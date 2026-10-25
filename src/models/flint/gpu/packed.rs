//! Explicit GGUF type-40 or native Tensor planes inside one ordinary F32 matvec.
//! No tensor cores or activation quantization. Native payload global must be1;
//! projection scales remain separate post-matmul operations in either layout.
//! Decode has its own bitwise test, independent of accumulation diagnostics.

use anyhow::{Context, Result, ensure};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::time::Instant;
use super::super::weights::Work;

use super::{Buffer, Device, H, KV, D};
use crate::formats::gguf::nvfp4;
use triblespace::core::blob::encodings::tensor::TensorView;

#[cube]
fn decode_value(raw: &Array<u8>, row_bytes: u32, row: u32, i: u32) -> f32 {
    let base = row * row_bytes + (i / 64) * 36;
    let group = (i % 64) / 16;
    let scale_byte = u32::cast_from(raw[(base + group) as usize]);
    let exponent = (scale_byte >> 3) & 15;
    let mantissa = scale_byte & 7;
    let mut scale = f32::new(0.0);
    // The special case is the full byte 0x7f, NOT (byte & 0x7f) == 0x7f.
    if scale_byte != 0 && scale_byte != 127 {
        if exponent == 0 {
            scale = mantissa as f32 * 0.001953125;
        } else {
            let mut power = f32::new(0.0009765625);
            let mut step = u32::new(0);
            while step < exponent { power *= 2.0; step += 1; }
            scale = (8 + mantissa) as f32 * power;
        }
    }
    let byte = u32::cast_from(raw[(base + 4 + group * 8 + i % 8) as usize]);
    let code = if i % 16 < 8 { byte & 15 } else { byte >> 4 };
    let magnitude = code & 7;
    let mut value = f32::new(0.0);
    if magnitude == 1 { value = 0.5; }
    if magnitude >= 2 {
        let mut power = f32::new(0.5);
        let mut step = u32::new(1);
        while step < magnitude / 2 { power *= 2.0; step += 1; }
        value = (2 + (magnitude & 1)) as f32 * power;
    }
    // CPU VALUES[0] and VALUES[8] are both positive zero. Negative nonzero
    // codes times a zero scale must still produce negative zero.
    if code >= 8 && magnitude != 0 { value = -value; }
    value * scale
}

#[cube]
fn decode_native_value(raw:&Array<u8>,codes:u32,index:u32)->f32 {
    let byte=u32::cast_from(raw[(index/2) as usize]);
    let code=if index%2==0 {byte&15} else {byte>>4};
    let scale_byte=u32::cast_from(raw[(codes+index/16) as usize]);
    // Host admission requires positive finite E4M3 (0..=126), global exactly1.
    let exponent=(scale_byte>>3)&15;
    let mantissa=scale_byte&7;
    let mut scale=f32::new(0.0);
    if exponent==0 { scale=mantissa as f32*0.001953125; }
    else {
        let mut power=f32::new(0.0009765625);
        let mut step=u32::new(0);
        while step<exponent { power*=2.0; step+=1; }
        scale=(8+mantissa) as f32*power;
    }
    let magnitude=code&7;
    let mut value=f32::new(0.0);
    if magnitude==1 { value=0.5; }
    if magnitude>=2 {
        let mut power=f32::new(0.5); let mut step=u32::new(1);
        while step<magnitude/2 { power*=2.0; step+=1; }
        value=(2+(magnitude&1)) as f32*power;
    }
    // Unlike GGUF, native code8 really is negative zero.
    if code>=8 { value=-value; }
    value*scale
}

#[cube(launch_unchecked)]
fn packed_matvec_kernel(
    x: &Array<f32>, raw: &Array<u8>, y: &mut Array<f32>,
    ni: u32, no: u32, batch: u32, row_bytes: u32, native_codes:u32,
    input_offset: u32, output_offset: u32,
) {
    // Same striding, accumulator and tree as the accepted decoded matvec.
    let lane = UNIT_POS_X % 32;
    let row = CUBE_POS_X * 8 + UNIT_POS_X / 32;
    let mut acc = f32::new(0.0);
    if row < no * batch {
        let token = row / no;
        let wr = row % no;
        let mut i = lane;
        while i < ni {
            let weight = if native_codes==0 { decode_value(raw, row_bytes, wr, i) }
                else { decode_native_value(raw,native_codes,wr*ni+i) };
            acc += weight * x[(input_offset + token * ni + i) as usize];
            i += 32;
        }
    }
    let mut red = SharedMemory::<f32>::new(256usize);
    red[UNIT_POS_X as usize] = acc;
    sync_cube();
    let mut stride = u32::new(16);
    while stride > 0 {
        if lane < stride {
            red[UNIT_POS_X as usize] = red[UNIT_POS_X as usize] + red[(UNIT_POS_X + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && row < no * batch {
        y[(output_offset + row) as usize] = red[UNIT_POS_X as usize];
    }
}

// Separate from the four-byte Buffer: logical length and allocation are bytes.
pub(super) struct PackedBuffer { handle: Handle, bytes: usize, ni: usize, no: usize, native_codes:u32 }

fn packed_bytes(ni: usize, no: usize) -> Result<usize> {
    ensure!(ni > 0 && ni % nvfp4::ELEMENTS == 0 && no > 0, "invalid packed matrix geometry");
    ensure!(ni <= (u32::MAX - 31) as usize, "packed width exceeds kernel index range");
    let bytes = (ni / nvfp4::ELEMENTS).checked_mul(nvfp4::BYTES)
        .and_then(|row| row.checked_mul(no)).context("packed byte count overflow")?;
    ensure!(bytes <= u32::MAX as usize, "packed bytes exceed kernel index range");
    Ok(bytes)
}

fn launch_geometry(ni: usize, no: usize, bytes: usize, batch: usize,
    input_len: usize, output_len: usize, input_offset: usize, output_offset: usize) -> Result<()> {
    ensure!(bytes == packed_bytes(ni, no)?, "packed matrix byte length mismatch");
    ensure!((1..=3).contains(&batch), "packed diagnostic supports T1..T3 only");
    let outputs = no.checked_mul(batch).context("packed output count overflow")?;
    ensure!(outputs <= (u32::MAX - 7) as usize, "packed launch exceeds kernel index range");
    let input_end = ni.checked_mul(batch).and_then(|n| input_offset.checked_add(n))
        .context("packed input end overflow")?;
    let output_end = output_offset.checked_add(outputs).context("packed output end overflow")?;
    ensure!(input_end <= input_len && output_end <= output_len, "packed buffer range outside allocation");
    ensure!(input_end <= u32::MAX as usize && output_end <= u32::MAX as usize,
        "packed buffer offsets exceed kernel index range");
    Ok(())
}

impl Device {
    pub(super) fn upload_nvfp4(&self, raw: &[u8], ni: usize, no: usize) -> Result<PackedBuffer> {
        ensure!(raw.len() == packed_bytes(ni, no)?, "packed upload byte length mismatch");
        Ok(PackedBuffer { handle: self.client.create_from_slice(raw), bytes: raw.len(), ni, no, native_codes:0 })
    }
    pub(super) fn upload_native_nvfp4(&self,view:&TensorView,ni:usize,no:usize)->Result<PackedBuffer> {
        self.upload_native_nvfp4_observed(view,ni,no,None)
    }
    pub(super) fn upload_native_nvfp4_observed(&self,view:&TensorView,ni:usize,no:usize,
        mut timing:Option<(&mut Work,&mut Work)>)->Result<PackedBuffer> {
        let start=timing.as_ref().map(|_|Instant::now());
        let layout=native_geometry(view,ni,no);
        if let (Some(start),Some((geometry,_)))=(start,timing.as_mut()) {
            geometry.record(start,view.payload().len(),0);
        }
        let (bytes,codes)=layout?;
        let start=timing.as_ref().map(|_|Instant::now());
        let handle=self.client.create_from_slice(view.payload());
        if let (Some(start),Some((_,upload)))=(start,timing.as_mut()) {
            upload.record(start,bytes,bytes);
        }
        Ok(PackedBuffer {handle,bytes,ni,no,native_codes:codes})
    }
    pub(super) fn packed_matvec(&self, x: &Buffer, w: &PackedBuffer, y: &Buffer,
        batch: usize, input_offset: usize, output_offset: usize) -> Result<()> {
        // Native adds only the4-byte intrinsic global; the code/scale plane
        // total is exactly the GGUF byte count. Geometry is otherwise identical.
        let layout_bytes=if w.native_codes==0 {w.bytes} else {w.bytes-4};
        launch_geometry(w.ni, w.no, layout_bytes, batch, x.len, y.len, input_offset, output_offset)?;
        let row_bytes = (w.ni / nvfp4::ELEMENTS) * nvfp4::BYTES; // checked above
        unsafe { packed_matvec_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d((w.no * batch).div_ceil(8) as u32), CubeDim::new_1d(256),
            x.arg(), ArrayArg::from_raw_parts(w.handle.clone(), w.bytes), y.arg(),
            w.ni as u32, w.no as u32, batch as u32, row_bytes as u32,w.native_codes,
            input_offset as u32, output_offset as u32,
        ); }
        Ok(())
    }
}

fn native_geometry(view:&TensorView,ni:usize,no:usize)->Result<(usize,u32)> {
    let gguf_bytes=packed_bytes(ni,no)?;
    let elements=ni.checked_mul(no).context("native matrix element overflow")?;
    ensure!(elements<=u32::MAX as usize,"native matrix index exceeds u32");
    ensure!(view.dims()==[no as u64,ni as u64],"native matrix logical dimensions differ");
    let bytes=gguf_bytes.checked_add(4).context("native byte count overflow")?;
    ensure!(bytes<=u32::MAX as usize && view.payload().len()==bytes,"native payload byte length differs");
    let (_,scales,global)=crate::nvfp4::split_payload(view.payload(),elements)?;
    ensure!(global.to_bits()==1.0f32.to_bits(),"native Flint import requires intrinsic global1; keep post-matmul scale separate");
    ensure!(scales.iter().all(|&b|b<=0x7e),"native Flint import requires positive finite E4M3 scale bytes");
    Ok((bytes,(elements/2) as u32))
}

/// A standalone comparison deliberately uploads a decoded reference matrix.
/// The packed layer mode does not call this diagnostic or allocate that copy.
pub struct KComparison {
    pub packed_unscaled: Vec<f32>,
    pub decoded_unscaled: Vec<f32>,
    pub packed_scaled: Vec<f32>,
    pub decoded_scaled: Vec<f32>,
}

pub fn compare_k(raw: &[u8], input: &[f32], scale: f32) -> Result<KComparison> {
    ensure!(input.len() % H == 0 && (1..=3).contains(&(input.len() / H)), "invalid K probe input length");
    ensure!(input.iter().all(|v| v.is_finite()) && scale.is_finite(), "nonfinite K probe input/scale");
    let no = KV * D;
    let t = input.len() / H;
    ensure!(raw.len() == packed_bytes(H, no)?, "K probe packed byte length mismatch");
    let decoded = nvfp4::decode(raw, H * no)?;
    let dev = Device::new();
    let x = dev.upload(input);
    let packed = dev.upload_nvfp4(raw, H, no)?;
    let plain = dev.upload(&decoded);
    let a = dev.empty(no * t);
    let b = dev.empty(no * t);
    dev.packed_matvec(&x, &packed, &a, t, 0, 0)?;
    dev.matvec(&x, &plain, &b, H, no, t, 0, 0)?;
    let scale = dev.upload(&[scale]);
    Ok(KComparison {
        packed_unscaled: dev.read(&a)?, decoded_unscaled: dev.read(&b)?,
        packed_scaled: dev.read(&dev.point(&a, &scale, a.len, 1)?)?,
        decoded_scaled: dev.read(&dev.point(&b, &scale, b.len, 1)?)?,
    })
}

#[cfg(test)]
#[cube(launch_unchecked)]
fn decode_probe_kernel(raw: &Array<u8>, out: &mut Array<f32>, elements: u32, row_bytes: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < elements { out[i as usize] = decode_value(raw, row_bytes, 0, i); }
}

#[cfg(test)]
#[cube(launch_unchecked)]
fn native_decode_probe_kernel(raw:&Array<u8>,out:&mut Array<f32>,elements:u32) {
    let i=ABSOLUTE_POS as u32;
    if i<elements { out[i as usize]=decode_native_value(raw,elements/2,i); }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_view(raw:&[u8],ni:usize,no:usize)->Result<TensorView> {
        crate::leaf::read_leaf(nvfp4::typed::to_tensor(raw,[no as u64,ni as u64])?)
    }

    #[test]
    fn native_geometry_admits_only_the_imported_arithmetic_domain()->Result<()> {
        use triblespace::core::blob::encodings::tensor::elements::NVFP4;
        let mut raw=[0x21;36]; raw[..4].fill(0x38);
        let view=native_view(&raw,64,1)?;
        assert_eq!(native_geometry(&view,64,1)?,(40,32));
        assert!(native_geometry(&view,128,1).is_err());
        for (offset,byte) in [(32,0x80),(32,0x7f),(39,0x40)] {
            let mut payload=view.payload().to_vec(); payload[offset]=byte;
            let bad=crate::leaf::read_leaf(crate::leaf::leaf_blob::<NVFP4,2>([1,64],anybytes::Bytes::from_source(payload))?)?;
            assert!(native_geometry(&bad,64,1).is_err());
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation"]
    fn cuda_native_setup_upload_spans_and_refusal()->Result<()> {
        let dev=Device::new();
        let mut raw=[0x11;36];raw[..4].fill(0x38);
        let view=native_view(&raw,64,1)?;
        let mut geometry=Work::default();let mut upload=Work::default();
        let packed=dev.upload_native_nvfp4_observed(&view,64,1,Some((&mut geometry,&mut upload)))?;
        assert_eq!((geometry.calls,geometry.input_bytes,geometry.output_bytes),(1,40,0));
        assert_eq!((upload.calls,upload.input_bytes,upload.output_bytes),(1,40,40));
        let x=dev.upload(&[1.0;64]);let y=dev.empty(1);
        dev.packed_matvec(&x,&packed,&y,1,0,0)?;
        assert_eq!(dev.read(&y)?,[32.0]); // Existing dyadic math, no timing tolerance.
        let mut bad_geometry=Work::default();let mut bad_upload=Work::default();
        assert!(dev.upload_native_nvfp4_observed(&view,128,1,Some((&mut bad_geometry,&mut bad_upload))).is_err());
        assert_eq!(bad_geometry.calls,1);assert_eq!(bad_upload.calls,0);
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation"]
    fn cuda_native_nvfp4_decode_all_admitted_scale_code_bits()->Result<()> {
        let mut raw=Vec::new();
        for scale in 0..255u8 {
            raw.extend([scale;4]);
            for group in 0..4u8 { for j in 0..8u8 {
                raw.push(((j+group*3)%16)|(((j+8+group*3)%16)<<4));
            }}
        }
        let elements=255*64;
        let expected=nvfp4::decode(&raw,elements)?;
        let dev=Device::new();
        let packed=dev.upload_native_nvfp4(&native_view(&raw,elements,1)?,elements,1)?;
        let out=dev.empty(elements);
        unsafe { native_decode_probe_kernel::launch_unchecked::<CudaRuntime>(
            &dev.client,CubeCount::new_1d(elements.div_ceil(256) as u32),CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(packed.handle.clone(),packed.bytes),out.arg(),elements as u32);
        }
        for (i,(&a,&e)) in dev.read(&out)?.iter().zip(&expected).enumerate() {
            assert_eq!(a.to_bits(),e.to_bits(),"native decode coordinate{i}");
        }
        // Converter canonicalizes GGUF code8 to native0; also exercise native8
        // directly, so this decoder cannot silently inherit GGUF's +zero rule.
        use triblespace::core::blob::encodings::tensor::elements::NVFP4;
        let mut payload=vec![0x88;32]; payload.extend([0,0x38,0,0x38]);
        payload.extend(1.0f32.to_le_bytes());
        let view=crate::leaf::read_leaf(crate::leaf::leaf_blob::<NVFP4,2>([1,64],anybytes::Bytes::from_source(payload))?)?;
        let packed=dev.upload_native_nvfp4(&view,64,1)?;
        let out=dev.empty(64);
        unsafe { native_decode_probe_kernel::launch_unchecked::<CudaRuntime>(
            &dev.client,CubeCount::new_1d(1),CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(packed.handle.clone(),packed.bytes),out.arg(),64u32);
        }
        for a in dev.read(&out)? { assert_eq!(a.to_bits(),(-0.0f32).to_bits()); }
        Ok(())
    }

    #[test]
    fn packed_geometry_rejects_invalid_before_launch() {
        assert_eq!(packed_bytes(2048, 512).unwrap(), 589824);
        assert_eq!(packed_bytes(6144, 2048).unwrap(), 7077888);
        assert!(launch_geometry(6144, 9, 31104, 3, 24576, 45, 6144, 9).is_ok());
        assert!(launch_geometry(6144, 9, 31103, 3, 24576, 45, 6144, 9).is_err());
        assert!(launch_geometry(6144, 9, 31104, 3, 24575, 45, 6144, 9).is_err());
        assert!(packed_bytes(63, 1).is_err());
        assert!(packed_bytes(0, 1).is_err());
        assert!(packed_bytes(64, 0).is_err());
        assert!(packed_bytes(64, usize::MAX).is_err());
        assert!(packed_bytes(usize::MAX, 1).is_err());
        assert!(launch_geometry(64, 9, 324, 3, 256, 45, 64, 9).is_ok());
        assert!(launch_geometry(64, 9, 323, 3, 256, 45, 64, 9).is_err());
        assert!(launch_geometry(64, 9, 324, 4, 512, 50, 0, 0).is_err());
        assert!(launch_geometry(64, 9, 324, 3, 255, 45, 64, 9).is_err());
        assert!(launch_geometry(64, 9, 324, 3, 256, 35, 64, 9).is_err());
        assert!(launch_geometry(64, 9, 324, 3, usize::MAX, 45, usize::MAX, 9).is_err());
        assert!(launch_geometry(64, 9, 324, 3, 256, usize::MAX, 64, usize::MAX).is_err());
        assert!(launch_geometry(64, 9, 324, 3, usize::MAX, 45, u32::MAX as usize, 9).is_err());
    }

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation"]
    fn cuda_nvfp4_decode_all_scale_code_bits() -> Result<()> {
        let mut raw = Vec::with_capacity(256 * 36);
        for scale in 0..=255u8 {
            raw.extend([scale; 4]);
            for group in 0..4u8 { for j in 0..8u8 {
                // Every group contains all 16 codes, with group-dependent order.
                let low = (j + group * 3) % 16;
                let high = (j + 8 + group * 3) % 16;
                raw.push(low | (high << 4));
            }}
        }
        let expected = nvfp4::decode(&raw, 256 * 64)?;
        let dev = Device::new();
        let packed = dev.upload_nvfp4(&raw, 256 * 64, 1)?;
        let out = dev.empty(expected.len());
        unsafe { decode_probe_kernel::launch_unchecked::<CudaRuntime>(
            &dev.client, CubeCount::new_1d((expected.len() / 256) as u32), CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(packed.handle.clone(), packed.bytes), out.arg(),
            expected.len() as u32, raw.len() as u32,
        ); }
        let actual = dev.read(&out)?;
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "decode coordinate {i}, scale {}, code-group {}", i / 64, (i % 64) / 16);
        }
        println!("NVFP4 decode: 256 scales x 16 codes x 4 groups = 16384 bitwise matches; signed zeros included");
        Ok(())
    }

    #[test]
    #[ignore = "requires an exclusive GB10 GPU reservation"]
    fn cuda_nvfp4_matvec_exact_t1_t3() -> Result<()> {
        let dev = Device::new();
        for ni in [64, 128, 192, 512, 1024, 2048, 4096, 6144] { for t in [1, 3] {
            let no = 9;
            let mut raw = Vec::new();
            for row in 0..no { for block in 0..ni / 64 {
                for group in 0..4 { raw.push(0x28 + ((row + block + group) % 4) as u8 * 8); }
                for group in 0..4 { for j in 0..8 {
                    let low = ((row * 3 + block * 5 + group * 7 + j) % 16) as u8;
                    let high = ((row * 7 + block * 3 + group * 5 + j + 9) % 16) as u8;
                    raw.push(low | (high << 4));
                }}
            }}
            let decoded = nvfp4::decode(&raw, ni * no)?;
            let input: Vec<f32> = (0..ni * (t + 1)).map(|i| (i as i32 % 19 - 9) as f32 / 16.0).collect();
            let x = dev.upload(&input);
            let packed = dev.upload_nvfp4(&raw, ni, no)?;
            let native = dev.upload_native_nvfp4(&native_view(&raw,ni,no)?,ni,no)?;
            let a = dev.upload(&vec![-123.0; no * (t + 2)]);
            let b = dev.upload(&vec![-123.0; no * (t + 2)]);
            let c = dev.upload(&vec![-123.0; no * (t + 2)]);
            dev.packed_matvec(&x, &packed, &a, t, ni, no)?;
            dev.matvec(&x, &dev.upload(&decoded), &b, ni, no, t, ni, no)?;
            dev.packed_matvec(&x,&native,&c,t,ni,no)?;
            let mut expected = vec![-123.0; no * (t + 2)];
            for token in 0..t { for row in 0..no {
                expected[(token + 1) * no + row] = (0..ni).map(|i|
                    decoded[row * ni + i] as f64 * input[(token + 1) * ni + i] as f64).sum::<f64>() as f32;
            }}
            let actual = dev.read(&a)?;
            let reference = dev.read(&b)?;
            let native_actual=dev.read(&c)?;
            for (&a,&b) in native_actual.iter().zip(&reference) { assert_eq!(a.to_bits(),b.to_bits(),"native/decoded ni={ni} T={t}"); }
            for (i, ((&a, &b), &e)) in actual.iter().zip(&reference).zip(&expected).enumerate() {
                assert!(a.is_finite());
                assert_eq!(a.to_bits(), b.to_bits(), "packed/decoded ni={ni} T={t} index={i}");
                assert_eq!(a, e, "exact dyadic ni={ni} T={t} index={i}");
            }
            let scale = dev.upload(&[0.75]);
            let scaled = dev.read(&dev.point(&a, &scale, a.len, 1)?)?;
            for (&s, &e) in scaled.iter().zip(&expected) { assert_eq!(s, e * 0.75); }
            println!("NVFP4 exact matvec: ni={ni} no={no} T={t}; offsets, sentinels, post-scale checked");
        }}
        Ok(())
    }
}
