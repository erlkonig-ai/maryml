//! GGUF type-40 bytes decoded inside an ordinary F32 matvec. Not Mary's typed
//! FP4 planes, not tensor cores, and not activation quantization. No repacking.
//! Decode has its own bitwise test, independent of accumulation diagnostics.

use anyhow::{Context, Result, ensure};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{Buffer, Device, H, KV, D};
use crate::formats::gguf::nvfp4;

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

#[cube(launch_unchecked)]
fn packed_matvec_kernel(
    x: &Array<f32>, raw: &Array<u8>, y: &mut Array<f32>,
    ni: u32, no: u32, batch: u32, row_bytes: u32,
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
            let weight = decode_value(raw, row_bytes, wr, i);
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
pub(super) struct PackedBuffer { handle: Handle, bytes: usize, ni: usize, no: usize }

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
        Ok(PackedBuffer { handle: self.client.create_from_slice(raw), bytes: raw.len(), ni, no })
    }
    pub(super) fn packed_matvec(&self, x: &Buffer, w: &PackedBuffer, y: &Buffer,
        batch: usize, input_offset: usize, output_offset: usize) -> Result<()> {
        launch_geometry(w.ni, w.no, w.bytes, batch, x.len, y.len, input_offset, output_offset)?;
        let row_bytes = (w.ni / nvfp4::ELEMENTS) * nvfp4::BYTES; // checked above
        unsafe { packed_matvec_kernel::launch_unchecked::<CudaRuntime>(
            &self.client, CubeCount::new_1d((w.no * batch).div_ceil(8) as u32), CubeDim::new_1d(256),
            x.arg(), ArrayArg::from_raw_parts(w.handle.clone(), w.bytes), y.arg(),
            w.ni as u32, w.no as u32, batch as u32, row_bytes as u32,
            input_offset as u32, output_offset as u32,
        ); }
        Ok(())
    }
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
    dev.matvec(&x, &plain, &b, H, no, t, 0, 0);
    let scale = dev.upload(&[scale]);
    Ok(KComparison {
        packed_unscaled: dev.read(&a)?, decoded_unscaled: dev.read(&b)?,
        packed_scaled: dev.read(&dev.point(&a, &scale, a.len, 1))?,
        decoded_scaled: dev.read(&dev.point(&b, &scale, b.len, 1))?,
    })
}

#[cfg(test)]
#[cube(launch_unchecked)]
fn decode_probe_kernel(raw: &Array<u8>, out: &mut Array<f32>, elements: u32, row_bytes: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < elements { out[i as usize] = decode_value(raw, row_bytes, 0, i); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_geometry_rejects_invalid_before_launch() {
        assert_eq!(packed_bytes(2048, 512).unwrap(), 589824);
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
        for ni in [64, 128, 192, 512, 1024, 2048, 4096] { for t in [1, 3] {
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
            let a = dev.upload(&vec![-123.0; no * (t + 2)]);
            let b = dev.upload(&vec![-123.0; no * (t + 2)]);
            dev.packed_matvec(&x, &packed, &a, t, ni, no)?;
            dev.matvec(&x, &dev.upload(&decoded), &b, ni, no, t, ni, no);
            let mut expected = vec![-123.0; no * (t + 2)];
            for token in 0..t { for row in 0..no {
                expected[(token + 1) * no + row] = (0..ni).map(|i|
                    decoded[row * ni + i] as f64 * input[(token + 1) * ni + i] as f64).sum::<f64>() as f32;
            }}
            let actual = dev.read(&a)?;
            let reference = dev.read(&b)?;
            for (i, ((&a, &b), &e)) in actual.iter().zip(&reference).zip(&expected).enumerate() {
                assert!(a.is_finite());
                assert_eq!(a.to_bits(), b.to_bits(), "packed/decoded ni={ni} T={t} index={i}");
                assert_eq!(a, e, "exact dyadic ni={ni} T={t} index={i}");
            }
            let scale = dev.upload(&[0.75]);
            let scaled = dev.read(&dev.point(&a, &scale, a.len, 1))?;
            for (&s, &e) in scaled.iter().zip(&expected) { assert_eq!(s, e * 0.75); }
            println!("NVFP4 exact matvec: ni={ni} no={no} T={t}; offsets, sentinels, post-scale checked");
        }}
        Ok(())
    }
}
