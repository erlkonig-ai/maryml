//! Compact E4M3 weights in eight-row warp fragments. Scales retain their
//! checkpoint values. Only a register fragment is expanded for BF16 MMA.
use crate::models::inkling::moegroup::BlockPlanDev;
use crate::models::inkling2_loop::experts::{MXFP8_BLOCK, decode_mxfp8_row};
use crate::models::inkling2_loop::moe::{Proj, Stack};
use cubecl::features::TypeUsage;
use cubecl::ir::MatrixIdent;
use cubecl::server::Handle;
use cubecl::{e4m3, prelude::*};
use half::bf16;

type Client = ComputeClient<cubecl::cuda::CudaRuntime>;

pub(super) fn supported(client: &Client) -> bool {
    client
        .properties()
        .features
        .type_usage(e4m3::cube_type())
        .contains(TypeUsage::Conversion)
}

/// Interleave eight adjacent columns' eight-byte K fragments so each warp
/// loads 64 contiguous codes. W13's alternating gate/up rows are separated
/// while packing. Row selection restarts naturally at each expert boundary.
pub(super) fn bind(
    client: &Client,
    codes: &[u8],
    scales: &[u8],
    experts: usize,
    n: usize,
    k: usize,
    first: usize,
    stride: usize,
) -> Result<Proj, String> {
    let per = n * k;
    let mut packed = vec![0; experts * per];
    let mut exponents = vec![0; experts * per / MXFP8_BLOCK];
    super::sliding::fill_experts(
        experts,
        [&mut packed, &mut exponents],
        |expert, [packed, exponents]| {
            for tile in 0..n / 8 {
                for row in 0..8 {
                    let source_row = (expert * n + tile * 8 + row) * stride + first;
                    for block in 0..k / 32 {
                        let index = source_row * k / 32 + block;
                        let exponent = scales[index];
                        let src = &codes[index * 32..(index + 1) * 32];
                        // Every finite E4M3 value at these scales is exactly BF16.
                        // Otherwise apply the original binder's representability test.
                        if !(3..=246).contains(&exponent) {
                            let mut values = [0f32; 32];
                            decode_mxfp8_row(src, &[exponent], MXFP8_BLOCK, &mut values);
                            if values.iter().any(|&v| {
                                !v.is_nan() && bf16::from_f32(v).to_f32().to_bits() != v.to_bits()
                            }) {
                                return Err(format!(
                                    "MXFP8 block {index} has an inexact BF16 weight"
                                ));
                            }
                        }
                        let group = tile * k / 32 + block;
                        exponents[group * 8 + row] = exponent;
                        for (fragment, bytes) in src.chunks_exact(8).enumerate() {
                            let dst = (group * 4 + fragment) * 64 + row * 8;
                            packed[dst..dst + 8].copy_from_slice(bytes);
                        }
                    }
                }
            }
            Ok(())
        },
    )?;
    Ok(Proj {
        n,
        k,
        stack: Stack::Mxfp8 {
            codes: client.create_from_slice(&packed),
            scales: client.create_from_slice(&exponents),
        },
    })
}

#[cube]
fn expand(pair: u16, exponent: u8) -> Vector<bf16, Const<2>> {
    let exp = u32::cast_from(exponent);
    let mut bits = exp << 23;
    if exp == 0 {
        bits = 0x0040_0000u32;
    }
    if exp == 255 {
        bits = 0x7fc0_0000u32;
    }
    let values = Vector::<f32, Const<2>>::cast_from(Vector::<e4m3, Const<2>>::reinterpret(pair));
    Vector::<bf16, Const<2>>::cast_from(
        values * Vector::<f32, Const<2>>::cast_from(f32::reinterpret(bits)),
    )
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn grouped_compact(
    a: &Array<Vector<bf16, Const<2>>>,
    codes: &Array<u16>,
    scales: &Array<u8>,
    slots: &Array<u32>,
    tile0: &Array<u32>,
    counts: &Array<u32>,
    out: &mut Array<Vector<f32, Const<2>>>,
    #[comptime] k: usize,
    #[comptime] n: usize,
) {
    let group = CUBE_POS_Y as usize;
    if counts[group] == 0u32 {
        terminate!();
    }
    let row0 = tile0[group] as usize * 16usize;
    let col_tile = CUBE_POS_X as usize * 4usize + PLANE_POS as usize;
    let expert = slots[group] as usize;
    let tile = expert * (n / 8usize) + col_tile;
    let def = cmma::MmaDefinition::<bf16, bf16, f32>::new(16usize, 8usize, 16usize);
    let lane = UNIT_POS_PLANE;
    let mut aa = Array::<Vector<bf16, Const<2usize>>>::new(4usize);
    let mut bb = Array::<Vector<bf16, Const<2usize>>>::new(2usize);
    let mut acc = Array::<Vector<f32, Const<2usize>>>::new(2usize);
    #[unroll]
    for i in 0usize..2usize {
        acc[i] = Vector::<f32, Const<2usize>>::cast_from(0f32);
    }
    for block in 0usize..k / 32usize {
        let scale_group = tile * (k / 32usize) + block;
        let (_, bcol) = def.position_of_nth(lane, 0u32, MatrixIdent::B);
        let exponent = scales[scale_group * 8usize + bcol as usize];
        #[unroll]
        for half in 0usize..2usize {
            #[unroll]
            for i in 0usize..4usize {
                let (r, c) = def.position_of_nth(lane, (i * 2usize) as u32, MatrixIdent::A);
                aa[i] =
                    a[((row0 + r as usize) * k + block * 32usize + half * 16usize + c as usize)
                        / 2usize];
            }
            #[unroll]
            for i in 0usize..2usize {
                let (r, c) = def.position_of_nth(lane, (i * 2usize) as u32, MatrixIdent::B);
                let p = (scale_group * 4usize + half * 2usize + r as usize / 8usize) * 32usize
                    + c as usize * 4usize
                    + (r as usize % 8usize) / 2usize;
                bb[i] = expand(codes[p], exponent);
            }
            let d = def.execute(&aa, &bb, &acc);
            #[unroll]
            for i in 0usize..2usize {
                acc[i] = d[i];
            }
        }
    }
    #[unroll]
    for i in 0usize..2usize {
        let (r, c) = def.position_of_nth(lane, (i * 2usize) as u32, MatrixIdent::Accumulator);
        out[((row0 + r as usize) * n + col_tile * 8usize + c as usize) / 2usize] = acc[i];
    }
}

pub(super) fn project(
    client: &Client,
    a: &Handle,
    codes: &Handle,
    scales: &Handle,
    plan: &BlockPlanDev,
    m: usize,
    n: usize,
    k: usize,
) -> Handle {
    assert_eq!(plan.planes, 1);
    let out = client.empty(m * n * 4);
    unsafe {
        grouped_compact::launch_unchecked::<cubecl::cuda::CudaRuntime>(
            client,
            CubeCount::Static((n / 32) as u32, plan.blocks as u32, 1),
            CubeDim::new_1d(128),
            AddressType::from_len((m * n.max(k)).max(codes.size() as usize)),
            ArrayArg::from_raw_parts(a.clone(), m * k),
            ArrayArg::from_raw_parts(codes.clone(), codes.size() as usize / 2),
            ArrayArg::from_raw_parts(scales.clone(), scales.size() as usize),
            ArrayArg::from_raw_parts(plan.slot.clone(), plan.blocks),
            ArrayArg::from_raw_parts(plan.tile0.clone(), plan.blocks),
            ArrayArg::from_raw_parts(plan.cnt.clone(), plan.blocks),
            ArrayArg::from_raw_parts(out.clone(), m * n),
            k,
            n,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cube(launch_unchecked)]
    fn edge_probe(src: &Array<u16>, exps: &Array<u8>, out: &mut Array<Vector<bf16, Const<2>>>) {
        let i = ABSOLUTE_POS;
        out[i] = expand(src[i], exps[i]);
    }

    #[test]
    fn mxfp8_device_conversion() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        let client = CudaRuntime::client(&CudaDevice::default());
        if !supported(&client) {
            return;
        }
        use crate::models::inkling::nvfp4::e4m3_to_f32;
        use crate::models::inkling2_loop::experts::e8m0_to_f32;
        let pairs: Vec<u16> = (0..256)
            .flat_map(|_| (0..256u16).map(|x| x | ((255 - x) << 8)))
            .collect();
        let scales: Vec<u8> = (0..256)
            .flat_map(|x| std::iter::repeat_n(x as u8, 256))
            .collect();
        let a = client.create_from_slice(u16::as_bytes(&pairs));
        let s = client.create_from_slice(&scales);
        let out = client.empty(pairs.len() * 4);
        unsafe {
            edge_probe::launch_unchecked::<cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::Static(256, 1, 1),
                CubeDim::new_1d(256),
                ArrayArg::from_raw_parts(a, pairs.len()),
                ArrayArg::from_raw_parts(s, scales.len()),
                ArrayArg::from_raw_parts(out.clone(), pairs.len() * 2),
            );
        }
        let bytes = client.read_one(out).unwrap();
        for (i, bb) in bytes.chunks_exact(2).enumerate() {
            let got = bf16::from_le_bytes([bb[0], bb[1]]);
            let code = pairs[i / 2].to_le_bytes()[i % 2];
            let want = bf16::from_f32(e4m3_to_f32(code) * e8m0_to_f32(scales[i / 2]));
            assert!(
                want.to_bits() == got.to_bits() || (want.is_nan() && got.is_nan()),
                "code {code} exponent {}: {got:?} expected {want:?}",
                scales[i / 2]
            );
        }
    }
}
