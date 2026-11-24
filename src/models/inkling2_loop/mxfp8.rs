//! Compact E4M3 weights in eight-row warp fragments. Scales retain their
//! checkpoint values for scaled E4M3 MMA or register expansion for BF16 MMA.
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

/// These scales have exact BF16 expansion for every finite weight code.
/// Other scales retain the existing per-block validation and BF16 projection.
pub(super) fn ordinary_scale(exponent: u8) -> bool {
    (3..=246).contains(&exponent)
}

fn scaled_instruction() -> cubecl::features::ScaledMmaConfig {
    cubecl::features::ScaledMmaConfig {
        a_type: e4m3::cube_type(),
        b_type: e4m3::cube_type(),
        cd_type: f32::cube_type(),
        scales_type: cubecl::ue8m0::cube_type(),
        m: 16,
        n: 8,
        k: 32,
        scales_factor: 1,
    }
}

fn has_scaled_instruction(features: &cubecl::features::MatmulFeatures) -> bool {
    features.scaled_mma.contains(&scaled_instruction())
}

fn qualified_toolchain(major: i32, minor: i32, driver: i32) -> bool {
    // CUDA 13.0 is the qualified NVRTC/PTX floor. The driver JIT must
    // understand the PTX version emitted by the actually loaded compiler.
    major >= 13 && driver >= major * 1000 + minor * 10
}

pub(super) fn native_supported(client: &Client) -> bool {
    if !has_scaled_instruction(&client.properties().features.matmul) {
        return false;
    }
    let (mut major, mut minor, mut driver) = (0, 0, 0);
    unsafe {
        cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor)
            .result()
            .is_ok()
            && cudarc::driver::sys::cuDriverGetVersion(&mut driver)
                .result()
                .is_ok()
            && qualified_toolchain(major, minor, driver)
    }
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
    native: bool,
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
                        if !ordinary_scale(exponent) {
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
            native,
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

#[cube(launch_unchecked, address_type = "dynamic")]
fn grouped_scaled(
    a: &Array<Vector<e4m3, Const<4>>>,
    a_scales: &Array<cubecl::ue8m0>,
    b: &Array<Vector<e4m3, Const<4>>>,
    b_scales: &Array<cubecl::ue8m0>,
    slots: &Array<u32>,
    tile0: &Array<u32>,
    counts: &Array<u32>,
    out: &mut Array<Vector<bf16, Const<2>>>,
    #[comptime] k: usize,
    #[comptime] n: usize,
) {
    let group = CUBE_POS_Y as usize;
    if counts[group] == 0u32 {
        terminate!();
    }
    let row0 = tile0[group] as usize * 16usize;
    let col_tile = CUBE_POS_X as usize * 4usize + PLANE_POS as usize;
    let tile = slots[group] as usize * (n / 8usize) + col_tile;
    let def = cmma::MmaDefinition::<e4m3, e4m3, f32>::new_scaled::<cubecl::ue8m0>(
        16usize, 8usize, 32usize, 1usize,
    );
    let lane = UNIT_POS_PLANE;
    let mut aa = Array::<Vector<e4m3, Const<4>>>::new(4usize);
    let mut bb = Array::<Vector<e4m3, Const<4>>>::new(2usize);
    let mut acc = Array::<Vector<f32, Const<2>>>::new(2usize);
    #[unroll]
    for i in 0usize..2usize {
        acc[i] = Vector::cast_from(0f32);
    }
    // Scales have a lane mapping of their own. A indexes one exponent
    // per activation row and K block. B indexes the same packed weight
    // tile as the code loads, with eight exponents per K block.
    let sa_row = row0 + def.scales_index(lane, MatrixIdent::A) as usize;
    let sb_col = def.scales_index(lane, MatrixIdent::B) as usize;
    for block in 0usize..k / 32usize {
        let group = tile * (k / 32usize) + block;
        #[unroll]
        for i in 0usize..4usize {
            let (r, c) = def.position_of_nth(lane, (i * 4usize) as u32, MatrixIdent::A);
            aa[i] = a[((row0 + r as usize) * k + block * 32usize + c as usize) / 4usize];
        }
        #[unroll]
        for i in 0usize..2usize {
            let (r, c) = def.position_of_nth(lane, (i * 4usize) as u32, MatrixIdent::B);
            let r = r as usize;
            bb[i] = b[(group * 4usize + r / 8usize) * 16usize
                + c as usize * 2usize
                + (r % 8usize) / 4usize];
        }
        // scale_vec::1X uses byte zero of each lane's scale register.
        let mut sa = Vector::<cubecl::ue8m0, Const<4>>::empty();
        let mut sb = Vector::<cubecl::ue8m0, Const<4>>::empty();
        let exp_a = a_scales[sa_row * (k / 32usize) + block];
        let exp_b = b_scales[group * 8usize + sb_col];
        #[unroll]
        for byte in 0usize..4usize {
            sa[byte] = exp_a;
            sb[byte] = exp_b;
        }
        let d = def.execute_scaled(&aa, &bb, &acc, sa, sb);
        #[unroll]
        for i in 0usize..2usize {
            acc[i] = d[i];
        }
    }
    #[unroll]
    for i in 0usize..2usize {
        let (r, c) = def.position_of_nth(lane, (i * 2usize) as u32, MatrixIdent::Accumulator);
        out[((row0 + r as usize) * n + col_tile * 8usize + c as usize) / 2usize] =
            Vector::<bf16, Const<2>>::cast_from(acc[i]);
    }
}

pub(super) fn scaled_project(
    client: &Client,
    a: &(Handle, Handle),
    codes: &Handle,
    scales: &Handle,
    plan: &BlockPlanDev,
    m: usize,
    n: usize,
    k: usize,
) -> Handle {
    assert_eq!(plan.planes, 1);
    let out = client.empty(m * n * size_of::<bf16>());
    unsafe {
        grouped_scaled::launch_unchecked::<cubecl::cuda::CudaRuntime>(
            client,
            CubeCount::Static((n / 32) as u32, plan.blocks as u32, 1),
            CubeDim::new_1d(128),
            AddressType::from_len((m * n.max(k)).max(codes.size() as usize)),
            ArrayArg::from_raw_parts(a.0.clone(), m * k),
            ArrayArg::from_raw_parts(a.1.clone(), m * k / 32),
            ArrayArg::from_raw_parts(codes.clone(), codes.size() as usize),
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
    #[test]
    fn mxfp8_instruction_and_toolchain() {
        let mut features = cubecl::features::MatmulFeatures::default();
        assert!(!has_scaled_instruction(&features));
        for other in [
            cubecl::features::ScaledMmaConfig {
                scales_type: e4m3::cube_type(),
                ..scaled_instruction()
            },
            cubecl::features::ScaledMmaConfig {
                scales_factor: 2,
                ..scaled_instruction()
            },
            cubecl::features::ScaledMmaConfig {
                k: 64,
                ..scaled_instruction()
            },
        ] {
            features.scaled_mma.insert(other);
        }
        assert!(!has_scaled_instruction(&features));
        features.scaled_mma.insert(scaled_instruction());
        assert!(has_scaled_instruction(&features));
        assert!(!qualified_toolchain(12, 9, 13000));
        assert!(!qualified_toolchain(13, 1, 13000));
        assert!(qualified_toolchain(13, 0, 13000));
    }

    #[test]
    fn mxfp8_native_activation_edges() {
        use crate::models::inkling::fp4quant::pack_mxfp8;
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        let client = CudaRuntime::client(&CudaDevice::default());
        if !native_supported(&client) {
            return;
        }
        let (m, n, k) = (16usize, 32usize, 32usize);
        let mut x = vec![bf16::ZERO; m * k];
        for (row, block) in x.chunks_mut(k).enumerate() {
            if row % 2 == 0 {
                block[0] = bf16::from_f32(480.0);
                block[1] = bf16::from_f32(-1.0 / 512.0);
                block[2] = bf16::from_f32(1.0625);
            } else {
                block[1] = bf16::from_bits(1);
            }
        }
        let x = client.create_from_slice(bf16::as_bytes(&x));
        let packed = pack_mxfp8(&client, &x, m, k);
        let codes = client.read_one(packed.0.clone()).unwrap();
        let scales = client.read_one(packed.1.clone()).unwrap();
        for r in 0..m {
            let (exp, prefix) = if r % 2 == 0 {
                (127, [0x7e, 0x81, 0x38])
            } else {
                (1, [0, 4, 0])
            };
            assert_eq!(scales[r], exp);
            assert_eq!(&codes[r * k..r * k + 3], &prefix);
        }

        // Only the tiny input contributes. An amplified product makes a
        // flushed subnormal observable even after the BF16 store.
        let mut weights = vec![0u8; n * k];
        for row in weights.chunks_mut(k) {
            row[1] = 0x38;
        }
        let p = bind(&client, &weights, &vec![227; n], 1, n, k, 0, 1, true).unwrap();
        let Stack::Mxfp8 { codes, scales, .. } = p.stack else {
            unreachable!()
        };
        let words = |v: &[u32]| client.create_from_slice(u32::as_bytes(v));
        let plan = BlockPlanDev {
            slot: words(&[0]),
            tile0: words(&[0]),
            cnt: words(&[16]),
            blocks: 1,
            planes: 1,
            rows_real: m,
        };
        let y = scaled_project(&client, &packed, &codes, &scales, &plan, m, n, k);
        let bytes = client.read_one(y).unwrap();
        for (r, row) in bf16::from_bytes(&bytes).chunks_exact(n).enumerate() {
            let expected = if r % 2 == 0 {
                -2f32.powi(91)
            } else {
                2f32.powi(-33)
            };
            assert!(
                row.iter().all(|v| v.to_f32() == expected),
                "row {r}: {:?}",
                row[0]
            );
        }
    }

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
