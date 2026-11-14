//! Portable BF16 linear projection for the wgpu build of the text decoder.
//!
//! `[M,K] x [N,K]^T -> [M,N]`: BF16 input and weight, F32 accumulation, one
//! BF16 rounding of each output, like the CUDA build's fixed MMA blueprint
//! (`gdn_mixer::project_mma`), but through shared memory and ordinary F32
//! multiply-adds instead of `mma.sync`, which neither Vulkan nor Metal has.
//! The K reduction order is a fixed function of K (32-wide stages in order,
//! each summed in order), independent of M and of the device, so a row's
//! output does not depend on how many rows share its launch.
//!
//! One cube of 16x16 units owns a 64x64 output tile; each unit owns a 4x4
//! block of it, strided by 16 in both axes so neighbouring units read
//! neighbouring shared-memory words. A query's few dozen rows fit in one M
//! tile, so each weight element is read from device memory once per call.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use half::bf16;

use super::Rt;

const TILE_M: usize = 64;
const TILE_N: usize = 64;
const TILE_K: usize = 32;
/// Shared B tile row pitch: one word of padding, so the transposed stores
/// of consecutive units land in different banks.
const PITCH_B: usize = TILE_N + 1;

#[cube(launch_unchecked)]
fn tiled_kernel(
    input: &Array<bf16>,
    weight: &Array<bf16>,
    output: &mut Array<bf16>,
    m: usize,
    n: usize,
    k: usize,
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let unit = ty * 16 + tx;
    let n0 = CUBE_POS_X as usize * TILE_N;
    let m0 = CUBE_POS_Y as usize * TILE_M;
    let mut a_tile = SharedMemory::<f32>::new(TILE_M * TILE_K);
    let mut b_tile = SharedMemory::<f32>::new(TILE_K * PITCH_B);
    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for i in 0..16usize {
        acc[i] = 0.0f32;
    }
    let mut k0 = 0usize;
    while k0 < k {
        // 64x32 of the input and 64x32 of the weight, 8 elements per unit,
        // consecutive units on consecutive K: coalesced device reads.
        #[unroll]
        for l in 0..8usize {
            let e = unit + 256 * l;
            let r = e / TILE_K;
            let c = e % TILE_K;
            let row = m0 + r;
            let col = k0 + c;
            a_tile[r * TILE_K + c] = if row < m && col < k {
                f32::cast_from(input[row * k + col])
            } else {
                f32::new(0.0f32)
            };
            let out = n0 + r;
            b_tile[c * PITCH_B + r] = if out < n && col < k {
                f32::cast_from(weight[out * k + col])
            } else {
                f32::new(0.0f32)
            };
        }
        sync_cube();
        for kk in 0..TILE_K {
            #[unroll]
            for i in 0..4usize {
                let a = a_tile[(ty + 16 * i) * TILE_K + kk];
                #[unroll]
                for j in 0..4usize {
                    acc[i * 4 + j] += a * b_tile[kk * PITCH_B + tx + 16 * j];
                }
            }
        }
        sync_cube();
        k0 += TILE_K;
    }
    #[unroll]
    for i in 0..4usize {
        #[unroll]
        for j in 0..4usize {
            let row = m0 + ty + 16 * i;
            let col = n0 + tx + 16 * j;
            if row < m && col < n {
                output[row * n + col] = bf16::cast_from(acc[i * 4 + j]);
            }
        }
    }
}

/// Called by `gdn_mixer::project_impl` after its descriptor, device and
/// u32-extent checks: `input` is contiguous BF16 `[B,T,K]`, `weight`
/// contiguous BF16 `[N,K]`, `shape` is `[B,T,N]` and `no` its element count.
pub(super) fn project(
    input: &CubeTensor<Rt>,
    weight: &CubeTensor<Rt>,
    shape: [usize; 3],
    no: usize,
) -> CubeTensor<Rt> {
    let k = input.meta.shape()[2];
    let (m, n) = (shape[0] * shape[1], shape[2]);
    let output = input.client.empty(no * 2);
    let count = CubeCount::Static(n.div_ceil(TILE_N) as u32, m.div_ceil(TILE_M) as u32, 1);
    // SAFETY: the caller checked both operands' contiguous extents (m*k and
    // n*k elements) and the u32 index domain; every read and write below is
    // bounds-guarded by m, n and k, and the output is a fresh buffer.
    unsafe {
        tiled_kernel::launch_unchecked::<Rt>(
            &input.client,
            count,
            CubeDim::new_2d(16, 16),
            ArrayArg::from_raw_parts(input.handle.clone(), m * k),
            ArrayArg::from_raw_parts(weight.handle.clone(), n * k),
            ArrayArg::from_raw_parts(output.clone(), no),
            m,
            n,
            k,
        );
    }
    CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape.into(),
        output,
        DType::BF16,
    )
}
