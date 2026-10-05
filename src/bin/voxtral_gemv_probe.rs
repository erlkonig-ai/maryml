//! Which matmul kernel streams Voxtral's single-row projections fastest on
//! CUDA? Times burn's default matmul against cubek's GEMV strategies on the
//! raw f16 CUDA backend, at the shapes the folded realtime lane multiplies
//! every 80 ms frame, and checks each against the default's output.
//!
//! Measured on sky (GB10, CUDA 13), per call, mean of 50 after 5 warm-up
//! calls, "GB/s" = weight bytes / time. At m = 1 the plane-parallel GEMV over
//! a col-major rhs (weights stored `[N, K]`, viewed transposed) beat burn's
//! default (which reads the `[K, N]` layout fast.rs stores) on every decoder
//! projection: wide qkv 0.45 vs 0.64 ms, o_proj 0.13 vs 0.58, gate|up 0.71 vs
//! 0.75, down 0.36 vs 1.36, lm head 4.89 vs 5.00. One decoder layer's four
//! projections: 1.65 vs 3.33 ms, i.e. 43 vs 87 ms across the 26 layers. At
//! m = 4 (the encoder) no GEMV applies and the default was fastest.
//!
//!   cargo run --release --features voxtral-cuda --bin voxtral_gemv_probe

use burn::prelude::*;
use burn::tensor::{Distribution, TensorPrimitive};
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cubek::matmul::definition::{MatmulElems, MatmulGlobalElems};
use cubek::matmul::launch::Strategy;
use cubek::matmul::routines::BlueprintStrategy;
use cubek::std::InputBinding;
use mary::nn::backend::hear::{Device, RawHalf as BR};
use std::time::Instant;

fn cube<const D: usize>(t: Tensor<BR, D>) -> CubeTensor<CudaRuntime> {
    match t.into_primitive() {
        TensorPrimitive::Float(c) => c,
        _ => unreachable!(),
    }
}

fn launch(
    strategy: &Strategy,
    x: &Tensor<BR, 3>,
    w: &Tensor<BR, 3>,
) -> Result<Tensor<BR, 3>, String> {
    let (lhs, rhs) = (cube(x.clone()), cube(w.clone()));
    let out = burn_cubecl::kernel::matmul::init_matmul_output(&lhs, &rhs, lhs.dtype);
    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: lhs.dtype.into(),
        rhs: rhs.dtype.into(),
        out: out.dtype.into(),
    });
    let client = lhs.client.clone();
    let (ld, rd) = (lhs.dtype, rhs.dtype);
    cubek::matmul::launch::launch_ref(
        strategy,
        &client,
        InputBinding::new(lhs.binding(), ld.into()),
        InputBinding::new(rhs.binding(), rd.into()),
        out.clone().binding(),
        &mut dtypes,
    )
    .map_err(|e| format!("{e:?}"))?;
    Ok(Tensor::from_primitive(TensorPrimitive::Float(out)))
}

fn sync(t: &Tensor<BR, 3>) {
    let _ = t.clone().slice([0..1, 0..1, 0..1]).into_data();
}

fn time(iters: usize, mut f: impl FnMut() -> Tensor<BR, 3>) -> f64 {
    for _ in 0..5 {
        sync(&f());
    }
    let t0 = Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(f());
    }
    sync(&last.unwrap());
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let dev = Device::default();
    // (label, m, k, n): decoder projections at m = 1, encoder at m = 4.
    let shapes = [
        ("dec wide qkv", 1, 3072, 11264),
        ("dec o_proj", 1, 4096, 3072),
        ("dec gate|up", 1, 3072, 18432),
        ("dec down", 1, 9216, 3072),
        ("dec lm head", 1, 3072, 131072),
        ("enc wide qkv", 4, 1280, 10240),
        ("enc gate|up", 4, 1280, 10240),
        ("enc down", 4, 5120, 1280),
    ];
    let iters = 50;
    for (label, m, k, n) in shapes {
        let x = Tensor::<BR, 3>::random([1, m, k], Distribution::Uniform(-1.0, 1.0), &dev);
        // [K, N] row-major: the layout fast.rs stores (pre-transposed).
        let w_kn = Tensor::<BR, 3>::random([1, k, n], Distribution::Uniform(-0.05, 0.05), &dev);
        // [N, K] row-major viewed as [K, N]: the col-major rhs a vecmat wants.
        let w_nk = w_kn.clone().swap_dims(1, 2).into_data();
        let w_nk = Tensor::<BR, 3>::from_data(w_nk, &dev);
        let w_col = w_nk.clone().swap_dims(1, 2);
        let bytes = (k * n * 2) as f64;

        let reference = x.clone().matmul(w_kn.clone());
        let ref_host: Vec<f32> = reference
            .clone()
            .into_data()
            .convert::<f32>()
            .into_vec()
            .unwrap();
        let ms = time(iters, || x.clone().matmul(w_kn.clone()));
        println!(
            "{label:13} m{m} k{k} n{n}: burn default     {ms:7.3} ms  {:6.1} GB/s",
            bytes / ms / 1e6
        );
        let ms = time(iters, || x.clone().matmul(w_col.clone()));
        println!(
            "{label:13}                 burn default col {ms:7.3} ms  {:6.1} GB/s",
            bytes / ms / 1e6
        );

        let candidates: Vec<(&str, Strategy, &Tensor<BR, 3>)> = vec![
            (
                "gemv unit-perp row",
                Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(Default::default())),
                &w_kn,
            ),
            (
                "gemv plane-par col",
                Strategy::GemvPlaneParallel(BlueprintStrategy::Inferred(Default::default())),
                &w_col,
            ),
            (
                "vecmat simple col",
                Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
                &w_col,
            ),
            (
                "vecmat double col",
                Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
                &w_col,
            ),
            (
                "simple unit row",
                Strategy::SimpleUnit(Default::default()),
                &w_kn,
            ),
        ];
        for (name, strategy, w) in candidates {
            match launch(&strategy, &x, w) {
                Err(e) => println!(
                    "{:13}                 {name:18} unavailable: {}",
                    "",
                    &e[..e.len().min(90)]
                ),
                Ok(out) => {
                    let host: Vec<f32> = out.into_data().convert::<f32>().into_vec().unwrap();
                    let err = host
                        .iter()
                        .zip(&ref_host)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let ms = time(iters, || launch(&strategy, &x, w).unwrap());
                    println!(
                        "{:13}                 {name:18} {ms:7.3} ms  {:6.1} GB/s  max|d| {err:.2e}",
                        "",
                        bytes / ms / 1e6
                    );
                }
            }
        }
    }
}
