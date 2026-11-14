//! Embed query texts with WeMM's text leg on the runtime this build selects.
//!
//! `wemm_query PILE COLLECTION ROOT QUERIES.txt OUT.f32`
//!   binds the decoder of model `ROOT` from the `models` collection
//!   `COLLECTION` (`blake3:HEX` or HEX) of the read-only pile `PILE`, embeds
//!   every non-empty line of QUERIES.txt exactly as a faculty query is
//!   embedded (`text::TextWemm::embed_text`, the path of
//!   `NativeWemm::embed_text`: the root's chat template frames the raw line,
//!   the `<embedding>` row is final-normed and L2-normalized), and writes the
//!   BF16 embeddings widened to little-endian f32, `[n, 4096]` in line order.
//!
//! The wgpu build (`wemm-query-vulkan`, `wemm-query-metal`) copies the 426
//! BF16 decoder leaves into device buffers and runs every projection in
//! BF16. The CUDA build (`wemm-input`) aliases them and runs
//! `WEMM_PRECISION` (bf16, w4a4, w4a16; default bf16), which is how the
//! production query vector is reproduced for comparison. Every query is run
//! `WEMM_PASSES` times (default 2); pass 1 includes kernel compilation. The
//! last pass's vectors are written. Timings go to stdout.
use anyhow::{Context, Result, ensure};
use half::bf16;
use mary::models::qwen3_5::{
    gdn_mixer::DecoderPrecision,
    prepared::CudaTensor,
    text::{Assets, TextWemm},
};
use std::{fs, path::Path, time::Instant};
use triblespace::prelude::Id;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn mem_available_kib() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find(|l| l.starts_with("MemAvailable"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
}

/// Waits for the tensor's producers and reads its BF16 words back.
fn read(t: &CudaTensor) -> Result<Vec<f32>> {
    let bytes = t
        .client
        .read_one(t.handle.clone())
        .map_err(|e| anyhow::anyhow!("read: {e:?}"))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

#[cfg(feature = "qwen3_5-wgpu")]
fn binder() -> Result<mary::models::qwen3_5::Binder> {
    use cubecl::wgpu::WgpuDevice;
    let device = match std::env::var("WEMM_WGPU_DEVICE").ok().as_deref() {
        None | Some("default") => WgpuDevice::DefaultDevice,
        Some(other) => {
            let index = other
                .strip_prefix("discrete:")
                .or_else(|| other.strip_prefix("integrated:"));
            match (
                other.starts_with("discrete:"),
                index.map(str::parse::<usize>),
            ) {
                (true, Some(Ok(i))) => WgpuDevice::DiscreteGpu(i),
                (false, Some(Ok(i))) => WgpuDevice::IntegratedGpu(i),
                _ => anyhow::bail!(
                    "WEMM_WGPU_DEVICE is default, discrete:N or integrated:N, not {other:?}"
                ),
            }
        }
    };
    let binder = mary::nn::wgpu_bf16_upload::WgpuBf16Uploads::new(device, 427)
        .map_err(anyhow::Error::msg)?;
    let client = binder.client();
    let props = client.properties();
    println!(
        "runtime\t{}\tbackend\t{:?}\tmax_shared\t{}\tmax_units_per_cube\t{}",
        <cubecl::wgpu::WgpuRuntime as cubecl::Runtime>::name(client),
        client.info(),
        props.hardware.max_shared_memory_size,
        props.hardware.max_units_per_cube,
    );
    Ok(binder)
}

#[cfg(feature = "qwen3_5-cuda")]
fn binder() -> Result<mary::models::qwen3_5::Binder> {
    let device = cubecl::cuda::CudaDevice { index: 0 };
    let (name, major, minor) =
        mary::models::qwen3_5::native::device_identity(&device).map_err(anyhow::Error::msg)?;
    println!("runtime\tcuda\tdevice\t{name}\tcompute\t{major}.{minor}");
    mary::nn::cuda_bf16_alias::CudaBf16Aliases::new(device, 427).map_err(anyhow::Error::msg)
}

fn precision() -> Result<DecoderPrecision> {
    DecoderPrecision::parse(&std::env::var("WEMM_PRECISION").unwrap_or_else(|_| "bf16".into()))
        .map_err(anyhow::Error::msg)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 5,
        "usage: wemm_query PILE COLLECTION ROOT QUERIES.txt OUT.f32"
    );
    let passes: usize = std::env::var("WEMM_PASSES")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(2);
    ensure!(passes >= 1, "WEMM_PASSES must be at least 1");
    let precision = precision()?;
    let queries: Vec<String> = fs::read_to_string(&args[3])
        .with_context(|| args[3].clone())?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    ensure!(!queries.is_empty(), "no queries in {}", args[3]);

    let t = Instant::now();
    let store = mary::persist::read_model_pile_snapshot_read_only(Path::new(&args[0]))?;
    let handle = mary::model_collection::parse_collection_handle(&args[1])?;
    let collection = mary::model_collection::ModelCollection::open(&store, handle)
        .map_err(|e| anyhow::anyhow!("open the model collection: {e}"))?;
    let snapshot = mary::model_collection::snapshot_model_collection_for(&store, collection)?;
    let (facts, _, reader) = snapshot.into_parts();
    let root = Id::from_hex(&args[2]).context("ROOT")?;
    let assets = Assets::from_model_root(&facts, &reader, root).map_err(anyhow::Error::msg)?;
    println!(
        "open_ms\t{:.1}\troot\t{root:X}\tprecision\t{}",
        ms(t),
        precision.label()
    );

    let avail = mem_available_kib();
    let t = Instant::now();
    let mut binder = binder()?;
    let init_ms = ms(t);
    let t = Instant::now();
    let (mut roles, mut bytes) = (0usize, 0u64);
    // SAFETY: the pile is append-only and this process never writes it; on
    // CUDA the snapshot's mapped prefix outlives the model, and on wgpu every
    // payload is copied at bind.
    let model = unsafe {
        TextWemm::from_frozen(
            &facts,
            &reader,
            root,
            assets,
            &mut binder,
            |_, _, _, payload| {
                roles += 1;
                bytes += payload.len() as u64;
                Ok(())
            },
            precision,
        )
    }
    .map_err(anyhow::Error::msg)?;
    // Bind returns with uploads possibly in flight; one tiny forward's sync
    // below would fold them into the first query, so wait here.
    cubecl::future::block_on(binder_client_sync(&binder))
        .map_err(|e| anyhow::anyhow!("sync after bind: {e:?}"))?;
    println!(
        "device_init_ms\t{init_ms:.1}\tbind_ms\t{:.1}\troles\t{roles}\tselected_bytes\t{bytes}\tdecoder_bound\t{:?}\tmemavail_drop_kib\t{}",
        ms(t),
        model.decoder_bound(),
        avail as i64 - mem_available_kib() as i64,
    );

    let mut last = Vec::new();
    for pass in 1..=passes {
        let mut vectors = Vec::with_capacity(queries.len() * 4096);
        let all = Instant::now();
        for (index, query) in queries.iter().enumerate() {
            let ids = model
                .codec()
                .text(query)
                .map_err(anyhow::Error::msg)?
                .as_slice()
                .len();
            let t = Instant::now();
            let embedding = model.embed_text(query).map_err(anyhow::Error::msg)?;
            let vector = read(&embedding)?;
            let elapsed = ms(t);
            ensure!(
                vector.len() == 4096,
                "embedding has {} values",
                vector.len()
            );
            let norm = vector
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            println!("pass\t{pass}\tquery\t{index}\tids\t{ids}\tms\t{elapsed:.1}\tnorm\t{norm:.5}");
            vectors.extend(vector);
        }
        println!(
            "pass\t{pass}\ttotal_ms\t{:.1}\tqueries\t{}",
            ms(all),
            queries.len()
        );
        last = vectors;
    }
    let out: Vec<u8> = last.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&args[4], out).with_context(|| args[4].clone())?;
    println!("wrote\t{}\tvectors\t{}", args[4], queries.len());
    Ok(())
}

#[cfg(feature = "qwen3_5-wgpu")]
fn binder_client_sync(
    binder: &mary::models::qwen3_5::Binder,
) -> impl std::future::Future<Output = Result<(), impl std::fmt::Debug>> + '_ {
    binder.client().sync()
}

#[cfg(feature = "qwen3_5-cuda")]
fn binder_client_sync(
    binder: &mary::models::qwen3_5::Binder,
) -> impl std::future::Future<Output = Result<(), impl std::fmt::Debug>> + '_ {
    use cubecl::Runtime;
    let client = cubecl::cuda::CudaRuntime::client(binder.device());
    async move { client.sync().await }
}
