//! Where WeMM derivation time goes, per phase, on the production path.
//!
//! `wemm_timing embed MODEL_PILE ASSETS_DIR TEXTS QUERIES OUT_DIR [CAP] [ROOT]`
//! binds the model exactly as `faculties maintain` does ([`NativeWemm`] over a
//! read-only pile, 759 BF16 aliases), cuts each text into windows the way
//! `triblespace_search::semantic_wemm` does (same greedy line/word/char rule,
//! same 248-ID bound, same cap), and times each phase with a device sync
//! after it: windowing (CPU, tokenizer calls counted), the forward of each
//! window, and the NVFP4 row encode (which reads its row back). TEXTS is a file
//! naming one text file per line; QUERIES one query per line. OUT_DIR receives
//! `windows.tsv` (one row per window), the BF16 embeddings of every window and
//! query, and `summary.txt`.
//!
//! `wemm_timing compare OLD_DIR NEW_DIR` compares two runs over the same
//! inputs: cosine of every window embedding and query embedding, and whether
//! each query's five nearest windows (max over a text's windows, as the index
//! scores) change.
//!
//! Host arithmetic here is diagnostic only (cosines over read-back BF16).
use anyhow::{Context, Result, bail, ensure};
use cubecl::cuda::CudaDevice;
use half::bf16;
use mary::models::qwen3_5::native::{Assets, NativeWemm};
use mary::nn::{cuda_bf16_alias::CudaBf16Aliases, nvfp4_cosine::cuda_encode::CudaRowEncoder4096};
use std::{fmt::Write as _, fs, path::Path, time::Instant};
use triblespace::prelude::Id;

const WINDOW_IDS: usize = 248;

/// Verbatim from triblespace-search semantic_wemm.rs (e6660fd3).
fn windows<'t>(text: &'t str, cap: u64, mut fits: impl FnMut(&str) -> bool) -> Vec<&'t str> {
    let units: [fn(&str) -> usize; 3] = [
        |s| s.find('\n').map_or(s.len(), |at| at + 1),
        |s| {
            let word = s.find(char::is_whitespace).unwrap_or(s.len());
            s.len() - s[word..].trim_start().len()
        },
        |s| s.chars().next().map_or(0, char::len_utf8),
    ];
    let mut windows = Vec::new();
    let mut rest = text.trim_start();
    while !rest.is_empty() && (windows.len() as u64) < cap {
        let mut end = 0;
        for unit in units {
            while end < rest.len() {
                let next = end + unit(&rest[end..]);
                if !fits(&rest[..next]) {
                    break;
                }
                end = next;
            }
            if end > 0 {
                break;
            }
        }
        let end = end.max(units[2](rest));
        windows.push(rest[..end].trim_end());
        rest = rest[end..].trim_start();
    }
    windows
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn sync(t: &mary::models::qwen3_5::multimodal::CudaTensor) -> Result<()> {
    cubecl::future::block_on(t.client.sync()).map_err(|e| anyhow::anyhow!("sync: {e:?}"))
}

fn bits(t: &mary::models::qwen3_5::multimodal::CudaTensor) -> Result<Vec<u8>> {
    Ok(t.client
        .read_one(t.handle.clone())
        .map_err(|e| anyhow::anyhow!("read: {e:?}"))?
        .to_vec())
}

fn embed(args: &[String]) -> Result<()> {
    ensure!(
        (5..=7).contains(&args.len()),
        "embed MODEL_PILE ASSETS_DIR TEXTS QUERIES OUT_DIR [CAP] [ROOT]"
    );
    let out = Path::new(&args[4]);
    ensure!(!out.exists(), "{} exists; keep old runs", out.display());
    fs::create_dir_all(out)?;
    let cap: u64 = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(16);
    let mut summary = String::new();
    macro_rules! say { ($($t:tt)*) => {{ let line = format!($($t)*); println!("{line}"); writeln!(summary, "{line}").unwrap(); }}; }

    let texts: Vec<(String, String)> = fs::read_to_string(&args[2])?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|p| Ok((p.to_owned(), fs::read_to_string(p).with_context(|| p.to_owned())?)))
        .collect::<Result<_>>()?;
    let queries: Vec<String> = fs::read_to_string(&args[3])?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(str::to_owned)
        .collect();

    let t = Instant::now();
    // WEMM_COLLECTION names the `models` collection explicitly (a working pile
    // carries several); without it the pile must hold exactly one.
    let model = match std::env::var("WEMM_COLLECTION") {
        Ok(hex) => {
            let store = mary::persist::read_model_pile_snapshot_read_only(Path::new(&args[0]))?;
            let handle = mary::model_collection::parse_collection_handle(&hex)?;
            let collection = mary::model_collection::ModelCollection::open(&store, handle)
                .map_err(|e| anyhow::anyhow!("open the model collection: {e}"))?;
            let snapshot = mary::model_collection::snapshot_model_collection_for(&store, collection)?;
            let (facts, _, store) = snapshot.into_parts();
            mary::persist::ModelPileSource { facts, store, collection }
        }
        Err(_) => mary::persist::read_model_pile_read_only(Path::new(&args[0]))?,
    };
    let root = match args.get(6) {
        Some(hex) => Id::from_hex(hex).context("ROOT")?,
        None => mary::selection::select_model_root(
            &model.facts,
            &model.store,
            mary::selection::ModelSelector::Only,
        )?,
    };
    say!("open_pile_ms\t{:.1}\troot\t{root:X}", ms(t));
    let dir = Path::new(&args[1]);
    let assets = Assets::from_bytes(
        &fs::read(dir.join("config.json"))?,
        &fs::read(dir.join("tokenizer.json"))?,
        &fs::read(dir.join("chat_template.jinja"))?,
    )
    .map_err(anyhow::Error::msg)?;
    let codec = assets.codec().clone();
    let device = CudaDevice { index: 0 };
    let t = Instant::now();
    let mut aliases = CudaBf16Aliases::new(device.clone(), 759).map_err(anyhow::Error::msg)?;
    // SAFETY: the model pile is an immutable read-only copy for this process.
    let mut native = unsafe {
        NativeWemm::from_frozen(&model.facts, &model.store, root, assets, &mut aliases, |_, _, _, _| Ok(()))
    }?;
    let encoder = CudaRowEncoder4096::new(device).map_err(|e| anyhow::anyhow!("{e}"))?;
    say!("bind_ms\t{:.1}", ms(t));

    // First forward compiles every kernel; report it apart.
    let t = Instant::now();
    let warm = native.embed_text("warm up the kernels").map_err(anyhow::Error::msg)?;
    sync(&warm)?;
    encoder.encode(&warm).map_err(|e| anyhow::anyhow!("{e}"))?;
    say!("first_forward_ms\t{:.1}", ms(t));
    for n in [8usize, 64, 248] {
        // Steady-state forward vs token count, after warm-up.
        let text = "word ".repeat(n);
        let window = windows(&text, 1, |w| codec.text(w).is_ok_and(|ids| ids.as_slice().len() <= n));
        let ids = codec.text(window[0]).unwrap().as_slice().len();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let e = native.embed_text(window[0]).map_err(anyhow::Error::msg)?;
            sync(&e)?;
            best = best.min(ms(t));
        }
        say!("steady_forward\tids\t{ids}\tbest_of_3_ms\t{best:.1}");
    }

    let mut tsv = String::from("text\twindow\tchars\tids\twindow_cut_ms\tforward_ms\tencode_ms\n");
    let mut embeddings = Vec::new();
    let mut rows_bytes = Vec::new();
    let (mut cut_total, mut fwd_total, mut enc_total, mut ids_total, mut n_windows) = (0.0, 0.0, 0.0, 0usize, 0usize);
    let all = Instant::now();
    for (index, (path, text)) in texts.iter().enumerate() {
        let t = Instant::now();
        let mut calls = 0usize;
        let cut = windows(text, cap, |w| {
            calls += 1;
            codec.text(w).is_ok_and(|ids| ids.as_slice().len() <= WINDOW_IDS)
        });
        let cut_ms = ms(t);
        cut_total += cut_ms;
        let mut text_fwd = 0.0;
        for (w, window) in cut.iter().enumerate() {
            let ids = codec.text(window).map_err(anyhow::Error::msg)?.as_slice().len();
            let t = Instant::now();
            let e = native.embed_text(window).map_err(anyhow::Error::msg)?;
            sync(&e)?;
            let fwd = ms(t);
            let t = Instant::now();
            let row = encoder.encode(&e).map_err(|e| anyhow::anyhow!("{e}"))?;
            let enc = ms(t);
            for stage in row.stages() {
                rows_bytes.extend(stage.global_scale_bytes());
                rows_bytes.extend(stage.block_scales());
                rows_bytes.extend(stage.codes());
            }
            rows_bytes.extend(row.reconstruction_norm_bytes());
            rows_bytes.extend(row.error_bound_bytes());
            embeddings.extend(bits(&e)?);
            writeln!(tsv, "{index}\t{w}\t{}\t{ids}\t{}\t{fwd:.2}\t{enc:.2}", window.len(),
                if w == 0 { format!("{cut_ms:.2}") } else { String::new() })?;
            fwd_total += fwd;
            enc_total += enc;
            text_fwd += fwd;
            ids_total += ids;
            n_windows += 1;
        }
        println!("text {index} {} bytes {} windows {} tokenizer calls {calls} cut {cut_ms:.1} ms forward {text_fwd:.1} ms", path, text.len(), cut.len());
    }
    let wall = ms(all);
    say!("texts\t{}\twindows\t{n_windows}\tids\t{ids_total}\tcap\t{cap}", texts.len());
    say!("wall_ms\t{wall:.1}\twindow_cut_ms\t{cut_total:.1}\tforward_ms\t{fwd_total:.1}\tencode_ms\t{enc_total:.1}\tother_ms\t{:.1}", wall - cut_total - fwd_total - enc_total);
    say!("forward_ms_per_window\t{:.1}\tforward_ms_per_1k_ids\t{:.1}\tencode_ms_per_window\t{:.2}",
        fwd_total / n_windows as f64, fwd_total / ids_total as f64 * 1e3, enc_total / n_windows as f64);
    // 2 * params * tokens: 6.92e9 decoder linear weights per token (32 layers,
    // excluding the vision tower and the embedding gather).
    say!("decoder_tflops_effective\t{:.1}", 2.0 * 6.92e9 * ids_total as f64 / (fwd_total / 1e3) / 1e12);
    fs::write(out.join("windows.tsv"), tsv)?;
    fs::write(out.join("windows.bf16"), &embeddings)?;
    fs::write(out.join("rows.nvfp4"), &rows_bytes)?;
    let mut q = Vec::new();
    let t = Instant::now();
    for query in &queries {
        let e = native.embed_text(query).map_err(anyhow::Error::msg)?;
        q.extend(bits(&e)?);
    }
    say!("queries\t{}\tms\t{:.1}", queries.len(), ms(t));
    fs::write(out.join("queries.bf16"), q)?;
    fs::write(out.join("queries.txt"), queries.join("\n"))?;
    fs::write(out.join("summary.txt"), summary)?;
    Ok(())
}

fn rows(path: &Path) -> Result<Vec<Vec<f32>>> {
    let b = fs::read(path)?;
    ensure!(b.len() % (4096 * 2) == 0, "{}: not [n,4096] BF16", path.display());
    Ok(b.chunks_exact(8192)
        .map(|r| r.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect())
        .collect())
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut x, mut y) = (0f64, 0f64, 0f64);
    for (&a, &b) in a.iter().zip(b) {
        d += a as f64 * b as f64;
        x += a as f64 * a as f64;
        y += b as f64 * b as f64;
    }
    d / (x.sqrt() * y.sqrt())
}
fn text_of(tsv: &str) -> Vec<usize> {
    tsv.lines().skip(1).map(|l| l.split('\t').next().unwrap().parse().unwrap()).collect()
}
fn top(query: &[f32], windows: &[Vec<f32>], owner: &[usize], k: usize) -> Vec<usize> {
    let mut best: std::collections::BTreeMap<usize, f64> = Default::default();
    for (w, o) in windows.iter().zip(owner) {
        let c = cosine(query, w);
        let e = best.entry(*o).or_insert(f64::MIN);
        *e = e.max(c);
    }
    let mut v: Vec<_> = best.into_iter().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.into_iter().take(k).map(|(t, _)| t).collect()
}

fn compare(args: &[String]) -> Result<()> {
    ensure!(args.len() == 2, "compare OLD_DIR NEW_DIR");
    let (a, b) = (Path::new(&args[0]), Path::new(&args[1]));
    let (wa, wb) = (rows(&a.join("windows.bf16"))?, rows(&b.join("windows.bf16"))?);
    let owner = text_of(&fs::read_to_string(a.join("windows.tsv"))?);
    ensure!(owner == text_of(&fs::read_to_string(b.join("windows.tsv"))?), "different windowing");
    ensure!(wa.len() == wb.len(), "different window counts");
    let mut cos: Vec<f64> = wa.iter().zip(&wb).map(|(x, y)| cosine(x, y)).collect();
    let identical = wa.iter().zip(&wb).filter(|(x, y)| x == y).count();
    cos.sort_by(f64::total_cmp);
    let mean = cos.iter().sum::<f64>() / cos.len() as f64;
    println!("windows {} bit_identical {identical} cosine min {:.6} p01 {:.6} median {:.6} mean {:.6}",
        cos.len(), cos[0], cos[cos.len() / 100], cos[cos.len() / 2], mean);
    match (fs::read(a.join("rows.nvfp4")), fs::read(b.join("rows.nvfp4"))) {
        (Ok(x), Ok(y)) => {
            let same = x.chunks(4624).zip(y.chunks(4624)).filter(|(p, q)| p == q).count();
            println!("nvfp4 rows {} byte_identical {same}", x.len() / 4624);
        }
        _ => println!("nvfp4 rows: not recorded in both runs"),
    }
    let (qa, qb) = (rows(&a.join("queries.bf16"))?, rows(&b.join("queries.bf16"))?);
    let names: Vec<String> = fs::read_to_string(a.join("queries.txt"))?.lines().map(str::to_owned).collect();
    let mut same_top1 = 0;
    let mut same_top5 = 0;
    for (i, (x, y)) in qa.iter().zip(&qb).enumerate() {
        let (ta, tb) = (top(x, &wa, &owner, 5), top(y, &wb, &owner, 5));
        same_top1 += (ta[0] == tb[0]) as usize;
        same_top5 += (ta == tb) as usize;
        println!("query {i} cos {:.6} old_top5 {ta:?} new_top5 {tb:?} :: {}", cosine(x, y), names[i]);
    }
    println!("queries {} same_top1 {same_top1} same_top5_order {same_top5}", qa.len());
    Ok(())
}

// ---- gemm: the decoder's projection shapes, production vs forced tilings ----

type Tensor = mary::models::qwen3_5::multimodal::CudaTensor;

fn upload_bf16(client: &cubecl::client::ComputeClient<cubecl::cuda::CudaRuntime>, n: usize, seed: u64, scale: f32) -> cubecl::server::Handle {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut bytes = Vec::with_capacity(n * 2);
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        bytes.extend(bf16::from_f32(u * scale).to_le_bytes());
    }
    client.create_from_slice(&bytes)
}

#[derive(Clone, Copy, Debug)]
enum Routine { Simple, Double, SimpleAsync, DoubleAsync }

fn forced(routine: Routine, m: usize, n: usize, k: usize, part: (u32, u32, u32), stage: (u32, u32), double_partition: bool, swizzle: bool)
    -> Result<cubek::matmul::launch::Strategy, String> {
    use cubek::matmul::{components::{stage::PartitionBuffering, tile::TileMatmulKind},
        definition::{MatmulGlobalElems, MatmulProblem, TilingBlueprint, TilingScheme}, launch::Strategy,
        routines::BlueprintStrategy};
    use cubek::std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    let bf = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
    let globals = MatmulGlobalElems { lhs: bf, rhs: bf, out: StorageType::Scalar(ElemType::Float(FloatKind::F32)) };
    let problem = MatmulProblem::from_shapes_and_strides(
        [m, k].into(), [k, n].into(), [m, n].into(), [k, 1].into(), [1, k].into(), [n, 1].into(),
        globals, cubecl::ir::AddressType::U32, None, None).map_err(|e| format!("{e:?}"))?;
    let scheme = TilingScheme::builder().with_tile_size((16, 8, 16).into())
        .with_partition_size(part.into()).with_stage_size((stage.0, stage.1, 1).into()).build()?;
    let grid = HypercubeBlueprint::builder()
        .global_order(if swizzle { GlobalOrder::SwizzleRow(4) } else { GlobalOrder::RowMajor })
        .cube_count_strategy(CubeCountStrategy::FromProblem).build();
    let mut bp = TilingBlueprint::builder(TileMatmulKind::Mma, scheme, 32, &problem)
        .partition_buffering(if double_partition { PartitionBuffering::Double } else { PartitionBuffering::Single })
        .hypercube_blueprint(grid).build();
    bp.check_m_bounds = true; bp.check_n_bounds = true; bp.check_k_bounds = true;
    Ok(match routine {
        Routine::Simple => Strategy::SimpleCyclicMma(BlueprintStrategy::Forced(bp)),
        Routine::Double => Strategy::DoubleCyclicMma(BlueprintStrategy::Forced(bp)),
        Routine::SimpleAsync => Strategy::SimpleAsyncCyclicMma(BlueprintStrategy::Forced(bp)),
        Routine::DoubleAsync => Strategy::DoubleAsyncCyclicMma(BlueprintStrategy::Forced(bp)),
    })
}

fn gemm(args: &[String]) -> Result<()> {
    use burn::tensor::DType;
    use cubecl::{Runtime, cuda::CudaRuntime, prelude::TensorBinding};
    use cubek::matmul::definition::{MatmulElems, MatmulGlobalElems};
    use cubek::std::InputBinding;
    use cubecl::ir::{ElemType, FloatKind, StorageType};
    let m: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(248);
    let iters: usize = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(10);
    let only: Option<String> = args.get(2).cloned();
    let device = CudaDevice { index: 0 };
    let client = CudaRuntime::client(&device);
    let bf = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
    let f32s = StorageType::Scalar(ElemType::Float(FloatKind::F32));
    // (k, n, uses per 32-layer forward)
    let shapes: &[(usize, usize, usize)] = &[(4096, 12288, 64), (12288, 4096, 32), (4096, 8192, 32),
        (4096, 4096, 56), (4096, 1024, 16), (4096, 32, 48)];
    type C = (&'static str, Routine, (u32, u32, u32), (u32, u32), bool, bool);
    let candidates: &[C] = &[
        ("simple p2x1x4 s2x2 (=prod)", Routine::Simple, (2, 1, 4), (2, 2), false, false),
        ("simple p2x8x2 s4x1", Routine::Simple, (2, 8, 2), (4, 1), false, false),
        ("simple p2x8x2 s8x1", Routine::Simple, (2, 8, 2), (8, 1), false, false),
        ("simple p2x8x2 s2x1", Routine::Simple, (2, 8, 2), (2, 1), false, false),
        ("simple p2x8x2 s4x2", Routine::Simple, (2, 8, 2), (4, 2), false, false),
        ("simple p4x8x2 s2x1", Routine::Simple, (4, 8, 2), (2, 1), false, false),
        ("simple p4x8x2 s4x1", Routine::Simple, (4, 8, 2), (4, 1), false, false),
        ("simple p2x16x2 s4x1", Routine::Simple, (2, 16, 2), (4, 1), false, false),
        ("simple p4x4x2 s4x1", Routine::Simple, (4, 4, 2), (4, 1), false, false),
        ("simple p2x4x2 s4x1", Routine::Simple, (2, 4, 2), (4, 1), false, false),
        ("simple p2x8x1 s4x1", Routine::Simple, (2, 8, 1), (4, 1), false, false),
        ("simple p2x8x2 s4x1 swz", Routine::Simple, (2, 8, 2), (4, 1), false, true),
        ("sasync p4x8x2 s2x1", Routine::SimpleAsync, (4, 8, 2), (2, 1), false, false),
        ("dasync p2x8x2 s4x1", Routine::DoubleAsync, (2, 8, 2), (4, 1), false, false),
        ("dasync p4x8x2 s2x1", Routine::DoubleAsync, (4, 8, 2), (2, 1), false, false),
        ("dasync p2x8x2 s8x1", Routine::DoubleAsync, (2, 8, 2), (8, 1), false, false),
        ("dasync p2x8x4 s4x1", Routine::DoubleAsync, (2, 8, 4), (4, 1), false, false),
    ];
    let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        f()?; f()?;
        cubecl::future::block_on(client.sync()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let t = Instant::now();
        for _ in 0..iters { f()?; }
        cubecl::future::block_on(client.sync()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(ms(t) / iters as f64)
    };
    let mut totals = vec![0f64; candidates.len() + 1];
    let mut valid = vec![true; candidates.len() + 1];
    println!("m {m} iters {iters}");
    for &(k, n, uses) in shapes {
        let a = upload_bf16(&client, m * k, 1 + k as u64, 1.0);
        let w = upload_bf16(&client, n * k, 7 + n as u64, 0.02);
        let input = Tensor::new_contiguous(client.clone(), device.clone(), [1, m, k].as_slice().into(), a.clone(), DType::BF16);
        let weight = Tensor::new_contiguous(client.clone(), device.clone(), [n, k].as_slice().into(), w.clone(), DType::BF16);
        let project = || mary::models::qwen3_5::gdn_mixer::project(&input, &weight).map_err(anyhow::Error::msg);
        let reference = project()?;
        let reference = client.read_one(reference.handle.clone()).map_err(|e| anyhow::anyhow!("{e:?}"))?.to_vec();
        let t = time(&mut || { project()?; Ok(()) })?;
        let flops = 2.0 * (m * n * k) as f64;
        println!("k {k} n {n} uses {uses}: production {t:.3} ms {:.1} TFLOP/s -> {:.1} ms/forward", flops / t / 1e9, t * uses as f64);
        totals[0] += t * uses as f64;
        for (ci, &(name, routine, part, stage, dpart, swz)) in candidates.iter().enumerate() {
            if only.as_deref().is_some_and(|o| !name.contains(o)) { valid[ci + 1] = false; continue; }
            let strategy = match forced(routine, m, n, k, part, stage, dpart, swz) {
                Ok(s) => s,
                Err(e) => { println!("  {name}: setup {e}"); valid[ci + 1] = false; continue; }
            };
            let out = client.empty(m * n * 4);
            let mut run = || -> Result<()> {
                let bind = |h: &cubecl::server::Handle, shape: [usize; 2], strides: [usize; 2]| TensorBinding::<CudaRuntime> {
                    handle: h.clone().binding(), shape: shape.into(), strides: strides.into(), runtime: core::marker::PhantomData };
                let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems { lhs: bf, rhs: bf, out: f32s });
                cubek::matmul::launch::launch_ref(&strategy, &client,
                    InputBinding::new(bind(&a, [m, k], [k, 1]), bf), InputBinding::new(bind(&w, [k, n], [1, k]), bf),
                    bind(&out, [m, n], [n, 1]), &mut dtypes).map_err(|e| anyhow::anyhow!("{e:?}"))
            };
            if let Err(e) = run() { println!("  {name}: launch {e}"); valid[ci + 1] = false; continue; }
            let got = client.read_one(out.clone()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let rounded: Vec<u8> = got.chunks_exact(4).flat_map(|c| bf16::from_f32(f32::from_le_bytes(c.try_into().unwrap())).to_le_bytes()).collect();
            let differ = rounded.chunks_exact(2).zip(reference.chunks_exact(2)).filter(|(x, y)| x != y).count();
            let t = time(&mut run)?;
            println!("  {name}: {t:.3} ms {:.1} TFLOP/s bf16_differ {differ}/{}", flops / t / 1e9, m * n);
            totals[ci + 1] += t * uses as f64;
        }
    }
    println!("per-forward projection total (m {m}): production {:.1} ms", totals[0]);
    for (ci, c) in candidates.iter().enumerate() {
        if valid[ci + 1] { println!("  {}: {:.1} ms (excl. f32->bf16 round)", c.0, totals[ci + 1]); }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("embed") => embed(&args[1..]),
        Some("compare") => compare(&args[1..]),
        Some("gemm") => gemm(&args[1..]),
        _ => bail!("usage: wemm_timing embed ... | compare OLD NEW"),
    }
}
