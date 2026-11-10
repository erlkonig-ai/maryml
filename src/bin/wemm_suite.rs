//! Known-answer retrieval suite for WeMM, one decoder precision per process.
//!
//! `wemm_suite embed PILE COLLECTION ROOT ITEMS.jsonl OUT_DIR`
//!   binds the model `ROOT` from the `models` collection `COLLECTION`
//!   (`blake3:HEX` or HEX) of the read-only pile `PILE` as `faculties maintain`
//!   does ([`NativeWemm`] over the frozen collection, assets from the root),
//!   puts the decoder's linear projections on `WEMM_PRECISION` (bf16, w4a4,
//!   w4a16; default bf16) and embeds every item of ITEMS.jsonl. A text item
//!   longer than one 248-ID window is cut to its FIRST window by the greedy
//!   line/word/char rule of `triblespace_search::semantic_wemm`; an image item
//!   is one still image with an optional crop. Texts run `WEMM_BATCH` at a time
//!   (default 8) through `NativeWemm::embed_texts`, longest first; images run
//!   one at a time. `WEMM_PASSES` (default 2) repeats the whole set in-process;
//!   pass 1 includes kernel compilation and is reported apart. `WEMM_COPY=1`
//!   copies the bound BF16 weights into device memory at bind. An NVFP4
//!   precision quantizes each decoder projection on the device as it is
//!   selected and keeps only its NVFP4 planes (`NativeWemm::from_frozen_with`);
//!   `WEMM_POST_BIND=1` instead binds BF16 and quantizes afterwards.
//!   `WEMM_VISION_PRECISION` (bf16, w4a4 or w4a16; default bf16) puts the vision
//!   tower's block and merger linears on that precision at load
//!   (`NativeWemm::from_frozen_with_vision`); linears the NVFP4 kernel cannot
//!   take stay BF16, and the summary counts both.
//!   OUT_DIR receives `embeddings.pass<P>.bf16` ([n, 4096] BF16 in item
//!   order), `items.tsv`, `batches.tsv` and `summary.txt`.
//!
//! `wemm_suite score ITEMS.jsonl TASKS.json EMBEDDINGS.bf16 [RANKS.tsv]`
//!   prints recall@1/5/10 and MRR per task: each query's known answer ranked
//!   by cosine among the task's documents (ties counted against the query),
//!   and optionally writes every query's rank.
//!
//! `wemm_suite same A.bf16 B.bf16` counts differing BF16 words and rows.
//!
//! Model arithmetic is native CUDA. Host arithmetic here is evaluation only
//! (cosines over read-back BF16 embeddings).
use anyhow::{Context, Result, bail, ensure};
use cubecl::cuda::CudaDevice;
use half::bf16;
use mary::models::qwen3_5::{
    gdn_mixer::DecoderPrecision,
    image_prepare::Crop,
    native::{Assets, NativeWemm},
};
use mary::nn::cuda_bf16_alias::{CudaBf16Aliases, WeightResidence};
use serde_json::Value;
use std::{collections::BTreeMap, fmt::Write as _, fs, path::Path, time::Instant};
use triblespace::prelude::Id;

const WINDOW_IDS: usize = 248;

/// Verbatim from triblespace-search semantic_wemm.rs (as wemm_timing).
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

type Cuda = mary::models::qwen3_5::multimodal::CudaTensor;

fn sync(t: &Cuda) -> Result<()> {
    cubecl::future::block_on(t.client.sync()).map_err(|e| anyhow::anyhow!("sync: {e:?}"))
}

fn bits(t: &Cuda) -> Result<Vec<u8>> {
    let b = t.client.read_one(t.handle.clone()).map_err(|e| anyhow::anyhow!("read: {e:?}"))?;
    ensure!(b.len() >= 8192, "embedding shorter than 4096 BF16");
    Ok(b[..8192].to_vec())
}

fn mem_available() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find(|l| l.starts_with("MemAvailable"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
        * 1024
}

enum Item {
    Text(String),
    Image(String, Option<Crop>),
}

fn read_items(path: &Path) -> Result<Vec<(String, Item)>> {
    let mut out = Vec::new();
    for (n, line) in fs::read_to_string(path)?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).with_context(|| format!("item line {}", n + 1))?;
        let id = v["id"].as_str().context("item id")?.to_owned();
        let item = match v["kind"].as_str() {
            Some("text") => Item::Text(v["text"].as_str().context("text")?.to_owned()),
            Some("image") => {
                let crop = v["crop"].as_array().map(|c| Crop {
                    left: c[0].as_u64().unwrap() as usize,
                    top: c[1].as_u64().unwrap() as usize,
                    right: c[2].as_u64().unwrap() as usize,
                    bottom: c[3].as_u64().unwrap() as usize,
                });
                Item::Image(v["path"].as_str().context("path")?.to_owned(), crop)
            }
            other => bail!("item {id}: kind {other:?}"),
        };
        out.push((id, item));
    }
    Ok(out)
}

fn embed(args: &[String]) -> Result<()> {
    ensure!(args.len() == 5, "embed PILE COLLECTION ROOT ITEMS.jsonl OUT_DIR");
    let out = Path::new(&args[4]);
    ensure!(!out.exists(), "{} exists; keep old runs", out.display());
    fs::create_dir_all(out)?;
    let precision = DecoderPrecision::parse(&std::env::var("WEMM_PRECISION").unwrap_or_else(|_| "bf16".into()))
        .map_err(anyhow::Error::msg)?;
    let batch: usize = std::env::var("WEMM_BATCH").ok().map(|s| s.parse()).transpose()?.unwrap_or(8);
    let passes: usize = std::env::var("WEMM_PASSES").ok().map(|s| s.parse()).transpose()?.unwrap_or(2);
    let copy = std::env::var("WEMM_COPY").as_deref() == Ok("1");
    let vision = DecoderPrecision::parse(&std::env::var("WEMM_VISION_PRECISION").unwrap_or_else(|_| "bf16".into()))
        .map_err(anyhow::Error::msg)?;
    // Default: an NVFP4 precision quantizes each projection at load and never
    // holds it in BF16. WEMM_POST_BIND=1 binds BF16 first and quantizes after.
    let post_bind = std::env::var("WEMM_POST_BIND").as_deref() == Ok("1");
    let items = read_items(Path::new(&args[3]))?;
    let mut summary = String::new();
    macro_rules! say { ($($t:tt)*) => {{ let line = format!($($t)*); println!("{line}"); writeln!(summary, "{line}").unwrap(); }}; }
    say!("precision\t{}\tvision_precision\t{}\tbatch\t{batch}\tpasses\t{passes}\tresidence\t{}\titems\t{}", precision.label(),
        vision.label(), if copy { "device-copy" } else { "pile-aliased" }, items.len());

    let t = Instant::now();
    let store = mary::persist::read_model_pile_snapshot_read_only(Path::new(&args[0]))?;
    let handle = mary::model_collection::parse_collection_handle(&args[1])?;
    let collection = mary::model_collection::ModelCollection::open(&store, handle)
        .map_err(|e| anyhow::anyhow!("open the model collection: {e}"))?;
    let snapshot = mary::model_collection::snapshot_model_collection_for(&store, collection)?;
    let (facts, _, reader) = snapshot.into_parts();
    let root = Id::from_hex(&args[2]).context("ROOT")?;
    let assets = Assets::from_model_root(&facts, &reader, root).map_err(anyhow::Error::msg)?;
    let codec = assets.codec().clone();
    say!("open_ms\t{:.1}\troot\t{root:X}", ms(t));

    let device = CudaDevice { index: 0 };
    let avail0 = mem_available();
    let t = Instant::now();
    let mut aliases = CudaBf16Aliases::new(device.clone(), 759)
        .map_err(anyhow::Error::msg)?
        .with_residence(if copy { WeightResidence::DeviceCopy } else { WeightResidence::PileAliased });
    let mut selected = 0usize;
    let mut selected_bytes = 0u64;
    // SAFETY: the pile is append-only; this process never writes it, and the
    // snapshot's prefix stays valid while the mapping lives.
    let mut native = unsafe {
        NativeWemm::from_frozen_with_vision(&facts, &reader, root, assets, &mut aliases, |_, _, _, payload| {
            selected += 1;
            selected_bytes += payload.len() as u64;
            Ok(())
        }, if post_bind { DecoderPrecision::Bf16 } else { precision }, vision)
    }?;
    let stats = aliases.stats();
    say!("bind_ms\t{:.1}\troles\t{selected}\tselected_bytes\t{selected_bytes}\tcopied_bytes\t{}\tregistrations\t{}",
        ms(t), stats.copied_bytes, stats.registrations);
    if post_bind {
        let t = Instant::now();
        let nvfp4_bytes = native.quantize_decoder(precision).map_err(anyhow::Error::msg)?;
        say!("post_bind_quantize_ms\t{:.1}\tnvfp4_plane_bytes\t{nvfp4_bytes}", ms(t));
    }
    let (projections, bf16_bytes, fp4_bytes) = native.projection_bytes();
    say!("decoder_projections\t{projections}\tresident_bf16_bytes\t{bf16_bytes}\tresident_nvfp4_plane_bytes\t{fp4_bytes}");
    let (vl, vl_bf16, vl_bf16_bytes, vl_fp4, vl_fp4_bytes, vl_kinds) = native.vision_linear_bytes();
    say!("vision_linears\t{vl}\tbf16\t{vl_bf16}\tresident_bf16_bytes\t{vl_bf16_bytes}\tnvfp4\t{vl_fp4}\tresident_nvfp4_plane_bytes\t{vl_fp4_bytes}\tbound\t{vl_kinds:?}");
    let (name, major, minor) = native.device_identity().map_err(anyhow::Error::msg)?;
    say!("device\t{name}\tcompute\t{major}.{minor}\tmemavail_drop_after_bind_bytes\t{}", avail0 as i64 - mem_available() as i64);

    // Prepare: text windows and token counts, once.
    let mut tsv = String::from("index\tid\tkind\tids\tcut\n");
    let mut texts: Vec<(usize, String, usize)> = Vec::new();
    let mut images: Vec<(usize, Vec<u8>, Option<Crop>)> = Vec::new();
    for (index, (id, item)) in items.iter().enumerate() {
        match item {
            Item::Text(text) => {
                let fits = |w: &str| codec.text(w).is_ok_and(|ids| ids.as_slice().len() <= WINDOW_IDS);
                let cut = windows(text, 1, fits);
                ensure!(!cut.is_empty(), "item {id}: empty text");
                let window = cut[0].to_owned();
                let ids = codec.text(&window).map_err(anyhow::Error::msg)?.as_slice().len();
                let was_cut = window.len() < text.trim().len();
                writeln!(tsv, "{index}\t{id}\ttext\t{ids}\t{}", u8::from(was_cut))?;
                texts.push((index, window, ids));
            }
            Item::Image(path, crop) => {
                let bytes = fs::read(path).with_context(|| path.clone())?;
                writeln!(tsv, "{index}\t{id}\timage\t\t{}", u8::from(crop.is_some()))?;
                images.push((index, bytes, *crop));
            }
        }
    }
    fs::write(out.join("items.tsv"), tsv)?;
    let mut order: Vec<usize> = (0..texts.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(texts[i].2), texts[i].0));
    let groups: Vec<Vec<usize>> = order.chunks(batch.max(1)).map(|c| c.to_vec()).collect();

    let mut batches = String::from("pass\tkind\tgroup\tB\tmax_ids\tvalid_ids\tms\n");
    for pass in 1..=passes {
        let mut emb: Vec<Vec<u8>> = vec![Vec::new(); items.len()];
        let all = Instant::now();
        let (mut text_ms, mut image_ms, mut ids_total) = (0.0, 0.0, 0usize);
        for (g, group) in groups.iter().enumerate() {
            let window: Vec<&str> = group.iter().map(|&i| texts[i].1.as_str()).collect();
            let t = Instant::now();
            let es = native.embed_texts(&window).map_err(anyhow::Error::msg)?;
            for e in &es {
                sync(e)?;
            }
            let f = ms(t);
            text_ms += f;
            let max = group.iter().map(|&i| texts[i].2).max().unwrap_or(0);
            let valid: usize = group.iter().map(|&i| texts[i].2).sum();
            ids_total += valid;
            writeln!(batches, "{pass}\ttext\t{g}\t{}\t{max}\t{valid}\t{f:.2}", group.len())?;
            for (&i, e) in group.iter().zip(&es) {
                emb[texts[i].0] = bits(e)?;
            }
        }
        for (g, (index, bytes, crop)) in images.iter().enumerate() {
            let t = Instant::now();
            let e = native.embed_image(bytes, *crop).map_err(anyhow::Error::msg)?;
            sync(&e)?;
            let f = ms(t);
            image_ms += f;
            writeln!(batches, "{pass}\timage\t{g}\t1\t\t\t{f:.2}")?;
            emb[*index] = bits(&e)?;
        }
        let missing = emb.iter().filter(|e| e.len() != 8192).count();
        ensure!(missing == 0, "{missing} items without an embedding");
        fs::write(out.join(format!("embeddings.pass{pass}.bf16")), emb.concat())?;
        say!("pass\t{pass}\twall_ms\t{:.1}\ttext_forward_ms\t{text_ms:.1}\ttext_windows\t{}\ttext_valid_ids\t{ids_total}\timage_forward_ms\t{image_ms:.1}\timages\t{}",
            ms(all), texts.len(), images.len());
        say!("pass\t{pass}\ttext_ms_per_window\t{:.2}\ttext_ms_per_1k_ids\t{:.2}\timage_ms_per_image\t{:.2}",
            text_ms / texts.len().max(1) as f64, text_ms / ids_total.max(1) as f64 * 1e3, image_ms / images.len().max(1) as f64);
    }
    let pool = native_pool(&native)?;
    say!("pool_reserved_bytes\t{}\tpool_in_use_bytes\t{}\tmemavail_drop_end_bytes\t{}", pool.0, pool.1, avail0 as i64 - mem_available() as i64);
    fs::write(out.join("batches.tsv"), batches)?;
    fs::write(out.join("summary.txt"), summary)?;
    Ok(())
}

fn native_pool(native: &NativeWemm) -> Result<(u64, u64)> {
    let client = <cubecl::cuda::CudaRuntime as cubecl::Runtime>::client(native.device());
    let usage = client.memory_usage().map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok((usage.bytes_reserved, usage.bytes_in_use))
}

fn rows(path: &Path) -> Result<Vec<Vec<f32>>> {
    let b = fs::read(path)?;
    ensure!(b.len() % 8192 == 0, "{}: not [n,4096] BF16", path.display());
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

fn score(args: &[String]) -> Result<()> {
    ensure!((3..=4).contains(&args.len()), "score ITEMS.jsonl TASKS.json EMBEDDINGS.bf16 [RANKS.tsv]");
    let mut ranks_out = String::from("task\tquery\tanswer\trank\n");
    let items = read_items(Path::new(&args[0]))?;
    let index: BTreeMap<&str, usize> = items.iter().enumerate().map(|(i, (id, _))| (id.as_str(), i)).collect();
    let tasks: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let emb = rows(Path::new(&args[2]))?;
    ensure!(emb.len() == items.len(), "{} embeddings for {} items", emb.len(), items.len());
    println!("task\tqueries\tdocs\tR@1\tR@5\tR@10\tMRR\tmean_rank");
    for task in tasks.as_array().context("tasks array")? {
        let name = task["name"].as_str().context("task name")?;
        let docs: Vec<usize> = task["docs"].as_array().context("docs")?.iter()
            .map(|d| index.get(d.as_str().unwrap()).copied().context("doc id")).collect::<Result<_>>()?;
        let pairs = task["pairs"].as_array().context("pairs")?;
        let (mut r1, mut r5, mut r10, mut mrr, mut ranks) = (0usize, 0usize, 0usize, 0f64, 0f64);
        for pair in pairs {
            let q = index[pair[0].as_str().unwrap()];
            let a = index[pair[1].as_str().unwrap()];
            let target = cosine(&emb[q], &emb[a]);
            // Rank 1 + documents scoring at least the answer (ties against).
            let rank = 1 + docs.iter().filter(|&&d| d != a && cosine(&emb[q], &emb[d]) >= target).count();
            r1 += usize::from(rank <= 1);
            r5 += usize::from(rank <= 5);
            r10 += usize::from(rank <= 10);
            mrr += 1.0 / rank as f64;
            ranks += rank as f64;
            writeln!(ranks_out, "{name}\t{}\t{}\t{rank}", pair[0].as_str().unwrap(), pair[1].as_str().unwrap())?;
        }
        let n = pairs.len() as f64;
        println!("{name}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.2}", pairs.len(), docs.len(),
            r1 as f64 / n, r5 as f64 / n, r10 as f64 / n, mrr / n, ranks / n);
    }
    if let Some(path) = args.get(3) {
        fs::write(path, ranks_out)?;
    }
    Ok(())
}

fn same(args: &[String]) -> Result<()> {
    ensure!(args.len() == 2, "same A.bf16 B.bf16");
    let a = fs::read(&args[0])?;
    let b = fs::read(&args[1])?;
    ensure!(a.len() == b.len() && a.len() % 8192 == 0, "sizes {} and {}", a.len(), b.len());
    let words = a.chunks_exact(2).zip(b.chunks_exact(2)).filter(|(x, y)| x != y).count();
    let rows = a.chunks_exact(8192).zip(b.chunks_exact(8192)).filter(|(x, y)| x != y).count();
    let (ra, rb) = (rows_of_bytes(&a), rows_of_bytes(&b));
    let mut cos: Vec<f64> = ra.iter().zip(&rb).map(|(x, y)| cosine(x, y)).collect();
    cos.sort_by(f64::total_cmp);
    let q = |p: f64| cos[((cos.len() - 1) as f64 * p) as usize];
    let mean = cos.iter().sum::<f64>() / cos.len() as f64;
    println!("rows\t{}\tdiffering_rows\t{rows}\tdiffering_words\t{words}\tmin_row_cosine\t{:.6}\tp01\t{:.6}\tmedian\t{:.6}\tmean\t{mean:.6}",
        a.len() / 8192, q(0.0), q(0.01), q(0.5));
    Ok(())
}

fn rows_of_bytes(b: &[u8]) -> Vec<Vec<f32>> {
    b.chunks_exact(8192)
        .map(|r| r.chunks_exact(2).map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect())
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("embed") => embed(&args[1..]),
        Some("score") => score(&args[1..]),
        Some("same") => same(&args[1..]),
        _ => bail!("wemm_suite embed|score|same ... (see the module docs)"),
    }
}
