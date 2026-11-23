//! Images through the Inkling 2 Mini session ([`MiniSession::append_with_image_rows`])
//! on the release's weights in a pile, against upstream's whole run of the
//! same prompt. Nothing is sampled.
//!
//! The reference is `golden/capture_inkling2_images.py` (transformers 046bdb8,
//! CPU, f32 and bf16, each run whole): the published processor and chat
//! template render one user message with the two vision fixture PNGs and
//! text, and `Inkling2LoopForConditionalGeneration.forward(input_ids,
//! pixel_values, image_grid_thw)` runs it. It records the decoder's
//! `inputs_embeds` after the `masked_scatter`, the tower's features and every
//! row's logits.
//!
//! What is compared:
//!
//! 1. **Binding.** The model opens without the vision tower; an empty
//!    [`MiniModel::project_images`] leaves it unbound; the first real one binds
//!    it. Device memory is read before and after.
//! 2. **Features.** Mary's own preprocessing of the PNGs and its tower,
//!    against the reference's `pixel_values` and tower output.
//! 3. **Decoder input.** [`MiniModel::decoder_inputs`] against upstream's
//!    merged `inputs_embeds`: text rows (the normed embedding) and image rows
//!    (the scattered features) separately.
//! 4. **Logits.** One call over the whole prompt, every row against upstream's
//!    f32 and bf16 logits; the last row is the next token at the session
//!    boundary.
//! 5. **Chunks.** The same prompt in three calls, the first boundary inside the
//!    first image's placeholders and the second inside the second's, then two
//!    fixed text tokens one at a time; every row against the one-call replay
//!    of the whole sequence. Reset and replay, bit for bit.
//! 6. **Refusals.** Wrong row counts, rows for a text chunk, non-finite values,
//!    an out-of-table ID, an empty call and a bad row request are refused with
//!    the position unchanged and the session not poisoned; the same session
//!    then continues bit for bit like an undisturbed one.
//! 7. **Text unchanged.** On text IDs, `append` and `append_with_image_rows`
//!    with no rows are bit-equal, and (with `--text-ref`) the rows against the
//!    independent whole-model text reference.
//! 8. **Negative controls.** The placeholder's own embedding (plain `append`),
//!    zeroed rows, and the rows in reverse order must each move the decoder
//!    input and the logits away from upstream by much more than the real run.
//!
//! Logit rows are judged as the text session check judges them (relative L2
//! to upstream f32 within 1.5x upstream's own bf16-vs-f32 distance, argmax,
//! top-5) and every row is printed. The decoder input's text rows are an f32
//! norm of the same bf16 table on both sides.
//!
//! First run (2026-11-23, sky): reference sha256 022d264d..., 49 IDs with 10
//! placeholders, 0 failures; the receipt is the lane's
//! `out/run-001016/receipt.json` (mini-session-images-20261123).
//!
//! usage: inkling2_session_image_check <pile> <collection> <root> <vision_fixture_dir> <image_reference.safetensors>
//!        [--text-ref <whole_reference.safetensors>] [--receipt <out.json>]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use mary::model_collection::parse_collection_handle;
use mary::models::inkling2_loop::session::{Logits, MiniModel, MiniSession, Rows};
use mary::models::inkling2_loop::vision_input::{Geometry, ImageTokens, Pixels, Raster, preprocess};
use mary::models::inkling2_loop::Inkling2LoopConfig;
use serde_json::json;
use triblespace::prelude::Id;

const TOL: f64 = 1.5;
/// The fixed text the chunked run decodes after the prompt.
const AFTER: [u32; 2] = [371, 157021];
/// The independent text reference's 27 IDs (mini-independent-reference-20261122).
const TEXT_IDS: [u32; 27] = [
    371, 157021, 161319, 220, 32329, 160581, 82, 150053, 149876, 161703, 153350, 13, 149954, 152743, 11,
    149876, 150131, 149890, 153939, 149876, 150908, 149879, 149878, 150299, 149884, 151015, 151083,
];

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (f64::from(*x) - f64::from(*y)).powi(2);
        n += f64::from(*y).powi(2);
    }
    (d / n.max(f64::MIN_POSITIVE)).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs()).fold(0.0, f64::max)
}

fn top(v: &[f32], k: usize) -> Vec<usize> {
    let mut i: Vec<usize> = (0..v.len()).collect();
    i.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    i.truncate(k);
    i
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn gpu_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let me = std::process::id().to_string();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let mut f = l.split(',').map(str::trim);
            (f.next()? == me).then(|| f.next()?.parse().ok())?
        })
        .next()
}

/// A safetensors file's f32 (or i32, widened) tensors.
struct Reference {
    tensors: std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    ids: Vec<u32>,
    grids: Vec<[usize; 3]>,
}

impl Reference {
    fn open(path: &str) -> Reference {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("safetensors");
        let mut tensors = std::collections::BTreeMap::new();
        let (mut ids, mut grids) = (Vec::new(), Vec::new());
        for (name, t) in st.tensors() {
            let d = t.data();
            match t.dtype() {
                safetensors::Dtype::F32 => {
                    let v = d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    tensors.insert(name.to_string(), (t.shape().to_vec(), v));
                }
                safetensors::Dtype::I32 => {
                    let v: Vec<i32> = d.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                    match name.as_str() {
                        "ids" => ids = v.iter().map(|&x| x as u32).collect(),
                        "image_grid_thw" => grids = v.chunks_exact(3).map(|g| [g[0] as usize, g[1] as usize, g[2] as usize]).collect(),
                        _ => {}
                    }
                }
                other => panic!("{name}: {other:?}"),
            }
        }
        Reference { tensors, ids, grids }
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.tensors.get(name).unwrap_or_else(|| panic!("reference has no {name}")).1
    }
}

struct Report {
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, what: &str, ok: bool, detail: String) -> bool {
        println!("{} {what}: {detail}", if ok { "ok  " } else { "FAIL" });
        if !ok {
            self.failures.push(format!("{what}: {detail}"));
        }
        ok
    }

    fn row(&self, what: &str, got: &[f32], want: &[f32]) -> serde_json::Value {
        let (r, m) = (rel_l2(got, want), max_abs(got, want));
        let (tg, tw) = (top(got, 5), top(want, 5));
        let overlap = tg.iter().collect::<BTreeSet<_>>().intersection(&tw.iter().collect()).count();
        println!(
            "     {what}: rel_l2 {r:.3e} max_abs {m:.3e} argmax {} vs {} top5 {overlap}/5 bit-equal {}",
            tg[0],
            tw[0],
            bits_equal(got, want)
        );
        json!({"what": what, "rel_l2": r, "max_abs": m, "argmax": tg[0], "want_argmax": tw[0],
               "same_argmax": tg[0] == tw[0], "top5_overlap": overlap, "bit_equal": bits_equal(got, want)})
    }
}

/// Rows `[a, b)` of a row-major `[_, w]` buffer.
fn rows(v: &[f32], w: usize, a: usize, b: usize) -> &[f32] {
    &v[a * w..b * w]
}

/// Image rows and text rows of a `[t, h]` buffer, each gathered in order.
fn split_rows(v: &[f32], h: usize, mask: &[bool]) -> (Vec<f32>, Vec<f32>) {
    let (mut img, mut txt) = (Vec::new(), Vec::new());
    for (i, &m) in mask.iter().enumerate() {
        (if m { &mut img } else { &mut txt }).extend_from_slice(rows(v, h, i, i + 1));
    }
    (img, txt)
}

fn all_rows(l: &Logits) -> Vec<Vec<f32>> {
    (0..l.rows).map(|i| l.row(i).to_vec()).collect()
}

/// `chunks` (IDs and their image rows) through `s`, every row returned; then
/// each of `after` alone.
fn chunked(s: &mut MiniSession, chunks: &[(&[u32], &[f32])], after: &[u32]) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for (ids, img) in chunks {
        let l = s.append_with_image_rows(ids, img, Rows::Tail(ids.len())).expect("chunk");
        out.extend(all_rows(&l));
    }
    for &id in after {
        out.push(s.decode(id).expect("decode").last().to_vec());
    }
    out
}

fn decode_png(path: &str, geometry: Geometry) -> ImageTokens {
    let png = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let img = image::load_from_memory_with_format(&png, image::ImageFormat::Png).expect("decode");
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (bytes, alpha) = if img.color().has_alpha() {
        (img.into_rgba8().into_raw(), true)
    } else {
        (img.into_rgb8().into_raw(), false)
    };
    let pixels = if alpha { Pixels::Rgba8(&bytes) } else { Pixels::Rgb8(&bytes) };
    preprocess(geometry, Raster { width: w, height: h, pixels }).expect("preprocess")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 6,
        "usage: inkling2_session_image_check <pile> <collection> <root> <vision_fixture_dir> <image_reference.safetensors> [--text-ref <f>] [--receipt <f>]"
    );
    let flag = |f: &str| args.iter().position(|a| a == f).map(|i| args[i + 1].clone());
    let (pile, coll, root, fix, refp) = (&args[1], &args[2], &args[3], &args[4], &args[5]);
    let collection = parse_collection_handle(coll).expect("collection");
    let root_id = Id::from_hex(root).expect("root must be a 32-hex id");
    let mut rep = Report { failures: Vec::new() };
    let mut receipt = json!({"reference": refp});
    let r = Reference::open(refp);

    // 1. binding
    let t = Instant::now();
    let model = Arc::new(MiniModel::open(pile.as_ref(), collection, root_id).expect("open"));
    let open_s = t.elapsed().as_secs_f64();
    let gpu_text = gpu_mib();
    println!("opened {:X} in {open_s:.1} s, device {gpu_text:?} MiB", model.model_root());
    let h = model.config().hidden_size;
    let image_id = model.image_token_id();
    rep.check("image token id", image_id == 200_054, format!("{image_id}"));
    rep.check("vision unbound after open", !model.vision_bound(), String::new());
    let none = model.project_images(&[]).expect("empty project");
    rep.check("empty project_images binds nothing", none.is_empty() && !model.vision_bound(), String::new());

    // 2. features
    let cfg = Inkling2LoopConfig::from_json(&model.root().config().expect("config").to_string()).expect("config");
    let geometry = Geometry::of(&cfg.vision_config).expect("geometry");
    let images: Vec<ImageTokens> = ["inkling2_mini_rgb_100x60.png", "inkling2_mini_rgba_52x44.png"]
        .iter()
        .map(|f| decode_png(&format!("{fix}/{f}"), geometry))
        .collect();
    let grids: Vec<[usize; 3]> = images.iter().map(|im| [1, im.grid.0, im.grid.1]).collect();
    rep.check("image grids = upstream's", grids == r.grids, format!("{grids:?} vs {:?}", r.grids));
    let pv: Vec<f32> = images.iter().flat_map(|im| im.pixel_values.iter().copied()).collect();
    let pv_max = max_abs(&pv, r.get("pixel_values"));
    rep.check("pixel values = upstream processor's", pv.len() == r.get("pixel_values").len() && pv_max <= 1e-6, format!("max_abs {pv_max:.3e}"));
    let t = Instant::now();
    let feats = model.project_images(&images).expect("project");
    let project_cold_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let again = model.project_images(&images).expect("project again");
    let project_warm_s = t.elapsed().as_secs_f64();
    let gpu_vision = gpu_mib();
    rep.check("vision bound after projecting", model.vision_bound(), format!("device {gpu_text:?} -> {gpu_vision:?} MiB"));
    let image_rows: Vec<f32> = feats.iter().flat_map(|f| f.values.iter().copied()).collect();
    let again_rows: Vec<f32> = again.iter().flat_map(|f| f.values.iter().copied()).collect();
    rep.check("projection repeats bit for bit", bits_equal(&image_rows, &again_rows), String::new());
    let (e32, e16) = (r.get("f32.image_embeds"), r.get("bf16.image_embeds"));
    let feat_rel = rel_l2(&image_rows, e32);
    let feat_spread = rel_l2(e16, e32);
    rep.check(
        "features within 1.5x upstream's bf16-vs-f32",
        image_rows.len() == e32.len() && feat_rel <= TOL * feat_spread,
        format!("rel_l2 {feat_rel:.3e}, spread {feat_spread:.3e}, ratio {:.2}", feat_rel / feat_spread),
    );
    println!("projected {} images ({} rows) cold {project_cold_s:.2} s, warm {project_warm_s:.2} s", feats.len(), image_rows.len() / h);

    // 3. decoder input
    let ids = r.ids.clone();
    let n = ids.len();
    let mask: Vec<bool> = ids.iter().map(|&i| i == image_id).collect();
    let k = mask.iter().filter(|&&m| m).count();
    rep.check("placeholders = projected rows", k * h == image_rows.len(), format!("{k} placeholders, {} rows, {n} ids", image_rows.len() / h));
    let x = model.decoder_inputs(&ids, &image_rows).expect("decoder inputs");
    let (u32m, u16m) = (r.get("f32.inputs_embeds"), r.get("bf16.inputs_embeds"));
    let (xi, xt) = split_rows(&x, h, &mask);
    let (ui, ut) = split_rows(u32m, h, &mask);
    let (bi, bt) = split_rows(u16m, h, &mask);
    let (text_rel, text_max) = (rel_l2(&xt, &ut), max_abs(&xt, &ut));
    let (img_rel, img_spread) = (rel_l2(&xi, &ui), rel_l2(&bi, &ui));
    rep.check("decoder input text rows = upstream f32 normed embedding", text_rel <= 1e-6, format!("rel_l2 {text_rel:.3e} max_abs {text_max:.3e} (bf16 run {:.3e})", rel_l2(&bt, &ut)));
    rep.check(
        "decoder input image rows within 1.5x upstream's spread",
        img_rel <= TOL * img_spread,
        format!("rel_l2 {img_rel:.3e} spread {img_spread:.3e} ratio {:.2}", img_rel / img_spread),
    );
    rep.check("image rows land where upstream scatters them", bits_equal(&xi, &image_rows), String::new());
    receipt["decoder_input"] = json!({"text_rel_l2": text_rel, "text_max_abs": text_max, "image_rel_l2": img_rel, "image_spread": img_spread});

    // 4. logits, one call
    let full: Vec<u32> = ids.iter().copied().chain(AFTER).collect();
    let t = Instant::now();
    let mut a = MiniSession::new(Arc::clone(&model));
    let replay = all_rows(&a.append_with_image_rows(&full, &image_rows, Rows::Tail(full.len())).expect("replay"));
    let replay_s = t.elapsed().as_secs_f64();
    rep.check("position after the replay", a.position() == full.len(), format!("{}", a.position()));
    let (l32, l16) = (r.get("f32.logits"), r.get("bf16.logits"));
    let v = model.vocab();
    let (mut within, mut am32, mut am16, mut t4) = (0, 0, 0, 0);
    let mut upstream_rows = Vec::new();
    for p in 0..n {
        let (w32, w16) = (rows(l32, v, p, p + 1), rows(l16, v, p, p + 1));
        let mut row = rep.row(&format!("row {p}{} vs upstream f32", if mask[p] { " (image)" } else { "" }), &replay[p], w32);
        let spread = rel_l2(w16, w32);
        let ratio = row["rel_l2"].as_f64().unwrap() / spread;
        within += (ratio <= TOL) as usize;
        am32 += row["same_argmax"].as_bool().unwrap() as usize;
        am16 += (top(&replay[p], 1)[0] == top(w16, 1)[0]) as usize;
        t4 += (row["top5_overlap"].as_u64().unwrap() >= 4) as usize;
        println!("       spread {spread:.3e} ratio {ratio:.2} {}", if ratio <= TOL { "ok" } else { "MISS" });
        row["spread"] = json!(spread);
        row["ratio"] = json!(ratio);
        upstream_rows.push(row);
    }
    let last = n - 1;
    let last_rel = rel_l2(&replay[last], rows(l32, v, last, last + 1));
    let next = top(&replay[last], 1)[0];
    let (next32, next16) = (top(rows(l32, v, last, last + 1), 1)[0], top(rows(l16, v, last, last + 1), 1)[0]);
    rep.check("next token at the boundary = upstream's", next == next32 || next == next16, format!("Mary {next}, upstream f32 {next32}, bf16 {next16}, rel_l2 {last_rel:.3e}"));
    rep.check(
        "rows within 1.5x upstream's spread (reported; the text path holds 19/27)",
        true,
        format!("{within}/{n}; argmax = f32 {am32}/{n}, = bf16 {am16}/{n}; top5 >= 4 on {t4}/{n}"),
    );
    receipt["upstream"] = json!({"rows": upstream_rows, "within": within, "argmax_f32": am32, "argmax_bf16": am16,
                                 "top5_ge4": t4, "next": next, "next_f32": next32, "next_bf16": next16, "last_rel_l2": last_rel});

    // 5. chunks
    let first = mask.iter().position(|&m| m).expect("an image");
    let second = (first..n).find(|&i| !mask[i]).and_then(|e| (e..n).find(|&i| mask[i])).expect("two images");
    let (c1, c2) = (first + 3, second + 2);
    let k1 = mask[..c1].iter().filter(|&&m| m).count();
    let k2 = mask[..c2].iter().filter(|&&m| m).count();
    let plan: Vec<(&[u32], &[f32])> = vec![
        (&ids[..c1], &image_rows[..k1 * h]),
        (&ids[c1..c2], &image_rows[k1 * h..k2 * h]),
        (&ids[c2..], &image_rows[k2 * h..]),
    ];
    println!("chunks: [0, {c1}) {k1} rows | [{c1}, {c2}) {} rows | [{c2}, {n}) {} rows | then {AFTER:?} one at a time", k2 - k1, k - k2);
    let t = Instant::now();
    let mut b = MiniSession::new(Arc::clone(&model));
    let inc = chunked(&mut b, &plan, &AFTER);
    let chunked_s = t.elapsed().as_secs_f64();
    let (mut worst, mut same) = (0f64, 0);
    let mut cached = Vec::new();
    for (p, row) in inc.iter().enumerate() {
        let c = rep.row(&format!("chunked row {p} vs replay"), row, &replay[p]);
        worst = worst.max(c["rel_l2"].as_f64().unwrap());
        same += c["same_argmax"].as_bool().unwrap() as usize;
        cached.push(c);
    }
    // the scatter at the chunk boundaries: each chunk's decoder input, concatenated, is the one-call input
    let mut joined = Vec::new();
    for (i, img) in &plan {
        joined.extend(model.decoder_inputs(i, img).expect("chunk decoder inputs"));
    }
    rep.check("chunked decoder inputs = one-call decoder input, bit for bit", bits_equal(&joined, &x), String::new());
    // the decoder's own call-shape spread: the same IDs as text (plain append), the same calls
    let mut ta = MiniSession::new(Arc::clone(&model));
    let treplay = all_rows(&ta.append(&full, Rows::Tail(full.len())).expect("text replay"));
    let mut tb = MiniSession::new(Arc::clone(&model));
    let mut tinc = Vec::new();
    for (i, _) in &plan {
        tinc.extend(all_rows(&tb.append(i, Rows::Tail(i.len())).expect("text chunk")));
    }
    for &id in &AFTER {
        tinc.push(tb.decode(id).expect("text decode").last().to_vec());
    }
    let (mut tworst, mut tsame) = (0f64, 0);
    let mut tmean = 0f64;
    for (p, row) in tinc.iter().enumerate() {
        let rl = rel_l2(row, &treplay[p]);
        tworst = tworst.max(rl);
        tmean += rl;
        tsame += (top(row, 1)[0] == top(&treplay[p], 1)[0]) as usize;
    }
    tmean /= tinc.len() as f64;
    let mean = cached.iter().map(|c| c["rel_l2"].as_f64().unwrap()).sum::<f64>() / cached.len() as f64;
    rep.check(
        "chunked against replay: no wider than the decoder's own call-shape spread on text",
        mean <= 2.0 * tmean && worst <= 2.0 * tworst,
        format!(
            "images: argmax {same}/{n1}, mean rel_l2 {mean:.3e}, worst {worst:.3e}; text, same calls: argmax {tsame}/{n1}, mean {tmean:.3e}, worst {tworst:.3e}",
            n1 = inc.len()
        ),
    );
    let blast = full.len() - 1;
    rep.check(
        "chunked next token after the prompt and after the decodes = replay's",
        top(&inc[n - 1], 1)[0] == top(&replay[n - 1], 1)[0] && top(&inc[blast], 1)[0] == top(&replay[blast], 1)[0],
        format!("row {} {} vs {}, row {blast} {} vs {}", n - 1, top(&inc[n - 1], 1)[0], top(&replay[n - 1], 1)[0], top(&inc[blast], 1)[0], top(&replay[blast], 1)[0]),
    );
    receipt["text_chunk_baseline"] = json!({"argmax_same": tsame, "mean_rel_l2": tmean, "worst_rel_l2": tworst});
    rep.check("position after the chunked run", b.position() == full.len(), format!("{}", b.position()));
    b.reset();
    rep.check("reset", b.position() == 0 && !b.is_poisoned(), String::new());
    let again = chunked(&mut b, &plan, &AFTER);
    rep.check("reset and replay bit for bit", again.iter().zip(&inc).all(|(x, y)| bits_equal(x, y)), String::new());
    receipt["chunked"] = json!({"rows": cached, "worst_rel_l2": worst, "mean_rel_l2": mean, "argmax_same": same, "split": [c1, c2]});

    // 6. refusals, mid-sequence: after the first chunk, before the second
    let mut d = MiniSession::new(Arc::clone(&model));
    d.append_with_image_rows(plan[0].0, plan[0].1, Rows::Last).expect("first chunk");
    let (ids2, img2) = plan[1];
    let mut nan = img2.to_vec();
    nan[h + 7] = f32::NAN;
    let mut inf = img2.to_vec();
    inf[3] = f32::INFINITY;
    let mut oob = ids2.to_vec();
    oob[0] = model.embedding_rows() as u32;
    let text_only: &[u32] = &AFTER;
    let cases: Vec<(&str, &[u32], &[f32], Rows)> = vec![
        ("one row short", ids2, &img2[..img2.len() - h], Rows::Last),
        ("one row extra", ids2, &image_rows[k1 * h..(k2 + 1) * h], Rows::Last),
        ("a partial row", ids2, &img2[..img2.len() - 1], Rows::Last),
        ("rows for a text chunk", text_only, &image_rows[..h], Rows::Last),
        ("a NaN", ids2, &nan, Rows::Last),
        ("an infinity", ids2, &inf, Rows::Last),
        ("an out-of-table id", &oob, img2, Rows::Last),
        ("an empty call", &[], &[], Rows::Last),
        ("Rows::Tail(0)", ids2, img2, Rows::Tail(0)),
        ("Rows::Tail(len + 1)", ids2, img2, Rows::Tail(ids2.len() + 1)),
    ];
    let before = d.position();
    for (what, i, img, rr) in cases {
        let e = d.append_with_image_rows(i, img, rr);
        let why = e.as_ref().err().map(|e| e.to_string()).unwrap_or_default();
        rep.check(&format!("refuses {what}"), e.is_err() && d.position() == before && !d.is_poisoned(), why);
    }
    let rest = chunked(&mut d, &plan[1..], &AFTER);
    rep.check("after refusals the session continues bit for bit", rest.iter().zip(&inc[c1..]).all(|(x, y)| bits_equal(x, y)) && rest.len() == inc.len() - c1, String::new());

    // 7. text unchanged
    let mut e = MiniSession::new(Arc::clone(&model));
    let plain = e.append(&TEXT_IDS, Rows::Tail(TEXT_IDS.len())).expect("text");
    let mut f = MiniSession::new(Arc::clone(&model));
    let none = f.append_with_image_rows(&TEXT_IDS, &[], Rows::Tail(TEXT_IDS.len())).expect("text, no rows");
    rep.check("text: append = append_with_image_rows(no rows), bit for bit", bits_equal(&plain.values, &none.values), String::new());
    let xt0 = model.decoder_inputs(&TEXT_IDS, &[]).expect("text decoder inputs");
    rep.check("text decoder input has no image rows", xt0.len() == TEXT_IDS.len() * h, String::new());
    if let Some(p) = flag("--text-ref") {
        let tr = Reference::open(&p);
        rep.check("text reference ids", tr.ids == TEXT_IDS, String::new());
        let (t32, t16) = (tr.get("f32.logits"), tr.get("bf16.logits"));
        let mut within = 0;
        let mut per = Vec::new();
        for q in 0..TEXT_IDS.len() {
            let (w32, w16) = (rows(t32, v, q, q + 1), rows(t16, v, q, q + 1));
            let rl = rel_l2(plain.row(q), w32);
            let ratio = rl / rel_l2(w16, w32);
            within += (ratio <= TOL) as usize;
            per.push(json!({"row": q, "rel_l2": rl, "ratio": ratio, "argmax": top(plain.row(q), 1)[0], "want": top(w32, 1)[0]}));
        }
        println!("text rows vs the independent reference: {within}/27 within 1.5x");
        receipt["text_reference"] = json!({"path": p, "within": within, "rows": per});
    }

    // 8. negative controls
    let mut controls = Vec::new();
    let real_last = last_rel;
    let real_mean: f64 = (0..n).map(|p| rel_l2(&replay[p], rows(l32, v, p, p + 1))).sum::<f64>() / n as f64;
    let zeros = vec![0f32; image_rows.len()];
    let reversed: Vec<f32> = image_rows.chunks_exact(h).rev().flatten().copied().collect();
    let variants: Vec<(&str, Option<&[f32]>)> =
        vec![("placeholder embedding (plain append)", None), ("zeroed rows", Some(&zeros)), ("rows reversed", Some(&reversed))];
    for (what, img) in variants {
        let mut s = MiniSession::new(Arc::clone(&model));
        let l = match img {
            None => s.append(&ids, Rows::Tail(n)).expect("control"),
            Some(img) => s.append_with_image_rows(&ids, img, Rows::Tail(n)).expect("control"),
        };
        // plain append feeds the placeholder's own normed embedding: upstream's
        // pre-scatter `embeds` at those rows
        let ci = match img {
            None => split_rows(r.get("f32.embeds"), h, &mask).0,
            Some(img) => split_rows(&model.decoder_inputs(&ids, img).expect("x"), h, &mask).0,
        };
        let in_rel = rel_l2(&ci, &ui);
        let lr = rel_l2(l.row(last), rows(l32, v, last, last + 1));
        let mean: f64 = (0..n).map(|p| rel_l2(l.row(p), rows(l32, v, p, p + 1))).sum::<f64>() / n as f64;
        let cnext = top(l.row(last), 1)[0];
        rep.check(
            &format!("control '{what}' is discriminated"),
            in_rel > 10.0 * img_rel && mean > 2.0 * real_mean,
            format!(
                "image input rel_l2 {in_rel:.3e} (real {img_rel:.3e}); logits mean rel_l2 {mean:.3e} (real {real_mean:.3e}), last {lr:.3e} (real {real_last:.3e}), next {cnext}"
            ),
        );
        controls.push(json!({"what": what, "image_input_rel_l2": in_rel, "mean_rel_l2": mean, "last_rel_l2": lr, "next": cnext}));
    }
    receipt["controls"] = json!({"real_mean_rel_l2": real_mean, "variants": controls});

    receipt["timing_s"] = json!({"open": open_s, "project_cold": project_cold_s, "project_warm": project_warm_s,
                                 "replay_51": replay_s, "chunked_51": chunked_s});
    receipt["device_mib"] = json!({"text_only": gpu_text, "with_vision": gpu_vision});
    receipt["failures"] = json!(rep.failures);
    println!("timing open {open_s:.1} s, project cold {project_cold_s:.2} warm {project_warm_s:.2}, replay {replay_s:.2}, chunked {chunked_s:.2}");
    println!("{} failure(s)", rep.failures.len());
    if let Some(p) = flag("--receipt") {
        std::fs::write(&p, serde_json::to_vec_pretty(&receipt).unwrap()).expect("receipt");
    }
    std::process::exit(if rep.failures.is_empty() { 0 } else { 1 });
}
