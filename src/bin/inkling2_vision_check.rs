//! Inkling 2 Mini's vision tower and projector, natively, against the
//! published tower's own run on the two saved fixture images.
//!
//! Reads the model from a pile alone (the pile, the exact model collection and
//! the model root, opened as a [`MiniRoot`]); decodes the two fixture PNGs the
//! way Drive does, keeping alpha; preprocesses them with
//! `inkling2_loop::vision_input`; runs `inkling2_loop::vision::VisionTower` on
//! the device; and compares the `[10, 2304]` projected features with the
//! fixture's `pooler_output` from transformers 046bdb8's
//! `Inkling2VisionModel` run on the CPU at f32 and at bf16
//! (`fixtures/vision/make_fixture.py`, full arrays in its `out/`).
//!
//! The tolerance is calibrated, not chosen, as in `inkling2_layer0`: Mary
//! passes when its relative L2 distance to upstream's f32 run is within `TOL`
//! times upstream's own bf16-vs-f32 distance. That makes it a bug detector for
//! the port (a wrong position, a dropped bias, a misplaced class row), not a
//! precision verdict.
//!
//! usage: inkling2_vision_check <pile> <collection> <root> <fixture_dir> <reference_out_dir> [receipt.json]

use std::time::Instant;

use sha2::{Digest, Sha256};

use mary::model_collection::parse_collection_handle;
use mary::models::inkling2_loop::Inkling2LoopConfig;
use mary::models::inkling2_loop::pile::MiniRoot;
use mary::models::inkling2_loop::vision::VisionTower;
use mary::models::inkling2_loop::vision_input::{Geometry, Pixels, Raster, preprocess};
use triblespace::prelude::Id;

/// Within this many times upstream's own bf16-vs-f32 spread.
const TOL: f64 = 1.5;

fn hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn f32_file(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut n, mut d) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        n += (f64::from(*x) - f64::from(*y)).powi(2);
        d += f64::from(*y).powi(2);
    }
    (n / d).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

fn min_row_cos(a: &[f32], b: &[f32], w: usize) -> f64 {
    a.chunks_exact(w)
        .zip(b.chunks_exact(w))
        .map(|(x, y)| {
            let (mut xy, mut xx, mut yy) = (0f64, 0f64, 0f64);
            for (p, q) in x.iter().zip(y) {
                let (p, q) = (f64::from(*p), f64::from(*q));
                xy += p * q;
                xx += p * p;
                yy += q * q;
            }
            xy / (xx.sqrt() * yy.sqrt())
        })
        .fold(1.0, f64::min)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 6,
        "usage: inkling2_vision_check <pile> <collection> <root> <fixture_dir> <reference_out_dir> [receipt.json]"
    );
    let (pile, coll, root, fix, out) = (&args[1], &args[2], &args[3], &args[4], &args[5]);
    let collection = parse_collection_handle(coll).expect("collection");
    let root_id = Id::from_hex(root).expect("root must be a 32-hex id");
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{fix}/inkling2_mini_vision_reference.json"))
            .expect("reference json"),
    )
    .expect("reference json");

    let t0 = Instant::now();
    let mini =
        MiniRoot::open(pile.as_ref(), collection, root_id).expect("open the Mini model root");
    let cfg =
        Inkling2LoopConfig::from_json(&mini.config().expect("config").to_string()).expect("config");
    let open_s = t0.elapsed().as_secs_f64();
    let dev = Default::default();
    let t1 = Instant::now();
    let tower = VisionTower::bind(&mini, &cfg.vision_config, &dev).expect("bind the vision tower");
    let bind_s = t1.elapsed().as_secs_f64();
    println!(
        "opened root {:X} in {open_s:.1} s, bound the tower in {bind_s:.1} s: {:?}",
        mini.root(),
        tower.dims()
    );

    let geometry = Geometry::of(&cfg.vision_config).expect("geometry");
    let mut images = Vec::new();
    for im in reference["images"].as_array().expect("images") {
        let file = im["file"].as_str().expect("file");
        let png = std::fs::read(format!("{fix}/{file}")).expect("png");
        assert_eq!(
            hex(&png),
            im["png_sha256"].as_str().unwrap(),
            "{file}: not the fixture's PNG"
        );
        let img =
            image::load_from_memory_with_format(&png, image::ImageFormat::Png).expect("decode");
        let (w, h) = (img.width() as usize, img.height() as usize);
        let (bytes, alpha) = if img.color().has_alpha() {
            (img.into_rgba8().into_raw(), true)
        } else {
            (img.into_rgb8().into_raw(), false)
        };
        let pixels = if alpha {
            Pixels::Rgba8(&bytes)
        } else {
            Pixels::Rgb8(&bytes)
        };
        images.push(
            preprocess(
                geometry,
                Raster {
                    width: w,
                    height: h,
                    pixels,
                },
            )
            .expect("preprocess"),
        );
    }

    // malformed geometry is refused at the call boundary, not a panic
    let mut bad = images[0].clone();
    bad.pixel_values.pop();
    let refused_short = tower.forward(&bad).unwrap_err();
    let mut bad = images[0].clone();
    bad.geometry.merge = 1;
    let refused_geometry = tower.forward(&bad).unwrap_err();
    println!("refused: {refused_short} | {refused_geometry}");

    let t2 = Instant::now();
    let feats = tower.forward_images(&images).expect("forward");
    let fwd_s = t2.elapsed().as_secs_f64();
    let t3 = Instant::now();
    let again = tower.forward_images(&images).expect("forward");
    let warm_s = t3.elapsed().as_secs_f64();
    let width = feats[0].width;
    let mary: Vec<f32> = feats
        .iter()
        .flat_map(|f| f.values.iter().copied())
        .collect();
    let repeat: Vec<f32> = again
        .iter()
        .flat_map(|f| f.values.iter().copied())
        .collect();
    let deterministic = mary
        .iter()
        .zip(&repeat)
        .all(|(a, b)| a.to_bits() == b.to_bits());

    let r32 = f32_file(&format!("{out}/pooler_output_float32.f32"));
    let r16 = f32_file(&format!("{out}/pooler_output_bfloat16.f32"));
    let vo = &reference["vision_output"];
    assert_eq!(
        hex(&std::fs::read(format!("{out}/pooler_output_float32.f32")).unwrap()),
        vo["float32"]["pooler_output_f32_sha256"].as_str().unwrap()
    );
    assert_eq!(
        hex(&std::fs::read(format!("{out}/pooler_output_bfloat16.f32")).unwrap()),
        vo["bfloat16"]["pooler_output_f32_sha256"].as_str().unwrap()
    );
    assert_eq!(
        mary.len(),
        r32.len(),
        "Mary [{}, {width}] against upstream {} values",
        mary.len() / width,
        r32.len()
    );

    let mut rows = Vec::new();
    let mut ok = true;
    let mut start = 0;
    let mut spans = Vec::new();
    for (k, f) in feats.iter().enumerate() {
        spans.push((format!("image{k}"), start, start + f.values.len()));
        start += f.values.len();
    }
    spans.push(("all".into(), 0, mary.len()));
    for (name, a, b) in &spans {
        let (m, x32, x16) = (&mary[*a..*b], &r32[*a..*b], &r16[*a..*b]);
        let e32 = rel_l2(m, x32);
        let e16 = rel_l2(m, x16);
        let spread = rel_l2(x16, x32);
        let ratio = e32 / spread;
        let pass = ratio <= TOL;
        ok &= pass;
        let line = format!(
            "{name:7} vs f32 rel {e32:.3e} max {:.3e} min-row-cos {:.6}  vs bf16 rel {e16:.3e}  upstream bf16-vs-f32 rel {spread:.3e} max {:.3e}  ratio {ratio:.2}  {}",
            max_abs(m, x32),
            min_row_cos(m, x32, width),
            max_abs(x16, x32),
            if pass { "ok" } else { "OUTSIDE" }
        );
        println!("{line}");
        rows.push(serde_json::json!({"span": name, "rel_l2_vs_f32": e32, "max_abs_vs_f32": max_abs(m, x32),
            "min_row_cos_vs_f32": min_row_cos(m, x32, width), "rel_l2_vs_bf16": e16, "upstream_spread": spread,
            "upstream_max_abs": max_abs(x16, x32), "ratio": ratio, "pass": pass}));
    }
    // the published probes and per-row norms, as a second reading of the same arrays
    let mut probe_max = 0f64;
    for s in vo["float32"]["selected"].as_array().unwrap() {
        let (r, c) = (
            s["row"].as_u64().unwrap() as usize,
            s["col"].as_u64().unwrap() as usize,
        );
        probe_max =
            probe_max.max((f64::from(mary[r * width + c]) - s["value"].as_f64().unwrap()).abs());
    }
    let norms: Vec<f64> = mary
        .chunks_exact(width)
        .map(|r| r.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt())
        .collect();
    let want_norms: Vec<f64> = vo["float32"]["row_l2"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let norm_rel = norms
        .iter()
        .zip(&want_norms)
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0, f64::max);
    println!(
        "probes max |mary - f32| {probe_max:.3e}; row L2 max rel {norm_rel:.3e}; deterministic {deterministic}; forward of both images {fwd_s:.2} s cold, {warm_s:.2} s warm"
    );
    println!(
        "tokens {:?}",
        feats.iter().map(|f| f.tokens).collect::<Vec<_>>()
    );
    if let Some(path) = args.get(6) {
        let receipt = serde_json::json!({
            "what": "inkling2_vision_check: native Mini vision tower + projector vs transformers 046bdb8 pooler_output",
            "pile": pile, "collection": coll, "root": format!("{:X}", mini.root()),
            "images": reference["images"].as_array().unwrap().iter().map(|i| i["file"].clone()).collect::<Vec<_>>(),
            "tokens": feats.iter().map(|f| f.tokens).collect::<Vec<_>>(), "width": width,
            "mary_f32_sha256": hex(&mary.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
            "reference_f32_sha256": vo["float32"]["pooler_output_f32_sha256"],
            "tol": TOL, "spans": rows, "probe_max_abs": probe_max, "row_l2_max_rel": norm_rel,
            "deterministic": deterministic, "open_s": open_s, "bind_s": bind_s, "forward_cold_s": fwd_s, "forward_warm_s": warm_s,
            "refused": [refused_short, refused_geometry], "pass": ok,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).expect("receipt");
    }
    if !ok {
        eprintln!("OUTSIDE tolerance");
        std::process::exit(1);
    }
    println!("vision check passed");
}
