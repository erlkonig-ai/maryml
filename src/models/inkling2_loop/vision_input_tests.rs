//! Held to the published processor on saved images.
//!
//! `fixtures/vision/`: two PNGs with the published processor's and vision
//! tower's outputs on them (`inkling2_mini_vision_reference.json`, written by
//! `make_fixture.py`), and five more PNGs through the processor alone
//! (`extra_cases.json`, `extra_cases.py`): exact multiples of the tile, and
//! the L, LA and palette-with-tRNS conversions. Both ran on the CPU in
//! vllm/vllm-openai v0.26.0 (torch 2.11.0, PIL 12.3.0) with transformers
//! 046bdb8 and the release's `processor_config.json` (b85c3cfa, copied here).
//! The PNGs are decoded here with the `image` crate, the way Drive decodes,
//! keeping alpha when the decoded image has it.

use super::*;
use crate::models::inkling2_loop::Inkling2LoopConfig;
use serde_json::Value;
use sha2::{Digest, Sha256};

const CONFIG: &str = include_str!("fixtures/config.json");
const PROCESSOR: &str = include_str!("fixtures/vision/processor_config.json");
const REFERENCE: &str = include_str!("fixtures/vision/inkling2_mini_vision_reference.json");
const EXTRA: &str = include_str!("fixtures/vision/extra_cases.json");

fn png(name: &str) -> &'static [u8] {
    match name {
        "inkling2_mini_rgb_100x60.png" => {
            include_bytes!("fixtures/vision/inkling2_mini_rgb_100x60.png")
        }
        "inkling2_mini_rgba_52x44.png" => {
            include_bytes!("fixtures/vision/inkling2_mini_rgba_52x44.png")
        }
        "rgb_80x40.png" => include_bytes!("fixtures/vision/rgb_80x40.png"),
        "rgba_40x80.png" => include_bytes!("fixtures/vision/rgba_40x80.png"),
        "la_33x47.png" => include_bytes!("fixtures/vision/la_33x47.png"),
        "p_trns_30x20.png" => include_bytes!("fixtures/vision/p_trns_30x20.png"),
        "l_45x41.png" => include_bytes!("fixtures/vision/l_45x41.png"),
        other => panic!("no fixture {other}"),
    }
}

fn geometry() -> Geometry {
    Geometry::of(&Inkling2LoopConfig::from_json(CONFIG).unwrap().vision_config).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn f32_sha(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    hex(&bytes)
}

/// Decoded bytes and whether they carry alpha.
fn decode(name: &str) -> (usize, usize, Vec<u8>, bool) {
    let img = image::load_from_memory_with_format(png(name), image::ImageFormat::Png).unwrap();
    let color = img.color();
    assert_eq!(
        color.bits_per_pixel() / u16::from(color.channel_count()),
        8,
        "{name}: 8-bit only"
    );
    let (w, h) = (img.width() as usize, img.height() as usize);
    if color.has_alpha() {
        (w, h, img.into_rgba8().into_raw(), true)
    } else {
        (w, h, img.into_rgb8().into_raw(), false)
    }
}

fn run(name: &str) -> (Vec<u8>, ImageTokens) {
    let (width, height, bytes, alpha) = decode(name);
    let pixels = if alpha {
        Pixels::Rgba8(&bytes)
    } else {
        Pixels::Rgb8(&bytes)
    };
    let tokens = preprocess(
        geometry(),
        Raster {
            width,
            height,
            pixels,
        },
    )
    .unwrap();
    (bytes, tokens)
}

fn max_abs(got: &[f32], want: &[(usize, f32)]) -> f32 {
    want.iter()
        .map(|&(i, w)| (got[i] - w).abs())
        .fold(0.0, f32::max)
}

#[test]
fn the_constants_are_the_release_processor_s() {
    let p: Value = serde_json::from_str(PROCESSOR).unwrap();
    let ip = &p["image_processor"];
    assert_eq!(ip["image_processor_type"], "Inkling2ImageProcessor");
    for k in ["do_convert_rgb", "do_rescale", "do_normalize"] {
        assert_eq!(ip[k], true, "{k}");
    }
    assert_eq!(ip["image_mean"], serde_json::json!([0.5, 0.5, 0.5]));
    assert_eq!(ip["image_std"], serde_json::json!([0.5, 0.5, 0.5]));
    let tile = geometry().tile();
    assert_eq!(ip["size"]["height"].as_u64(), Some(tile as u64));
    assert_eq!(ip["size"]["width"].as_u64(), Some(tile as u64));
    assert_eq!(
        geometry(),
        Geometry {
            native_patch: 20,
            merge: 2
        }
    );
    // The fused mean and std: 0.5 over the rescale factor, in f32 as the
    // torchvision backend builds them, is exactly the centre used here.
    let rescale = ip["rescale_factor"].as_f64().unwrap();
    assert_eq!((0.5f32 * (1.0 / rescale) as f32), CENTRE);
    assert_eq!(0.5 * (1.0 / rescale), f64::from(CENTRE));
}

#[test]
fn the_saved_fixtures_match_the_published_tensors() {
    let r: Value = serde_json::from_str(REFERENCE).unwrap();
    let g = geometry();
    let mut pixel_values = Vec::new();
    let mut native = Vec::new();
    let mut positions = Vec::new();
    let mut grids = Vec::new();
    let mut patch_grids = Vec::new();
    for im in r["images"].as_array().unwrap() {
        let (bytes, t) = run(im["file"].as_str().unwrap());
        // The codec gave the bytes PIL gave.
        assert_eq!(
            hex(&bytes),
            im["raw_sha256"].as_str().unwrap(),
            "{}",
            im["name"]
        );
        assert_eq!(t.pixel_values.len(), t.tokens() * g.token_values());
        grids.push(serde_json::json!([1, t.grid.0, t.grid.1]));
        patch_grids.push(serde_json::json!(t.patch_grid_thw()));
        pixel_values.extend_from_slice(&t.pixel_values);
        native.extend(t.native_patches());
        positions.extend(t.native_positions());
    }
    let p = &r["processor"];
    assert_eq!(Value::Array(grids), p["image_grid_thw"]);
    let want: Vec<(usize, f32)> = p["selected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let at = |k: &str| s[k].as_u64().unwrap() as usize;
            let i = ((at("token") * g.tile() + at("y")) * g.tile() + at("x")) * 3 + at("c");
            (i, s["value"].as_f64().unwrap() as f32)
        })
        .collect();
    assert_eq!(want.len(), 70);
    assert_eq!(
        f32_sha(&pixel_values),
        p["pixel_values_f32_sha256"].as_str().unwrap(),
        "pixel_values differ; max |error| over the {} published probes: {}",
        want.len(),
        max_abs(&pixel_values, &want)
    );
    for &(i, w) in &want {
        assert_eq!(pixel_values[i].to_bits(), w.to_bits(), "pixel_values[{i}]");
    }
    assert_eq!(pixel_values.len(), 10 * 40 * 40 * 3);
    assert_eq!(
        p["pad_value"].as_f64().unwrap() as f32,
        (PAD_RAW - CENTRE) / CENTRE
    );

    let v = &r["vision_input"];
    assert_eq!(Value::Array(patch_grids), v["patch_grid_thw"]);
    assert_eq!(native.len(), 40 * g.native_values());
    let want: Vec<(usize, f32)> = v["selected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let i = s["row"].as_u64().unwrap() as usize * g.native_values()
                + s["col"].as_u64().unwrap() as usize;
            (i, s["value"].as_f64().unwrap() as f32)
        })
        .collect();
    assert_eq!(
        f32_sha(&native),
        v["patch_embedding_input_f32_sha256"].as_str().unwrap(),
        "native patches differ; max |error| over the {} published probes: {}",
        want.len(),
        max_abs(&native, &want)
    );
    for &(i, w) in &want {
        assert_eq!(native[i].to_bits(), w.to_bits(), "native[{i}]");
    }
    assert_eq!(
        positions.len(),
        v["position_ids_shape"][0].as_u64().unwrap() as usize
    );
    assert_eq!(serde_json::json!(positions[..8]), v["position_ids_first8"]);
}

#[test]
fn exact_multiples_and_conversions_match_the_processor() {
    let cases: Vec<Value> = serde_json::from_str(EXTRA).unwrap();
    assert_eq!(cases.len(), 5);
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let (bytes, t) = run(c["file"].as_str().unwrap());
        // The `image` crate keeps an alpha channel exactly when PIL reports
        // transparency data (tRNS included), so the composite runs on the
        // same images.
        let channels =
            bytes.len() / (c["width"].as_u64().unwrap() * c["height"].as_u64().unwrap()) as usize;
        assert_eq!(
            channels == 4,
            c["has_transparency_data"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(
            serde_json::json!([1, t.grid.0, t.grid.1]),
            c["image_grid_thw"],
            "{name}"
        );
        assert_eq!(
            f32_sha(&t.pixel_values),
            c["pixel_values_f32_sha256"].as_str().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn the_grid_pads_only_what_the_tile_does_not_fill() {
    let g = geometry();
    assert_eq!(g.grid(40, 40), (1, 1));
    assert_eq!(g.grid(80, 40), (1, 2));
    assert_eq!(g.grid(41, 40), (1, 2));
    assert_eq!(g.grid(100, 60), (2, 3));
    assert_eq!(g.grid(1, 1), (1, 1));

    // An opaque RGBA8 raster is its RGB8 raster.
    let (w, h) = (45usize, 37usize);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 256) as u8).collect();
    let rgba: Vec<u8> = rgb
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    let a = preprocess(
        g,
        Raster {
            width: w,
            height: h,
            pixels: Pixels::Rgb8(&rgb),
        },
    )
    .unwrap();
    let b = preprocess(
        g,
        Raster {
            width: w,
            height: h,
            pixels: Pixels::Rgba8(&rgba),
        },
    )
    .unwrap();
    assert_eq!(a, b);
    // A fully transparent pixel is the checkerboard: 204 at (0, 0), 153 at (0, 8).
    let clear = vec![0u8; w * h * 4];
    let c = preprocess(
        g,
        Raster {
            width: w,
            height: h,
            pixels: Pixels::Rgba8(&clear),
        },
    )
    .unwrap();
    assert_eq!(c.pixel_values[0], (204.0 - CENTRE) / CENTRE);
    assert_eq!(c.pixel_values[8 * 3], (153.0 - CENTRE) / CENTRE);
    // x = 45.. of the second token column is pad.
    let at = |y: usize, x: usize| ((y * 40) + (x % 40)) * 3 + (x / 40) * g.token_values();
    assert_eq!(c.pixel_values[at(0, 44)], (153.0 - CENTRE) / CENTRE);
    assert_eq!(c.pixel_values[at(0, 45)], (PAD_RAW - CENTRE) / CENTRE);

    assert!(
        preprocess(
            g,
            Raster {
                width: 0,
                height: 4,
                pixels: Pixels::Rgb8(&[])
            }
        )
        .is_err()
    );
    assert!(
        preprocess(
            g,
            Raster {
                width: w,
                height: h,
                pixels: Pixels::Rgb8(&rgba)
            }
        )
        .is_err()
    );
    assert!(
        preprocess(
            g,
            Raster {
                width: w,
                height: h,
                pixels: Pixels::Rgba8(&rgb)
            }
        )
        .is_err()
    );
}

/// The learned position table's taps and the rotary positions, against the
/// published helpers' outputs on the two fixture images (indices as i64 LE,
/// weights as f32 LE, both images concatenated as the reference batches them).
#[test]
fn position_taps_and_rotary_positions_match_the_published_helpers() {
    let r: Value = serde_json::from_str(REFERENCE).unwrap();
    let v = &r["vision_input"];
    let side = Inkling2LoopConfig::from_json(CONFIG).unwrap().vision_config.posemb_grid_size;
    assert_eq!(side, 32);
    let (mut idx, mut wts, mut taps_all) = (Vec::new(), Vec::new(), Vec::new());
    for im in r["images"].as_array().unwrap() {
        let (_, t) = run(im["file"].as_str().unwrap());
        let taps = posemb_taps(&t.native_positions(), t.patch_grid(), side).unwrap();
        for tap in &taps {
            for &(i, w) in tap {
                idx.extend_from_slice(&(i as i64).to_le_bytes());
                wts.push(w);
            }
        }
        taps_all.extend(taps);
        // the class token first at (0, 0), then the patches 1-indexed
        let seq = t.sequence_positions();
        let native = t.native_positions();
        assert_eq!(seq.len(), 1 + native.len());
        assert_eq!(seq[0], [0, 0]);
        for (s, n) in seq[1..].iter().zip(&native) {
            assert_eq!(*s, [n[0] + 1, n[1] + 1]);
        }
    }
    assert_eq!(hex(&idx), v["interp_indices_sha256"].as_str().unwrap());
    assert_eq!(f32_sha(&wts), v["interp_weights_f32_sha256"].as_str().unwrap());
    for (k, want) in v["interp_indices_first8"].as_array().unwrap().iter().enumerate() {
        let got: Vec<u64> = taps_all[k].iter().map(|&(i, _)| i as u64).collect();
        assert_eq!(serde_json::json!(got), *want, "taps of row {k}");
    }
    for (k, want) in v["interp_weights_first8"].as_array().unwrap().iter().enumerate() {
        for (j, w) in want.as_array().unwrap().iter().enumerate() {
            assert_eq!(taps_all[k][j].1.to_bits(), (w.as_f64().unwrap() as f32).to_bits(), "weight {k}.{j}");
        }
    }
}

/// Upsampling past the table (more native patches than its side) clamps to
/// the border, and a position outside the stated grid is refused, not read.
#[test]
fn position_taps_clamp_at_the_border_and_refuse_outside_the_grid() {
    let taps = posemb_taps(&[[0, 0], [63, 63]], (64, 64), 32).unwrap();
    // src = 0.5 * 32 / 64 - 0.5 = -0.25: both taps on row/col 0, weights 0.25 and 0.75
    let w: f32 = taps[0].iter().map(|&(_, w)| w).sum();
    assert_eq!(w, 1.0);
    assert!(taps[0].iter().all(|&(i, _)| i == 0));
    assert!(taps[1].iter().all(|&(i, _)| i == 31 * 32 + 31));
    assert!(posemb_taps(&[[4, 0]], (4, 6), 32).is_err());
    assert!(posemb_taps(&[[0, 0]], (0, 6), 32).is_err());
    assert!(posemb_taps(&[[0, 0]], (4, 6), 0).is_err());
}
