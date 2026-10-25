//! Compare the raw codec with an independently captured b11853 vocab-only oracle.
//! Usage: flint_tokenizer_gate MODEL.gguf ORACLE.json. Metadata reads only.
use std::path::Path;
use anyhow::{Context, Result, ensure};
use mary::models::flint::tokenizer::{FlintTokenizer, EncodeOptions, REFERENCE_COMMIT, B11853_EOG_IDS, DECLARED_EOG_IDS};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    reference_commit: String, model_metadata_sha256: String, model_metadata_bytes: u64,
    source_observation: SourceObservation, cases: Vec<Case>,
}
#[derive(Deserialize)]
struct SourceObservation { bytes: u64, mtime_ns: u128, dev: u64, ino: u64 }
#[derive(Deserialize)]
struct Case {
    name: String, text: Option<String>, add_special: Option<bool>, parse_special: Option<bool>,
    result: NativeResult,
}
#[derive(Deserialize)]
struct NativeResult {
    ids: Vec<u32>, visible_hex: String, all_hex: String,
    pieces_visible_hex: Vec<String>, pieces_all_hex: Vec<String>, native_eog: Vec<bool>,
}

fn hex(bytes: &[u8]) -> String { bytes.iter().map(|byte| format!("{byte:02x}")).collect() }

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(args.len() == 2, "usage: flint_tokenizer_gate MODEL.gguf ORACLE.json");
    let fixture_path = Path::new(&args[1]);
    ensure!(fixture_path.metadata()?.len() <= 16 * 1024 * 1024, "oracle fixture too large");
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(fixture_path)?)?;
    ensure!(fixture.reference_commit == REFERENCE_COMMIT, "wrong native reference commit");
    ensure!(fixture.model_metadata_sha256 == "e5e5e957b8e7f1a02e3155bcb3899ec50ef534c477a3f8c6e1c18163b2653b88", "wrong oracle metadata identity");
    ensure!(fixture.model_metadata_bytes == 5436996, "wrong oracle metadata boundary");
    ensure!(!fixture.cases.is_empty() && fixture.cases.len() <= 256, "invalid oracle case count");
    let (mut codec, identity, metadata_end) = FlintTokenizer::open(Path::new(&args[0]))?;
    ensure!(identity.bytes == fixture.source_observation.bytes && identity.modified_ns == fixture.source_observation.mtime_ns,
        "candidate file differs from the oracle's hash-checked source observation");
    #[cfg(unix)]
    ensure!(identity.dev == fixture.source_observation.dev && identity.ino == fixture.source_observation.ino,
        "candidate file identity differs from oracle source");
    ensure!(metadata_end == fixture.model_metadata_bytes, "candidate metadata boundary differs");
    let mut failures = Vec::new();
    let mut encoded = 0;
    let mut decoded = 0;
    for case in &fixture.cases {
        let check = || -> Result<()> {
            let expected = &case.result;
            ensure!(expected.ids.len() <= 65536, "oversized oracle sequence");
            ensure!(expected.pieces_visible_hex.len() == expected.ids.len()
                && expected.pieces_all_hex.len() == expected.ids.len()
                && expected.native_eog.len() == expected.ids.len(), "malformed oracle piece count");
            for (render, whole, pieces) in [(false, &expected.visible_hex, &expected.pieces_visible_hex), (true, &expected.all_hex, &expected.pieces_all_hex)] {
                ensure!(hex(&codec.decode_bytes(&expected.ids, render)?) == *whole, "whole decode differs, render_controls={render}");
                let mut streamed = Vec::new();
                for (&id, piece) in expected.ids.iter().zip(pieces) {
                    let bytes = codec.token_bytes(id, render)?;
                    ensure!(hex(bytes) == *piece, "piece {id} differs, render_controls={render}");
                    streamed.extend_from_slice(bytes);
                }
                ensure!(hex(&streamed) == *whole, "streamed bytes differ");
            }
            for (&id, &native_eog) in expected.ids.iter().zip(&expected.native_eog) {
                ensure!(native_eog == B11853_EOG_IDS.contains(&id), "native EOG behavior changed at {id}");
                ensure!(codec.is_eog(id) == DECLARED_EOG_IDS.contains(&id), "declared EOG differs at {id}");
            }
            Ok(())
        };
        if let Err(error) = check() { failures.push(format!("{}: {error:#}", case.name)); }
        decoded += 1;
        if let Some(text) = &case.text {
            ensure!(text.len() <= 65536, "oversized oracle text");
            let options = EncodeOptions {
                add_special: case.add_special.context("missing add_special")?,
                parse_special: case.parse_special.context("missing parse_special")?,
            };
            match codec.encode(text, options) {
                Ok(ids) if ids == case.result.ids => {},
                Ok(ids) => failures.push(format!("{}: encode differs: actual {ids:?}; native {:?}", case.name, case.result.ids)),
                Err(error) => failures.push(format!("{}: encode: {error:#}", case.name)),
            }
            encoded += 1;
        }
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "status": if failures.is_empty() { "pass" } else { "fail" },
        "reference_commit": REFERENCE_COMMIT, "candidate_file_observation": identity,
        "candidate_metadata_end": metadata_end, "vocab_size": codec.vocab_size(),
        "encoded_cases": encoded, "decoded_cases": decoded, "declared_eog_ids": DECLARED_EOG_IDS,
        "native_b11853_eog_ids": B11853_EOG_IDS, "failures": failures,
        "limits": "raw codec only; no template rendering, tensor reads, model execution, or GPU parity"
    }))?);
    ensure!(failures.is_empty(), "{} tokenizer oracle checks failed", failures.len());
    Ok(())
}
