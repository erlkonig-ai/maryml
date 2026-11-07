//! Empirical concurrent-fit check: hold PersonaPlex-7B AND Gemma-4-31B resident
//! on the same Metal GPU (unified memory) at once, run a forward on the 31B, and
//! report peak RSS. Confirms both fit 128 GB with headroom for KV caches +
//! activations during concurrent operation (the background LLM + the voice model
//! live together). If the 31B-dense does NOT leave safe headroom, this is where
//! the 26B-A4B MoE fallback gets flagged.
//!
//!   cargo run --release --features gemma,qwen3tts,f16gen --bin gemma31b_fit -- \
//!     --gemma models/gemma_31b.pile --plex models/personaplex.pile
//!
//! The 31B loads as f16 via the zero-copy aliased loader (mmap → GPU, no
//! copy). PersonaPlex is selected from its exact signed model member, then each
//! f32 leaf is converted into a resident f16 GPU tensor. GPU allocations
//! on Apple silicon are unified memory, so peak RSS is the real system
//! footprint.

#[cfg(target_os = "macos")]
mod imp {

    use burn::prelude::*;
    use mary::models::gemma::gemma4::config::Gemma4Config;
    use mary::nn::backend::BHalf;
    use std::path::Path;
    use std::process::Command;

    fn find_hf_file(model_id: &str, filename: &str) -> String {
        let o = Command::new("python3")
            .args([
                "-c",
                &format!(
                "from huggingface_hub import hf_hub_download; print(hf_hub_download('{}', '{}'))",
                model_id, filename
            ),
            ])
            .output()
            .unwrap_or_else(|e| panic!("hf_hub_download {model_id}/{filename}: {e}"));
        String::from_utf8(o.stdout).unwrap().trim().to_string()
    }

    fn arg(args: &[String], k: &str) -> Option<String> {
        args.iter()
            .position(|s| s == k)
            .map(|i| args[i + 1].clone())
    }

    /// Current process resident-set size in GiB (macOS `ps`).
    fn rss_gib() -> f64 {
        let pid = std::process::id().to_string();
        let o = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .unwrap();
        let kb: f64 = String::from_utf8(o.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap_or(0.0);
        kb / 1048576.0 // KiB -> GiB
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().collect();
        let gemma_pile = arg(&args, "--gemma").expect("--gemma <gemma_31b.pile>");
        let plex_pile = arg(&args, "--plex").expect("--plex <personaplex.pile>");

        let device = mary::models::gemma::metal_device::init_metal_device_16gb();
        eprintln!("[fit] baseline RSS: {:.2} GiB", rss_gib());

        // ── PersonaPlex-7B: convert every exact leaf onto the GPU, hold resident ───
        // Freeze and strictly select the same signed model the realtime loader
        // uses; each exact leaf is uploaded at f16 and kept in
        // `plex_resident` so it stays live.
        eprintln!("[fit] loading PersonaPlex-7B weights ({plex_pile}) onto GPU...");
        let plex_model = mary::persist::personaplex_model(Path::new(&plex_pile))
            .expect("canonical signed PersonaPlex model");
        let plex = plex_model.weights();
        let mut plex_resident: Vec<Tensor<BHalf, 1>> = Vec::new();
        let mut plex_bytes: u64 = 0;
        let mut plex_params: u64 = 0;
        for leaf in plex.exact().values() {
            let data = leaf
                .view_f32()
                .expect("PersonaPlex admission rejects non-f32 leaves");
            let n = data.len();
            plex_params += n as u64;
            plex_bytes += (n * 2) as u64; // f16 resident width
            let t = Tensor::<BHalf, 1>::from_floats(&data[..], &device);
            plex_resident.push(t);
        }
        // Force the uploads to complete.
        let _ = plex_resident
            .iter()
            .map(|t| t.clone().sum().into_scalar())
            .count();
        eprintln!(
            "[fit] PersonaPlex resident: {} tensors, {:.2} G params, {:.2} GiB f16. RSS now {:.2} GiB",
            plex_resident.len(),
            plex_params as f64 / 1e9,
            plex_bytes as f64 / (1u64 << 30) as f64,
            rss_gib()
        );

        // ── Gemma-4-31B: zero-copy aliased load onto the SAME GPU ─────────────────
        eprintln!("[fit] loading Gemma-4-31B ({gemma_pile}) zero-copy aliased onto GPU...");
        let config_path = find_hf_file("google/gemma-4-31B-it", "config.json");
        let tokenizer_path = find_hf_file("google/gemma-4-31B-it", "tokenizer.json");
        let mut config = Gemma4Config::load(Path::new(&config_path));
        config.vision_config = None;
        config.audio_config = None;
        let lm = mary::models::gemma::gemma4::lm::GemmaLM::<BHalf>::from_aliased_pile(
            config,
            Path::new(&gemma_pile),
            mary::selection::ModelSelector::Source {
                source: "google/gemma-4-31B-it",
                quantization: mary::persist::QUANTIZATION_NATIVE,
            },
            Path::new(&tokenizer_path),
            device.clone(),
        );
        eprintln!(
            "[fit] Gemma-4-31B loaded (aliased). RSS now {:.2} GiB",
            rss_gib()
        );

        // ── Forward on the 31B WHILE PersonaPlex stays resident ───────────────────
        eprintln!("[fit] running a 31B forward with PersonaPlex still resident...");
        let out = lm.generate("Name one planet.", 6);
        let peak = rss_gib();
        eprintln!("[fit] 31B says: {out:?}");
        eprintln!("[fit] plex tensors still live: {}", plex_resident.len());

        println!("\n=== CONCURRENT FIT ===");
        println!(
            "PersonaPlex f16 weights : {:.2} GiB ({:.2} G params)",
            plex_bytes as f64 / (1u64 << 30) as f64,
            plex_params as f64 / 1e9
        );
        println!(
            "Gemma-4-31B pile        : {:.2} GiB (aliased, zero-copy)",
            std::fs::metadata(&gemma_pile).map(|m| m.len()).unwrap_or(0) as f64
                / (1u64 << 30) as f64
        );
        println!("PEAK RSS (both + fwd)   : {:.2} GiB", peak);
        println!("System RAM              : 128 GiB");
        println!("Headroom                : {:.2} GiB", 128.0 - peak);
        // keep both alive to the very end
        std::hint::black_box(&plex_resident);
        std::hint::black_box(&lm);
    }
}

#[cfg(target_os = "macos")]
fn main() {
    imp::run()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemma31b_fit: macOS/Metal-only lane (zero-copy aliased 31B load).");
    std::process::exit(2);
}
