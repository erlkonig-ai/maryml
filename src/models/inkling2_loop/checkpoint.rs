//! What an Inkling 2 Mini checkpoint holds, read from its safetensors
//! headers alone, and what Mary still cannot run of it.
//!
//! [`inspect_dir`] reads `config.json`, `hf_quant_config.json` and the
//! *header* of every `*.safetensors` in a directory: the 8-byte length and
//! the JSON that follows it, never a tensor byte. It then compares every
//! tensor against the layout's prediction. The result says what it examined
//! (files, tensors, slots) next to what it found, so a clean report cannot
//! come from looking at nothing.
//!
//! [`native_gaps`] is the list of operators and storage formats the release
//! needs that Mary has no native implementation of yet. Any caller that would
//! *run* the model goes through [`require_native_forward`] and gets that list
//! as an error. Nothing is skipped and nothing falls back to the original
//! Inkling's code.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};

use super::config::Inkling2LoopConfig;
use super::layout::{self, Dtype, Expect, File, Precision, Precisions, Slot};
use super::schedule::Schedule;

/// One tensor as its file's header describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderEntry {
    pub file: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Largest header this reader accepts. The released model's is 148 KB; a
/// length beyond this is a damaged or foreign file, not a bigger model.
const MAX_HEADER: u64 = 64 << 20;

/// Read one safetensors file's header without reading its tensors.
pub fn read_header(path: &Path) -> Result<BTreeMap<String, HeaderEntry>, String> {
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut f =
        std::fs::File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .map_err(|e| format!("reading the header length of {}: {e}", path.display()))?;
    let len = u64::from_le_bytes(len);
    if len == 0 || len > MAX_HEADER {
        return Err(format!(
            "{}: header length {len} is not plausible",
            path.display()
        ));
    }
    let mut json = vec![0u8; len as usize];
    f.read_exact(&mut json)
        .map_err(|e| format!("reading the {len}-byte header of {}: {e}", path.display()))?;
    parse_header(&file_name, &json)
}

/// Parse a safetensors header's JSON.
pub fn parse_header(file: &str, json: &[u8]) -> Result<BTreeMap<String, HeaderEntry>, String> {
    let v: serde_json::Value =
        serde_json::from_slice(json).map_err(|e| format!("{file}: header is not JSON: {e}"))?;
    let obj = v
        .as_object()
        .ok_or_else(|| format!("{file}: header is not an object"))?;
    let mut out = BTreeMap::new();
    for (name, t) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = t
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| format!("{file}: {name} has no dtype"))?;
        let shape = t
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| format!("{file}: {name} has no shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| format!("{file}: {name} has a non-integer dimension"))?;
        out.insert(
            name.clone(),
            HeaderEntry {
                file: file.to_string(),
                dtype: dtype.to_string(),
                shape,
            },
        );
    }
    Ok(out)
}

/// Which layout file a safetensors file holds. The release keeps the drafter
/// in `mtp.safetensors` with unprefixed names.
fn layout_file(file: &str) -> File {
    if file == "mtp.safetensors" {
        File::Mtp
    } else {
        File::Model
    }
}

/// A disagreement between a header and the layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finding {
    /// A slot the config and precisions imply that no header holds.
    Missing { name: String, file: File },
    /// A header tensor no slot accounts for.
    Unexpected { name: String, file: String },
    /// A name that parses but does not render back to itself.
    NotCanonical { name: String },
    WrongDtype {
        name: String,
        want: Dtype,
        got: String,
    },
    WrongShape {
        name: String,
        want: Vec<usize>,
        got: Vec<usize>,
    },
    /// The same name in two files.
    Duplicate { name: String, files: [String; 2] },
    /// `hf_quant_config.json` declares a tensor the layout does not have.
    DeclaredNotLaidOut { name: String },
}

/// What [`inspect`] examined and found.
#[derive(Debug, Clone)]
pub struct Inspection {
    pub files: Vec<String>,
    pub tensors: usize,
    pub expected_slots: usize,
    /// Slots whose shape the layout predicted and the header confirmed.
    pub shape_checked: usize,
    /// Drafter tensors whose shape nothing public fixes; dtype checked only.
    pub shape_unpredicted: Vec<String>,
    pub findings: Vec<Finding>,
    /// Physical layers by expert precision, as declared.
    pub expert_precision: BTreeMap<usize, Precision>,
    pub schedule: Schedule,
    pub gaps: Vec<Gap>,
}

impl Inspection {
    pub fn is_bijection(&self) -> bool {
        self.findings.is_empty() && self.tensors > 0 && self.expected_slots == self.tensors
    }
}

/// Compare headers against the layout.
pub fn inspect(
    cfg: &Inkling2LoopConfig,
    precisions: &Precisions,
    headers: &[BTreeMap<String, HeaderEntry>],
) -> Result<Inspection, String> {
    let expected = layout::expected(cfg, precisions)?;
    let mut findings = Vec::new();
    for name in layout::undeclared_by_layout(&expected, precisions) {
        findings.push(Finding::DeclaredNotLaidOut { name });
    }

    let mut all: BTreeMap<(File, String), HeaderEntry> = BTreeMap::new();
    let mut files = BTreeSet::new();
    for h in headers {
        for (name, e) in h {
            files.insert(e.file.clone());
            let key = (layout_file(&e.file), name.clone());
            if let Some(prev) = all.insert(key, e.clone()) {
                findings.push(Finding::Duplicate {
                    name: name.clone(),
                    files: [prev.file, e.file.clone()],
                });
            }
        }
    }

    let mut matched: BTreeSet<Slot> = BTreeSet::new();
    let mut shape_checked = 0;
    let mut shape_unpredicted = Vec::new();
    for ((file, name), e) in &all {
        let Some(slot) = Slot::parse(*file, name) else {
            findings.push(Finding::Unexpected {
                name: name.clone(),
                file: e.file.clone(),
            });
            continue;
        };
        if slot.tensor_name() != *name {
            findings.push(Finding::NotCanonical { name: name.clone() });
            continue;
        }
        let Some(Expect { dtype, shape }) = expected.get(&slot) else {
            // Parses as a slot this config does not have, e.g. a QK-norm on
            // a layer that has an L2 clip, or a controller outside the core.
            findings.push(Finding::Unexpected {
                name: name.clone(),
                file: e.file.clone(),
            });
            continue;
        };
        matched.insert(slot);
        if Dtype::parse(&e.dtype) != Some(*dtype) {
            findings.push(Finding::WrongDtype {
                name: name.clone(),
                want: *dtype,
                got: e.dtype.clone(),
            });
        }
        match shape {
            Some(want) if *want != e.shape => findings.push(Finding::WrongShape {
                name: name.clone(),
                want: want.clone(),
                got: e.shape.clone(),
            }),
            Some(_) => shape_checked += 1,
            None => shape_unpredicted.push(name.clone()),
        }
    }
    for slot in expected.keys() {
        if !matched.contains(slot) {
            findings.push(Finding::Missing {
                name: slot.tensor_name(),
                file: slot.file(),
            });
        }
    }

    let t = &cfg.text_config;
    let mut expert_precision = BTreeMap::new();
    for i in 0..t.num_hidden_layers {
        let name = format!("model.llm.layers.{i}.mlp.experts.w13_weight");
        if let Some(p) = precisions.of(&name) {
            expert_precision.insert(i, p);
        }
    }

    Ok(Inspection {
        files: files.into_iter().collect(),
        tensors: all.len(),
        expected_slots: expected.len(),
        shape_checked,
        shape_unpredicted,
        findings,
        gaps: native_gaps(cfg, &expert_precision),
        expert_precision,
        schedule: Schedule::of(t),
    })
}

/// Inspect a checkpoint directory from its JSON sidecars and safetensors
/// headers.
pub fn inspect_dir(dir: &Path) -> Result<(Inkling2LoopConfig, Inspection), String> {
    let read = |name: &str| {
        std::fs::read_to_string(dir.join(name))
            .map_err(|e| format!("reading {}: {e}", dir.join(name).display()))
    };
    let cfg = Inkling2LoopConfig::from_json(&read("config.json")?)?;
    let precisions = Precisions::from_hf_quant_config(&read("hf_quant_config.json")?)?;
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("listing {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(format!("no *.safetensors in {}", dir.display()));
    }
    let headers = shards
        .iter()
        .map(|p| read_header(p))
        .collect::<Result<Vec<_>, _>>()?;
    let inspection = inspect(&cfg, &precisions, &headers)?;
    Ok((cfg, inspection))
}

/// Something the release needs that Mary does not have natively.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    pub what: &'static str,
    pub detail: String,
}

/// The operators and formats between this checkpoint and a native forward.
///
/// Mary's NVFP4 *weight decode* applies once the per-expert factor is
/// formed (`experts::nvfp4_scale2`), and MXFP8 has a host decode
/// (`experts::decode_mxfp8_row`); neither has a device GEMM on this path yet,
/// so both stay listed.
pub fn native_gaps(cfg: &Inkling2LoopConfig, experts: &BTreeMap<usize, Precision>) -> Vec<Gap> {
    let t = &cfg.text_config;
    let l = &t.loop_config;
    let layers = |p: fn(&Precision) -> bool| -> Vec<usize> {
        experts
            .iter()
            .filter(|(_, q)| p(q))
            .map(|(i, _)| *i)
            .collect()
    };
    let nvfp4 = layers(|p| matches!(p, Precision::Nvfp4 { .. }));
    let mxfp8 = layers(|p| matches!(p, Precision::Mxfp8 { .. }));
    let mut gaps = vec![
        Gap {
            what: "looped core",
            detail: format!(
                "layers {}..{} run {} passes with separate KV and conv state per pass; \
                 passes after the first restart from the pre-loop state",
                l.core_start, l.core_end, l.num_passes
            ),
        },
        Gap {
            what: "cross-pass messages",
            detail: format!(
                "attention over the previous pass's block states with 4 learned queries \
                 per core layer, then 4 rank-{} message gates (Q, K, V, MLP)",
                l.gate_rank
            ),
        },
        Gap {
            what: "block attention residual",
            detail: format!(
                "attention-residual inputs before every attention (3 queries on global \
                 layers), MLP and the final norm, blocks of {} layers",
                t.n_layers_per_attn_res_block
            ),
        },
        Gap {
            what: "attention variant",
            detail: format!(
                "per-head scaled L2 clip of Q/K ({}) and of the relative state ({}) in place \
                 of QK-norm, a learned sink per head, a sigmoid output gate, log scaling past {}",
                t.qk_l2_clip_threshold, t.rel_l2_clip_threshold, t.log_scaling_n_floor
            ),
        },
        Gap {
            what: "dynamic K/V convolution",
            detail: "global layers predict their K and V short-conv kernels from the input \
                     (weight_proj + static_w)"
                .to_string(),
        },
        Gap {
            what: "softcapped experts and logits",
            detail: format!(
                "expert gate softcap {}, up softcap {}, final logit softcap {}",
                t.mlp_gate_softcap, t.mlp_up_softcap, t.final_logit_softcapping
            ),
        },
    ];
    if !nvfp4.is_empty() {
        gaps.push(Gap {
            what: "NVFP4 expert lane",
            detail: format!(
                "layers {nvfp4:?}: weights decode through Mary's NVFP4 codec with \
                 scale2 = amax / outer_scale_denominator, but activations use the \
                 'fouroversix_mae' per-token scale rule, which Mary's quantiser does not \
                 implement"
            ),
        });
    }
    if !mxfp8.is_empty() {
        gaps.push(Gap {
            what: "MXFP8 expert lane",
            detail: format!(
                "layers {mxfp8:?}: E4M3 values with UE8M0 scales per 32; host decode only, \
                 no device GEMM (sm_121 kind::mxf8f6f4 is untested in Mary)"
            ),
        });
    }
    gaps.push(Gap {
        what: "vision tower",
        detail: format!(
            "{}-layer ViT with a class token and a merge-{} projector, not the original \
             Inkling's HMLP",
            cfg.vision_config.num_hidden_layers, cfg.vision_config.adapter_merge_size
        ),
    });
    if cfg.mtp_config.is_some() {
        gaps.push(Gap {
            what: "drafter",
            detail: "mtp.safetensors is a DSpark-style drafter with a Markov head, not the \
                     original Inkling's MTP stack (optional for inference)"
                .to_string(),
        });
    }
    gaps
}

/// Refuse to run a model whose forward Mary does not have.
pub fn require_native_forward(cfg: &Inkling2LoopConfig) -> Result<(), String> {
    // Without hf_quant_config the expert precisions are unknown, so name the
    // gaps that do not depend on them.
    let gaps = native_gaps(cfg, &BTreeMap::new());
    refuse(&gaps)
}

/// The refusal text for a list of gaps.
pub fn refuse(gaps: &[Gap]) -> Result<(), String> {
    if gaps.is_empty() {
        return Ok(());
    }
    let list: Vec<String> = gaps
        .iter()
        .map(|g| format!("{}: {}", g.what, g.detail))
        .collect();
    Err(format!(
        "Inkling 2 Mini (inkling2_loop) is recognised, but Mary has no native forward \
         for it yet; it is not loaded as the original Inkling. Missing: {}",
        list.join("; ")
    ))
}

/// The explanation a loader of the original Inkling gives when handed a
/// Mini config: parse it as what it is, then refuse with the gaps.
pub fn refusal_for_config(text: &str) -> String {
    match Inkling2LoopConfig::from_json(text) {
        Ok(cfg) => require_native_forward(&cfg)
            .err()
            .unwrap_or_else(|| "Inkling 2 Mini has no native forward in this crate".to_string()),
        Err(e) => e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("fixtures/config.json");
    const QUANT: &str = include_str!("fixtures/hf_quant_config.json");
    const HEADERS: &str = include_str!("fixtures/mini_headers.tsv");

    /// The pinned headers, as per-file maps.
    fn pinned() -> Vec<BTreeMap<String, HeaderEntry>> {
        let mut by_file: BTreeMap<String, BTreeMap<String, HeaderEntry>> = BTreeMap::new();
        for line in HEADERS.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 4, "{line}");
            let shape = f[3].split(',').map(|d| d.parse().unwrap()).collect();
            by_file.entry(f[0].to_string()).or_default().insert(
                f[1].to_string(),
                HeaderEntry {
                    file: f[0].to_string(),
                    dtype: f[2].to_string(),
                    shape,
                },
            );
        }
        by_file.into_values().collect()
    }

    fn setup() -> (Inkling2LoopConfig, Precisions) {
        (
            Inkling2LoopConfig::from_json(CONFIG).unwrap(),
            Precisions::from_hf_quant_config(QUANT).unwrap(),
        )
    }

    #[test]
    fn the_pinned_release_is_a_bijection_with_the_layout() {
        let (cfg, p) = setup();
        let h = pinned();
        assert_eq!(h.len(), 2, "model and mtp files");
        let ins = inspect(&cfg, &p, &h).unwrap();
        assert_eq!(ins.findings, vec![], "{:#?}", ins.findings);
        assert!(ins.is_bijection());
        // 1244 decoder and vision tensors (the index's count) plus 54 drafter.
        assert_eq!(ins.tensors, 1244 + 54);
        assert_eq!(
            ins.files,
            vec!["model-00001-of-00001.safetensors", "mtp.safetensors"]
        );
        assert_eq!(ins.shape_checked + ins.shape_unpredicted.len(), ins.tensors);
        assert_eq!(
            ins.shape_unpredicted,
            vec![
                "fuse_weight",
                "markov_head.rnn_proj.bias",
                "markov_head.rnn_proj.weight"
            ]
        );
        // The precision split is read, not assumed.
        let nv = ins
            .expert_precision
            .values()
            .filter(|q| matches!(q, Precision::Nvfp4 { block: 16 }))
            .count();
        let mx: Vec<usize> = ins
            .expert_precision
            .iter()
            .filter(|(_, q)| matches!(q, Precision::Mxfp8 { block: 32, .. }))
            .map(|(i, _)| *i)
            .collect();
        assert_eq!((nv, mx), (20, vec![20, 21, 22, 23]));
        assert_eq!(ins.schedule.len(), 48);
    }

    #[test]
    fn the_inspection_notices_what_it_should() {
        let (cfg, p) = setup();
        let mut h = pinned();
        let model = h
            .iter_mut()
            .find(|m| m.contains_key("model.llm.embed.weight"))
            .unwrap();
        // A tensor dropped, one reshaped, a Small-style QK-norm added.
        model.remove("model.llm.layers.3.attn.gate_attn.weight");
        model
            .get_mut("model.llm.layers.20.mlp.experts.w13_weight_scale")
            .unwrap()
            .shape = vec![128, 2304, 144];
        model.insert(
            "model.llm.layers.0.attn.q_norm.weight".to_string(),
            HeaderEntry {
                file: "model-00001-of-00001.safetensors".into(),
                dtype: "BF16".into(),
                shape: vec![128],
            },
        );
        let ins = inspect(&cfg, &p, &h).unwrap();
        assert!(!ins.is_bijection());
        assert!(ins.findings.contains(&Finding::Missing {
            name: "model.llm.layers.3.attn.gate_attn.weight".into(),
            file: File::Model
        }));
        assert!(
            ins.findings
                .iter()
                .any(|f| matches!(f, Finding::WrongShape { name, .. }
            if name == "model.llm.layers.20.mlp.experts.w13_weight_scale"))
        );
        assert!(
            ins.findings
                .iter()
                .any(|f| matches!(f, Finding::Unexpected { name, .. }
            if name == "model.llm.layers.0.attn.q_norm.weight"))
        );
        // An inspection of nothing is never a bijection.
        assert!(!inspect(&cfg, &p, &[]).unwrap().is_bijection());
    }

    #[test]
    fn the_original_inkling_checkpoint_is_not_a_mini_checkpoint() {
        // Small's real expert sidecar names do not parse here.
        for name in [
            "model.llm.layers.10.mlp.experts.w13_weight.scale",
            "model.llm.layers.10.mlp.experts.w13_weight.scale2",
            "model.llm.layers.0.attn.q_norm.weight",
            "model.llm.layers.0.mlp.w13_dn.weight",
        ] {
            let s = Slot::parse(File::Model, name);
            assert!(s.is_none(), "{name} parsed as {s:?}");
        }
    }

    #[test]
    fn running_it_is_refused_with_the_gaps() {
        let e = refusal_for_config(CONFIG);
        assert!(e.contains("recognised") && e.contains("looped core"), "{e}");
        assert!(e.contains("not loaded as the original Inkling"), "{e}");
        let (cfg, p) = setup();
        let ins = inspect(&cfg, &p, &pinned()).unwrap();
        assert!(ins.gaps.iter().any(|g| g.what == "MXFP8 expert lane"));
        assert!(refuse(&ins.gaps).is_err());
    }
}

#[cfg(test)]
mod original_inkling_loader {
    use crate::models::inkling::config::InklingConfig;

    #[test]
    fn the_original_loader_refuses_mini_and_still_reads_small() {
        let small = InklingConfig::from_json(include_str!("fixtures/inkling_small_config.json"))
            .expect("the original Inkling's published config still parses");
        assert_eq!(small.text_config.num_hidden_layers, 42);
        assert_eq!(small.text_config.dense_mlp_idx, 2);
        let e = InklingConfig::from_json(include_str!("fixtures/config.json"))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("Inkling 2 Mini (inkling2_loop) is recognised"),
            "{e}"
        );
        assert!(e.contains("cross-pass messages"), "{e}");
        let other = InklingConfig::from_json(r#"{"model_type":"inkling2"}"#).unwrap_err();
        assert!(other.to_string().contains("Inkling 2 Small"), "{other}");
    }
}
