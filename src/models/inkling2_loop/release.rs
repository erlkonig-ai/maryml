//! The pinned Inkling 2 Mini release: its files, and the names the importer
//! ([`super::import`]) and the reader ([`super::pile`]) agree on.

/// The release this importer is pinned to.
pub const SOURCE: &str = "thinkingmachines/Inkling-2-Mini";
pub const REVISION: &str = "b85c3cfa615fc413cc9af3379f491b963f3b586a";

/// The weights file and the drafter file.
pub const MODEL_FILE: &str = "model-00001-of-00001.safetensors";
pub const MTP_FILE: &str = "mtp.safetensors";
/// Label of the drafter root under the main root.
pub const DRAFTER: &str = "mtp";
/// Label of the source manifest under the main root.
pub const MANIFEST: &str = "native-source-manifest";

/// The document name under which a safetensors file's header is kept.
pub fn header_document(file: &str) -> String {
    format!("{file}#header")
}

/// The release's files at [`REVISION`]: path, size, and the LFS sha256 where
/// the repository publishes one. Read from the Hub's tree listing on
/// 2026-11-22 (`mini-spark-reference-20261122/pinned-files.txt`). The
/// repository's `.gitattributes` is git metadata, not part of the model, and
/// is not expected in the directory.
pub const RELEASE: &[(&str, u64, Option<&str>)] = &[
    ("README.md", 54870, None),
    ("chat_template.jinja", 6868, None),
    ("config.json", 2722, None),
    ("generation_config.json", 98, None),
    ("hf_quant_config.json", 67924, None),
    (
        MODEL_FILE,
        19526019168,
        Some("65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1"),
    ),
    ("model.safetensors.index.json", 109300, None),
    (
        MTP_FILE,
        635801416,
        Some("7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2"),
    ),
    ("processor_config.json", 461, None),
    ("special_tokens_map.json", 529, None),
    (
        "tiktoken/tokenizer.model",
        3386494,
        Some("f170372dd655728909228220d77cf6b8104a59f38a91262a7d2c4027906ec0d8"),
    ),
    (
        "tokenizer.json",
        26178678,
        Some("7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6"),
    ),
    ("tokenizer_config.json", 13235, None),
];
