//! Explicit fresh scratch-pile block0 witness, not a full model publication.
//! No default pile/key: every invocation names the local model, identity and
//! new output directory. A throwaway signer is generated in memory; it never
//! opens a home/default key or a live collection. Never overwrites an output.
use std::{fs,path::PathBuf};
use anyhow::{Context,Result,ensure};
use clap::Parser;
use serde_json::{Value,json};
use mary::models::flint::{weights::GgufWeights,pile::{import_block0,PileWeights}};
use triblespace::prelude::*;

#[derive(Parser)]
struct Args {
    model:PathBuf,
    new_output_dir:PathBuf,
    /// Exact FileIdentity JSON from prior admission; no whole-model hash pass.
    #[arg(long)] expected_identity:String,
}

fn main()->Result<()> {
    let args=Args::parse();
    ensure!(args.expected_identity.len()<=1024,"identity JSON exceeds1KiB");
    let expected:Value=serde_json::from_str(&args.expected_identity)?;
    let key=ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
    fs::create_dir(&args.new_output_dir).context("scratch output directory must be new")?;
    let mut source=GgufWeights::open_block(&args.model,0)?;
    ensure!(serde_json::to_value(&source.file.identity)?==expected,"source identity differs from admission");
    let (fragment,tensors)=import_block0(&mut source)?;
    let root=fragment.root().context("import root")?;
    let source_bytes:u64=tensors.iter().map(|t|t.source.absolute_end-t.source.absolute_start).sum();
    let tensor_bytes:usize=tensors.iter().map(|t|t.tensor_bytes).sum();
    ensure!(tensors.len()==19 && source_bytes==31_868_060,"block0 inventory/byte budget differs");
    let pile_path=args.new_output_dir.join("block0.pile");
    fs::OpenOptions::new().write(true).create_new(true).open(&pile_path)?;
    let mut pile=Pile::open(&pile_path).map_err(|e|anyhow::anyhow!("open fresh scratch pile: {e}"))?;
    let commit=mary::model_collection::publish_model_fragment(&mut pile,&key,fragment)?;
    pile.close().map_err(|e|anyhow::anyhow!("close imported scratch pile: {e}"))?;
    // Reopen through the normal frozen native model-collection path, after the
    // publishing Pile handle was closed. No GGUF decoder or path is in reader.
    let reopened=PileWeights::open(&pile_path,root)?;
    reopened.check_imported(&tensors)?;
    source.file.check_identity()?;
    let report=json!({"scope":"native typed block0 scratch import/reopen; no GPU/full model/tokenizer",
        "model_path":args.model,"model_identity":source.file.identity,"whole_model_freshly_hashed":false,
        "metadata_end":source.file.index.metadata_end,"metadata":source.file.index.metadata,
        "pile":pile_path,"model_root":format!("{root:X}"),"model_collection":reopened.collection(),
        "signer":"fresh throwaway key generated in memory; secret not saved",
        "signer_public_key_bytes":key.verifying_key().to_bytes().to_vec(),
        "block":format!("{:X}",reopened.block),"config_node":format!("{:X}",reopened.config_node),
        "config":reopened.config,"native_commit":format!("{commit:?}"),"support_members":reopened.support_members(),
        "tensor_count":tensors.len(),"logical_selected_source_payload_bytes":source_bytes,
        "typed_tensor_bytes_before_content_dedup":tensor_bytes,"max_source_read_bytes":7_077_888,
        "roundtrip":"all19 row-major shapes and typed payload digests matched after close/reopen",
        "layout":"native E2M1 codes/E4M3 scale planes/global1; separate post-matmul F32 leaves",
        "bf16":"supported by adapter synthetic fixtures; not consumed/imported by actual dense block0",
        "reads":tensors});
    fs::write(args.new_output_dir.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
