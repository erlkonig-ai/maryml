//! Import Step-Audio2 Mini's native BF16 decoder and exact tokenizer/config into
//! one task-owned model pile. This command performs no neural-network inference.
use anyhow::{Context, Result};
use clap::Parser;
use mary::models::step_audio2;
use std::path::PathBuf;
use triblespace::core::{repo::pile::Pile, signing_key_file};

#[derive(Parser)]
#[command(about = "Import Step-Audio2 Mini decoder + tokenizer/config into a native model pile")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    pile: PathBuf,
    /// Existing signing key; an identity is never generated implicitly.
    #[arg(long)]
    key: PathBuf,
    /// Declared source revision, independently of the actual content-addressed weights.
    #[arg(long)]
    revision: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let key = signing_key_file::load_existing(&args.key).context("load existing signing key")?;
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.pile)
    {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error).context("create task model pile"),
    }
    let mut pile = Pile::open(&args.pile).context("open model pile")?;
    let result = (|| -> Result<_> {
        pile.refresh().context("refresh model pile")?;
        mary::model_collection::model_graph_collection_or_create(&mut pile, &key)?;
        let candidate =
            step_audio2::import::ingest_checkpoint(&mut pile, &args.checkpoint, &args.revision)?;
        let ids = candidate.artifacts;
        let commit =
            mary::model_collection::publish_model_fragment(&mut pile, &key, candidate.fragment)?;
        Ok(serde_json::json!({
            "model_root": format!("{:X}", ids.model_root),
            "config_root": format!("{:X}", ids.config_root),
            "tokenizer_asset": format!("{:X}", ids.tokenizer_asset),
            "tensor_count": candidate.tensor_count,
            "parameter_count": candidate.parameter_count,
            "commit": triblespace::core::collection::CollectionRecord::Commit(commit).fingerprint().to_string(),
            "scope": "native BF16 text/speech decoder assets; no inference or token2wav networks"
        }))
    })();
    let close = pile.close();
    match (result, close) {
        (Ok(receipt), Ok(())) => {
            println!("{}", serde_json::to_string_pretty(&receipt)?);
            Ok(())
        }
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(anyhow::anyhow!("close model pile: {error}")),
        (Err(error), Err(close)) => {
            Err(error.context(format!("also failed to close pile: {close}")))
        }
    }
}
