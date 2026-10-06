//! One native model pile, exact source dtypes, ordinary model publication.
use anyhow::{Context, Result};
use clap::Parser;
use mary::models::breeze;
use std::path::PathBuf;
use triblespace::core::{repo::pile::Pile, signing_key_file};

#[derive(Parser)]
#[command(about = "Import complete Breeze and external codec assets into a native model pile")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    /// New task-owned destination; existing files are never overwritten/appended.
    #[arg(long)]
    pile: PathBuf,
    /// Explicit existing task signer, never generated implicitly.
    #[arg(long)]
    key: PathBuf,
    #[arg(long)]
    revision: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let key = signing_key_file::load_existing(&args.key).context("load task signer")?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.pile)
        .context("create new task model pile")?;
    let mut pile = Pile::open(&args.pile).context("open new model pile")?;
    let result = (|| -> Result<_> {
        mary::model_collection::model_graph_collection_or_create(&mut pile, &key)?;
        let candidate =
            breeze::import::ingest_checkpoint(&mut pile, &args.checkpoint, &args.revision)?;
        let ids = candidate.artifacts;
        let commit =
            mary::model_collection::publish_model_fragment(&mut pile, &key, candidate.fragment)?;
        Ok(serde_json::json!({
            "source":breeze::SOURCE, "revision":args.revision,
            "model_root":format!("{:X}", ids.model_root),
            "config_root":format!("{:X}", ids.config_root),
            "tokenizer_asset":format!("{:X}", ids.tokenizer_asset),
            "external_codec_root":format!("{:X}", ids.external_codec_root),
            "external_codec_config_root":format!("{:X}", ids.external_codec_config_root),
            "tensor_count":candidate.tensor_count,"parameter_count":candidate.parameter_count,
            "payload_bytes":candidate.payload_bytes,
            "commit":triblespace::core::collection::CollectionRecord::Commit(commit).fingerprint().to_string()
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
