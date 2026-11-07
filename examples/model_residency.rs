//! Model residency: is a model root's whole tensor closure in this pile?
//!
//! Freezes the pile's `models` collection through the ordinary
//! reader and loads the keymap of each named root the way every consumer
//! does (`selection::load_keymap_from_graph`), which reads every tensor leaf
//! and folds the packed roots' channel scales. No inference, no network, no
//! model directory. A missing or unreadable leaf fails that root loudly with
//! the reader's own message; an empty map is not success.
//!
//! Written 2026-09-13 for Sol's replication check of the two nomic roots in
//! self.pile:
//!
//! ```text
//! model_residency <pile> <root-id-hex>...
//! ```

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use mary::selection::{load_keymap_from_graph, ModelSelector};
use triblespace::prelude::*;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let pile = PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow!("usage: model_residency <pile> <root-id-hex>..."))?,
    );
    let roots: Vec<Id> = args
        .map(|hex| Id::from_hex(&hex).ok_or_else(|| anyhow!("not a 32-hex id: {hex}")))
        .collect::<Result<_>>()?;
    if roots.is_empty() {
        return Err(anyhow!("name at least one root id"));
    }

    let snapshot = mary::model_collection::load_model_collection_local_latest(&pile)
        .with_context(|| format!("freeze the model collection of {}", pile.display()))?;
    println!(
        "{}: model collection frozen, {} facts, {} support members",
        pile.display(),
        snapshot.facts().len(),
        snapshot.support().len()
    );

    let mut failed = 0usize;
    for root in roots {
        match load_keymap_from_graph(
            snapshot.facts(),
            snapshot.store(),
            ModelSelector::Root(root),
        ) {
            Ok(keymap) => {
                let elements: usize = keymap.values().map(|(data, _)| data.len()).sum();
                if keymap.is_empty() {
                    failed += 1;
                    println!("root {root:X}: EMPTY keymap, not resident");
                } else {
                    println!(
                        "root {root:X}: resident, {} tensors, {} elements",
                        keymap.len(),
                        elements
                    );
                }
            }
            Err(error) => {
                failed += 1;
                println!("root {root:X}: NOT resident: {error:#}");
            }
        }
    }
    if failed > 0 {
        std::process::exit(2);
    }
    Ok(())
}
