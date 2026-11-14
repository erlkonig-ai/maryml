//! Give old model commits the metadata archive the current writer would have
//! written: see [`mary::metadata_backfill`].
//!
//!   model_metadata_backfill --catalog
//!   model_metadata_backfill <PILE> --collection <64-hex> --key <signing-key>
//!   model_metadata_backfill <PILE> --collection <64-hex> --key <signing-key>
//!       --apply --expect-catalog <64-hex>
//!
//! The default is a dry run, which opens the pile read-only and prints what an
//! apply would write. `--apply` appends, for every commit the key signed whose
//! enriched metadata differs, the description blobs, the new metadata archive
//! and `COMMIT(collection, data, M')`; it requires `--expect-catalog` naming the
//! catalog hash a dry run of the same build printed. `--catalog` lists every
//! attribute id the catalog describes and the declarations behind it.
use anyhow::{Context, Result, bail};
use mary::metadata_backfill::{self, Catalog, Report, hex_upper};
use std::path::PathBuf;
use triblespace::core::signing_key_file;

const USAGE: &str = "usage: model_metadata_backfill --catalog\n       model_metadata_backfill <PILE> --collection <64-hex> --key <signing-key> [--apply --expect-catalog <64-hex>]";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut pile = None;
    let mut collection = None;
    let mut key = None;
    let mut apply = false;
    let mut expect = None;
    let mut list = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--collection" => collection = Some(args.next().context("--collection needs a value")?),
            "--key" => key = Some(PathBuf::from(args.next().context("--key needs a value")?)),
            "--apply" => apply = true,
            "--expect-catalog" => {
                expect = Some(args.next().context("--expect-catalog needs a value")?)
            }
            "--catalog" => list = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other if other.starts_with("--") => bail!("unknown flag {other}\n{USAGE}"),
            other if pile.is_none() => pile = Some(PathBuf::from(other)),
            other => bail!("unexpected argument {other}\n{USAGE}"),
        }
    }

    let catalog = Catalog::frozen();
    if list {
        for entry in catalog.entries() {
            println!("ATTR {:X} {}", entry.id, entry.declarations.join(" | "));
        }
        println!(
            "catalog blake3:{} ({} attribute ids, {} tribles)",
            hex_upper(&catalog.hash().raw),
            catalog.entries().len(),
            catalog.facts().len()
        );
        return Ok(());
    }

    let (Some(pile), Some(collection), Some(key)) = (pile, collection, key) else {
        bail!("{USAGE}");
    };
    let collection = mary::model_collection::parse_collection_handle(&collection)?;
    if apply {
        let Some(expect) = expect.as_deref() else {
            bail!("--apply requires --expect-catalog <hash printed by a dry run of this build>");
        };
        metadata_backfill::check_catalog(&catalog, expect)?;
    }
    let signer = signing_key_file::load_existing(&key)
        .with_context(|| format!("load signing key {}", key.display()))?;
    let report = metadata_backfill::run_path(&pile, &signer, &catalog, collection, apply)?;
    print_report(&report, &catalog);
    Ok(())
}

fn print_report(report: &Report, catalog: &Catalog) {
    println!(
        "{} collection {} signer {}",
        if report.applied { "APPLIED" } else { "DRY RUN" },
        hex_upper(&report.collection.raw),
        hex_upper(&report.signer)
    );
    println!("catalog blake3:{}", hex_upper(&report.catalog_hash.raw));
    println!(
        "commits {} (signer {}, other signers {}, bad signatures {})",
        report.commits, report.signer_commits, report.other_signers, report.bad_signatures
    );
    println!(
        "unreadable: data non-resident {} undecodable {}; metadata non-resident {} undecodable {}",
        report.data_non_resident,
        report.data_undecodable,
        report.metadata_non_resident,
        report.metadata_undecodable
    );
    println!(
        "current {} enriched {} (successor already present {})",
        report.current_commits, report.enriched_commits, report.present_commits
    );
    for row in &report.enriched {
        println!(
            "  data {} metadata {} -> {} ({} bytes, +{} tribles, {} attributes)",
            hex_upper(&row.data.raw),
            hex_upper(&row.old_metadata.raw),
            hex_upper(&row.metadata.raw),
            row.metadata_bytes,
            row.added_tribles,
            row.attributes
        );
    }
    println!(
        "{} {} commits, {} blobs, {} bytes",
        if report.applied {
            "wrote"
        } else {
            "would write"
        },
        report.new_commits,
        report.new_blobs,
        report.new_bytes
    );
    println!("known attributes {}:", report.known.len());
    for (id, commits) in &report.known {
        let declarations = catalog
            .get(*id)
            .map(|entry| entry.declarations.join(" | "))
            .unwrap_or_default();
        println!("  {id:X} commits {commits} {declarations}");
    }
    println!("unknown attributes {}:", report.unknown.len());
    for (id, unknown) in &report.unknown {
        println!(
            "  {id:X} commits {} already described in M {}",
            unknown.commits, unknown.described
        );
    }
}
