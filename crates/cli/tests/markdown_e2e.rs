//! End-to-end markdown ingest test.
//!
//! Drives the CLI `commands::{init, scan, verify}` functions with a
//! deterministic `FakeEmbedder` over the checked-in markdown fixture tree
//! at `tests/fixtures/markdown/`. No network, no model download.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ostk_recall_cli::commands::{self, InitOutcome};
use ostk_recall_pipeline::ChunkEmbedder;
use tempfile::TempDir;

const FAKE_DIM: usize = 16;

struct FakeEmbedder;

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        FAKE_DIM
    }
    #[allow(clippy::cast_precision_loss)]
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                // Deterministic but text-distinguishing.
                let seed = ((t.len() % 100) as f32) * 0.01;
                (0..FAKE_DIM)
                    .map(|i| (i as f32).mul_add(0.001, seed))
                    .collect()
            })
            .collect()
    }
}

fn fixture_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is the CLI crate root; fixtures live at
    // ../../tests/fixtures/markdown relative to that.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("markdown")
        .join("notes")
}

fn write_config(path: &Path, corpus_root: &Path, fixture: &Path) {
    let body = format!(
        r#"[corpus]
root = "{corpus}"

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "markdown"
project = "notes"
paths = ["{fixture}"]
"#,
        corpus = corpus_root.display(),
        fixture = fixture.display(),
    );
    std::fs::write(path, body).unwrap();
}

#[tokio::test]
async fn markdown_end_to_end() {
    let corpus = TempDir::new().unwrap();
    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("config.toml");

    let fixture = fixture_root();
    assert!(
        fixture.exists(),
        "fixture dir missing: {}",
        fixture.display()
    );
    write_config(&config_path, corpus.path(), &fixture);

    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);

    // 1. init — initializes the corpus at the configured root.
    let outcome = commands::init(&config_path, Arc::clone(&embedder))
        .await
        .expect("init");
    match outcome {
        InitOutcome::Initialized { root, dim, .. } => {
            assert_eq!(root, corpus.path());
            assert_eq!(dim, FAKE_DIM);
        }
        InitOutcome::WroteStarter { .. } => panic!("expected Initialized"),
    }

    // 2. scan — should upsert the fixture's chunks.
    let out1 = commands::scan(&config_path, Arc::clone(&embedder), None, false)
        .await
        .expect("scan");
    assert!(
        out1.totals.chunks_upserted > 0,
        "first scan should upsert: {:?}",
        out1.totals
    );
    assert_eq!(out1.totals.errors, 0);
    assert_eq!(out1.totals.chunks_skipped_dup, 0);

    // 3. scan again — idempotent, zero upserts.
    let out2 = commands::scan(&config_path, Arc::clone(&embedder), None, false)
        .await
        .expect("scan rerun");
    assert_eq!(
        out2.totals.chunks_upserted, 0,
        "rerun must upsert zero; got {:?}",
        out2.totals
    );
    assert_eq!(out2.totals.chunks_skipped_dup, out1.totals.chunks_upserted);

    // 4. verify — totals consistent between corpus + ingest.
    let ver = commands::verify(&config_path, Arc::clone(&embedder))
        .await
        .expect("verify");
    assert!(
        ver.report.is_consistent(),
        "drift detected: {:?}",
        ver.report
    );
    assert_eq!(ver.report.corpus_total, out1.totals.chunks_upserted);
}

#[tokio::test]
async fn init_writes_starter_when_missing() {
    let tmp = TempDir::new().unwrap();
    let config_path = tmp.path().join("missing").join("config.toml");

    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
    let out = commands::init(&config_path, embedder).await.expect("init");
    match out {
        InitOutcome::WroteStarter { path } => {
            assert_eq!(path, config_path);
            assert!(path.exists());
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(text.contains("[corpus]"));
        }
        InitOutcome::Initialized { .. } => panic!("expected WroteStarter"),
    }
}
