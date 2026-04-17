//! End-to-end test for the per-source `ignore` field on `code` sources.
//!
//! Builds a fixture with a `.gitignore` that excludes `node_modules/`,
//! drops a fake dependency tree under it, and confirms the scanner
//! emits exactly one chunk for the real source file.

use std::fs;
use std::path::Path;
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
                let seed = ((t.len() % 100) as f32) * 0.01;
                (0..FAKE_DIM)
                    .map(|i| (i as f32).mul_add(0.001, seed))
                    .collect()
            })
            .collect()
    }
}

fn write_config(path: &Path, corpus_root: &Path, fixture: &Path) {
    let body = format!(
        r#"[corpus]
root = "{corpus}"

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "code"
project = "ignore-test"
paths = ["{fixture}"]
extensions = ["js"]
"#,
        corpus = corpus_root.display(),
        fixture = fixture.display(),
    );
    fs::write(path, body).unwrap();
}

fn write_fixture(root: &Path) {
    // .git/ so the `ignore` crate honors .gitignore for this tree.
    fs::create_dir_all(root.join(".git")).unwrap();
    fs::write(root.join(".gitignore"), "node_modules\n").unwrap();

    // A heavy "dep" tree the scanner must NOT walk.
    fs::create_dir_all(root.join("node_modules/big-dep")).unwrap();
    fs::write(
        root.join("node_modules/big-dep/index.js"),
        "module.exports = { huge: true };\n",
    )
    .unwrap();
    fs::write(
        root.join("node_modules/big-dep/package.json"),
        r#"{"name":"big-dep"}"#,
    )
    .unwrap();

    // The single real source file.
    fs::create_dir_all(root.join("src")).unwrap();
    fs::write(root.join("src/real.js"), "console.log('real');\n").unwrap();
}

#[tokio::test]
async fn ignore_excludes_node_modules() {
    let fixture = TempDir::new().unwrap();
    write_fixture(fixture.path());

    let corpus = TempDir::new().unwrap();
    let cfg_dir = TempDir::new().unwrap();
    let cfg_path = cfg_dir.path().join("config.toml");
    write_config(&cfg_path, corpus.path(), fixture.path());

    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);

    match commands::init(&cfg_path, Arc::clone(&embedder))
        .await
        .unwrap()
    {
        InitOutcome::Initialized { .. } => {}
        InitOutcome::WroteStarter { .. } => panic!("expected Initialized"),
    }

    let out = commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .unwrap();
    assert_eq!(out.totals.errors, 0, "scan reported errors: {out:?}");
    assert_eq!(
        out.totals.chunks_upserted, 1,
        "expected exactly 1 chunk for src/real.js (node_modules must be skipped); got {}",
        out.totals.chunks_upserted
    );
}
