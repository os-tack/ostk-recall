//! End-to-end code-source ingest test.
//!
//! Writes a tiny source tree to a tempdir, points `ostk-recall scan` at
//! it through the CLI library entrypoints, and checks idempotency +
//! verify balance.

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
project = "code"
paths = ["{fixture}"]
extensions = ["rs"]
"#,
        corpus = corpus_root.display(),
        fixture = fixture.display(),
    );
    std::fs::write(path, body).unwrap();
}

fn write_fixture(root: &Path) {
    use std::fmt::Write as _;
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(root.join("short.rs"), "fn a() {}\nfn b() {}\n").unwrap();
    let mut big = String::new();
    for i in 0..400 {
        let _ = writeln!(big, "// line {i}");
    }
    std::fs::write(root.join("big.rs"), big).unwrap();
}

#[tokio::test]
async fn code_end_to_end() {
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
    assert_eq!(out.totals.errors, 0);
    assert!(out.totals.chunks_upserted >= 3, "short=1 + big>=2 chunks");

    let out2 = commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .unwrap();
    assert_eq!(out2.totals.chunks_upserted, 0, "idempotent rerun");
    assert_eq!(out2.totals.chunks_skipped_dup, out.totals.chunks_upserted);

    let v = commands::verify(&cfg_path, embedder).await.unwrap();
    assert!(v.report.is_consistent());
}
