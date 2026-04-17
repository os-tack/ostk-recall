//! End-to-end `file_glob` ingest test.

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

fn write_config(path: &Path, corpus_root: &Path, glob_pattern: &str) {
    let body = format!(
        r#"[corpus]
root = "{corpus}"

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "file_glob"
project = "notes"
paths = ["{glob_pattern}"]
"#,
        corpus = corpus_root.display(),
    );
    std::fs::write(path, body).unwrap();
}

fn write_fixture(root: &Path) {
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("a.txt"), "alpha paragraph.\n\nbeta paragraph.\n").unwrap();
    std::fs::write(root.join("sub/b.txt"), "gamma paragraph.\n").unwrap();
    // Binary noise that should be skipped.
    std::fs::write(root.join("sub/c.bin"), b"\x00\x01garbage").unwrap();
}

#[tokio::test]
async fn file_glob_end_to_end() {
    let fixture = TempDir::new().unwrap();
    write_fixture(fixture.path());
    let corpus = TempDir::new().unwrap();
    let cfg_dir = TempDir::new().unwrap();
    let cfg_path = cfg_dir.path().join("config.toml");
    let pat = format!("{}/**/*.txt", fixture.path().display());
    write_config(&cfg_path, corpus.path(), &pat);

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
    assert!(
        out.totals.chunks_upserted >= 2,
        "both txt files should chunk, got {:?}",
        out.totals
    );

    let out2 = commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .unwrap();
    assert_eq!(out2.totals.chunks_upserted, 0);

    let v = commands::verify(&cfg_path, embedder).await.unwrap();
    assert!(v.report.is_consistent());
}
