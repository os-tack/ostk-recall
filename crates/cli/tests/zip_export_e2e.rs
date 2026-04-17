//! End-to-end `zip_export` ingest test.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use ostk_recall_cli::commands::{self, InitOutcome};
use ostk_recall_pipeline::ChunkEmbedder;
use tempfile::TempDir;
use zip::write::SimpleFileOptions;

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
kind = "zip_export"
project = "claudeai"
paths = ["{glob_pattern}"]
"#,
        corpus = corpus_root.display(),
    );
    std::fs::write(path, body).unwrap();
}

fn build_fixture_zip(path: &Path) {
    let f = File::create(path).unwrap();
    let mut zw = zip::ZipWriter::new(f);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let body = serde_json::json!([
        {
            "uuid": "conv-1",
            "name": "First convo",
            "chat_messages": [
                {"uuid": "m1", "text": "hello", "sender": "human",
                 "created_at": "2024-01-01T00:00:00Z"},
                {"uuid": "m2", "text": "hi there", "sender": "assistant",
                 "created_at": "2024-01-01T00:00:01Z"},
                {"uuid": "m3", "text": "thanks", "sender": "human",
                 "created_at": "2024-01-01T00:00:02Z"}
            ]
        },
        {
            "uuid": "conv-2",
            "name": "Second convo",
            "chat_messages": [
                {"uuid": "n1", "text": "question", "sender": "human",
                 "created_at": "2024-02-01T00:00:00Z"},
                {"uuid": "n2", "text": "answer", "sender": "assistant",
                 "created_at": "2024-02-01T00:00:01Z"},
                {"uuid": "n3", "text": "ok", "sender": "human",
                 "created_at": "2024-02-01T00:00:02Z"}
            ]
        }
    ]);
    zw.start_file("conversations.json", opts).unwrap();
    zw.write_all(body.to_string().as_bytes()).unwrap();
    zw.finish().unwrap();
}

#[tokio::test]
async fn zip_export_end_to_end() {
    let fixture = TempDir::new().unwrap();
    let zip_path = fixture.path().join("claude-data-export-2024.zip");
    build_fixture_zip(&zip_path);

    let corpus = TempDir::new().unwrap();
    let cfg_dir = TempDir::new().unwrap();
    let cfg_path = cfg_dir.path().join("config.toml");
    let pat = format!("{}/claude-data-export-*.zip", fixture.path().display());
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
    assert_eq!(out.totals.chunks_upserted, 6, "2 convos x 3 msgs");

    let out2 = commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .unwrap();
    assert_eq!(out2.totals.chunks_upserted, 0);

    let v = commands::verify(&cfg_path, embedder).await.unwrap();
    assert!(v.report.is_consistent());
}
