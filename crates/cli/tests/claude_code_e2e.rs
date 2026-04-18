//! End-to-end `claude_code` ingest test.

use std::io::Write;
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

fn write_config(path: &Path, corpus_root: &Path, claude_root: &Path) {
    let body = format!(
        r#"[corpus]
root = "{corpus}"

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "claude_code"
project = "haystack"
paths = ["{root}"]
"#,
        corpus = corpus_root.display(),
        root = claude_root.display(),
    );
    std::fs::write(path, body).unwrap();
}

fn write_fixture(root: &Path) {
    let proj = root.join("-Users-x-projects-haystack");
    std::fs::create_dir_all(&proj).unwrap();

    let mut s1 = std::fs::File::create(proj.join("session-a.jsonl")).unwrap();
    writeln!(s1, r#"{{"type":"file-history-snapshot","messageId":"x"}}"#).unwrap();
    writeln!(
        s1,
        r#"{{"type":"user","message":{{"role":"user","content":"hi there"}},"timestamp":"2026-04-17T10:00:00Z"}}"#
    )
    .unwrap();
    writeln!(
        s1,
        r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"let me check"}},{{"type":"tool_use","name":"read","input":{{"path":"x"}}}},{{"type":"text","text":"done"}}]}},"timestamp":"2026-04-17T10:00:01Z"}}"#
    )
    .unwrap();

    let mut s2 = std::fs::File::create(proj.join("session-b.jsonl")).unwrap();
    writeln!(
        s2,
        r#"{{"type":"user","message":{{"role":"user","content":"q"}},"timestamp":"2026-04-17T11:00:00Z"}}"#
    )
    .unwrap();
    writeln!(
        s2,
        r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"a"}}]}},"timestamp":"2026-04-17T11:00:01Z"}}"#
    )
    .unwrap();
}

#[tokio::test]
async fn claude_code_end_to_end() {
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
    // Per-message chunking (Phase H):
    //   session-a: user "hi there" + assistant text "let me check" + tool_use + assistant text "done" = 4
    //   session-b: user "q" + assistant text "a" = 2
    //   total = 6
    assert_eq!(
        out.totals.chunks_upserted, 6,
        "expected 6 per-block chunks, got {:?}",
        out.totals
    );

    let out2 = commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .unwrap();
    assert_eq!(out2.totals.chunks_upserted, 0);

    let v = commands::verify(&cfg_path, embedder).await.unwrap();
    assert!(v.report.is_consistent());
}
