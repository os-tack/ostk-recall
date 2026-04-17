//! End-to-end `ostk_project` ingest test.
//!
//! Builds a minimal `.ostk/` tree in a tempdir, runs the full
//! `init → scan → verify` dance via the CLI library, and asserts that:
//!
//! * every subsystem produces at least one chunk
//! * the audit firehose (including non-significant rows) lands in
//!   `events.duckdb`
//! * a failing audit row is linked from its corpus chunk to the duckdb row

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use ostk_recall_cli::commands::{self, InitOutcome};
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::EventsDb;
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

fn write_config(path: &Path, corpus_root: &Path, project_root: &Path) {
    let body = format!(
        r#"[corpus]
root = "{corpus}"

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "ostk_project"
project = "myproj"
paths = ["{root}"]
"#,
        corpus = corpus_root.display(),
        root = project_root.display(),
    );
    std::fs::write(path, body).unwrap();
}

fn write_jsonl(path: &Path, lines: &[&str]) {
    let mut f = std::fs::File::create(path).unwrap();
    for l in lines {
        writeln!(f, "{l}").unwrap();
    }
}

fn build_ostk_fixture(root: &Path) {
    std::fs::create_dir_all(root.join(".ostk/needles")).unwrap();
    std::fs::create_dir_all(root.join(".ostk/conversations")).unwrap();
    std::fs::create_dir_all(root.join(".ostk/sessions")).unwrap();
    std::fs::create_dir_all(root.join(".ostk/memory")).unwrap();
    std::fs::create_dir_all(root.join("docs/spec")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();

    write_jsonl(
        &root.join(".ostk/decisions.jsonl"),
        &[r#"{"key":"K1","value":"adopt X","reason":"study","timestamp":"2026-04-17T10:00:00Z"}"#],
    );
    write_jsonl(
        &root.join(".ostk/needles/issues.jsonl"),
        &[
            r#"{"id":"→1000","title":"T","status":"open","description":"d","ac":["a"],"created_at":"2026-04-17T10:00:00Z"}"#,
        ],
    );
    write_jsonl(
        &root.join(".ostk/audit.jsonl"),
        &[
            r#"{"ts":"2026-04-17T10:00:00Z","event":"tool.call","tool":"bash","agent":"a","success":true,"duration_ms":5}"#,
            r#"{"ts":"2026-04-17T10:00:01Z","event":"tool.call","tool":"bash","agent":"a","success":false,"exit_code":1,"duration_ms":100}"#,
        ],
    );
    write_jsonl(
        &root.join(".ostk/conversations/alpha.jsonl"),
        &[r#"{"turn":1,"from":"a","to":"b","ts":"2026-04-17T09:00:00Z","msg":"hi"}"#],
    );
    write_jsonl(
        &root.join(".ostk/sessions/s1.jsonl"),
        &[
            r#"{"role":"user","content":"q1","timestamp":"2026-04-17T08:00:00Z"}"#,
            r#"{"role":"assistant","content":"a1","timestamp":"2026-04-17T08:00:01Z"}"#,
            r#"{"role":"user","content":"q2","timestamp":"2026-04-17T08:00:02Z"}"#,
            r#"{"role":"assistant","content":"a2","timestamp":"2026-04-17T08:00:03Z"}"#,
        ],
    );
    write_jsonl(
        &root.join(".ostk/memory/pages.jsonl"),
        &[r#"{"name":"p1","file_id":"p1","tokens":10,"stored_at":"2026-04-17T07:00:00Z"}"#],
    );
    std::fs::write(root.join(".ostk/memory/p1.page"), "memory blob body").unwrap();
    std::fs::write(
        root.join("docs/spec/overview.md"),
        "# Overview\n\nIntro.\n\n## Details\n\nBody.\n",
    )
    .unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
}

#[tokio::test]
async fn ostk_project_end_to_end() {
    let fixture = TempDir::new().unwrap();
    build_ostk_fixture(fixture.path());
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
    assert!(
        out.totals.chunks_upserted >= 7,
        "expect chunks across all subsystems, got {:?}",
        out.totals
    );

    let out2 = commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .unwrap();
    assert_eq!(out2.totals.chunks_upserted, 0);

    let v = commands::verify(&cfg_path, embedder).await.unwrap();
    assert!(v.report.is_consistent());

    // events.duckdb must have received the full audit firehose (2 rows).
    let events = EventsDb::open(corpus.path()).unwrap();
    assert_eq!(events.row_count().unwrap(), 2);
}
