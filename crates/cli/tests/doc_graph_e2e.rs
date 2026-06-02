//! Graph-only doc-topology harvest: scan-routing regression.
//!
//! A `graph_only` docs source shares its `project` with the ingesting source
//! over the same tree, and `--source` filters by project — so a project-scoped
//! scan must NOT harvest the doc graph as a side-effect (full-scan only). This
//! pins that gate with two same-project sources.

use std::path::Path;
use std::sync::Arc;

use ostk_recall_cli::commands;
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::ThreadsDb;
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

fn write_config(path: &Path, corpus_root: &Path, other: &Path, docs: &Path) {
    // Two same-project ("haystack") sources: a plain ingesting markdown source
    // and a graph-only docs source. A `--source haystack` scan matches both.
    let body = format!(
        r#"[corpus]
root = '{corpus}'

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "markdown"
project = "haystack"
paths = ['{other}']

[[sources]]
kind = "markdown"
project = "haystack"
paths = ['{docs}']
entity_type = "doc"
graph_only = true
"#,
        corpus = corpus_root.display(),
        other = other.display(),
        docs = docs.display(),
    );
    std::fs::write(path, body).unwrap();
}

fn write_doc(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
}

#[tokio::test]
async fn scoped_scan_does_not_harvest_doc_graph_full_scan_does() {
    let corpus = TempDir::new().unwrap();
    let other = TempDir::new().unwrap();
    let docs = TempDir::new().unwrap();
    let cfg_dir = TempDir::new().unwrap();
    let cfg_path = cfg_dir.path().join("config.toml");

    write_doc(
        other.path(),
        "readme.md",
        "# Readme\nplain ingested markdown",
    );
    write_doc(docs.path(), "spec/a.md", "# Doc A\nlinks [b](b.md)");
    write_doc(docs.path(), "spec/b.md", "# Doc B\nleaf");

    write_config(&cfg_path, corpus.path(), other.path(), docs.path());

    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
    commands::init(&cfg_path, Arc::clone(&embedder))
        .await
        .unwrap();

    // Project-scoped scan: matches BOTH same-project sources. The graph-only
    // source must skip the harvest (and never ingest) → no doc nodes.
    commands::scan(&cfg_path, Arc::clone(&embedder), Some("haystack"), false)
        .await
        .unwrap();
    let db = ThreadsDb::open(corpus.path()).unwrap();
    assert!(
        db.get_concept("haystack", "spec-a").unwrap().is_none(),
        "scoped --source scan must NOT harvest the doc graph"
    );

    // Full scan (no filter): the graph-only source harvests.
    commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .unwrap();
    let db = ThreadsDb::open(corpus.path()).unwrap();
    let a = db
        .get_concept("haystack", "spec-a")
        .unwrap()
        .expect("full scan harvests the doc graph");
    assert_eq!(a.kind.as_deref(), Some("doc"));
    assert!(
        db.get_concept("haystack", "spec-b").unwrap().is_some(),
        "linked doc node also seeded"
    );
}
