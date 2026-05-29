//! P1 critical gate: re-running a scan with different facet overrides
//! flips `embedding_input_sha256` (forcing re-embed) iff the changed
//! facets are in the EMBED allowlist; otherwise the hash stays put and
//! Tier-2 dedupe skips embedding.

use std::path::Path;
use std::sync::Arc;

use ostk_recall_core::{Config, SourceConfig, SourceKind};
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline};
use ostk_recall_store::{CorpusStore, IngestDb};
use tempfile::TempDir;

const DIM: usize = 16;

/// Counts every embed batch so we can assert "did we actually re-embed?"
struct CountingEmbedder {
    calls: std::sync::atomic::AtomicUsize,
}

impl CountingEmbedder {
    fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    fn count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl ChunkEmbedder for CountingEmbedder {
    fn dim(&self) -> usize {
        DIM
    }
    #[allow(clippy::cast_precision_loss)]
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        self.calls
            .fetch_add(texts.len(), std::sync::atomic::Ordering::Relaxed);
        texts
            .iter()
            .map(|t| {
                let seed = ((t.len() % 100) as f32) * 0.01;
                (0..DIM).map(|i| (i as f32).mul_add(0.001, seed)).collect()
            })
            .collect()
    }
}

async fn make_pipeline(corpus_root: &Path) -> (Pipeline, Arc<CountingEmbedder>) {
    let store = Arc::new(CorpusStore::open_or_create(corpus_root, DIM).await.unwrap());
    let ingest = Arc::new(IngestDb::open(corpus_root).unwrap());
    let counter = Arc::new(CountingEmbedder::new());
    let emb: Arc<dyn ChunkEmbedder> = counter.clone();
    (Pipeline::new(store, ingest, emb), counter)
}

fn make_cfg(fixtures_dir: &Path, facets_override: &[(&str, &[&str])]) -> SourceConfig {
    let mut overrides = std::collections::BTreeMap::new();
    for (k, vs) in facets_override {
        overrides.insert(
            (*k).to_string(),
            vs.iter().map(|v| (*v).to_string()).collect(),
        );
    }
    let mut cfg = Config {
        corpus: ostk_recall_core::CorpusConfig {
            root: fixtures_dir.to_string_lossy().into_owned(),
        },
        embedder: ostk_recall_core::EmbedderConfig {
            model: "fake".into(),
        },
        sources: vec![SourceConfig {
            kind: SourceKind::Markdown,
            project: Some("notes".into()),
            paths: vec![fixtures_dir.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            id: None,
            source_config_id: String::new(),
            facets: overrides,
        }],
        reranker: None,
        watch: None,
        runtime: None,
        lens: None,
        record_rules: None,
        weaver: None,
    };
    cfg.validate_and_seal().expect("seal");
    cfg.sources.remove(0)
}

#[tokio::test]
async fn changing_allowlisted_facet_forces_reembed() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    std::fs::write(
        fixtures.path().join("a.md"),
        "# Alpha\n\nbody text that becomes a single chunk\n",
    )
    .unwrap();

    let scanner = ostk_recall_scan::markdown::MarkdownScanner;

    // Run 1: facets = {project: alpha}
    let (pipeline, counter) = make_pipeline(corpus.path()).await;
    let cfg1 = make_cfg(fixtures.path(), &[("project", &["alpha"])]);
    let s1 = pipeline.ingest_source(&scanner, &cfg1).await;
    let calls_after_run1 = counter.count();
    assert!(s1.chunks_upserted > 0);
    assert!(calls_after_run1 > 0, "first run must embed");

    // Run 2 (fresh pipeline + counter, same corpus): change project facet
    // (project IS in EMBED_FACET_ALLOWLIST → must re-embed).
    let (pipeline2, counter2) = make_pipeline(corpus.path()).await;
    let cfg2 = make_cfg(fixtures.path(), &[("project", &["renamed"])]);
    let s2 = pipeline2.ingest_source(&scanner, &cfg2).await;
    assert!(
        counter2.count() > 0,
        "allowlisted facet change must re-embed: counter={}, stats={:?}",
        counter2.count(),
        s2
    );
}

#[tokio::test]
async fn changing_non_allowlisted_facet_skips_reembed() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    std::fs::write(fixtures.path().join("a.md"), "# Alpha\n\nbody text\n").unwrap();

    let scanner = ostk_recall_scan::markdown::MarkdownScanner;

    // Run 1
    let (pipeline, counter) = make_pipeline(corpus.path()).await;
    let cfg1 = make_cfg(fixtures.path(), &[("session_id", &["session-1"])]);
    pipeline.ingest_source(&scanner, &cfg1).await;
    let _calls1 = counter.count();

    // Force a fresh metadata read by bumping mtime so Tier-1 doesn't
    // short-circuit on (mtime, size) match.
    std::fs::write(fixtures.path().join("a.md"), "# Alpha\n\nbody text\n").unwrap();
    // Run 2: change session_id (NOT in allowlist).
    let (pipeline2, counter2) = make_pipeline(corpus.path()).await;
    let cfg2 = make_cfg(fixtures.path(), &[("session_id", &["session-2"])]);
    pipeline2.ingest_source(&scanner, &cfg2).await;
    assert_eq!(
        counter2.count(),
        0,
        "non-allowlisted facet change must NOT re-embed"
    );
}
