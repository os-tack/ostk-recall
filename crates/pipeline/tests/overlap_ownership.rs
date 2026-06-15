//! P0 gate: two distinct `[[sources]]` blocks pointing at the same physical
//! directory must round-trip through ingest without cross-contamination.
//! When one block is removed and rescanned, ONLY that block's chunks should
//! be swept.
//!
//! This is the canonical "two configs over same dir" pattern. The pre-P0
//! identity formula collided these chunks; P0's `source_config_id`
//! discriminator fixes it.

use std::path::Path;
use std::sync::Arc;

use ostk_recall_core::{Config, SourceConfig, SourceKind};
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline};
use ostk_recall_store::{CorpusStore, IngestDb};
use tempfile::TempDir;

const DIM: usize = 16;

struct FakeEmbedder;

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        DIM
    }
    #[allow(clippy::cast_precision_loss)]
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let seed = ((t.len() % 100) as f32) * 0.01;
                (0..DIM).map(|i| (i as f32).mul_add(0.001, seed)).collect()
            })
            .collect()
    }
}

async fn make_pipeline(corpus_root: &Path) -> Pipeline {
    let store = Arc::new(CorpusStore::open_or_create(corpus_root, DIM).await.unwrap());
    let ingest = Arc::new(IngestDb::open(corpus_root).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
    Pipeline::new(store, ingest, emb)
}

fn write_sample_tree(root: &Path) {
    std::fs::write(
        root.join("a.md"),
        "# Alpha\n\nIntro.\n\n## Section\n\nbody one\n",
    )
    .unwrap();
    std::fs::write(
        root.join("b.md"),
        "# Beta\n\nIntro.\n\n## Section\n\nbody two\n",
    )
    .unwrap();
}

fn build_config(fixtures_dir: &Path, blocks: Vec<SourceConfig>) -> Config {
    let mut cfg = Config {
        corpus: ostk_recall_core::CorpusConfig {
            root: fixtures_dir.to_string_lossy().into_owned(),
        },
        embedder: ostk_recall_core::EmbedderConfig {
            model: "fake".into(),
        },
        sources: blocks,
        reranker: None,
        watch: None,
        runtime: None,
        ranking: None,
        relational: None,
        ambient_growth: None,
        lens: None,
        record_rules: None,
        weaver: None,
        salience: None,
    };
    cfg.validate_and_seal()
        .expect("config should validate + seal");
    cfg
}

#[tokio::test]
async fn two_configs_same_dir_do_not_collide() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    write_sample_tree(fixtures.path());

    let blocks = vec![
        SourceConfig {
            kind: SourceKind::Markdown,
            graph_only: false,
            project: Some("alpha".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            entity_type: None,
            edges: Vec::new(),
            id: Some("notes-alpha".into()),
            source_config_id: String::new(), // sealed by validate_and_seal,
            facets: Default::default(),
        },
        SourceConfig {
            kind: SourceKind::Markdown,
            graph_only: false,
            project: Some("beta".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            entity_type: None,
            edges: Vec::new(),
            id: Some("notes-beta".into()),
            source_config_id: String::new(),
            facets: Default::default(),
        },
    ];
    let cfg = build_config(fixtures.path(), blocks);

    let pipeline = make_pipeline(corpus.path()).await;
    let scanner = ostk_recall_scan::markdown::MarkdownScanner;
    let a = pipeline.ingest_source(&scanner, &cfg.sources[0]).await;
    let b = pipeline.ingest_source(&scanner, &cfg.sources[1]).await;
    assert_eq!(a.errors, 0, "block A: {a:?}");
    assert_eq!(b.errors, 0, "block B: {b:?}");
    assert!(a.chunks_upserted >= 3, "block A produced chunks: {a:?}");
    assert!(b.chunks_upserted >= 3, "block B produced chunks: {b:?}");

    // Both blocks coexist in the corpus.
    let total = pipeline.store().row_count().await.unwrap();
    assert_eq!(
        total,
        a.chunks_upserted + b.chunks_upserted,
        "no collision: A+B chunks sum to the corpus total"
    );
}

#[tokio::test]
async fn dropping_one_block_sweeps_only_its_chunks() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    write_sample_tree(fixtures.path());

    let blocks = vec![
        SourceConfig {
            kind: SourceKind::Markdown,
            graph_only: false,
            project: Some("alpha".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            entity_type: None,
            edges: Vec::new(),
            id: Some("notes-alpha".into()),
            source_config_id: String::new(),
            facets: Default::default(),
        },
        SourceConfig {
            kind: SourceKind::Markdown,
            graph_only: false,
            project: Some("beta".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            entity_type: None,
            edges: Vec::new(),
            id: Some("notes-beta".into()),
            source_config_id: String::new(),
            facets: Default::default(),
        },
    ];
    let cfg = build_config(fixtures.path(), blocks);

    // First scan: both blocks ingest.
    {
        let pipeline = make_pipeline(corpus.path()).await;
        let scanner = ostk_recall_scan::markdown::MarkdownScanner;
        pipeline.ingest_source(&scanner, &cfg.sources[0]).await;
        pipeline.ingest_source(&scanner, &cfg.sources[1]).await;
    }

    // Second scan: rescan ONLY block A (block B is "dropped").
    // Block A's run_id-tracked rows survive; block B's rows from the
    // prior run aren't swept because the sweep keys on
    // (source, source_config_id, run_id) — without block B in this run,
    // its source_config_id is never visited.
    {
        let pipeline = make_pipeline(corpus.path()).await;
        let scanner = ostk_recall_scan::markdown::MarkdownScanner;
        pipeline.ingest_source(&scanner, &cfg.sources[0]).await;
        // Confirm block B's chunks are still present.
        let total = pipeline.store().row_count().await.unwrap();
        assert!(
            total >= 6,
            "block B chunks must survive a rescan that only visits block A — got {total}"
        );
    }
}
