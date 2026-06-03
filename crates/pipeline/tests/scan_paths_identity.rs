//! P0 gate: `Pipeline::scan_paths` (the incremental scan trigger) must
//! preserve the same identity that `ingest_source` produces. A path-aware
//! rescan after a content edit must reuse the same chunk_id, so dedupe
//! detects "content changed" instead of "new chunk".

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

fn make_cfg(fixtures_dir: &Path) -> SourceConfig {
    let mut cfg = Config {
        corpus: ostk_recall_core::CorpusConfig {
            root: fixtures_dir.to_string_lossy().into_owned(),
        },
        embedder: ostk_recall_core::EmbedderConfig {
            model: "fake".into(),
        },
        sources: vec![SourceConfig {
            kind: SourceKind::Markdown,
            graph_only: false,
            project: Some("notes".into()),
            paths: vec![fixtures_dir.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            entity_type: None,
            edges: Vec::new(),
            id: None,
            source_config_id: String::new(),
            facets: Default::default(),
        }],
        reranker: None,
        watch: None,
        runtime: None,
        ranking: None,
        relational: None,
        ambient_growth: None,
        lens: None,
        record_rules: None,
        weaver: None,
    };
    cfg.validate_and_seal().expect("seal");
    cfg.sources.remove(0)
}

#[tokio::test]
async fn scan_paths_preserves_chunk_id_across_edit() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    std::fs::write(
        fixtures.path().join("a.md"),
        "# Alpha\n\nIntro.\n\n## Section\n\nbody one\n",
    )
    .unwrap();
    let cfg = make_cfg(fixtures.path());

    // First: full ingest_source produces N chunks with chunk_ids X.
    let scanner = ostk_recall_scan::markdown::MarkdownScanner;
    let pipeline = make_pipeline(corpus.path()).await;
    let s1 = pipeline.ingest_source(&scanner, &cfg).await;
    assert_eq!(s1.errors, 0);
    let upserted_initial = s1.chunks_upserted;
    assert!(upserted_initial >= 1);

    // Edit the file's content (forces re-embed via content_sha256 change).
    std::fs::write(
        fixtures.path().join("a.md"),
        "# Alpha\n\nIntro EDIT.\n\n## Section\n\nbody one CHANGED\n",
    )
    .unwrap();

    // Path-targeted rescan via scan_paths. The chunk_ids must match the
    // first run — the formula keys on (source, source_id, chunk_index,
    // source_config_id), none of which changed.
    let paths = vec![fixtures.path().join("a.md")];
    let per = pipeline
        .scan_paths(&[(&scanner, &cfg)], &paths)
        .await
        .expect("scan_paths");
    assert_eq!(per.len(), 1, "one source matched");
    let s2 = per[0].1;
    // Re-embed should happen (content changed); same number of chunks
    // because the file structure is similar. chunks_upserted > 0 confirms
    // the merge_insert path ran.
    assert!(
        s2.chunks_upserted > 0 || s2.chunks_skipped_dup == upserted_initial,
        "either re-embedded or detected unchanged: {s2:?}"
    );

    // The corpus total should equal the initial upsert (merge_insert on
    // same chunk_ids leaves the row count unchanged).
    let total = pipeline.store().row_count().await.unwrap();
    assert_eq!(
        total, upserted_initial,
        "scan_paths reuses chunk_ids (no new rows): total={total}, initial={upserted_initial}"
    );
}
