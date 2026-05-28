//! P0 gate: `rebuild_ingest_manifest` reconstructs `ingest.sqlite` from a
//! `corpus.lance` directory alone. Used by `ostk-recall manifest rebuild`
//! when an ingest ledger is lost or corrupt.

use ostk_recall_core::{Chunk, Links, Source};
use ostk_recall_store::{
    CorpusStore, IngestDb, manifest::rebuild_ingest_manifest,
};
use tempfile::TempDir;

#[tokio::test]
async fn rebuilds_ledger_from_corpus_alone() {
    let tmp = TempDir::new().unwrap();
    let store = CorpusStore::open_or_create(tmp.path(), 4).await.unwrap();

    // Seed three chunks across two source configs.
    let chunks = vec![
        Chunk {
            chunk_id: "a-0".into(),
            source: Source::Markdown,
            project: Some("alpha".into()),
            source_id: "alpha/a.md".into(),
            source_config_id: "cfg-alpha".into(),
            chunk_index: 0,
            ts: None,
            role: None,
            text: "alpha 0".into(),
            sha256: Chunk::content_hash("alpha 0"),
            links: Links::default(),
            extra: serde_json::Value::Null,
        },
        Chunk {
            chunk_id: "a-1".into(),
            source: Source::Markdown,
            project: Some("alpha".into()),
            source_id: "alpha/a.md".into(),
            source_config_id: "cfg-alpha".into(),
            chunk_index: 1,
            ts: None,
            role: None,
            text: "alpha 1".into(),
            sha256: Chunk::content_hash("alpha 1"),
            links: Links::default(),
            extra: serde_json::Value::Null,
        },
        Chunk {
            chunk_id: "b-0".into(),
            source: Source::Markdown,
            project: Some("beta".into()),
            source_id: "beta/b.md".into(),
            source_config_id: "cfg-beta".into(),
            chunk_index: 0,
            ts: None,
            role: None,
            text: "beta 0".into(),
            sha256: Chunk::content_hash("beta 0"),
            links: Links::default(),
            extra: serde_json::Value::Null,
        },
    ];
    let embs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.5, 0.5, 0.5, 0.5],
    ];
    store.upsert(&chunks, &embs).await.unwrap();
    assert_eq!(store.row_count().await.unwrap(), 3);

    // Pretend the ledger was lost — open a fresh, empty IngestDb at a
    // separate path.
    let ingest_tmp = TempDir::new().unwrap();
    let ingest = IngestDb::open(ingest_tmp.path()).unwrap();
    assert!(ingest.count_by_source().unwrap().is_empty());

    let written = rebuild_ingest_manifest(&store, &ingest, "rebuild-run-1")
        .await
        .unwrap();
    assert_eq!(written, 3, "all three corpus rows reproduced");

    // Verify dedupe ledger sees the rebuilt chunks.
    for chunk in &chunks {
        assert!(
            ingest
                .content_already_ingested(&chunk.chunk_id, &chunk.sha256)
                .unwrap(),
            "rebuilt ledger must recognize chunk {}",
            chunk.chunk_id
        );
    }

    // Source metadata reconstructed too (per (source, source_config_id,
    // source_id) tuple — verify one entry per source_id).
    assert!(
        ingest
            .get_source_metadata("markdown", "cfg-alpha", "alpha/a.md")
            .unwrap()
            .is_some(),
        "alpha source row exists"
    );
    assert!(
        ingest
            .get_source_metadata("markdown", "cfg-beta", "beta/b.md")
            .unwrap()
            .is_some(),
        "beta source row exists"
    );
}
