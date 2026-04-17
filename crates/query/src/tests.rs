//! In-crate tests for `QueryEngine`.

use super::*;
use chrono::{DateTime, Utc};
use ostk_recall_core::{Chunk, Links, Source};
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::IngestChunkRow;
use std::sync::Arc;
use tempfile::TempDir;

struct FakeEmbedder {
    dim: usize,
}

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0; self.dim];
                let bucket = t.len() % self.dim;
                v[bucket] = 1.0;
                v
            })
            .collect()
    }
}

fn chunk(id: &str, text: &str, project: Option<&str>) -> Chunk {
    Chunk {
        chunk_id: id.to_string(),
        source: Source::Markdown,
        project: project.map(ToString::to_string),
        source_id: format!("{id}.md"),
        chunk_index: 0,
        ts: Some(DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap()),
        role: None,
        text: text.to_string(),
        sha256: Chunk::content_hash(text),
        links: Links::default(),
        extra: serde_json::Value::Null,
    }
}

async fn build_engine(dim: usize) -> (TempDir, QueryEngine) {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
    store.ensure_fts_index().await.unwrap();
    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let events = Arc::new(EventsDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });
    let engine = QueryEngine::new(store, ingest, Some(events), emb, "test-model");
    (tmp, engine)
}

fn record(engine: &QueryEngine, chunks: &[Chunk]) {
    for c in chunks {
        engine
            .ingest
            .record_chunk(&IngestChunkRow {
                chunk_id: c.chunk_id.clone(),
                source: c.source.as_str().to_string(),
                source_id: c.source_id.clone(),
                chunk_index: c.chunk_index,
                content_sha256: c.sha256.clone(),
            })
            .unwrap();
    }
}

#[tokio::test]
async fn recall_returns_matching_chunk() {
    let dim = 8;
    let (_tmp, engine) = build_engine(dim).await;

    let texts = [
        ("a", "alpha bravo charlie quickfox"),
        ("b", "delta echo foxtrot"),
        ("c", "quickfox jumped over the lazy dog"),
    ];
    let chunks: Vec<Chunk> = texts
        .iter()
        .map(|(id, t)| chunk(id, t, Some("proj")))
        .collect();
    let emb_vectors = engine
        .embedder
        .encode_batch(&texts.iter().map(|(_, t)| *t).collect::<Vec<_>>());
    engine.store.upsert(&chunks, &emb_vectors).await.unwrap();
    record(&engine, &chunks);

    let hits = engine
        .recall(RecallParams {
            query: "quickfox".into(),
            limit: Some(5),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(!hits.is_empty(), "expected at least one hit for 'quickfox'");
    let top_ids: Vec<_> = hits.iter().map(|h| h.chunk_id.as_str()).collect();
    assert!(
        top_ids.contains(&"a") || top_ids.contains(&"c"),
        "expected a or c in top hits, got {top_ids:?}"
    );
}

#[tokio::test]
async fn recall_filter_by_project() {
    let dim = 8;
    let (_tmp, engine) = build_engine(dim).await;
    let chunks = vec![
        chunk("x1", "alpha beta", Some("p1")),
        chunk("x2", "alpha beta gamma", Some("p2")),
    ];
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vs = engine.embedder.encode_batch(&texts);
    engine.store.upsert(&chunks, &vs).await.unwrap();

    let hits = engine
        .recall(RecallParams {
            query: "alpha".into(),
            project: Some("p1".into()),
            limit: Some(5),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(hits.iter().all(|h| h.project.as_deref() == Some("p1")));
}

#[tokio::test]
async fn recall_link_chases_parents() {
    let dim = 4;
    let (_tmp, engine) = build_engine(dim).await;

    let parent = chunk("parent", "parent text", Some("p"));
    let mut child = chunk("child", "child text", Some("p"));
    child.links.parent_ids.push("parent".into());
    let chunks = vec![parent, child];
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vs = engine.embedder.encode_batch(&texts);
    engine.store.upsert(&chunks, &vs).await.unwrap();

    let out = engine.recall_link("child").await.unwrap();
    assert_eq!(out.chunk.chunk_id, "child");
    assert_eq!(out.parents.len(), 1);
    assert_eq!(out.parents[0].chunk_id, "parent");
}

#[tokio::test]
async fn recall_stats_reports_counts() {
    let dim = 4;
    let (_tmp, engine) = build_engine(dim).await;

    let chunks = vec![
        chunk("a", "text a", Some("p")),
        chunk("b", "text b", Some("p")),
    ];
    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vs = engine.embedder.encode_batch(&texts);
    engine.store.upsert(&chunks, &vs).await.unwrap();
    record(&engine, &chunks);

    let stats = engine.recall_stats().await.unwrap();
    assert_eq!(stats.total, 2);
    assert_eq!(stats.dim, dim);
    assert_eq!(stats.model, "test-model");
    assert_eq!(stats.by_source.iter().map(|c| c.count).sum::<u64>(), 2);
    assert!(stats.last_scan_at.is_some());
}

#[tokio::test]
async fn recall_audit_select_allowed() {
    let dim = 4;
    let (_tmp, engine) = build_engine(dim).await;

    let out = engine.recall_audit("SELECT 1 AS one").unwrap();
    assert_eq!(out.columns, vec!["one".to_string()]);
    assert_eq!(out.rows.len(), 1);
    assert_eq!(out.rows[0][0].as_i64(), Some(1));
}

#[tokio::test]
async fn recall_audit_non_select_rejected() {
    let dim = 4;
    let (_tmp, engine) = build_engine(dim).await;
    let err = engine.recall_audit("DELETE FROM audit_events").unwrap_err();
    assert!(matches!(err, QueryError::Forbidden(_)), "{err:?}");
}

#[tokio::test]
async fn recall_audit_requires_events_store() {
    let dim = 4;
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });
    let engine = QueryEngine::new(store, ingest, None, emb, "no-events");
    let err = engine.recall_audit("SELECT 1").unwrap_err();
    assert!(matches!(err, QueryError::NoEventsStore), "{err:?}");
}
