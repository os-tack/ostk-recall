//! In-crate MCP tests. Exercises the `handle_request` path without
//! actually touching stdio.

use super::*;
use ostk_recall_core::{Chunk, Links, Source};
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_query::QueryEngine;
use ostk_recall_store::{CorpusStore, EventsDb, IngestChunkRow, IngestDb};
use serde_json::{Value, json};
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

async fn build_server() -> (TempDir, Server) {
    let dim = 8;
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
    store.ensure_fts_index().await.unwrap();
    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let events = Arc::new(EventsDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });

    // seed one chunk
    let chunk = Chunk {
        chunk_id: "seed".into(),
        source: Source::Markdown,
        project: Some("proj".into()),
        source_id: "seed.md".into(),
        chunk_index: 0,
        ts: None,
        role: None,
        text: "hello world alphabravo".into(),
        sha256: Chunk::content_hash("hello world alphabravo"),
        links: Links::default(),
        extra: Value::Null,
    };
    let vs = emb.encode_batch(&[chunk.text.as_str()]);
    store
        .upsert(std::slice::from_ref(&chunk), &vs)
        .await
        .unwrap();
    ingest
        .record_chunk(&IngestChunkRow {
            chunk_id: chunk.chunk_id.clone(),
            source: chunk.source.as_str().to_string(),
            source_id: chunk.source_id.clone(),
            chunk_index: chunk.chunk_index,
            content_sha256: chunk.sha256.clone(),
        })
        .unwrap();

    let engine = QueryEngine::new(store, ingest, Some(events), emb, "test");
    (tmp, Server::new(engine))
}

#[tokio::test]
async fn initialize_returns_server_info() {
    let (_tmp, server) = build_server().await;
    let req = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}});
    let resp = server.handle_request(req).await.unwrap();
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let r = resp.result.unwrap();
    assert_eq!(r["protocolVersion"], PROTOCOL_VERSION);
    assert_eq!(r["serverInfo"]["name"], "ostk-recall");
}

#[tokio::test]
async fn tools_list_includes_all_four() {
    let (_tmp, server) = build_server().await;
    let req = json!({"jsonrpc":"2.0","id":2,"method":"tools/list"});
    let resp = server.handle_request(req).await.unwrap();
    let r = resp.result.unwrap();
    let tools = r["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 4);
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"recall"));
    assert!(names.contains(&"recall_link"));
    assert!(names.contains(&"recall_stats"));
    assert!(names.contains(&"recall_audit"));
}

#[tokio::test]
async fn tools_list_without_events_skips_audit() {
    let dim = 8;
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });
    let engine = QueryEngine::new(store, ingest, None, emb, "test");
    let server = Server::new(engine);

    let req = json!({"jsonrpc":"2.0","id":3,"method":"tools/list"});
    let resp = server.handle_request(req).await.unwrap();
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    assert!(
        !tools
            .iter()
            .any(|t| t["name"].as_str() == Some("recall_audit"))
    );
}

#[tokio::test]
async fn tools_call_recall_stats_returns_content() {
    let (_tmp, server) = build_server().await;
    let req = json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": { "name": "recall_stats", "arguments": {} }
    });
    let resp = server.handle_request(req).await.unwrap();
    let r = resp.result.unwrap();
    let content = r["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    let inner: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(inner["total"], 1);
    assert_eq!(inner["dim"], 8);
}

#[tokio::test]
async fn tools_call_recall_returns_content() {
    let (_tmp, server) = build_server().await;
    let req = json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "recall",
            "arguments": { "query": "alphabravo", "limit": 5 }
        }
    });
    let resp = server.handle_request(req).await.unwrap();
    assert!(resp.error.is_none(), "{:?}", resp.error);
    let r = resp.result.unwrap();
    let text = r["content"][0]["text"].as_str().unwrap();
    let inner: Value = serde_json::from_str(text).unwrap();
    assert!(inner["hits"].is_array());
}

#[tokio::test]
async fn unknown_method_returns_error() {
    let (_tmp, server) = build_server().await;
    let req = json!({"jsonrpc":"2.0","id":6,"method":"bogus","params":{}});
    let resp = server.handle_request(req).await.unwrap();
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32601);
}

#[tokio::test]
async fn notification_initialized_returns_none() {
    let (_tmp, server) = build_server().await;
    let req = json!({"jsonrpc":"2.0","method":"notifications/initialized","params":{}});
    assert!(server.handle_request(req).await.is_none());
}

#[tokio::test]
async fn ping_returns_empty_result() {
    let (_tmp, server) = build_server().await;
    let req = json!({"jsonrpc":"2.0","id":7,"method":"ping"});
    let resp = server.handle_request(req).await.unwrap();
    assert!(resp.error.is_none());
    assert_eq!(resp.result.unwrap(), json!({}));
}

#[tokio::test]
async fn tools_call_recall_link_missing_arg() {
    let (_tmp, server) = build_server().await;
    let req = json!({
        "jsonrpc":"2.0","id":8,"method":"tools/call",
        "params": {"name":"recall_link","arguments":{}}
    });
    let resp = server.handle_request(req).await.unwrap();
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32602);
}
