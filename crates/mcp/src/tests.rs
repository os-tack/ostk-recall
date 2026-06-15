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
        facets: Default::default(),
        embedding_input_sha256: String::new(),
        source_config_id: "test-cfg".to_string(),
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
        .record_chunk(
            &IngestChunkRow {
                chunk_id: chunk.chunk_id.clone(),
                source: chunk.source.as_str().to_string(),
                source_config_id: "test-cfg".to_string(),
                source_id: chunk.source_id.clone(),
                chunk_index: chunk.chunk_index,
                content_sha256: chunk.sha256.clone(),
                embedding_input_sha256: chunk.embedding_input_sha256.clone(),
            },
            None,
        )
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
async fn tools_list_includes_all_five() {
    let (_tmp, server) = build_server().await;
    let req = json!({"jsonrpc":"2.0","id":2,"method":"tools/list"});
    let resp = server.handle_request(req).await.unwrap();
    let r = resp.result.unwrap();
    let tools = r["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 5);
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"recall"));
    assert!(names.contains(&"recall_link"));
    assert!(names.contains(&"recall_stats"));
    assert!(names.contains(&"recall_fault"));
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

// ---------------------------------------------------------------------
// Attention substrate wiring smoke tests
// ---------------------------------------------------------------------
//
// These tests prove that `Server::with_attention` exposes the attention
// dispatch surface end-to-end: tools/list includes the new tools, and
// tools/call routes through the dispatch correctly.

async fn build_server_with_attention() -> (TempDir, Server) {
    use ostk_recall_attention::{AttentionForwardStore, InMemoryAttention};
    use ostk_recall_attention_mcp::AttentionDispatch;
    use ostk_recall_store::ThreadsDb;

    let (tmp, server) = build_server().await;
    let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
    let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    let dispatch = Arc::new(AttentionDispatch::new(attention, threads));
    (tmp, server.with_attention(dispatch))
}

#[tokio::test]
async fn tools_list_includes_attention_when_wired() {
    let (_tmp, server) = build_server_with_attention().await;
    let req = json!({"jsonrpc":"2.0","id":100,"method":"tools/list"});
    let resp = server.handle_request(req).await.unwrap();
    let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
    let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"attention_attend"));
    assert!(names.contains(&"attention_surface"));
    assert!(names.contains(&"attention_fold"));
    assert!(names.contains(&"thread_create"));
    assert!(names.contains(&"thread_list"));
    // memory_* is the primary surface and must advertise alongside the
    // low-level admin tools when the attention substrate is wired.
    for m in [
        "memory_recall",
        "memory_surface",
        "memory_focus",
        "memory_remember",
        "memory_concept",
        "memory_connect",
        "memory_reflect",
    ] {
        assert!(names.contains(&m), "tools/list missing {m}");
    }
}

#[tokio::test]
async fn tools_call_thread_create_then_list_round_trip() {
    let (_tmp, server) = build_server_with_attention().await;

    let create_req = json!({
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "name": "thread_create",
            "arguments": { "handle": "smoke-thread", "tension": "active" }
        }
    });
    let create_resp = server.handle_request(create_req).await.unwrap();
    assert!(create_resp.error.is_none(), "{:?}", create_resp.error);

    let list_req = json!({
        "jsonrpc": "2.0",
        "id": 102,
        "method": "tools/call",
        "params": { "name": "thread_list", "arguments": {} }
    });
    let list_resp = server.handle_request(list_req).await.unwrap();
    assert!(list_resp.error.is_none(), "{:?}", list_resp.error);
    let text = list_resp.result.unwrap()["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    let inner: Value = serde_json::from_str(&text).unwrap();
    let handles: Vec<&str> = inner["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["handle"].as_str().unwrap())
        .collect();
    assert!(
        handles.contains(&"smoke-thread"),
        "thread_create result should be visible via thread_list: {handles:?}"
    );
}

#[tokio::test]
async fn attention_tools_unavailable_without_dispatch() {
    let (_tmp, server) = build_server().await;
    let req = json!({
        "jsonrpc": "2.0",
        "id": 103,
        "method": "tools/call",
        "params": { "name": "thread_create", "arguments": { "handle": "x" } }
    });
    let resp = server.handle_request(req).await.unwrap();
    assert!(resp.error.is_some(), "expected method-not-found");
    assert_eq!(resp.error.unwrap().code, -32601);
}

/// →1957 leg 3 helpers: build a server whose config declares one watched
/// `ostk_project` source rooted at `proj_root`, with one ingested audit
/// row at `row_ts`. Each call builds a fresh server (the stale probe
/// caches for 30s, so cases must not share one).
async fn build_server_with_watch(
    proj_root: &std::path::Path,
    row_ts: &str,
) -> (TempDir, Server) {
    let (tmp, server) = build_server().await;
    // Drop the plain server; rebuild with config + a seeded events row.
    drop(server);
    let dim = 8;
    let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let events = Arc::new(EventsDb::open(tmp.path()).unwrap());
    events
        .ingest_batch(&[ostk_recall_store::AuditEventRow {
            row_key: format!("p1:{row_ts}"),
            project: Some("p1".into()),
            ts: Some(
                chrono::DateTime::parse_from_rfc3339(row_ts)
                    .unwrap()
                    .with_timezone(&chrono::Utc),
            ),
            event: Some("tool.bash".into()),
            tool: None,
            agent: None,
            success: Some(true),
            exit_code: None,
            duration_ms: None,
            raw: "{}".into(),
        }])
        .unwrap();
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });
    let engine = QueryEngine::new(store, ingest, Some(events), emb, "test");
    let cfg: ostk_recall_core::Config = serde_json::from_value(json!({
        "corpus": { "root": tmp.path().to_str().unwrap() },
        "embedder": { "model": "test" },
        "sources": [{
            "kind": "ostk_project",
            "project": "p1",
            "paths": [proj_root.to_str().unwrap()],
        }],
        "watch": { "enabled": true, "freshness_threshold_secs": 60 },
    }))
    .unwrap();
    let server = Server::new(engine).with_config(Arc::new(cfg));
    (tmp, server)
}

async fn stats_response_body(server: &Server) -> Value {
    let req = json!({
        "jsonrpc": "2.0",
        "id": 200,
        "method": "tools/call",
        "params": { "name": "recall_stats", "arguments": {} }
    });
    let resp = server.handle_request(req).await.unwrap();
    let r = resp.result.unwrap();
    serde_json::from_str(r["content"][0]["text"].as_str().unwrap()).unwrap()
}

/// →1957: fresh journal + old ingested row = divergence → the envelope
/// carries `stale_ingest` naming the project.
#[tokio::test]
async fn stale_ingest_flags_journal_vs_ingest_divergence() {
    let proj = TempDir::new().unwrap();
    std::fs::create_dir_all(proj.path().join(".ostk")).unwrap();
    // Journal written NOW; newest ingested row months old.
    std::fs::write(proj.path().join(".ostk/journal.jsonl"), "{}\n").unwrap();
    let (_tmp, server) = build_server_with_watch(proj.path(), "2026-01-01T00:00:00+00:00").await;

    let body = stats_response_body(&server).await;
    let stale = body
        .get("stale_ingest")
        .expect("divergent project must flag stale_ingest");
    assert_eq!(stale["projects"][0]["project"], "p1");
    assert_eq!(stale["threshold_secs"], 60);
}

/// →1957 leg 2: the bridge's heal receipt must be accepted as a
/// notification (None response), logged, and appended durably to
/// `<corpus_root>/bridge_events.jsonl`.
#[tokio::test]
async fn bridge_reconnected_receipt_appends_durable_row() {
    let proj = TempDir::new().unwrap();
    std::fs::create_dir_all(proj.path().join(".ostk")).unwrap();
    std::fs::write(proj.path().join(".ostk/journal.jsonl"), "{}\n").unwrap();
    let now = chrono::Utc::now().to_rfc3339();
    let (tmp, server) = build_server_with_watch(proj.path(), &now).await;

    let req = json!({
        "jsonrpc": "2.0",
        "method": "notifications/bridge_reconnected",
        "params": { "attempts": 3, "elapsed_ms": 1200, "bridge_pid": 42 }
    });
    let resp = server.handle_request(req).await;
    assert!(resp.is_none(), "notification must get no response");

    let raw = std::fs::read_to_string(tmp.path().join("bridge_events.jsonl"))
        .expect("durable receipt row written");
    let row: Value = serde_json::from_str(raw.lines().next().unwrap()).unwrap();
    assert_eq!(row["event"], "bridge.reconnected");
    assert_eq!(row["params"]["attempts"], 3);
    assert_eq!(row["params"]["bridge_pid"], 42);
    assert!(row["ts"].as_str().is_some());
}

/// →1957: ingest keeping pace with the journal (row ts ≈ mtime ≈ now)
/// stays silent — the conditional push only fires on failure. (The pull
/// side, `audit_newest_ts` in stats, stays unconditional — that pairing
/// is the invariant.)
#[tokio::test]
async fn stale_ingest_silent_when_ingest_keeps_pace() {
    let proj = TempDir::new().unwrap();
    std::fs::create_dir_all(proj.path().join(".ostk")).unwrap();
    std::fs::write(proj.path().join(".ostk/journal.jsonl"), "{}\n").unwrap();
    let now = chrono::Utc::now().to_rfc3339();
    let (_tmp, server) = build_server_with_watch(proj.path(), &now).await;

    let body = stats_response_body(&server).await;
    assert!(
        body.get("stale_ingest").is_none(),
        "healthy ingest must not flag: {body}"
    );
    // Pull path stays unconditionally truthful.
    assert!(body.get("audit_newest_ts").is_some());
}

// Autonomous-salience axis 4 (self-audit): the salience-health PULL leg on
// `recall_stats` is present (and decomposed) when an attention surface is
// wired, and absent — degrade-to-omission — when it is not. Mirrors the
// `audit_newest_ts` pull-leg contract.
#[tokio::test]
async fn salience_health_pull_leg_present_with_attention() {
    let (_tmp, server) = build_server_with_attention().await;
    let body = stats_response_body(&server).await;
    let health = body
        .get("salience_health")
        .expect("pull leg present when attention is wired");
    // Decomposed block — every metric field is carried (R4
    // `abi-as-sovereign-boundary`), even on an empty fresh surface.
    for field in [
        "surface_score_spread",
        "curated_ratio",
        "curated_load_bearing",
        "never_used",
        "unattributable",
        "drift_forgotten",
        "thresholds",
        "unhealthy",
    ] {
        assert!(
            health.get(field).is_some(),
            "salience_health missing `{field}`: {health}"
        );
    }
    // A fresh, empty surface has no drift / collapse / never-used ⇒ healthy,
    // so the conditional PUSH leg must NOT fire on this same response.
    assert_eq!(health["unhealthy"], serde_json::json!(false));
    assert_eq!(
        health["thresholds"]["window_days"],
        serde_json::json!(14),
        "shipped [salience.health] default window"
    );
}

#[tokio::test]
async fn salience_health_absent_without_attention() {
    // No attention dispatch wired (the `--stdio` / embedded path) ⇒ the pull
    // field is omitted entirely so old MCP clients keep parsing, the same
    // degrade contract `audit_newest_ts` uses with no events DB.
    let (_tmp, server) = build_server().await;
    let body = stats_response_body(&server).await;
    assert!(
        body.get("salience_health").is_none(),
        "no attention surface ⇒ no salience_health pull field: {body}"
    );
}
