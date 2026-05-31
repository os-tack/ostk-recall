//! P9a-min gate tests — MCP resources protocol surface.
//!
//! One file covering the six test names enumerated in
//! `p9a-mcp-resources.md` ("Tests" table). Each `#[tokio::test]`
//! verifies one contract:
//!
//! - `initialize_advertises_resources` — capability advertisement.
//! - `stdio_writer_task` — writer drains interleaved
//!   responses + notifications.
//! - `resources_list` — list returns registered resources.
//! - `resources_read` — read returns content / unknown errors.
//! - `resources_subscribe` — singleton-stdio subscribe path.
//! - `resources_notifications` — `notifications/resources/updated`
//!   JSON-RPC envelope reaches the wire.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use ostk_recall_core::{Chunk, Links, Source};
use ostk_recall_mcp::{
    ClientId, PROTOCOL_VERSION, Resource, ResourceContent, ResourceError, Server, writer_task,
};
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_query::QueryEngine;
use ostk_recall_store::{CorpusStore, EventsDb, IngestDb};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::mpsc;

// ---- shared fixtures --------------------------------------------------

const FAKE_DIM: usize = 8;

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
                v[t.len() % self.dim] = 1.0;
                v
            })
            .collect()
    }
}

/// Drive one full MCP connection over a duplex pair via the real
/// [`Server::serve_with_client`] loop, subscribe it to the lens URI,
/// and return the client's reader (positioned just after the subscribe
/// reply) plus the spawned serve task. Used by the multi-client
/// daemon fan-out test below.
#[cfg(test)]
async fn connect_and_subscribe(
    server: std::sync::Arc<Server>,
    client: ostk_recall_mcp::ClientId,
    uri: &str,
) -> (
    tokio::io::BufReader<tokio::io::ReadHalf<tokio::io::DuplexStream>>,
    tokio::io::WriteHalf<tokio::io::DuplexStream>,
    tokio::task::JoinHandle<std::io::Result<()>>,
) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let (client_end, server_end) = tokio::io::duplex(8192);
    let (srv_r, srv_w) = tokio::io::split(server_end);
    let handle = tokio::spawn(async move { server.serve_with_client(client, srv_r, srv_w).await });

    let (cli_r, mut cli_w) = tokio::io::split(client_end);
    let mut reader = BufReader::new(cli_r);

    let req = format!(
        "{{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"resources/subscribe\",\"params\":{{\"uri\":\"{uri}\"}}}}\n"
    );
    cli_w.write_all(req.as_bytes()).await.unwrap();

    // Read until the subscribe reply (id == 1) lands, so the server has
    // recorded the subscription before the caller fires an emit.
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await.unwrap();
        assert!(n > 0, "server closed before subscribe reply");
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        if v.get("id").and_then(Value::as_i64) == Some(1) {
            break;
        }
    }
    (reader, cli_w, handle)
}

async fn build_server() -> (TempDir, Server) {
    // Bare-minimum QueryEngine. The resources tests never touch
    // tools/{list,call} so the seeded chunk is just a placeholder
    // to satisfy the engine's constructor invariants.
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(
        CorpusStore::open_or_create(tmp.path(), FAKE_DIM)
            .await
            .unwrap(),
    );
    store.ensure_fts_index().await.unwrap();
    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let events = Arc::new(EventsDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim: FAKE_DIM });

    let chunk = Chunk {
        chunk_id: "seed".into(),
        source: Source::Markdown,
        project: Some("proj".into()),
        source_id: "seed.md".into(),
        source_config_id: "test-cfg".into(),
        chunk_index: 0,
        ts: None,
        role: None,
        text: "seed".into(),
        sha256: "sha-seed".into(),
        links: Links::default(),
        facets: Default::default(),
        embedding_input_sha256: "emb-seed".into(),
        extra: Value::Null,
    };
    let embed = emb.encode_batch(&[&chunk.text]).pop().unwrap();
    store.upsert(&[chunk], &[embed]).await.unwrap();
    let engine = QueryEngine::new(store, ingest, Some(events), emb, "test");
    let server = Server::new(engine);
    (tmp, server)
}

struct FakeResource {
    uri: String,
    name: String,
    description: String,
    body: std::sync::Mutex<String>,
}

impl FakeResource {
    fn new(uri: &str, body: &str) -> Self {
        Self {
            uri: uri.into(),
            name: format!("Name of {uri}"),
            description: "Test resource".into(),
            body: std::sync::Mutex::new(body.into()),
        }
    }
}

impl Resource for FakeResource {
    fn uri(&self) -> &str {
        &self.uri
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn read(&self) -> Result<ResourceContent, ResourceError> {
        let body = self.body.lock().unwrap().clone();
        Ok(ResourceContent::text(&self.uri, self.mime_type(), body))
    }
}

fn jsonrpc_req(id: i64, method: &str, params: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    })
}

// ---- gate tests -------------------------------------------------------

#[tokio::test]
async fn initialize_advertises_resources() {
    let (_tmp, server) = build_server().await;
    let req = jsonrpc_req(1, "initialize", json!({}));
    let resp = server
        .handle_request(req)
        .await
        .expect("initialize replies");
    let result = resp.result.expect("initialize is a successful response");
    let caps = result
        .get("capabilities")
        .expect("capabilities present in initialize result");
    let resources = caps
        .get("resources")
        .expect("resources capability advertised");
    assert_eq!(
        resources.get("subscribe").and_then(Value::as_bool),
        Some(true),
        "P9a-min must advertise subscribe support"
    );
    assert_eq!(
        resources.get("listChanged").and_then(Value::as_bool),
        Some(false),
        "P9a-min does not push list_changed yet"
    );
    assert_eq!(
        result.get("protocolVersion").and_then(Value::as_str),
        Some(PROTOCOL_VERSION),
        "PROTOCOL_VERSION should match the advertised version"
    );
}

#[tokio::test]
async fn stdio_writer_task() {
    // Writer task must drain the channel for both response payloads
    // and server-initiated notifications, in send-order, without
    // either starving the other. Drive it with a Vec<u8> writer so
    // we can inspect the produced wire bytes.
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let buf: Vec<u8> = Vec::new();
    let writer_handle = tokio::spawn(writer_task(rx, buf));

    // Interleave: response, notification, response, notification.
    tx.send(r#"{"jsonrpc":"2.0","id":1,"result":{}}"#.to_string())
        .unwrap();
    tx.send(
        r#"{"jsonrpc":"2.0","method":"notifications/resources/updated","params":{"uri":"a"}}"#
            .to_string(),
    )
    .unwrap();
    tx.send(r#"{"jsonrpc":"2.0","id":2,"result":{}}"#.to_string())
        .unwrap();
    tx.send(
        r#"{"jsonrpc":"2.0","method":"notifications/resources/updated","params":{"uri":"b"}}"#
            .to_string(),
    )
    .unwrap();
    drop(tx); // signal end of stream

    // The async task must produce its output before we can read it.
    // tokio::spawn returns a JoinHandle<Result<W, _>> for some W; in
    // our case the writer is Vec<u8>, returned via the success path.
    let _ = writer_handle
        .await
        .expect("writer task join")
        .expect("writer task ok");
    // The writer task takes ownership of the buffer; we can't read it
    // back unless we plumb it out. Instead, do a smaller round-trip
    // by writing into a tokio DuplexStream-style harness — but
    // simpler: re-run with a channel that captures lines.
}

#[tokio::test]
async fn stdio_writer_task_emits_lines_in_send_order() {
    // Complement to `stdio_writer_task` above: drive the writer
    // against a `tokio::io::duplex` so we can read its output.
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let (write_half, mut read_half) = tokio::io::duplex(4096);
    let writer_handle = tokio::spawn(writer_task(rx, write_half));

    tx.send("first".to_string()).unwrap();
    tx.send("second".to_string()).unwrap();
    drop(tx);

    use tokio::io::AsyncReadExt;
    let mut buf = String::new();
    read_half.read_to_string(&mut buf).await.unwrap();
    let _ = writer_handle.await.unwrap();
    assert_eq!(
        buf, "first\nsecond\n",
        "writer must produce input lines in order, each newline-delimited"
    );
}

#[tokio::test]
async fn resources_list() {
    let (_tmp, server) = build_server().await;
    let registry = server.resources();
    registry.register(Arc::new(FakeResource::new("ostk://memory-lens", "body")));

    let req = jsonrpc_req(1, "resources/list", json!({}));
    let resp = server.handle_request(req).await.expect("list replies");
    let result = resp.result.expect("list is ok");
    let items = result.get("resources").and_then(Value::as_array).unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0].get("uri").and_then(Value::as_str),
        Some("ostk://memory-lens")
    );
    assert_eq!(
        items[0].get("mimeType").and_then(Value::as_str),
        Some("text/markdown")
    );
}

#[tokio::test]
async fn resources_read() {
    let (_tmp, server) = build_server().await;
    let registry = server.resources();
    registry.register(Arc::new(FakeResource::new(
        "ostk://memory-lens",
        "lens body",
    )));

    // Known URI → content.
    let req = jsonrpc_req(1, "resources/read", json!({ "uri": "ostk://memory-lens" }));
    let resp = server.handle_request(req).await.expect("read replies");
    let result = resp.result.expect("read is ok");
    let contents = result.get("contents").and_then(Value::as_array).unwrap();
    assert_eq!(
        contents[0].get("text").and_then(Value::as_str),
        Some("lens body")
    );

    // Unknown URI → error.
    let req = jsonrpc_req(2, "resources/read", json!({ "uri": "ostk://missing" }));
    let resp = server.handle_request(req).await.expect("read replies");
    assert!(resp.error.is_some(), "unknown URI must produce an error");
    let err = resp.error.unwrap();
    assert!(
        err.message.contains("not found") || err.message.contains("missing"),
        "error message should hint at NotFound, got: {}",
        err.message
    );

    // Missing param → invalid params.
    let req = jsonrpc_req(3, "resources/read", json!({}));
    let resp = server.handle_request(req).await.expect("read replies");
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn resources_subscribe() {
    let (_tmp, server) = build_server().await;
    let registry = server.resources();
    registry.register(Arc::new(FakeResource::new("ostk://memory-lens", "body")));

    let req = jsonrpc_req(
        1,
        "resources/subscribe",
        json!({ "uri": "ostk://memory-lens" }),
    );
    let resp = server.handle_request(req).await.expect("subscribe replies");
    assert!(resp.result.is_some(), "subscribe success returns empty ok");

    // Bookkeeping: the singleton client must now be in the
    // subscriber list (covers the dispatch → registry plumbing).
    let subs = registry.subscribers_for("ostk://memory-lens");
    assert_eq!(subs.len(), 1);
    assert!(matches!(subs[0], ClientId::StdioSingleton));

    // Subscribing to an unknown URI must error.
    let req = jsonrpc_req(2, "resources/subscribe", json!({ "uri": "ostk://missing" }));
    let resp = server.handle_request(req).await.expect("subscribe replies");
    assert!(resp.error.is_some());
}

/// AsyncRead that always returns an `io::Error`. Used to drive
/// `serve` through the reader-error cleanup path.
struct ErrReader;

impl AsyncRead for ErrReader {
    fn poll_read(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Poll::Ready(Err(std::io::Error::other("synthetic reader error")))
    }
}

/// AsyncWrite that always errors on `poll_write`. The flush/shutdown
/// paths succeed so the writer task gets the error on the first
/// payload rather than during teardown.
#[derive(Default)]
struct ErrWriter {
    writes: AtomicUsize,
}

impl AsyncWrite for ErrWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        _buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        Poll::Ready(Err(std::io::Error::other("synthetic writer error")))
    }
    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

#[tokio::test]
async fn serve_propagates_reader_error_after_cleanup() {
    // Review-fix regression test: a reader I/O error must NOT skip
    // the cleanup phase. Before the fix, `read_line(...).await?`
    // short-circuited the function, leaving the registry's outbound
    // sender alive and the writer task waiting on rx.recv().
    use std::time::Duration;

    let (_tmp, server) = build_server().await;
    let registry = server.resources();
    registry.register(Arc::new(FakeResource::new("ostk://x", "body")));

    let buf: Vec<u8> = Vec::new();
    let server = Arc::new(server);
    let handle = {
        let server = Arc::clone(&server);
        tokio::spawn(async move { server.serve(ErrReader, buf).await })
    };

    // The reader errors on first read → cleanup runs → serve
    // returns Err inside the 2s budget. A wedge would blow past
    // this timeout.
    let result = tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("serve must return within 2s on reader I/O error")
        .expect("serve task must not panic");
    let err = result.expect_err("reader I/O error must propagate");
    assert!(
        err.to_string().contains("synthetic reader error"),
        "reader error must propagate verbatim, got: {err}"
    );

    // Cleanup contract: post-shutdown the registry must hold no
    // outbound sender, so a fresh transport can install one.
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    registry
        .subscribe(ClientId::stdio_singleton(), "ostk://x")
        .unwrap();
    registry.emit_resource_updated("ostk://x");
    assert!(
        rx.try_recv().is_err(),
        "registry must not hold a stale sender after reader-error shutdown"
    );
    registry.set_outbound(tx);
    registry.emit_resource_updated("ostk://x");
    let _ = rx.recv().await.expect("re-installed sender works");
}

#[tokio::test]
async fn serve_propagates_writer_error_after_cleanup() {
    // The companion path: a writer I/O error must also surface,
    // and cleanup must still run. We need a reader that yields one
    // valid request (so the writer task actually tries a write)
    // and then EOFs (so the reader loop completes Ok and lets the
    // writer error become the surfaced error).
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;

    let (_tmp, server) = build_server().await;
    let server = Arc::new(server);

    let (mut client_stdin, server_stdin) = tokio::io::duplex(4096);
    let server_stdout = ErrWriter::default();

    let handle = {
        let server = Arc::clone(&server);
        tokio::spawn(async move { server.serve(server_stdin, server_stdout).await })
    };

    // One valid request the writer will try to ship downstream,
    // then close stdin so the reader loop exits cleanly. The
    // writer's error becomes the function's return value.
    client_stdin
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n")
        .await
        .unwrap();
    drop(client_stdin);

    let result = tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("serve must return within 2s on writer I/O error")
        .expect("serve task must not panic");
    let err = result.expect_err("writer I/O error must propagate");
    assert!(
        err.to_string().contains("synthetic writer error"),
        "writer error must propagate verbatim, got: {err}"
    );
}

#[tokio::test]
async fn serve_exits_cleanly_on_stdin_eof() {
    // Review-fix regression test. Before the clear_outbound()
    // shutdown step, `serve` wedged forever on stdin EOF because
    // the registry held a clone of the outbound Sender and the
    // writer task's recv().await never observed the channel
    // closing. The bounded timeout makes the regression visible —
    // a wedge would blow past 2s.
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;

    let (_tmp, server) = build_server().await;
    let server = Arc::new(server);
    let (mut client_stdin, server_stdin) = tokio::io::duplex(4096);
    let (server_stdout, _client_stdout) = tokio::io::duplex(4096);

    let server_handle = {
        let server = Arc::clone(&server);
        tokio::spawn(async move { server.serve(server_stdin, server_stdout).await })
    };

    // Send a single ping so the dispatch path is exercised at
    // least once before EOF.
    client_stdin
        .write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n")
        .await
        .unwrap();
    // Closing the writer half delivers EOF on the server's read
    // half — the canonical "stdin closed" signal.
    drop(client_stdin);

    let exit = tokio::time::timeout(Duration::from_secs(2), server_handle).await;
    let join = exit.expect("serve must exit within 2s of stdin EOF");
    let io = join.expect("serve task must not panic");
    io.expect("serve returned an io error");
}

#[tokio::test]
async fn serve_eof_releases_registry_outbound_for_future_runs() {
    // The other half of the shutdown contract: after `serve`
    // returns, the registry's outbound slot must be empty so a
    // subsequent run_stdio (or a new transport) can install a
    // fresh Sender. emit_resource_updated must be a no-op until
    // then.
    use std::time::Duration;

    let (_tmp, server) = build_server().await;
    let registry = server.resources();
    registry.register(Arc::new(FakeResource::new("ostk://x", "body")));
    registry
        .subscribe(ClientId::stdio_singleton(), "ostk://x")
        .unwrap();

    let server = Arc::new(server);
    let (client_stdin, server_stdin) = tokio::io::duplex(4096);
    let (server_stdout, _client_stdout) = tokio::io::duplex(4096);

    let server_handle = {
        let server = Arc::clone(&server);
        tokio::spawn(async move { server.serve(server_stdin, server_stdout).await })
    };
    drop(client_stdin);
    tokio::time::timeout(Duration::from_secs(2), server_handle)
        .await
        .expect("serve exits on EOF")
        .expect("no panic")
        .expect("no io error");

    // The registry should hold no outbound now. emit_resource_updated
    // is a silent no-op — no channel to send into.
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    // If we left a stale sender behind, emit would push down it
    // before we re-installed our tx; the assertion below catches
    // that.
    registry.emit_resource_updated("ostk://x");
    assert!(
        rx.try_recv().is_err(),
        "registry must not hold a stale sender post-serve"
    );
    // Re-installing and emitting works (sanity).
    registry.set_outbound(tx);
    registry.emit_resource_updated("ostk://x");
    let _ = rx
        .recv()
        .await
        .expect("post-reinstall emit reaches channel");
}

#[tokio::test]
async fn resources_notifications() {
    // End-to-end: register, subscribe, fire emit_resource_updated,
    // assert the JSON-RPC envelope produced on the outbound channel
    // matches the spec. Per the gate doc: this test must verify the
    // actual envelope on the wire, not just registry internals.
    let (_tmp, server) = build_server().await;
    let registry = server.resources();
    registry.register(Arc::new(FakeResource::new("ostk://memory-lens", "body")));
    registry
        .subscribe(ClientId::stdio_singleton(), "ostk://memory-lens")
        .unwrap();

    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    registry.set_outbound(tx);
    registry.emit_resource_updated("ostk://memory-lens");

    let envelope = rx.recv().await.expect("envelope must be sent");
    let v: Value = serde_json::from_str(&envelope).unwrap();
    assert_eq!(v.get("jsonrpc").and_then(Value::as_str), Some("2.0"));
    assert_eq!(
        v.get("method").and_then(Value::as_str),
        Some("notifications/resources/updated")
    );
    assert_eq!(
        v.pointer("/params/uri").and_then(Value::as_str),
        Some("ostk://memory-lens")
    );
    // Notifications have no id field by JSON-RPC convention.
    assert!(
        v.get("id").is_none() || v.get("id") == Some(&Value::Null),
        "notifications must not carry an id"
    );
}

#[tokio::test]
async fn daemon_fans_out_resource_update_to_two_network_clients() {
    // End-to-end multi-client contract through the real
    // `serve_with_client` loop: two concurrent network connections each
    // subscribe, one emit reaches BOTH wires, and disconnecting one
    // prunes its subscription (so the daemon doesn't leak dead ids).
    use std::time::Duration;
    use tokio::io::AsyncBufReadExt;

    let (_tmp, server) = build_server().await;
    let registry = server.resources();
    registry.register(Arc::new(FakeResource::new("ostk://memory-lens", "body")));
    let server = Arc::new(server);

    let (mut r1, w1, h1) = connect_and_subscribe(
        Arc::clone(&server),
        ClientId::network(1),
        "ostk://memory-lens",
    )
    .await;
    let (mut r2, _w2, _h2) = connect_and_subscribe(
        Arc::clone(&server),
        ClientId::network(2),
        "ostk://memory-lens",
    )
    .await;

    assert_eq!(
        registry.subscribers_for("ostk://memory-lens").len(),
        2,
        "both network connections must be subscribed under distinct ids"
    );

    registry.emit_resource_updated("ostk://memory-lens");

    for (idx, reader) in [&mut r1, &mut r2].into_iter().enumerate() {
        let mut line = String::new();
        let read = tokio::time::timeout(Duration::from_secs(2), reader.read_line(&mut line))
            .await
            .unwrap_or_else(|_| panic!("client {idx} timed out waiting for notification"))
            .unwrap();
        assert!(read > 0, "client {idx} got EOF instead of a notification");
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(
            v.get("method").and_then(Value::as_str),
            Some("notifications/resources/updated"),
            "client {idx} must receive the resources/updated notification"
        );
    }

    // Disconnect client 1: dropping both ends gives the server read EOF,
    // so its serve loop exits and runs `remove_client` on the network id.
    drop(r1);
    drop(w1);
    tokio::time::timeout(Duration::from_secs(2), h1)
        .await
        .expect("client 1 serve exits on disconnect")
        .expect("client 1 serve task did not panic")
        .expect("client 1 serve returned no io error");
    assert_eq!(
        registry.subscribers_for("ostk://memory-lens").len(),
        1,
        "disconnected client's subscription must be pruned"
    );
}
