//! End-to-end MCP test that spawns `ostk-recall serve --stdio` as a
//! subprocess and exchanges JSON-RPC messages over stdin/stdout.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Arc;

use ostk_recall_cli::commands::{self, InitOutcome};
use ostk_recall_pipeline::ChunkEmbedder;
use tempfile::TempDir;

const FAKE_DIM: usize = 8;

struct FakeEmbedder;

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        FAKE_DIM
    }
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0; FAKE_DIM];
                let bucket = t.len() % FAKE_DIM;
                v[bucket] = 1.0;
                v
            })
            .collect()
    }
}

fn write_config(path: &Path, corpus_root: &Path, fixture: &Path) {
    let body = format!(
        r#"[corpus]
root = "{corpus}"

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "markdown"
project = "m"
paths = ["{fixture}"]
"#,
        corpus = corpus_root.display(),
        fixture = fixture.display(),
    );
    std::fs::write(path, body).unwrap();
}

fn write_fixture(root: &Path) {
    std::fs::create_dir_all(root).unwrap();
    std::fs::write(
        root.join("a.md"),
        "# Alpha\n\nalphabravo charlie quickfox jumps\n",
    )
    .unwrap();
}

/// Spawn the binary, send a newline-delimited JSON-RPC request, return the
/// decoded response (first line). Caller is responsible for termination.
fn read_line(reader: &mut BufReader<std::process::ChildStdout>) -> String {
    let mut s = String::new();
    reader.read_line(&mut s).unwrap();
    s
}

fn send(stdin: &mut std::process::ChildStdin, req: &serde_json::Value) {
    let line = serde_json::to_string(req).unwrap();
    stdin.write_all(line.as_bytes()).unwrap();
    stdin.write_all(b"\n").unwrap();
    stdin.flush().unwrap();
}

fn cargo_bin() -> std::path::PathBuf {
    env!("CARGO_BIN_EXE_ostk-recall").into()
}

#[tokio::test]
async fn serve_stdio_initialize_tools_list_tools_call_recall_stats() {
    let fixture = TempDir::new().unwrap();
    write_fixture(fixture.path());

    let corpus = TempDir::new().unwrap();
    let cfg_dir = TempDir::new().unwrap();
    let cfg_path = cfg_dir.path().join("config.toml");
    write_config(&cfg_path, corpus.path(), fixture.path());

    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);

    match commands::init(&cfg_path, Arc::clone(&embedder))
        .await
        .expect("init")
    {
        InitOutcome::Initialized { .. } => {}
        InitOutcome::WroteStarter { path } => {
            panic!("unexpected init outcome: wrote starter at {path:?}");
        }
    }
    let scan = commands::scan(&cfg_path, Arc::clone(&embedder), None, false)
        .await
        .expect("scan");
    assert!(scan.totals.chunks_upserted > 0);

    // Spawn the real binary with the fake-embedder escape hatch so the
    // subprocess doesn't need a real model on disk.
    //
    // NOTE: RUST_LOG=info forces the tracing subscriber to actually emit
    // log records — this is the gate that proves they're routed to stderr
    // and never interleaved with the JSON-RPC frames on stdout.
    let mut child = Command::new(cargo_bin())
        .env("OSTK_RECALL_FAKE_EMBEDDER", FAKE_DIM.to_string())
        // Don't pull the ONNX reranker in e2e — keeps the test offline.
        .env("OSTK_RECALL_SKIP_RERANKER", "1")
        .env("RUST_LOG", "info")
        .arg("--config")
        .arg(&cfg_path)
        .arg("serve")
        .arg("--stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ostk-recall serve --stdio");

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    // initialize
    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc":"2.0","id":1,"method":"initialize","params":{}
        }),
    );
    let line = read_line(&mut stdout);
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(v["result"]["serverInfo"]["name"], "ostk-recall");

    // tools/list
    send(
        &mut stdin,
        &serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
    );
    let line = read_line(&mut stdout);
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    let names: Vec<&str> = v["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"recall"));
    assert!(names.contains(&"recall_stats"));

    // tools/call recall_stats
    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc":"2.0","id":3,"method":"tools/call",
            "params": {"name":"recall_stats","arguments":{}}
        }),
    );
    let line = read_line(&mut stdout);
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    let stats: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(stats["total"].as_u64().unwrap() >= 1);

    // tools/call recall
    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc":"2.0","id":4,"method":"tools/call",
            "params": {"name":"recall","arguments":{"query":"alphabravo","limit":5}}
        }),
    );
    let line = read_line(&mut stdout);
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    let text = v["result"]["content"][0]["text"].as_str().unwrap();
    let body: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(body["hits"].is_array());

    // Close stdin to make server exit.
    drop(stdin);
    let exit = child.wait().expect("child exit");
    assert!(exit.success(), "child exit {exit:?}");
}

/// Stdout in `serve --stdio` must be reserved for JSON-RPC frames. A live
/// MCP client (Claude Desktop / Cursor / Claude Code) parses each line as
/// JSON; a stray log line breaks the session immediately.
///
/// We boot the server with `RUST_LOG=trace` (so the subscriber actually
/// emits records), send a real `initialize`, and assert the FIRST line on
/// stdout parses as JSON and is a valid JSON-RPC response.
#[tokio::test]
async fn serve_stdio_keeps_stdout_clean_of_logs() {
    let fixture = TempDir::new().unwrap();
    write_fixture(fixture.path());

    let corpus = TempDir::new().unwrap();
    let cfg_dir = TempDir::new().unwrap();
    let cfg_path = cfg_dir.path().join("config.toml");
    write_config(&cfg_path, corpus.path(), fixture.path());

    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
    commands::init(&cfg_path, Arc::clone(&embedder))
        .await
        .expect("init");

    let mut child = Command::new(cargo_bin())
        .env("OSTK_RECALL_FAKE_EMBEDDER", FAKE_DIM.to_string())
        // Don't pull the ONNX reranker in e2e — keeps the test offline.
        .env("OSTK_RECALL_SKIP_RERANKER", "1")
        // Crank logging up — every subsystem will try to write something.
        .env("RUST_LOG", "trace")
        .arg("--config")
        .arg(&cfg_path)
        .arg("serve")
        .arg("--stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ostk-recall serve --stdio");

    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        &serde_json::json!({
            "jsonrpc":"2.0","id":1,"method":"initialize","params":{}
        }),
    );
    let line = read_line(&mut stdout);

    let v: serde_json::Value = serde_json::from_str(&line).unwrap_or_else(|e| {
        panic!("first line of stdout must parse as JSON (logs leaked?): {e}\nline = {line:?}");
    });
    assert_eq!(v["jsonrpc"], "2.0", "expected JSON-RPC envelope, got {v}");
    assert_eq!(v["id"], 1);
    assert!(
        v["result"].is_object(),
        "expected initialize result, got {v}"
    );

    drop(stdin);
    let _ = child.wait();
}
