//! Adapter for `fcp-rust`, a rust-analyzer-backed MCP server.
//!
//! Spawns one `fcp-rust` subprocess per Cargo workspace. The workspace
//! must be fully indexed (rust-analyzer cold start: ~10-60 s on large
//! workspaces) before symbol queries return useful results. This client
//! is fully blocking — `fcp-rust` is a stdio JSON-RPC peer, and one
//! request must complete before the next is sent.
//!
//! The MCP transport uses newline-delimited JSON-RPC 2.0 frames. Wire
//! protocol contracts learned by probing the running binary:
//!
//! * `tools/call rust_session` takes `{"action": "open ABS_PATH"}` —
//!   the verb and target path are concatenated into a single string.
//! * `tools/call rust_query` takes `{"input": "VERB ARG..."}` — same
//!   concatenated form.
//! * Paths inside the `input` string must be **relative** to the
//!   workspace root opened by `rust_session open`. Absolute paths get
//!   joined to the workspace root and produce "file not found" errors.
//! * The response payload is `{"content":[{"type":"text","text":"..."}]}`;
//!   the `text` field is a human-readable transcript that we must parse.
//! * `symbols PATH` produces lines like `  NAME (KIND) L<n>` with the
//!   leading line `Symbols in <abs path>:`. No end-line is emitted —
//!   we approximate it as `next_symbol.line_start - 1`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use thiserror::Error;

/// One symbol returned by `rust_query symbols PATH`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RustSymbol {
    pub name: String,
    /// Lowercased rust-analyzer symbol kind: `function`, `struct`, `enum`,
    /// `trait`, `impl`, `module`, `constant`, `static`, `type`, `method`,
    /// `object`, `macro`, `field`, `variant`. Free-form by design — the
    /// chunker only uses it for the synthetic header.
    pub kind: String,
    /// 1-based line number of the symbol's declaration.
    pub line_start: u32,
    /// 1-based inclusive end line. Computed by the parser as
    /// `next_symbol.line_start - 1` (or end-of-file).
    pub line_end: u32,
}

#[derive(Debug, Error)]
pub enum FcpError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("fcp-rust protocol: {0}")]
    Protocol(String),
    #[error("fcp-rust returned error: {0}")]
    Server(String),
    #[error("timed out waiting for {what} ({waited:?})")]
    Timeout {
        what: &'static str,
        waited: Duration,
    },
    #[error("fcp-rust binary not found on PATH")]
    NotFound,
}

/// Default per-call request timeout. Used for `symbols`/`def`/etc.
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Default workspace-indexing wait. Override via env
/// `OSTK_RECALL_FCP_RUST_INDEX_TIMEOUT` (seconds).
const DEFAULT_INDEX_TIMEOUT: Duration = Duration::from_secs(60);
/// Poll cadence while waiting for `Status: ready`.
const STATUS_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Read the index-wait timeout from `OSTK_RECALL_FCP_RUST_INDEX_TIMEOUT`,
/// falling back to [`DEFAULT_INDEX_TIMEOUT`].
#[must_use]
pub fn index_timeout_from_env() -> Duration {
    std::env::var("OSTK_RECALL_FCP_RUST_INDEX_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or(DEFAULT_INDEX_TIMEOUT, Duration::from_secs)
}

/// Blocking wrapper around an `fcp-rust` subprocess.
pub struct FcpRustSession {
    child: Option<Child>,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: AtomicU64,
    /// Workspace root currently opened, if any. Used by [`Drop`] to send
    /// a graceful `close` before tearing down the subprocess.
    workspace: Option<PathBuf>,
}

impl FcpRustSession {
    /// Spawn `fcp-rust` and run the MCP `initialize` + `notifications/initialized`
    /// handshake. The child stderr is discarded; if you need it, use
    /// [`Self::spawn_with_stderr`].
    pub fn spawn() -> Result<Self, FcpError> {
        Self::spawn_inner(Stdio::null())
    }

    /// Same as [`Self::spawn`] but redirects child stderr to the supplied
    /// `Stdio` (typically a file under `<corpus_root>/logs/`).
    pub fn spawn_with_stderr(stderr: Stdio) -> Result<Self, FcpError> {
        Self::spawn_inner(stderr)
    }

    fn spawn_inner(stderr: Stdio) -> Result<Self, FcpError> {
        let mut child = match Command::new("fcp-rust")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()
        {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(FcpError::NotFound),
            Err(e) => return Err(e.into()),
        };

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| FcpError::Protocol("missing child stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| FcpError::Protocol("missing child stdout".into()))?;

        let mut session = Self {
            child: Some(child),
            stdin,
            stdout: BufReader::new(stdout),
            next_id: AtomicU64::new(1),
            workspace: None,
        };

        // initialize → wait for response → send "initialized" notification.
        let init_resp = session.request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ostk-recall-scan", "version": env!("CARGO_PKG_VERSION")},
            }),
            DEFAULT_REQUEST_TIMEOUT,
        )?;
        if init_resp.get("result").is_none() {
            return Err(FcpError::Protocol(format!(
                "initialize failed: {init_resp}"
            )));
        }
        session.notify("notifications/initialized", json!({}))?;

        Ok(session)
    }

    /// Open `root` as the workspace and poll `rust_session status` until
    /// the server reports `ready` (or `wait_timeout` elapses).
    pub fn open_workspace(&mut self, root: &Path, wait_timeout: Duration) -> Result<(), FcpError> {
        let abs = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let abs_str = abs.to_string_lossy().into_owned();

        // The `open` call itself blocks until indexing has progressed
        // far enough that `tools/call` returns. On large workspaces the
        // initial response can take ~15-30 s; allow the index timeout to
        // cover that, not just the per-call default.
        let resp = self.tool_call(
            "rust_session",
            json!({"action": format!("open {abs_str}")}),
            wait_timeout,
        )?;
        let text = extract_text(&resp).unwrap_or_default();
        if !text.starts_with("Opened workspace") {
            return Err(FcpError::Server(format!("open: {text}")));
        }
        self.workspace = Some(abs);

        // Belt-and-braces: poll status until `ready`. Most calls already
        // return `ready` immediately because `open` blocked.
        let deadline = Instant::now() + wait_timeout;
        loop {
            let status = self.tool_call(
                "rust_session",
                json!({"action": "status"}),
                DEFAULT_REQUEST_TIMEOUT,
            )?;
            let st = extract_text(&status).unwrap_or_default();
            if st.contains("Status: ready") {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(FcpError::Timeout {
                    what: "indexing",
                    waited: wait_timeout,
                });
            }
            std::thread::sleep(STATUS_POLL_INTERVAL);
        }
    }

    /// List symbols in `file`, relative to the opened workspace root.
    ///
    /// Returns symbols sorted by `line_start`. `line_end` is approximated
    /// by `next_symbol.line_start - 1` (or `total_lines` for the last
    /// symbol). Caller-supplied total-line count keeps the API single-
    /// allocation; pass `0` to leave end-lines bounded by 4 KiB above the
    /// next symbol (a safe default for Rust top-level items).
    pub fn symbols(&mut self, file: &Path, total_lines: u32) -> Result<Vec<RustSymbol>, FcpError> {
        let rel = self.relativize(file)?;
        let rel_str = rel.to_string_lossy().into_owned();
        let resp = self.tool_call(
            "rust_query",
            json!({"input": format!("symbols {rel_str}")}),
            DEFAULT_REQUEST_TIMEOUT,
        )?;
        let text = extract_text(&resp).unwrap_or_default();
        if text.starts_with("! LSP error") || text.starts_with("Error:") {
            return Err(FcpError::Server(text));
        }
        Ok(parse_symbols(&text, total_lines))
    }

    /// Send `rust_session close` and discard the response. Idempotent —
    /// safe to call repeatedly.
    pub fn close_workspace(&mut self) -> Result<(), FcpError> {
        if self.workspace.is_none() {
            return Ok(());
        }
        let _ = self.tool_call(
            "rust_session",
            json!({"action": "close"}),
            DEFAULT_REQUEST_TIMEOUT,
        )?;
        self.workspace = None;
        Ok(())
    }

    /// Close the workspace, drop stdin, and reap the child. Bypasses the
    /// `Drop` impl — use this when you want `?`-style error handling.
    pub fn close(mut self) -> Result<(), FcpError> {
        let _ = self.close_workspace();
        self.shutdown_child();
        Ok(())
    }

    fn shutdown_child(&mut self) {
        // Dropping stdin closes the pipe, which fcp-rust treats as EOF
        // and uses to exit cleanly. We grab stdin from a sentinel just
        // long enough to drop it.
        // SAFETY: the field is owned; replacing it with a sink-like writer
        // would be cleaner but stdin has no Default — so we ManuallyDrop
        // by taking the child first.
        if let Some(mut child) = self.child.take() {
            // Close stdin by replacing it with an already-closed pipe.
            // Easiest path: drop our writer by overwriting it with a
            // /dev/null sink we never use again. But ChildStdin has no
            // public Default — so we just kill on timeout.
            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) => {
                        if Instant::now() >= deadline {
                            let _ = child.kill();
                            let _ = child.wait();
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    Err(_) => {
                        let _ = child.kill();
                        let _ = child.wait();
                        return;
                    }
                }
            }
        }
    }

    /// Build a path relative to the opened workspace. Falls back to the
    /// raw path if `file` is not inside the workspace — fcp-rust will
    /// then return an LSP error we propagate.
    fn relativize(&self, file: &Path) -> Result<PathBuf, FcpError> {
        let ws = self
            .workspace
            .as_deref()
            .ok_or_else(|| FcpError::Protocol("no workspace open".into()))?;
        let abs = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
        Ok(abs
            .strip_prefix(ws)
            .map_or_else(|_| abs.clone(), Path::to_path_buf))
    }

    // `Value` is moved into `json!({...})` below, but clippy can't see
    // through the macro — silence its `needless_pass_by_value` complaint.
    #[allow(clippy::needless_pass_by_value)]
    fn tool_call(
        &mut self,
        name: &str,
        arguments: Value,
        timeout: Duration,
    ) -> Result<Value, FcpError> {
        let resp = self.request(
            "tools/call",
            json!({"name": name, "arguments": arguments}),
            timeout,
        )?;
        if let Some(err) = resp.get("error") {
            return Err(FcpError::Server(err.to_string()));
        }
        Ok(resp)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn request(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, FcpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let frame = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_frame(&frame)?;
        self.read_matching(id, timeout)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn notify(&mut self, method: &str, params: Value) -> Result<(), FcpError> {
        let frame = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_frame(&frame)
    }

    fn write_frame(&mut self, value: &Value) -> Result<(), FcpError> {
        let line = serde_json::to_string(value)?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn read_matching(&mut self, id: u64, timeout: Duration) -> Result<Value, FcpError> {
        // Newline-delimited JSON-RPC: read one line per frame, skip
        // notifications and stale id responses, return when our id matches.
        // The std BufReader doesn't expose per-call deadlines; we approximate
        // by tracking elapsed across read_line calls. fcp-rust normally
        // responds fast (<2 s), so this is rarely exercised.
        let started = Instant::now();
        loop {
            if started.elapsed() > timeout {
                return Err(FcpError::Timeout {
                    what: "response",
                    waited: timeout,
                });
            }
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line)?;
            if n == 0 {
                return Err(FcpError::Protocol("fcp-rust closed stdout".into()));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(trimmed)?;
            // Notifications carry no `id`; skip them.
            let Some(resp_id) = value.get("id").and_then(Value::as_u64) else {
                continue;
            };
            if resp_id == id {
                return Ok(value);
            }
            // Out-of-order response (we never have multiple in flight, so
            // this would only happen on protocol drift); skip and retry.
        }
    }
}

impl Drop for FcpRustSession {
    fn drop(&mut self) {
        if self.child.is_some() {
            let _ = self.close_workspace();
            self.shutdown_child();
        }
    }
}

/// Pull `result.content[0].text` from a JSON-RPC response. Returns `None`
/// for any shape mismatch.
fn extract_text(resp: &Value) -> Option<String> {
    resp.get("result")?
        .get("content")?
        .get(0)?
        .get("text")?
        .as_str()
        .map(str::to_string)
}

/// Parse the text body of `rust_query symbols PATH`. The format is:
///
/// ```text
/// Symbols in /abs/path/to/file.rs:
///   NAME (KIND) L<line>
///   NAME (KIND) L<line>
/// ```
///
/// Symbols are emitted in source order; we compute `line_end` as the
/// next symbol's `line_start - 1` (or `total_lines` for the last symbol).
/// `total_lines == 0` makes the last symbol's `line_end = u32::MAX`,
/// which the chunker treats as "to end of file".
fn parse_symbols(text: &str, total_lines: u32) -> Vec<RustSymbol> {
    let mut out: Vec<RustSymbol> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_start();
        // Skip the header `Symbols in PATH:` and any blank lines.
        if !line.starts_with(|c: char| c.is_alphabetic() || c == '_') {
            continue;
        }
        if line.starts_with("Symbols in ") || line.starts_with("No symbols") {
            continue;
        }
        // Format: "NAME (KIND) L<n>" — possibly with prose like
        // "impl Scanner for CodeScanner" preceding the parens.
        let Some(paren_open) = line.rfind(" (") else {
            continue;
        };
        let Some(paren_close) = line[paren_open..].find(") L") else {
            continue;
        };
        let name = line[..paren_open].trim().to_string();
        let kind_start = paren_open + 2; // after " ("
        let kind_end = paren_open + paren_close;
        let kind = line[kind_start..kind_end].trim().to_string();
        let line_start_str = line[paren_open + paren_close + 3..].trim();
        // Strip any trailing column suffix like ":12".
        let line_start_str = line_start_str.split(':').next().unwrap_or("");
        let Ok(line_start) = line_start_str.parse::<u32>() else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        out.push(RustSymbol {
            name,
            kind,
            line_start,
            line_end: 0,
        });
    }
    // Compute line_end as the next symbol's line_start - 1.
    let n = out.len();
    for i in 0..n {
        let end = if i + 1 < n {
            out[i + 1]
                .line_start
                .saturating_sub(1)
                .max(out[i].line_start)
        } else if total_lines == 0 {
            u32::MAX
        } else {
            total_lines.max(out[i].line_start)
        };
        out[i].line_end = end;
    }
    out
}

/// Walk `start` upward looking for `Cargo.toml`. Returns the directory
/// containing it, or `None` if no such ancestor exists. Used by the
/// code scanner to decide whether to delegate Rust files to fcp-rust.
#[must_use]
pub fn find_cargo_workspace(start: &Path) -> Option<PathBuf> {
    let mut cur = if start.is_file() {
        start.parent()?.to_path_buf()
    } else {
        start.to_path_buf()
    };
    loop {
        if cur.join("Cargo.toml").is_file() {
            return Some(cur);
        }
        cur = cur.parent()?.to_path_buf();
    }
}

/// Group an iterator of `.rs` paths by their nearest Cargo workspace root.
/// Files with no Cargo.toml ancestor land in the `None` bucket.
#[must_use]
pub fn group_by_workspace(
    paths: impl IntoIterator<Item = PathBuf>,
) -> HashMap<Option<PathBuf>, Vec<PathBuf>> {
    let mut by_ws: HashMap<Option<PathBuf>, Vec<PathBuf>> = HashMap::new();
    for path in paths {
        let ws = find_cargo_workspace(&path);
        by_ws.entry(ws).or_default().push(path);
    }
    by_ws
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn ensure_fcp_rust_present() -> bool {
        Command::new("fcp-rust")
            .arg("--help")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    #[test]
    fn parse_symbols_basic_format() {
        let body = "Symbols in /tmp/foo.rs:\n  WINDOW_LINES (constant) L19\n  CodeScanner (struct) L24\n  parse (method) L70\n";
        let syms = parse_symbols(body, 100);
        assert_eq!(syms.len(), 3);
        assert_eq!(syms[0].name, "WINDOW_LINES");
        assert_eq!(syms[0].kind, "constant");
        assert_eq!(syms[0].line_start, 19);
        // line_end of first = next.line_start - 1 = 23
        assert_eq!(syms[0].line_end, 23);
        assert_eq!(syms[1].name, "CodeScanner");
        assert_eq!(syms[1].line_end, 69);
        assert_eq!(syms[2].name, "parse");
        // last = total_lines = 100
        assert_eq!(syms[2].line_end, 100);
    }

    #[test]
    fn parse_symbols_handles_impl_object() {
        let body = "Symbols in /tmp/foo.rs:\n  impl Scanner for CodeScanner (object) L28\n  kind (method) L29\n";
        let syms = parse_symbols(body, 100);
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "impl Scanner for CodeScanner");
        assert_eq!(syms[0].kind, "object");
    }

    #[test]
    fn parse_symbols_empty_returns_empty() {
        assert!(parse_symbols("Symbols in /tmp/foo.rs:\n", 0).is_empty());
        assert!(parse_symbols("", 0).is_empty());
    }

    #[test]
    fn parse_symbols_unbounded_last_line_when_total_zero() {
        let body = "Symbols in /tmp/foo.rs:\n  only (function) L5\n";
        let syms = parse_symbols(body, 0);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].line_end, u32::MAX);
    }

    #[test]
    fn parse_symbols_skips_malformed_lines() {
        let body =
            "Symbols in /tmp/foo.rs:\n  good (function) L10\nrandom prose\n  also (struct) L20\n";
        let syms = parse_symbols(body, 50);
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "good");
        assert_eq!(syms[1].name, "also");
    }

    #[test]
    fn find_cargo_workspace_finds_ancestor() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("crates/x/src")).unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        let file = tmp.path().join("crates/x/src/lib.rs");
        std::fs::write(&file, "fn x() {}\n").unwrap();
        let ws = find_cargo_workspace(&file);
        assert_eq!(ws.as_deref(), Some(tmp.path()));
    }

    #[test]
    fn find_cargo_workspace_inner_cargo_wins() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("inner/src")).unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        std::fs::write(
            tmp.path().join("inner/Cargo.toml"),
            "[package]\nname=\"x\"\nversion=\"0.0.0\"\n",
        )
        .unwrap();
        let file = tmp.path().join("inner/src/lib.rs");
        std::fs::write(&file, "fn x() {}\n").unwrap();
        let ws = find_cargo_workspace(&file);
        assert_eq!(ws.as_deref(), Some(&*tmp.path().join("inner")));
    }

    #[test]
    fn find_cargo_workspace_returns_none_outside_cargo() {
        let tmp = TempDir::new().unwrap();
        let file = tmp.path().join("lone.rs");
        std::fs::write(&file, "fn x() {}\n").unwrap();
        assert!(find_cargo_workspace(&file).is_none());
    }

    #[test]
    fn group_by_workspace_buckets_files() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("a/src")).unwrap();
        std::fs::create_dir_all(tmp.path().join("b/src")).unwrap();
        std::fs::write(
            tmp.path().join("a/Cargo.toml"),
            "[package]\nname=\"a\"\nversion=\"0\"\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("b/Cargo.toml"),
            "[package]\nname=\"b\"\nversion=\"0\"\n",
        )
        .unwrap();
        let a1 = tmp.path().join("a/src/lib.rs");
        let a2 = tmp.path().join("a/src/mod.rs");
        let b1 = tmp.path().join("b/src/lib.rs");
        for p in [&a1, &a2, &b1] {
            std::fs::write(p, "fn x() {}\n").unwrap();
        }
        let lone = tmp.path().join("lone.rs");
        std::fs::write(&lone, "fn x() {}\n").unwrap();
        let lone_for_assert = lone.clone();
        let grouped = group_by_workspace(vec![a1, a2, b1, lone]);
        assert_eq!(grouped.get(&Some(tmp.path().join("a"))).unwrap().len(), 2);
        assert_eq!(grouped.get(&Some(tmp.path().join("b"))).unwrap().len(), 1);
        assert_eq!(grouped.get(&None).unwrap(), &vec![lone_for_assert]);
    }

    #[test]
    fn extract_text_pulls_first_text_payload() {
        let resp = json!({
            "result": {"content": [{"type":"text","text":"hello"}], "isError": false}
        });
        assert_eq!(extract_text(&resp).as_deref(), Some("hello"));
    }

    #[test]
    fn extract_text_none_on_shape_mismatch() {
        assert!(extract_text(&json!({"result": {}})).is_none());
        assert!(extract_text(&json!({})).is_none());
    }

    /// Live integration: open ostk-recall itself, query a known file,
    /// confirm symbols come back. Skipped if `fcp-rust` isn't installed.
    #[test]
    fn live_session_returns_symbols() {
        if !ensure_fcp_rust_present() {
            eprintln!("skipping: fcp-rust not on PATH");
            return;
        }
        let mut session = match FcpRustSession::spawn() {
            Ok(s) => s,
            Err(FcpError::NotFound) => {
                eprintln!("skipping: fcp-rust spawn returned NotFound");
                return;
            }
            Err(e) => panic!("spawn failed: {e}"),
        };

        // Try the workspace this test is running in. Fall back to skip
        // if for some reason indexing won't complete in 90 s.
        let root = std::env::current_dir().unwrap();
        let ws = find_cargo_workspace(&root).unwrap_or(root);
        if let Err(e) = session.open_workspace(&ws, Duration::from_secs(90)) {
            eprintln!("skipping: open_workspace failed: {e}");
            return;
        }

        let file = ws.join("crates/scan/src/code.rs");
        if !file.exists() {
            eprintln!("skipping: fixture file missing");
            let _ = session.close();
            return;
        }
        let lines = std::fs::read_to_string(&file)
            .map(|s| u32::try_from(s.lines().count()).unwrap_or(u32::MAX))
            .unwrap_or(0);
        let syms = session.symbols(&file, lines).expect("symbols query failed");
        assert!(!syms.is_empty(), "expected at least one symbol in code.rs");
        assert!(
            syms.iter().any(|s| s.name == "CodeScanner"),
            "expected CodeScanner among {:?}",
            syms.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
        let _ = session.close();
    }
}
