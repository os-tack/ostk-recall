//! ostk-recall-serve — peer-process daemon for the recall surface.
//!
//! Reads JSON-RPC 2.0 requests from stdin (newline-framed), writes
//! responses to stdout. Designed to be spawned by the haystack kernel
//! driver registry as `fcp-recall` and connected via the existing
//! concurrent driver relay (`kernel::driver_relay`).
//!
//! Spec: `docs/spec/driver-protocol.md` in this repo.
//!
//! Lifecycle: read until EOF or SIGTERM. Each line is one request.
//! Responses are emitted as soon as they're ready; the kernel relay
//! demuxes by JSON-RPC `id`.

use std::io::{self, Write};

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};

mod handlers;
mod protocol;
mod state;

use handlers::{handle_fault, handle_initialize, handle_ping};
use protocol::{ErrorCode, Request, Response, RpcError};
use state::State;

fn install_tracing() {
    // Logs go to stderr to keep stdout clean for JSON-RPC.
    let _ = tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("OSTK_RECALL_SERVE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    install_tracing();
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "ostk-recall-serve starting"
    );

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();

    let mut state: Option<State> = None;

    while let Some(line) = reader.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let response = process_line(line, &mut state).await;
        emit(&response);
    }

    tracing::info!("stdin EOF — shutting down");
    Ok(())
}

async fn process_line(line: &str, state: &mut Option<State>) -> Response {
    let req: Request = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            return Response::err(
                Value::Null,
                RpcError::new(ErrorCode::ParseError, format!("parse error: {e}")),
            );
        }
    };

    if req.jsonrpc != "2.0" {
        return Response::err(
            req.id,
            RpcError::new(ErrorCode::InvalidRequest, "jsonrpc must be \"2.0\""),
        );
    }

    match req.method.as_str() {
        "initialize" => {
            let (resp, new_state) = handle_initialize(req.id, req.params).await;
            if let Some(s) = new_state {
                *state = Some(s);
            }
            resp
        }
        "recall.fault" => match state.as_ref() {
            Some(s) => handle_fault(req.id, req.params, s).await,
            None => Response::err(
                req.id,
                RpcError::new(
                    ErrorCode::NotInitialized,
                    "must call `initialize` before `recall.fault`",
                ),
            ),
        },
        "ping" => handle_ping(req.id),
        other => Response::err(
            req.id,
            RpcError::new(
                ErrorCode::MethodNotFound,
                format!("unknown method: {other}"),
            )
            .with_data(json!({ "method": other })),
        ),
    }
}

/// Write a response to stdout, newline-framed.
fn emit(resp: &Response) {
    let mut stdout = io::stdout().lock();
    if let Ok(s) = serde_json::to_string(resp) {
        let _ = stdout.write_all(s.as_bytes());
        let _ = stdout.write_all(b"\n");
        let _ = stdout.flush();
    }
}
