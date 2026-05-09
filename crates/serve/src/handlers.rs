//! JSON-RPC method handlers for the recall driver.

use std::path::Path;
use std::sync::Arc;

use ostk_recall_core::RecallParams;
use ostk_recall_embed::Embedder;
use ostk_recall_query::{recall, Synthesizer};
use ostk_recall_store::CorpusStore;
use serde_json::{json, Value};
use tracing::info;

use crate::protocol::{
    EmbedderInfo, ErrorCode, FaultParams, FaultResult, InitializeParams, InitializeResult,
    NamedPage, PingResult, Response, RpcError,
};
use crate::state::State;

/// Default embedding model — matches the haystack squasher pipeline so
/// the corpus dim stays compatible across the in-process embedder
/// (squasher) and the daemon embedder (here).
const DEFAULT_MODEL: &str = "minishlab/potion-base-8M";

pub async fn handle_initialize(id: Value, params: Value) -> (Response, Option<State>) {
    let p: InitializeParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => {
            return (
                Response::err(
                    id,
                    RpcError::new(ErrorCode::InvalidParams, format!("invalid params: {e}")),
                ),
                None,
            );
        }
    };

    let ostk_dir = Path::new(&p.ostk_dir);
    if !ostk_dir.is_absolute() {
        return (
            Response::err(
                id,
                RpcError::new(
                    ErrorCode::InvalidParams,
                    "ostk_dir must be an absolute path",
                ),
            ),
            None,
        );
    }
    if !ostk_dir.exists() {
        return (
            Response::err(
                id,
                RpcError::new(
                    ErrorCode::CorpusNotFound,
                    format!("ostk_dir does not exist: {}", ostk_dir.display()),
                )
                .with_data(json!({ "ostk_dir": ostk_dir.display().to_string() })),
            ),
            None,
        );
    }

    info!(ostk_dir = %ostk_dir.display(), "loading embedder + corpus store");

    let embedder = match Embedder::load(DEFAULT_MODEL) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            return (
                Response::err(
                    id,
                    RpcError::new(
                        ErrorCode::EmbedderLoadFailed,
                        format!("embedder load failed: {e}"),
                    )
                    .with_data(json!({ "model": DEFAULT_MODEL })),
                ),
                None,
            );
        }
    };

    if embedder.dim() != p.embed_dim {
        return (
            Response::err(
                id,
                RpcError::new(
                    ErrorCode::EmbedderLoadFailed,
                    format!(
                        "embedder dim mismatch: requested {}, model {} provides {}",
                        p.embed_dim,
                        DEFAULT_MODEL,
                        embedder.dim(),
                    ),
                )
                .with_data(json!({
                    "requested_dim": p.embed_dim,
                    "actual_dim": embedder.dim(),
                    "model": DEFAULT_MODEL,
                })),
            ),
            None,
        );
    }

    let store = match CorpusStore::open_or_create(ostk_dir, embedder.dim()).await {
        Ok(s) => Arc::new(s),
        Err(e) => {
            return (
                Response::err(
                    id,
                    RpcError::new(
                        ErrorCode::CorpusNotFound,
                        format!("failed to open corpus store: {e}"),
                    )
                    .with_data(json!({ "ostk_dir": ostk_dir.display().to_string() })),
                ),
                None,
            );
        }
    };

    let corpus_root = ostk_dir.join("recall").display().to_string();
    let state = State {
        corpus_root: ostk_dir.join("recall"),
        store: store.clone(),
        embedder: embedder.clone(),
        embedder_model: DEFAULT_MODEL.to_string(),
    };

    let result = InitializeResult {
        name: "ostk-recall-serve",
        version: env!("CARGO_PKG_VERSION"),
        embedder: EmbedderInfo {
            model: DEFAULT_MODEL.to_string(),
            dim: embedder.dim(),
        },
        corpus_root,
    };

    (
        Response::ok(id, serde_json::to_value(&result).unwrap()),
        Some(state),
    )
}

pub async fn handle_fault(id: Value, params: Value, state: &State) -> Response {
    let p: FaultParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => {
            return Response::err(
                id,
                RpcError::new(ErrorCode::InvalidParams, format!("invalid params: {e}")),
            );
        }
    };

    if p.query.is_empty() {
        return Response::err(
            id,
            RpcError::new(ErrorCode::InvalidParams, "query must be non-empty"),
        );
    }

    let params = RecallParams {
        query: p.query,
        intent: p.intent,
        limit: p.limit,
        max_per_source_id: p.max_per_source_id,
        ..Default::default()
    };

    let hits = match recall(
        state.store.connection(),
        state.embedder.as_ref(),
        None, // no reranker — out of scope for cut #3 v0.2.0; daemon ships without
        &params,
    )
    .await
    {
        Ok(h) => h,
        Err(e) => {
            return Response::err(
                id,
                RpcError::new(ErrorCode::RecallFailed, format!("recall failed: {e}")),
            );
        }
    };

    let pages = Synthesizer::collapse(hits);
    let mut named = Vec::with_capacity(pages.len());
    for page in &pages {
        match NamedPage::from_page(page) {
            Ok(np) => named.push(np),
            Err(e) => {
                return Response::err(
                    id,
                    RpcError::new(
                        ErrorCode::InternalError,
                        format!("failed to serialize synthesized page: {e}"),
                    ),
                );
            }
        }
    }

    Response::ok(
        id,
        serde_json::to_value(FaultResult { pages: named }).unwrap(),
    )
}

pub fn handle_ping(id: Value) -> Response {
    Response::ok(id, serde_json::to_value(PingResult { ok: true }).unwrap())
}
