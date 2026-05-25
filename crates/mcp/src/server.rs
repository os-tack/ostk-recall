//! MCP server: JSON-RPC over stdio.

use std::sync::Arc;

use ostk_recall_attention_mcp::{
    AttentionDispatch, AttentionHandlersError, DefaultAttentionHandlers, attention_tools,
    thread_tools,
};
use ostk_recall_core::AttentionBiasParams;
use ostk_recall_query::{
    QueryEngine, QueryError, RecallHit, RecallParams, SynthesizedPage, Synthesizer,
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{error, info, warn};

use crate::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::tools::tool_list;

pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// MCP stdio server. Holds an `Arc<QueryEngine>` so it can be cloned cheaply
/// across tasks if the caller ever wants to run multiple transports.
///
/// The optional `attention` dispatch unlocks the
/// `attention_*` / `thread_*` tool families when the caller (typically
/// `cli::commands::serve`) constructs a long-lived `AttentionDispatch`
/// and threads it through.
pub struct Server {
    engine: Arc<QueryEngine>,
    attention: Option<Arc<AttentionDispatch>>,
}

impl Server {
    #[must_use]
    pub fn new(engine: QueryEngine) -> Self {
        Self {
            engine: Arc::new(engine),
            attention: None,
        }
    }

    #[must_use]
    pub const fn from_arc(engine: Arc<QueryEngine>) -> Self {
        Self {
            engine,
            attention: None,
        }
    }

    /// Attach an `AttentionDispatch` so the attention/thread MCP tools
    /// become callable. Without one, those tools are not advertised in
    /// `tools/list` and `tools/call` returns method-not-found.
    #[must_use]
    pub fn with_attention(mut self, dispatch: Arc<AttentionDispatch>) -> Self {
        self.attention = Some(dispatch);
        self
    }

    pub const fn engine(&self) -> &Arc<QueryEngine> {
        &self.engine
    }

    /// Read newline-delimited JSON requests from stdin, dispatch, write
    /// responses to stdout. Returns when EOF is reached on stdin.
    pub async fn run_stdio(&self) -> std::io::Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        info!("mcp server ready on stdio");
        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    let resp = JsonRpcResponse::err(
                        Value::Null,
                        JsonRpcError::parse(format!("parse error: {e}")),
                    );
                    write_response(&mut stdout, &resp).await?;
                    continue;
                }
            };
            let reply = self.handle_request(value).await;
            if let Some(r) = reply {
                write_response(&mut stdout, &r).await?;
            }
        }
        info!("mcp server shutting down");
        Ok(())
    }

    /// Dispatch one JSON-RPC message. Returns None for notifications (no id).
    pub async fn handle_request(&self, raw: Value) -> Option<JsonRpcResponse> {
        let req: JsonRpcRequest = match serde_json::from_value(raw.clone()) {
            Ok(r) => r,
            Err(e) => {
                let id = raw.get("id").cloned().unwrap_or(Value::Null);
                return Some(JsonRpcResponse::err(
                    id,
                    JsonRpcError::invalid_request(format!("malformed request: {e}")),
                ));
            }
        };
        let id = req.id.clone().unwrap_or(Value::Null);
        let is_notification = req.id.is_none();

        match req.method.as_str() {
            "initialize" => Some(JsonRpcResponse::ok(id, Self::handle_initialize())),
            "initialized" | "notifications/initialized" | "notifications/cancelled" => None,
            "ping" => Some(JsonRpcResponse::ok(id, json!({}))),
            "tools/list" => Some(JsonRpcResponse::ok(id, self.handle_tools_list())),
            "tools/call" => match self.handle_tools_call(req.params).await {
                Ok(v) => Some(JsonRpcResponse::ok(id, v)),
                Err(err) => Some(JsonRpcResponse::err(id, err)),
            },
            _ => {
                if is_notification {
                    warn!(method = %req.method, "unknown notification — ignoring");
                    None
                } else {
                    Some(JsonRpcResponse::err(
                        id,
                        JsonRpcError::method_not_found(&req.method),
                    ))
                }
            }
        }
    }

    fn handle_initialize() -> Value {
        json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": {
                "name": "ostk-recall",
                "version": env!("CARGO_PKG_VERSION"),
            }
        })
    }

    fn handle_tools_list(&self) -> Value {
        let mut tools = tool_list(self.engine.has_audit());
        if self.attention.is_some() {
            tools.extend(attention_tools());
            tools.extend(thread_tools());
        }
        json!({ "tools": tools })
    }

    async fn handle_tools_call(&self, params: Value) -> std::result::Result<Value, JsonRpcError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| JsonRpcError::invalid_params("missing tool name"))?
            .to_string();
        let args = params.get("arguments").cloned().unwrap_or(Value::Null);

        let result_json = match name.as_str() {
            "recall" => {
                let p: RecallParams = serde_json::from_value(args.clone())
                    .map_err(|e| JsonRpcError::invalid_params(format!("recall args: {e}")))?;
                let bias = p.attention_bias.clone();
                let mut hits = self.engine.recall(p).await.map_err(query_error_to_rpc)?;
                if let Some(bias) = bias {
                    if let Some(dispatch) = self.attention.as_ref() {
                        apply_attention_bias(&mut hits, &bias, dispatch).await;
                    }
                }
                json!({ "hits": hits })
            }
            "recall_link" => {
                let chunk_id = args
                    .get("chunk_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| JsonRpcError::invalid_params("missing chunk_id"))?;
                let out = self
                    .engine
                    .recall_link(chunk_id)
                    .await
                    .map_err(query_error_to_rpc)?;
                serde_json::to_value(out)
                    .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?
            }
            "recall_stats" => {
                let out = self
                    .engine
                    .recall_stats()
                    .await
                    .map_err(query_error_to_rpc)?;
                serde_json::to_value(out)
                    .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?
            }
            "recall_audit" => {
                let sql = args
                    .get("sql")
                    .and_then(Value::as_str)
                    .ok_or_else(|| JsonRpcError::invalid_params("missing sql"))?;
                let out = self.engine.recall_audit(sql).map_err(query_error_to_rpc)?;
                serde_json::to_value(out)
                    .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?
            }
            "recall_fault" => {
                // →1848 cut #3: synthesize-style recall for haystack
                // `mem.fault_recall`. Embed + recall + Synthesizer::collapse;
                // return (name, content) pairs for the caller to write into
                // its page table. The daemon does NOT touch any page table.
                let p: RecallParams = serde_json::from_value(args.clone())
                    .map_err(|e| JsonRpcError::invalid_params(format!("recall_fault args: {e}")))?;
                if p.query.is_empty() {
                    return Err(JsonRpcError::invalid_params("query must be non-empty"));
                }
                let hits = self.engine.recall(p).await.map_err(query_error_to_rpc)?;
                let pages = Synthesizer::collapse(hits);
                let named: Vec<Value> = pages
                    .iter()
                    .map(named_page_value)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| JsonRpcError::internal(format!("serialize page: {e}")))?;
                json!({ "pages": named })
            }
            other => {
                if let Some(d) = self.attention.as_ref() {
                    if is_attention_tool(other) {
                        let out = d
                            .dispatch(other, args)
                            .await
                            .map_err(attention_error_to_rpc)?;
                        let text = serde_json::to_string(&out).map_err(|e| {
                            JsonRpcError::internal(format!("serialize: {e}"))
                        })?;
                        return Ok(json!({
                            "content": [{ "type": "text", "text": text }],
                            "isError": false,
                        }));
                    }
                }
                return Err(JsonRpcError::method_not_found(&format!(
                    "tools/call/{other}"
                )));
            }
        };

        // Wrap in MCP content block.
        let text = serde_json::to_string(&result_json)
            .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?;
        Ok(json!({
            "content": [
                { "type": "text", "text": text }
            ],
            "isError": false
        }))
    }
}

fn is_attention_tool(name: &str) -> bool {
    name.starts_with("attention_") || name.starts_with("thread_")
}

fn attention_error_to_rpc(err: AttentionHandlersError) -> JsonRpcError {
    match err {
        AttentionHandlersError::InvalidParams(m) => JsonRpcError::invalid_params(m),
        AttentionHandlersError::InvalidHandle(e) => {
            JsonRpcError::invalid_params(format!("invalid handle: {e}"))
        }
        AttentionHandlersError::PrivacyForbidden(tier) => {
            JsonRpcError::invalid_params(format!("privacy tier {tier:?} not permitted"))
        }
        other => JsonRpcError::internal(other.to_string()),
    }
}

async fn write_response<W: AsyncWriteExt + Unpin>(
    w: &mut W,
    resp: &JsonRpcResponse,
) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(resp).map_err(|e| {
        error!(error = %e, "serialize response");
        std::io::Error::other(e)
    })?;
    line.push(b'\n');
    w.write_all(&line).await?;
    w.flush().await?;
    Ok(())
}

/// Re-rank recall hits by what the caller is attending to right now.
///
/// For each hit, look up every thread that anchors on its chunk or
/// has an evidence link resolved to it, then take the max
/// `score_thread(handle)` from the in-memory attention store. The hit
/// keeps its original `score` in `base_score`, gains `attention_score`
/// (clamped to `[0, 1]`) and `attention_weight`, and ends up with
/// `score = base_score + attention_weight * attention_score`. Hits
/// with no anchoring thread get `attention_score = 0.0` and ride on
/// their base score alone.
///
/// Discipline: this is the operator's lens, not the substrate's.
/// `weight = 0.0` is identity; non-zero weights blend visibly through
/// the per-hit attribution. Resort is stable on `score` descending.
async fn apply_attention_bias(
    hits: &mut Vec<RecallHit>,
    bias: &AttentionBiasParams,
    dispatch: &ostk_recall_attention_mcp::AttentionDispatch,
) {
    use ostk_recall_core::attention::ThreadHandle;

    if hits.is_empty() {
        return;
    }
    let weight = if bias.weight.is_finite() && bias.weight >= 0.0 {
        bias.weight
    } else {
        0.0
    };

    for hit in hits.iter_mut() {
        let base = hit.score;
        let handles = match dispatch.threads.find_threads_for_chunk(&hit.chunk_id) {
            Ok(h) => h,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    chunk_id = %hit.chunk_id,
                    "attention-bias: find_threads_for_chunk failed; treating as 0"
                );
                Vec::<ThreadHandle>::new()
            }
        };
        let mut max_thread_score = 0.0_f32;
        for handle in &handles {
            match dispatch.attention.score_thread(handle).await {
                Ok(s) if s.is_finite() && s > max_thread_score => max_thread_score = s,
                Ok(_) => {}
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        handle = %handle,
                        "attention-bias: score_thread failed; treating as 0"
                    );
                }
            }
        }
        let attention = max_thread_score.clamp(0.0, 1.0);
        hit.base_score = Some(base);
        hit.attention_score = Some(attention);
        hit.attention_weight = Some(weight);
        hit.score = base + weight * attention;
    }

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn query_error_to_rpc(err: QueryError) -> JsonRpcError {
    match err {
        QueryError::Forbidden(msg) => JsonRpcError::invalid_params(msg),
        QueryError::NoEventsStore => JsonRpcError::invalid_request("events store not configured"),
        QueryError::NotFound(id) => JsonRpcError::invalid_params(format!("not found: {id}")),
        QueryError::Decode(m) => JsonRpcError::internal(format!("decode: {m}")),
        other => JsonRpcError::internal(other.to_string()),
    }
}

/// Build the `{name, content}` pair for one synthesized page. The name
/// is the kernel page-table key; content is the JSON-encoded
/// `SynthesizedPage` the kernel writes via `store_page_owned`.
fn named_page_value(page: &SynthesizedPage) -> std::result::Result<Value, serde_json::Error> {
    let slug = page
        .head
        .source_id
        .replace(['/', '\\'], ":")
        .replace(' ', "_");
    let name = format!("recall:{slug}");
    let content = serde_json::to_string(page)?;
    Ok(json!({ "name": name, "content": content }))
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod attention_bias_tests {
    use super::*;
    use ostk_recall_attention::{AttentionForwardStore, InMemoryAttention};
    use ostk_recall_attention_mcp::AttentionDispatch;
    use ostk_recall_core::Links;
    use ostk_recall_core::attention::{AttentionScope, PrivacyTier, ThreadHandle};
    use ostk_recall_store::{TensionState, ThreadRecord, ThreadsDb};
    use tempfile::TempDir;

    fn scope() -> AttentionScope {
        AttentionScope {
            project: Some("p".into()),
            session_id: Some("s".into()),
            agent: Some("test".into()),
            privacy_tier: PrivacyTier::T1Project,
        }
    }

    fn hit(chunk_id: &str, score: f32) -> RecallHit {
        RecallHit {
            chunk_id: chunk_id.into(),
            project: Some("p".into()),
            source: "markdown".into(),
            source_id: format!("{chunk_id}.md"),
            ts: None,
            snippet: format!("text-{chunk_id}"),
            score,
            links: Links::default(),
            extra: serde_json::Value::Null,
            stale: false,
            role: None,
            base_score: None,
            attention_score: None,
            attention_weight: None,
        }
    }

    async fn build_dispatch() -> (TempDir, Arc<AttentionDispatch>) {
        let tmp = TempDir::new().unwrap();
        let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
        let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
        (
            tmp,
            Arc::new(AttentionDispatch::new(attention, threads)),
        )
    }

    /// Seed a thread anchored to a chunk + light up its in-memory score.
    async fn seed_anchored_thread(
        d: &AttentionDispatch,
        handle_str: &str,
        anchor_chunk: &str,
    ) -> ThreadHandle {
        let handle = ThreadHandle::new(handle_str).unwrap();
        let now = chrono::Utc::now();
        d.threads
            .upsert_thread(&ThreadRecord {
                handle: handle.clone(),
                tension: TensionState::Active,
                familiarity: 0,
                last_touched_at: now,
                anchor_chunk_id: Some(anchor_chunk.into()),
                fold_override: None,
                created_at: now,
                created_scope_key: None,
                privacy_tier: PrivacyTier::T1Project,
            })
            .unwrap();
        // attend + familiarize so the InMemoryAttention has a non-zero
        // score for the handle. Bump familiarity multiple times so the
        // floor stays above ARCHIVE_THRESHOLD.
        d.attention.attend(&scope(), "active context").await.unwrap();
        for _ in 0..5 {
            d.attention.familiarize(&scope(), &handle).await.unwrap();
        }
        let score = d.attention.score_thread(&handle).await.unwrap();
        assert!(
            score > 0.0,
            "seed_anchored_thread: score must be positive ({score})"
        );
        handle
    }

    #[tokio::test]
    async fn bias_lifts_anchored_hit_above_unrelated_hit() {
        let (_tmp, d) = build_dispatch().await;
        let _h = seed_anchored_thread(&d, "fade-is-concentration", "anchored").await;

        // Two hits with equal base scores; one anchored, one not.
        let mut hits = vec![hit("anchored", 0.5), hit("unrelated", 0.5)];
        let bias = AttentionBiasParams {
            scope: scope(),
            weight: 1.0,
        };
        apply_attention_bias(&mut hits, &bias, &d).await;

        // Resorted: the anchored hit is first.
        assert_eq!(hits[0].chunk_id, "anchored");
        assert_eq!(hits[1].chunk_id, "unrelated");

        // Anchored hit gained an attention contribution.
        let a = &hits[0];
        assert_eq!(a.base_score, Some(0.5));
        assert!(a.attention_score.unwrap() > 0.0);
        assert_eq!(a.attention_weight, Some(1.0));
        // Math is decomposable: score == base_score + weight * attention_score
        let expected = a.base_score.unwrap()
            + a.attention_weight.unwrap() * a.attention_score.unwrap();
        assert!(
            (a.score - expected).abs() < 1e-5,
            "score must equal base + weight*attention (got {} vs expected {expected})",
            a.score
        );

        // Unrelated hit is fully attributed too; attention_score is 0.
        let u = &hits[1];
        assert_eq!(u.base_score, Some(0.5));
        assert_eq!(u.attention_score, Some(0.0));
        assert_eq!(u.score, 0.5);
    }

    #[tokio::test]
    async fn bias_with_zero_weight_preserves_order_and_scores() {
        let (_tmp, d) = build_dispatch().await;
        seed_anchored_thread(&d, "abi-as-sovereign-boundary", "anchored").await;

        let mut hits = vec![hit("anchored", 0.5), hit("unrelated", 0.7)];
        let bias = AttentionBiasParams {
            scope: scope(),
            weight: 0.0,
        };
        apply_attention_bias(&mut hits, &bias, &d).await;

        // weight=0 is identity on score. The base_score / attention_score
        // fields are still populated for caller-side reasoning, but the
        // final scores (and therefore the rank) are unchanged from the
        // pre-bias state.
        assert_eq!(hits[0].chunk_id, "unrelated");
        assert_eq!(hits[0].score, 0.7);
        assert_eq!(hits[1].chunk_id, "anchored");
        assert_eq!(hits[1].score, 0.5);
        for h in &hits {
            assert_eq!(h.attention_weight, Some(0.0));
            assert!(h.base_score.is_some());
            assert!(h.attention_score.is_some());
        }
    }

    #[tokio::test]
    async fn bias_with_no_anchor_leaves_score_unchanged() {
        // A hit whose chunk has no anchoring thread carries
        // attention_score=0 and its final score equals its base score
        // regardless of `weight`. Proves the bias is local to anchored
        // hits — it doesn't accidentally re-rank the whole list.
        let (_tmp, d) = build_dispatch().await;
        let mut hits = vec![hit("alpha", 0.6)];
        let bias = AttentionBiasParams {
            scope: scope(),
            weight: 2.0,
        };
        apply_attention_bias(&mut hits, &bias, &d).await;
        assert_eq!(hits[0].score, 0.6);
        assert_eq!(hits[0].attention_score, Some(0.0));
    }
}
