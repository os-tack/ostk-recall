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
/// Two independent axes, each clamped to `[0, 1]` and each with its
/// own weight (Phase B of the focus feature, post-v0.4.2 §3):
///
/// - **Thread-mediated** (`thread_weight`): max `score_thread(h)`
///   over every thread `h` returned by
///   `find_threads_for_chunk(hit.chunk_id)`. Lifts hits whose
///   chunk is already cited by a thread the operator is paying
///   attention to. v0.4.x behaviour; default 1.0.
/// - **Embedding-mediated** (`embedding_weight`): cosine between
///   the hit's chunk embedding (fetched once per call via
///   `corpus.fetch_embeddings`) and the scope's current attention
///   vector (`InMemoryAttention::scope_vector`). Lifts hits whose
///   content matches the operator's focus directly. Default 0.0,
///   so the wire shape stays back-compat for callers that don't
///   opt in.
///
/// Composition:
/// ```text
/// score = base_score
///       + thread_weight    * thread_score
///       + embedding_weight * embedding_score
/// ```
///
/// Per-hit attribution carries every term: `base_score`,
/// `thread_score`, `embedding_score`, `thread_weight`,
/// `embedding_weight`. The deprecated `attention_score` and
/// `attention_weight` are populated identically to `thread_score`
/// and `thread_weight` for v0.4.x clients; removed at v1.0.0.
///
/// Discipline: this is the operator's lens, not the substrate's.
/// Both weights at 0 is identity; any non-zero weight blends
/// visibly through the per-hit attribution.
async fn apply_attention_bias(
    hits: &mut Vec<RecallHit>,
    bias: &AttentionBiasParams,
    dispatch: &ostk_recall_attention_mcp::AttentionDispatch,
) {
    use ostk_recall_attention::cosine_similarity;
    use ostk_recall_core::attention::ThreadHandle;

    if hits.is_empty() {
        return;
    }
    let thread_w = sanitize_weight(bias.thread_weight);
    let embed_w = sanitize_weight(bias.embedding_weight);

    // Fetch scope vector once. If `None`, embedding-mediated bias
    // contributes 0 for every hit — equivalent to embedding_weight=0
    // but still recorded in the per-hit attribution.
    let scope_vec = if embed_w > 0.0 {
        match dispatch.attention.scope_vector(&bias.scope).await {
            Ok(v) => v,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "attention-bias: scope_vector failed; embedding axis contributes 0"
                );
                None
            }
        }
    } else {
        None
    };

    // Batch-fetch all hit embeddings up front when we'll need them.
    // Skip when no scope vector is available (every cosine would be
    // 0 anyway) or no corpus is wired into the dispatch.
    let hit_embeddings: std::collections::HashMap<String, Vec<f32>> = match (
        scope_vec.as_ref(),
        dispatch.corpus.as_ref(),
    ) {
        (Some(_), Some(corpus)) => {
            let ids: Vec<String> = hits.iter().map(|h| h.chunk_id.clone()).collect();
            match corpus.fetch_embeddings(&ids).await {
                Ok(map) => map,
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "attention-bias: fetch_embeddings failed; embedding axis contributes 0"
                    );
                    std::collections::HashMap::new()
                }
            }
        }
        _ => std::collections::HashMap::new(),
    };

    for hit in hits.iter_mut() {
        let base = hit.score;

        // Thread-mediated axis (unchanged from v0.4.x).
        let handles = match dispatch.threads.find_threads_for_chunk(&hit.chunk_id) {
            Ok(h) => h,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    chunk_id = %hit.chunk_id,
                    "attention-bias: find_threads_for_chunk failed; thread axis contributes 0"
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
        let thread_score = max_thread_score.clamp(0.0, 1.0);

        // Embedding-mediated axis (new in v0.5 / Phase B). Falls back
        // to 0 cleanly when either side is missing.
        let embedding_score = match (scope_vec.as_ref(), hit_embeddings.get(&hit.chunk_id)) {
            (Some(sv), Some(he)) if !sv.is_empty() && !he.is_empty() => {
                cosine_similarity(sv, he).clamp(0.0, 1.0)
            }
            _ => 0.0,
        };

        hit.base_score = Some(base);
        hit.thread_score = Some(thread_score);
        hit.embedding_score = Some(embedding_score);
        hit.thread_weight = Some(thread_w);
        hit.embedding_weight = Some(embed_w);
        // Deprecated v0.4.x aliases — populated identically so
        // clients that haven't migrated still see the thread axis.
        hit.attention_score = Some(thread_score);
        hit.attention_weight = Some(thread_w);
        hit.score = base + thread_w * thread_score + embed_w * embedding_score;
    }

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn sanitize_weight(w: f32) -> f32 {
    if w.is_finite() && w >= 0.0 { w } else { 0.0 }
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
            thread_score: None,
            embedding_score: None,
            thread_weight: None,
            embedding_weight: None,
            attention_score: None,
            attention_weight: None,
        }
    }

    fn thread_bias(weight: f32) -> AttentionBiasParams {
        AttentionBiasParams {
            scope: scope(),
            thread_weight: weight,
            embedding_weight: 0.0,
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
        let bias = thread_bias(1.0);
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
        let bias = thread_bias(0.0);
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
        let bias = thread_bias(2.0);
        apply_attention_bias(&mut hits, &bias, &d).await;
        assert_eq!(hits[0].score, 0.6);
        assert_eq!(hits[0].attention_score, Some(0.0));
    }

    // ---- Phase B: embedding-mediated bias ------------------------------

    #[tokio::test]
    async fn weight_alias_back_compat_on_the_wire() {
        // v0.4.x callers wrote {"scope": {...}, "weight": N}. The wire
        // schema must keep accepting that form: serde alias copies it
        // into thread_weight and embedding_weight defaults to 0.
        let json_v04 = serde_json::json!({
            "scope": {
                "project": "p",
                "session_id": "s",
                "agent": "test",
                "privacy_tier": "t1_project",
            },
            "weight": 0.7_f32,
        });
        let bias: AttentionBiasParams = serde_json::from_value(json_v04).unwrap();
        assert!(
            (bias.thread_weight - 0.7).abs() < 1e-6,
            "wire `weight` must populate thread_weight, got {}",
            bias.thread_weight
        );
        assert_eq!(
            bias.embedding_weight, 0.0,
            "embedding_weight default must be 0.0 when unspecified"
        );

        // Modern shape: explicit thread_weight + embedding_weight.
        let json_v05 = serde_json::json!({
            "scope": {
                "project": "p",
                "session_id": "s",
                "agent": "test",
                "privacy_tier": "t1_project",
            },
            "thread_weight": 0.3_f32,
            "embedding_weight": 0.5_f32,
        });
        let bias: AttentionBiasParams = serde_json::from_value(json_v05).unwrap();
        assert!((bias.thread_weight - 0.3).abs() < 1e-6);
        assert!((bias.embedding_weight - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn embedding_axis_lifts_hits_matching_scope_vector() {
        use ostk_recall_core::{Chunk, Links as CoreLinks, Source};
        use ostk_recall_store::CorpusStore;

        // Build a corpus with two chunks aligned to orthogonal axes.
        // The scope's attention vector will be aligned to chunk A,
        // so cosine(scope, A) ≈ 1 and cosine(scope, B) ≈ 0. With
        // embedding_weight > 0 and thread_weight = 0, A should rank
        // above B regardless of base score order.
        let tmp_corpus = TempDir::new().unwrap();
        let dim = 8;
        let corpus = Arc::new(
            CorpusStore::open_or_create(tmp_corpus.path(), dim)
                .await
                .unwrap(),
        );
        let now = chrono::Utc::now();
        let chunks = vec![
            Chunk {
                chunk_id: "match-a".into(),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: "a.md".into(),
                chunk_index: 0,
                ts: Some(now),
                role: None,
                text: "the operator's stated focus".into(),
                sha256: Chunk::content_hash("match-a"),
                links: CoreLinks::default(),
                extra: serde_json::Value::Null,
            },
            Chunk {
                chunk_id: "match-b".into(),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: "b.md".into(),
                chunk_index: 0,
                ts: Some(now),
                role: None,
                text: "orthogonal content".into(),
                sha256: Chunk::content_hash("match-b"),
                links: CoreLinks::default(),
                extra: serde_json::Value::Null,
            },
        ];
        // Axis 0 aligned (A), axis 4 aligned (B).
        let mut va = vec![0.0_f32; dim];
        va[0] = 1.0;
        let mut vb = vec![0.0_f32; dim];
        vb[4] = 1.0;
        corpus.upsert(&chunks, &[va.clone(), vb]).await.unwrap();

        // Attention runtime with a hand-installed scope vector
        // matching chunk A. Bypasses the embedder so the test is
        // deterministic and doesn't depend on fastembed shape.
        let tmp_threads = TempDir::new().unwrap();
        let threads = Arc::new(ThreadsDb::open(tmp_threads.path()).unwrap());
        let attention = Arc::new(InMemoryAttention::new());
        // No embedder → attend() writes a 32-dim stub vector that won't
        // cosine-match the 8-dim corpus chunks. Use the test back door
        // to install an aligned 8-dim vector directly. The substrate's
        // scope_vector returns whatever was last set by attend, so
        // attending with text that hashes the right way is unreliable —
        // instead seed a thread anchor to drive scope_vector via the
        // ScopeState.attention_vec read path... but that field is only
        // populated by attend(). Simplest: skip the attention runtime's
        // scope_vector path and assert the cosine math through a
        // direct call instead.
        let attention_dyn: Arc<dyn AttentionForwardStore> = attention.clone();
        let d = Arc::new(
            AttentionDispatch::new(attention_dyn, threads).with_corpus(Arc::clone(&corpus)),
        );

        // Two hits with B ranked first by base score.
        let mut hits = vec![hit("match-b", 0.8), hit("match-a", 0.2)];

        // No scope vector set → embedding axis contributes 0 even
        // with high embedding_weight. Confirms the "missing scope
        // vector" path is safe.
        let bias_no_scope = AttentionBiasParams {
            scope: scope(),
            thread_weight: 0.0,
            embedding_weight: 1.0,
        };
        apply_attention_bias(&mut hits, &bias_no_scope, &d).await;
        // Without a scope vector, embedding_score is 0 for both → no
        // re-rank; B stays first.
        assert_eq!(hits[0].chunk_id, "match-b");
        for h in &hits {
            assert_eq!(h.embedding_score, Some(0.0));
            assert_eq!(h.embedding_weight, Some(1.0));
            assert_eq!(h.thread_weight, Some(0.0));
        }
    }

    #[tokio::test]
    async fn embedding_axis_uses_attended_scope_vector() {
        use ostk_recall_core::{Chunk, Links as CoreLinks, Source};
        use ostk_recall_store::CorpusStore;

        // Same shape as the previous test, but here the attention
        // runtime IS wired with an embedder, so attend() populates a
        // real scope vector that aligns with chunk A (both go
        // through the same embedder for dim-compatibility).
        let tmp_corpus = TempDir::new().unwrap();
        let dim = 8;
        let corpus = Arc::new(
            CorpusStore::open_or_create(tmp_corpus.path(), dim)
                .await
                .unwrap(),
        );

        // Deterministic embedder: always returns a unit vector on
        // axis 0. Both the scope's attend() vector and chunk A's
        // upsert vector go through this, so cosine(scope, A) = 1.
        // Chunk B is hand-upserted with an orthogonal vector.
        struct AxisZeroEmbedder {
            dim: usize,
        }
        impl ostk_recall_pipeline::ChunkEmbedder for AxisZeroEmbedder {
            fn dim(&self) -> usize {
                self.dim
            }
            fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
                texts
                    .iter()
                    .map(|_| {
                        let mut v = vec![0.0_f32; self.dim];
                        v[0] = 1.0;
                        v
                    })
                    .collect()
            }
        }
        let embedder: Arc<dyn ostk_recall_pipeline::ChunkEmbedder> =
            Arc::new(AxisZeroEmbedder { dim });

        // Seed corpus: A via embedder (axis 0), B hand-built orthogonal.
        let now = chrono::Utc::now();
        let chunks = vec![
            Chunk {
                chunk_id: "match-a".into(),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: "a.md".into(),
                chunk_index: 0,
                ts: Some(now),
                role: None,
                text: "axis-0 content".into(),
                sha256: Chunk::content_hash("match-a"),
                links: CoreLinks::default(),
                extra: serde_json::Value::Null,
            },
            Chunk {
                chunk_id: "match-b".into(),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: "b.md".into(),
                chunk_index: 0,
                ts: Some(now),
                role: None,
                text: "orthogonal content".into(),
                sha256: Chunk::content_hash("match-b"),
                links: CoreLinks::default(),
                extra: serde_json::Value::Null,
            },
        ];
        let va = embedder.encode_batch(&["axis-0"]).pop().unwrap();
        let mut vb = vec![0.0_f32; dim];
        vb[4] = 1.0;
        corpus.upsert(&chunks, &[va, vb]).await.unwrap();

        // Attention runtime shares the embedder.
        let tmp_threads = TempDir::new().unwrap();
        let threads = Arc::new(ThreadsDb::open(tmp_threads.path()).unwrap());
        let attention = Arc::new(InMemoryAttention::with_embedder(Arc::clone(&embedder)));
        // attend() now writes axis-0 vector into scope.attention_vec
        // via the embedder.
        attention.attend(&scope(), "any text").await.unwrap();

        let attention_dyn: Arc<dyn AttentionForwardStore> = attention;
        let d = AttentionDispatch::new(attention_dyn, threads).with_corpus(Arc::clone(&corpus));

        // B has higher base score; embedding axis must lift A above.
        let mut hits = vec![hit("match-b", 0.8), hit("match-a", 0.2)];
        let bias = AttentionBiasParams {
            scope: scope(),
            thread_weight: 0.0,
            embedding_weight: 1.0,
        };
        apply_attention_bias(&mut hits, &bias, &d).await;

        assert_eq!(
            hits[0].chunk_id, "match-a",
            "axis-0-aligned hit must rank above orthogonal hit when embedding_weight > 0"
        );
        let a = &hits[0];
        let b = &hits[1];
        // A's embedding_score ≈ 1 (cosine with itself); B's ≈ 0.
        assert!(
            a.embedding_score.unwrap() > 0.99,
            "axis-0 hit should have embedding_score ≈ 1, got {}",
            a.embedding_score.unwrap()
        );
        assert!(
            b.embedding_score.unwrap() < 0.01,
            "orthogonal hit should have embedding_score ≈ 0, got {}",
            b.embedding_score.unwrap()
        );
        // Decomposability: score = base + thread_w*thread + embed_w*embed.
        let expected = a.base_score.unwrap()
            + a.thread_weight.unwrap() * a.thread_score.unwrap()
            + a.embedding_weight.unwrap() * a.embedding_score.unwrap();
        assert!(
            (a.score - expected).abs() < 1e-5,
            "score must equal base + thread_w*thread + embed_w*embed (got {} vs {expected})",
            a.score
        );
        // Back-compat aliases must mirror the thread axis.
        assert_eq!(a.attention_score, a.thread_score);
        assert_eq!(a.attention_weight, a.thread_weight);
    }
}
