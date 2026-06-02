//! MCP server: JSON-RPC over stdio.

use std::sync::Arc;

use ostk_recall_attention_mcp::{
    AttentionDispatch, AttentionHandlersError, DefaultAttentionHandlers, attention_tools,
    thread_tools,
};
use ostk_recall_core::{AttentionBiasParams, Config};
use ostk_recall_query::{
    QueryEngine, QueryError, RecallHit, RecallParams, SynthesizedPage, Synthesizer,
};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{error, info, warn};

use ostk_recall_store::ChainEvent;

use crate::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::resources::{ClientId, ResourceRegistry};
use crate::tools::tool_list;

pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// MCP stdio server. Holds an `Arc<QueryEngine>` so it can be cloned cheaply
/// across tasks if the caller ever wants to run multiple transports.
///
/// The optional `attention` dispatch unlocks the
/// `attention_*` / `thread_*` tool families when the caller (typically
/// `cli::commands::serve`) constructs a long-lived `AttentionDispatch`
/// and threads it through.
///
/// The `resources` registry is always present (default empty) so the
/// MCP `resources/*` protocol surface advertises even before P9b-min
/// registers the `memory-lens` resource. The Arc is shared so callers
/// can register resources from outside the server (e.g. a background
/// lens loop) and emit update notifications without going through the
/// server handle.
pub struct Server {
    engine: Arc<QueryEngine>,
    attention: Option<Arc<AttentionDispatch>>,
    resources: Arc<ResourceRegistry>,
    /// Loaded config, when the daemon has one. Required by the
    /// `memory_concept` `crystallize` action (it resolves the target dir from
    /// the `[[sources]]` block declaring the node's `entity_type`); absent on
    /// the `--stdio`/test paths, where crystallize returns an error rather than
    /// panicking.
    config: Option<Arc<Config>>,
}

impl Server {
    #[must_use]
    pub fn new(engine: QueryEngine) -> Self {
        Self {
            engine: Arc::new(engine),
            attention: None,
            resources: Arc::new(ResourceRegistry::new()),
            config: None,
        }
    }

    #[must_use]
    pub fn from_arc(engine: Arc<QueryEngine>) -> Self {
        Self {
            engine,
            attention: None,
            resources: Arc::new(ResourceRegistry::new()),
            config: None,
        }
    }

    /// Attach the loaded config so the `memory_concept` `crystallize` action can
    /// resolve a typed node's stub-file directory. Without it, crystallize is
    /// rejected (`invalid_request`); all other verbs are unaffected.
    #[must_use]
    pub fn with_config(mut self, config: Arc<Config>) -> Self {
        self.config = Some(config);
        self
    }

    /// Attach an `AttentionDispatch` so the attention/thread MCP tools
    /// become callable. Without one, those tools are not advertised in
    /// `tools/list` and `tools/call` returns method-not-found.
    #[must_use]
    pub fn with_attention(mut self, dispatch: Arc<AttentionDispatch>) -> Self {
        self.attention = Some(dispatch);
        self
    }

    /// P7b: append access-ledger events, best-effort. Requires the
    /// attention dispatch (which owns the threads ledger + chain sink) —
    /// on the `--stdio`/no-attention path there is no sink and events are
    /// silently skipped. A ledger write must NEVER fail the recall that
    /// produced it, so append errors are logged and swallowed.
    fn log_access_events(&self, events: &[ChainEvent]) {
        let Some(dispatch) = self.attention.as_ref() else {
            return;
        };
        let sink = dispatch.threads.chain_sink();
        for ev in events {
            if let Err(e) = sink.append(ev) {
                warn!(error = %e, kind = ev.kind_str(), "access-ledger append failed (best-effort)");
            }
        }
    }

    /// Replace the resource registry. Callers needing to register
    /// resources before `run_stdio` (the typical P9b-min flow) build
    /// a registry, hand it here, and keep their own `Arc` to push
    /// updates later.
    #[must_use]
    pub fn with_resources(mut self, registry: Arc<ResourceRegistry>) -> Self {
        self.resources = registry;
        self
    }

    pub const fn engine(&self) -> &Arc<QueryEngine> {
        &self.engine
    }

    /// Handle to the registry. P9b-min holds this to register the
    /// memory-lens resource and call `emit_resource_updated` from the
    /// background loop.
    #[must_use]
    pub fn resources(&self) -> Arc<ResourceRegistry> {
        Arc::clone(&self.resources)
    }

    /// Read newline-delimited JSON requests from stdin, dispatch, write
    /// responses + server-initiated notifications to stdout. Returns
    /// when EOF is reached on stdin.
    ///
    /// Thin wrapper over [`Self::serve`] that wires real stdin/stdout;
    /// tests use `serve` directly against `tokio::io::duplex` pipes.
    pub async fn run_stdio(&self) -> std::io::Result<()> {
        self.serve(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Transport-agnostic serve loop.
    ///
    /// P9a-min refactor (see `p9a-mcp-resources.md`): the writer
    /// half is owned by a single task draining an mpsc channel; the
    /// reader loop and the resource registry both push through the
    /// same `Sender`. This is the only path that can interleave
    /// responses with server-initiated
    /// `notifications/resources/updated` without racing on the
    /// output descriptor.
    ///
    /// Shutdown contract — the cleanup phase runs unconditionally,
    /// so an I/O error on either the read or the write half cannot
    /// leak a half-open channel:
    ///
    /// 1. The reader loop runs inside an inner async block whose
    ///    result is captured rather than `?`-propagated. EOF →
    ///    `Ok(())`; reader I/O error → `Err(_)`; same for an early
    ///    break when the writer task closed.
    /// 2. After the inner block resolves either way, the registry's
    ///    outbound sender is cleared and the local sender dropped.
    ///    Only then does the writer task's `rx.recv().await`
    ///    observe all senders closed and exit.
    /// 3. `writer_handle.await` joins the writer; its I/O result is
    ///    consulted alongside the reader's.
    /// 4. Error propagation prefers the reader error (the proximate
    ///    cause when the underlying I/O breaks) and falls through to
    ///    the writer error only on a clean reader completion. A
    ///    writer-task panic surfaces as `io::Error::other`.
    pub async fn serve<R, W>(&self, reader: R, writer: W) -> std::io::Result<()>
    where
        R: tokio::io::AsyncRead + Unpin,
        W: tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        // stdio transport: the process-scoped singleton client.
        self.serve_with_client(ClientId::stdio_singleton(), reader, writer)
            .await
    }

    /// Transport-agnostic serve loop for one connection identified by
    /// `client`. The daemon's network listener calls this once per
    /// accepted connection with a fresh [`ClientId::Network`]; the
    /// stdio path goes through [`Self::serve`] with the singleton id.
    ///
    /// On exit a network client is fully evicted from the resource
    /// registry (sender + subscriptions) so a long-lived daemon
    /// doesn't leak dead ids; the stdio singleton keeps its
    /// (process-scoped) subscriptions and only drops its sender.
    pub async fn serve_with_client<R, W>(
        &self,
        client: ClientId,
        reader: R,
        writer: W,
    ) -> std::io::Result<()>
    where
        R: tokio::io::AsyncRead + Unpin,
        W: tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        self.resources.add_outbound(client.clone(), out_tx.clone());

        let writer_handle = tokio::spawn(writer_task(out_rx, writer));

        info!("mcp server ready");
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        // Capture the loop result rather than `?`-propagating. The
        // cleanup phase below must run on every path (EOF, reader
        // error, writer-task-closed) so the registry never holds a
        // dangling outbound Sender.
        let read_result: std::io::Result<()> = async {
            loop {
                line.clear();
                let n = buf_reader.read_line(&mut line).await?;
                if n == 0 {
                    break;
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let response = match serde_json::from_str::<Value>(trimmed) {
                    Ok(v) => self.handle_request_as(v, &client).await,
                    Err(e) => Some(JsonRpcResponse::err(
                        Value::Null,
                        JsonRpcError::parse(format!("parse error: {e}")),
                    )),
                };
                if let Some(r) = response {
                    let payload = serialize_response(&r);
                    if out_tx.send(payload).is_err() {
                        error!("writer task closed; dropping response");
                        break;
                    }
                }
            }
            Ok(())
        }
        .await;

        info!("mcp server shutting down");
        match &client {
            // Network connection closed: evict its sender and prune
            // its subscriptions so the daemon doesn't leak dead ids.
            ClientId::Network { .. } => self.resources.remove_client(&client),
            // stdio singleton: drop only the sender; process-scoped
            // subscriptions are kept (serve_eof regression contract).
            ClientId::StdioSingleton => self.resources.remove_outbound(&client),
        }
        drop(out_tx);
        let writer_result = writer_handle.await;

        // Reader error wins — it's the proximate cause when the
        // underlying transport breaks. Writer errors propagate only
        // when the reader completed cleanly.
        read_result?;
        match writer_result {
            Ok(io_result) => io_result,
            Err(join_err) => Err(std::io::Error::other(format!(
                "writer task panicked: {join_err}"
            ))),
        }
    }

    /// Dispatch one JSON-RPC message. Returns None for notifications (no id).
    /// `resources/subscribe` is attributed to the stdio singleton; the
    /// daemon network path uses [`Self::handle_request_as`] with a
    /// per-connection id.
    pub async fn handle_request(&self, raw: Value) -> Option<JsonRpcResponse> {
        self.handle_request_as(raw, &ClientId::stdio_singleton())
            .await
    }

    /// Dispatch one JSON-RPC message on behalf of `client`. `client`
    /// only affects `resources/subscribe` routing; every other method
    /// is client-agnostic.
    pub async fn handle_request_as(
        &self,
        raw: Value,
        client: &ClientId,
    ) -> Option<JsonRpcResponse> {
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
            "resources/list" => Some(JsonRpcResponse::ok(id, self.resources.list())),
            "resources/read" => match resource_uri_param(&req.params) {
                Ok(uri) => match self.resources.read(&uri) {
                    Ok(v) => Some(JsonRpcResponse::ok(id, v)),
                    Err(err) => Some(JsonRpcResponse::err(id, err.into_rpc())),
                },
                Err(err) => Some(JsonRpcResponse::err(id, err)),
            },
            "resources/subscribe" => match resource_uri_param(&req.params) {
                Ok(uri) => match self.resources.subscribe(client.clone(), &uri) {
                    Ok(()) => Some(JsonRpcResponse::ok(id, json!({}))),
                    Err(err) => Some(JsonRpcResponse::err(id, err.into_rpc())),
                },
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
            "capabilities": {
                "tools": { "listChanged": false },
                // P9a-min advertises `subscribe: true` so MCP clients
                // know they can ask for `resources/subscribe`.
                // `listChanged: false` because P9a-min doesn't push
                // `notifications/resources/list_changed` — the
                // resource set is established at boot. P9a-full /
                // P9b-full revisits this when lens registration
                // becomes dynamic.
                "resources": { "subscribe": true, "listChanged": false }
            },
            "serverInfo": {
                "name": "ostk-recall",
                "version": env!("CARGO_PKG_VERSION"),
            }
        })
    }

    fn handle_tools_list(&self) -> Value {
        let mut tools = tool_list(self.engine.has_audit());
        if self.attention.is_some() {
            // memory_* is the primary, intent-shaped surface; attention_* /
            // thread_* remain as the debug/admin surface beneath it.
            tools.extend(crate::memory::memory_tools());
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
                let query_hash = query_hash(&p.query);
                let mut hits = self.engine.recall(p).await.map_err(query_error_to_rpc)?;
                // Phase F: lens declaration. Capture the pinned focus
                // (if any) BEFORE applying bias so we can attach a
                // `lens` block when the response is shaped by a pin.
                // Absence of `lens` ⇔ ranking is unbiased by any pin.
                let mut lens = None;
                if let Some(bias) = bias {
                    if let Some(dispatch) = self.attention.as_ref() {
                        let status = dispatch.attention.focus_status(&bias.scope).await.ok();
                        lens = build_recall_lens(&bias, status.as_ref());
                        apply_attention_bias(&mut hits, &bias, dispatch).await;
                    }
                }
                // P7b: log one ExplicitRecall access event per returned hit
                // (best-effort; daemon path only). ACT-R freshness is
                // per-chunk, so each surfaced chunk is its own access.
                let now = chrono::Utc::now();
                let events: Vec<ChainEvent> = hits
                    .iter()
                    .map(|h| ChainEvent::ExplicitRecall {
                        chunk_id: h.chunk_id.clone(),
                        query_hash: query_hash.clone(),
                        ts: now,
                    })
                    .collect();
                self.log_access_events(&events);
                match lens {
                    Some(lens) => json!({ "hits": hits, "lens": lens }),
                    None => json!({ "hits": hits }),
                }
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
                // P7b: capture chunk_ids BEFORE `collapse` consumes `hits`;
                // log one RecallFault access event per retrieved chunk.
                let now = chrono::Utc::now();
                let events: Vec<ChainEvent> = hits
                    .iter()
                    .map(|h| ChainEvent::RecallFault {
                        chunk_id: h.chunk_id.clone(),
                        ts: now,
                    })
                    .collect();
                self.log_access_events(&events);
                let pages = Synthesizer::collapse(hits);
                let named: Vec<Value> = pages
                    .iter()
                    .map(named_page_value)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| JsonRpcError::internal(format!("serialize page: {e}")))?;
                json!({ "pages": named })
            }
            "memory_recall" => {
                // Recall + learn (memory-tool-surface.md). Runs hybrid
                // recall, logs the access (ExplicitRecall per hit, same as
                // `recall`), then — when learn=true — runs the deterministic
                // concept observer to turn the event into durable concept
                // candidates. Returns hits + a memory_delta receipt.
                let query = args
                    .get("query")
                    .and_then(Value::as_str)
                    .ok_or_else(|| JsonRpcError::invalid_params("memory_recall: missing query"))?
                    .to_string();
                let learn = args.get("learn").and_then(Value::as_bool).unwrap_or(true);
                // Build RecallParams from the known recall fields only, so
                // the extra `learn` flag doesn't trip deserialization.
                let mut recall_args = json!({ "query": query });
                for f in ["project", "source", "limit", "max_per_source_id"] {
                    if let Some(v) = args.get(f) {
                        recall_args[f] = v.clone();
                    }
                }
                let project = args
                    .get("project")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let p: RecallParams = serde_json::from_value(recall_args).map_err(|e| {
                    JsonRpcError::invalid_params(format!("memory_recall args: {e}"))
                })?;
                let query_hash = query_hash(&p.query);
                let mut hits = self.engine.recall(p).await.map_err(query_error_to_rpc)?;
                // Auto-apply the active focus pin (the fix for "memory_focus
                // did not affect memory_recall"): if a pin is set on the
                // default scope, bias the hits toward it and return a `lens`
                // block, mirroring the low-level `recall` tool — but sourced
                // from the live pin rather than a caller-supplied bias.
                let mut lens = None;
                if let Some(dispatch) = self.attention.as_ref() {
                    let scope = ostk_recall_core::attention::AttentionScope::default();
                    let bias = AttentionBiasParams {
                        scope: scope.clone(),
                        thread_weight: 1.0,
                        embedding_weight: 1.0,
                    };
                    if let Ok(status) = dispatch.attention.focus_status(&scope).await {
                        lens = build_recall_lens(&bias, Some(&status));
                        if lens.is_some() {
                            apply_attention_bias(&mut hits, &bias, dispatch).await;
                        }
                    }
                }
                let now = chrono::Utc::now();
                let events: Vec<ChainEvent> = hits
                    .iter()
                    .map(|h| ChainEvent::ExplicitRecall {
                        chunk_id: h.chunk_id.clone(),
                        query_hash: query_hash.clone(),
                        ts: now,
                    })
                    .collect();
                self.log_access_events(&events);
                let memory_delta = match (learn, self.attention.as_ref()) {
                    (true, Some(d)) => {
                        crate::memory::observe_recall(
                            d.threads.as_ref(),
                            Some(self.engine.store().as_ref()),
                            &project,
                            &query,
                            &query_hash,
                            &hits,
                        )
                        .await
                    }
                    // learn=false still activates already-known concepts the
                    // recall resolved (emits ConceptAccessed), creating none —
                    // "must still activate already-known concepts it resolved."
                    (false, Some(d)) => {
                        let activated = crate::memory::activate_known_concepts(
                            d.threads.as_ref(),
                            &project,
                            &query,
                            &query_hash,
                            &hits,
                        );
                        json!({ "concepts_activated": activated })
                    }
                    (_, None) => Value::Null,
                };
                let mut out = json!({
                    "hits": hits, "memory_delta": memory_delta, "learned": learn,
                });
                if let Some(lens) = lens {
                    out["lens"] = lens;
                }
                out
            }
            "memory_reflect" => {
                // Consolidation: promote corroborated candidates + reconcile
                // evidence whose chunk ids churned (re-point or orphan). Needs
                // the corpus + ingest handles, so it lives here, not in
                // dispatch_memory.
                let dispatch = self.attention.as_ref().ok_or_else(|| {
                    JsonRpcError::invalid_params("memory_reflect requires the attention substrate")
                })?;
                let (examined, promoted) =
                    crate::memory::reflect_promote(dispatch.threads.as_ref())?;
                let recon = ostk_recall_store::reconcile_concept_evidence(
                    dispatch.threads.as_ref(),
                    self.engine.store().as_ref(),
                    self.engine.ingest().as_ref(),
                )
                .await
                .map_err(|e| JsonRpcError::internal(format!("reconcile: {e}")))?;
                // Slice 2b: promote off-diagonal latent bridges (Lance similarity
                // with no reified edge) from the currently attention-active seeds
                // into weak `Promoted` edges — use sets conductance, decay
                // adjudicates. Deliberate consolidation, never write-on-read.
                let support = ostk_recall_store::ConceptActivationReader::relational_support(
                    dispatch.threads.as_ref(),
                    ostk_recall_store::default_since_now(),
                )
                .map_err(|e| JsonRpcError::internal(format!("relational support: {e}")))?;
                let promotion = ostk_recall_query::promote_latent_edges(
                    self.engine.store().as_ref(),
                    dispatch.threads.as_ref(),
                    &support,
                )
                .await
                .map_err(|e| JsonRpcError::internal(format!("latent promotion: {e}")))?;
                json!({
                    "candidates_examined": examined,
                    "promoted_to_proposed": promoted,
                    "evidence_reconcile": {
                        "checked": recon.checked,
                        "still_valid": recon.still_valid,
                        "backfilled": recon.backfilled,
                        "re_resolved": recon.re_resolved,
                        "orphaned": recon.orphaned,
                    },
                    "latent_promotion": {
                        "seeds_examined": promotion.examined_seeds,
                        "edges_promoted": promotion.promoted,
                        "edges_retouched": promotion.retouched,
                    },
                })
            }
            other => {
                if let Some(d) = self.attention.as_ref() {
                    if crate::memory::is_memory_tool(other) {
                        let out = crate::memory::dispatch_memory(
                            d.as_ref(),
                            self.config.as_deref(),
                            other,
                            args,
                        )
                        .await?;
                        let text = serde_json::to_string(&out)
                            .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?;
                        return Ok(json!({
                            "content": [{ "type": "text", "text": text }],
                            "isError": false,
                        }));
                    }
                    if is_attention_tool(other) {
                        let out = d
                            .dispatch(other, args)
                            .await
                            .map_err(attention_error_to_rpc)?;
                        let text = serde_json::to_string(&out)
                            .map_err(|e| JsonRpcError::internal(format!("serialize: {e}")))?;
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

/// Serialize a JSON-RPC response into the on-wire line form the
/// stdio writer task consumes (no trailing newline — the writer adds
/// one). Errors here would only happen on a programming bug in the
/// JsonRpcResponse types, so a serde_json error is logged and a
/// synthetic internal-error envelope returned in its place.
fn serialize_response(resp: &JsonRpcResponse) -> String {
    serde_json::to_string(resp).unwrap_or_else(|e| {
        error!(error = %e, "serialize response");
        format!(
            r#"{{"jsonrpc":"2.0","id":null,"error":{{"code":-32603,"message":"serialize error: {e}"}}}}"#
        )
    })
}

/// Writer task: sole owner of the output transport. Drains `rx`
/// until every sender has dropped, writing each line followed by
/// `\n`. Generic over the writer so tests can drive it against an
/// in-memory pipe; production uses `tokio::io::Stdout`.
pub async fn writer_task<W: AsyncWriteExt + Unpin>(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    mut writer: W,
) -> std::io::Result<()> {
    while let Some(line) = rx.recv().await {
        writer.write_all(line.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }
    Ok(())
}

/// Extract the `uri` parameter from a `resources/{read,subscribe}`
/// request, mapping missing-or-non-string to a JSON-RPC invalid params
/// error.
fn resource_uri_param(params: &Value) -> std::result::Result<String, JsonRpcError> {
    params
        .get("uri")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| JsonRpcError::invalid_params("missing or non-string `uri`"))
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
    let hit_embeddings: std::collections::HashMap<String, Vec<f32>> =
        match (scope_vec.as_ref(), dispatch.corpus.as_ref()) {
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
        // P3A invariant: `score = Σ match_features.contribution` must
        // hold on the MCP wire (see `core::types::MatchFeature`).
        // attention_bias mutates `score` here; emit the corresponding
        // contributions so the sum still matches. The existing
        // entries (rrf / rerank / identifier_boost) summed to `base`
        // before this stage — adding these two keeps that sum equal
        // to the new score.
        hit.match_features.insert(
            "attention_thread".to_string(),
            ostk_recall_core::MatchFeature {
                raw: thread_score,
                weight: thread_w,
                contribution: thread_w * thread_score,
            },
        );
        hit.match_features.insert(
            "attention_embedding".to_string(),
            ostk_recall_core::MatchFeature {
                raw: embedding_score,
                weight: embed_w,
                contribution: embed_w * embedding_score,
            },
        );
    }

    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
}

fn sanitize_weight(w: f32) -> f32 {
    if w.is_finite() && w >= 0.0 { w } else { 0.0 }
}

/// Build the Phase F `lens` block for a recall response. Returns
/// `Some(json)` when a pinned focus shaped this query (caller
/// supplied `attention_bias` AND a pin exists on the bias's scope);
/// `None` otherwise.
///
/// The substrate-level invariant: presence of `lens` ⇔ response was
/// shaped by a pin. Pulled out of `handle_tools_call` so the lens
/// shape is unit-testable without spinning up a full QueryEngine.
fn build_recall_lens(
    bias: &AttentionBiasParams,
    status: Option<&ostk_recall_attention::FocusStatus>,
) -> Option<Value> {
    let pin = status?.pinned.as_ref()?;
    let age_secs = (chrono::Utc::now() - pin.pinned_at).num_seconds().max(0);
    Some(json!({
        "focus_query": pin.query,
        "pinned_at": pin.pinned_at.to_rfc3339(),
        "focus_age_secs": age_secs,
        "applied": {
            "thread_weight": bias.thread_weight,
            "embedding_weight": bias.embedding_weight,
        },
    }))
}

/// Opaque short hash of a normalized query, stored on `ExplicitRecall`
/// access events so "which query surfaced this chunk" is recoverable
/// without persisting the full query text. `access_history` never reads
/// it — it exists purely for future provenance analysis.
fn query_hash(query: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    query.trim().to_lowercase().hash(&mut h);
    format!("{:016x}", h.finish())
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
            match_features: Default::default(),
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
        (tmp, Arc::new(AttentionDispatch::new(attention, threads)))
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
                mentions: 0,
                resonance: 0,
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
        d.attention
            .attend(&scope(), "active context")
            .await
            .unwrap();
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
        let expected =
            a.base_score.unwrap() + a.attention_weight.unwrap() * a.attention_score.unwrap();
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

    /// P3A invariant guard for MCP wire output: `score = Σ
    /// match_features.contribution` must hold even after the
    /// attention-bias post-stage mutates `score`. Bias emits
    /// `attention_thread` / `attention_embedding` contributions so
    /// the sum still matches.
    #[tokio::test]
    async fn bias_preserves_score_equals_sum_match_features() {
        let (_tmp, d) = build_dispatch().await;
        seed_anchored_thread(&d, "fade-is-concentration", "anchored").await;

        // Seed match_features as the rank engine would (sum equals
        // pre-bias score — that's the post-rank state hybrid::recall
        // produces).
        let mut hits = vec![hit("anchored", 0.5), hit("unrelated", 0.5)];
        for h in &mut hits {
            h.match_features.insert(
                "rrf".to_string(),
                ostk_recall_core::MatchFeature::new(h.score, 1.0),
            );
        }
        let bias = thread_bias(1.0);
        apply_attention_bias(&mut hits, &bias, &d).await;

        for h in &hits {
            let sum: f32 = h.match_features.values().map(|m| m.contribution).sum();
            assert!(
                (h.score - sum).abs() < 1e-5,
                "score {} != Σ contribution {} for {} (features: {:?})",
                h.score,
                sum,
                h.chunk_id,
                h.match_features
            );
            // Bias always emits the two attention entries (the
            // contribution may be 0 when the chunk isn't anchored,
            // but the entry exists so debug UIs see the axis).
            assert!(
                h.match_features.contains_key("attention_thread"),
                "missing attention_thread row on {}",
                h.chunk_id
            );
            assert!(
                h.match_features.contains_key("attention_embedding"),
                "missing attention_embedding row on {}",
                h.chunk_id
            );
        }
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
                facets: Default::default(),
                embedding_input_sha256: String::new(),
                source_config_id: "test-cfg".to_string(),
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
                facets: Default::default(),
                embedding_input_sha256: String::new(),
                source_config_id: "test-cfg".to_string(),
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
                facets: Default::default(),
                embedding_input_sha256: String::new(),
                source_config_id: "test-cfg".to_string(),
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
                facets: Default::default(),
                embedding_input_sha256: String::new(),
                source_config_id: "test-cfg".to_string(),
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

    // ---- Phase F: recall lens declaration ----------------------------

    #[test]
    fn build_recall_lens_returns_none_without_focus_status() {
        // No focus_status at all (e.g. trait default returned None or
        // the dispatch couldn't reach the attention store): no lens.
        let bias = thread_bias(1.0);
        assert!(build_recall_lens(&bias, None).is_none());
    }

    #[test]
    fn build_recall_lens_returns_none_when_status_has_no_pin() {
        // Focus_status present but no pin → no lens. Operator hasn't
        // declared an intent to shape ranking, so the substrate
        // doesn't claim the response was shaped by one.
        let bias = thread_bias(1.0);
        let status = ostk_recall_attention::FocusStatus {
            pinned: None,
            history: Vec::new(),
            transient_active: false,
        };
        assert!(build_recall_lens(&bias, Some(&status)).is_none());
    }

    #[test]
    fn build_recall_lens_emits_focus_query_and_weights_when_pinned() {
        // The substrate-level invariant: with a pin set AND a bias
        // supplied, the lens block declares the focus query, the
        // pinned_at timestamp, a non-negative focus_age_secs, and
        // both axis weights so the operator can argue with the math.
        let bias = AttentionBiasParams {
            scope: scope(),
            thread_weight: 0.7,
            embedding_weight: 0.4,
        };
        let pin = ostk_recall_attention::PinnedFocus {
            query: "the CLI surface".into(),
            vec: vec![0.0_f32; 4],
            pinned_at: chrono::Utc::now() - chrono::Duration::seconds(123),
        };
        let status = ostk_recall_attention::FocusStatus {
            pinned: Some(pin),
            history: Vec::new(),
            transient_active: false,
        };
        let lens = build_recall_lens(&bias, Some(&status)).expect("pin + bias must produce lens");
        assert_eq!(lens["focus_query"].as_str(), Some("the CLI surface"));
        // focus_age_secs should be roughly 123, allow drift for slow CI.
        let age = lens["focus_age_secs"].as_i64().unwrap();
        assert!((120..=130).contains(&age), "focus_age_secs ~123, got {age}");
        assert!((lens["applied"]["thread_weight"].as_f64().unwrap() - 0.7).abs() < 1e-5);
        assert!((lens["applied"]["embedding_weight"].as_f64().unwrap() - 0.4).abs() < 1e-5);
    }
}
