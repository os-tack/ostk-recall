//! `memory_*` MCP façade — intent-shaped verbs over the substrate.
//!
//! See `.ostk/threads/cognitive-memory/memory-tool-surface.md`. The
//! low-level `recall*` / `attention_*` / `thread_*` tools expose
//! *mechanics*; this surface exposes the *mental acts*: remember this,
//! recall and learn, what's active, make this a concept, connect these,
//! show/correct the memory. Nothing low-level is removed — those tools
//! remain the debug/admin surface.
//!
//! Composition only — every verb delegates to the existing query engine,
//! attention dispatch, and the concept ledger ([`ThreadsDb`] concept
//! methods). The novel piece is the **concept observer** ([`observe_recall`]):
//! a deterministic, conservative pass that turns a recall event into
//! durable, correctable concept objects with mandatory provenance.
//!
//! `memory_recall` and `memory_reflect` are handled in `server.rs` (they
//! need the `QueryEngine` for recall + corpus/ingest reconciliation); the
//! remaining verbs run through [`dispatch_memory`] (ledger + attention only).

use std::collections::HashMap;

use serde_json::{Value, json};

use ostk_recall_attention_mcp::{AttentionDispatch, DefaultAttentionHandlers};
use ostk_recall_core::RecallHit;
use ostk_recall_store::concepts::HitView;
use ostk_recall_store::{
    AliasSource, ConceptActivation, ConceptActivationReader, ConceptStatus, CorpusStore,
    EdgeDirection, EvidenceAttach, ThreadsDb, default_since_now, extract_concept_terms, slugify,
};

use crate::protocol::JsonRpcError;

/// Names of the tools this façade owns.
pub const MEMORY_TOOL_NAMES: &[&str] = &[
    "memory_recall",
    "memory_surface",
    "memory_focus",
    "memory_remember",
    "memory_concept",
    "memory_connect",
    "memory_reflect",
];

#[must_use]
pub fn is_memory_tool(name: &str) -> bool {
    MEMORY_TOOL_NAMES.contains(&name)
}

/// Max concept→concept `pairs_with` edges proposed from a single recall
/// (conservative: only among path-sourced concepts).
const MAX_PROPOSED_EDGES: usize = 8;
const PROPOSED_EDGE_CONFIDENCE: f32 = 0.2;
const OPERATOR_EDGE_CONFIDENCE: f32 = 0.6;
const REFLECT_EVIDENCE_THRESHOLD: usize = 2;
const REFLECT_ALIAS_TOUCH_THRESHOLD: u32 = 3;

// ---------------------------------------------------------------------
// Tool schemas
// ---------------------------------------------------------------------

fn project_prop() -> Value {
    json!({
        "type": "string",
        "description": "Project scope. Omit (or \"\") for a global, cross-project concept; set it to namespace the concept to one project so common handles (auth, client, kernel) don't collide."
    })
}

#[must_use]
pub fn memory_tools() -> Vec<Value> {
    vec![
        json!({
            "name": "memory_recall",
            "description": "Recall and learn. Runs hybrid retrieval, logs the access, applies any active focus pin (returning a lens block), and — when learn=true — observes the query + top hits to update/propose durable concept candidates. Returns hits plus a memory_delta describing what the memory learned.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "project": project_prop(),
                    "source": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
                    "learn": { "type": "boolean", "default": true,
                        "description": "When true, turn this recall into durable concept candidates (low-trust; never affects ranking until promoted)." }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "memory_surface",
            "description": "What is active right now. Working-memory view: current focus pin, active threads, active concepts, and concept proposals awaiting confirmation. Not raw implementation state.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "enum": ["now", "concepts", "open_loops"],
                        "default": "now"
                    },
                    "project": project_prop(),
                    "limit": { "type": "integer", "minimum": 1, "maximum": 200, "default": 50 }
                }
            }
        }),
        json!({
            "name": "memory_focus",
            "description": "Pay attention to this. Pins a focus so subsequent recall/lens is biased toward it. Target is free text OR a concept handle (resolved to its aliases). Delegates to the attention pin under the hood.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": { "type": "string", "description": "Free text or a concept handle." },
                    "project": project_prop()
                },
                "required": ["target"]
            }
        }),
        json!({
            "name": "memory_remember",
            "description": "Remember this durably. kind=concept_seed mints a proposed concept named by `text`. Narrative kinds (note|decision|fact|open_question) attach `text` to an existing concept (require `concept`). Operator-asserted memory always carries provenance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": { "type": "string", "enum": ["concept_seed", "note", "decision", "fact", "open_question"] },
                    "text": { "type": "string" },
                    "concept": { "type": "string", "description": "Target concept handle for narrative kinds." },
                    "project": project_prop()
                },
                "required": ["kind", "text"]
            }
        }),
        json!({
            "name": "memory_concept",
            "description": "Front door for durable concept cards. Actions: show | list | promote | reject | merge | alias | summarize. Concepts are use-driven (candidate→proposed→active) with mandatory evidence; only `active` concepts may bias recall.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["show", "list", "promote", "reject", "merge", "alias", "summarize"] },
                    "handle": { "type": "string" },
                    "project": project_prop(),
                    "status": { "type": "string", "enum": ["candidate", "proposed", "active", "rejected", "merged"], "description": "Filter for action=list." },
                    "to": { "type": "string", "description": "Target status for promote (proposed|active), or canonical handle for merge." },
                    "from": { "type": "string", "description": "Source handle for merge." },
                    "alias": { "type": "string", "description": "Alias text for action=alias." },
                    "source": { "type": "string", "enum": ["query", "path", "symbol", "user", "model"], "default": "user" },
                    "summary": { "type": "string", "description": "Summary text for action=summarize." }
                },
                "required": ["action"]
            }
        }),
        json!({
            "name": "memory_connect",
            "description": "Connect two ideas. Creates a concept→concept relation edge (e.g. mish pairs_with slipstream), creating either endpoint as a candidate if absent. Always records provenance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from": { "type": "string" },
                    "relation": { "type": "string", "description": "e.g. pairs_with, part_of, supersedes, depends_on." },
                    "to": { "type": "string" },
                    "project": project_prop(),
                    "evidence": { "type": "string", "description": "Free-text or JSON provenance note." }
                },
                "required": ["from", "relation", "to"]
            }
        }),
        json!({
            "name": "memory_reflect",
            "description": "Consolidation pass. Promotes candidate concepts that have accumulated corroboration (multi-source evidence or repeated aliases) to `proposed`, re-resolves evidence whose chunk ids churned (re-pointing or orphaning), and reports what changed. Deterministic; does not delete.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
    ]
}

// ---------------------------------------------------------------------
// memory_recall observer (the learning step) — deterministic, conservative
// ---------------------------------------------------------------------

fn hit_symbols(hit: &RecallHit) -> Vec<String> {
    hit.extra
        .get("symbols")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn basename_stem(source_id: &str) -> Option<&str> {
    if !source_id.contains('/') {
        return None;
    }
    let last = source_id.rsplit('/').next()?;
    let stem = last.split('.').next().unwrap_or(last);
    (!stem.is_empty()).then_some(stem)
}

/// Best (highest-score) hit that supports a concept handle. Path/symbol
/// terms resolve to the hit that mentions them; query terms fall back to
/// the top hit. `None` ⇒ no supporting evidence ⇒ the term is dropped
/// (mandatory-provenance junk-guard: a concept must have evidence).
fn supporting_hit<'a>(
    handle: &str,
    source: AliasSource,
    hits: &'a [RecallHit],
) -> Option<&'a RecallHit> {
    match source {
        AliasSource::Path => hits
            .iter()
            .filter(|h| basename_stem(&h.source_id).and_then(slugify).as_deref() == Some(handle))
            .max_by(|a, b| a.score.total_cmp(&b.score)),
        AliasSource::Symbol => hits
            .iter()
            .filter(|h| {
                hit_symbols(h).iter().any(|s| {
                    let tail = s.rsplit("::").next().unwrap_or(s);
                    slugify(tail).as_deref() == Some(handle)
                })
            })
            .max_by(|a, b| a.score.total_cmp(&b.score)),
        _ => hits.first(),
    }
}

/// Observe a recall event: turn the query + top hits into durable concept
/// candidates and return a `memory_delta` JSON object. Deterministic and
/// conservative — candidates are low-trust and never affect ranking.
///
/// Provenance is mandatory: a term with no supporting hit is dropped, so a
/// zero-hit recall mints nothing. Terminal concepts (`rejected`/`merged`)
/// are never re-touched. When `corpus` is available, each evidence row
/// captures the chunk's `content_sha256` + embedding (`anchor_vec`) so the
/// reconciler can re-resolve it after the `chunk_id` churns.
pub async fn observe_recall(
    threads: &ThreadsDb,
    corpus: Option<&CorpusStore>,
    project: &str,
    query: &str,
    query_hash: &str,
    hits: &[RecallHit],
) -> Value {
    let symbols_owned: Vec<Vec<String>> = hits.iter().map(hit_symbols).collect();
    let views: Vec<HitView> = hits
        .iter()
        .zip(&symbols_owned)
        .map(|(h, syms)| HitView {
            source_id: &h.source_id,
            symbols: syms.as_slice(),
        })
        .collect();
    let terms = extract_concept_terms(query, &views);

    // Prefetch chunk sha256 + embedding for the hit set (one batch) so
    // evidence rows can carry change-detection + a re-resolution anchor.
    let chunk_meta: HashMap<String, (String, Option<Vec<f32>>)> = match corpus {
        Some(c) => {
            let ids: Vec<String> = hits.iter().map(|h| h.chunk_id.clone()).collect();
            c.fetch_chunks_by_ids(&ids)
                .await
                .map(|m| {
                    m.into_iter()
                        .map(|(k, (chunk, emb))| (k, (chunk.sha256, emb)))
                        .collect()
                })
                .unwrap_or_default()
        }
        None => HashMap::new(),
    };

    let mut touched = Vec::new();
    let mut created = Vec::new();
    let mut aliases_noticed = Vec::new();
    let mut evidence_attached = Vec::new();
    let mut path_handles: Vec<String> = Vec::new();

    for term in &terms {
        // Mandatory provenance: no supporting hit → no concept.
        let Some(hit) = supporting_hit(&term.handle, term.source, hits) else {
            continue;
        };
        // Terminal-state guard: never re-touch a rejected/merged concept.
        if let Ok(Some(existing)) = threads.get_concept(project, &term.handle) {
            if existing.status.is_terminal() {
                continue;
            }
        }
        let (rec, was_created) = match threads.ensure_concept(
            project,
            &term.handle,
            ConceptStatus::Candidate,
        ) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(handle = %term.handle, error = %e, "ensure_concept failed; skipping");
                continue;
            }
        };
        touched.push(term.handle.clone());
        if was_created {
            created.push(term.handle.clone());
        }
        // Activation: this recall touched the concept. Emit ConceptAccessed
        // so the activation surface (frame + lens concept slot) sees it; the
        // distinct-query salience gate keys on `query_hash`. Best-effort —
        // a failed append never fails the recall.
        if let Err(e) = threads.record_concept_accessed(
            project,
            &term.handle,
            &format!("recall:{}", term.source.as_str()),
            query_hash,
        ) {
            tracing::warn!(handle = %term.handle, error = %e, "ConceptAccessed emit failed");
        }
        if threads
            .touch_alias(rec.id, &term.alias, term.source)
            .is_ok()
        {
            aliases_noticed.push(json!({
                "concept": term.handle, "alias": term.alias, "source": term.source.as_str(),
            }));
        }
        if matches!(term.source, AliasSource::Path) {
            path_handles.push(term.handle.clone());
        }
        let (sha, vec) = chunk_meta
            .get(&hit.chunk_id)
            .map_or((None, None), |(s, v)| (Some(s.as_str()), v.as_deref()));
        let reason = format!("recall:{}", term.source.as_str());
        if threads
            .attach_concept_evidence(&EvidenceAttach {
                concept_id: rec.id,
                project,
                source: &hit.source,
                source_id: &hit.source_id,
                chunk_id: Some(&hit.chunk_id),
                content_sha256: sha,
                anchor_vec: vec,
                score: Some(hit.score),
                reason: Some(&reason),
            })
            .is_ok()
        {
            evidence_attached.push(json!({
                "concept": term.handle, "chunk_id": hit.chunk_id, "score": hit.score,
            }));
        }
    }

    // Conservative relation proposal: only among path-sourced concepts
    // (real co-recalled entities), unordered pairs, capped, low-confidence.
    let mut relations_proposed = Vec::new();
    path_handles.sort();
    path_handles.dedup();
    let top_chunks: Vec<&str> = hits.iter().take(3).map(|h| h.chunk_id.as_str()).collect();
    let evidence_json = json!({ "chunks": top_chunks }).to_string();
    'outer: for i in 0..path_handles.len() {
        for j in (i + 1)..path_handles.len() {
            if relations_proposed.len() >= MAX_PROPOSED_EDGES {
                break 'outer;
            }
            let (from, to) = (&path_handles[i], &path_handles[j]);
            if threads
                .add_concept_edge(
                    project,
                    from,
                    "pairs_with",
                    to,
                    PROPOSED_EDGE_CONFIDENCE,
                    Some(&evidence_json),
                )
                .is_ok()
            {
                relations_proposed
                    .push(json!({ "from": from, "relation": "pairs_with", "to": to }));
            }
        }
    }

    json!({
        "concepts_touched": touched,
        "concepts_created": created,
        "concepts_proposed": Vec::<String>::new(),
        "aliases_noticed": aliases_noticed,
        "relations_proposed": relations_proposed,
        "evidence_attached": evidence_attached,
    })
}

/// Activate already-known concepts a recall resolved, creating none — the
/// `learn=false` path. The hardening discipline is *no model belief becomes
/// memory without provenance*; this mints nothing, only **activates** concepts
/// that already exist (non-terminal). Emits `ConceptAccessed` (reason
/// `recall:known`) for each so the activation surface warms what the recall
/// surfaced, and returns the activated handles for the receipt.
pub fn activate_known_concepts(
    threads: &ThreadsDb,
    project: &str,
    query: &str,
    query_hash: &str,
    hits: &[RecallHit],
) -> Vec<String> {
    let symbols_owned: Vec<Vec<String>> = hits.iter().map(hit_symbols).collect();
    let views: Vec<HitView> = hits
        .iter()
        .zip(&symbols_owned)
        .map(|(h, syms)| HitView {
            source_id: &h.source_id,
            symbols: syms.as_slice(),
        })
        .collect();
    let mut activated = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for term in extract_concept_terms(query, &views) {
        if !seen.insert(term.handle.clone()) {
            continue;
        }
        // Only activate what already exists and is non-terminal — never mint.
        if let Ok(Some(rec)) = threads.get_concept(project, &term.handle) {
            if rec.status.is_terminal() {
                continue;
            }
            if let Err(e) =
                threads.record_concept_accessed(project, &rec.handle, "recall:known", query_hash)
            {
                tracing::warn!(handle = %rec.handle, error = %e, "ConceptAccessed emit failed");
            }
            activated.push(rec.handle);
        }
    }
    activated
}

/// Promote corroborated candidates to `proposed` (the deterministic half
/// of `memory_reflect`; the evidence reconciler is wired in the server
/// where the corpus + ingest handles live). Returns the promoted handles.
pub fn reflect_promote(threads: &ThreadsDb) -> Result<(usize, Vec<String>), JsonRpcError> {
    let candidates = threads
        .list_concepts(None, Some(ConceptStatus::Candidate))
        .map_err(map_store_err)?;
    let mut promoted = Vec::new();
    for c in &candidates {
        let evidence_count = threads
            .list_concept_evidence(c.id)
            .map_err(map_store_err)?
            .iter()
            .filter(|e| e.relation_state == ostk_recall_store::EvidenceState::Active)
            .count();
        let max_alias_touch = threads
            .list_aliases(c.id)
            .map_err(map_store_err)?
            .iter()
            .map(|a| a.touch_count)
            .max()
            .unwrap_or(0);
        if evidence_count >= REFLECT_EVIDENCE_THRESHOLD
            || max_alias_touch >= REFLECT_ALIAS_TOUCH_THRESHOLD
        {
            if threads
                .set_concept_status(&c.project, &c.handle, ConceptStatus::Proposed, Some(0.4))
                .is_ok()
            {
                let _ = threads.record_concept_promoted(&c.project, &c.handle, "proposed");
                promoted.push(c.handle.clone());
            }
        }
    }
    Ok((candidates.len(), promoted))
}

// ---------------------------------------------------------------------
// Dispatch for the threads-only verbs
// ---------------------------------------------------------------------

/// Dispatch `memory_*` verbs other than `memory_recall` / `memory_reflect`
/// (those need the `QueryEngine` and are handled in `server.rs`).
pub async fn dispatch_memory(
    dispatch: &AttentionDispatch,
    name: &str,
    args: Value,
) -> Result<Value, JsonRpcError> {
    let threads = dispatch.threads.as_ref();
    match name {
        "memory_surface" => memory_surface(dispatch, &args).await,
        "memory_focus" => memory_focus(dispatch, args).await,
        "memory_remember" => memory_remember(threads, &args),
        "memory_concept" => memory_concept(threads, &args),
        "memory_connect" => memory_connect(threads, &args),
        other => Err(JsonRpcError::method_not_found(&format!(
            "tools/call/{other}"
        ))),
    }
}

fn str_arg(args: &Value, field: &str) -> Result<String, JsonRpcError> {
    args.get(field)
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| JsonRpcError::invalid_params(format!("missing `{field}`")))
}

/// Project scope from args; "" (global) when absent.
fn project_arg(args: &Value) -> String {
    args.get("project")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn map_store_err(e: ostk_recall_store::StoreError) -> JsonRpcError {
    JsonRpcError::invalid_params(e.to_string())
}

fn concept_to_json(rec: &ostk_recall_store::ConceptRecord) -> Value {
    json!({
        "project": rec.project,
        "handle": rec.handle,
        "summary": rec.summary,
        "status": rec.status.as_str(),
        "confidence": rec.confidence,
        "merged_into": rec.merged_into,
        "created_at": rec.created_at.to_rfc3339(),
        "updated_at": rec.updated_at.to_rfc3339(),
    })
}

/// Render an active concept's activation + decomposed `why` for the frame.
/// The `abi-as-sovereign-boundary` law: the surface ranks, but must expose
/// the math — every field that moved `activation` is shown so the operator
/// can argue with it, not just trust it. `next_actions` are the correction
/// handles the frame always offers.
fn activation_to_json(a: &ConceptActivation) -> Value {
    json!({
        "project": a.project,
        "handle": a.handle,
        "activation": a.activation,
        "why": {
            "confidence": a.why.confidence,
            "decayed_access": a.why.decayed_access,
            "focus_lift": a.why.focus_lift,
            "edge_lift": a.why.edge_lift,
            "note_recency": a.why.note_recency,
            "distinct_queries": a.why.distinct_queries,
            "distinct_sources": a.why.distinct_sources,
            "time_since_touch_secs": a.why.time_since_touch_secs,
        },
        "next_actions": ["summarize", "promote", "connect", "reject"],
    })
}

fn edge_to_json(e: &ostk_recall_store::ConceptEdge) -> Value {
    json!({
        "from": e.from_handle, "relation": e.relation, "to": e.to_handle,
        "confidence": e.confidence, "evidence": e.evidence_json, "touch_count": e.touch_count,
    })
}

/// Full concept card, resolving `handle` within `project` (global fallback).
fn concept_card(threads: &ThreadsDb, project: &str, handle: &str) -> Result<Value, JsonRpcError> {
    let Some(rec) = threads
        .resolve_concept(project, handle)
        .map_err(map_store_err)?
    else {
        return Err(JsonRpcError::invalid_params(format!(
            "concept `{handle}` not found"
        )));
    };
    let aliases = threads.list_aliases(rec.id).map_err(map_store_err)?;
    let evidence = threads
        .list_concept_evidence(rec.id)
        .map_err(map_store_err)?;
    let edges_from = threads
        .list_concept_edges(&rec.project, &rec.handle, EdgeDirection::From)
        .map_err(map_store_err)?;
    let edges_to = threads
        .list_concept_edges(&rec.project, &rec.handle, EdgeDirection::To)
        .map_err(map_store_err)?;
    let notes = threads.list_concept_notes(rec.id).map_err(map_store_err)?;
    Ok(json!({
        "concept": concept_to_json(&rec),
        "aliases": aliases.iter().map(|a| json!({
            "alias": a.alias, "source": a.source.as_str(),
            "touch_count": a.touch_count, "last_seen_at": a.last_seen_at.to_rfc3339(),
        })).collect::<Vec<_>>(),
        "evidence": evidence.iter().map(|e| json!({
            "source": e.source, "source_id": e.source_id,
            "chunk_id": e.last_resolved_chunk_id, "score": e.score, "reason": e.reason,
            "state": e.relation_state.as_str(), "touch_count": e.touch_count,
        })).collect::<Vec<_>>(),
        "edges_from": edges_from.iter().map(edge_to_json).collect::<Vec<_>>(),
        "edges_to": edges_to.iter().map(edge_to_json).collect::<Vec<_>>(),
        "notes": notes.iter().map(|n| json!({
            "kind": n.kind, "text": n.text, "created_at": n.created_at.to_rfc3339(),
        })).collect::<Vec<_>>(),
    }))
}

async fn memory_surface(dispatch: &AttentionDispatch, args: &Value) -> Result<Value, JsonRpcError> {
    let threads = dispatch.threads.as_ref();
    let view = args.get("view").and_then(Value::as_str).unwrap_or("now");
    let project = project_arg(args);
    // Empty project ⇒ Some("") ⇒ global-only (list_concepts treats Some(p)
    // as "p OR global"); a real project shows its own concepts + globals.
    // This keeps project-scoped concepts (e.g. acme/auth) out of the global
    // working-memory view.
    let project_filter = Some(project.as_str());
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(50usize, |v| usize::try_from(v).unwrap_or(50));
    let cards = |status: ConceptStatus| -> Result<Vec<Value>, JsonRpcError> {
        Ok(threads
            .list_concepts(project_filter, Some(status))
            .map_err(map_store_err)?
            .iter()
            .take(limit)
            .map(concept_to_json)
            .collect())
    };
    match view {
        "concepts" => Ok(json!({
            "view": "concepts",
            "active": cards(ConceptStatus::Active)?,
            "proposed": cards(ConceptStatus::Proposed)?,
            "candidates": cards(ConceptStatus::Candidate)?,
        })),
        "open_loops" => Ok(json!({
            "view": "open_loops",
            "pending_proposals": cards(ConceptStatus::Proposed)?,
        })),
        // "now" — the working-memory snapshot: the live focus pin + active
        // threads + active concepts + proposals awaiting confirmation.
        _ => {
            let scope = ostk_recall_core::attention::AttentionScope::default();
            let focus = dispatch.attention.focus_status(&scope).await.ok().map(|s| {
                json!({
                    "pinned": s.pinned.as_ref().map(|p| p.query.clone()),
                    "history_len": s.history.len(),
                })
            });
            let active_threads: Vec<Value> = threads
                .list_threads(Some(ostk_recall_store::TensionState::Active))
                .map_err(map_store_err)?
                .iter()
                .take(limit)
                .map(|t| json!({ "handle": t.handle.as_str(), "resonance": t.resonance }))
                .collect();
            // Active concepts ranked by activation, each carrying its `why`
            // (the working-memory frame, memory-activation-frame.md). Reads the
            // same activation surface that fuels the lens concept slot.
            let active_concepts: Vec<Value> = threads
                .concept_activations(project_filter, default_since_now())
                .map_err(map_store_err)?
                .iter()
                .take(limit)
                .map(activation_to_json)
                .collect();
            Ok(json!({
                "view": "now",
                "focus": focus,
                "active_threads": active_threads,
                "active_concepts": active_concepts,
                "pending_proposals": cards(ConceptStatus::Proposed)?,
            }))
        }
    }
}

async fn memory_focus(dispatch: &AttentionDispatch, args: Value) -> Result<Value, JsonRpcError> {
    let target = str_arg(&args, "target")?;
    let project = project_arg(&args);
    // If the target names a concept, enrich the focus query with its top
    // aliases so the pin's vector is grounded in evidence, not the bare handle.
    let query = match dispatch.threads.resolve_concept(&project, &target) {
        Ok(Some(rec)) => {
            // Activation: focus pinned onto a concept. Best-effort emission.
            if let Err(e) = dispatch
                .threads
                .record_concept_focused(&rec.project, &rec.handle)
            {
                tracing::warn!(handle = %rec.handle, error = %e, "ConceptFocused emit failed");
            }
            let aliases = dispatch.threads.list_aliases(rec.id).unwrap_or_default();
            let alias_str = aliases
                .iter()
                .take(3)
                .map(|a| a.alias.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            if alias_str.is_empty() {
                rec.handle
            } else {
                format!("{}: {alias_str}", rec.handle)
            }
        }
        _ => target,
    };
    let mut focus_args = json!({ "query": query });
    if let Some(scope) = args.get("scope") {
        focus_args["scope"] = scope.clone();
    }
    dispatch
        .dispatch("attention_focus", focus_args)
        .await
        .map_err(|e| JsonRpcError::invalid_params(e.to_string()))
}

fn memory_remember(threads: &ThreadsDb, args: &Value) -> Result<Value, JsonRpcError> {
    let kind = str_arg(args, "kind")?;
    let text = str_arg(args, "text")?;
    let project = project_arg(args);
    match kind.as_str() {
        "concept_seed" => {
            let handle = slugify(&text).ok_or_else(|| {
                JsonRpcError::invalid_params(
                    "concept_seed `text` did not yield a valid handle (need 3-40 alnum chars)",
                )
            })?;
            let (rec, _) = threads
                .ensure_concept(&project, &handle, ConceptStatus::Proposed)
                .map_err(map_store_err)?;
            // Operator-asserted → force at least `proposed` (ensure won't
            // upgrade an existing candidate).
            threads
                .set_concept_status(&project, &handle, ConceptStatus::Proposed, Some(0.5))
                .map_err(map_store_err)?;
            if let Err(e) = threads.record_concept_promoted(&project, &handle, "proposed") {
                tracing::warn!(handle = %handle, error = %e, "ConceptPromoted emit failed");
            }
            threads
                .touch_alias(rec.id, &text, AliasSource::User)
                .map_err(map_store_err)?;
            if rec.summary.is_none() {
                threads.set_concept_summary(&project, &handle, &text).ok();
            }
            Ok(
                json!({ "kind": "concept_seed", "project": project, "concept": handle, "status": "proposed" }),
            )
        }
        "note" | "decision" | "fact" | "open_question" => {
            let concept = args.get("concept").and_then(Value::as_str).ok_or_else(|| {
                JsonRpcError::invalid_params(format!(
                    "kind=`{kind}` requires a `concept` handle in v1 (or use kind=concept_seed)"
                ))
            })?;
            let rec = threads
                .resolve_concept(&project, concept)
                .map_err(map_store_err)?
                .ok_or_else(|| {
                    JsonRpcError::invalid_params(format!("concept `{concept}` not found"))
                })?;
            // Durable, timestamped provenance row — not summary-append.
            let note_id = threads
                .add_concept_note(rec.id, &kind, &text)
                .map_err(map_store_err)?;
            if let Err(e) = threads.record_concept_note_added(&rec.project, &rec.handle, &kind) {
                tracing::warn!(handle = %rec.handle, error = %e, "ConceptNoteAdded emit failed");
            }
            Ok(json!({ "kind": kind, "concept": rec.handle, "note_id": note_id }))
        }
        other => Err(JsonRpcError::invalid_params(format!(
            "unknown remember kind: {other}"
        ))),
    }
}

fn memory_concept(threads: &ThreadsDb, args: &Value) -> Result<Value, JsonRpcError> {
    let action = str_arg(args, "action")?;
    let project = project_arg(args);
    match action.as_str() {
        "show" => {
            let handle = str_arg(args, "handle")?;
            concept_card(threads, &project, &handle)
        }
        "list" => {
            // Some("") ⇒ globals only; Some("acme") ⇒ acme + globals.
            let project_filter = Some(project.as_str());
            let status = args
                .get("status")
                .and_then(Value::as_str)
                .map(ConceptStatus::parse)
                .transpose()
                .map_err(map_store_err)?;
            let rows = threads
                .list_concepts(project_filter, status)
                .map_err(map_store_err)?;
            Ok(json!({ "concepts": rows.iter().map(concept_to_json).collect::<Vec<_>>() }))
        }
        "promote" => {
            let handle = str_arg(args, "handle")?;
            let rec = resolve_or_err(threads, &project, &handle)?;
            let to = args.get("to").and_then(Value::as_str).unwrap_or("active");
            let status = ConceptStatus::parse(to).map_err(map_store_err)?;
            if !matches!(status, ConceptStatus::Proposed | ConceptStatus::Active) {
                return Err(JsonRpcError::invalid_params(
                    "promote `to` must be proposed or active",
                ));
            }
            let confidence = if matches!(status, ConceptStatus::Active) {
                1.0
            } else {
                0.5
            };
            threads
                .set_concept_status(&rec.project, &rec.handle, status, Some(confidence))
                .map_err(map_store_err)?;
            if let Err(e) =
                threads.record_concept_promoted(&rec.project, &rec.handle, status.as_str())
            {
                tracing::warn!(handle = %rec.handle, error = %e, "ConceptPromoted emit failed");
            }
            concept_card(threads, &rec.project, &rec.handle)
        }
        "reject" => {
            let handle = str_arg(args, "handle")?;
            let rec = resolve_or_err(threads, &project, &handle)?;
            threads
                .set_concept_status(
                    &rec.project,
                    &rec.handle,
                    ConceptStatus::Rejected,
                    Some(0.0),
                )
                .map_err(map_store_err)?;
            Ok(json!({ "project": rec.project, "handle": rec.handle, "status": "rejected" }))
        }
        "merge" => {
            let from = str_arg(args, "from")?;
            let into = args
                .get("to")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .or_else(|| {
                    args.get("handle")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                })
                .ok_or_else(|| {
                    JsonRpcError::invalid_params("merge requires `to` (canonical handle)")
                })?;
            threads
                .merge_concept(&project, &from, &into)
                .map_err(map_store_err)?;
            concept_card(threads, &project, &into)
        }
        "alias" => {
            let handle = str_arg(args, "handle")?;
            let alias = str_arg(args, "alias")?;
            let source = args
                .get("source")
                .and_then(Value::as_str)
                .map_or(Ok(AliasSource::User), AliasSource::parse)
                .map_err(map_store_err)?;
            let (rec, _) = threads
                .ensure_concept(&project, &handle, ConceptStatus::Candidate)
                .map_err(map_store_err)?;
            threads
                .touch_alias(rec.id, &alias, source)
                .map_err(map_store_err)?;
            concept_card(threads, &rec.project, &rec.handle)
        }
        "summarize" => {
            let handle = str_arg(args, "handle")?;
            let summary = str_arg(args, "summary")?;
            let rec = resolve_or_err(threads, &project, &handle)?;
            threads
                .set_concept_summary(&rec.project, &rec.handle, &summary)
                .map_err(map_store_err)?;
            concept_card(threads, &rec.project, &rec.handle)
        }
        "split" => Err(JsonRpcError::invalid_params(
            "memory_concept split is deferred to a follow-up; use merge + alias for now",
        )),
        other => Err(JsonRpcError::invalid_params(format!(
            "unknown memory_concept action: {other}"
        ))),
    }
}

fn resolve_or_err(
    threads: &ThreadsDb,
    project: &str,
    handle: &str,
) -> Result<ostk_recall_store::ConceptRecord, JsonRpcError> {
    threads
        .resolve_concept(project, handle)
        .map_err(map_store_err)?
        .ok_or_else(|| JsonRpcError::invalid_params(format!("concept `{handle}` not found")))
}

fn memory_connect(threads: &ThreadsDb, args: &Value) -> Result<Value, JsonRpcError> {
    let from = str_arg(args, "from")?;
    let relation = str_arg(args, "relation")?;
    let to = str_arg(args, "to")?;
    let project = project_arg(args);
    if from == to {
        return Err(JsonRpcError::invalid_params(
            "cannot connect a concept to itself",
        ));
    }
    threads
        .ensure_concept(&project, &from, ConceptStatus::Candidate)
        .map_err(map_store_err)?;
    threads
        .ensure_concept(&project, &to, ConceptStatus::Candidate)
        .map_err(map_store_err)?;
    // Provenance is mandatory: fall back to a self-describing note rather
    // than writing a null evidence edge ("always records provenance").
    let evidence = args
        .get("evidence")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| "operator-asserted via memory_connect".to_string());
    let id = threads
        .add_concept_edge(
            &project,
            &from,
            &relation,
            &to,
            OPERATOR_EDGE_CONFIDENCE,
            Some(&evidence),
        )
        .map_err(map_store_err)?;
    // Activation: a new live association. Best-effort chain emission.
    if let Err(e) = threads.record_concept_connected(&project, &from, &relation, &to) {
        tracing::warn!(error = %e, "ConceptConnected emit failed");
    }
    Ok(json!({ "edge_id": id, "from": from, "relation": relation, "to": to }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn db() -> (TempDir, ThreadsDb) {
        let tmp = TempDir::new().unwrap();
        let db = ThreadsDb::open(tmp.path()).unwrap();
        (tmp, db)
    }

    fn hit(source_id: &str, chunk_id: &str, score: f32, symbols: &[&str]) -> RecallHit {
        serde_json::from_value(json!({
            "chunk_id": chunk_id,
            "source": "code",
            "source_id": source_id,
            "snippet": "…",
            "score": score,
            "links": {},
            "extra": { "symbols": symbols },
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn observe_mints_coordinate_evidence_no_corpus() {
        let (_t, db) = db();
        let hits = vec![
            hit("~/projects/mish", "c-mish", 0.9, &["MishServer"]),
            hit("~/projects/slipstream", "c-slip", 0.8, &[]),
        ];
        let delta = observe_recall(&db, None, "", "mish and slipstream", "qh-1", &hits).await;
        let created: Vec<&str> = delta["concepts_created"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(created.contains(&"mish") && created.contains(&"slipstream"));
        // Evidence keyed on coordinate, not chunk_id.
        let mish = db.resolve_concept("", "mish").unwrap().unwrap();
        let ev = db.list_concept_evidence(mish.id).unwrap();
        assert_eq!(ev[0].source_id, "~/projects/mish");
        assert_eq!(ev[0].last_resolved_chunk_id.as_deref(), Some("c-mish"));
    }

    #[tokio::test]
    async fn observe_drops_unsupported_and_terminal() {
        let (_t, db) = db();
        // Zero hits → no concept (mandatory provenance).
        let delta = observe_recall(&db, None, "", "ghost concept", "qh-2", &[]).await;
        assert!(delta["concepts_created"].as_array().unwrap().is_empty());

        // A rejected concept is not re-touched by observation.
        db.ensure_concept("", "mish", ConceptStatus::Candidate)
            .unwrap();
        db.set_concept_status("", "mish", ConceptStatus::Rejected, None)
            .unwrap();
        let hits = vec![hit("~/projects/mish", "c1", 0.9, &[])];
        let _ = observe_recall(&db, None, "", "mish", "qh-3", &hits).await;
        assert_eq!(
            db.resolve_concept("", "mish").unwrap().unwrap().status,
            ConceptStatus::Rejected
        );
    }

    #[test]
    fn concept_facade_verbs_round_trip() {
        let (_t, db) = db();
        db.ensure_concept("", "mish", ConceptStatus::Candidate)
            .unwrap();
        let promoted = memory_concept(
            &db,
            &json!({"action":"promote","handle":"mish","to":"active"}),
        )
        .unwrap();
        assert_eq!(promoted["concept"]["status"], json!("active"));
        let card = memory_concept(&db, &json!({"action":"show","handle":"mish"})).unwrap();
        assert_eq!(card["concept"]["handle"], json!("mish"));

        let conn = memory_connect(
            &db,
            &json!({"from":"mish","relation":"pairs_with","to":"slipstream"}),
        )
        .unwrap();
        assert!(conn["edge_id"].as_i64().is_some());
    }

    #[test]
    fn remember_concept_seed_scoped() {
        let (_t, db) = db();
        memory_remember(
            &db,
            &json!({"kind":"concept_seed","text":"auth","project":"acme"}),
        )
        .unwrap();
        assert_eq!(
            db.get_concept("acme", "auth").unwrap().unwrap().status,
            ConceptStatus::Proposed
        );
        // Global lookup does not see the project-scoped concept.
        assert!(db.get_concept("", "auth").unwrap().is_none());
    }

    #[test]
    fn reflect_promotes_corroborated() {
        let (_t, db) = db();
        let (rec, _) = db
            .ensure_concept("", "mish", ConceptStatus::Candidate)
            .unwrap();
        for (sid, cid) in [("~/projects/mish", "c1"), ("docs/mish.md", "c2")] {
            db.attach_concept_evidence(&EvidenceAttach {
                concept_id: rec.id,
                project: "",
                source: "code",
                source_id: sid,
                chunk_id: Some(cid),
                content_sha256: None,
                anchor_vec: None,
                score: Some(0.5),
                reason: None,
            })
            .unwrap();
        }
        let (examined, promoted) = reflect_promote(&db).unwrap();
        assert_eq!(examined, 1);
        assert!(promoted.contains(&"mish".to_string()));
    }
}
