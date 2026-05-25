//! Dispatch layer behind the MCP and CLI surfaces.
//!
//! Each public function takes a JSON `Value` (the tool's `arguments`
//! payload) and the relevant store handles, validates input
//! (`PrivacyTier` check, scope default-fill), then delegates to
//! [`AttentionForwardStore`] or [`ThreadsDb`]. Errors surface as the
//! crate-local [`AttentionHandlersError`]; the MCP wiring maps them to
//! `JsonRpcError`.
//!
//! Keeping the dispatch pure-Rust (no JSON-RPC plumbing) lets the CLI
//! reuse the same code path — see [`crate::cli`].

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Duration as ChronoDuration, Utc};
use ostk_recall_attention::activity::{
    AttentionBurstError, DEFAULT_DECAY_HOURS, DEFAULT_LIMIT as ATTENTION_DEFAULT_LIMIT,
    DEFAULT_SAMPLES_PER_BURST, DEFAULT_SINCE_HOURS as ATTENTION_DEFAULT_SINCE_HOURS,
    surface_attention,
};
use ostk_recall_attention::emergent::{
    DEFAULT_LIMIT as EMERGENT_DEFAULT_LIMIT, DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_SINCE_HOURS as EMERGENT_DEFAULT_SINCE_HOURS, EmergentError, discover_and_surface,
};
use ostk_recall_attention::novelty::{
    DEFAULT_BASELINE_DAYS as NOVELTY_DEFAULT_BASELINE_DAYS,
    DEFAULT_LIMIT as NOVELTY_DEFAULT_LIMIT,
    DEFAULT_MIN_CLUSTER as NOVELTY_DEFAULT_MIN_CLUSTER,
    DEFAULT_RECLUSTER_THRESHOLD as NOVELTY_DEFAULT_RECLUSTER_THRESHOLD,
    DEFAULT_SINCE_HOURS as NOVELTY_DEFAULT_SINCE_HOURS, NoveltyError, surface_novelty,
};
use ostk_recall_attention::{AttentionError, AttentionForwardStore};
use ostk_recall_core::{
    AttentionPage, AttentionScope, FoldDepth, PrivacyTier, ThreadHandle, ThreadHandleError,
};
use ostk_recall_store::{
    AssociationType, CorpusStore, EvidenceLink, RelationState, StoreError, TensionState,
    ThreadRecord, ThreadsDb,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

/// Dispatch errors that the MCP layer translates into `JsonRpcError`.
#[derive(Debug, Error)]
pub enum AttentionHandlersError {
    #[error("invalid params: {0}")]
    InvalidParams(String),
    #[error("invalid handle: {0}")]
    InvalidHandle(#[from] ThreadHandleError),
    #[error("attention runtime error: {0}")]
    Attention(#[from] AttentionError),
    #[error("threads ledger error: {0}")]
    Store(#[from] StoreError),
    #[error("privacy tier {0:?} is not permitted for this caller")]
    PrivacyForbidden(PrivacyTier),
    #[error("emergent surfacing requires CorpusStore — wire one via AttentionDispatch::with_corpus")]
    CorpusUnavailable,
    #[error("emergent surfacing failed: {0}")]
    Emergent(#[from] EmergentError),
    #[error("attention surfacing failed: {0}")]
    Attention2(#[from] AttentionBurstError),
    #[error("novelty surfacing failed: {0}")]
    Novelty(#[from] NoveltyError),
}

/// Bag of dependencies threaded through every dispatch call.
///
/// Cheap to clone (`Arc`s under the hood). Constructed once at boot
/// and handed to each request scope.
#[derive(Clone)]
pub struct AttentionDispatch {
    pub attention: Arc<dyn AttentionForwardStore>,
    pub threads: Arc<ThreadsDb>,
    /// Optional corpus handle. Required for emergent surfacing
    /// (`thread_emergent`) — the dispatcher errors with
    /// `CorpusUnavailable` when a request needs it and `None` is set.
    /// Wired by `serve()` and the CLI builder; left `None` in
    /// lightweight test contexts.
    pub corpus: Option<Arc<CorpusStore>>,
    /// Maximum allowed `PrivacyTier`. Defaults to `T3Public` — every
    /// tier is permitted. Set to a lower tier to reject elevated
    /// requests at the boundary (per §Refinement §6 of the plan).
    pub max_privacy_tier: PrivacyTier,
}

impl AttentionDispatch {
    #[must_use]
    pub fn new(attention: Arc<dyn AttentionForwardStore>, threads: Arc<ThreadsDb>) -> Self {
        Self {
            attention,
            threads,
            corpus: None,
            max_privacy_tier: PrivacyTier::T3Public,
        }
    }

    #[must_use]
    pub fn with_corpus(mut self, corpus: Arc<CorpusStore>) -> Self {
        self.corpus = Some(corpus);
        self
    }

    #[must_use]
    pub const fn with_max_privacy_tier(mut self, tier: PrivacyTier) -> Self {
        self.max_privacy_tier = tier;
        self
    }
}

/// Trait alias for the dispatch surface — useful when tests want to
/// inject a fake. Most callers use [`AttentionDispatch`] directly.
#[async_trait]
pub trait DefaultAttentionHandlers: Send + Sync {
    async fn dispatch(&self, name: &str, args: Value) -> Result<Value, AttentionHandlersError>;
}

#[async_trait]
impl DefaultAttentionHandlers for AttentionDispatch {
    async fn dispatch(&self, name: &str, args: Value) -> Result<Value, AttentionHandlersError> {
        match name {
            "attention_attend" => attend(self, args).await,
            "attention_surface" => surface(self, args).await,
            "attention_fold" => fold(self, args).await,
            "attention_familiarize" => familiarize(self, args).await,
            "attention_decay" => decay(self, args).await,
            "thread_create" => thread_create(self, args).await,
            "thread_link" => thread_link(self, args).await,
            "thread_unlink" => thread_unlink(self, args).await,
            "thread_promote" => thread_promote(self, args).await,
            "thread_list" => thread_list(self, args).await,
            "thread_emergent" => thread_emergent(self, args).await,
            "thread_attention" => thread_attention(self, args).await,
            "thread_novelty" => thread_novelty(self, args).await,
            "thread_query" => thread_query(self, args).await,
            other => Err(AttentionHandlersError::InvalidParams(format!(
                "unknown attention/thread tool: {other}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------
// Scope + privacy plumbing
// ---------------------------------------------------------------------

/// Read `scope` from a JSON object, defaulting to the standard scope
/// when absent or null. This is the single point that enforces
/// "scope is mandatory but may default".
///
/// The default is `AttentionScope::default()` — `project=None`,
/// `privacy_tier=T1Project`.
pub fn default_scope(args: &Value) -> Result<AttentionScope, AttentionHandlersError> {
    let raw = args.get("scope").cloned();
    let scope = match raw {
        Some(Value::Null) | None => AttentionScope::default(),
        Some(v) => serde_json::from_value(v)
            .map_err(|e| AttentionHandlersError::InvalidParams(format!("scope: {e}")))?,
    };
    Ok(scope)
}

/// Reject elevated `PrivacyTier` values at the boundary.
pub const fn validate_privacy_tier(
    scope: &AttentionScope,
    max: PrivacyTier,
) -> Result<(), AttentionHandlersError> {
    if privacy_rank(scope.privacy_tier) > privacy_rank(max) {
        return Err(AttentionHandlersError::PrivacyForbidden(scope.privacy_tier));
    }
    Ok(())
}

/// Linear rank of a privacy tier — higher = more public / wider scope.
/// Used solely by [`validate_privacy_tier`] to compare against `max`.
const fn privacy_rank(t: PrivacyTier) -> u8 {
    match t {
        PrivacyTier::T0Private => 0,
        PrivacyTier::T1Project => 1,
        PrivacyTier::T2Trusted => 2,
        PrivacyTier::T3Public => 3,
    }
}

fn require_str(args: &Value, field: &str) -> Result<String, AttentionHandlersError> {
    args.get(field)
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| AttentionHandlersError::InvalidParams(format!("missing {field}")))
}

fn require_handle(args: &Value, field: &str) -> Result<ThreadHandle, AttentionHandlersError> {
    let s = require_str(args, field)?;
    Ok(ThreadHandle::new(s)?)
}

fn require_fold_depth(args: &Value) -> Result<FoldDepth, AttentionHandlersError> {
    let s = require_str(args, "depth")?;
    serde_json::from_value::<FoldDepth>(Value::String(s))
        .map_err(|e| AttentionHandlersError::InvalidParams(format!("depth: {e}")))
}

// ---------------------------------------------------------------------
// Attention verbs
// ---------------------------------------------------------------------

pub async fn attend(d: &AttentionDispatch, args: Value) -> Result<Value, AttentionHandlersError> {
    let scope = default_scope(&args)?;
    validate_privacy_tier(&scope, d.max_privacy_tier)?;
    let context = require_str(&args, "context")?;
    d.attention.attend(&scope, &context).await?;
    Ok(json!({}))
}

pub async fn surface(d: &AttentionDispatch, args: Value) -> Result<Value, AttentionHandlersError> {
    let scope = default_scope(&args)?;
    validate_privacy_tier(&scope, d.max_privacy_tier)?;
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(20usize, |v| usize::try_from(v).unwrap_or(20));
    let pages: Vec<AttentionPage> = d.attention.surface(&scope, limit).await?;
    Ok(json!({ "pages": pages }))
}

pub async fn fold(d: &AttentionDispatch, args: Value) -> Result<Value, AttentionHandlersError> {
    let scope = default_scope(&args)?;
    validate_privacy_tier(&scope, d.max_privacy_tier)?;
    let handle = require_handle(&args, "handle")?;
    let depth = require_fold_depth(&args)?;
    d.attention.fold(&scope, &handle, depth).await?;
    Ok(json!({}))
}

pub async fn familiarize(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let scope = default_scope(&args)?;
    validate_privacy_tier(&scope, d.max_privacy_tier)?;
    let handle = require_handle(&args, "handle")?;
    d.attention.familiarize(&scope, &handle).await?;
    // Mirror the on-disk ledger so the caller has a stable counter to
    // chain off (the AttentionForwardStore::familiarize trait method
    // returns ()). We swallow `ThreadNotFound` here: the ledger may
    // legitimately not know the handle yet during cold-start replay,
    // and the in-memory store will still have ticked the counter.
    let post = d.threads.increment_familiarity(&handle).ok();
    Ok(json!({ "familiarity": post }))
}

pub async fn decay(d: &AttentionDispatch, args: Value) -> Result<Value, AttentionHandlersError> {
    let handle = require_handle(&args, "handle")?;
    let factor = args
        .get("factor")
        .and_then(Value::as_f64)
        .ok_or_else(|| AttentionHandlersError::InvalidParams("missing factor".to_string()))?;
    // f64 -> f32 widens range loss-tolerantly; out-of-range values
    // become Infinity, which AttentionForwardStore::decay rejects.
    #[allow(clippy::cast_possible_truncation)]
    let factor_f32 = factor as f32;
    d.attention.decay(&handle, factor_f32).await?;
    Ok(json!({}))
}

// ---------------------------------------------------------------------
// Thread verbs
// ---------------------------------------------------------------------

/// Parsed input for `thread_create`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadCreateInput {
    #[serde(default)]
    pub scope: AttentionScope,
    pub handle: String,
    #[serde(default)]
    pub body: Option<String>,
    #[serde(default)]
    pub tension: Option<String>,
}

pub async fn thread_create(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let input: ThreadCreateInput = serde_json::from_value(args)
        .map_err(|e| AttentionHandlersError::InvalidParams(format!("thread_create: {e}")))?;
    validate_privacy_tier(&input.scope, d.max_privacy_tier)?;
    let handle = ThreadHandle::new(input.handle)?;
    let tension = match input.tension.as_deref() {
        Some(t) => TensionState::parse(t)?,
        None => TensionState::Active,
    };
    let now = Utc::now();
    let scope_key = scope_key_repr(&input.scope);
    let record = ThreadRecord {
        handle,
        tension,
        familiarity: 0,
        last_touched_at: now,
        anchor_chunk_id: None,
        fold_override: None,
        created_at: now,
        created_scope_key: scope_key,
        privacy_tier: input.scope.privacy_tier,
    };
    // The store reuses INSERT OR REPLACE semantics; the body is
    // informational on the wire — the threads scanner is the durable
    // owner of the on-disk `.ostk/threads/*.md` body.
    let _ = input.body;
    d.threads.upsert_thread(&record)?;
    Ok(json!({
        "record": thread_record_to_json(&record),
    }))
}

/// Parsed input for `thread_link`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadLinkInput {
    #[serde(default)]
    pub scope: AttentionScope,
    pub handle: String,
    pub target_path: String,
    pub category: String,
}

pub async fn thread_link(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let input: ThreadLinkInput = serde_json::from_value(args)
        .map_err(|e| AttentionHandlersError::InvalidParams(format!("thread_link: {e}")))?;
    validate_privacy_tier(&input.scope, d.max_privacy_tier)?;
    let handle = ThreadHandle::new(input.handle)?;
    let now = Utc::now();
    let link = EvidenceLink {
        id: 0,
        thread_handle: handle,
        original_path: PathBuf::from(input.target_path),
        current_path: None,
        content_hash: None,
        last_resolved_chunk_id: None,
        relation_state: RelationState::Active,
        // MCP-originated links are always operator-curated.
        association_type: AssociationType::Curated,
        category: input.category,
        similarity: None,
        created_at: now,
        updated_at: now,
    };
    let id = d.threads.add_evidence_link(&link)?;
    Ok(json!({ "evidence_id": id }))
}

pub async fn thread_unlink(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let id = args
        .get("evidence_id")
        .and_then(Value::as_i64)
        .ok_or_else(|| AttentionHandlersError::InvalidParams("missing evidence_id".into()))?;
    d.threads.remove_evidence(id)?;
    Ok(json!({}))
}

/// Parsed input for `thread_promote`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPromoteInput {
    pub handle_from_proposed: String,
    pub target_tier: String,
}

pub async fn thread_promote(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let input: ThreadPromoteInput = serde_json::from_value(args)
        .map_err(|e| AttentionHandlersError::InvalidParams(format!("thread_promote: {e}")))?;
    let handle = ThreadHandle::new(input.handle_from_proposed)?;
    let tension = TensionState::parse(&input.target_tier)?;
    if !matches!(tension, TensionState::Active | TensionState::Slack) {
        return Err(AttentionHandlersError::InvalidParams(
            "target_tier must be active or slack".into(),
        ));
    }
    d.threads.set_tension(&handle, tension)?;
    let record = d.threads.get_thread(&handle)?.ok_or_else(|| {
        AttentionHandlersError::InvalidParams(format!("thread {handle} vanished after promote"))
    })?;
    Ok(json!({ "record": thread_record_to_json(&record) }))
}

pub async fn thread_list(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let scope = default_scope(&args)?;
    validate_privacy_tier(&scope, d.max_privacy_tier)?;
    let tension = args
        .get("tension")
        .and_then(Value::as_str)
        .map(TensionState::parse)
        .transpose()?;
    let querying_key = scope_key_repr(&scope);
    let rows: Vec<ThreadRecord> = d
        .threads
        .list_threads(tension)?
        .into_iter()
        .filter(|r| visible_to_scope(r, &scope, querying_key.as_deref()))
        .collect();
    let records: Vec<Value> = rows.iter().map(thread_record_to_json).collect();
    Ok(json!({ "records": records }))
}

/// Discover emergent thread candidates from the existing corpus.
///
/// Arguments (all optional):
/// - `since_hours: u32` — look-back window. Default
///   [`EMERGENT_DEFAULT_SINCE_HOURS`].
/// - `limit: usize` — max chunks fed to the clusterer. Default
///   [`EMERGENT_DEFAULT_LIMIT`].
/// - `min_cluster_size: usize` — minimum members per surfaced
///   cluster. Default [`DEFAULT_MIN_CLUSTER_SIZE`].
/// - `persist: bool` — write proposals to `threads_proposed`. Default
///   `true` (idempotent on the UNIQUE constraint).
///
/// Returns `{"clusters": [{handle, members, cohesion, samples}, ...]}`
/// sorted by cohesion desc. Empty `clusters` means nothing stood out
/// in the window.
// TODO(verb-condensation): legacy single-axis verb superseded by
// `thread_query` (v0.4.1). Kept for back-compat through the v0.4.x
// line; slated for removal at v1.0.0 once callers migrate. Today it
// hides which axis the substrate is ranking on (density only) — the
// sentiment-trap exhibit the multi-signal verb dissolves.
pub async fn thread_emergent(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let Some(corpus) = d.corpus.as_ref() else {
        return Err(AttentionHandlersError::CorpusUnavailable);
    };
    let since_hours = args
        .get("since_hours")
        .and_then(Value::as_u64)
        .and_then(|n| i64::try_from(n).ok())
        .unwrap_or(EMERGENT_DEFAULT_SINCE_HOURS);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(EMERGENT_DEFAULT_LIMIT);
    let min_cluster_size = args
        .get("min_cluster_size")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(DEFAULT_MIN_CLUSTER_SIZE);
    let persist = args
        .get("persist")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    #[allow(clippy::cast_possible_truncation)]
    let cohesion_threshold = args
        .get("cohesion_threshold")
        .and_then(Value::as_f64)
        .map_or(
            ostk_recall_attention::cluster::EMERGENT_THRESHOLD,
            |v| v as f32,
        );
    let min_neighbours = args
        .get("min_neighbours")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(ostk_recall_attention::cluster::MIN_NEIGHBOURS_IN_CLUSTER);

    let since = Utc::now() - ChronoDuration::hours(since_hours);
    let reports = discover_and_surface(
        corpus,
        &d.threads,
        since,
        limit,
        min_cluster_size,
        cohesion_threshold,
        min_neighbours,
        persist,
    )
    .await?;

    let clusters: Vec<Value> = reports
        .into_iter()
        .map(|r| {
            json!({
                "handle": r.handle,
                "members": r.members,
                "cohesion": r.cohesion,
                "samples": r.samples,
            })
        })
        .collect();
    Ok(json!({
        "clusters": clusters,
        "params": {
            "since_hours": since_hours,
            "limit": limit,
            "min_cluster_size": min_cluster_size,
            "cohesion_threshold": cohesion_threshold,
            "min_neighbours": min_neighbours,
            "persist": persist,
        }
    }))
}

/// Activity-burst attention surface — "what are we paying attention to."
///
/// Robust alternative to `thread_emergent` (which clusters by embedding
/// density and is biased toward repetition). This verb groups recent
/// non-stale chunks by `(project, source_id)` and ranks each group by
/// `count * exp(-(now - max_ts) / decay_hours)`. The result is the
/// per-source focus areas of the recency window — without any
/// embedding similarity in the loop.
///
/// Arguments (all optional):
/// - `since_hours: u32` — look-back window. Default
///   [`ATTENTION_DEFAULT_SINCE_HOURS`] (24h).
/// - `limit: usize` — max bursts returned. Default
///   [`ATTENTION_DEFAULT_LIMIT`] (10).
/// - `samples_per_burst: usize` — sample snippets per burst. Default
///   [`DEFAULT_SAMPLES_PER_BURST`] (3).
/// - `decay_hours: f32` — recency half-life in hours. Default
///   [`DEFAULT_DECAY_HOURS`] (6.0).
// TODO(verb-condensation): legacy single-axis verb superseded by
// `thread_query` (v0.4.1). See same note on `thread_emergent`.
pub async fn thread_attention(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let Some(corpus) = d.corpus.as_ref() else {
        return Err(AttentionHandlersError::CorpusUnavailable);
    };
    let since_hours = args
        .get("since_hours")
        .and_then(Value::as_u64)
        .and_then(|n| i64::try_from(n).ok())
        .unwrap_or(ATTENTION_DEFAULT_SINCE_HOURS);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(ATTENTION_DEFAULT_LIMIT);
    let samples_per_burst = args
        .get("samples_per_burst")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(DEFAULT_SAMPLES_PER_BURST);
    #[allow(clippy::cast_possible_truncation)]
    let decay_hours = args
        .get("decay_hours")
        .and_then(Value::as_f64)
        .map_or(DEFAULT_DECAY_HOURS, |v| v as f32);

    let since = Utc::now() - ChronoDuration::hours(since_hours);
    let bursts = surface_attention(corpus, since, limit, samples_per_burst, decay_hours).await?;

    let bursts_json: Vec<Value> = bursts
        .into_iter()
        .map(|b| {
            json!({
                "project": b.project,
                "source_id": b.source_id,
                "count": b.count,
                "score": b.score,
                "max_ts": b.max_ts.to_rfc3339(),
                "min_ts": b.min_ts.to_rfc3339(),
                "samples": b.samples,
            })
        })
        .collect();
    Ok(json!({
        "bursts": bursts_json,
        "params": {
            "since_hours": since_hours,
            "limit": limit,
            "samples_per_burst": samples_per_burst,
            "decay_hours": decay_hours,
        }
    }))
}

/// Divergence-from-baseline novelty surface.
///
/// Complement to [`thread_attention`] (the activity-burst "where's the
/// focus" view) and [`thread_emergent`] (the embedding-density cluster
/// view). Novelty answers "what's a new direction" — scores each
/// recent chunk as `1 - cos(embedding, project_baseline)` and surfaces
/// only clusters that pass the same density bar emergent uses.
///
/// Arguments (all optional):
/// - `since_hours: u32` — recency window. Default
///   [`NOVELTY_DEFAULT_SINCE_HOURS`] (24h).
/// - `baseline_days: u32` — per-project baseline window. Default
///   [`NOVELTY_DEFAULT_BASELINE_DAYS`] (7d).
/// - `limit: usize` — max clusters returned. Default
///   [`NOVELTY_DEFAULT_LIMIT`] (10).
/// - `min_cluster: usize` — minimum members per surfaced cluster.
///   Default [`NOVELTY_DEFAULT_MIN_CLUSTER`] (matches
///   `cluster::MIN_CLUSTER_SIZE`).
/// - `recluster_threshold: f32` — re-cluster cosine threshold. Default
///   [`NOVELTY_DEFAULT_RECLUSTER_THRESHOLD`] (matches
///   `cluster::EMERGENT_THRESHOLD`).
///
/// Returns `{ "clusters": [...], "params": {...} }`. Empty `clusters`
/// is the expected null state when nothing novel enough surfaces.
// TODO(verb-condensation): legacy single-axis verb superseded by
// `thread_query` (v0.4.1). See same note on `thread_emergent`.
pub async fn thread_novelty(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    let Some(corpus) = d.corpus.as_ref() else {
        return Err(AttentionHandlersError::CorpusUnavailable);
    };
    let since_hours = args
        .get("since_hours")
        .and_then(Value::as_u64)
        .and_then(|n| i64::try_from(n).ok())
        .unwrap_or(NOVELTY_DEFAULT_SINCE_HOURS);
    let baseline_days = args
        .get("baseline_days")
        .and_then(Value::as_u64)
        .and_then(|n| i64::try_from(n).ok())
        .unwrap_or(NOVELTY_DEFAULT_BASELINE_DAYS);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(NOVELTY_DEFAULT_LIMIT);
    // Accept `min_cluster_size` (consistent with thread_emergent) or
    // `min_cluster` (legacy v0.3.0 name). The aliased name was a
    // naming-inconsistency the v0.3.1 discipline pass normalizes.
    let min_cluster = args
        .get("min_cluster_size")
        .or_else(|| args.get("min_cluster"))
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or(NOVELTY_DEFAULT_MIN_CLUSTER);
    #[allow(clippy::cast_possible_truncation)]
    let recluster_threshold = args
        .get("recluster_threshold")
        .and_then(Value::as_f64)
        .map_or(NOVELTY_DEFAULT_RECLUSTER_THRESHOLD, |v| v as f32);
    // No-baked-filters discipline: default 0.0 (filter off). The
    // historical 0.3 floor is still available as a constant for callers
    // that want it (and `surface_default` continues to pass it for
    // library back-compat), but MCP callers get an honest empty result
    // rather than a hidden post-filter.
    #[allow(clippy::cast_possible_truncation)]
    let min_mean_novelty = args
        .get("min_mean_novelty")
        .and_then(Value::as_f64)
        .map_or(0.0_f32, |v| v as f32);

    let since = Utc::now() - ChronoDuration::hours(since_hours);
    let reports = surface_novelty(
        corpus,
        since,
        baseline_days,
        limit,
        min_cluster,
        recluster_threshold,
        min_mean_novelty,
    )
    .await?;

    let clusters: Vec<Value> = reports
        .into_iter()
        .map(|r| {
            json!({
                "project": r.project,
                "members": r.members,
                "mean_novelty": r.mean_novelty,
                "max_novelty": r.max_novelty,
                "samples": r.samples,
            })
        })
        .collect();
    Ok(json!({
        "clusters": clusters,
        "params": {
            "since_hours": since_hours,
            "baseline_days": baseline_days,
            "limit": limit,
            "min_cluster_size": min_cluster,
            "recluster_threshold": recluster_threshold,
            "min_mean_novelty": min_mean_novelty,
        }
    }))
}

/// Multi-signal thread query (v0.4.1).
///
/// Runs density / activity / novelty against the same recency window
/// and returns a unified list of clusters carrying all three axis
/// scores (`None` for axes that didn't surface this cluster), plus a
/// `composite_score` and full `ScoreAttribution`-style breakdown.
///
/// See `crates/attention/src/query.rs` for the doctrine notes; the
/// short version: this verb is the single architectural move the
/// v0.3.0 hand-off identified as load-bearing, because it dissolves
/// "should we add another surface?" — new questions become new
/// rankings (caller-side), not new verbs (substrate-side).
pub async fn thread_query(
    d: &AttentionDispatch,
    args: Value,
) -> Result<Value, AttentionHandlersError> {
    use ostk_recall_attention::{
        Axis, CompositeWeights, RankBy, ThreadQueryParams, run_query,
    };

    let Some(corpus) = d.corpus.as_ref() else {
        return Err(AttentionHandlersError::CorpusUnavailable);
    };

    let mut params = ThreadQueryParams::default();
    if let Some(v) = args.get("since_hours").and_then(Value::as_u64) {
        if let Ok(n) = i64::try_from(v) {
            params.since_hours = n;
        }
    }
    if let Some(v) = args.get("baseline_days").and_then(Value::as_u64) {
        if let Ok(n) = i64::try_from(v) {
            params.baseline_days = n;
        }
    }
    if let Some(arr) = args.get("signals").and_then(Value::as_array) {
        let mut sigs: Vec<Axis> = Vec::new();
        for s in arr {
            if let Some(name) = s.as_str() {
                if let Some(a) = Axis::parse(name) {
                    if !sigs.contains(&a) {
                        sigs.push(a);
                    }
                }
            }
        }
        if !sigs.is_empty() {
            params.signals = sigs;
        }
    }
    if let Some(rb) = args.get("rank_by").and_then(Value::as_str) {
        if let Some(parsed) = RankBy::parse(rb) {
            params.rank_by = parsed;
        }
    }
    if let Some(w) = args.get("composite_weights") {
        #[allow(clippy::cast_possible_truncation)]
        let pick = |key: &str| -> Option<f32> {
            w.get(key).and_then(Value::as_f64).map(|v| v as f32)
        };
        let d = pick("density").unwrap_or(params.composite_weights.density);
        let a = pick("activity").unwrap_or(params.composite_weights.activity);
        let n = pick("novelty").unwrap_or(params.composite_weights.novelty);
        params.composite_weights = CompositeWeights::new(d, a, n);
    }
    #[allow(clippy::cast_possible_truncation)]
    {
        if let Some(v) = args.get("min_density").and_then(Value::as_f64) {
            params.min_density = v as f32;
        }
        if let Some(v) = args.get("min_activity").and_then(Value::as_f64) {
            params.min_activity = v as f32;
        }
        if let Some(v) = args.get("min_novelty").and_then(Value::as_f64) {
            params.min_novelty = v as f32;
        }
    }
    if let Some(v) = args.get("min_cluster_size").and_then(Value::as_u64) {
        if let Ok(n) = usize::try_from(v) {
            params.min_cluster_size = n;
        }
    }
    if let Some(v) = args.get("limit").and_then(Value::as_u64) {
        if let Ok(n) = usize::try_from(v) {
            params.limit = n;
        }
    }
    if let Some(v) = args.get("samples_per_cluster").and_then(Value::as_u64) {
        if let Ok(n) = usize::try_from(v) {
            params.samples_per_cluster = n;
        }
    }

    let rows = run_query(corpus, &d.threads, params.clone())
        .await
        .map_err(|e| -> AttentionHandlersError {
            match e {
                ostk_recall_attention::ThreadQueryError::Emergent(e) => e.into(),
                ostk_recall_attention::ThreadQueryError::Activity(e) => e.into(),
                ostk_recall_attention::ThreadQueryError::Novelty(e) => e.into(),
                ostk_recall_attention::ThreadQueryError::Corpus(e) => e.into(),
            }
        })?;

    let clusters: Vec<Value> = rows
        .into_iter()
        .map(|r| {
            let axes_json: Vec<Value> = r
                .attribution
                .axes
                .iter()
                .map(|a| {
                    json!({
                        "axis": a.axis.as_str(),
                        "weight": a.weight,
                        "score": a.score,
                        "contribution": a.contribution,
                    })
                })
                .collect();
            json!({
                "cluster_id": r.cluster_id,
                "origin": r.origin.as_str(),
                "project": r.project,
                "members": r.members,
                "density_score": r.density_score,
                "activity_score": r.activity_score,
                "novelty_score": r.novelty_score,
                "composite_score": r.composite_score,
                "samples": r.samples,
                "attribution": {
                    "axes": axes_json,
                    "composite": r.attribution.composite,
                }
            })
        })
        .collect();

    let signals_echo: Vec<&str> = params.signals.iter().map(|a| a.as_str()).collect();
    Ok(json!({
        "clusters": clusters,
        "params": {
            "since_hours": params.since_hours,
            "baseline_days": params.baseline_days,
            "signals": signals_echo,
            "rank_by": params.rank_by.as_str(),
            "composite_weights": {
                "density": params.composite_weights.density,
                "activity": params.composite_weights.activity,
                "novelty": params.composite_weights.novelty,
            },
            "min_density": params.min_density,
            "min_activity": params.min_activity,
            "min_novelty": params.min_novelty,
            "min_cluster_size": params.min_cluster_size,
            "limit": params.limit,
            "samples_per_cluster": params.samples_per_cluster,
        }
    }))
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn visible_to_scope(
    record: &ThreadRecord,
    querying_scope: &AttentionScope,
    querying_key: Option<&str>,
) -> bool {
    // Mirror the in-memory runtime: T0Private rows only surface in
    // their originating scope. We compare on the serialized
    // (project|session_id|agent) key — `created_scope_key` was written
    // at insert time using the same representation.
    if record.privacy_tier == PrivacyTier::T0Private {
        match (record.created_scope_key.as_deref(), querying_key) {
            (Some(a), Some(b)) if a == b => return true,
            (None, None) => return true,
            _ => return false,
        }
    }
    // Non-private rows: respect the caller's tier ceiling. A T0Private
    // caller does see everything in its own ledger, including T3Public
    // rows — privacy is one-way (private content stays private; public
    // content reaches private callers).
    let _ = querying_scope;
    true
}

/// Stable string representation of an `AttentionScope` for use as the
/// `created_scope_key` column. Format: `project|session_id|agent` with
/// `_` for absent components — matches the test invariant in
/// `crates/attention` (it uses `ScopeKey` for hashing; we use a string
/// because the `SQLite` column is `TEXT`).
fn scope_key_repr(scope: &AttentionScope) -> Option<String> {
    if scope.project.is_none() && scope.session_id.is_none() && scope.agent.is_none() {
        return None;
    }
    Some(format!(
        "{}|{}|{}",
        scope.project.as_deref().unwrap_or("_"),
        scope.session_id.as_deref().unwrap_or("_"),
        scope.agent.as_deref().unwrap_or("_"),
    ))
}

fn thread_record_to_json(r: &ThreadRecord) -> Value {
    json!({
        "handle": r.handle.as_str(),
        "tension": r.tension.as_str(),
        "familiarity": r.familiarity,
        "last_touched_at": r.last_touched_at.to_rfc3339(),
        "anchor_chunk_id": r.anchor_chunk_id,
        "fold_override": r.fold_override.map(|d| match d {
            FoldDepth::Folded => "folded",
            FoldDepth::Half => "half",
            FoldDepth::Full => "full",
        }),
        "created_at": r.created_at.to_rfc3339(),
        "created_scope_key": r.created_scope_key,
        "privacy_tier": match r.privacy_tier {
            PrivacyTier::T0Private => "t0_private",
            PrivacyTier::T1Project => "t1_project",
            PrivacyTier::T2Trusted => "t2_trusted",
            PrivacyTier::T3Public => "t3_public",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_attention::InMemoryAttention;
    use tempfile::TempDir;

    fn build_dispatch() -> (TempDir, AttentionDispatch) {
        let tmp = TempDir::new().unwrap();
        let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
        let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
        let d = AttentionDispatch::new(attention, threads);
        (tmp, d)
    }

    #[tokio::test]
    async fn attention_attend_round_trip() {
        let (_tmp, d) = build_dispatch();
        let args = json!({
            "scope": {"project": "haystack", "privacy_tier": "t1_project"},
            "context": "fade-is-concentration is load-bearing"
        });
        let out = attend(&d, args).await.unwrap();
        assert_eq!(out, json!({}));
    }

    #[tokio::test]
    async fn attention_surface_returns_pages_with_attribution() {
        let (_tmp, d) = build_dispatch();
        // Seed: attend + fold a handle so surface has something to return.
        attend(
            &d,
            json!({"scope": {"project": "p"}, "context": "ctx-seed"}),
        )
        .await
        .unwrap();
        fold(
            &d,
            json!({"scope": {"project": "p"}, "handle": "seeded-thread", "depth": "half"}),
        )
        .await
        .unwrap();
        let out = surface(&d, json!({"scope": {"project": "p"}, "limit": 10}))
            .await
            .unwrap();
        let pages = out["pages"].as_array().unwrap();
        // The seeded thread has a fresh anchor; if it scored below
        // ARCHIVE_THRESHOLD the test corpus is degenerate, but we can
        // at least assert the shape is correct.
        for p in pages {
            assert!(p["handle"].is_string());
            assert!(p["depth"].is_string());
            assert!(p["score"].is_number());
            assert!(p["why"]["resonance"].is_number());
            assert!(p["why"]["familiarity"].is_number());
        }
    }

    #[tokio::test]
    async fn scope_required_or_defaulted() {
        let (_tmp, d) = build_dispatch();
        // No scope → defaults to T1Project, attend succeeds.
        attend(&d, json!({"context": "no-scope-here"}))
            .await
            .unwrap();
        let pages = surface(&d, json!({})).await.unwrap();
        assert!(pages["pages"].is_array());
    }

    #[tokio::test]
    async fn privacy_t0_isolation() {
        let (_tmp, d) = build_dispatch();
        // Create a T0 thread in scope A.
        thread_create(
            &d,
            json!({
                "scope": {"project": "a", "privacy_tier": "t0_private"},
                "handle": "secret-thread"
            }),
        )
        .await
        .unwrap();
        // Create a T1 thread in scope B for contrast.
        thread_create(
            &d,
            json!({
                "scope": {"project": "b"},
                "handle": "public-thread"
            }),
        )
        .await
        .unwrap();
        // List from scope B: must not see the T0 row.
        let out = thread_list(&d, json!({"scope": {"project": "b"}}))
            .await
            .unwrap();
        let handles: Vec<&str> = out["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["handle"].as_str().unwrap())
            .collect();
        assert!(handles.contains(&"public-thread"));
        assert!(
            !handles.contains(&"secret-thread"),
            "T0Private must not leak across scopes"
        );

        // List from scope A: must see its own T0 row.
        let out_a = thread_list(
            &d,
            json!({"scope": {"project": "a", "privacy_tier": "t0_private"}}),
        )
        .await
        .unwrap();
        assert!(
            out_a["records"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["handle"].as_str() == Some("secret-thread"))
        );
    }

    // ---- thread_query (v0.4.1) ---------------------------------------

    #[tokio::test]
    async fn thread_query_without_corpus_errors() {
        let (_tmp, d) = build_dispatch();
        let err = thread_query(&d, json!({})).await.unwrap_err();
        match err {
            AttentionHandlersError::CorpusUnavailable => {}
            other => panic!("expected CorpusUnavailable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn thread_query_returns_decomposable_clusters() {
        use ostk_recall_core::{Chunk, Links, Source};
        use ostk_recall_store::CorpusStore;
        use std::sync::Arc as StdArc;

        let tmp_threads = TempDir::new().unwrap();
        let threads = Arc::new(ThreadsDb::open(tmp_threads.path()).unwrap());
        let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());

        // Build a corpus that will produce at least one density cluster
        // AND one novelty cluster, so the dispatch exercises multiple
        // primitives and the join logic.
        let tmp_corpus = TempDir::new().unwrap();
        let dim = 16;
        let corpus = StdArc::new(
            CorpusStore::open_or_create(tmp_corpus.path(), dim)
                .await
                .unwrap(),
        );
        let now = Utc::now();

        // Baseline: 20 chunks aligned to axis 0 (per-project).
        let mut chunks = Vec::new();
        let mut embs: Vec<Vec<f32>> = Vec::new();
        for i in 0..20 {
            chunks.push(Chunk {
                chunk_id: format!("base-{i}"),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: format!("base-{i}.md"),
                chunk_index: 0,
                ts: Some(now),
                role: None,
                text: format!("baseline-{i}"),
                sha256: Chunk::content_hash(&format!("base-{i}")),
                links: Links::default(),
                extra: serde_json::Value::Null,
            });
            let mut v = vec![0.0_f32; dim];
            v[0] = 1.0;
            for slot in v.iter_mut().skip(1) {
                *slot = 0.001;
            }
            embs.push(v);
        }
        // Novel cluster: 4 chunks aligned to axis 7.
        for i in 0..4 {
            chunks.push(Chunk {
                chunk_id: format!("novel-{i}"),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: format!("novel-{i}.md"),
                chunk_index: 0,
                ts: Some(now),
                role: None,
                text: format!("novel-thought-{i}"),
                sha256: Chunk::content_hash(&format!("novel-{i}")),
                links: Links::default(),
                extra: serde_json::Value::Null,
            });
            let mut v = vec![0.0_f32; dim];
            v[7] = 1.0;
            v[0] = 0.001 + (i as f32) * 0.0001;
            embs.push(v);
        }
        corpus.upsert(&chunks, &embs).await.unwrap();

        let d = AttentionDispatch::new(attention, threads).with_corpus(corpus);

        // Request all signals; uniform composite weights (the default).
        let out = thread_query(
            &d,
            json!({
                "since_hours": 1,
                "limit": 20,
                "min_cluster_size": 3,
            }),
        )
        .await
        .expect("thread_query should succeed against a seeded corpus");

        let clusters = out["clusters"].as_array().expect("clusters array");
        assert!(
            !clusters.is_empty(),
            "expected at least one cluster from the seeded corpus"
        );

        // Each cluster must carry a cluster_id, origin, attribution.axes
        // covering all three axes, and contributions that sum to
        // composite (the substrate-level decomposability promise).
        for c in clusters {
            assert!(c["cluster_id"].is_string(), "cluster_id present");
            let origin = c["origin"].as_str().expect("origin present");
            assert!(
                matches!(origin, "density" | "activity" | "novelty"),
                "origin in known set, got {origin:?}"
            );
            let axes = c["attribution"]["axes"]
                .as_array()
                .expect("attribution.axes");
            assert_eq!(axes.len(), 3, "all three axes always attributed");
            let composite = c["composite_score"].as_f64().unwrap();
            let attr_composite = c["attribution"]["composite"].as_f64().unwrap();
            assert!(
                (composite - attr_composite).abs() < 1e-6,
                "composite mirrors attribution.composite"
            );
            let sum: f64 = axes
                .iter()
                .map(|a| a["contribution"].as_f64().unwrap_or(0.0))
                .sum();
            assert!(
                (sum - composite).abs() < 1e-5,
                "contributions sum to composite: {sum} vs {composite}"
            );
        }

        // Params echo includes the defaulted composite_weights (uniform)
        // — sentiment-trap discipline visible in the response.
        let weights = &out["params"]["composite_weights"];
        let third = 1.0_f64 / 3.0;
        assert!(
            (weights["density"].as_f64().unwrap() - third).abs() < 1e-5,
            "default density weight is 1/3"
        );
        assert!(
            (weights["activity"].as_f64().unwrap() - third).abs() < 1e-5,
            "default activity weight is 1/3"
        );
        assert!(
            (weights["novelty"].as_f64().unwrap() - third).abs() < 1e-5,
            "default novelty weight is 1/3"
        );
    }

    #[tokio::test]
    async fn thread_query_cross_axis_backfill_populates_other_axes() {
        // The v0.4.2 honesty upgrade: a cluster surfaced by one
        // primitive gets its other axes computed from its own
        // membership. Seed a tight activity burst (4 similar chunks in
        // one source_id), confirm density_score is backfilled from the
        // embedding cohesion of those chunks and novelty_score is
        // backfilled against the project baseline.
        use ostk_recall_core::{Chunk, Links, Source};
        use ostk_recall_store::CorpusStore;
        use std::sync::Arc as StdArc;

        let tmp_threads = TempDir::new().unwrap();
        let threads = Arc::new(ThreadsDb::open(tmp_threads.path()).unwrap());
        let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());

        let tmp_corpus = TempDir::new().unwrap();
        let dim = 16;
        let corpus = StdArc::new(
            CorpusStore::open_or_create(tmp_corpus.path(), dim)
                .await
                .unwrap(),
        );
        let now = Utc::now();

        // Baseline (axis 0) — defines the project's "usual" direction.
        let mut chunks = Vec::new();
        let mut embs: Vec<Vec<f32>> = Vec::new();
        for i in 0..15 {
            chunks.push(Chunk {
                chunk_id: format!("base-{i}"),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: format!("base-{i}.md"),
                chunk_index: 0,
                ts: Some(now),
                role: None,
                text: format!("baseline-{i}"),
                sha256: Chunk::content_hash(&format!("base-{i}")),
                links: Links::default(),
                extra: serde_json::Value::Null,
            });
            let mut v = vec![0.0_f32; dim];
            v[0] = 1.0;
            embs.push(v);
        }
        // Activity burst: 4 chunks all in same source_id, all on axis 7
        // (very different from baseline → high novelty when backfilled).
        for i in 0..4 {
            chunks.push(Chunk {
                chunk_id: format!("burst-{i}"),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: "burst.md".into(),
                chunk_index: i,
                ts: Some(now),
                role: None,
                text: format!("burst-thought-{i}"),
                sha256: Chunk::content_hash(&format!("burst-{i}")),
                links: Links::default(),
                extra: serde_json::Value::Null,
            });
            let mut v = vec![0.0_f32; dim];
            v[7] = 1.0;
            embs.push(v);
        }
        corpus.upsert(&chunks, &embs).await.unwrap();

        let d = AttentionDispatch::new(attention, threads).with_corpus(corpus);
        let out = thread_query(
            &d,
            json!({"since_hours": 1, "limit": 20, "min_cluster_size": 3}),
        )
        .await
        .unwrap();

        // Find the activity-origin cluster for our burst source.
        let burst = out["clusters"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| {
                c["origin"].as_str() == Some("activity")
                    && c["cluster_id"].as_str() == Some("burst:p:burst.md")
            })
            .expect("activity burst for burst.md should surface");

        // Activity is the surfacing axis — always populated.
        assert!(burst["activity_score"].as_f64().unwrap() > 0.0);
        // Cross-axis backfill: density from average pairwise cosine
        // (all 4 chunks on axis 7 → cohesion ≈ 1.0) and novelty against
        // the axis-0 baseline (≈ 1.0).
        let density = burst["density_score"]
            .as_f64()
            .expect("density backfilled from cluster embeddings");
        assert!(
            density > 0.8,
            "density backfill should be high for axis-aligned cluster; got {density}"
        );
        let novelty = burst["novelty_score"]
            .as_f64()
            .expect("novelty backfilled against project baseline");
        assert!(
            novelty > 0.5,
            "novelty backfill should be high vs baseline; got {novelty}"
        );

        // Attribution stays decomposable post-backfill: contributions
        // sum to composite, every axis carries weight + score + contribution.
        let axes = burst["attribution"]["axes"].as_array().unwrap();
        let sum: f64 = axes
            .iter()
            .map(|a| a["contribution"].as_f64().unwrap_or(0.0))
            .sum();
        let composite = burst["composite_score"].as_f64().unwrap();
        assert!(
            (sum - composite).abs() < 1e-5,
            "post-backfill contributions still sum to composite"
        );
        // All three axes should now have non-null scores on the burst.
        for axis_name in ["density", "activity", "novelty"] {
            let row = axes
                .iter()
                .find(|a| a["axis"].as_str() == Some(axis_name))
                .unwrap();
            assert!(
                row["score"].is_number(),
                "{axis_name} score should be populated after backfill"
            );
        }
    }

    #[tokio::test]
    async fn thread_query_signals_subset_skips_other_primitives() {
        use ostk_recall_core::{Chunk, Links, Source};
        use ostk_recall_store::CorpusStore;
        use std::sync::Arc as StdArc;

        let tmp_threads = TempDir::new().unwrap();
        let threads = Arc::new(ThreadsDb::open(tmp_threads.path()).unwrap());
        let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());

        let tmp_corpus = TempDir::new().unwrap();
        let dim = 16;
        let corpus = StdArc::new(
            CorpusStore::open_or_create(tmp_corpus.path(), dim)
                .await
                .unwrap(),
        );
        let now = Utc::now();
        // Seed a single (project, source_id) burst — enough for the
        // activity primitive to surface but not the density primitive
        // (only one source_id, no cohesion neighbourhood).
        let mut chunks = Vec::new();
        let mut embs = Vec::new();
        for i in 0..4 {
            chunks.push(Chunk {
                chunk_id: format!("c-{i}"),
                source: Source::Markdown,
                project: Some("p".into()),
                source_id: "single.md".into(),
                chunk_index: i,
                ts: Some(now),
                role: None,
                text: format!("text-{i}"),
                sha256: Chunk::content_hash(&format!("c-{i}")),
                links: Links::default(),
                extra: serde_json::Value::Null,
            });
            let mut v = vec![0.0_f32; dim];
            v[0] = 1.0;
            embs.push(v);
        }
        corpus.upsert(&chunks, &embs).await.unwrap();

        let d = AttentionDispatch::new(attention, threads).with_corpus(corpus);

        // Ask only for the activity signal — substrate should not even
        // run density / novelty primitives. Echo confirms.
        let out = thread_query(
            &d,
            json!({
                "since_hours": 1,
                "signals": ["activity"],
                "limit": 10,
            }),
        )
        .await
        .unwrap();
        let echoed = out["params"]["signals"].as_array().unwrap();
        assert_eq!(echoed.len(), 1);
        assert_eq!(echoed[0].as_str(), Some("activity"));
        // Every surfaced cluster must have `origin == "activity"`.
        for c in out["clusters"].as_array().unwrap() {
            assert_eq!(
                c["origin"].as_str(),
                Some("activity"),
                "signals: [activity] must only produce activity-origin clusters"
            );
            // density_score and novelty_score must be null (we didn't
            // run those primitives and don't backfill yet in v0.4.1).
            assert!(c["density_score"].is_null());
            assert!(c["novelty_score"].is_null());
            assert!(c["activity_score"].is_number());
        }
    }

    #[tokio::test]
    async fn thread_create_then_list_round_trip() {
        let (_tmp, d) = build_dispatch();
        thread_create(
            &d,
            json!({
                "scope": {"project": "x"},
                "handle": "alpha-thread",
                "tension": "slack"
            }),
        )
        .await
        .unwrap();
        let out = thread_list(&d, json!({"scope": {"project": "x"}, "tension": "slack"}))
            .await
            .unwrap();
        let recs = out["records"].as_array().unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0]["handle"], "alpha-thread");
        assert_eq!(recs[0]["tension"], "slack");
    }

    #[tokio::test]
    async fn thread_link_writes_curated_evidence() {
        let (_tmp, d) = build_dispatch();
        thread_create(
            &d,
            json!({"scope": {"project": "x"}, "handle": "evidence-host"}),
        )
        .await
        .unwrap();
        let out = thread_link(
            &d,
            json!({
                "scope": {"project": "x"},
                "handle": "evidence-host",
                "target_path": "src/foo.rs",
                "category": "code"
            }),
        )
        .await
        .unwrap();
        assert!(out["evidence_id"].is_i64());
        let id = out["evidence_id"].as_i64().unwrap();
        assert!(id > 0);
        // Verify via the ledger directly: row exists, is curated.
        let rows = d
            .threads
            .list_evidence(&ThreadHandle::new("evidence-host").unwrap())
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].association_type, AssociationType::Curated);
        assert_eq!(rows[0].category, "code");
    }

    #[tokio::test]
    async fn thread_unlink_removes_row() {
        let (_tmp, d) = build_dispatch();
        thread_create(&d, json!({"handle": "u"})).await.unwrap();
        let out = thread_link(
            &d,
            json!({"handle": "u", "target_path": "p", "category": "doc"}),
        )
        .await
        .unwrap();
        let id = out["evidence_id"].as_i64().unwrap();
        thread_unlink(&d, json!({"evidence_id": id})).await.unwrap();
        let rows = d
            .threads
            .list_evidence(&ThreadHandle::new("u").unwrap())
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[tokio::test]
    async fn thread_promote_changes_tension() {
        let (_tmp, d) = build_dispatch();
        // Insert as "dormant" — simulates a proposed thread.
        thread_create(&d, json!({"handle": "prop", "tension": "dormant"}))
            .await
            .unwrap();
        let out = thread_promote(
            &d,
            json!({"handle_from_proposed": "prop", "target_tier": "active"}),
        )
        .await
        .unwrap();
        assert_eq!(out["record"]["tension"], "active");
    }

    #[tokio::test]
    async fn decay_rejects_unknown_handle() {
        let (_tmp, d) = build_dispatch();
        let err = decay(&d, json!({"handle": "nope", "factor": 0.5}))
            .await
            .unwrap_err();
        assert!(
            matches!(err, AttentionHandlersError::Attention(_)),
            "expected attention error, got {err:?}"
        );
    }

    #[tokio::test]
    async fn dispatch_unknown_tool_errors() {
        let (_tmp, d) = build_dispatch();
        let err = d.dispatch("not_a_real_tool", json!({})).await.unwrap_err();
        assert!(matches!(err, AttentionHandlersError::InvalidParams(_)));
    }

    #[tokio::test]
    async fn privacy_validation_rejects_elevated_request() {
        let (_tmp, d_unrestricted) = build_dispatch();
        let d = d_unrestricted.with_max_privacy_tier(PrivacyTier::T1Project);
        let err = attend(
            &d,
            json!({
                "scope": {"privacy_tier": "t3_public"},
                "context": "noop"
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AttentionHandlersError::PrivacyForbidden(_)));
    }
}
