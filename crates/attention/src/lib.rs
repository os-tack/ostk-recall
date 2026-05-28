//! In-memory `AttentionForwardStore` (Phase 2 of the attention substrate).
//!
//! This crate implements the runtime — scoring math, decay, per-scope
//! state, and an in-memory score tier with rebuild-on-boot. The wire
//! types (`AttentionScope`, `AttentionPage`, `ScoreAttribution`,
//! `FoldDepth`, `ThreadHandle`) live in `ostk-recall-core::attention`.
//!
//! Companion threads:
//! - `three-time-scales` — score tier is in-memory, never persisted
//! - `fade-is-concentration` — fade is the load-bearing primitive
//! - `abi-as-sovereign-boundary` — every page carries `ScoreAttribution`
//!
//! Persistence (chain replay, `SQLite` ledger) is intentionally not in
//! this crate; the store-backed `AttentionForwardStore` impl is a
//! future workpiece. Keeping the runtime abstract makes the math and
//! scope-isolation invariants testable on their own.

pub mod activity;
pub mod cluster;
pub mod curator;
pub mod emergent;
pub mod novelty;
pub mod observer;
pub mod query;
pub mod weaver;

pub use cluster::{EMERGENT_THRESHOLD, EmergentCluster, find_clusters, find_clusters_with};
pub use curator::{CuratorConfig, CuratorError, CuratorTick, IdleCurator, TensionTransition};
pub use observer::{
    ObservationResult, ObserverError, ProposedThreadStub, TurnObserver, ambient_scope_default,
};
pub use query::{
    Axis, AxisAttribution, CompositeWeights, RankBy, ThreadQueryAttribution, ThreadQueryError,
    ThreadQueryParams, ThreadQueryReport, run_query,
};
pub use weaver::{AutoWeaver, ProposedWeave, WeaverError, WeaverOutcome, WeaverThresholds};

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ostk_recall_core::attention::{
    AttentionPage, AttentionScope, AttentionSkipReason, FoldDepth, PrivacyTier, ScoreAttribution,
    ThreadHandle,
};
use ostk_recall_pipeline::ChunkEmbedder;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::RwLock;

// --- scoring constants -------------------------------------------------

/// Decay per day for a freshly-touched thread with familiarity 0.
pub const DECAY_RATE_BASE: f32 = 0.05;
/// Decay per day for a thread with familiarity >= 20.
pub const DECAY_RATE_FAMILIAR: f32 = 0.005;
/// Resonance weight (alpha).
pub const ALPHA: f32 = 1.0;
/// Off-diagonal lift weight (beta).
pub const BETA: f32 = 0.5;
/// Surfacer skips pages with score < this threshold.
pub const ARCHIVE_THRESHOLD: f32 = 0.1;

/// Default EMA update rate for the per-scope `rolling_vec`. `0.3` is
/// the tuning baseline documented in `p6-attention-ema.md`; a future
/// phase exposes it via config. λ ∈ (0, 1]: smaller values make the
/// rolling vector slower to drift; larger values track the latest
/// embedding more eagerly.
pub const DEFAULT_ATTENTION_LAMBDA: f32 = 0.3;

/// Familiarity at which decay flattens to `DECAY_RATE_FAMILIAR`.
pub const FAMILIARITY_SATURATION: u32 = 20;

/// Tension below this counts as "dormant enough" for off-diagonal lift.
pub const OFF_DIAGONAL_TENSION_MAX: f32 = 0.2;
/// Resonance above this counts as "resonant enough" for off-diagonal lift.
pub const OFF_DIAGONAL_RESONANCE_MIN: f32 = 0.7;

/// Linear interpolation between BASE and FAMILIAR over F in `[0, FAMILIARITY_SATURATION]`.
#[must_use]
// familiarity is small (saturates at 20); FAMILIARITY_SATURATION is a const.
// Both are well within f32 mantissa range — precision loss is irrelevant here.
#[allow(clippy::cast_precision_loss)]
pub fn decay_rate(familiarity: u32) -> f32 {
    let f = (familiarity as f32 / FAMILIARITY_SATURATION as f32).min(1.0);
    (DECAY_RATE_FAMILIAR - DECAY_RATE_BASE).mul_add(f, DECAY_RATE_BASE)
}

/// Floor rises from 0.1 to 1.0 as familiarity climbs to `FAMILIARITY_SATURATION`.
#[must_use]
// Same bounded-range justification as `decay_rate`.
#[allow(clippy::cast_precision_loss)]
pub fn familiarity_floor(familiarity: u32) -> f32 {
    0.9f32.mul_add(
        (familiarity as f32 / FAMILIARITY_SATURATION as f32).min(1.0),
        0.1,
    )
}

/// Cosine similarity. Returns 0.0 if either vector is zero-norm or empty,
/// and 0.0 if dimensions mismatch (the safe choice — scope-vs-thread
/// dimension drift should never propagate as a spurious score).
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Gate for the off-diagonal "surprising discovery" quadrant.
///
/// Returns 1.0 when both `tension < OFF_DIAGONAL_TENSION_MAX` and
/// `resonance > OFF_DIAGONAL_RESONANCE_MIN`; 0.0 otherwise. The score
/// formula multiplies this gate by `BETA`, and the attribution carries
/// that already-weighted value so attribution rows sum to `score`
/// cleanly.
#[must_use]
pub fn off_diagonal_lift_gate(tension: f32, resonance: f32) -> f32 {
    if tension < OFF_DIAGONAL_TENSION_MAX && resonance > OFF_DIAGONAL_RESONANCE_MIN {
        1.0
    } else {
        0.0
    }
}

// --- errors ------------------------------------------------------------

/// Outcome of an `attend()` call. The observer-mediated path
/// translates these into chain events; direct test callers may
/// ignore the value.
///
/// P6A only ever returns `Updated`; the multi-signal noise gate
/// that produces `Skipped` lands in P6-full (see
/// `p6-attention-ema.md`).
#[derive(Debug, Clone)]
pub enum AttendOutcome {
    /// The scope's rolling attention vector advanced. `rolling_vec`
    /// is the post-EMA value (L2-normalized); `lambda` is the
    /// runtime's update rate at the time of the call, so chain
    /// replay can reproduce the math even if the default changes.
    Updated { rolling_vec: Vec<f32>, lambda: f32 },
    /// The noise gate (P6-full) rejected the turn. The variant
    /// exists in P6A so the `ChainEvent::AttentionTurnSkipped` wire
    /// shape is stable from day one.
    Skipped { reason: AttentionSkipReason },
}

#[derive(Debug, Error)]
pub enum AttentionError {
    #[error("scope not found: {0:?}")]
    ScopeNotFound(ScopeKey),
    #[error("thread not found: {0}")]
    ThreadNotFound(String),
    #[error("invalid score: {0}")]
    InvalidScore(String),
    /// Refocus was called with a query that isn't in the scope's
    /// focus history. The operator can call `focus()` instead to
    /// pin a fresh query, or `focus_status()` to see what's available
    /// to refocus to.
    #[error("focus history miss: query '{0}' not in scope's history")]
    FocusHistoryMiss(String),
}

// --- scope key ---------------------------------------------------------

/// Stable hash-able projection of `AttentionScope`.
///
/// Two scopes with the same `(project, session_id, agent)` triple share
/// state regardless of privacy tier — tier is enforced at surface time,
/// not at attend time.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ScopeKey {
    pub project: Option<String>,
    pub session_id: Option<String>,
    pub agent: Option<String>,
}

impl From<&AttentionScope> for ScopeKey {
    fn from(scope: &AttentionScope) -> Self {
        Self {
            project: scope.project.clone(),
            session_id: scope.session_id.clone(),
            agent: scope.agent.clone(),
        }
    }
}

// --- trait -------------------------------------------------------------

/// Forward-attention store: every call carries scope; surfacing is
/// scoped and privacy-filtered.
#[async_trait]
pub trait AttentionForwardStore: Send + Sync {
    /// Ingest current conversational / tool context; update the
    /// rolling attention vector for the given scope.
    ///
    /// Returns an [`AttendOutcome`] so the observer-mediated path
    /// can persist [`crate::observer::TurnObserver`]'s chain events
    /// (`RollingVectorSnapshot` on update, `AttentionTurnSkipped` on
    /// noise-gate rejection). Direct callers may discard the value.
    async fn attend(
        &self,
        scope: &AttentionScope,
        context: &str,
    ) -> Result<AttendOutcome, AttentionError>;

    /// Surface pages above `ARCHIVE_THRESHOLD` for the given scope,
    /// honouring `PrivacyTier` rules.
    async fn surface(
        &self,
        scope: &AttentionScope,
        limit: usize,
    ) -> Result<Vec<AttentionPage>, AttentionError>;

    /// Set fold-depth state for a thread handle within the scope.
    async fn fold(
        &self,
        scope: &AttentionScope,
        handle: &ThreadHandle,
        depth: FoldDepth,
    ) -> Result<(), AttentionError>;

    /// Increment familiarity for a handle (called per turn-end).
    async fn familiarize(
        &self,
        scope: &AttentionScope,
        handle: &ThreadHandle,
    ) -> Result<(), AttentionError>;

    /// Apply an explicit fade factor (multiplies the floor).
    async fn decay(&self, handle: &ThreadHandle, factor: f32) -> Result<(), AttentionError>;

    /// Compute the current fade score for a thread, scope-agnostic.
    ///
    /// The idle curator calls this once per thread per tick and compares
    /// the result against configured tension thresholds. Implementations
    /// that maintain per-scope state (the in-memory runtime) return the
    /// maximum score across all scopes containing the handle — a thread
    /// that is "still warm somewhere" should not be archived.
    ///
    /// Default implementation returns 0.0 so callers without a real
    /// scoring runtime fall through to the lowest tension bucket; the
    /// in-memory runtime overrides this.
    async fn score_thread(&self, _handle: &ThreadHandle) -> Result<f32, AttentionError> {
        Ok(0.0)
    }

    /// Current attention vector for the scope. The in-memory runtime
    /// returns the latest `attend(scope, …)` write; stores without
    /// per-scope state return `None`.
    ///
    /// Used by `attention-biased recall` to blend hybrid retrieval
    /// scores with cosine similarity to the caller's current focus.
    /// Default `None` keeps the trait additive for non-state stores.
    async fn scope_vector(
        &self,
        _scope: &AttentionScope,
    ) -> Result<Option<Vec<f32>>, AttentionError> {
        Ok(None)
    }

    // --- focus pin (Phase C/D) ----------------------------------------
    // Default impls fall through to an empty outcome so stores without
    // per-scope state remain trait-compatible; the in-memory runtime
    // overrides each.

    async fn focus(
        &self,
        _scope: &AttentionScope,
        _query: String,
    ) -> Result<FocusOutcome, AttentionError> {
        Ok(FocusOutcome {
            previous: None,
            pinned: None,
            history: Vec::new(),
        })
    }

    async fn refocus(
        &self,
        _scope: &AttentionScope,
        query: String,
    ) -> Result<FocusOutcome, AttentionError> {
        Err(AttentionError::FocusHistoryMiss(query))
    }

    async fn unfocus(&self, _scope: &AttentionScope) -> Result<FocusOutcome, AttentionError> {
        Ok(FocusOutcome {
            previous: None,
            pinned: None,
            history: Vec::new(),
        })
    }

    async fn focus_status(&self, _scope: &AttentionScope) -> Result<FocusStatus, AttentionError> {
        Ok(FocusStatus {
            pinned: None,
            history: Vec::new(),
            transient_active: false,
        })
    }
}

// --- in-memory implementation -----------------------------------------

/// Per-thread state held inside a single scope.
#[derive(Debug, Clone)]
struct ThreadState {
    tension: f32,
    familiarity: u32,
    last_touched_at: DateTime<Utc>,
    depth: FoldDepth,
    /// Operator-applied multiplicative fade (1.0 = no fade).
    fade_multiplier: f32,
    /// Anchor vector — for now identical to the scope's attention vector
    /// at the moment of creation (a stand-in until the threads scanner
    /// supplies real anchors).
    anchor: Vec<f32>,
    /// Scope that created this thread; `T0Private` threads only surface here.
    origin: ScopeKey,
    /// Cached from `origin.privacy_tier` at creation time so cross-scope
    /// surface checks stay O(1) without re-reading scope metadata.
    origin_was_private: bool,
}

/// Operator-pinned focus for a scope. Set via `attention_focus`,
/// rotated via `attention_refocus`, cleared via `attention_unfocus`.
/// While a pin is active it drives all ranking in the scope (surface,
/// recall bias, score_thread, thread_query resonance); the transient
/// per-turn vector continues to be updated by `attend` but is shadowed
/// at read-time so the operator's stated lens stays authoritative.
#[derive(Debug, Clone)]
pub struct PinnedFocus {
    /// Verbatim natural-language input the operator pinned.
    pub query: String,
    /// `embedder.embed(query)` at pin time. Carried verbatim across
    /// rotations so a refocus on the same query is exactly the same
    /// lens — important for stochastic embedders.
    pub vec: Vec<f32>,
    pub pinned_at: DateTime<Utc>,
}

/// Maximum entries kept in `ScopeState::focus_history`. LRU eviction:
/// pushing onto a full ring drops the oldest. 9 holds an arc of
/// focus changes without bloating responses; raise it via config if a
/// session ever needs more (cheap, just a `VecDeque` cap bump).
pub const FOCUS_HISTORY_MAX: usize = 9;

/// Result of any focus-mutating call (`focus` / `refocus` / `unfocus`).
/// Carries the pre- and post-state so the MCP layer can return a
/// fully-described transition to the operator without re-querying.
#[derive(Debug, Clone)]
pub struct FocusOutcome {
    /// The pin that was active *before* this call, if any.
    pub previous: Option<PinnedFocus>,
    /// The pin that is active *after* this call. `None` for
    /// `unfocus`.
    pub pinned: Option<PinnedFocus>,
    /// Snapshot of `focus_history` after the operation, most-recent
    /// first.
    pub history: Vec<PinnedFocus>,
}

/// Read-only focus snapshot returned by `focus_status`. `pinned ==
/// None` AND `transient_active == true` means ranking is being
/// driven by the conversational transient — no explicit lens. Both
/// `pinned == None` AND `transient_active == false` means the scope
/// has never been attended to in this process run.
#[derive(Debug, Clone)]
pub struct FocusStatus {
    pub pinned: Option<PinnedFocus>,
    pub history: Vec<PinnedFocus>,
    pub transient_active: bool,
}

#[derive(Debug, Default)]
struct ScopeState {
    /// Last-turn embedding. `attend()` overwrites this verbatim on
    /// every successful call. Drives no ranking when a pin or a
    /// rolling vector exists; serves as the fallback signal for very
    /// young scopes before the rolling channel is seeded.
    transient_vec: Vec<f32>,
    /// EMA-blended attention vector. `None` until the first
    /// successful `attend()` seeds it; subsequent updates run
    /// `rolling = normalize((1 - lambda) * rolling + lambda * new)`
    /// per `p6-attention-ema.md`. Sits between `pinned_focus` and
    /// `transient_vec` in the [`Self::effective_vec`] priority chain
    /// — the operator's pin always wins, but rolling beats last-turn
    /// once it exists.
    rolling_vec: Option<Vec<f32>>,
    /// Operator-controlled focus. When `Some`, `effective_vec`
    /// returns this vector instead of rolling / transient.
    pinned_focus: Option<PinnedFocus>,
    /// Bounded ring of previously-pinned foci. Front is most-recent;
    /// LRU eviction pops the back when capacity is hit.
    focus_history: std::collections::VecDeque<PinnedFocus>,
    threads: HashMap<ThreadHandle, ThreadState>,
}

impl ScopeState {
    /// Priority chain used by every ranking caller:
    /// `pinned_focus → rolling_vec → transient_vec`.
    ///
    /// - Pinned wins absolutely: the operator's stated lens is
    ///   authoritative.
    /// - When no pin is set, the rolling EMA drives ranking — it
    ///   smooths out one-turn jitter and is the input the first
    ///   active memory lens (P9b-min) reads.
    /// - Transient is the cold-start fallback: before the rolling
    ///   channel has been seeded, the latest embed is the best
    ///   signal available.
    ///
    /// Returns an empty slice when nothing is populated; every
    /// cosine downstream treats an empty slice as 0, so the result
    /// is "no contribution" — same as the pre-rolling behaviour.
    fn effective_vec(&self) -> &[f32] {
        if let Some(pin) = &self.pinned_focus {
            return &pin.vec;
        }
        if let Some(rv) = &self.rolling_vec {
            if !rv.is_empty() {
                return rv;
            }
        }
        &self.transient_vec
    }

    /// Last-turn vector, bypassing the priority chain.
    ///
    /// Reserved for callers that explicitly want "what was the most
    /// recent turn about" rather than "what is the scope's current
    /// stance." No P6A caller uses it; exposed for the eventual
    /// P6-full noise-gate audit case in
    /// `final-corrections-addendum.md` B1.
    #[allow(dead_code)]
    fn transient_vec(&self) -> &[f32] {
        &self.transient_vec
    }
}

/// Compute `normalize((1 - lambda) * prev + lambda * next)`.
///
/// Falls back to `next.to_vec()` when the two vectors disagree on
/// dimension — that case can only arise if the embedder dimension
/// changed mid-run, and the safer move is to reset rolling state
/// rather than blend two incompatible spaces.
fn ema_blend_normalized(prev: &[f32], next: &[f32], lambda: f32) -> Vec<f32> {
    if prev.is_empty() || prev.len() != next.len() {
        return next.to_vec();
    }
    let mut out: Vec<f32> = prev
        .iter()
        .zip(next.iter())
        .map(|(p, n)| (1.0 - lambda).mul_add(*p, lambda * *n))
        .collect();
    let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut out {
            *x /= norm;
        }
    }
    out
}

#[derive(Debug, Default)]
struct Inner {
    scopes: HashMap<ScopeKey, ScopeState>,
}

/// In-memory attention runtime. Cloning is cheap (Arc-shared).
/// Per-thread score breakdown shared by `surface` and `score_thread`.
/// Centralises the formula so the two callers cannot drift.
struct ScoreParts {
    score: f32,
    resonance: f32,
    lift_term: f32,
    dt_secs: u64,
}

#[allow(
    clippy::similar_names,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn compute_score_parts(
    state: &ThreadState,
    attention_vec: &[f32],
    now: DateTime<Utc>,
) -> ScoreParts {
    let resonance = cosine_similarity(&state.anchor, attention_vec);
    let dt_secs = (now - state.last_touched_at).num_seconds().max(0) as u64;
    let dt_days = dt_secs as f32 / 86_400.0;
    let floor = familiarity_floor(state.familiarity) * state.fade_multiplier;
    let decay_term = floor * (-decay_rate(state.familiarity) * dt_days).exp();
    let resonance_term = ALPHA * resonance;
    let lift_term = BETA * off_diagonal_lift_gate(state.tension, resonance);
    let score = decay_term + resonance_term + lift_term;
    ScoreParts {
        score,
        resonance,
        lift_term,
        dt_secs,
    }
}

#[derive(Clone)]
pub struct InMemoryAttention {
    inner: Arc<RwLock<Inner>>,
    /// Optional real embedder. When `Some`, `attend()` and the
    /// re-anchoring path embed text through it; when `None`,
    /// `stub_embed` (32-dim SHA-256) is used. The `None` path stays
    /// supported so unit tests can construct a runtime without
    /// loading a model and so the bootstrap order in
    /// `cli::commands::serve` does not have to put the embedder
    /// first.
    embedder: Option<Arc<dyn ChunkEmbedder>>,
    /// EMA update rate for the per-scope rolling vector. See
    /// [`DEFAULT_ATTENTION_LAMBDA`].
    lambda: f32,
}

impl Default for InMemoryAttention {
    fn default() -> Self {
        Self {
            inner: Arc::default(),
            embedder: None,
            lambda: DEFAULT_ATTENTION_LAMBDA,
        }
    }
}

impl InMemoryAttention {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a real embedder wired in. Production
    /// (`cli::commands::serve`) MUST use this so attention vectors
    /// are dimension-compatible with corpus chunk embeddings —
    /// resonance / cosine ranking is meaningless otherwise. Pass
    /// the *same* `Arc<dyn ChunkEmbedder>` that the
    /// `QueryEngine` was constructed with; the substrate has no
    /// way to detect a mismatch once both are running.
    #[must_use]
    pub fn with_embedder(embedder: Arc<dyn ChunkEmbedder>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner::default())),
            embedder: Some(embedder),
            lambda: DEFAULT_ATTENTION_LAMBDA,
        }
    }

    /// Override the EMA update rate λ. Builder-style so production
    /// wiring stays a single chained expression. Clamped to
    /// `(0.0, 1.0]` so callers can't disable the rolling channel
    /// (λ=0 would freeze it) or overshoot (λ>1 would amplify noise).
    #[must_use]
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        let clamped = lambda.clamp(f32::EPSILON, 1.0);
        self.lambda = clamped;
        self
    }

    /// Current EMA update rate λ. Returned by `AttendOutcome::Updated`
    /// alongside the rolling vector so observers and chain replay see
    /// the same value.
    #[must_use]
    pub const fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Embed `text` through the configured embedder, or fall back
    /// to [`stub_embed`] when no embedder is wired. Centralises the
    /// branch so callers never re-implement it.
    #[must_use]
    pub fn embed(&self, text: &str) -> Vec<f32> {
        if let Some(emb) = &self.embedder {
            let mut out = emb.encode_batch(&[text]);
            out.pop().unwrap_or_else(|| stub_embed(text))
        } else {
            stub_embed(text)
        }
    }

    /// Dimension of vectors produced by [`Self::embed`]. Used by
    /// `cli::commands::serve`'s startup assertion to confirm the
    /// corpus dim matches the attention dim before anything is
    /// served.
    #[must_use]
    pub fn embedder_dim(&self) -> Option<usize> {
        self.embedder.as_ref().map(|e| e.dim())
    }

    /// Install or replace a thread's anchor vector. Used by the
    /// boot-time re-anchoring pass in `cli::commands::serve` to
    /// pull each thread's persistent `anchor_chunk_id` from the
    /// corpus and seed its in-memory anchor with the real
    /// embedder's vector. Without this, threads materialised by
    /// chain replay carry empty anchors (resonance = 0) until
    /// they're touched by a fresh `familiarize` or `fold` — and
    /// even then, the anchor would only reflect the scope's
    /// current attention vector, not the thread's persistent
    /// identity in the corpus.
    ///
    /// The corresponding scope is implied by the caller (replay
    /// scope today; could be the thread's `created_scope_key`
    /// once that's plumbed through). If the thread already has
    /// in-memory state, only the anchor is replaced — tension,
    /// familiarity, last_touched_at, and depth are preserved.
    #[allow(clippy::significant_drop_tightening)]
    pub async fn seed_anchor(
        &self,
        scope: &AttentionScope,
        handle: ThreadHandle,
        anchor: Vec<f32>,
    ) -> Result<(), AttentionError> {
        let key = ScopeKey::from(scope);
        let was_private = scope.privacy_tier == PrivacyTier::T0Private;
        let now = Utc::now();
        let mut inner = self.inner.write().await;
        let scope_state = inner.scopes.entry(key.clone()).or_default();
        scope_state
            .threads
            .entry(handle)
            .and_modify(|t| t.anchor.clone_from(&anchor))
            .or_insert_with(|| ThreadState {
                tension: 0.0,
                familiarity: 0,
                last_touched_at: now,
                depth: FoldDepth::Folded,
                fade_multiplier: 1.0,
                anchor,
                origin: key,
                origin_was_private: was_private,
            });
        Ok(())
    }

    /// Snapshot the per-scope rolling vector.
    ///
    /// Returns the raw EMA channel — not the priority-chain
    /// `effective_vec()`. P9b-min's lens loop reads this to feed
    /// the drift detector (cosine distance against the prior
    /// rolling); `scope_vector()` continues to be the ranking
    /// channel (pin precedence applied).
    pub async fn rolling_vec(
        &self,
        scope: &AttentionScope,
    ) -> Result<Option<Vec<f32>>, AttentionError> {
        let key = ScopeKey::from(scope);
        let inner = self.inner.read().await;
        let Some(state) = inner.scopes.get(&key) else {
            return Ok(None);
        };
        Ok(state.rolling_vec.clone())
    }

    /// Replay-seed the per-scope rolling vector from a chain
    /// snapshot.
    ///
    /// Used by `cli::commands::replay_chain_into_attention` to
    /// restore `rolling_vec` on boot from the most-recent
    /// `ChainEvent::RollingVectorSnapshot` per scope. The vec is
    /// carried verbatim from the chain row — no re-embedding — so
    /// stochastic embedders don't drift across restarts.
    ///
    /// Only `rolling_vec` is touched; `transient_vec` stays empty
    /// until the first post-boot `attend()`. `effective_vec()`'s
    /// priority chain (`pinned → rolling → transient`) then returns
    /// the seeded value as the scope vector until either a pin lands
    /// or the EMA blend advances it.
    #[allow(clippy::significant_drop_tightening)]
    pub async fn seed_rolling_vec(
        &self,
        scope: &AttentionScope,
        rolling: Vec<f32>,
    ) -> Result<(), AttentionError> {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let entry = inner.scopes.entry(key).or_default();
        entry.rolling_vec = Some(rolling);
        Ok(())
    }

    /// Pin a focus query to a scope. If `query` is already in
    /// `focus_history`, the existing entry is **promoted** to the
    /// pin slot — its original `vec` and `pinned_at` are preserved
    /// (merge-on-dedupe; important for stochastic embedders). The
    /// previously-pinned focus, if any, is pushed onto
    /// `focus_history` front; LRU eviction drops the oldest when
    /// the ring overflows [`FOCUS_HISTORY_MAX`].
    ///
    /// Dedupe is by exact `query` string equality. Operators who
    /// want to distinguish two similar foci should vary the wording.
    #[allow(clippy::significant_drop_tightening)]
    pub async fn focus(
        &self,
        scope: &AttentionScope,
        query: String,
    ) -> Result<FocusOutcome, AttentionError> {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let scope_state = inner.scopes.entry(key).or_default();

        // Merge-on-dedupe path: if the query is already in history,
        // take it out (preserving its original vec + pinned_at).
        let existing = scope_state
            .focus_history
            .iter()
            .position(|p| p.query == query)
            .map(|i| {
                scope_state
                    .focus_history
                    .remove(i)
                    .expect("position is valid")
            });

        // Construct the new pin: either the existing history entry,
        // or a fresh one with current timestamp and a freshly-embedded
        // vec. The freshly-embedded path uses self.embed() so it
        // shares the same embedder as attend() — pin and transient
        // are cosine-comparable in the same vector space.
        let new_pin = existing.unwrap_or_else(|| PinnedFocus {
            vec: self.embed(&query),
            query,
            pinned_at: Utc::now(),
        });

        // Demote current pin (if any) to history front; cap to MAX.
        let previous = scope_state.pinned_focus.take();
        if let Some(prev) = previous.clone() {
            scope_state.focus_history.push_front(prev);
            while scope_state.focus_history.len() > FOCUS_HISTORY_MAX {
                scope_state.focus_history.pop_back();
            }
        }
        scope_state.pinned_focus = Some(new_pin.clone());

        Ok(FocusOutcome {
            previous,
            pinned: Some(new_pin),
            history: scope_state.focus_history.iter().cloned().collect(),
        })
    }

    /// Promote a focus from history to the pin slot. Errors with
    /// [`AttentionError::FocusHistoryMiss`] when the query is not
    /// in the scope's `focus_history` — callers wanting "pin this
    /// query whether new or known" should use [`Self::focus`].
    ///
    /// The currently-pinned focus, if any, is pushed to history
    /// front (same eviction rule as `focus()`). Refocus never
    /// embeds — it always reuses the stored vec.
    #[allow(clippy::significant_drop_tightening)]
    pub async fn refocus(
        &self,
        scope: &AttentionScope,
        query: String,
    ) -> Result<FocusOutcome, AttentionError> {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let scope_state = inner.scopes.entry(key).or_default();

        let pos = scope_state
            .focus_history
            .iter()
            .position(|p| p.query == query)
            .ok_or_else(|| AttentionError::FocusHistoryMiss(query.clone()))?;
        let promoted = scope_state
            .focus_history
            .remove(pos)
            .expect("position is valid");

        let previous = scope_state.pinned_focus.take();
        if let Some(prev) = previous.clone() {
            scope_state.focus_history.push_front(prev);
            // No eviction needed: we just removed one entry and added
            // one, so len is unchanged. The cap check is still cheap
            // and guards against future code paths that get this
            // arithmetic wrong.
            while scope_state.focus_history.len() > FOCUS_HISTORY_MAX {
                scope_state.focus_history.pop_back();
            }
        }
        scope_state.pinned_focus = Some(promoted.clone());

        Ok(FocusOutcome {
            previous,
            pinned: Some(promoted),
            history: scope_state.focus_history.iter().cloned().collect(),
        })
    }

    /// Clear the scope's pin. The cleared focus is pushed to
    /// history front (with the same LRU eviction). Idempotent —
    /// unfocusing an already-unfocused scope is a no-op and returns
    /// a `FocusOutcome` with `previous = pinned = None`.
    #[allow(clippy::significant_drop_tightening)]
    pub async fn unfocus(&self, scope: &AttentionScope) -> Result<FocusOutcome, AttentionError> {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let scope_state = inner.scopes.entry(key).or_default();

        let previous = scope_state.pinned_focus.take();
        if let Some(prev) = previous.clone() {
            scope_state.focus_history.push_front(prev);
            while scope_state.focus_history.len() > FOCUS_HISTORY_MAX {
                scope_state.focus_history.pop_back();
            }
        }

        Ok(FocusOutcome {
            previous,
            pinned: None,
            history: scope_state.focus_history.iter().cloned().collect(),
        })
    }

    /// Replay-side application of a chain `FocusSet` row. Used by
    /// `cli::commands::serve`'s boot path to reconstruct
    /// `pinned_focus + focus_history` from the durable chain.
    ///
    /// Differs from [`Self::focus`] / [`Self::unfocus`] in two ways:
    /// (1) the embedder is never called — the vec is supplied by the
    /// chain row verbatim, so stochastic embedders never drift across
    /// restart; (2) the `pinned_at` timestamp is taken from the chain
    /// row rather than `Utc::now()`. The mutation logic (push-current
    /// to history, merge-on-dedupe, LRU eviction) matches the live
    /// runtime exactly so the derived view is identical.
    ///
    /// `query.is_some() != vec.is_some()` is treated as an unfocus —
    /// the writer-side schema invariant should prevent this, but
    /// replay must never refuse to make progress on a malformed row.
    #[allow(clippy::significant_drop_tightening)]
    pub async fn apply_focus_set_from_chain(
        &self,
        scope: &AttentionScope,
        query: Option<String>,
        vec: Option<Vec<f32>>,
        ts: DateTime<Utc>,
    ) -> Result<(), AttentionError> {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let scope_state = inner.scopes.entry(key).or_default();

        match (query, vec) {
            (Some(q), Some(v)) => {
                // Merge-on-dedupe: if this query was already in
                // history, take it out first so we don't double up
                // after we install the new pin.
                if let Some(pos) = scope_state.focus_history.iter().position(|p| p.query == q) {
                    let _ = scope_state.focus_history.remove(pos);
                }
                // Demote current pin to history front.
                if let Some(prev) = scope_state.pinned_focus.take() {
                    scope_state.focus_history.push_front(prev);
                    while scope_state.focus_history.len() > FOCUS_HISTORY_MAX {
                        scope_state.focus_history.pop_back();
                    }
                }
                scope_state.pinned_focus = Some(PinnedFocus {
                    query: q,
                    vec: v,
                    pinned_at: ts,
                });
            }
            _ => {
                // Unfocus (both None, or the malformed-mixed case
                // which we treat as unfocus). Push current pin to
                // history if any.
                if let Some(prev) = scope_state.pinned_focus.take() {
                    scope_state.focus_history.push_front(prev);
                    while scope_state.focus_history.len() > FOCUS_HISTORY_MAX {
                        scope_state.focus_history.pop_back();
                    }
                }
            }
        }
        Ok(())
    }

    /// Read-only snapshot of the scope's focus state. Cheap; clones
    /// the history into a Vec for the caller's convenience.
    pub async fn focus_status(
        &self,
        scope: &AttentionScope,
    ) -> Result<FocusStatus, AttentionError> {
        let key = ScopeKey::from(scope);
        let inner = self.inner.read().await;
        let Some(state) = inner.scopes.get(&key) else {
            return Ok(FocusStatus {
                pinned: None,
                history: Vec::new(),
                transient_active: false,
            });
        };
        Ok(FocusStatus {
            pinned: state.pinned_focus.clone(),
            history: state.focus_history.iter().cloned().collect(),
            transient_active: state.pinned_focus.is_none() && !state.transient_vec.is_empty(),
        })
    }

    /// Replay a deterministic event sequence into a fresh store. Used by
    /// `restart_rebuilds_scores_from_chain` to demonstrate that score
    /// tier is reconstructible from the chain.
    pub async fn replay(&self, events: &[ReplayEvent]) -> Result<(), AttentionError> {
        for ev in events {
            match ev {
                ReplayEvent::Attend { scope, context } => {
                    self.attend(scope, context).await?;
                }
                ReplayEvent::Familiarize { scope, handle } => {
                    self.familiarize(scope, handle).await?;
                }
                ReplayEvent::Decay { handle, factor } => {
                    self.decay(handle, *factor).await?;
                }
                ReplayEvent::Fold {
                    scope,
                    handle,
                    depth,
                } => {
                    self.fold(scope, handle, *depth).await?;
                }
            }
        }
        Ok(())
    }
}

/// Replayable event for `InMemoryAttention::replay`. Mirrors the
/// chain-of-events shape from `three-time-scales` — the score tier is
/// rebuilt by replaying these in order.
#[derive(Debug, Clone)]
pub enum ReplayEvent {
    Attend {
        scope: AttentionScope,
        context: String,
    },
    Familiarize {
        scope: AttentionScope,
        handle: ThreadHandle,
    },
    Decay {
        handle: ThreadHandle,
        factor: f32,
    },
    Fold {
        scope: AttentionScope,
        handle: ThreadHandle,
        depth: FoldDepth,
    },
}

/// Deterministic embedding stub: SHA-256 of context, expanded to a
/// fixed-length f32 vector.
///
/// Replaced by a real embedder via dependency injection in a later
/// phase; the runtime never assumes anything about the embedding source
/// beyond "cosine-comparable vectors of equal dimension".
#[must_use]
pub fn stub_embed(context: &str) -> Vec<f32> {
    const DIM: usize = 32;
    let hash = Sha256::digest(context.as_bytes());
    let mut out = Vec::with_capacity(DIM);
    for chunk in hash.chunks(2).take(DIM / 2) {
        // 16 bits -> [-1.0, 1.0]
        let v = u16::from_be_bytes([chunk[0], chunk[1]]);
        let f = (f32::from(v) / f32::from(u16::MAX)).mul_add(2.0, -1.0);
        out.push(f);
    }
    while out.len() < DIM {
        out.push(0.0);
    }
    out
}

// The `RwLock` guard pattern intentionally holds the lock across the
// `entry()/get()` borrow plus the work that follows; clippy's "merge
// the temporary into its single usage" rewrites either lose the
// `&mut Inner` we need or split a read/write boundary that must stay
// atomic. Same precedent as `crates/store/src/threads.rs`.
#[allow(clippy::significant_drop_tightening)]
#[async_trait]
impl AttentionForwardStore for InMemoryAttention {
    async fn attend(
        &self,
        scope: &AttentionScope,
        context: &str,
    ) -> Result<AttendOutcome, AttentionError> {
        let vec = self.embed(context);
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let entry = inner.scopes.entry(key).or_default();
        // attend() writes the transient + rolling channels but never
        // touches the pin: `effective_vec()` makes pinned win at
        // read time, so the operator's stated lens stays
        // authoritative even while the conversational stream keeps
        // updating attention.
        entry.transient_vec = vec.clone();
        let updated = match entry.rolling_vec.as_deref() {
            None => vec,
            Some(prev) if prev.is_empty() => vec,
            Some(prev) => ema_blend_normalized(prev, &vec, self.lambda),
        };
        entry.rolling_vec = Some(updated.clone());
        Ok(AttendOutcome::Updated {
            rolling_vec: updated,
            lambda: self.lambda,
        })
    }

    async fn surface(
        &self,
        scope: &AttentionScope,
        limit: usize,
    ) -> Result<Vec<AttentionPage>, AttentionError> {
        let key = ScopeKey::from(scope);
        let inner = self.inner.read().await;
        let Some(scope_state) = inner.scopes.get(&key) else {
            return Ok(vec![]);
        };

        let now = Utc::now();
        let mut pages: Vec<AttentionPage> = scope_state
            .threads
            .iter()
            .filter(|(_, t)| should_surface_thread(t, &key))
            .filter_map(|(handle, state)| {
                let parts = compute_score_parts(state, scope_state.effective_vec(), now);
                if parts.score < ARCHIVE_THRESHOLD {
                    return None;
                }
                Some(AttentionPage {
                    handle: handle.to_string(),
                    depth: state.depth,
                    score: parts.score,
                    why: ScoreAttribution {
                        tension: state.tension,
                        resonance: parts.resonance,
                        familiarity: state.familiarity,
                        // Already-weighted contribution so attribution
                        // axes sum (within float tolerance) to `score`.
                        off_diagonal_lift: parts.lift_term,
                        time_since_touch_secs: parts.dt_secs,
                    },
                })
            })
            .collect();

        pages.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        pages.truncate(limit);
        Ok(pages)
    }

    async fn fold(
        &self,
        scope: &AttentionScope,
        handle: &ThreadHandle,
        depth: FoldDepth,
    ) -> Result<(), AttentionError> {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let scope_state = inner.scopes.entry(key.clone()).or_default();
        // Anchor reads transient (conversational state), not pinned
        // focus. Anchor is the thread's identity at materialisation;
        // pinned focus is a ranking lens. Seeding anchors from a pin
        // would conflate identity with stance.
        let anchor_seed = scope_state.transient_vec.clone();
        let was_private = scope.privacy_tier == PrivacyTier::T0Private;
        let thread = scope_state
            .threads
            .entry(handle.clone())
            .or_insert_with(|| ThreadState {
                tension: 0.0,
                familiarity: 0,
                last_touched_at: Utc::now(),
                depth,
                fade_multiplier: 1.0,
                anchor: anchor_seed,
                origin: key.clone(),
                origin_was_private: was_private,
            });
        thread.depth = depth;
        Ok(())
    }

    async fn familiarize(
        &self,
        scope: &AttentionScope,
        handle: &ThreadHandle,
    ) -> Result<(), AttentionError> {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let scope_state = inner.scopes.entry(key.clone()).or_default();
        // Same anchor-seed reasoning as fold(): transient only.
        let anchor_seed = scope_state.transient_vec.clone();
        let was_private = scope.privacy_tier == PrivacyTier::T0Private;
        let thread = scope_state
            .threads
            .entry(handle.clone())
            .or_insert_with(|| ThreadState {
                tension: 0.0,
                familiarity: 0,
                last_touched_at: Utc::now(),
                depth: FoldDepth::Folded,
                fade_multiplier: 1.0,
                anchor: anchor_seed,
                origin: key.clone(),
                origin_was_private: was_private,
            });
        thread.familiarity = thread.familiarity.saturating_add(1);
        thread.last_touched_at = Utc::now();
        Ok(())
    }

    async fn decay(&self, handle: &ThreadHandle, factor: f32) -> Result<(), AttentionError> {
        if !factor.is_finite() || factor < 0.0 {
            return Err(AttentionError::InvalidScore(format!(
                "decay factor must be a finite non-negative float, got {factor}"
            )));
        }
        let mut inner = self.inner.write().await;
        let mut touched = false;
        for scope_state in inner.scopes.values_mut() {
            if let Some(thread) = scope_state.threads.get_mut(handle) {
                thread.fade_multiplier *= factor;
                touched = true;
            }
        }
        if !touched {
            return Err(AttentionError::ThreadNotFound(handle.to_string()));
        }
        Ok(())
    }

    // Curator-side scoring: walk every scope that contains the handle
    // and return the max score. Same formula as `surface()`; "still
    // warm somewhere" should keep a thread alive globally even if it
    // has cooled in the scope that originated it.
    async fn score_thread(&self, handle: &ThreadHandle) -> Result<f32, AttentionError> {
        let inner = self.inner.read().await;
        let now = Utc::now();
        let mut best: Option<f32> = None;
        for scope_state in inner.scopes.values() {
            let Some(state) = scope_state.threads.get(handle) else {
                continue;
            };
            let score = compute_score_parts(state, scope_state.effective_vec(), now).score;
            best = Some(best.map_or(score, |b| b.max(score)));
        }
        Ok(best.unwrap_or(0.0))
    }

    /// Return the current attention vector for the scope, or `None`
    /// when the scope has never been attended to (so no `attend(...)`
    /// has populated a vector yet). Returns an empty vec as `None` —
    /// callers ranking by cosine treat an unattended scope as a no-op.
    async fn scope_vector(
        &self,
        scope: &AttentionScope,
    ) -> Result<Option<Vec<f32>>, AttentionError> {
        let key = ScopeKey::from(scope);
        let inner = self.inner.read().await;
        let Some(state) = inner.scopes.get(&key) else {
            return Ok(None);
        };
        let eff = state.effective_vec();
        if eff.is_empty() {
            return Ok(None);
        }
        Ok(Some(eff.to_vec()))
    }

    // --- focus pin (Phase D dispatch) ---------------------------------
    // The inherent impl on InMemoryAttention owns the mutation logic;
    // these trait method bodies just forward so callers holding an
    // `Arc<dyn AttentionForwardStore>` (the MCP dispatch path) reach
    // the same behavior. Inherent methods take precedence on the
    // concrete type, so existing direct call sites are unaffected.

    async fn focus(
        &self,
        scope: &AttentionScope,
        query: String,
    ) -> Result<FocusOutcome, AttentionError> {
        Self::focus(self, scope, query).await
    }

    async fn refocus(
        &self,
        scope: &AttentionScope,
        query: String,
    ) -> Result<FocusOutcome, AttentionError> {
        Self::refocus(self, scope, query).await
    }

    async fn unfocus(&self, scope: &AttentionScope) -> Result<FocusOutcome, AttentionError> {
        Self::unfocus(self, scope).await
    }

    async fn focus_status(&self, scope: &AttentionScope) -> Result<FocusStatus, AttentionError> {
        Self::focus_status(self, scope).await
    }
}

/// Privacy and scope-isolation filter applied to every surfaced thread.
///
/// A thread is dropped from `surface()` if its origin scope was
/// `T0Private` and the querying scope key differs, regardless of the
/// querying scope's own tier.
fn should_surface_thread(state: &ThreadState, key: &ScopeKey) -> bool {
    if state.origin_was_private && &state.origin != key {
        return false;
    }
    true
}

// --- helpers for tests / internal use ---------------------------------

impl InMemoryAttention {
    /// Test helper: install a thread directly with controlled state.
    /// Tests use this to assemble specific score-shape scenarios
    /// without driving the full ingest path.
    #[doc(hidden)]
    // Test-only helper; bundling the args into a struct would obscure
    // the call sites that exist to set every axis independently.
    // significant_drop_tightening: same `RwLock` guard pattern as the
    // trait impl above — the guard must span the entry()/insert() pair.
    #[allow(clippy::too_many_arguments, clippy::significant_drop_tightening)]
    pub async fn __install_thread_for_test(
        &self,
        scope: &AttentionScope,
        handle: ThreadHandle,
        tension: f32,
        familiarity: u32,
        last_touched_at: DateTime<Utc>,
        depth: FoldDepth,
        anchor: Vec<f32>,
    ) {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let scope_state = inner.scopes.entry(key.clone()).or_default();
        scope_state.threads.insert(
            handle,
            ThreadState {
                tension,
                familiarity,
                last_touched_at,
                depth,
                fade_multiplier: 1.0,
                anchor,
                origin: key,
                origin_was_private: scope.privacy_tier == PrivacyTier::T0Private,
            },
        );
    }
}

// --- tests -------------------------------------------------------------

#[cfg(test)]
// Tests assemble small bounded indices and exact 0.0 outputs from the
// cosine_similarity contract ("Returns 0.0 if zero-norm or mismatched").
// The cast/float_cmp suppressions are test-data ergonomics, not behavior.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn scope_for(project: &str) -> AttentionScope {
        AttentionScope {
            project: Some(project.to_string()),
            session_id: Some("s1".into()),
            agent: Some("claude".into()),
            privacy_tier: PrivacyTier::T1Project,
        }
    }

    fn handle(s: &str) -> ThreadHandle {
        ThreadHandle::new(s).expect("valid handle")
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[tokio::test]
    async fn serde_round_trip_for_attention_page() {
        let store = InMemoryAttention::new();
        let scope = scope_for("haystack");
        store
            .attend(&scope, "fade-is-concentration is load-bearing")
            .await
            .unwrap();
        let h = handle("fade-is-concentration");
        store
            .__install_thread_for_test(
                &scope,
                h.clone(),
                0.5,
                3,
                Utc::now(),
                FoldDepth::Half,
                stub_embed("fade-is-concentration"),
            )
            .await;
        let pages = store.surface(&scope, 10).await.unwrap();
        assert!(!pages.is_empty(), "expected at least one surfaced page");
        let json = serde_json::to_string(&pages[0]).unwrap();
        let back: AttentionPage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.handle, pages[0].handle);
        assert_eq!(back.depth, pages[0].depth);
    }

    #[tokio::test]
    async fn scope_isolates_resonance_across_projects() {
        let store = InMemoryAttention::new();
        let scope_a = scope_for("project-a");
        let scope_b = scope_for("project-b");

        store.attend(&scope_a, "alpha context one").await.unwrap();
        store.attend(&scope_b, "beta context two").await.unwrap();

        let h = handle("shared-handle");
        store
            .__install_thread_for_test(
                &scope_a,
                h.clone(),
                0.5,
                5,
                Utc::now(),
                FoldDepth::Half,
                stub_embed("alpha context one"),
            )
            .await;

        let pages_a = store.surface(&scope_a, 10).await.unwrap();
        let pages_b = store.surface(&scope_b, 10).await.unwrap();

        assert!(pages_a.iter().any(|p| p.handle == "shared-handle"));
        assert!(
            !pages_b.iter().any(|p| p.handle == "shared-handle"),
            "scope B must not see scope A's threads"
        );
    }

    #[tokio::test]
    async fn privacy_tier_t0_never_surfaces_cross_scope() {
        let store = InMemoryAttention::new();
        let mut scope_a_private = scope_for("project-a");
        scope_a_private.privacy_tier = PrivacyTier::T0Private;
        let scope_b = scope_for("project-b");

        store
            .attend(&scope_a_private, "secret context")
            .await
            .unwrap();
        store.attend(&scope_b, "public context").await.unwrap();

        let h = handle("private-thread");
        store
            .__install_thread_for_test(
                &scope_a_private,
                h.clone(),
                0.0,
                5,
                Utc::now(),
                FoldDepth::Half,
                stub_embed("secret context"),
            )
            .await;

        let pages_b = store.surface(&scope_b, 10).await.unwrap();
        assert!(
            !pages_b.iter().any(|p| p.handle == "private-thread"),
            "T0Private threads must not leak across scopes"
        );

        let pages_a = store.surface(&scope_a_private, 10).await.unwrap();
        assert!(pages_a.iter().any(|p| p.handle == "private-thread"));
    }

    #[tokio::test]
    async fn restart_rebuilds_scores_from_chain() {
        let scope = scope_for("haystack");
        let h = handle("rebuilt-thread");
        let events = vec![
            ReplayEvent::Attend {
                scope: scope.clone(),
                context: "rebuild seed".into(),
            },
            ReplayEvent::Fold {
                scope: scope.clone(),
                handle: h.clone(),
                depth: FoldDepth::Half,
            },
            ReplayEvent::Familiarize {
                scope: scope.clone(),
                handle: h.clone(),
            },
            ReplayEvent::Familiarize {
                scope: scope.clone(),
                handle: h.clone(),
            },
        ];

        let a = InMemoryAttention::new();
        a.replay(&events).await.unwrap();
        let pages_a = a.surface(&scope, 10).await.unwrap();

        let b = InMemoryAttention::new();
        b.replay(&events).await.unwrap();
        let pages_b = b.surface(&scope, 10).await.unwrap();

        let pa = pages_a
            .iter()
            .find(|p| p.handle == "rebuilt-thread")
            .unwrap();
        let pb = pages_b
            .iter()
            .find(|p| p.handle == "rebuilt-thread")
            .unwrap();
        assert_eq!(pa.why.familiarity, 2);
        assert_eq!(pb.why.familiarity, 2);
        // Scores depend on Utc::now timing inside surface; familiarity
        // and depth state are the stable invariants the chain replay
        // must reconstruct.
        assert_eq!(pa.depth, FoldDepth::Half);
        assert_eq!(pb.depth, FoldDepth::Half);
    }

    #[tokio::test]
    async fn score_attribution_explains_every_surfaced_page() {
        let store = InMemoryAttention::new();
        let scope = scope_for("haystack");
        store
            .attend(&scope, "abi-as-sovereign-boundary")
            .await
            .unwrap();
        let anchor = stub_embed("abi-as-sovereign-boundary");

        for (i, name) in ["thread-a", "thread-b", "thread-c"].iter().enumerate() {
            store
                .__install_thread_for_test(
                    &scope,
                    handle(name),
                    0.1 * i as f32,
                    (i as u32) * 3,
                    Utc::now() - Duration::seconds(i as i64 * 60),
                    FoldDepth::Half,
                    anchor.clone(),
                )
                .await;
        }
        let pages = store.surface(&scope, 10).await.unwrap();
        assert!(!pages.is_empty());
        for p in &pages {
            // score = floor*exp(-decay*dt) + ALPHA*resonance + lift
            // We don't have the raw floor in the attribution, but we
            // can check: score - (ALPHA*resonance) - off_diagonal_lift
            // must be a non-negative number bounded by familiarity_floor(F).
            let decay_term = ALPHA.mul_add(-p.why.resonance, p.score) - p.why.off_diagonal_lift;
            assert!(
                decay_term >= -1e-4,
                "decay term went negative: {decay_term}"
            );
            let floor = familiarity_floor(p.why.familiarity);
            assert!(
                decay_term <= floor + 1e-4,
                "decay term {decay_term} exceeds floor {floor}"
            );
            // Recompute the full score from attribution and compare.
            let dt_days = p.why.time_since_touch_secs as f32 / 86_400.0;
            let expected_decay = floor * (-decay_rate(p.why.familiarity) * dt_days).exp();
            let expected_total =
                ALPHA.mul_add(p.why.resonance, expected_decay) + p.why.off_diagonal_lift;
            assert!(
                approx_eq(expected_total, p.score, 1e-4),
                "reconstructed {expected_total} != score {}",
                p.score
            );
        }
    }

    #[tokio::test]
    async fn off_diagonal_lift_surfaces_dormant_resonant() {
        let store = InMemoryAttention::new();
        let scope = scope_for("haystack");
        store
            .attend(&scope, "membrane of possibility")
            .await
            .unwrap();
        let anchor_resonant = stub_embed("membrane of possibility");
        // Mild low-resonance anchor for the contrast thread: orthogonal
        // by construction (every other component zeroed and shifted).
        // We want a small positive resonance — high enough to keep the
        // page above ARCHIVE_THRESHOLD, low enough that the dormant
        // page's off-diagonal lift makes it win.
        let anchor_low_resonance: Vec<f32> = anchor_resonant
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if i.is_multiple_of(2) {
                    0.1 * x
                } else {
                    -0.05 * x
                }
            })
            .collect();

        let dormant = handle("dormant-resonant");
        let active = handle("active-orthogonal");

        store
            .__install_thread_for_test(
                &scope,
                dormant.clone(),
                0.05,
                1,
                Utc::now() - Duration::days(7),
                FoldDepth::Folded,
                anchor_resonant.clone(),
            )
            .await;

        store
            .__install_thread_for_test(
                &scope,
                active.clone(),
                0.9,
                1,
                Utc::now(),
                FoldDepth::Half,
                anchor_low_resonance,
            )
            .await;

        let pages = store.surface(&scope, 10).await.unwrap();
        let dormant_page = pages
            .iter()
            .find(|p| p.handle == "dormant-resonant")
            .expect("dormant-resonant must surface");
        let active_page = pages
            .iter()
            .find(|p| p.handle == "active-orthogonal")
            .expect("active-orthogonal must surface (above archive threshold)");
        assert!(
            dormant_page.why.off_diagonal_lift > 0.0,
            "dormant-resonant should receive off-diagonal lift"
        );
        assert!(
            active_page.why.off_diagonal_lift == 0.0,
            "active thread with high tension must not receive off-diagonal lift"
        );
        assert!(
            dormant_page.score > active_page.score,
            "dormant-resonant (score {}) must surface above active-orthogonal (score {})",
            dormant_page.score,
            active_page.score
        );
    }

    #[tokio::test]
    async fn decay_lowers_score() {
        let store = InMemoryAttention::new();
        let scope = scope_for("haystack");
        store.attend(&scope, "ctx").await.unwrap();
        let h = handle("decay-target");
        store
            .__install_thread_for_test(
                &scope,
                h.clone(),
                0.0,
                10,
                Utc::now(),
                FoldDepth::Half,
                stub_embed("ctx"),
            )
            .await;
        let before = store.surface(&scope, 10).await.unwrap();
        let s_before = before
            .iter()
            .find(|p| p.handle == "decay-target")
            .unwrap()
            .score;

        store.decay(&h, 0.5).await.unwrap();
        let after = store.surface(&scope, 10).await.unwrap();
        let s_after = after
            .iter()
            .find(|p| p.handle == "decay-target")
            .unwrap()
            .score;
        assert!(
            s_after < s_before,
            "decay must lower score: before={s_before} after={s_after}"
        );
    }

    #[tokio::test]
    async fn familiarize_increments_counter() {
        let store = InMemoryAttention::new();
        let scope = scope_for("haystack");
        store.attend(&scope, "fam ctx").await.unwrap();
        let h = handle("fam-target");
        for _ in 0..3 {
            store.familiarize(&scope, &h).await.unwrap();
        }
        let pages = store.surface(&scope, 10).await.unwrap();
        let p = pages.iter().find(|p| p.handle == "fam-target").unwrap();
        assert_eq!(p.why.familiarity, 3);
    }

    #[tokio::test]
    async fn archive_threshold_filters_surface() {
        let store = InMemoryAttention::new();
        let scope = scope_for("haystack");
        store.attend(&scope, "ctx").await.unwrap();
        // Build a thread guaranteed to score below ARCHIVE_THRESHOLD:
        // - resonance ~ 0 (anchor opposite to scope vec)
        // - dt huge (decay term collapses)
        // - tension high (no off-diagonal lift)
        let h = handle("archived");
        let opposite_anchor: Vec<f32> = stub_embed("ctx").iter().map(|x| -x).collect();
        store
            .__install_thread_for_test(
                &scope,
                h.clone(),
                0.9,
                0,
                Utc::now() - Duration::days(365),
                FoldDepth::Folded,
                opposite_anchor,
            )
            .await;
        let pages = store.surface(&scope, 10).await.unwrap();
        assert!(
            !pages.iter().any(|p| p.handle == "archived"),
            "score-below-threshold thread must be filtered"
        );
    }

    #[test]
    fn cosine_similarity_basic() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 1.0, 1e-6));
        let c = vec![-1.0, 0.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &c), -1.0, 1e-6));
        let z = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &z), 0.0);
        let mismatched = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &mismatched), 0.0);
    }

    #[test]
    fn decay_rate_interpolates() {
        assert!(approx_eq(decay_rate(0), DECAY_RATE_BASE, 1e-6));
        assert!(approx_eq(decay_rate(20), DECAY_RATE_FAMILIAR, 1e-6));
        assert!(approx_eq(decay_rate(40), DECAY_RATE_FAMILIAR, 1e-6));
        let mid = decay_rate(10);
        assert!(mid > DECAY_RATE_FAMILIAR && mid < DECAY_RATE_BASE);
    }

    #[test]
    fn familiarity_floor_clamps() {
        assert!(approx_eq(familiarity_floor(0), 0.1, 1e-6));
        assert!(approx_eq(familiarity_floor(20), 1.0, 1e-6));
        assert!(approx_eq(familiarity_floor(40), 1.0, 1e-6));
    }

    // ---- Phase A: real-embedder path ---------------------------------

    /// Deterministic non-trivial-dim fake embedder. 32 != dim, so a
    /// regression that silently falls back to `stub_embed` (32-dim)
    /// would fail the dim assertion below.
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
                    let seed = ((t.len() % 100) as f32) * 0.01;
                    (0..self.dim)
                        .map(|i| (i as f32).mul_add(0.001, seed))
                        .collect()
                })
                .collect()
        }
    }

    #[tokio::test]
    async fn with_embedder_uses_real_dim_through_attend() {
        const DIM: usize = 64;
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: DIM }));
        assert_eq!(store.embedder_dim(), Some(DIM));
        assert_eq!(store.embed("anything").len(), DIM);

        let scope = scope_for("phase-a");
        store.attend(&scope, "first attend").await.unwrap();
        let vec = store
            .scope_vector(&scope)
            .await
            .unwrap()
            .expect("scope_vector after attend should be Some");
        assert_eq!(
            vec.len(),
            DIM,
            "scope_vector dim must match embedder, not 32-dim stub"
        );
    }

    #[tokio::test]
    async fn new_without_embedder_falls_back_to_stub() {
        // Back-compat: tests + bootstrapping paths that don't wire an
        // embedder still get the stub. Important so existing tests
        // never need to thread a FakeEmbedder through everywhere.
        let store = InMemoryAttention::new();
        assert_eq!(store.embedder_dim(), None);
        let scope = scope_for("legacy");
        store.attend(&scope, "no embedder here").await.unwrap();
        let vec = store.scope_vector(&scope).await.unwrap().unwrap();
        assert_eq!(vec.len(), 32, "stub_embed must produce 32-dim vectors");
    }

    #[tokio::test]
    async fn seed_anchor_creates_thread_with_given_anchor() {
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("seed");
        let h = handle("first-thread");
        let anchor = vec![0.1_f32; 8];

        store
            .seed_anchor(&scope, h.clone(), anchor.clone())
            .await
            .unwrap();

        // Familiarize must NOT clobber the seeded anchor — or_insert
        // skips because the entry exists, so the stored anchor wins.
        store.familiarize(&scope, &h).await.unwrap();

        // Surface and confirm the thread is materialised. We can't
        // read .anchor directly (private field), so cosine-equality
        // through scope_vector is the available probe:
        store.attend(&scope, "first-thread").await.unwrap();
        let pages = store.surface(&scope, 10).await.unwrap();
        assert!(
            pages.iter().any(|p| p.handle == h.as_str()),
            "seeded thread should surface; got {pages:?}"
        );
    }

    #[tokio::test]
    async fn seed_anchor_replaces_existing_anchor_in_place() {
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("replace");
        let h = handle("renamed-thread");

        // First seed installs the thread.
        store
            .seed_anchor(&scope, h.clone(), vec![0.0_f32; 8])
            .await
            .unwrap();
        // Familiarize bumps the familiarity counter; the seeded
        // entry should still exist after, with familiarity > 0.
        store.familiarize(&scope, &h).await.unwrap();
        store.familiarize(&scope, &h).await.unwrap();

        // Second seed replaces the anchor without resetting the
        // familiarity / last_touched_at fields — same thread,
        // different identity vector.
        store
            .seed_anchor(&scope, h.clone(), vec![1.0_f32; 8])
            .await
            .unwrap();

        // surface() proves the thread is still materialised and the
        // familiarity bump survived (entry was preserved, not
        // recreated).
        store.attend(&scope, "renamed-thread").await.unwrap();
        let pages = store.surface(&scope, 10).await.unwrap();
        let page = pages
            .iter()
            .find(|p| p.handle == h.as_str())
            .expect("thread present");
        assert!(
            page.why.familiarity >= 2,
            "familiarity bump must survive seed_anchor replace; got {}",
            page.why.familiarity
        );
    }

    // ---- Phase C: focus pin + history -------------------------------

    #[tokio::test]
    async fn focus_sets_pin_and_attend_does_not_clobber() {
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("focus-1");

        let outcome = store.focus(&scope, "the API surface".into()).await.unwrap();
        assert!(outcome.previous.is_none());
        assert_eq!(outcome.pinned.as_ref().unwrap().query, "the API surface");
        let pinned_vec = outcome.pinned.as_ref().unwrap().vec.clone();

        // attend() must NOT replace the pin's vector — it writes to
        // the transient channel, which is shadowed by the pin at read.
        store
            .attend(&scope, "something else entirely")
            .await
            .unwrap();
        let v = store.scope_vector(&scope).await.unwrap().unwrap();
        assert_eq!(
            v, pinned_vec,
            "scope_vector must return pinned vec, not transient"
        );
    }

    #[tokio::test]
    async fn focus_replaces_pin_and_demotes_previous_to_history() {
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("focus-2");

        store.focus(&scope, "A".into()).await.unwrap();
        let out = store.focus(&scope, "B".into()).await.unwrap();
        assert_eq!(out.previous.as_ref().unwrap().query, "A");
        assert_eq!(out.pinned.as_ref().unwrap().query, "B");
        assert_eq!(out.history.len(), 1);
        assert_eq!(out.history[0].query, "A");
    }

    #[tokio::test]
    async fn focus_merge_on_dedupe_promotes_existing_history_entry() {
        // Hand-off locked decision: focus(query=X) where X is already
        // in history must promote the EXISTING entry (preserving its
        // original vec) rather than re-embedding. Critical for
        // stochastic embedders.
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("focus-3");

        let out_a = store.focus(&scope, "A".into()).await.unwrap();
        let original_a_vec = out_a.pinned.as_ref().unwrap().vec.clone();
        let original_a_ts = out_a.pinned.as_ref().unwrap().pinned_at;

        store.focus(&scope, "B".into()).await.unwrap();
        // A is now in history. Re-focus on A.
        let out = store.focus(&scope, "A".into()).await.unwrap();
        let promoted = out.pinned.as_ref().unwrap();
        assert_eq!(promoted.query, "A");
        assert_eq!(
            promoted.vec, original_a_vec,
            "merge must preserve the original vec, not re-embed"
        );
        assert_eq!(
            promoted.pinned_at, original_a_ts,
            "merge must preserve the original pinned_at"
        );
        // B was demoted to history; A came out of history; history
        // contains exactly [B], no duplicate A.
        assert_eq!(out.history.len(), 1);
        assert_eq!(out.history[0].query, "B");
    }

    #[tokio::test]
    async fn refocus_promotes_from_history_and_errors_on_miss() {
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("focus-4");

        store.focus(&scope, "A".into()).await.unwrap();
        store.focus(&scope, "B".into()).await.unwrap(); // A → history
        let out = store.refocus(&scope, "A".into()).await.unwrap();
        assert_eq!(out.pinned.as_ref().unwrap().query, "A");
        assert_eq!(out.previous.as_ref().unwrap().query, "B");
        // Ping-pong: history stays at [B] (B was demoted, A was taken
        // out). No growth from refocus.
        assert_eq!(out.history.len(), 1);
        assert_eq!(out.history[0].query, "B");

        // refocus on a query that's never been pinned must error.
        let err = store.refocus(&scope, "C".into()).await.unwrap_err();
        assert!(matches!(err, AttentionError::FocusHistoryMiss(_)));
    }

    #[tokio::test]
    async fn unfocus_clears_pin_and_pushes_to_history() {
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("focus-5");

        store.focus(&scope, "A".into()).await.unwrap();
        let out = store.unfocus(&scope).await.unwrap();
        assert_eq!(out.previous.as_ref().unwrap().query, "A");
        assert!(out.pinned.is_none());
        assert_eq!(out.history[0].query, "A");

        // Unfocus on an already-unfocused scope is idempotent.
        let out = store.unfocus(&scope).await.unwrap();
        assert!(out.previous.is_none());
        assert!(out.pinned.is_none());
        // History unchanged: still has A from the first unfocus.
        assert_eq!(out.history.len(), 1);
    }

    #[tokio::test]
    async fn focus_history_caps_at_max() {
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("focus-6");

        // Each focus(N) (for N >= 1) demotes pin-{N-1} to history,
        // so after N focuses the history has N-1 entries. To force
        // an eviction we need N - 1 > FOCUS_HISTORY_MAX, i.e. N >=
        // MAX + 2 focuses.
        let total = FOCUS_HISTORY_MAX + 2;
        for i in 0..total {
            store.focus(&scope, format!("pin-{i}")).await.unwrap();
        }
        let status = store.focus_status(&scope).await.unwrap();
        assert_eq!(
            status.history.len(),
            FOCUS_HISTORY_MAX,
            "history must cap at FOCUS_HISTORY_MAX"
        );
        // pin-0 was the oldest demoted entry — it must have been
        // evicted by the LRU pop on the (MAX+2)th focus.
        assert!(
            !status.history.iter().any(|p| p.query == "pin-0"),
            "oldest pin must be evicted; got {:?}",
            status.history.iter().map(|p| &p.query).collect::<Vec<_>>()
        );
        // pin-1 should still be present (it's now the oldest).
        assert!(
            status.history.iter().any(|p| p.query == "pin-1"),
            "pin-1 should be the new oldest"
        );
    }

    #[tokio::test]
    async fn focus_status_reports_transient_active() {
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("focus-7");

        let st = store.focus_status(&scope).await.unwrap();
        assert!(st.pinned.is_none());
        assert!(!st.transient_active, "fresh scope has no transient");

        store.attend(&scope, "hello").await.unwrap();
        let st = store.focus_status(&scope).await.unwrap();
        assert!(st.transient_active, "attend must light up transient_active");

        store.focus(&scope, "X".into()).await.unwrap();
        let st = store.focus_status(&scope).await.unwrap();
        assert!(
            !st.transient_active,
            "with a pin set, transient_active must be false"
        );
    }

    #[tokio::test]
    async fn apply_focus_set_from_chain_rebuilds_pin_and_history() {
        // Simulates the boot replay path: feed a sequence of
        // FocusSet rows in chronological order, then verify the
        // reconstructed pin + history match what live runtime
        // would produce.
        let store = InMemoryAttention::with_embedder(Arc::new(FakeEmbedder { dim: 8 }));
        let scope = scope_for("focus-replay");
        let vec_a = vec![0.1_f32; 8];
        let vec_b = vec![0.2_f32; 8];
        let t0 = Utc::now();
        let t1 = t0 + Duration::seconds(1);
        let t2 = t0 + Duration::seconds(2);

        // Row 1: focus("A")
        store
            .apply_focus_set_from_chain(&scope, Some("A".into()), Some(vec_a.clone()), t0)
            .await
            .unwrap();
        // Row 2: focus("B")
        store
            .apply_focus_set_from_chain(&scope, Some("B".into()), Some(vec_b.clone()), t1)
            .await
            .unwrap();
        // Row 3: re-focus("A") — chain carries vec_a verbatim (the
        // writer-side merge-on-dedupe reuses the original).
        store
            .apply_focus_set_from_chain(&scope, Some("A".into()), Some(vec_a.clone()), t2)
            .await
            .unwrap();

        let st = store.focus_status(&scope).await.unwrap();
        let pinned = st.pinned.as_ref().expect("pin must be set");
        assert_eq!(pinned.query, "A");
        assert_eq!(pinned.vec, vec_a, "vec must come from chain row verbatim");
        assert_eq!(
            pinned.pinned_at, t2,
            "pinned_at must reflect chain row ts, not Utc::now()"
        );
        // History: [B] only — A was deduped out before B was demoted.
        assert_eq!(st.history.len(), 1);
        assert_eq!(st.history[0].query, "B");

        // Row 4: unfocus — pin clears, A goes to history front.
        store
            .apply_focus_set_from_chain(&scope, None, None, t2 + Duration::seconds(1))
            .await
            .unwrap();
        let st = store.focus_status(&scope).await.unwrap();
        assert!(st.pinned.is_none());
        assert_eq!(st.history[0].query, "A");
        assert_eq!(st.history[1].query, "B");
    }
}
