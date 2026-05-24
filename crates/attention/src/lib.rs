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

pub mod observer;
pub use observer::{ObservationResult, ObserverError, ProposedThreadStub, TurnObserver};

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use ostk_recall_core::attention::{
    AttentionPage, AttentionScope, FoldDepth, PrivacyTier, ScoreAttribution, ThreadHandle,
};
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

#[derive(Debug, Error)]
pub enum AttentionError {
    #[error("scope not found: {0:?}")]
    ScopeNotFound(ScopeKey),
    #[error("thread not found: {0}")]
    ThreadNotFound(String),
    #[error("invalid score: {0}")]
    InvalidScore(String),
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
    /// Ingest current conversational / tool context; update the rolling
    /// attention vector for the given scope.
    async fn attend(&self, scope: &AttentionScope, context: &str) -> Result<(), AttentionError>;

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

#[derive(Debug, Default)]
struct ScopeState {
    attention_vec: Vec<f32>,
    threads: HashMap<ThreadHandle, ThreadState>,
}

#[derive(Debug, Default)]
struct Inner {
    scopes: HashMap<ScopeKey, ScopeState>,
}

/// In-memory attention runtime. Cloning is cheap (Arc-shared).
#[derive(Clone, Default)]
pub struct InMemoryAttention {
    inner: Arc<RwLock<Inner>>,
}

impl InMemoryAttention {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
    async fn attend(&self, scope: &AttentionScope, context: &str) -> Result<(), AttentionError> {
        let key = ScopeKey::from(scope);
        let mut inner = self.inner.write().await;
        let entry = inner.scopes.entry(key).or_default();
        entry.attention_vec = stub_embed(context);
        Ok(())
    }

    // `*_term` triplet plus `scope`/`scope_state` triggers
    // similar_names; the names are deliberate (each term mirrors an
    // attribution axis) and renaming would obscure the score formula.
    // Casts are bounded: dt_secs is clamped >=0 then expressed in days,
    // well within f32 precision for any realistic uptime.
    #[allow(
        clippy::similar_names,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
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
                let resonance = cosine_similarity(&state.anchor, &scope_state.attention_vec);
                let dt_secs = (now - state.last_touched_at).num_seconds().max(0) as u64;
                let dt_days = dt_secs as f32 / 86_400.0;
                let floor = familiarity_floor(state.familiarity) * state.fade_multiplier;
                let decay_term = floor * (-decay_rate(state.familiarity) * dt_days).exp();
                let resonance_term = ALPHA * resonance;
                let lift_gate = off_diagonal_lift_gate(state.tension, resonance);
                let lift_term = BETA * lift_gate;
                let score = decay_term + resonance_term + lift_term;
                if score < ARCHIVE_THRESHOLD {
                    return None;
                }
                Some(AttentionPage {
                    handle: handle.to_string(),
                    depth: state.depth,
                    score,
                    why: ScoreAttribution {
                        tension: state.tension,
                        resonance,
                        familiarity: state.familiarity,
                        // Already-weighted contribution so attribution
                        // axes sum (within float tolerance) to `score`.
                        off_diagonal_lift: lift_term,
                        time_since_touch_secs: dt_secs,
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
        let anchor_seed = scope_state.attention_vec.clone();
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
        let anchor_seed = scope_state.attention_vec.clone();
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
}
