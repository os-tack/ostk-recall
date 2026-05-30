//! P9b-min — lens daemon loop + memory-lens MCP resource.
//!
//! Two surfaces:
//!
//! 1. [`MemoryLensResource`] — a [`Resource`] impl whose body is
//!    shared with the daemon via `Arc<RwLock<String>>`. The
//!    registry serves `resources/read` against the current body;
//!    the daemon updates the body via the same handle.
//! 2. [`try_refresh_lens`] / [`run_lens_loop`] — the decision +
//!    driver pair. `try_refresh_lens` is the pure step (sans
//!    sleep, sans the persistence write) so the gate suite can
//!    cover empty-mind / drift / pin-fingerprint / content-fp
//!    behaviour without spawning a tokio daemon.
//!
//! Shutdown contract: `run_lens_loop` reads a
//! `CancellationToken`; firing the token unblocks the sleep and
//! the loop exits cleanly.

use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use blake3;
use chrono::Utc;
use ostk_recall_attention::{AttentionForwardStore, InMemoryAttention};
use ostk_recall_core::attention::AttentionScope;
use ostk_recall_mcp::{Resource, ResourceContent, ResourceError, ResourceRegistry};
use ostk_recall_query::context::AttentionContext;
use ostk_recall_query::lens::{Lens, LensConfig, build_lens};
use ostk_recall_store::ChainEvent;
use ostk_recall_store::corpus::CorpusStore;
use ostk_recall_store::threads::ChainSink;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::lens_state::{LensState, save_lens_state};

/// Canonical URI for the memory-lens resource. Stable wire string;
/// clients (Claude Code, Claude Desktop, Cursor) subscribe with
/// exactly this URI.
pub const MEMORY_LENS_URI: &str = "ostk://memory-lens";
pub const MEMORY_LENS_NAME: &str = "ostk-recall memory lens";

/// Filename within the state directory where the loop writes a
/// copy of the most-recent rendered markdown. The MCP transport is
/// the authoritative wire, but a side-of-disk copy lets
/// `ostk-recall lens show` dump the current lens without spawning
/// an MCP client.
pub const LENS_MARKDOWN_FILE: &str = "lens.md";
pub const MEMORY_LENS_DESCRIPTION: &str = "Ambient memory lens — automatically surfaces chunks aligned with current attention. \
     Re-rendered when attention drifts or the pinned focus changes. \
     Resource content is markdown; subscribe for `notifications/resources/updated` to be \
     told when to re-read.";

// ---------------------------------------------------------------------
// MemoryLensResource
// ---------------------------------------------------------------------

/// MCP resource backing the `memory-lens` URI.
///
/// The body lives in a shared `Arc<RwLock<String>>` so the daemon
/// loop and the dispatch handler can talk to the same storage.
/// `Resource::read` clones the body under a short read lock; the
/// loop writes under a write lock. Both happen on a tokio runtime
/// but the critical section is microseconds — a `std::sync::RwLock`
/// is cheaper than crossing the async layer for what amounts to a
/// `String::clone`.
#[derive(Clone)]
pub struct MemoryLensResource {
    body: Arc<RwLock<String>>,
}

impl MemoryLensResource {
    /// Construct with an initial markdown body. Empty string is
    /// fine — the first refresh will overwrite it.
    #[must_use]
    pub fn new(initial: String) -> Self {
        Self {
            body: Arc::new(RwLock::new(initial)),
        }
    }

    /// Handle for the daemon. The loop calls `set` after each
    /// successful build; emitting the notification happens through
    /// `ResourceRegistry::emit_resource_updated`.
    pub fn set(&self, body: String) {
        if let Ok(mut guard) = self.body.write() {
            *guard = body;
        }
    }

    /// Snapshot the current body. Exposed for `lens show` and
    /// tests; production reads route through the MCP dispatch
    /// path.
    #[must_use]
    pub fn snapshot(&self) -> String {
        match self.body.read() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }
}

impl Resource for MemoryLensResource {
    fn uri(&self) -> &str {
        MEMORY_LENS_URI
    }
    fn name(&self) -> &str {
        MEMORY_LENS_NAME
    }
    fn description(&self) -> &str {
        MEMORY_LENS_DESCRIPTION
    }
    fn read(&self) -> Result<ResourceContent, ResourceError> {
        let body = match self.body.read() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        };
        Ok(ResourceContent::text(
            MEMORY_LENS_URI,
            self.mime_type(),
            body,
        ))
    }
}

// ---------------------------------------------------------------------
// Decision step
// ---------------------------------------------------------------------

/// What `try_refresh_lens` decided. Loops translate this into the
/// concrete side-effects (registry write, notification, chain
/// events, state persistence) — keeping it as an enum makes the
/// gate suite trivial to assert against.
#[derive(Debug, Clone)]
pub enum LensRefreshDecision {
    /// `rolling_vec.is_none() AND pinned == false`. P9b-min spec C2:
    /// don't fire on a zero/null state.
    EmptyMind,
    /// No drift on rolling, no pin change. Lens is stale-but-current.
    NoTrigger,
    /// Pin change OR drift triggered, build_lens errored. The loop
    /// keeps last lens visible; no registry update, no notification,
    /// no `LensIncluded` log. State is NOT advanced — the next poll
    /// retries.
    BuildFailed(String),
    /// Rendered bytes match the prior fingerprint AND pin didn't
    /// change. State advances (so we don't re-detect drift against
    /// a stale baseline), but no registry update / notification /
    /// `LensIncluded`. The carried `LensState` is what the loop
    /// must persist — without it, drift comparison stays anchored
    /// on the original rolling baseline forever, and the next
    /// genuine refresh fires for what is semantically unchanged
    /// content.
    UnchangedContent { new_state: LensState },
    /// Full refresh: registry update + notification + per-entry
    /// `LensIncluded`. Carries the rendered markdown and the new
    /// state to persist.
    Refresh {
        rendered: String,
        lens: Lens,
        new_state: LensState,
    },
}

/// Inputs to the decision step. Caller assembles these under the
/// attention read guard (sync, microseconds), drops the guard, then
/// calls `try_refresh_lens`. Mirrors the two-phase snapshot from
/// `p9b-lens-portfolio.md` "Background loop".
#[derive(Debug, Clone)]
pub struct LensTickSnapshot {
    pub rolling_vec: Option<Vec<f32>>,
    pub scope_vector: Option<Vec<f32>>,
    /// `blake3(pinned_vec || scope_bytes)` if pinned, else `None`.
    /// Caller computes this so we don't pull in
    /// `AttentionScope::serialize` here.
    pub pin_fingerprint: Option<Vec<u8>>,
}

impl LensTickSnapshot {
    /// Build an `AttentionContext` mirroring the snapshot's vectors.
    /// `scope_vector` reflects the effective vector (pin precedence
    /// already applied upstream); `rolling_vec` is raw.
    fn to_attention_context(&self) -> AttentionContext {
        AttentionContext {
            scope_vector: self.scope_vector.clone(),
            rolling_vec: self.rolling_vec.clone(),
            // P7b: the ledger reader is wired into the lens by P9b-full
            // when the freshness slot lands; the attention-only P9b-min
            // lens has no freshness feature, so None is correct here.
            chain_log: None,
        }
    }
}

/// Compute the lens-refresh decision for a single tick.
///
/// Pure modulo `build_lens` (which hits Lance). Caller applies the
/// returned side-effects in this order, per
/// `p9b-lens-portfolio.md` "Background loop":
///
/// 1. Update the registry body (clear → write `rendered`).
/// 2. `emit_resource_updated("memory-lens")`.
/// 3. Per-entry `LensIncluded` chain events.
/// 4. Persist `new_state` to `lens_state.json`.
///
/// Steps 2-3 are gated on a successful step 1 so a half-committed
/// refresh can't surface to clients. P9b-min owns this ordering;
/// P9b-full lifts it into a transactional wrapper.
pub async fn try_refresh_lens(
    snapshot: &LensTickSnapshot,
    last_state: &LensState,
    corpus: &CorpusStore,
    config: &LensConfig,
) -> LensRefreshDecision {
    // 1. Empty-mind skip — C2.
    let pinned = snapshot.pin_fingerprint.is_some();
    if snapshot.rolling_vec.is_none() && !pinned {
        return LensRefreshDecision::EmptyMind;
    }

    // 2. Decide whether the operator pin changed.
    let pin_changed = match (
        snapshot.pin_fingerprint.as_deref(),
        last_state.last_pin_fingerprint.as_deref(),
    ) {
        (Some(now), Some(prev)) => now != prev,
        (Some(_), None) | (None, Some(_)) => true,
        (None, None) => false,
    };

    // 3. Decide whether rolling drifted past the threshold.
    let rolling_drifted = match (
        snapshot.rolling_vec.as_deref(),
        last_state.last_rolling_vec.as_deref(),
    ) {
        (Some(now), Some(prev)) => cosine_distance(prev, now) >= config.drift_threshold,
        (Some(_), None) => true, // first valid rolling sample post-empty-mind
        (None, _) => false,      // no rolling channel → only pin change can trigger
    };

    if !(pin_changed || rolling_drifted) {
        return LensRefreshDecision::NoTrigger;
    }

    // 4. Build the lens. Errors return BuildFailed without
    //    touching last_state — next poll retries.
    let attn = snapshot.to_attention_context();
    let lens = match build_lens(&attn, corpus, config).await {
        Ok(l) => l,
        Err(e) => return LensRefreshDecision::BuildFailed(e.to_string()),
    };

    // 5. Render + fingerprint.
    let rendered = lens.to_markdown();
    let content_fp = blake3::hash(rendered.as_bytes()).as_bytes().to_vec();

    // 6. Unchanged-content skip — only when the pin didn't change.
    //    Pin changes always force a re-emit so the operator can
    //    see they're now driving the lens.
    if !pin_changed && last_state.last_content_fp.as_deref() == Some(content_fp.as_slice()) {
        // Advance the rolling baseline + pin fingerprint so the
        // next poll's drift check compares against *this* tick, not
        // the original baseline. Without this, accumulated drift
        // against a long-stable lens would eventually flip the
        // pin-unchanged branch and fire a spurious notification
        // for byte-identical content.
        let mut new_state = last_state.clone();
        new_state.last_rolling_vec = snapshot.rolling_vec.clone();
        new_state.last_pin_fingerprint = snapshot.pin_fingerprint.clone();
        new_state.last_lens_ts = Some(Utc::now());
        return LensRefreshDecision::UnchangedContent { new_state };
    }

    // 7. Full refresh.
    let new_state = LensState {
        last_rolling_vec: snapshot.rolling_vec.clone(),
        last_pin_fingerprint: snapshot.pin_fingerprint.clone(),
        last_portfolio_chunk_ids: lens.entries.iter().map(|e| e.chunk_id.clone()).collect(),
        last_content_fp: Some(content_fp),
        last_lens_ts: Some(Utc::now()),
    };
    LensRefreshDecision::Refresh {
        rendered,
        lens,
        new_state,
    }
}

/// Cosine distance — `1.0 - cosine_similarity`. Returns 2.0 on
/// empty / dim-mismatched inputs so a vector swap can't be confused
/// for "no drift."
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 2.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 2.0;
    }
    let sim = dot / (na.sqrt() * nb.sqrt());
    (1.0 - sim).max(0.0)
}

/// Build a `LensTickSnapshot` from the current attention state for
/// `scope`. Held lock duration is one async call to `scope_vector` —
/// the implementation grabs the inner `RwLock` synchronously inside.
/// The pin fingerprint hashes `(pinned_focus_vec || scope_bytes)`;
/// p6 stores the pinned vec but not directly accessible, so we
/// fall back to fingerprinting `scope_vector || scope_bytes` when
/// `scope_vector` itself reflects a pin (= different from
/// `rolling_vec`). Good-enough heuristic for P9b-min; P9b-full
/// gets a dedicated accessor on `InMemoryAttention`.
pub async fn snapshot_attention(
    attention: &InMemoryAttention,
    scope: &AttentionScope,
) -> LensTickSnapshot {
    let scope_vector = attention.scope_vector(scope).await.ok().flatten();
    let rolling_vec = attention.rolling_vec(scope).await.ok().flatten();
    let pin_fingerprint = compute_pin_fingerprint_from_vectors(
        scope_vector.as_deref(),
        rolling_vec.as_deref(),
        scope,
    );
    LensTickSnapshot {
        rolling_vec,
        scope_vector,
        pin_fingerprint,
    }
}

/// If `scope_vector` differs from `rolling_vec`, a pin is driving
/// ranking — fingerprint it. Otherwise return `None`.
fn compute_pin_fingerprint_from_vectors(
    scope_vec: Option<&[f32]>,
    rolling_vec: Option<&[f32]>,
    scope: &AttentionScope,
) -> Option<Vec<u8>> {
    let pinned = match (scope_vec, rolling_vec) {
        (Some(s), Some(r)) => s != r,
        (Some(_), None) => true,
        _ => false,
    };
    if !pinned {
        return None;
    }
    let sv = scope_vec?;
    let mut hasher = blake3::Hasher::new();
    for f in sv {
        hasher.update(&f.to_le_bytes());
    }
    let scope_bytes = serde_json::to_vec(scope).ok()?;
    hasher.update(&scope_bytes);
    Some(hasher.finalize().as_bytes().to_vec())
}

// ---------------------------------------------------------------------
// Daemon loop
// ---------------------------------------------------------------------

/// Background loop. Polls attention at `config.poll_interval_secs`
/// and applies the `try_refresh_lens` decision. Exits when `cancel`
/// fires.
///
/// Cleanup is best-effort: state persistence runs after every
/// decision (including `UnchangedContent`); a final `save_lens_state`
/// is not strictly necessary because every successful tick persisted.
#[allow(clippy::too_many_arguments)]
pub async fn run_lens_loop(
    attention: Arc<InMemoryAttention>,
    corpus: Arc<CorpusStore>,
    registry: Arc<ResourceRegistry>,
    resource: Arc<MemoryLensResource>,
    chain_sink: Arc<dyn ChainSink>,
    config: LensConfig,
    scope: AttentionScope,
    state_dir: PathBuf,
    initial_state: LensState,
    cancel: CancellationToken,
) {
    let mut state = initial_state;
    let mut interval = tokio::time::interval(Duration::from_secs(config.poll_interval_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    info!(
        scope = ?scope,
        uri = MEMORY_LENS_URI,
        "lens loop online"
    );
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("lens loop received cancel; exiting");
                break;
            }
            _ = interval.tick() => {
                let snapshot = snapshot_attention(&attention, &scope).await;
                let decision = try_refresh_lens(&snapshot, &state, &corpus, &config).await;
                apply_decision(
                    decision,
                    &mut state,
                    &resource,
                    &registry,
                    chain_sink.as_ref(),
                    &state_dir,
                );
            }
        }
    }
}

/// Translate a `LensRefreshDecision` into side-effects. Exposed
/// (crate-public) so integration tests can drive the loop's
/// effect surface without spinning up `interval.tick()`.
pub(crate) fn apply_decision(
    decision: LensRefreshDecision,
    state: &mut LensState,
    resource: &MemoryLensResource,
    registry: &ResourceRegistry,
    chain_sink: &dyn ChainSink,
    state_dir: &Path,
) {
    match decision {
        LensRefreshDecision::EmptyMind => {
            debug!("lens: empty mind, skipping");
        }
        LensRefreshDecision::NoTrigger => {
            debug!("lens: no drift, no pin change");
        }
        LensRefreshDecision::BuildFailed(err) => {
            warn!(error = %err, "lens build failed; keeping last lens");
        }
        LensRefreshDecision::UnchangedContent { new_state } => {
            // Adopt the advanced baseline so the next tick's drift
            // check compares against this snapshot, not the original
            // pre-stable baseline.
            *state = new_state;
            if let Err(err) = save_lens_state(state_dir, state) {
                warn!(error = %err, "lens_state.json save failed (unchanged path)");
            }
        }
        LensRefreshDecision::Refresh {
            rendered,
            lens,
            new_state,
        } => {
            // Steps 1-4 from try_refresh_lens's doc:
            //   1. Write rendered into the resource body.
            resource.set(rendered.clone());
            //   1b. Side copy on disk so `lens show` (a separate
            //       process) can read the current lens without
            //       spawning an MCP client.
            if let Err(err) =
                std::fs::write(state_dir.join(LENS_MARKDOWN_FILE), rendered.as_bytes())
            {
                warn!(error = %err, "lens.md write failed");
            }
            //   2. Tell subscribed clients to re-read. Must happen
            //      AFTER the body write so the next resources/read
            //      sees the new content.
            registry.emit_resource_updated(MEMORY_LENS_URI);
            //   3. Audit per entry. Logged only after the
            //      successful update + notify (p9b spec).
            let ts = Utc::now();
            for entry in &lens.entries {
                let event = ChainEvent::LensIncluded {
                    chunk_id: entry.chunk_id.clone(),
                    slot: entry.slot_name.to_string(),
                    ts,
                };
                if let Err(err) = chain_sink.append(&event) {
                    warn!(error = %err, chunk_id = %entry.chunk_id, "LensIncluded append failed");
                }
            }
            //   4. Persist the advanced state.
            *state = new_state;
            if let Err(err) = save_lens_state(state_dir, state) {
                warn!(error = %err, "lens_state.json save failed (refresh path)");
            }
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_core::attention::PrivacyTier;
    use ostk_recall_mcp::ResourceRegistry;
    use ostk_recall_query::lens::{Lens, LensConfig, LensEntry};
    use ostk_recall_store::{ChainEvent, ChainSink, NoopChainSink, StoreError};
    use std::sync::Mutex as StdMutex;
    use tempfile::TempDir;

    fn scope() -> AttentionScope {
        AttentionScope {
            project: Some("p9b-min".into()),
            session_id: Some("test".into()),
            agent: Some("test".into()),
            privacy_tier: PrivacyTier::T1Project,
        }
    }

    #[test]
    fn cosine_distance_identical_vectors_is_zero() {
        let d = cosine_distance(&[1.0, 0.0], &[1.0, 0.0]);
        assert!(d.abs() < 1e-6, "got {d}");
    }

    #[test]
    fn cosine_distance_orthogonal_is_one() {
        let d = cosine_distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-6, "got {d}");
    }

    #[test]
    fn cosine_distance_dim_mismatch_returns_two() {
        let d = cosine_distance(&[1.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!(
            (d - 2.0).abs() < 1e-6,
            "dim mismatch must not look like no-drift"
        );
    }

    #[test]
    fn pin_fingerprint_none_when_scope_equals_rolling() {
        let s = scope();
        let v = vec![1.0, 0.0, 0.0];
        let fp = compute_pin_fingerprint_from_vectors(Some(&v), Some(&v), &s);
        assert!(fp.is_none());
    }

    #[test]
    fn pin_fingerprint_some_when_scope_differs_from_rolling() {
        let s = scope();
        let pin = vec![1.0, 0.0, 0.0];
        let rolling = vec![0.0, 1.0, 0.0];
        let fp = compute_pin_fingerprint_from_vectors(Some(&pin), Some(&rolling), &s);
        assert!(fp.is_some());
        assert_eq!(fp.unwrap().len(), 32, "blake3 produces 32 bytes");
    }

    #[test]
    fn pin_fingerprint_some_when_only_scope_present() {
        let s = scope();
        let pin = vec![1.0, 0.0, 0.0];
        let fp = compute_pin_fingerprint_from_vectors(Some(&pin), None, &s);
        assert!(
            fp.is_some(),
            "scope_vector with no rolling = pin freshly placed"
        );
    }

    #[test]
    fn empty_mind_decision_when_no_rolling_no_pin() {
        let snap = LensTickSnapshot {
            rolling_vec: None,
            scope_vector: None,
            pin_fingerprint: None,
        };
        let state = LensState::default();
        let config = LensConfig::default();
        let tmp = TempDir::new().unwrap();
        // build_lens won't be called on this path — we don't need
        // a real corpus, but try_refresh_lens's signature wants
        // one. Construct a real CorpusStore at the temp path.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let corpus =
            rt.block_on(async { CorpusStore::open_or_create(tmp.path(), 8).await.unwrap() });
        let d = rt.block_on(try_refresh_lens(&snap, &state, &corpus, &config));
        assert!(matches!(d, LensRefreshDecision::EmptyMind));
    }

    #[test]
    fn no_trigger_decision_when_drift_below_threshold() {
        let v = vec![1.0_f32, 0.0, 0.0];
        let snap = LensTickSnapshot {
            rolling_vec: Some(v.clone()),
            scope_vector: Some(v.clone()),
            pin_fingerprint: None,
        };
        // Exactly the same vector → drift = 0 → below 0.15 threshold.
        let state = LensState {
            last_rolling_vec: Some(v),
            ..LensState::default()
        };
        let config = LensConfig::default();
        let tmp = TempDir::new().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let corpus =
            rt.block_on(async { CorpusStore::open_or_create(tmp.path(), 8).await.unwrap() });
        let d = rt.block_on(try_refresh_lens(&snap, &state, &corpus, &config));
        assert!(matches!(d, LensRefreshDecision::NoTrigger), "got {d:?}");
    }

    #[test]
    fn first_rolling_sample_triggers_refresh_attempt() {
        // No prior rolling → first sample drift is treated as ∞.
        // build_lens against an empty corpus returns Ok(empty lens),
        // so we land in Refresh.
        let snap = LensTickSnapshot {
            rolling_vec: Some(vec![1.0, 0.0, 0.0]),
            scope_vector: Some(vec![1.0, 0.0, 0.0]),
            pin_fingerprint: None,
        };
        let state = LensState::default(); // empty
        let config = LensConfig::default();
        let tmp = TempDir::new().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let corpus =
            rt.block_on(async { CorpusStore::open_or_create(tmp.path(), 3).await.unwrap() });
        let d = rt.block_on(try_refresh_lens(&snap, &state, &corpus, &config));
        assert!(
            matches!(d, LensRefreshDecision::Refresh { .. }),
            "got {d:?}"
        );
    }

    #[test]
    fn pin_only_no_spam_no_refresh_when_pin_fp_unchanged() {
        // Pin set, no rolling. Same fingerprint as last time → no
        // trigger. This is the "drift = ∞ every poll" bug guard
        // from p9b's third-pass review.
        let fp = vec![42_u8; 32];
        let snap = LensTickSnapshot {
            rolling_vec: None,
            scope_vector: Some(vec![1.0, 0.0, 0.0]),
            pin_fingerprint: Some(fp.clone()),
        };
        let state = LensState {
            last_pin_fingerprint: Some(fp),
            ..LensState::default()
        };
        let config = LensConfig::default();
        let tmp = TempDir::new().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let corpus =
            rt.block_on(async { CorpusStore::open_or_create(tmp.path(), 3).await.unwrap() });
        let d = rt.block_on(try_refresh_lens(&snap, &state, &corpus, &config));
        assert!(matches!(d, LensRefreshDecision::NoTrigger), "got {d:?}");
    }

    #[test]
    fn pin_change_triggers_even_when_rolling_unchanged() {
        let v = vec![1.0_f32, 0.0, 0.0];
        let snap = LensTickSnapshot {
            rolling_vec: Some(v.clone()),
            scope_vector: Some(vec![0.0, 1.0, 0.0]),
            pin_fingerprint: Some(vec![1_u8; 32]),
        };
        let state = LensState {
            last_rolling_vec: Some(v),
            last_pin_fingerprint: Some(vec![2_u8; 32]), // different
            ..LensState::default()
        };
        let config = LensConfig::default();
        let tmp = TempDir::new().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let corpus =
            rt.block_on(async { CorpusStore::open_or_create(tmp.path(), 3).await.unwrap() });
        let d = rt.block_on(try_refresh_lens(&snap, &state, &corpus, &config));
        // Empty corpus → empty lens → Refresh (UnchangedContent
        // requires a prior fingerprint that matches).
        assert!(
            matches!(d, LensRefreshDecision::Refresh { .. }),
            "got {d:?}"
        );
    }

    // ---- apply_decision side-effects ----

    #[derive(Default)]
    struct RecordingSink {
        events: StdMutex<Vec<ChainEvent>>,
    }
    impl ChainSink for RecordingSink {
        fn append(&self, event: &ChainEvent) -> Result<(), StoreError> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    fn dummy_lens(chunk_ids: Vec<&str>) -> Lens {
        Lens {
            entries: chunk_ids
                .into_iter()
                .map(|id| LensEntry {
                    chunk_id: id.into(),
                    source_kind: "markdown".into(),
                    source_id: format!("{id}.md"),
                    slot_name: "attention",
                    slot_reason: "test".into(),
                    text_excerpt: "body".into(),
                    feature_breakdown: std::collections::BTreeMap::new(),
                    total_score: 0.0,
                    truncated: false,
                })
                .collect(),
            generated_at: Utc::now(),
            drift_basis: "rolling".into(),
            pinned: false,
        }
    }

    #[test]
    fn apply_refresh_updates_resource_emits_notification_and_logs_per_entry() {
        let tmp = TempDir::new().unwrap();
        let resource = MemoryLensResource::new(String::new());
        let registry = Arc::new(ResourceRegistry::new());
        registry.register(Arc::new(resource.clone()));
        registry
            .subscribe(
                ostk_recall_mcp::ClientId::stdio_singleton(),
                MEMORY_LENS_URI,
            )
            .unwrap();
        let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        registry.set_outbound(out_tx);

        let sink = RecordingSink::default();
        let mut state = LensState::default();

        let lens = dummy_lens(vec!["c1", "c2"]);
        let decision = LensRefreshDecision::Refresh {
            rendered: "rendered body".into(),
            lens,
            new_state: LensState {
                last_rolling_vec: Some(vec![1.0]),
                last_content_fp: Some(vec![1, 2, 3]),
                last_portfolio_chunk_ids: vec!["c1".into(), "c2".into()],
                ..LensState::default()
            },
        };
        apply_decision(
            decision,
            &mut state,
            &resource,
            &registry,
            &sink,
            tmp.path(),
        );

        // 1. Resource body updated.
        assert_eq!(resource.snapshot(), "rendered body");
        // 2. Notification on the channel.
        let env = out_rx.try_recv().expect("notification sent");
        assert!(env.contains("notifications/resources/updated"));
        assert!(env.contains(MEMORY_LENS_URI));
        // 3. Two LensIncluded events.
        let events = sink.events.lock().unwrap().clone();
        let lens_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ChainEvent::LensIncluded { .. }))
            .collect();
        assert_eq!(lens_events.len(), 2);
        // 4. State advanced + persisted.
        assert!(state.last_content_fp.is_some());
        assert!(tmp.path().join("lens_state.json").exists());
    }

    #[test]
    fn apply_build_failed_does_not_touch_resource_or_emit() {
        let tmp = TempDir::new().unwrap();
        let resource = MemoryLensResource::new("last good".into());
        let registry = Arc::new(ResourceRegistry::new());
        registry.register(Arc::new(resource.clone()));
        registry
            .subscribe(
                ostk_recall_mcp::ClientId::stdio_singleton(),
                MEMORY_LENS_URI,
            )
            .unwrap();
        let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        registry.set_outbound(out_tx);

        let sink = NoopChainSink;
        let mut state = LensState::default();

        apply_decision(
            LensRefreshDecision::BuildFailed("synthetic".into()),
            &mut state,
            &resource,
            &registry,
            &sink,
            tmp.path(),
        );

        assert_eq!(
            resource.snapshot(),
            "last good",
            "build failure must leave last lens visible"
        );
        assert!(
            out_rx.try_recv().is_err(),
            "build failure must not emit a notification"
        );
        assert!(
            state.last_content_fp.is_none(),
            "build failure must not advance state.last_content_fp"
        );
    }

    #[test]
    fn apply_unchanged_content_does_not_emit_but_persists_state() {
        let tmp = TempDir::new().unwrap();
        let resource = MemoryLensResource::new("stable body".into());
        let registry = Arc::new(ResourceRegistry::new());
        registry.register(Arc::new(resource.clone()));
        registry
            .subscribe(
                ostk_recall_mcp::ClientId::stdio_singleton(),
                MEMORY_LENS_URI,
            )
            .unwrap();
        let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        registry.set_outbound(out_tx);

        let sink = NoopChainSink;
        let mut state = LensState::default();
        apply_decision(
            LensRefreshDecision::UnchangedContent {
                new_state: LensState {
                    last_rolling_vec: Some(vec![0.5, 0.5]),
                    last_pin_fingerprint: None,
                    last_portfolio_chunk_ids: Vec::new(),
                    last_content_fp: Some(vec![9; 32]),
                    last_lens_ts: Some(Utc::now()),
                },
            },
            &mut state,
            &resource,
            &registry,
            &sink,
            tmp.path(),
        );

        assert_eq!(resource.snapshot(), "stable body", "body untouched");
        assert!(
            out_rx.try_recv().is_err(),
            "unchanged-content must not notify clients"
        );
        assert!(
            state.last_lens_ts.is_some(),
            "unchanged-content path must advance last_lens_ts"
        );
        assert_eq!(
            state.last_rolling_vec,
            Some(vec![0.5, 0.5]),
            "unchanged-content path must adopt the new rolling baseline"
        );
        assert_eq!(
            state.last_content_fp,
            Some(vec![9; 32]),
            "unchanged-content path must preserve the prior content fingerprint"
        );
        assert!(tmp.path().join("lens_state.json").exists());
    }

    #[tokio::test]
    async fn unchanged_content_advances_rolling_baseline_so_next_drift_compares_to_current_tick() {
        // Regression for the review fix. Before it, the
        // UnchangedContent arm of apply_decision only touched
        // last_lens_ts — the rolling baseline stayed anchored to
        // the original snapshot. Drift accumulated against the
        // stale baseline could eventually cross threshold for
        // byte-identical content and fire a spurious notification.
        //
        // Post-fix: UnchangedContent carries the advanced LensState
        // and apply_decision adopts it. The test asserts every
        // field the loop relies on for the next tick's gating
        // decision is copied forward.
        let tmp = TempDir::new().unwrap();
        let advanced = LensState {
            last_rolling_vec: Some(vec![0.0_f32, 1.0, 0.0]),
            last_pin_fingerprint: None,
            last_portfolio_chunk_ids: vec!["x".into()],
            last_content_fp: Some(vec![3_u8; 32]),
            last_lens_ts: Some(Utc::now()),
        };
        let resource = MemoryLensResource::new("body".into());
        let registry = Arc::new(ResourceRegistry::new());
        registry.register(Arc::new(resource.clone()));
        let sink = NoopChainSink;
        let mut state = LensState::default();
        apply_decision(
            LensRefreshDecision::UnchangedContent {
                new_state: advanced.clone(),
            },
            &mut state,
            &resource,
            &registry,
            &sink,
            tmp.path(),
        );
        assert_eq!(state.last_rolling_vec, advanced.last_rolling_vec);
        assert_eq!(state.last_content_fp, advanced.last_content_fp);
        assert_eq!(
            state.last_portfolio_chunk_ids,
            advanced.last_portfolio_chunk_ids
        );
    }
}
