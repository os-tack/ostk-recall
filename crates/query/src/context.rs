//! Query- and attention-context types consumed by the rank engine and
//! lane functions.
//!
//! `AttentionContext` grows by phase: `scope_vector` (P3A), `rolling_vec`
//! (P6A), and now `chain_log` (P7b — the access-ledger reader the ACT-R
//! freshness feature consumes). Remaining `architecture.md` fields
//! (`recent_entities`, `active_thread_handles`, `pseudo_query`, the
//! two-phase `enrich_for_lens` constructor) accrete as their owning
//! features (P7/P8/P9b-full) ship.

use std::sync::Arc;

use ostk_recall_store::ChainLogReader;

/// The query side of a rank call.
///
/// Two shapes, matching the two retrieval modes in
/// `architecture.md` § "Retrieval invariants":
/// - `Explicit` — the caller passes `recall(text)`. Full BM25 + dense
///   candidate generation runs.
/// - `Ambient` — the lens loop has no user text query. Dense-only
///   candidate generation; BM25 lane is OFF by invariant.
#[derive(Clone, Debug)]
pub enum QueryContext {
    Explicit { text: String, embedding: Vec<f32> },
    Ambient,
}

impl QueryContext {
    #[must_use]
    pub fn explicit(text: impl Into<String>, embedding: Vec<f32>) -> Self {
        Self::Explicit {
            text: text.into(),
            embedding,
        }
    }

    #[must_use]
    pub const fn is_ambient(&self) -> bool {
        matches!(self, Self::Ambient)
    }

    #[must_use]
    pub fn query_embedding(&self) -> Option<&[f32]> {
        match self {
            Self::Explicit { embedding, .. } => Some(embedding.as_slice()),
            Self::Ambient => None,
        }
    }

    #[must_use]
    pub fn query_text(&self) -> Option<&str> {
        match self {
            Self::Explicit { text, .. } => Some(text.as_str()),
            Self::Ambient => None,
        }
    }
}

/// Snapshot of attention state for a rank call.
///
/// **P3A — minimal.** Only `scope_vector` is consumed.
/// **P6A** adds `rolling_vec` — the EMA-blended channel used by the
/// drift trigger and the [`crate::rank::attention_affinity_score`]
/// path. The pin-precedence chain remains owned by
/// `InMemoryAttention::effective_vec()`, which already feeds
/// `scope_vector`. P9b-min adds the two-phase
/// `enrich_for_lens` constructor that does async concept-label /
/// pseudo-query work after the attention lock drops.
#[derive(Clone, Default)]
pub struct AttentionContext {
    /// Effective attention vector (`pinned → rolling → transient`).
    /// `None` only at empty-mind boot (no pin, no rolling, no
    /// transient — P9b-min's empty-mind-skip handles that case
    /// upstream).
    pub scope_vector: Option<Vec<f32>>,
    /// Raw rolling-EMA channel, snapshotted independently of the
    /// priority chain. `Some` once the scope has been attended at
    /// least once. P9b-min's drift trigger compares successive
    /// values directly; ranking still uses `scope_vector` so the
    /// operator's pin (if any) stays authoritative.
    pub rolling_vec: Option<Vec<f32>>,
    /// P7b — read handle on the chunk-access ledger, consumed by the
    /// ACT-R [`crate::freshness`] feature's `prepare()`. `None` on the
    /// explicit-recall path and until P9b-full wires the lens loop's
    /// reader; the Freshness feature degrades to `chunk.ts`-only
    /// (creation-recency) base activation when absent.
    pub chain_log: Option<Arc<dyn ChainLogReader>>,
}

// Hand-written: `Arc<dyn ChainLogReader>` is not `Debug`, so the struct
// can't derive it. Print the vecs as-is (preserving prior output) and the
// chain-log handle as a presence marker.
impl std::fmt::Debug for AttentionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttentionContext")
            .field("scope_vector", &self.scope_vector)
            .field("rolling_vec", &self.rolling_vec)
            .field(
                "chain_log",
                &self.chain_log.as_ref().map_or("none", |_| "set"),
            )
            .finish()
    }
}

impl AttentionContext {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            scope_vector: None,
            rolling_vec: None,
            chain_log: None,
        }
    }

    #[must_use]
    pub fn with_scope_vector(vec: Vec<f32>) -> Self {
        Self {
            scope_vector: Some(vec),
            rolling_vec: None,
            chain_log: None,
        }
    }

    /// Attach a read handle on the access ledger so the ACT-R
    /// [`crate::freshness`] feature can pull retrieval history in its
    /// `prepare()`. Additive builder; leaves the vectors untouched.
    #[must_use]
    pub fn with_chain_log(mut self, reader: Arc<dyn ChainLogReader>) -> Self {
        self.chain_log = Some(reader);
        self
    }

    /// Populate the rolling channel and, when no effective scope
    /// vector is set yet, seed `scope_vector` from the same vec.
    ///
    /// The two-field behaviour exists so callers using a bare
    /// `AttentionContext::default().with_rolling(v)` get a non-zero
    /// affinity score — [`crate::rank::attention_affinity_score`]
    /// reads `scope_vector` exclusively. But when `scope_vector`
    /// has already been set (typically by an operator pin via
    /// [`Self::with_scope_vector`]), pin precedence must win:
    /// `with_rolling` is purely additive in that case, and the pin
    /// stays authoritative for ranking. The runtime-side priority
    /// chain (`pinned → rolling → transient`) already resolved which
    /// vec belongs in `scope_vector`; the builder here must not
    /// rewrite that decision.
    #[must_use]
    pub fn with_rolling(mut self, rolling: Vec<f32>) -> Self {
        if self.scope_vector.is_none() {
            self.scope_vector = Some(rolling.clone());
        }
        self.rolling_vec = Some(rolling);
        self
    }
}
