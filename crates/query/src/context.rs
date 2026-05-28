//! Query- and attention-context types consumed by the rank engine and
//! lane functions.
//!
//! **P3A — minimal shape.** The full `AttentionContext` defined in
//! `architecture.md` § "Context types" carries `rolling_vec`,
//! `recent_entities`, `active_thread_handles`, `pseudo_query`,
//! `chain_log`, and the two-phase `enrich_for_lens` pattern. Those
//! fields land in P6A (rolling vector + scope-vector priority chain)
//! and P9b-min (lens loop) as their owners ship.
//!
//! P3A consumes only `scope_vector`, which already exists today via
//! `InMemoryAttention::scope_vector()`. Keeping the type minimal here
//! avoids dragging the chain-log trait and concept-store handles into
//! a phase that has no consumer for them yet — they accrete when the
//! features that need them do.

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
#[derive(Clone, Debug, Default)]
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
}

impl AttentionContext {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            scope_vector: None,
            rolling_vec: None,
        }
    }

    #[must_use]
    pub fn with_scope_vector(vec: Vec<f32>) -> Self {
        Self {
            scope_vector: Some(vec),
            rolling_vec: None,
        }
    }

    /// Set both the effective scope vector and the rolling channel
    /// in one call. Used by P9b-min's enrich step (snapshot before
    /// the attention lock drops); also handy in tests.
    #[must_use]
    pub fn with_rolling(mut self, rolling: Vec<f32>) -> Self {
        self.rolling_vec = Some(rolling);
        self
    }
}
