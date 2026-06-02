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

use ostk_recall_store::{ChainLogReader, ConceptActivationReader, RelationalSupport};

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
    /// Concept-activation reader (memory-activation-frame.md slice 1),
    /// consumed by the [`crate::concept::ConceptSupportFactory`] feature's
    /// `prepare()` to score candidates by the activation of the concepts
    /// that cite them. `None` on the explicit-recall path and until the
    /// lens loop wires the reader; the feature degrades to a zero
    /// contribution (the concept slot skips cleanly) when absent.
    pub concept_reader: Option<Arc<dyn ConceptActivationReader>>,
    /// Per-lens-build cache of the diffusion result (relational-substrate
    /// slice 2). `build_lens` computes `relational_support` ONCE (for the
    /// candidate-injection lane) and stashes it here so the `relational_lift`
    /// feature reuses it instead of re-running the BFS + evidence reads during
    /// ranking. `None` on the standalone-feature path (the feature then
    /// computes it from `concept_reader`).
    pub relational_support: Option<Arc<RelationalSupport>>,
    /// P9b-full — `true` when an operator pin is driving ranking. The lens
    /// loop sets this explicitly from its pin-fingerprint detection;
    /// [`crate::lens`]'s `pinned()` heuristic (scope ≠ rolling) is the
    /// fallback used only by the P9b-min direct-build path.
    pub pinned: bool,
    /// P9b-full (dormant) — entity handles in recent attention, fed to
    /// pseudo-query construction. Empty until P6/P7 wire the entity ring;
    /// an empty list contributes nothing to `pseudo_query` or the
    /// entity-dominant slot.
    pub recent_entities: Vec<String>,
    /// P9b-full (dormant) — the dominant concept label for the current
    /// scope. `None` until P8 (concept overlay) lands; feeds the
    /// concept-dominant slot and the pseudo-query.
    pub dominant_concept_label: Option<String>,
    /// P9b-full (dormant) — synthetic query built from `recent_entities` +
    /// `dominant_concept_label`, consumed ONLY by the MaxSim (P4) rank
    /// feature. `None` when both inputs are empty (today's steady state),
    /// in which case MaxSim returns 0 for every candidate.
    pub pseudo_query: Option<String>,
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
            .field(
                "concept_reader",
                &self.concept_reader.as_ref().map_or("none", |_| "set"),
            )
            .field(
                "relational_support",
                &self
                    .relational_support
                    .as_ref()
                    .map_or("none", |_| "cached"),
            )
            .field("pinned", &self.pinned)
            .field("recent_entities", &self.recent_entities)
            .field("dominant_concept_label", &self.dominant_concept_label)
            .field("pseudo_query", &self.pseudo_query)
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
            concept_reader: None,
            relational_support: None,
            pinned: false,
            recent_entities: Vec::new(),
            dominant_concept_label: None,
            pseudo_query: None,
        }
    }

    #[must_use]
    pub fn with_scope_vector(vec: Vec<f32>) -> Self {
        Self {
            scope_vector: Some(vec),
            ..Self::empty()
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

    /// Attach a concept-activation reader so the
    /// [`crate::concept::ConceptSupportFactory`] feature can pull concept
    /// support in its `prepare()`. Additive builder.
    #[must_use]
    pub fn with_concept_reader(mut self, reader: Arc<dyn ConceptActivationReader>) -> Self {
        self.concept_reader = Some(reader);
        self
    }

    /// Stash a precomputed diffusion result so the `relational_lift` feature
    /// reuses it instead of recomputing (slice 2). `build_lens` calls this
    /// after running the diffusion once for the candidate-injection lane.
    #[must_use]
    pub fn with_relational_support(mut self, support: Arc<RelationalSupport>) -> Self {
        self.relational_support = Some(support);
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

    /// Phase-B lens enrichment (P9b-full, two-phase snapshot). The lens
    /// loop captures the attention vectors + pin state under the read
    /// guard (Phase A, sync, microseconds), drops the guard, then calls
    /// this to attach the access-ledger reader, set the explicit `pinned`
    /// flag, and build the pseudo-query. `async` honors the two-phase
    /// contract and reserves the seam for P8's async concept lookup;
    /// today it awaits nothing (entities/concept arrive empty), so
    /// `pseudo_query` is `None` and the MaxSim (P4) feature contributes 0.
    #[must_use]
    #[allow(clippy::unused_async)] // async reserves the P8 concept-lookup seam
    #[allow(clippy::too_many_arguments)]
    pub async fn enrich_for_lens(
        scope_vector: Option<Vec<f32>>,
        rolling_vec: Option<Vec<f32>>,
        pinned: bool,
        chain_log: Option<Arc<dyn ChainLogReader>>,
        concept_reader: Option<Arc<dyn ConceptActivationReader>>,
        recent_entities: Vec<String>,
        dominant_concept_label: Option<String>,
    ) -> Self {
        let pseudo_query = build_pseudo_query(&recent_entities, dominant_concept_label.as_deref());
        Self {
            scope_vector,
            rolling_vec,
            chain_log,
            concept_reader,
            relational_support: None,
            pinned,
            recent_entities,
            dominant_concept_label,
            pseudo_query,
        }
    }
}

/// Build the synthetic pseudo-query the MaxSim (P4) rank feature consumes
/// from the dominant concept label + recent entity handles. Returns `None`
/// when there is no signal (both inputs empty), so MaxSim degrades to a
/// zero contribution rather than ranking on an empty string.
fn build_pseudo_query(entities: &[String], concept: Option<&str>) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    if let Some(c) = concept {
        if !c.is_empty() {
            parts.push(c);
        }
    }
    for e in entities {
        if !e.is_empty() {
            parts.push(e.as_str());
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}
