//! Rank engine and feature-attribution types.
//!
//! **P3B — factory / instance rank features with async `prepare()`.**
//!
//! A rank feature is a [`RankFeatureFactory`] (immutable config, shared
//! via `Arc`, configured once at engine-build time) that mints a fresh
//! [`RankFeatureInstance`] per `rank()` call. The instance has a
//! two-phase shape:
//! - async `prepare(&mut self, …)` — pool-level setup that may await
//!   Lance/SQLite I/O (bulk-fetch embeddings, access histories,
//!   centroids), runs once per query before any scoring.
//! - sync `score(&self, …)` — pure per-candidate math over the state
//!   `prepare()` computed.
//!
//! `RankEngine::rank` takes `&self` and is `async`, so concurrent recall
//! and lens calls do not serialize and never share mutable feature
//! state — each call gets its own instances, dropped when `rank()`
//! returns (P9b-min / codex-A1 concurrency invariant).
//!
//! Stateless score-only features (`RRF`, attention-affinity, the test
//! features) use the [`FnFactory`] adapter so they don't each hand-write
//! the trait; genuinely-stateful enrichment features (`Freshness` P7b,
//! `EntitySalience` P7, `ConceptSupport` P8, `MaxSim` P4) implement
//! [`RankFeatureInstance`] directly to own their `prepare()` I/O + scratch.
//!
//! Alpha.1 default weights: only `Rrf = 1.0`; all other features at
//! `0.0`. This makes retrieval numerically equivalent to v0.5 on
//! the golden corpus; non-zero defaults arrive once P5 measures.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};
use crate::error::Result;

/// Pure scoring closure shared by [`FnFactory`] and its per-query
/// `FnInstance`. `Arc` so a single closure backs every instance the
/// factory mints; `Fn` (not `FnMut`) so it cannot hold mutable state.
type ScoreClosure = Arc<dyn Fn(&Candidate, &QueryContext, &AttentionContext) -> f32 + Send + Sync>;

/// Per-feature score breakdown. `raw` is the `[0, 1]` feature output;
/// `weight` is the engine-configured multiplier; `contribution` is
/// `raw * weight`. P9b portfolio slot dominance reads `contribution`
/// (raw with a zero weight is misleadingly large); debug UIs surface
/// both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureAttribution {
    pub raw: f32,
    pub weight: f32,
    pub contribution: f32,
}

impl FeatureAttribution {
    #[must_use]
    pub fn new(raw: f32, weight: f32) -> Self {
        Self {
            raw,
            weight,
            contribution: raw * weight,
        }
    }
}

/// One ranked candidate plus its per-feature attribution.
///
/// `total` is `Σ feature.contribution`. The map is ordered by feature
/// name so debug output and golden-file comparison are stable across
/// runs.
#[derive(Clone, Debug)]
pub struct RankedHit {
    pub candidate: Candidate,
    pub total: f32,
    pub features: BTreeMap<&'static str, FeatureAttribution>,
}

/// Immutable feature config, configured once when the engine is built
/// and shared across concurrent `rank()` calls via `Arc`. Mints a fresh
/// [`RankFeatureInstance`] per call so per-query scratch state never
/// leaks across calls.
pub trait RankFeatureFactory: Send + Sync {
    fn name(&self) -> &'static str;
    fn build_instance(&self) -> Box<dyn RankFeatureInstance>;
}

/// Per-query feature state. Built fresh by `build_instance()` on every
/// `rank()` call and dropped when it returns. Scratch state (`HashMap`s,
/// pool min/max, fetched embeddings) lives here, never on the factory.
///
/// Two-phase: `prepare()` runs once per query before any scoring and may
/// await Lance/SQLite I/O; `score()` is pure per-candidate math over the
/// state prepared. Never call `block_on` inside either — `prepare()` is
/// async precisely so I/O has somewhere to await.
#[async_trait]
pub trait RankFeatureInstance: Send {
    fn name(&self) -> &'static str;

    /// Pool-level setup — runs once per query before scoring. Default
    /// no-op for stateless features.
    async fn prepare(
        &mut self,
        _candidates: &mut [Candidate],
        _query: &QueryContext,
        _attn: &AttentionContext,
    ) -> Result<()> {
        Ok(())
    }

    /// Per-candidate score in `[0, 1]`. Pure in-memory math after
    /// `prepare()`; synchronous on purpose (no `.await` in the hot loop).
    fn score(&self, candidate: &Candidate, query: &QueryContext, attn: &AttentionContext) -> f32;
}

/// Stateless score-only feature adapter. Wraps a pure closure into the
/// factory/instance shape so trivial features (RRF, attention-affinity)
/// don't each hand-write the trait.
///
/// **For STATELESS features only.** The closure is shared (`Arc`) across
/// every per-query instance, so it must not capture mutable shared state
/// — anything needing per-query scratch or `prepare()` I/O must implement
/// [`RankFeatureInstance`] directly.
#[derive(Clone)]
pub struct FnFactory {
    name: &'static str,
    score_fn: ScoreClosure,
}

impl FnFactory {
    pub fn new(
        name: &'static str,
        score_fn: impl Fn(&Candidate, &QueryContext, &AttentionContext) -> f32 + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            score_fn: Arc::new(score_fn),
        }
    }
}

impl RankFeatureFactory for FnFactory {
    fn name(&self) -> &'static str {
        self.name
    }
    fn build_instance(&self) -> Box<dyn RankFeatureInstance> {
        Box::new(FnInstance {
            name: self.name,
            score_fn: Arc::clone(&self.score_fn),
        })
    }
}

/// Per-query instance for an [`FnFactory`]. Holds no mutable state — the
/// `Arc<closure>` clone is a refcount bump — so it inherits the trait's
/// default no-op `prepare()`.
struct FnInstance {
    name: &'static str,
    score_fn: ScoreClosure,
}

#[async_trait]
impl RankFeatureInstance for FnInstance {
    fn name(&self) -> &'static str {
        self.name
    }
    fn score(&self, candidate: &Candidate, query: &QueryContext, attn: &AttentionContext) -> f32 {
        (self.score_fn)(candidate, query, attn)
    }
}

/// Arc-safe rank engine.
///
/// Build once at startup with the configured `(factory, weight)` set;
/// share via `Arc<RankEngine>` across recall and lens-loop tasks. The
/// engine itself is immutable. `rank` takes `&self` and is `async`
/// (features may await I/O in `prepare()`); per-query instances are
/// built and dropped inside each call, so there is no shared mutable
/// state and no contention across concurrent calls.
pub struct RankEngine {
    factories: Vec<(Arc<dyn RankFeatureFactory>, f32)>,
}

impl RankEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
        }
    }

    /// Builder: append a feature factory with its weight. Order does not
    /// matter — the attribution map is sorted by name.
    #[must_use]
    pub fn with_factory(mut self, factory: Arc<dyn RankFeatureFactory>, weight: f32) -> Self {
        self.factories.push((factory, weight));
        self
    }

    /// Convenience builder for a stateless score-only feature — wraps the
    /// closure in an [`FnFactory`]. Equivalent to
    /// `with_factory(Arc::new(FnFactory::new(name, f)), weight)`.
    #[must_use]
    pub fn with_fn_feature(
        self,
        name: &'static str,
        weight: f32,
        score_fn: impl Fn(&Candidate, &QueryContext, &AttentionContext) -> f32 + Send + Sync + 'static,
    ) -> Self {
        self.with_factory(Arc::new(FnFactory::new(name, score_fn)), weight)
    }

    /// Number of registered features.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.factories.len()
    }

    /// Names of registered features, useful for introspection / tests.
    pub fn feature_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.factories.iter().map(|(f, _)| f.name())
    }

    /// Score `candidates` and return them sorted by descending `total`.
    ///
    /// Phase 1: build a per-query instance from each factory and run its
    /// `prepare()` (sequentially — Lance shares a runtime; independent
    /// prepares could later run via `join_all`). Phase 2: pure per-candidate
    /// scoring. Raw feature outputs are clamped to `[0, 1]` so a
    /// misbehaving feature can't dominate the sum.
    pub async fn rank(
        &self,
        mut candidates: Vec<Candidate>,
        query: &QueryContext,
        attn: &AttentionContext,
    ) -> Result<Vec<RankedHit>> {
        // Phase 1: per-query instances + pool-level setup.
        let mut instances: Vec<(Box<dyn RankFeatureInstance>, f32)> = self
            .factories
            .iter()
            .map(|(f, w)| (f.build_instance(), *w))
            .collect();
        for (instance, _) in &mut instances {
            instance.prepare(&mut candidates, query, attn).await?;
        }

        // Phase 2: pure CPU scoring.
        let mut hits = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let mut features: BTreeMap<&'static str, FeatureAttribution> = BTreeMap::new();
            let mut total = 0.0_f32;
            for (instance, weight) in &instances {
                let raw = instance.score(&candidate, query, attn).clamp(0.0, 1.0);
                let attribution = FeatureAttribution::new(raw, *weight);
                total += attribution.contribution;
                features.insert(instance.name(), attribution);
            }
            hits.push(RankedHit {
                candidate,
                total,
                features,
            });
        }
        // Sort by descending `total`; break ties on ascending
        // `chunk_id` so equal-RRF outputs are deterministic across
        // runs. HashMap iteration upstream (lane union, in-memory
        // attention state) is unordered, so the engine has to enforce
        // determinism — golden tests would otherwise drift on ties.
        hits.sort_by(|a, b| {
            b.total
                .partial_cmp(&a.total)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.candidate.chunk.chunk_id.cmp(&b.candidate.chunk.chunk_id))
        });
        Ok(hits)
        // instances dropped here — all per-query scratch state released.
    }
}

impl Default for RankEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---- attention features ------------------------------------------------

/// Direct attention-affinity score, used by P9b-min and as a future
/// `RankFeature` instance (P3B will wrap it behind the
/// `RankFeatureFactory` shape once `prepare()`-needing features ship).
///
/// Returns cosine similarity between the attention scope vector and
/// the candidate's dense embedding, clamped to `[0, 1]`. Cosine on
/// normalized vectors actually lives in `[-1, 1]`, but anti-resonant
/// candidates should contribute zero — they are not "attended to" in
/// any meaningful sense, just orthogonal in the embedding space.
///
/// Inputs `None` ⇒ score 0:
/// - `attn.scope_vector == None`: empty-mind boot (no pin, no
///   rolling, no transient). P9b-min skips lens injection in this
///   case; we still need the rank path to be a no-op rather than
///   bias toward the all-zeros vector.
/// - `candidate.dense_embedding == None`: candidate row had no
///   embedding column (legacy or non-projected). The other lanes
///   still drove it into the candidate set, but attention-axis
///   cannot contribute without an embedding.
#[must_use]
pub fn attention_affinity_score(candidate: &Candidate, attn: &AttentionContext) -> f32 {
    let Some(scope_vec) = attn.scope_vector.as_deref() else {
        return 0.0;
    };
    let Some(emb) = candidate.dense_embedding.as_deref() else {
        return 0.0;
    };
    cosine_similarity(scope_vec, emb).max(0.0)
}

/// Cosine similarity over `[-1, 1]`. Returns 0 on empty vectors or
/// dimension mismatch — the safe choice so an upstream embedder swap
/// can't produce a spurious score.
///
/// Inlined here rather than depending on `ostk-recall-attention` for
/// the helper: the query crate sits below attention in the crate
/// graph and must stay independent of it.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
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
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// `RankFeatureFactory` wrapper around [`attention_affinity_score`] (P9b-full).
///
/// `attention_affinity` is stateless — the score is a pure cosine over the
/// candidate embedding and the attention scope vector, with no pool-level
/// `prepare()` I/O. P9b-min registered it via the anonymous [`FnFactory`]
/// adapter; P9b-full promotes it to a named factory so the lens engine's
/// feature set reads as first-class types alongside the stateful
/// [`crate::freshness::FreshnessFactory`], and so a future `prepare()`
/// (e.g. bulk embedding projection) has a home. Behaviorally identical to
/// the `with_fn_feature("attention_affinity", …)` registration it replaces.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttentionAffinityFactory;

impl RankFeatureFactory for AttentionAffinityFactory {
    fn name(&self) -> &'static str {
        "attention_affinity"
    }
    fn build_instance(&self) -> Box<dyn RankFeatureInstance> {
        Box::new(AttentionAffinityInstance)
    }
}

/// Per-query instance for [`AttentionAffinityFactory`]. Holds no state —
/// inherits the trait's default no-op `prepare()`.
struct AttentionAffinityInstance;

#[async_trait]
impl RankFeatureInstance for AttentionAffinityInstance {
    fn name(&self) -> &'static str {
        "attention_affinity"
    }
    fn score(&self, candidate: &Candidate, _query: &QueryContext, attn: &AttentionContext) -> f32 {
        attention_affinity_score(candidate, attn)
    }
}

// ---- config-driven engine construction (P5) ---------------------------

/// Build a [`RankEngine`] from a profile weight map (P5).
///
/// **Single source of truth for engine construction** — both production
/// `recall` / lens paths and the `rank_bench` harness build their engines
/// here, so a weight that ships and a weight that was benched can never
/// drift. The map comes from [`ostk_recall_core::Config::effective_ranking_weights`]
/// (a profile's compiled-in defaults overlaid by `[ranking.weights.*]`).
///
/// Only features named in `weights` are registered; an entry's value is the
/// engine weight. This preserves production parity: the explicit profile's
/// compiled default is `{rrf: 1.0}`, so the built engine registers exactly
/// `rrf` at weight 1.0 — numerically identical to the pre-P5 hardcoded
/// builder. Unknown feature ids are ignored (forward-compat for features
/// that land in a later phase but appear in a hand-edited config).
///
/// Known feature ids:
/// - `rrf` — normalized reciprocal-rank fusion of the lane evidence.
/// - `bm25` — soft-sigmoid-normalized raw BM25 (`s / (s + K_BM25)`).
/// - `attention_affinity` — cosine(scope_vector, candidate embedding).
/// - `freshness` — ACT-R access-recency (P7b); reads `attn.chain_log` when
///   present, else falls back to chunk creation time. Registered with
///   [`crate::freshness::FreshnessFactory::default`] tuning.
#[must_use]
pub fn build_engine_from_weights(weights: &BTreeMap<String, f32>) -> RankEngine {
    let mut engine = RankEngine::new();
    for (name, &weight) in weights {
        engine = match name.as_str() {
            "rrf" => engine.with_fn_feature("rrf", weight, |c, _q, _a| {
                c.rrf_score.map_or(0.0, crate::lanes::rrf_score_normalized)
            }),
            "bm25" => engine.with_fn_feature("bm25", weight, |c, _q, _a| {
                c.bm25_score
                    .map_or(0.0, |s| s / (s + crate::hybrid::K_BM25))
            }),
            "attention_affinity" => engine.with_factory(Arc::new(AttentionAffinityFactory), weight),
            "freshness" => engine.with_factory(
                Arc::new(crate::freshness::FreshnessFactory::default()),
                weight,
            ),
            // `concept_support` — lens concept slot (memory-activation-frame.md).
            // Reads `attn.concept_reader` when present (the lens loop wires it),
            // else degrades to a zero contribution. Registering it lights the
            // dormant concept portfolio slot with zero allocator changes.
            "concept_support" => engine.with_factory(
                Arc::new(crate::concept::ConceptSupportFactory::default()),
                weight,
            ),
            // Unknown id: a feature not (yet) known to this builder. Ignore
            // rather than panic so a forward-looking config doesn't break an
            // older binary.
            _ => engine,
        };
    }
    engine
}

#[cfg(test)]
mod tests {
    use ostk_recall_core::{Chunk, FacetSet, Links, Source};

    use super::*;

    #[test]
    fn build_engine_registers_only_mapped_known_features() {
        let mut w = BTreeMap::new();
        w.insert("rrf".to_string(), 1.0);
        w.insert("freshness".to_string(), 0.5);
        w.insert("unknown_feature_xyz".to_string(), 9.0);
        let engine = build_engine_from_weights(&w);
        let names: Vec<&str> = engine.feature_names().collect();
        assert!(names.contains(&"rrf"), "rrf registered");
        assert!(names.contains(&"freshness"), "freshness registered");
        assert!(
            !names.contains(&"unknown_feature_xyz"),
            "unknown id must be ignored, not registered"
        );
        assert_eq!(engine.feature_count(), 2, "only known mapped features");
    }

    #[test]
    fn build_engine_explicit_default_is_rrf_only() {
        // Production parity: the explicit profile's compiled default
        // ({rrf: 1.0}) must build an engine identical in shape to the
        // pre-P5 hardcoded rrf-only builder.
        let w = ostk_recall_core::default_profile_weights(ostk_recall_core::RankProfile::Explicit);
        let engine = build_engine_from_weights(&w);
        let names: Vec<&str> = engine.feature_names().collect();
        assert_eq!(names, vec!["rrf"]);
    }

    #[test]
    fn build_engine_ambient_default_is_attention_only() {
        let w = ostk_recall_core::default_profile_weights(ostk_recall_core::RankProfile::Ambient);
        let engine = build_engine_from_weights(&w);
        let names: Vec<&str> = engine.feature_names().collect();
        assert_eq!(names, vec!["attention_affinity"]);
    }

    fn dummy_chunk(id: &str) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            source: Source::Markdown,
            project: None,
            source_id: format!("src/{id}"),
            source_config_id: "test:cfg".to_string(),
            chunk_index: 0,
            ts: None,
            role: None,
            text: format!("hello {id}"),
            sha256: format!("sha-{id}"),
            links: Links::default(),
            facets: FacetSet::default(),
            embedding_input_sha256: format!("emb-{id}"),
            extra: serde_json::Value::Null,
        }
    }

    fn cand(id: &str) -> Candidate {
        Candidate::for_chunk(dummy_chunk(id))
    }

    #[test]
    fn feature_attribution_contribution_is_raw_times_weight() {
        let a = FeatureAttribution::new(0.5, 2.0);
        assert!((a.contribution - 1.0).abs() < 1e-6);
        let b = FeatureAttribution::new(0.0, 5.0);
        assert_eq!(b.contribution, 0.0);
        let c = FeatureAttribution::new(0.8, 0.0);
        assert_eq!(c.contribution, 0.0);
    }

    #[tokio::test]
    async fn rank_engine_clamps_raw_to_unit_interval() {
        let engine = RankEngine::new().with_fn_feature("out_of_range", 1.0, |_c, _q, _a| 2.5);
        let hits = engine
            .rank(
                vec![cand("a")],
                &QueryContext::Ambient,
                &AttentionContext::empty(),
            )
            .await
            .unwrap();
        let attr = hits[0].features.get("out_of_range").unwrap();
        assert_eq!(attr.raw, 1.0, "raw should clamp to [0, 1]");
        assert_eq!(attr.contribution, 1.0);
    }

    #[tokio::test]
    async fn rank_engine_sorts_by_total_descending() {
        // candidate "a" → 0.3, "b" → 0.7, "c" → 0.5
        let engine = RankEngine::new().with_fn_feature("byid", 1.0, |c, _q, _a| {
            match c.chunk.chunk_id.as_str() {
                "a" => 0.3,
                "b" => 0.7,
                "c" => 0.5,
                _ => 0.0,
            }
        });
        let hits = engine
            .rank(
                vec![cand("a"), cand("b"), cand("c")],
                &QueryContext::Ambient,
                &AttentionContext::empty(),
            )
            .await
            .unwrap();
        let order: Vec<&str> = hits
            .iter()
            .map(|h| h.candidate.chunk.chunk_id.as_str())
            .collect();
        assert_eq!(order, vec!["b", "c", "a"]);
    }

    #[tokio::test]
    async fn rank_engine_sums_contributions_across_features() {
        let engine = RankEngine::new()
            .with_fn_feature("f1", 1.0, |_c, _q, _a| 0.4)
            .with_fn_feature("f2", 2.0, |_c, _q, _a| 0.3);
        let hits = engine
            .rank(
                vec![cand("x")],
                &QueryContext::Ambient,
                &AttentionContext::empty(),
            )
            .await
            .unwrap();
        // 0.4 * 1.0 + 0.3 * 2.0 = 0.4 + 0.6 = 1.0
        assert!((hits[0].total - 1.0).abs() < 1e-6);
        assert_eq!(hits[0].features.len(), 2);
    }

    #[tokio::test]
    async fn rank_engine_empty_feature_list_produces_zero_totals() {
        let engine = RankEngine::new();
        let hits = engine
            .rank(
                vec![cand("a"), cand("b")],
                &QueryContext::Ambient,
                &AttentionContext::empty(),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
        for h in &hits {
            assert_eq!(h.total, 0.0);
            assert!(h.features.is_empty());
        }
    }

    #[tokio::test]
    async fn rank_engine_tiebreaks_on_chunk_id_deterministically() {
        // Three candidates with the same total — RankEngine must
        // order them by chunk_id ascending so golden tests stay
        // stable across runs.
        let engine = RankEngine::new().with_fn_feature("flat", 1.0, |_c, _q, _a| 0.5);
        // Submit in reversed order; expect ascending chunk_id back.
        let hits = engine
            .rank(
                vec![cand("c-charlie"), cand("a-alpha"), cand("b-bravo")],
                &QueryContext::Ambient,
                &AttentionContext::empty(),
            )
            .await
            .unwrap();
        let order: Vec<&str> = hits
            .iter()
            .map(|h| h.candidate.chunk.chunk_id.as_str())
            .collect();
        assert_eq!(order, vec!["a-alpha", "b-bravo", "c-charlie"]);
    }

    #[test]
    fn rank_engine_is_send_sync_via_arc() {
        // Compile-time check: an Arc<RankEngine> can be shared across
        // tasks (codex A1 / P9b-min concurrency invariant).
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<std::sync::Arc<RankEngine>>();
    }
}
