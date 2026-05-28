//! Rank engine and feature-attribution types.
//!
//! **P3A — minimal shape (Option A from the lens-first slice review).**
//!
//! Features are pure synchronous functions `(&Candidate, &QueryContext,
//! &AttentionContext) -> f32`. The engine owns an immutable list of
//! `(name, weight, score_fn)` triples and takes `&self` on `rank`, so
//! concurrent recall and lens calls do not serialize — P9b-min relies
//! on this (the lens loop runs on a dedicated task while explicit
//! recall continues on request tasks).
//!
//! The trait-and-factory machinery described in `p3-rank-evidence.md`
//! § "Phase cut: P3A vs P3B" (RankFeatureFactory + RankFeatureInstance
//! + async `prepare()`) does NOT ship here. It lands in P3B when the
//! first enrichment feature (MaxSim / Freshness / EntitySalience /
//! ConceptSupport) introduces per-query mutable state that needs
//! pool-level setup. Until then, direct functions are simpler.
//!
//! Alpha.1 default weights: only `Rrf = 1.0`; all other features at
//! `0.0`. This makes P3A retrieval numerically equivalent to v0.5 on
//! the golden corpus; non-zero defaults arrive once P5 measures.

use std::collections::BTreeMap;

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};

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

/// Boxed feature scoring closure. Sync on purpose — pool-level / async
/// setup is P3B work; P3A features (`Rrf`, `DenseSimilarity`, `Bm25`,
/// `AttentionAffinity`) all have the inputs they need on `Candidate`
/// or `AttentionContext` by the time `rank()` is called.
pub type ScoreFn = Box<dyn Fn(&Candidate, &QueryContext, &AttentionContext) -> f32 + Send + Sync>;

/// One feature: a name (used as the attribution-map key), a weight,
/// and the scoring closure.
pub struct Feature {
    pub name: &'static str,
    pub weight: f32,
    pub score_fn: ScoreFn,
}

impl std::fmt::Debug for Feature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Feature")
            .field("name", &self.name)
            .field("weight", &self.weight)
            .finish_non_exhaustive()
    }
}

/// Minimal Arc-safe rank engine.
///
/// Build once at startup with the configured `(feature, weight)` set;
/// share via `Arc<RankEngine>` across recall and lens-loop tasks. The
/// engine itself is immutable; per-candidate scoring is pure CPU.
///
/// Concurrency: `rank` takes `&self`, no internal locks. Each feature
/// closure is `Send + Sync`, so the same engine can serve N concurrent
/// recall + lens calls without contention.
#[derive(Debug)]
pub struct RankEngine {
    features: Vec<Feature>,
}

impl RankEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
        }
    }

    /// Builder: append a feature. Order does not matter — attribution
    /// map is sorted by name anyway.
    #[must_use]
    pub fn with_feature(mut self, feature: Feature) -> Self {
        self.features.push(feature);
        self
    }

    /// Number of registered features.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    /// Names of registered features, useful for introspection / tests.
    pub fn feature_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.features.iter().map(|f| f.name)
    }

    /// Score `candidates` and return them sorted by descending
    /// `total`. Raw feature outputs are clamped to `[0, 1]` so a
    /// misbehaving feature can't dominate the sum.
    #[must_use]
    pub fn rank(
        &self,
        candidates: Vec<Candidate>,
        query: &QueryContext,
        attn: &AttentionContext,
    ) -> Vec<RankedHit> {
        let mut hits = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let mut features: BTreeMap<&'static str, FeatureAttribution> = BTreeMap::new();
            let mut total = 0.0_f32;
            for f in &self.features {
                let raw = (f.score_fn)(&candidate, query, attn).clamp(0.0, 1.0);
                let attribution = FeatureAttribution::new(raw, f.weight);
                total += attribution.contribution;
                features.insert(f.name, attribution);
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
        hits
    }
}

impl Default for RankEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use ostk_recall_core::{Chunk, FacetSet, Links, Source};

    use super::*;

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

    #[test]
    fn rank_engine_clamps_raw_to_unit_interval() {
        let engine = RankEngine::new().with_feature(Feature {
            name: "out_of_range",
            weight: 1.0,
            score_fn: Box::new(|_c, _q, _a| 2.5),
        });
        let hits = engine.rank(
            vec![cand("a")],
            &QueryContext::Ambient,
            &AttentionContext::empty(),
        );
        let attr = hits[0].features.get("out_of_range").unwrap();
        assert_eq!(attr.raw, 1.0, "raw should clamp to [0, 1]");
        assert_eq!(attr.contribution, 1.0);
    }

    #[test]
    fn rank_engine_sorts_by_total_descending() {
        let engine = RankEngine::new().with_feature(Feature {
            name: "byid",
            weight: 1.0,
            // candidate "a" → 0.3, "b" → 0.7, "c" → 0.5
            score_fn: Box::new(|c, _q, _a| match c.chunk.chunk_id.as_str() {
                "a" => 0.3,
                "b" => 0.7,
                "c" => 0.5,
                _ => 0.0,
            }),
        });
        let hits = engine.rank(
            vec![cand("a"), cand("b"), cand("c")],
            &QueryContext::Ambient,
            &AttentionContext::empty(),
        );
        let order: Vec<&str> = hits
            .iter()
            .map(|h| h.candidate.chunk.chunk_id.as_str())
            .collect();
        assert_eq!(order, vec!["b", "c", "a"]);
    }

    #[test]
    fn rank_engine_sums_contributions_across_features() {
        let engine = RankEngine::new()
            .with_feature(Feature {
                name: "f1",
                weight: 1.0,
                score_fn: Box::new(|_c, _q, _a| 0.4),
            })
            .with_feature(Feature {
                name: "f2",
                weight: 2.0,
                score_fn: Box::new(|_c, _q, _a| 0.3),
            });
        let hits = engine.rank(
            vec![cand("x")],
            &QueryContext::Ambient,
            &AttentionContext::empty(),
        );
        // 0.4 * 1.0 + 0.3 * 2.0 = 0.4 + 0.6 = 1.0
        assert!((hits[0].total - 1.0).abs() < 1e-6);
        assert_eq!(hits[0].features.len(), 2);
    }

    #[test]
    fn rank_engine_empty_feature_list_produces_zero_totals() {
        let engine = RankEngine::new();
        let hits = engine.rank(
            vec![cand("a"), cand("b")],
            &QueryContext::Ambient,
            &AttentionContext::empty(),
        );
        assert_eq!(hits.len(), 2);
        for h in &hits {
            assert_eq!(h.total, 0.0);
            assert!(h.features.is_empty());
        }
    }

    #[test]
    fn rank_engine_tiebreaks_on_chunk_id_deterministically() {
        // Three candidates with the same total — RankEngine must
        // order them by chunk_id ascending so golden tests stay
        // stable across runs.
        let engine = RankEngine::new().with_feature(Feature {
            name: "flat",
            weight: 1.0,
            score_fn: Box::new(|_c, _q, _a| 0.5),
        });
        // Submit in reversed order; expect ascending chunk_id back.
        let hits = engine.rank(
            vec![cand("c-charlie"), cand("a-alpha"), cand("b-bravo")],
            &QueryContext::Ambient,
            &AttentionContext::empty(),
        );
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
