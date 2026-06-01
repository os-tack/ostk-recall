//! `ConceptSupport` — the lens's concept rank feature (the missing wire).
//!
//! `memory-activation-frame.md` slice 1. The lens portfolio
//! ([`crate::lens::default_slots`]) ships a dormant `concept` slot keyed on
//! the feature name `"concept_support"`. This is the feature that lights it:
//! a candidate chunk scores by the **activation of the most-active concept
//! that cites it as evidence**, read from the concept-activation surface
//! ([`ostk_recall_store::ConceptActivationReader`]) — the same source of
//! truth `memory_surface(now)`'s frame reads. One activation core, two
//! projections; registering this feature wires the lens projection with zero
//! allocator changes.
//!
//! Structured exactly like the P7b [`crate::freshness::FreshnessFactory`]:
//! a stateful [`RankFeatureInstance`] whose async `prepare()` does the
//! ledger I/O once per query and whose `score()` is pure lookup. It degrades
//! to a zero contribution (so the concept slot skips cleanly) whenever the
//! reader is absent or no active concept cites any candidate — the correct
//! null state for a fresh corpus.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{Duration, Utc};

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};
use crate::error::Result;
use crate::rank::{RankFeatureFactory, RankFeatureInstance};

/// How far back the concept chain is consulted (mirrors the store default).
pub const DEFAULT_WINDOW_DAYS: i64 = 30;

/// Immutable config for the `ConceptSupport` feature; cheap to `Arc`-share.
#[derive(Debug, Clone)]
pub struct ConceptSupportFactory {
    window: Duration,
}

impl ConceptSupportFactory {
    #[must_use]
    pub const fn new(window: Duration) -> Self {
        Self { window }
    }
}

impl Default for ConceptSupportFactory {
    fn default() -> Self {
        Self {
            window: Duration::days(DEFAULT_WINDOW_DAYS),
        }
    }
}

impl RankFeatureFactory for ConceptSupportFactory {
    fn name(&self) -> &'static str {
        "concept_support"
    }
    fn build_instance(&self) -> Box<dyn RankFeatureInstance> {
        Box::new(ConceptSupportInstance {
            window: self.window,
            pool_min: 0.0,
            pool_max: 0.0,
            raw: HashMap::new(),
        })
    }
}

/// Per-query instance. `raw` maps `chunk_id → supporting concept activation`;
/// pool min/max are the min-max normalization bounds. All dropped when
/// `rank()` returns.
pub struct ConceptSupportInstance {
    window: Duration,
    pool_min: f32,
    pool_max: f32,
    raw: HashMap<String, f32>,
}

#[async_trait]
impl RankFeatureInstance for ConceptSupportInstance {
    fn name(&self) -> &'static str {
        "concept_support"
    }

    async fn prepare(
        &mut self,
        candidates: &mut [Candidate],
        _query: &QueryContext,
        attn: &AttentionContext,
    ) -> Result<()> {
        // Absent reader (explicit path / not yet wired) → no support, the
        // concept slot skips cleanly.
        let Some(reader) = attn.concept_reader.as_ref() else {
            return Ok(());
        };
        let since = Utc::now() - self.window;
        // One ledger read for the whole pool: coordinate → highest-activation
        // citing concept. Keyed `(source, source_id)`, the durable coordinate
        // that matches `Chunk { source, source_id }`.
        let support = reader.concept_support_by_coord(since)?;

        let mut raw = HashMap::with_capacity(candidates.len());
        for c in candidates.iter() {
            let key = (
                c.chunk.source.as_str().to_string(),
                c.chunk.source_id.clone(),
            );
            let s = support.get(&key).map_or(0.0, |cs| cs.activation);
            raw.insert(c.chunk.chunk_id.clone(), s);
        }
        self.pool_min = raw.values().copied().fold(f32::INFINITY, f32::min);
        self.pool_max = raw.values().copied().fold(f32::NEG_INFINITY, f32::max);
        self.raw = raw;
        Ok(())
    }

    fn score(&self, candidate: &Candidate, _query: &QueryContext, _attn: &AttentionContext) -> f32 {
        let raw = self
            .raw
            .get(&candidate.chunk.chunk_id)
            .copied()
            .unwrap_or(0.0);
        // Unlike freshness, a flat/empty pool returns 0.0, NOT a neutral
        // 0.5: concept support is a *sparse* signal — most chunks support no
        // active concept. Absence means "no concept fires here," so the
        // dominance slot finds nothing and skips, rather than the feature
        // contributing a spurious 0.5 to every candidate.
        if self.pool_max <= 1e-6 || (self.pool_max - self.pool_min) < 1e-6 {
            return 0.0;
        }
        (raw - self.pool_min) / (self.pool_max - self.pool_min)
    }
}

#[cfg(test)]
mod tests {
    // Exact float asserts on normalized 0.0 / 1.0 feature outputs.
    #![allow(clippy::float_cmp)]
    use std::collections::HashMap as Map;
    use std::sync::Arc;

    use chrono::DateTime;
    use ostk_recall_core::{Chunk, FacetSet, Links, Source};
    use ostk_recall_store::{ConceptActivation, ConceptActivationReader, ConceptSupport};

    use super::*;
    use crate::rank::RankEngine;

    // A stub reader returning a fixed coordinate → support map, so the
    // feature can be tested without a SQLite ledger.
    struct StubReader(Map<(String, String), ConceptSupport>);

    impl ConceptActivationReader for StubReader {
        fn concept_activations(
            &self,
            _project: Option<&str>,
            _since: DateTime<chrono::Utc>,
        ) -> std::result::Result<Vec<ConceptActivation>, ostk_recall_store::StoreError> {
            Ok(Vec::new())
        }
        fn concept_support_by_coord(
            &self,
            _since: DateTime<chrono::Utc>,
        ) -> std::result::Result<Map<(String, String), ConceptSupport>, ostk_recall_store::StoreError>
        {
            Ok(self.0.clone())
        }
    }

    fn chunk_at(id: &str, source: Source, source_id: &str) -> Chunk {
        Chunk {
            chunk_id: id.to_string(),
            source,
            project: None,
            source_id: source_id.to_string(),
            source_config_id: "test:cfg".to_string(),
            chunk_index: 0,
            ts: None,
            role: None,
            text: format!("text {id}"),
            sha256: format!("sha-{id}"),
            links: Links::default(),
            facets: FacetSet::default(),
            embedding_input_sha256: format!("emb-{id}"),
            extra: serde_json::Value::Null,
        }
    }

    fn cand(id: &str, source: Source, source_id: &str) -> Candidate {
        Candidate::for_chunk(chunk_at(id, source, source_id))
    }

    #[tokio::test]
    async fn absent_reader_contributes_zero() {
        let engine =
            RankEngine::new().with_factory(Arc::new(ConceptSupportFactory::default()), 1.0);
        let hits = engine
            .rank(
                vec![cand("a", Source::Markdown, "src/a")],
                &QueryContext::Ambient,
                &AttentionContext::empty(),
            )
            .await
            .unwrap();
        assert_eq!(hits[0].features.get("concept_support").unwrap().raw, 0.0);
    }

    #[tokio::test]
    async fn supported_chunk_outranks_unsupported() {
        // Only the `thread/shared` coordinate has concept support.
        let mut map = Map::new();
        map.insert(
            ("thread".to_string(), "shared".to_string()),
            ConceptSupport {
                handle: "mish".to_string(),
                activation: 1.4,
            },
        );
        let reader: Arc<dyn ConceptActivationReader> = Arc::new(StubReader(map));
        let attn = AttentionContext::empty().with_concept_reader(reader);

        let engine =
            RankEngine::new().with_factory(Arc::new(ConceptSupportFactory::default()), 1.0);
        let hits = engine
            .rank(
                vec![
                    cand("supported", Source::Thread, "shared"),
                    cand("plain", Source::Markdown, "src/plain"),
                ],
                &QueryContext::Ambient,
                &attn,
            )
            .await
            .unwrap();
        // The supported chunk normalizes to 1.0 and leads; the plain one 0.0.
        assert_eq!(hits[0].candidate.chunk.chunk_id, "supported");
        assert_eq!(hits[0].features.get("concept_support").unwrap().raw, 1.0);
        assert_eq!(hits[1].features.get("concept_support").unwrap().raw, 0.0);
    }

    #[tokio::test]
    async fn no_supported_candidate_is_all_zero() {
        // Reader has support, but for a coordinate no candidate carries →
        // sparse-signal path returns 0.0 for everyone (slot skips cleanly).
        let mut map = Map::new();
        map.insert(
            ("thread".to_string(), "elsewhere".to_string()),
            ConceptSupport {
                handle: "x".to_string(),
                activation: 0.9,
            },
        );
        let reader: Arc<dyn ConceptActivationReader> = Arc::new(StubReader(map));
        let attn = AttentionContext::empty().with_concept_reader(reader);
        let engine =
            RankEngine::new().with_factory(Arc::new(ConceptSupportFactory::default()), 1.0);
        let hits = engine
            .rank(
                vec![cand("a", Source::Markdown, "src/a")],
                &QueryContext::Ambient,
                &attn,
            )
            .await
            .unwrap();
        assert_eq!(hits[0].features.get("concept_support").unwrap().raw, 0.0);
    }
}
