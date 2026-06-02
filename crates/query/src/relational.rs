//! `relational_lift` — the lens's diffusion rank feature (relational-substrate
//! slice 2).
//!
//! Where [`crate::concept::ConceptSupportFactory`] lifts a chunk that is
//! evidence of a *directly-active* concept (0-hop), `relational_lift` lifts a
//! chunk that is evidence of a concept reached by **spreading activation** from
//! the attention-lit seeds across the edge graph (multi-hop, conductance- and
//! hop-decayed). The diffusion + coordinate bridge live in the store
//! ([`ostk_recall_store::ConceptActivationReader::relational_support`]); this
//! feature is the lens projection of it — a stateful [`RankFeatureInstance`]
//! whose `prepare()` does the one ledger pass and whose `score()` is pure
//! lookup. Like `concept_support` it is a **sparse** signal: a flat/empty pool
//! scores 0.0 so the relational slot skips cleanly.
//!
//! Surfacing (as opposed to re-ranking) neighbour chunks that dense retrieval
//! never returned is handled by the **relational candidate lane**
//! ([`crate::lanes::relational_candidates`]); this feature then scores the
//! unioned pool.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};
use crate::error::Result;
use crate::rank::{RankFeatureFactory, RankFeatureInstance};

/// Immutable config for the `relational_lift` feature; cheap to `Arc`-share.
///
/// The diffusion window is intentionally **not** configurable here: `build_lens`
/// computes the cached [`ostk_recall_store::RelationalSupport`] with
/// `ostk_recall_store::default_since_now()`, and the standalone fallback path
/// uses the same call — one window source, so the cache and the feature can
/// never disagree.
#[derive(Debug, Clone, Default)]
pub struct RelationalLiftFactory;

impl RankFeatureFactory for RelationalLiftFactory {
    fn name(&self) -> &'static str {
        "relational_lift"
    }
    fn build_instance(&self) -> Box<dyn RankFeatureInstance> {
        Box::new(RelationalLiftInstance {
            pool_min: 0.0,
            pool_max: 0.0,
            raw: HashMap::new(),
        })
    }
}

/// Per-query instance. `raw` maps `chunk_id → diffused activation of the
/// most-active concept reaching it`; pool min/max are the normalization bounds.
pub struct RelationalLiftInstance {
    pool_min: f32,
    pool_max: f32,
    raw: HashMap<String, f32>,
}

#[async_trait]
impl RankFeatureInstance for RelationalLiftInstance {
    fn name(&self) -> &'static str {
        "relational_lift"
    }

    async fn prepare(
        &mut self,
        candidates: &mut [Candidate],
        _query: &QueryContext,
        attn: &AttentionContext,
    ) -> Result<()> {
        // Prefer the cache `build_lens` populated (one diffusion per lens
        // build, shared with the injection lane). Fall back to computing it
        // from the reader on the standalone-feature path; if neither is
        // present (explicit recall / not wired) the slot skips.
        let support = if let Some(cached) = attn.relational_support.as_ref() {
            std::sync::Arc::clone(cached)
        } else if let Some(reader) = attn.concept_reader.as_ref() {
            // Same window source as build_lens's cache → the two never drift.
            std::sync::Arc::new(reader.relational_support(ostk_recall_store::default_since_now())?)
        } else {
            return Ok(());
        };

        let mut raw = HashMap::with_capacity(candidates.len());
        for c in candidates.iter() {
            let key = (
                c.chunk.source.as_str().to_string(),
                c.chunk.source_id.clone(),
            );
            let s = support.coord_activation.get(&key).copied().unwrap_or(0.0);
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
        // Sparse signal (like concept_support): a flat/empty pool returns 0.0,
        // not a neutral 0.5 — most chunks are reached by no diffusion, so the
        // dominance slot finds nothing and skips rather than every candidate
        // gaining a spurious 0.5.
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
    use ostk_recall_store::{
        ConceptActivation, ConceptActivationReader, ConceptSupport, RelationalSupport,
    };

    use super::*;
    use crate::rank::RankEngine;

    /// A stub reader returning a fixed diffused coordinate map.
    struct StubReader(RelationalSupport);

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
            Ok(Map::new())
        }
        fn relational_support(
            &self,
            _since: DateTime<chrono::Utc>,
        ) -> std::result::Result<RelationalSupport, ostk_recall_store::StoreError> {
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
            RankEngine::new().with_factory(Arc::new(RelationalLiftFactory::default()), 1.0);
        let hits = engine
            .rank(
                vec![cand("a", Source::Markdown, "src/a")],
                &QueryContext::Ambient,
                &AttentionContext::empty(),
            )
            .await
            .unwrap();
        assert_eq!(hits[0].features.get("relational_lift").unwrap().raw, 0.0);
    }

    #[tokio::test]
    async fn diffused_chunk_outranks_unreached() {
        // Only the `markdown/people/ostk-recall` coordinate is reached by
        // diffusion (e.g. the neighbour of the focused `tori`).
        let mut coord = Map::new();
        coord.insert(
            ("markdown".to_string(), "people/ostk-recall".to_string()),
            0.42,
        );
        let support = RelationalSupport {
            coord_activation: coord,
            inject_chunk_ids: vec!["reached".to_string()],
        };
        let reader: Arc<dyn ConceptActivationReader> = Arc::new(StubReader(support));
        let attn = AttentionContext::empty().with_concept_reader(reader);

        let engine =
            RankEngine::new().with_factory(Arc::new(RelationalLiftFactory::default()), 1.0);
        let hits = engine
            .rank(
                vec![
                    cand("reached", Source::Markdown, "people/ostk-recall"),
                    cand("plain", Source::Markdown, "src/plain"),
                ],
                &QueryContext::Ambient,
                &attn,
            )
            .await
            .unwrap();
        assert_eq!(hits[0].candidate.chunk.chunk_id, "reached");
        assert_eq!(hits[0].features.get("relational_lift").unwrap().raw, 1.0);
        assert_eq!(hits[1].features.get("relational_lift").unwrap().raw, 0.0);
    }

    #[tokio::test]
    async fn no_reached_candidate_is_all_zero() {
        // Diffusion reached a coordinate no candidate carries → sparse path
        // returns 0.0 for everyone (slot skips cleanly).
        let mut coord = Map::new();
        coord.insert(("thread".to_string(), "elsewhere".to_string()), 0.9);
        let support = RelationalSupport {
            coord_activation: coord,
            inject_chunk_ids: Vec::new(),
        };
        let reader: Arc<dyn ConceptActivationReader> = Arc::new(StubReader(support));
        let attn = AttentionContext::empty().with_concept_reader(reader);
        let engine =
            RankEngine::new().with_factory(Arc::new(RelationalLiftFactory::default()), 1.0);
        let hits = engine
            .rank(
                vec![cand("a", Source::Markdown, "src/a")],
                &QueryContext::Ambient,
                &attn,
            )
            .await
            .unwrap();
        assert_eq!(hits[0].features.get("relational_lift").unwrap().raw, 0.0);
    }
}
