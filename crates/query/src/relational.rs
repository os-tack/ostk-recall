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

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use ostk_recall_store::{
    CORPUS_TABLE, ConceptActivationReader, CorpusStore, EdgeSource, PROMOTED_EDGE_CONFIDENCE,
    REL_HOP_DECAY, REL_LATENT_K, REL_LATENT_SIM_FLOOR, REL_PROMOTE_TOP_K, REL_PROMOTED_RELATION,
    RelationalSupport, SeedAnchor, ThreadsDb,
};

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};
use crate::error::Result;
use crate::lanes::lane_dense;
use crate::rank::{RankFeatureFactory, RankFeatureInstance, cosine_similarity};

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

/// A latent (similarity) neighbour of a seed concept (relational-substrate
/// slice 2b): a concept whose evidence chunk is ANN-near the seed's anchor chunk
/// but has **no reified edge** to the seed — an off-diagonal bridge. Carries the
/// full coordinate `relational_lift` scores by, plus the concept's authoritative
/// scope for the same-project tiebreak and the promoted-edge audit event.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentNeighbor {
    pub concept_id: i64,
    pub project: String,
    pub handle: String,
    pub chunk_id: String,
    pub source: String,
    pub source_id: String,
    pub cosine: f32,
}

/// Extra ANN results fetched beyond `k` to absorb the anchor itself + same-
/// document chunks before they are filtered out.
const LATENT_SLACK: usize = 8;

/// One seed's latent hop: ANN over chunk vectors from its anchor chunk, mapped
/// back to off-diagonal neighbour concepts. Shared by the read augmenter
/// (Part A) and the promoter (Part B).
///
/// Order matters — the concept filters need the reverse map first:
/// 1. fetch the anchor embedding; absent → no hop;
/// 2. `nearest_to` with `k + slack` (the anchor + same-doc chunks come back);
///    drop the anchor chunk id;
/// 3. fetch candidate chunks + embeddings, cosine vs the anchor, drop below floor;
/// 4. reverse-map survivors to `(chunk, concept)` associations;
/// 5. drop associations whose concept is the seed's own or in `exclude` (the
///    caller's off-diagonal filter — the seed's authored/observed neighbours;
///    `promoted` neighbours are intentionally *not* excluded so they stay
///    latent-readable and the promoter can re-touch them);
/// 6. emit one `LatentNeighbor` per surviving association; sort by cosine
///    (same-project, then id/chunk as deterministic tiebreaks) and take top-`k`.
///
/// # Errors
/// Propagates corpus (Lance) and ledger read errors.
// `exclude` is always a default-hasher set built here in the crate; no need to
// generalize the public signature over `BuildHasher`.
#[allow(clippy::implicit_hasher)]
pub async fn latent_neighbors(
    store: &CorpusStore,
    reader: &dyn ConceptActivationReader,
    anchor: &SeedAnchor,
    exclude: &HashSet<i64>,
    k: usize,
) -> Result<Vec<LatentNeighbor>> {
    // 1. anchor embedding (skip the seed if it has no vector in Lance).
    let embs = store
        .fetch_embeddings(std::slice::from_ref(&anchor.anchor_chunk_id))
        .await?;
    let Some(anchor_vec) = embs.get(&anchor.anchor_chunk_id).cloned() else {
        return Ok(Vec::new());
    };

    // 2. nearest chunks (k + slack for the anchor + same-document chunks).
    let table = store
        .connection()
        .open_table(CORPUS_TABLE)
        .execute()
        .await?;
    let near = lane_dense(&table, &anchor_vec, None, k + LATENT_SLACK).await?;
    let near_ids: Vec<String> = near
        .into_iter()
        .map(|e| e.0)
        .filter(|id| *id != anchor.anchor_chunk_id)
        .collect();
    if near_ids.is_empty() {
        return Ok(Vec::new());
    }

    // 3. fetch candidate chunks + embeddings; cosine vs the anchor; drop below floor.
    let fetched = store.fetch_chunks_by_ids(&near_ids).await?;
    let mut kept: Vec<(String, String, String, f32)> = Vec::new();
    for (chunk_id, (chunk, emb)) in &fetched {
        let Some(emb) = emb.as_ref() else { continue };
        let cos = cosine_similarity(&anchor_vec, emb);
        if cos < REL_LATENT_SIM_FLOOR {
            continue;
        }
        kept.push((
            chunk_id.clone(),
            chunk.source.as_str().to_string(),
            chunk.source_id.clone(),
            cos,
        ));
    }
    if kept.is_empty() {
        return Ok(Vec::new());
    }

    // 4. reverse-map to (chunk, concept) associations.
    let kept_ids: Vec<String> = kept.iter().map(|(id, ..)| id.clone()).collect();
    let by_chunk = reader.concepts_for_chunks(&kept_ids)?;

    // 5 + 6. drop the seed itself and any concept in `exclude` (the caller's
    // hard off-diagonal set); emit one neighbour per surviving association.
    let mut out: Vec<LatentNeighbor> = Vec::new();
    for (chunk_id, source, source_id, cos) in kept {
        let Some(concepts) = by_chunk.get(&chunk_id) else {
            continue;
        };
        for (cid, project, handle) in concepts {
            if *cid == anchor.concept_id || exclude.contains(cid) {
                continue;
            }
            out.push(LatentNeighbor {
                concept_id: *cid,
                project: project.clone(),
                handle: handle.clone(),
                chunk_id: chunk_id.clone(),
                source: source.clone(),
                source_id: source_id.clone(),
                cosine: cos,
            });
        }
    }
    out.sort_by(|a, b| {
        b.cosine
            .partial_cmp(&a.cosine)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Deterministic tiebreaks on equal cosine: same-project as the seed
            // first, then concept id, then chunk id (map/SQL order is unstable).
            .then_with(|| {
                let a_same = a.project == anchor.project;
                let b_same = b.project == anchor.project;
                b_same.cmp(&a_same)
            })
            .then_with(|| a.concept_id.cmp(&b.concept_id))
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    out.truncate(k);
    Ok(out)
}

/// Part A: fold each seed's latent neighbours into the reified-diffusion
/// [`RelationalSupport`].
///
/// Off-diagonal chunks then **surface** through the same lens slot + injection
/// lane slice 2 built. Additive and read-only: the reified payload is unchanged
/// where the two overlap (max-merge per coordinate).
///
/// # Errors
/// Propagates corpus (Lance) and ledger read errors.
pub async fn augment_relational_support_latent(
    store: &CorpusStore,
    reader: &dyn ConceptActivationReader,
    support: &mut RelationalSupport,
) -> Result<()> {
    if support.seed_anchors.is_empty() {
        return Ok(());
    }
    let anchors = support.seed_anchors.clone();
    let mut seen: HashSet<String> = support.inject_chunk_ids.iter().cloned().collect();
    for anchor in &anchors {
        // Exclude only HARD (authored/observed) neighbours — those are owned by
        // the reified path. A `promoted` neighbour stays latent-readable so a
        // freshly-promoted bridge (conf 0.1, too weak for reified diffusion to
        // lift for an ordinary seed) doesn't vanish from the lens the moment it
        // is promoted; it keeps surfacing via the latent hop until its reified
        // conductance clears the floor.
        for n in latent_neighbors(
            store,
            reader,
            anchor,
            &anchor.hard_neighbor_ids,
            REL_LATENT_K,
        )
        .await?
        {
            let lift = anchor.weight * n.cosine * REL_HOP_DECAY;
            support
                .coord_activation
                .entry((n.source, n.source_id))
                .and_modify(|cur| {
                    if lift > *cur {
                        *cur = lift;
                    }
                })
                .or_insert(lift);
            if seen.insert(n.chunk_id.clone()) {
                support.inject_chunk_ids.push(n.chunk_id);
            }
        }
    }
    Ok(())
}

/// Summary of one latent→reified promotion pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromotionReport {
    /// Seeds with a usable anchor that were examined.
    pub examined_seeds: usize,
    /// Newly-created promoted edges.
    pub promoted: usize,
    /// Already-existing edges re-touched (kept warm, not re-created).
    pub retouched: usize,
}

/// Part B: promote each active seed's top off-diagonal latent neighbours into
/// weak [`EdgeSource::Promoted`] edges.
///
/// A weak prior that conductance/decay then adjudicates. The promoter excludes
/// only *hard* (authored/observed) neighbours, so a `promoted` edge passes the
/// filter and is **re-touched** (kept warm) on a later pass — that is what keeps
/// a still-relevant bridge alive. A fresh [`ThreadsDb::edges_between`] recheck
/// skips a pair an operator already linked and re-touches an existing promoted
/// edge in its own direction; an in-pass pair set settles reciprocal active
/// seeds once. The audit event fires only on creation, stamped under the seed's
/// scope (the by-id row is authoritative — matches slice-4 cross-scope edges).
///
/// # Errors
/// Propagates corpus (Lance) and ledger read/write errors.
pub async fn promote_latent_edges(
    store: &CorpusStore,
    threads: &ThreadsDb,
    support: &RelationalSupport,
) -> Result<PromotionReport> {
    let mut report = PromotionReport::default();
    // Unordered pairs settled this pass — so two reciprocal active seeds
    // (A near B and B near A) settle their shared edge once, not twice.
    let mut handled: HashSet<(i64, i64)> = HashSet::new();
    for anchor in &support.seed_anchors {
        report.examined_seeds += 1;
        // Promoter excludes only *hard* (authored/observed) neighbours — a
        // `promoted` neighbour passes through so it can be re-touched (warmed).
        let neighbors = latent_neighbors(
            store,
            threads,
            anchor,
            &anchor.hard_neighbor_ids,
            REL_LATENT_K,
        )
        .await?;
        let mut per_seed: HashSet<i64> = HashSet::new();
        for n in neighbors {
            if per_seed.len() >= REL_PROMOTE_TOP_K {
                break;
            }
            // One edge per neighbour concept (reached via several chunks → once).
            if !per_seed.insert(n.concept_id) {
                continue;
            }
            let pair = (
                anchor.concept_id.min(n.concept_id),
                anchor.concept_id.max(n.concept_id),
            );
            if !handled.insert(pair) {
                continue;
            }
            // Fresh recheck (both directions): an operator/observed link is never
            // re-promoted; an existing `promoted` edge is re-touched in its own
            // direction rather than written reciprocally.
            let existing = threads.edges_between(anchor.concept_id, n.concept_id)?;
            if existing
                .iter()
                .any(|(_, _, _, s)| matches!(s, EdgeSource::Authored | EdgeSource::Observed))
            {
                continue;
            }
            let (from_id, to_id) = existing
                .iter()
                .find(|(_, _, rel, s)| *s == EdgeSource::Promoted && rel == REL_PROMOTED_RELATION)
                .map_or((anchor.concept_id, n.concept_id), |(f, t, _, _)| (*f, *t));
            let gloss = serde_json::json!({
                "via": "latent-promotion",
                "cosine": n.cosine,
                "anchor_chunk": anchor.anchor_chunk_id,
                "near_chunk": n.chunk_id,
            })
            .to_string();
            let (_edge_id, created) = threads.add_concept_edge_by_id(
                from_id,
                REL_PROMOTED_RELATION,
                to_id,
                PROMOTED_EDGE_CONFIDENCE,
                EdgeSource::Promoted,
                Some("diffusion"),
                Some(&gloss),
            )?;
            if created {
                report.promoted += 1;
                threads.record_concept_connected(
                    &anchor.project,
                    &anchor.handle,
                    REL_PROMOTED_RELATION,
                    &n.handle,
                    EdgeSource::Promoted,
                    Some("diffusion"),
                )?;
            } else {
                report.retouched += 1;
            }
        }
    }
    Ok(report)
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
            seed_anchors: Vec::new(),
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
            seed_anchors: Vec::new(),
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
