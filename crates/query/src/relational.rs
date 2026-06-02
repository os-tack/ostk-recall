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
    ConceptActivationReader, CorpusStore, EdgeSource, PROMOTED_EDGE_CONFIDENCE, REL_HOP_DECAY,
    REL_LATENT_K, REL_PROMOTED_RELATION, RelationalSupport, SeedAnchor, ThreadsDb,
};

use crate::candidate::Candidate;
use crate::context::{AttentionContext, QueryContext};
use crate::error::Result;
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

/// One concept's codebook entry (relational-substrate slice 2cA): its anchor
/// embedding + identity/scope + the coordinate Part A surfaces / injects.
#[derive(Debug, Clone)]
pub struct CodebookEntry {
    pub project: String,
    pub handle: String,
    pub source: String,
    pub source_id: String,
    pub chunk_id: String,
    pub sem: Vec<f32>,
}

/// The concept codebook: every non-terminal concept's anchor embedding, held in
/// its own small space. Concept↔concept latent adjacency is **exact cosine over
/// this map** — no ANN over the bulk corpus, so concept neighbours are never
/// swamped by the ~0.001%-dense transcript chunks (the slice-2b production
/// failure surfaced by the live bench).
#[derive(Debug, Clone, Default)]
pub struct ConceptCodebook {
    pub by_id: HashMap<i64, CodebookEntry>,
}

impl ConceptCodebook {
    /// Build from the reader's `concept_anchors` + the corpus embeddings of their
    /// anchor chunks. Entries whose anchor chunk has no vector in Lance are
    /// dropped. Bounded by concept count, not corpus size.
    ///
    /// # Errors
    /// Propagates ledger + corpus (Lance) read errors.
    pub async fn build(store: &CorpusStore, reader: &dyn ConceptActivationReader) -> Result<Self> {
        let anchors = reader.concept_anchors()?;
        if anchors.is_empty() {
            return Ok(Self::default());
        }
        let ids: Vec<String> = anchors.iter().map(|a| a.chunk_id.clone()).collect();
        let embs = store.fetch_embeddings(&ids).await?;
        let mut by_id = HashMap::with_capacity(anchors.len());
        for a in anchors {
            let Some(sem) = embs.get(&a.chunk_id).cloned() else {
                continue;
            };
            by_id.insert(
                a.concept_id,
                CodebookEntry {
                    project: a.project,
                    handle: a.handle,
                    source: a.source,
                    source_id: a.source_id,
                    chunk_id: a.chunk_id,
                    sem,
                },
            );
        }
        Ok(Self { by_id })
    }
}

/// One seed's latent hop over the concept codebook (relational-substrate slice
/// 2cA): exact cosine between the seed's anchor embedding and every *other*
/// concept's anchor — no ANN over the bulk corpus.
///
/// `exclude` is the caller's off-diagonal filter — the seed's authored/observed
/// neighbours; `promoted` neighbours are intentionally *not* excluded so they
/// stay latent-readable and the promoter can re-touch them. Drops self +
/// excluded + below `floor`; sorts by cosine with deterministic tiebreaks
/// (same-project, then concept id, then chunk id); takes top-`k`. Empty if the
/// seed has no codebook entry (no resolvable anchor embedding).
// `exclude` is always a default-hasher set built here in the crate; no need to
// generalize the public signature over `BuildHasher`.
#[allow(clippy::implicit_hasher)]
#[must_use]
pub fn latent_neighbors(
    codebook: &ConceptCodebook,
    anchor: &SeedAnchor,
    exclude: &HashSet<i64>,
    floor: f32,
    k: usize,
) -> Vec<LatentNeighbor> {
    let Some(seed) = codebook.by_id.get(&anchor.concept_id) else {
        return Vec::new();
    };
    let mut out: Vec<LatentNeighbor> = Vec::new();
    for (cid, entry) in &codebook.by_id {
        // Skip self, hard (off-diagonal) neighbours, and any concept anchored to
        // the SAME chunk as the seed — co-citation cosines at 1.0 but is not a
        // latent *bridge* (it's the same evidence, not a discovered link).
        if *cid == anchor.concept_id
            || exclude.contains(cid)
            || entry.chunk_id == anchor.anchor_chunk_id
        {
            continue;
        }
        let cosine = cosine_similarity(&seed.sem, &entry.sem);
        if cosine < floor {
            continue;
        }
        out.push(LatentNeighbor {
            concept_id: *cid,
            project: entry.project.clone(),
            handle: entry.handle.clone(),
            chunk_id: entry.chunk_id.clone(),
            source: entry.source.clone(),
            source_id: entry.source_id.clone(),
            cosine,
        });
    }
    out.sort_by(|a, b| {
        b.cosine
            .partial_cmp(&a.cosine)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Deterministic tiebreaks on equal cosine: same-project as the seed
            // first, then concept id, then chunk id (map order is unstable).
            .then_with(|| {
                let a_same = a.project == anchor.project;
                let b_same = b.project == anchor.project;
                b_same.cmp(&a_same)
            })
            .then_with(|| a.concept_id.cmp(&b.concept_id))
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
    out.truncate(k);
    out
}

/// Part A: fold each seed's latent neighbours into the reified-diffusion
/// [`RelationalSupport`] so off-diagonal chunks **surface** through the same lens
/// slot + injection lane slice 2 built. Additive, read-only, and sync (the
/// `codebook` is prebuilt once per pass).
pub fn augment_relational_support_latent(
    codebook: &ConceptCodebook,
    support: &mut RelationalSupport,
    floor: f32,
) {
    if support.seed_anchors.is_empty() {
        return;
    }
    let anchors = support.seed_anchors.clone();
    let mut seen: HashSet<String> = support.inject_chunk_ids.iter().cloned().collect();
    for anchor in &anchors {
        // Exclude only HARD (authored/observed) neighbours — those are owned by
        // the reified path. A `promoted` neighbour stays latent-readable so a
        // freshly-promoted bridge (conf 0.1, too weak for reified diffusion to
        // lift for an ordinary seed) doesn't vanish from the lens; it keeps
        // surfacing via the latent hop until its reified conductance clears.
        for n in latent_neighbors(
            codebook,
            anchor,
            &anchor.hard_neighbor_ids,
            floor,
            REL_LATENT_K,
        ) {
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
/// Propagates ledger read/write errors.
pub fn promote_latent_edges(
    codebook: &ConceptCodebook,
    threads: &ThreadsDb,
    support: &RelationalSupport,
    floor: f32,
    top_k: usize,
) -> Result<PromotionReport> {
    let mut report = PromotionReport::default();
    // Unordered pairs settled this pass — so two reciprocal active seeds
    // (A near B and B near A) settle their shared edge once, not twice.
    let mut handled: HashSet<(i64, i64)> = HashSet::new();
    for anchor in &support.seed_anchors {
        report.examined_seeds += 1;
        // Excludes only *hard* (authored/observed) neighbours — a `promoted`
        // neighbour passes through so it can be re-touched (warmed). `top_k` is
        // the cap: the codebook holds one anchor per concept, so neighbours are
        // already distinct concepts (no per-seed dedup needed).
        let neighbors = latent_neighbors(codebook, anchor, &anchor.hard_neighbor_ids, floor, top_k);
        for n in neighbors {
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
