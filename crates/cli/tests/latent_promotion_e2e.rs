//! Slice 2b — latent (Lance-similarity) union + latent→reified promotion,
//! daemon-path integration against a real Lance corpus + a real `ThreadsDb`.
//!
//! The thesis: `alpha` (focused) and `beta` (proposed) have **no reified edge**,
//! but `beta`'s evidence chunk is ANN-near `alpha`'s anchor chunk — an
//! off-diagonal bridge. Part A: focusing `alpha` surfaces `beta`'s chunk in the
//! `relational` slot via the latent hop, though no edge exists. Part B: a
//! `memory_reflect`-style consolidation (`promote_latent_edges`) writes a weak
//! `Promoted` `related` edge `alpha → beta`; it is idempotent (re-touch, not
//! duplicate), off-diagonal-only (a pair with an existing edge is skipped), and
//! gated by the cosine floor (a far neighbour is never promoted).

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::Utc;
use tempfile::TempDir;

use ostk_recall_cli::lens_loop::{LensRefreshDecision, LensTickSnapshot, try_refresh_lens};
use ostk_recall_cli::lens_state::LensState;
use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_query::lens::Lens;
use ostk_recall_query::lens::LensConfig;
use ostk_recall_query::rank::{RankEngine, build_engine_from_weights};
use ostk_recall_query::{ConceptCodebook, PromotionReport, promote_latent_edges};
use ostk_recall_store::corpus::CorpusStore;
use ostk_recall_store::threads::ThreadsDb;
use ostk_recall_store::{
    ConceptActivationReader, ConceptStatus, EdgeDirection, EdgeSource, EvidenceAttach,
    GLOBAL_PROJECT, SqliteChainSink, default_since,
};

const DIM: usize = 8;
const G: &str = GLOBAL_PROJECT;
/// Test floor: admits the cosine-1.0 same-axis bridge, rejects the orthogonal
/// (cosine-0) far neighbour.
const TEST_FLOOR: f32 = 0.5;

/// Build the codebook + support fresh and run one promotion pass — the real
/// `memory_reflect` shape (support + codebook recomputed each pass).
async fn promote(corpus: &CorpusStore, db: &ThreadsDb) -> PromotionReport {
    let support = db.relational_support(default_since(Utc::now())).unwrap();
    let codebook = ConceptCodebook::build(corpus, db).await.unwrap();
    promote_latent_edges(&codebook, db, &support, TEST_FLOOR, 4).unwrap()
}

/// A one-hot unit vector on `axis`.
fn axis(a: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[a] = 1.0;
    v
}

fn chunk(id: &str) -> Chunk {
    Chunk {
        chunk_id: id.into(),
        source: Source::Markdown,
        project: Some("memories".into()),
        source_id: format!("{id}.md"),
        source_config_id: "test:cfg".into(),
        chunk_index: 0,
        ts: Some(Utc::now()),
        role: None,
        text: format!("text {id}"),
        sha256: format!("sha-{id}"),
        links: Links::default(),
        facets: FacetSet::default(),
        embedding_input_sha256: format!("emb-{id}"),
        extra: serde_json::Value::Null,
    }
}

fn id_of(db: &ThreadsDb, handle: &str) -> i64 {
    db.get_concept(G, handle).unwrap().unwrap().id
}

fn evidence(db: &ThreadsDb, id: i64, chunk_id: &str) {
    db.attach_concept_evidence(&EvidenceAttach {
        concept_id: id,
        project: "memories",
        source: "markdown",
        source_id: &format!("{chunk_id}.md"),
        chunk_id: Some(chunk_id),
        content_sha256: None,
        anchor_vec: None,
        score: Some(0.5),
        reason: Some("test"),
    })
    .unwrap();
}

/// Corpus + concept graph with an off-diagonal latent bridge.
///
/// - `seed-aligned` (axis 0): a plain dense candidate aligned to the attention
///   vector, so the lens always has a non-injected candidate.
/// - `alpha-anchor` (axis 1): focused-seed `alpha`'s evidence chunk — the latent
///   anchor; orthogonal to attention, so it is NOT dense-returned.
/// - `beta-chunk` (`beta_axis`): proposed `beta`'s evidence chunk. At axis 1 it
///   is cosine 1.0 with the anchor (a bridge); at axis 2 it is cosine 0 (far).
///
/// No reified edge connects `alpha` and `beta`.
async fn build(beta_axis: usize) -> (TempDir, CorpusStore, ThreadsDb) {
    let tmp = TempDir::new().unwrap();
    let corpus = CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap();
    corpus
        .upsert(
            &[
                chunk("seed-aligned"),
                chunk("alpha-anchor"),
                chunk("beta-chunk"),
            ],
            &[axis(0), axis(1), axis(beta_axis)],
        )
        .await
        .unwrap();

    let sink = SqliteChainSink::open(tmp.path()).unwrap();
    let db = ThreadsDb::open_with_sink(tmp.path(), Arc::new(sink)).unwrap();
    db.ensure_concept(G, "alpha", ConceptStatus::Active)
        .unwrap();
    db.ensure_concept(G, "beta", ConceptStatus::Proposed)
        .unwrap();
    // A recent access gives `alpha` dynamic (attention) activation → a seed.
    db.record_concept_accessed(G, "alpha", "recall:path", "qh-1")
        .unwrap();
    evidence(&db, id_of(&db, "alpha"), "alpha-anchor");
    evidence(&db, id_of(&db, "beta"), "beta-chunk");
    (tmp, corpus, db)
}

fn snapshot(scope: &[f32]) -> LensTickSnapshot {
    LensTickSnapshot {
        rolling_vec: Some(scope.to_vec()),
        scope_vector: Some(scope.to_vec()),
        pin_fingerprint: None,
    }
}

async fn refresh(
    engine: &RankEngine,
    reader: Arc<dyn ConceptActivationReader>,
    corpus: &CorpusStore,
) -> Lens {
    let scope = axis(0);
    let config = LensConfig {
        candidate_k_per_lane: 1,
        ..LensConfig::default()
    };
    let decision = try_refresh_lens(
        &snapshot(&scope),
        &LensState::default(),
        engine,
        None,
        Some(reader),
        corpus,
        &config,
        false,
    )
    .await;
    match decision {
        LensRefreshDecision::Refresh { lens, .. } => lens,
        LensRefreshDecision::BuildFailed(e) => panic!("expected Refresh, got BuildFailed: {e}"),
        LensRefreshDecision::EmptyMind => panic!("expected Refresh, got EmptyMind"),
        LensRefreshDecision::NoTrigger => panic!("expected Refresh, got NoTrigger"),
        LensRefreshDecision::UnchangedContent { .. } => {
            panic!("expected Refresh, got UnchangedContent")
        }
    }
}

#[tokio::test]
async fn focus_surfaces_off_diagonal_latent_chunk_via_relational_slot() {
    let (_tmp, corpus, db) = build(1).await; // beta near alpha's anchor
    let reader: Arc<dyn ConceptActivationReader> = Arc::new(db);

    let mut weights = BTreeMap::new();
    weights.insert("attention_affinity".to_string(), 1.0);
    weights.insert("relational_lift".to_string(), 0.5);
    let engine = build_engine_from_weights(&weights);

    let lens = refresh(&engine, reader, &corpus).await;

    let neighbour = lens
        .entries
        .iter()
        .find(|e| e.chunk_id == "beta-chunk")
        .expect("off-diagonal latent neighbour surfaced (no reified edge exists)");
    assert_eq!(
        neighbour.slot_name, "relational",
        "the latent neighbour claimed the relational slot"
    );
    assert!(
        neighbour
            .feature_breakdown
            .get("relational_lift")
            .is_some_and(|a| a.contribution > 0.0),
        "latent neighbour carries a positive relational_lift contribution"
    );
}

#[tokio::test]
async fn reflect_promotes_off_diagonal_bridge_to_weak_promoted_edge() {
    let (_tmp, corpus, db) = build(1).await;
    let report = promote(&corpus, &db).await;
    assert_eq!(report.promoted, 1, "one off-diagonal bridge promoted");
    assert_eq!(report.retouched, 0);

    let edges = db
        .list_concept_edges(G, "alpha", EdgeDirection::From)
        .unwrap();
    let edge = edges
        .iter()
        .find(|e| e.to_handle == "beta")
        .expect("alpha --related--> beta written");
    assert_eq!(edge.relation, "related");
    assert_eq!(edge.source, EdgeSource::Promoted, "EdgeSource::Promoted");
    assert!(
        (edge.confidence - 0.1).abs() < 1e-6,
        "weak prior 0.1, got {}",
        edge.confidence
    );
    assert_eq!(edge.by.as_deref(), Some("diffusion"));

    // Idempotent in the REAL daemon path: recompute support so it sees the
    // freshly-promoted edge as a reified neighbour (the promoted edge is NOT in
    // the seed's hard set, so the bridge passes the filter and is re-touched,
    // keeping conductance warm) — never duplicated.
    // Recompute support + codebook (the real path) and promote again.
    let again = promote(&corpus, &db).await;
    assert_eq!(again.promoted, 0, "no new edge on re-run");
    assert_eq!(
        again.retouched, 1,
        "existing promoted edge re-touched (kept warm)"
    );
    let edges2 = db
        .list_concept_edges(G, "alpha", EdgeDirection::From)
        .unwrap();
    assert_eq!(
        edges2.iter().filter(|e| e.to_handle == "beta").count(),
        1,
        "still exactly one edge"
    );
}

#[tokio::test]
async fn pair_with_existing_edge_is_not_re_promoted() {
    let (_tmp, corpus, db) = build(1).await;
    // Pre-existing authored edge → beta is now a reified neighbour, not off-diagonal.
    let (a, b) = (id_of(&db, "alpha"), id_of(&db, "beta"));
    db.add_concept_edge_by_id(a, "works_on", b, 1.0, EdgeSource::Authored, None, None)
        .unwrap();

    let report = promote(&corpus, &db).await;
    assert_eq!(
        report.promoted, 0,
        "off-diagonal only — existing pair skipped"
    );

    let promoted = db
        .list_concept_edges(G, "alpha", EdgeDirection::From)
        .unwrap()
        .into_iter()
        .filter(|e| e.source == EdgeSource::Promoted)
        .count();
    assert_eq!(promoted, 0, "no promoted edge written for a reified pair");
}

#[tokio::test]
async fn far_neighbour_below_floor_is_not_promoted() {
    let (_tmp, corpus, db) = build(2).await; // beta orthogonal to the anchor (cosine 0)
    let report = promote(&corpus, &db).await;
    assert_eq!(
        report.promoted, 0,
        "a sub-floor latent neighbour is not a bridge"
    );
}

#[tokio::test]
async fn same_chunk_co_citation_is_not_a_bridge() {
    // active `alpha` and proposed `beta` cite the SAME evidence chunk — they
    // cosine at 1.0, but co-citation of one chunk is not a discovered latent
    // bridge, so nothing is promoted.
    let tmp = TempDir::new().unwrap();
    let corpus = CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap();
    corpus.upsert(&[chunk("shared")], &[axis(1)]).await.unwrap();
    let sink = SqliteChainSink::open(tmp.path()).unwrap();
    let db = ThreadsDb::open_with_sink(tmp.path(), Arc::new(sink)).unwrap();
    db.ensure_concept(G, "alpha", ConceptStatus::Active)
        .unwrap();
    db.ensure_concept(G, "beta", ConceptStatus::Proposed)
        .unwrap();
    db.record_concept_accessed(G, "alpha", "recall:path", "qh-a")
        .unwrap();
    evidence(&db, id_of(&db, "alpha"), "shared");
    evidence(&db, id_of(&db, "beta"), "shared");

    let report = promote(&corpus, &db).await;
    assert_eq!(
        report.promoted, 0,
        "co-citation of one chunk is not a latent bridge"
    );
}

#[tokio::test]
async fn reciprocal_active_seeds_promote_a_single_edge() {
    // Both `alpha` and `beta` are active seeds whose anchors sit on the same
    // axis — each is the other's latent neighbour. A single pass must settle one
    // shared edge, not a reciprocal A→B and B→A pair.
    let tmp = TempDir::new().unwrap();
    let corpus = CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap();
    corpus
        .upsert(
            &[chunk("alpha-anchor"), chunk("beta-anchor")],
            &[axis(1), axis(1)],
        )
        .await
        .unwrap();
    let sink = SqliteChainSink::open(tmp.path()).unwrap();
    let db = ThreadsDb::open_with_sink(tmp.path(), Arc::new(sink)).unwrap();
    db.ensure_concept(G, "alpha", ConceptStatus::Active)
        .unwrap();
    db.ensure_concept(G, "beta", ConceptStatus::Active).unwrap();
    db.record_concept_accessed(G, "alpha", "recall:path", "qh-a")
        .unwrap();
    db.record_concept_accessed(G, "beta", "recall:path", "qh-b")
        .unwrap();
    evidence(&db, id_of(&db, "alpha"), "alpha-anchor");
    evidence(&db, id_of(&db, "beta"), "beta-anchor");

    let report = promote(&corpus, &db).await;
    assert_eq!(
        report.promoted, 1,
        "exactly one shared edge, not two reciprocal"
    );

    let total: usize = ["alpha", "beta"]
        .iter()
        .map(|h| {
            db.list_concept_edges(G, h, EdgeDirection::From)
                .unwrap()
                .into_iter()
                .filter(|e| e.source == EdgeSource::Promoted)
                .count()
        })
        .sum();
    assert_eq!(
        total, 1,
        "one promoted edge total between the reciprocal pair"
    );
}

#[tokio::test]
async fn promoted_bridge_still_surfaces_after_promotion() {
    // Continuity: a freshly-promoted edge has confidence 0.1 — too weak for
    // reified diffusion to lift for an access-only seed. So Part A must keep
    // treating the (now-promoted) neighbour as latent-readable, or the bridge
    // would surface before reflect, get promoted, then vanish from the lens.
    let (_tmp, corpus, db) = build(1).await;
    let report = promote(&corpus, &db).await;
    assert_eq!(report.promoted, 1);

    // Rebuild the lens against the post-promotion graph.
    let reader: Arc<dyn ConceptActivationReader> = Arc::new(db);
    let mut weights = BTreeMap::new();
    weights.insert("attention_affinity".to_string(), 1.0);
    weights.insert("relational_lift".to_string(), 0.5);
    let engine = build_engine_from_weights(&weights);
    let lens = refresh(&engine, reader, &corpus).await;

    let neighbour = lens
        .entries
        .iter()
        .find(|e| e.chunk_id == "beta-chunk")
        .expect("the promoted bridge still surfaces via the latent hop");
    assert_eq!(
        neighbour.slot_name, "relational",
        "promoted bridge keeps the relational slot until its conductance clears the floor"
    );
}
