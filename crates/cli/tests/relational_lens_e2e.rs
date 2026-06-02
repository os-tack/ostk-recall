//! Slice 2 — diffusion read, daemon-path integration: `try_refresh_lens`
//! against a real Lance corpus + a real `ThreadsDb` concept graph.
//!
//! The marquee thesis: focusing `tori` lights its neighbour `ostk-recall` and
//! **surfaces `ostk-recall`'s chunk in the lens** — even though that chunk is
//! NOT in the dense candidate pool (it's orthogonal to the attention vector,
//! and `candidate_k_per_lane = 1` returns only the aligned `tori` chunk). The
//! relational candidate lane injects it and the `relational_lift` feature
//! scores it into the `relational` slot. The companion test pins the gate: with
//! `relational_lift` weight 0, the lane is off and the neighbour stays hidden.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::Utc;
use tempfile::TempDir;

use ostk_recall_cli::lens_loop::{LensRefreshDecision, LensTickSnapshot, try_refresh_lens};
use ostk_recall_cli::lens_state::LensState;
use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_query::lens::{Lens, LensConfig};
use ostk_recall_query::rank::{RankEngine, build_engine_from_weights};
use ostk_recall_store::corpus::CorpusStore;
use ostk_recall_store::threads::ThreadsDb;
use ostk_recall_store::{
    ConceptActivationReader, ConceptStatus, EdgeSource, EvidenceAttach, GLOBAL_PROJECT,
    SqliteChainSink,
};

const DIM: usize = 8;
const G: &str = GLOBAL_PROJECT;

fn chunk(id: &str, text: &str) -> Chunk {
    Chunk {
        chunk_id: id.into(),
        source: Source::Markdown,
        project: Some("memories".into()),
        source_id: format!("{id}.md"),
        source_config_id: "test:cfg".into(),
        chunk_index: 0,
        ts: Some(Utc::now()),
        role: None,
        text: text.into(),
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

/// Corpus with `tori` aligned to the attention axis and `ostk-recall`
/// orthogonal; a concept graph where focused-seed `tori` --works_on-->
/// `ostk-recall` (only PROPOSED — diffusion reaches non-active neighbours),
/// whose evidence resolves to its corpus chunk. Returns the loop's inputs.
async fn setup() -> (
    TempDir,
    CorpusStore,
    Arc<dyn ConceptActivationReader>,
    Vec<f32>,
) {
    let tmp = TempDir::new().unwrap();
    let corpus = CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap();
    let mut scope = vec![0.0_f32; DIM];
    scope[0] = 1.0;
    let mut orthogonal = vec![0.0_f32; DIM];
    orthogonal[1] = 1.0;
    corpus
        .upsert(
            &[
                chunk("tori", "tori works on the memory layer"),
                chunk("ostk-recall", "the local memory substrate"),
            ],
            &[scope.clone(), orthogonal],
        )
        .await
        .unwrap();

    let sink = SqliteChainSink::open(tmp.path()).unwrap();
    let db = ThreadsDb::open_with_sink(tmp.path(), Arc::new(sink)).unwrap();
    db.ensure_concept(G, "tori", ConceptStatus::Active).unwrap();
    db.ensure_concept(G, "ostk-recall", ConceptStatus::Proposed)
        .unwrap();
    db.record_concept_accessed(G, "tori", "recall:path", "qh-1")
        .unwrap();
    let (tori, orec) = (id_of(&db, "tori"), id_of(&db, "ostk-recall"));
    db.add_concept_edge_by_id(
        tori,
        "works_on",
        orec,
        1.0,
        EdgeSource::Authored,
        Some("scanner"),
        None,
    )
    .unwrap();
    db.attach_concept_evidence(&EvidenceAttach {
        concept_id: orec,
        project: "memories",
        source: "markdown",
        source_id: "ostk-recall.md",
        chunk_id: Some("ostk-recall"),
        content_sha256: None,
        anchor_vec: None,
        score: Some(0.5),
        reason: Some("test"),
    })
    .unwrap();
    let reader: Arc<dyn ConceptActivationReader> = Arc::new(db);
    (tmp, corpus, reader, scope)
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
    scope: &[f32],
) -> Lens {
    // k_per_lane = 1 → the dense lane returns only the aligned `tori` chunk;
    // `ostk-recall` can reach the lens ONLY via the relational injection lane.
    let config = LensConfig {
        candidate_k_per_lane: 1,
        ..LensConfig::default()
    };
    let decision = try_refresh_lens(
        &snapshot(scope),
        &LensState::default(),
        engine,
        None,         // chain_reader (refractory/freshness) — not needed
        Some(reader), // concept_reader — drives diffusion
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
async fn focus_surfaces_diffused_neighbour_chunk_via_relational_slot() {
    let (_tmp, corpus, reader, scope) = setup().await;
    // Focused engine: attention_affinity + relational_lift, so the relational
    // slot's claim is unambiguous (no freshness/concept competing).
    let mut weights = BTreeMap::new();
    weights.insert("attention_affinity".to_string(), 1.0);
    weights.insert("relational_lift".to_string(), 0.5);
    let engine = build_engine_from_weights(&weights);

    let lens = refresh(&engine, reader, &corpus, &scope).await;

    let neighbour = lens
        .entries
        .iter()
        .find(|e| e.chunk_id == "ostk-recall")
        .expect("diffused neighbour chunk surfaced via the relational lane");
    assert_eq!(
        neighbour.slot_name, "relational",
        "the neighbour was placed in the relational slot"
    );
    assert!(
        neighbour
            .feature_breakdown
            .get("relational_lift")
            .is_some_and(|a| a.contribution > 0.0),
        "neighbour carries a positive relational_lift contribution"
    );
    assert!(
        lens.entries.iter().any(|e| e.chunk_id == "tori"),
        "the focused seed chunk is also present (via attention)"
    );
}

#[tokio::test]
async fn relational_lane_off_when_feature_disabled() {
    let (_tmp, corpus, reader, scope) = setup().await;
    // No relational_lift weight → the injection lane is gated off, so the
    // orthogonal neighbour never enters the candidate pool.
    let mut weights = BTreeMap::new();
    weights.insert("attention_affinity".to_string(), 1.0);
    let engine = build_engine_from_weights(&weights);

    let lens = refresh(&engine, reader, &corpus, &scope).await;

    assert!(
        lens.entries.iter().all(|e| e.chunk_id != "ostk-recall"),
        "with relational_lift disabled, the neighbour is not injected"
    );
    assert!(
        lens.entries.iter().any(|e| e.chunk_id == "tori"),
        "the attention-aligned chunk still surfaces"
    );
}
