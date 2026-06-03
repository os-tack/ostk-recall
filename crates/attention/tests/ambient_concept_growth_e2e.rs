//! Phase 2 — ambient salience-gated concept-growth, daemon-path integration
//! against a real Lance corpus + a real `ThreadsDb` + `InMemoryAttention`.
//!
//! The thesis: a turn grows the **concept** graph, but only along the
//! resonance gate. Two concept anchors (`alpha`, `beta`) sit on the same axis;
//! a turn whose rolling vector aligns to that axis AND that names both → a
//! `co_occurs` edge. A turn that names a concept but does NOT resonate mints
//! nothing (occurrence≠salience). An unknown kebab term recurring across
//! resonant turns mints a `Proposed` node connected to its co-resonant context;
//! the same term across non-resonant turns never mints.
//!
//! Determinism: concept anchor embeddings are set directly via `corpus.upsert`
//! (one-hot `axis()` vectors); the turn rolling vector is produced by an
//! [`AxisEmbedder`] wired into `InMemoryAttention` that maps a marker token in
//! the turn text to a chosen axis — so `cosine(anchor, rolling)` is controlled.

use std::sync::{Arc, Mutex as StdMutex};

use chrono::Utc;
use tempfile::TempDir;

use ostk_recall_attention::{
    AttentionForwardStore, ConceptGrowthConfig, ConceptGrowthRuntime, InMemoryAttention,
    TurnObserver,
};
use ostk_recall_core::attention::{AttentionScope, PrivacyTier};
use ostk_recall_core::{Chunk, FacetSet, Links, Source};
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline};
use ostk_recall_store::{
    ChainEvent, ChainSink, ConceptStatus, CorpusStore, EdgeDirection, EvidenceAttach,
    GLOBAL_PROJECT, IngestDb, StoreError, ThreadsDb,
};

/// In-memory chain sink so tests can assert on the audit events (`ConceptConnected`,
/// `ConceptPromoted`) the observer emits — there is no general chain-event reader
/// on `ThreadsDb` (its `ChainLogReader` only covers chunk-access events).
#[derive(Default)]
struct RecordingSink {
    events: StdMutex<Vec<ChainEvent>>,
}

impl RecordingSink {
    fn snapshot(&self) -> Vec<ChainEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl ChainSink for RecordingSink {
    fn append(&self, event: &ChainEvent) -> Result<(), StoreError> {
        self.events.lock().unwrap().push(event.clone());
        Ok(())
    }
}

const DIM: usize = 8;
const G: &str = GLOBAL_PROJECT;
/// The axis concept anchors `alpha`/`beta` live on; a resonant turn's rolling
/// vector lands here too.
const RESONANT_AXIS: usize = 1;
/// An orthogonal axis a non-resonant turn's rolling vector lands on.
const QUIET_AXIS: usize = 7;

/// A one-hot unit vector on `a`.
fn axis(a: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; DIM];
    v[a] = 1.0;
    v
}

/// Embedder for the attention rolling vector: a turn containing the marker
/// `RESONATE` lands on [`RESONANT_AXIS`] (aligned with the seeded anchors),
/// otherwise on the orthogonal [`QUIET_AXIS`]. Lets each turn's resonance be
/// chosen by its text, independent of the gazetteer's lexical match.
struct AxisEmbedder;

impl ChunkEmbedder for AxisEmbedder {
    fn dim(&self) -> usize {
        DIM
    }
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                if t.contains("RESONATE") {
                    axis(RESONANT_AXIS)
                } else {
                    axis(QUIET_AXIS)
                }
            })
            .collect()
    }
}

fn anchor_chunk(id: &str) -> Chunk {
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

fn attach_anchor_in(db: &ThreadsDb, project: &str, handle: &str, chunk_id: &str) {
    let id = db.get_concept(project, handle).unwrap().unwrap().id;
    db.attach_concept_evidence(&EvidenceAttach {
        concept_id: id,
        project,
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

fn attach_anchor(db: &ThreadsDb, handle: &str, chunk_id: &str) {
    attach_anchor_in(db, G, handle, chunk_id);
}

fn scope() -> AttentionScope {
    AttentionScope {
        project: None, // None → gazetteer lists all projects (incl. globals)
        session_id: Some("ambient".into()),
        agent: Some("substrate".into()),
        privacy_tier: PrivacyTier::T1Project,
    }
}

fn test_cfg() -> ConceptGrowthConfig {
    ConceptGrowthConfig {
        resonance_floor: 0.5, // admits cosine 1.0, rejects orthogonal 0.0
        edge_top_k: 4,
        min_survivors: 2,
        node_mint_min_resonant_turns: 3,
        codebook_rebuild_turns: 32,
        node_mint_cap_per_session: 8,
    }
}

/// Full harness: a corpus carrying `alpha`/`beta` anchor chunks on
/// [`RESONANT_AXIS`], the ledger with both as `Active` global concepts anchored
/// to those chunks, and an observer wired with corpus + attention + the chain
/// sink. Returns everything the assertions need.
async fn harness(
    with_corpus: bool,
) -> (
    TempDir,
    Arc<CorpusStore>,
    Arc<ThreadsDb>,
    Arc<RecordingSink>,
    TurnObserver,
) {
    let tmp = TempDir::new().unwrap();
    let corpus = Arc::new(CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap());
    corpus
        .upsert(
            &[anchor_chunk("alpha-anchor"), anchor_chunk("beta-anchor")],
            &[axis(RESONANT_AXIS), axis(RESONANT_AXIS)],
        )
        .await
        .unwrap();

    let recorder: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let sink: Arc<dyn ChainSink> = recorder.clone();
    let db = Arc::new(ThreadsDb::open_with_sink(tmp.path(), sink).unwrap());
    db.ensure_concept(G, "alpha", ConceptStatus::Active)
        .unwrap();
    db.ensure_concept(G, "beta", ConceptStatus::Active).unwrap();
    attach_anchor(&db, "alpha", "alpha-anchor");
    attach_anchor(&db, "beta", "beta-anchor");

    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(AxisEmbedder);
    let pipeline = Arc::new(Pipeline::new(Arc::clone(&corpus), ingest, emb));

    let attn: Arc<dyn AttentionForwardStore> =
        Arc::new(InMemoryAttention::with_embedder(Arc::new(AxisEmbedder)));
    let mut observer = TurnObserver::new(pipeline, Arc::clone(&db))
        .with_attention(attn)
        .with_chain_sink(db.chain_sink())
        .with_concept_growth(test_cfg());
    if with_corpus {
        observer = observer.with_corpus(Arc::clone(&corpus));
    }
    (tmp, corpus, db, recorder, observer)
}

/// Count distinct `co_occurs` edges incident on a concept handle (either
/// direction — edges are stored in canonical `min→max` id order).
fn co_occurs_count(db: &ThreadsDb, handle: &str) -> usize {
    let mut ids = std::collections::BTreeSet::new();
    for dir in [EdgeDirection::From, EdgeDirection::To] {
        for e in db.list_concept_edges(G, handle, dir).unwrap() {
            if e.relation == "co_occurs" {
                ids.insert(e.id);
            }
        }
    }
    ids.len()
}

fn connected_audit_count(recorder: &RecordingSink) -> usize {
    recorder
        .snapshot()
        .into_iter()
        .filter(|e| matches!(e, ChainEvent::ConceptConnected { .. }))
        .count()
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn resonant_co_mention_mints_a_co_occurs_edge() {
    let (_t, _c, db, rec, observer) = harness(true).await;
    let res = observer
        .observe(
            &scope(),
            "today alpha and beta line up RESONATE",
            0,
            "s",
            None,
        )
        .await
        .unwrap();
    assert_eq!(res.concept_edges_minted, 1, "one co_occurs edge minted");
    assert_eq!(co_occurs_count(&db, "alpha"), 1);
    assert_eq!(co_occurs_count(&db, "beta"), 1);
    assert_eq!(
        connected_audit_count(&rec),
        1,
        "exactly one ConceptConnected audit row (creation-gated)"
    );
}

#[tokio::test]
async fn non_resonant_co_mention_mints_nothing() {
    let (_t, _c, db, _rec, observer) = harness(true).await;
    // Names both concepts, but no RESONATE marker → rolling lands on the quiet
    // axis, cosine 0 < floor → no survivors → no edge.
    let res = observer
        .observe(
            &scope(),
            "alpha and beta mentioned in passing",
            0,
            "s",
            None,
        )
        .await
        .unwrap();
    assert_eq!(res.concept_edges_minted, 0);
    assert_eq!(co_occurs_count(&db, "alpha"), 0);
}

#[tokio::test]
async fn idempotent_re_touch_does_not_duplicate_edge_or_audit() {
    let (_t, _c, db, rec, observer) = harness(true).await;
    let turn = "alpha beta together RESONATE";
    let first = observer
        .observe(&scope(), turn, 0, "s", None)
        .await
        .unwrap();
    assert_eq!(first.concept_edges_minted, 1);
    // Re-observe the identical resonant turn.
    let second = observer
        .observe(&scope(), turn, 1, "s", None)
        .await
        .unwrap();
    assert_eq!(second.concept_edges_minted, 0, "re-touch is not a new mint");
    assert_eq!(co_occurs_count(&db, "alpha"), 1, "still one edge");
    assert_eq!(
        connected_audit_count(&rec),
        1,
        "no second ConceptConnected row (creation-gated, not touch-gated)"
    );
}

#[tokio::test]
async fn resonant_recurrence_mints_a_proposed_node() {
    let (_t, _c, db, rec, observer) = harness(true).await;
    // The unknown kebab term appears ONCE per turn across 3 resonant turns
    // (each turn names alpha+beta and resonates) → resonance reaches the gate.
    for seq in 0..3 {
        let res = observer
            .observe(
                &scope(),
                "alpha beta with new-bridge-idea RESONATE",
                seq,
                "s",
                None,
            )
            .await
            .unwrap();
        if seq < 2 {
            assert_eq!(res.concept_nodes_minted, 0, "below the recurrence gate");
        } else {
            assert_eq!(res.concept_nodes_minted, 1, "third resonant turn mints");
        }
    }
    let node = db
        .resolve_concept(G, "new-bridge-idea")
        .unwrap()
        .expect("minted node exists");
    assert_eq!(node.status, ConceptStatus::Proposed);
    assert_eq!(node.kind, None, "ambient node is untyped");
    // Connected to its co-resonant context.
    assert!(co_occurs_count(&db, "new-bridge-idea") >= 2);
    assert!(
        rec.snapshot().iter().any(
            |e| matches!(e, ChainEvent::ConceptPromoted { handle, to_status, .. }
                if handle == "new-bridge-idea" && to_status == "proposed")
        ),
        "ConceptPromoted audit row present"
    );
}

#[tokio::test]
async fn high_frequency_low_resonance_term_never_mints() {
    let (_t, _c, db, _rec, observer) = harness(true).await;
    // The term recurs across MANY turns, but none resonate (no RESONATE marker,
    // so no survivors) → mentions climb, resonance stays 0 → never mints.
    for seq in 0..6 {
        let res = observer
            .observe(&scope(), "noise-term-here appears again", seq, "s", None)
            .await
            .unwrap();
        assert_eq!(res.concept_nodes_minted, 0);
    }
    assert!(
        db.resolve_concept(G, "noise-term-here").unwrap().is_none(),
        "occurrence is not salience — no node minted"
    );
}

#[tokio::test]
async fn observer_without_corpus_grows_nothing() {
    let (_t, _c, db, _rec, observer) = harness(false).await;
    let res = observer
        .observe(&scope(), "alpha beta together RESONATE", 0, "s", None)
        .await
        .unwrap();
    assert_eq!(res.concept_edges_minted, 0, "phase is gated on with_corpus");
    assert_eq!(res.concept_nodes_minted, 0);
    assert_eq!(co_occurs_count(&db, "alpha"), 0);
}

fn project_scope(project: &str) -> AttentionScope {
    AttentionScope {
        project: Some(project.into()),
        session_id: Some("ambient".into()),
        agent: Some("substrate".into()),
        privacy_tier: PrivacyTier::T1Project,
    }
}

/// Regression: one observer streaming turns from project A then project B
/// (before the rebuild cadence) must resolve EACH turn's own project's
/// concepts. The earlier single-gazetteer cache let the first project seen win
/// until the next codebook rebuild, so project B's turn would miss B's
/// concepts. Gazetteers are now keyed per `scope.project`.
#[tokio::test]
async fn gazetteer_is_keyed_per_project_not_first_project_wins() {
    let tmp = TempDir::new().unwrap();
    let corpus = Arc::new(CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap());
    corpus
        .upsert(
            &[
                anchor_chunk("a-one"),
                anchor_chunk("a-two"),
                anchor_chunk("b-one"),
                anchor_chunk("b-two"),
            ],
            &[
                axis(RESONANT_AXIS),
                axis(RESONANT_AXIS),
                axis(RESONANT_AXIS),
                axis(RESONANT_AXIS),
            ],
        )
        .await
        .unwrap();

    let recorder: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let sink: Arc<dyn ChainSink> = recorder.clone();
    let db = Arc::new(ThreadsDb::open_with_sink(tmp.path(), sink).unwrap());
    // Two project-scoped concepts per project (≥3-char handles so the gazetteer
    // floor admits them).
    for (proj, h, chunk) in [
        ("proja", "alpha-one", "a-one"),
        ("proja", "alpha-two", "a-two"),
        ("projb", "delta-one", "b-one"),
        ("projb", "delta-two", "b-two"),
    ] {
        db.ensure_concept(proj, h, ConceptStatus::Active).unwrap();
        attach_anchor_in(&db, proj, h, chunk);
    }

    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(AxisEmbedder);
    let pipeline = Arc::new(Pipeline::new(Arc::clone(&corpus), ingest, emb));
    let attn: Arc<dyn AttentionForwardStore> =
        Arc::new(InMemoryAttention::with_embedder(Arc::new(AxisEmbedder)));
    let observer = TurnObserver::new(pipeline, Arc::clone(&db))
        .with_attention(attn)
        .with_chain_sink(db.chain_sink())
        .with_concept_growth(test_cfg()) // codebook_rebuild_turns = 32
        .with_corpus(Arc::clone(&corpus));

    // Turn 1 — project A.
    let ra = observer
        .observe(
            &project_scope("proja"),
            "alpha-one with alpha-two RESONATE",
            0,
            "s",
            None,
        )
        .await
        .unwrap();
    assert_eq!(ra.concept_edges_minted, 1, "project A concepts resolve");

    // Turn 2 — project B, BEFORE the rebuild cadence (turns_since_build ≪ 32).
    let rb = observer
        .observe(
            &project_scope("projb"),
            "delta-one with delta-two RESONATE",
            1,
            "s",
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        rb.concept_edges_minted, 1,
        "project B concepts resolve via B's own gazetteer, not A's (regression)"
    );
    // The B edge is real and scoped to B's concepts.
    let b_edges = db
        .list_concept_edges("projb", "delta-one", EdgeDirection::From)
        .unwrap()
        .into_iter()
        .chain(
            db.list_concept_edges("projb", "delta-one", EdgeDirection::To)
                .unwrap(),
        )
        .filter(|e| e.relation == "co_occurs")
        .count();
    assert_eq!(b_edges, 1);
}

/// Trust gate (daemon path): a turn co-mentions an Active, a Proposed, and a
/// Candidate concept — all anchored on the resonant axis, so all *would* resonate
/// — but only the Active↔Proposed pair may mint. A Candidate is "known enough to
/// avoid duplicate minting, not trusted enough to shape topology": minting it
/// would let recurring operational lexis re-touch low-trust noise into durable
/// structure.
#[tokio::test]
async fn candidate_concepts_do_not_earn_co_occurs_edges() {
    let tmp = TempDir::new().unwrap();
    let corpus = Arc::new(CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap());
    corpus
        .upsert(
            &[
                anchor_chunk("act"),
                anchor_chunk("prop"),
                anchor_chunk("cand"),
            ],
            &[
                axis(RESONANT_AXIS),
                axis(RESONANT_AXIS),
                axis(RESONANT_AXIS),
            ],
        )
        .await
        .unwrap();

    let recorder: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let sink: Arc<dyn ChainSink> = recorder.clone();
    let db = Arc::new(ThreadsDb::open_with_sink(tmp.path(), sink).unwrap());
    db.ensure_concept(G, "active-one", ConceptStatus::Active)
        .unwrap();
    db.ensure_concept(G, "proposed-two", ConceptStatus::Proposed)
        .unwrap();
    db.ensure_concept(G, "candidate-three", ConceptStatus::Candidate)
        .unwrap();
    attach_anchor(&db, "active-one", "act");
    attach_anchor(&db, "proposed-two", "prop");
    attach_anchor(&db, "candidate-three", "cand");

    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(AxisEmbedder);
    let pipeline = Arc::new(Pipeline::new(Arc::clone(&corpus), ingest, emb));
    let attn: Arc<dyn AttentionForwardStore> =
        Arc::new(InMemoryAttention::with_embedder(Arc::new(AxisEmbedder)));
    let observer = TurnObserver::new(pipeline, Arc::clone(&db))
        .with_attention(attn)
        .with_chain_sink(db.chain_sink())
        .with_concept_growth(test_cfg())
        .with_corpus(Arc::clone(&corpus));

    let res = observer
        .observe(
            &scope(),
            "active-one with proposed-two beside candidate-three RESONATE",
            0,
            "s",
            None,
        )
        .await
        .unwrap();

    // Only the Active↔Proposed pair mints — the Candidate is trust-gated out.
    assert_eq!(res.concept_edges_minted, 1, "only active↔proposed mints");
    assert_eq!(co_occurs_count(&db, "active-one"), 1);
    assert_eq!(co_occurs_count(&db, "proposed-two"), 1);
    assert_eq!(
        co_occurs_count(&db, "candidate-three"),
        0,
        "candidate earns no co_occurs edge"
    );
}

/// Corpus + ledger with `alpha`/`beta` active on the resonant axis + a shared
/// attention store — but NO observer. Lets a test build several observers
/// against the same substrate, mirroring the per-trigger fresh observers in
/// `serve`.
async fn growth_fixture() -> (
    TempDir,
    Arc<CorpusStore>,
    Arc<ThreadsDb>,
    Arc<dyn AttentionForwardStore>,
) {
    let tmp = TempDir::new().unwrap();
    let corpus = Arc::new(CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap());
    corpus
        .upsert(
            &[anchor_chunk("alpha-anchor"), anchor_chunk("beta-anchor")],
            &[axis(RESONANT_AXIS), axis(RESONANT_AXIS)],
        )
        .await
        .unwrap();
    let db = Arc::new(
        ThreadsDb::open_with_sink(tmp.path(), Arc::new(RecordingSink::default())).unwrap(),
    );
    db.ensure_concept(G, "alpha", ConceptStatus::Active)
        .unwrap();
    db.ensure_concept(G, "beta", ConceptStatus::Active).unwrap();
    attach_anchor(&db, "alpha", "alpha-anchor");
    attach_anchor(&db, "beta", "beta-anchor");
    let attn: Arc<dyn AttentionForwardStore> =
        Arc::new(InMemoryAttention::with_embedder(Arc::new(AxisEmbedder)));
    (tmp, corpus, db, attn)
}

/// Build a fresh observer over a shared corpus/ledger/attention, sharing the
/// given concept-growth runtime — the production shape: each `serve` watch
/// trigger spawns a new observer but `ServeContext` shares the runtime.
fn observer_sharing(
    tmp: &TempDir,
    corpus: &Arc<CorpusStore>,
    db: &Arc<ThreadsDb>,
    attn: &Arc<dyn AttentionForwardStore>,
    runtime: ConceptGrowthRuntime,
) -> TurnObserver {
    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let pipeline = Arc::new(Pipeline::new(
        Arc::clone(corpus),
        ingest,
        Arc::new(AxisEmbedder) as Arc<dyn ChunkEmbedder>,
    ));
    TurnObserver::new(pipeline, Arc::clone(db))
        .with_attention(Arc::clone(attn))
        .with_chain_sink(db.chain_sink())
        .with_concept_growth(test_cfg())
        .with_corpus(Arc::clone(corpus))
        .with_concept_growth_runtime(runtime)
}

const RECUR_TURN: &str = "alpha beta with cross-scan-term RESONATE";

/// Regression for the scan-local node-recurrence bug: in `serve`, each watch
/// trigger spawns a fresh observer, so an unknown term recurring across
/// *separate* live turns would never reach the recurrence gate. With a shared
/// `ConceptGrowthRuntime`, recurrence persists across the observer boundary and
/// the node mints.
#[tokio::test]
async fn node_recurrence_accumulates_across_observers_via_shared_runtime() {
    let (tmp, corpus, db, attn) = growth_fixture().await;
    let runtime = ConceptGrowthRuntime::default();

    // Observer A (scan 1): two resonant turns — recurrence reaches 2, no mint.
    let obs_a = observer_sharing(&tmp, &corpus, &db, &attn, runtime.clone());
    for seq in 0..2 {
        let r = obs_a
            .observe(&scope(), RECUR_TURN, seq, "s", None)
            .await
            .unwrap();
        assert_eq!(r.concept_nodes_minted, 0, "below the recurrence gate");
    }
    drop(obs_a); // scan 1 ends; the observer is torn down

    // Observer B (scan 2): SAME runtime → recurrence continues 2 → 3 → mint.
    let obs_b = observer_sharing(&tmp, &corpus, &db, &attn, runtime.clone());
    let r = obs_b
        .observe(&scope(), RECUR_TURN, 2, "s", None)
        .await
        .unwrap();
    assert_eq!(
        r.concept_nodes_minted, 1,
        "shared runtime carries recurrence across the observer (scan) boundary"
    );
    assert!(
        db.resolve_concept(G, "cross-scan-term").unwrap().is_some(),
        "the cross-scan term minted a node"
    );
}

/// Control: without the shared runtime, the second observer starts fresh and
/// the recurrence resets — proving the runtime is what carries the state (the
/// bug's failure mode).
#[tokio::test]
async fn node_recurrence_resets_without_shared_runtime() {
    let (tmp, corpus, db, attn) = growth_fixture().await;

    // Observer A: two resonant turns on its OWN runtime.
    let obs_a = observer_sharing(&tmp, &corpus, &db, &attn, ConceptGrowthRuntime::default());
    for seq in 0..2 {
        obs_a
            .observe(&scope(), RECUR_TURN, seq, "s", None)
            .await
            .unwrap();
    }
    drop(obs_a);

    // Observer B: a DIFFERENT fresh runtime → recurrence resets to 1 → no mint.
    let obs_b = observer_sharing(&tmp, &corpus, &db, &attn, ConceptGrowthRuntime::default());
    let r = obs_b
        .observe(&scope(), RECUR_TURN, 2, "s", None)
        .await
        .unwrap();
    assert_eq!(
        r.concept_nodes_minted, 0,
        "fresh runtime loses the accumulation"
    );
    assert!(
        db.resolve_concept(G, "cross-scan-term").unwrap().is_none(),
        "no node minted without shared recurrence"
    );
}
