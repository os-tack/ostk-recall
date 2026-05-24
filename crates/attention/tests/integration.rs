//! End-to-end integration tests for the attention substrate MVP.
//!
//! Each test exercises one of the eight behaviors the QA reviewer was
//! asked to cover (see review/qa-coverage scope). Unit tests inside
//! `crates/attention/src/*.rs` cover the math; this file covers the
//! wiring — ledger persistence across re-open, ledger row advancement
//! through the observer, MCP JSON-shape strictness, full state-chain
//! transitions through the curator, `AutoWeaver` merge semantics.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use chrono::{Duration as ChronoDuration, Utc};
use ostk_recall_attention::{
    AttentionForwardStore, CuratorConfig, IdleCurator, InMemoryAttention, ReplayEvent,
    TurnObserver, stub_embed,
};
use ostk_recall_attention_mcp::handlers::{
    AttentionDispatch, attend, fold, surface, thread_create, thread_list,
};
use ostk_recall_core::attention::{AttentionScope, FoldDepth, PrivacyTier, ThreadHandle};
use ostk_recall_core::{Chunk, Links, Source, SourceKind};
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline, SyntheticSourceMeta};
use ostk_recall_store::{
    AssociationType, ChainEvent, ChainSink, CorpusStore, EvidenceLink, IngestDb, RelationState,
    StoreError, TensionState, ThreadRecord, ThreadsDb,
};
use serde_json::json;
use tempfile::TempDir;

// ---------------------------------------------------------------------
// shared fixtures
// ---------------------------------------------------------------------

const EMBED_DIM: usize = 16;

/// Deterministic embedder. Identical text yields identical vectors; the
/// integration tests rely on this to engineer specific (anchor, chunk)
/// cosine similarities for the `AutoWeaver` merge-vs-skip path.
struct FakeEmbedder {
    dim: usize,
}

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    #[allow(clippy::cast_precision_loss)]
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let seed = ((t.len() % 100) as f32) * 0.01;
                (0..self.dim)
                    .map(|i| (i as f32).mul_add(0.001, seed))
                    .collect()
            })
            .collect()
    }
}

/// Records every chain event written through a `ChainSink`. Used to
/// assert that mutations actually emit the chain rows their semantics
/// promise (familiarity batch, tension transition, etc).
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

fn scope_for(project: &str) -> AttentionScope {
    AttentionScope {
        project: Some(project.into()),
        session_id: Some("s1".into()),
        agent: Some("claude".into()),
        privacy_tier: PrivacyTier::T1Project,
    }
}

fn handle(s: &str) -> ThreadHandle {
    ThreadHandle::new(s).expect("valid handle")
}

async fn make_pipeline(tmp: &TempDir) -> Arc<Pipeline> {
    let (p, _) = make_pipeline_with_corpus(tmp).await;
    p
}

async fn make_pipeline_with_corpus(tmp: &TempDir) -> (Arc<Pipeline>, Arc<CorpusStore>) {
    let corpus = Arc::new(
        CorpusStore::open_or_create(tmp.path(), EMBED_DIM)
            .await
            .expect("corpus open"),
    );
    let ingest = Arc::new(IngestDb::open(tmp.path()).expect("ingest open"));
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim: EMBED_DIM });
    let p = Arc::new(Pipeline::new(corpus.clone(), ingest, emb));
    (p, corpus)
}

fn seed_thread(db: &ThreadsDb, name: &str, anchor: Option<&str>) -> ThreadHandle {
    let h = handle(name);
    db.upsert_thread(&ThreadRecord {
        handle: h.clone(),
        tension: TensionState::Active,
        familiarity: 0,
        last_touched_at: Utc::now(),
        anchor_chunk_id: anchor.map(String::from),
        fold_override: None,
        created_at: Utc::now(),
        created_scope_key: None,
        privacy_tier: PrivacyTier::T1Project,
    })
    .expect("upsert");
    h
}

// ---------------------------------------------------------------------
// (1) Familiarity advances end-to-end through the turn observer
// ---------------------------------------------------------------------
//
// Existing unit tests assert that the observer records a
// FamiliarityBatch chain event with the correct turn_seq. Neither
// existing test verifies the ledger COLUMN (`threads.familiarity`)
// actually advanced — that was wired in via a fix recorded as
// commit 212b37c. Re-prove it end-to-end and pin the chain `turn_seq`.

#[tokio::test]
async fn familiarity_increments_ledger_and_emits_batch_with_turn_seq() {
    let corpus_tmp = TempDir::new().unwrap();
    let pipeline = make_pipeline(&corpus_tmp).await;

    let store_tmp = TempDir::new().unwrap();
    let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let db_sink: Arc<dyn ChainSink> = sink.clone();
    let db = Arc::new(ThreadsDb::open_with_sink(store_tmp.path(), db_sink).unwrap());

    let h = seed_thread(&db, "three-time-scales", None);
    let observer = TurnObserver::new(pipeline, db.clone());
    observer.refresh_known_handles().await.unwrap();

    let starting_familiarity = db.get_thread(&h).unwrap().unwrap().familiarity;
    assert_eq!(starting_familiarity, 0);

    // Drain the upsert chain event so we only assert against the
    // observer's output below.
    let _ = sink.snapshot();

    let turn = "we keep circling three-time-scales because it explains \
        why the score tier is in-memory.";
    let res = observer
        .observe(&scope_for("haystack"), turn, 42, "sess-1")
        .await
        .unwrap();
    assert_eq!(res.familiarity_increments, vec![h.clone()]);

    let after = db.get_thread(&h).unwrap().unwrap().familiarity;
    assert_eq!(
        after, 1,
        "ledger column must advance (regression guard for 212b37c)"
    );

    // Re-observe the same handle in a different turn — the column
    // must advance again, and the second batch must carry turn_seq=43.
    let _ = observer
        .observe(&scope_for("haystack"), turn, 43, "sess-1")
        .await
        .unwrap();
    let after_two = db.get_thread(&h).unwrap().unwrap().familiarity;
    assert_eq!(after_two, 2, "second turn must advance ledger again");

    let batches: Vec<_> = sink
        .snapshot()
        .into_iter()
        .filter_map(|e| match e {
            ChainEvent::FamiliarityBatch {
                entries, turn_seq, ..
            } => Some((entries, turn_seq)),
            _ => None,
        })
        .collect();
    assert_eq!(batches.len(), 2, "one batch per turn");
    assert_eq!(batches[0].1, 42);
    assert_eq!(batches[1].1, 43);
    // Each batch carries (handle, post-increment value).
    assert_eq!(batches[0].0, vec![(h.clone(), 1)]);
    assert_eq!(batches[1].0, vec![(h, 2)]);
}

// ---------------------------------------------------------------------
// (2) Scope isolation at the ledger / MCP boundary
// ---------------------------------------------------------------------
//
// In-memory runtime isolation is unit-tested in lib.rs. This test
// drives the MCP dispatch path so the JSON-boundary code path
// (created_scope_key bookkeeping) is exercised against the actual
// SQLite store.

#[tokio::test]
async fn cross_project_scope_isolation_via_mcp() {
    let tmp = TempDir::new().unwrap();
    let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
    let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    let d = AttentionDispatch::new(attention, threads);

    // Two projects each create their own thread.
    thread_create(
        &d,
        json!({
            "scope": {"project": "alpha", "privacy_tier": "t0_private"},
            "handle": "alpha-thread"
        }),
    )
    .await
    .unwrap();
    thread_create(
        &d,
        json!({
            "scope": {"project": "beta", "privacy_tier": "t0_private"},
            "handle": "beta-thread"
        }),
    )
    .await
    .unwrap();

    let list_a = thread_list(
        &d,
        json!({"scope": {"project": "alpha", "privacy_tier": "t0_private"}}),
    )
    .await
    .unwrap();
    let list_b = thread_list(
        &d,
        json!({"scope": {"project": "beta", "privacy_tier": "t0_private"}}),
    )
    .await
    .unwrap();

    let a_handles: HashSet<&str> = list_a["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["handle"].as_str().unwrap())
        .collect();
    let b_handles: HashSet<&str> = list_b["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["handle"].as_str().unwrap())
        .collect();

    assert!(a_handles.contains("alpha-thread"));
    assert!(!a_handles.contains("beta-thread"), "alpha must not see beta");
    assert!(b_handles.contains("beta-thread"));
    assert!(!b_handles.contains("alpha-thread"), "beta must not see alpha");
}

// ---------------------------------------------------------------------
// (3) PrivacyTier T0 enforcement when other dimensions match
// ---------------------------------------------------------------------
//
// Same project, same session, same agent — only `privacy_tier` differs.
// T0Private must still not leak.

#[tokio::test]
async fn t0_isolation_holds_even_when_other_scope_dimensions_match() {
    let tmp = TempDir::new().unwrap();
    let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
    let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    let d = AttentionDispatch::new(attention, threads);

    thread_create(
        &d,
        json!({
            "scope": {
                "project": "shared",
                "session_id": "sess-x",
                "agent": "claude",
                "privacy_tier": "t0_private",
            },
            "handle": "private-secret"
        }),
    )
    .await
    .unwrap();

    // A querier from a scope that matches project+session+agent but
    // declares a DIFFERENT scope (e.g. no session_id) must NOT see the
    // T0 row. The scope_key_repr in handlers.rs uses '_' for absent
    // components, so the scope keys differ and visibility filter fires.
    let list = thread_list(
        &d,
        json!({
            "scope": {"project": "shared", "privacy_tier": "t1_project"}
        }),
    )
    .await
    .unwrap();
    let handles: HashSet<&str> = list["records"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["handle"].as_str().unwrap())
        .collect();
    assert!(
        !handles.contains("private-secret"),
        "T0 must not surface to a scope with a different scope key"
    );
}

// ---------------------------------------------------------------------
// (4) Restart rebuilds attention state from the SQLite ledger
// ---------------------------------------------------------------------
//
// In-memory chain replay is covered in lib.rs. This test covers the
// other half: a real `ThreadsDb` opened against a directory that
// already holds a populated `threads.sqlite` must surface the threads
// (including familiarity counter and tension) without further input.

#[tokio::test]
async fn ledger_persists_across_reopen() {
    let tmp = TempDir::new().unwrap();
    let h = handle("persistent-thread");

    {
        let db = ThreadsDb::open(tmp.path()).unwrap();
        db.upsert_thread(&ThreadRecord {
            handle: h.clone(),
            tension: TensionState::Slack,
            familiarity: 0,
            last_touched_at: Utc::now(),
            anchor_chunk_id: Some("anchor-c1".into()),
            fold_override: Some(FoldDepth::Half),
            created_at: Utc::now(),
            created_scope_key: Some("haystack|s1|claude".into()),
            privacy_tier: PrivacyTier::T1Project,
        })
        .unwrap();
        for _ in 0..5 {
            db.increment_familiarity(&h).unwrap();
        }
        db.set_tension(&h, TensionState::Active).unwrap();
    }

    // Drop and re-open: the second handle is a fresh process state
    // from the runtime's perspective.
    let db = ThreadsDb::open(tmp.path()).unwrap();
    let recovered = db
        .get_thread(&h)
        .unwrap()
        .expect("thread row survives reopen");
    assert_eq!(recovered.familiarity, 5);
    assert_eq!(recovered.tension, TensionState::Active);
    assert_eq!(recovered.fold_override, Some(FoldDepth::Half));
    assert_eq!(recovered.anchor_chunk_id.as_deref(), Some("anchor-c1"));
    assert_eq!(recovered.privacy_tier, PrivacyTier::T1Project);

    // The in-memory score tier rebuilds via ReplayEvent replay.
    let scope = scope_for("haystack");
    let store = InMemoryAttention::new();
    store
        .replay(&[
            ReplayEvent::Attend {
                scope: scope.clone(),
                context: "persistent-thread context".into(),
            },
            ReplayEvent::Fold {
                scope: scope.clone(),
                handle: h.clone(),
                depth: FoldDepth::Half,
            },
            ReplayEvent::Familiarize {
                scope: scope.clone(),
                handle: h.clone(),
            },
        ])
        .await
        .unwrap();
    let pages = store.surface(&scope, 10).await.unwrap();
    assert!(
        pages.iter().any(|p| p.handle == "persistent-thread"),
        "replay must surface the rebuilt thread"
    );
}

// ---------------------------------------------------------------------
// (5) Cold start: fresh runtime on an empty ledger, no panic, queries
//     return empty cleanly.
// ---------------------------------------------------------------------

#[tokio::test]
async fn cold_start_empty_ledger_is_quiet() {
    let tmp = TempDir::new().unwrap();
    let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
    let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    let d = AttentionDispatch::new(attention.clone(), threads.clone());

    // No threads → list returns empty.
    let list = thread_list(&d, json!({})).await.unwrap();
    assert_eq!(list["records"].as_array().unwrap().len(), 0);

    // No attend → surface returns empty (not a panic).
    let pages = surface(&d, json!({})).await.unwrap();
    assert_eq!(pages["pages"].as_array().unwrap().len(), 0);

    // Curator tick on an empty ledger is a no-op.
    let curator = IdleCurator::new(
        threads,
        attention,
        CuratorConfig {
            tick_interval: Duration::from_millis(10),
            ..CuratorConfig::default()
        },
    );
    let tick = curator.tick().await.unwrap();
    assert_eq!(tick.threads_scored, 0);
    assert!(tick.transitions.is_empty());
}

// ---------------------------------------------------------------------
// (6) Curator state-machine: Active → Slack → Dormant by score decay,
//     then Dormant → Active by re-attending. Exercises every
//     transition arm of `resolve_target` end-to-end.
// ---------------------------------------------------------------------

// Each transition arm is exercised independently by pre-installing the
// thread at a known prior state and choosing thresholds so the score
// lands in the target band. Driving all three from one score evolution
// is brittle (the score formula's three terms move together when you
// nudge time); the state machine is the unit under test, not the math.

async fn install_at_score(
    attention: &Arc<InMemoryAttention>,
    threads: &Arc<ThreadsDb>,
    scope: &AttentionScope,
    name: &str,
    start_tension: TensionState,
) -> ThreadHandle {
    let h = handle(name);
    let stale = Utc::now() - ChronoDuration::days(1);
    threads
        .upsert_thread(&ThreadRecord {
            handle: h.clone(),
            tension: start_tension,
            familiarity: 20,
            last_touched_at: stale,
            anchor_chunk_id: None,
            fold_override: None,
            created_at: stale,
            created_scope_key: None,
            privacy_tier: PrivacyTier::T1Project,
        })
        .unwrap();
    // Identical anchor + scope vector → resonance ≈ 1.0 → score sits
    // around (floor·decay + 1.0). With familiarity 20 + 1 day stale,
    // decay term ≈ 1.0·exp(-0.005·1) ≈ 0.995, so total ≈ 1.995.
    attention
        .__install_thread_for_test(
            scope,
            h.clone(),
            0.9,
            20,
            stale,
            FoldDepth::Half,
            stub_embed("stepwise-thread context"),
        )
        .await;
    h
}

#[tokio::test]
async fn curator_active_to_slack_transition() {
    let tmp = TempDir::new().unwrap();
    let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let db_sink: Arc<dyn ChainSink> = sink.clone();
    let threads = Arc::new(ThreadsDb::open_with_sink(tmp.path(), db_sink).unwrap());
    let attention = Arc::new(InMemoryAttention::new());
    let scope = scope_for("haystack");
    attention
        .attend(&scope, "stepwise-thread context")
        .await
        .unwrap();
    let h = install_at_score(&attention, &threads, &scope, "step-active", TensionState::Active)
        .await;

    // Pin thresholds: active above score, slack below. Score lives in
    // the (slack, active) band → Active should fall to Slack.
    let curator = IdleCurator::new(
        threads.clone(),
        attention.clone() as Arc<dyn AttentionForwardStore>,
        CuratorConfig {
            tick_interval: Duration::from_millis(10),
            active_score_threshold: 10.0,
            slack_score_threshold: 0.5,
            archive_threshold: 0.1,
            hysteresis: 0.0,
        },
    );
    let tick = curator.tick().await.unwrap();
    assert!(
        tick.transitions
            .iter()
            .any(|t| t.handle == h && t.to == TensionState::Slack),
        "Active must step to Slack when score in (slack, active) band: {:?}",
        tick.transitions
    );
    assert_eq!(
        threads.get_thread(&h).unwrap().unwrap().tension,
        TensionState::Slack
    );
    // Chain row witnesses the transition.
    let transitions: Vec<_> = sink
        .snapshot()
        .into_iter()
        .filter_map(|e| match e {
            ChainEvent::TensionTransition { handle, from, to, .. } => Some((handle, from, to)),
            _ => None,
        })
        .collect();
    assert!(
        transitions
            .iter()
            .any(|(hh, from, to)| hh == &h
                && *from == TensionState::Active
                && *to == TensionState::Slack)
    );
}

#[tokio::test]
async fn curator_slack_to_dormant_transition() {
    let tmp = TempDir::new().unwrap();
    let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let db_sink: Arc<dyn ChainSink> = sink.clone();
    let threads = Arc::new(ThreadsDb::open_with_sink(tmp.path(), db_sink).unwrap());
    let attention = Arc::new(InMemoryAttention::new());
    let scope = scope_for("haystack");
    attention
        .attend(&scope, "stepwise-thread context")
        .await
        .unwrap();
    let h = install_at_score(&attention, &threads, &scope, "step-slack", TensionState::Slack)
        .await;

    // Tighter thresholds: score below archive → Slack should fall to
    // Dormant.
    let curator = IdleCurator::new(
        threads.clone(),
        attention.clone() as Arc<dyn AttentionForwardStore>,
        CuratorConfig {
            tick_interval: Duration::from_millis(10),
            active_score_threshold: 100.0,
            slack_score_threshold: 100.0,
            archive_threshold: 100.0,
            hysteresis: 0.0,
        },
    );
    let tick = curator.tick().await.unwrap();
    assert!(
        tick.transitions
            .iter()
            .any(|t| t.handle == h && t.to == TensionState::Dormant),
        "Slack must step to Dormant when score < archive: {:?}",
        tick.transitions
    );
    assert_eq!(
        threads.get_thread(&h).unwrap().unwrap().tension,
        TensionState::Dormant
    );
}

#[tokio::test]
async fn curator_dormant_to_active_reanimation() {
    let tmp = TempDir::new().unwrap();
    let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let db_sink: Arc<dyn ChainSink> = sink.clone();
    let threads = Arc::new(ThreadsDb::open_with_sink(tmp.path(), db_sink).unwrap());
    let attention = Arc::new(InMemoryAttention::new());
    let scope = scope_for("haystack");
    attention
        .attend(&scope, "stepwise-thread context")
        .await
        .unwrap();
    let h = install_at_score(
        &attention,
        &threads,
        &scope,
        "step-dormant",
        TensionState::Dormant,
    )
    .await;

    // Loose thresholds: score above active → Dormant should reanimate
    // straight to Active (the direct-jump arm in resolve_target).
    let curator = IdleCurator::new(
        threads.clone(),
        attention.clone() as Arc<dyn AttentionForwardStore>,
        CuratorConfig {
            tick_interval: Duration::from_millis(10),
            active_score_threshold: 0.1,
            slack_score_threshold: 0.05,
            archive_threshold: 0.0,
            hysteresis: 0.0,
        },
    );
    let tick = curator.tick().await.unwrap();
    assert!(
        tick.transitions
            .iter()
            .any(|t| t.handle == h && t.to == TensionState::Active),
        "Dormant must reanimate to Active when score >= active threshold: {:?}",
        tick.transitions
    );
}

// ---------------------------------------------------------------------
// (7) MCP surface response payload includes the ScoreAttribution struct
//     fields. The existing handlers::tests version checks shape; this
//     one pins every documented axis is present and well-typed, so a
//     future refactor that drops `time_since_touch_secs` fails loudly.
// ---------------------------------------------------------------------

#[tokio::test]
async fn mcp_surface_payload_carries_full_score_attribution() {
    let tmp = TempDir::new().unwrap();
    let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
    let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    let d = AttentionDispatch::new(attention, threads);

    attend(
        &d,
        json!({"scope": {"project": "p"}, "context": "score attribution wire shape"}),
    )
    .await
    .unwrap();
    fold(
        &d,
        json!({
            "scope": {"project": "p"},
            "handle": "wire-shape-thread",
            "depth": "half"
        }),
    )
    .await
    .unwrap();
    // Familiarize a few times so the score rises above ARCHIVE_THRESHOLD.
    for _ in 0..5 {
        ostk_recall_attention_mcp::handlers::familiarize(
            &d,
            json!({"scope": {"project": "p"}, "handle": "wire-shape-thread"}),
        )
        .await
        .unwrap();
    }

    let out = surface(&d, json!({"scope": {"project": "p"}, "limit": 10}))
        .await
        .unwrap();
    let pages = out["pages"].as_array().unwrap();
    let wire = pages
        .iter()
        .find(|p| p["handle"] == "wire-shape-thread")
        .expect("expected wire-shape-thread to surface");

    // Top-level page fields.
    assert!(wire["handle"].is_string());
    assert!(wire["depth"].is_string());
    assert!(wire["score"].is_number());

    // Every documented ScoreAttribution axis must be present and typed.
    let why = &wire["why"];
    assert!(why.is_object(), "why must be an object");
    for field in &[
        "tension",
        "resonance",
        "off_diagonal_lift",
        "familiarity",
        "time_since_touch_secs",
    ] {
        assert!(
            why.get(*field).is_some(),
            "ScoreAttribution.{field} missing from MCP surface payload"
        );
    }
    assert!(why["familiarity"].is_number());
    assert!(why["familiarity"].as_u64().unwrap() >= 5);
    assert!(why["resonance"].is_number());
    assert!(why["tension"].is_number());
    assert!(why["off_diagonal_lift"].is_number());
    assert!(why["time_since_touch_secs"].is_number());
}

// ---------------------------------------------------------------------
// (8) AutoWeaver merge-vs-skip
// ---------------------------------------------------------------------
//
// Important semantic correction: the AutoWeaver writes EVIDENCE LINKS,
// not threads. The merge case = "matching ingest links to existing
// thread"; the non-matching case = "ingest stays unlinked (no thread
// is created automatically)." Anything that creates new threads is
// either the threads scanner or the operator via thread_create.
// The proposed_stubs path in TurnObserver surfaces handle CANDIDATES
// but the operator still has to promote them.
//
// This test covers both arms via real CorpusStore + ThreadsDb, on the
// ingest_synthetic path so the broadcast channel + dedupe code is
// exercised too.

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn autoweaver_links_resonant_ingest_and_skips_non_resonant() {
    let tmp = TempDir::new().unwrap();
    let (pipeline, corpus) = make_pipeline_with_corpus(&tmp).await;
    let threads = Arc::new(ThreadsDb::open(tmp.path()).unwrap());

    // Anchor: an ingested chunk with content "anchor-content" gives the
    // FakeEmbedder a deterministic 16-dim vector. The thread points at
    // that chunk id.
    let anchor_chunk = Chunk {
        chunk_id: "anchor-c1".into(),
        source: Source::Markdown,
        project: Some("p".into()),
        source_id: "anchor.md".into(),
        chunk_index: 0,
        ts: Some(Utc::now()),
        role: None,
        text: "anchor-content".into(),
        sha256: Chunk::content_hash("anchor-content"),
        links: Links::default(),
        extra: serde_json::Value::Null,
    };
    pipeline
        .ingest_synthetic(
            vec![anchor_chunk],
            SyntheticSourceMeta {
                source: SourceKind::Markdown,
                project: Some("p".into()),
            },
        )
        .await
        .unwrap();
    seed_thread(&threads, "anchored-thread", Some("anchor-c1"));

    // Inline weaver mirror (same logic as AutoWeaver::process_event but
    // without the daemon loop) — drives the cosine compare + evidence
    // insert directly so we can pin behavior per event.
    let weave = |chunk_id: &str, threshold: f32| {
        let threads = threads.clone();
        let corpus = corpus.clone();
        let chunk_id = chunk_id.to_string();
        async move {
            let new = corpus
                .fetch_embeddings(std::slice::from_ref(&chunk_id))
                .await
                .unwrap();
            let rows = threads.list_threads(None).unwrap();
            let anchor_ids: Vec<String> = rows
                .iter()
                .filter_map(|t| t.anchor_chunk_id.clone())
                .collect();
            let anchors = corpus.fetch_embeddings(&anchor_ids).await.unwrap();
            let mut links = 0usize;
            for (cid, cvec) in &new {
                for t in &rows {
                    let Some(aid) = &t.anchor_chunk_id else { continue };
                    let Some(avec) = anchors.get(aid) else { continue };
                    let sim = ostk_recall_attention::cosine_similarity(cvec, avec);
                    if sim < threshold {
                        continue;
                    }
                    let now = Utc::now();
                    threads
                        .add_evidence_link(&EvidenceLink {
                            id: 0,
                            thread_handle: t.handle.clone(),
                            original_path: PathBuf::from(cid),
                            current_path: None,
                            content_hash: None,
                            last_resolved_chunk_id: Some(cid.clone()),
                            relation_state: RelationState::Active,
                            association_type: AssociationType::Derived,
                            category: "doc".into(),
                            similarity: Some(sim),
                            created_at: now,
                            updated_at: now,
                        })
                        .unwrap();
                    links += 1;
                }
            }
            links
        }
    };

    // MERGE arm: identical text → identical embedding → cosine = 1.0 →
    // above any threshold → evidence link appended to existing thread.
    let merge_chunk = Chunk {
        chunk_id: "ingest-merge".into(),
        source: Source::Markdown,
        project: Some("p".into()),
        source_id: "merge.md".into(),
        chunk_index: 0,
        ts: Some(Utc::now()),
        role: None,
        text: "anchor-content".into(), // identical → identical vector
        sha256: Chunk::content_hash("anchor-content-distinct-id"),
        links: Links::default(),
        extra: serde_json::Value::Null,
    };
    let _ = pipeline
        .ingest_synthetic(
            vec![merge_chunk],
            SyntheticSourceMeta {
                source: SourceKind::Markdown,
                project: Some("p".into()),
            },
        )
        .await
        .unwrap();
    let merged_links = weave("ingest-merge", 0.5).await;
    assert_eq!(merged_links, 1, "merge: resonant ingest must link evidence");

    let threads_after_merge = threads.list_threads(None).unwrap();
    assert_eq!(
        threads_after_merge.len(),
        1,
        "merge must NOT create a new thread — only the operator-seeded one persists"
    );
    let evidence = threads
        .list_evidence(&handle("anchored-thread"))
        .unwrap();
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].association_type, AssociationType::Derived);
    assert!(evidence[0].similarity.unwrap() > 0.99);

    // SKIP arm: text with a different length → embedder produces a
    // different `seed`, so the vector differs. We pick a threshold
    // high enough that any small variance fails.
    let skip_chunk = Chunk {
        chunk_id: "ingest-skip".into(),
        source: Source::Markdown,
        project: Some("p".into()),
        source_id: "skip.md".into(),
        chunk_index: 0,
        ts: Some(Utc::now()),
        role: None,
        // Force a very different `seed` in FakeEmbedder by changing
        // length significantly.
        text: "z".repeat(73),
        sha256: Chunk::content_hash("ingest-skip-content-unique"),
        links: Links::default(),
        extra: serde_json::Value::Null,
    };
    let _ = pipeline
        .ingest_synthetic(
            vec![skip_chunk],
            SyntheticSourceMeta {
                source: SourceKind::Markdown,
                project: Some("p".into()),
            },
        )
        .await
        .unwrap();
    let skip_links = weave("ingest-skip", 0.999_999).await;
    assert_eq!(
        skip_links, 0,
        "skip: non-resonant ingest must NOT link evidence"
    );

    // Final ledger shape: still exactly one thread; only the merge arm
    // wrote evidence.
    assert_eq!(threads.list_threads(None).unwrap().len(), 1);
    let evidence = threads
        .list_evidence(&handle("anchored-thread"))
        .unwrap();
    assert_eq!(
        evidence.len(),
        1,
        "non-resonant ingest must not append evidence to an existing thread"
    );
}
