//! P6A gate test — observer-owned chain emission.
//!
//! Per `p6-attention-ema.md` ("Chain event ownership"):
//! - Direct `InMemoryAttention::attend()` calls (test paths,
//!   library users without an observer) emit **no** chain rows.
//! - Only the observer-mediated path — `TurnObserver::observe()`
//!   with `with_chain_sink` wired — persists
//!   `RollingVectorSnapshot` (on `Updated`) or
//!   `AttentionTurnSkipped` (on `Skipped`).
//!
//! In P6A only the `Updated` path fires — the noise gate that
//! produces `Skipped` ships in P6-full.

use std::sync::{Arc, Mutex as StdMutex};

use ostk_recall_attention::{
    AttendOutcome, AttentionForwardStore, InMemoryAttention, TurnObserver,
};
use ostk_recall_core::attention::{AttentionScope, PrivacyTier};
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline};
use ostk_recall_store::{ChainEvent, ChainSink, CorpusStore, IngestDb, StoreError, ThreadsDb};
use tempfile::TempDir;

#[derive(Default)]
struct RecordingSink {
    events: StdMutex<Vec<ChainEvent>>,
}

impl RecordingSink {
    fn take(&self) -> Vec<ChainEvent> {
        std::mem::take(&mut self.events.lock().unwrap())
    }
}

impl ChainSink for RecordingSink {
    fn append(&self, event: &ChainEvent) -> Result<(), StoreError> {
        self.events.lock().unwrap().push(event.clone());
        Ok(())
    }
}

struct FakeEmbedder {
    dim: usize,
}

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
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

fn scope() -> AttentionScope {
    AttentionScope {
        project: Some("p6a".into()),
        session_id: Some("chain-emit".into()),
        agent: Some("claude".into()),
        privacy_tier: PrivacyTier::T1Project,
    }
}

async fn build_pipeline(dim: usize) -> (Arc<Pipeline>, TempDir) {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
    let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
    let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });
    let p = Pipeline::new(store, ingest, emb);
    (Arc::new(p), tmp)
}

#[tokio::test]
async fn direct_attend_emits_no_chain_rows() {
    // No observer, no chain_sink — just the runtime. The
    // `AttendOutcome::Updated` is the only signal; no persistence.
    let attn = InMemoryAttention::new();
    let outcome = attn.attend(&scope(), "any text").await.unwrap();
    assert!(matches!(outcome, AttendOutcome::Updated { .. }));
    // No assertion about chain rows possible — there is no sink
    // wired. The test exists to document the contract: direct
    // attend is intentionally side-effect-free at the chain layer.
}

#[tokio::test]
async fn observer_mediated_attend_emits_rolling_vector_snapshot() {
    let (pipeline, _corpus_tmp) = build_pipeline(16).await;
    let store_tmp = TempDir::new().unwrap();
    let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let chain: Arc<dyn ChainSink> = sink.clone();
    let db = Arc::new(ThreadsDb::open_with_sink(store_tmp.path(), chain.clone()).unwrap());

    let attn: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    let observer = TurnObserver::new(pipeline, db)
        .with_attention(attn)
        .with_chain_sink(chain);

    observer
        .observe(&scope(), "first observed turn", 0, "sess-1", None)
        .await
        .unwrap();

    let events = sink.take();
    let snapshot = events.iter().find_map(|e| match e {
        ChainEvent::RollingVectorSnapshot { vec, lambda, .. } => Some((vec.clone(), *lambda)),
        _ => None,
    });
    let (vec, lambda) = snapshot.expect("observer must emit RollingVectorSnapshot");
    assert!(!vec.is_empty(), "snapshot vec must be populated");
    assert!(
        lambda > 0.0 && lambda <= 1.0,
        "snapshot lambda must be in valid range, got {lambda}"
    );

    // No `AttentionTurnSkipped` should fire in P6A (no noise gate).
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ChainEvent::AttentionTurnSkipped { .. })),
        "P6A must not emit AttentionTurnSkipped (gate is P6-full)"
    );
}

#[tokio::test]
async fn rolling_vec_snapshot_round_trips_through_replay() {
    // Review-fix regression test: a RollingVectorSnapshot captured
    // from one InMemoryAttention must, after seed_rolling_vec, drive
    // the same scope_vector on a fresh InMemoryAttention. This is
    // the restart-safety contract that P9a/P9b depend on.
    use ostk_recall_attention::InMemoryAttention;

    let (pipeline, _corpus_tmp) = build_pipeline(16).await;
    let store_tmp = TempDir::new().unwrap();
    let sink: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let chain: Arc<dyn ChainSink> = sink.clone();
    let db = Arc::new(ThreadsDb::open_with_sink(store_tmp.path(), chain.clone()).unwrap());

    // Original session: build attention, attend twice so rolling_vec
    // has actually advanced via EMA blend (not just seeded).
    let original: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    let observer = TurnObserver::new(pipeline, db)
        .with_attention(Arc::clone(&original))
        .with_chain_sink(chain);
    observer
        .observe(&scope(), "first observed turn", 0, "sess-1", None)
        .await
        .unwrap();
    observer
        .observe(&scope(), "second observed turn diverges", 1, "sess-1", None)
        .await
        .unwrap();

    let pre_restart = original.scope_vector(&scope()).await.unwrap().unwrap();

    // Find the latest snapshot for this scope (mirrors the cli
    // replay's "keep latest per scope" rule).
    let events = sink.take();
    let snapshot = events
        .iter()
        .filter_map(|e| match e {
            ChainEvent::RollingVectorSnapshot { vec, .. } => Some(vec.clone()),
            _ => None,
        })
        .last()
        .expect("at least one RollingVectorSnapshot must have been emitted");
    assert_eq!(
        snapshot, pre_restart,
        "the snapshot vec must equal the live rolling channel"
    );

    // Fresh session: seed the rolling channel from the snapshot,
    // exactly as cli::replay_chain_into_attention does.
    let fresh = InMemoryAttention::new();
    fresh
        .seed_rolling_vec(&scope(), snapshot.clone())
        .await
        .unwrap();
    let post_restart = fresh.scope_vector(&scope()).await.unwrap().unwrap();
    assert_eq!(
        post_restart, snapshot,
        "after replay seed, scope_vector must return the snapshot verbatim"
    );
}

#[tokio::test]
async fn observer_without_chain_sink_emits_no_attention_events() {
    // `with_chain_sink` is optional: observers wired only to a
    // ledger (legacy / test fixtures) still call `attend()` for
    // the score tier but emit zero attention-side chain rows.
    let (pipeline, _corpus_tmp) = build_pipeline(16).await;
    let store_tmp = TempDir::new().unwrap();
    let recording: Arc<RecordingSink> = Arc::new(RecordingSink::default());
    let ledger_sink: Arc<dyn ChainSink> = recording.clone();
    let db = Arc::new(ThreadsDb::open_with_sink(store_tmp.path(), ledger_sink).unwrap());

    let attn: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    // Note: NO with_chain_sink.
    let observer = TurnObserver::new(pipeline, db).with_attention(attn);

    observer
        .observe(&scope(), "another turn", 0, "sess-2", None)
        .await
        .unwrap();

    let events = recording.take();
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, ChainEvent::RollingVectorSnapshot { .. })),
        "observer without chain_sink must not emit RollingVectorSnapshot"
    );
}
