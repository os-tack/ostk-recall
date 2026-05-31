//! P9b-full — daemon-path integration: `try_refresh_lens` against a real
//! Lance corpus + a real `ThreadsDb` access ledger.
//!
//! Exercises the Commit-5 wiring end-to-end: the loop's two-phase
//! `enrich_for_lens` (attaching the ledger reader), the Lens-profile rank
//! engine (attention_affinity + freshness), and the refractory penalty
//! reading `LensIncluded` events from the live ledger. Where the unit tests
//! (`lens_refractory.rs`, `lens_portfolio_diversity.rs`) pin the math with
//! constructed hits, this drives the same path the `serve` daemon runs.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{Duration, Utc};
use tempfile::TempDir;

use ostk_recall_cli::lens_loop::{LensRefreshDecision, LensTickSnapshot, try_refresh_lens};
use ostk_recall_cli::lens_state::LensState;
use ostk_recall_core::{Chunk, FacetSet, Links, RankProfile, Source, default_profile_weights};
use ostk_recall_query::lens::LensConfig;
use ostk_recall_query::rank::build_engine_from_weights;
use ostk_recall_store::corpus::CorpusStore;
use ostk_recall_store::threads::{ChainSink, ThreadsDb};
use ostk_recall_store::{ChainEvent, ChainLogReader, SqliteChainSink};

const DIM: usize = 8;

/// Unwrap a `Refresh` decision or panic with the actual variant (and the
/// build error, if any) so failures are diagnosable.
fn expect_refresh(d: LensRefreshDecision) -> ostk_recall_query::lens::Lens {
    match d {
        LensRefreshDecision::Refresh { lens, .. } => lens,
        LensRefreshDecision::BuildFailed(e) => panic!("expected Refresh, got BuildFailed: {e}"),
        LensRefreshDecision::EmptyMind => panic!("expected Refresh, got EmptyMind"),
        LensRefreshDecision::NoTrigger => panic!("expected Refresh, got NoTrigger"),
        LensRefreshDecision::UnchangedContent { .. } => {
            panic!("expected Refresh, got UnchangedContent")
        }
    }
}

fn chunk(id: &str, text: &str) -> Chunk {
    Chunk {
        chunk_id: id.into(),
        source: Source::Markdown,
        project: Some("test".into()),
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

/// Seed a corpus with two chunks both aligned to `scope`, a ThreadsDb whose
/// ledger marks "stale" as lens-included `minutes_ago`, and return the pieces
/// the loop needs. The scope vector points at axis 0 so both chunks score
/// max attention affinity.
async fn fixture(minutes_ago: i64) -> (TempDir, CorpusStore, Arc<dyn ChainLogReader>, Vec<f32>) {
    let tmp = TempDir::new().unwrap();
    let corpus = CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap();
    let mut scope = vec![0.0_f32; DIM];
    scope[0] = 1.0;
    let chunks = vec![
        chunk("stale", "stale body alpha bravo charlie"),
        chunk("fresh", "fresh body delta echo foxtrot"),
    ];
    let embeddings = vec![scope.clone(), scope.clone()];
    corpus.upsert(&chunks, &embeddings).await.unwrap();

    // ThreadsDb::open installs a NoopChainSink, so write the event through a
    // real SqliteChainSink (same threads.sqlite). The ThreadsDb reader opens a
    // separate WAL connection and sees the committed row.
    let sink = SqliteChainSink::open(tmp.path()).unwrap();
    sink.append(&ChainEvent::LensIncluded {
        chunk_id: "stale".into(),
        slot: "attention".into(),
        ts: Utc::now() - Duration::minutes(minutes_ago),
    })
    .unwrap();
    let reader: Arc<dyn ChainLogReader> = Arc::new(ThreadsDb::open(tmp.path()).unwrap());
    (tmp, corpus, reader, scope)
}

fn snapshot(scope: &[f32]) -> LensTickSnapshot {
    LensTickSnapshot {
        // First valid rolling sample (last_state default has None) → drift
        // fires and the loop attempts a refresh.
        rolling_vec: Some(scope.to_vec()),
        scope_vector: Some(scope.to_vec()),
        pin_fingerprint: None,
    }
}

#[tokio::test]
async fn lens_engine_wires_freshness_and_refractory_rows() {
    let (_tmp, corpus, reader, scope) = fixture(2).await;
    // Probe: confirm the ledger reader actually sees the seeded event before
    // we rely on it through build_lens.
    let probe = reader
        .access_history(
            &["stale".to_string(), "fresh".to_string()],
            Utc::now() - Duration::days(1),
        )
        .unwrap();
    assert!(
        probe.get("stale").is_some_and(|v| !v.is_empty()),
        "ledger reader must see the seeded LensIncluded event for 'stale'; got {probe:?}"
    );
    // The real Lens-profile engine: attention_affinity + freshness.
    let engine = build_engine_from_weights(&default_profile_weights(RankProfile::Lens));
    let config = LensConfig::default();

    let decision = try_refresh_lens(
        &snapshot(&scope),
        &LensState::default(),
        &engine,
        Some(reader),
        &corpus,
        &config,
    )
    .await;

    let lens = expect_refresh(decision);
    assert!(
        !lens.entries.is_empty(),
        "lens surfaced attention-aligned chunks"
    );
    // Freshness feature registered in the Lens engine ⇒ every entry carries a
    // freshness attribution row. Proves the engine wiring is live.
    assert!(
        lens.entries
            .iter()
            .all(|e| e.feature_breakdown.contains_key("freshness")),
        "freshness feature is wired into the live lens engine"
    );
    // The recently lens-included chunk carries a refractory row read from the
    // real ledger. Proves the chain_reader → refractory wiring is live.
    let stale = lens
        .entries
        .iter()
        .find(|e| e.chunk_id == "stale")
        .expect("stale chunk present in portfolio");
    let refr = stale
        .feature_breakdown
        .get("refractory")
        .expect("recently-included chunk carries a refractory row");
    assert!(
        refr.contribution < 0.0,
        "refractory contribution is negative"
    );
}

#[tokio::test]
async fn refractory_demotes_recently_included_chunk_end_to_end() {
    let (_tmp, corpus, reader, scope) = fixture(2).await;
    // Attention-only engine isolates the refractory penalty (no freshness
    // boost, which would partly offset it — the two are coupled by design).
    let mut weights = BTreeMap::new();
    weights.insert("attention_affinity".to_string(), 1.0);
    let engine = build_engine_from_weights(&weights);
    let config = LensConfig::default();

    let decision = try_refresh_lens(
        &snapshot(&scope),
        &LensState::default(),
        &engine,
        Some(reader),
        &corpus,
        &config,
    )
    .await;

    let lens = expect_refresh(decision);
    assert_eq!(
        lens.entries.len(),
        2,
        "both aligned chunks in the portfolio"
    );
    let stale = lens.entries.iter().find(|e| e.chunk_id == "stale").unwrap();
    let fresh = lens.entries.iter().find(|e| e.chunk_id == "fresh").unwrap();
    assert!(
        stale.feature_breakdown.contains_key("refractory"),
        "stale chunk is penalized"
    );
    assert!(
        !fresh.feature_breakdown.contains_key("refractory"),
        "never-included chunk is not penalized"
    );
    assert!(
        stale.total_score < fresh.total_score,
        "refractory demotes the recently-included chunk: stale {} < fresh {}",
        stale.total_score,
        fresh.total_score
    );
    assert_eq!(
        lens.entries[0].chunk_id, "fresh",
        "the fresh chunk ranks first after refractory"
    );
}
