//! End-to-end tests for the P11 bulk/live boundary and the →007 serve
//! auto-weave.
//!
//! Both build on an identical fixture: a markdown corpus that (a) mentions a
//! pre-seeded thread handle and (b) contains a chunk whose `FakeEmbedder`
//! output collides with a pre-seeded anchor's vector (cosine ≈ 1.0 → above the
//! Markdown weaver threshold of 0.82, guaranteeing one Derived evidence link).
//!
//! - `bulk_scan_is_inert_and_explicit_weave_writes_evidence` guards the P11a
//!   gate: a one-shot `commands::scan` (ctx=None) is `IngestOrigin::Bulk`, not
//!   a TurnEnd, so the ambient daemons leave the ledger untouched; only the
//!   explicit `commands::weave` pass writes evidence.
//! - `serve_scan_auto_weaves_bulk_evidence` guards →007: the long-lived serve
//!   daemon (ctx=Some) runs an end-of-scan epoch weave, so bulk content is
//!   bound into the thread graph WITHOUT a manual `weave`. Serve has no
//!   operator to run one, so it must self-maintain.
//!
//! Familiarity (`mentions`) stays 0 throughout either path: it advances only on
//! live watched conversation-transcript TurnEnds (observer unit tests), never
//! on a bulk scan or a weave.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use ostk_recall_attention::{ConceptGrowthRuntime, InMemoryAttention};
use ostk_recall_cli::commands;
use ostk_recall_core::{Chunk, Links, PrivacyTier, Source, ThreadHandle};
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::{CorpusStore, TensionState, ThreadRecord, ThreadsDb};
use tempfile::TempDir;

const FAKE_DIM: usize = 16;

/// Deterministic embedder. The seed is `(text.len() % 100) * 0.01`, so two
/// texts of equal length produce identical vectors — the tests use that to
/// engineer a guaranteed cosine = 1.0 between the pre-seeded anchor chunk and
/// the resonant markdown fixture.
struct FakeEmbedder;

impl ChunkEmbedder for FakeEmbedder {
    fn dim(&self) -> usize {
        FAKE_DIM
    }
    #[allow(clippy::cast_precision_loss)]
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let seed = ((t.len() % 100) as f32) * 0.01;
                (0..FAKE_DIM)
                    .map(|i| (i as f32).mul_add(0.001, seed))
                    .collect()
            })
            .collect()
    }
}

fn write_config(path: &Path, corpus_root: &Path, fixture_dir: &Path) {
    let body = format!(
        r#"[corpus]
root = '{corpus}'

[embedder]
model = "unused-in-tests"

[[sources]]
kind = "markdown"
project = "notes"
paths = ['{fixture}']
"#,
        corpus = corpus_root.display(),
        fixture = fixture_dir.display(),
    );
    std::fs::write(path, body).unwrap();
}

/// Shared fixture: writes `resonant.md` (vector-collides with the anchor) and
/// `mention.md` (names the handle), the config, then pre-seeds the corpus with
/// the anchor chunk and an Active thread anchored to it. Returns the seeded
/// handle and the embedder.
async fn seed_resonant_corpus(
    corpus: &Path,
    fixture: &Path,
    config_path: &Path,
) -> (ThreadHandle, Arc<dyn ChunkEmbedder>) {
    let resonant_body = "Resonant anchor text body for the e2e fixture content.";
    let mention_body = "We keep circling three-time-scales as the same shape under three names.\n";
    std::fs::write(fixture.join("resonant.md"), resonant_body).unwrap();
    std::fs::write(fixture.join("mention.md"), mention_body).unwrap();
    write_config(config_path, corpus, fixture);

    let handle = ThreadHandle::new("three-time-scales").unwrap();
    let anchor_source_id = "anchor-fixture.md";
    let anchor_chunk_id = Chunk::make_id(Source::Markdown, anchor_source_id, 0, "");

    // Pre-seed the corpus with the anchor chunk (weaver reads CorpusStore).
    {
        let store = CorpusStore::open_or_create(corpus, FAKE_DIM).await.unwrap();
        let anchor_chunk = Chunk {
            chunk_id: anchor_chunk_id.clone(),
            source: Source::Markdown,
            project: Some("notes".into()),
            source_id: anchor_source_id.into(),
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            source_config_id: "test-cfg".to_string(),
            chunk_index: 0,
            ts: Some(Utc::now()),
            role: None,
            text: resonant_body.to_string(),
            sha256: Chunk::content_hash(resonant_body),
            links: Links::default(),
            extra: serde_json::Value::Null,
        };
        let vec = FakeEmbedder.encode_batch(&[resonant_body]).remove(0);
        store.upsert(&[anchor_chunk], &[vec]).await.unwrap();
    }

    // Pre-seed the ledger with an Active thread anchored to that chunk.
    {
        let db = ThreadsDb::open(corpus).unwrap();
        let now = Utc::now();
        db.upsert_thread(&ThreadRecord {
            handle: handle.clone(),
            tension: TensionState::Active,
            mentions: 0,
            resonance: 0,
            last_touched_at: now,
            anchor_chunk_id: Some(anchor_chunk_id.clone()),
            fold_override: None,
            created_at: now,
            created_scope_key: None,
            privacy_tier: PrivacyTier::T1Project,
        })
        .unwrap();
    }

    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
    (handle, embedder)
}

#[tokio::test]
async fn bulk_scan_is_inert_and_explicit_weave_writes_evidence() {
    let corpus = TempDir::new().unwrap();
    let fixture = TempDir::new().unwrap();
    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("config.toml");
    let (handle, embedder) =
        seed_resonant_corpus(corpus.path(), fixture.path(), &config_path).await;

    // Production one-shot scan entry point (ctx=None). A bulk scan emits only
    // `IngestOrigin::Bulk` events — none TurnEnds — so observer and weaver must
    // skip every one and leave the ledger untouched.
    let outcome = commands::scan(&config_path, Arc::clone(&embedder), None, false)
        .await
        .expect("scan");
    assert!(
        outcome.totals.chunks_upserted >= 2,
        "expected scan to ingest at least 2 markdown chunks; got {:?}",
        outcome.totals
    );

    // Gate: a bulk one-shot scan must NOT trigger live cognition.
    {
        let db = ThreadsDb::open(corpus.path()).unwrap();
        let after = db
            .get_thread(&handle)
            .unwrap()
            .expect("seeded thread missing after scan");
        let evidence = db.list_evidence(&handle).unwrap();
        assert_eq!(
            after.mentions, 0,
            "P11a gate regressed: bulk scan advanced mentions to {} \
             (observer fired on a non-TurnEnd)",
            after.mentions
        );
        assert!(
            evidence.is_empty(),
            "P11a gate regressed: bulk scan wrote {} evidence link(s) \
             (weaver fired on a non-TurnEnd)",
            evidence.len()
        );
    }

    // Explicit weave: the bulk-content coverage path (P11b).
    let weave_out = commands::weave(&config_path, Arc::clone(&embedder), None, 1)
        .await
        .expect("weave");
    let db = ThreadsDb::open(corpus.path()).unwrap();
    let after = db
        .get_thread(&handle)
        .unwrap()
        .expect("seeded thread missing after weave");
    let evidence = db.list_evidence(&handle).unwrap();
    assert!(
        !evidence.is_empty(),
        "explicit weave wrote no evidence link from resonant.md \
         (weave summary: {weave_out:?}); expected >= 1 against the \
         pre-seeded anchor",
    );
    assert_eq!(
        after.mentions, 0,
        "weave must not advance mentions (observer-only); got {}",
        after.mentions
    );
}

#[tokio::test]
async fn serve_scan_auto_weaves_bulk_evidence() {
    // →007: the serve daemon (ctx=Some) runs an end-of-scan epoch weave, so
    // bulk-ingested content is bound into the thread graph WITHOUT a manual
    // `weave` — unlike the inert one-shot path above. Guards the `ctx.is_some()`
    // end-of-scan weave block in `scan_with_context`.
    let corpus = TempDir::new().unwrap();
    let fixture = TempDir::new().unwrap();
    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("config.toml");
    let (handle, embedder) =
        seed_resonant_corpus(corpus.path(), fixture.path(), &config_path).await;

    // A minimal serve context: shares the ledger the end-of-scan weave binds to.
    let serve_ctx = commands::ServeContext {
        threads: Arc::new(ThreadsDb::open(corpus.path()).unwrap()),
        attention: Arc::new(InMemoryAttention::new()),
        concept_growth: ConceptGrowthRuntime::default(),
    };
    commands::scan_with_context(
        &config_path,
        Arc::clone(&embedder),
        None,
        false,
        Some(&serve_ctx),
    )
    .await
    .expect("serve scan");

    // The end-of-scan epoch weave must have bound resonant.md to the anchor —
    // no explicit `weave` was run.
    let db = ThreadsDb::open(corpus.path()).unwrap();
    let evidence = db.list_evidence(&handle).unwrap();
    assert!(
        !evidence.is_empty(),
        "→007 regressed: serve-path end-of-scan weave wrote no evidence link \
         (the ctx.is_some() weave block in scan_with_context is missing/broken)"
    );
    let after = db
        .get_thread(&handle)
        .unwrap()
        .expect("seeded thread missing after serve scan");
    assert_eq!(
        after.mentions, 0,
        "auto-weave must not advance mentions (observer-only); got {}",
        after.mentions
    );
}

#[tokio::test]
async fn serve_path_scan_auto_weaves_bulk_evidence() {
    // →007 regression guard for the PER-PATH watch trigger. serve's watcher
    // routes changed files through `scan_paths_with_context` (not
    // `scan_with_context`), so it must run the same end-of-scan weave. This is
    // the exact path that shipped un-woven until the shared
    // `run_end_of_scan_weave` helper covered both — caught by live validation.
    let corpus = TempDir::new().unwrap();
    let fixture = TempDir::new().unwrap();
    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("config.toml");
    let (handle, embedder) =
        seed_resonant_corpus(corpus.path(), fixture.path(), &config_path).await;

    let serve_ctx = commands::ServeContext {
        threads: Arc::new(ThreadsDb::open(corpus.path()).unwrap()),
        attention: Arc::new(InMemoryAttention::new()),
        concept_growth: ConceptGrowthRuntime::default(),
    };
    // The watcher passes the specific changed paths, not a full source scan.
    let changed: Vec<PathBuf> = vec![fixture.path().join("resonant.md")];
    commands::scan_paths_with_context(
        &config_path,
        Arc::clone(&embedder),
        &changed,
        false,
        Some(&serve_ctx),
    )
    .await
    .expect("serve per-path scan");

    let db = ThreadsDb::open(corpus.path()).unwrap();
    let evidence = db.list_evidence(&handle).unwrap();
    assert!(
        !evidence.is_empty(),
        "→007 regressed on the per-path watch trigger: scan_paths_with_context \
         wrote no evidence link (run_end_of_scan_weave not wired there?)"
    );
}
