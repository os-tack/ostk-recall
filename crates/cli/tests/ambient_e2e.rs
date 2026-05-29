//! End-to-end test for the P11 bulk/live boundary.
//!
//! Drives the production `commands::scan` entry point with a markdown
//! corpus that (a) mentions a pre-seeded thread handle and (b) contains
//! a chunk whose `FakeEmbedder` output collides with a pre-seeded
//! anchor's vector. A bulk scan is `IngestOrigin::Bulk`, which is NOT a
//! TurnEnd, so the ambient daemons (`TurnObserver` + `AutoWeaver`) must
//! leave the ledger untouched — this guards the P11a gate against
//! regressing back to "library load == live cognition" (the bug that
//! flooded the corpus with one membrane commit per historical chunk).
//!
//! Weaving bulk content is instead the job of the explicit
//! `commands::weave` pass (P11b / `weave_window`), so after `weave`
//! returns we expect the resonant chunk to have produced at least one
//! derived evidence link. Familiarity stays 0 throughout: it advances
//! only on live watched conversation-transcript TurnEnds (covered by the
//! observer unit tests), never on a bulk scan or an offline weave.

use std::path::Path;
use std::sync::Arc;

use chrono::Utc;
use ostk_recall_cli::commands;
use ostk_recall_core::{Chunk, Links, PrivacyTier, Source, ThreadHandle};
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::{CorpusStore, TensionState, ThreadRecord, ThreadsDb};
use tempfile::TempDir;

const FAKE_DIM: usize = 16;

/// Deterministic embedder. The seed is `(text.len() % 100) * 0.01`, so
/// two texts of equal length produce identical vectors — the test uses
/// that property to engineer a guaranteed cosine = 1.0 between the
/// pre-seeded anchor chunk and the resonant markdown fixture.
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

#[tokio::test]
async fn bulk_scan_is_inert_and_explicit_weave_writes_evidence() {
    let corpus = TempDir::new().unwrap();
    let fixture = TempDir::new().unwrap();
    let config_dir = TempDir::new().unwrap();
    let config_path = config_dir.path().join("config.toml");

    // Resonant body: this exact text will appear in both the pre-seeded
    // anchor chunk and the `resonant.md` markdown fixture. Since the
    // `FakeEmbedder` keys off `text.len() % 100`, equal-length texts
    // produce identical vectors → cosine ≈ 1.0 → above the Markdown
    // weaver threshold of 0.82, guaranteeing one Derived evidence link.
    let resonant_body = "Resonant anchor text body for the e2e fixture content.";

    // Mention fixture: contains the known handle as a kebab-case token,
    // surrounded by non-handle characters so the observer's word-boundary
    // check fires.
    let mention_body = "We keep circling three-time-scales as the same shape under three names.\n";

    std::fs::write(fixture.path().join("resonant.md"), resonant_body).unwrap();
    std::fs::write(fixture.path().join("mention.md"), mention_body).unwrap();

    write_config(&config_path, corpus.path(), fixture.path());

    let handle = ThreadHandle::new("three-time-scales").unwrap();
    let anchor_source_id = "anchor-fixture.md";
    let anchor_chunk_id = Chunk::make_id(Source::Markdown, anchor_source_id, 0, "");

    // --- pre-seed the corpus with the anchor chunk -------------------
    // commands::scan does CorpusStore::open_or_create with the same dim,
    // so opening it here first is fine — the second open_or_create is a
    // no-op migration. We don't need IngestDb at this stage because the
    // weaver only reads from CorpusStore.
    {
        let store = CorpusStore::open_or_create(corpus.path(), FAKE_DIM)
            .await
            .unwrap();
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
        let embedder = FakeEmbedder;
        let vec = embedder.encode_batch(&[resonant_body]).remove(0);
        store.upsert(&[anchor_chunk], &[vec]).await.unwrap();
    }

    // --- pre-seed the ledger ----------------------------------------
    {
        let db = ThreadsDb::open(corpus.path()).unwrap();
        let now = Utc::now();
        db.upsert_thread(&ThreadRecord {
            handle: handle.clone(),
            tension: TensionState::Active,
            familiarity: 0,
            last_touched_at: now,
            anchor_chunk_id: Some(anchor_chunk_id.clone()),
            fold_override: None,
            created_at: now,
            created_scope_key: None,
            privacy_tier: PrivacyTier::T1Project,
        })
        .unwrap();
    }

    // --- production scan entry point: spawns ambient daemons --------
    // The daemons are spawned, but a bulk scan emits only
    // `IngestOrigin::Bulk` events — none of them TurnEnds — so the
    // observer and weaver must skip every one and leave the ledger
    // untouched.
    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
    let outcome = commands::scan(&config_path, Arc::clone(&embedder), None, false)
        .await
        .expect("scan");
    assert!(
        outcome.totals.chunks_upserted >= 2,
        "expected scan to ingest at least 2 markdown chunks; got {:?}",
        outcome.totals
    );

    // --- gate assertion: a bulk scan must NOT trigger live cognition.
    // resonant.md resonates with the seeded anchor and mention.md names
    // the known handle, yet because the scan is Bulk (not a watched
    // conversation TurnEnd) neither daemon may fire. Regressing this is
    // the version-explosion bug returning.
    {
        let db = ThreadsDb::open(corpus.path()).unwrap();
        let after = db
            .get_thread(&handle)
            .unwrap()
            .expect("seeded thread missing after scan");
        let evidence = db.list_evidence(&handle).unwrap();
        assert_eq!(
            after.familiarity, 0,
            "P11a gate regressed: bulk scan advanced familiarity to {} \
             (observer fired on a non-TurnEnd)",
            after.familiarity
        );
        assert!(
            evidence.is_empty(),
            "P11a gate regressed: bulk scan wrote {} evidence link(s) \
             (weaver fired on a non-TurnEnd)",
            evidence.len()
        );
    }

    // --- explicit weave: the bulk-content coverage path (P11b) -------
    // `commands::weave` runs `weave_window(None)` over the whole corpus.
    // This is how bulk-ingested content is woven into the thread graph
    // now that the live daemon deliberately ignores it.
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
    // Familiarity is the observer's responsibility and advances only on
    // live watched TurnEnds — an offline weave must never touch it.
    assert_eq!(
        after.familiarity, 0,
        "weave must not advance familiarity (observer-only); got {}",
        after.familiarity
    );
}
