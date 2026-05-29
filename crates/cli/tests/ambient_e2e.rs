//! End-to-end test for the ambient-pickup path.
//!
//! Drives the production `commands::scan` entry point with a markdown
//! corpus that (a) mentions a pre-seeded thread handle and (b) contains
//! a chunk whose `FakeEmbedder` output collides with a pre-seeded
//! anchor's vector. The wiring in `commands::scan` spawns the
//! `TurnObserver` + `AutoWeaver` daemons against the same `Pipeline`,
//! so after the scan returns we expect the threads ledger to show
//! advanced familiarity AND at least one derived evidence link.

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
async fn ambient_pickup_advances_familiarity_and_writes_evidence() {
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
    let embedder: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
    let outcome = commands::scan(&config_path, Arc::clone(&embedder), None, false)
        .await
        .expect("scan");
    assert!(
        outcome.totals.chunks_upserted >= 2,
        "expected scan to ingest at least 2 markdown chunks; got {:?}",
        outcome.totals
    );

    // --- assertions: daemons should have mutated the ledger ---------
    let db = ThreadsDb::open(corpus.path()).unwrap();
    let after = db
        .get_thread(&handle)
        .unwrap()
        .expect("seeded thread missing after scan");
    let evidence = db.list_evidence(&handle).unwrap();

    // Aggregate both signals before panicking — surfacing both at once
    // is more informative when the V1 wiring has a partial bug.
    let mut failures: Vec<String> = Vec::new();
    if after.familiarity < 1 {
        failures.push(format!(
            "familiarity advancement: expected >= 1 after observer saw \
             mention.md; got {}",
            after.familiarity
        ));
    }
    if evidence.is_empty() {
        failures.push(
            "evidence_links row from auto-weaver: expected >= 1 from \
             resonant.md; got 0"
                .into(),
        );
    }
    assert!(
        failures.is_empty(),
        "ambient pickup did not mutate the ledger after scan \
         (familiarity={}, evidence_count={}):\n  - {}",
        after.familiarity,
        evidence.len(),
        failures.join("\n  - "),
    );
}
