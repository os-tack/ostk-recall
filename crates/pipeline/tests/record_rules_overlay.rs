//! P12 integration gates: the config-driven record-rule overlay applied in
//! the pipeline ingest path.
//!
//! Covers the wiring the core `record_rules` unit tests can't reach:
//! - a `tag` rule flips `record_kind` (allowlisted) → Tier-2 re-embeds;
//! - a `drop` rule purges a previously-ingested chunk (explicit purge, not
//!   orphan sweep);
//! - editing a rule flips the per-source-kind Tier-1 digest → re-parse even
//!   when the file is unchanged;
//! - bumping a scanner's `parse_version` likewise forces re-parse;
//! - the watch path (`scan_paths`) runs the same overlay.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ostk_recall_core::error::Result as CoreResult;
use ostk_recall_core::{
    Chunk, CompiledRecordRules, Config, RecordRule, RuleAction, RuleMatch, Scanner, Source,
    SourceConfig, SourceItem, SourceKind, default_record_rules,
};
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline};
use ostk_recall_store::{CorpusStore, IngestDb};
use tempfile::TempDir;

const DIM: usize = 16;

struct CountingEmbedder {
    calls: AtomicUsize,
}
impl CountingEmbedder {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
    fn count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}
impl ChunkEmbedder for CountingEmbedder {
    fn dim(&self) -> usize {
        DIM
    }
    #[allow(clippy::cast_precision_loss)]
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        self.calls.fetch_add(texts.len(), Ordering::Relaxed);
        texts
            .iter()
            .map(|t| {
                let seed = ((t.len() % 100) as f32) * 0.01;
                (0..DIM).map(|i| (i as f32).mul_add(0.001, seed)).collect()
            })
            .collect()
    }
}

async fn make_pipeline(
    corpus_root: &Path,
    rules: &[RecordRule],
) -> (Pipeline, Arc<CountingEmbedder>, Arc<CorpusStore>) {
    let store = Arc::new(CorpusStore::open_or_create(corpus_root, DIM).await.unwrap());
    let ingest = Arc::new(IngestDb::open(corpus_root).unwrap());
    let counter = Arc::new(CountingEmbedder::new());
    let emb: Arc<dyn ChunkEmbedder> = counter.clone();
    let compiled = Arc::new(CompiledRecordRules::build(rules).expect("rules compile"));
    let pipeline = Pipeline::new(Arc::clone(&store), ingest, emb).with_record_rules(compiled);
    (pipeline, counter, store)
}

fn markdown_cfg(fixtures_dir: &Path) -> SourceConfig {
    let mut cfg = Config {
        corpus: ostk_recall_core::CorpusConfig {
            root: fixtures_dir.to_string_lossy().into_owned(),
        },
        embedder: ostk_recall_core::EmbedderConfig {
            model: "fake".into(),
        },
        sources: vec![SourceConfig {
            kind: SourceKind::Markdown,
            project: Some("notes".into()),
            paths: vec![fixtures_dir.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            entity_type: None,
            edges: Vec::new(),
            id: None,
            source_config_id: String::new(),
            facets: std::collections::BTreeMap::new(),
        }],
        reranker: None,
        watch: None,
        runtime: None,
        ranking: None,
        relational: None,
        lens: None,
        record_rules: None,
        weaver: None,
    };
    cfg.validate_and_seal().expect("seal");
    cfg.sources.remove(0)
}

fn tag_rule(prefix: &str, record_kind: &str) -> RecordRule {
    RecordRule {
        r#match: RuleMatch {
            prefix: Some(prefix.to_string()),
            source_kind: Some(vec!["markdown".to_string()]),
            ..Default::default()
        },
        action: RuleAction::Tag {
            record_kind: record_kind.to_string(),
        },
    }
}

fn drop_rule(prefix: &str) -> RecordRule {
    RecordRule {
        r#match: RuleMatch {
            prefix: Some(prefix.to_string()),
            source_kind: Some(vec!["markdown".to_string()]),
            ..Default::default()
        },
        action: RuleAction::Drop,
    }
}

#[tokio::test]
async fn tag_rule_sets_record_kind_and_reembeds() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    std::fs::write(fixtures.path().join("a.md"), "# Heading\n\nbody text\n").unwrap();
    let scanner = ostk_recall_scan::markdown::MarkdownScanner;
    let cfg = markdown_cfg(fixtures.path());

    // Run 1: no rules.
    let (p1, c1, _) = make_pipeline(corpus.path(), &[]).await;
    let s1 = p1.ingest_source(&scanner, &cfg).await;
    assert!(s1.chunks_upserted > 0 && c1.count() > 0, "baseline embed");

    // Run 2 (same corpus): a tag rule stamps record_kind (allowlisted) →
    // embedding_input_sha256 changes → Tier-2 re-embeds.
    let (p2, c2, store) = make_pipeline(corpus.path(), &[tag_rule("# Heading", "test_kind")]).await;
    let s2 = p2.ingest_source(&scanner, &cfg).await;
    assert!(
        c2.count() > 0,
        "tag (record_kind) must re-embed; stats={s2:?}"
    );

    // And the stored chunk carries the facet.
    let n = store
        .count_active("array_contains(facets, 'record_kind:test_kind')")
        .await
        .unwrap();
    assert_eq!(n, 1, "the tagged chunk must carry record_kind:test_kind");
}

#[tokio::test]
async fn drop_rule_purges_previously_ingested_chunk() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    std::fs::write(fixtures.path().join("a.md"), "<apparatus> machine noise\n").unwrap();
    let scanner = ostk_recall_scan::markdown::MarkdownScanner;
    let cfg = markdown_cfg(fixtures.path());

    // Run 1: no rules — the chunk is ingested.
    let (p1, _, store1) = make_pipeline(corpus.path(), &[]).await;
    let s1 = p1.ingest_source(&scanner, &cfg).await;
    assert_eq!(s1.chunks_upserted, 1, "baseline ingest of one chunk");
    let before = store1.count_active("true").await.unwrap();
    assert_eq!(before, 1);

    // Run 2 (same corpus): a drop rule now matches. The rules-digest flip
    // forces Tier-1 re-parse; the chunk is dropped and the previously-ingested
    // row is explicitly purged (markdown is Stale retention — orphan sweep
    // would NOT remove it; the explicit purge does).
    let (p2, _, store2) = make_pipeline(corpus.path(), &[drop_rule("<apparatus>")]).await;
    let s2 = p2.ingest_source(&scanner, &cfg).await;
    assert_eq!(s2.chunks_dropped_by_rule, 1, "rule should drop the chunk");
    assert_eq!(
        s2.chunks_purged, 1,
        "previously-ingested row must be purged"
    );
    let after = store2.count_active("true").await.unwrap();
    assert_eq!(after, 0, "corpus must no longer contain the dropped chunk");
}

#[tokio::test]
async fn editing_a_record_rule_forces_reparse() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    std::fs::write(fixtures.path().join("a.md"), "# Heading\n\nbody\n").unwrap();
    let scanner = ostk_recall_scan::markdown::MarkdownScanner;
    let cfg = markdown_cfg(fixtures.path());

    // Run 1: tag with v1.
    let (p1, c1, _) = make_pipeline(corpus.path(), &[tag_rule("# Heading", "v1")]).await;
    p1.ingest_source(&scanner, &cfg).await;
    assert!(c1.count() > 0);

    // Run 2: SAME file (mtime/size/facets unchanged), but the rule's tag value
    // changed → per-source-kind digest flips → Tier-1 re-parses → re-embed.
    let (p2, c2, _) = make_pipeline(corpus.path(), &[tag_rule("# Heading", "v2")]).await;
    p2.ingest_source(&scanner, &cfg).await;
    assert!(
        c2.count() > 0,
        "editing a record rule must flip the Tier-1 digest and re-parse"
    );
}

#[tokio::test]
async fn unaffected_source_kind_does_not_reparse_under_default_rules() {
    // Regression guard: enabling P12 with the DEFAULT ruleset (scoped to
    // claude_code/ostk_project) must NOT force a markdown source to re-parse —
    // its Tier-1 hash has to stay byte-identical to the no-rules value, or the
    // whole corpus re-parses on upgrade (the slow-scan bug).
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    std::fs::write(fixtures.path().join("a.md"), "# H\n\nbody\n").unwrap();
    let scanner = ostk_recall_scan::markdown::MarkdownScanner;
    let cfg = markdown_cfg(fixtures.path());

    // Run 1: no rules at all.
    let (p1, c1, _) = make_pipeline(corpus.path(), &[]).await;
    p1.ingest_source(&scanner, &cfg).await;
    assert!(c1.count() > 0, "baseline embed");

    // Run 2: the DEFAULT ruleset is now active, but none of it targets
    // markdown → digest_for(Markdown) == "" → extra digest empty → same hash →
    // Tier-1 skip, no re-embed.
    let (p2, c2, _) = make_pipeline(corpus.path(), &default_record_rules()).await;
    let s2 = p2.ingest_source(&scanner, &cfg).await;
    assert_eq!(
        c2.count(),
        0,
        "markdown must Tier-1-skip under the default rules (no re-embed); stats={s2:?}"
    );
    assert_eq!(s2.items_skipped, 1, "the file should be Tier-1-skipped");
}

#[tokio::test]
async fn scan_paths_applies_record_rules() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    let file = fixtures.path().join("a.md");
    std::fs::write(&file, "<apparatus> noise\n").unwrap();
    let scanner = ostk_recall_scan::markdown::MarkdownScanner;
    let cfg = markdown_cfg(fixtures.path());

    // The watch/incremental path must run the same overlay helper.
    let (p, _, store) = make_pipeline(corpus.path(), &[drop_rule("<apparatus>")]).await;
    let out = p
        .scan_paths(&[(&scanner, &cfg)], &[file.clone()])
        .await
        .unwrap();
    let total: usize = out.iter().map(|(_, s)| s.chunks_dropped_by_rule).sum();
    assert_eq!(total, 1, "scan_paths must drop via the record-rule overlay");
    assert_eq!(
        store.count_active("true").await.unwrap(),
        0,
        "scan_paths drop must not ingest the apparatus chunk"
    );
}

// --- parse_version backstop ----------------------------------------------

/// Minimal scanner emitting one fixed chunk, with a configurable
/// `parse_version` so we can prove a bump forces re-parse via the Tier-1
/// digest fold.
struct FixedScanner {
    parse_version: u32,
    text: String,
    root: std::path::PathBuf,
}

impl Scanner for FixedScanner {
    fn kind(&self) -> SourceKind {
        SourceKind::Markdown
    }
    fn parse_version(&self) -> u32 {
        self.parse_version
    }
    fn discover<'a>(
        &'a self,
        _cfg: &'a SourceConfig,
    ) -> Box<dyn Iterator<Item = CoreResult<SourceItem>> + 'a> {
        Box::new(std::iter::once(Ok(SourceItem {
            source_id: "fixed.md".to_string(),
            path: Some(self.root.join("fixed.md")),
            ..Default::default()
        })))
    }
    fn parse(&self, item: SourceItem) -> CoreResult<Vec<Chunk>> {
        let mut c = Chunk {
            chunk_id: String::new(),
            source: Source::Markdown,
            project: None,
            source_id: item.source_id.clone(),
            source_config_id: item.source_config_id.clone(),
            chunk_index: 0,
            ts: None,
            role: None,
            text: self.text.clone(),
            sha256: Chunk::content_hash(&self.text),
            links: Default::default(),
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            extra: serde_json::Value::Null,
        };
        c.chunk_id = Chunk::make_id(Source::Markdown, &c.source_id, 0, &c.source_config_id);
        Ok(vec![c])
    }
}

#[tokio::test]
async fn bumping_parse_version_forces_reparse() {
    let fixtures = TempDir::new().unwrap();
    let corpus = TempDir::new().unwrap();
    // Real file so Tier-1 metadata (mtime/size) is stable across runs.
    std::fs::write(fixtures.path().join("fixed.md"), "content\n").unwrap();
    let cfg = markdown_cfg(fixtures.path());

    // Run 1: parse_version = 1.
    let (p1, c1, _) = make_pipeline(corpus.path(), &[]).await;
    let s1 = FixedScanner {
        parse_version: 1,
        text: "hello world".to_string(),
        root: fixtures.path().to_path_buf(),
    };
    p1.ingest_source(&s1, &cfg).await;
    assert!(c1.count() > 0, "baseline embed");

    // Run 2: same file + same emitted chunk, but parse_version = 2 → Tier-1
    // digest flips → re-parse. (Text identical, so Tier-2 would normally skip;
    // the point is that parse() ran again, proving Tier-1 didn't short-circuit.)
    let (p2, _c2, _) = make_pipeline(corpus.path(), &[]).await;
    let s2 = FixedScanner {
        parse_version: 2,
        text: "hello world".to_string(),
        root: fixtures.path().to_path_buf(),
    };
    let stats = p2.ingest_source(&s2, &cfg).await;
    assert_eq!(
        stats.items_skipped, 0,
        "parse_version bump must defeat the Tier-1 skip (items_skipped stays 0)"
    );
}
