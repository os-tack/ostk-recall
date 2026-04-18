//! Subcommand implementations.
//!
//! Each public function here is callable from both `main.rs` (the binary
//! wraps these) and from integration tests (they inject a `FakeEmbedder`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use ostk_recall_core::{Config, RerankerConfig, Scanner, SourceKind};
use ostk_recall_mcp::Server;
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline, PipelineStats, VerifyReport};
use ostk_recall_query::{QueryEngine, Reranker, RerankerLike};
use ostk_recall_scan::claude_code::ClaudeCodeScanner;
use ostk_recall_scan::code::CodeScanner;
use ostk_recall_scan::file_glob::FileGlobScanner;
use ostk_recall_scan::markdown::MarkdownScanner;
use ostk_recall_scan::ostk_project::OstkProjectScanner;
use ostk_recall_scan::zip_export::ZipExportScanner;
use ostk_recall_store::{CorpusStore, EventsDb, IngestDb};

/// Starter config written when the user runs `init` and no config exists.
pub const STARTER_CONFIG: &str = r#"# ostk-recall configuration.
# Edit this file, then re-run `ostk-recall init` and `ostk-recall scan`.

[corpus]
root = "~/.local/share/ostk-recall"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
project = "notes"
paths = ["~/notes"]
"#;

/// Return the built-in starter config body.
#[must_use]
pub const fn starter_config() -> &'static str {
    STARTER_CONFIG
}

/// `${XDG_CONFIG_HOME:-~/.config}/ostk-recall/config.toml`.
pub fn default_config_path() -> Result<PathBuf> {
    let base = dirs::config_dir()
        .ok_or_else(|| anyhow!("could not determine config dir (no $XDG_CONFIG_HOME, no $HOME)"))?;
    Ok(base.join("ostk-recall").join("config.toml"))
}

#[derive(Debug, Clone)]
pub enum InitOutcome {
    /// A starter config was written; the user should edit and re-run.
    WroteStarter { path: PathBuf },
    /// Corpus was initialized at `root`; `model_id` + `dim` reflect the
    /// embedder.
    Initialized {
        root: PathBuf,
        model_id: String,
        dim: usize,
    },
}

#[derive(Debug, Clone)]
pub struct ScanOutcome {
    pub per_source: Vec<(String, PipelineStats)>,
    pub totals: PipelineStats,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct VerifyOutcome {
    pub report: VerifyReport,
}

/// Options accepted by [`init_with_options`].
///
/// Builder defaults match the pre-Phase H behaviour: `force = false`,
/// `prefetch_reranker = false`. Production callers (the binary) explicitly
/// turn the prefetch on.
#[derive(Debug, Clone, Copy, Default)]
pub struct InitOptions {
    pub force: bool,
    pub prefetch_reranker: bool,
}

impl InitOptions {
    #[must_use]
    pub const fn with_force(mut self, force: bool) -> Self {
        self.force = force;
        self
    }

    #[must_use]
    pub const fn with_prefetch_reranker(mut self, prefetch: bool) -> Self {
        self.prefetch_reranker = prefetch;
        self
    }
}

/// Initialize a corpus.
///
/// If `config_path` doesn't exist, write a starter and return
/// `WroteStarter` without touching the corpus. Otherwise: create the corpus
/// root, open (or create) the `LanceDB` table sized to the embedder dim,
/// and open the ingest DB.
///
/// `embedder` is passed in so tests can supply a fake — the binary wraps
/// `Embedder::load` around the real one.
///
/// Note: `init` does not prefetch the reranker model; the binary calls
/// [`init_with_options`] with `prefetch_reranker = true` for the production
/// path so the first `serve` doesn't pay the download latency.
pub async fn init(config_path: &Path, embedder: Arc<dyn ChunkEmbedder>) -> Result<InitOutcome> {
    init_with_options(config_path, embedder, InitOptions::default()).await
}

/// Like [`init`] but with explicit options (force-wipe + reranker prefetch).
///
/// When `opts.force` is true, deletes `corpus.lance/`, `ingest.duckdb`, and
/// `events.duckdb` from the resolved corpus root before opening the store.
/// Each removal is best-effort and guarded by `.exists()` — missing files
/// are not errors. A `forcing re-init` warning is printed (stderr) before
/// each deletion.
///
/// When `opts.prefetch_reranker` is true and the reranker is enabled in
/// config, the cross-encoder ONNX model is downloaded into the corpus's
/// `models/` cache. Failures are logged to stderr but do not error the init.
pub async fn init_with_options(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    opts: InitOptions,
) -> Result<InitOutcome> {
    if !config_path.exists() {
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating config dir {}", parent.display()))?;
        }
        std::fs::write(config_path, STARTER_CONFIG)
            .with_context(|| format!("writing starter config at {}", config_path.display()))?;
        return Ok(InitOutcome::WroteStarter {
            path: config_path.to_path_buf(),
        });
    }

    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    std::fs::create_dir_all(&root)
        .with_context(|| format!("creating corpus root {}", root.display()))?;

    if opts.force {
        force_wipe_corpus(&root)?;
    }

    let dim = embedder.dim();
    let store = CorpusStore::open_or_create(&root, dim)
        .await
        .map_err(|e| anyhow!("open corpus store: {e}"))?;
    store
        .ensure_fts_index()
        .await
        .map_err(|e| anyhow!("ensure fts index: {e}"))?;
    let _ingest = IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?;

    // Pre-download the reranker so the first `serve` doesn't pay the
    // download latency. Only runs when the caller opts in (the production
    // binary does; tests do not) and the user hasn't disabled rerank in
    // config. Failures are logged but do not fail init.
    if opts.prefetch_reranker {
        let reranker_cfg = RerankerConfig::resolve(cfg.reranker.as_ref());
        if reranker_cfg.enabled && !skip_reranker_for_tests() {
            if let Err(e) = prefetch_reranker_model(&root, &reranker_cfg.model) {
                eprintln!("warning: reranker prefetch failed ({e}); will retry on first serve");
            }
        }
    }

    Ok(InitOutcome::Initialized {
        root,
        model_id: cfg.embedder.model.clone(),
        dim,
    })
}

/// Test escape hatch. Returns true when we should skip every reranker
/// network operation: explicit `OSTK_RECALL_SKIP_RERANKER=1` for
/// subprocesses, or `OSTK_RECALL_FAKE_EMBEDDER=...` (which already marks
/// the run as a test that must not touch real models).
fn skip_reranker_for_tests() -> bool {
    std::env::var("OSTK_RECALL_SKIP_RERANKER").is_ok()
        || std::env::var("OSTK_RECALL_FAKE_EMBEDDER").is_ok()
}

/// Best-effort: load the reranker so the ONNX file lands in the corpus's
/// `models/` cache. Prints a "downloading reranker model..." line to
/// stderr so the user understands what's happening on a cold cache.
fn prefetch_reranker_model(root: &Path, model_spec: &str) -> Result<()> {
    let model = Reranker::resolve_model(model_spec).ok_or_else(|| {
        anyhow!("unknown reranker model {model_spec:?}; see config docs for accepted aliases")
    })?;
    let cache_dir = root.join("models");
    std::fs::create_dir_all(&cache_dir)?;
    eprintln!(
        "downloading reranker model {model_spec} into {}...",
        cache_dir.display()
    );
    let _ = Reranker::load_with_model(&cache_dir, model)
        .map_err(|e| anyhow!("load reranker {model_spec}: {e}"))?;
    Ok(())
}

/// Best-effort delete of corpus artifacts under `root`. Prints a warning to
/// stderr before each removal; absence is not an error.
fn force_wipe_corpus(root: &Path) -> Result<()> {
    let lance = root.join("corpus.lance");
    if lance.exists() {
        eprintln!("forcing re-init: removing {}", lance.display());
        std::fs::remove_dir_all(&lance).with_context(|| format!("removing {}", lance.display()))?;
    }
    for db_name in ["ingest.duckdb", "events.duckdb"] {
        let p = root.join(db_name);
        if p.exists() {
            eprintln!("forcing re-init: removing {}", p.display());
            std::fs::remove_file(&p).with_context(|| format!("removing {}", p.display()))?;
        }
    }
    Ok(())
}

/// Scan configured sources.
///
/// * `source_filter` — if `Some`, only process `SourceConfig`s whose project
///   matches the given string (case-sensitive). Phase B only wires the
///   `Markdown` kind; other kinds print "TODO phase C" and are skipped.
/// * `dry_run` — no side effects; pipeline still parses + dedupes and reports
///   counts.
pub async fn scan(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    source_filter: Option<&str>,
    dry_run: bool,
) -> Result<ScanOutcome> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    std::fs::create_dir_all(&root)?;

    let store = Arc::new(
        CorpusStore::open_or_create(&root, embedder.dim())
            .await
            .map_err(|e| anyhow!("open corpus store: {e}"))?,
    );
    let ingest = Arc::new(IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?);
    let pipeline = Pipeline::new(store, ingest, Arc::clone(&embedder)).with_dry_run(dry_run);

    // Scanners that are stateless can be reused across sources. The
    // code scanner now caches per-workspace `fcp-rust` sessions across
    // `parse` calls; one instance per ingest run lets that cache stick.
    let markdown = MarkdownScanner;
    let code = CodeScanner;
    let claude_code = ClaudeCodeScanner;
    let file_glob = FileGlobScanner;
    let zip_export = ZipExportScanner;

    // `OstkProjectScanner` optionally carries an `Arc<EventsDb>` — build
    // one only if at least one `ostk_project` source is configured.
    let events_db: Option<Arc<EventsDb>> = if cfg
        .sources
        .iter()
        .any(|s| matches!(s.kind, SourceKind::OstkProject))
    {
        Some(Arc::new(
            EventsDb::open(&root).map_err(|e| anyhow!("open events db: {e}"))?,
        ))
    } else {
        None
    };
    let ostk_project = events_db
        .as_ref()
        .map_or_else(OstkProjectScanner::new, |db| {
            OstkProjectScanner::new().with_events_db(Arc::clone(db))
        });

    let mut per_source = Vec::new();
    let mut totals = PipelineStats::default();
    for (i, source_cfg) in cfg.sources.iter().enumerate() {
        if let Some(filter) = source_filter {
            if source_cfg.project.as_deref() != Some(filter) {
                continue;
            }
        }

        let label = source_cfg
            .project
            .clone()
            .unwrap_or_else(|| format!("{}[{i}]", source_cfg.kind.as_str()));

        let scanner: &dyn Scanner = match source_cfg.kind {
            SourceKind::Markdown => &markdown,
            SourceKind::Code => &code,
            SourceKind::ClaudeCode => &claude_code,
            SourceKind::FileGlob => &file_glob,
            SourceKind::ZipExport => &zip_export,
            SourceKind::OstkProject => &ostk_project,
        };

        let stats = pipeline.ingest_source(scanner, source_cfg).await;
        totals = totals.merge(stats);
        per_source.push((label, stats));
    }

    // Close every cached fcp-rust subprocess so they don't linger
    // between ingests. Safe to call even when no `.rs` files were
    // scanned (the cache is simply empty).
    ostk_recall_scan::fcp_rust::drain_session_cache();

    Ok(ScanOutcome {
        per_source,
        totals,
        dry_run,
    })
}

/// Verify: compare corpus row count vs ingest chunk count. Phase B: totals
/// only (per-source comparison is deferred).
pub async fn verify(config_path: &Path, embedder: Arc<dyn ChunkEmbedder>) -> Result<VerifyOutcome> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;

    let store = Arc::new(
        CorpusStore::open_or_create(&root, embedder.dim())
            .await
            .map_err(|e| anyhow!("open corpus store: {e}"))?,
    );
    let ingest = Arc::new(IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?);
    let pipeline = Pipeline::new(store, ingest, embedder);
    let report = pipeline
        .verify_counts()
        .await
        .map_err(|e| anyhow!("verify_counts: {e}"))?;
    Ok(VerifyOutcome { report })
}

/// Build the query engine from config + embedder. Shared between `serve` and
/// integration tests so spawning the real MCP server stays a one-liner.
pub async fn build_query_engine(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
) -> Result<QueryEngine> {
    build_query_engine_with_reranker(config_path, embedder, true).await
}

/// Like [`build_query_engine`] but lets the caller force-disable the
/// reranker even if config says otherwise. Tests pass `attach_reranker
/// = false` so they don't pull ONNX.
pub async fn build_query_engine_with_reranker(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    attach_reranker: bool,
) -> Result<QueryEngine> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    let store = Arc::new(
        CorpusStore::open_or_create(&root, embedder.dim())
            .await
            .map_err(|e| anyhow!("open corpus store: {e}"))?,
    );
    let ingest = Arc::new(IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?);
    let events = events_db_if_wanted(&cfg, &root)?;
    let mut engine = QueryEngine::new(store, ingest, events, embedder, cfg.embedder.model.clone());
    if attach_reranker && !skip_reranker_for_tests() {
        let reranker_cfg = RerankerConfig::resolve(cfg.reranker.as_ref());
        if reranker_cfg.enabled {
            match load_reranker(&root, &reranker_cfg.model) {
                Ok(r) => {
                    tracing::info!(model = %reranker_cfg.model, "reranker attached");
                    engine = engine.with_reranker(r);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "reranker load failed; falling back to RRF order");
                }
            }
        }
    }
    Ok(engine)
}

fn load_reranker(root: &Path, model_spec: &str) -> Result<Arc<dyn RerankerLike>> {
    let model = Reranker::resolve_model(model_spec).ok_or_else(|| {
        anyhow!("unknown reranker model {model_spec:?}; see config docs for accepted aliases")
    })?;
    let cache_dir = root.join("models");
    std::fs::create_dir_all(&cache_dir)?;
    let r: Arc<Reranker> = Reranker::load_with_model(&cache_dir, model)
        .map_err(|e| anyhow!("load reranker {model_spec}: {e}"))?;
    Ok(r)
}

fn events_db_if_wanted(cfg: &Config, root: &Path) -> Result<Option<Arc<EventsDb>>> {
    if cfg
        .sources
        .iter()
        .any(|s| matches!(s.kind, SourceKind::OstkProject))
    {
        Ok(Some(Arc::new(
            EventsDb::open(root).map_err(|e| anyhow!("open events db: {e}"))?,
        )))
    } else {
        Ok(None)
    }
}

/// Start the MCP server.
///
/// Phase D: only the stdio transport is implemented. The `stdio` flag is
/// therefore required in practice, but we keep the arg so the CLI surface is
/// forward-compatible.
pub async fn serve(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    stdio: bool,
) -> Result<()> {
    if !stdio {
        return Err(anyhow!(
            "only stdio transport is currently supported; pass --stdio"
        ));
    }
    let engine = build_query_engine(config_path, embedder).await?;
    tracing::info!(
        model = %engine.model(),
        dim = engine.store().dim(),
        audit = engine.has_audit(),
        "mcp serve --stdio starting"
    );
    let server = Server::new(engine);
    server
        .run_stdio()
        .await
        .map_err(|e| anyhow!("mcp stdio: {e}"))?;
    tracing::info!("mcp serve exited cleanly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_pipeline::ChunkEmbedder;
    use tempfile::TempDir;

    struct FakeEmbedder;
    impl ChunkEmbedder for FakeEmbedder {
        fn dim(&self) -> usize {
            8
        }
        fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
            texts
                .iter()
                .map(|t| {
                    let mut v = vec![0.0; 8];
                    let bucket = t.len() % 8;
                    v[bucket] = 1.0;
                    v
                })
                .collect()
        }
    }

    fn write_min_config(cfg_path: &Path, root: &Path) {
        let body = format!(
            "[corpus]\nroot = \"{}\"\n\n[embedder]\nmodel = \"unused-in-tests\"\n",
            root.display()
        );
        std::fs::write(cfg_path, body).unwrap();
    }

    #[tokio::test]
    async fn init_force_wipes_existing_corpus() {
        let cfg_dir = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        let cfg_path = cfg_dir.path().join("config.toml");
        write_min_config(&cfg_path, corpus.path());

        // First init: creates corpus.lance/ and ingest.duckdb.
        let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
        init(&cfg_path, Arc::clone(&emb)).await.expect("first init");
        let lance = corpus.path().join("corpus.lance");
        let ingest = corpus.path().join("ingest.duckdb");
        assert!(lance.exists(), "first init should create corpus.lance");
        assert!(ingest.exists(), "first init should create ingest.duckdb");

        // Drop a sentinel into corpus.lance so we can prove it was wiped.
        let sentinel = lance.join("sentinel.txt");
        std::fs::write(&sentinel, "before-force").unwrap();
        assert!(sentinel.exists());

        // Second init with force: must wipe the table dir before reopen.
        init_with_options(
            &cfg_path,
            Arc::clone(&emb),
            InitOptions::default().with_force(true),
        )
        .await
        .expect("forced re-init");
        assert!(
            !sentinel.exists(),
            "force should have wiped corpus.lance/sentinel.txt"
        );
        // And the store is healthy again afterward.
        assert!(lance.exists(), "force re-init should recreate corpus.lance");
        assert!(
            ingest.exists(),
            "force re-init should recreate ingest.duckdb"
        );
    }

    #[tokio::test]
    async fn init_force_tolerates_missing_artifacts() {
        // A clean corpus root has no corpus.lance / ingest.duckdb yet —
        // force must not error.
        let cfg_dir = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        let cfg_path = cfg_dir.path().join("config.toml");
        write_min_config(&cfg_path, corpus.path());

        let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder);
        let outcome = init_with_options(&cfg_path, emb, InitOptions::default().with_force(true))
            .await
            .expect("force init on empty root");
        assert!(matches!(outcome, InitOutcome::Initialized { .. }));
    }
}
