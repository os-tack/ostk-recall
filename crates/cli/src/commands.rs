//! Subcommand implementations.
//!
//! Each public function here is callable from both `main.rs` (the binary
//! wraps these) and from integration tests (they inject a `FakeEmbedder`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use ostk_recall_core::{Config, Scanner, SourceKind};
use ostk_recall_mcp::Server;
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline, PipelineStats, VerifyReport};
use ostk_recall_query::QueryEngine;
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

/// Initialize a corpus.
///
/// If `config_path` doesn't exist, write a starter and return
/// `WroteStarter` without touching the corpus. Otherwise: create the corpus
/// root, open (or create) the `LanceDB` table sized to the embedder dim,
/// and open the ingest DB.
///
/// `embedder` is passed in so tests can supply a fake — the binary wraps
/// `Embedder::load` around the real one.
pub async fn init(config_path: &Path, embedder: Arc<dyn ChunkEmbedder>) -> Result<InitOutcome> {
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

    let dim = embedder.dim();
    let store = CorpusStore::open_or_create(&root, dim)
        .await
        .map_err(|e| anyhow!("open corpus store: {e}"))?;
    store
        .ensure_fts_index()
        .await
        .map_err(|e| anyhow!("ensure fts index: {e}"))?;
    let _ingest = IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?;

    Ok(InitOutcome::Initialized {
        root,
        model_id: cfg.embedder.model.clone(),
        dim,
    })
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

    // Scanners that are stateless can be reused across sources.
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
    Ok(QueryEngine::new(
        store,
        ingest,
        events,
        embedder,
        cfg.embedder.model.clone(),
    ))
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
