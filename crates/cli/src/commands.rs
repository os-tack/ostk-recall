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
use ostk_recall_scan::gemini::GeminiScanner;
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
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Ok(PathBuf::from(xdg).join("ostk-recall").join("config.toml"));
        }
    }
    if let Some(home) = dirs::home_dir() {
        return Ok(home.join(".config").join("ostk-recall").join("config.toml"));
    }
    let base = dirs::config_dir()
        .ok_or_else(|| anyhow!("could not determine config dir (no $XDG_CONFIG_HOME, no $HOME)"))?;
    Ok(base.join("ostk-recall").join("config.toml"))
}

#[derive(Debug, Clone)]
pub enum InitOutcome {
    WroteStarter { path: PathBuf },
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

pub async fn init(config_path: &Path, embedder: Arc<dyn ChunkEmbedder>) -> Result<InitOutcome> {
    init_with_options(config_path, embedder, InitOptions::default()).await
}

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

fn skip_reranker_for_tests() -> bool {
    std::env::var("OSTK_RECALL_SKIP_RERANKER").is_ok()
        || std::env::var("OSTK_RECALL_FAKE_EMBEDDER").is_ok()
}

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

fn force_wipe_corpus(root: &Path) -> Result<()> {
    let lance = root.join("corpus.lance");
    if lance.exists() {
        eprintln!("forcing re-init: removing {}", lance.display());
        std::fs::remove_dir_all(&lance).with_context(|| format!("removing {}", lance.display()))?;
    }
    for db_name in ["ingest.sqlite", "events.sqlite"] {
        let p = root.join(db_name);
        if p.exists() {
            eprintln!("forcing re-init: removing {}", p.display());
            std::fs::remove_file(&p).with_context(|| format!("removing {}", p.display()))?;
        }
    }
    Ok(())
}

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

    let markdown = MarkdownScanner;
    let code = CodeScanner;
    let claude_code = ClaudeCodeScanner;
    let file_glob = FileGlobScanner;
    let zip_export = ZipExportScanner;
    let gemini = GeminiScanner;

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
            SourceKind::Gemini => &gemini,
        };

        let stats = pipeline.ingest_source(scanner, source_cfg).await;
        totals = totals.merge(stats);
        per_source.push((label, stats));
    }

    ostk_recall_scan::fcp_rust::drain_session_cache();

    Ok(ScanOutcome {
        per_source,
        totals,
        dry_run,
    })
}

pub async fn scan_reingest(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    reingest_project: &str,
) -> Result<ScanOutcome> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;

    let store = Arc::new(
        CorpusStore::open_or_create(&root, embedder.dim())
            .await
            .map_err(|e| anyhow!("open corpus store: {e}"))?,
    );
    let ingest = Arc::new(IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?);

    let ids = store
        .chunk_ids_for_project(reingest_project)
        .await
        .map_err(|e| anyhow!("chunk_ids_for_project({reingest_project}): {e}"))?;

    let lance_deleted = store
        .delete_by_project(reingest_project)
        .await
        .map_err(|e| anyhow!("delete_by_project({reingest_project}): {e}"))?;

    let ingest_deleted = ingest
        .delete_by_chunk_ids(&ids)
        .map_err(|e| anyhow!("delete_by_chunk_ids: {e}"))?;

    println!(
        "reingest {reingest_project}: deleted {lance_deleted} corpus rows, {ingest_deleted} ingest rows"
    );

    drop(store);
    drop(ingest);
    scan(config_path, embedder, Some(reingest_project), false).await
}

pub async fn inspect(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    chunk_id: &str,
) -> Result<ostk_recall_query::RecallLinkResult> {
    let engine = build_query_engine_with_reranker(config_path, embedder, false).await?;
    engine
        .recall_link(chunk_id)
        .await
        .map_err(|e| anyhow!("recall_link {chunk_id}: {e}"))
}

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

pub async fn build_query_engine(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
) -> Result<QueryEngine> {
    build_query_engine_inner(config_path, embedder, true, false).await
}

pub async fn build_query_engine_with_reranker(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    attach_reranker: bool,
) -> Result<QueryEngine> {
    build_query_engine_inner(config_path, embedder, attach_reranker, false).await
}

pub async fn build_query_engine_read_only(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
) -> Result<QueryEngine> {
    build_query_engine_inner(config_path, embedder, true, true).await
}

async fn build_query_engine_inner(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    attach_reranker: bool,
    read_only: bool,
) -> Result<QueryEngine> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    let store = Arc::new(
        CorpusStore::open_or_create(&root, embedder.dim())
            .await
            .map_err(|e| anyhow!("open corpus store: {e}"))?,
    );
    let ingest = Arc::new(if read_only {
        IngestDb::open_read_only(&root).map_err(|e| anyhow!("open ingest db (ro): {e}"))?
    } else {
        IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?
    });
    let events = events_db_if_wanted(&cfg, &root, read_only)?;
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

fn events_db_if_wanted(
    cfg: &Config,
    root: &Path,
    read_only: bool,
) -> Result<Option<Arc<EventsDb>>> {
    if cfg
        .sources
        .iter()
        .any(|s| matches!(s.kind, SourceKind::OstkProject))
    {
        let db = if read_only {
            EventsDb::open_read_only(root).map_err(|e| anyhow!("open events db (ro): {e}"))?
        } else {
            EventsDb::open(root).map_err(|e| anyhow!("open events db: {e}"))?
        };
        Ok(Some(Arc::new(db)))
    } else {
        Ok(None)
    }
}

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
    let engine = build_query_engine_read_only(config_path, Arc::clone(&embedder)).await?;
    let root = engine.store().root().to_path_buf();
    let sock_path = root.join("recall.sock");

    // Spawn background scan trigger listener
    let config_path_for_bg = config_path.to_path_buf();
    let embedder_for_bg = Arc::clone(&embedder);
    tokio::spawn(async move {
        if let Err(e) = run_socket_listener(&sock_path, config_path_for_bg, embedder_for_bg).await {
            tracing::error!(error = %e, "background socket listener failed");
        }
    });

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

/// Listen for scan-trigger pokes from a local single-operator surface and
/// trigger a background `scan` on each connection.
///
/// Transport branches by platform:
/// - **Unix**: `AF_UNIX` socket bound at `path`. Filesystem permissions on
///   `path` are the only access control — do not bind inside a directory
///   that other system users can reach.
/// - **Windows**: Named pipe at `\\.\pipe\ostk-recall-{stem}`, where
///   `{stem}` is `path.file_stem()`. Named pipes are the closest semantic
///   match to `AF_UNIX` (local-only, no other-user reach by default) and
///   tokio supports them natively. The `path` itself is not created on
///   disk; it just seeds the pipe name so different daemons get different
///   pipes.
///
/// In both cases this is a single-operator local surface — no message
/// protocol, no auth, no audit. It is intended for the operator's own
/// `ostk-recall serve` daemon to receive scan-trigger pokes (e.g. from a
/// file-watcher), not for multi-tenant or networked use.
///
/// Concurrency: at most one scan is in-flight. Connections that arrive
/// while a scan is running are accepted (so the client does not block on
/// `connect`) and immediately closed without spawning a new scan. This
/// prevents racing scans from corrupting orphan tracking, and bounds
/// spawn-per-connection memory pressure.
async fn run_socket_listener(
    path: &Path,
    config_path: PathBuf,
    embedder: Arc<dyn ChunkEmbedder>,
) -> Result<()> {
    use tokio::sync::Mutex;
    let scan_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));

    #[cfg(unix)]
    {
        use tokio::net::UnixListener;
        if path.exists() {
            let _ = std::fs::remove_file(path);
        }
        let listener = UnixListener::bind(path)?;
        tracing::info!(path = %path.display(), "listening for scan triggers (unix socket)");

        loop {
            match listener.accept().await {
                Ok((_stream, _addr)) => {
                    spawn_scan_trigger(&scan_lock, &config_path, &embedder);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "socket accept failed");
                }
            }
        }
    }

    #[cfg(windows)]
    {
        use tokio::net::windows::named_pipe::ServerOptions;
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("trigger");
        let pipe_name = format!(r"\\.\pipe\ostk-recall-{stem}");
        // first_pipe_instance(true) on the FIRST create only; subsequent
        // instances inherit security from the existing pipe namespace and
        // must omit it (tokio doc-comment behavior). Each accepted connect
        // consumes the current server instance, so we immediately spin up
        // the next one before handing the connected pipe to the spawn.
        let mut server = ServerOptions::new()
            .first_pipe_instance(true)
            .create(&pipe_name)?;
        tracing::info!(pipe = %pipe_name, "listening for scan triggers (named pipe)");

        loop {
            match server.connect().await {
                Ok(()) => {
                    let connected = server;
                    server = ServerOptions::new().create(&pipe_name)?;
                    spawn_scan_trigger(&scan_lock, &config_path, &embedder);
                    drop(connected);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "named pipe connect failed");
                    // Re-create the listener so a transient error doesn't
                    // wedge the loop on a half-broken instance.
                    server = ServerOptions::new().create(&pipe_name)?;
                }
            }
        }
    }
}

/// Shared scan-trigger handler — used by both the `AF_UNIX` and Named Pipe
/// listeners. Spawns a tokio task that grabs the `scan_lock` (`try_lock` so
/// we drop concurrent triggers instead of queueing them) and runs `scan`.
fn spawn_scan_trigger(
    scan_lock: &Arc<tokio::sync::Mutex<()>>,
    config_path: &Path,
    embedder: &Arc<dyn ChunkEmbedder>,
) {
    let scan_lock = Arc::clone(scan_lock);
    let config_path = config_path.to_path_buf();
    let embedder = Arc::clone(embedder);
    tokio::spawn(async move {
        let Ok(guard) = scan_lock.try_lock() else {
            tracing::warn!("scan trigger received while a scan is already in flight; dropping");
            return;
        };
        tracing::info!("scan trigger received");
        if let Err(e) = scan(&config_path, embedder, None, false).await {
            tracing::error!(error = %e, "background scan failed");
        } else {
            tracing::info!("background scan complete");
        }
        drop(guard);
    });
}
