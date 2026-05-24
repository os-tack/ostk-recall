//! Subcommand implementations.
//!
//! Each public function here is callable from both `main.rs` (the binary
//! wraps these) and from integration tests (they inject a `FakeEmbedder`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use ostk_recall_core::{Config, RerankerConfig, Scanner, SourceConfig, SourceKind, WatchMode};
use ostk_recall_mcp::Server;
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline, PipelineStats, VerifyReport};
use ostk_recall_query::{QueryEngine, Reranker, RerankerLike};
use ostk_recall_scan::claude_code::ClaudeCodeScanner;
use ostk_recall_scan::code::CodeScanner;
use ostk_recall_scan::file_glob::FileGlobScanner;
use ostk_recall_scan::gemini::GeminiScanner;
use ostk_recall_scan::markdown::MarkdownScanner;
use ostk_recall_scan::ostk_project::OstkProjectScanner;
use ostk_recall_scan::threads::ThreadScanner;
use ostk_recall_scan::zip_export::ZipExportScanner;
use ostk_recall_store::{CorpusStore, EventsDb, IngestDb};

use fs4::FileExt;

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
    WroteStarter {
        path: PathBuf,
    },
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
    let thread = ThreadScanner;

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
            SourceKind::Thread => &thread,
            // Membrane sources are in-process synthetic ingest only
            // (turn observer → Pipeline::ingest_synthetic); they have
            // no on-disk scanner, so the config-driven scan skips them.
            SourceKind::Membrane => continue,
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

/// Path-aware sibling of [`scan`].
///
/// Loads the same config + scanners, then dispatches each input path to the
/// matching `[[sources]]` via [`Pipeline::scan_paths`]. Sources whose roots
/// don't parent any input are silent (no entry in the returned per-source
/// vector).
///
/// Skips delete handling — gh#7 covers tombstone semantics for paths
/// that yield no `SourceItem` (file removed, gitignore'd, etc.).
pub async fn scan_paths(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    paths: &[PathBuf],
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
    let thread = ThreadScanner;

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

    let pairs: Vec<(&dyn Scanner, &SourceConfig)> = cfg
        .sources
        .iter()
        .map(|source_cfg| {
            let scanner: &dyn Scanner = match source_cfg.kind {
                // Membrane sources are in-process synthetic ingest only
                // (turn observer → Pipeline::ingest_synthetic); they have
                // no on-disk scanner, so for scan_paths fan-out we treat
                // them as markdown — the path-filter step will exclude
                // any actual work since membrane sources name no roots.
                SourceKind::Markdown | SourceKind::Membrane => &markdown,
                SourceKind::Code => &code,
                SourceKind::ClaudeCode => &claude_code,
                SourceKind::FileGlob => &file_glob,
                SourceKind::ZipExport => &zip_export,
                SourceKind::OstkProject => &ostk_project,
                SourceKind::Gemini => &gemini,
                SourceKind::Thread => &thread,
            };
            (scanner, source_cfg)
        })
        .collect();

    let per_source = pipeline
        .scan_paths(&pairs, paths)
        .await
        .map_err(|e| anyhow!("scan_paths: {e}"))?;

    let mut totals = PipelineStats::default();
    for (_, s) in &per_source {
        totals = totals.merge(*s);
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
    use std::io::Write as _;

    if !stdio {
        return Err(anyhow!(
            "only stdio transport is currently supported; pass --stdio"
        ));
    }

    // Singleton guard: only one `serve` per corpus root.
    //
    // Multiple `serve` instances destroy each other's `recall.sock`
    // (run_socket_listener does `remove_file` + `bind`) and race on
    // the shared lance and SQLite state, which surfaces as
    // deterministic SIGABRT crashes. Hold an advisory exclusive flock
    // on `<corpus_root>/.serve.lock` for the lifetime of the process.
    // Second-and-later serves exit cleanly so the MCP client sees EOF
    // instead of a corrupted shared corpus.
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config {}", config_path.display()))?;
    let corpus_root = cfg.expanded_root().context("resolving corpus.root")?;
    std::fs::create_dir_all(&corpus_root)
        .with_context(|| format!("creating corpus root {}", corpus_root.display()))?;
    let lock_path = corpus_root.join(".serve.lock");
    let lock_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("opening serve lock {}", lock_path.display()))?;
    match FileExt::try_lock_exclusive(&lock_file) {
        Ok(()) => {
            let _ = lock_file.set_len(0);
            let _ = writeln!(&lock_file, "{}", std::process::id());
            tracing::info!(
                lock = %lock_path.display(),
                pid = std::process::id(),
                "serve lock acquired"
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
            let holder = std::fs::read_to_string(&lock_path)
                .unwrap_or_default()
                .trim()
                .to_string();
            tracing::warn!(
                lock = %lock_path.display(),
                holder_pid = %holder,
                "another ostk-recall serve is already running for this corpus; exiting cleanly"
            );
            return Ok(());
        }
        Err(e) => {
            return Err(anyhow!(
                "acquiring serve lock at {}: {e}",
                lock_path.display()
            ));
        }
    }
    // `lock_file` is held to end-of-function so the OS releases the
    // flock on process exit.

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
/// Wire format: line-delimited UTF-8 paths, terminated by writer EOF.
/// Empty body = legacy "scan all sources" (back-compat with the
/// pre-path-frame watcher). Non-empty body = scan only those paths via
/// [`Pipeline::scan_paths`]. Frame is capped at [`MAX_TRIGGER_FRAME_BYTES`]
/// — anything larger logs a warning and falls back to legacy scan-all
/// (defensive; the watcher's debounce + per-batch split should never
/// produce that much).
///
/// Concurrency: at most one scan is in-flight. Connections that arrive
/// while a scan is running are accepted (so the client does not block on
/// `connect`); the body is read but the spawn drops if the lock is held,
/// preventing racing scans from corrupting orphan tracking and bounding
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
                Ok((mut stream, _addr)) => {
                    let paths = read_trigger_paths(&mut stream).await;
                    spawn_scan_trigger(&scan_lock, &config_path, &embedder, paths);
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
                    let mut connected = server;
                    server = ServerOptions::new().create(&pipe_name)?;
                    let paths = read_trigger_paths(&mut connected).await;
                    spawn_scan_trigger(&scan_lock, &config_path, &embedder, paths);
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

/// Maximum trigger frame size in bytes. Frames larger than this log a
/// warning and fall back to legacy scan-all. Sized to absorb a few
/// hundred path lines — well above any debounced batch the watcher
/// produces in practice (debounced + per-source filter + per-batch
/// split in `kick_trigger_socket`).
const MAX_TRIGGER_FRAME_BYTES: usize = 64 * 1024;

/// Read a single scan-trigger frame off `stream`, capped at
/// [`MAX_TRIGGER_FRAME_BYTES`]. The frame is one UTF-8 path per line,
/// LF-terminated, with writer EOF closing the frame. Returns the paths
/// in the order they were written.
///
/// Empty body (immediate writer close) returns an empty vec — the
/// caller treats that as a legacy "scan all sources" poke.
///
/// Errors and oversize frames are converted to "empty paths" with a
/// warning log — the legacy fallback is the right behavior on every
/// observable error mode (read error, oversize, non-UTF-8 byte, missing
/// final newline). The escape hatch is load-bearing for backward compat:
/// a malformed body should never wedge the daemon.
///
/// Read deadline: 2 s after first byte of the body. The watcher writes
/// the whole frame and shuts down its write half immediately (a single
/// syscall worth of bytes for any realistic batch); 2 s is generous
/// even on a swap-thrashing host. Without a deadline a hung writer
/// could pin the accept thread; with the deadline we just fall back to
/// legacy scan-all and accept the next connection.
async fn read_trigger_paths<R>(stream: &mut R) -> Vec<PathBuf>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use std::time::Duration;
    use tokio::io::AsyncReadExt;

    let mut buf = Vec::with_capacity(1024);
    let read_result = tokio::time::timeout(Duration::from_secs(2), async {
        // read_to_end on a tokio reader returns Ok when the writer half
        // shuts down (EOF). Cap collected bytes ourselves rather than
        // using take(N).read_to_end, so we can detect overflow vs honest
        // exact-N frames.
        let mut chunk = [0u8; 4096];
        loop {
            let n = stream.read(&mut chunk).await?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
            if buf.len() > MAX_TRIGGER_FRAME_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "trigger frame exceeded cap",
                ));
            }
        }
        Ok::<(), std::io::Error>(())
    })
    .await;

    match read_result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "trigger frame read failed; falling back to legacy scan-all");
            return Vec::new();
        }
        Err(_) => {
            tracing::warn!("trigger frame read timed out; falling back to legacy scan-all");
            return Vec::new();
        }
    }

    let body = match std::str::from_utf8(&buf) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "trigger frame contained non-UTF-8 bytes; falling back to legacy scan-all");
            return Vec::new();
        }
    };

    body.lines()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// `ostk-recall watch` — run a file-watcher.
///
/// Pokes the scan-trigger socket whenever a debounced batch of events
/// touches a configured source path. Reuses each `[[sources]].paths` (and
/// `extensions`) — the watcher does not declare its own paths. The scan
/// does the real filtering; the watcher's job is just "did anything we care
/// about change recently?".
///
/// Behavior:
/// - Loads `[watch]` from config; bails if absent or `enabled = false`.
/// - Resolves the trigger socket via `WatchConfig::resolve_socket`
///   (defaults to `corpus.root/recall.sock` — same path `serve` binds).
/// - For every `[[sources]]` whose `project` passes the optional
///   `[watch].projects` allowlist, registers each expanded path with a
///   recursive debouncer (`notify-debouncer-full`).
/// - Drains debounced batches; on any batch that contains a path under
///   one of the watched roots (and matching the source's `extensions`,
///   if any), connects to the socket and immediately closes the
///   connection. The serve daemon's accept-loop treats that as a poke
///   and runs `scan`.
///
/// Connect failures (e.g. serve not running yet) are warnings, not
/// fatal — the watcher keeps running so the next save after `serve`
/// starts will fire. Errors from the underlying notify backend are
/// surfaced through `tracing::warn`.
#[allow(clippy::too_many_lines)]
pub async fn watch(config_path: &Path) -> Result<()> {
    use std::time::Duration;

    use notify_debouncer_full::{DebounceEventResult, new_debouncer, notify::RecursiveMode};

    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;

    let watch_cfg = cfg
        .watch
        .as_ref()
        .ok_or_else(|| {
            anyhow!(
                "no [watch] block in config — add one with `enabled = true` to enable the watcher"
            )
        })?
        .clone();
    if !watch_cfg.is_active() {
        bail!("[watch].enabled = false — nothing to do");
    }

    let corpus_root = cfg.expanded_root().context("expanding corpus root")?;
    let socket_path = watch_cfg
        .resolve_socket(&corpus_root)
        .context("resolving trigger socket path")?;

    // Resolve which sources to watch and expand each declared path.
    // Skip sources that don't pass the project allowlist; skip individual
    // paths that don't exist (logged at warn so misconfig is visible).
    let mut watched_roots: Vec<(PathBuf, SourceConfig)> = Vec::new();
    for source in &cfg.sources {
        if !watch_cfg.watches_project(source.project.as_deref()) {
            continue;
        }
        for raw in &source.paths {
            let expanded = match ostk_recall_core::config::expand_path(raw) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(path = %raw, error = %e, "failed to expand watch path; skipping");
                    continue;
                }
            };
            if !expanded.exists() {
                tracing::warn!(
                    path = %expanded.display(),
                    "watch path does not exist; skipping (will not retry until restart)"
                );
                continue;
            }
            watched_roots.push((expanded, source.clone()));
        }
    }
    if watched_roots.is_empty() {
        bail!(
            "no watchable source paths after applying [watch].projects filter — refine the filter or add sources"
        );
    }

    // Tokio mpsc as the bridge: the debouncer's std-style closure can call
    // UnboundedSender::send (non-blocking, no async needed), and the main
    // task awaits on the receiver.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<DebounceEventResult>();
    let mut debouncer = new_debouncer(
        Duration::from_millis(watch_cfg.debounce_ms),
        None,
        move |result: DebounceEventResult| {
            let _ = tx.send(result);
        },
    )
    .context("creating filesystem debouncer")?;

    for (root, _) in &watched_roots {
        debouncer
            .watch(root, RecursiveMode::Recursive)
            .with_context(|| format!("registering watch on {}", root.display()))?;
        tracing::info!(path = %root.display(), "watching");
    }
    tracing::info!(
        socket = %socket_path.display(),
        debounce_ms = watch_cfg.debounce_ms,
        sources = watched_roots.len(),
        "scan-trigger watcher started"
    );

    let mode = watch_cfg.mode;
    while let Some(result) = rx.recv().await {
        match result {
            Ok(events) => {
                // Collect every matched path across the debounced batch.
                // The same predicate that decides whether to kick decides
                // which paths go on the wire. Dedup preserves first-seen
                // order — line-delimited frame is order-insensitive but a
                // stable order makes test assertions and log output
                // friendlier.
                let mut matched: Vec<PathBuf> = Vec::new();
                for ev in &events {
                    for p in &ev.event.paths {
                        if matches_watched_root(p, &watched_roots) && !matched.contains(p) {
                            matched.push(p.clone());
                        }
                    }
                }
                if matched.is_empty() {
                    continue;
                }
                let frame: &[PathBuf] = match mode {
                    WatchMode::Legacy => &[],
                    WatchMode::Incremental => &matched,
                };
                if let Err(e) = kick_trigger_socket(&socket_path, frame).await {
                    tracing::warn!(
                        socket = %socket_path.display(),
                        error = %e,
                        "scan-trigger kick failed; will retry on next event"
                    );
                } else {
                    tracing::info!(paths = frame.len(), "scan-trigger kicked");
                }
            }
            Err(errors) => {
                for e in errors {
                    tracing::warn!(error = %e, "watch backend error");
                }
            }
        }
    }
    Ok(())
}

/// True if `path` falls under one of the watched roots AND, if that
/// root's source has a non-empty `extensions` filter, the path's
/// extension is in the filter. Roots without an extensions filter
/// match every path under them.
fn matches_watched_root(path: &Path, roots: &[(PathBuf, SourceConfig)]) -> bool {
    for (root, source) in roots {
        if !path.starts_with(root) {
            continue;
        }
        if source.extensions.is_empty() {
            return true;
        }
        let ext_ok = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| source.extensions.iter().any(|x| x == ext));
        if ext_ok {
            return true;
        }
    }
    false
}

/// Connect to the scan-trigger surface, write `paths` line-delimited
/// (UTF-8 + LF), then half-close the writer to signal EOF. Empty `paths`
/// is the legacy "scan all" poke (server reads zero bytes, treats as
/// legacy). Non-empty paths drive the server's per-path scan.
///
/// Frame size is capped at [`MAX_TRIGGER_FRAME_BYTES`]; if `paths` would
/// overflow, the batch is split across multiple connects (each connect
/// is its own frame from the server's perspective). Paths containing
/// embedded newlines are skipped — line-delimited frames cannot
/// represent them.
async fn kick_trigger_socket(socket: &Path, paths: &[PathBuf]) -> Result<()> {
    if paths.is_empty() {
        // Legacy poke: connect-and-close, server reads empty body and
        // runs full scan. Preserved for `[watch].mode = "legacy"` and as
        // the cross-version compat wire.
        kick_trigger_once(socket, &[]).await?;
        return Ok(());
    }

    let mut batch: Vec<&PathBuf> = Vec::new();
    let mut batch_bytes: usize = 0;
    for p in paths {
        let Some(s) = p.to_str() else {
            tracing::warn!(path = %p.display(), "skipping non-UTF-8 path in trigger frame");
            continue;
        };
        if s.contains('\n') {
            tracing::warn!(
                path = %p.display(),
                "skipping path with embedded newline (line-delimited frame can't represent it)"
            );
            continue;
        }
        let line_bytes = s.len() + 1;
        if line_bytes > MAX_TRIGGER_FRAME_BYTES {
            // A single line over the cap can never fit in any frame.
            tracing::warn!(
                path = %p.display(),
                len = line_bytes,
                cap = MAX_TRIGGER_FRAME_BYTES,
                "skipping path larger than trigger frame cap"
            );
            continue;
        }
        if batch_bytes + line_bytes > MAX_TRIGGER_FRAME_BYTES {
            // Flush current batch before adding this line.
            kick_trigger_once(socket, &batch).await?;
            batch.clear();
            batch_bytes = 0;
        }
        batch.push(p);
        batch_bytes += line_bytes;
    }
    if !batch.is_empty() {
        kick_trigger_once(socket, &batch).await?;
    }
    Ok(())
}

/// Write a single trigger frame: each path UTF-8 + `\n`, then half-close
/// the writer so the server sees EOF and dispatches.
async fn kick_trigger_once(socket: &Path, paths: &[&PathBuf]) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    #[cfg(unix)]
    {
        let mut stream = tokio::net::UnixStream::connect(socket)
            .await
            .with_context(|| format!("connect {}", socket.display()))?;
        for p in paths {
            if let Some(s) = p.to_str() {
                stream.write_all(s.as_bytes()).await?;
                stream.write_all(b"\n").await?;
            }
        }
        // Half-close the writer so the server's reader observes EOF.
        // Errors here are non-fatal — the bytes are already in the kernel
        // buffer.
        let _ = stream.shutdown().await;
        Ok(())
    }

    #[cfg(windows)]
    {
        let stem = socket
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("trigger");
        let pipe_name = format!(r"\\.\pipe\ostk-recall-{stem}");
        let mut client = tokio::net::windows::named_pipe::ClientOptions::new()
            .open(&pipe_name)
            .with_context(|| format!("open pipe {pipe_name}"))?;
        for p in paths {
            if let Some(s) = p.to_str() {
                client.write_all(s.as_bytes()).await?;
                client.write_all(b"\n").await?;
            }
        }
        // Named pipes don't have shutdown(); dropping the client closes
        // the write side, which the server's read_to_end observes as EOF.
        let _ = client.shutdown().await;
        Ok(())
    }
}

/// Shared scan-trigger handler — used by both the `AF_UNIX` and Named Pipe
/// listeners. Spawns a tokio task that grabs the `scan_lock` (`try_lock` so
/// we drop concurrent triggers instead of queueing them) and runs the
/// appropriate scan: empty `paths` → legacy [`scan`] (back-compat poke);
/// non-empty → [`scan_paths`] for the per-path code path.
fn spawn_scan_trigger(
    scan_lock: &Arc<tokio::sync::Mutex<()>>,
    config_path: &Path,
    embedder: &Arc<dyn ChunkEmbedder>,
    paths: Vec<PathBuf>,
) {
    let scan_lock = Arc::clone(scan_lock);
    let config_path = config_path.to_path_buf();
    let embedder = Arc::clone(embedder);
    tokio::spawn(async move {
        let Ok(guard) = scan_lock.try_lock() else {
            tracing::warn!("scan trigger received while a scan is already in flight; dropping");
            return;
        };
        let result = if paths.is_empty() {
            tracing::info!("scan trigger received (legacy scan-all)");
            scan(&config_path, embedder, None, false).await.map(|_| ())
        } else {
            tracing::info!(paths = paths.len(), "scan trigger received (per-path)");
            scan_paths(&config_path, embedder, &paths, false)
                .await
                .map(|_| ())
        };
        if let Err(e) = result {
            tracing::error!(error = %e, "background scan failed");
        } else {
            tracing::info!("background scan complete");
        }
        drop(guard);
    });
}

#[cfg(all(test, unix))]
mod trigger_wire_tests {
    //! Wire-protocol tests for the scan-trigger socket. Cover the
    //! line-delimited path frame end-to-end: read side, write side, and
    //! a round-trip via a real Unix socket.
    //!
    //! Unix-only: the Windows arm uses Named Pipes which need a `serve`
    //! daemon's `ServerOptions::create` to even bind. The protocol is
    //! identical (read until EOF on writer half-shutdown), so the
    //! Unix-side tests cover the wire contract.

    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::{UnixListener, UnixStream};

    #[tokio::test]
    async fn read_trigger_paths_empty_body_returns_empty_vec() {
        // Simulate connect-and-close: writer shuts down without sending
        // any bytes. This is the legacy poke shape.
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("trigger.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_trigger_paths(&mut stream).await
        });

        let mut client = UnixStream::connect(&sock).await.unwrap();
        client.shutdown().await.unwrap();
        drop(client);

        let paths = server.await.unwrap();
        assert!(paths.is_empty());
    }

    #[tokio::test]
    async fn read_trigger_paths_three_paths_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("trigger.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_trigger_paths(&mut stream).await
        });

        let mut client = UnixStream::connect(&sock).await.unwrap();
        client
            .write_all(b"/a/b.rs\n/c/d.md\n/e/f.txt\n")
            .await
            .unwrap();
        client.shutdown().await.unwrap();
        drop(client);

        let paths = server.await.unwrap();
        assert_eq!(
            paths,
            vec![
                PathBuf::from("/a/b.rs"),
                PathBuf::from("/c/d.md"),
                PathBuf::from("/e/f.txt"),
            ]
        );
    }

    #[tokio::test]
    async fn read_trigger_paths_oversized_frame_falls_back_to_legacy() {
        // 64 KiB + 1 of payload should trip the cap and return Vec::new().
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("trigger.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_trigger_paths(&mut stream).await
        });

        let mut client = UnixStream::connect(&sock).await.unwrap();
        // One absurdly long path line — easier to trip the cap than
        // 64 KiB of small lines and verifies the cap, not the line count.
        let mut huge = vec![b'x'; MAX_TRIGGER_FRAME_BYTES + 1];
        huge.push(b'\n');
        let _ = client.write_all(&huge).await;
        let _ = client.shutdown().await;
        drop(client);

        let paths = server.await.unwrap();
        assert!(
            paths.is_empty(),
            "oversize frame should fall back to legacy scan-all"
        );
    }

    #[tokio::test]
    async fn kick_trigger_socket_legacy_writes_empty_body() {
        // Watcher in legacy mode passes an empty path slice. The server
        // should observe a connection with zero body bytes.
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("trigger.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_trigger_paths(&mut stream).await
        });

        kick_trigger_socket(&sock, &[]).await.unwrap();

        let paths = server.await.unwrap();
        assert!(paths.is_empty());
    }

    #[tokio::test]
    async fn kick_trigger_socket_writes_each_path_line_delimited() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("trigger.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_trigger_paths(&mut stream).await
        });

        let payload = vec![
            PathBuf::from("/tmp/one.rs"),
            PathBuf::from("/tmp/two.md"),
            PathBuf::from("/tmp/three.txt"),
        ];
        kick_trigger_socket(&sock, &payload).await.unwrap();

        let paths = server.await.unwrap();
        assert_eq!(paths, payload);
    }

    #[tokio::test]
    async fn kick_trigger_socket_skips_paths_with_embedded_newline() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("trigger.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            read_trigger_paths(&mut stream).await
        });

        // Path with embedded \n must be filtered so the line-delimited
        // frame stays parseable on the server side.
        let bad = PathBuf::from("/tmp/a\nfile.rs");
        let good = PathBuf::from("/tmp/clean.rs");
        kick_trigger_socket(&sock, &[bad, good.clone()])
            .await
            .unwrap();

        let paths = server.await.unwrap();
        assert_eq!(paths, vec![good]);
    }

    #[tokio::test]
    async fn kick_trigger_socket_splits_oversized_batch_across_connects() {
        // 100 paths × ~1 KiB each easily exceeds the 64 KiB cap. Each
        // batch must arrive whole on its own connect; sum should equal
        // input.
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("trigger.sock");
        let listener = UnixListener::bind(&sock).unwrap();

        // Each path is ~750 bytes — 100 of them is ~75 KiB, must split
        // into ≥2 connects.
        let prefix = "/tmp/split/".to_string() + &"x".repeat(700);
        let payload: Vec<PathBuf> = (0..100)
            .map(|i| PathBuf::from(format!("{prefix}/{i:03}.rs")))
            .collect();
        let total = payload.len();

        let server = tokio::spawn(async move {
            let mut all = Vec::new();
            let mut connect_count = 0usize;
            // Each connect is a full frame. Stop accepting once we've
            // received every input path or hit a generous safety cap.
            while all.len() < total && connect_count < 8 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let batch = read_trigger_paths(&mut stream).await;
                all.extend(batch);
                connect_count += 1;
            }
            (all, connect_count)
        });

        kick_trigger_socket(&sock, &payload).await.unwrap();

        let (got, connects) = server.await.unwrap();
        assert!(connects >= 2, "expected ≥2 connects, got {connects}");
        assert_eq!(got, payload);
    }
}

#[cfg(test)]
mod watch_mode_tests {
    use super::WatchMode;
    use ostk_recall_core::WatchConfig;

    #[test]
    fn watch_mode_default_is_legacy() {
        let w = WatchConfig::default();
        assert_eq!(w.mode, WatchMode::Legacy);
    }
}
