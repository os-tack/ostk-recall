//! Subcommand implementations.
//!
//! Each public function here is callable from both `main.rs` (the binary
//! wraps these) and from integration tests (they inject a `FakeEmbedder`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use ostk_recall_attention::{
    AttentionForwardStore, AutoWeaver, CuratorConfig, IdleCurator, InMemoryAttention, ReplayEvent,
    TurnObserver, WeaverThresholds, ambient_scope_default,
};
use ostk_recall_attention_mcp::AttentionDispatch;
use ostk_recall_core::attention::{AttentionScope, PrivacyTier};
use ostk_recall_core::{
    CompiledRecordRules, Config, LensSettings, RerankerConfig, Scanner, SourceConfig, SourceKind,
    WatchMode, WeaverSettings,
};
use ostk_recall_mcp::{ClientId, ResourceRegistry, Server};
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline, PipelineStats, VerifyReport};
use ostk_recall_query::lens::LensConfig;
use ostk_recall_query::{QueryEngine, Reranker, RerankerLike};

use crate::lens_loop::{MemoryLensResource, run_lens_loop};
use crate::lens_state::load_lens_state;
use ostk_recall_scan::claude_code::ClaudeCodeScanner;
use ostk_recall_scan::code::CodeScanner;
use ostk_recall_scan::file_glob::FileGlobScanner;
use ostk_recall_scan::gemini::GeminiScanner;
use ostk_recall_scan::markdown::MarkdownScanner;
use ostk_recall_scan::ostk_project::OstkProjectScanner;
use ostk_recall_scan::threads::ThreadScanner;
use ostk_recall_scan::zip_export::ZipExportScanner;
use ostk_recall_store::{
    ChainLogReader, ChainSink, CorpusStore, EventsDb, IngestDb, SqliteChainSink, ThreadsDb,
};
use tokio_util::sync::CancellationToken;

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

pub type WeaveOutcome = ostk_recall_attention::WeaveWindowOutcome;
pub type ConsolidateOutcome = ostk_recall_attention::ConsolidateOutcome;

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
    store
        .ensure_auto_cleanup_disabled()
        .await
        .map_err(|e| anyhow!("strip auto_cleanup: {e}"))?;
    store
        .ensure_chunk_id_index()
        .await
        .map_err(|e| anyhow!("ensure chunk_id index: {e}"))?;
    store
        .ensure_project_index()
        .await
        .map_err(|e| anyhow!("ensure project index: {e}"))?;
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
    // Remove each SQLite DB *and* its WAL-mode sidecars. Deleting only the
    // main `.sqlite` while leaving an orphaned `-wal`/`-shm` behind puts
    // SQLite in a broken state: a later read-only open fails with "unable
    // to open database file" (which crashes `serve` → MCP `-32000`) and a
    // read-write open can abort the scan, silently stranding the corpus
    // empty. So wipe the sidecars in the same pass.
    for db_name in ["ingest.sqlite", "events.sqlite"] {
        for suffix in ["", "-wal", "-shm", "-journal"] {
            let p = root.join(format!("{db_name}{suffix}"));
            if p.exists() {
                eprintln!("forcing re-init: removing {}", p.display());
                std::fs::remove_file(&p).with_context(|| format!("removing {}", p.display()))?;
            }
        }
    }
    Ok(())
}

/// Shared mutable state owned by `serve()`.
///
/// Reused across every per-trigger scan. The serve binary constructs
/// this once at boot; the scan-trigger handler reads from it instead of
/// opening per-scan `ThreadsDb` instances so the long-lived
/// `IdleCurator` and the per-scan ambient daemons agree on familiarity,
/// tension, and proposed-thread state.
#[derive(Clone)]
pub struct ServeContext {
    pub threads: Arc<ThreadsDb>,
    pub attention: Arc<InMemoryAttention>,
}

/// Resolve the `Arc<ThreadsDb>` a scan should use.
///
/// When `ctx` is `Some` (i.e. the caller is the long-lived `serve()`
/// daemon) the scan shares the long-lived ledger. Otherwise it opens a
/// fresh per-scan handle. Open failures degrade to "ambient pickup
/// disabled" rather than failing the whole scan.
/// Idempotently ensure the lance corpus has the scalar + FTS indexes the
/// hot paths depend on, and that the per-commit auto-cleanup hook is
/// disabled. `init` already runs the index ensures once at create time;
/// we also run them at the start of every scan so corpora initialized
/// before these indexes existed get the one-time backfill on next
/// ingest. The auto-cleanup strip is critical — lance's defaults bake
/// `interval=20/older_than=14days` into the manifest and run
/// `cleanup_old_versions` (an O(versions) walk) every 20th commit, even
/// when nothing is old enough to delete.
async fn ensure_corpus_indexes(store: &CorpusStore) -> Result<()> {
    store
        .ensure_auto_cleanup_disabled()
        .await
        .map_err(|e| anyhow!("strip auto_cleanup: {e}"))?;
    store
        .ensure_chunk_id_index()
        .await
        .map_err(|e| anyhow!("ensure chunk_id index: {e}"))?;
    store
        .ensure_project_index()
        .await
        .map_err(|e| anyhow!("ensure project index: {e}"))?;
    Ok(())
}

fn resolve_threads_db(ctx: Option<&ServeContext>, root: &Path) -> Option<Arc<ThreadsDb>> {
    ctx.map_or_else(
        || match ThreadsDb::open(root) {
            Ok(db) => Some(Arc::new(db)),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "ambient daemons: ThreadsDb::open failed; skipping ambient pickup"
                );
                None
            }
        },
        |c| Some(Arc::clone(&c.threads)),
    )
}

/// Spawn the ambient-pickup daemons (`TurnObserver`, `AutoWeaver`).
///
/// Returns a `CancellationToken` to fire when the scan completes plus the
/// two `JoinHandle`s to await on cleanup. `pipeline` must be the same
/// `Arc<Pipeline>` the scan uses for ingest — both daemons subscribe to
/// its broadcast channel for `IngestEvent`s. `corpus` provides text
/// lookup for the observer's `fetch_texts` call. `threads` is shared
/// with the long-lived `IdleCurator` in `serve()` so the per-scan
/// daemons and the curator agree on familiarity and tension state.
/// When `serve()` is not the caller (e.g. one-shot `scan` / `scan_paths`),
/// callers may pass a freshly-opened `Arc<ThreadsDb>` — semantics are
/// identical, just without curator sharing.
type AmbientHandles = (
    CancellationToken,
    tokio::task::JoinHandle<()>,
    tokio::task::JoinHandle<()>,
);

/// Compile the effective record rules (P12) from config into the pipeline
/// overlay. Surfaces config-level rule errors (bad regex, unknown
/// `source_kind`/`source`, no-predicate rule) as a load-time failure.
fn compile_record_rules(cfg: &Config) -> Result<Arc<CompiledRecordRules>> {
    CompiledRecordRules::build(&cfg.effective_record_rules())
        .map(Arc::new)
        .map_err(|e| anyhow!("record rules: {e}"))
}

fn spawn_ambient_daemons(
    pipeline: &Arc<Pipeline>,
    corpus: &Arc<CorpusStore>,
    threads: &Arc<ThreadsDb>,
    attention: Option<Arc<dyn AttentionForwardStore>>,
    weaver_exclude_facets: Vec<String>,
) -> AmbientHandles {
    let mut observer = TurnObserver::new(Arc::clone(pipeline), Arc::clone(threads));
    if let Some(attn) = attention {
        observer = observer.with_attention(attn);
    }
    // P6A — share the ledger's chain sink with the observer so every
    // observe()-mediated `attend()` persists
    // `RollingVectorSnapshot` (or `AttentionTurnSkipped` once the
    // P6-full noise gate ships). Without this the rolling channel
    // lives only in memory and a restart erases it. The sink is
    // already `Arc`-shared, so this is a clone of a pointer.
    observer = observer.with_chain_sink(threads.chain_sink());
    let observer = Arc::new(observer);
    let weaver = Arc::new(
        AutoWeaver::new(
            Arc::clone(threads),
            Arc::clone(corpus),
            WeaverThresholds::default(),
        )
        .with_exclude_facets(weaver_exclude_facets),
    );
    let cancel = CancellationToken::new();

    let observer_handle = {
        let observer = Arc::clone(&observer);
        let corpus = Arc::clone(corpus);
        let cancel = cancel.clone();
        tokio::spawn(async move {
            if let Err(err) = observer
                .run_subscribed(corpus, ambient_scope_default(), cancel)
                .await
            {
                tracing::warn!(error = %err, "turn-observer task exited with error");
            }
        })
    };

    let weaver_handle = {
        let weaver = Arc::clone(&weaver);
        let pipeline = Arc::clone(pipeline);
        let cancel = cancel.clone();
        tokio::spawn(async move {
            if let Err(err) = weaver.run(&pipeline, cancel).await {
                tracing::warn!(error = %err, "auto-weaver task exited with error");
            }
        })
    };

    tracing::info!("ambient daemons spawned (turn-observer + auto-weaver)");
    (cancel, observer_handle, weaver_handle)
}

/// Cancel the ambient daemons and await their join handles.
async fn shutdown_ambient_daemons(handles: Option<AmbientHandles>) {
    let Some((cancel, observer_handle, weaver_handle)) = handles else {
        return;
    };
    cancel.cancel();
    let _ = observer_handle.await;
    let _ = weaver_handle.await;
}

pub async fn scan(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    source_filter: Option<&str>,
    dry_run: bool,
) -> Result<ScanOutcome> {
    scan_with_context(config_path, embedder, source_filter, dry_run, None).await
}

/// `scan` variant that reuses a caller-owned `ServeContext` instead of
/// opening a fresh `ThreadsDb`. Used by `serve()` so per-trigger scans
/// share state with the long-lived `IdleCurator`.
pub async fn scan_with_context(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    source_filter: Option<&str>,
    dry_run: bool,
    ctx: Option<&ServeContext>,
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
    ensure_corpus_indexes(&store).await?;
    let ingest = Arc::new(IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?);
    let pipeline = Arc::new(
        Pipeline::new(Arc::clone(&store), ingest, Arc::clone(&embedder))
            .with_dry_run(dry_run)
            .with_record_rules(compile_record_rules(&cfg)?),
    );

    let threads_db = resolve_threads_db(ctx, &root);
    let attention_dyn: Option<Arc<dyn AttentionForwardStore>> =
        ctx.map(|c| c.attention.clone() as Arc<dyn AttentionForwardStore>);
    let ambient = threads_db.as_ref().map(|threads| {
        spawn_ambient_daemons(
            &pipeline,
            &store,
            threads,
            attention_dyn.clone(),
            WeaverSettings::resolve(cfg.weaver.as_ref()).exclude_facets,
        )
    });

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

    // Drain ambient daemons FIRST so any synthetic ingests they fire
    // in response to source-loop events (TurnObserver auto-promotion,
    // AutoWeaver evidence linking, etc.) land in the corpus before we
    // reindex. The drain is cooperative — `shutdown_ambient_daemons`
    // cancels then awaits the join handles, so by the time it returns
    // the daemons' last `ingest_synthetic` call has committed.
    shutdown_ambient_daemons(ambient).await;

    // End-of-scan index maintenance: lance scalar indexes don't cover
    // fragments appended after the index was built, so each ingest
    // batch progressively shifts more work onto the unindexed-tail
    // scan leg of `Union(MapIndex, FilteredRead)`. Folding the new
    // fragments back into the existing indices is what restores fast
    // dedupe lookups for the *next* scan. We deliberately skip
    // `Compact` + `Prune` here — both scan with the version count, so
    // running them after every scan is O(commits) and adds hours to a
    // healthy corpus while contributing nothing to lookup latency. The
    // standalone `ostk-recall optimize` subcommand runs the full pass
    // when the operator wants to collapse a backlog.
    if !dry_run {
        let started = std::time::Instant::now();
        tracing::info!("running OptimizeAction::Index (fold new fragments into indices)");
        if let Err(err) = store.optimize_indices().await {
            tracing::warn!(error = %err, "end-of-scan optimize_indices failed");
        } else {
            tracing::info!(
                elapsed_s = started.elapsed().as_secs_f64(),
                "end-of-scan optimize_indices complete"
            );
        }
    }

    Ok(ScanOutcome {
        per_source,
        totals,
        dry_run,
    })
}

pub async fn weave(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    since: Option<Duration>,
    epoch_size: usize,
) -> Result<WeaveOutcome> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    std::fs::create_dir_all(&root)?;
    let store = Arc::new(
        CorpusStore::open_or_create(&root, embedder.dim())
            .await
            .map_err(|e| anyhow!("open corpus store: {e}"))?,
    );
    let threads = Arc::new(ThreadsDb::open(&root).map_err(|e| anyhow!("open threads db: {e}"))?);
    let weaver = AutoWeaver::new(threads, store, WeaverThresholds::default())
        .with_exclude_facets(WeaverSettings::resolve(cfg.weaver.as_ref()).exclude_facets);
    let since = since
        .map(chrono::Duration::from_std)
        .transpose()
        .map_err(|_| anyhow!("--since duration is too large for chrono"))?;
    weaver
        .weave_window(since, epoch_size.max(1))
        .await
        .map_err(|e| anyhow!("weave_window: {e}"))
}

/// Run a coarse consolidation cycle (`weave --consolidate`). Same setup as
/// [`weave`], but invokes [`AutoWeaver::consolidate`]: deep re-weave + anchor
/// bridging + proposal promotion + idle fade. Offline operator policy.
pub async fn consolidate(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    since: Option<Duration>,
    epoch_size: usize,
) -> Result<ConsolidateOutcome> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    std::fs::create_dir_all(&root)?;
    let store = Arc::new(
        CorpusStore::open_or_create(&root, embedder.dim())
            .await
            .map_err(|e| anyhow!("open corpus store: {e}"))?,
    );
    let threads = Arc::new(ThreadsDb::open(&root).map_err(|e| anyhow!("open threads db: {e}"))?);
    let weaver = AutoWeaver::new(threads, store, WeaverThresholds::default())
        .with_exclude_facets(WeaverSettings::resolve(cfg.weaver.as_ref()).exclude_facets);
    let since = since
        .map(chrono::Duration::from_std)
        .transpose()
        .map_err(|_| anyhow!("--since duration is too large for chrono"))?;
    weaver
        .consolidate(since, epoch_size.max(1))
        .await
        .map_err(|e| anyhow!("consolidate: {e}"))
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
    scan_paths_with_context(config_path, embedder, paths, dry_run, None).await
}

/// `scan_paths` variant that reuses a caller-owned `ServeContext` instead
/// of opening a fresh `ThreadsDb`. See [`scan_with_context`].
pub async fn scan_paths_with_context(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    paths: &[PathBuf],
    dry_run: bool,
    ctx: Option<&ServeContext>,
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
    ensure_corpus_indexes(&store).await?;
    let ingest = Arc::new(IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?);
    let pipeline = Arc::new(
        Pipeline::new(Arc::clone(&store), ingest, Arc::clone(&embedder))
            .with_dry_run(dry_run)
            .with_record_rules(compile_record_rules(&cfg)?),
    );

    let threads_db = resolve_threads_db(ctx, &root);
    let attention_dyn: Option<Arc<dyn AttentionForwardStore>> =
        ctx.map(|c| c.attention.clone() as Arc<dyn AttentionForwardStore>);
    let ambient = threads_db.as_ref().map(|threads| {
        spawn_ambient_daemons(
            &pipeline,
            &store,
            threads,
            attention_dyn.clone(),
            WeaverSettings::resolve(cfg.weaver.as_ref()).exclude_facets,
        )
    });

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

    shutdown_ambient_daemons(ambient).await;

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

    // Clear the per-file cursors too. Deleting chunk rows alone is not
    // enough: the Tier-1 metadata check in `ingest_source` would then see
    // every file as unchanged and skip re-parsing, leaving the project
    // under-filled. Clearing cursors for the project's sources forces the
    // follow-up scan to re-read everything from scratch.
    let mut cursors_cleared = 0u64;
    for source_cfg in &cfg.sources {
        if source_cfg.project.as_deref() == Some(reingest_project) {
            cursors_cleared += ingest
                .clear_source_metadata(&source_cfg.source_config_id)
                .map_err(|e| anyhow!("clear_source_metadata: {e}"))?;
        }
    }

    println!(
        "reingest {reingest_project}: deleted {lance_deleted} corpus rows, {ingest_deleted} ingest rows, cleared {cursors_cleared} source cursors"
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

/// Outcome of a one-shot [`optimize`] call.
pub struct OptimizeOutcome {
    /// Wall-clock duration of the optimize pass.
    pub elapsed: std::time::Duration,
    /// Number of historical versions pruned. `Some` only on the
    /// `--aggressive` path; `None` for the conservative default pass.
    pub versions_pruned: Option<u64>,
}

/// Run Lance's `OptimizeAction::All` against the corpus table —
/// compact fragments, prune old versions, fold appended data into
/// existing scalar / FTS indices. Maintenance entry point for users
/// who want to collapse a long backlog of small commits without
/// running a full re-scan.
pub async fn optimize(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    aggressive: bool,
) -> Result<OptimizeOutcome> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    let store = CorpusStore::open_or_create(&root, embedder.dim())
        .await
        .map_err(|e| anyhow!("open corpus store: {e}"))?;
    let started = std::time::Instant::now();
    let versions_pruned = if aggressive {
        let n = store
            .optimize_compact_and_prune()
            .await
            .map_err(|e| anyhow!("optimize: {e}"))?;
        Some(n)
    } else {
        store
            .optimize_all()
            .await
            .map_err(|e| anyhow!("optimize: {e}"))?;
        None
    };
    Ok(OptimizeOutcome {
        elapsed: started.elapsed(),
        versions_pruned,
    })
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

/// Map the optional `[lens]` config block onto the daemon's runtime
/// `LensConfig`. `None` (no `[lens]` block) yields the daemon
/// defaults. Kept as a free function so the mapping is unit-testable
/// without spinning up `serve`.
fn resolve_lens_config(settings: Option<&LensSettings>) -> LensConfig {
    settings.map_or_else(LensConfig::default, |l| LensConfig {
        token_budget: l.token_budget,
        min_excerpt_tokens: l.min_excerpt_tokens,
        drift_threshold: l.drift_threshold,
        poll_interval_secs: l.poll_interval_secs,
        exclude_facets: l.exclude_facets.clone(),
        candidate_k_per_lane: l.candidate_k_per_lane,
        dominance_threshold: l.dominance_threshold,
        refractory_tau_secs: l.refractory_tau_secs,
        refractory_weight: l.refractory_weight,
    })
}

#[allow(clippy::too_many_lines)]
pub async fn serve(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    stdio: bool,
    watch: bool,
) -> Result<()> {
    use std::io::Write as _;

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

    // Startup assertion: the embedder driving `attend()` (via
    // InMemoryAttention) and the embedder behind QueryEngine MUST
    // produce dimension-compatible vectors with the corpus, or
    // resonance / cosine-based ranking is meaningless. Both
    // QueryEngine and InMemoryAttention below receive the same
    // `Arc<dyn ChunkEmbedder>` clone, so the only way this assertion
    // fails is if the operator pointed a fresh embedder model at a
    // corpus that was indexed with a different model. Fail loud
    // rather than ship silently-wrong scores.
    let corpus_dim = engine.store().dim();
    let embedder_dim = embedder.dim();
    if corpus_dim != embedder_dim {
        return Err(anyhow!(
            "embedder/corpus dimension mismatch: embedder produces {embedder_dim}-dim vectors \
             but corpus at {} was indexed at {corpus_dim} dims — re-index the corpus or \
             configure the original embedder",
            root.display()
        ));
    }

    // Long-lived attention substrate state. The same Arc<ThreadsDb> backs
    // (a) the IdleCurator's periodic tension sweep, (b) every per-trigger
    // scan's ambient daemons (TurnObserver + AutoWeaver), and (c) the
    // attention-mcp dispatch surfaced through the MCP server. Sharing the
    // store means familiarity counters, tension transitions, and proposed
    // threads written by one path are visible to the others without
    // per-call re-open.
    //
    // Cancellation: the curator owns a CancellationToken that fires on
    // process exit (Drop semantics of the local guard). Serve has no
    // graceful-cancel surface yet — signal handling lives in main.rs and
    // is out of scope for this wiring.
    let serve_cancel = CancellationToken::new();
    let serve_ctx: Option<ServeContext> = match open_threads_db_with_chain(&root) {
        Ok(db) => {
            let threads_arc: Arc<ThreadsDb> = Arc::new(db);
            // Share the same embedder Arc with QueryEngine so
            // `attend()`-produced scope vectors are cosine-comparable
            // with corpus chunk embeddings — the prerequisite for
            // embedding-mediated attention bias (focus-feature Phase B).
            let attention_arc: Arc<InMemoryAttention> =
                Arc::new(InMemoryAttention::with_embedder(Arc::clone(&embedder)));

            if let Err(err) =
                replay_chain_into_attention(&threads_arc, attention_arc.as_ref()).await
            {
                tracing::warn!(
                    error = %err,
                    "chain replay failed; in-memory attention starts empty"
                );
            }

            // Re-anchor every thread from its persistent
            // anchor_chunk_id → corpus chunk embedding. This makes
            // resonance scoring meaningful from the first request:
            // chain replay alone leaves materialised threads with
            // empty anchors (familiarize takes its anchor from the
            // scope's attention_vec, which is empty after replay).
            // Sharing the same Arc<dyn ChunkEmbedder> as QueryEngine
            // (asserted above) guarantees these vectors are
            // cosine-comparable with the corpus.
            match re_anchor_threads_from_corpus(
                &threads_arc,
                engine.store(),
                attention_arc.as_ref(),
            )
            .await
            {
                Ok(n) => tracing::info!(seeded = n, "thread anchors re-seeded from corpus"),
                Err(err) => tracing::warn!(
                    error = %err,
                    "re-anchor pass failed; threads start with empty anchors"
                ),
            }

            let attention_dyn: Arc<dyn AttentionForwardStore> = attention_arc.clone();
            let curator = IdleCurator::new(
                Arc::clone(&threads_arc),
                Arc::clone(&attention_dyn),
                CuratorConfig::default(),
            );
            let cancel = serve_cancel.clone();
            tokio::spawn(async move {
                if let Err(err) = curator.run(cancel).await {
                    tracing::warn!(error = %err, "idle curator exited with error");
                }
            });

            tracing::info!("attention substrate online (idle curator + shared threads ledger)");
            Some(ServeContext {
                threads: threads_arc,
                attention: attention_arc,
            })
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "open threads ledger failed; serve runs without attention substrate"
            );
            None
        }
    };

    let dispatch: Option<Arc<AttentionDispatch>> = serve_ctx.as_ref().map(|c| {
        let attention_dyn: Arc<dyn AttentionForwardStore> = c.attention.clone();
        let dispatch = AttentionDispatch::new(attention_dyn, Arc::clone(&c.threads)).with_corpus(
            // Reuse the engine's CorpusStore so the emergent
            // surface reads the same lance corpus the MCP recall
            // verbs query.
            Arc::clone(engine.store()),
        );
        Arc::new(dispatch)
    });

    // Single scan-in-flight mutex shared by the socket listener and the
    // in-process `--watch` watcher, so a watcher-driven scan and an
    // external socket poke never run concurrently on the single-writer
    // corpus.
    let scan_lock: Arc<tokio::sync::Mutex<()>> = Arc::new(tokio::sync::Mutex::new(()));

    // Spawn background scan trigger listener
    let config_path_for_bg = config_path.to_path_buf();
    let embedder_for_bg = Arc::clone(&embedder);
    let ctx_for_bg = serve_ctx.clone();
    let scan_lock_for_bg = Arc::clone(&scan_lock);
    tokio::spawn(async move {
        if let Err(e) = run_socket_listener(
            &sock_path,
            config_path_for_bg,
            embedder_for_bg,
            ctx_for_bg,
            scan_lock_for_bg,
        )
        .await
        {
            tracing::error!(error = %e, "background socket listener failed");
        }
    });

    // `serve --watch`: run the filesystem watcher in-process and deliver
    // debounced batches straight to the scan path (no socket loopback),
    // sharing `scan_lock` with the socket listener so a watcher-driven
    // scan and an external poke never overlap. The watcher is best-effort:
    // if `[watch]` is misconfigured it logs and the daemon carries on.
    if watch {
        let config_path_for_watch = config_path.to_path_buf();
        let sink = ScanTriggerSink::InProcess {
            scan_lock: Arc::clone(&scan_lock),
            embedder: Arc::clone(&embedder),
            ctx: serve_ctx.clone(),
        };
        tokio::spawn(async move {
            if let Err(e) = run_watcher(&config_path_for_watch, sink).await {
                tracing::warn!(error = %e, "in-process watcher exited; daemon continues without it");
            }
        });
        tracing::info!("in-process watcher enabled (serve --watch)");
    }

    // P9b-min — construct the memory-lens registry + resource and,
    // when the attention substrate is online, spawn the background
    // refresh loop. The registry is always handed to Server::
    // with_resources so `resources/list` advertises the memory-lens
    // URI even before any refresh has rendered content.
    let lens_registry = Arc::new(ResourceRegistry::new());
    let lens_resource = Arc::new(MemoryLensResource::new(String::new()));
    lens_registry.register(Arc::clone(&lens_resource) as Arc<dyn ostk_recall_mcp::Resource>);

    let lens_disabled = std::env::var_os("OSTK_RECALL_LENS_DISABLED").is_some();
    if let (Some(ctx), false) = (serve_ctx.as_ref(), lens_disabled) {
        // P9b "Cold-start warmup C1" — encode a single token to
        // force model weights resident before the lens loop's first
        // poll. Best-effort: the embedder swallows internal errors,
        // and a slow first lens isn't fatal.
        let _ = embedder.encode_batch(&["warmup"]);

        let lens_state_dir = root.to_path_buf();
        let initial_state = match load_lens_state(&lens_state_dir) {
            Ok(state) => state.unwrap_or_default(),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "lens_state.json load failed; starting from default state"
                );
                Default::default()
            }
        };

        let attention = Arc::clone(&ctx.attention);
        let corpus = Arc::clone(engine.store());
        let chain_sink_clone: Arc<dyn ChainSink> = ctx.threads.chain_sink();
        // Read handle on the same ledger for the freshness feature + refractory
        // penalty. ThreadsDb implements both ChainSink (write) and
        // ChainLogReader (read); the reader is a fresh-handle clone of the Arc.
        let chain_reader: Arc<dyn ChainLogReader> = ctx.threads.clone();
        // Lens rank engine, built ONCE from the Lens-profile weights
        // (attention_affinity + freshness, overlaid by [ranking.weights.lens])
        // and shared across ticks via Arc — concurrent recall + lens never
        // serialize on it (RankEngine::rank is &self).
        let lens_engine = Arc::new(ostk_recall_query::build_engine_from_weights(
            &cfg.effective_ranking_weights(ostk_recall_core::RankProfile::Lens),
        ));
        let registry = Arc::clone(&lens_registry);
        let resource = Arc::clone(&lens_resource);
        let cancel = serve_cancel.clone();
        let scope = ambient_scope_default();
        let state_dir = lens_state_dir.clone();
        // Map the optional `[lens]` config block onto the daemon's
        // runtime LensConfig; absent block → defaults.
        let lens_config = resolve_lens_config(cfg.lens.as_ref());
        tokio::spawn(async move {
            run_lens_loop(
                attention,
                corpus,
                registry,
                resource,
                chain_sink_clone,
                chain_reader,
                lens_engine,
                lens_config,
                scope,
                state_dir,
                initial_state,
                cancel,
            )
            .await;
        });
        tracing::info!("memory-lens daemon spawned");
    } else if lens_disabled {
        tracing::info!("memory-lens daemon disabled via OSTK_RECALL_LENS_DISABLED");
    }

    tracing::info!(
        model = %engine.model(),
        dim = engine.store().dim(),
        audit = engine.has_audit(),
        attention = dispatch.is_some(),
        stdio,
        "ostk-recall serve starting"
    );
    let server_base = match dispatch {
        Some(d) => Server::new(engine).with_attention(d),
        None => Server::new(engine),
    };
    let server = server_base.with_resources(lens_registry);
    if stdio {
        // Direct stdio transport: this process IS the MCP server, talking
        // JSON-RPC over its own stdin/stdout. Used by a client that spawns
        // `serve --stdio` directly (and by the `connect` bridge's fallback).
        server
            .run_stdio()
            .await
            .map_err(|e| anyhow!("mcp stdio: {e}"))?;
    } else {
        // Standalone daemon: serve MCP to many clients over the
        // cross-platform local endpoint while the background scan-trigger
        // listener, idle curator, and memory-lens loop keep the corpus
        // fresh. All connections share this one read-only engine + lens
        // registry, so lens `resources/updated` notifications fan out to
        // every subscribed client. Block until a shutdown signal so the
        // background tasks stay alive; the `.serve.lock` flock releases on
        // exit regardless.
        let server = Arc::new(server);
        let mcp_sock = mcp_endpoint_path(&root);
        let mcp_server = Arc::clone(&server);
        tokio::spawn(async move {
            if let Err(e) = run_mcp_listener(&mcp_sock, mcp_server).await {
                tracing::error!(error = %e, "mcp listener exited with error");
            }
        });
        tracing::info!(
            "daemon mode: MCP listener + background freshening online; waiting for shutdown signal (Ctrl-C)"
        );
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::warn!(error = %e, "failed to listen for Ctrl-C; shutting down daemon");
        }
    }
    // Cancel the curator/lens on a clean exit so the tokio runtime can drain.
    serve_cancel.cancel();
    tracing::info!(stdio, "ostk-recall serve exited cleanly");
    Ok(())
}

/// Open `ThreadsDb` at `root` wired with a `SqliteChainSink`, so every
/// mutation through this handle is durably appended to the `chain_log`
/// table. The same `threads.sqlite` file holds both the row state and
/// the chain — boot reads chain rows back via `iter_chain` to rebuild
/// the in-memory score tier.
fn open_threads_db_with_chain(root: &Path) -> Result<ThreadsDb> {
    let sink: Arc<dyn ChainSink> =
        Arc::new(SqliteChainSink::open(root).map_err(|e| anyhow!("open chain sink: {e}"))?);
    ThreadsDb::open_with_sink(root, sink).map_err(|e| anyhow!("open threads db: {e}"))
}

/// Read every chain row from the threads ledger in order and replay the
/// reconstructable subset through `InMemoryAttention`. The score tier is
/// per-design rebuildable from the chain — every restart of `serve()`
/// arrives at the same in-memory state.
///
/// V1 mapping (only the events the in-memory runtime can act on):
/// - `ThreadCreate` → no-op (the thread row's anchor is per-scope; the
///   score tier materialises threads lazily via `Fold` / `Familiarize`)
/// - `FamiliarityBatch` → one `ReplayEvent::Familiarize` per entry
///
/// Other chain rows (`TensionTransition`, `EvidenceAdd`, etc.) describe
/// durable ledger state already on disk; they do not contribute score-tier
/// state that needs replay.
async fn replay_chain_into_attention(
    threads: &Arc<ThreadsDb>,
    attention: &InMemoryAttention,
) -> Result<()> {
    use ostk_recall_store::ChainEvent;

    let events = threads
        .iter_chain()
        .map_err(|e| anyhow!("iter_chain: {e}"))?;

    let scope = AttentionScope {
        project: None,
        session_id: Some("replay".into()),
        agent: Some("substrate".into()),
        privacy_tier: PrivacyTier::T1Project,
    };

    let mut replay: Vec<ReplayEvent> = Vec::new();
    // FocusSet rows are applied inline because their replay needs the
    // chain-row vec verbatim (no re-embed), the row's own timestamp,
    // and the chain row's scope (not the synthetic replay scope used
    // by Familiarize). Type inferred from the ChainEvent::FocusSet
    // pattern below.
    let mut focus_events = Vec::new();
    // P6A — RollingVectorSnapshot is idempotent per scope (each
    // event supersedes the previous), so we keep only the latest
    // per `(project, session_id, agent)` triple. Chain rows arrive
    // in chronological order, so HashMap::insert with the same key
    // naturally drops older entries. The vec is replayed verbatim
    // (no re-embed) so stochastic embedders don't drift on boot.
    let mut latest_rolling: std::collections::HashMap<
        (Option<String>, Option<String>, Option<String>),
        (AttentionScope, Vec<f32>),
    > = std::collections::HashMap::new();
    for ev in events {
        match ev {
            ChainEvent::FamiliarityBatch { entries, .. } => {
                // Restore both durable counters verbatim. Each entry is
                // the post-batch (mentions, resonance); replaying them in
                // chronological order means the final batch per handle
                // wins, reconstructing the on-disk counters exactly. We
                // cannot recompute resonance here (the replay scope has no
                // attention vector), so the chain is the source of truth.
                for (handle, mentions, resonance) in entries {
                    replay.push(ReplayEvent::Familiarize {
                        scope: scope.clone(),
                        handle,
                        mentions,
                        resonance,
                    });
                }
            }
            ChainEvent::FocusSet {
                scope: ev_scope,
                query,
                vec,
                ts,
            } => {
                focus_events.push((ev_scope, query, vec, ts));
            }
            ChainEvent::RollingVectorSnapshot {
                scope: ev_scope,
                vec,
                ..
            } => {
                let key = (
                    ev_scope.project.clone(),
                    ev_scope.session_id.clone(),
                    ev_scope.agent.clone(),
                );
                latest_rolling.insert(key, (ev_scope, vec));
            }
            ChainEvent::ThreadCreate { .. }
            | ChainEvent::ThreadRename { .. }
            | ChainEvent::ThreadDelete { .. }
            | ChainEvent::EvidenceAdd { .. }
            | ChainEvent::EvidenceRemove { .. }
            | ChainEvent::EvidenceStateChange { .. }
            | ChainEvent::TensionTransition { .. }
            | ChainEvent::ThreadLinkAdd { .. }
            | ChainEvent::ThreadLinkRemove { .. }
            // P6A: AttentionTurnSkipped is audit-only — no in-memory
            // state to restore for a turn that was rejected.
            | ChainEvent::AttentionTurnSkipped { .. }
            // P9b-min: LensIncluded is audit-only — the lens loop
            // reads recent events for the refractory penalty in
            // P9b-full but replay does not restore any in-memory
            // state. last_portfolio_chunk_ids is carried in
            // lens_state.json instead.
            | ChainEvent::LensIncluded { .. }
            // P7b: access-ledger events are audit-only on replay — the
            // ledger is read on demand by `ChainLogReader::access_history`
            // (for ACT-R freshness), never replayed into in-memory state.
            | ChainEvent::ExplicitRecall { .. }
            | ChainEvent::RecallFault { .. }
            | ChainEvent::OperatorSelected { .. } => {}
        }
    }

    let n = replay.len();
    attention
        .replay(&replay)
        .await
        .map_err(|e| anyhow!("replay: {e}"))?;
    let focus_n = focus_events.len();
    for (ev_scope, query, vec, ts) in focus_events {
        if let Err(err) = attention
            .apply_focus_set_from_chain(&ev_scope, query, vec, ts)
            .await
        {
            tracing::warn!(error = %err, "focus_set replay skipped");
        }
    }
    let rolling_n = latest_rolling.len();
    for (_key, (ev_scope, vec)) in latest_rolling {
        if let Err(err) = attention.seed_rolling_vec(&ev_scope, vec).await {
            tracing::warn!(error = %err, "rolling_vector_snapshot replay skipped");
        }
    }
    tracing::info!(
        events = n,
        focus_events = focus_n,
        rolling_snapshots = rolling_n,
        "chain replay applied to in-memory attention"
    );
    Ok(())
}

/// Re-derive every thread's in-memory anchor from the durable
/// `anchor_chunk_id` → corpus chunk embedding mapping. Run at boot
/// after `replay_chain_into_attention` so the substrate starts with
/// real-embedder anchors instead of:
///  - the empty vectors that chain replay produces (no Attend events
///    are replayed, so scope.attention_vec is empty when familiarize
///    materialises a thread); or
///  - stale stub-embedding anchors left over from a previous run
///    with a different embedder (the focus-feature Phase A prereq).
///
/// Threads without an `anchor_chunk_id` are skipped — there's
/// nothing to derive from. Threads whose anchor_chunk_id no longer
/// exists in the corpus are also skipped (corpus drift); the chain
/// row still attests to the thread's identity, but its anchor
/// remains empty until the operator reassigns or removes it.
///
/// Returns the number of anchors actually installed.
async fn re_anchor_threads_from_corpus(
    threads: &Arc<ThreadsDb>,
    corpus: &Arc<CorpusStore>,
    attention: &InMemoryAttention,
) -> Result<usize> {
    let inputs = threads
        .reanchor_inputs()
        .map_err(|e| anyhow!("reanchor_inputs: {e}"))?;
    if inputs.is_empty() {
        return Ok(0);
    }
    // Only threads without a cached anchor_vec need a corpus lookup; the rest
    // are restored from their durable embedding, independent of corpus churn.
    let need_fetch: Vec<String> = inputs
        .iter()
        .filter(|i| i.anchor_vec.is_none())
        .filter_map(|i| i.anchor_chunk_id.clone())
        .collect();
    let fetched = corpus
        .fetch_embeddings(&need_fetch)
        .await
        .map_err(|e| anyhow!("fetch_embeddings for re-anchor: {e}"))?;

    let mut seeded = 0usize;
    let mut orphaned: Vec<String> = Vec::new();
    for input in inputs {
        // Same single "replay"/"substrate" scope shape as
        // `replay_chain_into_attention`, so boot-materialised threads are
        // visible to cross-scope `surface()`. Carrying `privacy_tier` keeps
        // origin-private threads origin-restricted.
        let scope = AttentionScope {
            project: None,
            session_id: Some("replay".into()),
            agent: Some("substrate".into()),
            privacy_tier: input.privacy_tier,
        };
        // Prefer the cached, corpus-independent anchor vector.
        if let Some(vec) = input.anchor_vec {
            attention
                .seed_anchor(&scope, input.handle, vec)
                .await
                .map_err(|e| anyhow!("seed_anchor: {e}"))?;
            seeded += 1;
            continue;
        }
        // Fall back to the corpus chunk; backfill anchor_vec so the next boot
        // no longer depends on the (content-hash, churnable) chunk id.
        if let Some(vec) = input
            .anchor_chunk_id
            .as_ref()
            .and_then(|id| fetched.get(id))
        {
            attention
                .seed_anchor(&scope, input.handle.clone(), vec.clone())
                .await
                .map_err(|e| anyhow!("seed_anchor: {e}"))?;
            if let Err(e) = threads.set_anchor_vec(&input.handle, vec) {
                tracing::warn!(handle = %input.handle.as_str(), error = %e, "anchor_vec backfill failed");
            }
            seeded += 1;
            continue;
        }
        // Neither a cached vector nor a resolvable chunk: the anchor is
        // orphaned (its chunk's content changed before anchor_vec was cached).
        // Surface it instead of silently dropping it.
        orphaned.push(input.handle.as_str().to_string());
    }
    if !orphaned.is_empty() {
        tracing::warn!(
            count = orphaned.len(),
            handles = %orphaned.join(", "),
            "threads with unresolved anchors (chunk id not in corpus and no cached anchor_vec); \
             re-anchor with `thread create --anchor <chunk_id>`"
        );
    }
    Ok(seeded)
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
    ctx: Option<ServeContext>,
    scan_lock: Arc<tokio::sync::Mutex<()>>,
) -> Result<()> {
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
                    spawn_scan_trigger(&scan_lock, &config_path, &embedder, paths, ctx.clone());
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
                    spawn_scan_trigger(&scan_lock, &config_path, &embedder, paths, ctx.clone());
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

/// Filesystem path of the daemon's MCP endpoint, distinct from the
/// scan-trigger socket (`recall.sock`). On Unix this is the `AF_UNIX`
/// path that [`run_mcp_listener`] binds and the `connect` bridge
/// dials; on Windows it only seeds a distinct named-pipe name via
/// [`pipe_name_for`] (the file itself is never created on disk).
fn mcp_endpoint_path(root: &Path) -> PathBuf {
    root.join("recall-mcp.sock")
}

/// Derive the Windows named-pipe name for a local IPC endpoint from
/// its socket `path`. Unix binds `path` directly as an `AF_UNIX`
/// socket; Windows has no filesystem sockets, so the path's file stem
/// seeds a pipe name instead. The MCP listener and the `connect`
/// bridge both route through this so the two sides agree on the name.
#[cfg(windows)]
fn pipe_name_for(path: &Path, fallback_stem: &str) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(fallback_stem);
    format!(r"\\.\pipe\ostk-recall-{stem}")
}

/// Accept MCP client connections on the daemon's cross-platform local
/// endpoint and serve each one as an independent JSON-RPC session.
///
/// Mirrors [`run_socket_listener`]'s transport split (`AF_UNIX` on
/// Unix, named pipe on Windows) but, instead of reading a one-shot
/// trigger frame, hands each accepted connection's read/write halves
/// to [`Server::serve_with_client`] with a fresh [`ClientId::network`]
/// id. The single `Arc<Server>` is shared across all connections —
/// every connection reads through the same read-only `QueryEngine` and
/// the same lens `ResourceRegistry`, so memory-lens
/// `resources/updated` notifications fan out to every subscribed
/// client. Each connection runs in its own task, so a slow or stuck
/// client can't block the accept loop or its peers.
async fn run_mcp_listener(path: &Path, server: Arc<Server>) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    let next_id = AtomicU64::new(1);

    #[cfg(unix)]
    {
        use tokio::net::UnixListener;
        if path.exists() {
            let _ = std::fs::remove_file(path);
        }
        let listener = UnixListener::bind(path)
            .with_context(|| format!("binding MCP socket {}", path.display()))?;
        tracing::info!(path = %path.display(), "listening for MCP clients (unix socket)");

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let server = Arc::clone(&server);
                    tokio::spawn(async move {
                        let (rd, wr) = stream.into_split();
                        if let Err(e) = server.serve_with_client(ClientId::network(id), rd, wr).await
                        {
                            tracing::warn!(client = id, error = %e, "mcp connection ended with error");
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!(error = %e, "mcp socket accept failed");
                }
            }
        }
    }

    #[cfg(windows)]
    {
        use tokio::net::windows::named_pipe::ServerOptions;
        let pipe_name = pipe_name_for(path, "mcp");
        // first_pipe_instance(true) on the FIRST create only; each accepted
        // connect consumes the current instance, so spin up the next one
        // before serving the connected pipe — that's what lets multiple
        // MCP clients connect simultaneously.
        let mut pipe = ServerOptions::new()
            .first_pipe_instance(true)
            .create(&pipe_name)?;
        tracing::info!(pipe = %pipe_name, "listening for MCP clients (named pipe)");

        loop {
            match pipe.connect().await {
                Ok(()) => {
                    let connected = pipe;
                    pipe = ServerOptions::new().create(&pipe_name)?;
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let server = Arc::clone(&server);
                    tokio::spawn(async move {
                        let (rd, wr) = tokio::io::split(connected);
                        if let Err(e) = server.serve_with_client(ClientId::network(id), rd, wr).await
                        {
                            tracing::warn!(client = id, error = %e, "mcp connection ended with error");
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!(error = %e, "mcp named pipe connect failed");
                    // Re-create the listener so a transient error doesn't
                    // wedge the loop on a half-broken instance.
                    pipe = ServerOptions::new().create(&pipe_name)?;
                }
            }
        }
    }
}

/// Bridge this process's stdin/stdout to a running daemon's MCP
/// endpoint (the `connect` command). A dumb byte pump: it opens no
/// engine, takes no corpus lock, and runs no scan — it just splices
/// bytes so a stdio-only MCP client reaches the shared daemon.
///
/// Fails with an actionable hint when no daemon is listening rather
/// than silently spawning one (a second writer would be refused by the
/// `.serve.lock` singleton guard anyway).
pub async fn connect(config_path: &Path) -> Result<()> {
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config {}", config_path.display()))?;
    let root = cfg.expanded_root().context("resolving corpus.root")?;
    let endpoint = mcp_endpoint_path(&root);

    #[cfg(unix)]
    {
        use tokio::net::UnixStream;
        let stream = UnixStream::connect(&endpoint).await.map_err(|e| {
            anyhow!(
                "connecting to ostk-recall MCP daemon at {}: {e}\n\
                 is a daemon running? start one with `ostk-recall serve --watch`",
                endpoint.display()
            )
        })?;
        let (mut sock_r, mut sock_w) = stream.into_split();
        bridge_stdio(&mut sock_r, &mut sock_w).await
    }

    #[cfg(windows)]
    {
        use tokio::net::windows::named_pipe::ClientOptions;
        let pipe_name = pipe_name_for(&endpoint, "mcp");
        let client = ClientOptions::new().open(&pipe_name).map_err(|e| {
            anyhow!(
                "connecting to ostk-recall MCP daemon pipe {pipe_name}: {e}\n\
                 is a daemon running? start one with `ostk-recall serve --watch`"
            )
        })?;
        let (mut sock_r, mut sock_w) = tokio::io::split(client);
        bridge_stdio(&mut sock_r, &mut sock_w).await
    }
}

/// Bidirectional splice for [`connect`]: copy stdin → socket and
/// socket → stdout concurrently, each running to completion. When
/// stdin hits EOF the socket write half is shut down so the daemon
/// sees the client disconnect (and closes its side, ending the
/// socket→stdout copy); when the daemon closes first, the stdout copy
/// ends and the stdin copy unblocks once the client closes stdin.
/// Neither direction is cut off early, so trailing JSON-RPC frames are
/// not dropped.
async fn bridge_stdio<R, W>(sock_r: &mut R, sock_w: &mut W) -> Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    splice_bidirectional(
        sock_r,
        sock_w,
        &mut tokio::io::stdin(),
        &mut tokio::io::stdout(),
    )
    .await
}

/// Core of [`bridge_stdio`]: copy `input` → `sock_w` and `sock_r` →
/// `output` concurrently, each running to completion. When `input`
/// hits EOF the socket write half is shut down so the daemon's reader
/// sees the disconnect; the reverse copy ends when the daemon closes
/// its side. Neither direction is cut off early, so trailing JSON-RPC
/// frames survive. Split out from the stdin/stdout wiring so it can be
/// driven over in-memory pipes in tests.
async fn splice_bidirectional<R, W, In, Out>(
    sock_r: &mut R,
    sock_w: &mut W,
    input: &mut In,
    output: &mut Out,
) -> Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
    In: tokio::io::AsyncRead + Unpin,
    Out: tokio::io::AsyncWrite + Unpin,
{
    use tokio::io::{AsyncWriteExt, copy};

    let to_sock = async {
        let _ = copy(input, sock_w).await;
        // Half-close the write side so the daemon's reader sees EOF.
        let _ = sock_w.shutdown().await;
    };
    let to_output = async {
        let _ = copy(sock_r, output).await;
        let _ = output.flush().await;
    };
    tokio::join!(to_sock, to_output);
    Ok(())
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
/// Per-source minimum interval between scan-trigger kicks. Paths
/// accumulated during a rate-limited gap are buffered and flushed on the
/// next allowed kick.
const KICK_RATE_LIMIT: Duration = Duration::from_secs(3);

/// Per-source kick gate: rate-limits scan-trigger sends while
/// accumulating any paths suppressed during the gap so the eventual
/// kick covers the full window.
#[derive(Default)]
struct KickGate {
    last_kick: Option<Instant>,
    pending: Vec<PathBuf>,
}

impl KickGate {
    fn add(&mut self, paths: Vec<PathBuf>) {
        self.pending.extend(paths);
    }

    /// Returns Some(paths) if a kick should fire now. `paths` is the
    /// deduped, accumulated set since the last kick; the internal buffer
    /// is cleared on emit.
    fn try_emit(&mut self, now: Instant) -> Option<Vec<PathBuf>> {
        if self.pending.is_empty() {
            return None;
        }
        let allowed = self
            .last_kick
            .is_none_or(|t| now.duration_since(t) >= KICK_RATE_LIMIT);
        if !allowed {
            return None;
        }
        self.last_kick = Some(now);
        let mut seen: HashSet<PathBuf> = HashSet::new();
        let drained = std::mem::take(&mut self.pending);
        let unique = drained
            .into_iter()
            .filter(|p| seen.insert(p.clone()))
            .collect();
        Some(unique)
    }
}

/// Source label used as the `KickGate` key. Falls back to the source
/// kind when the source has no explicit `project`.
fn source_label(source: &SourceConfig) -> String {
    source
        .project
        .clone()
        .unwrap_or_else(|| source.kind.as_str().to_string())
}

/// Where a watcher delivers a debounced scan-trigger batch.
///
/// The standalone `watch` command writes to the daemon's scan-trigger
/// socket ([`ScanTriggerSink::Socket`]); `serve --watch` runs the same
/// loop in-process and hands batches straight to [`spawn_scan_trigger`]
/// ([`ScanTriggerSink::InProcess`]), skipping the socket loopback and
/// sharing the daemon's scan mutex so the two never scan at once.
enum ScanTriggerSink {
    Socket,
    InProcess {
        scan_lock: Arc<tokio::sync::Mutex<()>>,
        embedder: Arc<dyn ChunkEmbedder>,
        ctx: Option<ServeContext>,
    },
}

impl ScanTriggerSink {
    const fn is_in_process(&self) -> bool {
        matches!(self, Self::InProcess { .. })
    }
}

/// Deliver one debounced `frame` to the configured sink. The socket
/// arm awaits the kick (and can fail on a transient socket error); the
/// in-process arm fires [`spawn_scan_trigger`] and returns immediately
/// (the spawned task drops itself if the shared scan lock is held).
async fn dispatch_scan_trigger(
    sink: &ScanTriggerSink,
    socket_path: &Path,
    config_path: &Path,
    frame: &[PathBuf],
) -> Result<()> {
    match sink {
        ScanTriggerSink::Socket => kick_trigger_socket(socket_path, frame).await,
        ScanTriggerSink::InProcess {
            scan_lock,
            embedder,
            ctx,
        } => {
            spawn_scan_trigger(scan_lock, config_path, embedder, frame.to_vec(), ctx.clone());
            Ok(())
        }
    }
}

/// Standalone `watch` command: drive [`run_watcher`] delivering pokes
/// over the scan-trigger socket to a separately-running `serve`.
pub async fn watch(config_path: &Path) -> Result<()> {
    run_watcher(config_path, ScanTriggerSink::Socket).await
}

/// Filesystem-watch loop shared by the standalone `watch` command and
/// the in-process watcher spawned by `serve --watch`. `sink` selects
/// socket delivery (separate process) vs. direct in-process scans.
#[allow(clippy::too_many_lines)]
async fn run_watcher(config_path: &Path, sink: ScanTriggerSink) -> Result<()> {
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

    // Register each root once as Recursive and let the OS deliver every
    // event in the subtree. `notify::RecursiveMode::NonRecursive` on
    // macOS is a client-side filter, not an OS-level subscription bound
    // — every descendant event still arrives and is rejected per-event
    // inside the notify callback. Per-leaf NonRecursive (commit 14d7434)
    // *grew* notify's filter HashMap from O(roots) to O(all-source-dirs),
    // costing millions of `starts_with` ops/sec inside the FFI callback
    // during cargo-build storms. Recursive-per-root collapses that work;
    // the cheap event-side `path_has_noise_segment` filter (below) drops
    // `target/`/`node_modules/` events at O(NOISE_PATH_SEGMENTS) each.
    for (root, _source) in &watched_roots {
        let mode = if root.is_file() {
            RecursiveMode::NonRecursive
        } else {
            RecursiveMode::Recursive
        };
        debouncer
            .watch(root, mode)
            .with_context(|| format!("registering watch on {}", root.display()))?;
        tracing::info!(
            path = %root.display(),
            recursive = !root.is_file(),
            "watching"
        );
    }
    tracing::info!(
        socket = %socket_path.display(),
        debounce_ms = watch_cfg.debounce_ms,
        roots = watched_roots.len(),
        in_process = sink.is_in_process(),
        "scan-trigger watcher started"
    );

    let mode = watch_cfg.mode;
    let mut kick_gates: HashMap<String, KickGate> = HashMap::new();
    // 1 Hz wake-up so kick gates with pending paths flush even when no
    // new events arrive within the rate-limit window.
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;
            maybe = rx.recv() => {
                let Some(result) = maybe else { break };
                match result {
                    Ok(events) => {
                        // Recursive-per-root means FSEvents delivers every
                        // descendant event; the cheap segment-match filter
                        // below drops `target/`/`node_modules/` etc. at
                        // O(NOISE_PATH_SEGMENTS) per event. No dynamic
                        // registration or ignorefile rewalk needed — the
                        // OS-level subscription covers the whole subtree.

                        // Collect matched paths grouped by source.
                        // `matches_watched_root` already short-circuits on
                        // noise segments, so node_modules/etc. events that
                        // somehow slipped through (e.g. before a `.gitignore`
                        // rewalk completed) still get dropped here.
                        let mut by_source: HashMap<String, Vec<PathBuf>> = HashMap::new();
                        for ev in &events {
                            for p in &ev.event.paths {
                                if path_has_noise_segment(p) {
                                    continue;
                                }
                                for (root, source) in &watched_roots {
                                    if !p.starts_with(root) {
                                        continue;
                                    }
                                    let ext_ok = source.extensions.is_empty()
                                        || p.extension()
                                            .and_then(|e| e.to_str())
                                            .is_some_and(|ext| {
                                                source.extensions.iter().any(|x| x == ext)
                                            });
                                    if ext_ok {
                                        by_source
                                            .entry(source_label(source))
                                            .or_default()
                                            .push(p.clone());
                                        break;
                                    }
                                }
                            }
                        }

                        // Phase 4: rate-limited kicks per source. Paths
                        // suppressed during the rate-limit gap accumulate
                        // in the gate so the eventual kick covers the
                        // whole window.
                        let now = Instant::now();
                        for (label, paths) in by_source {
                            let gate = kick_gates.entry(label.clone()).or_default();
                            gate.add(paths);
                            if let Some(to_send) = gate.try_emit(now) {
                                let frame: &[PathBuf] = match mode {
                                    WatchMode::Legacy => &[],
                                    WatchMode::Incremental => &to_send,
                                };
                                if let Err(e) =
                                    dispatch_scan_trigger(&sink, &socket_path, config_path, frame)
                                        .await
                                {
                                    tracing::warn!(
                                        socket = %socket_path.display(),
                                        label = %label,
                                        error = %e,
                                        "scan-trigger kick failed; will retry on next event"
                                    );
                                } else {
                                    tracing::info!(
                                        label = %label,
                                        paths = frame.len(),
                                        "scan-trigger kicked"
                                    );
                                }
                            }
                        }
                    }
                    Err(errors) => {
                        for e in errors {
                            tracing::warn!(error = %e, "watch backend error");
                        }
                    }
                }
            }
            _ = tick.tick() => {
                // Wake-up flush: catches paths that landed in a kick gate
                // during a rate-limited gap when no new events arrived to
                // re-drive the loop.
                let now = Instant::now();
                for (label, gate) in &mut kick_gates {
                    let Some(to_send) = gate.try_emit(now) else { continue };
                    let frame: &[PathBuf] = match mode {
                        WatchMode::Legacy => &[],
                        WatchMode::Incremental => &to_send,
                    };
                    if let Err(e) =
                        dispatch_scan_trigger(&sink, &socket_path, config_path, frame).await
                    {
                        tracing::warn!(
                            socket = %socket_path.display(),
                            label = %label,
                            error = %e,
                            "scan-trigger flush-kick failed; will retry on next tick"
                        );
                    } else {
                        tracing::info!(
                            label = %label,
                            paths = frame.len(),
                            "scan-trigger kicked (flush)"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// Path segments that should never trigger a scan even when they
/// fall under a watched root. The depth-1 ignore filter applied at
/// debouncer registration catches direct children, but nested
/// occurrences (e.g. `<project>/shared/node_modules/...`) ride along
/// under the parent's recursive watch and produce a flood of
/// scan-trigger kicks for content that `scan()` would itself reject.
///
/// Matching is segment-exact (`node_modules` matches `/foo/node_modules/bar`
/// but not `/foo/anode_modules/bar`).
const NOISE_PATH_SEGMENTS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    ".ostk",
    ".worktrees",
    ".next",
    "dist",
    "build",
];

/// True if any component of `path` matches a noise segment exactly.
fn path_has_noise_segment(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .is_some_and(|s| NOISE_PATH_SEGMENTS.contains(&s))
    })
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
    ctx: Option<ServeContext>,
) {
    let scan_lock = Arc::clone(scan_lock);
    let config_path = config_path.to_path_buf();
    let embedder = Arc::clone(embedder);
    tokio::spawn(async move {
        let Ok(guard) = scan_lock.try_lock() else {
            tracing::warn!("scan trigger received while a scan is already in flight; dropping");
            return;
        };
        let ctx_ref = ctx.as_ref();
        let result = if paths.is_empty() {
            tracing::info!("scan trigger received (legacy scan-all)");
            scan_with_context(&config_path, embedder, None, false, ctx_ref)
                .await
                .map(|_| ())
        } else {
            tracing::info!(paths = paths.len(), "scan trigger received (per-path)");
            scan_paths_with_context(&config_path, embedder, &paths, false, ctx_ref)
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

#[cfg(test)]
mod lens_config_tests {
    use super::{LensSettings, resolve_lens_config};

    /// The `[lens]` config defaults are duplicated in `core` (serde
    /// can't reach across to `query`). This guard fails loudly if the
    /// two ever drift, which would silently change daemon behavior for
    /// users who omit the block.
    #[test]
    fn lens_settings_default_matches_query_default() {
        let mapped = resolve_lens_config(Some(&LensSettings::default()));
        let dflt = ostk_recall_query::lens::LensConfig::default();
        assert_eq!(mapped.token_budget, dflt.token_budget);
        assert_eq!(mapped.min_excerpt_tokens, dflt.min_excerpt_tokens);
        assert!((mapped.drift_threshold - dflt.drift_threshold).abs() < f32::EPSILON);
        assert_eq!(mapped.poll_interval_secs, dflt.poll_interval_secs);
        assert_eq!(mapped.exclude_facets, dflt.exclude_facets);
        assert_eq!(mapped.candidate_k_per_lane, dflt.candidate_k_per_lane);
        assert!((mapped.dominance_threshold - dflt.dominance_threshold).abs() < f32::EPSILON);
        assert_eq!(mapped.refractory_tau_secs, dflt.refractory_tau_secs);
        assert!((mapped.refractory_weight - dflt.refractory_weight).abs() < f32::EPSILON);
    }

    /// Absent `[lens]` block resolves to the daemon default verbatim.
    #[test]
    fn resolve_lens_config_none_is_default() {
        let mapped = resolve_lens_config(None);
        let dflt = ostk_recall_query::lens::LensConfig::default();
        assert_eq!(mapped.token_budget, dflt.token_budget);
        assert_eq!(mapped.poll_interval_secs, dflt.poll_interval_secs);
    }

    /// Explicit overrides flow through.
    #[test]
    fn resolve_lens_config_applies_overrides() {
        let settings = LensSettings {
            token_budget: 1234,
            poll_interval_secs: 9,
            exclude_facets: vec!["status:archived".into()],
            ..LensSettings::default()
        };
        let mapped = resolve_lens_config(Some(&settings));
        assert_eq!(mapped.token_budget, 1234);
        assert_eq!(mapped.poll_interval_secs, 9);
        assert_eq!(mapped.exclude_facets, vec!["status:archived".to_string()]);
    }
}

#[cfg(test)]
mod force_wipe_tests {
    use super::force_wipe_corpus;
    use std::fs;
    use tempfile::TempDir;

    /// Regression: `init --force` must remove SQLite WAL/SHM sidecars,
    /// not just the main `.sqlite`. An orphaned `-wal` (main file gone)
    /// breaks the next open — read-only fails outright (crashing `serve`
    /// → MCP `-32000`), read-write can abort mid-scan.
    #[test]
    fn force_wipe_removes_sqlite_wal_shm_sidecars() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        let files = [
            "events.sqlite",
            "events.sqlite-wal",
            "events.sqlite-shm",
            "ingest.sqlite",
            "ingest.sqlite-wal",
            "ingest.sqlite-shm",
        ];
        for f in files {
            fs::write(root.join(f), b"x").unwrap();
        }
        force_wipe_corpus(root).unwrap();
        for f in files {
            assert!(!root.join(f).exists(), "{f} should have been removed");
        }
    }
}

#[cfg(test)]
mod daemon_transport_tests {
    use super::{mcp_endpoint_path, splice_bidirectional};
    use std::path::Path;

    /// The daemon's MCP endpoint must not collide with the scan-trigger
    /// socket: they're two independent listeners on the same corpus
    /// root, and on Windows the distinct file stems seed distinct named
    /// pipes. A collision would wedge one listener on the other's bind.
    #[test]
    fn mcp_endpoint_distinct_from_trigger_socket() {
        let root = Path::new("/tmp/ostk-corpus");
        let mcp = mcp_endpoint_path(root);
        assert_eq!(mcp, root.join("recall-mcp.sock"));
        assert_ne!(
            mcp,
            root.join("recall.sock"),
            "MCP endpoint must differ from the scan-trigger socket"
        );
    }

    /// The `connect` bridge core must pump bytes in BOTH directions and
    /// not drop the daemon's reply: client stdin → socket, and the
    /// daemon's response → client stdout.
    #[tokio::test]
    async fn splice_bidirectional_pumps_both_directions() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        // Stand-in for the daemon socket: a duplex whose far end is the
        // "daemon" task.
        let (sock_client, daemon_end) = tokio::io::duplex(1024);
        let (mut sock_r, mut sock_w) = tokio::io::split(sock_client);
        let (mut daemon_r, mut daemon_w) = tokio::io::split(daemon_end);

        // Client "stdin": test writes a request, then EOF (drop).
        let (mut stdin_w, mut stdin_r) = tokio::io::duplex(1024);
        // Client "stdout": splice writes here, test reads it back.
        let (mut stdout_w, mut stdout_r) = tokio::io::duplex(1024);

        // Daemon: read the request, echo a reply, then close its side.
        let daemon = tokio::spawn(async move {
            let mut buf = vec![0u8; 5];
            daemon_r.read_exact(&mut buf).await.unwrap();
            daemon_w.write_all(b"pong\n").await.unwrap();
            buf
            // daemon_r / daemon_w drop here → socket EOF to the client.
        });

        stdin_w.write_all(b"ping\n").await.unwrap();
        drop(stdin_w);

        splice_bidirectional(&mut sock_r, &mut sock_w, &mut stdin_r, &mut stdout_w)
            .await
            .unwrap();

        assert_eq!(&daemon.await.unwrap(), b"ping\n", "request reached daemon");

        // Close the write side so read_to_end on the client stdout sees EOF.
        drop(stdout_w);
        let mut got = Vec::new();
        stdout_r.read_to_end(&mut got).await.unwrap();
        assert_eq!(got, b"pong\n", "daemon reply reached client stdout");
    }
}
