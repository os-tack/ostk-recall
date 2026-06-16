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
    AttentionForwardStore, AutoWeaver, ConceptGrowthConfig, ConceptGrowthRuntime, CuratorConfig,
    IdleCurator, InMemoryAttention, ReplayEvent, TurnObserver, WeaverThresholds,
    ambient_scope_default,
};
use ostk_recall_attention_mcp::AttentionDispatch;
use ostk_recall_core::attention::{AttentionScope, PrivacyTier};
use ostk_recall_core::{
    AmbientGrowthConfig, Chunk, CompiledRecordRules, Config, LensSettings, RelationalConfig,
    RerankerConfig, SalienceSettings, Scanner, SourceConfig, SourceKind, WatchMode, WeaverSettings,
};
use ostk_recall_mcp::{ClientId, ResourceRegistry, Server};
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline, PipelineStats, VerifyReport};
use ostk_recall_query::lens::LensConfig;
use ostk_recall_query::{QueryEngine, Reranker, RerankerLike};

use crate::lens_loop::{MemoryLensResource, load_lens_markdown, run_lens_loop};
use crate::lens_state::load_lens_state;
use ostk_recall_scan::claude_code::ClaudeCodeScanner;
use ostk_recall_scan::code::CodeScanner;
use ostk_recall_scan::codex::CodexScanner;
use ostk_recall_scan::file_glob::FileGlobScanner;
use ostk_recall_scan::gemini::GeminiScanner;
use ostk_recall_scan::markdown::MarkdownScanner;
use ostk_recall_scan::ostk_project::OstkProjectScanner;
use ostk_recall_scan::threads::ThreadScanner;
use ostk_recall_scan::zip_export::ZipExportScanner;
use ostk_recall_store::{
    ChainLogReader, ChainSink, CorpusStore, EventsDb, IngestChunkRow, IngestDb, SqliteChainSink,
    ThreadsDb,
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
    /// Long-lived concept-growth state, shared across every per-trigger
    /// `TurnObserver` so node-recurrence accumulates across separate live
    /// turns (each watch trigger spawns a fresh observer). Created once at
    /// `serve` boot; threaded into `spawn_ambient_daemons`.
    pub concept_growth: ConceptGrowthRuntime,
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
    // FTS (inverted) index on `text` — hybrid recall's BM25 leg cannot run
    // without it. `init` builds it once at create time, but a freshly-created
    // corpus is empty then, so the index covers zero rows; the first populated
    // scan must (re)build it. Omitting this call here is exactly what stranded
    // the v0.8.0 rebuilt corpus with no inverted index — every recall errored
    // "Cannot perform full text search unless an INVERTED index has been
    // created". `ensure_fts_index` is idempotent (no-op once a `text` index
    // exists), so running it every scan is cheap.
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
    weaver_stop_handles: Vec<String>,
    concept_growth: ConceptGrowthConfig,
    growth_runtime: Option<ConceptGrowthRuntime>,
) -> AmbientHandles {
    let mut observer = TurnObserver::new(Arc::clone(pipeline), Arc::clone(threads))
        .with_stop_handles(weaver_stop_handles.clone());
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
    // Phase 2 — the concept-growth phase needs the corpus (anchor codebook)
    // and the resolved `[ambient_growth]` tuning. Gated on `with_corpus` +
    // `with_attention`; both are present on the live `serve` path.
    observer = observer
        .with_corpus(Arc::clone(corpus))
        .with_concept_growth(concept_growth);
    // Share the long-lived concept-growth state (from `ServeContext`) so
    // node-recurrence persists across the fresh observer each watch trigger
    // spawns. `None` on one-shot scan paths → fresh per-scan state.
    if let Some(runtime) = growth_runtime {
        observer = observer.with_concept_growth_runtime(runtime);
    }
    let observer = Arc::new(observer);
    let weaver = Arc::new(
        AutoWeaver::new(
            Arc::clone(threads),
            Arc::clone(corpus),
            WeaverThresholds::default(),
        )
        .with_exclude_facets(weaver_exclude_facets)
        .with_stop_handles(weaver_stop_handles),
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
        Pipeline::new(
            Arc::clone(&store),
            Arc::clone(&ingest),
            Arc::clone(&embedder),
        )
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
            WeaverSettings::resolve(cfg.weaver.as_ref()).stop_handles,
            resolve_concept_growth(cfg.ambient_growth.as_ref()),
            ctx.map(|c| c.concept_growth.clone()),
        )
    });

    let markdown = MarkdownScanner;
    let code = CodeScanner;
    let claude_code = ClaudeCodeScanner;
    let codex = CodexScanner;
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
    // Slice-4 mention-linking runs after the loop over the sources that were
    // actually scanned (honoring `--source`), so the gazetteer is complete.
    let mut scanned_entity_sources: Vec<&SourceConfig> = Vec::new();
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

        // Graph-only source (relational-substrate doc-topology harvest): never
        // ingested/embedded (an ingesting source already owns these chunks) and
        // never added to the gazetteer mention pass. Harvest the doc nodes +
        // authorial link graph straight into the ledger, then skip ingest.
        //
        // Full-scan only: `--source` filters by project, and a graph-only docs
        // source shares its project with the ingesting source over the same
        // tree, so a project-scoped `--source <project>` would mutate the doc
        // graph as a side-effect of an unrelated scan. Gate on no filter;
        // always `continue` so the source never ingests regardless.
        if source_cfg.graph_only {
            if source_filter.is_none() && !dry_run {
                if let Some(threads) = threads_db.as_ref() {
                    let s =
                        crate::seed::harvest_doc_graph_from_source(threads, &ingest, source_cfg);
                    if s.nodes_seeded + s.edges_seeded > 0 {
                        tracing::info!(
                            source = %label,
                            nodes = s.nodes_seeded,
                            edges = s.edges_seeded,
                            evidence = s.evidence_attached,
                            "harvested doc-topology graph"
                        );
                    }
                }
            }
            continue;
        }

        let scanner: &dyn Scanner = match source_cfg.kind {
            SourceKind::Markdown => &markdown,
            SourceKind::Code => &code,
            SourceKind::ClaudeCode => &claude_code,
            SourceKind::Codex => &codex,
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
        per_source.push((label.clone(), stats));

        // Relational-substrate slice 3: seed typed nodes + authored edges
        // from this source's markdown files (only when it declares an
        // `entity_type`). Synchronous ledger pass — the Pipeline has no
        // ThreadsDb. Idempotent, so re-scans are safe.
        if source_cfg.entity_type.is_some() {
            scanned_entity_sources.push(source_cfg);
            if !dry_run {
                if let Some(threads) = threads_db.as_ref() {
                    let s = crate::seed::seed_nodes_from_source(threads, source_cfg);
                    if s.nodes_seeded + s.edges_seeded > 0 {
                        tracing::info!(
                            source = %label, nodes = s.nodes_seeded, edges = s.edges_seeded,
                            "seeded graph nodes/edges"
                        );
                    }
                }
            }
        }
    }

    // Relational-substrate slice 4: link prose mentions once node-seeding has
    // run for every scanned source, so the gazetteer is complete (e.g. a
    // person doc can mention a meeting defined in another source).
    if !dry_run && !scanned_entity_sources.is_empty() {
        if let Some(threads) = threads_db.as_ref() {
            let m = crate::seed::link_mentions_from_sources(threads, &scanned_entity_sources);
            if m.mentions_linked > 0 {
                tracing::info!(
                    files = m.files_scanned,
                    mentions = m.mentions_linked,
                    ambiguous = m.ambiguous_skipped,
                    "linked prose mentions"
                );
            }
        }
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
        .with_exclude_facets(WeaverSettings::resolve(cfg.weaver.as_ref()).exclude_facets)
        .with_stop_handles(WeaverSettings::resolve(cfg.weaver.as_ref()).stop_handles);
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
        .with_exclude_facets(WeaverSettings::resolve(cfg.weaver.as_ref()).exclude_facets)
        .with_stop_handles(WeaverSettings::resolve(cfg.weaver.as_ref()).stop_handles);
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
        Pipeline::new(
            Arc::clone(&store),
            Arc::clone(&ingest),
            Arc::clone(&embedder),
        )
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
            WeaverSettings::resolve(cfg.weaver.as_ref()).stop_handles,
            resolve_concept_growth(cfg.ambient_growth.as_ref()),
            ctx.map(|c| c.concept_growth.clone()),
        )
    });

    let markdown = MarkdownScanner;
    let code = CodeScanner;
    let claude_code = ClaudeCodeScanner;
    let codex = CodexScanner;
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
        // Graph-only doc-topology sources never ingest, and Phase 1 does not
        // auto-harvest them on the incremental path (full-scan only); exclude
        // them from the changed-path ingest fan-out entirely.
        .filter(|source_cfg| !source_cfg.graph_only)
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
                SourceKind::Codex => &codex,
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

    // Relational-substrate slice 3: seed nodes/edges for the changed paths
    // that fall under an `entity_type` source. Mirrors scan_with_context but
    // scoped to `paths` so live edits seed without rewalking every dir.
    if !dry_run {
        if let Some(threads) = threads_db.as_ref() {
            let mut scanned_entity_sources: Vec<&SourceConfig> = Vec::new();
            for (_, source_cfg) in &pairs {
                if source_cfg.entity_type.is_some() {
                    scanned_entity_sources.push(*source_cfg);
                    let s = crate::seed::seed_nodes_for_paths(threads, source_cfg, paths);
                    if s.nodes_seeded + s.edges_seeded > 0 {
                        tracing::info!(
                            nodes = s.nodes_seeded,
                            edges = s.edges_seeded,
                            "seeded graph nodes/edges (incremental)"
                        );
                    }
                }
            }
            // Slice 4: link prose mentions for the changed paths against the
            // now-complete gazetteer (reuses the same delete/ignore guards).
            if !scanned_entity_sources.is_empty() {
                let m =
                    crate::seed::link_mentions_for_paths(threads, &scanned_entity_sources, paths);
                if m.mentions_linked > 0 {
                    tracing::info!(
                        files = m.files_scanned,
                        mentions = m.mentions_linked,
                        ambiguous = m.ambiguous_skipped,
                        "linked prose mentions (incremental)"
                    );
                }
            }
        }
    }

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

/// Outcome of a [`recover_orphans`] pass.
pub struct RecoverOrphansOutcome {
    /// Total rows in the backup corpus.
    pub backup_rows: usize,
    /// Total rows in the live corpus (before recovery).
    pub live_rows: usize,
    /// `chunk_id`s present in the backup but absent from the live corpus.
    pub orphans: usize,
    /// Rows actually re-ingested into the live corpus (0 on a dry run).
    pub recovered: usize,
    /// Whether this was a dry run (diff only, no writes).
    pub dry_run: bool,
    /// Wall-clock duration of the pass.
    pub elapsed: std::time::Duration,
}

/// Recover backup-only corpus rows into the live corpus.
///
/// Targets chunks whose source files were rotated away (e.g. expired Claude
/// transcripts), for which the backup is the only surviving copy — the
/// "reindex missing data from the original" half of a backup → wipe → rescan
/// → recover rebuild.
///
/// It diffs `chunk_id`s, reads the backup-only rows' full content (including
/// their stored embeddings), and `append`s them to the live corpus —
/// preserving each chunk's original `chunk_id` / `source_config_id` and its
/// vector. Because the orphans are provably absent from the live corpus and the
/// backup was embedded by the same model, this copies vectors and appends (no
/// re-embed, no merge scan), staying O(rows) instead of O(rows²). Idempotent:
/// re-running re-diffs, so already-recovered rows are no longer orphans.
///
/// `from` is the **recall data root** of the backup (the directory that
/// contains `corpus.lance/`), not the `.lance` directory itself. Run with
/// `serve`/`watch` stopped — this mutates the live corpus.
pub async fn recover_orphans(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    from: &Path,
    batch: usize,
    dry_run: bool,
) -> Result<RecoverOrphansOutcome> {
    let started = Instant::now();
    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    if from == root {
        bail!(
            "--from ({}) must differ from the live corpus root ({})",
            from.display(),
            root.display()
        );
    }
    if !from.join("corpus.lance").exists() {
        bail!(
            "no corpus.lance under --from {} (point it at the backup's recall data root)",
            from.display()
        );
    }

    let live = CorpusStore::open_or_create(&root, embedder.dim())
        .await
        .map_err(|e| anyhow!("open live corpus: {e}"))?;
    let backup = CorpusStore::open_or_create(from, embedder.dim())
        .await
        .map_err(|e| anyhow!("open backup corpus {}: {e}", from.display()))?;

    // Diff chunk_ids: orphans = backup-only.
    let backup_ids = backup
        .all_chunk_ids()
        .await
        .map_err(|e| anyhow!("read backup chunk ids: {e}"))?;
    let live_ids: HashSet<String> = live
        .all_chunk_ids()
        .await
        .map_err(|e| anyhow!("read live chunk ids: {e}"))?
        .into_iter()
        .collect();
    let backup_rows = backup_ids.len();
    let live_rows = live_ids.len();
    let orphan_ids: Vec<String> = backup_ids
        .into_iter()
        .filter(|id| !live_ids.contains(id))
        .collect();
    let orphans = orphan_ids.len();

    if dry_run || orphans == 0 {
        return Ok(RecoverOrphansOutcome {
            backup_rows,
            live_rows,
            orphans,
            recovered: 0,
            dry_run,
            elapsed: started.elapsed(),
        });
    }

    let ingest = IngestDb::open(&root).map_err(|e| anyhow!("open ingest db: {e}"))?;
    let dim = embedder.dim();

    let mut recovered = 0usize;
    let mut skipped = 0usize;
    for id_batch in orphan_ids.chunks(batch) {
        let fetched = backup
            .fetch_chunks_by_ids(id_batch)
            .await
            .map_err(|e| anyhow!("fetch orphan chunks from backup: {e}"))?;
        // Copy the backup's stored vector and `append` (no merge scan):
        // orphans are provably absent from the live corpus, so this is a plain
        // O(rows) insert. The backup was embedded by the same model, so the
        // vector is valid as-is — no re-embed needed. Rows whose backup vector
        // is missing or the wrong dim are skipped (a model change calls for a
        // full re-scan, not orphan recovery).
        let mut chunks: Vec<Chunk> = Vec::with_capacity(fetched.len());
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(fetched.len());
        for (chunk, embedding) in fetched.into_values() {
            match embedding {
                Some(e) if e.len() == dim => {
                    chunks.push(chunk);
                    embs.push(e);
                }
                _ => skipped += 1,
            }
        }
        if chunks.is_empty() {
            continue;
        }
        live.append(&chunks, &embs)
            .await
            .map_err(|e| anyhow!("append recovered rows: {e}"))?;
        // Mirror the rows into the ingest ledger so `verify` reconciles and a
        // later scan dedupes against them.
        for chunk in &chunks {
            let row = IngestChunkRow {
                chunk_id: chunk.chunk_id.clone(),
                source: chunk.source.as_str().to_string(),
                source_id: chunk.source_id.clone(),
                source_config_id: chunk.source_config_id.clone(),
                chunk_index: chunk.chunk_index,
                content_sha256: chunk.sha256.clone(),
                embedding_input_sha256: chunk.embedding_input_sha256.clone(),
            };
            ingest
                .record_chunk(&row, Some("recover-orphans"))
                .map_err(|e| anyhow!("record ingest row: {e}"))?;
        }
        recovered += chunks.len();
    }
    if skipped > 0 {
        tracing::warn!(skipped, "orphans skipped (backup vector missing or wrong dim)");
    }

    Ok(RecoverOrphansOutcome {
        backup_rows,
        live_rows,
        orphans,
        recovered,
        dry_run,
        elapsed: started.elapsed(),
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
    let mut engine = QueryEngine::new(store, ingest, events, embedder, cfg.embedder.model.clone())
        .with_corpus_root(root.clone());
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
fn resolve_lens_config(
    settings: Option<&LensSettings>,
    relational: Option<&RelationalConfig>,
) -> LensConfig {
    // The latent floor comes from `[relational]`, not `[lens]` — fold it in
    // after mapping the lens block (default from the canonical core config).
    let mut cfg = settings.map_or_else(LensConfig::default, |l| LensConfig {
        token_budget: l.token_budget,
        min_excerpt_tokens: l.min_excerpt_tokens,
        drift_threshold: l.drift_threshold,
        poll_interval_secs: l.poll_interval_secs,
        exclude_facets: l.exclude_facets.clone(),
        candidate_k_per_lane: l.candidate_k_per_lane,
        dominance_threshold: l.dominance_threshold,
        refractory_tau_secs: l.refractory_tau_secs,
        refractory_weight: l.refractory_weight,
        latent_sim_floor: RelationalConfig::default().latent_sim_floor,
    });
    if let Some(r) = relational {
        cfg.latent_sim_floor = r.latent_sim_floor;
    }
    cfg
}

/// Map the optional `[ambient_growth]` config block onto the observer's runtime
/// [`ConceptGrowthConfig`]. `None` (no block) yields the calibrated defaults.
/// Free function so the mapping is unit-testable; a guard test pins the
/// `AmbientGrowthConfig` ↔ `ConceptGrowthConfig` defaults in lock-step (`core`
/// cannot depend on `attention`).
fn resolve_concept_growth(settings: Option<&AmbientGrowthConfig>) -> ConceptGrowthConfig {
    settings.map_or_else(ConceptGrowthConfig::default, |a| ConceptGrowthConfig {
        resonance_floor: a.resonance_floor,
        edge_top_k: a.edge_top_k,
        min_survivors: a.min_survivors,
        node_mint_min_resonant_turns: a.node_mint_min_resonant_turns,
        codebook_rebuild_turns: a.codebook_rebuild_turns,
        node_mint_cap_per_session: a.node_mint_cap_per_session,
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
            let salience_settings = SalienceSettings::resolve(cfg.salience.as_ref());
            let attention_arc: Arc<InMemoryAttention> = Arc::new(
                InMemoryAttention::with_embedder(Arc::clone(&embedder))
                    .with_stop_handles(WeaverSettings::resolve(cfg.weaver.as_ref()).stop_handles)
                    .with_salience_settings(&salience_settings),
            );

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
                &salience_settings,
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
                concept_growth: ConceptGrowthRuntime::default(),
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

    // →1957/→1958 seam: liveness + freshness sidecar for `ostk ps` —
    // a zero-IPC attestation surface at `<corpus_root>/stats.json`.
    // Contract v1 (p48651): `written_at` + `scan_cadence_secs` are
    // REQUIRED — without them a wedged serve would freeze the sidecar
    // at last-good values and read healthy forever (→1947 one level
    // up). Cadence here is the attestation-WRITE cadence (60s timer,
    // not per-scan): a wedged daemon stops the timer, written_at ages
    // past 2× cadence, and ps renders the seat wedged.
    {
        let ingest = Arc::clone(engine.ingest());
        let events = engine.events().cloned();
        let sidecar_root = root.clone();
        let started_at = chrono::Utc::now().to_rfc3339();
        const SIDECAR_CADENCE_SECS: u64 = 60;
        tokio::spawn(async move {
            let mut tick =
                tokio::time::interval(std::time::Duration::from_secs(SIDECAR_CADENCE_SECS));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                let last_scan_at = ingest.latest_upserted_at().ok().flatten();
                let audit_newest_ts: Vec<serde_json::Value> = events
                    .as_ref()
                    .and_then(|db| db.newest_ts_by_project().ok())
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(project, newest_ts)| {
                        serde_json::json!({ "project": project, "newest_ts": newest_ts })
                    })
                    .collect();
                let body = serde_json::json!({
                    "v": 1,
                    "pid": std::process::id(),
                    "started_at": started_at,
                    "host_build": env!("CARGO_PKG_VERSION"),
                    "last_scan_at": last_scan_at,
                    "written_at": chrono::Utc::now().to_rfc3339(),
                    "scan_cadence_secs": SIDECAR_CADENCE_SECS,
                    "audit_newest_ts": audit_newest_ts,
                });
                let path = sidecar_root.join("stats.json");
                let tmp = sidecar_root.join("stats.json.tmp");
                let write = std::fs::write(&tmp, body.to_string())
                    .and_then(|()| std::fs::rename(&tmp, &path));
                if let Err(e) = write {
                    tracing::warn!(path = %path.display(), error = %e, "stats sidecar write failed");
                }
            }
        });
    }

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
    // Seed the resource body from the persisted lens.md side-copy so a
    // serve restart restores the last-rendered lens immediately. lens.md
    // and lens_state.json are a persisted pair; reloading only the state
    // (below) while leaving the body empty desyncs them — the gate then
    // reads "nothing changed" against the restored fingerprints and never
    // re-renders. The daemon loop additionally forces one live re-render
    // on its first qualifying tick to supersede a stale seed. Absent file
    // or read error → empty body (logged), never a wedged startup.
    let initial_lens_body = match load_lens_markdown(&root) {
        Ok(Some(body)) => body,
        Ok(None) => String::new(),
        Err(err) => {
            tracing::warn!(error = %err, "lens.md load failed; serving empty body");
            String::new()
        }
    };
    let lens_resource = Arc::new(MemoryLensResource::new(initial_lens_body));
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
        // Concept-activation reader for the lens concept slot
        // (memory-activation-frame.md). ThreadsDb implements
        // ConceptActivationReader over the same chain_log + concept ledger;
        // a fresh-handle Arc clone, like chain_reader above.
        let concept_reader: Arc<dyn ostk_recall_store::ConceptActivationReader> =
            ctx.threads.clone();
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
        let lens_config = resolve_lens_config(cfg.lens.as_ref(), cfg.relational.as_ref());
        tokio::spawn(async move {
            run_lens_loop(
                attention,
                corpus,
                registry,
                resource,
                chain_sink_clone,
                chain_reader,
                concept_reader,
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
    let server = server_base
        .with_resources(lens_registry)
        // Slice 5: hand the daemon's config to the server so the
        // `memory_concept` `crystallize` action can resolve a typed node's
        // stub-file directory from its `[[sources]]` block.
        .with_config(Arc::new(cfg.clone()));
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
    // Aggregate (project-agnostic) ambient rolling vector for the scope the
    // memory-lens polls (`ambient_scope_default()`). The live observer
    // mirrors each turn into that scope but emits NO aggregate snapshot, so
    // on restart we re-derive it from the most recent ambient per-project
    // snapshot (the last one in chronological chain order).
    let mut latest_ambient: Option<Vec<f32>> = None;
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
                // Derive the aggregate ambient rolling vector from the most
                // recent ambient snapshot, so a restart restores the lens's
                // read scope immediately. The observer's ambient turns carry
                // `session_id="ambient"` / `agent="substrate"`; the last such
                // row in chronological order is the most recent — the same
                // "later row supersedes" rule `latest_rolling` relies on per
                // scope above (`iter_chain` yields seq-ASC = chronological).
                // The original wall-clock `ts` is intentionally NOT used:
                // `ChainEvent::from_row` synthesizes a decode-time `ts`, so it
                // is unavailable here and would only restate the seq order.
                if ev_scope.session_id.as_deref() == Some("ambient")
                    && ev_scope.agent.as_deref() == Some("substrate")
                {
                    latest_ambient = Some(vec.clone());
                }
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
            | ChainEvent::OperatorSelected { .. }
            // memory-activation-frame.md slice 1: concept-activation events
            // are audit-only on replay — the ConceptActivationReader reads
            // them on demand (frame + lens concept slot), never into
            // in-memory state, the same as the P7b access ledger above.
            | ChainEvent::ConceptAccessed { .. }
            | ChainEvent::ConceptFocused { .. }
            | ChainEvent::ConceptConnected { .. }
            | ChainEvent::ConceptPromoted { .. }
            | ChainEvent::ConceptNoteAdded { .. } => {}
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
    // Seed the project-agnostic aggregate scope the memory-lens reads, so
    // the lens has live ambient attention the moment serve restarts —
    // before any new turn (the lens loop's forced first tick then renders
    // it). The running observer keeps this scope fresh via its per-turn
    // mirror; here we restore it from the persisted ambient snapshots.
    let aggregate_seeded = latest_ambient.is_some();
    if let Some(vec) = latest_ambient {
        if let Err(err) = attention
            .seed_rolling_vec(&ambient_scope_default(), vec)
            .await
        {
            tracing::warn!(error = %err, "aggregate ambient rolling seed skipped");
        }
    }
    tracing::info!(
        events = n,
        focus_events = focus_n,
        rolling_snapshots = rolling_n,
        aggregate_seeded,
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
    salience: &SalienceSettings,
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

    // --- autonomous-salience precompute (THESIS axis 1) ----------------
    // Skip all the work when the master flag is off (factors stay empty ⇒ the
    // scorer is bit-identical to v1). Later axes (value, negative-transfer)
    // hang their precompute off this same single walk (design §2.2: ONE pass);
    // I1 fills only the specificity field.
    if salience.scorer_v2 {
        if let Err(err) =
            precompute_salience_factors(threads, corpus, attention, salience).await
        {
            // Never block boot on the precompute — a failure just leaves the
            // factors neutral (the scorer falls back to v1 behavior).
            tracing::warn!(error = %err, "salience precompute failed; factors left neutral");
        }
    }

    Ok(seeded)
}

/// Boot-time per-handle salience precompute (THESIS axes 1-3) — ONE pass that
/// fills all three fields of the factor map (design §2.2 "design the boot
/// wiring as ONE pass").
///
/// Reads the whole evidence graph once (`list_evidence_all`), resolves the
/// union of resonating + anchor chunk-ids to their `(project, source, source_id)`
/// metadata via the corpus, then per handle computes:
/// - **specificity** (axis 1): co-occurrence entropy `1 − H/H_max`;
/// - **value** (axis 3): the surfaced/used ledger join (`surfaced_vs_used`) +
///   curated-confidence propagation (decision/needle evidence +
///   `concept_support_by_coord`), monotone in positive evidence (v1
///   `value_neutral = 1.0` ⇒ a pass-through);
/// - **`neg_penalty`** (axis 2): centered-kNN proximity to the dormant/rejected
///   exemplar set (built store-level in passes A/B above).
///
/// Each axis is skipped entirely when its toggle is off (no DB reads for it),
/// and its field stays the neutral identity. Cost is bounded: one evidence
/// scan + one corpus fetch + one ledger scan + one concept-support read, paid
/// once at boot.
#[allow(clippy::too_many_lines)]
async fn precompute_salience_factors(
    threads: &Arc<ThreadsDb>,
    corpus: &Arc<CorpusStore>,
    attention: &InMemoryAttention,
    salience: &SalienceSettings,
) -> Result<()> {
    use std::collections::{HashMap, HashSet};

    use ostk_recall_attention::{
        SalienceFactors, SourceMeta, center, negative_penalty,
        salience::{specificity_from_project_dist, value_from},
    };
    use ostk_recall_store::{ConceptActivationReader, RelationState, default_since_now};

    // Per-handle anchors: the negative-transfer query side, plus the global-
    // mean input. `reanchor_inputs` already carries each anchored thread's
    // cached `anchor_vec` AND its `anchor_chunk_id`; read once and split into
    // the per-handle anchor-vector map (negative-transfer query side + global
    // mean) and the per-handle anchor-chunk map (value's surfaced/used join
    // unions the anchor chunk with the evidence chunks).
    let reanchor = threads
        .reanchor_inputs()
        .map_err(|e| anyhow!("reanchor_inputs for salience: {e}"))?;
    let anchors: HashMap<String, Vec<f32>> = reanchor
        .iter()
        .filter_map(|i| {
            i.anchor_vec
                .clone()
                .map(|v| (i.handle.as_str().to_string(), v))
        })
        .collect();
    let anchor_chunk: HashMap<String, String> = reanchor
        .iter()
        .filter_map(|i| {
            i.anchor_chunk_id
                .clone()
                .map(|c| (i.handle.as_str().to_string(), c))
        })
        .collect();

    // ---- negative-transfer passes (A: global mean, B: exemplar set) ----
    // R3: built ONCE here, store-level. Skipped entirely when the axis is off
    // (no DB reads, no centering); the neg factor then stays at the neutral
    // 0.0 and the scorer's `damp` term is the v1 identity.
    let mut global_mean: Vec<f32> = Vec::new();
    let mut neg_exemplars: Vec<Vec<f32>> = Vec::new();
    if salience.negative_enabled {
        // (A) global anchor mean — the anisotropy correction `center()` needs
        //     (R3 §TL;DR; design §2.2 A). Mean of every normalized thread anchor.
        global_mean = mean_normalized(anchors.values());
        // (B) exemplar set: dormant-thread anchors (STRONG, R3 §1A) + rejected-
        //     concept evidence anchors (WEAKER, R3 §1B), each centered.
        let mut neg_raw = threads
            .dormant_anchor_vecs()
            .map_err(|e| anyhow!("dormant_anchor_vecs: {e}"))?;
        neg_raw.extend(
            threads
                .rejected_concept_anchors()
                .map_err(|e| anyhow!("rejected_concept_anchors: {e}"))?,
        );
        neg_exemplars = neg_raw.iter().map(|v| center(v, &global_mean)).collect();
        attention
            .set_negative_exemplars(global_mean.clone(), neg_exemplars.clone())
            .await;
        tracing::info!(
            exemplars = neg_exemplars.len(),
            "negative-transfer exemplar set installed"
        );
    }

    let ev_by_handle = threads
        .list_evidence_all()
        .map_err(|e| anyhow!("list_evidence_all: {e}"))?;
    // The negative penalty keys on a handle's anchor, not its evidence graph,
    // and value can key on an anchor chunk's use even with no evidence links —
    // so a thread with an anchor but no evidence still earns those axes. Only
    // bail when there is nothing to compute on ANY axis.
    let have_anchored_work =
        !anchors.is_empty() && (salience.negative_enabled || salience.value_enabled);
    if ev_by_handle.is_empty() && neg_exemplars.is_empty() && !have_anchored_work {
        return Ok(());
    }

    // Union of every resonating chunk-id across all handles, resolved once.
    // Anchor chunks join too (value's surfaced/used ledger and judgment lookup
    // key on the anchor as well as the evidence links).
    let mut chunk_ids: HashSet<String> = HashSet::new();
    for links in ev_by_handle.values() {
        for link in links {
            if let Some(id) = link
                .last_resolved_chunk_id
                .clone()
                .or_else(|| link.original_path.to_str().map(str::to_string))
            {
                chunk_ids.insert(id);
            }
        }
    }
    for id in anchor_chunk.values() {
        chunk_ids.insert(id.clone());
    }
    let chunk_ids: Vec<String> = chunk_ids.into_iter().collect();
    let fetched = corpus
        .fetch_chunks_by_ids(&chunk_ids)
        .await
        .map_err(|e| anyhow!("fetch_chunks_by_ids for salience: {e}"))?;
    let chunk_meta: HashMap<String, SourceMeta> = fetched
        .into_iter()
        .map(|(id, (chunk, _emb))| {
            (
                id,
                SourceMeta {
                    project: chunk.project,
                    source: chunk.source.as_str().to_string(),
                    source_id: chunk.source_id,
                },
            )
        })
        .collect();

    // ---- value pass inputs (I3) — the SHARED surfaced/used ledger join +
    // the curated-confidence sources. Built once here, skipped entirely when
    // the value axis is off (no ledger scan, no concept-support read). ----
    let now = chrono::Utc::now();
    let since = default_since_now();
    let (use_ledger, active_coords) = if salience.value_enabled {
        // Each handle's attesting chunks: {anchor} ∪ {Active evidence chunks}.
        let mut handle_chunks: HashMap<String, Vec<String>> = HashMap::new();
        let handle_union: HashSet<&String> = ev_by_handle.keys().chain(anchors.keys()).collect();
        for handle in handle_union {
            let mut chunks: Vec<String> = Vec::new();
            if let Some(cid) = anchor_chunk.get(handle) {
                chunks.push(cid.clone());
            }
            if let Some(links) = ev_by_handle.get(handle) {
                for link in links {
                    if link.relation_state != RelationState::Active {
                        continue;
                    }
                    if let Some(id) = link
                        .last_resolved_chunk_id
                        .clone()
                        .or_else(|| link.original_path.to_str().map(str::to_string))
                    {
                        chunks.push(id);
                    }
                }
            }
            handle_chunks.insert(handle.clone(), chunks);
        }
        let ledger = threads
            .surfaced_vs_used(&handle_chunks, since)
            .map_err(|e| anyhow!("surfaced_vs_used for value: {e}"))?;
        // Active-concept support per coordinate — judgment propagation source
        // (active concepts carry confidence 1.0). One ledger-backed read.
        let coords = threads
            .concept_support_by_coord(since)
            .map(|m| {
                m.into_iter()
                    .map(|(coord, sup)| (coord, sup.activation))
                    .collect::<HashMap<(String, String), f32>>()
            })
            .unwrap_or_default();
        (ledger, coords)
    } else {
        (HashMap::new(), HashMap::new())
    };

    // ---- specificity pass: lexical project co-occurrence (recal approach 1) ----
    // Specificity = how CONCENTRATED a handle's whole-phrase mentions are across
    // the corpus's PROJECTS, normalized by the GLOBAL project count. A coined
    // concept concentrates in few projects (SPECIFIC); generic plumbing spreads
    // across many (GENERIC). Document-frequency inverts here (mono-thematic
    // corpus — the core concepts have the most DOCS), but project-span
    // separates them (diagnostic: concepts ≤6 projects, plumbing 8–14, N=23).
    // One projected (project, text) scan, run only when the axis is on; a scan
    // error degrades every handle to neutral specificity.
    let (n_projects_global, project_dist): (usize, HashMap<String, Vec<f32>>) =
        if salience.specificity_enabled {
            // Specificity is a property of the handle STRING (its text
            // co-occurrence across projects) — independent of whether the
            // thread has a cached anchor_vec. Source from EVERY thread handle
            // (`reanchor`), not just the anchored subset, so anchor-poor but
            // text-rich plumbing (in-memory, hand-off) is measured + demoted
            // rather than escaping to neutral.
            let handle_strings: Vec<String> = reanchor
                .iter()
                .map(|i| i.handle.as_str().to_string())
                .collect::<HashSet<String>>()
                .into_iter()
                .collect();
            match corpus.project_phrase_cooccurrence(&handle_strings).await {
                Ok((n, d)) => (n, d),
                Err(e) => {
                    tracing::warn!(error = %e, "salience: project_phrase_cooccurrence failed; specificity stays neutral");
                    (0, HashMap::new())
                }
            }
        } else {
            (0, HashMap::new())
        };

    // ---- pass C: per-handle factors (one map, all three axes) ----
    // Specificity = 1 − H(project_dist)/ln(N_projects_global) (concentrated in
    // few projects = high); negative-transfer keys on the anchor; value keys on
    // the surfaced/used ledger + judgment evidence. Iterate the UNION so a
    // handle qualifying for any one axis gets it; the others stay their neutral
    // identity (`specificity = 1.0`, `value = 1.0`, `neg_penalty = 0.0`).
    let mut factors: HashMap<String, SalienceFactors> = HashMap::new();
    // Union includes project_dist.keys() so an anchor-poor but text-measured
    // handle (in-memory, hand-off) still gets a specificity factor written —
    // otherwise it falls through to the scorer's neutral 1.0 and escapes
    // demotion despite having a computed low specificity.
    let handle_union: HashSet<&String> = ev_by_handle
        .keys()
        .chain(anchors.keys())
        .chain(project_dist.keys())
        .collect();
    let empty_ev: Vec<ostk_recall_store::EvidenceLink> = Vec::new();
    for handle in handle_union {
        let evidence = ev_by_handle.get(handle).unwrap_or(&empty_ev);
        // Project-distribution specificity, GLOBAL-normalized (recal approach 1).
        // A handle with no project matches (coined-unwritten / axis off) ⇒
        // neutral 1.0 (rests on the other axes).
        let specificity = match project_dist.get(handle) {
            Some(counts) if !counts.is_empty() => {
                specificity_from_project_dist(counts, n_projects_global)
            }
            _ => 1.0,
        };
        let neg_penalty = if salience.negative_enabled {
            anchors.get(handle).map_or(0.0, |anchor| {
                negative_penalty(anchor, &global_mean, &neg_exemplars, salience)
            })
        } else {
            0.0
        };
        // value = value_neutral + (1 − value_neutral)·positive (monotone in
        // positive evidence; v1 value_neutral = 1.0 ⇒ a constant pass-through).
        let value = if salience.value_enabled {
            let used = use_ledger
                .get(handle)
                .map(|(_, accesses)| accesses.as_slice())
                .unwrap_or(&[]);
            value_from(used, evidence, &chunk_meta, &active_coords, now, salience)
        } else {
            1.0
        };
        factors.insert(
            handle.clone(),
            SalienceFactors {
                specificity,
                value,
                neg_penalty,
            },
        );
    }
    let n = factors.len();
    attention.set_salience_factors(factors).await;
    tracing::info!(
        handles = n,
        "salience factors computed (specificity + value + negative-transfer)"
    );
    Ok(())
}

// --- salience A/B diagnostic (read-only) ------------------------------------

/// Confirmed coherent-NOISE handles (THESIS / review). Tagged NOISE in the A/B
/// table; the real-data P1 verdict checks these rank below every CONCEPT.
const AB_NOISE: &[&str] = &[
    "turn-digest",
    "squad-lead",
    "re-run",
    "non-blocking",
    "re-read",
    "pre-existing",
    "follow-up",
    "top-level",
    "system-reminder",
    "exec-replace",
    "watch-notify",
    "post-restart",
    "no-op",
    "read-only",
    "team-lead",
    "teammate-message",
];

/// Confirmed REAL concepts — coined IDEAS, not repo/tool/project NAMES. The
/// P1 verdict checks every NOISE handle ranks below the worst of these.
/// `ostk-recall` was removed: it's the REPO NAME (present in ~11 projects,
/// legitimately GENERIC under project-span), so tagging it CONCEPT made P1 fail
/// on a mislabel, not a signal error. Repo/tool names sit with the plumbing.
const AB_CONCEPT: &[&str] = &[
    "cognitive-memory",
    "ostk-cache",
    "dereference-or-void",
    "relational-substrate-docgraph",
    "slipstream",
    "mish",
    "attention-is-sovereign",
    "abi-as-sovereign-boundary",
];

/// One row of the merged off-vs-on ranking.
#[derive(Debug, serde::Serialize)]
struct AbRow {
    handle: String,
    tag: &'static str, // NOISE | CONCEPT | (other)
    /// The specificity FACTOR applied on the v2 surface (`why.specificity`) —
    /// the recalibration's key proof: < 1.0 (and low) for diffuse plumbing,
    /// ~1.0 for concentrated concepts. `None` if the handle isn't on the v2
    /// surface.
    on_specificity: Option<f32>,
    off_score: Option<f32>,
    off_rank: Option<usize>,
    on_score: Option<f32>,
    on_rank: Option<usize>,
    /// `off_rank - on_rank` (positive = moved UP under the new scorer). `None`
    /// if the handle is missing from either surface.
    rank_delta: Option<i64>,
}

/// READ-ONLY salience A/B: run the real boot-pass scoring over the live ledger
/// with `scorer_v2` off and on, and report the ranking delta + the THESIS P1
/// ordering verdict + the salience-health scoreboard for each.
///
/// Opens the corpus + threads **read-only** (`ThreadsDb::open_read_only`,
/// `build_query_engine_read_only`) and never scans/ingests; the in-memory
/// scoring (replay + a write-free re-anchor + the factor precompute) is exactly
/// what `serve` runs at boot, so the surface this prints is the IDLE,
/// floor-driven surface where coherent-noise dominates today. No mutation.
pub async fn salience_ab(
    config_path: &Path,
    embedder: Arc<dyn ChunkEmbedder>,
    top: usize,
    json: bool,
) -> Result<()> {
    use ostk_recall_core::SalienceHealth;

    let cfg = Config::load(config_path)
        .with_context(|| format!("loading config {}", config_path.display()))?;
    // Read-only engine (no scan/ingest), shared embedder for cosine-comparable
    // vectors — same prerequisite serve asserts.
    let engine = build_query_engine_read_only(config_path, Arc::clone(&embedder)).await?;
    let root = engine.store().root().to_path_buf();

    if engine.store().dim() != embedder.dim() {
        return Err(anyhow!(
            "embedder/corpus dimension mismatch ({} vs {}) — point the original \
             embedder model at this corpus",
            embedder.dim(),
            engine.store().dim()
        ));
    }
    let weaver_stop = WeaverSettings::resolve(cfg.weaver.as_ref()).stop_handles;

    // Run the boot-pass scoring for one value of `scorer_v2`, returning the
    // ranked surface (handle → score, descending) of the substrate scope plus
    // the health block. READ-ONLY: opens its own read-only ThreadsDb handle.
    async fn run_one(
        scorer_v2: bool,
        cfg: &Config,
        root: &Path,
        engine: &QueryEngine,
        embedder: &Arc<dyn ChunkEmbedder>,
        weaver_stop: &[String],
    ) -> Result<(Vec<ostk_recall_core::AttentionPage>, ostk_recall_core::SalienceHealth)> {
        use ostk_recall_attention::salience_health;
        // Read-only threads handle: SQLITE_OPEN_READ_ONLY + NoopChainSink, so
        // even a stray chain-emitting/anchor-backfill call cannot write.
        let threads = Arc::new(
            ThreadsDb::open_read_only(root)
                .map_err(|e| anyhow!("open threads db (read-only): {e}"))?,
        );
        let mut salience = SalienceSettings::resolve(cfg.salience.as_ref());
        salience.scorer_v2 = scorer_v2; // the A/B knob

        let attention = InMemoryAttention::with_embedder(Arc::clone(embedder))
            .with_stop_handles(weaver_stop.to_vec())
            .with_salience_settings(&salience);

        // Replay the chain into attention (restores counters/folds/anchors),
        // exactly as serve does — read-only (replay emits no new chain rows).
        if let Err(e) = replay_chain_into_attention(&threads, &attention).await {
            tracing::warn!(error = %e, "salience-ab: chain replay failed; attention starts empty");
        }
        // Write-free re-anchor: seed every thread's anchor into attention from
        // its cached anchor_vec (the common path) or the corpus chunk — WITHOUT
        // the `set_anchor_vec` persistence serve does (that is the only DB write
        // in the boot re-anchor; skipping it keeps this read-only while the
        // in-memory anchors — hence the scoring — stay identical to serve).
        reanchor_into_attention_read_only(&threads, engine.store(), &attention).await?;
        // The factor precompute (specificity/value/negative) — runs only when
        // scorer_v2 is on; reads threads/corpus, writes only in-memory attention.
        if salience.scorer_v2 {
            if let Err(e) =
                precompute_salience_factors(&threads, engine.store(), &attention, &salience).await
            {
                tracing::warn!(error = %e, "salience-ab: precompute failed; factors neutral");
            }
        }

        // Surface the substrate scope (project=None) — where the boot pass lands
        // every durable thread; a wide limit so the whole active set is ranked.
        let scope = AttentionScope {
            project: None,
            session_id: Some("replay".into()),
            agent: Some("substrate".into()),
            privacy_tier: PrivacyTier::T1Project,
        };
        let surface = attention.surface(&scope, 1000).await?;

        // Health scoreboard over this surface (no ledger join needed for the
        // entropy/ratio/spread the A/B reads; never-used/drift left empty).
        let curated = attention.curated_handles();
        let health = salience_health(
            &surface,
            &curated,
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
            &salience.health,
        );
        Ok((surface, health))
    }

    let (off_surface, off_health) =
        run_one(false, &cfg, &root, &engine, &embedder, &weaver_stop).await?;
    let (on_surface, on_health) =
        run_one(true, &cfg, &root, &engine, &embedder, &weaver_stop).await?;

    // Merge into per-handle rows.
    let rank_map = |surface: &[ostk_recall_core::AttentionPage]| {
        surface
            .iter()
            .enumerate()
            .map(|(i, p)| (p.handle.clone(), (p.score, i + 1)))
            .collect::<std::collections::HashMap<String, (f32, usize)>>()
    };
    let off = rank_map(&off_surface);
    let on = rank_map(&on_surface);
    // The specificity factor applied on the v2 surface, per handle — the proof
    // the recalibration actually fires (was uniformly 1.0 / inert before).
    let on_spec: std::collections::HashMap<String, f32> = on_surface
        .iter()
        .map(|p| (p.handle.clone(), p.why.specificity))
        .collect();
    let tag_of = |h: &str| -> &'static str {
        if AB_NOISE.contains(&h) {
            "NOISE"
        } else if AB_CONCEPT.contains(&h) {
            "CONCEPT"
        } else {
            "-"
        }
    };
    let mut handles: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    handles.extend(off.keys().cloned());
    handles.extend(on.keys().cloned());
    let mut rows: Vec<AbRow> = handles
        .into_iter()
        .map(|h| {
            let o = off.get(&h).copied();
            let n = on.get(&h).copied();
            let rank_delta = match (o, n) {
                (Some((_, orank)), Some((_, nrank))) => {
                    Some(orank as i64 - nrank as i64) // + = moved up under v2
                }
                _ => None,
            };
            AbRow {
                tag: tag_of(&h),
                on_specificity: on_spec.get(&h).copied(),
                handle: h,
                off_score: o.map(|x| x.0),
                off_rank: o.map(|x| x.1),
                on_score: n.map(|x| x.0),
                on_rank: n.map(|x| x.1),
                rank_delta,
            }
        })
        .collect();
    // Sort by the new scorer's rank (handles absent from the on-surface last).
    rows.sort_by(|a, b| match (a.on_rank, b.on_rank) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.handle.cmp(&b.handle),
    });

    // Real-data P1 verdict: every NOISE handle on the on-surface ranks below
    // every CONCEPT handle on the on-surface.
    let worst_concept_on_rank = rows
        .iter()
        .filter(|r| r.tag == "CONCEPT")
        .filter_map(|r| r.on_rank)
        .max();
    let best_noise_on_rank = rows
        .iter()
        .filter(|r| r.tag == "NOISE")
        .filter_map(|r| r.on_rank)
        .min();
    let p1_holds = match (worst_concept_on_rank, best_noise_on_rank) {
        (Some(worst_concept), Some(best_noise)) => best_noise > worst_concept,
        // No noise (or no concept) present on the surface ⇒ vacuously holds.
        _ => true,
    };

    // Biggest rank drops (noise pushed down = positive-then-negative delta) and
    // rises, for the summary.
    let mut by_delta: Vec<&AbRow> = rows.iter().filter(|r| r.rank_delta.is_some()).collect();
    by_delta.sort_by_key(|r| r.rank_delta.unwrap());
    let drops: Vec<&AbRow> = by_delta.iter().take(5).copied().collect(); // most negative = dropped
    let rises: Vec<&AbRow> = by_delta.iter().rev().take(5).copied().collect();

    if json {
        let out = serde_json::json!({
            "rows": rows.iter().take(top).collect::<Vec<_>>(),
            "p1_noise_below_concept": p1_holds,
            "health_off": off_health,
            "health_on": on_health,
            "top_drops": drops.iter().map(|r| (&r.handle, r.tag, r.rank_delta)).collect::<Vec<_>>(),
            "top_rises": rises.iter().map(|r| (&r.handle, r.tag, r.rank_delta)).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    println!("salience A/B — scorer_v2 OFF vs ON (read-only, idle surface)\n");
    println!(
        "{:<34} {:>5} {:>7} {:>5} {:>7} {:>5} {:>6}  {}",
        "handle", "spec", "off", "o#", "on", "n#", "Δrank", "tag"
    );
    for r in rows.iter().take(top) {
        let fmt_s = |s: Option<f32>| s.map_or("-".to_string(), |v| format!("{v:.3}"));
        let fmt_r = |r: Option<usize>| r.map_or("-".to_string(), |v| v.to_string());
        let fmt_d = |d: Option<i64>| d.map_or("-".to_string(), |v| format!("{v:+}"));
        println!(
            "{:<34} {:>5} {:>7} {:>5} {:>7} {:>5} {:>6}  {}",
            r.handle,
            fmt_s(r.on_specificity),
            fmt_s(r.off_score),
            fmt_r(r.off_rank),
            fmt_s(r.on_score),
            fmt_r(r.on_rank),
            fmt_d(r.rank_delta),
            r.tag,
        );
    }
    println!("\n── summary ──");
    println!(
        "P1 (every NOISE ranks below every CONCEPT under v2): {}",
        if p1_holds { "PASS" } else { "FAIL" }
    );
    let board = |label: &str, h: &SalienceHealth| {
        println!(
            "  {label:<4} entropy={:?} spread={:.3} curated_ratio={:.3}",
            h.surface_entropy, h.surface_score_spread, h.curated_ratio
        );
    };
    println!("salience-health scoreboard:");
    board("off", &off_health);
    board("on", &on_health);
    println!("top rank drops (NOISE should dominate):");
    for r in &drops {
        println!("  {:<34} Δ{:+} {}", r.handle, r.rank_delta.unwrap_or(0), r.tag);
    }
    println!("top rank rises (CONCEPT should dominate):");
    for r in &rises {
        println!("  {:<34} Δ{:+} {}", r.handle, r.rank_delta.unwrap_or(0), r.tag);
    }
    Ok(())
}

/// Write-free variant of the boot re-anchor: seeds each thread's anchor into
/// `attention` from its cached `anchor_vec`, or the corpus chunk embedding,
/// but does NOT persist the `set_anchor_vec` backfill (the only DB write in
/// `re_anchor_threads_from_corpus`). Keeps the salience A/B diagnostic
/// read-only while the in-memory anchors — and therefore the scoring — stay
/// identical to what `serve` produces at boot.
async fn reanchor_into_attention_read_only(
    threads: &Arc<ThreadsDb>,
    corpus: &Arc<CorpusStore>,
    attention: &InMemoryAttention,
) -> Result<()> {
    let inputs = threads
        .reanchor_inputs()
        .map_err(|e| anyhow!("reanchor_inputs: {e}"))?;
    if inputs.is_empty() {
        return Ok(());
    }
    let need_fetch: Vec<String> = inputs
        .iter()
        .filter(|i| i.anchor_vec.is_none())
        .filter_map(|i| i.anchor_chunk_id.clone())
        .collect();
    let fetched = corpus
        .fetch_embeddings(&need_fetch)
        .await
        .map_err(|e| anyhow!("fetch_embeddings for re-anchor: {e}"))?;
    for input in inputs {
        let scope = AttentionScope {
            project: None,
            session_id: Some("replay".into()),
            agent: Some("substrate".into()),
            privacy_tier: input.privacy_tier,
        };
        let vec = input
            .anchor_vec
            .or_else(|| input.anchor_chunk_id.as_ref().and_then(|id| fetched.get(id).cloned()));
        if let Some(vec) = vec {
            attention
                .seed_anchor(&scope, input.handle, vec)
                .await
                .map_err(|e| anyhow!("seed_anchor: {e}"))?;
        }
    }
    Ok(())
}

/// Element-wise mean of L2-normalized vectors — the global anchor mean the
/// negative-transfer `center()` subtracts (R3 anisotropy correction). Empty /
/// dimension-mismatched input yields an empty mean (a no-op for `center`).
fn mean_normalized<'a>(vecs: impl Iterator<Item = &'a Vec<f32>>) -> Vec<f32> {
    let mut acc: Vec<f32> = Vec::new();
    let mut n = 0usize;
    for v in vecs {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= 0.0 {
            continue;
        }
        if acc.is_empty() {
            acc = vec![0.0; v.len()];
        } else if acc.len() != v.len() {
            continue; // dimension drift — skip, never corrupt the mean
        }
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += x / norm;
        }
        n += 1;
    }
    if n > 0 {
        #[allow(clippy::cast_precision_loss)] // n is a thread count, well within range
        let nf = n as f32;
        for a in &mut acc {
            *a /= nf;
        }
    }
    acc
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
                        if let Err(e) = server
                            .serve_with_client(ClientId::network(id), rd, wr)
                            .await
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
                        if let Err(e) = server
                            .serve_with_client(ClientId::network(id), rd, wr)
                            .await
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
        connect_with_heal(&endpoint).await
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

/// →1957 legs 1+2: line-oriented bridge with daemon heal (port of
/// haystack's →1935.1 `reconnect_with_backoff` + →1943 receipt). The
/// old byte-splice died with the daemon — the operator's restart of
/// serve killed every connected client's toolset mid-request. This
/// loop caches the client's `initialize`/`notifications/initialized`
/// handshake, reconnects with backoff when the daemon goes away, and
/// REPLAYS the handshake against the new daemon so the client never
/// has to re-`initialize`. Requests arriving while the daemon is down
/// queue in the OS pipe buffer and deliver on heal.
#[cfg(unix)]
struct DaemonConn {
    reader: tokio::io::BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
}

#[cfg(unix)]
async fn connect_daemon(endpoint: &Path) -> std::io::Result<DaemonConn> {
    let stream = tokio::net::UnixStream::connect(endpoint).await?;
    let (r, w) = stream.into_split();
    Ok(DaemonConn {
        reader: tokio::io::BufReader::new(r),
        writer: w,
    })
}

/// Cache the client's handshake lines so they can be replayed against
/// any daemon we reconnect to. Notifications and non-handshake methods
/// pass through untouched.
#[cfg(unix)]
fn cache_handshake(
    line: &str,
    cached_initialize: &mut Option<String>,
    cached_initialized: &mut Option<String>,
) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return;
    };
    match v.get("method").and_then(serde_json::Value::as_str) {
        Some("initialize") => *cached_initialize = Some(line.to_string()),
        Some("initialized" | "notifications/initialized") => {
            *cached_initialized = Some(line.to_string());
        }
        _ => {}
    }
}

/// Reconnect budget. Default 300s rides out a `make install` + daemon
/// restart; override via `OSTK_RECALL_RECONNECT_BUDGET_S`.
#[cfg(unix)]
fn reconnect_budget() -> Duration {
    const FALLBACK_SECS: u64 = 300;
    let secs = std::env::var("OSTK_RECALL_RECONNECT_BUDGET_S")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(FALLBACK_SECS);
    Duration::from_secs(secs)
}

/// Reconnect with exponential backoff (250ms → 5s cap) until the
/// budget is exhausted, replaying the cached handshake on the fresh
/// socket. On success, sends a `notifications/bridge_reconnected`
/// receipt THROUGH the healed daemon (→1943 sibling): the receipt is
/// self-attesting — it can only land if the new socket works — and
/// turns "silently healed" into an attested event. Best-effort: the
/// heal never fails because the receipt couldn't be written.
#[cfg(unix)]
async fn reconnect_with_backoff(
    endpoint: &Path,
    cached_initialize: Option<&str>,
    cached_initialized: Option<&str>,
) -> Result<DaemonConn> {
    use tokio::io::AsyncWriteExt;

    let budget = reconnect_budget();
    let deadline = tokio::time::Instant::now() + budget;
    let started = std::time::Instant::now();
    let mut delay = Duration::from_millis(250);
    let mut attempts: u32 = 0;
    let mut conn = loop {
        attempts += 1;
        match connect_and_replay(endpoint, cached_initialize, cached_initialized).await {
            Ok(c) => {
                if attempts > 1 {
                    eprintln!("[recall-bridge] daemon reconnect recovered after {attempts} attempts");
                }
                break c;
            }
            Err(e) => {
                if attempts == 1 || attempts % 8 == 0 {
                    eprintln!("[recall-bridge] daemon reconnect attempt {attempts} failed: {e}");
                }
                if tokio::time::Instant::now() + delay > deadline {
                    bail!(
                        "daemon reconnect budget ({budget:?}) exhausted after {attempts} attempts: {e}"
                    );
                }
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_secs(5));
            }
        }
    };

    let frame = format!(
        "{}\n",
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/bridge_reconnected",
            "params": {
                "attempts": attempts,
                "elapsed_ms": u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                "bridge_pid": std::process::id(),
            }
        })
    );
    if let Err(e) = async {
        conn.writer.write_all(frame.as_bytes()).await?;
        conn.writer.flush().await
    }
    .await
    {
        eprintln!("[recall-bridge] bridge_reconnected receipt failed (heal itself OK): {e}");
    }
    Ok(conn)
}

/// Connect and replay the cached handshake. The replayed `initialize`
/// RESPONSE is consumed here so it doesn't reach stdout as a duplicate
/// answer to the client's original request; the `initialized`
/// notification gets no response.
#[cfg(unix)]
async fn connect_and_replay(
    endpoint: &Path,
    cached_initialize: Option<&str>,
    cached_initialized: Option<&str>,
) -> Result<DaemonConn> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let mut conn = connect_daemon(endpoint)
        .await
        .with_context(|| format!("connecting {}", endpoint.display()))?;
    if let Some(init) = cached_initialize {
        conn.writer
            .write_all(init.as_bytes())
            .await
            .context("replay initialize")?;
        conn.writer.flush().await.context("flush initialize")?;
        let mut discard = String::new();
        let n = conn
            .reader
            .read_line(&mut discard)
            .await
            .context("read replayed initialize response")?;
        if n == 0 {
            bail!("daemon EOF during initialize replay");
        }
    }
    if let Some(initialized) = cached_initialized {
        conn.writer
            .write_all(initialized.as_bytes())
            .await
            .context("replay initialized")?;
        conn.writer.flush().await.context("flush initialized")?;
    }
    Ok(conn)
}

/// The healing bridge loop. First connect fails fast with the
/// daemon-hint message (a missing daemon at startup is operator error,
/// not an outage to ride out); after that, every daemon-side failure
/// heals via [`reconnect_with_backoff`]. stdin EOF / stdout close are
/// the client leaving — clean exit, never healed.
#[cfg(unix)]
async fn connect_with_heal(endpoint: &Path) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let mut conn = connect_daemon(endpoint).await.map_err(|e| {
        anyhow!(
            "connecting to ostk-recall MCP daemon at {}: {e}\n\
             is a daemon running? start one with `ostk-recall serve --watch`",
            endpoint.display()
        )
    })?;

    let mut cached_initialize: Option<String> = None;
    let mut cached_initialized: Option<String> = None;
    let mut stdin_reader = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();
    let mut stdin_line = String::new();
    let mut daemon_line = String::new();

    loop {
        tokio::select! {
            r = stdin_reader.read_line(&mut stdin_line) => {
                match r {
                    Ok(0) => {
                        // Client closed stdin — half-close so the daemon
                        // sees the disconnect, then exit clean.
                        let _ = conn.writer.shutdown().await;
                        return Ok(());
                    }
                    Ok(_) => {
                        cache_handshake(&stdin_line, &mut cached_initialize, &mut cached_initialized);
                        if conn.writer.write_all(stdin_line.as_bytes()).await.is_err()
                            || conn.writer.flush().await.is_err()
                        {
                            eprintln!("[recall-bridge] daemon write failed — reconnecting");
                            conn = reconnect_with_backoff(
                                endpoint,
                                cached_initialize.as_deref(),
                                cached_initialized.as_deref(),
                            ).await?;
                            // Retry the original write once on the fresh
                            // connection; a second failure drops the
                            // message (the client's own timeout covers it).
                            if conn.writer.write_all(stdin_line.as_bytes()).await.is_err()
                                || conn.writer.flush().await.is_err()
                            {
                                eprintln!("[recall-bridge] write still failing after reconnect — dropping message");
                            }
                        }
                        stdin_line.clear();
                    }
                    Err(_) => return Ok(()),
                }
            }
            r = conn.reader.read_line(&mut daemon_line) => {
                match r {
                    Ok(0) => {
                        // Daemon EOF. In-flight responses are lost;
                        // reconnect + replay so the next request still
                        // passes the daemon's initialized gate.
                        eprintln!("[recall-bridge] daemon disconnected — reconnecting");
                        conn = reconnect_with_backoff(
                            endpoint,
                            cached_initialize.as_deref(),
                            cached_initialized.as_deref(),
                        ).await?;
                        daemon_line.clear();
                    }
                    Ok(_) => {
                        if stdout.write_all(daemon_line.as_bytes()).await.is_err()
                            || stdout.flush().await.is_err()
                        {
                            return Ok(()); // stdout closed — client gone
                        }
                        daemon_line.clear();
                    }
                    Err(_) => {
                        eprintln!("[recall-bridge] daemon read error — reconnecting");
                        conn = reconnect_with_backoff(
                            endpoint,
                            cached_initialize.as_deref(),
                            cached_initialized.as_deref(),
                        ).await?;
                        daemon_line.clear();
                    }
                }
            }
        }
    }
}

/// Bidirectional splice for the **Windows** [`connect`] path (the unix
/// path heals via [`connect_with_heal`]): copy stdin → socket and
/// socket → stdout concurrently, each running to completion. When
/// stdin hits EOF the socket write half is shut down so the daemon
/// sees the client disconnect (and closes its side, ending the
/// socket→stdout copy); when the daemon closes first, the stdout copy
/// ends and the stdin copy unblocks once the client closes stdin.
/// Neither direction is cut off early, so trailing JSON-RPC frames are
/// not dropped.
#[cfg(windows)]
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
#[cfg(any(windows, test))]
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

/// Minimum gap between →1966 rescan recovery kicks. Recovery is a full
/// scan (heavy), and rescan hints recur for as long as the storm that
/// caused them lasts, so the gate bounds recovery to one full scan per
/// window instead of one per hint. Worst-case staleness during a storm
/// is this window; before →1966 it was unbounded (freeze until bounce).
const RESCAN_KICK_RATE_LIMIT: Duration = Duration::from_secs(300);

/// →1966 rescan recovery gate. A `Flag::Rescan` event is the watch
/// backend saying "I dropped events" (FSEvents kernel/user queue
/// overflow under e.g. worktree-build `target/` storms). The dropped
/// window is unrecoverable from the event stream — the only correct
/// response is a full-scan kick. `pending` survives rate-limit
/// suppression, so the *last* hint of a storm always produces a final
/// recovery kick.
#[derive(Default)]
struct RescanGate {
    last_kick: Option<Instant>,
    pending: bool,
}

impl RescanGate {
    /// Record a rescan hint; the next allowed [`Self::try_emit`] fires.
    fn note(&mut self) {
        self.pending = true;
    }

    /// True exactly when a recovery kick should fire now. Clears
    /// `pending` and stamps the rate-limit window; callers re-[`Self::note`]
    /// on dispatch failure so the kick is retried.
    fn try_emit(&mut self, now: Instant) -> bool {
        if !self.pending {
            return false;
        }
        let allowed = self
            .last_kick
            .is_none_or(|t| now.duration_since(t) >= RESCAN_KICK_RATE_LIMIT);
        if !allowed {
            return false;
        }
        self.pending = false;
        self.last_kick = Some(now);
        true
    }
}

/// →1957 watcher observability: every drop in the event loop was
/// previously an unobservable non-event (→1947's 4-day freeze was three
/// stacked silent drops). The watcher maintains these counters and
/// persists them to `<corpus_root>/watch_status.json` so `recall_stats`
/// can surface them out-of-process.
#[derive(Default, serde::Serialize)]
struct WatchStatus {
    /// Events dropped by the `NOISE_PATH_SEGMENTS` filter.
    noise_filtered: u64,
    /// Events that passed the noise filter but matched no watched
    /// root + extension pair — the →1947 starvation class.
    unmatched: u64,
    /// Rescan hints from the watch backend — each one is a window of
    /// dropped events (→1966: FSEvents queue overflow under churn).
    rescans: u64,
    /// Watch backend errors (previously warn-logged only — invisible
    /// to `recall_stats`).
    backend_errors: u64,
    /// RFC 3339 stamp of the last successful full-scan recovery kick
    /// (→1966). Absent until the first rescan hint is recovered.
    #[serde(skip_serializing_if = "Option::is_none")]
    last_rescan_kick: Option<String>,
    /// Per-source-label timestamp (RFC 3339) of the last successful
    /// scan-trigger kick.
    kicks: std::collections::BTreeMap<String, String>,
    /// When this snapshot was written (RFC 3339).
    updated: String,
}

impl WatchStatus {
    /// Atomic snapshot write (tmp + rename). Failures are warned, not
    /// fatal — observability must never take the watcher down.
    fn persist(&mut self, path: &Path) {
        self.updated = chrono::Utc::now().to_rfc3339();
        let Ok(json) = serde_json::to_vec_pretty(self) else {
            return;
        };
        let tmp = path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp, &json).and_then(|()| std::fs::rename(&tmp, path)) {
            tracing::warn!(path = %path.display(), error = %e, "watch_status persist failed");
        }
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
            spawn_scan_trigger(
                scan_lock,
                config_path,
                embedder,
                frame.to_vec(),
                ctx.clone(),
            );
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
    // →1957 (from →1947 forensics): an `ostk_project` source that is
    // declared but excluded by the [watch].projects allowlist is almost
    // certainly a mistake — its `.ostk` journal will never live-ingest
    // and the staleness is silent (haystack froze 4 days this way).
    for source in &cfg.sources {
        if matches!(source.kind, SourceKind::OstkProject)
            && !watch_cfg.watches_project(source.project.as_deref())
        {
            tracing::warn!(
                project = source.project.as_deref().unwrap_or("<unnamed>"),
                "ostk_project source declared but absent from [watch].projects — \
                 its audit/decisions/needles will NOT live-ingest (→1947)"
            );
        }
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
    // →1966: pending-rescan state — set by Rescan-flagged events,
    // drained by rate-limited full-scan recovery kicks.
    let mut rescan_gate = RescanGate::default();
    // →1957: observable drop counters + last-kick stamps, persisted to
    // the corpus root for the stats surface.
    let status_path = corpus_root.join("watch_status.json");
    let mut status = WatchStatus::default();
    let mut status_dirty = false;
    let mut status_last_write = Instant::now();
    const STATUS_FLUSH_INTERVAL: Duration = Duration::from_secs(60);
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
                            // →1966: a Rescan-flagged event is the backend
                            // saying "I dropped events" (FSEvents kernel/
                            // user queue overflow under e.g. worktree-build
                            // target/ storms). Its path is the affected
                            // *directory*, so the filters below ate the
                            // hint (target/ → noise_filtered; bare dir →
                            // unmatched) and the dropped window was never
                            // recovered — haystack kicks froze 04:12Z while
                            // other roots kept flowing. Route it to the
                            // rescan gate: a rate-limited full-scan kick is
                            // the only correct recovery.
                            if ev.event.need_rescan() {
                                status.rescans += 1;
                                status_dirty = true;
                                rescan_gate.note();
                                tracing::warn!(
                                    paths = ?ev.event.paths,
                                    info = ev.event.info().unwrap_or("none"),
                                    "watch backend dropped events; full-scan recovery kick scheduled"
                                );
                                continue;
                            }
                            for p in &ev.event.paths {
                                if path_has_noise_segment(p) {
                                    status.noise_filtered += 1;
                                    status_dirty = true;
                                    continue;
                                }
                                let mut matched = false;
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
                                        matched = true;
                                        break;
                                    }
                                }
                                if !matched {
                                    // →1947 starvation class: the event
                                    // arrived but no root+ext pair claimed
                                    // it — count it so the silence shows.
                                    status.unmatched += 1;
                                    status_dirty = true;
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
                                    status
                                        .kicks
                                        .insert(label.clone(), chrono::Utc::now().to_rfc3339());
                                    status.persist(&status_path);
                                    status_dirty = false;
                                    status_last_write = Instant::now();
                                }
                            }
                        }

                        // →1966 recovery: emit the pending full-scan kick
                        // once the rescan gate allows. Empty frame = full
                        // scan in both watch modes.
                        if rescan_gate.try_emit(now) {
                            if let Err(e) =
                                dispatch_scan_trigger(&sink, &socket_path, config_path, &[]).await
                            {
                                tracing::warn!(
                                    error = %e,
                                    "rescan recovery kick failed; will retry"
                                );
                                rescan_gate.note();
                            } else {
                                tracing::info!("rescan recovery: full-scan kick dispatched");
                                status.last_rescan_kick =
                                    Some(chrono::Utc::now().to_rfc3339());
                                status.persist(&status_path);
                                status_dirty = false;
                                status_last_write = Instant::now();
                            }
                        }
                    }
                    Err(errors) => {
                        for e in errors {
                            status.backend_errors += 1;
                            status_dirty = true;
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
                // →1966: a rescan hint suppressed by the rate limit must
                // still recover after the storm goes quiet — the tick is
                // what guarantees the final full-scan kick.
                if rescan_gate.try_emit(now) {
                    if let Err(e) =
                        dispatch_scan_trigger(&sink, &socket_path, config_path, &[]).await
                    {
                        tracing::warn!(
                            error = %e,
                            "rescan recovery kick failed; will retry"
                        );
                        rescan_gate.note();
                    } else {
                        tracing::info!("rescan recovery: full-scan kick dispatched (flush)");
                        status.last_rescan_kick = Some(chrono::Utc::now().to_rfc3339());
                        status.persist(&status_path);
                        status_dirty = false;
                        status_last_write = Instant::now();
                    }
                }
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
                        status
                            .kicks
                            .insert(label.clone(), chrono::Utc::now().to_rfc3339());
                        status_dirty = true;
                    }
                }
                // Periodic flush so drop counters surface even when no
                // kick fires (a starved watcher is exactly the case the
                // counters exist for).
                if status_dirty && status_last_write.elapsed() >= STATUS_FLUSH_INTERVAL {
                    status.persist(&status_path);
                    status_dirty = false;
                    status_last_write = Instant::now();
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
    ".pnpm",
    ".ostk",
    ".worktrees",
    ".next",
    ".turbo",
    "dist",
    "build",
];

/// Basenames under `.ostk/` that carry sub-area SIGNAL and must pass the
/// noise filter despite the `.ostk` segment. These are exactly the files
/// `path_sub_area` routes (journal/audit → Audit, decisions → Decisions,
/// issues → Needles): →1947 found the blanket `.ostk` noise entry starved
/// the audit_events ingest for 4 days — incremental mode never delivered
/// `journal.jsonl`, so audit/decisions/needles only refreshed on explicit
/// full `scan` runs. The rest of `.ostk/` (proc/ inboxes, locks/, gen
/// tables, journal-seals/ churn) stays filtered — that flood is the
/// reason the noise entry exists.
const OSTK_SIGNAL_BASENAMES: &[&str] = &[
    "journal.jsonl",
    "audit.jsonl",
    "decisions.jsonl",
    "issues.jsonl",
];

/// True if any component of `path` matches a noise segment exactly.
/// `.ostk`-resident sub-area files (see [`OSTK_SIGNAL_BASENAMES`]) are
/// exempt — they are the signal the ostk_project scanner exists to read.
/// Paths through `.ostk/vfs/` are unconditionally noise (→2040): vfs is a
/// loopback NFS mount served by the ostk daemon, so even a single
/// `is_file()` stat downstream becomes daemon work — and vfs mirrors
/// signal basenames like `issues.jsonl` that would otherwise ride the
/// exemption straight into `discover_paths`.
fn path_has_noise_segment(path: &Path) -> bool {
    if path_enters_ostk_vfs(path) {
        return true;
    }
    let noisy = path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .is_some_and(|s| NOISE_PATH_SEGMENTS.contains(&s))
    });
    if !noisy {
        return false;
    }
    // Exemption only applies when `.ostk` is the sole noise segment —
    // a journal.jsonl under target/ or .worktrees/ is still noise.
    let only_ostk_noise = path.components().all(|c| {
        c.as_os_str()
            .to_str()
            .is_none_or(|s| s == ".ostk" || !NOISE_PATH_SEGMENTS.contains(&s))
    });
    if only_ostk_noise
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| OSTK_SIGNAL_BASENAMES.contains(&n))
    {
        return false;
    }
    noisy
}

/// True if `path` contains the adjacent segments `.ostk/vfs` — the
/// daemon-served loopback mountpoint (→2040). Segment-exact, so a
/// project directory merely *named* `vfs` outside `.ostk` is unaffected.
fn path_enters_ostk_vfs(path: &Path) -> bool {
    let mut prev_was_ostk = false;
    for c in path.components() {
        let seg = c.as_os_str().to_str();
        if prev_was_ostk && seg == Some("vfs") {
            return true;
        }
        prev_was_ostk = seg == Some(".ostk");
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
mod noise_filter_tests {
    use std::path::Path;

    use super::path_has_noise_segment;

    #[test]
    fn ostk_signal_basenames_pass() {
        assert!(!path_has_noise_segment(Path::new(
            "/p/haystack/.ostk/journal.jsonl"
        )));
        assert!(!path_has_noise_segment(Path::new(
            "/p/haystack/.ostk/needles/issues.jsonl"
        )));
    }

    #[test]
    fn ostk_vfs_is_unconditional_noise() {
        // →2040: vfs mirrors signal basenames; the exemption must not
        // let them through — a downstream stat hits the NFS mount.
        assert!(path_has_noise_segment(Path::new(
            "/p/haystack/.ostk/vfs/needles/issues.jsonl"
        )));
        assert!(path_has_noise_segment(Path::new(
            "/p/haystack/.ostk/vfs/journal.jsonl"
        )));
        assert!(path_has_noise_segment(Path::new("/p/haystack/.ostk/vfs")));
    }

    #[test]
    fn vfs_segment_outside_ostk_is_not_noise() {
        assert!(!path_has_noise_segment(Path::new("/p/proj/vfs/notes.md")));
        assert!(!path_has_noise_segment(Path::new(
            "/p/proj/src/vfs/namespace.rs"
        )));
    }
}

#[cfg(test)]
mod rescan_gate_tests {
    use std::time::{Duration, Instant};

    use super::{RESCAN_KICK_RATE_LIMIT, RescanGate};

    #[test]
    fn idle_gate_never_emits() {
        let mut g = RescanGate::default();
        assert!(!g.try_emit(Instant::now()));
    }

    #[test]
    fn first_hint_emits_immediately() {
        let mut g = RescanGate::default();
        g.note();
        assert!(g.try_emit(Instant::now()));
        // Drained: no double-emit for the same hint.
        assert!(!g.try_emit(Instant::now()));
    }

    #[test]
    fn hint_inside_window_is_suppressed_then_recovers() {
        let mut g = RescanGate::default();
        let t0 = Instant::now();
        g.note();
        assert!(g.try_emit(t0));
        // Storm continues: a second hint lands inside the window.
        g.note();
        assert!(!g.try_emit(t0 + Duration::from_secs(1)));
        // →1966 guarantee: pending survives suppression, so once the
        // window passes the final recovery kick still fires — bounded
        // staleness, not an indefinite freeze.
        assert!(g.try_emit(t0 + RESCAN_KICK_RATE_LIMIT));
    }

    #[test]
    fn failed_dispatch_renote_retries_after_window() {
        let mut g = RescanGate::default();
        let t0 = Instant::now();
        g.note();
        assert!(g.try_emit(t0));
        // Caller's dispatch failed and re-noted; the retry obeys the
        // rate limit rather than hot-looping.
        g.note();
        assert!(!g.try_emit(t0 + Duration::from_secs(1)));
        assert!(g.try_emit(t0 + RESCAN_KICK_RATE_LIMIT));
    }
}

#[cfg(test)]
mod lens_config_tests {
    use ostk_recall_core::RelationalConfig;

    use super::{LensSettings, resolve_lens_config};

    /// The `[lens]` config defaults are duplicated in `core` (serde
    /// can't reach across to `query`). This guard fails loudly if the
    /// two ever drift, which would silently change daemon behavior for
    /// users who omit the block.
    #[test]
    fn lens_settings_default_matches_query_default() {
        let mapped = resolve_lens_config(Some(&LensSettings::default()), None);
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
        let mapped = resolve_lens_config(None, None);
        let dflt = ostk_recall_query::lens::LensConfig::default();
        assert_eq!(mapped.token_budget, dflt.token_budget);
        assert_eq!(mapped.poll_interval_secs, dflt.poll_interval_secs);
        // No `[relational]` block → the canonical core default floor.
        assert!(
            (mapped.latent_sim_floor - RelationalConfig::default().latent_sim_floor).abs()
                < f32::EPSILON
        );
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
        let mapped = resolve_lens_config(Some(&settings), None);
        assert_eq!(mapped.token_budget, 1234);
        assert_eq!(mapped.poll_interval_secs, 9);
        assert_eq!(mapped.exclude_facets, vec!["status:archived".to_string()]);
    }

    /// The `[relational]` floor maps onto `LensConfig.latent_sim_floor`.
    #[test]
    fn resolve_lens_config_maps_relational_floor() {
        let rel = RelationalConfig {
            latent_sim_floor: 0.2,
            promote_top_k: 4,
        };
        let mapped = resolve_lens_config(None, Some(&rel));
        assert!((mapped.latent_sim_floor - 0.2).abs() < f32::EPSILON);
    }
}

#[cfg(test)]
mod ambient_growth_config_tests {
    use ostk_recall_core::AmbientGrowthConfig;

    use super::resolve_concept_growth;

    /// The `[ambient_growth]` defaults are duplicated in `core` (serde can't
    /// reach across to `attention`). This guard fails loudly if the two ever
    /// drift, which would silently change ambient concept-growth behavior for
    /// users who omit the block.
    #[test]
    fn ambient_growth_default_matches_attention_default() {
        let mapped = resolve_concept_growth(Some(&AmbientGrowthConfig::default()));
        let dflt = ostk_recall_attention::ConceptGrowthConfig::default();
        assert!((mapped.resonance_floor - dflt.resonance_floor).abs() < f32::EPSILON);
        assert_eq!(mapped.edge_top_k, dflt.edge_top_k);
        assert_eq!(mapped.min_survivors, dflt.min_survivors);
        assert_eq!(
            mapped.node_mint_min_resonant_turns,
            dflt.node_mint_min_resonant_turns
        );
        assert_eq!(mapped.codebook_rebuild_turns, dflt.codebook_rebuild_turns);
        assert_eq!(
            mapped.node_mint_cap_per_session,
            dflt.node_mint_cap_per_session
        );
    }

    /// Absent `[ambient_growth]` block resolves to the runtime default verbatim.
    #[test]
    fn resolve_concept_growth_none_is_default() {
        let mapped = resolve_concept_growth(None);
        let dflt = ostk_recall_attention::ConceptGrowthConfig::default();
        assert!((mapped.resonance_floor - dflt.resonance_floor).abs() < f32::EPSILON);
        assert_eq!(mapped.edge_top_k, dflt.edge_top_k);
    }

    /// Explicit overrides flow through the mapping.
    #[test]
    fn resolve_concept_growth_applies_overrides() {
        let settings = AmbientGrowthConfig {
            resonance_floor: 0.42,
            edge_top_k: 6,
            min_survivors: 3,
            node_mint_min_resonant_turns: 5,
            codebook_rebuild_turns: 16,
            node_mint_cap_per_session: 2,
        };
        let mapped = resolve_concept_growth(Some(&settings));
        assert!((mapped.resonance_floor - 0.42).abs() < f32::EPSILON);
        assert_eq!(mapped.edge_top_k, 6);
        assert_eq!(mapped.min_survivors, 3);
        assert_eq!(mapped.node_mint_min_resonant_turns, 5);
        assert_eq!(mapped.codebook_rebuild_turns, 16);
        assert_eq!(mapped.node_mint_cap_per_session, 2);
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

    /// →1957 legs 1+2: a heal must (1) reconnect, (2) REPLAY the cached
    /// `initialize` (consuming its response so it never duplicates onto
    /// client stdout), (3) replay `notifications/initialized`, and
    /// (4) send the self-attesting `bridge_reconnected` receipt — in
    /// that order, all over the fresh socket.
    #[cfg(unix)]
    #[tokio::test]
    async fn reconnect_replays_handshake_then_sends_receipt() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let tmp = tempfile::TempDir::new().unwrap();
        let sock = tmp.path().join("recall-mcp.sock");
        let listener = tokio::net::UnixListener::bind(&sock).unwrap();

        let daemon = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (r, mut w) = stream.into_split();
            let mut lines = BufReader::new(r).lines();

            let init = lines.next_line().await.unwrap().unwrap();
            // Respond to the replayed initialize so the bridge can
            // consume (and discard) it.
            w.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n")
                .await
                .unwrap();
            let initialized = lines.next_line().await.unwrap().unwrap();
            let receipt = lines.next_line().await.unwrap().unwrap();
            (init, initialized, receipt)
        });

        let init_line = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}\n";
        let initialized_line = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
        let conn =
            super::reconnect_with_backoff(&sock, Some(init_line), Some(initialized_line))
                .await
                .expect("heal against live socket");
        drop(conn);

        let (init, initialized, receipt) = daemon.await.unwrap();
        assert!(init.contains("\"method\":\"initialize\""), "{init}");
        assert!(
            initialized.contains("notifications/initialized"),
            "{initialized}"
        );
        let receipt: serde_json::Value = serde_json::from_str(&receipt).unwrap();
        assert_eq!(receipt["method"], "notifications/bridge_reconnected");
        assert_eq!(receipt["params"]["attempts"], 1);
        assert!(receipt["params"]["bridge_pid"].as_u64().is_some());
    }

    /// `cache_handshake` must capture exactly the two handshake frames
    /// and ignore everything else.
    #[cfg(unix)]
    #[test]
    fn cache_handshake_captures_only_handshake_frames() {
        let mut init = None;
        let mut initialized = None;
        super::cache_handshake(
            "{\"method\":\"initialize\",\"id\":1}\n",
            &mut init,
            &mut initialized,
        );
        super::cache_handshake(
            "{\"method\":\"notifications/initialized\"}\n",
            &mut init,
            &mut initialized,
        );
        super::cache_handshake(
            "{\"method\":\"tools/call\",\"id\":2}\n",
            &mut init,
            &mut initialized,
        );
        super::cache_handshake("not json\n", &mut init, &mut initialized);
        assert!(init.unwrap().contains("initialize"));
        assert!(initialized.unwrap().contains("notifications/initialized"));
    }
}

#[cfg(test)]
mod replay_aggregate_tests {
    //! Replay must seed the project-agnostic AGGREGATE ambient scope the
    //! memory-lens polls (`ambient_scope_default()`) from the most recent
    //! ambient `RollingVectorSnapshot`, so a restarted daemon restores live
    //! lens attention immediately (before any new turn). The original
    //! wall-clock `ts` is not recoverable via `iter_chain`
    //! (`ChainEvent::from_row` synthesizes a decode-time `ts`), so "most
    //! recent" is the last ambient row in chronological chain order — the
    //! same supersede rule the per-scope `latest_rolling` map relies on.
    use super::*;
    use chrono::Utc;
    use ostk_recall_store::ChainEvent;
    use tempfile::TempDir;

    fn ambient_project_scope(project: &str) -> AttentionScope {
        AttentionScope {
            project: Some(project.into()),
            session_id: Some("ambient".into()),
            agent: Some("substrate".into()),
            privacy_tier: PrivacyTier::T1Project,
        }
    }

    fn snapshot(scope: AttentionScope, vec: Vec<f32>) -> ChainEvent {
        ChainEvent::RollingVectorSnapshot {
            scope,
            vec,
            lambda: 0.3,
            ts: Utc::now(),
        }
    }

    #[tokio::test]
    async fn replay_seeds_aggregate_from_latest_ambient_snapshot() {
        let tmp = TempDir::new().unwrap();
        // `ThreadsDb::open` installs a Noop sink; write chain rows through a
        // real `SqliteChainSink` on the same file so `iter_chain()` reads them
        // back (it decodes payloads only, in seq-ASC = chronological order).
        let db = ThreadsDb::open(tmp.path()).unwrap();
        let sink = SqliteChainSink::open(tmp.path()).unwrap();

        let vec_a = vec![0.0_f32, 1.0, 0.0];
        let vec_b = vec![1.0_f32, 0.0, 0.0];
        let vec_other = vec![0.0_f32, 0.0, 1.0];

        // Two ambient snapshots under DIFFERENT projects (proves the aggregate
        // is cross-project), proj-b last → most recent ambient. Then a
        // NON-ambient snapshot appended LAST overall: it must be ignored even
        // though it is the final row, proving the session/agent filter.
        sink.append(&snapshot(ambient_project_scope("proj-a"), vec_a))
            .unwrap();
        sink.append(&snapshot(ambient_project_scope("proj-b"), vec_b.clone()))
            .unwrap();
        let non_ambient = AttentionScope {
            project: Some("proj-c".into()),
            session_id: Some("interactive".into()),
            agent: Some("operator".into()),
            privacy_tier: PrivacyTier::T1Project,
        };
        sink.append(&snapshot(non_ambient, vec_other)).unwrap();

        let attention = InMemoryAttention::new();
        replay_chain_into_attention(&Arc::new(db), &attention)
            .await
            .unwrap();

        let aggregate = attention
            .rolling_vec(&ambient_scope_default())
            .await
            .unwrap();
        assert_eq!(
            aggregate,
            Some(vec_b),
            "aggregate must seed from the most recent ambient snapshot (cross-project), \
             ignoring the later non-ambient row"
        );
    }
}
