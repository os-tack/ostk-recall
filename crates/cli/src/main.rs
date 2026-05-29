//! `ostk-recall` CLI — Phase B binary.
//!
//! Handlers live in [`ostk_recall_cli::commands`]; this module parses args,
//! constructs a real [`ostk_recall_embed::Embedder`], and delegates.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ostk_recall_attention::{AttentionForwardStore, InMemoryAttention};
use ostk_recall_attention_mcp::{AttentionDispatch, cli as attn_cli};
use ostk_recall_cli::commands::{self, InitOptions, InitOutcome};
use ostk_recall_core::{Config, default_worker_threads};
use ostk_recall_embed::Embedder;
use ostk_recall_pipeline::ChunkEmbedder;
use ostk_recall_store::ThreadsDb;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "ostk-recall",
    version,
    about = "Local, private semantic recall"
)]
struct Cli {
    /// Path to config file. Defaults to
    /// `${XDG_CONFIG_HOME:-~/.config}/ostk-recall/config.toml`.
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Initialize a new ostk-recall corpus using the resolved config.
    Init {
        /// Wipe `corpus.lance/`, `ingest.duckdb`, `events.duckdb` from the
        /// configured corpus root before re-initializing. Best-effort —
        /// missing files are not an error.
        #[arg(long)]
        force: bool,
    },
    /// Scan configured sources and ingest chunks.
    Scan {
        /// Optional source project filter; scans all when omitted.
        #[arg(long, conflicts_with = "reingest")]
        source: Option<String>,
        /// Wipe every corpus + ingest row whose `project` matches this
        /// value, then rescan just that source. Implies `--source=<NAME>`.
        /// Use when a chunker / parser change needs to propagate to
        /// already-ingested data without a full `init --force` rebuild.
        #[arg(long)]
        reingest: Option<String>,
        /// Report what would be ingested without writing to the store.
        /// Ignored when combined with `--reingest` (the delete is
        /// never dry — if you asked for it, we do it).
        #[arg(long)]
        dry_run: bool,
    },
    /// Verify corpus integrity (counts).
    Verify,
    /// Run Lance's `OptimizeAction::All` against the corpus — compact
    /// small fragments, prune old versions, fold appended data into
    /// existing scalar / FTS indices. Cheap to re-run; idempotent. Use
    /// after a long ingest backlog to restore fast indexed lookups.
    Optimize {
        /// Also prune ALL historical versions, collapsing the corpus to
        /// its latest version. Overrides Lance's default ~14-day
        /// retention (the normal pass leaves recent versions in place),
        /// so this is how you undo a version explosion from a heavy scan.
        /// ONLY safe when nothing else is writing the corpus — stop
        /// `serve`/`watch` and any running `scan` first.
        #[arg(long)]
        aggressive: bool,
    },
    /// Inspect a single chunk by id.
    Inspect {
        /// Chunk id to inspect.
        chunk_id: String,
    },
    /// Serve the MCP endpoint.
    Serve {
        /// Use stdio transport.
        #[arg(long)]
        stdio: bool,
    },
    /// Watch configured source paths and poke the running `serve`'s
    /// scan-trigger socket whenever a debounced batch of events lands.
    /// Requires `[watch].enabled = true` in config; reuses each
    /// `[[sources]].paths` and `extensions`.
    Watch,
    /// Attention substrate verbs (forward-attention runtime).
    Attention {
        /// Emit JSON instead of the human-readable summary.
        #[arg(long, global = true)]
        json: bool,
        #[command(subcommand)]
        verb: AttentionVerb,
    },
    /// Thread ledger verbs (durable thread + evidence ledger).
    Thread {
        #[arg(long, global = true)]
        json: bool,
        #[command(subcommand)]
        verb: ThreadVerb,
    },
    /// Manifest tooling (recover `ingest.sqlite` from `corpus.lance`).
    Manifest {
        #[command(subcommand)]
        verb: ManifestVerb,
    },
    /// Memory-lens controls (P9b-min). The lens is the daemon's
    /// background-injected MCP resource; these verbs read its state
    /// from the local recall directory without spawning a client.
    Lens {
        #[command(subcommand)]
        verb: LensVerb,
    },
}

#[derive(Debug, Subcommand)]
enum LensVerb {
    /// Dump the most recently rendered lens markdown. Reads
    /// `{root}/lens.md`, the side-of-disk copy the loop writes
    /// alongside each registry update. Prints an empty-state hint
    /// when the file is absent (daemon hasn't refreshed yet, or
    /// the daemon was started with OSTK_RECALL_LENS_DISABLED).
    Show,
    /// Print the env-var incantation that disables the lens for
    /// the next `serve` invocation. The disable is per-session by
    /// design — bake-in opt-out behaviour requires touching
    /// config.toml directly.
    Disable,
}

#[derive(Debug, Subcommand)]
enum ManifestVerb {
    /// Reconstruct `ingest.sqlite` from `corpus.lance` rows alone.
    ///
    /// Use after a partial copy or disk failure that lost the SQLite
    /// ledger but preserved the Lance directory. The rebuild populates
    /// `ingest_chunks` + `ingest_sources` from the embedded
    /// `source_config_id` column (P0); `mtime/size` are stamped as 0
    /// and refresh on the next scan.
    Rebuild,
}

#[derive(Debug, Subcommand)]
enum AttentionVerb {
    /// Ingest current context into the attention vector for `--scope-project`.
    Attend {
        #[arg(long)]
        scope_project: Option<String>,
        #[arg(long)]
        session_id: Option<String>,
        #[arg(long)]
        agent: Option<String>,
        #[arg(long)]
        privacy_tier: Option<String>,
        /// Inline context. Mutually exclusive with `--context-from-stdin`.
        #[arg(long, conflicts_with = "context_from_stdin")]
        context: Option<String>,
        /// Read the context from stdin to EOF.
        #[arg(long)]
        context_from_stdin: bool,
    },
    /// Surface attention pages above the archive threshold.
    Surface {
        #[arg(long)]
        scope_project: Option<String>,
        #[arg(long)]
        session_id: Option<String>,
        #[arg(long)]
        agent: Option<String>,
        #[arg(long)]
        privacy_tier: Option<String>,
        #[arg(long)]
        limit: Option<u64>,
    },
    /// Set fold depth for a thread handle within the scope.
    Fold {
        #[arg(long)]
        scope_project: Option<String>,
        #[arg(long)]
        session_id: Option<String>,
        #[arg(long)]
        agent: Option<String>,
        #[arg(long)]
        privacy_tier: Option<String>,
        #[arg(long)]
        handle: String,
        /// folded | half | full
        #[arg(long)]
        depth: String,
    },
    /// Increment familiarity for a handle within the scope.
    Familiarize {
        #[arg(long)]
        scope_project: Option<String>,
        #[arg(long)]
        session_id: Option<String>,
        #[arg(long)]
        agent: Option<String>,
        #[arg(long)]
        privacy_tier: Option<String>,
        #[arg(long)]
        handle: String,
    },
    /// Apply a multiplicative fade factor to a handle's floor.
    Decay {
        #[arg(long)]
        handle: String,
        #[arg(long)]
        factor: f64,
    },
}

#[derive(Debug, Subcommand)]
enum ThreadVerb {
    /// Insert-or-replace a thread row.
    Create {
        #[arg(long)]
        scope_project: Option<String>,
        #[arg(long)]
        handle: String,
        #[arg(long)]
        body_from_file: Option<PathBuf>,
        #[arg(long)]
        tension: Option<String>,
    },
    /// Add a curated evidence link.
    Link {
        #[arg(long)]
        scope_project: Option<String>,
        #[arg(long)]
        handle: String,
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        category: String,
    },
    /// Remove an evidence row by id.
    Unlink {
        #[arg(long)]
        evidence_id: i64,
    },
    /// Promote a proposed thread to active or slack.
    Promote {
        #[arg(long = "from")]
        handle_from_proposed: String,
        #[arg(long = "to")]
        target_tier: String,
    },
    /// List threads, optionally filtered by tension.
    List {
        #[arg(long)]
        scope_project: Option<String>,
        #[arg(long)]
        tension: Option<String>,
    },
    /// List emergent-thread proposals that the auto-weaver detected
    /// from chunk clusters but no operator has promoted yet.
    ProposedList {
        /// Limit the number of rows shown (most recent first).
        #[arg(long)]
        limit: Option<usize>,
    },
}

fn load_embedder(model_id: &str) -> Result<Arc<dyn ChunkEmbedder>> {
    // Escape hatch for tests / offline smoke: OSTK_RECALL_FAKE_EMBEDDER=<dim>
    // yields a deterministic toy embedder instead of pulling a real model.
    if let Ok(dim_str) = std::env::var("OSTK_RECALL_FAKE_EMBEDDER") {
        let dim: usize = dim_str
            .parse()
            .with_context(|| format!("OSTK_RECALL_FAKE_EMBEDDER parse: {dim_str}"))?;
        tracing::warn!(dim, "using fake embedder — do not use in production");
        return Ok(Arc::new(TestFakeEmbedder { dim }));
    }
    let embedder = Embedder::load(model_id).context("loading embedder")?;
    Ok(Arc::new(embedder))
}

/// Deterministic length-bucket embedder used only when
/// `OSTK_RECALL_FAKE_EMBEDDER=<dim>` is set. Kept in the binary (not a
/// public API) so integration tests can exercise the real subprocess path.
struct TestFakeEmbedder {
    dim: usize,
}

impl ChunkEmbedder for TestFakeEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0; self.dim];
                let bucket = t.len() % self.dim;
                v[bucket] = 1.0;
                v
            })
            .collect()
    }
}

fn resolve_embedder(config: Option<&PathBuf>) -> Result<Arc<dyn ChunkEmbedder>> {
    // Peek at the config to learn which model to load. If the config file
    // doesn't exist yet, fall back to the default.
    let path = match config {
        Some(p) => p.clone(),
        None => commands::default_config_path()?,
    };
    let model_id = if path.exists() {
        let cfg = ostk_recall_core::Config::load(&path)
            .with_context(|| format!("loading config from {}", path.display()))?;
        cfg.embedder.model
    } else {
        "minishlab/potion-retrieval-32M".to_string()
    };
    load_embedder(&model_id)
}

/// Resolve the runtime worker-thread cap.
///
/// Precedence: `OSTK_RECALL_WORKERS` env var > `[runtime].worker_threads`
/// in config > [`default_worker_threads`] (4). Returns the effective
/// value plus whether it was overridden, so we can log the source.
///
/// Loading the config here is a deliberate up-front cost paid before
/// the tokio runtime is built — we need the value to construct the
/// runtime with the right `worker_threads`, and the same value sets
/// `DATAFUSION_TARGET_PARTITIONS` / `RAYON_NUM_THREADS` env vars
/// which must be in place before the first lance / rayon call.
fn resolve_worker_threads(config_path: Option<&Path>) -> usize {
    if let Ok(v) = std::env::var("OSTK_RECALL_WORKERS") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
        eprintln!(
            "warning: OSTK_RECALL_WORKERS={v:?} could not be parsed as a positive integer; \
             falling back to config / default"
        );
    }
    if let Some(p) = config_path {
        if let Ok(cfg) = Config::load(p) {
            if let Some(n) = cfg.runtime.as_ref().and_then(|r| r.worker_threads) {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    default_worker_threads()
}

fn main() -> Result<()> {
    // CLI parsing happens before the runtime is built so we can read
    // `--config` and use it to resolve runtime resource caps. Errors
    // here short-circuit cleanly without spinning up tokio.
    let cli = Cli::parse();
    let config_path_for_runtime = cli.config.clone();

    let workers = resolve_worker_threads(config_path_for_runtime.as_deref());

    // Cap the global rayon pool. Lance pushes compute (protobuf
    // decoding, percent-encoding object paths, ingest batch
    // processing, index maintenance) onto rayon's default pool, which
    // sizes itself to `num_cpus()` unless we override here. Calling
    // `build_global()` is the safe, programmatic equivalent of setting
    // `RAYON_NUM_THREADS=N` — we prefer it over `std::env::set_var`
    // because the workspace forbids `unsafe_code`. `build_global` can
    // only succeed once per process; doing it here, before any lance
    // call, ensures we win the race. The `Err` arm is a no-op: it
    // means rayon was already initialized (e.g. in a test harness),
    // and we accept whatever sizing the prior caller chose.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("build tokio runtime: {e}"))?;
    rt.block_on(async_main(cli, workers))
}

#[allow(clippy::too_many_lines)]
async fn async_main(cli: Cli, worker_threads: usize) -> Result<()> {
    // Stdout is reserved for tool output (and JSON-RPC frames in
    // `serve --stdio`); logs go to stderr unconditionally so MCP clients
    // never see interleaved log noise on stdout.
    //
    // The writer is non-blocking: a background thread drains a bounded
    // queue and does the actual stderr writes, so the foreground tracing
    // path can never deadlock the tokio runtime when the consumer (pipe
    // / terminal / log forwarder) goes slow. Default policy is lossy —
    // a saturated queue drops lines rather than back-pressuring callers.
    // `_trace_guard` flushes pending events on drop; bind it to `main`'s
    // stack frame so it lives until shutdown.
    let (non_blocking_stderr, _trace_guard) =
        tracing_appender::non_blocking(std::io::stderr());
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(non_blocking_stderr)
        .init();

    tracing::info!(
        worker_threads,
        "runtime caps: tokio worker_threads + rayon global pool"
    );

    // Raise the open-file soft limit before any Lance work. macOS
    // ships a 256-fd soft cap (`launchctl limit maxfiles`) that
    // GUI-spawned processes inherit — the MCP server under Claude
    // Code, scan subprocesses. Lance opens many fragment files when
    // reading a large corpus, exhausting 256 mid-scan and surfacing
    // as `Too many open files (os error 24)` in the weaver /
    // turn-observer; that in turn starves the ambient attention
    // scope the memory-lens watches. `increase_nofile_limit` clamps
    // to the hard limit (and to `kern.maxfilesperproc` on macOS), so
    // the worst case is a no-op. Best-effort: log and continue.
    match rlimit::increase_nofile_limit(u64::MAX) {
        Ok(limit) => tracing::info!(nofile_soft = limit, "raised RLIMIT_NOFILE soft limit"),
        Err(e) => tracing::warn!(
            error = %e,
            "could not raise RLIMIT_NOFILE; large scans may hit os error 24"
        ),
    }

    let config_path = match cli.config.clone() {
        Some(p) => p,
        None => commands::default_config_path()?,
    };

    match cli.command {
        Command::Init { force } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            // Production init: prefetch the reranker so the first `serve`
            // doesn't pay the download latency mid-session.
            let opts = InitOptions::default()
                .with_force(force)
                .with_prefetch_reranker(true);
            let outcome = commands::init_with_options(&config_path, embedder, opts).await?;
            match outcome {
                InitOutcome::WroteStarter { path } => {
                    println!(
                        "wrote starter config at {}; please edit, then re-run",
                        path.display()
                    );
                }
                InitOutcome::Initialized {
                    root,
                    model_id,
                    dim,
                } => {
                    println!(
                        "initialized at {} with model {model_id} (dim={dim})",
                        root.display()
                    );
                }
            }
        }
        Command::Scan {
            source,
            reingest,
            dry_run,
        } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            let out = if let Some(project) = reingest.as_deref() {
                commands::scan_reingest(&config_path, embedder, project).await?
            } else {
                commands::scan(&config_path, embedder, source.as_deref(), dry_run).await?
            };
            println!("scan summary (dry_run={}):", out.dry_run);
            for (name, s) in &out.per_source {
                println!(
                    "  {name}: items={} chunks={} upserted={} dup={} errors={}",
                    s.items_seen,
                    s.chunks_emitted,
                    s.chunks_upserted,
                    s.chunks_skipped_dup,
                    s.errors
                );
            }
            let t = out.totals;
            println!(
                "  total: items={} chunks={} upserted={} dup={} errors={}",
                t.items_seen, t.chunks_emitted, t.chunks_upserted, t.chunks_skipped_dup, t.errors
            );
        }
        Command::Optimize { aggressive } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            if aggressive {
                println!(
                    "running aggressive optimize (compact + index + prune ALL old versions)…"
                );
            } else {
                println!("running OptimizeAction::All (compact + prune + index)…");
            }
            let out = commands::optimize(&config_path, embedder, aggressive).await?;
            match out.versions_pruned {
                Some(n) => println!(
                    "done in {:.1}s; pruned {n} old versions",
                    out.elapsed.as_secs_f64()
                ),
                None => println!("done in {:.1}s", out.elapsed.as_secs_f64()),
            }
        }
        Command::Verify => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            let out = commands::verify(&config_path, embedder).await?;
            let r = &out.report;
            println!(
                "verify: corpus_rows={} ingest_rows={}",
                r.corpus_total, r.ingest_total
            );
            for (source, n) in &r.by_source {
                println!("  {source}: {n}");
            }
            if !r.is_consistent() {
                eprintln!("drift: corpus={} ingest={}", r.corpus_total, r.ingest_total);
                std::process::exit(1);
            }
        }
        Command::Inspect { chunk_id } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            let result = commands::inspect(&config_path, embedder, &chunk_id).await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Serve { stdio } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            commands::serve(&config_path, embedder, stdio).await?;
        }
        Command::Watch => {
            commands::watch(&config_path).await?;
        }
        Command::Attention { json, verb } => {
            let dispatch = build_attention_dispatch(&config_path)?;
            let out = run_attention(&dispatch, verb).await?;
            print_attention_output(json, &out);
        }
        Command::Thread { json, verb } => {
            let dispatch = build_attention_dispatch(&config_path)?;
            let out = run_thread(&dispatch, verb).await?;
            print_attention_output(json, &out);
        }
        Command::Manifest { verb } => {
            run_manifest(&config_path, verb).await?;
        }
        Command::Lens { verb } => {
            run_lens(&config_path, verb)?;
        }
    }

    Ok(())
}

/// Dispatch `ostk-recall lens <verb>`.
fn run_lens(config_path: &Path, verb: LensVerb) -> Result<()> {
    match verb {
        LensVerb::Show => {
            let cfg = ostk_recall_core::Config::load(config_path)
                .with_context(|| format!("loading config from {}", config_path.display()))?;
            let root = cfg.expanded_root()?;
            let lens_md = root.join(ostk_recall_cli::lens_loop::LENS_MARKDOWN_FILE);
            if !lens_md.exists() {
                println!(
                    "_No lens rendered yet._ Start `ostk-recall serve --stdio` and wait for the first refresh.\n\
                     (Empty-mind boot, OSTK_RECALL_LENS_DISABLED, or simply no attention drift since the daemon started.)"
                );
                return Ok(());
            }
            let body = std::fs::read_to_string(&lens_md)
                .with_context(|| format!("reading {}", lens_md.display()))?;
            print!("{body}");
            if !body.ends_with('\n') {
                println!();
            }
        }
        LensVerb::Disable => {
            // Per-session opt-out. Documented in p9b-lens-portfolio.md
            // "Privacy + control". Baking-in disabled state requires
            // editing config.toml; we deliberately don't muck with the
            // user's config from a CLI verb.
            println!(
                "Set OSTK_RECALL_LENS_DISABLED=1 in the environment before launching\n\
                 `ostk-recall serve --stdio` to disable the memory-lens daemon for the\n\
                 lifetime of that serve invocation. For a persistent opt-out edit the\n\
                 [lens] section of your config.toml (planned for v0.6.0-rc.1)."
            );
        }
    }
    Ok(())
}

/// Dispatch `ostk-recall manifest <verb>`.
async fn run_manifest(config_path: &Path, verb: ManifestVerb) -> Result<()> {
    match verb {
        ManifestVerb::Rebuild => {
            let cfg = ostk_recall_core::Config::load(config_path)
                .with_context(|| format!("loading config from {}", config_path.display()))?;
            let root = cfg.expanded_root()?;
            std::fs::create_dir_all(&root)
                .with_context(|| format!("creating corpus root {}", root.display()))?;
            // Open both stores; the rebuild reads from corpus.lance and
            // writes into ingest.sqlite. If ingest.sqlite holds the
            // legacy v0.5 schema, the open path errors loudly per P0.
            let store = ostk_recall_store::CorpusStore::open_or_create(
                &root,
                // Dim is fixed at corpus-create time. We open with the
                // embedder's nominal dim; rebuild doesn't read embeddings.
                0,
            )
            .await
            .map_err(|e| anyhow::anyhow!("open corpus: {e}"))?;
            let ingest = ostk_recall_store::IngestDb::open(&root)
                .map_err(|e| anyhow::anyhow!("open ingest ledger: {e}"))?;
            let run_id = format!(
                "manifest-rebuild-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
            );
            let n = ostk_recall_store::rebuild_ingest_manifest(&store, &ingest, &run_id)
                .await
                .map_err(|e| anyhow::anyhow!("rebuild_ingest_manifest: {e}"))?;
            println!("rebuilt {n} chunks into ingest.sqlite (run_id={run_id})");
            println!("next step: run a full `ostk-recall scan` to refresh mtime/size metadata.");
        }
    }
    Ok(())
}

/// Build the attention dispatch from the resolved config.
///
/// The score tier is an `InMemoryAttention` (per Phase 2 — the chain
/// rebuilds it on boot in the daemon; the CLI uses an empty store for
/// one-shot verbs). The thread ledger is the durable
/// `<root>/threads.sqlite`.
fn build_attention_dispatch(config_path: &std::path::Path) -> Result<AttentionDispatch> {
    let cfg = ostk_recall_core::Config::load(config_path)
        .with_context(|| format!("loading config from {}", config_path.display()))?;
    let root = cfg.expanded_root()?;
    std::fs::create_dir_all(&root)
        .with_context(|| format!("creating corpus root {}", root.display()))?;
    let threads =
        Arc::new(ThreadsDb::open(&root).map_err(|e| anyhow::anyhow!("open threads ledger: {e}"))?);
    let attention: Arc<dyn AttentionForwardStore> = Arc::new(InMemoryAttention::new());
    Ok(AttentionDispatch::new(attention, threads))
}

async fn run_attention(d: &AttentionDispatch, verb: AttentionVerb) -> Result<serde_json::Value> {
    let v = match verb {
        AttentionVerb::Attend {
            scope_project,
            session_id,
            agent,
            privacy_tier,
            context,
            context_from_stdin,
        } => {
            let ctx = resolve_context(context, context_from_stdin)?;
            attn_cli::run_attend(d, scope_project, session_id, agent, privacy_tier, ctx)
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        }
        AttentionVerb::Surface {
            scope_project,
            session_id,
            agent,
            privacy_tier,
            limit,
        } => attn_cli::run_surface(d, scope_project, session_id, agent, privacy_tier, limit)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        AttentionVerb::Fold {
            scope_project,
            session_id,
            agent,
            privacy_tier,
            handle,
            depth,
        } => attn_cli::run_fold(
            d,
            scope_project,
            session_id,
            agent,
            privacy_tier,
            handle,
            depth,
        )
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        AttentionVerb::Familiarize {
            scope_project,
            session_id,
            agent,
            privacy_tier,
            handle,
        } => attn_cli::run_familiarize(d, scope_project, session_id, agent, privacy_tier, handle)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        AttentionVerb::Decay { handle, factor } => attn_cli::run_decay(d, handle, factor)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
    };
    Ok(v)
}

async fn run_thread(d: &AttentionDispatch, verb: ThreadVerb) -> Result<serde_json::Value> {
    let v = match verb {
        ThreadVerb::Create {
            scope_project,
            handle,
            body_from_file,
            tension,
        } => {
            let body = body_from_file
                .as_deref()
                .map(std::fs::read_to_string)
                .transpose()
                .with_context(|| format!("reading body from {body_from_file:?}"))?;
            attn_cli::run_thread_create(d, scope_project, handle, body, tension)
                .await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        }
        ThreadVerb::Link {
            scope_project,
            handle,
            target,
            category,
        } => attn_cli::run_thread_link(
            d,
            scope_project,
            handle,
            target.to_string_lossy().into_owned(),
            category,
        )
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        ThreadVerb::Unlink { evidence_id } => attn_cli::run_thread_unlink(d, evidence_id)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        ThreadVerb::Promote {
            handle_from_proposed,
            target_tier,
        } => attn_cli::run_thread_promote(d, handle_from_proposed, target_tier)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        ThreadVerb::List {
            scope_project,
            tension,
        } => attn_cli::run_thread_list(d, scope_project, tension)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        ThreadVerb::ProposedList { limit } => run_thread_proposed_list(d, limit)?,
    };
    Ok(v)
}

/// Read the threads ledger directly for the proposed-list verb.
///
/// Proposed threads have no MCP surface in V1; we go straight to the
/// `ThreadsDb` the dispatch was built with, format the rows as JSON,
/// and let the outer printer render them.
fn run_thread_proposed_list(
    d: &AttentionDispatch,
    limit: Option<usize>,
) -> Result<serde_json::Value> {
    let proposals = d
        .threads
        .list_proposed_threads()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let rows: Vec<serde_json::Value> = proposals
        .into_iter()
        .take(limit.unwrap_or(usize::MAX))
        .map(|p| {
            serde_json::json!({
                "id": p.id,
                "handle": p.proposed_handle,
                "chunk_count": p.chunk_ids.len(),
                "chunks": p.chunk_ids,
                "cohesion": p.cohesion,
                "created_at": p.created_at.to_rfc3339(),
                "promoted_to": p.promoted_to,
            })
        })
        .collect();
    Ok(serde_json::json!({ "proposals": rows }))
}

fn resolve_context(inline: Option<String>, from_stdin: bool) -> Result<String> {
    if let Some(s) = inline {
        return Ok(s);
    }
    if from_stdin {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        return Ok(buf);
    }
    Err(anyhow::anyhow!(
        "one of --context or --context-from-stdin is required"
    ))
}

fn print_attention_output(json: bool, value: &serde_json::Value) {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        );
    } else {
        // Human-readable rendering for the common cases; falls back
        // to JSON for unknown shapes.
        if let Some(pages) = value.get("pages").and_then(|v| v.as_array()) {
            println!("pages ({}):", pages.len());
            for p in pages {
                println!(
                    "  {}: score={:.3} depth={} resonance={:.3} familiarity={}",
                    p["handle"].as_str().unwrap_or("?"),
                    p["score"].as_f64().unwrap_or(0.0),
                    p["depth"].as_str().unwrap_or("?"),
                    p["why"]["resonance"].as_f64().unwrap_or(0.0),
                    p["why"]["familiarity"].as_u64().unwrap_or(0),
                );
            }
        } else if let Some(records) = value.get("records").and_then(|v| v.as_array()) {
            println!("threads ({}):", records.len());
            for r in records {
                println!(
                    "  {}: tension={} familiarity={} privacy_tier={}",
                    r["handle"].as_str().unwrap_or("?"),
                    r["tension"].as_str().unwrap_or("?"),
                    r["familiarity"].as_u64().unwrap_or(0),
                    r["privacy_tier"].as_str().unwrap_or("?"),
                );
            }
        } else if let Some(rec) = value.get("record") {
            println!(
                "thread: {}",
                serde_json::to_string_pretty(rec).unwrap_or_default()
            );
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
            );
        }
    }
}
