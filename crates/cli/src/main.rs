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
    /// Run the explicit windowed auto-weaver pass over indexed content.
    ///
    /// This is the bulk/library counterpart to live TurnEnd weaving:
    /// it does not make bulk scans look like watched conversation turns.
    Weave {
        /// Only weave chunks newer than this duration, e.g. 1h, 24h, 7d.
        /// Omit for the whole active corpus. With `--consolidate` this is
        /// the horizon of the consolidation layer (the cron tier picks it).
        #[arg(long)]
        since: Option<String>,
        /// Number of chunk ids per synthesized source-kind batch.
        #[arg(long, default_value_t = 256)]
        epoch_size: usize,
        /// Run a coarse **consolidation** cycle, not just capture: after the
        /// windowed re-weave, bridge consolidated threads across canyons,
        /// promote recurring high-cohesion proposals to durable threads, and
        /// fade idle threads. Offline policy — schedule under cron/launchd at
        /// the cadences in `config.example.toml`, never inside `serve`.
        #[arg(long)]
        consolidate: bool,
    },
    /// Inspect a single chunk by id.
    Inspect {
        /// Chunk id to inspect.
        chunk_id: String,
    },
    /// Serve the MCP endpoint.
    ///
    /// Without `--stdio`, runs as a standalone daemon: serves MCP to
    /// many clients over a local socket / named pipe (dial it with
    /// `ostk-recall connect`) and keeps the corpus fresh in the
    /// background. With `--stdio`, this process is itself the MCP
    /// server, talking JSON-RPC over its own stdin/stdout.
    Serve {
        /// Use stdio transport (this process is the MCP server).
        #[arg(long)]
        stdio: bool,
        /// Daemon mode only: also run the filesystem watcher in-process
        /// so file changes trigger a rescan without a separate
        /// `ostk-recall watch`. Ignored under `--stdio`.
        #[arg(long)]
        watch: bool,
    },
    /// Watch configured source paths and poke the running `serve`'s
    /// scan-trigger socket whenever a debounced batch of events lands.
    /// Requires `[watch].enabled = true` in config; reuses each
    /// `[[sources]].paths` and `extensions`.
    Watch,
    /// Bridge a stdio MCP client to a running `serve` daemon.
    ///
    /// A dumb byte pump: splices this process's stdin/stdout to the
    /// daemon's MCP socket / named pipe. No engine, no corpus lock, no
    /// scanning — point a stdio-only MCP client (Claude Code/Desktop)
    /// at `ostk-recall connect` so it reaches the shared daemon instead
    /// of spawning its own `serve`.
    Connect,
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
    let (non_blocking_stderr, _trace_guard) = tracing_appender::non_blocking(std::io::stderr());
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
            // The TurnEnd gate means bulk-scanned content is NOT woven into the
            // thread graph by the live daemon — only watched conversation turns
            // are. Surface the bridge so a fresh scan isn't a silent, un-woven
            // corpus: point the operator at the explicit consolidation pass.
            if !out.dry_run && t.chunks_upserted > 0 {
                println!(
                    "  -> {} new chunk(s) indexed but not yet woven; run `ostk-recall weave` to weave them into the thread graph",
                    t.chunks_upserted
                );
            }
        }
        Command::Optimize { aggressive } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            if aggressive {
                println!("running aggressive optimize (compact + index + prune ALL old versions)…");
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
        Command::Weave {
            since,
            epoch_size,
            consolidate,
        } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            let since = since.as_deref().map(parse_since_duration).transpose()?;
            if consolidate {
                let out =
                    commands::consolidate(&config_path, embedder, since, epoch_size).await?;
                let w = &out.window;
                println!(
                    "consolidate summary: batches={} chunks={} evidence_links={} evidence_touched={} proposed_threads={} proposals_pruned={} | bridges_written={} bridges_touched={} proposals_promoted={} threads_faded={}",
                    w.batches_processed,
                    w.chunks_seen,
                    w.evidence_links_written,
                    w.evidence_links_touched,
                    w.proposed_threads_written,
                    w.proposals_pruned,
                    out.anchor_bridges_written,
                    out.anchor_bridges_touched,
                    out.proposals_promoted,
                    out.threads_faded,
                );
            } else {
                let out = commands::weave(&config_path, embedder, since, epoch_size).await?;
                println!(
                    "weave summary: batches={} chunks={} evidence_links={} evidence_touched={} proposed_weaves={} proposed_threads={} proposals_pruned={}",
                    out.batches_processed,
                    out.chunks_seen,
                    out.evidence_links_written,
                    out.evidence_links_touched,
                    out.proposed_weaves,
                    out.proposed_threads_written,
                    out.proposals_pruned
                );
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
        Command::Serve { stdio, watch } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            commands::serve(&config_path, embedder, stdio, watch).await?;
        }
        Command::Watch => {
            commands::watch(&config_path).await?;
        }
        Command::Connect => {
            commands::connect(&config_path).await?;
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

fn parse_since_duration(input: &str) -> Result<std::time::Duration> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        anyhow::bail!("--since must not be empty; use values like 1h, 24h, 7d");
    }
    let split = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (num, unit) = trimmed.split_at(split);
    if num.is_empty() || unit.is_empty() || !unit.chars().all(|c| c.is_ascii_alphabetic()) {
        anyhow::bail!("invalid --since {input:?}; use values like 1h, 24h, 7d");
    }
    let value: u64 = num
        .parse()
        .with_context(|| format!("invalid --since number in {input:?}"))?;
    let seconds = match unit {
        "s" | "sec" | "secs" => value,
        "m" | "min" | "mins" => value
            .checked_mul(60)
            .ok_or_else(|| anyhow::anyhow!("--since is too large"))?,
        "h" | "hr" | "hrs" => value
            .checked_mul(60 * 60)
            .ok_or_else(|| anyhow::anyhow!("--since is too large"))?,
        "d" | "day" | "days" => value
            .checked_mul(24 * 60 * 60)
            .ok_or_else(|| anyhow::anyhow!("--since is too large"))?,
        "w" | "week" | "weeks" => value
            .checked_mul(7 * 24 * 60 * 60)
            .ok_or_else(|| anyhow::anyhow!("--since is too large"))?,
        _ => anyhow::bail!("invalid --since unit {unit:?}; use s, m, h, d, or w"),
    };
    Ok(std::time::Duration::from_secs(seconds))
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

#[cfg(test)]
mod tests {
    use super::parse_since_duration;

    #[test]
    fn parse_since_duration_accepts_supported_units() {
        assert_eq!(parse_since_duration("30s").unwrap().as_secs(), 30);
        assert_eq!(parse_since_duration("15min").unwrap().as_secs(), 900);
        assert_eq!(parse_since_duration("24h").unwrap().as_secs(), 86_400);
        assert_eq!(parse_since_duration("7d").unwrap().as_secs(), 604_800);
        assert_eq!(parse_since_duration("2weeks").unwrap().as_secs(), 1_209_600);
    }

    #[test]
    fn parse_since_duration_rejects_invalid_forms() {
        assert!(parse_since_duration("").is_err());
        assert!(parse_since_duration("h").is_err());
        assert!(parse_since_duration("12").is_err());
        assert!(parse_since_duration("12 months").is_err());
    }
}
