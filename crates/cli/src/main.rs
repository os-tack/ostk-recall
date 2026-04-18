//! `ostk-recall` CLI — Phase B binary.
//!
//! Handlers live in [`ostk_recall_cli::commands`]; this module parses args,
//! constructs a real [`ostk_recall_embed::Embedder`], and delegates.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ostk_recall_cli::commands::{self, InitOutcome};
use ostk_recall_embed::Embedder;
use ostk_recall_pipeline::ChunkEmbedder;
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
        #[arg(long)]
        source: Option<String>,
        /// Report what would be ingested without writing to the store.
        #[arg(long)]
        dry_run: bool,
    },
    /// Verify corpus integrity (counts).
    Verify,
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

#[tokio::main]
async fn main() -> Result<()> {
    // Stdout is reserved for tool output (and JSON-RPC frames in
    // `serve --stdio`); logs go to stderr unconditionally so MCP clients
    // never see interleaved log noise on stdout.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let config_path = match cli.config.clone() {
        Some(p) => p,
        None => commands::default_config_path()?,
    };

    match cli.command {
        Command::Init { force } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            let outcome = commands::init_with_options(&config_path, embedder, force).await?;
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
        Command::Scan { source, dry_run } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            let out = commands::scan(&config_path, embedder, source.as_deref(), dry_run).await?;
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
            println!("TODO: phase C — inspect {chunk_id}");
        }
        Command::Serve { stdio } => {
            let embedder = resolve_embedder(cli.config.as_ref())?;
            commands::serve(&config_path, embedder, stdio).await?;
        }
    }

    Ok(())
}
