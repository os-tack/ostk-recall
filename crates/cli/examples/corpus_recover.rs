//! One-off scanner-adapter: recover lost historical chunks into the live
//! corpus by re-processing them through the CURRENT ingest pipeline.
//!
//! The input (`/tmp/lost_chunks.jsonl` by default) is a set of already-chunked
//! historical chunks recovered from a backup corpus — transcripts, markdown,
//! membrane recognitions — that are absent from the live corpus. This adapter
//! rebuilds a [`Chunk`] per line, re-applies the CURRENT P12 record-rule
//! overlay (so apparatus that today's rules would drop never lands), and feeds
//! the survivors through [`Pipeline::ingest_synthetic`] — which re-embeds with
//! the current model, upserts by `chunk_id`, and emits
//! `IngestOrigin::Synthetic` (bypassing the turn observer, exactly what a
//! backfill wants).
//!
//! Recovered chunks carry a dedicated `source_config_id`
//! (`recovered-pre-v0.6.0`) so their deterministic `chunk_id`s never collide
//! with live native-scan ids.
//!
//! ```bash
//! # Validate parse + record-rules on a subset, no model / no writes:
//! HF_HUB_OFFLINE=1 cargo run --release --example corpus_recover -- \
//!   --dry-run --limit 5000
//!
//! # Real recovery (operator — run with `serve` STOPPED; this mutates the
//! # live corpus):
//! cargo run --release --example corpus_recover -- \
//!   --jsonl /tmp/lost_chunks.jsonl --corpus ~/.local/share/ostk-recall
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use ostk_recall_core::{
    Chunk, CompiledRecordRules, FacetSet, Links, RecordRule, RuleDecision, Source, SourceKind,
    default_record_rules, merge_override,
};
use ostk_recall_embed::Embedder;
use ostk_recall_pipeline::{ChunkEmbedder, Pipeline, SyntheticSourceMeta};
use ostk_recall_store::{CorpusStore, IngestDb};
use serde::Deserialize;

/// Dedicated recovery discriminator. Recovered chunks get a `source_config_id`
/// distinct from any live `[[sources]]`-driven scan, so their deterministic
/// `chunk_id`s never collide with native-scan ids.
const RECOVERY_SOURCE_CONFIG_ID: &str = "recovered-pre-v0.6.0";

/// Raw JSONL record. One per line in the recovered dump.
#[derive(Debug, Deserialize)]
struct RawChunk {
    #[allow(dead_code)]
    sha256: String,
    source: String,
    source_id: String,
    project: Option<String>,
    role: Option<String>,
    ts: Option<String>,
    chunk_index: u32,
    text: String,
    /// JSON-encoded string (e.g. "{\"confidence\":0.75}").
    extra_json: String,
}

/// Map the JSONL `source` string to the concrete [`Source`] row enum.
fn map_source(s: &str) -> Result<Source> {
    Ok(match s {
        "claude_code" => Source::ClaudeCode,
        "gemini" => Source::Gemini,
        "markdown" => Source::Markdown,
        "membrane" => Source::Membrane,
        "ostk_spec" => Source::OstkSpec,
        "ostk_session" => Source::OstkSession,
        other => return Err(anyhow!("unknown source {other:?}")),
    })
}

/// Map a concrete [`Source`] to the [`SourceKind`] the record-rule predicate +
/// `ingest_synthetic` routing use. `ostk_spec` / `ostk_session` are concrete
/// subtypes of the `ostk_project` kind.
fn source_kind_for(source: Source) -> SourceKind {
    match source {
        Source::ClaudeCode => SourceKind::ClaudeCode,
        Source::Gemini => SourceKind::Gemini,
        Source::Markdown => SourceKind::Markdown,
        Source::Membrane => SourceKind::Membrane,
        Source::OstkSpec | Source::OstkSession => SourceKind::OstkProject,
        // Defensive: the JSONL only carries the six sources above, but keep a
        // total mapping so a future source can't silently mis-route.
        Source::Code => SourceKind::Code,
        Source::FileGlob => SourceKind::FileGlob,
        Source::ZipExport => SourceKind::ZipExport,
        Source::Thread => SourceKind::Thread,
        Source::OstkDecision
        | Source::OstkNeedle
        | Source::OstkAuditSignificant
        | Source::OstkConversation
        | Source::OstkMemory => SourceKind::OstkProject,
    }
}

/// Build a fully-formed [`Chunk`] from a raw JSONL record. The pipeline fills
/// `embedding_input_sha256` at ingest, so it is left empty here.
fn build_chunk(raw: &RawChunk) -> Result<Chunk> {
    let source = map_source(&raw.source)?;
    let ts: Option<DateTime<Utc>> = match &raw.ts {
        Some(s) if !s.is_empty() => Some(
            DateTime::parse_from_rfc3339(s)
                .with_context(|| format!("parsing ts {s:?}"))?
                .with_timezone(&Utc),
        ),
        _ => None,
    };
    let chunk_id = Chunk::make_id(
        source,
        &raw.source_id,
        raw.chunk_index,
        RECOVERY_SOURCE_CONFIG_ID,
    );
    let extra = serde_json::from_str(&raw.extra_json).unwrap_or(serde_json::Value::Null);
    Ok(Chunk {
        chunk_id,
        source,
        project: raw.project.clone(),
        source_id: raw.source_id.clone(),
        source_config_id: RECOVERY_SOURCE_CONFIG_ID.to_string(),
        chunk_index: raw.chunk_index,
        ts,
        role: raw.role.clone(),
        sha256: Chunk::content_hash(&raw.text),
        text: raw.text.clone(),
        links: Links::default(),
        facets: FacetSet::default(),
        embedding_input_sha256: String::new(),
        extra,
    })
}

/// Apply the current record-rule overlay to one chunk, mirroring
/// `Pipeline::apply_record_rules` (pipeline/src/lib.rs). Returns `false` to
/// drop, `true` to keep (stamping any `record_kind` tag in place).
fn apply_record_rules(rules: &CompiledRecordRules, chunk: &mut Chunk) -> bool {
    let kind = source_kind_for(chunk.source);
    match rules.decide(&chunk.text, chunk.source, chunk.role.as_deref(), kind) {
        RuleDecision::Drop => false,
        RuleDecision::Tag(record_kind) => {
            merge_override(&mut chunk.facets, "record_kind", vec![record_kind]);
            true
        }
        RuleDecision::Keep => true,
    }
}

/// Resolve the effective record rules the same way the production scan/serve
/// path does (`Config::effective_record_rules()`), falling back to the
/// built-in defaults if no config file is present.
fn resolve_record_rules() -> Result<(Arc<CompiledRecordRules>, &'static str)> {
    let rules: Vec<RecordRule>;
    let provenance: &'static str;
    match ostk_recall_cli::commands::default_config_path() {
        Ok(path) if path.exists() => {
            match ostk_recall_core::Config::load(&path) {
                Ok(cfg) => {
                    rules = cfg.effective_record_rules();
                    provenance = "config.effective_record_rules()";
                }
                Err(e) => {
                    eprintln!(
                        "warning: config at {} failed to load ({e}); using default_record_rules()",
                        path.display()
                    );
                    rules = default_record_rules();
                    provenance = "default_record_rules() (config load failed)";
                }
            }
        }
        _ => {
            rules = default_record_rules();
            provenance = "default_record_rules() (no config file)";
        }
    }
    let compiled = CompiledRecordRules::build(&rules).map_err(|e| anyhow!("record rules: {e}"))?;
    Ok((Arc::new(compiled), provenance))
}

/// Expand a leading `~` to the user's home directory.
fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    if p == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    }
    PathBuf::from(p)
}

#[derive(Default)]
struct Counts {
    read: usize,
    parse_errors: usize,
    dropped: usize,
    kept: usize,
    upserted: usize,
    /// kept-count by source string
    by_source_kept: BTreeMap<String, usize>,
    by_source_dropped: BTreeMap<String, usize>,
    /// a few example dropped snippets
    dropped_examples: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut jsonl = PathBuf::from("/tmp/lost_chunks.jsonl");
    let mut corpus = expand_tilde("~/.local/share/ostk-recall");
    let mut model = "minishlab/potion-retrieval-32M".to_string();
    let mut limit: Option<usize> = None;
    let mut batch: usize = 2000;
    let mut dry_run = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--jsonl" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    jsonl = PathBuf::from(p);
                }
            }
            "--corpus" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    corpus = expand_tilde(p);
                }
            }
            "--model" => {
                i += 1;
                if let Some(m) = args.get(i) {
                    model = m.clone();
                }
            }
            "--limit" => {
                i += 1;
                limit = args.get(i).and_then(|v| v.parse().ok());
            }
            "--batch" => {
                i += 1;
                if let Some(b) = args.get(i).and_then(|v| v.parse().ok()) {
                    batch = b;
                }
            }
            "--dry-run" => dry_run = true,
            other => eprintln!("warning: ignoring unknown arg {other}"),
        }
        i += 1;
    }

    eprintln!(
        "corpus_recover: jsonl={} corpus={} model={model} limit={:?} batch={batch} dry_run={dry_run}",
        jsonl.display(),
        corpus.display(),
        limit,
    );

    let (rules, rules_provenance) = resolve_record_rules()?;
    eprintln!("record rules: {rules_provenance}");

    // --- parse + build + apply rules, grouping survivors by source ----------
    // Grouped by SourceKind because `ingest_synthetic` takes one kind per call.
    // We keep the originating concrete `Source` per chunk on the chunk itself;
    // the group key is the routing SourceKind + project.
    let mut groups: HashMap<(SourceKind, Option<String>), Vec<Chunk>> = HashMap::new();
    let mut counts = Counts::default();

    let file = File::open(&jsonl).with_context(|| format!("opening {}", jsonl.display()))?;
    let reader = BufReader::new(file);

    for line_res in reader.lines() {
        if let Some(n) = limit {
            if counts.read >= n {
                break;
            }
        }
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                eprintln!("warning: read error: {e}");
                counts.parse_errors += 1;
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        counts.read += 1;

        let raw: RawChunk = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                counts.parse_errors += 1;
                if counts.parse_errors <= 5 {
                    eprintln!("warning: parse error on line {}: {e}", counts.read);
                }
                continue;
            }
        };
        let source_str = raw.source.clone();
        let mut chunk = match build_chunk(&raw) {
            Ok(c) => c,
            Err(e) => {
                counts.parse_errors += 1;
                if counts.parse_errors <= 5 {
                    eprintln!("warning: build error on line {}: {e}", counts.read);
                }
                continue;
            }
        };

        if apply_record_rules(&rules, &mut chunk) {
            counts.kept += 1;
            *counts.by_source_kept.entry(source_str).or_insert(0) += 1;
            let key = (source_kind_for(chunk.source), chunk.project.clone());
            groups.entry(key).or_default().push(chunk);
        } else {
            counts.dropped += 1;
            *counts.by_source_dropped.entry(source_str).or_insert(0) += 1;
            if counts.dropped_examples.len() < 5 {
                let snippet: String = chunk.text.chars().take(80).collect();
                counts.dropped_examples.push(snippet.replace('\n', " "));
            }
        }

        if counts.read % 10_000 == 0 {
            eprintln!(
                "  ...processed {} lines (kept {}, dropped {})",
                counts.read, counts.kept, counts.dropped
            );
        }
    }

    eprintln!(
        "parse complete: read={} kept={} dropped={} parse_errors={}",
        counts.read, counts.kept, counts.dropped, counts.parse_errors
    );

    // --- ingest survivors (guarded behind !dry_run) -------------------------
    if !dry_run {
        let embedder: Arc<dyn ChunkEmbedder> =
            Arc::new(Embedder::load(&model).with_context(|| format!("loading model {model}"))?);
        std::fs::create_dir_all(&corpus)
            .with_context(|| format!("creating corpus root {}", corpus.display()))?;
        let store = Arc::new(
            CorpusStore::open_or_create(&corpus, embedder.dim())
                .await
                .with_context(|| format!("opening corpus {}", corpus.display()))?,
        );
        let ingest = Arc::new(
            IngestDb::open(&corpus).map_err(|e| anyhow!("open ingest db: {e}"))?,
        );
        // We apply record-rules ourselves above, so the pipeline overlay is
        // left empty (no `.with_record_rules`).
        let pipeline = Pipeline::new(Arc::clone(&store), ingest, Arc::clone(&embedder));

        for ((kind, project), chunks) in groups {
            let total = chunks.len();
            let mut group_upserted = 0usize;
            for sub in chunks.chunks(batch) {
                let meta = SyntheticSourceMeta {
                    source: kind,
                    project: project.clone(),
                };
                let stats = pipeline
                    .ingest_synthetic(sub.to_vec(), meta)
                    .await
                    .with_context(|| format!("ingest_synthetic kind={}", kind.as_str()))?;
                group_upserted += stats.chunks_upserted;
                counts.upserted += stats.chunks_upserted;
                eprintln!(
                    "  ingested kind={} project={:?}: +{} upserted ({} dup-skipped) [group {}/{}]",
                    kind.as_str(),
                    project,
                    stats.chunks_upserted,
                    stats.chunks_skipped_dup,
                    group_upserted,
                    total,
                );
            }
        }
    }

    // --- final report -------------------------------------------------------
    println!("\n=== corpus_recover summary ===");
    println!("mode:           {}", if dry_run { "DRY-RUN (no embed/no write)" } else { "INGEST" });
    println!("lines read:     {}", counts.read);
    println!("parse errors:   {}", counts.parse_errors);
    println!("dropped (rules):{}", counts.dropped);
    println!("kept:           {}", counts.kept);
    if !dry_run {
        println!("upserted:       {}", counts.upserted);
    }
    println!("\nby-source (kept):");
    for (src, n) in &counts.by_source_kept {
        println!("  {src:<14} {n}");
    }
    if !counts.by_source_dropped.is_empty() {
        println!("by-source (dropped):");
        for (src, n) in &counts.by_source_dropped {
            println!("  {src:<14} {n}");
        }
    }
    if !counts.dropped_examples.is_empty() {
        println!("\nexample dropped snippets:");
        for ex in &counts.dropped_examples {
            println!("  - {ex}");
        }
    }

    Ok(())
}
