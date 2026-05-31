//! `ostk-recall-pipeline` — scan → chunk → dedupe → embed → persist.
//!
//! The [`Pipeline`] owns the three side-effecting resources — a
//! [`CorpusStore`], an [`IngestDb`], and something that can embed text
//! (the [`ChunkEmbedder`] trait) — and wires a [`Scanner`] up to them.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use tokio::sync::broadcast;
use uuid::Uuid;

use ostk_recall_core::{
    Chunk, CompiledRecordRules, IngestOrigin, RetentionPolicy, RuleDecision, Scanner, SourceConfig,
    SourceItem, SourceKind, cfg_overlay_hash, compose_header, filter_to_allowlist, merge_override,
};
use ostk_recall_embed::Embedder;
use ostk_recall_store::{CorpusStore, IngestChunkRow, IngestDb};

/// Batch size used when calling the embedder. Kept small for model
/// throughput / memory; this is the *embed* unit, not the *commit* unit.
pub const EMBED_BATCH: usize = 64;

/// Number of chunks accumulated before a single corpus `merge_insert`
/// (one Lance commit / version). Decoupled from [`EMBED_BATCH`] so a
/// full scan creates one version per `COMMIT_BATCH`, not one per 64 —
/// otherwise version count (and thus fragment/fd/iops pressure) scales
/// ~linearly with corpus size and never prunes (auto-cleanup is off).
/// At ~2048, a 100k-chunk corpus is ~50 commits instead of ~1560.
pub const COMMIT_BATCH: usize = 2048;

/// Capacity of the post-ingest broadcast channel.
///
/// Each `IngestEvent` occupies one slot until every live subscriber has
/// consumed it; once the channel is full, slow subscribers see
/// [`broadcast::error::RecvError::Lagged`] on their next `recv()` and
/// recover from the next live slot.
pub const INGEST_BROADCAST_CAPACITY: usize = 256;

/// Post-ingest event broadcast by [`Pipeline`] after `merge_insert` completes.
///
/// Subscribers (auto-weaver, converger, turn observer feedback loops)
/// receive one event per ingest call — both scanner-driven
/// [`Pipeline::ingest_source`] and in-process
/// [`Pipeline::ingest_synthetic`].
///
/// Defined canonically in `ostk-recall-core`; re-exported here so existing
/// consumers can keep importing `ostk_recall_pipeline::IngestEvent`.
pub use ostk_recall_core::IngestEvent;

/// Caller-provided metadata accompanying a batch of synthetic chunks.
///
/// The pipeline already knows `source_ids` and `chunk_ids` (read from
/// the chunks themselves), so the caller only supplies the routing
/// pair `(source_kind, project)`.
#[derive(Debug, Clone)]
pub struct SyntheticSourceMeta {
    pub source: SourceKind,
    pub project: Option<String>,
}

pub trait ChunkEmbedder: Send + Sync {
    fn dim(&self) -> usize;
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>>;
}

impl ChunkEmbedder for Embedder {
    fn dim(&self) -> usize {
        Self::dim(self)
    }
    fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        Self::encode_batch(self, texts)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStats {
    pub items_seen: usize,
    pub items_skipped: usize,
    pub chunks_emitted: usize,
    pub chunks_upserted: usize,
    pub chunks_skipped_dup: usize,
    pub chunks_purged: usize,
    pub chunks_staled: usize,
    /// Chunks a record rule (P12) dropped this run. The `chunks_purged` count
    /// covers the corpus rows actually deleted (dropped chunks that had been
    /// ingested before); this counts the rule matches.
    pub chunks_dropped_by_rule: usize,
    pub errors: usize,
}

impl PipelineStats {
    #[must_use]
    pub const fn merge(mut self, other: Self) -> Self {
        self.items_seen += other.items_seen;
        self.items_skipped += other.items_skipped;
        self.chunks_emitted += other.chunks_emitted;
        self.chunks_upserted += other.chunks_upserted;
        self.chunks_skipped_dup += other.chunks_skipped_dup;
        self.chunks_purged += other.chunks_purged;
        self.chunks_staled += other.chunks_staled;
        self.chunks_dropped_by_rule += other.chunks_dropped_by_rule;
        self.errors += other.errors;
        self
    }
}

/// Outcome of applying the record-rule overlay to a single chunk (P12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleOutcome {
    /// Keep the chunk (possibly with a `record_kind` tag stamped).
    Keep,
    /// Drop the chunk: skip embedding and purge any previously-ingested copy.
    Drop,
}

pub struct Pipeline {
    store: Arc<CorpusStore>,
    ingest: Arc<IngestDb>,
    embedder: Arc<dyn ChunkEmbedder>,
    dry_run: bool,
    run_id: String,
    events: broadcast::Sender<IngestEvent>,
    /// P12 interpretation overlay. Applied to every chunk at ingest (both the
    /// full-scan and watch/scan_paths paths). Defaults to an empty ruleset
    /// (no-op) so test/programmatic construction is unchanged; production
    /// callers thread `Config::effective_record_rules()` via
    /// [`Pipeline::with_record_rules`].
    record_rules: Arc<CompiledRecordRules>,
}

impl Pipeline {
    pub fn new(
        store: Arc<CorpusStore>,
        ingest: Arc<IngestDb>,
        embedder: Arc<dyn ChunkEmbedder>,
    ) -> Self {
        let (events, _) = broadcast::channel(INGEST_BROADCAST_CAPACITY);
        Self {
            store,
            ingest,
            embedder,
            dry_run: false,
            run_id: Uuid::now_v7().to_string(),
            events,
            record_rules: Arc::new(CompiledRecordRules::empty()),
        }
    }

    #[must_use]
    pub const fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Install the compiled record-rule overlay (P12). Production callers
    /// pass `Config::effective_record_rules()` compiled via
    /// `CompiledRecordRules::build`.
    #[must_use]
    pub fn with_record_rules(mut self, rules: Arc<CompiledRecordRules>) -> Self {
        self.record_rules = rules;
        self
    }

    pub fn store(&self) -> &CorpusStore {
        &self.store
    }

    pub fn ingest_db(&self) -> &IngestDb {
        &self.ingest
    }

    /// P9b-min cold-start warmup. Encode a tiny dummy input through
    /// the embedder so the model weights are resident before the
    /// lens loop's first poll lands. The `cold_start_budget_ms`
    /// default (500ms) in `p9b-lens-portfolio.md` is a target on
    /// the first lens latency; this call is the lever.
    ///
    /// Best-effort: `encode_batch` already absorbs and logs its own
    /// errors, and the lens loop tolerates a slow first refresh —
    /// so we don't propagate a result.
    pub async fn warmup_models(&self) {
        let _ = self.embedder.encode_batch(&["warmup"]);
    }

    /// Subscribe to post-`merge_insert` events. Multiple subscribers are
    /// supported via [`tokio::sync::broadcast`]; lagged receivers see an
    /// explicit `RecvError::Lagged(n)` on their next `recv()` and
    /// recover from the next live slot. The publisher never blocks on
    /// backpressure (slow subscribers lose history, not liveness).
    ///
    /// Events fire from BOTH [`Pipeline::ingest_source`] (after the
    /// orphan sweep completes successfully) and
    /// [`Pipeline::ingest_synthetic`].
    pub fn subscribe_ingest(&self) -> broadcast::Receiver<IngestEvent> {
        self.events.subscribe()
    }

    /// P1: apply the per-chunk facet overlay and compute the embedding
    /// input hash in-place.
    ///
    /// Merges `cfg.facets` (operator override) into `chunk.facets`
    /// (scanner-emitted) per per-key cardinality. Legacy
    /// `[[sources]].project = "x"` is desugared to a `project` facet
    /// here too — the scalar field is in flight for v0.7 deprecation
    /// but the override-merge path is the canonical home.
    ///
    /// Then composes the deterministic header from the allowlisted
    /// subset and stamps `embedding_input_sha256` on the chunk.
    fn apply_p1_overlay(&self, chunk: &mut Chunk, cfg: &SourceConfig) {
        // Legacy `project` desugar: prepend the scalar value before the
        // operator override runs so an operator-supplied `facets.project`
        // unions on top.
        if let Some(legacy_project) = cfg.project.as_deref() {
            if !legacy_project.is_empty() {
                merge_override(
                    &mut chunk.facets,
                    "project",
                    vec![legacy_project.to_string()],
                );
            }
        }
        for (key, values) in &cfg.facets {
            merge_override(&mut chunk.facets, key, values.clone());
        }
        self.stamp_embedding_input_sha256(chunk);
    }

    /// Compute and stamp `embedding_input_sha256` from `chunk.facets` +
    /// `chunk.text`. Used by both the scanner-driven path (after the
    /// override merge) and the synthetic path (no override).
    fn stamp_embedding_input_sha256(&self, chunk: &mut Chunk) {
        let allow = filter_to_allowlist(&chunk.facets);
        let header = compose_header(&allow);
        chunk.embedding_input_sha256 =
            Chunk::embedding_input_hash(self.embedder_model_id(), &header, &chunk.text);
    }

    /// Returns the embedder's model identifier used to scope the
    /// `embedding_input_sha256`. The pipeline doesn't itself carry the
    /// model id (the [`ChunkEmbedder`] trait is dim-only), so for now
    /// we use a placeholder — bumping this is equivalent to bumping
    /// `ALLOWLIST_VERSION`: it forces a corpus-wide re-embed.
    fn embedder_model_id(&self) -> &'static str {
        // TODO(P5): pipe through the actual embedder model id from
        // ChunkEmbedder once we add `fn model_id(&self) -> &str`.
        // Until then, every re-tag/text-change hashes against a fixed
        // version; bumping this constant (or ALLOWLIST_VERSION /
        // HEADER_FORMAT_VERSION) forces a re-embed.
        "default"
    }

    /// Hash the per-source overlay state — legacy `project` + the operator
    /// `facets` override (P1), plus the per-source-kind record-rule digest and
    /// the scanner's `parse_version` (P12) — so Tier-1 (mtime/size) forces a
    /// re-parse when the operator changes facets, edits a record rule that can
    /// match this source, or a scanner bumps its parse version, even when the
    /// file itself is unchanged. Delegates to `core::cfg_overlay_hash`.
    fn cfg_overlay_hash(&self, cfg: &SourceConfig, scanner: &dyn Scanner) -> String {
        let rules_digest = self.record_rules.digest_for(cfg.kind);
        let parse_version = scanner.parse_version();
        // Preserve the pre-P12 hash exactly for sources with NO applicable
        // record rules AND parse_version 0 — otherwise the P12 upgrade would
        // flip every source's Tier-1 key and re-parse the whole corpus
        // (including the expensive code re-index), not just the affected
        // source kinds.
        let extra = if rules_digest.is_empty() && parse_version == 0 {
            String::new()
        } else {
            format!("rr:{rules_digest}|pv:{parse_version}")
        };
        cfg_overlay_hash(cfg.project.as_deref(), &cfg.facets, &extra)
    }

    /// P12: apply the config-driven record-rule overlay to one chunk. Returns
    /// [`RuleOutcome::Drop`] when a `drop` rule matched (the caller skips and
    /// purges the chunk); otherwise stamps any `record_kind` tag and returns
    /// [`RuleOutcome::Keep`]. Runs at the top of BOTH ingest loops (full scan
    /// and watch/scan_paths) — before `apply_p1_overlay` — so a tag is visible
    /// to the operator-override merge and the embedding-input hash.
    fn apply_record_rules(&self, chunk: &mut Chunk, cfg: &SourceConfig) -> RuleOutcome {
        match self.record_rules.decide(
            &chunk.text,
            chunk.source,
            chunk.role.as_deref(),
            cfg.kind,
        ) {
            RuleDecision::Drop => RuleOutcome::Drop,
            RuleDecision::Tag(record_kind) => {
                merge_override(&mut chunk.facets, "record_kind", vec![record_kind]);
                RuleOutcome::Keep
            }
            RuleDecision::Keep => RuleOutcome::Keep,
        }
    }

    /// Purge chunks dropped by a record rule from both the corpus and the
    /// ingest ledger. Needed because `RetentionPolicy::Keep` sources (e.g.
    /// `claude_code`, `ostk_project`) are not cleaned by the orphan sweep, so a
    /// newly-dropped, previously-ingested chunk would otherwise linger. No-op
    /// for ids that were never ingested. Runs after each ingest loop.
    async fn purge_dropped(&self, dropped: &[String], stats: &mut PipelineStats) {
        if dropped.is_empty() || self.dry_run {
            return;
        }
        match self.store.delete_chunks(dropped).await {
            Ok(n) => stats.chunks_purged += n as usize,
            Err(e) => {
                tracing::warn!(error = %e, "record-rule drop purge: corpus delete failed");
                stats.errors += 1;
            }
        }
        if let Err(e) = self.ingest.delete_by_chunk_ids(dropped) {
            tracing::warn!(error = %e, "record-rule drop purge: ingest delete failed");
            stats.errors += 1;
        }
    }

    // Casts: `d.as_micros() as i64` and `meta.len() as i64` are intentional —
    // micros within i64 range until year 294247; file lengths under 8 EiB.
    // Length and nested-if shape track the parallel `ingest_paths_for_source`.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::collapsible_if
    )]
    pub async fn ingest_source(&self, scanner: &dyn Scanner, cfg: &SourceConfig) -> PipelineStats {
        let mut stats = PipelineStats::default();
        let mut to_embed: Vec<Chunk> = Vec::new();
        let mut dropped: Vec<String> = Vec::new();
        let mut event_source_ids: HashSet<String> = HashSet::new();
        let source_kind_str = scanner.kind().as_str();

        for item_res in scanner.discover(cfg) {
            let item = match item_res {
                Ok(it) => it,
                Err(e) => {
                    tracing::warn!(error = %e, "discover failed");
                    stats.errors += 1;
                    continue;
                }
            };
            stats.items_seen += 1;

            // Tier 1: File-level metadata check, gated by cfg-overlay
            // hash so operator facet edits force re-parse even when the
            // file mtime/size are unchanged (P1), plus record-rule digest +
            // scanner parse_version (P12).
            let overlay_hash = self.cfg_overlay_hash(cfg, scanner);
            if let Some(path) = &item.path {
                if let Ok(meta) = std::fs::metadata(path) {
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map_or(0, |d| d.as_micros() as i64);
                    let size = meta.len() as i64;

                    if let Ok(Some((old_mtime, old_size, old_overlay))) =
                        self.ingest.get_source_metadata(
                            source_kind_str,
                            &cfg.source_config_id,
                            &item.source_id,
                        )
                    {
                        if old_mtime == mtime && old_size == size && old_overlay == overlay_hash {
                            if self
                                .ingest
                                .touch_source_chunks(
                                    source_kind_str,
                                    &cfg.source_config_id,
                                    &item.source_id,
                                    &self.run_id,
                                )
                                .is_ok()
                            {
                                stats.items_skipped += 1;
                                continue;
                            }
                        }
                    }

                    let _ = self.ingest.update_source_metadata(
                        source_kind_str,
                        &cfg.source_config_id,
                        &item.source_id,
                        mtime,
                        size,
                        &overlay_hash,
                        &self.run_id,
                    );
                }
            }

            let chunks = match scanner.parse(SourceItem {
                source_config_id: cfg.source_config_id.clone(),
                ..item
            }) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "parse failed");
                    stats.errors += 1;
                    continue;
                }
            };
            stats.chunks_emitted += chunks.len();

            for mut chunk in chunks {
                // P12: record-rule overlay BEFORE P1. A drop skips the chunk
                // (and purges any previously-ingested copy); a tag stamps
                // record_kind so it flows into the embedding-input hash.
                if self.apply_record_rules(&mut chunk, cfg) == RuleOutcome::Drop {
                    stats.chunks_dropped_by_rule += 1;
                    dropped.push(chunk.chunk_id);
                    continue;
                }

                // P1: stamp facets + embedding_input_sha256 BEFORE dedupe
                // so the Tier-2 check observes facet edits.
                self.apply_p1_overlay(&mut chunk, cfg);

                // Tier 2: skip when both chunk_id AND embedding input
                // are unchanged (text + allowlisted facets identical).
                match self
                    .ingest
                    .content_already_embedded(&chunk.chunk_id, &chunk.embedding_input_sha256)
                {
                    Ok(true) => {
                        stats.chunks_skipped_dup += 1;
                        let row = IngestChunkRow {
                            chunk_id: chunk.chunk_id.clone(),
                            source: chunk.source.as_str().to_string(),
                            source_config_id: chunk.source_config_id.clone(),
                            source_id: chunk.source_id.clone(),
                            chunk_index: chunk.chunk_index,
                            content_sha256: chunk.sha256.clone(),
                            embedding_input_sha256: chunk.embedding_input_sha256.clone(),
                        };
                        let _ = self.ingest.record_chunk(&row, Some(&self.run_id));
                    }
                    Ok(false) => to_embed.push(chunk),
                    Err(e) => {
                        tracing::warn!(error = %e, "ingest dedupe check failed");
                        stats.errors += 1;
                    }
                }
            }
        }

        // P12: purge chunks dropped by a record rule from corpus + ingest
        // ledger (orphan sweep won't, for Keep-retention sources).
        self.purge_dropped(&dropped, &mut stats).await;

        let mut upserted_chunk_ids: Vec<String> = Vec::new();
        if !to_embed.is_empty() && !self.dry_run {
            // Embed in EMBED_BATCH units (model throughput) but commit in
            // COMMIT_BATCH units (one Lance version per flush). Streaming
            // the embeds keeps peak memory bounded; the buffer flushes
            // when it reaches COMMIT_BATCH and once more at the end.
            let mut pending: Vec<Chunk> = Vec::new();
            let mut pending_embeds: Vec<Vec<f32>> = Vec::new();
            for batch in to_embed.chunks(EMBED_BATCH) {
                let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
                let embeddings = self.embedder.encode_batch(&texts);
                if embeddings.len() != batch.len() {
                    tracing::warn!(error = "dim mismatch", "embed batch failed");
                    stats.errors += 1;
                    continue;
                }
                pending.extend_from_slice(batch);
                pending_embeds.extend(embeddings);
                if pending.len() >= COMMIT_BATCH {
                    self.flush_pending(
                        &mut pending,
                        &mut pending_embeds,
                        &mut stats,
                        &mut upserted_chunk_ids,
                        &mut event_source_ids,
                    )
                    .await;
                }
            }
            self.flush_pending(
                &mut pending,
                &mut pending_embeds,
                &mut stats,
                &mut upserted_chunk_ids,
                &mut event_source_ids,
            )
            .await;
        }

        // Orphan Sweep — multi-project safety contract
        //
        // The orphan sweep filters by (source, project, run_id). If `cfg.project`
        // is None we'd fall back to a placeholder ("default"), and any chunks
        // ingested under that placeholder by a *different* unspecified-project
        // scan would get swept under this run_id. That's the cross-project
        // wipe footgun. Refuse to run the sweep when project is unset; let the
        // caller fix their config.
        if !self.dry_run && cfg.source_config_id.is_empty() {
            tracing::error!(
                source_kind = %source_kind_str,
                "orphan sweep skipped: SourceConfig.source_config_id is empty — Config::validate_and_seal must run before any scan."
            );
            stats.errors += 1;
            return stats;
        }

        // Orphan Sweep
        if !self.dry_run {
            match cfg.kind.retention_policy() {
                RetentionPolicy::Delete => {
                    for s in cfg.kind.sources() {
                        match self.ingest.delete_orphans(
                            s.as_str(),
                            &cfg.source_config_id,
                            &self.run_id,
                        ) {
                            Ok(orphans) => {
                                if !orphans.is_empty() {
                                    if let Err(e) = self.store.delete_chunks(&orphans).await {
                                        tracing::warn!(error = %e, source = %s.as_str(), "failed to delete orphan chunks from corpus store");
                                    } else {
                                        stats.chunks_purged += orphans.len();
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, source = %s.as_str(), "orphan sweep failed");
                            }
                        }
                    }
                }
                RetentionPolicy::Stale => {
                    for s in cfg.kind.sources() {
                        match self.ingest.mark_orphans_stale(
                            s.as_str(),
                            &cfg.source_config_id,
                            &self.run_id,
                        ) {
                            Ok(orphans) => {
                                if !orphans.is_empty() {
                                    if let Err(e) = self.store.mark_chunks_stale(&orphans).await {
                                        tracing::warn!(error = %e, source = %s.as_str(), "failed to mark orphan chunks as stale in corpus store");
                                    }
                                }
                                stats.chunks_staled += orphans.len();
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, source = %s.as_str(), "stale marking failed");
                            }
                        }
                    }
                }
                RetentionPolicy::Keep => {}
            }
        }

        if !self.dry_run && self.events.receiver_count() > 0 {
            let _ = self.events.send(IngestEvent {
                project: cfg.project.clone(),
                source: cfg.kind,
                source_ids: event_source_ids.into_iter().collect(),
                chunk_ids_upserted: upserted_chunk_ids,
                chunks_upserted: stats.chunks_upserted,
                chunks_stale: stats.chunks_staled,
                ts: Utc::now(),
                // Full-corpus scan = bulk library load, never a TurnEnd.
                origin: IngestOrigin::Bulk,
            });
        }

        stats
    }

    /// Ingest a batch of pre-formed chunks bypassing the scanner +
    /// discover phase. Used by in-process producers (turn observer,
    /// surfacer feedback loops) that have already constructed chunks
    /// and just need them embedded + merged into the corpus.
    ///
    /// Reuses the existing embed → `merge_insert` path
    /// ([`Pipeline::embed_and_persist`]) and runs the same Tier 2
    /// content-hash dedupe check so re-ingesting an identical chunk is
    /// idempotent. Emits one [`IngestEvent`] on the broadcast channel
    /// after `merge_insert` completes, exactly like a scanner-driven
    /// [`Pipeline::ingest_source`] call. Honors `dry_run` (skips
    /// embed + emit if set).
    pub async fn ingest_synthetic(
        &self,
        chunks: Vec<Chunk>,
        source_meta: SyntheticSourceMeta,
    ) -> Result<PipelineStats, PipelineError> {
        let mut stats = PipelineStats {
            items_seen: 1,
            chunks_emitted: chunks.len(),
            ..PipelineStats::default()
        };

        let project = source_meta.project.as_deref().unwrap_or("default");
        let mut to_embed: Vec<Chunk> = Vec::with_capacity(chunks.len());
        let mut event_source_ids: HashSet<String> = HashSet::new();

        for mut chunk in chunks {
            event_source_ids.insert(chunk.source_id.clone());
            // P1: synthetic chunks bypass `[[sources]]` override but
            // still need an `embedding_input_sha256` for the new dedupe
            // path. Whatever facets the synthetic producer set survive.
            self.stamp_embedding_input_sha256(&mut chunk);
            match self
                .ingest
                .content_already_embedded(&chunk.chunk_id, &chunk.embedding_input_sha256)
            {
                Ok(true) => {
                    stats.chunks_skipped_dup += 1;
                    let row = IngestChunkRow {
                        chunk_id: chunk.chunk_id.clone(),
                        source: chunk.source.as_str().to_string(),
                        source_config_id: chunk.source_config_id.clone(),
                        source_id: chunk.source_id.clone(),
                        chunk_index: chunk.chunk_index,
                        content_sha256: chunk.sha256.clone(),
                        embedding_input_sha256: chunk.embedding_input_sha256.clone(),
                    };
                    let _ = self.ingest.record_chunk(&row, Some(&self.run_id));
                }
                Ok(false) => to_embed.push(chunk),
                Err(e) => {
                    tracing::warn!(error = %e, "ingest dedupe check failed");
                    stats.errors += 1;
                }
            }
        }

        let mut upserted_chunk_ids: Vec<String> = Vec::new();
        if !to_embed.is_empty() && !self.dry_run {
            for batch in to_embed.chunks(EMBED_BATCH) {
                let n = self.embed_and_persist(batch, project).await?;
                stats.chunks_upserted += n;
                upserted_chunk_ids.extend(batch.iter().map(|c| c.chunk_id.clone()));
            }
        }

        if !self.dry_run && self.events.receiver_count() > 0 {
            let _ = self.events.send(IngestEvent {
                project: source_meta.project.clone(),
                source: source_meta.source,
                source_ids: event_source_ids.into_iter().collect(),
                chunk_ids_upserted: upserted_chunk_ids,
                chunks_upserted: stats.chunks_upserted,
                chunks_stale: 0,
                ts: Utc::now(),
                // Substrate self-write (membrane/attention). Never
                // re-observed — closes the observer feedback loop.
                origin: IngestOrigin::Synthetic,
            });
        }

        Ok(stats)
    }

    /// Path-aware ingest: group `paths` by source root, then per-source
    /// dispatch through `scanner.discover_paths` → parse → chunk → embed →
    /// upsert (same flow as [`Pipeline::ingest_source`] minus the orphan
    /// sweep — delete handling is gh#7).
    ///
    /// `sources` carries the same `(&dyn Scanner, &SourceConfig)` pairs the
    /// caller would pass to `ingest_source` for a full scan; the pipeline
    /// fans out only to the sources whose expanded `paths` parent at least
    /// one input. A single input path can match multiple sources (e.g.
    /// `code` + `markdown` both rooted at `~/projects/foo`); both fire.
    /// Per-source `extensions` filter still applies.
    ///
    /// Returns one `(label, PipelineStats)` per source that received work
    /// (skipped sources are absent from the returned vector). Label format
    /// matches CLI [`commands::scan`]: `project` if set, else
    /// `<kind>[<index>]`.
    pub async fn scan_paths(
        &self,
        sources: &[(&dyn Scanner, &SourceConfig)],
        paths: &[PathBuf],
    ) -> Result<Vec<(String, PipelineStats)>, PipelineError> {
        let mut out = Vec::new();
        for (i, (scanner, cfg)) in sources.iter().enumerate() {
            let label = cfg
                .project
                .clone()
                .unwrap_or_else(|| format!("{}[{i}]", cfg.kind.as_str()));

            let matched = paths_under_source(cfg, paths);
            if matched.is_empty() {
                continue;
            }
            let (mut stats, yielded) = self.ingest_paths_for_source(*scanner, cfg, &matched).await;
            stats = self
                .purge_missing_paths(cfg, &matched, &yielded, stats)
                .await;
            out.push((label, stats));
        }
        Ok(out)
    }

    /// Per-source path-filtered ingest. Mirrors [`Pipeline::ingest_source`]
    /// but drives discovery via [`Scanner::discover_paths`] and skips the
    /// orphan sweep — missing-on-disk paths are handled separately by
    /// [`Pipeline::purge_missing_paths`] (gh#7).
    ///
    /// Returns the per-source [`PipelineStats`] and the set of input
    /// paths that `discover_paths` actually yielded a [`SourceItem`] for.
    /// The caller diffs that set against `paths` to find missing-on-disk
    /// candidates for the delete pass.
    // Mirrors `ingest_source`; same cast + nested-if intentional patterns.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::collapsible_if
    )]
    async fn ingest_paths_for_source(
        &self,
        scanner: &dyn Scanner,
        cfg: &SourceConfig,
        paths: &[PathBuf],
    ) -> (PipelineStats, HashSet<PathBuf>) {
        let mut stats = PipelineStats::default();
        let mut to_embed: Vec<Chunk> = Vec::new();
        let mut dropped: Vec<String> = Vec::new();
        let mut yielded: HashSet<PathBuf> = HashSet::new();
        let source_kind_str = scanner.kind().as_str();
        let project = cfg.project.as_deref().unwrap_or("default");

        for item_res in scanner.discover_paths(cfg, paths) {
            let item = match item_res {
                Ok(it) => it,
                Err(e) => {
                    tracing::warn!(error = %e, "discover_paths failed");
                    stats.errors += 1;
                    continue;
                }
            };
            if let Some(p) = &item.path {
                yielded.insert(p.clone());
            }
            stats.items_seen += 1;

            // Tier 1: file-level metadata short-circuit (same as ingest_source).
            let overlay_hash = self.cfg_overlay_hash(cfg, scanner);
            if let Some(path) = &item.path {
                if let Ok(meta) = std::fs::metadata(path) {
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map_or(0, |d| d.as_micros() as i64);
                    let size = meta.len() as i64;

                    if let Ok(Some((old_mtime, old_size, old_overlay))) =
                        self.ingest.get_source_metadata(
                            source_kind_str,
                            &cfg.source_config_id,
                            &item.source_id,
                        )
                    {
                        if old_mtime == mtime && old_size == size && old_overlay == overlay_hash {
                            if self
                                .ingest
                                .touch_source_chunks(
                                    source_kind_str,
                                    &cfg.source_config_id,
                                    &item.source_id,
                                    &self.run_id,
                                )
                                .is_ok()
                            {
                                stats.items_skipped += 1;
                                continue;
                            }
                        }
                    }

                    let _ = self.ingest.update_source_metadata(
                        source_kind_str,
                        &cfg.source_config_id,
                        &item.source_id,
                        mtime,
                        size,
                        &overlay_hash,
                        &self.run_id,
                    );
                }
            }

            let chunks = match scanner.parse(SourceItem {
                source_config_id: cfg.source_config_id.clone(),
                ..item
            }) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "parse failed");
                    stats.errors += 1;
                    continue;
                }
            };
            stats.chunks_emitted += chunks.len();

            for mut chunk in chunks {
                // P12: same shared overlay as the full-scan path — a watched
                // transcript append cannot bypass record rules.
                if self.apply_record_rules(&mut chunk, cfg) == RuleOutcome::Drop {
                    stats.chunks_dropped_by_rule += 1;
                    dropped.push(chunk.chunk_id);
                    continue;
                }
                self.apply_p1_overlay(&mut chunk, cfg);
                match self
                    .ingest
                    .content_already_embedded(&chunk.chunk_id, &chunk.embedding_input_sha256)
                {
                    Ok(true) => {
                        stats.chunks_skipped_dup += 1;
                        let row = IngestChunkRow {
                            chunk_id: chunk.chunk_id.clone(),
                            source: chunk.source.as_str().to_string(),
                            source_config_id: chunk.source_config_id.clone(),
                            source_id: chunk.source_id.clone(),
                            chunk_index: chunk.chunk_index,
                            content_sha256: chunk.sha256.clone(),
                            embedding_input_sha256: chunk.embedding_input_sha256.clone(),
                        };
                        let _ = self.ingest.record_chunk(&row, Some(&self.run_id));
                    }
                    Ok(false) => to_embed.push(chunk),
                    Err(e) => {
                        tracing::warn!(error = %e, "ingest dedupe check failed");
                        stats.errors += 1;
                    }
                }
            }
        }

        // P12: purge record-rule-dropped chunks (Keep-retention safe).
        self.purge_dropped(&dropped, &mut stats).await;

        let mut upserted_chunk_ids: Vec<String> = Vec::new();
        let mut event_source_ids: HashSet<String> = HashSet::new();
        if !to_embed.is_empty() && !self.dry_run {
            for batch in to_embed.chunks(EMBED_BATCH) {
                match self.embed_and_persist(batch, project).await {
                    Ok(n) => {
                        stats.chunks_upserted += n;
                        upserted_chunk_ids.extend(batch.iter().map(|c| c.chunk_id.clone()));
                        for c in batch {
                            event_source_ids.insert(c.source_id.clone());
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "embed/persist batch failed");
                        stats.errors += 1;
                    }
                }
            }
        }

        // Broadcast a Watch-origin event. This is the incremental/live
        // path: a watched ingest of a conversation transcript is a
        // TurnEnd, which is the ONLY thing that drives the ambient
        // cognition daemons. (The full-scan path emits `Bulk`, which the
        // daemons skip.) Non-transcript watched sources still emit, but
        // `IngestEvent::is_turn_end()` filters them out downstream.
        if !self.dry_run && !upserted_chunk_ids.is_empty() && self.events.receiver_count() > 0 {
            let _ = self.events.send(IngestEvent {
                project: cfg.project.clone(),
                source: cfg.kind,
                source_ids: event_source_ids.into_iter().collect(),
                chunk_ids_upserted: upserted_chunk_ids,
                chunks_upserted: stats.chunks_upserted,
                chunks_stale: stats.chunks_staled,
                ts: Utc::now(),
                origin: IngestOrigin::Watch,
            });
        }

        (stats, yielded)
    }

    /// Delete-event branch of [`Pipeline::scan_paths`] (gh#7).
    ///
    /// For each input path under this source's roots that
    /// `discover_paths` did NOT yield (file deleted, gitignored, or
    /// extension-filtered out), purge the chunk family from the ledger
    /// and apply the source's [`RetentionPolicy`] against the corpus
    /// (delete or mark-stale). Paths that were never ingested fall
    /// through as no-ops.
    ///
    /// Atomicity ordering mirrors [`Pipeline::ingest_source`]'s orphan
    /// sweep: ledger tombstone first, then corpus mutation. A crash
    /// between the two leaves the same shape of drift the existing
    /// `verify` already detects (corpus row stranded with no active
    /// ledger entry — recoverable via `--reingest`).
    ///
    /// `RetentionPolicy::Keep` sources skip the corpus side per the
    /// orphan-sweep policy table; we still tombstone their ledger so a
    /// later full scan won't see stale rows.
    async fn purge_missing_paths(
        &self,
        cfg: &SourceConfig,
        matched: &[PathBuf],
        yielded: &HashSet<PathBuf>,
        mut stats: PipelineStats,
    ) -> PipelineStats {
        if self.dry_run {
            return stats;
        }
        let Ok(roots) = cfg.expanded_paths() else {
            return stats;
        };
        // Canonicalize roots once. relative_source_id below also
        // canonicalizes the input path so the strip_prefix succeeds
        // even on macOS where /var/folders symlinks to /private/var.
        let canon_roots: Vec<PathBuf> = roots
            .iter()
            .map(|r| std::fs::canonicalize(r).unwrap_or_else(|_| r.clone()))
            .collect();
        let retention = cfg.kind.retention_policy();

        for path in matched {
            if yielded.contains(path) {
                continue;
            }
            let canon_path = canonicalize_lossy(path);
            let source_id = relative_source_id(&canon_roots, &canon_path);
            // Each (source, project, source_id) ledger row keys a chunk
            // family. SourceKind expands to multiple Source variants
            // (notably ostk_project), so iterate the kind's sources to
            // catch all of them; mismatched variants tombstone nothing.
            for src in cfg.kind.sources() {
                let chunk_ids = match self.ingest.tombstone_chunks_by_path(
                    src.as_str(),
                    &cfg.source_config_id,
                    &source_id,
                ) {
                    Ok(ids) => ids,
                    Err(e) => {
                        tracing::warn!(error = %e, source = %src.as_str(), path = %path.display(), "tombstone failed");
                        stats.errors += 1;
                        continue;
                    }
                };
                if chunk_ids.is_empty() {
                    continue;
                }
                match retention {
                    RetentionPolicy::Delete => match self.store.delete_chunks(&chunk_ids).await {
                        Ok(_) => stats.chunks_purged += chunk_ids.len(),
                        Err(e) => {
                            tracing::warn!(error = %e, source = %src.as_str(), "corpus delete failed for missing path");
                            stats.errors += 1;
                        }
                    },
                    RetentionPolicy::Stale => {
                        if let Err(e) = self.store.mark_chunks_stale(&chunk_ids).await {
                            tracing::warn!(error = %e, source = %src.as_str(), "corpus stale-mark failed for missing path");
                            stats.errors += 1;
                        }
                        stats.chunks_staled += chunk_ids.len();
                    }
                    RetentionPolicy::Keep => {
                        // Ledger tombstoned above; corpus rows preserved.
                    }
                }
            }
        }
        stats
    }

    async fn embed_and_persist(
        &self,
        batch: &[Chunk],
        project: &str,
    ) -> Result<usize, PipelineError> {
        let _ = project;
        let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
        let embeddings = self.embedder.encode_batch(&texts);
        if embeddings.len() != batch.len() {
            return Err(PipelineError::Embedder("dim mismatch".into()));
        }
        self.commit_batch(batch, &embeddings).await?;
        Ok(batch.len())
    }

    /// Persist one already-embedded batch: a single corpus
    /// `merge_insert` (one Lance commit / version) plus the per-chunk
    /// ingest-ledger rows. Callers that want fewer versions accumulate
    /// across several [`EMBED_BATCH`] embeds and call this once per
    /// [`COMMIT_BATCH`].
    async fn commit_batch(
        &self,
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> Result<(), PipelineError> {
        if chunks.is_empty() {
            return Ok(());
        }
        self.store
            .upsert(chunks, embeddings)
            .await
            .map_err(|e| PipelineError::Store(e.to_string()))?;

        for chunk in chunks {
            let row = IngestChunkRow {
                chunk_id: chunk.chunk_id.clone(),
                source: chunk.source.as_str().to_string(),
                source_config_id: chunk.source_config_id.clone(),
                source_id: chunk.source_id.clone(),
                chunk_index: chunk.chunk_index,
                content_sha256: chunk.sha256.clone(),
                embedding_input_sha256: chunk.embedding_input_sha256.clone(),
            };
            self.ingest
                .record_chunk(&row, Some(&self.run_id))
                .map_err(|e| PipelineError::Store(e.to_string()))?;
        }
        Ok(())
    }

    /// Commit the accumulated buffer as one Lance version and fold the
    /// outcome into the scan's running totals, then clear the buffers. A
    /// commit error is logged + counted (not propagated), preserving the
    /// prior per-batch resilience — one bad flush doesn't abort the scan.
    async fn flush_pending(
        &self,
        pending: &mut Vec<Chunk>,
        pending_embeds: &mut Vec<Vec<f32>>,
        stats: &mut PipelineStats,
        upserted_chunk_ids: &mut Vec<String>,
        event_source_ids: &mut HashSet<String>,
    ) {
        if pending.is_empty() {
            return;
        }
        match self.commit_batch(pending, pending_embeds).await {
            Ok(()) => {
                stats.chunks_upserted += pending.len();
                upserted_chunk_ids.extend(pending.iter().map(|c| c.chunk_id.clone()));
                for c in pending.iter() {
                    event_source_ids.insert(c.source_id.clone());
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, count = pending.len(), "commit batch failed");
                stats.errors += 1;
            }
        }
        pending.clear();
        pending_embeds.clear();
    }

    pub async fn verify_counts(&self) -> Result<VerifyReport, PipelineError> {
        let corpus_total = self
            .store
            .row_count()
            .await
            .map_err(|e| PipelineError::Store(e.to_string()))?;
        let ingest_by_source = self
            .ingest
            .count_active_by_source()
            .map_err(|e| PipelineError::Store(e.to_string()))?;
        let ingest_total: u64 = ingest_by_source.iter().map(|(_, n)| *n).sum();
        // u64 -> usize: chunk counts won't exceed usize::MAX on 64-bit targets.
        #[allow(clippy::cast_possible_truncation)]
        let ingest_total_usize = ingest_total as usize;
        Ok(VerifyReport {
            corpus_total,
            ingest_total: ingest_total_usize,
            by_source: ingest_by_source,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub corpus_total: usize,
    pub ingest_total: usize,
    pub by_source: Vec<(String, u64)>,
}

impl VerifyReport {
    pub const fn is_consistent(&self) -> bool {
        self.corpus_total == self.ingest_total
    }
}

/// Filter `paths` down to those that fall under one of `cfg`'s expanded
/// roots AND match its per-source `extensions` (if any). Used by
/// [`Pipeline::scan_paths`] to compute the per-source path subset before
/// dispatching to `discover_paths`.
fn paths_under_source(cfg: &SourceConfig, paths: &[PathBuf]) -> Vec<PathBuf> {
    let Ok(roots) = cfg.expanded_paths() else {
        return Vec::new();
    };
    paths
        .iter()
        .filter(|p| roots.iter().any(|root| path_under_root(p, root)))
        .filter(|p| matches_extensions(p, &cfg.extensions))
        .cloned()
        .collect()
}

fn path_under_root(path: &Path, root: &Path) -> bool {
    let path = canonicalize_lossy(path);
    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    path.starts_with(&root)
}

/// Canonicalize a path even if it doesn't exist on disk (delete-event
/// case in [`Pipeline::purge_missing_paths`]). Falls back to
/// `parent.canonicalize() + file_name` so the leading symlinks resolve
/// the same way they do for paths whose target still exists. Without
/// this, on macOS a deleted `/var/folders/...` path stays uncanonical
/// while its still-existing root canonicalizes to `/private/var/...`,
/// and `starts_with` falsely rejects it.
fn canonicalize_lossy(path: &Path) -> PathBuf {
    if let Ok(p) = std::fs::canonicalize(path) {
        return p;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        if let Ok(p) = std::fs::canonicalize(parent) {
            return p.join(name);
        }
    }
    path.to_path_buf()
}

/// Compute a `source_id` from an absolute path the same way the
/// path-rooted scanners (markdown, code, `file_glob`) do — relative to the
/// first matching configured root, joined with `/`. Used by
/// [`Pipeline::purge_missing_paths`] (gh#7) to look up a chunk family in
/// the ledger by `(source, project, source_id)` for delete-event
/// handling. Sources whose `source_id` scheme is NOT path-relative
/// (`ostk_project` route markers, `zip_export` archive keys) won't match the
/// ledger and the tombstone collapses to a no-op — desired for the
/// first-cut scope (append-only sources are deferred per EPIC gh#8).
fn relative_source_id(roots: &[PathBuf], path: &Path) -> String {
    for root in roots {
        if let Ok(rel) = path.strip_prefix(root) {
            return rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
        }
    }
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn matches_extensions(path: &Path, extensions: &[String]) -> bool {
    if extensions.is_empty() {
        return true;
    }
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| extensions.iter().any(|x| x == ext))
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("embedder: {0}")]
    Embedder(String),
    #[error("store: {0}")]
    Store(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ostk_recall_core::SourceKind;
    use ostk_recall_scan::markdown::MarkdownScanner;
    use std::path::Path;
    use tempfile::TempDir;

    struct FakeEmbedder {
        dim: usize,
    }

    impl ChunkEmbedder for FakeEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }
        #[allow(clippy::cast_precision_loss)]
        fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
            texts
                .iter()
                .map(|t| {
                    let seed = ((t.len() % 100) as f32) * 0.01;
                    (0..self.dim)
                        .map(|i| (i as f32).mul_add(0.001, seed))
                        .collect()
                })
                .collect()
        }
    }

    fn write_sample_tree(root: &Path) {
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(
            root.join("a.md"),
            "# Alpha\n\nIntro paragraph.\n\n## One\n\ncontent one\n\n## Two\n\ncontent two\n",
        )
        .unwrap();
        std::fs::write(root.join("sub/b.md"), "# Beta\n\nJust one blob.\n").unwrap();
    }

    fn cfg_for(root: &Path) -> SourceConfig {
        SourceConfig {
            kind: SourceKind::Markdown,
            project: Some("test".into()),
            paths: vec![root.to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        }
    }

    async fn make_pipeline(corpus_root: &Path, dim: usize) -> Pipeline {
        let store = Arc::new(CorpusStore::open_or_create(corpus_root, dim).await.unwrap());
        let ingest = Arc::new(IngestDb::open(corpus_root).unwrap());
        let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });
        Pipeline::new(store, ingest, emb)
    }

    /// Regression for the version-explosion fix: a single source whose
    /// chunk count spans several EMBED_BATCHes (but stays under
    /// COMMIT_BATCH) must produce ONE Lance commit, not one per 64.
    #[tokio::test]
    async fn ingest_source_commits_in_batches_not_per_embed_batch() {
        use ostk_recall_scan::code::CodeScanner;
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        // ~30k lines → CodeScanner (≈200-line windows, 20 overlap) yields
        // well over 2 * EMBED_BATCH chunks, far under COMMIT_BATCH.
        let mut big = String::with_capacity(400_000);
        for i in 0..30_000 {
            big.push_str(&format!("let v{i} = {i};\n"));
        }
        std::fs::write(fixtures.path().join("big.rs"), &big).unwrap();

        let pipeline = make_pipeline(corpus.path(), 16).await;
        // Snapshot version after corpus create+index, before ingest.
        let store = CorpusStore::open_or_create(corpus.path(), 16)
            .await
            .unwrap();
        let v_before = store.version().await.unwrap();

        let scanner = CodeScanner;
        let cfg = SourceConfig {
            kind: SourceKind::Code,
            project: Some("commit-batch-test".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec!["rs".into()],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };
        let stats = pipeline.ingest_source(&scanner, &cfg).await;

        assert!(
            stats.chunks_upserted > 2 * EMBED_BATCH,
            "test needs >2 embed batches to be meaningful; got {} chunks",
            stats.chunks_upserted
        );
        let v_after = store.version().await.unwrap();
        let delta = v_after - v_before;
        // Batched commit: one COMMIT_BATCH flush = 1 version (+slack for any
        // index/sweep commit). Per-EMBED_BATCH would be ceil(chunks/64) ≥ 3.
        assert!(
            delta <= 2,
            "expected ~1 commit for {} chunks (COMMIT_BATCH={COMMIT_BATCH}), got {delta} versions",
            stats.chunks_upserted
        );
    }

    #[tokio::test]
    async fn end_to_end_ingest_emits_chunks() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = MarkdownScanner;
        let stats = pipeline
            .ingest_source(&scanner, &cfg_for(fixtures.path()))
            .await;

        assert_eq!(stats.items_seen, 2, "two md files");
        assert!(stats.chunks_emitted >= 3);
        assert_eq!(stats.chunks_upserted, stats.chunks_emitted);
        assert_eq!(stats.chunks_skipped_dup, 0);
        assert_eq!(stats.errors, 0);
    }

    #[tokio::test]
    async fn incremental_scan_skips_unchanged() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        // First run
        let s1 = pipeline.ingest_source(&scanner, &cfg).await;
        assert_eq!(s1.items_seen, 2);
        assert_eq!(s1.items_skipped, 0);

        // Second run (no changes)
        let s2 = pipeline.ingest_source(&scanner, &cfg).await;
        assert_eq!(s2.items_seen, 2);
        assert_eq!(s2.items_skipped, 2);
        assert_eq!(s2.chunks_upserted, 0);
    }

    #[tokio::test]
    async fn orphan_sweep_stales_vectors() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        // First run
        {
            let pipeline = make_pipeline(corpus.path(), 16).await;
            pipeline.ingest_source(&scanner, &cfg).await;
            let count = pipeline.store().row_count().await.unwrap();
            assert!(count > 0);
        }

        // Remove a file
        std::fs::remove_file(fixtures.path().join("sub/b.md")).unwrap();

        // Second run (new pipeline -> new run_id)
        let pipeline = make_pipeline(corpus.path(), 16).await;
        let s2 = pipeline.ingest_source(&scanner, &cfg).await;
        assert_eq!(s2.items_seen, 1);
        assert_eq!(s2.chunks_purged, 0);
        assert!(s2.chunks_staled > 0);

        let count_after = pipeline.store().row_count().await.unwrap();
        assert_eq!(count_after, 4);
    }

    #[tokio::test]
    async fn orphan_sweep_deletes_code_vectors() {
        use ostk_recall_core::SourceConfig;
        use ostk_recall_scan::code::CodeScanner;

        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();

        let code_path = fixtures.path().join("main.rs");
        std::fs::write(&code_path, "fn main() { println!(\"hello\"); }").unwrap();

        let scanner = CodeScanner;
        let cfg = SourceConfig {
            kind: SourceKind::Code,
            project: Some("code-test".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec!["rs".into()],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };

        // First run
        {
            let pipeline = make_pipeline(corpus.path(), 16).await;
            pipeline.ingest_source(&scanner, &cfg).await;
            let count = pipeline.store().row_count().await.unwrap();
            assert_eq!(count, 1);
        }

        // Remove the file
        std::fs::remove_file(code_path).unwrap();

        // Second run
        let pipeline = make_pipeline(corpus.path(), 16).await;
        let s2 = pipeline.ingest_source(&scanner, &cfg).await;
        assert_eq!(s2.items_seen, 0);
        assert_eq!(s2.chunks_purged, 1);
        assert_eq!(s2.chunks_staled, 0);

        let count_after = pipeline.store().row_count().await.unwrap();
        assert_eq!(count_after, 0, "code chunks must be physically deleted");
    }

    #[tokio::test]
    async fn scan_paths_direct_paths_upserts_only_those_files() {
        use ostk_recall_scan::markdown::MarkdownScanner;

        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());
        // Add a third file we will NOT pass to scan_paths.
        std::fs::write(
            fixtures.path().join("ignored.md"),
            "# Ignored\n\nshould not be ingested\n",
        )
        .unwrap();

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        let before = pipeline.store().row_count().await.unwrap();

        let paths = vec![
            fixtures.path().join("a.md"),
            fixtures.path().join("sub/b.md"),
        ];
        let per = pipeline
            .scan_paths(&[(&scanner, &cfg)], &paths)
            .await
            .unwrap();
        assert_eq!(per.len(), 1, "one source matched");
        let stats = per[0].1;
        assert_eq!(stats.items_seen, 2, "exactly the 2 input files");
        assert!(stats.chunks_emitted >= 3);
        assert_eq!(stats.chunks_upserted, stats.chunks_emitted);
        assert_eq!(stats.errors, 0);

        let after = pipeline.store().row_count().await.unwrap();
        assert_eq!(after - before, stats.chunks_upserted);
    }

    #[tokio::test]
    async fn scan_paths_multi_source_path_fires_each_source() {
        use ostk_recall_scan::code::CodeScanner;
        use ostk_recall_scan::markdown::MarkdownScanner;

        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        // One markdown file and one rust file under the same root.
        std::fs::write(
            fixtures.path().join("notes.md"),
            "# Notes\n\nIntro.\n\n## Section\n\nbody\n",
        )
        .unwrap();
        std::fs::write(
            fixtures.path().join("main.rs"),
            "fn main() { println!(\"hi\"); }\n",
        )
        .unwrap();

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let md_scanner = MarkdownScanner;
        let code_scanner = CodeScanner;
        let md_cfg = SourceConfig {
            kind: SourceKind::Markdown,
            project: Some("md-proj".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec![],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };
        let code_cfg = SourceConfig {
            kind: SourceKind::Code,
            project: Some("code-proj".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec!["rs".into()],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };

        let paths = vec![
            fixtures.path().join("notes.md"),
            fixtures.path().join("main.rs"),
        ];
        let per = pipeline
            .scan_paths(
                &[(&md_scanner, &md_cfg), (&code_scanner, &code_cfg)],
                &paths,
            )
            .await
            .unwrap();
        assert_eq!(per.len(), 2, "both sources fired");

        let md_stats = per
            .iter()
            .find(|(l, _)| l == "md-proj")
            .map(|(_, s)| *s)
            .expect("md source stats");
        let code_stats = per
            .iter()
            .find(|(l, _)| l == "code-proj")
            .map(|(_, s)| *s)
            .expect("code source stats");

        // Markdown's `extensions` is empty so it sees both inputs but only
        // ingests the .md (the rust file falls through `is_markdown`); code's
        // ext filter narrows to the .rs.
        assert!(md_stats.chunks_upserted >= 1);
        assert_eq!(code_stats.items_seen, 1);
        assert_eq!(code_stats.chunks_upserted, 1);
        assert_eq!(md_stats.errors, 0);
        assert_eq!(code_stats.errors, 0);
    }

    #[tokio::test]
    async fn scan_paths_path_outside_every_source_returns_empty() {
        use ostk_recall_scan::markdown::MarkdownScanner;

        let fixtures = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());
        std::fs::write(outside.path().join("stray.md"), "# Stray\n\nbody\n").unwrap();

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        let before = pipeline.store().row_count().await.unwrap();
        let paths = vec![outside.path().join("stray.md")];
        let per = pipeline
            .scan_paths(&[(&scanner, &cfg)], &paths)
            .await
            .unwrap();
        assert!(per.is_empty(), "no source matched, no per-source entry");

        let after = pipeline.store().row_count().await.unwrap();
        assert_eq!(
            after, before,
            "no upserts when path lies outside every source"
        );
    }

    #[tokio::test]
    async fn orphan_sweep_surfaces_stale_in_query() {
        use futures::TryStreamExt;
        use lancedb::query::ExecutableQuery;

        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        // First run
        {
            let pipeline = make_pipeline(corpus.path(), 16).await;
            pipeline.ingest_source(&scanner, &cfg).await;
        }

        // Remove a file
        std::fs::remove_file(fixtures.path().join("sub/b.md")).unwrap();

        // Second run
        let pipeline = make_pipeline(corpus.path(), 16).await;
        let stats = pipeline.ingest_source(&scanner, &cfg).await;
        assert!(stats.chunks_staled > 0);

        // Verify stale flag in LanceDB
        let table = pipeline
            .store()
            .connection()
            .open_table(ostk_recall_store::CORPUS_TABLE)
            .execute()
            .await
            .unwrap();
        let stream = table.query().execute().await.unwrap();
        let batches: Vec<arrow_array::RecordBatch> = stream.try_collect().await.unwrap();

        let mut found_stale = false;
        for batch in batches {
            let stale_col = batch
                .column_by_name("stale")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::BooleanArray>()
                .unwrap();
            for i in 0..batch.num_rows() {
                if stale_col.value(i) {
                    found_stale = true;
                }
            }
        }
        assert!(found_stale, "at least one chunk should be marked stale");
    }

    // gh#7 — delete-event handling in `Pipeline::scan_paths`
    //
    // Watch fires on a delete event → the path lands in the trigger
    // payload → `scan_paths` should purge the corpus rows for that path.
    // `code` retention is `Delete`, so we use it to assert physical
    // corpus row count goes down.
    #[tokio::test]
    async fn scan_paths_delete_event_purges_corpus_for_code_source() {
        use ostk_recall_scan::code::CodeScanner;

        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        let code_path = fixtures.path().join("main.rs");
        std::fs::write(&code_path, "fn main() { println!(\"hi\"); }").unwrap();

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = CodeScanner;
        let cfg = SourceConfig {
            kind: SourceKind::Code,
            project: Some("code-test".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec!["rs".into()],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };

        // Initial ingest via scan_paths (single-path).
        let per = pipeline
            .scan_paths(&[(&scanner, &cfg)], std::slice::from_ref(&code_path))
            .await
            .unwrap();
        assert_eq!(per.len(), 1);
        assert_eq!(per[0].1.chunks_upserted, 1);
        assert_eq!(pipeline.store().row_count().await.unwrap(), 1);

        // Delete the file on disk; trigger payload still names it.
        std::fs::remove_file(&code_path).unwrap();

        let per = pipeline
            .scan_paths(&[(&scanner, &cfg)], std::slice::from_ref(&code_path))
            .await
            .unwrap();
        assert_eq!(per.len(), 1, "source still matched on input path");
        let stats = per[0].1;
        assert_eq!(stats.items_seen, 0, "discover_paths yielded nothing");
        assert_eq!(
            stats.chunks_purged, 1,
            "missing-on-disk path purged 1 chunk"
        );
        assert_eq!(stats.errors, 0);
        assert_eq!(
            pipeline.store().row_count().await.unwrap(),
            0,
            "corpus row physically deleted under RetentionPolicy::Delete"
        );

        // verify_counts must not widen: corpus_total == ingest_total.
        let report = pipeline.verify_counts().await.unwrap();
        assert!(
            report.is_consistent(),
            "verify must not show drift after delete"
        );
        assert_eq!(report.corpus_total, 0);
        assert_eq!(report.ingest_total, 0);
    }

    // gh#7 acceptance #3 — rename arrives as (old: Remove) + (new: Create)
    // in the same trigger payload. First-cut behavior is delete-old +
    // ingest-new.
    #[tokio::test]
    async fn scan_paths_rename_event_purges_old_and_ingests_new() {
        use ostk_recall_scan::code::CodeScanner;

        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        let old_path = fixtures.path().join("old.rs");
        let new_path = fixtures.path().join("new.rs");
        std::fs::write(&old_path, "fn old() { println!(\"old\"); }").unwrap();

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = CodeScanner;
        let cfg = SourceConfig {
            kind: SourceKind::Code,
            project: Some("rename-test".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec!["rs".into()],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };

        pipeline
            .scan_paths(&[(&scanner, &cfg)], std::slice::from_ref(&old_path))
            .await
            .unwrap();
        assert_eq!(pipeline.store().row_count().await.unwrap(), 1);

        // Simulate rename: old is gone, new exists, both in payload.
        std::fs::rename(&old_path, &new_path).unwrap();
        let payload = vec![old_path.clone(), new_path.clone()];

        let per = pipeline
            .scan_paths(&[(&scanner, &cfg)], &payload)
            .await
            .unwrap();
        assert_eq!(per.len(), 1);
        let stats = per[0].1;
        assert_eq!(stats.items_seen, 1, "only new path yields");
        assert_eq!(stats.chunks_upserted, 1, "new path ingested");
        assert_eq!(stats.chunks_purged, 1, "old path purged");
        assert_eq!(stats.errors, 0);

        // Net delta zero — same number of chunks under a different id.
        assert_eq!(pipeline.store().row_count().await.unwrap(), 1);
        let report = pipeline.verify_counts().await.unwrap();
        assert!(report.is_consistent());
    }

    // gh#7 acceptance #4 — path that was never ingested: no-op, no
    // errors, counter stays at 0.
    #[tokio::test]
    async fn scan_paths_delete_for_never_ingested_path_is_noop() {
        use ostk_recall_scan::code::CodeScanner;

        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();

        // Write one file, ingest it, then ask scan_paths to "delete" a
        // sibling that was never seen.
        let known = fixtures.path().join("known.rs");
        std::fs::write(&known, "fn known() {}").unwrap();
        let unknown = fixtures.path().join("unknown.rs");
        // unknown is NOT created on disk and was never ingested.

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = CodeScanner;
        let cfg = SourceConfig {
            kind: SourceKind::Code,
            project: Some("noop-test".into()),
            paths: vec![fixtures.path().to_string_lossy().into_owned()],
            ignore: vec![],
            extensions: vec!["rs".into()],
            id: None,
            source_config_id: "test-cfg".to_string(),
            facets: Default::default(),
        };

        pipeline
            .scan_paths(&[(&scanner, &cfg)], std::slice::from_ref(&known))
            .await
            .unwrap();
        let baseline = pipeline.store().row_count().await.unwrap();
        assert_eq!(baseline, 1);

        let per = pipeline
            .scan_paths(&[(&scanner, &cfg)], &[unknown])
            .await
            .unwrap();
        assert_eq!(per.len(), 1, "source still matched on input path");
        let stats = per[0].1;
        assert_eq!(stats.items_seen, 0);
        assert_eq!(stats.chunks_purged, 0, "never-ingested path => no-op");
        assert_eq!(stats.chunks_staled, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(
            pipeline.store().row_count().await.unwrap(),
            baseline,
            "corpus untouched"
        );
    }

    // gh#7 — markdown retention is Stale, so a missing-on-disk path
    // marks chunks stale (not deletes them). Asserts the policy fan-out
    // in `purge_missing_paths` matches the orphan-sweep contract.
    #[tokio::test]
    async fn scan_paths_delete_event_stales_markdown_corpus() {
        use ostk_recall_scan::markdown::MarkdownScanner;

        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        let md_path = fixtures.path().join("doomed.md");
        std::fs::write(&md_path, "# Doomed\n\nbody\n").unwrap();

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        let per = pipeline
            .scan_paths(&[(&scanner, &cfg)], std::slice::from_ref(&md_path))
            .await
            .unwrap();
        let baseline = per[0].1.chunks_upserted;
        assert!(baseline > 0);

        std::fs::remove_file(&md_path).unwrap();
        let per = pipeline
            .scan_paths(&[(&scanner, &cfg)], std::slice::from_ref(&md_path))
            .await
            .unwrap();
        let stats = per[0].1;
        assert_eq!(stats.chunks_staled, baseline);
        assert_eq!(stats.chunks_purged, 0);
        assert_eq!(stats.errors, 0);

        // Markdown retention keeps the row in corpus (stale=true), so
        // verify_counts shows: ingest decremented, corpus unchanged —
        // matches existing Stale semantics for the markdown source.
    }

    // ===== phase 4: ingest_synthetic + subscribe_ingest =====

    use ostk_recall_core::{Chunk, Links, Source};

    fn make_synthetic_chunk(source_id: &str, idx: u32, text: &str) -> Chunk {
        let chunk_id = Chunk::make_id(Source::Gemini, source_id, idx, "");
        let sha = Chunk::content_hash(text);
        Chunk {
            chunk_id,
            source: Source::Gemini,
            project: Some("phase4-test".into()),
            source_id: source_id.to_string(),
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            source_config_id: "test-cfg".to_string(),
            chunk_index: idx,
            ts: None,
            role: None,
            text: text.to_string(),
            sha256: sha,
            links: Links::default(),
            extra: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn ingest_synthetic_round_trip() {
        let corpus = TempDir::new().unwrap();
        let pipeline = make_pipeline(corpus.path(), 16).await;
        let before = pipeline.store().row_count().await.unwrap();

        let chunks = vec![
            make_synthetic_chunk("turn:001", 0, "alpha body"),
            make_synthetic_chunk("turn:001", 1, "beta body"),
            make_synthetic_chunk("turn:002", 0, "gamma body"),
        ];
        let expected_ids: Vec<String> = chunks.iter().map(|c| c.chunk_id.clone()).collect();

        let stats = pipeline
            .ingest_synthetic(
                chunks,
                SyntheticSourceMeta {
                    source: SourceKind::Gemini,
                    project: Some("phase4-test".into()),
                },
            )
            .await
            .unwrap();

        assert_eq!(stats.chunks_emitted, 3);
        assert_eq!(stats.chunks_upserted, 3);
        assert_eq!(stats.errors, 0);

        let after = pipeline.store().row_count().await.unwrap();
        assert_eq!(after - before, 3, "exactly 3 new corpus rows");

        // Re-ingest the same chunks; Tier 2 content-hash dedupe should
        // collapse them to chunks_skipped_dup with no new upserts.
        let again = pipeline
            .ingest_synthetic(
                expected_ids
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let (sid, idx, text) = match i {
                            0 => ("turn:001", 0u32, "alpha body"),
                            1 => ("turn:001", 1u32, "beta body"),
                            _ => ("turn:002", 0u32, "gamma body"),
                        };
                        make_synthetic_chunk(sid, idx, text)
                    })
                    .collect(),
                SyntheticSourceMeta {
                    source: SourceKind::Gemini,
                    project: Some("phase4-test".into()),
                },
            )
            .await
            .unwrap();
        assert_eq!(again.chunks_upserted, 0);
        assert_eq!(again.chunks_skipped_dup, 3);
        assert_eq!(
            pipeline.store().row_count().await.unwrap(),
            after,
            "no new rows on dedupe re-ingest"
        );
    }

    #[tokio::test]
    async fn subscribe_ingest_delivers_event() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let mut rx = pipeline.subscribe_ingest();

        let scanner = MarkdownScanner;
        let stats = pipeline
            .ingest_source(&scanner, &cfg_for(fixtures.path()))
            .await;
        assert!(stats.chunks_upserted > 0);

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .expect("event delivered within timeout")
            .expect("recv ok");
        assert!(
            !event.chunk_ids_upserted.is_empty(),
            "event carries upserted chunk_ids"
        );
        assert_eq!(event.chunks_upserted, stats.chunks_upserted);
        assert_eq!(event.source, SourceKind::Markdown);
        assert_eq!(event.project.as_deref(), Some("test"));
        assert!(
            !event.source_ids.is_empty(),
            "event carries source_ids covering this batch"
        );
    }

    #[tokio::test]
    async fn multiple_subscribers_each_receive() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let mut rx1 = pipeline.subscribe_ingest();
        let mut rx2 = pipeline.subscribe_ingest();

        let scanner = MarkdownScanner;
        pipeline
            .ingest_source(&scanner, &cfg_for(fixtures.path()))
            .await;

        let ev1 = tokio::time::timeout(std::time::Duration::from_secs(1), rx1.recv())
            .await
            .unwrap()
            .unwrap();
        let ev2 = tokio::time::timeout(std::time::Duration::from_secs(1), rx2.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(ev1.chunk_ids_upserted, ev2.chunk_ids_upserted);
        assert_eq!(ev1.chunks_upserted, ev2.chunks_upserted);
    }

    #[tokio::test]
    async fn lagged_subscriber_recovers() {
        use tokio::sync::broadcast::error::RecvError;

        let corpus = TempDir::new().unwrap();
        let pipeline = make_pipeline(corpus.path(), 16).await;
        let mut slow = pipeline.subscribe_ingest();

        // Send INGEST_BROADCAST_CAPACITY + 1 synthetic ingests without
        // draining `slow`. The first event slot is evicted; the slow
        // receiver's next recv() yields RecvError::Lagged(n) and then
        // continues from the oldest still-live slot.
        let overflow = INGEST_BROADCAST_CAPACITY + 1;
        for i in 0..overflow {
            let sid = format!("turn:{i:05}");
            let chunk = make_synthetic_chunk(&sid, 0, &format!("body-{i}"));
            pipeline
                .ingest_synthetic(
                    vec![chunk],
                    SyntheticSourceMeta {
                        source: SourceKind::Gemini,
                        project: Some("phase4-test".into()),
                    },
                )
                .await
                .unwrap();
        }

        let first = slow.recv().await;
        match first {
            Err(RecvError::Lagged(n)) => {
                assert!(n >= 1, "lag count reports >= 1 dropped event");
            }
            other => panic!("expected Lagged, got {other:?}"),
        }

        // After the Lagged report, the receiver recovers and delivers
        // the next live event.
        let next = slow.recv().await.expect("recover after lag");
        assert!(!next.chunk_ids_upserted.is_empty());
    }

    #[tokio::test]
    async fn synthetic_ingest_emits_broadcast() {
        let corpus = TempDir::new().unwrap();
        let pipeline = make_pipeline(corpus.path(), 16).await;
        let mut rx = pipeline.subscribe_ingest();

        let chunks = vec![
            make_synthetic_chunk("turn:syn", 0, "synthetic body one"),
            make_synthetic_chunk("turn:syn", 1, "synthetic body two"),
        ];
        let expected: Vec<String> = chunks.iter().map(|c| c.chunk_id.clone()).collect();

        let stats = pipeline
            .ingest_synthetic(
                chunks,
                SyntheticSourceMeta {
                    source: SourceKind::Gemini,
                    project: Some("phase4-test".into()),
                },
            )
            .await
            .unwrap();
        assert_eq!(stats.chunks_upserted, 2);

        let event = tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(event.source, SourceKind::Gemini);
        assert_eq!(event.project.as_deref(), Some("phase4-test"));
        assert_eq!(event.chunks_upserted, 2);
        assert_eq!(event.chunks_stale, 0);
        let mut got = event.chunk_ids_upserted;
        let mut want = expected.clone();
        got.sort();
        want.sort();
        assert_eq!(got, want, "event carries the synthetic chunk_ids");
    }
}
