//! Auto-weaver (Phase 7 of the attention substrate).
//!
//! Subscribes to `Pipeline::subscribe_ingest`. For each batch of
//! freshly-upserted chunks it computes cosine similarity against the
//! anchor vector of every thread that has one, and writes a `Derived`
//! evidence link for each (chunk, anchor) pair above a per-source
//! threshold. Chunks that resonate with two or more anchors surface as
//! `ProposedWeave` values — the caller decides what to do with them
//! (a future phase will materialize these as substrate proposals).
//!
//! The weaver runs as a single sequential consumer; lag in the
//! broadcast channel is observed and skipped (not fatal). The pipeline
//! never blocks on us — the broadcast publisher is always non-blocking.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use ostk_recall_core::attention::{FoldDepth, PrivacyTier};
use ostk_recall_core::{IngestOrigin, SourceKind, ThreadHandle};
use ostk_recall_pipeline::{IngestEvent, Pipeline};
use ostk_recall_store::{
    AssociationType, CorpusStore, EvidenceLink, RelationState, StoreError, TensionState,
    ThreadRecord, ThreadsDb,
};
use tokio_util::sync::CancellationToken;

use crate::{AttentionError, cosine_similarity};

/// Per-source-kind similarity thresholds.
///
/// Code is permissive (smaller anchors, more partial matches); transcript
/// is strict (long noisy chunks dominated by chatter); prose / default
/// fall in between.
#[derive(Debug, Clone, Copy)]
pub struct WeaverThresholds {
    /// Cosine cut-off for `SourceKind::Markdown` chunks.
    pub prose: f32,
    /// Cosine cut-off for `SourceKind::Code` chunks (lower → permissive).
    pub code: f32,
    /// Cosine cut-off for transcript-shaped sources
    /// (`ClaudeCode`, `Gemini`, `ZipExport`) — set high to filter chatter.
    pub transcript: f32,
    /// Cosine cut-off for source kinds not listed above.
    pub default: f32,
}

impl Default for WeaverThresholds {
    fn default() -> Self {
        Self {
            prose: 0.82,
            code: 0.78,
            transcript: 0.85,
            default: 0.80,
        }
    }
}

impl WeaverThresholds {
    /// Look up the cut-off for a given source kind.
    #[must_use]
    pub const fn for_source(&self, source: SourceKind) -> f32 {
        match source {
            SourceKind::Markdown => self.prose,
            SourceKind::Code => self.code,
            SourceKind::ClaudeCode | SourceKind::Gemini | SourceKind::ZipExport => self.transcript,
            _ => self.default,
        }
    }
}

/// A chunk that resonated above threshold with two or more anchors.
///
/// The auto-weaver does not write proposed-weave rows itself — it
/// returns them per-event for the caller to surface (CLI / MCP / future
/// substrate writer).
#[derive(Debug, Clone)]
pub struct ProposedWeave {
    /// Thread handles whose anchors all resonated with the same chunk.
    pub anchors: Vec<ThreadHandle>,
    /// Chunk ids shared by the resonating anchors (typically one per
    /// `ProposedWeave`; the field is a `Vec` so future grouping passes
    /// can collapse co-resonant chunks into a single weave).
    pub shared_chunks: Vec<String>,
}

/// Result of processing a single ingest event.
#[derive(Debug)]
pub struct WeaverOutcome {
    /// The ingest event the weaver consumed.
    pub event_seen: IngestEvent,
    /// Count of `evidence_links` rows the weaver inserted for this
    /// event. Idempotent collisions (already-written edges) do not
    /// count.
    pub evidence_links_written: usize,
    /// Count of existing edges re-touched (touch_count bumped) because a
    /// chunk re-resonated with an anchor it had already linked. The
    /// future→past curation signal (P11b-full).
    pub evidence_links_touched: usize,
    /// Per-chunk groupings the weaver detected but did not write —
    /// the caller decides whether to surface them as candidate threads.
    pub proposed_weaves: Vec<ProposedWeave>,
    /// Count of `threads_proposed` rows the weaver inserted for this
    /// event (one per emergent cluster found in the unmatched
    /// chunks).
    pub proposed_threads_written: usize,
}

/// Aggregate result from a windowed corpus weave pass.
#[derive(Debug, Default)]
pub struct WeaveWindowOutcome {
    /// Number of synthesized source-kind batches processed.
    pub batches_processed: usize,
    /// Number of active corpus chunks handed to `process_event`.
    pub chunks_seen: usize,
    /// Count of newly inserted `evidence_links` rows. Existing links are
    /// idempotent no-ops and are not counted here.
    pub evidence_links_written: usize,
    /// Count of existing edges re-touched (strengthened) across the pass.
    pub evidence_links_touched: usize,
    /// Count of detected chunk-to-multiple-anchor groupings.
    pub proposed_weaves: usize,
    /// Count of new `threads_proposed` rows.
    pub proposed_threads_written: usize,
    /// Count of stale, unpromoted proposals pruned at the end of the pass
    /// (the consolidation counterweight; promoted proposals are kept).
    pub proposals_pruned: usize,
}

/// Aggregate result from a coarse `consolidate` pass — the P11b-full
/// consolidation cycle layered on top of `weave_window`.
#[derive(Debug, Default)]
pub struct ConsolidateOutcome {
    /// The deep re-weave of the window (binds recent arrivals, re-touches
    /// edges, prunes stale proposals).
    pub window: WeaveWindowOutcome,
    /// New anchor↔thread bridge edges written by the anchor re-weave.
    pub anchor_bridges_written: usize,
    /// Existing anchor↔thread bridges re-touched (strengthened) this pass.
    pub anchor_bridges_touched: usize,
    /// Recurring, high-cohesion proposals promoted to durable threads.
    pub proposals_promoted: usize,
    /// Near-duplicate threads merged away (anchors near-identical).
    pub threads_merged: usize,
    /// Deeply-familiar stable threads folded to a sparse summary depth.
    pub threads_abstracted: usize,
    /// Threads down-transitioned (Active→Slack→Dormant) by the idle fade.
    pub threads_faded: usize,
}

/// Unpromoted emergent proposals older than this are pruned at the end of a
/// `weave_window` pass. Generous enough to leave recent proposals for
/// operator review; the content-based handle prevents re-accumulation, so
/// this only clears the genuinely stale tail.
const PROPOSED_PRUNE_AGE_DAYS: i64 = 14;

// --- consolidation tunables (P11b-full) -----------------------------------

/// Cosine cut-off for the anchor↔anchor bridge re-weave in a consolidation
/// pass. High on purpose: a consolidated thread only bridges to another
/// when their anchors genuinely resonate (a canyon-spanning link, not
/// same-day adjacency).
const ANCHOR_BRIDGE_THRESHOLD: f32 = 0.85;
/// A proposal is auto-promoted to a real thread only when its
/// distinct-content cluster is at least this large …
const PROMOTE_MIN_DISTINCT_SIZE: usize = 5;
/// … and at least this cohesive. Auto-promotion writes a durable thread,
/// so the bar is "clearly a real, recurring unit," not a passing cluster.
const PROMOTE_MIN_COHESION: f32 = 0.9;
/// Base idle days after which an `Active` thread fades to `Slack`. Scaled
/// up by familiarity (see `idle_fade_multiplier`) so deeply-familiar threads
/// persist longer — the present curates which past stays load-bearing.
const FADE_SLACK_IDLE_DAYS: f64 = 14.0;
/// Base idle days after which a thread fades to `Dormant`.
const FADE_DORMANT_IDLE_DAYS: f64 = 60.0;
/// Anchor cosine at/above which two threads are near-duplicates and merge.
/// Deliberately strict — merging deletes a thread, so the bar is "these are
/// the same thread under two handles," not "these are related."
const MERGE_ANCHOR_SIMILARITY: f32 = 0.95;

/// Familiarity stretches the idle-fade windows: a saturated thread tolerates
/// up to ~2× the idle before fading. Mirrors the `familiarity_floor` curve.
#[allow(clippy::cast_precision_loss)]
fn idle_fade_multiplier(familiarity: u32) -> f64 {
    1.0 + f64::from(familiarity.min(crate::FAMILIARITY_SATURATION))
        / f64::from(crate::FAMILIARITY_SATURATION)
}

/// Coarseness ranking for tension, so the offline fade only ever
/// *down*-transitions (Active → Slack → Dormant); reanimation is the live
/// curator's job (it has the resonance signal the offline pass lacks).
const fn tension_rank(t: TensionState) -> u8 {
    match t {
        TensionState::Active => 2,
        TensionState::Slack => 1,
        TensionState::Dormant => 0,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WeaverError {
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("attention error: {0}")]
    Attention(#[from] AttentionError),
}

/// Resonance-driven evidence-link writer.
///
/// Holds only the dependencies `process_event` needs (the ledger, the
/// corpus, and per-source thresholds). `run` adds the broadcast wiring
/// on top — separating the two lets unit tests drive `process_event`
/// directly without standing up a `Pipeline`.
pub struct AutoWeaver {
    threads: Arc<ThreadsDb>,
    corpus: Arc<CorpusStore>,
    thresholds: WeaverThresholds,
}

impl AutoWeaver {
    #[must_use]
    pub const fn new(
        threads: Arc<ThreadsDb>,
        corpus: Arc<CorpusStore>,
        thresholds: WeaverThresholds,
    ) -> Self {
        Self {
            threads,
            corpus,
            thresholds,
        }
    }

    /// Weave an explicit corpus window without weakening the live
    /// TurnEnd gate in [`Self::run`].
    ///
    /// `since = None` scans the whole active corpus. `since = Some(d)`
    /// scans chunks whose corpus timestamp is within the last `d`.
    /// Batches are grouped by source kind so existing per-source
    /// thresholds continue to apply. Synthetic membrane chunks are
    /// skipped: they are substrate self-writes, not library content.
    pub async fn weave_window(
        &self,
        since: Option<chrono::Duration>,
        epoch_size: usize,
    ) -> Result<WeaveWindowOutcome, WeaverError> {
        let cutoff = since.map(|d| Utc::now() - d);
        let anchor_snapshot = self.load_anchor_snapshot().await?;
        let batches = self
            .corpus
            .chunk_id_batches_by_source_window(cutoff, epoch_size)
            .await?;
        let mut aggregate = WeaveWindowOutcome::default();
        for batch in batches {
            if batch.source == SourceKind::Membrane {
                continue;
            }
            let event = IngestEvent {
                project: None,
                source: batch.source,
                source_ids: vec![],
                chunks_upserted: batch.chunk_ids.len(),
                chunk_ids_upserted: batch.chunk_ids,
                chunks_stale: 0,
                ts: Utc::now(),
                // This is intentionally Bulk. The live gate remains:
                // only `run()` decides TurnEnd eligibility, while this
                // explicit pass invokes the processing primitive.
                origin: IngestOrigin::Bulk,
            };
            aggregate.chunks_seen += event.chunk_ids_upserted.len();
            let outcome = self
                .process_event_with_anchors(event, &anchor_snapshot)
                .await?;
            aggregate.batches_processed += 1;
            aggregate.evidence_links_written += outcome.evidence_links_written;
            aggregate.evidence_links_touched += outcome.evidence_links_touched;
            aggregate.proposed_weaves += outcome.proposed_weaves.len();
            aggregate.proposed_threads_written += outcome.proposed_threads_written;
        }
        // Consolidation counterweight: fade the stale, unpromoted tail.
        aggregate.proposals_pruned = self
            .threads
            .prune_proposed_threads_older_than(chrono::Duration::days(PROPOSED_PRUNE_AGE_DAYS))?;
        Ok(aggregate)
    }

    /// Run a coarse **consolidation cycle** over a temporal window — the
    /// P11b-full layer that goes beyond capture. Where `weave_window` only
    /// binds and proposes, `consolidate` also: deep-re-weaves recent arrivals,
    /// bridges consolidated threads across canyons, promotes recurring
    /// high-quality proposals into durable threads, and fades idle threads so
    /// the surfaced past stays load-bearing without drowning the present.
    ///
    /// Offline by design: the operator schedules it via `weave --consolidate`
    /// under cron/launchd, never `serve`. It does not make bulk content look
    /// like live TurnEnds — it invokes the processing primitives directly.
    pub async fn consolidate(
        &self,
        since: Option<chrono::Duration>,
        epoch_size: usize,
    ) -> Result<ConsolidateOutcome, WeaverError> {
        let mut out = ConsolidateOutcome::default();

        // 1. Deep re-weave: bind recent arrivals to the anchor set, re-touch
        //    edges that re-resonate, prune the stale proposal tail.
        out.window = self.weave_window(since, epoch_size).await?;

        // 2. Anchor↔anchor re-weave: cross-link consolidated threads whose
        //    anchors genuinely resonate (canyon-spanning bridges). Feeding the
        //    anchor set as both candidates and anchors reuses the matched-
        //    anchor pass; self-links are skipped (anchor_id == chunk_id guard).
        let snap = self.load_anchor_snapshot().await?;
        if !snap.anchor_embeds.is_empty() {
            let pass = self.match_against_anchors(
                &snap.anchor_embeds,
                &snap.threads,
                &snap.anchor_embeds,
                ANCHOR_BRIDGE_THRESHOLD,
                "bridge",
            )?;
            out.anchor_bridges_written = pass.links_written;
            out.anchor_bridges_touched = pass.links_touched;
        }

        // 3. Merge near-duplicate threads (same thread under two handles).
        //    Uses the same anchor snapshot; safe because it mutates the DB,
        //    not the snapshot, and later steps re-query fresh.
        out.threads_merged = self.merge_near_duplicate_threads(&snap)?;

        // 4. Promote recurring, high-cohesion proposals to durable threads.
        out.proposals_promoted = self.promote_recurring_proposals()?;

        // 5. Abstract deeply-familiar stable threads to a sparse summary fold.
        out.threads_abstracted = self.abstract_stable_threads()?;

        // 6. Fade idle threads (the present curates the past). Down-only; the
        //    live curator owns reanimation (it has the resonance signal this
        //    offline pass lacks).
        out.threads_faded = self.fade_idle_threads()?;

        Ok(out)
    }

    /// Merge near-duplicate threads — pairs whose anchors are near-identical
    /// (`MERGE_ANCHOR_SIMILARITY`). Greedy: the more-familiar thread (tiebreak:
    /// older) absorbs the other, so the consolidated map keeps the
    /// load-bearing handle. Operates over the snapshot but mutates the DB;
    /// merged-away handles are skipped for the rest of the pass.
    fn merge_near_duplicate_threads(
        &self,
        snap: &AnchorSnapshot,
    ) -> Result<usize, WeaverError> {
        let threads = &snap.threads;
        let mut merged_away: HashSet<String> = HashSet::new();
        let mut count = 0usize;
        for i in 0..threads.len() {
            if merged_away.contains(threads[i].handle.as_str()) {
                continue;
            }
            let Some(vi) = threads[i]
                .anchor_chunk_id
                .as_ref()
                .and_then(|id| snap.anchor_embeds.get(id))
            else {
                continue;
            };
            for j in (i + 1)..threads.len() {
                if merged_away.contains(threads[j].handle.as_str()) {
                    continue;
                }
                let Some(vj) = threads[j]
                    .anchor_chunk_id
                    .as_ref()
                    .and_then(|id| snap.anchor_embeds.get(id))
                else {
                    continue;
                };
                if cosine_similarity(vi, vj) < MERGE_ANCHOR_SIMILARITY {
                    continue;
                }
                // Keep the stronger thread (more familiar; tiebreak: older).
                let a = &threads[i];
                let b = &threads[j];
                let keep_a = a.familiarity > b.familiarity
                    || (a.familiarity == b.familiarity && a.created_at <= b.created_at);
                let (into, from) = if keep_a { (a, b) } else { (b, a) };
                if self.threads.merge_thread(&from.handle, &into.handle)? {
                    merged_away.insert(from.handle.as_str().to_string());
                    count += 1;
                    // If the outer thread was merged away, stop pairing it.
                    if from.handle == threads[i].handle {
                        break;
                    }
                }
            }
        }
        Ok(count)
    }

    /// Fold deeply-familiar, stable (`Active`) threads to the most-collapsed
    /// summary depth so the lens surfaces them sparsely — structural
    /// abstraction via the existing `fold_override` primitive (no text
    /// generation; the anchor already names the thread). Idempotent: threads
    /// already folded are skipped.
    fn abstract_stable_threads(&self) -> Result<usize, WeaverError> {
        let mut count = 0usize;
        for t in self.threads.list_threads(None)? {
            if t.familiarity >= crate::FAMILIARITY_SATURATION
                && t.tension == TensionState::Active
                && t.fold_override != Some(FoldDepth::Folded)
            {
                self.threads
                    .set_fold_override(&t.handle, Some(FoldDepth::Folded))?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Promote proposals whose distinct-content cluster is large and cohesive
    /// enough to be a real, recurring unit. A *surviving* proposal is recurring
    /// by construction: the date-free handle (RT-5) means the weaver re-proposes
    /// a still-live cluster idempotently rather than re-creating it, and stale
    /// proposals are pruned — so a proposal that is still present and clears the
    /// size/cohesion bar has recurred across passes. The new thread anchors on
    /// the cluster's first chunk. Idempotent: already-promoted rows are skipped.
    fn promote_recurring_proposals(&self) -> Result<usize, WeaverError> {
        let proposals = self.threads.list_proposed_threads()?;
        let mut promoted = 0usize;
        for p in proposals {
            if p.promoted_to.is_some()
                || p.chunk_ids.len() < PROMOTE_MIN_DISTINCT_SIZE
                || p.cohesion < PROMOTE_MIN_COHESION
            {
                continue;
            }
            let Some(anchor) = p.chunk_ids.first() else {
                continue;
            };
            // The proposed handle is already a content-derived `proposed-<hex>`
            // string; reuse it as the thread handle so the lineage is explicit.
            let Ok(handle) = ThreadHandle::new(&p.proposed_handle) else {
                continue;
            };
            let now = Utc::now();
            let rec = ThreadRecord {
                handle: handle.clone(),
                tension: TensionState::Active,
                familiarity: 0,
                last_touched_at: now,
                anchor_chunk_id: Some(anchor.clone()),
                fold_override: None,
                created_at: now,
                created_scope_key: None,
                privacy_tier: PrivacyTier::T1Project,
            };
            self.threads.upsert_thread(&rec)?;
            self.threads
                .mark_proposed_thread_promoted(&p.proposed_handle, &handle)?;
            promoted += 1;
        }
        Ok(promoted)
    }

    /// Fade threads that have gone idle, down-transitioning tension by time
    /// since `last_touched_at`. Familiarity stretches the windows so deeply-
    /// familiar threads persist longer. Down-only — never reanimates.
    #[allow(clippy::cast_precision_loss)]
    fn fade_idle_threads(&self) -> Result<usize, WeaverError> {
        let now = Utc::now();
        let mut faded = 0usize;
        for t in self.threads.list_threads(None)? {
            let idle_days = (now - t.last_touched_at).num_seconds().max(0) as f64 / 86_400.0;
            let mult = idle_fade_multiplier(t.familiarity);
            let target = if idle_days >= FADE_DORMANT_IDLE_DAYS * mult {
                TensionState::Dormant
            } else if idle_days >= FADE_SLACK_IDLE_DAYS * mult {
                TensionState::Slack
            } else {
                continue;
            };
            if tension_rank(target) < tension_rank(t.tension) {
                self.threads.set_tension(&t.handle, target)?;
                faded += 1;
            }
        }
        Ok(faded)
    }

    /// Subscribe to `pipeline` and drive the weaver until `cancel`
    /// fires or the pipeline channel closes. Per-event errors are
    /// logged at `warn` and the loop continues — daemon survival is
    /// more valuable than per-event strict failure, and the audit
    /// chain still records the event that triggered it. Channel lag
    /// is logged at `warn` with the skipped count so it's visible.
    pub async fn run(
        &self,
        pipeline: &Pipeline,
        cancel: CancellationToken,
    ) -> Result<(), WeaverError> {
        let mut rx = pipeline.subscribe_ingest();
        loop {
            tokio::select! {
                biased;
                res = rx.recv() => {
                    match res {
                        Ok(event) => {
                            // Weave live TurnEnds incrementally; bulk-
                            // ingested content is woven by the periodic
                            // whole-corpus epoch pass, not per event
                            // (avoids per-bulk-event corpus queries).
                            if !event.is_turn_end() {
                                continue;
                            }
                            if let Err(err) = self.process_event(event).await {
                                tracing::warn!(error = %err, "auto-weaver: process_event failed");
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(skipped = n, "auto-weaver: ingest channel lagged");
                        }
                    }
                }
                () = cancel.cancelled() => {
                    // Drain anything already queued before exiting so the
                    // shutdown handshake doesn't race the broadcast.
                    while let Ok(event) = rx.try_recv() {
                        if !event.is_turn_end() {
                            continue;
                        }
                        if let Err(err) = self.process_event(event).await {
                            tracing::warn!(error = %err, "auto-weaver: process_event failed during drain");
                        }
                    }
                    return Ok(());
                }
            }
        }
    }

    /// Process a single ingest event end-to-end. Returns counts and
    /// any proposed weaves; on success the evidence links are already
    /// committed to the ledger.
    pub async fn process_event(&self, event: IngestEvent) -> Result<WeaverOutcome, WeaverError> {
        let anchor_snapshot = self.load_anchor_snapshot().await?;
        self.process_event_with_anchors(event, &anchor_snapshot)
            .await
    }

    async fn process_event_with_anchors(
        &self,
        event: IngestEvent,
        anchor_snapshot: &AnchorSnapshot,
    ) -> Result<WeaverOutcome, WeaverError> {
        if event.chunk_ids_upserted.is_empty() {
            return Ok(WeaverOutcome {
                event_seen: event,
                evidence_links_written: 0,
                evidence_links_touched: 0,
                proposed_weaves: vec![],
                proposed_threads_written: 0,
            });
        }

        let new_embeds = self
            .corpus
            .fetch_embeddings(&event.chunk_ids_upserted)
            .await?;
        let threshold = self.thresholds.for_source(event.source);
        let category = source_kind_to_category(event.source);

        let MatchPassOutcome {
            links_written,
            links_touched,
            chunk_to_anchors,
            matched_chunks,
        } = self.match_against_anchors(
            &new_embeds,
            &anchor_snapshot.threads,
            &anchor_snapshot.anchor_embeds,
            threshold,
            category,
        )?;

        let proposed_weaves: Vec<ProposedWeave> = chunk_to_anchors
            .into_iter()
            .filter(|(_, anchors)| anchors.len() >= 2)
            .map(|(chunk_id, anchors)| ProposedWeave {
                anchors,
                shared_chunks: vec![chunk_id],
            })
            .collect();

        // Emergent-cluster pass: chunks that didn't resonate with any
        // existing anchor are candidates for proposing brand-new
        // threads. Density clustering is deterministic and cheap at
        // batch sizes <= ~50 (per-scan). Failure to write a proposal
        // never blocks the matched-path work above.
        let unmatched: Vec<(String, Vec<f32>)> = new_embeds
            .iter()
            .filter(|(id, _)| {
                !matched_chunks.contains(*id) && !anchor_snapshot.anchor_embeds.contains_key(*id)
            })
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();
        let proposed_threads_written =
            self.write_emergent_proposals(&unmatched)
                .unwrap_or_else(|err| {
                    tracing::warn!(error = %err, "auto-weaver: emergent-cluster write failed");
                    0
                });

        Ok(WeaverOutcome {
            event_seen: event,
            evidence_links_written: links_written,
            evidence_links_touched: links_touched,
            proposed_weaves,
            proposed_threads_written,
        })
    }

    async fn load_anchor_snapshot(&self) -> Result<AnchorSnapshot, WeaverError> {
        let threads = self.threads.list_threads(None)?;
        let anchor_ids: Vec<String> = threads
            .iter()
            .filter_map(|t| t.anchor_chunk_id.clone())
            .collect();
        let anchor_embeds = if anchor_ids.is_empty() {
            HashMap::new()
        } else {
            self.corpus.fetch_embeddings(&anchor_ids).await?
        };
        Ok(AnchorSnapshot {
            threads,
            anchor_embeds,
        })
    }

    /// Match every (chunk, anchor) pair and persist the resulting
    /// `Derived` evidence links. Extracted from `process_event` so the
    /// outer function stays under the line budget; the body and error
    /// shape are identical to the inlined version.
    fn match_against_anchors(
        &self,
        new_embeds: &HashMap<String, Vec<f32>>,
        threads: &[ostk_recall_store::ThreadRecord],
        anchor_embeds: &HashMap<String, Vec<f32>>,
        threshold: f32,
        category: &str,
    ) -> Result<MatchPassOutcome, WeaverError> {
        let mut out = MatchPassOutcome::default();
        for (chunk_id, chunk_vec) in new_embeds {
            for thread in threads {
                let Some(anchor_id) = &thread.anchor_chunk_id else {
                    continue;
                };
                if anchor_id == chunk_id {
                    continue;
                }
                let Some(anchor_vec) = anchor_embeds.get(anchor_id) else {
                    continue;
                };
                let sim = cosine_similarity(chunk_vec, anchor_vec);
                if sim < threshold {
                    continue;
                }
                out.matched_chunks.insert(chunk_id.clone());
                let now = Utc::now();
                let link = EvidenceLink {
                    id: 0,
                    thread_handle: thread.handle.clone(),
                    // The corpus chunk-id is the only durable handle we
                    // have on the resonating content; the threads
                    // scanner is what knows about source paths.
                    original_path: PathBuf::from(chunk_id),
                    current_path: None,
                    content_hash: None,
                    last_resolved_chunk_id: Some(chunk_id.clone()),
                    relation_state: RelationState::Active,
                    association_type: AssociationType::Derived,
                    category: category.to_string(),
                    similarity: Some(sim),
                    created_at: now,
                    updated_at: now,
                    touch_count: 1,
                    last_touched_at: now,
                };
                match self.threads.add_evidence_link(&link) {
                    Ok(_) => {
                        out.links_written += 1;
                        out.chunk_to_anchors
                            .entry(chunk_id.clone())
                            .or_default()
                            .push(thread.handle.clone());
                    }
                    Err(StoreError::UniqueViolation { .. }) => {
                        // The (thread, path, category) edge already exists.
                        // A re-resonance is the future→past curation signal
                        // (P11b-full): bump touch_count + last_touched_at so
                        // recurring canyon edges strengthen instead of the
                        // signal being dropped. Counts as a link touched.
                        match self.threads.touch_evidence_link(
                            &thread.handle,
                            &link.original_path,
                            &link.category,
                            now,
                        ) {
                            Ok(true) => out.links_touched += 1,
                            Ok(false) => {
                                tracing::trace!(
                                    thread = %thread.handle,
                                    chunk = %chunk_id,
                                    "auto-weaver: re-touch found no matching edge",
                                );
                            }
                            Err(err) => return Err(WeaverError::from(err)),
                        }
                    }
                    Err(err) => return Err(WeaverError::from(err)),
                }
            }
        }
        Ok(out)
    }

    /// Cluster unmatched chunks and write one `threads_proposed` row
    /// per emergent cluster. Returns the number of proposals written.
    fn write_emergent_proposals(
        &self,
        unmatched: &[(String, Vec<f32>)],
    ) -> Result<usize, WeaverError> {
        let clusters = crate::cluster::find_clusters(unmatched, crate::cluster::EMERGENT_THRESHOLD);
        if clusters.is_empty() {
            return Ok(0);
        }
        let now = Utc::now();
        let mut written = 0usize;
        for cluster in clusters {
            let handle = generate_proposed_handle(&cluster.chunk_ids);
            let rec = ostk_recall_store::ProposedThreadRecord {
                id: 0,
                proposed_handle: handle,
                chunk_ids: cluster.chunk_ids,
                centroid_vec: cluster.centroid,
                cohesion: cluster.cohesion,
                created_at: now,
                promoted_to: None,
            };
            match self.threads.insert_proposed_thread(&rec) {
                Ok(_) => written += 1,
                Err(StoreError::UniqueViolation { .. }) => {
                    // A proposal with this handle already exists —
                    // possible if two scans nominate near-identical
                    // clusters in the same second. Skip; the existing
                    // row is still valid.
                    tracing::trace!(
                        handle = %rec.proposed_handle,
                        "auto-weaver: proposed thread already exists",
                    );
                }
                Err(err) => return Err(WeaverError::from(err)),
            }
        }
        Ok(written)
    }
}

/// Intermediate result from the matched-anchor pass.
#[derive(Debug, Default)]
struct MatchPassOutcome {
    links_written: usize,
    links_touched: usize,
    chunk_to_anchors: HashMap<String, Vec<ThreadHandle>>,
    matched_chunks: std::collections::HashSet<String>,
}

#[derive(Debug)]
struct AnchorSnapshot {
    threads: Vec<ThreadRecord>,
    anchor_embeds: HashMap<String, Vec<f32>>,
}

/// Build a kebab-case proposed-thread handle from a cluster's member
/// chunks. Format: `proposed-<8 hex chars>` — a pure function of the
/// (sorted) member set, short enough to fit `ThreadHandle`'s 64-char /
/// 4-hyphen budget.
///
/// Content-based, with no timestamp: the same cluster re-surfaced on any
/// later weave collapses to the same handle and the UNIQUE constraint
/// absorbs it (idempotent). Previously the handle mixed in the date, so an
/// identical cluster re-proposed every day — the main driver of unbounded
/// proposal accumulation.
pub(crate) fn generate_proposed_handle(chunk_ids: &[String]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for id in chunk_ids {
        hasher.update(id.as_bytes());
        hasher.update(b"\n");
    }
    let digest = hasher.finalize();
    let short = hex::encode(&digest[..4]);
    format!("proposed-{short}")
}

const fn source_kind_to_category(source: SourceKind) -> &'static str {
    match source {
        SourceKind::Code => "code",
        SourceKind::Markdown => "doc",
        SourceKind::ClaudeCode | SourceKind::Gemini | SourceKind::ZipExport => "transcript",
        _ => "other",
    }
}

// -----------------------------------------------------------------------
// tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use ostk_recall_core::{Chunk, Links, PrivacyTier, Source};
    use ostk_recall_store::{
        CorpusStore, ProposedThreadRecord, TensionState, ThreadRecord, ThreadsDb,
    };
    use tempfile::TempDir;

    const DIM: usize = 4;

    struct Fixture {
        _tmp: TempDir,
        corpus: Arc<CorpusStore>,
        threads: Arc<ThreadsDb>,
    }

    async fn fixture() -> Fixture {
        let tmp = TempDir::new().unwrap();
        let corpus = CorpusStore::open_or_create(tmp.path(), DIM).await.unwrap();
        let threads = ThreadsDb::open(tmp.path()).unwrap();
        Fixture {
            _tmp: tmp,
            corpus: Arc::new(corpus),
            threads: Arc::new(threads),
        }
    }

    fn chunk(id: &str) -> Chunk {
        chunk_with(id, Source::Markdown, None)
    }

    fn chunk_with(id: &str, source: Source, ts: Option<chrono::DateTime<Utc>>) -> Chunk {
        Chunk {
            chunk_id: id.into(),
            source,
            project: Some("test".into()),
            source_id: format!("{id}.md"),
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            source_config_id: "test-cfg".to_string(),
            chunk_index: 0,
            ts,
            role: None,
            text: format!("text for {id}"),
            sha256: Chunk::content_hash(id),
            links: Links::default(),
            extra: serde_json::Value::Null,
        }
    }

    fn thread(handle: &str, anchor_chunk_id: Option<&str>) -> ThreadRecord {
        let now = Utc::now();
        ThreadRecord {
            handle: ThreadHandle::new(handle).unwrap(),
            tension: ostk_recall_store::TensionState::Active,
            familiarity: 0,
            last_touched_at: now,
            anchor_chunk_id: anchor_chunk_id.map(String::from),
            fold_override: None,
            created_at: now,
            created_scope_key: None,
            privacy_tier: PrivacyTier::T1Project,
        }
    }

    fn event(source: SourceKind, chunk_ids: &[&str]) -> IngestEvent {
        IngestEvent {
            project: Some("test".into()),
            source,
            source_ids: vec![],
            chunk_ids_upserted: chunk_ids.iter().map(|s| (*s).to_string()).collect(),
            chunks_upserted: chunk_ids.len(),
            chunks_stale: 0,
            ts: Utc::now(),
            origin: ostk_recall_core::IngestOrigin::Watch,
        }
    }

    fn weaver(fx: &Fixture) -> AutoWeaver {
        AutoWeaver::new(
            fx.threads.clone(),
            fx.corpus.clone(),
            WeaverThresholds::default(),
        )
    }

    #[tokio::test]
    async fn empty_event_returns_early() {
        let fx = fixture().await;
        let out = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &[]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 0);
        assert!(out.proposed_weaves.is_empty());
    }

    #[tokio::test]
    async fn process_event_with_no_anchors_yields_zero_links() {
        let fx = fixture().await;
        fx.threads.upsert_thread(&thread("orphan", None)).unwrap();
        fx.corpus
            .upsert(&[chunk("c1")], &[vec![1.0, 0.0, 0.0, 0.0]])
            .await
            .unwrap();
        let out = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &["c1"]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 0);
        assert!(out.proposed_weaves.is_empty());
    }

    #[tokio::test]
    async fn process_event_writes_link_above_threshold() {
        let fx = fixture().await;
        let anchor_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        fx.corpus
            .upsert(
                &[chunk("anchor-1"), chunk("new-1")],
                &[anchor_vec.clone(), anchor_vec.clone()],
            )
            .await
            .unwrap();
        fx.threads
            .upsert_thread(&thread("t1", Some("anchor-1")))
            .unwrap();
        let out = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &["new-1"]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 1);
        let evidence = fx
            .threads
            .list_evidence(&ThreadHandle::new("t1").unwrap())
            .unwrap();
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].association_type, AssociationType::Derived);
        assert!(evidence[0].similarity.unwrap() > 0.99);
    }

    #[tokio::test]
    async fn re_resonance_touches_existing_edge() {
        // First pass writes the edge; a second pass over the same chunk
        // re-resonates with the same anchor → the edge is re-touched
        // (touch_count bumps, last_touched_at advances), not duplicated.
        // This is the P11b-full future→past curation loop.
        let fx = fixture().await;
        let anchor_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        fx.corpus
            .upsert(
                &[chunk("anchor-1"), chunk("new-1")],
                &[anchor_vec.clone(), anchor_vec.clone()],
            )
            .await
            .unwrap();
        fx.threads
            .upsert_thread(&thread("t1", Some("anchor-1")))
            .unwrap();

        let first = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &["new-1"]))
            .await
            .unwrap();
        assert_eq!(first.evidence_links_written, 1);
        assert_eq!(first.evidence_links_touched, 0);

        let second = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &["new-1"]))
            .await
            .unwrap();
        assert_eq!(
            second.evidence_links_written, 0,
            "the edge already exists; no new row"
        );
        assert_eq!(
            second.evidence_links_touched, 1,
            "the re-resonance strengthens the existing edge"
        );

        let evidence = fx
            .threads
            .list_evidence(&ThreadHandle::new("t1").unwrap())
            .unwrap();
        assert_eq!(evidence.len(), 1, "re-touch must not duplicate the row");
        assert_eq!(evidence[0].touch_count, 2);
    }

    #[tokio::test]
    async fn process_event_does_not_link_thread_to_its_own_anchor() {
        let fx = fixture().await;
        let anchor_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        fx.corpus
            .upsert(&[chunk("anchor-1")], &[anchor_vec])
            .await
            .unwrap();
        fx.threads
            .upsert_thread(&thread("t1", Some("anchor-1")))
            .unwrap();
        let out = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &["anchor-1"]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 0);
        assert!(
            fx.threads
                .list_evidence(&ThreadHandle::new("t1").unwrap())
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn process_event_skips_below_threshold() {
        let fx = fixture().await;
        let anchor_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let orth_vec = vec![0.0_f32, 1.0, 0.0, 0.0];
        fx.corpus
            .upsert(
                &[chunk("anchor-1"), chunk("new-1")],
                &[anchor_vec, orth_vec],
            )
            .await
            .unwrap();
        fx.threads
            .upsert_thread(&thread("t1", Some("anchor-1")))
            .unwrap();
        let out = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &["new-1"]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 0);
    }

    #[tokio::test]
    async fn per_domain_threshold_applies() {
        // Build a (chunk, anchor) pair whose cosine sim is exactly 0.79.
        // Code threshold is 0.78 → above → write.
        // Transcript threshold is 0.85 → below → skip.
        let cos = 0.79_f32;
        let anchor_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let other_vec = vec![cos, cos.mul_add(-cos, 1.0).sqrt(), 0.0, 0.0];
        let actual = cosine_similarity(&anchor_vec, &other_vec);
        assert!((actual - 0.79).abs() < 1e-5, "got cos = {actual}");

        // Code event → above 0.78 threshold → writes link.
        {
            let fx = fixture().await;
            fx.corpus
                .upsert(
                    &[chunk("a"), chunk("b")],
                    &[anchor_vec.clone(), other_vec.clone()],
                )
                .await
                .unwrap();
            fx.threads.upsert_thread(&thread("tc", Some("a"))).unwrap();
            let out = weaver(&fx)
                .process_event(event(SourceKind::Code, &["b"]))
                .await
                .unwrap();
            assert_eq!(
                out.evidence_links_written, 1,
                "code @ 0.79 should be above 0.78 threshold"
            );
        }

        // Transcript event (ClaudeCode) → below 0.85 threshold → skip.
        {
            let fx = fixture().await;
            fx.corpus
                .upsert(&[chunk("a"), chunk("b")], &[anchor_vec, other_vec])
                .await
                .unwrap();
            fx.threads.upsert_thread(&thread("tt", Some("a"))).unwrap();
            let out = weaver(&fx)
                .process_event(event(SourceKind::ClaudeCode, &["b"]))
                .await
                .unwrap();
            assert_eq!(
                out.evidence_links_written, 0,
                "transcript @ 0.79 should be below 0.85 threshold"
            );
        }
    }

    #[tokio::test]
    async fn proposed_weave_when_two_anchors_share_chunk() {
        let fx = fixture().await;
        let common = vec![1.0_f32, 0.0, 0.0, 0.0];
        fx.corpus
            .upsert(
                &[chunk("a1"), chunk("a2"), chunk("new")],
                &[common.clone(), common.clone(), common.clone()],
            )
            .await
            .unwrap();
        fx.threads.upsert_thread(&thread("t1", Some("a1"))).unwrap();
        fx.threads.upsert_thread(&thread("t2", Some("a2"))).unwrap();
        let out = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &["new"]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 2);
        assert_eq!(out.proposed_weaves.len(), 1);
        let weave = &out.proposed_weaves[0];
        assert_eq!(weave.shared_chunks, vec!["new".to_string()]);
        assert_eq!(weave.anchors.len(), 2);
    }

    #[tokio::test]
    async fn unmatched_chunks_become_proposed_thread_when_dense() {
        // Three chunks share an anchor vector; no anchors exist yet,
        // so the weaver routes them through the emergent-cluster path
        // and writes a `threads_proposed` row.
        let fx = fixture().await;
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        // Tiny jitter so the cluster vectors are not bit-identical but
        // still resonate above the EMERGENT_THRESHOLD (0.82).
        let v2 = vec![1.0_f32, 0.01, 0.0, 0.0];
        let v3 = vec![1.0_f32, 0.0, 0.01, 0.0];
        fx.corpus
            .upsert(&[chunk("u1"), chunk("u2"), chunk("u3")], &[v, v2, v3])
            .await
            .unwrap();
        let out = weaver(&fx)
            .process_event(event(SourceKind::Markdown, &["u1", "u2", "u3"]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 0, "no anchors → no links");
        assert_eq!(out.proposed_threads_written, 1, "one emergent cluster");
        let proposals = fx.threads.list_proposed_threads().unwrap();
        assert_eq!(proposals.len(), 1);
        let p = &proposals[0];
        assert_eq!(p.chunk_ids.len(), 3);
        assert!(p.proposed_handle.starts_with("proposed-"));
        assert!(p.cohesion > 0.99);
        assert!(p.promoted_to.is_none());
    }

    #[tokio::test]
    async fn weave_window_groups_by_source_threshold_and_skips_membrane() {
        let fx = fixture().await;
        let cos = 0.79_f32;
        let anchor_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let other_vec = vec![cos, cos.mul_add(-cos, 1.0).sqrt(), 0.0, 0.0];
        let now = Utc::now();
        fx.corpus
            .upsert(
                &[
                    chunk_with("anchor", Source::Markdown, Some(now)),
                    chunk_with("code-hit", Source::Code, Some(now)),
                    chunk_with("transcript-skip", Source::ClaudeCode, Some(now)),
                    chunk_with("membrane-skip", Source::Membrane, Some(now)),
                ],
                &[
                    anchor_vec.clone(),
                    other_vec.clone(),
                    other_vec.clone(),
                    anchor_vec.clone(),
                ],
            )
            .await
            .unwrap();
        fx.threads
            .upsert_thread(&thread("t1", Some("anchor")))
            .unwrap();

        let out = weaver(&fx).weave_window(None, 1).await.unwrap();
        assert_eq!(out.chunks_seen, 3, "membrane chunks are not woven");
        assert_eq!(out.evidence_links_written, 1);

        let evidence = fx
            .threads
            .list_evidence(&ThreadHandle::new("t1").unwrap())
            .unwrap();
        assert_eq!(evidence.len(), 1);
        assert_eq!(
            evidence[0].last_resolved_chunk_id.as_deref(),
            Some("code-hit")
        );
        assert_eq!(evidence[0].category, "code");
    }

    #[tokio::test]
    async fn weave_window_since_filters_candidate_chunks_not_anchors() {
        let fx = fixture().await;
        let now = Utc::now();
        let old = now - chrono::Duration::hours(2);
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        fx.corpus
            .upsert(
                &[
                    chunk_with("anchor", Source::Markdown, Some(old)),
                    chunk_with("old-candidate", Source::Markdown, Some(old)),
                    chunk_with("new-candidate", Source::Markdown, Some(now)),
                ],
                &[v.clone(), v.clone(), v.clone()],
            )
            .await
            .unwrap();
        fx.threads
            .upsert_thread(&thread("t1", Some("anchor")))
            .unwrap();

        let out = weaver(&fx)
            .weave_window(Some(chrono::Duration::hours(1)), 8)
            .await
            .unwrap();
        assert_eq!(out.chunks_seen, 1);
        assert_eq!(out.evidence_links_written, 1);

        let evidence = fx
            .threads
            .list_evidence(&ThreadHandle::new("t1").unwrap())
            .unwrap();
        assert_eq!(evidence.len(), 1);
        assert_eq!(
            evidence[0].last_resolved_chunk_id.as_deref(),
            Some("new-candidate")
        );
    }

    #[test]
    fn thresholds_for_source_routing() {
        let t = WeaverThresholds::default();
        assert!((t.for_source(SourceKind::Code) - 0.78).abs() < 1e-6);
        assert!((t.for_source(SourceKind::Markdown) - 0.82).abs() < 1e-6);
        assert!((t.for_source(SourceKind::ClaudeCode) - 0.85).abs() < 1e-6);
        assert!((t.for_source(SourceKind::Gemini) - 0.85).abs() < 1e-6);
        assert!((t.for_source(SourceKind::ZipExport) - 0.85).abs() < 1e-6);
        assert!((t.for_source(SourceKind::FileGlob) - 0.80).abs() < 1e-6);
    }

    fn proposal(handle: &str, n_chunks: usize, cohesion: f32) -> ProposedThreadRecord {
        ProposedThreadRecord {
            id: 0,
            proposed_handle: handle.to_string(),
            chunk_ids: (0..n_chunks).map(|i| format!("{handle}-c{i}")).collect(),
            centroid_vec: vec![1.0, 0.0, 0.0, 0.0],
            cohesion,
            created_at: Utc::now(),
            promoted_to: None,
        }
    }

    fn thread_touched(handle: &str, tension: TensionState, days_idle: i64, fam: u32) -> ThreadRecord {
        let touched = Utc::now() - chrono::Duration::days(days_idle);
        ThreadRecord {
            handle: ThreadHandle::new(handle).unwrap(),
            tension,
            familiarity: fam,
            last_touched_at: touched,
            anchor_chunk_id: None,
            fold_override: None,
            created_at: touched,
            created_scope_key: None,
            privacy_tier: PrivacyTier::T1Project,
        }
    }

    #[tokio::test]
    async fn promote_recurring_proposals_gates_on_size_and_cohesion() {
        let fx = fixture().await;
        // Passing: large + cohesive.
        fx.threads
            .insert_proposed_thread(&proposal("proposed-bigcohesive", 6, 0.95))
            .unwrap();
        // Failing: too small.
        fx.threads
            .insert_proposed_thread(&proposal("proposed-small", 2, 0.99))
            .unwrap();
        // Failing: too loose.
        fx.threads
            .insert_proposed_thread(&proposal("proposed-loose", 8, 0.5))
            .unwrap();

        let promoted = weaver(&fx).promote_recurring_proposals().unwrap();
        assert_eq!(promoted, 1, "only the large+cohesive proposal promotes");

        // The promoted proposal became a real, anchored thread …
        let t = fx
            .threads
            .get_thread(&ThreadHandle::new("proposed-bigcohesive").unwrap())
            .unwrap()
            .expect("promoted thread exists");
        assert_eq!(t.anchor_chunk_id.as_deref(), Some("proposed-bigcohesive-c0"));
        // … and the proposal row is marked promoted (won't re-promote).
        let again = weaver(&fx).promote_recurring_proposals().unwrap();
        assert_eq!(again, 0, "already-promoted proposals are skipped");

        // The rejected ones did not become threads.
        assert!(
            fx.threads
                .get_thread(&ThreadHandle::new("proposed-loose").unwrap())
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn fade_idle_threads_down_transitions_by_idle_time() {
        let fx = fixture().await;
        // Fresh → stays Active (idle below the slack window).
        fx.threads
            .upsert_thread(&thread_touched("fresh", TensionState::Active, 1, 0))
            .unwrap();
        // 20 days idle, familiarity 0 → past the 14d slack window → Slack.
        fx.threads
            .upsert_thread(&thread_touched("idle-slack", TensionState::Active, 20, 0))
            .unwrap();
        // 90 days idle → past the 60d dormant window → Dormant.
        fx.threads
            .upsert_thread(&thread_touched("idle-dormant", TensionState::Active, 90, 0))
            .unwrap();
        // 20 days idle but saturated familiarity (×2 window = 28d slack) →
        // still within tolerance → stays Active.
        fx.threads
            .upsert_thread(&thread_touched("familiar", TensionState::Active, 20, 20))
            .unwrap();

        let faded = weaver(&fx).fade_idle_threads().unwrap();
        assert_eq!(faded, 2, "idle-slack and idle-dormant fade; others hold");

        let get = |h: &str| {
            fx.threads
                .get_thread(&ThreadHandle::new(h).unwrap())
                .unwrap()
                .unwrap()
                .tension
        };
        assert_eq!(get("fresh"), TensionState::Active);
        assert_eq!(get("idle-slack"), TensionState::Slack);
        assert_eq!(get("idle-dormant"), TensionState::Dormant);
        assert_eq!(
            get("familiar"),
            TensionState::Active,
            "familiarity stretches the idle window"
        );
    }

    #[tokio::test]
    async fn fade_idle_threads_never_up_transitions() {
        let fx = fixture().await;
        // A Dormant thread that is fresh must NOT be reanimated by the
        // offline fade — that's the live curator's job.
        fx.threads
            .upsert_thread(&thread_touched("dormant-fresh", TensionState::Dormant, 0, 0))
            .unwrap();
        let faded = weaver(&fx).fade_idle_threads().unwrap();
        assert_eq!(faded, 0);
        assert_eq!(
            fx.threads
                .get_thread(&ThreadHandle::new("dormant-fresh").unwrap())
                .unwrap()
                .unwrap()
                .tension,
            TensionState::Dormant
        );
    }

    #[tokio::test]
    async fn merge_near_duplicate_threads_absorbs_into_stronger() {
        let fx = fixture().await;
        // Two threads with byte-identical anchor embeddings (cosine 1.0).
        let v = vec![1.0_f32, 0.0, 0.0, 0.0];
        fx.corpus
            .upsert(&[chunk("anchor-keep"), chunk("anchor-dup")], &[v.clone(), v])
            .await
            .unwrap();
        // `keep` is more familiar → it absorbs `dup`.
        let mut keep = thread("keepthread", Some("anchor-keep"));
        keep.familiarity = 10;
        fx.threads.upsert_thread(&keep).unwrap();
        fx.threads
            .upsert_thread(&thread("dupthread", Some("anchor-dup")))
            .unwrap();
        // An evidence row on the soon-to-be-merged-away thread.
        let now = Utc::now();
        fx.threads
            .add_evidence_link(&EvidenceLink {
                id: 0,
                thread_handle: ThreadHandle::new("dupthread").unwrap(),
                original_path: PathBuf::from("evidence-1"),
                current_path: None,
                content_hash: None,
                last_resolved_chunk_id: None,
                relation_state: RelationState::Active,
                association_type: AssociationType::Derived,
                category: "doc".into(),
                similarity: Some(0.9),
                created_at: now,
                updated_at: now,
                touch_count: 1,
                last_touched_at: now,
            })
            .unwrap();

        let snap = weaver(&fx).load_anchor_snapshot().await.unwrap();
        let merged = weaver(&fx).merge_near_duplicate_threads(&snap).unwrap();
        assert_eq!(merged, 1);

        // `dup` is gone; `keep` remains and inherited the evidence row.
        assert!(
            fx.threads
                .get_thread(&ThreadHandle::new("dupthread").unwrap())
                .unwrap()
                .is_none()
        );
        let kept_evidence = fx
            .threads
            .list_evidence(&ThreadHandle::new("keepthread").unwrap())
            .unwrap();
        assert!(
            kept_evidence
                .iter()
                .any(|e| e.original_path == PathBuf::from("evidence-1")),
            "merged-away thread's evidence is re-pointed onto the survivor"
        );
    }

    #[tokio::test]
    async fn abstract_stable_threads_folds_only_familiar_active() {
        let fx = fixture().await;
        let mut familiar = thread("familiar-active", None);
        familiar.familiarity = crate::FAMILIARITY_SATURATION;
        fx.threads.upsert_thread(&familiar).unwrap();

        let mut casual = thread("casual-active", None);
        casual.familiarity = 5;
        fx.threads.upsert_thread(&casual).unwrap();

        let mut fam_slack = thread("familiar-slack", None);
        fam_slack.familiarity = crate::FAMILIARITY_SATURATION;
        fam_slack.tension = TensionState::Slack;
        fx.threads.upsert_thread(&fam_slack).unwrap();

        let n = weaver(&fx).abstract_stable_threads().unwrap();
        assert_eq!(n, 1, "only the deeply-familiar Active thread folds");
        assert_eq!(
            fx.threads
                .get_thread(&ThreadHandle::new("familiar-active").unwrap())
                .unwrap()
                .unwrap()
                .fold_override,
            Some(FoldDepth::Folded)
        );
        // Idempotent: a second pass folds nothing new.
        assert_eq!(weaver(&fx).abstract_stable_threads().unwrap(), 0);
    }
}
