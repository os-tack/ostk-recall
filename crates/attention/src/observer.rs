//! Turn observer (Phase 6 of the attention substrate).
//!
//! Single-pass producer over each conversational turn that emits three
//! outputs the rest of the substrate consumes:
//!
//! 1. **Membrane chunks** — recognition-language fragments, ingested
//!    through [`Pipeline::ingest_synthetic`] as `SourceKind::Membrane`.
//! 2. **Familiarity increments** — one tick per known thread handle
//!    that appears in the turn (de-duplicated; the unit is "turns
//!    containing this handle," not "occurrences").
//! 3. **Proposed thread stubs** — kebab-case noun phrases that look
//!    handle-shaped, recur in the turn, and aren't yet in the ledger.
//!    Returned as values; Phase 9 surfaces them through MCP.
//!
//! Companion docs:
//! - `daemons` (refinement-pass §1) — observer is one of the three
//!   in-process daemons; explicitly NOT a `Scanner`.
//! - `fade-is-concentration` — familiarity feeds the decay floor.
//! - `abi-as-sovereign-boundary` — synthetic chunks still carry
//!   structured attribution in `extra`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use chrono::Utc;
use ostk_recall_core::attention::{AttentionScope, PrivacyTier, ThreadHandle, ThreadHandleError};
use ostk_recall_core::{Chunk, Links, Source};
use ostk_recall_pipeline::{Pipeline, PipelineError, SyntheticSourceMeta};
use ostk_recall_store::corpus::{CorpusStore, StoreError};
use ostk_recall_store::{
    ChainEvent, ChainSink, ConceptActivationReader, ConceptStatus, EdgeSource, GLOBAL_PROJECT,
    OBSERVED_MENTION_CONFIDENCE, ProposedThreadRecord, TensionState, ThreadRecord, ThreadsDb,
    slugify,
};
use regex::Regex;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::concept_growth::{
    self, AMBIENT_EDGE_BY, CO_MENTION_RELATION, ConceptGrowthCache, ConceptGrowthConfig,
    TermRecurrence,
};

// --- public types -----------------------------------------------------

/// Outputs of a single [`TurnObserver::observe`] call.
#[derive(Debug, Clone, Default)]
pub struct ObservationResult {
    /// How many membrane chunks the pipeline actually ingested (post
    /// dedupe). Equals `chunks_upserted` on the inner pipeline call;
    /// duplicate chunks from re-observing an identical turn collapse.
    pub membrane_chunks_ingested: usize,
    /// Handles that were already known AND appeared in this turn.
    /// One entry per distinct handle; per-turn de-duped before counting.
    pub familiarity_increments: Vec<ThreadHandle>,
    /// Handle-shaped phrases that appeared >=`STUB_MIN_OCCURRENCES`
    /// times in the turn but didn't clear the promotion gate. Persisted
    /// to `threads_proposed` and surfaced here for operator review.
    pub proposed_stubs: Vec<ProposedThreadStub>,
    /// Handles auto-promoted to `Slack` threads in this observation.
    /// Empty unless the originating chunk hit
    /// `PROMOTE_MIN_OCCURRENCES` AND the session-wide promotion cap
    /// has room. See `TurnObserver::PROMOTION_CAP_PER_SESSION`.
    pub promoted_handles: Vec<ThreadHandle>,
    /// Count of `threads_proposed` rows written during this observation
    /// (both promotable and not-yet-promoted, idempotent on the proposed
    /// handle). Zero when `originating_chunk_id` is `None` — proposals
    /// require an anchor chunk for the audit trail.
    pub proposed_persisted: usize,
    /// Phase 2: newly-created `co_occurs` concept edges minted from the
    /// turn's resonance-gated co-mentions. Zero unless [`TurnObserver::with_corpus`]
    /// AND [`TurnObserver::with_attention`] are wired (the concept-growth phase
    /// is gated on both). Idempotent re-touch of an existing edge does not count.
    pub concept_edges_minted: usize,
    /// Phase 2: new `Proposed` concept nodes minted from salient recurring
    /// unknown terms. Same gating as `concept_edges_minted`.
    pub concept_nodes_minted: usize,
}

/// A candidate thread the observer wants the operator to consider.
#[derive(Debug, Clone)]
pub struct ProposedThreadStub {
    /// Kebab-case candidate (validated against [`ThreadHandle`] rules
    /// — anything that fails validation is silently dropped at detect
    /// time, so this string is always a legal handle).
    pub handle_guess: String,
    /// Up to `CONTEXT_SNIPPET_TOKENS` of turn text centered on the
    /// first occurrence.
    pub context_snippet: String,
    /// Heuristic strength in `[0, 1]`. v1: linear in occurrence count
    /// (capped at `MAX_CONFIDENCE_OCCURRENCES`).
    pub confidence: f32,
}

#[derive(Debug, Error)]
pub enum ObserverError {
    #[error("pipeline ingest: {0}")]
    Ingest(#[from] PipelineError),
    #[error("threads ledger: {0}")]
    Store(#[from] StoreError),
    #[error("invalid thread handle: {0}")]
    Handle(#[from] ThreadHandleError),
}

// --- tuning knobs -----------------------------------------------------

/// Approximate tokens around a recognition trigger captured in the
/// membrane chunk. "Token" here is whitespace-separated word, not BPE.
pub const MEMBRANE_CONTEXT_TOKENS: usize = 200;

/// Same budget reused for proposed-stub `context_snippet`.
pub const CONTEXT_SNIPPET_TOKENS: usize = 60;

/// A handle-shaped phrase must appear at least this many times in the
/// turn to count as a stub proposal. Single mentions are too noisy.
pub const STUB_MIN_OCCURRENCES: usize = 2;

/// Past this many occurrences the confidence stops climbing.
pub const MAX_CONFIDENCE_OCCURRENCES: usize = 6;

/// Minimum within-chunk occurrences for auto-promotion (count is the
/// binding gate; confidence is informational only — they're coupled in
/// the linear formula and over-constraining produces no usable knob).
/// Sized for transcript chunks, which are the only source kind where
/// repeated kebab-shaped phrases reliably surface; code chunks rarely
/// repeat handle-shaped tokens and will not auto-promote.
pub const PROMOTE_MIN_OCCURRENCES: usize = 3;

/// Hard cap on auto-promotions per `TurnObserver` instance (which
/// equals one session for the ambient daemon). Curator fade is the
/// long-term hygiene; this cap protects against the runaway
/// "mass-mention spam → hundreds of fake rows" failure mode.
pub const PROMOTION_CAP_PER_SESSION: usize = 8;

// --- the observer -----------------------------------------------------

/// In-process daemon that observes each conversational turn.
///
/// One per `(scope, pipeline, ledger)` triple. Cheap to clone — every
/// field is already `Arc`-shared.
#[derive(Clone)]
pub struct TurnObserver {
    pipeline: Arc<Pipeline>,
    store: Arc<ThreadsDb>,
    known_handles: Arc<RwLock<HashSet<ThreadHandle>>>,
    /// Count of auto-promotions performed by this observer instance.
    /// Compared against [`PROMOTION_CAP_PER_SESSION`] before each new
    /// promotion attempt. Atomic so cloned observers share the cap.
    promotions_this_session: Arc<AtomicUsize>,
    /// Optional handle to the in-memory attention store. When set, the
    /// observer calls `attend` + `familiarize` on every auto-promotion
    /// so the in-memory score tier sees the new thread immediately
    /// (otherwise the curator's stale-touch grace would expire one
    /// tick later, demoting the freshly-promoted thread to Dormant).
    /// Optional so test fixtures and library callers without a live
    /// score tier work unchanged.
    attention: Option<Arc<dyn crate::AttentionForwardStore>>,
    /// Optional chain sink for attention-side events. When set, every
    /// successful `attend()` emits `ChainEvent::RollingVectorSnapshot`
    /// and every noise-gate rejection emits
    /// `ChainEvent::AttentionTurnSkipped`. The Skipped path is wired
    /// in P6A but only fired in P6-full once the noise gate ships.
    /// Independent from the `ThreadsDb` sink so the observer can
    /// participate in chain emission without owning ledger writes.
    chain_sink: Option<Arc<dyn ChainSink>>,
    /// Phase 2: corpus handle for concept-anchor embeddings. `None` disables
    /// the concept-growth phase entirely (legacy/library callers, ledger-only
    /// tests) — the phase is gated on `corpus.is_some() && attention.is_some()`.
    corpus: Option<Arc<CorpusStore>>,
    /// Phase 2 tuning (resonance floor + per-turn / per-session caps), resolved
    /// from the `[ambient_growth]` config block by `serve`.
    concept_growth: ConceptGrowthConfig,
    /// Phase 2: the concept-growth read cache (anchor codebook + gazetteer),
    /// rebuilt every `concept_growth.codebook_rebuild_turns` observed turns.
    growth_cache: Arc<RwLock<ConceptGrowthCache>>,
    /// Observed turns since the last `growth_cache` rebuild. Atomic so the
    /// per-turn bump never needs the cache write lock; reset to 0 on rebuild.
    turns_since_build: Arc<AtomicU64>,
    /// Streaming recurrence accumulator for the node-minting half: unknown term
    /// → its mentions/resonance split + co-resonant context. Lives on the
    /// long-lived observer (one instance == one ambient session).
    term_recurrence: Arc<RwLock<HashMap<String, TermRecurrence>>>,
    /// Count of nodes minted by this observer instance, compared against
    /// [`ConceptGrowthConfig::node_mint_cap_per_session`]. Atomic so cloned
    /// observers share the cap (mirrors `promotions_this_session`).
    node_mints_this_session: Arc<AtomicUsize>,
}

impl TurnObserver {
    /// Build a new observer. `known_handles` starts empty; call
    /// [`Self::refresh_known_handles`] before the first turn (or
    /// periodically thereafter) to sync from the ledger.
    ///
    /// The in-memory attention store is not wired by default. Call
    /// [`Self::with_attention`] to attach one — production wiring in
    /// `cli::commands::spawn_ambient_daemons` does this; tests that
    /// only exercise the ledger path skip it.
    #[must_use]
    pub fn new(pipeline: Arc<Pipeline>, store: Arc<ThreadsDb>) -> Self {
        Self {
            pipeline,
            store,
            known_handles: Arc::new(RwLock::new(HashSet::new())),
            promotions_this_session: Arc::new(AtomicUsize::new(0)),
            attention: None,
            chain_sink: None,
            corpus: None,
            concept_growth: ConceptGrowthConfig::default(),
            growth_cache: Arc::new(RwLock::new(ConceptGrowthCache::default())),
            turns_since_build: Arc::new(AtomicU64::new(0)),
            term_recurrence: Arc::new(RwLock::new(HashMap::new())),
            node_mints_this_session: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Attach an in-memory attention store. Used by production wiring
    /// to make auto-promoted threads visible to the score tier
    /// immediately (rather than waiting for chain replay on next boot).
    #[must_use]
    pub fn with_attention(mut self, attention: Arc<dyn crate::AttentionForwardStore>) -> Self {
        self.attention = Some(attention);
        self
    }

    /// Attach a [`ChainSink`] so the observer-mediated attend path
    /// persists `RollingVectorSnapshot` / `AttentionTurnSkipped`
    /// events. Direct calls into `InMemoryAttention::attend()`
    /// (test-only) bypass this and emit no chain rows — the observer
    /// is the only path that persists, per the design split in
    /// `p6-attention-ema.md` ("Chain event ownership").
    #[must_use]
    pub fn with_chain_sink(mut self, sink: Arc<dyn ChainSink>) -> Self {
        self.chain_sink = Some(sink);
        self
    }

    /// Attach a [`CorpusStore`] so the concept-growth phase (Phase 2) can build
    /// its anchor codebook (`cosine(concept_anchor, turn_rolling_vec)` is the
    /// resonance gate). Without this — or without [`Self::with_attention`] —
    /// the phase is inert and `observe` grows no concept edges/nodes.
    /// Production wiring in `cli::commands::spawn_ambient_daemons` sets this.
    #[must_use]
    pub fn with_corpus(mut self, corpus: Arc<CorpusStore>) -> Self {
        self.corpus = Some(corpus);
        self
    }

    /// Override the concept-growth tuning (resonance floor + caps). `serve`
    /// resolves this from the optional `[ambient_growth]` config block.
    #[must_use]
    pub fn with_concept_growth(mut self, cfg: ConceptGrowthConfig) -> Self {
        self.concept_growth = cfg;
        self
    }

    /// Reload the in-memory known-handles set from the ledger.
    ///
    /// Cheap (one full-table scan over `threads`); call after batches
    /// of mutations or on a coarse timer. v1 takes the entire table.
    pub async fn refresh_known_handles(&self) -> Result<(), ObserverError> {
        let rows = self.store.list_threads(None)?;
        let fresh: HashSet<ThreadHandle> = rows.into_iter().map(|r| r.handle).collect();
        {
            let mut guard = self.known_handles.write().await;
            *guard = fresh;
        }
        Ok(())
    }

    /// Test/seed helper: insert handles directly into the known set
    /// without going through the ledger. Production code should call
    /// [`Self::refresh_known_handles`].
    pub async fn seed_known_handles<I>(&self, handles: I)
    where
        I: IntoIterator<Item = ThreadHandle>,
    {
        let mut guard = self.known_handles.write().await;
        guard.extend(handles);
    }

    /// Observe a single turn. Single pass over `turn_text` produces
    /// the outputs documented on [`ObservationResult`].
    ///
    /// `originating_chunk_id` is the corpus chunk_id whose text drove
    /// this observation. When `Some`, proposed stubs are persisted to
    /// `threads_proposed` and candidates at or above
    /// [`PROMOTE_MIN_OCCURRENCES`] auto-promote to `Slack` threads with
    /// this chunk_id as anchor (subject to
    /// [`PROMOTION_CAP_PER_SESSION`]). When `None`, no rows are written
    /// and the proposed-stubs path returns candidates only — used by
    /// unit tests where no real chunk exists.
    ///
    /// When [`Self::with_attention`] has attached an in-memory store,
    /// every auto-promotion also calls `attend(scope, turn_text)` and
    /// `familiarize(scope, handle)` on it so the score tier sees the
    /// thread immediately. v0.3.0 had a known gap here — the curator's
    /// stale-touch grace expired after one tick (default 60s) and
    /// demoted promotions to Dormant; v0.4.1 closes it.
    #[allow(clippy::too_many_arguments)]
    pub async fn observe(
        &self,
        scope: &AttentionScope,
        turn_text: &str,
        turn_seq: u64,
        session_id: &str,
        originating_chunk_id: Option<&str>,
    ) -> Result<ObservationResult, ObserverError> {
        let known_snapshot: HashSet<ThreadHandle> = self.known_handles.read().await.clone();

        // Hoisted: every observation updates the scope's in-memory
        // attention vector with the current turn's text. Done once per
        // call so the auto-promotion and known-handle paths share the
        // same fresh anchor seed when they call `familiarize` below.
        // Best-effort — a failing in-memory store must never abort the
        // durable observation.
        if let Some(attn) = &self.attention {
            match attn.attend(scope, turn_text).await {
                Ok(outcome) => {
                    // P6A — observer owns chain emission. Direct
                    // `attend()` calls (tests) bypass this path and
                    // emit no chain rows; only the observer-mediated
                    // route persists rolling-vector / turn-skipped
                    // events. See `p6-attention-ema.md` "Chain event
                    // ownership".
                    if let Some(sink) = &self.chain_sink {
                        let event = match outcome {
                            crate::AttendOutcome::Updated {
                                rolling_vec,
                                lambda,
                            } => ChainEvent::RollingVectorSnapshot {
                                scope: scope.clone(),
                                vec: rolling_vec,
                                lambda,
                                ts: Utc::now(),
                            },
                            crate::AttendOutcome::Skipped { reason } => {
                                ChainEvent::AttentionTurnSkipped {
                                    scope: scope.clone(),
                                    reason,
                                    ts: Utc::now(),
                                }
                            }
                        };
                        if let Err(err) = sink.append(&event) {
                            tracing::warn!(
                                error = %err,
                                "persistent-attention: chain append failed"
                            );
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "persistent-attention: attend failed at observe entry"
                    );
                }
            }
        }

        // (a) membrane chunks — recognition-language detection.
        let triggers = detect_recognition_triggers(turn_text);
        let chunks = build_membrane_chunks(turn_text, &triggers, session_id, turn_seq);
        let membrane_chunks_ingested = if chunks.is_empty() {
            0
        } else {
            let meta = SyntheticSourceMeta {
                source: ostk_recall_core::SourceKind::Membrane,
                project: scope.project.clone(),
            };
            let stats = self.pipeline.ingest_synthetic(chunks, meta).await?;
            // Newly-ingested OR re-observed-duplicate both count as a
            // recognized recognition moment for the caller; both shapes
            // are "we saw a trigger we'd ingest if it were fresh."
            stats.chunks_upserted + stats.chunks_skipped_dup
        };

        // (b) familiarity increments — one tick per distinct known
        // handle that shows up in the turn (case-sensitive, kebab-case
        // word-boundary match).
        //
        // Two-step per the chain-as-cognition-history policy:
        //   1. increment_familiarity per distinct handle → bumps counter,
        //      returns new value, NO chain row.
        //   2. record_familiarity_batch with (handle, familiarity_after)
        //      tuples → emits ONE chained event for the whole turn.
        // Calling record_familiarity_batch alone (the original Phase 6 shape)
        // chained the event but never advanced the counter, leaving
        // familiarity-based fold-depth defaults inert.
        let mut mentioned = handles_mentioned(turn_text, &known_snapshot);
        if !mentioned.is_empty() {
            // Skip handles that are in the in-memory cache but not in the
            // ledger (cache can drift briefly ahead of the ledger,
            // especially under seed_known_handles in tests). Increment
            // returns Err for those; we drop them silently rather than
            // failing the whole observation.
            // Resonance-gated familiarity: `mentions` advances on every
            // observed mention (raw recurrence), but the salience counter
            // `resonance` advances ONLY when the mention lands in a turn
            // that resonates with the thread anchor. The in-memory
            // `familiarize` returns that gate decision so the durable
            // `increment_resonance` is gated on the *same* cosine — one
            // compute, no divergence between the tiers.
            let mut entries: Vec<(ThreadHandle, u32, u32)> = Vec::with_capacity(mentioned.len());
            for handle in &mentioned {
                // Durable `mentions` first; this also filters cache-drift
                // handles (increment errors for handles not yet in the
                // ledger) — drop them silently rather than fail the turn.
                let Ok(mentions_after) = self.store.increment_mentions(handle) else {
                    continue;
                };
                // Mirror into the in-memory score tier and read back
                // whether the mention resonated. Best-effort: a failure
                // here just means resonance holds this turn.
                let resonant = if let Some(attn) = &self.attention {
                    match attn.familiarize(scope, handle).await {
                        Ok(r) => r,
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                handle = %handle,
                                "persistent-attention: familiarize failed on known-handle mention"
                            );
                            false
                        }
                    }
                } else {
                    false
                };
                let resonance_after = if resonant {
                    self.store.increment_resonance(handle).unwrap_or(0)
                } else {
                    self.store.current_resonance(handle).unwrap_or(0)
                };
                entries.push((handle.clone(), mentions_after, resonance_after));
            }
            if !entries.is_empty() {
                self.store.record_familiarity_batch(entries, turn_seq)?;
            }
        }

        // (c) candidate scan — kebab-case phrases recurring in the
        // turn that aren't known yet. `counts` carries the raw counts
        // so we can route between the "propose only" and "auto-promote"
        // arms without a second regex pass.
        let counts = count_unknown_kebab_phrases(turn_text, &known_snapshot);

        let mut proposed_stubs: Vec<ProposedThreadStub> = Vec::new();
        let mut promoted_handles: Vec<ThreadHandle> = Vec::new();
        let mut proposed_persisted: usize = 0;

        for (phrase, count, first_start, first_end) in counts {
            if count < STUB_MIN_OCCURRENCES {
                continue;
            }
            let snippet = token_window(turn_text, first_start, first_end, CONTEXT_SNIPPET_TOKENS);
            let confidence = stub_confidence(count);
            let stub = ProposedThreadStub {
                handle_guess: phrase.clone(),
                context_snippet: snippet,
                confidence,
            };

            // Persistence requires an originating chunk for the audit
            // trail. Without one, fall through to "candidate only" so
            // unit tests can still exercise the detection logic.
            let Some(anchor_chunk_id) = originating_chunk_id else {
                proposed_stubs.push(stub);
                continue;
            };

            let qualifies_for_promotion = count >= PROMOTE_MIN_OCCURRENCES
                && self.promotions_this_session.load(Ordering::Relaxed) < PROMOTION_CAP_PER_SESSION;
            let promoted_to_handle = if qualifies_for_promotion {
                ThreadHandle::new(phrase.clone()).ok().filter(|h| {
                    // Derived stop-handle gate (secondary defense): never
                    // re-promote a token an existing thread already shows
                    // is ubiquitous-but-unresonant (high mentions, near-zero
                    // resonance). The resonance-driven floor is the primary
                    // collapse; this stops a known stopword re-forming as a
                    // fresh Active thread once its stats exist.
                    match self.store.get_thread(h) {
                        Ok(Some(rec)) => !crate::is_stop_handle(rec.mentions, rec.resonance),
                        _ => true,
                    }
                })
            } else {
                None
            };

            // Persist a `threads_proposed` row in both arms — keeps a
            // clean audit trail of every kebab-shape the substrate
            // noticed, with `promoted_to` set IFF we also promoted.
            let proposed_record = ProposedThreadRecord {
                id: 0,
                proposed_handle: phrase.clone(),
                chunk_ids: vec![anchor_chunk_id.to_string()],
                centroid_vec: Vec::new(),
                cohesion: confidence,
                created_at: Utc::now(),
                promoted_to: promoted_to_handle.as_ref().map(|h| h.as_str().to_string()),
            };
            match self.store.insert_proposed_thread(&proposed_record) {
                Ok(_) => proposed_persisted += 1,
                Err(StoreError::UniqueViolation { .. }) => {
                    // Phrase was already proposed (earlier observation
                    // this session or a previous boot). If we're
                    // promoting now, repoint the existing row.
                    if let Some(target) = &promoted_to_handle {
                        let _ = self.store.mark_proposed_thread_promoted(&phrase, target);
                    }
                }
                Err(e) => return Err(e.into()),
            }

            if let Some(handle) = promoted_to_handle {
                // Promotion: upsert the threads row with the
                // originating chunk as anchor, then bump familiarity to
                // 1 so the InMemoryAttention sees a non-zero floor when
                // chain replay materializes the thread. Cache the
                // handle so subsequent chunks in this batch route
                // through the familiarity path, not stub detection.
                let now = Utc::now();
                let thread_record = ThreadRecord {
                    handle: handle.clone(),
                    tension: TensionState::Slack,
                    mentions: 0,
                    resonance: 0,
                    last_touched_at: now,
                    anchor_chunk_id: Some(anchor_chunk_id.to_string()),
                    fold_override: None,
                    created_at: now,
                    created_scope_key: None,
                    privacy_tier: scope.privacy_tier,
                };
                self.store.upsert_thread(&thread_record)?;
                // Seed mentions=1 AND resonance=1 on mint: a promoted
                // thread was distilled from the current resonant turn, so
                // it earns a non-baseline floor immediately (it must keep
                // resonating to climb further). Both counters round-trip
                // through the same FamiliarityBatch row.
                let mentions_after = self.store.increment_mentions(&handle).unwrap_or(0);
                let resonance_after = self.store.increment_resonance(&handle).unwrap_or(0);
                self.store.record_familiarity_batch(
                    vec![(handle.clone(), mentions_after, resonance_after)],
                    turn_seq,
                )?;
                // Mirror the durable familiarity bump into the in-memory
                // score tier so the curator's next tick sees a non-zero
                // score and doesn't demote a freshly-promoted thread to
                // Dormant. The scope's attention_vec was already populated
                // by the hoisted `attend` at the top of `observe`, so the
                // resonance term will see real signal here. Best-effort.
                if let Some(attn) = &self.attention {
                    if let Err(err) = attn.familiarize(scope, &handle).await {
                        tracing::warn!(
                            error = %err,
                            handle = %handle,
                            "persistent-attention: familiarize failed on auto-promotion"
                        );
                    }
                }
                self.known_handles.write().await.insert(handle.clone());
                self.promotions_this_session.fetch_add(1, Ordering::Relaxed);
                promoted_handles.push(handle.clone());
                if !mentioned.iter().any(|h| *h == handle) {
                    mentioned.push(handle);
                }
            } else {
                proposed_stubs.push(stub);
            }
        }

        // (d) concept growth — grow the *concept* graph from this turn's
        // resonance-gated co-mentions (edges) and salient recurring unknown
        // terms (nodes). Gated on both a corpus (anchor codebook) and an
        // attention store (rolling vector); best-effort, never aborts the turn.
        let (concept_edges_minted, concept_nodes_minted) = match (&self.corpus, &self.attention) {
            (Some(corpus), Some(attn)) => match attn.scope_vector(scope).await {
                Ok(Some(rolling_vec)) if !rolling_vec.is_empty() => {
                    self.grow_concepts(scope, turn_text, turn_seq, &rolling_vec, corpus)
                        .await
                }
                Ok(_) => (0, 0),
                Err(err) => {
                    tracing::warn!(error = %err, "concept-growth: scope_vector failed");
                    (0, 0)
                }
            },
            _ => (0, 0),
        };

        // Stable order: highest confidence first, then alphabetical.
        proposed_stubs.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.handle_guess.cmp(&b.handle_guess))
        });
        promoted_handles.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        mentioned.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        mentioned.dedup();

        Ok(ObservationResult {
            membrane_chunks_ingested,
            familiarity_increments: mentioned,
            proposed_stubs,
            promoted_handles,
            proposed_persisted,
            concept_edges_minted,
            concept_nodes_minted,
        })
    }

    /// Subscribe to the pipeline's post-ingest broadcast and observe every
    /// chunk that lands in the corpus. The chunk text is fetched from
    /// `corpus` by id (the broadcast carries ids only). Each chunk is
    /// treated as one observation — handle mentions advance familiarity,
    /// unknown handle candidates surface as proposed stubs.
    ///
    /// `scope_default` is the scope used when the event has no project
    /// (corpus-wide ingest); event-provided projects override it.
    ///
    /// Runs until `cancel` fires or the pipeline channel closes. Per-event
    /// errors are logged at `warn` and the loop continues — daemon survival
    /// is more valuable than per-event strict failure.
    pub async fn run_subscribed(
        &self,
        corpus: Arc<CorpusStore>,
        scope_default: AttentionScope,
        cancel: CancellationToken,
    ) -> Result<(), ObserverError> {
        let seq = AtomicU64::new(0);
        let mut rx = self.pipeline.subscribe_ingest();
        loop {
            tokio::select! {
                biased;
                res = rx.recv() => {
                    match res {
                        Ok(event) => {
                            // Live cognition fires ONLY on a TurnEnd: a
                            // watched conversation-transcript ingest. Bulk
                            // scans (replaying history through the live
                            // pipeline) and synthetic self-writes (membrane
                            // feedback) are skipped.
                            if !event.is_turn_end() {
                                continue;
                            }
                            let scope = AttentionScope {
                                project: event
                                    .project
                                    .clone()
                                    .or_else(|| scope_default.project.clone()),
                                session_id: scope_default.session_id.clone(),
                                agent: scope_default.agent.clone(),
                                privacy_tier: scope_default.privacy_tier,
                            };
                            // Refresh in case threads have been added since
                            // the daemon started. Cheap (read all rows; bounded).
                            if let Err(err) = self.refresh_known_handles().await {
                                tracing::warn!(error = %err, "turn-observer: refresh failed");
                                continue;
                            }
                            let texts = match corpus
                                .fetch_texts(&event.chunk_ids_upserted)
                                .await
                            {
                                Ok(t) => t,
                                Err(err) => {
                                    tracing::warn!(error = %err, "turn-observer: fetch_texts failed");
                                    continue;
                                }
                            };
                            for (chunk_id, text) in texts {
                                let s = seq.fetch_add(1, Ordering::Relaxed);
                                let session = format!("ambient:{chunk_id}");
                                if let Err(err) = self
                                    .observe(&scope, &text, s, &session, Some(chunk_id.as_str()))
                                    .await
                                {
                                    tracing::warn!(
                                        error = %err,
                                        chunk = %chunk_id,
                                        "turn-observer: observe failed",
                                    );
                                }
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(skipped = n, "turn-observer: ingest channel lagged");
                        }
                    }
                }
                () = cancel.cancelled() => {
                    // Drain queued events before exiting so the shutdown
                    // handshake doesn't race the broadcast. Best-effort
                    // observe; per-event errors stay non-fatal.
                    while let Ok(event) = rx.try_recv() {
                        if !event.is_turn_end() {
                            continue;
                        }
                        let scope = AttentionScope {
                            project: event
                                .project
                                .clone()
                                .or_else(|| scope_default.project.clone()),
                            session_id: scope_default.session_id.clone(),
                            agent: scope_default.agent.clone(),
                            privacy_tier: scope_default.privacy_tier,
                        };
                        if self.refresh_known_handles().await.is_err() {
                            break;
                        }
                        let Ok(texts) = corpus.fetch_texts(&event.chunk_ids_upserted).await
                        else {
                            continue;
                        };
                        for (chunk_id, text) in texts {
                            let s = seq.fetch_add(1, Ordering::Relaxed);
                            let session = format!("ambient:{chunk_id}");
                            let _ = self
                                .observe(&scope, &text, s, &session, Some(chunk_id.as_str()))
                                .await;
                        }
                    }
                    return Ok(());
                }
            }
        }
    }
}

// --- Phase 2: ambient concept growth ---------------------------------

impl TurnObserver {
    /// Grow the concept graph from one turn. Returns `(edges_minted,
    /// nodes_minted)` — newly-created edges/nodes only (idempotent re-touch
    /// does not count). Best-effort throughout: a store or codebook error is
    /// logged and skipped, never propagated, so the durable observation that
    /// already happened is preserved. Two halves, both gated on the SAME
    /// per-turn resonance decision (the occurrence≠salience law):
    ///
    /// - **edges** — existing concepts literally named in the turn (gazetteer)
    ///   whose anchor resonates with the turn get pairwise `co_occurs` edges;
    /// - **nodes** — unknown terms recurring across resonant turns mint a
    ///   `Proposed` node connected to its co-resonant context.
    async fn grow_concepts(
        &self,
        scope: &AttentionScope,
        turn_text: &str,
        turn_seq: u64,
        rolling_vec: &[f32],
        corpus: &Arc<CorpusStore>,
    ) -> (usize, usize) {
        let cfg = self.concept_growth;

        // Maybe-rebuild the (project-agnostic) anchor codebook on the turn
        // cadence. Build OUTSIDE the cache write lock (the embedding fetch is
        // async) and swap it in under the lock — never hold the guard across
        // `.await`. A rebuild also clears the per-project gazetteers so they
        // refresh with any concepts added since.
        let prev_turns = self.turns_since_build.fetch_add(1, Ordering::Relaxed);
        let need_anchor_build = {
            let g = self.growth_cache.read().await;
            !g.anchors_built || prev_turns >= cfg.codebook_rebuild_turns
        };
        if need_anchor_build {
            let anchors = Self::build_anchor_map(&self.store, corpus).await;
            {
                let mut g = self.growth_cache.write().await;
                g.anchors = anchors;
                g.anchors_built = true;
                g.gazetteers.clear();
            }
            self.turns_since_build.store(0, Ordering::Relaxed);
        }

        // Ensure a gazetteer for THIS turn's project scope. Keyed by
        // `scope.project` so an observer streaming turns from different projects
        // never reuses the wrong project's matcher/`known` oracle. Built lazily
        // (sync — no `.await`); insert under the write lock if absent.
        let proj_key = scope.project.clone();
        {
            let have = {
                let g = self.growth_cache.read().await;
                g.gazetteers.contains_key(&proj_key)
            };
            if !have {
                let gaz =
                    concept_growth::build_ambient_gazetteer(&self.store, scope.project.as_deref());
                let mut g = self.growth_cache.write().await;
                g.gazetteers.entry(proj_key.clone()).or_insert(gaz);
            }
        }

        // Read-side work under the guard, captured into owned data so the store
        // writes below run without holding the lock.
        let body_norm = concept_growth::normalize(turn_text);
        let known_threads: HashSet<ThreadHandle> = self.known_handles.read().await.clone();
        let raw_candidates = raw_unknown_kebab(turn_text, &known_threads);
        let (survivors, fresh_candidates): (Vec<(i64, f32, String)>, Vec<String>) = {
            let g = self.growth_cache.read().await;
            // The active project's gazetteer was just ensured above; if a
            // concurrent rebuild cleared it, skip this turn (best-effort).
            let Some(gaz) = g.gazetteers.get(&proj_key) else {
                return (0, 0);
            };
            let matched = gaz.matches(&body_norm);
            let survivors = concept_growth::select_survivors(
                &matched,
                &g.anchors,
                rolling_vec,
                cfg.resonance_floor,
                cfg.edge_top_k,
            )
            .into_iter()
            .filter_map(|(id, res)| gaz.handles.get(&id).map(|h| (id, res, h.clone())))
            .collect();
            // Exclude any raw candidate that is already a known concept form
            // (the gazetteer `known` oracle) so existing concepts never
            // re-materialize as fresh nodes.
            let fresh = raw_candidates
                .into_iter()
                .filter(|p| !gaz.known.contains(&concept_growth::normalize(p)))
                .collect();
            (survivors, fresh)
        };

        // A turn is "resonant" iff it produced enough co-resonant existing
        // concepts — the same decision the edge half uses, reused verbatim by
        // the node-recurrence accumulator (one compute, no divergence).
        let resonant_turn = survivors.len() >= cfg.min_survivors;

        // --- edge half: pairwise co_occurs among the survivors -----------
        let mut edges_minted = 0usize;
        if resonant_turn {
            for (i, a) in survivors.iter().enumerate() {
                for b in survivors.iter().skip(i + 1) {
                    // Deterministic min→max id order so A,B and B,A collapse
                    // under the directional UNIQUE(from,relation,to) constraint.
                    let (lo, hi) = if a.0 <= b.0 { (a, b) } else { (b, a) };
                    let evidence = serde_json::json!({
                        "via": "ambient-observe",
                        "resonance_from": lo.1,
                        "resonance_to": hi.1,
                        "turn_seq": turn_seq,
                    })
                    .to_string();
                    if self.mint_co_occurs(lo.0, &lo.2, hi.0, &hi.2, &evidence) {
                        edges_minted += 1;
                    }
                }
            }
        }

        // --- node half: salient recurring unknown terms ------------------
        let mut nodes_minted = 0usize;
        if !fresh_candidates.is_empty() {
            let mut ready: Vec<String> = Vec::new();
            {
                let mut rec = self.term_recurrence.write().await;
                for cand in fresh_candidates {
                    let entry = rec.entry(cand.clone()).or_default();
                    // `mentions` advances every turn the term appears; salience
                    // counter `resonance` advances ONLY on resonant turns.
                    entry.mentions = entry.mentions.saturating_add(1);
                    if resonant_turn {
                        entry.resonance = entry.resonance.saturating_add(1);
                        for (id, _res, handle) in &survivors {
                            entry.co_resonant.insert(*id, handle.clone());
                        }
                    }
                    if entry.resonance >= cfg.node_mint_min_resonant_turns {
                        ready.push(cand);
                    }
                }
            }
            for cand in ready {
                if self.node_mints_this_session.load(Ordering::Relaxed)
                    >= cfg.node_mint_cap_per_session
                {
                    break; // session mint cap reached
                }
                let co = {
                    let rec = self.term_recurrence.read().await;
                    rec.get(&cand)
                        .map(|e| e.co_resonant.clone())
                        .unwrap_or_default()
                };
                if self.mint_node(&cand, &co) {
                    nodes_minted += 1;
                    self.node_mints_this_session.fetch_add(1, Ordering::Relaxed);
                    self.term_recurrence.write().await.remove(&cand);
                }
            }
        }

        (edges_minted, nodes_minted)
    }

    /// Mint (or re-touch) a single `co_occurs` edge between two concept ids in
    /// canonical `lo→hi` order. Returns `true` iff the edge was newly created
    /// (so the caller's count reflects mints, not re-touches). The
    /// creation-gated `ConceptConnected` audit uses the SAME from/to order as
    /// the inserted row — audit-only for cross-scope edges (the by-id row is
    /// authoritative; the event's single `project` is cosmetic across scopes).
    fn mint_co_occurs(&self, lo: i64, lo_h: &str, hi: i64, hi_h: &str, evidence: &str) -> bool {
        match self.store.add_concept_edge_by_id(
            lo,
            CO_MENTION_RELATION,
            hi,
            OBSERVED_MENTION_CONFIDENCE,
            EdgeSource::Observed,
            Some(AMBIENT_EDGE_BY),
            Some(evidence),
        ) {
            Ok((_, created)) => {
                if created {
                    if let Err(err) = self.store.record_concept_connected(
                        GLOBAL_PROJECT,
                        lo_h,
                        CO_MENTION_RELATION,
                        hi_h,
                        EdgeSource::Observed,
                        Some(AMBIENT_EDGE_BY),
                    ) {
                        tracing::warn!(error = %err, "concept-growth: ConceptConnected emit failed");
                    }
                }
                created
            }
            Err(err) => {
                tracing::warn!(error = %err, "concept-growth: add_concept_edge_by_id failed");
                false
            }
        }
    }

    /// Mint a new `Proposed` global concept node from a salient recurring term
    /// and connect it to its co-resonant context via `co_occurs` edges. Returns
    /// `true` iff a new node was created. Mirrors slice-5 prose promotion:
    /// race-guard, born `Candidate` → promoted `Proposed` in-pass, creation-
    /// gated audit. `kind = None` — ambient turns carry no `entity_type`.
    fn mint_node(&self, candidate: &str, co_resonant: &BTreeMap<i64, String>) -> bool {
        let Some(slug) = slugify(candidate) else {
            return false;
        };
        // Race guard: if the term resolved to a concept since it was staged,
        // skip (no duplicate, no status downgrade).
        if matches!(
            self.store.resolve_concept(GLOBAL_PROJECT, &slug),
            Ok(Some(_))
        ) {
            return false;
        }
        let (rec, created) = match self.store.ensure_typed_concept(
            GLOBAL_PROJECT,
            &slug,
            ConceptStatus::Candidate,
            None,
        ) {
            Ok(v) => v,
            Err(err) => {
                tracing::warn!(error = %err, "concept-growth: ensure_typed_concept failed");
                return false;
            }
        };
        if !created {
            return false; // lost a race; leave the existing concept alone
        }
        // Promote in-pass so it joins the diffusible graph this session.
        if let Err(err) =
            self.store
                .set_concept_status(GLOBAL_PROJECT, &slug, ConceptStatus::Proposed, Some(0.4))
        {
            tracing::warn!(error = %err, "concept-growth: set_concept_status failed");
        }
        if let Err(err) = self
            .store
            .record_concept_promoted(GLOBAL_PROJECT, &slug, "proposed")
        {
            tracing::warn!(error = %err, "concept-growth: ConceptPromoted emit failed");
        }
        // Connect to the co-resonant context that birthed it.
        for (nbr_id, nbr_h) in co_resonant {
            if *nbr_id == rec.id {
                continue;
            }
            let ((lo, lo_h), (hi, hi_h)) = if rec.id <= *nbr_id {
                ((rec.id, slug.as_str()), (*nbr_id, nbr_h.as_str()))
            } else {
                ((*nbr_id, nbr_h.as_str()), (rec.id, slug.as_str()))
            };
            let evidence =
                serde_json::json!({ "via": "ambient-observe-node", "term": slug }).to_string();
            self.mint_co_occurs(lo, lo_h, hi, hi_h, &evidence);
        }
        true
    }

    /// Build the concept anchor codebook map (`concept_id` → anchor embedding)
    /// from the ledger's `concept_anchors` + the corpus embeddings of their
    /// anchor chunks. Mirrors `query::ConceptCodebook::build` but uses only
    /// store + corpus primitives (no `query` dependency). Best-effort: any
    /// error yields an empty map (concept growth degrades to a no-op).
    async fn build_anchor_map(store: &ThreadsDb, corpus: &CorpusStore) -> HashMap<i64, Vec<f32>> {
        let anchors = match store.concept_anchors() {
            Ok(a) => a,
            Err(err) => {
                tracing::warn!(error = %err, "concept-growth: concept_anchors failed");
                return HashMap::new();
            }
        };
        if anchors.is_empty() {
            return HashMap::new();
        }
        let ids: Vec<String> = anchors.iter().map(|a| a.chunk_id.clone()).collect();
        let embs = match corpus.fetch_embeddings(&ids).await {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, "concept-growth: fetch_embeddings failed");
                return HashMap::new();
            }
        };
        let mut by_id = HashMap::with_capacity(anchors.len());
        for a in anchors {
            if let Some(sem) = embs.get(&a.chunk_id).cloned() {
                by_id.insert(a.concept_id, sem);
            }
        }
        by_id
    }
}

/// Distinct unknown kebab-case candidate terms PRESENT in the turn — counted
/// once per turn, NOT the intra-turn `>= STUB_MIN_OCCURRENCES` gate the thread
/// path uses. A term appearing even once must register the turn, else "recurs
/// across N resonant turns" could never mint. Excludes known thread handles;
/// the caller additionally excludes known concept forms via the gazetteer oracle.
fn raw_unknown_kebab(turn_text: &str, known_threads: &HashSet<ThreadHandle>) -> Vec<String> {
    let lowered = turn_text.to_lowercase();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for caps in kebab_phrase_regex().captures_iter(&lowered) {
        let Some(m) = caps.get(1) else { continue };
        let phrase = m.as_str().to_string();
        if ThreadHandle::new(phrase.clone()).is_err() {
            continue;
        }
        if known_threads.iter().any(|h| h.as_str() == phrase) {
            continue;
        }
        seen.insert(phrase);
    }
    seen.into_iter().collect()
}

/// Build a default `AttentionScope` for ambient observation.
///
/// Used by `TurnObserver::run_subscribed` when callers don't supply
/// project-specific scopes. Privacy tier defaults to `T1Project` —
/// ambient pickup does not write to `T0Private` scopes.
#[must_use]
pub fn ambient_scope_default() -> AttentionScope {
    AttentionScope {
        project: None,
        session_id: Some("ambient".into()),
        agent: Some("substrate".into()),
        privacy_tier: PrivacyTier::T1Project,
    }
}

// --- detection primitives --------------------------------------------

/// Regex source for canonical recognition triggers. Centralized so
/// tests can match against the same shape the runtime does.
///
/// The trigger families are:
/// - `yes, exactly`
/// - `that's the move` / `that is the move`
/// - `oh, like the X thing` / `oh — like X` / `oh it's like X`
/// - `i see what you mean`
/// - bare `exactly`
/// - `got it`
/// - `right, X`
///
/// Word-boundary anchored so `yes` alone or `not exactly` don't fire.
const RECOGNITION_PATTERN: &str = r"(?ix)
    (?:^|[^\w])                                           # left boundary (start or non-word)
    (?:
        yes\s*,?\s*exactly                              | # 'yes exactly' / 'yes, exactly'
        that(?:'s|\s+is)\s+the\s+move                   | # that's the move
        oh\s*[,\-—]?\s*(?:like|it'?s\s+like)\s+\w  | # oh, like X / oh it's like X
        i\s+see\s+what\s+you\s+mean                     |
        exactly                                         | # bare exactly (caller-side de-dupes contradicts)
        got\s+it                                        |
        right\s*,\s*\w                                    # right, X (forces a follower word)
    )
";

/// Compiled lazily — `OnceLock` keeps the cost off the hot path and out
/// of every struct-construction. Once observed, the regex is shared
/// across observers (cheap; `Regex` is `Send + Sync`).
fn recognition_regex() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(RECOGNITION_PATTERN)
            .expect("recognition regex is a const literal and must compile")
    })
}

/// "negation precedes 'exactly'" guard — we don't want `no, exactly not`
/// to trip the bare `exactly` arm. The recognition regex matches inside
/// `no, exactly not`, so post-filter trigger matches whose immediate
/// left context is a negation word.
fn is_negated(turn_text: &str, match_start: usize) -> bool {
    let prefix = &turn_text[..match_start];
    let tail: String = prefix
        .chars()
        .rev()
        .take(16)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let lower = tail.to_lowercase();
    // Common negations directly preceding the trigger.
    lower.trim_end().ends_with("no,")
        || lower.trim_end().ends_with("not")
        || lower.trim_end().ends_with("no")
}

/// Information about one recognition match the observer found.
#[derive(Debug, Clone)]
struct RecognitionTrigger {
    /// Byte offset of the matched trigger in `turn_text`.
    start: usize,
    end: usize,
    /// The matched text (lowercased trigger phrase, leading boundary
    /// char trimmed).
    text: String,
    /// Confidence heuristic in `[0, 1]`. v1 = 0.75 for every match;
    /// per-trigger weighting is a Phase-9 follow-up.
    confidence: f32,
}

fn detect_recognition_triggers(turn_text: &str) -> Vec<RecognitionTrigger> {
    let mut out = Vec::new();
    for m in recognition_regex().find_iter(turn_text) {
        // The pattern grabs an optional leading boundary char; skip it
        // so `text` is the actual trigger.
        let trimmed_start = turn_text[m.start()..m.end()]
            .char_indices()
            .find(|(_, c)| c.is_alphanumeric())
            .map_or_else(|| m.start(), |(i, _)| m.start() + i);
        let raw = turn_text[trimmed_start..m.end()].to_string();
        if is_negated(turn_text, trimmed_start) {
            continue;
        }
        out.push(RecognitionTrigger {
            start: trimmed_start,
            end: m.end(),
            text: raw.to_lowercase(),
            confidence: 0.75,
        });
    }
    out
}

/// Slice ~`token_budget` whitespace-separated tokens centered on
/// `[match_start, match_end]`. Falls back to the whole string when the
/// turn is shorter than the budget.
fn token_window(text: &str, match_start: usize, match_end: usize, token_budget: usize) -> String {
    let tokens: Vec<(usize, &str)> = text
        .split_whitespace()
        .scan(0usize, |cursor, tok| {
            // Find each token's byte offset from cursor, advancing.
            text[*cursor..].find(tok).map(|off| {
                let start = *cursor + off;
                *cursor = start + tok.len();
                (start, tok)
            })
        })
        .collect();

    if tokens.len() <= token_budget {
        return text.to_string();
    }

    // Find the token whose span overlaps the match.
    let center = tokens
        .iter()
        .position(|(off, t)| *off <= match_end && off + t.len() > match_start)
        .unwrap_or(tokens.len() / 2);

    let half = token_budget / 2;
    let lo = center.saturating_sub(half);
    let hi = (center + half + 1).min(tokens.len());

    let start_byte = tokens[lo].0;
    let last = tokens[hi - 1];
    let end_byte = last.0 + last.1.len();
    text[start_byte..end_byte].to_string()
}

fn build_membrane_chunks(
    turn_text: &str,
    triggers: &[RecognitionTrigger],
    session_id: &str,
    turn_seq: u64,
) -> Vec<Chunk> {
    triggers
        .iter()
        .enumerate()
        .map(|(idx, trig)| {
            let window = token_window(turn_text, trig.start, trig.end, MEMBRANE_CONTEXT_TOKENS);
            let source_id = format!("{session_id}:{turn_seq}");
            // Each match within a turn is its own chunk; the idx makes
            // `chunk_id` unique within the (session, turn) pair.
            // u32::try_from is bounded by triggers.len() in a single
            // turn — astronomically below u32::MAX, but be tidy.
            let chunk_index = u32::try_from(idx).unwrap_or(u32::MAX);
            // Synthetic (membrane) chunks bypass `[[sources]]` config; use
            // the reserved `synthetic:<kind>` discriminator per P0.
            let source_config_id =
                Chunk::synthetic_source_config_id(ostk_recall_core::SourceKind::Membrane);
            let chunk_id =
                Chunk::make_id(Source::Membrane, &source_id, chunk_index, &source_config_id);
            let sha = Chunk::content_hash(&window);
            let extra = serde_json::json!({
                "kind": "recognition",
                "trigger": trig.text,
                "confidence": trig.confidence,
                "turn_seq": turn_seq,
                "session_id": session_id,
            });
            Chunk {
                chunk_id,
                source: Source::Membrane,
                project: None, // pipeline ingest_synthetic carries project via meta
                source_id,
                source_config_id,
                facets: Default::default(),
                embedding_input_sha256: String::new(),
                chunk_index,
                ts: Some(Utc::now()),
                role: Some("recognition".into()),
                text: window,
                sha256: sha,
                links: Links::default(),
                extra,
            }
        })
        .collect()
}

/// Find each known handle that appears in `turn_text` as a kebab-case
/// word (bounded by non-`[a-z0-9-]` chars). Returns one entry per
/// distinct handle, regardless of occurrence count — per the spec the
/// unit is "turns containing this handle," not occurrences.
fn handles_mentioned(turn_text: &str, known: &HashSet<ThreadHandle>) -> Vec<ThreadHandle> {
    let mut seen: HashSet<ThreadHandle> = HashSet::new();
    for handle in known {
        if find_handle_in_text(turn_text, handle.as_str()) {
            seen.insert(handle.clone());
        }
    }
    let mut out: Vec<ThreadHandle> = seen.into_iter().collect();
    // Stable order for deterministic tests / chain rows.
    out.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    out
}

/// Word-boundary check for kebab-case handles. The default regex `\b`
/// treats `-` as a word boundary, which would false-positive on
/// `not-a-real` vs `real`. So we hand-roll the boundary: char before
/// and char after must NOT be `[a-z0-9-]`.
fn find_handle_in_text(text: &str, needle: &str) -> bool {
    let bytes = text.as_bytes();
    let n_bytes = needle.as_bytes();
    if n_bytes.is_empty() || n_bytes.len() > bytes.len() {
        return false;
    }
    let mut i = 0usize;
    while i + n_bytes.len() <= bytes.len() {
        if &bytes[i..i + n_bytes.len()] == n_bytes {
            let left_ok = i == 0 || !is_handle_char(bytes[i - 1]);
            let right_idx = i + n_bytes.len();
            let right_ok = right_idx == bytes.len() || !is_handle_char(bytes[right_idx]);
            if left_ok && right_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

const fn is_handle_char(b: u8) -> bool {
    b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'
}

/// Compiled lazily; same precedent as `recognition_regex`.
fn kebab_phrase_regex() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // 2..=4 lowercase-word kebab phrases, bounded by non-handle chars.
        // The leading `(?:^|[^a-z0-9-])` and trailing `(?:[^a-z0-9-]|$)`
        // ensure we don't match the middle of a longer kebab token.
        Regex::new(r"(?:^|[^a-z0-9-])([a-z]+(?:-[a-z]+){1,3})(?:[^a-z0-9-]|$)")
            .expect("kebab phrase regex literal must compile")
    })
}

/// Scan `turn_text` for kebab-case phrases not already in `known`.
/// Returns `(phrase, count, first_match_start, first_match_end)` for
/// each candidate that passes handle-validation. The caller is
/// responsible for filtering by occurrence count and ordering.
fn count_unknown_kebab_phrases(
    turn_text: &str,
    known: &HashSet<ThreadHandle>,
) -> Vec<(String, usize, usize, usize)> {
    // Lower-case so phrases survive the regex regardless of the
    // surrounding text's casing; handles themselves are kebab-lowercase
    // by construction.
    let lowered = turn_text.to_lowercase();
    let mut counts: HashMap<String, (usize, usize, usize)> = HashMap::new();
    for caps in kebab_phrase_regex().captures_iter(&lowered) {
        let Some(m) = caps.get(1) else { continue };
        let phrase = m.as_str().to_string();
        if ThreadHandle::new(phrase.clone()).is_err() {
            continue;
        }
        if known.iter().any(|h| h.as_str() == phrase) {
            continue;
        }
        let entry = counts
            .entry(phrase)
            .or_insert_with(|| (0, m.start(), m.end()));
        entry.0 += 1;
    }
    let mut out: Vec<(String, usize, usize, usize)> = counts
        .into_iter()
        .map(|(phrase, (n, start, end))| (phrase, n, start, end))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[allow(clippy::cast_precision_loss)]
fn stub_confidence(occurrences: usize) -> f32 {
    let capped = occurrences.min(MAX_CONFIDENCE_OCCURRENCES) as f32;
    let max = MAX_CONFIDENCE_OCCURRENCES as f32;
    (capped / max).clamp(0.0, 1.0)
}

// `record_familiarity_batch` only emits a chain row; it doesn't update
// the score-tier counter (that runs through `AttentionForwardStore::
// familiarize`). The observer's contract is per the spec: emit the
// batch chain row. Score-tier wiring is the surfacer's job (Phase 10+).

// ===== tests =====

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use ostk_recall_core::attention::PrivacyTier;
    use ostk_recall_pipeline::ChunkEmbedder;
    use ostk_recall_store::{ChainEvent, ChainSink, CorpusStore, IngestDb};
    use std::sync::Mutex as StdMutex;
    use tempfile::TempDir;

    // ---- test harness ----

    /// Deterministic fake embedder — avoids spinning up `fastembed` for
    /// every test. Mirrors the shape of `pipeline::tests::FakeEmbedder`.
    struct FakeEmbedder {
        dim: usize,
    }

    impl ChunkEmbedder for FakeEmbedder {
        fn dim(&self) -> usize {
            self.dim
        }
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

    #[derive(Default)]
    struct RecordingSink {
        events: StdMutex<Vec<ChainEvent>>,
    }

    impl RecordingSink {
        fn take(&self) -> Vec<ChainEvent> {
            std::mem::take(&mut self.events.lock().unwrap())
        }
    }

    impl ChainSink for RecordingSink {
        fn append(&self, event: &ChainEvent) -> Result<(), StoreError> {
            self.events.lock().unwrap().push(event.clone());
            Ok(())
        }
    }

    async fn make_pipeline(dim: usize) -> (Arc<Pipeline>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let store = Arc::new(CorpusStore::open_or_create(tmp.path(), dim).await.unwrap());
        let ingest = Arc::new(IngestDb::open(tmp.path()).unwrap());
        let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });
        let p = Pipeline::new(store, ingest, emb);
        (Arc::new(p), tmp)
    }

    async fn make_observer() -> (TurnObserver, Arc<RecordingSink>, TempDir, TempDir) {
        let (pipeline, corpus_tmp) = make_pipeline(16).await;
        let store_tmp = TempDir::new().unwrap();
        let sink = Arc::new(RecordingSink::default());
        let store: Arc<dyn ChainSink> = sink.clone();
        let db = ThreadsDb::open_with_sink(store_tmp.path(), store).unwrap();
        let obs = TurnObserver::new(pipeline, Arc::new(db));
        (obs, sink, corpus_tmp, store_tmp)
    }

    fn scope() -> AttentionScope {
        AttentionScope {
            project: Some("haystack".into()),
            session_id: Some("sess-1".into()),
            agent: Some("claude".into()),
            privacy_tier: PrivacyTier::T1Project,
        }
    }

    fn handle(s: &str) -> ThreadHandle {
        ThreadHandle::new(s).unwrap()
    }

    // ---- (0) Phase-2 node candidate presence ----

    #[test]
    fn raw_unknown_kebab_counts_presence_and_excludes_known_threads() {
        let known: HashSet<ThreadHandle> = [handle("known-thread")].into_iter().collect();
        // A known thread handle is excluded; an unknown kebab term survives.
        let got = raw_unknown_kebab(
            "we touched known-thread and a new-bridge-idea today",
            &known,
        );
        assert_eq!(got, vec!["new-bridge-idea".to_string()]);
    }

    #[test]
    fn raw_unknown_kebab_registers_single_occurrence() {
        // Presence, not the intra-turn >=2 gate: one mention must register so
        // "once across N resonant turns" can mint.
        let got = raw_unknown_kebab("just one mention of fresh-term here", &HashSet::new());
        assert_eq!(got, vec!["fresh-term".to_string()]);
        // Non-kebab words (no hyphen) are not candidates.
        assert!(raw_unknown_kebab("alpha and beta plain words", &HashSet::new()).is_empty());
    }

    // ---- (1) recognition regex positive cases ----

    #[test]
    fn recognition_regex_matches_canonical_triggers() {
        let positives = [
            "yes, exactly that thing",
            "yes exactly the same shape",
            "ok — that's the move here",
            "right, that is the move",
            "oh, like the Hoberman thing comes back",
            "oh like a membrane situation",
            "oh — like the fold-depth one",
            "oh it's like the surfacer thing",
            "i see what you mean about decay",
            "exactly. that lines up.",
            "got it, will fold.",
            "right, that lines up",
        ];
        for s in positives {
            let trigs = detect_recognition_triggers(s);
            assert!(!trigs.is_empty(), "expected at least one trigger in: {s:?}");
        }
    }

    // ---- (2) recognition regex negative cases ----

    #[test]
    fn recognition_regex_rejects_non_triggers() {
        let negatives = [
            "yes",                           // bare yes
            "that move was good",            // wrong shape
            "no, exactly not the move",      // negation precedes
            "not exactly",                   // negation precedes
            "the right move is unclear",     // no comma + follower
            "they exactly captured nothing", // 'exactly' but in a clause
        ];
        // The "they exactly captured" case is the regex-vs-spec
        // boundary — bare `exactly` mid-sentence DOES match the regex.
        // The spec says the bare-`exactly` arm is allowed; we accept
        // that as a known false-positive class and only assert it
        // doesn't fire in the no/negation cases below.
        for s in &negatives[..5] {
            let trigs = detect_recognition_triggers(s);
            assert!(trigs.is_empty(), "unexpected trigger in {s:?}: {trigs:?}");
        }
    }

    // ---- (3) familiarity de-duplication ----

    #[tokio::test]
    async fn familiarity_per_distinct_handle_not_per_occurrence() {
        let (obs, sink, _c, _s) = make_observer().await;
        let h = handle("hoberman-thread-primitive");
        // Seed the ledger so refresh + observe sees it as known.
        obs.store
            .upsert_thread(&ostk_recall_store::ThreadRecord {
                handle: h.clone(),
                tension: ostk_recall_store::TensionState::Active,
                mentions: 0,
                resonance: 0,
                last_touched_at: Utc::now(),
                anchor_chunk_id: None,
                fold_override: None,
                created_at: Utc::now(),
                created_scope_key: None,
                privacy_tier: PrivacyTier::T1Project,
            })
            .unwrap();
        obs.refresh_known_handles().await.unwrap();
        let _ = sink.take(); // discard upsert chain event

        let turn = "We keep returning to hoberman-thread-primitive, \
            and hoberman-thread-primitive shows up again in this turn, \
            because hoberman-thread-primitive is everywhere.";
        let res = obs
            .observe(&scope(), turn, 5, "sess-1", None)
            .await
            .unwrap();
        assert_eq!(
            res.familiarity_increments.len(),
            1,
            "exactly one distinct handle ticked"
        );

        let ev = sink.take();
        let batch_count = ev
            .iter()
            .filter(|e| matches!(e, ChainEvent::FamiliarityBatch { .. }))
            .count();
        assert_eq!(batch_count, 1);
        if let Some(ChainEvent::FamiliarityBatch {
            entries, turn_seq, ..
        }) = ev
            .iter()
            .find(|e| matches!(e, ChainEvent::FamiliarityBatch { .. }))
        {
            assert_eq!(*turn_seq, 5);
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].0, h);
            assert_eq!(entries[0].1, 1, "one increment, not three");
        } else {
            panic!("expected a FamiliarityBatch event");
        }
    }

    // ---- (4) only known handles count ----

    #[tokio::test]
    async fn familiarity_only_for_known_handles() {
        let (obs, sink, _c, _s) = make_observer().await;
        // Empty known set on purpose.
        obs.refresh_known_handles().await.unwrap();
        let _ = sink.take();

        let turn = "We talked about not-a-real-handle and never mentioned it again.";
        let res = obs
            .observe(&scope(), turn, 1, "sess-1", None)
            .await
            .unwrap();
        assert!(
            res.familiarity_increments.is_empty(),
            "unknown handle must not increment familiarity"
        );
        let ev = sink.take();
        assert!(
            !ev.iter()
                .any(|e| matches!(e, ChainEvent::FamiliarityBatch { .. })),
            "no familiarity batch emitted for empty increments"
        );
    }

    // ---- (5) proposed-stub threshold ----

    #[tokio::test]
    async fn proposed_stub_threshold_two_occurrences() {
        let (obs, _sink, _c, _s) = make_observer().await;
        obs.refresh_known_handles().await.unwrap();

        // Single mention => no stub.
        let single = "I keep thinking about idle-curator-fade in passing.";
        let res = obs
            .observe(&scope(), single, 1, "sess-1", None)
            .await
            .unwrap();
        assert!(
            !res.proposed_stubs
                .iter()
                .any(|s| s.handle_guess == "idle-curator-fade"),
            "single occurrence should not propose a stub"
        );

        // Double mention => stub.
        let double = "We saw idle-curator-fade come up twice. \
            Then idle-curator-fade again, in the second sentence.";
        let res = obs
            .observe(&scope(), double, 2, "sess-1", None)
            .await
            .unwrap();
        assert!(
            res.proposed_stubs
                .iter()
                .any(|s| s.handle_guess == "idle-curator-fade"),
            "two occurrences should propose a stub: got {:?}",
            res.proposed_stubs
        );
    }

    // ---- (6) known handles aren't proposed as stubs ----

    #[tokio::test]
    async fn proposed_stub_excludes_known_handles() {
        let (obs, _sink, _c, _s) = make_observer().await;
        let h = handle("hoberman-thread-primitive");
        obs.seed_known_handles([h.clone()]).await;

        let turn = "Look at hoberman-thread-primitive and \
            hoberman-thread-primitive once more — it keeps showing up.";
        let res = obs
            .observe(&scope(), turn, 3, "sess-1", None)
            .await
            .unwrap();
        assert!(
            !res.proposed_stubs
                .iter()
                .any(|s| s.handle_guess == "hoberman-thread-primitive"),
            "known handle must NOT be proposed as a stub"
        );
        // It SHOULD be counted as a familiarity increment instead.
        assert_eq!(res.familiarity_increments, vec![h]);
    }

    // ---- (7) membrane chunk window size ----

    #[test]
    fn membrane_chunk_window_size() {
        // Build a long turn: 500 tokens of filler, "exactly" in the
        // middle, another 500 tokens of filler.
        let prefix = "alpha ".repeat(500);
        let suffix = " omega".repeat(500);
        let turn = format!("{prefix}exactly{suffix}");

        let triggers = detect_recognition_triggers(&turn);
        assert!(!triggers.is_empty());
        let chunks = build_membrane_chunks(&turn, &triggers, "sess-1", 9);
        assert_eq!(chunks.len(), triggers.len());

        for chunk in chunks {
            let token_count = chunk.text.split_whitespace().count();
            assert!(
                token_count <= MEMBRANE_CONTEXT_TOKENS + 4,
                "chunk text exceeds ~{MEMBRANE_CONTEXT_TOKENS} tokens: {token_count}"
            );
            assert!(
                chunk.text.to_lowercase().contains("exactly"),
                "window must contain the trigger; got: {}",
                &chunk.text[..chunk.text.len().min(120)]
            );
        }

        // And: short turn -> whole-turn fallback.
        let short = "yes, exactly the move";
        let triggers = detect_recognition_triggers(short);
        let chunks = build_membrane_chunks(short, &triggers, "sess-1", 1);
        assert_eq!(chunks[0].text, short);
    }

    // ---- (8) full dispatch ----

    #[tokio::test]
    async fn observe_dispatches_three_outputs() {
        let (obs, sink, _c, _s) = make_observer().await;
        let h1 = handle("hoberman-thread-primitive");
        let h2 = handle("fade-is-concentration");
        for h in [&h1, &h2] {
            obs.store
                .upsert_thread(&ostk_recall_store::ThreadRecord {
                    handle: h.clone(),
                    tension: ostk_recall_store::TensionState::Active,
                    mentions: 0,
                    resonance: 0,
                    last_touched_at: Utc::now(),
                    anchor_chunk_id: None,
                    fold_override: None,
                    created_at: Utc::now(),
                    created_scope_key: None,
                    privacy_tier: PrivacyTier::T1Project,
                })
                .unwrap();
        }
        obs.refresh_known_handles().await.unwrap();
        let _ = sink.take();

        let turn = "yes, exactly — hoberman-thread-primitive and \
            fade-is-concentration tie together. \
            That also points at attention-substrate-mvp, \
            and attention-substrate-mvp is worth a thread.";

        let res = obs
            .observe(&scope(), turn, 11, "sess-1", None)
            .await
            .unwrap();

        assert!(
            res.membrane_chunks_ingested >= 1,
            "expected at least one membrane chunk; got {}",
            res.membrane_chunks_ingested
        );
        let mut fam: Vec<String> = res
            .familiarity_increments
            .iter()
            .map(|h| h.as_str().to_string())
            .collect();
        fam.sort();
        assert_eq!(
            fam,
            vec![
                "fade-is-concentration".to_string(),
                "hoberman-thread-primitive".to_string(),
            ]
        );
        assert!(
            res.proposed_stubs
                .iter()
                .any(|s| s.handle_guess == "attention-substrate-mvp"),
            "expected attention-substrate-mvp as a stub, got: {:?}",
            res.proposed_stubs
        );
    }

    // ---- (9) promotion at PROMOTE_MIN_OCCURRENCES with anchor ----

    #[tokio::test]
    async fn promotes_phrase_at_three_occurrences_with_anchor() {
        let (obs, sink, _c, _s) = make_observer().await;
        let _ = sink.take();

        let chunk_id = "membrane:sess-1:42:0";
        // Three within-chunk mentions of the same kebab phrase.
        let turn = "we keep returning to fade-is-concentration, \
            and fade-is-concentration is the move here, \
            because fade-is-concentration explains the curator.";

        let res = obs
            .observe(&scope(), turn, 7, "sess-1", Some(chunk_id))
            .await
            .unwrap();

        // The promoted handle appears in res.promoted_handles and NOT
        // in res.proposed_stubs (it cleared the gate).
        assert_eq!(
            res.promoted_handles
                .iter()
                .map(|h| h.as_str().to_string())
                .collect::<Vec<_>>(),
            vec!["fade-is-concentration".to_string()],
        );
        assert!(
            !res.proposed_stubs
                .iter()
                .any(|s| s.handle_guess == "fade-is-concentration"),
            "promoted phrase must not appear in proposed_stubs"
        );
        assert!(
            res.proposed_persisted >= 1,
            "at least one proposed row written"
        );

        // The threads row is Slack with the originating chunk as anchor.
        let row = obs
            .store
            .get_thread(&handle("fade-is-concentration"))
            .unwrap()
            .expect("promoted thread must exist in ledger");
        assert_eq!(row.tension, TensionState::Slack);
        assert_eq!(row.anchor_chunk_id.as_deref(), Some(chunk_id));

        // The threads_proposed row carries promoted_to.
        let props = obs.store.list_proposed_threads().unwrap();
        let p = props
            .iter()
            .find(|p| p.proposed_handle == "fade-is-concentration")
            .expect("proposed-thread row must exist for audit trail");
        assert_eq!(
            p.promoted_to.as_deref(),
            Some("fade-is-concentration"),
            "promoted_to must point at the new thread for provenance"
        );

        // The chain log records the ThreadCreate + a FamiliarityBatch
        // (familiarity bumped on promote so the in-memory floor is
        // non-zero on next chain replay).
        let events = sink.take();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ChainEvent::ThreadCreate { handle, .. }
                    if handle.as_str() == "fade-is-concentration")),
            "ThreadCreate must chain for the promoted handle"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ChainEvent::FamiliarityBatch { entries, .. }
                    if entries.iter().any(|(h, _, _)| h.as_str() == "fade-is-concentration"))),
            "FamiliarityBatch must include the promoted handle"
        );
    }

    // ---- (10) below-threshold mentions stay as proposed stubs ----

    #[tokio::test]
    async fn two_mentions_propose_but_do_not_promote() {
        let (obs, _sink, _c, _s) = make_observer().await;
        let chunk_id = "membrane:sess-1:1:0";

        let turn = "saw idle-curator-fade twice. \
            then idle-curator-fade again. \
            but that's it for now.";
        let res = obs
            .observe(&scope(), turn, 1, "sess-1", Some(chunk_id))
            .await
            .unwrap();

        assert!(
            res.promoted_handles.is_empty(),
            "2 mentions must not promote"
        );
        assert!(
            res.proposed_stubs
                .iter()
                .any(|s| s.handle_guess == "idle-curator-fade"),
            "2 mentions should appear in proposed_stubs"
        );
        assert!(
            res.proposed_persisted >= 1,
            "the stub is persisted to threads_proposed with promoted_to=NULL"
        );

        let props = obs.store.list_proposed_threads().unwrap();
        let p = props
            .iter()
            .find(|p| p.proposed_handle == "idle-curator-fade")
            .expect("proposed-thread row must exist");
        assert!(
            p.promoted_to.is_none(),
            "below-threshold stub must have promoted_to=NULL"
        );
    }

    // ---- (11) promotion cap enforced per session ----

    #[tokio::test]
    async fn session_promotion_cap_enforced() {
        let (obs, _sink, _c, _s) = make_observer().await;
        // Build a turn with N distinct phrases each mentioned 3 times,
        // where N exceeds PROMOTION_CAP_PER_SESSION. Only the first
        // PROMOTION_CAP_PER_SESSION should promote; the rest fall
        // through to proposed-stub.
        // Build N distinct kebab phrases (all-letter segments so they
        // pass ThreadHandle validation) each mentioned 3+ times. The
        // suffix bank gives PROMOTION_CAP_PER_SESSION + 3 = 11 phrases
        // when the cap is 8.
        let n = PROMOTION_CAP_PER_SESSION + 3;
        assert!(
            n <= 12,
            "test assumes <=12 phrases; expand suffixes if cap grows"
        );
        let suffixes = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu",
        ];
        let mut turn = String::new();
        for suffix in suffixes.iter().take(n) {
            let phrase = format!("cap-trigger-{suffix}");
            // Interleave non-kebab "fillerN" tokens between mentions so
            // the kebab-phrase regex (which consumes its leading
            // boundary char) fires on every occurrence, not just every
            // other one. Mirrors the existing 2-occurrence test shape.
            for k in 0..3 {
                turn.push_str(&phrase);
                turn.push_str(&format!(" filler{k} "));
            }
        }

        let res = obs
            .observe(&scope(), &turn, 1, "sess-1", Some("anchor-chunk"))
            .await
            .unwrap();

        assert_eq!(
            res.promoted_handles.len(),
            PROMOTION_CAP_PER_SESSION,
            "promotion count must hit cap, not exceed it"
        );
        // Remaining candidates must surface as stubs (not silently dropped).
        assert!(
            res.proposed_stubs.len() >= n - PROMOTION_CAP_PER_SESSION,
            "post-cap candidates must surface as proposed_stubs: got {} stubs from {} excess",
            res.proposed_stubs.len(),
            n - PROMOTION_CAP_PER_SESSION
        );

        // Second observation: cap already saturated, zero new promotions
        // even with a fresh chunk anchor.
        let res2 = obs
            .observe(&scope(), &turn, 2, "sess-1", Some("anchor-chunk-2"))
            .await
            .unwrap();
        assert!(
            res2.promoted_handles.is_empty(),
            "session cap persists across calls; second observe must not promote"
        );
    }

    // ---- (12) None anchor → no persistence, still detects stubs ----

    #[tokio::test]
    async fn no_anchor_chunk_means_no_persistence() {
        let (obs, _sink, _c, _s) = make_observer().await;
        let turn = "anchor-less-phrase came up. \
            anchor-less-phrase showed up. \
            anchor-less-phrase one more.";
        let res = obs
            .observe(&scope(), turn, 1, "sess-1", None)
            .await
            .unwrap();

        // 3 mentions but no anchor → no row written; falls through to
        // the proposed-stubs return path (legacy behavior for tests).
        assert!(res.promoted_handles.is_empty());
        assert_eq!(res.proposed_persisted, 0);
        assert!(
            res.proposed_stubs
                .iter()
                .any(|s| s.handle_guess == "anchor-less-phrase"),
            "stub should still surface in the return value"
        );
        assert_eq!(obs.store.proposed_thread_count().unwrap(), 0);
    }

    // ---- (11) v0.4.1 — persistent attention: auto-promotion writes
    //               through to the in-memory score tier ----

    #[tokio::test]
    async fn auto_promotion_lights_up_in_memory_score() {
        // Build the observer, then attach a fresh InMemoryAttention via
        // with_attention. Before v0.4.1, the curator would demote this
        // freshly-promoted thread to Dormant after one tick because the
        // in-memory score was 0; the wiring this test exercises closes
        // that gap.
        let (pipeline, _corpus_tmp) = make_pipeline(16).await;
        let store_tmp = TempDir::new().unwrap();
        let sink: Arc<dyn ChainSink> = Arc::new(RecordingSink::default());
        let db = ThreadsDb::open_with_sink(store_tmp.path(), sink).unwrap();
        let attention: Arc<dyn crate::AttentionForwardStore> =
            Arc::new(crate::InMemoryAttention::new());
        let obs = TurnObserver::new(pipeline, Arc::new(db)).with_attention(Arc::clone(&attention));

        let chunk_id = "membrane:sess-1:42:0";
        let turn = "we keep returning to fade-is-concentration, \
            and fade-is-concentration is the move here, \
            because fade-is-concentration explains the curator.";

        let res = obs
            .observe(&scope(), turn, 7, "sess-1", Some(chunk_id))
            .await
            .unwrap();
        assert_eq!(
            res.promoted_handles.len(),
            1,
            "exactly one auto-promotion expected"
        );

        // The acceptance criterion from the v0.3.0 hand-off: after
        // promotion, score_thread(handle) > 0 (no waiting on chain
        // replay or the curator's stale-touch grace).
        let h = handle("fade-is-concentration");
        let score = attention.score_thread(&h).await.unwrap();
        assert!(
            score > 0.0,
            "in-memory score must be > 0 immediately after promotion; got {score}"
        );

        // And surface(...) sees the thread (score above ARCHIVE_THRESHOLD).
        let pages = attention.surface(&scope(), 10).await.unwrap();
        assert!(
            pages.iter().any(|p| p.handle == h.as_str()),
            "surface() must include the promoted thread"
        );
    }

    // ---- (12) known-handle familiarity bumps light up in-memory score ----
    //
    // Symmetric to (11): v0.4.0 closed the gap for new threads via
    // auto-promotion. This closes the same gap for *existing* threads.
    // Before this wiring, a turn that mentioned a known handle only
    // updated the SQLite counter and the chain; the in-memory store
    // stayed at score=0 until the next chain-replay on boot, leaving
    // `thread_query`'s activity axis quietly dishonest for any
    // long-running thread.

    #[tokio::test]
    async fn known_handle_mention_lights_up_in_memory_score() {
        let (pipeline, _corpus_tmp) = make_pipeline(16).await;
        let store_tmp = TempDir::new().unwrap();
        let sink: Arc<dyn ChainSink> = Arc::new(RecordingSink::default());
        let db = ThreadsDb::open_with_sink(store_tmp.path(), sink).unwrap();
        let attention: Arc<dyn crate::AttentionForwardStore> =
            Arc::new(crate::InMemoryAttention::new());

        // Seed an existing thread directly through the ledger so the
        // observer's known_handles cache picks it up.
        let h = handle("abi-as-sovereign-boundary");
        let now = chrono::Utc::now();
        db.upsert_thread(&ostk_recall_store::ThreadRecord {
            handle: h.clone(),
            tension: ostk_recall_store::TensionState::Slack,
            mentions: 0,
            resonance: 0,
            last_touched_at: now,
            anchor_chunk_id: None,
            fold_override: None,
            created_at: now,
            created_scope_key: None,
            privacy_tier: PrivacyTier::T1Project,
        })
        .unwrap();

        let obs = TurnObserver::new(pipeline, Arc::new(db)).with_attention(Arc::clone(&attention));
        obs.refresh_known_handles().await.unwrap();

        // Pre-condition: no in-memory score yet.
        let pre = attention.score_thread(&h).await.unwrap();
        assert_eq!(pre, 0.0, "in-memory score must be zero before observation");

        let turn = "back to abi-as-sovereign-boundary — the ABI is what \
            makes the implementation replaceable.";
        let res = obs
            .observe(&scope(), turn, 1, "sess-known", None)
            .await
            .unwrap();
        assert!(
            res.familiarity_increments.iter().any(|m| *m == h),
            "the known handle must be reported as a familiarity increment"
        );

        // Post-condition: score is now > 0 in-memory, without any chain
        // replay or curator tick.
        let post = attention.score_thread(&h).await.unwrap();
        assert!(
            post > 0.0,
            "in-memory score must be > 0 after a turn mentioning the handle; got {post}"
        );

        let pages = attention.surface(&scope(), 10).await.unwrap();
        assert!(
            pages.iter().any(|p| p.handle == h.as_str()),
            "surface() must include the mentioned thread"
        );
    }
}
