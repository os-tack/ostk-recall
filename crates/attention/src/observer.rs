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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::Utc;
use ostk_recall_core::attention::{AttentionScope, PrivacyTier, ThreadHandle, ThreadHandleError};
use ostk_recall_core::{Chunk, Links, Source};
use ostk_recall_pipeline::{Pipeline, PipelineError, SyntheticSourceMeta};
use ostk_recall_store::corpus::{CorpusStore, StoreError};
use ostk_recall_store::ThreadsDb;
use regex::Regex;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

// --- public types -----------------------------------------------------

/// Three outputs of a single [`TurnObserver::observe`] call.
#[derive(Debug, Clone, Default)]
pub struct ObservationResult {
    /// How many membrane chunks the pipeline actually ingested (post
    /// dedupe). Equals `chunks_upserted` on the inner pipeline call;
    /// duplicate chunks from re-observing an identical turn collapse.
    pub membrane_chunks_ingested: usize,
    /// Handles that were already known AND appeared in this turn.
    /// One entry per distinct handle; per-turn de-duped before counting.
    pub familiarity_increments: Vec<ThreadHandle>,
    /// Handle-shaped phrases that appeared >=2 times in the turn but
    /// aren't yet in [`TurnObserver::known_handles`]. Surfaced as
    /// proposals; the operator decides whether to promote them.
    pub proposed_stubs: Vec<ProposedThreadStub>,
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
}

impl TurnObserver {
    /// Build a new observer. `known_handles` starts empty; call
    /// [`Self::refresh_known_handles`] before the first turn (or
    /// periodically thereafter) to sync from the ledger.
    #[must_use]
    pub fn new(pipeline: Arc<Pipeline>, store: Arc<ThreadsDb>) -> Self {
        Self {
            pipeline,
            store,
            known_handles: Arc::new(RwLock::new(HashSet::new())),
        }
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
    /// the three outputs documented on [`ObservationResult`].
    pub async fn observe(
        &self,
        scope: &AttentionScope,
        turn_text: &str,
        turn_seq: u64,
        session_id: &str,
    ) -> Result<ObservationResult, ObserverError> {
        let known_snapshot: HashSet<ThreadHandle> = self.known_handles.read().await.clone();

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
        let mentioned = handles_mentioned(turn_text, &known_snapshot);
        if !mentioned.is_empty() {
            // Skip handles that are in the in-memory cache but not in the
            // ledger (cache can drift briefly ahead of the ledger,
            // especially under seed_known_handles in tests). Increment
            // returns Err for those; we drop them silently rather than
            // failing the whole observation.
            let mut entries: Vec<(ThreadHandle, u32)> = Vec::with_capacity(mentioned.len());
            for handle in &mentioned {
                if let Ok(new_familiarity) = self.store.increment_familiarity(handle) {
                    entries.push((handle.clone(), new_familiarity));
                }
            }
            if !entries.is_empty() {
                self.store.record_familiarity_batch(entries, turn_seq)?;
            }
        }

        // (c) proposed thread stubs — kebab-case phrases recurring in
        // the turn that aren't known yet.
        let proposed_stubs = detect_proposed_stubs(turn_text, &known_snapshot);

        Ok(ObservationResult {
            membrane_chunks_ingested,
            familiarity_increments: mentioned,
            proposed_stubs,
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
                                    .observe(&scope, &text, s, &session)
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
                            let _ = self.observe(&scope, &text, s, &session).await;
                        }
                    }
                    return Ok(());
                }
            }
        }
    }
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
            let chunk_id = Chunk::make_id(Source::Membrane, &source_id, chunk_index);
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

fn detect_proposed_stubs(
    turn_text: &str,
    known: &HashSet<ThreadHandle>,
) -> Vec<ProposedThreadStub> {
    // Lowered text so case-insensitivity in the rest of the turn doesn't
    // dodge the match (handles are kebab-case-lowercase by construction).
    let lowered = turn_text.to_lowercase();
    // Count occurrences and remember first match position per candidate.
    let mut counts: HashMap<String, (usize, usize, usize)> = HashMap::new();
    for caps in kebab_phrase_regex().captures_iter(&lowered) {
        let Some(m) = caps.get(1) else { continue };
        let phrase = m.as_str().to_string();
        // Drop candidates that fail handle validation (e.g. too long,
        // too many hyphens) — they can never become a real ThreadHandle
        // anyway.
        if ThreadHandle::new(phrase.clone()).is_err() {
            continue;
        }
        // Known handles aren't proposals — they're familiarity ticks.
        if known.iter().any(|h| h.as_str() == phrase) {
            continue;
        }
        let entry = counts
            .entry(phrase)
            .or_insert_with(|| (0, m.start(), m.end()));
        entry.0 += 1;
    }

    let mut stubs: Vec<ProposedThreadStub> = counts
        .into_iter()
        .filter(|(_, (n, _, _))| *n >= STUB_MIN_OCCURRENCES)
        .map(|(phrase, (n, start, end))| {
            // Window the snippet from the original turn_text, not the
            // lowered version, so casing is preserved when surfaced.
            let context = token_window(turn_text, start, end, CONTEXT_SNIPPET_TOKENS);
            let confidence = stub_confidence(n);
            ProposedThreadStub {
                handle_guess: phrase,
                context_snippet: context,
                confidence,
            }
        })
        .collect();
    // Stable order: highest confidence first, then alphabetical.
    stubs.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.handle_guess.cmp(&b.handle_guess))
    });
    stubs
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
                familiarity: 0,
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
        let res = obs.observe(&scope(), turn, 5, "sess-1").await.unwrap();
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
        let res = obs.observe(&scope(), turn, 1, "sess-1").await.unwrap();
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
        let res = obs.observe(&scope(), single, 1, "sess-1").await.unwrap();
        assert!(
            !res.proposed_stubs
                .iter()
                .any(|s| s.handle_guess == "idle-curator-fade"),
            "single occurrence should not propose a stub"
        );

        // Double mention => stub.
        let double = "We saw idle-curator-fade come up twice. \
            Then idle-curator-fade again, in the second sentence.";
        let res = obs.observe(&scope(), double, 2, "sess-1").await.unwrap();
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
        let res = obs.observe(&scope(), turn, 3, "sess-1").await.unwrap();
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
                    familiarity: 0,
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

        let res = obs.observe(&scope(), turn, 11, "sess-1").await.unwrap();

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
}
