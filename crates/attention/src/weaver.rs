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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use ostk_recall_core::{SourceKind, ThreadHandle};
use ostk_recall_pipeline::{IngestEvent, Pipeline};
use ostk_recall_store::{
    AssociationType, CorpusStore, EvidenceLink, RelationState, StoreError, ThreadsDb,
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
    /// Per-chunk groupings the weaver detected but did not write —
    /// the caller decides whether to surface them as candidate threads.
    pub proposed_weaves: Vec<ProposedWeave>,
    /// Count of `threads_proposed` rows the weaver inserted for this
    /// event (one per emergent cluster found in the unmatched
    /// chunks).
    pub proposed_threads_written: usize,
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
        if event.chunk_ids_upserted.is_empty() {
            return Ok(WeaverOutcome {
                event_seen: event,
                evidence_links_written: 0,
                proposed_weaves: vec![],
                proposed_threads_written: 0,
            });
        }

        let new_embeds = self
            .corpus
            .fetch_embeddings(&event.chunk_ids_upserted)
            .await?;
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
        let threshold = self.thresholds.for_source(event.source);
        let category = source_kind_to_category(event.source);

        let MatchPassOutcome {
            links_written,
            chunk_to_anchors,
            matched_chunks,
        } = self.match_against_anchors(
            &new_embeds,
            &threads,
            &anchor_embeds,
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
            .filter(|(id, _)| !matched_chunks.contains(*id))
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();
        let proposed_threads_written =
            self.write_emergent_proposals(&unmatched).unwrap_or_else(|err| {
                tracing::warn!(error = %err, "auto-weaver: emergent-cluster write failed");
                0
            });

        Ok(WeaverOutcome {
            event_seen: event,
            evidence_links_written: links_written,
            proposed_weaves,
            proposed_threads_written,
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
                        // The (thread, path, category) edge was written
                        // in a prior batch. Idempotent no-op.
                        tracing::trace!(
                            thread = %thread.handle,
                            chunk = %chunk_id,
                            "auto-weaver: evidence link already exists",
                        );
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
            let handle = generate_proposed_handle(&cluster.chunk_ids, now);
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
    chunk_to_anchors: HashMap<String, Vec<ThreadHandle>>,
    matched_chunks: std::collections::HashSet<String>,
}

/// Build a kebab-case proposed-thread handle from a cluster's member
/// chunks. Format: `proposed-<8 hex chars>` — deterministic enough that
/// repeated scans of the same cluster collapse to one row (the unique
/// constraint absorbs the duplicate), short enough to fit
/// `ThreadHandle`'s 64-char / 4-hyphen budget.
pub(crate) fn generate_proposed_handle(chunk_ids: &[String], ts: chrono::DateTime<Utc>) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for id in chunk_ids {
        hasher.update(id.as_bytes());
        hasher.update(b"\n");
    }
    // Include the date (not the full ts) so the same cluster surfaced
    // in two scans on the same day collides; a different day yields a
    // new proposal (which the operator can still merge).
    hasher.update(ts.format("%Y%m%d").to_string().as_bytes());
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
    use ostk_recall_store::{CorpusStore, ThreadRecord, ThreadsDb};
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
        Chunk {
            chunk_id: id.into(),
            source: Source::Markdown,
            project: Some("test".into()),
            source_id: format!("{id}.md"),
            facets: Default::default(),
            embedding_input_sha256: String::new(),
            source_config_id: "test-cfg".to_string(),
            chunk_index: 0,
            ts: None,
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
            .upsert(
                &[chunk("u1"), chunk("u2"), chunk("u3")],
                &[v, v2, v3],
            )
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
}
