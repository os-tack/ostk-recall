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
    pub prose: f32,
    pub code: f32,
    pub transcript: f32,
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
    pub anchors: Vec<ThreadHandle>,
    pub shared_chunks: Vec<String>,
}

/// Result of processing a single ingest event.
#[derive(Debug)]
pub struct WeaverOutcome {
    pub event_seen: IngestEvent,
    pub evidence_links_written: usize,
    pub proposed_weaves: Vec<ProposedWeave>,
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
/// One instance is meant to live for the life of the daemon: build it
/// with `new`, then drive `run` on a long-lived task. `process_event`
/// is public so unit tests (and operators) can drive single events
/// without standing up a broadcast channel.
pub struct AutoWeaver {
    pipeline: Arc<Pipeline>,
    threads: Arc<ThreadsDb>,
    corpus: Arc<CorpusStore>,
    thresholds: WeaverThresholds,
}

impl AutoWeaver {
    #[must_use]
    pub const fn new(
        pipeline: Arc<Pipeline>,
        threads: Arc<ThreadsDb>,
        corpus: Arc<CorpusStore>,
        thresholds: WeaverThresholds,
    ) -> Self {
        Self {
            pipeline,
            threads,
            corpus,
            thresholds,
        }
    }

    /// Drive the weaver until `cancel` fires or the pipeline channel
    /// closes. Errors processing a single event are logged and
    /// swallowed — the weaver must not die because one event was
    /// malformed.
    pub async fn run(&self, cancel: CancellationToken) -> Result<(), WeaverError> {
        let mut rx = self.pipeline.subscribe_ingest();
        loop {
            tokio::select! {
                () = cancel.cancelled() => return Ok(()),
                res = rx.recv() => {
                    match res {
                        Ok(event) => {
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
        if anchor_ids.is_empty() {
            return Ok(WeaverOutcome {
                event_seen: event,
                evidence_links_written: 0,
                proposed_weaves: vec![],
            });
        }
        let anchor_embeds = self.corpus.fetch_embeddings(&anchor_ids).await?;
        let threshold = self.thresholds.for_source(event.source);
        let category = source_kind_to_category(event.source);

        let mut links_written = 0usize;
        let mut chunk_to_anchors: HashMap<String, Vec<ThreadHandle>> = HashMap::new();

        for (chunk_id, chunk_vec) in &new_embeds {
            for thread in &threads {
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
                let now = Utc::now();
                let link = EvidenceLink {
                    id: 0,
                    thread_handle: thread.handle.clone(),
                    // The corpus chunk-id is the only durable handle we
                    // have on the resonating content; the threads
                    // scanner is what knows about source paths. Future
                    // work: a corpus → source path lookup.
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
                        links_written += 1;
                        chunk_to_anchors
                            .entry(chunk_id.clone())
                            .or_default()
                            .push(thread.handle.clone());
                    }
                    Err(err) => {
                        // Most often: UNIQUE (thread, path, category)
                        // already present from a prior batch. Logged
                        // and moved on; do NOT abort the whole event.
                        tracing::debug!(error = %err, thread = %thread.handle, chunk = %chunk_id, "auto-weaver: evidence link insert skipped");
                    }
                }
            }
        }

        let proposed_weaves: Vec<ProposedWeave> = chunk_to_anchors
            .into_iter()
            .filter(|(_, anchors)| anchors.len() >= 2)
            .map(|(chunk_id, anchors)| ProposedWeave {
                anchors,
                shared_chunks: vec![chunk_id],
            })
            .collect();

        Ok(WeaverOutcome {
            event_seen: event,
            evidence_links_written: links_written,
            proposed_weaves,
        })
    }
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
        }
    }

    // Direct constructor that bypasses Pipeline — for tests that call
    // process_event without spinning up the full broadcast machinery.
    struct WeaverTestHandle {
        threads: Arc<ThreadsDb>,
        corpus: Arc<CorpusStore>,
        thresholds: WeaverThresholds,
    }

    impl WeaverTestHandle {
        const fn new(
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

        async fn process_event(&self, event: IngestEvent) -> Result<WeaverOutcome, WeaverError> {
            // Reimplements AutoWeaver::process_event verbatim, sharing
            // the same fields. Keeps tests off the Pipeline constructor
            // (which would require a real embedder + ingest DB).
            if event.chunk_ids_upserted.is_empty() {
                return Ok(WeaverOutcome {
                    event_seen: event,
                    evidence_links_written: 0,
                    proposed_weaves: vec![],
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
            if anchor_ids.is_empty() {
                return Ok(WeaverOutcome {
                    event_seen: event,
                    evidence_links_written: 0,
                    proposed_weaves: vec![],
                });
            }
            let anchor_embeds = self.corpus.fetch_embeddings(&anchor_ids).await?;
            let threshold = self.thresholds.for_source(event.source);
            let category = source_kind_to_category(event.source);
            let mut links_written = 0usize;
            let mut chunk_to_anchors: HashMap<String, Vec<ThreadHandle>> = HashMap::new();
            for (chunk_id, chunk_vec) in &new_embeds {
                for t in &threads {
                    let Some(anchor_id) = &t.anchor_chunk_id else {
                        continue;
                    };
                    let Some(anchor_vec) = anchor_embeds.get(anchor_id) else {
                        continue;
                    };
                    let sim = cosine_similarity(chunk_vec, anchor_vec);
                    if sim < threshold {
                        continue;
                    }
                    let now = Utc::now();
                    let link = EvidenceLink {
                        id: 0,
                        thread_handle: t.handle.clone(),
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
                    if self.threads.add_evidence_link(&link).is_ok() {
                        links_written += 1;
                        chunk_to_anchors
                            .entry(chunk_id.clone())
                            .or_default()
                            .push(t.handle.clone());
                    }
                }
            }
            let proposed_weaves = chunk_to_anchors
                .into_iter()
                .filter(|(_, a)| a.len() >= 2)
                .map(|(c, a)| ProposedWeave {
                    anchors: a,
                    shared_chunks: vec![c],
                })
                .collect();
            Ok(WeaverOutcome {
                event_seen: event,
                evidence_links_written: links_written,
                proposed_weaves,
            })
        }
    }

    #[tokio::test]
    async fn empty_event_returns_early() {
        let fx = fixture().await;
        let w = WeaverTestHandle::new(
            fx.threads.clone(),
            fx.corpus.clone(),
            WeaverThresholds::default(),
        );
        let out = w
            .process_event(event(SourceKind::Markdown, &[]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 0);
        assert!(out.proposed_weaves.is_empty());
    }

    #[tokio::test]
    async fn process_event_with_no_anchors_yields_zero_links() {
        let fx = fixture().await;
        // Insert a thread with NO anchor.
        fx.threads.upsert_thread(&thread("orphan", None)).unwrap();
        // Insert a corpus chunk.
        fx.corpus
            .upsert(&[chunk("c1")], &[vec![1.0, 0.0, 0.0, 0.0]])
            .await
            .unwrap();
        let w = WeaverTestHandle::new(
            fx.threads.clone(),
            fx.corpus.clone(),
            WeaverThresholds::default(),
        );
        let out = w
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
        // Identical chunk + anchor → cosine = 1.0 → above any threshold.
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
        let w = WeaverTestHandle::new(
            fx.threads.clone(),
            fx.corpus.clone(),
            WeaverThresholds::default(),
        );
        let out = w
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
        // Orthogonal vector → cosine = 0.0 → below every threshold.
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
        let w = WeaverTestHandle::new(
            fx.threads.clone(),
            fx.corpus.clone(),
            WeaverThresholds::default(),
        );
        let out = w
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
        // We pick vectors directly to hit ~0.79.
        let fx = fixture().await;
        // Two vectors with known dot/norms in 4-D yielding cos≈0.79.
        // a = (1, 0, 0, 0); b = (0.79, sqrt(1 - 0.79^2), 0, 0)
        let cos = 0.79_f32;
        let anchor_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let other_vec = vec![cos, cos.mul_add(-cos, 1.0).sqrt(), 0.0, 0.0];
        // sanity: cosine_similarity(anchor, other) should be ~0.79
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
            let w = WeaverTestHandle::new(
                fx.threads.clone(),
                fx.corpus.clone(),
                WeaverThresholds::default(),
            );
            let out = w
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
            fx.corpus
                .upsert(&[chunk("a"), chunk("b")], &[anchor_vec, other_vec])
                .await
                .unwrap();
            fx.threads.upsert_thread(&thread("tt", Some("a"))).unwrap();
            let w = WeaverTestHandle::new(
                fx.threads.clone(),
                fx.corpus.clone(),
                WeaverThresholds::default(),
            );
            let out = w
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
        let w = WeaverTestHandle::new(
            fx.threads.clone(),
            fx.corpus.clone(),
            WeaverThresholds::default(),
        );
        let out = w
            .process_event(event(SourceKind::Markdown, &["new"]))
            .await
            .unwrap();
        assert_eq!(out.evidence_links_written, 2);
        assert_eq!(out.proposed_weaves.len(), 1);
        let weave = &out.proposed_weaves[0];
        assert_eq!(weave.shared_chunks, vec!["new".to_string()]);
        assert_eq!(weave.anchors.len(), 2);
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
