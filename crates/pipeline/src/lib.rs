//! `ostk-recall-pipeline` — scan → chunk → dedupe → embed → persist.
//!
//! The [`Pipeline`] owns the three side-effecting resources — a
//! [`CorpusStore`], an [`IngestDb`], and something that can embed text
//! (the [`ChunkEmbedder`] trait) — and wires a [`Scanner`] up to them.
//!
//! The embedder is behind a trait so tests can inject a deterministic
//! `FakeEmbedder` without requiring model weights on disk.

use std::sync::Arc;

use ostk_recall_core::{Chunk, Scanner, Source, SourceConfig};
use ostk_recall_embed::Embedder;
use ostk_recall_store::{CorpusStore, IngestChunkRow, IngestDb};

/// Batch size used when calling the embedder. Keeps peak memory bounded and
/// gives a natural checkpoint between upsert/record phases.
pub const EMBED_BATCH: usize = 64;

/// Anything that can turn a batch of texts into a batch of vectors.
///
/// Defined here (instead of in `ostk-recall-embed`) so the pipeline crate
/// can impl it for `Embedder` without orphan-rule issues, and so tests can
/// provide a pure-Rust fake without touching the embed crate.
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

/// Per-run counters returned from [`Pipeline::ingest_source`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStats {
    pub items_seen: usize,
    pub chunks_emitted: usize,
    pub chunks_upserted: usize,
    pub chunks_skipped_dup: usize,
    pub errors: usize,
}

impl PipelineStats {
    #[must_use]
    pub const fn merge(mut self, other: Self) -> Self {
        self.items_seen += other.items_seen;
        self.chunks_emitted += other.chunks_emitted;
        self.chunks_upserted += other.chunks_upserted;
        self.chunks_skipped_dup += other.chunks_skipped_dup;
        self.errors += other.errors;
        self
    }
}

/// The orchestrator. Cheap to clone — holds `Arc`s.
pub struct Pipeline {
    store: Arc<CorpusStore>,
    ingest: Arc<IngestDb>,
    embedder: Arc<dyn ChunkEmbedder>,
    dry_run: bool,
}

impl Pipeline {
    pub fn new(
        store: Arc<CorpusStore>,
        ingest: Arc<IngestDb>,
        embedder: Arc<dyn ChunkEmbedder>,
    ) -> Self {
        Self {
            store,
            ingest,
            embedder,
            dry_run: false,
        }
    }

    /// Set dry-run mode. In dry-run the pipeline still discovers, parses, and
    /// classifies chunks as dup/new — but it never calls the embedder or
    /// writes to the stores.
    #[must_use]
    pub const fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    pub fn store(&self) -> &CorpusStore {
        &self.store
    }

    pub fn ingest_db(&self) -> &IngestDb {
        &self.ingest
    }

    /// Ingest one source.
    ///
    /// Discovers items, parses them into chunks, filters out chunks already
    /// in `ingest_chunks` with matching `(chunk_id, content_sha256)`,
    /// embeds the survivors in batches of `EMBED_BATCH`, and upserts into
    /// `CorpusStore` + records into `IngestDb`.
    pub async fn ingest_source(&self, scanner: &dyn Scanner, cfg: &SourceConfig) -> PipelineStats {
        let mut stats = PipelineStats::default();

        let mut to_embed: Vec<Chunk> = Vec::new();

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

            let chunks = match scanner.parse(item) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(error = %e, "parse failed");
                    stats.errors += 1;
                    continue;
                }
            };
            stats.chunks_emitted += chunks.len();

            for chunk in chunks {
                match self
                    .ingest
                    .content_already_ingested(&chunk.chunk_id, &chunk.sha256)
                {
                    Ok(true) => stats.chunks_skipped_dup += 1,
                    Ok(false) => to_embed.push(chunk),
                    Err(e) => {
                        tracing::warn!(error = %e, "ingest dedupe check failed");
                        stats.errors += 1;
                    }
                }
            }
        }

        if to_embed.is_empty() || self.dry_run {
            return stats;
        }

        for batch in to_embed.chunks(EMBED_BATCH) {
            match self.embed_and_persist(batch).await {
                Ok(n) => stats.chunks_upserted += n,
                Err(e) => {
                    tracing::warn!(error = %e, "embed/persist batch failed");
                    stats.errors += 1;
                }
            }
        }

        stats
    }

    async fn embed_and_persist(&self, batch: &[Chunk]) -> Result<usize, PipelineError> {
        let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
        let embeddings = self.embedder.encode_batch(&texts);
        if embeddings.len() != batch.len() {
            return Err(PipelineError::Embedder(format!(
                "embedder returned {} vectors for {} inputs",
                embeddings.len(),
                batch.len()
            )));
        }
        // Dim sanity check (first vector only — they're all from the same
        // model in one call).
        if let Some(first) = embeddings.first() {
            let want = self.embedder.dim();
            if first.len() != want {
                return Err(PipelineError::Embedder(format!(
                    "embedder dim mismatch: got {}, want {want}",
                    first.len()
                )));
            }
        }

        self.store
            .upsert(batch, &embeddings)
            .await
            .map_err(|e| PipelineError::Store(e.to_string()))?;

        for chunk in batch {
            let row = IngestChunkRow {
                chunk_id: chunk.chunk_id.clone(),
                source: chunk.source.as_str().to_string(),
                source_id: chunk.source_id.clone(),
                chunk_index: chunk.chunk_index,
                content_sha256: chunk.sha256.clone(),
            };
            self.ingest
                .record_chunk(&row)
                .map_err(|e| PipelineError::Store(e.to_string()))?;
        }
        Ok(batch.len())
    }

    /// Compare ingest-db count vs corpus row count.
    pub async fn verify_counts(&self) -> Result<VerifyReport, PipelineError> {
        let corpus_total = self
            .store
            .row_count()
            .await
            .map_err(|e| PipelineError::Store(e.to_string()))?;
        let ingest_by_source = self
            .ingest
            .count_by_source()
            .map_err(|e| PipelineError::Store(e.to_string()))?;
        let ingest_total: u64 = ingest_by_source.iter().map(|(_, n)| *n).sum();
        Ok(VerifyReport {
            corpus_total,
            ingest_total: ingest_total.try_into().unwrap_or(usize::MAX),
            by_source: ingest_by_source,
        })
    }

    /// Count ingested chunks for a particular `Source` (Phase-B helper).
    pub fn count_for_source(&self, source: Source) -> Result<u64, PipelineError> {
        self.ingest
            .count_for_source(source)
            .map_err(|e| PipelineError::Store(e.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub corpus_total: usize,
    pub ingest_total: usize,
    pub by_source: Vec<(String, u64)>,
}

impl VerifyReport {
    #[must_use]
    pub const fn is_consistent(&self) -> bool {
        self.corpus_total == self.ingest_total
    }
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

    /// Deterministic embedder for tests. Produces `dim` floats derived from
    /// the input length so two different texts get distinct-but-stable
    /// vectors without needing a model on disk.
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
            extensions: vec![],
        }
    }

    async fn make_pipeline(corpus_root: &Path, dim: usize) -> Pipeline {
        let store = Arc::new(CorpusStore::open_or_create(corpus_root, dim).await.unwrap());
        let ingest = Arc::new(IngestDb::open(corpus_root).unwrap());
        let emb: Arc<dyn ChunkEmbedder> = Arc::new(FakeEmbedder { dim });
        Pipeline::new(store, ingest, emb)
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
    async fn rerun_is_idempotent() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        let s1 = pipeline.ingest_source(&scanner, &cfg).await;
        assert!(s1.chunks_upserted > 0);

        let s2 = pipeline.ingest_source(&scanner, &cfg).await;
        assert_eq!(
            s2.chunks_upserted, 0,
            "second run should upsert nothing (all dup)"
        );
        assert_eq!(s2.chunks_skipped_dup, s1.chunks_upserted);
        assert_eq!(s2.errors, 0);
    }

    #[tokio::test]
    async fn stats_merge_is_arithmetic() {
        let a = PipelineStats {
            items_seen: 1,
            chunks_emitted: 2,
            chunks_upserted: 3,
            chunks_skipped_dup: 4,
            errors: 5,
        };
        let b = PipelineStats {
            items_seen: 10,
            chunks_emitted: 20,
            chunks_upserted: 30,
            chunks_skipped_dup: 40,
            errors: 50,
        };
        let m = a.merge(b);
        assert_eq!(m.items_seen, 11);
        assert_eq!(m.chunks_emitted, 22);
        assert_eq!(m.chunks_upserted, 33);
        assert_eq!(m.chunks_skipped_dup, 44);
        assert_eq!(m.errors, 55);
    }

    #[tokio::test]
    async fn dry_run_does_not_upsert() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let pipeline = make_pipeline(corpus.path(), 16).await.with_dry_run(true);
        let scanner = MarkdownScanner;
        let stats = pipeline
            .ingest_source(&scanner, &cfg_for(fixtures.path()))
            .await;

        assert!(stats.chunks_emitted > 0);
        assert_eq!(stats.chunks_upserted, 0);
    }

    #[tokio::test]
    async fn verify_counts_balanced_after_ingest() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());
        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = MarkdownScanner;
        pipeline
            .ingest_source(&scanner, &cfg_for(fixtures.path()))
            .await;
        let report = pipeline.verify_counts().await.unwrap();
        assert!(report.is_consistent(), "report: {report:?}");
        assert!(report.corpus_total > 0);
    }
}
