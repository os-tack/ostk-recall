//! `ostk-recall-pipeline` — scan → chunk → dedupe → embed → persist.
//!
//! The [`Pipeline`] owns the three side-effecting resources — a
//! [`CorpusStore`], an [`IngestDb`], and something that can embed text
//! (the [`ChunkEmbedder`] trait) — and wires a [`Scanner`] up to them.

use std::sync::Arc;
use uuid::Uuid;

use ostk_recall_core::{Chunk, Scanner, SourceConfig, RetentionPolicy};
use ostk_recall_embed::Embedder;
use ostk_recall_store::{CorpusStore, IngestChunkRow, IngestDb};

/// Batch size used when calling the embedder.
pub const EMBED_BATCH: usize = 64;

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
        self.errors += other.errors;
        self
    }
}

pub struct Pipeline {
    store: Arc<CorpusStore>,
    ingest: Arc<IngestDb>,
    embedder: Arc<dyn ChunkEmbedder>,
    dry_run: bool,
    run_id: String,
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
            run_id: Uuid::now_v7().to_string(),
        }
    }

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

    pub async fn ingest_source(&self, scanner: &dyn Scanner, cfg: &SourceConfig) -> PipelineStats {
        let mut stats = PipelineStats::default();
        let mut to_embed: Vec<Chunk> = Vec::new();
        let source_kind_str = scanner.kind().as_str();
        let project = cfg.project.as_deref().unwrap_or("default");

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

            // Tier 1: File-level metadata check.
            //
            // A `--dry-run` invocation in v0.1.0 wrote to ingest_sources
            // without ever persisting the corresponding chunks. Future
            // scans then matched the cached mtime and silently skipped
            // parse, leaving the corpus permanently empty for that
            // source. v0.1.1 fixes this with two guards:
            //
            //   1. Dry-runs never write to ingest_sources at all
            //      (observation-only).
            //   2. The skip path only fires when chunks for this
            //      source_id actually exist in ingest_chunks — so a
            //      previously-interrupted scan won't poison the cache
            //      either.
            if let Some(path) = &item.path {
                if let Ok(meta) = std::fs::metadata(path) {
                    let mtime = meta.modified().ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_micros() as i64)
                        .unwrap_or(0);
                    let size = meta.len() as i64;

                    if let Ok(Some((old_mtime, old_size))) = self.ingest.get_source_metadata(source_kind_str, project, &item.source_id) {
                        if old_mtime == mtime && old_size == size {
                            // Defensive: a metadata row with no chunks
                            // means the previous run wrote metadata but
                            // never persisted any chunks (typical of an
                            // interrupted scan or a v0.1.0 dry-run).
                            // Re-parse instead of silently skipping.
                            let has_chunks = self
                                .ingest
                                .has_chunks_for_source(source_kind_str, project, &item.source_id)
                                .unwrap_or(false);
                            if has_chunks {
                                if self.ingest.touch_source_chunks(source_kind_str, project, &item.source_id, &self.run_id).is_ok() {
                                    stats.items_skipped += 1;
                                    continue;
                                }
                            }
                        }
                    }

                    if !self.dry_run {
                        let _ = self.ingest.update_source_metadata(source_kind_str, project, &item.source_id, mtime, size, &self.run_id);
                    }
                }
            }

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
                // Tier 2: Chunk-level dedupe
                match self.ingest.content_already_ingested(&chunk.chunk_id, &chunk.sha256) {
                    Ok(true) => {
                        stats.chunks_skipped_dup += 1;
                        let row = IngestChunkRow {
                            chunk_id: chunk.chunk_id.clone(),
                            source: chunk.source.as_str().to_string(),
                            project: project.to_string(),
                            source_id: chunk.source_id.clone(),
                            chunk_index: chunk.chunk_index,
                            content_sha256: chunk.sha256.clone(),
                        };
                        let _ = self.ingest.record_chunk(&row, Some(&self.run_id));
                    },
                    Ok(false) => to_embed.push(chunk),
                    Err(e) => {
                        tracing::warn!(error = %e, "ingest dedupe check failed");
                        stats.errors += 1;
                    }
                }
            }
        }

        if !to_embed.is_empty() && !self.dry_run {
            for batch in to_embed.chunks(EMBED_BATCH) {
                match self.embed_and_persist(batch, project).await {
                    Ok(n) => stats.chunks_upserted += n,
                    Err(e) => {
                        tracing::warn!(error = %e, "embed/persist batch failed");
                        stats.errors += 1;
                    }
                }
            }
        }

        // Orphan Sweep — multi-project safety contract
        //
        // The orphan sweep filters by (source, project, run_id). If `cfg.project`
        // is None we'd fall back to a placeholder ("default"), and any chunks
        // ingested under that placeholder by a *different* unspecified-project
        // scan would get swept under this run_id. That's the cross-project
        // wipe footgun. Refuse to run the sweep when project is unset; let the
        // caller fix their config.
        if !self.dry_run && cfg.project.is_none() {
            tracing::error!(
                source_kind = %source_kind_str,
                "orphan sweep skipped: SourceConfig.project is None — multi-project safety contract requires an explicit project for any source whose retention policy mutates the corpus. See crates/pipeline/src/lib.rs orphan-sweep guard."
            );
            stats.errors += 1;
            return stats;
        }

        // Orphan Sweep
        if !self.dry_run {
            match cfg.kind.retention_policy() {
                RetentionPolicy::Delete => {
                    for s in cfg.kind.sources() {
                        match self.ingest.delete_orphans(s.as_str(), project, &self.run_id) {
                            Ok(orphans) => {
                                if !orphans.is_empty() {
                                    if let Err(e) = self.store.delete_chunks(&orphans).await {
                                        tracing::warn!(error = %e, source = %s.as_str(), "failed to delete orphan chunks from corpus store");
                                    } else {
                                        stats.chunks_purged += orphans.len();
                                    }
                                }
                            }
                            Err(e) => tracing::warn!(error = %e, source = %s.as_str(), "orphan sweep failed"),
                        }
                    }
                }
                RetentionPolicy::Stale => {
                    for s in cfg.kind.sources() {
                        match self.ingest.mark_orphans_stale(s.as_str(), project, &self.run_id) {
                            Ok(orphans) => {
                                if !orphans.is_empty() {
                                    if let Err(e) = self.store.mark_chunks_stale(&orphans).await {
                                        tracing::warn!(error = %e, source = %s.as_str(), "failed to mark orphan chunks as stale in corpus store");
                                    }
                                }
                                stats.chunks_staled += orphans.len();
                            }
                            Err(e) => tracing::warn!(error = %e, source = %s.as_str(), "stale marking failed"),
                        }
                    }
                }
                RetentionPolicy::Keep => {}
            }
        }

        stats
    }

    async fn embed_and_persist(&self, batch: &[Chunk], project: &str) -> Result<usize, PipelineError> {
        let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
        let embeddings = self.embedder.encode_batch(&texts);
        if embeddings.len() != batch.len() {
            return Err(PipelineError::Embedder("dim mismatch".into()));
        }

        self.store.upsert(batch, &embeddings).await
            .map_err(|e| PipelineError::Store(e.to_string()))?;

        for chunk in batch {
            let row = IngestChunkRow {
                chunk_id: chunk.chunk_id.clone(),
                source: chunk.source.as_str().to_string(),
                project: project.to_string(),
                source_id: chunk.source_id.clone(),
                chunk_index: chunk.chunk_index,
                content_sha256: chunk.sha256.clone(),
            };
            self.ingest.record_chunk(&row, Some(&self.run_id))
                .map_err(|e| PipelineError::Store(e.to_string()))?;
        }
        Ok(batch.len())
    }

    pub async fn verify_counts(&self) -> Result<VerifyReport, PipelineError> {
        let corpus_total = self.store.row_count().await.map_err(|e| PipelineError::Store(e.to_string()))?;
        let ingest_by_source = self.ingest.count_active_by_source().map_err(|e| PipelineError::Store(e.to_string()))?;
        let ingest_total: u64 = ingest_by_source.iter().map(|(_, n)| *n).sum();
        Ok(VerifyReport {
            corpus_total,
            ingest_total: ingest_total as usize,
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
        use ostk_recall_scan::code::CodeScanner;
        use ostk_recall_core::SourceConfig;

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

    /// Regression: v0.1.0 dry-runs wrote source metadata before the
    /// dry-run gate, which meant a subsequent dry-run (or a real run)
    /// matched the cached mtime, hit the metadata-skip path, and
    /// emitted zero chunks even though the corpus was empty. The fix
    /// is two-pronged: dry-runs no longer write metadata, and the
    /// skip-path requires the chunks table to actually contain rows
    /// for the source_id.
    #[tokio::test]
    async fn dry_run_does_not_poison_metadata_cache() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        // First run: dry-run only. Discovers items but should NOT
        // populate ingest_sources (which would poison cache).
        {
            let pipeline = make_pipeline(corpus.path(), 16).await.with_dry_run(true);
            let s = pipeline.ingest_source(&scanner, &cfg).await;
            assert_eq!(s.items_seen, 2);
            assert_eq!(s.items_skipped, 0);
            // No actual ingest happened.
            assert_eq!(s.chunks_upserted, 0);
        }

        // Second run: real ingest. If dry-run poisoned the cache, this
        // would items_skipped=2 chunks_emitted=0 — the v0.1.0 bug. With
        // the fix it parses fresh.
        let pipeline = make_pipeline(corpus.path(), 16).await;
        let s2 = pipeline.ingest_source(&scanner, &cfg).await;
        assert_eq!(s2.items_seen, 2);
        assert_eq!(s2.items_skipped, 0, "dry-run must not poison metadata cache");
        assert!(s2.chunks_emitted > 0, "second run must emit chunks");
        assert!(s2.chunks_upserted > 0);
    }

    /// Regression: a run that wrote metadata but crashed before
    /// recording any chunks (or v0.1.0 in dry-run mode) leaves
    /// `ingest_sources` populated and `ingest_chunks` empty. v0.1.1
    /// detects that mismatch and re-parses instead of skipping.
    #[tokio::test]
    async fn metadata_without_chunks_re_parses() {
        let fixtures = TempDir::new().unwrap();
        let corpus = TempDir::new().unwrap();
        write_sample_tree(fixtures.path());

        let pipeline = make_pipeline(corpus.path(), 16).await;
        let scanner = MarkdownScanner;
        let cfg = cfg_for(fixtures.path());

        // Manually simulate the broken state: write metadata for one of
        // the items but never record chunks.
        let path = fixtures.path().join("a.md");
        let meta = std::fs::metadata(&path).unwrap();
        let mtime = meta
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;
        let size = meta.len() as i64;
        pipeline
            .ingest
            .update_source_metadata("markdown", "test", "a.md", mtime, size, "stale-run-id")
            .unwrap();

        // Now run normally. The skip-path must NOT trigger for a.md
        // because the chunks table is empty.
        let s = pipeline.ingest_source(&scanner, &cfg).await;
        assert_eq!(s.items_seen, 2);
        assert_eq!(s.items_skipped, 0, "must not skip when chunks table is empty");
        assert!(s.chunks_emitted > 0);
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
        let table = pipeline.store().connection().open_table(ostk_recall_store::CORPUS_TABLE).execute().await.unwrap();
        let stream = table.query().execute().await.unwrap();
        let batches: Vec<arrow_array::RecordBatch> = stream.try_collect().await.unwrap();
        
        let mut found_stale = false;
        for batch in batches {
            let stale_col = batch.column_by_name("stale").unwrap().as_any().downcast_ref::<arrow_array::BooleanArray>().unwrap();
            for i in 0..batch.num_rows() {
                if stale_col.value(i) {
                    found_stale = true;
                }
            }
        }
        assert!(found_stale, "at least one chunk should be marked stale");
    }
}
