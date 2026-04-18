//! Cross-encoder reranker. Optional second-pass over hybrid results.
//!
//! After hybrid (dense + BM25 + RRF) returns the top-N candidates, a
//! cross-encoder model scores each `(query, doc_text)` pair jointly and
//! re-orders. This is the production-standard ranking pipeline: BM25 +
//! dense return broad recall, then a small cross-encoder tightens
//! precision in the top-K.
//!
//! The default model is [`fastembed::RerankerModel::JINARerankerV1TurboEn`]
//! — ~80 MB ONNX, English-tuned, ~5 ms per `(query, doc)` pair on CPU.
//! `fastembed` v5 does *not* expose `MSMarcoMiniLML6V2`; the JINA Turbo
//! model is the closest English-only equivalent (similar size, similar
//! latency, MS-MARCO-class quality on retrieval rerank).
//!
//! # Threading
//!
//! `fastembed::TextRerank::rerank` takes `&mut self`, but `QueryEngine`
//! shares the reranker behind `Arc<dyn RerankerLike>`. We wrap the
//! underlying `TextRerank` in a `Mutex` — the model is fast enough on
//! CPU (~5 ms / pair) that lock contention isn't material at single-user
//! recall throughput.
//!
//! # Testability
//!
//! [`RerankerLike`] is the trait the hybrid path actually consumes; the
//! real ONNX-backed [`Reranker`] is one impl, but tests can supply a
//! deterministic fake without pulling the model.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RerankerError {
    #[error("fastembed: {0}")]
    Fastembed(String),
}

pub type Result<T> = std::result::Result<T, RerankerError>;

/// One reranked candidate: index back into the input slice + relevance score.
#[derive(Debug, Clone, Copy)]
pub struct RerankedHit {
    pub idx: usize,
    pub score: f32,
}

/// Trait the hybrid path consumes. Real impl wraps fastembed; tests can
/// inject a deterministic fake without ONNX.
pub trait RerankerLike: Send + Sync {
    /// Rerank `docs` against `query`. Return indices into `docs` ordered
    /// by score, descending. Take at most `top_k` (caller decides; the
    /// implementation may return fewer if `docs.len() < top_k`).
    fn rerank(&self, query: &str, docs: &[String], top_k: usize) -> Result<Vec<RerankedHit>>;

    /// Model identifier, used by `recall_stats`.
    fn model_id(&self) -> &str;
}

/// Production cross-encoder reranker.
///
/// Holds an ONNX-backed `fastembed::TextRerank` behind a `Mutex` so the
/// engine can share an `Arc<Self>` across async tasks. The mutex is held
/// only for the duration of one `rerank` call; latency dominates.
pub struct Reranker {
    inner: Mutex<TextRerank>,
    model_id: String,
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reranker")
            .field("model_id", &self.model_id)
            .finish_non_exhaustive()
    }
}

impl Reranker {
    /// Default model identifier surfaced to config / `recall_stats`.
    pub const DEFAULT_MODEL_ID: &'static str = "jina-reranker-v1-turbo-en";

    /// Load the default reranker model into `cache_dir`.
    ///
    /// First call downloads the ONNX file (~80 MB) into `cache_dir`;
    /// subsequent calls reuse the cached file.
    pub fn load(cache_dir: &Path) -> Result<Arc<Self>> {
        Self::load_with_model(cache_dir, Self::default_model())
    }

    /// Load a specific reranker variant. Use [`Self::resolve_model`] to
    /// translate a config string into the enum variant.
    pub fn load_with_model(cache_dir: &Path, model: RerankerModel) -> Result<Arc<Self>> {
        let model_id = model_to_id(&model).to_string();
        let opts = RerankInitOptions::new(model)
            .with_cache_dir(PathBuf::from(cache_dir))
            .with_show_download_progress(false);
        let inner = TextRerank::try_new(opts)
            .map_err(|e| RerankerError::Fastembed(format!("try_new: {e}")))?;
        Ok(Arc::new(Self {
            inner: Mutex::new(inner),
            model_id,
        }))
    }

    /// The model used as the workspace default. Currently
    /// `JINARerankerV1TurboEn`: 80 MB, English-only, MS-MARCO-class
    /// retrieval quality.
    #[must_use]
    pub const fn default_model() -> RerankerModel {
        RerankerModel::JINARerankerV1TurboEn
    }

    /// Map a string from config to a `RerankerModel`. Accepts both the
    /// short alias (`"jina-reranker-v1-turbo-en"`) and the upstream
    /// `HuggingFace` id (`"jinaai/jina-reranker-v1-turbo-en"`).
    ///
    /// Unknown values return `None` so the caller can decide whether to
    /// hard-fail or fall back to the default.
    ///
    /// The historical MS-MARCO `MiniLM` identifier doesn't ship in
    /// fastembed v5; we accept it as an alias for the JINA Turbo English
    /// model so old configs keep working without a hard error.
    #[must_use]
    pub fn resolve_model(spec: &str) -> Option<RerankerModel> {
        match spec {
            "jina-reranker-v1-turbo-en"
            | "jinaai/jina-reranker-v1-turbo-en"
            | "ms-marco-MiniLM-L-6-v2"
            | "MSMarcoMiniLML6V2" => Some(RerankerModel::JINARerankerV1TurboEn),
            "jina-reranker-v2-base-multilingual" | "jinaai/jina-reranker-v2-base-multilingual" => {
                Some(RerankerModel::JINARerankerV2BaseMultiligual)
            }
            "bge-reranker-base" | "BAAI/bge-reranker-base" => Some(RerankerModel::BGERerankerBase),
            "bge-reranker-v2-m3" | "rozgo/bge-reranker-v2-m3" => {
                Some(RerankerModel::BGERerankerV2M3)
            }
            _ => None,
        }
    }
}

impl RerankerLike for Reranker {
    fn rerank(&self, query: &str, docs: &[String], top_k: usize) -> Result<Vec<RerankedHit>> {
        if docs.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        // Scope the lock tight: drop the guard the moment the call returns
        // so concurrent recall callers don't queue behind us longer than
        // strictly needed.
        let results = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|e| RerankerError::Fastembed(format!("mutex poisoned: {e}")))?;
            guard
                .rerank(query, refs, false, None)
                .map_err(|e| RerankerError::Fastembed(format!("rerank: {e}")))?
        };
        Ok(results
            .into_iter()
            .take(top_k)
            .map(|r| RerankedHit {
                idx: r.index,
                score: r.score,
            })
            .collect())
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

const fn model_to_id(m: &RerankerModel) -> &'static str {
    match m {
        RerankerModel::JINARerankerV1TurboEn => "jina-reranker-v1-turbo-en",
        RerankerModel::JINARerankerV2BaseMultiligual => "jina-reranker-v2-base-multilingual",
        RerankerModel::BGERerankerBase => "bge-reranker-base",
        RerankerModel::BGERerankerV2M3 => "bge-reranker-v2-m3",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial fake reranker used to test the hybrid wire-up. Returns the
    /// docs in their original order with score = 1.0 / (rank+1).
    struct IdentityReranker;

    impl RerankerLike for IdentityReranker {
        fn rerank(&self, _query: &str, docs: &[String], top_k: usize) -> Result<Vec<RerankedHit>> {
            #[allow(clippy::cast_precision_loss)]
            Ok((0..docs.len().min(top_k))
                .map(|i| RerankedHit {
                    idx: i,
                    score: 1.0 / ((i + 1) as f32),
                })
                .collect())
        }
        fn model_id(&self) -> &'static str {
            "identity"
        }
    }

    #[test]
    fn resolve_model_known_aliases() {
        assert!(Reranker::resolve_model("jina-reranker-v1-turbo-en").is_some());
        assert!(Reranker::resolve_model("bge-reranker-base").is_some());
        // Legacy MS-MARCO id maps to the default English model.
        assert!(Reranker::resolve_model("ms-marco-MiniLM-L-6-v2").is_some());
        assert!(Reranker::resolve_model("nope").is_none());
    }

    #[test]
    fn identity_reranker_preserves_order() {
        let r = IdentityReranker;
        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let out = r.rerank("q", &docs, 5).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].idx, 0);
        assert_eq!(out[1].idx, 1);
        assert_eq!(out[2].idx, 2);
    }

    /// E2E test gated on `OSTK_RECALL_E2E=1` since it pulls the ONNX
    /// reranker model (~80 MB).
    #[test]
    #[ignore = "requires network / model download — set OSTK_RECALL_E2E=1 to run"]
    fn loads_real_reranker_and_scores() {
        if std::env::var("OSTK_RECALL_E2E").is_err() {
            return;
        }
        let tmp = tempfile::TempDir::new().unwrap();
        let r = Reranker::load(tmp.path()).expect("load reranker");
        let docs: Vec<String> = vec![
            "rust async runtime tokio".into(),
            "the giant panda is a bear native to china".into(),
            "kubernetes pod scheduling".into(),
            "I went to the store yesterday".into(),
            "lifecycle of a panda cub in the wild".into(),
        ];
        let out = r.rerank("what is a panda?", &docs, 5).unwrap();
        // Most relevant should be one of the panda docs (idx 1 or 4).
        assert!(
            matches!(out[0].idx, 1 | 4),
            "expected panda doc on top, got idx={}",
            out[0].idx
        );
    }
}
