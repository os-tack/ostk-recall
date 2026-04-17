//! In-process static-text embedder backed by model2vec-rs.
//!
//! Default model: `minishlab/potion-retrieval-32M` (512-dim).
//! Alt: `minishlab/potion-base-8M` (256-dim). Switching models requires
//! re-creating the corpus table — dim is fixed at table-create time.

use std::path::{Path, PathBuf};

use model2vec_rs::model::StaticModel;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("model load: {0}")]
    Load(String),
}

pub type Result<T> = std::result::Result<T, EmbedError>;

/// Thin wrapper around `model2vec_rs::StaticModel`.
///
/// `model2vec-rs` 0.1.x has no public accessor for embedding dim, so we probe
/// once at load time by encoding a tiny sentinel.
pub struct Embedder {
    model: StaticModel,
    dim: usize,
    model_id: String,
}

impl Embedder {
    /// Load a model by repo id (e.g. `"minishlab/potion-retrieval-32M"`).
    ///
    /// model2vec-rs will search `$HF_HOME`/local caches and fall back to
    /// `HuggingFace` Hub download. For bundled-first behavior, callers can set
    /// `HF_HOME` to the corpus's `models/` dir before calling.
    pub fn load(model_id: &str) -> Result<Self> {
        Self::load_with_subfolder(model_id, None)
    }

    pub fn load_with_subfolder(model_id: &str, subfolder: Option<&str>) -> Result<Self> {
        tracing::info!(model = model_id, "loading embedder");
        let model = StaticModel::from_pretrained(model_id, None, None, subfolder)
            .map_err(|e| EmbedError::Load(format!("from_pretrained({model_id}): {e}")))?;

        let probe: Vec<String> = vec!["dim probe".into()];
        let probe_vec = model.encode(&probe);
        let dim = probe_vec
            .first()
            .map(Vec::len)
            .ok_or_else(|| EmbedError::Load("probe encode returned empty".into()))?;

        tracing::info!(model = model_id, dim, "embedder ready");
        Ok(Self {
            model,
            dim,
            model_id: model_id.to_string(),
        })
    }

    /// Point `HF_HOME` at a corpus-local models dir before constructing.
    /// Returns the path that was set, so callers can log it.
    pub fn pin_hf_home(corpus_root: &Path) -> PathBuf {
        let models = corpus_root.join("models");
        // SAFETY: HF_HOME is read once by the model loader; we set it before
        // calling model2vec-rs. Single-threaded at the point of call in init.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("HF_HOME", &models);
        }
        models
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Encode a batch of strings. Order is preserved.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        let owned: Vec<String> = texts.iter().map(|&s| s.to_string()).collect();
        self.model.encode(&owned)
    }

    /// Encode one string.
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let batch = self.encode_batch(&[text]);
        batch
            .into_iter()
            .next()
            .expect("batch of one yields one embedding")
    }
}

/// Cosine similarity between two equal-length vectors. Pure helper so callers
/// don't have to pull in ndarray. Returns 0.0 if either side is zero-norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identity() {
        let v = vec![1.0f32, 2.0, 3.0];
        let c = cosine_similarity(&v, &v);
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_norm() {
        let a = [0.0f32, 0.0];
        let b = [1.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < f32::EPSILON);
    }

    /// End-to-end encode test. Requires network on first run to fetch the
    /// model. Gated behind `OSTK_RECALL_E2E=1` so CI runs offline by default.
    #[test]
    #[ignore = "requires network / model download — set OSTK_RECALL_E2E=1 to run"]
    fn loads_real_model_and_encodes() {
        if std::env::var("OSTK_RECALL_E2E").is_err() {
            return;
        }
        let emb = Embedder::load("minishlab/potion-base-8M").expect("load");
        assert_eq!(emb.dim(), 256);
        let v = emb.encode("hello world");
        assert_eq!(v.len(), 256);
    }
}
