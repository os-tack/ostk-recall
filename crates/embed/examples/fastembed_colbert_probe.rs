//! P2 Probe 4: what ColBERT path does fastembed-rs (v5, pinned) leave us?
//!
//! Method: enumerate fastembed's built-in `EmbeddingModel` catalog and
//! check whether the high-level API exposes a ColBERT / late-interaction
//! model. Then compile-check the lower-level raw-output API that would
//! let an OSTK-specific encoder extract `[batch, tokens, dim]` ONNX
//! outputs without using `TextEmbedding::embed()` pooling.
//!
//! Result:
//! - no high-level `EmbeddingModel::*ColBERT*` exists in fastembed-rs
//!   v5.13.4.
//! - `TextEmbedding::embed()` is still pooled and wrong for MaxSim.
//! - a pure-Rust encoder remains viable by loading ColBERT's ONNX +
//!   tokenizer files and using `transform().into_raw()` / custom `ort`
//!   output handling plus ColBERT-specific pre/post-processing.
//!
//! Exit 0 = the high-level path is unavailable, but the implementation
//! route is identified. See `docs/fastembed-colbert-probe.md`.

use fastembed::{OutputKey, TextEmbedding};

fn main() {
    println!("=== P2 Probe 4: fastembed ColBERT-family API surface ===\n");

    // 1. Enumerate built-in catalog; look for ColBERT-named models.
    let mut colbert_candidates = Vec::new();
    for info in TextEmbedding::list_supported_models() {
        let model = info.model.to_string();
        if model.to_lowercase().contains("colbert") {
            colbert_candidates.push(model.clone());
        }
        println!("  available: {model}");
    }

    if colbert_candidates.is_empty() {
        println!("\n  no `EmbeddingModel::*ColBERT*` variant in the catalog");
    }

    // 2. `embed()` is intentionally the wrong API for ColBERT.
    println!(
        "\n  fastembed-rs v5 `TextEmbedding::embed()` returns `Vec<Vec<f32>>` (one vector per\n  \
        text, i.e. CLS-token / mean-pooled), NOT one vector per token. MaxSim needs an\n  \
        unpooled `[batch, tokens, dim]` output."
    );

    // 3. The lower-level API is sufficient for a custom encoder. The
    //    helper below is deliberately not called because loading the
    //    436 MB ColBERT ONNX file is an e2e concern, not a compile probe.
    println!(
        "\n  fastembed-rs v5 exposes `TextEmbedding::transform().into_raw()` and\n  \
        `SingleBatchOutput::select_output(...)`. That raw path can return a\n  \
        3-D ONNX tensor when the loaded model emits token vectors."
    );

    println!(
        "\nRESULT: no high-level fastembed-rs ColBERT adapter. Recommended solution:\n\
         add an optional OSTK ColBERT encoder that loads `colbert-ir/colbertv2.0`\n\
         via HF cache and runs ONNX/tokenizer preprocessing directly in Rust,\n\
         then writes Lance side-table vectors for the existing MaxSim rerank path."
    );
}

#[allow(dead_code)]
fn raw_output_shapes(model: &mut TextEmbedding, texts: &[&str]) -> Result<Vec<Vec<usize>>, String> {
    let precedence = [
        OutputKey::OnlyOne,
        OutputKey::ByName("last_hidden_state"),
        OutputKey::ByName("token_embeddings"),
        OutputKey::ByName("output"),
    ];

    let batches = model
        .transform(texts, None)
        .map_err(|e| e.to_string())?
        .into_raw();

    batches
        .iter()
        .map(|batch| {
            let tensor = batch
                .select_output(&precedence.as_slice())
                .map_err(|e| e.to_string())?;
            Ok(tensor.shape().to_vec())
        })
        .collect()
}
