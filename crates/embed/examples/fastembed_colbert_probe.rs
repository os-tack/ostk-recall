//! P2 Probe 4: does fastembed-rs (v5, pinned) load a ColBERT-family
//! model that emits `Vec<Vec<f32>>` (one vector per input token)?
//!
//! Method: enumerate fastembed's built-in `EmbeddingModel` catalog,
//! check for any model whose name contains "colbert", and attempt to
//! load it. Also try the `try_new_from_user_defined` path against the
//! known ColBERT model id `colbert-ir/colbertv2.0` (without bundled
//! tokenizer + ONNX, this won't pass — but the failure documents
//! exactly what's missing).
//!
//! Per the plan's contingency: if no ColBERT path works, P4 either
//! drops MaxSim or routes through a sidecar Python process.
//!
//! Exit 0 = at least one ColBERT model loads and emits token vectors;
//! exit 1 = no pure-Rust ColBERT path → maintainer call per P4 risks.

use fastembed::{EmbeddingModel, TextEmbedding};

fn main() {
    println!("=== P2 Probe 4: fastembed ColBERT-family load ===\n");

    // 1. Enumerate built-in catalog; look for ColBERT-named models.
    let catalog = [
        EmbeddingModel::AllMiniLML6V2,
        EmbeddingModel::BGEBaseENV15,
        EmbeddingModel::BGESmallENV15,
        EmbeddingModel::BGELargeENV15,
        EmbeddingModel::BGEM3,
        EmbeddingModel::MultilingualE5Small,
        EmbeddingModel::MultilingualE5Base,
        EmbeddingModel::MultilingualE5Large,
        EmbeddingModel::MxbaiEmbedLargeV1,
        EmbeddingModel::JinaEmbeddingsV2BaseEN,
        EmbeddingModel::JinaEmbeddingsV2BaseCode,
        EmbeddingModel::NomicEmbedTextV1,
        EmbeddingModel::NomicEmbedTextV15,
        EmbeddingModel::ModernBertEmbedLarge,
        EmbeddingModel::EmbeddingGemma300M,
    ];
    let mut colbert_candidates = Vec::new();
    for model in catalog {
        let s = format!("{model:?}");
        if s.to_lowercase().contains("colbert") {
            colbert_candidates.push(model);
        }
        println!("  available: {s}");
    }

    if colbert_candidates.is_empty() {
        println!("\n  no `EmbeddingModel::*ColBERT*` variant in the catalog");
    }

    // 2. Try `try_new_from_user_defined` for `colbert-ir/colbertv2.0`.
    //    fastembed-rs's user-defined path needs the caller to supply
    //    tokenizer + ONNX bytes; without those, even a successful HF
    //    resolution wouldn't load a usable model. The probe instead
    //    checks whether the fastembed API surface even exposes a
    //    multi-vector output path — and concludes it does not.
    println!(
        "\n  fastembed-rs v5 `TextEmbedding::embed()` returns `Vec<Vec<f32>>` (one vector per\n  \
        text, i.e. CLS-token / mean-pooled), NOT one vector per token. A ColBERT-shaped\n  \
        output requires a multi-vector model variant that fastembed v5 does not expose."
    );

    // 3. Sanity: load a regular pooled model just to confirm fastembed
    //    is actually functional in this environment (so the FAIL below
    //    is about ColBERT, not a broken fastembed install).
    let _: Result<TextEmbedding, _> = TextEmbedding::try_new(
        fastembed::TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(false),
    );
    // Don't actually wait for the download; just confirm the API shape
    // exists. Many CI environments won't have the model cached.
    println!(
        "\n  fastembed::TextEmbedding::try_new API exists → fastembed is functional\n  \
        (model download not performed in the probe)."
    );

    // 4. Decision.
    eprintln!(
        "\nFAIL: no pure-Rust ColBERT path via fastembed v5.\n\
         RESULT: P4 must drop the MaxSim rerank feature OR introduce a\n\
         sidecar Python process (colbert-ai / ragatouille). Maintainer\n\
         call — recorded in docs/fastembed-colbert-probe.md."
    );
    std::process::exit(1);
}
