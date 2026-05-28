# fastembed ColBERT probe result (P2 Probe 4)

Date: 2026-05-27.
fastembed version: **5.13.4** (`fastembed = "5"` workspace pin resolved).

## Probe

`crates/embed/examples/fastembed_colbert_probe.rs`.

**Question**: Can fastembed-rs v5 (at the pinned workspace version)
load a ColBERT-family model and emit `Vec<Vec<f32>>` per text (one
vector per token)?

## Result: FAIL (per design expectation)

Enumeration of fastembed v5.13.4's `EmbeddingModel` catalog returned
**no `*ColBERT*` variants**:

```
AllMiniLM{L6V2,L12V2}{,Q}, AllMpnetBaseV2,
BGE{Base,Small,Large}ENV15{,Q}, BGE{Small,Large}ZHV15, BGEM3,
ClipVitB32, EmbeddingGemma300M,
GTE{Base,Large}ENV15{,Q},
JinaEmbeddingsV2BaseEN, JinaEmbeddingsV2BaseCode,
ModernBertEmbedLarge,
MultilingualE5{Small,Base,Large},
MxbaiEmbedLargeV1{,Q},
NomicEmbedTextV1, NomicEmbedTextV15{,Q},
ParaphraseMLMiniLML12V2{,Q}, ParaphraseMLMpnetBaseV2,
SnowflakeArcticEmbed{S,M,L,XS,MLong}{,Q}
```

The fastembed `TextEmbedding::embed()` API returns `Vec<Vec<f32>>`
where each inner vector is the *pooled* representation for one input
text (CLS / mean-pool), NOT one vector per token. A ColBERT-shaped
output requires a multi-vector model variant that fastembed v5 does
not expose. The `try_new_from_user_defined` path can in principle
load arbitrary ONNX models, but without the right tokenizer + post-
processing (token-level pooling disabled), even loading
`colbert-ir/colbertv2.0` would produce the wrong output shape.

## Decision

Per `p4-multivector.md` (and the explicit contingency in
`risks.md`):

> If fastembed-rs has no ColBERT support → P4 drops; jina-turbo
> cross-encoder remains the only post-rank precision feature.
> Maintainer call after probe results land.

**Recommendation**: defer P4 (MaxSim rerank feature) entirely until
either:
1. A Rust-native ColBERT encoder lands (track candle, ort, or a
   fastembed PR for `EmbeddingModel::Colbert*`).
2. A sidecar Python process is acceptable operationally (significant
   complexity — own process lifecycle, IPC, model loading).

Until then, the cross-encoder reranker (already in
`crates/query/src/rerank.rs`) remains the precision lane.
