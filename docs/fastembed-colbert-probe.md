# fastembed ColBERT probe result (P2 Probe 4)

Date: 2026-05-27.
fastembed version: **5.13.4** (`fastembed = "5"` workspace pin resolved).

## Probe

`crates/embed/examples/fastembed_colbert_probe.rs`.

**Question**: Can fastembed-rs v5 (at the pinned workspace version)
load a ColBERT-family model and emit `Vec<Vec<f32>>` per text (one
vector per token)?

## Result: high-level fastembed path FAILS; pure-Rust path remains viable

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
not expose at the high level.

That does **not** mean P4 requires Python. fastembed-rs v5 exposes a
lower-level `TextEmbedding::transform().into_raw()` path and
`SingleBatchOutput::select_output(...)`, which can surface the raw ONNX
tensor before pooling. For a ColBERT ONNX model, that tensor is the
`[batch, tokens, dim]` shape P4 needs.

The generic fastembed `try_new_from_user_defined` path is still not a
complete ColBERT adapter by itself. Correct ColBERT inference needs
model-specific preprocessing and postprocessing:

- Load `colbert-ir/colbertv2.0` files from HF/cache: `model.onnx`,
  `tokenizer.json`, `config.json`, `tokenizer_config.json`, and
  `special_tokens_map.json`.
- Query preprocessing: insert the query marker token id (`[unused0]`,
  id 1) after CLS and pad short queries with `[MASK]` to ColBERT's
  minimum query length.
- Document preprocessing: insert the document marker token id
  (`[unused1]`, id 2) after CLS, with max length reduced by one so the
  extra marker cannot overflow.
- ONNX inputs: `input_ids`, `attention_mask`, and `token_type_ids` if
  the session declares it.
- Document postprocessing: mask pad/punctuation tokens, L2-normalize
  each token vector, then retain only active token rows.
- Query postprocessing should match FastEmbed Python's ColBERT path so
  MaxSim scores are comparable.

## Decision

Per `p4-multivector.md` (and the explicit contingency in `risks.md`),
the original high-level fastembed-rs contingency is triggered:

> If fastembed-rs has no ColBERT support → P4 drops; jina-turbo
> cross-encoder remains the only post-rank precision feature.
> Maintainer call after probe results land.

**Updated recommendation**: do not introduce a Python sidecar. If P4 is
still desired, implement a narrow, optional Rust-native ColBERT encoder
inside `ostk-recall-embed` rather than waiting for a fastembed-rs catalog
variant.

Suggested shape:

1. Gate it behind an optional feature such as `colbert` /
   `multivector`.
2. Add direct dependencies on the same low-level crates fastembed-rs
   already uses (`hf-hub`, `tokenizers`, `ort`, and `ndarray`), pinned
   compatibly with the current lockfile and rustls policy.
3. Cache model files under the existing corpus-local model cache.
4. Expose two APIs:
   - `encode_passages(&[&str]) -> Vec<Vec<Vec<f32>>>`
   - `encode_queries(&[&str]) -> Vec<Vec<Vec<f32>>>`
5. Feed passage vectors into the Lance side table proven by
   `crates/store/examples/multivector_probe.rs`; use query vectors only
   at rerank time for in-memory MaxSim.

If that feature is not enabled, keep the existing cross-encoder reranker
(`crates/query/src/rerank.rs`) as the precision lane.
