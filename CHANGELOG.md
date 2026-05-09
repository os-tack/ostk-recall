# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v0.1.4 — Feature-gate Reranker; downstream haystack drops ORT

### Changed
- **`Reranker` (and its `fastembed` / ORT dependency) now sits behind a
  cargo feature `reranker`, default-on.** Consumers that set
  `default-features = false` on `ostk-recall-query` drop the entire
  ORT runtime + fastembed graph from their build (haystack v5.2.0 is
  the immediate beneficiary — its v5.0.x saga was driven by ORT's
  glibc / cross-build constraints).
- Trait surface unchanged: `RerankerLike`, `RerankerError`, and
  `RerankerStats` are always-on so default-features-off callers can
  still type their reranker-injection code paths and inject their own
  `Arc<dyn RerankerLike>` implementations.
- Workspace consumers (`ostk-recall` CLI, `ostk-recall-mcp`) consume
  default features and are unaffected — fastembed remains in their
  dependency tree exactly as before.

Closes →1847.

## [0.1.3] - 2026-05-09 — switch fastembed to rustls (downstream cross-musl unblock)

### Changed
- **Workspace `fastembed = "5"` switched to `default-features = false` with
  `["hf-hub-rustls-tls", "ort-download-binaries-rustls-tls", "image-models"]`.**
  Fastembed's defaults pull `hf-hub-native-tls` and
  `ort-download-binaries-native-tls`, which transitively activate hf-hub's
  `native-tls` feature and ort's `tls-native` feature — both drag
  `openssl-sys` and `native-tls` into the dependency tree via `native-tls`.
  Cargo features only union (never subtract), so a downstream-only override
  cannot remove these once fastembed's defaults are in scope. Setting
  `default-features = false` here keeps openssl-sys out of every downstream's
  cross-musl build (haystack v5.0.7 is the immediate consumer). No API or
  behavior change — fastembed exposes parallel `*-rustls-tls` features that
  resolve the same ort + hf-hub deps with rustls instead of native-tls.

## [0.1.2] - earlier

### Added
- Initial workspace with core, embed, store, scan, pipeline, query, mcp, cli crates.
- Six source scanners: markdown, code, claude_code, file_glob, zip_export, ostk_project (composite).
- LanceDB-backed corpus with Tantivy FTS index.
- DuckDB-backed ingest manifest and audit events store.
- MCP stdio server exposing recall, recall_link, recall_stats, recall_audit tools.
- Hybrid dense + BM25 retrieval with RRF fusion.
