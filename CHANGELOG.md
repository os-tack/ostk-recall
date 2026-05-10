# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v0.1.6 — Remove redundant `crates/serve` proof-of-concept

Cleanup release. No public API change.

### Removed

- `crates/serve/` workspace member. This was a standalone
  `ostk-recall-serve` binary speaking a custom JSON-RPC `recall.fault`
  protocol, built mid-development before the maintainer realized the
  haystack kernel-driver protocol IS MCP and that `ostk-recall serve
  --stdio` (the existing CLI subcommand backed by `crates/mcp`) was the
  correct daemon entry point. The `recall_fault` MCP tool added to
  `crates/mcp` in v0.1.5 superseded the standalone binary; nothing in
  the production stack ever called `ostk-recall-serve`.

The CLI subcommand `ostk-recall serve --stdio` (in `crates/cli`) is
unchanged — that's the production daemon haystack v6.0.0 spawns.

## v0.1.5 — Move serde result/parameter types to `ostk-recall-core` (cut #3 prep)

Preparation for haystack →1846 cut #3 (daemonize the recall surface as a
peer-process driver). Haystack will eventually drop the `ostk-recall-query`
dep entirely and consume only schema-only types via `ostk-recall-core`;
this release moves the types it needs into the right home.

### Added

- `ostk_recall_core::types` — new module containing the serde-only result
  and parameter types previously defined in `ostk_recall_query`:
  `RecallParams`, `RecallHit`, `RecallLinkResult`, `SourceCount`,
  `RecallStats`, `RerankerStats`, `AuditResult`, and the data shape
  `SynthesizedPage` (the runtime `Synthesizer` impl stays in `query`).
- `ostk_recall_core` now re-exports all of these at crate root.

### Changed

- `ostk_recall_query::types` is now a thin re-export shim of
  `ostk_recall_core::types`. No source-level breaking change for
  existing consumers — `ostk_recall_query::{RecallHit, RecallParams, …}`
  still resolves through the re-export.
- `ostk_recall_query::SynthesizedPage` is now re-exported from
  `ostk_recall_core` as well; `ostk_recall_query::synthesis` keeps the
  `Synthesizer` runtime impl that builds them.

### Why

Cut #3 ships `ostk-recall-serve` as a peer-process daemon that owns
`CorpusStore` + `Synthesizer` + `Reranker` + the lance/lancedb runtime.
Haystack dispatches `recall` / `recall_search` / `recall_outline` opcodes
over the kernel sock and only needs to deserialize results — for which
the schema-only types live in `core`. This is the minimum upstream change
that lets haystack switch its dep from `ostk-recall-query` to
`ostk-recall-core` and drop the entire lance/lancedb/datafusion/arrow
chain (~30–42 MB binary).

### Compatibility

Additive: no public API removed. `ostk-recall-{cli,mcp}` consumers see
no change. Workspace-internal code paths still resolve through the same
`use ostk_recall_query::…` imports.

### Also included in v0.1.5

Two commits from parallel work landed on `main` between v0.1.4 and
v0.1.5 and ship as part of this release:

- **`feat(cli): file-watcher subcommand drives scan-trigger socket`**
  (`09cc7fd`). Adds `ostk-recall watch` — a thin watcher that pokes the
  running `serve` daemon's scan-trigger socket whenever a debounced
  batch of events lands under any configured source path. Uses
  `notify-debouncer-full`. Opt-in via `[watch].enabled = true`.
- **`feat(watch): per-platform debounce defaults`** (`11f4d58`).
  Lifts the debounce window into a `cfg!()` block (Linux/inotify 800 ms;
  other platforms tuned to match their fs-event characteristics).

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
