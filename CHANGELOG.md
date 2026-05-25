# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- `ThreadThreadLink` Rust API on `ThreadsDb` (insert, list-from,
  list-to, delete, count) now chain-emitting. MCP surface still
  deferred — will land alongside the v0.4.x verb consolidation pass
  (`thread_evidence`).

### Added

- **Chain emission for thread → thread evidence edges.** Two new
  `ChainEvent` variants — `ThreadLinkAdd { from, to, category, note,
  ts }` and `ThreadLinkRemove { id, ts }` — emitted by
  `add_thread_thread_link` and `delete_thread_thread_link`
  respectively. Wire format covers both directions via `kind_str`,
  `to_payload`, and `from_row`; chain replay path acknowledges the
  variants (graph tier is itself durable, so the in-memory score tier
  ignores them — see comment in `replay_chain_into_attention`).
  Closes the v0.4.0 deferral: every link mutation now has an audit
  row, so a future recovery utility can reconstruct the graph from
  `chain_log` alone.
- Four new tests in `crates/store/src/threads.rs`:
  `add_thread_thread_link_emits_chain_event`,
  `delete_thread_thread_link_emits_chain_event`,
  `thread_link_chain_survives_reopen` (process boundary),
  `chain_event_thread_link_payload_round_trip` (wire format).
  Workspace gate: 329/0 (was 325/0).

- **Cross-axis backfill for `thread_query` (v0.4.2 honesty upgrade).**
  Every cluster surfaced by any primitive now gets its other two axis
  scores computed from its own membership: density via
  `cluster::mean_pairwise_cosine` over member embeddings, activity via
  per-chunk timestamps + recency decay, novelty via mean per-chunk
  novelty against the project baseline. `composite_score` is no longer
  degenerate for cross-axis questions ("activity ∩ novelty");
  attribution remains decomposable (`contributions.sum() ≈ composite`).
- `ThreadQueryReport.chunk_ids` exposes full cluster membership.
  Plumbed by additively extending `ActivityBurst`, `EmergentReport`,
  `NoveltyReport`, and the corpus-level `ActivityBurst` with
  `chunk_ids` (sorted lexicographically for stable identity).
- `CorpusStore::fetch_timestamps(chunk_ids)` — one-shot ts lookup
  used by the activity backfill.
- `cluster::mean_pairwise_cosine(embeddings)` — public flat-slice
  variant of the existing private cohesion metric, used by density
  backfill.
- One new test in `crates/attention-mcp/src/handlers.rs`:
  `thread_query_cross_axis_backfill_populates_other_axes` — seeds a
  tight activity burst (4 chunks on axis 7) against a baseline (axis
  0), confirms the burst's `density_score` backfills to > 0.8 and
  `novelty_score` > 0.5. Workspace: 330/0 (was 329/0).

- **`thread_query` — the multi-signal verb (v0.4.1).** Single tool
  that runs density, activity, and novelty against the same recency
  window and returns a unified cluster list with per-axis scores, a
  caller-rankable `composite_score`, and a decomposable
  `ThreadQueryAttribution` (`{ axes: [{axis, weight, score,
  contribution}], composite }`). Defaults are honest:
  `composite_weights` is uniform (1/3 each — the only neutral default
  for a combiner), all axis floors are 0.0, all three signals on.
  Caller picks `rank_by` and weights; substrate stops having opinions
  about which axis matters. Supersedes `thread_emergent`,
  `thread_attention`, and `thread_novelty` for callers that want a
  composable answer; the three legacy verbs remain available
  (marked `TODO(verb-condensation)`) and slated for removal at v1.0.0.
- New crate module `crates/attention/src/query.rs` exporting `Axis`,
  `CompositeWeights`, `RankBy`, `ThreadQueryParams`,
  `ThreadQueryReport`, `ThreadQueryAttribution`, and `run_query(...)`.
- New tool schema `tool_thread_query()` + dispatch wiring in
  `crates/attention-mcp`. Tool count: 14 (was 13).
- **Persistent attention on known-handle mentions.** `TurnObserver`
  now mirrors every `record_familiarity_batch` call through to the
  attached `AttentionForwardStore` via `familiarize(scope, handle)`,
  not just on auto-promotion. Closes the parallel gap to v0.4.0 for
  long-running threads: a turn that mentions an existing handle now
  lights up the in-memory score tier immediately, with no
  stale-until-next-boot lag. Without this, `thread_query`'s activity
  axis (v0.4.1) would be quietly dishonest for every thread already in
  the ledger.
- **Hoisted `attend(scope, turn_text)`** to fire once per `observe()`
  call when an in-memory store is wired. Previously the auto-promotion
  path called `attend` per promotion; the hoist makes the
  attention-vector update unconditional and removes the
  redundant per-promotion call. Both code paths (known-handle and
  auto-promotion) now share the same fresh anchor seed when they call
  `familiarize`.
- New test `known_handle_mention_lights_up_in_memory_score` in
  `crates/attention/src/observer.rs` — symmetric to the v0.4.0
  `auto_promotion_lights_up_in_memory_score` test, asserting
  pre/post-observation that `score_thread(handle)` transitions from
  0.0 to > 0 within a single turn.
- `thread_thread_links` table for thread → thread evidence edges.
  Schema: `(id, from_thread, to_thread, category, note, created_at)`
  with `UNIQUE(from_thread, to_thread, category)` and
  `CHECK(from_thread <> to_thread)`. Both endpoints CASCADE on thread
  delete.
- `ThreadThreadLink` struct exported from `ostk-recall-store`.
- `ThreadsDb` methods: `add_thread_thread_link`,
  `list_thread_thread_links_from`, `list_thread_thread_links_to`,
  `delete_thread_thread_link`, `thread_thread_link_count`.
- Use case: the v0.3.0 hand-off cites `abi-as-sovereign-boundary`,
  `three-time-scales`, and `fade-is-concentration` as evidence —
  the data shape now exists. v0 is hand-edited only; weaver
  auto-proposal of thread → thread edges deferred.

### Tests

- Five new tests in `crates/store/src/threads.rs`: round-trip,
  unique constraint, self-loop rejected, cascade delete on either
  endpoint, delete-by-id. Workspace: 315/0 (was 310/0).

### Internal

- Chain integration deferred: `add_thread_thread_link` does not yet
  emit a `ChainEvent`. v0 rows are recoverable from the table
  directly, not from chain_log. A future `ChainEvent::ThreadLinkAdd`
  variant + replay handler is tracked alongside verb consolidation.

## v0.4.0 — Persistent attention (TurnObserver → InMemoryAttention)

Closes the v0.3.0 hand-off gap where auto-promoted threads existed in
`threads.sqlite` but had `score_thread(handle) == 0` in the running
`InMemoryAttention` — the curator's stale-touch grace expired after
one tick (~60s) and demoted the freshly-promoted thread to Dormant.

The hand-off named this v0.4.1 (after `thread_query` as v0.4.0); we
flipped the order because this work is plumbing the multi-signal verb
will load-bear on. Cleaner to ship it first and let `thread_query`
build on an honest in-real-time activity axis.

### Added

- `TurnObserver::with_attention(Arc<dyn AttentionForwardStore>)` —
  builder method that attaches the in-memory score tier to the
  observer. Optional; observers without an attached store work
  exactly as before (test fixtures unchanged).
- `spawn_ambient_daemons` (cli) now takes the attention store as a
  parameter and threads it through to the observer it spawns. Both
  `serve` call sites already had `ctx: &ServeContext`, which carries
  the `Arc<InMemoryAttention>`.

### Changed

- On every successful auto-promotion, `TurnObserver::observe` now
  calls `attend(scope, turn_text)` and `familiarize(scope, handle)`
  on the in-memory store. Failures are logged but don't abort the
  durable observation (best-effort).
- The `TODO(persistent-attention)` comment in `observer.rs` is
  replaced with a doc paragraph describing the wiring.

### Tests

- New: `observer::tests::auto_promotion_lights_up_in_memory_score`.
  Builds an observer with `with_attention(InMemoryAttention::new())`,
  promotes a stub at `PROMOTE_MIN_OCCURRENCES`, asserts
  `score_thread(handle) > 0` *immediately* (no waiting on chain
  replay) and that `surface()` includes the new thread.
- Workspace: 310/0 (was 309/0).

## v0.3.1 — No-baked-filters discipline (constants → tool args)

Applies the v0.3.0 hand-off's discipline rule ("any threshold not derived
from a parser-level property is a tool arg") to the three thread surfaces.
Purely additive on the MCP wire — every prior call site keeps working.

### Added

- `thread_emergent` accepts two new optional args:
  - `cohesion_threshold: f32` (default `0.82`, was the baked
    `cluster::EMERGENT_THRESHOLD`)
  - `min_neighbours: usize` (default `2`, was the baked
    `cluster::MIN_NEIGHBOURS_IN_CLUSTER`)
- `thread_novelty` accepts two new optional args:
  - `min_mean_novelty: f32` (default `0.0` — filter off, permissive per
    discipline rule; pass `0.3` to recover pre-v0.3.1 behavior)
  - `min_cluster_size: usize` — naming-consistency alias for the
    existing `min_cluster` arg (both accepted; `min_cluster_size`
    matches `thread_emergent`)

### Changed

- **`thread_novelty` MCP default behavior**: the post-cluster
  `mean_novelty >= 0.3` filter is now off by default. Callers may see
  more clusters surfaced, including coherent-but-low-novelty ones.
  This is intentional — the substrate stops encoding "what's
  interesting" and lets the caller decide. Pass
  `min_mean_novelty: 0.3` to restore the v0.3.0 floor.
- `thread_novelty` response `params` block reports `min_cluster_size`
  (uniform with `thread_emergent`) and adds `min_mean_novelty`.
- Library defaults preserved: `surface_default` in `novelty.rs` and
  `discover_default` in `emergent.rs` still pass the historical
  constants. Only the MCP handler defaults moved.

### Internal

- `discover_and_surface` (emergent.rs) takes a new
  `min_in_cluster_neighbours` parameter and now calls
  `find_clusters_with` directly (was wrapping `find_clusters` then
  re-filtering).
- `surface_novelty` (novelty.rs) takes a new `min_mean_novelty`
  parameter; `0.0` disables the post-cluster filter entirely.

### Tests

- New test `novelty::tests::min_mean_novelty_zero_keeps_clusters_filter_drops`
  proves the arg propagates: `0.0` floor surfaces ≥1 cluster (and
  reveals that the historical `0.3` was hiding baseline-aligned
  clusters); `1.5` floor returns empty. Workspace: 309/0 (was 308/0).

## v0.1.7 — Path-aware incremental scan (EPIC gh#8)

Closes the loop between editor save and `recall_fault`: the watcher now
emits debounced changed paths over the trigger socket, and `serve`
dispatches per-path ingest instead of full corpus re-scan. Save → 200 ms
debounce → ~50 ms scan of the changed file → corpus updated. Kernel
agents calling `recall_fault` from any session see the fresh chunk
within ~250 ms of the save.

### Added

- `Scanner::discover_paths(cfg, paths)` trait method on
  `ostk_recall_core::Scanner`. Default impl walks `discover()` and
  filters by component-wise path match (`q == p || q.starts_with(p)`),
  so directory-yielding scanners (`OstkProjectScanner`-style) work
  correctly even without an override. Per-scanner overrides on
  markdown / code / file_glob / ostk_project / zip_export skip the
  walk for O(|paths|) instead of O(|tree|). `claude_code` and
  `gemini` defer to default-filter — append-only JSONL needs the
  per-file offset cursor follow-up.

- `Pipeline::scan_paths(sources, paths)` — async per-path ingest. Groups
  paths by `[[sources]]` they fall under (a single path can match
  multiple sources, e.g. `code` + `markdown` rooted at the same
  directory; both fire). Honors `RetentionPolicy` for delete events:
  `Delete` purges corpus rows, `Stale` marks chunks inactive, `Keep`
  tombstones ledger only.

- `commands::scan_paths` CLI wrapper mirroring `commands::scan` for
  remote callers (the trigger-socket dispatch path).

- `WatchConfig::mode: WatchMode` (`Legacy | Incremental`, default
  `Legacy`). When `Incremental`, the watcher writes line-delimited
  matched paths over the trigger socket and `serve` dispatches
  `Pipeline::scan_paths`. Empty body = legacy "scan all" remains the
  fallback in both directions, so old↔new mixes degrade cleanly to
  full re-scan instead of silent under-indexing.

- `IngestDb::tombstone_chunks_by_path` — public helper used by the
  delete-event branch in `Pipeline::scan_paths`.

### Changed

- `kick_trigger_socket` in `crates/cli/src/commands.rs` now writes
  paths + half-shutdown to signal EOF when called with a non-empty
  slice. The `kick_trigger_socket(_, &[])` shape is the explicit
  legacy poke (empty body), preserving the v0.1.6 wire shape.

- Scan-trigger listener (`run_socket_listener`) now reads the connected
  stream up to EOF (2 s deadline, 64 KiB cap) and dispatches to
  `Pipeline::scan_paths` when paths are present, falls back to
  `scan(...)` on empty body, read error, timeout, oversize frame, or
  non-UTF-8 byte. Legacy fallback is load-bearing in 5 places — any
  malformed frame degrades to full re-scan, never silent
  under-indexing of the changed file.

### Migration

`[watch].mode = "incremental"` is opt-in for this release. Existing
configs without a `mode` field default to `Legacy` (current behavior).
The default flips to `Incremental` in a follow-up minor version after
field bake-in.

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
