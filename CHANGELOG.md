# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- **Autonomous-salience scorer, axis 1 (specificity) — behind a flag.** New
  `[salience]` config block with a master `scorer_v2` flag (default **OFF**, so
  `compute_score_parts` stays bit-identical to v1 until opted in). The
  specificity axis computes each thread handle's co-occurrence entropy across
  the distinct source documents its evidence links resonate with
  (`specificity = 1 − H/H_max`) and multiplies it into the idle floor — the
  principled, continuous replacement for the binary `is_stop_handle` cliff. A
  handle that resonates with *everything* (diffuse, high-entropy — `re-read`,
  `top-level`) self-demotes toward the unresonant baseline with **no hand-list**;
  a concentrated concept (`cognitive-memory`, `dereference-or-void`) is
  preserved. Factors are precomputed once at boot (one evidence-graph scan +
  corpus join) and carried scope-independently on the attention runtime; the
  scorer reads them by value with zero added score-time cost. Value (axis 3) and
  negative-transfer (axis 2) fields ship as the neutral identity for now.
  `ScoreAttribution` gains `specificity`/`value`/`neg_penalty` (serde-default
  neutral, so older clients still parse). The curated `[weaver] stop_handles`
  set stays wired as a safety net and A/B control.
- **Autonomous-salience axis 4 (self-audit) — salience-health metrics on
  `recall_stats`.** The surfacer can now see its own drift. A new
  `recall_stats.salience_health` block (pull leg, always present when an
  attention surface is wired; omitted otherwise so old MCP clients keep
  parsing) reports four decomposed metrics over the active surface: (1)
  normalized score-share **entropy** + a cheap score-spread companion (the
  collapse / flat-ribbon alarm); (2) the **curated:autonomous ratio** + the
  hand-list handles still doing load-bearing work (the receipt that the
  stop-set is becoming redundant); (3) **surfaced-but-never-used** handles
  (`LensIncluded` ≥ N with zero distinct used queries), via the shared
  `surfaced_vs_used` ledger join, with `unattributable` handles flagged
  separately rather than miscounted; (4) **active-vs-decided drift** (Jaccard
  distance between the active surface and the recently-judged-salient concept
  set) plus the `J \ A` *forgotten* tail — handles the operator judged salient
  that the surfacer dropped (the dangling-anchor failure mode). When a
  threshold breaches, a compact loud-on-failure verdict is also pushed into
  tool responses (the `stale_ingest` pull+push pattern). The compute path is
  **pure observation, NOT flag-gated** — it watches both the v1 and `scorer_v2`
  scorers, so its metrics double as the A/B scoreboard. Thresholds live in a
  `[salience.health]` config block (live-tunable); the result is cached on a
  30s TTL. Every metric is decomposable (carries its `why`).

## [0.8.3] - 2026-06-15
### Changed
- `[weaver] stop_handles` default: added `non-blocking` — a generic adjective
  that surfaces a grab-bag of unrelated incidental uses (status checks, review
  nits, logging writers, TUI channels) with no unifying concept. Coherent noise
  the derived frequency gate misses (rate ~0.19, above the threshold).

## [0.8.2] - 2026-06-15
### Added
- **`[weaver] stop_handles` — curated thread-handle stop-set.** Handles listed
  here are forced to `is_stop` in the attention scorer (floor clamped to the
  unresonant baseline, off-diagonal lift zeroed) so the `IdleCurator` decays
  them to dormant and keeps them there, and the weaver/observer never mint or
  re-promote them. This complements the *derived* frequency stopword classifier
  (`is_stop_handle`), which only catches high-frequency / **low**-resonance
  generic vocab — harness mechanism vocabulary (`turn-digest`, `squad-lead`,
  `re-run`, `post-restart`, …) is "coherent noise" (high-frequency **and**
  high-resonance) that evades it and dominates the active surface, and a manual
  tension demote is recomputed straight back to Active by the curator. Seeded
  with the common harness / coordination / test-plumbing handles; override via
  the `[weaver]` block. Gate, don't delete: stopped handles stay reachable via
  `recall`/`thread_list`. Takes effect on `serve` restart.

## [0.8.1] - 2026-06-14
### Fixed
- **Fresh corpora had no FTS index, so hybrid recall failed on every query**
  (`Cannot perform full text search unless an INVERTED index has been
  created`). `ensure_corpus_indexes` runs on every scan and (per its own doc
  comment) is meant to backfill the scalar **and** FTS indexes, but it only
  ever ensured the scalar ones — the `text` inverted index was built solely by
  `init`, which runs against a still-empty table, so the index covered zero
  rows and nothing rebuilt it once data landed. Existing corpora only had a
  working FTS index because they had been re-`init`'d over populated data at
  some point. A corpus created by `init` → `scan` (the normal path, and the
  v0.8.0 rebuild) was left with no inverted index. Scan now calls
  `ensure_fts_index` too, so the first populated scan builds it; the call is
  idempotent (no-op once a `text` index exists).

## [0.8.0] - 2026-06-14
### Added
- **`recover-orphans` command** — rebuild the corpus's irreplaceable rows after
  a backup → wipe → rescan. Diffs `chunk_id`s against a backup corpus
  (`--from`) and appends the backup-only rows — chunks whose source files were
  rotated away (e.g. expired Claude transcripts) for which the backup is the
  only surviving copy — copying their stored vectors (no re-embed, no merge
  scan), so recovery is O(rows). Reads facets in either the new or legacy form;
  `--dry-run` reports the diff without writing.
### Fixed
- **`optimize` (`OptimizeAction::All`) failed on large corpora, so the corpus
  never compacted** and grew without bound (observed: 17 GB / 18k versions /
  11k fragments / 350+ un-merged index deltas). Two columns defeated lance's
  miniblock structural encoder: oversized `text` chunks (multi-hundred-KiB tool
  outputs) tripped the 32 KiB chunk assertion, and the sparse `facets` list
  overflowed the u16 rep/def buffer-length field. Fixed by pinning `text` to
  the `fullzip` encoding and storing `facets` as a flat `Utf8` JSON column.
- **`.ostk` substrate stopped ingesting incrementally** (→2040). The watcher
  noise filter blanket-vetoed `.ostk`, starving the `journal`/`decisions`/
  `issues` signal files; now only `.ostk/vfs` (the loopback NFS mount) is
  vetoed, and `walk_filtered` no longer crosses that mount
  (`same_file_system`).
### Changed
- **Corpus on-disk format**: `facets` is now a flat `Utf8` JSON array instead of
  `List<Utf8>`. Reads stay backward-compatible (legacy `List` corpora and
  backups still decode), but a corpus written by 0.8.0 is **not** readable by
  0.7.x, and existing corpora need a rescan to benefit from the `optimize` fix.
- **Dependencies**: `lancedb` 0.29 → 0.30 (lance 7.0).

## [0.7.2] - 2026-06-03
### Added
- **Relational substrate — ambient salience-gated concept-growth (Phase 1 + 2).**
  The `TurnObserver` now grows the *concept* graph from the live conversation
  stream, not just the thread graph: resonance-gated `co_occurs` edges among
  scope-visible concepts named in a turn (trust-gated to `Active`/`Proposed`
  endpoints), plus `Proposed` node minting for novel kebab terms that recur
  across resonant turns (`occurrence ≠ salience`). Recurrence + per-session
  mint caps persist across scan triggers via a shared `ConceptGrowthRuntime`;
  gazetteer lookup is scoped per chunk project. Phase 1 seeds a graph-only
  doc-topology harvest (markdown link graph → `doc` concept nodes/edges, zero
  corpus duplication).
- **Concept-codebook latent adjacency** — latent-similarity union +
  latent→reified promotion, with a config-exposed similarity floor.
- **Codex CLI history ingest** — initial `SourceKind::Codex` scanner source.
### Fixed
- **Ambient memory-lens froze across restart.** The observer's `attend()` was
  routed through the per-chunk *project* scope (correct for the concept
  gazetteer) but that sharded the rolling-attention vector away from the
  project-agnostic `ambient_scope_default()` the lens reads — so the lens saw
  `rolling_vec=None` → `EmptyMind` every tick and never re-rendered. Attention
  is now mirrored into the project-agnostic *aggregate* scope (memory stays
  project-scoped), and chain replay seeds that aggregate on boot so the lens is
  live immediately after a restart.
- **memory-lens resource repopulates across `serve` restart** (seeded from the
  persisted `lens.md` + a forced first re-render).

## [0.7.1] - 2026-06-02
### Added
- **Relational substrate — diffusion read (slice 2).** `relational_lift` lens
  lane + slot: recall diffuses over the salient concept graph.
- **Gazetteer prose mention-linking (slice 4)** — prose mentions of known
  concepts earn evidence links.
- **Automagic promotion (slice 5)** — recurring prose names promote to typed
  nodes; `crystallize` writes stub files for proposed typed nodes.
### Fixed
- **`serve` shutdown is bounded** so Ctrl-C reliably kills the daemon.

## [0.7.0] - 2026-06-01
### Added
- **Concept ledger + `memory_*` MCP façade** — durable concept cards
  (candidate→proposed→active) with mandatory evidence; only `active` concepts
  bias recall.
- **Relational substrate foundations** — edge provenance + earned conductance
  (slice 1); scanner config → typed-node + authored-edge seeding (slice 3).
- **Concept activation on the chain** — lens concept slot + frame "why surfaced".
- Runnable examples for distinct memory concerns; docs for indexing private
  `.ostk/` design docs.
### Fixed
- Alias resolution + merge-forward, coherent evidence, scope, and provenance
  hardening in the concept façade.
- `watch`: add `.pnpm` / `.turbo` to `NOISE_PATH_SEGMENTS`.

## [0.6.0] - 2026-05-31

### Added

- **Structural weave-apparatus gate — tool-call envelopes + `<task-notification>`
  can't seed threads.** Generalizing RT-7 beyond facets: the weaver now treats
  Claude Code tool-call blocks (`block_kind` `tool_use` / `tool_result`) and
  `<task-notification>` monitor events as *structural* apparatus and skips them
  for anchor-matching + emergent proposals, so they can't form degenerate
  high-cohesion threads (post-recovery they dominated the proposal pool).
  `block_kind`/content properties aren't facets, so this is the structural
  counterpart to `[weaver] exclude_facets` (`is_structural_apparatus`).
  Weave-only — the chunks stay in the corpus and remain recall-able (cf.
  `CorpusStore::mark_tool_blocks_stale` to drop them from recall entirely).

- **RT-7 — demux Claude Code multi-agent orchestration off the content
  channel.** `<teammate-message>` envelopes are harness apparatus (the same
  class as `<system-reminder>`): being templated they formed degenerate
  high-familiarity threads (`team-lead`, `teammate-message`) and surfaced in
  the ambient lens. `tag_harness_orchestration` (claude_code parse) now stamps
  them `record_kind=harness_orchestration` — **tag, don't drop**, since they
  carry task-description history; the lens denylist
  (`config::default_lens_exclude_facets`) keeps them out of ambient surfacing
  and the weaver skips them as anchor/proposal candidates, so they can't form
  threads. New `ostk-recall thread delete <handle>` verb (chains `ThreadDelete`)
  to prune apparatus / frequency-promoted threads without forgetting any chunk.

### Fixed

- **`scan --reingest <project>` no longer under-fills the project.** It
  deleted the corpus + `ingest_chunks` rows but left the `ingest_sources`
  cursors, so the follow-up rescan saw every file as unchanged (Tier-1
  metadata check) and skipped re-parsing — restoring only files that had
  changed since the last scan. `scan_reingest` now clears the per-config
  cursors (`IngestDb::clear_source_metadata`) for the project's sources, so
  the rescan re-reads everything from scratch. (Surfaced while completing
  RT-6's `<system-reminder>` purge, which is exactly a "parser changed,
  files unchanged" reingest.)

### Added

- **P11b-full — temporal consolidation: the past weaves the future.** The
  weaver previously only *captured* (wrote evidence, proposed clusters);
  this adds the reverse direction — recency re-strengthens which past stays
  load-bearing, and the unreferenced fades.
  - **Edge activation.** `evidence_links` gains `touch_count` +
    `last_touched_at` (idempotent migration + backfill). `edge_activation(
    similarity, age_days, idle_days, touch_count)` is a derived, componentized
    score: a recency gate (decays toward zero if never re-touched — so the
    lens cannot become a hairball) times a similarity bridge (time-distance
    upweights high-similarity long-range edges; weak matches are not rescued).
    The weaver re-touches an edge on each re-resonance instead of dropping the
    signal. Durable rows are never deleted as activation fades.
  - **`ostk-recall weave --consolidate`.** A coarse consolidation cycle over a
    `--since` window: deep re-weave → anchor↔anchor bridging across canyons →
    near-duplicate thread merge → promotion of recurring high-cohesion
    proposals to durable threads → structural abstraction of deeply-familiar
    stable threads (fold) → idle-time thread fade. Offline operator policy
    (cron/launchd), never coupled to `serve`. See the README "Consolidation
    cadence" section for the scheduling shape.

## [0.6.0-alpha.4] - 2026-05-29

Daemon transport milestone: `serve` becomes a real standalone daemon
that serves MCP to many clients over a cross-platform local socket /
named pipe, with a thin `connect` bridge for stdio clients and an
in-process `--watch`. Fixes the long-standing "only stdio transport is
currently supported" failure (surfaced on Windows).

### Fixed

- **`ostk-recall serve` (no `--stdio`) now actually runs.** It previously
  errored with "only stdio transport is currently supported" on every
  platform — the standalone daemon mode the README documented was never
  wired. (Surfaced on Windows, where nothing else incidentally bound the
  scan-trigger pipe.) `serve` now runs as a long-lived daemon and blocks
  on Ctrl-C (`tokio` gains the `signal` feature).

### Added

- **Single daemon, many clients.** The standalone daemon serves MCP to
  any number of clients over a cross-platform local endpoint (`AF_UNIX`
  socket / Windows named pipe). The MCP resource-notification layer is
  now genuinely multi-client: `ClientId::Network`, per-connection
  outbound channels, lens `resources/updated` fan-out to all subscribers,
  and subscription pruning on disconnect.
- **`ostk-recall connect`** — a thin stdio↔socket bridge (no engine, no
  lock, no scan) so a stdio-only MCP client reaches a running daemon
  instead of spawning its own `serve`.
- **`ostk-recall serve --watch`** — runs the filesystem watcher
  in-process, delivering debounced batches straight to the scan path
  (no socket loopback) and sharing one scan mutex with the trigger
  socket so scans never overlap on the single-writer corpus. The
  standalone `ostk-recall watch` process is retained for the decoupled
  case.

## [0.6.0-alpha.3] - 2026-05-28

The cognitive-memory substrate milestone. ostk-recall gains an ambient,
query-less memory lens, attention-aware recall, a thread/anchor graph, and
the P11 TurnEnd gate + temporal-consolidation base.

> **Pre-release (maintainers / early adopters).** This is a **destructive
> upgrade for v0.5 corpora**: P0 changed the chunk_id formula, and P10
> migration tooling is not yet shipped (nor is the `reset --keep-threads`
> escape hatch). Re-scan from sources after upgrading. The full v0.6.0 is
> reserved for when migration + the enrichment/RC phases land — see
> `rollout.md`.

### Cognitive-memory phases (P0–P11)

- **P0 — identity foundation** (`a0387e2`): physical `source_config_id` +
  chunk_id rekey + Lance ownership column. *Destructive: invalidates v0.5
  corpora.*
- **P1 — facets overlay** (`90345c9`): per-chunk facets, `before` filter,
  `embedding_input_sha256` re-embed/skip dedupe key.
- **P2 — capability probes** (`61ba8d3`, `b37c4f8`).
- **P3A — lens-first rank foundation** (`f32a5e9`): `Candidate` /
  `RankedHit` / `RankEngine` over weighted lane evidence (BM25 + dense, RRF
  preserved), with full per-feature attribution on every hit.
- **P6A — rolling attention EMA** (`8821b8a`): `rolling_vec` EMA blend;
  `effective_vec` priority chain pinned → rolling → transient.
- **P9a-min — MCP resources protocol** (`8b47988`):
  `resources/list|read|subscribe`; writer-task stdio transport.
- **P9b-min — first active lens** (`37d265b`…`69615ea`): the ambient,
  query-less `ostk://memory-lens` resource — diversity-aware allocator,
  token budget, refractory penalty, persisted lens state.
- **P11a — TurnEnd gate** (`a1c71c2`): `IngestOrigin{Bulk,Watch,Synthetic}`;
  the observer + weaver now fire *only* on watched conversation-transcript
  TurnEnds. Fixes the bulk-replay corpus-version explosion and the synthetic
  feedback loop, and turns on live cognition for real.
- **P11b base — temporal consolidation** (`e07b12b`): `weave_window` +
  `ostk-recall weave [--since]` — explicit whole-corpus / windowed weave for
  the bulk content the live daemon deliberately skips. `scan` prints a
  `weave` hint so a fresh scan is never a silent un-woven corpus.

### Scan robustness & version hygiene (`a1c71c2`)

- Raise `RLIMIT_NOFILE` at startup — fixes "too many open files" bulk-scan
  crashes under the macOS 256-fd soft cap.
- Gemini scanner accepts string-or-array message content.
- Decisions scanner accepts both `{key}` and `{decision,status}` schemas.
- Commit batching decoupled from the embed batch (stops Lance version
  explosion); `ostk-recall optimize --aggressive` collapses old versions.
- `init --force` removes orphaned sqlite WAL/SHM sidecars.
- `[lens]` config block wired into the schema.

### Tooling

- Repo-wide `cargo fmt` sweep on edition 2024 (`789da54`).

### Earlier unreleased work (pre-v0.6 attention groundwork)

- `ThreadThreadLink` Rust API on `ThreadsDb` (insert, list-from,
  list-to, delete, count) now chain-emitting AND exposed through MCP
  via the new `thread_evidence` action-routed verb.

### Added

- **Attention-biased recall (Task #6 — the inflection point).**
  Optional `attention_bias: { scope, weight }` rider on `RecallParams`.
  When present, every hit's chunk_id is looked up in the threads graph
  (via the new `ThreadsDb::find_threads_for_chunk` —
  `anchor_chunk_id == chunk_id` ∪ `evidence_links.last_resolved_chunk_id`),
  the max `score_thread(handle)` across those threads becomes the
  hit's `attention_score` (clamped to `[0, 1]`), and the final ranking
  uses `score = base_score + weight * attention_score`. Each hit now
  carries decomposable attribution — `base_score`, `attention_score`,
  `attention_weight` — when bias is applied; without bias, behavior is
  unchanged and the new fields stay `None`.

  This is the architectural transition the post-v0.4.0 hand-off named:
  the substrate stops being a memory store and starts being a thinking
  partner — corpus hits get reordered by what the operator is attending
  to right now, with full attribution visible on every row.

- New `AttentionForwardStore::scope_vector(scope)` trait method (default
  `Ok(None)`) — overridden on `InMemoryAttention` to expose the current
  attention vector. Not used by the v0.4.2 thread-mediated bias path,
  but in place for a future embedding-mediated variant.
- New `ThreadsDb::find_threads_for_chunk(chunk_id)` — lookup union of
  `threads.anchor_chunk_id` and `evidence_links.last_resolved_chunk_id`.
- `RecallParams.attention_bias: Option<AttentionBiasParams>` and three
  new optional fields on `RecallHit` (`base_score`, `attention_score`,
  `attention_weight`). Wire format additive: existing clients see no
  change.
- Three new tests in `crates/mcp/src/server.rs`:
  `bias_lifts_anchored_hit_above_unrelated_hit` (the load-bearing
  case — equal base scores, one anchored, anchored ranks higher),
  `bias_with_zero_weight_preserves_order_and_scores` (weight=0 is
  identity), `bias_with_no_anchor_leaves_score_unchanged` (no
  accidental re-rank for chunks without anchoring threads). Workspace
  gate: 336/0 (was 333/0).

- **`thread_evidence` MCP verb (v0.4.2).** Action-routed surface
  exposing the v0.4.x `thread_thread_links` graph through MCP. Single
  tool, three actions: `add` (from, to, category, note? → id +
  ChainEvent::ThreadLinkAdd), `list` (handle + direction:from|to →
  edges), `delete` (id → removed + ChainEvent::ThreadLinkRemove).
  Mirrors the verb-condensation pattern `thread_query` introduced —
  one tool, caller-chosen action, instead of three near-identical
  verbs. Tool count: 15 (was 14). Three new dispatch tests
  (`thread_evidence_add_list_delete_round_trip`,
  `thread_evidence_rejects_unknown_action`,
  `thread_evidence_missing_required_fields`). Workspace gate: 333/0
  (was 330/0).

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
