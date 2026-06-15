# R4 — Self-Audit Metrics for Salience Health

> Team `salience`, axis #4. Research only. Maps the precedent, recommends
> concrete metrics, says where they compute and how they surface. Modeled
> on the `stale_ingest` probe (`crates/mcp/src/server.rs` L744–841).

## TL;DR

Axis #4 is `dereference-or-void` + `projection-truth` applied to the
surfacer's own state: *the substrate cannot currently see its own salience
drift.* The optimize bug and the dangling anchors were found by accident,
not surfaced. Build a **computed health block** — modeled exactly on
`stale_ingest` — that reports four metrics over the live active surface:

1. **Surface entropy** `H(active_surface)` — is the surface spread across
   distinct concepts, or collapsed onto a few coherent-noise handles?
2. **Curated:autonomous ratio** — what fraction of the surface is held up
   by the curated `stop_handles` / hand-authored anchors vs. earned
   autonomously? (Health = *low* curated dependence.)
3. **Surfaced-N-times-never-used** flags — handles that keep surfacing
   (`LensIncluded`) but never get used (`ExplicitRecall` / `OperatorSelected`
   / cited in a decision). This is the click-through-loop's negative tail.
4. **Active-vs-decided drift** — divergence between the active thread set
   and the handles that recently showed up in `ostk_decision` /
   `ConceptAccessed` / `ConceptPromoted`. Health = the active set tracks
   what the operator actually judged salient.

**Surface it the way `stale_ingest` is surfaced**: a computed block, cached
~30s, attached as a new `salience_health` field on `recall_stats`
(always-on, truthful pull) AND conditionally pushed into responses *only when
unhealthy* (loud-on-failure). Behind no config flag for the read path — it is
pure observation — but its **thresholds** live in config so they can be tuned
without a recompile, same as `[watch].freshness_threshold_secs`.

The deeper point: this axis is the **A/B harness's instrument panel**. The
THESIS demands the new scorer (specificity × value × recency) be A/B'd
against the old one on the live thread set. These four metrics ARE the
scoreboard for that A/B — entropy up, curated dependence down, never-used
tail shrinking, drift shrinking = the new scorer working. So R4 is not just
"add some gauges"; it is the verification surface for R1–R3.

---

## 1. The precedent: `stale_ingest` is the template

`crates/mcp/src/server.rs`:

- **`compute_stale_ingest()` (L767–841)** — a pure function that derives a
  health signal (journal-mtime vs. newest-ingested-row divergence per
  watched project) and returns `Some(json!{...})` only when something is
  wrong, `None` when healthy. Carries a human-readable `hint` and the
  `threshold_secs` it judged against.
- **`stale_ingest()` (L750–765)** — a 30s TTL cache wrapper (`Mutex<Option<
  (Instant, Option<Value>)>>`) so the probe doesn't recompute per request.
- **Two surfacing legs**:
  - **Pull (unconditional, always truthful):** `recall_stats` carries
    `audit_newest_ts` *every time* (`crates/query/src/stats.rs` L33–43;
    field on `RecallStats` in `crates/core/src/types.rs` L266–289). The
    live probe today returns the full `stale_ingest` block inline on
    `recall_stats` — confirmed by the live probe below.
  - **Push (conditional, loud-on-failure):** server.rs L719–731 injects a
    `stale_ingest` key into *any* tool response when (and only when) the
    probe fires. The INVARIANT comment (L719–725) is load-bearing doctrine:
    *"pull always-truthful, push loud-on-failure. Do not optimize stats to
    conditional too; that recreates →1947's silent-stale with extra steps."*
- **Honesty guard (L770–776):** the probe documents its own blind spot
  (only covers `[watch].projects`-allowlisted sources) and pairs with a
  complementary boot-time WARN. *"change one, change both."*

**Every salience-health metric should inherit this shape:** a pure compute
fn, a TTL cache, an unconditional pull field + a conditional loud push,
self-documented blind spots, tunable thresholds in config.

The other precedent is `crates/store/src/activation.rs` — the
`memory-activation-frame` slice. Its doctrine is exactly axis #4's:
*"rank what matters now, and expose the math"* (`abi-as-sovereign-boundary`:
argue with the math, not the vibe). Its `ConceptWhy` (L71–89) decomposes an
activation score into named contributions. **Self-audit metrics must
likewise be decomposable** — never a bare scalar; always the components that
produced it, so a regression can be argued against the numbers.

---

## 2. Live substrate probe (what's surfaced today)

`recall_stats` (live): the response already carries the `stale_ingest` block
(two stale projects: `osteak` 63-day lag, `ostk-site` never-ingested), plus
`watch`, `audit_newest_ts`, `by_source`. There is **no salience-health
field of any kind today** — the surfacer's own state is invisible at the
stats surface. That is the gap R4 fills.

`attention_surface` (live, top of the active surface, `limit=40`):

| handle | score | mentions | resonance | resonance_count | curated? |
|---|---|---|---|---|---|
| attention-is-sovereign | 1.268 | 30 | 0.268 | 30 | (concept) |
| abi-as-sovereign-boundary | 1.267 | 28 | 0.268 | 28 | (concept) |
| memory-lens | 1.244 | 48 | 0.244 | 47 | (concept) |
| cognitive-memory | 1.239 | 104 | 0.239 | 93 | real ✓ |
| resonance-gated | 1.223 | 49 | 0.223 | 49 | (concept) |
| re-read | 1.213 | 24 | 0.213 | 24 | **noise?** |
| haystack-boot | 1.208 | 21 | 0.209 | 21 | **noise?** |
| kernel-cpu | 1.208 | 39 | 0.209 | 39 | (concept) |
| trust-root | 1.204 | 66 | 0.204 | 66 | (concept) |
| ... | ... | ... | ... | ... | ... |
| pre-existing | 1.072 | 67 | 0.072 | 67 | **noise** |
| follow-up | 1.072 | 51 | 0.072 | 51 | **noise** |
| system-reminder | 1.047 | 14 | 0.318 | 14 | **noise** |
| ostk-cache | 1.139 | 206 | 0.139 | 198 | real ✓ |

Three things this surface *screams* that no metric currently reports:

1. **The score band is tiny and flat.** 38 of 40 pages sit in `[1.02, 1.27]`
   — a `0.25`-wide ribbon. Score is dominated by `familiarity_floor(
   resonance_count)` (the frequency proxy), so high-mention handles bunch at
   the top regardless of whether they discriminate. This is *visible* as low
   inter-page score variance — a cheap, scorer-agnostic health proxy.
2. **`off_diagonal_lift` is `0.0` for all but one page** (`haystack-sliver-
   backport`, the lone bridge that earned the reserved slot at score `0.74`).
   The surface is almost entirely floor-driven; the bridge mechanism is
   doing nothing for 39/40 rows. A health metric should report *what fraction
   of the surface's score comes from the floor vs. resonance vs. lift.*
3. **Confirmed coherent-noise is on the surface right now**: `re-read`,
   `pre-existing`, `follow-up`, `system-reminder`, `haystack-boot`,
   `best-effort`, `full-suite`, `re-run`-class handles. These are exactly the
   THESIS's "looks like what's happening, a lot" terms. None are flagged.

The live surface is the proof that the gauges are needed and the seed corpus
for testing them.

---

## 3. Recommended metrics (concrete)

All four compute over **the active surface** = the set `surface()` returns
above `ARCHIVE_THRESHOLD` (`crates/attention/src/lib.rs` L1477; curator
`archive_threshold` default `0.1`, `crates/attention/src/curator.rs` L69).
That is the canonical "what's active" set — reuse it; do not invent a second
notion of active.

### Metric 1 — Surface entropy `H`

Shannon entropy of the surface's score-share distribution. For surface pages
`p_i` with score `s_i`, let `w_i = s_i / Σs_j`; then
`H = −Σ w_i·ln(w_i)`, and report **normalized** `H_norm = H / ln(N)` in
`[0,1]` (N = page count). 

- `H_norm → 1`: score spread evenly across many handles (healthy, diverse).
- `H_norm → 0`: a few handles dominate the surface mass (collapse / a
  coherent-noise handle eating the surface).

This is the *macro* mirror of R1's *per-handle* specificity entropy. R1
computes `H` over a handle's co-occurrence distribution to demote a single
non-discriminating term; R4 computes `H` over the *surface's* score
distribution to detect when the surfacer as a whole has collapsed. Same
information-theoretic primitive, different axis — note the shared math so
the implementer factors one `shannon_entropy(&[f32])` helper.

Cheap companion (no embeddings): **score-band variance / spread** =
`max_score − min_score` and stddev over the surface. The live `0.25` ribbon
is a one-number drift alarm.

### Metric 2 — Curated:autonomous ratio

The surfacer holds a curated set (`InMemoryAttention.stop_handles`, the
`forced_stop` flag at `compute_score_parts`; THESIS key-facts §). The
*inverse* — hand-authored waypoint anchors + curated concepts — is the
"immaculate curated tier." Define over the surface:

- `curated_on_surface` = pages whose handle is in the curated anchor set
  (hand-seeded threads + active concepts).
- `autonomous_on_surface` = the rest (observer-minted, weaver-promoted).
- `ratio = curated_on_surface / N`.

Health = ratio **trending down without surface quality dropping** — the
THESIS success bar for R1 is literally *"the curated stop-set becomes
redundant."* This metric measures that redundancy directly: if the new
scorer auto-demotes noise, the curated dependence falls and the ratio is the
receipt. Report both the count and the specific curated handles still doing
load-bearing work (so a human sees *which* hand-list entries are still
propping the surface up).

This is also the cleanest A/B discriminator: run old vs. new scorer, compare
`curated:autonomous` on each. The THESIS asks to prove the stop-set becomes
redundant; this is the number that proves it.

### Metric 3 — Surfaced-N-times-never-used

The click-through negative tail. The ledger already records both halves:

- **Surfaced**: `ChainEvent::LensIncluded { chunk_id, slot, ts }` (P9b,
  `crates/store/src/threads.rs` L339) — the weakest "useful" weight (0.5 in
  `AccessWeights`, L1114). Read via `ChainLogReader::access_history(
  chunk_ids, since)` (L1144–1148), keyed by `AccessKind::LensIncluded`.
- **Used**: `ExplicitRecall` (operator recalled the chunk), `RecallFault`,
  `OperatorSelected` (strongest, L358–359), and at the *concept* level
  `ConceptAccessed` / `ConceptNoteAdded` / `ConceptPromoted`.

Define per handle, over a window: `surfaced = count(LensIncluded)`,
`used = count(ExplicitRecall|OperatorSelected|RecallFault|cited-in-decision)`.
Flag handles with `surfaced ≥ N` (config threshold, default ~10) and
`used == 0`. Those are pure surface cost — they consume context-budget and
return nothing. This is the operational form of the THESIS's
*"decay surface-but-never-landed."*

**Implementation seam (important, document it):** the access ledger is keyed
by `chunk_id` (`LensIncluded`, `ExplicitRecall`), while the surface is keyed
by **thread/concept handle**. The two join only through the evidence graph
(`thread_evidence` / `evidence_links`: handle → chunk coords) or, for
concepts, directly via `ConceptAccessed { project, handle }` (handle-keyed —
no join needed). So:

- **Concept-handle never-used** is *cheap*: `ConceptAccessed` already carries
  the handle. Count `LensIncluded` of the concept's evidence chunks vs.
  `ConceptAccessed` for the handle.
- **Thread-handle never-used** needs a join through evidence links. Bound the
  cost the way `thread_query`'s backfill does (one batched `access_history`
  call over the union of evidence chunk_ids, L-block in `query.rs`), not a
  per-handle query.

Honor the `salience-vs-familiarity` distinct-query gate (`activation.rs`
L21–24): count **distinct `query_hash`**, not raw events, so one chatty
recall loop doesn't fake "used." That gate is doctrine — the thread layer
already got bitten by raw counts; don't reintroduce it.

### Metric 4 — Active-vs-decided drift

The active set should track recorded operator judgment. The judgment
artifacts are `ostk_decision` (562 rows live), `ostk_needle` (212),
active concepts (the curated tier), and the chain's `ConceptPromoted` /
`ConceptAccessed` / `ConceptNoteAdded` events (handle-keyed, windowed).

Define two handle sets over a recent window:
- `A` = active-surface handles (Metric 1's set).
- `J` = handles appearing in recent decisions / promotions / concept
  accesses (the "recently judged salient" set).

Report **Jaccard drift** `1 − |A ∩ J| / |A ∪ J|` plus the two asymmetric
tails:
- `A \ J` — active but never judged salient ("surfacing noise the operator
  ignored"; overlaps Metric 3).
- `J \ A` — judged salient but NOT active ("the operator cares, the surfacer
  forgot"; the dangling-anchor / dropped-thread failure mode that was found
  by accident). **This tail is the highest-value alarm** — it's the
  `projection-truth` violation where the substrate's projection (the surface)
  has diverged from ground truth (operator judgment).

Health = low drift, and specifically a small `J \ A`. This is the metric that
would have surfaced the dangling-anchor bug proactively.

---

## 4. Where it computes

Two viable homes; recommend **both, layered** (matching the
stale_ingest pull+push split):

### 4a. `recall_stats` extension (the pull leg) — PRIMARY

Add a `salience_health: Option<SalienceHealth>` field to `RecallStats`
(`crates/core/src/types.rs` L266; `#[serde(skip_serializing_if =
"Option::is_none")]` so old MCP clients keep parsing — the exact pattern
`audit_newest_ts` / `watch` / `reranker` already use). Populate it in
`recall_stats()` (`crates/query/src/stats.rs`), gated on the attention/threads
handles being wired (degrade to `None` when absent, like the events-DB guard
for `audit_newest_ts`). The `SalienceHealth` struct is decomposable:

```
struct SalienceHealth {
    surface_entropy: f32,        // H_norm in [0,1]
    surface_score_spread: f32,   // max - min (cheap drift proxy)
    curated_ratio: f32,          // curated_on_surface / N
    curated_load_bearing: Vec<String>,   // which hand-list entries still prop
    never_used: Vec<NeverUsed>,  // {handle, surfaced, used, window_days}
    active_decided_drift: f32,    // Jaccard distance
    drift_forgotten: Vec<String>, // J \ A — judged-but-not-active (top alarm)
    threshold: SalienceThresholds,  // what we judged against (self-documenting)
}
```

Compute cost: `surface()` is already an in-memory scan; entropy/ratio/spread
are O(N) over ≤ `limit` pages — negligible. Metric 3 + 4 touch the ledger
(`access_history`, a windowed `chain_log` scan) — bound them like the
`thread_query` backfill (one batched call, windowed `since`), and cache the
whole block on the `stale_ingest` TTL pattern.

### 4b. A dedicated `salience_health` MCP tool / CLI verb (the push leg) — SECONDARY

Two reasons to also have a standalone surface:

1. **The conditional-push leg** (server.rs L727) wants a compact "unhealthy?"
   verdict to inject into arbitrary responses. A `compute_salience_health()`
   that returns `Some(block)` *only when a threshold is breached* (entropy
   below floor, drift above ceiling, any `never_used` over N) is the direct
   analogue of `compute_stale_ingest()`. Same loud-on-failure semantics.
2. **The A/B harness** (THESIS proof bar) needs to run these metrics over the
   *old* and *new* scorer's surfaces side by side. A pure function over a
   surface (not bolted to `recall_stats`) is reusable by the harness. So the
   compute logic belongs in a library fn (`crates/attention` or
   `crates/query`) that *both* `recall_stats` and the A/B harness call.

**Recommendation:** put the pure compute in a new module
(`crates/attention/src/health.rs`, since it reads attention surface +
thread/concept state) exposing `fn salience_health(surface, ledger, judgment,
thresholds) -> SalienceHealth`. `recall_stats` calls it for the pull field;
`server.rs` calls a thin `compute_salience_health() -> Option<Value>` wrapper
(threshold-gated) for the conditional push; the A/B harness calls it twice.
Keep the curated-set knowledge (`stop_handles`, anchor set) flowing in as a
parameter, not hardcoded — the THESIS wants the hand-list to die, so don't
re-bake it into the auditor.

---

## 5. How it surfaces (operator-facing)

Mirror `stale_ingest` exactly:

- **`recall_stats.salience_health`** — always present (when handles wired),
  decomposed, truthful. The dashboard.
- **Conditional in-band push** — when any threshold breaches, a compact
  `salience_health` key appears in tool responses with a `hint`, e.g.
  *"salience surface drift high (J\\A=12 judged-but-inactive handles); the
  active set has diverged from recent decisions — see recall_stats
  salience_health"*. Healthy responses stay clean.
- **CLI**: a `ostk recall health` / extend `ostk show`-style verb is the
  operator's manual check (user-side; we suggest, can't invoke).

Self-documenting blind spots (the `stale_ingest` "change one, change both"
discipline):
- Entropy/ratio over the surface only see what cleared `ARCHIVE_THRESHOLD` —
  a handle suppressed *below* archive is invisible to Metric 1/2 (by design;
  that's the floor doing its job). Document it.
- Never-used and drift are **windowed** (`since`); a handle used 90 days ago
  but not lately reads as "never used" within the window. Report the window
  in the block (`window_days`) so the number is interpretable.
- Thread-handle never-used depends on the evidence graph being populated; a
  thread with no evidence links can't be use-attributed. Flag those
  separately ("unattributable") rather than miscounting them as never-used.

---

## 6. Thresholds (config, not constants)

Following `[watch].freshness_threshold_secs`, add a `[salience.health]`
config block (or fold into the new `[salience]` scorer block R1–R3 introduce)
with: `min_surface_entropy` (~0.6), `max_active_decided_drift` (~0.7),
`never_used_min_surfaced` (~10), `health_window_days` (~14), `ttl_secs` (30).
Live-tunable, A/B-friendly, no recompile. The compute path itself is **not**
behind a flag (it's pure observation, always safe to surface) — only the
scorer changes (R1–R3) are flag-gated. The auditor must watch *both* old and
new scorer, so it cannot be conditional on the new scorer being on.

---

## 7. Integration notes for D (design) and I4 (implement)

- **Order:** I4 is last (deps I1–I3) so the metrics can validate the new
  scorer. But the *pure compute fn* can land early and run against the
  *current* scorer to establish a baseline (today's `0.25` score-ribbon,
  curated ratio, never-used tail). Recommend shipping Metric 1+2 (surface-only,
  zero ledger cost) first as the A/B instrument, then 3+4 (ledger joins).
- **Shared primitive with R1:** `shannon_entropy(&[f32])` is used by both R1
  (per-handle co-occurrence) and R4 (surface score-share). Factor once.
- **Shared ledger reader with R2 (value):** Metric 3's surfaced-vs-used join
  is the *same* `access_history` / `ChainEvent` machinery R2 (value axis)
  uses for use-feedback. R4 reports it; R2 feeds it into the score. Same data
  source, two consumers — coordinate the join helper in design so it's
  written once.
- **`abi-as-sovereign-boundary` discipline:** every metric ships its `why`
  (the component handles/counts that produced it), never a bare scalar —
  matching `ConceptWhy` and `ScoreAttribution`.

## File / symbol index (for the implementer)

- Surfacing template: `crates/mcp/src/server.rs` L719–841
  (`compute_stale_ingest`, `stale_ingest` cache, pull+push legs, invariant).
- `recall_stats` compute: `crates/query/src/stats.rs`;
  `RecallStats` struct: `crates/core/src/types.rs` L266–302.
- Active surface: `crates/attention/src/lib.rs` `surface()` L1430–1556;
  `ScoreAttribution` per page L1484–1493; `ARCHIVE_THRESHOLD` filter L1477.
- Curator thresholds: `crates/attention/src/curator.rs` L56–70.
- MCP dispatch (where a new tool registers): `crates/attention-mcp/src/
  handlers.rs` L126–149 (match arm), `surface()` handler L227–236.
- Use-feedback ledger: `crates/store/src/threads.rs` — `ChainEvent` enum
  L228–419 (`LensIncluded` L339, `ExplicitRecall` L349, `OperatorSelected`
  L359, `ConceptAccessed` L377, `ConceptPromoted` L405); `AccessKind` /
  `AccessWeights` L1098–1130; `ChainLogReader::access_history` L1139–1148.
- Decomposable-`why` precedent + distinct-query gate: `crates/store/src/
  activation.rs` L16–24, `ConceptWhy` L71–89.
- Batched-ledger-join cost model: `crates/attention/src/query.rs`
  `backfill_cross_axis` (one batched fetch over the union of chunk_ids).
