# Plan — Autonomous Salience Implementation

> Team `salience`. The sequenced build plan. Input: `design.md` (the unified
> scorer, patched at the lead gate with the value-axis monotonicity invariant
> §4.5). Output of this phase: this file. Consumed by the four implementation
> tasks (I1–I4) and the review/verify task (V).
>
> **Branch:** `feat/autonomous-salience` — all implementation work lands here,
> off `main`. Never commit to `main` without the lead's review gate (THESIS
> §conventions). Each axis is one or more commits on this branch; the V phase
> reviews the whole branch before default-on.
>
> **Master flag default: OFF.** `[salience] scorer_v2 = false` ships in every
> increment. With it off, `compute_score_parts` is bit-identical to today
> (proven by I1 test #2). Default-on is a V-phase decision after the A/B
> harness passes — it is NOT part of any I-step.
>
> Code anchors are against the gens read during design/plan (lib.rs gen 103,
> commands.rs gen 144, config.rs gen 49, attention.rs gen 13, curator.rs,
> weaver.rs gen 67, threads.rs, activation.rs, stats.rs, core/types.rs). Symbol
> names are the durable anchors; line numbers may drift a few lines.

---

## 0. Sequencing rationale (why this order)

All of I1–I3 edit the **same function** (`compute_score_parts`) and the **same
boot pass** (`re_anchor_threads_from_corpus`). To avoid scorer merge conflicts
and keep each step independently testable + reviewable:

```
S0  scaffolding (branch + config + SalienceFactors + map wiring + flag)   ← shared substrate, no behavior change
 └─ I1  specificity   (the core; introduces the factors lookup + floor damp + lift cut)
     └─ I2  negative-transfer  (adds the (1−γ·neg) damp + exemplar set; depends on I1's signature)
     └─ I3  value      (fills the value factor via the shared ledger join; depends on I1's signature)
 └─ I4  self-audit     (pure observation; metrics 1+2 can start in parallel with I1 as the A/B instrument;
                        metrics 3+4 depend on I3's shared surfaced_vs_used helper)
 └─ V   review + A/B harness + default-on decision   (depends on I1–I4)
```

- **S0 first** so I1–I3 all build on a settled `compute_score_parts` signature
  (`+ factors: SalienceFactors, + cfg: &SalienceScorer`) and a settled
  `SalienceFactors` map — the signature change happens **once**, in S0/I1, not
  three times.
- **I1 before I2/I3** because I2 and I3 only *fill different fields* of the
  `SalienceFactors` the I1 signature already threads; they do not re-touch the
  signature. I2 adds the `damp` term + the store-level exemplar set; I3 adds the
  `value` field population. They touch disjoint lines of `compute_score_parts`
  (I2: the `damp` multiply on floor + resonance_term + lift cut; I3: nothing in
  the scorer beyond what I1 already wired — `val` is applied by I1's floor
  expression, I3 just makes it non-1.0) and disjoint blocks of the boot pass
  (pass A/B for I2, pass C-value for I3).
- **I2 and I3 are mutually independent** given I1 — they can be built in either
  order or in parallel; the plan lists I2 then I3 to match the THESIS DAG, but
  the only hard dep is I1.
- **I4 is pure observation, not flag-gated** (design §7.2): its compute path
  watches BOTH scorers, so it cannot wait behind `scorer_v2`. Metrics 1+2
  (surface-only, zero ledger cost) are the A/B instrument and can land as early
  as alongside I1 to capture the "before" baseline. Metrics 3+4 reuse I3's
  `surfaced_vs_used` helper, so they sequence after I3.

---

## S0 — Scaffolding (shared substrate; no behavior change)

**Goal:** land everything that I1–I3 share, behind the OFF flag, with zero
change to scoring when off. This is the step that introduces the
`compute_score_parts` signature change exactly once.

### S0.1 Branch
- `git checkout -b feat/autonomous-salience` off `main` (Bash,
  `dangerouslyDisableSandbox: true` for cargo/git reaching `~/.cargo`).

### S0.2 Config — `SalienceSettings` (`crates/core/src/config.rs`)
- Add `SalienceSettings` + `SalienceHealthSettings` structs mirroring
  `WeaverSettings` (config.rs:397) exactly: `#[derive(Debug, Clone, Serialize,
  Deserialize)]`, `#[serde(deny_unknown_fields)]`, per-field
  `#[serde(default = "default_…")]`, a `Default` impl, a
  `resolve(slot: Option<&Self>) -> Self` (config.rs:472-478).
- Fields = the full `[salience]` + `[salience.health]` block from design §7.1.
  Critical knobs: `scorer_v2 = false`, `specificity_enabled/value_enabled/
  negative_enabled`, `specificity_min_evidence = 5`, `specificity_lift_cutoff
  = 0.2`, `value_neutral = 1.0` (NOT the old bool — design §4.3/§7.1),
  `value_w_use/value_w_judg = 0.5`, `neg_gamma = 0.8`, `negative_knn_k = 3`,
  `negative_tau = 0.45`, `negative_lift_cutoff = 0.5`, `damper_floor = 0.02`;
  health: `min_surface_entropy = 0.6`, `max_active_decided_drift = 0.7`,
  `never_used_min_surfaced = 10`, `health_window_days = 14`, `ttl_secs = 30`.
- Add the `salience: Option<SalienceSettings>` field to the top-level config
  struct (sibling to `weaver: Option<WeaverSettings>`).
- **Test:** a `default_salience_block` test mirroring the
  `default_weaver_stop_handles` test (config.rs:1153) — absent block resolves
  to defaults; `scorer_v2 == false`; `value_neutral == 1.0`.

### S0.3 Types — `SalienceFactors` + `SalienceScorer` (`crates/attention/src/salience.rs`, NEW)
- New module `crates/attention/src/salience.rs`; add `mod salience;` (+ `pub
  use` the items the boot pass/scorer need) to `lib.rs`.
- `pub struct SalienceFactors { specificity: f32, value: f32, neg_penalty: f32 }`
  with `Default = {1.0, 1.0, 0.0}` (design §2.1).
- `pub struct SalienceScorer { … }` — the hot-path `Copy` snapshot of the
  scorer knobs (`scorer_v2`, the three `*_enabled`, `specificity_lift_cutoff`,
  `neg_gamma`, `negative_lift_cutoff`, `damper_floor`). Derived once from
  `SalienceSettings` via `SalienceScorer::from(&SalienceSettings)`.
- `pub fn shannon_entropy(weights: &[f32]) -> f32` — the SHARED primitive
  (design §3.1): normalize internally, return raw `H = −Σ pᵢ ln pᵢ`;
  empty/single-bin/all-zero → 0.0. **Used by I1 and I4.** Land it here in S0 so
  both axes import the one impl.
- **Tests (pure-fn, in `salience.rs`):** `shannon_entropy_*` — single-bin → 0;
  uniform-N → `ln(N)`; peaked → low; empty/zero → 0.

### S0.4 `InMemoryAttention` wiring (`crates/attention/src/lib.rs:853`) — P-1 resolved
- `salience_scorer: SalienceScorer` — a plain `Copy` field **on
  `InMemoryAttention`** next to `stop_handles` (lib.rs:871), set at construction
  (config is known then).
- The mutable, boot-populated data goes **on `Inner`** (the struct already
  behind `inner: Arc<RwLock<Inner>>`, lib.rs:854) — NOT a new `Arc`/`ArcSwap`
  (P-1 resolution: reuse the lock the scorer already traverses, no new dep):
  - `salience_factors: HashMap<String, SalienceFactors>` (default empty).
  - `negative_exemplars: Vec<Vec<f32>>` and `global_anchor_mean: Vec<f32>`
    (default empty = neutral; filled by I2's boot pass).
- Builder `with_salience_settings(self, cfg: &SalienceSettings) -> Self` (sets
  `salience_scorer`), analogous to `with_stop_handles` (lib.rs:915). Thread into
  `Default` and `with_embedder` (lib.rs:874, 899) so all constructors initialize
  to neutral/empty.
- Setters (post-construction, for the boot pass) take a brief `.write()` guard:
  `set_salience_factors(&self, map)`, `set_negative_exemplars(&self, mean,
  exemplars)` — each `let mut inner = self.inner.write().await;` then assign.
- Wire `with_salience_settings` into the `serve` construction
  (commands.rs:1428-1431) right after `with_stop_handles`, resolving
  `SalienceSettings::resolve(cfg.salience.as_ref())`.

### S0.5 Scorer signature + neutral pass-through (`compute_score_parts`, lib.rs:717)
- Change the signature to `(state, attention_vec, now, forced_stop, factors:
  SalienceFactors, cfg: &SalienceScorer)`.
- Body: implement the FULL design §1.2 expression but guarded so that with
  `cfg.scorer_v2 == false` (or all `*_enabled == false`) it is **bit-identical**
  to today — `spec = val = 1.0`, `neg = 0.0`, `damp = 1.0`, `lift_is_stop ==
  is_stop`. (S0 lands the expression; I1/I2/I3 flip the per-axis gates on and
  fill the factors. Landing the whole expression once avoids re-editing the
  function three times.)
- `ScoreParts` (lib.rs:705): add `specificity/value/neg_penalty: f32` axis
  fields. `ScoreAttribution` (core attention.rs:112): add the same three
  (`#[serde(default)]` so old MCP clients keep parsing). Populate at the
  `AttentionPage` build site (lib.rs:1484).
- Update **both** call sites to look up the factor + pass the snapshot. Both
  already hold the `inner.read().await` guard, so they read
  `inner.salience_factors` / `inner.negative_exemplars` directly — no extra
  lock, no clone of the whole map:
  - `surface` (lib.rs:1476): inside the existing read guard, pass
    `inner.salience_factors.get(handle).copied().unwrap_or_default()` and
    `&self.salience_scorer`.
  - `score_thread` (lib.rs:1683): same lookup, per handle, under its read guard.
  - **No lock inside `compute_score_parts`** — it stays pure; the caller (which
    already holds the read guard) passes the factor by value.
- **Test:** `scorer_v2_off_is_bit_identical` — seed a thread, score with flag
  off + default factors, assert equality with a hand-computed today-scorer value
  (or a snapshot captured before the change). This is the A/B no-regression
  guard and the gate for every later step.

### S0.6 Boot-pass skeleton (`re_anchor_threads_from_corpus`, commands.rs:1967)
- After the existing anchor seeding, add the guarded precompute skeleton from
  design §2.2: `if !salience.scorer_v2 { return Ok(seeded) }`, then the three
  labelled blocks **as no-op stubs** that I1/I2/I3 fill:
  - `// (A) global mean` (I2)
  - `// (B) negative exemplars` (I2)
  - `// (C) per-handle factors` — loop building the `HashMap<String,
    SalienceFactors>`; in S0 it produces all-default factors and calls
    `attention.set_salience_factors(map)`.
- The pass already holds `&Arc<ThreadsDb>`, `&Arc<CorpusStore>`,
  `&InMemoryAttention`, and iterates every thread — I1/I2/I3 hang their compute
  off this single walk (design §2.2: ONE pass, not three).
- Thread the resolved `SalienceSettings` into the function signature (it needs
  `scorer_v2` + the knobs); pass it from the `serve` call site
  (commands.rs:1451).

**S0 exit criteria:** `cargo test -p ostk-recall-attention -p ostk-recall-core`
green; flag off ⇒ no behavior change (S0.5 test); the config block round-trips.

---

## I1 — Specificity (the core; ships first)

**Depends on:** S0. **Toggle:** `specificity_enabled` (+ master `scorer_v2`).

### Files / functions
- `crates/attention/src/salience.rs`:
  - `pub fn specificity_from_evidence(evidence: &[EvidenceLink], chunk_meta:
    &HashMap<String, SourceMeta>, cfg: &SalienceSettings) -> f32` — build the
    `(project, source_id)` histogram (design §4-I1, R1 §1b/§5-Q2) from each
    evidence row's `last_resolved_chunk_id` → `source_id`; call
    `shannon_entropy`; return `1 − H/ln(N_eff)`. Guards: `N_eff ≤ 1 → 1.0`;
    `total_resonating_chunks < specificity_min_evidence → 1.0`.
- `crates/store/src/threads.rs`:
  - `pub fn list_evidence_all(&self) -> Result<HashMap<String, Vec<EvidenceLink>>>`
    — batched reader (one query, grouped by handle), so the boot pass does not
    call `list_evidence` (threads.rs:2394) per handle.
- `crates/store/src/corpus.rs`: reuse `fetch_chunks_by_ids` (corpus.rs:799) to
  map the union of evidence chunk_ids → `source_id` (carried at corpus.rs:853).
- `crates/cli/src/commands.rs` (`re_anchor_threads_from_corpus`, pass C): for
  each handle, `specificity_from_evidence(...)` → set
  `factors[handle].specificity`.
- `compute_score_parts` (already signature-ready from S0): flip the
  `specificity_enabled` gate live — `spec` multiplies the earned floor; `spec <
  specificity_lift_cutoff` ORs into `lift_is_stop` (design §1.2 terms 1+3).

### Tests (model on `stop_handle_floor_clamped_by_context_diversity` lib.rs:2383, `curated_stop_set_clamps_high_resonance_harness_handle` lib.rs:2431; tests mod at lib.rs:1819)
1. `specificity_demotes_diffuse_handle_with_no_hand_list` — two threads,
   identical counters (300/290) + identical anchors (equal live resonance);
   inject `specificity = 0.05` vs `0.95` via the factors map (a test setter or
   direct `set_salience_factors`); assert the concentrated one out-ranks the
   diffuse one with NO `with_stop_handles`. (Direct cliff-replacement proof.)
2. `specificity_one_reproduces_current_scorer` — default factors / flag off ⇒
   score bit-identical to today (the S0.5 guard, re-asserted under I1).
3. `specificity_from_evidence_*` (pure fn) — single-bin → 1.0; uniform-N → ≈0.0;
   peaked (one dominant + tail) → high; below `min_evidence` → 1.0 neutral.
4. `specificity_does_not_block_active_discussion` — diffuse handle (`spec ≈ 0`)
   with high *live* resonance against the attention vector still surfaces (floor
   clamped, `ALPHA·resonance` intact).

**I1 exit:** the four tests green; A/B fixture run (§A/B harness) shows
specificity-alone demotes the coherent-noise set below the real-concept set
(pass criterion P2).

---

## I2 — Negative-transfer (depends on I1)

**Depends on:** I1 (the factors signature + boot pass C). **Toggle:**
`negative_enabled`.

### Files / functions
- `crates/attention/src/salience.rs`:
  - `fn center(v: &[f32], global_mean: &[f32]) -> Vec<f32>` —
    `normalize(normalize(v) − global_mean)` (R3 MANDATORY; design §3.3).
  - `fn negative_penalty(anchor, global_mean, exemplars, cfg) -> f32` — centered
    kNN, k = `negative_knn_k` (default 3), τ-floored rescale to [0,1] (design
    §3.3). Reuse `cosine_similarity` (lib.rs:238).
- `crates/store/src/threads.rs`: `pub fn dormant_anchor_vecs(&self) ->
  Result<Vec<Vec<f32>>>` — `tension='dormant' AND anchor_vec IS NOT NULL`,
  decoded via `bytes_to_f32_vec`; mirrors `reanchor_inputs` (threads.rs:1935)
  filtered to dormant (R3 §1A; STRONG, 279 handles / 432 anchored).
- `crates/store/src/concepts.rs`: `pub fn rejected_concept_anchors(&self) ->
  Result<Vec<Vec<f32>>>` — evidence `anchor_vec` for `status='rejected'`,
  de-duped per concept by mean; mirrors `evidence_to_reconcile` (concepts.rs:1229)
  filtered to rejected (R3 §1B; WEAKER, 47 handles).
  **Exclude** the bare session-rejected handle strings (R3 §1 — do NOT embed).
- `crates/cli/src/commands.rs` (`re_anchor_threads_from_corpus`):
  - Pass (A): `global_anchor_mean = mean(normalize(anchor))` over all anchored
    inputs.
  - Pass (B): `neg_exemplars = [center(v) for v in dormant_anchor_vecs ++
    rejected_concept_anchors]`; optional scrub (drop any exemplar with centered
    cosine > 0.97 to an active-concept anchor — R3 §1 hardening);
    `attention.set_negative_exemplars(mean, exemplars)`.
  - Pass (C): `factors[handle].neg_penalty = negative_penalty(anchor, mean,
    exemplars, cfg)`.
- `compute_score_parts`: flip `negative_enabled` gate — `damp = 1 − neg_gamma·neg`
  multiplies the floor AND `resonance_term`; `neg > negative_lift_cutoff` ORs into
  `lift_is_stop` (design §1.2 term 2 + 3).

### Tests
1. `negative_penalty_keeps_recoverable_concept` — **the ostk-cache case** (R3
   named A/B hard case): fixture anchor at centered cosine 1.0 to ONE exemplar
   but far from the other k−1; assert with k=3 + γ=0.8 the damp leaves it
   surfaceable above a true-noise handle. (Proves the soft bounded form
   recovers a real concept reusing rejected sub-vocab.)
2. `negative_penalty_damps_harness_twin` — fixture anchor near several
   exemplars; assert its floor damps below a clean handle's.
3. `centering_is_required_for_separation` — raw-space mean-top-k AUC ≈ random on
   a fixture exemplar set; centered kNN AUC high (R3 §empirical).
4. `negative_penalty_neutral_when_no_exemplars` — empty set ⇒ 0.0 ⇒ no effect
   (flag-off / cold-substrate safety).

**I2 exit:** tests green; A/B run confirms ostk-cache survives (P3) and the
novel-harness-term pre-damp works.

---

## I3 — Value (depends on I1)

**Depends on:** I1 (factors signature + boot pass C). **Toggle:**
`value_enabled`. **v1 ships `value_neutral = 1.0` ⇒ value is a constant-1.0
pass-through** (design §4.3-4.5); the join runs as attribution-only scaffold,
the multiplier is inert. This is the lead-gate monotonicity fix — do NOT
average a near-zero `value_use` into present judgment.

### Files / functions
- `crates/store/src/threads.rs`: `pub fn surfaced_vs_used(&self, handles,
  since) -> Result<HashMap<String, UseLedger>>` — **the SHARED join** (design
  §3.4; R4 metric 3 reuses it). Mechanics: union each handle's `{anchor_chunk_id}
  ∪ {evidence_links.last_resolved_chunk_id for Active links}`; **one batched**
  `access_history` (threads.rs:1144/1152 trait) over the union across all
  handles; classify by `AccessKind` (LensIncluded=surfaced 0.5;
  ExplicitRecall/OperatorSelected/RecallFault=used); **distinct-`query_hash`
  gate** (activation.rs:326 doctrine); `unattributable: true` for no-evidence
  handles. Returns `UseLedger { surfaced, used_weighted, distinct_used_queries,
  unattributable }`.
- `crates/attention/src/salience.rs` (or a thin assembler near the boot pass):
  `fn value_from(handle, use_ledger, judgment_hits, active_coords, cfg) -> f32`:
  - `value_use = squash(used_weighted-derived raw)` — **reuse activation.rs
    verbatim**: `act_r_base` (activation.rs:188), `squash` (:198),
    `age_hours_floored` (:463), `ACT_R_DECAY_D` (:43), `AccessWeights`.
  - `value_judgment = 1 − exp(−Σ)` over (i) Active evidence links resolving to
    `source ∈ {ostk_decision, ostk_needle}` chunks and (ii) intersection with
    `concept_support_by_coord(since)` (activation.rs:515) — mirrors
    `edge_lift_for` saturation (:443). **No new join table** (design §4.2).
  - `positive = clamp01(w_use·value_use + w_judg·value_judgment)`.
  - `value = value_neutral + (1 − value_neutral)·positive` (design §4.3 monotone
    form). **v1: `value_neutral = 1.0` ⇒ value = 1.0 regardless.**
- `crates/cli/src/commands.rs` (`re_anchor_threads_from_corpus`, pass C): call
  the join + `value_from` → set `factors[handle].value`. Reuse the weaver's
  already-fetched anchor snapshot for judgment cosine where possible (design
  §R2-Q4) rather than re-querying Lance.
- `compute_score_parts`: no new edit beyond I1 — I1's floor already multiplies
  `val`; I3 only makes `val` non-1.0 once `value_neutral < 1.0` ships.

### Tests
1. `value_monotone_in_positive_evidence` — **THE invariant test** (design §4.5):
   two threads identical specificity/anchor/counters; A has an `ExplicitRecall`
   on its evidence chunk (or an `ostk_decision`-sourced Active evidence link), B
   has none. Assert `value(A) ≥ value(B)` AND `score(A) ≥ score(B)` **for every
   `value_neutral ∈ {1.0, 0.7}`** — the regression guard against the
   "average-a-zero-down" inversion.
2. `value_v1_is_pass_through` — with `value_neutral = 1.0` (shipped v1 config),
   any handle (judged, used, or neither) gets `value == 1.0`; score equals the
   specificity/negative-only score (no double-count).
3. `value_judgment_raises_proven_handle_when_unlocked` — with `value_neutral =
   0.7`: a thread whose Active evidence link resolves to an `ostk_decision`
   chunk gets `value > 0.7` (raised toward 1.0); a plumbing-only thread stays at
   0.7; the judged thread out-ranks. (Lift direction is UP.)
4. `value_distinct_query_gate` — 50 accesses sharing one `query_hash` collapse
   to one in `positive(H)` (model on `distinct_query_gate_not_raw_count`,
   activation.rs:805); asserted at the `positive`/join level so it holds for any
   `value_neutral`.

### DEFERRED (NOT in I3 v1 — second increment)
- `ChainEvent::ThreadSurfaced { handle, ts }` at the surface build site
  (lib.rs:1480) — the only legitimate source of the *wasted* signal (design
  §4.4). Unlocks `value_neutral < 1.0` (ceiling-raiser) + the
  surfaced-but-never-landed penalty. Log it as a follow-up needle; do not build
  in I3 v1.

**I3 exit:** the four tests green; with the shipped `value_neutral = 1.0`, the
A/B run is unchanged by value (it's a pass-through) — confirming value adds no
regression and the join/attribution plumbing is live for the second increment.

---

## I4 — Self-audit (pure observation; NOT flag-gated)

**Depends on:** metrics 1+2 can start alongside I1 (surface-only, the A/B
instrument); metrics 3+4 depend on I3's `surfaced_vs_used` helper. **No
toggle** — the compute path watches BOTH old and new scorer (design §7.2);
only its thresholds are config (`[salience.health]`).

### Files / functions
- `crates/attention/src/health.rs` (NEW): `pub fn salience_health(surface,
  ledger, judgment, thresholds) -> SalienceHealth` — pure, reusable by
  `recall_stats`, the conditional push, AND the A/B harness (called twice).
  Imports `shannon_entropy` from `salience.rs` (the one impl from S0.3).
- `crates/core/src/types.rs` (next to `RecallStats`, types.rs:266): `pub struct
  SalienceHealth { surface_entropy, surface_score_spread, curated_ratio,
  curated_load_bearing: Vec<String>, never_used: Vec<NeverUsed>,
  active_decided_drift, drift_forgotten: Vec<String>, threshold }` —
  decomposable, every metric ships its `why` (design §8-I4; R4
  `abi-as-sovereign-boundary`).
- Metrics (design §8-I4; R4 §3):
  1. surface entropy `H_norm = shannon_entropy(score_shares)/ln(N)` + cheap
     `surface_score_spread = max − min` (the live ~0.25-ribbon alarm).
  2. curated:autonomous ratio over the surface + the load-bearing curated
     handles still propping it.
  3. surfaced-N-never-used via the SHARED `surfaced_vs_used` (I3), distinct-
     query-gated, `unattributable` flagged separately.
  4. active-vs-decided Jaccard drift + the `J \ A` forgotten tail (top alarm).
- Surfacing (model on `stale_ingest`, server.rs:719-841):
  - **Pull leg:** `recall_stats.salience_health: Option<SalienceHealth>`
    (`#[serde(skip_serializing_if = "Option::is_none")]`), populated in
    `recall_stats` (the free fn at `crates/query/src/stats.rs:11`); degrade to
    `None` when the attention/threads handles are absent (mirror the events-DB
    guard for `audit_newest_ts`, stats.rs:34).
  - **Push leg:** `compute_salience_health() -> Option<Value>`
    (threshold-gated, loud-on-failure, 30s TTL cache) injected into responses
    only when unhealthy — the direct analogue of `compute_stale_ingest`.

### Tests
1. `surface_entropy_collapses_on_dominant_handle` — one dominant score → low
   `H_norm`; even spread → high.
2. `never_used_flags_surfaced_not_recalled` — fixture: handle with
   `LensIncluded ≥ N`, no `ExplicitRecall` → flagged; one with recalls → not.
3. `drift_reports_forgotten_judged_handle` — a handle in `J` but not `A` appears
   in `drift_forgotten`.
4. `health_runs_under_both_scorers` — `salience_health` produces a block with
   `scorer_v2` both on and off (observation, not gated).

**I4 exit:** tests green; the four metrics computed over the live surface
establish the "before" baseline (R4 §2: flat ~0.25 ribbon, off_diagonal_lift=0
on 39/40) and serve as the A/B scoreboard.

---

## A/B harness (design §7.3) — built in I4/V, run at V

A reusable pure-function harness. **Fixture form (P-4 resolved): a
`#[test]`/bench in `crates/attention`** — gates I1–I3 in CI. **Live form: driven
from the review session via the `attention_surface` MCP tool** against running
`serve` (the V-phase recipe below) — no new CLI subcommand for v1:

1. Load the thread set (fixture, or live via MCP `attention_surface` + the boot
   pass inputs).
2. Compute `salience_factors` for every handle via the §2.2 pass.
3. Rank the set **three ways**: (a) `scorer_v2 = false` (today's scorer); (b)
   `scorer_v2 = true` (all axes); (c) `scorer_v2 = true` with `stop_handles`
   disabled (`forced_stop = false` for all) — the redundancy probe.
4. Emit I4's four metrics for each run (the scoreboard).

**Pass criteria (the V gate):**
- **P1 (ordering):** `{turn-digest, squad-lead, re-run, non-blocking, re-read,
  pre-existing, follow-up, top-level, system-reminder}` all rank BELOW
  `{cognitive-memory, ostk-cache, dereference-or-void,
  relational-substrate-docgraph}` under the new scorer.
- **P2 (specificity alone):** the specificity-only run reproduces P1's ordering
  (the stop-set effect with no hand-list).
- **P3 (ostk-cache survives):** ostk-cache stays above `ARCHIVE_THRESHOLD` and
  above all P1 noise in the negative-enabled run.
- **P4 (redundancy):** stop-set-disabled run shows `curated_ratio` falling
  without P1 ordering regressing.
- **Baseline to beat:** today's flat ~0.25 ribbon, `off_diagonal_lift = 0` on
  39/40 (R4 §2).

---

## V phase — verify / make it so (operational recipe; task #11)

V is a separate task (#11), but the recipe is captured here so the reviewer
isn't inventing it. The fixture A/B (above) gates correctness in CI; this is
the **live** proof on the running substrate, plus the default-on flip.

1. **Build release.** `cargo build --release -p ostk-recall-cli` (Bash,
   `dangerouslyDisableSandbox: true` — reaches `~/.cargo`). The flag is still
   OFF at this point, so the new binary is behaviorally identical to the running
   one until the config is flipped (step 4) — that's intentional: install first,
   prove identical, then flip.

2. **Atomic install — do NOT `cp` over the running `serve` binary.** A `cp` (or
   `install`, or `cargo install` overwriting in place) onto the live binary
   invalidates its Mach-O code signature and the kernel **SIGKILLs the running
   `serve` process mid-write** (this bit us before — macOS re-validates the
   signature of a running image's backing file). Install via **temp + atomic
   rename on the same filesystem**:
   ```sh
   cp target/release/ostk-recall "<install_dir>/ostk-recall.new"   # write a NEW path
   mv -f "<install_dir>/ostk-recall.new" "<install_dir>/ostk-recall" # atomic rename = new inode
   ```
   The `mv` swaps the directory entry to a fresh inode; the already-running
   process keeps its original inode open and is untouched until it is
   deliberately restarted (step 3). Never write bytes into the path the live
   process is executing from.

3. **Restart `serve --watch`.** The factors map lives in `InMemoryAttention`,
   populated by `re_anchor_threads_from_corpus` at boot (the §2.2 pass) — a
   running `serve` will NOT pick up the new scorer or recompute factors without
   a restart. Cleanly stop the existing `serve` and relaunch
   `ostk-recall serve --watch` from the freshly-installed binary so the new
   `InMemoryAttention` constructs, replays the chain, runs the boot pass, and
   loads the factors. (Operator-side: suggest the user run it; agents can't
   restart their own substrate.)

4. **Flip the flag on for the A/B (config, not code).** Set `[salience]
   scorer_v2 = true` in the operator's config (leave the per-axis toggles at
   their defaults), restart `serve` once more so the resolved `SalienceSettings`
   threads in. Because the flag is config-resolved at construction, no rebuild
   is needed between the OFF and ON runs — same binary, two configs. This is
   what makes the live before/after a clean A/B.

5. **Live A/B run.** Drive `attention_surface` (MCP) against the running
   substrate in both configs and confirm the gate:
   - **P1** ordering (coherent-noise below real concepts),
   - **P2** specificity-alone reproduces the stop-set effect (toggle
     `value_enabled`/`negative_enabled` off),
   - **P3** **ostk-cache survives** — still above `ARCHIVE_THRESHOLD` and above
     all P1 noise with `negative_enabled = true` (the named hard case),
   - **P4** stop-set-disabled run shows `curated_ratio` falling without P1
     regressing.
   Read the scoreboard off `recall_stats.salience_health` (I4's pull leg) in
   each config: entropy UP, curated_ratio DOWN, never-used tail shrinking vs the
   captured baseline.

6. **Make it so (default-on).** Only after P1–P4 pass and the review checklist
   is clean: flip the `default_…` for `scorer_v2` to `true` in
   `crates/core/src/config.rs`, rebuild + atomic-install + restart (steps 1–3),
   and let the lead hold the merge gate on `feat/autonomous-salience → main`.
   **Retiring `default_weaver_stop_handles` is a SEPARATE follow-up** after the
   redundancy (P4) holds for a soak period — not part of this branch.

---

## Test & build commands (every step)

- Per-crate: `cargo test -p ostk-recall-attention -p ostk-recall-core`
  (Bash, `dangerouslyDisableSandbox: true`). Add `-p ostk-recall-store` and
  `-p ostk-recall-query` for the I3/I4 store/stats changes.
- Lint clean: `cargo clippy -p ostk-recall-attention -p ostk-recall-core
  -p ostk-recall-store -p ostk-recall-query` — the scorer carries
  `#[allow(clippy::similar_names, …)]` already (lib.rs:712); keep new pure fns
  clippy-clean.
- Full workspace `cargo test` before the V gate (no cross-crate regressions).

---

## Review checklist (the V gate — adversarial across all 4 axes)

**Correctness**
- [ ] Flag OFF ⇒ `compute_score_parts` bit-identical to pre-branch
      (`scorer_v2_off_is_bit_identical` + `specificity_one_reproduces_current_scorer`).
- [ ] Value-axis monotonicity invariant holds (`value_monotone_in_positive_evidence`
      green at `value_neutral ∈ {1.0, 0.7}`) — a decision-cited handle is NEVER
      more-damped than unproven noise (the lead-gate fix).
- [ ] `value_neutral = 1.0` shipped ⇒ value is a pass-through; no value-driven
      ranking change in v1 (`value_v1_is_pass_through`).
- [ ] ostk-cache survives the negative penalty (`negative_penalty_keeps_recoverable_concept`
      + A/B P3); centering is load-bearing (`centering_is_required_for_separation`).
- [ ] Specificity demotes coherent-noise with no hand-list (A/B P1/P2).
- [ ] `compute_score_parts` stays pure (no locks/DB); the Arc is read in the
      caller, factors passed by value.
- [ ] ε `damper_floor` clamp present on each `[0,1]` damper (gate, don't delete).
- [ ] Distinct-`query_hash` gate honored in the shared join (no raw-count fake).

**Composition / one-pass discipline**
- [ ] ONE boot pass fills specificity + value + neg-exemplars + global-mean
      (not three scans); batched joins read corpus/threads once.
- [ ] `shannon_entropy` + `surfaced_vs_used` each have exactly ONE impl, shared
      by their two consumers (I1/I4 and I3/I4).
- [ ] Per-axis toggles work independently; self-audit is NOT flag-gated and
      watches both scorers.

**Surface / safety**
- [ ] `ScoreAttribution` adds the three axes with `#[serde(default)]` (old MCP
      clients parse); `recall_stats.salience_health` is `skip_serializing_if`.
- [ ] `stop_handles`/`forced_stop` left wired (safety net + A/B control); the
      A/B P4 shows it's redundant before any retirement.
- [ ] No commit to `main`; all on `feat/autonomous-salience`; CHANGELOG entry.

**A/B scoreboard (I4 metrics)**
- [ ] New scorer: surface entropy UP, curated_ratio DOWN, never-used tail
      shrinking vs the captured baseline.

**Default-on decision (V only, after all boxes):** flip `scorer_v2 = true`
default in `default_…` config only once P1–P4 pass and the review is clean.
Retiring `default_weaver_stop_handles` is a SEPARATE follow-up after the
redundancy (P4) holds for a soak period — not part of this branch.

---

## Open questions carried from design (P-phase resolutions)

- **P-1 (interior mutability for the factors map). RESOLVED: store inside the
  existing `Arc<RwLock<Inner>>` — NO new dependency, NO second lock.** Checked
  the tree: `arc_swap` is NOT a workspace dep (absent from every `Cargo.toml`
  and `Cargo.lock`), and `InMemoryAttention` already holds all mutable state in
  `inner: Arc<RwLock<Inner>>` (lib.rs:854, tokio async `RwLock` — callers do
  `inner.read().await` / `.write().await`). So put `salience_factors:
  HashMap<String, SalienceFactors>`, `negative_exemplars: Vec<Vec<f32>>`, and
  `global_anchor_mean: Vec<f32>` as fields **on `Inner`**. The setters
  (`set_salience_factors`, `set_negative_exemplars`) take a brief `.write()`
  guard; the scorer's callers (`surface`/`score_thread`) already hold the
  `.read()` guard, so they read the map directly from `inner` — no extra lock,
  no clone-the-whole-map, no `ArcSwap` crate. `SalienceScorer` (the small `Copy`
  knob snapshot) stays a plain field on `InMemoryAttention` set at construction
  (`with_salience_settings`). This is strictly cleaner than either
  originally-listed option and adds zero dependency surface. **Settle this shape
  in S0.4; I1 builds on it.**
- **P-2 (specificity persistence). RESOLVED: recompute-on-boot for v1.** The
  pass already walks every thread, so recompute is free and always fresh; no
  schema change. A `threads.specificity REAL` column is a post-stabilization
  optimization (saves the boot recompute, lets `attention_surface why` expose
  it) — explicitly a follow-up, NOT this branch.
- **P-3 (consolidate-cadence refresh). RESOLVED: v1 is boot-only refresh.**
  Factors are computed once in the boot pass and held under the `RwLock`;
  between boots they are static (acceptable — anchors and evidence drift slowly,
  and the A/B runs against the boot-computed map anyway). The
  `AutoWeaver::consolidate` pass (weaver.rs:355) is the natural place to later
  re-run pass (C) and update the map under the same `.write()` guard so factors
  track drift — but `consolidate` does not obviously hold the corpus handle the
  judgment/specificity joins need, so wiring that is a **FAST-FOLLOW after the
  branch lands**, not v1. Say it plainly in the doc: v1 refreshes factors at
  boot only.
- **P-4 (A/B harness home). RESOLVED: fixture `#[test]` in `crates/attention`
  gates I1–I3; the V-phase live run is MCP-driven (no new CLI subcommand for
  v1).** The fixture form (load a captured thread-set fixture, compute factors,
  rank 3 ways, assert P1–P4) is enough to gate every I-step and runs in CI. The
  live "before/after on the running substrate" run at V is driven from the
  review session via the `attention_surface` MCP tool against `serve` (recipe in
  §"V phase" below) — a hidden `ostk-recall salience-ab` subcommand is a
  nice-to-have only if the team wants a repeatable one-shot, deferred unless V
  asks for it.

P-1 is the only one that must be settled inside S0 (the `Inner`-field shape);
P-2/P-3/P-4 are decided above and need no further gating — P-2/P-3 are
fast-follows, P-4's fixture form is built with I1.
