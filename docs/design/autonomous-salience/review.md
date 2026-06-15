# Review — Autonomous Salience (V gate)

> Independent adversarial review of branch `feat/autonomous-salience`
> (6 commits on `main`: S0 → I1 → I2 → I3 → I4 → docs/compile-fix).
> Reviewer did not build this. Goal: try to break it. Working tree clean
> throughout; no concurrent writer observed.

## VERDICT: PASS-WITH-FIXES

The scorer evolution is faithful to `design.md`: the flag-off path is
provably bit-identical to v1, shared machinery is factored exactly once,
the composition math is sound, the value monotonicity invariant holds for
all inputs, the boot pass is lock-clean, and self-audit is correctly
un-gated. All 6 targeted crates compile and test green (0 failures). The
issues below are real but none block landing **with the flag default-OFF**
(the shipped state); two should be fixed before `scorer_v2` is ever flipped
default-ON, and one (the A/B harness) is a V-gate *coverage* gap the live
run must close.

---

## Test & clippy results

`cargo test -p ostk-recall-attention -p ostk-recall-core -p ostk-recall-store
-p ostk-recall-query -p ostk-recall-mcp -p ostk-recall` (the CLI package is
`ostk-recall`): **0 failures** across the suite. Notable totals: attention
143 lib unit tests + integration suites green; core 97; store 95 (+ store
integration suites); query 81; mcp resources 12; CLI e2e suites all green.
Salience-specific tests present and green:
- salience.rs pure-fn: `entropy_*` (4), `specificity_*` (5), `value_*` (5),
  `negative_penalty_*` / `center_*` / `centering_*` (6), scorer-derivation (2).
- lib.rs scorer-level: `specificity_demotes_diffuse_handle_with_no_hand_list`,
  `specificity_one_reproduces_current_scorer`,
  `specificity_does_not_block_active_discussion`,
  `negative_penalty_damps_harness_twin`,
  `negative_penalty_keeps_recoverable_concept`,
  `value_v1_pass_through_is_bit_identical_to_flag_off`,
  `value_monotone_raises_proven_handle_in_surface`.
- health.rs: `surface_entropy_*` (2), `curated_ratio_*`, `never_used_*`,
  `drift_*` (2), `health_runs_under_both_scorers`.
- types.rs: `ScoreAttribution` round-trip + legacy-payload-decodes-to-neutral.
- store threads.rs: `surfaced_vs_used_*` (3, incl. distinct-query gate).

`cargo clippy` on those crates: **exit 0, no errors.** The workspace lint
profile is `pedantic`+`nursery` at **warn** (not deny) — so clippy does not
fail the build. There is a high *volume* of pre-existing-style warnings
(missing doc backticks, "first doc paragraph too long", `const fn`
suggestions, `derive Eq`), consistent with the rest of the workspace; the
new code carries appropriate `#[allow(...)]` where it deviates (e.g.
`struct_excessive_bools`, `too_many_lines`, `cast_precision_loss`). One
genuinely-new advisory: an **unused `AccessWeights` import** in
`crates/query/tests/freshness_act_r.rs` (a test, warn-only). Not a gate
failure; trivially fixable.

## Pre-existing failure confirmation

`tag_rule_sets_record_kind_and_reembeds`
(crates/pipeline/tests/record_rules_overlay.rs:132) fails on the branch with
`Lance(... Plan("array_has does not support type Utf8 ... array_has(Utf8, Utf8)"))`
at `lance-7.0.0/src/dataset/scanner.rs:445`. I built a **detached worktree on
`main`** and ran the identical test: it fails identically on `main`.
Corroborating evidence it is NOT from this branch: lance is `7.0.0` on both
refs; the branch's only change to `crates/pipeline/` is adding `salience:
None` to four `Config` struct literals (mechanical compile fixes for the new
field); the corpus/lance query layer (`store/src/corpus.rs`, `pipeline/src`,
`query/src` except the +6-line stats pull-leg) is **untouched**. **Verdict:
pre-existing, not attributable to salience.**

---

## Checklist (the V gate)

### Correctness

- [x] **Flag OFF ⇒ `compute_score_parts` bit-identical to pre-branch.**
  Verified against `git show main:` source. With `scorer_v2=false`,
  `SalienceScorer::from` forces every `*_enabled=false`, so `spec=val=1.0`,
  `neg=0.0`, `damp=(1.0−γ·0).clamp(eps,1.0)=1.0`, `lift_is_stop==is_stop`.
  Every new multiplier is `×1.0` (exactly representable in IEEE-754, so
  `x*1.0==x` bit-exact) and the floor branch / lift call are unchanged. The
  off-path is bit-identical at the source level. **Caveat — see Finding 1:**
  the `damp` clamp `(...).clamp(eps,1.0)` runs *unconditionally* and panics if
  `damper_floor > 1.0`, so "bit-identical when off" holds only for sane
  config. Guard test `specificity_one_reproduces_current_scorer` uses
  `approx_eq(_, _, 1e-6)` rather than `==`, and compares flag-off vs
  flag-on-with-neutral-factors (not a `main`-captured snapshot) — see
  Finding 4.

- [x] **Value-axis monotonicity invariant holds for ALL inputs.**
  `value_from`: `positive = clamp01(w_use·v_use + w_judg·v_judg)`,
  `value = (v_neutral + (1−v_neutral)·positive).clamp(0,1)`. With
  `v_neutral∈[0,1]` (clamped) and `positive∈[0,1]`, `value∈[v_neutral,1]`
  by construction — no code path produces `value < v_neutral`. The design-
  review "average-a-zero-down" inversion bug is genuinely fixed; positive
  evidence can only *raise*. Guarded by `value_monotone_in_positive_evidence`
  (pure, both v_neutral∈{1.0,0.7}) and end-to-end by
  `value_monotone_raises_proven_handle_in_surface`.

- [x] **`value_neutral = 1.0` shipped ⇒ value is a pass-through.** Default is
  1.0 (config.rs:601); `value_v1_pass_through_is_bit_identical_to_flag_off`
  proves a proven handle scores identically to flag-off.

- [~] **ostk-cache survives the negative penalty.** The *property* (bounded
  damp keeps a live-resonant handle surfaceable above equally-penalized noise)
  is proven by `negative_penalty_keeps_recoverable_concept`, and the *k=3 < k=1
  averaging* by `negative_penalty_k3_robust_to_single_collision`. But neither
  is the named end-to-end case, and the surface test **injects `neg_penalty=1.0`
  directly** rather than exercising `negative_penalty()` — see Finding 2.
  `centering_is_required_for_separation` confirms centering is load-bearing.

- [~] **Specificity demotes coherent-noise with no hand-list (P1/P2).** Proven
  only at the *2-thread, injected-factor* level
  (`specificity_demotes_diffuse_handle_with_no_hand_list`). The named
  coherent-noise SET vs real-concept SET ordering (P1) and "specificity alone
  reproduces the stop-set" (P2) are **not asserted by any CI test** — see
  Finding 3 (A/B harness gap).

- [x] **`compute_score_parts` stays pure (no locks/DB).** Signature takes
  `factors: SalienceFactors` (by value) + `cfg: &SalienceScorer`; no `inner`
  access, no `.await`, no DB. Both callers (`surface` lib.rs:1634,
  `score_thread` lib.rs:1850) read the factor under the read guard they
  already hold and pass it by value. Confirmed.

- [x] **ε `damper_floor` clamp on each [0,1] damper.** `spec` (L783), `val`
  (L788), `damp` (L800) each `.clamp(eps,1.0)`. Demotion is to the unresonant
  baseline, never zero. (But see Finding 1 for the `eps>1.0` panic.)

- [x] **Distinct-`query_hash` gate honored in the shared join.**
  `surfaced_vs_used` (threads.rs:1279) buckets used accesses by
  `(query_hash, kind)` keeping most-recent ts; `distinct_used_queries` counts
  distinct `query_hash`. One batched SQL scan over the four access kinds, fanned
  chunk→handles. `surfaced_vs_used_distinct_query_gate_and_classification`
  asserts it.

### Composition / one-pass discipline

- [x] **ONE boot pass fills all factors.** `precompute_salience_factors`
  (commands.rs:2085) does pass A (global mean) + B (exemplars) + C (per-handle
  spec/value/neg) off `reanchor_inputs()` + one `list_evidence_all()` + one
  `fetch_chunks_by_ids()` + one `surfaced_vs_used()` + one
  `concept_support_by_coord()`. Batched, not per-handle. Each axis's reads are
  skipped when its toggle is off. The whole pass is skipped when
  `scorer_v2=false`.

- [x] **`shannon_entropy` and `surfaced_vs_used` each have exactly ONE impl.**
  `grep` confirms a single `fn shannon_entropy` (salience.rs:136) imported by
  health.rs, and a single `fn surfaced_vs_used` (threads.rs:1279) consumed by
  BOTH the value boot pass (commands.rs:2236, takes the `Vec<UsedAccess>` leg)
  and the health pull leg (server.rs:999, takes the `UseLedger` leg). Also one
  `center`, one `negative_penalty`, one `UseLedger`.

- [x] **Per-axis toggles independent; self-audit NOT flag-gated.**
  `SalienceScorer::from` ANDs each axis with `scorer_v2`; the scorer gates each
  damper on its own `*_enabled`. `compute_salience_health` (server.rs:947)
  checks no flag — it surfaces whatever the live scorer produced.
  `health_runs_under_both_scorers` asserts it.

### Surface / safety

- [x] **`ScoreAttribution` adds axes with serde-default; `salience_health` is
  skip-if-none.** `specificity`/`value` use
  `#[serde(default="default_neutral_multiplier")]` (→1.0), `neg_penalty`
  `#[serde(default)]` (→0.0); a legacy-payload round-trip test proves old JSON
  decodes to the neutral identity. `RecallStats.salience_health:
  Option<SalienceHealth>` is `#[serde(skip_serializing_if="Option::is_none")]`.
  Old MCP clients parse cleanly.

- [x] **`stop_handles` / `forced_stop` left wired.** `with_stop_handles` chained
  at every construction site (commands.rs:360/391/677/709/1431);
  `forced_stop = stop_handles.contains(handle)` flows into the scorer at both
  call sites; `is_stop = forced_stop || is_stop_handle(...)` preserved. When
  `is_stop` fires, the floor is `familiarity_floor(0)` and the `spec·val`
  damper is a genuine no-op for that handle (no double-count — design §1.4).
  `curated_handles()` added for the health curated-ratio metric.

- [x] **No commit to `main`; all on `feat/autonomous-salience`; CHANGELOG entry.**
  Confirmed (6 commits, clean tree). CHANGELOG documents axes 1+4 accurately.

### Concurrency / purity

- [x] **Boot pass never holds an `Inner` guard across a setter.**
  `precompute_salience_factors` computes everything into locals, never reads
  `inner`, and calls `set_negative_exemplars` / `set_salience_factors` (each
  takes its own brief `.write().await`) as the only guard points. No deadlock
  on the non-reentrant tokio `RwLock`.

- [x] **Push leg: unhealthy-only, TTL-cached, computed outside the lock.**
  `salience_health()` (server.rs:884) computes the block *after* dropping the
  cache `Mutex` (explicit comment: "never hold a std Mutex across an await"),
  caches for `[salience.health].ttl_secs` (default 30s); `salience_health_push`
  returns `Some` only when `unhealthy==true`.

### A/B scoreboard (I4 metrics)

- [~] **New scorer: entropy UP / curated_ratio DOWN / never-used shrinking.**
  The four metrics are *computed* and unit-tested in isolation, but the live
  before/after comparison against the baseline (flat ~0.25 ribbon) is the
  V-phase live run — NOT yet executed, and no fixture stands in for it (Finding 3).

---

## Findings

### Finding 1 — `damper_floor > 1.0` (or NaN) panics `compute_score_parts` even with the flag OFF. [MEDIUM]
`crates/attention/src/lib.rs:800` —
`let damp = (1.0 - cfg.neg_gamma * neg).clamp(eps, 1.0);` runs on **every**
score, regardless of any toggle (the `spec`/`val` clamps are inside their
`if *_enabled` guards, but `damp` is not). `eps = cfg.damper_floor` is read
straight from config with **no validation** (config.rs: no range check; the
`resolve` is a bare `cloned().unwrap_or_default()`). `f32::clamp` panics when
`min > max` — empirically confirmed: `1.0_f32.clamp(1.5, 1.0)` →
`panicked: min > max, or either was NaN`. So an operator who sets
`[salience] damper_floor = 1.5` (or any `>1.0`, or NaN) panics the thread
scorer on every `surface`/`score_thread` call — and this fires **even with
`scorer_v2=false`**, breaking the "flag-off is always safe" guarantee. Same
class: `neg_gamma`/`negative_tau`/`value_neutral` are unvalidated, though only
`damper_floor>1.0` and NaN reach the panic (others are absorbed by later
clamps or the `(1.0-tau).max(EPSILON)` guard).
**Fix:** clamp/validate `damper_floor` into `[0, 1)` (and the other knobs into
their documented ranges) in `SalienceSettings::resolve` or
`SalienceScorer::from`; or compute `damp` as
`(1.0 - cfg.neg_gamma*neg).clamp(eps.min(1.0), 1.0)`. Cheap, removes a
misconfig DoS.

### Finding 2 — `negative_penalty_keeps_recoverable_concept` does not exercise `negative_penalty()` and oversells the k=3/ostk-cache claim. [LOW]
`crates/attention/src/lib.rs:2885` — the test **injects `neg_penalty=1.0`**
into the factors map for both `ostk-cache` and `pure-noise`, then proves the
*bounded damp* (γ=0.8) lets the live-resonant handle outrank the non-resonant
one. That is a valid recoverability property, but it proves nothing about the
**k=3 centered-kNN mechanism** R3 names as the actual ostk-cache survival
path. The k=3-vs-k=1 distinction is tested separately
(`negative_penalty_k3_robust_to_single_collision`) on a 4-dim toy fixture, and
`centering_is_required_for_separation` on a synthetic anisotropy. So the
"ostk-cache survives" guarantee is split across two tests, **neither of which
is the named end-to-end hard case** (a real concept at centered-cosine ~1.0 to
a rejected sub-term it shares an anchor chunk with, with `negative_penalty`
producing a recoverable < k=1 value). The mechanism is plausibly correct, but
the test names imply coverage that is not there.
**Fix:** add a test that drives the real `negative_penalty()` with an exemplar
set where one exemplar collides with the query anchor and the rest are far,
asserting the k=3 penalty is bounded enough that the surface keeps the handle —
i.e. wire the pure-fn k=3 result through the surface, not an injected 1.0.

### Finding 3 — The integrated A/B harness (P1–P4) does not exist as a CI test/bench. [MEDIUM — V-gate coverage gap]
`plan.md` §"A/B harness" / P-4 promised a **fixture `#[test]` in
crates/attention** that loads a thread-set, computes factors via the boot pass,
ranks 3 ways (v1 / v2-all / v2-stopset-disabled), and asserts P1 (named
coherent-noise set ranks below named real-concept set), P2 (specificity-alone
reproduces the stop-set effect), P3 (ostk-cache stays above
`ARCHIVE_THRESHOLD` and above all P1 noise), P4 (curated_ratio falls without P1
regressing). **No such test or bench exists** — grep for the named handle sets
(`turn-digest`/`squad-lead` + `cognitive-memory`/`ostk-cache` together in a
ranking assertion) finds only source comments and the health-metric unit
fixtures. What exists are per-axis unit tests that each prove a *fragment*
(one diffuse vs one concentrated handle; the metric *computes*; the bounded
damp recovers). So the plan's own checklist boxes "A/B P1/P2/P3/P4" are
satisfiable today only by the **live MCP-driven V-phase run**, which is
operator-side and not yet performed. This is the single biggest gap for a
"make it so / default-on" decision: there is no automated regression guard that
the new scorer actually achieves the THESIS ordering.
**Fix:** build the promised fixture harness before default-on (it gates every
future scorer change in CI), or explicitly down-scope the V gate to "live run
only" and record the live P1–P4 results in this doc before flipping the default.

### Finding 4 — The bit-identical guard test is weaker than `==` and is not anchored to `main`. [LOW]
`crates/attention/src/lib.rs:2767` `specificity_one_reproduces_current_scorer`
uses `approx_eq(scores[0], scores[1], 1e-6)` and compares **flag-off vs
flag-on-with-explicitly-neutral-factors within the branch** — it does not
compare against a value captured from `main`. The source-level analysis (above)
shows the off-path is bit-exact, so this is not a correctness defect, but the
test would not catch a future sub-`1e-6` drift, and it does not by itself prove
"identical to *pre-branch*." (The branch surface scores go through sort/float
aggregation, so a strict `==` may be defensible only at the
`compute_score_parts` unit level.)
**Fix (optional):** add a `compute_score_parts`-level assertion with `==` on
the raw `ScoreParts.score` for flag-off vs a hand-computed v1 value, or snapshot
a `main` score into the test as the anchor.

### Finding 5 — Unused-import warning introduced in a test. [TRIVIAL]
`crates/query/tests/freshness_act_r.rs` — `unused import: AccessWeights`
(warn-only; clippy/test both still pass). Drop the import.

---

## Adversarial probes that did NOT find a problem

- **Flag-off differs from main:** no — every added factor is `×1.0` and the
  branch/lift logic is untouched (modulo Finding 1's misconfig panic).
- **Value < v_neutral on some path:** no — `[v_neutral,1]` by construction.
- **Three dampers over-suppress a legit handle:** in v1 `val=1.0` always, so
  the floor product is two dampers, and each is ε-floored; `is_stop` short-
  circuits the `spec·val` damper entirely. No over-suppression to zero.
- **Negative-penalty mean-centers before kNN, k=3, damp bounded:** yes —
  `center()` is mandatory and applied; `negative_knn_k` default 3 (clamped to
  `[1, sims.len()]`, and `sims` is non-empty by the `exemplars.is_empty()`
  guard, so no `clamp(1,0)` panic); `γ=0.8<1` ⇒ `damp ≥ 1−γ = 0.2 > 0`.
- **Boot pass deadlock on the non-reentrant RwLock:** no — setters are the only
  guard points and are never nested under a held guard; the pass reads only
  `threads`/`corpus`/`attention` APIs.
- **Push leg holds a lock across an await:** no — compute is outside the cache
  Mutex.
- **Old MCP clients break on the new fields:** no — serde-default + skip-if-none,
  with a legacy-decode round-trip test.
- **Self-audit accidentally flag-gated:** no — `compute_salience_health` checks
  no flag.

## Recommendation

Land the branch **with `scorer_v2` default-OFF** (its shipped state) — it is
safe and well-tested in that configuration. Before flipping `scorer_v2`
default-ON, require: (a) Finding 1 fixed (config validation — it is a flag-off
panic), (b) Finding 3 closed (either the CI A/B fixture harness or recorded
live P1–P4 results), and ideally (c) Finding 2's end-to-end ostk-cache test.
Findings 4 and 5 are nice-to-haves.
