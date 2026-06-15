# Design — Unified Salience Scorer

> Team `salience`. The single input to the Plan phase (P) and the four
> implementation tasks (I1 specificity, I2 negative-transfer, I3 value,
> I4 self-audit). Synthesizes THESIS.md + research-specificity.md (R1),
> research-value.md (R2), research-negative-transfer.md (R3),
> research-self-audit.md (R4). It is precise enough to build each axis from
> without re-deriving the research; where it depends on a research number,
> the anchor is cited. Code line numbers are against the gens read during
> design (lib.rs gen 103, commands.rs gen 144, config.rs gen 49,
> attention.rs gen 13) and may drift by a few lines — the symbol names are
> the durable anchors.

---

## 0. The one-paragraph synthesis

Today `compute_score_parts` (`crates/attention/src/lib.rs:717`) scores a
thread as `decay_term + ALPHA·resonance + BETA·off_diagonal_lift`, where
`decay_term = familiarity_floor(resonance_count)·fade·decay` and the only
brake is the binary `is_stop` flag (the curated `stop_handles` set OR the
frequency cliff `is_stop_handle`). Every term rewards "looks like now, a
lot." We evolve this into

    salience = specificity × value × recency − negative_penalty

by introducing **three precomputed per-handle scalars** —
`specificity ∈ [0,1]`, `value ∈ [v_neutral,1]`, `neg_penalty ∈ [0,1]` —
computed in **one boot-time pass**, carried in **one store-level map** on
`InMemoryAttention`, and folded into the scorer as **two multiplicative
factors on the idle floor** (`specificity` damps; `value` is *monotone in
positive evidence* — it only raises proven handles toward 1.0, never damps
below the no-evidence point) plus **one bounded multiplicative damp on the
resonance-driven terms** (`1 − γ·neg_penalty`). **v1 ships `value` as a
constant-1.0 pass-through** (`v_neutral = 1.0`): a `[0,1]` floor multiplier can
only damp, and the only legitimate downward signal — "surfaced-but-never-used"
— needs the `ThreadSurfaced` event, which v1 defers; so v1's behavior is driven
by specificity + negative_penalty, and the value join runs as
attribution-only scaffold. This guarantees a decision-cited handle is never
*more* suppressed than unproven noise (the value-axis monotonicity invariant,
§4.5). The curated `stop_handles` set stays wired as `forced_stop` (a safety
net and an A/B control); the design proves it becomes *redundant* under
specificity rather than ripping it out. Self-audit (axis #4) is pure
observation, **not** flag-gated, watches both the old and new scorer, and its
four metrics double as the A/B scoreboard.

---

## 1. The unified scorer (exact expression)

### 1.1 Current expression (verbatim, `lib.rs:749-760`)

```rust
let is_stop = forced_stop || is_stop_handle(state.mentions, state.resonance);
let resonance_floor = if is_stop {
    familiarity_floor(0)
} else {
    familiarity_floor(state.resonance)
};
let floor = resonance_floor * state.fade_multiplier;
let decay_term = floor * (-decay_rate(state.mentions) * dt_days).exp();
let resonance_term = ALPHA * resonance;                  // ALPHA = 1.0
let lift_term = BETA                                     // BETA  = 0.5
    * off_diagonal_lift(state.tension, resonance, state.resonance, is_stop);
let score = decay_term + resonance_term + lift_term;
```

Three additive terms. `decay_term` is the **idle floor** (the lever the
stop-set exists to clamp); `resonance_term` is the **live** cosine to the
current attention vector; `lift_term` is the off-diagonal surprise bonus
(already `0.0` for 39/40 live pages — R4 §2).

### 1.2 New expression (exact)

`compute_score_parts` gains one parameter, a `SalienceFactors` lookup
(resolved by the caller from the store-level map — see §2), and a config
snapshot. When the master flag is off, `factors` is the neutral identity
`{specificity: 1.0, value: 1.0, neg_penalty: 0.0}` and the expression is
**bit-identical to 1.1** (proven by unit test, §8 I1-test-2).

```rust
fn compute_score_parts(
    state: &ThreadState,
    attention_vec: &[f32],
    now: DateTime<Utc>,
    forced_stop: bool,
    factors: SalienceFactors,        // NEW: {specificity, value, neg_penalty}
    cfg: &SalienceScorer,            // NEW: resolved [salience] knobs + per-axis toggles
) -> ScoreParts {
    let resonance = cosine_similarity(&state.anchor, attention_vec);
    let dt_secs = (now - state.last_touched_at).num_seconds().max(0) as u64;
    let dt_days = dt_secs as f32 / 86_400.0;

    // --- axis gating (each individually toggleable; identity when off) ----
    let spec = if cfg.specificity_enabled { factors.specificity } else { 1.0 };
    let val  = if cfg.value_enabled       { factors.value       } else { 1.0 };
    let neg  = if cfg.negative_enabled    { factors.neg_penalty } else { 0.0 };
    let damp = 1.0 - cfg.neg_gamma * neg;        // ∈ [1 − γ, 1]; γ ≈ 0.8 → never to 0

    let is_stop = forced_stop || is_stop_handle(state.mentions, state.resonance);

    // (1) IDLE FLOOR — gated by specificity × value (both dampers, both [0,1]).
    //     This is the "does it distinguish, and did it ever pay off?" lever.
    //     A diffuse (spec→0) handle collapses to the unresonant baseline at
    //     idle — the principled, continuous form of the is_stop cliff.
    //     VALUE is monotone in positive evidence and can only damp BELOW its
    //     neutral point via a wasted signal, which v1 defers (§4.3-4.5); so in
    //     v1 `val == 1.0` for every handle (constant pass-through) — value is
    //     attribution-only scaffold here, NOT a cold-start damper. Only
    //     specificity (live evidence today) and `damp` move the v1 floor.
    let earned_floor = familiarity_floor(state.resonance) * spec * val;
    let resonance_floor = if is_stop { familiarity_floor(0) } else { earned_floor };

    // (2) NEGATIVE-TRANSFER DAMP — bounded multiplicative on the floor AND on
    //     the live resonance term, so a handle that is a near-twin of something
    //     we rejected is suppressed but, because damp ≥ 1−γ > 0, can still climb
    //     on a strong fresh resonance (recoverable — the ostk-cache case).
    let resonance_floor = resonance_floor * damp;

    let floor = resonance_floor * state.fade_multiplier;
    let decay_term = floor * (-decay_rate(state.mentions) * dt_days).exp();
    let resonance_term = ALPHA * resonance * damp;

    // (3) OFF-DIAGONAL LIFT — a "surprise" from a diffuse / negatively-twinned
    //     handle is not a surprise. Treat low specificity or high neg like
    //     is_stop for lift purposes (lift's own is_stop arg zeroes the bonus).
    let lift_is_stop = is_stop
        || spec < cfg.specificity_lift_cutoff
        || neg  > cfg.negative_lift_cutoff;
    let lift_term = BETA
        * off_diagonal_lift(state.tension, resonance, state.resonance, lift_is_stop);

    let score = decay_term + resonance_term + lift_term;
    ScoreParts { score, resonance, lift_term, dt_secs,
                 // NEW attribution axes (see §1.4)
                 specificity: spec, value: val, neg_penalty: neg }
}
```

### 1.3 Where each axis multiplies — the resolved composition

This is the central composition decision the THESIS demanded (R1 §5-Q1,
R2 §combine, R3 §4 all flag it). Resolution:

| Axis | Maps onto | Form | Why |
|---|---|---|---|
| **specificity** (×) | the **idle floor** `familiarity_floor(resonance_count)` | multiplicative `[0,1]` damper | the floor is the idle-dominance lever the stop-set patches; gating it is the continuous `is_stop_handle` (R1 §3.2). Leaving the *live* `resonance_term` ungated preserves "light up while discussed, never idle-dominate" (lib.rs:745-748). |
| **value** (×) | the **idle floor**, alongside specificity | multiplicative `[v_neutral, 1.0]`, **monotone in positive evidence; v1 `v_neutral = 1.0` (pass-through)** | value is "did surfacing ever pay off?" Because it is a `[0,1]` floor multiplier it can only damp, and the only legitimate damp is a *wasted* signal (deferred, §4.4); so positive evidence may only raise value toward 1.0, never lower it (the §4.5 invariant). v1 pins `v_neutral = 1.0` ⇒ value is attribution-only and never resurrects or suppresses idle dominance; the ceiling-raiser regime (`v_neutral < 1.0`) arrives with `ThreadSurfaced`. |
| **negative_penalty** (−, as `×(1−γ·neg)`) | the **floor AND the live `resonance_term`** | bounded multiplicative damp, `γ<1` | R3 is emphatic: SOFT bounded, never a hard gate, so a real concept reusing rejected sub-vocab (ostk-cache) stays recoverable on fresh resonance. Damping the live term too is what lets a *novel* harness term be pre-damped even while "discussed." |
| **recency** | unchanged: `decay_term`'s `exp(−decay_rate·dt_days)` and `last_touched_at` | (already present) | THESIS: "the system already has recency (decay)." We do not touch it. |

**Floor product ordering and the ε guard.** The floor becomes
`familiarity_floor(resonance_count) · spec · val · damp`. Three `[0,1]`
multipliers compounding could over-suppress a *high-value-but-diffuse* or
*high-specificity-but-collision* handle (R1 §5-Q1). Two protections:

1. **Value is monotone-up-only (§4.5):** by the value-axis invariant, positive
   evidence may only *raise* `val` toward 1.0, never lower it. In **v1**
   `v_neutral = 1.0`, so `val = 1.0` for every handle and only `spec` and
   `damp` move the floor — the product is effectively two dampers, not three,
   so the over-suppression hazard does not even arise in v1. When the
   ceiling-raiser regime lands (`v_neutral < 1.0`, with `ThreadSurfaced`),
   `val` *raises* floors for proven handles and the only sub-neutral path is
   the wasted-signal penalty — it still never drops a never-used-but-specific
   handle below an unevidenced peer of equal specificity (the invariant).
2. **ε floor on each damper:** clamp `spec`, `val`, and `damp` to
   `[cfg.damper_floor, 1.0]` (suggest `damper_floor = 0.02`) so no single
   axis can zero the score outright; demotion is to the *unresonant
   baseline*, matching `is_stop`'s `familiarity_floor(0)` behavior, not to
   nothing. This is the "gate, don't delete" doctrine (lib.rs:742).

Why `−negative_penalty` is implemented as `×(1−γ·neg)` and not a literal
subtraction: a subtractive term has unbounded units relative to the
`[0.1,1.0]`-scaled floor and would need its own calibration; the bounded
multiplicative form composes cleanly with the two `[0,1]` dampers (R3 §4
"composes multiplicatively with axis-1 specificity rather than fighting it")
and is the exact shape R3 validated (AUC 0.917 with this damp form).

### 1.4 Composition with the existing `forced_stop` / stop-set path

The curated `stop_handles` set and `forced_stop` **stay wired exactly as
today** (lib.rs:1475 surface, lib.rs:1681 score_thread). They are:

- **A safety net during A/B**: while `scorer_v2` is being validated, the
  hand-list still clamps known noise, so a specificity miscalibration cannot
  regress the live surface.
- **An A/B control / redundancy proof**: the success bar (THESIS, §7 here)
  is that specificity *alone* (stop-set disabled in the A/B run) reproduces
  the stop-set's ranking effect. The self-audit `curated_ratio` metric (R4
  Metric 2) measures this redundancy directly — if it falls without surface
  quality dropping, the stop-set is provably redundant and the V phase can
  retire `default_weaver_stop_handles`.

`forced_stop` and specificity **agree on the same handles by independent
evidence** (R1 §3.5): `forced_stop` is a hand-curated boolean; specificity
is unsupervised entropy. When both fire, `is_stop` already clamps the floor
to `familiarity_floor(0)`, so the `spec·val` damper on `earned_floor` is a
no-op for that handle (the `if is_stop` branch wins) — they do not
double-count. The overlap is the proof; keeping both is intentional.

---

## 2. ONE boot-time precompute pass + the storage decision

### 2.1 RESOLVED storage decision

**All three per-handle scalars (specificity, value, neg_penalty) live in ONE
store-level map on `InMemoryAttention`, NOT on `ThreadState`.** This resolves
the R1-vs-R2 split (R1 proposed `ThreadState.specificity: f32`; R2 proposed
`InMemoryAttention.value_scores: Arc<HashMap>`) in favour of R2's mechanism,
for both, plus neg_penalty.

```rust
/// Precomputed per-handle salience factors. Neutral identity = no effect.
#[derive(Debug, Clone, Copy)]
pub struct SalienceFactors {
    pub specificity: f32,   // [0,1], 1.0 = maximally discriminating / neutral
    pub value: f32,         // [v_neutral,1], monotone in positive evidence; v1 const 1.0 (§4.5)
    pub neg_penalty: f32,   // [0,1], 0.0 = no negative proximity (neutral)
}
impl Default for SalienceFactors {
    fn default() -> Self { Self { specificity: 1.0, value: 1.0, neg_penalty: 0.0 } }
}

// On InMemoryAttention, next to `stop_handles: Arc<HashSet<String>>` (lib.rs:871):
salience_factors: Arc<HashMap<String, SalienceFactors>>,   // keyed by handle string
negative_exemplars: Arc<Vec<Vec<f32>>>,                    // centered+normalized (R3)
global_anchor_mean: Arc<Vec<f32>>,                         // for center() (R3)
```

**Justification (why the map, not `ThreadState` fields):**

1. **The boot pass runs after construction.** `re_anchor_threads_from_corpus`
   (commands.rs:1967) runs *after* `InMemoryAttention::with_embedder(...)
   .with_stop_handles(...)` (commands.rs:1428-1431) and after
   `replay_chain_into_attention`. A builder like `with_stop_handles` (consumed
   at construction) cannot carry data the pass computes *later*. So the
   mechanism must be a **post-construction setter** (`set_salience_factors`,
   `set_negative_exemplars`) on the `Arc<InMemoryAttention>` — the map pattern
   accommodates this naturally; per-handle `ThreadState` fields would force the
   pass to re-`seed_*` every thread with the new scalars.

2. **`ThreadState` has four construction sites.** `seed_counters` (lib.rs:985),
   `seed_anchor` (lib.rs:1416), `fold` (lib.rs:1576), `familiarize`
   (lib.rs:1607). Adding `specificity`/`value`/`neg_penalty` fields means
   touching all four `or_insert_with` blocks and keeping them in sync, with no
   benefit — the scorer reads the factors by handle anyway. The map keeps
   `ThreadState` unchanged.

3. **Cross-scope correctness.** A handle exists in multiple scopes (the boot
   `replay`/`substrate` scope plus live `ambient` scopes — see `surface`'s
   cross-scope loop, lib.rs:1456). A per-`ThreadState` field would have to be
   replicated into every scope's copy and could drift; a single handle-keyed
   map is the natural "one factor per handle, scope-independent" store, exactly
   like `stop_handles.contains(handle)` already is (lib.rs:1475, 1681).

4. **Refresh is an `Arc` swap.** The consolidate/ambient cadence recomputes the
   map and swaps the `Arc` (like `stop_handles`), with zero per-thread
   mutation and no write-lock on the hot `scopes` map.

**Setter, not builder, for the factors map** (because the data is computed
after construction):

```rust
impl InMemoryAttention {
    /// Install/replace the precomputed per-handle salience factors. Called
    /// once at boot after `re_anchor_threads_from_corpus` computes them, and
    /// again on each consolidate cycle (Arc swap; cheap, lock-free for readers).
    pub fn set_salience_factors(&self, factors: HashMap<String, SalienceFactors>) { /* store via ArcSwap or Mutex<Arc> */ }
    pub fn set_negative_exemplars(&self, mean: Vec<f32>, exemplars: Vec<Vec<f32>>) { /* idem */ }
}
```

Implementation note for P/I: `stop_handles` is a plain `Arc<HashSet>` set at
build time and never mutated. The factors map needs *interior mutability* to
be swapped at runtime by the consolidate pass. Use `arc_swap::ArcSwap`
(already a workspace-friendly pattern) or `Mutex<Arc<HashMap<...>>>` cloned
under a short lock in the scorer's caller — **not** inside `compute_score_parts`
(which must stay pure, no locks). The caller (`surface`/`score_thread`) reads
the current `Arc` once per call before the per-thread loop and passes
`factors.get(handle).copied().unwrap_or_default()` per thread.

### 2.2 The single pass (fills all three + the store-level exemplars/mean)

The pass extends `re_anchor_threads_from_corpus` (commands.rs:1967), which
**already** iterates every thread, holds `&Arc<ThreadsDb>`,
`&Arc<CorpusStore>`, and `&InMemoryAttention`, and fetches anchor embeddings.
We fold the three precomputes into that same walk so it is **one scan, not
three**.

```
re_anchor_threads_from_corpus(threads, corpus, attention):
    inputs = threads.reanchor_inputs()              # existing: (handle, tier, chunk_id, anchor_vec)
    ... existing anchor seeding (unchanged) ...

    # ---- NEW: single salience precompute, after anchors are seeded ----
    if !cfg.salience.scorer_v2 { return Ok(seeded) }    # skip work when flag off

    # (A) global anchor mean — needed by negative-transfer centering (R3 §1).
    #     Mean of all normalized thread anchors we just seeded (inputs carry anchor_vec
    #     or we fetched it). One pass over `inputs`.
    global_mean = mean( normalize(anchor) for each input with an anchor )

    # (B) negative exemplar set (R3 §1-2) — STORE-LEVEL, not per-handle.
    neg_raw = threads.dormant_anchor_vecs()                    # 279 dormant handles' anchor_vec (NEW reader; §3.2)
            ++ threads.rejected_concept_anchors()              # 47 rejected concepts' evidence anchor_vec (NEW reader; §3.2)
    neg_exemplars = [ center(v, global_mean) for v in neg_raw ]   # center = normalize(normalize(v) − global_mean)
    # optional scrub (R3 §1 hardening): drop any exemplar with centered cosine > 0.97 to an ACTIVE-concept anchor
    attention.set_negative_exemplars(global_mean, neg_exemplars)

    # (C) per-handle factors — ONE map, all three scalars.
    #     Batched joins so we read the ledger/evidence once, not per-handle (R4 cost model).
    ev_by_handle   = threads.list_evidence_all()               # handle -> [evidence rows]  (batched; §3)
    chunk_meta     = corpus.fetch_chunks_by_ids( union of all evidence chunk_ids )  # chunk_id -> source_id (R1 §1b)
    access_hist    = threads.access_history( union of all evidence+anchor chunk_ids, since )  # SHARED join (§4.2)
    judgment_set   = corpus chunks where source ∈ {ostk_decision, ostk_needle}     # for value_judgment (R2)
    active_coords  = activation.concept_support_by_coord(since)                     # for value_judgment (R2)

    factors = {}
    for handle, evidence in ev_by_handle:
        spec = specificity_from_evidence(evidence, chunk_meta, cfg.salience)        # R1 §3.1 (uses shannon_entropy)
        val  = value_from(handle, evidence, access_hist, judgment_set, active_coords, cfg.salience)  # R2
        neg  = negative_penalty(anchor_of(handle), global_mean, neg_exemplars, cfg.salience)         # R3 §3
        factors[handle] = SalienceFactors { specificity: spec, value: val, neg_penalty: neg }
    attention.set_salience_factors(factors)
    return Ok(seeded)
```

This honours the THESIS cross-axis directive (R3 §slot note: "three boot-time
precomputes now converge here; design the boot wiring as ONE pass"). The pass
reads the corpus/threads DBs **once each** for the shared joins, then computes
all three scalars per handle from in-memory data. Cost is `O(total
evidence_links + |neg_exemplars|·dim)` once per boot — the same order the
re-anchor walk already pays. Zero added cost at score time beyond three f32
ops and (for neg) one cached centered-anchor kNN (§3.3 perf note).

### 2.3 Where store-level vs per-handle data lives — summary

| Datum | Scope | Home | Built by |
|---|---|---|---|
| `specificity` | per-handle | `salience_factors[handle]` | pass (C) |
| `value` | per-handle | `salience_factors[handle]` | pass (C) |
| `neg_penalty` | per-handle | `salience_factors[handle]` | pass (C) |
| `global_anchor_mean` | store-level | `InMemoryAttention.global_anchor_mean` | pass (A) |
| `negative_exemplars` | store-level (the SET) | `InMemoryAttention.negative_exemplars` | pass (B) |

`neg_penalty` is computed per-handle in the pass (so the scorer needs no
embedding work), but its *inputs* — the exemplar set + global mean — are
store-level. (Alternative: store only the set+mean and compute `neg_penalty`
lazily at score time from `state.anchor`. **Rejected for v1**: keeps the
scorer pure and lock-free, and the anchor is static between boots so
precomputing once is strictly cheaper — R3 §5 perf note endorses precompute.)

---

## 3. Shared machinery, factored once

The THESIS cross-axis sections name two pieces of shared machinery that MUST
be written once. Plus the negative-transfer primitives.

### 3.1 `shannon_entropy(&[f32]) -> f32` (R1 + R4)

Used by R1 (per-handle co-occurrence entropy over the source-doc histogram)
and R4 Metric 1 (entropy of the active surface's score-share distribution).
**One impl.**

- **Home:** a new module `crates/attention/src/salience.rs` (the natural home
  — it also hosts `center`, `negative_penalty`, `specificity_from_evidence`,
  and the `SalienceFactors`/`SalienceScorer` types; all pure, unit-testable in
  isolation, next to `off_diagonal_lift` in spirit). R4's health compute
  (`crates/attention/src/health.rs`, §6) imports it from there.
- **Signature & contract:** takes a slice of non-negative weights (counts or
  scores), normalizes to a probability distribution internally, returns raw
  `H = −Σ pᵢ ln pᵢ`. Callers normalize by `ln(N_eff)` themselves (R1 wants
  `1 − H/ln(N_eff)`; R4 wants `H/ln(N)`), so the helper stays
  normalization-agnostic. Empty / single-bin / all-zero → `0.0`.

### 3.2 New `ThreadsDb` readers for the negative label sources (R3)

R3 §1 names the exact sources. Two new readers on `ThreadsDb`, mirroring
existing decode paths:

- `dormant_anchor_vecs() -> Vec<Vec<f32>>` — the 279 dormant handles'
  `anchor_vec` BLOBs (`tension='dormant' AND anchor_vec IS NOT NULL`),
  decoded via existing `bytes_to_f32_vec`. Mirrors `reanchor_inputs()`
  (threads.rs:1935) filtered to dormant. **STRONG source** (432/490 anchored;
  R3 §1A).
- `rejected_concept_anchors() -> Vec<Vec<f32>>` — evidence-row `anchor_vec`
  for `status='rejected'` concepts, de-duplicated per concept by mean.
  Mirrors `evidence_to_reconcile()` (concepts.rs:1229) filtered to rejected.
  **WEAKER source** (47 handles; R3 §1B).
- **Excluded** (R3 §1 emphatic): the 10 bare session-rejected handle *strings*
  (`follow-up`, `pre-existing`, `re-touch`…) have ZERO evidence/anchor. **Do
  NOT embed the bare handle string** — `potion` embeds 2-token handles
  noisily, and the dormant-thread anchors already cover that vocab space.
  Specificity (axis #1) catches these instead.

### 3.3 Negative-transfer primitives (R3 §3) — pure fns in `salience.rs`

```rust
/// Mean-center then re-normalize. R3 §TL;DR: this is MANDATORY — raw cosine to
/// a single centroid is useless (+0.0195 separation, anisotropy); centering is
/// a 5× improvement on its own.
fn center(v: &[f32], global_mean: &[f32]) -> Vec<f32> {
    // normalize(normalize(v) − global_mean)
}

/// kNN to nearest negative exemplars in centered space, then a floored,
/// rescaled penalty. R3 §3: k = 3 (AUC 0.917; k=1 is 0.957 but BRITTLE — the
/// ostk-cache cosine-1.0 collision; k=3 averages the neighborhood). τ ≈ 0.45.
fn negative_penalty(anchor: &[f32], global_mean: &[f32], exemplars: &[Vec<f32>], cfg: &SalienceScorer) -> f32 {
    if exemplars.is_empty() || anchor.is_empty() { return 0.0; }   // neutral
    let a = center(anchor, global_mean);
    let mut sims: Vec<f32> = exemplars.iter().map(|e| cosine_similarity(&a, e)).collect();
    sims.sort_unstable_by(|x, y| y.partial_cmp(x).unwrap_or(Ordering::Equal));
    let k = cfg.negative_knn_k.min(sims.len());                    // default 3
    let prox = sims[..k].iter().sum::<f32>() / k as f32;           // mean top-k cosine
    ((prox - cfg.negative_tau) / (1.0 - cfg.negative_tau)).clamp(0.0, 1.0)   // τ-floored, rescaled to [0,1]
}
```

Reuse `cosine_similarity` (lib.rs:238) for the dot products. The k=3 + bounded
soft penalty is **the ostk-cache survival mechanism** — design/I2 must prove
it (§8 I2-test-1).

### 3.4 The surfaced-vs-used ledger join (R2 scores it, R4 measures it)

ONE helper, used by both the value precompute (R2/I3) and the self-audit
never-used metric (R4/I4). This is the THESIS "write the join helper once"
directive.

- **Home:** a reader method on `ThreadsDb` (it owns both `evidence_links` and
  `chain_log`): `surfaced_vs_used(handles, since) -> HashMap<String, UseLedger>`
  where `UseLedger { surfaced: u32, used_weighted: f32, distinct_used_queries: u32, unattributable: bool }`.
- **Mechanics** (R2 §a, R4 §3 Metric 3):
  - For each handle, union `{anchor_chunk_id} ∪ {evidence_links.last_resolved_chunk_id for Active links}`.
  - **Batched**: call `ChainLogReader::access_history(union_of_all_chunk_ids, since)`
    ONCE over the union across all handles (R4 cost model: bound it like
    `thread_query`'s `backfill_cross_axis` — one batched fetch, not per-handle).
  - Classify each access by `AccessKind`: `LensIncluded` = "surfaced" (weight
    0.5); `ExplicitRecall`/`OperatorSelected`/`RecallFault` = "used".
  - **Distinct-`query_hash` gate** (doctrine — activation.rs:326, the
    `salience-vs-familiarity` discipline): collapse accesses sharing a
    `query_hash` so a chatty recall loop can't fake "used."
  - A handle with no evidence links → `unattributable: true` (R4 §5: flag
    separately, don't miscount as never-used).
- **R2 consumes it** to compute `value_use` (apply `activation.rs` curves —
  §4 below). **R4 consumes it** to compute Metric 3 (`surfaced ≥ N ∧ used == 0`).
  Same data, two readers, one helper.

---

## 4. Value specifics (R2)

`value ∈ [v_neutral, 1.0]`, computed in pass (C), cached as
`SalienceFactors.value`, folded onto the idle floor as a `[0,1]` multiplier
(§1.3). **The hard invariant (§4.5): value is monotone non-decreasing in
positive evidence — a handle WITH use/judgment evidence is never more damped
than an identical handle with none.** Because `value` is a `[0,1]` floor
multiplier it can only *damp*; the only thing that may legitimately push it
**below** the no-evidence neutral point is a genuine *wasted* signal
("surfaced-but-never-landed"), and that signal needs the `ThreadSurfaced`
event which **v1 defers** (§4.4). Therefore in **v1 there is no valid negative
value signal**, so `v_neutral = 1.0` and value is a **protective pass-through
(constant 1.0)**: positive evidence is recorded as attribution but cannot pull
the floor below where specificity/negative already put it. The damping regime
(`value < 1.0`) unlocks with `ThreadSurfaced` in the second increment.

### 4.1 `value_use` — the click-through loop (WEAK fidelity, BUILD FIRST)

R2 is explicit: build the **weak** fidelity ("the thread's content was used")
first — it needs **no schema change**. Strict click-through ("surfaced THEN
used") needs a new `ThreadSurfaced` event (lib.rs:1480) and is **DEFERRED** to
a second increment (§4.4).

```
value_use(H) = squash( Σ over distinct (query_hash, kind) of
                          AccessWeights.weight_of(kind) · act_r_base(age_of(ts)) )
```

- Source: the `surfaced_vs_used` join (§3.4) — `used_weighted` already applies
  `AccessWeights` and the distinct-query gate.
- **REUSE `activation.rs` curves verbatim** (R2 §combine, THESIS cross-axis):
  `act_r_base`, `squash`, `age_hours_floored`, `ACT_R_DECAY_D = 0.5`,
  `AccessWeights` (`explicit_recall = operator_selected = 1.0`,
  `recall_fault = 0.7`, `lens_included = 0.5`). No new curve.
- `operator_selected` has **no producer yet** (threads.rs:356) — weight it (it's
  in `AccessWeights`) but don't lean on it for v1; `explicit_recall` carries
  the loop today.

### 4.2 `value_judgment` — curated-confidence propagation (no new join table)

R2 §b: the curated artifacts (`ostk_decision` 562, `ostk_needle` 212, active
concepts) carry **no handle FK** — the only link is the embedding bridge the
weaver already computes. So:

```
value_judgment(H) = saturating_combine(j_evidence, j_concept)      # 1 − exp(−Σ), like edge_lift_for
  where
   j_evidence = max similarity over H's Active evidence links whose resolved
                chunk has source ∈ {ostk_decision, ostk_needle}   # READ existing Derived links, filter by source
   j_concept  = max active_coords[coord] over H's evidence coords  # intersect concept_support_by_coord (active = conf 1.0)
```

- `j_evidence` reads **already-materialized** `Derived` evidence links
  (weaver.rs:760 writes them; `list_evidence` exposes `category`/`similarity`)
  filtered by the resolved chunk's `source` — no new similarity computation.
- `j_concept` intersects `concept_support_by_coord(since)` (activation.rs:515,
  active concepts are `confidence = 1.0`) with H's evidence coordinates —
  generalizing the lens feature from chunks to handles. **No new join table**
  (R2 §b).
- Saturation `1 − exp(−Σ)` mirrors `edge_lift_for` (activation.rs:443): many
  weak judgment links ≈ one strong one, bounded to 1.

### 4.3 Cold-start and the v1 formula (RESOLVED — §4.5 / §5 cross-reference this)

`value_use ≈ 0` across the board today (R2 live probe: `distinct_queries = 0`
on nearly all active concepts; the recall surface that fires `ExplicitRecall`
is narrow). The naive "average a near-zero `value_use` into present judgment"
formula is **wrong** — with `value` a `[0,1]` floor multiplier and
`w_use = w_judg = 0.5`, a decision-cited handle would get
`0.5·value_judgment ≤ 0.5`, i.e. *more damped than an evidence-less handle that
keeps 1.0*. That inverts axis-2 ("promote what gets used"). The bug is treating
absent use-evidence as a zero to average down, rather than as "no signal yet."

**The correct form: positive evidence only moves value UP from the
no-evidence neutral point, never down.** Express value as the neutral point
plus a bounded positive lift, so it is monotone non-decreasing in evidence:

```
positive(H)  = clamp01(w_use · value_use(H) + w_judg · value_judgment(H))   // ∈ [0,1], 0 = no signal
value(H)     = v_neutral + (1.0 − v_neutral) · positive(H)                  // ∈ [v_neutral, 1.0]
```

- A handle with **no** use/judgment evidence ⇒ `positive = 0` ⇒
  `value = v_neutral` (the minimum — unproven is the floor, not punished).
- A handle **with** evidence ⇒ `positive > 0` ⇒ `value > v_neutral`
  (strictly *less* damped — never more). Monotone by construction. This is
  what I3 test #2/#3 assert (§8).

**v1 sets `v_neutral = 1.0`** (the protective pass-through, §4 headline): there
is no valid *negative* (wasted) signal in v1 to justify any `value < 1.0` — the
"surfaced-but-never-landed → damp" signal is deferred to `ThreadSurfaced`
(§4.4). So in v1, value is a constant 1.0 and the `value_use`/`value_judgment`
computations are **scaffold + attribution**: they run, populate the
`SalienceFactors.value` attribution axis (and the `why`), and exercise the
shared `surfaced_vs_used` join, but the multiplier itself is 1.0 so the floor is
untouched. This matches the THESIS framing — "specificity ships first; value is
the *ceiling-raiser* later." A noise handle with neither use nor judgment is
demoted by **specificity and negative_penalty**, not by value (which is the
double-count R2 §Q1 warned against, concept.rs:116-131's skip-don't-penalize
convention).

**The ceiling-raiser unlock (second increment, with `ThreadSurfaced`):** set
`v_neutral < 1.0` (config `value_neutral`, suggest `0.7`) so the
no-evidence floor is mildly damped and judgment/use evidence *raises* a proven
handle back toward 1.0 — the genuine "promote what gets used" lift — while the
`ThreadSurfaced`-derived wasted penalty supplies the only legitimate path
*below* `v_neutral`. Default weights `w_use = w_judg = 0.5`; the A/B harness
tunes `v_neutral` and the weights together. v1 ships `value_neutral = 1.0`.

### 4.4 Deferred: the wasted signal + the damping regime (`ThreadSurfaced` event)

Second increment, not v1. Add `ChainEvent::ThreadSurfaced { handle, ts }`
emitted at the surface build site (lib.rs:1480) — threads currently have no
surface event (`LensIncluded` logs chunks only). This event is **the only
legitimate source of a negative (wasted) value signal**: it lets `value_use`
compute the true correlation(surfaced_ts, subsequent-use) and, crucially, the
explicit "surfaced-but-never-landed → damp" penalty. *That* is what justifies
lowering `v_neutral` below 1.0 (§4.3 unlock) — a handle that has been surfaced
N times and never used has earned a damp, whereas a handle that was never
surfaced has earned nothing either way and must stay neutral. Until the event
lands, there is no way to distinguish "never surfaced" from "surfaced and
ignored," so v1 cannot damp on absence and pins `v_neutral = 1.0`. Deferring
keeps I3 v1 a pure join with zero schema change; v1 value is monotone-safe
scaffold (§4.5), and the genuine ceiling-raiser + wasted-penalty dynamic
arrives intact in the second increment.

### 4.5 The value-axis invariant (the lead-gate fix — implementers MUST honor)

> **Monotonicity in positive evidence.** For any two handles A, B with
> identical specificity, anchor, counters, and negative_penalty, if A has
> use/judgment evidence and B has none, then `value(A) ≥ value(B)`, hence
> `score(A) ≥ score(B)`. Positive evidence may only *raise* value toward 1.0;
> it may never lower it.

This invariant is what rules out the inverted-cold-start bug. Two equivalent
ways to satisfy it, both compatible with `value` being a `[0,1]` floor
multiplier and with the resolved composition (§1.3) — no change to the
floor-multiplier math:

1. **(v1, shipped)** `v_neutral = 1.0` ⇒ value is a constant-1.0 pass-through;
   the invariant holds trivially (all handles get exactly 1.0). Evidence is
   attribution-only.
2. **(2nd increment)** `v_neutral < 1.0` with `value = v_neutral + (1−v_neutral)·positive`
   ⇒ value rises strictly with `positive`, so an evidenced handle is `≥` an
   unevidenced one; the invariant holds by the monotone form, and the
   `ThreadSurfaced` wasted-penalty supplies the only sub-`v_neutral` path.

A `#[test]` (I3 test #3, §8) encodes the invariant directly so a future change
that reintroduces the "average a zero down" form fails CI.

---

## 5. Cold-start & cross-axis interaction (the over-suppression guard, restated)

The single most important calibration fact, consolidated so the implementer
cannot miss it:

- **`value` is monotone in positive evidence and is a constant 1.0 in v1**
  (§4.3-4.5). It can only damp below its neutral point on a *wasted* signal,
  which v1 defers (`v_neutral = 1.0`). So on day one value is a pure
  pass-through (attribution-only scaffold), and the new scorer's behavior is
  driven entirely by **specificity** (live evidence today) and
  **negative_penalty** (live exemplars today). This is deliberate: it makes
  specificity the ship-first lever (THESIS), avoids three compounding `[0,1]`
  dampers collapsing a legitimate handle, and — critically — guarantees a
  decision-cited handle is never *more* suppressed than unproven noise (the
  inverted-cold-start bug). Value becomes a live ceiling-raiser (`v_neutral <
  1.0`, raising proven handles) only when `ThreadSurfaced` lands.
- **ε floor (`damper_floor = 0.02`) on each of `spec`, `val`, `damp`** so the
  worst case is demotion to the unresonant baseline, never to zero (gate, don't
  delete).
- **negative_penalty is bounded by `γ < 1`** (`neg_gamma = 0.8`) so even a
  perfect negative twin keeps ≥ 20% of its resonance-driven score — the
  ostk-cache recoverability guarantee (R3).

---

## 6. Negative-transfer specifics (R3) — consolidated

(Most detail is in §2 boot pass, §3.2 readers, §3.3 primitives. Restating the
non-negotiables R3 was emphatic about.)

1. **Mean-center FIRST.** `center(v) = normalize(normalize(v) − global_mean)`.
   Without it, separation is +0.0195 (useless); with it, 5× better. The
   `global_anchor_mean` is built in pass (A) and stored store-level.
2. **kNN k=3** in centered space (AUC 0.917). k=1 (0.957) is brittle — it would
   kill ostk-cache, which sits at centered cosine 1.0 to rejected sub-terms
   (`tiers`/`recalls`/`paging`/`page`/`handler`) that share its anchor chunk.
   k=3 averages the neighborhood.
3. **SOFT bounded multiplicative damp** `(1 − γ·penalty)`, never a hard gate
   (§1.2 term 2). `γ = 0.8`, `τ = 0.45` (R3 §3 says τ ∈ [0.45, 0.6] cleanly
   zeroes most good concepts while penalizing noise: noise mean kNN +0.77,
   good mean +0.44).
4. **Label source = the 279 dormant thread handles' `anchor_vec`**
   (432/490 present) + 47 rejected-concept evidence anchors, **NOT** the 10
   bare session-rejected handle strings (§3.2).
5. **ostk-cache is the named A/B hard case** — I2's test must prove it survives
   (§8 I2-test-1) and the A/B harness must confirm it stays surfaceable.
6. **Self-reinforcing loop** (R3 §integration): a damped score lowers
   `score_thread`, the `IdleCurator` demotes the handle toward dormant on its
   own (no manual tension push), and on the next boot that handle's own anchor
   joins the negative exemplar set. Whack-a-mole retires itself.

---

## 7. Config, flag & A/B harness

### 7.1 The `[salience]` config block (new, sibling to `[weaver]`)

A new `SalienceSettings` in `crates/core/src/config.rs`, mirroring
`WeaverSettings` exactly: `#[serde(deny_unknown_fields)]`, per-field
`#[serde(default = "...")]`, a `Default` impl, and a
`resolve(slot: Option<&Self>) -> Self` (config.rs:472-478). `WeaverSettings`
is left as-is (stop_handles stays there — it's still wired).

```toml
[salience]
# --- master + per-axis toggles (A/B; each axis individually toggleable) ---
scorer_v2          = false   # master: false ⇒ today's scorer, factors = identity
specificity_enabled = true   # axis 1 (the core; ship first)
value_enabled       = true   # axis 3
negative_enabled    = true   # axis 2

# --- specificity (R1) ---
specificity_min_evidence = 5     # neutral (1.0) below this many resonating chunks
specificity_lift_cutoff  = 0.2   # deny off-diagonal lift below this specificity

# --- value (R2) ---
# value = value_neutral + (1 − value_neutral)·positive,  positive ∈ [0,1].
# Monotone in positive evidence (§4.5): evidence only raises value toward 1.0.
value_neutral = 1.0             # v1: 1.0 ⇒ value is a pass-through (NO cold-start damp).
                                # 2nd increment (with ThreadSurfaced): set < 1.0 (e.g. 0.7)
                                # to turn value into a ceiling-raiser; the wasted-signal
                                # penalty is then the ONLY path below value_neutral.
value_w_use   = 0.5             # weight on value_use within `positive`
value_w_judg  = 0.5             # weight on value_judgment within `positive`

# --- negative-transfer (R3) ---
neg_gamma           = 0.8        # max damp = 1 − γ (so floor never below 20%)
negative_knn_k      = 3          # kNN k in centered space (k=1 brittle; do not set to 1 in prod)
negative_tau        = 0.45       # proximity floor below which neg_penalty = 0
negative_lift_cutoff = 0.5       # treat neg above this like is_stop for lift

# --- shared guards ---
damper_floor = 0.02              # ε clamp on each [0,1] damper (gate, don't delete)

# --- self-audit thresholds (R4) — NOT flag-gated; pure observation ---
[salience.health]
min_surface_entropy     = 0.6
max_active_decided_drift = 0.7
never_used_min_surfaced = 10
health_window_days      = 14
ttl_secs                = 30
```

The resolved `SalienceSettings` is threaded into `InMemoryAttention` (it needs
`scorer_v2` + the per-axis toggles + knobs at score time) — add it as a field
set at construction (it's known at build time, unlike the factors), e.g.
`with_salience_settings(cfg)` builder analogous to `with_stop_handles`. The
scorer reads a small `SalienceScorer` snapshot (the subset of knobs
`compute_score_parts` needs) — derive it from `SalienceSettings` once and store
on `InMemoryAttention` so the hot path reads a `Copy` struct, not the full
config.

### 7.2 Self-audit is NOT flag-gated

R4 §6, restated: the health compute path is pure observation and **must watch
BOTH the old and new scorer** — so it cannot be conditional on `scorer_v2`.
Only the scorer *changes* (axes 1-3) are flag-gated. The `[salience.health]`
thresholds live in config (live-tunable) but the compute always runs. Its four
metrics double as the A/B scoreboard (§7.3).

### 7.3 A/B harness spec

**Mechanism.** A test/bench harness (a `#[test]` in `crates/attention` or a
hidden CLI subcommand for the V phase) that:

1. Loads the live thread set (or a captured fixture of it — the boot pass
   inputs: `reanchor_inputs()` + the dormant/rejected anchors + a sample
   attention vector). The V phase runs it against the live `serve` substrate
   via the MCP `attention_surface` tool; the unit-level harness uses a fixture.
2. Computes `salience_factors` for every handle via the §2.2 pass.
3. Ranks the thread set **twice**: once with `scorer_v2 = false` (identity
   factors — today's scorer) and once with `scorer_v2 = true`. Optionally a
   third run with `scorer_v2 = true` but the curated `stop_handles` **disabled**
   (`forced_stop = false` for all) to test the redundancy claim.
4. Emits R4's four health metrics (entropy, curated:autonomous ratio,
   never-used tail, active-vs-decided drift) for each run — the scoreboard.

**Pass criteria** (THESIS proof bar):

- **P1 (ordering):** confirmed coherent-noise
  `{turn-digest, squad-lead, re-run, non-blocking, re-read, pre-existing,
  follow-up, top-level, system-reminder}` all rank **below** confirmed real
  concepts `{cognitive-memory, ostk-cache, dereference-or-void,
  relational-substrate-docgraph}` under the new scorer.
- **P2 (specificity alone):** ideally the `specificity_enabled = true,
  value/negative off` run alone reproduces P1's ordering effect — i.e.
  specificity reproduces the stop-set's effect with no hand-list.
- **P3 (ostk-cache survives):** ostk-cache stays on the surface (above
  `ARCHIVE_THRESHOLD`) and above all P1 noise handles in the negative-enabled
  run — the bounded-soft-penalty guarantee.
- **P4 (redundancy):** in the stop-set-disabled run, `curated_ratio` (R4
  Metric 2) falls without P1 ordering regressing — the stop-set is provably
  redundant.
- **Baseline to beat** (R4 §2, THESIS): today's surface is a flat ~0.25-wide
  score ribbon with `off_diagonal_lift = 0.0` on 39/40 pages — almost pure
  familiarity-floor. Health metrics should move: surface entropy up, curated
  ratio down, never-used tail shrinking.

---

## 8. Per-axis implementation contract

Sequencing: **I1 (specificity) first** — it's the core and the cheapest
unsupervised win. **I2 (negative) and I3 (value) depend on I1** (they layer
onto the same `SalienceFactors` map + `compute_score_parts` signature I1
introduces). **I4 (self-audit)** can land its surface-only metrics (1+2) early
as the A/B instrument, then ledger metrics (3+4) after I3. Each axis is
**individually toggleable** via its `*_enabled` flag.

### I1 — Specificity (the core; ships first)

- **New (in `crates/attention/src/salience.rs`):**
  - `pub fn shannon_entropy(&[f32]) -> f32` (shared with R4; §3.1).
  - `pub fn specificity_from_evidence(evidence, chunk_meta, cfg) -> f32`
    — builds the `source_id` histogram (R1 §1b: join chunk_id → source_id;
    bin by `(project, source_id)` per R1 §5-Q2), calls `shannon_entropy`,
    returns `1 − H/ln(N_eff)` with guards: `N_eff ≤ 1 → 1.0`;
    `total_chunks < specificity_min_evidence → 1.0` (R1 §3.1).
  - `pub struct SalienceFactors`, `pub struct SalienceScorer` (the hot-path
    snapshot), `Default` impls.
- **Modified:**
  - `compute_score_parts` (lib.rs:717): add `factors: SalienceFactors` +
    `cfg: &SalienceScorer` params; apply the `spec` damper on the floor and
    `spec < cutoff` to lift (§1.2). Update both call sites: `surface`
    (lib.rs:1476) and `score_thread` (lib.rs:1683) to look up
    `factors.get(handle)` from the store map and pass the snapshot.
  - `InMemoryAttention` (lib.rs:853): add `salience_factors`,
    `salience_scorer` fields + `set_salience_factors` setter +
    `with_salience_settings` builder; thread into `Default`/`with_embedder`.
  - `ScoreParts` (lib.rs:705) + `ScoreAttribution` (core attention.rs:112):
    add `specificity`/`value`/`neg_penalty` axes so `why` decomposes (R4
    `abi-as-sovereign-boundary` discipline).
  - `re_anchor_threads_from_corpus` (commands.rs:1967): add pass (C) for
    specificity (the value/neg parts arrive with I3/I2).
  - `ThreadsDb`: `list_evidence_all()` batched reader; reuse
    `fetch_chunks_by_ids` (corpus.rs:799) for source_id.
  - `SalienceSettings` in config.rs.
- **Unit tests** (model on `stop_handle_floor_clamped_by_context_diversity`
  lib.rs:2383 and `curated_stop_set_clamps_high_resonance_harness_handle`
  lib.rs:2431):
  1. `specificity_demotes_diffuse_handle_with_no_hand_list` — two threads,
     identical counters (300/290) + identical anchors (equal live resonance);
     seed `specificity = 0.05` vs `0.95` via the factors map; assert the
     concentrated one out-ranks the diffuse one with NO `with_stop_handles`.
  2. `specificity_one_reproduces_current_scorer` — `factors = default` (or
     `scorer_v2 = false`) ⇒ score bit-identical to flag-off (A/B no-regression
     guard).
  3. `specificity_from_evidence_*` pure-fn tests — single-bin → 1.0;
     uniform-N-bins → ≈0.0; peaked → high; below `min_evidence` → 1.0.
  4. `specificity_does_not_block_active_discussion` — diffuse handle with high
     *live* resonance still surfaces (floor clamped, `ALPHA·resonance` intact).

### I2 — Negative-transfer (depends on I1)

- **New (in `salience.rs`):** `center`, `negative_penalty` (§3.3).
- **New (`ThreadsDb`):** `dormant_anchor_vecs`, `rejected_concept_anchors`
  (§3.2).
- **Modified:**
  - `InMemoryAttention`: add `negative_exemplars`, `global_anchor_mean` +
    `set_negative_exemplars` setter.
  - `compute_score_parts`: apply the `damp = 1 − γ·neg` term (§1.2 term 2) and
    `neg > cutoff` to lift.
  - `re_anchor_threads_from_corpus`: pass (A) global mean + (B) exemplar set +
    (C) `neg_penalty` per handle (the convergent single pass).
- **Unit tests:**
  1. `negative_penalty_keeps_recoverable_concept` — the **ostk-cache** case:
     a fixture anchor at centered cosine 1.0 to one exemplar but far from the
     other k−1; assert with k=3 + γ=0.8 the damp leaves it surfaceable above a
     true-noise handle. (The named A/B hard case, R3.)
  2. `negative_penalty_damps_harness_twin` — a fixture anchor near several
     exemplars; assert its floor is damped below a clean handle's.
  3. `centering_is_required_for_separation` — assert raw-space mean-top-k AUC ≈
     random on a fixture set, centered kNN AUC high (R3 §empirical).
  4. `negative_penalty_neutral_when_no_exemplars` — empty set → 0.0 → no effect.

### I3 — Value (depends on I1)

- **New (`ThreadsDb`):** `surfaced_vs_used(handles, since)` (§3.4, the SHARED
  join — R4 reuses it); a `value_from(...)` assembler (may live in
  `salience.rs`, pulling the join result + judgment set + active coords).
- **Reuse verbatim:** `activation.rs` `act_r_base`, `squash`,
  `age_hours_floored`, `AccessWeights`, `concept_support_by_coord`,
  `edge_lift_for` saturation.
- **Modified:**
  - `compute_score_parts`: apply `val` as a floor multiplier (§1.2 term 1) — the
    param already exists from I1. **In v1 `val = 1.0` always** (the monotone
    pass-through, §4.5), so this edit is the wiring; the multiplier only moves
    the floor once `v_neutral < 1.0` ships with `ThreadSurfaced`.
  - `re_anchor_threads_from_corpus`: pass (C) — compute `positive(H)` (the
    `value_use`/`value_judgment` join + `concept_support_by_coord` intersect)
    and store `value = v_neutral + (1−v_neutral)·positive`. With v1
    `v_neutral = 1.0` this is the constant 1.0, but the join runs so the
    attribution axis + `surfaced_vs_used` plumbing are live and tested.
- **Value is monotone in positive evidence (§4.5 invariant); v1 `v_neutral = 1.0`
  (pass-through, NOT a cold-start damper).** Config `value_neutral = 1.0` in v1.
- **DEFER:** the `ThreadSurfaced` event, the sub-neutral wasted penalty, and the
  ceiling-raiser regime `v_neutral < 1.0` (§4.4).
- **Unit tests:**
  1. `value_monotone_in_positive_evidence` (THE invariant test, §4.5) — two
     threads with identical specificity/anchor/counters; A has an
     `ExplicitRecall` on its evidence chunk (or an `ostk_decision`-sourced
     Active evidence link), B has none. Assert `value(A) ≥ value(B)` AND
     `score(A) ≥ score(B)` **for every `v_neutral ∈ {1.0, 0.7}`** — proving
     evidence never *more*-damps. This is the regression guard against the
     "average a zero down" inversion.
  2. `value_v1_is_pass_through` — with `value_neutral = 1.0` (the shipped v1
     config), any handle (judged, used, or neither) gets `value == 1.0`, so the
     score equals the specificity/negative-only score (value contributes
     nothing but attribution; no double-count with specificity).
  3. `value_judgment_raises_proven_handle_when_unlocked` — with
     `value_neutral = 0.7` (the 2nd-increment regime), a thread whose Active
     evidence link resolves to an `ostk_decision`-sourced chunk gets
     `value > 0.7` (raised toward 1.0), while a plumbing-only thread stays at
     `0.7` — and the judged thread out-ranks the plumbing one. (Asserts the
     ceiling-raiser direction is *up*, not down.)
  4. `value_distinct_query_gate` — 50 accesses sharing one `query_hash` collapse
     to one in `positive(H)` (model on `distinct_query_gate_not_raw_count`,
     activation.rs:805). Tested at the `positive`/join level so it holds
     regardless of `v_neutral`.

### I4 — Self-audit (depends on I1-I3 for the full A/B; metrics 1+2 can land early)

- **New module `crates/attention/src/health.rs`:**
  `pub fn salience_health(surface, ledger, judgment, thresholds) -> SalienceHealth`
  (pure, reusable by `recall_stats`, the conditional push, and the A/B harness —
  twice). Imports `shannon_entropy` from `salience.rs` (§3.1).
- **New struct (`crates/core/src/types.rs`, next to `RecallStats`):**
  `SalienceHealth { surface_entropy, surface_score_spread, curated_ratio,
  curated_load_bearing, never_used, active_decided_drift, drift_forgotten,
  threshold }` (R4 §4a) — every metric decomposable (`why`), never a bare
  scalar.
- **Metrics** (R4 §3): (1) surface entropy `H_norm` via `shannon_entropy` over
  score-share + cheap score-spread companion; (2) curated:autonomous ratio +
  the load-bearing curated handles; (3) surfaced-N-never-used via the SHARED
  `surfaced_vs_used` join (§3.4), distinct-query-gated, `unattributable`
  flagged separately; (4) active-vs-decided Jaccard drift + the `J \ A`
  forgotten tail (the top alarm).
- **Surfacing** (R4 §4-5, model on `stale_ingest` server.rs:719-841):
  - Pull leg: `recall_stats.salience_health: Option<SalienceHealth>`
    (`#[serde(skip_serializing_if = "Option::is_none")]`), populated in
    `recall_stats()` (query/src/stats.rs), degrade to `None` when handles
    absent.
  - Push leg: a `compute_salience_health() -> Option<Value>` wrapper,
    threshold-gated (loud-on-failure), 30s TTL cache, injected into responses
    only when unhealthy.
- **NOT flag-gated** (§7.2) — watches both scorers.
- **Unit tests:**
  1. `surface_entropy_collapses_on_dominant_handle` — a surface with one
     dominant score → low `H_norm`; an even spread → high.
  2. `never_used_flags_surfaced_not_recalled` — fixture: handle with
     `LensIncluded` ≥ N but no `ExplicitRecall` → flagged; one with recalls →
     not.
  3. `drift_reports_forgotten_judged_handle` — a handle in the judged set `J`
     but not the active surface `A` appears in `drift_forgotten`.
  4. `health_runs_under_both_scorers` — `salience_health` produces a block with
     `scorer_v2` both on and off (it's observation, not gated).

---

## 9. Open questions for Plan (P)

1. **Interior mutability primitive for the factors map.** Design picks "store
   on `InMemoryAttention`, swap an `Arc` from the consolidate pass" (§2.1).
   P should confirm `arc_swap::ArcSwap` is acceptable in the workspace (it's a
   common, lock-free fit) vs `Mutex<Arc<HashMap>>` read-cloned in the caller.
   Either works; ArcSwap is cleaner for the hot read path. **Not blocking** —
   both are mechanical.
2. **Specificity persistence (R1 §5-Q3).** Recompute-on-boot (chosen for v1,
   always fresh) vs a `threads.specificity REAL` column (saves boot recompute,
   lets `attention_surface why` expose it for I4). Design leans recompute-on-
   boot for v1 since the boot pass already walks every thread; revisit a column
   once the formula stabilizes. **Not blocking I1.**
3. **Consolidate-cadence refresh.** The boot pass fills the map; the
   `AutoWeaver::consolidate` pass (weaver.rs:299) should re-run pass (C) so
   factors track drift (R1 §3.4 option 2, R2 §refresh-cadence). P should
   confirm the consolidate pass has (or can borrow) the corpus+threads handles
   to call the shared joins, and that the `Arc` swap is the refresh mechanism.
   **Not blocking v1** (boot-only refresh is correct, just staler).
4. **A/B harness home.** A `#[test]`/bench in `crates/attention` (fixture) plus
   a V-phase live run via MCP `attention_surface` (§7.3). P should decide
   whether the live run warrants a hidden CLI subcommand or is driven from the
   review session directly. **Not blocking implementation.**

None of these block the sequenced implementation; they are P-phase
refinements. There is no genuine research gap — all four axes have a concrete,
code-anchored mechanism and the composition is resolved.
