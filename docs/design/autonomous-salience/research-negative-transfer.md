# R3 — Negative-Transfer Signal (research)

> Team `salience`, axis #3 of THESIS.md. Damp a handle's salience by its
> embedding-proximity to a centroid/exemplar-set of *rejected* concepts and
> *demoted/dormant* thread handles, so new harness-ish vocab inherits suspicion
> without being hand-listed. **All claims below are measured against the LIVE
> substrate** (`/Users/scottmeyer/.local/share/ostk-recall/threads.sqlite`,
> embedder `minishlab/potion-retrieval-32M`, dim 512).

## TL;DR / verdict

Feasible, cheap, and useful — **but only with two non-obvious corrections, and
only as a soft bounded penalty, not a hard gate.**

1. The raw "cosine to a single rejected/stopped centroid" is **useless**: good
   concepts and confirmed noise both sit at ~0.62 cosine to the negative
   centroid (separation +0.0195). This is embedding **anisotropy** — the
   `potion` space has a strong common component (global-mean norm = 0.62).
2. **Mean-centering** (subtract the global anchor mean, then normalize) is
   *mandatory* and raises gross separation ~5x (+0.0195 → +0.1061), but a single
   centroid still overlaps heavily (a known-good concept `ostk-cache` scores
   *higher* than every noise handle).
3. **k-NN to the nearest negative exemplars (k=3) in centered space** is the
   discriminator that works: **AUC 0.917** (P(noise ranked above good)). It
   beats the single centroid and beats a neg-vs-pos margin (AUC 0.709).
4. Recommended mechanism: a **centered-space kNN negative penalty**, computed at
   score time against an in-memory exemplar set rebuilt at boot, applied as a
   bounded multiplicative damp `(1 − γ·penalty)` on the resonance/floor terms.
   Cost is negligible: 72 exemplars × 512 dims = **144 KiB**, ~72 dot products
   per scored thread.

## Where the labels and anchors live (the map)

### Negative label sources (two, with different quality)

**(A) Demoted / dormant thread handles — the strong source.**
- `crates/store/src/threads.rs`: the `threads` table. Columns `tension`
  (`active|slack|dormant`), `anchor_chunk_id TEXT`, and **`anchor_vec BLOB`**
  (the cached 512-d chunk embedding, added by migration L1828; codec
  `f32_vec_to_bytes` / `bytes_to_f32_vec`). `set_anchor_vec` (L1922) writes it;
  `reanchor_inputs()` (L1935) reads `(handle, privacy_tier, anchor_chunk_id,
  anchor_vec)` for every thread with an anchor source.
- Live: **490 threads, 432 carry `anchor_vec`** (88%). `thread_list` (default)
  returns all **279 dormant** rows — these ARE the demoted/stop set
  (`per-trigger`, `post-hoc`, `compile-check`, `anthropic-*`, `merge-gate`,
  `def-buffer`, `audit-ingest`, …). This is the cleanest negative source: every
  one is a real demotion decision with a real corpus-grounded anchor.

**(B) Rejected concepts — usable but polluted.**
- `crates/store/src/concepts.rs`: `concepts` table (`status` ∈
  `candidate|proposed|active|rejected|merged`), and `concept_evidence` with its
  own `anchor_vec BLOB` per evidence row (`evidence_to_reconcile()` L1229 decodes
  it; `EvidenceReconcileRow.anchor_vec`). `ConceptStatus::Rejected` is explicitly
  documented as "anti-pattern memory" (L103) — terminal, never re-touched.
- Live: **57 rejected concepts; 47 of them carry embedded evidence anchors** (50
  evidence rows, all 50 with `anchor_vec`). These are the *older* rejections
  (`cache`, `boot`, `daemon`, `gate`, `page`, `handler`, `tiers`, …).
- **Caveat — the 10 session-rejected handles have NO anchor.** `follow-up`,
  `pre-existing`, `re-touch`, `re-touching`, `re-renders`, `per-chunk`,
  `cross-scan`, `pin-clear`, `same-project`, `load-bearing` are bare
  observer-minted term candidates with **0 evidence rows** (probed
  `memory_concept show follow-up` → `evidence: []`, only `co_occurs` edges). They
  contribute nothing to an anchor-based centroid unless we embed the handle
  string. **Recommendation: do not embed the bare handle string** — it adds the
  least-grounded labels and `potion` embeds 2-token handles noisily. The 432
  dormant-thread anchors already cover the harness-vocab space; the bare 10 are
  better consumed by axis #1 (specificity) and the hand-list's eventual retirement.

### In-memory anchors (where the scorer reads)

- `crates/attention/src/lib.rs`: `ThreadState.anchor: Vec<f32>` (L550). Seeded at
  boot by `re_anchor_threads_from_corpus` (`crates/cli/src/commands.rs:1967`),
  which pulls `reanchor_inputs()`, prefers the cached `anchor_vec`, and calls
  `seed_anchor(scope, handle, vec)` into the `project=None / session="replay" /
  agent="substrate"` scope. `resolve_anchor` (L813) resolves a handle's true
  anchor cross-scope.
- The scorer is `compute_score_parts` (L717): `resonance = cosine(state.anchor,
  attention_vec)`; floor from `familiarity_floor(resonance)`; `is_stop` forces
  the floor to baseline. The curated stop-set lives on `InMemoryAttention`
  (`stop_handles: Arc<HashSet<String>>`, wired by `with_stop_handles`, L915).

**This is the slot.** The negative penalty needs exactly one input the scorer
already has in hand: `state.anchor`. Everything else (the exemplar set, the
global mean) is built once at boot and held on `InMemoryAttention`.

## The empirical experiment (why the design is what it is)

Built three candidate signals over the live anchors and scored two labeled sets:
known-good real concepts (`cognitive-memory`, `ostk-cache`,
`dereference-or-void`, `relational-substrate-docgraph`, `three-time-scales`,
`fade-is-concentration`, `abi-as-sovereign-boundary`, `projection-truth`,
`north-star`, `cross-scope`) vs. 25 confirmed coherent-noise dormant handles.
Metric: AUC = P(a noise handle is ranked more-harness-ish than a good concept);
1.0 = perfect, 0.5 = random.

| Signal | Space | AUC | Verdict |
|---|---|---|---|
| cosine to single neg centroid | raw | ~0.5 (sep +0.0195) | useless — anisotropy |
| cosine to single neg centroid | **centered** | overlaps (sep +0.106) | better, still fails: good `ostk-cache` > all noise |
| margin: cos(neg)−cos(pos) | centered | 0.709 | weak; pos centroid noisy (only 5 active concepts have evidence anchors) |
| **kNN to nearest neg exemplars, k=3** | **centered** | **0.917** | **ship this** |
| kNN k=1 | centered | 0.957 | higher but brittle (see collisions) |
| kNN k=5 / k=10 / k=20 | centered | 0.909 / 0.811 / 0.760 | degrades as k grows past the local cluster |

**Anisotropy is the headline.** Global-mean anchor norm is 0.62 — a huge common
component. Without centering, every handle is ~0.62 from everything and no
centroid discriminates. `cosine(centered(anchor), …)` where
`centered(v) = normalize(normalize(v) − global_mean)` is the fix and is the same
trick BM25/IDF encodes on the corpus side (THESIS axis #1 calls this out).

**Why kNN beats a centroid.** The negative set is multi-modal: ostk-internals
vocab (`cache`/`boot`/`daemon`), anthropic/harness vocab (`anthropic-*`,
`about-claude`), and generic process vocab (`merge-gate`, `compile-check`). A
single mean blurs these into a point near nobody. Nearest-exemplar asks the right
question — "is this handle a near-twin of something we already rejected?" —
which is exactly the negative-transfer intuition.

**Why k=3, not k=1.** k=1 scores AUC 0.957 but the diagnostic exposed why it's
dangerous: `ostk-cache` (a *real* concept) sits at **cosine 1.0000** to rejected
concepts `tiers`, `recalls`, `paging`, `pages`, `page`, `handler`. A 1.0 in
centered space means they share the *identical anchor chunk* — those rejections
are observer-minted sub-fragments of the very paging/cache docs `ostk-cache` is
about. They are not "harness noise"; they are rejected *sub-terms of a good
concept*. With k=1, that single poisoned twin would damp `ostk-cache` to the
floor. k=3 averages over the local neighborhood and is far more robust to a
single colliding exemplar. **This is also the case for the penalty being soft
and bounded, never a hard cliff** — a real concept can legitimately reuse
rejected sub-vocabulary, and the system must be able to recover it via the other
axes (specificity, value).

## Recommended mechanism

### 1. Label source
- **Primary:** `anchor_vec` of every **dormant** thread (`tension='dormant'`,
  `anchor_vec IS NOT NULL`) — 279 handles, ~all anchored.
- **Secondary:** `anchor_vec` of evidence rows for **rejected** concepts
  (`status='rejected'`), de-duplicated per concept by mean. 47 handles.
- **Excluded:** bare rejected concepts with no evidence anchor (the 10 session
  ones). Don't embed the handle string.
- All exemplars are **mean-centered then normalized** before storage.
- (Optional hardening, follow-up) drop any negative exemplar whose centered
  cosine to an *active*-concept anchor exceeds ~0.97 — that's the `ostk-cache`
  collision class (rejected sub-terms of real concepts). Cheap one-time scrub at
  build; removes the worst false-positive source.

### 2. Centroid → exemplar set + global mean (built once at boot)
Not a single centroid. Build, on `InMemoryAttention` (next to `stop_handles`):
- `global_anchor_mean: Vec<f32>` — mean of all normalized thread anchors.
- `negative_exemplars: Arc<Vec<Vec<f32>>>` — the centered+normalized negatives.

Both rebuilt in `re_anchor_threads_from_corpus` / `serve` boot, from
`reanchor_inputs()` + a new `ThreadsDb::rejected_concept_anchors()` reader
(mirrors `evidence_to_reconcile` filtered to `status='rejected'`). Wire via a
builder (`with_negative_exemplars(mean, exemplars)`) exactly like
`with_stop_handles`. Behind the `[salience]` A/B flag.

### 3. Proximity at score time
```
center(v)        = normalize(normalize(v) − global_anchor_mean)
neg_proximity(a) = mean of top-k cosine(center(a), e_i) over negative_exemplars,
                   k = 3   // NEGATIVE_KNN_K
penalty(a)       = clamp((neg_proximity(a) − τ) / (1 − τ), 0, 1)   // τ ≈ 0.45
```
`τ` is a floor below which proximity is ignored (centered noise sits ~0..0.4;
real collisions clear 0.45). Tune τ on the A/B set; the AUC=0.917 separation has
the noise mean at +0.77 kNN and good mean +0.44, so τ in [0.45, 0.6] cleanly
zeroes most good concepts while keeping noise penalized.

### 4. Where it slots in `compute_score_parts`
It is the `− negative_penalty` term of the THESIS unified scorer
`salience = specificity × value × recency − negative_penalty`. Concretely, apply
it as a **bounded multiplicative damp on the resonance-driven terms** so it
composes with axis #1's specificity multiplier rather than fighting it:
```
let neg = negative_penalty(&state.anchor);          // 0..1, default 0 when flag off / no exemplars
let damp = 1.0 - GAMMA * neg;                        // GAMMA ≈ 0.8 → max 80% damp, never to 0
let resonance_floor = (is_stop ? floor(0) : floor(state.resonance)) * damp;
let resonance_term  = ALPHA * resonance * damp;
// off-diagonal lift already gated by is_stop; also damp it or OR neg>threshold into is_stop
```
Multiplicative + bounded (`GAMMA<1`) means a harness term is suppressed but a
real concept that briefly collides can still climb on a strong fresh resonance —
recoverable, per the k=3 / soft-penalty argument above. Reuse `cosine_similarity`
(L238) for the dot products; add `center()` and `negative_penalty()` as free
functions next to `off_diagonal_lift` so they're unit-testable in isolation.

### 5. Cost
- Memory: 72 centered exemplars × 512 × 4 B = **144 KiB** (grows linearly; even
  all 279 dormant + 47 rejected = ~650 KiB). Plus one 2 KiB global mean.
- Per score: one centering (1 sub + 1 normalize over 512) + `|exemplars|` dot
  products of dim 512 (~72 today). `score_thread` runs per thread per curator
  tick and `surface` per thread per call — both already O(scopes×threads); this
  adds a constant ~72× factor on a 512-vector dot, microseconds. Negligible. If
  it ever matters, precompute `center(anchor)` once per thread at seed time
  (anchors are static between boots) so score time is pure kNN.

## Integration notes for D (design) / I2 (impl)
- The penalty must be **toggleable and default-off** until the A/B harness
  (THESIS verification bar) shows it ranks `turn-digest`/`squad-lead`/`re-run`/
  `non-blocking` below `cognitive-memory`/`ostk-cache`/`dereference-or-void`.
  Note `ostk-cache` is the known hard case — the A/B must confirm the soft
  bounded form keeps it surfaceable.
- It is **composable with axis #1**: specificity is an unsupervised entropy
  multiplier; negative-transfer is a supervised proximity damp. They multiply
  cleanly (`spec × (1 − γ·neg)`). Sequence I1 (specificity) first per THESIS;
  I2 layers on top.
- Curator interaction: a damped score lowers `score_thread`, which the
  `IdleCurator` (recomputes tension from score every tick) will then demote
  toward dormant on its own — no manual tension push needed. This is the
  self-reinforcing loop the THESIS wants: a novel harness term gets pre-damped,
  demoted, and (eventually) its own anchor joins the negative exemplar set on the
  next boot. Whack-a-mole retires itself.
- Unit tests to model on (per THESIS): `negative_penalty_damps_harness_twin`,
  `negative_penalty_bounded_preserves_recoverable_concept`,
  `centering_is_required_for_separation` (assert raw-space AUC ≈ random,
  centered kNN AUC high on a fixture exemplar set).

## Files referenced
- `crates/attention/src/lib.rs` — `compute_score_parts` L717, `ThreadState.anchor`
  L550, `cosine_similarity` L238, `off_diagonal_lift` L284, `InMemoryAttention`
  + `with_stop_handles` L853/L915, `resolve_anchor` L813, `seed_anchor` L1401.
- `crates/store/src/threads.rs` — `anchor_vec` schema/migration L1583/L1828,
  `set_anchor_vec` L1922, `reanchor_inputs` L1935, codec `f32_vec_to_bytes` /
  `bytes_to_f32_vec`.
- `crates/store/src/concepts.rs` — `ConceptStatus::Rejected` L87/L103,
  `concept_evidence.anchor_vec`, `evidence_to_reconcile` L1229,
  `EvidenceReconcileRow.anchor_vec` L1313.
- `crates/cli/src/commands.rs` — `re_anchor_threads_from_corpus` L1967 (boot
  seed slot for the exemplar-set + global-mean build), `with_stop_handles`
  wiring L355/L677/L709.
