# Autonomous Salience — Team Spec (canonical brief)

> Team `salience`. This doc is the north star. Every phase reads it. It is
> self-contained so the work survives a context compaction of the lead.

## Mission
Improve **autonomous salience** in the ostk-recall cognition substrate — the
mechanisms that decide, without being asked, what threads/concepts are "active"
and what the surfacer/lens injects into context.

## The thesis (the one sentence)
Today salience ≈ **resemblance-to-now + frequency**, and both are backwards.
`compute_score_parts` scores a thread as
`decay·floor(resonance_count) + α·cosine(anchor, attention_now) + β·off_diagonal_lift`.
Every term rewards "looks like what's happening, a lot." Harness/plumbing vocab
looks like what's happening *constantly*, so it wins — that's the "coherent
noise" problem (high frequency AND high resonance) that the current
`[weaver] stop_handles` hand-list only *patches*.

**Reframe we are building toward:**

    salience = specificity × value × recency      (NOT frequency × recency)

The system already has recency (decay) and a frequency proxy (anti-correlated
with what we want). It is missing **specificity** (does this term *discriminate*?)
and **value** (did surfacing it ever *matter*?). Those two axes are the work.
Mantra: stop asking "how often does this recur and resemble now?" — ask
"**how much does this distinguish, and did it ever pay off?**"

## Key mechanism facts (so nobody re-discovers them)
- `crates/attention/src/lib.rs`:
  - `compute_score_parts(state, attention_vec, now, forced_stop)` (~L717): the
    scorer. `resonance = cosine(anchor, attention_vec)`; `resonance_floor =
    is_stop ? familiarity_floor(0) : familiarity_floor(resonance_count)`;
    `score = decay_term + ALPHA·resonance + BETA·off_diagonal_lift(...)`.
  - `is_stop_handle(mentions, resonance)` (~L110): frequency-only cliff
    (`resonance/mentions < STOPWORD_RATE_MAX = 0.05`, gated by
    `STOPWORD_MENTION_MIN = 50`). Comment literally says "no hand-list". This is
    a crude 1-bin entropy estimate with a binary cutoff.
  - `InMemoryAttention` carries `stop_handles` (the curated set, wired
    `with_stop_handles`); `score_thread`/`surface` pass `forced_stop`.
- `crates/attention/src/curator.rs`: `IdleCurator.resolve_target(tension, score)`
  recomputes tension from score thresholds + hysteresis every tick. A manual
  demote bounces back unless the *score* changes — so all real fixes must move
  the score, not the tension.
- `crates/attention/src/weaver.rs`: `AutoWeaver` writes thread↔chunk evidence
  edges and proposal seeding; `is_stop_handle` gate at the promotion site. The
  **co-occurrence data for specificity lives here / in the threads evidence
  graph** (thread_evidence edges, evidence_links).
- `crates/attention/src/observer.rs`: `TurnObserver` node-minter (term_recurrence
  → concept candidates), tuned by `[ambient_growth]`. Carries `stop_handles`.
- `crates/core/src/config.rs`: `WeaverSettings` ([weaver]) holds `exclude_facets`
  + `stop_handles` + `default_weaver_stop_handles()`. New scorer knobs land here
  (or a new `[salience]`/`[attention]` block) — **behind a flag for A/B**.
- Live signals available via MCP (probe the running serve): `thread_list`,
  `thread_evidence`, `attention_surface` (carries per-row `why`: resonance,
  off_diagonal_lift, mentions, resonance_count), `recall` (+access logging),
  `memory_recall`/`memory_concept`, `recall_stats`. Judgment artifacts:
  `ostk_decision` (562), `ostk_needle` (212), active concepts (the immaculate
  curated tier). Thread anchors are stored embedding vectors.

## The four axes

### 1. Specificity — root-cause fix, unsupervised, SHIP FIRST
A term that co-occurs with *everything* carries no information; one that
co-occurs *selectively* does. This is IDF/entropy — the corpus side already uses
it (BM25); the attention side does not. For each handle compute the Shannon
entropy `H` of its co-occurrence distribution (which contexts/anchor-neighbors/
source-docs it resonates with — the weaver already tracks this). Define
`specificity = 1 − H/H_max` and make it a **continuous multiplier** on the
resonance term/floor in `compute_score_parts`. This is the principled version of
`is_stop_handle`'s binary cliff. **Success = it auto-demotes `turn-digest`/
`squad-lead`/`non-blocking` with NO hand-list, while preserving real concepts
(`cognitive-memory`, `ostk-cache`, `dereference-or-void`) — and the curated
stop-set becomes redundant.**

### 2. Value — the ceiling-raiser
Ground salience in what was actually used.
- **Use feedback**: `memory_recall` logs access; did a surfaced thread then get
  cited in a decision / lead to an edit / get pinned? Promote what gets used
  after surfacing; decay surface-but-never-landed. (Click-through loop.)
- **Judgment propagation**: handles appearing in `ostk_decision`/`ostk_needle`/
  active concepts are *recorded operator salience judgments*. Propagate
  confidence FROM those curated artifacts TO co-occurring handles.
Success = salience tracks meaning, not statistics; surfaced items correlate with
downstream use.

### 3. Negative transfer — cheap, uses labels we just generated
We rejected 10 concept proposals and demoted 34 thread handles this session.
They cluster in embedding space. Damp a handle's salience by proximity to the
rejected/stopped centroid, so new harness-ish terms inherit suspicion — retiring
the whack-a-mole hand-list. Success = a *novel* harness term is pre-damped
without being hand-listed.

### 4. Self-audit — pay down the doctrine debt
The substrate cannot see its own drift (the optimize bug + dangling anchors were
found by accident, not surfaced). Add salience-health metrics: entropy of the
active surface, curated:autonomous ratio on the surface, "surfaced-N-times-
never-used" flags. This is `dereference-or-void` + `projection-truth` applied to
the surfacer's own state. Success = drift surfaces itself; no manual gardening
session required to notice it.

## Integration constraint (CRITICAL)
Axes #1, #2, #3 ALL modify the scoring path (`compute_score_parts` / the
curator's score). They must be designed as **one coherent scorer evolution**
(`salience = specificity × value × recency − negative_penalty`), NOT three
conflicting patches. Implement **sequenced** (specificity first; it's the core),
each composable and individually toggleable. Everything ships **behind a config
flag** so the new scorer can be **A/B'd against the current one on the live
thread set** before becoming default.

## Proof / verification bar ("tests for proof, verify, make it so")
- Unit tests per axis in `crates/attention` (model on
  `stop_handle_floor_clamped_by_context_diversity` and
  `curated_stop_set_clamps_high_resonance_harness_handle`).
- An **A/B harness**: run old vs new scorer over the live thread set; prove the
  new scorer ranks confirmed coherent-noise (`turn-digest`, `squad-lead`,
  `re-run`, `non-blocking`) BELOW confirmed real concepts (`cognitive-memory`,
  `ostk-cache`, `dereference-or-void`, `relational-substrate-docgraph`) — and
  ideally that #1 alone reproduces the stop-set's effect.
- `cargo test -p ostk-recall-attention -p ostk-recall-core` green; no regressions.
- Adversarial review across all 4 before "make it so" (land + default-on).

## Conventions
- Prefer `mcp__ostk__*` tools (fs_ops/read/bash/search) for kernel-tracked files;
  native fallback if a tool is missing. Cargo/git via Bash need
  `dangerouslyDisableSandbox: true` (reaches ~/.cargo).
- Implementation work on a feature branch (e.g. `feat/autonomous-salience`),
  never commit to `main` without the lead's review gate.
- Probe the LIVE substrate via the ostk-recall MCP tools for real data.

## Pipeline & artifacts (file-mediated, durable)
- Research → `research-specificity.md`, `research-value.md`,
  `research-negative-transfer.md`, `research-self-audit.md` (this dir).
- Design  → `design.md` (unified scorer).
- Plan    → `plan.md` (sequenced steps + test strategy).
- Implement → code + tests on the branch.
- Review  → `review.md` (verdict + A/B results).

## Task DAG (team `salience`)
R1–R4 research (parallel, no deps) → D design (deps R1–R4) → P plan (deps D) →
I1 specificity → I2 negative-transfer → I3 value → I4 self-audit (I2/I3 dep I1;
all dep P) → V review+verify (deps I1–I4). Lead sequences and holds the review
gate before default-on.

## Cross-axis coordination (from R4 research, 2026-06-15) — DESIGN MUST HONOR
Self-audit research surfaced shared machinery; factor each ONCE in design (D):
- `shannon_entropy(&[f32])` is needed by BOTH R1 (per-handle co-occurrence
  entropy) and R4 (active-surface entropy). One shared impl.
- The **surfaced-vs-used ledger join** (ChainLogReader::access_history —
  LensIncluded "surfaced" vs ExplicitRecall/OperatorSelected/ConceptAccessed
  "used", honoring the distinct query_hash gate) is the SAME machinery R2
  (value) *scores* and R4 *reports*. Write the join helper once: R2 scores it,
  R4 measures it.
- Self-audit's compute path is NOT flag-gated (pure observation must watch BOTH
  old and new scorer); its 4 metrics double as the A/B scoreboard for R1–R3.
- Ledger access events are chunk_id-keyed: thread-handle "never-used" needs a
  join through the evidence graph (bound it like thread_query's batched
  backfill); concept-handle never-used is cheap (ConceptAccessed is
  (project,handle)-keyed).
- A/B baseline captured live now (the "before" to beat): the active surface is a
  flat ~0.25-wide score ribbon with off_diagonal_lift=0.0 on 39/40 pages —
  almost pure familiarity-floor — with confirmed coherent-noise (re-read,
  pre-existing, follow-up, system-reminder) present on it.

## Cross-axis coordination (from R2 value research, 2026-06-15) — DESIGN MUST HONOR
- **Both halves of value are new WIRES into existing data, not new collection.**
  `compute_score_parts` reads only in-memory `ThreadState` today — zero ledger
  access. The chain_log access ledger (explicit_recall/operator_selected/
  lens_included) is already logged but wired only for the CONCEPT tier
  (`activation.rs` decayed_access) and chunk-freshness, never the thread scorer.
- value = **value_use × value_judgment**, precomputed offline into a new
  `InMemoryAttention.value_scores: Arc<HashMap<String,f32>>` (the EXACT
  `stop_handles` pattern, lib.rs:871), folded as a multiplier on resonance/floor,
  flag-gated. **Design decision to resolve:** R1 caches specificity on a
  `ThreadState.specificity` field; R2 caches value in an `Arc<HashMap>`. Unify
  the per-handle-scalar precompute mechanism (one of the two patterns for both).
- `value_use` = join chain_log access_history (threads.rs:1144) → threads via
  `evidence_links.last_resolved_chunk_id`, distinct-query gate, **REUSE
  activation.rs curves (act_r_base/squash/AccessWeights) verbatim.** This is the
  SAME ledger join R4 reports — confirms the shared helper (R2 scores, R4 measures).
- `value_judgment` = ostk_decision(562)/ostk_needle(212) are corpus chunks with
  NO handle FK; the only link is the embedding bridge the weaver ALREADY
  computes. Read existing Derived evidence links filtered by source + intersect
  `concept_support_by_coord` (active concepts = confidence 1.0). **No new join
  table.**
- **Two reconstruction fidelities:** weak ("thread's content was used") works
  today with no schema change — BUILD FIRST. Strict click-through ("surfaced
  THEN used") needs one new `ThreadSurfaced` event at lib.rs:1480 (threads have
  no surface event; LensIncluded logs chunks only) — DEFER to 2nd increment.
- **Cold-start (design must address):** value_use≈0 across the board today
  (loop not yet wired), so value_judgment carries the early lift. Decide
  neutral-1.0 (no damp until evidence) vs damp-by-default. Note `operator_selected`
  has no producer yet — don't lean on it for v1.

## Cross-axis coordination (from R3 negative-transfer research, 2026-06-15) — DESIGN MUST HONOR
- **A naive single rejected-centroid cosine is USELESS** — measured live,
  separation +0.0195; good concepts and confirmed noise both sit ~0.62 from the
  centroid (embedding anisotropy, global-mean norm 0.62, is the killer).
- The mechanism that works (validated, AUC=P(noise ranked above good)):
  1. **MUST mean-center the space first:** `center(v) = normalize(normalize(v) −
     global_mean)`. 5× improvement alone.
  2. Then **kNN to nearest negative exemplars, k=3**, in centered space → AUC
     0.917. A single centroid fails; k=1 scores higher (0.957) but is BRITTLE.
  3. Apply as a **SOFT bounded multiplicative damp `(1 − γ·penalty)`, NOT a hard
     gate.** Composes multiplicatively with axis-1 specificity.
- **Label sources mapped live** (threads.sqlite, dim 512): STRONG = 279 dormant
  thread handles, 432/490 carry `anchor_vec` BLOB (the demoted set: per-trigger,
  anthropic-*, compile-check…) — USE THESE. WEAKER = 47 rejected concepts with
  evidence anchor_vec. The 10 session-rejected handles (follow-up, pre-existing,
  re-touch…) have ZERO evidence/anchor — **do NOT embed the bare handle string**;
  dormant-thread anchors already cover that space.
- **ostk-cache is the A/B hard case:** a REAL concept that sits at cosine 1.0 to
  rejected sub-terms (tiers/recalls/paging/page/handler) because they share the
  identical anchor chunk. k=1 would kill it; **k=3 + bounded soft penalty keeps
  it recoverable.** Design/I2 must prove ostk-cache survives.
- Slot: `compute_score_parts`. Exemplar-set + global-mean built at boot in
  `re_anchor_threads_from_corpus` (crates/cli/src/commands.rs:1967), wired like
  `with_stop_handles`. (NOTE: R1 says specificity precompute, R2 says value
  precompute — three boot-time precomputes now converge here; design the boot
  wiring as ONE pass that fills specificity + value + neg-exemplars together.)
