# Research R2 — The VALUE / use-feedback signal

> Team `salience`, axis #2 of THESIS.md. Read-only research. The one
> sentence: today the thread scorer rewards `frequency × recency`; this axis
> grounds salience in **what was actually used** — surfaced-then-cited / pinned
> / edited (use-feedback), and confidence propagated **from** the curated
> judgment artifacts (`ostk_decision`, `ostk_needle`, active concepts) **to**
> the handles that co-occur with them.

## TL;DR (the headline)

The substrate **already logs every ingredient of a use-feedback loop** — but
the loop is wired for the **concept** tier and the **chunk-freshness** tier,
**not for the thread scorer** the salience work targets. `compute_score_parts`
(the function the THESIS calls "backwards") reads *only* the in-memory
`ThreadState` (`mentions`, `resonance`, `anchor`, `last_touched_at`); it has
**zero** access to the access ledger, concept activation, or the curated
judgment artifacts. So both halves of axis #2 are *new wires into an existing
data substrate*, not new data collection.

Two concrete mechanisms, both precomputed-into-`ThreadState` (the exact
`stop_handles` pattern), both behind a flag:

1. **Use-feedback (`value_use`)**: join the P7b access ledger (`chain_log`
   `explicit_recall` / `operator_selected` / `lens_included`) to threads via
   `evidence_links.last_resolved_chunk_id`, with a **distinct-query gate** (the
   `salience-vs-familiarity` discipline already proven in `activation.rs`).
   A thread whose evidence chunks get explicitly recalled / operator-selected
   after being surfaced earns value; one surfaced-but-never-landed decays.
2. **Judgment propagation (`value_judgment`)**: the curated artifacts are
   corpus chunks (`source ∈ {ostk_decision, ostk_needle}`) and `active`
   concepts (`confidence = 1.0`). Propagate their confidence to a thread by the
   *same embedding bridge the weaver already computes* — a thread whose anchor
   resonates with a decision/needle/active-concept chunk inherits a value
   prior. This is `concept_support_by_coord` (activation.rs) generalized from
   "lens candidate chunks" to "thread handles."

Success per THESIS: salience tracks *meaning, not statistics*; surfaced items
correlate with downstream use; `cognitive-memory` / `ostk-cache` /
`dereference-or-void` (all cited by decisions and heavily recalled) rise, while
`pre-existing` / `follow-up` / `re-read` / `top-level` (high resonance, zero
use, zero judgment-citation) fall.

---

## (a) What the access logging captures — and is "surfaced → then used" reconstructable?

### The access ledger exists and is rich: `chain_log` in `threads.sqlite`

The cognition stream is a single append-only table `chain_log(ts, kind,
payload)` in `<root>/threads.sqlite`, written by `SqliteChainSink`
(`crates/store/src/threads.rs:1213`, `append` / `append_within`). It is **not**
the DuckDB `audit_events` store (that is harness tool-call telemetry — 421k
`claude_code` rows — reachable via `recall_audit`; the two are unrelated for
this purpose). The five **access** event kinds
(`crates/store/src/threads.rs:339-359`, `AccessKind` at `:1063`):

| Kind | Emitted where | Meaning | What it keys on |
|---|---|---|---|
| `ExplicitRecall` | `crates/mcp/src/server.rs:445` (`recall`), `:571` (`memory_recall`) — one per returned hit | A chunk was surfaced by an explicit recall | `chunk_id`, `query_hash` |
| `RecallFault` | `crates/mcp/src/server.rs:504` (`recall_fault`) | Chunk pulled by synthesis paging | `chunk_id` |
| `LensIncluded` | (P9b-min — ambient lens render) | Chunk injected into the rendered lens | `chunk_id`, `slot` |
| `OperatorSelected` | **variant + ledger support ship; NO producer yet** (`threads.rs:356-359`) | Operator explicitly picked a chunk from a surface — the strongest "proven-useful" signal | `chunk_id` |
| `Creation` | synthetic (from `chunk.ts`, not a chain row) | Document recency baseline | — |

`AccessWeights` (`threads.rs:1098-1117`) already encodes a value ordering:
`explicit_recall = operator_selected = 1.0`, `recall_fault = 0.7`,
`lens_included = 0.5`. The reader is `ChainLogReader::access_history`
(`threads.rs:1144-1205`): for a set of `chunk_id`s it returns `(AccessKind, ts)`
pairs since a cutoff — an indexed `ts >= ?1 AND kind IN (...)` scan. **This is
the join primitive the value loop needs.**

It is consumed today **only by chunk-freshness** (the P7b `FreshnessFactory` in
`crates/query/src/freshness.rs`, an ACT-R base-activation curve over per-chunk
accesses) — i.e. it re-ranks *recall candidate chunks*, never *threads*.

### The crucial gap: access events key on `chunk_id`; threads key on `handle`

This is the whole reconstruction problem in one line. To answer "was a
**thread** surfaced then used?" you must bridge:

```
chain_log access event   →  chunk_id
thread                   →  handle  +  anchor_chunk_id  +  evidence_links[*].last_resolved_chunk_id
```

The bridge **exists and is durable**: every thread carries an
`anchor_chunk_id` (`ThreadRecord`, `threads.rs:143`) and a set of
`evidence_links` rows, each with a `last_resolved_chunk_id`
(`EvidenceLink`, `threads.rs:188`), written by the `AutoWeaver` whenever a
corpus chunk resonates with the thread's anchor above threshold
(`crates/attention/src/weaver.rs:760-778`). `ThreadsDb::list_evidence(handle)`
(`threads.rs:2394`) returns them. So:

> **thread → {anchor_chunk_id} ∪ {evidence_links.last_resolved_chunk_id} →
> access_history(those chunk_ids) → did any get ExplicitRecall /
> OperatorSelected after the thread was surfaced?**

is fully reconstructable from durable state. No new logging is required.

### Two reconstruction fidelities — and which to build

- **Weak / available now ("was the thread's content used?")**: union a thread's
  evidence chunk_ids, call `access_history`, sum weighted accesses (reuse
  `AccessWeights`) with a distinct-`query_hash` gate. This says *the material
  this thread binds got recalled/selected* — a strong proxy for value. **Build
  this.** Every input is durable today.
- **Strict / needs one new event ("was this thread surfaced, *then* used?")**:
  the true click-through loop wants a `surfaced` timestamp to compare against
  the `used` timestamp. Threads are surfaced via `attention_surface` /
  `score_thread` (the lens), but **there is no chain event for "thread surfaced
  in the lens."** `LensIncluded` logs *chunks*, not thread handles. **Closing
  the strict loop = add one event** (`ThreadSurfaced { handle, ts }` or extend
  `LensIncluded` with an optional `handle`) at the surface emit site
  (`crates/attention/src/lib.rs:1480`, where the `AttentionPage` is built).
  Then value = correlation(surfaced_ts, subsequent access on its chunks).

Recommendation: ship the **weak** loop first (no schema change, pure join), and
land the `ThreadSurfaced` event as the second increment so the strict
click-through becomes available — but the weak loop already moves the score in
the right direction and satisfies the THESIS success bar.

### Where the access loop is *already* a closed loop — and the template to copy

For the **concept** tier the loop is fully wired and is the exact blueprint:

- `memory_recall` (`server.rs:518`) runs recall, logs `ExplicitRecall` per hit
  (`:571`), then `observe_recall` mints/touches concept candidates and emits
  `ConceptAccessed{project, handle, reason, query_hash}` per resolved concept
  (`threads.rs:1343` `record_concept_accessed`; emit site in the memory façade
  `crates/mcp/src/memory.rs`, reason `recall:path` / `recall:known`).
- `activation.rs` (`scan_concept_signals` `:281`, `activations_internal`
  `:352`) reads those events and computes a concept's activation as
  `confidence + W_ACCESS·decayed_access + W_FOCUS·focus_lift + W_EDGE·edge_lift
  + W_NOTE·note_recency`. **`decayed_access` is the use-feedback term**: an
  ACT-R base-activation decay over **distinct `query_hash`** (the
  `salience-vs-familiarity` gate at `:326` — 50 hits from one chatty query
  collapse to one; proven by `distinct_query_gate_not_raw_count`, `:805`).

The thread scorer should grow the **same** `decayed_access` term, fed from the
**same** ledger, joined through `evidence_links`. The math is already written,
tested, and tuned (`ACT_R_DECAY_D = 0.5`, the `squash`/`act_r_base` pair) — axis
#2 is largely *reusing* `activation.rs`'s curves on the thread side.

### Live-substrate evidence that the loop is needed

`memory_surface(now)` on the live store shows the disconnect plainly. The
`active_concepts` whose use-feedback fires are a tiny minority:

- `ostk`: `distinct_queries = 6`, `decayed_access = 0.23` — genuinely used.
- `slipstream`: `distinct_queries = 1`, `decayed_access = 0.05`.
- **Every other active concept** (`ostk-recall`, `ostk-cache`,
  `cognitive-memory`'s neighbors, `dereference-or-void`, `the-lens`, …):
  `distinct_queries = 0`, `decayed_access = 0.0`. Their activation is carried
  almost entirely by `edge_lift` (the concept graph) + a trickle of
  `note_recency`. Use-feedback is *barely contributing* even on the tier where
  it is wired — because the recall surface that *fires* it is narrow.

Meanwhile the `active_threads` list (the tier with NO value signal) is exactly
the coherent-noise zoo the THESIS targets: `top-level` (resonance 75),
`pre-existing` (72), `self-sovereign` (77), `trust-root` (66), `follow-up`
(55), `re-read` (26), `hand-off` (39) — all ranked by raw `resonance` with no
notion of whether surfacing them ever paid off. Real ideas sit in the same list
(`cognitive-memory` 97, `ostk-cache` 200, `resonance-gated` 50,
`off-diagonal-bridge` 32) but are scored by the *same backwards metric*. A value
term is what separates them.

---

## (b) How the judgment artifacts link to handles — for confidence propagation

### The artifacts and how they are stored

| Artifact | Live count | Storage | Carries a thread/concept handle? |
|---|---|---|---|
| `ostk_decision` | 562 | **corpus chunk**, `source = "ostk_decision"`, `source_id` = decision slug (e.g. `durable_doctrine_requires_canonical_store`), durable `chunk_id`, `links.file_path → .ostk/decisions.jsonl` | **No direct handle** |
| `ostk_needle` | 212 | **corpus chunk**, `source = "ostk_needle"` | **No direct handle** |
| active concepts | ~28 | `concepts` table (`status='active'`, `confidence=1.0`), each with `concept_evidence` rows keyed on `(source, source_id)` and `concept_aliases` | The concept **handle** is the row key |

Verified via the live `recall(source="ostk_decision")` probe: decisions come
back as ordinary corpus hits with `source: "ostk_decision"`, `source_id`, and a
stable `chunk_id`. They are **content**, not graph nodes — there is **no
`decision → thread_handle` foreign key anywhere**. The only thing that relates
a decision to a thread/concept handle is **semantic proximity** (their
embeddings) or **textual co-occurrence** (a decision body mentions a concept,
or cites a code path the concept also cites).

This is the central design fact for judgment propagation: **the link is
embedding-mediated, not relational.** That is *fine* — it is precisely the
bridge the weaver and `concept_support_by_coord` already compute. We do not
need a new join table; we need to run the existing similarity bridge against the
curated chunk set.

### The propagation channels that already exist

1. **Active concept → coordinate → chunk (`concept_support_by_coord`,
   `activation.rs:515`).** For every evidence coordinate `(source, source_id)`
   of an *active* concept, it returns the highest-activation citing concept.
   This is *already* judgment propagation onto chunks (it's how the dormant lens
   `concept_support` slot lights, `crates/query/src/concept.rs`). Today its
   output flows to **recall candidate chunks**; axis #2 wants the same map keyed
   so it can reach **thread handles** (via the thread's anchor/evidence chunk
   coordinates).

2. **The weaver's anchor-similarity bridge (`weaver.rs:734-815`).** Every thread
   with an anchor is matched against incoming chunks; an above-threshold cosine
   writes a `Derived` evidence link. If the incoming/active chunk is a
   *decision* or *needle*, that evidence link is *already* a recorded
   "this thread resonates with a judgment artifact" edge — we just never read it
   as a *value* signal. `list_evidence(handle)` exposes the
   `category`/`similarity` per link; a link whose resolved chunk has
   `source ∈ {ostk_decision, ostk_needle}` is a propagation hit.

3. **`ConceptAccessed` reasons already distinguish surfaces** (`recall:path`,
   `recall:known`, `recall:symbol`) — `distinct_sources` is tracked
   (`activation.rs:382`). The same `reason` discipline can tag a
   judgment-sourced touch (`judgment:decision`) so propagated confidence is
   auditable in the `why`.

### Why "confidence FROM curated TO co-occurring handles" is sound here

Active concepts sit at `confidence = 1.0`; decisions/needles are
operator-authored records (the immaculate tier). A handle that **co-occurs**
with them — by embedding proximity (anchor resonates with the decision chunk) or
by citing the same `(source, source_id)` coordinate — is *more likely to be a
real idea than coherent noise*, because harness vocab (`top-level`,
`re-read`, `turn-digest`) does **not** resonate with the *content* of a
considered decision; it resonates with the *plumbing* of every turn. This is the
value-side complement to axis #1's specificity: specificity asks "does this term
discriminate across contexts?"; judgment propagation asks "does this term land
near the things a human/curator deemed worth recording?" The two are
correlated but independent lifts, and both demote the same noise.

---

## Recommended mechanism (one coherent value lift, two composable terms)

Both terms resolve to a **single per-handle scalar `value ∈ [0,1]`** precomputed
outside the hot scorer and folded into `compute_score_parts`. This honors the
THESIS integration constraint (`salience = specificity × value × recency −
negative_penalty`, sequenced & individually toggleable).

### Integration seam (where it hooks)

`compute_score_parts` (`crates/attention/src/lib.rs:717`) reads only
`ThreadState`. It must **not** do ledger I/O in the hot path (it runs per
thread per surface). Mirror `stop_handles` exactly:

- `InMemoryAttention` already carries `stop_handles: Arc<HashSet<String>>`
  (`lib.rs:871`), installed via `with_stop_handles` (`:915`) and consulted as
  `forced_stop` at the surface site (`:1475`). **Add a sibling
  `value_scores: Arc<HashMap<String, f32>>`** installed via a
  `with_value_scores(...)` builder, refreshed by the same offline/ambient cadence
  that recomputes stop handles (the `AutoWeaver::consolidate` pass /
  `spawn_ambient_daemons`).
- In `compute_score_parts`, look up `value = value_scores.get(handle)` and apply
  it as a **multiplier on the resonance/floor terms** (the principled form;
  composes with axis #1's specificity multiplier on the same terms) plus a small
  additive `GAMMA · value` so a high-value idle thread can clear
  `ARCHIVE_THRESHOLD` even at low present resonance. Flag-gated: `value = 1.0`
  (identity) when the feature is off, so the A/B harness can diff old vs new on
  the live thread set.

This keeps the scorer pure and O(1), and makes the value contribution appear in
the `ScoreAttribution`/`why` (add a `value` axis next to `off_diagonal_lift` at
`lib.rs:1484`) — satisfying `abi-as-sovereign-boundary` ("argue with the math").

### Term 1 — `value_use` (the click-through loop)

Computed by a new reader method on `ThreadsDb` (it already owns both
`evidence_links` and `chain_log`), run at refresh cadence:

```
for each thread handle H:
    chunks = {anchor_chunk_id(H)} ∪ {ev.last_resolved_chunk_id for ev in list_evidence(H)
                                     if ev.relation_state == Active}
    accesses = access_history(chunks, since)                 # threads.rs:1144
    # distinct-query gate, exactly like scan_concept_signals (activation.rs:326):
    # collapse accesses sharing a query_hash; weight by AccessKind via AccessWeights
    raw = Σ over distinct (query_hash | kind) of AccessWeights.weight_of(kind)
          · age_decay(ts)                                    # reuse act_r_base / squash
    value_use(H) = squash(raw)                               # [0,1)
```

Notes:
- **Reuse `activation.rs` wholesale**: `act_r_base`, `squash`, `age_hours_floored`,
  `ACT_R_DECAY_D`, `AccessWeights`. No new curve.
- **`operator_selected` is the gold signal** but has no producer yet
  (`threads.rs:356`). The loop should weight it (it's in `AccessWeights`), and a
  cheap win is to add its producer where a future operator-select surface lands;
  until then `explicit_recall` carries the loop.
- **Decay surfaced-but-never-landed**: a thread with surface exposure (lens
  inclusions of its chunks) but no `explicit_recall`/`operator_selected` follow
  gets `value_use → 0`, so it loses the multiplier and falls. With the strict
  `ThreadSurfaced` event this becomes an explicit penalty
  (`surfaced_count > 0 ∧ used_count == 0 → damp`), the negative complement of
  the loop.

### Term 2 — `value_judgment` (curated-confidence propagation)

Computed at the same cadence, embedding-bridge based (no new join table):

```
judgment_chunks = corpus chunks where source ∈ {ostk_decision, ostk_needle}
active_coords   = concept_support_by_coord(since)           # activation.rs:515  (already judgment-weighted)
for each thread handle H with anchor vector v(H):
    # (i) direct evidence: any evidence link of H resolves to a judgment chunk
    j_evidence = max similarity over ev in list_evidence(H)
                 where chunk(ev).source ∈ {ostk_decision, ostk_needle}
    # (ii) active-concept coordinate overlap: H's evidence coords intersect active_coords
    j_concept  = max active_coords[coord] over coord in evidence_coords(H)
    value_judgment(H) = saturating_combine(j_evidence, j_concept)   # 1 - exp(-Σ), like edge_lift_for
```

Notes:
- `(i)` is **already materialized** — the weaver writes `Derived` evidence links
  from threads to any resonant chunk, including decision/needle chunks; this
  just *reads* those links and filters by source. `category`/`similarity` are on
  the link.
- `(ii)` is the `concept_support_by_coord` map (active concepts are
  `confidence=1.0`, so their support *is* propagated curator confidence)
  intersected with the thread's evidence coordinates — generalizing the lens
  feature from chunks to handles. Reuse verbatim.
- Saturation curve mirrors `edge_lift_for` (`activation.rs:443`,
  `1 - exp(-Σconf)`), so many weak judgment links ≈ one strong one, bounded to 1.

### Combine

```
value(H) = clamp01( w_use · value_use(H) + w_judg · value_judgment(H) )
```

with `w_use`, `w_judg`, the multiplier/additive split, and the on/off flag in a
new `[salience]` (or `[attention]`) config block in
`crates/core/src/config.rs` next to `WeaverSettings` (THESIS: "behind a flag for
A/B"). Default weights start equal; the A/B harness tunes them against the
confirmed coherent-noise vs real-concept sets.

### Why this satisfies the success bar

- `cognitive-memory`, `ostk-cache`, `ostk-recall`, `dereference-or-void`,
  `relational-substrate-docgraph` are all (a) active concepts / cited by
  decisions (high `value_judgment`) and (b) the subjects agents actually recall
  (rising `value_use` as the recall surface widens). They get lifted.
- `top-level`, `pre-existing`, `follow-up`, `re-read`, `hand-off`,
  `turn-digest`, `squad-lead` resonate with *plumbing*, not with decision/needle
  *content*, and their evidence chunks are not what agents explicitly recall →
  `value ≈ 0` → the multiplier collapses their resonance/floor advantage. The
  curated stop-set becomes redundant for the same reason axis #1 predicts —
  value and specificity demote the same handles by independent routes, which is
  the cross-check the design wants.

---

## Open questions / risks to hand to Design (D)

1. **Cold-start sparsity.** `value_use` is ~0 for almost every handle today
   (the recall surface that fires `ExplicitRecall` is narrow — see the live
   `distinct_queries=0` across active concepts). Until recall traffic grows,
   `value_judgment` carries the lift. Design must decide the `w_use:w_judg`
   ratio and whether `value` defaults to a **neutral 1.0 multiplier** (so a
   value-less thread is unpenalized) vs a **<1 damp** (so absence of value is
   itself mild evidence of noise). The concept-tier convention (`ConceptSupport`
   returns 0.0 for the sparse case, *skipping* rather than penalizing,
   `concept.rs:116-131`) argues for neutral-1.0 to avoid double-counting with
   axis #1/#3.
2. **Refresh cadence & staleness.** The value map is recomputed offline (like
   stop handles). The live store currently shows **audit ingest lag** on some
   projects (`recall_stats.stale_ingest`); the value reader reads `chain_log`
   (threads.sqlite) not the lagging DuckDB audit, so it is unaffected — but the
   `evidence_links` it joins depend on the weaver having run. Cadence must align
   with `consolidate` so value doesn't lag the thread set it scores.
3. **Strict click-through needs the `ThreadSurfaced` event.** Decide whether to
   ship it in I3 (cleanest: emit at `lib.rs:1480`) or defer. The weak loop is
   enough for the first A/B; the strict loop is the "decay surfaced-but-never-
   landed" penalty that fully closes the THESIS framing.
4. **Embedding bridge cost.** `value_judgment` does cosine over the
   decision/needle chunk set per thread anchor. At 562 + 212 curated chunks ×
   ~44 threads this is cheap and offline, but Design should confirm it reuses
   the weaver's already-fetched anchor snapshot rather than re-querying Lance.
5. **`operator_selected` has no producer.** The strongest value signal is
   defined but never emitted. Flag for the team: a tiny producer (log it when an
   operator acts on a surfaced item) would massively strengthen the loop and is
   independent of the scorer work.

---

## Code anchors (for D / I3)

- Scorer (integration target): `crates/attention/src/lib.rs:717`
  `compute_score_parts`; surface emit + `why` at `:1480`; `ThreadState` at
  `:532`; `InMemoryAttention` + `stop_handles` pattern at `:853`/`:871`/`:915`;
  `forced_stop` consult at `:1475`.
- Access ledger: `crates/store/src/threads.rs` — `ChainEvent` access variants
  `:339-359`; `AccessKind` `:1063`; `AccessWeights` `:1098`; `ChainLogReader::
  access_history` `:1144`; `SqliteChainSink` `:1213`; `list_evidence` `:2394`;
  `EvidenceLink.last_resolved_chunk_id` `:188`; `ThreadRecord.anchor_chunk_id`
  `:143`.
- Concept use-feedback template (reuse the curves): `crates/store/src/
  activation.rs` — `scan_concept_signals` `:281` (distinct-query gate `:326`),
  `activations_internal` `:352` (`W_ACCESS·decayed_access` `:407`),
  `concept_support_by_coord` `:515` (judgment propagation onto coords),
  `act_r_base`/`squash` `:188`/`:198`, `edge_lift_for` saturation `:443`.
- Emit sites: `crates/mcp/src/server.rs:445` (`recall`→`ExplicitRecall`), `:504`
  (`recall_fault`→`RecallFault`), `:571` (`memory_recall`→`ExplicitRecall` +
  `observe_recall`); concept emit `record_concept_accessed`
  `crates/store/src/threads.rs:1343`.
- Weaver thread↔chunk bridge (where evidence links to judgment chunks are
  already written): `crates/attention/src/weaver.rs:734-815`
  (`match_against_anchors`), `:760` (`EvidenceLink` write).
- Lens concept feature (the chunk-side projection of judgment propagation):
  `crates/query/src/concept.rs` (`ConceptSupportInstance`, sparse-signal 0.0
  convention `:116-131`).
- Config seam: `crates/core/src/config.rs` `WeaverSettings` — add `[salience]`
  block + flag here.
