# R1 — Specificity (co-occurrence entropy / IDF) signal

> Research deliverable for THESIS §axis 1. Read-only investigation of the
> code paths + a live-substrate probe. Recommends a concrete, cheap
> specificity multiplier and exactly where it slots into the scorer.
> Author: `researcher-spec`. Blocks: D (design).

## TL;DR (the recommendation in one screen)

- **Signal.** For each thread handle, compute the Shannon entropy `H` of how
  its *resonances spread across distinct source documents*, and define
  `specificity = 1 − H/H_max`. A handle whose resonances all land in one
  doc/topic → `H≈0` → `specificity≈1`. A handle that resonates everywhere →
  `H≈H_max` → `specificity≈0`. This is IDF, restated as normalized entropy.
- **Source of the distribution.** The `evidence_links` table
  (`crates/store/src/threads.rs`) already records, per thread, **every corpus
  chunk that resonated with its anchor above threshold** — keyed
  `(thread_handle, original_path, category)` where `original_path` *is the
  corpus chunk-id*. Join each chunk-id → its `source_id` via
  `CorpusStore::fetch_chunks_by_ids` (the corpus carries `source_id` per
  chunk) and you have the per-handle co-occurrence histogram for free. No new
  schema; the weaver already writes these edges (`match_against_anchors`,
  `weaver.rs:734`).
- **When to compute.** **Precompute, not score-time.** `compute_score_parts`
  (`lib.rs:717`) only sees an in-memory `ThreadState` — it has *no DB handle*
  and runs in a hot `RwLock` read. Compute `specificity ∈ [0,1]` once per
  thread at the same cadence the counters are loaded (boot replay / the
  consolidate pass), cache it on a new `ThreadState.specificity: f32` field
  (default `1.0` = neutral), and have the scorer just multiply.
- **Where it slots in.** Multiply the **floor** (and deny lift below a cut)
  in `compute_score_parts`. The floor is the idle-dominance lever; gating it
  by specificity is the principled, continuous replacement for the binary
  `is_stop_handle` cliff at `lib.rs:749-754`. See "Recommended mechanism".
- **Why it works (proven on live data below).** Coherent-noise handles
  (`top-level`, `re-read`) resonate across many unrelated source docs (flat,
  high-entropy → low specificity); real concepts (`cognitive-memory`,
  `dereference-or-void`) concentrate in one topic/file (peaked, low-entropy →
  high specificity). The separation is large and unsupervised — it demotes
  the stop-set with no hand-list.

---

## 1. Where the co-occurrence data lives (code-anchored)

### 1a. `evidence_links` = the chunk→thread co-occurrence record (THE source)

`crates/store/src/threads.rs:1450` schema:

```sql
CREATE TABLE IF NOT EXISTS evidence_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_handle TEXT NOT NULL REFERENCES threads(handle) ON DELETE CASCADE,
    original_path TEXT NOT NULL,          -- == corpus chunk-id (see below)
    ...
    last_resolved_chunk_id TEXT,
    similarity REAL,
    touch_count INTEGER NOT NULL DEFAULT 1,
    ...
    UNIQUE(thread_handle, original_path, category)
);
CREATE INDEX idx_evidence_thread ON evidence_links(thread_handle);
```

The weaver writes one row per `(chunk, anchor)` pair that clears the
per-source cosine threshold (`weaver.rs:760`):

```rust
let link = EvidenceLink {
    thread_handle: thread.handle.clone(),
    // The corpus chunk-id is the only durable handle we have on the
    // resonating content; the threads scanner is what knows about paths.
    original_path: PathBuf::from(chunk_id),   // <-- chunk-id stored as path
    last_resolved_chunk_id: Some(chunk_id.clone()),
    similarity: Some(sim),
    touch_count: 1,
    ...
};
self.threads.add_evidence_link(&link)        // UNIQUE → idempotent
// on UniqueViolation: touch_evidence_link() bumps touch_count   (weaver.rs:793)
```

So **the set of `original_path` values for a given `thread_handle` is exactly
the set of corpus chunks that have ever resonated with that thread's anchor**,
and `touch_count` is how many times each one re-resonated. That is the raw
co-occurrence histogram. Read it with the existing
`ThreadsDb::list_evidence(handle)` (`threads.rs:2394`), which already returns
`original_path`, `last_resolved_chunk_id`, `similarity`, and `touch_count`.

This is the **right** source. Note THESIS §key-facts says "co-occurrence data
lives here / in the threads evidence graph" — the chunk→thread `evidence_links`
is the dense, populated half; the thread→thread graph is the sparse,
hand-edited half (see 1c).

### 1b. Chunk-id → source document, for document-frequency entropy

Chunk-ids are per-chunk; entropy over distinct *source documents* is more
robust to chunking artifacts than entropy over raw chunk-ids. The corpus
provides the join cheaply: `CorpusStore::fetch_chunks_by_ids`
(`corpus.rs:799`) returns each chunk with its `source_id` (`corpus.rs:853,946`).
So the histogram bins are `source_id` (or `(project, source_id)`), counted by
distinct resonating chunks (or summed `touch_count`).

This is the same `(project, source_id)` grouping the activity-burst surface
already uses (`corpus.rs:1456`, "one row per `(project, source_id)` pair"), so
the pattern is established in-codebase.

### 1c. The thread→thread graph is NOT the entropy source (sparse)

`thread_thread_links` (`threads.rs:1506`) is "v0 is hand-edited only;
auto-proposal by the weaver is deferred." Live probe confirms it is empty for
the handles that matter:

```
thread_evidence(handle="top-level", direction="from") -> { "edges": [] }
```

Do **not** base specificity on `thread_evidence` — it has no autonomous
population path. Use the chunk→thread `evidence_links` (1a/1b).

### 1d. What the scorer has in scope at score time (the precompute constraint)

`compute_score_parts(state: &ThreadState, attention_vec, now, forced_stop)`
(`lib.rs:717`) sees **only** the in-memory `ThreadState` (`lib.rs:531`):

```rust
struct ThreadState {
    tension: f32,
    mentions: u32,        // doc-frequency proxy (drives decay rate only)
    resonance: u32,       // resonance-gated salience counter (drives floor)
    last_touched_at: DateTime<Utc>,
    depth: FoldDepth,
    fade_multiplier: f32,
    anchor: Vec<f32>,
    origin: ScopeKey,
    origin_was_private: bool,
}
```

There is **no `CorpusStore`/`ThreadsDb` handle in the scorer**, and
`score_thread`/`surface` (`lib.rs:1673`, `lib.rs:1430`) iterate the in-memory
`scopes` map under an `RwLock` read guard. **Conclusion: specificity must be a
precomputed scalar carried on `ThreadState`, not a live SQL query inside the
scorer.** This mirrors how `resonance`/`mentions` are themselves durable
counters loaded via `seed_counters` (`lib.rs:969`) on the `replay` path
(`lib.rs:1286`), and seeded into `ThreadState` at the four `or_insert_with`
sites (`lib.rs:985, 1416, 1576, 1607`).

---

## 2. Live-substrate probe — the real co-occurrence distributions

The threads DB at `~/.local/share/ostk-recall/threads.sqlite` is not directly
readable from this sandbox (macOS TCC blocks `~/.local/share`; the serve
process also holds it), so I probed the *running* substrate via the
ostk-recall MCP tools. `recall(query=handle)` returns the corpus chunks a
handle co-occurs with, across source docs — a direct read-out of the
co-occurrence distribution the entropy is computed over.

### 2a. `mentions` / `resonance` confirm the THESIS frequency-is-backwards premise

From `thread_list(active)` + `attention_surface` (`why` block), the
confirmed coherent-noise handles carry huge frequency and tiny resonance rate:

| handle            | mentions | resonance | rate (res/men) | class            |
|-------------------|---------:|----------:|---------------:|------------------|
| `ostk-recall`     |    2019  |     334   |   0.165        | (mixed: name+noise) |
| `top-level`       |    1237  |      71   |   0.057        | coherent noise   |
| `attention-mcp`   |     324  |      26   |   0.080        | coherent noise   |
| `long-lived`      |     255  |      26   |   0.102        | coherent noise   |
| `hand-off`        |     225  |      38   |   0.169        | coherent noise   |
| `per-process`     |     137  |      43   |   0.314        | borderline       |
| `cognitive-memory`|     106  |      95   |   0.896        | real concept     |
| `ostk-cache`      |     208  |     200   |   0.962        | real concept     |
| `re-read`         |      24  |      24   |   1.000        | **coherent noise, rate-blind!** |

The last row is the killer: `re-read` has rate `1.0`, so the frequency-only
`is_stop_handle` classifier (`rate < 0.05`) **cannot** see it — yet it is pure
procedural filler. This is precisely the gap §axis 1 says specificity must
close, and it is exactly the kind of handle the curated stop-set was invented
to patch by hand.

### 2b. Co-occurrence spread — the entropy signal, measured

`recall(query=...)` top-10 hits, classified by how concentrated the
co-occurring docs are:

**`top-level` (coherent noise) — FLAT / high entropy:**
Distinct, topically-unrelated contexts across **8+ source docs**, 2 projects,
Jan–Jun span:
- "get the top-level structure" · "create the top-level README" ·
  "gate a new top-level verb" · `top_level_matches_whitelist` ontology test ·
  "Walk top-level `ostk --help`" · "add a `Verify` command at the top-level".
No shared topic — `top-level` is a generic English modifier. **High `H`.**

**`re-read` (coherent noise, rate-blind) — MAXIMALLY FLAT:**
Every hit is a different unrelated file-reading event across **10 distinct
sessions / 5+ months / multiple projects**:
- "re-read the spec" · "re-read the actual content" · "re-read the exact text"
  · "re-read it to satisfy the tool requirement" · "file changed … let me
  re-read". Zero topical cohesion. **Maximal `H` → specificity ≈ 0.**

**`cognitive-memory` (real concept) — PEAKED / low entropy:**
All 10 hits are about the *one* cognitive-memory feature, concentrated in 2
projects, nearly all literally referencing `.ostk/thread/cognitive-memory`:
- "reading the cognitive-memory thread to identify P0" · "update
  .ostk/thread/cognitive-memory" · "Implement … P0 of cognitive-memory" ·
  "fast-forward `cognitive-memory-v06`". The term's *meaning* constrains where
  it appears. **Low `H` → high specificity.**

**`dereference-or-void` (real concept) — MAXIMALLY PEAKED:**
All 10 hits collapse onto a **single source file**
`src/machined/dereference.rs` and its tests, one project (haystack):
- `dereference_evidence` fn · `DereferencedEvidence` struct · rule-N tests.
**Lowest `H` → specificity ≈ 1.**

### 2c. The quantified separation

Using "number of distinct source docs the top-k resonances span" as a cheap
proxy for entropy spread (more docs spanned at equal hit count ⇒ higher `H`):

| handle               | distinct source docs (top-10) | topic cohesion | specificity (expected) |
|----------------------|------------------------------:|----------------|------------------------:|
| `re-read`            | ~10 (every hit unique)        | none           | ~0.0 (demote)          |
| `top-level`          | ~8                            | none           | ~0.1–0.2 (demote)      |
| `cognitive-memory`   | ~5 but all one topic/handle   | single feature | ~0.7–0.8 (keep)        |
| `dereference-or-void`| 1 file                        | single symbol  | ~0.95 (keep)           |

The coherent-noise / real-concept gap in this proxy is wide and monotonic with
the THESIS-confirmed labels — including `re-read`, which the rate cliff misses.
This is the empirical case that the entropy multiplier reproduces the stop-set
effect with **no hand-list**.

---

## 3. Recommended mechanism (concrete, code-anchored)

### 3.1 Definition

For thread `h`, let its evidence edges resonate with documents `d`, each with a
weight `w_d` (count of distinct resonating chunks in `d`, or summed
`touch_count` — see 3.4). Let `p_d = w_d / Σ w` and

```
H(h)         = − Σ_d p_d · log(p_d)          # Shannon entropy of the doc spread
H_max(h)     = log(N_eff)                     # N_eff = number of populated bins
specificity  = 1 − H(h) / H_max(h)            # ∈ [0, 1]; 1 = peaked, 0 = flat
```

Guards (so the multiplier never punishes thin-but-real concepts):
- **`N_eff <= 1` → `specificity = 1.0`** (a handle seen in one doc is maximally
  specific; `H_max = 0`, avoid div-by-zero).
- **Evidence floor**: if a handle has fewer than `SPECIFICITY_MIN_EVIDENCE`
  total resonating chunks (suggest `5`), set `specificity = 1.0` (neutral) —
  not enough evidence to call it diffuse, mirroring `STOPWORD_MENTION_MIN`'s
  "need enough frequency first" guard at `lib.rs:96`.
- Normalize by `log(N_eff)` (entropy of the *populated* bins), not
  `log(corpus_doc_count)`, so a handle with 30 resonances spread evenly across
  30 docs reads as maximally diffuse regardless of total corpus size.

This is the principled generalization of `is_stop_handle`: that function is a
1-bin entropy estimate with a binary cutoff (`lib.rs:103-116`); this is the
full multi-bin entropy as a continuous multiplier.

### 3.2 Where it slots into `compute_score_parts` (the exact edit shape)

Add `specificity: f32` to `ThreadState` (default `1.0`) and gate the **floor**
— the idle-dominance lever — by it. Current code (`lib.rs:749-760`):

```rust
let is_stop = forced_stop || is_stop_handle(state.mentions, state.resonance);
let resonance_floor = if is_stop {
    familiarity_floor(0)
} else {
    familiarity_floor(state.resonance)
};
let floor = resonance_floor * state.fade_multiplier;
let decay_term = floor * (-decay_rate(state.mentions) * dt_days).exp();
let resonance_term = ALPHA * resonance;
let lift_term = BETA * off_diagonal_lift(state.tension, resonance, state.resonance, is_stop);
let score = decay_term + resonance_term + lift_term;
```

Recommended (behind the A/B flag — see 3.3):

```rust
// Specificity gate: a handle whose resonances spread across many unrelated
// docs (high co-occurrence entropy) carries little information; one that
// concentrates does. `specificity ∈ [0,1]` is precomputed from evidence_links
// (default 1.0 = neutral). It multiplies the idle floor — the principled,
// continuous replacement for the `is_stop_handle` cliff — so a diffuse handle
// decays to the unresonant baseline without a hand-list.
let spec = if salience_v2 { state.specificity } else { 1.0 };
let resonance_floor = if is_stop {
    familiarity_floor(0)
} else {
    familiarity_floor(state.resonance) * spec      // <-- continuous clamp
};
let floor = resonance_floor * state.fade_multiplier;
let decay_term = floor * (-decay_rate(state.mentions) * dt_days).exp();
let resonance_term = ALPHA * resonance;
// Deny the off-diagonal surprise to diffuse handles too: a "surprise" from a
// resonate-with-everything term is not a surprise. Treat low specificity like
// is_stop for lift purposes.
let lift_is_stop = is_stop || spec < SPECIFICITY_LIFT_CUTOFF;   // suggest 0.2
let lift_term = BETA * off_diagonal_lift(state.tension, resonance, state.resonance, lift_is_stop);
let score = decay_term + resonance_term + lift_term;
```

Rationale for gating the **floor** (not the live `resonance_term`):
- The floor is what buys *idle dominance* — the exact failure mode the
  curated stop-set exists to suppress (`lib.rs:745-748`, `with_stop_handles`).
  Clamping it to baseline for diffuse handles is the principled version of
  `forced_stop`.
- Leaving `ALPHA · resonance` (the live term) ungated preserves the desirable
  "can light up while actively discussed but never buys idle dominance"
  behavior the comment at `lib.rs:745-748` already articulates for stop-handles.
  A diffuse handle still surfaces *when you're literally talking about it*, it
  just doesn't sit warm at idle.

Continuous, not binary: at `specificity = 1.0` (real concept) the new scorer is
identical to today's; at `specificity ≈ 0` (re-read/top-level) the floor
collapses to `familiarity_floor(state.resonance) · ε`, ranking it with the
`is_stop` baseline — reproducing the stop-set's clamp without naming anything.

### 3.3 Config flag (A/B, per integration constraint)

Add to `crates/core/src/config.rs`. Two viable shapes; recommend a small new
`[salience]` block so all four axes share one toggle surface (THESIS calls for
"one coherent scorer evolution"), with `WeaverSettings` left as-is:

```toml
[salience]
scorer_v2 = false                 # master A/B flag; false = today's scorer
specificity_enabled = true        # axis #1 (composable, individually toggleable)
specificity_min_evidence = 5      # neutral below this many resonating chunks
specificity_lift_cutoff = 0.2     # deny off-diagonal lift below this
```

Mirror `WeaverSettings::resolve(Option<&toml::Value>)` (`config.rs:473`) and its
serde-default pattern (`config.rs:411`, the `default_weaver_stop_handles` test
at `config.rs:1153`) so an absent block resolves to defaults. The flag threads
into `InMemoryAttention` exactly like `stop_handles` does today
(`Arc`-wrapped field at `lib.rs:871`, wired via a `with_salience_config(...)`
builder analogous to `with_stop_handles`).

### 3.4 Precompute + cache: where and how

Compute `specificity` once per thread, off the hot path, and seed it onto
`ThreadState` alongside the counters. Two equivalent entry points (both already
load every thread):

1. **Boot / replay** (preferred for v1): when the score tier is rebuilt, for
   each thread call `threads.list_evidence(handle)` → collect `original_path`
   (chunk-ids) + `touch_count` → `corpus.fetch_chunks_by_ids` to map chunk-id →
   `source_id` → build the `source_id` histogram → `specificity_from_hist(...)`.
   Seed it with the counters (extend `seed_counters` or add a sibling
   `seed_specificity`, and a `ReplayEvent` field, analogous to the
   `Familiarize` event at `lib.rs:1316`).
2. **Consolidate pass** (`weave_window`/`consolidate`, `weaver.rs:299`): the
   weaver already walks threads + evidence each cycle to re-touch edges and
   fade tension. Recompute specificity there so it tracks drift as new
   resonances accrue, and persist it (a `threads.specificity REAL` column, or a
   derived view) so the next boot loads it without a recompute.

Cost: `O(total evidence_links)` once per boot/consolidate — the same order as
the work the weaver already does. **Zero added cost at score time** (one f32
multiply). This satisfies §axis 1 "cheap, unsupervised."

Histogram weight choice (`w_d`):
- **Recommended v1: distinct resonating chunks per `source_id`** (count of
  evidence rows grouped by source doc). Simplest, robust, and directly the
  "how many places does this resonate" signal.
- **v2 option: sum of `touch_count`** per doc — weights by how *insistently*
  each doc re-resonates. Slightly richer but `touch_count` is already consumed
  by `edge_activation`; keep v1 simple and revisit if needed.

### 3.5 Interaction with the existing stop-set and the other axes

- **Stop-set becomes redundant (the success criterion).** Once specificity is
  on, `top-level`/`re-read`/`turn-digest` self-demote via low specificity. The
  A/B harness should show specificity-alone reproduces the curated stop-set's
  ranking effect (THESIS proof bar). Keep `forced_stop` wired during A/B as a
  safety net; the V phase decides when to retire `default_weaver_stop_handles`.
- **Composes multiplicatively with value (axis #2) and the negative penalty
  (axis #3)**, per the unified form `salience = specificity × value × recency −
  negative_penalty`. Specificity and value both gate the floor; design D must
  decide order (recommend `floor · specificity · value`, both ∈ [0,1], so each
  is an independent dampener and neither can resurrect idle dominance).
- **Negative transfer (axis #3)** is a *complementary* embedding-space signal
  (proximity to rejected/stopped centroid); specificity is a *statistical*
  co-occurrence signal. They agree on `top-level`/`re-read` but via independent
  evidence — good for robustness.

---

## 4. Unit-test strategy (model on the two named tests)

Model on `stop_handle_floor_clamped_by_context_diversity` (`lib.rs:2383`) and
`curated_stop_set_clamps_high_resonance_harness_handle` (`lib.rs:2431`), which
already use `seed_anchor` + `seed_counters` + `surface` to assert relative
ranking. Add a `seed_specificity` test helper (or extend `seed_counters`) and:

1. **`specificity_demotes_diffuse_handle_with_no_hand_list`**: two threads,
   identical counters (e.g. `300/290`) and identically-aligned anchors so the
   live resonance term is equal; one seeded `specificity = 0.05` (the `re-read`
   case), one `specificity = 0.95`. Assert the concentrated one out-ranks the
   diffuse one — *without* any `with_stop_handles`. This is the direct
   replacement-of-the-cliff proof.
2. **`specificity_one_reproduces_current_scorer`**: `specificity = 1.0` ⇒ score
   bit-identical to flag-off. Guards the A/B "no regression at neutral."
3. **`specificity_from_hist_*` pure-fn tests** (in the store or a `salience`
   module): single-bin → `1.0`; uniform-N-bins → `≈0.0`; peaked (one dominant
   bin + tail) → high; below evidence floor → `1.0` neutral.
4. **`specificity_does_not_block_active_discussion`**: a diffuse handle with a
   high *live* resonance against the current attention vector still surfaces
   (floor clamped, but `ALPHA·resonance` intact) — proving "light up while
   discussed, never idle-dominate."

A/B harness (V phase): load the live thread set, compute specificity for each,
rank old vs new, and assert `{re-run, turn-digest, squad-lead, non-blocking,
top-level, re-read}` all fall below `{cognitive-memory, ostk-cache,
dereference-or-void, relational-substrate-docgraph}` under the new scorer, and
ideally that specificity-alone ≈ the stop-set's effect.

---

## 5. Open questions for design (D)

1. **Floor-only vs floor+resonance gating.** Recommendation is floor-only (3.2).
   D should confirm against the value axis: if value also gates the floor, the
   product of three [0,1] dampeners could over-suppress a high-value-but-diffuse
   handle. Mitigation: make value *additive/boosting* and specificity/negative
   *multiplicative/damping*, or floor each dampener at a small ε.
2. **`source_id` vs `(project, source_id)` bins.** Cross-project spread (a term
   appearing in haystack *and* ostk-recall) is arguably *more* diffuse. Suggest
   binning by `(project, source_id)` — strictly more bins for cross-project
   terms → higher entropy → lower specificity, which is the desired direction.
3. **Persistence.** Add a `threads.specificity` column vs recompute-on-boot.
   Recompute-on-boot is simpler and always fresh; a column saves the boot
   recompute and lets `attention_surface why` expose specificity for the
   self-audit axis (#4). Lean toward the column once the formula stabilizes.
4. **Decay of specificity.** As a handle accrues new diffuse resonances its
   specificity should *fall* over time; recomputing each consolidate pass
   (3.4 option 2) gives this for free. Confirm cadence in D.

## Appendix — exact code anchors

- Scorer: `crates/attention/src/lib.rs:717` `compute_score_parts`; floor at
  `:749-755`; `is_stop_handle` `:110`; `familiarity_floor` `:172`;
  `off_diagonal_lift` `:284`; consts `ALPHA=1.0 :64`, `BETA=0.5 :66`,
  `FAMILIARITY_SATURATION=20 :79`, `STOPWORD_MENTION_MIN=50 :96`,
  `STOPWORD_RATE_MAX=0.05 :101`.
- `ThreadState` (precompute slot): `lib.rs:531`; seed path `seed_counters`
  `:969`; `or_insert_with` sites `:985,1416,1576,1607`; `replay`/`ReplayEvent`
  `:1269,:1316`; `stop_handles` field + `with_stop_handles` `:871`.
- Reference tests: `lib.rs:2383`, `lib.rs:2431`.
- Evidence data: `crates/store/src/threads.rs` schema `:1450`,
  `EvidenceLink` `:182`, `list_evidence` `:2394`, `add_evidence_link` `:2326`,
  `touch_evidence_link` `:2371`; thread→thread (sparse) `:1506`.
- Weaver edge writer: `crates/attention/src/weaver.rs:734` `match_against_anchors`,
  link build `:760`, re-touch `:793`; `weave_window`/consolidate `:299`.
- Corpus chunk→source_id join: `crates/store/src/corpus.rs:799`
  `fetch_chunks_by_ids` (carries `source_id` `:853,946`); `(project,source_id)`
  grouping precedent `:1456`.
- Config flag home: `crates/core/src/config.rs` `WeaverSettings` `:397`,
  `resolve` `:473`, serde-default pattern `:411`, test `:1153`.
