# Recalibration — specificity fires on real data + drift J-set fix

> Task #19, DIAGNOSE → PROPOSE (review gate). Triggered by the live A/B: with
> `scorer_v2` ON on the operator's real ledger, **specificity = 1.0 on every
> surfaced handle** (the axis is inert) and **`active_decided_drift = 0.97`**
> (the alarm is meaningless). This doc diagnoses both and proposes concrete
> recalibrations. NO code yet — review this first.
>
> Measurement basis: `attention_surface` + `recall_stats.salience_health` +
> `thread_list` on the live ledger (read-only MCP), plus the source paths for
> how `evidence_links` is populated and how the J-set is built. The threads
> SQLite is TCC-blocked from direct shell SQL in this seat, so the per-handle
> evidence-count *distribution* is argued from the population mechanism +
> corroborated by the live surface, not a raw `SELECT count(*)` — see §1.3 for
> the exact query to confirm the number if you want it before implementing.

---

## 0. TL;DR

- **Specificity is inert because its INPUT is wrong, not merely thin.**
  `specificity_from_evidence` reads `evidence_links`, but `evidence_links` is
  written **only by the `AutoWeaver`** (weave/consolidate), gated behind high
  per-source cosine thresholds (prose 0.82 / code 0.78 / transcript 0.85). The
  boot/serve path that the scorer runs in **never weaves**, and few corpus
  chunks clear those thresholds, so most thread handles have **< 5 Active
  evidence links** → `specificity_min_evidence` guard returns the neutral
  `1.0`. Lowering the threshold alone won't help: the signal is the wrong one.
- **The richer signal is already on the ledger and is non-degenerate for
  exactly the handles that matter.** Each thread has a `mentions` counter
  (kebab-token literal recurrence) that is huge for plumbing (`ostk-recall`
  2099, `in-memory` 1280, `top-level` 1274, `per-row` 296) and modest for real
  concepts (`cognitive-memory` 142, `dereference-or-void` 34). Frequency alone
  is the backwards signal — but the corpus chunks those mentions land in,
  grouped by `(project, source_id)`, give the **co-occurrence spread**
  specificity needs. **Recommendation: compute the histogram from the corpus
  chunks a handle's kebab-token occurs in (a corpus query keyed on the handle),
  not from `evidence_links`.** §2.
- **Drift is meaningless because the J-set is polluted.** `active_decided_drift
  = 0.97` and `drift_forgotten` lists **hundreds of `spec-*` / `draft-*` /
  `patents-*` / `llmos-*` / `man-*` doc-concepts** — file-seeded markdown nodes
  that were never "active threads." Scope J to genuine operator judgments
  (recent decisions + recently-*accessed* active concepts, excluding the bulk
  file-seeded doc population). §3.
- **One design fork for the operator** (§2.3): the corpus-mention histogram
  (cheap, no new resonance math, available today) vs an anchor-kNN
  doc-distribution (truer to "what this concept resonates with" but needs a
  per-handle ANN query in the boot pass). I recommend the former for v1.

---

## 1. Diagnosis — why specificity is inert

### 1.1 The input is weaver-populated, and the scorer's path never weaves

`evidence_links` rows are written exclusively by `AutoWeaver`
(`weave_window` / `consolidate` / `process_event`, `weaver.rs:760`
`add_evidence_link`; driven from `commands.rs:683/715` and the ambient daemons
at `commands.rs:385`). The match gate is a **per-source cosine threshold**
(`WeaverThresholds`: prose `0.82`, code `0.78`, transcript `0.85`, default
`0.80`) — deliberately high to keep chatter out of the thread graph.

The scorer runs in the **boot/serve path** (`re_anchor_threads_from_corpus` →
`precompute_salience_factors`), which reads `list_evidence_all()` but **never
runs the weaver**. So specificity sees whatever evidence the weaver happened to
accrue in prior scans — and at a 0.78–0.85 cosine bar, a given thread anchor
matches only a handful of corpus chunks. Net: **most handles have < 5 Active
evidence links**, tripping the `total_chunks < specificity_min_evidence (5)`
guard at `salience.rs:222`, which returns the neutral `1.0`.

> **Input wrong, not just thin (the question the brief asked).** Both, but the
> binding one is *wrong source*. Even if we dropped `specificity_min_evidence`
> to 2, a handle with 2 evidence links in 2 docs reads as maximally diffuse
> (specificity ≈ 0) on 2 data points — noisy and unfair to thin-but-real
> concepts. The evidence graph is a *curated, high-threshold* artifact; it was
> never meant to be a dense co-occurrence distribution. Specificity needs a
> dense per-handle signal, which `evidence_links` structurally is not.

### 1.2 The live surface corroborates it

`attention_surface` / `thread_list` (scorer_v2 ON), the handles specificity was
meant to demote, all sitting at the idle floor plateau, **un-demoted**:

| handle        | mentions | resonance | on the surface? | specificity (live) |
|---------------|---------:|----------:|-----------------|--------------------|
| `ostk-recall` |   2099   |    414    | yes, top        | 1.0 (inert)        |
| `in-memory`   |   1280   |     90    | yes             | 1.0                |
| `top-level`   |   1274   |    108    | yes             | 1.0                |
| `per-row`     |    296   |     23    | yes             | 1.0                |
| `attention-mcp`|   326   |     28    | yes             | 1.0                |
| `hand-off`    |    233   |     46    | yes             | 1.0                |
| `pre-existing`|    108   |    108    | yes             | 1.0                |
| `follow-up`   |     97   |     97    | yes             | 1.0                |
| `re-read`     |     62   |     62    | yes             | 1.0                |
| `cognitive-memory` | 142 |    131    | yes             | 1.0 (should be HIGH)|
| `dereference-or-void`| 34|     34    | yes             | 1.0 (should be HIGH)|

Every specificity reads `1.0` — confirming the axis contributes nothing; the
order is resonance-only, and the high-mention plumbing rides the floor. (The
A/B's near-zero rank delta is this: with spec=val=1.0 everywhere, v1 and v2
differ only by the tiny `neg_penalty` damp — `re-read` 0.075, `follow-up`/
`pre-existing` 0.060 — which barely reorders a flat ribbon.)

### 1.3 Exact confirmation query (optional, before implementing)

If you want the raw distribution before coding (TCC blocks it from my seat),
this is read-only against the live threads DB:

```sql
-- evidence-link count per thread handle (Active only), bucketed.
SELECT n_links, COUNT(*) AS handles FROM (
  SELECT thread_handle, COUNT(*) AS n_links
  FROM evidence_links WHERE relation_state = 'active'
  GROUP BY thread_handle
) GROUP BY n_links ORDER BY n_links;
```

Prediction: the mass is at `n_links ∈ {0,1,2,3,4}` (below the guard), with a
thin tail. `SELECT COUNT(*) FROM evidence_links` sizes the whole graph; I
expect it small relative to ~50 active threads × the corpus.

---

## 2. Proposal — a specificity input that fires

### 2.1 The signal: corpus chunks the handle's kebab-token occurs in, by `(project, source_id)`

Every thread handle is a kebab-case token (`cognitive-memory`, `top-level`).
The corpus already indexes the text those tokens appear in. The co-occurrence
distribution specificity wants is: **for handle `h`, the set of corpus chunks
whose text contains `h`'s token, bucketed by `(project, source_id)`** — then the
same `1 − H/ln(N_eff)` over that histogram.

Why this discriminates where evidence_links doesn't:
- `top-level` / `in-memory` / `per-row` / `ostk-recall` are generic English or
  ubiquitous project nouns — their token appears in **hundreds of distinct
  source docs across projects** → flat, high-entropy histogram → **low
  specificity** (demoted). This is the IDF the corpus side already computes for
  BM25 (R1 §1), now applied to the handle.
- `cognitive-memory` / `dereference-or-void` are coined terms — their token
  concentrates in a **handful of docs/files on one topic** → peaked, low
  entropy → **high specificity** (preserved).

This is dense (mentions are plentiful — the very frequency that makes plumbing
"coherent noise" *is* the broad co-occurrence that should sink it), available
today, and needs no weaver run.

### 2.2 Mechanism (keep the SAME single boot walk)

`precompute_salience_factors` already holds `&Arc<CorpusStore>`. Add ONE batched
corpus read keyed on the handle tokens (a substring/FTS match over
`chunk.text`, projected to `(project, source_id, chunk_id)`), then bucket per
handle. Concretely:
- New `CorpusStore` reader: `token_cooccurrence(handles: &[String]) ->
  HashMap<handle, Vec<SourceMeta>>` (or counts per `(project, source_id)`),
  one FTS/scan over the corpus restricted to the ~50 active-thread tokens.
  Bound it like the existing `fetch_chunks_by_ids` batch — one query, not
  per-handle.
- Feed that histogram into the existing `specificity_from_evidence` entropy
  math (rename to `specificity_from_histogram` taking the `(project,
  source_id)` counts, so the evidence-link path can stay as a fallback/secondary
  and the pure entropy core is unchanged + still unit-tested).
- `specificity_min_evidence`: with the dense signal, raise the floor to a
  meaningful "enough mentions to judge" (e.g. ≥ 20 token-hits) so a rarely-
  mentioned real concept isn't judged on noise; below it → neutral 1.0.

Cost: one extra corpus query in the boot pass (same scan budget as the existing
`fetch_chunks_by_ids`), still ONE walk. Score-time unchanged (precomputed
factor).

### 2.3 DESIGN FORK for the operator

Two viable co-occurrence sources; I recommend (A) for v1:

- **(A) corpus token co-occurrence** (recommended): handle-token → distinct
  `(project, source_id)` it appears in. Cheap, dense, available now, and it IS
  textbook IDF. Risk: a token that is a substring of unrelated words
  (`per-row` in "supper-rowing"? rare for kebab tokens) — mitigated by
  word-boundary matching. Also: the *literal token* may diverge from the
  *concept* (a doc about cognitive-memory that never writes the token) — but
  for demoting high-frequency plumbing (the actual goal) the token IS the
  signal.
- **(B) anchor-kNN doc distribution**: embed the handle's anchor, ANN the
  corpus, bucket the top-K neighbors by `(project, source_id)`. Truer to
  "what this concept *resonates* with" (semantic, not lexical), and reuses the
  embedding the weaver already has. Cost: a per-handle ANN query in the boot
  pass (~50 queries) — heavier, and K-choice is a new knob. Better signal for
  concepts whose token is rare but whose meaning is concentrated; overkill for
  the immediate "sink the plumbing" win.

Recommend shipping (A) now (it directly fixes the live inertness with the
cheapest faithful signal), and noting (B) as a follow-up if (A)'s lexical
proxy proves too coarse for specific real concepts.

---

## 3. Proposal — scope the drift J-set to real judgments

### 3.1 Why it's 0.97

`compute_salience_health` (server.rs) builds J from
`threads.concept_activations(None, since)` — **every** activated concept in the
window. The live `drift_forgotten` is dominated by file-seeded markdown
doc-concepts: `spec-*` (≈100+), `draft-*` (≈120+), `patents-*`, `llmos-*`,
`man-*`, `notes-*`, `release-hygiene-*`. These are crystallized typed nodes
from the docs corpus — they "activate" because they're in the graph, but they
are **not** operator judgments about what's salient *now*, and they are never
on the active *thread* surface A. So `J \ A` ≈ all of them → drift ≈ 1.0, and
the alarm is pure noise.

### 3.2 The fix: J = genuine, recent judgment signal

Scope J to handles that represent an actual recent operator judgment, not the
bulk doc population. Options (combine 1+2):

1. **Exclude file-seeded doc-concept kinds.** `ConceptActivation` carries
   `kind: Option<String>` (typed-node kind). The doc nodes are typed
   (`spec`/`draft`/`doc`/`patent`/…); real judgments are untyped observed
   concepts or decisions. Filter J to `kind == None` (untyped/observed) — or
   explicitly drop the known doc kinds. (Confirm the exact `kind` values the
   crystallizer assigns these — quick `concept list` check — before hardcoding.)
2. **Require a DYNAMIC activation signal, not just graph presence.** Use the
   `ConceptWhy` breakdown already computed in `activations_internal`: require
   `why.decayed_access + why.focus_lift > 0` (the concept was actually
   *accessed*/*focused* recently, mirroring `relational_support`'s
   `REL_SEED_THRESHOLD` seed filter, activation.rs:561) — a doc-concept that
   only carries durable `confidence` and was never touched is excluded.
3. **Add recent `ostk_decision` / `ostk_needle` handles** as the positive J
   core (these ARE recorded operator judgments — R4 metric 4's intent). The
   evidence-source filter already exists for value's `j_evidence`.

Recommended v1: **J = {active concepts with a non-zero recent dynamic signal
(option 2)} ∪ {handles cited in recent ostk_decision/ostk_needle (option 3)},
minus file-seeded doc kinds (option 1)**. That makes `J \ A` mean "a thing the
operator recently engaged that the surfacer dropped" — the real
`projection-truth` alarm — instead of "the docs corpus exists."

Cost: filter on the existing `concept_activations` result (no new query) + the
decision/needle handles (the value axis already reads those). Same boot/health
budget.

### 3.3 Note

This is a self-audit *metric* fix (observation only), independent of the
scorer; it can land in the same branch but gates nothing — it just makes the
`unhealthy` push and the `active_decided_drift` number trustworthy.

---

## 4. Recommended plan (for the implement phase, post-review)

On `feat/salience-specificity-recal` off main (e52bc4b):
1. Corpus `token_cooccurrence` reader (one batched query) + rewire
   `precompute_salience_factors` pass C to build the specificity histogram from
   it; keep the entropy core (`specificity_from_histogram`) + its unit tests;
   retune `specificity_min_evidence` to the token-hit scale.
2. Scope the health J-set (§3.2 v1) in `compute_salience_health`.
3. Re-verify with `salience-ab` against the live ledger: specificity should now
   be < 1.0 (and low) for `ostk-recall`/`in-memory`/`top-level`/`per-row`/
   `hand-off`, high for `cognitive-memory`/`dereference-or-void`; the P1 verdict
   should flip toward PASS; `active_decided_drift` should drop well below 0.97
   with a short, meaningful `drift_forgotten`.
4. Unit tests: token-cooccurrence histogram discriminates a synthetic
   plumbing-vs-concept fixture; J-set filter excludes doc-kinds + keeps a
   decision-cited handle.

Open question for the operator (the fork, §2.3): lexical token co-occurrence
(A, recommended, cheap) vs semantic anchor-kNN (B, truer, heavier). I propose A
for v1; flag if the operator wants B's semantic fidelity from the start.

---

## 5. FINAL IMPLEMENTATION (as shipped)

The implementation went through three bases under the live proof gate; this
records what shipped and why, so the audit trail isn't only in the thread.

### 5.1 What shipped: project-bucketed RAW SPAN

`CorpusStore::project_phrase_cooccurrence(handles)` — ONE projected
`(project, source, links_json, text)` scan (no FTS; lance-7 `full_text_search`
is broken against this corpus — see 5.2). For each handle, count chunks whose
text contains the de-hyphenated handle as an **ordered whole-phrase substring**
(`cognitive-memory` → `"cognitive memory"`), bucketed by **derived project**.
The boot pass (`precompute_salience_factors`) feeds each handle's per-project
counts to `salience::specificity_from_project_dist(counts, n_projects_global)`:

    specificity = 1 − (project_span − 1) / (N_projects_global − 1)

i.e. **raw distinct-project SPAN**, NOT entropy. A handle in 1 project → 1.0
(specific); in all → 0.0 (generic); linear between. `min_evidence` is dropped
(an IDF-like signal needs no "insufficient evidence" floor — rare → specific).

**Derived project, not the raw column (load-bearing).** ~80% of the corpus is
claude_code transcripts, ALL carrying the single shared `project =
claude-code-history` label. Bucketing on the raw column would collapse that 80%
into ONE bin (a constant +1 to every handle), riding the discrimination on the
~20% project-ful sources alone. So for claude_code chunks we derive the REAL
per-session repo from the transcript `file_path` in `links_json`
(`…/projects/-Users-…-projects-<REPO>/…` → `<REPO>`, via
`repo_from_transcript_path`), falling back to the `project` column for
non-transcript / unparseable rows (defensive — never panics). `N_global` is the
derived distinct-project count (38 on this corpus, up from 23). This makes the
bulk of the corpus contribute its true project structure and fixes a real
mis-scoring class: a transcript-ONLY handle previously had span≈1 → falsely
specific; it now spans its real repos.

Live verification (`salience-ab`, honest gate = the organic concept above the
text-rich plumbing), AFTER the derivation (N_global=38) — and it SHARPENED vs
the raw-column version (N=23, shown in parens):
**cognitive-memory 0.838** (was 0.773; organic, 171 mentions) clearly above
**in-memory 0.459** (was 0.409; span 14→21), **hand-off 0.676** (0.545),
**top-level 0.784** (0.682). in-memory demoted hardest (−72 ranks). PASS — the
concept rose, the worst plumbing stayed lowest. (Strict full-set P1 still trips
only on the §5.3 project-local-noise cases — pre-existing/system-reminder — not
on the named handles; documented limitation, not a regression.)

### 5.2 The path here (three bases, two failed)

1. **Lexical token-AND** (preserved on `wip/salience-lexical-cooccurrence`):
   de-hyphenated handle, all tokens present anywhere (FTS `Operator::And`).
   **INVERTED** — a coined multi-word concept's individually-common component
   words AND-match everywhere → looked generic. (FTS was abandoned anyway: the
   index lacks token positions, and lance-7 `full_text_search` trips a
   `record batch must have the same length` assertion regardless of query shape.)
2. **Semantic anchor-kNN** (the documented follow-up basis): mean top-K cosine
   of the handle's anchor neighborhood. Directionally right for *measured*
   concepts, but **coverage hole** — the worst plumbing (in-memory, top-level,
   per-row, hand-off, ostk-recall) all have NULL `anchor_vec` (old, text-rich,
   never re-anchored), so they defaulted to neutral 1.0 and escaped demotion.
   Complementary to approach 1 (which measures exactly those text-rich handles).
3. **Project span (shipped).** Two corrections made it work:
   - **Entropy → raw span.** The first project version normalized Shannon
     entropy of the project-COUNT distribution by `ln(N_global)`. Entropy
     weights by match VOLUME, and every handle is bursty in its home project, so
     concepts and plumbing both collapsed into ~0.6–0.7 (top-level 0.707 >
     cognitive-memory 0.660 — FAIL). The diagnostic proved distinct project
     COUNT (6 vs 14) discriminates; raw span measures count and ignores volume,
     so the home-project burst can't rescue a broad-span term. Fixed it.
   - **Two coverage bugs.** Specificity was sourced from, and applied only to,
     ANCHORED handles — so anchor-poor text-rich plumbing (in-memory/hand-off)
     was never measured or written, defaulting to neutral 1.0. Now sourced from
     EVERY thread handle, and the factor loop includes `project_dist.keys()`.

### 5.3 KNOWN LIMITATION — project-local noise (delegated, not fixed)

Specificity is a clean **focused-vs-broad** signal: a handle concentrated in
few projects is specific. It therefore **cannot distinguish a rare coined
concept from a rare noise phrase** — both are project-local, both read as
specific. Examples: `dereference-or-void` (0 prose matches → 1.0, correct — we
WANT rare coinages elevated) and `system-reminder` / `pre-existing` (noise
phrases local to one project → ~1.0, a false positive). Any span/IDF measure is
blind to this; a min-doc threshold would re-break the rare-coined-concept case
we explicitly want elevated, so we do NOT tighten specificity here.

Noise suppression is the job of the OTHER axes, by design:
- **value** (axis 3): a surfaced-but-never-used handle decays to low value.
- **negative-transfer** (axis 2): proximity to the rejected/dormant centroid.
- the curated **`[weaver] stop_handles`** set: the explicit harness-vocab lever.
Specificity stays focused; these carry the project-local-noise case.

### 5.4 Bankable (kept) + dead-code note

- **drift-J fix** (`compute_salience_health`): J scoped to concepts with a
  non-zero recent dynamic signal — `active_decided_drift` curated_ratio off→on
  0.066 → 0.000; the spec-*/draft-* doc-concept flood no longer inflates the
  alarm. Shipped.
- **`salience-ab` `spec` column** + the read-only diagnostic harness. Shipped.
- **Entropy core (`specificity_from_histogram`) + `specificity_from_evidence`**:
  NO LONGER on the live boot path (replaced by `specificity_from_project_dist`).
  Still exercised by the A/B fixture (`ab_harness.rs` uses
  `specificity_from_evidence` as a synthetic input) and unit tests; the entropy
  core is also the documented alternative basis. Kept as tested, fixture-used
  code — not dead, but no longer production-path. (If the fixture migrates to
  the span basis, both become removable.)
### 5.5 FOLLOW-UP BASIS — semantic anchor-kNN (documented, not committed)

The semantic basis was prototyped and set aside; it was never committed (the
prototype is not on any branch — this prose is its durable record, by decision:
reflog can't reliably recover overwritten working-tree edits, and the value is
the idea + the fallback, not the half-experiment). Pursue it when the project-
local-noise limitation (§5.3) or the coined-unwritten gap (0-text handles
defaulting to 1.0) needs closing.

- **Mechanism.** For each handle, embed/reuse its anchor (the re-anchor
  centroid), ANN the corpus top-K, and measure neighborhood CONCENTRATION =
  mean top-K cosine (or cosine at rank K). Tight/high = specific; diffuse/low =
  generic. `nearest_to` is the proven dense lane; ~50 ANN at boot, bounded. Do
  NOT bucket the neighbors by source_id + take entropy — that re-imports the
  doc-frequency proxy that inverted (a coherent concept with many close docs
  would read "spread").
- **Why it COMPLEMENTS the shipped span basis (the key insight).** Span (text)
  measures the *plumbing* side precisely — those handles are anchor-poor (NULL
  `anchor_vec`) but text-rich, so semantic ANN defaults them to neutral and
  MISSES them, while span nails them. Conversely a coined-unwritten concept
  (0 prose) is invisible to span (→ 1.0 default) but its anchor neighborhood is
  tight, so semantic NAILS it. The two gaps are disjoint: span covers the
  text-rich/anchor-poor plumbing; semantic covers the anchor-rich/text-poor
  coinage. A future scorer could take `min(span, semantic)` over the handles
  each can measure.
- **Coverage-fallback insight (preserve this).** When a handle's ANN
  neighborhood is degenerate (anchor-poor → only a self-match, or no anchor),
  fall the density back to **intra-thread member dispersion**: the mean cosine
  of the thread's member chunks to their centroid (tight = specific). The
  members are already in hand during `re_anchor`, so no extra query — and it
  rescues exactly the anchor-poor handles that ANN can't measure. This is the
  enhancement that would close the semantic basis's own coverage hole.
