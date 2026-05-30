# Rank-bench results (P5)

Harness: `crates/cli/examples/rank_bench.rs`. Fixtures:
`tests/fixtures/bench/{queries.json,lens_turns.jsonl}`. Engine weights are built
through the single `build_engine_from_weights` path (query crate), the same one
production recall uses — so a weight that is benched and a weight that ships can
never drift.

## How to run

```bash
# CI-safe synthetic fixture (no model download, deterministic):
HF_HUB_OFFLINE=1 cargo run -p ostk-recall --example rank_bench -- --fixture

# Real corpus (operator step — fills in the §Operator results below):
cargo run --release -p ostk-recall --example rank_bench -- \
  --corpus ~/.local/share/ostk-recall --model potion-retrieval-32M \
  --queries tests/fixtures/bench/queries.json --out docs/rank-bench-results.md
```

## Method

- **Corpus**: operator's own (~10K chunks across claude_code / code / ostk_project
  / markdown) for the real run; a small deterministic in-memory corpus for the
  fixture run.
- **Query set**: `queries.json`. Each query carries relevance markers under
  `relevant.{source_ids,chunk_ids}`. A hit is **relevant** iff its `source_id`
  is in `source_ids` OR its `chunk_id` is in `chunk_ids`. `source_id`/path is the
  durable identity (chunk_ids churn across re-scans), so the real set should
  prefer `source_ids`.
- **Metrics** (per query, then averaged):
  - **MRR@10** — `1/rank` of the first relevant hit. *Primary.*
  - **Success@10** (hit rate) — fraction of queries with ≥1 relevant in top-10.
    This is the **honest stand-in for "Recall@10"**: a 10K-corpus marker set
    can't enumerate every relevant chunk, so true recall has no denominator.
  - **Recall@10 vs labeled set** — relevant-in-top-10 / labeled-markers.
    Secondary; the denominator is the labeled markers, not the full relevant set.
  - **Per-source diversity** (distinct `source_id` in top-10) and
    **duplicate-source concentration** (max hits from one `source_id`).
- **Latency** (p50 / p95 / p99, µs): candidate-gen (lanes + union + RRF + fetch),
  rank (`RankEngine::rank`), post-rank (**derived** = e2e − candgen − rank;
  the rerank / identifier-boost / self-reference / diversify stages inside
  `hybrid::recall` are not separately instrumented), end-to-end.
- **Memory**: best-effort process peak RSS — Linux `/proc/self/status` `VmHWM`;
  `n/a` elsewhere (macOS). The meaningful figure is the operator's Linux run.
- **Sensitivity sweep**: each weight over `{0.0, 0.25, 0.5, 1.0, 2.0}`, others
  fixed; reports MRR per setting (monotonic improver vs noise).
- **Lens rotation**: replays `lens_turns.jsonl` through the ambient
  (attention-only) path and reports turn→turn repeated-chunk rate and rotation
  rate. This is the **P9b-min baseline** (no refractory); the P9b-full refractory
  penalty is expected to raise rotation — that comparison lands with P9b-full.

## Configurations

| # | Config | Status |
|---|---|---|
| 1 | BM25 + dense + RRF (no rerank) | run (`C1 rrf=1.0`) |
| 2 | (1) + cross-encoder rerank | **skipped in fixture** (model download); runnable in `--corpus` with a reranker wired |
| 3 | RankEngine rrf=1.0 only | == config 1 numerically (compiled explicit default) |
| 4 | RankEngine tuned `{rrf, bm25}` no rerank | run (`C4 rrf=1.0+bm25=0.5`) |
| 5 | (4) + attention (ambient) | ambient profile — see **Lens rotation** |
| 6 | (5) + multivector (P4) | **skipped: multivector unavailable** — P4 not started |
| 7 | (6) + rerank | **skipped** — depends on P4 |
| 8 | header-enriched vs raw dense | **deferred** to its own phase (needs an embed-header change + full corpus re-embed) |

## Fixture results (plumbing check — NOT a retrieval verdict)

> ⚠️ Synthetic in-memory corpus + `FakeEmbedder` (length-bucket single-hot).
> These prove the harness runs and every column populates. The retrieval verdict
> comes from the operator `--corpus` run below. Numbers from
> `--fixture --iters 5` (6 synthetic queries):

**Quality (explicit profile)**

| Config | MRR@10 | Success@10 | Recall@10* | avg distinct src | avg max dup |
|---|---|---|---|---|---|
| C1 rrf=1.0 (no rerank) | 0.764 | 1.000 | 0.833 | 4.83 | 1.00 |
| C4 rrf=1.0+bm25=0.5 (no rerank) | 0.764 | 1.000 | 0.833 | 4.83 | 1.00 |

**Latency (µs, p50/p95/p99)** — fixture machine, indicative only

| Config | candidate-gen | rank | post-rank† | end-to-end |
|---|---|---|---|---|
| C1 | 924 | 1056 | 5018 | 7159 |
| C4 | 1023 | 1304 | 5325 | 7723 |

**Sensitivity sweep (MRR@10 vs weight)**
- `bm25`: 0→0.764, 0.25→0.764, 0.5→0.764, 1→0.764, 2→0.764 (flat on this
  corpus — RRF already orders the small candidate pool; bm25's soft-sigmoid adds
  no separation here).
- `rrf`: 0→0.250, 0.25→0.764, 0.5→0.764, 1→0.764, 2→0.764 (**rrf is
  load-bearing** — zeroing it collapses ordering).

**Lens rotation (8 turns)**: repeated-chunk rate 0.286, rotation rate 0.714.

Peak RSS: n/a (fixture run on macOS).

## §Operator results (real corpus — 2026-05-30)

Run on the maintainer's live corpus. Release build, reranker **OFF** (the bench
measures the RRF + rank-engine path P5 tunes; the production reranker runs on top
of this and replaces the score — see the frame note in the gate decision).

- **Corpus**: `~/.local/share/ostk-recall/corpus.lance`, **119,965 chunks**,
  `minishlab/potion-retrieval-32M` (512-dim). Source mix: claude_code 55,274 ·
  code 54,462 · markdown 6,609 · ostk_spec 2,357 · ostk_decision 521 ·
  ostk_session 200 · ostk_audit_significant 259 · ostk_needle 95 · gemini 132 ·
  membrane 40 · ostk_memory 3.
- **Query set**: 15 queries (`tests/fixtures/bench/queries.corpus.json`),
  `source_id`-labeled from live recall. Categories: topical ×9, entity ×4,
  vague ×2. (`--iters` defaulted to 5 — a known harness arg-parse nit, see
  below; affects latency sample count only, not quality.)

**Quality (explicit profile, reranker off)**

| Config | MRR@10 | Success@10 | Recall@10* |
|---|---|---|---|
| C1 rrf=1.0 | 0.581 | 0.733 | 0.430 |
| C4 rrf=1.0 + bm25=0.5 | 0.581 | 0.733 | 0.430 |

Adding bm25 as an additive rank feature at 0.5 changes nothing vs rrf-only.

**Latency (µs, p50/p95/p99 — release, 120k chunks)**

| Config | candidate-gen | rank | post-rank† | end-to-end |
|---|---|---|---|---|
| C1 | 215,897 / 229,466 / 231,736 | 38 / 43 / 47 | 76,333 / 87,798 / 92,667 | 295,748 / 313,749 / 320,995 |
| C4 | 218,905 / 238,139 / 239,408 | 41 / 47 / 52 | 73,716 / 89,008 / 97,462 | 293,344 / 315,937 / 320,464 |

- **candidate-gen (Lance I/O) dominates** end-to-end (~216 ms of ~296 ms p50).
- **The rank engine is ~free** (38 µs p50). This empirically validates the P5
  frame: feature weights cost nothing; the latency budget is all retrieval I/O.
- post-rank (76 ms) is the derived remainder absorbing run-to-run Lance
  variance; with no reranker loaded the real in-memory stages are microseconds.

**Sensitivity sweep (MRR@10 vs weight)**

- **bm25**: 0→0.559, 0.25→**0.581**, 0.5→0.581, 1→0.559, 2→0.503. A tiny bump at
  0.25–0.5, then it *regresses* at weight ≥1 (the additive bm25 term starts
  out-voting rrf's fused ordering on some queries).
- **rrf**: 0→**0.000** (total collapse), 0.25→0.581, 0.5→0.581, 1→0.581,
  2→0.581. RRF is load-bearing and **saturating** — any weight ≥0.25 is
  identical; magnitude past that is irrelevant.

**Lens rotation (ambient, 8 turns)**: repeated-chunk rate 0.129, rotation rate
**0.871** — the attention-only ambient path rotates well on the real corpus.

### Gate decision (P5 → P9b-full): **NEUTRAL — ship framework, keep defaults**

Per `p5-bench.md`'s pre-registered criteria, a config must beat baseline by ≥10%
relative MRR@10 (or Recall@10 with no >5% regression elsewhere). **No config
clears the bar:**

- bm25 as an additive rank feature is at best **+0.0** (it's within noise at low
  weight) and **regresses** at weight ≥1 — not a 10% improvement under any
  setting.
- rrf saturates at ≥0.25; there's no weight magnitude that improves on the
  `rrf=1.0` default.

This is exactly the `p5-bench.md` "if nothing meets the gate" path:

- [x] **Neutral** → **keep the compiled defaults** (explicit `rrf=1.0`, ambient
      `attention_affinity=1.0`). No re-weighting is committed to `config.rs`. The
      `[ranking.weights]` config + `effective(profile)` resolver +
      `build_engine_from_weights` ship as the tunable framework; the substrate
      value (P6/P7/P7b/P8/P9) is independent of a retrieval-quality win.
- [ ] ~~Improvement found~~ — not observed.

**Frame caveat (load-bearing).** This measures the **reranker-off** rank path.
In production the cross-encoder reranker runs and **replaces** the engine score,
so these explicit-path rank weights only reshape the candidate pool fed to the
reranker — consistent with the ratified relevance/salience split. The rank-engine
weights are decisive in the **ambient/lens path** (no reranker), which is what
P9b-full consumes; the neutral explicit-path result does **not** argue against
the lens-path freshness/entity/concept slots P9b-full adds. A future bench pass
*with* the reranker wired into the example would measure the full explicit stack
(config 2); that's a harness extension, not a P5 blocker.

Header-format choice (`EMBED_HEADER_FORMAT_VERSION`) and the
`[ranking.multivector] enabled` default remain **out of scope** (header
experiment deferred; P4 not started).

### Known harness nits (follow-ups, non-blocking)

- `--iters N` is dropped (arg-parse advances the index twice for valued flags),
  so latency used the default 5 samples. Fix: don't double-increment in valued
  arms. Quality is unaffected.
- Peak RSS prints `n/a` on macOS (Linux-only `/proc` path). Expected.
- The reranker is not wired into the example (config 2 shows skipped). Wiring it
  would let the bench measure the full production explicit stack.
