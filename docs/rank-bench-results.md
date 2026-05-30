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

## §Operator results (real corpus — PENDING)

> This section is the seam for the maintainer's `--corpus` run. Replace the
> placeholders; do not ship fabricated numbers.

- Corpus mix (counts by source): _pending_
- Query set size + categories: _pending_
- Quality table (configs 1–5 + 8): _pending_
- Latency + memory (Linux): _pending_
- Sensitivity sweep per weight: _pending_
- Qualitative spot-check (top-3 for 5 representative queries, 1–5): _pending_

### Gate decision (P5 → P9b-full) — PENDING

Per `p5-bench.md`: a config beats baseline (config 2) if ≥10% relative MRR@10
improvement, OR ≥10% relative Recall@10 with no >5% regression on the other, AND
no qualitative regression.

- [ ] **Improvement found** → commit tuned defaults to
      `crates/core/src/config.rs` `[ranking.weights.{explicit,ambient}]` (the
      `effective(profile)` resolver and `build_engine_from_weights` are already
      wired; only the default values change).
- [ ] **Neutral** → keep the compiled defaults (explicit `rrf=1.0`, ambient
      `attention_affinity=1.0`); document neutral retrieval, ship the framework.
      The substrate value (P6/P7/P7b/P8/P9) is independent of a retrieval-quality
      win.

Header-format choice (`EMBED_HEADER_FORMAT_VERSION`) and the
`[ranking.multivector] enabled` default are **out of scope for this phase** (the
header experiment is deferred; P4 multivector is not started).
