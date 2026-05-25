# Architecture

ostk-recall is a single-process, local-first memory substrate. It ingests
content from heterogeneous sources, embeds and indexes it into a unified
store, observes a live conversational stream, and serves hybrid (dense +
BM25) queries plus a live thread/attention runtime over the Model Context
Protocol (MCP).

Two organs share one binary: **recall** (corpus retrieval, query-shaped)
and **attention** (live thread/scope runtime, process-shaped). They share
the Lance store and the MCP server; the v0.4.x roadmap composes them.

## Shape at a glance

```
  sources ──▶ scanners ──▶ pipeline ──▶ store ──┬─▶ query ──▶ MCP server
   (fs)     (8 kinds +    chunk+embed   │        │  hybrid    (stdio,
              membrane    + merge_insert │        │  RRF       18 tools)
              synthetic)                 │        │  + rerank
              ┌─────────────────────────┘        │
              ▼                                  │
         ┌─────────────────┐   ┌────────────┐    │
         │  corpus.lance   │   │  manifest  │    │
         │  + Tantivy BM25 │   │  + audit   │    │
         │  (chunks + vec) │   │  (SQLite)  │    │
         └────────┬────────┘   └────────────┘    │
                  │                              │
                  ├──── threads.sqlite ──────────┤
                  │   (threads, evidence_links, │
                  │    threads_proposed,        │
                  │    chain_events ledger)     │
                  │                              │
   pipeline ─────▶│      IngestEvent            │
   broadcast      │      (subscribe_ingest)     │
                  │                              │
                  ▼                              │
           ┌──────────────────────────────┐     │
           │   live attention runtime     │     │
           │                              │     │
           │  • TurnObserver (per turn)   │     │
           │    ↳ membrane chunks,        │     │
           │      familiarity ticks,      │     │
           │      proposed stubs          │     │
           │                              │     │
           │  • AutoWeaver (per ingest)   │     │
           │    ↳ anchor cosine →         │     │
           │      derived evidence_links  │     │
           │                              │     │
           │  • IdleCurator (timer)       │     │
           │    ↳ fade-score every thread │     │
           │      → tension transitions   │     │
           │                              │     │
           │  • InMemoryAttention         │     │
           │    (score tier, rebuilt on   │     │
           │     boot via chain replay)   │     │
           └──────────────┬───────────────┘     │
                          │                      │
                          └──────────────────────┴───▶ attention tools
                                                       (attention_*, thread_*)
```

Read and write daemons share `corpus.lance` via Lance MVCC.

## Layered overview

- **Core** (`crates/core`) — wire types shared across crates:
  `ChunkRecord`, `Source`, `SynthesizedPage`, and the attention wire
  types (`AttentionScope`, `AttentionPage`, `ThreadHandle`,
  `ScoreAttribution`, `FoldDepth`).

- **Embedder** (`crates/embed`) — wraps `model2vec-rs` for static
  embeddings. Default `potion-retrieval-32M` (512-dim), fallback
  `potion-base-8M` (256-dim). A fake embedder is available behind a
  feature flag for unit and CI tests.

- **Store** (`crates/store`) — owns the persistent state:
  - `corpus.lance` (Lance + Tantivy BM25) — chunks with dense vectors
    and a full-text index side by side. Dimension locked at table
    creation as `FixedSizeList<Float32, DIM>`.
  - **SQLite manifest** — one row per known `source:source_id` with
    content hashes; idempotency anchor for re-scans.
  - **SQLite `audit_events`** — ingested from haystack `ostk_project`
    sources; queryable via `recall_audit`.
  - **`threads.sqlite`** — durable thread identity: `threads`,
    `evidence_links`, `threads_proposed`, and the `chain_events`
    ledger that lets `InMemoryAttention` rebuild on boot.

- **Scanners** (`crates/scan`) — one module per source kind (seven
  today). Each scanner yields `ChunkRecord`s; scanners do no embedding
  or storage. Includes a `threads` scanner that turns `.ostk/threads/`
  files into corpus chunks.

- **Pipeline** (`crates/pipeline`) — orchestrates `scan ▶ chunk ▶ embed
  ▶ merge_insert`. Batching, the `subscribe_ingest` broadcast channel
  (which the auto-weaver and observer listen on), and the synthetic
  ingest path used by `TurnObserver`'s membrane chunks all live here.

- **Query** (`crates/query`) — executes hybrid retrieval: a dense kNN
  search and a BM25 search, both parameterised by the same filters
  (`project`, `source`, `since`). Results are fused with reciprocal rank
  fusion (RRF) and optionally re-ordered by a fastembed-rs cross-encoder
  rerank pass. The `Synthesizer::collapse` path turns hits into
  `SynthesizedPage`s for `recall_fault`.

- **Attention** (`crates/attention`) — the live runtime. Owns
  `InMemoryAttention` (the score tier, in-memory, rebuilt on boot via
  chain replay) and three daemons:
  - **TurnObserver** — per conversational turn; emits membrane
    chunks, familiarity ticks, and proposed thread stubs.
  - **AutoWeaver** — subscribes to `Pipeline::subscribe_ingest`;
    cosine-matches new chunks against thread anchor vectors and
    writes `Derived` evidence links.
  - **IdleCurator** — timer-driven; recomputes each thread's fade
    score, transitions across tension thresholds with hysteresis.

  Three time-scales: chain (durable, append-only), graph
  (`threads.sqlite`, persistent), score (`InMemoryAttention`,
  ephemeral). The attention runtime never writes to the chain
  directly — that's the store's job — but every state nudge does.

- **Attention MCP** (`crates/attention-mcp`) — JSON-RPC dispatch for
  the 13 attention/thread tools. Mounted into the same MCP server as
  the recall tools (see below).

- **MCP** (`crates/mcp`) — the stdio server implementing MCP 2025-06-18.
  Exposes 18 tools total: 5 `recall_*` from this crate
  (`recall`, `recall_link`, `recall_stats`, `recall_audit`,
  `recall_fault`) plus the 13 attention/thread tools from
  `attention-mcp`. `recall_audit` is only registered when at least
  one configured source is `ostk_project`.

- **CLI** (`crates/cli`) — the `ostk-recall` binary. Subcommands:
  `init`, `scan`, `verify`, `serve` (read-only `--stdio` driver mode
  or read-write standalone with a scan-trigger socket), `watch`
  (FSEvents → debounce → trigger socket).

## Scanner contract + chunking rules

Every scanner yields a deterministic stream of chunks for a given input:

```rust
struct ChunkRecord {
    chunk_id: String,      // sha256(source:source_id:chunk_index), hex
    source: String,        // configured project + kind
    source_id: String,     // stable per-source identifier (path, turn, …)
    chunk_index: u32,      // 0-based within source_id
    text: String,
    metadata: serde_json::Value,
}
```

Chunking rules per kind:

| kind           | chunking strategy                                                    |
| -------------- | -------------------------------------------------------------------- |
| `markdown`     | split on ATX / setext headings, soft-wrap long sections at ~400 tok  |
| `code`         | sliding line window; `fcp-rust` provides symbol-bounded chunks for `.rs` |
| `claude_code`  | one chunk per user / assistant turn in the `.jsonl` log              |
| `gemini`       | one chunk per user/gemini exchange pair (`session-*.json`)           |
| `file_glob`    | paragraph split, soft-wrap at ~400 tokens                            |
| `zip_export`   | per-conversation-turn chunks extracted from the zip bundle           |
| `ostk_project` | composite; decisions, needles, audit rows, specs, and code surface   |
| `threads`      | one chunk per `.ostk/threads/*.md` file (tension state in metadata)  |

## Idempotency guarantee

`chunk_id` is a stable function of the triple `source:source_id:chunk_index`:

```
chunk_id = sha256(format!("{source}:{source_id}:{chunk_index}"))
```

The pipeline writes to LanceDB via `merge_insert(["chunk_id"])`. Re-scans
are idempotent: unchanged chunks update in place, truly new chunks append,
and content-hash checks short-circuit embedding for unchanged bodies. No
duplicate rows, no drift.

## FTS + dense hybrid

LanceDB keeps the vector column and the Tantivy-backed text index in the
same table. A single query fans out into two searches:

1. Dense kNN over the `FixedSizeList<Float32, DIM>` vector column.
2. BM25 over the Tantivy FTS index on the `text` column.

Both searches apply the same filter predicate (`project`, `source`,
`since`). Their ranked lists are fused with reciprocal rank fusion:

```
score(d) = Σ  1 / (k + rank_i(d))
          i∈{dense, bm25}
```

with `k = 60` (Cormack et al. default). The fused top-`limit` rows are
returned to the caller.

## Dimension lock

LanceDB stores the embedding column as `FixedSizeList<Float32, DIM>`. The
Arrow schema is fixed at table creation time and cannot be mutated to a
different dimension. Switching from `potion-retrieval-32M` (512) to
`potion-base-8M` (256), or vice versa, therefore requires:

1. `ostk-recall init --force` to recreate the table.
2. A full re-scan of every configured source.

This is an Arrow schema constraint, not an ostk-recall regression.

## haystack integration

**haystack v6.0.0+** ships with `fcp-recall` baked into the driver
defaults — `mem.fault_recall` routes through `ostk-recall serve --stdio`
automatically. Have `ostk-recall` on `$PATH`; no HUMANFILE entry needed.

For pre-v6 haystack or other ostk-shaped projects, register manually
via HUMANFILE:

```
DRIVER mine fcp ostk-recall serve --stdio
```

haystack's `ostk _relay` wraps any stdio MCP subprocess into a Unix
socket — no kernel change needed. The verb shows up on next boot and
routes through the same attenuation, audit, and timeout machinery as
every other driver.

See [`docs/spec/driver-protocol.md`](spec/driver-protocol.md) for the
full MCP tool surface (18 tools across the recall and attention
families).
