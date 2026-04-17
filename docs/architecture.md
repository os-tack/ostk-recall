# Architecture

ostk-recall is a single-process, local-first retrieval server. It ingests
content from heterogeneous sources, embeds and indexes it into a unified
store, and serves hybrid (dense + BM25) queries over the Model Context
Protocol (MCP).

## Shape at a glance

```
  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
  │  sources    │    │  scanners    │    │  pipeline    │
  │  (fs, zips, │──▶ │  (6 kinds)   │──▶ │  chunk +     │
  │   .ostk)    │    │              │    │  embed +     │
  └─────────────┘    └──────────────┘    │  merge_insert│
                                          └──────┬───────┘
                                                 │
                     ┌───────────────────────────┴───────────────┐
                     ▼                                           ▼
              ┌──────────────┐                          ┌────────────────┐
              │   LanceDB    │                          │    DuckDB      │
              │  dense vec   │                          │  manifest +    │
              │  + Tantivy   │                          │  audit_events  │
              │  BM25        │                          │                │
              └──────┬───────┘                          └────────┬───────┘
                     │                                           │
                     └──────────────┬────────────────────────────┘
                                    ▼
                           ┌──────────────────┐
                           │     query        │
                           │  dense + BM25    │
                           │  RRF fusion      │
                           └────────┬─────────┘
                                    ▼
                           ┌──────────────────┐
                           │   MCP server     │
                           │   (stdio, 4 tools)│
                           └──────────────────┘
```

## Layered overview

- **Embedder** (`crates/embed`) — wraps `model2vec-rs` for static
  embeddings. Default `potion-retrieval-32M` (512-dim), fallback
  `potion-base-8M` (256-dim). A fake embedder is available behind a
  feature flag for unit and CI tests.

- **Store** (`crates/store`) — owns the LanceDB table and the DuckDB
  database. The LanceDB table carries dense vectors and the Tantivy-backed
  full-text index side by side. DuckDB holds two logical tables: the
  ingest manifest (one row per known `source:source_id`, with content
  hashes) and `audit_events` (ingested from haystack `ostk_project`
  sources).

- **Scanners** (`crates/scan`) — one module per source kind. Each scanner
  implements a contract that yields `ChunkRecord { chunk_id, source,
  source_id, chunk_index, text, metadata }`. Scanners do no embedding or
  storage — that is the pipeline's job.

- **Pipeline** (`crates/pipeline`) — orchestrates `scan ▶ chunk ▶ embed ▶
  merge_insert`. Batching is handled here; sources never see the embedder
  or the store directly.

- **Query** (`crates/query`) — executes hybrid retrieval: a dense kNN
  search and a BM25 search, both parameterised by the same filters
  (`project`, `source`, `since`). Results are fused with reciprocal rank
  fusion (RRF).

- **MCP** (`crates/mcp`) — a stdio server implementing the Model Context
  Protocol. Exposes four tools: `recall`, `recall_link`, `recall_stats`,
  and `recall_audit`. `recall_audit` is only registered when at least one
  configured source is `ostk_project` (DuckDB `audit_events` is
  otherwise empty).

- **CLI** (`crates/cli`) — the `ostk-recall` binary. Subcommands: `init`,
  `scan`, `verify`, `serve`.

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
| `code`         | sliding line window (tree-sitter deferred — see roadmap)             |
| `claude_code`  | one chunk per user / assistant turn in the `.jsonl` log              |
| `file_glob`    | paragraph split, soft-wrap at ~400 tokens                            |
| `zip_export`   | per-conversation-turn chunks extracted from the zip bundle           |
| `ostk_project` | composite; decisions, needles, audit rows, specs, and code surface   |

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

haystack's `ostk _relay` already knows how to wrap an arbitrary stdio MCP
subprocess into a Unix socket. Integration is a single HUMANFILE line:

```
DRIVER mine fcp ostk-recall serve --stdio
```

No kernel change, no new relay plumbing. The `mine` verb appears on next
boot and routes through the same attenuation, audit, and timeout
machinery as every other driver.
