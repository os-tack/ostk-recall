# ostk-recall

Local-first cross-corpus retrieval MCP server. One binary built on
[model2vec-rs](https://github.com/MinishLab/model2vec-rs) for static
embeddings, [LanceDB](https://lancedb.com/) (Arrow + Tantivy BM25) for the
vector + full-text store, and [DuckDB](https://duckdb.org/) for the ingest
manifest and audit event log.

## What it solves

Claude, Cursor, and ChatGPT each maintain their own siloed context — notes
you dumped into one app are invisible to the others, and every new session
starts cold. ostk-recall ingests your markdown trees, Claude Code session
logs, haystack `.ostk/` directories, or arbitrary globs into a single local
corpus. It then exposes a hybrid (dense + BM25) search tool over MCP so any
MCP-speaking client can recall context across projects without shipping
your data off-box.

## Status

Pre-alpha. Core plumbing is in.

Works today:

- Six source scanners (markdown, code, claude_code, file_glob, zip_export,
  ostk_project composite).
- Four MCP tools (`recall`, `recall_link`, `recall_stats`, `recall_audit`).
- Hybrid retrieval with RRF fusion over LanceDB's dense and Tantivy BM25
  indexes.
- Idempotent re-ingest via `chunk_id = sha256(source:source_id:chunk_index)`
  and LanceDB `merge_insert`.

Deferred:

- Tree-sitter aware code chunking (line-window fallback today).
- ChatGPT export scanner (zip layout differs from Claude exports).
- Live ingest / file-watcher (scan is one-shot today).
- Optional reranker pass (RRF only for now).

## Install

```
git clone https://github.com/os-tack/ostk-recall
cd ostk-recall
cargo install --path crates/cli
```

`lance-encoding` needs `protoc` at build time:

- macOS: `brew install protobuf`
- Debian / Ubuntu: `apt-get install protobuf-compiler`

## Configure

Default path: `${XDG_CONFIG_HOME:-~/.config}/ostk-recall/config.toml`.

Minimal example:

```toml
[corpus]
root = "~/.local/share/ostk-recall"

[embedder]
model = "potion-retrieval-32M"

[[sources]]
kind = "markdown"
project = "notes"
paths = ["~/notes"]
```

See [`config.example.toml`](./config.example.toml) for every source kind
and all tunables.

## Quickstart

```
ostk-recall init                # create corpus root, download model
ostk-recall scan                # ingest configured sources
ostk-recall verify              # reconcile counts across store + manifest
ostk-recall serve --stdio       # MCP server on stdio
```

## Source kinds

| kind           | what it ingests                                                  | chunking                                         |
| -------------- | ---------------------------------------------------------------- | ------------------------------------------------ |
| `markdown`     | `.md` / `.markdown` trees                                        | split on headings, soft-wrap at ~400 tokens      |
| `code`         | source files filtered by `extensions = [...]`                    | sliding line window (tree-sitter deferred)       |
| `claude_code`  | Claude Code project session logs (`<slug>/*.jsonl`)              | one chunk per user / assistant turn              |
| `file_glob`    | arbitrary glob, ingested as plain text                           | paragraph split, soft-wrap at ~400 tokens        |
| `zip_export`   | Claude.ai data-export `.zip` bundles                             | per-conversation-turn chunks                     |
| `ostk_project` | haystack `.ostk/` dirs — decisions, needles, audit, specs, code  | composite; one chunk per record or source chunk  |

## MCP tools exposed

| tool           | input schema (one-line)                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `recall`       | `{ query: string, project?: string, source?: string, since?: rfc3339, limit?: 1..100 }`                      |
| `recall_link`  | `{ chunk_id: string }` — returns the chunk plus its parent chain                                             |
| `recall_stats` | `{}` — returns total count, breakdown by source, model info, last-scan timestamp                             |
| `recall_audit` | `{ sql: string }` — raw SELECT over DuckDB `audit_events` (ostk_project sources only; single statement)      |

## Hook into clients

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ostk-recall": {
      "command": "ostk-recall",
      "args": ["serve", "--stdio"]
    }
  }
}
```

### Cursor

Edit `~/.cursor/mcp.json` (or project-local `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "ostk-recall": {
      "command": "ostk-recall",
      "args": ["serve", "--stdio"]
    }
  }
}
```

### Claude Code

Edit `.mcp.json` at user or project level:

```json
{
  "mcpServers": {
    "ostk-recall": {
      "type": "stdio",
      "command": "ostk-recall",
      "args": ["serve", "--stdio"]
    }
  }
}
```

### haystack (llmOS)

Add one line to your HUMANFILE (`~/.ostk/HUMANFILE` or project-local):

```
DRIVER mine fcp ostk-recall serve --stdio
```

haystack's `ostk _relay` already wraps any stdio MCP subprocess into a Unix
socket — no kernel change needed. The `mine` verb shows up on next boot.

## Embedder options

| model                  | dim | notes                              |
| ---------------------- | --- | ---------------------------------- |
| `potion-retrieval-32M` | 512 | default; best recall               |
| `potion-base-8M`       | 256 | lighter, faster, lower ceiling     |

Dimension is locked into the LanceDB schema as `FixedSizeList<Float32, DIM>`.
Switching models requires `ostk-recall init --force` followed by a full
re-scan. This is an Arrow schema constraint, not a regression.

## Architecture

```
  sources ──► scanners ──► pipeline ──► store ──► query ──► MCP
   (fs)       (.rs x6)    (embed +     (LanceDB   (hybrid   (stdio
                          chunk +       + DuckDB)  + RRF)    server)
                          merge)
```

Embedder (model2vec-rs) ▶ Store (LanceDB vector + Tantivy BM25; DuckDB
manifest + audit) ▶ Pipeline (scan ▶ chunk ▶ embed ▶ `merge_insert`) ▶
Query (dense + BM25 with RRF fusion) ▶ MCP (stdio server, four tools).

See [`docs/architecture.md`](./docs/architecture.md) for the full writeup.

## Roadmap

- Tree-sitter aware code chunking.
- ChatGPT export scanner.
- Live file-watcher ingest loop.
- Optional cross-encoder reranker stage after RRF.
- Alternate embedder backends (ONNX, Candle).

## Development

```
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
OSTK_RECALL_E2E=1 cargo test --workspace    # network-enabled tests (model download)
```

## License

Apache-2.0 — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
