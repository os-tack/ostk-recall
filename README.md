# ostk-recall

Local-first cross-corpus retrieval MCP server. One binary built on
[model2vec-rs](https://github.com/MinishLab/model2vec-rs) for static
embeddings, [LanceDB](https://lancedb.com/) (Arrow + Tantivy BM25) for the
vector + full-text store, [SQLite](https://www.sqlite.org/) (via rusqlite)
for the ingest manifest and audit event log, and [fastembed-rs](https://github.com/Anush008/fastembed-rs)
for the optional cross-encoder reranker.

## What it solves

Claude, Cursor, and ChatGPT each maintain their own siloed context — notes
you dumped into one app are invisible to the others, and every new session
starts cold. ostk-recall ingests your markdown trees, Claude Code session
logs, haystack `.ostk/` directories, or arbitrary globs into a single local
corpus. It then exposes a hybrid (dense + BM25) search tool over MCP so any
MCP-speaking client can recall context across projects without shipping
your data off-box.

## Status

Pre-alpha but functional. Used in production by the maintainer's own stack.

Works today:

- Seven source scanners (markdown, code, claude_code, file_glob, zip_export,
  gemini, ostk_project composite).
- Five MCP tools — `recall`, `recall_link`, `recall_stats`, `recall_audit`,
  and `recall_fault` (synthesizes hits into virtual-memory pages for
  haystack's [`mem.fault_recall`](https://github.com/os-tack/haystack)
  driver-relay path).
- Hybrid retrieval with RRF fusion over LanceDB's dense and Tantivy BM25
  indexes, plus an optional cross-encoder rerank pass (fastembed-rs;
  default `jina-reranker-v1-turbo-en`).
- Idempotent re-ingest via `chunk_id = sha256(source:source_id:chunk_index)`
  and LanceDB `merge_insert`.
- File-watcher (`ostk-recall watch`) that pokes a running `serve` whenever
  edits land under any configured source path. Path-aware incremental
  ingest: ~250 ms from save to corpus, opt-in via `[watch].mode = "incremental"`
  in config (default `legacy` runs a full re-scan per kick for safe
  rollout).
- Two daemon modes that share `corpus.lance` via Lance MVCC: read-only
  driver mode (`ostk-recall serve --stdio`, kernel-managed by haystack)
  and read-write standalone mode (`ostk-recall serve`, operator-managed
  with the scan-trigger socket the watcher pokes).

Deferred:

- Tree-sitter aware code chunking (line-window fallback today).
- ChatGPT export scanner (zip layout differs from Claude exports).
- Per-file offset cursors for incremental scan of `claude_code` and
  `gemini` (append-only JSONL falls back to full-source scan when poked
  via the watcher; everything else gets the per-path speedup).

## Install

### From a release tarball

Pre-built binaries for Linux x86_64, macOS arm64, and Windows x86_64 ship
with every tagged release:
<https://github.com/os-tack/ostk-recall/releases>. Untar (or unzip on
Windows), drop the binary on `$PATH`, done.

### From source

```
git clone https://github.com/os-tack/ostk-recall
cd ostk-recall
make install     # → ~/.cargo/bin/ostk-recall
```

`make install` is a thin wrapper around `cargo install --path crates/cli
--locked --force`. `make help` lists every target (build, scan, serve,
verify, lint, test, …); `make version` shows what's currently installed
plus the workspace's git sha.

Native deps (the cargo build pulls these in transitively, but you need
the system tools at compile time):

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
ostk-recall serve --stdio       # MCP server on stdio (driver mode, RO)
ostk-recall serve               # standalone w/ scan-trigger socket (RW)
ostk-recall watch               # file-watcher → poke standalone serve
```

### Live updates

`serve` (no `--stdio`) binds a Unix socket at `corpus.root/recall.sock`
that accepts scan-trigger pokes. Run `watch` next to it and edits under
any configured source path get debounced and re-ingested without
manual re-scan. Add to your config:

```toml
[watch]
enabled = true
# debounce_ms defaults: 800 ms (Linux), 1200 ms (Windows), 1500 ms (macOS).
# projects = ["notes", "docs"]   # optional allowlist; defaults to all sources.
mode = "incremental"             # opt-in path-aware ingest (~250 ms save → corpus).
                                 # default "legacy" runs a full re-scan per kick.
```

The Makefile wraps the same loop:

```
make install                    # build + install the binary
make scan                       # incremental ingest (uses installed binary)
make scan-source SOURCE=<name>  # only one project from config
make reingest    SOURCE=<name>  # wipe + rescan one project (chunker change, etc.)
make rebuild                    # nuke corpus + manifest, full re-init + scan (prompts)
make verify
make serve
```

## Source kinds

| kind           | what it ingests                                                  | chunking                                         |
| -------------- | ---------------------------------------------------------------- | ------------------------------------------------ |
| `markdown`     | `.md` / `.markdown` trees                                        | split on headings, soft-wrap at ~400 tokens      |
| `code`         | source files filtered by `extensions = [...]`                    | sliding line window (tree-sitter deferred)       |
| `claude_code`  | Claude Code project session logs (`<slug>/*.jsonl`)              | one chunk per user / assistant turn              |
| `gemini`       | Gemini CLI session JSON (`session-*.json`, walks recursively)    | one chunk per user/gemini exchange pair          |
| `file_glob`    | arbitrary glob, ingested as plain text                           | paragraph split, soft-wrap at ~400 tokens        |
| `zip_export`   | Claude.ai data-export `.zip` bundles                             | per-conversation-turn chunks                     |
| `ostk_project` | haystack `.ostk/` dirs — decisions, needles, audit, specs, code  | composite; one chunk per record or source chunk  |

## MCP tools exposed

| tool           | input schema (one-line)                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `recall`       | `{ query: string, project?: string, source?: string, since?: rfc3339, limit?: 1..100 }`                      |
| `recall_link`  | `{ chunk_id: string }` — returns the chunk plus its parent chain                                             |
| `recall_stats` | `{}` — returns total count, breakdown by source, model info, last-scan timestamp                             |
| `recall_fault` | `{ query: string, intent?: "symbol"\|"narrative"\|"trace"\|"general", limit?: int, max_per_source_id?: int }` — synthesizes hits into named virtual-memory pages; haystack's `mem.fault_recall` calls this |
| `recall_audit` | `{ sql: string }` — raw SELECT over the SQLite `audit_events` table (ostk_project sources only; single statement) |

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

**haystack v6.0.0+** ships with `fcp-recall` baked into the driver
defaults — `mem.fault_recall` and the `recall` family route through
`ostk-recall serve --stdio` automatically. Just have `ostk-recall` on
`$PATH`. No HUMANFILE entry needed.

For pre-v6 haystack or other ostk-shaped projects, register manually
via HUMANFILE (`~/.ostk/HUMANFILE` or project-local):

```
DRIVER mine fcp ostk-recall serve --stdio
```

haystack's driver-relay wraps any stdio MCP subprocess into a Unix
socket — no kernel change needed. The verb shows up on next boot.

## Embedder options

| model                  | dim | notes                              |
| ---------------------- | --- | ---------------------------------- |
| `potion-retrieval-32M` | 512 | default; best recall               |
| `potion-base-8M`       | 256 | lighter, faster, lower ceiling     |

Dimension is locked into the LanceDB schema as `FixedSizeList<Float32, DIM>`.
Switching models requires `ostk-recall init --force` followed by a full
re-scan. This is an Arrow schema constraint, not a regression.

## Reranker options

After hybrid (dense + BM25 + RRF) returns a top-N candidate pool the
reranker scores each (query, doc) pair jointly and re-orders. ~150 ms per
query on CPU, ~80 MB to the model cache. Opt out with
`[reranker] enabled = false`.

| model id                                | notes                                  |
| --------------------------------------- | -------------------------------------- |
| `jina-reranker-v1-turbo-en`             | default; English, fastest              |
| `jina-reranker-v2-base-multilingual`    | multilingual                           |
| `bge-reranker-base`                     | BAAI baseline                          |
| `bge-reranker-v2-m3`                    | BAAI multi-task                        |

## Architecture

```
                                           ┌─► MCP (stdio, RO)  ──► clients
  sources ──► scanners ──► pipeline ──► store
   (fs)       (.rs x7)    (embed +     (LanceDB   ▲
              ▲           chunk +      + SQLite)  │
              │           merge)                  │
              │                                   │
              └─── scan_paths ◄─── trigger.sock ◄─┘
                   (per-path)      (line-delim    │
                                    UTF-8)        │
                                                  │
              fs events ──► watch ────────────────┘
              (debounced)
```

**Read path** (kernel agents → corpus): `serve --stdio` runs as the
`fcp-recall` driver under haystack; `recall_fault` MCP tool synthesizes
hits into virtual-memory pages.

**Write path** (operator edits → corpus): `serve` (no `--stdio`) binds
`recall.sock`; `watch` debounces filesystem events and pokes the socket
with the changed paths; `Pipeline::scan_paths` does per-path ingest.
Read and write daemons share `corpus.lance` via Lance MVCC.

Stack: model2vec-rs (embedder) ▶ LanceDB vector + Tantivy BM25 (store)
+ SQLite (manifest + audit) ▶ pipeline (scan ▶ chunk ▶ embed ▶
`merge_insert`) ▶ query (dense + BM25 with RRF fusion, then optional
fastembed-rs cross-encoder rerank) ▶ MCP server (five tools).

See [`docs/architecture.md`](./docs/architecture.md) and
[`docs/spec/driver-protocol.md`](./docs/spec/driver-protocol.md) for the
full writeups.

## Roadmap

- Symbol-aware chunking for non-Rust source ([#11](https://github.com/os-tack/ostk-recall/issues/11)). Rust is already done — `code` scanner shells out to `fcp-rust` (rust-analyzer) for symbol-bounded chunks. Python / Go / JS are next; tree-sitter is the likely vehicle.
- ChatGPT export scanner ([#12](https://github.com/os-tack/ostk-recall/issues/12)). Today's `zip_export` only handles Claude.ai exports; ChatGPT's zip layout differs.
- Per-file offset cursors so `claude_code` and `gemini` benefit from path-aware incremental scan ([#13](https://github.com/os-tack/ostk-recall/issues/13)). Speed item, not correctness — content-addressed chunk_ids mean today's full re-parse is idempotent, just wasteful.
- MCP transport diversity — HTTP / SSE per the 2025-06-18 spec ([#14](https://github.com/os-tack/ostk-recall/issues/14)). stdio-only today.
- Retrieval feedback loop — query log + citation tracking, future personal reranker ([#15](https://github.com/os-tack/ostk-recall/issues/15)).
- Flip `[watch].mode` default from `legacy` to `incremental` after field bake-in.
- Alternate embedder backends (ONNX, Candle).

## Development

```
make check                                   # fmt-check + clippy + tests (CI parity)
make test                                    # tests only
make lint-strict                             # clippy with -D warnings
OSTK_RECALL_E2E=1 cargo test --workspace     # network-enabled tests (model download)
```

## License

Dual-licensed under either:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE) (with [`NOTICE`](NOTICE))

at your option. Contributions are accepted under the same terms (Apache-2.0 §5).
