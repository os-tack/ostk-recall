# ostk-recall

Local-first cross-corpus memory substrate. One binary built on
[model2vec-rs](https://github.com/MinishLab/model2vec-rs) for static
embeddings, [LanceDB](https://lancedb.com/) (Arrow + Tantivy BM25) for the
vector + full-text store, [SQLite](https://www.sqlite.org/) (via rusqlite)
for the ingest manifest, the audit event log, and the threads ledger, and
[fastembed-rs](https://github.com/Anush008/fastembed-rs) for the optional
cross-encoder reranker.

Two organs share one binary: **recall** (corpus retrieval, query-shaped)
and **attention** (live thread/scope runtime, process-shaped). 24 MCP
tools total — see below.

## What it solves

Claude, Cursor, and ChatGPT each maintain their own siloed context — notes
you dumped into one app are invisible to the others, and every new session
starts cold. ostk-recall ingests your markdown trees, Claude Code session
logs, ostk `.ostk/` directories, or arbitrary globs into a single local
corpus, then exposes a hybrid (dense + BM25) recall surface plus a live
thread/attention surface over MCP so any MCP-speaking client can recall
context across projects without shipping your data off-box.

A turn observer running in the same process watches the conversational
stream, proposes thread stubs from recurring noun phrases, and lets
high-confidence ones auto-promote. An auto-weaver matches new chunks
against thread anchor vectors and writes derived evidence links. An
idle curator fades inactive threads through tension thresholds. The
substrate stays mechanical — surfaces report axes, the caller decides
what's interesting.

## Status

Pre-alpha but functional. Used in production by the maintainer's own stack.

Works today:

- Eight source scanners (markdown, code, claude_code, gemini, file_glob,
  zip_export, ostk_project composite, thread) plus a synthetic `membrane`
  kind for in-process observer chunks.
- **24 MCP tools** across two families:
  - **Recall** (5): `recall`, `recall_link`, `recall_stats`,
    `recall_audit`, `recall_fault` (synthesizes hits into virtual-memory
    pages for ostk's
    [`mem_fault_recall`](https://github.com/os-tack/ostk.ai)
    driver-relay path).
  - **Attention/threads** (19): nine `attention_*` verbs (attend,
    surface, fold, familiarize, decay, focus, refocus, unfocus, status)
    and ten `thread_*` verbs (create, link, unlink, promote, list,
    emergent, attention, novelty, query, evidence).
- Hybrid retrieval with RRF fusion over LanceDB's dense and Tantivy BM25
  indexes, plus an optional cross-encoder rerank pass (fastembed-rs;
  default `jina-reranker-v1-turbo-en`).
- Live attention runtime: TurnObserver + AutoWeaver fire on watched
  conversation-transcript **TurnEnds** (the v0.6 gate — bulk scans no
  longer replay as live cognition); IdleCurator on a timer.
  `InMemoryAttention` score tier rebuilt on boot via chain replay from
  `threads.sqlite`.
- **Ambient memory lens**: `serve` maintains the `ostk://memory-lens` MCP
  *resource* — a query-less portfolio of corpus excerpts aligned to the
  current attention vector and refreshed as focus drifts, so subscribing
  clients get proactive context without asking. Bulk-scanned content is
  woven into the thread graph by `ostk-recall weave`. Tune via `[lens]`;
  `ostk-recall lens show` / `lens disable` to inspect or stop the loop.
- Idempotent re-ingest via `chunk_id = sha256(source:source_id:chunk_index)`
  and LanceDB `merge_insert`.
- File-watcher that pokes a running daemon whenever edits land under any
  configured source path. Run it in-process with `ostk-recall serve
  --watch`, or as a separate `ostk-recall watch` process. Path-aware
  incremental ingest: ~250 ms from save to corpus, opt-in via
  `[watch].mode = "incremental"` in config (default `legacy` runs a full
  re-scan per kick for safe rollout).
- One daemon, many clients. `ostk-recall serve` (no `--stdio`) runs as a
  standalone daemon that owns the corpus: it serves MCP to any number of
  clients over a local socket / named pipe, keeps the corpus fresh
  (scan-trigger socket + optional in-process `--watch`), and runs the
  attention loop. A `.serve.lock` flock makes it the single writer per
  corpus. Stdio clients reach it through the thin `ostk-recall connect`
  bridge; `ostk-recall serve --stdio` remains as the direct in-process
  transport (this process is itself the MCP server) for a client that
  prefers to spawn its own.

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
ostk-recall weave               # weave bulk-scanned content into the thread graph
ostk-recall weave --consolidate # coarse consolidation cycle (bridge/promote/merge/fade)
ostk-recall verify              # reconcile counts across store + manifest
ostk-recall optimize            # compact + fold indexes (--aggressive to collapse old versions)
ostk-recall serve               # standalone daemon: MCP over socket/pipe + ambient lens
ostk-recall serve --watch       # daemon + in-process file-watcher (keep corpus fresh)
ostk-recall serve --stdio       # direct stdio MCP transport (this process is the server)
ostk-recall connect             # bridge a stdio MCP client to a running daemon
ostk-recall watch               # standalone file-watcher → poke a running daemon
ostk-recall lens show           # print the current ambient memory lens
```

### Live updates

The daemon binds a scan-trigger socket at `corpus.root/recall.sock`
(named pipe `\\.\pipe\ostk-recall-recall` on Windows) that accepts
pokes. The simplest setup is `ostk-recall serve --watch`, which runs
the watcher in-process and skips the socket round-trip; alternatively
run a separate `ostk-recall watch` next to the daemon. Either way,
edits under any configured source path get debounced and re-ingested
without a manual re-scan. Add to your config:

```toml
[watch]
enabled = true
# debounce_ms defaults: 800 ms (Linux), 1200 ms (Windows), 1500 ms (macOS).
# projects = ["notes", "docs"]   # optional allowlist; defaults to all sources.
mode = "incremental"             # opt-in path-aware ingest (~250 ms save → corpus).
                                 # default "legacy" runs a full re-scan per kick.
```

### Consolidation cadence (scheduling)

`serve` owns *live* cognition (a watched transcript turn → observer +
weaver). Offline **consolidation** is separate and operator-scheduled — it
must not run inside `serve` or compete with lens responsiveness. The
maintenance shape is explicit and one-directional; nothing is implicitly
coupled:

```
ostk-recall scan                          # ingest (prints a `weave` hint on new chunks)
ostk-recall weave --since 24h             # capture: bind recent arrivals to anchors
ostk-recall weave --consolidate --since 1w  # coarse cycle (see below)
ostk-recall optimize                      # compact fragments when you want it
```

`weave --consolidate` runs the full cycle over its `--since` window:
deep re-weave → anchor↔anchor bridge → near-duplicate thread merge →
promote recurring high-cohesion proposals → fold deeply-familiar stable
threads → idle-fade. The `--since` value *is* the consolidation horizon;
pick it per cron tier. A sketch (`p11-temporal-consolidation.md` horizons):

```cron
# crontab — offline consolidation tiers (single-writer; runs beside serve via WAL,
# or stop serve first for a clean measurement window).
17 * * * *   ostk-recall weave --since 90m                       # hourly capture
37 */6 * * * ostk-recall weave --consolidate --since 24h         # 6h: cohere day-threads
53 4 * * *   ostk-recall weave --consolidate --since 7d          # daily: 1-week consolidate
23 5 * * 0   ostk-recall weave --consolidate --since 30d && ostk-recall optimize  # weekly: month + compact
```

On macOS, wrap the same commands in a `launchd` `StartCalendarInterval`
agent (`~/Library/LaunchAgents/ai.ostk.recall.consolidate.plist`); on
Linux a systemd timer. The scheduler is policy — the CLI is the only
contract. Prefix offline runs with `HF_HUB_OFFLINE=1` once the model is
cached.

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
| `ostk_project` | ostk `.ostk/` dirs — decisions, needles, audit, specs, code  | composite; one chunk per record or source chunk  |
| `thread`       | `.ostk/threads/*.md` files; tension state captured as metadata    | one chunk per thread file                        |

## MCP tools exposed

24 tools total across two families. All callable from any MCP client
(Claude Desktop, Cursor, Claude Code, ostk kernel, etc.) — either by
spawning `ostk-recall serve --stdio` directly, or, when a shared daemon
is running, through the `ostk-recall connect` bridge.

Alongside the tools, `serve` exposes an MCP **resources** surface —
`resources/list`, `resources/read`, `resources/subscribe` — currently
serving the ambient `ostk://memory-lens` (see the memory lens under
[Status](#status)). Clients that subscribe receive change notifications as
the lens refreshes.

### Recall family — corpus retrieval (`crates/mcp`)

| tool           | input schema (one-line)                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| `recall`       | `{ query: string, project?: string, source?: string, since?: rfc3339, limit?: 1..100 }`                      |
| `recall_link`  | `{ chunk_id: string }` — returns the chunk plus its parent chain                                             |
| `recall_stats` | `{}` — total count, breakdown by source, model info, last-scan timestamp                                     |
| `recall_fault` | `{ query: string, intent?: "symbol"\|"narrative"\|"trace"\|"general", limit?: int, max_per_source_id?: int }` — synthesizes hits into named virtual-memory pages; ostk's `mem_fault_recall` calls this |
| `recall_audit` | `{ sql: string }` — raw SELECT over the SQLite `audit_events` table (ostk_project sources only; single statement) |

### Attention family — live thread/scope runtime (`crates/attention-mcp`)

Every tool that takes an `AttentionScope` carries it as
`{ project?, session_id?, agent?, privacy_tier? }` — absence defaults to
`(project=None, privacy_tier=t1_project)`. Pages carry `ScoreAttribution`
per the `abi-as-sovereign-boundary` doctrine.

| tool                    | input schema (one-line)                                                                |
| ----------------------- | -------------------------------------------------------------------------------------- |
| `attention_attend`      | `{ scope?, context: string }` — ingest context into the scope's attention vector       |
| `attention_surface`     | `{ scope?, limit?: 1..200 }` — pages above `ARCHIVE_THRESHOLD`                         |
| `attention_fold`        | `{ scope?, handle: string, depth: folded\|half\|full }`                                |
| `attention_familiarize` | `{ scope?, handle: string }` — increment familiarity counter                           |
| `attention_decay`       | `{ handle: string, factor: number }` — multiplicative fade factor on the floor        |
| `attention_focus`       | `{ scope?, query: string, surface_limit?: 0..100 }` — pin a focus query; ranking uses the pinned vector until cleared |
| `attention_refocus`     | `{ scope?, query: string, surface_limit?: 0..100 }` — rotate pin to a query already in the scope's focus history |
| `attention_unfocus`     | `{ scope?, surface_limit?: 0..100 }` — clear the pin; ranking returns to the conversational transient |
| `attention_status`      | `{ scope?, surface_limit?: 0..100 }` — read-only snapshot: current pin, history, transient state |
| `thread_create`         | `{ scope?, handle: string, body?: string, tension?: active\|slack\|dormant }`         |
| `thread_link`           | `{ scope?, handle: string, target_path: string, category: string }`                   |
| `thread_unlink`         | `{ evidence_id: integer }`                                                             |
| `thread_promote`        | `{ handle_from_proposed: string, target_tier: active\|slack }`                        |
| `thread_list`           | `{ scope?, tension?: active\|slack\|dormant }`                                        |
| `thread_emergent`       | `{ since_hours?, limit?, min_cluster_size?, cohesion_threshold?, min_neighbours?, persist? }` — embedding-density clusters from the corpus |
| `thread_attention`      | `{ since_hours?, limit?, samples_per_burst?, decay_hours? }` — activity-burst surface  |
| `thread_novelty`        | `{ since_hours?, baseline_days?, limit?, min_cluster_size?, recluster_threshold?, min_mean_novelty? }` — divergence-from-baseline clusters |
| `thread_query`          | `{ signals?: [density\|activity\|novelty], rank_by?, composite_weights?, since_hours?, ... }` — unified multi-signal query; supersedes the three legacy verbs above (kept until v1.0.0) |
| `thread_evidence`       | `{ action: add\|list\|delete, ... }` — manage thread→thread evidence edges (cites/supersedes/etc.) |

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

This spawns a self-contained server per client. If you instead run a
shared daemon (`ostk-recall serve --watch`, for the background watcher
and ambient lens), point the client at the bridge so it joins the
daemon rather than colliding with its `.serve.lock`:

```json
{
  "mcpServers": {
    "ostk-recall": {
      "type": "stdio",
      "command": "ostk-recall",
      "args": ["connect"]
    }
  }
}
```

### ostk (llmOS)

**ostk v6.0.0+** ships with `fcp-recall` baked into the driver
defaults — `mem_fault_recall` and the `recall` family route through
`ostk-recall serve --stdio` automatically. Just have `ostk-recall` on
`$PATH`. No HUMANFILE entry needed.

For pre-v6 ostk or other ostk-shaped projects, register manually
via HUMANFILE (`~/.ostk/HUMANFILE` or project-local):

```
DRIVER recall ostk-recall serve --stdio
```

Form is `DRIVER <name> [transport] <command>`. Transport defaults to
`fcp` when omitted; ostk's driver-relay wraps the stdio MCP
subprocess into a Unix socket at `.ostk/drivers/fcp-<name>.sock`. The
verb shows up on next boot.

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
                                          ┌─► MCP (stdio)  ──► clients
  sources ──► scanners ──► pipeline ──► store    (24 tools)    ▲
   (fs)      (8 kinds)    chunk+embed   │                       │
                          + merge_insert │                       │
                                         │  ┌──────────────┐    │
                                         ├──┤ corpus.lance ├────┤
                                         │  │ + Tantivy    │    │ recall_*
                                         │  └──────────────┘    │ (5 tools)
                                         │                       │
                                         │  ┌──────────────┐    │
                                         ├──┤ manifest +   │    │
                                         │  │ audit_events │    │
                                         │  └──────────────┘    │
                                         │                       │
                                         │  ┌──────────────┐    │ attention_*
                                         └──┤ threads.sqlite├───┤ thread_*
                                            │ + chain ledger│    │ (19 tools)
                                            └──────┬───────┘    │
                                                   │             │
              ┌────────────────────────────────────┴─────┐       │
              ▼                                          ▼       │
       ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
       │ TurnObserver   │  │ AutoWeaver     │  │ IdleCurator  │──┘
       │ on TurnEnd     │  │ on TurnEnd     │  │ timer-driven │
       │ → membrane     │  │ → evidence     │  │ → fade +     │
       │   chunks +     │  │   links        │  │   tension    │
       │   stubs        │  │                │  │   transitions│
       └────────────────┘  └────────────────┘  └──────────────┘
                                                       ▲
                          fs events ──► watch ─────────┤
                          (debounced)                   │
                                          trigger.sock ─┘
                                          (per-path scan)
```

**Read path** (clients → corpus): the daemon (`serve`, no `--stdio`)
serves MCP to many clients over a local socket / named pipe; stdio
clients reach it through the `connect` bridge. `serve --stdio` is the
alternative direct transport — the process is itself the MCP server —
for a client (e.g. the `fcp-recall` driver under ostk v6+) that prefers
to spawn its own rather than share a daemon. A `.serve.lock` flock
admits one `serve` per corpus, so run *either* a shared daemon (clients
use `connect`) *or* per-client `serve --stdio`, not both at once.

**Write path** (edits → corpus): the daemon binds `recall.sock`; a
watcher (`serve --watch` in-process, or a separate `watch` process)
debounces filesystem events and delivers the changed paths; a single
scan mutex serialises every trigger so `Pipeline::scan_paths` does
per-path ingest without overlapping the single writer.

**Attention loop** (conversational stream → live state): on a watched
transcript **TurnEnd**, TurnObserver emits membrane chunks via
`Pipeline::ingest_synthetic` and AutoWeaver writes derived evidence links
— both gate on `IngestEvent::is_turn_end()`, so bulk ingest is skipped.
`ostk-recall weave` runs the same weaver over bulk content on demand.
IdleCurator scores fade on a timer and transitions tension states.

One daemon owns the corpus as the single writer (`.serve.lock`); its
read-only query engine reopens the Lance table per query, so background
scans are visible to in-flight reads immediately. The
`InMemoryAttention` score tier is per-process; it rebuilds from the
`threads.sqlite` chain ledger on boot.

Stack: model2vec-rs (embedder) ▶ LanceDB vector + Tantivy BM25 (store)
+ SQLite (manifest + audit + threads ledger) ▶ pipeline (scan ▶ chunk
▶ embed ▶ `merge_insert`) ▶ query (dense + BM25 with RRF fusion, then
optional fastembed-rs cross-encoder rerank) ▶ attention runtime
(observer/weaver/curator) ▶ MCP server (24 tools).

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
