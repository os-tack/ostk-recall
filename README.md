# ostk-recall

A local, single-binary search-and-recall service for your own files and chat
history. It indexes notes, source code, and assistant session logs into a local
corpus, runs hybrid semantic + keyword retrieval over them, and exposes the
results to any MCP client (Claude Desktop, Cursor, Claude Code, the ostk kernel,
and others). Data and indexes stay on your machine; nothing is sent off-box.

Built on [model2vec-rs](https://github.com/MinishLab/model2vec-rs) for static
embeddings, [LanceDB](https://lancedb.com/) (Arrow + Tantivy BM25) for the vector
and full-text store, [SQLite](https://www.sqlite.org/) (via rusqlite) for the
ingest manifest, audit log, and concept/thread ledger, and
[fastembed-rs](https://github.com/Anush008/fastembed-rs) for the optional
cross-encoder reranker.

## What it does

- **Ingests many source kinds into one corpus** — markdown trees, source code,
  Claude Code, Gemini, and Codex CLI session logs, arbitrary globs, Claude.ai
  `.zip` exports, and ostk `.ostk/` directories.
- **Hybrid retrieval** — model2vec dense vectors and Tantivy BM25, fused with
  Reciprocal Rank Fusion, with an optional cross-encoder rerank pass.
- **A concept ledger** — typed nodes and attributed, directed edges, durable in
  SQLite. Grow them from markdown frontmatter at scan time, or write them
  directly over MCP. Each edge records its origin (authored / observed /
  promoted) and derives a conductance from confidence and recency rather than
  storing a weight. Diffusion also walks the latent (vector-similarity) half of
  the graph; an off-diagonal bridge walked during consolidation is promoted into
  a weak reified edge that must then earn its conductance through use, or decay.
- **A live attention runtime** — a turn observer, an auto-weaver that links new
  chunks to thread anchors, and an idle curator that fades inactive threads. It
  also maintains an ambient "memory lens" (an MCP resource) aligned to whatever
  the current attention vector is focused on.
- **31 MCP tools** plus a resources surface, served over stdio or a shared
  local-socket daemon.

## Status

Pre-alpha but functional; the maintainer runs it as a daily driver.

Not yet built:

- Per-file offset cursors for `claude_code` / `gemini` incremental scan
  (append-only JSONL currently falls back to a full-source rescan when poked by
  the watcher; content-addressed chunk ids keep that idempotent, just wasteful).
- MCP transports beyond stdio (HTTP / SSE).

## Install

### Release binaries

Pre-built binaries for Linux x86_64, macOS arm64, and Windows x86_64 ship with
every tagged release: <https://github.com/os-tack/ostk-recall/releases>. Untar
(unzip on Windows), put the binary on your `PATH`, done.

### From source

```
git clone https://github.com/os-tack/ostk-recall
cd ostk-recall
make install     # → ~/.cargo/bin/ostk-recall
```

`make install` wraps `cargo install --path crates/cli --locked --force`.
`make help` lists every target; `make version` shows the installed version and
the workspace git sha.

Build-time native dependency (pulled in transitively, but the system tool must
be present):

- macOS: `brew install protobuf`
- Debian / Ubuntu: `apt-get install protobuf-compiler`

## Configure

Default path: `${XDG_CONFIG_HOME:-~/.config}/ostk-recall/config.toml`.

Minimal config:

```toml
[corpus]
root = "~/.local/share/ostk-recall"

[embedder]
model = "minishlab/potion-retrieval-32M"   # repo id "org/name" — the prefix is required

[[sources]]
kind = "markdown"
project = "notes"
paths = ["~/notes"]
```

[`config.example.toml`](./config.example.toml) is a complete, commented
reference for every option — globals (`[corpus]`, `[embedder]`, `[reranker]`,
`[runtime]`, `[lens]`, `[weaver]`, `[[record_rule]]`), all source kinds, the
relational `entity_type` / `edges` scanner, and the `[watch]` block.

## Quickstart

```
ostk-recall init                # create the corpus root, download the model
ostk-recall scan                # ingest the configured sources
ostk-recall serve               # run the daemon: MCP over a local socket + ambient lens
ostk-recall serve --watch       # daemon + in-process file-watcher (keeps the corpus fresh)
ostk-recall serve --stdio       # direct stdio MCP transport (this process is the server)
ostk-recall connect             # bridge a stdio MCP client to a running daemon
ostk-recall watch               # standalone file-watcher → pokes a running daemon
ostk-recall weave               # weave bulk-scanned content into the thread graph
ostk-recall verify              # reconcile counts across store + manifest
ostk-recall optimize            # compact and fold indexes
ostk-recall lens show           # print the current ambient memory lens
```

`ostk-recall --help` documents the full CLI; `ostk-recall <verb> --help` covers
each verb.

## Examples

The [`examples/`](./examples/) directory has four self-contained, runnable
setups — each with sample content, a `config.toml`, a `run.sh`, and a README
walking the `scan → query` loop:

| Example | Shows |
| --- | --- |
| [`01-engineering`](./examples/01-engineering/) | code + design decisions in one corpus; plain hybrid recall |
| [`02-personal-kb`](./examples/02-personal-kb/) | typed nodes + authored edges from markdown frontmatter (`entity_type` / `edges`) |
| [`03-persistent-agent`](./examples/03-persistent-agent/) | an agent's durable memory via `memory_remember` / `memory_connect`, surviving restarts |
| [`04-shared-substrate`](./examples/04-shared-substrate/) | one daemon, multiple clients over a shared corpus |

Each keeps its corpus inside the example directory (gitignored), so running them
never touches your real corpus.

## Source kinds

| kind | ingests | chunking |
| --- | --- | --- |
| `markdown` | `.md` / `.markdown` trees | split on headings, soft-wrap ~400 tokens |
| `code` | source files filtered by `extensions = [...]` | tree-sitter symbol chunks (rs/py/ts/js/go); line-window fallback |
| `claude_code` | Claude Code session logs (`<slug>/*.jsonl`) | one chunk per user / assistant turn |
| `gemini` | Gemini CLI session JSON (`session-*.json`, recursive) | one chunk per user/gemini exchange |
| `codex` | Codex CLI session logs (`~/.codex/sessions/**/rollout-*.jsonl`) | one chunk per user turn |
| `file_glob` | an arbitrary glob, as plain text | paragraph split, soft-wrap ~400 tokens |
| `zip_export` | Claude.ai data-export `.zip` bundles | per-conversation-turn chunks |
| `ostk_project` | ostk `.ostk/` dirs — decisions, needles, audit, specs, code | composite; one chunk per record |
| `thread` | `.ostk/threads/*.md` files; tension state as metadata | one chunk per thread file |

A markdown or code source can also set `entity_type` and `edges` to seed typed
concept nodes and authored edges from each file's frontmatter at scan time — see
[`examples/02-personal-kb`](./examples/02-personal-kb/) and
[`config.example.toml`](./config.example.toml).

## MCP tools

31 tools across four families, callable from any MCP client — either by spawning
`ostk-recall serve --stdio` directly, or, when a shared daemon is running,
through the `ostk-recall connect` bridge. `serve` also exposes an MCP
**resources** surface (`resources/list` / `read` / `subscribe`) serving the
ambient `ostk://memory-lens`; subscribers get a notification when it refreshes.

### Recall — corpus retrieval (5)

| tool | input (one-line) |
| --- | --- |
| `recall` | `{ query, project?, source?, since?, limit?: 1..100 }` |
| `recall_link` | `{ chunk_id }` — the chunk plus its parent chain |
| `recall_stats` | `{}` — total count, breakdown by source, model info, last scan |
| `recall_fault` | `{ query, intent?: symbol\|narrative\|trace\|general, limit?, max_per_source_id? }` — synthesizes hits into named virtual-memory pages |
| `recall_audit` | `{ sql }` — single read-only SELECT over the `audit_events` table |

### Memory — concept ledger façade (7)

A higher-level surface composing recall, attention, and the ledger. Pass
`project` to target a namespaced graph; absent a project, these operate on the
global scope.

| tool | input (one-line) |
| --- | --- |
| `memory_recall` | `{ query, project?, limit?, source?, learn? }` — recall, optionally learning concept candidates |
| `memory_concept` | `{ action: show\|list\|promote\|reject\|merge\|alias\|summarize\|crystallize, handle?, project?, ... }` — inspect/correct a concept card; `crystallize` writes a proposed typed node's stub file |
| `memory_surface` | `{ view?: now\|concepts\|open_loops, project?, limit? }` — working-memory view |
| `memory_remember` | `{ kind: concept_seed\|note\|decision\|fact\|open_question, text, concept?, project? }` |
| `memory_connect` | `{ from, relation, to, project?, evidence? }` — author a concept edge |
| `memory_focus` | `{ target, project? }` — pin a focus that biases recall |
| `memory_reflect` | `{}` — consolidation pass: promote corroborated candidates, re-resolve evidence, and promote off-diagonal latent bridges into weak `promoted` edges |

### Attention — live scope runtime (9)

Each tool takes an optional `AttentionScope` as
`{ project?, session_id?, agent?, privacy_tier? }`; absence defaults to
`(project=None, privacy_tier=t1_project)`.

| tool | input (one-line) |
| --- | --- |
| `attention_attend` | `{ scope?, context }` — ingest context into the scope's attention vector |
| `attention_surface` | `{ scope?, limit?: 1..200 }` — pages above the archive threshold |
| `attention_fold` | `{ scope?, handle, depth: folded\|half\|full }` |
| `attention_familiarize` | `{ scope?, handle }` — increment the familiarity counter |
| `attention_decay` | `{ handle, factor }` — multiplicative fade on the floor |
| `attention_focus` | `{ scope?, query, surface_limit?: 0..100 }` — pin a focus query |
| `attention_refocus` | `{ scope?, query, surface_limit? }` — rotate to a query already in focus history |
| `attention_unfocus` | `{ scope?, surface_limit? }` — clear the pin |
| `attention_status` | `{ scope?, surface_limit? }` — read-only snapshot of pin / history / transient |

### Thread — durable thread ledger (10)

| tool | input (one-line) |
| --- | --- |
| `thread_create` | `{ scope?, handle, body?, tension?: active\|slack\|dormant }` |
| `thread_link` | `{ scope?, handle, target_path, category }` |
| `thread_unlink` | `{ evidence_id }` |
| `thread_promote` | `{ handle_from_proposed, target_tier: active\|slack }` |
| `thread_list` | `{ scope?, tension? }` |
| `thread_emergent` | `{ since_hours?, limit?, min_cluster_size?, cohesion_threshold?, ... }` — embedding-density clusters |
| `thread_attention` | `{ since_hours?, limit?, samples_per_burst?, decay_hours? }` — activity-burst surface |
| `thread_novelty` | `{ since_hours?, baseline_days?, limit?, min_mean_novelty?, ... }` — divergence-from-baseline clusters |
| `thread_query` | `{ signals?: [density\|activity\|novelty], rank_by?, composite_weights?, ... }` — unified multi-signal query |
| `thread_evidence` | `{ action: add\|list\|delete, ... }` — thread→thread evidence edges |

## Connecting clients

`serve --stdio` spawns a self-contained server per client. To share one corpus
across clients, run a daemon (`ostk-recall serve` or `serve --watch`) and point
each client at the `connect` bridge instead — a `.serve.lock` admits only one
`serve` per corpus, so run *either* a shared daemon (clients use `connect`) *or*
per-client `serve --stdio`, not both.

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ostk-recall": { "command": "ostk-recall", "args": ["serve", "--stdio"] }
  }
}
```

### Cursor

`~/.cursor/mcp.json` (or project-local `.cursor/mcp.json`): same `mcpServers`
entry as above.

### Claude Code

`.mcp.json` at user or project level:

```json
{
  "mcpServers": {
    "ostk-recall": { "type": "stdio", "command": "ostk-recall", "args": ["serve", "--stdio"] }
  }
}
```

To join a shared daemon instead, replace the args with `["connect"]`.

### ostk

ostk v6.0.0+ ships `fcp-recall` in its driver defaults — `mem_fault_recall` and
the recall family route through `ostk-recall serve --stdio` automatically; just
keep `ostk-recall` on `PATH`. For earlier ostk, register via HUMANFILE:

```
DRIVER recall ostk-recall serve --stdio
```

## Live updates

The daemon binds a scan-trigger socket at `corpus.root/recall.sock` (named pipe
`\\.\pipe\ostk-recall-recall` on Windows). `ostk-recall serve --watch` runs the
watcher in-process; alternatively run a separate `ostk-recall watch`. Either way,
edits under a configured source path are debounced and re-ingested without a
manual rescan:

```toml
[watch]
enabled = true
# debounce_ms defaults: 800 (Linux), 1200 (Windows), 1500 (macOS).
# projects = ["notes", "docs"]   # optional allowlist; defaults to all sources.
mode = "incremental"             # path-aware ingest; default "legacy" rescans per kick.
```

As of the v0.6 TurnEnd gate, only watched conversation transcripts
(`claude_code` / `gemini`) drive the live attention runtime; other sources are
ingested and searchable but do not wake the observer or weaver.

## Consolidation

`serve` handles live cognition (a watched transcript turn wakes the observer and
weaver). Offline consolidation is a separate, operator-scheduled step — run it
from cron / launchd / systemd, not inside `serve`:

```
ostk-recall weave --since 24h                 # bind recent arrivals to thread anchors
ostk-recall weave --consolidate --since 1w    # deep re-weave, bridge, merge, promote, fade
ostk-recall optimize                          # compact fragments
```

`weave --consolidate` runs the full cycle over its `--since` window; the window
is the consolidation horizon, so pick it per schedule tier. The CLI is the only
contract — the scheduler is yours. Set `HF_HUB_OFFLINE=1` for offline runs once
the model is cached.

## Architecture

```
                                          ┌─► MCP (stdio / socket) ──► clients
  sources ──► scanners ──► pipeline ──► store      (31 tools)         ▲
   (fs)      (8 kinds)    chunk+embed   │                              │
                          + merge_insert │                              │
                                         │  ┌──────────────┐           │
                                         ├──┤ corpus.lance ├───────────┤ recall_*
                                         │  │ + Tantivy    │           │ memory_*
                                         │  └──────────────┘           │
                                         │  ┌──────────────┐           │
                                         ├──┤ manifest +   │           │
                                         │  │ audit_events │           │
                                         │  └──────────────┘           │
                                         │  ┌──────────────┐           │ attention_*
                                         └──┤ threads.sqlite├──────────┤ thread_*
                                            │ + chain ledger│           │
                                            └──────┬───────┘           │
              ┌────────────────────────────────────┴─────┐             │
              ▼                  ▼                  ▼                    │
       ┌────────────┐   ┌────────────┐   ┌──────────────┐              │
       │TurnObserver│   │ AutoWeaver │   │ IdleCurator  │──────────────┘
       │on TurnEnd  │   │on TurnEnd  │   │ timer-driven │
       │→ membrane  │   │→ evidence  │   │→ fade +      │
       │  chunks    │   │  links     │   │  tension     │
       └────────────┘   └────────────┘   └──────────────┘
                                                ▲
                          fs events ──► watch ──┘  (debounced → trigger.sock)
```

**Read path.** The daemon serves MCP to many clients over a local socket; stdio
clients reach it through `connect`. `serve --stdio` is the alternative direct
transport for a client that prefers to spawn its own server.

**Write path.** The daemon binds `recall.sock`; a watcher debounces filesystem
events and delivers changed paths; a scan mutex serialises every trigger so
per-path ingest never overlaps the single writer.

**Attention loop.** On a watched transcript TurnEnd, TurnObserver emits membrane
chunks via `Pipeline::ingest_synthetic` and AutoWeaver writes derived evidence
links — both gate on `IngestEvent::is_turn_end()`, so bulk ingest is skipped.
`ostk-recall weave` runs the same weaver over bulk content on demand. IdleCurator
fades inactive threads on a timer. The `InMemoryAttention` score tier is
per-process and rebuilds from the `threads.sqlite` chain ledger on boot.

Stack: model2vec-rs (embedder) ▶ LanceDB + Tantivy (store) + SQLite (manifest,
audit, ledger) ▶ pipeline (scan ▶ chunk ▶ embed ▶ `merge_insert`) ▶ query (dense
+ BM25 with RRF, then optional cross-encoder rerank) ▶ attention runtime ▶ MCP
server.

See [`docs/architecture.md`](./docs/architecture.md) and
[`docs/spec/driver-protocol.md`](./docs/spec/driver-protocol.md) for full
writeups.

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
