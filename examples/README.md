# ostk-recall examples

Each sub-directory is a **self-contained** ostk-recall instance: sample content, a
`config.toml`, a `run.sh`, and a README walking the `scan → query` loop. They demonstrate the
distinct memory concerns the substrate supports, and double as fixtures for the
relational-substrate slices.

| Example | Concern | What it shows |
|---|---|---|
| [`01-engineering`](01-engineering/) | code + design decisions | two source kinds (`code` + `markdown`) over one corpus; pure recall — the dogfood case |
| [`02-personal-kb`](02-personal-kb/) | personal knowledge base | **typed nodes + authored edges** from frontmatter (`entity_type`/`edges`) — the relational-substrate slice-3 showcase |
| [`03-persistent-agent`](03-persistent-agent/) | an agent's durable memory | the `memory_remember` / `memory_connect` write path; a graph that survives daemon restarts in `threads.sqlite` |
| [`04-shared-substrate`](04-shared-substrate/) | many minds, one substrate | one `serve` daemon, two `connect` clients over a shared corpus; documents the current attention/pin bleed |

## The portable-corpus convention

Every example keeps its corpus **inside the example directory** and never touches your real
ostk-recall corpus. The configs reference an env var `$EX` (the example's own absolute path),
which ostk-recall expands at load time (`shellexpand` supports `$VAR` and `~`):

```toml
[corpus]
root = "$EX/.recall"
[[sources]]
paths = ["$EX/content/people"]
```

The `run.sh` in each example sets `EX` for you:

```sh
export EX="$(cd "$(dirname "$0")" && pwd)"
```

If you run `ostk-recall` by hand instead of via `run.sh`, export it first from the example dir:
`export EX="$PWD"`. The generated `$EX/.recall/` corpus is gitignored.

## The universal loop

```sh
cd examples/<name>
bash run.sh                       # init + scan; run.sh sets $EX itself (first run downloads the embedder)

# For any command you run yourself, export $EX first — the config references it:
export EX="$PWD"
ostk-recall --config config.toml serve     # start the MCP daemon (binds <root>/recall.sock)
```

Then point an MCP client at it. Register a stdio bridge in your client's MCP config:

```json
{
  "mcpServers": {
    "recall-example": {
      "command": "ostk-recall",
      "args": ["--config", "/abs/path/to/examples/<name>/config.toml", "connect"],
      "env": { "EX": "/abs/path/to/examples/<name>" }
    }
  }
}
```

`connect` bridges the client's stdio to the running `serve` daemon. Because that bridge process
loads the config (to find the daemon's socket), it needs `$EX` in its environment too — hence
the `env` block above. From there, query via the `memory_*` tools — `memory_recall`,
`memory_concept`, `memory_surface`, and (for writes) `memory_remember` / `memory_connect` /
`memory_focus`. Pass `project="..."` matching the example's source `project` (e.g. `memories`,
`agent-memory`) — absent a project, `memory_*` operates on the global scope and won't see the
example's seeded graph.

> **One `serve` per corpus root.** The daemon holds a singleton advisory lock
> (`<root>/.serve.lock`); a second `serve` over the same root exits cleanly. To run many
> *clients* over one substrate, start one `serve` and register multiple `connect` bridges —
> that's exactly what [`04-shared-substrate`](04-shared-substrate/) demonstrates.
