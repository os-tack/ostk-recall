# 04 — One substrate, many minds

The concrete form of the thesis that **minds are *users* of a shared substrate, not nodes in
it**. Several clients query and write against one corpus + `threads.sqlite`. This example shows
what is correctly shared — and surfaces a known, not-yet-built seam: attention/pins are not yet
partitioned per mind.

## One daemon, many clients (not many daemons)

`serve` holds a singleton advisory lock (`<root>/.serve.lock`) per corpus root; a second
`serve` over the same root exits cleanly. So the topology is **one daemon, many `connect`
clients**:

```sh
bash run.sh                                   # init + scan the shared corpus
export EX="$PWD"                               # the config references $EX
ostk-recall --config config.toml serve        # the ONE daemon
```

Register two MCP clients, both bridging to that daemon:

```json
{
  "mcpServers": {
    "recall-A": { "command": "ostk-recall", "args": ["--config", "/abs/…/04-shared-substrate/config.toml", "connect"], "env": { "EX": "/abs/…/04-shared-substrate" } },
    "recall-B": { "command": "ostk-recall", "args": ["--config", "/abs/…/04-shared-substrate/config.toml", "connect"], "env": { "EX": "/abs/…/04-shared-substrate" } }
  }
}
```

## What's shared (correctly)

The **substrate** — concepts, threads, the chain. A concept written by client A
(`memory_remember` / `memory_connect`) is immediately visible to client B via
`memory_concept` / `memory_recall`. This is the design: the durable graph is common ground.

```
# via client A
memory_remember(kind="concept_seed", text="slipstream", project="shared")
# via client B
memory_concept(action="show", handle="slipstream", project="shared")   # → present. Shared substrate works.
```

## What leaks (the known seam)

**Attention/pins are NOT partitioned per mind.** The `memory_*` façade defaults to the empty
`AttentionScope`, so focus state is global, not per-client:

```
# via client A
memory_focus(target="slipstream")
# via client B
memory_surface(view="now")     # → shows A's focus on slipstream. That's the bleed.
memory_recall(query="…")        # → B's ranking is biased by A's pin.
```

This is the **pin-partitions** seam (`.ostk/threads/cognitive-memory/pin-partitions.md`): the
substrate wants to be shared, but attention wants to be partitioned, per mind, keyed by
`AttentionScope{agent, session, project}`. The slot exists; the façade just doesn't yet default
to the caller's own scope. Two layers, opposite sharing policies — documented here as the next
step, not fixed in this example.

> Takeaway: sharing the substrate is a feature; sharing the *spotlight* is a bug. The first is
> what this example demonstrates; the second is what it honestly exposes.
