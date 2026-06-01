---
relates_to: [project-ostk]
---
# Agent

The agent itself — the mind that accumulates this memory. It works primarily on the
`project-ostk` effort, so it authors a `relates_to` edge there from the start.

Everything else the agent learns across sessions hangs off this seed: notes, decisions, and
edges it writes at runtime via `memory_remember` / `memory_connect`.
