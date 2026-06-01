# 03 — Persistent agent (durable cross-session memory)

An agent that remembers across runs. Its memory has two layers, both durable in
`threads.sqlite`:

1. **Seeded files** — `content/memory/*.md` become typed `note` nodes with `relates_to` edges
   (same authoring mechanism as [02](../02-personal-kb/)). This is the agent's starting graph.
2. **Runtime writes** — during a session the agent grows its memory through the `memory_*`
   write tools, with no file involved.

## The write surface

- `memory_remember(kind="concept_seed", text="slipstream", project="agent-memory")` — mint a
  new proposed concept.
- `memory_remember(kind="note"|"decision"|"fact"|"open_question", concept="project-ostk", project="agent-memory", text="…")`
  — attach a timestamped narrative record to an existing concept.
- `memory_connect(from="agent", relation="relates_to", to="slipstream", project="agent-memory")`
  — draw an edge (creating either endpoint as a candidate if absent), always with provenance.

> Pass `project="agent-memory"` on every call — this example's graph lives in that project;
> absent a project, `memory_*` uses the global scope and won't touch the seeded graph.

## The point: persistence across sessions

```
bash run.sh                                   # seed the starting graph
export EX="$PWD"                               # the config references $EX
ostk-recall --config config.toml serve        # session 1 daemon
```

**Session 1** — through a connected client, the agent writes:

```
memory_remember(kind="decision", concept="project-ostk", project="agent-memory",
                text="Chose a gazetteer over wiki-links for mention-linking.")
memory_connect(from="project-ostk", relation="relates_to", to="gazetteer", project="agent-memory")
```

Stop the daemon (the in-memory process is gone). **Session 2** — restart `serve`, reconnect,
and the agent's additions are still there:

```
memory_concept(action="show", handle="project-ostk", project="agent-memory")
#   → edges_from includes relates_to → gazetteer; notes include the decision text
memory_concept(action="list", project="agent-memory")
memory_surface(view="concepts", project="agent-memory")
```

## What is and isn't restored on boot

The concept **ledger** — nodes, edges, notes — is durable in `threads.sqlite` and read back
directly, so `memory_concept` / `memory_surface` show the full accreted graph after a restart.
Concept **activation** (how strongly something is lit *right now*) is computed from the
`chain_log` on demand and is not replayed into in-memory attention at boot. So prove durability
with the ledger views above — not by expecting recall *ranking* to remember the last session's
focus.
