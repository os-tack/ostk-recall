# 02 — Personal knowledge base (typed nodes + authored edges)

The relational-substrate **slice-3 showcase**, end to end. The directory layout is the
ontology; dropping a markdown file authors a typed node, and frontmatter fields author edges.
No `[[wiki-links]]`, no API calls — the filesystem is the authoring surface (the anti-Obsidian
move: the graph is primary, files crystallize from it).

## How the ontology is declared

`config.toml` has one `[[sources]]` block per node-kind, each adding two fields beyond a normal
markdown source:

```toml
[[sources]]
kind = "markdown"
project = "memories"
paths = ["$EX/content/people"]
entity_type = "person"          # every file here becomes a `person` node
edges = ["families", "works_on"] # these frontmatter fields become edge types
```

So `content/people/tori.md` with:

```yaml
---
families: [sarah]
works_on: [ostk-recall]
---
```

seeds a `person` node `tori` and two **authored** edges: `tori --families--> sarah` and
`tori --works_on--> ostk-recall`. An edge target with no file of its own (`ostk-recall`) is
auto-created as an untyped node; a target that *does* have a file (`sarah`, via
`people/sarah.md`) resolves to that existing typed `person` node — never shadowed by a
duplicate. (Resolution is handle/alias/merge-aware, across project and global scope.)

## Authored ≠ evidence

Authored edges enter at a **low prior** (`AUTHORED_EDGE_CONFIDENCE = 0.1`), not as fact.
"Authoring sets the topology hypothesis; use sets the conductance." An authored edge is a
guess about where attention *might* flow; it earns conductance only as real work traverses it,
and decays if it doesn't. Re-scanning is idempotent — nodes aren't duplicated and edges are
re-touched (their `touch_count` bumps, origin preserved).

## Run + inspect

```sh
bash run.sh
```

`run.sh` runs `init` + `scan`, then dumps the resulting `concepts` and `concept_edges` straight
from `threads.sqlite` so you can see the typed nodes and authored edges without a client. The
seeding happens *inside* `scan` (the CLI's `seed_nodes_from_source` pass).

## Query the live graph

Start the daemon and connect a client (see the top-level
[README](../README.md#the-universal-loop)), then follow [`smoke.md`](smoke.md) — a short,
asserted walk through `memory_concept`, `memory_surface`, and `memory_recall`, including a
restart to prove the graph is durable.
