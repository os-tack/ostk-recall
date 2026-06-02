# 02 — Personal knowledge base (typed nodes, authored + observed edges, prose-name promotion)

The relational-substrate showcase for **slices 3–5**, end to end. The directory layout is the
ontology; dropping a markdown file creates a typed node, frontmatter fields create edges, prose
that names a known node creates observed edges, and a capitalized name that recurs in prose
across several same-typed docs — with no file of its own — is proposed as a new typed node you
can later write to a file. No `[[wiki-links]]`, no API calls, no NER model — the filesystem is
the authoring surface, and the graph proposes the rest from what you already wrote.

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

## Prose mentions (observed edges)

Beyond frontmatter, plain prose in a file's **body** that *names* a known node lights an
`observed` `mentions` edge automatically — no markup. After a scan, `mike.md`'s body (“He and
Tori both…”) yields `mike --mentions--> tori` even though Mike's frontmatter never declares it,
and the standup's body yields `2026-05-20-standup --mentions--> ostk-recall`. These coexist
with the slice-3 authored edges to the same targets (different `relation`). The match is against
a **gazetteer** of every known node's handle + aliases, matched in resolution order — a handle
before an alias, and the file's project before global within each — so it agrees with how
`memory_*` resolves the same name; ambiguous names (one surface form, two nodes in the same
tier) are skipped rather than guessed. `run.sh` prints the observed
edges alongside the authored ones.

## Prose-name promotion (slice 5)

The gazetteer (above) only links names already in the graph. Slice 5 handles the names that
*aren't* there yet: a capitalized word in prose that does not resolve to any concept, but recurs
across several same-typed documents, is proposed as a new node. In this example **Priya** appears
in three `person` bodies (`mike`, `sarah`, `tori`) and has no file; after a scan she is a
`priya | person | proposed` node with an observed `mentions` edge from each of the three docs.

**What is extracted (per document body, no ML):** a candidate is a single capitalized token that
is *not* at a sentence start (sentence-initial capitals are ambiguous and skipped), is ≥3 chars,
is not a stopword, with any trailing possessive (`Sarah's` → `Sarah`) stripped. Tokens that
resolve to an existing concept — by handle, by alias, or as a `rejected`/`merged` one — are
dropped, so known and previously-declined names never reappear as candidates.

**What is promoted:** a candidate is materialized only if it recurs across **≥3 distinct
documents of one dominant `entity_type`** (the gate; a tie between two kinds is dropped, not
guessed). It is typed by that dominant kind — `person` here, because that is the kind of the docs
it recurs in. The new node is created `proposed` (confidence `0.4`) with an `observed` `mentions`
edge (confidence `0.1`, `by=scanner`) from each contributing document. **No file is written.**
Promotion runs on full scans only; an incremental single-file scan never has the cross-document
view to cross the gate.

**Writing the file (`crystallize`).** A proposed node is a suggestion; turning it into a real,
content-bearing node is an explicit step that writes one file and never overwrites:

```sh
ostk-recall --config config.toml crystallize priya --project memories
```

This resolves the target directory from the `[[sources]]` block whose `entity_type` matches the
node's kind (`person` → `content/people/`), writes `priya.md` with frontmatter pre-filled for
that source's edge vocabulary (`project`, `kind`, `families: []`, `works_on: []`), and returns
`created: false` if the file already exists. The next scan then ingests it as a normal typed
node. The same operation is available to MCP clients as `memory_concept` with
`{ "action": "crystallize", "handle": "priya", "project": "memories" }`; both require the node to
be `proposed` and typed.

## Run + inspect

```sh
bash run.sh
```

`run.sh` runs `init` + `scan`, dumps the resulting `concepts` and `concept_edges` straight from
`threads.sqlite` (typed nodes + authored/observed edges, including the promoted
`priya | person | proposed`), then crystallizes `priya` to a file and prints it. Seeding,
mention-linking, and promotion all happen *inside* `scan`; the script deletes the generated
`priya.md` afterward so the example stays re-runnable.

## Query the live graph

Start the daemon and connect a client (see the top-level
[README](../README.md#the-universal-loop)), then follow [`smoke.md`](smoke.md) — a short,
asserted walk through `memory_concept`, `memory_surface`, and `memory_recall`, including a
restart to prove the graph is durable.
