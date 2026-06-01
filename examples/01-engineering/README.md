# 01 — Engineering substrate (code + decisions)

The dogfood case: index a codebase and its design decisions into one corpus and recall across
both. No `entity_type`, no edges — just two source kinds and hybrid retrieval.

## Layout

```
content/
  code/        retry.rs, cache.py     # kind = "code"  (tree-sitter chunked)
  decisions/   0001-*.md, 0002-*.md   # kind = "markdown" (ADR-style design notes)
```

The two `[[sources]]` blocks in `config.toml` map each directory to a `SourceKind`. Code is
chunked per function/block by the tree-sitter chunker; markdown is chunked per doc.

## Run

```sh
bash run.sh                                  # init + scan into ./.recall
export EX="$PWD"                              # the config references $EX
ostk-recall --config config.toml serve       # MCP daemon
```

## Query (via the `memory_*` MCP tools)

- `memory_recall(query="exponential backoff retry")` → surfaces `retry.rs` (code lane).
- `memory_recall(query="why derive conductance instead of storing a weight")` → surfaces
  decision `0002` (markdown lane). Cross-kind recall over one corpus is the point.
- `memory_surface(view="now")` → what the substrate currently considers active.

This example has no typed graph — it's the baseline. See [`02-personal-kb`](../02-personal-kb/)
for typed nodes and authored edges.
