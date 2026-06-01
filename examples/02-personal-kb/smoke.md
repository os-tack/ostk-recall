# Smoke checks — 02-personal-kb

A short, asserted walk to confirm the scanned graph is correct, queryable through the `memory_*`
façade, and durable across a daemon restart. Run after `bash run.sh`, with a `serve` daemon up
and an MCP client connected (see the top-level [README](../README.md#the-universal-loop)).

## 1. The typed node + its authored edges

```
memory_concept(action="show", handle="tori", project="memories")
```

**Assert:**
- `kind == "person"`
- `edges_from` contains `{relation: "families", to: "sarah"}` and
  `{relation: "works_on", to: "ostk-recall"}`
- on each of those edges: `source == "authored"`, `by == "scanner"`, `confidence ≈ 0.1`

## 2. The working set

```
memory_surface(view="concepts", project="memories")   # the active/proposed concept cards
memory_surface(view="now", project="memories")        # current focus + active threads + concepts
```

**Assert:** the personal-KB concepts (`tori`, `mike`, `office`, the meeting …) appear with
their `kind`. (`now` may be sparse on a fresh corpus — nothing has been *used* yet; that's the
point of derived conductance.)

## 3. Recall reads, it does not mutate

```
memory_recall(query="tori sarah office", learn=false)
```

**Assert:** returns hits from the scanned corpus; with `learn=false` the call does **not**
mint or mutate concept candidates — the façade is querying, not writing.

## 4. Durability across a restart

Stop the `serve` daemon, start it again, reconnect, then repeat step 1:

```
memory_concept(action="show", handle="tori", project="memories")
```

**Assert:** identical to step 1. The concept ledger (nodes, edges, notes) lives in
`threads.sqlite`; it survives the daemon. (Concept *activation* is read from `chain_log` on
demand — it is not replayed into in-memory attention on boot — so durability is proven by the
ledger card, not by recall ranking.)
