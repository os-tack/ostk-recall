# 0001 — Use LanceDB for the latent adjacency

- **Status:** accepted
- **Date:** 2026-04-02

## Context

We need an embedded vector store for similarity search over chunk embeddings. It must run
in-process (no server), support approximate nearest-neighbour over hundreds of thousands of
vectors, and coexist with a relational store for the reified graph.

## Decision

Use **LanceDB** as the corpus store. ANN over chunk vectors *is* a latent-edge query we never
have to materialize: "what is semantically near X" is computed on demand. SQLite holds the
reified, typed, attributed edges; Lance holds the dense similarity adjacency. Recall walks the
union of the two.

## Consequences

- No separate vector server to operate.
- The "off-diagonal bridge" — a Lance-similar pair with no SQLite edge yet — falls out for
  free as the surprising, un-reified neighbour.
