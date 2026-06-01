# 0002 — Derive edge conductance, never store it

- **Status:** accepted
- **Date:** 2026-05-30
- **Supersedes:** the early "store a weight column" sketch

## Context

Edges in the relational substrate need a notion of how strongly current flows through them.
The naive design stores a mutable `weight` column and updates it on every use — but that
conflates *authoring* (drawing an edge) with *evidence* (the edge earning its keep).

## Decision

**Conductance is derived, never stored:** `conductance(edge, now) = confidence × recency`,
where recency decays from `last_seen_at`. Authoring sets the topology hypothesis at a low
prior; *use* sets the conductance. An authored edge that real work never traverses stays dim
and decays — the data declines to corroborate it. There is no backdoor for unfalsifiable bias.

## Consequences

- Decay is free from the `last_seen_at` timestamp — no background sweep.
- `authored ≠ evidence`: a drawn edge is a prior that must survive contact with use.
- Same discipline as candidate→active concepts: occurrence ≠ salience.
