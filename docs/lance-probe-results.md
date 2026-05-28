# Lance probe results (P2)

Date: 2026-05-27 (initial; re-run on Lance upgrades).
Lance version: **0.29.0** (lancedb crate; via `Cargo.toml` workspace pin).
Arrow version: **58.3.0** (bumped from 57 alongside the Lance bump).

## Probe 1: `array_contains` on `List<Utf8>`

**Question**: Does `lance.table.query().only_if("array_contains(facets,
'project:auth')")` work in lance 0.29.0?

**Method**: `crates/store/examples/array_contains_probe.rs` — builds a
3-row table with a `facets: List<Utf8>` column matching P1's encoding
(`"key:value"` strings), then issues two `array_contains` filters and
asserts row counts.

**Result**: **PASS.**
- `array_contains(facets, 'project:auth')` → 1 row (correct).
- `array_contains(facets, 'lang:rust')` → 2 rows (correct).

**Decision**: P1 facet filtering uses `array_contains` natively in
`build_filter`. P3's lane evidence does not need a post-filter step
for facet predicates.

## Probe 2: multivector side-table + bulk-fetch + in-memory MaxSim

**Question**: Can a `List<FixedSizeList<F32, D>>` column be stored in a
Lance side table, bulk-fetched via `chunk_id IN (...)` in a single
query, and decoded into `Vec<Vec<f32>>` per chunk for in-memory MaxSim?

**Method**: `crates/store/examples/multivector_probe.rs` — writes a
1000-row side table with variable-length token-vector lists (3..=7
tokens of dim 8 each), issues a single `chunk_id IN (100 ids)` query,
decodes into `HashMap<String, Vec<Vec<f32>>>`, and verifies
in-memory MaxSim against a hand-computed reference.

**Result**: **PASS.**
- Bulk fetch returned exactly 100 rows in one query (not N).
- Decode round-trips into `Vec<Vec<f32>>` with correct cardinalities.
- MaxSim score (6.346154) matches reference implementation (6.346154).

**Decision**: P4 ships the rerank-only MaxSim feature with side-table
storage + bulk fetch + in-memory MaxSim. The design committed to in
P4 holds against Lance 0.29.0.

## Probe 3: online `add_column` schema evolution

**Question**: Does Lance 0.29.0 support online `add_column` schema
evolution?

**Method**: `crates/store/examples/add_column_probe.rs` — creates a
3-row table with 2 columns, then calls
`table.add_columns(NewColumnTransform::AllNulls(...), None)` to add a
`facets: List<Utf8>` column. Re-opens the table; verifies the column
is in the schema and that all existing rows have NULL values for the
new column.

**Result**: **PASS.**
- `add_columns(AllNulls List<Utf8>)` returns Ok.
- `facets` column present in schema after the add.
- All 3 existing rows have NULL facets values (no rewrite needed).

**Decision**: P10 migration can evolve in-place via `add_columns` for
the columns added in P0/P1 (`source_config_id`, `facets`,
`embedding_input_sha256`). No wipe-and-rescan required for the
schema-evolution path.
