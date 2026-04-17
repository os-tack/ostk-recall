# tests/fixtures/

Read-only fixtures used by the recall end-to-end tests and by the
verification panel (`crates/cli/tests/verification_panel.rs`).

The panel reads [`tests/queries.yaml`](../queries.yaml). Each query
targets a distinctive keyword planted in exactly one fixture so BM25
retrieval pins the expected chunk. **Do not rewrite these files
without updating `queries.yaml`** — the panel will silently regress.

| scanner       | fixture path                                                          | pinned keyword(s)                                  | query that exercises it                                 |
|---------------|-----------------------------------------------------------------------|----------------------------------------------------|---------------------------------------------------------|
| markdown      | `markdown/notes/topics/rebase.md`                                     | `tier2_line_rebase`                                | `markdown cross-heading retrieval`                      |
| code          | `code/arithmetic.rs`                                                  | `compute_lagrangian_delta`, `invert_hoberman_matrix` | `code identifier retrieval`                             |
| claude_code   | `claude_code/-Users-x-projects-demo/session-memory-model.jsonl`       | `hoberman shared memory`                           | `claude_code conversation recall`                       |
| file_glob     | `file_glob/alpha.txt`                                                 | `fleet_heartbeat_skew`                             | `file_glob alpha keyword`                               |
| file_glob     | `file_glob/sub/beta.txt`                                              | `arboreal_scheduler`, `canopy_checkpoint`          | `file_glob nested folder recursion`                     |
| zip_export    | `zip_export/conversations.json` (zipped at test time)                 | `trapezoidal_integrator`                           | `zip_export message recall`, `recall_link finds parent` |
| ostk_project/decisions | `ostk_project/.ostk/decisions.jsonl`                         | `big.LITTLE scheduler`                             | `ostk_project decision`                                 |
| ostk_project/needles   | `ostk_project/.ostk/needles/issues.jsonl`                    | `hoberman_panel_needle`                            | `ostk_project needle panel marker`                      |
| ostk_project/audit     | `ostk_project/.ostk/audit.jsonl`                             | 2 `success=false` rows                             | `recall_audit counts failures`                          |
| ostk_project/conversations | `ostk_project/.ostk/conversations/panel.jsonl`           | `panel_conversation_marker`                        | `ostk_project conversation handoff`                     |
| ostk_project/sessions  | `ostk_project/.ostk/sessions/session-panel.jsonl`            | `panel_session_marker`                             | `ostk_project session marker`                           |
| ostk_project/spec      | `ostk_project/docs/spec/overview.md`                         | `panel_spec_anchor`                                | `ostk_project spec anchor`                              |

## Why fixtures are small

All fixtures are kept under ~2 KB so the panel is fast and the repo
doesn't bloat. The `zip_export` fixture is a plain `conversations.json`
checked in at rest; the test zips it into a tempdir before running the
scanner.

## Known limitation: BM25-only ranking with the fake embedder

`verification_panel` uses a `FakeEmbedder` with length-bucket vectors, so
the dense side of the hybrid ranker is effectively a no-op and BM25 FTS
drives placement. Queries are written to be keyword-matchable.

A second, gated test `verification_panel_semantic` exercises the same
YAML with the real embedder (set `OSTK_RECALL_E2E=1` to run). That
variant is optional and is skipped by default.
