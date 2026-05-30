#!/usr/bin/env bash
# Regenerate / document the P5 rank-bench fixtures.
#
# These fixtures are hand-authored, not machine-generated — this script
# documents what they are and how to (re)produce a real-corpus run. It is
# intentionally not a one-shot generator: the CI fixture is curated to match
# the synthetic corpus in `crates/cli/examples/rank_bench.rs`, and the real
# query set is the maintainer's judgment call.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../../.." && pwd)"

cat <<'EOF'
P5 rank-bench fixtures
======================

queries.json     CI-safe synthetic query set. Relevance is labeled by
                 `source_id` against the deterministic in-memory corpus that
                 `rank_bench --fixture` builds. Edit alongside that corpus.

lens_turns.jsonl Simulated attention timeline (one focus update per line) for
                 the lens rotation / refractory metric. Each line:
                   {"turn": N, "ts": RFC3339, "focus_text": "..."}

Run the CI-safe bench (no model download, deterministic):
  HF_HUB_OFFLINE=1 cargo run -p ostk-recall-cli --example rank_bench -- --fixture

Run against a REAL corpus (operator step — produces docs/rank-bench-results.md):
  1. Author ~50 queries in queries.json labeled by source_id against your corpus.
  2. Stop `serve` (one writer on corpus.lance).
  3. cargo run --release -p ostk-recall-cli --example rank_bench -- \
       --corpus ~/.local/share/ostk-recall \
       --model potion-retrieval-32M \
       --queries tests/fixtures/bench/queries.json \
       --out docs/rank-bench-results.md
EOF

echo
echo "fixtures present under: $here"
ls -1 "$here"
echo "repo root: $repo_root"
