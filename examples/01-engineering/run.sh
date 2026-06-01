#!/usr/bin/env bash
# Engineering example — scan code + decisions into a throwaway corpus.
set -euo pipefail
export EX="$(cd "$(dirname "$0")" && pwd)"
CFG="$EX/config.toml"

ostk-recall --config "$CFG" init
ostk-recall --config "$CFG" scan

cat <<EOF

Scanned into $EX/.recall
Query it:
  ostk-recall --config "$CFG" serve        # MCP daemon, in one terminal
  # then register an MCP client: command=ostk-recall, args=[--config, $CFG, connect]
  # and call:  memory_recall(query="why did we derive conductance instead of storing it")
EOF
