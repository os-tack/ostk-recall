#!/usr/bin/env bash
# Shared-substrate example — scan once, then run ONE daemon for MANY clients.
set -euo pipefail
export EX="$(cd "$(dirname "$0")" && pwd)"
CFG="$EX/config.toml"

ostk-recall --config "$CFG" init
ostk-recall --config "$CFG" scan

cat <<EOF

Start ONE daemon over the shared corpus:
  ostk-recall --config "$CFG" serve

Then attach TWO clients to it — register both in your MCP client config, each as:
  command = ostk-recall
  args    = ["--config", "$CFG", "connect"]
(e.g. servers "recall-A" and "recall-B"). Both bridge to the same daemon and see the same
concepts/threads/chain. See README.md for the shared-vs-partitioned walk — including the
current attention/pin bleed across clients.
EOF
