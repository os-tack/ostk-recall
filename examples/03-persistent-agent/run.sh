#!/usr/bin/env bash
# Persistent-agent example — seed the agent's starting memory from files.
set -euo pipefail
export EX="$(cd "$(dirname "$0")" && pwd)"
CFG="$EX/config.toml"
DB="$EX/.recall/threads.sqlite"

ostk-recall --config "$CFG" init
ostk-recall --config "$CFG" scan

echo
echo "=== seeded note nodes ==="
sqlite3 "$DB" "select handle, kind, status from concepts order by handle;"

cat <<EOF

The files above are the agent's STARTING memory. At runtime the agent grows it through
MCP writes (see README): memory_remember(kind=...) and memory_connect(from,relation,to).
Those writes persist in $DB and survive a daemon restart — that's the "persistent" part.

  ostk-recall --config "$CFG" serve
EOF
