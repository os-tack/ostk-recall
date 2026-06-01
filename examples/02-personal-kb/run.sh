#!/usr/bin/env bash
# Personal-KB example — seed typed nodes + authored edges from frontmatter.
set -euo pipefail
export EX="$(cd "$(dirname "$0")" && pwd)"
CFG="$EX/config.toml"
DB="$EX/.recall/threads.sqlite"

ostk-recall --config "$CFG" init
ostk-recall --config "$CFG" scan

echo
echo "=== concepts (typed nodes) ==="
sqlite3 "$DB" "select project, handle, kind, status from concepts order by handle;"

echo
echo "=== authored edges ==="
sqlite3 "$DB" \
  "select c1.handle, e.relation, c2.handle, e.source, e.[by], e.confidence, e.touch_count
     from concept_edges e
     join concepts c1 on c1.id = e.from_concept
     join concepts c2 on c2.id = e.to_concept
    order by c1.handle, e.relation;"

cat <<EOF

Expect: tori|person, sarah|person, mike|person, office|place, the standup meeting node,
and ostk-recall (untyped — it is only an edge target, no file of its own). Edges (all
source=authored, by=scanner, confidence~0.1):
  tori --families--> sarah
  sarah --families--> tori
  tori --works_on--> ostk-recall
  mike --works_on--> ostk-recall
  <standup> --people--> tori, mike
  <standup> --places--> office

Then query the live graph — see smoke.md.
EOF
