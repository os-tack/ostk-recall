#!/usr/bin/env bash
# Personal-KB example — typed nodes + authored edges (slice 3), prose mentions
# (slice 4), and automagic promotion + crystallize (slice 5).
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
echo "=== edges (authored from frontmatter + observed from prose) ==="
sqlite3 "$DB" \
  "select c1.handle, e.relation, c2.handle, e.source, e.[by], e.confidence, e.touch_count
     from concept_edges e
     join concepts c1 on c1.id = e.from_concept
     join concepts c2 on c2.id = e.to_concept
    order by c1.handle, e.relation;"

echo
echo "=== slice 5: automagic promotion → crystallize ==="
# 'Priya' is named in three person bodies (mike, sarah, tori) but has no file.
# The scan recognized the recurrence and minted a context-typed PROPOSED node
# (see priya|person|proposed above) wired by observed mentions edges. Now
# confirm it into a real file under the person source — propose, then confirm.
ostk-recall --config "$CFG" crystallize priya --project memories
echo "--- generated content/people/priya.md ---"
cat "$EX/content/people/priya.md"
# Keep the example idempotent: drop the stub so a re-run reproduces the
# promotion. A real workflow keeps it — the next scan then ingests priya as a
# first-class, content-bearing node (no longer just a proposal).
rm -f "$EX/content/people/priya.md"

cat <<EOF

Expect: tori|person, sarah|person, mike|person, office|place, the standup meeting node,
ostk-recall (untyped — only an edge target, no file of its own), and — NEW (slice 5) —
priya|person|proposed, promoted from recurring prose (3 person docs name 'Priya'), no file yet.

Authored edges (from frontmatter; source=authored, by=scanner, confidence~0.1):
  tori --families--> sarah        sarah --families--> tori
  tori --works_on--> ostk-recall  mike --works_on--> ostk-recall
  <standup> --people--> tori, mike    <standup> --places--> office

Observed edges (slices 4 + 5 — bare prose mentions; source=observed, relation=mentions):
  tori --mentions--> sarah, ostk-recall, priya
  sarah --mentions--> tori, priya
  mike --mentions--> tori, ostk-recall, priya   # mike->tori has no authored counterpart
  <standup> --mentions--> tori, mike, office, ostk-recall

Slice 5: 'Priya' recurred across 3 person bodies → minted as a proposed person node +
observed mentions edges from each mentioning doc, then 'crystallize' wrote its stub file
(propose, never auto-write). Then query the live graph — see smoke.md.
EOF
