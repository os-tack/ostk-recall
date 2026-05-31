#!/usr/bin/env bash
# recall-backup.sh — on-box, tiered checkpoint of the ostk-recall stores.
#
# Strategy (see the backup design discussion):
#   Tier 0  threads.sqlite + events.sqlite   irreplaceable earned cognition.
#                                            Always snapshotted. Fast (~2s).
#   Tier 2  ingest.sqlite                    derivable (manifest recover) but
#                                            cheap to carry; makes a restore
#                                            self-sufficient. Always included.
#   Tier 1  corpus.lance/                    4.6G, slow to rebuild. Only with
#                                            --corpus (APFS COW clone, ~instant).
#
# SQLite stores use `VACUUM INTO` — a consistent hot snapshot that is safe
# while `serve` is running (WAL readers don't block writers) and arrives
# defragmented + integrity-checkable. The corpus is an APFS clonefile copy
# (cp -c): near-zero extra disk until the original diverges.
#
# Usage:
#   recall-backup.sh                      # Tier 0 + Tier 2 (pre-weave default)
#   recall-backup.sh --corpus             # + corpus clone (full / pre-optimize)
#   recall-backup.sh --label pre-weave    # tag the checkpoint dir
#   recall-backup.sh --keep 14            # prune oldest beyond N (default: keep all)
#
# Env:
#   RECALL_ROOT   source dir   (default ~/.local/share/ostk-recall)
#   BACKUP_ROOT   dest parent  (default ~/.local/share/ostk-recall-backups)
set -euo pipefail

RECALL_ROOT="${RECALL_ROOT:-$HOME/.local/share/ostk-recall}"
BACKUP_ROOT="${BACKUP_ROOT:-$HOME/.local/share/ostk-recall-backups}"
WITH_CORPUS=0
LABEL=""
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus)     WITH_CORPUS=1; shift ;;
    --label)      LABEL="$2"; shift 2 ;;
    --keep)       KEEP="$2"; shift 2 ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$RECALL_ROOT" ]] || { echo "recall root not found: $RECALL_ROOT" >&2; exit 1; }
command -v sqlite3 >/dev/null || { echo "sqlite3 not on PATH" >&2; exit 1; }

STAMP="$(date +%Y-%m-%d_%H%M%S)"
SUFFIX="${LABEL:+_$LABEL}"
DST="$BACKUP_ROOT/${STAMP}${SUFFIX}"
mkdir -p "$DST"
echo "checkpoint -> $DST"

# --- SQLite tiers: VACUUM INTO + integrity check ---------------------------
SQLITE_DBS=(threads events ingest)
for db in "${SQLITE_DBS[@]}"; do
  src="$RECALL_ROOT/${db}.sqlite"
  [[ -f "$src" ]] || { echo "  [$db] SKIP (missing)"; continue; }
  sqlite3 "$src" "VACUUM INTO '$DST/${db}.sqlite'"
  ok="$(sqlite3 "$DST/${db}.sqlite" 'PRAGMA integrity_check;')"
  sz="$(du -h "$DST/${db}.sqlite" | cut -f1)"
  echo "  [$db] $sz  integrity:$ok"
  [[ "$ok" == "ok" ]] || { echo "  [$db] INTEGRITY FAILED — aborting" >&2; exit 1; }
done

# --- Tier 1 corpus: APFS clone (opt-in) ------------------------------------
if [[ "$WITH_CORPUS" == 1 ]]; then
  src="$RECALL_ROOT/corpus.lance"
  if [[ -d "$src" ]]; then
    if df -t apfs "$RECALL_ROOT" >/dev/null 2>&1; then
      cp -cR "$src" "$DST/corpus.lance"; mode="APFS clone"
    else
      cp -R "$src" "$DST/corpus.lance"; mode="plain copy"
    fi
    echo "  [corpus] $(du -sh "$DST/corpus.lance" | cut -f1) logical ($mode)"
  else
    echo "  [corpus] SKIP (missing)"
  fi
fi

# --- manifest --------------------------------------------------------------
{
  echo "ostk-recall checkpoint"
  echo "created:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "label:     ${LABEL:-none}"
  echo "corpus:    $([[ "$WITH_CORPUS" == 1 ]] && echo included || echo "NOT included (Tier 0/2 only)")"
  echo "restore:   stop serve; cp -cR each artifact back into $RECALL_ROOT/; reboot serve."
} > "$DST/MANIFEST.txt"

# --- retention -------------------------------------------------------------
if [[ "$KEEP" -gt 0 ]]; then
  mapfile -t old < <(ls -1dt "$BACKUP_ROOT"/*/ 2>/dev/null | tail -n +$((KEEP+1)))
  for d in "${old[@]:-}"; do
    [[ -n "$d" ]] || continue
    echo "  [prune] $d"; rm -rf "$d"
  done
fi

echo "done: $(du -sh "$DST" | cut -f1) logical at $DST"
