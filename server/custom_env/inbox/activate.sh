#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INBOX="$BASE_DIR/inbox"
ACTIVE="$BASE_DIR/active"
ARCHIVE="$BASE_DIR/archive"

mkdir -p "$ARCHIVE"

# --- active 内の JSON を archive へ退避 ---
shopt -s nullglob
active_jsons=( "$ACTIVE"/*.json )
shopt -u nullglob

if (( ${#active_jsons[@]} > 0 )); then
  echo "[activate] Archiving ${#active_jsons[@]} active JSON(s) -> $ARCHIVE"
  mv "$ACTIVE"/*.json "$ARCHIVE"/
fi

# --- inbox 内の JSON を確認 ---
shopt -s nullglob
inbox_jsons=( "$INBOX"/*.json )
shopt -u nullglob

if (( ${#inbox_jsons[@]} == 0 )); then
  echo "[activate] Error: inbox に JSON がありません。"
  exit 1
fi
if (( ${#inbox_jsons[@]} > 1 )); then
  echo "[activate] Error: inbox に JSON が複数あります。1件のみ残してください。"
  for f in "${inbox_jsons[@]}"; do echo " - $f"; done
  exit 1
fi

# --- inbox の1件を active へ移動 ---
JSON_FILE="${inbox_jsons[0]}"
echo "[activate] Activating $(basename "$JSON_FILE")"
mv "$JSON_FILE" "$ACTIVE/"

echo "[activate] Done."