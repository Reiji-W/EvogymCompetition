#!/usr/bin/env bash
set -euo pipefail

# === 設定ファイル ===
CONFIG="client/config/remote.yaml"

# === remote.yaml から remote ブロックを Bash だけで抽出 ===
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

# remote: ブロックだけを抜き出し、コメント(#)を除去
REMOTE_BLOCK="$(sed -n '/^remote:/,/^[^[:space:]]/p' "$CONFIG" | sed -E 's/#.*//')"

yaml_val () {
  # 使い方: yaml_val "key"
  printf '%s\n' "$REMOTE_BLOCK" \
    | grep -E "^[[:space:]]*$1:" \
    | head -n1 \
    | sed -E "s/^[[:space:]]*$1:[[:space:]]*//" \
    | tr -d '"'"'"
}

HOST="$(yaml_val host)"
USER="$(yaml_val user)"
KEY="$(yaml_val ssh_key)"
RPATH="$(yaml_val path)"

if [[ -z "${HOST:-}" || -z "${USER:-}" || -z "${RPATH:-}" ]]; then
  echo "remote.yaml の remote: { host, user, path } を確認してください。" >&2
  exit 1
fi

# 鍵ファイルは任意。未指定なら ssh の通常設定(~/.ssh/config 等)に任せる
SSH_OPT=(-o IdentitiesOnly=yes -o BatchMode=yes)
if [[ -n "${KEY:-}" && "${KEY:-}" != "null" && -f "$KEY" ]]; then
  SSH_OPT=(-i "$KEY" "${SSH_OPT[@]}")
fi

# === rsync のソース/宛先設定 ===
# 末尾のスラッシュが「中身だけ」を同期するポイント！
RSRC="${USER}@${HOST}:${RPATH%/}/server/saved_data/"
RDST="client/mnt/"

mkdir -p "$RDST"

echo "-> Syncing ONLY ${USER}@${HOST}:${RPATH%/}/server/saved_data/ -> ${RDST}"
rsync -az --delete \
  -e "ssh ${SSH_OPT[*]}" \
  "$RSRC" "$RDST"

echo "Done. (synced contents into ${RDST})"