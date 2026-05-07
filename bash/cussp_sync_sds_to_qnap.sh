#!/usr/bin/env bash

set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/data/mseed_archive}"
QNAP_SSH_TARGET="${QNAP_SSH_TARGET:-}"
QNAP_DST_ROOT="${QNAP_DST_ROOT:-}"
STATE_ROOT="${STATE_ROOT:-/home/gmf/.cache/cussp_sds_backup}"
LOG_DIR="${LOG_DIR:-$STATE_ROOT/logs}"
LOCK_FILE="${LOCK_FILE:-$STATE_ROOT/cussp_sync.lock}"
SSH_KEY="${SSH_KEY:-}"
BWLIMIT_KBPS="${BWLIMIT_KBPS:-0}"

require_var() {
    local name="$1"
    local value="$2"

    if [[ -z "$value" ]]; then
        echo "Missing required setting: $name" >&2
        exit 1
    fi
}

mkdir -p "$STATE_ROOT" "$LOG_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Sync already running; exiting"
    exit 0
fi

require_var "QNAP_SSH_TARGET" "$QNAP_SSH_TARGET"
require_var "QNAP_DST_ROOT" "$QNAP_DST_ROOT"

if [[ ! -d "$SRC_ROOT" ]]; then
    echo "Source archive not found: $SRC_ROOT" >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$LOG_DIR/sync_${timestamp}.log"

ssh_args=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
    ssh_args=(-i "$SSH_KEY" "${ssh_args[@]}")
fi

remote_sh() {
    local command="$1"
    ssh "${ssh_args[@]}" "$QNAP_SSH_TARGET" "sh -c $(printf '%q' "$command")"
}

rsync_ssh_cmd="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
if [[ -n "$SSH_KEY" ]]; then
    rsync_ssh_cmd="ssh -i $SSH_KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
fi

rsync_args=(
    --archive
    --hard-links
    --numeric-ids
    --human-readable
    --omit-dir-times
    --partial
    --append-verify
    --stats
    --log-file="$log_file"
    -e "$rsync_ssh_cmd"
)

if [[ "$BWLIMIT_KBPS" != "0" ]]; then
    rsync_args+=(--bwlimit="$BWLIMIT_KBPS")
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Preparing remote target $QNAP_SSH_TARGET:$QNAP_DST_ROOT"
remote_sh "mkdir -p '$QNAP_DST_ROOT' '$QNAP_DST_ROOT/.backup_meta/manifests' '$QNAP_DST_ROOT/.backup_meta/verified'"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting rsync from $SRC_ROOT"
rsync "${rsync_args[@]}" "$SRC_ROOT/" "$QNAP_SSH_TARGET:$QNAP_DST_ROOT/"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sync completed"
echo "Log written to $log_file"