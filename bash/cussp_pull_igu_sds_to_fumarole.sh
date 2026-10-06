#!/usr/bin/env bash

set -euo pipefail

SOURCE_SSH_TARGET="${SOURCE_SSH_TARGET:-chopp@lrc-xfer}"
SOURCE_ROOT="${SOURCE_ROOT:-/global/scratch/projects/pc_aieqsim/nnakata/Cape/IGU/seiscomp_staging}"
DEST_ROOT="${DEST_ROOT:-/mnt/qnap_hdd/SDS}"
MOUNT_ROOT="${MOUNT_ROOT:-/mnt/qnap_hdd}"
STATE_ROOT="${STATE_ROOT:-${HOME:-/tmp}/.cache/cussp_igu_sds_pull}"
LOG_DIR="${LOG_DIR:-$STATE_ROOT/logs}"
LOCK_FILE="${LOCK_FILE:-$STATE_ROOT/cussp_igu_sds_pull.lock}"
SSH_KEY="${SSH_KEY:-}"
SSH_BATCH_MODE="${SSH_BATCH_MODE:-false}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-$STATE_ROOT/ssh-%C}"
BWLIMIT_KBPS="${BWLIMIT_KBPS:-0}"
DRY_RUN="${DRY_RUN:-true}"

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
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pull already running; exiting" >&2
    exit 0
fi

require_var "SOURCE_SSH_TARGET" "$SOURCE_SSH_TARGET"
require_var "SOURCE_ROOT" "$SOURCE_ROOT"
require_var "DEST_ROOT" "$DEST_ROOT"

case "$DRY_RUN" in
    true|1)
        dry_run=true
        ;;
    false|0)
        dry_run=false
        ;;
    *)
        echo "DRY_RUN must be true, false, 1, or 0 (got: $DRY_RUN)" >&2
        exit 1
        ;;
esac

case "$SSH_BATCH_MODE" in
    true|1)
        ssh_batch_mode=true
        ;;
    false|0)
        ssh_batch_mode=false
        ;;
    *)
        echo "SSH_BATCH_MODE must be true, false, 1, or 0 (got: $SSH_BATCH_MODE)" >&2
        exit 1
        ;;
esac

if ! command -v mountpoint >/dev/null 2>&1; then
    echo "mountpoint is required to verify the QNAP mount" >&2
    exit 1
fi

if [[ "$DEST_ROOT" != "$MOUNT_ROOT" && "$DEST_ROOT" != "$MOUNT_ROOT/"* ]]; then
    echo "Destination must be inside the QNAP mount $MOUNT_ROOT: $DEST_ROOT" >&2
    exit 1
fi

if ! mountpoint -q "$MOUNT_ROOT"; then
    echo "QNAP mount is not mounted: $MOUNT_ROOT" >&2
    exit 1
fi

ssh_args=(-o ControlMaster=auto -o ControlPersist=5m -o "ControlPath=$SSH_CONTROL_PATH" -o StrictHostKeyChecking=accept-new)
if [[ "$ssh_batch_mode" == "true" ]]; then
    ssh_args+=(-o BatchMode=yes)
fi
if [[ -n "$SSH_KEY" ]]; then
    ssh_args=(-i "$SSH_KEY" "${ssh_args[@]}")
fi

remote_sh() {
    local command="$1"
    ssh "${ssh_args[@]}" "$SOURCE_SSH_TARGET" "sh -c $(printf '%q' "$command")"
}

if ! remote_sh "test -d '$SOURCE_ROOT'"; then
    echo "Remote source directory is not accessible: $SOURCE_SSH_TARGET:$SOURCE_ROOT" >&2
    exit 1
fi

if ! mkdir -p "$DEST_ROOT"; then
    echo "Cannot create destination directory: $DEST_ROOT" >&2
    exit 1
fi

if [[ ! -w "$DEST_ROOT" ]]; then
    echo "Destination is not writable: $DEST_ROOT" >&2
    exit 1
fi

available_kb="$(df -Pk "$MOUNT_ROOT" | awk 'NR == 2 {print $4}')"
if [[ ! "$available_kb" =~ ^[0-9]+$ ]] || (( available_kb == 0 )); then
    echo "Unable to confirm free space on QNAP mount: $MOUNT_ROOT" >&2
    exit 1
fi

if [[ "$dry_run" == "true" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Dry run enabled; no files will be copied."
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$LOG_DIR/pull_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

rsync_ssh_cmd="ssh"
if [[ -n "$SSH_KEY" ]]; then
    rsync_ssh_cmd+=" -i $SSH_KEY"
fi
rsync_ssh_cmd+=" -o ControlMaster=auto -o ControlPersist=5m -o ControlPath=$SSH_CONTROL_PATH -o StrictHostKeyChecking=accept-new"
if [[ "$ssh_batch_mode" == "true" ]]; then
    rsync_ssh_cmd+=" -o BatchMode=yes"
fi

rsync_args=(
    --archive
    --no-owner
    --no-group
    --hard-links
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

if [[ "$dry_run" == "true" ]]; then
    rsync_args+=(--dry-run --itemize-changes)
fi

printf '[%s] Pulling from %s:%s to %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_SSH_TARGET" "$SOURCE_ROOT" "$DEST_ROOT"
rsync "${rsync_args[@]}" "$SOURCE_SSH_TARGET:$SOURCE_ROOT/" "$DEST_ROOT/"

if [[ "$dry_run" == "true" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Dry run complete; no files were transferred."
else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pull complete. Log written to $log_file"
fi
