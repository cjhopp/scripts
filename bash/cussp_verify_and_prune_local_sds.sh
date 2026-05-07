#!/usr/bin/env bash

set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/data/mseed_archive}"
QNAP_SSH_TARGET="${QNAP_SSH_TARGET:-}"
QNAP_DST_ROOT="${QNAP_DST_ROOT:-}"
STATE_ROOT="${STATE_ROOT:-/home/gmf/.cache/cussp_sds_backup}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$STATE_ROOT/manifests}"
LOG_DIR="${LOG_DIR:-$STATE_ROOT/logs}"
LOCK_FILE="${LOCK_FILE:-$STATE_ROOT/cussp_verify_prune.lock}"
SSH_KEY="${SSH_KEY:-}"
KEEP_DAYS="${KEEP_DAYS:-10}"
VERIFY_MIN_AGE_DAYS="${VERIFY_MIN_AGE_DAYS:-1}"
DRY_RUN="${DRY_RUN:-true}"

require_var() {
    local name="$1"
    local value="$2"

    if [[ -z "$value" ]]; then
        echo "Missing required setting: $name" >&2
        exit 1
    fi
}

mkdir -p "$STATE_ROOT" "$MANIFEST_ROOT" "$LOG_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "Verify/prune already running; exiting"
    exit 0
fi

require_var "QNAP_SSH_TARGET" "$QNAP_SSH_TARGET"
require_var "QNAP_DST_ROOT" "$QNAP_DST_ROOT"

if [[ ! -d "$SRC_ROOT" ]]; then
    echo "Source archive not found: $SRC_ROOT" >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$LOG_DIR/verify_prune_${timestamp}.log"
exec > >(tee -a "$log_file") 2>&1

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

archive_days() {
    (
        cd "$SRC_ROOT"
        find . -type f -regextype posix-extended -regex '.*\.[0-9]{4}\.[0-9]{3}$' -printf '%P\n' |
            awk -F. '{print $(NF-1), $NF}' |
            sort -u
    )
}

day_epoch() {
    local year="$1"
    local julian_day="$2"

    date -u -d "${year}-01-01 +$((10#$julian_day - 1)) days" +%s
}

day_age() {
    local year="$1"
    local julian_day="$2"
    local now_epoch

    now_epoch="$(date -u +%s)"
    echo $(((now_epoch - $(day_epoch "$year" "$julian_day")) / 86400))
}

manifest_path() {
    local year="$1"
    local julian_day="$2"

    echo "$MANIFEST_ROOT/$year/$julian_day.sha256"
}

build_manifest() {
    local year="$1"
    local julian_day="$2"
    local manifest
    local manifest_tmp

    manifest="$(manifest_path "$year" "$julian_day")"
    manifest_tmp="$manifest.tmp"
    mkdir -p "$(dirname "$manifest")"

    (
        cd "$SRC_ROOT"
        find "$year" -type f -name "*.${year}.${julian_day}" -print0 |
            sort -z |
            xargs -0 -r sha256sum
    ) > "$manifest_tmp"

    if [[ ! -s "$manifest_tmp" ]]; then
        rm -f "$manifest_tmp"
        return 1
    fi

    mv "$manifest_tmp" "$manifest"
}

upload_and_verify_manifest() {
    local year="$1"
    local julian_day="$2"
    local manifest

    manifest="$(manifest_path "$year" "$julian_day")"

    remote_sh "mkdir -p '$QNAP_DST_ROOT/.backup_meta/manifests/$year' '$QNAP_DST_ROOT/.backup_meta/verified/$year'"
    rsync -a -e "$rsync_ssh_cmd" "$manifest" "$QNAP_SSH_TARGET:$QNAP_DST_ROOT/.backup_meta/manifests/$year/$julian_day.sha256"

    remote_sh "
        set -eu
        cd '$QNAP_DST_ROOT'
        sha256sum -c '.backup_meta/manifests/$year/$julian_day.sha256' >/dev/null
        date -u '+%Y-%m-%dT%H:%M:%SZ' > '.backup_meta/verified/$year/$julian_day.ok'
    "
}

remote_verified() {
    local year="$1"
    local julian_day="$2"

    remote_sh "test -s '$QNAP_DST_ROOT/.backup_meta/verified/$year/$julian_day.ok'"
}

prune_local_day() {
    local year="$1"
    local julian_day="$2"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: would prune local files for ${year}.${julian_day}"
        return 0
    fi

    find "$SRC_ROOT/$year" -type f -name "*.${year}.${julian_day}" -delete
    find "$SRC_ROOT/$year" -depth -type d -empty -delete
    echo "Pruned local files for ${year}.${julian_day}"
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting verify/prune pass"
echo "DRY_RUN=$DRY_RUN KEEP_DAYS=$KEEP_DAYS VERIFY_MIN_AGE_DAYS=$VERIFY_MIN_AGE_DAYS"

while read -r year julian_day; do
    [[ -n "$year" ]] || continue

    age_days="$(day_age "$year" "$julian_day")"
    echo "Inspecting ${year}.${julian_day} (age ${age_days} day(s))"

    if (( age_days >= VERIFY_MIN_AGE_DAYS )); then
        if remote_verified "$year" "$julian_day"; then
            echo "Already verified on QNAP: ${year}.${julian_day}"
        else
            echo "Building manifest for ${year}.${julian_day}"
            if build_manifest "$year" "$julian_day"; then
                echo "Uploading and verifying manifest for ${year}.${julian_day}"
                upload_and_verify_manifest "$year" "$julian_day"
            else
                echo "No files found for ${year}.${julian_day}; skipping"
                continue
            fi
        fi
    fi

    if (( age_days >= KEEP_DAYS )); then
        if remote_verified "$year" "$julian_day"; then
            prune_local_day "$year" "$julian_day"
        else
            echo "Refusing to prune ${year}.${julian_day}: not verified on QNAP"
        fi
    fi
done < <(archive_days)

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Verify/prune pass completed"