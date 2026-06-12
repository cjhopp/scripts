#!/usr/bin/env bash
set -euo pipefail

# Robust sync for CUSSP CASSM archive from Google Drive via rclone.
#
# Default behavior:
# - resumes partial transfers
# - retries failed chunks/files
# - keeps destination in sync with source
# - writes a transfer log
#
# Usage:
#   bash cussp_sync_cassm_sigmav_rclone.sh \
#     <remote_path> \
#     /media/chopp/HDD1/chet-cussp/wavs/cassm/sigmav_archive
#
# Example remote_path:
#   gdrive:chet-cussp/wavs/cassm/sigmav_archive

REMOTE_PATH="${1:-}"
DEST_DIR="${2:-/media/chopp/HDD1/chet-cussp/wavs/cassm/sigmav_archive}"
# Space-separated set of VALID source (dat file) counts per epoch.
# Current config:  16 sources (TS-well sources decommissioned, ~2026-05)
# Previous config: 20 sources (full complement)
# Any epoch NOT in this set is flagged as INCOMPLETE; any count in the set is EXPECTED.
VALID_DAT_COUNTS="${VALID_DAT_COUNTS:-16 20}"
LOG_FILE="${LOG_FILE:-${DEST_DIR%/}/rclone_sync.log}"
DRIVE_SHARED_WITH_ME="${DRIVE_SHARED_WITH_ME:-1}"

if [[ -z "${REMOTE_PATH}" ]]; then
  echo "ERROR: remote path required" >&2
  echo "Usage: $0 <remote_path> [dest_dir]" >&2
  exit 2
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "ERROR: rclone not found in PATH" >&2
  exit 3
fi

mkdir -p "${DEST_DIR}"

RCLONE_SHARED_FLAG=()
if [[ "${DRIVE_SHARED_WITH_ME}" == "1" ]]; then
  RCLONE_SHARED_FLAG=(--drive-shared-with-me)
fi

echo "[$(date -Is)] Sync start" | tee -a "${LOG_FILE}"
echo "  remote: ${REMOTE_PATH}" | tee -a "${LOG_FILE}"
echo "  dest:   ${DEST_DIR}" | tee -a "${LOG_FILE}"

# copy is safer than sync while source is still being populated.
# If you want strict mirror semantics, replace 'copy' with 'sync'.
rclone copy "${REMOTE_PATH}" "${DEST_DIR}" \
  --drive-acknowledge-abuse \
  "${RCLONE_SHARED_FLAG[@]}" \
  --checkers 16 \
  --transfers 8 \
  --retries 20 \
  --retries-sleep 10s \
  --low-level-retries 30 \
  --contimeout 30s \
  --timeout 10m \
  --stats 20s \
  --stats-one-line \
  --fast-list \
  --create-empty-src-dirs \
  --metadata \
  --log-file "${LOG_FILE}" \
  --log-level INFO

# Optional integrity pass using size+modtime only (fast and practical for Drive).
rclone check "${REMOTE_PATH}" "${DEST_DIR}" \
  "${RCLONE_SHARED_FLAG[@]}" \
  --size-only \
  --one-way \
  --checkers 16 \
  --log-file "${LOG_FILE}" \
  --log-level INFO || true

# Quick local completeness scan by epoch folder.
echo "[$(date -Is)] Epoch completeness check (valid .dat counts: ${VALID_DAT_COUNTS})" | tee -a "${LOG_FILE}"
declare -A _count_hist
while IFS= read -r epoch_dir; do
  n_dat=$(find "${epoch_dir}" -maxdepth 1 -type f -name '*.dat' | wc -l)
  _count_hist["${n_dat}"]=$(( ${_count_hist["${n_dat}"]:-0} + 1 ))
  is_valid=0
  for vc in ${VALID_DAT_COUNTS}; do
    if [[ "${n_dat}" -eq "${vc}" ]]; then is_valid=1; break; fi
  done
  if [[ "${is_valid}" -eq 0 ]]; then
    echo "INCOMPLETE ${epoch_dir##*/}: ${n_dat} dat files (expected one of: ${VALID_DAT_COUNTS})" | tee -a "${LOG_FILE}"
  fi
done < <(find "${DEST_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)

# Emit a histogram so operator can see the config distribution at a glance.
echo "[$(date -Is)] Source-count histogram (dat files per epoch):" | tee -a "${LOG_FILE}"
for n in $(echo "${!_count_hist[@]}" | tr ' ' '\n' | sort -n); do
  is_valid=0
  for vc in ${VALID_DAT_COUNTS}; do
    if [[ "${n}" -eq "${vc}" ]]; then is_valid=1; break; fi
  done
  label="OK"
  [[ "${is_valid}" -eq 0 ]] && label="UNEXPECTED"
  echo "  ${n} dat files: ${_count_hist[${n}]} epoch(s)  [${label}]" | tee -a "${LOG_FILE}"
done

echo "[$(date -Is)] Sync finished" | tee -a "${LOG_FILE}"
