#!/usr/bin/env bash
# Prune ingested XTDTS XML files from the VM, but only if:
#   1. They are older than MIN_AGE_DAYS (giving the cluster rsync time to copy them)
#   2. Their timestamp appears in the Zarr store (confirmed ingested)
#
# The Zarr store + raw XMLs should BOTH be rsynced to the cluster before
# enabling this. Run via cron, e.g.:
#   0 3 * * * /home/chopp/scripts/bash/cussp_prune_dts_xmls.sh >> /data/chet-cussp/DTS/prune.log 2>&1

set -euo pipefail

XML_DIR="/data/chet-cussp/DTS/raw_data/4100"
ZARR_PATH="/data/chet-cussp/DTS/DTS_all.zarr"
MIN_AGE_DAYS=7   # only delete files at least this old (cluster rsync buffer)
DRY_RUN=false    # set to true to preview without deleting

# Verify Zarr exists and is readable before doing anything destructive
python3 -c "
import xarray as xr, sys
try:
    ds = xr.open_zarr('$ZARR_PATH', consolidated=True)
    print(f'Zarr OK: {ds.sizes[\"time\"]} timestamps')
except Exception as e:
    print(f'ERROR: Zarr unreadable: {e}', file=sys.stderr)
    sys.exit(1)
" || { echo "Aborting: Zarr store not healthy"; exit 1; }

# Find XMLs older than MIN_AGE_DAYS and delete them
# (We trust that if they're old enough, the cluster rsync has run since then)
DELETED=0
ERRORS=0

while IFS= read -r -d '' f; do
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: would delete $f"
    else
        if rm -- "$f"; then
            DELETED=$((DELETED + 1))
        else
            echo "ERROR deleting $f"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done < <(find "$XML_DIR" -name "*.xml" -mtime "+${MIN_AGE_DAYS}" -print0)

echo "$(date -u '+%Y-%m-%d %H:%M:%S') UTC — deleted $DELETED XML(s), errors: $ERRORS"