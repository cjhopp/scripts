#!/usr/bin/env python3
"""One-time repair for a DTS_all.zarr store with mismatched 'time' lengths.

combine_XTDTS.py appends one timestep at a time to /data/chet-cussp/DTS/DTS_all.zarr.
If that writer is killed mid-append (systemd restart, or resuming after a long
outage of the upstream DTS source), individual arrays in the store can end up
with different lengths along the 'time' dimension. xr.open_dataset() then
raises "conflicting sizes for dimension 'time'" on every read.

This script finds every array in the store with a 'time' dimension (using the
'_ARRAY_DIMENSIONS' attribute xarray's zarr backend writes on each array),
computes the shortest common length, and truncates the longer arrays down to
that length so the store is internally consistent again.

Usage:
    # Inspect only, makes no changes (default):
    python3 repair_dts_zarr.py

    # Actually truncate the mismatched arrays:
    python3 repair_dts_zarr.py --apply

IMPORTANT: stop the ingest service first so it isn't writing to the store
concurrently:
    systemctl stop cussp-dts-combine
    python3 repair_dts_zarr.py --apply
    systemctl start cussp-dts-combine
"""
import argparse

import zarr

DEFAULT_ZARR_PATH = "/data/chet-cussp/DTS/DTS_all.zarr"


def find_time_lengths(group):
    """Return {array_name: (time_axis_index, length_along_time)} for every
    array in the group that has a 'time' dimension."""
    lengths = {}
    for name, arr in group.arrays():
        dims = arr.attrs.get("_ARRAY_DIMENSIONS")
        if not dims or "time" not in dims:
            continue
        axis = dims.index("time")
        lengths[name] = (axis, arr.shape[axis])
    return lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr-path", default=DEFAULT_ZARR_PATH, help=f"Path to the Zarr store (default: {DEFAULT_ZARR_PATH})")
    parser.add_argument("--apply", action="store_true", help="Actually truncate mismatched arrays (default is dry-run/report only)")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt when used with --apply")
    args = parser.parse_args()

    group = zarr.open_group(args.zarr_path, mode="r")
    lengths = find_time_lengths(group)
    if not lengths:
        print(f"No arrays with a 'time' dimension found in {args.zarr_path}")
        return

    min_len = min(length for _, length in lengths.values())

    print(f"Zarr store: {args.zarr_path}")
    print(f"{'array':<28}{'time length':>14}{'action':>16}")
    mismatched = []
    for name, (axis, length) in sorted(lengths.items()):
        if length > min_len:
            action = f"truncate -{length - min_len}"
            mismatched.append((name, axis, length))
        else:
            action = "ok"
        print(f"{name:<28}{length:>14}{action:>16}")

    if not mismatched:
        print(f"\nStore is already consistent at {min_len} timesteps. Nothing to do.")
        return

    print(f"\n{len(mismatched)} array(s) will be truncated to {min_len} timesteps.")

    if not args.apply:
        print("\nDry run only — no changes made. Re-run with --apply to truncate.")
        return

    if not args.yes:
        reply = input(
            f"\nThis will permanently truncate {len(mismatched)} array(s) in {args.zarr_path}.\n"
            "Make sure cussp-dts-combine is stopped first. Continue? [y/N] "
        )
        if reply.strip().lower() != "y":
            print("Aborted.")
            return

    rw_group = zarr.open_group(args.zarr_path, mode="r+")
    for name, axis, length in mismatched:
        arr = rw_group[name]
        new_shape = list(arr.shape)
        new_shape[axis] = min_len
        arr.resize(tuple(new_shape))
        print(f"Truncated {name}: {length} -> {min_len}")

    zarr.consolidate_metadata(args.zarr_path)
    print("\nDone. Re-consolidated metadata. Restart cussp-dts-combine and cussp-dts now.")


if __name__ == "__main__":
    main()
