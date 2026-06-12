#!/usr/bin/env python
"""
Setup script for NLLoc Grid2Time and location of Newberry DAS catalog events.

Workflow:
  1. Build a unique channel table (CHAN, distance_m, mapped_MD) from all DAS
     pick CSVs.  x/y/z from the picks CSVs are NOT used.
  2. Interpolate easting, northing, elevation (metres ASL) from the borehole
     deviation survey at each channel's mapped_MD.
  3. Subsample channels to every ~10 m of fiber distance.
  4. Convert UTM Zone 10N -> WGS84; write GTSRCE lines into the NLLoc config.
  5. For each event CSV write a NLLOC_OBS obs file, keeping only picks on
     subsampled channels.

Usage:
  python das_nlloc_setup.py [--spacing 10] [--csv-dir ...] [--obs-dir ...]
                            [--nlloc-cfg ...]

After this script, run:
  cd /home/chopp/NLLoc/Newberry_DAS/run
  Grid2Time locate_newberry_DAS.nlloc   # P then S pass
  # Then per event:
  NLLoc locate_newberry_DAS.nlloc       # after patching LOCFILES
"""

import argparse
import os
import re
from glob import glob

import numpy as np
import pandas as pd
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CSV_DIR = "/media/chopp/HDD1/chet-meq/newberry/catalogs/DAS/phase_2000_2660"
DEVIATION_CSV = (
    "/media/chopp/HDD1/chet-meq/newberry/boreholes/55-29/GDR_submission/"
    "Deviation_corrected_with-depth_w-TD.csv"
)
NLLOC_CFG = "/home/chopp/NLLoc/Newberry_DAS/run/locate_newberry_DAS.nlloc"
OBS_DIR = "/home/chopp/NLLoc/Newberry_DAS/obs"
SPACING_M = 10.0  # subsample spacing in meters of fiber distance

# Default Grid2Time output root (basename, no extension).
# Derived from the GTFILES line in the NLLoc config; can be overridden with
# --grid-root on the command line.
GRID_ROOT = (
    "/media/chopp/HDD1/chet-meq/newberry/vmods/NLL_grids/Newberry_1d-topo"
)

# Epoch date used for all obs files (NLLoc solves origin time as free param)
EPOCH_DATE = "20010101"
EPOCH_HHMM = "0000"

# Pick time uncertainty in seconds (Gaussian)
P_ERROR_S = 1.0e-3
S_ERROR_S = 2.0e-3


# ---------------------------------------------------------------------------
# Step 1: Build unique channel table
# ---------------------------------------------------------------------------
def build_channel_table(csv_dir: str) -> pd.DataFrame:
    """
    Read all pick CSVs and return a DataFrame of unique channels with their
    along-fiber distance and measured depth.  Columns retained: CHAN,
    distance_m, mapped_MD.  x/y/z are NOT taken from the picks CSV; they
    are interpolated from the deviation survey in Step 2.
    """
    frames = []
    for path in sorted(glob(os.path.join(csv_dir, "*_picks.csv"))):
        df = pd.read_csv(path, usecols=["CHAN", "distance_m", "mapped_MD"])
        frames.append(df.drop_duplicates(subset="CHAN"))
    combined = pd.concat(frames, ignore_index=True)
    channel_table = (
        combined.drop_duplicates(subset="CHAN")
        .sort_values("CHAN")
        .reset_index(drop=True)
    )
    print(
        f"  Found {len(channel_table)} unique DAS channels "
        f"(CHAN {channel_table['CHAN'].min()}–{channel_table['CHAN'].max()})"
    )
    print(
        f"  mapped_MD range: {channel_table['mapped_MD'].min():.1f} – "
        f"{channel_table['mapped_MD'].max():.1f} m"
    )
    return channel_table


# ---------------------------------------------------------------------------
# Step 2: Interpolate x, y, z from deviation survey
# ---------------------------------------------------------------------------
def interpolate_from_deviation(
    channel_table: pd.DataFrame, deviation_csv: str
) -> pd.DataFrame:
    """
    Add UTM easting (x), northing (y), and elevation in metres ASL (z) to
    the channel table by linearly interpolating the borehole deviation survey
    at each channel's mapped_MD.

    The deviation file columns are:
      easting, northing, elevation, MD_ft, MD_m, TD
    where elevation is in metres ASL (positive up).
    """
    dev = pd.read_csv(deviation_csv).sort_values("MD_m").reset_index(drop=True)
    md = channel_table["mapped_MD"].values
    channel_table = channel_table.copy()
    channel_table["x"] = np.interp(md, dev["MD_m"].values, dev["easting"].values)
    channel_table["y"] = np.interp(md, dev["MD_m"].values, dev["northing"].values)
    channel_table["z"] = np.interp(md, dev["MD_m"].values, dev["elevation"].values)
    z = channel_table["z"].values
    print(
        f"  Deviation-interpolated elevation range: "
        f"{z.min():.1f} – {z.max():.1f} m ASL"
    )
    return channel_table


# ---------------------------------------------------------------------------
# Step 3: Subsample channels
# ---------------------------------------------------------------------------
def subsample_channels(
    channel_table: pd.DataFrame, spacing_m: float = 10.0
) -> pd.DataFrame:
    """
    Keep one channel per *spacing_m* metres of fiber distance.  The
    distance_m column is the along-fiber distance from channel 0; we bin it
    and take the channel closest to each bin centre.
    """
    dist = channel_table["distance_m"].values
    max_dist = dist.max()
    bins = np.arange(0.0, max_dist + spacing_m, spacing_m)
    keep_indices = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (dist >= lo) & (dist < hi)
        if mask.any():
            # Pick the channel closest to the bin centre
            centre = (lo + hi) / 2.0
            idx = channel_table.index[mask]
            best = idx[np.argmin(np.abs(dist[mask] - centre))]
            keep_indices.append(best)
    subsampled = channel_table.loc[keep_indices].reset_index(drop=True)
    print(
        f"  Subsampled to {len(subsampled)} channels "
        f"(every {spacing_m:.0f} m; "
        f"fiber distance {dist.min():.1f}–{max_dist:.1f} m)"
    )
    return subsampled


# ---------------------------------------------------------------------------
# Step 4: Convert UTM → WGS84 and build GTSRCE lines
# ---------------------------------------------------------------------------
def build_gtsrce_lines(
    channel_table: pd.DataFrame,
    existing_labels: set[str] | None = None,
) -> list[str]:
    """
    Convert UTM Zone 10N (EPSG:32610) easting/northing to WGS84 lat/lon and
    format as NLLoc GTSRCE lines.

    NLLoc LATLON convention (SIMPLE projection, sea-level datum):
      GTSRCE label  LATLON  lat(+N)  lon(+E)  depth_km(+down)  elev_km(+up)

    Both fields are relative to sea level.
    - depth_km > 0  : station is below sea level
    - depth_km < 0  : station is above sea level (same as existing regional
                      stations, e.g. NN07 depth = -1.4336)
    - elev_km = 0.0 always (depth already encodes the full elevation)

    z is elevation in metres ASL (from deviation survey interpolation).
    Convert: depth_km = -z / 1000.

    If *existing_labels* is provided, channels whose label is already in that
    set (i.e. Grid2Time grids already exist) are written as commented-out lines
    so they are preserved for reference but skipped by Grid2Time.  Channels
    not in *existing_labels* are written as active lines for Grid2Time to
    process.
    """
    transformer = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    lines = []
    for _, row in channel_table.iterrows():
        lon, lat = transformer.transform(row["x"], row["y"])
        depth_km = -row["z"] / 1000.0  # z negative (below sea level) → depth positive
        label = str(int(row["CHAN"]))
        gtsrce = (
            f"GTSRCE {label:<6s} LATLON "
            f"{lat:.6f}  {lon:.6f}  {depth_km:.4f}  0.0"
        )
        if existing_labels is not None and label in existing_labels:
            lines.append(f"# {gtsrce}  # grid exists")
        else:
            lines.append(gtsrce)
    return lines


# ---------------------------------------------------------------------------
# Step 4b: Detect channels that already have Grid2Time grids
# ---------------------------------------------------------------------------
def find_existing_grid_labels(grid_root: str, phases: list[str]) -> set[str]:
    """
    Return the set of station labels that already have a complete set of
    Grid2Time travel-time grid files (both ``.buf`` and ``.hdr``) for
    *every* phase in *phases*.

    Grid2Time names files:
        <grid_root>.<PHASE>.<label>.time.buf
        <grid_root>.<PHASE>.<label>.time.hdr

    Only labels that have **all** phases present are considered complete.
    """
    grid_dir = os.path.dirname(grid_root)
    root_base = os.path.basename(grid_root)

    # Collect labels present for each phase
    phase_label_sets: dict[str, set[str]] = {}
    for phase in phases:
        present: set[str] = set()
        pattern = os.path.join(grid_dir, f"{root_base}.{phase}.*.time.buf")
        for fpath in glob(pattern):
            fname = os.path.basename(fpath)
            # fname = <root_base>.<phase>.<label>.time.buf
            prefix = f"{root_base}.{phase}."
            suffix = ".time.buf"
            if fname.startswith(prefix) and fname.endswith(suffix):
                label = fname[len(prefix):-len(suffix)]
                hdr = fpath.replace(".buf", ".hdr")
                if os.path.isfile(hdr):
                    present.add(label)
        phase_label_sets[phase] = present

    if not phase_label_sets:
        return set()

    # Intersection: only labels complete for ALL phases
    complete = set.intersection(*phase_label_sets.values())
    return complete


def parse_gtfiles_phases(cfg_path: str) -> tuple[str | None, list[str]]:
    """
    Parse active (non-commented) GTFILES lines from the NLLoc config and
    return ``(output_root, [phases])``.  Returns ``(None, [])`` if none found.
    """
    output_root: str | None = None
    phases: list[str] = []
    try:
        with open(cfg_path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped.startswith("GTFILES"):
                    continue
                parts = stripped.split()
                # GTFILES  vel_root  time_root  phase
                if len(parts) >= 4:
                    root = parts[2]
                    phase = parts[3]
                    if output_root is None:
                        output_root = root
                    phases.append(phase)
    except FileNotFoundError:
        pass
    return output_root, phases


# ---------------------------------------------------------------------------
# Step 5: Patch GTSRCE lines into the NLLoc config file
# ---------------------------------------------------------------------------
GTSRCE_SENTINEL_START = "# === DAS GTSRCE START ==="
GTSRCE_SENTINEL_END = "# === DAS GTSRCE END ==="


def patch_nlloc_config(cfg_path: str, gtsrce_lines: list[str]) -> None:
    """
    Replace the block between the two sentinel comments in the NLLoc config
    with the new GTSRCE lines. If the sentinels are not present the block is
    appended after the first existing GTSRCE line (or after GTMODE).
    """
    with open(cfg_path) as fh:
        content = fh.read()

    new_block = (
        GTSRCE_SENTINEL_START
        + "\n"
        + "\n".join(gtsrce_lines)
        + "\n"
        + GTSRCE_SENTINEL_END
    )

    if GTSRCE_SENTINEL_START in content:
        # Replace existing block
        pattern = re.compile(
            re.escape(GTSRCE_SENTINEL_START)
            + r".*?"
            + re.escape(GTSRCE_SENTINEL_END),
            re.DOTALL,
        )
        content = pattern.sub(new_block, content)
    else:
        # Insert after the line containing GT_PLFD (just before it)
        content = content.replace(
            "GT_PLFD", new_block + "\n\nGT_PLFD", 1
        )

    with open(cfg_path, "w") as fh:
        fh.write(content)
    print(f"  Written {len(gtsrce_lines)} GTSRCE lines to {cfg_path}")


# ---------------------------------------------------------------------------
# Step 6: Write NLLOC_OBS files
# ---------------------------------------------------------------------------
def _parse_time_utc(series: pd.Series):
    """
    Parse a time_utc column (ISO-8601 strings or anything pandas understands)
    into a DatetimeIndex (UTC-aware).  Returns None if parsing fails.
    """
    try:
        return pd.to_datetime(series, utc=True, infer_datetime_format=True)
    except Exception:
        return None


def _detect_time_utc_col(df: pd.DataFrame) -> str | None:
    """
    Return the name of the UTC-time column in *df*, handling case variants
    (``time_utc``, ``time_UTC``, ``time_Utc``, …).  Returns None if absent.
    """
    for col in df.columns:
        if col.lower() == "time_utc":
            return col
    return None


def write_obs_files(
    csv_dir: str,
    channel_set: set,
    obs_dir: str,
    p_error: float = P_ERROR_S,
    s_error: float = S_ERROR_S,
) -> None:
    """
    For each event CSV write one NLLOC_OBS file containing only picks on
    channels in *channel_set*.

    Pick time source (in priority order):
      1. ``time_utc`` (case-insensitive) – absolute UTC timestamp
         (ISO-8601 or pandas-parseable).  The NLLoc line uses the real
         date/HHMM/seconds extracted from this column.  The epoch constants
         (EPOCH_DATE / EPOCH_HHMM) are NOT used.
      2. ``time_rel_s`` – relative seconds since an event epoch.
         The epoch constants are used for the date/HHMM fields (legacy
         behaviour).
    """
    os.makedirs(obs_dir, exist_ok=True)
    csv_files = sorted(glob(os.path.join(csv_dir, "*_picks.csv")))
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  WARNING: no pick rows in {csv_path}")
            continue
        # Filter to subsampled channels only
        df = df[df["CHAN"].isin(channel_set)].copy()
        if df.empty:
            print(f"  WARNING: no picks on selected channels in {csv_path}")
            continue

        # Detect which time column is available (case-insensitive: time_utc / time_UTC / …)
        utc_col = _detect_time_utc_col(df)

        # Group by event_id (there may be multiple events per file in future)
        for event_id, ev_df in df.groupby("event_id"):
            ev_df = ev_df.copy()

            if utc_col is not None:
                # --- UTC-based pick times ---
                parsed_utc = _parse_time_utc(ev_df[utc_col])
                if parsed_utc is None:
                    print(
                        f"  WARNING: could not parse {utc_col!r} for {event_id} "
                        f"in {csv_path}; falling back to time_rel_s"
                    )
                    use_utc_this = False
                else:
                    ev_df["_dt"] = parsed_utc
                    ev_df = ev_df[ev_df["_dt"].notna()].copy()
                    use_utc_this = True
            else:
                use_utc_this = False

            if not use_utc_this:
                # --- Relative-time fallback ---
                if "time_rel_s" not in ev_df.columns:
                    print(
                        f"  WARNING: neither time_utc nor time_rel_s found "
                        f"for {event_id} in {csv_path}; skipping"
                    )
                    continue
                ev_df["time_rel_s"] = pd.to_numeric(ev_df["time_rel_s"], errors="coerce")
                ev_df = ev_df[np.isfinite(ev_df["time_rel_s"])].copy()

            if ev_df.empty:
                print(f"  WARNING: no valid pick times for {event_id} in {csv_path}")
                continue

            # Keep unique arrival times per phase.
            # If phase confidence exists, retain the highest-confidence row.
            before_n = len(ev_df)
            dedupe_time_col = "_dt" if use_utc_this else "time_rel_s"
            dedupe_keys = ["phase", dedupe_time_col]
            if "phase_confidence" in ev_df.columns:
                ev_df["phase_confidence"] = pd.to_numeric(ev_df["phase_confidence"], errors="coerce")
                ev_df = ev_df.sort_values("phase_confidence", ascending=False, na_position="last")
            ev_df = ev_df.drop_duplicates(subset=dedupe_keys, keep="first").sort_values(dedupe_time_col)

            lines = []
            for _, row in ev_df.iterrows():
                phase = str(row["phase"]).strip()
                err = p_error if phase == "P" else s_error
                label = str(int(row["CHAN"])).ljust(6)

                if use_utc_this:
                    dt = row["_dt"]
                    date_str = dt.strftime("%Y%m%d")
                    hhmm_str = dt.strftime("%H%M")
                    time_s = dt.second + dt.microsecond / 1e6
                else:
                    date_str = EPOCH_DATE
                    hhmm_str = EPOCH_HHMM
                    time_s = float(row["time_rel_s"])

                line = (
                    f"{label} ?    ?    ? {phase:<6s} ? "
                    f"{date_str} {hhmm_str} "
                    f"{time_s:9.4f} "
                    f"GAU {err:9.2e} -1.00e+00 -1.00e+00 -1.00e+00"
                )
                lines.append(line)
            lines.append("")  # blank line terminates the event block

            out_path = os.path.join(obs_dir, f"{event_id}.nll")
            with open(out_path, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            time_src = utc_col if use_utc_this else "time_rel_s"
            print(
                f"  {event_id}: {len(ev_df)} picks (deduped from {before_n}) "
                f"({(ev_df['phase']=='P').sum()} P, "
                f"{(ev_df['phase']=='S').sum()} S) "
                f"[time src: {time_src}] -> {out_path}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DAS NLLoc setup script")
    parser.add_argument("--csv-dir", default=CSV_DIR)
    parser.add_argument("--deviation-csv", default=DEVIATION_CSV)
    parser.add_argument("--nlloc-cfg", default=NLLOC_CFG)
    parser.add_argument("--obs-dir", default=OBS_DIR)
    parser.add_argument("--spacing", type=float, default=SPACING_M,
                        help="Channel subsampling spacing in metres (default 10)")
    parser.add_argument(
        "--grid-root",
        default=None,
        help=(
            "Grid2Time output root path (no extension). "
            "Auto-detected from GTFILES in --nlloc-cfg if not supplied. "
            f"Falls back to: {GRID_ROOT}"
        ),
    )
    parser.add_argument(
        "--gtsrce-only",
        action="store_true",
        help="Only print GTSRCE lines (dry-run; do not write any files)",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Keep all channels from pick files; skip subsampling entirely",
    )
    args = parser.parse_args()

    print("=== Step 1: Building channel table ===")
    channel_table = build_channel_table(args.csv_dir)

    print("\n=== Step 2: Interpolating x/y/z from deviation survey ===")
    channel_table = interpolate_from_deviation(channel_table, args.deviation_csv)

    if args.all_channels:
        print("\n=== Step 3: Keeping ALL channels (no subsampling) ===")
        subsampled = channel_table.copy()
        print(f"  Keeping all {len(subsampled)} channels")
    else:
        print(f"\n=== Step 3: Subsampling channels (every {args.spacing:.0f} m) ===")
        subsampled = subsample_channels(channel_table, args.spacing)

    # --- Detect which channels already have Grid2Time grids ---
    cfg_grid_root, cfg_phases = parse_gtfiles_phases(args.nlloc_cfg)
    grid_root = args.grid_root or cfg_grid_root or GRID_ROOT
    phases = cfg_phases or ["P", "S"]
    print(f"\n=== Step 4b: Checking existing Grid2Time grids ===")
    print(f"  Grid root : {grid_root}")
    print(f"  Phases    : {phases}")
    existing_labels = find_existing_grid_labels(grid_root, phases)
    all_labels = {str(int(c)) for c in subsampled["CHAN"].values}
    new_labels = all_labels - existing_labels
    print(f"  Channels with complete grids : {len(existing_labels)}")
    print(f"  New channels needing Grid2Time: {len(new_labels)}")
    if new_labels:
        sorted_new = sorted(new_labels, key=lambda x: int(x))
        print(f"  New labels: {sorted_new}")

    print("\n=== Step 4: Building GTSRCE lines ===")
    gtsrce_lines = build_gtsrce_lines(subsampled, existing_labels=existing_labels)
    active = [l for l in gtsrce_lines if not l.startswith("#")]
    commented = [l for l in gtsrce_lines if l.startswith("#")]
    print(f"  Active (new, needs Grid2Time) : {len(active)}")
    print(f"  Commented (grid exists)       : {len(commented)}")
    if active:
        print(f"  First active:\n  " + "\n  ".join(active[:3]))

    if args.gtsrce_only:
        print("\nAll GTSRCE lines:")
        print("\n".join(gtsrce_lines))
        return

    # Step 5: Patch GTSRCE lines into config.
    # Active lines are for new channels Grid2Time will process.
    # Commented lines preserve the geometry record for channels already done.
    print(f"\n=== Step 5: Patching NLLoc config: {args.nlloc_cfg} ===")
    patch_nlloc_config(args.nlloc_cfg, gtsrce_lines)

    print(f"\n=== Step 6: Writing obs files to {args.obs_dir} ===")
    channel_set = set(subsampled["CHAN"].values)
    write_obs_files(
        args.csv_dir,
        channel_set,
        args.obs_dir,
        p_error=P_ERROR_S,
        s_error=S_ERROR_S,
    )

    print("\nDone.")
    if new_labels:
        sorted_new = sorted(new_labels, key=lambda x: int(x))
        print(
            f"\n*** {len(new_labels)} new channel(s) need Grid2Time grids. ***\n"
            f"    Run Grid2Time ONLY after confirming the config has been updated.\n"
            f"    Channels: {sorted_new}\n"
            f"    Existing grids ({len(existing_labels)} channels) will NOT be "
            f"recomputed (Grid2Time skips stations whose .buf already exists).\n"
        )
        print("Next step (Grid2Time for new channels only):")
        print(f"  cd {os.path.dirname(args.nlloc_cfg)}")
        print("  Grid2Time locate_newberry_DAS.nlloc")
        print(
            "  (Grid2Time will skip channels that already have .buf/.hdr files.)"
        )
    else:
        print(
            f"\nAll {len(all_labels)} channels already have Grid2Time grids. "
            "No Grid2Time run needed — proceed directly to NLLoc."
        )
        print(f"\n  cd {os.path.dirname(args.nlloc_cfg)}")
        print("  NLLoc locate_newberry_DAS.nlloc")


if __name__ == "__main__":
    main()
