#!/usr/bin/env python3
"""
Pull CUSSP injection CSV files from Google Drive and atomically publish the
latest data+metadata pair into a live directory for dashboards.

Requires rclone configured with a remote (e.g. gdrive:).
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


log = logging.getLogger("cussp_pull_injection")

# Restrict pipeline to campaign files expected by the CUSSP dashboard.
INJ_DATA_PATTERN = "CUSSP*.INJ_data.csv"
INJ_META_PATTERN = "CUSSP*.INJ_metadata.csv"


def is_root_remote(remote_folder):
    """Return True when remote points at top-level root (e.g. 'name:')."""
    if ":" not in remote_folder:
        return False
    remote, path = remote_folder.split(":", 1)
    return bool(remote) and path.strip() == ""


def run_cmd(cmd):
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Command failed: %s", " ".join(cmd))
        if result.stdout:
            log.error("stdout: %s", result.stdout.strip())
        if result.stderr:
            log.error("stderr: %s", result.stderr.strip())
        raise RuntimeError(f"Command failed with code {result.returncode}")
    if result.stdout.strip():
        log.debug("stdout: %s", result.stdout.strip())
    return result


def resolve_metadata_path(data_path):
    return data_path.with_name(data_path.name.replace("data", "metadata"))


def _date_key_from_name(path):
    """Extract YYYY_MM_DD from CUSSPYYYY_MM_DD.INJ_data.csv names."""
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _parse_injection_time_column(series, data_path):
    """Parse injection time values using robust fallbacks.

    Handles:
    - MM/DD/YY HH:MM:SS strings
    - Fractional day offsets (0-2) relative to filename date
    - Excel serial day numbers
    - Generic datetime strings
    """
    s = series.astype(str).str.strip()

    # 1) Expected string format in CUSSP files.
    parsed = pd.to_datetime(s, format="%m/%d/%y %H:%M:%S", errors="coerce")
    if parsed.notna().mean() >= 0.80:
        return parsed

    # 2) Numeric fallback for files that encode time as numbers.
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_valid = numeric.notna().mean()
    if numeric_valid >= 0.80:
        base_date_key = _date_key_from_name(data_path)
        if base_date_key is not None:
            y, m, d = base_date_key
            base = pd.Timestamp(year=y, month=m, day=d)
        else:
            base = pd.Timestamp("1970-01-01")

        # Fractional day in [0, 2] -> offset from file date.
        num_nonan = numeric.dropna()
        if not num_nonan.empty and ((num_nonan >= 0) & (num_nonan <= 2)).mean() >= 0.80:
            parsed_num = base + pd.to_timedelta(numeric, unit="D")
            if parsed_num.notna().mean() >= 0.80:
                return parsed_num

        # Excel serial date numbers.
        if not num_nonan.empty and (num_nonan > 20000).mean() >= 0.80:
            parsed_num = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            if parsed_num.notna().mean() >= 0.80:
                return parsed_num

    # 3) Generic parser for remaining string-like values.
    parsed = pd.to_datetime(s, errors="coerce")
    return parsed


def find_latest_pair(staging_dir):
    data_files = sorted(staging_dir.glob(INJ_DATA_PATTERN))
    if not data_files:
        return None, None

    # Prefer newest by date encoded in filename; fall back to mtime.
    dated = []
    undated = []
    for p in data_files:
        dk = _date_key_from_name(p)
        if dk is None:
            undated.append(p)
        else:
            dated.append((dk, p.stat().st_mtime, p))

    candidates = []
    if dated:
        dated.sort(key=lambda t: (t[0], t[1]), reverse=True)
        candidates.extend([t[2] for t in dated])
    if undated:
        undated.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(undated)

    for data_path in candidates:
        metadata_path = resolve_metadata_path(data_path)
        if metadata_path.exists():
            return data_path, metadata_path

    return candidates[0], None


def atomic_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def sync_from_drive(remote_folder, staging_dir, drive_shared_with_me=False):
    staging_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rclone",
        "copy",
        remote_folder,
        str(staging_dir),
        "--include",
        INJ_DATA_PATTERN,
        "--include",
        INJ_META_PATTERN,
        "--checkers",
        "4",
        "--transfers",
        "2",
        "--contimeout",
        "30s",
        "--timeout",
        "60s",
    ]
    if drive_shared_with_me:
        cmd.append("--drive-shared-with-me")
    run_cmd(cmd)


def publish_latest(staging_dir, live_dir):
    data_path, metadata_path = find_latest_pair(staging_dir)
    if data_path is None:
        log.warning("No injection data files found in staging: %s", staging_dir)
        return False
    if metadata_path is None:
        log.warning("Latest data file has no metadata pair: %s", data_path.name)
        return False

    live_dir.mkdir(parents=True, exist_ok=True)
    live_data = live_dir / data_path.name
    live_metadata = live_dir / metadata_path.name

    atomic_copy(data_path, live_data)
    atomic_copy(metadata_path, live_metadata)

    # Stable aliases for dashboards to read without scanning.
    atomic_copy(data_path, live_dir / "latest_INJ_data.csv")
    atomic_copy(metadata_path, live_dir / "latest_INJ_metadata.csv")

    log.info("Published latest injection pair: %s + %s", data_path.name, metadata_path.name)
    return True


def resample_injection_data(staging_dir, output_path, resample_freq='1min'):
    """Build a complete downsampled injection dataset from all files.
    
    Reads ALL *INJ_data.csv files, concatenates them in time order,
    resamples to 1-min mean, and writes to output.
    Falls back silently if any step fails (this is not fatal).
    """
    try:
        data_files = sorted(staging_dir.glob(INJ_DATA_PATTERN))
        if not data_files:
            log.warning("No %s files found in staging for resampling", INJ_DATA_PATTERN)
            return False
        
        dfs = []
        required_cols = ["Time", "PT 503", "Net Flow"]
        for data_path in data_files:
            try:
                # Read all columns first so BOM stripping can happen before column selection.
                # Passing usecols before stripping BOM causes a ValueError when the first
                # column header is '\ufeffTime', silently skipping every file.
                #
                # index_col=False is required: CUSSP CSVs have a trailing comma on every
                # data row, producing one more field than header columns. Without this flag
                # pandas (C engine) raises "Expected N fields, saw N+1" and the python engine
                # silently drops every data row via on_bad_lines='skip'.  index_col=False
                # tells pandas not to promote the spurious empty last field to a row index,
                # which absorbs the extra field cleanly without dropping any rows.
                df = pd.read_csv(
                    data_path,
                    skiprows=[1, 2],
                    index_col=False,
                    engine="python",
                    on_bad_lines="skip",
                )
                df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
                if any(c not in df.columns for c in required_cols):
                    log.warning(
                        "Skipping %s: missing required columns %s (found: %s)",
                        data_path.name, required_cols, list(df.columns),
                    )
                    continue
                df = df[required_cols]
                
                # Parse Time column with robust fallbacks (string, fractional-day, excel serial).
                df['Time'] = _parse_injection_time_column(df['Time'], data_path)
                
                df['PT 503'] = pd.to_numeric(df['PT 503'], errors='coerce')
                df['Net Flow'] = pd.to_numeric(df['Net Flow'], errors='coerce')
                df = df.dropna(subset=['Time'])
                if not df.empty:
                    dfs.append(df)
                    log.debug("Loaded %d rows from %s", len(df), data_path.name)
            except Exception as e:
                log.warning("Failed to read %s: %s", data_path.name, e)
                continue
        
        if not dfs:
            log.warning("No valid injection data found; skipping resample")
            return False
        
        # Concatenate only the columns required by the dashboard.
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['Time']).sort_values('Time')
        log.info("Combined %d files into %d total rows", len(data_files), len(df_combined))
        
        df_combined.set_index('Time', inplace=True)
        df_combined.index = pd.DatetimeIndex(df_combined.index)
        
        # Resample required series to 1-min mean, then drop NaN to keep only actual data periods
        df_resampled = df_combined.resample(resample_freq).agg({
            'PT 503': 'mean',
            'Net Flow': 'mean',
        })
        # Drop rows where both columns are NaN (no data in that minute)
        df_resampled = df_resampled.dropna(how='all')
        df_resampled.reset_index(inplace=True)
        
        # Write resampled data (pandas handles datetime serialization automatically)
        df_resampled.to_csv(output_path, index=False)
        log.info("Resampled complete injection history to %s: %d -> %d rows (after dropping empty periods)", resample_freq, len(df_combined), len(df_resampled))

        # Record which inputs produced this output so needs_resample() can detect
        # future changes without relying on rclone/Drive-influenced mtimes.
        _write_staging_manifest(staging_dir, _manifest_path(output_path))
        return True
    except Exception as e:
        log.warning("Failed to resample injection data: %s", e)
        return False


def _manifest_path(output_path):
    return output_path.with_name(output_path.stem + ".manifest.json")


def _staging_manifest(staging_dir):
    """Return {filename: size} for all staging INJ data files.

    Used as a deterministic stand-in for mtime, since rclone-preserved
    Drive modtimes don't reliably advance when a file's content changes.
    """
    return {p.name: p.stat().st_size for p in sorted(staging_dir.glob(INJ_DATA_PATTERN))}


def _write_staging_manifest(staging_dir, manifest_path):
    manifest_path.write_text(json.dumps(_staging_manifest(staging_dir)))


def needs_resample(staging_dir, output_path):
    """Return True when output/manifest is missing or staging files changed."""
    data_files = list(staging_dir.glob(INJ_DATA_PATTERN))
    if not data_files:
        return False
    if not output_path.exists():
        return True

    manifest_path = _manifest_path(output_path)
    if not manifest_path.exists():
        return True

    try:
        last_manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return True

    return _staging_manifest(staging_dir) != last_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Pull and publish CUSSP injection CSV files")
    parser.add_argument(
        "--remote-folder",
        default="shared_cussp_inj:",
        help="Rclone remote folder containing INJ data/metadata CSV files",
    )
    parser.add_argument(
        "--staging-dir",
        default="/data/chet-cussp/injection/staging",
        help="Local staging directory for synced files",
    )
    parser.add_argument(
        "--live-dir",
        default="/data/chet-cussp/injection/live",
        help="Local live directory consumed by the dashboard",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--allow-root-remote",
        action="store_true",
        help="Allow syncing from remote root (disabled by default as a safety guard)",
    )
    parser.add_argument(
        "--drive-shared-with-me",
        action="store_true",
        help="Enable rclone Google Drive shared-with-me listing mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    staging_dir = Path(args.staging_dir)
    live_dir = Path(args.live_dir)

    if is_root_remote(args.remote_folder) and not args.allow_root_remote:
        log.error(
            "Refusing to sync from remote root '%s'. Set --remote-folder to a specific folder path.",
            args.remote_folder,
        )
        log.error(
            "Example: --remote-folder shared_cussp_inj:chet-cussp/CUSSP_Data/Pressure_Flow_Data"
        )
        return 2

    sync_from_drive(args.remote_folder, staging_dir, drive_shared_with_me=args.drive_shared_with_me)
    ok = publish_latest(staging_dir, live_dir)
    
    # After publishing, create/update downsampled version from ALL historical data.
    # Only rebuild when there are newer inputs than the current output.
    if ok:
        resampled_data = live_dir / "latest_INJ_data_1min.csv"
        if needs_resample(staging_dir, resampled_data):
            resample_injection_data(staging_dir, resampled_data, resample_freq='1min')
        else:
            log.info("Skipping resample: %s is up to date", resampled_data)
    
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log.exception("Injection pull failed: %s", exc)
        raise SystemExit(2)
