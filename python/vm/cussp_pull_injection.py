#!/usr/bin/env python3
"""
Pull CUSSP injection CSV files from Google Drive and atomically publish the
latest data+metadata pair into a live directory for dashboards.

Requires rclone configured with a remote (e.g. gdrive:).
"""

import argparse
import io
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


log = logging.getLogger("cussp_pull_injection")


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


def find_latest_pair(staging_dir):
    data_files = sorted(staging_dir.glob("*INJ_data.csv"))
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
        "*INJ_data.csv",
        "--include",
        "*INJ_metadata.csv",
        "--checkers",
        "4",
        "--transfers",
        "2",
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
        data_files = sorted(staging_dir.glob("*INJ_data.csv"))
        if not data_files:
            log.warning("No INJ_data.csv files found in staging for resampling")
            return False
        
        dfs = []
        for data_path in data_files:
            try:
                # Read file, stripping trailing commas
                with open(data_path, 'r') as f:
                    lines = [line.rstrip('\r\n').rstrip(',') + '\n' for line in f]
                csv_text = ''.join(lines)
                
                # Load with skiprows=[1,2] to skip units row
                df = pd.read_csv(io.StringIO(csv_text), skiprows=[1, 2])
                df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
                
                # Parse Time column
                try:
                    df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%y %H:%M:%S', errors='raise')
                except (ValueError, TypeError):
                    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
                
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
        
        # Concatenate all files
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['Time']).sort_values('Time')
        log.info("Combined %d files into %d total rows", len(data_files), len(df_combined))
        
        df_combined.set_index('Time', inplace=True)
        df_combined.index = pd.DatetimeIndex(df_combined.index)
        
        # Resample PT 503 and Net Flow to 1-min mean, keep first value of other cols
        agg_dict = {}
        for col in df_combined.columns:
            if col in ['PT 503', 'Net Flow']:
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'first'  # Just keep first value for non-numeric cols
        
        df_resampled = df_combined.resample(resample_freq).agg(agg_dict)
        df_resampled.reset_index(inplace=True)
        
        # Write resampled data
        df_resampled.to_csv(output_path, index=False)
        log.info("Resampled complete injection history to %s: %d -> %d rows", resample_freq, len(df_combined), len(df_resampled))
        return True
    except Exception as e:
        log.warning("Failed to resample injection data: %s", e)
        return False


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
    
    # After publishing, create downsampled version from ALL historical data
    if ok:
        resampled_data = live_dir / "latest_INJ_data_1min.csv"
        resample_injection_data(staging_dir, resampled_data, resample_freq='1min')
    
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log.exception("Injection pull failed: %s", exc)
        raise SystemExit(2)
