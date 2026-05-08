#!/usr/bin/env python3
"""
Pull CUSSP injection CSV files from Google Drive and atomically publish the
latest data+metadata pair into a live directory for dashboards.

Requires rclone configured with a remote (e.g. gdrive:).
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def find_latest_pair(staging_dir):
    data_files = sorted(staging_dir.glob("*INJ_data.csv"))
    if not data_files:
        return None, None

    candidates = sorted(data_files, key=lambda p: p.stat().st_mtime, reverse=True)
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
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log.exception("Injection pull failed: %s", exc)
        raise SystemExit(2)
