#!/usr/bin/env python3
"""Publish CUSSP CASSM artifacts from recorder server to remote VM.

This helper is intentionally transport-agnostic and supports:
- rsync over SSH
- rclone remote targets

Typical usage:
- publish metrics bundle + manifest directory
- publish inversion products directory used by the viz app
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path


LOG = logging.getLogger("cussp_cassm_publish")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_cmd(cmd):
    LOG.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout.strip():
            LOG.error("stdout: %s", result.stdout.strip())
        if result.stderr.strip():
            LOG.error("stderr: %s", result.stderr.strip())
        raise RuntimeError(f"Command failed with code {result.returncode}")
    if result.stdout.strip():
        LOG.debug("stdout: %s", result.stdout.strip())


def sync_rsync(src_dir: Path, dst: str, delete: bool = False):
    cmd = ["rsync", "-az", "--mkpath"]
    if delete:
        cmd.append("--delete")
    cmd.extend([f"{src_dir}/", dst.rstrip("/") + "/"])
    run_cmd(cmd)


def sync_rclone(src_dir: Path, dst: str, delete: bool = False):
    cmd = ["rclone", "copy", str(src_dir), dst, "--checkers", "4", "--transfers", "2"]
    if delete:
        cmd[1] = "sync"
    run_cmd(cmd)


def sync_dir(src_dir: Path, dst: str, mode: str, delete: bool = False):
    if not src_dir.exists():
        LOG.warning("Source directory missing, skipping: %s", src_dir)
        return
    if mode == "rsync":
        sync_rsync(src_dir, dst, delete=delete)
    elif mode == "rclone":
        sync_rclone(src_dir, dst, delete=delete)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def run_once(args) -> int:
    live_dir = Path(args.live_dir)
    inv_dir = Path(args.inversion_dir) if args.inversion_dir else None

    sync_dir(
        src_dir=live_dir,
        dst=args.live_remote,
        mode=args.mode,
        delete=args.delete,
    )
    LOG.info("Published live artifacts from %s to %s", live_dir, args.live_remote)

    if inv_dir is not None:
        sync_dir(
            src_dir=inv_dir,
            dst=args.inversion_remote,
            mode=args.mode,
            delete=args.delete,
        )
        LOG.info("Published inversion artifacts from %s to %s", inv_dir, args.inversion_remote)

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Publish CUSSP CASSM artifacts to remote VM")
    p.add_argument("--mode", choices=["rsync", "rclone"], default="rsync")
    p.add_argument("--live-dir", default="/data/chet-cussp/cassm/live")
    p.add_argument("--live-remote", required=True, help="Remote target for live dir (ssh path or rclone remote)")
    p.add_argument("--inversion-dir", default="/data/chet-cussp/cassm/inversion/live")
    p.add_argument("--inversion-remote", default="")
    p.add_argument("--delete", action="store_true", help="Mirror deletions on remote")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--period-s", type=int, default=60)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.inversion_dir and not args.inversion_remote:
        raise SystemExit("--inversion-remote is required when --inversion-dir is set")

    if not args.watch:
        return run_once(args)

    LOG.info("Starting publish watch mode (%ds)", args.period_s)
    while True:
        try:
            run_once(args)
        except Exception as exc:
            LOG.exception("Publish cycle failed: %s", exc)
        time.sleep(max(args.period_s, 5))


if __name__ == "__main__":
    raise SystemExit(main())
