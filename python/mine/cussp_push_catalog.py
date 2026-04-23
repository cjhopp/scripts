#!/usr/bin/env python3
"""
cussp_push_catalog.py

Runs on the mine SeisComP server (behind VPN). Fetches the seismicity catalog
from the local FDSN service and pushes it to the web VM via rsync over SSH.

Designed to be called by a systemd timer (see cussp-catalog-push.timer).

SSH key setup (one-time, on the mine server):
    ssh-keygen -t ed25519 -f ~/.ssh/id_cussp_vm -N ""
    ssh-copy-id -i ~/.ssh/id_cussp_vm.pub chopp@cussp-vm.lbl.gov

Usage:
    python cussp_push_catalog.py [options]
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

# SURF_converter lives alongside this script in the lbnl package on the mine server
try:
    from lbnl.coordinates import SURF_converter
    _SURF = SURF_converter()
except ImportError:
    _SURF = None
    logging.getLogger(__name__).warning(
        "lbnl.coordinates not importable — HMC annotation will be skipped"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — override via CLI flags or the systemd service ExecStart line
# ---------------------------------------------------------------------------
DEFAULT_FDSN_URL = "http://localhost:8080"
DEFAULT_DAYS = 365
DEFAULT_OUTPUT = "/tmp/cussp_catalog.quakeml"
DEFAULT_RSYNC_TARGET = "chopp@cussp-vm.lbl.gov:/data/chet-cussp/seismicity/catalog.quakeml"
DEFAULT_SSH_KEY = str(Path.home() / ".ssh" / "id_cussp_vm")
DEFAULT_LAT = 44.3517
DEFAULT_LON = -103.7508
DEFAULT_RADIUS_DEG = 0.5    # degrees (~55 km); tighten once site coordinates confirmed

# HMC z of the Earth surface directly above the 4100L experiment volume.
# Calibrate by comparing a known SeisComP event against the mine's HMC survey:
#   SURF_SURFACE_HMC_Z_M = hmc_elev_from_survey + origin.depth_for_that_event
# Rough estimate: borehole tops ~355 m HMC + ~1250 m to surface ≈ 1605 m.
SURF_SURFACE_HMC_Z_M = 1605.0


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def annotate_hmc(catalog):
    """Add hmc_east / hmc_north / hmc_elev extra attributes to every origin.

    hmc_east  — metres East in the Homestake Mine Coordinate system
    hmc_north — metres North in HMC
    hmc_elev  — HMC elevation (metres), computed as::

        hmc_elev = SURF_SURFACE_HMC_Z_M - origin.depth

    where origin.depth is metres positive-downward (standard QuakeML / ObsPy).
    Skips silently if SURF_converter is unavailable.
    """
    if _SURF is None:
        return catalog

    n = 0
    for ev in catalog:
        for orig in ev.origins:
            try:
                east, north, _ = _SURF.to_HMC(
                    (orig.longitude, orig.latitude, 0.0)
                )
                elev = SURF_SURFACE_HMC_Z_M - orig.depth  # depth in metres
                orig.extra["hmc_east"]  = {"value": str(east),  "namespace": "CUSSP"}
                orig.extra["hmc_north"] = {"value": str(north), "namespace": "CUSSP"}
                orig.extra["hmc_elev"]  = {"value": str(elev),  "namespace": "CUSSP"}
                n += 1
            except Exception as exc:
                log.debug("HMC annotation failed for origin %s: %s", orig.resource_id, exc)
    log.info("Annotated %d origin(s) with HMC coordinates", n)
    return catalog


def fetch_catalog(fdsn_url, days, lat, lon, radius_deg):
    log.info("Connecting to FDSN at %s", fdsn_url)
    client = Client(fdsn_url)
    starttime = UTCDateTime(datetime.utcnow() - timedelta(days=days))
    endtime = UTCDateTime()

    try:
        catalog = client.get_events(
            starttime=starttime,
            endtime=endtime,
            latitude=lat,
            longitude=lon,
            maxradius=radius_deg,
            includeallmagnitudes=True,
        )
    except FDSNNoDataException:
        log.warning("No events found for the requested time/space window")
        from obspy import Catalog
        catalog = Catalog()

    log.info("Fetched %d event(s)", len(catalog))
    return catalog


def write_catalog(catalog, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.write(str(path), format="QUAKEML")
    log.info("Wrote %d event(s) to %s", len(catalog), path)


def push_to_vm(local_path, rsync_target, ssh_key):
    ssh_opts = f"ssh -i {ssh_key} -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
    cmd = [
        "rsync",
        "-az",
        "--checksum",
        "-e", ssh_opts,
        str(local_path),
        rsync_target,
    ]
    log.info("rsync: %s → %s", local_path, rsync_target)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("rsync failed (exit %d): %s", result.returncode, result.stderr.strip())
        sys.exit(1)
    log.info("Push complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch CUSSP seismicity catalog and push to web VM"
    )
    parser.add_argument(
        "--fdsn-url", default=DEFAULT_FDSN_URL,
        help=f"FDSN base URL (default: {DEFAULT_FDSN_URL})",
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Days of history to fetch (default: {DEFAULT_DAYS})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Local QuakeML output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--rsync-target", default=DEFAULT_RSYNC_TARGET,
        help=f"rsync destination user@host:path (default: {DEFAULT_RSYNC_TARGET})",
    )
    parser.add_argument(
        "--ssh-key", default=DEFAULT_SSH_KEY,
        help=f"Path to SSH private key (default: {DEFAULT_SSH_KEY})",
    )
    parser.add_argument(
        "--lat", type=float, default=DEFAULT_LAT,
        help=f"Centre latitude for event search (default: {DEFAULT_LAT})",
    )
    parser.add_argument(
        "--lon", type=float, default=DEFAULT_LON,
        help=f"Centre longitude for event search (default: {DEFAULT_LON})",
    )
    parser.add_argument(
        "--radius", type=float, default=DEFAULT_RADIUS_DEG,
        help=f"Search radius in degrees (default: {DEFAULT_RADIUS_DEG})",
    )
    args = parser.parse_args()

    catalog = fetch_catalog(args.fdsn_url, args.days, args.lat, args.lon, args.radius)
    catalog = annotate_hmc(catalog)
    write_catalog(catalog, args.output)
    push_to_vm(args.output, args.rsync_target, args.ssh_key)


if __name__ == "__main__":
    main()
