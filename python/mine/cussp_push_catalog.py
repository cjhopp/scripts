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
import csv
import fcntl
import logging
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNNoServiceException
from obspy.core.util.attribdict import AttribDict
from pyproj import Transformer

# Ensure local package imports work when run via systemd from arbitrary CWD.
_LOCAL_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_LOCAL_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PYTHON_DIR))

# SURF_converter lives alongside this script in the lbnl package on the mine server
try:
    from lbnl.coordinates import SURF_converter
    _SURF = SURF_converter()
except ImportError:
    _SURF = None
    logging.getLogger(__name__).warning(
        "lbnl.coordinates not importable — HMC annotation will be skipped"
    )

_WGS84_TO_NAD27_UTM13 = Transformer.from_crs(
    "EPSG:4326", "EPSG:26713", always_xy=True
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
DEFAULT_LOCK_FILE = "/tmp/cussp_catalog_push.lock"
DEFAULT_RSYNC_TIMEOUT = 60
DEFAULT_SSH_CONNECT_TIMEOUT = 10
DEFAULT_LAT = 44.3517
DEFAULT_LON = -103.7508
DEFAULT_RADIUS_DEG = 0.5    # degrees (~55 km); tighten once site coordinates confirmed
DEFAULT_HMC_MODE = "surf-datum-aware"
DEFAULT_STATION_OUTPUT = "/tmp/cussp_stations_hmc.csv"
DEFAULT_STATION_RSYNC_TARGET = "chopp@cussp-vm.lbl.gov:/data/chet-cussp/seismicity/stations_hmc.csv"

# Wellhead-anchored linear approximation used by /tmp/plot_hypocenters.py.
# Keep these in sync with that plotting script when recalibrated.
WH_LAT = 44.35105719
WH_LON = -103.75035647
WH_HMC_E = 1217.0
WH_HMC_N = -862.0
LAT_SCALE = 111000.0   # m / deg latitude
LON_SCALE = 79411.0    # m / deg longitude at ~44.35 N

# HMC z of the Earth surface directly above the 4100L experiment volume.
# Calibrate by comparing a known SeisComP event against the mine's HMC survey:
#   SURF_SURFACE_HMC_Z_M = hmc_elev_from_survey + origin.depth_for_that_event
# Rough estimate: borehole tops ~355 m HMC + ~1250 m to surface ≈ 1605 m.
SURF_SURFACE_HMC_Z_M = 1605.0


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def _depth_to_hmc_elev(depth_m):
    """Map origin depth to HMC elevation for local CUSSP catalogs."""
    if depth_m is None:
        return 0.0
    depth_m = float(depth_m)
    # Local scanloc catalogs store 4100L events as negative depth values.
    if depth_m < 0 and abs(depth_m) < 1000:
        return -depth_m
    return SURF_SURFACE_HMC_Z_M - depth_m


def _lonlat_to_hmc_xy(lon, lat, hmc_mode):
    """Convert lon/lat to HMC easting/northing for the selected conversion mode."""
    if hmc_mode == "linear-wellhead":
        east = WH_HMC_E + (float(lon) - WH_LON) * LON_SCALE
        north = WH_HMC_N + (float(lat) - WH_LAT) * LAT_SCALE
        return east, north

    if _SURF is None:
        raise RuntimeError("SURF_converter unavailable for requested HMC mode")

    if hmc_mode == "surf-datum-aware":
        utm_e, utm_n = _WGS84_TO_NAD27_UTM13.transform(float(lon), float(lat))
        return utm_e - _SURF.orig_utm[0], utm_n - _SURF.orig_utm[1]

    east, north, _ = _SURF.to_HMC((float(lon), float(lat), 0.0))
    return east, north


def annotate_hmc(catalog, hmc_mode):
    """Add hmc_east / hmc_north / hmc_elev extra attributes to every origin.

    hmc_east  — metres East in the Homestake Mine Coordinate system
    hmc_north — metres North in HMC
    hmc_elev  — HMC elevation (metres), computed as::

        hmc_elev = SURF_SURFACE_HMC_Z_M - origin.depth

    where origin.depth is metres positive-downward (standard QuakeML / ObsPy).
    Skips silently if SURF_converter is unavailable.
    """
    if hmc_mode in {"surf-nad27", "surf-datum-aware"} and _SURF is None:
        log.warning("HMC mode '%s' requested but SURF_converter is unavailable", hmc_mode)
        return catalog

    n = 0
    failed = 0
    for ev in catalog:
        for orig in ev.origins:
            try:
                east, north = _lonlat_to_hmc_xy(orig.longitude, orig.latitude, hmc_mode)

                elev = _depth_to_hmc_elev(orig.depth)
                if getattr(orig, "extra", None) is None:
                    orig.extra = AttribDict()
                orig.extra["hmc_east"]  = {"value": str(east),  "namespace": "CUSSP"}
                orig.extra["hmc_north"] = {"value": str(north), "namespace": "CUSSP"}
                orig.extra["hmc_elev"]  = {"value": str(elev),  "namespace": "CUSSP"}
                n += 1
            except Exception as exc:
                failed += 1
                log.debug("HMC annotation failed for origin %s: %s", orig.resource_id, exc)
    log.info("Annotated %d origin(s) with HMC coordinates (mode=%s)", n, hmc_mode)
    if failed:
        log.warning("HMC annotation failed for %d origin(s)", failed)
    return catalog


def fetch_stations_hmc(fdsn_url, hmc_mode):
    """Fetch station metadata and convert to HMC coordinates.

    Exports one row per channel to preserve per-channel depth metadata.
    """
    log.info("Fetching station metadata from FDSN at %s", fdsn_url)
    client = Client(fdsn_url)
    try:
        inv = client.get_stations(level="channel")
    except FDSNNoServiceException as exc:
        log.error("FDSN station service is not available at %s", fdsn_url)
        log.error("Enable it by setting 'serveStation = true' in fdsnws.cfg")
        log.error("Original error: %s", exc)
        return []
    except FDSNNoDataException:
        log.warning("No station metadata returned by FDSN station service")
        return []

    rows = []
    failed = 0
    for net in inv.networks:
        for sta in net.stations:
            for cha in sta.channels:
                try:
                    lat = cha.latitude if cha.latitude is not None else sta.latitude
                    lon = cha.longitude if cha.longitude is not None else sta.longitude
                    elev = cha.elevation if cha.elevation is not None else sta.elevation
                    depth = cha.depth if cha.depth is not None else 0.0
                    east, north = _lonlat_to_hmc_xy(lon, lat, hmc_mode)
                    rows.append(
                        {
                            "network": net.code,
                            "station": sta.code,
                            "location": cha.location_code,
                            "channel": cha.code,
                            "latitude": float(lat),
                            "longitude": float(lon),
                            "elevation_m_asl": float(elev) if elev is not None else "",
                            "channel_depth_m": float(depth),
                            "hmc_east_m": float(east),
                            "hmc_north_m": float(north),
                            "hmc_z_m_asl": float(elev) if elev is not None else "",
                            "hmc_z_minus_depth_m": (
                                float(elev) - float(depth) if elev is not None else ""
                            ),
                        }
                    )
                except Exception:
                    failed += 1

    log.info("Prepared %d station/channel HMC row(s)", len(rows))
    if failed:
        log.warning("Skipped %d station/channel row(s) due to conversion errors", failed)
    return rows


def write_station_csv(rows, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "network",
        "station",
        "location",
        "channel",
        "latitude",
        "longitude",
        "elevation_m_asl",
        "channel_depth_m",
        "hmc_east_m",
        "hmc_north_m",
        "hmc_z_m_asl",
        "hmc_z_minus_depth_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %d station/channel row(s) to %s", len(rows), path)


def fetch_catalog(fdsn_url, days, lat, lon, radius_deg):
    log.info("Connecting to FDSN at %s", fdsn_url)
    client = Client(fdsn_url)
    starttime = UTCDateTime(datetime.now(timezone.utc) - timedelta(days=days))
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
    except FDSNNoServiceException as exc:
        log.error("FDSN event service is not available at %s", fdsn_url)
        log.error(
            "Enable it in SeisComP by setting 'serveEvent = true' in fdsnws.cfg "
            "and restart fdsnws."
        )
        log.error("Original error: %s", exc)
        sys.exit(2)
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


def parse_rsync_target(rsync_target):
    """Split rsync target of the form user@host:/absolute/path."""
    if ":" not in rsync_target:
        raise ValueError(f"Invalid rsync target (missing ':'): {rsync_target}")
    host, remote_path = rsync_target.split(":", 1)
    if not host or not remote_path:
        raise ValueError(f"Invalid rsync target: {rsync_target}")
    return host, remote_path


def push_to_vm(local_path, rsync_target, ssh_key, rsync_timeout, ssh_connect_timeout):
    host, remote_path = parse_rsync_target(rsync_target)
    remote_tmp_path = f"{remote_path}.tmp"
    ssh_target_tmp = f"{host}:{remote_tmp_path}"
    ssh_base = [
        "ssh",
        "-i", str(ssh_key),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={int(ssh_connect_timeout)}",
    ]
    ssh_opts = " ".join(shlex.quote(x) for x in ssh_base)

    cmd = [
        "rsync",
        "-az",
        "--checksum",
        "--timeout", str(int(rsync_timeout)),
        "-e", ssh_opts,
        str(local_path),
        ssh_target_tmp,
    ]
    log.info("rsync: %s -> %s (tmp)", local_path, ssh_target_tmp)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("rsync failed (exit %d): %s", result.returncode, result.stderr.strip())
        sys.exit(1)

    mv_cmd = ssh_base + [host, "mv", "-f", remote_tmp_path, remote_path]
    log.info("remote publish: %s", " ".join(shlex.quote(x) for x in mv_cmd))
    mv_result = subprocess.run(mv_cmd, capture_output=True, text=True)
    if mv_result.returncode != 0:
        log.error(
            "remote publish failed (exit %d): %s",
            mv_result.returncode,
            mv_result.stderr.strip(),
        )
        sys.exit(1)

    log.info("Push complete")


def acquire_lock(lock_path):
    """Acquire a non-blocking lock for singleton execution."""
    lock_file = open(lock_path, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.warning("Another catalog push is already running; exiting")
        sys.exit(0)

    lock_file.write(str(Path("/proc/self").resolve().name) + "\n")
    lock_file.flush()
    return lock_file


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
        "--lock-file", default=DEFAULT_LOCK_FILE,
        help=f"Lock file to prevent overlapping runs (default: {DEFAULT_LOCK_FILE})",
    )
    parser.add_argument(
        "--rsync-timeout", type=int, default=DEFAULT_RSYNC_TIMEOUT,
        help=f"Idle I/O timeout in seconds for rsync (default: {DEFAULT_RSYNC_TIMEOUT})",
    )
    parser.add_argument(
        "--ssh-connect-timeout", type=int, default=DEFAULT_SSH_CONNECT_TIMEOUT,
        help=(
            "SSH connect timeout in seconds "
            f"(default: {DEFAULT_SSH_CONNECT_TIMEOUT})"
        ),
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
    parser.add_argument(
        "--hmc-mode",
        default=DEFAULT_HMC_MODE,
        choices=["surf-datum-aware", "linear-wellhead", "surf-nad27"],
        help=(
            "HMC conversion mode: 'surf-datum-aware' (default) applies "
            "WGS84->NAD27 before HMC conversion; 'surf-nad27' uses raw "
            "SURF_converter; 'linear-wellhead' matches /tmp/plot_hypocenters.py"
        ),
    )
    parser.add_argument(
        "--station-output",
        default=DEFAULT_STATION_OUTPUT,
        help=f"Local station CSV output path (default: {DEFAULT_STATION_OUTPUT})",
    )
    parser.add_argument(
        "--station-rsync-target",
        default=DEFAULT_STATION_RSYNC_TARGET,
        help=(
            "rsync destination for station CSV user@host:path "
            f"(default: {DEFAULT_STATION_RSYNC_TARGET})"
        ),
    )
    args = parser.parse_args()

    _lock_handle = acquire_lock(args.lock_file)
    log.debug("Acquired lock file %s", args.lock_file)
    log.info("Using HMC conversion mode: %s", args.hmc_mode)

    catalog = fetch_catalog(args.fdsn_url, args.days, args.lat, args.lon, args.radius)
    catalog = annotate_hmc(catalog, args.hmc_mode)
    write_catalog(catalog, args.output)
    push_to_vm(
        args.output,
        args.rsync_target,
        args.ssh_key,
        args.rsync_timeout,
        args.ssh_connect_timeout,
    )

    station_rows = fetch_stations_hmc(args.fdsn_url, args.hmc_mode)
    if station_rows:
        write_station_csv(station_rows, args.station_output)
        push_to_vm(
            args.station_output,
            args.station_rsync_target,
            args.ssh_key,
            args.rsync_timeout,
            args.ssh_connect_timeout,
        )


if __name__ == "__main__":
    main()
