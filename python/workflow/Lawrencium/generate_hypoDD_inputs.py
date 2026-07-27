#!/usr/bin/env python
"""
generate_hypoDD_inputs.py
--------------------------
Generate HypoDD input files (dt.ct, dt.cc, event.dat, phase.dat, station.dat)
from a Smackover detection Party object and extracted waveforms.

Uses eqcorrscan.utils.catalog_to_dd functions to convert detection waveforms
into HypoDD-format files suitable for double-difference relocation.

Applies quality filters:
  - Removes detections from excluded templates (TEMPLATE_EXCLUSIONS)
  - Removes detections on spike days (SPIKE_DAY_EXCLUSIONS)

The detections are promoted to "events" by creating fake origins at estimated
depths, then used to generate HypoDD input files.

Usage:
    conda run -n py311 python generate_hypoDD_inputs.py
    or
    /home/chopp/miniconda3/envs/py311/bin/python generate_hypoDD_inputs.py

Output files (in OUTPUT_DIR):
    - dt.ct           (catalog-based differential times)
    - dt.cc           (cross-correlation refined differential times)
    - event.dat       (hypocenter data)
    - phase.dat       (phase picks with travel times)
    - station.dat     (station coordinates)
    - hypoDD_summary.txt  (processing report)
"""

import argparse
import logging
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import numpy as np
from obspy import read_events, read_inventory, read, UTCDateTime, Catalog
from obspy.core.event import (
    Event, Origin, Magnitude, CreationInfo, EventDescription, Pick, WaveformStreamID
)
from obspy.core.inventory import Inventory
from obspy.core.stream import Stream
from obspy.clients.fdsn import Client

# EQcorrscan imports
from eqcorrscan import Party
from eqcorrscan.core.match_filter import Family, Template
from eqcorrscan.utils import catalog_to_dd
from eqcorrscan.utils.pre_processing import multi_process

# ── Monkey-patch Family._process_streams ──────────────────────────────────────
# EQcorrscan's _process_streams calls stream.merge() (default method=0) on the
# pre_processed=True branch, then immediately calls .split().  The default merge
# fills every gap with a masked array covering the full time span between
# non-adjacent traces of the same SEED ID.  For a family whose detections span
# many years this allocates 10s–100s of GB.  Since .split() follows immediately,
# gap-filling is pointless; method=-1 (remove exact overlaps/duplicates only)
# produces the same result without the memory bomb.
_orig_process_streams = Family._process_streams

def _patched_process_streams(self, stream, pre_processed, **kwargs):
    if pre_processed:
        return stream.merge(method=-1).split()
    return _orig_process_streams(self, stream, pre_processed, **kwargs)

Family._process_streams = _patched_process_streams
# ─────────────────────────────────────────────────────────────────────────────

# ── Monkey-patch catalog_to_dd._compute_dt_correlations ───────────────────────
# Bug in installed EQcorrscan: when a master event has no neighbors within
# MAX_SEP (sub_catalog is empty), _compute_dt_correlations still tries to
# create a Pool with len(event_ids)=0 when max_workers > 1, raising:
#   ValueError: Number of processes must be at least 1
# Guard against the empty-catalog case by returning immediately.
_orig_compute_dt_correlations = catalog_to_dd._compute_dt_correlations

def _safe_compute_dt_correlations(catalog, master, *args, **kwargs):
    if not catalog:
        return []
    return _orig_compute_dt_correlations(catalog, master, *args, **kwargs)

catalog_to_dd._compute_dt_correlations = _safe_compute_dt_correlations
# ─────────────────────────────────────────────────────────────────────────────

# ── Monkey-patch catalog_to_dd.write_station ───────────────────────────────────
# Deduplicate stations by (code, latitude, longitude) to avoid redundant entries
# in station.dat when the same station appears in multiple networks or is
# duplicated in the inventory.
_orig_write_station = catalog_to_dd.write_station

def _dedup_write_station(inventory, use_elevation=False, filename="station.dat"):
    """Write station.dat with automatic deduplication."""
    station_strings = []
    seen = set()
    formatter = "{sta:<7s} {lat:>9.5f} {lon:>10.5f}"
    if use_elevation:
        formatter = " ".join([formatter, "{elev:>5.0f}"])

    for network in inventory.networks:
        for station in network.stations:
            # Deduplicate by code and coordinates rounded to 5 decimals (matching formatter)
            key = (station.code, round(station.latitude, 5), round(station.longitude, 5))
            if key in seen:
                continue
            seen.add(key)
            
            parts = dict(sta=station.code, lat=station.latitude,
                         lon=station.longitude)
            if use_elevation:
                channel_depths = {chan.depth for chan in station.channels}
                if len(channel_depths) == 0:
                    depth = 0.0
                else:
                    depth = channel_depths.pop()
                if len(channel_depths) > 1:
                    pass  # Multiple depths warning omitted to reduce log noise
                parts.update(dict(elev=station.elevation - depth))
            station_strings.append(formatter.format(**parts))
    
    with open(filename, "w") as f:
        f.write("\n".join(station_strings))

catalog_to_dd.write_station = _dedup_write_station
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# Input paths
PARTY_FILE = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/Smackover_analyzed_decluter10.tgz"
)
INVENTORY_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/templates/tribe_analysis"
    "/station_inventory.xml"
)

# FDSN Client & cache settings for lag_calc raw streams
FDSN_CLIENT_NAME = "IRIS"
FDSN_CACHE_DIR = "./fdsn_cache"
FETCH_BEFORE = 10.0   # seconds before detection time
FETCH_AFTER = 30.0   # seconds after detection time

# Unified Picking Parameters
LAG_MIN_CC = 0.4       # Slack correlation threshold for both lag_calc and dt.cc
LAG_SHIFT_LEN = 0.5   # Shift len for both lag_calc and dt.cc

# Output directory
OUTPUT_DIR = "./"

MIN_CHANS = 1

DEFAULT_DEPTH_KM = 5.0  # Default depth for events without depth estimates

# Number of detections to process per lag_calc call within a family.
# Caps peak memory to: CHUNK_SIZE × n_template_channels × window_length × samp_rate × 4 bytes.
# E.g. 50 × 2 × 40s × 100sps × 4B ≈ 1.6 MB per chunk stream. Tune up if memory permits.
LAG_CALC_CHUNK_SIZE = 50

# Quality filters (from plot_smackover_detections.py)
# Templates to exclude entirely (noise/artefacts)
TEMPLATE_EXCLUSIONS = [
    "us2000h85v",
    "tx2025qlwgec",
    "us70003tzm",
    "us6000e1q3",
    "tx2023zock",
    "tx2024ywip",
    "tx2024zbdb",
    "tx2024zocv",
    "tx2024yvww",
    "us6000pi49",
    "us70008ee1",
]

# Spike days to exclude per template (YYYY-MM-DD dates)
SPIKE_DAY_EXCLUSIONS: Dict[str, List[str]] = {
    "nm60081223": ["2015-03-08", "2013-02-14"],
    "nm60351847": ["2015-03-08", "2013-02-14", "2013-02-07"],
    "nm60120628": ["2023-06-26", "2023-08-08"],
    "nm60080523": ["2012-12-25", "2013-02-07", "2013-02-14", "2020-09-14"],
    "us70003tzm": ["2012-12-25", "2013-02-07"],
    "us7000rfpr": ["2012-12-25"],
    "us6000m33c": ["2012-12-25"],
    "us6000pkzk": ["2012-12-25", "2013-02-07"],
    "nm60163943": ["2020-09-14"],
    "tx2024ywip": ["2012-12-25"],
    "us70008ee1": ["2012-12-25"],
    "us6000e1z3": ["2013-02-07"],
    "us6000dy5c": ["2013-02-07"],
}

# HypoDD parameters
MAX_SEP = 25.0          # km, max hypocentral separation to link events
MIN_LINK = 1           # minimum shared phase observations

# Correlation parameters for dt.cc
EXTRACT_LEN = 3.0      # seconds around pick
PRE_PICK = 0.5         # seconds before pick
LOWCUT = 0.5           # Hz
HIGHCUT = 19.0         # Hz

# Advanced options
MAX_WORKERS = 1     # None = auto; set to limit parallel processing
USE_ELEVATION = True   # Include elevation in station.dat if available
PARALLEL_PROCESS = False  # Process streams in parallel (memory intensive)
WEIGHT_BY_SQUARE = True   # Weight by correlation² vs raw correlation

# ─────────────────────────────────────────────────────────────────────────────
# ── LOGGING SETUP ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ── UTILITY FUNCTIONS ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def load_party(party_path: str) -> Party:
    """
    Load a EQcorrscan Party object from a tgz/tar file.
    
    Parameters
    ----------
    party_path : str
        Path to party.tgz file.
        
    Returns
    -------
    Party
        EQcorrscan Party object.
        
    Raises
    ------
    FileNotFoundError
        If party file does not exist.
    """
    log.info(f"Loading Party from: {party_path}")
    
    if not os.path.exists(party_path):
        raise FileNotFoundError(f"Party file not found: {party_path}")
    
    try:
        # Load party without the detection catalog to avoid massive memory usage (OOM)
        # and extremely slow linear lookups in _read_family (O(N*M)).
        # Since lag_calc immediately overwrites/regenerates all detection events/picks,
        # loading the catalog.xml from the archive is completely redundant.
        party = Party().read(party_path, read_detection_catalog=False)
    except Exception as exc:
        log.error(f"Failed to read Party: {exc}")
        raise
    
    log.info(f"  Loaded Party with {len(party.families)} families.")
    total_detections = sum(len(f.detections) for f in party.families)
    log.info(f"  Total detections: {total_detections}")
    
    return party


def fix_event_origins(
    catalog: Catalog, template_map: Dict[str, Template]
) -> Catalog:
    """
    Fix event origins using template coordinates and detection times.
    
    For each lag_calc detection event in the catalog:
    - Extract template name from resource_id
    - Look up template in template_map (EQcorrscan Template object)
    - Use template's location (latitude, longitude, depth) for event's origin
    - Compute origin.time = template_origin.time + (detect_time - t_template_start)
      This produces travel times (pick.time - origin.time) equal to template travel
      times adjusted by per-channel lag_calc shifts.
    
    Parameters
    ----------
    catalog : Catalog
        Catalog with events from lag_calc.
    template_map : Dict[str, Template]
        Mapping of template name -> EQcorrscan Template object.
        
    Returns
    -------
    Catalog
        Updated catalog with origins added/fixed.
    """
    log.info(f"Fixing event origins using {len(template_map)} templates...")
    
    n_fixed = 0
    n_failed = 0
    
    for event in catalog:
        # Extract resource_id string (format: smi:local/{template_name}_{YYYYMMDD}_{HHMMSSNNNNNN})
        # or just {template_name}_{YYYYMMDD}_{HHMMSSNNNNNN}
        event_id_str = str(event.resource_id.id)
        if "/" in event_id_str:
            event_id_short = event_id_str.split("/")[1]
        else:
            event_id_short = event_id_str
        
        # Parse template name and detection time from resource_id
        # Format: template_name_YYYYMMDD_HHMMSSNNNNNN
        # Split on '_' to get [...template_parts, YYYYMMDD, HHMMSSNNNNNN]
        # Handle template names with underscores by using last two tokens for date/time
        tokens = event_id_short.split("_")
        if len(tokens) < 3:
            log.debug(
                f"  Malformed resource_id {event.resource_id.id} "
                f"(expected at least 3 underscore-separated tokens)"
            )
            n_failed += 1
            continue
        
        # Last two tokens: YYYYMMDD and HHMMSSNNNNNN
        datetime_token = tokens[-2]  # YYYYMMDD
        time_token = tokens[-1]       # HHMMSSNNNNNN (12 digits: HHMMSS + 6-digit microseconds)
        template_name = "_".join(tokens[:-2])  # Everything before the date
        
        # Validate and parse datetime tokens
        if len(datetime_token) != 8 or len(time_token) != 12:
            log.debug(
                f"  Malformed date/time tokens in {event.resource_id.id}: "
                f"datetime={datetime_token}, time={time_token}"
            )
            n_failed += 1
            continue
        
        try:
            year = int(datetime_token[0:4])
            month = int(datetime_token[4:6])
            day = int(datetime_token[6:8])
            hour = int(time_token[0:2])
            minute = int(time_token[2:4])
            second = int(time_token[4:6])
            microsecond = int(time_token[6:12])
            detect_time = UTCDateTime(year, month, day, hour, minute, second, microsecond)
        except (ValueError, IndexError) as exc:
            log.debug(
                f"  Failed to parse detection time from {event.resource_id.id}: {exc}"
            )
            n_failed += 1
            continue
        
        # Lookup template
        if template_name not in template_map:
            log.debug(
                f"  Template '{template_name}' not in template_map for event {event.resource_id.id}"
            )
            n_failed += 1
            continue
        
        template = template_map[template_name]
        
        # Get template origin
        try:
            template_origin = (
                template.event.preferred_origin()
                or template.event.origins[0]
            )
        except (IndexError, AttributeError, TypeError):
            log.warning(f"  Template {template_name} has no valid origin")
            n_failed += 1
            continue
        
        # Compute template trace start time (minimum starttime across all traces)
        try:
            t_template_start = min(tr.stats.starttime for tr in template.st)
        except (ValueError, AttributeError):
            log.warning(f"  Template {template_name} has no traces or invalid trace stats")
            n_failed += 1
            continue
        
        # Compute detection origin time using the formula:
        # origin_time = template_origin.time + (detect_time - t_template_start)
        # This makes: tt_detection = pick.time - origin_time
        #                          = (pick.time - t_template_start + shift)
        #                          = (template_tt + shift) ✓
        event_time = template_origin.time + (detect_time - t_template_start)
        
        # Create/update origin with template coords and computed detection time
        new_origin = Origin(
            time=event_time,
            latitude=template_origin.latitude,
            longitude=template_origin.longitude,
            depth=template_origin.depth,
            creation_info=CreationInfo(
                author="generate_hypoDD_inputs.py",
                creation_time=UTCDateTime.now()
            )
        )
        
        # Clear existing origins and add new one
        event.origins = [new_origin]
        event.preferred_origin_id = new_origin.resource_id
        
        n_fixed += 1
    
    log.info(f"  Fixed {n_fixed} event origins")
    if n_failed > 0:
        log.warning(f"  Failed to fix {n_failed} events")
    
    return catalog


def load_and_validate_inventory(inventory_path: str) -> Inventory:
    """
    Load and validate the station inventory.
    
    Parameters
    ----------
    inventory_path : str
        Path to StationXML file.
        
    Returns
    -------
    Inventory
        ObsPy Inventory object.
        
    Raises
    ------
    FileNotFoundError
        If inventory file does not exist.
    Exception
        If inventory is invalid or unreadable.
    """
    log.info(f"Loading station inventory from: {inventory_path}")
    
    if not os.path.exists(inventory_path):
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")
    
    try:
        inventory = read_inventory(inventory_path)
    except Exception as exc:
        log.error(f"Failed to read inventory: {exc}")
        raise
    
    n_stations = sum(len(net.stations) for net in inventory.networks)
    log.info(f"  Loaded {len(inventory.networks)} networks with {n_stations} stations")
    
    return inventory


def apply_quality_filters_party(party: Party) -> Party:
    """
    Apply quality filters to the Party:
      1. Remove families whose templates are in TEMPLATE_EXCLUSIONS
      2. Remove individual detections on spike days (SPIKE_DAY_EXCLUSIONS)
      3. Remove individual detections with fewer than MIN_CHANS template channels
    
    Parameters
    ----------
    party : Party
        Input Party.
        
    Returns
    -------
    Party
        Filtered Party (rebuilt with filtered families).
    """
    log.info("Applying quality filters to Party...")
    
    # ── Filter 1: Whole-template exclusions ──────────────────────────────────
    n_before_templates = len(party.families)
    filtered_families = []
    for family in party.families:
        if family.template.name in TEMPLATE_EXCLUSIONS:
            continue
        filtered_families.append(family)
        
    n_dropped_templates = n_before_templates - len(filtered_families)
    log.info(
        f"  Template exclusions: dropped {n_dropped_templates} families "
        f"for {len(TEMPLATE_EXCLUSIONS)} templates"
    )
    
    # ── Filter 2 & 3: Spike-day and MIN_CHANS ───────────────────────────────
    final_families = []
    total_original = 0
    total_kept = 0
    
    for family in filtered_families:
        template_name = family.template.name
        spike_dates = SPIKE_DAY_EXCLUSIONS.get(template_name, [])
        
        filtered_detections = []
        for detection in family.detections:
            total_original += 1
            
            # Spike day filter
            detect_date_str = detection.detect_time.datetime.strftime("%Y-%m-%d")
            if detect_date_str in spike_dates:
                continue
            
            # MIN_CHANS check (detection.chans is list of (station, channel) tuples)
            if len(detection.chans) < MIN_CHANS:
                continue
                
            filtered_detections.append(detection)
            
        total_kept += len(filtered_detections)
        
        # Only keep family if it contains detections after filtering
        if filtered_detections:
            # Recreate Family with its template and filtered detections list
            new_family = Family(
                template=family.template,
                detections=filtered_detections
            )
            final_families.append(new_family)
            
    filtered_party = Party(families=final_families)
    
    log.info(f"  Original detections: {total_original:,}")
    log.info(f"  Final filtered detections: {total_kept:,}")
    log.info(f"  Detections removed: {total_original - total_kept:,}")
    
    return filtered_party


def fetch_family_streams(
    family: Family,
    client: Client,
    cache_dir: str,
    fetch_before: float,
    fetch_after: float,
) -> Stream:
    """
    Fetch and return a merged stream for the given family.
    
    Checks cache_dir first. If cached mseed file exists, reads it and processes it.
    Otherwise, fetches from FDSN client, saves raw to cache, and processes it.
    
    Processes each detection individually, then merges with method=-1 to remove
    only exact duplicates/overlaps. NEVER use merge() or merge(method=0) — for a
    family whose detections span many years, they would allocate a masked array
    covering the entire time range (potentially TBs of RAM) to bridge gaps between
    short detection windows. lag_calc handles gappy multi-trace streams fine.
    """
    os.makedirs(cache_dir, exist_ok=True)
    family_stream = Stream()
    
    # We need the SEED identifiers from the template traces
    template_seed_ids = []
    for tr in family.template.st:
        template_seed_ids.append((
            tr.stats.network,
            tr.stats.station,
            tr.stats.location,
            tr.stats.channel
        ))
        
    lowcut = getattr(family.template, "lowcut", 1.0)
    highcut = getattr(family.template, "highcut", 20.0)
    filt_order = getattr(family.template, "filt_order", 4)
    samp_rate = getattr(family.template, "samp_rate", 100.0)
        
    log.info(
        f"  Fetching/loading streams for family {family.template.name} "
        f"({len(family.detections)} detections, {len(template_seed_ids)} template channels)..."
    )
    
    n_cached = 0
    n_fetched = 0
    n_failed = 0
    
    for detection in family.detections:
        cache_path = os.path.join(cache_dir, f"{detection.id}_raw.mseed")
        
        detect_time = detection.detect_time
        start_time = detect_time - fetch_before
        end_time = detect_time + fetch_after
        
        # Check cache
        if os.path.exists(cache_path):
            try:
                st = read(cache_path)
                # Pre-process the cached raw stream immediately using template parameters
                processed_st = multi_process(
                    st=st,
                    lowcut=lowcut,
                    highcut=highcut,
                    filt_order=filt_order,
                    samp_rate=samp_rate,
                    starttime=start_time,
                    endtime=end_time,
                    ignore_length=True,
                    ignore_bad_data=True
                )
                family_stream += processed_st
                n_cached += 1
                continue
            except Exception as exc:
                log.warning(f"    Failed to read/process cache {cache_path}, will re-fetch: {exc}")
                
        # Cache miss - fetch from FDSN
        bulk_request = []
        for net, sta, loc, cha in template_seed_ids:
            bulk_request.append((net, sta, loc, cha, start_time, end_time))
            
        try:
            log.debug(f"    FDSN bulk fetch for detection {detection.id}...")
            st = client.get_waveforms_bulk(bulk_request)
            
            # Simple check: make sure we got some data
            if len(st) > 0:
                st.write(cache_path, format="MSEED")
                # Pre-process the newly fetched raw stream immediately
                processed_st = multi_process(
                    st=st,
                    lowcut=lowcut,
                    highcut=highcut,
                    filt_order=filt_order,
                    samp_rate=samp_rate,
                    starttime=start_time,
                    endtime=end_time,
                    ignore_length=True,
                    ignore_bad_data=True
                )
                family_stream += processed_st
                n_fetched += 1
            else:
                log.warning(f"    No data returned for detection {detection.id}")
                n_failed += 1
        except Exception as exc:
            log.error(f"    Failed fetching FDSN data for detection {detection.id}: {exc}")
            n_failed += 1
            
    log.info(
        f"  Family stream loading summary: {n_cached} from cache, "
        f"{n_fetched} fetched, {n_failed} failed. Total traces: {len(family_stream)}"
    )
    
    # Stream will be deduplicated by _process_streams when lag_calc runs.
    # Do not merge here to avoid gap-filling on multi-year spans.
    return family_stream


def run_lag_calc_all_families(
    party: Party,
    client: Client,
    cache_dir: str,
    fetch_before: float,
    fetch_after: float,
    plot: bool = False,
    plotdir: Optional[str] = None,
) -> Catalog:
    """
    Run lag_calc on each family in chunks of LAG_CALC_CHUNK_SIZE detections.

    Processing the whole family stream at once can exhaust memory for large families
    (e.g. 452 detections × 2 channels = 904 traces held simultaneously). Chunking
    caps peak memory to chunk_size × n_channels × window traces at any one time.
    """
    import gc
    log.info(
        f"Running lag_calc on {len(party.families)} families with "
        f"min_cc={LAG_MIN_CC}, shift_len={LAG_SHIFT_LEN}, chunk_size={LAG_CALC_CHUNK_SIZE}..."
    )

    compiled_catalog = Catalog()

    for family in party.families:
        detections = family.detections
        n_det = len(detections)
        log.info(
            f"Processing family: {family.template.name} "
            f"({n_det} detections in chunks of {LAG_CALC_CHUNK_SIZE})"
        )

        family_events = 0
        for chunk_start in range(0, n_det, LAG_CALC_CHUNK_SIZE):
            chunk = detections[chunk_start: chunk_start + LAG_CALC_CHUNK_SIZE]
            chunk_end = chunk_start + len(chunk)
            log.info(
                f"  Chunk {chunk_start + 1}-{chunk_end} / {n_det} "
                f"for family {family.template.name}"
            )

            # Build a temporary sub-Family for just this chunk so that
            # fetch_family_streams only fetches/loads this chunk's traces.
            chunk_family = Family(template=family.template, detections=chunk)

            chunk_stream = fetch_family_streams(
                chunk_family, client, cache_dir, fetch_before, fetch_after
            )

            if len(chunk_stream) == 0:
                log.warning(
                    f"  No streams for chunk {chunk_start + 1}-{chunk_end}, skipping."
                )
                del chunk_stream
                gc.collect()
                continue

            try:
                chunk_catalog = chunk_family.lag_calc(
                    stream=chunk_stream,
                    pre_processed=True,
                    shift_len=LAG_SHIFT_LEN,
                    min_cc=LAG_MIN_CC,
                    interpolate=True,
                    use_new_resamp_method=True,
                    ignore_length=True,
                    ignore_bad_data=True,
                    plot=plot,
                    plotdir=plotdir,
                )
                family_events += len(chunk_catalog)
                compiled_catalog += chunk_catalog
            except Exception as exc:
                log.error(
                    f"  Error in lag_calc chunk {chunk_start + 1}-{chunk_end} "
                    f"for {family.template.name}: {exc}",
                    exc_info=True,
                )
            finally:
                del chunk_stream
                gc.collect()

        log.info(
            f"  ✓ Family {family.template.name}: "
            f"{family_events} events from {n_det} detections"
        )

    # Print picks summary
    pick_counts = [len(e.picks) if e.picks else 0 for e in compiled_catalog]
    log.info(f"lag_calc completed: {len(compiled_catalog)} total events with picks.")
    if pick_counts:
        log.info(
            f"  Picks per event: min={min(pick_counts)}, "
            f"max={max(pick_counts)}, median={int(np.median(pick_counts))}"
        )

    return compiled_catalog


def filter_catalog_by_cc(catalog: Catalog, min_cc: float, min_chans: int) -> Catalog:
    """
    Filter picks in a catalog keeping only those with correlation coefficient >= min_cc.
    If an event has fewer than min_chans picks after filtering, the event is omitted.
    """
    filtered_catalog = Catalog()
    
    for event in catalog:
        # Create a copy of the event (deep pick data copied)
        event_copy = event.copy()
        
        filtered_picks = []
        for pick in event_copy.picks:
            # Check comment texts for 'cc_max=<value>'
            pick_cc = None
            for comment in pick.comments:
                if comment.text.startswith("cc_max="):
                    try:
                        pick_cc = float(comment.text.split("=")[1])
                    except (IndexError, ValueError):
                        pass
                    break
            
            # If we couldn't find/parse the comment, keep it as fallback (assume 1.0)
            if pick_cc is None:
                log.debug(f"      No cc_max found in comments for pick {pick.resource_id}, keeping.")
                filtered_picks.append(pick)
            elif pick_cc >= min_cc:
                filtered_picks.append(pick)
                
        if len(filtered_picks) >= min_chans:
            event_copy.picks = filtered_picks
            filtered_catalog.append(event_copy)
            
    return filtered_catalog


def _extract_template_name(event: Event) -> str:
    """
    Extract template name from event resource_id.
    
    Resource_id format: smi:local/{template_name}_{YYYYMMDD}_{HHMMSSNNNNNN}
    Example: smi:local/nm60163943_20210306_102345100000
    
    Parameters
    ----------
    event : Event
        ObsPy Event object.
        
    Returns
    -------
    str
        Template name, or empty string if parse fails.
    """
    try:
        event_id_full = str(event.resource_id.id)
        
        # Remove 'smi:local/' prefix if present
        if "/" in event_id_full:
            event_id_short = event_id_full.split("/", 1)[1]
        else:
            event_id_short = event_id_full
        
        # Split on first underscore to get template name
        if "_" not in event_id_short:
            return ""
        
        parts = event_id_short.split("_", 1)
        if len(parts) >= 1:
            return parts[0]
    except Exception:
        pass
    
    return ""


def build_stream_dict(
    catalog: Catalog, cache_dir: str
) -> Tuple[Dict[str, Stream], int, int]:
    """
    Build a dictionary of streams keyed by event resource_id.
    
    Loads, trims, detrends, and bandpass filters cached raw streams to keep
    memory usage extremely low and bypass slow/redundant filtering loops.
    
    Parameters
    ----------
    catalog : Catalog
        Input catalog.
    cache_dir : str
        FDSN cache directory.
        
    Returns
    -------
    Tuple[Dict[str, Stream], int, int]
        (stream_dict, n_loaded, n_missing) where:
        - stream_dict: {event.resource_id.id: Stream}
        - n_loaded: number of events with waveforms found
        - n_missing: number of events with no waveforms
    """
    log.info(f"DEBUG: Entered build_stream_dict()")
    
    stream_dict = {}
    n_loaded = 0
    n_missing = 0
    
    if not (cache_dir and os.path.isdir(cache_dir)):
        log.error("Valid FDSN cache directory does not exist.")
        raise ValueError("Must provide a valid FDSN cache directory.")
    
    log.info(f"Loading, trimming, and pre-filtering streams directly from FDSN cache: {cache_dir}")
    
    for event in catalog:
        event_id_full = str(event.resource_id.id)
        
        # Extract the part after '/' to remove 'smi:local/' prefix
        # Format: smi:local/{template_name}_{YYYYMMDD}_{HHMMSSNNNNNN}
        # e.g., smi:local/nm60163943_20210306_102345100000
        if "/" in event_id_full:
            event_id_short = event_id_full.split("/")[1]
        else:
            event_id_short = event_id_full
            
        # ISO resource_id timestamp → detection.id filename format
        # nm60163943_20210306T102345.100000 → nm60163943_20210306_102345100000
        detection_id = event_id_short.replace('T', '_').replace('.', '')
        cache_path = os.path.join(cache_dir, f"{detection_id}_raw.mseed")
        if os.path.exists(cache_path):
            try:
                st = read(cache_path)
                
                # Trim to a tight window around the picks to save 90% of waveform memory
                if event.picks:
                    pick_times = [p.time for p in event.picks]
                    # Required slicing is typically pick.time - 1.0 to pick.time + 3.0.
                    # We add a 2.0s buffer on start and a 4.0s buffer on end for safety.
                    t_start = min(pick_times) - 2.0
                    t_end = max(pick_times) + 4.0
                    st.trim(starttime=t_start, endtime=t_end, pad=False)
                
                # Pre-apply detrend and bandpass filtering here
                st.detrend()
                if LOWCUT is not None and HIGHCUT is not None:
                    st.filter("bandpass", freqmin=LOWCUT, freqmax=HIGHCUT, corners=4, zerophase=True)
                elif LOWCUT is None and HIGHCUT is not None:
                    st.filter("lowpass", freq=HIGHCUT, corners=4, zerophase=True)
                elif LOWCUT is not None and HIGHCUT is None:
                    st.filter("highpass", freq=LOWCUT, corners=4, zerophase=True)
                
                # Merge segments after filtering
                st.merge()
                
                stream_dict[event_id_full] = st
                n_loaded += 1
                continue
            except Exception as exc:
                log.debug(f"  Failed to read/process cached stream {event_id_short}_raw.mseed: {exc}")
        log.debug(f"  No cached waveform found in {cache_dir} for {event_id_short}")
        n_missing += 1
        continue
    
    log.info(f"  Loaded and processed waveforms: {n_loaded}")
    log.info(f"  Missing waveforms: {n_missing}")
    
    return stream_dict, n_loaded, n_missing


def generate_hypodd_files(
    catalog: Catalog,
    inventory: Inventory,
    stream_dict: Dict[str, Stream],
    output_dir: str,
    min_cc_val: float,
) -> Dict[str, any]:
    """
    Generate all HypoDD input files using eqcorrscan.utils.catalog_to_dd.
    
    Parameters
    ----------
    catalog : Catalog
        Filtered input catalog.
    inventory : Inventory
        Station inventory.
    stream_dict : Dict[str, Stream]
        Dictionary of waveforms by event ID.
    output_dir : str
        Output directory for HypoDD files.
    min_cc_val : float
        Min correlation coefficient for cross-correlations.
        
    Returns
    -------
    Dict[str, any]
        Summary statistics of generated files.
    """
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output directory: {output_dir}")
    
    summary = {
        "n_events": len(catalog),
        "n_waveforms": len(stream_dict),
        "files_generated": [],
        "errors": [],
    }
    
    # Change to output directory for file generation
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # ──────────────────────────────────────────────────────────────────────
        # 1. Generate dt.ct (catalog-based differential times)
        # ──────────────────────────────────────────────────────────────────────
        log.info("Generating dt.ct (catalog differential times)...")
        try:
            event_id_mapper = catalog_to_dd.write_catalog(
                catalog, max_sep=MAX_SEP, min_link=MIN_LINK
            )
            if os.path.exists("dt.ct"):
                size_ct = os.path.getsize("dt.ct")
                log.info(f"  ✓ Generated dt.ct ({size_ct} bytes)")
                summary["files_generated"].append("dt.ct")
            else:
                log.warning("  dt.ct was not generated (check event/phase links)")
                summary["errors"].append("dt.ct generation failed")
        except Exception as exc:
            log.error(f"  Error generating dt.ct: {exc}")
            summary["errors"].append(f"dt.ct: {str(exc)}")
            event_id_mapper = None
        
        # ──────────────────────────────────────────────────────────────────────
        # 2. Generate dt.cc (cross-correlation refined differential times)
        # ──────────────────────────────────────────────────────────────────────
        log.info(f"Generating dt.cc (cross-correlation differential times at CC >= {min_cc_val})...")
        if len(stream_dict) > 0:
            try:
                # Pass lowcut=None and highcut=None because we pre-filter and detrend
                # streams during build_stream_dict. This avoids creating copy dictionaries
                # of the wave streams inside write_correlations, drastically reducing memory.
                event_id_mapper = catalog_to_dd.write_correlations(
                    catalog,
                    stream_dict,
                    extract_len=EXTRACT_LEN,
                    pre_pick=PRE_PICK,
                    shift_len=LAG_SHIFT_LEN,
                    event_id_mapper=event_id_mapper,
                    lowcut=None,
                    highcut=None,
                    max_sep=MAX_SEP,
                    min_link=MIN_LINK,
                    min_cc=min_cc_val,
                    interpolate=True,
                    all_horiz=False,
                    max_workers=MAX_WORKERS,
                    parallel_process=PARALLEL_PROCESS,
                    weight_by_square=WEIGHT_BY_SQUARE,
                    max_trace_workers=35,
                )
                if os.path.exists("dt.cc"):
                    size_cc = os.path.getsize("dt.cc")
                    log.info(f"  ✓ Generated dt.cc ({size_cc} bytes)")
                    summary["files_generated"].append("dt.cc")
                else:
                    log.warning("  dt.cc was not generated (check correlation results)")
                    summary["errors"].append("dt.cc generation failed")
            except Exception as exc:
                log.error(f"  Error generating dt.cc: {exc}")
                summary["errors"].append(f"dt.cc: {str(exc)}")
        else:
            log.warning("  Skipping dt.cc: no waveforms available")
            summary["errors"].append("dt.cc: no waveforms")
        
        # ──────────────────────────────────────────────────────────────────────
        # 3. Generate event.dat (hypocenter data)
        # ──────────────────────────────────────────────────────────────────────
        log.info("Generating event.dat (hypocenter data)...")
        try:
            catalog_to_dd.write_event(catalog, event_id_mapper=event_id_mapper)
            if os.path.exists("event.dat"):
                size_ev = os.path.getsize("event.dat")
                log.info(f"  ✓ Generated event.dat ({size_ev} bytes)")
                summary["files_generated"].append("event.dat")
            else:
                log.warning("  event.dat was not generated")
                summary["errors"].append("event.dat generation failed")
        except Exception as exc:
            log.error(f"  Error generating event.dat: {exc}")
            summary["errors"].append(f"event.dat: {str(exc)}")
        
        # ──────────────────────────────────────────────────────────────────────
        # 4. Generate phase.dat (phase picks)
        # ──────────────────────────────────────────────────────────────────────
        log.info("Generating phase.dat (phase picks)...")
        try:
            catalog_to_dd.write_phase(catalog, event_id_mapper=event_id_mapper)
            if os.path.exists("phase.dat"):
                size_ph = os.path.getsize("phase.dat")
                log.info(f"  ✓ Generated phase.dat ({size_ph} bytes)")
                summary["files_generated"].append("phase.dat")
            else:
                log.warning("  phase.dat was not generated")
                summary["errors"].append("phase.dat generation failed")
        except Exception as exc:
            log.error(f"  Error generating phase.dat: {exc}")
            summary["errors"].append(f"phase.dat: {str(exc)}")
        
        # ──────────────────────────────────────────────────────────────────────
        # 5. Generate station.dat (station coordinates)
        # ──────────────────────────────────────────────────────────────────────
        log.info("Generating station.dat (station coordinates)...")
        try:
            catalog_to_dd.write_station(
                inventory, use_elevation=USE_ELEVATION, filename="station.dat"
            )
            if os.path.exists("station.dat"):
                size_st = os.path.getsize("station.dat")
                log.info(f"  ✓ Generated station.dat ({size_st} bytes)")
                summary["files_generated"].append("station.dat")
            else:
                log.warning("  station.dat was not generated")
                summary["errors"].append("station.dat generation failed")
        except Exception as exc:
            log.error(f"  Error generating station.dat: {exc}")
            summary["errors"].append(f"station.dat: {str(exc)}")
    
    finally:
        os.chdir(original_cwd)
    
    return summary


def write_summary_report(
    summary: Dict[str, any],
    filtered_catalog: Catalog,
    stream_dict: Dict[str, Stream],
    output_dir: str,
    original_detections_count: int,
    filtered_detections_count: int,
) -> None:
    """
    Write a text summary report of the processing.
    
    Parameters
    ----------
    summary : Dict[str, any]
        Summary dictionary from generate_hypodd_files().
    filtered_catalog : Catalog
        Filtered catalog after quality checks and lag_calc.
    stream_dict : Dict[str, Stream]
        Stream dictionary.
    output_dir : str
        Output directory.
    original_detections_count : int
        Original detections count.
    filtered_detections_count : int
        Filtered detections count.
    """
    report_path = os.path.join(output_dir, "hypoDD_summary.txt")
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HypoDD Input Files Generation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        # Input files
        f.write("INPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Party File:   {PARTY_FILE}\n")
        f.write(f"Inventory:    {INVENTORY_PATH}\n\n")
        
        # Catalog statistics
        f.write("DETECTION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Detections before filtering:  {original_detections_count}\n")
        f.write(f"Detections after filtering:   {filtered_detections_count}\n")
        f.write(f"Detections removed:           {original_detections_count - filtered_detections_count}\n\n")
        
        # Waveform statistics
        f.write("WAVEFORM STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total detections:             {summary['n_events']}\n")
        f.write(f"Waveforms available:          {summary['n_waveforms']}\n")
        coverage = 100.0 * summary['n_waveforms'] / max(summary['n_events'], 1)
        f.write(f"Coverage:                     {coverage:.1f}%\n\n")
        
        # HypoDD parameters
        f.write("HypoDD PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Max hypocentral separation (km):  {MAX_SEP}\n")
        f.write(f"Min linked phases:                 {MIN_LINK}\n")
        f.write(f"Min correlation coefficient (picking): {LAG_MIN_CC}\n")
        f.write(f"Default event depth (km):         {DEFAULT_DEPTH_KM}\n")
        f.write(f"Extract length (s):                {EXTRACT_LEN}\n")
        f.write(f"Pre-pick time (s):                 {PRE_PICK}\n")
        f.write(f"Max pick shift (s):                {LAG_SHIFT_LEN}\n")
        f.write(f"Bandpass filter (Hz):              {LOWCUT} - {HIGHCUT}\n\n")
        
        # Generated files
        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        if summary["files_generated"]:
            for fname in summary["files_generated"]:
                fpath = os.path.join(output_dir, fname)
                if os.path.exists(fpath):
                    fsize = os.path.getsize(fpath)
                    f.write(f"✓ {fname:15s}  ({fsize:,} bytes)\n")
                else:
                    f.write(f"✗ {fname:15s}  (not found)\n")
        else:
            f.write("No files generated.\n")
        
        f.write("\n")
        
        # Errors/warnings
        if summary["errors"]:
            f.write("WARNINGS/ERRORS\n")
            f.write("-" * 80 + "\n")
            for error in summary["errors"]:
                f.write(f"⚠ {error}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    log.info(f"Summary report written to: {report_path}")


def _regenerate_station(args: argparse.Namespace, threshold_list: list) -> int:
    """
    Regenerate station.dat files only from the inventory.
    
    Loads the station inventory, deduplicates, and writes station.dat for each
    threshold subdirectory. Skips all correlation and waveform operations.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    threshold_list : list
        List of correlation coefficient thresholds.
        
    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    log.info("=" * 80)
    log.info("Regenerating station.dat files only (--regenerate-station)")
    log.info("=" * 80)
    
    try:
        inventory = load_and_validate_inventory(args.inventory)
    except Exception as exc:
        log.error(f"Failed to load inventory: {exc}")
        return 1
    
    for threshold in threshold_list:
        sub_output_dir = os.path.join(args.output, f"min_cc_{threshold:.1f}")
        if not os.path.isdir(sub_output_dir):
            log.warning(f"Output directory does not exist: {sub_output_dir} — skipping")
            continue
        
        log.info(f"Writing station.dat for min_cc={threshold:.2f} in {sub_output_dir}...")
        original_cwd = os.getcwd()
        os.chdir(sub_output_dir)
        try:
            catalog_to_dd.write_station(
                inventory, use_elevation=USE_ELEVATION, filename="station.dat"
            )
            if os.path.exists("station.dat"):
                station_size = os.path.getsize("station.dat")
                log.info(f"  ✓ station.dat written ({station_size:,} bytes)")
            else:
                log.warning("  station.dat was not generated")
        except Exception as exc:
            log.error(f"  Error writing station.dat: {exc}", exc_info=True)
        finally:
            os.chdir(original_cwd)
    
    log.info("=" * 80)
    log.info("station.dat regeneration complete.")
    log.info("=" * 80)
    return 0


def _resume_dtcc(args: argparse.Namespace, threshold_list: list) -> int:
    """
    Resume dt.ct and dt.cc generation from pre-written catalog XMLs.
    
    Skips party loading and lag_calc; loads per-threshold catalogs already
    written to --output and regenerates both dt.ct (catalog differential times)
    and dt.cc (cross-correlation differential times) using FDSN-cached waveforms.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    threshold_list : list
        List of correlation coefficient thresholds.
        
    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    import gc
    
    log.info("=" * 80)
    log.info("Resuming dt.ct and dt.cc generation (--resume-dtcc)")
    log.info("=" * 80)
    
    for threshold in threshold_list:
        catalog_xml = os.path.join(args.output, f"catalog_min_cc_{threshold:.1f}.xml")
        if not os.path.exists(catalog_xml):
            log.error(f"Catalog XML not found: {catalog_xml} — skipping threshold {threshold:.2f}")
            continue
        
        log.info(f"\n{'=' * 60}")
        log.info(f"Threshold min_cc = {threshold:.2f}")
        log.info(f"{'=' * 60}")
        log.info(f"Loading catalog: {catalog_xml}")
        
        try:
            threshold_catalog = read_events(catalog_xml)
            log.info(f"  {len(threshold_catalog)} events loaded.")
        except Exception as exc:
            log.error(f"Failed to load catalog XML: {exc}")
            continue
        
        stream_dict, n_loaded, n_missing = build_stream_dict(
            threshold_catalog, cache_dir=args.fdsn_cache_dir
        )
        log.info(f"  Waveforms: {n_loaded} loaded, {n_missing} missing.")
        
        if n_loaded == 0:
            log.warning(f"  No waveforms found — skipping dt.cc for threshold {threshold:.2f}")
            del stream_dict
            gc.collect()
            continue
        
        sub_output_dir = os.path.join(args.output, f"min_cc_{threshold:.1f}")
        os.makedirs(sub_output_dir, exist_ok=True)
        
        original_cwd = os.getcwd()
        os.chdir(sub_output_dir)
        try:
            # Generate dt.ct (catalog differential times)
            log.info(f"  Generating dt.ct (catalog differential times) ...")
            event_id_mapper = catalog_to_dd.write_catalog(
                threshold_catalog,
                max_sep=MAX_SEP,
                min_link=MIN_LINK,
            )
            if os.path.exists("dt.ct"):
                dt_ct_size = os.path.getsize("dt.ct")
                log.info(f"  ✓ dt.ct written ({dt_ct_size:,} bytes)")
            else:
                log.warning("  dt.ct was not generated")
            
            # Generate dt.cc (cross-correlation differential times)
            log.info(f"  Running write_correlations in {sub_output_dir} ...")
            catalog_to_dd.write_correlations(
                threshold_catalog,
                stream_dict,
                extract_len=EXTRACT_LEN,
                pre_pick=PRE_PICK,
                shift_len=LAG_SHIFT_LEN,
                event_id_mapper=event_id_mapper,
                lowcut=None,      # already filtered in build_stream_dict
                highcut=None,
                max_sep=MAX_SEP,
                min_link=MIN_LINK,
                min_cc=threshold,
                interpolate=True,
                all_horiz=False,
                max_workers=MAX_WORKERS,
                parallel_process=PARALLEL_PROCESS,
                weight_by_square=WEIGHT_BY_SQUARE,
                max_trace_workers=35,
            )
            if os.path.exists("dt.cc"):
                dt_cc_size = os.path.getsize("dt.cc")
                log.info(f"  ✓ dt.cc written ({dt_cc_size:,} bytes)")
            else:
                log.warning("  dt.cc was not generated (no event pairs met criteria)")
        except Exception as exc:
            log.error(f"  Error generating dt files for min_cc={threshold:.2f}: {exc}", exc_info=True)
        finally:
            os.chdir(original_cwd)
            del stream_dict
            gc.collect()
    
    log.info("=" * 80)
    log.info("dt.ct and dt.cc resume complete.")
    log.info("=" * 80)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# ── MAIN EXECUTION ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def main(args: Optional[argparse.Namespace] = None) -> int:
    """
    Main entry point.
    
    Parameters
    ----------
    args : Optional[argparse.Namespace]
        Command-line arguments (if None, parsed from sys.argv).
        
    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--party",
            default=PARTY_FILE,
            help=f"Path to party (.tgz) (default: {PARTY_FILE})",
        )
        parser.add_argument(
            "--inventory",
            default=INVENTORY_PATH,
            help=f"Path to station inventory XML (default: {INVENTORY_PATH})",
        )
        parser.add_argument(
            "--fdsn-client",
            default=FDSN_CLIENT_NAME,
            help=f"FDSN client name (default: {FDSN_CLIENT_NAME})",
        )
        parser.add_argument(
            "--fdsn-cache-dir",
            default=FDSN_CACHE_DIR,
            help=f"FDSN raw stream cache directory (default: {FDSN_CACHE_DIR})",
        )
        parser.add_argument(
            "--output",
            default=OUTPUT_DIR,
            help=f"Output directory (default: {OUTPUT_DIR})",
        )
        parser.add_argument(
            "--max-sep",
            type=float,
            default=MAX_SEP,
            help=f"Max hypocentral separation in km (default: {MAX_SEP})",
        )
        parser.add_argument(
            "--min-link",
            type=int,
            default=MIN_LINK,
            help=f"Min linked phases (default: {MIN_LINK})",
        )
        parser.add_argument(
            "--cc-thresholds",
            default="0.4,0.5,0.6,0.7",
            help="Comma-separated list of correlation coefficient thresholds to generate catalogs and HypoDD inputs for (default: 0.4,0.5,0.6,0.7)",
        )
        parser.add_argument(
            "--plot-lag-calc",
            action="store_true",
            help="Enable granular plotting of waveform repicking during lag_calc",
        )
        parser.add_argument(
            "--plotdir",
            default="./lag_calc_plots",
            help="Directory to save repick diagnostic plots (default: ./lag_calc_plots)",
        )
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging output level (default: INFO)",
        )
        parser.add_argument(
            "--skip-correlation",
            action="store_true",
            help="Skip dt.cc generation (only generate dt.ct)",
        )
        parser.add_argument(
            "--resume-dtcc",
            action="store_true",
            help=(
                "Skip party loading and lag_calc. Load per-threshold catalog XMLs "
                "already written to --output and regenerate dt.ct and dt.cc."
            ),
        )
        parser.add_argument(
            "--regenerate-station",
            action="store_true",
            help="Regenerate station.dat only from inventory (no correlations or waveforms).",
        )
        args = parser.parse_args()
    
    # Parse thresholds list and update config with the lowest threshold for lag_calc
    try:
        threshold_list = sorted([float(x.strip()) for x in args.cc_thresholds.split(",")])
        if not threshold_list:
            raise ValueError()
    except Exception as exc:
        log.error(f"Invalid correlation thresholds list '{args.cc_thresholds}': must be comma-separated floats.")
        return 1
        
    globals()["LAG_MIN_CC"] = min(threshold_list)
    log.info(f"Using LAG_MIN_CC={LAG_MIN_CC:.2f} for baseline lag_calc picking.")
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("eqcorrscan").setLevel(log_level)
    
    # Propagate warnings to help diagnose too-few-points warnings from lag_calc
    import warnings
    warnings.filterwarnings("always", category=UserWarning, module="eqcorrscan")
    
    # Update global variables if custom arguments provided
    if args.max_sep != MAX_SEP:
        globals()["MAX_SEP"] = args.max_sep
    if args.min_link != MIN_LINK:
        globals()["MIN_LINK"] = args.min_link
    if args.party != PARTY_FILE:
        globals()["PARTY_FILE"] = args.party
    if args.inventory != INVENTORY_PATH:
        globals()["INVENTORY_PATH"] = args.inventory
    if args.fdsn_cache_dir != FDSN_CACHE_DIR:
        globals()["FDSN_CACHE_DIR"] = args.fdsn_cache_dir
    
    if args.resume_dtcc:
        return _resume_dtcc(args, threshold_list)
    
    if args.regenerate_station:
        return _regenerate_station(args, threshold_list)
    
    log.info("=" * 80)
    log.info("HypoDD Input Files Generation")
    log.info("=" * 80)
    
    try:
        # Load the Party object
        party = load_party(args.party)
        
        # Check template parameter consistency with configuration
        template_lowcuts = []
        template_highcuts = []
        template_prepicks = []
        template_lengths = []
        for family in party.families:
            if family.template:
                if getattr(family.template, "lowcut", None) is not None:
                    template_lowcuts.append(family.template.lowcut)
                if getattr(family.template, "highcut", None) is not None:
                    template_highcuts.append(family.template.highcut)
                if getattr(family.template, "prepick", None) is not None:
                    template_prepicks.append(family.template.prepick)
                if family.template.st:
                    tr = family.template.st[0]
                    template_lengths.append(tr.stats.npts * tr.stats.delta)
                    
        # Log template attributes overview and check for alignment with CONFIGURATION
        if template_lowcuts:
            log.info("Template Parameter Summary from Loaded Party:")
            log.info(f"  Lowcut (Hz):  min={min(template_lowcuts)}, max={max(template_lowcuts)}")
            log.info(f"  Highcut (Hz): min={min(template_highcuts)}, max={max(template_highcuts)}")
            log.info(f"  Pre-pick (s): min={min(template_prepicks):.3f}, max={max(template_prepicks):.3f}")
            log.info(f"  Length (s):   min={min(template_lengths):.3f}, max={max(template_lengths):.3f}")
            
            mismatches = []
            mean_lowcut = float(np.mean(template_lowcuts))
            mean_highcut = float(np.mean(template_highcuts))
            mean_prepick = float(np.mean(template_prepicks))
            mean_length = float(np.mean(template_lengths))
            
            if abs(mean_lowcut - LOWCUT) > 0.01:
                mismatches.append(f"LOWCUT config ({LOWCUT} Hz) vs Template average ({mean_lowcut:.1f} Hz)")
            if abs(mean_highcut - HIGHCUT) > 0.01:
                mismatches.append(f"HIGHCUT config ({HIGHCUT} Hz) vs Template average ({mean_highcut:.1f} Hz)")
            if abs(mean_prepick - PRE_PICK) > 0.01:
                mismatches.append(f"PRE_PICK config ({PRE_PICK} s) vs Template average ({mean_prepick:.3f} s)")
            if abs(mean_length - EXTRACT_LEN) > 0.01:
                mismatches.append(f"EXTRACT_LEN config ({EXTRACT_LEN} s) vs Template average ({mean_length:.3f} s)")
                
            if mismatches:
                log.warning("⚠ CORRELATION PARAMETER MISMATCH DETECTED! To get identical cross-correlations between lag_calc and write_correlations, config parameters should match the templates:")
                for m in mismatches:
                    log.warning(f"  - {m}")
            else:
                log.info("✓ Correlation parameters are perfectly aligned between configuration and loaded templates.")
        
        # Keep track of original detection count for the summary report
        original_detections_count = sum(len(f.detections) for f in party.families)
        
        # Apply quality filtering to the Party
        filtered_party = apply_quality_filters_party(party)
        filtered_detections_count = sum(len(f.detections) for f in filtered_party.families)
        
        # Load and validate station inventory
        inventory = load_and_validate_inventory(args.inventory)
        
        # Initialize FDSN Client
        client = Client(args.fdsn_client)
        
        # Run lag_calc on each family of the filtered Party using raw waveforms
        filtered_catalog = run_lag_calc_all_families(
            filtered_party,
            client=client,
            cache_dir=args.fdsn_cache_dir,
            fetch_before=FETCH_BEFORE,
            fetch_after=FETCH_AFTER,
            plot=args.plot_lag_calc,
            plotdir=args.plotdir,
        )
        
        # Build the template mapping from Party families
        template_map = {f.template.name: f.template for f in filtered_party.families}
        
        # Fix event origins on the baseline catalog
        filtered_catalog = fix_event_origins(filtered_catalog, template_map)
        
        # Loop through each target threshold to filter catalog and create HypoDD inputs
        for threshold in threshold_list:
            log.info("")
            log.info("=" * 60)
            log.info(f"Processing Threshold Option: min_cc = {threshold:.2f}")
            log.info("=" * 60)
            
            # Filter the catalog picks by the threshold, enforcing the same MIN_CHANS filter
            threshold_catalog = filter_catalog_by_cc(filtered_catalog, min_cc=threshold, min_chans=MIN_CHANS)
            log.info(f"  Picks filtering completed: {len(threshold_catalog)} events meet criteria.")
            
            # Write out catalog XML
            catalog_xml_path = os.path.join(args.output, f"catalog_min_cc_{threshold:.1f}.xml")
            try:
                threshold_catalog.write(catalog_xml_path, format="QUAKEML")
                log.info(f"  ✓ Saved sub-catalog to {catalog_xml_path}")
            except Exception as exc:
                log.error(f"  Failed writing catalog XML for min_cc={threshold:.1f}: {exc}")
            
            # Create a threshold-specific subdirectory
            sub_output_dir = os.path.join(args.output, f"min_cc_{threshold:.1f}")
            
            # Build stream dictionary for threshold-specific catalog to avoid OOM
            if not args.skip_correlation:
                log.info("Loading workflows for threshold catalog from FDSN cache...")
                threshold_stream_dict, n_loaded, n_missing = build_stream_dict(
                    threshold_catalog, cache_dir=args.fdsn_cache_dir
                )
            else:
                threshold_stream_dict = {}
                
            # Generate HypoDD input files for this threshold
            summary = generate_hypodd_files(
                threshold_catalog,
                inventory,
                threshold_stream_dict,
                sub_output_dir,
                min_cc_val=threshold,
            )
            
            # Write summary report inside the subdirectory
            write_summary_report(
                summary=summary,
                filtered_catalog=threshold_catalog,
                stream_dict=threshold_stream_dict,
                output_dir=sub_output_dir,
                original_detections_count=original_detections_count,
                filtered_detections_count=filtered_detections_count,
            )
            
            log.info(f"  ✓ Subdirectory output generation completed for min_cc={threshold:.1f}")
            
            # Garbage collect to free memory
            del threshold_stream_dict
            import gc
            gc.collect()
        
        log.info("=" * 80)
        log.info("Generation complete!")
        log.info(f"All outputs generated in subdirectories under: {args.output}")
        log.info("=" * 80)
        
        return 0
        
        log.info("=" * 80)
        log.info("Generation complete!")
        log.info(f"Output files in: {args.output}")
        log.info("=" * 80)
        
        return 0
    
    except Exception as exc:
        log.error(f"Fatal error: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
