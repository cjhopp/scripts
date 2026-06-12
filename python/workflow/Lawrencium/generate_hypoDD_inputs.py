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

# EQcorrscan imports
from eqcorrscan import Party
from eqcorrscan.utils import catalog_to_dd

# ─────────────────────────────────────────────────────────────────────────────
# ── CONFIGURATION ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# Input paths
PARTY_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/Smackover_analyzed_raw.tgz"
)
WAVEFORM_DIR = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/waveforms/smackover_north_analyzed/MAD12_2hr"
)
INVENTORY_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/templates/tribe_analysis"
    "/station_inventory.xml"
)

# Output directory
OUTPUT_DIR = "./"

MIN_CHANS = 3

DEFAULT_DEPTH_KM = 5.0  # Default depth for events without depth estimates

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
MIN_LINK = 3           # minimum shared phase observations
MIN_CC = 0.6           # correlation coefficient threshold

# Correlation parameters for dt.cc
EXTRACT_LEN = 3.0      # seconds around pick
PRE_PICK = 0.5         # seconds before pick
SHIFT_LEN = 0.5        # max allowed pick shift (seconds)
LOWCUT = 1.0           # Hz
HIGHCUT = 20.0         # Hz

# Advanced options
MAX_WORKERS = None     # None = auto; set to limit parallel processing
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

def load_party_as_catalog(party_path: str) -> Catalog:
    """
    Load Party file and convert detections to a Catalog for HypoDD processing.
    
    Each Detection already has an Event object with real picks including proper
    network/station/channel codes. We use those directly, preserving all the
    pick information. The no_chans field is preserved via the pick count.
    
    Parameters
    ----------
    party_path : str
        Path to Party .tgz file.
        
    Returns
    -------
    Catalog
        ObsPy Catalog object with events extracted from detections.
    """
    log.info(f"Loading Party from: {party_path}")
    
    if not os.path.exists(party_path):
        raise FileNotFoundError(f"Party file not found: {party_path}")
    
    party = Party().read(party_path, read_detection_catalog=False)
    log.info(f"  Loaded Party with {len(party.families)} families")
    
    events = []
    for family in party.families:
        template_name = family.template.name
        
        for detection in family.detections:
            # Use the Detection's event directly (it has real picks with proper codes)
            if detection.event is None:
                log.warning(f"Detection for {template_name} at {detection.detect_time} has no event, skipping")
                continue
            
            event = detection.event.copy()
            
            # Update origin time to match detection time (more accurate)
            if len(event.origins) > 0:
                event.origins[0].time = detection.detect_time
            else:
                # Fallback: create origin if not present
                template_origin = family.template.event.origins[0] if family.template.event.origins else None
                if template_origin:
                    origin = Origin(
                        time=detection.detect_time,
                        latitude=template_origin.latitude,
                        longitude=template_origin.longitude,
                        depth=template_origin.depth if template_origin.depth else (DEFAULT_DEPTH_KM * 1000),
                    )
                else:
                    origin = Origin(
                        time=detection.detect_time,
                        latitude=0.0,
                        longitude=0.0,
                        depth=DEFAULT_DEPTH_KM * 1000,
                    )
                event.origins.append(origin)
            
            # Update resource_id to include template name and detection time for mapping
            iso_timestamp = detection.detect_time.isoformat()
            event.resource_id = f"smi:local/{template_name}_{iso_timestamp}"
            
            events.append(event)
    
    catalog = Catalog(events=events)
    log.info(f"  Converted to Catalog: {len(catalog)} events")
    
    # Log pick (no_chans) distribution
    pick_counts = [len(e.picks) if e.picks else 0 for e in catalog]
    if pick_counts:
        log.info(f"  picks per event distribution: min={min(pick_counts)}, max={max(pick_counts)}, median={int(np.median(pick_counts))}")
        n_with_3plus = sum(1 for c in pick_counts if c >= 3)
        log.info(f"  Detections with >= 3 picks: {n_with_3plus}/{len(catalog)}")
        
        # Sample a few picks to show they have proper codes
        if len(catalog) > 0 and len(catalog[0].picks) > 0:
            sample_pick = catalog[0].picks[0]
            log.info(f"  Sample pick: {sample_pick.waveform_id.network_code}.{sample_pick.waveform_id.station_code}."
                    f"{sample_pick.waveform_id.location_code}.{sample_pick.waveform_id.channel_code}")
    
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


def apply_quality_filters(catalog: Catalog) -> Tuple[Catalog, Catalog]:
    """
    Apply quality filters to the catalog, matching plot_smackover_detections.py order:
      1. Remove events from excluded templates (TEMPLATE_EXCLUSIONS)
      2. Remove events on spike days (SPIKE_DAY_EXCLUSIONS)
      3. Remove events with fewer than MIN_CHANS picks
    
    Parameters
    ----------
    catalog : Catalog
        Input catalog.
        
    Returns
    -------
    Tuple[Catalog, Catalog]
        (filtered_catalog, raw_catalog) where:
        - raw_catalog: after template/spike-day filters but BEFORE MIN_CHANS
        - filtered_catalog: after all filters including MIN_CHANS
    """
    log.info("Applying quality filters (matching plot_smackover_detections.py)...")
    original_len = len(catalog)
    
    events = list(catalog.events) if catalog.events else []
    n_excluded_template = 0
    n_excluded_spike = 0
    
    # ── Filter 1: Whole-template exclusions ──────────────────────────────────
    if TEMPLATE_EXCLUSIONS:
        n_before = len(events)
        events = [
            e for e in events
            if _extract_template_name(e) not in TEMPLATE_EXCLUSIONS
        ]
        n_excluded_template = n_before - len(events)
        log.info(
            f"  Template exclusions: dropped {n_excluded_template:,} events "
            f"for {len(TEMPLATE_EXCLUSIONS)} templates: {TEMPLATE_EXCLUSIONS}"
        )
    
    # ── Filter 2: Spike-day exclusions (applied BEFORE MIN_CHANS filter) ────────
    if SPIKE_DAY_EXCLUSIONS:
        n_before = len(events)
        events_keep = []
        for event in events:
            template_name = _extract_template_name(event)
            
            # Check if this event is on a spike day for its template
            if template_name in SPIKE_DAY_EXCLUSIONS and len(event.origins) > 0:
                origin_time = event.origins[0].time
                event_date_str = origin_time.datetime.strftime("%Y-%m-%d")
                
                if event_date_str in SPIKE_DAY_EXCLUSIONS[template_name]:
                    continue  # Skip this event
            
            events_keep.append(event)
        
        n_excluded_spike = n_before - len(events_keep)
        events = events_keep
        log.info(
            f"  Spike-day exclusions: dropped {n_excluded_spike:,} events "
            f"({n_before:,} → {len(events):,})"
        )
    
    # ── Create raw_catalog BEFORE MIN_CHANS filter ────────────────────────────
    raw_catalog = Catalog(events=events.copy())
    log.info(f"  Raw catalog (before MIN_CHANS): {len(raw_catalog)} events")
    
    # ── Filter 3: Minimum number of channels (picks) ────────────────────────────
    n_before = len(events)
    events_filtered = [
        e for e in events
        if len(e.picks) >= MIN_CHANS
    ]
    n_excluded_minchan = n_before - len(events_filtered)
    
    filtered_catalog = Catalog(events=events_filtered)
    
    log.info(f"  Original catalog: {original_len} events")
    log.info(f"  Excluded (templates): {n_excluded_template}")
    log.info(f"  Excluded (spike days): {n_excluded_spike}")
    log.info(f"  Excluded (no_chans < {MIN_CHANS}): {n_excluded_minchan}")
    log.info(f"  Final filtered catalog: {len(filtered_catalog)} events")
    
    return filtered_catalog, raw_catalog


def _extract_template_name(event: Event) -> str:
    """
    Extract template name from event resource_id.
    
    Resource_id format: smi:local/{template_name}_{ISO_timestamp}
    Example: smi:local/tx2024istc_20091007T073702.525000
    
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
        
        # Split on last underscore to separate template from timestamp
        if "_" not in event_id_short:
            return ""
        
        parts = event_id_short.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
    except Exception:
        pass
    
    return ""


def build_stream_dict(
    catalog: Catalog, waveform_dir: str
) -> Tuple[Dict[str, Stream], int, int]:
    """
    Build a dictionary of streams keyed by event resource_id.
    
    Maps waveforms from {template_name}_{YYYYMMDD}_{HHMMSSNNNNNN}.mseed files to events.
    resource_id format: {template_name}_{ISO_timestamp}
    waveform filename format: {template_name}_{YYYYMMDD}_{HHMMSSNNNNNN}.mseed
    Example: resource_id = tx2024istc_20091007T073702.525000
             waveform = tx2024istc_20091007_073702525000.mseed
    
    Parameters
    ----------
    catalog : Catalog
        Input catalog.
    waveform_dir : str
        Directory containing .mseed waveform files.
        
    Returns
    -------
    Tuple[Dict[str, Stream], int, int]
        (stream_dict, n_loaded, n_missing) where:
        - stream_dict: {event.resource_id.id: Stream}
        - n_loaded: number of events with waveforms found
        - n_missing: number of events with no waveforms
    """
    log.info(f"DEBUG: Entered build_stream_dict()")
    log.info(f"Building stream dictionary from: {waveform_dir}")
    
    if not os.path.isdir(waveform_dir):
        log.error(f"Waveform directory not found: {waveform_dir}")
        raise FileNotFoundError(f"Waveform directory not found: {waveform_dir}")
    
    stream_dict = {}
    n_loaded = 0
    n_missing = 0
    
    # Get all .mseed files in directory
    waveform_files = sorted(
        [f for f in os.listdir(waveform_dir) if f.endswith(".mseed")]
    )
    
    for event in catalog:
        event_id_full = str(event.resource_id.id)
        
        # Extract the part after '/' to remove 'smi:local/' prefix
        # Format: smi:local/{template_name}_{ISO_timestamp}
        # e.g., smi:local/tx2024istc_20091007T073702.525000
        if "/" in event_id_full:
            event_id_short = event_id_full.split("/")[1]
        else:
            event_id_short = event_id_full
        
        # Parse resource_id format: {template_name}_{ISO_timestamp}
        # e.g., tx2024istc_20091007T073702.525000
        if "_" not in event_id_short:
            log.debug(f"  Cannot parse resource_id: {event_id_short}")
            n_missing += 1
            continue
        
        # Split on the last underscore to separate template from timestamp
        parts = event_id_short.rsplit("_", 1)
        if len(parts) != 2:
            n_missing += 1
            continue
        
        template_name, iso_timestamp = parts
        
        # Parse ISO timestamp to get date/time components
        # ISO format can be: 2013-09-22T23:48:20.975000 or similar
        try:
            if "T" in iso_timestamp:
                date_part, time_part = iso_timestamp.split("T")
            else:
                # Assume format is already date_time
                date_part, time_part = iso_timestamp.split("_", 1) if "_" in iso_timestamp else (iso_timestamp, "00:00:00")
            
            # Remove dashes from date
            date_clean = date_part.replace("-", "")  # YYYY-MM-DD → YYYYMMDD
            
            # Handle time with possible colons and microseconds
            if "." in time_part:
                time_str, microsecond_str = time_part.split(".")
            else:
                time_str = time_part
                microsecond_str = "000000"
            
            # Remove colons from time
            time_clean = time_str.replace(":", "")  # HH:MM:SS → HHMMSS
            
            # Pad/truncate microseconds to 6 digits
            microseconds = microsecond_str[:6].ljust(6, "0")
            
            # Expected format: YYYYMMDD_HHMMSSNNNNNN
            expected_timestamp = f"{date_clean}_{time_clean}{microseconds}"
            
            # Parse for comparison (extract date and time without microseconds)
            expected_yyyymmdd = date_clean
            expected_hhmmss = time_clean
            expected_dt = datetime.strptime(
                f"{expected_yyyymmdd}_{expected_hhmmss}", "%Y%m%d_%H%M%S"
            )
        except Exception as exc:
            log.debug(f"  Failed to parse ISO timestamp {iso_timestamp}: {exc}")
            n_missing += 1
            continue
        
        # Build expected filename
        expected_basename = f"{template_name}_{expected_timestamp}.mseed"
        
        # Try exact match first
        waveform_path = os.path.join(waveform_dir, expected_basename)
        if os.path.exists(waveform_path):
            try:
                st = read(waveform_path)
                stream_dict[event_id_full] = st
                n_loaded += 1
                continue
            except Exception as exc:
                log.debug(f"  Failed to read {expected_basename}: {exc}")
        
        # Try fuzzy match: same template, close timestamp (within 1 second)
        waveform_found = False
        for wf_file in waveform_files:
            if not wf_file.startswith(f"{template_name}_"):
                continue
            
            # Extract timestamp from filename (format varies)
            wf_ts_str = wf_file.replace(f"{template_name}_", "").replace(".mseed", "")
            
            # Try to parse the waveform filename timestamp
            # Could be: YYYYMMDD_HHMMSSNNNNNN or YYYY-MM-DD_HH:MM:SSNNNNNN or similar
            try:
                # Try to extract and normalize date/time from filename
                if "_" in wf_ts_str:
                    wf_date_part, wf_time_part = wf_ts_str.split("_", 1)
                else:
                    continue
                
                # Clean up date and time
                wf_date_clean = wf_date_part.replace("-", "")  # Handle YYYY-MM-DD
                
                # Extract HHMMSS (first 6 chars of time part, removing colons)
                wf_time_clean = wf_time_part[:8].replace(":", "")  # Get first ~6 chars and remove colons
                if len(wf_time_clean) < 6:
                    continue
                
                # Only take HHMMSS (6 chars)
                wf_hhmmss = wf_time_clean[:6]
                
                wf_dt = datetime.strptime(
                    f"{wf_date_clean}_{wf_hhmmss}", "%Y%m%d_%H%M%S"
                )
                
                # Check if within 1 second
                time_diff = abs((expected_dt - wf_dt).total_seconds())
                if time_diff < 1.0:
                    waveform_path = os.path.join(waveform_dir, wf_file)
                    try:
                        st = read(waveform_path)
                        stream_dict[event_id_full] = st
                        n_loaded += 1
                        waveform_found = True
                        break
                    except Exception as exc:
                        log.debug(f"  Failed to read {wf_file}: {exc}")
                        continue
            except Exception:
                continue
        
        if not waveform_found:
            log.debug(f"  No waveform found for {event_id_short}")
            n_missing += 1
    
    log.info(f"  Loaded waveforms: {n_loaded}")
    log.info(f"  Missing waveforms: {n_missing}")
    
    return stream_dict, n_loaded, n_missing


def generate_hypodd_files(
    catalog: Catalog,
    inventory: Inventory,
    stream_dict: Dict[str, Stream],
    output_dir: str,
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
        log.info("Generating dt.cc (cross-correlation differential times)...")
        if len(stream_dict) > 0:
            try:
                event_id_mapper = catalog_to_dd.write_correlations(
                    catalog,
                    stream_dict,
                    extract_len=EXTRACT_LEN,
                    pre_pick=PRE_PICK,
                    shift_len=SHIFT_LEN,
                    event_id_mapper=event_id_mapper,
                    lowcut=LOWCUT,
                    highcut=HIGHCUT,
                    max_sep=MAX_SEP,
                    min_link=MIN_LINK,
                    min_cc=MIN_CC,
                    interpolate=False,
                    all_horiz=False,
                    max_workers=MAX_WORKERS,
                    parallel_process=PARALLEL_PROCESS,
                    weight_by_square=WEIGHT_BY_SQUARE,
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
    catalog: Catalog,
    filtered_catalog: Catalog,
    stream_dict: Dict[str, Stream],
    output_dir: str,
) -> None:
    """
    Write a text summary report of the processing.
    
    Parameters
    ----------
    summary : Dict[str, any]
        Summary dictionary from generate_hypodd_files().
    catalog : Catalog
        Original catalog.
    filtered_catalog : Catalog
        Filtered catalog after quality checks.
    stream_dict : Dict[str, Stream]
        Stream dictionary.
    output_dir : str
        Output directory.
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
        f.write(f"Party:        {PARTY_PATH}\n")
        f.write(f"Waveforms:    {WAVEFORM_DIR}\n")
        f.write(f"Inventory:    {INVENTORY_PATH}\n\n")
        
        # Catalog statistics
        f.write("CATALOG STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Detections before filtering:  {len(catalog)}\n")
        f.write(f"Detections after filtering:   {len(filtered_catalog)}\n")
        f.write(f"Detections removed:           {len(catalog) - len(filtered_catalog)}\n\n")
        
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
        f.write(f"Min correlation coefficient:       {MIN_CC}\n")
        f.write(f"Default event depth (km):         {DEFAULT_DEPTH_KM}\n")
        f.write(f"Extract length (s):                {EXTRACT_LEN}\n")
        f.write(f"Pre-pick time (s):                 {PRE_PICK}\n")
        f.write(f"Max pick shift (s):                {SHIFT_LEN}\n")
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
            default=PARTY_PATH,
            help=f"Path to Party .tgz file (default: {PARTY_PATH})",
        )
        parser.add_argument(
            "--waveforms",
            default=WAVEFORM_DIR,
            help=f"Path to waveform directory (default: {WAVEFORM_DIR})",
        )
        parser.add_argument(
            "--inventory",
            default=INVENTORY_PATH,
            help=f"Path to station inventory XML (default: {INVENTORY_PATH})",
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
            "--skip-correlation",
            action="store_true",
            help="Skip dt.cc generation (only generate dt.ct)",
        )
        args = parser.parse_args()
    
    # Update global variables if custom arguments provided
    if args.max_sep != MAX_SEP:
        globals()["MAX_SEP"] = args.max_sep
    if args.min_link != MIN_LINK:
        globals()["MIN_LINK"] = args.min_link
    
    log.info("=" * 80)
    log.info("HypoDD Input Files Generation")
    log.info("=" * 80)
    
    try:
        # Load Party and convert to Catalog
        catalog = load_party_as_catalog(args.party)
        
        # Load and validate inventory
        inventory = load_and_validate_inventory(args.inventory)
        
        # Apply quality filters (returns both filtered and raw catalogs)
        filtered_catalog, raw_catalog = apply_quality_filters(catalog)
        
        # Build stream dictionary (use filtered catalog)
        log.info(f"DEBUG: args.skip_correlation = {args.skip_correlation}")
        if not args.skip_correlation:
            log.info("DEBUG: Calling build_stream_dict...")
            stream_dict, n_loaded, n_missing = build_stream_dict(
                filtered_catalog, args.waveforms
            )
            log.info(f"DEBUG: build_stream_dict returned {n_loaded} loaded, {n_missing} missing")
        else:
            stream_dict = {}
            log.info("Skipping waveform loading (--skip-correlation)")
        
        # Generate HypoDD files
        summary = generate_hypodd_files(
            filtered_catalog,
            inventory,
            stream_dict,
            args.output,
        )
        
        # Write summary report
        write_summary_report(
            summary,
            catalog,
            filtered_catalog,
            stream_dict,
            args.output,
        )
        
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
