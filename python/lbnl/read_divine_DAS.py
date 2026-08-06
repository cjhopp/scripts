import pandas as pd
import numpy as np
from obspy import Catalog, UTCDateTime, Inventory
from obspy.core.event import (
    Event, Origin, Magnitude, Pick, WaveformStreamID,
    ResourceIdentifier, Arrival, OriginQuality
)
import re
import warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from pyproj import Transformer
from obspy.core.inventory import (
    Network, Station, Channel, Site,
    InstrumentSensitivity, Response
)

# ── Channel mapping configuration ─────────────────────────────────────────────
CHANNEL_CONFIGS = {
    "Gold_PB": {
        "fiber_start": 1123, "fiber_end": 2240, "step": 3,
        "n_channels": 373, "network": "DAS", "station_prefix": "GOLD",
    },
    "Delano1": {
        "fiber_start": 898, "fiber_end": 1521, "step": 2,
        "n_channels": 312, "network": "DAS", "station_prefix": "DVN1",
    },
    "16B_PT": {
        "fiber_start": 864, "fiber_end": 1215, "step": 2,
        "n_channels": 176, "network": "DAS", "station_prefix": "16BP",
    },
    "Gold_NEW": {
        "fiber_start": 1159, "fiber_end": 2293, "step": 3,
        "n_channels": 379, "network": "DAS", "station_prefix": "GNEW",
    },
    "Delano_NEW": {
        "fiber_start": 1931, "fiber_end": 2551, "step": 2,
        "n_channels": 311, "network": "DAS", "station_prefix": "DNEW",
    },
}

# Reference wellhead coordinates (UTM, metres)
X0, Y0, Z0 = 334641.1891, 4263443.693, 1650.0249
SAMPLING_RATE = 1000.0  # Hz
FT_TO_M = 0.3048

# Config name to 2-character SEED code mapping
CONFIG_TO_CODE = {
    "16B_PT":     "16",
    "Delano1":    "DL",
    "Gold_PB":    "GP",
    "Gold_NEW":   "GN",
    "Delano_NEW": "DN",
}

# ── Feb/Mar fiber group ranges ─────────────────────────────────────────────────
# Channels listed in numerical order per the Divine export description:
#   Gold PB (1-373), Delano-1 (374-685), 16B PT (686-861)
FEBMAR_FIBER_RANGES = [
    ("Gold_PB",  1,   373),
    ("Delano1",  374, 685),
    ("16B_PT",   686, 861),
]

# ── Jul/Aug filename prefix → fiber config mapping ─────────────────────────────
# ** Verify prefix→fiber assignments against your data **
JULAUG_PREFIX_CONFIGS = {
    "2ib":       [("Gold_NEW",   1, 379)],
    "6ib":       [("Delano_NEW", 1, 311)],
    "7pa":       [("Gold_NEW",   1, 379), ("Delano_NEW", 380, 690)],
    "post_stim": [("Gold_NEW",   1, 379), ("Delano_NEW", 380, 690)],
    "p_test":    [("Gold_NEW",   1, 379), ("Delano_NEW", 380, 690)],
}


def build_channel_index(config_name):
    """Return dict {group_number (1-based): absolute fiber channel index}."""
    cfg = CHANNEL_CONFIGS[config_name]
    channels = list(range(cfg["fiber_start"], cfg["fiber_end"] + 1, cfg["step"]))
    return {grp + 1: ch for grp, ch in enumerate(channels)}


def load_rec_file(rec_path):
    """
    Load a Divine .rec trajectory file (fixed-width, not CSV).

    Format (after header block):
        col 0: receiver number
        col 1: X (ft from wellhead)
        col 2: Y (ft from wellhead)
        col 3: Z (ft, depth positive downward from wellhead)
        cols 4-7: ignored (angles etc.)

    The header block ends when we hit a line that starts with a numeric
    receiver number. We detect this by trying to parse the first token
    as an integer.

    Parameters
    ----------
    rec_path : str or Path

    Returns
    -------
    pd.DataFrame with columns:
        receiver_num, X_ft, Y_ft, Z_ft,
        Easting(m), Northing(m), Depth(m)
    indexed 0-based by row order.
    """
    rows = []
    with open(rec_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            # Data lines start with an integer receiver number
            try:
                rec_num = int(tokens[0])
            except ValueError:
                continue  # header / comment line, skip
            try:
                x_ft = float(tokens[1])
                y_ft = float(tokens[2])
                z_ft = float(tokens[3])
            except (IndexError, ValueError) as exc:
                warnings.warn(f"Could not parse rec line '{line}': {exc}")
                continue
            rows.append({
                "receiver_num": rec_num,
                "X_ft": x_ft,
                "Y_ft": y_ft,
                "Z_ft": z_ft,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        warnings.warn(f"No data rows parsed from {rec_path}")
        return df

    df["Easting(m)"]  = X0 + df["X_ft"] * FT_TO_M
    df["Northing(m)"] = Y0 + df["Y_ft"] * FT_TO_M
    df["Depth(m)"]    = Z0 - df["Z_ft"] * FT_TO_M

    return df


def _parse_group_columns(raw_header):
    """
    Walk the raw header list and return a dict:
        {group_number: {"Tp": col_idx, "Ts": col_idx, "dp": col_idx}}
    Only fields actually present are included.
    """
    group_pattern = re.compile(r"Group\s+(\d+)", re.IGNORECASE)
    group_fields = {}
    current_group = None

    for idx, raw_col in enumerate(raw_header):
        col = raw_col.strip()
        m = group_pattern.match(col)
        if m:
            current_group = int(m.group(1))
            group_fields.setdefault(current_group, {})
        elif col in ("Tp", "Ts", "dp") and current_group is not None:
            group_fields[current_group].setdefault(col, idx)

    return group_fields


def _is_snr_file(filepath):
    """Return True if this is an _snr file (to be skipped)."""
    return "_snr" in Path(filepath).stem.lower()


def _get_phase_from_filename(filepath):
    """
    Extract phase hint from filename suffix (_P or _S).
    Returns None if not determinable (file contains both P and S picks).
    """
    stem = Path(filepath).stem.upper()
    if stem.endswith("_P"):
        return "P"
    elif stem.endswith("_S"):
        return "S"
    return None


def _infer_julaug_config(filename_stem):
    """
    Return the fiber range list for a Jul/Aug file based on its prefix.
    """
    stem = filename_stem.lower().replace("-", "_")
    for prefix, ranges in JULAUG_PREFIX_CONFIGS.items():
        if stem.startswith(prefix):
            return ranges
    warnings.warn(f"Could not infer fiber config for '{filename_stem}', "
                  f"defaulting to Gold_NEW only.")
    return [("Gold_NEW", 1, 379)]


def _collect_seen_stations(seen, picks):
    """
    Update *seen* dict in-place from a list of Pick objects.

    *seen* is keyed by (network, station, location, channel); values hold
    coordinates and fiber metadata extracted from pick.extra.  Already-seen
    keys are skipped so the first pick per channel wins.
    """
    for pick in picks:
        wf  = pick.waveform_id
        key = (wf.network_code, wf.station_code,
               wf.location_code, wf.channel_code)
        if key in seen:
            continue
        extra    = getattr(pick, 'extra', {}) or {}
        easting  = extra.get("sta_easting_m",  {}).get("value")
        northing = extra.get("sta_northing_m", {}).get("value")
        depth_m  = extra.get("sta_depth_m",    {}).get("value")
        seen[key] = {
            "cfg_name": extra.get("config",        {}).get("value"),
            "fiber_ch": extra.get("fiber_channel", {}).get("value"),
            "grp_num":  extra.get("group_number",  {}).get("value"),
            "easting":  easting  if easting  is not None else float("nan"),
            "northing": northing if northing is not None else float("nan"),
            "depth_m":  depth_m  if depth_m  is not None else float("nan"),
        }


def read_single_evt_file(pick_file, fiber_ranges, phase_filter=None,
                         rec_files=None, min_dp=None, tz_name="America/Denver"):
    """
    Read one .evt file into a list of ObsPy Events.

    Parameters
    ----------
    pick_file : str
        Path to the .evt file.
    fiber_ranges : list of (config_name, group_start, group_end)
        Which fiber configs to apply and which group numbers belong to each.
    phase_filter : str or None
        If "P" or "S", only read picks of that phase. None reads both.
    rec_files : dict or None
        Dict mapping config_name -> rec_file_path for station coordinates.
    min_dp : float or None
        Minimum dp value to keep a pick.
    tz_name : str
        Timezone name (IANA format, e.g., "America/Denver", "UTC").
        Divine times are interpreted as local to this timezone before
        conversion to UTC. Default: "America/Denver".

    Returns
    -------
    list of obspy.core.event.Event
    """
    phase_hint_map = {"Tp": "P", "Ts": "S"}
    if phase_filter == "P":
        phase_hint_map = {"Tp": "P"}
    elif phase_filter == "S":
        phase_hint_map = {"Ts": "S"}

    # Load rec DataFrames for each config present
    rec_dfs = {}
    if rec_files:
        for cfg_name, _, _ in fiber_ranges:
            if cfg_name in rec_files and rec_files[cfg_name] is not None:
                try:
                    rec_dfs[cfg_name] = load_rec_file(rec_files[cfg_name])
                except Exception as exc:
                    warnings.warn(f"Could not load rec file for {cfg_name}: {exc}")

    # ── Load and parse file ────────────────────────────────────────────────
    # Pass 1: scan for header line without storing all lines
    header_line_idx = None
    raw_header = None
    with open(pick_file, "r") as fh:
        for i, line in enumerate(fh):
            if line.strip().startswith("ID"):
                header_line_idx = i
                raw_header = [c.strip() for c in line.split(",")]
                break

    if header_line_idx is None:
        warnings.warn(f"No header row found in {pick_file}, skipping.")
        return

    group_fields = _parse_group_columns(raw_header)

    # Pass 2: use pandas to read directly from file, skipping to header row
    df = pd.read_csv(pick_file, skiprows=header_line_idx, skipinitialspace=True,
                     low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    src_file = Path(pick_file).stem

    # Pre-build channel indexes for all configs in fiber_ranges
    ch_indexes = {}
    for cfg_name, _, _ in fiber_ranges:
        ch_indexes[cfg_name] = build_channel_index(cfg_name)

    # ── Pre-compute static per-group metadata ──────────────────────────────
    # Convert rec_dfs to index-dicts so coordinate lookup is O(1) per group.
    # Keys are 1-based row positions — rec file receiver_num values are absolute
    # fiber channel indices and do NOT match the group numbers in .evt files.
    rec_lookup = {}
    for cn, rdf in rec_dfs.items():
        if not rdf.empty:
            rows = rdf[["Easting(m)", "Northing(m)", "Depth(m)"]].to_dict("records")
            rec_lookup[cn] = {i + 1: r for i, r in enumerate(rows)}

    # Cache timezone objects — ZoneInfo construction is not free.
    _tz     = ZoneInfo(tz_name)
    _tz_utc = ZoneInfo("UTC")

    # Resolve cfg_name, fiber_ch, station_code, and coords for every group
    # once here.  Result is identical for all event rows in this file.
    grp_meta = {}  # grp_num -> (cfg_name, fiber_ch, station_code, e, n, d)
    for grp_num in group_fields:
        _cfg = _local = None
        for cn, g_start, g_end in fiber_ranges:
            if g_start <= grp_num <= g_end:
                _cfg   = cn
                _local = grp_num - g_start + 1
                break
        if _cfg is None:
            continue
        fiber_ch = ch_indexes.get(_cfg, {}).get(_local)
        if fiber_ch is None:
            continue
        sta_code  = f"{CONFIG_TO_CODE.get(_cfg, 'XX')}{grp_num:03d}"
        cfg_lookup = rec_lookup.get(_cfg, {})
        # .rec files use sequential receiver numbers, not absolute fiber-channel
        # indices.  The Feb/Mar combined file is numbered by global grp_num
        # (1-861); per-config Jul/Aug files are numbered by local group (1-N).
        # Try global first, fall back to local so both layouts are handled.
        coords = cfg_lookup.get(grp_num) or cfg_lookup.get(_local) or {}
        grp_meta[grp_num] = (
            _cfg, fiber_ch, sta_code,
            coords.get("Easting(m)",  float("nan")),
            coords.get("Northing(m)", float("nan")),
            coords.get("Depth(m)",    float("nan")),
        )

    for _, row in df.iterrows():

        # ── Event metadata ─────────────────────────────────────────────────
        try:
            event_id = str(row["ID"]).strip()
            date_str = str(row["Date"]).strip()
            time_str = str(row["Time"]).strip()
            t_zero   = float(row["T zero"])
            src_x    = float(row["Source X"])
            src_y    = float(row["Source Y"])
            src_z    = float(row["Source Z"])
            mw       = float(row["Mw"])
            rms      = float(row["rms"])
        except (KeyError, ValueError) as exc:
            warnings.warn(f"Skipping row in {src_file}: {exc}")
            continue

        try:
            doy, year = [int(x) for x in date_str.split("/")]
            # Parse as naive datetime in local timezone, then convert to UTC
            dt_naive = datetime.strptime(f"{year}-{doy:03d}T{time_str}", "%Y-%jT%H:%M:%S.%f")
            dt_local = dt_naive.replace(tzinfo=_tz)
            dt_utc   = dt_local.astimezone(_tz_utc)
            otime = UTCDateTime(dt_utc)
        except Exception as exc:
            warnings.warn(f"Bad time in {src_file} event {event_id}: {exc}")
            continue

        src_easting  = X0 + src_x * FT_TO_M
        src_northing = Y0 + src_y * FT_TO_M
        # Elevation above sea level (m); QuakeML depth = positive downward from
        # sea level, so depth_quakeml = -(elevation) when below sea level.
        src_elevation_m = Z0 - src_z * FT_TO_M
        src_lat, src_lon = utm_to_latlon(src_easting, src_northing)

        # T zero is in samples at SAMPLING_RATE; otime is trace start, not origin time
        origin_time = otime + t_zero / SAMPLING_RATE
        origin = Origin(
            time=origin_time,
            latitude=src_lat,
            longitude=src_lon,
            depth=-src_elevation_m,   # metres, positive downward from sea level
            quality=OriginQuality(standard_error=rms),
        )
        origin.extra = {
            "easting_m":          {"value": src_easting,     "namespace": "DAS"},
            "northing_m":         {"value": src_northing,    "namespace": "DAS"},
            "elevation_m":        {"value": src_elevation_m, "namespace": "DAS"},
            "t_zero_samples":     {"value": t_zero,          "namespace": "DAS"},
            "otime":              {"value": str(otime),      "namespace": "DAS"},
        }
        magnitude = Magnitude(mag=mw, magnitude_type="Mw")
        event = Event(
            resource_id=ResourceIdentifier(
                id=f"das_pick/{src_file}/{event_id}"),
            origins=[origin],
            magnitudes=[magnitude],
        )
        event.preferred_origin_id   = origin.resource_id
        event.preferred_magnitude_id = magnitude.resource_id

        # ── Per-channel picks ──────────────────────────────────────────────
        for grp_num, fields in group_fields.items():
            meta = grp_meta.get(grp_num)
            if meta is None:
                continue
            cfg_name, fiber_ch, station_code, sta_e, sta_n, sta_d = meta

            for pick_field, phase_hint in phase_hint_map.items():
                if pick_field not in fields:
                    continue

                try:
                    t_sample = float(row.iloc[fields[pick_field]])
                except (ValueError, TypeError):
                    continue

                if min_dp is not None and "dp" in fields:
                    try:
                        if float(row.iloc[fields["dp"]]) < min_dp:
                            continue
                    except (ValueError, TypeError):
                        pass

                pick_time = otime + t_sample / SAMPLING_RATE

                _extra = {
                    "group_number":  {"value": grp_num,  "namespace": "DAS"},
                    "fiber_channel": {"value": fiber_ch, "namespace": "DAS"},
                    "config":        {"value": cfg_name, "namespace": "DAS"},
                    "t_sample":      {"value": t_sample, "namespace": "DAS"},
                    "source_file":   {"value": src_file, "namespace": "DAS"},
                }
                if not np.isnan(sta_e):
                    _extra.update({
                        "sta_easting_m":  {"value": sta_e, "namespace": "DAS"},
                        "sta_northing_m": {"value": sta_n, "namespace": "DAS"},
                        "sta_depth_m":    {"value": sta_d, "namespace": "DAS"},
                    })
                pick = Pick(
                    resource_id=ResourceIdentifier(
                        id=f"das_pick/{src_file}/{event_id}"
                           f"/grp{grp_num}/{pick_field}"),
                    time=pick_time,
                    waveform_id=WaveformStreamID(
                        network_code="CF",
                        station_code=station_code,
                        location_code="",
                        channel_code="FSF",
                    ),
                    phase_hint=phase_hint,
                )
                pick.extra = _extra
                event.picks.append(pick)

        yield event


def read_all_das_picks(base_dir, output_dir, min_dp=None, tz_name="America/Denver"):
    """
    Read all .evt files from PicksBearskinFebMar2025/ and
    PicksGoldBearskinJulyAugust/, writing each event to a separate SCML file.
    Also builds and writes a StationXML Inventory containing all unique
    DAS groups/channels referenced in the picks.

    Parameters
    ----------
    base_dir : str
        Path to the DAS_picks folder containing both pick subdirectories
        and the .rec files.
    output_dir : str or Path
        Directory where per-event SCML files will be written.
    min_dp : float or None
        Minimum dp threshold; None keeps all picks.
    tz_name : str
        Timezone name (IANA format, e.g., "America/Denver", "UTC").
        Divine times are interpreted as local to this timezone before
        conversion to UTC. Default: "America/Denver".

    Returns
    -------
    dict with keys:
        'event_paths': list of Path objects to written SCML files
        'inventory': obspy.Inventory of all DAS groups with picks
        'inventory_path': Path to written StationXML file
    """
    base_dir   = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    febmar_dir = base_dir / "PicksBearskinFebMar2025"
    julaug_dir = base_dir / "PicksGoldBearskinJulyAugust"

    # ── .rec file paths ────────────────────────────────────────────────────
    rec_febmar      = base_dir / "gold-delano-16b.rec"
    rec_julaug_july = julaug_dir / "July_Gold_Delano.rec"
    rec_julaug_aug  = julaug_dir / "AUG_Delano_Gold.rec"

    rec_files_febmar = {
        "16B_PT":  str(rec_febmar),
        "Delano1": str(rec_febmar),
        "Gold_PB": str(rec_febmar),
    }
    rec_files_julaug = {
        "Gold_NEW":   str(rec_julaug_july),
        "Delano_NEW": str(rec_julaug_aug),
    }

    seen_stations = {}  # (net, sta, loc, cha) -> coord/config info
    written_paths = []
    total_events = 0
    total_picks = 0

    # ── Feb/Mar files ──────────────────────────────────────────────────────
    print("Reading Feb/Mar picks...")
    for evt_file in sorted(febmar_dir.glob("*.evt")):
        phase_filter = _get_phase_from_filename(evt_file)
        file_events = 0
        file_picks = 0
        for event in read_single_evt_file(
            pick_file=str(evt_file),
            fiber_ranges=FEBMAR_FIBER_RANGES,
            phase_filter=phase_filter,
            rec_files=rec_files_febmar,
            min_dp=min_dp,
            tz_name=tz_name,
        ):
            event_id = str(event.resource_id).split("/")[-1]
            src_file = Path(evt_file).stem
            out_path = output_dir / f"{src_file}__{event_id}.xml"
            _collect_seen_stations(seen_stations, event.picks)
            if not out_path.exists():
                cat = Catalog([event])
                cat.write(str(out_path), format="SC3ML")
            written_paths.append(out_path)
            file_events += 1
            file_picks += len(event.picks)
            total_events += 1
            total_picks += len(event.picks)
        print(f"  {evt_file.name}: {file_picks} picks across {file_events} events")

    # ── Jul/Aug files ──────────────────────────────────────────────────────
    print("Reading Jul/Aug picks...")
    for evt_file in sorted(julaug_dir.glob("*.evt")):
        if _is_snr_file(evt_file):
            continue  # skip _snr files per your instructions

        phase_filter = _get_phase_from_filename(evt_file)
        fiber_ranges = _infer_julaug_config(evt_file.stem)
        file_events = 0
        file_picks = 0
        for event in read_single_evt_file(
            pick_file=str(evt_file),
            fiber_ranges=fiber_ranges,
            phase_filter=phase_filter,
            rec_files=rec_files_julaug,
            min_dp=min_dp,
            tz_name=tz_name,
        ):
            event_id = str(event.resource_id).split("/")[-1]
            src_file = Path(evt_file).stem
            out_path = output_dir / f"{src_file}__{event_id}.xml"
            _collect_seen_stations(seen_stations, event.picks)
            if not out_path.exists():
                cat = Catalog([event])
                cat.write(str(out_path), format="SC3ML")
            written_paths.append(out_path)
            file_events += 1
            file_picks += len(event.picks)
            total_events += 1
            total_picks += len(event.picks)
        print(f"  {evt_file.name}: {file_picks} picks across {file_events} events")

    print(f"\nTotal events processed: {total_events}")
    print(f"Total picks processed: {total_picks}")
    print(f"Output directory: {output_dir}")

    # ── Build and write StationXML Inventory ───────────────────────────────
    print("\nBuilding StationXML inventory...")
    inventory = build_das_inventory(seen_stations, rec_files={
        "16B_PT":     str(rec_febmar),
        "Delano1":    str(rec_febmar),
        "Gold_PB":    str(rec_febmar),
        "Gold_NEW":   str(rec_julaug_july),
        "Delano_NEW": str(rec_julaug_aug),
    })
    inv_path = output_dir / "DAS_stations.xml"
    inventory.write(str(inv_path), format="STATIONXML")
    print(f"Wrote StationXML to: {inv_path}")

    return {
        'event_paths': written_paths,
        'inventory': inventory,
        'inventory_path': inv_path,
    }



_transformer = Transformer.from_crs("EPSG:6341", "EPSG:4326", always_xy=True)

def utm_to_latlon(easting, northing):
    """Convert EPSG:6341 easting/northing to WGS84 lat/lon."""
    lon, lat = _transformer.transform(easting, northing)
    return lat, lon


def build_das_inventory(catalog_or_seen, rec_files=None):
    """
    Build an ObsPy Inventory from DAS pick data.

    Each DAS channel becomes a Station + Channel entry. A dummy Response
    is attached at the Channel level representing a generic strain-rate
    sensor at 1000 Hz with unit sensitivity.

    Parameters
    ----------
    catalog_or_seen : obspy.Catalog or dict
        Either an obspy.Catalog whose picks carry the DAS extra fields, or
        a pre-built ``seen`` dict as populated by ``_collect_seen_stations``
        (keyed by ``(network, station, location, channel)``).  Passing the
        pre-built dict avoids re-scanning the catalog and is preferred when
        calling from ``read_all_das_picks``.
    rec_files : dict or None
        Optional dict mapping config_name -> rec_file_path.  Used as a
        fallback when station coordinates are missing or NaN.

    Returns
    -------
    obspy.Inventory
    """
    from obspy.core.inventory.response import (
        InstrumentSensitivity, Response
    )
    from obspy.core.utcdatetime import UTCDateTime as _UTC

    # ── Optionally pre-load rec DataFrames ────────────────────────────────
    rec_dfs = {}
    if rec_files:
        for cfg_name, rec_path in rec_files.items():
            try:
                rec_dfs[cfg_name] = load_rec_file(rec_path)
            except Exception as exc:
                warnings.warn(f"Could not load rec file for {cfg_name}: {exc}")

    # ── Dummy response ─────────────────────────────────────────────────────
    # Represents a linear, flat instrument with unit sensitivity.
    # input_units = "STRAIN/S" (DAS measures strain rate)
    # output_units = "COUNTS"
    dummy_sensitivity = InstrumentSensitivity(
        value=1.0,
        frequency=1.0,
        input_units="STRAIN/S",
        input_units_description="Strain rate along fiber axis",
        output_units="COUNTS",
        output_units_description="Digital counts",
    )
    dummy_response = Response(instrument_sensitivity=dummy_sensitivity)

    # ── Accept pre-built seen dict or collect from catalog ──────────────────
    if isinstance(catalog_or_seen, dict):
        seen = catalog_or_seen
    else:
        seen = {}
        for event in catalog_or_seen:
            _collect_seen_stations(seen, event.picks)

    # Apply rec_dfs fallback for any stations with missing/NaN coordinates
    for info in seen.values():
        cfg_name = info["cfg_name"]
        fiber_ch = info["fiber_ch"]
        grp_num  = info["grp_num"]
        if (np.isnan(info["easting"]) and cfg_name in rec_dfs
                and grp_num is not None):
            rec_df  = rec_dfs[cfg_name]
            # Keys are 1-based row positions (same convention as rec_lookup
            # in read_single_evt_file).  Try global group number first,
            # fall back to local fiber index.
            for key in (grp_num, fiber_ch):
                if key is None or not (1 <= key <= len(rec_df)):
                    continue
                row = rec_df.iloc[key - 1]
                info["easting"]  = row["Easting(m)"]
                info["northing"] = row["Northing(m)"]
                info["depth_m"]  = row["Depth(m)"]
                break

    # ── Build Inventory structure ──────────────────────────────────────────
    # Group stations by network code
    networks_dict = {}  # network_code -> {station_code -> info}

    for (net, sta, loc, cha), info in seen.items():
        networks_dict.setdefault(net, {})
        networks_dict[net].setdefault(sta, {
            "locations": {},
            "easting":  info["easting"],
            "northing": info["northing"],
            "depth_m":  info["depth_m"],
            "cfg_name": info["cfg_name"],
            "fiber_ch": info["fiber_ch"],
        })
        networks_dict[net][sta]["locations"][(loc, cha)] = info

    inv_networks = []
    missing_coords = []   # collect stations with no coordinates for one summary warning

    for net_code, stations in networks_dict.items():
        inv_stations = []

        for sta_code, sta_info in stations.items():
            easting  = sta_info["easting"]
            northing = sta_info["northing"]
            depth_m  = sta_info["depth_m"]

            # Convert UTM to geographic for ObsPy Station
            if not (np.isnan(easting) or np.isnan(northing)):
                lat, lon = utm_to_latlon(easting, northing)
            else:
                lat, lon = 0.0, 0.0
                missing_coords.append(sta_code)

            # depth_m        = absolute sensor elevation above sea level (m)
            #                 = Z0 - Z_ft * FT_TO_M
            # Station.elevation = surface elevation at wellhead = Z0 (our only
            #                     known surface reference for all downhole channels)
            # Channel.depth     = depth below that surface (m, positive downward)
            #                     = Z0 - depth_m
            # Channel.elevation = absolute sensor elevation = Station.elevation
            #                     - Channel.depth = depth_m  ✓
            sensor_elevation_m = depth_m if not np.isnan(depth_m) else 0.0
            depth_below_wh     = (Z0 - depth_m) if not np.isnan(depth_m) else 0.0

            station = Station(
                code=sta_code,
                latitude=lat,
                longitude=lon,
                elevation=Z0,   # surface reference = wellhead elevation
                site=Site(name=f"DAS fiber channel – {sta_info.get('cfg_name', '')} "
                               f"ch {sta_info.get('fiber_ch', '')}"),
                creation_date=_UTC(0),
            )

            # One Channel entry per unique (location, channel_code)
            for (loc_code, cha_code), info in sta_info["locations"].items():
                channel = Channel(
                    code=cha_code,
                    location_code=loc_code,
                    latitude=lat,
                    longitude=lon,
                    elevation=sensor_elevation_m,   # absolute sensor elevation (m)
                    depth=depth_below_wh,           # below wellhead surface (m)
                    azimuth=0.0,       # along-fiber; adjust if known
                    dip=0.0,           # horizontal; adjust for deviated wells
                    sample_rate=SAMPLING_RATE,
                    response=dummy_response,
                )
                station.channels.append(channel)

            inv_stations.append(station)

        inv_networks.append(
            Network(
                code=net_code,
                stations=inv_stations,
                description="CF DAS fiber network",
            )
        )

    if missing_coords:
        warnings.warn(
            f"{len(missing_coords)} station(s) had no .rec coordinates and were "
            f"defaulted to (0, 0): {', '.join(sorted(missing_coords))}"
        )
    inventory = Inventory(networks=inv_networks, source="DAS picks reader")
    return inventory


def write_gtsrce(inventory, output_path, decimate=10):
    """
    Write a NonLinLoc GTSRCE station file from a DAS StationXML inventory.

    Format per line:
        #GTSRCE <station> LATLON <lat> <lon> <elev_km> 0.0

    Parameters
    ----------
    inventory : obspy.Inventory or str or Path
        Inventory object or path to a StationXML file.
    output_path : str or Path
        Destination file for the GTSRCE lines.
    decimate : int, optional
        Keep every Nth channel within each well (identified by the 2-char
        station-code prefix).  Default 10 reduces ~300-channel wells to
        ~30 entries.  Use 1 to write every channel.

    Notes
    -----
    Coordinates are taken from the Channel level (not Station), so each DAS
    gauge gets its own line with its absolute downhole position.
    The 5th column is Channel.elevation / 1000 — elevation above sea level
    in km; negative values indicate sensors below sea level.
    The 6th column is always 0.0 (surface elevation / water depth, unused
    for borehole sensors).
    """
    from collections import defaultdict

    if isinstance(inventory, (str, Path)):
        from obspy import read_inventory
        inventory = read_inventory(str(inventory))

    # Group stations by the 2-char well prefix (e.g. "DL", "GP", "GN")
    wells = defaultdict(list)
    for network in inventory:
        for station in network:
            wells[station.code[:2]].append(station)

    output_path = Path(output_path)
    total = 0
    with open(output_path, "w") as fh:
        for prefix in sorted(wells):
            stations_sorted = sorted(wells[prefix], key=lambda s: s.code)
            for station in stations_sorted[::decimate]:
                for channel in station:
                    elev_km = channel.elevation / -1000.0
                    fh.write(
                        f"GTSRCE {station.code} LATLON "
                        f"{channel.latitude:.6f} {channel.longitude:.6f} "
                        f"{elev_km:.4g} 0.0\n"
                    )
                    total += 1

    print(f"  {total} GTSRCE entries written "
          f"(decimate={decimate}, {len(wells)} well(s): "
          f"{', '.join(sorted(wells))})")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DAS pick reader / station file builder")
    parser.add_argument("base_dir", nargs="?",
                        default="/media/chopp/HDD1/chet-meq/cape_modern/catalogs/fervo/DAS_picks")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <base_dir>/scml_events)")
    parser.add_argument("--stations-only", action="store_true",
                        help="Skip SCML event writing; only rebuild DAS_stations.xml "
                             "and the GTSRCE file.")
    parser.add_argument("--decimate", type=int, default=10,
                        help="GTSRCE channel decimation factor (default: 10)")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "scml_events_v2"

    if args.stations_only:
        # ── Fast path: rebuild station files only ─────────────────────────
        # Re-parse evt files to collect channel coordinates, then write
        # DAS_stations.xml and GTSRCE without touching any SCML files.
        output_dir.mkdir(parents=True, exist_ok=True)

        febmar_dir      = base_dir / "PicksBearskinFebMar2025"
        julaug_dir      = base_dir / "PicksGoldBearskinJulyAugust"
        rec_febmar      = base_dir / "gold-delano-16b.rec"
        rec_julaug_july = julaug_dir / "July_Gold_Delano.rec"
        rec_julaug_aug  = julaug_dir / "AUG_Delano_Gold.rec"

        rec_files_febmar = {"16B_PT": str(rec_febmar), "Delano1": str(rec_febmar),
                            "Gold_PB": str(rec_febmar)}
        rec_files_julaug = {"Gold_NEW": str(rec_julaug_july),
                            "Delano_NEW": str(rec_julaug_aug)}

        seen_stations = {}
        print("Scanning Feb/Mar evt files for channel coordinates...")
        for evt_file in sorted(febmar_dir.glob("*.evt")):
            phase_filter = _get_phase_from_filename(evt_file)
            for event in read_single_evt_file(str(evt_file), FEBMAR_FIBER_RANGES,
                                              phase_filter=phase_filter,
                                              rec_files=rec_files_febmar):
                _collect_seen_stations(seen_stations, event.picks)

        print("Scanning Jul/Aug evt files for channel coordinates...")
        for evt_file in sorted(julaug_dir.glob("*.evt")):
            if _is_snr_file(evt_file):
                continue
            fiber_ranges = _infer_julaug_config(evt_file.stem)
            phase_filter = _get_phase_from_filename(evt_file)
            for event in read_single_evt_file(str(evt_file), fiber_ranges,
                                              phase_filter=phase_filter,
                                              rec_files=rec_files_julaug):
                _collect_seen_stations(seen_stations, event.picks)

        print(f"Collected {len(seen_stations)} unique channels.")
        inventory = build_das_inventory(seen_stations, rec_files={
            "16B_PT": str(rec_febmar), "Delano1": str(rec_febmar),
            "Gold_PB": str(rec_febmar), "Gold_NEW": str(rec_julaug_july),
            "Delano_NEW": str(rec_julaug_aug),
        })
        inv_path = output_dir / "DAS_stations.xml"
        inventory.write(str(inv_path), format="STATIONXML")
        print(f"Wrote StationXML to: {inv_path}")
        gtsrce_path = write_gtsrce(inventory, output_dir / "DAS_stations.gtsrce",
                                   decimate=args.decimate)
        print(f"GTSRCE file: {gtsrce_path}")

    else:
        # ── Full run ──────────────────────────────────────────────────────
        result = read_all_das_picks(base_dir, output_dir=output_dir, min_dp=None)
        print(f"\nWrote {len(result['event_paths'])} SCML files to {output_dir}")
        print(f"StationXML inventory: {result['inventory_path']}")
        gtsrce_path = write_gtsrce(result["inventory"],
                                   output_dir / "DAS_stations.gtsrce",
                                   decimate=args.decimate)
        print(f"GTSRCE file: {gtsrce_path}")