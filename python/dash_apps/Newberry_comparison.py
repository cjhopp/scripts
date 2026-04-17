#!/usr/bin/python
"""
Newberry Comparison Dashboard
Compares three injection periods side-by-side:
  - 2014:  15 Sep 2014 – 30 Nov 2014  (well 55-29)
  - Jan 2025: 01 Jan 2025 – 10 Feb 2025 (well 55A-29)
  - Aug 2025: 01 Aug 2025 – 31 Aug 2025 (well 55A-29)

Seismicity x-axis: hours since injection start (normalised).
Injection plots: flow rate, pressure, cumulative volume vs. hours since start.
Spatial plots (map, N-S, E-W cross-section): one column per period.
"""

import pyproj
import sys

import panel as pn
import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models import NumeralTickFormatter

from obspy import UTCDateTime, Catalog, read_events
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from glob import glob
from scipy.interpolate import interp1d
from holoviews.core.data.interface import DataError

np.set_printoptions(threshold=sys.maxsize)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WELLPATH_55_29 = '/media/chopp/HDD1/chet-meq/newberry/boreholes/55-29/GDR_submission/Deviation_corrected_with-depth.csv'
WELLPATH_55A_29 = '/media/chopp/HDD1/chet-meq/newberry/boreholes/55A-29/55A-29_trajectory.csv'
INJECTION_2014_DIR = '/media/chopp/HDD1/chet-meq/newberry/boreholes/injection_data/2014'
INJECTION_2025_55_29_GLOB  = '/media/chopp/HDD1/chet-meq/newberry/boreholes/injection_data/2025/55-29/**/*.csv'
INJECTION_2025_55A_29_GLOB = '/media/chopp/HDD1/chet-meq/newberry/boreholes/injection_data/2025/55A-29/*.csv'
SEISMICITY_2025_CSV = '/media/chopp/HDD1/chet-meq/newberry/catalogs/Mazama Newberry Microseismic Catalog with Distance From Perfs Halliburton.csv'

# ---------------------------------------------------------------------------
# Injection periods definition
# ---------------------------------------------------------------------------
PERIODS = {
    '2014': {
        'start': UTCDateTime(2014, 9, 15),
        'end':   UTCDateTime(2014, 11, 30),
        'label': '2014 (55-29)',
        'color': '#1f77b4',
        'spatial_min_marker_size': 0.0,
        'injection_source': '2014_dir',
        'injection_path': INJECTION_2014_DIR,
        'wellpaths': [WELLPATH_55_29],
        'wellpath_years': [2014],
    },
    'Jan 2025': {
        'start': UTCDateTime(2025, 1, 1),
        'end':   UTCDateTime(2025, 2, 10),
        'label': 'Jan 2025 (55-29)',
        'color': '#ff7f0e',
        'spatial_min_marker_size': 4.0,
        'injection_source': '2025_csv',
        'injection_path': INJECTION_2025_55_29_GLOB,
        'wellpaths': [WELLPATH_55_29],
        'wellpath_years': [2025],
    },
    'Aug 2025': {
        'start': UTCDateTime(2025, 8, 1),
        'end':   UTCDateTime(2025, 8, 31),
        'label': 'Aug 2025 (55A-29)',
        'color': '#2ca02c',
        'spatial_min_marker_size': 4.0,
        'injection_source': '2025_csv',
        'injection_path': INJECTION_2025_55A_29_GLOB,
        'wellpaths': [WELLPATH_55_29, WELLPATH_55A_29],
        'wellpath_years': [2025, 2025],
    },
}

# Optional catalog overrides by period key.
# Value can be:
#   - None: use FDSN query
#   - ObsPy Catalog object
#   - path to a catalog file readable by obspy.read_events
CUSTOM_CATALOGS = {
    '2014': '/media/chopp/HDD1/chet-meq/newberry/seiscomp_output/testing_2014_chopp-picks_topo-model/scrtdd-relocations_1d-topo_QML.xml',
    'Jan 2025': None,
    'Aug 2025': None,
}

# Optional seismic DataFrame sources by period key.
# Value should be a CSV file path containing event locations/times.
CUSTOM_SEISMICITY_SOURCES = {
    '2014': None,
    'Jan 2025': SEISMICITY_2025_CSV,
    'Aug 2025': SEISMICITY_2025_CSV,
}

# 2025 stage depths by well, converted to meters.
# If you have well-specific stage depths, update these arrays.
# STAGES_55_29 = np.array([9005., 9070., 9170., 9270., 9370., 9470., 9570., 9670., 9770., 9870.]) * 0.3048
# STAGES_55A_29 = np.array([9005., 9070., 9170., 9270., 9370., 9470., 9570., 9670., 9770., 9870.]) * 0.3048

STAGES_55_29 = np.array([9320., 9427., 9452., 9522.5, 9606., 9636., 9769.5, 9851.875]) * 0.3048
STAGES_55_29_Jul25 = np.array([9415., 9486.]) * 0.3048
STAGES_55A_29 = np.array([9575., 9625., 9660., 9750., 9850., 9980., 10064.]) * 0.3048

STAGE_STYLE_BY_WELL = {
    WELLPATH_55_29: {
        'stage_sets': [
            {
                'label': '55-29 stages (2025)',
                'color': '#e67e22',
                'marker': 'circle',
                'stages': STAGES_55_29,
            },
            {
                'label': '55-29 stages (Jul 2025)',
                'color': '#87CEFA',
                'marker': 'circle',
                'stages': STAGES_55_29_Jul25,
            },
        ],
        'east_offset': 0.0,
        'north_offset': 0.0,
    },
    WELLPATH_55A_29: {
        'stage_sets': [
            {
                'label': '55A-29 stages (2025)',
                'color': '#16a085',
                'marker': 'circle',
                'stages': STAGES_55A_29,
            },
        ],
        'east_offset': 0.0,
        'north_offset': 0.0,
    },
}

# Wellhead UTM (used as spatial origin for relative coordinates)
WH_LOC = np.array([635642.0, 4842835.0])

# Injection point relative to WH (for trajectory calcs)
INJECTION_PT = np.array([511, -10, -1184.])

# Colour palette for injection lines
FLOW_COLOR = 'steelblue'
PRESSURE_COLOR = 'firebrick'
CUMVOL_COLOR = 'seagreen'

hv.extension('bokeh')

# ---------------------------------------------------------------------------
# Helpers shared with original script
# ---------------------------------------------------------------------------

def ecdf_transform(data):
    return len(data) - data.rank(method="first")


def cumulative_bbl_from_bps(index: pd.DatetimeIndex, bps: pd.Series) -> pd.Series:
    """Integrate rate (bbl/s) over time using actual sample intervals.

    Uses trapezoidal integration so irregular sampling is handled correctly.
    """
    if len(bps) == 0:
        return pd.Series(dtype=float, index=index)
    t_sec = index.asi8.astype(np.float64) / 1e9
    r = pd.to_numeric(bps, errors='coerce').to_numpy(dtype=np.float64)

    # dt[i] is time between sample i-1 and i (seconds).
    dt = np.diff(t_sec)
    area_steps = 0.5 * (r[1:] + r[:-1]) * dt
    cumulative = np.concatenate(([0.0], np.cumsum(np.nan_to_num(area_steps, nan=0.0))))
    return pd.Series(cumulative, index=index)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_catalog(start: UTCDateTime, end: UTCDateTime) -> Catalog:
    cli = Client('http://131.243.224.19:8085', timeout=600)
    try:
        cat = cli.get_events(
            starttime=start, endtime=end,
            latitude=43.726, longitude=-121.316, maxradius=0.05,
            includeallmagnitudes=True,
        )
    except FDSNNoDataException:
        cat = Catalog()
    except TimeoutError:
        print(f"WARNING: ObsPy FDSN client timed out for {start} to {end}. Returning empty Catalog.")
        cat = Catalog()
    except Exception as e:
        print(f"WARNING: ObsPy FDSN client error: {e}. Returning empty Catalog.")
        cat = Catalog()
    cat.events.sort(key=lambda x: x.preferred_origin().time)
    return cat


def _filter_catalog_to_window(cat: Catalog, start: UTCDateTime, end: UTCDateTime) -> Catalog:
    """Filter a catalog to [start, end] using preferred origin times."""
    events = []
    for ev in cat:
        try:
            o = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
            if o is None:
                continue
            if start <= o.time <= end:
                events.append(ev)
        except Exception:
            continue
    out = Catalog(events=events)
    out.events.sort(key=lambda x: (x.preferred_origin() or x.origins[0]).time)
    return out


def load_period_catalog(period_key: str, start: UTCDateTime, end: UTCDateTime) -> Catalog:
    """Load a period catalog from override if supplied, else query FDSN."""
    override = CUSTOM_CATALOGS.get(period_key)
    if override is None:
        return get_catalog(start, end)

    try:
        if isinstance(override, Catalog):
            cat = override
        elif isinstance(override, str):
            cat = read_events(override)
        else:
            print(f"WARNING: Unsupported catalog override type for {period_key}: {type(override)}. Falling back to FDSN.")
            return get_catalog(start, end)
    except Exception as e:
        print(f"WARNING: Failed to load override catalog for {period_key}: {e}. Falling back to FDSN.")
        return get_catalog(start, end)

    return _filter_catalog_to_window(cat, start, end)


def load_seismicity_from_csv(csv_path: str, start: UTCDateTime, end: UTCDateTime,
                             t_origin) -> pd.DataFrame:
    """Load seismicity locations from CSV and return dashboard-ready DataFrame.

        Expected columns used when present:
      - time: 'date_time_UTC' (fallback: Origin Date/Time + Time Zone Offset)
            - coords (preferred): 'Longitude', 'Latitude' projected to UTM
            - coords (fallback): local/absolute Easting/Northing CSV columns
      - depth/elevation: 'Depth ft (TVDSS)' (fallback: 'Depth ft (TVD)')
      - magnitude: 'Magnitude'
      - lat/lon: 'Latitude', 'Longitude'
    """
    df = pd.read_csv(csv_path)

    # Parse event times, preferring explicit UTC column.
    if 'date_time_UTC' in df.columns:
        times = pd.to_datetime(df['date_time_UTC'], errors='coerce', utc=True)
    elif all(col in df.columns for col in ['Origin Date (Local)', 'Origin Time (Local)', 'Time Zone Offset']):
        dt_local = pd.to_datetime(
            df['Origin Date (Local)'].astype(str) + ' ' +
            df['Origin Time (Local)'].astype(str) + ' ' +
            df['Time Zone Offset'].astype(str),
            errors='coerce', utc=True,
        )
        times = dt_local
    else:
        raise KeyError('No usable time columns found in seismic CSV')

    # Filter to requested window.
    t0 = pd.Timestamp(start.datetime, tz='UTC')
    t1 = pd.Timestamp(end.datetime, tz='UTC')
    in_window = (times >= t0) & (times <= t1)
    df = df.loc[in_window].copy()
    times = times.loc[in_window]

    if df.empty:
        return pd.DataFrame(columns=[
            'latitude', 'longitude', 'depth', 'magnitude', 'marker size',
            'timestamp', 'east', 'north', 'elevation', 'hours', 'cumulative number',
        ])

    # Coordinates: prefer lon/lat projection for consistency with FDSN-derived plotting.
    latitude = pd.to_numeric(df.get('Latitude', np.nan), errors='coerce')
    longitude = pd.to_numeric(df.get('Longitude', np.nan), errors='coerce')
    if latitude.notna().any() and longitude.notna().any():
        utm = pyproj.Proj("EPSG:32610")
        east_abs, north_abs = utm(longitude.to_numpy(), latitude.to_numpy())
        east = pd.Series(east_abs, index=df.index) - WH_LOC[0]
        north = pd.Series(north_abs, index=df.index) - WH_LOC[1]
    elif all(c in df.columns for c in ['Easting ft (Relative Local Tangent Plane True North)',
                                       'Northing ft (Relative Local Tangent Plane True North)']):
        # Relative columns are in feet; convert to meters for local plot axes.
        east = pd.to_numeric(
            df['Easting ft (Relative Local Tangent Plane True North)'], errors='coerce'
        ) * 0.3048
        north = pd.to_numeric(
            df['Northing ft (Relative Local Tangent Plane True North)'], errors='coerce'
        ) * 0.3048
    elif all(c in df.columns for c in ['Easting ft (Abs)', 'Northing ft (Abs)']):
        # Legacy fallback used previously.
        east = pd.to_numeric(df['Easting ft (Abs)'], errors='coerce') - WH_LOC[0]
        north = pd.to_numeric(df['Northing ft (Abs)'], errors='coerce') - WH_LOC[1]
    else:
        raise KeyError('Missing usable coordinate columns in seismic CSV')

    if 'Depth ft (TVDSS)' in df.columns:
        elevation = pd.to_numeric(df['Depth ft (TVDSS)'], errors='coerce') * 0.3048
    elif 'Depth ft (TVD)' in df.columns:
        elevation = -pd.to_numeric(df['Depth ft (TVD)'], errors='coerce') * 0.3048
    else:
        raise KeyError('Missing depth/elevation columns in seismic CSV')

    magnitude = pd.to_numeric(df.get('Magnitude', np.nan), errors='coerce')
    timestamp = times.astype('int64') / 1e9

    out = pd.DataFrame({
        'latitude': latitude,
        'longitude': longitude,
        'depth': -elevation,
        'magnitude': magnitude,
        'marker size': (magnitude + 1.5) ** 2,
        'timestamp': timestamp,
        'east': east,
        'north': north,
        'elevation': elevation,
    }).dropna(subset=['east', 'north', 'elevation', 'magnitude', 'timestamp'])

    if isinstance(t_origin, pd.Timestamp):
        t_origin_ts = t_origin.timestamp()
    else:
        t_origin_ts = float(t_origin)
    out['hours'] = (out['timestamp'] - t_origin_ts) / 3600.0
    out['cumulative number'] = out['magnitude'].transform(ecdf_transform)
    return out


def get_injection_2025(glob_pattern: str) -> pd.DataFrame:
    """Load 2025 injection CSV files; return DataFrame with DatetimeIndex."""
    inj_files = glob(glob_pattern, recursive=True)
    if not inj_files:
        print('WARNING: no 2025 injection files matched {}'.format(glob_pattern))
        empty = pd.DataFrame(columns=['psi', 'bpm', 'bps', 'cumulative [bbl]'])
        empty.index = pd.DatetimeIndex([])
        return empty
    all_df = []
    import csv
    for inj_file in inj_files:
        # Read the first non-skipped line to determine column count
        with open(inj_file, 'r') as f:
            for _ in range(15):
                next(f, None)
            try:
                first_row = next(f)
            except StopIteration:
                print(f"WARNING: File {inj_file} is empty after skipping 15 rows.")
                continue
            col_count = len(list(csv.reader([first_row]))[0])

        # Decide usecols based on file type and column count
        if 'SurgiFrac' in inj_file:
            needed_cols = [0, 1, 3]
        elif 'Test' in inj_file:
            needed_cols = [0, 1, 4]
        else:
            needed_cols = [0, 1, 2]

        if max(needed_cols) >= col_count:
            print(f"WARNING: File {inj_file} has only {col_count} columns, but usecols={needed_cols} requested. Skipping file.")
            continue

        try:
            df = pd.read_csv(
                inj_file, skiprows=15, names=['time', 'psi', 'bpm'],
                index_col=0, usecols=needed_cols
            )
        except Exception as e:
            print(f"WARNING: Failed to read {inj_file} with usecols={needed_cols}: {e}")
            continue
        # Coerce bpm to numeric, warn if any non-numeric
        df['bpm'] = pd.to_numeric(df['bpm'], errors='coerce')
        if df['bpm'].isna().any():
            print(f"WARNING: Non-numeric bpm values found in {inj_file}, coerced to NaN.")
        df['bps'] = df['bpm'] / 60
        all_df.append(df)
    full = pd.concat(all_df, axis=0)
    # Robustly convert index to datetime, drop rows where it fails
    idx_dt = pd.to_datetime(full.index, errors='coerce')
    bad_rows = idx_dt.isna().sum()
    if bad_rows > 0:
        print(f"WARNING: {bad_rows} rows with invalid datetime index dropped from injection data.")
    full = full.loc[~idx_dt.isna()].copy()
    full.index = pd.to_datetime(full.index)
    full = full.sort_index()
    # Force all numeric columns to numeric, coerce errors
    for col in ['bpm', 'psi', 'bps', 'cumulative [bbl]']:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors='coerce')
    full['cumulative [bbl]'] = cumulative_bbl_from_bps(full.index, full['bps'])
    return full


def get_injection_2014(directory: str) -> pd.DataFrame:
    """Load 2014 Excel injection files (same schema as get_old_injection)."""
    frames = []
    for excel in glob('{}/*.xlsx'.format(directory)):
        frames.append(pd.read_excel(excel))
    if not frames:
        empty = pd.DataFrame(columns=['psi', 'bpm', 'bps', 'cumulative [bbl]'])
        empty.index = pd.DatetimeIndex([])
        return empty
    df = pd.concat(frames)
    df = df.set_index('Date + Time')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Rename to common schema used below
    df = df.rename(columns={
        'WHP Corrected (psi)': 'psi',
        'Corrected UltraSonic (gpm)': 'bpm',
    })
    if 'bpm' not in df.columns:
        raise KeyError('Cannot find flow-rate column in 2014 injection data')
    df['bpm'] = pd.to_numeric(df['bpm'], errors='coerce')
    # Convert GPM to BPM (1 BPM = 42 GPM)
    df['bpm'] = df['bpm'] / 42.0
    df['bps'] = df['bpm'] / 60
    df['cumulative [bbl]'] = cumulative_bbl_from_bps(df.index, df['bps'])
    return df


def slice_injection(df: pd.DataFrame, start: UTCDateTime, end: UTCDateTime) -> pd.DataFrame:
    """Return rows within [start, end] (no hours column yet)."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df.iloc[0:0].copy()
    t0 = pd.Timestamp(start.datetime).tz_localize(df.index.tz)
    t1 = pd.Timestamp(end.datetime).tz_localize(df.index.tz)
    return df.loc[(df.index >= t0) & (df.index <= t1)].copy()


def first_injection_time(df: pd.DataFrame):
    """Return the pd.Timestamp of the first row, or None if empty."""
    if df.empty:
        return None
    return df.index[0]


def add_hours(df: pd.DataFrame, t_origin) -> pd.DataFrame:
    """Add/overwrite 'hours' column relative to t_origin (pd.Timestamp)."""
    df = df.copy()
    df['hours'] = (df.index - t_origin).total_seconds() / 3600.0
    return df


def catalog_to_dataset(catalog: Catalog, t_origin) -> pd.DataFrame:
    """Convert ObsPy catalog to DataFrame; add spatial columns and hours-since-start.
    t_origin may be a pd.Timestamp or a POSIX float."""
    utm = pyproj.Proj("EPSG:32610")
    well_origin = np.loadtxt(WELLPATH_55_29, delimiter=',', skiprows=1)
    wh_east = well_origin[0][0]
    wh_north = well_origin[0][1]

    if len(catalog) < 1:
        return pd.DataFrame(columns=[
            'latitude', 'longitude', 'depth', 'magnitude', 'marker size',
            'timestamp', 'east', 'north', 'elevation', 'hours', 'cumulative number',
        ])

    rows = [
        (ev.preferred_origin().longitude,
         ev.preferred_origin().latitude,
         ev.preferred_origin().depth,
         ev.preferred_magnitude().mag,
         float(ev.preferred_origin().time.timestamp))
        for ev in catalog
        if ev.preferred_origin() is not None and ev.preferred_magnitude() is not None
    ]
    if not rows:
        return pd.DataFrame(columns=[
            'latitude', 'longitude', 'depth', 'magnitude', 'marker size',
            'timestamp', 'east', 'north', 'elevation', 'hours', 'cumulative number',
        ])

    params = np.array(rows)
    east, north = utm(params[:, 0], params[:, 1])
    east -= wh_east
    north -= wh_north
    elevation = params[:, 2] * -1.

    df = pd.DataFrame({
        'latitude':   params[:, 1],
        'longitude':  params[:, 0],
        'depth':      params[:, 2],
        'magnitude':  params[:, 3],
        'marker size': (params[:, 3] + 1.5) ** 2,
        'timestamp':  params[:, 4],
        'east':       east,
        'north':      north,
        'elevation':  elevation,
    })
    if isinstance(t_origin, pd.Timestamp):
        t_origin_ts = t_origin.timestamp()
    else:
        t_origin_ts = float(t_origin)
    df['hours'] = (df['timestamp'] - t_origin_ts) / 3600.0
    df['cumulative number'] = df['magnitude'].transform(ecdf_transform)
    return df


# ---------------------------------------------------------------------------
# Well-path geometry
# ---------------------------------------------------------------------------

def build_wellpath_overlays(wp_file: str, year: int):
    """Return (map_paths, NS_paths, EW_paths) lists of hv elements."""
    liner_color = {'cased': 'black', 'slotted': 'red'}
    liner_style = {'cased': 'solid', 'slotted': 'dotted'}
    if year < 2023:
        slotted = {'cased': [(0, 1912), (2289, 2493)],
                   'slotted': [(1912, 2289), (2493, 3045)]}
    else:
        slotted = {'cased': [(0, 3200)], 'slotted': [(0, 0)]}

    raw = np.loadtxt(wp_file, delimiter=',', skiprows=1)
    raw[:, 0] -= WH_LOC[0]
    raw[:, 1] -= WH_LOC[1]
    # Interpolate against MD_m (column 4) since stages are provided in MD.
    fx = interp1d(raw[:, 4], raw[:, 0])
    fy = interp1d(raw[:, 4], raw[:, 1])
    fe = interp1d(raw[:, 4], raw[:, 2])
    new_z = np.linspace(raw[-1, 4], raw[0, 4], 3009)[::-1]
    well_ds = pd.DataFrame({'east': fx(new_z), 'north': fy(new_z),
                             'elevation': fe(new_z), 'depth': new_z})

    map_paths, NS_paths, EW_paths = [], [], []

    # Lithology spans (only add once — caller controls)
    lith_colors = {
        'Welded Tuff': 'darkkhaki', 'Tuff': 'khaki',
        'Basalt': 'darkgray',       'Granodiorite': 'bisque',
    }
    lith_depths = {
        'Welded Tuff': [[1966, 2057]], 'Tuff': [[2057, 2439]],
        'Basalt': [[2439, 2634], [2908, 3067]], 'Granodiorite': [[2634, 2908]],
    }
    elev_wh = 1770.
    for unit, depth_intervals in lith_depths.items():
        for i, d in enumerate(depth_intervals):
            line = 'dotted' if i == 1 else 'solid'
            top = elev_wh - d[0]
            bottom = elev_wh - d[1]
            middle = np.mean([top, bottom])
            span = hv.HSpan(bottom, top).opts(color=lith_colors[unit], alpha=0.3, line_dash=line)
            label = hv.Text(-1500, middle, unit, fontsize=8).opts(color='black')
            NS_paths.extend([span, label])
            EW_paths.extend([span, label])
    NS_paths.append(hv.Text(-1500, elev_wh - 3067, '??', fontsize=8).opts(color='black'))
    EW_paths.append(hv.Text(-1500, elev_wh - 3067, '??', fontsize=8).opts(color='black'))

    for liner, intervals in slotted.items():
        for s in intervals:
            q = '{} > depth and {} < depth'.format(s[1], s[0])
            seg = well_ds.query(q)
            map_paths.append(
                hv.Curve(seg, 'east', 'north').opts(
                    color=liner_color[liner], line_dash=liner_style[liner], line_width=3.))
            NS_paths.append(
                hv.Curve(seg, 'north', 'elevation').opts(
                    color=liner_color[liner], line_dash=liner_style[liner], line_width=3.))
            EW_paths.append(
                hv.Curve(seg, 'east', 'elevation').opts(
                    color=liner_color[liner], line_dash=liner_style[liner], line_width=3.))

    if year >= 2023 and wp_file in STAGE_STYLE_BY_WELL:
        style = STAGE_STYLE_BY_WELL[wp_file]
        stage_sets = style.get('stage_sets')
        if stage_sets is None:
            # Backward-compatible path if stage_sets are not defined.
            stage_sets = [{
                'label': style.get('label', 'stages'),
                'color': style.get('color', 'black'),
                'marker': style.get('marker', 'circle'),
                'stages': style.get('stages', []),
            }]
        for stage_set in stage_sets:
            marker = stage_set.get('marker', 'circle')
            for stage in stage_set['stages']:
                idx = (well_ds['depth'] - stage).abs().idxmin()
                row = well_ds.loc[idx]
                stage_north = row['north'] + style['north_offset']
                stage_east = row['east'] + style['east_offset']
                map_paths.append(
                    hv.Scatter((stage_east, stage_north)).opts(
                        size=10., color=stage_set['color'], marker=marker, line_color='black'))
                NS_paths.append(
                    hv.Scatter((stage_north, row['elevation'])).opts(
                        size=10., color=stage_set['color'], marker=marker, line_color='black'))
                EW_paths.append(
                    hv.Scatter((stage_east, row['elevation'])).opts(
                        size=10., color=stage_set['color'], marker=marker, line_color='black'))

    return map_paths, NS_paths, EW_paths


def stage_legend_panel() -> pn.pane.Markdown:
    """Single global legend for 2025 stage symbols."""
    items = []
    for style in STAGE_STYLE_BY_WELL.values():
        stage_sets = style.get('stage_sets')
        if stage_sets is None:
            stage_sets = [{
                'label': style.get('label', 'stages'),
                'color': style.get('color', 'black'),
                'marker': style.get('marker', 'circle'),
            }]
        for stage_set in stage_sets:
            marker_text = stage_set.get('marker', 'circle')
            items.append(
                "<div style='margin-bottom:2px;'>"
                f"<span style='display:inline-block; width:12px; height:12px; "
                f"background:{stage_set['color']}; border:1px solid #222; margin-right:6px;'></span>"
                f"{stage_set['label']} (marker: {marker_text})"
                + "</div>"
            )
    html = "<div style='font-size:12px; padding:4px 0;'><b>Stage Legend (all spatial plots):</b>" + "".join(items) + "</div>"
    return pn.pane.Markdown(html)


# ---------------------------------------------------------------------------
# Per-period spatial plot (map + 2 cross-sections)
# ---------------------------------------------------------------------------

XLIM = (-2000, 2000)
YLIM_CROSS = (-2200, 1800)

def _seis_scatter(df: pd.DataFrame, x_col: str, y_col: str, color: str,
                  title: str = '', min_marker_size: float = 0.0) -> hv.Scatter:
    """Scatter plot coloured uniformly by period colour."""
    if df.empty:
        return hv.Scatter(pd.DataFrame({x_col: [], y_col: [], 'magnitude': [], 'marker size': []}),
                          x_col, vdims=[y_col, 'magnitude', 'marker size'])
    plot_df = df.copy()
    plot_df['_plot_marker_size'] = np.maximum(plot_df['marker size'], float(min_marker_size))
    return hv.Scatter(plot_df, x_col, vdims=[y_col, 'magnitude', '_plot_marker_size']).opts(
        color=color, size='_plot_marker_size', marker='circle', alpha=0.35,
        fill_alpha=0.2, line_alpha=0.7, line_color=color,
        xlim=XLIM, ylim=XLIM if y_col == 'north' else YLIM_CROSS,
        bgcolor='whitesmoke', responsive=True, backend='bokeh',
        title=title,
    )


def spatial_column(period_key: str, seis_df: pd.DataFrame, period_cfg: dict):
    """Build map + NS + EW overlays for one period column."""
    color = period_cfg['color']
    label = period_cfg['label']
    min_marker_size = period_cfg.get('spatial_min_marker_size', 0.0)

    scatters = {
        'map': _seis_scatter(seis_df, 'east', 'north', color, title=label,
                             min_marker_size=min_marker_size),
        'NS':  _seis_scatter(seis_df, 'north', 'elevation', color,
                             min_marker_size=min_marker_size),
        'EW':  _seis_scatter(seis_df, 'east', 'elevation', color,
                             min_marker_size=min_marker_size),
    }

    all_map_paths, all_NS_paths, all_EW_paths = [], [], []

    for wp_file, wp_year in zip(period_cfg['wellpaths'], period_cfg['wellpath_years']):
        map_paths, NS_paths, EW_paths = build_wellpath_overlays(wp_file, wp_year)
        all_map_paths.extend(map_paths)
        all_NS_paths.extend(NS_paths)
        all_EW_paths.extend(EW_paths)

    map_overlay = hv.Overlay(all_map_paths + [scatters['map']]).opts(
        xlim=XLIM, ylim=XLIM, backend='bokeh')
    NS_overlay = hv.Overlay(all_NS_paths + [scatters['NS']]).opts(
        xlim=XLIM, ylim=YLIM_CROSS, backend='bokeh')
    EW_overlay = hv.Overlay(all_EW_paths + [scatters['EW']]).opts(
        xlim=XLIM, ylim=YLIM_CROSS, backend='bokeh')

    return map_overlay, NS_overlay, EW_overlay


# ---------------------------------------------------------------------------
# Injection plots (normalised time axis = hours since period start)
# ---------------------------------------------------------------------------

def injection_comparison_plots(inj_slices: dict) -> hv.Layout:
    """
    Three stacked rows (flow, pressure, cumulative volume), one curve per period,
    all on a shared "hours since injection start" x-axis.
    """
    flow_curves, pres_curves, cumvol_curves = [], [], []

    for key, (df, cfg) in inj_slices.items():
        if df.empty:
            continue
        color = cfg['color']
        label = cfg['label']

        # Determine correct column names
        flow_col = 'bpm' if 'bpm' in df.columns else 'Corrected UltraSonic (gpm)'
        pres_col = 'psi' if 'psi' in df.columns else 'WHP Corrected (psi)'

        flow_curves.append(
            hv.Curve(df, 'hours', flow_col, label=label).opts(color=color, line_width=1.2))
        pres_curves.append(
            hv.Curve(df, 'hours', pres_col, label=label).opts(color=color, line_width=1.2))
        cumvol_curves.append(
            hv.Curve(df, 'hours', 'cumulative [bbl]', label=label).opts(color=color, line_width=1.2))

    def _overlay(curves, ylabel, title):
        if not curves:
            return hv.Curve([]).opts(ylabel=ylabel, title=title)
        ov = hv.Overlay(curves)
        return ov.opts(
            hv.opts.Curve(backend='bokeh'),
            hv.opts.Overlay(
                xlabel='Hours since injection start', ylabel=ylabel,
                title=title, bgcolor='whitesmoke', responsive=True,
                legend_position='top_left', backend='bokeh',
            ),
        )

    flow_plot   = _overlay(flow_curves,   'Flow rate [bpm]',      'Flow rate comparison')
    pres_plot   = _overlay(pres_curves,   'Pressure [psi]',       'Pressure comparison')
    cumvol_plot = _overlay(cumvol_curves, 'Cumulative vol [bbl]', 'Cumulative volume comparison')

    # Avoid scientific notation on cumulative volume axis.
    def _cumvol_axis_formatter(plot, element):
        plot.state.yaxis[0].formatter = NumeralTickFormatter(format='0,0')

    cumvol_plot = cumvol_plot.opts(hooks=[_cumvol_axis_formatter])

    return pn.Column(
        pn.pane.HoloViews(flow_plot,   height=250, sizing_mode='stretch_width'),
        pn.pane.HoloViews(pres_plot,   height=250, sizing_mode='stretch_width'),
        pn.pane.HoloViews(cumvol_plot, height=250, sizing_mode='stretch_width'),
    )


# ---------------------------------------------------------------------------
# Magnitude-time plot (normalised) — one scatter per period
# ---------------------------------------------------------------------------

def mag_time_comparison(seis_datasets: dict) -> pn.pane.HoloViews:
    curves = []
    for key, (df, cfg) in seis_datasets.items():
        if df.empty:
            continue
        stems = hv.Spikes(df, 'hours', 'magnitude').opts(
            color=cfg['color'], line_width=1.0, alpha=0.55,
            backend='bokeh', show_legend=False,
        )
        heads = hv.Scatter(df, 'hours', 'magnitude', label=cfg['label']).opts(
            color=cfg['color'], marker='circle', size=7,
            fill_alpha=0.0, line_alpha=1.0, line_width=1.5,
            backend='bokeh',
        )
        curves.extend([stems, heads])
    if not curves:
        return pn.pane.HoloViews(hv.Scatter([]))
    ov = hv.Overlay(curves).opts(
        hv.opts.Overlay(
            xlabel='Hours since injection start', ylabel='Magnitude',
            bgcolor='whitesmoke', responsive=True, legend_position='top_right',
            backend='bokeh', title='Seismicity magnitude vs. time',
        )
    )
    return pn.pane.HoloViews(ov, height=300, sizing_mode='stretch_width')


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class ComparisonApp(pn.viewable.Viewer):

    def __init__(self, **params):
        super().__init__(**params)
        self._layout = pn.Column(
            '## Loading data…',
            sizing_mode='stretch_width',
        )
        self._build()

    def _build(self):
        # -- Load injection data per-source (cache by path to avoid re-reading) --
        _inj_cache: dict = {}

        def _load_inj(cfg: dict) -> pd.DataFrame:
            path = cfg['injection_path']
            if path not in _inj_cache:
                if cfg['injection_source'] == '2014_dir':
                    _inj_cache[path] = get_injection_2014(path)
                else:
                    _inj_cache[path] = get_injection_2025(path)
            return _inj_cache[path]

        # Map period key → (sliced injection df, period cfg)
        inj_slices = {}
        seis_datasets = {}

        for key, cfg in PERIODS.items():
            t0, t1 = cfg['start'], cfg['end']

            # Injection slice — derive actual origin from first datapoint
            raw_inj = _load_inj(cfg)
            inj_slice_raw = slice_injection(raw_inj, t0, t1)
            t_origin = first_injection_time(inj_slice_raw)
            if t_origin is None:
                # No injection data for this period; fall back to window boundary
                t_origin = pd.Timestamp(t0.datetime)
            inj_slice = add_hours(inj_slice_raw, t_origin)
            inj_slices[key] = (inj_slice, cfg)

            # Seismicity — prefer per-period CSV locations if supplied, fallback to catalog/FDSN.
            seis_source = CUSTOM_SEISMICITY_SOURCES.get(key)
            if seis_source:
                try:
                    seis_df = load_seismicity_from_csv(seis_source, t0, t1, t_origin)
                except Exception as e:
                    print(f"WARNING: Failed to load seismic CSV for {key}: {e}. Falling back to catalog/FDSN.")
                    cat = load_period_catalog(key, t0, t1)
                    seis_df = catalog_to_dataset(cat, t_origin)
            else:
                cat = load_period_catalog(key, t0, t1)
                seis_df = catalog_to_dataset(cat, t_origin)
            seis_datasets[key] = (seis_df, cfg)

        # -- Build injection comparison panel --
        inj_panel = injection_comparison_plots(inj_slices)

        # -- Build spatial columns --
        spatial_cols = {}
        for key, (seis_df, cfg) in seis_datasets.items():
            map_ov, ns_ov, ew_ov = spatial_column(key, seis_df, cfg)
            spatial_cols[key] = (map_ov, ns_ov, ew_ov)

        keys = list(PERIODS.keys())

        def _titled_pane(hv_obj, title='', height=400):
            return pn.Column(
                '**{}**'.format(title),
                pn.pane.HoloViews(hv_obj, height=height, sizing_mode='stretch_width'),
                sizing_mode='stretch_width',
            )

        map_row = pn.Row(
            *[_titled_pane(spatial_cols[k][0], PERIODS[k]['label'], height=400) for k in keys],
            sizing_mode='stretch_width',
        )
        ns_row = pn.Row(
            *[_titled_pane(spatial_cols[k][1], 'N–S cross-section', height=350) for k in keys],
            sizing_mode='stretch_width',
        )
        ew_row = pn.Row(
            *[_titled_pane(spatial_cols[k][2], 'E–W cross-section', height=350) for k in keys],
            sizing_mode='stretch_width',
        )

        mag_time_row = mag_time_comparison(seis_datasets)

        save_button = pn.widgets.Button(name='Save report', button_type='primary')
        save_button.on_click(self._save)

        self._layout[:] = [
            '# Newberry Injection Comparison',
            save_button,
            '## Seismicity locations',
            stage_legend_panel(),
            map_row, ns_row, ew_row,
            '## Magnitude vs. time (hours since injection start)',
            mag_time_row,
            '## Injection parameters',
            inj_panel,
        ]

    def _save(self, event):
        from bokeh.resources import INLINE
        print('Saving to Newberry_comparison.html')
        self._layout.save('Newberry_comparison.html', resources=INLINE)

    def __panel__(self):
        return self._layout


app = pn.template.VanillaTemplate(
    title='Newberry Injection Comparison',
    logo='',
    main=ComparisonApp(),
).servable()
