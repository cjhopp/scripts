import dash
import argparse
import json
import logging
import os
import re
from glob import glob
from functools import lru_cache
import numpy as np
import pandas as pd
import xarray as xr
import pyproj

import plotly.graph_objs as go

from dash import dcc, html
from obspy import read_events, UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude, ResourceIdentifier
from osgeo import gdal
from datetime import datetime
from shapely.geometry import Polygon
from shapely import wkt

gdal.UseExceptions()

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Set your local data directory here
data_directory = '/media/chopp/HDD1/chet-meq'
cape_stage_table_path = f'{data_directory}/cape_modern/injection/timing_location/cape_injection_stage_table.csv'

site_polygons = {
    'Newberry': Polygon([(-121.0736, 43.8988), (-121.0736, 43.5949), (-121.4918, 43.5949), (-121.4918, 43.8988)]),
    'JV': Polygon([(-117.40, 40.2357), (-117.5692, 40.2357), (-117.5692, 40.107), (-117.40, 40.107)]),
    'DAC': Polygon([(-118.1979, 38.9604), (-118.1979, 38.7943), (-118.4046, 38.7943), (-118.4046, 9604)]),
    'TM': Polygon([(-117.5956, 39.7353), (-117.5956, 39.6056), (-117.7649, 39.6056), (-117.7649, 39.7353)]),
    'Cape': Polygon([(-112.6924, 38.3912), (-112.6924, 38.6512), (-113.1358, 38.6512), (-113.1358, 38.3912)])
}

datasets = {
    'Newberry': {
        '55-29': f'{data_directory}/newberry/boreholes/55-29/GDR_submission/Deviation_corrected_with-depth_w-TD.csv',
        '55A-29': f'{data_directory}/newberry/boreholes/55A-29/55A-29_trajectory.csv',
        'DEM': f'{data_directory}/newberry/DEMs/USGS_13_merged_epsg-26910_just_edifice_very-coarse.tif',
    },
    'JV': {
        'Offset Wells': f'{data_directory}/JV/vector/boreholes/Offset_Wells_Surveys_JV.csv',
    },
    'DAC': {
        'Offset Wells': f'{data_directory}/DAC/vector/boreholes/Offset_Wells_Surveys_DAC.csv',
    },
    'Cape': {
        'Topography': f'{data_directory}/cape_modern/spatial_data/DEM/Cape-modern_Lidar_downsample.tif',
        'Basement': f'{data_directory}/cape_modern/spatial_data/vmods/ToB_50m_grid_3-1-24.nc',
        'Frisco-1': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Frisco-1_trajectory.csv',
        'Frisco-2': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Frisco-2_trajectory.csv',
        'Frisco-3': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Frisco-3_trajectory.csv',
        'Frisco-4': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Frisco-4_trajectory.csv',
        'Bearskin-1IA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_1IA_xyz.csv',
        'Bearskin-2IB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_2IB_xyz.csv',
        'Bearskin-4PB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_4PB_xyz.csv',
        'Bearskin-6IB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_6IB_xyz.csv',
        'Bearskin-7PA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_7PA_xyz.csv',
        'Bearskin-8IA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_8IA_xyz.csv',
        'Gold-1PB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Gold_1PB_trajectory.csv',
        'Gold-2IB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Gold_2IB_trajectory.csv',
        'Gold-3PA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Gold_3PA_trajectory.csv',
        'Gold-4PB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Gold_4PB_trajectory.csv',
        'Gold-5IA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Gold_5IA_edit.csv',
        'Gold-6IB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Gold_6IB_trajectory.csv',
        'Gold-7PA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Gold_7PA_trajectory.csv',
        'Gold-8PB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Gold_8PB_trajectory.csv',
        '16A(78)-32': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Well_16A(78)-32_points_depths.csv',
        '16B(78)-32': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/16B(78)-32 P2.xlsx',
    }
}

projections = {'cape': pyproj.Proj("EPSG:26912"),
               'newberry': pyproj.Proj("EPSG:32610"),
               'JV': pyproj.Proj("EPSG:32611"),
               'DAC': pyproj.Proj("EPSG:26911"),}


color_dict = {
    'JV': {
        ('14-34'): 'black',
        ('18A-27', '46-28', '14-27', '81-28', '81A-28'): 'steelblue',
        ('86-28', '87-28', '77A-28'): 'firebrick',
    },
    'DAC': {
        ('68-1RD'): 'black',
        ('24-6', '24A-6', '26-6', '26A-6', '36-6', '24-6', '24A-6'): 'steelblue',
        ('64-11', '64A-11', '64B-11', '64C-11', '65-11', '65A-11', '85-11', '85A-11', '54-11', '54A-11'): 'firebrick',
    },
}

depth_correction = {
    'JV': 1446.,
    'DAC': 1286.,
}

catalog_colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def is_nlloc_loc_catalog_path(path):
    """Return True for NLLoc location hypothesis files (.loc.hyp variants)."""
    p = str(path).lower()
    return p.endswith('.loc.hyp')


def is_halliburton_catalog_path(path):
    """Return True for Halliburton catalog inputs that should plot in black."""
    p = os.path.basename(str(path)).lower()
    return 'halliburton' in p


def catalog_color_by_group(index, catalog_paths):
    """Assign categorical colors per catalog/event input."""
    if catalog_paths and index < len(catalog_paths) and is_halliburton_catalog_path(catalog_paths[index]):
        return 'black'
    return catalog_colors[index % len(catalog_colors)]

stage_colors = {
    'Frisco': '#f28e2b',
    'Bearskin': '#59a14f',
    'Gold': '#edc948',
}

newberry_stations = {
    'NN07': (634844.640, 4845632.818),
    'NN09': (634284.865, 4843588.322),
    'NN17': (634718.817, 4842218.195),
    'NN18': (636725.806, 4844127.781),
    'NN24': (636195.156, 4843503.594),
    'NN32': (634759.129, 4840340.145),
    'NNVM': (638819.167, 4841236.633),
    'NN19': (636380.039, 4841978.593),
    'NN21': (637721.337, 4843658.363),
}

# Wellhead location and surface elevation (UTM Zone 10N)
NEWBERRY_WH_LOC = np.array([635642.0, 4842835.0])
NEWBERRY_WH_ELEV = 1770.0
NEWBERRY_DAS_NLLOC_CFG = '/home/chopp/NLLoc/Newberry_DAS/run/locate_newberry_DAS.nlloc'

# Stage depths from Newberry_comparison.py (feet → metres)
STAGES_55_29 = np.array([9320., 9427., 9452., 9522.5, 9606., 9636., 9769.5, 9851.875]) * 0.3048
STAGES_55_29_JUL25 = np.array([9415., 9486.]) * 0.3048
STAGES_55A_29 = np.array([9575., 9625., 9660., 9750., 9850., 9980., 10064.]) * 0.3048

newberry_stage_sets = {
    '55-29': [
        {'label': '55-29 stages (2025)',     'color': '#e67e22', 'stages': STAGES_55_29},
        {'label': '55-29 stages (Jul 2025)', 'color': '#87CEFA', 'stages': STAGES_55_29_JUL25},
    ],
    '55A-29': [
        {'label': '55A-29 stages (2025)', 'color': '#16a085', 'stages': STAGES_55A_29},
    ],
}

# Lithology units: depth intervals in MD_m from wellhead (55-29 only)
# Top elevation ASL = NEWBERRY_WH_ELEV - depth_m
newberry_lith = {
    'Welded Tuff':  {'depths': [1966, 2057], 'color': 'darkkhaki'},
    'Tuff':         {'depths': [2057, 2439], 'color': 'khaki'},
    'Basalt':       {'depths': [2439, 2634], 'color': 'darkgray'},
    'Granodiorite': {'depths': [2634, 2908], 'color': 'bisque'},
    'Basalt (deep)':{'depths': [2908, 3067], 'color': 'darkgray'},
}


def add_newberry_station_markers(objects):
    """Add surface seismic stations as inverted triangles (UTM Zone 10N)."""
    east = [v[0] for v in newberry_stations.values()]
    north = [v[1] for v in newberry_stations.values()]
    # Station elevations are not in the dict; place markers at z=1770 (approx wellhead ASL)
    elev = [1770.0] * len(east)
    objects.append(go.Scatter3d(
        x=east, y=north, z=elev,
        mode='markers+text',
        name='Stations',
        marker=dict(size=6, color='darkgray', symbol='diamond', opacity=0.8,
                    line=dict(color='black', width=0.5)),
        text=list(newberry_stations.keys()),
        textposition='top center',
        textfont=dict(size=9, color='darkgray'),
        hoverinfo='text',
    ))


@lru_cache(maxsize=1)
def load_newberry_das_channels(cfg_path=NEWBERRY_DAS_NLLOC_CFG):
    """Load DAS channels with Grid2Time grids from GTSRCE lines in a .nlloc config.

    Both active and commented-out GTSRCE lines are included — active lines are
    channels queued for a new Grid2Time run; commented lines (``# GTSRCE …``)
    are channels whose travel-time grids already exist.  All of them can be
    used by NLLoc for location.
    """
    if not os.path.exists(cfg_path):
        return None
    channels = []
    with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as fobj:
        for line in fobj:
            line = line.strip()
            if not line:
                continue
            # Strip leading comment marker(s) so we can parse commented GTSRCE lines
            stripped = line.lstrip('# ').strip()
            if not stripped.startswith('GTSRCE'):
                continue
            parts = stripped.split()
            # Expected: GTSRCE <id> LATLON <lat> <lon> <depth_km> <elev>
            if len(parts) < 7:
                continue
            try:
                chan = parts[1]
                lat = float(parts[3])
                lon = float(parts[4])
                depth_km = float(parts[5])
            except (ValueError, IndexError):
                continue
            channels.append((chan, lat, lon, depth_km))
    if len(channels) == 0:
        return None
    utm = projections['newberry']
    lon = np.array([c[2] for c in channels], dtype=float)
    lat = np.array([c[1] for c in channels], dtype=float)
    east, north = utm(lon, lat)
    elev = -1000.0 * np.array([c[3] for c in channels], dtype=float)
    chan = [c[0] for c in channels]
    return chan, np.asarray(east), np.asarray(north), elev


def add_newberry_das_channel_markers(objects):
    """Add blue markers for DAS channels included in the NLLoc source list."""
    loaded = load_newberry_das_channels()
    if loaded is None:
        return None
    chan, east, north, elev = loaded
    objects.append(go.Scatter3d(
        x=east, y=north, z=elev,
        mode='markers',
        name='DAS channels (used)',
        marker=dict(size=3, color='royalblue', symbol='circle', opacity=0.85,
                    line=dict(color='midnightblue', width=0.5)),
        hovertext=[f'CHAN {c}' for c in chan],
        hoverinfo='text',
    ))
    return east, north, elev


def add_newberry_stage_markers(objects, datasets):
    """Interpolate stage positions along each borehole and add as Scatter3d markers.

    Uses column 4 (MD_m for 55-29, TVD metres for 55A-29) as the depth axis for
    interpolation — same convention as Newberry_comparison.py build_wellpath_overlays.
    """
    for well_label, stage_sets in newberry_stage_sets.items():
        wp_path = datasets.get(well_label)
        if not wp_path or not os.path.exists(wp_path):
            continue
        try:
            raw = np.loadtxt(wp_path, delimiter=',', skiprows=1)
        except Exception as e:
            logging.warning('Could not read %s for stage markers: %s', wp_path, e)
            continue
        md = raw[:, 4]
        east = raw[:, 0]
        north = raw[:, 1]
        elev = raw[:, 2]
        order = np.argsort(md)
        md, east, north, elev = md[order], east[order], north[order], elev[order]
        for stage_set in stage_sets:
            depths = stage_set['stages']
            se = np.interp(depths, md, east,  left=np.nan, right=np.nan)
            sn = np.interp(depths, md, north, left=np.nan, right=np.nan)
            sz = np.interp(depths, md, elev,  left=np.nan, right=np.nan)
            valid = np.isfinite(se) & np.isfinite(sn) & np.isfinite(sz)
            if not np.any(valid):
                continue
            objects.append(go.Scatter3d(
                x=se[valid], y=sn[valid], z=sz[valid],
                mode='markers',
                name=stage_set['label'],
                marker=dict(size=8, color=stage_set['color'], symbol='circle',
                            line=dict(color='black', width=0.8)),
                hovertext=[f"{stage_set['label']}: MD {v:.0f} m" for v in depths[valid]],
                hoverinfo='text',
            ))


def add_newberry_lithology_planes(objects):
    """Add semi-transparent horizontal planes marking the top of each lithology unit.

    Planes are 1 × 1 km squares centred on the wellhead (±500 m) at the ASL
    elevation of each unit top (NEWBERRY_WH_ELEV − depth_m), matching the
    HSpan bands in Newberry_comparison.py / Newberry_daily_report.py.
    """
    wh_e, wh_n = NEWBERRY_WH_LOC
    hw = 500.0  # half-width of plane in metres
    xc = [wh_e - hw, wh_e + hw, wh_e + hw, wh_e - hw]
    yc = [wh_n - hw, wh_n - hw, wh_n + hw, wh_n + hw]
    for unit, cfg in newberry_lith.items():
        top_elev = NEWBERRY_WH_ELEV - cfg['depths'][0]
        objects.append(go.Mesh3d(
            x=xc, y=yc, z=[top_elev] * 4,
            i=[0, 0], j=[1, 2], k=[2, 3],
            name=unit, color=cfg['color'],
            opacity=0.12, showlegend=True, hoverinfo='name',
        ))


def get_pixel_coords(dataset):
    band = dataset.GetRasterBand(1)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    transform = dataset.GetGeoTransform()
    xo = transform[0]
    yo = transform[3]
    pixw = transform[1]
    pixh = transform[5]
    return (np.arange(cols) * pixw) + xo, (np.arange(rows) * pixh) + yo, band


def read_trajectory_csv(path):
    """Read trajectory CSV (or survey XLSX) with or without a header row."""
    # --- XLSX survey format (e.g. FORGE 16B) ---
    if str(path).endswith('.xlsx'):
        return _read_survey_xlsx(path)
    try:
        frame = pd.read_csv(path)
        if 'geometry' in frame.columns and 'elevation_meters' in frame.columns:
            geometries = frame['geometry'].astype(str)
            lon = []
            lat = []
            for value in geometries:
                try:
                    point = wkt.loads(value)
                    lon.append(float(point.x))
                    lat.append(float(point.y))
                except Exception:
                    lon.append(np.nan)
                    lat.append(np.nan)
            lon = pd.to_numeric(pd.Series(lon), errors='coerce')
            lat = pd.to_numeric(pd.Series(lat), errors='coerce')
            good = lon.notna() & lat.notna()
            if good.any():
                transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:26912', always_xy=True)
                east, north = transformer.transform(lon[good].to_numpy(), lat[good].to_numpy())
                elev = pd.to_numeric(frame.loc[good, 'elevation_meters'], errors='coerce').to_numpy()
                coords = np.column_stack([east, north, elev])
                coords = coords[np.isfinite(coords).all(axis=1)]
                if len(coords) > 0:
                    return coords[:, :3]
        named_columns = [
            ('easting', 'northing', 'elevation'),
            ('Map Easting', 'Map Northing', 'Elevation'),
            ('easting_meters', 'northing_meters', 'elevation_meters'),
            ('utm_e_m', 'utm_n_m', 'elevation_meters'),
            ('UTM x (m)', 'UTM y (m)', 'elev z (m)'),
            ('easting_m', 'northing_m', 'elevation_m'),
            ('UTM_E', 'UTM_N', 'Elev_msl_m'),
            ('x', 'y', 'z'),
        ]
        for columns in named_columns:
            if all(column in frame.columns for column in columns):
                coords = frame.loc[:, list(columns)].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(dtype=float)
                if len(coords) > 0:
                    return coords[:, :3]
        if all(pd.api.types.is_numeric_dtype(frame[column]) or pd.to_numeric(frame[column], errors='coerce').notna().any() for column in frame.columns[:3]):
            numeric = frame.apply(pd.to_numeric, errors='coerce').dropna()
            if numeric.shape[1] >= 3 and len(numeric) > 0:
                return numeric.iloc[:, :3].to_numpy(dtype=float)
    except Exception:
        pass
    arr = np.genfromtxt(path, delimiter=',')
    if arr.ndim == 1:
        arr = np.atleast_2d(arr)
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.shape[1] < 3:
        raise ValueError(f'Trajectory CSV must have at least 3 numeric columns: {path}')
    return arr[:, :3]


def _read_survey_xlsx(path):
    """Read a directional-survey XLSX with a metadata header block.

    Handles the FORGE/Cape format where:
      - A header row contains 'MD' in column 0 (case-insensitive)
      - Units row follows immediately
      - Data rows begin at the first numeric row after the header
      - All depths are in US Survey Feet; NORTHING (col 6) and EASTING (col 8)
        are UTM Zone 12N in ftUS; SSTVD (col 4) is elevation above MSL in ft.
    """
    FTUS_TO_M = 1200 / 3937
    raw = pd.read_excel(path, header=None)
    header_row = None
    for i, row in raw.iterrows():
        if str(row.iloc[0]).strip().upper() == 'MD':
            header_row = i
            break
    if header_row is None:
        raise ValueError(f'Could not find MD header row in {path}')
    # Skip header + units rows, then drop non-numeric rows
    data = raw.iloc[header_row + 1:].copy()
    data = data[pd.to_numeric(data.iloc[:, 0], errors='coerce').notna()].reset_index(drop=True)
    east  = pd.to_numeric(data.iloc[:, 8], errors='coerce') * FTUS_TO_M
    north = pd.to_numeric(data.iloc[:, 6], errors='coerce') * FTUS_TO_M
    elev  = pd.to_numeric(data.iloc[:, 4], errors='coerce') * FTUS_TO_M
    valid = east.notna() & north.notna() & elev.notna()
    coords = np.column_stack([east[valid].to_numpy(dtype=float),
                              north[valid].to_numpy(dtype=float),
                              elev[valid].to_numpy(dtype=float)])
    coords = coords[np.isfinite(coords).all(axis=1)]
    if len(coords) == 0:
        raise ValueError(f'No valid numeric rows found in {path}')
    return coords


def _get_linestring_coords(geometry):
    gtype = geometry.get('type')
    coords = geometry.get('coordinates', [])
    if gtype == 'LineString':
        return coords
    if gtype == 'MultiLineString' and len(coords) > 0:
        return coords[0]
    return []


def _is_plausible_cape_utm(east, north):
    if len(east) == 0 or len(north) == 0:
        return False
    med_e = float(np.nanmedian(east))
    med_n = float(np.nanmedian(north))
    return 300000.0 <= med_e <= 400000.0 and 4200000.0 <= med_n <= 4300000.0


def _cape_spatial_root():
    return f'{data_directory}/cape_modern/spatial_data'


def load_cape_geojson_xy(label, trajectory_path):
    """Load Cape well XY from GeoJSON when available and plausible."""
    stem = os.path.basename(trajectory_path)
    stem = stem.replace('_trajectory.csv', '').replace('_xyz.csv', '').replace('.csv', '')
    aliases = {
        'Bearskin-6IB': 'Bearskin_6IA',
        'Bearskin_6IB': 'Bearskin_6IA',
    }
    root = _cape_spatial_root()
    candidates = []
    for key in {label, label.replace('-', '_'), stem, aliases.get(label), aliases.get(stem)}:
        if key is None:
            continue
        candidates.append(f'{root}/vector/{key}.geojson')
        candidates.append(f'{root}/vector/{key}_latlon.geojson')
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                gj = json.load(f)
            features = gj.get('features', [])
            if len(features) == 0:
                continue
            coords = _get_linestring_coords(features[0].get('geometry', {}))
            if len(coords) == 0:
                continue
            xy = np.array(coords, dtype=float)
            east = xy[:, 0]
            north = xy[:, 1]
            if np.nanmax(np.abs(east)) <= 180 and np.nanmax(np.abs(north)) <= 90:
                transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:26912', always_xy=True)
                east, north = transformer.transform(east, north)
            east = np.array(east)
            north = np.array(north)
            if not np.all(np.isfinite(east)) or not np.all(np.isfinite(north)):
                logging.warning('Skipping non-finite GeoJSON XY for %s from %s', label, path)
                continue
            if not _is_plausible_cape_utm(east, north):
                logging.warning('Skipping implausible Cape GeoJSON XY for %s from %s', label, path)
                continue
            return east, north
        except Exception as e:
            logging.warning('Failed to parse well GeoJSON %s: %s', path, e)
    return None, None


def estimate_cape_shared_offset(datasets_dict):
    """Estimate shared (dE, dN) offset from trustworthy GeoJSON-to-CSV pairs."""
    offsets = []
    for label, data in datasets_dict.items():
        if data.endswith(('tif', 'nc')) or data.endswith('JV.csv'):
            continue
        try:
            traj = read_trajectory_csv(data)
        except Exception:
            continue
        geo_east, geo_north = load_cape_geojson_xy(label, data)
        if geo_east is None or geo_north is None:
            continue
        n = min(len(traj), len(geo_east), len(geo_north))
        if n < 2:
            continue
        de = np.nanmedian(geo_east[:n] - traj[:n, 0])
        dn = np.nanmedian(geo_north[:n] - traj[:n, 1])
        if np.isfinite(de) and np.isfinite(dn):
            offsets.append((de, dn, label))
    if len(offsets) == 0:
        return 0.0, 0.0
    nontrivial = [(de, dn) for de, dn, _ in offsets if np.hypot(de, dn) > 50.0]
    pool = nontrivial if len(nontrivial) > 0 else [(de, dn) for de, dn, _ in offsets]
    dE = float(np.nanmean([de for de, _ in pool]))
    dN = float(np.nanmean([dn for _, dn in pool]))
    logging.info('Estimated Cape shared CSV offset from %d wells: dE=%.3f dN=%.3f', len(pool), dE, dN)
    return dE, dN


@lru_cache(maxsize=1)
def load_cape_stage_table():
    """Load normalized Cape stage table and keep only rows with plottable XYZ."""
    if not os.path.exists(cape_stage_table_path):
        logging.warning('Cape stage table not found: %s', cape_stage_table_path)
        return pd.DataFrame()
    try:
        table = pd.read_csv(cape_stage_table_path)
    except Exception as exc:
        logging.warning('Failed reading Cape stage table %s: %s', cape_stage_table_path, exc)
        return pd.DataFrame()
    expected = {'field', 'well', 'stage', 'x_m', 'y_m', 'z_m'}
    if not expected.issubset(set(table.columns)):
        logging.warning('Cape stage table missing expected columns: %s', expected - set(table.columns))
        return pd.DataFrame()
    table = table[table['field'].astype(str).isin({'Frisco', 'Bearskin', 'Gold'})].copy()
    for col in ['x_m', 'y_m', 'z_m', 'stage']:
        table[col] = pd.to_numeric(table[col], errors='coerce')
    if 'start_time' in table.columns:
        table['start_time'] = pd.to_datetime(table['start_time'], errors='coerce')
    if 'end_time' in table.columns:
        table['end_time'] = pd.to_datetime(table['end_time'], errors='coerce')
    table = table.dropna(subset=['x_m', 'y_m', 'z_m', 'stage'])
    table = table.sort_values(['well', 'stage'])
    return table


def add_cape_stage_markers(objects, bounds):
    """Append stage markers and stage-number text traces to the 3D figure objects."""
    stage_table = load_cape_stage_table()
    if stage_table.empty:
        return

    def update_bounds(x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(good):
            return
        xg = x[good]
        yg = y[good]
        zg = z[good]
        bounds['xmin'] = min(bounds['xmin'], float(np.min(xg)))
        bounds['xmax'] = max(bounds['xmax'], float(np.max(xg)))
        bounds['ymin'] = min(bounds['ymin'], float(np.min(yg)))
        bounds['ymax'] = max(bounds['ymax'], float(np.max(yg)))
        bounds['zmin'] = min(bounds['zmin'], float(np.min(zg)))
        bounds['zmax'] = max(bounds['zmax'], float(np.max(zg)))

    for well, group in stage_table.groupby('well'):
        group = group.sort_values('stage')
        field_name = str(group['field'].iloc[0])
        color = stage_colors.get(field_name, '#4e79a7')
        start_txt = group.get('start_time', pd.Series([pd.NaT] * len(group))).astype(str)
        end_txt = group.get('end_time', pd.Series([pd.NaT] * len(group))).astype(str)
        hover = [
            (
                f'Well: {w}<br>'
                f'Stage: {int(s)}<br>'
                f'Start: {st if st != "NaT" else "N/A"}<br>'
                f'End: {et if et != "NaT" else "N/A"}'
            )
            for w, s, st, et in zip(group['well'], group['stage'], start_txt, end_txt)
        ]
        objects.append(go.Scatter3d(
            x=group['x_m'],
            y=group['y_m'],
            z=group['z_m'],
            mode='markers',
            name=f'{well} stages',
            legendgroup='injection-stages',
            marker=dict(size=6, color=color, symbol='diamond', opacity=0.9, line=dict(color='black', width=0.4)),
            hoverinfo='text',
            text=hover,
        ))
        objects.append(go.Scatter3d(
            x=group['x_m'],
            y=group['y_m'],
            z=group['z_m'],
            mode='text',
            name=f'{well} stage labels',
            legendgroup='injection-stages',
            showlegend=False,
            text=[str(int(s)) for s in group['stage']],
            textposition='top center',
            textfont=dict(size=9, color=color),
            hoverinfo='skip',
        ))
        update_bounds(group['x_m'], group['y_m'], group['z_m'])

def plot_datasets_3d(datasets, field='cape', show_lith=False):
    objects = []
    well_x = []
    well_y = []
    well_z = []
    data_xmin, data_xmax = np.inf, -np.inf
    data_ymin, data_ymax = np.inf, -np.inf
    data_zmin, data_zmax = np.inf, -np.inf

    def update_bounds(x, y, z):
        nonlocal data_xmin, data_xmax, data_ymin, data_ymax, data_zmin, data_zmax
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(good):
            return
        xg = x[good]
        yg = y[good]
        zg = z[good]
        data_xmin = min(data_xmin, float(np.min(xg)))
        data_xmax = max(data_xmax, float(np.max(xg)))
        data_ymin = min(data_ymin, float(np.min(yg)))
        data_ymax = max(data_ymax, float(np.max(yg)))
        data_zmin = min(data_zmin, float(np.min(zg)))
        data_zmax = max(data_zmax, float(np.max(zg)))
    cape_offset_e = 0.0
    cape_offset_n = 0.0
    if field == 'cape':
        cape_offset_e, cape_offset_n = estimate_cape_shared_offset(datasets)
    if field == 'newberry':
        add_newberry_station_markers(objects)
        das_xyz = add_newberry_das_channel_markers(objects)
        if das_xyz is not None:
            update_bounds(das_xyz[0], das_xyz[1], das_xyz[2])
        if show_lith:
            add_newberry_lithology_planes(objects)
    for label, data in datasets.items():
        if not data.endswith(('tif', 'nc')):
            wellpath = read_trajectory_csv(data)
            east = wellpath[:, 0]
            north = wellpath[:, 1]
            dep_m = wellpath[:, 2]
            if field == 'cape':
                geo_east, geo_north = load_cape_geojson_xy(label, data)
                if geo_east is not None and geo_north is not None:
                    npts = min(len(dep_m), len(geo_east), len(geo_north))
                    east = geo_east[:npts]
                    north = geo_north[:npts]
                    dep_m = dep_m[:npts]
                else:
                    east = east + cape_offset_e
                    north = north + cape_offset_n
            objects.append(go.Scatter3d(
                x=east, y=north, z=dep_m,
                name=label,
                mode='lines',
                line=dict(color='black', width=6),
                hoverinfo='skip'
            ))
            well_x.append(np.asarray(east))
            well_y.append(np.asarray(north))
            well_z.append(np.asarray(dep_m))
            update_bounds(east, north, dep_m)
        elif data.endswith('tif'):
            topo = gdal.Open(data, gdal.GA_ReadOnly)
            x, y, band = get_pixel_coords(topo)
            X, Y = np.meshgrid(x, y, indexing='xy')
            raster_values = band.ReadAsArray()
            topo_mesh = go.Mesh3d(
                x=X.flatten(), y=Y.flatten(), z=raster_values.flatten(),
                name=label, color='gray', opacity=0.18, delaunayaxis='z', showlegend=True,
                hoverinfo='skip'
            )
            objects.append(topo_mesh)
            update_bounds(X.flatten(), Y.flatten(), raster_values.flatten())
        elif data.endswith('nc'):
            tob = xr.load_dataarray(data)
            tob = tob.interp(easting=tob.easting[::10], northing=tob.northing[::10])
            X, Y = np.meshgrid(tob.easting, tob.northing, indexing='xy')
            Z = tob.values.flatten()
            tob_mesh = go.Mesh3d(
                x=X.flatten(), y=Y.flatten(), z=Z,
                name=label, color='gray', opacity=0.12, delaunayaxis='z', showlegend=True,
                hoverinfo='skip'
            )
            objects.append(tob_mesh)
            update_bounds(X.flatten(), Y.flatten(), Z)

    if field == 'newberry':
        add_newberry_stage_markers(objects, datasets)

    bounds = dict(
        xmin=data_xmin, xmax=data_xmax,
        ymin=data_ymin, ymax=data_ymax,
        zmin=data_zmin, zmax=data_zmax,
    )
    return objects, well_x, well_y, well_z, bounds

def get_catalog_params(catalog, utm):
    params = []
    for ev in catalog:
        o = ev.preferred_origin()
        try:
            m = ev.preferred_magnitude().mag
        except Exception:
            m = 0.5
        params.append([ev.resource_id.id, o.time.timestamp, o.latitude, o.longitude, o.depth, m])
    params = np.array(params)
    if len(params) == 0:
        return None
    id, t, lat, lon, depth, m = np.split(params, 6, axis=1)
    t = t.astype('f').flatten()
    lat = lat.astype('f').flatten()
    lon = lon.astype('f').flatten()
    depth = depth.astype('f').flatten()
    m = m.astype('f').flatten()
    ev_east, ev_north = utm(lon, lat)
    depth = np.array(depth) * -1
    return id, t, lat, lon, depth, m, ev_east, ev_north

def make_3d_figure(catalogs, catalog_names, datasets, field='cape', scale_by_magnitude=False, color_by_time=False, apply_cape_correction=False, show_lith=False, show_scat=False, catalog_paths=None):
    objects, well_x, well_y, well_z, bounds = plot_datasets_3d(datasets, field=field, show_lith=show_lith)
    mfact = 2.5
    utm = projections[field]

    def dynamic_base_marker_size(n_events):
        """Auto-size markers: larger for sparse catalogs, ~2 for ~10k events."""
        n = max(int(n_events), 1)
        # n=10 -> ~6.5, n=100 -> ~5, n=1000 -> ~3.5, n=10000 -> ~2
        return float(np.clip(8.0 - 1.5 * np.log10(n), 2.0, 10.0))

    cape_offset_e = 0.0
    cape_offset_n = 0.0
    if field == 'cape' and apply_cape_correction:
        cape_offset_e, cape_offset_n = estimate_cape_shared_offset(datasets)

    if field == 'cape':
        add_cape_stage_markers(objects, bounds)

    def update_bounds(x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(good):
            return
        xg = x[good]
        yg = y[good]
        zg = z[good]
        bounds['xmin'] = min(bounds['xmin'], float(np.min(xg)))
        bounds['xmax'] = max(bounds['xmax'], float(np.max(xg)))
        bounds['ymin'] = min(bounds['ymin'], float(np.min(yg)))
        bounds['ymax'] = max(bounds['ymax'], float(np.max(yg)))
        bounds['zmin'] = min(bounds['zmin'], float(np.min(zg)))
        bounds['zmax'] = max(bounds['zmax'], float(np.max(zg)))
    for i, catalog in enumerate(catalogs):
        result = get_catalog_params(catalog, utm)
        if result is None:
            continue
        id, t, lat, lon, depth, m, ev_east, ev_north = result
        if field == 'cape' and apply_cape_correction:
            ev_east = ev_east + cape_offset_e
            ev_north = ev_north + cape_offset_n
        tickvals = np.linspace(min(t), max(t), 10)
        ticktext = [datetime.fromtimestamp(int(tv)).strftime('%d %b %Y: %H:%M') for tv in tickvals]
        base_size = dynamic_base_marker_size(len(m))
        if scale_by_magnitude:
            # Keep relative magnitude scaling, but adapt absolute size to event density.
            density_scale = base_size / 2.0
            marker_size = np.maximum((mfact * np.array(m)) ** 2 * density_scale, base_size)
        else:
            marker_size = np.full_like(m, base_size, dtype=float)
        if color_by_time:
            marker_color = t
            marker_dict = dict(
                color=marker_color,
                cmin=min(tickvals),
                cmax=max(tickvals),
                size=marker_size,
                symbol='circle',
                line=dict(color=marker_color, width=1, colorscale='Cividis'),
                colorbar=dict(
                    title=dict(text='Timestamp', font=dict(size=18)),
                    x=-0.2,
                    ticktext=ticktext,
                    tickvals=tickvals
                ),
                colorscale='Bluered',
                opacity=0.5
            )
        else:
            marker_color = catalog_color_by_group(i, catalog_paths)
            marker_dict = dict(
                color=marker_color,
                size=marker_size,
                symbol='circle',
                line=dict(color=marker_color, width=1),
                opacity=0.5
            )
        scat_obj = go.Scatter3d(
            x=ev_east, y=ev_north, z=depth,
            mode='markers',
            name=catalog_names[i],
            hoverinfo='text',
            text=np.array(id),
            marker=marker_dict
        )
        objects.append(scat_obj)
        update_bounds(ev_east, ev_north, depth)

        if catalog_paths and i < len(catalog_paths):
            cloud = load_nlloc_cloud_for_hyp(catalog_paths[i], field)
            if cloud is not None:
                cloud_e, cloud_n, cloud_z, cloud_w = cloud
                if cloud_e.size > 8000:
                    idx = np.argsort(cloud_w)[-8000:]
                    cloud_e = cloud_e[idx]
                    cloud_n = cloud_n[idx]
                    cloud_z = cloud_z[idx]
                    cloud_w = cloud_w[idx]
                w_max = float(np.nanmax(cloud_w)) if cloud_w.size > 0 else 0.0
                w_norm = cloud_w / w_max if w_max > 0 else np.ones_like(cloud_w)
                cloud_obj = go.Scatter3d(
                    x=cloud_e, y=cloud_n, z=cloud_z,
                    mode='markers',
                    name=f'{catalog_names[i]} cloud',
                    visible=True if show_scat else 'legendonly',
                    marker=dict(size=2, color=w_norm, colorscale='Viridis', opacity=0.25)
                )
                objects.append(cloud_obj)
                update_bounds(cloud_e, cloud_n, cloud_z)

    x_range = None
    y_range = None
    z_range = None
    aspectratio = dict(x=1, y=1, z=1)
    if len(well_x) > 0 and len(well_y) > 0 and len(well_z) > 0 and np.isfinite(bounds['xmin']):
        all_x = np.concatenate(well_x)
        all_y = np.concatenate(well_y)
        all_z = np.concatenate(well_z)
        cx = float(np.nanmean(all_x))
        cy = float(np.nanmean(all_y))
        cz = float(np.nanmean(all_z))
        half_span = max(
            cx - bounds['xmin'], bounds['xmax'] - cx,
            cy - bounds['ymin'], bounds['ymax'] - cy,
            cz - bounds['zmin'], bounds['zmax'] - cz,
            1.
        ) * 1.02
        x_range = [cx - half_span, cx + half_span]
        y_range = [cy - half_span, cy + half_span]
        z_range = [cz - half_span, cz + half_span]
        x_span = max(float(x_range[1] - x_range[0]), 1.)
        y_span = max(float(y_range[1] - y_range[0]), 1.)
        z_span = max(float(z_range[1] - z_range[0]), 1.)
        scale = max(x_span, y_span, z_span)
        aspectratio = dict(x=x_span / scale, y=y_span / scale, z=z_span / scale)

    fig = go.Figure(data=objects)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Easting (m)', range=x_range, autorange=False),
            yaxis=dict(title='Northing (m)', range=y_range, autorange=False),
            zaxis=dict(title='Elevation (m)', range=z_range, autorange=False),
            aspectmode='manual',
            aspectratio=aspectratio,
            bgcolor="rgb(244, 244, 248)",
            uirevision='cape-3d'
        ),
        title='3D Seismicity',
        legend=dict(itemsizing='constant', bgcolor='whitesmoke', bordercolor='gray', borderwidth=1),
        height=900,
        uirevision='cape-3d'
    )
    return fig

def make_cumulative_figure(catalogs, catalog_names, catalog_paths=None):
    curves = []
    pick_curves = []
    for i, catalog in enumerate(catalogs):
        times = [ev.preferred_origin().time.datetime for ev in catalog]
        times = sorted(times)
        if not times:
            continue
        df = pd.DataFrame({'time': times, 'count': np.arange(1, len(times)+1)})
        curves.append(go.Scatter(
            x=df['time'], y=df['count'],
            mode='lines+markers',
            name=f'{catalog_names[i]} Events',
            yaxis='y1',
            line=dict(color=catalog_color_by_group(i, catalog_paths))
        ))
        # Cumulative picks
        pick_times = []
        for ev in catalog:
            for arr in ev.origins[0].arrivals:
                try:
                    pick = arr.pick_id.get_referred_object()
                    pick_times.append(pick.time.datetime)
                except Exception:
                    continue
        pick_times = sorted(pick_times)
        if pick_times:
            dfp = pd.DataFrame({'time': pick_times, 'count': np.arange(1, len(pick_times)+1)})
            pick_curves.append(go.Scatter(
                x=dfp['time'], y=dfp['count'],
                mode='lines',
                name=f'{catalog_names[i]} Picks',
                yaxis='y2',
                line=dict(dash='dot', color=catalog_color_by_group(i, catalog_paths))
            ))
    fig = go.Figure(data=curves + pick_curves)
    fig.update_layout(
        title='Cumulative Number of Events and Picks',
        xaxis_title='Time',
        yaxis=dict(title='Cumulative Events'),
        yaxis2=dict(title='Cumulative Picks', overlaying='y', side='right', showgrid=False),
        legend=dict(itemsizing='constant', bgcolor='whitesmoke', bordercolor='gray', borderwidth=1),
        height=450
    )
    return fig

def make_arrivals_histogram(catalogs, catalog_names, catalog_paths=None):
    hists = []
    for i, catalog in enumerate(catalogs):
        seed_ids = []
        for ev in catalog:
            for arr in ev.origins[0].arrivals:
                try:
                    pick = arr.pick_id.get_referred_object()
                    seed_ids.append(pick.waveform_id.get_seed_string())
                except Exception:
                    continue
        if not seed_ids:
            continue
        s, counts = np.unique(seed_ids, return_counts=True)
        hists.append(go.Bar(
            x=s, y=counts,
            name=catalog_names[i],
            marker=dict(color=catalog_color_by_group(i, catalog_paths))
        ))
    fig = go.Figure(data=hists)
    fig.update_layout(
        title='Histogram of Arrivals by SEED ID',
        xaxis_title='SEED ID',
        yaxis_title='Arrivals',
        barmode='group',
        legend=dict(itemsizing='constant', bgcolor='whitesmoke', bordercolor='gray', borderwidth=1),
        height=450
    )
    return fig


def get_catalog_time_range(catalogs):
    """Return (min_date, max_date) as datetime.date across all catalogs."""
    times = []
    for cat in catalogs:
        for ev in cat:
            o = ev.preferred_origin()
            if o and o.time:
                times.append(o.time.datetime)
    if not times:
        today = datetime.now().date()
        return today, today
    return min(times).date(), max(times).date()


def filter_catalogs_by_date(catalogs, start_date, end_date):
    """Return catalogs filtered to [start_date, end_date] (inclusive).

    Uses direct UTCDateTime comparison rather than ObsPy's string-filter
    API, which can silently drop events when timestamps are passed as floats.
    """
    from obspy import UTCDateTime, Catalog
    t0 = UTCDateTime(start_date)
    t1 = UTCDateTime(end_date) + 86400  # include full end day
    result = []
    for cat in catalogs:
        kept = [
            ev for ev in cat
            if ev.preferred_origin() is not None
            and ev.preferred_origin().time is not None
            and t0 <= ev.preferred_origin().time <= t1
        ]
        logging.info('Date filter %s–%s: %d/%d events kept', start_date, end_date, len(kept), len(cat))
        result.append(Catalog(events=kept))
    return result


def normalize_field(field_name):
    key = str(field_name).strip().lower()
    mapping = {
        'cape': 'Cape',
        'newberry': 'Newberry',
        'jv': 'JV',
        'dac': 'DAC',
    }
    if key not in mapping:
        raise ValueError(f"Unsupported field '{field_name}'. Choose one of: Cape, Newberry, JV, DAC")
    dataset_key = mapping[key]
    projection_key = key if key in projections else dataset_key
    return dataset_key, projection_key


def nlloc_event_key(path):
    """Return a stable key that collapses summary/full NLLoc .hyp variants."""
    name = os.path.basename(path)
    if not name.endswith('.hyp'):
        return os.path.splitext(name)[0]
    stem = name[:-4]
    stem = re.sub(r'\.sum(?=\.grid\d+\.loc$)', '', stem)
    stem = re.sub(r'\.\d{8}\.\d{6}(?=\.grid\d+\.loc$)', '', stem)
    return stem


def dedupe_catalog_paths(catalog_paths):
    """Keep one .hyp per event key, preferring .sum and dropping redundant .last."""
    chosen = {}
    order = []
    removed_last = 0
    for path in catalog_paths:
        name = os.path.basename(str(path)).lower()
        if name.endswith('.hyp') and '.last.' in name:
            removed_last += 1
            continue
        key = nlloc_event_key(path)
        if key not in chosen:
            chosen[key] = path
            order.append(key)
            continue
        old_path = chosen[key]
        old_is_sum = '.sum.grid' in os.path.basename(old_path)
        new_is_sum = '.sum.grid' in os.path.basename(path)
        if new_is_sum and not old_is_sum:
            chosen[key] = path
    deduped = [chosen[key] for key in order]
    removed_dupes = len(catalog_paths) - len(deduped) - removed_last
    if removed_dupes > 0:
        logging.info('Removed %d duplicate NLLoc .hyp inputs (summary/full duplicates).', removed_dupes)
    if removed_last > 0:
        logging.info('Removed %d redundant NLLoc .last catalogs.', removed_last)
    return deduped


def parse_nlloc_hdr(hdr_path):
    """Parse NLLoc .hdr grid and SIMPLE transform metadata."""
    if not os.path.exists(hdr_path):
        return None
    with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as fobj:
        lines = fobj.readlines()
    if len(lines) < 2:
        return None

    parts = lines[0].split()
    if len(parts) < 9:
        return None
    nx, ny, nz = int(parts[0]), int(parts[1]), int(parts[2])
    x0, y0, z0 = float(parts[3]), float(parts[4]), float(parts[5])
    dx, dy, dz = float(parts[6]), float(parts[7]), float(parts[8])

    match = re.search(r'LatOrig\s+([-\d.]+)\s+LongOrig\s+([-\d.]+)', lines[1])
    if not match:
        return None
    lat0 = float(match.group(1))
    lon0 = float(match.group(2))
    return {
        'nx': nx, 'ny': ny, 'nz': nz,
        'x0': x0, 'y0': y0, 'z0': z0,
        'dx': dx, 'dy': dy, 'dz': dz,
        'lat0': lat0, 'lon0': lon0,
    }


def find_scat_for_hyp(hyp_path):
    """Find matching .scat path for a .hyp file when available."""
    direct = hyp_path[:-4] + '.scat'
    if os.path.exists(direct):
        return direct
    folder = os.path.dirname(hyp_path)
    stem = os.path.basename(hyp_path)[:-4]
    # Build a stable event root that matches both summary and full/timestamped outputs.
    root = re.sub(r'\.sum(?=\.grid\d+\.loc$)', '', stem)
    root = re.sub(r'\.\d{8}\.\d{6}(?=\.grid\d+\.loc$)', '', root)
    root = re.sub(r'\.grid\d+\.loc$', '', root)
    candidates = sorted(glob(os.path.join(folder, f'{root}*.grid*.loc.scat')))
    return candidates[0] if candidates else None


def read_nlloc_scat_points(scat_path, hdr_meta):
    """Read NLLoc .scat binary as float32 quadruples with offset auto-detection."""
    raw = np.fromfile(scat_path, dtype=np.uint8)
    if raw.size < 32:
        return None

    xmin = hdr_meta['x0'] - hdr_meta['dx']
    xmax = hdr_meta['x0'] + (hdr_meta['nx'] - 1) * hdr_meta['dx'] + hdr_meta['dx']
    ymin = hdr_meta['y0'] - hdr_meta['dy']
    ymax = hdr_meta['y0'] + (hdr_meta['ny'] - 1) * hdr_meta['dy'] + hdr_meta['dy']
    zmin = hdr_meta['z0'] - hdr_meta['dz']
    zmax = hdr_meta['z0'] + (hdr_meta['nz'] - 1) * hdr_meta['dz'] + hdr_meta['dz']

    best_rows = None
    best_valid = None
    best_score = -1

    for offset in (0, 4, 8, 12):
        if raw.size <= offset + 16:
            continue
        arr = np.frombuffer(raw[offset:], dtype='<f4')
        n4 = (arr.size // 4) * 4
        if n4 < 16:
            continue
        rows = arr[:n4].reshape(-1, 4)
        x_vals, y_vals, z_vals, w_vals = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        valid = (
            np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(z_vals) & np.isfinite(w_vals)
            & (w_vals > 0.0)
            & (x_vals >= xmin) & (x_vals <= xmax)
            & (y_vals >= ymin) & (y_vals <= ymax)
            & (z_vals >= zmin) & (z_vals <= zmax)
        )
        score = int(np.sum(valid))
        if score > best_score:
            best_score = score
            best_rows = rows
            best_valid = valid

    if best_rows is None or best_score <= 0:
        return None
    picked = best_rows[best_valid]
    return picked[:, 0], picked[:, 1], picked[:, 2], picked[:, 3]


@lru_cache(maxsize=256)
def load_nlloc_cloud_for_hyp(hyp_path, field):
    """Return cloud arrays (east, north, elev, weight) for a .hyp, if available."""
    scat_path = find_scat_for_hyp(hyp_path)
    if not scat_path:
        return None

    hdr_path = scat_path[:-5] + '.hdr'
    hdr_meta = parse_nlloc_hdr(hdr_path)
    if hdr_meta is None:
        return None

    parsed = read_nlloc_scat_points(scat_path, hdr_meta)
    if parsed is None:
        return None

    # NLLoc SIMPLE coordinates are local Cartesian (km) around (LatOrig, LongOrig),
    # not native UTM metres. Convert x/y km -> lat/lon first, then project to UTM.
    x_km, y_km, z_km, weight = parsed
    lat0 = hdr_meta['lat0']
    lon0 = hdr_meta['lon0']
    deg_per_km = 1.0 / 111.111
    lat = lat0 + y_km * deg_per_km
    cos_lat0 = np.cos(np.deg2rad(lat0))
    if np.abs(cos_lat0) < 1e-8:
        return None
    lon = lon0 + x_km * (deg_per_km / cos_lat0)
    utm = projections[field]
    east, north = utm(lon, lat)
    elev = -z_km * 1000.0
    return east, north, elev, weight


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description='Catalog comparison dashboard with 3D borehole context.'
    )
    parser.add_argument(
        'inputs',
        nargs='*',
        help='Catalog file paths followed optionally by field name (Cape/Newberry/JV/DAC).'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Dash host interface (default: 0.0.0.0).'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Dash port (default: 8050).'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run Dash in debug mode.'
    )
    parser.add_argument(
        '--export-html',
        metavar='OUTPUT.html',
        default=None,
        help='Export a static HTML snapshot of all figures and exit (no server started).'
    )
    args = parser.parse_args()

    field = 'Cape'
    catalog_paths = []
    if len(args.inputs) > 0:
        possible_field = args.inputs[-1].strip().lower()
        if possible_field in {'cape', 'newberry', 'jv', 'dac'}:
            field = args.inputs[-1]
            catalog_paths = args.inputs[:-1]
        else:
            catalog_paths = args.inputs

    if len(catalog_paths) == 0:
        catalog_paths = [
            '/home/chopp/Cape_noise.xml',
            '/home/chopp/Cape_events.xml',
        ]

    catalog_names = [os.path.splitext(os.path.basename(p))[0] for p in catalog_paths]
    dataset_key, projection_key = normalize_field(field)
    return args, catalog_paths, catalog_names, dataset_key, projection_key


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def load_catalog_from_csv(path):
    """Load CSV with event rows into an ObsPy Catalog for plotting compatibility."""
    frame = pd.read_csv(path)
    basename = os.path.basename(path).lower()
    latlon_authoritative_csv = 'mazama newberry microseismic catalog with distance from perfs halliburton.csv'
    time_mask = pd.Series(True, index=frame.index)

    if 'date_time_UTC' in frame.columns:
        times = pd.to_datetime(frame['date_time_UTC'], errors='coerce', utc=True)
    elif all(col in frame.columns for col in ['Origin Date (Local)', 'Origin Time (Local)', 'Time Zone Offset']):
        times = pd.to_datetime(
            frame['Origin Date (Local)'].astype(str) + ' '
            + frame['Origin Time (Local)'].astype(str) + ' '
            + frame['Time Zone Offset'].astype(str),
            errors='coerce', utc=True,
        )
    else:
        raise ValueError(f'CSV has no usable timestamp columns: {path}')

    if basename == latlon_authoritative_csv:
        # Keep only January events for this catalog.
        time_mask = times.dt.month.eq(1)
        logging.info(
            'CSV %s: keeping January events only (%d of %d rows)',
            os.path.basename(path),
            int(time_mask.fillna(False).sum()),
            int(len(frame)),
        )

    if basename == latlon_authoritative_csv:
        lat = pd.to_numeric(frame.get('Latitude', np.nan), errors='coerce')
        lon = pd.to_numeric(frame.get('Longitude', np.nan), errors='coerce')
        valid_en = pd.Series(True, index=frame.index)
        if not (lat.notna() & lon.notna()).any():
            raise ValueError(f'CSV requires valid Latitude/Longitude rows for {path}')
        logging.info('CSV %s: using Latitude/Longitude as authoritative plotting coordinates', os.path.basename(path))
    else:
        required_spatial_cols = ['Easting ft (Abs)', 'Northing ft (Abs)']
        missing_spatial_cols = [col for col in required_spatial_cols if col not in frame.columns]
        if missing_spatial_cols:
            raise ValueError(
                f'CSV requires projected columns {required_spatial_cols} for strict UTM parsing; '
                f'missing {missing_spatial_cols}: {path}'
            )

        east = pd.to_numeric(frame['Easting ft (Abs)'], errors='coerce')
        north = pd.to_numeric(frame['Northing ft (Abs)'], errors='coerce')
        valid_en = east.notna() & north.notna()
        if not valid_en.any():
            raise ValueError(f'CSV has no valid projected E/N rows in required columns {required_spatial_cols}: {path}')

        src_epsg = 26910
        try:
            transformer = pyproj.Transformer.from_crs(f'EPSG:{src_epsg}', 'EPSG:4326', always_xy=True)
            lon_vals, lat_vals = transformer.transform(
                east.to_numpy(dtype=float),
                north.to_numpy(dtype=float),
            )
        except Exception as exc:
            raise ValueError(
                f'Failed strict UTM->lat/lon transform from EPSG:{src_epsg} for {path}: {exc}'
            ) from exc

        lon = pd.Series(lon_vals, index=frame.index)
        lat = pd.Series(lat_vals, index=frame.index)

    if 'Depth ft (TVDSS)' in frame.columns:
        depth_m = -pd.to_numeric(frame['Depth ft (TVDSS)'], errors='coerce') * 0.3048
    elif 'Depth ft (TVD)' in frame.columns:
        depth_m = pd.to_numeric(frame['Depth ft (TVD)'], errors='coerce') * 0.3048
    else:
        depth_m = pd.Series(np.nan, index=frame.index)

    mag = pd.to_numeric(frame.get('Magnitude', np.nan), errors='coerce')
    evid = frame.get('Event Number', pd.Series(frame.index + 1)).astype(str)

    good = times.notna() & lat.notna() & lon.notna() & valid_en & time_mask
    if not np.any(good):
        return Catalog(events=[])

    cat = Catalog(events=[])
    for idx in np.where(good.to_numpy())[0]:
        tstamp = UTCDateTime(times.iloc[idx].to_pydatetime())
        origin = Origin(
            time=tstamp,
            latitude=float(lat.iloc[idx]),
            longitude=float(lon.iloc[idx]),
            depth=_safe_float(depth_m.iloc[idx]) if np.isfinite(_safe_float(depth_m.iloc[idx])) else 0.0,
            resource_id=ResourceIdentifier(f'{os.path.basename(path)}#origin-{idx+1}')
        )
        event = Event(resource_id=ResourceIdentifier(f'{os.path.basename(path)}#event-{evid.iloc[idx]}'))
        event.origins = [origin]
        event.preferred_origin_id = origin.resource_id
        mval = _safe_float(mag.iloc[idx])
        if np.isfinite(mval):
            magnitude = Magnitude(
                mag=float(mval),
                resource_id=ResourceIdentifier(f'{os.path.basename(path)}#mag-{idx+1}')
            )
            event.magnitudes = [magnitude]
            event.preferred_magnitude_id = magnitude.resource_id
        cat.events.append(event)
    return cat


def load_catalog_input(path):
    """Load either standard event files (ObsPy) or CSV event tables."""
    if str(path).lower().endswith('.csv'):
        return load_catalog_from_csv(path)
    return read_events(path)

# --- MAIN DASH APP ---

cli_args, catalog_paths, catalog_names, dataset_key, projection_key = parse_cli_args()
catalog_paths = dedupe_catalog_paths(catalog_paths)
catalog_names = [os.path.splitext(os.path.basename(path))[0] for path in catalog_paths]
datas = datasets[dataset_key]
catalogs = [load_catalog_input(path) for path in catalog_paths]

app_state = {
    'catalogs': catalogs,
    'catalog_names': catalog_names,
    'catalog_paths': catalog_paths,
    'datas': datas,
    'field': projection_key,
}

app = dash.Dash(__name__)

catalog_date_min, catalog_date_max = get_catalog_time_range(catalogs)

app.layout = html.Div([
    html.H1("Seismic Catalog Comparison Dashboard"),
    html.Div([
        html.Label("Date range:", style={'marginRight': '8px', 'fontWeight': 'bold'}),
        dcc.DatePickerRange(
            id='date-range-picker',
            min_date_allowed=catalog_date_min,
            max_date_allowed=catalog_date_max,
            start_date=catalog_date_min,
            end_date=catalog_date_max,
            display_format='YYYY-MM-DD',
            style={'display': 'inline-block'},
        ),
        html.Span(id='event-count-label', style={'marginLeft': '16px', 'color': '#555', 'fontStyle': 'italic'}),
    ], style={'marginBottom': '12px'}),
    html.Label("Scale 3D markers by magnitude:"),
    dcc.RadioItems(
        id='scale-mag-toggle',
        options=[
            {'label': 'Yes', 'value': 'yes'},
            {'label': 'No', 'value': 'no'}
        ],
        value='no',
        inline=True
    ),
    html.Label("Color 3D markers by time:"),
    dcc.RadioItems(
        id='color-by-time-toggle',
        options=[
            {'label': 'Yes', 'value': 'yes'},
            {'label': 'No', 'value': 'no'}
        ],
        value='no',
        inline=True
    ),
    html.Label("Apply Cape CRS correction to catalogs (+433m N):"),
    dcc.RadioItems(
        id='apply-correction-toggle',
        options=[
            {'label': 'No', 'value': 'no'},
            {'label': 'Yes', 'value': 'yes'}
        ],
        value='no',
        inline=True
    ),
    html.Label("Show Newberry lithology planes:"),
    dcc.RadioItems(
        id='show-lith-toggle',
        options=[
            {'label': 'Off', 'value': 'no'},
            {'label': 'On',  'value': 'yes'}
        ],
        value='no',
        inline=True
    ),
    html.Label("Show NLLoc scatter clouds:"),
    dcc.RadioItems(
        id='show-scat-toggle',
        options=[
            {'label': 'Off', 'value': 'no'},
            {'label': 'On', 'value': 'yes'}
        ],
        value='no',
        inline=True
    ),
    dcc.Store(id='camera-store', data=None),
    dcc.Graph(id='3d-plot', figure=make_3d_figure(app_state['catalogs'], app_state['catalog_names'], app_state['datas'], field=app_state['field'], catalog_paths=app_state['catalog_paths']), config={'scrollZoom': True, 'displaylogo': False}),
    dcc.Graph(id='cumulative-plot', figure=make_cumulative_figure(app_state['catalogs'], app_state['catalog_names'], app_state['catalog_paths']), config={'displaylogo': False}),
    dcc.Graph(id='arrivals-hist', figure=make_arrivals_histogram(app_state['catalogs'], app_state['catalog_names'], app_state['catalog_paths']), config={'displaylogo': False}),
])


app.clientside_callback(
    """
    function(relayoutData, restyleData, currentCamera) {
        var ctx = window.dash_clientside.callback_context;
        var triggered = ctx && ctx.triggered && ctx.triggered.length > 0
                        ? ctx.triggered[0].prop_id : '';

        // Legend item clicked: restyleData fired. Restore saved camera after Plotly re-renders.
        if (triggered.indexOf('restyleData') !== -1) {
            if (currentCamera) {
                setTimeout(function() {
                    var el = document.getElementById('3d-plot');
                    if (el) { Plotly.relayout(el, {'scene.camera': currentCamera}); }
                }, 50);
            }
            return window.dash_clientside.no_update;
        }

        // Camera moved: save new position.
        if (relayoutData && relayoutData['scene.camera']) {
            return relayoutData['scene.camera'];
        }
        return window.dash_clientside.no_update;
    }
    """,
    dash.dependencies.Output('camera-store', 'data'),
    dash.dependencies.Input('3d-plot', 'relayoutData'),
    dash.dependencies.Input('3d-plot', 'restyleData'),
    dash.dependencies.State('camera-store', 'data'),
    prevent_initial_call=True,
)

@app.callback(
    dash.dependencies.Output('3d-plot', 'figure'),
    [
        dash.dependencies.Input('scale-mag-toggle', 'value'),
        dash.dependencies.Input('color-by-time-toggle', 'value'),
        dash.dependencies.Input('apply-correction-toggle', 'value'),
        dash.dependencies.Input('show-lith-toggle', 'value'),
        dash.dependencies.Input('show-scat-toggle', 'value'),
        dash.dependencies.Input('date-range-picker', 'start_date'),
        dash.dependencies.Input('date-range-picker', 'end_date'),
    ],
    [dash.dependencies.State('camera-store', 'data')],
    prevent_initial_call=True,
)
def update_3d_plot(scale_mag_value, color_by_time_value, apply_correction_value, show_lith_value, show_scat_value, start_date, end_date, camera_data):
    scale_by_magnitude = (scale_mag_value == 'yes')
    color_by_time = (color_by_time_value == 'yes')
    apply_cape_correction = (apply_correction_value == 'yes')
    show_lith = (show_lith_value == 'yes')
    show_scat = (show_scat_value == 'yes')
    cats = filter_catalogs_by_date(app_state['catalogs'], start_date, end_date) if (start_date and end_date) else app_state['catalogs']
    fig = make_3d_figure(
        cats, app_state['catalog_names'], app_state['datas'], field=app_state['field'],
        scale_by_magnitude=scale_by_magnitude,
        color_by_time=color_by_time,
        apply_cape_correction=apply_cape_correction,
        show_lith=show_lith,
        show_scat=show_scat,
        catalog_paths=app_state['catalog_paths'],
    )
    if camera_data:
        fig.update_layout(scene_camera=camera_data)
    return fig


@app.callback(
    dash.dependencies.Output('cumulative-plot', 'figure'),
    [
        dash.dependencies.Input('date-range-picker', 'start_date'),
        dash.dependencies.Input('date-range-picker', 'end_date'),
    ],
    prevent_initial_call=True,
)
def update_cumulative_plot(start_date, end_date):
    cats = filter_catalogs_by_date(app_state['catalogs'], start_date, end_date) if (start_date and end_date) else app_state['catalogs']
    return make_cumulative_figure(cats, app_state['catalog_names'], app_state['catalog_paths'])


@app.callback(
    dash.dependencies.Output('arrivals-hist', 'figure'),
    [
        dash.dependencies.Input('date-range-picker', 'start_date'),
        dash.dependencies.Input('date-range-picker', 'end_date'),
    ],
    prevent_initial_call=True,
)
def update_arrivals_hist(start_date, end_date):
    cats = filter_catalogs_by_date(app_state['catalogs'], start_date, end_date) if (start_date and end_date) else app_state['catalogs']
    return make_arrivals_histogram(cats, app_state['catalog_names'], app_state['catalog_paths'])


@app.callback(
    dash.dependencies.Output('event-count-label', 'children'),
    [
        dash.dependencies.Input('date-range-picker', 'start_date'),
        dash.dependencies.Input('date-range-picker', 'end_date'),
    ],
    prevent_initial_call=True,
)
def update_event_count(start_date, end_date):
    cats = filter_catalogs_by_date(app_state['catalogs'], start_date, end_date) if (start_date and end_date) else app_state['catalogs']
    parts = [f'{app_state["catalog_names"][i]}: {len(c)} events' for i, c in enumerate(cats)]
    return '  |  '.join(parts)


if __name__ == '__main__':
    if cli_args.export_html:
        _fig3d  = make_3d_figure(app_state['catalogs'], app_state['catalog_names'], app_state['datas'], field=app_state['field'], catalog_paths=app_state['catalog_paths'])
        _figcum = make_cumulative_figure(app_state['catalogs'], app_state['catalog_names'], app_state['catalog_paths'])
        _fighist = make_arrivals_histogram(app_state['catalogs'], app_state['catalog_names'], app_state['catalog_paths'])
        _html_parts = [
            '<!DOCTYPE html><html><head><meta charset="utf-8">',
            f'<title>Seismic Catalog Comparison – {dataset_key}</title></head><body>',
            f'<h1>Seismic Catalog Comparison – {dataset_key}</h1>',
            f'<p><em>Catalogs: {", ".join(catalog_names)}  |  '
            f'Total events: {", ".join(str(len(c)) for c in catalogs)}</em></p>',
            _fig3d.to_html(full_html=False, include_plotlyjs=True),
            _figcum.to_html(full_html=False, include_plotlyjs=False),
            _fighist.to_html(full_html=False, include_plotlyjs=False),
            '</body></html>',
        ]
        out_path = cli_args.export_html
        with open(out_path, 'w', encoding='utf-8') as _f:
            _f.write('\n'.join(_html_parts))
        logging.info('Exported static HTML to %s', out_path)
        print(f'Exported: {out_path}')
    else:
        app.run(host=cli_args.host, port=cli_args.port, debug=cli_args.debug)

