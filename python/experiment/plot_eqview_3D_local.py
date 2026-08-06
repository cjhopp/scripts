#!/home/chopp/miniconda3/envs/geo-plotting/bin/python

import sys
import glob
import os
import re
import json
import plotly
import pyproj
import fileinput
import logging

import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objs as go
import colorlover as cl

from itertools import cycle
from datetime import datetime
from obspy import read_events
from osgeo import ogr, osr, gdal
from shapely.geometry import Polygon, MultiLineString, LineString

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG, filename='log.txt')

data_directory = '/media/chopp/HDD1/chet-meq/cape_modern/spatial_data'

site_polygons = {
    'Newberry': Polygon([(-121.0736, 43.8988), (-121.0736, 43.5949), (-121.4918, 43.5949), (-121.4918, 43.8988)]),
    'JV': Polygon([(-117.40, 40.2357), (-117.5692, 40.2357), (-117.5692, 40.107), (-117.40, 40.107)]),
    'DAC': Polygon([(-118.1979, 38.9604), (-118.1979, 38.7943), (-118.4046, 38.7943), (-118.4046, 9604)]),
    'TM': Polygon([(-117.5956, 39.7353), (-117.5956, 39.6056), (-117.7649, 39.6056), (-117.7649, 39.7353)]),
    'Cape': Polygon([(-112.6924, 38.3912), (-112.6924, 38.6512), (-113.1358, 38.6512), (-113.1358, 38.3912)])
}

datasets = {
    'Newberry': ['{}/newberry/boreholes/Deviation_corrected.csv'.format(data_directory),
                 '{}/newberry/DEMs/USGS_13_merged_epsg-26910_just_edifice_very-coarse.tif'.format(data_directory)],
    'JV': {'wells': '/media/chopp/HDD1/chet-amplify/spatial_data/wells/Offset_Wells_Surveys_JV.csv'},
    'DAC': {'wells': '/media/chopp/HDD1/chet-amplify/spatial_data/wells/DAC/Offset_Wells_Surveys_DAC.csv'},
    'TM': [],
    'Cape': {'Topography': '{}/DEM/Cape-modern_Lidar_downsample.tif'.format(data_directory),
             'Delano': '{}/vector/boreholes/Delano_trajectory.csv'.format(data_directory),
             'Frisco-1': '{}/vector/boreholes/Frisco-1_trajectory.csv'.format(data_directory),
             'Frisco-2': '{}/vector/boreholes/Frisco-2_trajectory.csv'.format(data_directory),
             'Frisco-3': '{}/vector/boreholes/Frisco-3_trajectory.csv'.format(data_directory),
             'Frisco-4': '{}/vector/boreholes/Frisco-4_trajectory.csv'.format(data_directory),
             'Basement': '{}/vmods/ToB_50m_grid_3-1-24.nc'.format(data_directory),
             'Bearskin-1IA': '{}/vector/boreholes/Bearskin_1IA_trajectory.csv'.format(data_directory),
             'Bearskin-2IB': '{}/vector/boreholes/Bearskin_2IB_trajectory.csv'.format(data_directory),
            #  'Bearskin-3PA': '{}/vector/boreholes/Bearskin_3PA_trajectory.csv'.format(data_directory),
             'Bearskin-4PB': '{}/vector/boreholes/Bearskin_4PB_trajectory.csv'.format(data_directory),
            #  'Bearskin-5IA': '{}/vector/boreholes/Bearskin_5IA_trajectory.csv'.format(data_directory),
             'Bearskin-6IB': '{}/vector/boreholes/Bearskin_6IB_trajectory.csv'.format(data_directory),
             'Bearskin-7PA': '{}/vector/boreholes/Bearskin_7PA_trajectory.csv'.format(data_directory),
             'Bearskin-8IA': '{}/vector/boreholes/Bearskin_8IA_trajectory.csv'.format(data_directory),
             'Gold-1PB': '{}/vector/boreholes/Gold_1PB_trajectory.csv'.format(data_directory),
             'Gold-2IB': '{}/vector/boreholes/Gold_2IB_trajectory.csv'.format(data_directory),
             'Gold-3PA': '{}/vector/boreholes/Gold_3PA_trajectory.csv'.format(data_directory),
             'Gold-4PB': '{}/vector/boreholes/Gold_4PB_trajectory.csv'.format(data_directory),
            #  'Gold-5IA': '{}/vector/boreholes/Gold_5IA_trajectory.csv'.format(data_directory),
             'Gold-6IB': '{}/vector/boreholes/Gold_6IB_trajectory.csv'.format(data_directory),
             'Gold-7PA': '{}/vector/boreholes/Gold_7PA_trajectory.csv'.format(data_directory),
             'Gold-8PB': '{}/vector/boreholes/Gold_8PB_trajectory.csv'.format(data_directory),
             'Kings-1PB': '{}/vector/boreholes/Kings_1PB_trajectory.csv'.format(data_directory),}
}


projections = {'cape': pyproj.Proj("EPSG:26912"),
               'newberry': pyproj.Proj("EPSG:32610"),
               'jv': pyproj.Proj("EPSG:32611"),
               'dac': pyproj.Proj("EPSG:26911"),}


color_dict = {
    'jv': {
        ('14-34'): 'black',
        ('18A-27', '46-28', '14-27', '81-28', '81A-28'): 'steelblue',
        ('86-28', '87-28', '77A-28'): 'firebrick',
    },
    'dac': {
        ('68-1RD'): 'black',
        ('24-6', '24A-6', '26-6', '26A-6', '36-6', '24-6', '24A-6'): 'steelblue',
        ('64-11', '64A-11', '64B-11', '64C-11', '65-11', '65A-11', '85-11', '85A-11', '54-11', '54A-11'): 'firebrick',
    },
}

depth_correction = {
    'jv': 1446.,
    'dac': 1286.,
}

def read_stdin():
    return [ln for ln in fileinput.input()]


def get_selection_area(lines):
    line = lines[0].split(',')
    coords = []
    for part in line:
        coord = float(part.split('=')[-1])
        coords.append(coord)
    poly = Polygon([(coords[1], coords[0]), (coords[1], coords[2]), (coords[3], coords[2]), (coords[3], coords[0])])
    return poly


def get_events(lines):
    """
    Differs from the version on the seiscomp servers

    Pipe a scrtdd catalog (e.g.) into this file
    """
    events = []
    for ln in lines[1:]:
        events.append(ln.split(',')[:6])  # Seems like the only difference between scrtdd output and GAPS is delimiter
    for e in events:
        e[1] = datetime.strptime(e[1], '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()  # Also different time format
        e[2] = float(e[2])
        e[3] = float(e[3])
        try:
            e[4] = float(e[4])
        except ValueError:
            e[4] = 0.  # Case of no depth
        try:
            e[5] = float(e[5])
        except ValueError:
            e[5] = 1.
        # e[-2] = float(e[-2])
    return events


def check_if_in_field(selection):
    """
    Check if the selected region falls into any of the geothermal fields. If so, return list of datasets
    :return:
    """
    for name, site_poly in site_polygons.items():
        if selection.intersects(site_poly):
            return datasets[name]
    return []


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


def expand_catalog_paths(catalog_args):
    paths = []
    labels = []
    label_pattern = re.compile(r'(lbnl202.....)')
    for arg in catalog_args:
        matches = sorted(glob.glob(arg))
        if matches:
            for match in matches:
                paths.append(match)
                match_label = label_pattern.search(match)
                if match_label:
                    labels.append(match_label.group(1))
                else:
                    labels.append(os.path.splitext(os.path.basename(match))[0])
        elif os.path.exists(arg):
            paths.append(arg)
            match_label = label_pattern.search(arg)
            if match_label:
                labels.append(match_label.group(1))
            else:
                labels.append(os.path.splitext(os.path.basename(arg))[0])
        else:
            logging.warning('Catalog path did not match: %s', arg)
    return paths, labels


def read_trajectory_csv(path):
    """Read trajectory CSV with or without a header row."""
    arr = np.genfromtxt(path, delimiter=',')
    if arr.ndim == 1:
        arr = np.atleast_2d(arr)
    # Drop non-numeric rows (e.g., header rows parsed as NaN).
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.shape[1] < 3:
        raise ValueError('Trajectory CSV must have at least 3 numeric columns: {}'.format(path))
    return arr[:, :3]


def _get_linestring_coords(geometry):
    gtype = geometry.get('type')
    coords = geometry.get('coordinates', [])
    if gtype == 'LineString':
        return coords
    if gtype == 'MultiLineString' and len(coords) > 0:
        return coords[0]
    return []


def _is_plausible_cape_utm(east, north):
    """Basic plausibility bounds for Cape UTM coordinates (EPSG:26912)."""
    if len(east) == 0 or len(north) == 0:
        return False
    med_e = float(np.nanmedian(east))
    med_n = float(np.nanmedian(north))
    return 300000.0 <= med_e <= 400000.0 and 4200000.0 <= med_n <= 4300000.0


def load_cape_geojson_xy(label, trajectory_path):
    """Load Cape well XY from GeoJSON when available.

    This avoids mixed export offsets in some trajectory CSV products.
    """
    stem = os.path.basename(trajectory_path).replace('_trajectory.csv', '')
    aliases = {
        'Bearskin-6IB': 'Bearskin_6IA',
        'Bearskin_6IB': 'Bearskin_6IA',
    }
    candidates = []
    for key in {label, label.replace('-', '_'), stem, aliases.get(label), aliases.get(stem)}:
        if key is None:
            continue
        candidates.append('{}/vector/{}.geojson'.format(data_directory, key))
        candidates.append('{}/vector/{}_latlon.geojson'.format(data_directory, key))
    logging.debug('GeoJSON candidates for %s: %s', label, candidates)
    for path in candidates:
        if not os.path.exists(path):
            logging.debug('GeoJSON not found: %s', path)
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                gj = json.load(f)
            features = gj.get('features', [])
            if len(features) == 0:
                logging.debug('GeoJSON has no features: %s', path)
                continue
            # Support both a single LineString/MultiLineString and a
            # FeatureCollection of Point features (e.g. survey stations).
            first_type = features[0].get('geometry', {}).get('type')
            if first_type == 'Point':
                coords = [f['geometry']['coordinates'] for f in features
                          if f.get('geometry', {}).get('type') == 'Point']
            else:
                coords = _get_linestring_coords(features[0].get('geometry', {}))
            if len(coords) == 0:
                logging.debug('GeoJSON has no usable coordinates: %s', path)
                continue
            xy = np.array(coords, dtype=float)
            x = xy[:, 0]
            y = xy[:, 1]
            # GeoJSON products here are lon/lat, so convert to Cape UTM if needed.
            if np.nanmax(np.abs(x)) <= 180 and np.nanmax(np.abs(y)) <= 90:
                transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:26912', always_xy=True)
                x, y = transformer.transform(x, y)
            x = np.array(x)
            y = np.array(y)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                logging.warning('Skipping non-finite GeoJSON XY for %s from %s', label, path)
                continue
            if not _is_plausible_cape_utm(x, y):
                logging.warning('Skipping implausible Cape GeoJSON XY for %s from %s', label, path)
                continue
            logging.info('Loaded GeoJSON XY for %s from %s: median E=%.1f N=%.1f',
                         label, path, float(np.nanmedian(x)), float(np.nanmedian(y)))
            return x, y
        except Exception as e:
            logging.warning('Failed to parse well GeoJSON %s: %s', path, e)
    return None, None


def estimate_cape_shared_offset(datasets):
    """Estimate shared (dE, dN) offset between trustworthy GeoJSON XY and CSV XY.

    Uses robust median per well, then averages offsets with meaningful magnitude.
    """
    offsets = []
    for label, data in datasets.items():
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

    # Prefer non-trivial offsets so Frisco-like ~0 shifts do not dilute Gold/Bearskin correction.
    nontrivial = [(de, dn) for de, dn, _ in offsets if np.hypot(de, dn) > 50.0]
    pool = nontrivial if len(nontrivial) > 0 else [(de, dn) for de, dn, _ in offsets]
    dE = float(np.nanmean([de for de, _ in pool]))
    dN = float(np.nanmean([dn for _, dn in pool]))
    logging.info('Estimated Cape shared CSV offset from %d wells: dE=%.3f dN=%.3f', len(pool), dE, dN)
    return dE, dN


def plot_3D(datasets, catalogs, field, catalog_labels=None, use_time_color=True):
    """
    Make plotly html of selected earthquakes

    :param datasets: List of paths to included datasets
    :param catalog: Catalog of seismicity

    :return:
    """
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

    try:
        utm = projections[field.lower()]
    except KeyError:
        return
    cape_offset_e = 0.0
    cape_offset_n = 0.0
    if field.lower() == 'cape':
        cape_offset_e, cape_offset_n = estimate_cape_shared_offset(datasets)
    for label, data in datasets.items():
        if not data.endswith(('tif', 'nc')):
            if data.endswith('JV.csv'):
                wells = pd.read_csv(data)
                # Get unique well IDs and iterate over them
                for well_id in wells['Well_ID'].unique():
                    well_data = wells[wells['Well_ID'] == well_id]
                    # Process each well's data here
                    east = well_data['X'].values
                    north = well_data['Y'].values
                    dep_m = well_data['Z'].values
                    try:
                        col = [col for tup, col in color_dict['jv'].items() if well_id in tup][0]
                    except IndexError:
                        col = 'gray'  # Default color if no match found
                    objects.append(go.Scatter3d(x=east,
                                                y=north,
                                                z=dep_m,
                                                name='Well {}'.format(well_id),
                                                mode='lines',
                                                line=dict(color=col, width=6),
                                                hoverinfo='skip'),
                                    )
                    well_x.append(np.asarray(east))
                    well_y.append(np.asarray(north))
                    well_z.append(np.asarray(dep_m))
                    update_bounds(east, north, dep_m)
            else:
                # Add objects
                wellpath = read_trajectory_csv(data)
                east = wellpath[:, 0]
                north = wellpath[:, 1]
                dep_m = wellpath[:, 2]
                if field.lower() == 'cape':
                    geo_east, geo_north = load_cape_geojson_xy(label, data)
                    if geo_east is not None and geo_north is not None:
                        npts = min(len(dep_m), len(geo_east), len(geo_north))
                        east = geo_east[:npts]
                        north = geo_north[:npts]
                        dep_m = dep_m[:npts]
                    else:
                        # Fallback for wells without trustworthy GeoJSON: apply shared Cape offset.
                        east = east + cape_offset_e
                        north = north + cape_offset_n
                objects.append(go.Scatter3d(x=east,
                                            y=north,
                                            z=dep_m,
                                            name=label,
                                            mode='lines',
                                            line=dict(color='black', width=6),
                                            hoverinfo='skip'),
                                            )
                well_x.append(np.asarray(east))
                well_y.append(np.asarray(north))
                well_z.append(np.asarray(dep_m))
                update_bounds(east, north, dep_m)
        elif data.endswith('tif'):
            topo = gdal.Open(data, gdal.GA_ReadOnly)
            x, y, band = get_pixel_coords(topo)
            X, Y = np.meshgrid(x, y, indexing='xy')
            raster_values = band.ReadAsArray()
            topo_mesh = go.Mesh3d(x=X.flatten(), y=Y.flatten(),
                                  z=raster_values.flatten(), name=label, color='gray',
                                  opacity=0.3, delaunayaxis='z', showlegend=True,
                                  hoverinfo='skip')
            objects.append(topo_mesh)
            update_bounds(X.flatten(), Y.flatten(), raster_values.flatten())
        elif data.endswith('nc'):
            tob = xr.load_dataarray(data)
            tob = tob.interp(easting=tob.easting[::10], northing=tob.northing[::10])
            X, Y = np.meshgrid(tob.easting, tob.northing, indexing='xy')
            Z = tob.values.flatten()
            tob_mesh = go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z,
                                 name=label, color='gray', opacity=0.5, delaunayaxis='z', showlegend=True,
                                 hoverinfo='skip')
            objects.append(tob_mesh)
            update_bounds(X.flatten(), Y.flatten(), Z)
    mfact = 2.5  # Magnitude scaling factor
    # Add arrays to the plotly objects
    if not catalog_labels:
        catalog_labels = ['Catalog {}'.format(i + 1) for i in range(len(catalogs))]
    palette = cl.scales.get('9', {}).get('qual', {}).get('Set1', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    color_cycle = cycle(palette)
    for i, catalog in enumerate(catalogs):
        label = catalog_labels[i] if i < len(catalog_labels) else 'Catalog {}'.format(i + 1)
        color = next(color_cycle)
        try:
            # id, t, lat, lon, depth, m, agency, status, phases, geo, _, _, _, _, _ = zip(*catalog)
            id, t, lat, lon, depth, m = zip(*catalog)
        except ValueError:  # When passing an obspy Catalog
            params = []
            if len(catalog.events) == 0:
                continue
            for ev in catalog:
                ev.origins.sort(key=lambda o: o.time)  # Sort origins by time to ensure consistent selection of preferred origin
                o = ev.preferred_origin()
                if o is None:
                    try:
                        o = ev.origins[0]
                    except IndexError:
                        continue
                try:
                    m = ev.preferred_magnitude().mag
                except AttributeError:
                    m = 1.
                params.append([ev.resource_id.id, o.time.timestamp, o.latitude, o.longitude, o.depth, m])
            params = np.array(params)
            id, t, lat, lon, depth, m = np.split(params, 6, axis=1)
            id = id.flatten()
            t = t.astype('f').flatten()
            lat = lat.astype('f').flatten()
            lon = lon.astype('f').flatten()
            depth = depth.astype('f').flatten()
            m = m.astype('f').flatten()
        if use_time_color:
            tickvals = np.linspace(min(t), max(t), 10)
            ticktext = [datetime.fromtimestamp(int(t)).strftime('%d %b %Y: %H:%M')
                        for t in tickvals]
        ev_east, ev_north = utm(lon, lat)
        depth = np.array(depth) * -1#000
        m_arr = np.array(m).astype(float)
        m_min, m_max = float(np.min(m_arr)), float(np.max(m_arr))
        if m_min == m_max:
            marker_size = np.full_like(m_arr, 8.0)
        else:
            # Radius is linear in magnitude; plotly size = r^2, so marker area
            # grows quadratically per magnitude unit and gives clear visual spread.
            r = np.interp(m_arr, [m_min, m_max], [0.15, 4.])
            marker_size = r ** 2.5
        if use_time_color:
            marker = dict(color=t,
                          cmin=min(tickvals),
                          cmax=max(tickvals),
                          size=marker_size,
                          symbol='circle',
                          line=dict(color=t,
                                    width=1,
                                    colorscale='Cividis'),
                          colorbar=dict(
                              title=dict(text='Timestamp',
                                         font=dict(size=18)),
                              x=-0.2,
                              ticktext=ticktext,
                              tickvals=tickvals),
                          colorscale='Bluered',
                          opacity=0.5)
        else:
            marker = dict(color=color,
                          size=marker_size,
                          symbol='circle',
                          line=dict(color=color,
                                    width=1),
                          opacity=0.4)
        scat_obj = go.Scatter3d(x=ev_east, y=ev_north, z=depth,
                                mode='markers',
                                name=label,
                                hoverinfo='text',
                                text=np.array(id),
                                marker=marker)
        objects.append(scat_obj)
        update_bounds(ev_east, ev_north, depth)
    # Start figure
    fig = go.Figure(data=objects)
    x_range = None
    y_range = None
    z_range = None
    if len(well_x) > 0 and len(well_y) > 0 and len(well_z) > 0:
        all_x = np.concatenate(well_x)
        all_y = np.concatenate(well_y)
        all_z = np.concatenate(well_z)
        cx = float(np.nanmean(all_x))
        cy = float(np.nanmean(all_y))
        cz = float(np.nanmean(all_z))
        # Keep center at wells, but span full plotted data (wells + seismicity + surfaces).
        if np.isfinite(data_xmin) and np.isfinite(data_xmax):
            half_span = max(
                cx - data_xmin, data_xmax - cx,
                cy - data_ymin, data_ymax - cy,
                cz - data_zmin, data_zmax - cz,
                1.
            ) * 1.02
            x_range = [cx - half_span, cx + half_span]
            y_range = [cy - half_span, cy + half_span]
            z_range = [cz - half_span, cz + half_span]
    xax = go.layout.scene.XAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Easting (m)',
                                showline=True, mirror=True,
                                linecolor='black', linewidth=2.,
                                range=x_range)
    yax = go.layout.scene.YAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Northing (m)',
                                showline=True, mirror=True,
                                linecolor='black', linewidth=2.,
                                range=y_range)
    zax = go.layout.scene.ZAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Elevation (m)',
                                range=z_range)
    aspectratio = dict(x=1, y=1, z=1)
    if x_range is not None and y_range is not None and z_range is not None:
        x_span = max(float(x_range[1] - x_range[0]), 1.)
        y_span = max(float(y_range[1] - y_range[0]), 1.)
        z_span = max(float(z_range[1] - z_range[0]), 1.)
        scale = max(x_span, y_span, z_span)
        aspectratio = dict(x=x_span / scale, y=y_span / scale, z=z_span / scale)
    layout = go.Layout(scene=dict(xaxis=xax, yaxis=yax, zaxis=zax,
                                  xaxis_showspikes=False,
                                  yaxis_showspikes=False,
                                  aspectmode='manual',
                                  aspectratio=aspectratio,
                                  bgcolor="rgb(244, 244, 248)"),
                       # autosize=True,
                       title='3D Seismicity',
                       legend=dict(traceorder='normal',
                                   itemsizing='constant',
                                   font=dict(
                                       family="sans-serif",
                                       size=14,
                                       color="black"),
                                   bgcolor='whitesmoke',
                                   bordercolor='gray',
                                   borderwidth=1,
                                   tracegroupgap=3))
    fig.update_layout(layout)
    return fig


if __name__ in '__main__':
    lines = sys.argv
    print(lines)
    # bbox = get_selection_area(lines)
    # catalog = get_events(lines)
    catalog_paths, catalog_labels = expand_catalog_paths(lines[1:-1])
    catalogs = [read_events(path) for path in catalog_paths]
    datas = datasets[lines[-1]]
    fig = plot_3D(datas, catalogs, lines[-1], catalog_labels=catalog_labels,
                  use_time_color=len(catalogs) == 1)
    html = plotly.io.to_html(fig)
    fig.write_html('eqview_3d_compare.html')
    # fig.write_html('output.html')
    # sys.stdout.write(html)
