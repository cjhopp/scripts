#!/home/chopp/miniconda3/envs/geo-plotting/bin/python

import sys
import plotly
import pyproj
import fileinput
import logging

import numpy as np
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

data_directory = '/home/chopp/spatial_data'

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
    'JV': [],
    'DAC': [],
    'TM': [],
    'Cape': ['{}/DEM/Cape-modern_Lidar_downsample.tif'.format(data_directory),
             '{}/Cape_share/Frisco-1_trajectory.csv'.format(data_directory),
             '{}/Cape_share/Frisco-2_trajectory.csv'.format(data_directory),
             '{}/Cape_share/Frisco-3_trajectory.csv'.format(data_directory),
             '{}/Cape_share/Frisco-4_trajectory.csv'.format(data_directory),
             '{}/vmods/ToB_50m_grid_3-1-24.nc'.format(data_directory)]
}

projections = {'cape': pyproj.Proj("EPSG:26912"),
               'newberry': pyproj.Proj("EPSG:32610")}

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


def plot_3D(datasets, catalog):
    """
    Make plotly html of selected earthquakes

    :param datasets: List of paths to included datasets
    :param catalog: Catalog of seismicity

    :return:
    """
    objects = []
    # What field is this?
    # field = datasets[0].split('/')[-3]
    field = 'newberry'
    try:
        utm = projections[field]
    except KeyError:
        return
    for data in datasets:
        if not data.endswith(('tif', 'nc')):
            # Add objects
            wellpath = np.loadtxt(data, delimiter=',', skiprows=1)
            east = wellpath[:, 0]
            north = wellpath[:, 1]
            dep_m = wellpath[:, 2]
            objects.append(go.Scatter3d(x=east,
                                        y=north,
                                        z=dep_m,
                                        mode='lines',
                                        line=dict(color='black', width=6),
                                        hoverinfo='skip'))
        elif data.endswith('tif'):
            topo = gdal.Open(data, gdal.GA_ReadOnly)
            x, y, band = get_pixel_coords(topo)
            X, Y = np.meshgrid(x, y, indexing='xy')
            raster_values = band.ReadAsArray()
            topo_mesh = go.Mesh3d(x=X.flatten(), y=Y.flatten(),
                                  z=raster_values.flatten(), name='Topography', color='gray',
                                  opacity=0.3, delaunayaxis='z', showlegend=True,
                                  hoverinfo='skip')
            objects.append(topo_mesh)
        elif data.endswith('nc'):
            tob = xr.load_dataarray(data)
            tob = tob.interp(easting=tob.easting[::10], northing=tob.northing[::10])
            X, Y = np.meshgrid(tob.easting, tob.northing, indexing='xy')
            Z = tob.values.flatten()
            tob_mesh = go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z,
                                 name='Basement', color='gray', opacity=0.5, delaunayaxis='z', showlegend=True,
                                 hoverinfo='skip')
            objects.append(tob_mesh)
    mfact = 2.5  # Magnitude scaling factor
    # Add arrays to the plotly objects
    try:
        # id, t, lat, lon, depth, m, agency, status, phases, geo, _, _, _, _, _ = zip(*catalog)
        id, t, lat, lon, depth, m = zip(*catalog)
    except ValueError:  # When passing an obspy Catalog
        params = []
        for ev in catalog:
            o = ev.preferred_origin()
            try:
                m = ev.preferred_magnitude().mag
            except AttributeError:
                m = 0.5
            params.append([ev.resource_id.id, o.time.timestamp, o.latitude, o.longitude, o.depth, m])
        params = np.array(params)
        id, t, lat, lon, depth, m = np.split(params, 6, axis=1)
        t = t.astype('f').flatten()
        lat = lat.astype('f').flatten()
        lon = lon.astype('f').flatten()
        depth = depth.astype('f').flatten()
        m = m.astype('f').flatten()
    print(t)
    tickvals = np.linspace(min(t), max(t), 10)
    ticktext = [datetime.fromtimestamp(t).strftime('%d %b %Y: %H:%M')
                for t in tickvals]
    ev_east, ev_north = utm(lon, lat)
    depth = np.array(depth) * -1#000
    scat_obj = go.Scatter3d(x=ev_east, y=ev_north, z=depth,
                            mode='markers',
                            name='Seismic event',
                            hoverinfo='text',
                            text=np.array(id),
                            marker=dict(color=t,
                                        cmin=min(tickvals),
                                        cmax=max(tickvals),
                                        size=(mfact * np.array(m)) ** 2,
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
                                        opacity=0.5))
    objects.append(scat_obj)
    # Start figure
    fig = go.Figure(data=objects)
    xax = go.layout.scene.XAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Easting (m)',
                                showline=True, mirror=True,
                                linecolor='black', linewidth=2.)
    yax = go.layout.scene.YAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Northing (m)',
                                showline=True, mirror=True,
                                linecolor='black', linewidth=2.)
    zax = go.layout.scene.ZAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Elevation (m)')
    layout = go.Layout(scene=dict(xaxis=xax, yaxis=yax, zaxis=zax,
                                  xaxis_showspikes=False,
                                  yaxis_showspikes=False,
                                  aspectmode='data',
                                  aspectratio=dict(x=1, y=1, z=1.),
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
    catalog = read_events(lines[1])
    datas = datasets[lines[2]]
    fig = plot_3D(datas, catalog)
    html = plotly.io.to_html(fig)
    fig.write_html('eqview_3d_compare.html')
    # fig.write_html('output.html')
    # sys.stdout.write(html)
