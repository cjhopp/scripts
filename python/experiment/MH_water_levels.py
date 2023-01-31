#!/usr/bin/python

"""
Script to pull down Michigan-Huron water level data and plot to html.

Intended to be posted on website
"""
import json

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import plotly.express as px
import plotly.graph_objs as go

from pyproj import Proj
from glob import glob
from datetime import datetime
from geojsoncontour import contourf_to_geojson


'''
NSIDC ftp:
ftp://sidads.colorado.edu/DATASETS/NOAA/G10029/ascii
'''

## Reading/parsing funcs ##

def read_waterlevels(path):
    df = pd.read_csv(path, header=2, index_col='year')
    start = datetime(year=1918, month=1, day=1)
    end_yr = df.index.values[-1]
    # Whats the current (last) month in the data? Future are NaN
    end_month = df.iloc[-1].last_valid_index()
    end = datetime.strptime('{}-{}'.format(end_yr, end_month), '%Y-%b')
    months = pd.date_range(start, end, freq='BM')
    levels = df.values.flatten()
    return months, levels


def read_ice_grid(path):
    """Read in ASCII ice cover grids to xarray"""
    grid = xr.open_rasterio(path).squeeze()
    # Deproject coords
    proj = Proj('+proj=merc +lon_0=0 +k=1 +x_0=0 ' +
                '+y_0=-24 +datum=WGS84 +units=m +no_defs')
    lon, lat = proj(grid.coords['x'].values, grid.coords['y'].values,
                    inverse=True)
    grid = grid.assign_coords(x=lon, y=lat)
    return grid


def read_shorelines(dir):
    """Read individual lake shorelines"""
    shps = glob('{}/**/*.shp'.format(dir), recursive=True)
    df = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in shps]))
    return df


def contour_ice(dataarray):
    """Contour xarray dataarray of ice cover to geojson"""
    contours = dataarray.plot.contourf(vmin=0, vmax=100, levels=5)
    ice_json = json.loads(contourf_to_geojson(contourf=contours))
    print(ice_json['type'])
    # Remove land polygons
    rm_feats = [f for f in ice_json['features']
                if f['properties']['title'] == '<0.00 ']
    for rm in rm_feats:
        ice_json['features'].remove(rm)
    print([np.array(f['geometry']['coordinates']).shape
           for f in ice_json['features']])
    geodf_ice = gpd.GeoDataFrame.from_features(ice_json)
    return geodf_ice


## Plotting funcs ##

def plot_ice_map(ice_path, shore_dir):
    """
    Plot contours of ice thicknesses on Great Lakes

    :param ice_path: Path to .ct file
    :param shore_dir: Path to directory of shorline vectors

    :return:
    """
    # Get date from title of ice grid
    date = datetime.strptime(ice_path[-11:-3], '%Y%m%d').date()
    ice_array = read_ice_grid(ice_path)
    geodf_ice = contour_ice(ice_array)
    shores = read_shorelines(shore_dir)
    fig = px.choropleth(
        geodf_ice, geojson=geodf_ice.geometry, locations=geodf_ice.index,
        color='fill', title='Ice Cover: {}'.format(date))
    for i, shore in shores.iterrows():
        lon, lat = shore.geometry.exterior.coords.xy
        fig.add_trace(go.Scattergeo(lon=np.array(lon), lat=np.array(lat),
                                    mode='lines', fill='toself',
                                    fillcolor='rgba(211, 211, 211, 0.0)',
                                    line=dict(color='rgba(100, 100, 100, 0.3)'),
                                    showlegend=False))
    fig.update_geos(fitbounds='locations', projection_type='mercator',
                    visible=False)
    fig.update_layout(#margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      title_text='Ice Cover: {}'.format(date))
    fig.data = fig.data[::-1]
    fig.show()
    return


def plot_waterlevels(path):
    """
    Make plotly figure of the water levels (and eventually other stuff)

    :return:
    """
    fig = go.Figure()
    # Read the file
    months, levels = read_waterlevels(path)
    fig.add_trace(go.Scatter(x=months, y=levels, name='Monthly Avg. Level',
                             line=dict(color='navy', width=1.),
                             mode='lines'))
    fig.add_trace(go.Scatter(x=[months[0], months[-1]],
                             y=[np.nanmean(levels), np.nanmean(levels)],
                             name='Avg. Level 1918-present',
                             line=dict(color='red'),
                             mode='lines'))
    fig.update_layout(template='plotly',
                      xaxis=dict(
                          rangeselector=dict(
                              buttons=list([
                                  dict(count=5,
                                       label="5y",
                                       step="year",
                                       stepmode="backward"),
                                  dict(count=10,
                                       label="10y",
                                       step="year",
                                       stepmode="backward"),
                                  dict(count=30,
                                       label="30y",
                                       step="year",
                                       stepmode="backward"),
                                  dict(count=50,
                                       label="50y",
                                       step="year",
                                       stepmode="backward"),
                                  dict(step="all"),
                              ])
                          ),
                          rangeslider=dict(
                              visible=True
                          ),
                          type="date"
                      )
                      )
    fig.show(renderer='firefox')
    return