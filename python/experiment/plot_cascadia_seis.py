#!/usr/bin/python

import sys
import plotly

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objs as go
import colorlover as cl

from itertools import cycle
from datetime import datetime
from obspy import read_events
from shapely.geometry import Polygon, MultiLineString, LineString


def read_slab_model(slab_mod_path):
    # Helper to read in slab model for cascadia and return (x, 3) ndarray
    slab_grd = []
    with open(slab_mod_path, 'r') as f:
        next(f)
        for ln in f:
            line = ln.strip()
            line = line.split(',')
            slab_grd.append((float(line[0]), float(line[1]), float(line[2])))
    return np.array(slab_grd)


def add_catalog(catalog):
    # UTM Zone 10N
    crs = ccrs.UTM(10)
    # Establish color scales from colorlover (import colorlover as cl)
    colors = cycle(cl.scales['11']['qual']['Paired'])
    lats = np.array([ev.preferred_origin().latitude for ev in catalog])
    lons = np.array([ev.preferred_origin().longitude for ev in catalog])
    transforms = crs.transform_points(ccrs.Geodetic(), lons, lats)
    easts = transforms[:, 0]
    norths = transforms[:, 1]
    depths = [-ev.preferred_origin().depth for ev in catalog]
    mags = [ev.magnitudes[-1].mag for ev in catalog]
    times = [ev.preferred_origin().time.datetime.timestamp() for ev in catalog]
    eids = [ev.resource_id.id for ev in catalog]
    tickvals = np.linspace(min(times), max(times), 10)
    ticktext = [datetime.fromtimestamp(t).strftime('%Y-%m-%d')
                for t in tickvals]
    scat_obj = go.Scatter3d(x=easts, y=norths, z=np.array(depths),
                            mode='markers',
                            name='Seismic event',
                            hoverinfo='text',
                            text=eids,
                            marker=dict(color=times,
                                        cmin=min(tickvals),
                                        cmax=max(tickvals),
                                        size=(1.5 * np.array(mags)) ** 2,
                                        symbol='circle',
                                        line=dict(color=times,
                                                  width=1,
                                                  colorscale='Cividis'),
                                        colorbar=dict(
                                            title=dict(text='Timestamp',
                                                       font=dict(size=18)),
                                            x=-0.2,
                                            ticktext=ticktext,
                                            tickvals=tickvals),
                                        colorscale='Cividis',
                                        opacity=0.5))
    return scat_obj


def add_coastlines():
    """Helper to add coastlines at zero depth"""
    polys = []
    # UTM Zone 10N
    crs = ccrs.UTM(10)
    # Clipping polygon
    box = Polygon([[-126.5, 46.5], [-126.5, 50.],
                   [-121.5, 50.], [-121.5, 46.5]])
    coasts = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    for geo in coasts.geometries():
        clipped = geo.intersection(box)
        if type(clipped) == MultiLineString:
            for line in clipped.geoms:
                coords = np.array(line.coords)
                if len(coords) == 0:
                    continue
                pts = crs.transform_points(ccrs.Geodetic(), coords[:, 0],
                                           coords[:, 1])
                easts = pts[:, 0]
                norths = pts[:, 1]
                z = np.array([0. for x in easts])
                coast = go.Scatter3d(
                    x=easts, y=norths, z=z,
                    marker=dict(color='black', line=dict(color='black')),
                    name='Coastlines',
                    mode='lines',
                    opacity=1.,
                    showlegend=False)
                polys.append(coast)
        elif type(clipped) == LineString:
            coords = np.array(clipped.coords)
            if len(coords) == 0:
                continue
            pts = crs.transform_points(ccrs.Geodetic(), coords[:, 0],
                                       coords[:, 1])
            easts = pts[:, 0]
            norths = pts[:, 1]
            z = np.array([0. for x in easts])
            coast = go.Scatter3d(
                x=easts, y=norths, z=z,
                marker=dict(color='black', line=dict(color='black')),
                name='Coastlines',
                mode='lines',
                opacity=1.,
                showlegend=False)
            polys.append(coast)
    return polys


def plot_cascadia_3D(slab_file, catalog, outfile):
    """
    Plot Cascadia locations in 3D with slab model and coastlines

    :param slab_mod: Path to slab model file
    :param catalog: Catalog of seismicity

    :return:
    """
    # UTM Zone 10N
    crs = ccrs.UTM(10)
    # Plot rough slab interface
    slab_grd = read_slab_model(slab_file)
    pts_trans = crs.transform_points(ccrs.Geodetic(), slab_grd[:, 0],
                                     slab_grd[:, 1])
    slab = go.Mesh3d(x=pts_trans[:, 0], y=pts_trans[:, 1],
                     z=slab_grd[:, 2] * 1000,
                     name='Slab model', color='gray', opacity=0.15,
                     delaunayaxis='z', showlegend=True, hoverinfo='skip')
    cat = add_catalog(catalog)
    # Map limits are catalog extents
    lims_x = (np.min(cat['x']), np.max(cat['x']))
    lims_y = (np.min(cat['y']), np.max(cat['y']))
    # Add cartopy coastlines
    coasts = add_coastlines()
    data = [cat, slab]
    data.extend(coasts)
    # Start figure
    fig = go.Figure(data=data)
    xax = go.layout.scene.XAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Easting (m)',
                                range=lims_x, showline=True, mirror=True,
                                linecolor='black', linewidth=2.)
    yax = go.layout.scene.YAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Northing (m)',
                                range=lims_y, showline=True, mirror=True,
                                linecolor='black', linewidth=2.)
    zax = go.layout.scene.ZAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Elevation (m)')
    layout = go.Layout(scene=dict(xaxis=xax, yaxis=yax, zaxis=zax,
                                  xaxis_showspikes=False,
                                  yaxis_showspikes=False,
                                  aspectmode='manual',
                                  aspectratio=dict(x=1, y=1, z=1.),
                                  bgcolor="rgb(244, 244, 248)"),
                       autosize=True,
                       # title=title,
                       legend=dict(title=dict(text='Legend',
                                              font=dict(size=18)),
                                   traceorder='normal',
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
    plotly.offline.iplot(fig, filename='{}.html'.format(outfile))
    return


if __name__ == '__main__':
    slab, cat = sys.argv[-2:]
    catalog = read_events(cat)
    plot_cascadia_3D(slab, catalog, outfile='test.html')
