#!/usr/bin/python

"""
Plotting functions for the lbnl module
"""
import obspy

try:
    import dxfgrabber
except ImportError:
    print('Dont plot SURF dxf, dxfgrabber not installed')
import plotly
import rasterio
import fiona
import pickle
import trimesh
import pyproj

import numpy as np
import colorlover as cl
import seaborn as sns
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cf
import chart_studio.plotly as py
import plotly.graph_objs as go
import shapely.geometry as geometry
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import earthpy.spatial as es

from rasterio import mask as msk
from rasterio.merge import merge
from rasterio.plot import plotting_extent
from shapely.geometry import mapping
from shapely.geometry.point import Point
from descartes import PolygonPatch
from itertools import cycle
from glob import glob
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from scipy.io import loadmat
from scipy.linalg import lstsq, norm
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from obspy.imaging.beachball import beach
from cartopy.mpl.geoaxes import GeoAxes
from obspy import Trace, Catalog
from plotly.figure_factory import create_gantt
from matplotlib import animation, cm
from matplotlib.dates import DayLocator, HourLocator
from matplotlib.patches import Circle
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.gridspec import GridSpec
from matplotlib.dates import date2num, num2date
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap

# Local imports (assumed to be in python path)
from lbnl.boreholes import (parse_surf_boreholes, create_FSB_boreholes,
                            structures_to_planes, depth_to_xyz,
                            distance_to_borehole, make_4100_boreholes)
from lbnl.coordinates import SURF_converter
try:
    from lbnl.DSS import (interpolate_picks, extract_channel_timeseries,
                          get_frac_piercepoint, extract_strains, fault_depths)
except ModuleNotFoundError:
    print('Error on DSS import. Change env')


csd_well_colors = {'D1': 'blue', 'D2': 'blue', 'D3': 'green',
                   'D4': 'green', 'D5': 'green', 'D6': 'green', 'D7': 'black'}

fsb_well_colors = {'B1': 'k', 'B2': 'steelblue', 'B3': 'goldenrod',
                   'B4': 'goldenrod', 'B5': 'goldenrod', 'B6': 'goldenrod',
                   'B7': 'goldenrod', 'B8': 'firebrick', 'B9': 'firebrick',
                   'B10': 'k'}

cols_4850 = {'PDT': 'black', 'PDB': 'black', 'PST': 'black', 'PSB': 'black',
             'OT': 'black', 'OB': 'black', 'I': '#4682B4', 'P': '#B22222',
             'DMU': 'black', 'DML': 'black', 'AMU': 'black', 'AML': 'black',
             'TS': '#4682B4', 'TN': '#4682B4', 'TC': '#4682B4', 'TU': '#4682B4',
             'TL': '#4682B4'}

collab_4100_zone_depths = {
    'TC': [148.95, 156.85, 164.75, 172.65, 180.55, 188.45, 196.35, 204.25,
           212.15],
    'TU': [178.5]
}

# Mont Terri fault depths by borehole
fault_depths = {'D1': (14.34, 19.63), 'D2': (11.04, 16.39), 'D3': (17.98, 20.58),
                'D4': (27.05, 28.44), 'D5': (19.74, 22.66), 'D6': (28.5, 31.4),
                'D7': (22.46, 25.54), 'B2': (41.25, 45.65), 'B1': (34.8, 42.25),
                'B9': (55.7, 55.7), 'B10': (17.75, 21.7), '1': (38.15, 45.15),
                '2': (44.23, 49.62), '3': (38.62, 43.39)}

dts_hits_4100 = {
    'AMU': [34.0, 54.9],
    'AML': [45.5],
    'DMU': [18.5, 31.0, 43.0],
    'DML': [42.9, 43.9, 7.1, 14.9, 6.7, 22.3, 19.9]
}

dts_cooling_4100 = {
    'AMU': [[43.4, 48.4]],
    'DMU': [[18.7, 28.7], [32.7, 39.7]]
}


def plotly_timeseries(DSS_dict, DAS_dict, simfip, hydro, seismic, packers=None,
                      accel_dict=None):
    """
    DataFrame of timeseries data of any kind

    :param DSS_dict:
    :param DAS_dict:
    :param simfip: Simfip dataframe
    :param hydro: Hydro dataframe
    :param packers: Packer pressure dataframe
    :param seismic: Dict of seismicity of keys 'times' and 'dists'
    :param accel_dict: Keys 'data' and 'times'

    :return:
    """

    # Try to go back to subplots?
    fig = go.Figure()
    # Establish cycle of accelerometer traces
    acc_cols = cycle(sns.color_palette().as_hex())
    # Fibers traces
    fig.add_trace(go.Scatter(x=DSS_dict['times'],
                             y=DSS_dict['DSS_median'] - DSS_dict['DSS_std'],
                             line=dict(color="mediumvioletred", width=0),
                             fill=None, showlegend=False, hoverinfo='skip',
                             legendgroup='1'))
    fig.add_trace(go.Scatter(x=DSS_dict['times'],
                             y=DSS_dict['DSS_median'] + DSS_dict['DSS_std'],
                             line=dict(color="mediumvioletred", width=0),
                             fill='tonexty', showlegend=False,
                             hoverinfo='skip', legendgroup='1'))
    fig.add_trace(go.Scatter(x=DSS_dict['times'], y=DSS_dict['DSS_median'],
                             name="DSS: 20 min rolling median",
                             line=dict(color='mediumvioletred'),
                             fill=None, legendgroup='1'))
    fig.add_trace(go.Scatter(x=DAS_dict['times'], y=DAS_dict['data'],
                             name='DAS at {} m'.format(DAS_dict['depth']),
                             line=dict(color='#9467bd', width=2.5), yaxis='y7'))
    fig.add_trace(go.Scatter(x=DSS_dict['times'], y=DSS_dict['DTS'],
                             name="DTS at {} m".format(DSS_dict['depth']),
                             yaxis="y2", line=dict(color='#ee854a')))
    # Hydraulic data
    fig.add_trace(go.Scatter(x=hydro.index, y=hydro['Flow'], name="Pump Flow",
                             line=dict(color='steelblue'),
                             yaxis='y3'))
    fig.add_trace(go.Scatter(x=hydro.index, y=hydro['Pressure'],
                             name="Pump Pressure", line=dict(color='red'),
                             yaxis='y4'))
    # Packer pressures
    if packers:
        fig.add_trace(go.Scatter(
            x=packers.index,
            y=packers['Pressure_Packer_Upper_Injection_SNL06'] / 145.,
            name="E1-I: Upper Packer Pressure", line=dict(color='lime'),
            yaxis='y4'))
        fig.add_trace(go.Scatter(
            x=packers.index,
            y=packers['Pressure_Packer_Lower_Injection_SNL07'] / 145.,
            name="E1-I: Lower Packer Pressure", line=dict(color='olive'),
            yaxis='y4'))
        fig.add_trace(go.Scatter(
            x=packers.index,
            y=packers['Pressure_Packer_Upper_Production_SNL08'] / 145.,
            name="E1-P: Upper Packer Pressure", line=dict(color='maroon'),
            yaxis='y4'))
        fig.add_trace(go.Scatter(
            x=packers.index,
            y=packers['Pressure_Packer_Lower_Production_SNL09'] / 145.,
            name="E1-P: Lower Packer Pressure", line=dict(color='darkred'),
            yaxis='y4'))
    # Earthquakes
    fig.add_trace(go.Scatter(x=seismic['times'], y=seismic['dists'],
                             name='Seismic event', mode='markers',
                             marker=dict(color='rgba(0,0,0,0.05)',
                                         line=dict(color='rgba(0,0,0,1.0)',
                                                   width=0.5)), yaxis='y6'))
    # Add reference line for distance to OT
    fig.add_trace(go.Scatter(x=[seismic['times'][0], seismic['times'][-1]],
                             y=[7.457, 7.457],
                             name='Distance to OT', mode='lines',
                             line=dict(color='red', dash='dot',
                                       width=1.),
                             yaxis='y6'))
    # SIMFIP
    fig.add_trace(go.Scatter(x=simfip.index, y=simfip['P Yates'],
                             name='E1-P Yates', yaxis='y5'))
    fig.add_trace(go.Scatter(x=simfip.index, y=simfip['P Top'],
                             name='E1-P Top', yaxis='y5'))
    fig.add_trace(go.Scatter(x=simfip.index, y=simfip['P Axial'],
                             name='E1-P Axial', yaxis='y5'))
    fig.add_trace(go.Scatter(x=simfip.index, y=simfip['I Yates'],
                             name='E1-I Yates', yaxis='y5'))
    fig.add_trace(go.Scatter(x=simfip.index, y=simfip['I Top'],
                             name='E1-I Top', yaxis='y5'))
    fig.add_trace(go.Scatter(x=simfip.index, y=simfip['I Axial'],
                             name='E1-I Axial', yaxis='y5'))
    # Synthetic DSS for production
    # simfip['P shear'] = np.sqrt(simfip['P Yates']**2 +
    #                             simfip['P Top']**2) / 2.5
    # simfip['Arc'] = ((simfip['P shear'] / 2.5)**2) / 2
    # simfip['Synthetic DSS'] = ((simfip['P Axial'] / 2.5) +
    #                             simfip['Arc'])
    # fig.add_trace(go.Scatter(x=simfip.index, y=simfip['Synthetic DSS'],
    #                          name='E1-P Synth DSS', yaxis='y5'))
    # fig.add_trace(go.Scatter(x=simfip.index, y=simfip['P shear'],
    #                          name='E1-P Shear', yaxis='y5'))
    # fig.add_trace(go.Scatter(x=simfip.index, y=simfip['Arc'],
    #                          name='E1-P Arc', yaxis='y5'))
    # Create axis objects
    fig.update_layout(
        xaxis=dict(showspikes=True, spikemode='across', spikethickness=1.5,
                   spikedash='dot', domain=[0., 0.95], type='date',
                   anchor='y5'),
        yaxis=dict(title="$\mu\epsilon$", titlefont=dict(color="black"),
                   tickfont=dict(color="black"), domain=[0.725, 0.9],
                   anchor='x', zeroline=False),
        yaxis2=dict(title=r"$\Delta{T}\;\text{[}^{\circ}\text{C]}$",
                    titlefont=dict(color='#ee854a'),
                    tickfont=dict(color='#ee854a'), overlaying="y",
                    side="right", showgrid=False, anchor='x'),
        yaxis3=dict(title="Flow [L/min]", titlefont=dict(color="steelblue"),
                    tickfont=dict(color="steelblue"), side="left",
                    rangemode='nonnegative', domain=[0.9, 1.0],
                    anchor='x'),
        yaxis4=dict(title="Pressure [MPa]", titlefont=dict(color="red"),
                    tickfont=dict(color="red"), overlaying="y3", side="right",
                    rangemode='nonnegative', anchor='x'),
        yaxis5=dict(title="Displacement [m]", titlefont=dict(color="black"),
                    tickfont=dict(color="black"), domain=[0., 0.45],
                    anchor='x'),
        yaxis6=dict(title="Distance from 50 m notch [m]",
                    titlefont=dict(color="black"), range=[0, 30],
                    tickfont=dict(color="black"), domain=[0., 0.45],
                    anchor='x', side='right', overlaying='y5'),
        yaxis7=dict(title="$\mu\epsilon$", titlefont=dict(color="black"),
                   tickfont=dict(color="black"), domain=[0.45, 0.725],
                   anchor='x', zeroline=False))
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_layout(template="ggplot2", legend=dict(traceorder='reversed'))
    fig.show(renderer='firefox')
    return


def plot_lab_3D(outfile, location, catalog=None, inventory=None, well_file=None,
                wells=None, title=None, offline=True, dd_only=False,
                surface=None, fault=None, DSS_picks=None, structures=None,
                meshes=None, line=None, simfip=None, fracs=False, surfaces=None,
                xrange=(2579250, 2579400), yrange=(1247500, 1247650),
                zrange=(450, 500), sampling=0.5, eye=None, export=False,
                drift_2D=False):
    """
    Plot boreholes, seismicity, monitoring network, etc in 3D in plotly

    :param outfile: Name of plot in plotly
    :param location: Either 'fsb' or 'surf' as of 12-18-19
    :param catalog: Optional catalog of seismicity
    :param inventory: Optional inventory for monitoring network
    :param well_file: If field == 'surf', must provide well (x, y, z) file
    :param wells: List of wells to plot
    :param title: Title of plot
    :param offline: Boolean for whether to plot to plotly account (online)
        or to local disk (offline)
    :param dd_only: Are we only plotting dd locations?
    :param surface: What type of surface to fit to points? Supports 'plane'
        and 'ellipsoid' for now.
    :param fault: Tuple of 2D arrays for X, Y, Z of fault
    :param DSS_picks: Dictionary {well name: {'heights': array,
                                              'widths': array,
                                              'depths': list}}
    :param structures: None or path to root well_info directory
    :param meshes: list of tup (Layer name, path) for files containing xyz
        vertices for mesh (only used for FSB Gallery at the moment; can be
        expanded)
    :param line: Bool to add excavation progress at Mont Terri
    :param simfip: Dict of {well name: (top packer depth, bottom packer depth)}
    :param fracs: bool plot fracture planes at surf or not
    :param surfaces: Plot CASSM result surfaces as FSB or empty
    :param xrange: List of min and max x of volume to interpolate DSS over
    :param yrange: List of min and max y of volume to interpolate DSS over
    :param zrange: List of min and max z of volume to interpolate DSS over
    :param sampling: Sampling interval for ranges above (meters)
    :param export: Bool to just return fig object for manual export

    :return:
    """
    if not title:
        title = '3D Plot'
    # Make well point lists and add to Figure
    datas = []
    if location == '4850':
        well_dict = parse_surf_boreholes(well_file)
    elif location == '4100':
        well_dict = make_4100_boreholes(well_file)
    elif location == 'fsb':
        # Too many points in asbuilt file to upload to plotly
        well_dict = create_FSB_boreholes()
    else:
        print('Location {} not supported'.format(location))
        return
    datas = add_wells(well_dict, objects=datas, structures=structures,
                      wells=wells)
    if inventory:
        datas = add_inventory(inventory=inventory, location=location,
                              objects=datas)
        if location == 'surf':
            datas = add_surf_sources(well_dict, datas)
    if DSS_picks:
        try:
            datas = add_DSS(DSS_picks=DSS_picks, objects=datas,
                            well_dict=well_dict)
        except KeyError as e:
            print('Havent provided discrete picks with height, width')
        datas = add_DSS_volume_slices(objects=datas, pick_dict=DSS_picks,
                                      xrange=xrange, yrange=yrange,
                                      zrange=zrange, sampling=sampling)
    if meshes:
        datas = add_meshes(meshes=meshes, objects=datas)
    if structures:
        datas = add_structures(structures=structures, objects=datas,
                               well_dict=well_dict)
    if catalog:
        datas = add_catalog(catalog=catalog, dd_only=dd_only, objects=datas,
                            surface=surface, location=location)
    if line:
        add_time_colored_line(objects=datas)
    if simfip:
        for wl, deps in simfip.items():
            add_fsb_simfip(datas, wl, deps)
    if fracs:
        add_4850_fracs(datas, well_file)
    if surfaces:
        add_surface(surfaces, datas)
    if drift_2D:
        add_drift_2D(drift_2D, datas)
    try:
        print(fault)
        add_fault(fault[0, :], fault[1, :], fault[2, :], datas)
    except TypeError:
        pass
    # Start figure
    fig = go.Figure(data=datas)
    # Manually find the data limits, and scale appropriately
    all_x = np.ma.masked_invalid(np.concatenate(
        [d['x'].flatten() for d in fig.data if type(d['y']) == np.ndarray
         and d.name != 'Drift']))
    all_y = np.ma.masked_invalid(np.concatenate(
        [d['y'].flatten() for d in fig.data if type(d['y']) == np.ndarray
         and d.name != 'Drift']))
    all_z = np.ma.masked_invalid(np.concatenate(
        [d['z'].flatten() for d in fig.data if type(d['z']) == np.ndarray
         and d.name != 'Drift']))
    xrange = np.abs(np.max(all_x) - np.min(all_x))
    yrange = np.abs(np.max(all_y) - np.min(all_y))
    if location == 'fsb':
        zmin = 400.
    else:
        zmin = np.min(all_z)
    zrange = np.abs(np.max(all_z) - zmin)
    if not eye:
        eye = (1.25, 1.25, 1.25)
    xax = go.layout.scene.XAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Easting (m)',
                                range=(np.min(all_x), np.max(all_x)))
    yax = go.layout.scene.YAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Northing (m)',
                                range=(np.min(all_y), np.max(all_y)))
    zax = go.layout.scene.ZAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Elevation (m)',
                                range=(zmin, np.max(all_z)))

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    # frames = []
    # for t in np.arange(0, 6.26, 0.1):
    #     xe, ye, ze = rotate_z(eye[0], eye[1], eye[2], -t)
    #     frames.append(dict(
    #         layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    #
    # fig.frames = frames
    layout = go.Layout(scene=dict(xaxis=xax, yaxis=yax, zaxis=zax,
                                  xaxis_showspikes=False,
                                  yaxis_showspikes=False,
                                  aspectmode='manual',
                                  aspectratio=dict(x=1, y=yrange / xrange,
                                                   z=zrange / xrange),
                                  bgcolor="rgb(244, 244, 248)",
                                  camera=dict(eye=dict(x=eye[0], y=eye[1],
                                                       z=eye[2]))),
                       autosize=True,
                       title=title,
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
                                   tracegroupgap=3),
                       # updatemenus=[dict(type='buttons',
                       #                   showactive=True,
                       #                   active=-1,
                       #                   direction="left",
                       #                   pad={"r": 10, "t": 70},
                       #                   x=0.1, y=0.1,
                       #                   buttons=[dict(label="&#9654;",
                       #                                 method='animate',
                       #                                 args=[None, dict(frame=dict(duration=1, redraw=False),
                       #                                                  transition=dict(duration=1, easing='linear'),
                       #                                                  fromcurrent=True,
                       #                                                  mode='immediate'
                       #                                                  )]
                       #                                 ),
                       #                            {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                       #                                               "mode": "immediate",
                       #                                               "transition": {"duration": 0}}],
                       #                             "label": "&#9724;",
                       #                             "method": "animate"}]
                       #                   )
                       #              ]
                       )
    fig.update_layout(layout)
    if export:
        return fig
    if offline:
        plotly.offline.iplot(fig, filename='{}.html'.format(outfile))
    else:
        py.plot(fig, filename=outfile)
    return fig


def animate_dug_seis_location(outfile, well_dict, active_events=None,
                              passive_events=None, cluster=False,
                              inventory=None):
    """
    Matplotlib animation of a dug-seis location at FSB

    :param outfile: Path to output file
    :param well_dict: Dict from create_FSB_boreholes()
    :param active_events: DataFrame or Catalog
    :param passive_events: DataFrame or Catalog
    :param cluster: Bool for clustering
    :param inventory: obspy Inventory
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=90, azim=-90)
    plot_dug_seis_locations_fsb(well_dict, active_events,
                                passive_events,
                                cluster,
                                inventory,
                                fig, ax)

    def animate(i):
        if i < 90:
            el = 90 - i
            az = -90
        else:
            el = 0
            az = -i
        ax.view_init(elev=el, azim=az)
        return fig,

    # Animate
    anim = animation.FuncAnimation(
        fig, animate, frames=450, interval=20)
    # Save
    anim.save(outfile, extra_args=['-vcodec', 'libx264'],
              dpi=200)
    return


def plot_dug_seis_locations_fsb(well_dict, active_events=None,
                                passive_events=None, cluster=False,
                                inventory=None, figure=None, axes=None):
    """
    Plot location output of dug-seis on wells at FSB

    :param well_dict: output from create_FSB_boreholes
    :param active_events: DataFrame or catalog of "active" source shots
    :param passive_events: DataFrame or catalog of "passive" sources
    :param cluster: False or number of kmeans clusters to create
    :param inventory: obspy.core.Inventory of stations

    :return:
    """
    if not axes:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=90, azim=-90)
    else:
        fig = figure
        ax = axes
    # Make a data array for active
    if isinstance(active_events, pd.DataFrame):
        data_act = active_events[['x', 'y', 'z']].to_numpy()
    elif isinstance(active_events, Catalog):
        oextra = [ev.origins[-1].extra for ev in active_events]
        times = [ev.picks[0].time.datetime.timestamp() for ev in active_events]
        data_act = np.array([[float(d.ch1903_east.value),
                              float(d.ch1903_north.value),
                              float(d.ch1903_elev.value),]
                             for d in oextra])
        if len(active_events) == 1:
            act_scatter = active_events[0].origins[-1].nonlinloc_scatter
    else:
        data_act = np.array([])
    if isinstance(passive_events, pd.DataFrame):
        data_pass = passive_events[['x', 'y', 'z']].to_numpy()
    elif isinstance(passive_events, Catalog):
        oextra = [ev.origins[-1].extra for ev in passive_events]
        oarrs = [ev.origins[-1].arrivals for ev in passive_events]
        data_pass = np.array([[float(d.ch1903_east.value),
                               float(d.ch1903_north.value),
                               float(d.ch1903_elev.value)]
                         for d in oextra])
        if len(passive_events) == 1:
            title = passive_events[0].origins[0].time
            oarrs = passive_events[0].origins[-1].arrivals
            arr_stas = [arr.pick_id.get_referred_object().waveform_id.station_code
                        for arr in oarrs]
            pass_scatter = passive_events[0].origins[-1].nonlinloc_scatter
    else:
        data_pass = np.array([])
    # If cluster, plot em up
    if cluster and data_pass.size > 0:
        km = KMeans(n_clusters=cluster, init='random',
                    n_init=10, max_iter=300,
                    tol=1e-04, random_state=0)
        clusters = km.fit_predict(data_pass)
        clusts = {}
        for c in list(set(clusters)):
            events = data_pass[clusters == c]
            clusts[c] = events
            ax.scatter(events[:, 0], events[:, 1], events[:, 2], alpha=0.5,
                       s=5., label=c)
    else:
        if data_act.size > 0:
            if data_act.shape[0] == 1:
                # Plot scatter too
                ax.scatter(act_scatter[:, 0], act_scatter[:, 1],
                           act_scatter[:, 2], color='lightgray',
                           alpha=0.1, s=2)
            else:
                map = ax.scatter(data_act[:, 0], data_act[:, 1], data_act[:, 2],
                                 c=times, s=10, alpha=0.5, zorder=120, cmap='copper')
                cax = plt.colorbar(map)
                cax_labs = [int(l.get_text()) for l in fig.axes[-1].get_yticklabels()]
                new_labs = [datetime.fromtimestamp(ts) for ts in cax_labs]
                fig.axes[-1].set_yticklabels(new_labs)
        if data_pass.size > 0:
            if data_pass.shape[0] == 1:
                # Plot scatter too
                ax.scatter(pass_scatter[:, 0], pass_scatter[:, 1],
                           pass_scatter[:, 2], color='k',
                           alpha=0.1, s=2, zorder=-1)
                if inventory:
                    # Get xyz for all stations in arrivals list
                    for sta in arr_stas:
                        # Plot ray
                        xyz = inventory.select(station=sta)[0][0].extra
                        xyz = [float(xyz.ch1903_east.value),
                               float(xyz.ch1903_north.value),
                               float(xyz.ch1903_elev.value)]
                        ax.scatter(xyz[0], xyz[1], xyz[2], marker='x',
                                   color='k')
                        ax.plot([xyz[0], data_pass[0, 0]],
                                [xyz[1], data_pass[0, 1]],
                                [xyz[2], data_pass[0, 2]], color='k',
                                alpha=0.3, linewidth=0.75)
                ax.set_title(title.strftime('%d-%b %H:%M:%S.%f'), fontsize=14)
                ax.text2D(-0.3, 0.75, 'East: {}\nNorth: {}\nElev: {}'.format(
                    data_pass[0, 0], data_pass[0, 1], data_pass[0, 2]),
                    transform=ax.transAxes)
            ax.scatter(data_pass[:, 0], data_pass[:, 1], data_pass[:, 2],
                       color='r', marker='s', s=40, alpha=0.7, zorder=5)
            ax.plot(data_pass[:, 0], data_pass[:, 1], data_pass[:, 2],
                    color='r', linewidth=1., marker=None, alpha=0.7, zorder=5)
    # Plot up the well bores
    for w, pts in well_dict.items():
        if w.startswith('B'):
            wx = pts[:, 0]# + 579300
            wy = pts[:, 1]# + 247500
            wz = pts[:, 2]# + 500
            ax.scatter(wx[0], wy[0], wz[0], s=10., marker='s',
                       color=fsb_well_colors[w])
            ax.plot(wx, wy, wz, color=fsb_well_colors[w])
    # ax.set_xlim([3158610, 3158655])
    # ax.set_ylim([1495055, 1495100])
    # ax.set_zlim([985, 1030])
    if not axes:
        plt.show()
    if cluster:
        return clusters
    return fig,


def plot_fsb_inventory(well_dict, inventory, plot_asbuilt=False):
    """
    Plot stations on boreholes

    :param well_dict: Dictionary of well tracks
    :param inventory: obspy Inventory object
    :param stas: List of stas in inventory to plot
    """
    if plot_asbuilt:
        stas = [sta.code for sta in inventory[0] if sta.code[0] != 'S']
    else:
        stas = [
            'B301', 'B302', 'B303', 'B304', 'B305', 'B306', 'B307', 'B308',
            'B309', 'B310', 'B311', 'B312', 'B313', 'B314', 'B315', 'B316',
            'B317', 'B318', 'B319', 'B320', 'B321', 'B322',
            'B401', 'B402', 'B403', 'B404', 'B405', 'B406', 'B407', 'B408',
            'B409', 'B410', 'B411', 'B412', 'B413', 'B414', 'B415', 'B416',
            'B417', 'B418', 'B419', 'B420', 'B421', 'B422',
            'B81', 'B82', 'B83', 'B91'
        ]
    fig = plt.figure()
    print('fo')
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=90, azim=-90)
    # Plot up the well bores
    for w, pts in well_dict.items():
        if w.startswith('B'):
            wx = pts[:, 0]# + 579300
            wy = pts[:, 1]# + 247500
            wz = pts[:, 2]# + 500
            ax.scatter(wx[0], wy[0], wz[0], s=10., marker='s',
                       color=fsb_well_colors[w])
            ax.plot(wx, wy, wz, color=fsb_well_colors[w])
    for sta in stas:
        # Plot ray
        xyz = inventory.select(station=sta)[0][0].extra
        xyz = [float(xyz.ch1903_east.value),
               float(xyz.ch1903_north.value),
               float(xyz.ch1903_elev.value)]
        if sta[:2] in ['B8', 'B9']:
            color = 'r'
            label = 'AE'
        elif len(sta) == 3 or sta[:2] in ['B5', 'B6', 'B7']:
            color = 'b'
            label = 'Accelerometer'
        elif len(sta) == 4:
            color = 'g'
            label = 'Hydrophone'
        ax.scatter(xyz[0], xyz[1], xyz[2], marker='x',
                   color=color, label=label)
    legend_without_duplicate_labels(ax)
    return


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def plot_4850_2D(autocad_path, strike=347.,
                 origin=np.array((811.61, -1296.63, 105.28)),
                 seismicity=None):
    """
    Plot SURF 4850 in a combination of 3D, map view, and cross section

    :param autocad_path: Path to file with arcs and lines etc
    :param strike: Strike of main fault to project piercepoints onto
    :param origin: Origin point for the cross section
    :param seismicity: obspy Catalog

    :return:
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 12))
    spec = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    ax3d = fig.add_subplot(spec[:4, :4], projection='3d')
    ax_x = fig.add_subplot(spec[:4, 4:])
    ax_map = fig.add_subplot(spec[4:, :4])
    ax_fault = fig.add_subplot(spec[4:, 4:])
    well_dict = parse_surf_boreholes(
        'data/chet-collab/boreholes/surf_4850_wells.csv')
    # Cross section plane
    r = np.deg2rad(360 - strike)
    normal = np.array([-np.cos(r), -np.sin(r), 0.])
    normal /= norm(normal)
    new_strk = np.array([np.sin(r), -np.cos(r), 0.])
    new_strk /= norm(new_strk)
    change_b_mat = np.array([new_strk, [0, 0, 1], normal])
    # Theoretical fracture
    frac = {'strike': 77, 'dip': 79, 'radius': 15,
            'center': depth_to_xyz(well_dict, 'I', 50.2),
            'color': 'purple'}
    s = np.deg2rad(frac['strike'])
    d = np.deg2rad(frac['dip'])
    # Define fault normal
    n = np.array([np.sin(d) * np.cos(s),
                  -np.sin(d) * np.sin(s),
                  np.cos(d)])
    u = np.array([np.sin(s), np.cos(s), 0])
    # Equ from https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
    Pc = [(frac['radius'] * np.cos(t) * u) +
          (frac['radius'] * np.sin(t) * np.cross(n, u)) + frac['center']
          for t in np.linspace(0, 2 * np.pi, 50)]
    Pc = np.array(Pc)
    verts = [list(zip(Pc[:, 0], Pc[:, 1], Pc[:, 2]))]
    poly = Poly3DCollection(verts, alpha=0.3, color='blue')
    ax3d.add_collection3d(poly)
    # Now drift mesh
    Xs, Ys, Zs, tris = dxf_to_mpl(autocad_path)
    if not seismicity:
        ax3d.plot_trisurf(Xs, Ys, Zs, triangles=tris, color='gray', alpha=0.3)
    # Proj
    pts_t = np.column_stack([Pc[:, 0].flatten(), Pc[:, 1].flatten(),
                             Pc[:, 2].flatten()])
    proj_pts_t = np.dot(pts_t - origin, normal)[:, None] * normal
    proj_pts_t = pts_t - origin - proj_pts_t
    proj_pts_t = np.matmul(change_b_mat, proj_pts_t.T)
    # 3D mesh with triangles
    ax_map.fill(Pc[:, 0], Pc[:, 1], color='blue', alpha=0.3,
                label='50 m fracture', edgecolor='b', linewidth=1.)
    ax_x.fill(proj_pts_t[0, :], proj_pts_t[1, :], color='blue', alpha=0.3,
              edgecolor='b', linewidth=1.)
    # Convex hull in 2D views
    map_pts = np.column_stack([Xs, Ys, Zs])
    map_hull = ConvexHull(map_pts[:, :2])
    ax_map.fill(map_pts[map_hull.vertices, 0],
                map_pts[map_hull.vertices, 1], color='gray', alpha=0.3)
    drift_proj = np.dot(map_pts - origin, normal)[:, None] * normal
    drift_proj = map_pts - origin - drift_proj
    drift_proj = np.matmul(change_b_mat, drift_proj.T)
    drift_proj = drift_proj.T[:, :2]
    x_hull = ConvexHull(drift_proj)
    ax_x.fill(drift_proj[x_hull.vertices, 0],
              drift_proj[x_hull.vertices, 1],
              color='gray', alpha=0.3)
    # Plot notch location
    notch_pos = depth_to_xyz(well_dict, 'I', 50.2)
    ax_map.scatter(notch_pos[0], notch_pos[1], marker='x',
                   s=70, color='b', label='50-m notch')
    ax_x.scatter(0, 0, marker='x', s=70, color='b')
    ax3d.scatter(origin[0], origin[1], origin[2], marker='x', s=70, color='b')
    for well, pts in well_dict.items():
        if well.startswith('SW'):
            continue
        col = cols_4850[well]
        # Proj
        pts = pts[:, :3]
        proj_pts = np.dot(pts - origin, normal)[:, None] * normal
        proj_pts = pts - origin - proj_pts
        proj_pts = np.matmul(change_b_mat, proj_pts.T)
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col,
                  linewidth=1.5)
        ax_x.plot(proj_pts[0, :], proj_pts[1, :], color=col)
        ax_map.plot(pts[:, 0], pts[:, 1], color=col, linewidth=2.)
        ax_map.scatter(pts[:, 0][0], pts[:, 1][0], color=col, s=15.)
        ax_map.annotate(
            text=well, xy=(pts[:, 0][0], pts[:, 1][1]), fontsize=10,
            weight='bold', xytext=(3, 0), textcoords="offset points",
            color=col)
    # Seismicity
    if seismicity:
        eq_pts = [(float(ev.origins[-1].extra.hmc_east.value),
                   float(ev.origins[-1].extra.hmc_north.value),
                   float(ev.origins[-1].extra.hmc_elev.value))
                  for ev in seismicity]
        eq_pts = np.array(eq_pts)
        proj_eq_pts = np.dot(eq_pts - origin, normal)[:, None] * normal
        proj_eq_pts = eq_pts - origin - proj_eq_pts
        proj_eq_pts = np.matmul(change_b_mat, proj_eq_pts.T)
        ax3d.scatter(eq_pts[:, 0], eq_pts[:, 1], eq_pts[:, 2], color='gray',
                     s=1., alpha=0.5)
        ax_x.scatter(proj_eq_pts[0, :], proj_eq_pts[1, :], color='gray', s=1.,
                     alpha=0.5)
        ax_map.scatter(eq_pts[:, 0], eq_pts[:, 1], color='gray', s=1.,
                       alpha=0.5)
        # Onto fract plane
        fs_rad = np.deg2rad(frac['strike'])
        fd_rad = np.deg2rad(frac['dip'])
        frac_norm = np.array((np.sin(fd_rad) * np.cos(fs_rad),
                              -np.sin(fd_rad) * np.sin(fs_rad),
                              np.cos(fd_rad)))
        frac_norm /= norm(frac_norm)
        frac_strk = np.array([np.sin(fs_rad), np.cos(r), 0.])
        frac_strk /= norm(frac_strk)
        frac_dip = np.array([-np.cos(fs_rad) * np.cos(fd_rad),
                             np.sin(fs_rad) * np.cos(fd_rad),
                             np.sin(fd_rad)])
        frac_dip /= norm(frac_dip)
        change_b_frac = np.array([frac_strk, frac_dip, frac_norm])
        frac_pts = np.dot(eq_pts - origin, frac_norm)[:, None] * frac_norm
        frac_pts = eq_pts - origin - frac_pts
        frac_pts = np.matmul(change_b_frac, frac_pts.T).T
        ax_fault.scatter(frac_pts[:, 0], frac_pts[:, 1], color='gray',
                         s=1., alpha=0.5)
    # Plot fault coords and piercepoints
    grdx, grdy = np.meshgrid(Pc[:, 0], Pc[:, 1])
    grdz = ((-0.238 * grdx) + grdy + 1510.9) / 0.198
    plot_pierce_points(grdx, grdy, grdz, strike=frac['strike'],
                       dip=frac['dip'], ax=ax_fault, location='surf')
    ax_fault.add_artist(Circle((0, 0), radius=frac['radius'],
                               alpha=0.3, color='b'))
    # Formatting
    if seismicity:
        ax3d.set_xlim([800, 820])
        ax3d.set_ylim([-1310, -1290])
        ax3d.set_zlim([90, 120])
    else:
        ax3d.set_xlim([790, 840])
        ax3d.set_ylim([-1330, -1280])
        ax3d.set_zlim([80, 130])
    ax3d.view_init(elev=30., azim=-158)
    ax3d.margins(0.)
    ax3d.set_xticks([])
    ax3d.set_xticklabels([])
    ax3d.set_yticks([])
    ax3d.set_yticklabels([])
    ax3d.set_zticks([])
    ax3d.set_zticklabels([])
    # Overview map
    ax_map.axis('equal')
    if not seismicity:
        ax_map.axis('off')
    else:
        ax_map.set_xlabel('Easting [m]')
        ax_map.set_ylabel('Northing [m]')
    if seismicity:
        ax_map.set_xlim([795, 835])
        ax_map.set_ylim([-1315, -1275])
    else:
        ax_map.set_xlim([780, 860])
        ax_map.set_ylim([-1350, -1270])
    # Fault map
    ax_fault.axis('equal')
    ax_fault.tick_params(direction='in', left=False, labelleft=False)
    if seismicity:
        ax_fault.set_xlim([-15, 15])
        ax_fault.set_xticks([-20, -10, 0, 10, 20])
        ax_fault.set_xticklabels(['0', '10', '20', '30', '40'])
    else:
        ax_fault.set_xticks([-10, -5, 0, 5, 10])
        ax_fault.set_xticklabels(['0', '5', '10', '15', '20'])
    ax_fault.set_xlabel('Meters')
    # Cross section
    if seismicity:
        ax_x.set_xlim([-20, 10])
    else:
        ax_x.set_xlim([-20, 50])
    ax_x.axis('equal')
    ax_x.spines['top'].set_visible(False)
    ax_x.spines['bottom'].set_visible(False)
    ax_x.spines['left'].set_visible(False)
    ax_x.yaxis.set_ticks_position('right')
    ax_x.tick_params(direction='in', bottom=False, labelbottom=False)
    if seismicity:
        ax_x.spines['right'].set_bounds(-15, 10)
        ax_x.set_yticks([-15, -10, -5, 0, 5, 10])
        ax_x.set_yticklabels(['25', '20', '15', '10', '5', '0'])
    else:
        ax_x.spines['right'].set_bounds(-30, 20)
        ax_x.set_yticks([-30, -20, -10, 0, 10, 20])
        ax_x.set_yticklabels(['50', '40', '30', '20', '10', '0'])
    ax_x.set_ylabel('Meters', labelpad=15)
    ax_x.yaxis.set_label_position("right")
    fig.legend(loc=10)
    plt.show()
    return


def plot_CSD_2D(autocad_path, strike=305.,
                origin=np.array([2579325., 1247565., 514.])):
    """
    Plot the Mont Terri lab in a combination of 3D, map view, and cross section

    :param autocad_path: Path to file with arcs and lines etc
    :param strike: Strike of main fault to project piercepoints onto
    :param origin: Origin point for the cross section

    :return:
    """
    fig = plt.figure(figsize=(12, 12))
    spec = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    ax3d = fig.add_subplot(spec[:4, :4], projection='3d')
    ax_x = fig.add_subplot(spec[:4, 4:])
    ax_map = fig.add_subplot(spec[4:, :4])
    ax_fault = fig.add_subplot(spec[4:, 4:])
    well_dict = create_FSB_boreholes()
    # Cross section plane (strike 320)
    r = np.deg2rad(360 - strike)
    normal = np.array([-np.sin(r), -np.cos(r), 0.])
    normal /= norm(normal)
    new_strk = np.array([np.sin(r), -np.cos(r), 0.])
    new_strk /= norm(new_strk)
    change_b_mat = np.array([new_strk, [0, 0, 1], normal])
    for afile in glob('{}/*.csv'.format(autocad_path)):
        # if 'FSB' in afile:
        #     continue
        df_cad = pd.read_csv(afile)
        lines = df_cad.loc[df_cad['Name'] == 'Line']
        arcs = df_cad.loc[df_cad['Name'] == 'Arc']
        for i, line in lines.iterrows():
            xs = np.array([line['Start X'], line['End X']])
            ys = np.array([line['Start Y'], line['End Y']])
            zs = np.array([line['Start Z'], line['End Z']])
            # Proj
            pts = np.column_stack([xs, ys, zs])
            proj_pts = np.dot(pts - origin, normal)[:, None] * normal
            proj_pts = pts - origin - proj_pts
            proj_pts = np.matmul(change_b_mat, proj_pts.T)
            ax3d.plot(xs, ys, zs, color='darkgray', zorder=110)
            ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='darkgray',
                      zorder=110)
            ax_map.plot(xs, ys, color='darkgray')
        for i, arc in arcs.iterrows():
            # Stolen math from Melchior
            if not np.isnan(arc['Extrusion Direction X']):
                rotaxang = [arc['Extrusion Direction X'],
                            arc['Extrusion Direction Y'],
                            arc['Extrusion Direction Z'],
                            arc['Total Angle']]
                rad = np.linspace(arc['Start Angle'], arc['Start Angle'] +
                                  arc['Total Angle'])
                dx = np.sin(np.deg2rad(rad)) * arc['Radius']
                dy = np.cos(np.deg2rad(rad)) * arc['Radius']
                dz = np.zeros(dx.shape[0])
                phi1 = -np.arctan2(
                    norm(np.cross(np.array([rotaxang[0], rotaxang[1], rotaxang[2]]),
                                  np.array([0, 0, 1]))),
                    np.dot(np.array([rotaxang[0], rotaxang[1], rotaxang[2]]),
                           np.array([0, 0, 1])))
                DX = dx * np.cos(phi1) + dz * np.sin(phi1)
                DY = dy
                DZ = dz * np.cos(phi1) - dx * np.sin(phi1)
                # ax.plot(DX, DY, DZ, color='r')
                phi2 = np.arctan(rotaxang[1] / rotaxang[0])
                fdx = (DX * np.cos(phi2)) - (DY * np.sin(phi2))
                fdy = (DX * np.sin(phi2)) + (DY * np.cos(phi2))
                fdz = DZ
                x = fdx + arc['Center X']
                y = fdy + arc['Center Y']
                z = fdz + arc['Center Z']
                # projected pts
                pts = np.column_stack([x, y, z])
                proj_pts = np.dot(pts - origin, normal)[:, None] * normal
                proj_pts = pts - origin - proj_pts
                proj_pts = np.matmul(change_b_mat, proj_pts.T)
                ax3d.plot(x, y, z, color='darkgray', zorder=110)
                ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='darkgray',
                          zorder=110)
                ax_map.plot(x, y, color='darkgray')
            elif not np.isnan(arc['Start X']):
                v1 = -1. * np.array([arc['Center X'] - arc['Start X'],
                                     arc['Center Y'] - arc['Start Y'],
                                     arc['Center Z'] - arc['Start Z']])
                v2 = -1. * np.array([arc['Center X'] - arc['End X'],
                                     arc['Center Y'] - arc['End Y'],
                                     arc['Center Z'] - arc['End Z']])
                rad = np.linspace(0, np.deg2rad(arc['Total Angle']), 50)
                # get rotation vector (norm is rotation angle)
                rotvec = np.cross(v2, v1)
                rotvec /= norm(rotvec)
                rotvec = rotvec[:, np.newaxis] * rad[np.newaxis, :]
                Rs = R.from_rotvec(rotvec.T)
                pt = np.matmul(v1, Rs.as_matrix())
                # Projected pts
                x = arc['Center X'] + pt[:, 0]
                y = arc['Center Y'] + pt[:, 1]
                z = arc['Center Z'] + pt[:, 2]
                pts = np.column_stack([x, y, z])
                proj_pts = np.dot(pts - origin, normal)[:, None] * normal
                proj_pts = pts - origin - proj_pts
                proj_pts = np.matmul(change_b_mat, proj_pts.T)
                ax3d.plot(x, y, z, color='darkgray', zorder=110)
                ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='darkgray',
                          zorder=110)
                ax_map.plot(x, y, color='darkgray')
    # Fault model
    fault_mod = '{}/faultmod.mat'.format(autocad_path)
    faultmod = loadmat(fault_mod, simplify_cells=True)['faultmod']
    x = faultmod['xq']
    y = faultmod['yq']
    zt = faultmod['zq_top']
    zb = faultmod['zq_bot']
    # ax3d.plot_surface(x, y, zt, color='bisque', alpha=.5)
    # ax3d.plot_surface(x, y, zb, color='bisque', alpha=.5)
    # Proj
    pts_t = np.column_stack([x.flatten(), y.flatten(), zt.flatten()])
    proj_pts_t = np.dot(pts_t - origin, normal)[:, None] * normal
    proj_pts_t = pts_t - origin - proj_pts_t
    proj_pts_t = np.matmul(change_b_mat, proj_pts_t.T)
    pts_b = np.column_stack([x.flatten(), y.flatten(), zb.flatten()])
    proj_pts_b = np.dot(pts_b - origin, normal)[:, None] * normal
    proj_pts_b = pts_b - origin - proj_pts_b
    proj_pts_b = np.matmul(change_b_mat, proj_pts_b.T)
    # ax_x.fill(proj_pts_t[0, :], proj_pts_t[1, :], color='bisque', alpha=0.7,
    #           label='Main Fault')
    # ax_x.fill(proj_pts_b[0, :], proj_pts_b[1, :], color='bisque', alpha=0.7)
    for well, pts in well_dict.items():
        if well[0] not in ['D', 'B']:
            continue
        try:
            col = csd_well_colors[well]
            zdr = 109
        except KeyError:
            col = 'lightgray'
            zdr = 90
        # Proj
        pts = pts[:, :3]
        proj_pts = np.dot(pts - origin, normal)[:, None] * normal
        proj_pts = pts - origin - proj_pts
        proj_pts = np.matmul(change_b_mat, proj_pts.T)
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col,
                  linewidth=1.5, zorder=zdr)
        ax_x.plot(proj_pts[0, :], proj_pts[1, :], color=col, zorder=zdr)
        ax_map.scatter(pts[:, 0][0], pts[:, 1][0], color=col, s=15.,
                       zorder=111)
        ax_map.annotate(text=well, xy=(pts[:, 0][0], pts[:, 1][1]), fontsize=10,
                        weight='bold', xytext=(3, 0),
                        textcoords="offset points", color=col)
    # Plot fault coords and piercepoints
    plot_pierce_points(x, y, zt, strike=47, dip=57, ax=ax_fault, location='fsb')
    # Formatting
    ax3d.set_xlim([2579295, 2579340])
    ax3d.set_ylim([1247555, 1247600])
    ax3d.set_zlim([485, 530])
    # ax3d.view_init(elev=30., azim=-112)
    ax3d.view_init(elev=32, azim=-175)
    ax3d.margins(0.)
    ax3d.set_xticks([])
    ax3d.set_xticklabels([])
    ax3d.set_yticks([])
    ax3d.set_yticklabels([])
    ax3d.set_zticks([])
    ax3d.set_zticklabels([])
    # Overview map
    ax_map.axis('equal')
    ax_map.axis('off')
    ax_map.set_xlim([2579300, 2579338])
    ax_map.set_ylim([1247540, 1247582])
    # Fault map
    ax_fault.axis('equal')
    ax_fault.spines['top'].set_visible(False)
    ax_fault.spines['left'].set_visible(False)
    ax_fault.spines['right'].set_visible(False)
    ax_fault.spines['bottom'].set_bounds(-5, 5)
    ax_fault.tick_params(direction='in', left=False, labelleft=False)
    ax_fault.set_xticks([-5, 0, 5])
    ax_fault.set_xticklabels(['0', '5', '10'])
    ax_fault.set_xlabel('Meters')
    # Cross section
    ax_x.set_xlim([-30, 5])
    ax_x.axis('equal')
    ax_x.spines['top'].set_visible(False)
    ax_x.spines['bottom'].set_visible(False)
    ax_x.spines['left'].set_visible(False)
    ax_x.yaxis.set_ticks_position('right')
    ax_x.tick_params(direction='in', bottom=False, labelbottom=False)
    ax_x.set_yticks([-30, -20, -10, 0])
    ax_x.set_yticklabels(['30', '20', '10', '0'])
    ax_x.set_ylabel('Meters', labelpad=15)
    ax_x.yaxis.set_label_position("right")
    ax_x.spines['right'].set_bounds(0, -30)
    fig.legend()
    plt.show()
    return


def plot_FSB_2D(autocad_path, strike=120.,
                origin=np.array([2579332., 1247600., 514.])):
    """
    Plot the Mont Terri lab in a combination of 3D, map view, and cross section

    :param autocad_path: Path to file with arcs and lines etc
    :param strike: Strike of main fault to project piercepoints onto
    :param origin: Origin point for the cross section

    :return:
    """
    fig = plt.figure(figsize=(12, 12))
    spec = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    ax3d = fig.add_subplot(spec[:4, :4], projection='3d')
    ax_x = fig.add_subplot(spec[:4, 4:])
    ax_map = fig.add_subplot(spec[4:, :4])
    ax_fault = fig.add_subplot(spec[4:, 4:])
    well_dict = create_FSB_boreholes()
    # Cross section plane (strike 320)
    r = np.deg2rad(360 - strike)
    normal = np.array([-np.sin(r), -np.cos(r), 0.])
    normal /= norm(normal)
    new_strk = np.array([np.sin(r), -np.cos(r), 0.])
    new_strk /= norm(new_strk)
    change_b_mat = np.array([new_strk, [0, 0, 1], normal])
    for afile in glob('{}/*.csv'.format(autocad_path)):
        # if 'FSB' in afile:
        #     continue
        df_cad = pd.read_csv(afile)
        lines = df_cad.loc[df_cad['Name'] == 'Line']
        arcs = df_cad.loc[df_cad['Name'] == 'Arc']
        for i, line in lines.iterrows():
            xs = np.array([line['Start X'], line['End X']])
            ys = np.array([line['Start Y'], line['End Y']])
            zs = np.array([line['Start Z'], line['End Z']])
            # Proj
            pts = np.column_stack([xs, ys, zs])
            proj_pts = np.dot(pts - origin, normal)[:, None] * normal
            proj_pts = pts - origin - proj_pts
            proj_pts = np.matmul(change_b_mat, proj_pts.T)
            ax3d.plot(xs, ys, zs, color='lightgray', zorder=110,
                      linewidth=0.5)
            ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='lightgray',
                      zorder=110, alpha=0.5, linewidth=0.5)
            ax_map.plot(xs, ys, color='darkgray', linewidth=0.5)
        for i, arc in arcs.iterrows():
            # Stolen math from Melchior
            if not np.isnan(arc['Extrusion Direction X']):
                rotaxang = [arc['Extrusion Direction X'],
                            arc['Extrusion Direction Y'],
                            arc['Extrusion Direction Z'],
                            arc['Total Angle']]
                rad = np.linspace(arc['Start Angle'], arc['Start Angle'] +
                                  arc['Total Angle'])
                dx = np.sin(np.deg2rad(rad)) * arc['Radius']
                dy = np.cos(np.deg2rad(rad)) * arc['Radius']
                dz = np.zeros(dx.shape[0])
                phi1 = -np.arctan2(
                    norm(np.cross(np.array([rotaxang[0], rotaxang[1], rotaxang[2]]),
                                  np.array([0, 0, 1]))),
                    np.dot(np.array([rotaxang[0], rotaxang[1], rotaxang[2]]),
                           np.array([0, 0, 1])))
                DX = dx * np.cos(phi1) + dz * np.sin(phi1)
                DY = dy
                DZ = dz * np.cos(phi1) - dx * np.sin(phi1)
                # ax.plot(DX, DY, DZ, color='r')
                phi2 = np.arctan(rotaxang[1] / rotaxang[0])
                fdx = (DX * np.cos(phi2)) - (DY * np.sin(phi2))
                fdy = (DX * np.sin(phi2)) + (DY * np.cos(phi2))
                fdz = DZ
                x = fdx + arc['Center X']
                y = fdy + arc['Center Y']
                z = fdz + arc['Center Z']
                # projected pts
                pts = np.column_stack([x, y, z])
                proj_pts = np.dot(pts - origin, normal)[:, None] * normal
                proj_pts = pts - origin - proj_pts
                proj_pts = np.matmul(change_b_mat, proj_pts.T)
                ax3d.plot(x, y, z, color='lightgray', zorder=110,
                          linewidth=0.5)
                ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='lightgray',
                          zorder=110, alpha=0.5, linewidth=0.5)
                ax_map.plot(x, y, color='darkgray', linewidth=0.5)
            elif not np.isnan(arc['Start X']):
                v1 = -1. * np.array([arc['Center X'] - arc['Start X'],
                                     arc['Center Y'] - arc['Start Y'],
                                     arc['Center Z'] - arc['Start Z']])
                v2 = -1. * np.array([arc['Center X'] - arc['End X'],
                                     arc['Center Y'] - arc['End Y'],
                                     arc['Center Z'] - arc['End Z']])
                rad = np.linspace(0, np.deg2rad(arc['Total Angle']), 50)
                # get rotation vector (norm is rotation angle)
                rotvec = np.cross(v2, v1)
                rotvec /= norm(rotvec)
                rotvec = rotvec[:, np.newaxis] * rad[np.newaxis, :]
                Rs = R.from_rotvec(rotvec.T)
                pt = np.matmul(v1, Rs.as_matrix())
                # Projected pts
                x = arc['Center X'] + pt[:, 0]
                y = arc['Center Y'] + pt[:, 1]
                z = arc['Center Z'] + pt[:, 2]
                pts = np.column_stack([x, y, z])
                proj_pts = np.dot(pts - origin, normal)[:, None] * normal
                proj_pts = pts - origin - proj_pts
                proj_pts = np.matmul(change_b_mat, proj_pts.T)
                ax3d.plot(x, y, z, color='lightgray', zorder=110,
                          linewidth=0.5)
                ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='lightgray',
                          zorder=110, alpha=0.5, linewidth=0.5)
                ax_map.plot(x, y, color='darkgray', linewidth=0.5)
    # Fault model
    fault_mod = '{}/faultmod.mat'.format(autocad_path)
    faultmod = loadmat(fault_mod, simplify_cells=True)['faultmod']
    x = faultmod['xq']
    y = faultmod['yq']
    zt = faultmod['zq_top']
    zb = faultmod['zq_bot']
    # ax3d.plot_surface(x, y, zt, color='bisque', alpha=.5)
    # ax3d.plot_surface(x, y, zb, color='bisque', alpha=.5)
    # Proj
    pts_t = np.column_stack([x.flatten(), y.flatten(), zt.flatten()])
    proj_pts_t = np.dot(pts_t - origin, normal)[:, None] * normal
    proj_pts_t = pts_t - origin - proj_pts_t
    proj_pts_t = np.matmul(change_b_mat, proj_pts_t.T)
    pts_b = np.column_stack([x.flatten(), y.flatten(), zb.flatten()])
    proj_pts_b = np.dot(pts_b - origin, normal)[:, None] * normal
    proj_pts_b = pts_b - origin - proj_pts_b
    proj_pts_b = np.matmul(change_b_mat, proj_pts_b.T)
    ax_x.fill((8., -30., -30, 2.), (0., -63., -50., 0.), color='lightgray',
              alpha=0.7, label='Main Fault Zone')
    for well, pts in well_dict.items():
        if well[0] not in ['D', 'B']:
            continue
        try:
            col = fsb_well_colors[well]
            zdr = 109
        except KeyError:
            col = csd_well_colors[well]
            # col = 'lightgray'
            zdr = 90
        # Proj
        pts = pts[:, :3]
        print(pts.shape)
        proj_pts = np.dot(pts - origin, normal)[:, None] * normal
        proj_pts = pts - origin - proj_pts
        proj_pts = np.matmul(change_b_mat, proj_pts.T)
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col,
                  linewidth=1.5, zorder=zdr)
        ax3d.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=col,
                     linewidth=1.5, zorder=zdr, s=5.)
        if well[0] == 'B':
            ax_x.plot(proj_pts[0, :], proj_pts[1, :], color=col, zorder=zdr)
        ax_map.scatter(pts[:, 0][0], pts[:, 1][0], color=col, s=15.,
                       zorder=111)
        ax_map.annotate(text=well, xy=(pts[:, 0][0], pts[:, 1][1]), fontsize=10,
                        weight='bold', xytext=(3, 0),
                        textcoords="offset points", color=col)
    # Plot fault coords and piercepoints
    plot_pierce_points(x, y, zt, strike=47, dip=57, ax=ax_fault, location='fsb')
    # Formatting
    ax3d.set_xlim([2579310, 2579355])
    ax3d.set_ylim([1247555, 1247600])
    ax3d.set_zlim([485, 530])
    # ax3d.view_init(elev=30., azim=-112)
    # ax3d.view_init(elev=75, azim=-120.)
    ax3d.view_init(elev=80, azim=-17.)
    ax3d.margins(0.)
    ax3d.set_xticks([])
    ax3d.set_xticklabels([])
    ax3d.set_yticks([])
    ax3d.set_yticklabels([])
    # ax3d.set_zticks([])
    ax3d.set_zticklabels([])
    # Overview map
    ax_map.axis('equal')
    ax_map.axis('off')
    ax_map.set_xlim([2579300, 2579353])
    ax_map.set_ylim([1247560, 1247612])
    # Fault map
    ax_fault.axis('equal')
    ax_fault.spines['top'].set_visible(False)
    ax_fault.spines['left'].set_visible(False)
    ax_fault.spines['right'].set_visible(False)
    ax_fault.spines['bottom'].set_bounds(-5, 5)
    ax_fault.tick_params(direction='in', left=False, labelleft=False)
    ax_fault.set_xticks([-5, 0, 5])
    ax_fault.set_xticklabels(['0', '5', '10'])
    ax_fault.set_xlabel('Meters')
    fig.legend()
    plt.show()
    return


def plot_4100(boreholes, inventory=None, drift_polygon=None, hull=None,
              catalog=None, filename=None, view=None, stimulation_data=None,
              circulation_data=None, plot_zones=False, dates=None, DTS_points=False,
              fractures=None):
    """
    Plot overview of 4100L with map, 3D and timeseries (if requested)

    :param boreholes: Dictionary output from lbnl.boreholes
    :param inventory: Obspy inventory
    :param drift_polygon: Path to pickled Shapely polygon
    :param hull: Path to JSON of the Trimesh for the drift
    :param catalog: Optional Obspy catalog
    :param filename: Optional path to figure file
    :param view: Optional view initialization for the 3D plot (elevation, az)
    :param stimulation_data:
    """
    # Read the polygon and alpha hull from file
    if drift_polygon:
        with open(drift_polygon, 'rb') as f:
            drift_polygon = pickle.load(f)
    if hull:
        with open(hull, 'r') as f:
            hull = trimesh.load_mesh(f, file_type='json')
    # Define injection zones of interest
    zone_dict = {'TU': [], 'TC': []}
    for key, d_list in collab_4100_zone_depths.items():
        for d in d_list:
            zone_dict[key].append(depth_to_xyz(boreholes, key, d * 0.3048))
    # fig = plt.figure(constrained_layout=False, figsize=(18, 13))
    # # fig.suptitle('Realtime MEQ: {} UTC'.format(datetime.utcnow()), fontsize=20)
    # gs = GridSpec(ncols=18, nrows=13, figure=fig)
    # axes_map = fig.add_subplot(gs[:9, :9])
    # axes_3D = fig.add_subplot(gs[:9, 9:], projection='3d')
    # axes_time = fig.add_subplot(gs[9:11, :])
    # hydro_ax = fig.add_subplot(gs[11:, :], sharex=axes_time)
    # Or individual plots
    fig_map, axes_map = plt.subplots(figsize=(10, 7))
    fig = plt.figure(figsize=(10, 7))
    axes_3D = fig.add_subplot(projection='3d')
    fig2, axes = plt.subplots(nrows=2, figsize=(18, 6), sharex=True)
    axes_time, hydro_ax = axes
    # Convert to HMC system
    if catalog:
        catalog = [ev for ev in catalog if len(ev.origins) > 0]
        catalog.sort(key=lambda x: x.origins[-1].time)
        if dates:
            catalog = [ev for ev in catalog if dates[0] < ev.picks[0].time
                       < dates[1]]
        hmc_locs = [(float(ev.preferred_origin().extra.hmc_east.value),
                     float(ev.preferred_origin().extra.hmc_north.value),
                     float(ev.preferred_origin().extra.hmc_elev.value))
                    for ev in catalog]
        colors = [date2num(ev.picks[0].time) for ev in catalog]
        times = [ev.preferred_origin().time.datetime for ev in catalog]
        mags = []
        for ev in catalog:
            if len(ev.magnitudes) > 0:
                mags.append(ev.preferred_magnitude().mag)
            else:
                mags.append(-999.)

        x, y, z = zip(*hmc_locs)
        mag_inds = np.where(np.array(mags) > -999.)
        mags = np.array(mags)[mag_inds]
        TU_center = zone_dict['TU'][0]
        dists = [np.sqrt((l[0] - TU_center[0])**2 +
                         (l[1] - TU_center[1])**2 +
                         (l[2] - TU_center[2])**2)
                 for l in hmc_locs]
        distance = np.array(dists)
    endtime = datetime.utcnow()
    starttime = endtime - timedelta(seconds=3600)
    if hull:
        axes_3D.plot_trisurf(*zip(*hull.vertices), triangles=hull.faces,
                             color='darkgray')
    if fractures:
        for frac_file in fractures:
            verts = np.loadtxt(frac_file, delimiter=',', skiprows=1)
            print(verts)
            axes_3D.plot_trisurf(verts[:, 0] * 0.3048, verts[:, 1] * 0.3048, verts[:, 2] * 0.3048, color='tan', alpha=0.5)
    if inventory:
        stations = [(float(sta.extra.hmc_east.value) * 0.3048,
                     float(sta.extra.hmc_north.value) * 0.3048,
                     float(sta.extra.hmc_elev.value))
                    for sta in inventory[0] if sta.code[-2] != 'S']
        sx, sy, sz = zip(*stations)
    for well, xyzd in boreholes.items():
        if well[0] == 'T':
            color = 'steelblue'
            linewidth = 1.0
        else:
            color = 'k'
            linewidth = 1.0
        axes_3D.plot(xyzd[:, 0], xyzd[:, 1], xyzd[:, 2], color=color,
                     linewidth=linewidth, alpha=0.8, zorder=500)
    # Plot Zone 1, 7
    if plot_zones:
        for well, zone_list in zone_dict.items():
            if well == 'TC':
                color = 'k'
            else:
                color = 'r'
            for z in zone_list:
                axes_3D.scatter(z[0], z[1], z[2], marker='*', s=100, c=color)
    if DTS_points:
        # Plot J-T frac hits
        for bh, deps in dts_hits_4100.items():
            for dep in deps:
                xyz = depth_to_xyz(boreholes, bh, dep)
                axes_3D.scatter(xyz[0], xyz[1], xyz[2], marker='x', s=100, color='firebrick')
                axes_map.scatter(xyz[0], xyz[1], marker='x', s=100, color='firebrick')
        for bh, dep_lists in dts_cooling_4100.items():
            for dep in dep_lists:
                pt1 = depth_to_xyz(boreholes, bh, dep[0])
                pt2 = depth_to_xyz(boreholes, bh, dep[1])
                axes_3D.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], linewidth=5, color='dodgerblue')
                axes_map.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], linewidth=5., color='dodgerblue')
    # Stations
    if inventory:
        axes_3D.scatter(sx, sy, sz, marker='v', color='r', label='Seismic sensor')
    if catalog:
        sizes = ((mags - np.min(mags)))**2 + 20.
        mpl = axes_3D.scatter(
            np.array(x)[mag_inds], np.array(y)[mag_inds],
            np.array(z)[mag_inds], marker='o',
            # color='k',
            c=np.array(colors)[mag_inds],
            s=sizes, alpha=0.7)
        axes_map.scatter(np.array(x)[mag_inds], np.array(y)[mag_inds],
                         marker='o',
                         c=np.array(colors)[mag_inds],
                         # color='k',
                         s=sizes)
        axes_time.scatter(
            np.array(times)[mag_inds], np.array(distance)[mag_inds],
            c=np.array(colors)[mag_inds], s=sizes)
        ax2 = axes_time.twinx()
        ax2.step(times, np.arange(len(times)), color='firebrick')
        axes_time.set_ylabel('Distance [m]', fontsize=18)
        ax2.set_ylabel('# events', fontsize=18)
        axes_time.tick_params(which='both', axis='x', labelbottom='False', labelsize=18)
        axes_time.tick_params(axis='y', labelsize=18)
        ax2.tick_params(axis='y', labelsize=18)
        plt.setp(axes_time.get_xticklabels(), visible=False)
    if type(circulation_data) == pd.DataFrame:
        # df = hydro_data[date_range[0]:date_range[1]]
        ax2 = hydro_ax.twinx()
        hydro_ax.plot(circulation_data.index, circulation_data['Net Flow'],
                      color='steelblue')
        ax2.plot(
            circulation_data.index, circulation_data['Injection Pressure'],
            color='firebrick')
        if type(stimulation_data) == pd.DataFrame:
            Q = stimulation_data.filter(like='Flow')
            quizP = stimulation_data.filter(like='Quizix P')
            Q.plot(
                ax=hydro_ax, color=sns.color_palette('Blues', 12).as_hex(),
                legend=False)
            quizP.plot(
                ax=ax2, color=sns.color_palette('Reds', 6).as_hex(),
                legend=False)
            stimulation_data['PT 403'].plot(ax=ax2, color='firebrick')
        hydro_ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        hydro_ax.set_ylabel('L/min', color='steelblue', fontsize=22)
        ax2.set_ylabel('psi', color='firebrick', fontsize=22)
        hydro_ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                             color='steelblue', labelsize=15)
        ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                        color='firebrick', labelsize=15)
        ax2.set_xlabel('Date', fontsize=24)
        # ax2.xaxis.set_major_locator(HourLocator(interval=4))
        ax2.xaxis.set_major_locator(DayLocator(interval=7))
        ax2.tick_params(axis='x', pad=15)
        plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", fontsize=18)
    hydro_ax.set_xlabel('Date', fontsize=24)
    # hydro_ax.xaxis.set_major_locator(HourLocator(interval=4))
    hydro_ax.xaxis.set_major_locator(DayLocator(interval=7))
    hydro_ax.tick_params(axis='x', pad=15)
    plt.setp(hydro_ax.get_xticklabels(), rotation=20, ha="right", fontsize=18)
    axes_3D.set_xlabel('Easting [HMC]', fontsize=18)
    axes_3D.set_ylabel('Northing [HMC]', fontsize=18)
    axes_3D.set_zlabel('Elevation [m]', fontsize=18)
    axes_3D.tick_params(which='major', labelsize=13)
    axes_3D.set_ylim([-905, -855])
    axes_3D.set_xlim([1215, 1265])
    axes_3D.set_zlim([305, 355])
    if view:
        axes_3D.view_init(*view)
    else:
        axes_3D.view_init(10, -10)
    if dates:
        axes_time.set_xlim(*dates)
    # Plot boreholes
    for well, xyzd in boreholes.items():
        if well[0] == 'T':
            color = 'steelblue'
            linewidth = 1.
        else:
            color = 'k'
            linewidth = 1.0
        axes_map.plot(xyzd[:, 0], xyzd[:, 1], color=color,
                      linewidth=linewidth, alpha=0.8, zorder=100)
    if plot_zones:
        for well, zone_list in zone_dict.items():
            if well == 'TC':
                color = 'k'
            else:
                color = 'r'
            for z in zone_list:
                axes_map.scatter(z[0], z[1], marker='*', s=100, c=color)
    if inventory:
        axes_map.scatter(sx, sy, marker='v', color='r')
    if drift_polygon:
        axes_map.add_patch(PolygonPatch(drift_polygon, fc='darkgray', ec='k'))
    # axes_map.plot(hull_pts[0, :], hull_pts[1, :], linewidth=0.9, color='k')
    axes_map.set_ylim([-920, -840])
    axes_map.set_xlim([1200, 1280])
    axes_map.set_xlabel('Easting [HMC]', fontsize=22)
    axes_map.set_ylabel('Northing [HMC]', fontsize=22)
    axes_map.tick_params(axis='both', which='major', labelsize=18)
    axes_map.set_aspect('equal')
    fig.tight_layout()
    fig_map.tight_layout()
    fig2.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.93, bottom=0.08, top=0.95, hspace=1.4)
    if filename:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    return


def get_well_piercepoint(wells):
    """
    Return the xyz points of the main fault for a list of wells

    :param wells: List
    :return:
    """
    well_dict = create_FSB_boreholes()
    pierce_dict = {}
    for well in wells:
        pierce_dict[well] = {'top': depth_to_xyz(well_dict, well,
                                                 fault_depths[well][0])}
        pierce_dict[well]['bottom'] = depth_to_xyz(well_dict, well,
                                                   fault_depths[well][1])
    return pierce_dict


def plot_pierce_points(x, y, z, strike, dip, ax, location='fsb'):
    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    origin = np.array((np.nanmean(x), np.nanmean(y), np.nanmean(z)))
    normal = np.array((np.sin(d) * np.cos(s), -np.sin(d) * np.sin(s),
                       np.cos(d)))
    strike_new = np.array([np.sin(s), np.cos(s), 0])
    up_dip = np.array([-np.cos(s) * np.cos(d), np.sin(s) * np.cos(d), np.sin(d)])
    change_B_mat = np.array([strike_new, up_dip, normal])
    grid_pts = np.subtract(np.column_stack([x, y, z]), origin)
    newx, newy, newz = change_B_mat.dot(grid_pts.T)
    newx = newx[~np.isnan(newx)]
    newy = newy[~np.isnan(newy)]
    pts = np.column_stack([newx, newy])
    hull = ConvexHull(pts)
    if location == 'fsb':
        pierce_points = get_well_piercepoint(['D1', 'D2', 'D3', 'D4', 'D5',
                                              'D6', 'D7'])
        ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color='white',
                alpha=0.0)
        size = 20.
        fs = 10
    elif location == 'surf':
        pierce_points = get_frac_piercepoint(
            ['I', 'OB', 'OT', 'P'],
            well_file='data/chet-collab/boreholes/surf_4850_wells.csv')
        size = 70.
        fs = 12
    # Plot well pierce points
    projected_pts = {}
    for well, pts in pierce_points.items():
        try:
            col = csd_well_colors[well]
        except KeyError as e:
            col = cols_4850[well]
        p = np.array(pts['top'])
        # Project onto plane in question
        proj_pt = p - (normal.dot(p - origin)) * normal
        trans_pt = proj_pt - origin
        new_pt = change_B_mat.dot(trans_pt.T)
        ax.scatter(new_pt[0], new_pt[1], marker='+', color='k', s=size,
                   zorder=103)
        ax.annotate(text=well, xy=(new_pt[0], new_pt[1]), fontsize=fs,
                    weight='bold', xytext=(3, 0),
                    textcoords="offset points", color=col)
        projected_pts[well] = new_pt
    return projected_pts


###### Adding various objects to the plotly figure #######

def add_frac_surf(strike, dip, center, radius, datas, name, color):
    """
    Add a circular plane for a fracture

    :param strike: Strike in deg from N
    :param dip: Dip in deg from horizontal
    :param center: 3D point of center of plane
    :param radius: Radius of the circle
    :return:
    """
    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    # Define fault normal
    n = np.array([np.sin(d) * np.cos(s),
                  -np.sin(d) * np.sin(s),
                  np.cos(d)])
    u = np.array([np.sin(s), np.cos(s), 0])
    # Equ from https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
    Pc = [(radius * np.cos(t) * u) +
          (radius * np.sin(t) * np.cross(n, u)) + center
          for t in np.linspace(0, 2 * np.pi, 50)]
    Pc = np.array(Pc)
    datas.append(go.Scatter3d(
        x=Pc[:, 0], y=Pc[:, 1], z=Pc[:, 2], surfaceaxis=1, surfacecolor=color,
        line=dict(color=color, width=4),
        name=name, mode='lines', opacity=0.3))
    return


def add_4850_fracs(datas, well_file):
    well_dict = parse_surf_boreholes(well_file)
    fracs = {'OT-P Connector': {'strike': 150.9, 'dip': 87.5, 'radius': 10.,
                                'center': depth_to_xyz(well_dict, 'P', 37.11),
                                'color': 'gray'},
             'W-S': {'strike': 77, 'dip': 79, 'radius': 4, 'color': '#3CB371',
                     'center': depth_to_xyz(well_dict, 'OT', 45.)},
             'W-N': {'strike': 72, 'dip': 81, 'radius': 4, 'color': '#CD5C5C',
                     'center': depth_to_xyz(well_dict, 'OT', 46.4)},
             'W-L': {'strike': 113, 'dip': 86, 'radius': 3.,
                     'center': (816.60, -1296.7, 102.18),
                     'color': '#FF4500'},
             'E-N': {'strike': 248, 'dip': 82, 'radius': 4, 'color': '#663399',
                     'center': (821.142, -1290.464, 116.281)},
             'E-S': {'strike': 60, 'dip': 88, 'radius': 4., 'color': '#1E90FF',
                     'center': (822.213, -1293.600, 106.827)}}
    for frac, sd in fracs.items():
        add_frac_surf(sd['strike'], sd['dip'], sd['center'], sd['radius'],
                      datas, frac, sd['color'])
    return


def get_plane_z(X, Y, strike, dip, point):
    """
    Helper to return the Z values of a fault/frac on a grid defined by X, Y

    :param X: Array defining the X coordinates
    :param Y: Array defining the Y coordinates
    :param strike: Strike of plane (deg clockwise from N)
    :param dip: Dip of plane (deg down from horizontal; RHR applies)
    :param point: Point that lies on the plane
    """
    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    # Define fault normal
    a, b, c = (np.sin(d) * np.cos(s), -np.sin(d) * np.sin(s), np.cos(d))
    d = (a * point[0]) + (b * point[1]) + (c * point[2])
    Z = (d - (a * X) - (b * Y)) / c
    return Z


"""
Following functions from notebook here:
https://nbviewer.jupyter.org/github/empet/Plotly-plots/blob/master/Plotly-Slice-in-volumetric-data.ipynb
"""

def get_the_slice(x, y, z, surfacecolor, name='', showscale=True):
    ticks = np.arange(-200, 200, 20)
    tick_labs = [str(t) for t in ticks]
    return go.Surface(x=x, y=y, z=z,
                      surfacecolor=surfacecolor,
                      colorbar=dict(
                          title=dict(text=r'microstrain',
                                     font=dict(size=18),
                                     side='top'),
                          ticks='outside', x=0.05, y=0.5, len=0.5,
                          ticktext=tick_labs, tickvals=ticks),
                      colorscale='RdBu', reversescale=True,
                      showscale=showscale,
                      name=name, showlegend=True)


def get_lims_colors(surfacecolor): # color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)


def get_strain(volume, gridz, planez):
    # Get the 2D index array. This is the z-index closest to the plane for
    # each X-Y pair
    inds = np.argmin(np.abs(gridz - planez), axis=2)
    # Index the 3D strain array with this index array
    strains = volume[np.arange(inds.shape[0])[:, None],
                     np.arange(inds.shape[1]), inds]
    return strains


def add_time_colored_line(objects):
    """
    Helper to create a colored line object for excavation progress at Mont Terri
    :return:
    """
    well_dict_fsb = create_FSB_boreholes()
    df_excavation = distance_to_borehole(
        well_dict_fsb, 'D7', depth=20.,
        gallery_pts='data/chet-FS-B/excavation/points_along_excavation.csv',
        excavation_times='data/chet-FS-B/excavation/G18excavationdistance.txt')
    df_excavation = df_excavation.loc[df_excavation.index.dropna()]
    pts = df_excavation[['X', 'Y', 'Z']].values
    ts = df_excavation.index
    cyan_inds = np.where(ts < datetime(2019, 5, 13))
    blue_inds = np.where((ts >= datetime(2019, 5, 13)) &
                         (ts < datetime(2019, 5, 17)))
    red_inds = np.where(ts >= datetime(2019, 5, 17))
    labs = ['Before 13 June', '13 June -- 17 June', 'After 17 June']
    for i, (col_inds, col) in enumerate([(cyan_inds, 'rgba(0, 191, 191, 1.)'),
                                         (blue_inds, 'rgba(0, 0, 255, 1.)'),
                                         (red_inds, 'rgba(255, 0, 0, 1.)')]):
        x = np.squeeze(pts[col_inds, 0])
        y = np.squeeze(pts[col_inds, 1])
        z = np.squeeze(pts[col_inds, 2])
        c = np.array([col for i in range(x.shape[0])])
        # Make the scatter obj
        scat = go.Scatter3d(x=x, y=y, z=z, mode='lines',
                            line=dict(color=col, width=10),
                            name=labs[i])
        objects.append(scat)
    return


def add_fsb_simfip(objects, well, depth_range):
    """
    Add symbol for a simfip

    :param objects: List of plotly objects to add to
    :param well: Which well is it in?
    :param depth_range: Top and bottom packer depths
    """
    well_dict = create_FSB_boreholes()
    well = well_dict[well]
    pts = well[np.where((well[:, -1] > depth_range[0]) &
                        (well[:, -1] < depth_range[1]))]
    scat = go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='lines',
                        line=dict(color='black', width=12),
                        name='SIMFIP')
    objects.append(scat)
    return


def add_DSS_volume_slices(objects, pick_dict, xrange, yrange, zrange, sampling,
                          clims=(-100, 100)):
    """
    Interpolate onto a volume between DSS measurements, then plot the top
    and bottom of the Main Fault where it intersects this volume.

    :param pick_dict:
    :param xrange:
    :param yrange:
    :param zrange:
    :param sampling:
    :param which:
    :return:
    """
    Xs = np.arange(xrange[0], xrange[1], sampling)
    Ys = np.arange(yrange[0], yrange[1], sampling)
    Zs = np.arange(zrange[0], zrange[1], sampling)
    gridx, gridy, gridz = np.meshgrid(Xs, Ys, Zs, indexing='xy', sparse=False)
    volume = interpolate_picks(pick_dict, gridx, gridy, gridz, method='linear')
    # Get z values within grid for fault/frac
    faultZ_top = get_plane_z(gridx, gridy, strike=52., dip=57.,
                             point=(2579327.55063806, 1247523.80743839,
                                    419.14869573))
    faultZ_bot = get_plane_z(gridx, gridy, strike=52., dip=57.,
                             point=(2579394.34498769, 1247583.94281201,
                                    425.28368236))
    color_top = get_strain(volume=volume, gridz=gridz, planez=faultZ_top)
    color_bot = get_strain(volume=volume, gridz=gridz, planez=faultZ_bot)
    # Use strains to mask x, y, z values
    slicez_t = faultZ_top[:, :, 0]
    slicez_t[np.where(np.isnan(color_top))] = np.nan
    slicez_b = faultZ_bot[:, :, 0]
    slicez_b[np.where(np.isnan(color_bot))] = np.nan
    slice_top = get_the_slice(Xs, Ys, slicez_t, color_top,
                              name='Main Fault Strain: Top')
    slice_bot = get_the_slice(Xs, Ys, slicez_b, color_bot,
                              name='Main Fault Strain: Bottom',
                              showscale=False)
    if not clims:
        cmin_t, cmax_t = get_lims_colors(color_top)
        cmin_b, cmax_b = get_lims_colors(color_bot)
        slice_top.update(cmin=np.min([cmin_t, cmin_b]),
                         cmax=np.max([cmax_t, cmax_b]))
        slice_bot.update(cmin=np.min([cmin_t, cmin_b]),
                         cmax=np.max([cmax_t, cmax_b]))
    else:
        slice_top.update(cmin=clims[0], cmax=clims[1])
        slice_bot.update(cmin=clims[0], cmax=clims[1])
    objects.extend([slice_top, slice_bot])
    return objects


def add_wells(well_dict, objects, structures, wells):
    well_colors = cycle(sns.color_palette().as_hex())
    for i, (key, pts) in enumerate(well_dict.items()):
        if key in wells or wells == 'all':
            try:
                x, y, z = zip(*pts)
            except ValueError:
                x, y, z, d = zip(*pts)
        else:
            continue
        if key in cols_4850:
            col = cols_4850[key]
        elif structures or wells != 'all':
            col = 'gray'
        else:
            col = next(well_colors)
        if key.startswith('D') and len(key) == 2:
            group = 'CSD'
            viz = True
        elif key.startswith('B'):
            group = 'FS-B'
            viz = True
        elif len(key) == 1:
            group = 'FS'
            viz = True
        elif key[0] in ['O', 'P', 'I', 'P', 'D', 'T', 'A']:
            group = 'Collab'
            viz = True
        else:
            group = 'Other projects'
            viz = False
        objects.append(go.Scatter3d(x=np.array([x[0], x[-1]]),
                                    y=np.array([y[0], y[-1]]),
                                    z=np.array([z[0], z[-1]]),
                                    mode='lines',
                                    visible=viz,
                                    name='{}: {}'.format(group, key),
                                    line=dict(color=col, width=6),
                                    hoverinfo='skip'))
    return objects


def add_DSS(DSS_picks, objects, well_dict):
    # Over each well
    frac_list = []
    for well, pick_dict in DSS_picks.items():
        easts, norths, zs, deps = np.hsplit(well_dict[well], 4)
        if well.startswith('D'):  # Scale CSD signal way down visually
            loc = 1
            scale = 2.
        elif well.startswith('B'):
            loc = 2
            scale = 1.1
        # Over each picked feature
        for i, dep in enumerate(pick_dict['depths']):
            if dep < 5.:
                # Crude skip of shallow anomalies that overrun everything
                continue
            dists = np.squeeze(np.abs(dep - deps))
            x = easts[np.argmin(dists)][0]
            y = norths[np.argmin(dists)][0]
            z = zs[np.argmin(dists)][0]
            strain = pick_dict['strains'][i]
            width = pick_dict['widths'][i]
            frac_list.append((x, y, z, strain, width, loc, scale))
    fracx, fracy, fracz, strains, fracw, locs, scales = zip(*frac_list)
    scales = 1 / np.array(scales)
    ticks = np.arange(-200, 200, 20)
    tick_labs = [str(t) for t in ticks]
    # Add to plot
    objects.append(go.Scatter3d(
        x=np.array(fracx), y=np.array(fracy),
        z=np.array(fracz),
        mode='markers',
        legendgroup='DSS',
        name='DSS picks',
        hoverinfo='text',
        text=strains,
        visible='legendonly',
        marker=dict(
            color=strains, cmin=-200., cmax=200.,
            size=np.abs(np.array(strains))**scales,
            symbol='circle',
            line=dict(color=strains, width=1,
                      colorscale='RdBu'),
            colorbar=dict(
                title=dict(text=r'microstrain',
                           font=dict(size=18),
                           side='top'),
                ticks='outside', x=0.05, y=0.5, len=0.5,
                ticktext=tick_labs, tickvals=ticks),
            colorscale='RdBu', reversescale=True,
            opacity=0.9)))
    return objects


def read_mesh(fsb_mesh_file):
    """Helper to read verts and triangles from gocad formatted mesh files"""
    mesh = pd.read_csv(fsb_mesh_file, header=None, delimiter=' ')
    vertices = mesh[mesh.iloc[:, 0] == 'VRTX']
    triangles = mesh[mesh.iloc[:, 0] == 'TRGL']
    X = vertices.iloc[:, 1].values
    Y = vertices.iloc[:, 2].values
    Z = vertices.iloc[:, 3].values
    I = triangles.iloc[:, 1].values - 1
    J = triangles.iloc[:, 2].values - 1
    K = triangles.iloc[:, 3].values - 1
    return X, Y, Z, I, J, K


def add_meshes(meshes, objects):
    fault_colors = cycle(sns.xkcd_palette(['tan', 'light brown']).as_hex())
    other_colors = cycle(sns.xkcd_palette(['pale purple']).as_hex())
    for mesh_name, mesh_file in meshes:
        if 'FAULT' in mesh_name.upper():
            col = next(fault_colors)
        else:
            col = next(other_colors)
        # SURF vtk courtesy of Pengcheng
        if mesh_file.endswith(".dxf"):
            dxf_to_xyz(mesh_file, mesh_name, objects)
        else:
            X, Y, Z, I, J, K = read_mesh(mesh_file)
            objects.append(go.Mesh3d(
                x=X, y=Y, z=Z, i=I, j=J, k=K,
                name=mesh_name,
                color=col, opacity=0.3,
                alphahull=0,
                showlegend=True,
                hoverinfo='skip'))
    return objects


def add_drift_2D(shapely_pickle, objects):
    with open(shapely_pickle, 'rb') as f:
        poly = pickle.load(f)
    XY = np.array(poly.exterior.coords)
    Z = np.zeros(XY.shape[0]) + 343.5
    print(XY)
    print(Z)
    surf = go.Scatter3d(x=XY[:, 0], y=XY[:, 1], z=Z,
                        line=dict(color='black', width=5),
                        showlegend=False, mode='lines')
    objects.append(surf)
    return


def add_surface(surface_file, objects):
    """
    Read Tanners CASSM results and add to plotly
    """
    cassm = loadmat(surface_file)['DataSet']
    for i in range(6):
        surf = go.Surface(x=cassm[:, 0].reshape(76, 116),
                          y=cassm[:, 1].reshape(76, 116),
                          z=cassm[:, 2].reshape(76, 116),
                          surfacecolor=cassm[:, 3+i].reshape(76, 116),
                          showlegend=True,
                          name='Cycle {}'.format(i),
                          opacity=0.5, hoverinfo='skip',
                          colorbar=dict(x=1.02, len=0.5, y=0.25,
                                        title='dVp [m/s]'),
                          colorscale='Cividis', cmin=-16, cmax=5)
        objects.append(surf)
    return

def add_fault(x, y, z, objects):
    fault = go.Mesh3d(x=x, y=y, z=z,
                      showlegend=True,
                      name='Fracture?',
                      opacity=0.7, hoverinfo='skip',
                      color='burlywood')
    objects.append(fault)
    return

def add_structures(structures, objects, well_dict):
    struct_files = glob('{}/**/B*_structures.xlsx'.format(structures),
                        recursive='True')
    used_ftype = []
    for struct_file in struct_files:
        frac_planes = structures_to_planes(struct_file, well_dict)
        for X, Y, Z, ftype, color in frac_planes:
            if (Z > 550).any():  # One strange foliation?
                continue
            if ftype in used_ftype:
                objects.append(go.Mesh3d(
                    x=X, y=Y, z=Z, name=ftype,
                    color=color, opacity=0.3,
                    visible='legendonly',
                    delaunayaxis='z', text=ftype,
                    legendgroup='frac_logs',
                    showlegend=False,
                    hoverinfo='skip'))
            else:
                objects.append(go.Mesh3d(
                    x=X, y=Y, z=Z, name=ftype,
                    color=color, opacity=0.3,
                    visible='legendonly',
                    delaunayaxis='z', text=ftype,
                    legendgroup='frac_logs',
                    showlegend=True,
                    hoverinfo='skip'))
                used_ftype.append(ftype)
    return objects


def add_inventory(inventory, location, objects):
    """
    Handle adding of inventory scatter object to 3D viz

    :param inventory: obspy Inventory
    :param location: 'surf' or 'fsb'
    :param objects: list of plotly objects
    :return:
    """
    fsb_accel = ['B31', 'B34', 'B42', 'B43', 'B551', 'B585', 'B647', 'B659',
                 'B748', 'B75']
    surf_accel = ['PDB3', 'PDB4', 'PDB6', 'PDT1', 'PSB7', 'PSB9', 'PST10',
                  'PST12', 'AMU1', 'AMU2', 'AMU3', 'AMU4', 'AML1', 'AML2', 'AML3', 'AML4',
                  'DMU1', 'DMU2', 'DMU3', 'DMU4', 'DML1', 'DML2', 'DML3', 'DML4']
    sta_list = []
    if isinstance(inventory, dict):
        for sta, pt in inventory.items():
            (sx, sy, sz) = pt
            sta_list.append((sx, sy, sz, sta))
    else:
        # Do the same for the inventory
        for sta in inventory[0]:  # Assume single network for now
            if location in ['4850', '4100']:
                loc_key = 'hmc'
            elif location == 'fsb':
                loc_key = 'ch1903'
            else:
                print('Location {} not supported'.format(location))
                raise KeyError
            if sta.code.startswith('S'):
                legend = 'CASSM Source'
                color = 'blue'
                symbol = 'circle'
            elif sta.code in fsb_accel or sta.code in surf_accel:
                legend = 'Accelerometer'
                color = 'magenta'
                symbol = 'square'
            elif len(sta.code) == 3:
                legend = 'AE sensor'
                color = 'red'
                symbol = 'diamond'
            elif sta.code[-2] == 'S':
                legend = 'CASSM Source'
                color = 'red'
                symbol = 'cross'
            else:
                legend = 'Hydrophone'
                color = 'green'
                symbol = 'square'
            sx = float(sta.extra['{}_east'.format(loc_key)].value)
            sy = float(sta.extra['{}_north'.format(loc_key)].value)
            sz = float(sta.extra['{}_elev'.format(loc_key)].value)
            name = sta.code
            sta_list.append((sx, sy, sz, name, legend, color, symbol))
            if location == '4100':
                scale = 0.3048
            else:
                scale = 1
    _, _, _, _, leg, _, _ = zip(*sta_list)
    for sensor_type in list(set(leg)):
        stax, stay, staz, nms, leg, col, sym = zip(*[sta for sta in sta_list if
                                                     sta[4] == sensor_type])
        objects.append(go.Scatter3d(x=np.array(stax) * scale, y=np.array(stay) * scale,
                                    z=np.array(staz),
                                    mode='markers',
                                    name=sensor_type,
                                    legendgroup='Seismic network',
                                    hoverinfo='text',
                                    text=nms,
                                    marker=dict(color=col,
                                                size=3.,
                                                symbol=sym,
                                                line=dict(color=col,
                                                        width=1),
                                                opacity=0.9)))
    return objects


def add_surf_sources(well_dict, objects):
    surf_cassm_deps = {'OB': [30.1, 34.2, 38.5, 47., 51.5],
                       'PST': [19.7, 22.9, 25.7, 28.7, 32., 35.1],
                       'PSB': [9.1, 19.8, 30.5, 41.1, 51.8], 'PDT': [30.5]}
    sta_list = []
    # Add CASSM sources too
    for bh, deps in surf_cassm_deps.items():
        for d in deps:
            x, y, z = depth_to_xyz(well_dict, bh, d)
            sta_list.append((x, y, z))
    xs, ys, zs = zip(*sta_list)
    objects.append(go.Scatter3d(
        x=np.array(xs), y=np.array(ys), z=np.array(zs), mode='markers',
        name='CASSM', legendgroup='Seismic network',
        marker=dict(color='black', size=2., symbol='x',
        opacity=0.9)))
    return objects


def add_catalog(catalog, dd_only, objects, surface, location=None):
    # Establish color scales from colorlover (import colorlover as cl)
    colors = cycle(cl.scales['11']['qual']['Paired'])
    pt_lists = []
    pt_list = []
    if len(catalog) < 100:
        mfact = 3.
    elif len(catalog) < 1000:
        mfact = 1.5
    else:
        mfact = 0.5
    for ev in catalog:
        try:
            # o = ev.preferred_origin()
            o = ev.origins[-1]
        except IndexError:
            continue
        if location == 'fsb':
            try:
                ex = float(o.extra.ch1903_east.value)
                ey = float(o.extra.ch1903_north.value)
                ez = float(o.extra.ch1903_elev.value)
            except AttributeError:  # Case of only dug-seis location
                try:
                    ex = float(ev.extra.x.value)
                    ey = float(ev.extra.y.value)
                    ez = float(o.depth) - 500.
                except AttributeError:
                    continue
        elif location == 'surf':
            ex = float(o.extra.hmc_east.value)
            ey = float(o.extra.hmc_north.value)
            ez = float(o.extra.hmc_elev.value)
        else:
            ex = float(o.longitude)
            ey = float(o.latitude)
            ez = float(o.depth)
        if dd_only and not o.method_id:
            print('Not accepting non-dd locations')
            continue
        elif dd_only and not o.method_id.id.endswith('GrowClust'):
            print('Not accepting non-GrowClust locations')
            continue
        try:
            m = ev.magnitudes[-1].mag
        except IndexError:
            # Default to M 1
            m = 1
        t = o.time.datetime.timestamp()
        pt_list.append((ex, ey, ez, m, t,
                        ev.resource_id.id.split('/')[-1]))
    # if len(pt_list) > 0:
    pt_lists.append(pt_list)
    # Add arrays to the plotly objects
    for i, lst in enumerate(pt_lists):
        if len(lst) == 0:
            continue
        x, y, z, m, t, id = zip(*lst)
        # z = -np.array(z)
        clust_col = next(colors)
        tickvals = np.linspace(min(t), max(t), 10)
        ticktext = [datetime.fromtimestamp(t).strftime('%d %b %Y: %H:%M')
                    for t in tickvals]
        scat_obj = go.Scatter3d(x=np.array(x), y=np.array(y), z=np.array(z),
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
                                            colorscale='Plotly3',
                                            opacity=0.5))
        objects.append(scat_obj)
        if surface == 'plane':
            if len(x) <= 2:
                continue  # Cluster just 1-2 events
            # Fit plane to this cluster
            X, Y, Z, stk, dip = pts_to_plane(np.array(x), np.array(y),
                                             np.array(z))
            # Add mesh3d object to plotly
            objects.append(go.Mesh3d(x=X, y=Y, z=Z, color=clust_col,
                                     opacity=0.3, delaunayaxis='z',
                                     text='Strike: {}, Dip {}'.format(stk, dip),
                                     showlegend=True))
        elif surface == 'ellipsoid':
            if len(x) <= 2:
                continue  # Cluster just 1-2 events
            # Fit plane to this cluster
            center, radii, evecs, v = pts_to_ellipsoid(np.array(x),
                                                       np.array(y),
                                                       np.array(z))
            X, Y, Z = ellipsoid_to_pts(center, radii, evecs)
            # Add mesh3d object to plotly
            objects.append(go.Mesh3d(x=X, y=Y, z=Z, color=clust_col,
                                     opacity=0.3, delaunayaxis='z',
                                     showlegend=True))
        else:
            print('No surfaces fitted')
    return objects


def dxf_to_mpl(path):
    """Return xyz and triangles for mpl3d"""
    dxf = dxfgrabber.readfile(path)
    xs = []
    ys = []
    zs = []
    tris = []
    j = 0
    for obj in dxf.entities:
        x = np.array([p[0] / 3.28084 for p in obj.points[:-1]])
        y = np.array([p[1] / 3.28084 for p in obj.points[:-1]])
        z = np.array([p[2] / 3.28084 for p in obj.points[:-1]])
        if  not (np.all((780 < x) & (x <= 860)) and
                 np.all((-1350 < y) & (y <=-1270)) and
                 np.all((60 < z) & (z <= 140))):
            continue
        xs.extend(x.tolist())
        ys.extend(y.tolist())
        zs.extend(z.tolist())
        tris.append((j * 3, (j * 3) + 1, (j * 3) + 2))
        j += 1
    return xs, ys, zs, tris


def dxf_to_xyz(mesh_file, mesh_name, datas):
    """Helper for reading dxf files of surf levels to xyz"""
    # SURF CAD files
    dxf = dxfgrabber.readfile(mesh_file)
    if dxf.entities[0].dxftype == '3DFACE':
        xs = []
        ys = []
        zs = []
        iz = []
        jz = []
        kz = []
        for i, obj in enumerate(dxf.entities):
            xs.extend([p[0] for p in obj.points[:-1]])
            ys.extend([p[1] for p in obj.points[:-1]])
            zs.extend([p[2] for p in obj.points[:-1]])
            iz.append(i * 3)
            jz.append((i * 3) + 1)
            kz.append((i * 3) + 2)
        datas.append(go.Mesh3d(x=np.array(xs) / 3.28084,
                               y=np.array(ys) / 3.28084,
                               z=np.array(zs) / 3.28084,
                               i=iz, j=jz, k=kz,
                               color='gray',
                               name=mesh_name, opacity=0.5))
    else:
        for obj in dxf.entities:
            xs = []
            ys = []
            zs = []
            if obj.dxftype == 'POLYLINE':
                for pt in obj.points:
                    x, y, z = pt[:3]
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                datas.append(go.Scatter3d(
                    x=np.array(xs) / 3.28084, y=np.array(ys) / 3.28084,
                    z=(np.array(zs) / 3.28084) - 2.,
                    name=mesh_name,
                    mode='lines',
                    line={'color': 'black', 'width': 7.},
                    opacity=1.,
                    showlegend=False))
            elif obj.dxftype == 'LINE':
                sx, sy, sz = obj.start[:3]
                ex, ey, ez = obj.end[:3]
                xs.append(sx)
                ys.append(sy)
                zs.append(sz)
                xs.append(ex)
                ys.append(ey)
                zs.append(ez)
                datas.append(go.Scatter3d(
                    x=np.array(xs) / 3.28084, y=np.array(ys) / 3.28084,
                    z=np.array(zs) / 3.28084,
                    name=mesh_name,
                    mode='lines',
                    opacity=1.,
                    line={'color': 'black', 'width':7.},
                    showlegend=False))
    return datas


def pts_to_plane(x, y, z, method='lstsq'):
    # Create a grid over the desired area
    # Here just define it over the x and y range of the cluster (100 pts)
    x_ran = max(x) - min(x)
    y_ran = max(y) - max(y)
    if method == 'lstsq':
        # Add 20 percent to x and y dimensions for viz purposes
        X, Y = np.meshgrid(np.arange(min(x) - (0.2 * x_ran),
                                     max(x) + (0.2 * x_ran),
                                     (max(x) - min(x)) / 10.),
                           np.arange(min(y) - (0.2 * y_ran),
                                     max(y) + (0.2 * y_ran),
                                     (max(y) - min(y)) / 10.))
        # Now do the linear fit and generate the coefficients of the plane
        A = np.c_[x, y, np.ones(len(x))]
        C, _, _, _ = lstsq(A, z)  # Coefficients (also the vector normal?)
    elif method == 'svd':
        print('SVD not implemented')
        return
    # Evaluate the plane for the points of the grid
    Z = C[0] * X + C[1] * Y + C[2]
    # strike and dip
    pt1 = (X[0][2], Y[0][2], Z[0][2])
    pt2 = (X[3][1], Y[3][1], Z[3][1])
    pt3 = (X[0][0], Y[0][0], Z[0][0])
    strike, dip = strike_dip_from_pts(pt1, pt2, pt3)
    return X.flatten(), Y.flatten(), Z.flatten(), strike, dip


def pts_to_ellipsoid(x, y, z):
    # Function from:
    # https://github.com/aleksandrbazhin/ellipsoid_fit_python/blob/master/ellipsoid_fit.py
    # http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
    # for arbitrary axes
    D = np.array([x * x,
                  y * y,
                  z * z,
                  2 * x * y,
                  2 * x * z,
                  2 * y * z,
                  2 * x,
                  2 * y,
                  2 * z])
    DT = D.conj().T
    v = np.linalg.solve(D.dot(DT), D.dot(np.ones(np.size(x))))
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], -1]])
    center = np.linalg.solve(- A[:3, :3], [[v[6]], [v[7]], [v[8]]])
    T = np.eye(4)
    T[3, :3] = center.T
    R = T.dot(A).dot(T.conj().T)
    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    radii = np.sqrt(1. / np.abs(evals)) # Absolute value to eliminate imaginaries?
    return center, radii, evecs, v


def strike_dip_from_pts(pt1, pt2, pt3):
    # Take the output from the best fit plane and calculate strike and dip
    vec_1 = np.array(pt3) - np.array(pt1)
    vec_2 = np.array(pt3) - np.array(pt2)
    U = np.cross(vec_1, vec_2)
    # Standard rectifying for right-hand rule
    if U[2] < 0:
        easting = U[1]
        northing = -U[0]
    else:
        easting = -U[1]
        northing = U[0]
    if easting >= 0:
        partA_strike = easting**2 + northing**2
        strike = np.rad2deg(np.arccos(northing / np.sqrt(partA_strike)))
    else:
        partA_strike = northing / np.sqrt(easting**2 + northing**2)
        strike = 360. - np.rad2deg(np.arccos(partA_strike))
    part1_dip = np.sqrt(U[1]**2 + U[0]**2)
    part2_dip = np.sqrt(part1_dip**2 + U[2]**2)
    dip = np.rad2deg(np.arcsin(part1_dip / part2_dip))
    return strike, dip


def ellipsoid_to_pts(center, radii, evecs):
    """
    Take the center and radii solved for in pts_to_ellipsoid and convert them
    to a bunch of points to be meshed by plotly
    :param center: Center of ellipsoid
    :param radii: Radii of ellipsoid
    :return:
    """
    center = center.flatten()
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    # cartesian coordinates that correspond to the spherical angles:
    X = radii[0] * np.outer(np.cos(u), np.sin(v))
    Y = radii[1] * np.outer(np.sin(u), np.sin(v))
    Z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(X)):
        for j in range(len(X)):
            [X[i, j], Y[i, j], Z[i, j]] = np.dot([X[i, j], Y[i, j], Z[i, j]],
                                                 evecs) + center
    return X.flatten(), Y.flatten(), Z.flatten()


def alpha_shape(points, alpha):
    """
    Stolen: https://gist.github.com/dwyerk/10561690

    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.simplices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points



# Amplify overview figures

field_locations = {
    'JV': (-117.476, 40.181),
    'TM': (-117.687, 39.672),
    'PAT': (-119.075, 39.582),
    'DAC': (-118.327, 38.837),
    'COSO': (-117.796, 36.019)
}


def write_bbox_shp(extents, filename):
    schema = {'geometry': 'Polygon',
              'properties': [('Name', 'str')]
    }
    polyshp = fiona.open(filename, 'w',
                         driver='ESRI Shapefile', schema=schema, crs='EPSG:4326')
    rowDict = {
        'geometry': {'type': 'Polygon',
                     'coordinates': [extents]},
        'properties': {'Name': ''},
    }
    polyshp.write(rowDict)
    polyshp.close()
    return


def clip_raster(gdf, img):
     clipped_array, clipped_transform = msk.mask(
         img, [mapping(gdf.iloc[0]['geometry'])], crop=True)
     clipped_array, clipped_transform = msk.mask(
         img, [mapping(gdf.iloc[0]['geometry'])],
         crop=True, nodata=(np.amax(clipped_array[0]) + 1))
     out_meta = img.meta
     out_meta.update({
         'driver': 'GTiff',
         'height': clipped_array.shape[1],
         'width': clipped_array.shape[2],
         'transform': clipped_transform
     })
     clipped_array[0] = clipped_array[0] + abs(np.amin(clipped_array))
     value_range = np.amax(clipped_array) + abs(np.amin(clipped_array))
     return clipped_array, out_meta, value_range


def plot_catalog(catalog, dem_dir, vector_dir):
    """
    Plot catalog on top of imagery
    """
    dem_file = glob('{}/*mea075.tif'.format(dem_dir))[0]
    overview = rasterio.open(dem_file)
    ca_agua = glob('{}/CA_Lakes/*.shp'.format(vector_dir))[0]
    nv_agua = glob('{}/nv_water/*.shp'.format(vector_dir))[0]
    nv_roads = glob('{}/tl_2021_06_prisecroads/*.shp'.format(vector_dir))[0]
    ca_roads = glob('{}/tl_2021_32_prisecroads/*.shp'.format(vector_dir))[0]
    towns = glob('{}/USA_Major_Cities/*.shp'.format(vector_dir))[0]
    map_box_shp = glob('{}/*extent_v2.shp'.format(vector_dir))[0]
    borders = glob('{}/ne_50m_admin_1_states_provinces/*.shp'.format(vector_dir))[0]
    # Read in various shapefiles
    df = gpd.read_file(map_box_shp)
    nv = gpd.read_file(borders)
    nv_water = gpd.read_file(nv_agua).to_crs(4326)
    ca_water = gpd.read_file(ca_agua).to_crs(4326)
    nv_roads = gpd.read_file(nv_roads).to_crs(4326)
    ca_roads = gpd.read_file(ca_roads).to_crs(4326)
    towns = gpd.read_file(towns).to_crs(4326)
    # Filter out lesser features
    nv = nv.loc[nv['iso_3166_2'] == 'US-NV']
    topo, meta, value_range = clip_raster(df, overview)
    extent = plotting_extent(topo[0], meta['transform'])
    # Hillshade
    hillshade = es.hillshade(topo[0].copy(), azimuth=90, altitude=20)

    return


def plot_numo(catalog, dem_dir, vector_dir=None, plot_focmecs=False):
    """

    :param catalog:
    :param dem_dir:
    :param vector_dir:
    :return:
    """
    numo_location = (-121.572758, 36.873258)
    numo_extents = [(-121.70, 37.05), (-121.70, 36.70),
                     (-121.35, 36.70), (-121.35, 37.05)]
    dem_file = glob('{}/NUMO_merged_DEM.tif'.format(dem_dir))[0]
    overview = rasterio.open(dem_file)
    write_bbox_shp(numo_extents, './tmp_bbox.shp')
    bbox = gpd.read_file('./tmp_bbox.shp')
    topo, meta, value_range = clip_raster(bbox, overview)
    extent = plotting_extent(topo[0], meta['transform'])
    # Hillshade
    hillshade = es.hillshade(topo[0].copy(), azimuth=90, altitude=20)
    # Vectors
    ca_highways = glob('{}/NUMO_highways/*.shp'.format(vector_dir))[0]
    ca_cities = glob('{}/City_Boundaries/*.shp'.format(vector_dir))[0]
    ca_faults = glob('{}/NUMO_Faults/*.shp'.format(vector_dir))[0]
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    # Only top half of colormap
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap = plt.get_cmap('gist_earth')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    cmap2 = LinearSegmentedColormap.from_list('Upper Half', colors)
    # Bottom up, first DEM
    ax.imshow(topo[0], cmap=cmap2, extent=extent, alpha=0.3)
    # Then hillshade
    ax.imshow(hillshade, cmap="Greys", alpha=0.3, extent=extent)
    # Plot vector layers
    ca_highways = gpd.read_file(ca_highways).to_crs(4326)
    ca_cities = gpd.read_file(ca_cities).to_crs(4326)
    ca_faults = gpd.read_file(ca_faults).to_crs(4326)
    # Plot them
    ca_cities.plot(ax=ax, facecolor='dimgray', edgecolor="none", alpha=0.5)
    ca_highways.plot(ax=ax, linewidth=1., color='black')
    ca_faults.plot(ax=ax, linewidth=1., color='firebrick')
    ax.scatter(numo_location[0], numo_location[1], marker='^', color='r', s=100, zorder=400, edgecolor='k')
    # ax.annotate('NUMO', xy=numo_location, xytext=(5, 5), textcoords='offset points',
    #             fontsize=26, fontweight='bold')
    # Now make the beachballs
    catalog.events.sort(key=lambda x: x.preferred_magnitude().mag)
    cnums = [date2num(ev.preferred_origin().time) for ev in catalog]
    cmap = cm.magma
    norm = Normalize(vmin=min(cnums), vmax=max(cnums))
    for ev in catalog:
        o = ev.preferred_origin()
        m = ev.preferred_magnitude().mag
        c = cmap(norm(date2num(o.time)))
        if plot_focmecs:
            if len(ev.focal_mechanisms) > 0:
                fm = ev.focal_mechanisms[0]
                try:
                    tens = fm.moment_tensor.tensor
                    mt = [tens.m_rr, tens.m_tt, tens.m_pp, tens.m_rt, tens.m_rp, tens.m_tp]
                    beach_ = beach(mt, width=m**2 * 0.003, xy=(o.longitude, o.latitude), facecolor=c)
                except AttributeError:
                    np1 = fm.nodal_planes.nodal_plane_1
                    if hasattr(fm, "_beachball"):
                        beach_ = copy.copy(fm._beachball)
                    else:
                        beach_ = beach([np1.strike, np1.dip, np1.rake], facecolor=c,
                                       width=m**2 * 0.003, xy=(o.longitude, o.latitude))
                ax.add_collection(beach_)
                if m > 2.5:
                    ax.annotate(ev.resource_id.id.split('=')[-2].split('&')[0], xy=(o.longitude, o.latitude),
                                xytext=(m*6, m*6), textcoords="offset points", fontsize=12, fontstyle='italic')
        else:
            ax.scatter(o.longitude, o.latitude, color=c, s=m**2 * 2)
    # Scale bar
    points = gpd.GeoSeries([Point(-121.6, extent[2]),
                            Point(-121.65, extent[2])], crs=4326)
    points = points.to_crs(32611)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    # ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    ax.set_xlabel('Longitude [$^o$]')
    ax.set_ylabel('Latitude [$^o$]')
    ax.set_aspect('equal')
    fig.colorbar(cm.ScalarMappable(norm, cmap), location='bottom', shrink=0.5, fraction=0.1, pad=0.1)
    cax = fig.axes[-1]
    cbar_labs = cax.get_xticklabels()
    new_labs = [num2date(int(l.get_text())).strftime('%Y/%m/%d') for l in cbar_labs]
    cax.set_xticklabels(new_labs, rotation=30, horizontalalignment='right')
    plt.show()
    return


def plot_amplify_sites(dem_dir, vector_dir, catalog=None, outfile=None):
    """
    Plot figures of the Amplify fields

    Using post here:
    https://towardsdatascience.com/creating-beautiful-topography-maps-with-python-efced5507aa3
    """
    dem_file = glob('{}/*mea075.tif'.format(dem_dir))[0]
    overview = rasterio.open(dem_file)
    ca_agua = glob('{}/CA_Lakes/*.shp'.format(vector_dir))[0]
    nv_agua = glob('{}/nv_water/*.shp'.format(vector_dir))[0]
    nv_roads = glob('{}/tl_2021_06_prisecroads/*.shp'.format(vector_dir))[0]
    ca_roads = glob('{}/tl_2021_32_prisecroads/*.shp'.format(vector_dir))[0]
    towns = glob('{}/USA_Major_Cities/*.shp'.format(vector_dir))[0]
    map_box_shp = glob('{}/*extent_v3.shp'.format(vector_dir))[0]
    borders = glob('{}/ne_50m_admin_1_states_provinces/*.shp'.format(vector_dir))[0]
    # Read in various shapefiles
    df = gpd.read_file(map_box_shp)
    nv = gpd.read_file(borders)
    nv_water = gpd.read_file(nv_agua).to_crs(4326)
    ca_water = gpd.read_file(ca_agua).to_crs(4326)
    nv_roads = gpd.read_file(nv_roads).to_crs(4326)
    ca_roads = gpd.read_file(ca_roads).to_crs(4326)
    towns = gpd.read_file(towns).to_crs(4326)
    # Filter out lesser features
    nv = nv.loc[nv['iso_3166_2'] == 'US-NV']
    topo, meta, value_range = clip_raster(df, overview)
    extent = plotting_extent(topo[0], meta['transform'])
    # Hillshade
    hillshade = es.hillshade(topo[0].copy(), azimuth=90, altitude=45)
    # Figure setup
    fig, ax = plt.subplots(figsize=(20, 20))
    # Only top half of colormap
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap = plt.get_cmap('gist_earth')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    cmap2 = LinearSegmentedColormap.from_list('Upper Half', colors)
    # Bottom up, first DEM
    ax.imshow(topo[0], cmap=cmap2, extent=extent, alpha=0.3)
    # Then hillshade
    ax.imshow(hillshade, cmap="Greys", alpha=0.3, extent=extent)
    # Now water
    nv_water.loc[nv_water.TYPE.isin(['Major Lake', 'Major River',
                                     'Major Reservoir'])].boundary.plot(
        ax=ax, facecolor='steelblue', edgecolor="none", alpha=0.5)
    ca_water.loc[ca_water.TYPE.isin(['perennial'])].boundary.plot(
        ax=ax, facecolor='steelblue', edgecolor="none", alpha=0.5)
    # Roads
    nv_roads.loc[nv_roads.RTTYP.isin(['I', 'U'])].plot(
        ax=ax, column='RTTYP', linewidth=1., color='firebrick')
    ca_roads.loc[ca_roads.RTTYP.isin(['I', 'U'])].plot(
        ax=ax, column='RTTYP', linewidth=1., color='firebrick')
    big_towns = towns.loc[towns.NAME.isin(['Reno', 'Carson City',
                                           'Fresno', 'Fernley'])]
    big_towns.plot(ax=ax, color='k', markersize=5.)
    # Label cities
    for x, y, label in zip(big_towns.geometry.x, big_towns.geometry.y, big_towns.NAME):
        if label == 'Fernley':
            xytext = (-50, 5)
        else:
            xytext = (3, 3)
        ax.annotate(label, xy=(x, y), xytext=xytext, textcoords="offset points",
                    fontsize=15, fontstyle='italic', fontweight='bold')
    nv.boundary.plot(ax=ax, linewidth=1.0, linestyle='--', color='k')
    # Annotate border
    ax.annotate('Nevada', xy=(-119.5, 38.5), xytext=(10, 21),
                textcoords="offset points", fontsize=18, fontstyle='italic',
                rotation=-43)
    ax.annotate('California', xy=(-119.5, 38.5), xytext=(-15, -7),
                textcoords="offset points", fontsize=18, fontstyle='italic',
                rotation=-43)
    # Geothermal fields
    for lab, loc in field_locations.items():
        ax.scatter(loc[0], loc[1], marker='s', color='k', s=20)
        ax.annotate(lab, xy=loc, xytext=(5, 5), textcoords='offset points',
                    fontsize=26, fontweight='bold')
    # Seismic catalog
    if catalog:
        locs = np.array([[ev.origins[-1].longitude, ev.origins[-1].latitude,
                          int(ev.comments[-1].text.split('=')[-1])]
                         for ev in catalog])
        ax.scatter(locs[:, 0], locs[:, 1], marker='o', color='k',
                   s=locs[:, 2] / 3)
    # Inset map
    ax2 = inset_axes(ax, width=5, height=5, loc=2,
                     axes_class=GeoAxes,
                     axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
    # ax2.stock_img()
    ax2.add_feature(cf.COASTLINE)
    ax2.add_feature(cf.BORDERS)
    ax2.add_feature(cf.OCEAN)
    ax2.add_feature(cf.LAND)
    ax2.add_feature(cf.ShapelyFeature(df.geometry, crs=ccrs.CRS('epsg:4326')),
                    edgecolor='r', linewidth=2., facecolor='none')
    ax2.add_feature(
        cf.NaturalEarthFeature(
            category='cultural', name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none'
        )
    )
    ax2.set_extent([-128, -109, 24, 49], crs=ccrs.PlateCarree())
    # Scale bar
    points = gpd.GeoSeries([Point(-117., extent[2]),
                            Point(-118., extent[2])], crs=4326)
    points = points.to_crs(32611)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel(r'Longitude [$^o$]', fontsize=20)
    ax.set_ylabel(r'Latitude [$^o$]', fontsize=20)
    if outfile:
        plt.savefig(outfile, dpi=200)
        plt.close('all')
    else:
        plt.show()
    return


def plot_patua(dem_dir, vector_dir, inventory, catalog):
    """Patua overview plot"""
    # Station --> RT130 serial mapping
    RT130_serial = {
        '2115': 'ABB6',
        '2221': 'B2A1',
        '2128': 'AAE2',
        '5230': 'AC06',
        '4509': 'B2C8',
        '23A-17': 'B2C8'
    }
    patua_extents = [(-119.17, 39.63), (-119.17, 39.51),
                     (-119.01, 39.51), (-119.01, 39.63)]
    dem_file = glob('{}/*n40w120*.tif'.format(dem_dir))[0]
    overview = rasterio.open(dem_file)
    write_bbox_shp(patua_extents, './tmp_bbox.shp')
    bbox = gpd.read_file('./tmp_bbox.shp')
    topo, meta, value_range = clip_raster(bbox, overview)
    extent = plotting_extent(topo[0], meta['transform'])
    # Hillshade
    hillshade = es.hillshade(topo[0].copy(), azimuth=90, altitude=20)
    # Seismic catalog
    if catalog:
        cat = pd.read_excel(catalog, skiprows=[0, 1])
    # Read in vectors
    ch_roads = gpd.read_file('{}/ChurchillRoads.shp'.format(vector_dir)).to_crs(4326)
    ly_roads = gpd.read_file('{}/LyonRoads.shp'.format(vector_dir)).to_crs(4326)
    rr = gpd.read_file('{}/Patua_RRs.shp'.format(vector_dir)).to_crs(4326)
    plant = gpd.read_file('{}/Patua_Plant.shp'.format(vector_dir)).to_crs(4326)
    wells = gpd.read_file('{}/geothermal_wells.shp'.format(vector_dir)).to_crs(4326)
    I_pipe = gpd.read_file('{}/Injection_Pipelines.shp'.format(vector_dir)).to_crs(4326)
    P_pipe = gpd.read_file('{}/Production_Pipelines.shp'.format(vector_dir)).to_crs(4326)
    springs = gpd.read_file('{}/Patua_Hotsprings.shp'.format(vector_dir)).to_crs(4326)
    circle1 = gpd.read_file('{}/Patua_3546-m_radius.shp'.format(vector_dir)).to_crs(4326)
    circle2 = gpd.read_file('{}/Patua_5319-m_radius.shp'.format(vector_dir)).to_crs(4326)
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    # Only top half of colormap
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap = plt.get_cmap('gist_earth')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    cmap2 = LinearSegmentedColormap.from_list('Upper Half', colors)
    # Bottom up, first DEM
    ax.imshow(topo[0], cmap=cmap2, extent=extent, alpha=0.3)
    # Then hillshade
    ax.imshow(hillshade, cmap="Greys", alpha=0.3, extent=extent)
    # Vector layers
    # ly_roads.loc[ly_roads.RTTYP.isin(['S', 'C', 'U', 'M', 'O'])].plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    # ch_roads.loc[ch_roads.STATE_ROAD == 'YES'].plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    ly_roads.plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    ch_roads.plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    rr.plot(ax=ax, linewidth=1., color='firebrick', alpha=0.5)
    I_pipe.plot(ax=ax, color='b', alpha=0.5)
    P_pipe.plot(ax=ax, color='r', alpha=0.5)
    plant.geometry.plot(ax=ax, color='k')
    springs.plot(ax=ax, markersize=10., marker='*', color='dodgerblue')
    # circle1.plot(ax=ax, color='dodgerblue', linewidth=1.)
    # circle2.plot(ax=ax, color='dodgerblue', linewidth=1.)
    # Labels
    ax.annotate('Hot Springs', xy=springs.geometry[0].coords[0], xytext=(-30, 10),
                textcoords='offset points', fontsize=8, fontstyle='italic',
                color='dodgerblue')
    # Injection wells
    wells.loc[wells.status == 'injector'].plot(ax=ax, markersize=10, color='b')
    # Production wells
    wells.loc[wells.status == 'producer'].plot(ax=ax, markersize=10, color='r')
    # Leidos catalog
    if catalog:
        ax.scatter(cat['Longitude'], cat['Latitude'], marker='o', color='k',
                   facecolor=None, s=1., alpha=0.3)
    # Seismic stations
    for sta in inventory.select(location='10')[0]:
        if sta.code == '2317':
            continue
        ax.scatter(sta.longitude, sta.latitude, marker='v', s=40., color='purple')
        ax.annotate(
            '{}\n{}'.format(sta.code, RT130_serial[sta.code]),
            xy=(sta.longitude, sta.latitude), xytext=(6, -8),
            textcoords='offset points', fontsize=10, fontweight='bold',
            color='purple')
    # Add 23A-17
    well_23a17 = wells.loc[wells.name.isin(['23A-17'])]
    well_23a17.plot(ax=ax, marker='v', markersize=40, color='purple')
    ax.annotate(xy=(well_23a17.geometry.x,
                    well_23a17.geometry.y), text='23A-17\nB2C8',
                textcoords='offset points', xytext=(6, 0),
                fontsize=10, fontweight='bold', color='purple')
    # Injection well
    wells.loc[wells.name == '16-29'].plot(ax=ax, marker='*', color='yellow',
                                          markersize=60.)
    # Potentially functioning seismometers
    wells.loc[wells.name.isin(['36-5 TGH', '87-25 TGH', '26-31 TGH',
                               '77-31 TGH', '35-33 TGH', '88-33 TGH',
                               '36-15', '28-13', '45-27', '33-23',
                               '28-13', '36-15', '21A-19', '27-29'])].plot(
        ax=ax, marker='v', color='sienna', markersize=40.)
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    # Scale bar
    points = gpd.GeoSeries([Point(-117., extent[2]),
                            Point(-118., extent[2])], crs=4326)
    points = points.to_crs(32611)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel(r'Longitude [$^o$]')
    ax.set_ylabel(r'Latitude [$^o$]')
    ax.set_title('Patua')
    plt.show()
    return


def plot_cape(dem_dir, vector_dir, catalog):
    """Cape modern overview plot"""

    cape_extents = [(-113.03, 38.60), (-113.03, 38.39),
                    (-112.78, 38.39), (-112.78, 38.60)]
    delano = [333669, 4262718]
    dem_file = glob('{}/Cape-modern_merged_clip.tif'.format(dem_dir))[0]
    overview = rasterio.open(dem_file)
    write_bbox_shp(cape_extents, './tmp_bbox.shp')
    bbox = gpd.read_file('./tmp_bbox.shp')
    topo, meta, value_range = clip_raster(bbox, overview)
    extent = plotting_extent(topo[0], meta['transform'])
    # Hillshade
    hillshade = es.hillshade(topo[0].copy(), azimuth=90, altitude=20)
    # Seismic catalog
    if catalog:
        cat = pd.read_excel(catalog, skiprows=[0, 1])
    # Read in vectors
    delano = gpd.read_file('{}/Delano.shp'.format(vector_dir)).to_crs(4326)
    roads = gpd.read_file('{}/Utah_roads.shp'.format(vector_dir)).to_crs(4326)
    plant = gpd.read_file('{}/Blundel-plant.shp'.format(vector_dir)).to_crs(4326)
    UU_stations = gpd.read_file('{}/UU_stations.shp'.format(vector_dir)).to_crs(4326)
    K_stations = gpd.read_file('{}/6K_stations.shp'.format(vector_dir)).to_crs(4326)
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    # Only top half of colormap
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap = plt.get_cmap('gist_earth')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    cmap2 = LinearSegmentedColormap.from_list('Upper Half', colors)
    # Bottom up, first DEM
    ax.imshow(topo[0], cmap=cmap2, extent=extent, alpha=0.3)
    # Then hillshade
    ax.imshow(hillshade, cmap="Greys", alpha=0.3, extent=extent)
    # Vector layers
    roads[roads['SPEED_LMT'] > 35].plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    # plant.geometry.plot(ax=ax, color='dimgray')
    delano.plot(ax=ax, marker='x', markersize=80, color='k')
    # Labels
    ax.annotate('Cape Modern', xy=delano.geometry[0].coords[0], xytext=(-80, 6),
                textcoords='offset points', fontsize=12, fontstyle='italic', color='k')
    # Catalog
    if catalog:
        ax.scatter(cat['Longitude'], cat['Latitude'], marker='o', color='k',
                   facecolor=None, s=1., alpha=0.3)
    # Seismic stations
    UU_stations.plot(ax=ax, marker='^', markersize=80, color='k')
    K_stations.plot(ax=ax, marker='v', markersize=80, color='firebrick')
    for i, row in K_stations.iterrows():
        loc = row.geometry.coords[0][:-1]
        sta = '6K.'+ row.Name.split()[0].replace('FG', 'CS')
        ax.annotate(
            sta, xy=loc, xytext=(4, 4),
            textcoords='offset points', fontsize=10, fontweight='bold',
            color='firebrick')
    # Inset map
    ax2 = inset_axes(ax, width=2, height=2, loc=2,
                     axes_class=GeoAxes,
                     axes_kwargs=dict(projection=ccrs.AlbersEqualArea(-112, 40)))
    # ax2.stock_img()
    ax2.add_feature(cf.BORDERS)
    ax2.add_feature(cf.LAND)
    ax2.add_feature(cf.ShapelyFeature(bbox.geometry, crs=ccrs.CRS('epsg:4326')),
                    edgecolor='r', linewidth=2., facecolor='none')
    ax2.add_feature(
        cf.NaturalEarthFeature(
            category='cultural', name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none'
        )
    )
    ax2.set_extent([-116, -108, 36, 44], crs=ccrs.CRS('epsg:4326'))
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    # Scale bar
    points = gpd.GeoSeries([Point(-117., extent[2]),
                            Point(-118., extent[2])], crs=4326)
    points = points.to_crs(32611)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel(r'Longitude [$^o$]')
    ax.set_ylabel(r'Latitude [$^o$]')
    ax.set_title('Cape Modern Seismic Sites')
    fig.savefig('Cape_modern_seismic.png', dpi=300)
    # plt.show()
    return


def plot_newberry(dem_dir, vector_dir, inventory):
    """Patua overview plot"""
    dem = rasterio.open('{}/USGS_13_merged_epsg-26910.tif'.format(dem_dir))
    lidar = rasterio.open('{}/USGS_Lidar_merged_rasterio.tif'.format(dem_dir))
    bbox = gpd.read_file('{}/Newberry_outline_plotting_epsg-26910.shp'.format(vector_dir))
    dem_vals, meta_dem, value_range = clip_raster(bbox, dem)
    lidar_vals, meta_lidar, value_range = clip_raster(bbox, lidar)
    extent_lidar = plotting_extent(lidar_vals[0], meta_lidar['transform'])
    extent_dem = plotting_extent(dem_vals[0], meta_dem['transform'])
    # Hillshade
    dem_hillshade = es.hillshade(dem_vals[0].copy(), azimuth=90, altitude=20)
    lidar_hillshade = es.hillshade(lidar_vals[0].copy(), azimuth=90, altitude=20)
    # Read in vectors
    sensors = obspy.read_inventory(inventory)
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    # Only top half of colormap
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap = plt.get_cmap('gist_earth')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    cmap2 = LinearSegmentedColormap.from_list('Upper Half', colors)
    # Bottom up, first DEM
    ax.imshow(dem_vals[0], cmap=cmap2, alpha=0.2, extent=extent_lidar)
    ax.imshow(dem_hillshade, cmap="Greys", alpha=0.3, extent=extent_lidar)
    ax.imshow(lidar_vals[0], cmap=cmap2, alpha=0.3, extent=extent_lidar)
    # Then hillshade
    ax.imshow(lidar_hillshade, cmap="Greys", alpha=0.3, extent=extent_lidar)
    # Vector layers
    roads = gpd.read_file('{}/Newberry_roads_clipped.shp'.format(vector_dir)).to_crs(26910)
    lakes = gpd.read_file('{}/Newberry_lakes_clipped.shp'.format(vector_dir)).to_crs(26910)
    # roads.plot(ax=ax, linewidth=0.5, color='dimgray', alpha=0.5)
    ax = lakes.geometry.plot(ax=ax, color='dodgerblue', alpha=0.5)
    # Seismic stations
    marker_dict = {'UW': ['^', 'dodgerblue'], 'CC': ['^', 'r'],
                   '9G': ['v', 'purple']}
    for net in sensors:
        for sta in net:
            if net.code == 'UW' and sta.code in ['NN17', 'NN32', 'NNVM']:
                continue
            elif net.code == '9G' and (sta.code.startswith('NM')
                                       or sta.code in ['NN19', 'NN21']):
                continue
            pt = gpd.GeoSeries([Point(sta.longitude, sta.latitude)], crs=4326)
            pt = pt.to_crs(26910)
            ax.scatter(pt.x, pt.y, marker=marker_dict[net.code][0], s=50.,
                       color=marker_dict[net.code][1], label=net.code,
                       edgecolors='k')
    # Proposed locations
    ax.set_xlim([extent_lidar[0], extent_lidar[1]])
    ax.set_ylim([extent_lidar[2], extent_lidar[3]])
    # Scale bar
    # points = gpd.GeoSeries([Point(630000, extent_lidar[2]),
    #                         Point(635000, extent_lidar[2])], crs=4326)
    # points = points.to_crs(32611)  # Projected WGS 84 - meters
    # distance_meters = points[0].distance(points[1])
    # ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel(r'Easting [m]')
    ax.set_ylabel(r'Northing [m]')
    ax.set_title('Newberry seismic networks')
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.legend()
    legend_without_duplicate_labels(ax)
    plt.show()
    return


def plot_TM(dem_dir, vector_dir):
    """Patua overview plot"""
    TM_extents = [(-117.72, 39.6975), (-117.72, 39.6472),
                  (-117.65, 39.6472), (-117.65, 39.6975)]
    TM_plant_extents = [(-117.68896, 39.67314), (-117.68848, 39.67211),
                        (-117.69158, 39.67115), (-117.69227, 39.67239)]
    stations = {'ROK': (-117.70123, 39.67387), 'SED': (-117.68897, 39.66543)}
    write_bbox_shp(TM_plant_extents, '{}/TM_plant.shp'.format(vector_dir))
    dem_file = glob('{}/*n40w118*.tif'.format(dem_dir))[0]
    overview = rasterio.open(dem_file)
    write_bbox_shp(TM_extents, './tmp_bbox.shp')
    bbox = gpd.read_file('./tmp_bbox.shp')
    topo, meta, value_range = clip_raster(bbox, overview)
    extent = plotting_extent(topo[0], meta['transform'])
    # Hillshade
    hillshade = es.hillshade(topo[0].copy(), azimuth=90, altitude=20)
    # Read in vectors
    ch_roads = gpd.read_file('{}/ChurchillRoads.shp'.format(vector_dir)).to_crs(4326)
    plant = gpd.read_file('{}/TM_plant.shp'.format(vector_dir)).to_crs(4326)
    sensors = gpd.read_file('{}/TM_sensors.shp'.format(vector_dir)).to_crs(4326)
    lease = gpd.read_file('{}/TM_lease.shp'.format(vector_dir)).to_crs(4326)
    woo_well = gpd.read_file('{}/TM_Stim_Well_24A-23.shp'.format(vector_dir)).to_crs(4326)
    circle1 = gpd.read_file('{}/TM_1280-m_radius.shp'.format(vector_dir)).to_crs(4326)
    circle2 = gpd.read_file('{}/TM_1920-m_radius.shp'.format(vector_dir)).to_crs(4326)
    tracks = gpd.read_file('{}/TM_tracks.shp'.format(vector_dir)).to_crs(4326)
    solar_array = gpd.read_file('{}/Tungsten_solar_array.shp'.format(vector_dir)).to_crs(4326)
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    # Only top half of colormap
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap = plt.get_cmap('gist_earth')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    cmap2 = LinearSegmentedColormap.from_list('Upper Half', colors)
    # Bottom up, first DEM
    ax.imshow(topo[0], cmap=cmap2, extent=extent, alpha=0.3)
    # Then hillshade
    ax.imshow(hillshade, cmap="Greys", alpha=0.3, extent=extent)
    # Vector layers
    ch_roads.plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    tracks.plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    ax = plant.geometry.plot(ax=ax, color='k', label='Plant')
    ax = solar_array.geometry.plot(ax=ax, color='gray', label='Solar array')
    ax = lease.plot(ax=ax, linestyle=':', color='firebrick')
    ax = woo_well.plot(ax=ax, marker='*', color='yellow',
                       markersize=60., legend=True, label='WOO Well',
                       legend_kwds=dict(loc='upper left'))
    circle1.plot(ax=ax, color='dodgerblue', linewidth=1.)
    circle2.plot(ax=ax, color='dodgerblue', linewidth=1.)
    # Labels
    # Seismic stations
    for sta, loc in stations.items():
        ax.scatter(loc[0], loc[1], marker='^', s=40., color='purple')
        ax.annotate(
            sta, xy=loc, xytext=(3, 3),
            textcoords='offset points', fontsize=10, fontweight='bold',
            color='purple')
    # Proposed locations
    sensors.plot(ax=ax, marker='v', color='indigo', markersize=40)
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    # Scale bar
    points = gpd.GeoSeries([Point(-117., extent[2]),
                            Point(-118., extent[2])], crs=4326)
    points = points.to_crs(32611)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel(r'Longitude [$^o$]')
    ax.set_ylabel(r'Latitude [$^o$]')
    ax.set_title('Tungsten Mountain')
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.legend()
    plt.show()
    return


def plot_JV(dem_dir, vector_dir):
    """Patua overview plot"""
    JV_extents = [(-117.505, 40.195), (-117.447, 40.195),
                  (-117.447, 40.145), (-117.505, 40.145)]
    JV_plant_extents = [(-117.47696, 40.18190), (-117.47471, 40.18190),
                        (-117.47471, 40.18005), (-117.47696, 40.18005)]
    stations = {'ROK': (-117.47066, 40.17340), 'SED': (-117.49506, 40.17577)}
    write_bbox_shp(JV_plant_extents, '{}/JV_plant.shp'.format(vector_dir))
    dem_file = glob('{}/*n41w118*.tif'.format(dem_dir))[0]
    overview = rasterio.open(dem_file)
    write_bbox_shp(JV_extents, './tmp_bbox.shp')
    bbox = gpd.read_file('./tmp_bbox.shp')
    topo, meta, value_range = clip_raster(bbox, overview)
    extent = plotting_extent(topo[0], meta['transform'])
    # Hillshade
    hillshade = es.hillshade(topo[0].copy(), azimuth=90, altitude=20)
    # Read in vectors
    ch_roads = gpd.read_file('{}/ChurchillRoads.shp'.format(vector_dir)).to_crs(4326)
    plant = gpd.read_file('{}/JV_plant.shp'.format(vector_dir)).to_crs(4326)
    circle1 = gpd.read_file('{}/JV_973-m_radius.shp'.format(vector_dir)).to_crs(4326)
    circle2 = gpd.read_file('{}/JV_1460-m_radius.shp'.format(vector_dir)).to_crs(4326)
    lease = gpd.read_file('{}/JV_lease.shp'.format(vector_dir)).to_crs(4326)
    tracks = gpd.read_file('{}/JV_tracks.shp'.format(vector_dir)).to_crs(4326)
    two_track = gpd.read_file('{}/JV_S-two-track.shp'.format(vector_dir)).to_crs(4326)
    sensors = gpd.read_file('{}/JV_sensors.shp'.format(vector_dir)).to_crs(4326)
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    # Only top half of colormap
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap = plt.get_cmap('gist_earth')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    cmap2 = LinearSegmentedColormap.from_list('Upper Half', colors)
    # Bottom up, first DEM
    ax.imshow(topo[0], cmap=cmap2, extent=extent, alpha=0.3)
    # Then hillshade
    ax.imshow(hillshade, cmap="Greys", alpha=0.3, extent=extent)
    # Vector layers
    ch_roads.plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    plant.geometry.plot(ax=ax, color='k')
    circle1.plot(ax=ax, color='dodgerblue', linewidth=1.)
    circle2.plot(ax=ax, color='dodgerblue', linewidth=1.)
    lease.boundary.plot(ax=ax, linestyle=':', color='firebrick')
    tracks.plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    two_track.plot(ax=ax, linewidth=1., color='dimgray', linestyle='--',
                   alpha=0.5)
    # Proposed locations
    sensors.plot(ax=ax, marker='v', color='indigo', markersize=40)
    # Stim well
    ax.scatter(-117.47384, 40.17023, marker='*', color='yellow',
               s=60.)
    # Labels
    # Seismic stations
    for sta, loc in stations.items():
        ax.scatter(loc[0], loc[1], marker='^', s=40., color='purple')
        ax.annotate(
            sta, xy=loc, xytext=(3, 3),
            textcoords='offset points', fontsize=10, fontweight='bold',
            color='purple')
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    # Scale bar
    points = gpd.GeoSeries([Point(-117., extent[2]),
                            Point(-118., extent[2])], crs=4326)
    points = points.to_crs(32611)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel(r'Longitude [$^o$]')
    ax.set_ylabel(r'Latitude [$^o$]')
    ax.set_title('Jersey Valley')
    ax.ticklabel_format(style='plain', useOffset=False)
    plt.show()
    return


def plot_DAC(dem_dir, vector_dir, catalog=None):
    """Patua overview plot"""
    DAC_extents = [(-118.39, 38.87), (-118.28, 38.87),
                  (-118.28, 38.80), (-118.39, 38.80)]
    DAC1_plant_extents = [(-118.32956, 38.83626), (-118.32452, 38.83626),
                          (-118.32452, 38.83530), (-118.32956, 38.83530)]
    DAC2_plant_extents = [(-118.32623, 38.83769), (-118.32140, 38.83769),
                          (-118.32140, 38.83676), (-118.32623, 38.83676)]
    write_bbox_shp(DAC1_plant_extents, '{}/DAC1_plant.shp'.format(vector_dir))
    write_bbox_shp(DAC2_plant_extents, '{}/DAC2_plant.shp'.format(vector_dir))
    dem_file = glob('{}/*n39w119*.tif'.format(dem_dir))[0]
    overview = rasterio.open(dem_file)
    write_bbox_shp(DAC_extents, './tmp_bbox.shp')
    bbox = gpd.read_file('./tmp_bbox.shp')
    topo, meta, value_range = clip_raster(bbox, overview)
    extent = plotting_extent(topo[0], meta['transform'])
    # Hillshade
    hillshade = es.hillshade(topo[0].copy(), azimuth=90, altitude=20)
    # Read in vectors
    ch_roads = gpd.read_file('{}/tl_2021_32021_roads.shp'.format(vector_dir)).to_crs(4326)
    plant1 = gpd.read_file('{}/DAC1_plant.shp'.format(vector_dir)).to_crs(4326)
    plant2 = gpd.read_file('{}/DAC2_plant.shp'.format(vector_dir)).to_crs(4326)
    lease = gpd.read_file('{}/DAC_Unit_Boundary-polygon.shp'.format(vector_dir)).to_crs(4326)
    stations = gpd.read_file('{}/Final-station-locations_DAC-point.shp'.format(vector_dir)).to_crs(4326)
    dac_rok = gpd.read_file('{}/DAC_ROK.shp'.format(vector_dir)).to_crs(4326)
    dac_sed = gpd.read_file('{}/DAC_SED.shp'.format(vector_dir)).to_crs(4326)
    woo_well = gpd.read_file('{}/DAC_WOO_well.shp'.format(vector_dir)).to_crs(4326)
    circle0 = gpd.read_file('{}/950-m_radius_circle_centered_on_inj.shp'.format(vector_dir)).to_crs(4326)
    circle1 = gpd.read_file('{}/1800-m_radius_circle_centered_on_inj.shp'.format(vector_dir)).to_crs(4326)
    circle2 = gpd.read_file('{}/2700-m_radius_circle_centered_on_inj.shp'.format(vector_dir)).to_crs(4326)
    sand = gpd.read_file('{}/sand-dunes.shp'.format(vector_dir)).to_crs(4326)
    I_pipe = gpd.read_file('{}/DAC_injection_pipelines.shp'.format(vector_dir)).to_crs(4326)
    P_pipe = gpd.read_file('{}/DAC_production_pipelines.shp'.format(vector_dir)).to_crs(4326)
    I_wells = gpd.read_file('{}/DAC_injection_wells.shp'.format(vector_dir)).to_crs(4326)
    P_wells = gpd.read_file('{}/DAC_production_wells.shp'.format(vector_dir)).to_crs(4326)
    # EQ catalog
    if catalog is not None:
        lats = [ev.preferred_origin().latitude for ev in catalog]
        longs = [ev.preferred_origin().longitude for ev in catalog]
        deps = [ev.preferred_origin().depth for ev in catalog]
        mags = [ev.preferred_magnitude().mag for ev in catalog]
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    # Only top half of colormap
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    cmap = plt.get_cmap('gist_earth')
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))
    # Create a new colormap from those colors
    cmap2 = LinearSegmentedColormap.from_list('Upper Half', colors)
    # Bottom up, first DEM
    ax.imshow(topo[0], cmap=cmap2, extent=extent, alpha=0.3)
    # Then hillshade
    ax.imshow(hillshade, cmap="Greys", alpha=0.3, extent=extent)
    # Sand
    sand.plot(ax=ax, facecolor='beige', alpha=0.3)
    ax.annotate('DUNES', xy=(-118.31, 38.86), fontsize=14, fontweight='bold',
                color='beige', alpha=0.7)
    # Vector layers
    ch_roads.plot(ax=ax, linewidth=1., color='dimgray', alpha=0.5)
    plant1.geometry.plot(ax=ax, color='k')
    plant2.geometry.plot(ax=ax, color='k')
    lease.boundary.plot(ax=ax, linestyle=':', color='firebrick')
    I_pipe.plot(ax=ax, color='b', alpha=0.5)
    P_pipe.plot(ax=ax, color='r', alpha=0.5)
    I_wells.plot(ax=ax, marker='o', color='b', markersize=20, alpha=0.5)
    P_wells.plot(ax=ax, marker='o', color='r', markersize=20, alpha=0.5)
    woo_well.plot(ax=ax, marker='*', color='yellow',
                  markersize=60.)
    circle0.plot(ax=ax, color='dodgerblue', linewidth=1.)
    circle1.plot(ax=ax, color='dodgerblue', linewidth=1.)
    circle2.plot(ax=ax, color='dodgerblue', linewidth=1.)
    if catalog is not None:
        ax.scatter(longs, lats, marker='o', color='k',
                   facecolor=None, s=10.*(1**np.array(mags)), alpha=0.3)
    # Labels
    # Seismic stations
    stations.plot(ax=ax, marker='v', markersize=50, color='indigo')
    dac_rok.plot(ax=ax, marker='^', markersize=60, color='purple')
    ax.annotate('ROK', dac_rok.geometry[0].coords[0][:2], xytext=(3, 3),
                textcoords='offset points', fontsize=10, fontweight='bold',
                color='purple')
    dac_sed.plot(ax=ax, marker='^', markersize=60, color='purple')
    ax.annotate('SED', dac_sed.geometry[0].coords[0][:2], xytext=(3, 3),
                textcoords='offset points', fontsize=10, fontweight='bold',
                color='purple')
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    # Scale bar
    points = gpd.GeoSeries([Point(-117., extent[2]),
                            Point(-118., extent[2])], crs=4326)
    points = points.to_crs(32611)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel(r'Longitude [$^o$]')
    ax.set_ylabel(r'Latitude [$^o$]')
    ax.set_title('Don A Campbell')
    fig.tight_layout()
    plt.show()
    return


def plot_5529_seismicity(well_path, catalog, location_method='NonLinLoc'):
    """
    Plot seismicity relative to 55-29 with a NS and EW cross-section

    :param well_path:
    :param zone_path:
    :param catalog:
    :return:
    """
    utm = pyproj.Proj("EPSG:32610")
    lith_colors = {'Welded Tuff': 'darkkhaki', 'Tuff': 'khaki', 'Basalt': 'darkgray', 'Granodiorite': 'bisque'}
    method_colors = {'RTDD': 'r', 'NonLinLoc': 'b', 'LOCSAT': 'k', 'SimulPS': 'purple'}
    depth_factor = {'RTDD': -1., 'NonLinLoc': -1., 'LOCSAT': -1., 'SimulPS': -1000.}
    depth_correction = {'RTDD': 0., 'NonLinLoc': 0., 'LOCSAT': 0., 'SimulPS': 0.}  # Elevation correction for LOCSAT
    lim = [-2000, 2000]
    fig = plt.figure(figsize=(8, 16))
    spec = gridspec.GridSpec(ncols=8, nrows=16, figure=fig)
    ax_map = fig.add_subplot(spec[:8, :])
    ax_e = fig.add_subplot(spec[8:, :4])
    ax_n = fig.add_subplot(spec[8:, 4:])
    elev_wh = 1703.2488  # Wellhead elevation
    Lith_depths = {'Welded Tuff': [[1966, 2057]], 'Tuff': [[2057, 2439]], 'Basalt': [[2439, 2634], [2908, 3067]],
                   'Granodiorite': [[2634, 2908]],}
    slotted = [(1912, 2289), (2493, 3045)]
    # Lithology first
    for unit, depths in Lith_depths.items():
        for i, ax in enumerate([ax_e, ax_n]):
            for j, d in enumerate(depths):
                if i == 0 and j == 0:
                    lab = unit
                else:
                    lab = ''
                ax.axhspan(elev_wh - d[1], elev_wh - d[0], color=lith_colors[unit], alpha=.3, label=lab)
    # Add objects
    wellpath = np.loadtxt(well_path, delimiter=',', skiprows=1)
    east = wellpath[:, 0] - wellpath[0, 0]
    north = wellpath[:, 1] - wellpath[0, 1]
    elev_m = wellpath[:, 2]
    # Upper well
    top_e = east[np.where(elev_m > elev_wh - slotted[0][0])]
    top_n = north[np.where(elev_m > elev_wh - slotted[0][0])]
    top_d = elev_m[np.where(elev_m > elev_wh - slotted[0][0])]
    mid_e = east[np.where((elev_m < elev_wh - slotted[0][1]) & (elev_m > elev_wh - slotted[1][0]))]
    mid_n = north[np.where((elev_m < elev_wh - slotted[0][1]) & (elev_m > elev_wh - slotted[1][0]))]
    mid_d = elev_m[np.where((elev_m < elev_wh - slotted[0][1]) & (elev_m > elev_wh - slotted[1][0]))]
    bot_e = east[np.where(elev_m < (elev_wh - slotted[1][1]))]
    bot_n = north[np.where(elev_m < (elev_wh - slotted[1][1]))]
    bot_d = elev_m[np.where(elev_m < (elev_wh - slotted[1][1]))]
    # Plot them
    ax_map.plot(top_e, top_n, color='darkgray')
    ax_map.plot(mid_e, mid_n, color='darkgray')
    ax_map.plot(bot_e, bot_n, color='darkgray')
    ax_e.plot(top_e, top_d, color='darkgray')
    ax_e.plot(mid_e, mid_d, color='darkgray')
    ax_e.plot(bot_e, bot_d, color='darkgray')
    ax_n.plot(top_n, top_d, color='darkgray')
    ax_n.plot(mid_n, mid_d, color='darkgray')
    ax_n.plot(bot_n, bot_d, color='darkgray')
    # Slotted sections
    for slot in slotted:
        e = east[np.where((elev_m > elev_wh - slot[1]) & (elev_m < elev_wh - slot[0]))]
        n = north[np.where((elev_m > elev_wh - slot[1]) & (elev_m < elev_wh - slot[0]))]
        d = elev_m[np.where((elev_m > elev_wh - slot[1]) & (elev_m < elev_wh - slot[0]))]
        ax_map.plot(e, n, color='goldenrod', linestyle=':')
        ax_e.plot(e, d, color='goldenrod', linestyle=':')
        ax_n.plot(n, d, color='goldenrod', linestyle=':')
    # Now catalog
    preferred = [ev.preferred_origin() for ev in catalog]
    methods = list(set([o.method_id for ev in catalog for o in ev.origins]))
    for method in methods:
        for origin in preferred:
            if origin.method_id == method:
                meth = method.id.split('/')[-1]
                x, y = utm(origin.longitude, origin.latitude)
                x -= wellpath[0, 0]
                y -= wellpath[0, 1]
                d = (origin.depth * depth_factor[meth]) + depth_correction[meth]
                ax_map.scatter(x, y, edgecolor=method_colors[meth], facecolor='none', s=2., marker='o', linewidth=0.25)
                ax_e.scatter(x, d, edgecolor=method_colors[meth], facecolor='none',
                             s=2., marker='o', linewidth=0.25)
                ax_n.scatter(y, d, edgecolor=method_colors[meth], facecolor='none',
                             s=2., marker='o', linewidth=0.25)
    # Set limits
    ax_map.set_ylim(lim)
    ax_map.set_xlim(lim)
    ax_n.set_xlim(lim)
    ax_e.set_xlim(lim)
    ax_e.set_ylim((-3000, 1800))
    ax_n.set_ylim((-3000, 1800))
    # label and tick positioning
    ax_map.xaxis.set_label_position('top')
    ax_map.set_xlabel('Easting [m]')
    ax_map.set_ylabel('Northing [m]')
    ax_map.tick_params(axis='x', labelbottom=False, labeltop=True, bottom=False, top=True)
    ax_e.set_ylabel('Depth [m]')
    ax_e.set_xticks([-1700, 1700], ['E', 'W'], size=16, fontweight='bold')
    ax_n.set_xticks([-1700, 1700], ['N', 'S'], size=16, fontweight='bold')
    ax_n.tick_params(axis='y', labelleft=False, left=False)
    fig.legend(loc=4)
    plt.show()
    return


#### One-off Collab plotting
optasense_dicts = [dict(Task='Optasense', Start=datetime(2022, 3, 17, 22, 3),
                        Finish=datetime(2022, 6, 1, 16, 55),
                        Resource="Good timing"),
                   dict(Task='Optasense', Start=datetime(2022, 6, 1, 17, 4),
                        Finish=datetime(2022, 7, 20, 4, 10),
                        Resource="Good timing")
                   ]
iDAS_dicts = [dict(Task='iDAS', Start=datetime(2022, 3, 19, 1, 21),
                        Finish=datetime(2022, 5, 11),
                   Resource="Good timing"),
              dict(Task='iDAS', Start=datetime(2022, 5, 17, 6, 12),
                   Finish=datetime(2022, 5, 18, 22, 16),
                   Resource="Good timing"),
              # Next one has bad timing (diff color?)
              dict(Task='iDAS', Start=datetime(2022, 5, 23, 18, 5),
                   Finish=datetime(2022, 5, 26, 15, 13),
                   Resource="Bad timing"),
              dict(Task='iDAS', Start=datetime(2022, 5, 26, 15, 13),
                   Finish=datetime(2022, 7, 19, 4, 10),
                   Resource="Good timing"),
              dict(Task='iDAS', Start=datetime(2022, 7, 26, 19, 46),
                   Finish=datetime(2022, 9, 7, 21, 51),
                   Resource="Good timing")
              ]
terra15_dicts = [dict(Task='Terra15', Start=datetime(2022, 3, 18, 15, 57),
                      Finish=datetime(2022, 3, 20, 18, 34),
                      Resource="Good timing"),
                 dict(Task='Terra15', Start=datetime(2022, 3, 22, 18, 25),
                      Finish=datetime(2022, 4, 19, 22, 56),
                      Resource="Good timing"),
                 # Guessing at how short the reconfiguration was (15 min?)
                 dict(Task='Terra15', Start=datetime(2022, 4, 19, 23, 10),
                      Finish=datetime(2022, 4, 26, 2, 41),
                      Resource="Good timing"),
                 dict(Task='Terra15', Start=datetime(2022, 5, 6, 3, 38),
                      Finish=datetime(2022, 5, 9, 19, 7),
                      Resource="Good timing"),
                 dict(Task='Terra15', Start=datetime(2022, 5, 17, 6, 2),
                      Finish=datetime(2022, 6, 21, 18, 20),
                      Resource="Good timing")
                 ]
def plot_Collab_gantt(cassm_files, vbox_files, dss_files, dts_files, outfile):
    """
    Plot gantt chart of data coverage for various monitoring systems
    """
    # Programatically calculate time spans from CASSM file names
    times = []
    with open(cassm_files, 'r') as f:
        for l in f:
            try:
                dt = datetime.strptime(l.strip(), '%Y%m%d%H%M%S')
            except ValueError:
                continue
            times.append(dt)
    dates = pd.DataFrame(times, columns=['date'])
    deltas = dates['date'].diff()
    gaps = deltas[deltas > timedelta(hours=1)]
    gap_indices = gaps.index.values
    cassm_dicts = [dict(Task="CASSM", Start=times[0],
                        Finish=times[gap_indices[0]-1],
                        Resource="Good timing")]
    # Vbox parsing
    vbox_times = []
    for f in vbox_files:
        with open(f, 'r') as file:
            for l in file:
                # Get both times
                dts = l.split()
                for fn in dts:
                    try:
                        dt = datetime.strptime(fn.rstrip(), 'vbox_%Y%m%d%H%M%S%f.dat')
                    except ValueError:
                        continue
                    vbox_times.append(dt)
    vbox_times.sort()
    vbox_dates = pd.DataFrame(vbox_times, columns=['date'])
    vbox_deltas = vbox_dates['date'].diff()
    vbox_gaps = vbox_deltas[vbox_deltas > timedelta(hours=1)]
    vbox_gap_indices = vbox_gaps.index.values
    vbox_dicts = [dict(Task="VBox", Start=vbox_times[0],
                       Finish=vbox_times[vbox_gap_indices[0]-1],
                       Resource="Good timing")]
    # DSS parsing
    dss_times = []
    with open(dss_files, 'r') as file:
        for l in file:
            dstr = l.split()[-2].split('/')[-1]
            try:
                dt = datetime.strptime(dstr, '%Y_%m_%d_%H_%M_%S')
            except ValueError:
                continue
            dss_times.append(dt)
    dss_times.sort()
    dss_dates = pd.DataFrame(dss_times, columns=['date'])
    dss_deltas = dss_dates['date'].diff()
    dss_gaps = dss_deltas[dss_deltas > timedelta(hours=1)]
    dss_gap_indices = dss_gaps.index.values
    dss_dicts = [dict(Task="DSS", Start=dss_times[0],
                      Finish=dss_times[dss_gap_indices[0]-1],
                      Resource="Good timing")]
    # DTS parsing
    dts_times = []
    with open(dts_files, 'r') as file:
        for l in file:
            dstr = l.split()[-1]
            try:
                dt = datetime.strptime(dstr, '1_%Y%m%d%H%M%S%f.xml')
            except ValueError:
                continue
            dts_times.append(dt)
    dts_times.sort()
    dts_dates = pd.DataFrame(dts_times, columns=['date'])
    dts_deltas = dts_dates['date'].diff()
    dts_gaps = dts_deltas[dts_deltas > timedelta(hours=1)]
    dts_gap_indices = dts_gaps.index.values
    dts_dicts = [dict(Task="DTS", Start=dts_times[0],
                      Finish=dts_times[dts_gap_indices[0]-1],
                      Resource="Good timing")]
    for j, ind in enumerate(gap_indices):
        try:
            cassm_dicts.append(dict(Task="CASSM", Start=times[ind],
                                    Finish=times[gap_indices[j+1]-1],
                                    Resource="Good timing"))
        except IndexError:
            cassm_dicts.append(dict(Task="CASSM", Start=times[ind],
                                    Finish=times[-1],
                                    Resource="Good timing"))
    for j, ind in enumerate(vbox_gap_indices):
        try:
            vbox_dicts.append(dict(Task="VBox", Start=vbox_times[ind],
                                    Finish=vbox_times[vbox_gap_indices[j+1]-1],
                                    Resource="Good timing"))
        except IndexError:
            vbox_dicts.append(dict(Task="VBox", Start=vbox_times[ind],
                                    Finish=vbox_times[-1],
                                    Resource="Good timing"))
    for j, ind in enumerate(dss_gap_indices):
        try:
            dss_dicts.append(dict(Task="DSS", Start=dss_times[ind],
                                  Finish=dss_times[dss_gap_indices[j+1]-1],
                                  Resource="Good timing"))
        except IndexError:
            dss_dicts.append(dict(Task="DSS", Start=dss_times[ind],
                                  Finish=dss_times[-1],
                                  Resource="Good timing"))
    for j, ind in enumerate(dts_gap_indices):
        try:
            dts_dicts.append(dict(Task="DTS", Start=dts_times[ind],
                                  Finish=dts_times[dts_gap_indices[j+1]-1],
                                  Resource="Good timing"))
        except IndexError:
            dts_dicts.append(dict(Task="DTS", Start=dts_times[ind],
                                  Finish=dts_times[-1],
                                  Resource="Good timing"))
    all_dicts = (optasense_dicts + iDAS_dicts + terra15_dicts + cassm_dicts
                 + vbox_dicts + dss_dicts + dts_dicts)
    fig = create_gantt(all_dicts, group_tasks=True, index_col="Resource",
                       show_colorbar=True)
    plotly.offline.plot(fig, filename=outfile)
    fig.show()
    return