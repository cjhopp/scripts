#!/usr/bin/python

"""
Plotting functions for the lbnl module
"""
try:
    import dxfgrabber
except ImportError:
    print('Dont plot SURF dxf, dxfgrabber not installed')
import plotly
import rasterio
import fiona

import numpy as np
import colorlover as cl
import seaborn as sns
import pandas as pd
import geopandas as gpd
import chart_studio.plotly as py
import plotly.graph_objs as go
import shapely.geometry as geometry
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import earthpy.spatial as es

from rasterio import mask as msk
from rasterio.plot import plotting_extent
from shapely.geometry import mapping
from shapely.geometry.point import Point
from itertools import cycle
from glob import glob
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.io import loadmat
from scipy.linalg import lstsq, norm
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from obspy import Trace, Catalog
from plotly.subplots import make_subplots
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib import animation
from matplotlib.patches import Circle
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from scipy.signal import resample, detrend

# Local imports (assumed to be in python path)
from lbnl.boreholes import (parse_surf_boreholes, create_FSB_boreholes,
                            structures_to_planes, depth_to_xyz,
                            distance_to_borehole)
from lbnl.coordinates import SURF_converter
try:
    from lbnl.DSS import (interpolate_picks, extract_channel_timeseries,
                          get_well_piercepoint, get_frac_piercepoint,
                          extract_strains, fault_depths)
except ModuleNotFoundError:
    print('Error on DSS import. Change env')


csd_well_colors = {'D1': 'blue', 'D2': 'blue', 'D3': 'green',
                   'D4': 'green', 'D5': 'green', 'D6': 'green', 'D7': 'black'}

fsb_well_colors = {'B1': 'k', 'B2': 'steelblue', 'B3': 'goldenrod',
                   'B4': 'goldenrod', 'B5': 'goldenrod', 'B6': 'goldenrod',
                   'B7': 'goldenrod', 'B8': 'firebrick', 'B9': 'firebrick',
                   'B10': 'k'}

cols_4850 = {'PDT': 'black', 'PDB': 'black', 'PST': 'black', 'PSB': 'black',
             'OT': 'black', 'OB': 'black', 'I': '#4682B4', 'P': '#B22222'}

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
                zrange=(450, 500), sampling=0.5, eye=None, export=False):
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
    if location == 'surf':
        well_dict = parse_surf_boreholes(well_file)
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
    if fault:
        add_fault(fault[0], fault[1], fault[2], datas)
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
        data_act = np.array([[float(d.ch1903_east.value),
                              float(d.ch1903_north.value),
                              float(d.ch1903_elev.value)]
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
            ax.scatter(data_act[:, 0], data_act[:, 1], data_act[:, 2],
                       color='k', s=5, alpha=0.1, zorder=120)
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
            col = 'lightgray'
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
    ax3d.view_init(elev=75, azim=-120.)
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
    ax_map.set_xlim([2579305, 2579353])
    ax_map.set_ylim([1247565, 1247612])
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
        if key.startswith('D'):
            group = 'CSD'
            viz = True
        elif key.startswith('B'):
            group = 'FS-B'
            viz = True
        elif len(key) == 1:
            group = 'FS'
            viz = True
        elif key[0] in ['O', 'P', 'I', 'P']:
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
    fault = go.Surface(x=x, y=y, z=z,
                      showlegend=True,
                      name='Fault fit',
                      opacity=0.5, hoverinfo='skip')
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
                  'PST12']
    sta_list = []
    if isinstance(inventory, dict):
        for sta, pt in inventory.items():
            (sx, sy, sz) = pt
            sta_list.append((sx, sy, sz, sta))
    else:
        # Do the same for the inventory
        for sta in inventory[0]:  # Assume single network for now
            if location == 'surf':
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
            else:
                legend = 'Hydrophone'
                color = 'green'
                symbol = 'square'
            sx = float(sta.extra['{}_east'.format(loc_key)].value)
            sy = float(sta.extra['{}_north'.format(loc_key)].value)
            sz = float(sta.extra['{}_elev'.format(loc_key)].value)
            name = sta.code
            sta_list.append((sx, sy, sz, name, legend, color, symbol))
    _, _, _, _, leg, _, _ = zip(*sta_list)
    for sensor_type in list(set(leg)):
        stax, stay, staz, nms, leg, col, sym = zip(*[sta for sta in sta_list if
                                                     sta[4] == sensor_type])
        objects.append(go.Scatter3d(x=np.array(stax), y=np.array(stay),
                                    z=np.array(staz),
                                    mode='markers',
                                    name=sensor_type,
                                    legendgroup='Seismic network',
                                    hoverinfo='text',
                                    text=nms,
                                    marker=dict(color=col,
                                                size=2.,
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


def plot_amplify_sites(dem_dir, vector_dir, catalog):
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
    # Now water
    nv_water.loc[nv_water.TYPE.isin(['Major Lake', 'Major River',
                                     'Major Reservoir'])].boundary.plot(
        ax=ax, facecolor='steelblue', edgecolor="none", alpha=0.5)
    ca_water.loc[ca_water.TYPE.isin(['perennial'])].boundary.plot(
        ax=ax, facecolor='steelblue', edgecolor="none", alpha=0.5)
    # Roads
    nv_roads.loc[nv_roads.RTTYP.isin(['I', 'U'])].plot(
        ax=ax, column='RTTYP', linewidth=0.5, color='firebrick')
    ca_roads.loc[ca_roads.RTTYP.isin(['I', 'U'])].plot(
        ax=ax, column='RTTYP', linewidth=0.5, color='firebrick')
    big_towns = towns.loc[towns.NAME.isin(['Reno', 'Carson City',
                                           'Fresno', 'Fernley'])]
    big_towns.plot(ax=ax, color='k', markersize=2.)
    # Label cities
    for x, y, label in zip(big_towns.geometry.x, big_towns.geometry.y, big_towns.NAME):
        if label == 'Fernley':
            xytext = (-20, 3)
        else:
            xytext = (3, 3)
        ax.annotate(label, xy=(x, y), xytext=xytext, textcoords="offset points",
                    fontsize=6, fontstyle='italic', fontweight='bold')
    nv.boundary.plot(ax=ax, linewidth=1.0, linestyle='--', color='k')
    # Annotate border
    ax.annotate('Nevada', xy=(-117.760, 37.170), xytext=(2, 2),
                textcoords="offset points", fontsize=10, fontstyle='italic',
                fontweight='bold', rotation=-43)
    ax.annotate('California', xy=(-117.760, 37.170), xytext=(-14, -14),
                textcoords="offset points", fontsize=10, fontstyle='italic',
                fontweight='bold', rotation=-43)
    # Geothermal fields
    for lab, loc in field_locations.items():
        ax.scatter(loc[0], loc[1], marker='s', color='k', s=10)
        ax.annotate(lab, xy=loc, xytext=(3, 3), textcoords='offset points',
                    fontsize=12, fontweight='bold')
    # Seismic catalog
    locs = np.array([[ev.origins[-1].longitude, ev.origins[-1].latitude,
                      int(ev.comments[-1].text.split('=')[-1])]
                     for ev in catalog])
    ax.scatter(locs[:, 0], locs[:, 1], marker='o', color='k',
               s=locs[:, 2] / 3)
    # Scale bar
    points = gpd.GeoSeries([Point(-117., extent[2]),
                            Point(-118., extent[2])], crs=4326)
    points = points.to_crs(32611)  # Projected WGS 84 - meters
    distance_meters = points[0].distance(points[1])
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    ax.add_artist(ScaleBar(distance_meters))
    ax.set_xlabel(r'Longitude [$^o$]')
    ax.set_ylabel(r'Latitude [$^o$]')
    plt.show()
    return


def plot_patua(dem_dir, vector_dir, inventory, catalog):
    """Patua overview plot"""
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
    circle1.plot(ax=ax, color='dodgerblue', linewidth=1.)
    circle2.plot(ax=ax, color='dodgerblue', linewidth=1.)
    # Labels
    ax.annotate('Hot Springs', xy=springs.geometry[0].coords[0], xytext=(-30, 10),
                textcoords='offset points', fontsize=8, fontstyle='italic',
                color='dodgerblue')
    # Injection wells
    wells.loc[wells.status == 'injector'].plot(ax=ax, markersize=10, color='b')
    # Production wells
    wells.loc[wells.status == 'producer'].plot(ax=ax, markersize=10, color='r')
    # Leidos catalog
    ax.scatter(cat['Longitude'], cat['Latitude'], marker='o', color='k',
               facecolor=None, s=1., alpha=0.3)
    # Seismic stations
    for sta in inventory.select(location='10')[0]:
        if sta.code == '4509':
            continue
        ax.scatter(sta.longitude, sta.latitude, marker='v', s=40., color='purple')
        ax.annotate(
            sta.code, xy=(sta.longitude, sta.latitude), xytext=(6, 0),
            textcoords='offset points', fontsize=10, fontweight='bold',
            color='purple')
    # Add 23A-17
    well_23a17 = wells.loc[wells.name.isin(['23A-17'])]
    well_23a17.plot(ax=ax, marker='v', markersize=40, color='purple')
    ax.annotate(xy=(well_23a17.geometry.x,
                    well_23a17.geometry.y), text='23A-17',
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


def plot_DAC(dem_dir, vector_dir):
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
    woo_well.plot(ax=ax, marker='*', color='yellow',
                  markersize=60.)
    circle0.plot(ax=ax, color='dodgerblue', linewidth=1.)
    circle1.plot(ax=ax, color='dodgerblue', linewidth=1.)
    circle2.plot(ax=ax, color='dodgerblue', linewidth=1.)
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
    plt.show()
    return
