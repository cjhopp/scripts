#!/usr/bin/python

"""
Plotting functions for the lbnl module
"""

import dxfgrabber

import numpy as np
import colorlover as cl
import seaborn as sns
import pandas as pd
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go

from itertools import cycle
from glob import glob
from datetime import datetime
from scipy.linalg import lstsq
from plotly.subplots import make_subplots
from vtk.util.numpy_support import vtk_to_numpy
from matplotlib.colors import ListedColormap
from scipy.signal import resample

# Local imports (assumed to be in python path)
from lbnl.boreholes import (parse_surf_boreholes, create_FSB_boreholes,
                            structures_to_planes, depth_to_xyz)
from lbnl.coordinates import SURF_converter
from lbnl.DSS import interpolate_picks, extract_channel_timeseries


def plotly_timeseries(DSS_dict, DAS_dict, simfip, hydro):
    """
    DataFrame of timeseries data of any kind

    :param DSS_dict:
    :param DAS_dict:
    :param simfip: Simfip dataframe
    :param hydro: Hydro dataframe
    :return:
    """
    # Top figure: fibers and hydro?
    # trace_dss = go.Scatter(x=DSS_dict['times'], y=DSS_dict['DSS'],
    #                        name="DSS at {} m".format(DSS_dict['depth']),
    #                        opacity=0.5, visible=False)
    trace_dss_std_bot = go.Scatter(x=DSS_dict['times'],
                                   y=DSS_dict['DSS_median'] - DSS_dict['DSS_std'],
                                   line=dict(color="mediumvioletred", width=0),
                                   fill=None)
    trace_dss_std_top = go.Scatter(x=DSS_dict['times'],
                                   y=DSS_dict['DSS_median'] + DSS_dict['DSS_std'],
                                   line=dict(color="mediumvioletred", width=0),
                                   fill='tonexty')
    trace_dss_median = go.Scatter(x=DSS_dict['times'], y=DSS_dict['DSS_median'],
                                  name="DSS: 20 min rolling median",
                                  line=dict(color='mediumvioletred'),
                                  fill=None)
    das_trace = go.Scatter(x=DAS_dict['times'], y=DAS_dict['data'],
                           name='DAS at 41 m', yaxis="y6",
                           line=dict(color='purple'))
    trace_dts = go.Scatter(x=DSS_dict['times'], y=DSS_dict['DTS'],
                           name="DTS at {} m".format(DSS_dict['depth']),
                           yaxis="y2",
                           line=dict(color='saddlebrown'))
    trace_flow = go.Scatter(x=hydro.index, y=hydro['Flow'], name="Flow",
                            yaxis="y3",
                            line=dict(color='steelblue'))
    trace_pres = go.Scatter(x=hydro.index, y=hydro['Pressure'],
                            name="Pressure", yaxis="y4",
                            line=dict(color='red'))
    trace_simfip_Y = go.Scatter(x=simfip.index, y=simfip['P Yates'],
                                name='P Yates', yaxis="y5")
    trace_simfip_T = go.Scatter(x=simfip.index, y=simfip['P Top'],
                                name='P Top', yaxis="y5")
    trace_simfip_A = go.Scatter(x=simfip.index, y=simfip['P Axial'],
                                name='P Axial', yaxis="y5", xaxis='x')
    # Create axis objects
    layout = go.Layout(
        xaxis=dict(domain=[0.1, 0.9]),
        yaxis=dict(title="DSS microstrain", titlefont=dict(color="black"),
                   tickfont=dict(color="black"), domain=[0.66, 1.0]),
        yaxis2=dict(title="Temperature (C)", titlefont=dict(color="saddlebrown"),
                    tickfont=dict(color="saddlebrown"), anchor="x",
                    overlaying="y", side="right"),
        yaxis3=dict(title="Flow [L/min]", titlefont=dict(color="steelblue"),
                    tickfont=dict(color="steelblue"), anchor="x",
                    side="left", domain=[0.33, 0.63]),
        yaxis4=dict(title="Pressure [MPa]", titlefont=dict(color="red"),
                    tickfont=dict(color="red"), anchor="x",
                    overlaying="y3", side="right"),
        yaxis5=dict(title="Microns", titlefont=dict(color="black"),
                    tickfont=dict(color="black"), domain=[0., 0.3]),
        yaxis6=dict(title="DAS microstrain", titlefont=dict(color="purple"),
                    tickfont=dict(color="purple"), overlaying='y', side='left',
                    position=0.04)
    )
    fig = go.Figure(data=[trace_dts, trace_flow, trace_pres,
                          trace_simfip_A, trace_simfip_T, trace_simfip_Y,
                          trace_dss_std_bot, trace_dss_std_top,
                          trace_dss_median, das_trace],
                    layout=layout)
    fig.show()
    return


def plot_lab_3D(outfile, location, catalog=None, inventory=None, well_file=None,
                title=None, offline=True, dd_only=False, surface='plane',
                DSS_picks=None, structures=None, meshes=None,
                xrange=(2579250, 2579400), yrange=(1247500, 1247650),
                zrange=(450, 500), sampling=0.5):
    """
    Plot boreholes, seismicity, monitoring network, etc in 3D in plotly

    :param outfile: Name of plot in plotly
    :param location: Either 'fsb' or 'surf' as of 12-18-19
    :param catalog: Optional catalog of seismicity
    :param inventory: Optional inventory for monitoring network
    :param well_file: If field == 'surf', must provide well (x, y, z) file
    :param wells: Boolean for whether to plot the wells
    :param video: Deprecated because it's impossible to deal with
    :param animation: (See above)
    :param title: Plot title
    :param offline: Boolean for whether to plot to plotly account (online)
        or to local disk (offline)
    :param dd_only: Are we only plotting dd locations?
    :param surface: What type of surface to fit to points? Supports 'plane'
        and 'ellipsoid' for now.
    :param DSS_picks: Dictionary {well name: {'heights': array,
                                              'widths': array,
                                              'depths': list}}
    :param structures: None or path to root well_info directory
    :param meshes: list of tup (Layer name, path) for files containing xyz
        vertices for mesh (only used for FSB Gallery at the moment; can be
        expanded)
    :param xrange: List of min and max x of volume to interpolate DSS over
    :param yrange: List of min and max y of volume to interpolate DSS over
    :param zrange: List of min and max z of volume to interpolate DSS over
    :param sampling: Sampling interval for ranges above (meters)

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
    datas = add_wells(well_dict, objects=datas, structures=structures)
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
                            surface=surface)
    # Start figure
    fig = go.Figure(data=datas)
    # Manually find the data limits, and scale appropriately
    all_x = np.ma.masked_invalid(np.concatenate(
        [d['x'].flatten() for d in fig.data if type(d['y']) == np.ndarray]))
    all_y = np.ma.masked_invalid(np.concatenate(
        [d['y'].flatten() for d in fig.data if type(d['y']) == np.ndarray]))
    all_z = np.ma.masked_invalid(np.concatenate(
        [d['z'].flatten() for d in fig.data if type(d['z']) == np.ndarray]))
    xrange = np.abs(np.max(all_x) - np.min(all_x))
    yrange = np.abs(np.max(all_y) - np.min(all_y))
    if location == 'fsb':
        zmin = 300.
    else:
        zmin = np.min(all_z)
    zrange = np.abs(np.max(all_z) - zmin)
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
    layout = go.Layout(scene=dict(xaxis=xax, yaxis=yax, zaxis=zax,
                                  xaxis_showspikes=False,
                                  yaxis_showspikes=False,
                                  aspectmode='manual',
                                  aspectratio=dict(x=1, y=yrange / xrange,
                                                   z=zrange / xrange),
                                  bgcolor="rgb(244, 244, 248)"),
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
                                   tracegroupgap=3))
    fig.update_layout(layout)
    if offline:
        plotly.offline.iplot(fig, filename='{}.html'.format(outfile))
    else:
        py.plot(fig, filename=outfile)
    return fig

###### Adding various objects to the plotly figure #######

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


def add_wells(well_dict, objects, structures):
    well_colors = cycle(sns.color_palette().as_hex())
    for i, (key, pts) in enumerate(well_dict.items()):
        try:
            x, y, z = zip(*pts)
        except ValueError:
            x, y, z, d = zip(*pts)
        if structures:
            col = 'gray'
        else:
            col = next(well_colors)
        if key.startswith('D'):
            group = 'CSD'
            viz = True
        elif key.startswith('B'):
            group = 'FS-B'
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
                                    line=dict(color=col, width=4),
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
            mesh = pd.read_csv(mesh_file, header=None, delimiter=' ')
            vertices = mesh[mesh.iloc[:, 0] == 'VRTX']
            triangles = mesh[mesh.iloc[:, 0] == 'TRGL']
            X = vertices.iloc[:, 1].values
            Y = vertices.iloc[:, 2].values
            Z = vertices.iloc[:, 3].values
            I = triangles.iloc[:, 1].values - 1
            J = triangles.iloc[:, 2].values - 1
            K = triangles.iloc[:, 3].values - 1
            objects.append(go.Mesh3d(
                x=X, y=Y, z=Z, i=I, j=J, k=K,
                name=mesh_name,
                color=col, opacity=0.3,
                alphahull=0,
                showlegend=True,
                hoverinfo='skip'))
    return objects


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


def add_catalog(catalog, dd_only, objects, surface):
    # Establish color scales from colorlover (import colorlover as cl)
    colors = cycle(cl.scales['11']['qual']['Paired'])
    pt_lists = []
    pt_list = []
    for ev in catalog:
        o = ev.origins[-1]
        ex = float(o.extra.hmc_east.value)
        ey = float(o.extra.hmc_north.value)
        ez = float(o.extra.hmc_elev.value)
        if dd_only and not o.method_id:
            print('Not accepting non-dd locations')
            continue
        elif dd_only and not o.method_id.id.endswith('GrowClust'):
            print('Not accepting non-GrowClust locations')
            continue
        try:
            m = ev.magnitudes[-1].mag
        except IndexError:
            print('No magnitude. Wont plot.')
            continue
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
        ticktext = [datetime.fromtimestamp(t) for t in tickvals]
        print(np.max(z))
        scat_obj = go.Scatter3d(x=np.array(x), y=np.array(y), z=np.array(z),
                                mode='markers',
                                name='Seismic event',
                                hoverinfo='text',
                                text=id,
                                marker=dict(color=t,
                                            cmin=min(tickvals),
                                            cmax=max(tickvals),
                                            size=(1.5 * np.array(m)) ** 2,
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
                                            colorscale='Cividis',
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


def dxf_to_xyz(mesh_file, mesh_name, datas):
    """Helper for reading dxf files of surf levels to xyz"""
    # SURF CAD files
    dxf = dxfgrabber.readfile(mesh_file)
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
                z=np.array(zs) / 3.28084,
                name=mesh_name,
                mode='lines',
                opacity=0.3,
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
                opacity=0.3,
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
    print(center)
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
