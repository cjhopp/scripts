#!/usr/bin/python

"""
Plotting functions for the lbnl module
"""

import numpy as np
import colorlover as cl
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go

from itertools import cycle
from datetime import datetime
from scipy.linalg import lstsq

# Local imports (assumed to be in python path)
from lbnl.boreholes import parse_surf_boreholes

def plot_surf_3D(catalog, inventory, well_file, outfile, xlims=None, ylims=None,
                 zlims=None, video=False, animation=False, title=None,
                 offline=False, dd_only=False, surface='plane'):
    """
    Plot a list of catalogs as a plotly 3D figure
    :param cluster_cats: List of obspy.event.Catalog objects
    :param outfile: Name of the output figure
    :param field: Either 'Rot' or 'Nga' depending on which field we want
    :param xlims: List of [min, max] longitude to plot
    :param ylims: List of [max, min] latitude to plot
    :param zlims: List of [max neg., max pos.] depths to plot
    :param wells: Boolean for whether to plot the wells
    :param video: Deprecated because it's impossible to deal with
    :param animation: (See above)
    :param title: Plot title
    :param offline: Boolean for whether to plot to plotly account (online)
        or to local disk (offline)
    :param dd_only: Are we only plotting dd locations?
    :param surface: What type of surface to fit to points? Supports 'plane'
        and 'ellipsoid' for now.
    :return:
    """
    pt_lists = []
    # Establish color scales from colorlover (import colorlover as cl)
    colors = cycle(cl.scales['11']['qual']['Paired'])
    well_colors = cl.scales['9']['seq']['BuPu']
    if not title:
        title = '3D Plot'
    # Make well point lists and add to Figure
    datas = []
    wells = parse_surf_boreholes(well_file)
    for i, (key, pts) in enumerate(wells.items()):
        x, y, z = zip(*pts)
        datas.append(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                  name='Borehole: {}'.format(key),
                                  line=dict(color=next(colors), width=7)))
    # Do the same for the inventory
    sta_list = []
    for sta in inventory[0]: # Assume single network for now
        sx = float(sta.extra.hmc_east.value)
        sy = float(sta.extra.hmc_north.value)
        sz = float(sta.extra.hmc_elev.value)
        name = sta.code
        sta_list.append((sx, sy, sz, name))
    stax, stay, staz, nms = zip(*sta_list)
    datas.append(go.Scatter3d(x=np.array(stax), y=np.array(stay),
                              z=np.array(staz),
                              mode='markers',
                              name='Station',
                              hoverinfo='text',
                              text=nms,
                              marker=dict(color='black',
                                size=3.,
                                symbol='diamond',
                                line=dict(color='gray',
                                          width=1),
                                opacity=0.9)))
    # If no limits specified, take them from boreholes
    if not xlims:
        xs = [pt[0] for bh, pts in wells.items() for pt in pts]
        ys = [pt[1] for bh, pts in wells.items() for pt in pts]
        xlims = [min(xs), max(xs)]
        ylims = [min(ys), max(ys)]
        zlims = [60, 130]
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
        if (xlims[0] < ex < xlims[1]
            and ylims[0] < ey < ylims[1]
            and zlims[0] < ez < zlims[1]):
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
        scat_obj = go.Scatter3d(x=np.array(x), y=np.array(y), z=z,
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
        datas.append(scat_obj)
        if surface == 'plane':
            if len(x) <= 2:
                continue # Cluster just 1-2 events
            # Fit plane to this cluster
            X, Y, Z, stk, dip = pts_to_plane(np.array(x), np.array(y),
                                             np.array(z))
            # Add mesh3d object to plotly
            datas.append(go.Mesh3d(x=X, y=Y, z=Z, color=clust_col,
                                   opacity=0.3, delaunayaxis='z',
                                   text='Strike: {}, Dip {}'.format(stk, dip),
                                   showlegend=True))
        elif surface == 'ellipsoid':
            if len(x) <= 2:
                continue # Cluster just 1-2 events
            # Fit plane to this cluster
            center, radii, evecs, v = pts_to_ellipsoid(np.array(x),
                                                       np.array(y),
                                                       np.array(z))
            X, Y, Z = ellipsoid_to_pts(center, radii, evecs)
            # Add mesh3d object to plotly
            datas.append(go.Mesh3d(x=X, y=Y, z=Z, color=clust_col,
                                   opacity=0.3, delaunayaxis='z',
                                   # text='A-axis Trend: {}, Plunge {}'.format(stk, dip),
                                   showlegend=True))
        else:
            print('No surfaces fitted')
    xax = go.layout.scene.XAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Easting (m)',
                                autorange=True, range=xlims)
    yax = go.layout.scene.YAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Northing (m)',
                                autorange=True, range=ylims)
    zax = go.layout.scene.ZAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Elevation (m)',
                                autorange=True, range=zlims)
    layout = go.Layout(scene=dict(xaxis=xax, yaxis=yax, zaxis=zax,
                                  bgcolor="rgb(244, 244, 248)"),
                       autosize=True,
                       title=title)
    # This is a bunch of hooey
    if video and animation:
        layout.update(
            updatemenus=[{'type': 'buttons', 'showactive': False,
                          'buttons': [{'label': 'Play',
                                       'method': 'animate',
                                       'args': [None,
                                                {'frame': {'duration': 1,
                                                           'redraw': True},
                                                 'fromcurrent': True,
                                                 'transition': {
                                                    'duration': 0,
                                                    'mode': 'immediate'}}
                                                          ]},
                                      {'label': 'Pause',
                                       'method': 'animate',
                                       'args': [[None],
                                                {'frame': {'duration': 0,
                                                           'redraw': True},
                                                           'mode': 'immediate',
                                                           'transition': {
                                                               'duration': 0}}
                                                          ]}]}])
    # Start figure
    fig = go.Figure(data=datas, layout=layout)
    # trace = fig.data[-1]
    # new_tick_text = [datetime.fromtimestamp(t) for t in
    #                  trace.marker.colorbar.tickvals]
    # trace.marker.colorbar.ticktext = new_tick_text
    if video and animation:
        zoom = 2
        frames = [dict(layout=dict(
            scene=dict(camera={'eye':{'x': np.cos(rad) * zoom,
                                      'y': np.sin(rad) * zoom,
                                      'z': 0.2}})))
                  for rad in np.linspace(0, 6.3, 630)]
        fig.frames = frames
        if offline:
            plotly.offline.plot(fig, filename=outfile)
        else:
            py.plot(fig, filename=outfile)
    elif video and not animation:
        print('You dont need a video')
    else:
        if offline:
            plotly.offline.iplot(fig, filename=outfile)
        else:
            py.plot(fig, filename=outfile)
    return

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