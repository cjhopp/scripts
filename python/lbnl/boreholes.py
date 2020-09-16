#!/usr/bin/python

"""
Utilities for handling various formats of borehole trajectory/orientation files
"""
import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from itertools import cycle
from glob import glob
from pathlib import Path
from matplotlib.dates import date2num, num2date


def depth_to_xyz(well_dict, well, depth):
    """
    Return xyz coords for depth in a given borehole

    :param well: Well string
    :param depth: Depth float
    :return:
    """
    easts, norths, zs, deps = np.hsplit(well_dict[well], 4)
    # Get closest depth point
    dists = np.squeeze(np.abs(depth - deps))
    x = easts[np.argmin(dists)][0]
    y = norths[np.argmin(dists)][0]
    z = zs[np.argmin(dists)][0]
    return (x, y, z)


def distance_to_borehole(well_dict, well, depth, gallery_pts,
                         excavation_times):
    """
    Calculate distance, x, y, azimuth and dip to SIMFIP from Gallery excavation
    front at Mont Terri

    :param well_dict: Output of parse_FSB_wells
    :param well: String of well to calculate distance to
    :param depth: Depth in well to calculate distances to
    :param gallery_pts: Path to file with 1-m spacing gallery distances
    :param excavation_times: Path to file with times and distances along gallery
        for G18 excavation
    :return:
    """
    borehole_xyz = depth_to_xyz(well_dict, well, depth)
    df_gallery = read_gallery_distances(gallery_pts)
    gallery_dist = df_gallery['distance']
    df_excavation = read_gallery_excavation(excavation_times).copy()
    exc_dist = df_excavation['distance [m]']
    # Calculate value of closest point along my line to measured dist
    dists_along = np.array([gallery_dist[np.argmin(np.abs(gallery_dist - d))]
                            for d in exc_dist])
    # Get X, Y, Z as array for each of these distances
    pts_along = np.vstack([df_gallery.loc[df_gallery['distance'] == d].values[0]
                           for d in dists_along])
    # Pythagoras dance
    bh_dx = borehole_xyz[0] - pts_along[:, 0]
    bh_dy = borehole_xyz[1] - pts_along[:, 1]
    bh_dz = borehole_xyz[2] - pts_along[:, 2]
    bh_dists = np.sqrt(bh_dx**2 + bh_dy**2 + bh_dz**2)
    bh_az = np.rad2deg(np.arctan(bh_dx / bh_dy))
    bh_plunge = np.rad2deg(np.arcsin(bh_dz / bh_dists))
    # Put back into DataFrame
    df_excavation.insert(0, 'dh', np.sqrt(bh_dx**2 + bh_dy**2))
    df_excavation.insert(0, 'dz', bh_dz)
    df_excavation.insert(0, 'Distance to SIMFIP', bh_dists)
    df_excavation.insert(0, 'Azimuth to SIMFIP', bh_az)
    df_excavation.insert(0, 'Plunge to SIMFIP', bh_plunge)
    df_excavation = df_excavation.rename(
        mapper={'distance [m]': 'Excavation distance [m]'}, axis='columns')
    return df_excavation


def read_gallery_excavation(path):
    """Parse gallery excavation progress"""
    df = pd.read_csv(path, parse_dates=[0])
    df = df.set_index('time')
    return df


def read_gallery_distances(path):
    """Helper to read points along gallery"""
    df = pd.read_csv(path)
    # Call elevation of center of gallery ~517m
    df.insert(2, 'Z', np.ones(df['X'].values.shape[0]) * 517.)
    return df[['X', 'Y', 'Z', 'distance']]


def calc_mesh_area(X, Y, Z):
    """Assume rotated mesh is rhombus and calculate area"""
    v1 = (X[0, 0], Y[0, 0], Z[0, 0])
    v2 = (X[0, 2], Y[0, 2], Z[0, 2])
    v3 = (X[2, 0], Y[2, 0], Z[2, 0])
    v4 = (X[2, 2], Y[2, 2], Z[2, 2])
    p = np.sqrt((v1[0] - v4[0])**2 +
                (v1[1] - v4[1])**2 +
                (v1[2] - v4[2])**2)
    q = np.sqrt((v2[0] - v3[0])**2 +
                (v2[1] - v3[1])**2 +
                (v2[2] - v3[2])**2)
    return p * q / 2


def scale_mesh(X, Y, Z, center):
    """Scale points of mesh to area of 1"""
    A = calc_mesh_area(X, Y, Z)
    scale = 1. / A
    Xs = ((X - center[0]) * scale) + center[0]
    Ys = ((Y - center[1]) * scale) + center[1]
    Zs = ((Z - center[2]) * scale) + center[2]
    return Xs.flatten(), Ys.flatten(), Zs.flatten()


def make_frac_mesh(center, dip_direction, dip):
    # Dip direction to strike
    strike = dip_direction - 90.
    if strike < 0.:
        strike += 360.
    # Helper to return X, Y, Z arrays to be plotted
    # 2m square grid around point
    x = np.linspace(center[0] - 0.5, center[0] + 0.5, 3)
    y = np.linspace(center[1] - 0.5, center[1] + 0.5, 3)
    # Mesh x and y arrays
    X, Y = np.meshgrid(x, y)
    dip_rad = np.deg2rad(dip)
    strike_rad = np.deg2rad(strike)
    # Normal to plane
    a = np.sin(dip_rad) * np.cos(strike_rad)  # East
    b = -np.sin(dip_rad) * np.sin(strike_rad)  # North
    c = np.cos(dip_rad)
    # Evaluate d at borehole xyz
    d = np.dot(np.array([a, b, c]), center)
    Z = (d - a * X - b * Y) / c
    return X, Y, Z


def structures_to_planes(path, well_dict):
    """
    Take the Optical TV picked structures for a well (dip direction-dip)
    and return lists of X, Y, Z arrays defining a plane for plotting.

    :param path: Path to Terratek excel
    :return: list of (X, Y, Z)
    """
    # Custom color palette similar to wellcad convention
    cols = {'open/undif. fracture': 'blue',
            'sealed fracture / vein': 'lightblue',
            'foliation / bedding': 'red',
            'induced fracture': 'magenta',
            'sedimentary structures/color changes undif.': 'green',
            'uncertain type': 'orange',
            'lithology change': 'yellow'}
    # Read excel sheet
    well = path.split('_')[-2]
    fracs = pd.read_excel(path, skiprows=np.arange(9),
                          usecols=np.arange(1, 9), header=None)
    frac_planes = []
    for i, frac in fracs.iterrows():
        dep = frac[1]
        dd = frac[4]
        dip = frac[5]
        frac_type = frac[7]
        # Get borehole xyz of feature
        try:
            bh_point = depth_to_xyz(well_dict, well, dep)
        except KeyError:
            print('No borehole info yet for {}'.format(well))
            return []
        X, Y, Z = make_frac_mesh(center=bh_point, dip_direction=dd, dip=dip)
        Xs, Ys, Zs = scale_mesh(X, Y, Z, bh_point)
        frac_planes.append((Xs, Ys, Zs, frac_type, cols[frac_type]))
    return frac_planes


def read_frac_cores(path, well):
    """
    Return dictionary of {well: {'fracture type': fracture density array}}

    :param path: Path to core fracture density excel file
    :return: dict
    """
    well_dict = {'All fractures': {}}
    fracs = pd.read_excel(path, sheet_name=None, skiprows=[2], header=1)
    for full_name, items in fracs.items():
        if 'BCS-{}'.format(well) == full_name:
            bin_centers = ((items['TopDepth'] +
                            items['BottomDepth']) / 2).values
            well_dict['All fractures'] = np.stack(
                (bin_centers, items['Total Counts Core 1m'])).T
    return well_dict


def calculate_frac_density(path, well_dict):
    """
    Return dict of {well: {'fracture type': fracture density array}}

    :param path:
    :param well_dict:
    :return:
    """
    well = path.split('_')[-2]
    fracs = pd.read_excel(path, skiprows=np.arange(9),
                          usecols=np.arange(1, 9), header=None)
    deps = fracs[1].values
    types = fracs[7].values
    unique_types = list(set(types))
    # 0.5 step and 0.5 overlap
    dep_bins = np.arange(0, well_dict[well][-1, -1], 0.5)
    frac_dict = {}
    # All fractures first
    total_density = np.array(
        [deps[np.where(np.logical_and(a - 1 <= deps, deps < a + 1))].shape[0]
         for a in dep_bins])
    frac_dict['All fractures'] = np.stack((dep_bins, total_density)).T
    for t in unique_types:
        ds = fracs[fracs[7] == t][1].values
        t_dens = np.array(
            [deps[np.where(np.logical_and(a - 1 <= ds, ds < a + 1))].shape[0]
             for a in dep_bins])
        frac_dict[t] = np.stack((dep_bins, t_dens)).T
    return frac_dict


def create_FSB_boreholes(gocad_dir='/media/chet/data/chet-FS-B/Mont_Terri_model/',
                         asbuilt_dir='/media/chet/data/chet-FS-B/wells/'):
    """
    Return dictionary of FSB well coordinates

    :param asbuilt_dir: Directory holding the gamma logs for each well
    """
    if not os.path.isdir(asbuilt_dir):
        asbuilt_dir = '/media/chet/hdd/seismic/chet_FS-B/wells/'
    if not os.path.isdir(asbuilt_dir):
        asbuilt_dir = 'data/chet-FS-B/wells'
    excel_asbuilts = glob('{}/**/*Gamma_Deviation.xlsx'.format(asbuilt_dir))
    well_dict = {}
    if not os.path.isdir(gocad_dir):
        gocad_dir = '/media/chet/hdd/seismic/chet_FS-B/Mont_Terri_model'
    if not os.path.isdir(gocad_dir):
        gocad_dir = 'data/chet-FS-B/Mont_Terri_model'
    gocad_asbuilts =  glob('{}/*.wl'.format(gocad_dir))
    for gocad_f in gocad_asbuilts:
        name = str(gocad_f).split('-')[-1].split('.')[0]
        well_dict[name] = []
        # Multispace delimiter
        top = pd.read_csv(gocad_f, header=None, skiprows=np.arange(13),
                          delimiter='\s+', index_col=False, engine='python',
                          nrows=1)
        rows = pd.read_csv(gocad_f, header=None, skiprows=np.arange(14),
                           delimiter='\s+', index_col=False, engine='python',
                           skipfooter=1)
        lab, x_top, y_top, z_top = top.values.flatten()
        well_dict[name] = np.stack(((x_top + rows.iloc[:, 3]).values,
                                    (y_top + rows.iloc[:, 4]).values,
                                    rows.iloc[:, 2].values,
                                    rows.iloc[:, 1].values)).T
        if well_dict[name].shape[0] < 1000:  # Read in gamma instead
            # If so, make a more highly-sampled interpolation
            x, y, z, d = zip(*well_dict[name])
            td = d[-1]
            if td == 'Top':
                td = float(d[-3])
            well_dict[name] = np.stack((np.linspace(x_top, x[-1], 1000),
                                        np.linspace(y_top, y[-1], 1000),
                                        np.linspace(z_top, z[-1], 1000),
                                        np.linspace(0, td, 1000))).T
    return well_dict


def parse_surf_boreholes(file_path):
    """
    Parse the surf 4850 xyz file to dict of hole: {[(x, y, z), (x1, y1, z1)]}

    :param file_path: Path to borehole text file
    :return: dict
    """
    well_dict = {}
    with open(file_path, 'r') as f:
        next(f)
        for ln in f:
            ln = ln.rstrip('\n')
            line = ln.split(',')
            xm = float(line[3]) / 3.28084
            ym = float(line[4]) / 3.28084
            zm = float(line[5]) / 3.28084
            dp = float(line[2])
            name = line[0].split('-')[1]
            if name in well_dict:
                well_dict[name] = np.concatenate(
                    (well_dict[name],
                     np.array([xm, ym, zm, dp]).reshape(1, 4)))
            else:
                well_dict[name] = np.array([xm, ym, zm, dp]).reshape(1, 4)
    return well_dict


def wells_4850_to_gmt(outfile):
    colors = cycle(sns.color_palette())
    well_file = '/media/chet/data/chet-collab/boreholes/surf_4850_wells.csv'
    wells = parse_surf_boreholes(well_file)
    with open(outfile, 'w') as f:
        for key, pts in wells.items():
            col = np.array(next(colors))
            col *= 255
            col_str = '{}/{}/{}'.format(int(col[0]), int(col[1]), int(col[2]))
            f.write('>-W1.0,{} -L{}\n'.format(col_str, key))
            for pt in pts:
                f.write('{} {}\n'.format(pt[0], pt[1]))
    return

# Plotting

def plot_excavation_vector(df_excavation):
    """Plot excavation progress with time and lower hemi projection"""
    fig = plt.figure(figsize=(7, 10))
    ax1 = fig.add_subplot(211)
    df_excavation[['Distance to SIMFIP', 'Excavation distance [m]',
                   'dh', 'dz']].plot(ax=ax1)
    ax1.set_ylabel('Distance [m]')
    ax2 = fig.add_subplot(212, projection='polar')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    times = df_excavation.index.values
    colors = date2num(times)
    dots = ax2.scatter(
        np.deg2rad(df_excavation['Azimuth to SIMFIP']),
        df_excavation['Plunge to SIMFIP'],
        c=colors)
    ticks = np.linspace(np.nanmin(colors),
                        np.nanmax(colors), 5)
    cax = plt.colorbar(dots, ax=ax2, ticks=ticks)
    ticklabs = [num2date(t).strftime('%b-%d') for t in ticks]
    cax.ax.set_yticklabels(ticklabs)
    ax2.set_ylim([-90., 0])
    ax2.set_yticklabels([])
    plt.subplots_adjust(hspace=0.4, wspace=0.6)
    plt.show()
    return