#!/usr/bin/python

"""
Utilities for handling various formats of borehole trajectory/orientation files
"""
import os

import numpy as np
import seaborn as sns
import pandas as pd

from itertools import cycle
from glob import glob
from pathlib import Path


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
    excel_asbuilts = glob('{}/**/*Gamma_Deviation.xlsx'.format(asbuilt_dir))
    well_dict = {}
    if not os.path.isdir(gocad_dir):
        gocad_dir = '/media/chet/hdd/seismic/chet_FS-B/Mont_Terri_model'
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
            name = line[0].split('-')[1]
            if name in well_dict:
                well_dict[name].append((xm, ym, zm))
            else:
                well_dict[name] = [(xm, ym, zm)]
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
