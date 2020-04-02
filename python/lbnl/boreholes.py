#!/usr/bin/python

"""
Utilities for handling various formats of borehole trajectory/orientation files
"""
import os

import numpy as np
import seaborn as sns
import pandas as pd

from itertools import cycle
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


def make_frac_mesh(center, strike, dip):
    # Helper to return X, Y, Z arrays to be plotted
    # 2m square grid around point
    x = np.linspace(center[0] - 0.5, center[0] + 0.5, 3)
    y = np.linspace(center[1] - 0.5, center[1] + 0.5, 3)
    # Mesh x and y arrays
    X, Y = np.meshgrid(x, y)
    dip_rad = np.deg2rad(dip)
    strike_rad = np.deg2rad(strike)
    # Normal to plane
    a = -np.sin(dip_rad) * np.sin(strike_rad)
    b = np.sin(dip_rad) * np.cos(strike_rad)
    c = -np.cos(dip_rad)
    # Evaluate d at borehole xyz
    d = np.dot(np.array([a, b, c]), center)
    Z = (d - a * X - b * Y) / c
    return X.flatten(), Y.flatten(), Z.flatten()


def structures_to_planes(path, well_dict):
    """
    Take the Optical TV picked structures for a well (strike-dip) and return
    lists of X, Y, Z arrays defining a plane for plotting.

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
        strike = frac[4]
        dip = frac[5]
        frac_type = frac[7]
        # Get borehole xyz of feature
        try:
            bh_point = depth_to_xyz(well_dict, well, dep)
        except KeyError:
            print('No borehole info yet for {}'.format(well))
            return []
        X, Y, Z = make_frac_mesh(center=bh_point, strike=strike, dip=dip)
        frac_planes.append((X, Y, Z, frac_type, cols[frac_type]))
    return frac_planes


def create_FSB_boreholes(method='asbuilt',
                         asbuilt_dir='/media/chet/data/chet-FS-B/wells/'):
    """
    Return dictionary of FSB well coordinates

    :param method: From asbuilt or asplanned specs
    :param asbuilt_dir: Directory holding the gamma logs for each well
    """
    well_dict = {
                 # As planned
                 'B1': [(2579345.22, 1247580.39, 513.20, 0),
                        (2579345.22, 1247580.39, 450.00, 0)],
                 'B2': [(2579334.48, 1247570.98, 513.20, 0),
                        (2579329.36, 1247577.86, 460.41, 0)],
                 # Changed to as built B3-7 1-14-20 CJH
                 'B3': [(2579324.915, 1247611.678, 514.13, 0),
                        (2579324.01, 1247557.61, 450.05, 0)],
                 'B4': [(2579325.497, 1247612.048, 514.07, 0),
                         (2579337.15, 1247569.67, 448.33, 0)],
                 'B5': [(2579332.571, 1247597.289, 513.78, 0),
                        (2579319.59, 1247557.19, 472.50, 0)],
                 'B6': [(2579333.568, 1247598.048, 513.70, 0),
                        (2579337.52, 1247568.85, 474.53, 0)],
                 'B7': [(2579335.555, 1247599.383, 513.70, 0),
                        (2579352.08, 1247578.44, 475.78, 0)],
                 # As planned
                 # 'B8': [(2579334.00, 1247602.00, 513.78, 0),
                 #        (2579326.50, 1247563.50, 472.50, 0)],
                 # 'B9': [(2579328.00, 1247609.00, 513.20, 0),
                 #        (2579335.00, 1247570.00, 458.00, 0)],
                 # As built tops: CJH 3-2-20
                 'B8': [(2579331.9512, 1247600.6754, 513.7908, 0),
                        (2579326.50, 1247563.50, 472.50, 0)],
                 'B9': [(2579327.8493, 1247608.9225, 513.9813, 0),
                        (2579335.00, 1247570.00, 458.00, 0)]
    }
    if method == 'asplanned':
        return well_dict
    elif method == 'asbuilt':
        if not os.path.isdir(asbuilt_dir):
            asbuilt_dir = '/media/chet/hdd/seismic/chet_FS-B/wells/'
        excel_asbuilts = []
        for fname in Path(asbuilt_dir).rglob(
                '*Gamma_Deviation.xlsx'):
            excel_asbuilts.append(fname)
        for excel_f in excel_asbuilts:
            name = str(excel_f).split('-')[-1][:2]
            top = well_dict[name][0]
            df = pd.read_excel(excel_f, header=6, skiprows=[7])
            well_dict[name] = np.stack(((top[0] + df['Easting']).values,
                                        (top[1] + df['Northing']).values,
                                        (top[2] - df['TVD']).values,
                                        df['Depth'].values)).T
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
