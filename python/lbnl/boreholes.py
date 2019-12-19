#!/usr/bin/python

"""
Utilities for handling various formats of borehole trajectory/orientation files
"""
import numpy as np
import seaborn as sns

from itertools import cycle


def create_FSB_boreholes():
    """
    Return dictionary of FSB well coordinates
    """
    well_dict = {'B1': [(2579345.22, 1247580.39, 513.20),
                        (2579345.22, 1247580.39, 450.00)],
                 'B2': [(2579334.48, 1247570.98, 513.20),
                        (2579329.36, 1247577.86, 460.41)],
                 'B3': [(2579324.91, 1247611.68, 514.13),
                        (2579322.61, 1247556.79, 449.53)],
                 'B4': [(2579325.50, 1247612.05, 514.07),
                        (2579338.71, 1247569.11, 447.96)],
                 'B5': [(2579332.57, 1247597.29, 513.78),
                        (2579321.52, 1247556.01, 473.52)],
                 'B6': [(2579334.35, 1247598.44, 513.72),
                        (2579338.50, 1247569.01, 473.70)],
                 'B7': [(2579336.22, 1247599.75, 513.76),
                        (2579351.79, 1247579.12, 474.15)],
                 'B8': [(2579334.00, 1247602.00, 513.78),
                        (2579326.50, 1247563.50, 472.50)],
                 'B9': [(2579328.00, 1247609.00, 513.20),
                        (2579335.00, 1247570.00, 458.00)]
                 }
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
