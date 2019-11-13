#!/usr/bin/python

"""
Utilities for handling various formats of borehole trajectory/orientation files
"""
import numpy as np
import seaborn as sns

from itertools import cycle

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
