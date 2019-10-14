#!/usr/bin/python

"""
Utilities for handling various formats of borehole trajectory/orientation files
"""

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