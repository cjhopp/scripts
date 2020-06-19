#!/usr/bin/python

"""
Functions for processing and plotting DTS data
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.dates import num2date, date2num
from scipy.io import loadmat

surf_wells = ['OT', 'OB', 'PSB', 'PST', 'PDB', 'PDT']

attr_map = {'OT': ['otDepths', 'otTemps'], 'OB': ['obDepths', 'obTemps'],
            'PDB': ['pdbDepths', 'pdbTemps'], 'PDT': ['pdtDepths', 'pdtTemps'],
            'PSB': ['psbDepths', 'psbTemps'], 'PST': ['pstDepths', 'pstTemps'],}

def read_struct(f):
    # Return the parts of the struct we actually want
    struct = loadmat(f, struct_as_record=False,
                     squeeze_me=True)['monthSet'].dayCell
    well_dict = {w: {'depth': [], 'temp': []}
                 for w in surf_wells}
    # Concatenate each day cell along zero axis and rotate into preferred shape
    for i, day_struct in enumerate(struct):
        if i == 0:
            for w, w_dict in well_dict.items():
                w_dict['depth'] = getattr(day_struct, attr_map[w][0])
                w_dict['temp'] = getattr(day_struct, attr_map[w][1]).T
        for w, w_dict in well_dict.items():
            w_dict['temp'] = np.concatenate((w_dict['temp'],
                                             getattr(day_struct,
                                                     attr_map[w][1]).T),
                                            axis=1)
    well_dict['times'] = np.array([])
    for s in struct:
        well_dict['times'] = np.concatenate((well_dict['times'],
                                             num2date(s.dates)))
    return well_dict


def plot_DTS(well_dict, well):
    """
    Plot the selected
    :param struct:
    :param well:
    :return:
    """
    times = well_dict['times']
    temps = well_dict[well]['temp']
    depths = well_dict[well]['depth']
    fig, ax = plt.subplots()
    im = plt.imshow(temps, extent=(date2num(times[0]), date2num(times[-1]),
                                   depths[-1], 0), axes=ax)
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Degrees C', fontsize=12)
    ax.xaxis_date()
    plt.show()
    return