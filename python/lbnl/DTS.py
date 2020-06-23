#!/usr/bin/python

"""
Functions for processing and plotting DTS data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.io import loadmat
from datetime import datetime, timedelta
from matplotlib.dates import num2date, date2num

surf_wells = ['OT', 'OB', 'PSB', 'PST', 'PDB', 'PDT']

attr_map = {'OT': ['otDepths', 'otTemps'], 'OB': ['obDepths', 'obTemps'],
            'PDB': ['pdbDepths', 'pdbTemps'], 'PDT': ['pdtDepths', 'pdtTemps'],
            'PSB': ['psbDepths', 'psbTemps'], 'PST': ['pstDepths', 'pstTemps']}


def datenum_to_datetime(datenums):
    # Helper to correctly convert matlab datenum to python datetime
    # SO source:
    # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    return [datetime.fromordinal(int(d)) +
            timedelta(days=d % 1) - timedelta(days=366)
            for d in datenums]


def read_struct(f, date_range=None):
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
                # Correct for wrong years (this will come back to bite me...)
                w_dict['times'] = datenum_to_datetime(day_struct.dates)
        for w, w_dict in well_dict.items():
            w_dict['temp'] = np.concatenate((w_dict['temp'],
                                             getattr(day_struct,
                                                     attr_map[w][1]).T),
                                            axis=1)
            dates = datenum_to_datetime(day_struct.dates)
            w_dict['times'] = np.concatenate((w_dict['times'], dates))
    return well_dict


def plot_DTS(well_dict, well, measure='dt'):
    """
    Plot the selected
    :param struct:
    :param well:
    :param measure: We plotting 'dt' or 'abs_t'
    :return:
    """
    times = well_dict['times']
    temps = well_dict[well]['temp']
    if measure == 'dt':
        temps = np.diff(temps, axis=1)
    depths = well_dict[well]['depth']
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(temps, extent=(date2num(times[0]), date2num(times[-1]),
                                  depths[-1], 0))
    ax.set_aspect('auto')
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Degrees C', fontsize=12)
    # Formatting
    date_formatter = mdates.DateFormatter('%b-%d %H')
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(date_formatter)
    plt.show()
    return