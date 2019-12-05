#!/usr/bin/python
"""
Functions for processing and plotting DSS data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import Stream, Trace
from scipy.signal import detrend
from datetime import datetime

chan_map_fsb = {'B3': [237.7, 404.07], 'B4': [413.52, 571.90],
                'B5': [80.97, 199.63], 'B6': [594.76, 694.32],
                'B7': [700.43, 793.47]}


def read_ascii(path, header=42, encoding='iso-8859-1'):
    """Read in a raw DSS file"""
    return np.loadtxt(path, skiprows=header, encoding=encoding)[::-1]


def datetime_parse(t):
    # Parse the date format of the DSS headers
    return datetime.strptime(t, '%Y/%m/%d %H:%M:%S')


def read_times(path, encoding='iso-8859-1'):
    """Read timestamps from ascii header"""
    strings = np.genfromtxt(path, skip_header=10, max_rows=1, encoding=encoding,
                            dtype=None, delimiter='\t')
    return np.array([datetime_parse(t) for t in strings[1:-1]])[::-1]


def plot_DSS(path, channel_range=(0, -1), date_range=None,
             denoise_method='demean'):
    """
    Plot a colormap of DSS data

    :param path: Path to raw data file
    :param channel_range: [start channel, end channel]
    :param date_range: [start date, end date]
    :return:
    """
    fig, ax = plt.subplots()
    data = read_ascii(path)
    times = read_times(path)
    # Take selected channels
    data = data[channel_range[0]:channel_range[1], :]
    if date_range:
        times = times[np.where(date_range[0] < times < date_range[1])]
        data = data[:, np.where(date_range[0] < times < date_range[1])]
    mpl_times = mdates.date2num(times)
    if denoise_method:
        data = denoise(data, denoise_method)
    im = plt.imshow(data, cmap='seismic',
                    extent=[mpl_times[0], mpl_times[-1], data.shape[0], 0],
                    aspect='auto')
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.colorbar()
    return data


def denoise(data, method='detrend'):
    if method == 'demean':
        mean = data.mean(axis=0)
        data -= mean[np.newaxis, :]
    elif method == 'detrend':
        data = np.apply_along_axis(detrend, 0, data)
    return data