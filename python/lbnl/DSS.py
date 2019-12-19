#!/usr/bin/python
"""
Functions for processing and plotting DSS data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from obspy import Stream, Trace
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import detrend
from datetime import datetime
from matplotlib.colors import ListedColormap

chan_map_fsb = {'B3': (237.7, 404.07), 'B4': (413.52, 571.90),
                'B5': (80.97, 199.63), 'B6': (594.76, 694.32),
                'B7': (700.43, 793.47)}

chan_map_maria = {'B3': (232.21, 401.37), 'B4': (406.56, 566.58),
                  'B5': (76.46, 194.11), 'B6': (588.22, 688.19),
                  'B7': (693.37, 789.86)}


def read_ascii(path, header=42, encoding='iso-8859-1'):
    """Read in a raw DSS file (flipped about axis 1 for left-to-right time"""
    return np.flip(np.loadtxt(path, skiprows=header, encoding=encoding), 1)


def datetime_parse(t):
    # Parse the date format of the DSS headers
    return datetime.strptime(t, '%Y/%m/%d %H:%M:%S')


def read_times(path, encoding='iso-8859-1'):
    """Read timestamps from ascii header"""
    strings = np.genfromtxt(path, skip_header=10, max_rows=1, encoding=encoding,
                            dtype=None, delimiter='\t')
    return np.array([datetime_parse(t) for t in strings[1:-1]])[::-1]


def plot_DSS(path, well='all',
             date_range=(datetime(2019, 5, 21), datetime(2019, 6, 5)),
             denoise_method='detrend', vrange=(-30, 30), title=None):
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
    # Take first column as the length along the fiber and remove it from data
    depth = data[:, -1]
    data = data[:, :-1]
    # Take selected channels
    if well == 'all':
        channel_range = (0, -1)
    else:
        start_chan = np.abs(depth - chan_map_fsb[well][0])
        end_chan = np.abs(depth - chan_map_fsb[well][1])
        # Find the closest integer channel to meter mapping
        channel_range = (np.argmin(start_chan), np.argmin(end_chan))
    data = data[channel_range[0]:channel_range[1], :]
    depth = depth[channel_range[0]:channel_range[1]]
    if date_range:
        indices = np.where((date_range[0] < times) & (times < date_range[1]))
        times = times[indices]
        data = data[:, indices]
        data = np.squeeze(data)
    mpl_times = mdates.date2num(times)
    if denoise_method:
        data = denoise(data, denoise_method)
    cmap = ListedColormap(sns.color_palette("RdBu_r", 21).as_hex())
    im = plt.imshow(data, cmap=cmap,
                    extent=[mpl_times[0], mpl_times[-1],
                            depth[-1] - depth[0], 0],
                    aspect='auto', vmin=vrange[0], vmax=vrange[1])
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.set_ylabel('Length along fiber [m]')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'$\mu\varepsilon$', fontsize=16, fontweight='bold')
    if not title:
        ax.set_title('DSS well {}: {}'.format(well, denoise_method))
    plt.tight_layout()
    return


def denoise(data, method='detrend'):
    if method == 'demean':
        mean = data.mean(axis=0)
        data -= mean[np.newaxis, :]
    elif method == 'demedian':
        median = np.median(data, axis=0)
        data -= median[np.newaxis, :]
    elif method == 'detrend':
        data = np.apply_along_axis(detrend, 0, data)
    elif method == 'gaussian':
        data = gaussian_filter(data, 2)
    elif method == 'median':
        data = median_filter(data, 2)
    return data