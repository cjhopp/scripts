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
from matplotlib.dates import num2date
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

# Local imports
from lbnl.simfip import read_excavation, plot_displacement_components

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


def plot_DSS(path, well='all', inset_channels=True, simfip=False,
             date_range=(datetime(2019, 5, 19), datetime(2019, 6, 5)),
             denoise_method='demedian', vrange=(-60, 60), title=None):
    """
    Plot a colormap of DSS data

    :param path: Path to raw data file
    :param well: Which well to plot
    :param inset_channels: Bool for picking channels to plot in separate axes
    :param simfip: Give path to data file if simfip data over same timespan
    :param date_range: [start date, end date]
    :param denoise_method: String stipulating the method in denoise() to use
    :param vrange: Colorbar range (in microstrains)
    :param title: Title of plot

    :return:
    """
    fig = plt.figure(constrained_layout=True, figsize=(6, 12))
    if inset_channels and simfip:
        # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 12))
        gs = GridSpec(ncols=5, nrows=9, figure=fig)
        axes1 = fig.add_subplot(gs[:3, :-1])
        axes2 = fig.add_subplot(gs[3:6, :-1], sharex=axes1)
        axes3 = fig.add_subplot(gs[6:, :-1], sharex=axes1)
        cax = fig.add_subplot(gs[:6, -1])
        df = read_excavation(simfip)
    elif inset_channels:
        gs = GridSpec(ncols=5, nrows=6, figure=fig)
        axes1 = fig.add_subplot(gs[:3, :-1])
        axes2 = fig.add_subplot(gs[3:6, :-1])
        cax = fig.add_subplot(gs[:, -1])
        # fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 8))
    else:
        gs = GridSpec(ncols=5, nrows=3, figure=fig)
        axes1 = fig.add_subplot(gs[:3, :-1])
        cax = fig.add_subplot(gs[:, -1])
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
    im = axes1.imshow(data, cmap=cmap,
                      extent=[mpl_times[0], mpl_times[-1],
                              depth[-1] - depth[0], 0],
                      aspect='auto', vmin=vrange[0], vmax=vrange[1])
    # If simfip, plot these data here
    if simfip:
        plot_displacement_components(df, starttime=date_range[0],
                                     endtime=date_range[1], new_axes=axes3,
                                     remove_clamps=False,
                                     rotated=True)
    date_formatter = mdates.DateFormatter('%b-%d %H')
    fig.axes[-2].xaxis_date()
    fig.axes[-2].xaxis.set_major_formatter(date_formatter)
    plt.setp(fig.axes[-2].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes1.set_ylabel('Length along fiber [m]', fontsize=16)
    if simfip:
        axes2.set_ylabel(r'$\mu\varepsilon$', fontsize=16)
        axes3.xaxis_date()
        axes3.xaxis.set_major_formatter(date_formatter)
        plt.setp(axes3.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.setp(axes1.get_xticklabels(), visible=False)
        plt.setp(axes2.get_xticklabels(), visible=False)
    elif inset_channels:
        axes2.xaxis_date()
        axes2.xaxis.set_major_formatter(date_formatter)
        plt.setp(axes2.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.setp(axes1.get_xticklabels(), visible=False)
    else:
        axes1.xaxis_date()
        axes1.xaxis.set_major_formatter(date_formatter)
        plt.setp(axes1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel(r'$\mu\varepsilon$', fontsize=16, fontweight='bold')
    if not title:
        axes1.set_title('DSS well {}: {}'.format(well, denoise_method))
    plt.tight_layout()
    # If plotting 1D channel traces, do this last
    if inset_channels:
        # Grid lines on axes 1
        axes2.grid(which='both', axis='y')

        # Define class for plotting new traces
        class TracePlotter():
            def __init__(self, figure, data, depth, cmap):
                self.figure = figure
                self.cmap = cmap
                self.data = data
                self.depth = depth - depth[0]
                self.xlim = self.figure.axes[0].get_xlim()
                self.times = np.linspace(self.xlim[0], self.xlim[1],
                                         data.shape[1])
                self.cid = self.figure.canvas.mpl_connect('button_press_event',
                                                          self)

            def __call__(self, event):
                print('click', event.xdata, event.ydata)
                global counter
                if event.inaxes != self.figure.axes[0]:
                    return
                # Get channel corresponding to ydata (which was modified to
                # units of meters during imshow...?
                chan_dist = np.abs(self.depth - event.ydata)
                chan = np.argmin(chan_dist)
                trace = self.data[chan, :]
                depth = self.depth[chan]
                # Plot trace for this channel colored by strain as LineCollection
                points = np.array([self.times, trace]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=self.cmap,
                                    label='Depth {:0.2f}'.format(depth),
                                    norm=plt.Normalize(vmin=vrange[0],
                                                       vmax=vrange[1]))
                # Set the values used for colormapping
                lc.set_array(trace)
                lc.set_linewidth(1.5)
                line = self.figure.axes[1].add_collection(lc)
                self.figure.axes[1].set_ylim([-100, 100]) # Need dynamic way
                self.figure.axes[1].legend()
                self.figure.canvas.draw()
                counter += 1

        # Make a better cursor for picking channels
        class Cursor(object):
            def __init__(self, ax, fig):
                self.ax = ax
                self.figure = fig
                self.lx = ax.axhline(ax.get_ylim()[0], color='k')  # the horiz line
                self.ly = ax.axvline(ax.get_xlim()[0], color='k')  # the vert line

            def mouse_move(self, event):
                if event.inaxes != self.ax:
                    return

                x, y = event.xdata, event.ydata
                # update the line positions
                self.lx.set_ydata(y)
                self.ly.set_xdata(num2date(x))

                self.figure.canvas.draw()

        # Connect cursor to ax1
        cursor = Cursor(axes1, fig)
        fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)

        global counter
        counter = 0 # Click counter for trace spacing
        plotter = TracePlotter(fig, data, depth, cmap)
        plt.show()
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