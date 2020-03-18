#!/usr/bin/python
"""
Functions for processing and plotting DSS data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from obspy import Stream, Trace
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import detrend, welch
from scipy.stats import median_absolute_deviation
from datetime import datetime, timedelta
from itertools import cycle
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


def read_metadata(path, encoding='iso-8859-1'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.read().split('\n')
        for ln in lines:
            line = ln.split()
            if len(line) > 0:
                if line[:2] == ['Measurement', 'Mode']:
                    mode = line[-1]
                elif line[:2] == ['Data', 'type']:
                    type = ' '.join(line[2:])
    return mode, type

def extract_wells(path, wells):
    """
    Helper to extract only the channels in specific wells
    """
    data = read_ascii(path)
    times = read_times(path)
    mode, type = read_metadata(path)
    # Take first column as the length along the fiber and remove it from data
    depth = data[:, -1]
    data = data[:, :-1]
    # First realign this shiz....soooooo slow
    data = madjdabadi_realign(data)
    # Take selected channels
    if wells == 'all':
        channel_ranges = [('all', (0, -1))]
    else:
        channel_ranges = []
        for well in wells:
            start_chan = np.abs(depth - chan_map_fsb[well][0])
            end_chan = np.abs(depth - chan_map_fsb[well][1])
            # Find the closest integer channel to meter mapping
            channel_range = (np.argmin(start_chan), np.argmin(end_chan))
            channel_ranges.append((well, channel_range))
    well_data = {'times': times, 'mode': mode, 'type': type}
    for rng in channel_ranges:
        data_tmp = data[rng[1][0]:rng[1][1], :]
        depth_tmp = depth[rng[1][0]:rng[1][1]]
        # Get median absolute deviation averaged across all channels
        noise = estimate_noise(data_tmp)
        well_data[rng[0]] = {'data': data_tmp, 'depth': depth_tmp,
                             'noise': noise}
    return well_data


def plot_wells_over_time(well_data, wells, date_range=(datetime(2019, 5, 19),
                                                  datetime(2019, 6, 5)),
                         vrange=(-40, 40), pick_dict=None, alpha=1.):
    """
    Plot wells side-by-side with each curve over a given time slice

    :param path: Path to raw data file
    :param wells: List of well names to plot
    :param date_range: List of [start datetime, end datetime]
    :param vrange: Xlims for all axes
    :param pick_dict: Dictionary {well name: [pick depths, ...]}
    :return:
    """
    # Read in data
    # Make list of times within date range
    times = well_data['times']
    times = [t for t in times if date_range[0] < t < date_range[1]]
    # Initialize the figure
    fig, axes = plt.subplots(nrows=1, ncols=len(wells) * 2, sharey=True,
                             sharex=True, figsize=(len(wells) * 2, 8))
    # Cmap
    cmap = sns.cubehelix_palette(as_cmap=True)
    for i, t in enumerate(times):
        pick_col = cmap(float(i) / len(times))
        plot_well_timeslices(well_data, wells, date=t, vrange=vrange,
                             pick_dict=pick_dict, fig=fig, pick_col=pick_col,
                             alpha=alpha, plot_noise=False)
    # Hack-tron 5000
    if len(times) % 2 == 0:
        axes[0].invert_yaxis()
    return


def plot_well_timeslices(well_data, wells, date, vrange=(-40, 40),
                         pick_dict=None, fig=None, pick_col=None,
                         alpha=None, plot_noise=False):
    """
    Plot a time slice up and down each specified well

    :param path: Well_data dict from extract_wells
    :param wells: List of well names to plot
    :param date: datetime to plot lines for
    :param vrange: Xlims for all axes
    :param pick_dict: Dictionary {well name: [pick depths, ...]}

    :return:
    """
    if not fig:
        fig, axes = plt.subplots(nrows=1, ncols=len(wells) * 2, sharey=True,
                                 sharex=True, figsize=(len(wells) * 2, 8))
    else:
        axes = fig.axes
    # Initialize counter
    i = 0
    if not pick_col and not alpha:
        cat_cmap = cycle(sns.color_palette('dark'))
        pick_col = next(cat_cmap)
        alpha = 1.
    times = well_data['times']
    for well in wells:
        ax1, ax2 = axes[i:i + 2]
        data = well_data[well]['data']
        depth = well_data[well]['depth']
        noise = well_data[well]['noise']
        down_d, up_d = np.array_split(depth - depth[0], 2)
        if down_d.shape[0] != up_d.shape[0]:
            # prepend last element of down to up if unequal lengths by 1
            up_d = np.insert(up_d, 0, down_d[-1])
        # Remove first
        data = data - data[:, 0, np.newaxis]
        reference_vect = data[:, 0]
        ref_time = times[0]
        # Also reference vector
        down_ref, up_ref = np.array_split(reference_vect, 2)
        # Again account for unequal down and up arrays
        if down_ref.shape[0] != up_ref.shape[0]:
            up_ref = np.insert(up_ref, 0, down_ref[-1])
        up_d_flip = up_d[-1] - up_d
        ax1.plot(down_ref, down_d, color='k', linestyle=':',
                 label=ref_time.date(), lw=1.)
        ax2.plot(up_ref, up_d_flip, color='k', linestyle=':', lw=1.)
        if plot_noise:
            # Fill between noise bounds
            ax1.fill_betweenx(y=down_d, x1=down_ref - noise, x2=down_ref + noise,
                              alpha=0.2, color='k')
            ax2.fill_betweenx(y=up_d_flip, x1=up_ref - noise, x2=up_ref + noise,
                              alpha=0.2, color='k')
        # Get column corresponding to xdata time
        dts = np.abs(times - date)
        time_int = np.argmin(dts)
        # Grab along-fiber vector
        fiber_vect = data[:, time_int]
        # Plot two traces for downgoing and upgoing trace at user-
        # picked time
        down_vect, up_vect = np.array_split(fiber_vect, 2)
        # Again account for unequal down and up arrays
        if down_vect.shape[0] != up_vect.shape[0]:
            up_vect = np.insert(up_vect, 0, down_vect[-1])
        ax1.plot(down_vect, down_d, color=pick_col, label=date.date(),
                 alpha=alpha, linewidth=0.5)
        ax2.plot(up_vect, up_d_flip, color=pick_col, alpha=alpha, linewidth=0.5)
        # If picks provided, plot them
        if pick_dict and well in pick_dict:
            for pick in pick_dict[well]:
                well_length = (chan_map_fsb[well][1] -
                               chan_map_fsb[well][0]) / 2
                if pick[0] < well_length:
                    ax1.fill_between(x=np.array([-500, 500]), y1=pick[0] - 0.5,
                                     y2=pick[0] + 0.5, alpha=0.5, color='gray')
                else:
                    ax2.fill_between(x=np.array([-500, 500]), y1=pick[0] - 0.5,
                                     y2=pick[0] + 0.5, alpha=0.5, color='gray')
        # Legend only at last well
        if i == (len(wells) * 2) - 2 and len(times) < 4:
            ax1.legend(fontsize=16, bbox_to_anchor=(0.15, 0.25),
                       framealpha=1.).set_zorder(103)
        elif i == 0:
            ax1.set_ylabel('Depth [m]', fontsize=18)
        # Formatting
        # Common title for well subplots
        ax1_x = ax1.get_window_extent().x1 / 1000.
        ax2_x = ax2.get_window_extent().x0 / 1000.
        fig.text(x=(ax1_x + ax2_x) / 2, y=0.92, s=well, ha='center',
                 fontsize=22)
        ax2.set_facecolor('lightgray')
        ax1.set_facecolor('lightgray')
        ax1.set_xlim([vrange[0], vrange[1]])
        ax1.margins(y=0)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(5.))
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1.))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(5.))
        ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1.))
        ax1.set_title('Down')
        ax2.set_title('Up')
        i += 2
    label = r'$\mu\varepsilon$'
    fig.text(0.5, 0.04, label, ha='center', fontsize=20)  # Commmon xlabel
    ax1.invert_yaxis()
    return


def plot_DSS(well_data, well='all', derivative=False, colorbar_type='light',
             inset_channels=True, simfip=False,
             date_range=(datetime(2019, 5, 19), datetime(2019, 6, 5)),
             denoise_method=None, vrange=(-60, 60), title=None):
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
    fig = plt.figure(constrained_layout=False, figsize=(12, 12))
    if inset_channels and simfip:
        gs = GridSpec(ncols=12, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:3, 5:-1])
        axes1b = fig.add_subplot(gs[3:6, 5:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[6:9, 5:-1], sharex=axes1)
        axes3 = fig.add_subplot(gs[9:, 5:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, :2])
        axes5 = fig.add_subplot(gs[:, 2:4], sharex=axes4)
        cax = fig.add_subplot(gs[:6, -1])
        df = read_excavation(simfip)
    elif inset_channels:
        gs = GridSpec(ncols=12, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:4, 5:-1])
        axes1b = fig.add_subplot(gs[4:8, 5:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[8:, 5:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, :2])
        axes5 = fig.add_subplot(gs[:, 2:4], sharex=axes4)
        cax = fig.add_subplot(gs[:8, -1])
    # Get just the channels from the well in question
    times = well_data['times']
    data = well_data[well]['data']
    depth = well_data[well]['depth']
    noise = well_data[well]['depth']
    mode = well_data['mode']
    type = well_data['type']
    if date_range:
        indices = np.where((date_range[0] < times) & (times < date_range[1]))
        times = times[indices]
        data = data[:, indices]
        data = np.squeeze(data)
    mpl_times = mdates.date2num(times)
    if mode == 'Absolute':
        data = data - data[:, 0, np.newaxis]
    # Denoise methods are not mature yet
    if denoise_method:
        data = denoise(data, denoise_method)
    if colorbar_type == 'dark':
        cmap = ListedColormap(sns.diverging_palette(
            240, 10, n=21, center='dark').as_hex())
    elif colorbar_type == 'light':
        cmap = ListedColormap(sns.color_palette('RdBu_r', 21).as_hex())
    if derivative == 'time':
        data = np.gradient(data, axis=1)
        label = r'$\mu{}$m $hr^{-1}$'
    elif derivative == 'length':
        data = np.gradient(data, axis=0)
        label = r'$m^{-1}$'
    elif type == 'Strain':
        label = r'$\mu\varepsilon$'
    elif type == 'Brillouin Gain':
        label = r'%'
    elif type == 'Brillouin Frequency':
        label = r'GHz'
    # Split the array in two and plot both separately
    down_data, up_data = np.array_split(data, 2, axis=0)
    down_d, up_d = np.array_split(depth - depth[0], 2)
    if down_d.shape[0] != up_d.shape[0]:
        # prepend last element of down to up if unequal lengths by 1
        up_data = np.insert(up_data, 0, down_data[-1, :], axis=0)
        up_d = np.insert(up_d, 0, down_d[-1])
    im = axes1.imshow(down_data, cmap=cmap, origin='upper',
                      extent=[mpl_times[0], mpl_times[-1],
                              down_d[-1] - down_d[0], 0],
                      aspect='auto', vmin=vrange[0], vmax=vrange[1])
    imb = axes1b.imshow(up_data, cmap=cmap, origin='lower',
                        extent=[mpl_times[0], mpl_times[-1],
                                up_d[-1] - up_d[0], 0],
                        aspect='auto', vmin=vrange[0], vmax=vrange[1])
    date_formatter = mdates.DateFormatter('%b-%d %H')
    # If simfip, plot these data here
    if simfip:
        plot_displacement_components(df, starttime=date_range[0],
                                     endtime=date_range[1], new_axes=axes3,
                                     remove_clamps=False,
                                     rotated=True)
        axes3.set_ylabel(r'Displacement [$\mu$m]', fontsize=16)
        axes3.xaxis_date()
        axes3.xaxis.set_major_formatter(date_formatter)
        plt.setp(axes3.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.setp(axes1.get_xticklabels(), visible=False)
        plt.setp(axes1b.get_xticklabels(), visible=False)
        plt.setp(axes2.get_xticklabels(), visible=False)
    else:
        axes2.xaxis_date()
        axes2.xaxis.set_major_formatter(date_formatter)
        plt.setp(axes2.xaxis.get_majorticklabels(), rotation=30, ha='right')
        axes2.xaxis_date()
        axes2.xaxis.set_major_formatter(date_formatter)
        plt.setp(axes2.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.setp(axes1.get_xticklabels(), visible=False)
        plt.setp(axes1b.get_xticklabels(), visible=False)
    axes1.set_ylabel('Depth [m]', fontsize=16)
    axes1b.set_ylabel('Depth [m]', fontsize=16)
    axes1.set_title('Downgoing')
    axes1b.set_title('Upgoing')
    axes2.set_ylabel(label, fontsize=16)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel(label, fontsize=16)
    if not title:
        fig.suptitle('DSS BFS-{}'.format(well), fontsize=20)
    plt.subplots_adjust(wspace=1., hspace=1.)
    # If plotting 1D channel traces, do this last
    if inset_channels:
        # Plot reference time (first point)
        reference_vect = data[:, 0]
        ref_time = times[0]
        # Also reference vector
        down_ref, up_ref = np.array_split(reference_vect, 2)
        # Again account for unequal down and up arrays
        if down_ref.shape[0] != up_ref.shape[0]:
            up_ref = np.insert(up_ref, 0, down_ref[-1])
        up_d_flip = up_d[-1] - up_d
        axes4.plot(down_ref, down_d, color='k', linestyle=':',
                   label=ref_time.date())
        axes5.plot(up_ref, up_d_flip, color='k', linestyle=':')
        # Fill between noise bounds
        axes4.fill_betweenx(y=down_d, x1=down_ref - noise, x2=down_ref + noise,
                            alpha=0.2, color='k')
        axes5.fill_betweenx(y=up_d_flip, x1=up_ref - noise, x2=up_ref + noise,
                            alpha=0.2, color='k')
        # Grid lines on axes 1
        axes2.grid(which='both', axis='y')
        axes4.grid(which='both', axis='x')
        axes5.grid(which='both', axis='x')
        axes2.set_ylim([vrange[0], vrange[1]])
        axes2.set_facecolor('lightgray')
        axes5.set_facecolor('lightgray')
        axes4.set_facecolor('lightgray')
        axes4.set_xlim([vrange[0], vrange[1]])
        axes4.set_ylim([down_d[-1], down_d[0]])
        axes5.set_ylim([up_d[-1] - up_d[0], 0])
        axes5.yaxis.set_major_locator(ticker.MultipleLocator(5.))
        axes5.yaxis.set_minor_locator(ticker.MultipleLocator(1.))
        axes4.yaxis.set_major_locator(ticker.MultipleLocator(5.))
        axes4.yaxis.set_minor_locator(ticker.MultipleLocator(1.))
        axes4.set_title('Downgoing')
        axes4.set_ylabel('Depth [m]', fontsize=16)
        axes4.set_xlabel(label, fontsize=16)
        axes5.set_xlabel(label, fontsize=16)
        axes5.set_title('Upgoing')

        # Define class for plotting new traces
        class TracePlotter():
            def __init__(self, figure, data, well, depth, cmap, cat_cmap, up_d,
                         down_d):
                self.figure = figure
                self.cmap = cmap
                self.cat_cmap = cat_cmap
                self.data = data
                self.up_d = up_d
                self.down_d = down_d
                self.depth = depth - depth[0]
                self.xlim = self.figure.axes[0].get_xlim()
                self.times = np.linspace(self.xlim[0], self.xlim[1],
                                         data.shape[1])
                self.pick_dict = {well: []}
                self.well = well
                self.cid = self.figure.canvas.mpl_connect('button_press_event',
                                                          self)

            def __call__(self, event):
                global counter
                if event.inaxes not in self.figure.axes[:2]:
                    return
                pick_ax = event.inaxes
                # Did we pick in the upgoing or downgoing fiber?
                if pick_ax == self.figure.axes[0]:
                    upgoing = False
                elif pick_ax == self.figure.axes[1]:
                    upgoing = True
                pick_col = next(self.cat_cmap)
                print('click', event.xdata, event.ydata)
                self.pick_dict[self.well].append((event.ydata, pick_col))
                # Get channel corresponding to ydata (which was modified to
                # units of meters during imshow...?
                # Separate depth vectors
                if upgoing:
                    chan_dist = np.abs(self.depth - (up_d[-1] - event.ydata))
                else:
                    chan_dist = np.abs(self.depth - event.ydata)
                chan = np.argmin(chan_dist)
                # Get column corresponding to xdata time
                dts = np.abs(self.times - event.xdata)
                time_int = np.argmin(dts)
                trace = self.data[chan, :]
                # depth = self.depth[chan]
                depth = event.ydata
                # Grab along-fiber vector
                fiber_vect = self.data[:, time_int]
                # Arrow patches for picks
                trans = pick_ax.get_yaxis_transform()
                arrow = mpatches.FancyArrowPatch((1.05, event.ydata),
                                                 (0.95, event.ydata),
                                                 mutation_scale=20,
                                                 transform=trans,
                                                 facecolor=pick_col,
                                                 clip_on=False,
                                                 zorder=103)
                pick_ax.add_patch(arrow)
                self.figure.axes[2].axvline(x=event.xdata, color=pick_col,
                                            linestyle='--', alpha=0.5)
                if len(self.figure.axes) == 7:
                    self.figure.axes[3].axvline(x=event.xdata, color=pick_col,
                                                linestyle='--', alpha=0.5)
                self.figure.axes[2].plot(self.times, trace, color=pick_col,
                                         label='Depth {:0.2f}'.format(depth))
                # Silly
                self.figure.axes[2].margins(x=0.)
                # Plot two traces for downgoing and upgoing trace at user-
                # picked time
                down_vect, up_vect = np.array_split(fiber_vect, 2)
                # Again account for unequal down and up arrays
                if down_vect.shape[0] != up_vect.shape[0]:
                    up_vect = np.insert(up_vect, 0, down_vect[-1])
                self.figure.axes[-3].plot(down_vect, down_d, color=pick_col,
                                          label=num2date(event.xdata).date())
                self.figure.axes[-2].plot(up_vect, up_d[-1] - up_d,
                                          color=pick_col)
                self.figure.axes[-3].legend(
                    loc=2, fontsize=12, bbox_to_anchor=(-0.95, 1.0),
                    framealpha=1.).set_zorder(103)
                # Swap out fiber length tick labels for depth
                # Plot ydata on axes4/5
                if upgoing:
                    self.figure.axes[-2].fill_between(x=np.array([-500, 500]),
                                                      y1=event.ydata - 0.5,
                                                      y2=event.ydata + 0.5,
                                                      alpha=0.5, color=pick_col)
                else:
                    self.figure.axes[-3].fill_between(x=np.array([-500, 500]),
                                                      y1=event.ydata - 0.5,
                                                      y2=event.ydata + 0.5,
                                                      alpha=0.5, color=pick_col)
                # TODO Need dynamic way of colorbar scaling
                self.figure.axes[2].legend(loc=2, bbox_to_anchor=(-0.2, 1.15),
                                           framealpha=1.).set_zorder(103)
                self.figure.axes[2].yaxis.tick_right()
                self.figure.axes[2].yaxis.set_label_position('right')
                self.figure.canvas.draw()
                counter += 1

        # Make a better cursor for picking channels
        class Cursor(object):
            def __init__(self, axes, fig):
                self.axes = axes
                self.figure = fig
                self.lx1 = axes[0].axhline(axes[0].get_ylim()[0], color='k')
                self.ly1 = axes[0].axvline(axes[0].get_xlim()[0], color='k')
                self.lx1 = axes[1].axhline(axes[1].get_ylim()[0], color='k')
                self.ly1 = axes[1].axvline(axes[1].get_xlim()[0], color='k')

            def mouse_move(self, event):
                if event.inaxes in self.axes:
                    return

                x, y = event.xdata, event.ydata
                # update the line positions
                if event.inaxes == self.axes[0]:
                    self.lx1.set_ydata(y)
                    self.ly1.set_xdata(num2date(x))
                elif event.inaxes == self.axes[1]:
                    self.lx2.set_ydata(y)
                    self.ly2.set_xdata(num2date(x))

                self.figure.canvas.draw()

        # Connect cursor to ax1
        cursor = Cursor([axes1, axes1b], fig)
        fig.canvas.mpl_connect('motion_notify_event', cursor.mouse_move)

        global counter
        counter = 0 # Click counter for trace spacing
        # Set up categorical color palette
        cat_cmap = cycle(sns.color_palette('dark'))
        plotter = TracePlotter(fig, data, well, depth, cmap,
                               cat_cmap, up_d, down_d)
        plt.show()
    return plotter.pick_dict


def madjdabadi_realign(data):
    """
    Spatial realignment based on Modjdabadi et al. 2016
    https://doi.org/10.1016/j.measurement.2015.08.040
    """
    # 'Up' shifted
    next_j = np.append(data[1:, :], data[-1, :]).reshape(data.shape)
    # 'Down' shifted
    prev_j = np.insert(data[:-1, :], 0, data[0, :]).reshape(data.shape)
    compare = np.stack([prev_j, data, next_j], axis=2)
    return np.min(compare, axis=2)


def estimate_noise(data):
    """
    Calculate the average MAD for all channels similar to Madjdabadi 2016,
    but replacing std with MAD.

    :param data: Numpy array of DSS data
    :return:
    """
    # Take MAD of each channel time series, then average
    return np.mean(median_absolute_deviation(data, axis=1))


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

def DSS_spectrum(path, well='all', domain='time'):
    times, data, depth = extract_well(path, well)
    if domain == 'time':
        # Frequency in hours
        freq, psd = welch(data, fs=1., axis=1)
        avg_psd = psd.sum(axis=0)
    elif domain == 'depth':
        # Frequency in feet
        freq, psd = welch(data, fs=0.255, axis=0)
        avg_psd = psd.sum(axis=1)
    else:
        print('Invalid domain string')
    # Plot
    plt.semilogy(freq, avg_psd)
    return