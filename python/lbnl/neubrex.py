#!/usr/bin/python

"""
Functions for reading, processing, plotting Neubrex fiber optic data
"""

import h5py
import os

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from scipy.io import loadmat, savemat
from lxml import etree
from lxml.etree import XMLSyntaxError
from copy import deepcopy
from glob import glob
from obspy import UTCDateTime
from pandas.errors import ParserError
from datetime import datetime, timedelta
from matplotlib.dates import num2date
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import ListedColormap
from scipy.signal import detrend
from itertools import cycle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.dates import DayLocator, HourLocator

# Local imports
from lbnl.boreholes import (parse_surf_boreholes, create_FSB_boreholes,
                            calculate_frac_density, read_frac_cores,
                            depth_to_xyz, distance_to_borehole,
                            read_gallery_distances, read_gallery_excavation)
from lbnl.hydraulic_data import (read_collab_hydro, read_csd_hydro,
                                 plot_csd_hydro, plot_collab_ALL,
                                 read_fsb_hydro, plot_fsb_hydro)

# Assorted parameters
# Custom color palette similar to wellcad convention
frac_cols = {'All fractures': 'black',
             'open/undif. fracture': 'blue',
             'sealed fracture / vein': 'lightblue',
             'foliation / bedding': 'red',
             'induced fracture': 'magenta',
             'sedimentary structures/color changes undif.': 'green',
             'uncertain type': 'orange',
             'lithology change': 'yellow'}

fault_depths = {'D1': (14.34, 19.63), 'D2': (11.04, 16.39), 'D3': (17.98, 20.58),
                'D4': (27.05, 28.44), 'D5': (19.74, 22.66), 'D6': (28.5, 31.4),
                'D7': (22.46, 25.54), 'B2': (41.25, 45.65), 'B1': (34.8, 42.25),
                'B9': (55.7, 55.7), 'B10': (17.75, 21.7)}

resin_depths = {'D3': (2.5, 3.), 'D4': (9., 10.), 'D5': (17., 18.),
                'D6': (12., 14.)}

# Channel mappings
chan_map_injection_fsb = {
    'B1': 98.5, 'B2': 1515.3, 'B3': 888., 'B4': 1062., 'B5': 709., 'B6': 1216.,
    'B7': 1320., 'B8': 428., 'B9': 267., 'B10': 565.}

chan_map_numo = {'NW1': (166., 1251.), 'NW2': (1476., 2600.)}

channel_mapping = {'fsb': chan_map_injection_fsb, 'numo': chan_map_numo}

# Fiber winding mapping
cable_wind = {'numo': 13.59}

#Fiber depth mapping
fiber_depth_mapping = {'numo': {'NW1': 595., 'NW2': 595.}}

# make_date, distance_step, read_data_file are taken from Neubrex python file
def make_date(text, fmt='%m/%d/%Y %H:%M'):
    """
    Returns date-time stamp from a string format.

    :param text: date string
    :param fmt: format string for the stamp
    :return: Python datetime object.
    """
    return datetime.strptime(text, fmt)


def distance_step(dt, ior=1.49):
    """
    Calculates distance step for given index of refraction
    and time step.

    Parameters:
    -----------
    dt : float
        time step in seconds.
    ior : float
        index of refraction.

    Returns:
    ---------
    dt : float
        distance step in meters.
    """
    speed_of_light = 3.0e8  # speed of light [m/s]

    # group speed
    vg = speed_of_light / ior

    # time step
    dz = dt * vg / 2.0

    # return to caller
    return dz


def read_data_file(file_path, mapping=None, wells=None, extract_wells=False):
    """
    Reads and transposes the data set from the deliverables format H5 file.

    :param file_path: file name
    :return: tuple of depth, time stamps, and data
    """
    # date format in file --> stamps
    frmt = '%m/%d/%Y %H:%M:%S.%f'

    # read file
    with h5py.File(file_path, "r") as f:
        # convert data set to numpy arrays
        v = f["data"][()].T
        s = f["stamps"][()]
        z = f["depth"][()]

    # convert to list
    ts = [make_date(t.decode('ascii'), frmt) for t in s]
    fiber_data = {'depth': np.array(z), 'times': np.array(ts), 'data': v}
    if not extract_wells:
        return fiber_data
    chan_map = channel_mapping[mapping]
    fiber_depths = fiber_depth_mapping[mapping]
    fiber_wind = cable_wind[mapping]
    well_data = {}
    for well in wells:
        if well not in chan_map:
            print('{} not in mapping'.format(well))
            continue
        # This accounts for XX% greater fiber depth than TD
        fiber_depth = (fiber_depths[well] /
                       np.cos(np.deg2rad(cable_wind[mapping])))
        depth = fiber_data['depth'].copy()
        data = fiber_data['data'].copy()
        times = fiber_data['times'].copy()
        if mapping == 'fsb':  # Case of one symmetry point on looped wells
            start_chan = np.abs(depth - (chan_map[well] - fiber_depth))
            end_chan = np.abs(depth - (chan_map[well] + fiber_depth))
        elif mapping in ['numo']:  # Non-looped well
            start_chan = np.abs(depth - chan_map[well][0])
            end_chan = np.abs(depth - chan_map[well][1])
        else:
            print('Mapping string {} not recognized'.format(mapping))
            return
        # Find the closest integer channel to meter mapping
        data_tmp = data[np.argmin(start_chan):np.argmin(end_chan), :]
        depth_tmp = depth[np.argmin(start_chan):np.argmin(end_chan)]
        # Account for cable winding
        depth_tmp *= np.cos(np.deg2rad(fiber_wind))
        noise = estimate_noise(data_tmp)
        well_data[well] = {'data': data_tmp, 'depth': depth_tmp,
                           'times': times, 'noise': noise,
                           'type': None}
        return well_data


def estimate_noise(data, method='madjdabadi'):
    """
    Calculate the average MAD for all channels similar to Madjdabadi 2016,
    but replacing std with MAD

    Alternatively, don't take the average and return both the mean and MAD
    as arrays

    :param data: Numpy array of DSS data
    :return:
    """
    if method == 'madjdabadi':
        # Take MAD of each channel time series, then average
        return np.mean(3 * np.std(data, axis=1)), None
        # return np.mean(median_absolute_deviation(data, axis=1)), None
    elif method == 'Krietsch':
        return np.mean(np.percentile(data, q=[10, 90], axis=1), axis=1)
    else:
        print('Invalid method for denoise')
        return

def plot_full_fiber(well_data, dates, xlim, ylim, write_frames=False,
                    frame_interval=timedelta(hours=2)):
    """
    Plot the entire fiber at various times

    :param well_data: Output of read_XTDTS_dir where extract_wells=False
    :param dates: List of datetime.datetime unless write_frames=True,
        In while case the dates will be autogenerated at a given interval
    :param xlim: Xlimits of the figure (along fiber length)
    :param ylim: Ylimits of the figure (temperature range)
    :param write_frames: Boolean whether to write regularly spaced frames,
        to be made into a movie
    :param frame_interval: Time interval between frames (as timedelta)
    """
    if not write_frames:
        fig, axes = plt.subplots()
    cat_cmap = cycle(sns.color_palette('Dark2'))
    if write_frames:
        no_frames = (dates[1] - dates[0]) // frame_interval
        dates = [dates[0] + (frame_interval * i) for i in range(no_frames)]
    for i, date in enumerate(dates):
        if write_frames:
            color = 'firebrick'
            fig, axes = plt.subplots()
        else:
            color = next(cat_cmap)
        axes.plot(
            well_data['depth'],
            well_data['data'][:, np.argmin(np.abs(
                date - well_data['times']))],
            label=date.strftime('%m-%d-%Y %H:%M'), color=color,
            linewidth=1.25)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.set_ylabel('Temperature [C]')
        axes.set_xlabel('Distance along fiber [m]')
        if write_frames:
            fig.text(0.05, 0.90, date, ha="left", va="bottom", fontsize=14,
                     bbox=dict(boxstyle="round",
                               ec='k', fc='white'))
            fig.savefig('frame_{:03d}.png'.format(i), dpi=300)
            plt.close('all')
        else:
            fig.legend()
    if not write_frames:
        plt.show()
    return

def plot_neubrex(well_data, well='all', derivative=False, inset_channels=True,
                 date_range=(datetime(2020, 11, 19), datetime(2020, 11, 23)),
                 vrange=(14, 17), title=None, tv_picks=None, prominence=30.,
                 hydro_data=None, offset_samps=None, mask_depths=None,
                 delta_T=False):
    """
    Plot a colormap of DSS data

    :param path: Path to raw data file
    :param well: Which well to plot
    :param inset_channels: Bool for picking channels to plot in separate axes
    :param date_range: [start date, end date]
    :param window: Window for pandas rolling funcs (only rolling_mean denoise
        at the moment.
    :param vrange: Colorbar range (in measurand unit)
    :param title: Title of plot
    :param tv_picks: Path to excel file with optical televiewer picks
    :param prominence: Prominence (in measurand units) fed to
        scipy.signal.find_peaks
    :param hydro_data: Path to hydraulic data file (just Martin's at collab rn)
    :param offset_samps: Number of time samples to use to compute/remove noise
    :param mask_depths: Depths in borehole to mask in plotting (e.g. for
        the 3D string heating in BFS-B1)
    :param delta_T: Boolean to plot temperature relative to first sample or not
    :return:
    """
    if inset_channels and well != 'D5' and not hydro_data:
        fig = plt.figure(constrained_layout=False, figsize=(14, 14))
        gs = GridSpec(ncols=12, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:4, 7:-1])
        axes1b = fig.add_subplot(gs[4:8, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[8:, 7:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        cax = fig.add_subplot(gs[:8, -1])
    elif inset_channels and hydro_data:
        fig = plt.figure(constrained_layout=False, figsize=(14, 14))
        gs = GridSpec(ncols=14, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:3, 7:-1])
        axes1b = fig.add_subplot(gs[3:6, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[6:9, 7:-1], sharex=axes1)
        hydro_ax = fig.add_subplot(gs[9:, 7:-1], sharex=axes1)
        pres_ax = hydro_ax.twinx()
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        cax = fig.add_subplot(gs[:6, -1])
    elif inset_channels:
        fig = plt.figure(constrained_layout=False, figsize=(14, 14))
        gs = GridSpec(ncols=14, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:4, 7:-1])
        axes1b = fig.add_subplot(gs[4:8, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[8:, 7:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        cax = fig.add_subplot(gs[:6, -1])
    # Get just the channels from the well in question
    times = well_data[well]['times'].copy()
    data = well_data[well]['data'].copy()
    try:
        gain = well_data[well]['gain'].copy()
    except KeyError:
        print('No gain correction')
    depth_vect = well_data[well]['depth']
    if well_data[well]['noise'][1] is None:
        noise = well_data[well]['noise'][0]
    else:
        noise = well_data[well]['noise']
    type = well_data[well]['type']
    if delta_T:
        data = data - data[:, 0, np.newaxis]
    if date_range:
        indices = np.where((date_range[0] < times) & (times < date_range[1]))
        times = times[indices]
        data = np.squeeze(data[:, indices])
        if times.shape[0] == 0:
            print('No data inside time range')
            return
    mpl_times = mdates.date2num(times)
    if offset_samps:
        data = data - data[:, 0:offset_samps, np.newaxis].mean(axis=1)
    cmap = ListedColormap(sns.color_palette('magma', 21).as_hex())
    if derivative:
        data = np.gradient(data, axis=1)
        label = r'$\Delta^O$C'
    elif type == None:
        label = r'$^O$C'
    if well in ['3339', '3359']:
        # Split the array in two and plot both separately
        down_data = data
        up_data = data[::-1, :]
        down_d = depth_vect
        up_d = depth_vect
    else:
        # Split the array in two and plot both separately
        down_data, up_data = np.array_split(data, 2, axis=0)
        down_d, up_d = np.array_split(depth_vect - depth_vect[0], 2)
    if down_d.shape[0] != up_d.shape[0]:
        # prepend last element of down to up if unequal lengths by 1
        up_data = np.insert(up_data, 0, down_data[-1, :], axis=0)
        up_d = np.insert(up_d, 0, down_d[-1])
    if mask_depths:
        for i, dr in enumerate(mask_depths):
            if i == 0:
                mask_inds = ((down_d > dr[0]) & (down_d <= dr[1]))
            else:
                mask_inds += ((down_d > dr[0]) & (down_d <= dr[1]))
        mask_arr = np.stack([mask_inds] * down_data.shape[1], -1)
        down_data = np.ma.MaskedArray(down_data, mask_arr)
        up_data = np.ma.MaskedArray(up_data, mask_arr[::-1, :])
    # Run the integration for D1/2
    im = axes1.imshow(down_data, cmap=cmap, origin='upper',
                      extent=[mpl_times[0], mpl_times[-1],
                              down_d[-1] - down_d[0], 0],
                      aspect='auto', vmin=vrange[0], vmax=vrange[1])
    imb = axes1b.imshow(up_data, cmap=cmap, origin='lower',
                        extent=[mpl_times[0], mpl_times[-1],
                                up_d[-1] - up_d[0], 0],
                        aspect='auto', vmin=vrange[0], vmax=vrange[1])
    # Plot fault bounds on images
    if well in fault_depths:
        try:
            axes1.axhline(fault_depths[well][0], linestyle='--', linewidth=1.,
                          color='k')
            axes1.axhline(fault_depths[well][1], linestyle='--', linewidth=1.,
                          color='k', label='Main Fault Zone')
            axes1b.axhline(fault_depths[well][0], linestyle='--', linewidth=1.,
                          color='k')
            axes1b.axhline(fault_depths[well][1], linestyle='--', linewidth=1.,
                          color='k')
            axes1.legend(loc=2, fontsize=12, bbox_to_anchor=(0.65, 1.3),
                         framealpha=1.).set_zorder(110)
        except IndexError as e:
            print(e)
    date_formatter = mdates.DateFormatter('%m-%d %H:%M')
    # If simfip, plot these data here
    if hydro_data:
        try:
            df = read_collab_hydro(hydro_data)
        except ParserError:
            df = read_csd_hydro(hydro_data)
        df = df[date_range[0]:date_range[1]]
        hydro_ax.plot(df['Flow'], color='steelblue',
                      label='Flow')
        pres_ax.plot(df['Pressure'], color='red', label='Pressure')
        hydro_ax.margins(x=0., y=0.)
        pres_ax.margins(x=0., y=0.)
        hydro_ax.set_ylim(bottom=0.)
        pres_ax.set_ylim(bottom=0.)
        hydro_ax.set_ylabel('L/min')
        pres_ax.set_ylabel('MPa')
        hydro_ax.yaxis.label.set_color('steelblue')
        hydro_ax.tick_params(axis='y', colors='steelblue')
        pres_ax.yaxis.label.set_color('red')
        pres_ax.tick_params(axis='y', colors='red')
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
    if hydro_data:
        hydro_ax.xaxis_date()
        hydro_ax.xaxis.set_major_formatter(date_formatter)
        plt.setp(hydro_ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        # plt.setp(hydro_ax.get_xticklabels(), visible=False)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel(label, fontsize=16)
    if not title:
        if well.startswith('D') and len(well) == 2:
            exp = 'BCS'
        elif well.startswith('B'):
            exp = 'BFS'
        else:
            exp = 'Collab'
        fig.suptitle('DTS {}-{}'.format(exp, well), fontsize=20)
    plt.subplots_adjust(wspace=1., hspace=1.)
    # If plotting 1D channel traces, do this last
    if inset_channels:
        # Plot reference time (first point)
        reference_vect = data[:, 0]
        ref_time = times[0]
        if well in ['3339', '3359']:
            # Also reference vector
            down_ref = reference_vect
            up_ref = reference_vect[::-1]
        else:
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
        axes4.fill_betweenx(y=down_d, x1=down_ref - noise,
                            x2=down_ref + noise, alpha=0.2, color='k')
        axes5.fill_betweenx(y=up_d_flip, x1=up_ref - noise,
                            x2=up_ref + noise, alpha=0.2, color='k')
        # Plot fracture density too TODO Enable other logs here too
        if tv_picks:
            pass
            # try:
            #     frac_dict = calculate_frac_density(
            #         tv_picks, create_FSB_boreholes())
            # except KeyError:
            #     # Try core fracture counts instead
            #     frac_dict = read_frac_cores(tv_picks, well)
            # for frac_type, dens in frac_dict.items():
            #     if not frac_type.startswith('sed'):
            #         log_ax.plot(dens[:, 1], dens[:, 0],
            #                     color=frac_cols[frac_type],
            #                     label=frac_type)
            # log_ax.legend(
            #     loc=2, fontsize=12, bbox_to_anchor=(-1.2, 1.13),
            #     framealpha=1.).set_zorder(110)
        # Grid lines on axes 1
        axes2.grid(which='both', axis='y')
        axes4.grid(which='both', axis='x')
        axes5.grid(which='both', axis='x')
        axes2.set_ylim([vrange[0], vrange[1]])
        axes2.set_facecolor('lightgray')
        axes5.set_facecolor('lightgray')
        axes4.set_facecolor('lightgray')
        log_ax.set_facecolor('lightgray')
        # log_ax.set_title('Televiewer picks')
        log_ax.set_xlabel('Count / m', fontsize=12)
        axes4.set_xlim([vrange[0], vrange[1]])
        axes4.set_ylim([down_d[-1], down_d[0]])
        axes5.set_ylim([up_d[-1] - up_d[0], 0])
        if well in ['3339', '3359']:
            axes5.yaxis.set_major_locator(ticker.MultipleLocator(500.))
            axes5.yaxis.set_minor_locator(ticker.MultipleLocator(100.))
            axes4.yaxis.set_major_locator(ticker.MultipleLocator(500.))
            axes4.yaxis.set_minor_locator(ticker.MultipleLocator(100.))
        else:
            axes5.yaxis.set_major_locator(ticker.MultipleLocator(5.))
            axes5.yaxis.set_minor_locator(ticker.MultipleLocator(1.))
            axes4.yaxis.set_major_locator(ticker.MultipleLocator(5.))
            axes4.yaxis.set_minor_locator(ticker.MultipleLocator(1.))
        axes4.set_title('Downgoing')
        log_ax.set_ylabel('Depth [m]', fontsize=16)
        axes4.set_xlabel(label, fontsize=16)
        axes5.set_xlabel(label, fontsize=16)
        axes5.set_title('Upgoing')

        # Define class for plotting new traces
        class TracePlotter():
            def __init__(self, figure, data, times, well, depth, cmap, cat_cmap,
                         up_d, down_d, noise, prominence):
                self.figure = figure
                self.cmap = cmap
                self.cat_cmap = cat_cmap
                self.prominence = prominence
                self.noise = noise
                self.data = data
                self.up_d = up_d
                self.down_d = down_d
                self.depth = depth - depth[0]
                self.xlim = self.figure.axes[0].get_xlim()
                self.times = times
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
                else:
                    return
                print('click', mdates.num2date(event.xdata), event.ydata)
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
                # Also stack a range of channels in case we want that?
                stack = np.sum(self.data[chan-10:chan+10, :], axis=0)
                stack *= (np.max(np.abs(trace)) / np.max(np.abs(stack)))
                # depth = self.depth[chan]
                depth = event.ydata
                # Grab along-fiber vector
                fiber_vect = self.data[:, time_int]
                self.figure.axes[2].axvline(x=event.xdata, color='k',
                                            linestyle='--', alpha=0.5)
                # Silly
                self.figure.axes[2].margins(x=0.)
                # Plot two traces for downgoing and upgoing trace at user-
                # picked time
                if well in ['3339', '3359']:
                    down_vect = fiber_vect
                    up_vect = fiber_vect[::-1]
                else:
                    down_vect, up_vect = np.array_split(fiber_vect, 2)
                # Adjustment flag for pick plotting on upgoing vector
                pick_adjust = 0
                # Again account for unequal down and up arrays
                if down_vect.shape[0] != up_vect.shape[0]:
                    up_vect = np.insert(up_vect, 0, down_vect[-1])
                    pick_adjust = 1
                self.figure.axes[-4].plot(down_vect, down_d, color='b',
                                          label=num2date(event.xdata).date())
                self.figure.axes[-3].plot(up_vect, up_d[-1] - up_d,
                                          color='b')
                if well in fault_depths:
                    try:
                        for i in range(-4, -1):
                            self.figure.axes[i].axhline(fault_depths[well][0],
                                                        linestyle='--',
                                                        linewidth=1., color='k')
                            self.figure.axes[i].axhline(fault_depths[well][1],
                                                        linestyle='--',
                                                        linewidth=1., color='k')
                            # Fill between resin plug
                            self.figure.axes[i].fill_between(
                                x=np.array([-500, 500]), y1=resin_depths[well][0],
                                y2=resin_depths[well][1], hatch='/',
                                alpha=0.5, color='bisque')
                            self.figure.axes[i].fill_between(
                                x=np.array([-500, 500]), y1=resin_depths[well][0],
                                y2=resin_depths[well][1], hatch='/',
                                alpha=0.5, color='bisque', label='Resin plug')
                    except (IndexError, KeyError) as e:
                        print(e)
                self.figure.axes[-4].legend(
                    loc=2, fontsize=12, bbox_to_anchor=(0.5, 1.13),
                    framealpha=1.).set_zorder(110)
                pick_col = next(self.cat_cmap)
                # Populate pick_dict
                self.pick_dict[self.well].append((event.ydata, pick_col))
                # Plot ydata on axes4/5 if manual
                if upgoing:
                    self.figure.axes[-3].fill_between(
                        x=np.array([-500, 500]), y1=event.ydata - 0.5,
                                   y2=event.ydata + 0.5,
                                   alpha=0.5, color=pick_col)
                else:
                    self.figure.axes[-4].fill_between(
                        x=np.array([-500, 500]), y1=event.ydata - 0.5,
                        y2=event.ydata + 0.5,
                        alpha=0.5, color=pick_col)
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
                self.figure.axes[2].plot(
                    self.times, trace, color=pick_col,
                    label='Depth {:0.2f}'.format(depth), alpha=0.7)
                self.figure.axes[2].legend(loc=2, bbox_to_anchor=(-0.2, 1.15),
                                           framealpha=1.).set_zorder(103)
                self.figure.axes[2].yaxis.tick_right()
                self.figure.axes[2].yaxis.set_label_position('right')
                # Ensure xlims don't modify from original date range
                self.figure.axes[0].set_xlim(self.xlim)
                self.figure.canvas.draw()
                self.figure.axes[-1].set_zorder(
                    self.figure.axes[-2].get_zorder() - 1)
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
        plotter = TracePlotter(fig, data, mpl_times, well, depth_vect, cmap,
                               cat_cmap, up_d, down_d,
                               noise=well_data[well]['noise'],
                               prominence=prominence)
        plt.show()
    return plotter.pick_dict