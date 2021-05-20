#!/usr/bin/python

import h5py
import os

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.linalg as linalg
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from glob import glob
from itertools import cycle
from datetime import datetime, timedelta
from scipy.io import loadmat
from scipy.signal import detrend
from scipy.integrate import trapz
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial.transform import Rotation as R
from pandas.errors import ParserError
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.dates import num2date

from lbnl.boreholes import (parse_surf_boreholes, create_FSB_boreholes,
                            calculate_frac_density, read_frac_cores,
                            depth_to_xyz, distance_to_borehole,
                            read_gallery_distances, read_gallery_excavation)

from lbnl.hydraulic_data import (read_collab_hydro, read_csd_hydro,
                                 plot_csd_hydro, plot_collab_ALL,
                                 read_fsb_hydro, plot_fsb_hydro)

resin_depths = {'D3': (2.5, 3.), 'D4': (9., 10.), 'D5': (17., 18.),
                'D6': (12., 14.)}

fault_depths = {'D1': (14.34, 19.63), 'D2': (11.04, 16.39), 'D3': (17.98, 20.58),
                'D4': (27.05, 28.44), 'D5': (19.74, 22.66), 'D6': (28.5, 31.4),
                'D7': (22.46, 25.54), 'B2': (41.25, 45.65), 'B1': (34.8, 42.25),
                'B9': (55.7, 55.7), 'B10': (17.75, 21.7)}

# Custom color palette similar to wellcad convention
frac_cols = {'All fractures': 'black',
             'open/undif. fracture': 'blue',
             'sealed fracture / vein': 'lightblue',
             'foliation / bedding': 'red',
             'induced fracture': 'magenta',
             'sedimentary structures/color changes undif.': 'green',
             'uncertain type': 'orange',
             'lithology change': 'yellow'}


def datenum_to_datetime(datenums):
    # Helper to correctly convert matlab datenum to python datetime
    # SO source:
    # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    return [datetime.fromordinal(int(d)) +
            timedelta(days=d % 1) - timedelta(days=366)
            for d in datenums]


def read_struct(f, depth=41):
    # Return the parts of the struct we actually want
    try:
        struct = loadmat(f, struct_as_record=False,
                         squeeze_me=True)
        data = struct['OT_strain'].data[depth, :]
        # Convert nano to microstrain
        data /= 1000.
        try:
            datenums = struct['OT_strain'].dn
        except AttributeError as e:
            datenums = struct['OT_strain'].datenum
    except NotImplementedError:
        # In case of Matlab 7.3 format
        with h5py.File(f, 'r') as f:
            # print(f['dasRate_CleanMedian']['date'][()])
            try:
                data = f['DAS_1min_rs']['strain'][()]
                datenums = np.concatenate(f['DAS_1min_rs']['dates'][()])
                depths = np.concatenate(f['DAS_1min_rs']['depths'][()])
                depth = np.argmin(np.abs(depths - depth))
            except KeyError as e:
                print(e)
                print(f['strain_OT']['strain'].shape)
                data = f['strain_OT']['strain'][()].T
                datenums = np.concatenate(f['strain_OT']['datenum'][()])
            data = data[depth, :] / 1000.
    try:
        time = datenum_to_datetime(datenums)
    except:
        time = None
    return time, data


def read_struct_FSB(f):
    # Return the parts of the struct we actually want
    # In case of Matlab 7.3 format
    bh = f.split('/')[-1].split('_')[0]
    sname = 'strain{}'.format(bh)
    if sname.endswith('b10'):
        sname = 'strainb10_l1'
    with h5py.File(f, 'r') as f:
        try:
            # DAS data in nanostrain
            data = f[sname]['strain'][()].T * 1e-3
        except KeyError:
            sname = 'strainb10_l2'
            data = f[sname]['strain'][()].T * 1e-3
        datenums = np.concatenate(f[sname]['datenum'][()])
        depths = np.concatenate(f[sname]['distance'][()])
    try:
        time = datenum_to_datetime(datenums)
    except:
        time = None
    return bh, time, depths, data


def extract_wells_FSB(das_dir):
    """Read in all DAS data to well dict"""
    das_files = glob('{}/*.mat'.format(das_dir))
    well_dict = {}
    for df in das_files:
        well, time, depths, data = read_struct_FSB(df)
        well_dict[well.upper()] = {'data': data, 'depth': depths,
                                   'times': np.array(time)}
    return well_dict


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


def denoise(data, method='detrend', depth=None, times=None, window='2h'):
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
    elif method == 'rolling_mean':
        data = rolling_stats(data, times, depth, window, stat='mean')
    elif method == 'rolling_median':
        data = rolling_stats(data, times, depth, window, stat='median')
    return data


def rolling_stats(data, times, depth, window='2h', stat='mean'):
    """
    Run a rolling mean on a data matrix with pandas rolling framework

    :param data: values from DSS reading functions
    :param times: Time array (will be used as index)
    :param depth: Depth (column indices)
    :param window: Time window to use in rolling calcs, default 2h
    :param stat: 'mean' or 'median'

    :return:
    """

    df = pd.DataFrame(data=data.T, index=times, columns=depth)
    df = df.sort_index()
    if stat == 'mean':
        roll = df.rolling(window, min_periods=1).mean()
    elif stat == 'median':
        roll = df.rolling(window, min_periods=1).median()
    elif stat == 'std':
        roll = df.rolling(window, min_periods=1).std()
    else:
        print('{} is not a supported statistic'.format(stat))
        return None
    return roll.values.T


def extract_channel_timeseries(well_data, well, depth, direction='down',
                               window='20T'):
    """
    Return a time series of the selected well and depth

    :param well_data: Dict from extract_wells
    :param well: String, wellname
    :param depth: Depth to channel
    :param direction: 'up' or 'down', defaults to 'down'
    :return: times, strains, both arrays
    """
    well_d = well_data[well]
    depths = well_d['depth'] - well_d['depth'][0]
    data = well_d['data']
    times = well_d['times']
    data_median = rolling_stats(data, times, depths, window, stat='median')
    data_std = rolling_stats(data, times, depths, window, stat='std')
    if direction == 'up':
        down_d, up_d = np.array_split(depths, 2)
        down_data, up_data = np.array_split(data, 2)
        down_median, up_median = np.array_split(data_median, 2)
        down_std, up_std = np.array_split(data_std, 2)
        if down_d.shape[0] != up_d.shape[0]:
            # prepend last element of down to up if unequal lengths by 1
            up_d = np.insert(up_d, 0, down_d[-1])
            up_data = np.insert(up_data, 0, down_data[-1, :], axis=0)
            up_median = np.insert(up_median, 0, down_median[-1, :], axis=0)
            up_std = np.insert(up_std, 0, down_std[-1, :], axis=0)
        depths = np.abs(up_d - up_d[-1])
        data = up_data
        data_median = up_median
        data_std = up_std
    # Find closest channel
    chan = np.argmin(np.abs(depth - depths))
    strains = data[chan, :]
    strain_median = data_median[chan, :]
    strain_std = data_std[chan, :]
    return times, strains, strain_median, strain_std


def integrate_depth_interval(well_data, depths, well, leg, dates=None):
    """
    Return timeseries of channels integrated over a depth range

    :param data: data array for well loop
    :param depths: [shallow, deep] range to integrate over
    :param well: Well name
    :param leg: Down or up
    :return:
    """
    data = well_data[well]['data'].copy()
    depth = well_data[well]['depth'].copy()
    times = well_data[well]['times']
    depth -= depth[0].copy()
    if leg == 'down':
        chans = (np.argmin(np.abs(depth - depths[0])),
                 np.argmin(np.abs(depth - depths[1])))
    elif leg == 'up':
        chans = (np.argmin(np.abs(depth - (depth[-1] - depths[1]))),
                 np.argmin(np.abs(depth - (depth[-1] - depths[0]))))
    else:
        print('Only up or down leg, hoss')
        return
    if not dates:
        int_data = data[chans[0]:chans[1] + 1, :]
        # Relative to first sample
        int_data = int_data - int_data[:, 0]
        integral = trapz(int_data, axis=0)
        integral = np.squeeze(integral) * (depth[1] - depth[0])
    else:
        d_inds = np.where((times >= dates[0]) &
                          (times < dates[1]))
        times = times[d_inds]
        int_data = np.squeeze(data[chans[0]:chans[1] + 1, d_inds])
        # Relative to first sample
        int_data = int_data - int_data[:, 0, np.newaxis]
        integral = trapz(int_data, axis=0)
        # Squeeze and scale to channel spacing
        integral = np.squeeze(integral) * (depth[1] - depth[0])
    return integral, times # units are displacement

## Plotting funcs ##

def plot_DAS(well_data, well='all', derivative=False, inset_channels=True,
             date_range=(datetime(2020, 11, 19), datetime(2020, 11, 23)),
             denoise_method=None, window='2h', vrange=(14, 17), title=None,
             tv_picks=None, prominence=30., pot_data=None, hydro_data=None,
             offset_samps=None, filter_params=None):
    """
    Plot a colormap of DSS data

    :param path: Path to raw data file
    :param well: Which well to plot
    :param inset_channels: Bool for picking channels to plot in separate axes
    :param date_range: [start date, end date]
    :param denoise_method: String stipulating the method in denoise() to use
    :param window: Window for pandas rolling funcs (only rolling_mean denoise
        at the moment.
    :param vrange: Colorbar range (in measurand unit)
    :param title: Title of plot
    :param tv_picks: Path to excel file with optical televiewer picks
    :param prominence: Prominence (in measurand units) fed to
        scipy.signal.find_peaks
    :param pot_data: Path to potentiometer data file
    :param hydro_data: Path to hydraulic data file (just Martin's at collab rn)
    :param offset_samps: Number of time samples to use to compute/remove noise
    :param filter_params: Nested dict of various bandstop parameters

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
    # if well_data[well]['noise'][1] is None:
    #     noise = well_data[well]['noise'][0]
    # else:
    #     noise = well_data[well]['noise']
    # type = well_data[well]['type']
    if date_range:
        indices = np.where((date_range[0] < times) & (times < date_range[1]))
        times = times[indices]
        data = np.squeeze(data[:, indices])
    mpl_times = mdates.date2num(times)
    # Denoise methods are not mature yet
    if denoise_method:
        data = denoise(data, denoise_method, times=times, depth=depth_vect,
                       window=window)
    if filter_params:
        for key, f in filter_params.items():
            data = filter(data, freqmin=f['freqmin'],
                          freqmax=f['freqmax'],
                          df=1 / (times[1] - times[0]).seconds)
    if offset_samps:
        data = data - data[:, 0:offset_samps, np.newaxis].mean(axis=1)
    cmap = ListedColormap(sns.color_palette('RdBu_r', 21).as_hex())
    label = r'$\mu\epsilon$'
    # Split the array in two and plot both separately
    down_data, up_data = np.array_split(data, 2, axis=0)
    down_d, up_d = np.array_split(depth_vect - depth_vect[0], 2)
    if down_d.shape[0] != up_d.shape[0]:
        # prepend last element of down to up if unequal lengths by 1
        up_data = np.insert(up_data, 0, down_data[-1, :], axis=0)
        up_d = np.insert(up_d, 0, down_d[-1])
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
        if well.startswith('D'):
            exp = 'BCS'
        elif well.startswith('B'):
            exp = 'BFS'
        else:
            exp = 'Collab'
        fig.suptitle('DAS {}-{}'.format(exp, well), fontsize=20)
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
        # axes4.fill_betweenx(y=down_d, x1=down_ref - noise,
        #                     x2=down_ref + noise, alpha=0.2, color='k')
        # axes5.fill_betweenx(y=up_d_flip, x1=up_ref - noise,
        #                     x2=up_ref + noise, alpha=0.2, color='k')
        # Plot fracture density too TODO Enable other logs here too
        if tv_picks:
            try:
                frac_dict = calculate_frac_density(
                    tv_picks, create_FSB_boreholes())
            except KeyError:
                # Try core fracture counts instead
                frac_dict = read_frac_cores(tv_picks, well)
            for frac_type, dens in frac_dict.items():
                if not frac_type.startswith('sed'):
                    log_ax.plot(dens[:, 1], dens[:, 0],
                                color=frac_cols[frac_type],
                                label=frac_type)
            log_ax.legend(
                loc=2, fontsize=12, bbox_to_anchor=(-1.2, 1.13),
                framealpha=1.).set_zorder(110)
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
                         up_d, down_d, prominence):
                self.figure = figure
                self.cmap = cmap
                self.cat_cmap = cat_cmap
                self.prominence = prominence
                # self.noise = noise
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
                if pot_data:
                    self.figure.axes[4].axvline(x=event.xdata, color='k',
                                                linestyle='--', alpha=0.5)
                # Silly
                self.figure.axes[2].margins(x=0.)
                # Plot two traces for downgoing and upgoing trace at user-
                # picked time
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
                    for i in range(-4, -1):
                        self.figure.axes[i].axhline(fault_depths[well][0],
                                                    linestyle='--',
                                                    linewidth=1., color='k')
                        self.figure.axes[i].axhline(fault_depths[well][1],
                                                    linestyle='--',
                                                    linewidth=1., color='k')
                    try:
                        for i in range(-4, -1):
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
                               # noise=well_data[well]['noise'],
                               prominence=prominence)
        plt.show()
    return plotter.pick_dict


def minute_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int(((end_date - start_date).seconds) / 60.) + 1):
        yield start_date + timedelta(seconds=n * 60)


def martin_plot_fsb(well_data, date_range, autocad_path,
                     vrange, hydro_path, outdir, title):
    """
    Make a series of frames to animate into movie of DAS strain a la
    Martin's plots for Collab

    :param well_data: Output of extract_wells
    :param date_range: Two tuple of start and end date
    :param autocad_path: Path to directory of autocad gallery files
    :param vrange: Two tuple of low and high end of strain colorbar
    :param strike: Strike of xsection plot
    :param hydro_path: Path to hydraulic data file
    :param outdir: Output directory for plots

    :return:
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for i, date in enumerate(minute_generator(date_range[0], date_range[1])):
        fig = martin_plot_frame(
            well_data, time=date, vrange=vrange,
            autocad_path=autocad_path,
            hydro_path=hydro_path, title=title)
        fig.savefig('{}/{:04d}.png'.format(outdir, i), dpi=300)
        plt.close('all')
    return


def martin_plot_frame(well_data, time, vrange=(-100, 100),
                      autocad_path=None, strike=120., hydro_path=None,
                      origin=np.array([2579332., 1247600., 514.]),
                      title=None):
    """
    Plot single frame of wells colored by strain. Take map and cross
    section from Vero and my presentation to MT partners

    :param well_data: Output from extract_wells
    :param time: Time plot signals for

    :return:
    """
    fig = plt.figure(figsize=(15, 8))
    spec = GridSpec(ncols=8, nrows=8, figure=fig)
    ax3d = fig.add_subplot(spec[:6, :4], projection='3d')
    ax_x = fig.add_subplot(spec[:7, 4:])
    ax_hydro = fig.add_subplot(spec[7:, :])
    if title:
        fig.suptitle(title, fontsize=20)
    # Cross section plane (strike 320)
    r = np.deg2rad(360 - strike)
    normal = np.array([-np.sin(r), -np.cos(r), 0.])
    normal /= linalg.norm(normal)
    new_strk = np.array([np.sin(r), -np.cos(r), 0.])
    new_strk /= linalg.norm(new_strk)
    change_b_mat = np.array([new_strk, [0, 0, 1], normal])
    for afile in glob('{}/*.csv'.format(autocad_path)):
        df_cad = pd.read_csv(afile)
        lines = df_cad.loc[df_cad['Name'] == 'Line']
        arcs = df_cad.loc[df_cad['Name'] == 'Arc']
        for i, line in lines.iterrows():
            xs = np.array([line['Start X'], line['End X']])
            ys = np.array([line['Start Y'], line['End Y']])
            zs = np.array([line['Start Z'], line['End Z']])
            # Proj
            pts = np.column_stack([xs, ys, zs])
            proj_pts = np.dot(pts - origin, normal)[:, None] * normal
            proj_pts = pts - origin - proj_pts
            proj_pts = np.matmul(change_b_mat, proj_pts.T)
            ax3d.plot(xs, ys, zs, color='lightgray', zorder=210,
                      linewidth=0.5)
            ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='lightgray',
                      zorder=110, alpha=0.5, linewidth=0.5)
        for i, arc in arcs.iterrows():
            # Stolen math from Melchior
            if not np.isnan(arc['Extrusion Direction X']):
                rotaxang = [arc['Extrusion Direction X'],
                            arc['Extrusion Direction Y'],
                            arc['Extrusion Direction Z'],
                            arc['Total Angle']]
                rad = np.linspace(arc['Start Angle'], arc['Start Angle'] +
                                  arc['Total Angle'])
                dx = np.sin(np.deg2rad(rad)) * arc['Radius']
                dy = np.cos(np.deg2rad(rad)) * arc['Radius']
                dz = np.zeros(dx.shape[0])
                phi1 = -np.arctan2(
                    linalg.norm(
                        np.cross(np.array([rotaxang[0], rotaxang[1], rotaxang[2]]),
                        np.array([0, 0, 1]))),
                    np.dot(np.array([rotaxang[0], rotaxang[1], rotaxang[2]]),
                           np.array([0, 0, 1])))
                DX = dx * np.cos(phi1) + dz * np.sin(phi1)
                DY = dy
                DZ = dz * np.cos(phi1) - dx * np.sin(phi1)
                # ax.plot(DX, DY, DZ, color='r')
                phi2 = np.arctan(rotaxang[1] / rotaxang[0])
                fdx = (DX * np.cos(phi2)) - (DY * np.sin(phi2))
                fdy = (DX * np.sin(phi2)) + (DY * np.cos(phi2))
                fdz = DZ
                x = fdx + arc['Center X']
                y = fdy + arc['Center Y']
                z = fdz + arc['Center Z']
                # projected pts
                pts = np.column_stack([x, y, z])
                proj_pts = np.dot(pts - origin, normal)[:, None] * normal
                proj_pts = pts - origin - proj_pts
                proj_pts = np.matmul(change_b_mat, proj_pts.T)
                ax3d.plot(x, y, z, color='lightgray', zorder=210,
                          linewidth=0.5)
                ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='lightgray',
                          zorder=110, alpha=0.5, linewidth=0.5)
            elif not np.isnan(arc['Start X']):
                v1 = -1. * np.array([arc['Center X'] - arc['Start X'],
                                     arc['Center Y'] - arc['Start Y'],
                                     arc['Center Z'] - arc['Start Z']])
                v2 = -1. * np.array([arc['Center X'] - arc['End X'],
                                     arc['Center Y'] - arc['End Y'],
                                     arc['Center Z'] - arc['End Z']])
                rad = np.linspace(0, np.deg2rad(arc['Total Angle']), 50)
                # get rotation vector (norm is rotation angle)
                rotvec = np.cross(v2, v1)
                rotvec /= linalg.norm(rotvec)
                rotvec = rotvec[:, np.newaxis] * rad[np.newaxis, :]
                Rs = R.from_rotvec(rotvec.T)
                pt = np.matmul(v1, Rs.as_matrix())
                # Projected pts
                x = arc['Center X'] + pt[:, 0]
                y = arc['Center Y'] + pt[:, 1]
                z = arc['Center Z'] + pt[:, 2]
                pts = np.column_stack([x, y, z])
                proj_pts = np.dot(pts - origin, normal)[:, None] * normal
                proj_pts = pts - origin - proj_pts
                proj_pts = np.matmul(change_b_mat, proj_pts.T)
                ax3d.plot(x, y, z, color='lightgray', zorder=210,
                          linewidth=0.5)
                ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='lightgray',
                          zorder=210, alpha=0.5, linewidth=0.5)
    well_dict = create_FSB_boreholes()
    # cmap = ListedColormap(sns.color_palette('icefire', 21).as_hex())
    cmap = ListedColormap(sns.color_palette('coolwarm', 21).as_hex())
    for well, w_dict in well_data.items():
        pts = []
        if well == 'B4':
            continue
        for feature_dep in w_dict['depth']:
            feature_dep -= w_dict['depth'][0]
            pts.append(depth_to_xyz(well_dict, well, feature_dep))
        strains = w_dict['data'][:, np.argmin(np.abs(time - w_dict['times']))]
        # Project well points and fault intersection points onto cross section
        pts = np.array(pts)
        proj_pts = np.dot(pts - origin, normal)[:, None] * normal
        proj_pts = pts - origin - proj_pts
        proj_pts = np.matmul(change_b_mat, proj_pts.T)
        proj_pts = proj_pts[:2, :]
        proj_pts = proj_pts.T.reshape([-1, 1, 2])
        ax3d.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color='darkgray',
                     linewidth=1.5, s=15., zorder=110)
        pts = pts.reshape([-1, 1, 3])
        try:
            fault_pts = [depth_to_xyz(well_dict, well, fault_depths[well][i])
                         for i in (0, 1)]
            fault_pts = np.array(fault_pts)
            p_fault_pts = np.dot(fault_pts - origin, normal)[:, None] * normal
            p_fault_pts = fault_pts - origin - p_fault_pts
            p_fault_pts = np.matmul(change_b_mat, p_fault_pts.T)
            p_fault_pts = p_fault_pts[:2, :].T
            # Plot fault intersection points
            ax_x.scatter(p_fault_pts[:, 0], p_fault_pts[:, 1], c='purple',
                         marker='x', s=15., zorder=105)
        except KeyError as e:
            # Borehole doesn't intersect fault
            pass
        # Make segments
        proj_seggies = np.concatenate([proj_pts[:-1], proj_pts[1:]], axis=1)
        seggies = np.concatenate([pts[:-1], pts[1:]], axis=1)
        col_norm = plt.Normalize(vrange[0], vrange[1])
        lc = Line3DCollection(seggies, cmap=cmap, norm=col_norm)
        lc_proj = LineCollection(proj_seggies, cmap=cmap, norm=col_norm)
        # Set the values used for colormapping
        lc.set_array(strains)
        lc.set_linewidth(3.)
        lc_proj.set_array(strains)
        lc_proj.set_linewidth(3.)
        line = ax3d.add_collection3d(lc)
        line_x = ax_x.add_collection(lc_proj)
    fig.colorbar(line_x, ax=ax3d, label=r'$\mu\epsilon$')
    # Formatting
    ax3d.set_xlim([2579310, 2579355])
    ax3d.set_ylim([1247555, 1247600])
    ax3d.set_zlim([485, 530])
    # ax3d.view_init(elev=30., azim=-112)
    ax3d.view_init(elev=75, azim=-120.)
    ax3d.margins(0.)
    ax3d.set_xticks([])
    ax3d.set_xticklabels([])
    ax3d.set_yticks([])
    ax3d.set_yticklabels([])
    ax3d.set_zticks([])
    ax3d.set_zticklabels([])
    # Cross section
    ax_x.set_xlim([-30, 5])
    ax_x.axis('equal')
    ax_x.spines['top'].set_visible(False)
    ax_x.spines['bottom'].set_visible(False)
    ax_x.spines['left'].set_visible(False)
    ax_x.yaxis.set_ticks_position('right')
    ax_x.tick_params(direction='in', bottom=False, labelbottom=False)
    ax_x.set_yticks([-30, -20, -10, 0])
    ax_x.set_yticklabels(['30', '20', '10', '0'])
    ax_x.set_ylabel('                                      Meters', labelpad=15)
    ax_x.yaxis.set_label_position("right")
    ax_x.spines['right'].set_bounds(0, -30)
    # Plot the hydro
    df_hydro = read_fsb_hydro(hydro_path)
    pq_axes = plot_fsb_hydro(df_hydro, axes=ax_hydro)
    # Add 1 hour to Swiss local winter time
    pq_axes[0].axvline(x=time, linestyle=':', color='k', linewidth=2.)
    pq_axes[0].set_xlabel('Time', fontsize=14)
    return fig