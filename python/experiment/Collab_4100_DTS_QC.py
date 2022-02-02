#!/usr/bin/python

"""
Monitor the DTS xml directories and plot QC plots periodically
"""

import os
import time
import matplotlib

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from lxml import etree
from glob import glob
from obspy import UTCDateTime
from lxml.etree import XMLSyntaxError
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

# Avoid potential memory leaks
# See: https://github.com/matplotlib/matplotlib/issues/20490
matplotlib.use('agg')

chan_map_4100 = {'AMU': (85, 210), 'AML': (220, 343),
                 'DMU': (384, 495), 'DML': (505, 620)}

channel_mapping = {'collab41': chan_map_4100}

efsl_wind = 0

fiber_depth_4100 = {'AMU': 60, 'AML': 60, 'DMU': 55, 'DML': 55}


def starttime_generator(start_date, end_date, stride):
    # Generator for date looping
    from datetime import timedelta
    total_sec = (end_date - start_date).seconds + ((end_date -
                                                    start_date).days * 86400)
    for n in range(int(total_sec / stride) + 1):
        yield start_date + timedelta(seconds=n * stride)


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


def read_XTDTS(path, no_cols):
    # Read single xml file and return array for all values and time
    try:
        dts = etree.parse(path)
    except XMLSyntaxError as e:
        return None
    # Get root element
    root = dts.getroot()
    # Create one string for all values, comma sep
    measurements = np.fromstring(','.join(
        [l.text.replace('\n', '')
         for l in root[0].find('{*}logData').findall('{*}data')]),
        sep=',')
    # 6 columns in original data
    measurements = measurements.reshape(-1, no_cols)
    # Get time
    dto = UTCDateTime(root[0].find('{*}endDateTimeIndex').text).datetime
    return dto, measurements


def read_XTDTS_dir(files, wells, mapping, no_cols,
                   noise_method='madjdabadi'):
    """
    Read all files in a directory to 2D DTS arrays

    :param files: List of files to read
    :param wells: List of well names
    :param mapping: String for field location ('fsb' or 'efsl')
    :param no_cols: Number of columns in XT-DTS data file
    :param noise method: Method string for noise calculation

    :return:
    """
    results = [read_XTDTS(str(f), no_cols) for f in files]
    results = [r for r in results if r]
    times, measures = zip(*results)
    times = np.array(times)
    measures = np.stack(measures, axis=-1)
    # Make same dict as for other sources
    fiber_data = {'times': times, 'anti-stokes': measures[:, 2, :],
                  'stokes': measures[:, 1, :], 'data': measures[:, 5, :],
                  'depth': measures[:, 0, 0]}
    fiber_depths = fiber_depth_4100
    fiber_wind = efsl_wind
    well_data = {}
    chan_map = channel_mapping[mapping]
    for well in wells:
        if well not in chan_map:
            print('{} not in mapping'.format(well))
            continue
        # For FSB B* wells, this accounts for XX% greater fiber depth than TD
        fiber_depth = (fiber_depths[well] / np.cos(np.deg2rad(fiber_wind)))
        depth = fiber_data['depth'].copy()
        data = fiber_data['data'].copy()
        times = fiber_data['times'].copy()
        start_chan = np.abs(depth - chan_map[well][0])
        end_chan = np.abs(depth - chan_map[well][1])
        # Find the closest integer channel to meter mapping
        data_tmp = data[np.argmin(start_chan):np.argmin(end_chan), :]
        depth_tmp = depth[np.argmin(start_chan):np.argmin(end_chan)]
        # Account for cable winding
        depth_tmp *= np.cos(np.deg2rad(fiber_wind))
        noise = estimate_noise(data_tmp, method=noise_method)
        well_data[well] = {'data': data_tmp, 'depth': depth_tmp,
                           'noise': noise, 'times': times, 'mode': None,
                           'type': None}
    return well_data


def plot_4100_QC(well_data, well, depths, baseline, outfile,
                 date_range=None, vrange_T=(0, 80), vrange_dT=(-5, 5)):
    """
    Multi-panel QC plot of EFSL DTS data

    :param well_data: Dict output from read_XTDTS_dir
    :param well: String for which well to plot
    :param depths: List of depths to plot timeseries for
    :param baseline: Path to npy binary with baseline T vector
    :param outfile: Absolute path to output plot
    :param date_range: Tuple of start and end datetimes to plot
    :param vrange_T: Tuple of top and bottom temperature for colormap
    :param vrange_dT: Tuple of top and bottom dT for colormap
    """
    # Set up figure
    fig = plt.figure(constrained_layout=False, figsize=(22, 14))
    gs = GridSpec(ncols=12, nrows=12, figure=fig)
    axes_depth = fig.add_subplot(gs[:, :2])
    axes_T = fig.add_subplot(gs[:5, 2:-1])
    axes_dT = fig.add_subplot(gs[5:-2, 2:-1], sharex=axes_T)
    axes_ts = fig.add_subplot(gs[-2:, 2:-1], sharex=axes_T)
    cax_T = fig.add_subplot(gs[:5, -1])
    cax_dT = fig.add_subplot(gs[5:-2, -1])
    # Pull out the datetime vector
    times = well_data[well]['times']
    # Grab T data, np.gradient for dT
    T = well_data[well]['data']
    # dT = T.copy() - T[:, 0, np.newaxis]
    dT = np.gradient(T, axis=1)
    if date_range:
        indices = np.where((date_range[0] < times) & (times < date_range[1]))
        times = times[indices]
        T = np.squeeze(T[:, indices])
        dT = np.squeeze(dT[:, indices])
    # Set up depth and time vectors for plotting
    depth = well_data[well]['depth']
    # Make depth relative to "wellhead"
    depth = depth - depth[0]
    mpl_times = mdates.date2num(times)
    # Two different colormaps
    cmap_T = ListedColormap(sns.color_palette('magma', 40).as_hex())
    cmap_dT = ListedColormap(sns.color_palette('vlag', 40).as_hex())
    # Latex labels
    label_T = r'$^O$C'
    label_dT = r'$\Delta^O$C'
    # Plot waterfalls
    im_T = axes_T.imshow(T, cmap=cmap_T, origin='upper',
                         extent=[mpl_times[0], mpl_times[-1],
                                 depth[-1], depth[0]],
                         aspect='auto', vmin=vrange_T[0], vmax=vrange_T[1])
    im_dT = axes_dT.imshow(np.flip(dT, axis=0), cmap=cmap_dT, origin='lower',
                           extent=[mpl_times[0], mpl_times[-1],
                                   depth[-1], depth[0]],
                           aspect='auto', vmin=vrange_dT[0], vmax=vrange_dT[1])
    cbar_T = fig.colorbar(im_T, cax=cax_T, orientation='vertical')
    cbar_T.ax.set_ylabel(label_T, fontsize=16)
    cbar_dT = fig.colorbar(im_dT, cax=cax_dT, orientation='vertical')
    cbar_dT.ax.set_ylabel(label_dT, fontsize=16)
    # Now time snapshots along the well
    # By default, do baseline, and two equally spaced slices in date_range
    # Read in baseline
    baseline = np.load(baseline)
    axes_depth.plot(baseline, depth, label='Baseline')
    third = T.shape[1] // 3
    t1 = T[:,third]
    t2 = T[:,2 * third]
    axes_depth.plot(t1, depth, label=times[third], color='darkgray')
    axes_depth.plot(t2, depth, label=times[2*third], color='darkblue')
    # Plot times on T waterfall
    axes_T.axvline(mpl_times[third], linestyle=':', color='darkgray')
    axes_T.axvline(mpl_times[2 * third], linestyle=':', color='darkblue')
    # Now do depth timeseries
    cols = ['firebrick', 'steelblue']
    for i, d in enumerate(depths):
        d_ind = np.argmin(np.abs(d - depth))
        d_ts = T[d_ind, :]
        axes_ts.plot(mpl_times, d_ts, color=cols[i], label='{} m'.format(d))
        # Plot depth on waterfall
        axes_T.axhline(d, linestyle='--', color=cols[i])
    # Axes formatting
    axes_depth.invert_yaxis()
    axes_depth.margins(0.)
    axes_depth.set_xlim(vrange_T)
    axes_depth.set_xlabel(label_T)
    axes_depth.set_ylabel('Measured Depth [m]')
    axes_depth.legend(loc=2, bbox_to_anchor=(-0.2, 1.08),
                      framealpha=1.).set_zorder(103)
    axes_ts.margins(0.)
    axes_ts.set_ylim(vrange_T)
    axes_ts.xaxis_date()
    axes_ts.legend()
    date_formatter = mdates.DateFormatter('%m-%d %H:%M')
    axes_ts.xaxis.set_major_formatter(date_formatter)
    axes_ts.set_ylabel(label_T)
    plt.setp(axes_ts.xaxis.get_majorticklabels(), rotation=30, ha='right',
             fontsize=14)
    plt.setp(axes_T.get_xticklabels(), visible=False)
    plt.setp(axes_dT.get_xticklabels(), visible=False)
    axes_T.set_ylabel('MD [m]')
    axes_dT.set_ylabel('MD [m]')
    plt.subplots_adjust(wspace=1.)
    plt.suptitle('ACEFFL DTS: {}\n{} -- {}'.format(well, times[0], times[-1]),
                 fontsize=18)
    plt.savefig(outfile)
    plt.close('all')
    return


def launch_processing(files_c1, baselines, ping_interval, plot_length_seconds,
                      outpath):
    """
    Housekeeping and generate plots for back-recordings

    :param files_c1: List of files to process for channel 1
    :param baselines: Path to directory of np binary of the baseline T
    :param ping_interval: Time step between generating each plot
    :param plot_length_seconds: Length of plot in seconds
    :param outpath: Output plot directory
    """
    # Sort files by date (filename)
    files_c1 = sorted(files_c1)
    tstrings_c1 = [f.split('_')[-1][:-7] for f in files_c1]
    times_c1 = [datetime.strptime(ts, '%Y%m%d%H%M%S') for ts in tstrings_c1]
    print('Producing plots for back-recorded data:\n{} to {}'.format(
        times_c1[0], times_c1[-1]))
    baselines = glob('{}\*.npy'.format(baselines))
    base_dict = {f.split(os.sep)[-1].split('_')[0]: f for f in baselines}
    # Now loop over the number of intervals for this file list
    for start in starttime_generator(times_c1[0], times_c1[-1], ping_interval):
        # Get the file indices for this plot
        indices_c1 = np.where((start <= np.array(times_c1)) &
                              (start + timedelta(seconds=plot_length_seconds) >
                               np.array(times_c1)))[0]
        these_times_c1 = [times_c1[i] for i in list(indices_c1)]
        if (((these_times_c1[-1] - these_times_c1[0]).seconds + ping_interval) <
                plot_length_seconds):
            return
        well_data_c1 = read_XTDTS_dir(
            [files_c1[i] for i in list(indices_c1)],
            wells=['AMU', 'AML', 'DMU', 'DML'], mapping='collab41', no_cols=6)
        for w in well_data_c1.keys():
            out_c1 = '{}\Collab_4100_DTS_QC_{}_{}_{}.png'.format(
                outpath, w, these_times_c1[0].strftime('%Y-%m-%dT%H-%M'),
                these_times_c1[-1].strftime('%Y-%m-%dT%H-%M'))
            # If plots exist, skip this
            if not os.path.isfile(out_c1):
                print('Plotting {}'.format(out_c1))
                plot_4100_QC(well_data_c1, well=w, depths=[20, 40],
                             baseline=base_dict[w], outfile=out_c1,
                             vrange_T=(15, 40))
            else:
                print('Already plotted\n{}'.format(out_c1))
    return


### Stolen from DUG-seis live processing script

path_c1 = r'C:\Users\loaner\Google Drive\collabDataTransfer\rawDTSdata\data\XT16034\temperature\EGSCOLLAB-Grout\channel 1\*.xml'

f_c1 = glob(path_c1)

outpath = r'C:\Users\loaner\Google Drive\collabDataTransfer\rawDTSplots\Collab4100-grouting'

baselines = r'C:\Users\loaner\Google Drive\collabDataTransfer\rawDTSdata\data\baselines'

ping_interval_in_seconds = 1200  # How often to attempt to generate a plot

plot_length_seconds = 36000  # Length of each plot in seconds

# Monitor directory for files and wait until there are some
while True:
    try:  # Wait till arbitrary number of files in directory
        test39 = f_c1[15]
    except IndexError as err:
        print("Not enough data yet - trying again in {} seconds".format(
            ping_interval_in_seconds))
        time.sleep(ping_interval_in_seconds)
        raise err
    break


launch_processing(
    files_c1=f_c1, baselines=baselines,
    ping_interval=ping_interval_in_seconds,
    plot_length_seconds=plot_length_seconds,
    outpath=outpath)

# Wait
print('Done with back-recorded data')
time.sleep(ping_interval_in_seconds)

# Monitor the folders and launch the processing again.
while True:

    all_files_c1 = glob(path_c1)

    if all_files_c1 == f_c1:
        print('No new files written. Waiting')
        time.sleep(ping_interval_in_seconds)
        continue
    baselines = glob('{}\*.npy'.format(baselines))
    base_dict = {f.split(os.sep)[-1].split('_')[0]: f for f in baselines}
    # Determine endtime and backcalculate start
    endtime_c1 = str(sorted(all_files_c1)[-1]).split('_')[-1][:-7]
    endtime_c1 = datetime.strptime(endtime_c1, '%Y%m%d%H%M%S')
    starttime_c1 = endtime_c1 - timedelta(seconds=plot_length_seconds)
    print('Producing plot for {} to {}'.format(starttime_c1, endtime_c1))
    # Get the appropriate files
    all_files_chan1 = sorted(all_files_c1)
    tstrings_c1 = [f.split('_')[-1][:-7] for f in all_files_chan1]
    times_c1 = [datetime.strptime(ts, '%Y%m%d%H%M%S') for ts in tstrings_c1]
    indices_c1 = np.where((starttime_c1 <= np.array(times_c1)) &
                          (np.array(times_c1) < endtime_c1))[0]
    # Read them in
    well_data_c1 = read_XTDTS_dir([all_files_chan1[i] for i in list(indices_c1)],
                                  wells=['AMU', 'AML', 'DMU', 'DML'],
                                  mapping='collab41', no_cols=6)
    # Plot them
    these_times_c1 = [times_c1[i] for i in list(indices_c1)]
    for w in well_data_c1.keys():
        out_c1 = '{}\Collab_4100_DTS_QC_{}_{}_{}.png'.format(
            outpath, w, these_times_c1[0].strftime('%Y-%m-%dT%H-%M'),
            these_times_c1[-1].strftime('%Y-%m-%dT%H-%M'))
        # If plots exist, skip this
        if not os.path.isfile(out_c1):
            print('Plotting {}'.format(out_c1))
            plot_4100_QC(well_data_c1, well=w, depths=[20, 40],
                         baseline=base_dict[w], outfile=out_c1,
                         vrange_T=(15, 40))
        else:
            print('Already plotted\n{}'.format(out_c1))
    time.sleep(ping_interval_in_seconds)
