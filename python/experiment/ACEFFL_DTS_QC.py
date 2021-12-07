#!/usr/bin/python

"""
Monitor the DTS xml directories and plot QC plots periodically
"""

import os
import time
import pathlib

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


chan_map_EFSL = {'3359': (76.56, 5358.7), '3339': (99.25, 5193.25)}

channel_mapping = {'efsl': chan_map_EFSL}

efsl_wind = 0

fiber_depth_efsl = {'3359': 5399.617, '3339': 5249.653}


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
                  'stokes': measures[:, 1, :], 'data': measures[:, 3, :],
                  'depth': measures[:, 0, 0]}
    fiber_depths = fiber_depth_efsl
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


def plot_EFSL_QC(well_data, well, depths, baseline, outfile,
                 date_range=None, vrange_T=(0, 110), vrange_dT=(-5, 5)):
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


def launch_processing(files_39, files_59, baseline_39, baseline_59,
                      ping_interval, plot_length_seconds, outpath):
    """
    Housekeeping and generate plots for back-recordings

    :param files_39: List of files to process for well 3339
    :param files_59: List of files to process for well 3359
    :param baseline_39: Path to np binary of the baseline T for 3339
    :param baseline_59: Path to np binary of the baseline T for 3359
    :param ping_interval: Time step between generating each plot
    :param plot_length_seconds: Length of plot in seconds
    :param outpath: Output plot directory
    """
    # Sort files by date (filename)
    files_39 = sorted(files_39)
    files_59 = sorted(files_59)
    tstrings_39 = [''.join(f.split('_')[-2:])[:-8] for f in files_39]
    tstrings_59 = [''.join(f.split('_')[-2:])[:-8] for f in files_59]
    times_39 = [datetime.strptime(ts, '%Y%m%d%H%M%S') for ts in tstrings_39]
    times_59 = [datetime.strptime(ts, '%Y%m%d%H%M%S') for ts in tstrings_59]
    print('Producing plots for back-recorded data:\n{} to {}'.format(
        times_39[0], times_39[-1]))
    # Now loop over the number of intervals for this file list
    used_39 = set()
    used_59 = set()
    for start in starttime_generator(times_39[0], times_39[-1], ping_interval):
        # Get the file indices for this plot
        indices_39 = np.where((start <= np.array(times_39)) &
                              (start + timedelta(seconds=plot_length_seconds) >
                               np.array(times_39)))[0]
        indices_59 = np.where((start <= np.array(times_59)) &
                              (start + timedelta(seconds=plot_length_seconds) >
                               np.array(times_59)))[0]
        these_times_39 = [times_39[i] for i in list(indices_39)]
        these_times_59 = [times_39[i] for i in list(indices_59)]
        out_39 = '{}\ACEFFL_DTS_QC_{}_{}_{}.png'.format(
            outpath, '3339', these_times_39[0].strftime('%Y-%m-%dT%H-%M'),
            these_times_39[-1].strftime('%Y-%m-%dT%H-%M'))
        out_59 = '{}\ACEFFL_DTS_QC_{}_{}_{}.png'.format(
            outpath, '3359', these_times_59[0].strftime('%Y-%m-%dT%H-%M'),
            these_times_59[-1].strftime('%Y-%m-%dT%H-%M'))
        # If plots exist, skip this
        if not (os.path.isfile(out_39) and os.path.isfile(out_59)):
            well_data_39 = read_XTDTS_dir([files_39[i] for i in list(indices_39)],
                                          wells=['3339'], mapping='efsl', no_cols=4)
            well_data_59 = read_XTDTS_dir([files_59[i] for i in list(indices_59)],
                                          wells=['3359'], mapping='efsl', no_cols=4)
            plot_EFSL_QC(well_data_39, well='3339', depths=[2000, 5080],
                         baseline=baseline_39, outfile=out_39)
            plot_EFSL_QC(well_data_59, well='3359', depths=[2000, 5080],
                         baseline=baseline_59, outfile=out_59)
        else:
            print('{}\n{}\nAlready plotted'.format(out_39, out_59))
        # Update which files have been used
        used_39.update(set([files_39[i] for i in list(indices_39)]))
        used_59.update(set([files_59[i] for i in list(indices_59)]))
    return used_39, used_59


### Stolen from DUG-seis live processing script

f_3339 = glob(r'C:\Program Files (x86)\XT Client\XTClientCore\app data\data\XT20018\XT20018\temperature\ACEFFL 24 Nov 2021\channel 4\*.xml')
f_3359 = glob(r'C:\Program Files (x86)\XT Client\XTClientCore\app data\data\XT20018\XT20018\temperature\ACEFFL 24 Nov 2021\channel 1\*.xml')

outpath = 'Z:\91_QC\DTS'

baseline_39 = r'Z:\91_QC\DTS\3339_baseline.npy'
baseline_59 = r'Z:\91_QC\DTS\3359_baseline.npy'

ping_interval_in_seconds = 600  # How often to attempt to generate a plot

plot_length_seconds = 7200  # Length of each plot in seconds

# Monitor directory for files and wait until there are some
while True:
    try:  # Wait till arbitrary number of files in directory
        test39 = f_3339[150]
        test59 = f_3359[150]
    except IndexError as err:
        print("Not enough data yet - trying again in {} seconds".format(
            ping_interval_in_seconds))
        time.sleep(ping_interval_in_seconds)
        raise err
    break


used_39, used_59 = launch_processing(
    files_39=f_3339, files_59=f_3359,
    baseline_39=baseline_39, baseline_59=baseline_59,
    ping_interval=ping_interval_in_seconds,
    plot_length_seconds=plot_length_seconds,
    outpath=outpath)

# Wait
print('Done with back-recorded data')
time.sleep(ping_interval_in_seconds)

# Monitor the folders and launch the processing again.
while True:
    all_files_3339 = set()
    all_files_3359 = set()
    all_files_3339 = all_files_3339.union(set(f_3339))
    all_files_3359 = all_files_3359.union(set(f_3359))

    new_files_39 = all_files_3339.difference(used_39)
    new_files_59 = all_files_3359.difference(used_59)

    if not (new_files_39 and new_files_59):
        print("No new files yet - trying again in {} seconds".format(
            ping_interval_in_seconds))
        time.sleep(ping_interval_in_seconds)
        continue
    # Determine endtime and backcalculate start
    endtime_39 = ''.join(str(sorted(new_files_39)[-1]).split('_')[-2:])[:-8]
    endtime_39 = datetime.strptime(endtime_39, '%Y%m%d%H%M%S')
    starttime_39 = endtime_39 - timedelta(seconds=plot_length_seconds)
    endtime_59 = ''.join(str(sorted(new_files_59)[-1]).split('_')[-2:])[:-8]
    endtime_59 = datetime.strptime(endtime_59, '%Y%m%d%H%M%S')
    starttime_59 = endtime_59 - timedelta(seconds=plot_length_seconds)
    print('Producing plot for {} to {}'.format(starttime_39, endtime_39))
    # Get the appropriate files
    all_files_3339 = sorted(all_files_3339)
    all_files_3359 = sorted(all_files_3359)
    tstrings_39 = [''.join(f.split('_')[-2:])[:-8] for f in all_files_3339]
    tstrings_59 = [''.join(f.split('_')[-2:])[:-8] for f in all_files_3359]
    times_39 = [datetime.strptime(ts, '%Y%m%d%H%M%S') for ts in tstrings_39]
    times_59 = [datetime.strptime(ts, '%Y%m%d%H%M%S') for ts in tstrings_59]
    indices_39 = np.where((starttime_39 <= np.array(times_39)) &
                          (np.array(times_39) < endtime_39))[0]
    indices_59 = np.where((starttime_59 <= np.array(times_59)) &
                          (np.array(times_59) < endtime_59))[0]
    # Read them in
    well_data_39 = read_XTDTS_dir([all_files_3339[i] for i in list(indices_39)],
                                  wells=['3339'], mapping='efsl', no_cols=4)
    well_data_59 = read_XTDTS_dir([all_files_3359[i] for i in list(indices_59)],
                                  wells=['3359'], mapping='efsl', no_cols=4)
    # Plot them
    these_times_39 = [times_39[i] for i in list(indices_39)]
    these_times_59 = [times_39[i] for i in list(indices_59)]
    out_39 = '{}\ACEFFL_DTS_QC_{}_{}_{}.png'.format(
        outpath, '3339', these_times_39[0].strftime('%Y-%m-%dT%H-%M'),
        these_times_39[-1].strftime('%Y-%m-%dT%H-%M'))
    out_59 = '{}\ACEFFL_DTS_QC_{}_{}_{}.png'.format(
        outpath, '3359', these_times_59[0].strftime('%Y-%m-%dT%H-%M'),
        these_times_59[-1].strftime('%Y-%m-%dT%H-%M'))
    plot_EFSL_QC(well_data_39, well='3339', depths=[2000, 5080],
                 baseline=baseline_39, outfile=out_39)
    plot_EFSL_QC(well_data_59, well='3359', depths=[2000, 5080],
                 baseline=baseline_59, outfile=out_59)
    # Update which files have been used
    used_39.update(set([all_files_3339[i] for i in list(indices_39)]))
    used_59.update(set([all_files_3359[i] for i in list(indices_59)]))
