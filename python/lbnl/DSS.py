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

from glob import glob
from obspy import Stream, Trace
from eqcorrscan.core.match_filter import normxcorr2
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import detrend, welch, find_peaks
from scipy.stats import median_absolute_deviation
from datetime import datetime, timedelta
from itertools import cycle
from matplotlib.dates import num2date
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

# Local imports
from lbnl.coordinates import cartesian_distance
from lbnl.boreholes import parse_surf_boreholes, create_FSB_boreholes,\
    calculate_frac_density, read_frac_cores
from lbnl.simfip import read_excavation, plot_displacement_components


######### SURF CHANNEL MAPPING ############
chan_map_surf = {}

########## FSB DSS CHANNEL MAPPINGS ###########
# Michelle DataViewer mapping (tug test)
chan_map_fsb = {'B3': (237.7, 404.07), 'B4': (413.52, 571.90),
                'B5': (80.97, 199.63), 'B6': (594.76, 694.32),
                'B7': (700.43, 793.47)}
# Maria mapping (via ft markings on cable and scaling)
chan_map_maria = {'B3': (232.21, 401.37), 'B4': (406.56, 566.58),
                  'B5': (76.46, 194.11), 'B6': (588.22, 688.19),
                  'B7': (693.37, 789.86)}

########## CSD CHANNEL MAPPINGS ###########
# Antonio Dataviewer channel mapping
chan_map_csd_1256 = {# Loop 1, 2, 5, 6
                     'D1': (336.56, 379.08), 'D2': (259.99, 294.19),
                     'D5': (67.16, 126.04), 'D6': (157.42, 219.57)}
chan_map_csd_34 = {# Loop 3, 4
                   'D3': (47.26, 106.66), 'D4': (132.34, 201.49)}

# Solexperts channel mapping
chan_map_solexp_1256 = {# Loop 1, 2, 5, 6
                        'D1': (336.56, 379.08), 'D2': (259.99, 294.19),
                        'D5': (68.61, 131.40), 'D6': (154.62, 227.21)}
chan_map_solexp_34 = {# Loop 3, 4
                      'D3': (48.60, 111.44), 'D4': (134.82, 206.84)}

# Bottom hole mapping (by me from DataViewer)
# chan_map_bottom_1256 = {# Loop 1, 2, 5, 6
#                         'D1': None, 'D2': None,
#                         'D5': 97.03, 'D6': 187.65}
# Swapped mapping
chan_map_bottom_1256 = {# Loop 1, 2, 5, 6
                        'D1': None, 'D2': None,
                        'D5': 187.65, 'D6': 97.03}

chan_map_bottom_34 = {# Loop 3, 4
                      'D3': 76.65, 'D4': 167.24}

# Excavation correlation mapping
chan_map_excav_56 = {# Loop 5, 6
                     'D5': 187.535,
                     'D6': 97.145}
chan_map_excav_34 = {# Loop 3, 4
                     'D3': 76.61,
                     'D4': 167.22}
chan_map_co2_5612 = {# Loop 5, 6
                     'D5': 95.92,
                     'D6': 186.74,
                     'D1': 353.64,
                     'D2': 272.91}
chan_map_co2_34 = {# Loop 3, 4
                   'D3': 76.12,
                   'D4': 166.93}

######### DRILLING FAULT DEPTH ############
# Dict of drilled depths
drilled_depths = {'D1': 25.2, 'D2': 18.55, 'D3': 31.65, 'D4': 36.9, 'D5': 31.79,
                  'D6': 36.65, 'D7': 29.7}

fault_depths = {'D1': (), 'D2': (13.25, 16.45), 'D3': (17.98, 20.58),
                'D4': (27., 30.), 'D5': (19.65, 22.65), 'D6': (28.5, 31.36),
                'D7': (22.3, 25.45)}

mapping_dict = {'solexperts': {'CSD3': chan_map_solexp_34,
                               'CSD5': chan_map_solexp_1256,
                               'FSB': chan_map_fsb},
                'antonio': {'CSD3': chan_map_csd_34,
                            'CSD5': chan_map_csd_1256,
                            'FSB': chan_map_fsb},
                'bottom': {'CSD3': chan_map_bottom_34,
                           'CSD5': chan_map_bottom_1256,
                           'FSB': chan_map_fsb},
                'excavation': {'CSD3': chan_map_excav_34,
                               'CSD5': chan_map_excav_56,
                               'FSB': chan_map_fsb},
                'co2_injection': {'CSD3': chan_map_co2_34,
                                  'CSD5': chan_map_co2_5612,
                                  'FSB': chan_map_fsb}}

# Custom color palette similar to wellcad convention
frac_cols = {'All fractures': 'black',
             'open/undif. fracture': 'blue',
             'sealed fracture / vein': 'lightblue',
             'foliation / bedding': 'red',
             'induced fracture': 'magenta',
             'sedimentary structures/color changes undif.': 'green',
             'uncertain type': 'orange',
             'lithology change': 'yellow'}


def read_ascii(path, header=42, encoding='iso-8859-1'):
    """Read in a raw DSS file (flipped about axis 1 for left-to-right time"""
    return np.flip(np.loadtxt(path, skiprows=header, encoding=encoding), 1)


def read_neubrex(path, header=105, encoding='iso-8859-1'):
    """Read in raw Neubrex (SolExperts) measurement"""
    # Flip time axis back as only single measurements
    try:
        data = np.flip(read_ascii(path, header, encoding), 1)
    except ValueError:
        data = np.flip(read_ascii(path, header=122, encoding=encoding), 1)
    depths = data[:, 1]
    data = data[:, -1]
    try:
        times = read_times(path, header=67, encoding=encoding,
                           time_fmt='%Y/%m/%d %H:%M:%S.%f')
    except ValueError:
        times = read_times(path, header=75, encoding=encoding,
                           time_fmt='%Y/%m/%d %H:%M:%S.%f')
    return data, depths, times


def read_potentiometer(path):
    """Read Antonio's potentiometer data file"""
    data = np.loadtxt(path, skiprows=7, encoding='iso-8859-1').T
    depths = np.genfromtxt(path, skip_header=4, max_rows=1,
                           encoding='iso-8859-1')
    times = read_times(path, header=1, time_fmt='%d/%m/%Y %H:%M:%S')[::-1]
    return data, depths, times


def datetime_parse(t, fmt):
    # Parse the date format of the DSS headers
    return datetime.strptime(t, fmt)


def read_times(path, encoding='iso-8859-1', header=10,
               time_fmt='%Y/%m/%d %H:%M:%S'):
    """Read timestamps from ascii header"""
    strings = np.genfromtxt(path, skip_header=header, max_rows=1,
                            encoding=encoding, dtype=None, delimiter='\t')
    if header == 1:  # Potentiometer file
        return np.array([datetime_parse(t, time_fmt)
                         for t in strings[:-1]])[::-1]
    elif header == 10:  # Omnisens output
        return np.array([datetime_parse(t, time_fmt)
                         for t in strings[1:-1]])[::-1]
    elif header == 67:  # Neubrex output file
        row_str = str(strings).split()
        time_str = ' '.join([row_str[-2], row_str[-1]])
        return np.array([datetime_parse(time_str, time_fmt)])


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


def extract_wells(root, measure, mapping, wells=None, fibers=None,
                  noise_method='majdabadi'):
    """
    Helper to extract only the channels in specific wells

    :param root: Root directory for all measurement text files
    :param measure: Which measure to read:
        Absolute_Strain
        Absolute_Freq
        Absolute_Gain
        Relative_Freq
        Relative_Strain
    :param mapping: For Mont Terri, specifically, who's channel mapping do
        we use? The preferred mappings are now 'excavation' or 'co2_injection'
    :param wells: List of well name strings to return
    :param fibers: Optionaly specify individual fiber loops (FSB, CSD3 or CSD5)
    :param noise_method: 'majdabadi' or 'by_channel' to estimate noise.
        'majdabadi' returns scalar, 'by_channel' an array

    :returns: dict {well name: {'data':, 'depth':, 'noise':}
    """
    if not fibers and not wells:
        print('Must specify either fibers or wells')
        return
    data_files = glob('{}/*{}.txt'.format(root, measure))
    well_data = {}
    for f in data_files:
        if f.split('/')[-1].startswith('FSB-SMF-1'):
            # Skip fiber 1
            continue
        file_root = f.split('/')[-1].split('-')[0]
        if fibers:
            if file_root not in fibers:
                continue
        chan_map = mapping_dict[mapping][file_root]
        data = read_ascii(f)
        times = read_times(f)
        mode, type = read_metadata(f)
        # Take first column as the length along the fiber and remove from data
        depth = data[:, -1]
        data = data[:, :-1]
        # First realign
        data = madjdabadi_realign(data)
        # Take selected channels
        if wells == 'all' or fibers:
            channel_range = (0, -1)
        channel_ranges = []
        if wells:
            for well in wells:
                if well not in chan_map:
                    continue
                if (mapping in ['bottom', 'excavation', 'co2_injection'] and
                    well.startswith('D')):
                    start_chan = np.abs(depth - (chan_map[well] -
                                                 drilled_depths[well]))
                    end_chan = np.abs(depth - (chan_map[well] +
                                               drilled_depths[well]))
                else:
                    start_chan = np.abs(depth - chan_map[well][0])
                    end_chan = np.abs(depth - chan_map[well][1])
                # Find the closest integer channel to meter mapping
                channel_range = (np.argmin(start_chan), np.argmin(end_chan))
                channel_ranges.append((well, channel_range))
                well_data[well] = {'times': times, 'mode': mode,
                                   'type': type}
        elif fibers:
            for fiber in fibers:
                if fiber == file_root:
                    well_data[fiber] = {'times': times, 'mode': mode,
                                        'type': type}
                    channel_ranges.append((fiber, channel_range))
        for rng in channel_ranges:
            data_tmp = data[rng[1][0]:rng[1][1], :]
            depth_tmp = depth[rng[1][0]:rng[1][1]]
            # Get median absolute deviation averaged across all channels
            noise = estimate_noise(data_tmp, method=noise_method)
            well_data[rng[0]].update({'data': data_tmp, 'depth': depth_tmp,
                                      'noise': noise})
    return well_data


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


def estimate_noise(data, method='majdabadi'):
    """
    Calculate the average MAD for all channels similar to Madjdabadi 2016,
    but replacing std with MAD

    Alternatively, don't take the average and return both the mean and MAD
    as arrays

    :param data: Numpy array of DSS data
    :return:
    """
    if method == 'majdabadi':
        # Take MAD of each channel time series, then average
        return np.mean(median_absolute_deviation(data, axis=1)), None
    elif method == 'Krietsch':
        return np.mean(np.percentile(data, q=[10, 90], axis=1), axis=1)
    elif method == 'by_channel':
        return np.mean(data, axis=1), median_absolute_deviation(data, axis=1)
    else:
        print('Invalid method for denoise')
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


def DSS_spectrum(path, well='all', domain='time'):
    times, data, depth = extract_wells(path, well)
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


def pick_anomalies(data, noise_mean, noise_mad, thresh=1.):
    """
    Pick every point where the data exceeds the noise and return the width and
    amplitude of the peak.

    :param data: Strain data to pick
    :param noise_mean: Mean value of the noise (integer or array)
    :param noise_mad: MAD of the data, returned from extract_wells
        Array or integer
    :param thresh: MAD multiplier for threshold
    :return:
    """
    return find_peaks(np.abs(data), height=np.abs(noise_mean) +
                      (np.abs(noise_mad) * thresh),
                      width=(None, None), prominence=30.)


def correlate_fibers(template, template_lengths, image, image_lengths,
                     plot=False, title='Fiber Cross Correlation'):
    """
    Correlate one fiber with another (probably for channel mapping purposes)

    Just a thin wrapper on eqcorrscan normxcorr2 to handle the mapping
    to/from fiber length

    :param template: "Template" waveform, as in Matched Filtering
    :param template_lengths: Array of same shape as template, with values
        representing actual fiber length for each element of template
    :param image: Image waveform to scan over
    :param image: Array of same shape as image, with values
        representing actual fiber length for each element of image
    :param plot: Plot flag to output shifted template, image and ccc

    :return: Array of correlation coefficients, shift for temp in length unit
    """

    # Check sampling intervals for consistency
    if not np.isclose(np.diff(template_lengths)[0], np.diff(image_lengths)[0]):
        print('Sampling intervals not equal. Fix this first')
        return
    ccc = normxcorr2(template, image)[0]
    samp_int = np.diff(template_lengths)[0]  # Make sure this is stable
    shift = (samp_int * np.argmax(ccc)) - template_lengths[0]
    if plot:
        # Make length array for ccc
        ccc_length = (np.arange(ccc.size) * samp_int) + template_lengths[0]
        plot_fiber_correlation(template, template_lengths, image, image_lengths,
                               ccc, ccc_length, shift, title)
    return ccc, shift


################  Plotting  Funcs  ############################################

def plot_fiber_correlation(template, template_lengths, image, image_lengths,
                           ccc, ccc_length, shift, title):
    """
    Two-axes plot of template, image and cross correlation coefficient
    for two lengths of fiber

    :param template: Template waveform
    :param template_lengths: Array of lengths for each sample in template
    :param image: Image waveform
    :param image_lengths: Array of lengths for each sample in image
    :param ccc: CCC array
    :param ccc_length: Array of lengths corresponding to ccc elements
    :param shift: Shift (in length units) to achieve maximum ccc
    :return:
    """

    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), sharex=True)
    axes[0].plot(template_lengths, template, label='Template', linewidth=1.,
                 color='r', alpha=0.7)
    axes[0].plot(image_lengths, image, label='Image', linewidth=1., color='k')
    axes[0].plot(template_lengths + shift, template, label='Shifted Template',
                 linewidth=1., color='b', alpha=0.7)
    axes[1].plot(ccc_length, ccc, label='CCC')
    max_ccc_line = ccc_length[ccc.argmax()]
    axes[1].axvline(max_ccc_line, linestyle='--', color='k')
    axes[1].annotate(xy=(max_ccc_line, ccc.max()),
                     xytext=(max_ccc_line + (120 * np.diff(image_lengths)[0]),
                             ccc.max() - 0.2),
                     s='CCC: {:0.3f}\nShift: {:0.2f} m'.format(ccc.max(),
                                                               shift),
                     arrowprops={'arrowstyle': '->'},
                     fontsize=14, bbox={'facecolor': 'w', 'edgecolor': 'k'})
    # Format
    axes[0].set_ylabel('Brillouin Frequency [GHz]', fontsize=16)
    axes[1].set_ylabel('Cross correlation coefficient', fontsize=16)
    axes[1].set_xlabel('Length [m]', fontsize=16)
    axes[0].legend()
    axes[1].legend()
    axes[0].set_ylim(10.5, 11.1)
    axes[1].set_ylim(-1, 1)
    if not title:
        title = 'Fiber cross correlation'
    fig.suptitle(title, fontsize=20)
    plt.show()
    return


def plot_strains_w_dist(location, DSS_picks, point):
    """
    Plot DSS strain values with distance from a point (e.g. excavation front)

    :param location: 'fsb' or 'surf'
    :param DSS_picks: Dictionary {well name: [pick depths, ...]}
    :param point: (x, y, z) point to calculate distances from
    :return:
    """

    if location == 'surf':
        well_dict = parse_surf_boreholes(well_file)
    elif location == 'fsb':
        # Too many points in asbuilt file to upload to plotly
        well_dict = create_FSB_boreholes()
    else:
        print('Location {} not supported'.format(location))
        return
    dist_list = []
    for well, pick_dict in DSS_picks.items():
        easts, norths, zs, deps = np.hsplit(well_dict[well], 4)
        if well.startswith('D'):  # Scale CSD signal way down visually
            loc = 1
            scale = 1.1
        elif well.startswith('B'):
            loc = 2
            scale = 1.1
        # Over each picked feature
        for i, dep in enumerate(pick_dict['depths']):
            if dep < 5.:
                # Crude skip of shallow anomalies that overrun everything
                continue
            dists = np.squeeze(np.abs(dep - deps))
            x = easts[np.argmin(dists)][0]
            y = norths[np.argmin(dists)][0]
            z = zs[np.argmin(dists)][0]
            strain = pick_dict['strains'][i]
            width = pick_dict['widths'][i]
            dist_list.append((strain, cartesian_distance(pt1=(x, y, z),
                                                         pt2=point)))
    # Unpack and plot
    strains, dists = zip(*dist_list)
    plt.scatter(dists, strains, color='blue', alpha=0.7, label='DSS picks')
    return


def plot_wells_over_time(well_data, wells,
                         date_range=(datetime(2019, 5, 19),
                                     datetime(2019, 6, 4)),
                         vrange=(-40, 40), pick_dict=None, alpha=1.,
                         plot_noise=False, frames=False):
    """
    Plot wells side-by-side with each curve over a given time slice

    :param well_data: Dict output of extract_wells
    :param wells: List of well names to plot
    :param date_range: List of [start datetime, end datetime]
    :param vrange: Xlims for all axes
    :param pick_dict: Dictionary {well name: [pick depths, ...]}
    :param alpha: Alpha value for curves (1 is opaque)
    :param frames: Save each time slice as a single frame, to be compiled into
        an animation
    :return:
    """
    # Read in data
    # Make list of times within date range
    times = []
    for well, well_dict in well_data.items():
        times.extend(list(well_dict['times']))
    times = list(set(times))
    times = [t for t in times if date_range[0] < t < date_range[1]]
    time_labs = [t.strftime('%m-%d') for t in times]
    # Initialize the figure
    if frames:
        fig = None
    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(wells) * 2, sharey=True,
                                 sharex=True, figsize=(len(wells) * 2, 8))
    # Cmap
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)
    for i, t in enumerate(times):
        pick_col = cmap(float(i) / len(times[::2]))
        if frames:
            pick_col = 'k'
        if i == 0:
            formater = True
        elif frames:
            formater = True
        else:
            formater = False
        plot_well_timeslices(well_data, wells, date=t, vrange=vrange,
                             pick_dict=pick_dict, fig=fig, pick_col=pick_col,
                             alpha=alpha, plot_noise=plot_noise,
                             ref_date=date_range[0], formater=formater,
                             frame=frames)
    if not frames:
        tick_indices = np.linspace(0, len(times) - 1, 8).astype(int)
        cbar = fig.colorbar(
            ScalarMappable(norm=Normalize(0, len(times)), cmap=cmap),
                            ax=axes, fraction=0.04, location='bottom',
                            ticks=tick_indices)
        cbar.ax.set_xticklabels(np.array(time_labs)[tick_indices])
        cbar.ax.set_xlabel('{}'.format(times[-1].year), fontsize=16)
    return fig


def plot_well_timeslices(well_data, wells, ref_date, date, remove_ref=True,
                         vrange=(-40, 40), pick_dict=None, fig=None,
                         pick_col=None, alpha=None, plot_noise=False,
                         formater=True, frame=False):
    """
    Plot a time slice up and down each specified well

    :param path: Well_data dict from extract_wells
    :param wells: List of well names to plot
    :param ref_date: Reference date to plot
    :param date: datetime to plot lines for
    :param remove_ref: Subtract first time sample from whole array?
    :param vrange: Xlims for all axes
    :param pick_dict: Dictionary {well name: [pick depths, ...]}
    :param fig: Figure to plot into
    :param pick_col: Color of the curve
    :param alpha: Alpha of curve
    :param plot_noise: Plot noise estimate or not
    :param formater: On the first pass, do the formatting, otherwise skip it
    :param frame: Save as frame of animation?

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
    for well in wells:
        times = well_data[well]['times']
        ax1, ax2 = axes[i:i + 2]
        data = well_data[well]['data']
        depth = well_data[well]['depth']
        noise = well_data[well]['noise']
        down_d, up_d = np.array_split(depth - depth[0], 2)
        if down_d.shape[0] != up_d.shape[0]:
            # prepend last element of down to up if unequal lengths by 1
            up_d = np.insert(up_d, 0, down_d[-1])
        # Remove reference first
        ref_ind = np.argmin(np.abs(times - ref_date))
        # TODO Do we remove the reference time signal or not??
        if remove_ref:
            data = data - data[:, ref_ind, np.newaxis]
        reference_vect = data[:, ref_ind]
        ref_time = times[ref_ind]
        # Also reference vector
        down_ref, up_ref = np.array_split(reference_vect, 2)
        # Again account for unequal down and up arrays
        if down_ref.shape[0] != up_ref.shape[0]:
            up_ref = np.insert(up_ref, 0, down_ref[-1])
        up_d_flip = up_d[-1] - up_d
        ax1.plot(down_ref, down_d, color='k', linestyle=':',
                 label=ref_time.date(), lw=.5)
        ax2.plot(up_ref, up_d_flip, color='k', linestyle=':', lw=.5)
        if plot_noise and formater:
            # If single noise measure
            if noise[1] is None:
                noise_mean = reference_vect
                noise_mad = np.zeros(noise_mean.shape[0]) + noise[0]
            # If noise measured per channel along fiber
            else:
                noise_mean = noise[0]
                noise_mad = noise[1]
            down_noise, up_noise = np.array_split(noise_mean, 2)
            down_noise_mad, up_noise_mad = np.array_split(noise_mad, 2)
            # Again account for unequal down and up arrays
            if down_noise.shape[0] != up_noise.shape[0]:
                up_noise = np.insert(up_noise, 0, down_noise[-1])
                up_noise_mad = np.insert(up_noise_mad, 0, down_noise_mad[-1])
            # Fill between noise bounds
            ax1.fill_betweenx(y=down_d, x1=down_noise - down_noise_mad,
                              x2=down_noise + down_noise_mad,
                              alpha=0.2, color='k')
            ax2.fill_betweenx(y=up_d_flip, x1=up_noise - up_noise_mad,
                              x2=up_noise + up_noise_mad,
                              alpha=0.2, color='k')
        # Get column corresponding to xdata time
        dts = np.abs(times - date)
        time_int = np.argmin(dts)
        # Grab along-fiber vector
        fiber_vect = data[:, time_int]
        # If frame for animation, keep last 10 time samples too
        if frame:
            old_vects = data[:, time_int - 20:time_int]
            old_down, old_up = np.array_split(old_vects, 2)
            if old_down.shape[0] != old_up.shape[0]:
                old_up = np.append(old_up, old_down[-1]).reshape(old_down.shape)
            ax1.plot(old_down, down_d[:, None], color='grey', alpha=alpha,
                     linewidth=0.5)
            ax2.plot(old_up, up_d_flip[:, None], color='grey', alpha=alpha,
                     linewidth=0.5)
        # Plot two traces for downgoing and upgoing trace at user-
        # picked time
        down_vect, up_vect = np.array_split(fiber_vect, 2)
        # Again account for unequal down and up arrays
        if down_vect.shape[0] != up_vect.shape[0]:
            up_vect = np.insert(up_vect, 0, down_vect[-1])
        if frame:
            lw = 2.
        else:
            lw = 0.5
        ax1.plot(down_vect, down_d, color=pick_col, label=date.date(),
                 alpha=alpha, linewidth=lw)
        ax2.plot(up_vect, up_d_flip, color=pick_col, alpha=alpha, linewidth=lw)
        # If picks provided, plot them
        if pick_dict and well in pick_dict and formater:
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
        if formater:
            if i == (len(wells) * 2) - 2 and len(times) < 4:
                ax1.legend(fontsize=16, bbox_to_anchor=(0.15, 0.25),
                           framealpha=1.).set_zorder(103)
            elif i == 0:
                ax1.set_ylabel('Depth [m]', fontsize=18)
            # Formatting
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
        # Always increment, obviously
        i += 2
    if formater:
        if frame:
            lab_y = 0.04
        else:
            lab_y = 0.19
        label = r'$\mu\varepsilon$'
        fig.text(0.5, lab_y, label, ha='center', fontsize=20)  # Commmon xlabel
        ax1.invert_yaxis()
        # Now loop axes and place well titles
        # Crappy calculation for title spacing
        start = 1 / len(wells)
        title_xs = np.linspace(1 / len(wells), 1 - (1 / len(wells)),
                               len(wells))
        title_ys = np.ones(len(wells)) * 0.92
        # Common title for well subplots
        for i, pax in enumerate(axes[::2]):
            fig.text(x=title_xs[i], y=title_ys[i], s=wells[i], ha='center',
                     fontsize=22)
    if frame:
        # Put the date on the plot if animating
        fig.text(0.7, 0.2, date, ha="center", va="center", fontsize=20,
                 bbox=dict(boxstyle="round",
                           ec='k', fc='white'))
        fig.savefig('frame_{}.png'.format(date))
        plt.close('all')
    return fig


def plot_DSS(well_data, well='all', derivative=False, colorbar_type='light',
             inset_channels=True, simfip=False, pick_mode='auto', thresh=1.,
             date_range=(datetime(2019, 5, 19), datetime(2019, 6, 4)),
             denoise_method=None, vrange=(-60, 60), title=None,
             tv_picks=None):
    """
    Plot a colormap of DSS data

    :param path: Path to raw data file
    :param well: Which well to plot
    :param inset_channels: Bool for picking channels to plot in separate axes
    :param simfip: Give path to data file if simfip data over same timespan
    :param pick_mode: "manual" or "auto". User will still need to manually
        pick the time of the signal on which the anomalies will be picked.
    :param thresh: MAD multiplier that serves as the threshold for auto picking
    :param date_range: [start date, end date]
    :param denoise_method: String stipulating the method in denoise() to use
    :param vrange: Colorbar range (in measurand unit)
    :param title: Title of plot
    :param tv_picks: Path to excel file with optical televiewer picks

    :return:
    """
    fig = plt.figure(constrained_layout=False, figsize=(14, 12))
    if inset_channels and simfip:
        gs = GridSpec(ncols=14, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:3, 7:-1])
        axes1b = fig.add_subplot(gs[3:6, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[6:9, 7:-1], sharex=axes1)
        axes3 = fig.add_subplot(gs[9:, 7:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        cax = fig.add_subplot(gs[:6, -1])
        df = read_excavation(simfip)
    elif inset_channels:
        gs = GridSpec(ncols=12, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:4, 7:-1])
        axes1b = fig.add_subplot(gs[4:8, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[8:, 7:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        cax = fig.add_subplot(gs[:8, -1])
    # Get just the channels from the well in question
    times = well_data[well]['times']
    data = well_data[well]['data']
    depth_vect = well_data[well]['depth']
    if well_data[well]['noise'][1] is None:
        noise = well_data[well]['noise'][0]
    else:
        noise = well_data[well]['noise']
    mode = well_data[well]['mode']
    type = well_data[well]['type']
    if date_range:
        indices = np.where((date_range[0] < times) & (times < date_range[1]))
        times = times[indices]
        data = data[:, indices]
        data = np.squeeze(data)
    mpl_times = mdates.date2num(times)
    # TODO still don't know the best way to deal with relative values?
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
    down_d, up_d = np.array_split(depth_vect - depth_vect[0], 2)
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
        if well.startswith('D'):
            exp = 'BCS'
        elif well.startswith('B'):
            exp = 'BFS'
        fig.suptitle('DSS {}-{}'.format(exp, well), fontsize=20)
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
        axes4.fill_betweenx(y=down_d, x1=down_ref - (noise * thresh),
                            x2=down_ref + (noise * thresh),
                            alpha=0.2, color='k')
        axes5.fill_betweenx(y=up_d_flip, x1=up_ref - (noise * thresh),
                            x2=up_ref + (noise * thresh),
                            alpha=0.2, color='k')
        # Plot fracture density too TODO Enable other logs here too
        if tv_picks:
            try:
                frac_dict = calculate_frac_density(tv_picks, create_FSB_boreholes())
            except KeyError:
                # Try core fracture counts instead
                frac_dict = read_frac_cores(tv_picks, well)
            for frac_type, dens in frac_dict.items():
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
                         up_d, down_d, pick_mode, noise, thresh):
                self.figure = figure
                self.cmap = cmap
                self.cat_cmap = cat_cmap
                self.pick_mode = pick_mode
                self.thresh = thresh
                self.noise = noise
                self.data = data
                self.up_d = up_d
                self.down_d = down_d
                self.depth = depth - depth[0]
                self.xlim = self.figure.axes[0].get_xlim()
                self.times = times
                if self.pick_mode == 'manual':
                    self.pick_dict = {well: []}
                elif self.pick_mode == 'auto':
                    self.pick_dict = {well: {}}
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
                # depth = self.depth[chan]
                depth = event.ydata
                # Grab along-fiber vector
                fiber_vect = self.data[:, time_int]
                self.figure.axes[2].axvline(x=event.xdata, color='k',
                                            linestyle='--', alpha=0.5)
                if simfip:
                    self.figure.axes[3].axvline(x=event.xdata, color='k',
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
                    try:
                        self.figure.axes[-3].axhline(fault_depths[well][0],
                                                     linestyle='--',
                                                     linewidth=1., color='k')
                        self.figure.axes[-3].axhline(fault_depths[well][1],
                                                     linestyle='--',
                                                     linewidth=1., color='k')
                        self.figure.axes[-4].axhline(fault_depths[well][0],
                                                     linestyle='--',
                                                     linewidth=1., color='k')
                        self.figure.axes[-4].axhline(fault_depths[well][1],
                                                     linestyle='--',
                                                     linewidth=1., color='k')
                    except IndexError as e:
                        print(e)
                self.figure.axes[-4].legend(
                    loc=2, fontsize=12, bbox_to_anchor=(0.5, 1.13),
                    framealpha=1.).set_zorder(110)
                if pick_mode == 'manual':
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
                        label='Depth {:0.2f}'.format(depth))
                else:
                    if self.noise[1] is None:
                        noise_mean = 0.
                        noise_mad = self.noise[0]
                    else:
                        noise_mean = self.noise[0]
                        noise_mad = self.noise[1]
                    peak_inds, peak_dict = pick_anomalies(
                        fiber_vect, noise_mean=noise_mean,
                        noise_mad=noise_mad, thresh=self.thresh)
                    # Populate pick_dict
                    samp_int = self.depth[1] - self.depth[0]
                    self.pick_dict[self.well]['strains'] = fiber_vect[peak_inds]
                    self.pick_dict[self.well]['heights'] = peak_dict['peak_heights']
                    self.pick_dict[self.well]['widths'] = peak_dict['widths'] * samp_int
                    self.pick_dict[self.well]['depths'] = []
                    # Now plot all picks at peak index with width calculated
                    # from find_widths
                    for pk in zip(peak_inds, peak_dict['widths']):
                        pick_col = next(self.cat_cmap)
                        half_width = (pk[1] * samp_int) / 2.
                        trace = self.data[pk[0], :]
                        if self.depth[pk[0]] > down_d[-1]: # Upgoing peak
                            # Precalculate axes depth
                            rel_dep = (self.depth[-1] -
                                       self.depth[pk[0] + pick_adjust])
                            if rel_dep < 5:  # Ignore shallow picks for now
                                continue
                            self.pick_dict[self.well]['depths'].append(rel_dep)
                            self.figure.axes[-3].fill_between(
                                x=np.array([-500, 500]),
                                y1=rel_dep - half_width,
                                y2=rel_dep + half_width,
                                alpha=0.5, color=pick_col)
                            arr_ax = self.figure.axes[1]
                            trans = arr_ax.get_yaxis_transform()
                        else: # Downgoing
                            rel_dep = self.depth[pk[0]]
                            if rel_dep < 5:  # Ignore shallow picks for now
                                continue
                            self.pick_dict[self.well]['depths'].append(rel_dep)
                            self.figure.axes[-4].fill_between(
                                x=np.array([-500, 500]),
                                y1=rel_dep - half_width,
                                y2=rel_dep + half_width,
                                alpha=0.5, color=pick_col)
                            arr_ax = self.figure.axes[0]
                            trans = arr_ax.get_yaxis_transform()
                        # Arrow patches for picks
                        arrow = mpatches.FancyArrowPatch((1.05, rel_dep),
                                                         (0.95, rel_dep),
                                                         mutation_scale=20,
                                                         transform=trans,
                                                         facecolor=pick_col,
                                                         clip_on=False,
                                                         zorder=103)
                        arr_ax.add_patch(arrow)
                        self.figure.axes[2].plot(
                            self.times, trace, color=pick_col,
                            label='Depth {:0.2f}'.format(rel_dep))
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
                               cat_cmap, up_d, down_d, pick_mode,
                               noise=well_data[well]['noise'], thresh=thresh)
        plt.show()
    return plotter.pick_dict


def plot_potentiometer(times, depths, data,
                       date_range=(datetime(2019, 5, 19), datetime(2019, 6, 4)),
                       vrange=(-400, 400), simfip=False):
    """
    Plot CSD potentiometer string as contoured image and individual traces

    :param data: data output from read_potentiometer
    :param depths: Depths of each row of data
    :param times: Times of each column of data

    :return: matplotlib.Figure
    """
    # Cut data to daterange
    indices = np.where((date_range[0] < times) & (times < date_range[1]))
    times = times[indices]
    data = data[indices, :]
    data = np.squeeze(data)
    if simfip:
        fig = plt.figure(constrained_layout=False, figsize=(10, 12))
        gs = GridSpec(ncols=8, nrows=10, figure=fig)
        axes1 = fig.add_subplot(gs[:4, :-1])
        axes2 = fig.add_subplot(gs[4:7, :-1], sharex=axes1)
        axes3 = fig.add_subplot(gs[7:, :-1], sharex=axes1)
        cax = fig.add_subplot(gs[:7, -1])
        df = read_excavation(simfip)
    else:
        fig = plt.figure(constrained_layout=False, figsize=(10, 12))
        gs = GridSpec(ncols=8, nrows=8, figure=fig)
        axes1 = fig.add_subplot(gs[:4, :-1])
        axes2 = fig.add_subplot(gs[4:8, :-1], sharex=axes1)
        cax = fig.add_subplot(gs[:4, -1])
    cmap = ListedColormap(sns.color_palette('RdBu_r', 21).as_hex())
    # Flip sign of data to conform to extension == positive strain
    im = axes1.contourf(times, depths, data.T * -1., cmap=cmap,
                        vmin=vrange[0], vmax=vrange[1])
    axes1.invert_yaxis()
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel('Microstrain', fontsize=16)
    for i, depth in enumerate(depths):
        # Only plot the lower instruments
        if depth > 18.:
            # Again flip data so + strain is extensional
            axes2.plot(times, data[:, i] * -1, label=depth)
            axes2.invert_yaxis()  # Negative strain convention
            axes2.legend(title='Depth')
    # Formatting
    date_formatter = mdates.DateFormatter('%b-%d %H')
    axes1.set_ylabel('Depth [m]', fontsize=16)
    axes2.set_ylabel(r'$\mu\varepsilon$', fontsize=16)
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
        plt.setp(axes2.get_xticklabels(), visible=False)
    else:
        axes2.set_ylabel(r'Displacement [$\mu$m]', fontsize=16)
        axes2.xaxis_date()
        axes2.xaxis.set_major_formatter(date_formatter)
        plt.setp(axes2.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.setp(axes1.get_xticklabels(), visible=False)
    plt.tight_layout()
    return fig
