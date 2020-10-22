#!/usr/bin/python
"""
Functions for processing and plotting DSS data
"""

import os

import numpy as np
import pytz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from glob import glob
from copy import deepcopy
from obspy import Stream, Trace
from pytz import timezone
from eqcorrscan.core.match_filter import normxcorr2
from pandas.errors import ParserError
from scipy.integrate import trapz
from scipy.interpolate import griddata, interp2d
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import detrend, welch, find_peaks, zpk2sos, sosfilt, iirfilter
from scipy.stats import median_absolute_deviation
from datetime import datetime, timedelta
from itertools import cycle
from matplotlib.dates import num2date, date2num, DateFormatter
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection

# Local imports
from lbnl.coordinates import cartesian_distance
from lbnl.boreholes import (parse_surf_boreholes, create_FSB_boreholes,
                            calculate_frac_density, read_frac_cores,
                            depth_to_xyz, distance_to_borehole)
from lbnl.DTS import read_struct
from lbnl.simfip import (read_excavation, plot_displacement_components,
                         read_collab, rotate_fsb_to_fault,
                         rotate_fsb_to_borehole)
from lbnl.hydraulic_data import (read_collab_hydro, read_csd_hydro,
                                 plot_csd_hydro)


######### SURF CHANNEL MAPPING ############
# Foot markings (actual feet...)
omnisens = 5360.36
# Jonathan mapping from scripts (Source ??)
chan_map_feet = {'OT': (6287., 291., 356.), 'OB': (411., 470.5, 530.),
                 'PST': (695., 737.5, 780.), 'PSB': (827., 886.5, 946.),
                 'PDT': (1179., 1238., 1297.), 'PDB': (995., 1054.5, 1114.)}

# Jonathan mapping from scripts (Source ??)
chan_map_surf = {#'OT': (226., 291., 356.), ORIGINAL
                 # 'OT': (226., 286., 346.),
                 'OT': (224.5, 286., 347.5),
                 'OB': (411., 470.5, 530.),
                 'PST': (695., 737.5, 780.), 'PSB': (827., 886.5, 946.),
                 'PDT': (1179., 1238., 1297.), 'PDB': (995., 1054.5, 1114.)}

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
# Loop 5, 6
chan_map_excav_56 = {'D5': 187.535,
                     'D6': 97.145}
# Loop 3, 4
chan_map_excav_34 = {'D3': 76.61,
                     'D4': 167.22}
# Loop 5, 6
chan_map_co2_5612 = {'D5': 96.42,
                     'D6': 186.74,
                     'D1': 354.577,
                     'D2': 272.39}
# Loop 3, 4
chan_map_co2_34 = {'D3': 79.62,
                   'D4': 170.43}

# 0.5 m resolution results in -0.25 m shift in measurements
chan_map_co2_34_pt1 = {'D3': 79.37,
                       'D4': 170.18}
chan_map_co2_5612_pt1 = {'D5': 96.17,
                         'D6': 186.49,
                         'D1': 354.327,
                         'D2': 272.14}

chan_map_august = {'D3': 108.,
                   'D4': 199.,
                   'D5': 391.63,
                   'D6': 482.,
                   'D2': 568.23,
                   'D1': 649.16}

# Anchor point mapping (depth in hole)
D1_anchor_map = {'seg3': (12.97, 15.37),
                 'seg2': (15.37, 17.17),
                 'seg1': (17.17, 19.37),
                 }
D2_anchor_map = {'seg5': (10.27, 11.67),
                 'seg4': (11.67, 12.72),
                 'seg3': (12.72, 13.52),
                 'seg2': (13.52, 14.32),
                 'seg1': (14.32, 15.27),
                 }
######### DRILLING FAULT DEPTH ############
# Dict of drilled depths
fiber_depths = {'D1': 21.26, 'D2': 17.1, 'D3': 31.65, 'D4': 36.9, 'D5': 31.79,
                'D6': 36.65, 'D7': 29.7}

fault_depths = {'D1': (14.34, 19.63), 'D2': (11.04, 16.39), 'D3': (17.98, 20.58),
                'D4': (27.05, 28.44), 'D5': (19.74, 22.66), 'D6': (28.5, 31.4),
                'D7': (22.46, 25.54)}

resin_depths = {'D3': (2.5, 3.), 'D4': (9., 10.), 'D5': (17., 18.),
                'D6': (12., 14.)}

potentiometer_depths = {'1': (2., 6.5, 11.), '2': (11., 15., 19.),
                        '3': (19., 19.25, 19.5), '4': (19.5, 19.75, 20.),
                        '5': (20., 20.25, 20.5), '6': (20.5, 20.75, 21.),
                        '7': (21., 21.25, 21.5), '8': (21.5, 21.75, 22.),
                        '9': (22., 22.25, 22.5), '10': (22.5, 22.75, 23.),
                        '11': (23., 23.25, 23.5), '12': (23.5, 23.75, 24.)}

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
                                  'FSB': chan_map_fsb},
                'co2_injection_pt1': {'CSD3': chan_map_co2_34_pt1,
                                      'CSD5': chan_map_co2_5612_pt1,
                                      'FSB': chan_map_fsb},
                'august_pulse': {'CSD1': chan_map_august},
                'surf': chan_map_surf}

well_fiber_map = {'D1': 'CSD5', 'D2': 'CSD5', 'D3': 'CSD3', 'D4': 'CSD3',
                  'D5': 'CSD5', 'D6': 'CSD5', 'B3': 'FSB', 'B4': 'FSB',
                  'B5': 'FSB', 'B6': 'FSB', 'B7': 'FSB'}

# Custom color palette similar to wellcad convention
frac_cols = {'All fractures': 'black',
             'open/undif. fracture': 'blue',
             'sealed fracture / vein': 'lightblue',
             'foliation / bedding': 'red',
             'induced fracture': 'magenta',
             'sedimentary structures/color changes undif.': 'green',
             'uncertain type': 'orange',
             'lithology change': 'yellow'}

csd_well_colors = {'D1': 'dodgerblue', 'D2': 'lightseagreen',
                   'D3': 'firebrick', 'D4': 'firebrick',
                   'D5': 'blueviolet', 'D6': 'firebrick',
                   'D7': 'k'}

def date_generator(start_date, end_date, frequency='day'):
    # Generator for date looping
    from datetime import timedelta
    if frequency == 'day':
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + timedelta(n)
    elif frequency == 'hour':
        for n in range((int((end_date - start_date).days * 86400) // 3600) + 1):
            yield start_date + timedelta(hours=n)
    else:
        print('Only day and hour frequency supported')
        return


def read_ascii(path, header=42, encoding='iso-8859-1'):
    """Read in a raw DSS file (flipped about axis 1 for left-to-right time"""
    return np.flip(np.loadtxt(path, skiprows=header, encoding=encoding), 1)


def read_ascii_directory(root_path, header, location):
    """Read single-measurement files into one data matrix"""
    asciis = glob('{}/**/*.txt'.format(root_path), recursive=True)
    asciis.sort()
    for i, ascii in enumerate(asciis):
        if i == 0:
            dd = read_ascii(ascii, header=header)
            times = read_times(ascii, location=location)
            depths = dd[:, -1]
            data = dd[:, 0]
        else:
            data = np.vstack([data, read_ascii(ascii, header=header)[:, 0]])
            times = np.vstack([times, read_times(ascii, location=location)])
    return data.T, depths, times.squeeze()


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


def read_potentiometer_co2(root_path):
    csvs = glob('{}/*.csv'.format(root_path))
    csvs.sort()
    df_dict = {}
    for csv in csvs:
        name = os.path.basename(csv).rstrip('.csv').split('-')[-1]
        name = '{:d}'.format(int(name))
        parser = lambda x: datetime.strptime(x, '%d.%m.%Y %H:%M:%S')
        df_temp = pd.read_csv(csv, usecols=[0, 1],
                              skiprows=list(np.arange(8)), header=0,
                              parse_dates=[0], date_parser=parser)
        df_temp = df_temp.set_index('dd.MM.yyyy  hh:mm:ss')
        df_temp.rename(columns={'Measurement': name}, inplace=True)
        # Length of potentiometer element in meters
        scale = potentiometer_depths[name][2] - potentiometer_depths[name][0]
        # Measurement is mm, so divide by length of element (in mm), then
        # convert to microstrain
        df_dict[name] = (df_temp - df_temp.iloc[0]) / (scale * 0.001)
    return df_dict


def read_potentiometer_raw(root_path):
    """
    Yet another parsing function for the raw data format **.DAT
    :return:
    """
    frames = []
    for f in glob('{}/*.DAT'.format(root_path)):
        parser = lambda x, y: datetime.strptime(x + y, '%d.%m.%Y%H:%M:%S')
        df_temp = pd.read_csv(f, encoding='ISO-8859-1', delimiter='\s+',
                              header=1, skiprows=[2, 3, 4],
                              parse_dates={'datetime': [0, 1]},
                              date_parser=parser)
        df_temp = df_temp.set_index('datetime')
        df_temp = df_temp[['D_B5_EXT{:02d}'.format(d + 1) for d in range(12)]]
        mapper = lambda x: x.split('_')[-1]
        df_temp = df_temp.rename(mapper=mapper, axis=1)
        # Scale the measures to pot length and convert to microns
        for pot_nm, endpts in potentiometer_depths.items():
            scale = endpts[2] - endpts[0]
            name = 'EXT{:02d}'.format(int(pot_nm))
            df_temp[name] = df_temp[name] / (scale * 0.001)
        frames.append(df_temp)
    df_pot = pd.concat(frames)
    df_pot = df_pot.sort_index()
    df_pot = df_pot - df_pot.iloc[0]
    return df_pot


def datetime_parse(t, fmt, location):
    # Parse the date format of the DSS headers; return as UTC for fsb
    if location == 'fsb':
        tz = timezone('Etc/GMT+2')
        tz.localize(datetime.strptime(t, fmt)).astimezone()
    return datetime.strptime(t, fmt)


def read_times(path, encoding='iso-8859-1', header=10,
               time_fmt='%Y/%m/%d %H:%M:%S', location='fsb'):
    """Read timestamps from ascii header"""
    # Create appropriate timezone object
    strings = np.genfromtxt(path, skip_header=header, max_rows=1,
                            encoding=encoding, dtype=None, delimiter='\t')
    if header == 1:  # Potentiometer file
        return np.array([datetime_parse(t, time_fmt, location)
                         for t in strings[:-1]])[::-1]
    elif header == 10 and location == 'fsb':  # Omnisens output
        return np.array([datetime_parse(t, time_fmt, location)
                         for t in strings[1:-1]])[::-1]
    elif header == 10 and location == 'surf':
        return np.array([datetime_parse(t, time_fmt, location)
                         for t in strings[1:]])[::-1]
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
    try:
        return mode, type
    except UnboundLocalError:
        return 'Absolute', 'Brillouin Frequency'


def integrate_anchors(data, depth, well):
    """
    Helper to replace the strain values between anchors with the integral over
    the anchor span
    """
    if well == 'D1':
        chan_map = {key: {'down': (np.argmin(np.abs(depth - tup[0])),
                                   np.argmin(np.abs(depth - tup[1]))),
                          'up': (np.argmin(np.abs(depth -
                                                  (depth[-1] - tup[0]))),
                                 np.argmin(np.abs(depth -
                                                  (depth[-1] - tup[1]))))}
                    for key, tup in D1_anchor_map.items()}
    elif well == 'D2':
        chan_map = {key: {'down': (np.argmin(np.abs(depth - tup[0])),
                                   np.argmin(np.abs(depth - tup[1]))),
                          'up': (np.argmin(np.abs(depth -
                                                  (depth[-1] - tup[0]))),
                                 np.argmin(np.abs(depth -
                                                  (depth[-1] - tup[1]))))}
                    for key, tup in D2_anchor_map.items()}
    for seg, chans in chan_map.items():
        up_chans = chans['up']
        down_chans = chans['down']
        if up_chans[0] > up_chans[1]:
            up_chans = (up_chans[1], up_chans[0])
        intg_up = trapz(data[up_chans[0]:up_chans[1] + 1, :], axis=0)
        intg_down = trapz(data[down_chans[0]:down_chans[1] + 1, :], axis=0)
        # Scale to channel spacing
        data[up_chans[0]:up_chans[1] + 1, :] = intg_up * np.abs(depth[1] - depth[0])
        data[down_chans[0]:down_chans[1] + 1, :] = intg_down * np.abs(depth[1] - depth[0])
    return data


def integrate_depth_interval(well_data, depths, well, leg, dates=None):
    """
    Return timeseries of channels integrated over a depth range

    :param data: data array for well loop
    :param depths: [shallow, deep] range to integrate over
    :param well: Well name
    :param leg: Down or up
    :return:
    """
    data = well_data[well]['data']
    depth = well_data[well]['depth']
    depth -= depth[0]
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
    else:
        d_inds = np.where((dates[0] <= well_data[well]['times']) &
                           (dates[1] > well_data[well]['times']))
        int_data = np.squeeze(data[chans[0]:chans[1] + 1, d_inds])
        # Relative to first sample
        int_data = int_data - int_data[:, 0, np.newaxis]
        integral = trapz(int_data, axis=0)
        integral = np.squeeze(integral)
    return integral


def scale_to_gain(data, gain, offset_samps):
    """Scale measure relative to starting gain"""
    gain /= gain[:, 0:offset_samps, np.newaxis].mean(axis=1)
    return data / gain


def extract_wells(root, measure=None, mapping=None, wells=None, fibers=None,
                  location=None, noise_method='madjdabadi', convert_freq=False,
                  DTS=None, DTS_interp='linear', gain_thresh=0.015,
                  correct_gain=False, debug=0):
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
    :param fibers: Optionally specify individual fiber loops (FSB, CSD3 or CSD5)
    :param noise_method: 'majdabadi' or 'by_channel' to estimate noise.
        'majdabadi' returns scalar, 'by_channel' an array
    :param convert_freq:
    :param DTS: Path to DTS data
    :param DTS_interp: Method of interpolation for DTS to DSS grid
    :param gain_thresh: Threshold for removal of changes from bulk gain shifts
        Unit is percent.
    :param correct_gain: bool to apply correction at gain jumps
    :param debug: Flag for plotting

    :returns: dict {well name: {'data':, 'depth':, 'noise':, ...}
    """
    if not fibers and not wells:
        print('Must specify either fibers or wells')
        return
    if not location:
        print('Specify location: surf or fsb')
        return
    well_data = {}
    fiber_data = {}
    print('Reading data')
    if location == 'fsb':
        chan_map = {}
        data_files = glob('{}/*{}.txt'.format(root, measure))
        gain_files = glob('{}/*Absolute_Gain.txt'.format(root))
        for f in data_files:
            if f.split('/')[-1].startswith('FSB-SMF-1'):
                # Skip fiber 1
                continue
            file_root = f.split('/')[-1].split('-')[0]
            fiber_data[file_root] = {}
            if fibers:
                if file_root not in fibers:
                    print('{} not in fibers'.format(file_root))
                    continue
            data = read_ascii(f)
            times = read_times(f)
            try:  # Grab absolute gain file and try to correct
                gain_file = [g for g in gain_files if file_root in g][0]
                gain = read_ascii(gain_file)
                gain = gain[:, :-1]
                fiber_data[file_root]['gain'] = gain
            except IndexError:
                print('No gain file. Need this first.')
                continue
            # Take first column as the length along the fiber and remove
            depth = data[:, -1]
            data = data[:, :-1]
            fiber_data[file_root]['data'] = data
            fiber_data[file_root]['depth'] = depth
            fiber_data[file_root]['times'] = times
            chan_map.update(mapping_dict[mapping][file_root])
            mode, type_m = read_metadata(f)
    elif location == 'surf':
        if DTS:
            # Read in DTS temps to remove response
            print('Reading DTS')
            temp_dict = read_struct(DTS)
        print('Reading DSS')
        data, depth, times = read_ascii_directory(root, header=34,
                                                  location=location)
        chan_map = mapping_dict[location]
        print('Reading metadata')
        mode, type_m = read_metadata(glob('{}/**/*bpr.txt'.format(root),
                                          recursive=True)[0])
        fiber_data['surf'] = {}
        fiber_data['surf']['data'] = data
        fiber_data['surf']['depth'] = depth
        fiber_data['surf']['times'] = times
    else:
        print('Provide valid location')
        return
    print('Realigning')
    # First realign
    for fib, f_dict in fiber_data.items():
        if not f_dict:
            print('Fiber {} returned empty dictionary'.format(fib))
            continue
        f_dict['data'] = madjdabadi_realign(f_dict['data'])
    if convert_freq and type_m.endswith('Frequency'):
        print('Converting from freq to strain')
        if mode == 'Absolute':
            # First convert to delta Freq
            for fib, f_dict in fiber_data.items():
                f_dict['data'] = f_dict['data'] - f_dict['data'][:, 0, np.newaxis]
            mode = 'Relative'  # overwrite mode
            type_m = 'Strain'
    print('Calculating channel mapping')
    if wells:
        for well in wells:
            if well not in chan_map:
                print('{} not in mapping'.format(well))
                continue
            if location == 'fsb':
                try:
                    depth = fiber_data[well_fiber_map[well]]['depth']
                    data = fiber_data[well_fiber_map[well]]['data']
                    times = fiber_data[well_fiber_map[well]]['times']
                    gain = fiber_data[well_fiber_map[well]]['gain']
                except KeyError as e:
                    depth = fiber_data['CSD1']['depth']
                    data = fiber_data['CSD1']['data']
                    times = fiber_data['CSD1']['times']
                    gain = fiber_data['CSD1']['gain']
            elif location == 'surf':
                depth = fiber_data['surf']['depth']
                data = fiber_data['surf']['data']
                times = fiber_data['surf']['times']
            else:
                print('{} not a location'.format(location))
                return
            if type(chan_map[well]) == float:
                start_chan = np.abs(depth - (chan_map[well] -
                                             fiber_depths[well]))
                end_chan = np.abs(depth - (chan_map[well] +
                                           fiber_depths[well]))
            else:
                start_chan = np.abs(depth - chan_map[well][0])
                end_chan = np.abs(depth - chan_map[well][-1])
            # Find the closest integer channel to meter mapping
            data_tmp = data[np.argmin(start_chan):np.argmin(end_chan), :]
            depth_tmp = depth[np.argmin(start_chan):np.argmin(end_chan)]
            gain_tmp = gain[np.argmin(start_chan):np.argmin(end_chan), :]
            if location == 'surf' and well == 'OT':
                depth_tmp *= 0.9642  # "Stretch factor"
            noise = estimate_noise(data_tmp, method=noise_method)
            well_data[well] = {'times': times, 'mode': mode,
                               'type': type_m}
            if DTS:
                print('Removing DTS-induced strain')
                # Remove temperature signal from strain
                full_loop_T = np.concatenate((temp_dict[well]['temp'],
                                              np.flip(temp_dict[well]['temp'],
                                                      axis=0)),
                                             axis=0)
                full_loop_d = np.concatenate((temp_dict[well]['depth'],
                                              temp_dict[well]['depth'] +
                                              temp_dict[well]['depth'][-1]))
                # DTS grid
                xt, yt = np.meshgrid(full_loop_d,
                                     date2num(temp_dict[well]['times']),
                                     indexing='ij')
                # DSS grid
                xd, yd = np.meshgrid(depth_tmp - depth_tmp[0], date2num(times),
                                     indexing='ij')
                temp_interp = griddata(
                    np.array([xt.flatten(), yt.flatten()]).T,
                    full_loop_T.flatten(),
                    np.array([xd.flatten(), yd.flatten()]).T,
                    method=DTS_interp, rescale=True).reshape(xd.shape)
                # We want delta T since start (relative to same time as DSS)
                temp_interp = temp_interp - temp_interp[:, 0, np.newaxis]
                well_data[well].update(
                    {'uncorrected_freq': data_tmp.copy(),
                     'uncorrected_strain': data_tmp.copy() * 5790.,
                     'interp_temp': temp_interp,
                     'temp-induced_freq': temp_interp.copy() * 0.00164,
                     'temp-induced_strain': temp_interp.copy() * 0.00164 * 5790.,
                     'raw_temp': temp_dict[well]['temp']})
                data_tmp = data_tmp - (temp_interp * 0.00164)
                well_data[well].update({'corrected_freq': data_tmp.copy()})
            if convert_freq:
                # Use conversion factor 0.579 GHz shift per 1% strain
                # For microstrain, factor is 5790
                data_tmp *= 5790.
            well_data[well].update({'data': data_tmp, 'depth': depth_tmp,
                                    'noise': noise, 'gain': gain_tmp})
    elif fibers:
        for fiber in fibers:
            if location != 'surf':
                for fib, f_dict in fiber_data.items():
                    if fiber == fib:
                        depth = fiber_data[fiber]['depth']
                        data = fiber_data[fiber]['data']
                        times = fiber_data[fiber]['times']
                        well_data[fiber] = {'times': times, 'mode': mode,
                                            'type': type}
                        data_tmp = data[0:-1, :]
                        depth_tmp = depth[0:-1]
                        # Get median absolute deviation averaged across all channels
                        noise = estimate_noise(data_tmp, method=noise_method)
                        well_data[fiber].update({'data': data_tmp,
                                                 'depth': depth_tmp,
                                                 'noise': noise})
            else:  # Case of surf
                data = fiber_data['surf']['data']
                depth = fiber_data['surf']['depth']
                times = fiber_data['surf']['times']
                well_data[fiber] = {'times': times, 'mode': mode,
                                    'type': type}
                data_tmp = data[0:-1, :]
                depth_tmp = depth[0:-1]
                # Get median absolute deviation averaged across all channels
                noise = estimate_noise(data_tmp, method=noise_method)
                well_data[fiber].update({'data': data_tmp, 'depth': depth_tmp,
                                         'noise': noise})
    # Calculate deviation between legs in each well
    calculate_leg_deviation(well_data)
    # Gain correction
    if correct_gain:
        try:
            gain_correction(well_data, gain_thresh, debug=debug)
        except KeyError as e:
            print(e)
            pass
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
    elif method == 'by_channel':
        return np.mean(data, axis=1), median_absolute_deviation(data, axis=1)
    else:
        print('Invalid method for denoise')
        return


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


def filter(data, freqmin, freqmax, df, corners=4, zerophase=False):
    """
    CJH Stolen from obspy.signal.bandstop to apply over axis=0

    Butterworth-Bandstop Filter.

    Filter data removing data between frequencies ``freqmin`` and ``freqmax``
    using ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Stop band low corner frequency.
    :param freqmax: Stop band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high > 1:
        high = 1.0
        msg = "Selected high corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        print(msg)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high],
                        btype='bandstop', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data, axis=1)
        return np.flip(sosfilt(sos, np.flip(firstpass[::-1], axis=1),
                               axis=1), axis=1)
    else:
        return sosfilt(sos, data, axis=1)


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


def calculate_leg_deviation(well_data):
    """Helper to calculate the difference between down and upgoing legs"""
    for well, w_dict in well_data.items():
        data = w_dict['data']
        down_data, up_data = np.array_split(data, 2)
        if down_data.shape[0] != up_data.shape[0]:
            up_data = np.insert(up_data, 0, down_data[-1, :], axis=0)
        diff = np.ma.subtract(down_data, np.flip(up_data, axis=0),
                              out=np.zeros_like(down_data),
                              where=down_data!=0)
        w_dict['leg_difference'] = diff
    return well_data


def gain_correction(well_data, gain_thresh=0.015, debug=0):
    """
    Correct data for jumps in gain that produce freq shift

    :param well_data: well_data dict
    :param gain_thresh: Threshold above which to remove measurement contribution
    :param debug: Debug flag for plotting

    :return:
    """
    for well, wdict in well_data.items():
        # Absolute diff along time axis
        g_prime = np.abs(np.diff(wdict['gain'], axis=1, append=0.))
        d_prime = np.diff(wdict['data'], axis=1, append=0.)
        offenders = np.where(g_prime > gain_thresh)
        new_data = wdict['data'].copy()
        for i, (row, col) in enumerate(np.c_[offenders]):
            new_data[row, col:] -= d_prime[row, col-1]
            try:
                # If next sample not in col but d_prime large, remove it
                if (np.abs(d_prime[row, col]) >= 5 and
                    offenders[1][i+1] != col + 1):
                    new_data[row, col+1:] -= d_prime[row, col]
            except IndexError:
                continue  # Skip final sample
        if debug > 0 and well in ['D5', 'D1', 'D2']:
            fig, axes = plt.subplots(nrows=2, sharex='col', figsize=(10, 7))
            fig.suptitle('Gain shift correction', fontsize=18, x=0.3, y=0.95,
                         weight='bold')
            ind = np.argmin(np.abs(wdict['depth'] - wdict['depth'][-1] +
                                   fault_depths[well][0]))
            axes[0].plot(wdict['times'], wdict['gain'][ind, :], color='r',
                      label='Gain')
            ax1 = axes[0].twinx()
            ax1.plot(wdict['times'], wdict['data'][ind, :], color='b',
                     label='Data')
            ax1.plot(wdict['times'], new_data[ind, :] - new_data[ind, :][0],
                     color='steelblue', label='Corrected data')
            axes[1].plot(wdict['times'], g_prime[ind, :], color='k',
                         label=r'$\Delta$Gain')
            ax4 = axes[1].twinx()
            ax4.plot(wdict['times'], d_prime[ind, :], color='g',
                     label=r'$\Delta$Strain')
            axes[1].axhline(gain_thresh, color='darkgray', linestyle=':',
                            label='Correction threshold')
            fig.legend(ncol=3)
            fig.autofmt_xdate()
            axes[0].set_ylabel('Gain [%]', fontsize=14, color='r')
            axes[0].set_ylim(bottom=0, top=1)
            ax1.set_ylabel(r'Strain [$\mu\epsilon$]', fontsize=14, color='b')
            axes[1].set_ylabel(r'Gain change [$\%$]', fontsize=14)
            ax4.set_ylabel(r'Strain change [$\mu\epsilon$]', fontsize=14,
                           color='g')
            # axes[0].set_xlim(right=datetime(2019, 6, 3))
            plt.show()
        # Assign to new key
        wdict['data'] = new_data
    return


def pick_anomalies(data, noise_mean, noise_mad, thresh=1., prominence=30.):
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
                      width=(None, None), prominence=prominence)


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


def interpolate_picks(pick_dict, gridx, gridy, gridz, method='linear', debug=0):
    pts = []
    strains = []
    well_dict = create_FSB_boreholes()
    for well, w_dict in pick_dict.items():
        for feature_dep in w_dict['depths']:
            pts.append(depth_to_xyz(well_dict, well, feature_dep))
        strains.extend(w_dict['strains'])
    interp = griddata(np.array(pts), np.array(strains), (gridx, gridy, gridz),
                      method=method)
    # Test plotting
    if debug > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scat = ax.scatter(gridx, gridy, gridz, c=interp.flatten(), alpha=0.5)
        fig.colorbar(scat)
        plt.show()
    return interp


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
    try:
        temp = well_d['interp_temp']
    except KeyError as e:
        temp = np.zeros(data.shape)
    times = well_d['times']
    try:
        gain = well_d['gain']
    except KeyError:
        print('No gain correction')
        gain = None
    data_median = rolling_stats(data, times, depths, window, stat='median')
    data_std = rolling_stats(data, times, depths, window, stat='std')
    if direction == 'up':
        down_d, up_d = np.array_split(depths, 2)
        down_data, up_data = np.array_split(data, 2)
        down_median, up_median = np.array_split(data_median, 2)
        down_std, up_std = np.array_split(data_std, 2)
        down_temp, up_temp = np.array_split(temp, 2)
        down_gain, up_gain = np.array_split(gain, 2)
        if down_d.shape[0] != up_d.shape[0]:
            # prepend last element of down to up if unequal lengths by 1
            up_d = np.insert(up_d, 0, down_d[-1])
            up_data = np.insert(up_data, 0, down_data[-1, :], axis=0)
            up_median = np.insert(up_median, 0, down_median[-1, :], axis=0)
            up_std = np.insert(up_std, 0, down_std[-1, :], axis=0)
            up_temp = np.insert(up_temp, 0, down_temp[-1, :], axis=0)
            up_gain = np.insert(up_gain, 0, down_gain[-1, :], axis=0)
        depths = np.abs(up_d - up_d[-1])
        data = up_data
        data_median = up_median
        data_std = up_std
        temp = up_temp
        gain = up_gain
    # Find closest channel
    chan = np.argmin(np.abs(depth - depths))
    strains = data[chan, :]
    strain_median = data_median[chan, :]
    strain_std = data_std[chan, :]
    temps = temp[chan, :]
    gain = gain[chan, :]
    return times, strains, strain_median, strain_std, temps, gain


def extract_strains(well_data, date, wells, average=True, reference_time=None):
    """
    For a given datetime, extract the strain along the borehole (averaged
    between down and upgoing legs...?)

    :param well_data: Output of extract_wells
    :param date: Datetime object to extract
    :param wells: List of well names
    :param average: Bool for averaging up and down, or returning both separately
    :return:
    """
    pick_dict = {}
    for well, well_dict in well_data.items():
        date_col = np.argmin(np.abs(well_dict['times'] - date))
        if reference_time:
            ref_col = np.argmin(np.abs(well_dict['times'] - reference_time))
        else:
            ref_col = 0
        print(ref_col)
        data_mat = well_dict['data'] - well_dict['data'][:, ref_col, np.newaxis]
        if well not in wells:
            continue
        pick_dict[well] = {}
        # Grab along-fiber distances, split in two
        deps = well_dict['depth'] - well_dict['depth'][0]
        down_d, up_d = np.array_split(deps, 2)
        # Same for data array
        data = data_mat[:, date_col]
        down_data, up_data = np.array_split(data, 2)
        if down_d.shape[0] != up_d.shape[0]:
            # prepend last element of down to up if unequal lengths by 1
            up_d = np.insert(up_d, 0, down_d[-1])
            up_data = np.insert(up_data, 0, down_data[-1])
        # Flip up_data to align
        if average:
            avg_data = (down_data + up_data[::-1]) / 2.
            pick_dict[well]['strains'] = avg_data
        else:
            pick_dict[well]['up_data'] = up_data[::-1]
            pick_dict[well]['down_data'] = down_data
        pick_dict[well]['depths'] = down_d
    return pick_dict


def get_plane_z(X, Y, strike, dip, point):
    """
    Helper to return the Z values of a fault/frac on a grid defined by X, Y

    :param X: Array defining the X coordinates
    :param Y: Array defining the Y coordinates
    :param strike: Strike of plane (deg clockwise from N)
    :param dip: Dip of plane (deg down from horizontal; RHR applies)
    :param point: Point that lies on the plane
    """
    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    # Define fault normal
    a, b, c = (np.sin(d) * np.cos(s), -np.sin(d) * np.sin(s), np.cos(d))
    d = (a * point[0]) + (b * point[1]) + (c * point[2])
    Z = (d - (a * X) - (b * Y)) / c
    return Z


def get_strain(volume, gridx, gridy, gridz, planez):
    # Get the 2D index array. This is the z-index closest to the plane for
    # each X-Y pair
    inds = np.argmin(np.abs(gridz - planez), axis=2)
    # Index the 3D strain array with this index array
    strains = volume[np.arange(inds.shape[0])[:, None],
                     np.arange(inds.shape[1]), inds]
    # Also return the x and y coords of this plane
    xgrid = gridx[np.arange(inds.shape[0])[:, None],
                  np.arange(inds.shape[1]), inds]
    ygrid = gridy[np.arange(inds.shape[0])[:, None],
                  np.arange(inds.shape[1]), inds]
    zgrid = gridz[np.arange(inds.shape[0])[:, None],
                  np.arange(inds.shape[1]), inds]
    return strains, xgrid, ygrid, zgrid


def get_well_piercepoint(wells):
    """
    Return the xyz points of the main fault for a list of wells

    :param wells: List
    :return:
    """
    well_dict = create_FSB_boreholes()
    pierce_dict = {}
    for well in wells:
        pierce_dict[well] = {'top': depth_to_xyz(well_dict, well,
                                                 fault_depths[well][0])}
        pierce_dict[well]['bottom'] = depth_to_xyz(well_dict, well,
                                                   fault_depths[well][1])
    return pierce_dict

################  Plotting  Funcs  ############################################

def plot_fiber_mapping(well_data, fiber, mapping, title='Fiber mapping',
                       xlims=None):
    fig, axes = plt.subplots(figsize=(10, 3))
    cols = cycle(sns.color_palette())
    # Just plot first time sample
    axes.plot(well_data[fiber]['depth'], well_data[fiber]['data'][:, 0],
              color='k', label='Data')
    mapping = mapping_dict[mapping][fiber]
    for well, depth in mapping.items():
        c = next(cols)
        if type(depth) == tuple:
            axes.axvline(depth[1], label=well, color=c)
            axes.axvline(depth[0], color=c, linestyle='--')
            axes.axvline(depth[2], color=c, linestyle='--')
        else:
            axes.axvline(depth, label=well, color=c)
            axes.axvline(depth - fiber_depths[well], color=c, linestyle='--')
            axes.axvline(depth + fiber_depths[well], color=c, linestyle='--')
    fig.suptitle(title, fontsize=16)
    fig.legend(ncol=4)
    axes.set_xlabel('Meters along fiber')
    axes.set_ylabel('Absolute Freq [GHz]')
    axes.xaxis.set_minor_locator(MultipleLocator(10))
    plt.grid(True, which='both', axis='x')
    if xlims:
        axes.set_xlim(xlims)
    plt.tight_layout()
    plt.show()
    return


def plot_temp_removal(well_data, well, vmin=-50, vmax=50):
    # Plot images of temperature removal at different stages
    fig, axes = plt.subplots(nrows=3, figsize=(6, 8), sharex='col')
    times = well_data[well]['times']
    depth = well_data[well]['depth'] - well_data[well]['depth'][0]
    it = axes[0].imshow(
        well_data[well]['temp-induced_strain'],
        extent=(date2num(times[0]), date2num(times[-1]),
                         depth[-1], depth[0]),
        origin='upper', aspect='auto',
    vmin=vmin, vmax=vmax)
    axes[0].set_title('Temp-induced strain', fontsize=16)
    plt.colorbar(it, label=r'$\mu\varepsilon$', ax=axes[0])
    us = axes[1].imshow(
        well_data[well]['uncorrected_strain'],
        extent=(date2num(times[0]), date2num(times[-1]),
                         depth[-1], depth[0]),
        origin='upper', aspect='auto',
        vmin=vmin, vmax=vmax)
    axes[1].set_title('Uncorrected strain', fontsize=16)
    plt.colorbar(us, label=r'$\mu\varepsilon$', ax=axes[1])
    axes[1].set_ylabel('Distance from borehole entry (full loop)', fontsize=18)
    cs = axes[2].imshow(
        well_data[well]['data'],
        extent=(date2num(times[0]), date2num(times[-1]),
                         depth[-1], depth[0]),
        origin='upper', aspect='auto',
        vmin=vmin, vmax=vmax)
    axes[2].set_title('Corrected strain', fontsize=16)
    axes[2].xaxis_date()
    plt.colorbar(cs, label=r'$\mu\varepsilon$', ax=axes[2])
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[2].set_xlabel('Date', fontsize=16)
    fig.autofmt_xdate()
    plt.show()
    return


def plot_channel_timeseries(well_data, well, depths):
    """
    Plot standalone multi-channel timeseries(es)

    :param well_data: From extract_wells
    :param well: Well string
    :param depths: List of tuples with (depth, direction (i.e. up/down))
    :param normalized: Whether to normalize the traces to max = 1
    :return:
    """
    cmap = cycle(sns.color_palette('dark', 8))
    fig, axes = plt.subplots(nrows=2, sharex='col', figsize=(8, 12))
    for depth in depths:
        print(depth)
        col = next(cmap)
        times, data, _, _, _, _ = extract_channel_timeseries(
            well_data, well, depth[0], direction=depth[1])
        data_norm = data / np.max(np.abs(data))
        axes[0].plot(times, data, color=col,
                     label='{}: {} m'.format(well, depth[0]))
        axes[1].plot(times, data_norm, color=col)
    fig.suptitle('Borehole BCS-{}'.format(well), fontsize=20)
    axes[0].set_ylabel('Microstrain', fontsize=14)
    axes[0].set_title('Strain', fontsize=16)
    axes[1].set_ylabel('Normalized strain', fontsize=14)
    axes[1].set_title('Normalized', fontsize=16)
    axes[1].set_xlabel('Date', fontsize=14)
    axes[0].legend(title='Depth', loc='upper left',
                   bbox_to_anchor=(0.0, 0.1), framealpha=1.)
    axes[0].set_zorder(1000)
    axes[0].grid(True, which='major', axis='both')
    axes[1].grid(True, which='major', axis='both')
    axes[0].set_facecolor('lightgray')
    axes[1].set_facecolor('lightgray')
    fig.autofmt_xdate()
    return fig


def plot_interpolation_frames(well_data, date_range, strike, dip, points,
                              xrange, yrange, zrange, sampling, wells,
                              vlims=(-80, 80), outdir=None, debug=0):
    """
    Save frames of DSS interpolation for video creation

    :param well_data: Dictionary from extract_wells
    :param date_range: Start and stop date tuple
    :param strike: list of strike of the planes through volume
    :param dip: Dips of the planes (to correct to actual units)
    :param points: Points that lie on the planes
    :param xrange: List of min and max values of X coordinates
    :param yrange: List of min and max values of Y coordinates
    :param zrange: List of min and max values of Z coordinates
    :param sampling: Sampling interval (same unit as above)
    :param wells: Which wells to include in the interp
    :param vlims: Colorbar limits passed to matplotlib
    :param outdir: Output directory for frames
    :param debug: Debug flag for extra plotting

    :return:
    """
    for date in date_generator(date_range[0], date_range[1], frequency='hour'):
        fname = os.path.join(outdir, '{}.png'.format(date))
        print(date)
        ax = plot_DSS_interpolation(well_data, date, strike, dip, points,
                                    xrange, yrange, zrange, sampling, wells,
                                    vlims, outfile=fname, debug=debug)
    return


def plot_DSS_interpolation(well_data, date, strike, dip, points,
                           xrange, yrange, zrange, sampling,
                           wells, vlims=(-80, 80), outfile=None,
                           debug=0):
    """
    Plot a 2D image of a slice taken through a 3D interpolation of the DSS
    results

    :param well_data: Dictionary from extract_wells
    :param date: Datetime of the image
    :param strike: list of strike of the planes through volume
    :param dip: Dips of the planes (to correct to actual units)
    :param points: Points that lie on the planes
    :param xrange: List of min and max values of X coordinates
    :param yrange: List of min and max values of Y coordinates
    :param zrange: List of min and max values of Z coordinates
    :param sampling: Sampling interval (same unit as above)
    :param wells: Which wells to include in the interp
    :param vlims: Colorbar limits passed to matplotlib
    :param outfile: Path to output file
    :param debug: Debug flag for extra plotting

    :return:
    """
    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    Xs = np.arange(xrange[0], xrange[1], sampling)
    Ys = np.arange(yrange[0], yrange[1], sampling)
    Zs = np.arange(zrange[0], zrange[1], sampling)
    print('Creating grid')
    gridx, gridy, gridz = np.meshgrid(Xs, Ys, Zs, indexing='xy', sparse=False)
    grid_shape = gridx.shape
    # Define temporary origin for rotation
    ox, oy, oz = (np.mean(gridx), np.mean(gridy), np.mean(gridz))
    print('Rotating grid to align with fault')
    if debug > 0:
        fig = plt.figure()
        ax = fig.add_subplot(311, projection='3d')
        ax.scatter(gridx, gridy, gridz, color='magenta', alpha=0.5)
    # Rotate grid parallel to fault (strike and dip)
    thetaz = np.deg2rad(90 - strike)
    thetax = np.deg2rad(dip)
    pts = np.vstack([gridx.flatten() - ox, gridy.flatten() - oy,
                     gridz.flatten() - oz])
    rot_matz = np.array([[np.cos(thetaz), -np.sin(thetaz), 0],
                         [np.sin(thetaz), np.cos(thetaz), 0],
                         [0, 0, 1]])
    rot_matx = np.array([[1, 0, 0],
                         [0, np.cos(thetax), -np.sin(thetax)],
                         [0, np.sin(thetax), np.cos(thetax)]])
    big_rot = rot_matz.dot(rot_matx)
    # Do rotation, then add the origin back in
    gridx, gridy, gridz = big_rot.dot(pts)
    if debug > 0:
        ax2 = fig.add_subplot(312, projection='3d')
        intx, inty, intz = rot_matz.dot(pts)
        intx2, inty2, intz2 = rot_matx.dot(np.array([intx, inty, intz]))
        ax2.scatter(intx, inty, intz, color='k', alpha=0.5)
        ax2.scatter(intx2, inty2, intz2, color='green', alpha=0.5)
        ax2.scatter(gridx, gridy, gridz, color='pink', alpha=0.5)
    gridx = (gridx + ox).reshape(grid_shape)
    gridy = (gridy + oy).reshape(grid_shape)
    gridz = (gridz + oz).reshape(grid_shape)
    print('Extracting DSS strain values from boreholes')
    pick_dict = extract_strains(well_data, date=date, wells=wells)
    top_point = (2579327.55063806, 1247523.80743839, 419.14869573)
    bottom_point = (2579394.34498769, 1247583.94281201, 425.28368236)
    faultZ_top = get_plane_z(gridx, gridy, strike=52., dip=57.,
                             point=top_point)
    faultZ_bot = get_plane_z(gridx, gridy, strike=52., dip=57.,
                             point=bottom_point)
    if debug > 0:
        ax.scatter(gridx, gridy, gridz, color='b', alpha=0.5)
        ax.plot_surface(gridx[:, :, -1], gridy[:, :, -1], faultZ_top[:, :, -1],
                        color='r', alpha=0.5)
    print('Running interpolation on rotated grid')
    volume = interpolate_picks(pick_dict, gridx, gridy, gridz, method='linear')
    print('Pulling strain values from interpolation points closest to fault')
    color_top, xs_t, ys_t, zs_t = get_strain(volume=volume, gridx=gridx,
                                             gridy=gridy, gridz=gridz,
                                             planez=faultZ_top)
    color_bot, xs_b, ys_b, zs_b = get_strain(volume=volume, gridx=gridx,
                                             gridy=gridy, gridz=gridz,
                                             planez=faultZ_bot)
    # Change of basis for fault xy to fault plane coordinates
    origin = np.array((np.mean(xs_t), np.mean(ys_t), np.mean(faultZ_top)))
    normal = np.array((np.sin(d) * np.cos(s), -np.sin(d) * np.sin(s),
                       np.cos(d)))
    strike_new = np.array([np.sin(s), np.cos(s), 0])
    up_dip = np.array([-np.cos(s) * np.cos(d), np.sin(s) * np.cos(d), np.sin(d)])
    change_B_mat = np.array([strike_new, up_dip, normal])
    grid_pts = np.subtract(np.array([xs_t.flatten(), ys_t.flatten(),
                                     zs_t.flatten()]),
                           origin[:, None])
    newx, newy, newz = change_B_mat.dot(grid_pts)
    newx = newx.reshape(xs_t.shape)
    newy = newy.reshape(xs_t.shape)
    if debug > 0:
        ax3 = fig.add_subplot(313, projection='3d')
        ax3.scatter(newx, newy, newz, c=color_top)
        plt.show()
    # Make the pierce points
    pierce_points = get_well_piercepoint(wells)
    if debug > 0:
        plt.imshow(color_top)
        plt.show()
    # Calculate distance between grid vertices as extents
    fig, ax = plt.subplots(nrows=1, ncols=1)
    cmap = ListedColormap(sns.color_palette('RdBu_r', 21).as_hex())
    cmap_norm = Normalize(vmin=vlims[0], vmax=vlims[1])
    levels = np.linspace(vlims[0], vlims[1], 21)
    img = ax.contourf(newx, newy, color_top, levels=levels, cmap=cmap,
                      norm=cmap_norm, vmin=vlims[0], vmax=vlims[1],
                      extend='both')
    # Plot well pierce points
    for well, pts in pierce_points.items():
        p = np.array(pts['top'])
        # Project onto plane in question
        proj_pt = p - (normal.dot(p - top_point)) * normal
        trans_pt = proj_pt - origin
        new_pt = change_B_mat.dot(trans_pt.T)
        ax.scatter(new_pt[0], new_pt[1], marker='+', color='k')
        ax.annotate(s=well, xy=(new_pt[0], new_pt[1]), fontsize=10)
    #     ax[1].scatter(pts['bottom'][0], pts['bottom'][1], marker='+', color='k')
    #     ax[1].annotate(s=well, xy=(pts['bottom'][0], pts['bottom'][1]),
    #                    fontsize=10)
    fig.colorbar(img)
    ax.set_ylabel('Up-dip distance [m]')
    ax.set_xlabel('Along strike distance [m]')
    fig.suptitle('Interpolated strain on Main Fault: {}'.format(date.date()))
    if outfile:
        plt.savefig(outfile, dpi=200)
        plt.close('all')
    else:
        plt.show()
    return ax


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
        well_dict = parse_surf_boreholes(
            '/media/chet/hdd/seismic/chet_collab/boreholes/surf_4850_wells.csv')
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
    times.sort()
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
        print(t)
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
                         formater=True, frame=False,
                         plot_preceding_times=None):
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
    :param plot_preceding_times: int for plotting the preceding time slices

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
        # If frame for animation, keep last 20 time samples too
        if frame and plot_preceding_times:
            old_vects = data[:, time_int - plot_preceding_times:time_int]
            old_down, old_up = np.array_split(old_vects, 2)
            if old_down.shape[0] != old_up.shape[0]:
                old_up = np.append(old_up, old_down[-1]).reshape(old_down.shape)
            try:
                ax1.plot(old_down, down_d[:, None], color='grey', alpha=alpha,
                         linewidth=0.5)
            except ZeroDivisionError as e:
                print('Trying to plot old timeslices that dont exist')
                print('Pick a later start date')
                return
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
            try:
                # Plot fault extents and resin plug, if CSD
                ax1.axhline(fault_depths[well][0], linestyle='--',
                            linewidth=1., color='k')
                ax1.axhline(fault_depths[well][1],
                            linestyle='--', label='Main Fault',
                            linewidth=1., color='k')
                ax2.axhline(fault_depths[well][0], linestyle='--',
                            linewidth=1., color='k')
                ax2.axhline(fault_depths[well][1],
                            linestyle='--',
                            linewidth=1., color='k')
            except KeyError:
                i += 2
                continue
            try:
                # Fill between resin plug
                ax1.fill_between(
                    x=np.array([-500, 500]), y1=resin_depths[well][0],
                    y2=resin_depths[well][1], hatch='/',
                    alpha=0.5, color='bisque')
                ax2.fill_between(
                    x=np.array([-500, 500]), y1=resin_depths[well][0],
                    y2=resin_depths[well][1], hatch='/',
                    alpha=0.5, color='bisque', label='Resin plug')
            except KeyError as e:
                i += 2
                continue
        # Always increment, obviously
        i += 2
    if formater:
        if frame:
            lab_y = 0.04
        else:
            lab_y = 0.04
            # lab_y = 0.19
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
        fig.text(0.05, 0.02, date, ha="left", va="bottom", fontsize=20,
                 bbox=dict(boxstyle="round",
                           ec='k', fc='white'))
        fig.savefig('frame_{}.pdf'.format(date))
        plt.close('all')
    return fig


def plot_DSS(well_data, well='all', derivative=False, colorbar_type='light',
             inset_channels=True, simfip=False, pick_mode='auto', thresh=1.,
             date_range=(datetime(2019, 5, 19), datetime(2019, 6, 4)),
             denoise_method=None, window='2h', vrange=(-60, 60), title=None,
             tv_picks=None, prominence=30., pot_data=None, hydro_data=None,
             offset_samps=120, filter_params=None, plot_stack=False,
             integrate_anchor_segs=True, gain_correction=False):
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
    :param plot_stack: Whether to plot the stack of 20 channels centered on
        manual pick.
    :param integrate_anchor_segs: For CSD boreholes D1-D2, integrate strain signal
        over the anchored segments?
    :param gain_correction: Scale to shifts in gain?

    :return:
    """
    # TODO This is ghastly
    if inset_channels and simfip and well != 'D5' and not hydro_data:
        fig = plt.figure(constrained_layout=False, figsize=(14, 14))
        gs = GridSpec(ncols=14, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:3, 7:-1])
        axes1b = fig.add_subplot(gs[3:6, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[6:9, 7:-1], sharex=axes1)
        axes3 = fig.add_subplot(gs[9:, 7:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        cax = fig.add_subplot(gs[:6, -1])
        try:
            df = read_excavation(simfip)
        except KeyError:
            df = read_collab(simfip)
    elif inset_channels and well != 'D5' and not hydro_data:
        fig = plt.figure(constrained_layout=False, figsize=(14, 14))
        gs = GridSpec(ncols=12, nrows=12, figure=fig)
        axes1 = fig.add_subplot(gs[:4, 7:-1])
        axes1b = fig.add_subplot(gs[4:8, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[8:, 7:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        cax = fig.add_subplot(gs[:8, -1])
    elif inset_channels and simfip and well == 'D5':
        fig = plt.figure(constrained_layout=False, figsize=(15, 16))
        gs = GridSpec(ncols=14, nrows=20, figure=fig)
        axes1 = fig.add_subplot(gs[:4, 7:-1])
        axes1b = fig.add_subplot(gs[4:8, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[8:12, 7:-1], sharex=axes1)
        axes3 = fig.add_subplot(gs[12:16, 7:-1], sharex=axes1)
        pot_ax = fig.add_subplot(gs[16:, 7:-1], sharex=axes1)
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        df = read_excavation(simfip)
        cax = fig.add_subplot(gs[:8, -1])
    elif inset_channels and simfip and hydro_data:
        fig = plt.figure(constrained_layout=False, figsize=(15, 16))
        gs = GridSpec(ncols=14, nrows=20, figure=fig)
        axes1 = fig.add_subplot(gs[:4, 7:-1])
        axes1b = fig.add_subplot(gs[4:8, 7:-1], sharex=axes1)
        axes2 = fig.add_subplot(gs[8:12, 7:-1], sharex=axes1)
        axes3 = fig.add_subplot(gs[12:16, 7:-1], sharex=axes1)
        hydro_ax = fig.add_subplot(gs[16:, 7:-1], sharex=axes1)
        pres_ax = hydro_ax.twinx()
        axes4 = fig.add_subplot(gs[:, 2:4])
        axes5 = fig.add_subplot(gs[:, 4:6], sharex=axes4)
        log_ax = fig.add_subplot(gs[:, :2], sharey=axes4)
        try:
            df = read_excavation(simfip)
        except KeyError:
            df = read_collab(simfip)
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
    times = well_data[well]['times']
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
    mode = well_data[well]['mode']
    type = well_data[well]['type']
    if date_range:
        indices = np.where((date_range[0] < times) & (times < date_range[1]))
        times = times[indices]
        data = np.squeeze(data[:, indices])
        if 'gain' in well_data[well].keys():
            gain = np.squeeze(gain[:, indices])
            if gain_correction:
                data = scale_to_gain(data, gain, offset_samps)
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
    if mode == 'Relative':
        data = data - data[:, 0:offset_samps, np.newaxis].mean(axis=1)
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
    if well in ['D1', 'D2'] and integrate_anchor_segs:
        data = integrate_anchors(data, depth_vect - depth_vect[0], well)
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
    if simfip:
        if 'P Top' in df.keys():
            plot_displacement_components(df, starttime=date_range[0],
                                         endtime=date_range[1], new_axes=axes3,
                                         location='collab')
        else:
            plot_displacement_components(df, starttime=date_range[0],
                                         endtime=date_range[1], new_axes=axes3,
                                         remove_clamps=False,
                                         rotated=True)
        axes3.set_ylabel(r'$\mu$m', fontsize=16)
        plt.setp(axes1.get_xticklabels(), visible=False)
        plt.setp(axes1b.get_xticklabels(), visible=False)
        plt.setp(axes2.get_xticklabels(), visible=False)
        plt.setp(axes3.get_xticklabels(), visible=False)
        if hydro_data:
            try:
                dfh = read_collab_hydro(hydro_data)
            except ParserError:
                dfh = read_csd_hydro(hydro_data)
            dfh = dfh[date_range[0]:date_range[1]]
            hydro_ax.plot(dfh['Flow'], color='steelblue',
                          label='Flow')
            pres_ax.plot(dfh['Pressure'], color='red', label='Pressure')
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
    elif hydro_data:
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
    if not pot_data and simfip and not hydro_data:
        axes3.xaxis_date()
        axes3.xaxis.set_major_formatter(date_formatter)
        plt.setp(axes3.xaxis.get_majorticklabels(), rotation=30, ha='right')
    elif pot_data:
        pot_ax.xaxis_date()
        pot_ax.xaxis.set_major_formatter(date_formatter)
        plt.setp(pot_ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        plt.setp(axes3.get_xticklabels(), visible=False)
    elif hydro_data:
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
                if not frac_type.startswith('sed'):
                    log_ax.plot(dens[:, 1], dens[:, 0],
                                color=frac_cols[frac_type],
                                label=frac_type)
            log_ax.legend(
                loc=2, fontsize=12, bbox_to_anchor=(-1.2, 1.13),
                framealpha=1.).set_zorder(110)
        if well == 'D5' and pot_data:  # Potentiometer elements
            pot_d, pot_depths, pot_times = read_potentiometer(pot_data)
            max_x = log_ax.get_xlim()[-1]
            ymin = []
            ymax = []
            cs = []
            pot_cols = cycle(sns.color_palette("Paired"))
            for i, dep in enumerate(pot_depths):
                deps = potentiometer_depths[str(12 - i)]
                if dep > 18.:
                    ymin.append(deps[0])
                    ymax.append(deps[-1])
                    cs.append(next(pot_cols))
            log_ax.vlines(x=np.ones(len(ymin)) * max_x,
                          ymax=ymax, ymin=ymin,
                          linewidth=8., color=cs)
            plot_potentiometer(pot_d, pot_depths, pot_times, axes=pot_ax,
                               colors='Paired')
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
                         up_d, down_d, pick_mode, noise, thresh, prominence):
                self.figure = figure
                self.cmap = cmap
                self.cat_cmap = cat_cmap
                self.pick_mode = pick_mode
                self.thresh = thresh
                self.prominence = prominence
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
                # Also stack a range of channels in case we want that?
                stack = np.sum(self.data[chan-10:chan+10, :], axis=0)
                stack *= (np.max(np.abs(trace)) / np.max(np.abs(stack)))
                # depth = self.depth[chan]
                depth = event.ydata
                # Grab along-fiber vector
                fiber_vect = self.data[:, time_int]
                self.figure.axes[2].axvline(x=event.xdata, color='k',
                                            linestyle='--', alpha=0.5)
                if simfip:
                    self.figure.axes[3].axvline(x=event.xdata, color='k',
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
                        label='Depth {:0.2f}'.format(depth), alpha=0.7)
                    if plot_stack:
                        self.figure.axes[2].plot(
                            self.times, stack, color='red',
                            label='Stack', alpha=0.7)
                else:
                    if self.noise[1] is None:
                        noise_mean = 0.
                        noise_mad = self.noise[0]
                    else:
                        noise_mean = self.noise[0]
                        noise_mad = self.noise[1]
                    peak_inds, peak_dict = pick_anomalies(
                        fiber_vect, noise_mean=noise_mean,
                        noise_mad=noise_mad, thresh=self.thresh,
                        prominence=self.prominence)
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
                            self.pick_dict[self.well]['depths'].append(rel_dep)
                            if rel_dep < 5:  # Ignore shallow picks for now
                                continue
                            self.figure.axes[-3].fill_between(
                                x=np.array([-500, 500]),
                                y1=rel_dep - half_width,
                                y2=rel_dep + half_width,
                                alpha=0.5, color=pick_col)
                            arr_ax = self.figure.axes[1]
                            trans = arr_ax.get_yaxis_transform()
                        else: # Downgoing
                            rel_dep = self.depth[pk[0]]
                            self.pick_dict[self.well]['depths'].append(rel_dep)
                            if rel_dep < 5:  # Ignore shallow picks for now
                                continue
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
                               noise=well_data[well]['noise'], thresh=thresh,
                               prominence=prominence)
        plt.show()
    return plotter.pick_dict


def plot_potentiometer(data, depths, times, colors=None, axes=None,
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
    data = data.T[indices, :]
    data = np.squeeze(data)
    # Divide by two for microns
    data *= 0.5
    if colors:
        cols = cycle(sns.color_palette(colors))
    else:
        cols = cycle(sns.color_palette())
    if axes:
        xlims = axes.get_xlim()
        axes2 = axes
    elif simfip:
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
    if not axes:
        cmap = ListedColormap(sns.color_palette('RdBu_r', 21).as_hex())
        # Flip sign of data to conform to extension == positive strain
        im = axes1.contourf(times, depths, data.T * -1., cmap=cmap,
                            vmin=vrange[0], vmax=vrange[1])
        axes1.invert_yaxis()
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel(r'$\mu$m', fontsize=16)
    for i, depth in enumerate(depths):
        # Only plot the lower instruments
        if depth > 18.:
            # Again flip data so + strain is extensional
            axes2.plot(times, data[:, i] * -1, label=depth, color=next(cols),
                       linewidth=1.5)
            axes2.invert_yaxis()  # Negative strain convention
    if axes:
        axes2.set_ylabel(r'$\mu$m', fontsize=16)
        axes2.set_xlim(xlims)
        axes2.text(0.05, 0.9, horizontalalignment='left', s='Potentiometer',
                   verticalalignment='center', transform=axes2.transAxes,
                   fontsize=16)
        return
    else:
        # Formatting
        axes2.legend(title='Depth')
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


def plot_D5_with_depth(well_data, time, tv_picks, pot_data, leg='up_data',
                       strain_range=(-170, 170), reference_time=None):
    """
    Plot D5 data with depth including DSS, fracture logs and potentiometer

    :param well_data: output from extract_wells
    :param time: Datetime object specifying time slice to pull from pot/DSS
    :param tv_picks: Path to excel file with optical televiewer picks
    :param pot_data: Path to potentiometer data
    :param leg: Either 'up_data' or 'down_data'
    :param strain_range:
    :return:
    """
    fig, axes = plt.subplots(ncols=3, figsize=(4, 10), sharey='row')
    dss_dict = extract_strains(well_data, date=time, wells=['D5'],
                               average=False, reference_time=reference_time)
    frac_dict = read_frac_cores(tv_picks, 'D5')
    for frac_type, dens in frac_dict.items():
        if not frac_type.startswith('sed'):
            axes[0].plot(dens[:, 1], dens[:, 0],
                         color=frac_cols[frac_type],
                         label=frac_type)
    pot_d, pot_depths, pot_times = read_potentiometer(pot_data)
    top_anchors = [d[0] for nm, d in potentiometer_depths.items()]
    top_anchors.sort()
    top_anchors = top_anchors[::-1]
    # Make depth series for potentiometer
    data = pot_d.T[(np.abs(pot_times - time)).argmin(), :]
    data = np.squeeze(data)
    # Divide by two for microns (flip for extension)
    data *= -0.5
    axes[1].step(data, top_anchors, label='Potentiometer',
                 color='r', where='post')
    # Plot anchors as lil black dots
    axes[1].scatter(data, top_anchors, c='k', s=2., zorder=101)
    # Lastly, DSS data
    axes[2].plot(dss_dict['D5'][leg], dss_dict['D5']['depths'],
                 label='DSS', color='purple')
    axes[0].invert_yaxis()
    axes[0].set_ylabel('Depth [m]')
    axes[0].set_xlabel('Fractures per m')
    axes[1].set_xlabel('Microns')
    axes[2].set_xlabel('Microns')
    axes[1].set_xlim(strain_range)
    axes[2].set_xlim(strain_range)
    # Zero line for DSS and potentiometer
    axes[1].axvline(0, linestyle=':', linewidth=0.5, color='k')
    axes[2].axvline(0, linestyle=':', linewidth=0.5, color='k')
    for i, ax in enumerate(axes):
        # Resin plug
        if i == 0:
            label = 'Main Fault'
        else:
            label = ''
            # Fill between resin plug
            ax.fill_between(
                x=np.array([-500, 500]), y1=resin_depths['D5'][0],
                y2=resin_depths['D5'][1], hatch='/',
                alpha=0.5, color='bisque')
        ax.axhline(fault_depths['D5'][0], linestyle='--',
                   linewidth=1., color='k')
        ax.axhline(fault_depths['D5'][1],
                   linestyle='--', label=label,
                   linewidth=1., color='k')
        ax.set_facecolor('lightgray')
        ax.set_ylim(top=0, bottom=fiber_depths['D5'])
    fig.text(0.1, 0.95, time.date(), ha="left", va="bottom", fontsize=14,
             bbox=dict(boxstyle="round",
                       ec='k', fc='white'))
    fig.legend()
    plt.show()
    return


def plot_D5_with_time(well_data, pot_data, depth, simfip,
                      dates=None):
    """
    Compare timeseries of potentiometer, DSS and SIMFIP (normal to fault)

    :param well_data: Output of extract_wells
    :param pot_data: Path to potentiometer file
    :param depth: Depth in well to plot
    :param simfip: Path to SIMFIP data
    :param dates: Date range to plot
    """
    fig, axes = plt.subplots(nrows=2, figsize=(8, 10), sharex='col')
    pot_d, pot_depths, pot_times = read_potentiometer(pot_data)
    # Which pot?
    pind = (np.abs(pot_depths - depth)).argmin()
    # Take 3 elements (1.5 m), sum and scale
    pot_strains = pot_d[pind-1:pind+2, :].sum(axis=0) / 1.5
    dss_times, dss_strains, _, _, _, _ = extract_channel_timeseries(
        well_data, 'D5', depth=depth, direction='up')
    if dates:
        date_inds = np.where((dates[0] <= dss_times) &
                             (dates[1] > dss_times))
        dss_times = dss_times[date_inds]
        dss_strains = dss_strains[date_inds]
    indices = np.where((pot_times > dss_times[0]) &
                       (pot_times < dss_times[-1]))
    pot_times = pot_times[indices]
    pot_d = pot_d[:, indices].squeeze()
    pot_strains = pot_strains[indices]
    # Relative to start of plot
    pot_strains = pot_strains - pot_strains[0]
    dss_strains = dss_strains - dss_strains[0]
    axes[0].plot(pot_times, -0.5 * pot_strains, color='r',
                 label='Potentiometer')
    axes[0].plot(dss_times, dss_strains, color='purple',
                 label='DSS')
    # Now simfip plot
    # Integrate DSS and Pot
    integrated_strain = integrate_depth_interval(well_data, depths=(20., 23.),
                                                 well='D5', leg='up',
                                                 dates=dates)
    # Sum potentiometer
    interval = np.where((pot_depths < 23.) & (pot_depths > 20.))
    integrated_pot = np.sum(-0.5 * pot_d[interval, :].squeeze(), axis=0)
    # Scale the integrated traces to sampling interval
    integrated_strain *= 0.255
    integrated_pot *= 0.5
    axes[1].plot(dss_times, integrated_strain, color='mediumorchid',
                 label='DSS')
    axes[1].plot(pot_times, integrated_pot - integrated_pot[0],
                 color='firebrick', label='Potentiometer')
    df_simfip = read_excavation(simfip)
    df_simfip = df_simfip.loc[((df_simfip.index < dss_times[-1])
                               & (df_simfip.index > dss_times[0]))]
    # Rotate simfip onto BCS-D5
    df_simfip = rotate_fsb_to_borehole(df_simfip, 'D5')
    df_simfip['Zf'] = df_simfip['Zf'] - df_simfip['Zf'][0]
    df_simfip['Yf'] = df_simfip['Yf'] - df_simfip['Yf'][0]
    df_simfip['Xf'] = df_simfip['Xf'] - df_simfip['Xf'][0]
    df_simfip['Sf'] = np.sqrt(df_simfip['Xf']**2 + df_simfip['Yf']**2)
    # Convert to shear strain
    df_simfip['ESf'] = df_simfip['Sf'] / 6.3
    # Theoretical arc length effect of SIMFIP shear as seen on DSS
    # assuming 6.3 m (SIMFIP interval) strained length
    df_simfip['Arc'] = (df_simfip['ESf'] / 6.3)**2 / 2
    df_simfip['Synthetic DSS'] = ((df_simfip['Zf'] / 6.3) +
                                  df_simfip['Arc']) * 6.3
    df_simfip['Zf'].plot(ax=axes[1], color='steelblue',
                         label='SIMFIP: Opening')
    df_simfip['Sf'].plot(ax=axes[1], color='lightseagreen',
                         label='SIMFIP: Shear')
    df_simfip['Synthetic DSS'].plot(ax=axes[1], color='blue',
                                    label='Synthetic DSS')
    for i, ax in enumerate(axes):
        ax.set_facecolor('lightgray')
        ax.set_ylabel('Microns', fontsize=14)
        if i == 0:
            ax.axvline(date2num(datetime(2019, 5, 27, 17)), linestyle='--',
                       color='gray', label='Breakthrough')
        else:
            ax.axvline(date2num(datetime(2019, 5, 27, 17)), linestyle='--',
                       color='gray')
    axes[1].xaxis.set_major_formatter(DateFormatter('%m-%d'))
    axes[1].set_xlabel(pot_times[0].year, fontsize=14)
    axes[0].set_title('DSS vs Potentiometer: {} m'.format(depth),
                      fontsize=16)
    axes[1].set_title('DSS, SIMFIP, and Potentiometer: Main Fault Interval',
                      fontsize=16)
    ax_ex = axes[0].twinx()
    df_excavation = distance_to_borehole(
        create_FSB_boreholes(), well='D5', depth=20.,
        gallery_pts='data/chet-FS-B/excavation/points_along_excavation.csv',
        excavation_times='data/chet-FS-B/excavation/G18excavationdistance.txt')
    exc_hand, = ax_ex.plot(
        df_excavation.index.values, df_excavation['Distance to SIMFIP'],
        label='Excavation front', color='k', linestyle='dotted')
    leg_hand, leg_lab = axes[0].get_legend_handles_labels()
    print(leg_hand, exc_hand)
    axes[0].legend()
    ax_ex.legend(loc='lower right')
    axes[1].legend()
    ax_ex.set_ylabel('Distance to fault [m]', fontsize=14)
    ax_ex.set_ylim((24.5, 27.))
    axes[0].margins(0.)
    axes[1].margins(0.)
    if dates:
        axes[0].set_xlim(dates)
    plt.show()
    return


def plot_csd_injection(well_data_1, time, depths, dates=None, leg='up_data',
                       well_data_2=None, strain_range=(-170, 170),
                       window='2h'):
    """
    Plot final figure for co2 injection period at CSD

    :param well_data_1: Well data from 0.5 m resolution of co2 test or whole
        august pulse
    :param time: Time point to plot the well traces
    :param depths: Depth at which to plot timeseries {well: depth}
    :param dates: Date range if clipping
    :param leg: 'up' or 'down'
    :param well_data_2: well data from 1.0 m resolution for co2 test
    :param strain_range: Strain range for full-well traces
    :param window: Window length for rolling stats of timeseries

    :return:
    """
    # Copy out of the way
    well_data1 = deepcopy(well_data_1)
    if well_data_2:
        well_data2 = deepcopy(well_data_2)
    fig = plt.figure(figsize=(11, 9))
    spec = GridSpec(ncols=7, nrows=6, figure=fig, wspace=0.13, hspace=0.1)
    ax5pot = fig.add_subplot(spec[:, 0])
    ax5dss = fig.add_subplot(spec[:, 1], sharey=ax5pot, sharex=ax5pot)
    ax1dss = fig.add_subplot(spec[:, 2], sharey=ax5pot, sharex=ax5pot)
    ax2dss = fig.add_subplot(spec[:, 3], sharey=ax5pot, sharex=ax5pot)
    ax_time = fig.add_subplot(spec[:3, 4:])
    ax_hydro = fig.add_subplot(spec[3:, 4:], sharex=ax_time)
    if well_data_2:
        pot_data = 'data/chet-FS-B/potentiometer/CO2_injection/dataGEOMONITOR'
    else:
        pot_data = 'data/chet-FS-B/potentiometer/pulse_test/dataGEOMONITOR'
    pot_df = read_potentiometer_raw(pot_data)
    pot_d = pot_df.values.T
    pot_times = pd.to_datetime(pot_df.index)
    dt = timedelta(hours=1)
    pot_times -= dt  # One hour ahead of UTC
    pot_depths = [d[1] for i, d in potentiometer_depths.items()]
    pot_depths.sort(reverse=True)
    pot_strains = pot_d[(np.abs(np.array(pot_depths) -
                                depths['D5'])).argmin(), :]
    # Integrate D1 and D2 over anchor intervals
    well_data1['D1']['data'] = integrate_anchors(well_data1['D1']['data'],
                                                 well_data1['D1']['depth'],
                                                 'D1')
    well_data1['D2']['data'] = integrate_anchors(well_data1['D2']['data'],
                                                 well_data1['D2']['depth'],
                                                 'D2')
    if well_data_2:
        well_data2['D1']['data'] = integrate_anchors(well_data2['D1']['data'],
                                                     well_data2['D1']['depth'],
                                                     'D1')
        well_data2['D2']['data'] = integrate_anchors(well_data2['D2']['data'],
                                                     well_data2['D2']['depth'],
                                                     'D2')
    # Make dict of DSS timeseries for each well
    dss_time_dict = {}
    for well in ['D5', 'D1', 'D2']:
        dep = depths[well]
        dss_times1, dss_strains1, dss_med1, dss_std1, _, _ = extract_channel_timeseries(
            well_data1, well, depth=dep, direction='up', window=window)
        if well_data_2:
            dss_times2, dss_strains2, _, _, _, _ = extract_channel_timeseries(
                well_data2, well, depth=dep, direction='up', window=window)
        if dates:
            date_inds1 = np.where((dates[0] <= dss_times1) &
                                 (dates[1] > dss_times1))
            dss_times1 = dss_times1[date_inds1]
            dss_strains1 = dss_strains1[date_inds1]
            dss_med1 = dss_med1[date_inds1]
            dss_std1 = dss_std1[date_inds1]
            if well_data_2:
                date_inds2 = np.where((dates[0] <= dss_times2) &
                                     (dates[1] > dss_times2))
                dss_times2 = dss_times2[date_inds2]
                dss_strains2 = dss_strains2[date_inds2]
        dss_times1 -= dt  # One hour ahead of UTC
        dss_strains1 = dss_strains1 - dss_strains1[0]
        dss_med1 = dss_med1 - dss_med1[0]
        if well_data_2:
            dss_times2 -= dt
            dss_strains2 = dss_strains2 + dss_strains1[-1]
            # Coreful here, we're cookin. Setting start of data2 to end of data1
            dss_time_dict[well] = {'1': [dss_times1, dss_strains1],
                                   '2': [dss_times2, dss_strains2]}
            # Time series for potentiometer
            indices = np.where((pot_times > dss_times1[0]) &
                               (pot_times < dss_times2[-1]))
        else:
            dss_time_dict[well] = {'1': [dss_times1, dss_strains1,
                                         dss_med1, dss_std1]}
            indices = np.where((pot_times > dss_times1[0]) &
                               (pot_times < dss_times1[-1]))
    pot_times = pot_times[indices]
    pot_d = pot_d[:, indices].squeeze()
    pot_strains = pot_strains[indices]
    # Relative to start of plot
    pot_strains = pot_strains - pot_strains[0]
    # Depth series for DSS
    if well_data_2:
        dss_depth_dict = extract_strains(well_data2, date=time,
                                         wells=['D5', 'D1', 'D2'],
                                         average=False)
    else:
        dss_depth_dict = extract_strains(well_data1, date=time,
                                         wells=['D5', 'D1', 'D2'],
                                         average=False)
    # Depth series for potentiometer
    top_anchors = [d[0] for nm, d in potentiometer_depths.items()]
    top_anchors.sort()
    top_anchors = top_anchors[::-1]
    # Make depth series for potentiometer
    pot_depth_data = pot_d.T[(np.abs(pot_times - time)).argmin(), :]
    pot_depth_data = np.squeeze(pot_depth_data)
    # Divide by two for microns (flip for extension)
    pot_depth_data *= -0.5
    # Now plot everything
    ax5pot.step(pot_depth_data, top_anchors, color='r', where='post')
    ax5dss.plot(dss_depth_dict['D5'][leg], dss_depth_dict['D5']['depths'],
                color=csd_well_colors['D5'])
    ax1dss.plot(dss_depth_dict['D1'][leg], dss_depth_dict['D1']['depths'],
                color=csd_well_colors['D1'])
    ax2dss.plot(dss_depth_dict['D2'][leg], dss_depth_dict['D2']['depths'],
                color=csd_well_colors['D2'])
    # Resin plugs and fault depths
    for well, ax in zip(['D5', 'D5', 'D1', 'D2'], [ax5pot, ax5dss,
                                                   ax1dss, ax2dss]):
        ax.axhline(fault_depths[well][0], linestyle='--',
                    linewidth=1., color='k')
        ax.axhline(fault_depths[well][1],
                    linestyle='--',
                    linewidth=1., color='k')
        ax.set_title(well, fontsize=18, weight='bold')
        # Zero reference line
        ax.axvline(0., linestyle=':', color='gray')
        # Fill between resin plug
        try:
            ax.fill_between(
                x=np.array([-500, 500]), y1=resin_depths[well][0],
                y2=resin_depths[well][1], hatch='/',
                alpha=0.5, color='bisque')
        except KeyError:
            continue
    # Time series plotting
    ax_time.plot(pot_times, -0.5 * pot_strains, color='r',
                 label='Potentiometer')
    for w in ['D1', 'D2', 'D5']:
        if well_data_2:
            ax_time.plot(dss_time_dict[w]['1'][0], dss_time_dict[w]['1'][1],
                         color=csd_well_colors[w],
                         label='{} m'.format(depths[w]))
            ax_time.plot(dss_time_dict[w]['2'][0], dss_time_dict[w]['2'][1],
                         color=csd_well_colors[w])
        else:
            ax_time.plot(dss_time_dict[w]['1'][0], dss_time_dict[w]['1'][2],
                         color=csd_well_colors[w],
                         label='{} m'.format(depths[w]))
    # Finally the hydraulic data
    if well_data_2:
        df_hydro = read_csd_hydro('data/chet-FS-B/pump/CO2_injection/dataDCAM')
    else:
        df_hydro = read_csd_hydro('data/chet-FS-B/pump/pulse_test')
    plot_csd_hydro(df_hydro, axes=ax_hydro)
    # Formatting
    ax5pot.invert_yaxis()
    ax5pot.set_xlim(strain_range)
    ax_time.set_ylim(strain_range)
    if well_data_2:
        ax_time.fill_between(
            x=np.array([dss_time_dict[w]['1'][0][0],
                        dss_time_dict[w]['1'][0][-1]]),
            y1=1000, y2=-1000, color='lightgray')
        ax_time.axvline(datetime(2019, 6, 12, 15, 11), linestyle=':', color='k',
                        linewidth=1.5)
    for a in fig.axes:
        a.set_facecolor('whitesmoke')
        a.margins(0.)
    ax_time.yaxis.set_ticks_position('right')
    ax_time.yaxis.set_label_position('right')
    ax_time.set_ylabel(r'$\mu\epsilon$', fontsize=16)
    ax_time.tick_params(labelright=True)
    ax5pot.set_ylabel('Depth [m]', fontsize=16)
    if well_data_2:
        ax_hydro.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax_hydro.axvline(datetime(2019, 6, 12, 15, 11), linestyle=':', color='k',
                         linewidth=1.5, label='FOP reached')
        ax_hydro.set_xlabel('Time on {}'.format(df_hydro.index.date[0]),
                            fontsize=14, labelpad=10)
    ax_time.axvline(time, linestyle='--', color='navy', alpha=0.5)
    ax_hydro.axvline(time, linestyle='--', color='navy', alpha=0.5,
                    label='Plotted measure')
    plt.setp(ax_hydro.xaxis.get_majorticklabels(), rotation=-30,
             horizontalalignment='left')
    ax_hydro.tick_params(which='minor', length=0.)
    ax_time.tick_params(which='minor', length=0.)
    if well_data_2:
        ax_time.legend(loc='lower left')
        ax_hydro.legend()
    else:
        ax_time.legend(loc='upper left')
        ax_hydro.legend(loc='lower left')
    ax5dss.annotate(text=r'$\mu\epsilon$', xy=(.95, -0.11),
                    xycoords='axes fraction',
                    ha='left', fontsize=22)
    plt.show()
    return


def plot_strain_gain(well_data, well, depth, direction, title=''):
    """Compare strain and gain timeseries"""
    dss_times, dss_strains, _, _, _, dss_gain = extract_channel_timeseries(
        well_data, well, depth=depth, direction=direction)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    print()
    gain_correct = dss_gain / dss_gain[0]
    corrected_gain = dss_strains / gain_correct
    ax.plot(dss_times, dss_strains, label='Strain: {} m'.format(depth),
            color='steelblue')
    ax.plot(dss_times, corrected_gain, label='Scaled strain',
            color='dodgerblue')
    ax2.plot(dss_times, dss_gain, label='Gain', color='firebrick')
    fig.legend()
    fig.autofmt_xdate()
    ax.set_ylabel(r'$\mu\epsilon$')
    ax2.set_ylabel('Gain [%]')
    fig.suptitle(title)
    plt.show()
    return