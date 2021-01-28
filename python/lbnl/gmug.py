#!/usr/bin/python

"""
Functions for reading GMuG waveform data and writing to obspy/h5
"""

import os
import yaml
import pyasdf

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from datetime import datetime, timedelta
from obspy import Stream, Trace, UTCDateTime
from scipy.stats import median_absolute_deviation
from eqcorrscan.core.match_filter import normxcorr2

# Local imports
from SURF_SEIS.surf_seis.vibbox import vibbox_read

# Multiplexed channel order from a-to-d board
AD_chan_order = np.array([1, 9, 2, 10, 3, 11, 4, 12,
                          5, 13, 6, 14, 7, 15, 8, 16]) - 1

def read_raw_continuous(path, chans=16):
    """Read a raw, multiplexed .dat continuous waveform file"""
    raw = np.fromfile(path, dtype=np.int16)
    raw = raw.reshape((-1, chans))
    # Require 32-bit floats for conversion to mV
    raw = np.require(raw, dtype=np.float32, requirements=["C"])
    raw *= 5000 * 2 / 65536  # +/- 5000 mV over 16 bits
    return raw


def parse_continuous_metadata(path):
    # Time info
    with open(path.replace('.dat', '.txt')) as f:
        lines = f.readlines()
        # Recording start time
        ts = ''.join(lines[8].split()[-2:])
        channels = int(lines[3].split()[-1])
        delta = 1 / int(lines[4].split()[-1])
    starttime = datetime(year=int(ts[:4]), month=int(ts[4:6]),
                         day=int(ts[6:8]), hour=int(ts[8:10]),
                         minute=int(ts[10:12]), second=int(ts[12:14]),
                         microsecond=int('{}000'.format(ts[14:])))
    return UTCDateTime(starttime), channels, delta


def gmug_to_stream(pattern, config):
    """
    Take binary continuous wav and header file and return obspy Stream

    :param pattern: Root filename without extension (will be added)
    :return:
    """
    # Read in the config file and grab sta.chan list
    with open(config, 'r') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    stachans = np.array(param['Mapping']['GMuG_stachans'])
    # Re-order per multiplexer order
    multi_stachans = stachans[AD_chan_order]
    starttime, no_chans, delta = parse_continuous_metadata(
        '{}.txt'.format(pattern))
    np_raw = read_raw_continuous('{}.dat'.format(pattern), chans=no_chans)
    st = Stream(traces=[
        Trace(data=np_raw[:, i],
              header=dict(delta=delta, starttime=starttime,
                          network='FS', station=multi_stachans[i].split('.')[0],
                          channel=multi_stachans[i].split('.')[1],
                          location=''))
        for i in range(no_chans) if multi_stachans[i] != '.'])
    return st


def cassm_clock_correct(gmug_tr, vbox_tr, trig_tr, which=0, debug=0, name=None):
    """
    Find first CASSM shot in common and use cross correlation to estimate
    the clock offset between the two systems.

    Will be done for each GMuG file, but not each Vibbox file

    :param gmug_st: Trace of B81 on gmug
    :param vbox_st: Trace of B81 on vbox
    :param trig_tr: Trace of the CASSM trigger
    :param which: 0 for first or -1 for last trigger
    :param debug: Debug flag for correlation plot
    :param name: Name of output h5 file for plot nameing if debug > 0

    :return:
    """
    # Use derivative of PPS signal to find pulse start
    dt = np.diff(trig_tr.data)
    # Use 70 * MAD threshold
    samp_to_trig = np.where(
        dt > np.mean(dt) + 70 * median_absolute_deviation(dt))[0][which]
    trig1_time = vbox_tr.stats.starttime + (float(samp_to_trig) /
                                            vbox_tr.stats.sampling_rate)
    cc_vbox = vbox_tr.copy().trim(trig1_time,
                                  endtime=trig1_time + 0.01).detrend('demean')
    cc_gmug = gmug_tr.copy().trim(trig1_time,
                                  endtime=trig1_time + 0.2).detrend('demean')
    cc_gmug.resample(cc_vbox.stats.sampling_rate)
    ccc = normxcorr2(cc_vbox.data, cc_gmug.data)
    max_cc = np.argmax(ccc[0])
    max_cc_sec = float(max_cc) / cc_vbox.stats.sampling_rate
    if debug > 0:
        fig, axes = plt.subplots(nrows=2)
        vbox_x = np.arange(start=max_cc, stop=max_cc + cc_vbox.data.shape[0])
        axes[0].plot(cc_gmug.data / np.max(cc_gmug.data), color='k',
                     linewidth=0.7)
        axes[1].axvline(x=max_cc, linestyle=':', color='gray')
        axes[0].axvline(x=max_cc, linestyle=':', color='gray')
        axes[0].plot(vbox_x, cc_vbox.data / np.max(cc_vbox.data), color='r',
                     linewidth=0.7)
        axes[1].plot(ccc[0], color='b', linewidth=0.7)
        plt.savefig(name.replace('.h5', 'time_cc.png'))
        plt.close('all')
    return max_cc_sec, ccc, trig1_time


def which_vbox_files(gmug_st, vbox_files):
    """Select only those vbox files in the time range of gmug stream"""
    gmug_start = gmug_st[0].stats.starttime
    gmug_end = gmug_st[0].stats.endtime
    vbox_starts = [datetime.strptime(s.split('_')[-1][:14], '%Y%m%d%H%M%S')
                   for s in vbox_files]
    vbox_ends = [v + timedelta(seconds=32) for v in vbox_starts]
    which_overlap = [i for i, (strt, end) in enumerate(zip(vbox_starts,
                                                           vbox_ends))
                     if strt < gmug_end and end > gmug_start]
    return [vbox_files[i] for i in which_overlap]


def combine_vbox_gmug(vbox_dir, gmug_dir, gmug_param, outdir, inventory,
                      dug_params, overwrite=True, debug=0):
    """
    Turn two directories, one with vbox waveforms, the other with gmug waves,
    into a single directory of 32 sec-long, combined asdf files

    :param vbox_dir: Path to root for vbox files
    :param gmug_dir: Root for gmug files
    :param gmug_param: Path to GMuG parameter file
    :param outdir: Root output directory
    :param inventory: obspy Inventory
    :param dug_param: Path to DUG-seis parameter file with metadata
    :param overwrite: Overwrite files or no?
    :param debug: Debug flag for plotting

    :return:
    """
    # Read in dugseis parameters
    with open(dug_params, 'r') as f:
        dug_params = yaml.load(f, Loader=yaml.FullLoader)
    vbox_files = glob('{}/**/*.dat'.format(vbox_dir), recursive=True)
    gmug_files = glob('{}/**/*.dat'.format(gmug_dir), recursive=True)
    vbox_files.sort()
    gmug_files.sort()
    # Big loop over gmug files (10-min length)
    clock_correct = []  # Save previous clock corrections if ccc too low
    for gmu_f in gmug_files:
        print('GMuG file: {}'.format(gmu_f))
        st_gmug = gmug_to_stream(gmu_f.rstrip('.dat'), gmug_param)
        # Change delta to account for slightly shorter samp relative to vbox
        for tr in st_gmug:
            tr.stats.delta *= 1.00002808
        for i, vb_f in enumerate(which_vbox_files(st_gmug, vbox_files)):
            fname = vb_f.split('/')[-1].replace(
                '.dat', '.h5').replace('vbox', 'FSB')
            name = os.path.join(outdir, fname)
            if os.path.exists(name) and not overwrite:
                print('{} already exists'.format(name))
                continue
            print('Vibbox file: {}'.format(vb_f))
            st_vbox = vibbox_read(vb_f, dug_params)
            if debug > 1:  # Don't plot these large wavs unless necessary
                vbox_B81 = st_vbox.select(station='B81')
                vbox_B81[0].stats.network = 'MT'
                st_B81 = st_gmug.select(station='B81') + vbox_B81
                st_B81.plot(method='full', equal_scale=False)
            which = 0
            if st_vbox[0].stats.starttime < st_gmug[0].stats.starttime:
                which = -1
            # Clock correct
            cc = cassm_clock_correct(
                    gmug_tr=st_gmug.select(station='B81')[0],
                    vbox_tr=st_vbox.select(station='B81')[0],
                    trig_tr=st_vbox.select(station='CTrg')[0],
                    which=which, debug=debug, name=name)
            if np.max(cc[1]) < 0.75:
                if which == 0:
                    which = -1
                else:
                    which = 0
                try:  # Try the oppostie cassm shot in case its higher amp
                    cc = cassm_clock_correct(
                        gmug_tr=st_gmug.select(station='B81')[0],
                        vbox_tr=st_vbox.select(station='B81')[0],
                        trig_tr=st_vbox.select(station='CTrg')[0],
                        which=which, debug=debug, name=name)
                except Exception as e: # For vbox at end or beginning of gmug wav
                    print(e)
            clock_correct.append(cc)
            # Correct the starttime
            # Find most recent high ccc value
            inds = [j for j, c in enumerate(clock_correct)
                    if np.max(c[1]) > 0.75]
            correct = clock_correct[np.max(inds)]
            print('Clock correction: {}'.format(correct))
            for tr in st_gmug:
                tr.stats.starttime -= correct[0]
            # Shitty checks on the start and end times
            if st_vbox[0].stats.starttime < st_gmug[0].stats.starttime:
                all_strt = st_gmug[0].stats.starttime
                all_end = st_vbox[0].stats.endtime
            elif (st_vbox[0].stats.starttime > st_gmug[0].stats.starttime
                  and st_vbox[0].stats.endtime > st_gmug[0].stats.endtime):
                all_strt = st_vbox[0].stats.starttime
                all_end = st_gmug[0].stats.endtime
            else:
                all_strt = st_vbox[0].stats.starttime
                all_end = st_vbox[0].stats.endtime
            # Slice both to appropriate time range
            slice_gmug = st_gmug.slice(starttime=all_strt,
                                       endtime=all_end).copy()
            slice_vbox = st_vbox.trim(starttime=all_strt, endtime=all_end)
            if debug > 0:
                vbox_B81 = slice_vbox.copy().select(station='B81').slice(
                    starttime=clock_correct[-1][-1],
                    endtime=clock_correct[-1][-1] + 0.01)
                vbox_B81[0].stats.network = 'MT'
                st_B81 = slice_gmug.copy().select(station='B81').slice(
                    starttime=correct[-1], endtime=correct[-1] + 0.01) + vbox_B81
                st_B81.plot(method='full', equal_scale=False,
                            outfile=name.replace('.h5', 'corrected.png'))
                plt.close('all')
            # Deselect AE sensors on vibbox
            slice_vbox = slice_vbox.select(station='[BCP][34567TEPM]*')
            st_all = slice_gmug + slice_vbox
            # Write it out
            print('Writing {}'.format(name))
            with pyasdf.ASDFDataSet(name, compression='gzip-3') as asdf:
                asdf.add_stationxml(inventory)
                asdf.add_waveforms(st_all, tag='raw_recording')
    return