#!/usr/bin/python

"""
Functions for reading/writing and processing waveform data
"""

# import matplotlib
# matplotlib.use('Agg')

import os
import copy

import scipy
import itertools
import yaml
import joblib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from glob import glob
from copy import deepcopy
from itertools import cycle
from datetime import timedelta, datetime
from joblib import Parallel, delayed
from obspy import read, Stream, Catalog, UTCDateTime, Trace, ObsPyException
from obspy.core.event import ResourceIdentifier
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal import PPSD
from obspy.signal.spectral_estimation import get_nhnm, get_nlnm
from obspy.signal.rotate import rotate2zne
from obspy.signal.cross_correlation import xcorr_pick_correction
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException, FDSNException
from eqcorrscan.core.match_filter import Tribe, Party, MatchFilterError
from eqcorrscan.core.match_filter.matched_filter import match_filter
from eqcorrscan.core.match_filter.matched_filter import get_stream_xcorr, multi_find_peaks
from eqcorrscan import Family, Detection
from eqcorrscan.core.template_gen import template_gen
from eqcorrscan.utils.pre_processing import shortproc, dayproc, _prep_data_for_correlation
from eqcorrscan.utils.stacking import align_traces
from eqcorrscan.utils import clustering
from eqcorrscan.utils.mag_calc import dist_calc
from eqcorrscan.utils.plotting import _match_filter_plot, detection_multiplot
from scipy.stats import special_ortho_group, median_absolute_deviation
from scipy.signal import find_peaks
from scipy import fftpack
from scipy.io import savemat
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

try:
    import pyasdf
    from pyasdf import WaveformNotInFileException
except:
    print('No pyasdf in this env')

# Local imports
try:
    from surf_seis.vibbox import vibbox_read, vibbox_preprocess
except:
    print('No surf_seis on this machine')

from lbnl.instruments import modify_SAULN_inventory

extra_stas = ['CMon', 'CTrig', 'CEnc', 'PPS']

three_comps = ['OB13', 'OB15', 'OT16', 'OT18', 'PDB3', 'PDB4', 'PDB6', 'PDT1',
               'PSB7', 'PSB9', 'PST10', 'PST12']

# Rough +/- 1 degree borehole orientations
borehole_dict = {'OB': [356., 62.5], 'OT': [359., 83.], 'PDB': [259., 67.],
                 'PDT': [263., 85.4], 'PSB': [260., 67.], 'PST': [265., 87.]}

cascadia_colors = {'NSMTC.B1': '#c6dbef', 'NSMTC.B2': '#6aaed6',
                   'NSMTC.B3': '#2070b4', 'NSMTC.G1': '#956cb4',
                   'NSMTC.G2': '#d65f5f', 'PGC.': '#ee854a',
                   'B011.': '#6acc64'}


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def read_rosemanowes_segy(segy_file):
    """
    Read the crap seg-y data from rosemanowes

    ..note Need to be in a modified env with edited obspy (to correct the trace
        header values)

    ..note GN* For 4 kHz sampling, optical accelerometers
    """

    rmws_ids = ['RMNW.REF1..GN1', 'RMNW.REF1..GN2', 'RMNW.REF1..GNZ',
                'RMNW.REF2..GN1', 'RMNW.REF2..GN2', 'RMNW.REF2..GNZ',
                'RMNW.REF3..GN1', 'RMNW.REF3..GN2', 'RMNW.REF3..GNZ',
                'RMNW.US04..GN1', 'RMNW.US04..GN2', 'RMNW.US04..GNZ',
                'RMNW.US05..GN1', 'RMNW.US05..GN2', 'RMNW.US05..GNZ',
                'RMNW.US06..GN1', 'RMNW.US06..GN2', 'RMNW.US06..GNZ',
                ]

    st = read(segy_file, unpack_trace_headers=True,
              textual_header_encoding='EBCDIC', format='SEGY')
    # Grab start time out of textual header (who knows why...)
    starttime = str(st.stats.textual_file_header).split()[-42]
    print(starttime)
    starttime = UTCDateTime(starttime)
    for i, tr in enumerate(st):
        tr.id = rmws_ids[i]
        tr.stats.starttime = starttime
    return st


def read_raw_wavs(wav_dir, event_type='MEQ'):
    """Read all the waveforms in the given directory to a dict"""
    mseeds = glob('{}/*'.format(wav_dir))
    wav_dict = {}
    for ms in mseeds:
        if event_type == 'MEQ':
            eid = ms.split('/')[-1].rstrip('_raw.mseed')
        elif event_type == 'CASSM':
            eid = ms.split('/')[-1].rstrip('_raw.mseed')[:-6]
            eid = eid.lstrip('cassm_')
        else:
            print('Invalid event type')
            return
        try:
            wav_dict[eid] = read(ms)
        except TypeError as e:
            print(e)
    return wav_dict


def _check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return


def clean_daylong(stream):
    """
    Convenience func to clean out traces that will raise Exceptions in
    EQcorrscan preprocessing functions (e.g. too many zeros and too short)
    :return:
    """
    rmtrs = []
    for tr in stream:
        if len(np.nonzero(tr.data)[0]) < 0.5 * len(tr.data):
            print('{} mostly zeros. Removing'.format(tr.id))
            rmtrs.append(tr)
            continue
        if tr.stats.endtime - tr.stats.starttime < 0.8 * 86400:
            print('{} less than 80 percent daylong. Removing'.format(tr.id))
            rmtrs.append(tr)
            continue
        # Check for spikes
        if (tr.data > 2 * np.max(np.sort(
                np.abs(tr.data))[0:int(0.99 * len(tr.data))]
                                 ) * 1e7).sum() > 0:
            print('{} is spiky. Removing'.format(tr.id))
            rmtrs.append(tr)
    for rt in rmtrs:
        stream.traces.remove(rt)
    return stream


def downsample_mseeds(wavs, samp_rate, start, end, outdir):
    """
    Loop a list of miniseed files, downsample, and save to new files.

    Intended for use with SA-ULN, looking at long-period trends

    :param wavs:
    :return:
    """
    st = Stream()
    for date in date_generator(start, end):
        dto = UTCDateTime(date)
        dwavs = [w for w in wavs
                 if '{}.{:03d}'.format(dto.year, dto.julday) in w]
        if len(dwavs) == 0:
            print('No waveforms for {}.{}'.format(dto.year, dto.julday))
            continue
        dwavs.sort()
        new_name = os.path.basename(dwavs[0]).rstrip('.ms') + '_1Hz.ms'
        new_name = new_name.replace('.CN1', '')
        outfile = '{}/{}'.format(outdir, new_name)
        if os.path.exists(outfile):
            print('{} already written'.format(outfile))
            st += read(outfile)
            continue
        tmp_st = Stream()
        for w in dwavs:
            print('Reading {}'.format(w))
            tmp_st += read(w)
        starttime = tmp_st[0].stats.starttime.date
        tmp_st.merge()
        try:
            print('Processing {}'.format(date))
            down_st = dayproc(
                st=tmp_st, samp_rate=samp_rate, starttime=starttime,
                lowcut=0., highcut=0.4, filt_order=3)
            print('Writing {}'.format(new_name))
            down_st.write(outfile, format="MSEED")
            st += down_st
        except (NotImplementedError, ValueError) as e:
            print(e)
            continue
    return st


def combine_ppsds(npz_dir, netstalocchans, outdir, inventory=None):
    """
    Combine a number of npz files (daylong) into one large one

    :param npz_dir: Root for npz files
    :param netstalocchans: List of net.sta.loc.chan for get_stations()
    :return:
    """
    cli = Client('IRIS')
    bulk = [n.split('.') for n in netstalocchans]
    for b in bulk:
        b.extend([UTCDateTime(2019, 2, 1), UTCDateTime(2019, 3, 31)])
    if not inventory:
        inventory = cli.get_stations_bulk(bulk, level='response')
        inventory = modify_SAULN_inventory(inventory)
    for nsl in netstalocchans:
        print('Combining {}'.format(nsl))
        wavs = glob(
            '/bear0-data/GMF_1/Cascadia/Chet/mseed/**/{}*.ms'.format(nsl),
            recursive=True)
        if len(wavs) == 0:
            continue
        st = read(wavs[0])
        # Deal with shitty CN sampling rates
        for tr in st:
            if not ((1 / tr.stats.delta).is_integer() and
                    tr.stats.sampling_rate.is_integer()):
                tr.stats.sampling_rate = round(tr.stats.sampling_rate)
        try:
            ppsd = PPSD(stats=st[0].stats, metadata=inventory)
            ppsd.add_npz(filename='{}/{}*.npz'.format(npz_dir, nsl))
        except AssertionError as e:
            print(e)
            continue
        if not os.path.isdir(outdir):
            os.makedirs(os.path.join(outdir, 'ppsds'))
            os.makedirs(os.path.join(outdir, 'plots'))
        ppsd.save_npz('{}/ppsds/{}_combined.npz'.format(outdir, nsl))
        ppsd.plot(filename='{}/plots/{}_combined.png'.format(outdir, nsl),
                  show_earthquakes=(0, 1.5, 10), xaxis_frequency=True,
                  show_noise_models=False)
    return


def calculate_ppsds(netstalocchans, wav_dir, date_range, outdir, inventory=None,
                    cores=8):
    """
    Crawl a waveform directory structure and calculate ppsds for each file.

    In this case, the files are daylong. We'll make a PPSD for each and
    save the matrices as well

    :param netstalocchans: List of net.sta.loc.chan for
    :param wav_dir: Path to root waveform directory
    :param date_range: (start, end)
    :param outdir: Output directory for both numpy arrays of PPSD and plots
    :param inventory: Optional inventory object, otherwise pulls from fdsn
    :param cores: Number of core to run this on

    :return:
    """
    # Get the raw inventory and then modify the resp information
    cli = Client('IRIS')
    if not inventory:
        bulk = [n.split('.') for n in netstalocchans]
        for b in bulk:
            b.extend([UTCDateTime(date_range[0]), UTCDateTime(date_range[1])])
        inventory = cli.get_stations_bulk(bulk, level='response')
        inventory = modify_SAULN_inventory(inventory)
    results = Parallel(n_jobs=cores, verbose=10)(
        delayed(ppsd_channel_loop)(nsl, date_range, outdir, inventory, wav_dir)
        for nsl in netstalocchans)
    return

def ppsd_channel_loop(nsl, date_range, outdir, inventory, wav_dir):
    print('Running station {}'.format(nsl))
    nsl_split = nsl.split('.')
    for date in date_generator(date_range[0], date_range[1]):
        f = '{}/{}/{}/{}/{}/{}.{}.{}.{}.{}.{:03d}.ms'.format(
            wav_dir, date.year, nsl_split[0], nsl_split[1],
            nsl_split[3], nsl_split[0], nsl_split[1],
            nsl_split[2], nsl_split[3], date.year,
            UTCDateTime(date).julday)
        print('Calculating {}'.format(f))
        root_name = os.path.basename(f).rstrip('.ms')
        # Check if output file exists
        out_file = '{}/ppsds/{}.npz'.format(outdir, root_name)
        if os.path.isfile(out_file):
            print('{} already exists. Skipping'.format(out_file))
            continue
        try:
            st = read(f)
        except FileNotFoundError:
            print('{} doesnt exist'.format(f))
            continue
        # Deal with shitty CN sampling rates
        for tr in st:
            if not ((1 / tr.stats.delta).is_integer() and
                    tr.stats.sampling_rate.is_integer()):
                tr.stats.sampling_rate = round(tr.stats.sampling_rate)
        lil_ppsd = PPSD(st[0].stats, inventory)
        flag = lil_ppsd.add(st)
        if not flag:
            print('Failed to add {}'.format(f))
            continue
        lil_ppsd.save_npz('{}/ppsds/{}.npz'.format(outdir, root_name))
        lil_ppsd.plot(filename='{}/plots/{}.png'.format(outdir, root_name),
                      show_earthquakes=(0, 1.5, 10), xaxis_frequency=True,
                      show_noise_models=False)
    return

def get_IRIS_waveforms(start_date, end_date, inventory, output_root):
    """
    Iterate over date range, pull IRIS waveforms pertaining to the obspy
    inventory provided, and output into a directory structure

    :param start_date: datetime.datetime start of data
    :param end_date: datetime.datetime end of data
    :param inventory: Obspy.core.Inventory for desired stations
    :param output_root: Root output directory
    :return:
    """
    client = Client('IRIS')
    for date in date_generator(start_date, end_date):
        year = str(date.year)
        _check_dir(os.path.join(output_root, year))
        print('Retrieving: {}'.format(date))
        jday = UTCDateTime(date).julday
        # If no directory
        t2 = UTCDateTime(date) + 86400.
        for net in inventory:
            _check_dir(os.path.join(output_root, year, net.code))
            for sta in net.stations:
                _check_dir(os.path.join(output_root, year, net.code,
                                        sta.code))
                used_chans = []
                for chan in sta.channels:
                    loc = chan.location_code
                    if '{}.{}'.format(loc, chan.code) in used_chans:
                        continue
                    _check_dir(os.path.join(output_root, year, net.code,
                                            sta.code, chan.code))
                    fname = '{}.{}.{}.{}.{}.{:03d}.ms'.format(
                        net.code, sta.code, loc, chan.code, year, jday)
                    out_path = os.path.join(output_root, year, net.code,
                                            sta.code, chan.code, fname)
                    if os.path.isfile(out_path):
                        print('{} already exists'.format(out_path))
                        continue
                    bulk = [(net.code, sta.code, loc, chan.code,
                             UTCDateTime(date), t2)]
                    try:
                        print('Making request for {}'.format(bulk))
                        st = client.get_waveforms_bulk(bulk)
                        print(st)
                    except (FDSNNoDataException, ConnectionResetError,
                            FDSNException) as e:
                        print(e)
                        continue
                    try:
                        print('Writing {}'.format(out_path))
                        st.select(location=loc,
                                  channel=chan.code).write(out_path,
                                                           format="MSEED")
                        used_chans.append('{}.{}'.format(loc, chan.code))
                    except ObsPyException as e:
                        print(e)
                        continue
    return


def write_event_mseeds(wav_root, catalog, outdir, pre_pick=60.,
                       post_pick=120.):
    """
    Cut event waveforms from daylong mseed for catalog. Will cut the same
    time window for all available channels. Saved waveforms will be used for
    obspycking.

    :param wav_root: Waveform root directory
    :param catalog: Catalog of events
    :param outdir: Output directory for cut waveforms
    :param pre_origin: Seconds before the origin time to clip
    :param post_origin: Seconds after the origin time to clip

    :return:
    """
    # Ensure catalog sorted (should be by default?)
    try:
        catalog.events.sort(key=lambda x: x.preferred_origin().time)
        # Define catalog start and end dates
        cat_start = catalog[0].preferred_origin().time.date
        cat_end = catalog[-1].preferred_origin().time.date
    except AttributeError as e:  # In case this is a catalog of detections
        catalog.events.sort(key=lambda x: x.picks[0].time)
        # Define catalog start and end dates
        cat_start = catalog[0].picks[0].time.date
        cat_end = catalog[-1].picks[0].time.date
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        print('Processing events on {}'.format(dto))
        # Establish which events are in this day
        tmp_cat = Catalog(
            events=[ev for ev in catalog
                    if dto <= ev.picks[0].time < dto + 86400])
        if len(tmp_cat) == 0:
            print('No events on: %s' % str(dto))
            continue
        # Read all available channels for this julian day
        print('Reading wavs')
        chan_files = glob('{}/**/*{}.ms'.format(wav_root, dto.julday),
                          recursive=True)
        st = Stream()
        for chan_f in chan_files:
            st += read(chan_f)
        print('Merging')
        try:
            st.merge(fill_value='interpolate')
        except Exception as e:
            print(e)
            print('Skipping {}'.format(date))
            continue
        for ev in tmp_cat:
            try:
                fname = ev.resource_id.id.split('&')[-2].split('=')[-1]
            except IndexError:  # Case of non-usgs catalog, try Detection name
                fname = ev.resource_id.id
            print('Slicing ev {}'.format(fname))
            pt = min([pk.time for pk in ev.picks])
            st_slice = st.slice(starttime=pt - pre_pick,
                                endtime=pt + post_pick)
            st_slice.write('{}/{}.ms'.format(outdir, fname), format='MSEED')
    return


def tribe_from_client(catalog, **params):
    """
    Small wrapper for creating a tribe from client

    Basically just to catch merge exceptions and skip

    :param catalog: Catalog of events (we'll loop over days)
    :param params: Rest of kwargs for Tribe.construct

    :return:
    """
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    # Define catalog start and end dates
    cat_start = catalog[0].preferred_origin().time.date
    cat_end = catalog[-1].preferred_origin().time.date
    tribe = Tribe()
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        print('Date: {}'.format(dto))
        # Establish which events are in this day
        sch_str_start = 'time >= {}'.format(dto)
        sch_str_end = 'time <= {}'.format((dto + 86400))
        tmp_cat = catalog.filter(sch_str_start, sch_str_end)
        if len(tmp_cat) == 0:
            continue
        try:
            tribe += Tribe().construct(catalog=tmp_cat, **params)
        except Exception as e:
            # This is probs same trace id with diff samp rates
            print(e)
            continue
    return tribe


def tribe_from_catalog(catalog, wav_dir, param_dict, single_station=False,
                       stations='all'):
    """
    Loop over a catalog and return a tribe of templates for each. Assumes we're
    looking in a directory with day-long miniSEED files with filenames formatted
    as: net.sta.loc.chan.julday.ms

    :param catalog: Catalog of events with picks
    :param wav_dir: Root directory that will globbed recursively (/**/) for the
        format specified above
    :param param_dict: Dictionary containing all necessary parameters for
        template creation e.g. {'highcut': , 'lowcut': , 'corners': ,
                                'samp_rate': , 'prepick': , 'length': ,
                                'filt_order': , }
    :param single_station: Flag to tell function to make a Template for each
        single station for each event.
    :param stations: 'all' or a list of station names to include in templates.
        Valid for both single_station=T and F

    :return:
    """
    # Ensure catalog sorted (should be by default?)
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    # Define catalog start and end dates
    cat_start = catalog[0].preferred_origin().time.date
    cat_end = catalog[-1].preferred_origin().time.date
    tribe = Tribe()
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        print('Date: {}'.format(dto))
        # Establish which events are in this day
        sch_str_start = 'time >= {}'.format(dto)
        sch_str_end = 'time <= {}'.format((dto + 86400))
        tmp_cat = catalog.filter(sch_str_start, sch_str_end)
        jday = dto.julday
        seeds = list(set([pk.waveform_id.get_seed_string()
                          for ev in tmp_cat for pk in ev.picks]))
        seeds = [s for s in seeds if s.split('.')[1] in stations]
        wav_files = []
        for seed in seeds:
            try:
                wav_files.append(glob('{}/**/{}.{}.{:03d}.ms'.format(
                    wav_dir, seed, date.year, jday), recursive=True)[0])
            except IndexError:
                print('{} not in wavform directory'.format(seed))
                continue
        daylong = Stream()
        for wav_file in wav_files:
            daylong += read(wav_file)
        if len(daylong.traces) == 0:
            print('No waveforms for any picks in day catalog')
            continue
        # Deal with shitty CN sampling rates
        for tr in daylong:
            if not ((1 / tr.stats.delta).is_integer() and
                    tr.stats.sampling_rate.is_integer()):
                tr.stats.sampling_rate = round(tr.stats.sampling_rate)
        # Clean out stream of Exception traces
        clean_daylong(daylong)
        if len(daylong.traces) == 0:  # Run away if we've removed them all
            continue
        if single_station:
            for ev in tmp_cat:
                try:
                    name = ev.resource_id.id.split('&')[-2].split('=')[-1]
                except IndexError:
                    name = ev.resource_id.id.split('/')[-1]
                # Otherwise, make a stand-alone template for each station
                # with P pick (S will get included as well)
                netstalocs = [(pk.waveform_id.network_code,
                               pk.waveform_id.station_code,
                               pk.waveform_id.location_code)
                              for pk in ev.picks if pk.phase_hint == 'P']
                for nsl in netstalocs:
                    if type(stations) == list and nsl[1] not in stations:
                        continue
                    tmp_ev = ev.copy()
                    t_nm = name + '_{}'.format('.'.join(nsl))
                    tmp_ev.picks = [pk for pk in tmp_ev.picks
                                    if pk.waveform_id.network_code == nsl[0]
                                    and pk.waveform_id.station_code == nsl[1]
                                    and pk.waveform_id.location_code == nsl[2]]
                    tmp_ev.resource_id = ResourceIdentifier(id=t_nm)
                    tmp_ev = Catalog(events=[tmp_ev])
                    try:
                        trb = Tribe().construct(method='from_meta_file',
                                                st=daylong, meta_file=tmp_ev,
                                                **param_dict)
                    except IndexError as e:
                        print(e)
                        continue
                    trb.templates[0].name = t_nm
                    tribe += trb
        else:
            # Eliminate unwanted picks
            if type(stations) == list:
                for ev in tmp_cat:
                    ev.picks = [pk for pk in ev.picks
                                if pk.waveform_id.station_code in stations]
                    if len(ev.picks) == 0:
                        print('No waveforms for any picks in {}'.format(
                            ev.resource_id.id))
                        continue
            try:
                trb = Tribe().construct(method='from_meta_file', st=daylong,
                                        meta_file=tmp_cat,
                                        **param_dict)
            except (ValueError, TypeError) as e:
                print(e)
                continue
            tribe += trb
    return tribe


def extract_raw_tribe_waveforms(tribe, wav_dir, outdir, prepick, length):
    tribe.templates.sort(key=lambda x: x.event.origins[0].time)
    start = tribe[0].event.origins[-1].time.datetime.date()
    end = tribe[-1].event.origins[-1].time.datetime.date()
    net_sta_loc_chans = list(set([(pk.waveform_id.network_code,
                                   pk.waveform_id.station_code,
                                   pk.waveform_id.location_code,
                                   pk.waveform_id.channel_code)
                                  for temp in tribe
                                  for pk in temp.event.picks]))
    for date in date_generator(start, end):
        dto = UTCDateTime(date)
        day_trb = Tribe(
            templates=[t for t in tribe
                       if dto <= t.event.origins[0].time < dto + 86400])
        if len(day_trb.templates) == 0:
            continue
        jday = dto.julday
        print('Running {}\nJday: {}'.format(dto, jday))
        wav_files = ['{}/{}/{}/{}/{}/{}.{}.{}.{}.{}.{:03d}.ms'.format(
            wav_dir, date.year, nslc[0], nslc[1], nslc[3], nslc[0], nslc[1],
            nslc[2], nslc[3], date.year, jday) for nslc in net_sta_loc_chans]
        daylong = Stream()
        print('Reading wavs')
        for wav_file in wav_files:
            try:
                daylong += read(wav_file)
            except FileNotFoundError as e:
                print(e)
                continue
        # Deal with shitty sampling rates
        for tr in daylong:
            if not ((1 / tr.stats.delta).is_integer() and
                    tr.stats.sampling_rate.is_integer()):
                tr.stats.sampling_rate = round(tr.stats.sampling_rate)
        for temp in day_trb:
            name = temp.name
            print('Extracting {}'.format(name))
            o = temp.event.origins[-1]
            print(o.time)
            wav_slice = daylong.slice(starttime=o.time - prepick,
                                      endtime=o.time - prepick + length)
            # Write event waveform
            outwav = '{}/{}.ms'.format(outdir, name)
            print('Writing {}'.format(outwav))
            wav_slice.write(outwav, format='MSEED')
    return


def detect_tribe_client(tribe, client, start, end, param_dict,
                        daylong_dir=None):
    """
    Run detect for tribe on specified wav client

    :param tribe:
    :param client:
    :param start:
    :param end:
    :param param_dict: Params necessary for running detect from client
    :return:
    """
    import logging

    logging.basicConfig(
        filename='tribe-detect_run.txt',
        level=logging.WARNING,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    party = Party()
    for date in date_generator(start.datetime, end.datetime):
        print('Running detect: {}'.format(date))
        try:
            if 'return_stream' in param_dict:
                if param_dict['return_stream'] == True and not daylong_dir:
                    print('Specify daylong dir if you want to save streams')
                    return
                elif param_dict['return_stream'] == True and daylong_dir:
                    day_party, day_st = tribe.client_detect(
                        client=client, starttime=UTCDateTime(date),
                        endtime=UTCDateTime(date) + 86400,
                        **param_dict)
                    if not os.path.exists(daylong_dir):
                        os.makedirs(daylong_dir)
                    # Ensure there are no masked arrays
                    for tr in day_st:
                        if isinstance(tr.data, np.ma.masked_array):
                            tr.data = tr.data.filled()
                    day_st.write('{}/{}.mseed'.format(daylong_dir, date),
                                 format='MSEED')
            else:
                day_party = tribe.client_detect(
                    client=client, starttime=UTCDateTime(date),
                    endtime=UTCDateTime(date) + 86400,
                    **param_dict)
            party += day_party
        except (OSError, IndexError, MatchFilterError, Exception) as e:
            # Any number of lame pre-processing errors, usually
            logging.exception(e)
            continue
    return party


def detect_tribe(tribe, wav_dir, start, end, param_dict):
    """
    Run matched filter detection on a tribe of Templates over waveforms in
    a waveform directory (formatted as in above Tribe construction func)

    :param tribe: Tribe of Templates
    :param wav_dir: Root directory that will globbed recursively (/**/) for the
        format specified above
    :param start_date: Start UTCDateTime object
    :param end_date: End UTCDateTime object
    :param param_dict: Dict of parameters to pass to Tribe.detect()
    :return:
    """
    import logging

    logging.basicConfig(
        filename='tribe-detect_run.txt',
        level=logging.ERROR,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    party = Party()
    net_sta_loc_chans = list(set([(pk.waveform_id.network_code,
                                   pk.waveform_id.station_code,
                                   pk.waveform_id.location_code,
                                   pk.waveform_id.channel_code)
                                  for temp in tribe
                                  for pk in temp.event.picks]))
    for date in date_generator(start.datetime, end.datetime):
        dto = UTCDateTime(date)
        jday = dto.julday
        print('Running {}\nJday: {}'.format(dto, jday))
        wav_files = []
        for nslc in net_sta_loc_chans:
            day_wav_fs = glob('{}/{}/{}/{}/**/{}.{}.{}.{}.{}.{:03d}.ms'.format(
                wav_dir, date.year, nslc[0], nslc[1], nslc[0], nslc[1],
                nslc[2], nslc[3], date.year, jday),
                              recursive=True)
            wav_files.extend(day_wav_fs)
        daylong = Stream()
        print('Reading wavs')
        for wav_file in wav_files:
            daylong += read(wav_file)
        # Deal with shitty CN sampling rates
        for tr in daylong:
            if not ((1 / tr.stats.delta).is_integer() and
                    tr.stats.sampling_rate.is_integer()):
                tr.stats.sampling_rate = round(tr.stats.sampling_rate)
        daylong = clean_daylong(daylong.merge(fill_value='interpolate'))
        print('Running detect')
        try:
            party += tribe.detect(stream=daylong, **param_dict)
        except (OSError, IndexError, MatchFilterError) as e:
            print(e)
            continue
    return party


def detect_tribe_h5(tribe, wav_dir, start, end, param_dict):
    """
    Run matched filter detection on a tribe of Templates over waveforms
    saved in asdf h5 format (FSB Vibbox-specific naming conventions)

    :param tribe: Tribe of Templates
    :param wav_dir: Root directory that will globbed
    :param start_date: Start UTCDateTime object
    :param end_date: End UTCDateTime object
    :param param_dict: Dict of parameters to pass to Tribe.detect()
    :return:
    """
    import logging

    logging.basicConfig(
        filename='tribe-detect_run.txt',
        level=logging.DEBUG,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    fam_dict = {t.name: Family(template=t) for t in tribe.templates}
    # Grab all the necessary files
    h5s = glob('{}/*.h5'.format(wav_dir))
    h5s.sort()
    # Establish list of needed stations
    stas = list(set([tr.stats.station for temp in tribe for tr in temp.st]))
    for h5 in h5s:
        continuous = Stream()
        filestart = datetime.strptime(
            h5.split('_')[-1].rstrip('.h5'), '%Y%m%d%H%M%S%f')
        if filestart < start or filestart > end:
            continue
        # Grab only the stations in the templates
        with pyasdf.ASDFDataSet(h5) as ds:
            for sta in ds.waveforms:
                try:
                    if sta.StationXML[0][0].code in stas:
                        try:
                            continuous += sta.raw_recording
                        except WaveformNotInFileException:
                            continue
                except AttributeError:  # Trigger traces
                    continue
        # Process
        continuous = shortproc(
            continuous, highcut=tribe[0].highcut, lowcut=tribe[0].lowcut,
            samp_rate=tribe[0].samp_rate, filt_order=tribe[0].filt_order)
        print('Running detect on {}'.format(h5))
        try:
            # Go lower level to get to epoch arg
            templates = [t.st.copy() for t in tribe]
            _template_names = [t.name for t in tribe]
            # All this just to not force_epoch
            stream, templates, _template_names = _prep_data_for_correlation(
                stream=continuous, templates=templates,
                template_names=_template_names, force_stream_epoch=False)
            if len(templates) == 0:
                raise IndexError("No matching data")
            multichannel_normxcorr = get_stream_xcorr(None, None)
            [cccsums, no_chans, chans] = multichannel_normxcorr(
                templates=templates, stream=stream, **param_dict)
            if len(cccsums[0]) == 0:
                raise MatchFilterError('Correlation has not run, zero length cccsum')
            detections = []
            thresholds = [param_dict['threshold'] * np.median(np.abs(cccsum))
                          for cccsum in cccsums]
            all_peaks = multi_find_peaks(
                arr=cccsums, thresh=thresholds, parallel=True,
                trig_int=int(param_dict['trig_int'] *
                             stream[0].stats.sampling_rate),
                full_peaks=False, cores=param_dict['cores'])
            for i, cccsum in enumerate(cccsums):
                # Set up a trace object for the cccsum as this is easier to plot and
                # maintains timing
                if param_dict['plot']:
                    plotdir = param_dict['plotdir']
                    _match_filter_plot(
                        stream=stream, cccsum=cccsum,
                        template_names=_template_names,
                        rawthresh=thresholds[i], plotdir=plotdir,
                        plot_format='png', i=i)
                if all_peaks[i]:
                    print("Found {0} peaks for template {1}".format(
                          len(all_peaks[i]), _template_names[i]))
                    for peak in all_peaks[i]:
                        detecttime = (
                                stream[0].stats.starttime +
                                peak[1] / stream[0].stats.sampling_rate)
                        detection = Detection(
                            template_name=_template_names[i],
                            detect_time=detecttime, no_chans=no_chans[i],
                            detect_val=peak[0], threshold=thresholds[i],
                            typeofdet='corr', chans=chans[i],
                            threshold_type=param_dict['threshold_type'],
                            threshold_input=param_dict['threshold'])
                        detections.append(detection)
                        # Plot detection plot if asked for
                        if param_dict['plot']:
                            background = stream.slice(
                                starttime=detecttime - 0.005,
                                endtime=detecttime + 0.02)
                            filename = '{}{}_{}.png'.format(
                                plotdir, _template_names[i], detecttime)
                            detection_multiplot(
                                stream=background, template=templates[i],
                                times=[detecttime], show=False,
                                save=True, savefile=filename)
                else:
                    print("Found 0 peaks for template {0}".format(
                          _template_names[i]))
        except (OSError, IndexError, MatchFilterError) as e:
            print(e)
            continue
        # Place each Detection in it's proper family
        for d in detections:
            fam_dict[d.template_name] += d
    # Now make party
    party = Party(families=[f for t, f in fam_dict.items()])
    return party


def party_multiplot_h5(party, h5_dir, plotdir, start, end):
    # Grab all the necessary files
    h5s = glob('{}/*.h5'.format(h5_dir))
    h5s.sort()
    # Establish list of needed stations
    stas = list(set([tr.stats.station for fam in party
                     for tr in fam.template.st]))
    template_dict = {f.template.name: f.template.st for f in party}
    for h5 in h5s:
        print('Running file: {}'.format(h5))
        continuous = Stream()
        filestart = datetime.strptime(
            h5.split('_')[-1].rstrip('.h5'), '%Y%m%d%H%M%S%f')
        if filestart < start or filestart > end:
            continue
        file_end = filestart + timedelta(seconds=32.)  # roughly...
        # Get all detections in this file
        detections = [d for f in party for d in f
                      if filestart <= d.detect_time <= file_end]
        if len(detections) == 0:
            print('No detections')
            continue
        # Grab only the stations in the templates
        with pyasdf.ASDFDataSet(h5) as ds:
            for sta in ds.waveforms:
                try:
                    if sta.StationXML[0][0].code in stas:
                        try:
                            continuous += sta.raw_recording
                        except WaveformNotInFileException:
                            continue
                except AttributeError:  # Trigger traces
                    continue
        # Process
        continuous = shortproc(
            continuous, highcut=party[0].template.highcut,
            lowcut=party[0].template.lowcut,
            samp_rate=party[0].template.samp_rate,
            filt_order=party[0].template.filt_order)
        for det in detections:
            background = continuous.slice(
                starttime=det.detect_time - 0.005,
                endtime=det.detect_time + 0.015)
            filename = '{}{}.png'.format(
                plotdir, det.id)
            detection_multiplot(
                stream=background, template=template_dict[det.template_name],
                times=[det.detect_time], show=False,
                save=True, savefile=filename)
    return

def party_multiplot_wavdir(party, wav_dir, plotdir, start, end):
    """
    Plot templates over data at each detection time in a party

    :param party: eqcorrscan Party instance
    :param wav_dir: Path to waveform directory
    :param plotdir: Path to plot directory
    :param start: Start time
    :param end: End time
    :return:
    """
    # Template lookup dict
    template_dict = {f.template.name: f.template.st for f in party}
    for date in date_generator(start, end):
        print('Plotting for {}'.format(date))
        # Get all detections in this file
        detections = [(d, f.template) for f in party for d in f
                      if date <= d.detect_time <= date + timedelta(days=1)]
        if len(detections) == 0:
            print('No detections')
            continue
        fns = [os.path.exists('{}/{}.png'.format(plotdir, det[0].id))
               for det in detections]
        if all(fns):
            print('All plots saved. Next')
            continue
        detections = [d for d, f in zip(detections, fns) if not f]
        net_sta_loc_chans = list(set([(pk.waveform_id.network_code,
                                       pk.waveform_id.station_code,
                                       pk.waveform_id.location_code,
                                       pk.waveform_id.channel_code)
                                      for d in detections
                                      for pk in d[1].event.picks]))
        dto = UTCDateTime(date)
        jday = dto.julday
        print('Running {}\nJday: {}'.format(dto, jday))
        wav_files = ['{}/{}/{}/{}/{}/{}.{}.{}.{}.{}.{:03d}.ms'.format(
            wav_dir, date.year, nslc[0], nslc[1], nslc[3], nslc[0], nslc[1],
            nslc[2], nslc[3], date.year, jday) for nslc in net_sta_loc_chans]
        if not any([os.path.isfile(f) for f in wav_files]):
            print('No waveforms?')
            continue
        daylong = Stream()
        print('Reading wavs')
        for wav_file in wav_files:
            try:
                daylong += read(wav_file)
            except FileNotFoundError as e:
                print(e)
                continue
        # Deal with shitty CN sampling rates
        for tr in daylong:
            if not ((1 / tr.stats.delta).is_integer() and
                    tr.stats.sampling_rate.is_integer()):
                tr.stats.sampling_rate = round(tr.stats.sampling_rate)
        daylong = clean_daylong(daylong.merge(fill_value='interpolate'))
        temp = party[0].template
        daylong = dayproc(st=daylong, lowcut=temp.lowcut, highcut=temp.highcut,
                          samp_rate=temp.samp_rate, filt_order=temp.filt_order,
                          starttime=date)
        for det in detections:
            background = daylong.slice(
                starttime=det[0].detect_time - 10,
                endtime=det[0].detect_time + 40)
            filename = '{}/{}.png'.format(
                plotdir, det[0].id)
            print('Writing {}'.format(filename))
            detection_multiplot(
                stream=background, template=template_dict[det[0].template_name],
                times=[det[0].detect_time], show=False,
                save=True, savefile=filename)
    return


def party_lag_extract(party, wav_dir, out_dir, prepick=30, length=90,
                      shift_len=0.2, min_cc=0.6, process_cores=1, cores=8):
    """
    Perform lag_calc and extract raw wav snippets for each detection in a party

    :param party: EQcorrscan Party object
    :param wav_dir: Path to root waveform directory
    :param out_dir: Output directory for trimmed waveforms
    :param prepick: Length to extract before anticipated pick time
    :param length: Total length of waveform to extract for each trace
    :param shift_len: Seconds to allow lag_calc shift (absolute value)
    :param min_cc: Minimum correlation coefficient for picking
    :param process_cores: Cores used in preprocessing
    :param cores: Cores used in lag_calc algorithm (post processing)

    :return:
    """
    import logging

    logging.basicConfig(
        filename='lag_calc_run.txt',
        level=logging.ERROR,
        format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    dets = [det for fam in party for det in fam]
    dets.sort(key=lambda x: x.detect_time)
    repicked_cat = Catalog()
    for date in date_generator(dets[0].detect_time.datetime,
                               dets[-1].detect_time.datetime):
        dto = UTCDateTime(date)
        jday = dto.julday
        print('Running {}\nJday: {}'.format(dto, jday))
        wav_files = []
        day_dets = [d for d in dets if dto <= d.detect_time < dto + 86400]
        day_temps = [party.select(d.template_name).template.st
                     for d in day_dets]
        day_seeds = list(set([tr.id for t in day_temps for tr in t]))
        for seed in day_seeds:
            nslc = seed.split('.')
            f = '{}/{}/{}/{}/{}/{}.{}.{}.{}.{}.{:03d}.ms'.format(
                wav_dir, date.year, nslc[0], nslc[1], nslc[3], nslc[0],
                nslc[1], nslc[2], nslc[3], date.year, jday)
            wav_files.append(f)
        daylong = Stream()
        print('Reading wavs')
        for wav_file in wav_files:
            try:
                daylong += read(wav_file)
            except FileNotFoundError:
                print('{} doesnt exist'.format(wav_file))
                continue
        # Deal with shitty CN sampling rates
        for tr in daylong:
            if not ((1 / tr.stats.delta).is_integer() and
                    tr.stats.sampling_rate.is_integer()):
                tr.stats.sampling_rate = round(tr.stats.sampling_rate)
        daylong = clean_daylong(daylong.merge(fill_value='interpolate'))
        # Do the lag calc
        print('Lag calc-ing')
        repicked_cat += party.lag_calc(
            stream=daylong, pre_processed=False, shift_len=shift_len,
            min_cc=min_cc, plot=True, process_cores=process_cores, cores=cores)
        print('Writing waveforms')
        # Extract and write streams
        for d in day_dets:
            d_st = daylong.slice(starttime=d.detect_time - prepick,
                                 endtime=d.detect_time - prepick + length)
            d_st.write('{}/{}.ms'.format(out_dir, d.id), format='MSEED')
    return repicked_cat


def stack_CASSM_directory(path, length, plotdir=None):
    """
    Wrapper on stack_CASSM_shots to loop over all mseeds in directory
    :param path: Path to directory of 16-shot records from Todd
    :param plotdir: Path to optional plotting directory
    :return:
    """
    # Get all miniseeds
    vibbox_files = glob('{}/**/vbox_*.mseed'.format(path), recursive=True)
    for vbox in vibbox_files:
        dir_name = os.path.dirname(vbox)
        fname = 'CASSM_stack_{}.mseed'.format(
            vbox.split('/')[-1].split('.')[0][6:])
        shot = stack_CASSM_shots(read(vbox), length=length)
        shot.write(os.path.join(dir_name, fname), format='MSEED')
        if plotdir:
            shot.plot(equal_scale=False, show=False,
                      outfile=os.path.join(plotdir,
                                           fname.replace('mseed', 'png')),
                      dpi=200, size=(1600, 2400))
    return


def stack_CASSM_shots(st, length):
    """
    Stack all shots in a 16 shot sequence
    :param st: Stream containing all 16 shots and a trigger trace
    :param length: Length of the stacked signals in seconds

    :return: Stream object with all Traces
    """
    start = st[0].stats.starttime
    # Use derivative of PPS signal to find pulse start
    dt = np.diff(st.select(station='CTrg')[0].data)
    # Use 70 * MAD threshold
    trig_samps = np.where(
        dt > np.mean(dt) + 70 * median_absolute_deviation(dt))[0]
    t_samps = []
    # Select only first sample of pulse
    for i, t in enumerate(trig_samps):
        if i == 0:
            t_samps.append(t)
        elif trig_samps[i - 1] != t - 1:
            t_samps.append(t)
    # Remove CASSM and timing info
    st_sensors = st.select(station='B*')
    # Loop trigger onsets, slice, add to list
    shots = Stream()
    for samp in t_samps:
        samp_time = samp / st[0].stats.sampling_rate
        shots += st_sensors.slice(starttime=start + samp_time,
                                  endtime=start + samp_time + length)
    shots.stack(group_by='id', stack_type='linear')
    return shots


def svd_denoise(st, plot=False, plot_tr=None):
    """
    Remove first input singular vector from stream to eliminate
    electrical noise

    :param st: obspy Stream
    :param plot: Plot flag for denoising
    :param plot_tr: Station to plot if plotting
    :return:
    """
    stream_list = [st.select(station=tr.stats.station).copy() for tr in st]
    for str in stream_list:
        str[0].stats.station = 'XXX'
    u, s, v, stachans = clustering.svd(stream_list=stream_list, full=False)
    # Reweight the first singular vector
    noise_vect = np.dot(u[0][:,0] * s[0][0], v[0][0, 0])
    sv2 = np.dot(u[0][:,1] * s[0][1], v[0][1, 1])
    sv3 = np.dot(u[0][:,2] * s[0][2], v[0][2, 2])
    st_denoise = st.copy()
    for tr in st_denoise:
        tr.data -= noise_vect
    if plot:
        fig, axes = plt.subplots(nrows=6, sharex='col', figsize=(5, 10))
        time_vec = np.arange(st[0].data.shape[0]) / 200.
        plt_ref = st.select(station=plot_tr)[0].data
        plt_denoise = st_denoise.select(station=plot_tr)[0].data
        axes[0].plot(time_vec, plt_ref, color='k', label='Raw data')
        axes[1].plot(time_vec, noise_vect, color='r', label='First SV: reweighted')
        axes[1].plot(time_vec, sv2, color='b', label='Second SV: reweighted')
        axes[1].plot(time_vec, sv3, color='g', label='Third SV: reweighted')
        axes[2].plot(time_vec, plt_ref, alpha=0.6, color='k')
        axes[2].plot(time_vec, noise_vect, alpha=0.6, color='r')
        axes[3].plot(time_vec, plt_denoise, color='steelblue', label='Denoised')
        fig.legend()
        labs = []
        for i, tr in enumerate(st_denoise):
            labs.append(tr.stats.station)
            norm = tr.copy().data / np.max(tr.data)
            raw_norm = st.select(station=tr.stats.station)[0].copy().data
            raw_norm /= np.max(raw_norm)
            axes[4].plot(time_vec, raw_norm + i, color='black')
            axes[5].plot(time_vec, norm + i, color='darkgray')
        axes[4].set_yticks(np.arange(len(st_denoise.traces)))
        axes[4].set_yticklabels(labs)
        axes[5].set_yticks(np.arange(len(st_denoise.traces)))
        axes[5].set_yticklabels(labs)
        axes[5].set_xlabel('ms')
        fig.suptitle(plot_tr, fontsize=16)
        plt.show()
    return


def SNR(signal, noise, log=True):
    """
    Simple SNR calculation (in decibels)
    SNR = 20log10(RMS(sig) / RMS(noise))
    """
    sig_pow = np.sqrt(np.mean(signal ** 2))
    noise_pow = np.sqrt(np.mean(noise ** 2))
    if log:
        return 20 * np.log10(sig_pow / noise_pow)
    else:
        return sig_pow / noise_pow


def rotate_catalog_streams(catalog, wav_dir, inv, ncores=8, **kwargs):
    """
    Return a list of rotated streams and a single

    :param catalog: Catalog of events used to orient seismometers
    :param wav_dir: Directory of waveform files to draw from
    :param inv: Station inventory
    :param ncores: Number of cores to use, default 8.
    :param kwargs: Keyword arguments passed to uniform_rotate_stream
    :return:
    """
    # Mseed reading function for error handling
    def mseed_read(afile):
        try: # Catch effed up MSEED files inside dictcomp
            return read(afile)
        except TypeError:
            # Empty stream otherwise
            return Stream()
    mseeds = glob('{}/*'.format(wav_dir))
    wav_ids = [ms.split('/')[-1].split('_')[1] for ms in mseeds]
    eids = [e.resource_id.id.split('/')[-1] for e in catalog]
    mseeds = {m.split('/')[-1].split('_')[1]: mseed_read(m)
              for m in mseeds if m.split('/')[-1].split('_')[1] in eids
              and m.split('/')[-1].split('_')[1] in wav_ids}
    print('Starting pool')
    results = Parallel(n_jobs=ncores, verbose=10)(
        delayed(uniform_rotate_stream)(mseeds[e.resource_id.id.split('/')[-1]],
                                       e, inv, **kwargs)
        for e in catalog if e.resource_id.id.split('/')[-1] in mseeds)
    rot_streams, sta_dicts = zip(*results)
    sd = {}
    # Combine all the dictionaries into one
    print('Retrieving results')
    for d in sta_dicts:
        for k, v in d.items():
            if k not in sd.keys():
                sd[k] = {}
            for ks, vs in v.items():
                if ks not in sd[k].keys():
                    sd[k][ks] = [vs]
                else:
                    sd[k][ks].append(vs)
    return rot_streams, sd


def uniform_rotate_stream(st, ev, inv, rotation='rand', n=1000,
                          amp_window=0.0002, metric='ratio', debug=0):
    """
    Sample a uniform distribution of rotations of a stream and return
    the rotation and stream of interest

    :param st: Stream to rotate
    :param ev: Event with picks used to define the P arrival window
    :param inv: Obspy Inventory object
    :param rotation: Defaults to a random set of rotations but can be given any
        axis which will entail a regular grid of rotations around that axis.
    :param n: Number of samples to draw
    :param amp_window: Length (sec) within which to measure the energy of the
        trace.
    :param metric: Whether to use the 'ratio' of radial to transverse components
        or simply the energy in the 'radial' component to decide on the best
        rotation matrix for each station.

    .. note: One complication is that the arrival orientation corresponding to
        the greatest energy in the signal can be parallel to either the
        source-reciever path or the reciever-source path (i.e. backazimuth).
        To account for this, we test both possibilities by rotating both
        possible arrival vectors by the matrix that provided the highest
        energy, and selecting the one closest to normal to the borehole axis.

    :return:
    """
    # Catch empty stream
    if len(st) == 0:
        print('Stream empty')
        return Stream(), {}
    # Make array of station names
    stas = list(set([tr.stats.station for tr in st
                     if tr.stats.station in three_comps]))
    if rotation == 'rand':
        # Make array of uniformly distributed rotations to apply to stream
        rots = [Rotation.from_dcm(special_ortho_group.rvs(3))
                for i in range(n)]
    else:
        # Just do every degree...
        rots = [Rotation.from_euler(rotation, d, degrees=True)
                for d in range(360)]
    rot_streams = []
    # Create a stream for all possible rotations (this may be memory expensive)
    amp_dict = {}
    for sta in stas:
        rot_st = Stream()
        amp_dict[sta] = []
        # Handle case where no pick for this station
        # Grab the pick
        try:
            pk = [pk for pk in ev.picks
                  if pk.waveform_id.station_code == sta
                  and pk.phase_hint == 'P'][-1]
        except IndexError:
            # If no pick at this station, break the rotations loop
            continue
        work_st = st.select(station=sta).copy().detrend()
        # Bandpass
        work_st.filter(type='bandpass', freqmin=3000,
                       freqmax=42000, corners=3)
        # Trim to 0.01s window around pick (hardcoded for indexing convenience)
        work_st.trim(starttime=pk.time - 0.005,
                     endtime=pk.time + 0.005)
        # Leave this unrotated for debug plotting
        og_signal = work_st.copy().select(channel='*Y')[0].data[495:550]
        if len(work_st.select(channel='*X')) == 0:
            continue
        for R in rots:
            r_st = work_st.copy()
            datax = r_st.select(channel='*X')[0].data
            datay = r_st.select(channel='*Y')[0].data
            dataz = r_st.select(channel='*Z')[0].data
            # Rotate
            datax, datay, dataz = np.dot(R.as_dcm(), [datax, datay, dataz])
            # Noise
            noisey = datay[:301]
            # Signal
            signaly = datay[495:550]
            # Calc E as sum of squared amplitudes
            energy_int = int(amp_window * work_st[0].stats.sampling_rate)
            # Amplitude window starting at pick
            ampx = np.copy(datax[496:496+energy_int])
            ampy = np.copy(datay[496:496+energy_int])
            ampz = np.copy(dataz[496:496+energy_int])
            Ex = np.sum(ampx ** 2)
            Ey = np.sum(ampy ** 2)
            Ez = np.sum(ampz ** 2)
            h_ratio = Ey / (Ez + Ex)
            snry = SNR(signaly, noisey)
            # Normalize data
            poldat = ampy / np.linalg.norm(ampy)
            # Decide polarity of arrival
            pp = find_peaks(poldat, prominence=0.05, distance=5)
            pn = find_peaks(-poldat, prominence=0.05, distance=5)
            try:
                if pp[0][0] < pn[0][0]:
                    pol = 1 # up
                else:
                    pol = 0 # down
            except IndexError as e: # Case of no suitable peaks found
                continue
            amp_dict[sta].append([Ex, Ey, Ez, snry, h_ratio, pol, ampy,
                                  pp[0][0], pn[0][0], signaly, og_signal])
    sta_dict = {}
    if debug > 0:
        fig, axes = plt.subplots(ncols=2, figsize=(10, 12))
        labs = []
        ticks = []
    for i, (sta, amps) in enumerate(amp_dict.items()):
        # No picks at this station
        if len(amps) == 0:
            continue
        x, y, z, snr, rat, p, ampy, pp, pn, sigy, os = zip(*amps)
        if metric == 'radial':
            best_ind = np.argmax(y)
        elif metric == 'ratio':
            best_ind = np.argmax(rat)
        # Take Y, but could be Z (X is along borehole)
        sta_dict[sta] = {'matrix': rots[best_ind]}
        sta_dict[sta]['polarity'] = p[best_ind]
        sta_dict[sta]['datay'] = ampy[best_ind]
        sta_dict[sta]['ratio'] = rat[best_ind]
        sta_dict[sta]['snr'] = snr[best_ind]
        # Test plotting
        if debug > 0:
            labs.append(sta)
            ticks.append(i)
            plot_dat = ampy[best_ind]
            plot_dat /= np.max(plot_dat)
            rot_sig = sigy[best_ind] / np.max(sigy[best_ind])
            # Plot data
            axes[0].plot(plot_dat + i, color='k')
            # Plot polarity picks
            # Up pick
            axes[0].scatter(pp[best_ind], plot_dat[pp[best_ind]] + i,
                            marker='o', color='r')
            # Down pick
            axes[0].scatter(pn[best_ind], plot_dat[pn[best_ind]] + i,
                            marker='o', color='b')
            try:
                axes[1].plot(rot_sig + i, color='k')
                axes[1].plot((os[best_ind] / np.max(os[best_ind])) + i,
                             color='grey')
            except IndexError as e:
                continue
    if debug > 0:
        axes[0].set_yticklabels(labs)
        axes[0].set_yticks(ticks)
        axes[1].set_yticklabels(labs)
        axes[1].set_yticks(ticks)
    # Rotate a final stream so that all instruments have Y oriented radially
    radial_stream = Stream()
    for sta in stas:
        # Hack to get correct length well names
        if sta.startswith('O'):
            well_int = 2
        else:
            well_int = 3
        try:
            rot = sta_dict[sta]['matrix']
            p = sta_dict[sta]['polarity']
        except KeyError:
            # No P pick at this station
            continue
        work_st = st.select(station=sta).copy()
        datax = work_st.select(channel='*X')[0].data
        statx = work_st.select(channel='*X')[0].stats
        datay = work_st.select(channel='*Y')[0].data
        staty = work_st.select(channel='*Y')[0].stats
        dataz = work_st.select(channel='*Z')[0].data
        statz = work_st.select(channel='*Z')[0].stats
        # As select() passes references to traces, can modify in-place
        datax, datay, dataz = np.dot(rot.as_dcm(), [datax, datay, dataz])
        new_trx = Trace(data=datax, header=statx)
        new_try = Trace(data=datay, header=staty)
        new_trz = Trace(data=dataz, header=statz)
        rot_st = Stream(traces=[new_trx, new_try, new_trz])
        radial_stream += rot_st
        # Sort out original orientation. If polarity of max energy arrival
        # is positive, it's the radial vector. Otherwise it's the backazimuth.
        arrival_vect = az_toa_vect(inv.select(station=sta)[0][0],
                                   ev.origins[-1])
        if not p:
            continue
        if p == 0: # Negative polarity
            arrival_vect *= -1. # Other direction
        # Multiply with inverse rotation matrix
        orig_vect = np.dot(rot.inv().as_dcm(), arrival_vect.T)
        # orig_vect_baz = np.dot(rot.inv().as_dcm(), baz_vect.T)
        # Do some trig
        # This comes out as deg from (1, 0) vector (i.e. positive E)
        az = np.rad2deg(np.arctan2(orig_vect[1],
                                   orig_vect[0])) + 90.
        # az_baz = np.rad2deg(np.arctan2(orig_vect_baz[1],
        #                                orig_vect_baz[0])) + 90.
        # Will be degrees from horizontal (negative value is upgoing ray)
        dip = np.rad2deg(np.arccos(orig_vect[2])) - 90.
        # dip_baz = np.rad2deg(np.arccos(orig_vect_baz[2])) - 90.
        # Make borehole vector
        bhz = np.sin(np.deg2rad(borehole_dict[sta[:well_int]][1]))
        bhh = np.sqrt(1 - bhz ** 2)
        bhx = np.sin(np.deg2rad(borehole_dict[sta[:well_int]][0])) * bhh
        bhy = np.cos(np.deg2rad(borehole_dict[sta[:well_int]][0])) * bhh
        bh_vect = np.array([bhx, bhy, bhz])
        # Sort out which is most normal to borehole
        angle = np.rad2deg(np.arccos(np.dot(orig_vect, bh_vect)))
        # angle_baz = np.rad2deg(np.arccos(np.dot(orig_vect_baz, bh_vect)))
        sta_dict[sta]['orientation'] = (az, dip)
        sta_dict[sta]['bh angle'] = angle
        # sta_dict[sta]['orientation from backazimuth'] = (az_baz, dip_baz)
        # sta_dict[sta]['bh angle from backazimuth'] = angle_baz
    return radial_stream, sta_dict


def az_toa_vect(station, origin):
    """
    Returns a unit radial vector from the source to station
    """
    n_diff = float(station.extra.hmc_north.value) - float(origin.extra.hmc_north.value)
    e_diff = float(station.extra.hmc_east.value) - float(origin.extra.hmc_east.value)
    elev_diff = float(station.extra.hmc_elev.value) - float(origin.extra.hmc_elev.value)
    unit_vect = np.array([e_diff, n_diff, elev_diff])
    unit_vect /= np.linalg.norm(unit_vect)
    return unit_vect


def coords2bazinc(station, origin):
    """
    Yoinked from Obspyck, per usual...CJH 3-12-20

    Returns backazimuth and incidence angle from station coordinates
    and event location specified in hmc cartesian coordinate system
    """
    n_diff = (float(station.extra.hmc_north.value) -
              float(origin.extra.hmc_north.value))
    e_diff = (float(station.extra.hmc_east.value) -
              float(origin.extra.hmc_east.value))
    dist = np.sqrt(n_diff**2 + e_diff**2)
    baz = np.rad2deg(np.arctan2(e_diff, n_diff))
    if baz < 0:
        baz += 360.
    elev_diff = (float(station.extra.hmc_elev.value) -
                 float(origin.extra.hmc_elev.value))
    # Angle up from vertical down
    inci = np.rad2deg(np.arctan2(dist, elev_diff))
    return baz, inci


def rotate_stream_to_LQT(st, inventory, origin):
    """
    Rotate stream into LQT orientation w respect to given origin.

    :param invenory:
    :param origin:
    :return:
    """
    rot_st = Stream()
    zne_st = Stream()
    for sta in inventory[0]:
        if sta.code not in three_comps:
            rot_st += st.select(station=sta.code).copy()
            continue
        if len(st.select(station=sta.code)) == 0:
            continue
        baz, incidence = coords2bazinc(sta, origin)
        # First to ZNE
        sta_st_zne = st.copy().select(station=sta.code).rotate(
            method='->ZNE', components='ZXY', inventory=inventory)
        zne_st += sta_st_zne
        # Then to LQT
        sta_st_lqt = sta_st_zne.copy().rotate(
            method='ZNE->LQT', back_azimuth=baz, inclination=incidence)
        rot_st += sta_st_lqt
    return rot_st, zne_st


def which_server_vibbox(cat, file_list, outfile):
    """
    Output a list of files we need from the server

    :param cat: Catalog of events we need
    :param file_list: File containing a list of all vibbox files on the server
    :param outfile: Path to the output file of only the files we need to scp

    :return:
    """

    vibbox_files = []
    with open(file_list, 'r') as f:
        for ln in f:
            if ln.endswith('.dat\n'):
                vibbox_files.append(ln.rstrip('\n'))
    vibbox_files.sort()
    # Make list of ranges for vibbox files
    ranges = [(int(vibbox_files[i - 1].split('_')[-1][:14]),
               int(ln.split('_')[-1][:14]))
              for i, ln in enumerate(vibbox_files)]
    out_list = []
    for ev in cat:
        time_int = int(ev.origins[-1].time.strftime('%Y%m%d%H%M%S'))
        for i, r in enumerate(ranges):
            if r[0] < time_int < r[1] and vibbox_files[i - 1] not in out_list:
                out_list.append(vibbox_files[i - 1])
                break
    with open(outfile, 'w') as of:
        for out in out_list:
            of.write('{}\n'.format(out))
    return


def extract_CASSM_wavs(catalog, vibbox_dir, outdir, pre_ot=0.01, post_ot=0.02):
    """
    Extract waveforms from vibbox files for cassm sources and write to event
    files for later processing.

    :param cat: Catalog of cassm events
    :param vibbox_dir: Directory with waveform files
    :param pre_ot: Time before origin time to start wavs
    :param post_ot: Time after origin time to end wavs

    :return:
    """
    for ev in catalog:
        eid = ev.resource_id.id.split('/')[-1]
        ot = ev.origins[-1].time
        pk1 = min([pk.time for pk in ev.picks])
        print(ot)
        print(pk1)
        dir = '{}/{}'.format(vibbox_dir, ot.strftime('%Y%m%d'))
        day_files = glob('{}/*'.format(dir))
        day_files.sort()
        if len(day_files) == 1:
            print('Only one file in dir')
            print(ev.resource_id.id)
            print(day_files[0])
            st = vibbox_read(day_files[0])
        else:
            ranges = [(int(day_files[i - 1].split('_')[-1][:14]),
                       int(ln.split('_')[-1][:14]))
                      for i, ln in enumerate(day_files)]
            print(ev.resource_id.id)
            time_int = int(ev.origins[-1].time.strftime('%Y%m%d%H%M%S'))
            for i, r in enumerate(ranges):
                if r[0] < time_int < r[1]:
                    print(day_files[i - 1])
                    st = vibbox_read(day_files[i - 1])
                    if st[0].stats.starttime < ot < st[0].stats.endtime:
                        print('Origin time within stream\n')
                    else:
                        print('Origin time outside stream\n')
                    break
        if len(st) == 0:
            print('No file for this event...\n')
            continue
        # # De-median the traces (in place)
        # st = vibbox_preprocess(st)
        # Trim that baby
        try:
            st.trim(starttime=pk1 - pre_ot, endtime=pk1 + post_ot)
        except UnboundLocalError:
            continue
        st.write('{}/cassm_{}_raw.mseed'.format(outdir, eid), format='MSEED')
    return print('boom')


def extract_event_signal(wav_dir, catalog, prepick=0.0001, duration=0.01,
                         event_type='MEQ'):
    """
    Trim around pick times and filter waveforms

    :param wav_dir: Path to waveform files
    :param catalog: obspy.core.Catalog
    :param prepick: How many seconds do we want before the pick?
    :param duration: How long do we want the traces?
    :param event_type: Nameing convention is different for CASSM/MEQ.
        Which is being used here?

    :return:
    """
    streams = {}
    wav_dict = read_raw_wavs(wav_dir, event_type=event_type)
    for ev in catalog:
        ot = ev.origins[-1].time
        if event_type == 'MEQ':
            t_stamp = ot.strftime('%Y%m%d%H%M%S%f')
        elif event_type == 'CASSM':
            t_stamp = ot.strftime('%Y%m%d%H%M%S')
        try:
            st = wav_dict[t_stamp]
        except KeyError as e:
            print(t_stamp)
            print(e)
        # De-median the traces (in place)
        st = vibbox_preprocess(st)
        # Append only those wavs with picks
        new_st = Stream()
        for pk in ev.picks:
            sta = pk.waveform_id.station_code
            chan = pk.waveform_id.channel_code
            if len(st.select(station=sta, channel=chan)) == 0 or \
                    pk.phase_hint == 'S':
                continue
            st_sta = st.select(station=sta).copy()
            st_sta.trim(starttime=pk.time - prepick, endtime=pk.time + duration)
            new_st += st_sta
        if len(new_st) > 0:
            streams[t_stamp] = new_st
    return streams

def rotate_channels(st, inv):
    """
    Take unoriented stream and return it rotated into ZNE

    :param st: stream to be rotated
    :param inv: Inventory with pertinent channel orientations
    :return:
    """
    rotated_st = Stream()
    # Loop each station in inv and append to new st
    for sta in inv[0]:
        sta_st = st.select(station=sta.code)
        if len(sta_st) < 3 and len(sta_st) > 0:
            # Ignore hydrophones here
            rotated_st += sta_st
            continue
        elif len(sta_st) == 0:
            continue
        data1 = sta_st.select(channel='*Z')[0]
        dip1 = sta.select(channel='*Z')[0].dip
        az1 = sta.select(channel='*Z')[0].azimuth
        data2 = sta_st.select(channel='*X')[0]
        dip2 = sta.select(channel='*X')[0].dip
        az2 = sta.select(channel='*X')[0].azimuth
        data3 = sta_st.select(channel='*Y')[0]
        dip3 = sta.select(channel='*Y')[0].dip
        az3 = sta.select(channel='*Y')[0].azimuth
        rot_np = rotate2zne(data_1=data1.data, azimuth_1=az1, dip_1=dip1,
                            data_2=data2.data, azimuth_2=az2, dip_2=dip2,
                            data_3=data3.data, azimuth_3=az3, dip_3=dip3,
                            inverse=False)
        # Reassemble rotated stream
        # TODO Without renaming XYZ, just assuming user understands
        # TODO that X is North, Y is East....fix this by adding channels
        # TODO to inventory later!
        new_trZ = Trace(data=rot_np[0], header=data1.stats)
        # new_trZ.stats.channel = 'XNZ'
        new_trN = Trace(data=rot_np[1], header=data2.stats)
        # new_trN.stats.channel = 'XNN'
        new_trE = Trace(data=rot_np[2], header=data3.stats)
        # new_trE.stats.channel = 'XNE'
        rot_st = Stream(traces=[new_trZ, new_trN, new_trE])
        rotated_st += rot_st
    return rotated_st


def find_largest_SURF(wav_dir, catalog, method='avg', sig=2):
    """
    Find the largest-amplitude events for the SURF catalog

    :param wav_dir: path to eventfiles_raw (or similar)
    :param catalog: obspy.core.Catalog
    :param method: 'avg' or a station name to solely use
    :param sig: How many sigma to use as a minimum amplitude threshold
    :return:
    """
    stream_dict = extract_event_signal(wav_dir, catalog)
    amp_dict = {}
    for eid, st in stream_dict.items():
        if method == 'avg':
            avg = 0.
            for tr in st:
                avg += np.max(np.abs(tr.data))
            amp_dict[eid] = (avg / len(st))
        else:
            if len(st.select(station=method)) > 0:
                amp_dict[eid] = np.max(np.abs(
                    st.select(station=method)[0].data))
    # Now determine largest and output corresponding catalog
    big_eids = []
    amps = [amp for eid, amp in amp_dict.items()]
    thresh = np.mean(amps) + (sig * np.std(amps))
    print('Amplitude threshold: {}'.format(thresh))
    for eid, amp in amp_dict.items():
        if amp >= thresh:
            big_eids.append(eid)
    big_ev_cat = Catalog(events=[ev for ev in catalog
                                 if ev.resource_id.id.split('/')[-1]
                                 in big_eids])
    return big_ev_cat


def cluster_from_dist_mat(dist_mat, temp_list, corr_thresh,
                          show=False, debug=1, method='single'):
    """
    In the case that the distance matrix has been saved, forego calculating it

    Functionality extracted from eqcorrscan.utils.clustering.cluster
    Consider adding this functionality and commiting to new branch
    :param dist_mat: Distance matrix of pair-wise template wav correlations
    :param temp_list: List of (Stream, Event) with same length as shape of
        distance matrix
    :param corr_thresh: Correlation thresholds corresponding to the method
        used in the linkage algorithm
    :param method: Method fed to scipy.heirarchy.linkage
    :return: Groups of templates
    """
    dist_vec = squareform(dist_mat)
    if debug >= 1:
        print('Computing linkage')
    Z = linkage(dist_vec, method=method)
    if show:
        if debug >= 1:
            print('Plotting the dendrogram')
        dendrogram(Z, color_threshold=1 - corr_thresh,
                   distance_sort='ascending')
        plt.show()
    # Get the indices of the groups
    if debug >= 1:
        print('Clustering')
    indices = fcluster(Z, t=1 - corr_thresh, criterion='distance')
    # Indices start at 1...
    group_ids = list(set(indices))  # Unique list of group ids
    if debug >= 1:
        msg = ' '.join(['Found', str(len(group_ids)), 'groups'])
        print(msg)
    # Convert to tuple of (group id, stream id)
    indices = [(indices[i], i) for i in range(len(indices))]
    # Sort by group id
    indices.sort(key=lambda tup: tup[0])
    groups = []
    if debug >= 1:
        print('Extracting and grouping')
    for group_id in group_ids:
        group = []
        for ind in indices:
            if ind[0] == group_id:
                group.append(temp_list[ind[1]])
            elif ind[0] > group_id:
                # Because we have sorted by group id, when the index is greater
                # than the group_id we can break the inner loop.
                # Patch applied by CJC 05/11/2015
                groups.append(group)
                break
    # Catch the final group
    groups.append(group)
    return groups


def cluster_cat(catalog, corr_thresh, corr_params=None, raw_wav_dir=None,
                dist_mat=False, show=False, method='average'):
    """
    Cross correlate all events in a catalog and return separate tribes for
    each cluster
    :param tribe: Tribe to cluster
    :param corr_thresh: Correlation threshold for clustering
    :param corr_params: Dictionary of filter parameters. Must include keys:
        lowcut, highcut, samp_rate, filt_order, pre_pick, length, shift_len,
        cores
    :param raw_wav_dir: Directory of waveforms to take from
    :param dist_mat: If there's a precomputed distance matrix, use this
        instead of doing all the correlations
    :param out_cat: Output catalog corresponding to the events
    :param show: Show the dendrogram? Careful as this can exceed max recursion
    :param wavs: Should we even bother with processing waveforms? Otherwise
        will just populate the tribe with an empty Stream
    :return:

    .. Note: Functionality here is pilaged from align design as we don't
        want the multiplexed portion of that function.
    """
    # Effing catalogs not being sorted by default
    catalog.events.sort(key=lambda x: x.origins[0].time)
    if corr_params and raw_wav_dir:
        shift_len = corr_params['shift_len']
        lowcut = corr_params['lowcut']
        highcut = corr_params['highcut']
        samp_rate = corr_params['samp_rate']
        filt_order = corr_params['filt_order']
        pre_pick = corr_params['pre_pick']
        length = corr_params['length']
        cores = corr_params['cores']
        wav_dict = read_raw_wavs(raw_wav_dir)
        eids = [ev.resource_id.id.split('/')[-1] for ev in catalog]
        try:
            wavs = [wav_dict[eid] for eid in eids]
        except KeyError as e:
            print(e)
            print('All catalog events not present in waveform directory')
            return
        print('Processing temps')
        temp_list = [(shortproc(tmp,lowcut=lowcut, highcut=highcut,
                                samp_rate=samp_rate, filt_order=filt_order,
                                parallel=True, num_cores=cores),
                      ev.resource_id.id.split('/')[-1])
                     for tmp, ev in zip(wavs, catalog)]
        print('Clipping traces')
        rm_temps = []
        rm_ev = []
        for i, temp in enumerate(temp_list):
            rm_ts = [] # Make a list of traces with no pick to remove
            for tr in temp[0]:
                pk = [pk for pk in catalog[i].picks
                      if pk.waveform_id.station_code == tr.stats.station
                      and pk.waveform_id.channel_code == tr.stats.channel]
                if len(pk) == 0:
                    # This also, inadvertently removes timing/trigger traces
                    rm_ts.append(tr)
                else:
                    tr.trim(starttime=pk[0].time - shift_len - pre_pick,
                            endtime=pk[0].time - pre_pick + length + shift_len)
                    if len(tr) == 0: # Errant pick
                        rm_ts.append(tr)
            # Remove pickless traces
            for rm in rm_ts:
                temp[0].traces.remove(rm)
            # If trace lengths are internally inconsistent, remove template
            if len(list(set([len(tr) for tr in temp[0]]))) > 1:
                rm_temps.append(temp)
            # If template is now length 0, remove it and associated event
            if len(temp[0]) == 0:
                rm_temps.append(temp)
                rm_ev.append(catalog[i])
        for t in rm_temps:
            temp_list.remove(t)
        # Remove the corresponding events as well so catalog and distmat
        # are the same shape
        for rme in rm_ev:
            catalog.events.remove(rme)
    print('Clustering')
    if isinstance(dist_mat, np.ndarray):
        print('Assuming the tribe provided is the same shape as dist_mat')
        # Dummy streams
        temp_list = [(Stream(), ev) for ev in catalog]
        groups = cluster_from_dist_mat(dist_mat=dist_mat, temp_list=temp_list,
                                       show=show, corr_thresh=corr_thresh,
                                       method=method)
    else:
        groups = clustering.cluster(temp_list, show=show,
                                    corr_thresh=corr_thresh,
                                    shift_len=shift_len * 2,
                                    save_corrmat=True, cores=cores)
    group_cats = []
    for grp in groups:
        group_cat = Catalog()
        for ev in grp:
            group_cat.events.extend([
                e for e in catalog if e.resource_id.id.split('/')[-1] == ev[1]
            ])
        group_cats.append(group_cat)
    return group_cats


def vibbox_to_LP(files, outdir, param_file):
    """
    Convert a list of vibbox files to downsampled, lowpassed mseed
    :param files: List of files to convert
    :return:
    """
    # Load in the parameters
    with open(param_file, 'r') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    for afile in files:
        name = os.path.join(
            outdir, afile.split('/')[-2],
            afile.split('/')[-1].replace('.dat', '_accels_10Hz.mseed'))
        if os.path.exists(name):
            print('File already written')
            continue
        if not os.path.isdir(os.path.dirname(name)):
            os.mkdir(os.path.dirname(name))
        print('Writing {} to {}'.format(afile, name))
        # Read raw
        st = vibbox_read(afile, param)
        # Select only accelerometers
        try:
            st = Stream(traces=[tr for tr in st
                                if tr.stats.station in three_comps])
        except TypeError as e:
            print(e)
            continue
        # Downsample, demean, merge, then filter
        st.resample(10.)
        # st.detrend('demean')
        # st.merge(fill_value=0.)
        # st.filter(type='lowpass', freq=1., corners=2)
        # st.resample(3.)
        # # Assume flat resp below 1 Hz, sens 1V/g
        # for tr in st:
        #     # Convert to m/s**2
        #     tr.data /= 9.8
        # # Double integral to DISP in m..?
        # st.integrate()
        # st.integrate()
        st.write(name, format='MSEED')
    return


def make_accelerometer_LP_dict(streams):
    """
    Convenience function to return the accel_dict arg for plotly_timeseries

    :param streams: List of Stream objects in chronological order
    """
    accel_dict = {}
    dt = streams[0][0].stats.delta
    for st in streams:
        for tr in st:
            tmp_data = tr.data
            tmp_data[np.array([0, 1])] = np.nan
            times = [st[0].stats.starttime + timedelta(seconds=i * dt)
                     for i in range(tr.data.shape[0])]
            stachan = '{}.{}'.format(tr.stats.station, tr.stats.channel)
            if stachan not in accel_dict:
                accel_dict[stachan] = {
                    'times': times,
                    'data': tmp_data}
            else:
                accel_dict[stachan]['times'].extend(
                    [st[0].stats.starttime + timedelta(seconds=i * dt)
                     for i in range(st[0].data.shape[0])])
                accel_dict[stachan]['data'] = np.append(
                    accel_dict[stachan]['data'], tmp_data)
    return accel_dict


def geores_read(fname):
    """Martins GEORES read func"""
    stations = ('OT01', 'OT02', 'OT03', 'OT04', 'OT05', 'OT06', 'OT07', 'OT08', 'OT09', 'OT10', 'OT11', 'OT12',
                'PDB01', 'PDB02', 'PDB03', 'PDB04', 'PDB05', 'PDB06', 'PDB07', 'PDB08', 'PDB09', 'PDB10', 'PDB11', 'PDB12',
                'PSB8','PSB8','PSB8','PSB7', 'PSB7', 'PSB7', 'PDB6', 'PDB6', 'PDB6', 'PDT5', 'PDT5', 'PDT5',
                'PDB4', 'PDB4', 'PDB4', 'PDB3', 'PDB3', 'PDB3', 'PDT2', 'PDT2', 'PDT2',  'PDT1', 'PDT1', 'PDT1',
                'OT16', 'OT16', 'OT16', 'OB15', 'OB15', 'OB15', 'OB14', 'OB14', 'OB14', 'OB13', 'OB13', 'OB13',
                'PST12', 'PST12', 'PST12', 'PST11', 'PST11', 'PST11', 'PST10', 'PST10', 'PST10', 'PSB9', 'PSB9', 'PSB9',
                'PPS', 'CMon', 'CEnc','CTrig', 'NC1', 'NC2', 'W4', 'W4', 'W4', 'W3', 'W3', 'W3', 'W2', 'W2', 'W2', 'W1', 'W1', 'W1',
                'OT18', 'OT18', 'OT18', 'OT17', 'OT17', 'OT17')
    location = ('', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''
                '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '')
    channel = ('XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1',
               'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1', 'XN1',
               'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ',
               'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ',
               'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ',
               'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ',
               '', '', '', '', '', '', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ',
               'XNY', 'XNX', 'XNZ', 'XNY', 'XNX', 'XNZ')
    channels=96
    st = read(fname)
    for ii in range(channels):
        st[ii].stats.network = 'SV'
        st[ii].stats.station = stations[ii]
        st[ii].stats.location = location[ii]
        st[ii].stats.channel = channel[ii]
    return st


########################## PLOTTING ######################################


def compare_NSMTC_inst(wav_files, cat, inv, signal_len, outdir='.',
                       log=False, plot_spectra=False):
    """
    Plot signal comparison between SA-ULNs and Geophone (B3 and G2)

    :param wav_files: List of extracted event files
    :param cat: Catalog with the picks
    :param inv: Inventory with response info
    :param signal_len: Length of signal for SNR calcs
    :param outdir: Output directory for plots
    :param log: Log SNR or not?
    :param plot_spectra: Plot the spectra in a bottom panel?

    :return:
    """
    # inv = modify_SAULN_inventory(inv)
    snrs = {}
    for wav_file in wav_files:
        eid = wav_file.split('_')[-1].rstrip('.ms')
        print(eid)
        if os.path.exists('{}/{}.png'.format(outdir, eid)):
            print('Output already made')
            continue
        try:
            ev = [ev for ev in cat
                  if ev.resource_id.id[-25:-15] == eid][0]
        except IndexError:
            try:
                ev = [ev for ev in cat
                      if ev.resource_id.id.split('/')[-1] == eid][0]
            except IndexError:
                print('{} not in wav archive'.format(eid))
                continue
        o = ev.origins[-1]
        sta = inv.select(station='NSMTC')[0][0]
        dist = dist_calc((o.latitude, o.longitude, o.depth / 1000.),
                         (sta.latitude, sta.longitude, 0.))
        if dist > 100:
            print('Too far away: {}'.format(dist))
            continue
        st = read(wav_file)
        # Remove response
        st = st.select(station='[NP]*C', channel='*Z')
        st.detrend()
        st.detrend('demean')
        st.taper(0.05)
        st.remove_response(inventory=inv, output='VEL', water_level=80)
        st.detrend()
        st.detrend('demean')
        st.taper(0.05)
        st.filter(type='bandpass', freqmin=2, freqmax=20, corners=3)
        try:
            pk_P = [pk for pk in ev.picks if pk.waveform_id.location_code
                    in ['B3', 'G2', 'G1'] and pk.phase_hint == 'P'][0]
        except IndexError:
            print('No P pick at either, skipping')
            continue
        st_b3 = st.select(location='B3', channel='*Z')
        st_g2 = st.select(location='G2', channel='*Z')
        st_g1 = st.select(location='G1', channel='*Z')
        st_pgc = st.select(station='PGC', channel='*Z')
        b3_noise = st_b3.slice(starttime=pk_P.time - 7,
                               endtime=pk_P.time - 0.25).copy()
        b3_signal = st_b3.slice(starttime=pk_P.time,
                                endtime=pk_P.time + signal_len).copy()
        g2_noise = st_g2.slice(starttime=pk_P.time - 7,
                               endtime=pk_P.time - 0.25).copy()
        g2_signal = st_g2.slice(starttime=pk_P.time,
                                endtime=pk_P.time + signal_len).copy()
        g1_noise = st_g1.slice(starttime=pk_P.time - 7,
                               endtime=pk_P.time - 0.25).copy()
        g1_signal = st_g1.slice(starttime=pk_P.time,
                                endtime=pk_P.time + signal_len).copy()
        pgc_noise = st_pgc.slice(starttime=pk_P.time - 7,
                                 endtime=pk_P.time - 0.25).copy()
        pgc_signal = st_pgc.slice(starttime=pk_P.time,
                                  endtime=pk_P.time + signal_len).copy()
        # Maximum amplitudes (displacement)
        g2_amp = np.max(np.abs(g2_signal[0].copy().integrate().data))
        g1_amp= np.max(np.abs(g1_signal[0].copy().integrate().data))
        b3_amp= np.max(np.abs(b3_signal[0].copy().integrate().data))
        pgc_amp= np.max(np.abs(pgc_signal[0].copy().integrate().data))
        g2_snr = SNR(g2_signal[0].data, g2_noise[0].data, log=log)
        g1_snr = SNR(g1_signal[0].data, g1_noise[0].data, log=log)
        b3_snr = SNR(b3_signal[0].data, b3_noise[0].data, log=log)
        pgc_snr = SNR(pgc_signal[0].data, pgc_noise[0].data, log=log)
        snrs[eid] = {'G1': g1_snr, 'G2': g2_snr, 'B3': b3_snr,
                     'PGC': pgc_snr, 'event': ev, 'G1_amp': g1_amp,
                     'G2_amp': g2_amp, 'B3_amp': b3_amp,
                     'PGC_amp': pgc_amp}
        g2_plot = st_g2.slice(starttime=pk_P.time - 5,
                              endtime=pk_P.time + 15).copy()
        b3_plot = st_b3.slice(starttime=pk_P.time - 5,
                              endtime=pk_P.time + 15).copy()
        g1_plot = st_g1.slice(starttime=pk_P.time - 5,
                              endtime=pk_P.time + 15).copy()
        pgc_plot = st_pgc.slice(starttime=pk_P.time - 5,
                                endtime=pk_P.time + 15).copy()
        if plot_spectra:
            fig, axes = plt.subplots(nrows=3, figsize=(8, 13))
        else:
            fig, axes = plt.subplots(nrows=2, sharex='col', figsize=(8, 10))
        start = g2_plot[0].stats.starttime.datetime
        dt = g2_plot[0].stats.delta
        time_vect = [start + timedelta(seconds=dt * i)
                     for i in range(g2_plot[0].data.shape[0])]
        axes[0].plot(time_vect, g1_plot[0].data,
                     color=cascadia_colors['NSMTC.G1'], alpha=0.6,
                     linewidth=1.5, label='NV.NSMTC.G1.CHZ')
        axes[0].plot(time_vect, pgc_plot[0].data, color=cascadia_colors['PGC.'],
                     alpha=0.6, linewidth=1.5, label='CN.PGC..HHZ')
        axes[1].plot(time_vect, g2_plot[0].data,
                     color=cascadia_colors['NSMTC.G2'],
                     alpha=0.6, linewidth=1.5, label='NV.NSMTC.G2.CNZ')
        axes[1].plot(time_vect, b3_plot[0].data,
                     color=cascadia_colors['NSMTC.B3'],
                     alpha=0.6, linewidth=1.5, label='NV.NSMTC.B3.CNZ')
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        axes[0].annotate(xy=(0.7, 0.9), text='SNR: {:0.2f} dB'.format(pgc_snr),
                         color=cascadia_colors['PGC.'],
                         xycoords='axes fraction',
                         fontsize=16, bbox=props)
        axes[0].annotate(xy=(0.7, 0.8), text='SNR: {:0.2f} dB'.format(g1_snr),
                         color=cascadia_colors['NSMTC.G1'],
                         xycoords='axes fraction', fontsize=16, bbox=props)
        axes[1].annotate(xy=(0.7, 0.9), text='SNR: {:0.2f} dB'.format(g2_snr),
                         color=cascadia_colors['NSMTC.G2'],
                         xycoords='axes fraction', fontsize=16, bbox=props)
        axes[1].annotate(xy=(0.7, 0.8), text='SNR: {:0.2f} dB'.format(b3_snr),
                         color=cascadia_colors['NSMTC.B3'],
                         xycoords='axes fraction', fontsize=16, bbox=props)
        if plot_spectra:
            axes[2].magnitude_spectrum(g1_plot[0].data,
                                       color=cascadia_colors['NSMTC.G1'],
                                       Fs=g1_plot[0].stats.sampling_rate)
            axes[2].magnitude_spectrum(pgc_plot[0].data, #scale='dB',
                                       color=cascadia_colors['PGC.'],
                                       Fs=pgc_plot[0].stats.sampling_rate)
            axes[2].magnitude_spectrum(g2_plot[0].data, #scale='dB',
                                       color=cascadia_colors['NSMTC.G2'],
                                       Fs=g2_plot[0].stats.sampling_rate)
            axes[2].magnitude_spectrum(b3_plot[0].data, #scale='dB',
                                       color=cascadia_colors['NSMTC.B3'],
                                       Fs=b3_plot[0].stats.sampling_rate)
            # axes[2].set_xscale('log')
            axes[2].set_xlim([1, 25])
        axes[1].xaxis_date()
        axes[0].legend(loc=2, fontsize=16)
        axes[1].legend(loc=2, fontsize=16)
        axes[0].set_ylabel('Velocity [m/s]', fontsize=16)
        axes[1].set_ylabel('Velocity [m/s]', fontsize=16)
        mag = ev.magnitudes[-1].mag
        fig.suptitle('M$_L$ {:0.2f} at {:0.2f} km'.format(mag, dist),
                     fontsize=18)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(fname='{}/{}.pdf'.format(outdir, eid))
        plt.close()
    return snrs


def plot_snr_distribution(snrs):
    g1_snrs = [d['G1'] for eid, d in snrs.items()]
    g2_snrs = [d['G2'] for eid, d in snrs.items()]
    b3_snrs = [d['B3'] for eid, d in snrs.items()]
    pgc_snrs = [d['PGC'] for eid, d in snrs.items()]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.distplot(g1_snrs, hist=False, color=cascadia_colors['NSMTC.G1'], ax=ax,
                 label='Surface geophone')
    sns.distplot(g2_snrs, hist=False, color=cascadia_colors['NSMTC.G2'], ax=ax,
                 label='Geophone: 308 m')
    sns.distplot(b3_snrs, hist=False, color=cascadia_colors['NSMTC.B3'], ax=ax,
                 label='SA-ULN: 305 m')
    sns.distplot(pgc_snrs, hist=False, color=cascadia_colors['PGC.'], ax=ax,
                 label='Broadband')
    ax.set_ylabel('Kernel density', fontsize=16)
    ax.set_xlabel('SNR [dB]', fontsize=16)
    ax.legend(fontsize=14)
    ax.axvline(x=np.median(g1_snrs), color=cascadia_colors['NSMTC.G1'],
               linestyle='--', linewidth=1.5)
    ax.axvline(x=np.median(g2_snrs), color=cascadia_colors['NSMTC.G2'],
               linestyle='--', linewidth=1.5)
    ax.axvline(x=np.median(b3_snrs), color=cascadia_colors['NSMTC.B3'],
               linestyle='--', linewidth=1.5)
    ax.axvline(x=np.nanmedian(pgc_snrs), color=cascadia_colors['PGC.'],
               linestyle='--', linewidth=1.5)
    ax.annotate(text='Median SNR'.format(np.median(g1_snrs)),
                xy=(.8, 0.58), xycoords='axes fraction', weight='bold',
                ha='center', fontsize=14, color='k')
    ax.annotate(text='Geophone: {:0.2f}'.format(np.median(g1_snrs)),
                xy=(.8, 0.5), xycoords='axes fraction',
                ha='center', fontsize=14, color=cascadia_colors['NSMTC.G1'])
    ax.annotate(text='Geophone 308 m: {:0.2f}'.format(np.median(g2_snrs)),
                xy=(.8, 0.34), xycoords='axes fraction',
                ha='center', fontsize=14, color=cascadia_colors['NSMTC.G2'])
    ax.annotate(text='SA-ULN 305 m: {:0.2f}'.format(np.median(b3_snrs)),
                xy=(.8, 0.26), xycoords='axes fraction',
                ha='center', fontsize=14, color=cascadia_colors['NSMTC.B3'])
    ax.annotate(text='Broadband: {:0.2f}'.format(np.nanmedian(pgc_snrs)),
                xy=(.8, 0.42), xycoords='axes fraction',
                ha='center', fontsize=14, color=cascadia_colors['PGC.'])
    fig.suptitle('Regional seismicity SNR', fontsize=18)
    plt.show()
    return


def plot_signal_w_dist(snrs, measure='snr', title=None, mag_correct=True):
    """
    Plot the snr with distance from output of above func

    :param snrs:
    :return:
    """
    loc_b3 = (48.65, -123.45, 0.3)
    dists = [dist_calc(loc_b3, (d['event'].origins[-1].latitude,
                                d['event'].origins[-1].longitude,
                                d['event'].origins[-1].depth / 1000.))
             for eid, d in snrs.items()]
    mags_correction = np.array([10**d['event'].magnitudes[-1].mag
                                for eid, d in snrs.items()])
    if measure == 'snr':
        ylab = 'SNR [dB]'
        b3 = np.array([d['B3'] for eid, d in snrs.items()])
        g2 = np.array([d['G2'] for eid, d in snrs.items()])
    elif measure == 'amp':
        ylab = 'Max amplitude [m/s]'
        b3 = np.array([d['B3_amp'] for eid, d in snrs.items()])
        g2 = np.array([d['G2_amp'] for eid, d in snrs.items()])
    else:
        print('Only supported measures: amp, snr')
        return
    if mag_correct:
        ylab = ylab.replace('[', '/ $10^{Ml}$ [')
        b3 /= mags_correction
        g2 /= mags_correction
    fig, axes = plt.subplots(nrows=2, sharex='col', sharey='col',
                             figsize=(6, 8))
    axes[0].scatter(dists, b3, alpha=0.5, label='SA-ULN: B3', color='r',
                    s=5)
    axes[1].scatter(dists, g2, alpha=0.5, label='Geophone: G2',
                    color='steelblue', s=5)
    axes[0].legend()
    axes[1].legend()
    axes[1].set_xlim([0, 250])
    axes[1].set_xlabel('Event-Station distance [km]', fontsize=14)
    axes[0].set_ylabel(ylab, fontsize=14)
    axes[1].set_ylabel(ylab, fontsize=14)
    axes[0].set_title(title, fontsize=18)
    plt.show()
    return


def plot_pick_corrections(catalog, stream_dir, t_before, t_after,
                          max_lag, ccc_thresh, plotdir):
    """
    Hard coded wrapper on xcorr_pick_correction

    Used for cascadia hypodd cc testing

    :param catalog: Catalog of events to generate plots for
    :param stream_dir: Path to directory of *raw.mseed files
    :param t_before:
    :param t_after:
    :param max_lag:
    :param plotdir: Path to root directory for plots

    :return:
    """
    for ev1, ev2 in itertools.combinations(catalog, r=2):
        eid1 = ev1.resource_id.id.split('/')[-1]
        eid2 = ev2.resource_id.id.split('/')[-1]
        # For FDSN pulled events from USGS
        if len(eid1.split('=')) > 1:
            eid1 = ev1.resource_id.id.split('=')[-2].split('&')[0]
        if len(eid2.split('=')) > 1:
            eid2 = ev2.resource_id.id.split('=')[-2].split('&')[0]
        try:
            st1 = read('{}/Event_{}.ms'.format(stream_dir, eid1))
        except FileNotFoundError as e:
            print(e)
            continue
        for pk1 in ev1.picks:
            seed_id = pk1.waveform_id.get_seed_string()
            seeddir = '{}/{}'.format(plotdir, seed_id)
            if not os.path.isdir(seeddir):
                os.mkdir(seeddir)
            try:
                st2 = read('{}/Event_{}.ms'.format(stream_dir, eid2))
            except FileNotFoundError as e:
                print(e)
                continue
            tr1 = st1.select(id=seed_id)
            tr2 = st2.select(id=seed_id)
            pk2 = [pk for pk in ev2.picks
                   if pk.waveform_id.get_seed_string() == seed_id]
            if len(pk2) > 0 and len(tr2) > 0:
                fname = '{}/{}/{}_{}_{}.pdf'.format(
                    plotdir, seed_id, eid1, eid2, seed_id)
                try:
                    ct, mccc = xcorr_pick_correction(
                        pk1.time, tr1[0], pk2[0].time, tr2[0],
                        t_before=t_before, t_after=t_after,
                        cc_maxlag=max_lag, plot=True,
                        filter='bandpass',
                        filter_options={'corners': 3,
                                        'freqmax': 15.,
                                        'freqmin': 1.},
                        filename='{}/{}/{}_{}_{}.pdf'.format(
                            plotdir, seed_id, eid1, eid2, seed_id))
                    if mccc < ccc_thresh:
                        # Remove this plot
                        os.remove(fname)
                except Exception as e:
                    print(e)
                    continue
    return


def plot_arrivals(st, ev, pre_pick, post_pick):
    """
    Simple plot of arrivals for showing polarities

    :param st: Stream containing arrivals
    :param ev: Event with pick information
    :param pre_pick:
    :param post_pick:
    :return:
    """
    plot_st = Stream()
    for pk in ev.picks:
        sta = pk.waveform_id.station_code
        chan = pk.waveform_id.channel_code
        if pk.polarity and len(st.select(station=sta, channel=chan)) != 0:
            tr = st.select(station=sta, channel=chan)[0]
            tr.trim(starttime=pk.time - pre_pick, endtime=pk.time + post_pick)
            plot_st += tr
    plot_st.plot(equal_scale=False)
    return


def plot_station_rot_stats(sta_dict, title='Station orientaton stats'):
    """
    Plot the statistics of the station dictionary output by
    rotate_catalog_streams()

    :param sta_dict: Nested dictionary with keys 'az-dip', 'borehole angle'
    :return:
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))#,
                             #subplot_kw=dict(polar=True))
    for sta, d in sta_dict.items():
        az, dip = zip(*d['orientation'])
        sns.distplot(d['bh angle'], label=sta, ax=axes[2],
                     hist=False, rug=True)
        sns.distplot(az, label=sta, ax=axes[0], hist=False, rug=True)
        sns.distplot(dip, label=sta, ax=axes[1], hist=False, rug=True)
        axes[0].legend()
        axes[0].set_title('Channel azimuth')
        # axes[0].set_theta_zero_location('N')
        # axes[0].set_theta_direction(-1)
        axes[1].legend()
        axes[1].set_title('Channel dip (from horizontal)')
        # axes[1].set_theta_zero_location('E')
        # axes[1].set_theta_direction(-1)
        axes[2].legend()
        axes[2].set_title('Channel angle with borehole axis')
        # axes[1].set_theta_zero_location('N')
        # axes[1].set_theta_direction(-1)
        plt.suptitle(title, fontsize=20)
    return axes


def family_stack_plot(family, wav_files, seed_id, selfs,
                      title='Detections', shift=True, shift_len=0.3,
                      pre_pick_plot=1., post_pick_plot=5., pre_pick_corr=0.05,
                      post_pick_corr=0.5, cc_thresh=0.7, spacing_param=2,
                      normalize=True, plot_mags=False, figsize=(8, 15),
                      savefig=None):
    """
    Plot list of traces for a stachan one just above the other (modified from
    subspace_util.stack_plot()

    Modified from workflow.util.plot_detections.family_stack_plot 9-23-2020

    :param events: List of events from which we'll extract picks for aligning
    :param wav_dirs: List of directories containing SAC files for above events
    :param seed_id: Net.Sta.Loc.Chan seed string for selecting picks/wavs
    :param selfs: List of self detection ids for coloring the template
    :param title: Plot title
    :param shift: Whether to allow alignment of the wavs
    :param shift_len: Length in seconds to allow wav to shift
    :param pre_pick_plot: Seconds before p-pick to plot
    :param post_pick_plot: Seconds after p-pick to plot
    :param pre_pick_corr: Seconds before p-pick to use in correlation alignment
    :param post_pick_corr: Seconds after p-pick to use in correlation alignment
    :param spacing_param: Parameter determining vertical spacing of traces.
        value of 1 indicates unit of 1 between traces. Less than one, tighter,
        greater than 1 looser.
    :param normalize: Flag to normalize traces
    :param figsize: Tuple of matplotlib figure size
    :param savefig: Name of the file to write
    :return:

    .. note: When using the waveforms clipped for stefan, they were clipped
    based on origin time (I think??), so should be clipped on the pick time
    to plot aligned on the P-pick.
    """
    streams = [] # List of tup: (Stream, name string)
    rm_evs = []
    events = [d.event for d in family.detections]
    temp_freqmax = family.template.highcut
    temp_freqmin = family.template.lowcut
    temp_samp_rate = family.template.samp_rate
    temp_order = family.template.filt_order
    prepick = family.template.prepick
    for i, ev in enumerate(events):
        eid = ev.resource_id.id
        eid = eid.split('_')
        if len(eid) == 3:
            eid = '{}_{}T{}.{}'.format(eid[0], eid[1], eid[2][:-6], eid[2][-6:])
        elif len(eid) == 4:
            eid = '{}_{}_{}T{}.{}'.format(eid[0], eid[1], eid[2], eid[3][:-6],
                                          eid[3][-6:])
        det_file = [f for f in wav_files
                    if f.split('/')[-1].rstrip('.ms') == eid]
        try:
            streams.append(read(det_file[0]))
        except IndexError: # If this directory doesn't exist, remove event
            print('{} doesnt exist'.format(eid))
            rm_evs.append(events[i])
    for rm in rm_evs:
        events.remove(rm)
    print('Have {} streams and {} events'.format(len(streams), len(events)))
    # Select all traces
    traces = []
    pk_offsets = []
    tr_evs = []
    colors = []  # Plotting colors
    for i, (st, ev) in enumerate(zip(streams, events)):
        if len(st.select(id=seed_id)) == 1:
            st1 = shortproc(st=st, lowcut=temp_freqmin, highcut=temp_freqmax,
                            filt_order=temp_order, samp_rate=temp_samp_rate)
            tr = st1.select(id=seed_id)[0]
            try:
                pk = [pk for pk in ev.picks
                      if pk.waveform_id.get_seed_string() == tr.id][0]
            except:
                print('No pick for this event')
                continue
            traces.append(tr)
            tr_evs.append(ev)
            if ev.resource_id.id.split('/')[-1] in selfs:
                colors.append('red')
                master_trace = tr
                pk_offsets.append(0.0)
            else:
                colors.append('k')
                pk_offsets.append(0.1) #  Deal with template pick offset
        else:
            print('No trace in stream for {}'.format(seed_id))
    # Normalize traces, demean and make dates vect
    date_labels = []
    print('{} traces found'.format(len(traces)))
    # Vertical space array
    vert_steps = np.linspace(0, len(traces) * spacing_param, len(traces))
    fig, ax = plt.subplots(figsize=figsize)
    shift_samp = int(shift_len * traces[0].stats.sampling_rate)
    pks = []
    for ev, pk_offset in zip(tr_evs, pk_offsets):
        pks.append([pk.time + pk_offset for pk in ev.picks
                    if pk.waveform_id.get_seed_string() == tr.id][0])
    mags = []
    for ev in tr_evs:
        try:
            mags.append(ev.preferred_magnitude().mag)
        except AttributeError:
            mags.append('')
    # Copy these out of the way for safe keeping
    if shift:
        cut_traces = [tr.copy().trim(starttime=p_time - pre_pick_corr,
                                     endtime=p_time + post_pick_corr)
                      for tr, p_time in zip(traces, pks)]
        shifts, ccs = align_traces(cut_traces, shift_len=shift_samp,
                                   master=master_trace)
        shifts = np.array(shifts)
        shifts /= tr.stats.sampling_rate  # shifts is in samples, we need sec
    # Now trim traces down to plotting length
    for tr, pk in zip(traces, pks):
        tr.trim(starttime=pk - pre_pick_plot,
                endtime=pk + post_pick_plot)
        date_labels.append(str(tr.stats.starttime.date))
        tr.data -= np.mean(tr.data)
        if normalize:
            tr.data /= max(tr.data)
    if shift:
        # Establish which sample the pick will be plotted at (prior to slicing)
        pk_samples = [(pk - tr.stats.starttime) * tr.stats.sampling_rate
                      for tr, pk in zip(traces, pks)]
        dt_vects = []
        pk_xs = []
        arb_dt = UTCDateTime(1970, 1, 1)
        td = timedelta(microseconds=int(1 / tr.stats.sampling_rate * 1000000))
        for shif, cc, tr, p_samp in zip(shifts, ccs, traces, pk_samples):
            # Make new arbitrary time vectors as they otherwise occur on
            # different dates
            if cc >= cc_thresh:
                dt_vects.append([(arb_dt + shif).datetime + (i * td)
                                 for i in range(len(tr.data))])
                pk_xs.append((arb_dt + shif).datetime + (p_samp * td))
            else:
                dt_vects.append([(arb_dt).datetime + (i * td)
                                 for i in range(len(tr.data))])
                pk_xs.append((arb_dt).datetime + (p_samp * td))
    else:
        pk_samples = [(pk - tr.stats.starttime) * tr.stats.sampling_rate
                      for tr, pk in zip(traces, pks)]
        dt_vects = []
        pk_xs = []
        arb_dt = UTCDateTime(1970, 1, 1)
        td = timedelta(microseconds=int(1 / tr.stats.sampling_rate * 1000000))
        for tr, p_samp in zip(traces, pk_samples):
            # Make new arbitrary time vectors as they otherwise occur on
            # different dates
            dt_vects.append([(arb_dt).datetime + (i * td)
                             for i in range(len(tr.data))])
            pk_xs.append((arb_dt).datetime + (p_samp * td))
    # Plotting chronologically from top
    for tr, vert_step, dt_v, col, pk_x, mag in zip(traces,
                                                   list(reversed(
                                                       vert_steps)),
                                                   dt_vects, colors, pk_xs,
                                                   mags):
        if shift:
            ax.vlines(x=pk_x, ymin=vert_step - spacing_param / 2.,
                      ymax=vert_step + spacing_param / 2., linestyle='--',
                      color='red')
        # Magnitude text
        try:
            mag_text = 'M$_L$={:0.2f}'.format(mag)
        except ValueError:
            mag_text = ''
        if shift:
            mag_x = (arb_dt + post_pick_plot + max(shifts)).datetime
        else:
            mag_x = (arb_dt + post_pick_plot + 1).datetime
        if not plot_mags or mag_text != '':
            ax.plot(dt_v, tr.data + vert_step, color=col)
        elif mag_text == '':
            ax.plot(dt_v, tr.data + vert_step, color='k', alpha=0.2)
        if plot_mags:
            ax.text(mag_x, vert_step, mag_text, fontsize=14,
                    verticalalignment='center', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="k", lw=1))
    # Plot the stack of all the waveforms (maybe with mean pick and then AIC
    # pick following Rowe et al. 2004 JVGR)
    data_stack = np.sum(np.array([tr.data for tr in traces]), axis=0)
    # Demean and normalize
    data_stack -= np.mean(data_stack)
    data_stack /= np.max(data_stack)
    # Plot using last datetime vector from loop above for convenience
    ax.plot(dt_vects[-1], (data_stack * 2) - vert_steps[2],
            color='b')
    # Have to suss out average pick time tho
    av_p_time = (arb_dt - prepick).datetime + (np.mean(pk_samples) * td)
    ax.vlines(x=av_p_time, ymin=-vert_steps[2] - (spacing_param * 2),
              ymax=-vert_steps[2] + (spacing_param * 2),
              color='green')
    ax.set_xlabel('Seconds', fontsize=19)
    # Second locator
    formatter = mdates.DateFormatter('%S')
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylabel('Date', fontsize=19)
    # Change y labels to dates
    ax.yaxis.set_ticks(vert_steps)
    date_labels[1::3] = ['' for d in date_labels[1::3]]
    date_labels[2::3] = ['' for d in date_labels[2::3]]
    ax.set_yticklabels(date_labels[::-1], fontsize=16, rotation=30)
    ax.set_title(title, fontsize=19)
    if savefig:
        fig.tight_layout()
        plt.savefig(savefig, dpi=300)
        plt.close()
    else:
        fig.tight_layout()
        plt.show()
    return


def compare_detection_wavs(mseeds, events, template, seed_ids,
                           prepick_plot=3., postpick_plot=7.):
    """
    Plot waveforms for the same detections at specified stations
    :param mseeds: List of wav files for detections
    :param events: List of events for each detection, in order of mseeds
    :param seed_ids: List of net.sta.loc for each station you want to plot
    :return:
    """
    cols = cycle(sns.color_palette('muted'))
    # Grab only the seed ids we want
    streams = []
    for ms in mseeds:
        st = read(ms)
        rms = []
        for tr in st:
            staloc = '.'.join(tr.id.split('.')[:-1])
            if (staloc not in seed_ids or tr.stats.channel[-1] in ['2', 'E'] or
                '{}.{}'.format(tr.stats.station, tr.stats.channel[1]) == 'PGC.N'):
                rms.append(tr)
        for rm in rms:
            st.traces.remove(rm)
        streams.append(st)
    for i, (st, ev) in enumerate(zip(streams, events)):
        fig, axes = plt.subplots(figsize=(15, 5))
        labs = []
        t_cols = []
        st1 = shortproc(st=st, lowcut=template.lowcut,
                        highcut=49.,
                        filt_order=template.filt_order,
                        samp_rate=100.)
        for j, sid in enumerate(seed_ids):
            seed_pts = sid.split('.')
            tmp_st = st1.select(station=seed_pts[1], location=seed_pts[2])
            pk = [pk for pk in ev.picks
                  if pk.waveform_id.location_code == seed_pts[2]
                  and pk.waveform_id.station_code == seed_pts[1]
                  and pk.waveform_id.channel_code[-1] == "Z"][0]
            tmp_st.trim(starttime=pk.time - prepick_plot,
                        endtime=pk.time + postpick_plot)
            for tr in tmp_st:
                if tr.stats.channel[-1] == 'Z':
                    alpha = 0.7
                else:
                    alpha = 0.4
                tr_data = tr.data / np.max(tr.data)
                c = cascadia_colors['.'.join(sid.split('.')[1:])]
                times = np.arange(tr_data.shape[0]) * tr.stats.delta
                axes.plot(times, tr_data + (j * 1.7),
                          color=c, label=tr.id, alpha=alpha)
            t_cols.append(c)
            labs.append(sid)
        # P and S times
        axes.axvline(prepick_plot, linestyle='--', color='firebrick',
                     label='P-pick', linewidth=1., alpha=0.4)
        plt.yticks(np.arange(len(labs)) * 1.7, labs)
        [t.set_color(t_cols[i]) for i, t in enumerate(axes.get_yticklabels())]
        axes.set_xlabel('Seconds', fontsize=14)
        axes.annotate(s='$M_L$ {:0.2f}'.format(ev.preferred_magnitude().mag),
                      xy=(0.1, 1.), xycoords='axes fraction',
                      bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                ec="k", lw=1),
                      ha='center', fontsize=18)
        fig.suptitle(ev.resource_id.id)
        plt.show()
    return


def plot_all_spectra(cat, wav_dir, inv, seed_ids, savedir=None):
    casc_dict = casc_wav_dict(cat, wav_dir)
    for eid, l in casc_dict.items():
        try:
            plot_raw_spectra(read(l[1]), l[0], seed_ids=seed_ids, inv=inv)
        except FileNotFoundError as e:
            print(e)
            continue
    return


def casc_wav_dict(cat, wav_dir):
    cat_dict = {}
    for ev in cat:
        eid1 = ev.resource_id.id.split('/')[-1]
        # For FDSN pulled events from USGS
        if len(eid1.split('=')) > 1:
            eid1 = ev.resource_id.id.split('=')[-2].split('&')[0]
        cat_dict[eid1] = [ev, '{}/Event_{}.ms'.format(wav_dir, eid1)]
    return cat_dict


def plot_raw_spectra(st, ev, seed_ids, inv=None, savefig=None):
    """
    Simple function to plot the displacement spectra of a trace

    :param st: obspy.core.trace.Stream
    :param ev: obspy.core.event.Event
    :param seed_ids: List of net.sta.loc to plot
    :param inv: Inventory if we want to remove response

    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 8))
    eid = str(ev.resource_id).split('/')[-1]
    for trace in st:
        tr = trace.copy()
        chan = tr.stats.channel
        staloc = '.'.join(tr.id.split('.')[:-1])
        if (staloc not in seed_ids or tr.stats.channel[-1] in ['2', 'E'] or
                '{}.{}'.format(tr.stats.station, tr.stats.channel[1]) == 'PGC.N'):
            continue
        if not chan.endswith(('1', 'Z')):
            # Only use Z comps for now
            continue
        pick = [pk for pk in ev.picks
                if pk.waveform_id.get_seed_string() == tr.id
                and tr.stats.channel[-1] == 'Z']
        if len(pick) == 0:
            print('No pick for {}'.format(tr.id))
            continue
        else:
            pick = pick[0]
        if inv:
            tr.remove_response(inventory=inv, output='DISP')
        else:
            print('No instrument response to remove. Raw spectrum only.')
        c = cascadia_colors['.'.join(tr.id.split('.')[1:-1])]
        # Hard coded for cascadia sources atm
        tr.trim(starttime=pick.time - 0.1, endtime=pick.time + 7)
        axes[0].plot(tr.data, color=c)
        N = fftpack.next_fast_len(tr.stats.npts)
        T = 1.0 / tr.stats.sampling_rate
        frequencies = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        fft = fftpack.rfft(tr.data, N)
        axes[1].loglog(frequencies, 2.0 / N * np.abs(fft[0: N // 2]),
                       'r', label=tr.id, color=c)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend()
        axes[1].set_title('{}: Displacement Spectra'.format(eid))
    if savefig:
        dir = '{}/{}'.format(savefig, eid)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        fig.savefig('{}/{}_spectra.png'.format(dir, eid))
    else:
        plt.show()
    return axes


def plot_psds(psd_dir, seeds, datetime=None, reference_seed='NV.NSMTC.B2.CNZ',
              eq_psd=None):
    """
    Take pre-computed ppsds and plot the means and diffs for all specified
    channels

    :param psd_dir: Root dir with the .npz files
    :param seeds: list of full seed ids
    :param datetime: Datetime for date we want (will only use year and julday)
    :param eq_psd: Path to file with periods and pds values for an event
        or set of events
    :return:
    """
    cols = cycle(sns.color_palette('muted'))
    next(cols)  # Skip first blue-ish one
    B_cols = cycle(sns.color_palette('Blues', 3))
    npz_files = glob('{}/*'.format(psd_dir))
    if datetime:
        day_str = '{}.{:03d}'.format(datetime.year,
                                     UTCDateTime(datetime).julday)
    else:
        day_str = 'FEB_MAR'
    ppsds = {}
    for seed in seeds:
        try:
            ppsds[seed] = PPSD.load_npz(
                [f for f in npz_files
                 if f[:-4].endswith('.'.join([seed, day_str]))][0])
        except IndexError:
            print('No file for {}.{}'.format(seed, day_str))
            continue
    # Plot em
    fig, axes = plt.subplots(ncols=2, sharex='row', figsize=(15, 5))
    plot_noise_and_sig_bands(axes[0])
    refx, refy = ppsds[reference_seed].get_mean()
    for seed, ppsd in ppsds.items():
        c = cascadia_colors['.'.join(seed.split('.')[1:-1])]
        xs, ys = ppsd.get_mean()
        try:
            diffx, diffy = ys - refy
        except ValueError:  # Case of lower samp rate data
            # Interpolate onto reference freqs
            f = interp1d(xs, ys, bounds_error=False, fill_value=np.nan)
            diffx = refx
            diffy = f(refx) - refy
        # Plot vs frequency
        axes[0].plot(1 / xs, ys, label=seed, color=c)
        axes[1].plot(1 / diffx, diffy, color=c)
        if seed == reference_seed:
            axes[0].fill_between(1 / xs, -180, ys, color='lightgray',
                                 alpha=0.8)
            axes[1].fill_between(1 / diffx, -45, diffy, color='lightgray',
                                 alpha=0.8)
    if eq_psd:
        df = pd.read_csv(eq_psd)
        pds = df['periods']
        psds = df['psd']
        axes[0].plot(1 / pds, psds, color='gray', linestyle=':')
        # Interpolate onto reference freqs
        f = interp1d(pds, psds, bounds_error=False, fill_value=np.nan)
        diffx = refx
        diffy = f(refx) - refy
        axes[1].plot(1 / diffx, diffy, label='MEQ', color='gray',
                     linestyle=':')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Freq [Hz]', fontsize=12)
    axes[1].set_xlabel('Freq [Hz]', fontsize=12)
    axes[0].set_ylabel('Amplitude [dB]', fontsize=12)
    axes[1].set_ylabel('Relative amplitude [dB]', fontsize=12)
    axes[0].set_facecolor('whitesmoke')
    axes[1].set_facecolor('whitesmoke')
    axes[0].margins(0.)
    axes[1].margins(0.)
    axes[0].set_xlim([0.002, 250.])
    axes[0].set_ylim(bottom=-180)
    fig.legend()
    plt.show()
    return


def plot_noise_and_sig_bands(axes=None, plot_brune=False, plot_bands=False,
                             sig0=10, Q=100, radius=1000, Vs=3600, Vr=2100):
    """
    Overview of signal bands, noise levels

    :param axes: mpl axes to plot into (optional)
    :param plot_brune: Bool for plotting eq models
    :param plot_bands: Plot background bands for various signals
    :param sig0: Stress drop [MPa]
    :param Q: Quality factor
    :param radius: Source-receiver distance [meters]
    :param Vp: P velocity [m/s]
    :param Vs: S velocity [m/s]
    :plot_source_spec
    """
    if not axes:
        fig, ax = plt.subplots()
    else:
        ax = axes
    hn_periods, hn_psd = get_nhnm()
    ln_periods, ln_psd = get_nlnm()
    ax.plot(1 / hn_periods, hn_psd, color='lightgray')
    ax.plot(1 / ln_periods, ln_psd, color='lightgray')
    if plot_brune:
        moments = Mw_to_moment()
        mag_cols = cycle(sns.color_palette('muted'))
        Mws = ['-3', '-2', '-1', '0', '1']
        freq = np.logspace(-2, 4, 1000)
        for i, mom in enumerate(moments):
            mcol = next(mag_cols)
            abercrom_spec = abercrombie_disp(freq, M0=mom, Q=Q, rad=radius,
                                             sig0=sig0, Vr=Vr, Vs=Vs)
            pow_spec = disp_spec_to_psd(freq, abercrom_spec)
            ax.plot(freq, pow_spec, linestyle=':', c=mcol)
            maxx = freq[np.argmax(pow_spec)]
            maxy = np.max(pow_spec)
            ax.text(x=maxx, y=maxy * 0.98, s='$M_W$ {}'.format(Mws[i]),
                    color=mcol, fontsize=11, horizontalalignment='center')
    if plot_bands:
        if axes:
            ax.fill_betweenx(y=[-200, -40], x1=0.05, x2=1.25, color='red',
                             alpha=0.1)
            ax.fill_betweenx(y=[-200, -40], x1=1.25, x2=5., color='purple',
                             alpha=0.1)
            ax.fill_betweenx(y=[-200, -40], x1=5., x2=1000., color='green',
                             alpha=0.1)
            ax.fill_betweenx(y=[-200, -40], x1=0., x2=0.05, color='lightgray',
                             alpha=0.1)
        else:
            ax.fill_betweenx(y=[-200, -40], x1=0.05, x2=1.25, color='red',
                            alpha=0.1, label='microseism')
            ax.fill_betweenx(y=[-200, -40], x1=1.25, x2=5., color='purple',
                            alpha=0.1, label='Tremor')
            ax.fill_betweenx(y=[-200, -40], x1=5., x2=100000., color='green',
                            alpha=0.1, label=r'$M_{w} < 1$')
            ax.fill_betweenx(y=[-200, -40], x1=0., x2=0.05, color='lightgray',
                             alpha=0.1, label='Quasi-static')
            ax.set_xscale('log')
            ax.set_xlim([0.001, 1000])
            ax.set_ylim([-200, -60])
            ax.legend()
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Noise [dB]')
            plt.show()
    return ax


def plot_cascadia_sensor_noise(psd_dir, seeds, reference_seed, daily_psd_dir,
                               eq_psd=None, plot_brune=False, plot_bands=False,
                               distance=2000, Q=50, stress_drop=30,
                               Vs=5000, Vr=3000, self_noise=False):
    """
    Plot change in noise spectra for surface and deep geophone

    :param psd_dir: Directory with npz files
    :param seeds: List of full seeds for stations to plot
    :param reference_seed: Full seed of reference station
    :param eq_psd: Path to text file for eq psd
    :param plot_brune: Flag for plotting theoretical eq spectra
    :param distance: Source-receiver distance for brune (meters)
    :param Q: Quality factor for brune
    :param stress_drop: Stress drop (bar) for brune
    :param Vs: Velocity at source
    :param Vr: Velocity at receiver
    :param self_noise: Path to csv of SA-ULN self noise estimate

    :return:
    """
    ppsds = {s: {} for s in seeds}
    files = glob('{}/*.npz'.format(psd_dir))
    for s in seeds:
        ppsd = PPSD.load_npz([f for f in files if s in f][0])
        ppsds[s]['ppsd'] = ppsd
        ppsds[s]['median'] = ppsd.get_percentile(50)
        ppsds[s]['10'] = ppsd.get_percentile(10)
        ppsds[s]['90'] = ppsd.get_percentile(90)
    # Calucalte the stats on the relative psds
    calculate_psd_diff_stats(daily_psd_dir, ppsds, reference_seed)
    if plot_brune:
        fig, axes = plt.subplots(nrows=2, figsize=(9, 9), sharex='col')
    else:
        fig, axes = plt.subplots(nrows=2, figsize=(6, 10), sharex='col')
    plot_noise_and_sig_bands(axes[0], plot_brune, plot_bands, sig0=stress_drop,
                             Q=Q, radius=distance, Vs=Vs, Vr=Vr)
    if eq_psd:
        df = pd.read_csv(eq_psd)
        pds = df['periods']
        psds = df['psd']
        axes[0].plot(1 / pds, psds, color='gray', linestyle=':')
        # Interpolate onto reference freqs
        f = interp1d(pds, psds, bounds_error=False, fill_value=np.nan)
        refx, refy = ppsds[reference_seed]['median']
        diffx = refx
        diffy = f(refx) - refy
        axes[1].plot(1 / diffx, diffy, label='MEQ', color='gray',
                     linestyle=':')
    # Plot vs frequency
    axes[1].fill_between(np.array([1e-3, 1e3]), -55, 0., color='lightgray')
    for seed, psd_dict in ppsds.items():
        col_code = '.'.join(seed.split('.')[1:3])
        col = cascadia_colors[col_code]
        x, y = psd_dict['median']
        x10, y10 = psd_dict['10']
        x90, y90 = psd_dict['90']
        diff = psd_dict['diff']
        diff10 = psd_dict['diff10']
        diff90 = psd_dict['diff90']
        axes[0].fill_between(1 / x, y1=y10, y2=y90, color=col,
                             alpha=0.2)
        axes[0].plot(1 / x, y, label=seed, color=col, linewidth=0.75)
        # Plot noise reduction
        try:
            axes[1].fill_between(1 / x, y1=diff10, y2=diff90, color=col,
                                 alpha=0.2)
            axes[1].plot(1 / x, diff, color=col, linewidth=0.75)
        except ValueError:
            axes[1].fill_between(1 / ppsds[reference_seed]['median'][0],
                                 y1=diff10, y2=diff90, color=col,
                                 alpha=0.2)
            axes[1].plot(1 / ppsds[reference_seed]['median'][0], diff,
                         color=col, linewidth=0.75)
    if self_noise:
        noise_sim = np.loadtxt(self_noise, skiprows=4, delimiter=',')
        noise_f = noise_sim[:, 0]
        noise_g = noise_sim[:, 1]
        noise_psd = 20 * np.log10(noise_g * 9.8 * 10**-9)
        # Interpolate onto reference freqs
        f = interp1d(noise_f, noise_psd, bounds_error=False, fill_value=np.nan)
        refx, refy = ppsds[reference_seed]['median']
        diffy = f(1 / refx) - refy
        axes[0].plot(noise_f, noise_psd,
                     label='Self noise', color='darkgray', linestyle='-.',
                     linewidth=0.8)
        axes[1].plot(1 / refx, diffy,
                     color='darkgray', linestyle='-.', linewidth=0.8)
    # Formatting
    if plot_brune:
        fig.suptitle('Detection threshold', x=0.4, y=0.95, fontsize=20)
    else:
        fig.suptitle('Sensor comparison', x=0.4, y=0.95, fontsize=20)
    axes[0].set_xscale('log')
    axes[0].set_ylabel('Amplitude [dB]', fontsize=12)
    axes[0].set_facecolor('whitesmoke')
    axes[0].margins(0.)
    axes[0].set_ylim([-180, -40])
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Freq [Hz]', fontsize=12)
    axes[1].set_ylabel('Noise relative to SA-ULN [dB]', fontsize=12)
    axes[1].set_facecolor('whitesmoke')
    axes[1].margins(0.)
    axes[1].set_xlim([0.002, 1000])
    fig.legend()
    plt.show()
    return


def calculate_psd_diff_stats(daily_psd_dir, ppsd_dict, reference_seed):
    """
    Helper for above func to calculate daily psd relative to reference
    channel and then calculate median and percentiles
    """
    all_psds = glob('{}/*.npz'.format(daily_psd_dir))
    for seed, psd_dict in ppsd_dict.items():
        print('Calculating diffs for {}'.format(seed))
        diffs = []
        seed_files = [f for f in all_psds if f.split('/')[-1].startswith(seed)]
        for f in seed_files:
            print(f)
            day_ppsd = PPSD.load_npz(f)
            med_p, median = day_ppsd.get_percentile(50)
            try:
                diffs.append(median -
                             ppsd_dict[reference_seed]['median'][1])
            except ValueError:
                refx, refy = ppsd_dict[reference_seed]['median']
                f = interp1d(med_p, median, bounds_error=False,
                             fill_value=np.nan)
                diffs.append(f(refx) - refy)
        diffs = np.array(diffs)
        median_diff = np.percentile(diffs, 50, axis=0)
        diff_10 = np.percentile(diffs, 10, axis=0)
        diff_90 = np.percentile(diffs, 90, axis=0)
        psd_dict['diff'] = median_diff
        psd_dict['diff10'] = diff_10
        psd_dict['diff90'] = diff_90
    return


def return_Brune(M0, stress_drop):
    """Theoretical Brune displacement spectra"""
    # This fc uses dyn cm
    mom_dyn = M0 / 1e-7
    fc = 4.9e6 * 3.8 * (stress_drop / mom_dyn)**0.333
    freq = np.logspace(-2, 5, 1000)
    ang_fc = 2 * np.pi * fc
    ws = 2 * np.pi * freq
    disp_spec = np.array([brune_disp(w, M0, ang_fc) for w in ws])
    return freq, disp_spec


def brune_disp(w, M0, ang_fc):
    return M0 / (1 + (w / ang_fc)**2)


def Mw_to_moment():
    # Return moment in Nm from Mw
    moments = np.array([10**((mw + 10.7) / 0.667) for mw in
                        np.array([-3, -2, -1, 0, 1])])
    # Dyn cm to N m
    moments *= 1e-7
    return moments


def abercrombie_disp(freqs, M0, sig0, Vs, Vr, Q, rad):
    """
    Return displacement spectra

    :param freqs: frequencies at which to calculate displacement
    :param M0: Moment for this event N-m
    :param sig0: Stress drop MPa
    :param Vs: Velocity at source
    :param Vr: Velocity at receiver
    :param Q: Quality factor
    :param rad: Source-reciever distance meters
    """
    mom_dyn = M0 / 1e-7
    vs_ft = Vs * 3.28
    vr_ft = Vr * 3.28
    # Density in kg/m^3
    ps = 230 * vs_ft**0.25
    pr = 230 * vr_ft**0.25
    Om0 = ((1. * 2. / np.pi) /
           (4 * np.pi * np.sqrt(ps * pr * Vs**5 * Vr))) * (M0 / rad)
    fc = 4.9e6 * 3.8 * (sig0 / mom_dyn)**0.333
    disp_spec = (Om0 / (1 + (freqs / fc)**4)**(0.5)) * np.exp(
        -((np.pi * freqs * rad) / (Q * np.sqrt(Vs * Vr))))
    return disp_spec


def disp_spec_to_psd(freq, disp_spec):
    """
    Use Ackerley 2012 method for comparing disp spectra with Clinton&Heaton psd
    https://geoconvention.com/wp-content/uploads/abstracts/2012/273_GC2012_Estimating_the_Spectra_of_Small_Events.pdf
    :return:
    """
    # To max accel
    disp_spec = disp_spec * (2 / np.sqrt(2)) * freq
    # Displacement to Acceleration
    ang_f = 2 * np.pi * freq
    vel_spec = ang_f * disp_spec
    acc_spec = ang_f * vel_spec
    xmax = (acc_spec / np.sqrt(np.pi / 2))**2 / (freq / np.sqrt(2))
    xmax = np.log10(xmax)
    xmax *= 10
    return xmax


def plot_meq_brune(stress_drop=10, ax=None):
    """
    Overview plot of Brune disp spectra for small Mw events

    :param stress_drop: Stress drop in bar
    :return:
    """
    cols = cycle(sns.color_palette('muted'))
    moments = Mw_to_moment()
    Mws = ['-3', '-2', '-1', '0', '1']
    print(moments)
    if not ax:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
    for i, mom in enumerate(moments):
        # model above uses dyn cm, I guess
        c = next(cols)
        freq, disp_spec = return_Brune(mom, stress_drop)
        ax.plot(freq, disp_spec, label=Mws[i], color=c)
        ax.text(x=2, y=np.max(disp_spec) * 1.3, s='$M_W$ {}'.format(Mws[i]),
                color=c, fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Moment [Nm]')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_xlim(right=1e5)
    ax.set_ylim(top=1e12)
    ax.set_facecolor('whitesmoke')
    # plt.legend(title='$M_W$')
    # plt.show()
    return ax


def plot_stacked_wavs(events, streams, station, prepick, postpick,
                      colors=None, labels=None):
    """
    Plot traces from given station for number of events

    :param events: list of events with picks
    :param streams: list of streams in order of events
    :param station: Station to plot
    :param prepick: Seconds before to clip
    :param postpick: Seconds after to clip
    :param colors: Colors of each trace, optional
    :param labels: List of labels for each trace

    :return:
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    labs = [r'$Time$']
    for i, (ev, st) in enumerate(zip(events, streams)):
        try:
            raw = st.select(station=station)[0]
        except IndexError:
            print('No wav for {}'.format(ev))
            labs.append('')
            continue
        try:
            ptime = [p.time for p in ev.picks
                     if p.waveform_id.station_code == station
                     and p.phase_hint == 'P'][0]
        except IndexError:
            print('No pick for {}'.format(ev))
            labs.append('')
            continue
        # Filter
        raw = raw.copy().filter('bandpass', freqmin=1000.,
                                freqmax=12000., corners=3)
        # Trim
        data = raw.trim(starttime=ptime - prepick,
                        endtime=ptime + postpick).data
        # Microsecond vector
        time_v = np.arange(data.shape[0]) / raw.stats.sampling_rate * 1e3
        data /= np.max(data)  # Normalize
        # Plot
        if colors:
            col = colors[i]
        else:
            col = 'k'
        if labels:
            lab = labels[i]
        else:
            lab = ''
        ax.plot(time_v, data - i, color=col, label=lab)
        labs.append('{}:{}:{}.{}'.format(ptime.hour,
                                      ptime.minute,
                                      ptime.second,
                                      ptime.microsecond))
    ax.set_yticks(-np.arange(len(streams) + 1) + 1)
    ax.set_xlabel('Milliseconds')
    ax.set_yticklabels(labs, fontsize=12)
    fig.suptitle(station, fontsize=20, fontweight='bold')
    fig.legend()
    plt.show()
    return


def mseed_to_mat(ms_dir, datestr='202110*'):
    """
    Convert a directory of mseed files to matlab structs with a time and
    data key
    """
    mseeds = glob('{}/*C?..{}*.mseed'.format(ms_dir, datestr))
    mseeds.sort()
    for mseed in mseeds:
        tr = read(mseed)[0]
        mat_name = mseed.replace('mseed', 'mat')
        mdict = {k: str(v) for k, v in tr.stats.items()}
        mdict['data'] = tr.data
        mdict['times'] = tr.times(type='timestamp')
        print('Saving {}'.format(mat_name))
        savemat(mat_name, mdict)
    return