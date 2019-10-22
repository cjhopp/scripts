#!/usr/bin/python

"""
Functions for reading/writing and processing waveform data
"""
import itertools
import os

import numpy as np

from glob import glob
from obspy import read, Stream, Catalog
from obspy.signal.cross_correlation import xcorr_pick_correction
from surf_seis.vibbox import vibbox_preprocess


def read_raw_wavs(wav_dir):
    """Read all the waveforms in the given directory to a dict"""
    mseeds = glob('{}/*'.format(wav_dir))
    wav_dict = {}
    for ms in mseeds:
        eid = ms.split('/')[-1].rstrip('_raw.mseed')
        try:
            wav_dict[eid] = read(ms)
        except TypeError as e:
            print(e)
    return wav_dict

def extract_event_signal(wav_dir, catalog, prepick=0.0001, duration=0.01):
    """
    Trim around pick times and filter waveforms

    :param wav_dir: Path to waveform files
    :param catalog: obspy.core.Catalog
    :param seconds: Length to trim to
    :param filt: Dictionary of filter parameters

    :return:
    """
    streams = {}
    wav_dict = read_raw_wavs(wav_dir)
    for ev in catalog:
        ot = ev.origins[-1].time
        t_stamp = ot.strftime('%Y%m%d%H%M%S%f')
        try:
            st = wav_dict[t_stamp]
        except KeyError as e:
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
            st_sta = st.select(station=sta, channel=chan).copy()
            st_sta.trim(starttime=pk.time - prepick, endtime=pk.time + duration)
            new_st += st_sta
        if len(new_st) > 0:
            streams[t_stamp] = new_st
    return streams

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

def plot_pick_corrections(catalog, stream_dir, plotdir):
    """
    Hard coded wrapper on xcorr pick correction to be fixed on Monday 10-21
    :param catalog: Catalog of events to generate plots for
    :param stream_dir: Path to directory of *raw.mseed files
    :param plotdir: Path to root directory for plots

    :return:
    """
    for ev1, ev2 in itertools.combinations(catalog, r=2):
        eid1 = ev1.resource_id.id.split('/')[-1]
        eid2 = ev2.resource_id.id.split('/')[-1]
        for pk1 in ev1.picks:
            sta = pk1.waveform_id.station_code
            chan = pk1.waveform_id.channel_code
            stachandir = '{}/{}.{}'.format(plotdir, sta, chan)
            if not os.path.isdir(stachandir):
                os.mkdir(stachandir)
            try:
                st1 = read('{}/{}_raw.mseed'.format(stream_dir, eid1))
                st2 = read('{}/{}_raw.mseed'.format(stream_dir, eid2))
            except FileNotFoundError as e:
                print(e)
                continue
            tr1 = st1.select(station=sta, channel=chan)
            tr2 = st2.select(station=sta, channel=chan)
            pk2 = [pk for pk in ev2.picks
                   if pk.waveform_id.station_code == sta
                   and pk.waveform_id.channel_code == chan]
            if len(pk2) > 0 and len(tr2) > 0:
                try:
                    xcorr_pick_correction(pk1.time, tr1[0], pk2[0].time, tr2[0],
                                          t_before=0.00003, t_after=0.00015,
                                          cc_maxlag=0.0001, plot=True,
                                          filter='bandpass',
                                          filter_options={'corners': 5,
                                                          'freqmax': 42000.,
                                                          'freqmin': 2000.},
                                          filename='{}/{}.{}/{}_{}_{}.{}.pdf'.format(
                                              plotdir, sta, chan, eid1, eid2,
                                              sta, chan
                                          ))
                except Exception as e:
                    print(e)
                    continue
    return
