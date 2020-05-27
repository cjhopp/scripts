#!/usr/bin/python

"""
Set of functions wrapping obspy triggering and phasepapy picking/association
"""

import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from obspy import UTCDateTime, read, Stream
from obspy.signal.trigger import coincidence_trigger, plot_trigger
from eqcorrscan.utils.pre_processing import dayproc


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def trigger(param_file, plot=False):
    """
    Wrapper on obspy coincidence trigger for a directory of waveforms

    :param param_file: Path to a yaml with the necessary parameters
    :param plot: Plotting flag

    :return:
    """
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
    trig_p = paramz['Trigger']
    pick_p = paramz['Picker']
    sta_lta_params = trig_p['channel_specific_params']
    trigs = []
    start = UTCDateTime(trig_p['start_time']).datetime
    end = UTCDateTime(trig_p['end_time']).datetime
    for date in date_generator(start.date(), end.date()):
        print('Triggering on {}'.format(date))
        utcdto = UTCDateTime(date)
        jday = utcdto.julday
        day_wavs = glob('{}/**/*{}.ms'.format(
            paramz['General']['wav_directory'], jday), recursive=True)
        st = Stream()
        for w in day_wavs:
            seed_parts = os.path.basename(w).split('.')
            seed_id = '.'.join(seed_parts[:-3])
            if seed_id in sta_lta_params:
                # TODO Do some checks for continuity, gaps, etc...
                st += read(w)
        st = st.merge(fill_value='interpolate')
        # Filter and downsample the wavs
        st = dayproc(st, lowcut=trig_p['lowcut'], num_cores=4,
                     highcut=trig_p['highcut'], filt_order=trig_p['corners'],
                     samp_rate=trig_p['sampling_rate'], starttime=utcdto)
        # Precompute characteristic functions for each station as tuned manually
        trigger_stream = Stream()
        for tr in st:
            try:
                seed_params = sta_lta_params[tr.id]
            except KeyError as e:
                print('No trigger params for {}'.format(tr.id))
                continue
            trigger_stream += tr.copy().trigger(
                type='recstalta',
                nsta=int(seed_params['sta'] * tr.stats.sampling_rate),
                nlta=int(seed_params['lta'] * tr.stats.sampling_rate))
        # Coincidence triggering on precomputed characteristic funcs
        trigs += coincidence_trigger(
            trigger_type=None, stream=trigger_stream,
            thr_on=seed_params['thr_on'],
            thr_off=seed_params['thr_off'],
            thr_coincidence_sum=trig_p['coincidence_sum'],
            details=True)
        if plot:
            plot_triggers(trigs, st, trigger_stream,
                          trig_p['threshold_on'],
                          trig_p['threshold_off'])
    return trigs


def plot_triggers(triggers, st, cft_stream, thr_on, thr_off):
    """Helper to plot triggers, traces and characteristic funcs"""
    for trig in triggers:
        seeds = trig['trace_ids']
        # Clip around trigger time
        st_slice = st.slice(starttime=trig['time'] - 3,
                            endtime=trig['time'] + 10)
        cft_slice = cft_stream.slice(starttime=trig['time'] - 3,
                                     endtime=trig['time'] + 10)
        fig, ax = plt.subplots(nrows=len(seeds), sharex='col')
        fig.suptitle('Detection: {}'.format(trig['time']))
        fig.subplots_adjust(hspace=0.)
        for i, sid in enumerate(seeds):
            tr_raw = st_slice.select(id=sid)[0]
            tr_cft= cft_slice.select(id=sid)[0].data
            ax[i].plot(tr_raw.data / np.max(tr_raw.data) * 0.6 * np.max(tr_cft),
                       color='k')
            ax[i].plot(tr_cft.data, color='gray')
            ax[i].axhline(thr_on, linestyle='--', color='r')
            ax[i].axhline(thr_off, linestyle='--', color='b')
            ax[i].annotate(text=sid, xy=(0.1, 0.8), xycoords='axes fraction')
            ax[i].set_yticks([])
        plt.show()
    return