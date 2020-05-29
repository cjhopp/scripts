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
    network_sta_lta = trig_p['network_specific_params']
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
                print('Reading in {}'.format(w))
                st += read(w)
            elif seed_id[-1] == 'Z':  # Triggering on Z comps only
                print('Reading in {}'.format(w))
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
            except KeyError as e:  # Take network general parameters
                seed_params = network_sta_lta[tr.id.split('.')[0]]
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
                          sta_lta_params)
    return trigs


def plot_triggers(triggers, st, cft_stream, params):
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
            tps = params[sid]
            tr_raw = st_slice.select(id=sid)[0]
            tr_cft= cft_slice.select(id=sid)[0].data
            time_vect = np.arange(tr_cft.shape[0]) * tr_raw.stats.delta
            ax[i].plot(time_vect,
                       tr_raw.data / np.max(tr_raw.data) * 0.6 * np.max(tr_cft),
                       color='k')
            ax[i].plot(time_vect, tr_cft.data, color='gray')
            ax[i].axhline(tps['thr_on'], linestyle='--', color='r')
            ax[i].axhline(tps['thr_off'], linestyle='--', color='b')
            bbox_props = dict(boxstyle="round,pad=0.2", fc="white",
                              ec="k", lw=1)
            ax[i].annotate(s=sid, xy=(0.0, 0.8), xycoords='axes fraction',
                           bbox=bbox_props, ha='center')
            ax[i].set_yticks([])
        ax[i].set_xlabel('Time [s]', fontsize=14)
        plt.show()
    return