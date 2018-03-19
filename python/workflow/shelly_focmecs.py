#!/usr/bin/python

"""
Functions for running Shelly et al. focal mechanism methods for MF detections
"""

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from multiprocessing import Pool
from scipy.signal import argrelmax
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from obspy import read
from eqcorrscan.core.match_filter import normxcorr2
from eqcorrscan.utils.pre_processing import shortproc

def make_stream_lists(cat_temps, cat_dets, temp_dir, det_dir):
    det_streams = []
    temp_streams = []
    print('Globbing waveforms')
    temp_wavs = glob('{}/*'.format(temp_dir))
    print('Template directories have the following pattern:\n{}'.format(
        temp_wavs[0].split('/')[-1]))
    det_wavs = glob('{}/*'.format(det_dir))
    print('Detection directories have the following pattern:\n{}'.format(
        det_wavs[0].split('/')[-1]))
    # Templates
    print('Creating template streams')
    for ev in list(cat_temps.events):
        wdir = [wavs for wavs in temp_wavs if
                wavs.split('/')[-1].split('_')[0] ==
                ev.resource_id.id.split('/')[-1]]
        if len(wdir) > 0:
            print('Adding template wavs from {}'.format(wdir[0]))
            temp_streams.append(read('{}/*'.format(wdir[0])))
        else:
            cat_temps.events.remove(ev)
    # Detections
    print('Creating detection streams')
    det_cat_ids = [ev.resource_id.id.split('/')[-1] for ev in cat_dets]
    det_wavs_only = [wav for wav in det_wavs if
                              wav.split('/')[-1] in det_cat_ids]
    det_wav_dict = {wav.split('/')[-1]: wav for wav in det_wavs_only}
    for ev in list(cat_dets.events):
        try:
            print('Reading wavs for ev {}'.format(ev.resource_id.id))
            det_streams.append(read('{}/*'.format(
                det_wav_dict[ev.resource_id.id.split('/')[-1]])))
        except KeyError:
            print(ev.resource_id.id)
            cat_dets.events.remove(ev)
    return temp_streams, det_streams

def _rel_polarity(data1, data2, min_cc, debug=0):
    """
    Compute the relative polarity between two traces

    :type data1: numpy.ndarray
    :param data1: Template data
    :type data2: numpy.ndarray
    :param data2: Detection data
    :type min_cc: float
    :param min_cc: Minimum accepted cros-correlation value for a polarity pick

    :returns:
        Value of the relative polarity measurement between template and
        detection on this sta/chan/phase
    :rtype: float
    """
    if not data1.any() or not data2.any():
        return 0.0
    ccc = normxcorr2(data1, data2)[0]
    raw_max = np.argmax(np.abs(ccc))
    if raw_max == 0:
        print('Max absolute data point is at end of ccc array. Skipping.')
        return 0.0
    elif raw_max == np.max(ccc.shape) - 1:
        print('Max absolute data point is at end of ccc array. Skipping.')
        return 0.0
    elif ccc[raw_max] < min_cc:
        return 0.0
    sign = np.sign(ccc[raw_max])
    # Find pks
    pk_locs = argrelmax(np.abs(ccc), order=2)[0]
    pk_ind = np.where(np.equal(raw_max, pk_locs))[0][0]
    # Now find the two peaks either side of the max peak
    if pk_ind == 0:
        # If max peak is first peak...
        second_pk_vals = np.abs(ccc)[np.array([pk_locs[pk_ind + 1]])]
        sec_pk_locs = np.array([pk_locs[pk_ind + 1]])
    elif pk_ind == np.max(pk_locs.shape) - 1:
        # If max peak is last peak...
        second_pk_vals = np.abs(ccc)[np.array([pk_locs[pk_ind - 1]])]
        sec_pk_locs = np.array([pk_locs[pk_ind - 1]])
    else:
        # All other cases
        second_pk_vals = np.abs(ccc)[np.array([pk_locs[pk_ind - 1],
                                               pk_locs[pk_ind + 1]])]
        sec_pk_locs = np.array([pk_locs[pk_ind - 1],
                                pk_locs[pk_ind + 1]])
    if debug > 0:
        plt.plot(np.abs(ccc), color='k')
        for loc in sec_pk_locs:
            plt.axvline(loc, color='blue', linestyle='-.')
        plt.axvline(raw_max, color='r')
        plt.axvline(pk_locs[pk_ind], color='grey', linestyle='--')
        plt.show()
        plt.close('all')
    rel_pol = sign * np.max(second_pk_vals)
    return rel_pol

def _prepare_data(template_streams, detection_streams, template_cat,
                  detection_cat, temp_traces, det_traces, filt_params,
                  phases, corr_dict, cores):
    # Filter data
    filt_temps = []
    filt_dets = []
    print('Filtering data')
    for st in template_streams:
        filt_temps.append(shortproc(st.copy(), filt_params['lowcut'],
                                    filt_params['highcut'],
                                    filt_params['filt_order'],
                                    filt_params['samp_rate'],
                                    num_cores=cores))
    for st in detection_streams:
        filt_dets.append(shortproc(st.copy(), filt_params['lowcut'],
                                   filt_params['highcut'],
                                   filt_params['filt_order'],
                                   filt_params['samp_rate'],
                                   num_cores=cores))
    # Populate trace arrays for all picks
    for i, (st, ev) in enumerate(zip(filt_temps, template_cat.events)):
        for pk in ev.picks:
            sta = pk.waveform_id.station_code
            chan = pk.waveform_id.channel_code
            stch = '{}.{}'.format(sta, chan)
            if pk.phase_hint not in phases:
                continue
            hint = pk.phase_hint
            try:
                tr = st.select(station=sta, channel=chan)[0]
            except IndexError:
                continue
            # Put this data in corresponding row of the array
            try:
                temp_traces[hint][stch][i] = tr.slice(
                    starttime=pk.time - corr_dict[hint]['pre_pick'],
                    endtime=pk.time + corr_dict[hint]['post_pick'],
                    nearest_sample=False).data
            except ValueError:
                # Clip last sample off data in this case?
                temp_traces[hint][stch][i] = tr.slice(
                    starttime=pk.time - corr_dict[hint]['pre_pick'],
                    endtime=pk.time + corr_dict[hint]['post_pick'],
                    nearest_sample=False).data[:-1]
    for i, (st, ev) in enumerate(zip(filt_dets, detection_cat.events)):
        for pk in ev.picks:
            sta = pk.waveform_id.station_code
            chan = pk.waveform_id.channel_code
            stch = '{}.{}'.format(sta, chan)
            if pk.phase_hint not in phases:
                continue
            try:
                tr = st.select(station=sta, channel=chan)[0]
            except IndexError:
                continue
            # Put this data in corresponding row of the array
            try:
                det_traces[hint][stch][i] = tr.slice(
                    starttime=pk.time - corr_dict[hint]['pre_pick'] -
                             corr_dict[hint]['shift_len'],
                    endtime=pk.time + corr_dict[hint]['post_pick'] +
                            corr_dict[hint]['shift_len'],
                    nearest_sample=False).data
            except ValueError:
                # Clip last sample off in this case
                det_traces[hint][stch][i] = tr.slice(
                    starttime=pk.time - corr_dict[hint]['pre_pick'] -
                             corr_dict[hint]['shift_len'],
                    endtime=pk.time + corr_dict[hint]['post_pick'] +
                            corr_dict[hint]['shift_len'],
                    nearest_sample=False).data[:-1]
    return temp_traces, det_traces

def _stachan_loop(phase, stachan, temp_traces, det_traces, min_cc, debug):
    """
    Inner loop to parallel over stachan matrices
    :return:
    """
    pol_array = np.zeros((len(det_traces), len(temp_traces)))
    print('Looping stachan: {}'.format(stachan))
    for m in range(len(temp_traces)):
        for n in range(len(det_traces)):
            pol = _rel_polarity(temp_traces[m],
                                det_traces[n],
                                min_cc, debug)
            pol_array[n][m] = pol
    return phase, stachan, pol_array

def make_corr_matrices(template_streams, detection_streams, template_cat,
                       detection_cat, corr_dict, min_cc, filt_params,
                       phases=('P', 'S'), cores=4, debug=0):
    """
    Create the correlation matrices
    :type template_streams: list
    :param template_streams: List of all template streams in order of events in
        temp_cat
    :type detection_streams: list
    :param detection_streams: List of all template streams in order of events
        in temp_cat
    :type template_cat: obspy.core.event.Catalog
    :param template_cat: Catalog of template events with event and pick info
    :type detection_cat: obspy.core.event.Catalog
    :param detection_cat: Catalog of detected events with event and pick info
    :type corr_dict: dict
    :param corr_dict: Nested dictionary of parameters for the correlation.
        Upper level keys are 'P' and/or 'S'. Beneath this are the keys:
        'pre_pick', 'post_pick', 'shift_len', and 'min_cc'.
    :type filt_params: dict
    :param filt_params: Dictionary containing filtering parameters for
        waveforms. Should include keys: 'lowcut', 'highcut', 'filt_order'
        and 'samp_rate' to be fed to shortproc.
    :type phases: list
    :param phases: List of phases used: ['P'], ['S'], or ['P', 'S']
    :type cores: int
    :param cores: Number of cores to use for multiprocessing
    :return:
    """
    # Get unique sta.chan combos
    stachan = ['{}.{}'.format(pk.waveform_id.station_code,
                              pk.waveform_id.channel_code)
               for ev in template_cat for pk in ev.picks]
    stachan.extend(['{}.{}'.format(pk.waveform_id.station_code,
                                   pk.waveform_id.channel_code)
                    for ev in detection_cat for pk in ev.picks])
    stachans = list(set(stachan))
    print(len(stachans), stachans)
    # Establish length of template and detection traces in samples
    s_rate = detection_streams[0][0].stats.sampling_rate
    temp_len = {}; det_len = {}; temp_traces = {}; det_traces = {}
    for p in phases:
        temp_len[p] = int((corr_dict[p]['pre_pick'] +
                           corr_dict[p]['post_pick']) * s_rate)
        det_len[p] = int((corr_dict[p]['pre_pick'] +
                          corr_dict[p]['post_pick'] +
                          (2 * corr_dict[p]['shift_len'])) * s_rate)
    print('Preallocating arrays')
    # Set up zero arrays for trace data
    for p in phases:
        temp_traces[p] = {stachan: np.zeros((len(template_cat), temp_len[p]))
                          for stachan in stachans}
        det_traces[p] = {stachan: np.zeros((len(detection_cat), det_len[p]))
                         for stachan in stachans}
    # Populate trace arrays for all picks
    # Pass to _prepare_data function to clean this up
    print('Preparing data for processing')
    temp_traces, det_traces = _prepare_data(
        template_streams, detection_streams, template_cat, detection_cat,
        temp_traces, det_traces, filt_params, phases, corr_dict, cores=cores)
    # Calculate relative polarities
    if cores > 1:
        print('Starting up pool')
        rel_pols = {}
        pool = Pool(processes=cores)
        for phase in phases:
            results = [pool.apply_async(
                _stachan_loop,
                (phase, stachan,
                 temp_traces[phase][stachan],
                 det_traces[phase][stachan]),
                 {'min_cc': min_cc, 'debug': debug})
                for stachan in stachans]
            pool.close()
            rel_pols[phase] = [p.get() for p in results]
    else:
        # Python loop..?
        rel_pols = []
        for phase in phases:
            for stachan in stachans:
                print('Looping stachan: {}'.format(stachan))
                pol_array = np.zeros((len(detection_streams),
                                      len(template_streams)))
                for m in range(len(template_streams)):
                    for n in range(len(detection_streams)):
                        pol = _rel_polarity(temp_traces[phase][stachan][m],
                                            det_traces[phase][stachan][n],
                                            min_cc, debug)
                        pol_array[n][m] = pol
                rel_pols.append((phase, stachan, pol_array))
    return rel_pols

def svd_matrix(rel_pol_dict):
    """
    Make the matrix of left singular vectors from all sta/chan/phase combos
    :param rel_pol_dict:
    :return:
    """
    phases = rel_pol_dict.keys()
    stachans = rel_pol_dict[phases[0]].keys()
    # Find how many sta/chan/phase combos we have
    if set(('P','S')).issubset(phases):
        combos = len(stachans) * 2
    else:
        combos = len(stachans)
    n = rel_pol_dict['P'][stachans[0]].shape[0]
    svd_mat = np.zeros((n, combos))
    for phase, stachan_dict in rel_pol_dict.items():
        for stachan, stachan_mat in stachan_dict.items():
            u, s, v = np.linalg.svd(stachan_mat, full_matrices=True)
            svd_mat.hstack(svd_mat, u[0])
    return svd_mat

def cluster_svd_mat(svd_mat, metric='cosine', show=False):
    """
    Function to cluster the rows of the nxk matrix of relative polarity
    measurements
    :return:
    """
    Z = linkage(svd_mat, method='single', metric='cosine')
    if show:
        dendrogram(Z)
    indices = fcluster(Z, t=1)
    return