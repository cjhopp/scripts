#!/usr/bin/python

"""
Functions for running Shelly et al. focal mechanism methods for MF detections
"""

import numpy as np

from glob import glob
from multiprocessing import Pool
from scipy.signal import argrelmax
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from eqcorrscan.core.match_filter import normxcorr2


def _rel_polarity(data1, data2, min_cc, samp_rate, i, j):
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
    ccc = normxcorr2(data1, data2)[0]
    raw_max = np.argmax(ccc)
    if ccc[raw_max] < min_cc:
        return 0.0
    sign = np.sign(ccc[raw_max])
    # Find pks
    pk_locs = argrelmax(np.abs(ccc), order=int(0.02 * samp_rate))[0]
    pk_ind = np.where(pk_locs == raw_max)[-1]
    # Now find the two peaks either side of the max peak
    second_pk_vals = np.abs(ccc)[0][pk_locs[np.array([pk_ind - 1,
                                                      pk_ind + 1])]]
    rel_pol = sign * np.max(second_pk_vals)
    return rel_pol, i, j

def _prepare_data(template_streams, detection_streams, template_cat,
                  detection_cat, temp_traces, det_traces, phases, corr_dict):
    # Populate trace arrays for all picks
    for i, (st, ev) in enumerate(zip(template_streams, template_cat.events)):
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
    for i, (st, ev) in enumerate(zip(detection_streams, detection_cat.events)):
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

def _stachan_loop():
    """
    Inner loop to parallel over stachan matrices
    :return:
    """

    return

def make_corr_matrices(template_streams, detection_streams, template_cat,
                       detection_cat, corr_dict, min_cc, phases=('P', 'S'),
                       cores=4, debug=0):
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
    # Preassign umbrella dict with nxm arrary of zeros for each sta/chan/phase
    rel_pol_dict = {ph: {stachan: np.zeros((len(detection_cat),
                                        len(template_cat)))}
                    for ph in phases for stachan in stachans}
    # Populate trace arrays for all picks
    # Pass to _prepare_data function to clean this up
    print('Preparing data for processing')
    temp_traces, det_traces = _prepare_data(
        template_streams, detection_streams, template_cat, detection_cat,
        temp_traces, det_traces, phases, corr_dict)
    # Calculate relative polarities
    if cores > 1:
        print('Starting up pool')
        pool = Pool(processes=cores)
        for phase in phases:
            for stachan in stachans:
                print('Calculating relative pols for:'
                      '\nPhase: {}\nStachan: {}'.format(phase, stachan))
                results = [pool.apply_async(
                    _rel_polarity,
                    (temp_traces[phase][stachan][m],
                     det_traces[phase][stachan][n],),
                    {'min_cc': min_cc, 'samp_rate': s_rate, 'i': m, 'j':n})
                    for m in range(len(template_streams))
                    for n in range(len(detection_streams))]
                pool.close()
                rel_pols = [p.get() for p in results]
                for pol in rel_pols:
                    rel_pol_dict[phase][stachan][pol[2]][pol[1]] = pol[0]
    else:
        # Python loop..?
        for phase in phases:
            for stachan in stachans:
                for m in range(len(template_streams)):
                    for n in range(len(detection_streams)):
                        pol = _rel_polarity(temp_traces[phase][stachan][m],
                                            det_traces[phase][stachan][n],
                                            min_cc, s_rate, m, n)
                        rel_pol_dict[phase][stachan][m][n] = pol[0]
    return rel_pol_dict

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