#!/usr/bin/python

"""
Functions for running Shelly et al. focal mechanism methods for MF detections
"""
# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import random
import unittest
import marshal as pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pyproj
try:
    import mplstereonet
    import colorlover as cl
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
except:
    print('Youre probably on the server. Dont try any plotting')
from glob import glob
from itertools import cycle
from operator import attrgetter
from collections import OrderedDict
from multiprocessing import Pool
from numpy.linalg import LinAlgError
from scipy.signal import argrelmax
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from obspy import read, Catalog, read_events, Stream
from obspy.core.event import (Comment, Event, Arrival, Origin, Pick,
                              WaveformStreamID, OriginUncertainty,
                              QuantityError)
from eqcorrscan.core.match_filter import normxcorr2
from eqcorrscan.utils.pre_processing import shortproc
from eqcorrscan.utils.synth_seis import seis_sim
from eqcorrscan.utils.plotting import xcorr_plot
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

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


def _rel_polarity(data1, data2, min_cc, m, n, stachan, phase, plotdir,
                  debug=0):
    """
    Compute the relative polarity between two traces

    :type data1: numpy.ndarray
    :param data1: Template data
    :type data2: numpy.ndarray
    :param data2: Detection data
    :type min_cc: float
    :param min_cc: Minimum accepted cros-correlation value for a polarity pick
    :type m: int
    :param m: Index of template event
    :type n: int
    :param n: Index of detection event
    :type stachan: str
    :param stachan: Station/channel string
    :type phase: str
    :param phase: Phase string

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
        if debug > 1:
            print('Max absolute data point is at end of ccc array. Skipping.')
        return 0.0
    elif raw_max == np.max(ccc.shape) - 1:
        if debug > 1:
            print('Max absolute data point is at end of ccc array. Skipping.')
        return 0.0
    elif np.abs(ccc[raw_max]) < min_cc:
        if debug > 1:
            print('Max absolute correlation below threshold. Skipping.')
        return 0.0
    # Sign of max abs corr
    sign = np.sign(ccc[raw_max])
    # Find pks
    pk_locs = argrelmax(np.abs(ccc), order=2)[0]
    # Make sure theres more than one peak
    if pk_locs.shape[0] <= 1:
        if debug > 0:
            print('Only one peak found. Skip this polarity.')
        return 0.0
    # Find index of the maximum peak in pk_locs
    try:
        pk_ind = np.where(np.equal(raw_max, pk_locs))[0][0]
    except IndexError:
        print('Raw max ccc not found by argrelmax?? Ignore this pair.')
        return 0.0
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
    rel_pol = sign * np.min(np.abs(ccc[raw_max]) - second_pk_vals)
    if debug > 1:
        print('Relative polarity: {}'.format(rel_pol))
        _rel_pol_plot(data1, data2, ccc, sec_pk_locs, raw_max, pk_locs,
                      pk_ind, rel_pol, second_pk_vals, m, n, stachan, phase,
                      plotdir)
    return rel_pol


def _rel_pol_plot(temp, image, ccc, sec_pk_locs, raw_max, pk_locs, pk_ind,
                  rel_pol, second_pk_vals, m, n, stachan, phase, plotdir):
    # Plot shifted waveforms and the correlation coefficient with time
    fig, (ax1, ax2) = plt.subplots(2)
    # Plot the shifted waveforms
    shift = np.abs(ccc).argmax()
    cc = ccc[shift]
    xi = np.arange(len(image))
    ax1.plot(xi, image / abs(image).max(), 'k', lw=1.3, label='Image')
    xt = np.arange(len(temp)) + shift
    ax1.plot(xt, temp / abs(temp).max(), 'r', lw=1.1, label='Template')
    ax1.set_title('Shift=%s, Correlation=%s' % (shift, cc))
    handles, labels = ax1.get_legend_handles_labels()
    lgd1 = ax1.legend(handles, labels, loc='upper center',
                     bbox_to_anchor=(1.2, 0.5))
    # Now plot the cc vector as in shelly 2016
    ax2.plot(np.abs(ccc), color='purple')
    for loc in sec_pk_locs:
        ax2.axvline(loc, color='blue', linestyle='-.', label='Secondary peak')
    ax2.axvline(raw_max, color='r', label='Absolute CCC max')
    ax2.axhline(np.max(second_pk_vals), color='gray', linestyle='-.',
                label='Secondary peak value')
    ax2.set_title('Relative polarity weight: {}'.format(rel_pol))
    handles, labels = ax2.get_legend_handles_labels()
    lgd2 = ax2.legend(handles, labels, loc='upper center',
                      bbox_to_anchor=(1.3, 0.5))
    plt.tight_layout()
    fig.savefig('{}/{}_{}_temp_{}_det_{}.png'.format(plotdir, stachan, phase,
                                                     m, n),
                bbox_extra_artists=(lgd1, lgd2), bbox_inches='tight')
    plt.close('all')
    return fig


def _prepare_data(template_streams, detection_streams, template_cat,
                  detection_cat, temp_traces, det_traces, filt_params,
                  phases, corr_dict, cores):
    # Filter data
    filt_temps = []
    filt_dets = []
    print('Filtering data')
    print('Filtering templates')
    for st in template_streams:
        try:
            filt_temps.append(shortproc(st.copy(), filt_params['lowcut'],
                                        filt_params['highcut'],
                                        filt_params['filt_order'],
                                        filt_params['samp_rate'],
                                        num_cores=cores))
        except ValueError:
            # If there's an issue with filtering the streams, add an empty
            # one. If the corresponding trace doesn't exist in the next loop
            # that gets accounted for by preallocation of zero arrays
            filt_temps.append(Stream())
    print('Filtering detections')
    for st in detection_streams:
        try:
            filt_dets.append(shortproc(st.copy(), filt_params['lowcut'],
                                       filt_params['highcut'],
                                       filt_params['filt_order'],
                                       filt_params['samp_rate'],
                                       num_cores=cores))
        except ValueError:
            filt_dets.append(Stream())
    # Populate trace arrays for all picks
    print('Populating trace array for templates')
    for i, (st, ev) in enumerate(zip(filt_temps, template_cat.events)):
        for pk in ev.picks:
            sta = pk.waveform_id.station_code
            chan = pk.waveform_id.channel_code
            stch = '{}.{}'.format(sta, chan)
            # Run some checks for wierdness
            if pk.phase_hint not in phases:
                continue
            hint = pk.phase_hint
            try:
                tr = st.select(station=sta, channel=chan)[0]
            except IndexError:
                continue
            if hint == 'P' and stch[-1] != 'Z':
                print('You have a P pick on a horizontal channel. Skipping')
                continue
            # Put this data in corresponding row of the array
            try:
                print(hint, stch)
                temp_traces[hint][stch][i] = tr.slice(
                    starttime=pk.time - corr_dict[hint]['pre_pick'],
                    endtime=pk.time + corr_dict[hint]['post_pick'],
                    nearest_sample=False).data
            except ValueError:
                # Try to clip last sample off data in this case
                try:
                    temp_traces[hint][stch][i] = tr.slice(
                        starttime=pk.time - corr_dict[hint]['pre_pick'],
                        endtime=pk.time + corr_dict[hint]['post_pick'],
                        nearest_sample=False).data[:-1]
                except ValueError:
                    # Just ignore now. We tried.
                    continue
    print('Populating trace array for detections')
    for i, (st, ev) in enumerate(zip(filt_dets, detection_cat.events)):
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
            if hint == 'P' and stch[-1] != 'Z':
                print('You have a P pick on a horizontal channel. Skipping')
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
                try:
                    det_traces[hint][stch][i] = tr.slice(
                        starttime=pk.time - corr_dict[hint]['pre_pick'] -
                                 corr_dict[hint]['shift_len'],
                        endtime=pk.time + corr_dict[hint]['post_pick'] +
                                corr_dict[hint]['shift_len'],
                        nearest_sample=False).data[:-1]
                except ValueError:
                    try:
                        det_traces[hint][stch][i] = tr.slice(
                            starttime=pk.time - corr_dict[hint]['pre_pick'] -
                                     corr_dict[hint]['shift_len'],
                            endtime=pk.time + corr_dict[hint]['post_pick'] +
                                    corr_dict[hint]['shift_len'],
                            nearest_sample=False).data[:-2]
                    except ValueError:
                        # Tried real hard again. Ignore.
                        continue
    return temp_traces, det_traces


def _stachan_loop(phase, stachan, temp_traces, det_traces, min_cc, plotdir,
                  debug):
    """
    Inner loop to parallel over stachan matrices
    :return:
    """
    pol_array = np.zeros((len(det_traces), len(temp_traces)))
    print('Looping stachan: {}'.format(stachan))
    for m in range(len(temp_traces)):
        for n in range(len(det_traces)):
            pol = _rel_polarity(temp_traces[m], det_traces[n], min_cc, m, n,
                                stachan, phase, plotdir, debug)
            pol_array[n][m] = pol
    return phase, stachan, pol_array


def make_corr_matrices(template_streams, detection_streams, template_cat,
                       detection_cat, corr_dict, filt_params,
                       phases=('P', 'S'), cores=4, debug=0, save=False,
                       plotdir='.'):
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
    :type debug: int
    :param debug: Debugging level for print output
    :type save: bool or str
    :param save: If a directory is provided, will save the relative polarity
        array to a pickle object called 'rel_pols.pkl' for later, repeated use
        by svd_matrix()
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
    ph_stachans = {}
    ph_stachans['P'] = [stachan for stachan in stachans if stachan[-1] == 'Z']
    ph_stachans['S'] = [stachan for stachan in stachans if stachan[-1] != 'Z']
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
        temp_traces[p] = {stachan: np.zeros((len(template_cat),
                                             temp_len[p]))
                          for stachan in ph_stachans[p]}
        det_traces[p] = {stachan: np.zeros((len(detection_cat),
                                            det_len[p]))
                         for stachan in ph_stachans[p]}
    # Populate trace arrays for all picks
    # Pass to _prepare_data function to clean this up
    print('Preparing data for processing')
    temp_traces, det_traces = _prepare_data(
        template_streams, detection_streams, template_cat, detection_cat,
        temp_traces, det_traces, filt_params, phases, corr_dict, cores=cores)
    # Calculate relative polarities
    if cores > 1:
        print('Starting up pool')
        rel_pols = []
        pool = Pool(processes=cores)
        results = [pool.apply_async(
            _stachan_loop,
            (phase, stachan,
             temp_traces[phase][stachan],
             det_traces[phase][stachan]),
             {'min_cc': corr_dict[phase]['min_cc'],
              'debug': debug,
              'plotdir': plotdir})
            for phase in phases
            for stachan in ph_stachans[phase]]
        pool.close()
        print('Retrieving results')
        rel_pols.extend([p.get() for p in results])
        pool.join()
    else:
        # Python loop..?
        rel_pols = []
        for phase in phases:
            for stachan in ph_stachans[phase]:
                print('Looping stachan: {}'.format(stachan))
                pol_array = np.zeros((len(detection_streams),
                                      len(template_streams)))
                for m in range(len(template_streams)):
                    for n in range(len(detection_streams)):
                        pol = _rel_polarity(temp_traces[phase][stachan][m],
                                            det_traces[phase][stachan][n],
                                            corr_dict[phase]['min_cc'], m, n,
                                            stachan, phase, plotdir, debug)
                        pol_array[n][m] = pol
                rel_pols.append((phase, stachan, pol_array))
    if save:
        with open('{}/rel_pols.pkl'.format(save), 'wb') as f:
            pickle.dump(rel_pols, f)
    return rel_pols


def svd_matrix(rel_pols):
    """
    Make the matrix of left singular vectors from all sta/chan/phase combos
    :param rel_pols: Output from
    :return:
    """
    stachans = []
    for i, rel_pol in enumerate(rel_pols):
        try:
            u, s, v = np.linalg.svd(rel_pol[2], full_matrices=True)
        # If rel_pols read back in from marshal binary, array is a buffer
        # Reshape it to size of detection_streams x template_streams
        # For Rotokawa case: 3114 template streams
        except LinAlgError:
            u, s, v = np.linalg.svd(np.frombuffer(rel_pol[2]).reshape(-1, 3114),
                                    full_matrices=True)
        lsv = u[:, 0] # First left sigular vector
        if i == 0:
            stachans.append((rel_pol[0], rel_pol[1]))
            svd_mat = lsv[~np.isnan(lsv)]
        else:
            stachans.append((rel_pol[0], rel_pol[1]))
            svd_mat = np.column_stack((svd_mat, lsv[~np.isnan(lsv)]))
    return svd_mat, stachans


def catalog_resolve(svd_mat, stachans, cat_dets, min_weight=1.e-5):
    """

    :param svd_mat: nxm matrix output from svd_matrix
    :param stachans: List of tuples of (phase, stachan) output from
        svd_matrix
    :param cat_dets: Original catalog of detections to copy and reassign
        relative polarities to
    :type cat_dets: obspy.core.Catalog
    :param min_weight: Threshold for final polarity weights to accept
    :type min_weight: float
    :return:

     ..Note We assume values for any of pick.time_errors.uncertainty
        pick.time_errors.upper_uncertainty or pick.time_errors.confidence_level
    """
    # Isolate columns of svd_mat corresponding to vertical channels
    # Not supporting P-polarities on horizontal channels yet.
    z_cols = np.array([i for i, stachan in enumerate(stachans)
                       if stachan[0] == 'P'])
    z_chans = [stachan[1] for stachan in stachans if stachan[0] == 'P']
    z_mat = svd_mat[:, z_cols]
    # Create dictionary of all weighted catalog polarities
    cat_pol_dict = {stachan: np.zeros((len(cat_dets))) for stachan in z_chans}
    for i, ev in enumerate(cat_dets):
        for pk in ev.picks:
            if pk.polarity and pk.phase_hint == 'P':
                sta = pk.waveform_id.station_code
                chan = pk.waveform_id.channel_code
                te = pk.time_errors
                if not te.uncertainty and not te.upper_uncertainty and not \
                        te.confidence_level:
                    print('Must have a measure of pick uncertainty for '
                          'weighting')
                    continue
                else:
                    # Invert uncertainty measure for confidence
                    if te.uncertainty or te.upper_uncertainty:
                        # Careful here as some of the picks in our catalog
                        # have uncertainties of up to 0.3...
                        try:
                            wt = 0.5 - te.uncertainty
                        except TypeError:
                            wt = 0.5 - te.upper_uncertainty
                    elif te.confidence_level:
                        wt = 5 - te.confidence_level
                if pk.polarity == 'negative':
                    pol = -1. * wt
                elif pk.polarity == 'positive':
                    pol = 1. * wt
                cat_pol_dict['{}.{}'.format(sta, chan)][i] = pol
    # Establish stachan weighting by comparing SVD pols with cat pols
    # Apply to z_mat
    stachan_wt = {}
    print('Establishing stachan weighting')
    for i, stachan in enumerate(z_chans):
        svd_pols = z_mat[:, i]
        cat_pols = cat_pol_dict[stachan]
        print('At {}:\n Numerator: {}\n Denominator:{}'.format(
            stachan, np.sum(svd_pols * cat_pols),
            np.sum(np.abs(svd_pols * cat_pols))))
        stachan_wt[stachan] = (np.sum(svd_pols * cat_pols) /
                               np.sum(np.abs(svd_pols * cat_pols)))
        # Multiply corresponding column of z_mat by this value
        if np.isnan(stachan_wt[stachan]):
            print('Station weight is nan')
            continue
        else:
            z_mat[:, i] *= stachan_wt[stachan]
    # Put the final polarities into a new catalog
    catalog_pols = cat_dets.copy()
    for i, ev in enumerate(catalog_pols):
        for pk in ev.picks:
            sta = pk.waveform_id.station_code
            chan = pk.waveform_id.channel_code
            stach = '{}.{}'.format(sta, chan)
            # Find the column index of z_mat for this stachan
            try:
                stach_i = [i for i, stch in enumerate(z_chans)
                           if stch == stach][0]
            # Unless not a vertical channel
            except IndexError:
                continue
            # Assign polarity by the sign. Put the weight in a Comment at the
            # moment. User will have to decide what to do with this.
            if np.abs(z_mat[i, stach_i]) < min_weight:
                continue  # Skip if below threshold
            if z_mat[i, stach_i] < 0.0:
                pk.polarity = 'negative'
                pk.comments.append(
                    Comment(text='pol_wt: {}'.format(
                        np.abs(z_mat[i, stach_i]))))
            elif z_mat[i, stach_i] > 0.0:
                pk.polarity = 'positive'
                pk.comments.append(
                    Comment(text='pol_wt: {}'.format(
                        np.abs(z_mat[i, stach_i]))))
    return catalog_pols, cat_pol_dict, z_mat, z_chans

def compare_rel_cat_pols(cat_pols, cat_dets, show=True):
    # Make dict of eid: event for detections with polarity picks
    cat_picks_dict = {ev.resource_id.id.split('/')[-1]: ev
                      for ev in cat_dets if len([pk for pk in ev.picks
                                                 if pk.polarity]) > 0}
    # This will contain bools for stachans and rel_pol_weights
    matches = {stachan: [] for stachan in
               list(set([pk.waveform_id.station_code for ev in cat_pols
                         for pk in ev.picks]))}
    for ev in cat_pols:
         if ev.resource_id.id.split('/')[-1] in cat_picks_dict:
             eid = ev.resource_id.id.split('/')[-1]
             for pk in ev.picks:
                 stachan = pk.waveform_id.station_code
                 if (pk.phase_hint == 'P' and pk.polarity
                     and len(pk.comments) != 0):
                    pol = [p for p in cat_picks_dict[eid].picks
                           if p.waveform_id.station_code ==
                           pk.waveform_id.station_code
                           and p.waveform_id.channel_code ==
                           pk.waveform_id.channel_code
                           and p.polarity]
                    if len(pol) == 1 and pk.time_errors.upper_uncertainty:
                        matches[stachan].append(
                            (pol[0].polarity == pk.polarity,
                             pk.comments[-1].text,
                             pk.time_errors.upper_uncertainty))
    # Plot distributions of weights and uncertainties for matches and non-match
    trues = [(float(mat[1].split()[-1]), mat[-1])
             for stach, mats in matches.items()
             for mat in mats if mat[0] == True]
    falses = [(float(mat[1].split()[-1]), mat[-1])
              for stach, mats in matches.items()
              for mat in mats if mat[0] == False]
    t_wts, t_uncerts = zip(*trues)
    f_wts, f_uncerts = zip(*falses)
    print(t_uncerts)
    fig, (ax1, ax2) = plt.subplots(2)
    sns.distplot(t_wts, ax=ax1, color='r',
                 label='Correct matches', kde=False)
    sns.distplot(f_wts, ax=ax1, color='b',
                 label='Incorrect matches', kde=False)
    sns.distplot(t_uncerts, ax=ax2, color='r',
                 label='Correct matches', kde=False)
    sns.distplot(f_uncerts, ax=ax2, color='b',
                 label='Incorrect matches', kde=False)
    ax1.text(0.7, 0.9, 'Correct matches: {}'.format(len(t_wts)),
             fontsize=10, transform=ax1.transAxes)
    ax1.text(0.7, 0.8, 'Incorrect matches: {}'.format(len(f_wts)),
             fontsize=10, transform=ax1.transAxes)
    ax1.set_title('Pick weights')
    ax2.set_title('Pick uncertainty')
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
        plt.close()
    return matches


def partition_Nga(cat, svd_mat=None, threshold='tight'):
    """
    Split the svd_mat and det_cat into NgaN and NgaS clusters
    :return:
    """
    NgaN_cat = Catalog(); NgaS_cat = Catalog()
    if svd_mat:
        NgaN_svd = np.zeros(svd_mat.shape[1])
        NgaS_svd = np.zeros(svd_mat.shape[1])
    for i, ev in enumerate(cat):
        o = ev.preferred_origin() or ev.origins[-1]
        if threshold == 'tight': # Stringent cropping to Nga clusters
            if -38.526 > o.latitude > -38.55 and 176.17 < o.longitude < 176.20:
                # Ngatamariki North
                NgaN_cat.append(ev)
                if svd_mat:
                    NgaN_svd = np.vstack((NgaN_svd, svd_mat[i]))
            elif -38.575 < o.latitude < -38.55 and 176.178 < o.longitude < 176.21:
                # Ngatamariki South
                NgaS_cat.append(ev)
                if svd_mat:
                    NgaS_svd = np.vstack((NgaS_svd, svd_mat[i]))
        elif threshold == 'loose': # More general split on latitude
            if o.latitude > -38.55:
                NgaN_cat.append(ev)
                if svd_mat:
                    NgaN_svd = np.vstack((NgaN_svd, svd_mat[i]))
            elif o.latitude < -38.55:
                NgaS_cat.append(ev)
                if svd_mat:
                    NgaS_svd = np.vstack((NgaS_svd, svd_mat[i]))
    if svd_mat:
        return NgaN_cat, NgaS_cat, NgaN_svd[1:], NgaS_svd[1:]
    else:
        return NgaN_cat, NgaS_cat


def cluster_svd_mat(svd_mat, stachans=None, exclude_sta=[],
                    exclude_phase=[], metric='cosine', criterion='maxclust',
                    thresh=None, show=False):
    """
    Function to cluster the rows of the nxk matrix of relative polarity
    measurements
    :return: List of group indices for each of the n detected events
    """
    # arg checks
    if (len(exclude_sta) > 0 or len(exclude_phase) > 0) and not stachans:
        print('If specifying phases or stations to use in clustering '
              + 'you must provide stachans for mapping to svd_mat')
        return
    if not thresh:
        print('Must provide a threshold for your chosen criterion: {}'.format(
            criterion))
        return
    if len(exclude_sta) > 0 or len(exclude_phase) > 0:
        inds = np.array([i for i, stach in enumerate(stachans)
                         if stach[0] not in exclude_phase
                         and stach[1].split('.')[0] not in exclude_sta])
        clust_mat = svd_mat[:, inds]
    else:
        clust_mat = svd_mat
    print('Matrix for clustering has shape: {}'.format(clust_mat.shape))
    # Compute dist matrix outside linkage for debugging
    Y = pdist(clust_mat, metric=metric)
    Z = linkage(Y[~np.isnan(Y)], method='single') # Mask nans
    if show:
        if criterion == 'distance':
            dendrogram(Z, color_threshold=thresh)
        else:
            dendrogram(Z, color_threshold=0)
        plt.show()
        plt.close('all')
    indices = fcluster(Z, t=thresh, criterion=criterion)
    return indices


def cluster_cat(indices, det_cat, min_events=2):
    """
    Group detection catalog into the clusters determined by the relative
    polarity measurements.
    :param indices:
    :param z_mat:
    :param det_cat:
    :return:
    """
    clust_ids = list(set(indices))
    indices = [(indices[i], i) for i in range(len(indices))]
    indices.sort(key=lambda x: x[0])
    clust_cats = []
    for clust_id in clust_ids:
        cat = Catalog()
        for ind in indices:
            if ind[0] == clust_id:
                ev = det_cat.events[ind[1]]
                # Go through picks and assign polarity from relative pols
                cat.append(ev)
            elif ind[0] > clust_id:
                if len(cat) > min_events:
                    clust_cats.append(cat)
                break
    # Get final group
    if len(cat) > min_events:
        clust_cats.append(cat)
    # Sort by size of catalog
    clust_cats.sort(key=lambda x: len(x), reverse=True)
    return clust_cats

def cluster_to_stereonets(cluster_cats, cons_cat_dir, outdir, pols='all'):
    """
    Helper to loop clusters and save output plots to a directory

    :param cluster_cats: List of cats where each catalog represents a polarity
        cluster
    :param cons_cat_dir: Directory where the consensus catalogs for each
        cluster are kept. Named as: Cat_consensus_i where i is cluster no.
    :param outdir: Output directory for plots
    :return:
    """
    for i, clust in enumerate(cluster_cats):
        cons_cat = read_events('{}/Cat_consensus_{}.xml'.format(cons_cat_dir,
                                                                i))
        plot_picks_on_stereonet(clust, fm_cat=cons_cat,
                                title='Cluster {} consensus'.format(i),
                                outdir=outdir, pols=pols,
                                savefig='Consensus_{}_{}.png'.format(i, pols))
    return

def plot_picks_on_stereonet(catalog, fm_cat=None, pols='all', title=None,
                            station_plot=False, savefig=False, outdir='.',
                            ax=None):
    """
    Plot relative polarities for catalog (presumably a cluster) on stereonet

    :param catalog: Catalog of events for which to plot pols
    :param fm_cat: Catalog of one event which contains fm solutions.
        Will plot the soln with lowest misfit
    :param pols: If 'all', will plot all polarities from cluster. If
        'consensus', will plot only the consensus polarities.
    :param title: Title of plot (optional)
    :param station_plot: If True, will plot the arrival points for each station
        as a different color. If False, will plot arrivals colored by polarity
        and sized by relative polarity weighting.
    :param savefig: If False, will show plot. Else savefig must be the name of
        the desired file (with extension).
    :param outdir: Output directory for files if savefig
    :param ax: Axes instance on top of which to plot
    :return:
    """
    # check args
    if not station_plot:
        pol_plot = True
    if pols == 'consensus' and not fm_cat:
        print('Cant plot consensus polarities without fm_cat arg')
        return
    fig = plt.figure()
    colors = cycle(['red', 'green', 'blue', 'cyan', 'magenta', 'yellow',
                    'black', 'firebrick', 'purple', 'darkgoldenrod', 'gray'])
    if not ax:
        ax = fig.add_subplot(111, projection='stereonet')
    sta_dict = {sta: {'plunge': [],
                      'bearing': [],
                      'pol': [],
                      'wt': []}
                for sta in list(set([pk.waveform_id.station_code
                                     for ev in catalog for pk in ev.picks]))}
    wts = []
    for ev in catalog:
        for pk in ev.picks:
            if pk.comments:
                if 'pol_wt' in pk.comments[-1].text:
                    wts.append(float(pk.comments[-1].text.split()[-1]))
    max_wt = np.max(np.abs(wts))
    for ev in catalog:
        if ev.preferred_origin().method_id:
            for arrival in ev.preferred_origin().arrivals:
                pk = arrival.pick_id.get_referred_object()
                if pk and pk.polarity:
                    sta = pk.waveform_id.station_code
                    toa = arrival.takeoff_angle
                    az = arrival.azimuth
                    try:
                        wt = float(pk.comments[-1].text.split()[-1]) / max_wt
                    except IndexError:
                        if pk.creation_info.version.startswith('ObsPyck'):
                            wt = max_wt
                        else:
                            continue
                    sta_dict[sta]['wt'].append(wt)
                    if toa > 90.:
                        up = True
                        sta_dict[sta]['plunge'].append(toa - 90.)
                    else:
                        up = False
                        sta_dict[sta]['plunge'].append(90. - toa)
                    if up and az < 180.:
                        sta_dict[sta]['bearing'].append(az + 180.)
                    elif up and az > 180.:
                        sta_dict[sta]['bearing'].append(az - 180.)
                    elif not up:
                        sta_dict[sta]['bearing'].append(az)
                    if pk.polarity == 'positive':
                        sta_dict[sta]['pol'].append('$+$')
                    elif pk.polarity == 'negative':
                        sta_dict[sta]['pol'].append('$-$')
    if station_plot:
        for sta, p_dict in sta_dict.items():
            col = next(colors)
            for i in range(len(p_dict['bearing'])):
                ax.line(p_dict['plunge'][i], p_dict['bearing'][i], label=sta,
                        markersize=2, color=col)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                   ncol=2, fontsize=5, loc='upper right', bbox_to_anchor=(1.3, 1.1))
    elif pol_plot:
        if fm_cat:
            fms = fm_cat[0].focal_mechanisms
            fm = min(fms, key=attrgetter('misfit'))
            ax.text(1.09, 0.85, 'Misfit: {:.3f}'.format(fm.misfit),
                    transform=ax.transAxes, fontdict=dict(fontsize=8))
            np1 = fm.nodal_planes.nodal_plane_1
            np2 = fm.nodal_planes.nodal_plane_2
            ax.plane(np1.strike, np1.dip, color='k')
            ax.plane(np2.strike, np2.dip, color='k')
        if pols == 'consensus':
            for arr in fm_cat[0].origins[0].arrivals:
                pk = arr.pick_id.get_referred_object()
                toa = arr.takeoff_angle
                az = arr.azimuth
                if toa > 90.:
                    up = True
                    plunge = toa - 90.
                else:
                    up = False
                    plunge = 90. - toa
                if up and az < 180.:
                    bearing = az + 180.
                elif up and az > 180.:
                    bearing = az - 180.
                elif not up:
                    bearing = az
                if pk.polarity == 'positive':
                    pol = '$+$'
                elif pk.polarity == 'negative':
                    pol = '$-$'
                if pol == '$+$':
                    color = 'blue'
                elif pol == '$-$':
                    color = 'red'
                ax.line(plunge, bearing, c=color, label=pol)
        elif pols == 'all':
            for sta, p_dict in sta_dict.items():
                for i in range(len(p_dict['bearing'])):
                    if p_dict['pol'][i] == '$+$':
                        color = 'blue'
                    elif p_dict['pol'][i] == '$-$':
                        color = 'red'
                    ax.line(p_dict['plunge'][i], p_dict['bearing'][i],
                            c=color, markersize=10 * p_dict['wt'][i],
                            label=p_dict['pol'][i])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        for text, handle in by_label.items():
            handle.set_markersize(5)
        plt.legend(by_label.values(), by_label.keys(), loc='upper right',
                   fontsize=12, bbox_to_anchor=(1.3, 1.1))
    if title:
        plt.suptitle(title)
    if savefig:
        plt.savefig('{}/{}'.format(outdir, savefig), dpi=300)
        plt.close('all')
    else:
        plt.show()
        plt.close('all')
    return


def plot_relative_pols(z_mat, z_chans, cat_pols, cat_pol_dict, show=True):
    """
    Plot weighted relative polarities vs catalog polarities
    :return:
    """
    # mags = [ev.magnitudes[-1].mag for ev in cat_pols]
    for i, stachan in enumerate(z_chans):
        fig, ax = plt.subplots()
        rectpos = Rectangle((0, 0), 0.08, 0.1)
        rectneg = Rectangle((-0.08, -0.1), 0.08, 0.1)
        patches = PatchCollection([rectneg, rectpos], facecolor='lightgray',
                                  alpha=0.5)
        ax.add_collection(patches)
        s = ax.scatter(z_mat[:, i], cat_pol_dict[stachan], s=2)
        ax.set_title(stachan)
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlim([-0.08, 0.08])
        ax.axvline(0)
        ax.axhline(0)
        # plt.colorbar(s)
        if show:
            plt.show()
            plt.close('all')
    return

def cluster_to_consensus(catalog):
    """
    Take the median location from a cluster, determine consensus polarity
    for each stachan, output to single event for HASH
    :return:
    """
    # TODO Add weighting based on fraction of stations with zero values?
    stas = list(set([pk.waveform_id.station_code for ev in catalog
                       for pk in ev.picks if pk.phase_hint == 'P'
                       and pk.comments]))
    print(stas)
    stach_dict = {stach: {'wts': [],
                          'consensus': 0}
                  for stach in stas}
    arr_dict = {stach: {'toas': [],
                        'dists': [],
                        'azs': []}
                for stach in stas}
    for ev in catalog:
        for pk in ev.picks:
            if pk.comments:
                if pk.comments[-1].text.startswith('pol_wt'):
                    sta = pk.waveform_id.station_code
                    if pk.polarity:
                        if pk.polarity == 'positive':
                            wt = float(pk.comments[-1].text.split()[-1])
                        elif pk.polarity == 'negative':
                            wt = -float(pk.comments[-1].text.split()[-1])
                        stach_dict[sta]['wts'].append(wt)
        if ev.preferred_origin():
            for arr in ev.preferred_origin().arrivals:
                sta = arr.pick_id.get_referred_object().waveform_id.station_code
                if sta in stas:
                    arr_dict[sta]['toas'].append(arr.takeoff_angle)
                    arr_dict[sta]['dists'].append(arr.distance)
                    arr_dict[sta]['azs'].append(arr.azimuth)
    for sta, sta_dict in stach_dict.items():
        sta_dict['consensus'] = (np.sum(sta_dict['wts'])
                                 / np.sum(np.abs(sta_dict['wts'])))
    consensus_catalog = Catalog()
    event = Event()
    first_ts = catalog[0].preferred_origin().time.timestamp
    med_lat = np.median([ev.preferred_origin().latitude
                         for ev in catalog])
    lat_dev = np.std([ev.preferred_origin().latitude
                         for ev in catalog])
    med_lonlon = np.median([ev.preferred_origin().longitude
                         for ev in catalog])
    lon_dev = np.std([ev.preferred_origin().longitude
                         for ev in catalog])
    med_dep = np.median([ev.preferred_origin().depth
                         for ev in catalog])
    dep_dev = np.std([ev.preferred_origin().depth
                         for ev in catalog])
    event.origins = [Origin(time=first_ts, latitude=med_lat,
                            longitude=med_lonlon, depth=med_dep,
                            origin_uncertainty=OriginUncertainty(),
                            longitude_errors=QuantityError(
                                uncertainty=lon_dev
                            ),
                            latitude_errors=QuantityError(
                                uncertainty=lat_dev
                            ),
                            depth_errors=QuantityError(
                                uncertainty=dep_dev
                            ))]
    event.preferred_origin_id = event.origins[-1].resource_id.id
    for sta, sta_dict in stach_dict.items():
        pk = Pick()
        ar = Arrival()
        if sta_dict['consensus'] < 0:
            pk.polarity = 'negative'
        elif sta_dict['consensus'] > 0:
            pk.polarity = 'positive'
        pk.waveform_id = WaveformStreamID(station_code=sta,
                                          channel_code='EHZ')
        pk.phase_hint = 'P'
        ar.pick_id = pk.resource_id.id
        ar.phase = 'P'
        ar.azimuth = np.median(arr_dict[sta]['azs'])
        ar.takeoff_angle = np.median(arr_dict[sta]['toas'])
        ar.distance = np.median(arr_dict[sta]['dists'])
        event.picks.append(pk)
        event.origins[-1].arrivals.append(ar)
    consensus_catalog.append(event)
    return consensus_catalog, stach_dict, arr_dict


def run_rel_pols(template_streams, detection_streams, template_cat,
                 detection_cat, corr_dict, min_cc, filt_params,
                 phases=('P', 'S'), cores=4, debug=0, method='joblib',
                 cluster_metric='cosine', cluster_criterion='maxclust',
                 cluster_maxclusts=100, plot=False):
    """
    :return:
    """
    # Check lengths of streams lists against length of catalogs.
    # We're assuming cat[0] corresponds to stream_list[0], etc...
    if len(template_streams) != len(template_cat) or len(detection_streams)\
        != len(detection_cat):
        print('Stream lists must have the same length and order as '
              'corresponding catalog')
        return
    rel_pols = make_corr_matrices(template_streams, detection_streams,
                                  template_cat, detection_cat, corr_dict,
                                  min_cc, filt_params, phases, cores, debug,
                                  method)
    svd_mat, stachans = svd_matrix(rel_pols)
    indices = cluster_svd_mat(svd_mat, metric=cluster_metric,
                              criterion=cluster_criterion,
                              clusts=cluster_maxclusts)
    cat_pol, cat_pol_dict, z_mat, z_chans = catalog_resolve(
        svd_mat, stachans, detection_cat, plot=plot)
    clust_cats = cluster_cat(indices, cat_pol)
    return clust_cats, cat_pol_dict, z_mat, z_chans


class TestRelPols(unittest.TestCase):
    """
    Testing class for above functions
    """
    def generate_data(self):
        # Generate test set of 10 template and 20 detection traces with random
        # pols for P and S
        rand_pols_temps = np.asarray([random.choice((-1, 1))
                                      for i in range(20)]).reshape((2, 10))
        rand_pols_dets = np.asarray([random.choice((-1, 1))
                                     for i in range(40)]).reshape((2, 20))
        temp_traces = [seis_sim(15, amp_ratio=1.2) for i in range(10)]
        det_traces = [seis_sim(15, amp_ratio=1.2) for i in range(20)]
        # Flip the phases around randomly
        for i in range(len(temp_traces)):
            temp_traces[i][:10] *= rand_pols_temps[0, i]
            temp_traces[i][10:] *= rand_pols_temps[1, i]
        for i in range(len(det_traces)):
            det_traces[i][:10] *= rand_pols_dets[0, i]
            det_traces[i][10:] *= rand_pols_dets[1, i]
        return temp_traces, det_traces

    def test_rel_pol(self):
        trace_1 = seis_sim(sp=10, amp_ratio=1.2)
        trace_2 = trace_1 * -1.
        data_1 = trace_1[10:20]
        data_2 = trace_2[5:25]
        rel_pol = _rel_polarity(data_1, data_2)
        self.assertEqual(rel_pol, 0)