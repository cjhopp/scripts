#!/usr/bin/python

"""
Functions for reading/writing and processing waveform data
"""
import os
import scipy
import itertools

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from obspy import read, Stream, Catalog
from obspy.signal.cross_correlation import xcorr_pick_correction
from surf_seis.vibbox import vibbox_preprocess
from eqcorrscan.utils.pre_processing import shortproc
from eqcorrscan.utils import stacking, clustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


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
                    xcorr_pick_correction(
                        pk1.time, tr1[0], pk2[0].time, tr2[0],
                        t_before=0.00003, t_after=0.00015,
                        cc_maxlag=0.0001, plot=True,
                        filter='bandpass',
                        filter_options={'corners': 5,
                                        'freqmax': 42000.,
                                        'freqmin': 2000.},
                        filename='{}/{}.{}/{}_{}_{}.{}.pdf'.format(
                            plotdir, sta, chan, eid1, eid2,
                            sta, chan))
                except Exception as e:
                    print(e)
                    continue
    return

def plot_raw_spectra(st, ev, inv=None, savefig=None):
    """
    Simple function to plot the displacement spectra of a trace
    :param tr: obspy.core.trace.Trace
    :param ev: obspy.core.event.Event
    :param inv: Inventory if we want to remove response

    :return:
    """
    fig, ax = plt.subplots()
    eid = str(ev.resource_id).split('/')[-1]
    for trace in st:
        tr = trace.copy()
        sta = tr.stats.station
        chan = tr.stats.channel
        if not chan.endswith(('1', 'Z')):
            # Only use Z comps for now
            continue
        pick = [pk for pk in ev.picks
                if pk.waveform_id.station_code == sta
                and pk.waveform_id.channel_code == chan]
        if len(pick) == 0:
            continue
        else:
            pick = pick[0]
        if inv:
            pf_dict = {'MERC': [0.5, 3.5, 40., 49.],
                       'GEONET': [0.2, 1.1, 40., 49.]}
            if sta.endswith('Z'):
                prefilt = pf_dict['GEONET']
            else:
                prefilt = pf_dict['MERC']
            tr.remove_response(inventory=inv, pre_filt=prefilt,
                               water_level=20, output='DISP')
        else:
            print('No instrument response to remove. Raw spectrum only.')
        tr.trim(starttime=pick.time - 0.005, endtime=pick.time + 0.02)
        N = len(tr.data)
        T = 1.0 / tr.stats.sampling_rate
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
        yf = scipy.fft(tr.data)
        ax.loglog(xf[1:N//2], 2.0 / N * np.abs(yf[1:N//2]), label=sta)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Displacement (m/Hz)')
        ax.legend()
        ax.set_title('{}: Displacement Spectra'.format(eid))
    if savefig:
        dir = '{}/{}'.format(savefig, eid)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        fig.savefig('{}/{}_spectra.png'.format(dir, eid))
    else:
        plt.show()
    return ax


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
                dist_mat=False, out_cat=None, show=False, method='average'):
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
        raw_wav_files = glob('%s/*' % raw_wav_dir)
        raw_wav_files.sort()
        all_wavs = [wav.split('/')[-1].split('_')[0]
                    for wav in raw_wav_files]
        names = [ev.resource_id.id.split('/')[-1] for ev in catalog
                 if ev.resource_id.id.split('/')[-1] in all_wavs]
        wavs = [wav for wav in raw_wav_files
                if wav.split('/')[-1].split('_')[0] in names]
        print(wavs[0])
        new_cat = Catalog(events=[ev for ev in catalog
                                  if ev.resource_id.id.split('/')[-1]
                                  in names])
        print(new_cat == catalog)
        print('Processing temps')
        temp_list = [(shortproc(read(tmp),lowcut=lowcut,
                                highcut=highcut, samp_rate=samp_rate,
                                filt_order=filt_order, parallel=True,
                                num_cores=cores),
                      ev.resource_id.id.split('/')[-1])
                     for tmp, ev in zip(wavs, new_cat)]
        print(list(set([len(tr) for tup in temp_list for tr in tup[0]])))
        print('Clipping traces')
        rm_temps = []
        for i, temp in enumerate(temp_list):
            # print('Clipping template %s' % new_cat[i].resource_id.id)
            rm_ts = [] # Make a list of traces with no pick to remove
            rm_ev = []
            for tr in temp[0]:
                pk = [pk for pk in new_cat[i].picks
                      if pk.waveform_id.station_code == tr.stats.station
                      and pk.waveform_id.channel_code == tr.stats.channel]
                if len(pk) == 0:
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
                rm_ev.append(new_cat[i])
        for t in rm_temps:
            temp_list.remove(t)
        # Remove the corresponding events as well so catalog and distmat
        # are the same shape
        for rme in rm_ev:
            new_cat.events.remove(rme)
    print(new_cat)
    # new_cat.write(out_cat, format="QUAKEML")
    print('Clustering')
    if isinstance(dist_mat, np.ndarray):
        print('Assuming the tribe provided is the same shape as dist_mat')
        # Dummy streams
        temp_list = [(Stream(), ev) for ev in catalog]
        groups = cluster_from_dist_mat(dist_mat=dist_mat, temp_list=temp_list,
                                       show=show, corr_thresh=corr_thresh,
                                       method=method)
    else:
        # try:
        groups = clustering.cluster(temp_list, show=show,
                                    corr_thresh=corr_thresh,
                                    shift_len=shift_len * 2,
                                    save_corrmat=True, cores=cores)
        # except AssertionError as e:
        #     print(e) # Probably errant picks with time outside traces?
        #     return temp_list
    return groups