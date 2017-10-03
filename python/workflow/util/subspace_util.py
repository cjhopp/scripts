#!/usr/bin/python
"""
Helper functions for subspace detectors
"""
import os
import copy
import fnmatch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from glob import glob
from itertools import chain
from obspy import Stream, read, UTCDateTime
import datetime
from datetime import timedelta
from matplotlib import dates
from eqcorrscan.core.match_filter import Tribe, Template
from eqcorrscan.utils import stacking, clustering
from eqcorrscan.utils.pre_processing import shortproc
from eqcorrscan.utils.stacking import align_traces
from eqcorrscan.core.subspace import Detector
from obspy.signal.trigger import classic_sta_lta
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def date_generator(start_date, end_date):
    # Generator for date looping
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def grab_day_wavs(wav_dirs, dto, stachans):
    # Helper to recursively crawl paths searching for waveforms for a dict of
    # stachans for one day

    st = Stream()
    wav_files = []
    for path, dirs, files in chain.from_iterable(os.walk(path)
                                                 for path in wav_dirs):
        print('Looking in %s' % path)
        for sta, chans in iter(stachans.items()):
            for chan in chans:
                for filename in fnmatch.filter(files,
                                               '*.%s.*.%s*%d.%03d'
                                                       % (
                                               sta, chan, dto.year,
                                               dto.julday)):
                    wav_files.append(os.path.join(path, filename))
    print('Reading into memory')
    for wav in wav_files:
        st += read(wav)
    st.merge(fill_value='interpolate')
    print('Checking for trace length. Removing if too short')
    rm_trs = []
    for tr in st:
        if len(tr.data) < (86400 * tr.stats.sampling_rate * 0.8):
            rm_trs.append(tr)
        if tr.stats.starttime != dto:
            print('Trimming trace %s.%s with starttime %s to %s'
                  % (tr.stats.station, tr.stats.channel,
                     str(tr.stats.starttime), str(dto)))
            tr.trim(starttime=dto, endtime=dto + 86400,
                    nearest_sample=False)
    if len(rm_trs) != 0:
        print('Removing traces shorter than 0.8 * daylong')
        for tr in rm_trs:
            st.remove(tr)
    else:
        print('All traces long enough to proceed to dayproc')
    return st.sort(['starttime'])

def cluster_from_dist_mat(dist_mat, temp_list, corr_thresh,
                          show=False, debug=1):
    """
    In the case that the distance matrix has been saved, forego calculating it

    Functionality extracted from eqcorrscan.utils.clustering.cluster
    Consider adding this functionality and commiting to new branch
    :param party: Party used to make the dist_mat
    :param dist_mat: Distance matrix of pair-wise template wav correlations
    :return: Groups of templates
    """
    dist_vec = squareform(dist_mat)
    if debug >= 1:
        print('Computing linkage')
    Z = linkage(dist_vec)
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

def cluster_map_plot(dmat_file, big_tribe, tribe_groups_dir, raw_wav_dir):
    """
    Wrapper on seaborn.clustermap to allow for coloring of rows/columns
    by multiplet
    :param dmat_file:
    :param big_tribe:
    :param tribe_groups_dir:
    :return:
    """
    # Make list of temp files which were actually used in the clustering
    # There were actually fewer than templates for some reason...?
    # XXX TODO May be worth using SAC directories instead?
    big_tribe.sort()
    raw_wav_files = glob('%s/*' % raw_wav_dir)
    raw_wav_files.sort()
    all_wavs = [wav.split('/')[-1].split('.')[0] for wav in raw_wav_files]
    names = [t.name for t in big_tribe if t.name in all_wavs]
    wavs = [wav for wav in raw_wav_files if wav.split('/')[-1].split('.')[0]
            in names]
    new_tribe = Tribe()
    new_tribe.templates = [temp for temp in big_tribe if temp.name in names]
    print('Processing temps')
    temp_list = [template.name
                 for tmp, template in zip(wavs, new_tribe)]
    matrix = np.abs(np.load(dmat_file)) # Take absolute value?
    dist_vec = squareform(matrix)
    Z = linkage(dist_vec)
    df_mat = pd.DataFrame(matrix)
    tribes = glob(tribe_groups_dir)
    grp_inds = []
    grp_nos = []
    for tribe in tribes:
        grp_nos.append(tribe.split('_')[-2])
        trb = Tribe().read(tribe)
        names = [temp.name for temp in trb]
        inds = []
        for i, nm in enumerate(temp_list):
            if nm in names:
                inds.append(i)
        grp_inds.append(tuple(inds))
    # Create a categorical palette to identify the networks
    multiplet_pal = sns.hls_palette(len(grp_inds))
    multiplet_lut = dict(zip(tuple(grp_inds), multiplet_pal))
    # Convert the palette to vectors that will be drawn on the side of the matrix
    temp_colors = {}
    temp_inds = np.arange(0, len(temp_list), 1)
    for i in temp_inds:
        for key in multiplet_lut.keys():
            if i in key:
                temp_colors[i] = multiplet_lut[key]
                break
    template_colors = pd.Series(temp_inds,
                                index=temp_inds,
                                name='Multiplet').map(temp_colors)
    cmg = sns.clustermap(df_mat, method='single', cmap='vlag_r',
                         vmin=0.1, vmax=1.0, #row_colors=template_colors,
                         col_colors=template_colors, row_linkage=Z,
                         col_linkage=Z, yticklabels=False, xticklabels=False,
                         cbar_kws={'label':'1 - CCC'})
    plt.show()
    return cmg

def stack_plot(tribe, wav_dir_pat, station, channel, title, shift=True,
               shift_len=0.3, savefig=None):
    """
    Plot list of traces for a stachan one just above the other
    :return:
    """
    wavs = glob(wav_dir_pat)
    streams = []
    events = [temp.event for temp in tribe]
    for temp in tribe:
        streams.append(read([
            f for f in wavs if f.split('/')[-1].split('.')[0] ==
            str(temp.event.resource_id).split('/')[-1]][0]))
    # Sort traces by starttime
    streams.sort(key=lambda x: x[0].stats.starttime)
    # Select all traces
    traces = []
    tr_evs = []
    for st, ev in zip(streams, events):
        if len(st.select(station=station, channel=channel)) == 1:
            tr = st.select(station=station,channel=channel)[0]
            tr.trim(starttime=tr.stats.starttime + 1.5,
                    endtime=tr.stats.endtime - 5)
            traces.append(tr)
            tr_evs.append(ev)
    if shift: # align traces on cc
        shift_samp = int(shift_len * traces[0].stats.sampling_rate)
        pks = [pk.time for ev in tr_evs for pk in ev.picks
               if pk.waveform_id.station_code == station and
               pk.waveform_id.channel_code == channel]
        cut_traces = [tr.slice(starttime=p_time - 0.2,
                               endtime=p_time + 0.4)
                      for tr, p_time in zip(traces, pks)]
        shifts, ccs = align_traces(cut_traces, shift_len=shift_samp)
        dt_vects = []
        for shif, tr in zip(shifts, traces):
            arb_dt = UTCDateTime(1970, 1, 1)
            td = datetime.timedelta(microseconds=
                                    int(1 / tr.stats.sampling_rate * 1000000))
            # Make new arbitrary time vectors as they otherwise occur on
            # different dates
            dt_vects.append([(arb_dt + shif).datetime + (i * td)
                             for i in range(len(tr.data))])
    # Normalize traces
    for tr in traces:
        tr.data = tr.data / max(tr.data)
    fig, ax = plt.subplots(figsize=(6, 15))
    vert_steps = np.linspace(0, len(traces), len(traces))
    if shift:
        # Plotting chronologically from top
        for tr, vert_step, dt_v in zip(list(reversed(traces)), vert_steps,
                                       dt_vects):
            ax.plot(dt_v, tr.data + vert_step, color='k')
    else:
        for tr, vert_step in zip(list(reversed(traces)), vert_steps):
            ax.plot(tr.data + vert_step, color='k')
    if shift:
        ax.set_xlabel('Seconds')
    else:
        ax.set_xlabel('Samples')
    ax.set_ylabel('Event count')
    ax.set_title(title)
    if savefig:
        fig.tight_layout()
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
    return

def cluster_tribe(tribe, raw_wav_dir, lowcut, highcut, samp_rate, filt_order,
                  pre_pick, length, shift_len, corr_thresh, cores,
                  dist_mat=False, show=False):
    """
    Cross correlate all templates in a tribe and return separate tribes for
    each cluster
    :param tribe:
    :return:

    .. Note: Functionality here is pilaged from align design as we don't
        want the multiplexed portion of that function.
    """

    tribe.sort()
    raw_wav_files = glob('%s/*' % raw_wav_dir)
    raw_wav_files.sort()
    all_wavs = [wav.split('/')[-1].split('.')[0] for wav in raw_wav_files]
    names = [t.name for t in tribe if t.name in all_wavs]
    wavs = [wav for wav in raw_wav_files if wav.split('/')[-1].split('.')[0]
            in names]
    new_tribe = Tribe()
    new_tribe.templates = [temp for temp in tribe if temp.name in names]
    print('Processing temps')
    temp_list = [(shortproc(read(tmp),lowcut=lowcut, highcut=highcut,
                            samp_rate=samp_rate, filt_order=filt_order,
                            parallel=True, num_cores=cores),
                  template)
                 for tmp, template in zip(wavs, new_tribe)]
    print('Clipping traces')
    for temp in temp_list:
        print('Clipping template %s' % temp[1].name)
        for tr in temp[0]:
            pk = [pk for pk in temp[1].event.picks
                  if pk.waveform_id.station_code == tr.stats.station
                  and pk.waveform_id.channel_code == tr.stats.channel][0]
            tr.trim(starttime=pk.time - shift_len - pre_pick,
                    endtime=pk.time - pre_pick + length + shift_len)
    trace_lengths = [tr.stats.endtime - tr.stats.starttime for st in
                     temp_list for tr in st[0]]
    clip_len = min(trace_lengths) - (2 * shift_len)
    stachans = list(set([(tr.stats.station, tr.stats.channel)
                         for st in temp_list for tr in st[0]]))
    print('Aligning traces')
    for stachan in stachans:
        trace_list = []
        trace_ids = []
        for i, st in enumerate(temp_list):
            tr = st[0].select(station=stachan[0], channel=stachan[1])
            if len(tr) > 0:
                trace_list.append(tr[0])
                trace_ids.append(i)
            if len(tr) > 1:
                warnings.warn('Too many matches for %s %s' % (stachan[0],
                                                              stachan[1]))
        shift_len_samples = int(shift_len * trace_list[0].stats.sampling_rate)
        shifts, cccs = stacking.align_traces(
            trace_list=trace_list, shift_len=shift_len_samples, positive=True)
        for i, shift in enumerate(shifts):
            st = temp_list[trace_ids[i]][0]
            start_t = st.select(
                station=stachan[0], channel=stachan[1])[0].stats.starttime
            start_t += shift_len
            start_t -= shift
            st.select(
                station=stachan[0], channel=stachan[1])[0].trim(
                start_t, start_t + clip_len)
    print('Clustering')
    if isinstance(dist_mat, np.ndarray):
        groups = cluster_from_dist_mat(dist_mat=dist_mat, temp_list=temp_list,
                                       show=show, corr_thresh=corr_thresh)
    else:
        groups = clustering.cluster(temp_list, show=show,
                                    corr_thresh=corr_thresh, allow_shift=False,
                                    save_corrmat=True,
                                    cores=cores)
    group_tribes = []
    for group in groups:
        group_tribes.append(Tribe(templates=[Template(st=tmp[0],
                                                      name=tmp[1].name,
                                                      event=tmp[1].event,
                                                      highcut=highcut,
                                                      lowcut=lowcut,
                                                      samp_rate=samp_rate,
                                                      filt_order=filt_order,
                                                      prepick=pre_pick)
                                             for tmp in group]))
    return group_tribes

def Tribe_2_Detector(tribe_dir, raw_wavs, outdir, lowcut, highcut, filt_order,
                     samp_rate, shift_len, reject, dimension, prepick,
                     length, multiplex=False):
    """
    Take a directory of cluster-defined Tribes and write them to Detectors
    :param tribe_dir:
    :return:
    """

    tribe_files = glob('%s/*.tgz' % tribe_dir)
    tribe_files.sort()
    wav_files = glob('%s/*' % raw_wavs)
    for tfile in tribe_files:
        tribe = Tribe().read(tfile)
        print('Working on Tribe: %s' % tfile)
        templates = []
        for temp in tribe:
            try:
                wav = read([wav for wav in wav_files
                            if wav.split('/')[-1].split('.')[0]
                            == temp.name][0])
            except IndexError:
                print('Event not above SNR 1.5')
                continue
            wav.traces = [tr.trim(starttime=tr.stats.starttime + 2 - prepick,
                                  endtime=tr.stats.starttime + 2 - prepick
                                  + length)
                          for tr in wav if tr.stats.channel[-1] == 'Z']
            templates.append(wav)
        # Now construct the detector
        detector = Detector()
        detector.construct(streams=templates, lowcut=lowcut, highcut=highcut,
                           filt_order=filt_order, sampling_rate=samp_rate,
                           multiplex=multiplex,
                           name=tfile.split('/')[-1].split('.')[0],
                           align=True, shift_len=shift_len,
                           reject=reject, no_missed=False)
        detector.write('%s/%s_detector' % (outdir,
                                           tfile.split('/')[-1].split('.')[0]))
    return

def rewrite_subspace(detector, outfile):
    """
    Rewrite old subspace with U and V matrices switched
    :param detector:
    :return:
    """

    new_u = copy.deepcopy(detector.v)
    new_v = copy.deepcopy(detector.u)
    final_u = [u.T for u in new_u]
    final_v = [v.T for v in new_v]
    final_data = copy.deepcopy(final_u)
    new_det = Detector(name=detector.name, sampling_rate=detector.sampling_rate,
                       multiplex=detector.multiplex, stachans=detector.stachans,
                       lowcut=detector.lowcut, highcut=detector.highcut,
                       filt_order=detector.filt_order, data=final_data,
                       u=final_u,sigma=detector.sigma,v=final_v,
                       dimension=detector.dimension)
    new_det.write(outfile)
    return

def get_nullspace(wav_dirs, detector, start, end, n, sta, lta, limit):
    """
    Function to grab a random sample of data from our dataset, check that
    it doesn't contain amplitude spikes (STA/LTA?), then feed it to subspace
    threshold calculation
    :type wav_dir: str
    :param wav_dir: Where the wavs live
    :type detector: eqcorrscan.core.subspace.Detector
    :param detector: Detector object we're calculating the threshold for
    :type start: obspy.core.event.UTCDateTime
    :param start: Start of range from which to draw random samples
    :type end: obspy.core.event.UTCDateTime
    :param end: End of range for random samples
    :type
    :return: list of obspy.core.stream.Stream
    """
    import numpy as np

    day_range = (end.datetime - start.datetime).days  # Number of days in range
    # Take a random sample of days since start of range
    rands = np.random.choice(day_range, size=n, replace=False)
    dtos = [start + (86400 * rand) for rand in rands]
    nullspace = []
    for dto in dtos:
        wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
        stachans = {stachan[0]: [stachan[1]] for stachan in detector.stachans}
        day_wavs = grab_day_wavs(wav_ds, dto, stachans)
        day_wavs.merge(fill_value='interpolate')
        day_wavs.detrend('simple')
        day_wavs.resample(100.)
        # Loop over the hours of this day and take ones with no events
        day_start = day_wavs[0].stats.starttime
        for hr in range(24):
            slice_start = day_start + (hr * 3600)
            slice_end = day_start + (hr * 3600) + 3600
            wav = day_wavs.slice(starttime=slice_start, endtime=slice_end,
                                 nearest_sample=True)
            # Check STA/LTA
            if _check_stalta(wav, sta, lta, limit):
                nullspace.append(wav)
            else:
                print('STA/LTA fail for %s' % slice_start)
                continue
    return nullspace

def calculate_threshold(wav_dirs, detector, start, end, n, Pf, plot=False):
    st = get_nullspace(wav_dirs=wav_dirs, detector=detector, start=start,
                       end=end, n=n)
    detector.set_threshold(streams=st, Pf=Pf, plot=plot)
    return

def _check_stalta(st, STATime, LTATime, limit):
    """
    Take a stream and make sure it's vert. component (or first comp
    if no vert) does not exceed limit given STATime and LTATime
    Return True if passes, false if fails

    .. Note: Taken from detex.fas
    """

    if limit is None:
        return True
    if len(st) < 1:
        return None
    try:
        stz = st.select(component='Z')[0]
    except IndexError:  # if no Z found on trace
        return None
    if len(stz) < 1:
        stz = st[0]
    sz = stz.copy()
    sr = sz.stats.sampling_rate
    ltaSamps = LTATime * sr
    staSamps = STATime * sr
    try:
        cft = classic_sta_lta(sz.data, staSamps, ltaSamps)
    except:
        return False
    if np.max(cft) <= limit:
        return True
    else:
        sta = sz.stats.station
        t1 = sz.stats.starttime
        t2 = sz.stats.endtime
        msg = ('%s fails sta/lta req of %d between %s and %s' % (sta, limit,
                                                                 t1, t2))
        print(msg)
        return False
