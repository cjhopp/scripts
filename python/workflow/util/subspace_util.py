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
try:
    import seaborn as sns
    import pandas as pd
except:
    print('Environment doesnt contain seaborn/pandas')

from glob import glob
from itertools import chain
from obspy import Stream, read, UTCDateTime, Catalog
import datetime
from datetime import timedelta
from eqcorrscan.core.match_filter import Tribe, Template
from eqcorrscan.utils import stacking, clustering
from eqcorrscan.utils.pre_processing import shortproc
from eqcorrscan.utils.stacking import align_traces, linstack, PWS_stack
from eqcorrscan.core.subspace import Detector, align_design
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

def heatmap_plot(dmat_file, big_tribe, raw_wav_dir, tick_int=20,
                 title=None, show=True):
    mat = 1.0 - np.load(dmat_file) # More intuitive to use CCC
    # Make list of dates
    big_tribe.sort()
    raw_wav_files = glob('%s/*' % raw_wav_dir)
    raw_wav_files.sort()
    all_wavs = [wav.split('/')[-1].split('.')[0] for wav in raw_wav_files]
    names = [t.name for t in big_tribe if t.name in all_wavs]
    new_tribe = Tribe()
    new_tribe.templates = [temp for temp in big_tribe if temp.name in names]
    times = [template.event.origins[-1].time.strftime('%Y-%m-%d')
             for template in new_tribe][::tick_int]
    ax = sns.heatmap(mat, vmin=-0.4, vmax=0.6, cmap='vlag',
                     yticklabels=tick_int, xticklabels=False,
                     cbar_kws={'label': 'CCC'})
    ax.set_yticklabels(times, fontsize=6)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if show:
        plt.show()
        plt.close()
    return ax

def cluster_map_plot(dmat_file, big_tribe, tribe_groups_dir, raw_wav_dir,
                     savefig=None):
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
    matrix = np.load(dmat_file) # Take absolute value? NO
    dist_vec = squareform(matrix)
    Z = linkage(dist_vec)
    df_mat = pd.DataFrame(matrix)
    tribes = glob('{}/*.tgz'.format(tribe_groups_dir))
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
                         vmin=0.4, vmax=1.4, #row_colors=template_colors,
                         col_colors=template_colors, row_linkage=Z,
                         col_linkage=Z, yticklabels=False, xticklabels=False,
                         cbar_kws={'label':'1 - CCC'}, figsize=(12, 12))
    if not savefig:
        plt.show()
    else:
        cmg.savefig(savefig, dpi=500)
    return cmg

def stack_plot(tribe, wav_dir_pat, station, channel, title, shift=True,
               shift_len=0.3, savefig=None):
    """
    Plot list of traces for a stachan one just above the other
    :param tribe: Tribe to plot
    :param wav_dir_pat: Glob pattern for all possible wavs
    :param station: Station to plot
    :param channel: channel to plot
    :param title: Plot title
    :param shift: Whether to allow alignment of the wavs
    :param shift_len: Length in seconds to allow wav to shift
    :param savefig: Name of the file to write
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
    # Normalize traces and make dates vect
    date_labels = []
    for tr in traces:
        date_labels.append(str(tr.stats.starttime.date))
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
        ax.set_xlabel('Seconds', fontsize=19)
    else:
        ax.set_xlabel('Samples', fontsize=19)
    ax.set_ylabel('Date', fontsize=19)
    # Change y labels to dates
    ax.yaxis.set_ticks(vert_steps)
    ax.set_yticklabels(date_labels[::-1], fontsize=16)
    ax.set_title(title, fontsize=19)
    if savefig:
        fig.tight_layout()
        plt.savefig(savefig)
        plt.close()
    else:
        fig.tight_layout()
        plt.show()
    return

def stack_party(party, sac_dir, method='linear', filt_params=None, align=True,
                shift_len=0.1, prepick=2., postpick=5., reject=0.7,
                normalize=False, plot=False, outdir=None):
    """
    Return a stream for the linear stack of the templates in a multiplet.

    The approach here is to first stack all of the detections in a family
    over the rejection ccc threshold and THEN stack the Family stacks into
    the final stack for the multiplet. This avoids attempting to correlate
    detections from different Families with each other, which is nonsensical.

    :param party: Party for the multiplet we're interested in
    :param sac_dir: Directory of SAC files made for Stefan
    :param method: Stacking method: 'linear' or 'PWS'
    :param filt_params: (optional) Dictionary of filter parameters to use
        before aligning waveforms. Keys must be 'highcut', 'lowcut',
        'filt_order', and 'samp_rate'
    :param align: Whether or not to align the waveforms
    :param shift_len: Allowed shift in aligning in seconds
    :param reject: Correlation coefficient cutoff in aligning
    :param normalize: Whether to normalize before stacking
    :param plot: Alignment plot flag
    :return:
    """

    sac_dirs = glob('{}/2*'.format(sac_dir))
    fam_stacks = {}
    for fam in party:
        fam_id = fam.template.event.resource_id
        print('For Family {}'.format(fam_id))
        eids = [str(ev.resource_id).split('/')[-1] for ev in fam.catalog]
        raws = []
        for s_dir in sac_dirs:
            if s_dir.split('/')[-1] in eids:
                raws.append(read('{}/*'.format(s_dir)).merge(
                    fill_value='interpolate'))
        # Stupid check for empty det directories. Not yet resolved
        lens = [len(raw) for raw in raws]
        if len(lens) == 0: continue
        if max(lens) == 0: continue
        print('Removing all traces without 3001 samples')
        for st in raws:
            for tr in st.copy():
                if len(tr.data) != 3001:
                    st.remove(tr)
        if filt_params:
            for raw in raws:
                shortproc(raw, lowcut=filt_params['lowcut'],
                          highcut=filt_params['highcut'],
                          filt_order=filt_params['filt_order'],
                          samp_rate=filt_params['samp_rate'])
        print('Now trimming around pick times')
        z_streams = []
        for raw in raws:
            z_stream = Stream()
            for tr in raw.copy():
                if 'a' in tr.stats.sac:
                    strt = tr.stats.starttime
                    z_stream += tr.trim(
                        starttime=strt + tr.stats.sac['a'] - prepick,
                        endtime=strt + tr.stats.sac['a'] + postpick)
            if len(z_stream) > 0:
                z_streams.append(z_stream)
        # At the moment, the picks are based on P-arrival correlation already!
        if align:
            z_streams = align_design(z_streams, shift_len=shift_len,
                                     reject=reject, multiplex=False,
                                     no_missed=False, plot=plot)
        if method == 'linear':
            fam_stacks[fam_id] = linstack(z_streams, normalize=normalize)
        elif method == 'PWS':
            fam_stacks[fam_id] = PWS_stack(z_streams, normalize=normalize)
    if plot:
        # Plot up the stacks of the Families first
        for id, fam_stack in fam_stacks.items():
            fam_stack.plot(equal_scale=False)
    if outdir:
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        for id, fam_stack in fam_stacks.items():
            filename = '{}/Family_{}_stack.mseed'.format(
                outdir, str(id).split('/')[-1])
            fam_stack.write(filename, format='MSEED')
    return fam_stacks

def cluster_cat(catalog, corr_thresh, corr_params=None, raw_wav_dir=None,
                dist_mat=False, out_cat=None, show=False, method='average'):
    """
    Cross correlate all templates in a tribe and return separate tribes for
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
        all_wavs = [wav.split('/')[-1].split('_')[-3]
                    for wav in raw_wav_files]
        print(all_wavs[0])
        names = [ev.resource_id.id.split('/')[-1] for ev in catalog
                 if ev.resource_id.id.split('/')[-1] in all_wavs]
        print(names[0])
        wavs = [wav for wav in raw_wav_files
                if wav.split('/')[-1].split('_')[-3] in names]
        print(wavs[0])
        new_cat = Catalog(events=[ev for ev in catalog
                                  if ev.resource_id.id.split('/')[-1]
                                  in names])
        print(new_cat)
        new_cat.write(out_cat, format="QUAKEML")
        print('Processing temps')
        temp_list = [(shortproc(read('{}/*'.format(tmp)),lowcut=lowcut,
                                highcut=highcut, samp_rate=samp_rate,
                                filt_order=filt_order, parallel=True,
                                num_cores=cores), ev)
                     for tmp, ev in zip(wavs, new_cat)]
        print('Clipping traces')
        rm_temps = []
        for temp in temp_list:
            print('Clipping template %s' % temp[1].resource_id.id)
            rm_ts = [] # Make a list of traces with no pick to remove
            for tr in temp[0]:
                pk = [pk for pk in temp[1].picks
                      if pk.waveform_id.station_code == tr.stats.station
                      and pk.waveform_id.channel_code == tr.stats.channel]
                if len(pk) == 0:
                    rm_ts.append(tr)
                else:
                    tr.trim(starttime=pk[0].time - shift_len - pre_pick,
                            endtime=pk[0].time - pre_pick + length + shift_len)
            # Remove pickless traces
            for rm in rm_ts:
                temp[0].traces.remove(rm)
            # If trace lengths are internally inconsistent, remove template
            if len(list(set([len(tr) for tr in temp[0]]))) > 1:
                rm_temps.append(temp)
        for t in rm_temps:
            temp_list.remove(t)
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
    group_tribes = []
    group_cats = []
    if corr_params:
        for group in groups:
            group_tribes.append(
                Tribe(templates=[Template(
                    st=tmp[0], name=tmp[1].resource_id.id.split('/')[-1],
                    event=tmp[1], highcut=highcut,
                    lowcut=lowcut, samp_rate=samp_rate,
                    filt_order=filt_order,
                    prepick=pre_pick)
                                 for tmp in group]))
            group_cats.append(Catalog(events=[tmp[1] for tmp in group]))
    else:
        for group in groups:
            group_tribes.append(
                Tribe(templates=[Template(
                    st=tmp[0], name=tmp[1].resource_id.id.split('/')[-1],
                    event=tmp[1].event, highcut=None,
                    lowcut=None, samp_rate=None,
                    filt_order=None, prepick=None)
                                 for tmp in group]))
            group_cats.append(Catalog(events=[tmp[1] for tmp in group]))
    return group_tribes, group_cats

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
