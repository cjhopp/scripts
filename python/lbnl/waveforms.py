#!/usr/bin/python

"""
Functions for reading/writing and processing waveform data
"""
import os
import copy
import scipy
import itertools

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from datetime import timedelta
from obspy import read, Stream, Catalog, UTCDateTime, Trace
from obspy.signal.rotate import rotate2zne
from obspy.signal.cross_correlation import xcorr_pick_correction
from surf_seis.vibbox import vibbox_preprocess
from eqcorrscan.utils.pre_processing import shortproc
from eqcorrscan.utils.stacking import align_traces
from eqcorrscan.utils import clustering
from scipy.stats import special_ortho_group
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

extra_stas = ['CMon', 'CTrig', 'CEnc', 'PPS']

three_comps = ['OB13', 'OB15', 'OT16', 'OT18', 'PDB3', 'PDB4', 'PDB6', 'PDT1',
               'PSB7', 'PSB9', 'PST10', 'PST12']

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

def uniform_rotate_stream(st, ev, measure='Pamp', n=1000, amp_window=0.0003,
                          plot=False, plot_station='OT16'):
    """
    Sample a uniform distribution of rotations of a stream and return
    the rotation and stream of interest

    :param st: Stream to rotate
    :param ev: Event with picks used to define the P arrival window
    :param measure: What measure determines which rotation is returned
        Defaults to 'Pamp' which finds the rotation which maximizes the
        P-arrival amplitude on the Z component (somewhat arbitrary, but
        is convention to pick on Z for surface stations)
    :param n: Number of samples to draw
    :param amp_window: Length (sec) within which to measure the energy of the
        trace.
    :param plot: Save images of the rotated stream to file. Can be ordered and
        made into a movie later...If yes, provide path as plot argument.
    :param plot_station: To avoid clutter, just give one station to generate
        plots for.

    :return:
    """
    # Make array of station names
    stas = list(set([tr.stats.station for tr in st
                     if tr.stats.station in three_comps]))
    # Make array of uniformly distributed rotations to apply to stream
    rand_Rots = [Rotation.from_dcm(special_ortho_group.rvs(3))
                 for i in range(n)]
    rot_streams = []
    # Create a stream for all possible rotations (this may be memory expensive)
    amp_dict = {}
    for sta in stas:
        rot_st = Stream()
        amp_dict[sta] = []
        for R in rand_Rots:
            # Grab the pick
            pk = [pk for pk in ev.picks if pk.waveform_id.station_code == sta
                  and pk.phase_hint == 'P'][0]
            # Take only the Z comps for triaxials
            work_st = st.select(station=sta).copy()
            # Bandpass
            work_st.filter(type='bandpass', freqmin=3000,
                           freqmax=42000, corners=3)
            # Trim to small window
            work_st.trim(starttime=pk.time - 0.0001,
                         endtime=pk.time + amp_window)
            datax = work_st.select(channel='*X')[0].data
            statx = work_st.select(channel='*X')[0].stats
            datay = work_st.select(channel='*Y')[0].data
            staty = work_st.select(channel='*Y')[0].stats
            dataz = work_st.select(channel='*Z')[0].data
            statz = work_st.select(channel='*Z')[0].stats
            # As select() passes references to traces, can modify in-place
            datax, datay, dataz = np.dot(R.as_dcm(), [datax, datay, dataz])
            rot_st += work_st
            # Calc E as sum of squared amplitudes
            Ex = np.sum([d ** 2 for d in datax])
            Ey = np.sum([d ** 2 for d in datay])
            Ez = np.sum([d ** 2 for d in dataz])
            amp_dict[sta].append([Ex, Ey, Ez])
            if plot and sta == plot_station:
                eulers = R.as_euler(seq='xyz')
                new_trx = Trace(data=datax, header=statx)
                new_try = Trace(data=datay, header=staty)
                new_trz = Trace(data=dataz, header=statz)
                rot_st = Stream(traces=[new_trx, new_try, new_trz])
                outfile = '{}/{}_{:0.2f}_{:0.2f}_{:0.2f}.png'.format(
                    plot, sta, np.rad2deg(eulers[0]), np.rad2deg(eulers[1]),
                    np.rad2deg(eulers[2]))
                rot_st.plot(outfile=outfile)
    sta_dict = {}
    for i, (sta, amps) in enumerate(amp_dict.items()):
        x, y, z = zip(*amps)
        # Take Y, but could be Z (X is along borehole)
        sta_dict[sta] = [rand_Rots[np.argmax(y)]]
    # TODO Back out original orientations of instruments and rotate each station
    # TODO so that X is radial.
    radial_stream = Stream()
    return radial_stream, sta_dict

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


def family_stack_plot(event_list, wavs, station, channel, selfs,
                      title='Detections', shift=True, shift_len=0.3,
                      pre_pick_plot=1., post_pick_plot=5., pre_pick_corr=0.05,
                      post_pick_corr=0.5, cc_thresh=0.5, spacing_param=2,
                      normalize=True, plot_mags=False, figsize=(6, 15),
                      savefig=None):
    """
    Plot list of traces for a stachan one just above the other (modified from
    subspace_util.stack_plot()

    :param events: List of events from which we'll extract picks for aligning
    :param wavs: List of files for above events
    :param station: Station to plot
    :param channel: channel to plot
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
    events = copy.deepcopy(event_list)
    for i, wav in enumerate(wavs):
        try:
            streams.append(read(wav))
        except Exception: # If this directory doesn't exist, remove event
            print('{} doesnt exist'.format(wav))
            rm_evs.append(events[i])
    for rm in rm_evs:
        events.remove(rm)
    print('Have {} streams and {} events'.format(len(streams), len(events)))
    # Select all traces
    traces = []
    tr_evs = []
    colors = []  # Plotting colors
    for i, (st, ev) in enumerate(zip(streams, events)):
        if len(st.select(station=station, channel=channel)) == 1:
            st1 = shortproc(st=st, lowcut=3000., highcut=42000.,
                            filt_order=3, samp_rate=100000.)
            tr = st1.select(station=station, channel=channel)[0]
            try:
                pk = [pk for pk in ev.picks
                      if pk.waveform_id.station_code == station
                      and pk.waveform_id.channel_code == channel][0]
            except:
                print('No pick for this event')
                continue
            traces.append(tr)
            tr_evs.append(ev)
            if ev.resource_id.id.split('/')[-1] in selfs:
                colors.append('red')
                master_trace = tr
            else:
                amps = [np.max(np.abs(tr.data) for tr in traces)]
                master_trace = traces[amps.index(max(amps))]
                colors.append('k')
        else:
            print('No trace in stream for {}.{}'.format(station, channel))
    # Normalize traces, demean and make dates vect
    date_labels = []
    print('{} traces found'.format(len(traces)))
    # for tr in traces:
    #     date_labels.append(str(tr.stats.starttime.date))
    #     tr.data -= np.mean(tr.data)
    #     if normalize:
    #         tr.data /= max(tr.data)
    # Vertical space array
    vert_steps = np.linspace(0, len(traces) * spacing_param, len(traces))
    fig, ax = plt.subplots(figsize=figsize)
    shift_samp = int(shift_len * traces[0].stats.sampling_rate)
    pks = []
    for ev in tr_evs:
        pks.append([pk.time for pk in ev.picks
                    if pk.waveform_id.station_code == station and
                    pk.waveform_id.channel_code == channel][0])
    mags = [ev.magnitudes[0].mag for ev in tr_evs]
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
        ax.plot(dt_v, tr.data + vert_step, color=col)
        if shift:
            ax.vlines(x=pk_x, ymin=vert_step - spacing_param / 2.,
                      ymax=vert_step + spacing_param / 2., linestyle='--',
                      color='red')
        # Magnitude text
        mag_text = 'M$_L$={:0.2f}'.format(mag)
        if shift:
            mag_x = (arb_dt + post_pick_plot + max(shifts)).datetime
        else:
            mag_x = (arb_dt + post_pick_plot).datetime
        if plot_mags:
            ax.text(mag_x, vert_step + spacing_param / 2., mag_text, fontsize=14,
                    verticalalignment='center', horizontalalignment='left',
                    bbox=dict(ec='k', fc='w'))
    # Plot the stack of all the waveforms (maybe with mean pick and then AIC
    # pick following Rowe et al. 2004 JVGR)
    data_stack = np.sum(np.array([tr.data for tr in traces]), axis=0)
    # Demean and normalize
    data_stack -= np.mean(data_stack)
    data_stack /= np.max(data_stack)
    # Plot using last datetime vector from loop above for convenience
    ax.plot(dt_vects[-1], data_stack * 2 - vert_steps[-1],
            color='b')
    # Have to suss out average pick time tho
    av_p_time = (arb_dt).datetime + (np.mean(pk_samples) * td)
    ax.vlines(x=av_p_time, ymin=-vert_steps[-1] - (spacing_param * 2),
              ymax=-vert_steps[-1] + (spacing_param * 2),
              color='green')
    ax.set_xlabel('Seconds', fontsize=19)
    ax.set_ylabel('Date', fontsize=19)
    # Change y labels to dates
    ax.yaxis.set_ticks(vert_steps)
    date_labels[1::3] = ['' for d in date_labels[1::3]]
    date_labels[2::3] = ['' for d in date_labels[2::3]]
    ax.set_yticklabels(date_labels[::-1], fontsize=16)
    ax.set_title(title, fontsize=19)
    if savefig:
        fig.tight_layout()
        plt.savefig(savefig, dpi=300)
        plt.close()
    else:
        fig.tight_layout()
        plt.show()
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