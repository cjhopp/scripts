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
import seaborn as sns

from glob import glob
from datetime import timedelta
from joblib import Parallel, delayed
from obspy import read, Stream, Catalog, UTCDateTime, Trace, ObsPyException
from obspy.core.event import ResourceIdentifier
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal import PPSD
from obspy.signal.rotate import rotate2zne
from obspy.signal.cross_correlation import xcorr_pick_correction
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from surf_seis.vibbox import vibbox_preprocess
from eqcorrscan.core.match_filter import Tribe, Party
from eqcorrscan.core.template_gen import template_gen
from eqcorrscan.utils.pre_processing import shortproc, _check_daylong
from eqcorrscan.utils.stacking import align_traces
from eqcorrscan.utils import clustering
from scipy.stats import special_ortho_group, median_absolute_deviation
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Local imports
from surf_seis.vibbox import vibbox_read, vibbox_preprocess

extra_stas = ['CMon', 'CTrig', 'CEnc', 'PPS']

three_comps = ['OB13', 'OB15', 'OT16', 'OT18', 'PDB3', 'PDB4', 'PDB6', 'PDT1',
               'PSB7', 'PSB9', 'PST10', 'PST12']
# Rough +/- 1 degree borehole orientations
borehole_dict = {'OB': [356., 62.5], 'OT': [359., 83.], 'PDB': [259., 67.],
                 'PDT': [263., 85.4], 'PSB': [260., 67.], 'PST': [265., 87.]}


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def read_raw_wavs(wav_dir, event_type='MEQ'):
    """Read all the waveforms in the given directory to a dict"""
    mseeds = glob('{}/*'.format(wav_dir))
    wav_dict = {}
    for ms in mseeds:
        if event_type == 'MEQ':
            eid = ms.split('/')[-1].rstrip('_raw.mseed')
        elif event_type == 'CASSM':
            eid = ms.split('/')[-1].rstrip('_raw.mseed')[:-6]
            eid = eid.lstrip('cassm_')
        else:
            print('Invalid event type')
            return
        try:
            wav_dict[eid] = read(ms)
        except TypeError as e:
            print(e)
    return wav_dict


def _check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return


def calculate_ppsds(netstalocchans, wav_dir, inventory, outdir):
    """
    Crawl a waveform directory structure and calculate ppsds for each file.

    In this case, the files are daylong. We'll make a PPSD for each and also
    compile a full-dataset PPSD (if memory allows)

    :param netstalocs: List of net.sta.loc for
    :param wav_dir: Path to root waveform directory
    :param inventory: Obspy Inventory object for stations of interest
    :param outdir: Output directory for both numpy arrays of PPSD and plots

    :return:
    """
    for nsl in netstalocchans:
        print('Running station {}'.format(nsl))
        nsl_split = nsl.split('.')
        wav_files = glob('{}/**/{}.{}.{}.{}*.ms'.format(
            wav_dir, nsl_split[0], nsl_split[1], nsl_split[2], nsl_split[3]),
            recursive=True)
        wav_files.sort()
        big_ppsd = PPSD(read(wav_files[0])[0].stats, inventory)
        for f in wav_files:
            print('Adding {}'.format(f))
            root_name = os.path.basename(f).rstrip('.ms')
            st = read(f)
            lil_ppsd = PPSD(st[0].stats, inventory)
            flag = lil_ppsd.add(st)
            if not flag:
                print('Failed to add {}'.format(f))
                continue
            big_ppsd.add(st)
            lil_ppsd.save_npz('{}/ppsds/{}.npz'.format(outdir, root_name))
            lil_ppsd.plot(filename='{}/plots/{}.png'.format(outdir, root_name))
        big_root = '.'.join(root_name.split('.')[:-2])
        big_ppsd.save_npz('{}/ppsds/{}_all.npz'.format(outdir, big_root))
        big_ppsd.plot(filename='{}/plots/{}_all.png'.format(outdir, big_root))
    return


def get_IRIS_waveforms(start_date, end_date, inventory, output_root):
    """
    Iterate over date range, pull IRIS waveforms pertaining to the obspy
    inventory provided, and output into a directory structure

    :param start_date: datetime.datetime start of data
    :param end_date: datetime.datetime end of data
    :param inventory: Obspy.core.Inventory for desired stations
    :param output_root: Root output directory
    :return:
    """
    client = Client('IRIS')
    for date in date_generator(start_date, end_date):
        year = str(date.year)
        _check_dir(os.path.join(output_root, year))
        print('Retrieving: {}'.format(date))
        jday = UTCDateTime(date).julday
        # If no directory
        t2 = UTCDateTime(date) + 86400.
        for net in inventory:
            _check_dir(os.path.join(output_root, year, net.code))
            for sta in net.stations:
                _check_dir(os.path.join(output_root, year, net.code,
                                        sta.code))
                for chan in sta.channels:
                    loc = chan.location_code
                    if loc == '':  # If empty, directory wont get written
                        loc = '00'
                    _check_dir(os.path.join(output_root, year, net.code,
                                            sta.code, loc))
                    _check_dir(os.path.join(output_root, year, net.code,
                                            sta.code, loc, chan.code))
                    fname = '{}.{}.{}.{}.{}.{}.ms'.format(net.code, sta.code,
                                                          loc, chan.code,
                                                          year, jday)
                    out_path = os.path.join(output_root, net.code, sta.code,
                                            loc, chan.code, fname)
                    if os.path.isfile(out_path):
                        print('{} already exists'.format(out_path))
                        continue
                    bulk = [(net.code, sta.code, '*', chan.code,
                             UTCDateTime(date), t2)]
                    try:
                        print('Making request for {}'.format(bulk))
                        st = client.get_waveforms_bulk(bulk)
                        print(st)
                    except FDSNNoDataException as e:
                        print(e)
                        continue
                    try:
                        print('Writing {}'.format(out_path))
                        st.select(location=loc,
                                  channel=chan.code).write(out_path,
                                                           format="MSEED")
                    except ObsPyException as e:
                        print(e)
                        continue
    return


def write_event_mseeds(wav_root, catalog, outdir, pre_pick=60.,
                       post_pick=120.):
    """
    Cut event waveforms from daylong mseed for catalog. Will cut the same
    time window for all available channels. Saved waveforms will be used for
    obspycking.

    :param wav_root: Waveform root directory
    :param catalog: Catalog of events
    :param outdir: Output directory for cut waveforms
    :param pre_origin: Seconds before the origin time to clip
    :param post_origin: Seconds after the origin time to clip

    :return:
    """
    # Ensure catalog sorted (should be by default?)
    try:
        catalog.events.sort(key=lambda x: x.preferred_origin().time)
        # Define catalog start and end dates
        cat_start = catalog[0].preferred_origin().time.date
        cat_end = catalog[-1].preferred_origin().time.date
    except AttributeError as e:  # In case this is a catalog of detections
        catalog.events.sort(key=lambda x: x.picks[0].time)
        # Define catalog start and end dates
        cat_start = catalog[0].picks[0].time.date
        cat_end = catalog[-1].picks[0].time.date
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        print('Processing events on {}'.format(dto))
        # Establish which events are in this day
        tmp_cat = Catalog(
            events=[ev for ev in catalog
                    if dto <= ev.picks[0].time < dto + 86400])
        if len(tmp_cat) == 0:
            print('No events on: %s' % str(dto))
            continue
        # Read all available channels for this julian day
        print('Reading wavs')
        chan_files = glob('{}/**/*{}.ms'.format(wav_root, dto.julday),
                          recursive=True)
        st = Stream()
        for chan_f in chan_files:
            st += read(chan_f)
        print('Merging')
        try:
            st.merge(fill_value='interpolate')
        except Exception as e:
            print(e)
            print('Skipping {}'.format(date))
            continue
        for ev in tmp_cat:
            try:
                fname = ev.resource_id.id.split('&')[-2].split('=')[-1]
            except IndexError:  # Case of non-usgs catalog, try Detection name
                fname = ev.resource_id.id
            print('Slicing ev {}'.format(fname))
            pt = min([pk.time for pk in ev.picks])
            st_slice = st.slice(starttime=pt - pre_pick,
                                endtime=pt + post_pick)
            st_slice.write('{}/{}.ms'.format(outdir, fname), format='MSEED')
    return


def tribe_from_catalog(catalog, wav_dir, param_dict, single_station=False,
                       stations='all'):
    """
    Loop over a catalog and return a tribe of templates for each. Assumes we're
    looking in a directory with day-long miniSEED files with filenames formatted
    as: net.sta.loc.chan.julday.ms

    :param catalog: Catalog of events with picks
    :param wav_dir: Root directory that will globbed recursively (/**/) for the
        format specified above
    :param param_dict: Dictionary containing all necessary parameters for
        template creation e.g. {'highcut': , 'lowcut': , 'corners': ,
                                'sampling_rate': , 'prepick': , 'length': }
    :param single_station: Flag to tell function to make a Template for each
        single station for each event.
    :param stations: 'all' or a list of station names to include in templates.
        Valid for both single_station=T and F

    :return:
    """
    # Ensure catalog sorted (should be by default?)
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    # Define catalog start and end dates
    cat_start = catalog[0].preferred_origin().time.date
    cat_end = catalog[-1].preferred_origin().time.date
    tribe = Tribe()
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        # Establish which events are in this day
        sch_str_start = 'time >= {}'.format(dto)
        sch_str_end = 'time <= {}'.format((dto + 86400))
        tmp_cat = catalog.filter(sch_str_start, sch_str_end)
        jday = dto.julday
        net_sta_loc_chans = [(pk.waveform_id.network_code,
                              pk.waveform_id.station_code,
                              pk.waveform_id.location_code,
                              pk.waveform_id.channel_code)
                             for ev in tmp_cat for pk in ev.picks]
        wav_files = [glob('{}/**/{}.{}.{}.{}.{}.ms'.format(wav_dir, nslc[0],
                                                           nslc[1], nslc[2],
                                                           nslc[3], jday,),
                          recursive=True)[0]
                     for nslc in net_sta_loc_chans]
        daylong = Stream()
        for wav_file in wav_files:
            daylong += read(wav_file)
        for ev in tmp_cat:
            name = ev.resource_id.id.split('&')[-2].split('=')[-1]
            if not single_station:
                # Eliminate unwanted picks
                if type(stations) == list:
                    ev.picks = [pk for pk in ev.picks
                                if pk.waveform_id.station_code in stations]
                trb = Tribe().construct(method='from_meta_file', st=daylong,
                                        meta_file=Catalog(events=[ev]),
                                        **param_dict)
                trb.templates[0].name = name
                tribe += trb
            else:
                # Otherwise, make a stand-alone template for each station
                # with P pick (S will get included as well)
                netstalocs = [(pk.waveform_id.network_code,
                               pk.waveform_id.station_code,
                               pk.waveform_id.location_code)
                              for pk in ev.picks if pk.phase_hint == 'P']
                for nsl in netstalocs:
                    if type(stations) == list and nsl[1] not in stations:
                        continue
                    tmp_ev = ev.copy()
                    t_nm = name + '_{}'.format('.'.join(nsl))
                    tmp_ev.picks = [pk for pk in tmp_ev.picks
                                    if pk.waveform_id.network_code == nsl[0]
                                    and pk.waveform_id.station_code == nsl[1]
                                    and pk.waveform_id.location_code == nsl[2]]
                    tmp_ev.resource_id = ResourceIdentifier(id=t_nm)
                    tmp_ev = Catalog(events=[tmp_ev])
                    trb = Tribe().construct(method='from_meta_file', st=daylong,
                                            meta_file=tmp_ev, **param_dict)
                    trb.templates[0].name = t_nm
                    tribe += trb
    return tribe


def detect_tribe(tribe, wav_dir, start, end, param_dict):
    """
    Run matched filter detection on a tribe of Templates over waveforms in
    a waveform directory (formatted as in above Tribe construction func)

    :param tribe: Tribe of Templates
    :param wav_dir: Root directory that will globbed recursively (/**/) for the
        format specified above
    :param start_date: Start UTCDateTime object
    :param end_date: End UTCDateTime object
    :param param_dict: Dict of parameters to pass to Tribe.detect()
    :return:
    """
    party = Party()
    net_sta_loc_chans = list(set([(pk.waveform_id.network_code,
                                   pk.waveform_id.station_code,
                                   pk.waveform_id.location_code,
                                   pk.waveform_id.channel_code)
                                  for temp in tribe
                                  for pk in temp.event.picks]))
    for date in date_generator(start.datetime, end.datetime):
        dto = UTCDateTime(date)
        jday = dto.julday
        wav_files = []
        for nslc in net_sta_loc_chans:
            wav_files.extend(glob('{}/**/{}.{}.{}.{}.{}.ms'.format(
                wav_dir, nslc[0], nslc[1], nslc[2], nslc[3], jday),
                recursive=True))
        daylong = Stream()
        for wav_file in wav_files:
            st = read(wav_file)
            # Ensure not simply zeros or less than 0.8 * day length
            if (_check_daylong(st[0]) and
                    st[0].stats.endtime - st[0].stats.starttime >= 69120.):
                daylong += st
        party += tribe.detect(stream=daylong.merge(), **param_dict)
    return party


def stack_CASSM_directory(path, length, plotdir=None):
    """
    Wrapper on stack_CASSM_shots to loop over all mseeds in directory
    :param path: Path to directory of 16-shot records from Todd
    :param plotdir: Path to optional plotting directory
    :return:
    """
    # Get all miniseeds
    vibbox_files = glob('{}/**/vbox_*.mseed'.format(path), recursive=True)
    for vbox in vibbox_files:
        dir_name = os.path.dirname(vbox)
        fname = 'CASSM_stack_{}.mseed'.format(
            vbox.split('/')[-1].split('.')[0][6:])
        shot = stack_CASSM_shots(read(vbox), length=length)
        shot.write(os.path.join(dir_name, fname), format='MSEED')
        if plotdir:
            shot.plot(equal_scale=False, show=False,
                      outfile=os.path.join(plotdir,
                                           fname.replace('mseed', 'png')),
                      dpi=200, size=(1600, 2400))
    return


def stack_CASSM_shots(st, length):
    """
    Stack all shots in a 16 shot sequence
    :param st: Stream containing all 16 shots and a trigger trace
    :param length: Length of the stacked signals in seconds

    :return: Stream object with all Traces
    """
    start = st[0].stats.starttime
    # Use derivative of PPS signal to find pulse start
    dt = np.diff(st.select(station='CTrg')[0].data)
    # Use 70 * MAD threshold
    trig_samps = np.where(
        dt > np.mean(dt) + 70 * median_absolute_deviation(dt))[0]
    t_samps = []
    # Select only first sample of pulse
    for i, t in enumerate(trig_samps):
        if i == 0:
            t_samps.append(t)
        elif trig_samps[i - 1] != t - 1:
            t_samps.append(t)
    # Remove CASSM and timing info
    st_sensors = st.select(station='B*')
    # Loop trigger onsets, slice, add to list
    shots = Stream()
    for samp in t_samps:
        samp_time = samp / st[0].stats.sampling_rate
        shots += st_sensors.slice(starttime=start + samp_time,
                                  endtime=start + samp_time + length)
    shots.stack(group_by='id', stack_type='linear')
    return shots


def SNR(signal, noise):
    """
    Simple SNR calculation (in decibels)
    SNR = 20log10(RMS(sig) / RMS(noise))
    """
    sig_pow = np.sqrt(np.mean(signal ** 2))
    noise_pow = np.sqrt(np.mean(noise ** 2))
    return 20 * np.log10(sig_pow / noise_pow)


def rotate_catalog_streams(catalog, wav_dir, inv, ncores=8, **kwargs):
    """
    Return a list of rotated streams and a single

    :param catalog: Catalog of events used to orient seismometers
    :param wav_dir: Directory of waveform files to draw from
    :param inv: Station inventory
    :param ncores: Number of cores to use, default 8.
    :param kwargs: Keyword arguments passed to uniform_rotate_stream
    :return:
    """
    # Mseed reading function for error handling
    def mseed_read(afile):
        try: # Catch effed up MSEED files inside dictcomp
            return read(afile)
        except TypeError:
            # Empty stream otherwise
            return Stream()
    mseeds = glob('{}/*'.format(wav_dir))
    wav_ids = [ms.split('/')[-1].split('_')[1] for ms in mseeds]
    eids = [e.resource_id.id.split('/')[-1] for e in catalog]
    mseeds = {m.split('/')[-1].split('_')[1]: mseed_read(m)
              for m in mseeds if m.split('/')[-1].split('_')[1] in eids
              and m.split('/')[-1].split('_')[1] in wav_ids}
    print('Starting pool')
    results = Parallel(n_jobs=ncores, verbose=10)(
        delayed(uniform_rotate_stream)(mseeds[e.resource_id.id.split('/')[-1]],
                                       e, inv, **kwargs)
        for e in catalog if e.resource_id.id.split('/')[-1] in mseeds)
    rot_streams, sta_dicts = zip(*results)
    sd = {}
    # Combine all the dictionaries into one
    print('Retrieving results')
    for d in sta_dicts:
        for k, v in d.items():
            if k not in sd.keys():
                sd[k] = {}
            for ks, vs in v.items():
                if ks not in sd[k].keys():
                    sd[k][ks] = [vs]
                else:
                    sd[k][ks].append(vs)
    return rot_streams, sd


def uniform_rotate_stream(st, ev, inv, rotation='rand', n=1000,
                          amp_window=0.0002, metric='ratio', debug=0):
    """
    Sample a uniform distribution of rotations of a stream and return
    the rotation and stream of interest

    :param st: Stream to rotate
    :param ev: Event with picks used to define the P arrival window
    :param inv: Obspy Inventory object
    :param rotation: Defaults to a random set of rotations but can be given any
        axis which will entail a regular grid of rotations around that axis.
    :param n: Number of samples to draw
    :param amp_window: Length (sec) within which to measure the energy of the
        trace.
    :param metric: Whether to use the 'ratio' of radial to transverse components
        or simply the energy in the 'radial' component to decide on the best
        rotation matrix for each station.

    .. note: One complication is that the arrival orientation corresponding to
        the greatest energy in the signal can be parallel to either the
        source-reciever path or the reciever-source path (i.e. backazimuth).
        To account for this, we test both possibilities by rotating both
        possible arrival vectors by the matrix that provided the highest
        energy, and selecting the one closest to normal to the borehole axis.

    :return:
    """
    # Catch empty stream
    if len(st) == 0:
        print('Stream empty')
        return Stream(), {}
    # Make array of station names
    stas = list(set([tr.stats.station for tr in st
                     if tr.stats.station in three_comps]))
    if rotation == 'rand':
        # Make array of uniformly distributed rotations to apply to stream
        rots = [Rotation.from_dcm(special_ortho_group.rvs(3))
                for i in range(n)]
    else:
        # Just do every degree...
        rots = [Rotation.from_euler(rotation, d, degrees=True)
                for d in range(360)]
    rot_streams = []
    # Create a stream for all possible rotations (this may be memory expensive)
    amp_dict = {}
    for sta in stas:
        rot_st = Stream()
        amp_dict[sta] = []
        # Handle case where no pick for this station
        # Grab the pick
        try:
            pk = [pk for pk in ev.picks
                  if pk.waveform_id.station_code == sta
                  and pk.phase_hint == 'P'][-1]
        except IndexError:
            # If no pick at this station, break the rotations loop
            continue
        work_st = st.select(station=sta).copy().detrend()
        # Bandpass
        work_st.filter(type='bandpass', freqmin=3000,
                       freqmax=42000, corners=3)
        # Trim to 0.01s window around pick (hardcoded for indexing convenience)
        work_st.trim(starttime=pk.time - 0.005,
                     endtime=pk.time + 0.005)
        # Leave this unrotated for debug plotting
        og_signal = work_st.copy().select(channel='*Y')[0].data[495:550]
        if len(work_st.select(channel='*X')) == 0:
            continue
        for R in rots:
            r_st = work_st.copy()
            datax = r_st.select(channel='*X')[0].data
            datay = r_st.select(channel='*Y')[0].data
            dataz = r_st.select(channel='*Z')[0].data
            # Rotate
            datax, datay, dataz = np.dot(R.as_dcm(), [datax, datay, dataz])
            # Noise
            noisey = datay[:301]
            # Signal
            signaly = datay[495:550]
            # Calc E as sum of squared amplitudes
            energy_int = int(amp_window * work_st[0].stats.sampling_rate)
            # Amplitude window starting at pick
            ampx = np.copy(datax[496:496+energy_int])
            ampy = np.copy(datay[496:496+energy_int])
            ampz = np.copy(dataz[496:496+energy_int])
            Ex = np.sum(ampx ** 2)
            Ey = np.sum(ampy ** 2)
            Ez = np.sum(ampz ** 2)
            h_ratio = Ey / (Ez + Ex)
            snry = SNR(signaly, noisey)
            # Normalize data
            poldat = ampy / np.linalg.norm(ampy)
            # Decide polarity of arrival
            pp = find_peaks(poldat, prominence=0.05, distance=5)
            pn = find_peaks(-poldat, prominence=0.05, distance=5)
            try:
                if pp[0][0] < pn[0][0]:
                    pol = 1 # up
                else:
                    pol = 0 # down
            except IndexError as e: # Case of no suitable peaks found
                continue
            amp_dict[sta].append([Ex, Ey, Ez, snry, h_ratio, pol, ampy,
                                  pp[0][0], pn[0][0], signaly, og_signal])
    sta_dict = {}
    if debug > 0:
        fig, axes = plt.subplots(ncols=2, figsize=(10, 12))
        labs = []
        ticks = []
    for i, (sta, amps) in enumerate(amp_dict.items()):
        # No picks at this station
        if len(amps) == 0:
            continue
        x, y, z, snr, rat, p, ampy, pp, pn, sigy, os = zip(*amps)
        if metric == 'radial':
            best_ind = np.argmax(y)
        elif metric == 'ratio':
            best_ind = np.argmax(rat)
        # Take Y, but could be Z (X is along borehole)
        sta_dict[sta] = {'matrix': rots[best_ind]}
        sta_dict[sta]['polarity'] = p[best_ind]
        sta_dict[sta]['datay'] = ampy[best_ind]
        sta_dict[sta]['ratio'] = rat[best_ind]
        sta_dict[sta]['snr'] = snr[best_ind]
        # Test plotting
        if debug > 0:
            labs.append(sta)
            ticks.append(i)
            plot_dat = ampy[best_ind]
            plot_dat /= np.max(plot_dat)
            rot_sig = sigy[best_ind] / np.max(sigy[best_ind])
            # Plot data
            axes[0].plot(plot_dat + i, color='k')
            # Plot polarity picks
            # Up pick
            axes[0].scatter(pp[best_ind], plot_dat[pp[best_ind]] + i,
                            marker='o', color='r')
            # Down pick
            axes[0].scatter(pn[best_ind], plot_dat[pn[best_ind]] + i,
                            marker='o', color='b')
            try:
                axes[1].plot(rot_sig + i, color='k')
                axes[1].plot((os[best_ind] / np.max(os[best_ind])) + i,
                             color='grey')
            except IndexError as e:
                continue
    if debug > 0:
        axes[0].set_yticklabels(labs)
        axes[0].set_yticks(ticks)
        axes[1].set_yticklabels(labs)
        axes[1].set_yticks(ticks)
    # Rotate a final stream so that all instruments have Y oriented radially
    radial_stream = Stream()
    for sta in stas:
        # Hack to get correct length well names
        if sta.startswith('O'):
            well_int = 2
        else:
            well_int = 3
        try:
            rot = sta_dict[sta]['matrix']
            p = sta_dict[sta]['polarity']
        except KeyError:
            # No P pick at this station
            continue
        work_st = st.select(station=sta).copy()
        datax = work_st.select(channel='*X')[0].data
        statx = work_st.select(channel='*X')[0].stats
        datay = work_st.select(channel='*Y')[0].data
        staty = work_st.select(channel='*Y')[0].stats
        dataz = work_st.select(channel='*Z')[0].data
        statz = work_st.select(channel='*Z')[0].stats
        # As select() passes references to traces, can modify in-place
        datax, datay, dataz = np.dot(rot.as_dcm(), [datax, datay, dataz])
        new_trx = Trace(data=datax, header=statx)
        new_try = Trace(data=datay, header=staty)
        new_trz = Trace(data=dataz, header=statz)
        rot_st = Stream(traces=[new_trx, new_try, new_trz])
        radial_stream += rot_st
        # Sort out original orientation. If polarity of max energy arrival
        # is positive, it's the radial vector. Otherwise it's the backazimuth.
        arrival_vect = az_toa_vect(inv.select(station=sta)[0][0],
                                   ev.origins[-1])
        if not p:
            continue
        if p == 0: # Negative polarity
            arrival_vect *= -1. # Other direction
        # Multiply with inverse rotation matrix
        orig_vect = np.dot(rot.inv().as_dcm(), arrival_vect.T)
        # orig_vect_baz = np.dot(rot.inv().as_dcm(), baz_vect.T)
        # Do some trig
        # This comes out as deg from (1, 0) vector (i.e. positive E)
        az = np.rad2deg(np.arctan2(orig_vect[1],
                                   orig_vect[0])) + 90.
        # az_baz = np.rad2deg(np.arctan2(orig_vect_baz[1],
        #                                orig_vect_baz[0])) + 90.
        # Will be degrees from horizontal (negative value is upgoing ray)
        dip = np.rad2deg(np.arccos(orig_vect[2])) - 90.
        # dip_baz = np.rad2deg(np.arccos(orig_vect_baz[2])) - 90.
        # Make borehole vector
        bhz = np.sin(np.deg2rad(borehole_dict[sta[:well_int]][1]))
        bhh = np.sqrt(1 - bhz ** 2)
        bhx = np.sin(np.deg2rad(borehole_dict[sta[:well_int]][0])) * bhh
        bhy = np.cos(np.deg2rad(borehole_dict[sta[:well_int]][0])) * bhh
        bh_vect = np.array([bhx, bhy, bhz])
        # Sort out which is most normal to borehole
        angle = np.rad2deg(np.arccos(np.dot(orig_vect, bh_vect)))
        # angle_baz = np.rad2deg(np.arccos(np.dot(orig_vect_baz, bh_vect)))
        sta_dict[sta]['orientation'] = (az, dip)
        sta_dict[sta]['bh angle'] = angle
        # sta_dict[sta]['orientation from backazimuth'] = (az_baz, dip_baz)
        # sta_dict[sta]['bh angle from backazimuth'] = angle_baz
    return radial_stream, sta_dict


def az_toa_vect(station, origin):
    """
    Returns a unit radial vector from the source to station
    """
    n_diff = float(station.extra.hmc_north.value) - float(origin.extra.hmc_north.value)
    e_diff = float(station.extra.hmc_east.value) - float(origin.extra.hmc_east.value)
    elev_diff = float(station.extra.hmc_elev.value) - float(origin.extra.hmc_elev.value)
    unit_vect = np.array([e_diff, n_diff, elev_diff])
    unit_vect /= np.linalg.norm(unit_vect)
    return unit_vect


def coords2bazinc(station, origin):
    """
    Yoinked from Obspyck, per usual...CJH 3-12-20

    Returns backazimuth and incidence angle from station coordinates
    and event location specified in hmc cartesian coordinate system
    """
    n_diff = (float(station.extra.hmc_north.value) -
              float(origin.extra.hmc_north.value))
    e_diff = (float(station.extra.hmc_east.value) -
              float(origin.extra.hmc_east.value))
    dist = np.sqrt(n_diff**2 + e_diff**2)
    baz = np.rad2deg(np.arctan2(e_diff, n_diff))
    if baz < 0:
        baz += 360.
    elev_diff = (float(station.extra.hmc_elev.value) -
                 float(origin.extra.hmc_elev.value))
    # Angle up from vertical down
    inci = np.rad2deg(np.arctan2(dist, elev_diff))
    return baz, inci


def rotate_stream_to_LQT(st, inventory, origin):
    """
    Rotate stream into LQT orientation w respect to given origin.

    :param invenory:
    :param origin:
    :return:
    """
    rot_st = Stream()
    zne_st = Stream()
    for sta in inventory[0]:
        if sta.code not in three_comps:
            rot_st += st.select(station=sta.code).copy()
            continue
        if len(st.select(station=sta.code)) == 0:
            continue
        baz, incidence = coords2bazinc(sta, origin)
        # First to ZNE
        sta_st_zne = st.copy().select(station=sta.code).rotate(
            method='->ZNE', components='ZXY', inventory=inventory)
        zne_st += sta_st_zne
        # Then to LQT
        sta_st_lqt = sta_st_zne.copy().rotate(
            method='ZNE->LQT', back_azimuth=baz, inclination=incidence)
        rot_st += sta_st_lqt
    return rot_st, zne_st


def which_server_vibbox(cat, file_list, outfile):
    """
    Output a list of files we need from the server

    :param cat: Catalog of events we need
    :param file_list: File containing a list of all vibbox files on the server
    :param outfile: Path to the output file of only the files we need to scp

    :return:
    """

    vibbox_files = []
    with open(file_list, 'r') as f:
        for ln in f:
            if ln.endswith('.dat\n'):
                vibbox_files.append(ln.rstrip('\n'))
    vibbox_files.sort()
    # Make list of ranges for vibbox files
    ranges = [(int(vibbox_files[i - 1].split('_')[-1][:14]),
               int(ln.split('_')[-1][:14]))
              for i, ln in enumerate(vibbox_files)]
    out_list = []
    for ev in cat:
        time_int = int(ev.origins[-1].time.strftime('%Y%m%d%H%M%S'))
        for i, r in enumerate(ranges):
            if r[0] < time_int < r[1] and vibbox_files[i - 1] not in out_list:
                out_list.append(vibbox_files[i - 1])
                break
    with open(outfile, 'w') as of:
        for out in out_list:
            of.write('{}\n'.format(out))
    return


def extract_CASSM_wavs(catalog, vibbox_dir, outdir, pre_ot=0.01, post_ot=0.02):
    """
    Extract waveforms from vibbox files for cassm sources and write to event
    files for later processing.

    :param cat: Catalog of cassm events
    :param vibbox_dir: Directory with waveform files
    :param pre_ot: Time before origin time to start wavs
    :param post_ot: Time after origin time to end wavs

    :return:
    """
    for ev in catalog:
        eid = ev.resource_id.id.split('/')[-1]
        ot = ev.origins[-1].time
        pk1 = min([pk.time for pk in ev.picks])
        print(ot)
        print(pk1)
        dir = '{}/{}'.format(vibbox_dir, ot.strftime('%Y%m%d'))
        day_files = glob('{}/*'.format(dir))
        day_files.sort()
        if len(day_files) == 1:
            print('Only one file in dir')
            print(ev.resource_id.id)
            print(day_files[0])
            st = vibbox_read(day_files[0])
        else:
            ranges = [(int(day_files[i - 1].split('_')[-1][:14]),
                       int(ln.split('_')[-1][:14]))
                      for i, ln in enumerate(day_files)]
            print(ev.resource_id.id)
            time_int = int(ev.origins[-1].time.strftime('%Y%m%d%H%M%S'))
            for i, r in enumerate(ranges):
                if r[0] < time_int < r[1]:
                    print(day_files[i - 1])
                    st = vibbox_read(day_files[i - 1])
                    if st[0].stats.starttime < ot < st[0].stats.endtime:
                        print('Origin time within stream\n')
                    else:
                        print('Origin time outside stream\n')
                    break
        if len(st) == 0:
            print('No file for this event...\n')
            continue
        # # De-median the traces (in place)
        # st = vibbox_preprocess(st)
        # Trim that baby
        try:
            st.trim(starttime=pk1 - pre_ot, endtime=pk1 + post_ot)
        except UnboundLocalError:
            continue
        st.write('{}/cassm_{}_raw.mseed'.format(outdir, eid), format='MSEED')
    return print('boom')


def extract_event_signal(wav_dir, catalog, prepick=0.0001, duration=0.01,
                         event_type='MEQ'):
    """
    Trim around pick times and filter waveforms

    :param wav_dir: Path to waveform files
    :param catalog: obspy.core.Catalog
    :param prepick: How many seconds do we want before the pick?
    :param duration: How long do we want the traces?
    :param event_type: Nameing convention is different for CASSM/MEQ.
        Which is being used here?

    :return:
    """
    streams = {}
    wav_dict = read_raw_wavs(wav_dir, event_type=event_type)
    for ev in catalog:
        ot = ev.origins[-1].time
        if event_type == 'MEQ':
            t_stamp = ot.strftime('%Y%m%d%H%M%S%f')
        elif event_type == 'CASSM':
            t_stamp = ot.strftime('%Y%m%d%H%M%S')
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
            st_sta = st.select(station=sta).copy()
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

########################## PLOTTING ######################################

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
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 8))
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
        # Hard coded for CASSM sources atm
        tr.trim(starttime=pick.time - 0.00005, endtime=pick.time + 0.005)
        axes[0].plot(tr.data)
        N = len(tr.data)
        print(N)
        T = 1.0 / tr.stats.sampling_rate
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
        yf = scipy.fft(tr.data)
        axes[1].loglog(xf[1:N//2], 2.0 / N * np.abs(yf[1:N//2]), label=sta)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Displacement (m/Hz)')
        axes[1].legend()
        axes[1].set_title('{}: Displacement Spectra'.format(eid))
    if savefig:
        dir = '{}/{}'.format(savefig, eid)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        fig.savefig('{}/{}_spectra.png'.format(dir, eid))
    else:
        plt.show()
    return axes


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


def plot_station_rot_stats(sta_dict, title='Station orientaton stats'):
    """
    Plot the statistics of the station dictionary output by
    rotate_catalog_streams()

    :param sta_dict: Nested dictionary with keys 'az-dip', 'borehole angle'
    :return:
    """
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))#,
                             #subplot_kw=dict(polar=True))
    for sta, d in sta_dict.items():
        az, dip = zip(*d['orientation'])
        sns.distplot(d['bh angle'], label=sta, ax=axes[2],
                     hist=False, rug=True)
        sns.distplot(az, label=sta, ax=axes[0], hist=False, rug=True)
        sns.distplot(dip, label=sta, ax=axes[1], hist=False, rug=True)
        axes[0].legend()
        axes[0].set_title('Channel azimuth')
        # axes[0].set_theta_zero_location('N')
        # axes[0].set_theta_direction(-1)
        axes[1].legend()
        axes[1].set_title('Channel dip (from horizontal)')
        # axes[1].set_theta_zero_location('E')
        # axes[1].set_theta_direction(-1)
        axes[2].legend()
        axes[2].set_title('Channel angle with borehole axis')
        # axes[1].set_theta_zero_location('N')
        # axes[1].set_theta_direction(-1)
        plt.suptitle(title, fontsize=20)
    return axes