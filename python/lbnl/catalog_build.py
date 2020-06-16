#!/usr/bin/python

"""
Set of functions wrapping obspy triggering and phasepapy picking/association
"""

import os
import yaml

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from joblib import Parallel, delayed
from obspy import UTCDateTime, read, Stream, Catalog, read_inventory, read_events
from obspy.core.event import Pick, Event, WaveformStreamID, QuantityError
from obspy.geodetics import degrees2kilometers, locations2degrees
from obspy.signal.trigger import coincidence_trigger, plot_trigger
from eqcorrscan.utils.pre_processing import dayproc, _check_daylong, shortproc
from phasepapy.phasepicker import aicdpicker, ktpicker
from phasepapy.associator import tables3D
from phasepapy.associator.assoc3D import LocalAssociator, PickModified
from phasepapy.associator.tables3D import Associated
from phasepapy.associator.tt_stations_3D import BaseTT3D, TTtable3D, SourceGrids
from phasepapy.associator.tt_stations_3D import Station3D

import obspy.taup as taup

sidney_stas = ['NSMTC', 'B009', 'B010', 'B011', 'PGC']
olympic_bhs = ['B005', 'B006', 'B007']

def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def build_databases(param_file, which='assoc'):
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
    db_name = paramz['Database']['name']
    # If the associator database exists delete it first
    if which == 'assoc':
        if os.path.exists('{}_associator.db'.format(db_name)):
            os.remove('{}_associator.db'.format(db_name))
    elif which == 'tt':
        if os.path.exists('{}_tt.db'.format(db_name)):
            os.remove('{}_tt.db'.format(db_name))
    elif which == 'both':
        if os.path.exists('{}_tt.db'.format(db_name)):
            os.remove('{}_tt.db'.format(db_name))
            os.remove('{}_associator.db'.format(db_name))
    # Our SQLite databases are:
    db_assoc = 'sqlite:///{}_associator.db'.format(db_name)
    db_tt = 'sqlite:///{}_tt.db'.format(db_name)  # Traveltime database
    # Connect to our databases
    engine_assoc = create_engine(db_assoc, echo=False)
    # Create the tables required to run the 1D associator
    tables3D.Base.metadata.create_all(engine_assoc)
    Session = sessionmaker(bind=engine_assoc)
    session = Session()
    return session, db_assoc, db_tt


def build_tt_tables(param_file, inventory, tt_db):
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
    assoc_paramz = paramz['Associator']
    # Create a connection to an sqlalchemy database
    tt_engine = create_engine(tt_db, echo=False)
    BaseTT3D.metadata.create_all(tt_engine)
    TTSession = sessionmaker(bind=tt_engine)
    tt_session = TTSession()
    # We will use IASP91 here but obspy.taup does let you build your own model
    velmod = taup.TauPyModel(model='iasp91')
    # Define our distances we want to use in our lookup table
    grid_shape_lat = assoc_paramz['grid_shape_lat']
    grid_shape_lon = assoc_paramz['grid_shape_lon']
    grid_origin_lat = assoc_paramz['grid_origin_lat']
    grid_origin_lon = assoc_paramz['grid_origin_lon']
    grid_spacing = assoc_paramz['grid_spacing']
    max_depth = assoc_paramz['max_depth']
    depth_spacing = assoc_paramz['depth_spacing']
    lats = np.arange(grid_origin_lat,
                     grid_origin_lat + (grid_shape_lat * grid_spacing),
                     grid_spacing)
    lons = np.arange(grid_origin_lon,
                     grid_origin_lon + (grid_shape_lon * grid_spacing),
                     grid_spacing)
    depth_km = np.arange(0, max_depth + depth_spacing,
                         depth_spacing)
    # Now add all individual stations to tt sesh
    seeds = list(set(['{}.{}.{}'.format(net.code, sta.code, chan.location_code)
                      for net in inventory for sta in net for chan in sta]))
    grid_ids = {}
    for i, seed in enumerate(seeds):
        print('Populating {}'.format(seed))
        net, sta, loc = seed.split('.')
        sta_inv = inventory.select(network=net, station=sta, location=loc)[0][0]
        chan = sta_inv[0]
        station = Station3D(sta, net, loc, sta_inv.latitude, sta_inv.longitude,
                            sta_inv.elevation - chan.depth)
        # Save the station locations in the database
        tt_session.add(station)
        tt_session.commit()
        # Now loop lat, lon depth (ijk); populate the SourceGrids and TTtable3D
        # for each grid point
        for glat in lats:
            for glon in lons:
                for dep in depth_km:
                    if i == 0:
                        print('Adding node {},{},{}'.format(glat, glon, dep))
                        src_grid = SourceGrids(latitude=glat, longitude=glon,
                                               depth=dep)
                        grid_ids[(glat, glon, dep)] = src_grid.id
                    d_deg = locations2degrees(
                        lat1=sta_inv.latitude, long1=sta_inv.longitude,
                        lat2=glat, long2=glon)
                    d_km = degrees2kilometers(d_deg)
                    p_arrivals = velmod.get_travel_times(
                        source_depth_in_km=dep, distance_in_degree=d_deg,
                        phase_list=['P', 'p', 'Pn'])
                    ptime = min([p.time for p in p_arrivals
                                 if p.name in ['P', 'p']])
                    s_arrivals = velmod.get_travel_times(
                        source_depth_in_km=dep, distance_in_degree=d_deg,
                        phase_list=['S', 's', 'Sn'])
                    stime = min([s.time for s in s_arrivals
                                 if s.name in ['S', 's']])
                    try:
                        pn_time = [p.time for p in p_arrivals
                                   if p.name in ['Pn']][0]
                        sn_time = [s.time for s in s_arrivals
                                   if s.name in ['Sn']][0]
                    except IndexError as e:
                        pn_time = 9999.
                        sn_time = 9999.
                    tt_entry = TTtable3D(
                        sta=sta, sgid=grid_ids[(glat, glon, dep)], d_km=d_km,
                        delta=d_deg, p_tt=ptime, s_tt=stime,
                        s_p=stime - ptime, pn_tt=pn_time,
                        sn_tt=sn_time, sn_pn=sn_time - pn_time)
                    tt_session.add(tt_entry)
                    tt_session.add(src_grid)
    tt_session.commit()
    tt_session.close()
    return


def associator(param_file):
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
    assoc_p = paramz['Associator']
    trig_p = paramz['Trigger']
    # Build tt databases
    print('Building tt databases')
    inv = read_inventory(assoc_p['inventory'])
    db_sesh, db_assoc, db_tt = build_databases(param_file)
    build_tt_tables(param_file, inv, db_tt)
    # Define our picker
    pick_p = paramz['Picker']
    if pick_p['method'] == 'aicd':
        picker = aicdpicker.AICDPicker(
            t_ma=pick_p['t_ma'], nsigma=pick_p['nsigma'], t_up=pick_p['t_up'],
            nr_len=pick_p['nr_len'], nr_coeff=pick_p['nr_coeff'],
            pol_len=pick_p['pol_len'], pol_coeff=pick_p['pol_coeff'],
            uncert_coeff=pick_p['uncert_coeff'])
    elif pick_p['method'] == 'kurtosis':
        picker = ktpicker.KTPicker(
            t_ma=pick_p['t_ma'], nsigma=pick_p['nsigma'], t_up=pick_p['t_up'],
            nr_len=pick_p['nr_len'], nr_coeff=pick_p['nr_coeff'],
            pol_len=pick_p['pol_len'], pol_coeff=pick_p['pol_coeff'],
            uncert_coeff=pick_p['uncert_coeff'])
    else:
        print('Only kpick and AICD supported')
        return
    # Define associator
    associator = LocalAssociator(
        db_assoc, db_tt, max_km=assoc_p['max_km'],
        aggregation=assoc_p['aggregation'], aggr_norm=assoc_p['aggr_norm'],
        cutoff_outlier=assoc_p['cutoff_outlier'],
        assoc_ot_uncert=assoc_p['assoc_ot_uncert'],
        nsta_declare=assoc_p['nsta_declare'],
        loc_uncert_thresh=assoc_p['loc_uncert_thresh'])
    # Run over all days individually
    start = UTCDateTime(trig_p['start_time']).datetime
    end = UTCDateTime(trig_p['end_time']).datetime
    for date in date_generator(start.date(), end.date()):
        print('Picking on {}'.format(date))
        utcdto = UTCDateTime(date)
        jday = utcdto.julday
        day_wavs = glob('{}/{}/**/*{:03d}.ms'.format(
            paramz['General']['wav_directory'], date.year, jday),
            recursive=True)
        st = Stream()
        for w in day_wavs:
            seed_parts = os.path.basename(w).split('.')
            seed_id = '.'.join(seed_parts[:-3])
            if seed_id[-1] != 'Z':
                continue
            print('Reading {}'.format(w))
            st += read(w)
        st = st.merge(fill_value='interpolate')
        # Filter and downsample the wavs
        print('Processing')
        st = dayproc(st, lowcut=trig_p['lowcut'], num_cores=trig_p['ncores'],
                     highcut=trig_p['highcut'], filt_order=trig_p['corners'],
                     samp_rate=trig_p['sampling_rate'], starttime=utcdto,
                     ignore_length=True)
        # Loop slices to speed calculations
        for tr in st:
            print('Picking on {}'.format(tr.id))
            results = Parallel(n_jobs=trig_p['ncores'], verbose=10)(
                delayed(picker.picks)(
                    tr.slice(starttime=utcdto + (3600. * hr),
                             endtime=utcdto + (3600. * (hr + 1))))
                for hr in np.arange(24))
            t_create = UTCDateTime().datetime
            # Add each pick to the database
            for res in results:
                scnl, picks, polarity, snr, uncert = res
                for i in range(len(picks)):
                    new_pick = tables3D.Pick(scnl, picks[i].datetime,
                                             polarity[i], snr[i], uncert[i],
                                             t_create)
                    db_sesh.add(new_pick)  # Add pick i to the database
        db_sesh.commit()  # Commit the pick to the database
    print('Associating events')
    associator.id_candidate_events()
    associator.associate_candidates()
    associator.single_phase()
    return db_sesh, db_assoc, db_tt


def extract_fdsn_events(param_file):
    """
    Loop all events detected via fdsn, extract waveforms and pick them (for CN
    events with no picks)

    :param param_file:
    :return:
    """
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
    extract_p = paramz['Extract']
    pick_p = paramz['Picker']
    trig_p = paramz['Trigger']
    # Read in catalog
    cat = read_events(extract_p['catalog'])
    cat.events.sort(key=lambda x: x.origins[-1].time)
    # Read in inventory
    inv = read_inventory(paramz['Associator']['inventory'])
    start = cat[0].origins[-1].time
    end = cat[-1].origins[-1].time
    # Basic vmod for anticipated arrival times
    velmod = taup.TauPyModel(model='iasp91')
    # Set up picker
    picker = aicdpicker.AICDPicker(
        t_ma=pick_p['t_ma'], nsigma=pick_p['nsigma'], t_up=pick_p['t_up'],
        nr_len=pick_p['nr_len'], nr_coeff=pick_p['nr_coeff'],
        pol_len=pick_p['pol_len'], pol_coeff=pick_p['pol_coeff'],
        uncert_coeff=pick_p['uncert_coeff'])
    for date in date_generator(start.date(), end.date()):
        print('Extracting events on {}'.format(date))
        utcdto = UTCDateTime(date)
        day_cat = Catalog(events=[ev for ev in cat
                                  if utcdto < ev.origins[-1].time
                                  < utcdto + 86400.])
        jday = utcdto.julday
        day_wavs = glob('{}/{}/**/*{:03d}.ms'.format(
            paramz['General']['wav_directory'], date.year, jday),
            recursive=True)
        day_st = Stream()
        for w in day_wavs:
            day_st += read(w)
        for ev in day_cat:
            eid = ev.resource_id.id.split('/')[-1]
            print('Extracting {}'.format(eid))
            o = ev.origins[-1]
            wav_slice = day_st.slice(starttime=o.time,
                                     endtime=o.time + extract_p['length'])
            # Write event waveform
            wav_slice.write('{}/Event_{}.ms'.format(
                extract_p['outdir'], eid, format='MSEED'))
            pick_seeds = ['{}.{}.{}.{}'.format(
                pk.waveform_id.network_code,
                pk.waveform_id.station_code,
                pk.waveform_id.location_code,
                pk.waveform_id.channel_code) for pk in ev.picks]
            # Pick traces with not pick already
            for tr in wav_slice:
                if tr.id in pick_seeds:
                    continue
                # Process and pick
                pk_tr = shortproc(
                    tr.copy(), lowcut=trig_p['lowcut'], parallel=False,
                    highcut=trig_p['highcut'], filt_order=trig_p['corners'],
                    samp_rate=trig_p['sampling_rate'])
                # Make picks
                scnl, picks, polarity, snr, uncert = picker.picks(pk_tr)
                if len(picks) == 0:
                    continue
                tr_inv = inv.select(station=tr.stats.station,
                                    location=tr.stats.location)[0][0]
                d_deg = locations2degrees(
                    lat1=tr_inv.latitude, long1=tr_inv.longitude,
                    lat2=o.latitude, long2=o.longitude)
                # Get the P and S arrivals from TauP iasp91
                p_arrivals = velmod.get_travel_times(
                    source_depth_in_km=o.depth / 1000.,
                    distance_in_degree=d_deg,
                    phase_list=['P', 'p'])
                ptime = min([p.time for p in p_arrivals
                             if p.name in ['P', 'p']])
                s_arrivals = velmod.get_travel_times(
                    source_depth_in_km=o.depth / 1000.,
                    distance_in_degree=d_deg,
                    phase_list=['S', 's'])
                stime = min([s.time for s in s_arrivals
                             if s.name in ['S', 's']])
                for i, pk in enumerate(picks):
                    if np.abs(pk.time - ptime) < 0.5:
                        ev.picks.append(Pick(
                            time=pk.time,
                            waveform_id=WaveformStreamID(
                                network_code=tr.stats.network,
                                station_code=tr.stats.station,
                                location_code=tr.stats.location,
                                channel_code=tr.stats.channel),
                            method_id=pick_p['method'],
                            time_error=QuantityError(uncertainty=uncert[i]),
                            phase_hint='P'
                        ))
                    elif np.abs(pk.time - stime) < 0.5:
                        ev.picks.append(Pick(
                            time=pk.time,
                            waveform_id=WaveformStreamID(
                                network_code=tr.stats.network,
                                station_code=tr.stats.station,
                                location_code=tr.stats.location,
                                channel_code=tr.stats.channel),
                            method_id=pick_p['method'],
                            time_error=QuantityError(uncertainty=uncert[i]),
                            phase_hint='S'
                        ))
            if 'plotdir' in pick_p:
                plot_picks(
                    wav_slice, ev, prepick=5, postpick=10,
                    outdir=pick_p['plotdir'],
                    name=os.path.basename(eid).split('_')[-1].split('.')[0])
    return cat


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
    sta_lta_params = trig_p['channel_specific_params']
    try:
        network_sta_lta = trig_p['network_specific_params']
    except KeyError as e:
        print('No network-specific parameters. Trigger only on listed stations')
        network_sta_lta = {}
    trigs = []
    start = UTCDateTime(trig_p['start_time']).datetime
    end = UTCDateTime(trig_p['end_time']).datetime
    for date in date_generator(start.date(), end.date()):
        print('Triggering on {}'.format(date))
        utcdto = UTCDateTime(date)
        jday = utcdto.julday
        day_wavs = glob('{}/{}/**/*{:03d}.ms'.format(
            paramz['General']['wav_directory'], date.year, jday),
            recursive=True)
        st = Stream()
        for w in day_wavs:
            seed_parts = os.path.basename(w).split('.')
            seed_id = '.'.join(seed_parts[:-3])
            if seed_id in sta_lta_params:
                print('Reading in {}'.format(w))
                st += read(w)
            elif (seed_id[:2] in network_sta_lta and
                  seed_id[-1] == 'Z'):  # Triggering on Z comps only
                print('Reading in {}'.format(w))
                st += read(w)
        st = st.merge()
        # Remove all traces with more zeros than data
        rms = []
        for tr in st:
            if not _check_daylong(tr):
                rms.append(tr)
        for rm in rms:
            st.traces.remove(rm)
        # Filter and downsample the wavs
        st = dayproc(st, lowcut=trig_p['lowcut'], num_cores=trig_p['ncores'],
                     highcut=trig_p['highcut'], filt_order=trig_p['corners'],
                     samp_rate=trig_p['sampling_rate'], starttime=utcdto,
                     ignore_length=True)
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
        day_trigs = coincidence_trigger(
            trigger_type=None, stream=trigger_stream,
            thr_on=seed_params['thr_on'],
            thr_off=seed_params['thr_off'],
            thr_coincidence_sum=trig_p['coincidence_sum'],
            max_trigger_length=trig_p['max_trigger_length'],
            details=True, trigger_off_extension=trig_p['trigger_off_extension'])
        # Enforce at least 5 non-sidney and non Olympic borehole stations
        day_trigs = [t for t in day_trigs
                     if len([sta for sta in t['stations']
                             if sta not in sidney_stas]) > 4
                     and len([sta for sta in t['stations']
                             if sta not in olympic_bhs]) > 4]
        if plot:
            plot_triggers(day_trigs, st, trigger_stream,
                          sta_lta_params, network_sta_lta,
                          outdir=trig_p['plot_outdir'])
        if not trig_p['output']['write_wavs']:
            print('Not writing waveforms')
            return trigs
        print('Writing triggered waveforms')
        output_param = trig_p['output']
        for t in day_trigs:
            trig_trs = Stream()
            # Only keep stations that triggered
            for sid in t['trace_ids']:
                trig_trs += st.select(id=sid)
            trig_s = trig_trs.slice(
                starttime=t['time'] - output_param['pre_trigger'],
                endtime=t['time'] + output_param['post_trigger'])
            trig_s.write(
                '{}/Trig_{}.ms'.format(output_param['waveform_outdir'],
                                       t['time']), format='MSEED')
        trigs += day_trigs
    return trigs


def picker(param_file):
    """
    Pick the first arrivals (P) for triggered waveforms
    :param method:
    :return:
    """
    cat = Catalog()
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
    assoc_p = paramz['Associator']
    pick_p = paramz['Picker']
    if pick_p['method'] == 'aicd':
        picker = aicdpicker.AICDPicker(
            t_ma=pick_p['t_ma'], nsigma=pick_p['nsigma'], t_up=pick_p['t_up'],
            nr_len=pick_p['nr_len'], nr_coeff=pick_p['nr_coeff'],
            pol_len=pick_p['pol_len'], pol_coeff=pick_p['pol_coeff'],
            uncert_coeff=pick_p['uncert_coeff'])
    elif pick_p['method'] == 'kurtosis':
        picker = ktpicker.KTPicker(
            t_ma=pick_p['t_ma'], nsigma=pick_p['nsigma'], t_up=pick_p['t_up'],
            nr_len=pick_p['nr_len'], nr_coeff=pick_p['nr_coeff'],
            pol_len=pick_p['pol_len'], pol_coeff=pick_p['pol_coeff'],
            uncert_coeff=pick_p['uncert_coeff'])
    else:
        print('Only kpick and AICD supported')
        return
    trigger_files = glob('{}/*'.format(
        paramz['Trigger']['output']['waveform_outdir']))
    # Force chronological order
    trigger_files.sort()
    for trig_f in trigger_files:
        # Make new db_assoc for each event
        db_sesh, db_assoc, db_tt = build_databases(param_file, which='assoc')
        associator = LocalAssociator(
            db_assoc, db_tt, max_km=assoc_p['max_km'],
            aggregation=assoc_p['aggregation'], aggr_norm=assoc_p['aggr_norm'],
            assoc_ot_uncert=assoc_p['assoc_ot_uncert'],
            nsta_declare=assoc_p['nsta_declare'],
            nt=assoc_p['grid_shape_lat'] + 1,
            np=assoc_p['grid_shape_lon'] + 1,
            nr=(assoc_p['max_depth'] / assoc_p['depth_spacing']) + 1)
        ev = Event()
        print('Picking {}'.format(trig_f))
        st = read(trig_f)
        for tr in st:
            # TODO Should process trace before picking!!
            scnl, picks, polarity, snr, uncert = picker.picks(tr)
            if len(picks) == 0:
                continue
            # Always take pick with largest SNR
            if pick_p['pick_measure'] == 'snr':
                ind = np.argmax(snr)
            elif pick_p['pick_measure'] == 'earliest':
                ind = 0
            # Do association?
            t_create = UTCDateTime().datetime
            for i in range(len(picks)):
                new_pick = tables3D.Pick(scnl, picks[i].datetime,
                                         polarity[i], snr[i], uncert[i],
                                         t_create)
                db_sesh.add(new_pick)  # Add pick i to the database
            db_sesh.commit()  # Commit the pick to the database
        print('Associating events')
        associator.id_candidate_events()
        associator.associate_candidates()
        # Query database for associated events
        try:
            event = db_sesh.query(Associated).all()[0]
        except IndexError:
            print('No event associated for these picks')
            continue
        picks = db_sesh.query(PickModified).filter(PickModified.assoc_id==event.id)
        for pick in picks:
            ev.picks.append(Pick(
                time=pick.time,
                waveform_id=WaveformStreamID(
                    network_code=pick.net,
                    station_code=pick.sta,
                    location_code=pick.loc,
                    channel_code=pick.chan),
                method_id=pick_p['method'],
                time_error=QuantityError(uncertainty=pick.error),
                phase_hint=pick.phase,
                ))
        cat.events.append(ev)
        if 'plotdir' in pick_p:
            plot_picks(
                st, ev, prepick=5, postpick=10, outdir=pick_p['plotdir'],
                name=os.path.basename(trig_f).split('_')[-1].split('.')[0])
    return cat

## Plotting ##

def plot_triggers(triggers, st, cft_stream, params, net_params, outdir):
    """Helper to plot triggers, traces and characteristic funcs"""
    for trig in triggers:
        seeds = trig['trace_ids']
        # Clip around trigger time
        st_slice = st.slice(starttime=trig['time'] - 10,
                            endtime=trig['time'] + 50)
        cft_slice = cft_stream.slice(starttime=trig['time'] - 10,
                                     endtime=trig['time'] + 50)
        fig, ax = plt.subplots(nrows=len(seeds), sharex='col',
                               figsize=(6, len(seeds) / 2.))
        fig.suptitle('Detection: {}'.format(trig['time']))
        fig.subplots_adjust(hspace=0.)
        for i, sid in enumerate(seeds):
            try:
                tps = params[sid]
            # Case where channel uses network-general params
            except KeyError as e:
                tps = net_params[sid.split('.')[0]]
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
                           bbox=bbox_props, ha='center', fontsize=8)
            ax[i].set_yticks([])
        ax[i].set_xlabel('Time [s]', fontsize=12)
        if os.path.isdir(outdir):
            plt.savefig('{}/Trig_{}.png'.format(outdir, trig['time']),
                        dpi=200)
            plt.close('all')
        else:
            plt.show()
    return


def plot_picks(st, ev, prepick, postpick, name, outdir):
    seeds = [tr.id for tr in st]
    # Clip around trigger time
    st_slice = st.slice(starttime=prepick,
                        endtime=postpick)
    time_v = np.arange(st_slice[0].data.shape[0]) * st_slice[0].stats.delta
    fig, ax = plt.subplots(nrows=len(seeds), sharex='col',
                           figsize=(6, len(seeds) / 2.), dpi=200)
    fig.suptitle('Detection: {}'.format(name))
    fig.subplots_adjust(hspace=0.)
    for i, sid in enumerate(seeds):
        tr_raw = st_slice.select(id=sid)[0]
        time_vect = np.arange(time_v.shape[0]) * tr_raw.stats.delta
        ax[i].plot(time_vect, tr_raw.data / np.max(tr_raw.data), color='k')
        try:
            pk = [pk for pk in ev.picks
                  if '{}.{}.{}.{}'.format(pk.waveform_id.network_code,
                                          pk.waveform_id.station_code,
                                          pk.waveform_id.location_code,
                                          pk.waveform_id.channel_code) == sid][0]
            pk_time = pk.time
            if pk.phase_hint == 'P':
                col = 'r'
            elif pk.phase_hint == 'S':
                col = 'b'
            pk_t = ((pk_time - st_slice[0].stats.starttime) *
                    st_slice[0].stats.sampling_rate) * st_slice[0].stats.delta
            ax[i].axvline(pk_t, linestyle='-', color=col)
        except IndexError as e:
            pass
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white",
                          ec="k", lw=1)
        ax[i].annotate(s=sid, xy=(0.0, 0.8), xycoords='axes fraction',
                       bbox=bbox_props, ha='center')
        ax[i].set_yticks([])
    fig.savefig('{}/Picks_{}.png'.format(outdir, name))
    plt.close('all')
    return