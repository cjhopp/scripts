#!/usr/bin/python

"""
Set of functions wrapping obspy triggering and phasepapy picking/association
"""

import os
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import colorlover as cl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go

from glob import glob
from itertools import cycle
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from shapely.geometry import Polygon, MultiLineString, LineString
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.dates import date2num

from joblib import Parallel, delayed
from obspy import UTCDateTime, read, Stream, Catalog, read_inventory, read_events
from obspy.core.event import Pick, Event, WaveformStreamID, QuantityError
from obspy.core.event import Origin, ResourceIdentifier
from obspy.geodetics import degrees2kilometers, locations2degrees
from obspy.signal.trigger import coincidence_trigger, plot_trigger
from eqcorrscan.utils.pre_processing import dayproc, _check_daylong, shortproc
from eqcorrscan.utils.mag_calc import dist_calc
try:
    from phasepapy.phasepicker import aicdpicker, ktpicker
    from phasepapy.associator import tables3D
    from phasepapy.associator.assoc3D import LocalAssociator, PickModified
    from phasepapy.associator.tables3D import Associated
    from phasepapy.associator.tt_stations_3D import BaseTT3D, TTtable3D
    from phasepapy.associator.tt_stations_3D import SourceGrids
    from phasepapy.associator.tt_stations_3D import Station3D
except ImportError:
    print('No PhasePApy on this machine/env')

import obspy.taup as taup

sidney_stas = ['NSMTC', 'B009', 'B010', 'B011', 'PGC']
olympic_bhs = ['B005', 'B006', 'B007']


def read_tremor(path, lats=(46.5, 50.), lons=(-126.5, -121.5)):
    """Read in a tremor catalog from PNSN"""
    trems = pd.read_csv(path, parse_dates=[3])
    trems.index = pd.to_datetime(trems['time'])
    del trems['time']
    trems.sort_index()
    trems = trems.loc[((50 > trems['lat']) & (trems['lat'] > 46.5)
                       & (trems['lon'] > -126.5) & (trems['lon'] < -121.5))]
    return trems.index.values, trems['lat'].values, trems['lon'].values, trems['depth'].values


def read_slab_model(slab_mod_path):
    # Helper to read in slab model for cascadia and return (x, 3) ndarray
    slab_grd = []
    with open(slab_mod_path, 'r') as f:
        next(f)
        for ln in f:
            line = ln.strip()
            line = line.split(',')
            slab_grd.append((float(line[0]), float(line[1]), float(line[2])))
    return np.array(slab_grd)


def distance_to_slab(event, slab_array):
    o = (event.preferred_origin() or event.origins[-1])
    ev_tup = (o.latitude, o.longitude, o.depth / 1000.)
    return np.min([dist_calc(ev_tup, (s[1], s[0], -s[2])) for s in slab_array])


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
    print('Reading catalog to memory')
    cat = read_events(extract_p['catalog'])
    print(cat)
    cat.events.sort(key=lambda x: x.origins[-1].time)
    # Read in inventory
    inv = read_inventory(paramz['Associator']['inventory'])
    if not extract_p['start_date']:
        start = cat[0].origins[-1].time.datetime
        end = cat[-1].origins[-1].time.datetime
    else:
        start = UTCDateTime(extract_p['start_date']).datetime
        end = UTCDateTime(extract_p['end_date']).datetime
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
        if not extract_p['overwrite']:
            donezos = []
            for ev in day_cat:
                eid = ev.resource_id.id.split('/')[-1]
                if len(eid.split('=')) > 1:
                    # For FDSN pulled events from USGS
                    eid = ev.resource_id.id.split('=')[-2].split('&')[0]
                if os.path.exists(
                    '{}/Event_{}.ms'.format(extract_p['outdir'], eid)):
                    donezos.append(ev)
            for rm in donezos:
                day_cat.events.remove(rm)
        if len(day_cat) == 0:
            continue
        jday = utcdto.julday
        day_wavs = glob('{}/{}/**/*{:03d}.ms'.format(
            paramz['General']['wav_directory'], date.year, jday),
            recursive=True)
        day_st = Stream()
        print('Reading wavs')
        for w in day_wavs:
            # Only pick on Z and N or 2 for speed (arbitrarily)
            if w.split('.')[-4][-1] in ['E', '1']:
                continue
            day_st += read(w)
        try:
            day_st.merge()
        except Exception as e:
            print(e)
            continue
        rms = []
        for tr in day_st:
            if not _check_daylong(tr):
                rms.append(tr)
            else:
                # Address silly sampling rate issues by rounding delta?
                tr.stats.delta = round(tr.stats.delta, 5)
        for rm in rms:
            day_st.traces.remove(rm)
        # Filter and downsample the wavs
        print('Processing daylong wavs')
        try:
            day_st = dayproc(
                day_st, lowcut=trig_p['lowcut'], num_cores=trig_p['ncores'],
                highcut=trig_p['highcut'], filt_order=trig_p['corners'],
                samp_rate=trig_p['sampling_rate'], starttime=utcdto,
                ignore_length=True, ignore_bad_data=True)
        except (ValueError, AttributeError) as e:
            print(e)
            continue
        for ev in day_cat:
            eid = ev.resource_id.id.split('/')[-1]
            if len(eid.split('=')) > 1:
                # For FDSN pulled events from USGS
                eid = ev.resource_id.id.split('=')[-2].split('&')[0]
            print('Extracting {}'.format(eid))
            o = ev.origins[-1]
            wav_slice = day_st.slice(starttime=o.time,
                                     endtime=o.time + extract_p['length'])
            # Write event waveform
            outwav = '{}/Event_{}.ms'.format(extract_p['outdir'], eid)
            wav_slice.write(outwav, format='MSEED')
            pick_seeds = ['{}.{}.{}.{}'.format(
                pk.waveform_id.network_code,
                pk.waveform_id.station_code,
                pk.waveform_id.location_code,
                pk.waveform_id.channel_code) for pk in ev.picks]
            # Pick traces with not pick already
            for tr in wav_slice:
                if tr.id in pick_seeds:
                    print('Pick already made')
                    continue
                print('Picking {}'.format(tr.id))
                # Make picks
                scnl, picks, polarity, snr, uncert = picker.picks(tr.copy())
                if len(picks) == 0:
                    continue
                tr_inv = inv.select(station=tr.stats.station,
                                    location=tr.stats.location)[0][0]
                d_deg = locations2degrees(
                    lat1=tr_inv.latitude, long1=tr_inv.longitude,
                    lat2=o.latitude, long2=o.longitude)
                print('Predicting arrival times')
                # Get the P and S arrivals from TauP iasp91
                if o.depth < 0.:
                    dep = 0.
                else:
                    dep = o.depth
                if tr.stats.channel.endswith('Z'):
                    p_arrivals = velmod.get_travel_times(
                        source_depth_in_km=dep / 1000.,
                        distance_in_degree=d_deg,
                        phase_list=['P', 'p'])
                    ptime = min([p.time for p in p_arrivals
                                 if p.name in ['P', 'p']])
                    phase = 'P'
                else:
                    s_arrivals = velmod.get_travel_times(
                        source_depth_in_km=dep / 1000.,
                        distance_in_degree=d_deg,
                        phase_list=['S', 's'])
                    ptime = min([s.time for s in s_arrivals
                                 if s.name in ['S', 's']])
                    phase = 'S'
                for i, pk in enumerate(picks):
                    pt = pk.datetime
                    pred_pt = (o.time + ptime).datetime
                    # P misfit
                    p_dt = np.abs((pt - pred_pt).total_seconds())
                    if p_dt < 2.:
                        ev.picks.append(Pick(
                            time=pk.datetime,
                            waveform_id=WaveformStreamID(
                                network_code=tr.stats.network,
                                station_code=tr.stats.station,
                                location_code=tr.stats.location,
                                channel_code=tr.stats.channel),
                            method_id=pick_p['method'],
                            time_error=QuantityError(uncertainty=uncert[i]),
                            phase_hint=phase
                        ))
            outev = '{}/Event_{}.xml'.format(extract_p['outdir'], eid)
            ev.write(outev, format='QUAKEML')
            if 'plotdir' in pick_p:
                try:
                    plot_picks(
                        wav_slice.copy(), ev, prepick=o.time,
                        postpick=o.time + 40,
                        outdir=extract_p['plotdir'],
                        name=os.path.basename(eid).split('_')[-1].split('.')[0])
                except ValueError as e:
                    print('Mismatch in x and y for some reason')
                    print(e)
                    continue
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

### HypoDD helpers to deal with NSMTC ###

casc_dd_map = {'G1': 'a', 'B1': 'b', 'B2': 'c', 'B3': 'd', 'G2': 'e'}


def modify_catalog(catalog):
    """Modify a catalog so that picks reflect new names for NSMTC"""
    for ev in catalog:
        for pk in ev.picks:
            wid = pk.waveform_id
            if wid.location_code in casc_dd_map:
                pk.waveform_id.station_code = 'NSMT{}'.format(
                    casc_dd_map[wid.location_code])
    return catalog


def write_station(inventory):
    """Modified from EQcorrscan.utils.catalog_to_dd"""
    station_strings = []
    unique_staloc = list(set(['{}.{}'.format(sta.code, chan.location_code)
                              for net in inventory for sta in net
                              for chan in sta]))
    used_staloc = []
    for network in inventory:
        for station in network:
            for channel in station:
                if channel.location_code in casc_dd_map:
                    write_sta = '{}{}'.format(
                        station.code[:-1], casc_dd_map[channel.location_code])
                else:
                    write_sta = station.code
                staloc = '{}.{}'.format(station.code,
                                        channel.location_code)
                if staloc not in used_staloc and staloc in unique_staloc:
                    station_strings.append(
                        "{:<7s} {:6.3f} {:6.3f} {:5.0f}".format(
                            write_sta, station.latitude,
                            station.longitude,
                            station.elevation - channel.depth))
                    used_staloc.append(staloc)
    with open("station.dat", "w") as f:
        f.write("\n".join(station_strings))


def make_stream_dict(catalog, wav_dir):
    stream_dict = {}
    for ev in catalog:
        eid = ev.resource_id.id.split('/')[-1]
        # TODO This needs to allow the Canada events!!
        if len(eid.split('=')) > 1:
            # For FDSN pulled events from USGS
            eid = ev.resource_id.id.split('=')[-2].split('&')[0]
        st = read('{}/Event_{}.ms'.format(wav_dir, eid))
        # Edit trace header for nsmtc
        for tr in st:
            if tr.stats.location in casc_dd_map:
                tr.stats.station = 'NSMT{}'.format(
                    casc_dd_map[tr.stats.location])
        stream_dict[ev.resource_id.id] = st
    return stream_dict


def read_dd_to_cat(ev_id_map, cat, dd_outfile):
    """
    Read dd output back into obspy Catalog as new origin

    :param ev_id_map:
    :param cat:
    :param dd_outfile:
    :return:
    """
    cat_new = cat.copy()
    dd_loc_dict = {}
    with open(dd_outfile, 'r') as f:
        for ln in f:
            fields = ln.split()
            o = Origin()
            o.latitude = float(fields[1])
            o.longitude = float(fields[2])
            o.depth = float(fields[3]) * 1000.
            t = UTCDateTime(year=int(fields[10]), month=int(fields[11]),
                            day=int(fields[12]), hour=int(fields[13]),
                            minute=int(fields[14]),
                            second=int(fields[15].split('.')[0]),
                            microsecond=int(fields[15].split('.')[1]) * 1000)
            o.time = t
            o.method_id = ResourceIdentifier(
                id="smi:de.erdbeben-in-bayern/location_method/hypoDD/2.1b")
            o.earth_model_id = ResourceIdentifier(
                id="smi:de.erdbeben-in-bayern/earth_model/Cascadia_slow_len")
            dd_loc_dict[int(fields[0])] = o
    # Now loop the catalog and add the events
    for ev in cat_new:
        eid = ev.resource_id.id
        try:
            dd_o = dd_loc_dict[ev_id_map[eid]]
            ev.origins.append(dd_o)
            ev.preferred_origin_id = dd_o.resource_id.id
        except KeyError as e:
            print(e)
            continue
    return cat_new

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
            if pk.phase_hint.lower().startswith('p'):
                col = 'r'
            elif pk.phase_hint.lower().startswith('s'):
                col = 'b'
            else:
                col = 'green'
            pk_t = ((pk_time - st_slice[0].stats.starttime) *
                    st_slice[0].stats.sampling_rate) * st_slice[0].stats.delta
            ax[i].axvline(pk_t, linestyle='-', color=col)
        except (IndexError, UnboundLocalError) as e:
            pass
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white",
                          ec="k", lw=1)
        ax[i].annotate(s=sid, xy=(0.0, 0.8), xycoords='axes fraction',
                       bbox=bbox_props, ha='center')
        ax[i].set_yticks([])
    fig.savefig('{}/Picks_{}.png'.format(outdir, name))
    plt.close('all')
    return


def add_catalog(catalog):
    # UTM Zone 10N
    crs = ccrs.UTM(10)
    # Establish color scales from colorlover (import colorlover as cl)
    colors = cycle(cl.scales['11']['qual']['Paired'])
    lats = np.array([ev.preferred_origin().latitude for ev in catalog])
    lons = np.array([ev.preferred_origin().longitude for ev in catalog])
    transforms = crs.transform_points(ccrs.Geodetic(), lons, lats)
    easts = transforms[:, 0]
    norths = transforms[:, 1]
    depths = [-ev.preferred_origin().depth for ev in catalog]
    mags = [ev.magnitudes[-1].mag for ev in catalog]
    times = [ev.preferred_origin().time.datetime.timestamp() for ev in catalog]
    eids = [ev.resource_id.id for ev in catalog]
    tickvals = np.linspace(min(times), max(times), 10)
    ticktext = [datetime.fromtimestamp(t).strftime('%Y-%m-%d')
                for t in tickvals]
    scat_obj = go.Scatter3d(x=easts, y=norths, z=np.array(depths),
                            mode='markers',
                            name='Seismic event',
                            hoverinfo='text',
                            text=eids,
                            marker=dict(color=times,
                                        cmin=min(tickvals),
                                        cmax=max(tickvals),
                                        size=(1.5 * np.array(mags)) ** 2,
                                        symbol='circle',
                                        line=dict(color=times,
                                                  width=1,
                                                  colorscale='Cividis'),
                                        colorbar=dict(
                                            title=dict(text='Timestamp',
                                                       font=dict(size=18)),
                                            x=-0.2,
                                            ticktext=ticktext,
                                            tickvals=tickvals),
                                        colorscale='Cividis',
                                        opacity=0.5))
    return scat_obj


def add_coastlines():
    """Helper to add coastlines at zero depth"""
    polys = []
    # UTM Zone 10N
    crs = ccrs.UTM(10)
    # Clipping polygon
    box = Polygon([[-126.5, 46.5], [-126.5, 50.],
                   [-121.5, 50.], [-121.5, 46.5]])
    coasts = cfeature.NaturalEarthFeature('physical', 'coastline', '10m')
    for geo in coasts.geometries():
        clipped = geo.intersection(box)
        if type(clipped) == MultiLineString:
            for line in clipped.geoms:
                coords = np.array(line.coords)
                if len(coords) == 0:
                    continue
                pts = crs.transform_points(ccrs.Geodetic(), coords[:, 0],
                                           coords[:, 1])
                easts = pts[:, 0]
                norths = pts[:, 1]
                z = np.array([0. for x in easts])
                coast = go.Scatter3d(
                    x=easts, y=norths, z=z,
                    marker=dict(color='black', line=dict(color='black')),
                    name='Coastlines',
                    mode='lines',
                    opacity=1.,
                    showlegend=False)
                polys.append(coast)
        elif type(clipped) == LineString:
            coords = np.array(clipped.coords)
            if len(coords) == 0:
                continue
            pts = crs.transform_points(ccrs.Geodetic(), coords[:, 0],
                                       coords[:, 1])
            easts = pts[:, 0]
            norths = pts[:, 1]
            z = np.array([0. for x in easts])
            coast = go.Scatter3d(
                x=easts, y=norths, z=z,
                marker=dict(color='black', line=dict(color='black')),
                name='Coastlines',
                mode='lines',
                opacity=1.,
                showlegend=False)
            polys.append(coast)
    return polys


def plot_cascadia_3D(slab_file, catalog, outfile):
    """
    Plot Cascadia locations in 3D with slab model and coastlines

    :param slab_mod: Path to slab model file
    :param catalog: Catalog of seismicity

    :return:
    """
    # UTM Zone 10N
    crs = ccrs.UTM(10)
    # Plot rough slab interface
    slab_grd = read_slab_model(slab_file)
    pts_trans = crs.transform_points(ccrs.Geodetic(), slab_grd[:, 0],
                                     slab_grd[:, 1])
    slab = go.Mesh3d(x=pts_trans[:, 0], y=pts_trans[:, 1],
                     z=slab_grd[:, 2] * 1000,
                     name='Slab model', color='gray', opacity=0.3,
                     delaunayaxis='z', showlegend=True, hoverinfo='skip')
    cat = add_catalog(catalog)
    # Map limits are catalog extents
    lims_x = (np.min(cat['x']), np.max(cat['x']))
    lims_y = (np.min(cat['y']), np.max(cat['y']))
    # Add cartopy coastlines
    coasts = add_coastlines()
    data = [cat, slab]
    data.extend(coasts)
    # Start figure
    fig = go.Figure(data=data)
    xax = go.layout.scene.XAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Easting (m)',
                                range=lims_x, showline=True, mirror=True,
                                linecolor='black', linewidth=2.)
    yax = go.layout.scene.YAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Northing (m)',
                                range=lims_y, showline=True, mirror=True,
                                linecolor='black', linewidth=2.)
    zax = go.layout.scene.ZAxis(nticks=10, gridcolor='rgb(200, 200, 200)',
                                gridwidth=2, zerolinecolor='rgb(200, 200, 200)',
                                zerolinewidth=2, title='Elevation (m)')
    layout = go.Layout(scene=dict(xaxis=xax, yaxis=yax, zaxis=zax,
                                  xaxis_showspikes=False,
                                  yaxis_showspikes=False,
                                  aspectmode='manual',
                                  aspectratio=dict(x=1, y=1, z=1.),
                                  bgcolor="rgb(244, 244, 248)"),
                       autosize=True,
                       # title=title,
                       legend=dict(title=dict(text='Legend',
                                              font=dict(size=18)),
                                   traceorder='normal',
                                   itemsizing='constant',
                                   font=dict(
                                       family="sans-serif",
                                       size=14,
                                       color="black"),
                                   bgcolor='whitesmoke',
                                   bordercolor='gray',
                                   borderwidth=1,
                                   tracegroupgap=3))
    fig.update_layout(layout)
    plotly.offline.iplot(fig, filename='{}.html'.format(outfile))
    return


def make_shift_lines(catalog):
    """
    Helper to make three line collections for eq location shift viz

    :param catalog:
    :return:
    """
    crs = ccrs.UTM(10)
    lats_dd = np.array([ev.origins[-1].latitude for ev in catalog])
    lats_nll = np.array([ev.origins[-2].latitude for ev in catalog])
    lons_dd = np.array([ev.origins[-1].longitude for ev in catalog])
    lons_nll = np.array([ev.origins[-2].longitude for ev in catalog])
    pts_dd = crs.transform_points(ccrs.Geodetic(), lons_dd, lats_dd)
    pts_nll = crs.transform_points(ccrs.Geodetic(), lons_nll, lats_nll)
    lines_map = [[pt[:2], pts_dd[i][:2]] for i, pt in enumerate(pts_nll)]
    lines_lat_xc = [[(catalog[i].origins[-1].depth / 1000, pt[1]),
                     (catalog[i].origins[-2].depth / 1000, pts_dd[i][1])]
                    for i, pt in enumerate(pts_nll)]
    lines_lon_xc = [[(pt[0], catalog[i].origins[-1].depth / 1000),
                     (pts_dd[i][0], catalog[i].origins[-2].depth / 1000)]
                    for i, pt in enumerate(pts_nll)]
    lc_map = LineCollection(lines_map, color='purple', linewidths=0.5,
                            alpha=0.3)
    lc_lat = LineCollection(lines_lat_xc, color='purple', linewidths=0.5,
                            alpha=0.7)
    lc_lon = LineCollection(lines_lon_xc, color='purple', linewidths=0.5,
                            alpha=0.7)
    return lc_map, lc_lat, lc_lon


def plot_locations(catalog, slab_file, title='EQ Locations', filename=None,
                   preferred_origins=True, show_shift=False, plot_tremor=True):
    """
    Cartopy-based plotting function for map view and simple cross sections

    :param catalog:
    :param slab_file:
    :return:
    """
    cols = {'2.1b': 'b', '7': 'k'}
    fig = plt.figure(figsize=(8., 8.5))
    gs = GridSpec(ncols=11, nrows=11, figure=fig)
    crs = ccrs.UTM(10)
    if preferred_origins:
        cat_pref = Catalog(events=[ev for ev in catalog
                                   if ev.preferred_origin()])
        lats = np.array([e.preferred_origin().latitude for e in cat_pref])
        lons = np.array([e.preferred_origin().longitude for e in cat_pref])
        mags = np.array([4 * e.magnitudes[0].mag**2 for e in cat_pref])
        depths = np.array([e.preferred_origin().depth / 1000 for e in cat_pref])
        try:
            colors = [cols[ev.preferred_origin().method_id.id.split('/')[-1]]
                      for ev in catalog]
        except:
            colors = 'k'
        if show_shift:
            lc_map, lc_lat, lc_lon = make_shift_lines(cat_pref)
    else:
        lats = np.array([e.origins[-1].latitude for e in catalog])
        lons = np.array([e.origins[-1].longitude for e in catalog])
        mags = np.array([4 * e.magnitudes[0].mag**2 for e in catalog])
        depths = np.array([e.origins[-1].depth / 1000 for e in catalog])
        try:
            colors = [cols[ev.origins[-1].method_id.id.split('/')[-1]]
                      for ev in catalog]
        except:
            colors = 'k'
    axes_map = fig.add_subplot(gs[:7, :7], projection=crs)
    axes_cs_lat = fig.add_subplot(gs[:7, 7:10], sharey=axes_map)
    axes_cs_lon = fig.add_subplot(gs[7:10, :7], sharex=axes_map)
    axes_map.coastlines(resolution='50m', color='black', linewidth=0.5)
    axes_map.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'ocean', '50m', edgecolor='face',
        facecolor=cfeature.COLORS['water']), alpha=0.7, zorder=0)
    if plot_tremor:
        t_times, t_lat, t_lon, t_dep = read_tremor(plot_tremor)
        t_pts = crs.transform_points(ccrs.Geodetic(), t_lon, t_lat)
        axes_map.scatter(t_pts[:, 0], t_pts[:, 1], marker='s', s=0.25,
                         alpha=0.3, facecolors='none', edgecolors='r',
                         linewidths=0.75)
        axes_cs_lat.scatter(t_dep, t_pts[:, 1], marker='s', s=0.25,
                            alpha=0.3, facecolors='none', edgecolors='r',
                            linewidths=0.75)
        axes_cs_lon.scatter(t_pts[:, 0], t_dep, marker='s', s=0.25,
                            alpha=0.3, facecolors='none', edgecolors='r',
                            linewidths=0.75)
    if show_shift and preferred_origins:
        axes_map.add_collection(lc_map)
        axes_cs_lon.add_collection(lc_lon)
        axes_cs_lat.add_collection(lc_lat)
    pts = crs.transform_points(ccrs.Geodetic(), lons, lats)
    x = pts[:, 0]
    y = pts[:, 1]
    axes_map.scatter(x, y, s=mags, marker='o', facecolors='none',
                     edgecolors=colors, linewidths=0.5)
    axes_cs_lat.scatter(depths, y, s=mags, marker='o', facecolors='none',
                        edgecolors=colors, linewidths=0.5)
    axes_cs_lon.scatter(x, depths, s=mags, marker='o', facecolors='none',
                        edgecolors=colors, linewidths=0.5)
    # Plot rough slab interface
    slab_grd = read_slab_model(slab_file)
    # Take single latitude for now
    line_pts = slab_grd[np.where(slab_grd[:, 1] == 47.9274994)]
    pts_trans = crs.transform_points(ccrs.Geodetic(), line_pts[:, 0],
                                     line_pts[:, 1])
    axes_cs_lon.plot(pts_trans[:, 0], -line_pts[:, 2], linewidth=0.75,
                     color='b', label='Modeled slab depth')
    axes_cs_lon.legend()
    # Formatting
    axes_cs_lon.invert_yaxis()
    axes_cs_lat.yaxis.set_ticks_position('right')
    axes_cs_lat.xaxis.set_ticks_position('top')
    axes_cs_lat.yaxis.set_label_position('right')
    axes_cs_lat.xaxis.set_label_position('top')
    axes_cs_lat.set_xlabel('Depth [km]')
    axes_cs_lon.set_ylabel('Depth [km]')
    axes_cs_lat.set_ylabel('Northing [m]')
    axes_cs_lon.set_xlabel('Easting [m]')
    axes_cs_lat.set_xlim([0, 60])
    axes_cs_lon.set_ylim([60, 0])
    axes_map.margins(0, 0)
    axes_cs_lon.margins(0, 0)
    axes_cs_lat.margins(0, 0)
    axes_cs_lon.ticklabel_format(axis='both', style='scientific',
                                 scilimits=(-5, 5))
    axes_cs_lat.ticklabel_format(axis='both', style='scientific',
                                 scilimits=(-5, 5))
    if title:
        plt.suptitle(title, fontsize=18)
    if filename:
        plt.savefig('{}.png'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
    return