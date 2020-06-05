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

from obspy import UTCDateTime, read, Stream
from obspy.geodetics import kilometer2degrees
from obspy.signal.trigger import coincidence_trigger, plot_trigger
from eqcorrscan.utils.pre_processing import dayproc
from phasepapy.phasepicker import aicdpicker, ktpicker
from phasepapy.associator import tables1D, assoc1D, plot1D
from phasepapy.associator import tt_stations_1D

import obspy.taup as taup

def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def build_databases(param_file):
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
    db_name = paramz['Database']['name']
    # If the associator database exists delete it first
    if os.path.exists('{}_associator.db'.format(db_name)):
        os.remove('{}_associator.db'.format(db_name))
        os.remove('{}_tt.db'.format(db_name))
    # Our SQLite databases are:
    db_assoc = 'sqlite:///{}_associator.db'.format(db_name)
    db_tt = 'sqlite:///{}_tt.db'.format(db_name)  # Traveltime database

    # Connect to our databases
    engine_assoc = create_engine(db_assoc, echo=False)
    # Create the tables required to run the 1D associator
    tables1D.Base.metadata.create_all(engine_assoc)
    Session = sessionmaker(bind=engine_assoc)
    session = Session()
    return session, db_assoc, db_tt

# TODO Read parameter file out here, then feed dates to trigger/picker to
# TODO facilitate parallel processing

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
        day_wavs = glob('{}/{}/**/*{}.ms'.format(
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
        st = st.merge(fill_value='interpolate')
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
            details=True)
        if plot:
            plot_triggers(day_trigs, st, trigger_stream,
                          sta_lta_params, outdir=trig_p['plot_outdir'])
        if not trig_p['output']['write_wavs']:
            print('Not writing waveforms')
            return trigs
        print('Writing triggered waveforms')
        output_param = trig_p['output']
        for t in day_trigs:
            trig_s = trigger_stream.slice(
                starttime=t['time'] - output_param['pre_trigger'],
                endtime=t['time'] + output_param['post_trigger'])
            trig_s.write(
                '{}/Trig_{}.ms'.format(output_param['waveform_outdir'],
                                       t['time']), format='MSEED')
        trigs += day_trigs
    return trigs


def picker(param_file, db_sesh):
    """
    Pick the first arrivals (P) for triggered waveforms
    :param method:
    :return:
    """
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
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
    for trig_f in trigger_files:
        print('Picking {}'.format(trig_f))
        st = read(trig_f)
        for tr in st:
            print(tr.id)
            print(tr)
            scnl, picks, polarity, snr, uncert = picker.picks(tr)
            t_create = UTCDateTime().datetime
            # Add each pick to the database
            print('Adding to db')
            for i, pick in enumerate(picks):
                new_pick = tables1D.Pick(scnl, pick.datetime, polarity[i],
                                         snr[i], uncert[i], t_create)
                db_sesh.add(new_pick)  # Add pick i to the database
            print('Committing to db')
            db_sesh.commit()  # Commit the pick to the database
    return


def build_tt_tables(param_file, inventory, tt_db):
    with open(param_file, 'r') as f:
        paramz = yaml.load(f, Loader=yaml.FullLoader)
    assoc_paramz = paramz['Associator']
    # Create a connection to an sqlalchemy database
    tt_engine = create_engine(tt_db, echo=False)
    tt_stations_1D.BaseTT1D.metadata.create_all(tt_engine)
    TTSession = sessionmaker(bind=tt_engine)
    tt_session = TTSession()
    # Now add all individual stations to tt sesh
    seeds = list(set(['{}.{}.{}'.format(net.code, sta.code, chan.location_code)
                      for net in inventory for sta in net for chan in sta]))
    for seed in seeds:
        net, sta, loc = seed.split('.')
        sta_inv = inventory.select(network=net, station=sta, location=loc)[0][0]
        chan = sta_inv[0]
        station = tt_stations_1D.Station1D(sta, net, loc, sta_inv.latitude,
                                           sta_inv.longitude,
                                           sta_inv.elevation - chan.depth)
        # Save the station locations in the database
        tt_session.add(station)
        tt_session.commit()
    # We will use IASP91 here but obspy.taup does let you build your own model
    velmod = taup.TauPyModel(model='iasp91')
    # Define our distances we want to use in our lookup table
    max_dist = assoc_paramz['max_dist']
    dist_spacing = assoc_paramz['dist_spacing']
    max_depth = assoc_paramz['max_depth']
    depth_spacing = assoc_paramz['depth_spacing']
    distance_km = np.arange(0, max_dist + dist_spacing, dist_spacing)
    depth_km = np.arange(0, max_depth + depth_spacing, depth_spacing)
    for d_km in distance_km:
        d_deg = kilometer2degrees(d_km)
        ptimes = []
        stimes = []
        p_arrivals = velmod.get_travel_times(
            source_depth_in_km=15., distance_in_degree=d_deg,
            phase_list=['P', 'p'])
        for p in p_arrivals:
            ptimes.append(p.time)
        s_arrivals = velmod.get_travel_times(
            source_depth_in_km=15., distance_in_degree=d_deg,
            phase_list=['S', 's'])
        for s in s_arrivals:
            stimes.append(s.time)
        tt_entry = tt_stations_1D.TTtable1D(d_km, d_deg, np.min(ptimes),
                                            np.min(stimes),
                                            np.min(stimes) - np.min(ptimes))
        tt_session.add(tt_entry)
        tt_session.commit()  # Probably faster to do the commit outside of loop but oh well
    tt_session.close()
    return


## Plotting ##

def plot_triggers(triggers, st, cft_stream, params, outdir):
    """Helper to plot triggers, traces and characteristic funcs"""
    for trig in triggers:
        seeds = trig['trace_ids']
        # Clip around trigger time
        st_slice = st.slice(starttime=trig['time'] - 10,
                            endtime=trig['time'] + 30)
        cft_slice = cft_stream.slice(starttime=trig['time'] - 10,
                                     endtime=trig['time'] + 30)
        fig, ax = plt.subplots(nrows=len(seeds), sharex='col')
        fig.suptitle('Detection: {}'.format(trig['time']))
        fig.subplots_adjust(hspace=0.)
        for i, sid in enumerate(seeds):
            tps = params[sid]
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
                           bbox=bbox_props, ha='center')
            ax[i].set_yticks([])
        ax[i].set_xlabel('Time [s]', fontsize=14)
        if os.path.isdir(outdir):
            plt.savefig('{}/Trig_{}.png'.format(outdir, trig['time']))
            plt.close('all')
        else:
            plt.show()
    return


def plot_picks():
    return