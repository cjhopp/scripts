#!/usr/bin/env python

from __future__ import division

import os
import shutil
from glob import glob
from obspy import read

def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def grab_day_wavs_stations(wav_dirs, dto, stations):
    # Helper to recursively crawl paths searching for waveforms for a dict of
    # stachans for one day
    import fnmatch
    from itertools import chain
    from obspy import Stream

    st = Stream()
    wav_files = []
    for path, dirs, files in chain.from_iterable(os.walk(path)
                                                 for path in wav_dirs):
        print('Looking in %s' % path)
        for sta in stations:
            for filename in fnmatch.filter(files, '*.%s.*%d.%03d'
                                           % (sta, dto.year, dto.julday)):
                wav_files.append(os.path.join(path, filename))
    print('Reading into memory')
    for wav in wav_files:
        st += read(wav)
    stachans = [(tr.stats.station, tr.stats.channel) for tr in st]
    for stachan in list(set(stachans)):
        tmp_st = st.select(station=stachan[0], channel=stachan[1])
        if len(tmp_st) > 1 and len(set([tr.stats.sampling_rate
                                        for tr in tmp_st])) > 1:
            print('Traces from %s.%s have differing samp rates'
                  % (stachan[0], stachan[1]))
            for tr in tmp_st:
                st.remove(tr)
            tmp_st.resample(sampling_rate=100.)
            st += tmp_st
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
    return st

def orient_boreholes(sac_dir):
    """
    Take Stefan's ngatamariki borehole orientations and add the azimuth
    to the SAC headers
    :param sac_dir: Directory of sac directories
    :return:
    """
    # Stefans orientations for EH1 from teleseisms
    bh_dict = {'NS12': [283.73, 13.73],
               'NS13': [292.03, 22.03],
               'NS14': [65.31, 155.31]}
    sac_dirs = glob('{}/*'.format(sac_dir))
    for dir in sac_dirs:
        sacs = glob('{}/*'.format(dir))
        bh_sacs = [sac for sac in sacs if sac.split('_')[-2]
                   in bh_dict.keys() and sac.rstrip('.sac')[-1] != 'Z']
        print(bh_sacs)
        if len(bh_sacs) == 0:
            print('No boreholes')
            continue
        for bh_sac in bh_sacs:
            tr = read(bh_sac)[0]
            if tr.stats.channel in ['EH1', 'EHN']:
                tr.stats.sac['cmpaz'] = bh_dict[tr.stats.station][0]
            elif tr.stats.channel in ['EH2', 'EHE']:
                tr.stats.sac['cmpaz'] = bh_dict[tr.stats.station][1]
            if tr.stats.channel == 'EHN':
                tr.stats.sac['kcmpnm'] = 'EH1'
                tr.stats.channel = 'EH1'
                bh_sac2 = bh_sac.replace('EHN', 'EH1')
                print('Writing file: {}'.format(bh_sac2))
                tr.write(bh_sac2, format='SAC')
                os.remove(bh_sac)
                continue
            elif tr.stats.channel == 'EHE':
                tr.stats.sac['kcmpnm'] = 'EH2'
                tr.stats.channel = 'EH2'
                bh_sac2 = bh_sac.replace('EHE', 'EH2')
                print('Writing file: {}'.format(bh_sac2))
                tr.write(bh_sac2, format='SAC')
                os.remove(bh_sac)
                continue
            print('Writing file: {}'.format(bh_sac))
            tr.write(bh_sac, format='SAC')
    return

def cat_2_stefan_SAC(cat, inv, wav_dirs, outdir, start=None, end=None):
    """
    Temp gen function for Stefan SAC files
    :param cat:
    :param wav_dirs:
    :param outdir:
    :param start:
    :param end:
    :return:
    """
    import os
    from obspy import UTCDateTime
    import datetime
    from eqcorrscan.utils import pre_processing

    cat.events.sort(key=lambda x: x.origins[-1].time)
    if start:
        cat_start = datetime.datetime.strptime(start, '%d/%m/%Y')
        cat_end = datetime.datetime.strptime(end, '%d/%m/%Y')
    else:
        cat_start = cat[0].origins[-1].time.date
        cat_end = cat[-1].origins[-1].time.date
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        print('Processing templates for: %s' % str(dto))
        # Establish which events are in this day
        sch_str_start = 'time >= %s' % str(dto)
        sch_str_end = 'time <= %s' % str(dto + 86400)
        tmp_cat = cat.filter(sch_str_start, sch_str_end)
        if len(tmp_cat) == 0:
            print('No events on: %s' % str(dto))
            continue
        stations = list(set([pk.waveform_id.station_code for ev in tmp_cat
                             for pk in ev.picks]))
        wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
        sta_st = grab_day_wavs_stations(wav_ds, dto, stations)
        print('Processing data:')
        # Process the stream
        try:
            st1 = pre_processing.dayproc(sta_st, lowcut=None,
                                         highcut=None,
                                         filt_order=None,
                                         samp_rate=100.,
                                         starttime=dto, debug=0,
                                         ignore_length=True,
                                         num_cores=2)
        except NotImplementedError or Exception as e:
            print('Found error in dayproc, noting date and continuing')
            print(e)
            with open('%s/dayproc_errors.txt' % outdir,
                      mode='a') as fo:
                fo.write('%s\n%s\n' % (str(date), e))
            continue
        for event in tmp_cat:
            if len(event.picks) < 5:
                print('Too few picks for event. Continuing.')
                continue
            ev_name = str(event.resource_id).split('/')[-1]
            if not os.path.exists('%s/%s' % (outdir, ev_name)):
                os.mkdir('%s/%s' % (outdir, ev_name))
            elif os.path.exists('%s/%s' % (outdir, ev_name)):
                print('Event already written. Moving to next.')
                continue
            big_o = event.origins[-1]
            ev_time = big_o.time
            tr_starttime = ev_time - 5
            tr_endtime = ev_time + 25
            for pick in event.picks:
                # Only take waveforms for stations with P-picks
                # Take all channels for these stations
                # Stefan will make S-picks himself
                pk_sta = pick.waveform_id.station_code
                if pick.phase_hint != 'P':
                    continue
                # Grab just this station from whole day stream
                sta_wavs = st1.select(station=pk_sta)
                # Copy it out of the way and trim
                work_st = sta_wavs.copy().trim(tr_starttime, tr_endtime)
                if work_st == None:
                    continue
                rel_origin_t = ev_time - work_st[0].stats.starttime
                # Grab stationXML
                sta_inv = inv.select(station=pick.waveform_id.station_code)
                for tr in work_st:
                    stachan = '%s.%s' % (tr.stats.station, tr.stats.channel)
                    print('Populating SAC header for ' + stachan)
                    # For each trace manually set the ref time to origin
                    # Create SAC dictionary
                    tr.stats.sac = {}
                    # Reference times (note microsec --> millisec change)
                    tr.stats['sac']['nzyear'] = tr_starttime.year
                    tr.stats['sac']['nzjday'] = tr_starttime.julday
                    tr.stats['sac']['nzhour'] = tr_starttime.hour
                    tr.stats['sac']['nzmin'] = tr_starttime.minute
                    tr.stats['sac']['nzsec'] = tr_starttime.second
                    tr.stats['sac']['nzmsec'] = int(tr_starttime.microsecond
                                                    // 1000)
                    # Origin time in relation to relative time
                    tr.stats['sac']['o'] = rel_origin_t
                    tr.stats['sac']['iztype'] = 9
                    # Event info
                    tr.stats['sac']['evdp'] = big_o.depth / 1000
                    tr.stats['sac']['evla'] = big_o.latitude
                    tr.stats['sac']['evlo'] = big_o.longitude
                    # Network/Station info
                    tr.stats['sac']['knetwk'] = sta_inv[0].code
                    tr.stats['sac']['kstnm'] = sta_inv[0][0].code
                    tr.stats['sac']['stla'] = sta_inv[0][0].latitude
                    tr.stats['sac']['stlo'] = sta_inv[0][0].longitude
                    tr.stats['sac']['stel'] = sta_inv[0][0].elevation
                    # Channel specific info
                    for chan in sta_inv[0][0]:
                        if chan.code == tr.stats.channel:
                            tr.stats['sac']['stdp'] = chan.depth
                            tr.stats['sac']['cmpaz'] = chan.azimuth
                            tr.stats['sac']['kcmpnm'] = chan.code
                            # SAC cmpinc is deg from vertical (not horiz)
                            if chan.dip == -90.0:
                                tr.stats['sac']['cmpinc'] = 180.0
                                tr.stats['sac']['lpspol'] = False
                            elif chan.dip == 90.0:
                                tr.stats['sac']['cmpinc'] = 0.0
                                tr.stats['sac']['lpspol'] = True
                            else:
                                tr.stats['sac']['cmpinc'] = 90.0
                    # Assign the pick time and type if exists
                    if tr.stats.channel == pick.waveform_id.channel_code and \
                            pick.phase_hint == 'P':
                        print('Writing pick to "a" header')
                        tr.stats['sac']['a'] = pick.time - tr.stats.starttime
                        tr.stats['sac']['ka'] = pick.phase_hint
                    elif tr.stats.channel == pick.waveform_id.channel_code and \
                            pick.phase_hint == 'S':
                        tr.stats['sac']['t0'] = pick.time - tr.stats.starttime
                        tr.stats['sac']['kt0'] = pick.phase_hint
                    else:
                        print('No pick on %s' % stachan)
                    filename = '%s/%s/%s%s_%s_%s.sac' % (outdir, ev_name,
                                                         ev_name,
                                                         tr.stats.network,
                                                         tr.stats.station,
                                                         tr.stats.channel)
                    print('Writing event ' + filename + ' to file...')
                    tr.write(filename, format="SAC")
    return