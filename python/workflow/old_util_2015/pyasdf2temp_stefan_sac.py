#!/usr/bin/env python

"""
This script is the start of the MRP project workflow. It takes a pre-made
pyasdf file and extracts the waveform data, cuts them around the arrival times
held in pyasdf.events and saves the templates as separate files
"""
from __future__ import division


def cat_2_stefan_SAC(cat, wav_dirs, outdir, start, end):
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
        q_start = dto - 10
        q_end = dto + 86410
        # Establish which events are in this day
        sch_str_start = 'time >= %s' % str(dto)
        sch_str_end = 'time <= %s' % str(dto + 86400)
        tmp_cat = cat.filter(sch_str_start, sch_str_end)
        if len(tmp_cat) == 0:
            print('No events on: %s' % str(dto))
            continue
        # Which stachans we got?
        stachans = {pk.waveform_id.station_code: [] for ev in tmp_cat
                    for pk in ev.picks}
        for ev in tmp_cat:
            for pk in ev.picks:
                chan_code = pk.waveform_id.channel_code
                if chan_code not in stachans[pk.waveform_id.station_code]:
                    stachans[pk.waveform_id.station_code].append(chan_code)
        wav_read_start = timer()
        # Be sure to go +/- 10 sec to account for GeoNet shit timing
        wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
        st = grab_day_wavs(wav_ds, dto, stachans)
        wav_read_stop = timer()
        print('Reading waveforms took %.3f seconds' % (wav_read_stop
                                                       - wav_read_start))

        temp_list = []
        for event in tmp_cat:
            ev_name = str(event.resource_id).split('/')[-1]
            os.mkdir('%s/%s' % (outdir, ev_name))
            big_o = event.preferred_origin()
            ev_time = big_o.time
            ev_date = UTCDateTime(ev_time.date)
            used_stachans = []
            for pick in event.picks:
                print('Processing data: ' + tmp_st[0].stats.station)
                tmp_st.merge(fill_value='interpolate')
                tmp_st.trim(tr_starttime, tr_endtime)
                if len(tmp_st) == 0:
                    continue
                rel_origin_t = ev_time - tmp_st[0].stats.starttime
                # Grab stationXML from pyasdf
                network = station.StationXML[0]
                sta_info = network[0]
                if 'st' not in locals():
                    st = tmp_st
                else:
                    st += tmp_st
                del tmp_st
                print('st contains ' + str(len(st)) + ' channels')
                for tr in st:
                    stachan = tr.stats.station + '.' +\
                              tr.stats.channel
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
                    tr.stats['sac']['nzmsec'] = int(tr_starttime.microsecond // 1000)
                    # Origin time in relation to relative time
                    tr.stats['sac']['o'] = rel_origin_t
                    tr.stats['sac']['iztype'] = 9
                    # Event info
                    tr.stats['sac']['evdp'] = big_o.depth / 1000
                    tr.stats['sac']['evla'] = big_o.latitude
                    tr.stats['sac']['evlo'] = big_o.longitude
                    # Network/Station info
                    tr.stats['sac']['knetwk'] = network.code
                    tr.stats['sac']['kstnm'] = sta_info.code
                    tr.stats['sac']['stla'] = sta_info.latitude
                    tr.stats['sac']['stlo'] = sta_info.longitude
                    tr.stats['sac']['stel'] = sta_info.elevation
                    # Channel specific info
                    for chan in sta_info:
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
                    # Assign the pick time and type
                    if tr.stats.channel == pick.waveform_id.channel_code and \
                            pick.phase_hint == 'P':
                        print('Writing pick to "a" header')
                        tr.stats['sac']['a'] = pick.time - tr.stats.starttime
                        tr.stats['sac']['ka'] = pick.phase_hint
                    elif tr.stats.channel == pick.waveform_id.channel_code and \
                            pick.phase_hint == 'S':
                        tr.stats['sac']['t0'] = pick.time - tr.stats.starttime
                        tr.stats['sac']['kt0'] = pick.phase_hint
                    filename = '/media/rotnga_data/templates/' +\
                               'stefan_30sec/2015_dets_sac/' + ev_name + '/' +\
                               ev_name + tr.stats.network + '_' +\
                               tr.stats.station + '_' + tr.stats.channel +\
                               '.sac'
                    print('Writing event ' + filename + ' to file...')
                    if stachan not in used_stachans:
                        tr.write(filename, format="SAC")
                        used_stachans.append(stachan)
                del st
            del used_stachans
        del cat
    return

# Correct channel names in the catalog to len(3)
for ev in cat:
    for pk in ev.picks:
        pk.waveform_id.channel_code = pk.waveform_id.channel_code[0] + 'H' + pk.waveform_id.channel_code[1]
# Write or_id origin_time and P_picks to text file
import csv
with open('/Users/home/hoppche/data/filt_cat_Stefan.csv', 'wb') as f:
    writer = csv.writer(f)
    for ev in cat:
        writer.writerow([str(ev.resource_id).split('/')[-1], ev.preferred_origin().time])
        for pk in ev.picks:
            writer.writerow([pk.phase_hint, pk.time])
