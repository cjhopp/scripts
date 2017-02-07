#!/usr/bin/env python

"""
This script is the start of the MRP project workflow. It takes a pre-made
pyasdf file and extracts the waveform data, cuts them around the arrival times
held in pyasdf.events and saves the templates as separate files
"""
from __future__ import division

import pyasdf
import os
from obspy import UTCDateTime, read_events

# Make list of catalog parts
cat = read_events('/Users/home/hoppche/data/2015_dets_nlloc_Sherburn_filt_Stefan_ccval0.5_pks9.xml')
with pyasdf.ASDFDataSet('/media/rotnga_data/pyasdf/mrp_rotnga.h5') as ds:
    # For each event and station/channel, cut around arrival times
    temp_list = []
    for event in cat:
        ev_name = str(event.resource_id).split('/')[-1]
        os.mkdir('/media/rotnga_data/templates/stefan_30sec/2015_dets_sac/' +
                 ev_name)
        big_o = event.preferred_origin()
        ev_time = big_o.time
        tr_starttime = ev_time - 5
        tr_endtime = ev_time + 25
        ev_date = UTCDateTime(ev_time.date)
        print('Reading event ' + ev_name + ' from pyasdf...')
        used_stachans = []
        for pick in event.picks:
            for station in ds.ifilter(ds.q.station == pick.waveform_id.station_code,
                                      ds.q.starttime >= UTCDateTime(pick.time.date) - 10,
                                      ds.q.endtime <= UTCDateTime(pick.time.date) + 86410):
                tmp_st = station.raw_recording
                print('Processing data: ' + tmp_st[0].stats.station)
                print(tmp_st)
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
