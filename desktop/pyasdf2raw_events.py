#/usr/bin/env python

"""
This script is the start of the MRP project workflow. It takes a pre-made
pyasdf file and extracts the waveform data, cuts them around the arrival times
held in pyasdf.events and saves the templates as separate files
"""

import pyasdf
from glob import glob
from obspy import UTCDateTime, read_events

# Make list of catalog parts
cat_list = glob('/Users/home/hoppche/data/catalog_parts/*part*')

with pyasdf.ASDFDataSet('/media/rotnga_data/pyasdf/mrp_rotnga.h5') as ds:
    for catalog in cat_list:
        # Read in catalog
        cat = read_events(catalog)
        # For each event and station/channel, cut around arrival times
        temp_list = []
        for event in cat:
            ev_name = str(event.resource_id).split('/')[2]
            ev_time = event.preferred_origin().time
            ev_date = UTCDateTime(ev_time.date)
            print('Reading event ' + ev_name + ' from pyasdf...')
            used_sta = []
            for pick in event.picks:
                if pick.waveform_id.station_code not in used_sta:
                    for station in ds.ifilter(ds.q.station == pick.waveform_id.station_code,
                                              ds.q.starttime >= UTCDateTime(pick.time.date) - 10,
                                              ds.q.endtime <= UTCDateTime(pick.time.date) + 86410):
                        if 'st' not in locals():
                            st = station.raw_recording
                        else:
                            st += station.raw_recording
                    used_sta.append(pick.waveform_id.station_code)
            if 'st' not in locals():
                print('No data extracted from pyasdf!!')
                continue
            else:
                print('Merging stream and trimming...')
                st.merge(fill_value='interpolate')
                # Cheap check if the waveforms were extracted for all the picks
                if len(event.picks) != len(st):
                    print('Not the same number of traces as picks!!!')
                st.trim(ev_time - 5, ev_time + 20)
                print('Writing event ' + ev_name + ' to file...')
                st.write('/media/rotnga_data/templates/events_raw/raw_' +
                         ev_name+'.mseed', format="MSEED")
                del st
        del cat
