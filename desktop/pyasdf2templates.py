#/usr/bin/env python

"""
This script is the start of the MRP project workflow. It takes a pre-made
pyasdf file and extracts the waveform data, cuts them around the arrival times
held in pyasdf.events and saves the templates as separate files
"""

import pyasdf
from glob import glob
from obspy import UTCDateTime, read_events
from eqcorrscan.core.template_gen import _template_gen
from eqcorrscan.utils import pre_processing
from eqcorrscan.utils.timer import Timer

#Make list of catalog parts
cat_list = glob('/Users/home/hoppche/data/catalog_parts/*part*')

with pyasdf.ASDFDataSet('/media/rotnga_data/pyasdf/mrp_rotnga.h5') as ds:
    for catalog in cat_list:
        #Read in catalog
        cat = read_events(catalog)
        # For each event and station/channel, cut around arrival times
        temp_list = []
        for event in cat:
            ev_name = str(event.resource_id).split('/')[2]
            ev_time = event.preferred_origin().time
            ev_date = UTCDateTime(ev_time.date)
            print('Reading event ' + ev_name + ' from pyasdf...')
            for pick in event.picks:
                for station in ds.ifilter(ds.q.station == pick.waveform_id.station_code,
                                          ds.q.channel == pick.waveform_id.channel_code,
                                          ds.q.starttime >= UTCDateTime(pick.time.date) - 10,
                                          ds.q.endtime <= UTCDateTime(pick.time.date) + 86410):
                    if 'st' not in locals():
                        st = station.raw_recording
                    else:
                        st += station.raw_recording
            if 'st' not in locals():
                print('No data extracted from pyasdf!!')
                continue
            else:
                print('Merging stream and preprocessing...')
                st.merge(fill_value='interpolate')
                #Cheap check if the waveforms were extracted for all the picks
                if len(event.picks) != len(st):
                    print('Not the same number of traces as picks!!!')
                st1 = pre_processing.dayproc(st, lowcut=1.0, highcut=20.0,
                                               filt_order=3, samp_rate=100,
                                               starttime=ev_date, debug=0)
                st.trim(ev_time - 5, ev_time + 20)
                print('Feeding stream to _template_gen...')
                template = _template_gen(event.picks, st1, length=4,
                                         swin='all', prepick=0.5)
                # temp_list.append(template)
                print('Writing event ' + ev_name + ' to file...')
                template.write('/media/rotnga_data/templates/2015_dayproc/' +
                               ev_name+'.mseed', format="MSEED")
                del st, st1, template
        del cat
