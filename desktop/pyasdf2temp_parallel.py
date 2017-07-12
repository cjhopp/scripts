#/usr/bin/env python

"""
This script is the start of the MRP project workflow. It takes a pre-made
pyasdf file and extracts the waveform data, cuts them around the arrival times
held in pyasdf.events and saves the templates as separate files
"""

import pyasdf
from obspy import UTCDateTime, read_events
from eqcorrscan.core.template_gen import _template_gen
from eqcorrscan.utils import pre_processing
from eqcorrscan.utils.timer import Timer
from datetime import timedelta
from dateutil import rrule
from glob import glob

sta_list = ['RT01', 'RT02', 'RT03', 'RT05', 'RT06', 'RT07', 'RT08', 'RT09',
            'RT10', 'RT11', 'RT12', 'RT12', 'RT13', 'RT14', 'RT15', 'RT16',
            'RT17', 'RT18', 'RT19', 'RT20', 'RT21', 'NS01', 'NS02', 'NS03',
            'NS04', 'NS05', 'NS06', 'NS07', 'NS08', 'NS09', 'NS10', 'NS11',
            'NS12', 'NS13', 'NS14', 'NS15', 'NS16', 'NS18', 'WPRZ', 'HRRZ',
            'PRRZ', 'ALRZ', 'ARAZ', 'THQ2', 'RT23', 'RT22']

cat_list = glob('/Users/home/hoppche/data/catalog_parts/*part*')

#Read dataset
with pyasdf.ASDFDataSet('/media/rotnga_data/pyasdf/mrp_rotnga.h5') as ds:
    #Read in catalogs
    for a_cat in cat_list:
        cat = read_events(a_cat)
        ev_times = []
        print('Establishing catalog start/end times...')
        for event in cat:
            ev_times.append(event.origins[0].time)
        #Establish start and end dates of this chunk of catalog
        startday = min(ev_times).date
        endday = max(ev_times).date
        #Loop over each possible day in catalog
        for dt in rrule.rrule(rrule.DAILY, dtstart=startday, until=endday):
            #Figure out start and end times for filter/pyasdf
            starttime = UTCDateTime(dt)
            endtime = UTCDateTime(dt + timedelta(days=1))
            #Convert to string for Catalog.filter
            start_str = 'time > ' + str(starttime)
            end_str = 'time < ' + str(endtime)
            print('Starting day loop for ' + start_str)
            day_cat = cat.filter(start_str, end_str)
            if len(day_cat) == 0:
                print('No events for this day...')
                continue
            print('Reading in waveforms from pyasdf for: ' + start_str
                  + ' --> ' + end_str)
            for station in sta_list:
                for ds_station in ds.ifilter(ds.q.starttime > starttime - 10,
                                             ds.q.endtime < endtime + 10):
                    if not 'st' in locals():
                        st = ds_station.raw_recording
                    else:
                        st += ds_station.raw_recording
            if not 'st' in locals():
                print('No data for this day from pyasdf?!')
                continue
            else:
                print('Merging stream...')
                st.merge(fill_value='interpolate')
                day_st = st.copy()
                for event in day_cat:
                    ev_name = str(event.resource_id).split('/')[2]
                    origin_time = event.origins[0].time
                    print('Trimming data around event time...')
                    day_st.trim(origin_time - 120, origin_time + 120)
                    print('Preprocessing data for day: ' + str(starttime.date))
                    temp_st = pre_processing.shortproc(day_st, lowcut=1.0,
                                                       highcut=20.0, filt_order=3,
                                                       samp_rate=100, debug=0)
                    del day_st
                    print('Feeding stream to _template_gen...')
                    template = _template_gen(event.picks, temp_st, length=4,
                                             swin='all', prepick=0.5)
                    print('Writing event ' + ev_name + ' to file...')
                    template.write('/media/rotnga_data/templates/2015/' +
                                   ev_name+'.mseed', format="MSEED")
                    del temp_st, template
                del day_cat
