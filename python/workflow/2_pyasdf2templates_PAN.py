#/usr/bin/env python

"""
This script is the start of the MRP project workflow. It takes a pre-made
pyasdf file and extracts the waveform data, cuts them around the arrival times
held in pyasdf.events and saves the templates as separate files
"""
import sys
sys.path.insert(0, "/projects/nesi00228/EQcorrscan")

import pyasdf
import copy
from obspy import UTCDateTime
from obspy import readEvents
from eqcorrscan.core.template_gen import _template_gen
from eqcorrscan.utils import pre_processing
from timeit import default_timer as timer

"""
Take input arguments --split and --instance from bash which specify slices of
days to run
"""
split = False
instance = False

#Be sure to only give --instance and --splits, otherwise following is not valid
args = sys.argv
if '--instance' in args:
    # Arguments to allow the code to be run in multiple instances
    split = True
    for i, arg in enumerate(args):
        if arg == '--instance':
            instance = int(args[i+1])
            print('I will run this for instance %d' % instance)
        elif arg == '--splits':
            splits = int(args[i+1])
            print('I will divide the days into %d chunks' % splits)

# Read in dat catalog
cat = readEvents('/projects/nesi00228/data/catalogs/2015_nlloc_final_run02_group_refined.xml')

# Establish date range for template creation
start_day = UTCDateTime(2015, 01, 01)
end_day = UTCDateTime(2015, 12, 31)
all_dates = range(start_day.julday, end_day.julday+1)
ndays = len(all_dates)
if split:
    #Determine date range
    split_size = ndays // splits
    instance_dates = [all_dates[i:i+split_size]
                      for i in range(0, ndays, split_size)]
    inst_dats = instance_dates[instance]
    print('This instance will run from day %03d to %03d' % (min(inst_dats),
                                                            max(inst_dats)))
else:
    inst_dats = all_dates

for day in inst_dats:
    print('Processing templates for julday: %03d' % day)
    dto = UTCDateTime('2015' + str('%03d' % day))
    q_start = dto - 10
    q_end = dto + 86410
    # Establish which events are in this day
    sch_str_start = 'time >= %s' % str(dto)
    sch_str_end = 'time <= %s' % str(dto + 86400)
    tmp_cat = cat.filter(sch_str_start, sch_str_end)
    if len(tmp_cat) == 0:
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
    with pyasdf.ASDFDataSet('/projects/nesi00228/data/pyasdf/mrp_rotnga.h5') as ds:
        for sta, chans in stachans.iteritems():
            for station in ds.ifilter(ds.q.station == sta,
                                      ds.q.channel == chans,
                                      ds.q.starttime >= q_start,
                                      ds.q.endtime <= q_end):
                if not 'st' in locals():
                    st = station.raw_recording
                else:
                    st += station.raw_recording
    wav_read_stop = timer()
    print('Reading waveforms took %.3f seconds' % (wav_read_stop
                                                   - wav_read_start))
    print('Merging stream and preprocessing...')
    st.merge(fill_value='interpolate')
    #Cut the stream to a manageable size
    st1 = pre_processing.dayproc(st, lowcut=1.0, highcut=20.0,
                                 filt_order=3, samp_rate=50.0,
                                 starttime=dto, debug=2, parallel=True,
                                 as_float32=True)
    print('Feeding stream to _template_gen...')
    for event in tmp_cat:
        print('Copying stream to keep away from the trim...')
        trim_st = copy.deepcopy(st1)
        ev_name = str(event.resource_id).split('/')[-1]
        template = _template_gen(event.picks, trim_st, length=4,
                                 swin='all', prepick=0.5)
        # temp_list.append(template)
        print('Writing event ' + ev_name + ' to file...')
        template.write('/projects/nesi00228/data/templates/nlloc_reloc/' +
                       'dayproc_4-27/' + ev_name +
                       '_50Hz.mseed', format="MSEED")
        del trim_st
    del tmp_cat, st1, st
