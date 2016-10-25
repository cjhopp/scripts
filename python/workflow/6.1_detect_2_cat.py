#!/usr/bin/env python

"""Script for taking a csv of detections from matched filter, reading them to DETECTION objects \
and creating a catalog for the whole set. Optionally, these can be written to their own miniseed files
"""
import sys
sys.path.insert(0, '/home/chet/EQcorrscan')
from eqcorrscan.core.match_filter import DETECTION
from eqcorrscan.utils import cat_util
from eqcorrscan.utils import pre_processing
from obspy import read, UTCDateTime, Stream
from glob import glob
import csv
import pyasdf
from timeit import default_timer as timer
import numpy as np


split=False
instance=False
args = sys.argv
if '--instance' in args:
    # Arguments to allow the code to be run in multiple instances
    split = True
    for i, arg in enumerate(args):
        if arg == '--instance':
            instance = int(args[i+1])
            print('I will run this for instance %d' % instance)
        elif arg == '--splits':
            splits = int(args[i+1]) - 1
            print('I will divide the days into %d chunks' % splits)

# Create template dict but key it to template name from matched filt (NOT URI)
temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/1_sec_5-2/*'
# temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/dayproc_4-27/*'
# temp_dir = '/projects/nesi00228/data/templates/nlloc_reloc/dayproc_4-27/*'
temp_files = glob(temp_dir)
template_dict = {}
for filename in temp_files:
    temp_name = filename.split('/')[-1].split('_')[0]
    template_dict[temp_name] = read(filename)
# Extract the station info from the templates
stachans = {tr.stats.station: [] for name, template in template_dict.iteritems()
            for tr in template}
for name, template in template_dict.iteritems():
    for tr in template:
        # chan_code = 'EH' + tr.stats.channel[1]
        chan_code = 'EHZ'
        if chan_code not in stachans[tr.stats.station]:
            stachans[tr.stats.station].append(chan_code)
detect_list = []
detect_csv = '/home/chet/data/detections/2015_alltemps_1sec/2015_all_det_1sec.csv'
with open(detect_csv, 'rb') as file:
    reader = csv.reader(file)
    for row in reader:
        if int(row[4]) > 5:
            detect_list.append(DETECTION(template_name=row[0].split('_')[0], detect_time=UTCDateTime(row[1]),
                                         detect_val=float(row[2]), threshold=float(row[3]), no_chans=int(row[4]),
                                         typeofdet='corr'))
# Establish date range for this run
start_day = UTCDateTime(2015, 01, 01)
end_day = UTCDateTime(2015, 01, 01)
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
    dto = UTCDateTime('2015' + str('%03d' % day))
    q_start = dto - 10
    q_end = dto + 86410
    wav_read_start = timer()
    # Be sure to go +/- 10 sec to account for GeoNet shit timing
    st = Stream()
    with pyasdf.ASDFDataSet('/media/chet/rotnga_data/pyasdf/mrp_rotnga.h5') as ds:
        for sta, chans in stachans.iteritems():
            for station in ds.ifilter(ds.q.station == str(sta),
                                      ds.q.channel == chans,
                                      ds.q.starttime >= q_start,
                                      ds.q.endtime <= q_end):
                st += station.raw_recording
    wav_read_stop = timer()
    print('Reading waveforms took %.3f seconds' % (wav_read_stop
                                                   - wav_read_start))
    merg_strt = timer()
    st.merge(fill_value='interpolate')
    merg_stp = timer()
    print('Merging took %.3f seconds' % (merg_stp - merg_strt))
    proc_strt = timer()
    st1 = pre_processing.dayproc(st, lowcut=1.0, highcut=20.0,
                                 filt_order=3, samp_rate=50.0,
                                 starttime=dto, debug=2, parallel=True)
    del st
    proc_stp = timer()
    print('Pre-processing took %.3f seconds' % (proc_stp - proc_strt))
    # Grab all detections from this day
    day_dets = [det for det in detect_list if det.detect_time.julday == day]
    # Select random sample from list of dets for plotting purposes
    rand_dets = [day_dets[i] for i in np.random.choice(range(len(day_dets)), 30)]
    # XXX TODO MAKE SURE YOU UNDERSTAND THESE PARAMETERS!! Specifically, pre_pick, post_pick, max_lag
    new_cat = cat_util.detections_2_cat(day_dets, template_dict, st1, temp_prepick=0.5,
                                        max_lag=5, cc_thresh=0.3, debug=1)