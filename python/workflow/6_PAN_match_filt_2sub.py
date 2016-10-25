#!/usr/bin/python

"""
Main script to organize and prepare data followed by
running EQcorrscan match_filter to generate detections
"""
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, "/projects/nesi00228/EQcorrscan")

from obspy import read, UTCDateTime
from eqcorrscan.core import match_filter
from eqcorrscan.utils import pre_processing
from glob import glob
from timeit import default_timer as timer
import pyasdf
import csv
import itertools

# Time this script
script_start = timer()

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
            splits = int(args[i+1]) - 1
            print('I will divide the days into %d chunks' % splits)

#Read in templates and names
temp_dir = '/projects/nesi00228/data/templates/nlloc_reloc/corr_groups/*2nd*'
temp_files = glob(temp_dir)
templates = [read(temp_file) for temp_file in temp_files]
template_names = [temp_file.split('/')[-1].rstrip('.mseed')
                  for temp_file in temp_files]
# Extract the station info from the templates
stachans = {tr.stats.station: [] for template in templates
            for tr in template}
for template in templates:
    for tr in template:
        # chan_code = 'EH' + tr.stats.channel[1]
        chan_code = 'EHZ'
        if chan_code not in stachans[tr.stats.station]:
            stachans[tr.stats.station].append(chan_code)
# Establish date range for this match filter run
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
out_name = '/projects/nesi00228/data/%d_%03d-%03d_2sub.csv' % (start_day.year,
                                                               min(inst_dats),
                                                               max(inst_dats))
# Set plot directory
plot_dir = '/projects/nesi00228/data/plots/'
with open(out_name, 'wb') as out_file:
    # Create the csv writer object for detections
    det_writer = csv.writer(out_file)
    for day in inst_dats:
        dto = UTCDateTime('2015' + str('%03d' % day))
        q_start = dto - 10
        q_end = dto + 86410
        wav_read_start = timer()
        # Be sure to go +/- 10 sec to account for GeoNet shit timing
        with pyasdf.ASDFDataSet('/projects/nesi00228/data/pyasdf/mrp_rotnga.h5') as ds:
            for sta in stachans:
                for station in ds.ifilter(ds.q.station == str(sta),
                                          ds.q.channel == stachans[sta],
                                          ds.q.starttime >= q_start,
                                          ds.q.endtime <= q_end):
                    if not 'st' in locals():
                        st = station.raw_recording
                    else:
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
                                     starttime=dto, debug=2, parallel=True,
                                     as_float32=True)
        del st
        proc_stp = timer()
        print('Pre-processing took %.3f seconds' % (proc_stp - proc_strt))
        # RUN MATCH FILTER (looping through chunks of templates due to RAM)
        chunk_size = len(templates) // 40
        chunk_temps = [templates[i:i+chunk_size]
                       for i in range(0, len(templates), chunk_size)]
        chunk_temp_names = [template_names[i:i+chunk_size]
                            for i in range(0, len(template_names), chunk_size)]
        for temps, temp_names in itertools.izip(chunk_temps, chunk_temp_names):
            detections = match_filter.match_filter(temp_names, temps, st1,
                                                   threshold=8.0,
                                                   threshold_type='MAD',
                                                   trig_int=6.0, plotvar=False,
                                                   cores='all', debug=2)
            # Write detections to a file to check later
            for detection in detections:
                det_writer.writerow([detection.template_name,
                                     detection.detect_time, detection.detect_val,
                                     detection.threshold, detection.no_chans])
            del detections
#Print out runtime
script_end = timer()
print('Instance took %.3f seconds' % (script_end - script_start))
