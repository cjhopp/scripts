#!/usr/bin/python

"""
Main script to organize and prepare data followed by
running EQcorrscan match_filter to generate detections
"""
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, "/projects/nesi00228/EQcorrscan")

from obspy import read, UTCDateTime, Catalog
from eqcorrscan.core import match_filter
from eqcorrscan.utils import pre_processing
from glob import glob
from timeit import default_timer as timer
import pyasdf
import csv
from datetime import datetime, timedelta

# Helper function for dividing catalog into --splits roughly-equal parts
def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))]
            for i in range(n)]
# Time this script
script_start = timer()
"""
Take input arguments --split and --instance from bash which specify slices of
days to run and which days to slice up.
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
        elif arg == '--start':
            cat_start = datetime.strptime(str(args[i + 1]), '%d/%m/%Y')
        elif arg == '--end':
            cat_end = datetime.strptime(str(args[i + 1]), '%d/%m/%Y')
        else:
            NotImplementedError('Argument %s not supported' % str(arg))
delta = (cat_end - cat_start).days + 1
all_dates = [cat_end - timedelta(days=x) for x in range(0, delta)][::-1]
# Sanity check. If splits > len(all_dates) overwrite splits to len(all_dates)
if splits > len(all_dates):
    print(
    'Splits > # dates in catalog. Splits will now equal len(all_dates)')
    splits = len(all_dates)
if split:
    split_dates = partition(all_dates, splits)
    # Determine date range
    try:
        inst_dats = split_dates[instance]
    except IndexError:
        print('Instance no longer needed. Downsize --splits for this job')
        sys.exit()
    inst_start = min(inst_dats)
    inst_end = max(inst_dats)
    print('This instance will run from %s to %s'
          % (inst_start.strftime('%Y/%m/%d'),
             inst_end.strftime('%Y/%m/%d')))
else:
    inst_dats = all_dates
#Read in templates and names
temp_dir = '/projects/nesi00228/data/templates/2013/1sec_3-20Hz/*'
temp_files = glob(temp_dir)
templates = [read(temp_file) for temp_file in temp_files]
template_names = [temp_file.split('/')[-1].rstrip('.mseed')
                  for temp_file in temp_files]
# Extract the station info from the templates
stachans = {tr.stats.station: [] for template in templates
            for tr in template}
for temp in templates:
    for tr in temp:
        # Don't hard code vertical channels!!
        chan_code = 'EH' + tr.stats.channel[1]
        if chan_code not in stachans[tr.stats.station]:
            stachans[tr.stats.station].append(chan_code)
# Create a catalog for this instance which gets added to then written
inst_cat = Catalog()
for day in inst_dats:
    dto = UTCDateTime(day)
    q_start = dto - 10
    q_end = dto + 86410
    wav_read_start = timer()
    # Be sure to go +/- 10 sec to account for GeoNet shit timing
    with pyasdf.ASDFDataSet('/projects/nesi00228/data/pyasdf/rotnga_%d.h5'
                            % dto.year) as ds:
        for sta, chans in iter(stachans.items()):
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
    merg_strt = timer()
    st.merge(fill_value='interpolate')
    merg_stp = timer()
    print('Merging took %.3f seconds' % (merg_stp - merg_strt))
    proc_strt = timer()
    try:
        st1 = pre_processing.dayproc(st, lowcut=3.0, highcut=20.0,
                                     filt_order=3, samp_rate=50.0,
                                     starttime=dto, debug=2, parallel=True,
                                     num_cores=6)
    except NotImplementedError or Exception:
        print('Found error in dayproc, noting date and continuing')
        with open('/projects/nesi00228/logs/dayproc_errors.txt', mode='a') as fo:
            fo.write('%s\n' % str(day))
        continue
    del st
    proc_stp = timer()
    print('Pre-processing took %.3f seconds' % (proc_stp - proc_strt))
    # RUN MATCH FILTER (looping through chunks of templates due to RAM)
    chunk_temps = partition(templates, 120)
    chunk_temp_names = partition(template_names, 120)
    print('Starting correlation runs for %s' % str(day))
    i = 0
    for temps, temp_names in zip(chunk_temps, chunk_temp_names):
        i += 1 # Silly counter for debug
        grp_corr_st = timer()
        print('On template group %d of %d' % (i, len(chunk_temps)))
        dets, cat, sts = match_filter.match_filter(temp_names, temps, st1,
                                                   threshold=8.0,
                                                   threshold_type='MAD',
                                                   trig_int=1.0,
                                                   plotvar=False,
                                                   cores=12,
                                                   output_cat=True,
                                                   extract_detections=True,
                                                   debug=2)
        # Append detections to a file for this instance to check later
        print('Correlations for group %d took %.3f sec, now extracting them'
              % (i, timer() - grp_corr_st))
        extrct_st = timer()
        with open('/projects/nesi00228/data/detections/raw_det_txt/%s/%d_dets.txt'
                  % (str(dto.year), instance), mode='a') as fo:
            det_writer = csv.writer(fo)
            for det, st in zip(dets, sts):
                print('Writing %s_%s.mseed to files' % (det.template_name,
                                                        det.detect_time))
                det_writer.writerow([det.template_name,
                                     det.detect_time, det.detect_val,
                                     det.threshold, det.no_chans])
                # Write wav for each detection
                st.write('/projects/nesi00228/data/detections/raw_det_wavs/' +
                         '%d/%s_%s.mseed' % (dto.year, det.template_name,
                                             det.detect_time), format='MSEED')
        print('Extracting wavs took %.3f seconds' % (timer() - extrct_st))
        cat.write('/projects/nesi00228/data/catalogs/raw_det_cats/%d/inst%d_group%d_dets.xml'
                  % (dto.year, instance, i), format='QUAKEML')
        del dets, cat, sts
#Print out runtime
script_end = timer()
print('Instance took %.3f seconds' % (script_end - script_start))
