#!/usr/bin/python

"""
Main script to organize and prepare data followed by
running EQcorrscan match_filter to generate detections
"""
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, "/projects/nesi00228/EQcorrscan")

from obspy import UTCDateTime
from eqcorrscan.core.match_filter import Tribe, Party
from timeit import default_timer as timer
from datetime import datetime, timedelta

def partition(lst, n):
    # Helper function for dividing catalog into --splits roughly-equal parts
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))]
            for i in range(n)]

def grab_day_wavs(wav_dirs, dto, stachans):
    # Helper to recursively crawl paths searching for waveforms for a dict of
    # stachans for one day
    import os
    import fnmatch
    from itertools import chain
    from obspy import read, Stream

    st = Stream()
    wav_files = []
    for path, dirs, files in chain.from_iterable(os.walk(path)
                                                 for path in wav_dirs):
        print('Looking in %s' % path)
        for sta, chans in iter(stachans.items()):
            for chan in chans:
                for filename in fnmatch.filter(files,
                                               '*.%s.*.%s*%d.%03d'
                                                       % (
                                               sta, chan, dto.year,
                                               dto.julday)):
                    wav_files.append(os.path.join(path, filename))
    print('Reading into memory')
    for wav in wav_files:
        print('Reading file: %s' % wav)
        st += read(wav)
    return st

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
            splits = int(args[i+1])
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
tribe_rd_strt = timer()
# Reading tribe
tribe = Tribe().read('/projects/nesi00228/data/templates/12-15/Tribe_12-15_P.tgz')
print('Reading Tribe tarball took %s seconds' % (timer() - tribe_rd_strt))
# Extract the station info from the templates
stachans = {tr.stats.station: [] for temp in tribe
            for tr in temp.st}
for temp in tribe:
    for tr in temp.st:
        # Don't hard code vertical channels!!
        chan_code = 'EH' + tr.stats.channel[-1]
        if chan_code not in stachans[tr.stats.station]:
            stachans[tr.stats.station].append(chan_code)
# Specify locations of waveform files
wav_dirs = ['/projects/nesi00228/data/miniseed/']
inst_partay = Party()
for day in inst_dats:
    dto = UTCDateTime(day)
    wav_read_start = timer()
    wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
    st = grab_day_wavs(wav_ds, dto, stachans)
    st.merge(fill_value='interpolate')
    wav_read_stop = timer()
    print('Reading waveforms took %.3f seconds' % (wav_read_stop
                                                   - wav_read_start))
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
    # RUN MATCH FILTER (looping through chunks of templates due to RAM)
    print('Starting correlation runs for %s' % str(day))
    inst_partay += tribe.detect(stream=st, threshold=8.0, threshold_type='MAD',
                                trig_int=2., plotvar=False, daylong=True,
                                group_size=500, debug=3,
                                parallel_process=False)
# Write out the Party object
print('Writing instance party object to file')
inst_partay.write('/projects/nesi00228/data/detections/parties_12-15/Party_%s_%s'
                  % (inst_start.strftime('%Y-%m-%d'),
                     inst_end.strftime('%Y-%m-%d')))
#Print out runtime
script_end = timer()
print('Instance took %.3f seconds' % (script_end - script_start))
