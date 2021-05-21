#!/usr/bin/python

"""
Run matched filter detection on a tribe of Templates over waveforms in
a waveform directory (formatted as in above Tribe construction func)
"""

import sys
import logging

from glob import glob
from datetime import datetime, timedelta
from obspy import UTCDateTime, Stream, read
from eqcorrscan.core.match_filter import Tribe, Party, MatchFilterError

import numpy as np
from timeit import default_timer as timer


logging.basicConfig(
    filename='tribe-detect_run.txt',
    level=logging.ERROR,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

# Helper function for dividing catalog into --splits roughly-equal parts
def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))]
            for i in range(n)]


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def clean_daylong(stream):
    """
    Convenience func to clean out traces that will raise Exceptions in
    EQcorrscan preprocessing functions (e.g. too many zeros and too short)
    :return:
    """
    rmtrs = []
    for tr in stream:
        if len(np.nonzero(tr.data)[0]) < 0.5 * len(tr.data):
            print('{} mostly zeros. Removing'.format(tr.id))
            rmtrs.append(tr)
            continue
        if tr.stats.endtime - tr.stats.starttime < 0.8 * 86400:
            print('{} less than 80 percent daylong. Removing'.format(tr.id))
            rmtrs.append(tr)
            continue
        # Check for spikes
        if (tr.data > 2 * np.max(np.sort(
                np.abs(tr.data))[0:int(0.99 * len(tr.data))]
                                 ) * 1e7).sum() > 0:
            print('{} is spiky. Removing'.format(tr.id))
            rmtrs.append(tr)
    for rt in rmtrs:
        stream.traces.remove(rt)
    return stream


# Time this script
script_start = timer()
"""
Take input arguments --split and --instance from bash which specify slices of
days to run and which days to slice up.
"""
split = False
instance = False


### User-defined file paths and detect parameters ###
tribe_file = ''
wav_dir = ''
outdir = ''

param_dict = {'threshold': 8., 'threshold_type': 'MAD', 'trig_int': 5.,
              'cores': 8, 'save_progress': False, 'parallel_process': False,
              'plot': False}

#Be sure to only give --instance and --splits, otherwise following is not valid
args = sys.argv
if '--instance' in args and '--splits' in args:
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
            cat_start = datetime.strptime(str(args[i+1]), '%d/%m/%Y')
        elif arg == '--end':
            cat_end = datetime.strptime(str(args[i+1]), '%d/%m/%Y')
        else:
            NotImplementedError('Argument %s not supported' % str(arg))
else:
    NotImplementedError('Need to provide --instance and --splits')
# Make datetime list
delta = (cat_end - cat_start).days + 1
all_dates = [cat_end - timedelta(days=x) for x in range(delta)][::-1]
# Sanity check. If splits > len(all_dates) overwrite splits to len(all_dates)
if splits > len(all_dates):
    print('Splits > # dates in catalog; splits will now equal len(all_dates)')
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

# Reading tribe
tribe = Tribe().read(tribe_file)

party = Party()
net_sta_loc_chans = list(set([(pk.waveform_id.network_code,
                               pk.waveform_id.station_code,
                               pk.waveform_id.location_code,
                               pk.waveform_id.channel_code)
                              for temp in tribe
                              for pk in temp.event.picks]))
for date in date_generator(inst_dats[0], inst_dats[-1]):
    dto = UTCDateTime(date)
    jday = dto.julday
    print('Running {}\nJday: {}'.format(dto, jday))
    wav_files = []
    for nslc in net_sta_loc_chans:
        day_wav_fs = glob('{}/{}/{}/{}/**/{}.{}.{}.{}.{}.{:03d}.ms'.format(
            wav_dir, date.year, nslc[0], nslc[1], nslc[0], nslc[1],
            nslc[2], nslc[3], date.year, jday),
                          recursive=True)
        wav_files.extend(day_wav_fs)
    daylong = Stream()
    print('Reading wavs')
    for wav_file in wav_files:
        daylong += read(wav_file)
    # Deal with shitty CN sampling rates
    for tr in daylong:
        if not ((1 / tr.stats.delta).is_integer() and
                tr.stats.sampling_rate.is_integer()):
            tr.stats.sampling_rate = round(tr.stats.sampling_rate)
    daylong = clean_daylong(daylong.merge(fill_value='interpolate'))
    print('Running detect')
    try:
        party += tribe.detect(stream=daylong, **param_dict)
    except (OSError, IndexError, MatchFilterError) as e:
        print(e)
        continue
# Write out the Party object
print('Writing instance party object to file')
party.write('{}/Party_{}_{}'.format(outdir,
                                    inst_start.strftime('%Y-%m-%d'),
                                    inst_end.strftime('%Y-%m-%d')))
# Print out runtime
script_end = timer()
print('Instance took %.3f seconds' % (script_end - script_start))