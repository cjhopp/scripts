#!/usr/bin/python

"""
Run matched filter detection on a tribe of Templates over waveforms in
a waveform directory (formatted as in above Tribe construction func)
"""

import sys
import logging

from datetime import datetime, timedelta
from obspy import UTCDateTime, Stream, read
from obspy.clients.fdsn import Client
from eqcorrscan.core.match_filter import Tribe, Party, MatchFilterError

import numpy as np
from timeit import default_timer as timer



##### User-defined stuff #####

tribe_file = '/clusterfs/bear/GMF_1/Amplify_EGS/tribes/dac/JVTM_top25-stations_2012-2021_clean.tgz'
# wav_dir = '/clusterfs/bear/chopp/chet-amplify/waveforms/dac'
outdir = '/clusterfs/bear/chopp/chet-amplify/parties/dac'

match_params = {'threshold': 10., 'threshold_type': 'MAD', 'trig_int': 5.,
                'cores': 16, 'save_progress': False, 'parallel_process': True,
                'plot': False, 'process_cores': 8, 'overlap': None, 'return_stream': True}

lag_params = {'min_cc': 0.7, 'shift_len': 0.4, 'interpolate': True,
              'cores': 16, 'process_cores': 8, 'plot': False,
              'plotdir': '/clusterfs/bear/chopp/chet-amplify/lag_calc_plots/dac'}

extract_params = {'prepick': 30., 'length': 90.,
                  'outdir': '/clusterfs/bear/chopp/chet-amplify/event_wavs/dac'}

##### end user-defined stuff #####

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
            continue
        # Specific to station LB.BMN. Ignore Guralp even if picks...
        if tr.stats.location == '02':
            rmtrs.append(tr)
            continue
    for rt in rmtrs:
        stream.traces.remove(rt)
    return stream


# Time this script
script_start = timer()
# Take input arguments --split and --instance from bash which specify slices of
# days to run and which days to slice up.

split = False
instance = False

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
# Set up logging

logging.basicConfig(
    filename='tribe-detect_{}.txt'.format(instance),
    level=logging.ERROR,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

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
# FDSN server on Amplify seiscomp machine
client = Client('http://131.243.224.51:8085')
for date in date_generator(inst_dats[0], inst_dats[-1]):
    dto = UTCDateTime(date)
    jday = dto.julday
    print('Running {}\nJday: {}'.format(dto, jday))
    try:
        day_party, daylong = tribe.client_detect(client=client, starttime=dto, endtime=dto + 86400, **match_params)
    except (OSError, IndexError, MatchFilterError) as e:
        print(e)
        continue
    print('Declustering')
    # Decluster day party
    day_party.decluster(trig_int=match_params['trig_int'])
    day_dets = [d for f in day_party for d in f]
    # Do the lag calcing
    print('Lag calc')
    day_party.lag_calc(
        stream=daylong, pre_processed=False, **lag_params)
    party += day_party
    print('Extracting event waveforms')
    # Extract the days detection waveforms
    pp = extract_params['prepick']
    length = extract_params['length']
    outd = extract_params['outdir']
    for d in day_dets:
        d_st = daylong.slice(starttime=d.detect_time - pp,
                             endtime=d.detect_time - pp + length)
        d_st.write('{}/{}.ms'.format(outd, d.id), format='MSEED')
# Write out the Party object
print('Writing instance party object to file')
party.write('{}/Party_{}_{}'.format(outdir,
                                    inst_start.strftime('%Y-%m-%d'),
                                    inst_end.strftime('%Y-%m-%d')))
# Print out runtime
script_end = timer()
print('Instance took %.3f seconds' % (script_end - script_start))