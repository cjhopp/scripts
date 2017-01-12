#/usr/bin/env python

"""
This script is the start of the MRP project workflow. It takes a pre-made
pyasdf file and extracts the waveform data, cuts them around the arrival times
held in pyasdf.events and saves the templates as separate files
"""
import sys
sys.path.insert(0, "/projects/nesi00228/EQcorrscan")
sys.path.insert(0, "/projects/nesi00228/scripts")

import datetime
from obspy import readEvents
from data_prep import pyasdf_2_templates

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
            instance = int(args[i + 1])
            print('I will run this for instance %d' % instance)
        elif arg == '--splits':
            splits = int(args[i + 1])
            print('I will divide the days into %d chunks' % splits)
# Read in dat catalog
cat = readEvents('/projects/nesi00228/data/catalogs/year_long/rotnga_raw_cat_2013.xml')

# Establish date range for template creation
# Establish date range for template creation
cat.events.sort(key=lambda x: x.preferred_origin().time)
cat_start = cat[0].origins[-1].time.date
cat_end = cat[-1].origins[-1].time.date
delta = cat_end - cat_start
all_dates = [cat_end - datetime.timedelta(days=x) for x in range(0, delta)]
if split:
    #Determine date range
    split_size = len(all_dates) // splits
    instance_dates = [all_dates[i:i + split_size]
                      for i in range(0, len(all_dates), split_size)]
    inst_dats = instance_dates[instance]
    inst_start = min(inst_dats)
    inst_end = max(inst_dats)
    print('This instance will run from %s to %s'
          % (inst_start.strftime('%Y/%m/%d'),
             inst_end.strftime('%Y/%m/%d')))
else:
    inst_dats = all_dates
# Establish which events are in this range
sch_str_start = 'time >= %s' % inst_start
sch_str_end = 'time < %s' % (inst_end + datetime.timedelta(days=1)).strftime('%Y/%m/%d')
day_cat = cat.filter(sch_str_start, sch_str_end)
# Call template generating function
pyasdf_2_templates('/projects/nesi00228/data/pyasdf/rotnga_2013.h5', day_cat,
                   '/projects/nesi00228/data/templates/2013/30s_raw',
                   length=30, prepick=5, highcut=None, lowcut=None,
                   f_order=None, samp_rate=100, debug=1)