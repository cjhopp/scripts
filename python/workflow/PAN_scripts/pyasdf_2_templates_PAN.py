#/usr/bin/env python

"""
This script is the start of the MRP project workflow. It takes a pre-made
pyasdf file and extracts the waveform data, cuts them around the arrival times
held in pyasdf.events and saves the templates as separate files
"""
import sys
sys.path.insert(0, "/projects/nesi00228/EQcorrscan")
sys.path.insert(0, "/projects/nesi00228/scripts/python/workflow")

from datetime import datetime, timedelta
from obspy import readEvents
from data_prep import pyasdf_2_templates

# Helper function for dividing catalog into --splits roughly-equal parts
def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))]
            for i in xrange(n)]
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
cat.events.sort(key=lambda x: x.preferred_origin().time)
cat_start = cat[0].origins[-1].time.date
cat_end = cat[-1].origins[-1].time.date
delta = (cat_end - cat_start).days
all_dates = [cat_end - timedelta(days=x) for x in range(0, delta)][::-1]
# Sanity check. If splits > len(all_dates) overwrite splits to len(all_dates)
if splits > len(all_dates):
    print('Splits > # dates in catalog. Splits will now equal len(all_dates)')
    splits = len(all_dates)
if split:
    split_dates = partition(all_dates, splits)
    #Determine date range
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
# Establish which events are in this range
sch_str_start = 'time >= %s' % (str(datetime.combine(inst_start,
                                                     datetime.min.time())))
sch_str_end = 'time < %s' % (str(datetime.combine(inst_end + timedelta(days=1),
                                                  datetime.min.time())))
day_cat = cat.filter(sch_str_start, sch_str_end)
# Call template generating function
pyasdf_2_templates('/projects/nesi00228/data/pyasdf/rotnga_2013.h5', day_cat,
                   '/projects/nesi00228/data/templates/2013/30s_raw',
                   length=30, prepick=5, highcut=None, lowcut=None,
                   f_order=None, samp_rate=100, debug=1)