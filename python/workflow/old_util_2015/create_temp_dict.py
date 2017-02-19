#!/usr/bin/env python

from obspy import read
from glob import glob
from obspy.core.event import ResourceIdentifier
# Use this for the _50Hz data
# temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/1_sec_5-2/*'
temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/dayproc_4-27/*'
# temp_dir = '/projects/nesi00228/data/templates/nlloc_reloc/dayproc_4-27/*'
temp_files = glob(temp_dir)
template_dict = {}
for filename in temp_files:
    uri_name = 'smi:org.gfz-potsdam.de/geofon/' +\
               filename.split('/')[-1].split('_')[0]
    uri = ResourceIdentifier(uri_name)
    template_dict[uri] = read(filename)

# Corr_group_temps
temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/corr_groups/*'
temp_files = glob(temp_dir)
template_dict = {}
for filename in temp_files:
    temp_name = filename.split('/')[-1].rstrip('.mseed')
    template_dict[temp_name] = read(filename)

# Or this for the non _50Hz stuff
temp_dir2 = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/events_raw/*'
temp_files = glob(temp_dir2)
template_dict2 = {}
for filename in temp_files:
    uri_name = 'smi:org.gfz-potsdam.de/geofon/' +\
               filename.split('/')[-1].split('_')[-1].rstrip('.mseed')
    uri = ResourceIdentifier(uri_name)
    template_dict2[uri] = read(filename)

# Check lengths are the same!!
crap_ids = []
for RID, temp in template_dict.iteritems():
    if len(set([tr.stats.npts for tr in temp])) > 1:
        crap_ids.append(RID)
