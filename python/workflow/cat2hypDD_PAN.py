#!/usr/bin/env python

import sys
sys.path.insert(0, '/projects/nesi00228/EQcorrscan')
# Working with obspy 0.10 on PAN so go with deprecated readEvents
from obspy import readEvents, UTCDateTime, read
from obspy.core.event import ResourceIdentifier
from glob import glob
from eqcorrscan.utils import catalog_to_dd
import warnings

split=False
instance=False

cat = readEvents('/projects/nesi00228/data/catalogs/2015_Rawlinson_tempSNR_improved_5+picks_0.50ccval.xml')
# Be aware this directory contains the full 10 second waveforms for each detection
# This is in contrast to the dir of the same name locally
temp_dir = '/projects/nesi00228/data/templates/2015_1sec_detections/*'
# temp_dir = '/projects/nesi00228/data/templates/nlloc_reloc/dayproc_4-27/*'
temp_files = glob(temp_dir)
template_dict = {}
for filename in temp_files:
    if filename.split('/')[-1].rstrip('.mseed').split('_')[-1] == 'self':
        uri_name = 'smi:local/' + \
                   filename.split('/')[-1].rstrip('.mseed')
    else:
        utc = UTCDateTime(filename.split('_')[-1].rstrip('.mseed'))
        uri_name = 'smi:local/' +\
                   filename.split('/')[-1].rstrip('.mseed')
    uri = ResourceIdentifier(uri_name)
    template_dict[uri] = read(filename)

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

#Determine catalog range
split_size = len(cat) // splits
start_index = instance * split_size
loop_inds = range(start_index, start_index + split_size)
for index in loop_inds:
    catalog_to_dd.write_corr_parallel(index, cat, template_dict, extract_len=0.2, pre_pick=0.1,
                                      shift_len=0.1,
                                      outdir='/projects/nesi00228/data/hypoDD/ccval_0.50_cats/weight_tempSNR/max_sep_2_min_link_6/',
                                      lowcut=1.0, highcut=20.0, max_sep=2,
                                      coh_thresh=0.60, min_link=6, plotvar=False)
