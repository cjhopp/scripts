#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/chet/EQcorrscan')

"""Creating the input files for HypoDD from nlloc relocated catalog"""
import numpy as np
import os
from collections import Counter
from obspy import read_events, read, UTCDateTime
from glob import glob
from obspy.core.event import ResourceIdentifier
from eqcorrscan.utils import catalog_to_dd

cat = read_events('/media/chet/hdd/seismic/NZ/catalogs/2015_dets_nlloc/2015_dets_nlloc_Sherburn.xml')

# Need to deal with events which have origins
cat.events = [ev for ev in cat if len(ev.origins) != 0]

# Remove dup events by rounding to nearest second and removing duplicates
times = [(i, np.round(float(ev.preferred_origin().time.strftime('%Y%m%d%H%M%S.%f')))) for i, ev in enumerate(cat)]
seen = {}
result = []
for time in times:
    if time[1] in seen:
        continue
    else:
        seen[time[1]] = 1
        result.append(time)
indices, times = zip(*result)
cat.events = [ev for i, ev in enumerate(cat) if i in indices]

# Now we'll remove any events which have no arrivals?
cat.events = [ev for ev in cat if len(ev.preferred_origin().arrivals) > 0]

# Write to HypoDD input files...
catalog_to_dd.write_catalog(cat)

####################################################
# Need a template dict for each located detection
# Read in catalog processed by above lines...
cat = read_events('/media/chet/hdd/seismic/NZ/catalogs/2015_dets_nlloc/2015_dets_nlloc_Sherburn_no_dups.xml')

temp_dir = '/media/rotnga_data/templates/2015_det2cats/*'
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

# Correcting different naming of resource_id and wav filenames
cnt = 0
for ev in cat:
    if str(ev.resource_id).split('_')[-1] == 'self':
        cnt += 1
        print('This is a self detection: %d' % cnt)
        # Should theoretically be able to extract detect time from first pick - template prepick time...
        det_time = min([pk.time for pk in ev.picks]) - 0.1
        wav_id = ResourceIdentifier(str(ev.resource_id).split('/')[-1].split('_')[0] + '_' + str(det_time))
        find_file = temp_dir.rstrip('*') + str(wav_id) + '.mseed'
        if os.path.isfile(find_file):
            new_fname = temp_dir.rstrip('*') + str(ev.resource_id).split('/')[-1].split('_')[0] + '_self.mseed'
            print('Renaming file: %s to %s' % (find_file, new_fname))
            os.rename(find_file, new_fname)
# Take subset of catalog for testing purposes
test_cat = cat[:100].copy()
# Write dt.cc files...
# Supress xcorr_pick_correction warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    catalog_to_dd.write_correlations(cat, template_dict, extract_len=3.0, pre_pick=0.5,
                                     shift_len=0.3, lowcut=1.0, highcut=20.0, max_sep=4,
                                     min_link=6, plotvar=False)
# Reset stdout
