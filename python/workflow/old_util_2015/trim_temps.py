#!/usr/bin/env python

"""
Take longer templates and trim them down to shorter ones
"""

from glob import glob
from obspy import read, read_events

temp_files = glob('/media/chet/hdd/seismic/NZ/templates/rotnga_2015/dayproc_4-27/*')

for temp_file in temp_files:
    temp_id = temp_file.split('/')[-1].split('_')[0]
    temp4sec = read(temp_file).copy()
    for tr in temp4sec:
        tr.trim(starttime=tr.stats.starttime + 0.4,
                endtime=tr.stats.starttime + 1.4)
    temp4sec.write('/media/chet/hdd/seismic/NZ/templates/rotnga_2015/' +
                   '1_sec_5-2/%s_1sec.mseed' % temp_id, format='MSEED')
