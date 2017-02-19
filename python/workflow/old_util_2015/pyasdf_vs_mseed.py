#!/usr/bin/env python

"""Testing the reading speed of pyasdf relative to straight miniseed files"""

from timeit import default_timer as timer
import pyasdf
from obspy import UTCDateTime, read
import fnmatch
import os

starttime = UTCDateTime(2015, 7, 31) - 10
day = starttime.julday
endtime = UTCDateTime(2015, 8, 1) + 10
stas = ['ALRZ', 'WPRZ', 'NS12', 'RT18', 'RT22', 'NS03']
# Read the same day for 5 stations from both datasets
# Pyasdf
pyasdf_strt = timer()
with pyasdf.ASDFDataSet('/media/chet/rotnga_data/pyasdf/mrp_rotnga.h5') as ds:
    for sta in stas:
        for station in ds.ifilter(ds.q.station == sta,
                                  ds.q.starttime >= starttime,
                                  ds.q.endtime <= endtime):
            if not 'st' in locals():
                st = station.raw_recording
            else:
                st += station.raw_recording
pyasdf_stp = timer()
print('pyasdf took %.3f seconds to read' % (pyasdf_stp - pyasdf_strt))

# Daylong mseed only
ms_strt = timer()
raw_dir = '/media/chet/rotnga_data/waveform_data/'
#Delete stream and list of raw_files after day loop
if 'st' in locals():
    del st
if 'raw_files' in locals():
    del raw_files
raw_files = []
for root, dirnames, filenames in os.walk(raw_dir):
    for sta in stas:
        for filename in fnmatch.filter(filenames, 'NZ.' + sta +
                                       '*.2015.' + str(day)):
            raw_files.append(os.path.join(root, filename))
for rawfile in raw_files:
    if not 'st' in locals():
        st = read(rawfile)
    else:
        st += read(rawfile)
ms_stp = timer()
print('mseed took %.3f seconds to read' % (ms_stp - ms_strt))
