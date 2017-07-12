#!/usr/bin/env python

"""
Messing with data availability via pyasdf and/or raw mseed files
"""
import sys
sys.path.insert(0, '/home/chet/EQcorrscan')
import pyasdf
import os
import fnmatch
from obspy import UTCDateTime

start = UTCDateTime(2015, 01, 01)
end = UTCDateTime(2015, 12, 31)
all_dates = range(start.julday, end.julday+1)
wav_dir = '/media/chet/rotnga_data/waveform_data/2015'
availables = {}
for date in all_dates:
    dto = '2015%03d' % date
    print(dto)
    for root, dirnames, filenames in os.walk(wav_dir):
        for filename in fnmatch.filter(filenames, '*2015.%03d' % date):
            sta = filename.split('.')[1]
            chan = filename.split('.')[3]
            if dto not in availables:
                availables[dto] = {sta: [chan]}
            else:
                if sta not in availables[dto]:
                    availables[dto][sta] = [chan]
                else:
                    availables[dto][sta].append(chan)
