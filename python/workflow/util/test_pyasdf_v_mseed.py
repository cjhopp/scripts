from timeit import default_timer as timer
from obspy import Stream, UTCDateTime, read
import pyasdf
import os
import fnmatch

# Make stachans dictionary (easiest tested with GeoNet wavs locally)
stachans = {'WPRZ': ['EHZ', 'EHN', 'EHE'],
            'ALRZ': ['EHZ', 'EHN', 'EHE'],
            'PRRZ': ['EHZ', 'EHN', 'EHE'],
            'HRRZ': ['EHZ', 'EHN', 'EHE'],
            'THQ2': ['EHZ', 'EH1', 'EH2'],
            'ARAZ': ['EHZ', 'EHN', 'EHE']}
date = UTCDateTime(2013, 1, 1)
q_start = date - 10
q_end = date + 86410

asdf_st = timer()
st = Stream()
with pyasdf.ASDFDataSet('/media/rotnga_data/pyasdf/rotnga_2013.h5') as ds:
    for sta, chans in iter(stachans.items()):
        for station in ds.ifilter(ds.q.station == sta,
                                  ds.q.channel == chans,
                                  ds.q.starttime >= q_start,
                                  ds.q.endtime <= q_end):
            st += station.raw_recording
print('ASDF read took %f.3 seconds' % (timer() - asdf_st))
del st

# Now for just miniseed with os.walk
mseed_st = timer()
st = Stream()
wav_files = []
for root, dirnames, filenames in os.walk('/media/rotnga_data/waveform_data/2013/'):
    for sta, chans in iter(stachans.items()):
        for chan in chans:
            for filename in fnmatch.filter(filenames, '*.%s.*.%s*%03d'
                                           % (sta, chan, date.julday)):
                wav_files.append(os.path.join(root, filename))
for afile in wav_files:
    st += read(afile)
print('Miniseed reading took %f.3 seconds' % (timer() - mseed_st))
del st
