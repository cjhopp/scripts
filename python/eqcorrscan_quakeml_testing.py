#!/usr/bin/python

r"""Testing of the new EQcorrscan functions for QuakeML integration
"""

from glob import glob
from eqcorrscan.core.template_gen import from_QuakeML, from_SeisHub, _template_gen, from_sfile
from obspy import readEvents, read
import fnmatch
import os

raw_files = []
raw_dir = '/home/chet/data/test_mseed'
#Recursively search a directory for specific files matching desired day and stachan
for root, dirnames, filenames in os.walk(raw_dir):
    for filename in fnmatch.filter(filenames, '*.177'):
        raw_files.append(os.path.join(root, filename))
for rawfile in raw_files:
    if not 'st' in locals():
        print('Adding: '+rawfile)
        st = read(rawfile)
        print(str(len(st)))
    else:
        print('Adding: '+rawfile)
        st += read(rawfile)
        print(str(len(st)))
st.merge()

# client = Client('http://localhost:8080')
xml_file = '/home/chet/xml_validation/obspyck_20151120033331.xml'
sfile = '/home/chet/seismo/REA/TEST_/1996/06/25-0337-31L.S199606'

st1 = from_QuakeML(xml_file, st, lowcut=2, highcut=20, samp_rate=100.00,
                   filt_order=3, length=10, prepick=1, swin='all')

st1 = from_sfile(sfile, lowcut=2, highcut=20, samp_rate=100.00,
                 filt_order=3, length=10, swin='all')
