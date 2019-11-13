#/usr/bin/env python

"""
Import waveforms into pyasdf database
"""

import os
import pyasdf
import fnmatch
from obspy import read

h5_file = '/media/rotnga_data/pyasdf/mrp_rotnga.h5'
wav_dir = '/media/rotnga_data/waveform_data/2015'

with pyasdf.ASDFDataSet(h5_file) as ds:

    wav_files = []
    for root, dirnames, filenames in os.walk(wav_dir):
        for filename in fnmatch.filter(filenames, '*2015.*'):
            wav_files.append(os.path.join(root, filename))

    for _i, a_file in enumerate(wav_files):
        print("Adding mseed file %i of %i..." % (_i + 1, len(wav_files)))
        #Add waveforms
        st = read(a_file)
        ds.add_waveforms(st, tag="raw_recording")
        del st
