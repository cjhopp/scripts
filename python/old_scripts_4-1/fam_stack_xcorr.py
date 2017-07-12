#!/user/bin/python

"""Cross correlate family stacks with each other to
see how different they really are

Must run from same directory at template_auto_corr.py
until I can figure out how to import from relative path
"""

import os
import cPickle as pickle
import pylab as pl
from template_auto_corr import template_auto_corr
from obspy import Stream, read, UTCDateTime
from obspy.signal.cross_correlation import xcorr
from glob import glob
import numpy as np

#shift_file = open('/home/chet/data/template_pha_shift.txt', 'w')
temp_dir = '/home/chet/data/templates/master_temps'
os.chdir(temp_dir)
#Streamline this if you want to do it for every threshold
ms_files = glob('1.5*.ms')
#Sort files in time
files = ms_files.sort()

xcorrs = template_auto_corr(ms_files)

xcorrs[np.isnan(xcorrs)] = 0
#Plot xcorrs as 'Tartan' diagram using pylab
pl.pcolor(xcorrs)
pl.colorbar()
pl.show()
