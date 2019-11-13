#!/usr/bin/python

"""
Testing out EQcorrscan for use with MRP dataset
"""

import pdb, sys, os, fnmatch
import matplotlib.pyplot as plt
from obspy import read, Stream, readEvents, UTCDateTime
from core import template_gen, match_filter
from utils import pre_processing
from utils.Sfile_util import PICK
from utils.EQcorrscan_plotting import pretty_template_plot as tplot#import numpy as np

pick_file = '/home/chet/data/test_pha_0607.txt'
#Generate list matching pattern given in fnmatch.filter() -recursive process-
wav_files = []
for root, dirnames, filenames in os.walk('/Volumes/GeoPhysics_07/users-data/hoppche/gabe_backup/'):
    for filename in fnmatch.filter(filenames, '*.SAC'):
        wav_files.append(os.path.join(root, filename))

#Read in all waveforms from list generated above
for wavefile in wav_files:
    if not 'st' in locals():
        st = read(wavefile)
    else:
        st += read(wavefile)

st.resample(50)
st.normalize()
# Process the data
for tr in st:
    day=tr.stats.starttime.date
    tr=pre_processing.dayproc(tr, 1.0, 20.0, 3, 50.0, matchdcf.debug, day)
"""
#Format for assigning PICK class

test_pick=PICK('FOZ', 'SZ', 'I', 'P', '1', 'C', UTCDateTime("2012-03-26")+1,
             coda=10, amplitude=0.2, peri=0.1,
             azimuth=10.0, velocity=20.0, AIN=0.1, SNR='',
             azimuthres=1, timeres=0.1,
             finalweight=4, distance=10.0,
             CAZ=2)
"""
templates = []
test_picks = []
prev_line_time = 0
ev_cnt = 0 #Event counter
#Create a PICK class for the P and S pick, then add to test_picks
with open(pick_file) as fid:
    for line in fid:
        s = line.split(" ")
        ev_time = UTCDateTime(s[0]) #To know if we're at a new event
        temp_pickP = PICK(s[1], 'EZ', 'I', 'P', '1', 'C', UTCDateTime(s[2]), coda=10, amplitude=0.2, peri=0.1,
        azimuth=10.0, velocity=20.0, AIN=0.1, SNR='',
        azimuthres=1, timeres=0.1,
        finalweight=4, distance=10.0,
        CAZ=2)
        s[3] = s[3][:-1]
        temp_pickS = PICK(s[1], 'EN', 'I', 'S', '1', 'C', UTCDateTime(s[3]), coda=10, amplitude=0.2, peri=0.1,
        azimuth=10.0, velocity=20.0, AIN=0.1, SNR='',
        azimuthres=1, timeres=0.1,
        finalweight=4, distance=10.0,
        CAZ=2)
#If this is a new event, append current template to templates and plot, then clear all temp vars
        if prev_line_time != 0:
            if ev_time != prev_line_time:
                ev_cnt += 1
                template = template_gen._template_gen(test_picks,st,3.0,'all', prepick = 1.0)
                templates += [template]
                #Save templates to miniseed
                template.write('/home/chet/data/'+str(ev_time.julday)+'_'+str(ev_cnt)+'_template.ms',\
                           format='MSEED')
                test_picks = [] #clear test_picks
                test_picks.append(temp_pickP) #Start appending new event picks
                test_picks.append(temp_pickS)
                prev_line_time = ev_time #Save ev_time as previous for next iteration
            else: #If prev_line_time same as last line, continue appending event picks
                test_picks.append(temp_pickP)
                test_picks.append(temp_pickS)
                prev_line_time = ev_time
        else:
            test_picks.append(temp_pickP)
            test_picks.append(temp_pickS)
            prev_line_time = ev_time
#Process last template outside of readline() loop
ev_cnt += 1
template = template_gen._template_gen(test_picks,st,3.0,'all', prepick = 1.0)
templates += [template]
template.write('/home/chet/data/'+str(ev_time.julday)+'_'+str(ev_cnt)+'_template.ms',\
           format='MSEED')
