#!/usr/bin/python
"""
Read in all waveforms and picks from Gabe's SAC files
Careful!! SAC headers user dependent. This 'a' = P pick, 't0' = S pick holds true
for Gabe but may not hold true with other user's data. Emma?
"""
import pdb, sys, os, fnmatch
import matplotlib.pyplot as plt
from obspy import read, Stream, readEvents, UTCDateTime
from glob import glob

length = 4 #Length of template in seconds
path = '/Volumes/GeoPhysics_07/users-data/matsonga/MRP_PROJ/data/mastersData/eventFilesLong/'

sub_dirs = filter(os.path.isdir, [os.path.join(path,f) for f in os.listdir(path)])
for a_dir in sub_dirs:
    if a_dir[-4:] != 'copy' and a_dir[-4:] != 'Test': #Specific to Gabe's directory
        os.chdir(a_dir)
        files = glob('*.SAC')

        for sac_file in files:
            st1 = read(sac_file)
            st = st1.copy()
            st.normalize()
            for tr in st:
                time_0 = UTCDateTime(tr.stats.starttime) #start time of SAC file
                origin = time_0 + 10 #For file naming consistency with Gabe's data
                p_picks = tr.stats.sac.a #P phase
                s_picks = tr.stats.sac.t0 #S phase
                chan = tr.stats.channel
                if p_picks != float(-12345.0):
                    pre_cut = (time_0 + p_picks) - 1 #Cut one second before arrival
                    post_cut = pre_cut + length
                    tr.trim(pre_cut, post_cut, nearest_sample = False)
                    if len(tr.data) == (tr.stats.sampling_rate*length)+1: #if one sample too long, correct
                        tr.data=tr.data[0:-1]
                    elif len(tr.data) < (tr.stats.sampling_rate*length):
                        break
                    #if tr.stats.station[0:2] == 'RT':
                        #print(a_dir)
                        #print(tr.stats.station)
                    if not 'template' in locals():
                        template = Stream(tr)
                    else:
                        template += tr
                if s_picks != float(-12345.0):
                    pre_cut = (time_0 + s_picks) - 1 #Cut one second before arrival
                    post_cut = pre_cut + length
                    tr.trim(pre_cut, post_cut, nearest_sample = False)
                    if len(tr.data) == (tr.stats.sampling_rate*length)+1: #if one sample too long, correct
                        tr.data=tr.data[0:-1]
                    elif len(tr.data) < (tr.stats.sampling_rate*length):
                        break
                    #if tr.stats.station[0:2] == 'RT':
                        #print(a_dir)
                        #print(tr.stats.station)
                    if not 'template' in locals():
                        template = Stream(tr)
                    else:
                        template += tr
    if 'template' in locals(): #Place zeros in from of 1-digit numbers
        if origin.month < 10:
            month = '0'+str(origin.month)
        else:
            month = str(origin.month)
        if origin.day < 10:
            day = '0'+str(origin.day)
        else:
            day = str(origin.day)
        if origin.hour < 10:
            hour = '0'+str(origin.hour)
        else:
            hour = str(origin.hour)
        if origin.minute < 10:
            minute = '0'+str(origin.minute)
        else:
            minute = str(origin.minute)
        if origin.second < 10:
            second = '0'+str(origin.second)
        else:
            second = str(origin.second)
        template.write('/home/chet/data/templates/'+str(origin.year)+'-'+month+'-'+day+'_'+\
                        hour+':'+minute+':'+second+'.'+\
                        str(origin.microsecond)+'_template.ms', format='MSEED')
        del template, st
