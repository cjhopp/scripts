#!/user/bin/python

"""
Take a set of templates and cross-correlate them with each other using openCV
similar to Calum

Will be used for identifying multiplets in MRP dataset
"""
import os
import matplotlib.pyplot as plt
import pylab as pl
from obspy import Stream, read, UTCDateTime
from obspy.signal.cross_correlation import xcorr
from glob import glob
#from core.match_filter import normxcorr2, _channel_loop
import numpy as np

temp_dir = '/home/chet/data/templates/'
os.chdir(temp_dir)
ms_files = glob('*.ms')
#Sort files by splitting filename at '_' characters (Year, month, day, hrmin.sec)
files = sorted(ms_files, key=lambda x: (int(x.split('_')[0]),\
    int(x.split('_')[1]), int(x.split('_')[2]), float(x.split('_')[3])))

xcorrs = [[0 for i in xrange(len(files))] for i in xrange(len(files))]
file_cnt = 0
#For each template, correlate with each other template and write value to xcorrs
for j in range(len(files)):
    print('Running template '+files[j])
    temp1 = read(files[j])
    temp1.resample(50)
    for i in range(len(files)):
        temp2 = read(files[i])
        temp2.resample(50)
        #print('correlating with '+files[i])
        #Make list of common sta.chans between both templates
        temp1_stachan = []
        temp2_stachan = []
        for tr1 in temp1:
            temp1_stachan.append(tr1.stats.station+'.'+tr1.stats.channel)
        for tr2 in temp2:
            temp2_stachan.append(tr2.stats.station+'.'+tr2.stats.channel)
        com_stachan = set(temp1_stachan).intersection(temp2_stachan)
        #Run the cross-correlation loop
        temp_xcorrs = [] #List of
        shift_dict = {}
        for stachan in com_stachan:
            #Use tr.select() to specify sta and chan from stachan list
            temp1_data = temp1.select(station = stachan[0:4], channel = stachan[5:])
            temp2_data = temp2.select(station = stachan[0:4], channel = stachan[5:])
            [index, ccc] = xcorr(temp1_data[0], temp2_data[0], 50)
            #Create list of list[stachan, ccc, index] to be stored in array xcorrs
            temp_xcorrs.append([stachan, ccc, index])
            #shift_dict[stachan] = index
        #What sort of correlation are we doing? Stacked CCC? Mean CCC?
        xcorrs[j, i] = temp_xcorrs
        mean_maxcorr[i, j] = np.mean(xcorrs[i,j,:,1])
        file_cnt += 1
#Replace NaNs in xcorrs with 0 or something more meaningful
mean_maxcorr[np.isnan(mean_maxcorr)] = -1

#Plot xcorrs as 'Tartan' diagram using pylab
pl.pcolor(mean_maxcorr)
pl.colorbar()
pl.show()
