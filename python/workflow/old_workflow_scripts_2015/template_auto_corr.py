#!/user/bin/python

import os
from obspy import Stream, read, UTCDateTime
from obspy.signal.cross_correlation import xcorr
import numpy as np

def template_auto_corr(files, write_shifts=False):
    if write_shifts:
        shift_file = open('/home/chet/data/template_pha_shift.txt', 'w')
    xcorrs = np.zeros((len(files), len(files)))
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
            temp_xcorrs = []
            #shifts = []
            for stachan in com_stachan:
                #Use tr.select() to specify sta and chan from stachan list
                temp1_data = temp1.select(station = stachan[0:4], channel = stachan[5:])
                temp2_data = temp2.select(station = stachan[0:4], channel = stachan[5:])
                [index, ccc] = xcorr(temp1_data[0], temp2_data[0], 50)
                temp_xcorrs.append(ccc)
                if write_shifts:
                    #Write phase shifts to file for possible use at later date
                    shift_file.write('%s %s %s %s\n' %(files[j], stachan, ccc, index))
            #What sort of correlation are we doing? Stacked CCC? Mean CCC?
            xcorrs[j, i] = np.mean(temp_xcorrs)
            file_cnt += 1
    if write_shifts:
        shift_file.close()
    return xcorrs
