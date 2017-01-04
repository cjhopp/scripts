#!/user/bin/python

import pickle
import csv
import os
import numpy as np
from glob import glob
from obspy import UTCDateTime, read, Stream, Trace

path = '/home/chet/data/'

#Recreate list of template files in chronological order
temp_dir = '/home/chet/data/templates/'
os.chdir(temp_dir)
ms_files = glob(temp_dir+'*.ms')
#Sort files in time
ms_files.sort()
files = ms_files

#Load in dictionaries for thresholds
with open('/home/chet/data/fam_dictionaries.pickle', 'r') as f:
    [avgdict, constdict] = pickle.load(f)

for key in avgdict: #For each threshold
    fam_num = 0
    for fam in avgdict[key][1]: #For each family that thresh produced
        stachan_dict = {} #Dictionary[stachan] for each family for easy summing of data
        delay_times = {} #Save avg delay times per family to input them back into stack
        for index in fam: #For each index in that family, add traces to stachan_dict
            st = read(files[index])
            arr_times = {} #Organize arrival times for each event
            for tr in st:
                stachan = tr.stats.station+'.'+tr.stats.channel #create stachan key for all dicts
                arr_times[stachan] = UTCDateTime(tr.stats.starttime) #Build dictionary of arrival times for this event
                if stachan in stachan_dict: #Build parallel dictionary of data
                    stachan_dict[stachan].append(tr)
                else:
                    stachan_dict[stachan] = [tr]
            min_stachan = min(arr_times, key=arr_times.get)
            for key2 in arr_times: #Now put relative tt delays for each stachan in delay_times dict
                if not key2 in delay_times:
                    delay_times[key2] = [arr_times[key2] - arr_times[min_stachan]]
                else:
                    delay_times[key2].append(arr_times[key2] - arr_times[min_stachan])
        #Sum the data in each stachan key, keep track of average delay time?
        master = Stream()
        i = 0
        for key3 in stachan_dict:
            if key3 in stachan_dict and key3 in delay_times:
                trace_cnt = 0
                for trace in stachan_dict[key3]:
                    if len(trace) != 400 and len(trace) != 800: #Remove stubborn traces of incorrect length
                        trace = np.zeros(trace.stats.sampling_rate*4)
                        trace_cnt += 1
                if len(stachan_dict[key3]) == 1:
                    master.append(stachan_dict[key3][0]) #No stack needed, already a Trace object
                else:
                    master.append(Trace(reduce(np.add, stachan_dict[key3])))
                master[i].stats['sampling_rate'] = stachan_dict[key3][0].stats.sampling_rate
                master[i].stats['network'] = 'NZ'
                master[i].stats['station'] = key3[:4]
                master[i].stats['channel'] = key3[-3:]
                # if key == 1.0: #Print standard deviation for each stachan for each family master for each threshold
                #     print(str(key)+' '+key3+str(np.std(delay_times[key3])))
                #Add avg delay time for that stachan to template stachan
                master[i].stats['starttime'] = UTCDateTime(0 + (sum(delay_times[key3])/float(len(delay_times[key3]))))
                i+=1
        master.sort()
        master.normalize()
        if fam_num < 10:
            master.write(temp_dir+'master_temps/'+str(key)+'_'+'f0'+str(fam_num)+'.ms', format='MSEED')
        else:
            master.write(temp_dir+'master_temps/'+str(key)+'_'+'f'+str(fam_num)+'.ms', format='MSEED')
        fam_num += 1
        #del master, delay_times, arr_times, stachan_dict
