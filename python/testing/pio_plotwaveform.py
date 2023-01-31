import logging
from obspy import Stream
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import pyasdf
from dug_seis.processing.dug_trigger import dug_trigger
#from dug_seis.processing.event_processing import event_processing
import re

asdf_folder = '/Users/rinaldia/Documents/DUG-Seis_Output/raw'   # 'raw'  # Location of .h5 file
files = sorted([f for f in os.listdir(asdf_folder) if f.endswith('.h5') ])  # generates a list of
stations = [i - 1 for i in [19,20,21,22,23,24,25,26]]

sta = Stream()


for f in files:
    ds = pyasdf.ASDFDataSet(asdf_folder + '/' +f, mode='r')
    wf_list = ds.waveforms.list()
    for k in stations:
        sta += ds.waveforms[wf_list[k]].raw_recording

for i in range(len(sta.traces)):
    sta.traces[i].stats.delta=5.e-6


sta.merge()

dt =UTCDateTime("2019-06-12T14:13:00.00")
dt2 = UTCDateTime("2019-06-12T14:13:10.00")

#dt =UTCDateTime("2019-06-12T14:00:00.00")
#dt2 = UTCDateTime("2019-06-12T14:30:00.00")


#sta.trim(dt,dt2)

#sta.decimate(8)

#sta[0].write(sta[0].id+"2019-06-12_1240-1435_ch21-22.mseed",format="MSEED")


for tr in sta:
    tr.write(tr.id+"2019-06-12_141300-141310.dat",format="SH_ASC")

#sta.write("2019-06-12_1240-1435_ch21-22.mseed",format="MSEED")

#sta.taper(0.05)

#sta.filter('bandpass',freqmin=1000., freqmax=10000., corners=4)
#sta.plot()
#plt(sta.slice(dt,dt2).traces[0].data); plt.show()

# daje=[]
# for _tr in sta.traces:
#     daje.append(_tr.data)
# daje_arr=np.array(daje)
#
# plt.figure()
# plt.plot(daje_arr)
# plt.show()