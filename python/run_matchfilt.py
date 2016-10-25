#!/usr/bin/python

"""
Main script to organize and prepare data followed by
running EQcorrscan match_filter to generate detections
"""
from obspy import read
from obspy.core.event import Catalog
from eqcorrscan.core import match_filter, lag_calc
from eqcorrscan.par import match_filter_par as matchdef
from eqcorrscan.utils import pre_processing
import matplotlib.pyplot as plt
import fnmatch
from glob import glob
import os

#Read in templates
temp_dir = '/home/chet/data/templates/master_temps/hierarchy_cluster/no_delays/empirical_SVD/'
#Streamline this if you want to do it for every threshold
ms_files = glob(temp_dir+'*.ms')
ms_files.sort()
template_names = []
temp_tup = []
for file1 in ms_files:
    if not 'templates' in locals():
        temp_data = read(file1)
        templates = [temp_data]
        temp_name = file1.split("/")[-1:][0][:-3]
        template_names.append(temp_name)
        temp_tup.append((temp_name, temp_data))
    else:
        temp_data = read(file1)
        templates += [temp_data]
        template_names.append(file1.split("/")[-1:][0][:-3])
        temp_name = file1.split("/")[-1:][0][:-3]
        template_names.append(temp_name)
        temp_tup.append((temp_name, temp_data))


# Extract the station info from the templates
for template in templates:
    #Filter and downsample sample data
    template = pre_processing.shortproc(template, 1.0, 20.0, 3, 100.0,
                                        matchdef.debug)
    if not 'stachans' in locals():
        stachans = [(tr.stats.station, tr.stats.channel) for tr in template]
    else:
        stachans += [(tr.stats.station, tr.stats.channel) for tr in template]

# Make this a unique list
stachans = list(set(stachans))

# Read in the continuous data for these station, channel combinations
raw_files = []
raw_dir = '/Volumes/GeoPhysics_07/users-data/matsonga/MRP_PROJ/data/mastersData/sac'
#Recursively search a directory for specific files amtching desired day and stachan
for root, dirnames, filenames in os.walk(raw_dir):
    for stachan in stachans:
        for filename in fnmatch.filter(filenames, 'NZ.'+stachan[0]+'*' +
                                       stachan[1][-1]+'*'+'171.SAC'):
            raw_files.append(os.path.join(root, filename))
for rawfile in raw_files:
    if not 'st' in locals():
        st = read(rawfile)
    else:
        st += read(rawfile)

# Merge the data to account for miniseed files being written in chunks
# We need continuous day-long data, so data are padded if there are gaps
st = st.merge(fill_value='interpolate')

# Work out what day we are working on, required as we will pad the data to be daylong
day = st[0].stats.starttime.date

# Process the data in the same way as the template
for tr in st:
    tr = pre_processing.dayproc(tr, 1.0, 20.0, 3, 100.0,
                                matchdef.debug, day)

#Set directory for match filter output plots
plot_dir = '/home/chet/data/plot'
# Compute detections
detections = match_filter.match_filter(template_names, templates, st,
                                       8.0, matchdef.threshtype,
                                       matchdef.trig_int, True, plot_dir,
                                       cores=5)

# Do the lag calculations
new_catalog = lag_calc.lag_calc(detections=detections, detect_data=st,
                                templates=temp_tup, min_cc=0.2)
# We now have a list of detections! We can output these to a file to check later
# f=open('/home/chet/data/test_detections.csv','w')
# for detection in detections:
#     f.write(detection.template_name+', '+str(detection.detect_time)+\
#             ', '+str(detection.detect_val)+', '+str(detection.threshold)+\
#             ', '+str(detection.no_chans)+'\n')
# f.close()

##Instead of saving all of these waveforms, just save the plots as pdf
# wav_dir = '/home/chet/data/detections/'
# det_wav = Stream()
# for detection in detections:
#     st.plot(starttime=detection.detect_time-2, endtime=detection.detect_time+8,
#             outfile=wav_dir+detection.template_name+' ' +
#             str(detection.detect_time)+'.pdf')
