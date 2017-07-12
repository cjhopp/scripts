#!/usr/bin/python

"""
Main script to organize and prepare data followed by
running EQcorrscan match_filter to generate detections
"""
import sys
sys.path.append("/projects/nesi00228/EQcorrscan")

from obspy import read, Stream, UTCDateTime
from eqcorrscan.core import template_gen, match_filter
from eqcorrscan.par import match_filter_par as matchdef
from eqcorrscan.utils import pre_processing, Sfile_util
import matplotlib.pyplot as plt
import fnmatch
import copy
from glob import glob
import os
import time

#Time this script
script_start = time.time()

#Take input arguments --split and --instance from bash which specify slices of days to run
Split = False
instance = False

#Be sure to only give --instance and --splits, otherwise following is not valid
args = sys.argv[1:len(sys.argv)]
if args[0] == '--instance' or args[1] == '--instance' or args[2] == '--instance':
    # Arguments to allow the code to be run in multiple instances
    Split = True
    Test = False
    Prep = False
    args = sys.argv[1:len(sys.argv)]
    for i in xrange(len(args)):
        if args[i] == '--instance':
            instance = int(args[i+1])
            print 'I will run this for instance '+str(instance)
        elif args[i] == '--splits':
            splits = int(args[i+1])
            print 'I will divide the days into '+str(splits)+' chunks'

#Read in templates made via clustering.empirical_SVD() or SVD()
#temp_dir = '/home/chet/data/templates/master_temps/hierarchy_cluster/no_delays/empirical_SVD/'
temp_dir = '/projects/nesi00228/data/templates/empirical_SVD/'
os.chdir(temp_dir)
ms_files = glob('*.ms')
ms_files.sort()
template_names = []
for file1 in ms_files:
    if not 'templates' in locals():
        templates = [read(file1)]
        template_names.append(file1[:-3])
    else:
        templates += [read(file1)]
        template_names.append(file1[:-3])
# print templates, template_names
# Extract the station info from the templates
for template in templates:
    #Filter and downsample sample data
    template = pre_processing.shortproc(template, 1.0, 20.0, 3, 100.0,
                                        debug=1)
    if not 'stachans' in locals():
        stachans = [(tr.stats.station, tr.stats.channel) for tr in template]
    else:
        stachans += [(tr.stats.station, tr.stats.channel) for tr in template]

# Make this a unique list
stachans = list(set(stachans))
# print stachans
# Read in the continuous data for these station, channel combinations
start_day = UTCDateTime(2012, 06, 18)
end_day = UTCDateTime(2012, 06, 27)
dates = range(start_day.julday, end_day.julday+1)
ndays = len(dates)

if Split:
    if instance == splits-1:
        ndays = ndays-(ndays/splits)*(splits-1)
        dates = dates[-ndays:]
    else:
        ndays = ndays/splits
        dates = dates[ndays*instance:(ndays*instance)+ndays]
    print 'This instance will run for '+str(ndays)+' days'
    print 'This instance will run from '+str(min(dates))
else:
    dates = dates

# Read in the continuous data for these station, channel combinations
raw_dir = '/projects/nesi00228/data/miniseed/'
#raw_dir='/home/chet/data/test_mseed/'
f = open('/projects/nesi00228/data/'+str(start_day.year)+'_'+str(min(dates)) +
         '-'+str(max(dates))+'_detections.csv', 'w+')

for day in dates:
    #Delete stream and list of raw_files after day loop
    if 'st' in locals():
        del st
    if 'raw_files' in locals():
        del raw_files
    raw_files = []
    for root, dirnames, filenames in os.walk(raw_dir):
        for stachan in stachans:
            for filename in fnmatch.filter(filenames, 'NZ.'+stachan[0] +
                                           '.10.EH'+stachan[1][-1]+'.D.2012.' +
                                           str(day)):
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
        tr = pre_processing.dayproc(tr, 1.0, 20.0, 3, 100.0,\
                                    1, day)

    #Set directory for match filter output plots
    plot_dir = '/projects/nesi00228/data/plots/'
    # Compute detections
    detections = match_filter.match_filter(template_names, templates, st,
                                           8.0, 'MAD', 6.0, False, plot_dir)

    # We now have a list of detections! We can output these to a file to check later
    for detection in detections:
        f.write(detection.template_name+', '+str(detection.detect_time) +
                ', '+str(detection.detect_val)+', '+str(detection.threshold) +
                ', '+str(detection.no_chans)+'\n')
    del detections
f.close()

#Print out runtime
print 'Script took ', time.time() - start, ' seconds.'

    ##Instead of saving all of these waveforms, just save the plots as pdf
    # wav_dir='/home/chet/data/detections/'
    # det_wav = Stream()
    # for detection in detections:
    #     st.plot(starttime=detection.detect_time-2, endtime=detection.detect_time+8, \
    #                 outfile=wav_dir+detection.template_name+' '+\
    #                 str(detection.detect_time)+'.pdf')
