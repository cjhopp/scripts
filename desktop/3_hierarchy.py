#!/usr/bin/python

"""
Cluster events in dataset by hierarchical clustering via
EQcorrscan clustering.cluster()

Templates have already been processed:
prepick = 0.5 sec
length = 4 sec
Passband: 1 - 20 Hz
Sample rate = 100 Hz
"""

from eqcorrscan.utils import clustering
from glob import glob
from obspy import read
import numpy as np

temp_dir = '/media/rotnga_data/templates/2015_dayproc/*'
temp_files = glob(temp_dir)
temp_files.sort()

template_list = []
files_wo_data = []
for filename in temp_files:
    try:
        template_list.append(read(filename))
    except TypeError:
        print('No actual data in this file')
        files_wo_data.append(filename)
#Run hierarchical clustering function
groups = clustering.cluster(template_list, show=False, corr_thresh=0.28,
                            save_corrmat=True, debug=2)

"""
Now compute the SVD (or empirical approximation) for each family
of MORE THAN ONE event
Use SVD() or empirical_SVD()
"""
#First, empirical_SVD
first_subspace = []
second_subspace = []
for group in groups:
    if len(group) > 1:
        [first, second] = clustering.empirical_SVD(group)
        #Account for np.diff() returning array with len one less than original
        for tr in second:
            tr.data = np.concatenate(([0.0], tr.data))
        first_subspace.append(first)
        second_subspace.append(second)

#Write out first and second subspace empirical "SVD" templates for each family
out_dir = '/home/chet/data/templates/master_temps/hierarchy_cluster/no_delays/'
for i in range(len(first_subspace)):
    if i < 10:
        first_subspace[i].write(out_dir+'f'+'0'+str(i)+'_time_stack.ms',
                                format='MSEED')
        second_subspace[i].write(out_dir+'f'+'0'+str(i)+'_stack_derivative.ms',
                                 format='MSEED')
    else:
        first_subspace[i].write(out_dir+'f'+str(i)+'_time_stack.ms',
                                format='MSEED')
        second_subspace[i].write(out_dir+'f'+str(i)+'_stack_derivative.ms',
                                 format='MSEED')

#Now for the straight SVD using SVD() and SVD_2_stream()
##MAKE SURE TIMING OF THESE TEMPLATES IS OK!!! Detections look bizzarre
grp_cnt = 0
for group in groups:
    #Do SVD
    if len(group) > 1:
        [SVectors, SValues, Uvectors, stachans] = clustering.SVD(group)
        #Convert first and second SVD to a stream
        SVstreams = clustering.SVD_2_stream(SVectors, stachans, 2, 100.0)
        if grp_cnt < 10:
            SVstreams[0].write(out_dir+'f'+'0'+str(grp_cnt)+'_SVD1.ms',
                               format='MSEED')
            SVstreams[1].write(out_dir+'f'+'0'+str(grp_cnt)+'_SVD2.ms',
                               format='MSEED')
        else:
            SVstreams[0].write(out_dir+'f'+str(grp_cnt)+'_SVD1.ms',
                               format='MSEED')
            SVstreams[1].write(out_dir+'f'+str(grp_cnt)+'_SVD2.ms',
                               format='MSEED')
        grp_cnt += 1
