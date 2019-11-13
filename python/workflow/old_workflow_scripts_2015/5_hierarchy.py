#!/usr/bin/python

"""
Cluster events in dataset by hierarchical clustering via
EQcorrscan clustering.cluster()

Templates have already been processed:
prepick = 0.5 sec
length = 1 sec
Passband: 1 - 20 Hz
Sample rate = 100 Hz
"""
import sys
sys.path.insert(0, '/home/chet/EQcorrscan')

from eqcorrscan.utils import clustering
from glob import glob
from obspy import read, read_events, Catalog
import numpy as np
from obspy.core.event import ResourceIdentifier
from multiprocessing import cpu_count
# Create the template dictionary
# temp_dir = '/media/chet/rotnga_data/templates/2015/*'

# temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/1_sec_5-2/*'
temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/dayproc_4-27/*'
temp_files = glob(temp_dir)

# Template dictionary keyed to event resource_id
template_dict = {}
for filename in temp_files:
    uri_name = 'smi:org.gfz-potsdam.de/geofon/' +\
               filename.split('/')[-1].split('_')[0].rstrip('.mseed')
    uri = ResourceIdentifier(uri_name)
    template_dict[uri] = read(filename)

# Read in refined catalog and group in space once more
space_cats = glob('/media/chet/hdd/seismic/NZ/catalogs/qml/space_groups/*')
# group_cats = glob('/media/chet/hdd/seismic/NZ/catalogs/qml/corr_groups/4_sec_temps/spacegrp_063_corrgrp_018*')

# Cluster each group by waveform cross-correlation and write to new files
for space_cat in space_cats:
    cat = read_events(space_cat)
    if len(cat) <= 2:
        continue
    elif len(cat) < cpu_count():
        cores = len(cat)
    elif len(cat) >= cpu_count():
        cores = 'all'
    grp_num = space_cat.split('/')[-1].split('_')[-1].rstrip('.xml')
    template_list = [(template_dict[ev.resource_id], ev.resource_id)
                     for ev in cat]
    plt_name = '/media/chet/hdd/seismic/NZ/catalogs/corr_figs/1_sec_temps/' +\
               'spacegrp_%s_dend_0.20.png' % grp_num
    corr_mat = '/media/chet/hdd/seismic/NZ/catalogs/corr_figs/1_sec_temps/' +\
               'spacegrp_%s_mat.npy' % grp_num
    groups = clustering.cluster(template_list, corr_thresh=0.30, allow_shift=True,
                                shift_len=25, save_corrmat=True, cores=cores, debug=2)
    for i, grp in enumerate(groups):
        corrgrp_cat = Catalog()
        f_name_root = '/media/chet/hdd/seismic/NZ/catalogs/'
        f_name = 'spacegrp_%s_corrgrp_%03d' % (grp_num, i)
        for e in cat:
            for temp_st in grp:
                if e.resource_id == temp_st[1]:
                    corrgrp_cat.append(e)
        corrgrp_cat.write(f_name_root + 'qml/corr_groups/1_sec_temps/'
                          + f_name + '.xml', format="QUAKEML")
        corrgrp_cat.write(f_name_root + 'shp/corr_groups/1_sec_temps/'
                          + f_name + '.shp', format="SHAPEFILE")

# Also trying correlation cluster for whole catalog
cat = read_events('/media/chet/hdd/seismic/NZ/catalogs/qml/2015_nlloc_final_run02_group_refined.xml')
template_list = [(template_dict[ev.resource_id], ev.resource_id)
                 for ev in cat]
plt_name = '/media/chet/hdd/seismic/NZ/catalogs/corr_figs/4_sec_temps/' +\
           'entire_cat_cluster_dend_shift25.png'
corr_mat = '/media/chet/hdd/seismic/NZ/catalogs/corr_figs/1_sec_temps/' +\
           'entire_cat_mat_shift25.npy'
groups = clustering.cluster(template_list, show=True, corr_thresh=0.30,
                            allow_shift=True, shift_len=25,
                            save_corrmat=True, cores='all', debug=2)
# If dist_mat already saved
groups = clustering.cluster(template_list, plot='dend', corr_thresh=0.44,
                            save_plot=plt_name, cores='all', debug=2,
                            corr_mat=corr_mat)
for i, grp in enumerate(groups):
    corrgrp_cat = Catalog()
    f_name_root = '/media/chet/hdd/seismic/NZ/catalogs/'
    f_name = 'entire_cat_shift_corrgrp_%03d_0.40' % i
    for e in cat:
        for temp_st in grp:
            if e.resource_id == temp_st[1]:
                corrgrp_cat.append(e)
    corrgrp_cat.write(f_name_root + 'qml/corr_groups/4_sec_temps/'
                      + f_name + '.xml', format="QUAKEML")
    corrgrp_cat.write(f_name_root + 'shp/corr_groups/4_sec_temps/'
                      + f_name + '.shp', format="SHAPEFILE")
