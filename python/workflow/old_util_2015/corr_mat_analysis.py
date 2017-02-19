#!/usr/bin/env python

"""
Small script to extract statistics from correlation or distance matrices\
for the purposes of establishing grouping thresholds, etc...
"""

from glob import glob
import numpy as np
from scipy.spatial.distance import squareform

corr_mats = glob('/media/chet/hdd/seismic/NZ/catalogs/corr_mats/*')
mat_stats = {}
for corr_mat in corr_mats:
    mat_name = corr_mat.split('/')[-1]
    mat = np.load(corr_mat)
    mat_stats[mat_name] = {}
    mat_stats[mat_name]['avg_dist'] = np.mean(squareform(mat))
    mat_stats[mat_name]['std_dist'] = np.std(mat)
    mat_stats[mat_name]['dist_thresh'] = np.mean(squareform(mat)) - np.std(mat)
