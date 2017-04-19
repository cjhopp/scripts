#!/usr/bin/env python

"""
Take pre-existing dist_mat and run last portion of eqcorrscan.utils.clusetering
on the matrix
"""
import sys
# Add EQcorrscan dev branch if needed
sys.path.insert(0, '/home/chet/EQcorrscan')

import numpy
import csv
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.cluster.hierarchy import inconsistent, maxinconsts
import matplotlib.pyplot as plt
from glob import glob
from obspy import read, read_events, Catalog
from eqcorrscan.utils.plotting import pretty_template_plot
from obspy.core.event import ResourceIdentifier

# Set correlation threshold
corr_thresh = 0.40
inconsis_thresh = 1.0

# Create template dict usually fed to clustering.cluster()
temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/events_raw/*'
temp_files = glob(temp_dir)

template_dict = {}
for filename in temp_files:
    uri_name = 'smi:org.gfz-potsdam.de/geofon/' +\
               filename.split('/')[-1].split('_')[-1].rstrip('.mseed')
    uri = ResourceIdentifier(uri_name)
    template_dict[uri] = read(filename)


# Read distance matrix from saved .npy file
dist_mat = numpy.load('/media/chet/hdd/seismic/NZ/clustering/dist_mat.npy')
# Flatten bottom half of distance matrix
dist_vec = squareform(dist_mat)

"""
At this point, we need have a distance matrix for the entire dataset, but we
need a robust way of clustering them which is actually representative of the
data. There are a number of params which can be changed and which greatly
affect how well the clustering will do this:

1. Linkage: the linking method (single, complete, average, weighted) affects
the cophenetic correlation distance. This is a measure of how well the linking
between two events reflects the actual distance between those same events in
the raw distance matrix.

2. Clustering cutoff: Can be done on the basis of cophenetic distance or on the
basis of inconsistency coefficient. This measures the ratio between the height
of a link and the average height of its direct children links. A group with a
low inconsistency is a group where the parent and children link heights are
similar. It can also be said that this group reflects real similarity between
children. The depth of the inconsistency cuttoff tells us how many levels for
which the cutoff applies for each group.
"""

# Then compute the linkages (average represents the MRP dataset best)
Z = linkage(dist_vec, method='average')

# Compute cophenetic correlation distance between Z and flat dist_mat
[c, d] = cophenet(Z, Y=dist_vec)

# Compute the inconsistency matrix for non-singleton cluster (d=2)
R = inconsistent(Z)
# Now compute the maximum inconsistency coefficient per Cluster
MI = maxinconsts(Z, R)

# Cluster the events based on inconsistency threshold of 1.2
indices = fcluster(Z, t=1.0, criterion='inconsistent')

# Visualize the distribution of correlation values
samp_inds = numpy.random.random_integers(0, len(dist_vec), 10000)
samp_corrs = []
for ind in samp_inds:
    samp_corrs.append(dist_vec[ind])

# Plot the dendrogram...if it's not way too huge
dendrogram(Z, color_threshold=1 - corr_thresh,
           distance_sort='ascending')
plt.show()

group_ids = list(set(indices))
indices = [(indices[i], i) for i in xrange(len(indices))]

# Eliminate groups smaller than 8 events (arbitrary)
real_groups = [g for g in groups if len(g) > 7]
group_lengths = [len(g) for g in real_groups]
# print('At corr_thresh: ' + str(corr_thresh))
print('Total number of groups: %d' % len(real_groups))
print('Total number of events: %d' % sum(group_lengths))

"""
Now visualizing the groups we've created
"""

# Read in catalog
cat = read_events('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/' +
                  'rotnga/final_cat/bbox_final_QML.xml')

# Seperate out the streams from the id's for iteration
big_group_ids = [bgi[1] for bgi in groups]
bob = dict(zip([bgi[1] for bgi in groups], [bgi[0] for bgi in groups]))

for group in real_groups:
    big_group_ids.append(list(zip(*group)[1]))
    big_group_streams.append(list(zip(*group)[0]))
for i, group_ids in enumerate(big_group_ids):
    file_names = '/home/chet/data/mrp_data/catalogs/2015/final/clustering/' +\
        'spatial/thresh_' + str(corr_thresh) + '_group_' + str(i)
    temp_cat = Catalog()
    with open(file_names + '.csv', 'wb') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for event in cat:
            ev_name = str(event.resource_id).split('/')[-1:][0]
            if ev_name in group_ids:
                x = str(event.preferred_origin().longitude)
                y = str(event.preferred_origin().latitude)
                z = str(event.preferred_origin().depth)
                csvwriter.writerow([x, y, z])
                temp_cat.append(event)
    temp_cat.write(file_names + '.shp', format="SHAPEFILE")

# If clutering using space_cluster(), much simpler for shapefile writing
for i, group_cat in enumerate(non_sing_cats):
    file_names = '/home/chet/data/mrp_data/catalogs/2015/final/qml/' +\
        'spatial_clusts/thresh_%.02f_group_%03d' % (d_thresh, i)
    group_cat.write(file_names + '.xml', format="QUAKEML")

# Below we'll plot picks over templates
for event in rand_cat:
    ev_id = str(event.resource_id).split('/').pop()
    fig_name = '/home/chet/figures/rand_cat_samp/filtered/filt_' +\
               ev_id + '_pick_fig.png'
    stream = template_list[ev_id]
    pretty_template_plot(stream, save=fig_name, picks=event.picks)
    plt.close()
