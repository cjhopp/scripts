#!/usr/bin/env python

"""
The first phase of clustering will be based upon inter-event distance.
Given the unsatisfactory quality of phase picks, they will be corrected
within these distance clustered groups. Finally, clustering based on
waveform cross correlation will be applied within these groups.
"""

import sys
sys.path.insert(0, '/home/chet/EQcorrscan')

from eqcorrscan.utils import clustering, plotting
from obspy import read_events

# Read in whole damn catalog
cat = read_events('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/' +
                  'rotnga/final_cat/bbox_final_QML.xml')

# Cluster events by distance
groups = clustering.space_cluster(cat, d_thresh=2.0)

# Eliminate groups with size below a certain threshold
real_groups = [g for g in refined_groups if len(g) > 1]
group_lengths = [len(g) for g in real_groups]
# print('At corr_thresh: ' + str(corr_thresh))
print('Total number of groups: %d' % len(real_groups))
print('Total number of events: %d' % sum(group_lengths))

# Here we can write the groups (which are Catalogs) to qml/shapefile/xyz
for i, group_cat in enumerate(real_groups):
    file_names = '/media/chet/hdd/seismic/NZ/catalogs/qml/space_groups/' +\
        'nlloc_thresh_%.02f_group_%03d' % (d_thresh, i)
    # Write shapefile first
    group_cat.write(file_names + '.shp', format="SHAPEFILE")
    # Now qml
    group_cat.write(file_names + '.xml', format="QUAKEML")
    # Now write xyz to csv for 3D viz
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
