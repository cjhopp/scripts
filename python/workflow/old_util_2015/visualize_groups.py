#!/usr/bin/env python

"""
Take groups from 3_hierarchy.py and map them

Should look to load catalog, then extract groups and write them to a shapefile?
"""

import cPickle
import csv
from obspy import read_events, Catalog
from eqcorrscan.util.plotting import pretty_template_plot
from obspy.core.event import ResourceIdentifier

# Read groups from pickle
with open('/media/chet/hdd/seismic/NZ/clustering/groups_w_ids.p', "rb") as file:
    groups = cPickle.load(file)
# Read in catalog
cat = read_events('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/' +
                  'rotnga/final_cat/bbox_final_QML.xml')

big_group_ids = []
big_group_streams = []
# Extract just the
for group in groups:
    if len(group) > 7:
        big_group_ids.append(list(zip(*group)[1]))
        big_group_streams.append(list(zip(*group)[0]))
for i, group_ids in enumerate(big_group_ids):
    file_names = '/home/chet/data/mrp_data/catalogs/2015/final/thresh_' +\
    str(corr_thresh) + '_group_' + str(i)
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

# Below we'll plot picks over templates for given indices
ev_id = '2015sora495962'
res_id = ResourceIdentifier('smi:org.gfz-potsdam.de/geofon/2015sora495962')
for event in cat:
    if event.resource_id == res_id:
        test_ev = event
for i, group_id in enumerate(big_group_ids):
    if group_id == ev_id:
        pretty_template_plot(big_group_streams[i], picks=test_ev.picks)
