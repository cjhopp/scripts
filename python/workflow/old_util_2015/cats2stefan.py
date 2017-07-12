#!/usr/bin/env python

"""Writing catalog ev_ids to text for stefan"""

import sys
sys.path.insert(0, '/home/chet/EQcorrscan')
from obspy import read_events
import csv
from glob import glob

# cats = glob('/media/chet/hdd/seismic/NZ/catalogs/qml/space_groups/*')
cats = glob('/media/chet/hdd/seismic/NZ/catalogs/qml/corr_groups/4_sec_temps/*')
for a_cat in cats:
    csv_file = a_cat.rstrip('.xml') + '.csv'
    cat = read_events(a_cat)
    with open(csv_file, 'wb') as cat_file:
        cat_writer = csv.writer(cat_file)
        for ev in cat:
            cat_writer.writerow([str(ev.resource_id).split('/')[-1]])
