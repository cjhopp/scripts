#!/usr/bin/env python

"""
Script intended to filter a catalog of data based on pre-set parameters.
I think it will return a list of event id's or similar
"""

from glob import glob
from obspy import read_events, Catalog

#Make list of catalog parts
cat_list = glob('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/' +
                'rotnga/final_cat/*part*')

#Search through all events in catalog and output list of names
new_cat = Catalog()
for catalog in cat_list:
    #Read in catalog
    cat = read_events(catalog)
    for event in cat:
        lat = event.preferred_origin().latitude
        lon = event.preferred_origin().longitude
        if lat > -38.661 and lat < -38.483 and lon > 176.094 and lon < 176.296:
            new_cat.append(event)
        else:
            print('Event outside bounding box...')

#Write catalog to various formats
#VELEST format
new_cat.write('/home/chet/data/mrp_data/catalogs/2015/final/cnv/rotnga2015.cnv',
              format="CNV")
#Shapefile
new_cat.write('/home/chet/data/mrp_data/catalogs/2015/final/shp/rotnga2015',
              format="SHAPEFILE")
#Loop to write single event NLLOC files
for event in new_cat:
    ev_name = str(event.resource_id).split('/')[2]
    event.write('/home/chet/data/mrp_data/catalogs/2015/final/nlloc/' +
                ev_name + '.nll', format="NLLOC_OBS")
#Loop to write to x, y, z text
import csv
with open('/home/chet/data/mrp_data/catalogs/2015/final/xyz/' +
          'rotnga_2015_wgs84.csv', 'wb') as f:
    csvwriter = csv.writer(f, delimiter=',')
    for event in new_cat:
        x = str(event.preferred_origin().longitude)
        y = str(event.preferred_origin().latitude)
        z = str(event.preferred_origin().depth)
        csvwriter.writerow([x, y, z])
