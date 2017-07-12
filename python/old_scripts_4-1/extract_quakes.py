#!/usr/bin/python

"""
Take CSV extracted from GeoNet quick search and
extract public ids (i.e. yyyyp######) to grab
single quakeML documents
"""
#Using idea from code from Nico Fournier at GNS (Wairakei?)
# set URL and URLFILTER
urlroot = "http://quakeml.geonet.org.nz/quakeml/1.2/"

import csv
import urllib

with open('/home/chet/xml_validation/misc/quakes.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        if 'f' in locals():
            del f
        temp_url = urlroot + str(row[0])
        # RETRIEVE ONLINE QUERY AND SAVE LOCALLY
        f = urllib.URLopener()
        f.retrieve(temp_url, '/home/chet/data/GeoNet_catalog/'+str(row[0])+'.xml')
        #Maybe read these into obspy catalog object and save single file??
