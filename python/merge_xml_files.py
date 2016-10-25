#!/usr/bin/python

"""
Merge all catalog xml files into obspy Catalog object
then write them to single xml to see if seishub will accept it
"""

from glob import glob
from obspy.core.event import readEvents, Catalog

xml_files = glob('/home/chet/data/GeoNet_catalog/20??p??????.xml')
cat = Catalog()
for file1 in xml_files:
    cat = cat + readEvents(file1)
