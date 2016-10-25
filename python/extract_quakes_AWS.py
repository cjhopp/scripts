#!/usr/bin/python

"""
Take CSV extracted from GeoNet quick search and
extract public ids (i.e. yyyyp######) to grab
single quakeML documents
"""
#Using idea from code from Nico Fournier at GNS (Wairakei?)

import csv
import subprocess

with open('/home/chet/xml_validation/misc/quakes.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        cmd_str = 'aws s3 cp s3://seiscompml07/'+row[0]+'.xml'+' s3://cjhcatalog'
        subprocess.call(cmd_str, shell=True)
