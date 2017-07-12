#!/usr/bin/python

from glob import glob
import os
import fnmatch

raw_files = []
raw_dir = '/home/chet/data/mrp_data/sherburn_catalog/sc3_event_xml'
for root, dirnames, filenames in os.walk(raw_dir):
    for filename in fnmatch.filter(filenames, '*.xml.zip'):
        raw_files.append(os.path.join(root, filename))

os.chdir('/home/chet/seiscomp3/lib/')
for afile in raw_files:
    name = afile[:-4]
    cmd_str = '/home/chet/seiscomp3/bin/sczip -d ' + afile + ' -o '+name
    os.system(cmd_str)
