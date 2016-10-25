#!/usr/bin/python
from glob import glob
import os
import fnmatch

raw_files = []
raw_dir = '/home/chet/data/mrp_data/sherburn_catalog/sc3_event_xml/rotnga'
for root, dirnames, filenames in os.walk(raw_dir):
    for filename in fnmatch.filter(filenames, '*.xml'):
        raw_files.append(os.path.join(root, filename))
#For running sczip from SC3
os.chdir('/home/chet/seiscomp3/lib/')
for afile in raw_files:
    name = afile[:-4]
    cmd_str = '/home/chet/seiscomp3/bin/sczip -d ' + afile + ' -o '+name
    os.system(cmd_str)
#Next step: convert sc3ml to QuakeML
for afile in raw_files:
    #Put new files in separate directory
    new_name = '/home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotnga/' + \
        os.path.basename(afile)[:-4] + '_QML.xml'
    cmd_str2 = 'xsltproc -o ' + new_name + \
        ' ~/data/sc3ml_quakeml_xslt/sc3ml_0.7__quakeml_1.2.xsl ' + afile
    os.system(cmd_str2)
#Remove all '#' from QuakeML (shady way of circumventing validation issues)
qml_files = glob('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/*QML.xml')
for one_file in qml_files:
    command = "sed -i 's/#//g' " + one_file
    os.system(command)
