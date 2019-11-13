#!/usr/bin/Python

"""
Script for plotting detections for family stacks over time
"""

import matplotlib.pyplot as plt
import csv
from obspy import UTCDateTime
import datetime as dt
times = []
fam_name = []
det_dict = {}
#Read in detection file
with open('/home/chet/data/PAN_output/detections/2012_170-179_detections.csv', 'r') as f1:
    detect_read = csv.reader(f1)
    temps, det_times, c, d, e = zip(*detect_read)
#Extract family number from column 1 string
for temp in temps:
    fam_name.append(temp)
#Convert times to Obspy UTCDatetime objects
times = [dt.datetime.strptime(time,' %Y-%m-%dT%H:%M:%S.%fZ') for time in det_times]
for i in range(len(times)):
    if not fam_name[i] in det_dict:
        det_dict[fam_name[i]] = [times[i]]
    else:
        det_dict[fam_name[i]].append(times[i])
#Make list of key names, sort it, then plot y value as index of name
families = []
for key in det_dict:
    families.append(key)
families.sort()
counter = 0
for index in range(len(families)):
    times = det_dict[families[index]]
    plt.plot(times, [counter]*len(times), 'ro--')
    counter+=1
plt.yticks(range(len(families)), families)
plt.ylim(plt.ylim()[0]-1, plt.ylim()[1]+1)
plt.show()
