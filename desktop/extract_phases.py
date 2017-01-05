#!/usr/bin/python

"""
Read in phases from Gabe's dataset and
rewrite them in order:
TIME STA P-ARR S-ARR
"""
from obspy.core import UTCDateTime
import subprocess

path1 = "/Users/home/hoppche/data/templatePhaseTimes.txt"
path2 = "/Users/home/hoppche/data/test_pha.txt"

pha_info = []
#Read in each field in each line as UTCDateTime object, store in pha_info
subprocess.call(["rm", path2])
fid2 = open(path2,'w')
with open(path1) as fid:
    for line in fid:
        s = line.split(" ")
        time1 = UTCDateTime(s[2])
        time2 = s[3]
        time2 = time2.split("\\")
        time2 = UTCDateTime(time2[0])
        mintime = min(time1,time2)
        maxtime = max(time1,time2)
        writeline = '%s %s %s %s\n' %(s[0], s[1], mintime, maxtime)
        #print writeline
        fid2.write(writeline)

fid2.close()
