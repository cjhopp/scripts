# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:03:59 2015

@author: matsonga
"""
import datetime
import matplotlib.pyplot as plt
import pyproj
import math
from obspy import UTCDateTime

wgs84=pyproj.Proj("+init=EPSG:4326")
nztm=pyproj.Proj("+init=EPSG:2135")
#pyproj.transform(wgs84,nztm,176.177214,-38.541174)

#file = '/Users/home/matsonga/seisan/REA/NM76C/hypoDD7/hypoDD.reloc'
file = '/Users/home/matsonga/seisan/REA/TEMP6/hypoDD19/hypoDD.reloc'
# file = '/Users/home/matsonga/seisan/REA/NM76G/hypoDD1/hypoDD2.csv'
# file = '/Users/home/matsonga/seisan/REA/TEMP6/hypoDD18/hypoDD.reloc'
templateFile = '/Users/home/matsonga/seisan/REA/NM76D/templateLocations.csv'
templates = []
with open(templateFile,'r') as tempfid:
    templateContents = tempfid.readlines()
for line in templateContents:
    template = line.split(',')[0]
    templates.append(UTCDateTime(template).datetime)
y = [15] * len(templates)
with open(file,'r') as fid:
    contents = fid.readlines()
e1 = -38.5329
n1=176.1782
e0,n0 = pyproj.transform(wgs84,nztm,n1,e1)
# e0 = 1877051.7416363815 #2787148.273 #
# n0 = 5730277.176951178 #6291842.492 #5730277.176951178
# d0 = 1975.477055
# e0=2787147.937
# n0=6291840.728
datetimeList = []
distanceList = []
for line in contents:
    year=line.split()[10]
    month = '%02i' %int(line.split()[11])
    day = '%02i' %int(line.split()[12])
    hour= '%02i' %int(line.split()[13])
    minute = '%02i' %int(line.split()[14])
    second = '%2.3f' %float(line.split()[15])
    datetime1=datetime.datetime.strptime('%s/%s/%s %s:%s:%s'%(year,month,day,hour,minute,second),'%Y/%m/%d %H:%M:%S.%f')
    datetimeList.append(datetime1)
    lat = float(line.split()[1])
    lon = float(line.split()[2])
    # north = float(line.split()[1])
    # east = float(line.split()[2])
    depth = float(line.split()[3])*1000
    east,north = pyproj.transform(wgs84,nztm,lon,lat)
    # distance = math.sqrt((east-e0)**2+(north-n0)**2+(depth-d0)**2)
    distance = math.sqrt((east-e0)**2+(north-n0)**2)
    distanceList.append(distance)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(datetimeList,distanceList,'o')
fig.autofmt_xdate()
# ax2=ax.twinx()
# ax.set_xlim([datetime.datetime(2012,6,17),datetime.datetime(2012,6,27)])
# ax.set_ylim(0,1000)
ax.plot(templates,y,'rv')
plt.show()

# file = 'NM8stimulationDetectionsEastNorth.txt'
#
# for line in contents:
#     east = line.split()[3]
#     north = line.split()[4]
#     depth = line.split()[5]
#     datetime1 = line.split()[6]
#     east = float(east)
#     north = float(north)
#     depth = float(depth)
#     datetime1 = UTCDateTime(datetime1)
#     distance = math.sqrt((east-e0)**2+(north-n0)**2+(depth-d0)**2)
#     distanceList.append(distance)
#     datetimeList.append(datetime1)
# datetimeList = []
# distanceList = []
# for line in contents:
#     east = line.split()[3]
#     north = line.split()[4]
#     depth = line.split()[5]
#     datetime1 = line.split()[6]
#     east = float(east)
#     north = float(north)
#     depth = float(depth)*(-1)
#     datetime1 = UTCDateTime(datetime1).datetime
#     distance = math.sqrt((east-e0)**2+(north-n0)**2+(depth-d0)**2)
#     distanceList.append(distance)
#     datetimeList.append(datetime1)
#
# startdate = datetime.date(year=2012,month=06,day=01)
# numdays =61
# dateList = []
# for x in range(0,numdays):
#     dateList.append(startdate +datetime.timedelta(days=x))
#
# # for date in dateList:
#
#
# minTime = min(datetimeList)
# datetimeList = [x-minTime for x in datetimeList]




# database = 'TEMP4'
# trial = '1'
# indir = '/Users/home/matsonga/seisan/REA/%s/hypoDD%s' %(database,trial)
# filename = '%s/hypoDD_withDistance.txt' %indir
# outfilename = '%s/hypoDDJulDay.reloc' %indir
# fid = open(filename, 'r')
# #outfid = open(outfilename, 'w')
# datetimeList = []
# distanceList = []
# for line in fid:
#     data = line.split()
#     eqID = data[0]
#     lat = data[1]
#     lon = data[2]
#     east = data[3]
#     north = data[4]
#     distance = data[6]
#     distanceList.append(distance)
#     depth = data[3]
#     year = data[7]
#     month = data[8]
#     day = data[9]
#     hr = data[10]
#     mn = data[11]
#     sc = data[12]
#     if '.' in sc:
#         datetime1=datetime.datetime.strptime('%s/%s/%s %s:%s:%s'%(year,month,day,hr,mn,sc),'%Y/%m/%d %H:%M:%S.%f')
#     else:
#         datetime1=datetime.datetime.strptime('%s/%s/%s %s:%s:%s'%(year,month,day,hr,mn,sc),'%Y/%m/%d %H:%M:%S')
# #    datetime1=datetime.datetime.strptime('%s/%s/%s %s:%s:%s'%(year,month,day,hr,mn,sc),'%Y/%m/%d %H:%M:%S.%f')
#     datetimeList.append(datetime1)
# #    timetuple1 = datetime1.timetuple()
# #    julday = timetuple1.tm_yday
# #    print ('%s %s %s %s %s %s' %(eqID,lat,lon,depth,year,julday))
# #    outfid.write('%s %s %s %s %s %s \n' %(eqID,lat,lon,depth,year,julday))
# fid.close()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(datetimeList,distanceList,'o')
# fig.autofmt_xdate()
# plt.show()
# #outfid.close()
