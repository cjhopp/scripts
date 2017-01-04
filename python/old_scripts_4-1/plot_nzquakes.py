#!/usr/bin/env python
# PLOT_NZQUAKES.PY
#
# Plot daily rate of seismcity in area of interest for given period
# Usage:
#	plot_nzquakes.py <START DATE/TIME> <END DATE/TIME> <MINLLON/MINLAT/MAXLON/MAXLAT/MINDEPTH/MAXDEPTH/MINMAG/MAXMAG>
#
#
# 	e.g., plot_nzquakes.py 2012-01-01T00:00:00  2012-12-31T11:00:00  '175.61 -39.158 175.7737 -39.053 0 30 0 5'
#
# NOTE:
#		from plot_dailyseismicity.py (NF, July 2012)
# 		adjusted to retrieve GeoNet earthquake catalogue from SeisComp3 instead of CUSP
#		this version does not plot error bars on depth (not available from sc3 catalogue at the time of writing)
#		
#		Heavily based on plot_dailyseismicity_sc3.py (NF)
#
# Nico Fournier, GNS Science, Taupo, New Zealand
# email: n.fournier@gns.cri.nz
# December 2012
#
# V2 is adapted for the newest version of pandas
# September 2013
#
# changed conversion to ordinal from:
#   t = date2num(dstrate)
# to
#   t = date2num(dstrate.to_pydatetime())
# December 2014
#
# cleaned up and changed tmp files location (now under tmp subdir) 
# April 2015


########################################
# import modules
########################################
import matplotlib.mlab
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
import numpy as np
import string
import datetime as dt
from subprocess import call
import scipy.signal
import sys
import pyproj
import os
from subprocess import call
from pandas import *

# added for sc3 catalogue
import urllib
import dateutil.parser

# maps
from matplotlib.colors import LightSource
import netCDF4 as ncdf
from mpl_toolkits.basemap import Basemap, cm

################################################################################
# SUBFUNCTIONS
################################################################################

########################################
# PRINT ERROR MESSAGE
def errmessage(hdrline):
    print "----------------------------------"
    print hdrline
    print "----------------------------------"
    print "PLOT_NZQUAKES.PY"
    print "----------------------------------"
    print "USAGE:"
    print "	plot_nzquakes.py <START DATE/TIME> <END DATE/TIME> <MINLON MINLAT MAXLON MAXLAT MINDEPTH MAXDEPTH MINMAG MAXMAG>"
    print "FORMAT:"
    print "	plot_nzquakes.py <YYYY-MM-DDTHH:MN-SS> <YYYY-MM-DDTHH:MN-SS> <'MINLON MINLAT MAXLON MAXLAT MINDEPTH MAXDEPTH MINMAG MAXMAG'>"
    print "EXAMPLE:"
    print "	run plot_nzquakes.py 2012-01-01T00:00:00  2013-01-01T00:00:00  '175.61 -39.158 175.7737 -39.053 0 30 0 5'"
    print "----------------------------------"
    print "EXAMPLES OF ZONES:"
    print "	TAUPO:		'175.6 -39.05 176.2 -38.55 0 15 0 8'"
    print "	TAUPO CALDERA:	'175.73 -38.88 176.07 -38.68 0 15 0 8'"
    print "	TNP:		'175.35 -39.45 175.80 -39.0 0 15 0 8'"
    print "	TNP WIDE:	'175.2 -39.6 176.0 -38.9 0 15 0 8'"
    print "	TONGARIRO:	'175.55 -39.2 175.75 -39.05 0 15 0 8'"
    print "	RUAPEHU:	'175.45 -39.39 175.705 -39.18 0 15 0 8'"
    print "	OKATAINA:	'176.1 -38.35 176.65 -37.97 0 15 0 8'"
    print "	TARANAKI:	'173.7 -39.6 174.45 -39.0 0 15 0 8'"
    print "	BOP:		'176.6 -38.2 178. -37.2 0 15 0 8'"

########################################

########################################
# RETRIEVE SC3 EVENTS OF INTEREST
def srchsc3(srchparam):

    #------------------------------
    # assign search parameter values to local variables
    minlon, maxlon, minlat, maxlat, mindepth, maxdepth, minmag, maxmag, startdate, enddate = srchparam

    # temporary file with output from search (not used for later input in this script, saved as conveniency only)
    locsrchfile = 'tmp/searchout.tmp'

    ########################################
    # PERFORM ONLINE QUERY
    ########################################
    #------------------------------
    # set individual filters
    # location
    bboxf = "BBOX(origin_geom," + str(minlon) + "," + str(maxlat) + "," + str(maxlon) + "," + str(minlat) + ")"
    # start and end time
    # origtf = "origintime%3E='" + startdate.isoformat() + "'"
    origtf = "origintime>='" + d1.isoformat() + "'+AND+origintime<='" + d2.isoformat() + "'"

    # magnitude
    magf = "magnitude>=" + str(minmag) + "+AND+magnitude<=" + str(maxmag)
    
    # depth
    depthf = "depth>=" + str(mindepth) + "+AND+depth<=" + str(maxdepth)

    #------------------------------
    # set URL and URLFILTER
    urlroot = "http://wfs.geonet.org.nz/geonet/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=geonet:quake_search_v1&outputFormat=csv"
    # urlroot = "http://wfs-beta.geonet.org.nz/geoserver/geonet/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=geonet:quake&outputFormat=csv"
    urlfilter = "&cql_filter=" + bboxf + "+AND+" + origtf + "+AND+" + magf + "+AND+" + depthf
    urlname = urlroot + urlfilter

    #------------------------------
    # RETRIEVE ONLINE QUERY AND SAVE LOCALLY
    # load info
    f = urllib.urlopen(urlname)
    # read info
    s = f.read()
    # write local csv file
    fid = open(locsrchfile,'w')
    fid.write(s)
    fid.close


    ########################################
    # PARSE AND PROCESS DATA
    ########################################

    # parse info
    a  = string.split(s,'\n')

    # concatenate info into single array
    dst = []		# datetime object
    lon = []		# float (degrees)
    lat = []		# float (degrees)
    depth = []	# float (km > 0)
    mag = []		# float

    for event in a[1:]:
        if event != '':
            # parse event info
            strev = string.split(event,',')
            # datetime object
            # dst.append(dateutil.parser.parse(strev[2]))
            dst.append(dateutil.parser.parse(strev[3]))
            # data - note string format change from July 2014
            # lon.append(float(strev[3]))
            # lat.append(float(strev[4]))
            # depth.append(float(strev[5]))
            # mag.append(float(strev[6]))
            lon.append(float(strev[6]))
            lat.append(float(strev[5]))
            depth.append(float(strev[7]))
            mag.append(float(strev[8]))

    # convert all data lists to arrays
    lon = np.array(lon)
    lat = np.array(lat)
    depth = np.array(depth)
    mag = np.array(mag)
    # total number of events
    numev = len(lon)

    #------------------------------
    # SAVE CATALOGUE LOCALLY ZMAP-FRIENDLY FILE (dec deg)
    fidzmap = open("searchout_zmap.txt",'w')
    for iev in np.arange(0,len(mag)):
        outstr = str(lon[iev]) + " " + str(lat[iev]) + " " + str(dst[iev].year) + " " + str(dst[iev].month) + " " + str(dst[iev].day) + " " + str(mag[iev]) + " " + str(depth[iev]) + " " + str(dst[iev].hour) + " " + str(dst[iev].minute) + "\n"
        fidzmap.write(outstr)
    fidzmap.close

    #------------------------------
    # SAVE CATALOGUE LOCALLY ZMAP-FRIENDLY FILE in NZMG
    e,n = deg2nzmg(lon,lat)
    fidzmap = open("searchout_zmap_nzmg.txt",'w')
    for iev in np.arange(0,len(mag)):
        outstr = str(e[iev]) + " " + str(n[iev]) + " " + str(dst[iev].year) + " " + str(dst[iev].month) + " " + str(dst[iev].day) + " " + str(mag[iev]) + " " + str(depth[iev]) + " " + str(dst[iev].hour) + " " + str(dst[iev].minute) + "\n"
        fidzmap.write(outstr)
    fidzmap.close

    # return data
    return numev, dst, lon, lat, depth, mag

########################################


########################################
# CONVERT DECIMAL DEGREES TO NZMG
########################################
def deg2nzmg(lon,lat):
    # define projections
    proj_wgs84 = pyproj.Proj(init="epsg:4326")
    proj_nzmg = pyproj.Proj(init="epsg:27200")

    # convert coordinates from deg to NZMG
    e,n = pyproj.transform(proj_wgs84,proj_nzmg,lon,lat)

    # return Easting and Northing in NZMG
    return e,n

########################################

# ########################################
# # COMPUTE B-VALUE
# ########################################
# def compbval():
# 	

########################################
# READ TOPO FILE (NETCDF GRID)
def readtopo(topofile,minmax):

    #------------------------------
    # Read file
    #------------------------------
    rootgrp = ncdf.Dataset(topofile, 'r', format='NETCDF4')
    xvar =  rootgrp.variables['x']
    yvar =  rootgrp.variables['y']
    zvar =  rootgrp.variables['z']
    nx = xvar.size
    ny = yvar.size
    x = xvar[:]
    y = yvar[:]

    #------------------------------
    # select area of interest
    #------------------------------
    ix = np.where((x>=minmax[0]) & (x<=minmax[1]))[0]
    iy = np.where((y>=minmax[2]) & (y<=minmax[3]))[0]
    xs = xvar[ix]
    ys = yvar[iy]
    Zs = zvar[iy,ix]

    # Xs, Ys = np.meshgrid(xs,ys)

    #------------------------------
    # BASEMAP
    #------------------------------
    # create Basemap instance with Mercator projection
    m = Basemap(projection='merc', resolution="l", llcrnrlon=minmax[0], llcrnrlat=minmax[2], urcrnrlon=minmax[1], urcrnrlat=minmax[3])
    # create grids and compute map projection coordinates for lon/lat grid
    Xs, Ys = m(*np.meshgrid(xs, ys))

    #------------------------------
    # return values
    #------------------------------
    return xs,ys,Xs,Ys,Zs,m

########################################

########################################
# SEND IMAGE TO TARAWERA (VOLCANO DEVELOPMENT PAGE)
def send2vdp(listname):
    server = 'volcano'
    usr = 'nicof'
    if listname == 'RUAP':
        remotedir = 'ruapehu'
    elif listname == 'TVZ':
        remotedir = 'tvz'
    elif listname == 'TAUPO':
        remotedir = 'taupo'
    elif listname == 'AKL':
        remotedir = 'auckland'
    elif listname == 'OKAT':
        remotedir = 'rotorua_okataina'
    elif listname == 'WI':
        remotedir = 'whiteis'
    elif listname == 'tong':
        remotedir = 'tongariro'

    # remotepath = '/opt/local/apache/htdocs/volcanoes/' + remotedir + '/eqs/'

    # tmp path (permissions issues)
    remotepath = '/opt/local/apache/htdocs/volcanoes/tvz/' + remotedir + '_seismicity/'

    cmdstr = '/usr/bin/scp '+ imagefile + ' nicof@' + server + ':' + remotepath
    call(cmdstr, shell=True)

########################################

################################################################################
# MAIN SCRIPT
################################################################################

################################################################################
# Get user argument, if any (date range for plots)
################################################################################

if len(sys.argv) < 3:
    errmessage('PLOT TIMESERIES OF QUAKE SEARCH ON NZ GEONET NETWORK')
    sys.exit(0)
else:
    sdst = str(sys.argv[1])
    edst = str(sys.argv[2])
    srchparam= str(sys.argv[3])
    if len(string.split(srchparam)) != 8:
        errmessage("ERROR: WRONG NUMBER SEARCH OF PARAMETERS")
        sys.exit(0)
    else:
        # minlon, maxlon, minlat, maxlat, mindepth, maxdepth, minmag, maxmag = string.split(srchparam)
        minlon, minlat, maxlon, maxlat, mindepth, maxdepth, minmag, maxmag = string.split(srchparam)
        try:
            d1 = dt.datetime.strptime(sdst,'%Y-%m-%dT%H:%M:%S')
        except:
            errmessage("ERROR: WRONG START DATE FORMAT")
            sys.exit(0)

        try:
            d2 = dt.datetime.strptime(edst,'%Y-%m-%dT%H:%M:%S')
        except:
            errmessage("ERROR: WRONG END DATE FORMAT")
            sys.exit(0)


########################################
# RETRIEVE SEISCOMP3 EVENTS
########################################

#------------------------------
# RUN DATA RETRIEVAL
#------------------------------
nevents, dst, lon, lat, z, mags = srchsc3([minlon, maxlon, minlat, maxlat, mindepth, maxdepth, minmag, maxmag, d1, d2])

#------------------------------
# PRINT SEARCH PARAMETERS AND NUMBER OF EVENTS FOUND
#------------------------------
print "----------------------------------"
print str(nevents) + ' EVENT(S) FOUND'
print "----------------------------------"
print "SEARCH PARAMETERS:"
print '	----------------'
print '	Start date/time:	', sdst
print '	End date/time:		', edst
print '	----------------'
print '	Min longitude:		', minlon
print '	Min latitute:		', minlat
print '	Max longitude:		', maxlon
print '	Max latitute:		', maxlat
print '	----------------'
print '	Min depth (km):		', mindepth
print '	Max depth (km):		', maxdepth
print '	----------------'
print '	Min magnitude:		', minmag
print '	Max magnitude:		', maxmag
print "----------------------------------"

# if no events found in the catalogue
if nevents == 0:
    sys.exit(0)

########################################
# BIN EVENTS daily
########################################

# Hourly
# dwin = datetools.Hour()
# nwin = 1./24
# pylabel = '# events per hour'
snum = np.ones(len(lon))
# Daily
# dwin = datetools.Day()
dwin = datetools.DateOffset()
nwin = 1
pylabel = '# events per day'

# all events
ser = TimeSeries(snum,dst)

# window - NOTE: CHANGE FOR SC3: FIRST EVENT = OLDEST --> REVERSE INDICES
# dststart = str(ser.index.year[0]) + '-' + str(ser.index.month[0]) + '-' + str(ser.index.day[0])
# dstend = str(ser.index.year[-1]) + '-' + str(ser.index.month[-1]) + '-' + str(ser.index.day[-1])

# dststart = str(ser.index.year[-1]) + '-' + str(ser.index.month[-1]) + '-' + str(ser.index.day[-1])
# dstend = str(ser.index.year[0]) + '-' + str(ser.index.month[0]) + '-' + str(ser.index.day[0])
dststart = str(ser.index[-1].year) + '-' + str(ser.index[-1].month) + '-' + str(ser.index[-1].day)
dstend = str(ser.index[0].year) + '-' + str(ser.index[0].month) + '-' + str(ser.index[0].day)

# rate = DateRange(dststart,dstend,offset=dwin)
rate = date_range(dststart,dstend)
ratesum = ser.groupby(rate.asof)
serrate = ratesum.sum()
rateval = serrate.values

# convert to datetime objects
# dstrate = serrate.index.to_datetime()
# adjusted in V2
dstrate = serrate.index
t = date2num(dstrate.to_pydatetime())

########################################
# Compute Seismic moment
# Note: time series ordered from recent to older events
# need to reverse that order to compute cumulative seismic moment
########################################
mo = 10.0**(1.5*mags+6.06)
dstcum = np.flipud(dst)
cummo = np.cumsum(np.flipud(mo))

########################################
# get current date and time for plot annotation
########################################
dnow = dt.datetime.strftime(dt.datetime.now(),'%Y-%m-%d %H:%M NZLT')

########################################
# PLOT
########################################

f = plt.figure(figsize=(20,11))

#------------------------------
# normalize dates to get colorscale
#------------------------------
cdst = (date2num(dst)-date2num(d1)) / (date2num(d2)-date2num(d1))


#------------------------------
# CUMULATIVE RATE OF SEISMICITY
#------------------------------
ax1 = plt.subplot(411)
plt.plot_date(date2num(dstcum),snum.cumsum(),'k')
plt.ylabel('Cumul. # events')
plt.title('Total number of events: ' + str(int(nevents)) + '     -     ' + dt.datetime.strftime(d1,'%Y-%m-%d %H:%M') + ' - ' + dt.datetime.strftime(d2,'%Y-%m-%d %H:%M'))
# second axis
ax2 = plt.twinx()
plt.plot_date(date2num(dstcum),cummo,':k')
ax2.set_ylabel('Cumul. seis. moment (N m)')

#------------------------------
# RATE OF SEISMICITY
#------------------------------
ax3 = plt.subplot(412,sharex=ax1)
plt.bar(num2date(t),rateval,width=nwin,color='b')
plt.ylabel(pylabel)

#------------------------------
# DEPTH OF SEISMICITY
ax4 = plt.subplot(413, sharex=ax1)
plt.scatter(dst,-z,c=date2num(dst))
v = plt.axis()
plt.axis((d1,d2,v[2],0))
plt.ylabel('Depth (km)')

#------------------------------
# MAGNITUDE
ax5 = plt.subplot(414, sharex=ax1)
# plt.plot_date(date2num(dst),mags,'ro')
plt.scatter(dst,mags,c=date2num(dst))
v = plt.axis()
plt.axis((d1,d2,v[2],v[3]))
plt.ylabel('Magnitude')

plt.show()

# # SAVE IMAGE TO LOCAL FOLDER
# imagefile = '/Users/nicof/scripts/geonet_images/' + net + '_earthquakerate_' + drange + '.png'
# plt.savefig(imagefile, dpi=100)
# plt.close()
# 
# # COPY PLOT IMAGE TO VOLCANO DEVELOPMENT WEB PAGE (TARAWERA)
# send2vdp(net)


# ########################################
# # PLOT B-VALUE
# ########################################
# plt.figure()
# plt.semilogx(ratevalmo,rateval,'ko')
# plt.xlabel('Seismic moment (N m)')
# plt.ylabel('# events')

########################################
# IMPORT AND PLOT TOPO
########################################
# topofile = '/Users/nicof/deformation/maps/nz_dtmvolcan200m_NZMG_nosea.grd'
# topofile='/Volumes/Data1/dem/nz_25m_20120403/tnp_wide_25m_dem.grd'
# topofile='/Volumes/Data1/dem/nz_25m_20120403/north_island_100m_dem.grd'
# topofile='/Users/nicof/Documents/data/dems/nz_25m_20120403/north_island_100m_dem.grd'
#topofile='/Volumes/data/topo/nz_25m_20120403/north_island_100m_dem.grd'
topofile='/Users/KTJ/Desktop/PostdocResources/katie/north_island_100m_dem.grd'

# boundaries for zone of interest
xmin = float(minlon)
xmax = float(maxlon)
ymin = float(minlat)
ymax = float(maxlat)

# set minmax
minmax = np.array([xmin,xmax,ymin,ymax])

#------------------------------
# read, crop and make basemap from topo file
#------------------------------
xs,ys,Xs,Ys,Zs,m = readtopo(topofile,minmax)

plt.figure()
# Make contour plot
cs = m.contour(Xs, Ys, Zs, 40, colors="k", lw=0.5, alpha=0.3)
# Plot parallels and meridians
m.drawparallels(np.linspace(ys.min(), ys.max(), 5), labels=[1, 1, 0, 0], fmt="%.2f", dashes=[2, 2])
m.drawmeridians(np.linspace(xs.min(), xs.max(), 5), labels=[0, 0, 1, 1], fmt="%.2f", dashes=[2, 2])
# set axes to equal
plt.axis('equal')


# add lightening
ls = LightSource(azdeg = 90, altdeg = 20)

# # OPTION 1 - coloured
# rgb = ls.shade(Zs, cm.GMT_haxby)
# rgb = ls.shade(Zs, cm.GMT_relief)
# OPTION 2 - grayed
rgb = ls.shade(Zs/100, plt.cm.Greys)

im = m.imshow(rgb,alpha=0.01)

#------------------------------
# plot epicentres
#------------------------------
xc,yc = m(lon,lat)

# m.scatter(xc,yc,20,marker='o',color='r',edgecolors='k')
# m.scatter(xc,yc,20,c=cdst,marker='o',color='r',edgecolors='k')

# size = magnitude ; colour = date/time
# m.scatter(xc,yc,s=mags*20,c=cdst,marker='o',color='r',edgecolors='k')
m.scatter(xc,yc,s=mags*20,c=date2num(dst),marker='o',color='r',edgecolors='k')


# show plot
plt.show()

########################################
# SAVE NUMBERS TO TMP FILE - COMMENT AS NEEDED
# FORMAT:
#	- dat
#	- gmt
########################################
#------------------------------
# rate of seismicity
#------------------------------
tmpfile = 'tmp/rateseis.tmp'
tmpfilegmt = 'tmp/rateseis_gmt.tmp'

fid = open(tmpfile,'w')
fidgmt = open(tmpfilegmt,'w')
# print one-line header
fid.write('#YYYY MM DD Nevents\n')
fidgmt.write('#YYYY-MM-DDT12:00:00 Nevents\n')

# print data
for i in np.arange(0,len(t)):
    outstr = dt.datetime.strftime(num2date(t[i]),'%Y %m %d ') + str(int(rateval[i])) + '\n'
    fid.write(outstr)
    outstr = dt.datetime.strftime(num2date(t[i]),'%Y-%m-%dT12:00:00 ') + str(int(rateval[i])) + '\n'
    fidgmt.write(outstr)
fid.close()
fidgmt.close()

#------------------------------
# cumulative rate of seismicity
#------------------------------
tmpfile = 'tmp/cumrateseis.tmp'
tmpfilegmt = 'tmp/cumrateseis_gmt.tmp'

fid = open(tmpfile,'w')
fidgmt = open(tmpfilegmt,'w')
# print one-line header
fid.write('#YYYY MM DD CumNevents\n')
fidgmt.write('#YYYY-MM-DDT12:00:00 CumNevents\n')

# print data
for i in np.arange(0,len(dstcum)):
    outstr = dt.datetime.strftime(dstcum[i],'%Y %m %d %H %M %S ') + str(int(np.cumsum(snum)[i])) + '\n'
    fid.write(outstr)
    outstr = dt.datetime.strftime(dstcum[i],'%Y-%m-%dT%H:%M:%S ') + str(int(np.cumsum(snum)[i])) + '\n'
    fidgmt.write(outstr)
fid.close()
fidgmt.close()

#------------------------------
# moment release
#------------------------------
tmpfile = 'tmp/momentrelease.tmp'
fid = open(tmpfile,'w')
# print one-line header
fid.write('#YYYY MM DD HH MN SS MomentRelease(N.m) CumulativeMomentRelease(N.m)\n')
# print data
for i in np.arange(0,len(dstcum)):
    outstr = dt.datetime.strftime(np.flipud(dst)[i],'%Y %m %d %H %M %S ') + str(np.flipud(mo)[i]) + ' ' + str(cummo[i]) + '\n'
    fid.write(outstr)
fid.close()


