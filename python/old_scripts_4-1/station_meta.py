#!/usr/bin/python

"""
Retrieve station info from GeoNet, convert to XSEED(?)
and save into SeisHub
"""

from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.xseed import Parser
from obspy.clients.fdsn.header import FDSNException
from glob import glob
#GeoNet FDSN client
client = Client('http://service.geonet.org.nz')
starttime = UTCDateTime(2012, 06, 19)
endtime = UTCDateTime(2012, 06, 28)

#Get station metadata in individual files
stations = ['RT01', 'RT02', 'RT03', 'RT05', 'RT06', 'RT07', 'RT08', 'RT09',
            'RT10', 'RT11', 'RT12', 'RT12', 'RT13', 'RT14', 'RT15', 'RT16',
            'RT17', 'RT18', 'RT19', 'RT20', 'RT21', 'NS01', 'NS02', 'NS03',
            'NS04', 'NS05', 'NS06', 'NS07', 'NS08', 'NS09', 'NS10', 'NS11',
            'NS12', 'NS13', 'NS14', 'NS15', 'NS16', 'NS18', 'WPRZ', 'HRRZ',
            'PRRZ', 'ALRZ', 'ARAZ', 'THQ2', 'RT23', 'RT22']
new_stas = ['RT23', 'RT22', 'NS15', 'NS16', 'NS18']
for station in new_stas:
    try:
        sta_inv = client.get_stations(station=station, level="response")
    except FDSNException:
        print('No StationXML available')
    sta_inv.write('/home/chet/data/GeoNet_catalog/stations/station_xml/'
                  + station + '_STATIONXML.xml', format='STATIONXML')

"""
Intermediate step to use stationxml-converter java app (IRIS)
Perhaps can be done from this script?
"""

dataless_files = glob('/home/chet/data/GeoNet_catalog/stations/*.dataless')
for file1 in dataless_files:
    #Read dataless to obspy, then write to XSEED
    sp = Parser(file1)
    sp.writeXSEED('/home/chet/data/GeoNet_catalog/stations/'+str(file1[-13:-9])+'_xseed.xml')
