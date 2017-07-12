#/usr/bin/env python

"""
Download GeoNet stations which are not in MRP Dataset

Stations are: WPRZ, HRRZ, PRRZ, ALRZ, ARAZ, THQ2
"""
import os
from obspy import Stream
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from obspy import UTCDateTime
from dateutil import rrule
from datetime import datetime, date, timedelta
import warnings

# stations = ['WPRZ', 'HRRZ', 'PRRZ', 'ALRZ', 'ARAZ', 'THQ2']
#
# cli = Client('GEONET')
# starttime = UTCDateTime(2014, 01, 01)
# endtime = UTCDateTime(2015, 01, 02)

def grab_geonet(stations, starttime, endtime, outdir):
    """
    Pull geonet waves to local directory
    :param start: UTCDateTime start date
    :param end: UTCDateTime end date
    :return: Nada
    """
    cli = Client('GEONET')
    for dt in rrule.rrule(rrule.DAILY, dtstart=starttime, until=endtime):
        print('Downloading data for '+str(dt)+':')
        st = Stream()
        start = UTCDateTime(dt)
        end = UTCDateTime(dt + timedelta(days=1))
        if not os.path.exists('%s/%d' % (outdir, start.year)):
            os.makedirs('%s/%d/NZ' % (outdir, start.year))
        for station in stations:
            if not os.path.exists('%s/%d/NZ/%s' % (outdir, start.year, station)):
                os.makedirs('%s/%d/NZ/%s' % (outdir, start.year, station))
            if station == 'THQ2':
                channels = ['EH1', 'EH2', 'EHZ']
            else:
                channels = ['EHE', 'EHN', 'EHZ']
            for chan in channels:
                if not os.path.exists('%s/%d/NZ/%s/%s.D' % (outdir, start.year, station, chan)):
                    os.makedirs('%s/%d/NZ/%s/%s.D' % (outdir, start.year, station, chan))
                print(station+'.'+chan)
                try:
                    st = cli.get_waveforms("*", station, "*", chan, start, end)
                except FDSNException:
                    warnings.warn('No data available for this station/channel')
                filename = '%s/%d/NZ/%s/%s.D/NZ.%s.10.%s.D.%d.%03d' % (outdir,
                                                                       start.year,
                                                                       station,
                                                                       chan,
                                                                       station,
                                                                       chan,
                                                                       dt.year,
                                                                       start.julday)
                if len(st) > 0:
                    print('Writing to file: %s' % filename)
                    st.write(filename, format="MSEED")
    return
