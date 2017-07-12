#/usr/bin/env python

"""
What the fuck is going on with this damn ifilter method??
"""

import pyasdf
from obspy import UTCDateTime

with pyasdf.ASDFDataSet('/media/chet/rotnga_data/pyasdf/mrp_rotnga.h5') as ds:
    for station in ds.ifilter(ds.q.station == 'NS12',
                              ds.q.starttime > (UTCDateTime(2015, 4, 30) - 10)):
        print(station)
        # st = station.raw_recording
