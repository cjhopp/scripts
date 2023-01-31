#!/usr/bin/python

"""
Utilities for manupulating/plotting seiscomp output
"""

from obspy import UTCDateTime

import matplotlib.dates as mdates
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def plot_station_availability(xml_file, networks):
    # Make namespace dict
    ns = {'sc': 'http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.11'}
    tree = ET.parse(xml_file)
    root = tree.getroot()
    fig, ax = plt.subplots()
    for extent in root[0].findall('sc:extent', ns):
        wid = extent.find('sc:waveformID', ns)
        if wid.get('networkCode') not in networks:
            continue
        seed = '.'.join([wid.get('networkCode'),
                         wid.get('stationCode')])
        start = UTCDateTime(extent.find('sc:start', ns).text).datetime
        end = UTCDateTime(extent.find('sc:end', ns).text).datetime
        start = mdates.date2num(start)
        end = mdates.date2num(end)
        ax.barh(seed, end-start, left=start)
    ax.set_title('Waveform availability')
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.show()
    return