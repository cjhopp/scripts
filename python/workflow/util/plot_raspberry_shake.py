#!/usr/bin/python
"""
Functions for retrieving and plotting data from a raspberry shake
on the local network
"""

import paramiko

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from itertools import cycle
from datetime import timedelta
from obspy import read, Stream, UTCDateTime
from obspy.taup import TauPyModel
from obspy.clients.fdsn import Client

def fetch_wavs(starttime, endtime):
    """
    Copy relevant waveforms from raspberry shake (if not already in directory).

    Will return a list of Stream objects for the necessary day(s)

    :param starttime: UTCDateTime object for starttime
    :param endtime: UTCDateTime object for endtime
    :return: List of Stream objects
    """
    local_archive = '/media/chet/hdd/seismic/rpi_shake/archive'
    remote_archive = '/opt/data/archive/2019/AM/RA8DB/SHZ.D'
    # '{ }' is a 'set literal' huh, fun. Thanks PyCharm
    jul_dates = list({starttime.julday, endtime.julday})
    archived_paths = glob('{}/*'.format(local_archive))
    archived_files = [p.split('/')[-1] for p in archived_paths]
    # Assuming we dont want 31 Dec and 1 Jan....
    we_want = ['AM.RA8DB.00.SHZ.D.{}.{}'.format(starttime.year, jd)
               for jd in jul_dates]
    fetch = [name for name in we_want if name not in archived_files]
    if len(fetch) > 0:
        # Open ssh and sftp to raspberry shake
        ssh = paramiko.SSHClient()
        paramiko.util.log_to_file('/tmp/paramiko.log')
        paramiko.util.load_host_keys('/home/chet/.ssh/known_hosts')
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname='192.168.1.9', username='myshake')
        sftp = ssh.open_sftp()
        for afile in fetch:
            sftp.get('{}/{}'.format(remote_archive, afile),
                     '{}/{}'.format(local_archive, afile))
        sftp.close()
        ssh.close()
    return [read('{}/{}'.format(local_archive, name)) for name in we_want]


def fetch_events(**kwargs):
    """
    Simple wrapper on obspy.fdsn client for USGS events
    :param kwargs: Any arguments passed to clients.fdsn.Client object
    :return:
    """
    return Client('USGS').get_events(**kwargs)


def calculate_arrivals(event, sta_lat=47.1314, sta_lon=-88.5947):
    """
    Use TauP to calculate theoretical arrivals from this event

    :param event: Event to calculate arrivals for at the shake
    :param sta_lat: Latitude of station (default to Hancock)
    :param sta_lon: Longitude of station (default to Hancock)
    :return:
    """
    model = TauPyModel(model='iasp91')
    o = event.preferred_origin()
    return model.get_travel_times_geo(source_depth_in_km=o.depth / 1000.,
                                      source_longitude_in_deg=o.longitude,
                                      source_latitude_in_deg=o.latitude,
                                      receiver_longitude_in_deg=sta_lon,
                                      receiver_latitude_in_deg=sta_lat)


def plot_event(event, pre_P_time=120., length=1200.):
    """
    Main function to be called for plotting waveforms from rpi shake
    overlain with picks predicted from TauP

    :param event: obspy.core.events.Event object to be plotted
    :param pre_P_time: Seconds before theoretical P to be plotted
    :param length: Total number of seconds to be plotted
    :return:
    """
    # Set up phase coloring cycle
    phase_colors = cycle(sns.color_palette())
    # Calculate the arrivals
    arrivals = calculate_arrivals(event)
    o = event.preferred_origin()
    P_arrival = [arr for arr in arrivals if arr.name == 'P'][0]
    # TauP arrivals only store travel time
    P_pick = (o.time + P_arrival.time)
    # Get the day-long wavform file(s)
    wavs = fetch_wavs(starttime=P_pick - pre_P_time,
                      endtime=P_pick - pre_P_time + length)
    st = Stream() # Preallocate Stream object and add waveforms
    for wav in wavs:
        st += wav
    # Merge for cases of segmented files or multiple days
    st.merge()
    # Cut to plotting length
    st.trim(starttime=P_pick - pre_P_time, endtime=P_pick -pre_P_time + length)
    tr = st[0] # Will only be one trace
    # Just set up the plot yourself....
    start_dt = tr.stats.starttime
    td = timedelta(microseconds=int(1 / tr.stats.sampling_rate * 1000000))
    dt_vect = [start_dt.datetime + (i * td) for i in range(len(tr.data))]
    data = tr.data
    # Plot em
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(dt_vect, data, color='k', linewidth=1.0)
    # Now loop all arrivals, plot line and annotate
    for arr in arrivals:
        ph_col = next(phase_colors)
        pk_time = (o.time + arr.time).datetime
        if pk_time > tr.stats.endtime:
            continue
        ax.axvline(pk_time, ymin=0.2, ymax=0.8, color=ph_col,
                   label='{}'.format(arr.name))
        ax.annotate(arr.name, xy=(pk_time, 0.8 * np.max(data)), xycoords='data',
                    color=ph_col, horizontalalignment='center',
                    fontsize=14)
    ax.set_title('{} | {} {} | {} km'.format(o.time, o.longitude, o.latitude,
                                             o.depth),
                 fontsize=16)
    fig.autofmt_xdate()
    plt.show()
    return fig