#!/usr/bin/python
"""
Functions for retrieving and plotting data from a raspberry shake
on the local network
"""

import os
import paramiko

from glob import glob
from obspy import read
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