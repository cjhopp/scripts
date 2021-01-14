#!/usr/bin/python

"""
Functions for reading GMuG waveform data and writing to obspy/h5
"""

import yaml

import numpy as np

from datetime import datetime
from obspy import Stream, Trace, UTCDateTime

# Multiplexed channel order from a-to-d board
AD_chan_order = np.array([1,9,2,10,3,11,4,12,5,13,6,14,7,15,8,16]) - 1

def read_raw_continuous(path, chans=16):
    """Read a raw, multiplexed .dat continuous waveform file"""
    raw = np.fromfile(path, dtype=np.int16)
    raw = raw.reshape((-1, chans))
    return raw


def parse_continuous_metadata(path):
    # Time info
    with open(path.replace('.dat', '.txt')) as f:
        lines = f.readlines()
        ts = ''.join(lines[1].split()[-2:])
        channels = int(lines[3].split()[-1])
        delta = 1 / int(lines[4].split()[-1])
    starttime = datetime(year=int(ts[:4]), month=int(ts[4:6]), day=int(ts[6:8]),
                         hour=int(ts[8:10]), minute=int(ts[10:12]),
                         second=int(ts[12:14]),
                         microsecond=int('{}000'.format(ts[14:])))
    return UTCDateTime(starttime), channels, delta


def gmug_to_stream(pattern, config):
    """
    Take binary continuous wav and header file and return obspy Stream

    :param pattern: Root filename without extension (will be added)
    :return:
    """
    # Read in the config file and grab sta.chan list
    with open(config, 'r') as f:
        param = yaml.load(f)
    stachans = np.array(param['Mapping']['GMuG_stachans'])
    # Re-order per multiplexer order
    multi_stachans = stachans[AD_chan_order]
    starttime, no_chans, delta = parse_continuous_metadata(
        '{}.txt'.format(pattern))
    np_raw = read_raw_continuous('{}.dat'.format(pattern), chans=no_chans)
    st = Stream(traces=[
        Trace(data=np_raw[:, i],
              header=dict(delta=delta, starttime=starttime,
                          network='FS', station=multi_stachans[i].split('.')[0],
                          channel=multi_stachans[i].split('.')[1],
                          location=''))
        for i in range(no_chans) if multi_stachans[i] != '.'])
    return st