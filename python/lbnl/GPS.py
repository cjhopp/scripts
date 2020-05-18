#!/usr/bin/python

"""Functions for parsing and plotting GPS waveforms"""

import numpy as np

from datetime import datetime, timedelta

def decyear_to_datetime(times):
    """
    Helper to convert decimal years to datetimes

    SO Answer here: https://stackoverflow.com/questions/20911015/decimal-years-to-datetime-in-python
    """

    # TODO  Will accumulate errors for leap years over multiple years!
    start = times[0]
    year = int(start)
    rems = [t - int(t) for t in times]
    bases = [datetime(int(t), 1, 1) for t in times]
    dtos = [b + timedelta(
        seconds=(b.replace(year=b.year + 1) - b).total_seconds() * rems[i])
            for i, b in enumerate(bases)]
    return dtos


def read_PANGA(in_file):
    """
    Read in data from CWU geodetic array website

    :param in_file: Path to data file
    :return:
    """
    arr = np.loadtxt(in_file, skiprows=30)
    times = decyear_to_datetime(arr[:, 0])
    return times, arr[:, 1], arr[:, 2]