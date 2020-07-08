#!/usr/bin/python

import h5py

import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from scipy.io import loadmat


def datenum_to_datetime(datenums):
    # Helper to correctly convert matlab datenum to python datetime
    # SO source:
    # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    return [datetime.fromordinal(int(d)) +
            timedelta(days=d % 1) - timedelta(days=366)
            for d in datenums]


def read_struct(f):
    # Return the parts of the struct we actually want
    struct = loadmat(f, struct_as_record=False,
                     squeeze_me=True)
    data = struct['OT_strain'].data[41, :]
    # Convert nano to microstrain
    data /= 1000.
    datenums = struct['OT_strain'].dn
    time = datenum_to_datetime(datenums)
    return time, data
