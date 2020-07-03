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
    f = h5py.File(f, 'r')
    das_dict = {}
    das_dict['data'] = np.array(f['OT_strain']['data'])
    das_dict['times'] = np.array(f['OT_strain']['time'])
    return das_dict
