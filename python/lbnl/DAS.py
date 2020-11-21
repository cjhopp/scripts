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


def read_struct(f, depth=41):
    # Return the parts of the struct we actually want
    try:
        struct = loadmat(f, struct_as_record=False,
                         squeeze_me=True)
        data = struct['OT_strain'].data[depth, :]
        # Convert nano to microstrain
        data /= 1000.
        try:
            datenums = struct['OT_strain'].dn
        except AttributeError as e:
            datenums = struct['OT_strain'].datenum
    except NotImplementedError:
        # In case of Matlab 7.3 format
        with h5py.File(f, 'r') as f:
            # print(f['dasRate_CleanMedian']['date'][()])
            try:
                data = f['DAS_1min_rs']['strain'][()]
                datenums = np.concatenate(f['DAS_1min_rs']['dates'][()])
                depths = np.concatenate(f['DAS_1min_rs']['depths'][()])
                depth = np.argmin(np.abs(depths - depth))
            except KeyError as e:
                print(e)
                print(f['strain_OT']['strain'].shape)
                data = f['strain_OT']['strain'][()].T
                datenums = np.concatenate(f['strain_OT']['datenum'][()])
            data = data[depth, :] / 1000.
    try:
        time = datenum_to_datetime(datenums)
    except:
        time = None
    return time, data
