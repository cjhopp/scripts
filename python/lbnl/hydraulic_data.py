#!/usr/bin/python

"""
Parsing and plotting hydraulic data
"""
import pandas as pd

from matplotlib.dates import date2num
from nptdms import TdmsFile
from datetime import datetime

def read_collab_hydro(path):
    """
    Read Martin's custom flow/pressure file

    :param path:
    :return:
    """
    df = pd.read_csv(path)
    df['dt'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    df = df.set_index('dt')
    df = df.drop(['date'], axis=1)
    df = df.drop(['datenum'], axis=1)
    df.rename(columns={'injection_Lpm': 'Flow', 'pressure_MPa': 'Pressure'},
              inplace=True)
    # Resample this shiz to shrink the dataset
    df = df.resample('10s').mean()
    return df


def read_packer_pressures(path):
    """
    Reading Martin's packer pressure matlab file

    :param path: File path to csv
    :return:
    """
    df = pd.read_csv(path)
    df['dt'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S')
    df = df.set_index('dt')
    df = df.drop(['Time'], axis=1)
    # Resample this shiz to shrink the dataset
    df = df.resample('10s').mean()
    return df


def read_csd_hydro(path):
    """
    Stolen from Antonio for reading CSD pump data (LabView)
    :param path: File path

    :return:
    """
    tdms_file = TdmsFile(path)
    pump_df = tdms_file['Channels'].as_dataframe()
    pump_df = pump_df.set_index('Time')
    pump_df.rename(columns={'Interval pressure': 'Pressure',
                            'Pump flow': 'Flow'},
                   inplace=True)
    pump_df = pump_df.sort_index()
    pump_df.index = pump_df.index.tz_localize('Etc/GMT+1')
    pump_df.index = pump_df.index.tz_convert('UTC')
    pump_df.index = pump_df.index.tz_convert(None)
    pump_df['Pressure'] /= 145.  # To MPa from psi
    return pump_df