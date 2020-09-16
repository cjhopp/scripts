#!/usr/bin/python

"""
Parsing and plotting hydraulic data
"""
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
from nptdms import TdmsFile
from glob import glob
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


def read_csd_hydro(root_path):
    """
    Stolen from Antonio for reading CSD pump data (LabView)
    :param path: File path

    :return:
    """
    frames = []
    for f in glob('{}/*.tdms'.format(root_path)):
        tdms_file = TdmsFile(f)
        pump_df = tdms_file['Channels'].as_dataframe()
        pump_df = pump_df.set_index('Time')
        pump_df.rename(columns={'Interval pressure': 'Pressure',
                                'Pump flow': 'Flow'},
                       inplace=True)
        pump_df = pump_df.sort_index()
        pump_df.index = pump_df.index.tz_localize('Etc/GMT+1')
        pump_df.index = pump_df.index.tz_convert('UTC')
        pump_df.index = pump_df.index.tz_convert(None)
        pump_df['Pressure'] /= 1000.  # To MPa from kPa
        frames.append(pump_df)
    all_df = pd.concat(frames)
    all_df = all_df.sort_index()
    return all_df


def plot_csd_hydro(df_hydro, title='Flow and Pressure'):
    """Simple Flow and Press plot"""
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    df_hydro['Flow'].plot(ax=ax, color='steelblue', label='Flow')
    ax.set_ylim(bottom=0)
    df_hydro['Pressure'].plot(ax=ax2, color='firebrick', label='Pressure')
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('MPa')
    ax.set_ylabel('ml/min')
    if (df_hydro.index[-1] - df_hydro.index[0]).days == 0:
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.set_xlabel('Time on {}'.format(df_hydro.index.date[0]))
    fig.legend()
    fig.suptitle(title, fontsize=16)
    plt.show()
    return