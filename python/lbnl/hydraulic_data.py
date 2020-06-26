#!/usr/bin/python

"""
Parsing and plotting hydraulic data
"""
import pandas as pd

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
    # Resample this shiz to eliminate
    df = df.resample('10s').mean()
    # Assuming these data are corrected for clamp effects now??
    return df