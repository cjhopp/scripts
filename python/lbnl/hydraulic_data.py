#!/usr/bin/python

"""
Parsing and plotting hydraulic data
"""
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
from nptdms import TdmsFile
from glob import glob
from scipy.io import loadmat
from datetime import datetime, timedelta



def datenum_to_datetime(datenums):
    # Helper to correctly convert matlab datenum to python datetime
    # SO source:
    # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    return [datetime.fromordinal(int(d)) +
            timedelta(days=d % 1) - timedelta(days=366)
            for d in datenums]


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
    df = df.drop(['datetime'], axis=1)
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


def read_martin_hydro_mat(path):
    """Helper to read-in raw array .mat files customized from Martin's .mat
    tables (Table objects don't translate to Python)"""
    # try:
    df = pd.DataFrame(loadmat(path)['dat_all'],
                      columns=['datenum', 'Quizix Press', 'Quizix Flow',
                               'Quizix Volume', 'Triplex Flow', 'SNL16',
                               'PNNL08', 'SNL10', 'SNL11'])
    # except KeyError:
    #     df = pd.DataFrame(loadmat(path)['dat_dec_ALL_array'],
    #                       columns=['datenum', 'Quizix Press', 'Quizix Flow',
    #                                'Quizix Volume', 'Triplex Flow', 'SNL16',
    #                                'PNNL08', 'SNL10', 'SNL11'])
    t = datenum_to_datetime(df.datenum)
    t = [t.replace(year=2018) for t in t]
    df.index = t
    df = df.drop(labels=['datenum', 'Quizix Volume'], axis=1)
    return df


def martin_cumulative_vols(df_early):
    """
    One-time func to generate cumulative volumes and max pressures for each
    stim in Martin's paper

    :param collab_df: Collab DataFrame

    :return:
    """

    stims = [
        (datetime(2018, 5, 22, 21, 40, 00), datetime(2018, 5, 22, 22, 10, 00)),  #Quizix
        (datetime(2018, 5, 23, 18, 20, 00), datetime(2018, 5, 23, 20, 10, 00)),  #Quizix
        (datetime(2018, 5, 24, 22, 10, 00), datetime(2018, 5, 24, 23, 00, 00)),  #Triplex
        (datetime(2018, 5, 25, 15, 00, 00), datetime(2018, 5, 25, 15, 45, 00)),  #Triplex
        (datetime(2018, 5, 25, 20, 20, 00), datetime(2018, 5, 25, 21, 30, 00)),  #Triplex
        (datetime(2018, 6, 25, 17, 00, 00), datetime(2018, 6, 25, 18, 30, 00)),  #
        (datetime(2018, 6, 25, 18, 30, 00), datetime(2018, 6, 25, 21, 30, 00)),  #
        (datetime(2018, 7, 19, 14, 00, 00), datetime(2018, 7, 19, 22, 00, 00)),  #Quizix
        (datetime(2018, 7, 20, 14, 30, 00), datetime(2018, 7, 21, 00, 00, 00)),  #Combo
        (datetime(2018, 12, 7, 23, 00, 00), datetime(2018, 12, 7, 23, 30, 00)),  #Triplex
        (datetime(2018, 12, 20, 16, 32, 00), datetime(2018, 12, 20, 21, 30, 00)),  #Triplex
        (datetime(2018, 12, 21, 18, 30, 00), datetime(2018, 12, 21, 23, 30, 00))]  #Triplex
    for stim in stims:
        # Stim name
        name = '{}'.format(stim[0].date())
        fig, ax = plt.subplots(figsize=(10, 7))
        ax2 = ax.twinx()
        stim_df = df_early[stim[0]:stim[1]]
        # Downsample to 30s
        stim_df = stim_df.resample('30s').mean()
        # Eliminate negative values
        stim_df[stim_df < 0] = 0.
        # Pressures to MPa
        if stim[0].month == 6:
            stim_df[['P1', 'P2']] = (stim_df[['PNNL08',
                                              'SNL11']] / 145.)
            # Do cumulative sum, account for L/min measure every 30 s
            stim_df['Cumulative'] = (
                    stim_df['SNL16'].cumsum() * 0.5 +
                    stim_df['Triplex Flow'].cumsum() + 0.5)
            stim_df[['Quizix Flow', 'Triplex Flow', 'SNL16']].plot(
                ax=ax, legend=False)
        else:
            stim_df[['P1', 'P2']] = (stim_df[['Quizix Press',
                                              'SNL10']] / 145.)
            stim_df['Cumulative'] = (
                    stim_df['Quizix Flow'].cumsum() * 0.5 +
                    stim_df['Triplex Flow'].cumsum() * 0.5)
            stim_df[['Quizix Flow', 'Triplex Flow']].plot(
                ax=ax, legend=False)
        stim_df['Cumulative'].plot(ax=ax2, color='k')
        ax.set_title('Stim {} - {}'.format(stim[0], stim[1]))
        ax.text(x=0.05, y=0.9,
                s='Max Volume: {:.3f} L\nMax Press: {:0.3f} MPa'.format(
                    stim_df['Cumulative'].max(),
                    stim_df[['P1', 'P2']].max(axis=1).max()),
                transform=ax.transAxes, fontsize=14,
                bbox=dict(boxstyle="round",
                          ec='k', fc='white',
                          zorder=103))
        ax.set_ylabel('L/min')
        ax2.set_ylabel('Liters')
        # Write csv
        stim_df.to_csv('Stim_{}.csv'.format(name))
        fig.legend()
        plt.savefig('Stim_{}.png'.format(name), dpi=200)
    return


###### Plotting #######

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