#!/usr/bin/python

"""
Parsing and plotting hydraulic data
"""
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, DayLocator, MonthLocator, HourLocator
from matplotlib.collections import LineCollection
from matplotlib.dates import date2num, DateFormatter
from matplotlib import gridspec
from glob import glob
from scipy.io import loadmat
from datetime import datetime, timedelta
from itertools import cycle
from pandas._libs.tslibs.parsing import DateParseError
try:
    from nptdms import TdmsFile
except ImportError:
    print('No TDMS package installed.')

def datenum_to_datetime(datenums):
    # Helper to correctly convert matlab datenum to python datetime
    # SO source:
    # https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    return [datetime.fromordinal(int(d)) +
            timedelta(days=d % 1) - timedelta(days=366)
            for d in datenums]


def CCCO2(x, a, b, c):  # Functional fit for CO2-H2O phase boundries of Wendland et al 1999
    return np.exp(a + (b / x) + (c * x))


def CCCO2_spycher(x):  # Fit to the high pressure CO2-H2O data from Spycher and Pruess
    return 44.6 * (1 + 32.33 * np.sqrt((x / 9.77) - 1) + 91.169 * ((x / 9.77) - 1))


def plot_CO2_phase_diagram(ax=None):
    """
    Take functions from Spycher & Pruess and Wendland et al. 1999 and plot the CO2-H2O phase space at low pressures
    :return:
    """
    styles = cycle([':', '-', '-.', '--'])
    # Leave these in
    phase_dict = {r'$L_{1}L_{2}V$': {'x': np.linspace(270.3, 304.6, 100), 'popt': [3.99263, -1335.97, 0.007865]},
                  r'$HL_{1}V$': {'x': np.linspace(270.3, 282.9, 100), 'popt': [-446.88395, 57308.65, 0.868805]}}
    if not ax:
        fig, ax = plt.subplots()
    # First the Wendland functions
    for lab, pdict in phase_dict.items():
        ax.plot(pdict['x'] - 273.15, CCCO2(pdict['x'], *pdict['popt']), color='k', linestyle=next(styles))#, label=lab
    # Now quadruple point from Spycher
    ax.scatter(9.62, 4.46, color='k')
    # Spycher high pressure function
    ax.plot(np.linspace(9.62, 12.77, 100), CCCO2_spycher(np.linspace(9.77, 12.77, 100) + 273.15) / 10., #label=r'$HL_{1}L_{2}$',
            color='k', linestyle=next(styles))
    # Annotate two-phase regions
    ax.annotate(text=r'$H_{2}O + LIQ_{CAR}$', xy=(14.85, 5.5), rotation=3)
    ax.annotate(text=r'$H_{2}O + CO_{2}$', xy=(14.85, 4.5), rotation=3)
    ax.annotate(text=r'$CLA + CO_{2}$', xy=(6.9, 3.5), rotation=20)
    ax.annotate(text=r'$CLA + LIQ_{CAR}$', xy=(7.85, 6.5), rotation=90)
    ax.set_ylim([0, 9.0])
    ax.set_xlim([14.35, 17.7])
    ax.legend()
    ax.set_ylabel('Pressure [MPa]')
    ax.set_xlabel('Temperature [C]')
    return ax


def plot_B12_minireudi(hydro_data, date_range=False, ax=None):
    """
    Plot the minireudi data from BFS-B12 on a time axis

    :param hydro_data:
    :param date_range:
    :param ax:
    :return:
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(16, 5))
    hydro_data = hydro_data[date_range[0]:date_range[1]]
    ax.plot(hydro_data.index.values, hydro_data['Temperature'], color='darkolivegreen', label='Temperature 39 m')
    ax.plot(hydro_data.index.values, hydro_data['pH'], color='dodgerblue', label='pH')
    # Pressures
    ax2 = ax.twinx()
    ax2.plot(date2num(hydro_data.index.values), hydro_data['downhole pressure [kPa]'], color='firebrick',
             label='Pressure')
    # If CO2, plot it on flow axis
    if 'CO2 partial pressure [kPa]' in hydro_data.keys():
        ax2.plot(date2num(hydro_data.index.values), hydro_data['CO2 partial pressure [kPa]'], label=r'$CO_{2} pp$',
                 color='magenta', alpha=0.3)
        # ax2.legend()
    ax.set_ylabel(r'$^{O}C$ or pH', color='darkolivegreen')
    ax2.set_ylabel('kPa', color='firebrick')
    ax.tick_params(axis='y', which='major', labelcolor='darkolivegreen',
                   color='darkolivegreen')
    ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                    color='firebrick')
    if date_range[1] - date_range[0] > timedelta(days=1):
        ax.set_xlabel('Date')
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))
    else:
        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlim([date2num(hydro_data.index.values[0]), date2num(hydro_data.index.values[-1])])
    # ax.set_ylim(bottom=0)
    ax2.set_xlim([date2num(hydro_data.index.values[0]), date2num(hydro_data.index.values[-1])])
    ax2.set_ylim(bottom=0)
    fig.legend()
    return ax, ax2


def plot_PT_timeseries(hydro_data, date_range, ax=None):
    """
    Plot Pressure and Temperature time series
    :param hydro_data: DataFrame with "Pressure" and "Temperature" columns
    :param date_range: Start and end datetimes
    :param ax: Optional axes to plot into
    :return:
    """
    if not ax:
        fig, ax = plt.subplots()
    hydro_data = hydro_data[date_range[0]:date_range[1]]
    hydro_data['Temperature'].plot(color='darkolivegreen', ax=ax, label='Temperature')
    # Pressure
    ax2 = ax.twinx()
    ax2.plot(date2num(hydro_data.index.values), hydro_data['Pressure'], color='firebrick', label='Pressure')
    # If CO2, plot it on flow axis
    if 'CO2' in hydro_data.keys():
        ax2.plot(date2num(hydro_data.index.values), hydro_data['CO2'], label=r'$CO_{2} [g/g]*10$',
                 color='magenta', alpha=0.3)
        ax2.legend()
    ax.set_ylabel(r'$^{O}C$', color='darkolivegreen')
    ax2.set_ylabel('MPa', color='firebrick')
    ax.tick_params(axis='y', which='major', labelcolor='darkolivegreen',
                   color='darkolivegreen')
    ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                    color='firebrick')
    if date_range[1] - date_range[0] > timedelta(days=1):
        ax.set_xlabel('Date')
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))
    else:
        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlim([date2num(hydro_data.index.values[0]), date2num(hydro_data.index.values[-1])])
    # ax.set_ylim(bottom=0)
    ax2.set_xlim([date2num(hydro_data.index.values[0]), date2num(hydro_data.index.values[-1])])
    ax2.set_ylim(bottom=0)
    return ax


def plot_hydro(hydro_data, date_range, color_by_flow=False, ax=None):
    """
    Plot injection paramters within a date range

    :param hydro_data: DataFrame with injection paramters 'Flow', 'Pressure', potentially CO2 concentration
    :param date_range: Iterable of start, end datetime
    :return:
    """
    if not ax:
        fig, ax = plt.subplots()
    hydro_data = hydro_data[date_range[0]:date_range[1]]
    if color_by_flow:
        # Line collection
        x = date2num(hydro_data.index.values)
        y = hydro_data['Flow'].values
        segments = np.vstack([x, y]).T
        segments = segments.reshape(-1, 1, 2)
        segments = np.hstack([segments[:-1], segments[1:]])
        segments = list(segments)
        coll = LineCollection(segments, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
        coll.set_array(hydro_data['Flow'].values)
        ax.add_collection(coll)
        ax.autoscale_view()
    else:
        hydro_data.plot('Flow', color='steelblue', ax=ax, label='Flow')
    # Pressure
    ax2 = ax.twinx()
    ax2.plot(date2num(hydro_data.index.values), hydro_data['Pressure'], color='firebrick', label='Pressure')
    # If CO2, plot it on flow axis * 100
    if 'CO2' in hydro_data.keys():
        ax.plot(date2num(hydro_data.index.values), hydro_data['CO2'] * 10, label=r'$CO_{2} [g/g]*10$',
                color='magenta', alpha=0.3)
    ax.legend()
    ax.set_ylabel('L/min', color='steelblue')
    ax2.set_ylabel('MPa', color='firebrick')
    ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                   color='steelblue')
    ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                    color='firebrick')
    ax.set_xlabel('Time')
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.set_xlim([date2num(hydro_data.index.values[0]), date2num(hydro_data.index.values[-1])])
    ax.set_ylim(bottom=0)
    ax2.set_xlim([date2num(hydro_data.index.values[0]), date2num(hydro_data.index.values[-1])])
    ax2.set_ylim(bottom=0)
    return ax


def plot_PT_curve(hydro_data, date_range, ax=None):
    """
    Plot a PT curve (onto a phase diagram, typically) from a pre-made dataframe with pressure and temperature columns,
    colored by a categorized flow rate column

    :param hydro_df: DataFrame that contains Pressure and Temperature columns
    :param date_range: Start and end times for data to plot
    :return:
    """
    if not ax:
        fig, ax = plt.subplots()
    hydro_data = hydro_data[date_range[0]:date_range[1]]
    # Line collection
    x = hydro_data['Temperature'].values
    y = hydro_data['Pressure'].values
    # Units of phase diagram are K vs bar
    ax.scatter(x[0], y[0], marker='o', s=60, color='green', label='Starting point')
    ax.scatter(x[-1], y[-1], marker='x', s=60, color='k', label='Ending point')
    segments = np.vstack([x, y]).T
    segments = segments.reshape(-1, 1, 2)
    segments = np.hstack([segments[:-1], segments[1:]])
    coll = LineCollection(segments, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
    try:
        coll.set_array(hydro_data['Flow'].values)
    except KeyError:
        pass
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.legend()
    return ax


def plot_PT_movie(dataframe, date_range, frame_rate, outdir):
    """
    Save frames for a movie of PT plot and injection parameters

    :param dataframe: DataFrame with columns 'Pressure', 'Temperature', and 'Flow'
    :param date_range: Iterable of start and end time for the plot
    :param frame_rate: timedelta corresponding to frame rate
    :param outdir: Output directory for frames
    :return:
    """
    end_times = [date_range[0] + (frame_rate * i) for i in range((date_range[1] - date_range[0]) // frame_rate)]
    for i, et in enumerate(end_times):
        fig, axes = plt.subplots(figsize=(7, 12), nrows=2)
        plot_CO2_phase_diagram(ax=axes[0])
        plot_PT_curve(dataframe, [date_range[0], et], ax=axes[0])
        plot_hydro(dataframe, [date_range[0], et], color_by_flow=True, ax=axes[1])
        fig.text(0.05, 0.90, et, ha="left", va="bottom", fontsize=14,
                 bbox=dict(boxstyle="round",
                           ec='k', fc='white'))
        fig.savefig('{}/frame_{:03d}.png'.format(outdir, i), dpi=300)
        plt.close('all')
    return


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
    df.rename(columns={'injection_Lpm': 'Flow', 'pressure_MPa': 'Pressure'},
              inplace=True)
    # Resample this shiz to shrink the dataset
    df = df.resample('10s').mean()
    return df


def read_4100_circulation(path):
    df = pd.read_csv(path, header=0, parse_dates=['Time'])
    df = df.set_index('Time')
    df.sort_index(inplace=True)
    return df


def read_4100_hydro(path):
    flow_files = glob('{}/*.csv'.format(path))
    flow_files.sort()
    df = pd.DataFrame()
    for f in flow_files:
        print(f)
        # zone = f.split('_')[0].split('/')[-1]
        cols_tri = ['Time', 'Quizix Flow', 'Quixiz Pressure',
                    'PT 403', 'Net Flow']
        ndf = pd.read_csv(f, usecols=cols_tri, skiprows=[1, 2], index_col=False)#, parse_dates=[0], date_format='%m/%d/%y %H:%M:%S')
        rename_dict = {
            'Quizix Flow': 'Quizix Flow',
            'Quixiz Pressure': 'Quizix P',
            'Net Flow': 'Triplex Flow'
        }
        ndf.rename(columns=rename_dict, inplace=True)
        print(ndf)
        try:
            ndf['Datetime'] = pd.to_datetime(ndf['Time'], format='%m/%d/%y %H:%M:%S')
        except ValueError:
            print(ndf)
            continue
        ndf = ndf.set_index('Datetime')
        ndf.sort_index(inplace=True)
        ndf.drop('Time', axis=1, inplace=True)
        ndf = ndf.resample('1Min').mean()
        df = pd.concat([df, ndf])
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


def read_fsb_hydro(path, year=2023, B1_path=False, B12_path=False):
    """Helper to read in Pauls hydraulic data"""
    if year == 2020:
        df = pd.read_csv(path, names=['Time', 'Pressure', 'Flow'], header=0)
        df['dt'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S.%f')
        tz = 'CET'
        df = df.set_index('dt')
        df = df.drop(['Time'], axis=1)
        df.index = df.index.tz_localize(tz)
        df.index = df.index.tz_convert('UTC')
        df.index = df.index.tz_convert(None)
    elif year == 2023:
        df = pd.read_csv(path, names=['Time', 'Flow', 'Pressure', 'CO2'], header=0)
        df['dt'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S.%f')
        tz = 'UTC'
        df = df.set_index('dt')
        df = df.drop(['Time'], axis=1)
        df.index = df.index.tz_localize(tz)
        df.index = df.index.tz_convert('UTC')
        df.index = df.index.tz_convert(None)
        df = df.resample('5s').mean()
        if B1_path:
            pkls = glob('{}/*.pkl'.format(B1_path))
            for pkl in pkls:
                with open(pkl, 'rb') as f:
                    df1 = pickle.load(f)
                    df1 = df1.rename({'Pressure': '{} Pressure'.format(pkl.split('/')[-1]),
                                      'Temperature': '{} Temperature'.format(pkl.split('/')[-1])}, axis=1)
                df1 = df1.resample('5s').mean()
                df = pd.concat([df, df1], join='outer')
        if B12_path:
            df12 = pd.read_csv(B12_path, names=['Time', 'B12 pressure [kPa]', 'CO2 pp', 'pH'], header=0)
            df12['dt'] = pd.to_datetime(df12['Time'], format='%m/%d/%Y %H:%M:%S.%f')
            tz = 'UTC'
            df12 = df12.set_index('dt')
            df12 = df12.drop(['Time'], axis=1)
            df12 = df12.resample('5s').mean().interpolate('linear')
            df12.plot()
            plt.show()
            df12.index = df12.index.tz_localize(tz)
            df12.index = df12.index.tz_convert('UTC')
            df12.index = df12.index.tz_convert(None)
            df = pd.concat([df, df12], join='outer')
    elif year == 2021:
        df = pd.read_csv(path, names=['Time', 'Pressure', 'Packer', 'Flow', 'Qin', 'Qout', 'Hz'], header=0)
        df['dt'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S')
        tz = 'CET'  # ????
        df = df.set_index('dt')
        df = df.drop(['Time'], axis=1)
        df.index = df.index.tz_localize(tz)
        df.index = df.index.tz_convert('UTC')
        df.index = df.index.tz_convert(None)
    return df


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
        name = '{}'.format(stim[0])
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


def plot_4100_circulation(df, date_range=None, ax=None, ax2=None,
                          legend=False):
    if date_range:
        df = df[date_range[0]:date_range[1]]
    if not ax:
        fig, ax = plt.subplots(figsize=(25, 6))
    if not ax2:
        ax2 = ax.twinx()
    ax.plot(df['Time'], df['Net Flow'], color='steelblue')
    ax2.plot(df['Time'], df['Injection Pressure'], color='firebrick')
    if legend:
        ax1ln, ax1lab = ax.get_legend_handles_labels()
        ax2ln, ax2lab = ax2.get_legend_handles_labels()
        ax.legend(ax1ln, ax1lab)
        ax2.legend(ax2ln, ax2lab)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.set_ylabel('L/min', color='steelblue')
    ax2.set_ylabel('psi', color='firebrick')
    ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                   color='steelblue')
    ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                    color='firebrick')
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(DayLocator(interval=7))
    ax2.xaxis.set_major_locator(DayLocator(interval=7))
    if legend:
        plt.legend()
    return


def plot_4100_hydro(df, date_range=None, ax=None, legend=False):
    if date_range:
        df = df[date_range[0]:date_range[1]]
    df.resample('20T')
    if not ax:
        fig, ax = plt.subplots(figsize=(25, 6))
    ax2 = ax.twinx()
    Q = df.filter(like='Flow')
    quizP = df.filter(like='Quizix P')
    Q.plot(ax=ax, color=sns.color_palette('Blues', 12).as_hex(), legend=legend)
    quizP.plot(ax=ax2, color=sns.color_palette('Reds', 6).as_hex(), legend=legend)
    df['PT 403'].plot(ax=ax2, color='firebrick')
    if legend:
        ax1ln, ax1lab = ax.get_legend_handles_labels()
        ax2ln, ax2lab = ax2.get_legend_handles_labels()
        ax.legend(ax1ln, ax1lab)
        ax2.legend(ax2ln, ax2lab)
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.set_ylabel('L/min', color='steelblue')
    ax2.set_ylabel('psi', color='firebrick')
    ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                   color='steelblue')
    ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                    color='firebrick')
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(DayLocator(interval=1))
    return fig, ax, ax2


def plot_collab_ALL(df_hydro, date_range=None, axes=None):
    if not axes:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax2 = ax.twinx()
    else:
        ax = axes
        ax2 = ax.twinx()
    # Downsample off the bat
    df_hydro = df_hydro.resample('30s').mean()
    df_hydro = df_hydro[date_range[0]:date_range[1]]
    stims = [
        (datetime(2018, 5, 22, 21, 40, 00), datetime(2018, 5, 22, 22, 10, 00)),  #Quizix
        (datetime(2018, 5, 23, 18, 20, 00), datetime(2018, 5, 23, 20, 10, 00)),  #Quizix
        (datetime(2018, 5, 24, 22, 10, 00), datetime(2018, 5, 24, 23, 00, 00)),  #Triplex
        (datetime(2018, 5, 25, 15, 00, 00), datetime(2018, 5, 25, 15, 45, 00)),  #Triplex
        (datetime(2018, 5, 25, 20, 20, 00), datetime(2018, 5, 25, 21, 30, 00)),#Triplex
        (datetime(2018, 6, 20, 12, 00, 00), datetime(2018, 6, 21, 00, 00, 00)),
        (datetime(2018, 6, 21, 12, 00, 00), datetime(2018, 6, 22, 00, 00, 00)),
        (datetime(2018, 6, 22, 12, 00, 00), datetime(2018, 6, 23, 00, 00, 00)),
        (datetime(2018, 6, 25, 17, 00, 00), datetime(2018, 6, 25, 18, 30, 00)),  #
        (datetime(2018, 6, 25, 18, 30, 00), datetime(2018, 6, 25, 21, 30, 00)),  #
        (datetime(2018, 7, 19, 14, 00, 00), datetime(2018, 7, 19, 22, 00, 00)),  #Quizix
        (datetime(2018, 7, 20, 14, 30, 00), datetime(2018, 7, 21, 00, 00, 00)),  #Combo
        (datetime(2018, 12, 7, 23, 00, 00), datetime(2018, 12, 7, 23, 30, 00)),  #Triplex
        (datetime(2018, 12, 20, 16, 32, 00), datetime(2018, 12, 20, 21, 30, 00)),  #Triplex
        (datetime(2018, 12, 21, 18, 30, 00), datetime(2018, 12, 21, 23, 30, 00))]  #Triplex
    for stim in stims:
        if ~np.all(np.array([date_range[0] <= s < date_range[1]
                             for s in stim])):
            continue
        # Stim name
        stim_df = df_hydro[stim[0]:stim[1]].copy()
        # Eliminate negative values
        stim_df[stim_df < 0] = 0.
        # Pressures to MPa
        if stim[0].month == 6:
            stim_df[['P1', 'P2']] = (stim_df[['PNNL08',
                                              'SNL11']] / 145.)
            # Do cumulative sum, account for L/min measure every 30 s
            stim_df[['Quizix Flow', 'Triplex Flow', 'SNL16']].plot(
                ax=ax, legend=False, color='steelblue')
        else:
            stim_df[['P1', 'P2']] = (stim_df[['Quizix Press',
                                              'SNL10']] / 145.)
            stim_df[['Quizix Flow', 'Triplex Flow']].plot(
                ax=ax, legend=False, color='steelblue')
        stim_df[['P1', 'P2']].plot(ax=ax2, color='firebrick', legend=False)
    ax.set_ylabel('L/min', color='steelblue')
    ax2.set_ylabel('MPa', color='firebrick')
    ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                   color='steelblue')
    ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                    color='firebrick')
    ax.set_ylim(bottom=0.)
    ax2.set_ylim(bottom=0.)
    if not axes:
        fig.legend()
    return


def plot_collab_hydro(df_hydro):
    """
    DataFrame from read_collab_hydro

    :param df_hydro: pandas DataFrame
    :return:
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    ax2 = ax.twinx()
    df_hydro = df_hydro[datetime(2018, 5, 22, 19):]
    ax.plot(df_hydro['Flow'], color='steelblue')
    ax2.plot(df_hydro['Pressure'], color='firebrick')
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.margins(0)
    fig.autofmt_xdate()
    ax.tick_params(axis='y', which='major', direction='out',
                   labelcolor='steelblue', color='steelblue',
                   length=6, width=1)
    ax2.tick_params(axis='y', which='major', direction='out',
                    labelcolor='firebrick', color='firebrick',
                    labelright=True, length=6, width=1)
    ax.set_ylabel('Flow [L/min]', fontsize=16, color='steelblue')
    ax2.set_ylabel('Pressure [MPa]', fontsize=16, color='firebrick')
    plt.show()
    return


def plot_csd_hydro(df_hydro, title='Flow and Pressure', axes=None, flow=False):
    """Simple Flow and Press plot"""
    if not axes:
        fig, ax = plt.subplots()
    else:
        ax = axes
    ax2 = ax.twinx()
    if axes:
        lab = ''
    else:
        lab = 'Flow'
    if flow:
        df_hydro['Flow'].plot(ax=ax, color='steelblue', label=lab)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('ml/min', fontsize=14, color='steelblue')
    else:
        ax.tick_params(labelleft=False, left=False)
    # Only take values past 13:47
    df_hydro = df_hydro[df_hydro.index > datetime(2019, 6, 12, 13, 47)]
    df_hydro['Pressure'].plot(ax=ax2, color='firebrick', label='Pressure')
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('MPa', fontsize=14, color='firebrick')
    if not axes:
        if (df_hydro.index[-1] - df_hydro.index[0]).days == 0:
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.set_xlabel('Time on {}'.format(df_hydro.index.date[0]),
                          fontsize=16, labelpad=5)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=-30,
                     horizontalalignment='left')
            ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                           color='steelblue')
            ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                            color='firebrick')
        fig.legend(loc='lower left')
        fig.suptitle(title, fontsize=16)
        plt.show()
    else:
        if flow:
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
            ax.set_ylabel('ml/min', fontsize=14, labelpad=-50,
                          color='steelblue')
            ax.tick_params(axis='y', which='major', direction='in', pad=-30,
                           labelcolor='steelblue', color='steelblue',
                           labelright=True, length=6, width=1)
            ax.set_yticks(np.arange(25, 225, 25))
            ax.set_yticklabels([str(n) for n in np.arange(25, 225, 25)])
        ax2.tick_params(axis='y', which='major', direction='out', pad=5,
                        labelcolor='firebrick', color='firebrick',
                        length=6, width=1)
    return


def plot_fsb_hydro(df_hydro, title='Flow and Pressure', axes=None, show=False):
    """Simple Flow and Press plot"""
    if not axes:
        fig, ax = plt.subplots(figsize=(15, 5))
    else:
        ax = axes
    ax2 = ax.twinx()
    if axes:
        lab = ''
    else:
        lab = 'Flow'
    # Mask funky signal between tests
    df_hydro_mask = df_hydro[((df_hydro.index > '2020-11-21 9:07') &
                               (df_hydro.index < '2020-11-21 9:58'))]
    df_hydro = df_hydro[~((df_hydro.index > '2020-11-21 9:07') &
                          (df_hydro.index < '2020-11-21 9:58'))]
    df_hydro_mask['Flow'].plot(ax=ax, color='steelblue', alpha=0.15,
                               legend=False, label='')
    df_hydro['Flow'].plot(ax=ax, color='steelblue', label=lab)
    ax.set_ylim(bottom=0)
    df_hydro_mask['Pressure'].plot(ax=ax2, color='firebrick', alpha=0.15,
                                   legend=False, label='')
    df_hydro['Pressure'].plot(ax=ax2, color='firebrick', label='Pressure')
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('MPa', fontsize=14, color='firebrick')
    ax.set_ylabel('L/min', fontsize=14, color='steelblue')
    if not axes:
        if (df_hydro.index[-1] - df_hydro.index[0]).days == 0:
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.set_xlabel('Time on {}'.format(df_hydro.index.date[0]),
                          fontsize=16, labelpad=5)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=-30,
                     horizontalalignment='left')
            ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                           color='steelblue')
            ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                            color='firebrick')
        fig.legend()
        fig.suptitle(title, fontsize=16)
        if show:
            plt.show()
    else:
        ax.set_ylabel('L/min', fontsize=14, color='steelblue')
        ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                       color='steelblue', length=6, width=1)
        ax2.tick_params(axis='y', which='major', direction='out',
                        labelcolor='firebrick', color='firebrick', width=1)
    return [ax, ax2]

def plot_fsb_hydro_panels(df_hydro, title='Flow and Pressure',
                          meq_times=None, big_times=None):
    """
    Plot same data as above, but zoom into each cycle in separate panels

    :param df_hydro:
    :param title:
    :return:
    """
    fig = plt.figure(figsize=(12, 12))
    spec = gridspec.GridSpec(ncols=8, nrows=12, wspace=2, hspace=3)
    fig.suptitle(title, fontsize=18)
    ax1 = fig.add_subplot(spec[:4, :4])
    ax2 = fig.add_subplot(spec[:4, 4:])
    ax3 = fig.add_subplot(spec[4:8, :4])
    ax4 = fig.add_subplot(spec[4:8, 4:])
    ax5 = fig.add_subplot(spec[8:, :4])
    ax6 = fig.add_subplot(spec[8:, 4:])
    cycle_ranges = [
        (datetime(2020, 11, 21, 7, 10), datetime(2020, 11, 21, 7, 40)),
        (datetime(2020, 11, 21, 8, 10), datetime(2020, 11, 21, 8, 40)),
        (datetime(2020, 11, 21, 9, 54), datetime(2020, 11, 21, 10, 30)),
        (datetime(2020, 11, 21, 11, 34), datetime(2020, 11, 21, 12, 17)),
        (datetime(2020, 11, 21, 13, 17), datetime(2020, 11, 21, 14)),
        (datetime(2020, 11, 21, 15), datetime(2020, 11, 21, 16)),
    ]
    # Mask funky signal between tests
    df_hydro_mask = df_hydro[((df_hydro.index > '2020-11-21 9:07') &
                               (df_hydro.index < '2020-11-21 9:58'))]
    df_hydro = df_hydro[~((df_hydro.index > '2020-11-21 9:07') &
                          (df_hydro.index < '2020-11-21 9:58'))]
    time = df_hydro.index.values
    pressure = df_hydro['Pressure']
    flow = df_hydro['Flow']
    for i, (ax, date_range) in enumerate(zip([ax1, ax2, ax3, ax4, ax5, ax6],
                                             cycle_ranges)):
        ax2 = ax.twinx()
        ax.plot(time, flow, color='steelblue', label='Flow')
        ax.set_ylim(bottom=0)
        ax2.plot(time, pressure, color='firebrick', label='')
        ax2.set_ylim(bottom=0)
        if i % 2 == 0:
            ax.set_ylabel('L/min', fontsize=14, color='steelblue')
            ax.tick_params(axis='y', which='major', labelcolor='steelblue',
                           color='steelblue')
            ax2.set_yticklabels([])
        else:
            ax2.set_ylabel('MPa', fontsize=14, color='firebrick')
            ax2.tick_params(axis='y', which='major', labelcolor='firebrick',
                            color='firebrick')
            ax.set_yticklabels([])
        # Plot stems for eq times
        if meq_times:
            ax.stem(big_times, [6 for i in range(len(big_times))],
                    markerfmt='ro', linefmt=':k')
            ax.step(meq_times, np.arange(0, len(meq_times)) * (8 / 32.),
                    color='k')
        # Cycle label
        ax.text(0.9, 0.8, i + 1, fontweight='bold', fontsize=16,
                transform=ax.transAxes)
        ax.set_xlim(date_range)
        ax.set_xlabel('')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), horizontalalignment='center')
        plt.setp(ax.xaxis.get_majorticklabels(), visible=True)
        plt.setp(ax2.xaxis.get_majorticklabels(), visible=True)
    fig.tight_layout()
    plt.show()
    return
