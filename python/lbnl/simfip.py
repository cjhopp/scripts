#!/usr/bin/python

"""SIMFIP I/O and analysis functions"""
import plotly

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import colorlover as cl
import chart_studio.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime
from glob import glob
from datetime import datetime
from itertools import cycle
from matplotlib.collections import LineCollection

# 6x6 calibration matrix for SIMFIP
mA1 = np.array([0.021747883, -0.142064305, 0.120457028, 0.120410153,
                -0.14201743, 0.021747883, -0.151614577, -0.05694289, 0.09461837,
                -0.09461837, 0.056949289, 0.151614577, 1.031246708, 1.030927816,
                1.030927816, 1.030927816, 1.030927816, 1.030927816, 0.019263011,
                0.038774139, 0.019522408, -0.019522408, -0.038774139,
                -0.019263011, -0.033663699, 0.000146658, 0.033505759,
                0.033517041, 0.000146658, -0.033663699, -0.005467588,
                0.005466153, -0.005467588, 0.005467588, -0.005466153,
                0.005467588]).reshape((6, 6))

# 6x6 matrix turning Forces/moments to Displacement/rotations
CO1 = np.array([0.001196667, 0, 3.375e-07, 0, 0.0023775, 0, 0, 0.001197333, 0,
                -0.0023725, 0, -0.000128, 2.03333e-05, 0, 0.00001603, 0,
                -0.00014125, 0, 0, -0.000114489, 0, 0.001625984, 0, -9.53402e-5,
                0.000114744, 0, 7.56304e-07, 0, 0.001628276, 0, 0, -7.63944e-7,
                0, -0.000139802, 0, 0.02293582]).reshape((6, 6))

# Borehole inclination and azimuth
csd_boreholes = {'D6': (90.0, 0.0), 'D3': (48.0, 324.0), 'D5': (49.7, 316.9),
                 'D2': (90.0, 0.0), 'D4': (48.0, 324.0), 'D7': (90.0, 0.0),
                 'D1': (90.0, 0.0)}


def raw_simfip_correction(files, angles, clamps=False, resample='S'):
    """
    Take a list of files and covert to displacement in the borehole coordinate
    system.

    :param files: List of raw SIMFIP files to read
    :param angles: Tuple of the angles (in radians) for rotation around
        the (X, Y, Z) axes
    :param clamps: Whether to remove clamp response or not
    :param resample: Resample string passed to pandas resample().
        Defaults to 'S'.
    :return: pandas DataFrame with three columns; one for each axis X, Y, Z
    """
    df = read_simfip(files)
    FM_df = wings_2_FM(df)
    UTheta = FM_2_UTheta(FM_df)
    df = rotate_UTheta(UTheta, df=df, angles=angles)
    if clamps:
        df = remove_clamps(df, angles)
    return df.resample('S').mean()

def read_simfip(files):
    """
    Read a list of raw simfip files to a dataframe

    :param files: List of paths to read
    :return: pandas.DataFrame
    """
    # Use Dask as its significantly faster for this many data points
    dask_df = dd.read_csv(files, encoding='ISO-8859-1')
    df = dask_df.compute()
    df['dt'] = pd.to_datetime(df['Unnamed: 0'], format='%m/%d/%Y %H:%M:%S.%f')
    df = df.set_index('dt')
    df = df.drop(['Unnamed: 0'], axis=1)
    # Sort index as this isn't forced by pandas for DateTime indices
    df = df.sort_index()
    df.index.name = None
    return df

def wings_2_FM(df):
    """
    Convert displacements from the six wings to forces and moments in x, y, z

    :param df: pandas DataFrame read from files
    :return:
    """
    # Multiply the rows of the calibration matrix mA1 by the six wing columns
    # ...then sum across the rows
    FM_df = pd.concat(
        ((df[['A1', 'B1', 'C1', 'D1', 'E1', 'F1']] * mA1[i]).sum(axis=1)
         for i in range(6)), axis=1)
    return FM_df

def FM_2_UTheta(FM_df):
    """Convert from force-moment matrix to displacement-rotation"""

    # Multiply the rows of the CO1 calibration matrix by the six columns of
    # 3 forces and 3 moments
    ut_df = pd.concat(((FM_df * CO1[i]).sum(axis=1) for i in range(6)), axis=1)
    return ut_df

def make_rotation_mat(angles):
    """Helper to make the rotation matrix"""

    # Rotation matrices around each axis
    rX = np.array([1., 0., 0., 0., np.cos(angles[0]), -np.sin(angles[0]),
                   0., np.sin(angles[0]), np.cos(angles[0])]).reshape((3, 3))
    rY = np.array([np.cos(angles[1]), 0., np.sin(angles[1]), 0., 1., 0.,
                   -np.sin(angles[1]), 0., np.cos(angles[1])]).reshape((3, 3))
    rZ = np.array([np.cos(angles[2]), -np.sin(angles[2]), 0., np.sin(angles[2]),
                   np.cos(angles[2]), 0., 0., 0., 1.]).reshape((3, 3))
    # Final rotation matrix
    R = rZ @ rY @ rX
    return R

def rotate_UTheta(ut_df, df, angles):
    """
    Rotate displacements into borehole coords. For the case of SURF at the
    4850 level, this only requires a rotation around Z (Not sure about this??)

    :param ut_df: pandas DataFrame from FM_2_UTheta
    :param df: original pandas DataFrame from read_simfip
    :param angles: tuple of angles (in radians) around the x, y and z axes
        (thetax, thetay, thetaz)
    :return: pandas DataFrame of rotated displacements in borehole coordinates
    """
    R = make_rotation_mat(angles)
    # Rotate (premultiplied as df are column vectors, I think)
    U = R.dot(ut_df[[0, 1, 2]].values.T)
    # Add columns to df for raw displacement (ux...) and rotated displacement
    df['ux'] = ut_df[0]
    df['uy'] = ut_df[1]
    df['uz'] = ut_df[2]
    df['X'] = U[0]
    df['Y'] = U[1]
    df['Z'] = U[2]
    return df

def remove_clamps(df, angles):
    """
    Remove clamp pressure response from displacements
    :param U: ndarray of rows x, y, z in borehole coordinates
    :param df: pandas DataFrame with pressure data in it

    :return: ndarray of corrected U matrix
    """
    # Grab pressures
    P = df['Pz1'].values
    # Correction matrix
    clamp_correct = np.concatenate((-17.2e-6 * P / 300.,
                                    23.e-6 * P / 300.,
                                    -6.01e-6 * P / 300)).reshape(3, -1)
    # Make rotation mat
    R = make_rotation_mat(angles)
    # Rotate 'em
    U_correction = R.dot(clamp_correct)
    df['Xc'] = df['X'] - U_correction[0]
    df['Yc'] = df['Y'] - U_correction[1]
    df['Zc'] = df['Z'] - U_correction[2]
    return df


def rotate_fsb_to_fault(df, strike=52, dip=57):
    """Rotate FSB SIMFIP data into fault coordinates"""
    rotz = np.deg2rad(90 - strike)
    rotx = np.deg2rad(dip)
    # Passive rotation matrices (coord system moves about axes)
    matz = np.array([[np.cos(rotz), np.sin(rotz), 0],
                     [-np.sin(rotz), np.cos(rotz), 0],
                     [0, 0, 1]])
    matx = np.array([[1, 0, 0],
                     [0, np.cos(rotx), np.sin(rotx)],
                     [0, -np.sin(rotx), np.cos(rotx)]])
    simfip_mat = df[['Xc', 'Yc', 'Zc']].values.T
    rotated_simfip = np.matmul(matz, simfip_mat)
    rotated_simfip = np.matmul(matx, rotated_simfip)
    df['Xf'] = rotated_simfip[0, :]
    df['Yf'] = rotated_simfip[1, :]
    df['Zf'] = rotated_simfip[2, :]
    return df


def rotate_fsb_to_borehole(df, well):
    """Rotate FSB SIMFIP data into borehole coordinates"""
    plunge, trend = csd_boreholes[well]
    rotz = np.deg2rad(np.abs(trend - 360))
    rotx = np.deg2rad(90 - plunge)
    # Passive rotation matrices (coord system moves about axes)
    matz = np.array([[np.cos(rotz), np.sin(rotz), 0],
                     [-np.sin(rotz), np.cos(rotz), 0],
                     [0, 0, 1]])
    matx = np.array([[1, 0, 0],
                     [0, np.cos(rotx), np.sin(rotx)],
                     [0, -np.sin(rotx), np.cos(rotx)]])
    simfip_mat = df[['Xc', 'Yc', 'Zc']].values.T
    rotated_simfip = np.matmul(matz, simfip_mat)
    rotated_simfip = np.matmul(matx, rotated_simfip)
    df['Xf'] = rotated_simfip[0, :]
    df['Yf'] = rotated_simfip[1, :]
    df['Zf'] = rotated_simfip[2, :]
    return df


def read_excavation(path):
    """
    Read excavation displacement data for SIMFIP, tiltmeters, borehole
    extensometers

    :param path: Path to file from Florian
    :return:
    """
    # Use Dask as its significantly faster for this many data points
    dask_df = dd.read_csv(path)
    df = dask_df.compute()
    try:
        df['dt'] = pd.to_datetime(df['TIME (UTC+2)'], format='%d.%m.%y %H:%M:%S')
    except KeyError:
        df['dt'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S')
    df = df.set_index('dt')
    # df = df.drop(['TIME (UTC+2)'], axis=1)
    # Sort index as this isn't forced by pandas for DateTime indices
    df = df.sort_index()
    df.index.name = None
    df.index = df.index.tz_localize('Etc/GMT+2')
    df.index = df.index.tz_convert('UTC')
    df.index = df.index.tz_convert(None)
    # Assuming these data are corrected for clamp effects now??
    df.rename(columns={'dispE': 'Xc', 'dispN': 'Yc', 'dispUp': 'Zc'},
              inplace=True)
    return df


def read_collab(path):
    """
    Read excavation displacement data for SIMFIP at surf 4850L

    :param path: Path to file from Yves
    :return:
    """
    col_names = ['date', 'time', 'Pz1', 'Pz2', 'S1Yates', 'S1Top',
                 'S1WellAxial', 'S1YatesE', 'S1TopE', 'S1WellAxialE',
                 'S1YatesIE', 'S1TopIE', 'S1WellAxialIE', 'S2YatesIE',
                 'S2TopIE', 'S2WellAxialIE', 'blank']
    # Use Dask as its significantly faster for this many data points
    df = pd.read_csv(path, sep=' ', names=col_names, header=0)
    df['dt'] = pd.to_datetime(df['date'], format='%d/%b/%YT%H:%M:%S.%f')
    df = df.set_index('dt')
    df = df.drop(['date'], axis=1)
    # Sort index as this isn't forced by pandas for DateTime indices
    df = df.sort_index()
    df.index = df.index.tz_localize('US/Mountain')
    df.index = df.index.tz_convert('UTC')
    df.index = df.index.tz_convert(None)
    df.index.name = None
    # Seems like Yates and Axial are swapped??
    df.rename(columns={
        'S2TopIE': 'P Top', 'S2YatesIE': 'P Yates', 'S2WellAxialIE': 'P Axial',
        'S1TopIE': 'I Top', 'S1YatesIE': 'I Yates', 'S1WellAxialIE': 'I Axial'},
        inplace=True)
    return df


def read_3D(data_dir):
    """
    Read the data from the FSB 3D string

    :param data_dir: 3D string data directory
    :return:
    """
    flz = glob('{}/*.csv'.format(data_dir))
    df = pd.DataFrame()
    for fl in flz:
        nm = 'T{}Z{}'.format(fl[-7], fl[-5])
        tdf = pd.read_csv(fl, header=0, names=['time', '{}E'.format(nm),
                                     '{}N'.format(nm), '{}U'.format(nm)])
        tdf['dt'] = pd.to_datetime(tdf['time'], format='%d-%b-%Y %H:%M:%S')
        tdf = tdf.set_index('dt')
        tdf = tdf.drop(['time'], axis=1)
        df = pd.concat([df, tdf])
    return df.sort_index()

################### Plotting functions below here #######################

def plot_drive2production(df_simfip, DSS_dict, DAS_dict):
    """
    Plot SIMFIP at I and P versus multiple fractures at OT from fiber
    """
    fig = plt.figure(figsize=(7, 10))
    spec = gridspec.GridSpec(ncols=1, nrows=13, figure=fig)
    ax_hydro = fig.add_subplot(spec[:1, :])
    ax_I = fig.add_subplot(spec[1:4, :], sharex=ax_hydro)
    ax_fiber = fig.add_subplot(spec[4:7, :], sharex=ax_hydro)#, sharey=ax_I)
    ax_P = fig.add_subplot(spec[7:10, :], sharex=ax_hydro)#, sharey=ax_I)
    ax_comp = fig.add_subplot(spec[10:, :], sharex=ax_hydro)#, sharey=ax_I)
    pres = df_simfip['Pz1'] / 145.038  # psi to MPa
    # Plot hydraulics
    ax_hydro.plot(df_simfip.index, pres, label='Pump pressure',
                  color='firebrick')
    ax_hydro.legend()
    # Add total shear SIMFIP column
    df_simfip['P shear'] = np.sqrt(df_simfip['P Yates']**2 +
                                   df_simfip['P Top']**2)
    # Plot SIMFIP
    (df_simfip[['I Yates', 'I Top', 'I Axial']] * 1e6).plot(ax=ax_I)
    (df_simfip[['P Yates', 'P Top', 'P Axial']] * 1e6).plot(ax=ax_P)
    # Plot fibers
    ax_fiber.plot(DSS_dict['times'], DSS_dict['44 m'], label='DSS: 44 m',
                  color='steelblue')
    ax_fiber.plot(DSS_dict['times'], DSS_dict['47 m'], label='DSS: 47 m',
                  color='dodgerblue')
    ax_fiber.plot(DAS_dict['times'], DAS_dict['44 m'], label='DAS: 44 m',
                  color='mediumpurple')
    ax_fiber.plot(DAS_dict['times'], DAS_dict['47 m'], label='DAS: 47 m',
                  color='indigo')
    handles, labels = ax_I.get_legend_handles_labels()
    ax_I.legend(handles, ['X', 'Y', 'Z'])
    handles, labels = ax_I.get_legend_handles_labels()
    ax_P.legend(handles, ['X', 'Y', 'Z'])
    # Comparison
    comp_times = DSS_dict['times'][np.where(
        (np.array(DSS_dict['times']) > df_simfip.index[0])
        & (np.array(DSS_dict['times']) < df_simfip.index[-1]))]
    norm_dss = DSS_dict['47 m'][np.where(
        (np.array(DSS_dict['times']) > df_simfip.index[0])
        & (np.array(DSS_dict['times']) < df_simfip.index[-1]))]
    norm_dss -= norm_dss[0]
    norm_dss /= np.max(np.abs(norm_dss))
    norm_das = DAS_dict['47 m'][np.where(
        (np.array(DAS_dict['times']) > df_simfip.index[0])
        & (np.array(DAS_dict['times']) < df_simfip.index[-1]))]
    norm_das -= norm_das[0]
    norm_das /= np.max(np.abs(norm_das))
    norm_P_shear = df_simfip['P shear'] / np.max(np.abs(df_simfip['P shear']))
    ax_comp.plot(comp_times, norm_dss, label='DSS: 47 m',
                  color='dodgerblue')
    ax_comp.plot(comp_times, norm_das, label='DAS: 47 m',
                  color='indigo')
    ax_comp.plot(
        df_simfip.index, -norm_P_shear, label='SIMFIP', color='goldenrod')
    # Shut in time
    ax_P.axvline(datetime(2018, 5, 24, 22, 51), linestyle=':', color='gray',
                 label='Shut-in')
    ax_I.axvline(datetime(2018, 5, 24, 22, 51), linestyle=':', color='gray')
    ax_fiber.axvline(datetime(2018, 5, 24, 22, 51), linestyle=':', color='gray')
    ax_comp.axvline(datetime(2018, 5, 24, 22, 51), linestyle=':', color='gray')
    # Formatting
    ax_P.margins(0.)
    ax_hydro.set_ylabel('MPa')
    ax_fiber.legend(loc=1)
    ax_comp.legend(loc=1)
    ax_I.set_ylabel('Microns')
    ax_P.set_ylabel('Microns')
    ax_fiber.set_ylabel('Microns')
    ax_fiber.set_ylim([-20, 20])
    ax_P.set_xlim([datetime(2018, 5, 24, 22), datetime(2018, 5, 25, 2)])
    fig.autofmt_xdate()
    plt.show()
    return


def plot_overview(df, starttime=UTCDateTime(2018, 5, 22, 10, 48),
                  endtime=UTCDateTime(2018, 5, 22, 18), corrected=True):
    """
    Three-panel plot with flow on top, packer/interval pressures in middle
    and xyz on bottom

    :param df: Pandas dataframe containing the raw header values and the XYZ
        columns from rotate_UTheta()
    :param starttime: Start time of the plot
    :param endtime: End time of the plot
    :param corrected: Plot the clamp-removed, rotated streams

    :return:
    """
    if corrected:
        heads = ('Xc', 'Yc', 'Zc')
    else:
        heads = ('X', 'Y', 'Z')
    date_formatter = mdates.DateFormatter('%b-%d %H:%M')
    df = df[starttime.datetime:endtime.datetime]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # These files are wack...
    axes[0].plot(df['Pt1'], label='Flow (mL/min)') # Header wrong
    axes[1].plot(df['Pz1'], label='Interval P')
    axes[1].plot(df['Tb2'], label='Upper Packer P') # Header wrong
    axes[1].plot(df['Pb2'], label='Bottom Packer P') # Header wrong
    axes[2].plot(df[heads[0]] - df[heads[0]][0], label='X-Yates')
    axes[2].plot(df[heads[1]] - df[heads[1]][0], label='Y-Top')
    axes[2].plot(df[heads[2]] - df[heads[2]][0], label='Z')
    axes[0].set_ylabel('Flow (mL/min)', fontsize=16)
    axes[1].set_ylabel('Pressure (psi)', fontsize=16)
    axes[2].set_ylabel('Displacement (microns)', fontsize=16)
    axes[2].set_xlabel('Date', fontsize=16)
    axes[0].legend(fontsize=12, loc=1)
    axes[1].legend(fontsize=12, loc=4)
    axes[2].legend(fontsize=12, loc=3)
    axes[2].xaxis.set_major_formatter(date_formatter)
    axes[2].tick_params(axis='x', which='major', labelsize=12)
    tstamp = df.index[0]
    axes[0].set_title('{}-{}-{}'.format(tstamp.year, tstamp.month, tstamp.day),
                      fontsize=22)
    return


def plot_displacement_components(df, starttime=datetime(2018, 5, 22, 11, 24),
                                 endtime=datetime(2018, 5, 22, 12, 36),
                                 new_axes=None, rotated=True, location='fsb',
                                 plot_clamp_curves=False, remove_clamps=False):
    """
    Plot X, Y and Z on same  or separate axes and compare to clamp effects
    if desired

    :param df: dataframe with X, Y, Z columns appended
    :param starttime: Plot start time
    :param endtime: Plot end time
    :param new_axes: Axes instance to plot all traces into
    :param rotated: Are the streams rotated into cardinal directions?
    :param plot_clamp_curves: Whether to plot Yves clamping curves or not
    :param remove_clamps: 'High' or 'Low' clamp curve to subtract

    :return:
    """
    colors = cycle(sns.color_palette("Paired"))
    # Set out date formatter
    date_formatter = mdates.DateFormatter('%b-%d %H:%M')
    df = df[starttime:endtime]
    if not new_axes:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    if location == 'collab':
        # headers = ('I Top', 'I Yates', 'I Axial', 'P Top', 'P Yates', 'P Axial')
        headers = ('P Top', 'P Yates', 'P Axial')
    elif rotated and not remove_clamps:
        headers = ('Xc', 'Yc', 'Zc')
    elif rotated:
        headers = ('X', 'Y', 'Z')
    else:
        headers = ('ux', 'uy', 'uz')
    if not new_axes:
        for i, header in enumerate(headers):
            new_axes.plot(df[header] - df[header][0],
                          label='{}'.format(header),
                          color=next(colors), linewidth=1.5)
    else:
        # Plot measurements first
        for i, header in enumerate(headers):
            new_axes.plot(df[header] - df[header][0],
                          label='{}'.format(header),
                          color=next(colors), linewidth=1.5)
    # If plotting possible clamp effects
    if plot_clamp_curves and not remove_clamps and not rotated:
        axes[0].plot(-0.00028805 + (-7.8e-6 * df['Pz1'] / 300.),
                     color='steelblue', alpha=0.3,
                     label='Ux clamp low')
        axes[0].plot(-0.00025719 + (-17.2e-6 * df['Pz1'] / 300.),
                     color='steelblue', alpha=0.5,
                     label='Ux clamp high')
        axes[1].plot(-0.00035549 + (10.e-6 * df['Pz1'] / 300.),
                     color='orange', alpha=0.3,
                     label='Uy clamp low')
        axes[1].plot(-0.00039795 + (23.e-6 * df['Pz1'] / 300.),
                     color='orange', alpha=0.5,
                     label='Uy clamp high')
        axes[2].plot(0.0000152 + (-2.31e-6 * df['Pz1'] / 300.),
                     color='green', alpha=0.3,
                     label='Uz clamp low')
        axes[2].plot(0.000027204 + (-6.01e-6 * df['Pz1'] / 300.),
                     color='green', alpha=0.5,
                     label='Uz clamp high')
    elif remove_clamps:
        axes[0].plot(df[headers[0]] - df[headers[0]][0] - (-17.2e-6 * df['Pz1']
                                                           / 300.),
                     color='steelblue', alpha=0.7,
                     label='{} - clamp'.format(headers[0]))
        axes[1].plot(df[headers[1]] - df[headers[1]][0] - (23.e-6 * df['Pz1']
                                                           / 300.),
                     color='orange', alpha=0.7,
                     label='{} - clamp'.format(headers[1]))
        axes[2].plot(df[headers[2]] - df[headers[2]][0] - (-6.01e-6 * df['Pz1']
                                                           / 300.),
                     color='green', alpha=0.7,
                     label='{} - clamp'.format(headers[2]))
    # Format it up
    if new_axes:
        new_axes.set_ylabel('Displacement (microns)', fontsize=16)
        new_axes.text(0.05, 0.1, horizontalalignment='left', s='SIMFIP',
                      verticalalignment='center', transform=new_axes.transAxes,
                      fontsize=16)
        new_axes.legend()
        new_axes.margins(x=0)
    else:
        axes[1].set_ylabel('Displacement (microns)', fontsize=20)
        axes[2].set_xlabel('Time', fontsize=16)
        axes[0].legend(fontsize=12, loc=1)
        axes[1].legend(fontsize=12, loc=1)
        axes[2].legend(fontsize=12, loc=1)
        axes[2].xaxis.set_major_formatter(date_formatter)
        axes[2].tick_params(axis='x', which='major', labelsize=12)
        tstamp = df.index[0]
        axes[0].set_title('{}-{}-{}'.format(tstamp.year, tstamp.month, tstamp.day),
                          fontsize=22)
        axes[0].margins(x=0)
    return


def plot_displacement_pressure(df, starttime, endtime):
    """
    Plot pressure, and displacement vs pressure

    TODO THIS THING AND THE FOLLOWING FUNC ARE NEARLY IDENTICAL. MERGE THEM.

    :param df: DataFrame containing corrected displacements
    :param starttime: Starttime of the plot
    :param endtime: Endtime of the plot
    :return:
    """
    fig = plt.figure(figsize=(9, 8))
    ax_P = fig.add_subplot(221)
    ax_X = fig.add_subplot(222)
    ax_Y = fig.add_subplot(223, sharex=ax_X, sharey=ax_X)
    ax_Z = fig.add_subplot(224, sharex=ax_X, sharey=ax_X)
    # Filter for time
    df = df[starttime.datetime:endtime.datetime]
    # Make date array
    mpl_times = mdates.date2num(df.index.to_pydatetime())
    # Make color array
    norm = plt.Normalize(mpl_times.min(), mpl_times.max())
    # Plot the pressure with continuous color
    # (Discrete colormap would require user input)
    points = np.array([mpl_times, df['Pz1']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='cividis', norm=norm)
    lc.set_array(mpl_times)
    lc.set_linewidth(1.)
    line = ax_P.add_collection(lc)
    ## Now X displacement
    pts_X = np.array([df['Pz1'], df['Xc'] - df['Xc'][0]]).T.reshape(-1, 1, 2)
    segs_X = np.concatenate([pts_X[:-1], pts_X[1:]], axis=1)
    lc_X = LineCollection(segs_X, cmap='cividis', norm=norm)
    line_X = ax_X.add_collection(lc_X)
    lc_X.set_array(mpl_times)
    lc_X.set_linewidth(1.)
    ## Now Y displacement
    pts_Y = np.array([df['Pz1'], df['Yc'] - df['Yc'][0]]).T.reshape(-1, 1, 2)
    segs_Y = np.concatenate([pts_Y[:-1], pts_Y[1:]], axis=1)
    lc_Y = LineCollection(segs_Y, cmap='cividis', norm=norm)
    line_Y = ax_Y.add_collection(lc_Y)
    lc_Y.set_array(mpl_times)
    lc_Y.set_linewidth(1.)
    ## Now Z displacement
    pts_Z = np.array([df['Pz1'], df['Zc'] - df['Zc'][0]]).T.reshape(-1, 1, 2)
    segs_Z = np.concatenate([pts_Z[:-1], pts_Z[1:]], axis=1)
    lc_Z = LineCollection(segs_Z, cmap='cividis', norm=norm)
    line_Z = ax_Z.add_collection(lc_Z)
    lc_Z.set_array(mpl_times)
    lc_Z.set_linewidth(1.)
    ## Formatting
    # ax_P
    ax_P.set_title('Pressure')
    ax_P.set_xlabel('Time')
    ax_P.set_ylabel('Pressure (psi)')
    ax_P.set_xlim([mpl_times.min(), mpl_times.max()])
    ax_P.set_ylim([df['Pz1'].min(), df['Pz1'].max()])
    # ax_X
    ax_X.set_title('Yates (X)')
    data = df['Xc'] - df['Xc'][0]
    ax_X.set_xlim([df['Pz1'].min(), df['Pz1'].max()])
    ax_X.set_ylim([data.min(), data.max()])
    plt.setp(ax_X.get_xticklabels(), visible=False)
    ax_X.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax_X.set_ylabel('Displacement (m)')
    # ax_Y
    ax_Y.set_title('Top (Y)')
    ax_Y.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax_Y.set_ylabel('Displacement (m)')
    ax_Y.set_xlabel('Pressure (psi)')
    # ax_Z
    ax_Z.set_title('Borehole axis (Z)')
    plt.setp(ax_Z.get_yticklabels(), visible=False)
    ax_Z.set_xlabel('Pressure (psi)')
    # Axis formatting
    ax_P.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9,
                        wspace=0.25, hspace=0.25)
    # Make colorbar
    cax = plt.axes([0.87, 0.1, 0.04, 0.8])
    cbar = fig.colorbar(line, cax=cax)
    # Change colorbar ticks
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    return


def plot_displacement_planes(df, starttime=UTCDateTime(2018, 5, 22, 11, 55),
                             endtime=UTCDateTime(2018, 5, 22, 12, 30)):
    """
    Plot 3D displacement on 3, 2D planes colored by time

    :param df: DataFrame with corrected displacements under headers Xc Yc Zc
    :param starttime: Time at start of plot
    :param endtime: Time at end of plot

    :return:
    """
    # Will use same layout as P-D plot above, including reference pressure
    # plot in top left
    fig = plt.figure(figsize=(9, 8))
    ax_P = fig.add_subplot(221)
    ax_XY = fig.add_subplot(222)
    ax_ZX = fig.add_subplot(223, sharex=ax_XY, sharey=ax_XY)
    ax_ZY = fig.add_subplot(224, sharex=ax_XY, sharey=ax_XY)
    # Filter for time
    df = df[starttime.datetime:endtime.datetime]
    # Make date array
    mpl_times = mdates.date2num(df.index.to_pydatetime())
    # Make color array
    norm = plt.Normalize(mpl_times.min(), mpl_times.max())
    # Plot the pressure with continuous color
    # (Discrete colormap would require user input)
    points = np.array([mpl_times, df['Pz1']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='cividis', norm=norm)
    lc.set_array(mpl_times)
    lc.set_linewidth(2.)
    line = ax_P.add_collection(lc)
    # Yates-Top plane
    pts_XY = np.array([df['Xc'] - df['Xc'][0],
                      df['Yc'] - df['Yc'][0]]).T.reshape(-1, 1, 2)
    segs_XY = np.concatenate([pts_XY[:-1], pts_XY[1:]], axis=1)
    lc_XY = LineCollection(segs_XY, cmap='cividis', norm=norm)
    line_XY = ax_XY.add_collection(lc_XY)
    lc_XY.set_array(mpl_times)
    lc_XY.set_linewidth(2.)
    ## Now Y displacement
    pts_ZX = np.array([df['Xc'] - df['Xc'][0],
                       df['Zc'] - df['Zc'][0]]).T.reshape(-1, 1, 2)
    segs_ZX = np.concatenate([pts_ZX[:-1], pts_ZX[1:]], axis=1)
    lc_ZX = LineCollection(segs_ZX, cmap='cividis', norm=norm)
    line_ZX = ax_ZX.add_collection(lc_ZX)
    lc_ZX.set_array(mpl_times)
    lc_ZX.set_linewidth(2.)
    ## Now Z displacement
    pts_ZY = np.array([df['Zc'] - df['Zc'][0],
                       df['Yc'] - df['Yc'][0]]).T.reshape(-1, 1, 2)
    segs_ZY = np.concatenate([pts_ZY[:-1], pts_ZY[1:]], axis=1)
    lc_ZY = LineCollection(segs_ZY, cmap='cividis', norm=norm)
    line_ZY = ax_ZY.add_collection(lc_ZY)
    lc_ZY.set_array(mpl_times)
    lc_ZY.set_linewidth(2.)
    ## Formatting
    # ax_P
    ax_P.set_title('Pressure')
    ax_P.set_xlabel('Time')
    ax_P.set_ylabel('Pressure (psi)')
    ax_P.set_xlim([mpl_times.min(), mpl_times.max()])
    ax_P.set_ylim([df['Pz1'].min(), df['Pz1'].max()])
    # ax_X
    ax_XY.set_title('Yates-Up plane')
    datz = df['Xc'] - df['Xc'][0]
    ax_XY.set_xlim([-datz.max(), datz.max()])
    ax_XY.set_ylim([-datz.max(), datz.max()])
    plt.setp(ax_XY.get_xticklabels(), visible=False)
    ax_XY.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax_XY.set_ylabel('Displacement (m)')
    # ax_Y
    ax_ZX.set_title('Yates-Z plane')
    ax_ZX.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax_ZX.set_ylabel('Displacement (m)')
    ax_ZX.set_xlabel('Displacement (m)')
    # ax_Z
    ax_ZY.set_title('Z-Up plane')
    plt.setp(ax_ZY.get_yticklabels(), visible=False)
    ax_ZY.set_xlabel('Displacement (m)')
    # Axis formatting
    ax_P.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9,
                        wspace=0.25, hspace=0.25)
    # Make colorbar
    cax = plt.axes([0.87, 0.1, 0.04, 0.8])
    cbar = fig.colorbar(line, cax=cax)
    # Change colorbar ticks
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    return


def plot_3D_displacement(df, starttime=UTCDateTime(2018, 5, 22, 11, 55),
                         endtime=UTCDateTime(2018, 5, 22, 12, 30),
                         outfile='SIMFIP_test'):
    """
    Make a plotly figure of 3D SIMFIP displacement

    :param df: DataFrame with corrected displacements under headers Xc Yc Zc
    :param starttime: Time at start of plot
    :param endtime: Time at end of plot

    :return:
    """
    # Filter for time
    df = df[starttime.datetime:endtime.datetime]
    # Make date array
    mpl_times = mdates.date2num(df.index.to_pydatetime())
    tickvals = np.linspace(min(mpl_times), max(mpl_times), 10)
    ticktext = [mdates.num2date(t).strftime('%H:%M') for t in tickvals]
    # Plot colored points first. Maybe line segments later
    fig = go.Figure(data=go.Scatter3d(x=df['Xc'] - df['Xc'][0],
                                      y=df['Yc'] - df['Yc'][0],
                                      z=df['Zc'] - df['Zc'][0],
                                      marker=dict(size=2., color=mpl_times,
                                                  colorbar=dict(
                                                      title=dict(text='Time',
                                                                 font=dict(size=18)),
                                                      x=-0.2,
                                                      ticktext=ticktext,
                                                      tickvals=tickvals),
                                                  colorscale='cividis'),
                                      line=dict(color='darkblue', width=2)))
    fig.update_layout(width=800, height=800, autosize=False,
                      scene=dict(camera=dict(up=dict(x=0, y=1, z=0),
                                             eye=dict(x=0, y=1.0707, z=3.)),
                                 aspectmode='data'),
                      title=outfile)
    py.plot(fig, filename=outfile)
    return