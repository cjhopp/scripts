#!/usr/bin/python

"""SIMFIP I/O and analysis functions"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import UTCDateTime

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


def raw_simfip_correction(files, angles, clamps=False):
    """
    Take a list of files and covert to displacement in the borehole coordinate
    system.

    :param files: List of raw SIMFIP files to read
    :param angles: Tuple of the angles (in radians) for rotation around
        the (X, Y, Z) axes
    :param clamps: Whether to remove clamp response or not

    :return: pandas DataFrame with three columns; one for each axis X, Y, Z
    """
    df = read_simfip(files)
    FM_df = wings_2_FM(df)
    UTheta = FM_2_UTheta(FM_df)
    U = rotate_UTheta(UTheta, df=df, angles=angles)
    if clamps:
        Uc = remove_clamps(U, df, angles)
        return Uc
    else:
        return U

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

def remove_clamps(U, df, angles):
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
    Uc = pd.DataFrame((U - U_correction).T)
    Uc = Uc.set_index(df.index)
    Uc['Pc'] = df['Pz1']
    return Uc

################### Plotting functions below here #######################

def plot_raw_overview(df, starttime=UTCDateTime(2018, 5, 22, 10, 48),
                      endtime=UTCDateTime(2018, 5, 22, 18)):
    """
    Three-panel plot with flow on top, packer/interval pressures in middle
    and xyz on bottom

    :param df: Pandas dataframe containing the raw header values and the XYZ
        columns from rotate_UTheta()
    :param starttime: Start time of the plot
    :param endtime: End time of the plot
    :return:
    """
    date_formatter = mdates.DateFormatter('%H:%M')
    df = df[starttime.datetime:endtime.datetime]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # These files are wack...
    axes[0].plot(df['Pt1'], label='Flow (mL/min)') # Header wrong
    axes[1].plot(df['Pz1'], label='Interval P')
    axes[1].plot(df['Tb2'], label='Upper Packer P') # Header wrong
    axes[1].plot(df['Pb2'], label='Bottom Packer P') # Header wrong
    axes[2].plot(df['X'] - df['X'][0], label='X-Yates')
    axes[2].plot(df['Y'] - df['Y'][0], label='Y-Top')
    axes[2].plot(df['Z'] - df['Z'][0], label='Z')
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


def plot_displacement_components(df, starttime=UTCDateTime(2018, 5, 22, 11, 24),
                                 endtime=UTCDateTime(2018, 5, 22, 12, 36),
                                 plot_clamp_curves=False):
    """
    Plot X, Y and Z on separate axes and compare to clamp effects if desired

    :param df: dataframe with X, Y, Z colums appended
    :param plot_clamp_curves: Whether to plot Yves clamping curves or not
    :return:
    """
    # Set out date formatter
    date_formatter = mdates.DateFormatter('%H:%M')
    df = df[starttime.datetime:endtime.datetime]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # Plot measurements first
    axes[0].plot(df['ux'] - df['ux'][0], label='Ux measurement',
                 color='steelblue', linewidth=1.5)
    axes[1].plot(df['uy'] - df['uy'][0], label='Uy measurement', color='orange',
                 linewidth=1.5)
    axes[2].plot(df['uz'] - df['uz'][0], label='Uz measurement', color='green',
                 linewidth=1.5)
    # If plotting possible clamp effects
    if plot_clamp_curves:
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
    # Format it up
    # axes[0].set_ylabel('Flow (mL/min)', fontsize=16)
    axes[1].set_ylabel('Displacement (microns)', fontsize=20)
    # axes[2].set_ylabel('Displacement (microns)', fontsize=16)
    axes[2].set_xlabel('Date', fontsize=16)
    axes[0].legend(fontsize=12, loc=1)
    axes[1].legend(fontsize=12, loc=1)
    axes[2].legend(fontsize=12, loc=1)
    axes[2].xaxis.set_major_formatter(date_formatter)
    axes[2].tick_params(axis='x', which='major', labelsize=12)
    tstamp = df.index[0]
    axes[0].set_title('{}-{}-{}'.format(tstamp.year, tstamp.month, tstamp.day),
                      fontsize=22)
    return


def plot_displacements(Uc, starttime=None, endtime=None, decimation=1000):
    """
    # TODO Needs review and change to use df with everything and correct cols
    Plot all three displacement components with interval pressure

    :param Uc: dataframe??
    :param starttime: Start time of plot
    :param endtime: End time of plot
    :param decimation: Decimation so plots are drawn more quickly
    :return:
    """
    # Filter by dates
    if starttime and endtime:
        filt_Uc = Uc[starttime:endtime]
    else:
        filt_Uc = Uc
    # Grab raw datetime array
    dtos = filt_Uc.index.to_pydatetime()[::decimation]
    # X, Y, Z as seperate arrays
    X = filt_Uc[0].values[::decimation]
    Y = filt_Uc[1].values[::decimation]
    Z = filt_Uc[2].values[::decimation]
    P = filt_Uc['Pc'].values[::decimation] * 0.00689476 # to MPa (real unit)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dtos, X - X[0], color='b', label='Ux (Yates)')
    ax.plot(dtos, Y - Y[0], color='r', label='Uy')
    ax.plot(dtos, (Z - Z[0]) * 10, color='g', label='Uz*10')
    hands, labs = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.plot(dtos, P, color='gray', label='Chamber P')
    hands2, labs2 = ax2.get_legend_handles_labels()
    labs.extend(labs2)
    hands.extend(hands2)
    ax.set_ylabel('Displacement (m?)', fontsize=16)
    ax2.set_ylabel('MPa', fontsize=16)
    ax.set_title('Corrected displacements with pressure', fontsize=20)
    plt.legend(handles=hands, labels=labs, loc=2)
    fig.autofmt_xdate()
    ax.set_xlabel('Date', fontsize=16)
    plt.tight_layout()
    plt.show()
    return ax