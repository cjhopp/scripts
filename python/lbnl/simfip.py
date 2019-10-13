#!/usr/bin/python

"""SIMFIP I/O and analysis functions"""

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

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


def raw_simfip_correction(files, angles):
    """
    Take a list of files and covert to displacement in the borehole coordinate
    system.

    :param files: List of raw SIMFIP files to read
    :param angles: Tuple of the angles (in radians) for rotation around
        the (X, Y, Z) axes

    :return: pandas DataFrame with three columns; one for each axis X, Y, Z
    """
    df = read_simfip(files)
    FM_df = wings_2_FM(df)
    UTheta = FM_2_UTheta(FM_df)
    U = rotate_UTheta(UTheta, angles=angles)
    Uc = remove_clamps(U, df, angles)
    return Uc

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

def FM_2_UTheta(df):
    """Convert from force-moment matrix to displacement-rotation"""

    # Multiply the rows of the CO1 calibration matrix by the six columns of
    # 3 forces and 3 moments
    return pd.concat(((df * CO1[i]).sum(axis=1) for i in range(6)), axis=1)

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

def rotate_UTheta(df, angles):
    """
    Rotate displacements into borehole coords. For the case of SURF at the
    4850 level, this only requires a rotation around Z (Not sure about this??)

    :param df: pandas DataFrame from FM_2_UTheta
    :param angles: tuple of angles (in radians) around the x, y and z axes
        (thetax, thetay, thetaz)
    :return: pandas DataFrame of rotated displacements in borehole coordinates
    """
    R = make_rotation_mat(angles)
    # Rotate (premultiplied as df are column vectors, I think)
    U = R.dot(df[[0, 1, 2]].values.T)
    return U

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
                                    -6.01e-6 * P /300)).reshape(3, -1)
    # Make rotation mat
    R = make_rotation_mat(angles)
    # Rotate 'em
    U_correction = R.dot(clamp_correct)
    Uc = pd.DataFrame((U - U_correction).T)
    Uc = Uc.set_index(df.index)
    Uc['Pc'] = df['Pz1']
    return Uc

### Plotting functions below here ###

def plot_displacements(Uc, starttime, endtime, decimation=1000):
    # Filter by dates
    filt_Uc = Uc[starttime:endtime]
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
    ax.plot(dtos, Z - Z[0], color='g', label='Uz')
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