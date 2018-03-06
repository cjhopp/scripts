#!/usr/bin/python

"""
Plotting functions for stress inversion results
"""
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

def plot_stress_time(outdir):
    """
    Plot all output from Rstress in an output directory
    :param outdir:
    :return:
    """
    outfiles = glob('{}/*.2dparameters.dat'.format(outdir))
    outfiles.sort()
    axes_dict = {'S1': {'Phi': [], 'Theta': []},
                 'S2': {'Phi': [], 'Theta': []},
                 'S3': {'Phi': [], 'Theta': []}}
    for file in outfiles:
        file_dict = {'S1': {'Phi': None, 'Theta': None},
                     'S2': {'Phi': None, 'Theta': None},
                     'S3': {'Phi': None, 'Theta': None}}
        with open(file, 'r') as f:
            for ln in f:
                ln = ln.split(',')
                param = ln[0].split(':')
                if param[0] in file_dict:
                    file_dict[param[0]][param[1]] = float(ln[1])
        for stress_ax, vect_dict in file_dict.items():
            if vect_dict['Theta'] > 90.:
                axes_dict[stress_ax]['Theta'].append(180. - vect_dict['Theta'])
                if vect_dict['Phi'] > 0.:
                    axes_dict[stress_ax]['Phi'].append(vect_dict['Phi'] - 180.)
                else:
                    axes_dict[stress_ax]['Phi'].append(vect_dict['Phi'] + 180.)
            else:
                axes_dict[stress_ax]['Theta'].append(vect_dict['Theta'])
                axes_dict[stress_ax]['Phi'].append(vect_dict['Phi'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.plot(axes_dict['S1']['Phi'], color='red', label='S1: Phi')
    ax1.plot(axes_dict['S2']['Phi'], color='green', label='S2: Phi')
    ax1.plot(axes_dict['S3']['Phi'], color='blue', label='S3: Phi')
    ax1.legend()
    ax2.plot(axes_dict['S1']['Theta'], color='pink', label='S1: Theta')
    ax2.plot(axes_dict['S2']['Theta'], color='lightblue', label='S2: Theta')
    ax2.plot(axes_dict['S3']['Theta'], color='lightgreen', label='S3: Theta')
    ax2.legend()
    plt.show()
    return

def plot_mohr_circle(sig1, sig2, sig3, t0, mu):
    """
    Simple schematic plotting of Mohr circles
    :param sig1:
    :param sig2:
    :param sig3:
    :param t0: Cohesion
    :return:
    """
    fig, ax = plt.subplots()
    rad1 = (sig1 - sig3) / 2.
    rad2 = (sig1 - sig2) / 2.
    rad3 = (sig2 - sig3) / 2.
    c1 = plt.Circle((sig3 + rad1, 0), radius=rad1, linestyle='solid',
                    linewidth=1.0, edgecolor='black', facecolor='lightblue')
    c2 = plt.Circle((sig2 + rad2, 0), radius=rad2, facecolor='white',
                    linestyle='solid', edgecolor='black')
    c3 = plt.Circle((sig3 + rad3, 0), radius=rad3, facecolor='white',
                    linestyle='solid', linewidth=1.0, edgecolor='black')
    ax.add_artist(c1)
    ax.add_artist(c2)
    ax.add_artist(c3)
    ax.scatter(sig1, 0, color='red', zorder=3)
    ax.scatter(sig2, 0, color='green', zorder=3)
    ax.scatter(sig3, 0, color='blue', zorder=3)
    ax.annotate('$\sigma_1$', xy=(sig1, 0), xytext=(sig1-1.5, 0.5), fontsize=12)
    ax.annotate('$\sigma_2$', xy=(sig2, 0), xytext=(sig2-1.5, 0.5), fontsize=12)
    ax.annotate('$\sigma_3$', xy=(sig3, 0), xytext=(sig3-1.5, 0.5), fontsize=12)
    ax.set_aspect('equal')
    ax.set_ylim([0, sig3 + 5])
    ax.set_xlim([0, sig1 + 5])
    fig.tight_layout()
    fig.show()
    return