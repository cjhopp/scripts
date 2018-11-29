#!/usr/bin/python

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from matplotlib.pyplot import GridSpec
from itertools import cycle
from plot_well_data import plot_well_seismicity

def parse_arnold_grid(file):
    """Return the vectors that define the phi, theta grid"""
    with open(file, 'r') as f:
        lines = []
        for ln in f:
            lines.append(ln.strip('\n'))
    phivec = np.array([float(ln) for ln in lines[6:57]])
    thetavec = np.array([float(ln) for ln in lines[58:]])
    return phivec, thetavec

def parse_arnold_params(files):
    """Parse the 1d and 2d parameter files to dictionary"""
    strs_params = {}
    for file in files:
        with open(file, 'r') as f:
            next(f)
            for ln in f:
                ln.rstrip('\n')
                line = ln.split(',')
                if len(line) == 4:
                    strs_params[line[0]] = {
                        'mean': float(line[1]), 'map': float(line[2])
                    }
                elif len(line) == 6:
                    strs_params[line[0]] = {
                        'mean': float(line[1]), 'map': float(line[2]),
                        'median': float(line[3]), 'X10': float(line[4]),
                        'X90': float(line[5])
                    }
    return strs_params

def plot_arnold_density(outdir, clust_name, ax=None, show=False):
    """
    Porting the contour plotting workflow from Richard's R code

    :param outdir: Output directory for Arnold-Townend inversion
    :param clust_name: String of cluster name to plot
    :param ax: matplotlib Axes object to plot onto. This should already
        be defined as polar projection.
    :param show: Automatically show this plot once we're done?
    :return: matplotlib Axes object
    """
    froot = '/'.join([outdir, clust_name])
    grid_f = '{}.{}.dat'.format(froot, 's123grid')
    param_files = glob('{}.*{}.dat'.format(froot, 'dparameters'))
    phivec, thetavec = parse_arnold_grid(grid_f)
    strs_params = parse_arnold_params(param_files)
    # Read in the density estimates for the cells of the grid defined by
    # thetavec and phivec
    z1 = np.loadtxt('{}.{}.dat'.format(froot, 's1density'), delimiter=',')
    z2 = np.loadtxt('{}.{}.dat'.format(froot, 's2density'), delimiter=',')
    z3 = np.loadtxt('{}.{}.dat'.format(froot, 's3density'), delimiter=',')
    # Now need to generate contour plot on passes Axes?
    # Need to convert the z1 values to degrees somehow?
    if not ax:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    # Contoured sigmas
    greens = sns.cubehelix_palette(5, light=0.6, hue=1., dark=0.5, start=1.9,
                                   rot=0.1, as_cmap=True, gamma=1.3)
    blues = sns.cubehelix_palette(5, light=0.6, hue=1., dark=0.5, start=2.7,
                                  rot=0.1, as_cmap=True, gamma=1.3)
    reds = sns.cubehelix_palette(5, light=0.6, hue=1., dark=0.5, start=0.9,
                                 rot=0.1, as_cmap=True, gamma=1.3)
    ax.contour(np.deg2rad(phivec), np.deg2rad(thetavec), z1.T, cmap=reds,
               linewidths=1.)
    ax.contour(np.deg2rad(phivec), np.deg2rad(thetavec), z2.T, cmap=greens,
               linewidths=1.)
    ax.contour(np.deg2rad(phivec), np.deg2rad(thetavec), z3.T, cmap=blues,
               linewidths=1.)
    # Plot them max likelihood vectors
    for i, sig in enumerate(['S1', 'S2', 'S3']):
        s_cols = ['r', 'g', 'b']
        s_labs = ['$\sigma_1$', '$\sigma_2$', '$\sigma_3$']
        phi = strs_params['{}:Phi'.format(sig)]['mean']
        theta = strs_params['{}:Theta'.format(sig)]['mean']
        # Sort out upwards vectors
        if theta > 90:
            theta = 180. - theta
            if phi < 0:
                phi = 180 + phi
            else:
                phi = phi + 180.
        else:
            if phi < 0:
                phi = 360 + phi
        ax.scatter(np.deg2rad(phi), np.deg2rad(theta), color=s_cols[i],
                   label=s_labs[i], zorder=3., s=50.)
    # SHmax
    mean = strs_params['Shmax']['mean']
    X10 = strs_params['Shmax']['X10']
    X90 = strs_params['Shmax']['X90']
    width = (np.abs(X10 - mean) + np.abs(X90 - mean)) / 2.
    w_rad = np.deg2rad(width)
    # Plot both sides of bow tie
    ax.bar(np.deg2rad(mean), 10., width=w_rad, color='lightgray', alpha=0.7)
    ax.bar(np.deg2rad(mean) + np.pi, 10., width=w_rad, color='lightgray',
           alpha=0.7, label='90% SH$_{max}$')
    ax.plot([np.deg2rad(mean) + np.pi, 0, np.deg2rad(mean)], [10, 0, 10],
            linewidth=2., linestyle='--', color='k', label='SH$_{max}$')
    ax.legend(bbox_to_anchor=(0.1, 1.1))
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.margins(0.0)
    # Set up to North, clockwise scale, 180 offset
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_yticklabels([])
    ax.set_ylim([0, np.pi / 2])
    ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W'])
    if show:
        plt.show()
    return ax

def plot_one_cluster():
    return

def plot_all_clusters(group_cats, outdir, plot_dir, wells=None, **kwargs):
    """
    Plot clusters in map view, xsection and stress orientations

    :param group_cat: Catalog of events in the cluster
    :param outdir: Arnold-Townend output directory
    :param plot_dir: Output directory for plots
    :param kwargs: kwargs passed to plot_well_seismicity
    :return:
    """
    # Just big loop over all clusters
    for i, cat in enumerate(group_cats):
        # Do some setting up of the plots based on wells
        if wells == 'Rotokawa':
            profile = [(176.185, -38.60), (176.21, -38.62)]
            xsection = [(176.185, -38.60), (176.21, -38.62)]
        else:
            # Figure out which wells to plot from median event latitude
            med_lat = np.median([ev.preferred_origin().latitude for ev in cat])
            if med_lat > -38.55:  # NgaN
                wells = ['NM08', 'NM09']
            elif med_lat < -38.55:
                wells = ['NM06', 'NM10']
            profile = 'EW'
            xsection = None
        if len(cat) < 10:
            print('Fewer than 10 events. Not plotting')
            continue
        clust_name = '{}_0'.format(i)
        # Set up subplots with gridspec
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(4, 4)#, hspace=0.1, wspace=0.1)
        ax_map = fig.add_subplot(gs[0:2, 0:2])
        ax_xsec = fig.add_subplot(gs[2:, :])
        ax_strs = fig.add_subplot(gs[0:2, 2:], polar=True)
        ax_map = plot_well_seismicity(cat, wells=wells, profile='map',
                                      ax=ax_map, xsection=xsection,
                                      color=False, **kwargs)
        ax_xsec = plot_well_seismicity(cat, wells=wells, profile=profile,
                                       ax=ax_xsec, color=False, **kwargs)
        try:
            ax_strs = plot_arnold_density(outdir=outdir, clust_name=clust_name,
                                          ax=ax_strs)
        except:
            print('Inversion output doesnt exist. Moving on.')
            continue
        plt.tight_layout()
        plt.savefig('{}/Cluster_{}.png'.format(plot_dir, clust_name), dpi=300,
                    bbox='tight')
        plt.close('all')
    return