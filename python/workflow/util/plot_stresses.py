#!/usr/bin/python

import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from matplotlib.pyplot import GridSpec
from itertools import cycle
from plot_well_data import plot_well_seismicity
from focal_mecs import rose_plot

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


def arnold_stress_to_gmt(out_dir, out_file, spacing, method='SHmax',
                         color_boxes=True):
    """
    Arnold stress output directory to gmt input file. Writing this specifically
    for plotting gridded principle axes to compare with PMG inversion

    :param out_dir: Path to output directory of Arnold stress package.
    :param out_file: Path to file that will be written for use with gmt
    :param spacing: Horizontal grid spacing in degrees.
    :return:
    """
    grid_file = glob('{}/*.grid'.format(out_dir))[0]
    grid_dict = {}
    # Dictionary of grid indices: grid coordinates
    with open(grid_file, 'r') as gf:
        for ln in gf:
            line = ln.rstrip('\n').split()
            grid_dict['{}_{}'.format(line[0], line[1])] = (line[2], line[3])
    # Make list of unique indices in directory
    out_ps = glob('{}/*.eps'.format(out_dir))
    indices = list(set([ps.rstrip('.eps').split('/')[-1] for ps in out_ps]))
    # Write sigmas and boxes file
    with open('{}.boxes'.format(out_file), 'w') as box_f:
        with open(out_file, 'w') as f:
            for ind in indices:
                param_files = glob('{}/{}.*{}.dat'.format(out_dir, ind,
                                                          'dparameters'))
                strs_params = parse_arnold_params(param_files)
                color = strs_params['nu']['mean']
                mean = strs_params['Shmax']['mean']
                X10 = strs_params['Shmax']['X10']
                X90 = strs_params['Shmax']['X90']
                if method == 'SHmax':
                    # Flip these around for other half of bowtie
                    back_10 = X10 - 180.
                    back_90 = X90 - 180.
                    if back_10 < 0.:
                        back_10 += 360.
                    if back_90 < 0.:
                        back_90 += 360.
                    if not color_boxes:
                        f.write('{} {} {} {} {}\n'.format(grid_dict[ind][0],
                                                          grid_dict[ind][1],
                                                          color, X10, X90))
                        f.write('{} {} {} {} {}\n'.format(grid_dict[ind][0],
                                                          grid_dict[ind][1],
                                                          color, back_10,
                                                          back_90))
                    else:
                        f.write('>-Glightgray\n')
                        f.write('{} {} {} {}\n'.format(grid_dict[ind][0],
                                                       grid_dict[ind][1],
                                                       X10, X90))
                        f.write('{} {} {} {}\n'.format(grid_dict[ind][0],
                                                       grid_dict[ind][1],
                                                       back_10, back_90))
                elif method == 'sigmas':
                    # Grab sigma trend and plunge values
                    s_cols = ['red', 'green', 'blue']
                    size = 1.5
                    for i, sig in enumerate(['S1', 'S2', 'S3']):
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
                        length = 0.6 * np.sin(np.deg2rad(theta))
                        f.write('>-W{},{}\n'.format(size, s_cols[i]))
                        # Size in 3rd column. Then 4 and 5 for az and length
                        f.write('{} {} 0 {} {}\n'.format(grid_dict[ind][0],
                                                         grid_dict[ind][1],
                                                         phi, length))
                # nu boxes
                # Put color zval in header
                lat = float(grid_dict[ind][1])
                lon = float(grid_dict[ind][0])
                h = spacing / 2.0
                if color_boxes:
                    box_f.write('>-Z{}\n'.format(color))
                else:
                    box_f.write('>\n')
                box_f.write(
                    '{} {}\n{} {}\n{} {}\n{} {}\n{} {}\n'.format(
                        lon - h, lat + h, lon + h, lat + h, lon + h,
                        lat - h, lon - h, lat - h, lon - h, lat + h
                ))
    return

def boxes_to_gmt(box_file, out_file, stress_dir=None):
    """
    Output gmt formatted file for quadtree boxes. Can color by various params

    :param box_file: Path to box file from matlab quadtree codes
    :param out_file: Path to output file
    :param stress_dir: Path to directory of corresponding inversion results
    :return:
    """
    with open(out_file, 'w') as out_f:
        with open(box_file, 'r') as in_f:
            for i, ln in enumerate(in_f):
                line = ln.rstrip('\n').split()
                if stress_dir:
                    froot = '/'.join([stress_dir, '{}_0'.format(i)])
                    param_files = glob('{}.*{}.dat'.format(froot,
                                                           'dparameters'))
                    if len(param_files) == 0:
                        out_f.write('>-ZNaN\n')
                    else:
                        strs_params = parse_arnold_params(param_files)
                        color = strs_params['nu']['mean']
                        # Put color zval in header
                        out_f.write('>-Z{}\n'.format(color))
                out_f.write('{} {}\n{} {}\n{} {}\n{} {}\n{} {}\n'.format(
                    line[0], line[2], line[0], line[3], line[1], line[3],
                    line[1], line[2], line[0], line[2]
                ))
    return

def plot_arnold_density(outdir, clust_name, ax=None, legend=False, show=False,
                        label=False, cardinal_dirs=False):
    """
    Porting the contour plotting workflow from Richard's R code

    :param outdir: Output directory for Arnold-Townend inversion
    :param clust_name: String of cluster name to plot
    :param ax: matplotlib Axes object to plot onto. This should already
        be defined as polar projection.
    :param legend: Whether we want the legend or not
    :param show: Automatically show this plot once we're done?
    :param label: Are we labeling in the top left by the cluster #?
    :param cardinal_dirs: NSEW around edge or plot, or no?

    :return: matplotlib Axes object
    """
    froot = '/'.join([outdir, clust_name])
    grid_f = '{}.{}.dat'.format(froot, 's123grid')
    param_files = glob('{}.*{}.dat'.format(froot, 'dparameters'))
    if not os.path.isfile(grid_f):
        print('{} doesnt exist. Cluster likely too small'.format(grid_f))
        return
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
    nu = strs_params['nu']['mean']
    width = (np.abs(X10 - mean) + np.abs(X90 - mean)) / 2.
    w_rad = np.deg2rad(width)
    # Plot both sides of bow tie
    ax.bar(np.deg2rad(mean), 10., width=w_rad, color='lightgray', alpha=0.7)
    ax.bar(np.deg2rad(mean) + np.pi, 10., width=w_rad, color='lightgray',
           alpha=0.7, label='90% SH$_{max}$')
    ax.plot([np.deg2rad(mean) + np.pi, 0, np.deg2rad(mean)], [10, 0, 10],
            linewidth=2., linestyle='--', color='k', label='SH$_{max}$')
    if legend:
        ax.legend(bbox_to_anchor=(0.1, 1.1))
    if label:
        ax.text(0., 0.9, clust_name.split('_')[0], fontsize=14,
                transform=ax.transAxes)
    # Text for nu
    ax.text(0.5, -0.15, '$\\nu$ = {:0.1f}'.format(nu), fontsize=14.,
            transform=ax.transAxes, horizontalalignment='center')
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.margins(0.0)
    # Set up to North, clockwise scale, 180 offset
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_yticklabels([])
    ax.set_ylim([0, np.pi / 2])
    if cardinal_dirs:
        ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W'])
    else:
        ax.set_xticklabels([])
    if show:
        plt.show()
    return ax


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

################### FRACTURE and Fault Plane Plotting ########################

def plot_fracs(well, label=True, cardinal_dirs=True, depth_interval=None,
               ax=None, show=False, outfile=None):
    """
    Plot density plot of poles to fractures from AFIT/FMI logs for a given well
    :param well: path to well file (depth, dip, strike, dip direction, ...)
    :param label: Are we labeling in the top left with well name?
    :param cardinal_dirs: Include cardinal direction labels?
    :param depth_interval: Start (top) and end (bottom) depths to plot
    :param ax: matplotlib.Axes object preconfigured as polar plot
    :return:
    """
    if not ax:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    # Read in arrays
    data = np.genfromtxt(well, delimiter=',', skip_header=1)
    depth = data[:,0]
    # Select only fractures within depth interval
    if depth_interval:
        # Check your depth interval is valid
        if (depth_interval[0] < data[:,0].min() or
            depth_interval[1] > data[:,0].max()):
            print('Depth interval provided doesnt exist for this well')
            return
        dep_data = data[np.where(
            np.logical_and(data[:,0] < depth_interval[1],
                           data[:,0] > depth_interval[0]))]
    else:
        dep_data = data
    dip = dep_data[:,1]
    dip_dir = dep_data[:,3]
    # Pole to plane is dip dir - 180
    pole_dir = dip_dir - 180.
    # Correct values less than 0
    pole_dir = np.where(pole_dir >= 0., pole_dir, pole_dir + 360.)
    pole_angle = np.deg2rad(dip) # Angle up from down
    pole_az = np.deg2rad(pole_dir) # East from North
    # Subtract 90 for strike values and eliminate negatives
    strk_az = pole_az - (np.pi / 2.)
    strk_az = np.where(strk_az >=0., strk_az, strk_az + 2 * np.pi)
    # Define the bin areas
    # Plot'em
    if label:
        lab = well.split('_')[-2].split('/')[-1]
    ax = rose_plot(trend=pole_az, plunge=pole_angle, strike=strk_az, label=lab,
                   cardinal_dirs=cardinal_dirs, ax=ax, show=show,
                   outfile=outfile)
    return ax