#!/usr/bin/python
"""
Plotting functions for well related data (reorganizing to clear up
plot_detections.py
"""
import matplotlib
import csv
import pytz

import numpy as np
import pandas as pd
import seaborn as sns
try:
    import colorlover as cl
except:
    print('On the server. No colorlover')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

from glob import glob
from itertools import cycle
from scipy import special
from datetime import timedelta
from obspy import Catalog, UTCDateTime
from obspy.geodetics import kilometer2degrees, degrees2kilometers
from obspy.imaging.beachball import beach
from scipy.interpolate import splev, splrep
from focal_mecs import beach_mod
from eqcorrscan.utils.mag_calc import dist_calc
from shelly_mags import local_to_moment_Majer
from obspy.imaging.scripts.mopad import MomentTensor as mopad_MT
from obspy.imaging.scripts.mopad import BeachBall as mopad_BB


def date_generator(start_date, end_date):
    # Generator for date looping
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def plot_stress_depth(field='NgaN', axes=None, show=False):
    """
    Place to keep Sigma_v with depth for each part of the fields

    :param field: 'NgaN', 'NgaS', 'Rot' or 'all'
    :return:
    """
    stress_dict = {'NgaN': {}, 'NgaS': {}, 'Rot': {}}
    stress_dict['NgaN']['elevs'] = np.array([52, 222, 297, 382, 707, 1067,
                                             2722, 2827, 3042, 3497]) - 350.
    stress_dict['NgaN']['MPas'] = np.array([0.74, 2.46, 3.8, 5.58, 12.09, 19.,
                                            57.28, 59.82, 65.57, 76.76])
    stress_dict['NgaS']['elevs'] = np.array([80., 240., 585., 825.,
                                             1205., 2210., 3385.]) - 350.
    stress_dict['NgaS']['MPas'] = np.array([1.14, 2.76, 8.91, 13.94, 21.54,
                                            44.78, 72.42])
    if not axes:
        fig, ax = plt.subplots(figsize=(4, 8))
    else:
        ax = axes.twiny()
    # Plot stress profile
    colors = cycle(['rebeccapurple', 'mediumaquamarine', 'darkorange'])
    for fld, f_dict in stress_dict.items():
        if fld == field or field == 'all' and fld != 'Rot':
            ax.scatter(f_dict['MPas'], f_dict['elevs'],
                       label='{}: Lithostatic'.format(fld),
                       marker='o', color=next(colors))
    # Plot hydrostaic line
    ax.plot((0., 39.2), (-350., 3650.), linestyle='--', color='darkgray')
    ax.text(39.2, 3650, 'Pp (hydrostatic)', color='darkgray')
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel('MPa', fontsize=16)
    ax.set_ylabel('Depth (m)', fontsize=16)
    ax.set_title('Stress with depth', fontsize=18)
    if show:
        plt.show()
    return ax

def parse_feedzone_file(fz_file, well):
    """
    Helper to parse the feedzone csvs from Mercury.
    Format: well_name, fz_top (mCT), fz_bottom (mCT), other shiz

    Will get passed to plot_PTS
    """
    fzs = []
    surf = 350. # CT in m asl
    with open(fz_file, 'r') as f:
        for line in f:
            ln = line.split(',')
            if ln[0] == well:
                fzs.append([float(ln[1]) - surf,
                            float(ln[2]) - surf])
    return fzs

def fix_legend(ax):
    # Helper to remove repeated legend handles
    hand, labl = ax.get_legend_handles_labels()
    handout=[]
    lablout=[]
    for h,l in zip(hand,labl):
       if l not in lablout:
            lablout.append(l)
            handout.append(h)
    ax.legend(handout, lablout)
    return ax

def plot_PTS(PTS_data, wells, NST=False, ax=None, show=False, title=False,
             outfile=False, feedzones=None, fz_labels=False):
    """
    Simple plots of Pressure-temperature-flow spinner data
    :param PTS_data: path to PTS excel sheet
    :param wells: list of well names to plot
    :param NST: False or path for plotting Natural State Temperatures for each
        well
    :param ax: matplotlib.Axes to plot into
    :param show: Show the plot?
    :param title: Plot title
    :param outfile: Output file for figure
    :param feedzones: Path to pertinent feedzone file
    :param fz_labels: Boolean for fz labels
    :return:
    """
    if ax:
        ax1 = ax
    else:
        fig, ax1 = plt.subplots(figsize=(5, 8), dpi=300)
    temp_colors = cycle(sns.color_palette('Blues', 3))
    nst_colors = cycle(sns.color_palette('Reds', 3))
    # Make little dict of flow rates for curves at wells
    fr_dict = {'NM08': [55, 130, 22], 'NM09': [130, 90, 50], 'NM10': [2.2, 67]}
    for well in wells: # Just to keep column namespace clear
        df = pd.read_excel(PTS_data, sheetname=well)
        if NST:
            df_nst = pd.read_excel(NST, sheetname='Data', header=[0, 1])
            # Make depth positive down to agree with PTS data
            elev = df_nst[('{} NST Interp 2016'.format(well), 'Elev')].values
            elev *= -1.
            t = df_nst[('{} NST Interp 2016'.format(well), 'T')].values
            ax1.plot(t, elev, label='{} NST'.format(well),
                     color=next(nst_colors))
        for i in range(len(fr_dict[well])):
            if i > 0:
                suffix = '.{}'.format(i)
            else:
                suffix = ''
            # Do the elevation conversion
            df['elev{}'.format(suffix)] = df['depth{}'.format(suffix)] - 350.
            ax1 = df.plot('temp{}'.format(suffix), 'elev{}'.format(suffix),
                          color=next(temp_colors), ax=ax1,
                          label='{} temps {} t/h'.format(well,
                                                         fr_dict[well][i]),
                          legend=False)
        ax1.set_xlim((0, 300))
        if feedzones:
            xlims = ax1.get_xlim()
            xz = [xlims[0], xlims[1], xlims[1], xlims[0]]
            for fz in parse_feedzone_file(feedzones, well):
                yz = [fz[0], fz[0], fz[1], fz[1]]
                ax1.fill(xz, yz, color='lightgray', zorder=0,
                         alpha=0.9, label='Feedzone')
                if fz_labels:
                    ax1.text(200., (fz[0] + fz[1]) / 2., 'Feedzone',
                             fontsize=8, color='gray',
                             verticalalignment='center')
    ax1.invert_yaxis()
    ax1.set_ylabel('Depth (m bsl)', fontsize=16)
    ax1.set_xlabel(r'Temperature ($\degree$C)', fontsize=16)
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('NST & Injection Temperatures')
    fix_legend(ax1)
    if show:
        plt.show()
    elif outfile:
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
    return ax1

def read_fm_file(fm_file, cat_format):
    sdrs = {}
    with open(fm_file, 'r') as f:
        next(f)
        for line in f:
            line = line.rstrip('\n')
            line = line.split(',')
            if cat_format == 'detections':
                fid = line[0].split('.')[0][:-6]
                sdrs[fid] = (float(line[1]), float(line[2]),
                             float(line[3]))
            elif cat_format == 'templates':
                sdrs[line[0].split('.')[0]] = (float(line[1]), float(line[2]),
                                               float(line[3]))
    return sdrs

def proj_point(a, b, p):
    """
    Project a point onto a line segment, return along-line distance in km
    and azimuth of line in geog cood system (-90 to 90)

    :param a: One end of line (lon, lat)
    :param b: Other end of line (lon, lat)
    :param p: Point to project (lon, lat)
    :return:
    """
    ap = np.array(p) - np.array(a)
    ab = np.array(b) - np.array(a)
    pt_on_line = np.array(a) + np.dot(ap, ab) / np.dot(ab, ab) * ab
    # Flip pts for dist calc
    pt2 = (pt_on_line[1], pt_on_line[0], 0.)
    a2 = (a[1], a[0], 0.)
    along_line = dist_calc(a2, pt2) # along line dist in km
    # pt defining equilateral triangle
    c = np.array((a[0], b[1]))
    ac = np.array(c) - np.array(a)
    az = np.rad2deg(np.arccos(np.dot(ab, ac) /
                              (np.linalg.norm(ab) * np.linalg.norm(ac))))
    # Return az -90 (West) to 90 (East)
    if a[1] > b[1]:
        azimuth = -1. * az
    else:
        azimuth = az
    return along_line, azimuth

def plot_well_seismicity(catalog, wells, profile='NS', dates=None, color=True,
                         ax=None, show=False, outfile=None, feedzones=None,
                         fz_labels=True, focal_mechs=None, cat_format=None,
                         fm_color='b', ylims=None, dd_only=True,
                         colorbar=False, c_axes=None, xsection=None,
                         half_width=None, errors=False):
    """
    Plot well with depth and seismicity

    :param catalog: Catalog of seismicity
    :param wells: List of strings specifying well names to plot or either
        'Rotokawa' or 'Ngatamariki' for whole field plots.
    :param profile: 'NS', 'EW', 'map', or a list of two tuples defining
        the line for a cross section e.g. [(lon, lat), (lon, lat)]
    :param dates: Start and end dates for the catalog
    :param color: True: color by time since start of plot
    :param ax: matplotlib.Axes object to plot into
    :param show: To show this axes or not
    :param outfile: Path to an output file
    :param feedzones: List of lists specifying the elevations of the tops and
        bottoms of any feedzones you wish to plot. Will plot them horizontally
        across entire axes, though.
    :param fz_labels: Boolean to label the feedzones at their central depth
    :param focal_mechs: None or path to arnold-townend fm file
    :param cat_format: Naming format for events in seismicity catalog. Used to
        find corresponding focal mech info in file focal_mechs, if it exists.
    :param fm_color: Either a mpl recognized color string, or None, in which
        case the beachballs will be colored by date of occurrence.
    :param ylims: Manually set ylims for the depth cross sections if you wish
    :param dd_only: Accept only HypoDD locations?
    :param colorbar: Boolean for whether to plot a colorbar
    :param c_axes: If colorbar == True, can supply an axes into which it will
        be plotted.
    :param xsection: If profile == 'map', then specify the endpts of the
        xsection to be plotted in map view.
    :param half_width: Secify the xy width of the plot in degrees. Otherwise
        is set to 0.02
    :param errors: If True, will plot location errors as crosses centered
        at hypocenter.
    :return:

    ..note: Optimum Rotokawa profile: [(176.185, -38.60), (176.21, -38.62)]
    """
    if ax:
        ax1 = ax
    else:
        fig, ax1 = plt.subplots()
    colors = cycle(['steelblue', 'skyblue'])
    if dd_only and not dates:
        catalog = Catalog(events=[ev for ev in catalog
                                  if ev.preferred_origin().method_id
                                  and not ev.preferred_origin().quality])
    elif dd_only and dates:
        catalog = Catalog(events=[ev for ev in catalog
                                  if ev.preferred_origin().method_id
                                  and ev.preferred_origin().origin_uncertainty
                                  and dates[0] < ev.preferred_origin().time
                                  < dates[1]])
        catalog.events = [
            ev for ev in catalog
            if ev.preferred_origin().method_id.id.endswith('GrowClust')
        ]
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    well_pt_lists = []
    # Dictionary of fm strike-dip-rake from Arnold/Townend pkg
    if focal_mechs:
        sdrs = read_fm_file(focal_mechs, cat_format)
        fm_tup = []
        for ev in catalog:
            if cat_format == 'detections':
                fm_id = '{}_{}_{}'.format(
                    ev.resource_id.id.split('/')[-1].split('_')[0],
                    ev.resource_id.id.split('_')[-2],
                    ev.resource_id.id.split('_')[-1][:6])
            elif cat_format == 'templates':
                fm_id = ev.resource_id.id.split('/')[-1]
            else:
                print('Provide relevant catalog format')
                return
            if fm_id in sdrs:
                fm_tup.append(sdrs[fm_id])
            else:
                fm_tup.append(None)
    if wells == 'Rotokawa':
        wells = []
        for well_file in glob('/home/chet/gmt/data/NZ/wells/RK??_xyz_pts.csv'):
            wells.append(well_file.split('/')[-1][:4])
            well_pt_lists.append(format_well_data(well_file, depth='mbsl'))
    elif wells == 'Ngatamariki':
        wells = []
        for well_file in glob('/home/chet/gmt/data/NZ/wells/NM??_xyz_pts.csv'):
            wells.append(well_file.split('/')[-1][:4])
            well_pt_lists.append(format_well_data(well_file, depth='mbsl'))
    else:
        for well in wells:
            well_file = '/home/chet/gmt/data/NZ/wells/{}_xyz_pts.csv'.format(well)
            # Grab well pts (these are depth (kmRF)) correct accordingly
            well_pt_lists.append(format_well_data(well_file, depth='mbsl'))
    if len(catalog) > 0:
        t0 = catalog[0].preferred_origin().time.datetime
        if errors:
            pts = [(ev.preferred_origin().longitude,
                    ev.preferred_origin().latitude,
                    ev.preferred_origin().depth,
                    ev.preferred_magnitude().mag,
                    (ev.preferred_origin().time.datetime - t0).total_seconds(),
                    kilometer2degrees(ev.preferred_origin().origin_uncertainty.horizontal_uncertainty / 1000.),
                    ev.preferred_origin().depth_errors.uncertainty)
                   for ev in catalog]
            lons, lats, ds, mags, secs, hus, dus = zip(*pts)
        else:
            pts = [(ev.preferred_origin().longitude,
                    ev.preferred_origin().latitude,
                    ev.preferred_origin().depth,
                    ev.preferred_magnitude().mag,
                    (ev.preferred_origin().time.datetime - t0).total_seconds())
                   for ev in catalog]
            lons, lats, ds, mags, secs = zip(*pts)
        mags /= max(np.array(mags))
        days = np.array(secs) / 86400.
        n_days = days / np.max(days)
        d_cols = [cm.viridis(d) for d in n_days]
        if type(profile) == list:
            proj_dists = [proj_point(profile[0], profile[1],
                                     (pt[0], pt[1]))[0]
                          for pt in pts]
        if errors:
            # Make arrays for plotting with hlines and vlines
            x_data = {'y': [], 'xmax': [], 'xmin': [], 'colors': 'gray'}
            y_data = {'x': [], 'ymax': [], 'ymin': [], 'colors': 'gray'}
            for i, d in enumerate(ds):
                if profile != 'map':
                    x_data['y'].append(d)
                    y_data['ymin'].append(d - dus[i])
                    y_data['ymax'].append(d + dus[i])
                if profile == 'NS':
                    y_data['x'].append(lats[i])
                    x_data['xmin'].append(lats[i] + hus[i]) #S Hemisphere only
                    x_data['xmax'].append(lats[i] - hus[i])
                elif profile == 'EW':
                    y_data['x'].append(lons[i])
                    x_data['xmin'].append(lons[i] - hus[i]) #S Hemisphere only
                    x_data['xmax'].append(lons[i] + hus[i])
                elif profile == 'map':
                    y_data['x'].append(lons[i])
                    x_data['xmin'].append(lons[i] - hus[i])  # S Hemisphere only
                    x_data['xmax'].append(lons[i] + hus[i])
                    x_data['y'].append(lats[i])
                    y_data['ymin'].append(lats[i] + hus[i])
                    y_data['ymax'].append(lats[i] - hus[i])
                elif type(profile) == list: # Projected coords
                    y_data['x'].append(proj_dists[i])
                    x_data['xmin'].append(proj_dists[i] -
                                          degrees2kilometers(hus[i]) * 1000.)
                    x_data['xmax'].append(proj_dists[i] +
                                          degrees2kilometers(hus[i]) * 1000.)
        # Plot seismicity
        if color:
            col = days # Days since first event
        else:
            col = 'darkgray'
        if errors:
            ax1.vlines(x=y_data['x'], ymin=y_data['ymin'], ymax=y_data['ymax'],
                       color='gray', alpha=0.1)
            ax1.hlines(y=x_data['y'], xmin=x_data['xmin'], xmax=x_data['xmax'],
                       color='gray', alpha=0.1)
        # If profile is a list of xy pts, reproject everything
        if type(profile) == list:
            # Get projected distances
            sc = ax1.scatter(proj_dists, ds, s=(12 * mags) ** 2, c=col,
                             alpha=0.7, label='Events')
            ax1.annotate('A', xy=(0., 1.), xytext=(0., 10), fontsize=14,
                         xycoords='axes fraction',
                         textcoords='offset points',
                         horizontalalignment='center')
            ax1.annotate("A'", xy=(1., 1.), xytext=(0., 10), fontsize=14,
                         xycoords='axes fraction',
                         textcoords='offset points',
                         horizontalalignment='center')
        elif profile == 'NS':
            sc = ax1.scatter(lats, ds, s=(12 * mags) ** 2, c=col, alpha=0.7,
                             label='Events')
            ax1.annotate('N', xy=(0., 1.), xytext=(0., 10), fontsize=14,
                         xycoords='axes fraction',
                         textcoords='offset points',
                         horizontalalignment='center')
            ax1.annotate('S', xy=(1., 1.), xytext=(0., 10), fontsize=14,
                         xycoords='axes fraction',
                         textcoords='offset points',
                         horizontalalignment='center')
        elif profile == 'EW':
            sc = ax1.scatter(lons, ds, s=(12 * mags) ** 2, c=col, alpha=0.7,
                             label='Events')
            ax1.annotate('W', xy=(0., 1.), xytext=(0., 10), fontsize=14,
                         xycoords='axes fraction',
                         textcoords='offset points',
                         horizontalalignment='center')
            ax1.annotate('E', xy=(1., 1.), xytext=(0., 10), fontsize=14,
                         xycoords='axes fraction',
                         textcoords='offset points',
                         horizontalalignment='center')
        elif profile == 'map':
            sc = ax1.scatter(lons, lats, s=(12 * mags) ** 2, c=col, alpha=0.7,
                             label='Events')
            ax1.set_ylabel('Latitude')
            ax1.set_xlabel('Longitude')
            if xsection:
                ax1.plot((xsection[0][0], xsection[1][0]),
                         (xsection[0][1], xsection[1][1]),
                         linestyle='--', color='k',
                         linewidth=2.)
                ax1.annotate('A', xy=(xsection[0][0], xsection[0][1]),
                             fontsize=20, xytext=(0., 6),
                             textcoords='offset points',
                             horizontalalignment='left')
                ax1.annotate("A'", xy=(xsection[1][0], xsection[1][1]),
                             fontsize=20, xytext=(6., 0),
                             textcoords='offset points',
                             horizontalalignment='left')
        if colorbar:
            if not c_axes:
                cbar = plt.colorbar(sc, ax=ax1)
            else:
                cbar = plt.colorbar(sc, cax=c_axes, orientation='horizontal',
                                    format='%d')
            cbar.set_label('Elapsed days', fontsize=14)
    for i, well_pts in enumerate(well_pt_lists):
        if wells[i] == 'RK19':
            print('Not plotting RK19, as it clutters injection field')
            continue
        if type(profile) == list:
            well_dists = [proj_point(profile[0], profile[1], (pt[1], pt[0]))[0]
                          for pt in well_pts]
        wlat, wlon, wkm = zip(*well_pts)
        wdp = (np.array(wkm) * 1000.)# - elevation
        if wells[i][:2] == 'RK' and 19 < int(wells[i][2:]) < 25:
            col = 'steelblue'
        elif wells[i][:2] == 'RK':
            col = 'firebrick'
        else:
            col = next(colors)
        if type(profile) == list:
            ax1.plot(well_dists, wdp, color=col,
                     label='{} wellbore'.format(wells[i]))
        elif profile == 'NS':
            ax1.plot(wlat, wdp, color=col,
                     label='{} wellbore'.format(wells[i]))
        elif profile == 'EW':
            ax1.plot(wlon, wdp, color=col,
                     label='{} wellbore'.format(wells[i]))
        elif profile == 'map':
            ax1.plot(wlon, wlat, color=col,
                     label='{} wellbore'.format(wells[i]))
            if i > 0:
                ax1.scatter(wlon[0], wlat[0], s=20., color='k',
                            zorder=3)
            else:
                ax1.scatter(wlon[0], wlat[0], s=20., color='k',
                            label='Wellhead',
                            zorder=3)
    # Plot beachballs if we have them
    if focal_mechs:
        for i, fm in enumerate(fm_tup):
            if fm:
                if not fm_color:
                    fm_col = d_cols[i]
                else:
                    fm_col = fm_color
                # Here do reprojection with MoPaD FM object by defining new
                # viewpoint. Then feed resulting fm into obspy beach obj
                if type(profile) == list:
                    pp, az = proj_point(profile[0], profile[1], (0., 0.))
                    # Calculte azimuth of xsection defined by 2 pts
                    # Always view xsection from the south quadrant
                    # Assuming profile is defined "left-to-right" a --> b
                    if az < 0.:
                        view = [az, -90, -90.]
                    else:
                        view = [-1. * az, 90, 90.]
                    bball = beach_mod(fm, xy=(proj_dists[i], ds[i]),
                                      width=(mags[i] ** 2) * 100,
                                      linewidth=1, axes=ax1,
                                      facecolor=fm_col,
                                      viewpoint=view)
                elif profile == 'NS':
                    bball = beach_mod(fm, xy=(lats[i], ds[i]),
                                      width=(mags[i] ** 2) * 100,
                                      linewidth=1, axes=ax1,
                                      facecolor=fm_col,
                                      viewpoint=[0., -90., -90.])
                elif profile == 'EW':
                    bball = beach_mod(fm, xy=(lons[i], ds[i]),
                                      width=(mags[i] ** 2) * 100,
                                      linewidth=1, axes=ax1,
                                      facecolor=fm_col,
                                      viewpoint=[-90., 0., 0.])
                elif profile == 'map':
                    bball = beach_mod(fm, xy=(lons[i], lats[i]),
                                      width=(mags[i] ** 2) * 100,
                                      linewidth=1, axes=ax1,
                                      facecolor=fm_col,
                                      viewpoint=[0., 0., 0.])
                ax1.add_collection(bball)
    # Redo the xaxis ticks to be in meters by calculating the distance from
    # origin to ticks
    # Extend bounds for deviated wells
    if not half_width:
        half_width = 0.02
    # Now center on wellhead position (should work for last wlat as only
    # wells from same wellpad should be plotted this way)
    print(wells)
    if type(profile) == list:
        xsec_len = dist_calc((profile[0][1], profile[0][0], 0.0),
                             (profile[1][1], profile[1][0], 0.0))
        ax1.set_xlim([0, xsec_len])
    if profile == 'NS':
        ax1.set_xlim([wlat[0] + half_width, wlat[0] - half_width])
    elif profile == 'EW':
        # Center axes on wellbore
        ax1.set_xlim([wlon[0] - half_width, wlon[0] + half_width])
    elif profile == 'map' and wells[0].startswith('NM'):
        ax1.set_xlim([wlon[0] - half_width, wlon[0] + half_width])
        ax1.set_ylim([wlat[0] - half_width, wlat[0] + half_width])
    elif profile == 'map' and wells[0].startswith('RK'):
        print('Plotting Rotokawa map')
        # Center north of RT14 coordinates
        ax1.set_xlim([176.1947 - half_width, 176.1947 + half_width])
        ax1.set_ylim([-38.61 - half_width, -38.61 + half_width])
    if profile != 'map' and not ylims: # Adjust depth limits depending on well
        if 'NM10' in wells or 'NM06' in wells:
            ax1.set_ylim([4000., 0.])
        elif wells[0].startswith('RK'):
            ax1.set_ylim([5000., 0.])
        else:
            ax1.set_ylim([4000., 0.])
    elif profile != 'map' and ylims:
        ax1.set_ylim(ylims)
    if not wells[0].startswith('RK') and len([x in ['NM08', 'NM09']
                                              for x in wells]) < 2:
        ax1.legend(fontsize=12, loc=2)
    if feedzones and profile != 'map':
        x0 = ax1.get_xlim()[0] * 1.00001  # silly hack
        xlims = ax1.get_xlim()
        xz = [xlims[0], xlims[1], xlims[1], xlims[0]]
        for fz in feedzones:
            yz = [fz[0], fz[0], fz[1], fz[1]]
            ax1.fill(xz, yz, color='lightgray', zorder=0,
                     alpha=0.5)
            if fz_labels:
                ax1.text(x0, (fz[0] + fz[1]) / 2., 'Feedzone',
                         fontsize=10, color='k', verticalalignment='center')
    new_labs = []
    new_labs_y  = []
    if profile == 'NS':
        orig = (ax1.get_xlim()[0], wlon[0], 0.)
    elif profile == 'EW':
        orig = (wlat[0], ax1.get_xlim()[0], 0.)
    elif profile == 'map':
        # ax1.set_aspect('equal')
        orig = (ax1.get_ylim()[0], ax1.get_xlim()[0], 0.)
        for laby in ax1.get_yticks():
            new_labs_y.append('{:4.0f}'.format(
                dist_calc(orig, (laby, ax1.get_xlim()[0], 0.)) * 1000.))
        ax1.set_yticklabels(new_labs_y)
    for lab in ax1.get_xticks():
        if profile == 'NS':
            new_labs.append('{:4.0f}'.format(
                dist_calc(orig, (lab, wlon[0], 0.)) * 1000.))
        elif profile == 'EW':
            new_labs.append('{:4.0f}'.format(
                dist_calc(orig, (wlat[0], lab, 0.)) * 1000.))
        elif profile == 'map':
            new_labs.append('{:4.0f}'.format(
                dist_calc(orig, (ax1.get_ylim()[0], lab, 0.)) * 1000.))
        elif type(profile) == list:
            new_labs.append('{:4.0f}'.format(lab * 1000.))
    ax1.set_xticklabels(new_labs)
    ax1.set_xlabel('Meters', fontsize=16)
    if profile != 'map':
        ax1.set_ylabel('Depth (m bsl)', fontsize=16)
    else:
        ax1.set_ylabel('Meters', fontsize=16)
    if show:
        plt.show()
    elif outfile:
        plt.savefig(outfile)
        plt.close('all')
    return ax1


def make_McGarr_moments():
    # Just return the dictionary of moments from McGarr 2014
    data_dict = {
        'KTB': [200, 1.43e11], 'Soultz': [3.98e4, 2.51e13],
        'Dallas-Ft. Worth': [2.82e5, 8.9e13], 'Basel': [1.15e4, 1.41e14],
        'Ashtabula 1987': [6.17e4, 2.82e14], 'Cooper Basin': [2.e4, 3.98e14],
        'Ashtabula 2001': [3.4e5, 8.e14], 'Youngstown': [8.34e4, 8.3e14],
        'Paradox Valley': [3.287e6, 3.16e15], 'Raton Basin 01': [4.26e5, 4.5e15],
        'Guy': [6.29e5, 1.2e16], 'Painesville': [1.19e6, 2.e16],
        'Rocky Mtn. Arsenal': [6.25e5, 2.1e16], 'Timpson': [9.91e5, 2.21e16],
        'Raton Basin 11': [7.84e6, 1.e17], 'Prague': [1.2e7, 3.92e17]
    }
    return data_dict

def plot_volume_Mmax(plot_moment=False, plot_McGarr=True, show=True):
    """
    Just a way of jotting down data from Figure 3b in Geobel&Brodsky for
    comparison with Merc data

    .. note: May expand this to take arguments for well and dates to pass to
        plot_well_data to get exact cumulative sums
    :return:
    """
    fig, ax = plt.subplots()
    mcgarr_moments = make_McGarr_moments()
    data_dict = {
        'Ml': {'St Gallen': [1165, 3.5], 'Fenton Hill 83': [21600, 1.0],
               'Fenton Hill 86': [37000, 1.0],
               'Landau': [1.13E4, 2.7], 'NM08': [7E4, 2.1], 'NM10': [6E4, 2.1]
               },
        # So I'm not sure how we should deal with the Ml vs Mw scaling??
        # How did Goebel-Brodsky do it???
        # 'Mw': {'Dallas-Ft. Worth': [5.E5, 3.3], 'Geysers': [3.5E6, 3.3],
        #        'Guy-Greenbriar': [6.29E5, 4.7], 'Habanero': [2.E4, 2.2],
        #        'Paralana': [3100, 2.5], 'Newberry 12': [4.E4, 2.4],
        #        'Newberry 14': [9500, 2.4], 'Paradox': [7.7E6, 4.3],
        #        }
    }
    for mag_type, mag_dict in data_dict.items():
        for location, xy in mag_dict.items():
            if location in ['Soultz', 'Fenton Hill 83']:
                align='right'
            else:
                align='left'
            if plot_moment:
                label = 'Maximum moment ($N\cdot{m}$)'
                if mag_type == 'Mw':
                    Mo = 10.0 ** (1.5 * xy[1] + 9.0 )
                elif mag_type == 'Ml':
                    Mo = local_to_moment_Majer(xy[1])
            else:
                Mo = xy[1]
                label = 'Maximum magnitude'
            if location not in ['NM08', 'NM10']:
                ax.scatter(xy[0], Mo, marker='o', label=location, s=3,
                           color='gray', alpha=0.5)
                ax.annotate(xy=(xy[0], Mo), s=location, xytext=(0.1, 2),
                            textcoords='offset points', fontsize=8,
                            horizontalalignment=align, color='gray')
            else:
                ax.scatter(xy[0], Mo, marker='o', label=location, s=10,
                           color='goldenrod')
                if location == 'NM08':
                    ax.annotate(xy=(xy[0], Mo), s='$NM08$', xytext=(400, 100),
                                arrowprops=dict(color='gray', shrink=0.05,
                                                width=0.1, headlength=2.,
                                                headwidth=2.),
                                textcoords='offset pixels', fontsize=10,
                                color='goldenrod')
                elif location == 'NM10':
                    ax.annotate(xy=(xy[0], Mo), s='$NM10$', xytext=(10, 150),
                                arrowprops=dict(color='gray', shrink=0.05,
                                                width=0.1, headlength=2.,
                                                headwidth=2.),
                                textcoords='offset pixels', fontsize=10,
                                color='goldenrod')
    for loc, pts in mcgarr_moments.items():
        if loc.startswith('Ashtabula'):
            align = 'left'
            align_v = 'center'
        elif loc == 'Timpson':
            align = 'left'
            align_v = 'bottom'
        elif loc == 'Painesville':
            align = 'left'
            align_v = 'top'
        else:
            align = 'right'
            align_v = None
        ax.scatter(pts[0], pts[1], marker='o', label=loc, s=3,
                   color='gray', alpha=0.5)
        ax.annotate(xy=(pts[0], pts[1]), s=loc, xytext=(0.1, 2),
                    textcoords='offset points', fontsize=8,
                    horizontalalignment=align, verticalalignment=align_v,
                    color='gray')
    if plot_McGarr:
        mmax = []
        vols = np.linspace(1E2, 3E7, 100)
        for v in vols:
            mmax.append(v * 3E10) # G = 3E10 Pa from McGarr 2014
        ax.plot(vols, mmax, '--', color='dimgray')
        ax.text(x=5E2, y=3E13, s='$M_{o}=GV$', rotation=28., color='dimgray',
                horizontalalignment='center', verticalalignment='center')
    if plot_moment:
        ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Total injected volume ($m^{3}$)')
    ax.set_ylabel(label)
    ax.set_xlim([vols[0], vols[-1]])
    ax.set_title('Maximum moment vs. total injected volume')
    if show:
        plt.show()
    return ax

def plot_transient(excel_file, sheetname, dates, II=False, falloff=False,
                   tp=None, fit_start=None, xlims=None, ylims=None,
                   show=False):
    """
    Pressure transient or II plotting for well tests and stimulations

    :param excel_file: Path to the excel file with the flow/pres data
    :param sheetname: Sheetname for the operation in question
    :param dates: List of start and end UTCDateTime for the data
    :param II: Flag for whether to plot II or delta P
    :param falloff: Is this a falloff test?
    :param tp: If falloff is True, this is a UTCDto for start of injection
    :param fit_start: Number of seconds to fit the semilog line.
        Presumably determined after a visual check of the plot.
    :param xlims: Manually defined list of xlims
    :param ylims: Manually defined list of ylims

    ..note: Data will be filtered between xlims and ylims for lsqr fitting
    :return:
    """
    well = sheetname.split()[0]
    df = pd.read_excel(excel_file, header=[0, 1], sheetname=sheetname)
    df.index = df.index.tz_localize('Pacific/Auckland')
    # Convert it to UTC
    df.index = df.index.tz_convert(None)
    # Clip off unwanted data
    df = df.truncate(before=dates[0].datetime, after=dates[1].datetime)
    # Do this outside of pandas now
    # Grab pressures
    values = df[(well, 'WHP (barg)')]
    # If plotting II (as in Clearwater 2015)
    # Grab datetime objects
    dtos = values.index.to_pydatetime()
    secs = np.array([(dto - dtos[0]).total_seconds() + 1
                     for dto in dtos])
    if falloff and tp:
        tot_inj_s = dates[0] - tp
    if II: # Plot II
        flows = df[(well, 'Flow (t/h)')]
        # Can just use hydrostatic for Pr?
        # Measured pres in NM09 @2050 = 133
        # Using 180 bar as pres @2350 in NgaN for best match with Clearwater
        Pr = 180 # guesstimation from press of ~130 at
        pgz = 940 * 2400 * 9.8 * 1e-5
        denom = values + pgz - Pr
        values = ((flows / denom) * 10).values # Convert to MPa
        # Do a fitty fit
        # Get only nonzero values within xlims and ylims
        if ylims:
            II_thresh = ylims[0]
        else:
            II_thresh = 0.0
        non_zeros = values[np.where(values > II_thresh)]
        nz_secs = secs[np.where(values > II_thresh)]
        f_values = non_zeros[np.where((nz_secs < xlims[1]) &
                                      (nz_secs > xlims[0]))]
        f_secs = nz_secs[np.where((nz_secs < xlims[1]) &
                                  (nz_secs > xlims[0]))]
        # Do the fit
        n, c = np.polyfit(np.log(f_secs), np.log(f_values), 1)
        # Grab y vals for line
        line_vals = np.exp(c) * f_secs**n
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plot raw data
        ax.plot(f_secs, line_vals, color='darkorange', linestyle='--',
                label='Fit: $n={:0.2f}$'.format(n))
        ax.scatter(f_secs, f_values, color='darkblue', label='II')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Seconds', fontsize=16)
        ax.set_ylabel('Injectivity (t/h/MPa)', fontsize=16)
        ax.grid(which='both')
        plt.legend(fontsize=14)
        return ax
    else: # Plot up pressures
        # Plot raw data
        if not falloff:
            fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(8, 8))
            dP = values.diff().values
            axes[0].plot(dtos, values, color='darkslateblue', label='WHP')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('WHP (bar)')
            axes[1].loglog(secs, values, color='tomato', label='Absolute dP')
            axes[1].scatter(secs, dP, color='darkslategray',
                            label='Derivative', s=0.8)
            axes[1].set_xlabel('Seconds', fontsize=16)
            axes[1].set_ylabel('$\Delta{P}$ (bar)', fontsize=16)
        else:
            # Do the fit
            fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
            values = values.values
            non_zeros = values[np.where(values > 0.0)]
            nz_secs = secs[np.where(values > 0.0)]
            if xlims:
                f_values = non_zeros[np.where((nz_secs < xlims[1]) &
                                              (nz_secs > xlims[0]))]
                f_secs = nz_secs[np.where((nz_secs < xlims[1]) &
                                          (nz_secs > xlims[0]))]
            else:
                f_secs = nz_secs# - nz_secs[0]
                f_values = non_zeros
            # Plot raw data
            axes[0, 0].plot(f_secs, f_values, color='darkslateblue', label='WHP')
            axes[0, 0].set_xlabel('Seconds')
            axes[0, 0].set_ylabel('WHP (bar)')
            axes[0, 0].set_title('Raw Data')
            # Fit semilog plot of
            dP = f_values[0] - f_values
            if fit_start:
                fit_secs = f_secs[np.where(f_secs >= fit_start)]
                fit_vals = f_values[np.where(f_secs >= fit_start)]
                n, c = np.polyfit(np.log(fit_secs), fit_vals, 1)
            # Grab y vals for line
            line_vals = n * np.log(f_secs) + c
            # Semilog P vs logT
            axes[0, 1].plot(f_secs, line_vals, color='darkorange', linestyle='--',
                         label='Fit: $n={:0.2f}$'.format(n), linewidth=0.5)
            axes[0, 1].scatter(f_secs, f_values, color='darkslategray',
                            label='Pressure (bar)', s=0.5)
            axes[0, 1].set_xscale('log')
            axes[0, 1].set_ylabel('Pressure')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].grid(which='both')
            axes[0, 1].set_axisbelow(True)
            axes[0, 1].set_title('P vs log T')
            axes[0, 1].set_xlim([100, 50000])
            axes[0, 1].set_ylim([0, 21])
            axes[0, 1].legend()
            # loglog log deltaP-log deltaT
            # First log dP vs log T
            axes[1, 0].scatter(f_secs, dP, color='k', label='Pressure', s=0.5)
            # With pressure derivative: tdp/dt
            # Spline fit to denoise
            f = splrep(f_secs, dP, k=3, s=3)
            tdp_dt = splev(f_secs, f, der=1) * f_secs # t dP/dT
            axes[1, 0].scatter(f_secs, tdp_dt, color='r',
                               label='${t} dP/dt$', s=0.5)
            axes[1, 0].set_ylabel('$\Delta{P}$ (bar)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_xlim([100, 50000])
            axes[1, 0].set_ylim([1, 30])
            axes[1, 0].grid(which='both')
            axes[1, 0].set_axisbelow(True)
            axes[1, 0].set_title('Derivative Plot')
            axes[1, 0].legend()
            # Horner Plot (dimensionless time)?
            h_time = (f_secs + tot_inj_s) / f_secs
            axes[1, 1].scatter(h_time, dP, color='g', s=0.5)
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_title('Horner Plot?')
            axes[1, 1].grid(which='both')
            axes[1, 1].set_axisbelow(True)
            plt.tight_layout()
        if show:
            plt.show()
        return axes

def plot_well_data(excel_file, sheetname, parameter, well_list,
                   colors=None, cumulative=False, ax=None, dates=None,
                   show=False, ylims=False, outdir=None, figsize=(8, 6),
                   tick_colors=False, twin_ax=True, labs=None):
    """
    New flow/pressure plotting function utilizing DataFrame functionality
    :param excel_file: Excel file to read
    :param sheetname: Which sheet of the spreadsheet do you want?
    :param parameter: Either 'WHP (bar)' or 'Flow (t/h)' at the moment
    :param well_list: List of wells you want plotted
    :param colors: List of colors to turn into a cycle
    :param cumulative: Plot the total injected volume?
    :param ax: If plotting on existing Axes, pass it here
    :param dates: Specify start and end dates if plotting to preexisting
        empty Axes.
    :param show: Are we showing this Axis automatically?
    :param ylims: To force the ylims for the well data.
    :param outdir: Directory to save plot to. Will not show figure.
    :param figsize: Figure size passed to subplots
    :param tick_colors: Boolean for ticks color-coded to curves
    :param twin_ax: If False, will plot on the passes ax directly
    :param labs: List of labels corresponding to wells. Really just a hack
        so I can manually force no labels in legend with labs=['_nolegend_']

    :return: matplotlib.pyplot.Axes
    """
    # Yet another shit hack for silly merc data
    if (sheetname in ['NM10 Stimulation', 'NM09 Stimulation']
        and parameter == 'Injectivity'):
        # Combine sheets for DHP and Flow with different samp rates into one
        df = pd.read_excel(excel_file, header=[0, 1], sheetname=sheetname)
        df2 = pd.read_excel(excel_file, header=[0, 1],
                            sheetname='NM10 Stimulation DHP')
        df[('NM10', 'DHP (barg)')] = df2[('NM10', 'DHP (barg)')].asof(df.index)
    elif sheetname == 'Injection' and parameter == 'Injectivity':
        # Rotokawa sheet has only one header line
        df = pd.read_excel(excel_file, header=[0, 1], sheetname=sheetname)
        df2 = pd.read_excel(excel_file, header=[0, 1],
                            sheetname='WHP Press tubings')
        if not well_list[0].startswith('RK'):
            df2 = df2.asof(df.index)
        df2.index = df2.index.tz_localize('Pacific/Auckland')
        df2.index = df2.index.tz_convert('UTC')
    else:
        df = pd.read_excel(excel_file, header=[0, 1], sheetname=sheetname)
    # All flow info is local time
    df.index = df.index.tz_localize('Pacific/Auckland')
    # Convert it to UTC
    df.index = df.index.tz_convert('UTC')
    if not colors:
        colors = cycle(sns.color_palette())
    else:
        colors = cycle(colors)
    print('Flow data tz set to: {}'.format(df.index.tzinfo))
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        handles = []
        plain = True
        if dates:
            start = dates[0].datetime
            end = dates[1].datetime
            df = df.truncate(before=start, after=end)
    else:
        plain = False
        xlims = ax.get_xlim()
        if not dates:
            try:
                start = mdates.num2date(xlims[0])
                end = mdates.num2date(xlims[1])
            except ValueError:
                print('If plotting on empty Axes, please specify start'
                      'and end date')
                return
        else:
            start = dates[0].datetime
            end = dates[1].datetime
        df = df.truncate(before=start, after=end)
        if parameter == 'Injectivity':
            df2 = df2.truncate(before=start, after=end)
        if twin_ax:
            try:
                handles = ax.legend().get_lines() # Grab these lines for legend
                if isinstance(ax.legend_, matplotlib.legend.Legend):
                    ax.legend_.remove() # Need to manually remove this, apparently
            except AttributeError:
                print('Empty axes. No legend to incorporate.')
                handles = []
    # Set color (this is only a good idea for one line atm)
    # Loop over well list (although there must be slicing option here)
    # Maybe do some checks here on your kwargs (Are these wells in this sheet?)
    if cumulative:
        # THIS IS SHITE UNLESS YOU ACCOUNT FOR SAMPLING RATE
        # FLow reported as T/h, so must be corrected to jive with samp rate
        # (sec, hr, day, etc...)
        # Stimulations are 5 min samples (roughly) (i.e. / 12.)
        # Post-startup are daily (i.e. * 24.)
        maxs = []
        if twin_ax:
            ax1a = ax.twinx()
        else:
            ax1a = ax
        for i, well in enumerate(well_list):
            dtos = df.xs((well, parameter), level=(0, 1),
                         axis=1).index.to_pydatetime()
            # Post startup daily samples
            if not 'Stimulation' in sheetname.split():
                values = df.xs((well, parameter), level=(0, 1),
                               axis=1).cumsum() * 24.
            # Stimulations are irregularly sampled, so resample them to hourly
            else:
                val = df.resample('H').mean()
                values = val.xs((well, parameter), level=(0, 1),
                                axis=1).cumsum()
                dtos = df.resample('H').mean().xs((well, parameter), level=(0, 1),
                             axis=1).index.to_pydatetime()
            if outdir:
                # Write to file
                filename = 'Cumulative_flow_{}'.format(well)
                with open('{}/{}.csv'.format(outdir, filename), 'w') as f:
                    for dto, val in zip(dtos, values):
                        f.write('{} {}'.format(dto.strftime('%Y-%m-%d'), val))
                continue
            colr = next(colors)
            ax1a.plot(dtos, values,
                      label='{}: {}'.format(well, 'Cumulative Vol.'),
                      color=colr)
            plt.legend() # This is annoying
            maxs.append(np.max(df.xs((well, parameter),
                               level=(0, 1), axis=1).values))
        ax1a.set_ylabel('Cumulative Volume (Tonnes)', fontsize=16)
        # Force scientific notation for cumulative y axis
        ax1a.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    else:
        # Loop over wells, slice dataframe to each and plot
        maxs = []
        if not plain and ax.get_ylim()[-1] != 1.0:
            if twin_ax: # If we want to combine stims and post startup at Nga
                ax1a = ax.twinx()
            else:
                ax1a = ax
            # Check for existing position of labels (and probably ticks as well)
            # then put the new ones on the opposite side
            if ax.yaxis.get_ticks_position() == 'right':
                ax1a.yaxis.set_label_position('left')
                ax1a.yaxis.set_ticks_position('left')
            elif ax.yaxis.get_ticks_position() == 'left':
                ax1a.yaxis.set_label_position('right')
                ax1a.yaxis.set_ticks_position('right')
        else:
            ax1a = ax
        for i, well in enumerate(well_list):
            # Just grab the dates for the flow column as it shouldn't matter
            if parameter == 'Injectivity' and not well.startswith('RK'):
                # Use WHP = Pr - pgz + W/II + KW^2 where W is flow rate
                # JC sets K to zero for NM08...
                Pr = 180  # Reservoir pressure at 2350 in NgaN (best match to JC)
                # p water at 140C = 0.926 g/cm3 50C = 0.988 g/cm3
                # NM08 fz = 2400 m depth
                pgz = 940 * 2400 * 9.8 * 1e-5  # Pascal to bar
                # neglect friction for now XXX TODO
                if (well in ['NM10', 'NM09']
                    and sheetname.endswith('Stimulation')):
                    vals = df[(well, 'Flow (t/h)')] / df[(well, 'DHP (barg)')]
                else: # should happen for NM10 stimulation
                    denom = df[(well, 'WHP (barg)')] + pgz - Pr
                    vals = df[(well, 'Flow (t/h)')] * 10 / denom # MPa
                values = vals.where(vals < 1000.) # Careful with this shiz
                dtos = values.index.to_pydatetime()
            elif parameter == 'Depth' and sheetname == 'NM10 Losses':
                values = df[('NM10', 'Depth')] - 372. # Correcting to m bsl
                dtos = values.index.to_pydatetime()
            elif well.startswith('RK') and parameter == 'Injectivity':
                # Use WHP = Pr - pgz + W/II + KW^2 where W is flow rate
                # JC sets K to zero for NM08...
                Pr = 90  # Reservoir pressure (bar -roughly)
                # p water at 140C = 0.926 g/cm3 50C = 0.988 g/cm3
                # NM08 fz = 2400 m depth
                pgz = 940 * 2400 * 9.8 * 1e-5  # Pascal to bar
                vals = df[(well, 'Flow (t/h)')] / \
                    (df2[well, 'WHP (barg)'] + pgz - Pr)
                values = vals * 10
                dtos = values.index.to_pydatetime()
            else:
                values = df.xs((well, parameter), level=(0, 1), axis=1)
                dtos = df.xs((well, parameter), level=(0, 1),
                             axis=1).index.to_pydatetime()
            maxs.append(np.max(values.dropna().values))
            if outdir:
                # Write to file
                filename = '{}_{}_{}'.format(well, sheetname.split()[-1],
                                             parameter.split()[0])
                with open('{}/{}.csv'.format(outdir, filename), 'w') as f:
                    for dto, val in zip(dtos, values.values):
                        f.write('{} {}\n'.format(
                            dto.strftime('%Y-%m-%dT%H:%M:%S'), val[0]))
                continue
            colr = next(colors)
            # Force MPa instead of bar units
            # Ugliest thing ever
            if parameter in ['WHP (bar)', 'WHP (barg)']:
                if labs:
                    label = labs[i]
                else:
                    label = '{}: WHP (MPa)'.format(well)
                values *= 0.1
            elif parameter == 'DHP (barg)':
                if labs:
                    label = labs[i]
                else:
                    label = '{}: DHP (MPa)'.format(well)
                values *= 0.1
            elif parameter == 'Depth':
                if labs:
                    label = labs[i]
                else:
                    label = 'NM10 Drilling depth'
            else:
                if labs:
                    label = labs[i]
                else:
                    label = '{}: {}'.format(well, parameter)
            if parameter == 'Injectivity':
                # Adjust II unit to MPa (unconventional though...?)
                ax1a.scatter(dtos, values, label=label, color=colr, s=0.4,
                             facecolor=colr, zorder=3)
            else:
                ax1a.plot(dtos, values, label=label, color=colr, linewidth=1.5,
                          zorder=3)
            if twin_ax:
                ax1a.legend()
        if parameter in ['WHP (bar)', 'WHP (barg)']:
            ax1a.set_ylabel('WHP (MPa)', fontsize=16)
        elif parameter == 'DHP (barg)':
            ax1a.set_ylabel('DHP (MPa)', fontsize=16)
        elif parameter == 'Injectivity':
            ax1a.set_ylabel('Injectivity (t/h/MPa)', fontsize=16)
        elif parameter == 'Depth':
            ax1a.set_ylabel('Depth (m bsl)', fontsize=16)
        else:
            ax1a.set_ylabel(parameter, fontsize=16)
        if ylims:
            ax1a.set_ylim(ylims)
        else:
            ax1a.set_ylim([0, max(maxs) * 1.2])
    if tick_colors:
        ax1a.yaxis.label.set_color(colr)
        ax1a.tick_params(axis='y', colors=colr)
    if outdir:
        # Not plotting if just writing to outfile
        return
    # No legend if not twinning axis
    if twin_ax:
        # Add the new handles to the prexisting ones
        if len(handles) == 0:
            print('Plotting on empty axes. No handles to add to.')
            ax1a.legend(fontsize=12, loc=2)
        else:
            handles.extend(ax1a.legend_.get_lines())
            # Redo the legend
            if len(handles) > 4:
                ax1a.legend(handles=handles, fontsize=12, loc=2)
            else:
                ax1a.legend(handles=handles, loc=2, fontsize=12)
    # Now plot formatting
    if not plain:
        ax.set_xlim(start, end)
    else:
        fig.autofmt_xdate()
    if not ylims:
        plt.ylim(ymin=0) # Make bottom always zero
    plt.tight_layout()
    # Remove margins
    ax.margins(0, 0)
    if show:
        plt.show()
    return ax1a, values

def format_well_data(well_file, depth='mRF'):
    """
    Helper to format well txt files into (lat, lon, depth(km)) tups
    :param well_file: Well txt file
    :param depth: Returning depth as m RF or m bsl
    :return: list of tuples
    """
    pts = []
    if depth == 'mRF':
        with open(well_file) as f:
            rdr = csv.reader(f, delimiter=' ')
            for row in rdr:
                if row[2] == '0':
                    pts.append((float(row[1]), float(row[0]),
                                float(row[4]) / 1000.))
                else:
                    pts.append((float(row[1]), float(row[0]),
                                float(row[3]) / 1000.))
    elif depth == 'mbsl':
        with open(well_file) as f:
            rdr = csv.reader(f, delimiter=' ')
            for row in rdr:
                if row[2] == '0':
                    pts.append((float(row[1]), float(row[0]),
                                float(row[-1]) / -1000.))
                else:
                    pts.append((float(row[1]), float(row[0]),
                                float(row[-1]) / -1000.))
    return pts

def calculate_pressure(D, r, q0, qt, t, t0=None):
    """
    Internal function to calculate pore fluid pressure analytically using
    Dinske 2010 eq 2.

    :param D: Diffusivity (m^2/s)
    :param r: Radius (m)
    :param q0: Initial pressure at source (Pa)
    :param qt: Rate of pressure increase (Pa/s)
    :param t: Time (seconds)
    :param t0: Shut-in time (optional)
    :return:
    """
    if t == 0:
        return q0
    term1 = ((q0 + (qt * t)) / 4 * np.pi * D * r) + \
        ((qt * r) / (8 * np.pi * (D**2)))
    # Complement to the Gaussian error function
    erfc = special.erfc(r / np.sqrt(4 * D * t))
    term2 = ((qt * np.sqrt(t)) /
             4 * ((np.pi * D)**1.5)) * \
            np.exp(-1 * r**2 / 4 * D * t)
    prt = (term1 * erfc) - term2
    return prt

def plot_pressure_rt(q0, qt, diffs, dates, dists, show=True, norm=True):
    """
    Plot pressure with distance and time from injection point assuming linear
    pore pressure diffusion

    :param p0: Pressure perturbation at injection point
    :param dates: Start and end dates to plot for (will be hourly)
    :param dists: Distances to plot time profiles for
    :return:
    """
    # Make the data
    t = pd.to_datetime(pd.date_range(dates[0].datetime, dates[1].datetime,
                                     freq='H'))
    d = np.linspace(0, max(dists), 100) # 100 intervals for plotting
    plot_dict = {'time': {}, 'dist': {}}
    tot_seconds = (t[-1] - t[0]).total_seconds()
    print(tot_seconds)
    max_WHP = q0 + (tot_seconds * qt)
    for diff in diffs:
        plot_dict['time'][diff] = {}
        for dist in dists:
            plot_dict['time'][diff][dist] = [
                calculate_pressure(D=diff, r=dist, q0=q0, qt=qt, t=ti * 3600.)
                for ti in range(1,len(t))]
    for diff in diffs:
        plot_dict['dist'][diff] = {}
        for ti in range(10, len(t), 80): # Every 80 hours for time steps
            plot_dict['dist'][diff][ti] = [
                calculate_pressure(D=diff, r=di, q0=q0, qt=qt, t=ti * 3600.)
                for di in d[1:]]
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    for diff, dict in plot_dict['time'].items():
        for dist, ps in dict.items():
            # normalize log of ps
            ys = np.log10(np.array(ps))
            if norm:
                ys /= np.log10(max_WHP)
            ax1.plot([t for t in range(len(ps))], ys,
                     label='D={}, r={} m'.format(diff, dist))
    for diff, dict in plot_dict['dist'].items():
        for ti, ps in dict.items():
            # normalize log of ps
            ys = np.log10(np.array(ps))
            if norm:
                ys /= np.log10(max_WHP)
            ax2.plot([di for di in d[1:]], ys,
                     label='D={}, t={} h'.format(diff, ti))
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Normalized log10 pore pressure')
    ax2.set_xlabel('Distance from injection point (m)')
    ax2.set_ylabel('Normalized log10 pore pressure')
    if norm:
        ax1.set_ylim([0, 1.5])
        ax2.set_ylim([0, 1.5])
    else:
        ax1.set_ylim([-10, 10])
        ax2.set_ylim([-10, 10])
    ax1.legend()
    ax2.legend()
    if show:
        plt.tight_layout()
        plt.show()
    return ax1, ax2

def plot_event_well_dist(catalog, well_fzs, flow_dict, centroid=False,
                         temp_list='all', thickness=None, method='scatter',
                         boxplots=False, dates=None, ylim=None, title=None,
                         show=False, axes=None):
    """
    Function to plot events with distance from well as a function of time.
    :param cat: catalog of events
    :param well_fzs: text file of xyz feedzone pts
        e.g. NM08 bottom hole: (176.1788 -38.5326 2.75)
        e.g. NM08 middle hole: (-38.5326, 176.1788, 2.0)
        e.g. NM08 main fz: (-38.5326, 176.1788, 2.00)
        e.g. NM10 main fz: (-38.5673, 176.1893, 2.128)
        e.g. NM09 main fz: (-38.5358, 176.1857, 2.45)
        e.g. NM06 main fz: (-38.5653, 176.1948, 2.88)
        e.g. RK24 main fz: (-38.615, 176.203, 2.515) (Middle of FZ: 2476-2550)
        *** DEPTHS HERE NEED TO BE M BSL!!! ***
    :param flow_dict: Dictionary of flow starts, stops and D, for example:
        {'start': {D: {planar: [starttime, endtime]}},
         'end': {D: {iso: [starttime, endtime, start_injection]}}}
    :param centroid: Boolean to calculate distance from centroid of seismicity
    :param temp_list: list of templates for which we'll plot detections
    :param thickness: Thickness of aquifer in m if planar flow being plotted
    :param method: plot the 'scatter' or daily 'average' distance or both
    :param boxplots: Plot a daily boxplot of distance distributions?
    :param dates: List of starttime and endtime for catalog and plot
    :param ylim: List of min and max values for yaxis
    :param title: Plot title?
    :param show: Show the plot?
    :param axes: Axes object to plot into

    :return: matplotlib.pyplot.Axes
    """
    if type(well_fzs) == str:
        well_pts = format_well_data(well_fzs)
    elif type(well_fzs) == tuple:
        well_pts = [well_fzs]
    elif type(well_fzs) == list:
        well_pts = well_fzs
    elif centroid:
        print('Using catalog centroid as feedzone')
    else:
        print('Well feedzones should be either a file with feedzones or'
              + ' an xyz tuple')
        return
    colors = cycle(['brickred', 'magenta', 'purple'])
    # Grab only templates in the list
    cat = Catalog()
    filt_cat = Catalog()
    if dates:
        filt_cat.events = [ev for ev in catalog
                           if dates[0] < ev.origins[-1].time < dates[1]]
    else:
        filt_cat = catalog
    cat.events = [ev for ev in filt_cat if
                  str(ev.resource_id).split('/')[-1].split('_')[0] in
                  temp_list or temp_list == 'all']
    time_dist_tups = []
    cat_start = min([ev.origins[-1].time.datetime for ev in cat])
    cat_end = max([ev.origins[-1].time.datetime for ev in cat])
    if centroid:
        # Pretend the cluster centroid is a feedzone
        os = [ev.preferred_origin() for ev in cat
              if ev.preferred_origin().method_id]
        lats = [o.latitude for o in os
                if o.method_id.id.endswith('GrowClust')]
        lons = [o.longitude for o in os
                if o.method_id.id.endswith('GrowClust')]
        dps = [o.depth for o in os
               if o.method_id.id.endswith('GrowClust')]
        well_pts = [(np.median(lats), np.median(lons), np.median(dps) / 1000.)]
        print(well_pts)
    for ev in cat:
        # Only do distance calculation for GrowClust origins
        if ev.preferred_origin():
            o = ev.preferred_origin()
            if o.method_id:
                if o.method_id.id.endswith('GrowClust'):
                    dist = min([dist_calc((o.latitude, o.longitude,
                                           o.depth / 1000.),
                                          pt) * 1000. for pt in well_pts])
                    time_dist_tups.append((o.time.datetime, dist))
    times, dists = zip(*time_dist_tups)
    # Make DataFrame for boxplotting
    dist_df = pd.DataFrame()
    dist_df['dists'] = pd.Series(dists, index=times)
    # Add daily grouping column to df (this is crap, but can't find better)
    dist_df['day_num'] =  [mdates.date2num(
        dto.replace(hour=12, minute=0, second=0,
                    microsecond=0).to_pydatetime())
                           for dto in dist_df.index]
    dist_df['dto_num'] =  [mdates.date2num(dt) for dt in dist_df.index]
    # Now create the pressure envelopes
    # Creating hourly datetime increments
    diff_ys = []
    ts = []
    labs = []
    for tb, flow_d in flow_dict.items():
        for D, g_dict in flow_d.items():
            for geom, tlist in g_dict.items():
                start = pd.Timestamp(tlist[0].datetime)
                if tlist[1]:
                    end = pd.Timestamp(tlist[-1].datetime)
                else:
                    end = pd.Timestamp(cat_end)
                t = pd.to_datetime(pd.date_range(start, end, freq='H'))
                tint = [mdates.date2num(d) for d in t]
                ts.append(tint)
                # Now diffusion y vals
                # Isotropic diffusion
                if geom.lower() == 'isotropic':
                    # Shapiro and Dinske 2009 (and all other such citations)
                    if tb == 'start': # Triggering front (Shapiro)
                        diff_ys.append([np.sqrt(3600 * D * i * 4. * np.pi)
                                        for i in range(len(t))])
                    elif tb == 'end': # Backfront (Parotidis 2004)
                        duration = tlist[0] - tlist[2] # Seconds of injection
                        diff_y = []
                        for i in range(1, len(t)):
                            secs_tot = (i * 3600) + duration
                            diff_y.append(
                                np.sqrt(secs_tot * D * 6. *
                                        ((secs_tot / duration) - 1) *
                                        np.log(secs_tot /
                                               (secs_tot - duration))))
                        diff_y.insert(0, 0)
                        diff_ys.append(diff_y)
                if geom.lower() == 'planar':
                    d = thickness # thickness of fz in meters
                    diff_ys.append([3600 * D * i / (2 * d) * np.pi
                                    for i in range(len(t))])
                elif geom.lower() == 'cubic':
                    # Yeilds volume of affected area. We will assume spherical
                    # for simplicity
                    diff_ys.append([0.5 * ((3600 * i * 100. / 0.2)**(1/3.))
                                    for i in range(len(t))])
                if tb == 'start':
                    labs.append('{} trig. front: D={} m$^2$/s'.format(geom, D))
                elif tb == 'end':
                    labs.append('{} back-front: D={} m$^2$/s'.format(geom, D))
    # Plot 'em up
    if not axes:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        ax = axes
    # First boxplots
    if boxplots:
        u_days = list(set(dist_df.day_num))
        bins = [dist_df.loc[dist_df['day_num'] == d]['dists'].values
                for d in u_days]
        good_bins = []
        positions = []
        for i, b in enumerate(bins):
            if len(b) > 5:
                good_bins.append(b)
                positions.append(u_days[i])
        bplots = ax.boxplot(good_bins, positions=positions, patch_artist=True,
                            flierprops={'markersize': 0}, manage_xticks=False,
                            widths=0.7)
        for patch in bplots['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.5)
    # Then diffusions
    for i, diff_y in enumerate(diff_ys):
        if 'trig' in labs[i]: #Triggering
            ax.plot(ts[i], diff_y, label=labs[i])
        elif 'back' in labs[i]: # Backfront
            ax.plot(ts[i], diff_y, '--', label=labs[i])
    # Now events
    if method != 'scatter':
        dates = []
        day_avg_dist = []
        for date in date_generator(cat_start, cat_end):
            dates.append(date)
            tdds = [tdd[1] for tdd in time_dist_tups if tdd[0] > date
                    and tdd[0] < date + timedelta(days=1)]
            day_avg_dist.append(np.mean(tdds))
    if method == 'scatter':
        ax.scatter(times, dists, color='gray', label='Event', s=10, alpha=0.5)
    elif method == 'average':
        ax.plot(dates, day_avg_dist)
    elif method == 'both':
        ax.scatter(times, dists)
        ax.plot(dates, day_avg_dist, color='r')
    # Plot formatting
    # fig.autofmt_xdate()
    ax.legend(fontsize=12, loc=2)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([0, 3000])
    if title:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title('Fluid diffusion envelopes with time', fontsize=20)
    if dates:
        ax.set_xlim([dates[0].datetime, dates[1].datetime])
    else:
        ax.set_xlim([min(t), max(t)])
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Distance (m)', fontsize=16)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=15)
    if show:
        fig.show()
    return ax


def place_Rot_times(fig=None, method='lines'):
    """
    Plot shutdown times, RK23 shutdown/startup
    :param fig: matplotlib Figure object to plot onto
    :param method: Can be 'lines', 'spans' or 'both'
    :return:
    """
    # Hardcoded important dates in UTC (NZDT - 12)
    # Shutdowns
    SD1 = [
        pytz.utc.localize(UTCDateTime(2012, 1, 16).datetime),
        pytz.utc.localize(UTCDateTime(2012, 1, 28).datetime)
    ]
    SD2 = [
        pytz.utc.localize(UTCDateTime(2012, 10, 24).datetime),
        pytz.utc.localize(UTCDateTime(2012, 10, 29).datetime)
    ]
    SD3 = [
        pytz.utc.localize(UTCDateTime(2013, 10, 17).datetime),
        pytz.utc.localize(UTCDateTime(2013, 11, 3).datetime)
    ]
    SD4 = [
        pytz.utc.localize(UTCDateTime(2013, 11, 28).datetime),
        pytz.utc.localize(UTCDateTime(2013, 12, 9).datetime)
    ]
    SD5 = [
        pytz.utc.localize(UTCDateTime(2014, 4, 13).datetime),
        pytz.utc.localize(UTCDateTime(2014, 4, 17).datetime)
    ]
    SD6 = [
        pytz.utc.localize(UTCDateTime(2014, 6, 23).datetime),
        pytz.utc.localize(UTCDateTime(2014, 6, 25).datetime)
    ]
    SD7 = [
        pytz.utc.localize(UTCDateTime(2014, 10, 10).datetime),
        pytz.utc.localize(UTCDateTime(2014, 10, 23).datetime)
    ]
    SD8 = [
        pytz.utc.localize(UTCDateTime(2015, 7, 20).datetime),
        pytz.utc.localize(UTCDateTime(2015, 8, 9).datetime)
    ]
    RK23_on = [
        pytz.utc.localize(UTCDateTime(2012, 11, 10).datetime),
    ]
    RK23_off = [
        pytz.utc.localize(UTCDateTime(2013, 7, 4).datetime),
    ]
    handles = [] # Only add handles to the well axes...I guess
    for ax in fig.axes:
        # Hard code only specific axes on which to plot spans
        if method in ['both', 'spans'] and any(
                [s in ax.get_ylabel() for s in ['Flow', '#', 'Events']]):
            for i, spn in enumerate([SD1, SD2, SD3, SD4, SD5, SD6, SD7, SD8]):
                if i == 7: # Only label final span for compact legend
                    ax.axvspan(spn[0], spn[1], alpha=0.3, color='firebrick',
                               label='Plant shutdown')
                else:
                    ax.axvspan(spn[0], spn[1], alpha=0.3, color='firebrick')
        if method in ['both', 'lines']:
            # Shade region of RK23 shutdown
            ax.axvspan(RK23_on, RK23_off, color='darkgray', linestyle='-.',
                       label='RK23 shutdown', alpha=0.2)
        if any([s in ax.get_ylabel() for s in ['WHP', 'Flow']]):
            handles.extend(ax.legend().get_lines())
            if 'Flow' in ax.get_ylabel():
                handles.extend(ax.legend().get_patches())
            if isinstance(ax.legend_, matplotlib.legend.Legend):
                ax.legend_.remove() # Need to manually remove this
    ax.legend(handles=handles, fontsize=12, loc=1)
    return


def place_NM08_times(fig=None, fill_between=False, diffs=False,
                     fill_color=False, fill_hatch=True, lines=False):
    """
    Place the time labels as axvline for NM08 stimulation
    :param ax: matplotlib.Axes to plot on
    :param fig: matplotlib.Figure in the case of fill_between where we'll
        find the WHP and II axes and fill the correct intervals
    :param fill_between: Whether to fill the periods of interest
    :param diffs: If this is a diffusion axes, shift up the vertical lines
    :return:
    """
    # Hardcoded important dates in UTC (NZDT - 12)
    ph1_start = pytz.utc.localize(
        UTCDateTime(2012, 6, 7, 7, 12).datetime)
    max_pres = pytz.utc.localize(
        UTCDateTime(2012, 6, 9, 14, 22).datetime)
    breakth = pytz.utc.localize(
        UTCDateTime(2012, 6, 13, 2, 10).datetime)
    whp_mod = pytz.utc.localize(
        UTCDateTime(2012, 6, 15, 5, 10).datetime)
    ph1_seis = pytz.utc.localize(
        UTCDateTime(2012, 6, 16, 23, 22).datetime)
    ph1_end = pytz.utc.localize(
        UTCDateTime(2012, 6, 25, 22, 15).datetime)
    ph2_start = pytz.utc.localize(
        UTCDateTime(2012, 6, 30, 22).datetime)
    ph2_seis = pytz.utc.localize(
        UTCDateTime(2012, 7, 6, 2, 50).datetime)
    ph2_end = pytz.utc.localize(
        UTCDateTime(2012, 7, 9, 4, 50).datetime)
    if diffs:
        corr = 0.15
    else:
        corr = 0.
    if fill_between and fig:
        for ax in fig.axes:
            if len(ax.collections) > 0:
                if 'Injectivity' in ax.collections[0].get_label():
                    print('Filling II plot')
                    xs, ys = zip(*ax.collections[0].get_offsets())
                    xdt = [mdates.num2date(nm, tz=pytz.UTC) for nm in xs]
                    datas = list(zip(xdt, ys))
                else:
                    datas = None
            elif 'MPa' in ax.lines[0].get_label():
                print('Filling WHP plot')
                # Fill below the WHP curve or II curve(?) for periods of interest
                xs, ys = ax.lines[0].get_data()
                datas = list(zip(xs, ys))
            else:
                datas = None
            if datas:
                plot_data = []
                labs = []
                # Trying a lot of these guys...
                # colors = cycle(sns.color_palette(["#9b59b6", "#3498db",
                #                                   "#95a5a6", "#2ecc71"]))
                # colors = cycle(sns.color_palette('deep'))
                # colors = cycle(sns.color_palette("GnBu_d", 4))
                colors = cycle(sns.color_palette(['dimgray', 'lightgray',
                                                  'dimgray', 'lightgray']))
                hatches = cycle(['+', 'x', 'o', '*'])
                # Near-field pressurization
                nf = [dat for dat in datas if ph1_start < dat[0] < breakth]
                x, y = zip(*nf)
                plot_data.append((x, y))
                labs.append('Near-field press.')
                # Breakthrough
                bt = [dat for dat in datas if breakth < dat[0] < ph1_seis]
                x, y = zip(*bt)
                plot_data.append((x, y))
                labs.append('Breakthrough')
                # FZ pressurization
                fzp = [dat for dat in datas if ph1_seis < dat[0] < ph1_end]
                x, y = zip(*fzp)
                plot_data.append((x, y))
                labs.append('Fracture-zone press.')
                #Repressurization
                rpres = [dat for dat in datas if ph2_start < dat[0] < ph2_seis]
                x, y = zip(*rpres)
                plot_data.append((x, y))
                labs.append('Repressurization')
                # FZ pressurization pt2
                fzp2 = [dat for dat in datas if ph2_seis < dat[0] < ph2_end]
                x, y = zip(*fzp2)
                plot_data.append((x, y))
                labs.append('Fracture-zone press.')
                for i, datz in enumerate(plot_data):
                    if fill_color:
                        # 3rd and 5th category should have same color
                        if i == 2:
                            fzp_col = next(colors)
                            col = fzp_col
                        elif i == 4:
                            col = fzp_col
                        else:
                            col = next(colors)
                        ax.fill_between(datz[0], datz[1], color=col,
                                        alpha=0.2, edgecolor=None, zorder=1)
                    elif fill_hatch:
                        ax.fill_between(datz[0], datz[1], edgecolor='gray',
                                        hatch=next(hatches), facecolor="none",
                                        zorder=1)
    if lines: # Just put some vertical lines on it
        for ax in fig.axes:
            # Place lines
            if len([line for line in ax.lines
                    if 'WHP' in line.get_label()]) > 0:
                ax.hlines(y=2.06, xmin=ph1_seis, xmax=ph2_seis,
                           linestyles=':', colors='darkgray')
            ax.axvline(ph1_start, linestyle='--', color='gray', linewidth=1.0)
            ax.axvline(breakth, linestyle='--', color='gray', linewidth=1.0)
            ax.axvline(ph1_seis, linestyle='--', color='gray', linewidth=1.0)
            ax.axvline(ph1_end, linestyle='--', color='gray', linewidth=1.0)
            ax.axvline(ph2_start, linestyle='--', color='gray', linewidth=1.0)
            ax.axvline(ph2_seis, linestyle='--', color='gray', linewidth=1.0)
            ax.axvline(ph2_end, linestyle='--', color='gray', linewidth=1.0)
        # # Place text
        # ax.text(UTCDateTime(2012, 6, 7).datetime, (0.61 - corr) * yz[1], 'T1',
        #         horizontalalignment='center', fontsize=10)
        # ax.text(UTCDateTime(2012, 6, 15).datetime, (0.61 - corr) * yz[1], 'T2',
        #         horizontalalignment='center', fontsize=10)
        # ax.text(UTCDateTime(2012, 6, 26).datetime, (0.71 - corr) * yz[1], 'T3',
        #         horizontalalignment='center', fontsize=10)
        # ax.text(UTCDateTime(2012, 7, 1).datetime, (0.86 - corr) * yz[1], 'T4',
        #         horizontalalignment='center', fontsize=10)
        # ax.text(UTCDateTime(2012, 7, 6).datetime, (0.91 - corr) * yz[1], 'T5',
        #         horizontalalignment='center', fontsize=10)
    return

