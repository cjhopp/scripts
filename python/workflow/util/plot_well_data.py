#!/usr/bin/python
"""
Plotting functions for well related data (reorganizing to clear up
plot_detections.py
"""
import matplotlib
import csv

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

from itertools import cycle
from scipy import special
from datetime import timedelta
from obspy import Catalog, UTCDateTime
from obspy.imaging.beachball import beach
from focal_mecs import beach_mod
from eqcorrscan.utils.mag_calc import dist_calc
from shelly_mags import local_to_moment_Majer
from obspy.imaging.scripts.mopad import MomentTensor as mopad_MT
from obspy.imaging.scripts.mopad import BeachBall as mopad_BB


def date_generator(start_date, end_date):
    # Generator for date looping
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def plot_II(excel_file, wells, P_type, period):
    """
    Plot the II in log-log space for the given wells using the specified
    pressure measurement type.
    :param excel_file: Path to excel file
    :param wells: List of well names to plot
    :param P_type: DHP or WHP
    :param period: For now, either stimulation or
    :return:
    """
    if (sheetname in ['NM10 Stimulation', 'NM09 Stimulation']
        and parameter == 'Injectivity'):
        # Combine sheets for DHP and Flow with different samp rates into one
        df = pd.read_excel(excel_file, header=[0, 1], sheetname=sheetname)
        df2 = pd.read_excel(excel_file, header=[0, 1],
                            sheetname='NM10 Stimulation DHP')
        df[('NM10', 'DHP (barg)')] = df2[('NM10', 'DHP (barg)')].asof(df.index)
    else:
        df = pd.read_excel(excel_file, header=[0, 1], sheetname=sheetname)

    return

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
    ax1.invert_yaxis()
    ax1.set_xlim((0, 300))
    if feedzones:
        xlims = ax1.get_xlim()
        xz = [xlims[0], xlims[1], xlims[1], xlims[0]]
        for fz in feedzones:
            yz = [fz[0], fz[0], fz[1], fz[1]]
            ax1.fill(xz, yz, color='lightgray', zorder=0,
                     alpha=0.5)
            if fz_labels:
                ax1.text(200., (fz[0] + fz[1]) / 2., 'Feedzone', fontsize=8,
                         color='gray', verticalalignment='center')
    ax1.set_ylabel('Depth (m bsl)', fontsize=16)
    ax1.set_xlabel(r'Temperature ($\degree$C)', fontsize=16)
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('NST & Injection Temperatures')
    ax1.legend()
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
                sdrs[line[0]] = (float(line[1]), float(line[2]),
                                 float(line[3]))
            elif cat_format == 'templates':
                sdrs[line[0].split('.')[0]] = (float(line[1]), float(line[2]),
                                               float(line[3]))
    return sdrs

def plot_well_seismicity(catalog, wells, profile='NS', dates=None, color=True,
                         ax=None, show=False, outfile=None, feedzones=None,
                         fz_labels=True, focal_mechs=None, cat_format=None,
                         fm_color='b', ylims=None, dd_only=True,
                         colorbar=False, c_axes=None):
    """
    Plot well with depth and seismicity

    :param catalog: Catalog of seismicity
    :param wells: List of strings specifying well names to plot
    :param profile: 'NS', 'EW', or 'map'
    :param dates: Start and end dates for the catalog
    :param color: Whether to color seismicity by date of occurrence
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
    :return:
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
                                  and not ev.preferred_origin().quality
                                  and dates[0] < ev.preferred_origin().time
                                  < dates[1]])
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    well_pt_lists = []
    # Dictionary of fm strike-dip-rake from Arnold/Townend pkg
    if focal_mechs:
        sdrs = read_fm_file(focal_mechs, cat_format)
        fm_tup = []
        for ev in catalog:
            if cat_format == 'detections':
                fm_id = '{}.{}.{}'.format(
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
    for well in wells:
        well_file = '/home/chet/gmt/data/NZ/wells/{}_xyz_pts.csv'.format(well)
        # Grab well pts (these are depth (kmRF)) correct accordingly
        well_pt_lists.append(format_well_data(well_file))
    t0 = catalog[0].preferred_origin().time.datetime
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
    # Plot seismicity
    if color:
        col = days # Days since first event
    else:
        col = 'darkgray'
    if profile == 'NS':
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
    if colorbar:
        if not c_axes:
            cbar = plt.colorbar(sc, ax=ax1)
        else:
            cbar = plt.colorbar(sc, cax=c_axes, orientation='horizontal',
                                format='%d')
        cbar.set_label('Elapsed days', fontsize=14)
    for i, well_pts in enumerate(well_pt_lists):
        # Elevation of wellhead correction
        if well in ['NM08', 'NM09']:
            elevation = 350.
        elif well in ['NM06', 'NM10']:
            elevation = 372.
        wlat, wlon, wkm = zip(*well_pts)
        wdp = (np.array(wkm) * 1000.) - elevation
        if profile == 'NS':
            ax1.plot(wlat, wdp, color=next(colors),
                     label='{} wellbore'.format(wells[i]))
        elif profile == 'EW':
            ax1.plot(wlon, wdp, color=next(colors),
                     label='{} wellbore'.format(wells[i]))
        elif profile == 'map':
            ax1.plot(wlon, wlat, color=next(colors),
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
                if profile == 'NS':
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
    half_width = 0.02
    # Now center on wellhead position (should work for last wlat as only
    # wells from same wellpad should be plotted this way)
    if profile == 'NS':
        ax1.set_xlim([wlat[0] + half_width, wlat[0] - half_width])
    elif profile == 'EW':
        # Center axes on wellbore
        ax1.set_xlim([wlon[0] - half_width, wlon[0] + half_width])
    elif profile == 'map':
        ax1.set_xlim([wlon[0] - half_width, wlon[0] + half_width])
        ax1.set_ylim([wlat[0] - half_width, wlat[0] + half_width])
    if profile != 'map' and not ylims: # Adjust depth limits depending on well
        if 'NM10' in wells or 'NM06' in wells:
            ax1.set_ylim([4000., 0.])
        else:
            ax1.set_ylim([4000., 0.])
    elif profile != 'map' and ylims:
        ax1.set_ylim(ylims)
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

def plot_volume_Mmax(plot_moment=False, plot_McGarr=True, show=True):
    """
    Just a way of jotting down data from Figure 3b in Geobel&Brodsky for
    comparison with Merc data

    .. note: May expand this to take arguments for well and dates to pass to
        plot_well_data to get exact cumulative sums
    :return:
    """
    fig, ax = plt.subplots()
    data_dict = {
        'Ml': {'Basel': [11570, 3.4], 'St Gallen': [1165, 3.5],
               'Fenton Hill 83': [21600, 1.0], 'Fenton Hill 86': [37000, 1.0],
               'KTB': [4000, 0.7], 'Soultz': [4.4E4, 2.],
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
    if plot_McGarr:
        mmax = []
        vols = np.linspace(1E2, 1E7, 100)
        for v in vols:
            mmax.append(v * 3E10) # G = 3E10 Pa from McGarr 2014
        ax.plot(vols, mmax, '--', color='lightgray')
        ax.text(x=2E5, y=1E16, s='$M_{o}=GV$', rotation=28., color='lightgray',
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

def plot_well_data(excel_file, sheetname, parameter, well_list, loglog=False,
                   color=False, cumulative=False, ax=None, dates=None,
                   show=True, ylims=False, outdir=None, figsize=(8, 6),
                   hall_plot=True):
    """
    New flow/pressure plotting function utilizing DataFrame functionality
    :param excel_file: Excel file to read
    :param sheetname: Which sheet of the spreadsheet do you want?
    :param parameter: Either 'WHP (bar)' or 'Flow (t/h)' at the moment
    :param well_list: List of wells you want plotted
    :param cumulative: Plot the total injected volume?
    :param ax: If plotting on existing Axis, pass it here
    :param dates: Specify start and end dates if plotting to preexisting
        empty Axes.
    :param show: Are we showing this Axis automatically?
    :param ylims: To force the ylims for the well data.
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
    else:
        df = pd.read_excel(excel_file, header=[0, 1], sheetname=sheetname)
    # All flow info is local time
    df.index = df.index.tz_localize('Pacific/Auckland')
    print(df.index[0])
    # Convert it to UTC
    df.index = df.index.tz_convert(None)
    print(df.index[0])
    colors = cycle(sns.color_palette())
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
        # THIS IS FUCKED UNLESS YOU ACCOUNT FOR SAMPLING RATE
        # FLow reported as T/h, so must be corrected to jive with samp rate
        # (sec, hr, day, etc...)
        # Stimulations are 5 min samples (roughly) (i.e. / 12.)
        # Post-startup are daily (i.e. * 24.)
        # Manually set sampling rate-related multiplier
        if 'Stimulation' in sheetname.split():
            multiplier = 1 / 12. # Stimulations will be overestimated
        else:
            multiplier = 24 # NM10 Losses and Post-startup will by underestimated
        maxs = []
        if hall_plot:
            plt.close('all')
            fig, (ax1a, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        else:
            ax1a = ax.twinx()
        for i, well in enumerate(well_list):
            if color:
                colr = color
            else:
                colr = next(colors)
            dtos = df.xs((well, parameter), level=(0, 1),
                         axis=1).index.to_pydatetime()
            values = df.xs((well, parameter), level=(0, 1),
                           axis=1).cumsum() * multiplier
            if outdir:
                # Write to file
                filename = 'Cumulative_flow_{}'.format(well)
                with open('{}/{}.csv'.format(outdir, filename), 'w') as f:
                    for dto, val in zip(dtos, values):
                        f.write('{} {}'.format(dto.strftime('%Y-%m-%d'), val))
                continue
            ax1a.plot(dtos, values, label='{}: {}'.format(well,
                                                          'Cumulative Vol.'),
                      color=next(colors))
            plt.legend() # This is annoying
            maxs.append(np.max(df.xs((well, parameter),
                               level=(0, 1), axis=1).values))
        ax1a.set_ylabel('Cumulative Volume (Tonnes)', fontsize=16)
        # Force scientific notation for cumulative y axis
        ax1a.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        if hall_plot:
            cum_pres = df.xs((well, 'WHP (barg)'), level=(0, 1),
                             axis=1).cumsum() * multiplier
            ax2.plot(values, cum_pres, color='r')
            ax2.set_ylabel('Cumulative P (bar)')
            ax2.set_xlabel('Cumulative Volume (t)')
            ax2.set_title('Hall plot')
    else:
        # Loop over wells, slice dataframe to each and plot
        maxs = []
        if not plain and ax.get_ylim()[-1] != 1.0:
            ax1a = ax.twinx()
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
            if parameter == 'Injectivity':
                # Use WHP = Pr - pgz + W/II + KW^2 where W is flow rate
                # JC sets K to zero for NM08...
                Pr = 90  # Reservoir pressure (bar -roughly)
                # p water at 140C = 0.926 g/cm3 50C = 0.988 g/cm3
                # NM08 fz = 2400 m depth
                pgz = 940 * 2400 * 9.8 * 1e-5  # Pascal to bar
                # neglect friction for now XXX TODO
                if (well in ['NM10', 'NM09']
                    and sheetname.endswith('Stimulation')):
                    vals = df[(well, 'Flow (t/h)')] / df[(well, 'DHP (barg)')]
                else: # should happen for NM10 stimulation
                    denom = df[(well, 'WHP (barg)')] + pgz - Pr
                    vals = df[(well, 'Flow (t/h)')] / denom
                values = vals.where(vals < 1000.) # Careful with this shiz
                dtos = values.index.to_pydatetime()
            elif parameter == 'Depth' and sheetname == 'NM10 Losses':
                values = df[('NM10', 'Depth')] - 372. # Correcting to m bsl
                dtos = values.index.to_pydatetime()
            elif parameter == 'P_derivative':
                values = df[(well, 'WHP (barg)')]
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
            if color:
                colr = color
            else:
                colr = next(colors)
            # Force MPa instead of bar units
            if parameter in ['WHP (bar)', 'WHP (barg)']:
                label = '{}: WHP (MPa)'.format(well)
                values *= 0.1
            elif parameter == 'DHP (barg)':
                label = '{}: DHP (MPa)'.format(well)
                values *= 0.1
            elif parameter == 'Depth':
                label = 'NM10 Drilling depth'
            else:
                label = '{}: {}'.format(well, parameter)
            if parameter == 'Injectivity':
                if loglog:
                    ax1a.loglog(dtos, values)
                else:
                    # Adjust II unit to MPa (unconventional though...?)
                    ax1a.scatter(dtos, values * 10, label=label, color=colr, s=0.05)
            elif parameter == 'P_derivative':
                secs = [int((dto - dtos[0]).total_seconds())
                        for dto in dtos]
                dP = values.diff()
                dP = dP.where(dP > 0.)
                dp_secs = [(dt - dtos[0]).total_seconds()
                           for dt in dP.index.to_pydatetime()]
                ax1a.scatter(secs, values)
                ax1a.scatter(dp_secs, dP, label=label)
                ax1a.set_yscale('log')
                ax1a.set_xscale('log')
            else:
                ax1a.plot(dtos, values, label=label, color=colr, linewidth=1.0)
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
    if outdir:
        # Not plotting if just writing to outfile
        return
    # try:
    #     # Add the new handles to the prexisting ones
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
    # except UnboundLocalError:
    #     print('Plotting on empty axes. No handles to add to.')
    #     ax.legend()
    # Now plot formatting
    if not plain:
        ax.set_xlim(start, end)
    else:
        fig.autofmt_xdate()
    if not ylims:
        plt.ylim(ymin=0) # Make bottom always zero
    plt.tight_layout()
    if show:
        plt.show()
    return ax1a, values

def format_well_data(well_file):
    """
    Helper to format well txt files into (lat, lon, depth(km)) tups
    :param well_file: Well txt file
    :return: list of tuples
    """
    pts = []
    with open(well_file) as f:
        rdr = csv.reader(f, delimiter=' ')
        for row in rdr:
            if row[2] == '0':
                pts.append((float(row[1]), float(row[0]),
                            float(row[4]) / 1000.))
            else:
                pts.append((float(row[1]), float(row[0]),
                            float(row[3]) / 1000.))
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

def plot_event_well_dist(catalog, well_fzs, flow_dict, temp_list='all',
                         thickness=None, method='scatter', boxplots=False,
                         starttime=None, ylim=None, endtime=None, title=None,
                         show=True, axes=None):
    """
    Function to plot events with distance from well as a function of time.
    :param cat: catalog of events
    :param well_fzs: text file of xyz feedzone pts
        e.g. NM08 bottom hole: (176.1788 -38.5326 3.3615)
        e.g. NM08 middle hole: (-38.5326, 176.1788, 2.85)
        e.g. NM08 main fz: (-38.5326, 176.1788, 2.35)
        e.g. NM10 main fz: (-38.5673, 176.1893, 2.5)
        e.g. NM09 main fz: (-38.5358, 176.1857, 2.8)
        DEPTHS HERE ARE mRF
    :param flow_dict: Dictionary of flow starts, stops and D, for example:
        {'starts': {D: {planar: [starttime, endtime]}},
         'ends': {D: {iso: [starttime, endtime, start_injection]}}}
    :param temp_list: list of templates for which we'll plot detections
    :param thickness: Thickness of aquifer in m if planar flow being plotted
    :param method: plot the 'scatter' or daily 'average' distance or both
    :return: matplotlib.pyplot.Axes
    """
    if type(well_fzs) == str:
        well_pts = format_well_data(well_fzs)
    elif type(well_fzs) == tuple:
        well_pts = [well_fzs]
    elif type(well_fzs) == list:
        well_pts = well_fzs
    else:
        print('Well feedzones should be either a file with feedzones or'
              + ' an xyz tuple')
        return
    colors = cycle(['brickred', 'magenta', 'purple'])
    # Grab only templates in the list
    cat = Catalog()
    filt_cat = Catalog()
    if starttime and endtime:
        filt_cat.events = [ev for ev in catalog if ev.origins[-1].time
                           < endtime and ev.origins[-1].time >= starttime]
    else:
        filt_cat = catalog
    cat.events = [ev for ev in filt_cat if
                  str(ev.resource_id).split('/')[-1].split('_')[0] in
                  temp_list or temp_list == 'all']
    time_dist_tups = []
    cat_start = min([ev.origins[-1].time.datetime for ev in cat])
    cat_end = max([ev.origins[-1].time.datetime for ev in cat])
    for ev in cat:
        if ev.preferred_origin():
            o = ev.preferred_origin()
            dist = min([dist_calc((o.latitude, o.longitude,
                                   (o.depth + 350.) / 1000.),
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
                    labs.append('{} trig. front: D={} $m^2/s$'.format(geom, D))
                elif tb == 'end':
                    labs.append('{} back-front: D={} $m^2/s$'.format(geom, D))
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
        ax.set_ylim([0, 3])
    if title:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title('Fluid diffusion envelopes with time', fontsize=20)
    if starttime:
        ax.set_xlim([mdates.date2num(starttime.datetime),
                     mdates.date2num(endtime.datetime)])
    else:
        ax.set_xlim([min(t), max(t)])
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Distance (m)', fontsize=16)
    # fig.tight_layout()
    if show:
        fig.show()
    return ax

def place_NM08_times(ax, diffs=False):
    """
    Place the time labels as axvline for NM08 stimulation
    :param ax: matplotlib.Axes to plot on
    :return:
    """
    if diffs:
        corr = 0.15
    else:
        corr = 0.
    # Get data coords
    yz = ax.get_ylim()
    # Place lines
    ax.axvline(UTCDateTime(2012, 6, 7).datetime, ymax=0.6 - corr,
               linestyle='--', color='k', linewidth=0.7)
    ax.axvline(UTCDateTime(2012, 6, 15).datetime, ymax=0.6 - corr,
               linestyle='--', color='k', linewidth=0.7)
    ax.axvline(UTCDateTime(2012, 6, 26).datetime, ymax=0.7 - corr,
               linestyle='--', color='k', linewidth=0.7)
    ax.axvline(UTCDateTime(2012, 7, 1).datetime, ymax=0.85 - corr,
               linestyle='--', color='k', linewidth=0.7)
    ax.axvline(UTCDateTime(2012, 7, 6).datetime, ymax=0.90 - corr,
               linestyle='--', color='k', linewidth=0.7)
    # Place text
    ax.text(UTCDateTime(2012, 6, 7).datetime, (0.61 - corr) * yz[1], 'T1',
            horizontalalignment='center', fontsize=10)
    ax.text(UTCDateTime(2012, 6, 15).datetime, (0.61 - corr) * yz[1], 'T2',
            horizontalalignment='center', fontsize=10)
    ax.text(UTCDateTime(2012, 6, 26).datetime, (0.71 - corr) * yz[1], 'T3',
            horizontalalignment='center', fontsize=10)
    ax.text(UTCDateTime(2012, 7, 1).datetime, (0.86 - corr) * yz[1], 'T4',
            horizontalalignment='center', fontsize=10)
    ax.text(UTCDateTime(2012, 7, 6).datetime, (0.91 - corr) * yz[1], 'T5',
            horizontalalignment='center', fontsize=10)
    return

