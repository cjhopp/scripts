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

from itertools import cycle
from datetime import timedelta
from obspy import Catalog, UTCDateTime
from eqcorrscan.utils.mag_calc import dist_calc

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
             outfile=False):
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
    ax1.set_ylabel('Depth (m bsl)', fontsize=16)
    ax1.set_xlabel(r'Temperature ($\degree$C)', fontsize=16)
    if title:
        plt.title(title)
    else:
        plt.title('NST & Injection Temperatures')
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    elif outfile:
        plt.savefig(outfile, dpi=300)
    return ax1

def plot_well_data(excel_file, sheetname, parameter, well_list, loglog=False,
                   color=False, cumulative=False, ax=None, dates=None,
                   show=True, ylims=False, outdir=None, figsize=(8, 6)):
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
    # Convert it to UTC
    df.index = df.index.tz_convert(None)
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
        if parameter != 'Flow (t/h)':
            print('Will not plot cumulative %s. Only for Flow (t/h)'
                  % parameter)
            return
        maxs = []
        ax1a = ax.twinx()
        for i, well in enumerate(well_list):
            if color:
                colr = color
            else:
                colr = next(colors)
            dtos = df.xs((well, parameter), level=(0, 1),
                         axis=1).index.to_pydatetime()
            values = df.xs((well, parameter), level=(0, 1), axis=1).cumsum()
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
        plt.gca().set_ylabel('Cumulative Volume (Tonnes)', fontsize=16)
        # Force scientific notation for cumulative y axis
        plt.gca().ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
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
                if (well in ['NM10', 'NM09']
                    and sheetname.endswith('Stimulation')):
                    # Use WHP = Pr - pgz + W/II + KW^2 where W is flow rate
                    # JC sets K to zero for NM08...TODO XXX
                    vals = df[(well, 'Flow (t/h)')] / df[(well, 'DHP (barg)')]
                else: # should happen for NM10 stimulation
                    vals = df[(well, 'Flow (t/h)')] / df[(well, 'WHP (barg)')]
                values = vals.where(vals < 1000.) # Careful with this shiz
                dtos = values.index.to_pydatetime()
            elif parameter == 'Depth' and sheetname == 'NM10 Losses':
                values = df[('NM10', 'Depth')]
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
                    ax1a.scatter(dtos, values, label=label, color=colr, s=0.05)
            else:
                ax1a.plot(dtos, values, label=label, color=colr, linewidth=1.0)
            ax1a.legend()
        if parameter in ['WHP (bar)', 'WHP (barg)']:
            ax1a.set_ylabel('WHP (MPa)', fontsize=16)
        elif parameter == 'DHP (barg)':
            ax1a.set_ylabel('DHP (MPa)', fontsize=16)
        elif parameter == 'Injectivity':
            ax1a.set_ylabel('Injectivity (t/h/bar)', fontsize=16)
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
        ax1a.legend()
    else:
        handles.extend(ax1a.legend_.get_lines())
        # Redo the legend
        if len(handles) > 4:
            ax1a.legend(handles=handles, fontsize=5, loc=2)
        else:
            ax1a.legend(handles=handles, loc=2)
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

def plot_event_well_dist(catalog, well_fzs, flow_start, diffs,
                         diffusion='isotropic', temp_list='all',
                         method='scatter', boxplots=False, starttime=None,
                         ylim=None, endtime=None, title=None, show=True,
                         axes=None):
    """
    Function to plot events with distance from well as a function of time.
    :param cat: catalog of events
    :param well_fzs: text file of xyz feedzone pts
        e.g. NM08 bottom hole: (176.1788 -38.5326 3.3615)
        e.g. NM08 top fz:
    :param flow_start: Start UTCdt of well flow to model
    :param diffs: list of diffusion values to plot
    :param diffusion: Either isotropic or planar flow
    :param temp_list: list of templates for which we'll plot detections
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
                                   (o.depth - 350.) / 1000.),
                                  # Account for hypoDD station elev
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
    start = pd.Timestamp(flow_start.datetime)
    if endtime:
        end = pd.Timestamp(endtime.datetime)
    else:
        end = pd.Timestamp(cat_end)
    t = pd.to_datetime(pd.date_range(start, end, freq='H'))
    t = [mdates.date2num(d) for d in t]
    # Now diffusion y vals
    diff_ys = []
    labs = []
    for d in diffs:
        # Isotropic diffusion
        if 'isotropic' in diffusion:
            # Shapiro and Dinske 2009 (and all other such citations)
            diff_ys.append([np.sqrt(3600 * d * i * 4. * np.pi)
                            for i in range(len(t))])
            labs.append('Isotropic')
        if 'planar' in diffusion:
            diff_ys.append([3600 * d * i / 2 * np.pi for i in range(len(t))])
            labs.append('Planar')
        elif 'cubic' in diffusion:
            # Yeilds volume of affected area. We will assume spherical for simplicity
            diff_ys.append([0.5 * ((3600 * i * 100. / 0.2)**(1/3.))
                            for i in range(len(t))])
            labs.append('Nonlinear diffusion')
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
        positions = [d for d in u_days]
        bplots = ax.boxplot(bins, positions=positions, patch_artist=True,
                            flierprops={'markersize': 0}, manage_xticks=False)
        for patch in bplots['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.5)
    # First diffusions
    for i, diff_y in enumerate(diff_ys):
        if labs[i] == 'Nonlinear diffusion':
            ax.plot(t, diff_y, label='{}'.format(labs[i]))
        else:
            ax.plot(t, diff_y, label='{}: D={} $m^2/s$'.format(labs[i],
                                                               str(diffs[i])))
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
    ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([0, 3])
    if title:
        ax.set_title(title, fontsize=19)
    else:
        ax.set_title('Fluid diffusion envelopes with time')
    if starttime:
        ax.set_xlim([date2num(starttime.datetime), max(t)])
    else:
        ax.set_xlim([min(t), max(t)])
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Distance (m)', fontsize=16)
    # fig.tight_layout()
    if show:
        fig.show()
    return ax

def place_NM08_times(ax):
    """
    Place the time labels as axvline for NM08 stimulation
    :param ax: matplotlib.Axes to plot on
    :return:
    """
    # Get data coords
    yz = ax.get_ylim()
    # Place lines
    ax.axvline(UTCDateTime(2012, 6, 7).datetime, ymax=0.6, linestyle='--',
               color='k', linewidth=0.7)
    ax.axvline(UTCDateTime(2012, 6, 15).datetime, ymax=0.6, linestyle='--',
               color='k', linewidth=0.7)
    ax.axvline(UTCDateTime(2012, 6, 26).datetime, ymax=0.7, linestyle='--',
               color='k', linewidth=0.7)
    ax.axvline(UTCDateTime(2012, 7, 1).datetime, ymax=0.85, linestyle='--',
               color='k', linewidth=0.7)
    ax.axvline(UTCDateTime(2012, 7, 6).datetime, ymax=0.90, linestyle='--',
               color='k', linewidth=0.7)
    # Place text
    ax.text(UTCDateTime(2012, 6, 7).datetime, 0.61 * yz[1], 'T1',
            horizontalalignment='center')
    ax.text(UTCDateTime(2012, 6, 15).datetime, 0.61 * yz[1], 'T2',
            horizontalalignment='center')
    ax.text(UTCDateTime(2012, 6, 26).datetime, 0.71 * yz[1], 'T3',
            horizontalalignment='center')
    ax.text(UTCDateTime(2012, 7, 1).datetime, 0.86 * yz[1], 'T4',
            horizontalalignment='center')
    ax.text(UTCDateTime(2012, 7, 6).datetime, 0.91 * yz[1], 'T5',
            horizontalalignment='center')
    return

