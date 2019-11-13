#!/usr/bin/env python
import matplotlib
import pytz
import scipy.spatial as ss
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
import seaborn as sns

from itertools import cycle
from dateutil import rrule
from scipy.io import loadmat
from scipy.spatial import KDTree
from operator import itemgetter
from datetime import timedelta
from obspy.imaging.beachball import beach
from obspy.geodetics import degrees2kilometers
from obspy import Catalog
from obspy.core.event import Comment
from eqcorrscan.utils.mag_calc import calc_max_curv, calc_b_value, dist_calc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# local files dependent upon paths set in ipython rc
from shelly_mags import local_to_moment, local_to_moment_Majer


def date_generator(start_date, end_date):
    # Generator for date looping
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def avg_dto(dto_list):
    srt_list = sorted(dto_list)
    return srt_list[0] + np.mean([dt - srt_list[0] for dt in srt_list])
######################### Shelly plotting stuffs #############################

def plot_svd_mag_example(temp_wav, det_wav, station, show=False):
    """
    Plot an example data matrix for svd magnitude calc

    :param temp_wav: obspy.core.Stream of detected event
    :param det_wav: obspy.core.Stream of detected event
    :param station: Station to select wavs for
    :return:
    """
    try:
        temp_dat = temp_wav.select(station=station)[0].data
        det_dat = det_wav.select(station=station)[0].data
    except IndexError:
        print('You dont have {} in one of your streams'.format(station))
        return
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(11, 5))
    # Do the mathsss for the U and V plots
    M = np.vstack((temp_dat, det_dat)).T
    U, sig, V = np.linalg.svd(M, full_matrices=True)
    # Normalize the streams for plotting
    # (obviously you don't do this for the actual calculation)
    norm_temp = temp_dat / np.max(temp_dat)
    norm_det = det_dat / np.max(temp_dat)
    # y vector for both
    y_vec = np.arange(norm_det.shape[0])
    axes[0].plot(norm_temp, y_vec, color='black', alpha=0.7)
    axes[0].plot(norm_det + 1.3, y_vec, color='black', alpha=0.7)
    axes[0].invert_yaxis()
    axes[0].axis('off')
    cols = ['black', 'dimgray', 'gray', 'darkgray', 'lightgray']
    for i in range(2):
        axes[1].plot(U[:,i] + (i * 0.3), y_vec, color=cols[i], alpha=0.4)
    axes[1].invert_yaxis()
    axes[1].axis('off')
    if show:
        plt.show()
    return U, sig, V

#########################Magnitude and b-val functions #######################

def plot_cumulative_mo(catalog, method='Ristau', dates=None,
                       color='firebrick', axes=None, tick_colors=False):
    """
    Plot cumulative seismic moment with time from a catalog
    :param catalog: catalog of events to plot for
    :param method: Method of Ml to Mo conversion: Ristau or Majer
    :param dates: Date range to plot
    :param color: color of the curve
    :param axes: Prexisting axes to plot into
    :param tick_colors: Coordinate tick colors with curve?
    :return:
    """
    if dates:
        cat = Catalog(events=[ev for ev in catalog
                              if dates[0] < ev.origins[-1].time < dates[-1]])
    else:
        cat = catalog
    if not axes:
        fig, ax = plt.subplots()
    elif axes.get_ylim()[-1] == 1.0:
        ax = axes
    else:
        ax = axes.twinx()
        try:
            # Grab these lines for legend
            handles, leg_labels = ax.get_legend_handles_labels()
            if isinstance(ax.legend_, matplotlib.legend.Legend):
                ax.legend_.remove()  # Need to manually remove this, apparently
        except AttributeError:
            print('Empty axes. No legend to incorporate.')
        if axes.yaxis.get_ticks_position() == 'right':
            ax.yaxis.set_label_position('left')
            ax.yaxis.set_ticks_position('left')
        elif axes.yaxis.get_ticks_position() == 'left':
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')
    dtos = [ev.preferred_origin().time.datetime for ev in cat]
    mags = [mag.mag for ev in cat for mag in ev.magnitudes
            if mag.magnitude_type in ['ML', 'M']]
    if method == 'Ristau':
        mos = [local_to_moment(m) for m in mags]
    elif method == 'Majer':
        mos = [local_to_moment_Majer(m) for m in mags]
    else:
        print('Must specify either Ristau or Majer method')
    ax.plot(dtos, np.cumsum(mos), color=color, label='Cumulative M$_0$')
    locs = ax.get_yticks()
    # exp = int('{:.2E}'.format(max(locs)).split('+')[-1])
    # ax.set_yticks(locs, ['{:.2E}'.format(l).split('+')[0] for l in locs])
    math_str = r'$\sum{M_0}}$ dyne-cm'
    ax.set_ylabel(math_str, fontsize=16)
    ax.set_ylim(bottom=0)
    if tick_colors:
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis='y', colors=color)
    if axes:
        try:
            ax.legend()
            hands, labs = ax.get_legend_handles_labels()
            # Add the new handles to the prexisting ones
            handles.extend(hands)
            leg_labels.extend(labs)
            # Redo the legend
            if len(handles) > 4:
                ax.legend(handles=handles, labels=leg_labels,
                          fontsize=12, loc=2, numpoints=1)
            else:
                ax.legend(handles=handles, labels=leg_labels, loc=2,
                          numpoints=1, fontsize=12)
        except UnboundLocalError:
            print('Plotting on empty axes. No handles to add to.')
            ax.legend(fontsize=12)
    return ax


def make_sdrs(fm_file, cat_format):
    # Take output from fm codes and parse to dictionary keyed to event id
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


def plot_mags(cats, dates=None, metric='time', ax=None, title=None, show=True,
              fm_file=None, fm_color=None, cat_format='templates',
              focmecs=False):
    """
    Plot earthquake magnitude as a function of time

    :param cat: list of catalogs of earthquakes with mag info
    :param dates: list of UTCDateTime objects defining start and end of plot.
        Defaults to None.
    :param metric: Are we plotting magnitudes with depth, time or
        depth_w_time?
    :param ax: matplotlib.Axes object to plot on top of (optional)
    :param title: Plot title (optional)
    :param show: whether or not to show plot
    :param fm_file: Path to focal mech output from AT code
    :param fm_color: 'by_date', string of single color or list of
        colors (one per catalog)
    :param cat_format: Event id naming convention for catalog (template or
        detection)
    :param focmecs: Plot focmec collection or not?
    :return: matplotlib.pyplot.Figure
    """
    # Lump all catalogs together
    cat = Catalog()
    for c in cats:
        cat += c
    if ax: # If axis passed in, set x-axis limits accordingly
        plain = False
        if ax.get_ylim()[-1] == 1.0 or metric == 'depth_w_time':
            ax1 = ax
        else:
            ax1 = ax.twinx()
            try:
                # Grab these lines for legend
                handles, leg_labels = ax.get_legend_handles_labels()
                if isinstance(ax.legend_, matplotlib.legend.Legend):
                    ax.legend_.remove()  # Need to manually remove this, apparently
            except AttributeError:
                print('Empty axes. No legend to incorporate.')
        if ax.yaxis.get_ticks_position() == 'right':
            ax1.yaxis.set_label_position('left')
            ax1.yaxis.set_ticks_position('left')
        elif ax.yaxis.get_ticks_position() == 'left':
            ax1.yaxis.set_label_position('right')
            ax1.yaxis.set_ticks_position('right')
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
            start = pytz.utc.localize(dates[0].datetime)
            end = pytz.utc.localize(dates[1].datetime)
    else:
        if dates:
            start = pytz.utc.localize(dates[0].datetime)
            end = pytz.utc.localize(dates[1].datetime)
        else:
            cat.events.sort(key=lambda x: x.origins[-1].time)
            start = pytz.utc.localize(cat[0].origins[0].time.datetime)
            end = pytz.utc.localize(cat[-1].origins[0].time.datetime)
    cat.events.sort(key=lambda x: x.origins[-1].time)
    # Make all event times UTC for purposes of dto compare
    mag_tup = []
    fm_tup = []
    # Dictionary of fm strike-dip-rake from Arnold/Townend pkg
    if fm_file:
        sdrs = make_sdrs(fm_file=fm_file, cat_format=cat_format)
    for ev in cat:
        if start < pytz.utc.localize(ev.origins[-1].time.datetime) < end:
            if cat_format == 'detections' and focmecs:
                fm_id = '{}_{}_{}'.format(
                    ev.resource_id.id.split('/')[-1].split('_')[0],
                    ev.resource_id.id.split('_')[-2],
                    ev.resource_id.id.split('_')[-1][:6])
            elif cat_format == 'templates' and focmecs:
                fm_id = ev.resource_id.id.split('/')[-1]
            elif not focmecs:
                fm_id = 'foo'
            try:
                if metric == 'time':
                    mag_tup.append(
                        (pytz.utc.localize(ev.origins[-1].time.datetime),
                         ev.magnitudes[-1].mag))
                    if fm_id in sdrs:
                        fm_tup.append(sdrs[fm_id])
                    else:
                        fm_tup.append(None)
                elif metric == 'depth':
                    if ev.preferred_origin().method_id:
                        if ev.preferred_origin().method_id.id.endswith('GrowClust'):
                            mag_tup.append((ev.preferred_origin().depth + 350.,
                                            ev.magnitudes[-1].mag))
                        else:
                            mag_tup.append((ev.preferred_origin().depth,
                                            ev.magnitudes[-1].mag))
                    else:
                        mag_tup.append((ev.preferred_origin().depth,
                                        ev.magnitudes[-1].mag))
                    if fm_id in sdrs:
                        fm_tup.append(sdrs[fm_id])
                    else:
                        fm_tup.append(None)
                elif metric == 'depth_w_time':
                    # Only plot events with dd locations
                    if ev.preferred_origin().method_id:
                        if ev.preferred_origin().method_id.id.endswith('GrowClust'):
                            mag_tup.append(
                                (pytz.utc.localize(ev.preferred_origin().time.datetime),
                                 ev.preferred_origin().depth))
            except AttributeError:
                print('Event {} has no associated magnitude'.format(
                    str(ev.resource_id)))
    xs, ys = zip(*mag_tup)
    if fm_color != 'by_date':
        if type(fm_color) == list:
            cols = []
            for i, ct in enumerate(cats):
                cols.extend([fm_color[i] for j in range(len(ct))])
            print(cols)
        else:
            cols = [fm_color for x in xs]
    else:
        cols = np.array([(x - xs[0]).days for x in xs])
        cols = cols / np.max(cols)
        cols = [cm.viridis(col) for col in cols]
    if not ax:
        fig, ax1 = plt.subplots()
    ax1.set_ylabel('Magnitude', fontsize=16)
    ax1.set_ylim([0, max(ys) * 1.2])
    # Eliminate the side padding
    ax1.margins(0, 0)
    if metric == 'depth':
        ax1.set_xlabel('Depth (m)', fontsize=16)
        fn = np.poly1d(np.polyfit(xs, ys, 1))
        ax1.plot(xs, ys, 'mo', xs, fn(xs), '--k',
                 markersize=2)
        ax1.set_xlim([0, 5000])
    elif metric == 'time':
        mkline, stlines, bsline = ax1.stem(xs, ys, markerfmt='o',
                                           label='Magnitudes', alpha=0.5)
        plt.setp(stlines, 'color', 'darkgray')
        plt.setp(stlines, 'linewidth', 1.)
        plt.setp(bsline, 'color', 'black')
        plt.setp(bsline, 'linewidth', 1.)
        plt.setp(mkline, 'color', 'darkgray')
        plt.setp(mkline, 'zorder', 3)
        plt.setp(mkline, 'markeredgecolor', 'k')
        plt.setp(mkline, 'markersize', 4.)
        # ax1.stem(xs, ys, 'gray', 'gray', label='Magnitudes')
        ax1.set_xlabel('Date', fontsize=16)
        ax1.set_xlim([start, end])
    elif metric == 'time':
        ax1.set_xlabel('Date', fontsize=16)
        ax1.set_xlim([start, end])
    elif metric == 'depth_w_time':
        ax1.set_ylabel('Depth (m bsl)')
        ax1.scatter(xs, ys, s=9., marker='o', color='green',
                    edgecolor='k', linewidth=0.3, label='Events',
                    alpha=0.3)
        # plot Rot Andesite extents
        ax1.set_xlim([start, end])
        xlims = ax1.get_xlim()
        xz = [xlims[0], xlims[1], xlims[1], xlims[0]]
        yz_tkri = [810., 810., 2118., 2118.]
        yz_and = [2118., 2118., 3000., 3000.]
        # Tahorakuri
        ax1.fill(xz, yz_tkri, color='palegoldenrod', zorder=0, alpha=0.5)
        ax1.text(0.2, 0.6, 'Tahorakuri', fontsize=10, color='k',
                transform=ax1.transAxes, horizontalalignment='center')
        # Andesite
        ax1.fill(xz, yz_and, color='gainsboro', zorder=0, alpha=0.5)
        ax1.text(0.2, 0.3, 'Andesite', fontsize=10, color='k',
                transform=ax1.transAxes, horizontalalignment='center')
        ax1.set_xlabel('Date', fontsize=16)
        ax1.invert_yaxis()
        ax1.set_ylim([4000, -500])
    for x, y, fm, col in zip(xs, ys, fm_tup, cols):
        if metric == 'time' and fm:
            bball = beach(fm, xy=(mdates.date2num(x), y), width=70,
                          linewidth=1, axes=ax1, facecolor=col)
            ax1.add_collection(bball)
        elif metric == 'depth' and fm:
            bball = beach(fm, xy=(x, y), width=70,
                          linewidth=1, axes=ax1, facecolor=col)
            ax1.add_collection(bball)
    if ax:
        try:
            ax1.legend()
            hands, labs = ax1.get_legend_handles_labels()
            # Add the new handles to the prexisting ones
            handles.extend(hands)
            leg_labels.extend(labs)
            # Redo the legend
            if len(handles) > 4:
                ax1.legend(handles=handles, labels=leg_labels,
                           fontsize=12, loc=2, numpoints=1)
            else:
                ax1.legend(handles=handles, labels=leg_labels, loc=2,
                           numpoints=1, fontsize=12)
        except UnboundLocalError:
            print('Plotting on empty axes. No handles to add to.')
            ax1.legend(fontsize=12)
    # ax1.set_xlim([0, 10000])
    if title:
        ax1.set_title(title)
        plt.tight_layout()
    if not ax:
        fig.autofmt_xdate()
    if show:
        plt.show()
        plt.close('all')
    return ax1

def Mc_test(cat, n_bins, start_mag=None):
    """
    Test the reliability of predetermined Mc
    :param cat: Catalog of events
    :param n_bins: Number of bins
    :param test_mag: Pre-calculated mag to test
    :param start_mag: Magnitude to start test from
    :param show: Plotting flag
    :return: (matplotlib.pyplot.Figure, best bval, cutoff mag)
    """
    mags = np.array([round(ev.magnitudes[-1].mag, 1)
                    for ev in cat if len(ev.magnitudes) > 0])
    mags = mags[np.isfinite(mags)].tolist()
    mags.sort()
    bin_vals, bins = np.histogram(mags, bins=n_bins) # Count mags in each bin
    inds = np.digitize(mags, bins) # Get bin index for each mag in mags
    # bin_cents = bins - ((bins[1] - bins[0]) / 2.)
    avg_mags = []
    for i, bin in enumerate(bins):
        avg_mags.append(np.mean([mag for mag, ind in zip(mags, inds)
                                 if ind >= i + 1]))
    bvals = [np.log10(np.exp(1)) / (avg_mag - bins[i])
             for i, avg_mag in enumerate(avg_mags)]
    # Errors for each bin
    errs = [2 * bval / np.sqrt(sum(bin_vals[i:]))
            for i, bval in enumerate(bvals)]
    # Error ranges for bins above start_mag
    err_rangs = [(cent, bval - err, bval + err)
                 for bval, err, cent in zip(bvals, errs, bins)
                 if cent > start_mag]
    # Now to test input mag against "best-fitting" bval within these errors
    bval_hits = [] # Count how many bins each value hits
    for test_bval in np.linspace(0, 2, 40):
        hits = [rang[0] for rang in err_rangs if rang[1] <= test_bval and
                rang[2] >= test_bval]
        # bval_hits is a tup: ((bval, cutoff mag), total number of matches)
        bval_hits.append(((test_bval, min(hits)), len(hits)))
    # Find max bval_hits and corresponding cuttoff mag
    best_bval_cut = max(bval_hits, key=itemgetter(1))[0]
    # Now plotting from premade fig from bval_plot
    return {'best_bval':best_bval_cut[0], 'M_cut': best_bval_cut[1],
            'bins': bins, 'bvals': bvals, 'errs': errs}


def bval_calc(cat, bin_size, MC, weight=False):
    """
    Helper function to run the calculation loop
    :param mags: list of magnitudes
    :param bins: size of bins in M
    :param method: whether to use weighted lsqr regression or MLE
    :return: (non_cum_bins, cum_bins, bval_vals, bval_bins, bval_wts)
    """
    mags = np.asarray([ev.magnitudes[-1].mag for ev in cat
                       if len(ev.magnitudes) > 0])
    mags = mags[np.isfinite(mags)].tolist()
    # Calculate Mc using max curvature method if not specified
    if not MC:
        Mc = calc_max_curv(mags)
    else:
        Mc = MC
    # Establish bin limits and spacing
    # bin_vals = np.arange(min(mags), max(mags), bin_size)
    bin_vals = np.arange(-1., 4., bin_size)
    non_cum_bins = []
    cum_bins = []
    bval_vals = []
    bval_bins = []
    bval_wts = []
    for i, val in enumerate(bin_vals):
        cum_val_count = len([ev for ev in cat if len(ev.magnitudes) > 0
                             and ev.magnitudes[-1].mag >= val])
        if i < len(bin_vals) - 1:
            non_cum_val_cnt = len([ev for ev in cat
                                   if len(ev.magnitudes) > 0
                                   and val < ev.magnitudes[-1].mag
                                   and bin_vals[i + 1] >=
                                   ev.magnitudes[-1].mag])
            non_cum_bins.append(non_cum_val_cnt)
        cum_bins.append(cum_val_count)
        if val >= Mc:
            bval_vals.append(cum_val_count)
            bval_bins.append(val)
            bval_wts.append(non_cum_val_cnt / float(len(mags)))
    if len(bval_bins) == 0:
        print('No bins above Mc. Ignore this catalog')
        return None
    # Tack 0 on end of non_cum_bins representing bin above max mag
    non_cum_bins.append(0)
    if weight:
        b, a = np.polyfit(bval_bins, np.log10(bval_vals), 1, w=bval_wts)
        b *= -1.
    else:
        b, a = np.polyfit(bval_bins, np.log10(bval_vals), 1)
        b *= -1.
    return {'bin_vals':bin_vals, 'non_cum_bins':non_cum_bins,
            'cum_bins':cum_bins, 'bval_vals':bval_vals,
            'bval_bins':bval_bins, 'bval_wts':bval_wts,
            'b': b, 'a': a, 'Mc': Mc}

def simple_bval_plot(catalogs, cat_names, bin_size=0.1, MC=None,
                     histograms=False, title=None, weight=False,
                     show=True, savefig=None, axes=None, colors=None,
                     linestyles=None, plot_text=False, xlim=None, ylim=None,
                     insets=False, reference=True, bplotvar=False):
    """
    Function to plot cumulative distributions of mag for an arbitrary
    number of catalogs on the same Axes

    :param catalogs: list of obspy.core.Catalog objects
    :param cat_names: Names of catalogs in the order of catalogs
    :param bin_size: If ploting
    :param MC: Manual magnitude of completeness if desired
    :param title: Title of plot
    :param show: Do we show the plot?
    :param savefig: None or name of saved file
    :param axes: Axes object to plot to (optional)
    :param colors: iterable of desired color strings
    :param linestyles: itertools.Cycle of desired linestyles
    :param plot_text: Boolean for placing text on plot
    :param xlim: Custom x limits
    :param ylim: Custom y limits
    :param insets: Plot inset plots of b value and std_dev
    :param reference: Plot a reference line of b = 1
    :param bplotvar: Plot flag for EQcorrscan mag_calc.calc_b_value

    :return:
    """
    if colors == False:
        colors = cycle(sns.color_palette('muted').as_hex())
    else:
        colors = cycle(colors)
    if not axes:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = axes
    # Array of b-values and std_errs for many catalogs for further plotting
    bs = []
    std_errs = []
    for i, (cat, name) in enumerate(zip(catalogs, cat_names)):
        mags = [ev.preferred_magnitude().mag for ev in cat]
        b_dict = bval_calc(cat, bin_size, MC, weight=weight)
        if not b_dict:
            print('b_dict went wrong. Next catalog.')
            continue
        bcalc = calc_b_value(
            magnitudes=mags,
            completeness=np.arange(min(mags), max(mags), 0.1),
            plotvar=bplotvar)
        bcalc.sort(key=lambda x: x[2])
        # b = bcalc[-1][1]
        Mc = bcalc[-1][0]
        comp_mags = [m for m in mags if m > Mc]
        mean_mag = np.mean(comp_mags)
        min_mag = min(comp_mags)
        neq = len(comp_mags)
        # Max likelihood (Aki 1965) with
        # Shi&Bolt 1982 Formulation for b std error
        b = (1 / (mean_mag - min_mag)) * np.log10(np.exp(1))
        std_dev = np.sum([(m - mean_mag) ** 2
                          for m in comp_mags]) / (neq * (neq - 1))
        std_err = 2.30 * np.sqrt(std_dev) * (b ** 2)
        col = next(colors)
        if not linestyles:
            lin = '-'
        else:
            lin = next(linestyles)
        if histograms:
            sns.distplot(mags, kde=False, color=col,
                         hist_kws={'alpha': 1.0},
                         ax=ax)
        # Reversed cumulative hist
        ax.plot(b_dict['bin_vals'], b_dict['cum_bins'], label=name,
                color=col, linestyle=lin, linewidth=1.5)
        if i == 0:
            y = 0.9
        elif i == 1:
            y = 0.8
        elif i == 2:
            y = 0.7
        bs.append(b)
        std_errs.append(std_err)
        text = 'B-value: {:.2f}$\pm${:.2f}'.format(b, std_err)
        if plot_text:
            ax.text(0.80, y - 0.05, text, transform=ax.transAxes,
                    color=col, horizontalalignment='center',
                    fontsize=14.)
            ax.text(0.80, y, 'Mc=%.2f' % Mc,
                    color=col, transform=ax.transAxes,
                    horizontalalignment='center', fontsize=14.)
    if insets:
        inset1 = inset_axes(ax, width='98%', height='40%',
                            bbox_to_anchor=(.6, .5, .4, .5),
                            bbox_transform=ax.transAxes, loc=1)
        inset2 = inset_axes(ax, width='98%', height='40%',
                            bbox_to_anchor=(.6, .5, .4, .5),
                            bbox_transform=ax.transAxes, loc=4)
        sns.distplot(bs, ax=inset1, kde=False, bins=8)
        sns.distplot(std_errs, ax=inset2, kde=False, bins=8)
        inset1.set_xlabel('$b$-value', fontsize=6)
        plt.setp(inset1.get_xticklabels(), fontsize=6)
        plt.setp(inset1.get_yticklabels(), fontsize=6)
        inset2.set_xlabel('Error', fontsize=6)
        plt.setp(inset2.get_xticklabels(), fontsize=6)
        plt.setp(inset2.get_yticklabels(), fontsize=6)
    if reference:
        a_ref = 4000
        mags = np.linspace(0.5, 3., 50)
        nums = 10**(np.log10(a_ref) - mags)
        ax.plot(mags, nums, label='$b$=1', color='black', linewidth=0.8,
                linestyle='--')
        ax.text(mags[-1] + 0.1, nums[-1], '$b=1$', fontsize=10, rotation=-36,
                horizontalalignment='center')
    ax.set_yscale('log')
    ax.tick_params(labelsize=14.)
    ax.set_ylabel('Number of events', fontsize=14.)
    ax.set_xlabel('Magnitude', fontsize=14.)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title, fontsize=18)
    else:
        ax.set_title('B-value plot', fontsize=18)
    if len(catalogs) < 6:
        leg = ax.legend(loc=3)
        leg.get_frame().set_alpha(0.9)
    if show:
        plt.show()
    if savefig:
        fig.savefig(savefig, dpi=300)
    return ax, bs, std_errs


def add_subplot_axes(ax, width, height, loc):
    """
    Simple wrapper on inset_axes
    """
    sub_ax = inset_axes(ax, width=width, height=height, loc=2)
    return sub_ax


def bval_null_prob(N1, N2, b1, b2):
    """
    Calculate probability that two catalogs come from same population per
    Utsu 1992 and Wiemer et al 1998:

    https://academic.oup.com/gji/article-abstract/134/2/409/662959

    :param cat1:
    :param cat2:
    :return:
    """
    N = N1 + N2
    dA = ((-2. * N * np.log(N)) +
          (2 * N1 * np.log(N1 + (N2 * b1 / b2))) +
          (2 * N2 * np.log((N1 * b2 / b1) + N2)) - 2)
    P0 = np.exp((-1 * dA / 2.) - 2)
    return P0


def map_bvalue(catalog, max_ev, no_above_Mc, Man_Mc=None, show=False, outfile=None,
               dimension=3, plotvar=False):
    """
    Do b-value mapping using a catalog, as described in Bachmann et al. 2012:

    doi:10.1029/2012GL051480

    :param catalog: Catalog of events for which to map b-value
    :param max_ev: Number of nearest events to use in calculation
    :param no_above_Mc: Required number of events above Mc for b calculation
    :param Man_Mc: If desired, can provide a static Mc for all clusters
    :param show: Plot flag for locations colored by b and Mc
    :param outfile: Path to output file for plotting with GMT
    :param dimension: 2 (xy) or 3 (xyz) dimensional inter-event distance calcs
    :param plotvar: Plot flag for the b-value calculation function
    :return:
    """
    # Sort catalog
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    # Make array of points, with units in meters
    if dimension == 3:
        pts = np.array([[
            degrees2kilometers(ev.preferred_origin().longitude) * 1000.,
            degrees2kilometers(ev.preferred_origin().latitude) * 1000.,
            ev.preferred_origin().depth]
                        for ev in catalog])
    elif dimension == 2:
        pts = np.array([[
            degrees2kilometers(ev.preferred_origin().longitude) * 1000.,
            degrees2kilometers(ev.preferred_origin().latitude) * 1000.]
                        for ev in catalog])
    # Make KDTree to query
    treebeard = KDTree(pts)
    bvals = []
    Mcs = []
    errs = []
    volumes = []
    # Make catalog of nearest points for each event
    for pt in pts:
        # print('Working on pt: {}'.format(pt))
        dists, ney_burs = treebeard.query(pt, k=max_ev)
        sub_cat = Catalog(events=[catalog[i] for i in ney_burs])
        lats = [degrees2kilometers(ev.preferred_origin().latitude) * 1000
                for ev in sub_cat]
        lons = [degrees2kilometers(ev.preferred_origin().longitude) * 1000
                for ev in sub_cat]
        zs = [ev.preferred_origin().depth for ev in sub_cat]
        mags = [ev.preferred_magnitude().mag for ev in sub_cat]
        # Do bval calculation
        # If Man_Mc, enforce b-value calculation for only one Mc value
        if not Man_Mc:
            completenesses = np.arange(min(mags), max(mags), 0.1)
        else:
            completenesses = np.array([Man_Mc])
        bcalc = calc_b_value(
            magnitudes=mags,
            completeness=completenesses,
            plotvar=plotvar)
        bcalc.sort(key=lambda x: x[2])
        # b = bcalc[-1][1]
        Mc = bcalc[-1][0]
        comp_mags = [m for m in mags if m > Mc]
        mean_mag = np.mean(comp_mags)
        min_mag = min(comp_mags)
        neq = len(comp_mags)
        # Max likelihood (Aki 1965) with
        # Shi&Bolt 1982 Formulation for b std error
        b = (1 / (mean_mag - min_mag)) * np.log10(np.exp(1))
        std_dev = np.sum([(m - mean_mag) ** 2
                          for m in comp_mags]) / (neq * (neq - 1))
        std_err = 2.30 * np.sqrt(std_dev) * (b ** 2)
        # If enough events, save b val
        if len([ev for ev in sub_cat
                if ev.preferred_magnitude().mag > Mc]) > no_above_Mc:
            bvals.append(b)
            Mcs.append(Mc)
            errs.append(std_err)
            hull = ss.ConvexHull(np.stack((lons, lats, zs)).T)
            volumes.append(hull.volume)
        else:
            # Otherwise dont save
            bvals.append(None)
            Mcs.append(None)
            errs.append(None)
            volumes.append(None)
    # Make output array of lon, lat, depth, mag, b, Mc
    print(len(catalog), len(bvals), len(Mcs)) # Check consistent lengths
    bval_out = []
    for i, ev in enumerate(catalog):
        # Add bvalue Comment to origin (in place)
        ev.preferred_origin().comments.append(
            Comment(text='b={}'.format(bvals[i])))
        ev.preferred_origin().comments.append(
            Comment(text='Mc={}'.format(Mcs[i])))
        ev.preferred_origin().comments.append(
            Comment(text='b error={}'.format(errs[i])))
        ev.preferred_origin().comments.append(
            Comment(text='b volume={}'.format(volumes[i])))
        bval_out.append([ev.preferred_origin().longitude,
                         ev.preferred_origin().latitude,
                         ev.preferred_origin().depth,
                         ev.preferred_magnitude().mag,
                         bvals[i], Mcs[i], errs[i], volumes[i]])
    if show:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        x, y, z, m, b, mc, err, vol = zip(*bval_out)
        scat_b = axes[0, 0].scatter(x, y, s=m, c=b)
        scat_mc = axes[0, 1].scatter(x, y, s=m, c=mc)
        scat_err = axes[1, 0].scatter(x, y, s=m, c=err)
        scat_vol = axes[1, 1].scatter(x, y, s=m, c=vol)
        fig.colorbar(scat_b, ax=axes[0, 0])
        fig.colorbar(scat_mc, ax=axes[0, 1])
        fig.colorbar(scat_err, ax=axes[1, 0])
        fig.colorbar(scat_vol, ax=axes[1, 1])
        axes[0, 0].set_title('$b$-value')
        axes[0, 1].set_title('M$_c$')
        axes[1, 0].set_title('Std error')
        axes[1, 1].set_title('Volume (m$^3$')
        plt.show()
        plt.close()
    if outfile:
        with open(outfile, 'w') as outf:
            for ln in bval_out:
                if ln[4] == None:
                    continue
                outf.write('{} {} {} {} {} {} {} {}\n'.format(ln[0], ln[1],
                                                              ln[2], ln[3],
                                                              ln[4], ln[5],
                                                              ln[6], ln[7]))
    return bval_out


def r_b_plot(catalog, injection_point, dimension=3, axes=None, show=False,
             xlim=[0, 1000], ylim=[0, 2.5], title=None, color=None, label=None):
    """
    Plot b-values with distance from an injection point
    :param catalog: Catalog of events with a Comment in the preferred_origin()
        with the bval in it.
    :param injection_point: Tuple of (lat, lon, depth) for the desired
        injection point
    :param dimension: Calculate distance in map distance or include depth (3)?
    :param axes: matlotlib Axes instance to plot into
    :param show: Show the plot?
    :param outfile: Save plot to a file?
    :param title: String for the plot title
    :param color: Curve color
    :param label: Label the curve with this string

    :return:

    .. note: RK24 Feedzones: (-38.6149, 176.2025, 2.9)
             RK23 Feedzones: (-38.6162, 176.2076, 2.9)
             NM06 feedzones: (-38.5653, 176.1948, 2.88)
             NM09 feedzones: (-38.5358, 176.1857, 2.45)
    """
    if not axes:
        fig, axes = plt.subplots(figsize=(7, 5))
    if dimension == 3:
        pts = [(dist_calc(injection_point,
                          (ev.preferred_origin().latitude,
                           ev.preferred_origin().longitude,
                           ev.preferred_origin().depth / 1000.)) * 1000.,
                ev.preferred_origin().comments[-1].text.split('=')[-1])
               for ev in catalog
               if ev.preferred_origin().comments[-1].text.split('=')[-1]
               != 'None']
    elif dimension == 2:
        pts = [(dist_calc((injection_point[0], injection_point[1], 0),
                          (ev.preferred_origin().latitude,
                           ev.preferred_origin().longitude, 0)) * 1000.,
                ev.preferred_origin().comments[-1].text.split('=')[-1])
               for ev in catalog
               if ev.preferred_origin().comments[-1].text.split('=')[-1]
               != 'None']
    elif dimension == 'depth':
        pts = [(ev.preferred_origin().depth,
                ev.preferred_origin().comments[-1].text.split('=')[-1])
               for ev in catalog
               if ev.preferred_origin().comments[-1].text.split('=')[-1]
               != 'None']
    # Sort by distance for putting into bins
    pts.sort(key=lambda x: x[0])
    x, y = zip(*pts)
    x = np.array(x)
    y_flts = np.array([float(s) for s in y])
    # Scatter plot
    axes.scatter(x, y_flts, marker='.', color='k', s=8.0, alpha=0.2)
    # Calculate averages/variance
    bins = np.arange(min(x), max(x), 50) #50 m bins
    digitized = np.digitize(x, bins)
    bin_means = [y_flts[digitized == i].mean() for i in range(1, len(bins))]
    bin_std = [y_flts[digitized == i].std() for i in range(1, len(bins))]
    if not color:
        col = 'r'
    else:
        col = color
    if not label:
        lab = 'Average'
    else:
        lab = label
    axes.errorbar(bins[:-1] + ((bins[1] - bins[0]) / 2.), bin_means,
                  color=col, linewidth=2.5, yerr=bin_std, ecolor='black',
                  elinewidth=1.5, capsize=2.5, marker='s',
                  markeredgecolor='black', label=lab)
    # Formatting
    if dimension != 'depth':
        axes.set_xlabel('Distance (m)', fontsize=16)
    else:
        axes.set_xlabel('Depth (m)', fontsize=16)
    axes.set_ylabel('$b$-value', fontsize=16)
    if title == None:
        axes.set_title('$b$-value with distance', fontsize=18)
    else:
        axes.set_title(title, fontsize=18)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.legend(fontsize=14)
    plt.setp(axes.get_xticklabels(), fontsize=14)
    plt.setp(axes.get_yticklabels(), fontsize=14)
    if show:
        plt.show()
        plt.close()
    return axes


def t_b_plot(catalog, window, overlap, dates=None, plotvar=False, color=None,
             label=None, axes=None, show=False):
    """
    Plot bvalue with time for a catalog

    :param catalog: Catalog of events
    :param window: Number of events to compute bval for
    :param overlap: How much overlap
    :param dates:
    :return:
    """
    # Ensure sorted by time
    if axes:
        ax = axes
    else:
        fig, ax = plt.subplots()
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    if dates:
        cat = Catalog(events=[ev for ev in catalog
                              if dates[0] < ev.origins[-1].time
                              < dates[1]])
    else:
        cat = catalog
    med_dates = []
    b_values = []
    mcs = []
    residuals = []
    for i in range(0, len(cat), window-overlap):
        sub_cat = cat[i:i+window]
        if len(sub_cat) != window:
            print('Sub cat not length of window. Skipping')
            continue
        med_ind = int(len(sub_cat) / 2)
        med_dates.append(sub_cat[med_ind].preferred_origin().time.datetime)
        # Do bval calculation
        mags = [ev.preferred_magnitude().mag for ev in sub_cat]
        bcalc = calc_b_value(
            magnitudes=mags,
            completeness=np.arange(min(mags), max(mags), 0.1),
            plotvar=plotvar)
        bcalc.sort(key=lambda x: x[2])
        b_values.append(bcalc[-1][1])
        mcs.append(bcalc[-1][0])
        print(bcalc[-1][2])
        residuals.append((100 - bcalc[-1][2]) * 0.1)
    # Plotting
    if color:
        col = color
    else:
        col = 'red'
    if label:
        lab = label
    else:
        lab = 'b-value'
    ax.errorbar(med_dates, b_values,
                color=col, linewidth=2.5, yerr=residuals, ecolor='black',
                elinewidth=1.5, capsize=2.5, marker='s',
                markeredgecolor='black', label=lab)
    fig.autofmt_xdate()
    if show:
        plt.show()
    return axes


def big_bval_plot(cat, bins=30, MC=None, title=None,
                  show=True, savefig=None):
    """
    Plotting the frequency-magnitude distribution on semilog axes
    :param cat: Catalog of events with magnitudes
    :param show: Plot flag
    :return: matplotlib.pyplot.Figure
    """
    ax1, test_dict, b_dict = simple_bval_plot(cat, bins, MC, title, show=False)
    ax2 = fig.add_subplot(122)
    ax2.set_ylim([0, 3])
    ax2.errorbar(test_dict['bins'], test_dict['bvals'],
                 yerr=test_dict['errs'], fmt='-o', color='k')
    ax2.axhline(test_dict['best_bval'], linestyle='--', color='b')
    ax2.axhline(b_dict['b'], linestyle='--', color='r')
    ax2.axvline(b_dict['Mc'], linestyle='--', color='r')
    ax2.axvline(test_dict['M_cut'], linestyle='--', color='b')
    # ax2.text(0.5, 0.9, 'B-value (wt-lsqr): %.3f' % b_dict['b'], color='r',
    #          transform=ax2.transAxes, horizontalalignment='center',
    #          fontsize=10.)
    # ax2.text(0.5, 0.95, 'Mc via max-curv: %.2f' % b_dict['Mc'], color='r',
    #          transform=ax2.transAxes, horizontalalignment='center',
    #          fontsize=10.)
    # ax2.text(0.5, 0.85, 'Modified Mc: %.2f' % test_dict['M_cut'],
    #          transform=ax2.transAxes, horizontalalignment='center',
    #          fontsize=10., color='b')
    # ax2.text(0.5, 0.8, 'Best MLE b-value: %.3f' % test_dict['best_bval'],
    #          color='b', transform=ax2.transAxes,
    #          horizontalalignment='center', fontsize=10.)
    ax2.set_title('MLE b-values v. Mc')
    ax2.set_xlabel('Mc')
    ax2.set_ylabel('B-value')
    # Plot magnitude histogram underneath ax2
    ax3 = ax2.twinx()
    mags = np.asarray([ev.magnitudes[-1].mag for ev in cat
                       if len(ev.magnitudes) > 0])
    mags = mags[np.isfinite(mags)].tolist()
    sns.distplot(mags, kde=False, ax=ax3, hist_kws={"alpha": 0.2})
    ax3.set_ylabel('Number of events')
    fig.tight_layout()
    if show:
        fig.show()
    if savefig:
        fig.savefig(savefig, dpi=300)
    return

def plot_mag_v_lat(cat, method='all'):
    """
    Plotting magnitude vs latitude of events in fields
    :param cat: obspy Catalog
    :param method: 'all' or 'avg'
    :return: matplotlib.pyplot.Figure
    """
    import numpy as np
    data = [(ev.origins[-1].latitude, ev.magnitudes[-1].mag)
            for ev in cat if ev.magnitude[-1]]
    lats, mags = zip(*data)
    fig, ax = plt.subplots()
    if method == 'all':
        ax.scatter(lats, mags)
        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Latitude (deg)')
        ax.set_xlim(min(lats), max(lats))
        ax.set_title('Magnitude vs Latitude')
    elif method == 'avg':
        avgs = []
        bins = np.linspace(min(lats), max(lats), 100)
        for i, bin in enumerate(bins):
            if i < len(bins) - 1:
                avgs.append(np.mean([tup[1] for tup in data
                                     if tup[0] <= bins[i + 1]
                                     and tup[0] > bin]))
        ax.plot(bins[:-1], avgs, marker='o')
        ax.set_ylabel('Magnitude')
        ax.set_xlabel('Latitude (deg)')
        ax.set_xlim(min(lats), max(lats))
        ax.set_title('Magnitude vs Latitude')
    fig.show()

############## vv Plotting of ZMAP .mat output vv ###################

def convert_frac_year(number):
    from datetime import timedelta, datetime
    year = int(number)
    d = timedelta(days=(number - year) * 365)
    day_one = datetime(year,1,1)
    date = d + day_one
    return date


def plot_zmap_b_w_time(mat_file, ax_in=None, color='b',
                       overwrite=False, show=True):
    """
    Take a .mat workspace created by ZMAP and plot it in python
    :param mat_file: Path to .mat file
    :return: matplotlib.pyplot.Figure
    """

    """
    Wkspace must contain variables 'mResult', 'mB' and 'mBstd1' which are
    created when ZMAP is asked to plot b-value with time
    """

    wkspace = loadmat(mat_file)
    # First, deal with time decimals
    time_decs = wkspace['mResult'][:,[0]]
    dtos = [convert_frac_year(dec[0]) for dec in time_decs]
    bvals = wkspace['mB']
    stds = wkspace['mBstd1']
    sigma_top = [bval + std for bval, std in zip(bvals, stds)]
    sigma_bottom = [bval - std for bval, std in zip(bvals, stds)]
    if ax_in and not overwrite:
        ax = ax_in.twinx()
    elif ax_in and overwrite:
        ax = ax_in
    else:
        fig, ax = plt.subplots()
    ax.plot(dtos, sigma_top, linestyle='--', color='0.60', linewidth=1)
    ax.plot(dtos, sigma_bottom, linestyle='--', color='0.60', linewidth=1)
    ax.plot(dtos, bvals, color=color, linewidth=2)
    for t in ax.get_yticklabels():
        t.set_color(color)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Date')
    ax.set_ylabel('b-value')
    ax.set_ylim([0.5, 2.])
    if show:
        fig.show()
    if ax_in:
        return [ax, ax_in]
    else:
        return ax


def plot_max_mag(mat_file, bins='monthly', ax_in=None, color='k',
                 overwrite=False, show=True):
    """
    Scatter plot of the maximum magnitude
    :param cat:
    :return:
    """
    from dateutil import rrule

    wkspace = loadmat(mat_file)
    time_decs = wkspace['s'] # Decimal years
    mags = list(wkspace['newt2'][:,5]) # 6th col holds magnitudes
    dtos = [convert_frac_year(dec[0]) for dec in time_decs]
    start = min(dtos).replace(day=1, hour=0, minute=0, second=0)
    end = max(dtos).replace(day=1, hour=0, minute=0, second=0)
    month_maxs = []
    firsts = list(rrule.rrule(rrule.MONTHLY, dtstart=start,
                              until=end, bymonthday=1))
    # Loop over first day of each month
    for i, dt in enumerate(firsts):
        if i < len(firsts) - 2:
            mags_tmp = [mag for mag, dto in zip(mags, dtos)
                        if dto >= dt and dto < firsts[i + 1]]
            if len(mags_tmp) > 0:
                month_maxs.append(max(mags_tmp))
            else:
                month_maxs.append(np.nan)
    mids = [fst + ((firsts[i + 1] - fst) / 2) for i, fst in enumerate(firsts)
            if i < len(firsts) - 2]
    if ax_in and not overwrite:
        ax = ax_in.twinx()
    elif ax_in and overwrite:
        ax = ax_in
    else:
        fig, ax = plt.subplots()
    ax.scatter(mids, month_maxs, color=color, label='Max magnitude')
    for t in ax.get_yticklabels():
        t.set_color(color)
    ax.set_ylabel('Maximum Mag')
    ax.xaxis_date()
    leg = ax.legend()
    leg.get_frame().set_alpha(0.5)
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylim([0, 4])
    if show:
        plt.show()
    if ax_in:
        return [ax, ax_in]
    else:
        return ax


def plot_mag_bins(mat_file, bin_size='monthly', ax_in=None, color='k',
                  overwrite=False, show=True):
    """
    Scatter plot of the maximum magnitude
    :param cat:
    :return:
    """

    # Read vars from matlab file
    wkspace = loadmat(mat_file)
    time_decs = wkspace['s'] # Decimal years
    mags = list(wkspace['newt2'][:,5]) # 6th col holds magnitudes
    dtos = [convert_frac_year(dec[0]) for dec in time_decs]
    start = min(dtos).replace(day=1, hour=0, minute=0, second=0)
    end = max(dtos).replace(day=1, hour=0, minute=0, second=0)
    month_no = []
    if bin_size == 'monthly':
        firsts = list(rrule.rrule(rrule.MONTHLY, dtstart=start,
                                  until=end, bymonthday=1))
        width = 15.
    elif bin_size == 'weekly':
        firsts = list(rrule.rrule(rrule.WEEKLY, dtstart=start,
                                  until=end))
        width=5.
    elif bin_size == 'daily':
        firsts = list(rrule.rrule(rrule.DAILY, dtstart=start,
                                  until=end))
    # Loop over first day of each month
    for i, dt in enumerate(firsts):
        if i < len(firsts) - 2:
            mags_tmp = [mag for mag, dto in zip(mags, dtos)
                        if dto >= dt and dto < firsts[i + 1]]
            month_no.append(len(mags_tmp))
    mids = [fst + ((firsts[i + 1] - fst) / 2) for i, fst in enumerate(firsts)
            if i < len(firsts) - 2]
    if ax_in and not overwrite:
        ax = ax_in.twinx()
    elif ax_in and overwrite:
        ax = ax_in
    else:
        fig, ax = plt.subplots()
    ax.bar(mids, month_no, width=width, color=color, linewidth=0,
           alpha=0.4, label='Number of events')
    for t in ax.get_yticklabels():
        t.set_color(color)
    ax.set_ylabel('# events')
    ax.xaxis_date()
    ax.set_xlabel('Date')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if show:
        plt.show()
    if ax_in:
        return [ax, ax_in]
    else:
        return ax


def plot_cum_moment(mat_file, ax_in=None, color='m', overwrite=False,
                    show=True):
    """
    Plotting cumulative moment from zmap .mat file
    :param cat: catalog of events
    :param method: path to file
    :return:
    """

    wkspace = loadmat(mat_file)
    time_decs = wkspace['s']
    dtos = [convert_frac_year(dec[0]) for dec in time_decs]
    cum_mo = wkspace['c']
    if ax_in and not overwrite:
        ax = ax_in.twinx()
    elif ax_in and overwrite:
        ax = ax_in
    else:
        fig, ax = plt.subplots()
    ax.plot(dtos, cum_mo, color=color)
    for t in ax.get_yticklabels():
        t.set_color(color)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Date')
    # yticks
    # First get the maximum exponent of scientific not
    locs, labels = plt.yticks()
    exp = int(str(max(locs)).split('+')[-1])
    plt.yticks(locs, map(lambda x: "%.2f" % x, locs / (10 ** exp)))
    math_str = r'$\sum M_0 x 10^{%d}$' % exp
    ax.set_ylabel(math_str)
    if show:
        plt.show()
    if ax_in:
        return [ax, ax_in]
    else:
        return ax


def plot_Mc(mat_file, ax_in=None, color='r', overwrite=False, show=True):
    """
    Plotting Mc with time from zmap .mat file
    :param mat_file: path to file
    :return:
    """

    wkspace = loadmat(mat_file)
    # First, deal with time decimals
    time_decs = wkspace['mResult'][:,[0]]
    dtos = [convert_frac_year(dec[0]) for dec in time_decs]
    Mcs = wkspace['mMc']
    stds = wkspace['mMcstd1']
    sigma_top = [bval + std for bval, std in zip(Mcs, stds)]
    sigma_bottom = [bval - std for bval, std in zip(Mcs, stds)]
    if ax_in and not overwrite:
        ax = ax_in.twinx()
    elif ax_in and overwrite:
        ax = ax_in
    else:
        fig, ax = plt.subplots()
    ax.plot(dtos, sigma_top, linestyle='--', color='0.60', linewidth=1)
    ax.plot(dtos, sigma_bottom, linestyle='--', color='0.60', linewidth=1)
    ax.plot(dtos, Mcs, color=color, linewidth=2)
    for t in ax.get_yticklabels():
        t.set_color(color)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Date')
    ax.set_ylabel(r'$M_c$')
    ax.set_ylim([0., 1.5])
    if show:
        plt.show()
    if ax_in:
        return [ax, ax_in]
    else:
        return ax


def make_big_plot(matfile, flow_dict, pres_dict, well_list='all',
                  method='flows', bar_bin='monthly',
                  show=False, savefig=False):
    """
    Combining all the above plotting functions into something resembling
    Martinez-Garzon Fig 6
    :param matfile:
    :param cat:
    :return:
    """
    from plot_detections import plot_flow_rates

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9.,13.), dpi=400)
    # Top subplot is flow rate and ev no per month
    plot_mag_bins(matfile, bin_size=bar_bin, ax_in=axes[0], overwrite=True, show=False)
    plot_flow_rates(flow_dict, pres_dict, '1/5/2012', '18/11/2015',
                    method=method, well_list=well_list,
                    ax_in=axes[0])
    # Next is max mag and cumulative moment
    plot_max_mag(matfile, ax_in=axes[1], overwrite=True, show=False)
    plot_cum_moment(matfile, ax_in=axes[1], show=False)
    # Last axis
    plot_zmap_b_w_time(matfile, ax_in=axes[2], overwrite=True, show=False)
    plot_Mc(matfile, ax_in=axes[2], show=False)
    if show:
        fig.show()
    if savefig:
        fig.tight_layout()
        fig.savefig(savefig)
    return fig


def plot_2D_bval_grid(matfile, savefig=None, show=True):
    """
    Take ZMAP generated bvalue workspace and plot 2-D geographic grid in python
    :param matfile: path to matlab .mat workspace
    :return:
    """
    from mpl_toolkits.basemap import Basemap
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon

    # Establish figure and axes
    fig, ax = plt.subplots()

    workspace = loadmat(matfile)
    bval_grid = workspace['mBvalue']
    bval_mask = np.ma.masked_invalid(bval_grid)
    lats = workspace['yvect']
    lons = workspace['xvect']
    x, y = np.meshgrid(lons, lats)
    ev_lons = workspace['a'][:,0]
    ev_lats = workspace['a'][:,1]
    mp = Basemap(projection='cyl',
                 urcrnrlon=np.max(lons),
                 urcrnrlat=np.max(lats),
                 llcrnrlon=np.min(lons),
                 llcrnrlat=np.min(lats),
                 resolution='h',
                 epsg=4326)
    mp.drawmapboundary(fill_color='white', zorder=-1)
    mp.fillcontinents(color='beige', lake_color='white', zorder=0)
    mp.drawcoastlines(color='0.6', linewidth=0.5)
    mp.drawcountries(color='0.6', linewidth=0.5)
    mp.pcolormesh(x, y, bval_mask, vmin=0.8,vmax=1.5, cmap='cool',
                  edgecolor='0.6', linewidth=0, latlon=True)
    # Fixup grid spacing params
    meridians = np.arange((np.min(lons) + (np.max(lons) - np.min(lons)) * 0.1),
                          np.max(lons), 0.1)
    parallels = np.arange((np.min(lats) + (np.max(lats) - np.min(lats)) * 0.1),
                          np.max(lats), 0.1)
    mp.drawmeridians(meridians,
                     fmt='%.2f',
                     labelstyle='+/-',
                     labels=[1,0,0,1],
                     linewidth=0.,
                     xoffset=2)
    mp.drawparallels(parallels,
                     fmt='%.2f',
                     labelstyle='+/-',
                     labels=[1,0,0,1],
                     linewidth=0.,
                     yoffset=2)
    cbar = mp.colorbar()
    cbar.solids.set_edgecolor("face")
    cbar.set_label('b-value', rotation=270, labelpad=15)
    cbar.set_ticks(np.arange(0.8, 1.5, 0.1))
    plt.title("B-value map")
    # Plot epicenters
    mp.plot(ev_lons, ev_lats, '.', color='k', markersize=1)
    # Shapefiles!
    # Wells
    mp.readshapefile('/home/chet/gmt/data/NZ/RK_tracks_injection', name='rki',
                     color='blue', linewidth=1.)
    mp.readshapefile('/home/chet/gmt/data/NZ/NM_tracks_injection', name='nmi',
                     color='blue', linewidth=1.)
    mp.readshapefile('/home/chet/gmt/data/NZ/NZ_faults_clipped', name='faults',
                     color='k', linewidth=0.75)
    mp.readshapefile('/home/chet/gmt/data/NZ/NM_tracks_production_wgs84',
                     name='nmp', color='firebrick', linewidth=1.)
    mp.readshapefile('/home/chet/gmt/data/NZ/RK_tracks_production', name='rkp',
                     color='firebrick', linewidth=1.)

    # Water
    mp.readshapefile('/home/chet/gmt/data/NZ/taupo_lakes', name='lakes',
                     color='steelblue', linewidth=0.2)
    mp.readshapefile('/home/chet/gmt/data/NZ/taupo_river_poly', name='rivers',
                     color='steelblue', linewidth=0.2)
    # Bullshit filling part
    lakes = []
    for info, shape in zip(mp.lakes_info, mp.lakes):
        lakes.append(Polygon(np.array(shape), True))
    ax.add_collection(
        PatchCollection(lakes, facecolor='steelblue', edgecolor='steelblue',
                        linewidths=0.2,
                        zorder=2))
    rivers = []
    for info, shape in zip(mp.rivers_info, mp.rivers):
        rivers.append(Polygon(np.array(shape), True))
    ax.add_collection(
        PatchCollection(rivers, facecolor='steelblue', edgecolor='steelblue',
                        linewidths=0.2,
                        zorder=2))
    if show:
        fig.show()
    if savefig:
        fig.tight_layout()
        fig.savefig(savefig, dpi=720)
    return


