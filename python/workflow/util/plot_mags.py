#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/chet/EQcorrscan/')
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

"""Magnitude and b-val functions"""

def plot_mag_w_time(cat, show=True):
    """
    Plot earthquake magnitude as a function of time
    :param cat: catalog of earthquakes with mag info
    :param show: whether or not to show plot
    :return: matplotlib.pyplot.Figure
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.dpi'] = 300
    mag_tup = []
    for ev in cat:
        try:
            mag_tup.append((ev.origins[-1].time.datetime,
                            ev.preferred_magnitude().mag))
        except AttributeError:
            print('Event %s has no associated magnitude' % str(ev.resource_id))
    dates, mags = zip(*mag_tup)
    fig, ax = plt.subplots()
    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Date')
    ax.scatter(dates, mags)
    if show:
        fig.show()
    return fig


def Mc_test(cat, n_bins, test_cutoff, maxcurv_bval, start_mag=None):
    """
    Test the reliability of predetermined Mc
    :param cat: Catalog of events
    :param n_bins: Number of bins
    :param test_mag: Pre-calculated mag to test
    :param start_mag: Magnitude to start test from
    :param show: Plotting flag
    :return: (matplotlib.pyplot.Figure, best bval, cutoff mag)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from operator import itemgetter
    mags = [round(ev.magnitudes[-1].mag, 1)
            for ev in cat if len(ev.magnitudes) > 0]
    mags.sort()
    bin_vals, bins = np.histogram(mags, bins=n_bins) # Count mags in each bin
    inds = np.digitize(mags, bins) # Get bin index for each mag in mags
    bin_cents = bins - ((bins[1] - bins[0]) / 2.)
    avg_mags = []
    for i, bin in enumerate(bins):
        avg_mags.append(np.mean([mag for mag, ind in zip(mags, inds)
                                 if ind >= i + 1]))
    bvals = [np.log10(np.exp(1)) / (avg_mag - bin_cents[i])
             for i, avg_mag in enumerate(avg_mags)]
    # Errors for each bin
    errs = [2 * bval / np.sqrt(sum(bin_vals[i:]))
            for i, bval in enumerate(bvals)]
    # Error ranges for bins above start_mag
    err_rangs = [(cent, bval - err, bval + err)
                 for bval, err, cent in zip(bvals, errs, bin_cents)
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
            'bin_cents': bin_cents, 'bvals': bvals, 'errs': errs}


def bval_calc(cat, bins, MC):
    """
    Helper function to run the calculation loop
    :param mags: list of magnitudes
    :param bins: int number of bins for calculation
    :return: (non_cum_bins, cum_bins, bval_vals, bval_bins, bval_wts)
    """
    import numpy as np
    from eqcorrscan.utils.mag_calc import calc_max_curv, calc_b_value
    mags = [ev.preferred_magnitude().mag for ev in cat
            if ev.preferred_magnitude()]
    # Calculate Mc using max curvature method if not specified
    if not MC:
        Mc = calc_max_curv(mags)
    else:
        Mc = MC
    # Establish bin limits and spacing
    bin_vals = np.linspace(min(mags), max(mags), bins)
    non_cum_bins = []
    cum_bins = []
    bval_vals = []
    bval_bins = []
    bval_wts = []
    for i, val in enumerate(bin_vals):
        cum_val_count = len([ev for ev in cat if ev.preferred_magnitude()
                         and ev.preferred_magnitude().mag >= val])
        if i < len(bin_vals) - 1:
            non_cum_val_cnt = len([ev for ev in cat
                                   if ev.preferred_magnitude()
                                   and val < ev.preferred_magnitude().mag
                                   and bin_vals[i + 1] >=
                                   ev.preferred_magnitude().mag])
            non_cum_bins.append(non_cum_val_cnt)
        cum_bins.append(cum_val_count)
        if val >= Mc:
            bval_vals.append(cum_val_count)
            bval_bins.append(val)
            bval_wts.append(non_cum_val_cnt / float(len(mags)))
    # Tack 0 on end of non_cum_bins representing bin above max mag
    non_cum_bins.append(0)
    b, a = np.polyfit(bval_bins, np.log10(bval_vals), 1, w=bval_wts)
    return {'bin_vals':bin_vals, 'non_cum_bins':non_cum_bins,
            'cum_bins':cum_bins, 'bval_vals':bval_vals,
            'bval_bins':bval_bins, 'bval_wts':bval_wts,
            'b': b*-1., 'a': a, 'Mc': Mc}


def bval_plot(cat, bins=30, MC=None, title=None, show=True):
    """
    Plotting the frequency-magnitude distribution on semilog axes
    :param cat: Catalog of events with magnitudes
    :param show: Plot flag
    :return: matplotlib.pyplot.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn.apionly as sns
    matplotlib.rcParams['figure.dpi'] = 300
    import numpy as np

    b_dict = bval_calc(cat, bins, MC)
    test_dict = Mc_test(cat, n_bins=bins, test_cutoff=b_dict['Mc'],
                       maxcurv_bval=b_dict['b'], start_mag=b_dict['Mc'])
    # Now re-compute b-value for new Mc if difference larger than bin size
    mag_diff = test_dict['M_cut'] - b_dict['Mc']
    bin_interval = b_dict['bin_vals'][1] - b_dict['bin_vals'][0]
    if abs(mag_diff) > bin_interval:
        b_dict2 = bval_calc(cat, bins, MC=test_dict['M_cut'])
    if show:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121, aspect=1.)
        # Plotting first bval line
        ax.plot(b_dict['bval_bins'],
                np.power([10],[b_dict['a']-b_dict['b']*aval
                               for aval in b_dict['bval_bins']]),
                color='r', linestyle='-', label='Max-curv: log(N)=a - bM')
        if 'b_dict2' in locals():
            ax.plot(b_dict2['bval_bins'],
                    np.power([10],[b_dict2['a']-b_dict2['b']*aval
                                   for aval in b_dict2['bval_bins']]),
                    color='b', linestyle='-',
                    label='Modified Mc: log(N)=a - bM')
        ax.set_yscale('log')
        # Put b-val on plot
        text = 'B-val via max-curv: %.3f' % b_dict['b']
        ax.text(0.8, 0.7, text, transform=ax.transAxes, color='r',
                horizontalalignment='center', fontsize=8.)
        ax.text(0.8, 0.75, 'Mc via max-curv=%.2f' % b_dict['Mc'], color='r',
                transform=ax.transAxes, horizontalalignment='center',
                fontsize=8.)
        if 'b_dict2' in locals():
            text = 'Modified Mc b-val: %.3f' % b_dict2['b']
            ax.text(0.8, 0.6, text, transform=ax.transAxes, color='b',
                    horizontalalignment='center', fontsize=8.)
            ax.text(0.8, 0.65, 'Modified Mc: %.2f' % b_dict2['Mc'],
                    color='b', transform=ax.transAxes,
                    horizontalalignment='center', fontsize=8.)
        ax.scatter(b_dict['bin_vals'], b_dict['cum_bins'], label='Cumulative',
                   color='k')
        ax.scatter(b_dict['bin_vals'] + (bin_interval / 2.),
                   b_dict['non_cum_bins'], color='m', marker='^',
                   label='Non-cumulative')
        ax.set_ylim(bottom=1)
        ax.set_ylabel('Number of events')
        ax.set_xlabel('Magnitude')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('B-value plot')
        ax.legend(fontsize=9., markerscale=0.7)
        ax2 = fig.add_subplot(122)
        ax2.set_ylim([0, 3])
        ax2.errorbar(test_dict['bin_cents'], test_dict['bvals'],
                     yerr=test_dict['errs'], fmt='-o', color='k')
        ax2.axhline(test_dict['best_bval'], linestyle='--', color='b')
        ax2.axhline(b_dict['b'], linestyle='--', color='r')
        ax2.axvline(b_dict['Mc'], linestyle='--', color='b')
        ax2.axvline(test_dict['M_cut'], linestyle='--', color='r')
        ax2.text(0.5, 0.8, 'Max-curv B-value: %.3f' % b_dict['b'], color='r',
                 transform=ax2.transAxes, horizontalalignment='center',
                 fontsize=8.)
        ax2.text(0.5, 0.85, 'Max-curv Mc: %.2f' % b_dict['Mc'], color='r',
                 transform=ax2.transAxes, horizontalalignment='center',
                 fontsize=8.)
        ax2.text(0.5, 0.95, 'Modified Mc: %.2f' % test_dict['M_cut'],
                 transform=ax2.transAxes, horizontalalignment='center',
                 fontsize=8., color='b')
        ax2.text(0.5, 0.9, 'Modified b-value: %.3f' % test_dict['best_bval'],
                 color='b', transform=ax2.transAxes,
                 horizontalalignment='center', fontsize=8.)
        ax2.set_title('B-values v. cut-off magnitude')
        ax2.set_xlabel('Cut-off magnitude')
        ax2.set_ylabel('B-value')
        # Plot magnitude histogram underneath ax2
        ax3 = ax2.twinx()
        mags = [ev.preferred_magnitude().mag for ev in cat
                if ev.preferred_magnitude()]
        sns.distplot(mags, kde=False, ax=ax3, hist_kws={"alpha": 0.2})
        ax3.set_ylabel('Number of events')
        fig.tight_layout()
        fig.show()
    return


def plot_mag_v_lat(cat, method='all'):
    """
    Plotting magnitude vs latitude of events in fields
    :param cat: obspy Catalog
    :param method: 'all' or 'avg'
    :return: matplotlib.pyplot.Figure
    """
    import numpy as np
    data = [(ev.origins[-1].latitude, ev.preferred_magnitude().mag)
            for ev in cat if ev.preferred_magnitude()]
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


def convert_frac_year(number):
    from datetime import timedelta, datetime
    year = int(number)
    d = timedelta(days=(number - year) * 365)
    day_one = datetime(year,1,1)
    date = d + day_one
    return date


def plot_zmap_b_w_time(mat_file):
    """
    Take a .mat file created by ZMAP and plot it in python
    :param mat_file: Path to .mat file
    :return: matplotlib.pyplot.Figure
    """
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    bval_mat = loadmat(mat_file)['mResult']
    # First, deal with time decimals
    time_decs = bval_mat[:,[0]]
    dtos = [convert_frac_year(dec[0]) for dec in time_decs]
    bvals = bval_mat[:,[3]]
    stds = bval_mat[:,[2]]
    sigma_top = [bval + std for bval, std in zip(bvals, stds)]
    sigma_bottom = [bval - std for bval, std in zip(bvals, stds)]
    fig, ax = plt.subplots()
    ax.plot(dtos, bvals, color='k')
    ax.plot(dtos, sigma_top, linestyle='--', color='0.60')
    ax.plot(dtos, sigma_bottom, linestyle='--', color='0.60')
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Date')
    ax.set_ylabel('b-value')
    return fig
