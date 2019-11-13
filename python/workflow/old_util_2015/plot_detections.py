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


# Read Rot, Nga_N and Nga_S temps from files to temp lists
def qgis2temp_list(filename):
    import csv
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        temp_names = [row[2].split('/')[-1] for row in reader]
    return temp_names


def which_self_detect(cat):
    """
    Figure out which detections are self detections and name them accordingly
    :type cat: obspy.Catalog
    :param cat: Catalog of detections including self detections
    :return: obspy.Catalog
    """
    import numpy as np
    from obspy.core.event import ResourceIdentifier
    avg_corrs = {ev.resource_id: np.mean([float(pk.comments[0].text.split('=')[-1]) for pk in ev.picks
                                          if len(pk.comments) > 0])
                 for ev in cat}
    for ev in cat:
        if avg_corrs[ev.resource_id] > 0.99 and str(ev.resource_id).split('/')[-1].split('_')[-1] != 'self':
            temp_str = str(ev.resource_id).split('/')[-1]
            ev.resource_id = ResourceIdentifier(id=temp_str.split('_')[0] + '_self')
    return cat


def template_det_cats(cat, temp_list, outdir=False):
    """
    Seperate a catalog of detections into catalogs for each template
    :type cat: obspy.Catalog
    :param cat: Catalog of detections for a number of templates
    :type temp_list: list
    :param temp_list: list of templates you want
    :type outdir: str
    :param outdir: Directory to write catalogs and shapefiles to (optional)
    :return: dict of {template_name: obspy.Catalog}
    """
    from obspy import Catalog
    temp_det_dict = {}
    for ev in cat:
        temp_name = str(ev.resource_id).split('/')[-1].split('_')[0]
        if temp_name in temp_list or temp_list == 'all':
            if temp_name not in temp_det_dict:
                temp_det_dict[temp_name] = Catalog(events = [ev])
            else:
                temp_det_dict[temp_name].append(ev)
    if outdir:
        for temp, cat in temp_det_dict.iteritems():
            cat.write('%s/%s_detections.xml' % (outdir, temp), format="QUAKEML")
            cat.write('%s/%s_detections.shp' % (outdir, temp), format="SHAPEFILE")
    return temp_det_dict


def format_well_data(well_file):
    """
    Helper to format well txt files into (lat, lon, depth(km)) tups
    :param well_file: Well txt file
    :return: list of tuples
    """
    import csv
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

###############################################################################
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


###############################################################################

def plot_det2well_dist(big_cat, well_file, temp_list='all', method='scatter', show=True):
    """
    Function to plot events with distance from well as a function of time.
    :param cat: catalog of events
    :param well_file: text file of xyz well pts
    :param temp_list: list of templates for which we'll plot detections
    :param method: plot either the 'scatter' or daily 'average' distance
    :return: matplotlib.pyplot.Figure
    """
    from eqcorrscan.utils.mag_calc import dist_calc
    from obspy import Catalog
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    from datetime import timedelta
    matplotlib.rcParams['figure.dpi'] = 300
    well_pts = format_well_data(well_file)
    # Grab only templates in the list
    cat = Catalog()
    cat.events = [ev for ev in big_cat if
                  str(ev.resource_id).split('/')[-1].split('_')[0] in
                  temp_list or temp_list == 'all']
    time_dist_tups = []
    cat_start = min([ev.origins[-1].time.datetime for ev in cat])
    cat_end = max([ev.origins[-1].time.datetime for ev in cat])
    for ev in cat:
        if ev.origins[-1]:
            dist = min([dist_calc((ev.origins[-1].latitude,
                                   ev.origins[-1].longitude,
                                   ev.origins[-1].depth / 1000.),
                                  pt) for pt in well_pts])
            time_dist_tups.append((ev.origins[-1].time.datetime,
                                  dist))
    times, dists = zip(*time_dist_tups)
    # Plot 'em up
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Distance (m)')
    if method == 'scatter':
        ax.scatter(times, dists)
    elif method == 'average':
        dates = []
        day_avg_dist = []
        for date in date_generator(cat_start, cat_end):
            dates.append(date)
            tdds = [tdd[1] for tdd in time_dist_tups if tdd[0] > date
                    and tdd[0] < date + timedelta(days=1)]
            day_avg_dist.append(np.mean(tdds))
        ax.plot(dates, day_avg_dist)
    elif method == 'both':
        ax.scatter(times, dists)
        dates = []
        day_avg_dist = []
        for date in date_generator(cat_start, cat_end):
            dates.append(date)
            tdds = [tdd[1] for tdd in time_dist_tups if tdd[0] > date
                    and tdd[0] < date + timedelta(days=1)]
            day_avg_dist.append(np.mean(tdds))
        ax.plot(dates, day_avg_dist, color='r')
    ax.set_ylim([0, max(dists)])
    if show:
        fig.show()
    return fig


def plot_detection_wavs(cat, temp_dict, det_dict, n_events):
    """
    Wrapper on detection_multiplot() for our dataset
    :param cat: catalog of detections
    :param temp_dir: template waveform dict
    :param det_dir: detection waveform dict
    :return: matplotlib.pyplot.Figure
    """
    import numpy as np
    import matplotlib
    from obspy.core.event import ResourceIdentifier
    from eqcorrscan.utils.plotting import detection_multiplot
    matplotlib.rcParams['figure.dpi'] = 300

    rand_inds = np.random.choice(range(len(cat)), n_events, replace=False)
    for i, ev in enumerate(cat):
        if i in rand_inds:
            det_st = det_dict[ev.resource_id].copy()
            for tr in det_st:
                tr.trim(tr.stats.starttime + 2, tr.stats.endtime - 4)
            temp_id = ResourceIdentifier('smi:local/' +
                                         str(ev.resource_id).split('/')[-1].split('_')[0] +
                                         '_1sec')
            temp_st = temp_dict[temp_id]
            times = [min([tr.stats.starttime + 0.9 for tr in det_st])]
            fig = detection_multiplot(det_st, temp_st, times, save=True,
                                      savefile='/home/chet/figures/NZ/det_mulplt/%s.ps' %
                                               str(ev.resource_id).split('/')[-1],
                                      title='Detection for template %s' %
                                            str(temp_id).split('/')[-1].split('_')[0])
    return


def bounding_box(cat, bbox, depth_thresh):
    from obspy import Catalog
    new_cat = Catalog()
    new_cat.events = [ev for ev in cat if min(bbox[0]) <= ev.origins[-1].longitude <= max(bbox[0])
                      and min(bbox[1]) <= ev.origins[-1].latitude <= max(bbox[1])
                      and ev.origins[-1].depth <= depth_thresh * 1000]
    return new_cat


def plot_detections_map(cat, temp_cat, bbox, stations=None, temp_list='all', threeD=False, show=True):
    """
    Plot the locations of detections for select templates
    :type cat: obspy.core.Catalog
    :param cat: catalog of detections
    :type temp_cat: obspy.core.Catalog
    :param temp_cat: catalog of template catalogs
    :type stations: obspy.core.Inventory
    :param stations: Station data in Inventory form
    :type temp_list: list
    :param temp_list: list of str of template names
    :return: matplotlib.pyplot.Figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap

    dets_dict = template_det_cats(cat, temp_list)
    if temp_list == 'all':
        temp_list = [str(ev.resource_id).split('/')[-1] for ev in temp_cat]
    temp_dict = {str(ev.resource_id).split('/')[-1]: ev for ev in temp_cat
                 if str(ev.resource_id).split('/')[-1] in temp_list}
    # Remove keys which aren't common
    for key in temp_dict.keys():
        if key not in dets_dict:
            del temp_dict[key]
    #Set up map and xsection grid
    temp_num = len(temp_dict)
    cm = plt.get_cmap('gist_rainbow')
    # with sns.color_palette(palette='colorblind', n_colors=temp_num):
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
    if threeD:
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
    for i, temp_str in enumerate(temp_dict):
        temp_o = temp_dict[temp_str].origins[-1]
        det_lons = [ev.origins[-1].longitude for ev in dets_dict[temp_str]]
        det_lats = [ev.origins[-1].latitude for ev in dets_dict[temp_str]]
        det_depths = [ev.origins[-1].depth / 1000. for ev in dets_dict[temp_str]]
        if threeD:
            ax3d.scatter(det_lons, det_lats, det_depths, s=3.0, color=cm(1. * i / temp_num))
            ax3d.scatter(temp_o.longitude, temp_o.latitude, temp_o.depth, s=2.0, color='k', marker='x')
            ax3d.set_xlim(left=bbox[0][0], right=bbox[0][1])
            ax3d.set_ylim(top=bbox[1][0], bottom=bbox[1][1])
            ax3d.set_zlim(bottom=10., top=-0.5)
        else:
            # Map
            # mp = Basemap(projection='merc', lat_0=bbox[1][1]-bbox[1][0], lon_0=bbox[0][1]-bbox[0][0],
            #              resolution='h', llcrnrlon=bbox[0][0], llcrnrlat=bbox[1][1],
            #              urcrnrlon=bbox[0][1], urcrnrlat=bbox[1][0])
            # x, y = mp(det_lons, det_lats)
            # mp.scatter(x, y, color=cm(1. * i / temp_num))
            ax1.scatter(det_lons, det_lats, s=1.5, color=cm(1. * i / temp_num))
            ax1.scatter(temp_o.longitude, temp_o.latitude, s=2.0, color='k', marker='x')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            if bbox:
                ax1.set_xlim(left=bbox[0][0], right=bbox[0][1])
                ax1.set_ylim(top=bbox[1][0], bottom=bbox[1][1])
            # N-S
            ax2.scatter(det_depths, det_lats, s=1.5, color=cm(1. * i / temp_num))
            ax2.scatter(temp_o.depth, temp_o.latitude, s=2.0, color='k', marker='x')
            ax2.yaxis.tick_right()
            ax2.ticklabel_format(useOffset=False)
            ax2.yaxis.set_label_position('right')
            ax2.set_ylabel('Latitude')
            ax2.set_xlabel('Depth (km)')
            ax2.xaxis.set_label_position('top')
            ax2.set_xlim(left=-0.50, right=10.)
            ax2.set_ylim(top=bbox[1][0], bottom=bbox[1][1])
            # E-W
            ax3.scatter(det_lons, det_depths, s=1.5, color=cm(1. * i / temp_num))
            ax3.scatter(temp_o.longitude, -1 * temp_o.depth, s=2.0, color='k', marker='x')
            ax3.set_xlabel('Longitude')
            ax3.ticklabel_format(useOffset=False)
            ax3.set_ylabel('Depth (km)')
            ax3.yaxis.set_label_position('left')
            ax3.invert_yaxis()
            ax3.set_xlim(left=bbox[0][0], right=bbox[0][1])
            ax3.set_ylim(top=-0.50, bottom=10.)
    fig = plt.gcf()
    if show:
        fig.show()
    return fig


def template_extents(cat, temp_cat, temp_list='all', param='avg_dist', show=True):
    """
    Measure parameters of the areal extent of template detections
    :param cat: Detections catalog
    :param temp_cat: Templates catalog
    :param param: What parameter are we measuring? Average template-detection distance or area?
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from eqcorrscan.utils.mag_calc import dist_calc
    import seaborn.apionly as sns

    dets_dict = template_det_cats(cat, temp_list)
    temp_dict = {str(ev.resource_id).split('/')[-1]: ev for ev in temp_cat
                 if str(ev.resource_id).split('/')[-1] in temp_list}
    # Remove keys which aren't common
    for key in temp_dict.keys():
        if key not in dets_dict:
            del temp_dict[key]
    param_dict = {}
    for key, ev in temp_dict.iteritems():
        temp_o = ev.origins[-1]
        if param == 'avg_dist':
            param_dict[key] = np.mean([dist_calc((temp_o.latitude, temp_o.longitude, temp_o.depth / 1000.0),
                                                 (det.origins[-1].latitude,
                                                  det.origins[-1].longitude,
                                                  det.origins[-1].depth / 1000.0)) for det in dets_dict[key]])
    ax = sns.distplot([avg for key, avg in param_dict.iteritems()])
    ax.set_title('')
    ax.set_xlabel('Average template-detection distance per template (km)')
    fig = plt.gcf()
    if show:
        fig.show()
    return fig


def add_patches(shapefiles):
    from descartes import PolygonPatch
    patches = []
    for file in shapefiles:
        for poly in file.geometry:
            # deal with single polygons and multipolygons
            if poly.geom_type == 'Polygon':
                p = PolygonPatch(poly, facecolor='#6699cc', alpha=1., zorder=1)
                patches.append(p)
            elif poly.geom_type == 'MultiPolygon':
                for single in poly:
                    q = PolygonPatch(single, facecolor='#6699cc', alpha=1., zorder=1)
                    patches.append(q)
    return patches


def plot_location_changes(cat, bbox, show=True):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from matplotlib.collections import PatchCollection
    import numpy as np

    fig, ax = plt.subplots()
    mp = Basemap(projection='merc', lat_0=bbox[1][1] - bbox[1][0], lon_0=bbox[0][1] - bbox[0][0],
                 resolution='h', llcrnrlon=bbox[0][0], llcrnrlat=bbox[1][1],
                 urcrnrlon=bbox[0][1], urcrnrlat=bbox[1][0],
                 suppress_ticks=True)
    mp.drawcoastlines()
    mp.fillcontinents(color='white')
    mp.readshapefile('/home/chet/gmt/data/NZ/taupo_river_poly', 'rivers', color='b')
    mp.readshapefile('/home/chet/gmt/data/NZ/taupo_lakes', 'lakes', color='b')
    # Goofy basemap filling for shapefile?
    # patches = add_patches([rivers, lakes])
    # for patch in patches:
    #     ax.add_patch(patch)
    mp.drawparallels(np.arange(bbox[1][1], bbox[0][1], 0.025), linewidth=0, labels=[1, 0, 0, 0])
    mp.drawmeridians(np.arange(bbox[0][0], bbox[1][0], 0.025), linewidth=0, labels=[0, 0, 0, 1])
    for ev in cat:
        # This is specific to the current origin order in my catalogs
        lats = [ev.origins[-1].latitude, ev.origins[-2].latitude]
        lons = [ev.origins[-1].longitude, ev.origins[-2].longitude]
        depths = [ev.origins[-1].depth / 1000., ev.origins[-2].depth / 1000.]
        x, y = mp(lons, lats)
        mp.plot(x, y, color='r', latlon=False)
        mp.plot(lons[0], lats[0], color='r', marker='o', mfc='none', latlon=True)
        mp.plot(lons[1], lats[1], color='r', marker='o', latlon=True)
        plt.xticks()
        plt.yticks()
    if show:
        fig.show()
    return fig

######## RATE-RELATED FUNCTIONS ########

def plot_detections_rate(cat, temp_list='all', bbox=None, depth_thresh=None, cumulative=False, detection_rate=False):
    """
    Plotting detections for some catalog of detected events
    :type cat: obspy.core.Catalog
    :param cat: Catalog containting detections. Should have detecting template info in the resource_id
    :type temp_list: list
    :param temp_list: list of template names which we want to consider for this plot
    :type bbox: tuple of tuple
    :param bbox: select only events within a certain geographic area bound by ((long, long), (lat, lat))
    :type depth_thresh: float
    :param depth_thresh: Depth cutoff for detections being plotted in km
    :type cumulative: bool
    :param cumulative: Whether or not to combine detections into
    :type detection_rate: bool
    :param detection_rate: Do we plot derivative of detections?
    :return:
    """
    import matplotlib.pyplot as plt
    from eqcorrscan.utils import plotting
    from datetime import timedelta
    import matplotlib.dates as mdates
    from obspy import Catalog

    # If specified, filter catalog to only events in geographic area
    if bbox:
        det_cat = bounding_box(cat, bbox, depth_thresh)
    # Sort det_cat by origin time
    else:
        det_cat = cat
    if not det_cat[0].origins[-1]:
        det_cat.events.sort(key=lambda x: x.origins[0].time)
    else:
        det_cat.events.sort(key=lambda x: x.origins[-1].time)
    # Put times and names into sister lists
    if detection_rate and not cumulative:
        cat_start = min([ev.origins[-1].time.datetime for ev in det_cat])
        cat_end = max([ev.origins[-1].time.datetime for ev in det_cat])
        det_rates = []
        dates = []
        if not det_cat[0].origins[-1]:
            for date in date_generator(cat_start, cat_end):
                dates.append(date)
                det_rates.append(len([ev for ev in det_cat if ev.origins[0].time.datetime > date
                                      and ev.origins[0].time.datetime < date + timedelta(days=1)]))
        else:
            for date in date_generator(cat_start, cat_end):
                dates.append(date)
                det_rates.append(len([ev for ev in det_cat if ev.origins[-1].time.datetime > date
                                       and ev.origins[-1].time.datetime < date + timedelta(days=1)]))
        fig, ax1 = plt.subplots()
        # ax1.step(dates, det_rates, label='All templates', linewidth=1.0, color='black')
        ax1 = plt.subplot(111)
        ax1.bar(dates, det_rates)
        ax1.xaxis_date()
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative detection rate (events/day)')
        plt.title('Cumulative detection rate for all templates')
    else:
        det_times = []
        temp_names = []
        temp_dict = {}
        for ev in det_cat:
            temp_name = str(ev.resource_id).split('/')[-1].split('_')[0]
            o = ev.origins[-1] or ev.origins[0]
            if temp_name not in temp_dict:
                temp_dict[temp_name] = [o.time.datetime]
            else:
                temp_dict[temp_name].append(o.time.datetime)
        for temp_name, det_time_list in temp_dict.iteritems():
            if temp_name in temp_list:
                det_times.append(det_time_list)
                temp_names.append(temp_name)
            elif temp_list == 'all':
                det_times.append(det_time_list)
                temp_names.append(temp_name)
        if cumulative:
            fig = plotting.cumulative_detections(dates=det_times, template_names=temp_names,
                                                 plot_grouped=True, show=False, plot_legend=False)
            if detection_rate:
                ax2 = fig.get_axes()[0].twinx()
                cat_start = min([ev.origins[-1].time.datetime for ev in det_cat])
                cat_end = max([ev.origins[-1].time.datetime for ev in det_cat])
                det_rates = []
                dates = []
                for date in date_generator(cat_start, cat_end):
                    dates.append(date)
                    det_rates.append(len([ev for ev in det_cat if ev.origins[-1].time.datetime > date
                                          and ev.origins[-1].time.datetime < date + timedelta(days=1)]))
                ax2.step(dates, det_rates, label='All templates', linewidth=2.0, color='black')
                ax2.set_ylabel('Cumulative detection rate (events/day)')
        else:
            fig = plotting.cumulative_detections(dates=det_times, template_names=temp_names)
    return fig


def mrp_2_flow_dict(flow_csv, well_list=None):
    """
    Handle taking the csv files made from mrp and creating the flow dict required for plot_flow_rates
    :type flow_csv: str
    :param flow_csv: Path to csv file with MRP flow rates
    :type well_list: None or list of str
    :param well_list: List of wells you would like to include
    :return: dict of flows {datetime: {well: flow rate}}
    """
    import datetime
    import numpy as np
    import pandas as pd
    flows = pd.read_csv(flow_csv, header=[0, 1], tupleize_cols=True)
    if not well_list:
        well_list, measure = zip(*list(flows.columns.values))
        well_list = [well for well in well_list if well != 'Date' and well != 'LP Brine']
    flow_dict = {datetime.datetime.strptime(row[('Date', 'Time')], "%m/%d/%Y %H:%M"):
                     {index[0]: flow for index, flow in row.iteritems() if index[0] != 'Date'
                      and index[0] != 'LP Brine' and index[0] in well_list and index[1] == 'Flow'}
                 for ind, row in flows.iterrows()}
    whp_dict = {datetime.datetime.strptime(row[('Date', 'Time')], "%m/%d/%Y %H:%M"):
                     {index[0]: whp for index, whp in row.iteritems() if index[0] != 'Date'
                      and index[0] != 'LP Brine' and index[0] in well_list and index[1] == 'WHP'}
                for ind, row in flows.iterrows()}
    # Sum the flows and add to the dto dict
    for dto, well_dict in flow_dict.iteritems(): # Convert nans to 0.0
        for well, flow in well_dict.iteritems():
            if type(flow) != float:
                try:
                    well_dict[well] = float(flow)
                except ValueError:
                    well_dict[well] = 0.0
            elif np.isnan(flow):
                well_dict[well] = 0.0
        well_dict['total'] = sum([flow for well, flow in well_dict.iteritems() if well != 'LP Brine'])
    for dto, well_dict in whp_dict.iteritems(): # Convert nans to 0.0
        for well, whp in well_dict.iteritems():
            if type(whp) != float:
                try:
                    well_dict[well] = float(whp)
                except ValueError:
                    well_dict[well] = 0.0
            elif np.isnan(whp):
                well_dict[well] = 0.0
        well_dict['total'] = sum([flow for well, flow in well_dict.iteritems() if well != 'LP Brine'])

    return flow_dict, whp_dict


def plot_flow_rates(flow_dict, pres_dict, start_date, end_date, well_list='all', pressure=False,
                    total=False, volume=False, fig=None):
    """
    Plotting function for injection flows for geothermal power production.
    :type flow_csv: str
    :param flow_csv: Path to csv file with flow rates (this should change to Python native format)
    :type start_date: str
    :param start_date: Plot start date in %d/%m/%Y format
    :type end_date: str
    :param end_date: Plot end date in %d/%m/%Y format
    :type well_list: str or list
    :param well_list: Either 'all' or a list of strings with wells of interest matching headers in csv
    :type total: bool
    :param total: Plot total injection rate for 'True', otherwise seperate out wells
    :type fig: None or matplotlib.figure
    :param fig: Input figure with detections already plotted
    :return: class matplotlib.figure
    """
    #TODO Generalize this so that flow_rate can be format agnostic?
    import datetime
    import matplotlib.pyplot as plt

    # Checks for plotting pressure or flow
    if pressure:
        plot_dict = pres_dict
    else:
        plot_dict = flow_dict
    # Date strings to datetime objects
    start = datetime.datetime.strptime(start_date, "%d/%m/%Y")
    end = datetime.datetime.strptime(end_date, "%d/%m/%Y")
    # Get flows into list of (dto, flow) for plotting
    if total:
        well_list = 'total'
        plot_tups_dict = {'total':[(dto, well_dict['total']) for dto, well_dict in plot_dict.iteritems()
                                   if dto > start and dto < end]}
    else:
        plot_tups_dict = {}
        for dto, well_dict in plot_dict.iteritems():
            if dto > start and dto < end:
                for well in well_dict.keys():
                    if well in well_list or well_list == 'all':
                        if well != 'total':
                            if well in plot_tups_dict:
                                plot_tups_dict[well].append((dto, well_dict[well]))
                            else:
                                plot_tups_dict[well] = [(dto, well_dict[well])]
    # Sort the tuple lists by datetime object
    for well, list_tups in plot_tups_dict.iteritems():
        list_tups.sort(key=lambda x: x[0])
    # Set up figure object
    if fig:
        fig_final = fig
        lines, labels = fig_final.get_axes()[0].get_legend_handles_labels()
        if fig_final.get_axes()[0].legend_:
            fig_final.get_axes()[0].legend_.remove() # Clear old legend
        axes = fig_final.get_axes()[0].twinx()
    else:
        fig_final = plt.figure()
        axes = plt.gca()
    for t in axes.get_yticklabels():
        t.set_color('r')
    if pressure:
        axes.set_ylabel('Wellhead Pressure (bar-g)', color='r')
    else:
        axes.set_ylabel('Flow rate (T/h)', color='r')
    axes.set_xlabel('Date')
    for well, flow_list in plot_tups_dict.iteritems():
        if well in well_list or well_list:
            dtos, flows = zip(*flow_list)
            if well_list == 'total' and not volume:
                label = 'Total flow rate'
                axes.plot(dtos, flows, label=label, color='r')
            elif well_list == 'total' and volume:
                label = 'Cumulative injected volume'
                cum_vols = []
                for i, flow in enumerate(flows):
                    if i == 0:
                        cum_vols = [flow * 24]
                    else:
                        cum_vols.append((flow * 24) + cum_vols[i-1])
                axes.plot(list(dtos), cum_vols, label=label)
            else:
                label = well
                axes.plot(dtos, flows, label=label)
    lines2, labels2 = axes.get_legend_handles_labels()
    if fig:
        axes.legend(lines + lines2, labels + labels2, loc=2,
                    prop={'size': 12}, ncol=3)
    else:
        axes.legend(lines2, labels2, loc=2, prop={'size': 8}, ncol=3)
    if not pressure:
        try:
            axes.set_ylim([min([fl[1] for well, list_tups in
                                plot_tups_dict.iteritems() for fl in list_tups]),
                           max([fl[1] for well, list_tups in
                                   plot_tups_dict.iteritems() for fl in list_tups])
                           + 200])
        except ValueError:
            print('Probably no flow data available for this time period')
    return fig_final


##### OTHER MISC FUNCTIONS #####

def plot_catalog_uncertainties(cat1, cat2=None, RMS=True, uncertainty_ellipse=False):
    """
    Plotting of various uncertainty and error parameters of catalog. Catalog must have
    event.origin.quality and event.origin.origin_uncertainty for each event.
    :type cat1: obspy.Catalog
    :param cat1: catalog with uncertainty info
    :type cat2: obspy.Catalog
    :param cat2: Catalog which we'd like to compare to cat1
    :return: matplotlib.Figure
    """
    import seaborn.apionly as sns
    from matplotlib.patches import Ellipse
    from pyproj import Proj, transform

    #kwarg sanity check
    if uncertainty_ellipse:
        RMS=False
    #Do a check on kwargs
    if RMS and uncertainty_ellipse:
        print('Will not plot RMS on top of uncertainty ellipse, choosing only RMS')
        uncertainty_ellipse=False
    # Sort the catalogs by time, in case we'd like to compare identical events with differing picks
    cat1.events.sort(key=lambda x: x.origins[-1].time)
    if cat2:
        cat2.events.sort(key=lambda x: x.origins[-1].time)
    if RMS:
        ax1 = sns.distplot([ev.origins[-1].quality.standard_error for ev in cat1],
                           label='Catalog 1', kde=False)
        if cat2:
            ax1 = sns.distplot([ev.origins[-1].quality.standard_error for ev in cat2],
                               label='Catalog 2', ax=ax1, kde=False)
        ax1.legend()
        ax1.set_xlabel('RMS (sec)')
        ax1.set_title('Catalog RMS')
        ax1.set_ylabel('Number of events')
        plt.show()
        return
    if uncertainty_ellipse:
        # Set input and output projections
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:2193')
        ax1 = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=2)
        # Lets try just a flat cartesian coord system
        # First, pyproj coordinates into meters, so that we can plot the ellipses in same units
        dict1 = {ev.resource_id: {'coords': transform(inProj, outProj, ev.origins[-1].longitude, ev.origins[-1].latitude),
                                  'depth': ev.origins[-1].depth,
                                  'ellps_max': ev.origins[-1].origin_uncertainty.max_horizontal_uncertainty,
                                  'ellps_min': ev.origins[-1].origin_uncertainty.min_horizontal_uncertainty,
                                  'ellps_az': ev.origins[-1].origin_uncertainty.azimuth_max_horizontal_uncertainty}
                 for ev in cat1}
        if cat2:
            dict2 = {ev.resource_id: {'coords': transform(inProj, outProj, ev.origins[-1].longitude, ev.origins[-1].latitude),
                                      'depth': ev.origins[-1].depth,
                                      'ellps_max': ev.origins[-1].origin_uncertainty.max_horizontal_uncertainty,
                                      'ellps_min': ev.origins[-1].origin_uncertainty.min_horizontal_uncertainty,
                                      'ellps_az': ev.origins[-1].origin_uncertainty.azimuth_max_horizontal_uncertainty}
                     for ev in cat2}
        for eid, ev_dict in dict1.iteritems():
            ax1.add_artist(Ellipse(xy=ev_dict['coords'], width=ev_dict['ellps_min'],
                                   height=ev_dict['ellps_max'], angle=180 - ev_dict['ellps_az'],
                                   color='r', fill=False))
            if cat2:
                ax1.add_artist(Ellipse(xy=dict2[eid]['coords'], width=dict2[eid]['ellps_min'],
                                   height=dict2[eid]['ellps_max'], angle=180 - dict2[eid]['ellps_az'],
                                   color='b', fill=False))
        # Set axis limits
        ax1.set_xlim(min(subdict['coords'][0] for subdict in dict1.values()),
                     max(subdict['coords'][0] for subdict in dict1.values()))
        ax1.set_ylim(min(subdict['coords'][1] for subdict in dict1.values()),
                     max(subdict['coords'][1] for subdict in dict1.values()))
        ax2 = plt.subplot2grid((3,3), (0,2), rowspan=2)
        # Lots of trig in here somewhere
        ax3 = plt.subplot2grid((3,3), (2,0), colspan=2)
        # Here too
        plt.gcf().show()
    return


def plot_pick_uncertainty(cat, show=True):
    import seaborn.apionly as sns
    import matplotlib.pyplot as plt
    sns.distplot([pk.time_errors.uncertainty for ev in cat for pk in ev.picks])
    fig = plt.gcf()
    if show:
        fig.show()
    return fig


def make_residuals_dict(cat):
    stas = set([pk.waveform_id.station_code for ev in cat for pk in ev.picks])
    residual_dict = {sta: {'P': [], 'S': []} for sta in stas}
    for ev in cat:
        for arr in ev.origins[-1].arrivals:
            pk = arr.pick_id.get_referred_object()
            residual_dict[pk.waveform_id.station_code][pk.phase_hint].append(arr.time_residual)
    return residual_dict


def plot_station_residuals(cat1, sta_list='all', plot_type='bar',
                           kde=False, hist=True, savefig=False):
    """ Plotting function to compare stations residuals between catalogs, stations and phases
    :type cat1: obspy.Catalog
    :param cat1: Catalog of events we're interested in
    :type cat2: obspy.Catalog (optional)
    :param cat2: Can be given a second catalog for comparison
    :type sta_list: list
    :param sta_list: List of stations you'd like to plot
    :type plot_type: str
    :param plot_type: either a 'bar' plot or a seaborn 'hist'ogram
    :type kde: bool
    :param kde: Tells seaborn whether or not to plot the kernel density estimate of the distributions
    :type hist: bool
    :param hist: Tells seaborn whether or not to plot the histogram
    :return: matplotlib.pyplot.Figure
    """
    import seaborn.apionly as sns
    import numpy as np
    # Create dict of {sta: {phase: residual}}
    residual_dict1 = make_residuals_dict(cat1)
    if plot_type == 'hist':
        fig, ax1 = plt.subplots(6, 5, figsize=(20, 20))
        axes = ax1.flat
        for i, sta in enumerate(residual_dict1):
            if sta in sta_list or sta_list == 'all':
                sns.distplot(residual_dict1[sta]['P'], label='Catalog 1: %s P-picks' % sta,
                             ax=axes[i], kde=kde, hist=hist)
                if len(residual_dict1[sta]['S']) > 1:
                    sns.distplot(residual_dict1[sta]['S'], label='Catalog 1: %s S-picks' % sta,
                                 ax=axes[i], kde=kde, hist=hist)
            axes[i].set_title(sta)
            axes[i].set_xlim([-2., 2.])
            # axes[i].legend()
        if savefig:
            plt.savefig(savefig)
        else:
            plt.show()
    else:
        sta_chans_P, P_avgs = zip(*sorted([(stachan[:4], np.mean(dict['P']))
                                           for stachan, dict in residual_dict1.iteritems()], key=lambda x: x[0]))
        sta_chans_S, S_avgs = zip(*sorted([(stachan[:4], np.mean(dict['S']))
                                           for stachan, dict in residual_dict1.iteritems()], key=lambda x: x[0]))
        fig, ax = plt.subplots()
        # Set bar width
        width = 0.50
        # Arrange indices
        ind = np.arange(len(sta_chans_P))
        barsP = ax.bar(ind, P_avgs, width, color='r')
        barsS = ax.bar(ind + width, S_avgs, width, color='b')
        # ax.set_xticks(ind + width)
        # ax.set_xticklabels(sta_chans_P)
        ax.legend((barsP[0], barsS[0]), ('P picks', 'S-picks'))
        ax.set_title('Average arrival residual by station and phase')
        ax.set_ylabel('Arrival residual (s)')
        for barP, barS, stachan in zip(barsP, barsS, sta_chans_P):
            height = max(abs(barP.get_height()), abs(barS.get_height()))
            # Account for large negative bars
            if max(barP.get_y(), barS.get_y()) < 0.0:
                ax.text(barP.get_x() + barP.get_width(), -1.0 * height - 0.05,
                        stachan, ha='center', fontsize=18)
            else:
                ax.text(barP.get_x() + barP.get_width(), height + 0.05,
                        stachan, ha='center', fontsize=18)
        fig.show()
    return fig


def event_diff_dict(cat1, cat2):
    """
    Generates a dictionary of differences between two catalogs with identical events
    :type cat1: obspy.Catalog
    :param cat1: catalog of events parallel to cat2
    :type cat2: obspy.Catalog
    :param cat2: catalog of events parallel to cat1
    :return: dict
    """
    from eqcorrscan.utils.mag_calc import dist_calc

    diff_dict = {}
    for ev1 in cat1:
        ev1_o = ev1.origins[-1]
        ev2 = [ev for ev in cat2 if ev.resource_id == ev1.resource_id][0]
        ev2_o = ev2.origins[-1]
        diff_dict[ev1.resource_id] = {'dist': dist_calc((ev1_o.latitude,
                                                         ev1_o.longitude,
                                                         ev1_o.depth / 1000.00),
                                                        (ev2_o.latitude,
                                                         ev2_o.longitude,
                                                         ev2_o.depth / 1000.00)),
                                      'pick_residuals1': {'P': [], 'S': []},
                                      'pick_residuals2': {'P': [], 'S': []},
                                      'cat1_RMS': ev1_o.quality.standard_error,
                                      'cat2_RMS': ev2_o.quality.standard_error,
                                      'RMS_change': ev2_o.quality.standard_error - ev1_o.quality.standard_error,
                                      'cat1_picks': len(ev1.picks),
                                      'cat2_picks': len(ev2.picks),
                                      'x_diff': ev2_o.longitude - ev1_o.longitude,
                                      'y_diff': ev2_o.latitude - ev1_o.latitude,
                                      'z_diff': ev2_o.depth - ev1_o.depth,
                                      'min_uncert_diff': ev2_o.origin_uncertainty.min_horizontal_uncertainty -
                                                         ev1_o.origin_uncertainty.min_horizontal_uncertainty,
                                      'max_uncert_diff': ev2_o.origin_uncertainty.max_horizontal_uncertainty -
                                                         ev1_o.origin_uncertainty.max_horizontal_uncertainty,
                                      'az_max_uncert_diff': ev2_o.origin_uncertainty.azimuth_max_horizontal_uncertainty -
                                                            ev1_o.origin_uncertainty.azimuth_max_horizontal_uncertainty}
        for ar in ev1_o.arrivals:
            phs = ar.pick_id.get_referred_object().phase_hint
            diff_dict[ev1.resource_id]['pick_residuals1'][phs].append((ar.pick_id.get_referred_object().waveform_id.station_code + '.' +
                                                                       ar.pick_id.get_referred_object().waveform_id.channel_code,
                                                                       ar.time_residual))
        for ar in ev2_o.arrivals:
            phs = ar.pick_id.get_referred_object().phase_hint
            diff_dict[ev1.resource_id]['pick_residuals2'][phs].append((ar.pick_id.get_referred_object().waveform_id.station_code + '.' +
                                                                       ar.pick_id.get_referred_object().waveform_id.channel_code,
                                                                       ar.time_residual))
    return diff_dict


def plot_catalog_differences(diff_dict, param='dist', param2=None, show=True):
    """
    Plot differences between catalogs with S or no S picks
    :param param:
    :return: matplotlib.axis
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn.apionly as sns

    param_list = [ev_dict[param] for rid, ev_dict in diff_dict.iteritems()]
    if param2:
        param_list2 = [ev_dict[param2] for rid, ev_dict in diff_dict.iteritems()]
    ax = sns.distplot(param_list)
    if param2:
        ax = sns.regplot(np.array(param_list), np.array(param_list2))
    if show:
        plt.gcf().show()
    return ax


def seis_viewer_compare(ev1, ev2):
    """
    Hard-coded crap to launch seismicity viewer for two events (one with S-picks, one without)
    :param ev1:
    :param ev2:
    :return:
    """
    import subprocess

    filename1 = '/media/chet/hdd/seismic/NZ/NLLoc/mrp/2015_Rawlinson_S_9-21/loc/%s.*.*.grid0.loc.hyp' % \
                str(ev1.resource_id).split('/')[-1]
    filename2 = '/media/chet/hdd/seismic/NZ/NLLoc/mrp/2015_Rawlinson_S_9-21/rewt_0.05_test/loc/%s.*.*.grid0.loc.hyp' % \
                str(ev2.resource_id).split('/')[-1]
    print(filename1)
    cmnd = 'java net.alomax.seismicity.Seismicity %s %s' % (filename1, filename2)
    subprocess.call(cmnd, shell=True)
    return


def seis_view_catalogs(cat1, cat2):
    for i, ev in enumerate(cat1):
        seis_viewer_compare(ev, cat2[i])
    return


def bbox_two_cat(cat1, cat2, bbox, depth_thresh):
    from obspy import Catalog

    new_cat1 = Catalog()
    new_cat2 = Catalog()
    for i, ev in enumerate(cat1):
        if min(bbox[0]) <= ev.origins[-1].longitude <= max(bbox[0]) \
                and min(bbox[1]) <= ev.origins[-1].latitude <= max(bbox[1]) \
                and ev.origins[-1].depth <= depth_thresh * 1000:
            new_cat1.events.append(ev)
            new_cat2.events.append(cat2[i])
    return new_cat1, new_cat2


def find_common_events(catP, catS):
    """
    Takes parallel catalogs, one with P only, the other with added S phases
    :param catP: Catalog with only p-picks
    :param catS: Catalog with S-picks added
    :return: two parallel catalogs including events with S-picks and their corresponding P-only versions
    """
    from obspy import Catalog
    comm_cat_S = Catalog()
    comm_cat_P = Catalog()
    for i, ev in enumerate(catS):
        if len([pk for pk in ev.picks if pk.phase_hint == 'S']) > 0:
            comm_cat_S.events.append(ev)
            comm_cat_P.events.append(catP[i])
    return comm_cat_P, comm_cat_S
