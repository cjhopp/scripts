#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt

import csv
import os
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates

from copy import deepcopy
from glob import glob
from collections import defaultdict
from datetime import timedelta, datetime
from itertools import cycle
# from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Ellipse
from matplotlib.dates import date2num
from pyproj import Proj, transform
from obspy import Catalog, UTCDateTime, Stream
from obspy.core.event import ResourceIdentifier
from eqcorrscan.utils import plotting, pre_processing
from eqcorrscan.utils.mag_calc import dist_calc
from eqcorrscan.utils.plotting import detection_multiplot
from eqcorrscan.core.match_filter import Detection, Family, Party, Template
# Import local stress functions
try:
    from plot_stresses import parse_arnold_params, parse_arnold_grid
except:
    print('On server. pathlib not installed')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def date_generator(start_date, end_date):
    # Generator for date looping
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def grab_day_wavs(wav_dirs, dto, stachans):
    # Helper to recursively crawl paths searching for waveforms from a dict of
    # stachans for one day
    import os
    import fnmatch
    from itertools import chain
    from obspy import read, Stream

    st = Stream()
    wav_files = []
    for path, dirs, files in chain.from_iterable(os.walk(path)
                                                 for path in wav_dirs):
        print('Looking in %s' % path)
        for sta, chans in iter(stachans.items()):
            for chan in chans:
                for filename in fnmatch.filter(files,
                                               '*.%s.*.%s*%d.%03d'
                                                       % (
                                               sta, chan, dto.year,
                                               dto.julday)):
                    wav_files.append(os.path.join(path, filename))
    print('Reading into memory')
    for wav in wav_files:
        st += read(wav)
    stachans = [(tr.stats.station, tr.stats.channel) for tr in st]
    for stachan in list(set(stachans)):
        tmp_st = st.select(station=stachan[0], channel=stachan[1])
        if len(tmp_st) > 1 and len(set([tr.stats.sampling_rate
                                        for tr in tmp_st])) > 1:
            print('Traces from %s.%s have differing samp rates'
                  % (stachan[0], stachan[1]))
            for tr in tmp_st:
                st.remove(tr)
            tmp_st.resample(sampling_rate=100.)
            st += tmp_st
    st.merge(fill_value='interpolate')
    print('Checking for trace length. Removing if too short')
    rm_trs = []
    for tr in st:
        if len(tr.data) < (86400 * tr.stats.sampling_rate * 0.8):
            rm_trs.append(tr)
        if tr.stats.starttime != dto:
            print('Trimming trace %s.%s with starttime %s to %s'
                  % (tr.stats.station, tr.stats.channel,
                     str(tr.stats.starttime), str(dto)))
            tr.trim(starttime=dto, endtime=dto + 86400,
                    nearest_sample=False)
    if len(rm_trs) != 0:
        print('Removing traces shorter than 0.8 * daylong')
        for tr in rm_trs:
            st.remove(tr)
    else:
        print('All traces long enough to proceed to dayproc')
    return st

def qgis2temp_list(filename):
    # Read Rot, Nga_N and Nga_S temps from files to temp lists
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

def catalog_to_gmt(catalogs, outfile, dd_only=True, centroids=False,
                   stress_dir=None, clust_nums=None, min_ev=2, color_nu=False,
                   sigmas=False):
    """
    Write a catalog to a file formatted for gmt plotting


    # XX TODO maybe add a formatting option for date strings instead of
    # XX TODO integer date.

    Format: lon, lat, depth(km bsl), integer day, normalized mag(?)
    :param catalogs: a list of catalogs, normally one for a list of clusters
    :param outfile: Path to output file for gmt plotting
    :param dd_only: Whether to only accept GrowClust locations
    :param centroids: Flag for plotting only the cluster centroid to declutter
        stress plots. Must provide stress_dir. If sigmas == True, will plot
        the principle stresses projected onto horizontal surface. Otherwise,
        will plot SHmax wedges.
        If you don't want to plot symbols at cluster centroid, you can also
        provide a list of (centroid x, centroid y) of boxes in case of
        quadtree clustering.
    :param stress_dir: Output directory for Arnold-Townend stress inversion
    :param clust_nums: If the cluster numbers are different than simply the
        order of the catalogs given, provide them as a list.
    :param min_ev: Minimum number of events per cluster. Will skip otherwise.
    :param color_nu: Color the wedges by the value of nu?
    :param sigmas: Plot principle stress vectors in map view?
    :return:
    """
    # Check centroid args
    if centroids and not stress_dir:
        print('Provide stress directory if plotting centroids')
        return
    # Make hex list
    pal_hex = sns.color_palette().as_hex()
    print(pal_hex)
    cols = cycle(pal_hex)
    syms = cycle(['c', 's', 'd', 't', 'n', 'a', 'i'])
    with open(outfile, 'w') as f:
        for j, cat in enumerate(catalogs):
            # Write clust no. comment
            if clust_nums:
                clust_name = '{}_0'.format(clust_nums[j])
                f.write('# Cluster {}\n'.format(clust_nums[j]))
            else:
                clust_name = '{}_0'.format(j)
                f.write('# Cluster {}\n'.format(j))
            if stress_dir:
                # Pull stress inversion results
                froot = '/'.join([stress_dir, clust_name])
                grid_f = '{}.{}.dat'.format(froot, 's123grid')
                param_files = glob('{}.*{}.dat'.format(froot, 'dparameters'))
                if not os.path.isfile(grid_f):
                    print('No output grid file...cluster probably not used')
                    continue
                phivec, thetavec = parse_arnold_grid(grid_f)
                strs_params = parse_arnold_params(param_files)
                # Grab means, 10% and 90% azimuths
                mean = strs_params['Shmax']['mean']
                X10 = strs_params['Shmax']['X10']
                X90 = strs_params['Shmax']['X90']
                nu = strs_params['nu']['mean']
            if len(cat) < min_ev:
                print('Too few events in cluster. Skipping')
                continue
            col = next(cols)
            # Plot various symbols for stress inv results at clust center
            if centroids:
                if type(centroids) == list:
                    cent_lat = centroids[j][1]
                    cent_lon = centroids[j][0]
                elif centroids == True:
                    # Plotting SHmax bowtie (maybe other stuff) at clust
                    # centroid
                    cent_lat = np.median([
                        ev.preferred_origin().latitude for ev in cat
                        if ev.preferred_origin().method_id.id.endswith('GrowClust')])
                    cent_lon = np.median([
                        ev.preferred_origin().longitude for ev in cat
                        if ev.preferred_origin().method_id.id.endswith('GrowClust')])
                if sigmas:
                    # Grab sigma trend and plunge values
                    s_cols = ['red', 'green', 'blue']
                    size=1.5
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
                        f.write('{} {} 0 {} {}\n'.format(cent_lon, cent_lat,
                                                         phi, length))
                else:
                    # Flip these around for other half of bowtie
                    back_10 = X10 - 180.
                    back_90 = X90 - 180.
                    if back_10 < 0.:
                        back_10 += 360.
                    if back_90 < 0.:
                        back_90 += 360.
                    if color_nu:
                        f.write('{} {} {} {} {}\n'.format(cent_lon, cent_lat,
                                                          nu, X10, X90))
                        f.write('{} {} {} {} {}\n'.format(cent_lon, cent_lat,
                                                          nu, back_10,
                                                          back_90))
                    else:
                        f.write('>-Glightgray\n')
                        f.write('{} {} {} {}\n'.format(cent_lon, cent_lat,
                                                       X10, X90))
                        f.write('{} {} {} {}\n'.format(cent_lon, cent_lat,
                                                       back_10, back_90))
                continue
            # Write rgb to header
            if color_nu:
                f.write('>\n')
            else:
                f.write('>-G{}\n'.format(col))
            if col == pal_hex[0]:
                sym = next(syms)
            cat.events.sort(key=lambda x: x.preferred_origin().time)
            t0 = cat[0].preferred_origin().time.datetime
            mags = np.array([ev.preferred_magnitude().mag for ev in cat])
            mags = list(mags / max(mags * 7)) # Squared, normalized mags
            days = [int((ev.preferred_origin().time.datetime
                         - t0).total_seconds() / 86400.)
                    for ev in cat] # Elapsed days since first event
            for i, ev in enumerate(cat):
                o = ev.preferred_origin()
                if not o.method_id:
                    continue
                # We want depths to in m bsl here
                if dd_only and o.method_id.id.endswith('GrowClust'):
                    dp = o.depth / 1000.
                    if color_nu:
                        # 4th field is nu
                        f.write(
                            '{} {} {} {} {} {}\n'.format(
                                o.longitude, o.latitude, nu, dp,
                                mags[i], sym))
                    else:
                        # 4th field is integer day elapsed
                        f.write(
                            '{} {} {} {} {} {}\n'.format(
                                o.longitude, o.latitude, dp, days[i],
                                mags[i], sym))
                else:
                    dp = o.depth
                    f.write(
                        '{} {} {} {} {} {}\n'.format(
                            o.longitude, o.latitude,
                            dp, days[i], mags[i], sym))
    return

def detection_multiplot_cjh(stream, template, times, events=None, title=None,
                            streamcolour='k', templatecolour='r',
                            size=(10.5, 7.5), cccsum=None, thresh=None):
    """
    Modified version of det_multiplot from eqcorrscan CJH

    Plot a stream of data with a template on top of it at detection times.

    :type stream: obspy.core.stream.Stream
    :param stream: Stream of data to be plotted as the background.
    :type template: obspy.core.stream.Stream
    :param template: Template to be plotted on top of the base stream.
    :type times: list
    :param times: list of detection times, one for each event
    :type events: list
    :param events: List of events corresponding to times
    :type streamcolour: str
    :param streamcolour: String of matplotlib colour types for the stream
    :type templatecolour: str
    :param templatecolour: Colour to plot the template in.
    :type size: tuple
    :param size: Figure size.
    :type cccsum: np.ndarray
    :param cccsum: Optionally provide cccsum as obspy stream to plot below

    :returns: :class:`matplotlib.figure.Figure`

    """
    import matplotlib.pyplot as plt
    plt.style.use('default')
    # Only take traces that match in both accounting for streams shorter than
    # templates
    template_stachans = [(tr.stats.station, tr.stats.channel)
                         for tr in template]
    stream_stachans = [(tr.stats.station, tr.stats.channel)
                       for tr in stream]
    temp = Stream([tr for tr in template
                   if (tr.stats.station,
                       tr.stats.channel) in stream_stachans])
    st = Stream([tr for tr in stream
                 if (tr.stats.station,
                     tr.stats.channel) in template_stachans])
    ntraces = len(temp)
    if cccsum:
        ntraces += 1
    fig, axes = plt.subplots(ntraces, 1, sharex=True, figsize=size)
    if len(temp) > 1:
        axes = axes.ravel()
    mintime = min([tr.stats.starttime for tr in temp])
    temp.sort(keys=['starttime'])
    for i, template_tr in enumerate(temp):
        if len(axes) > 1:
            axis = axes[i]
        else:
            axis = axes
        image = st.select(station=template_tr.stats.station,
                          channel='*' + template_tr.stats.channel[-1])
        if not image:
            msg = ' '.join(['No data for', template_tr.stats.station,
                            template_tr.stats.channel])
            print(msg)
            continue
        image = image.merge()[0]
        # Downsample if needed
        if image.stats.sampling_rate > 20 and image.stats.npts > 10000:
            image.decimate(int(image.stats.sampling_rate // 20))
            template_tr.decimate(int(template_tr.stats.sampling_rate // 20))
        # Get a list of datetime objects
        image_times = [image.stats.starttime.datetime +
                       timedelta((j * image.stats.delta) / 86400)
                       for j in range(len(image.data))]
        axis.plot(image_times, image.data / max(image.data),
                  streamcolour, linewidth=1.)
        for j, time in enumerate(times):
            lagged_time = UTCDateTime(time) + (template_tr.stats.starttime -
                                               mintime)
            lagged_time = lagged_time.datetime
            template_times = [lagged_time +
                              timedelta((j * template_tr.stats.delta) /
                                           86400)
                              for j in range(len(template_tr.data))]
            # Normalize the template according to the data detected in
            try:
                normalizer = max(image.data[int((template_times[0] -
                                                image_times[0]).
                                                total_seconds() /
                                                image.stats.delta):
                                            int((template_times[-1] -
                                                 image_times[0]).
                                                total_seconds() /
                                                image.stats.delta)] /
                                 max(image.data))
            except ValueError:
                # Occurs when there is no data in the image at this time...
                normalizer = max(image.data)
            normalizer /= max(template_tr.data)
            axis.plot(template_times,
                      template_tr.data * normalizer,
                      templatecolour, linewidth=1.)
        ylab = '.'.join([template_tr.stats.station,
                         template_tr.stats.channel])
        axis.set_ylabel(ylab, rotation=0, fontsize=14,
                        horizontalalignment='right',
                        verticalalignment='center')
        if events:
            ev = events[j]
            try:
                pk = [pk for pk in ev.picks if pk.waveform_id.station_code ==
                      template_tr.stats.station and pk.phase_hint == 'P'][0]
            except IndexError:
                print('No corresponding pick in event?')
            cc_text = 'CC={}'.format(pk.comments[-1].text.split('=')[-1])
            axis.text(0.9, 0.2, cc_text[:8], fontsize=14,
                      transform=axis.transAxes, verticalalignment='center',
                      bbox=dict(ec='k', fc='w'))
        axis.set_yticklabels([])
        axis.axis('on')
    if cccsum:
        ccc_times = [cccsum[0].stats.starttime.datetime +
                     timedelta((j * cccsum[0].stats.delta) / 86400)
                     for j in range(len(cccsum[0].data))]
        axes[-1].plot(ccc_times, cccsum[0].data, color='steelblue', linewidth=1.0)
        axes[-1].set_ylabel('Network CC Sum', fontsize=14)
        axes[-1].axhline(thresh, linestyle='--', color='r', linewidth=1.0,
                         label='Threshold')
        axes[-1].axis('on')
        axes[-1].legend()
    axes[len(axes) - 1].set_xlabel('Time', fontsize=14)
    if title:
        axes[0].set_title(title, fontsize=18)
    plt.subplots_adjust(hspace=0, left=0.175, right=0.95, bottom=0.07)
    plt.tight_layout()
    plt.margins(x=0.0)
    # plt.xticks(rotation=10)
    return axes

def simple_snr(noise, signal):
    """
    Simple ratio of variances SNR calculation provided a noise stream and
    a signal stream.

    :param noise: Obspy.Stream of just noise
    :param signal: Obspy.Stream of just signal
    :return:
    """
    snrs_var = []
    snrs_avg = []
    for tr in noise:
        var_n = np.sum(tr.data**2) / tr.stats.npts
        avg_n = np.sum(np.abs(tr.data)) / tr.stats.npts
        print(var_n)
        tr_sig = signal.select(station=tr.stats.station,
                               channel=tr.stats.channel)
        if len(tr_sig) == 1:
            var_sig = np.sum(tr_sig[0].data**2) / tr_sig[0].stats.npts
            avg_sig = np.sum(np.abs(tr_sig[0].data)) / tr_sig[0].stats.npts
            print(var_sig)
            snrs_var.append((var_sig - var_n) / var_n)
            snrs_avg.append(avg_sig / avg_n)
    return np.mean(snrs_var), np.mean(snrs_avg)

def plot_detection_wavs(family, tribe, wav_dirs, start=None, end=None,
                        save=False, save_dir=None, no_dets=5):
    """
    Wrapper on detection_multiplot() for our dataset
    :param cat: catalog of detections
    :param temp_dir: template waveform dict
    :param det_dir: detection waveform dict
    :return: matplotlib.pyplot.Figure
    """

    # Random range of dates in detections
    rand_inds = np.random.choice(range(len(family)), no_dets, replace=False)
    cat = Catalog(events=[det.event for i, det in enumerate(family)
                          if i in rand_inds])
    # Always plot self_detection
    cat += [det.event for det in family
            if det.detect_val / det.no_chans == 1.0][0]
    cat.events.sort(key=lambda x: x.picks[0].time)
    sub_fam = Family(template=family.template, detections=[det for i, det in
                                                           enumerate(family)
                                                           if i in rand_inds])
    sub_fam.detections.extend([det for det in family
                               if det.detect_val / det.no_chans == 1.0])
    temp = tribe[sub_fam.template.name]
    if start:
        cat_start = datetime.strptime(start, '%d/%m/%Y')
        cat_end = datetime.strptime(end, '%d/%m/%Y')
    else:
        cat_start = cat[0].picks[0].time.date
        cat_end = cat[-1].picks[0].time.date
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        dets = [det for det in sub_fam if dto
                < det.detect_time < dto + 86400]
        if len(dets) == 0:
            print('No detections on: {!s}'.format(dto))
            continue
        print('Running for date: %s' % str(dto))
        stachans = {}
        for det in dets:
            ev = det.event
            for pk in ev.picks:
                sta = pk.waveform_id.station_code
                chan = pk.waveform_id.channel_code
                if sta not in stachans:
                    stachans[sta] = [chan]
                elif chan not in stachans[sta]:
                    stachans[sta].append(chan)
        # Grab day's wav files
        wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
        stream = grab_day_wavs(wav_ds, dto, stachans)
        print('Preprocessing')
        st1 = pre_processing.dayproc(stream, temp.lowcut, temp.highcut,
                                        temp.filt_order, temp.samp_rate,
                                        starttime=dto, num_cores=3)
        for det in dets:
            det_st = st1.slice(starttime=det.detect_time - 3,
                               endtime=det.detect_time + 7).copy()
            fname = '{}/{}.png'.format(
                save_dir,
                str(det.event.resource_id).split('/')[-1])
            det_t = 'Template {}: {}'.format(temp.name, det.detect_time)
            detection_multiplot(det_st, temp.st, [det.detect_time],
                                save=save, savefile=fname, title=det_t)
            plt.close('all')
    return


def bounding_box(cat, bbox, depth_thresh):
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

def plot_two_cat_loc_changes(cat1, cat2, show=True):
    """
    Compare location changes between two catalogs

    :param cat1: obspy Catalog
    :param cat2: obspy Catalog
    :return:
    """
    cat1_dict = {}
    # Figz
    fig, ax = plt.subplots()
    for ev in cat1:
        cat1_dict[ev.resource_id] = ev.preferred_origin()
    for ev in cat2:
        if ev.resource_id in cat1_dict:
            loc1 = cat1_dict[ev.resource_id]
            loc2 = ev.preferred_origin()
            ax.scatter(x=(loc1.longitude, loc2.longitude),
                       y=(loc1.latitude, loc2.latitude),
                       facecolors=('none', 'red'),
                       edgecolors='red')
            ax.plot(x=(loc1.longitude, loc2.longitude),
                    y=(loc1.latitude,loc2.latitude))
    if show:
        plt.show()
    return ax

def plot_location_changes(cat, bbox, show=True):
    fig, ax = plt.subplots()
    mp = Basemap(projection='merc', lat_0=bbox[1][1] - bbox[1][0],
                 lon_0=bbox[0][1] - bbox[0][0],
                 resolution='h', llcrnrlon=bbox[0][0], llcrnrlat=bbox[1][1],
                 urcrnrlon=bbox[0][1], urcrnrlat=bbox[1][0],
                 suppress_ticks=True)
    mp.drawcoastlines()
    mp.fillcontinents(color='white')
    mp.readshapefile('/home/chet/gmt/data/NZ/taupo_river_poly', 'rivers', color='b')
    mp.readshapefile('/home/chet/gmt/data/NZ/taupo_lakes', 'lakes', color='b')
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

def plot_non_cumulative(party, dates=False, tribe_list=False):
    """
    Recreating something similar to Gabe's thesis fig. 4.9 plotting
    a party
    :param party:
    :param tribe_list:
    :return:
    """

    if dates:
        date_party = Party()
        for fam in party:
            date_party += Family(detections=[det for det in fam.detections
                                             if det.detect_time < dates[1]
                                             and det.detect_time > dates[0]],
                                 template=Template())
        party = date_party
    # Make list of list of template names for each tribe
    mult_list = [[temp.name for temp in tribe] for tribe in tribe_list]
    # Setup generator for colors as in cumulative_detections()
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'black',
                    'purple', 'darkgoldenrod', 'gray'])
    # Make color dict with key as mutliplet no
    col_dict = {i : (next(colors) if len(mult) > 1
                     else 'grey') for i, mult in enumerate(tribe_list)}
    detections = []
    for fam in party:
        detections.extend(fam.detections)
    dates = []
    template_names = []
    for detection in detections:
        if not type(detection) == Detection:
            msg = 'detection not of type: ' + \
                  'eqcorrscan.core.match_filter.Detection'
            raise IOError(msg)
        dates.append(detection.detect_time.datetime)
        template_names.append(detection.template_name)
    _dates = []
    _template_names = []
    mult_cols = []
    for template_name in sorted(set(template_names)):
        _template_names.append(template_name)
        _dates.append([date for i, date in enumerate(dates)
                       if template_names[i] == template_name])
        # Assign this template the color of its corresponding multiplet
        for i, mult in enumerate(mult_list):
            if template_name in mult:
                mult_cols.append(col_dict[i])
    dates = _dates
    template_names = _template_names
    fig, ax = plt.subplots()
    for i, (d_list, temp_name, mult_col) in enumerate(zip(dates,
                                                          template_names,
                                                          mult_cols)):
        y = np.empty(len(d_list))
        y.fill(i)
        d_list.sort()
        ax.plot(d_list, y, '--o', color=mult_col, linewidth=0.2,
                #markerfacecolor=colorsList[i - 1],
                markersize=3,
                markeredgewidth=0, markeredgecolor='k',
                label=temp_name)
    fig.autofmt_xdate()
    return ax

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

def plot_seismicity_with_dist(catalog, feedzone, dists=(200, 500, 1000),
                              dimension=3, shells=False, normalized=False,
                              ax=None, show=False, title=None):
    """
    Plot cumulative detection curves for given distances from a feedzone

    :param catalog: Catalog of seismicity
    :param feedzone: (lat, lon, depth (km)) for feedzone in question
    :param dists: Iterable of distances from feedzone
    :param shells: Plot distances as hollow, cylindrical bins (shells)
    :param normalized: Normalize the curves?
    :param ax: Plot onto a preexisting axes
    :param show: Show the plot?
    :param title: Title for plot?

    .. note: RK24 Feedzones: (-38.6149, 176.2025, 2.9)
             RK23 Feedzones: (-38.6162, 176.2076, 2.9)
    :return:
    """
    if not ax:
        fig, ax1 = plt.subplots()
    else:
        ax1 = ax
    dist_dates = []
    dist_names = []
    min_dates = []
    max_dates = []
    for i, dist in enumerate(dists):
        if shells and i == 0:
            # Events between well and first dist
            if dimension == 3:
                dist_cat = [ev for ev in catalog
                            if dist_calc(
                        (ev.preferred_origin().latitude,
                         ev.preferred_origin().longitude,
                         ev.preferred_origin().depth / 1000.),
                        feedzone) * 1000. < dist]
            elif dimension == 2:
                dist_cat = [ev for ev in catalog
                            if dist_calc(
                        (ev.preferred_origin().latitude,
                         ev.preferred_origin().longitude, 0),
                        (feedzone[0], feedzone[1], 0)) * 1000. < dist]
        elif shells:
            # Events between dist and previous dist
            if dimension == 3:
                dist_cat = [ev for ev in catalog
                            if dists[i - 1] <
                            dist_calc(
                        (ev.preferred_origin().latitude,
                         ev.preferred_origin().longitude,
                         ev.preferred_origin().depth / 1000.),
                        feedzone) * 1000. < dist]
            elif dimension == 2:
                dist_cat = [ev for ev in catalog
                            if dists[i - 1] < dist_calc(
                        (ev.preferred_origin().latitude,
                         ev.preferred_origin().longitude, 0),
                        (feedzone[0], feedzone[1], 0)) * 1000. < dist]
        else:
            # Make catalog within dist of feedzone
            if dimension == 3:
                dist_cat = [ev for ev in catalog
                            if dist_calc(
                        (ev.preferred_origin().latitude,
                         ev.preferred_origin().longitude,
                         ev.preferred_origin().depth / 1000.),
                        feedzone) * 1000. < dist]
            elif dimension == 2:
                dist_cat = [ev for ev in catalog
                            if dist_calc(
                        (ev.preferred_origin().latitude,
                         ev.preferred_origin().longitude, 0),
                        (feedzone[0], feedzone[1], 0)) * 1000. < dist]
        dates = [ev.preferred_origin().time.datetime
                 for ev in dist_cat]
        dist_dates.append(dates)
        dist_names.append('Within {} m'.format(dist))
        min_dates.append(min(dates))
        max_dates.append(max(dates))
    min_dates.sort()
    max_dates.sort()
    ax1 = cumulative_detections(dates=dist_dates, template_names=dist_names,
                                show=show, axes=ax1, normalized=normalized,
                                plot_dates=[min_dates[0], max_dates[-1]],
                                title=title)
    return ax1


def plot_swarms(catalog, threshold):
    """
    Isolate events in a catalog belonging to "swarms". This term in defined
    by a threshold number of events within the specified bin size.

    :param catalog: catalog in question
    :param bin_days: Width of bin in days
    :param threshold: Threshold number of events in a bin to trigger a 'swarm'
    :return:
    """
    # Sort by time of occurrence
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    start = UTCDateTime(catalog[0].preferred_origin().time.date).datetime
    elapsed_days = np.array([(ev.preferred_origin().time.datetime - start).days
                             for ev in catalog])
    print(elapsed_days)
    bins = np.arange(0, elapsed_days[-1])
    digitized = np.digitize(elapsed_days, bins)
    which_days = [i - 1 for i in range(1, len(bins))
                  if elapsed_days[digitized == i].shape[0] > threshold]
    print(which_days)
    swarm_cat = Catalog()
    for day in which_days:
        day_start = UTCDateTime(start) + (day * 86400)
        print(day_start)
        swarm_cat.events.extend([
            ev for ev in catalog if day_start < ev.preferred_origin().time <
            day_start + 86400
        ])
    return swarm_cat


def cumulative_detections(dates=None, template_names=None, detections=None,
                          plot_grouped=False, group_name=None, rate=False,
                          show=True, plot_legend=True, axes=None, save=False,
                          savefile=None, color=None, colors=None,
                          linestyles=None, tick_colors=None, normalized=False,
                          deviation=False, plot_dates=None, title=None,
                          thresh=None, rate_bin=None):
    """
    Plot cumulative detections or detecton rate in time.

    Simple plotting function to take a list of either datetime objects or
    :class:`eqcorrscan.core.match_filter.Detection` objects and plot
    a cumulative detections list.  Can take dates as a list of lists and will
    plot each list separately, e.g. if you have dates from more than one
    template it will overlay them in different colours.

    :type dates: list
    :param dates: Must be a list of lists of datetime.datetime objects
    :type template_names: list
    :param template_names: List of the template names in order of the dates
    :type detections: list
    :param detections: List of :class:`eqcorrscan.core.match_filter.Detection`
    :type plot_grouped: bool
    :param plot_grouped: Plot detections for each template individually, or \
        group them all together - set to False (plot template detections \
        individually) by default.
    :type rate: bool
    :param rate: Whether or not to plot the rate of detection per day. Only
        works for plot_grouped=True
    :type show: bool
    :param show: Whether or not to show the plot, defaults to True.
    :type plot_legend: bool
    :param plot_legend: Specify whether to plot legend of template names. \
        Defaults to True.
    :type axes: matplotlib.Axes
    :param axes: Axes object on which to plot cumulative detections
    :type save: bool
    :param save: Save figure or show to screen, optional
    :type savefile: str
    :param savefile: String to save to, required is save=True
    :param color: Define a color for a single line, will be red otherwise
    :type color: Str or None
    :param colors: Custom cycle of colors to be used
    :type colors: itertools.Cycle
    :param linestyles: Provide list of linestyles to cycle through
    :type linestyles: itertools.Cycle
    :param tick_colors: Whether to color axis ticks same as curve
    :type tick_colors: bool
    :param normalized: Whether to normalize the curves or leave absolute vals
    :type normalized: bool
    :param deviation: Plot the deviation from the average daily rate?
    :type deviation: bool
    :param plot_dates: List of datetime objects defining the start and end
        of the plot
    :type plot_dates: list
    :param title: Title for plot
    :type title: str or None
    :param thresh: Threshold of any kind to by plotted horizontally
    :type thresh: int or float
    :param rate_bin: int
    :type rate_bin: Number of days per bin in rate plotting

    :returns: :class:`matplotlib.figure.Figure`

    """
    from eqcorrscan.core.match_filter import Detection
    # Set up a default series of parameters for lines
    if not colors:
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black',
                  'firebrick', 'purple', 'darkgoldenrod', 'gray']
    cols = cycle(colors)
    if not linestyles:
        custom = False # Flag to cycle linestyle for each set of dates
        linestyles = ['-', '-.', '--', ':']
    else:
        custom = True
    lins = cycle(linestyles)
    # Check that dates is a list of lists
    if not detections:
        if type(dates[0]) != list:
            dates = [dates]
    else:
        dates = []
        template_names = []
        for detection in detections:
            if not type(detection) == Detection:
                msg = 'detection not of type: ' +\
                    'eqcorrscan.core.match_filter.Detection'
                raise IOError(msg)
            dates.append(detection.detect_time.datetime)
            template_names.append(detection.template_name)
        _dates = []
        _template_names = []
        for template_name in list(set(template_names)):
            _template_names.append(template_name)
            _dates.append([date for i, date in enumerate(dates)
                           if template_names[i] == template_name])
        dates = _dates
        template_names = _template_names
    if plot_grouped:
        _dates = []
        for template_dates in dates:
            _dates += template_dates
        dates = [_dates]
        if group_name:
            template_names = group_name
        else:
            template_names = ['All templates']
    if deviation:
        print('Not implemented yet')
        return
    if not axes:
        ax = plt.gca()
        if plot_dates:
            xlims = (plot_dates[0], plot_dates[1])
        else:
            xlims = None
    else:
        if axes.get_ylim()[-1] == 1.0:
            ax = axes
        else:
            ax = axes.twinx()
            try:
                # Grab these lines for legend
                handles, leg_labels = axes.get_legend_handles_labels()
                if isinstance(axes.legend_, matplotlib.legend.Legend):
                    axes.legend_.remove()  # Need to manually remove this, apparently
            except AttributeError:
                print('Empty axes. No legend to incorporate.')
        if not plot_dates and ax.get_xlim()[0] == 0:
            xlims = None
        elif not plot_dates:
            xlims = ax.get_xlim()
        else:
            xlims = (plot_dates[0], plot_dates[1])
    # Make sure not to pad at edges
    ax.margins(0, 0)
    min_date = min([min(_d) for _d in dates])
    max_date = max([max(_d) for _d in dates])
    for k, template_dates in enumerate(dates):
        template_dates.sort()
        final_dates = deepcopy(template_dates)
        # Account for step plot stopping
        color = next(cols)
        if color == colors[0]:
            linestyle = next(lins)
        elif custom == True:
            linestyle = next(lins)
        counts = np.arange(0, len(template_dates))
        if normalized:
            counts = [cnt / float(max(counts)) for cnt in counts]
        if rate:
            days = (max_date - min_date).days
            if rate_bin:
                bins = days // rate_bin
                ax.set_ylabel('Events / {} days'.format(rate_bin), fontsize=16)
            elif 31 < days < 365:
                bins = days
                ax.set_ylabel('Events / day', fontsize=16)
            elif days <= 31:
                bins = days * 4
                ax.set_ylabel('Events / 6 hour bin', fontsize=16)
            else:
                bins = days // 7
                ax.set_ylabel('Events / week', fontsize=16)
            ax.hist(mdates.date2num(final_dates), bins=bins,
                    label=template_names[k], color=color)
            if thresh:
                ax.axhline(thresh, linestyle='--', color='r',
                           linewidth=1.5)
        else:
            ax.step(final_dates, counts, linestyle=linestyle,
                    color=color, label=template_names[k],
                    linewidth=1.5, where='post',
                    zorder=1)
            if normalized:
                ax.set_ylabel('Normalized # of Events', fontsize=14)
            else:
                ax.set_ylabel('# of Events', fontsize=16)
    ax.set_xlabel('Date', fontsize=16)
    # Set formatters for x-labels
    mins = mdates.MinuteLocator()
    max_date = dates[0][0]
    min_date = max_date
    for date_list in dates:
        if max(date_list) > max_date:
            max_date = max(date_list)
        if min(date_list) < min_date:
            min_date = min(date_list)
    timedif = max_date - min_date
    if 10800 <= timedif.total_seconds() <= 25200:
        hours = mdates.MinuteLocator(byminute=[0, 30])
        mins = mdates.MinuteLocator(byminute=np.arange(0, 60, 10))
    elif 7200 <= timedif.total_seconds() < 10800:
        hours = mdates.MinuteLocator(byminute=[0, 15, 30, 45])
        mins = mdates.MinuteLocator(byminute=np.arange(0, 60, 5))
    elif timedif.total_seconds() <= 1200:
        hours = mdates.MinuteLocator(byminute=np.arange(0, 60, 2))
        mins = mdates.MinuteLocator(byminute=np.arange(0, 60, 0.5))
    elif 25200 < timedif.total_seconds() <= 86400:
        hours = mdates.HourLocator(byhour=np.arange(0, 24, 3))
        mins = mdates.HourLocator(byhour=np.arange(0, 24, 1))
    elif 86400 < timedif.total_seconds() <= 172800:
        hours = mdates.HourLocator(byhour=np.arange(0, 24, 6))
        mins = mdates.HourLocator(byhour=np.arange(0, 24, 1))
    elif timedif.total_seconds() > 172800:
        hours = mdates.AutoDateLocator()
        mins = mdates.HourLocator(byhour=np.arange(0, 24, 3))
    else:
        hours = mdates.MinuteLocator(byminute=np.arange(0, 60, 5))
    # Minor locator overruns maxticks for ~year-long datasets
    if timedif.total_seconds() < 172800:
        ax.xaxis.set_minor_locator(mins)
        hrFMT = mdates.DateFormatter('%Y/%m/%d %H:%M:%S')
    else:
        hrFMT = mdates.DateFormatter('%Y/%m/%d')
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(hrFMT)
    plt.gcf().autofmt_xdate()
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=15)
    if tick_colors:
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis='y', colors=color)
    if not rate and not normalized:
        ax.set_ylim([0, max([len(_d) for _d in dates]) * 1.1])
    elif not rate and normalized:
        ax.set_ylim([0, 1.2])
    if plot_legend:
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
                               fontsize=12, loc=2, scatterpoints=10)
                else:
                    ax.legend(handles=handles, labels=leg_labels, loc=2,
                              scatterpoints=10, fontsize=12)
            except UnboundLocalError:
                print('Plotting on empty axes. No handles to add to.')
                ax.legend(fontsize=10)
        elif ax.legend() is not None:
            leg = ax.legend(loc=2, prop={'size': 8}, ncol=2, fontsize=12)
            leg.get_frame().set_alpha(0.5)
    if xlims:
        ax.set_xlim(xlims)
    if title:
        ax.set_title(title, fontsize=16)
    if save:
        plt.gcf().savefig(savefile)
        plt.close()
    else:
        if show:
            plt.show()
    return ax

##### OTHER MISC FUNCTIONS #####

def simple_pick_plot(event, stream):
    """
    Plot picks from event over streams
    :return:
    """
    fig = plt.figure()
    pk_stachs = defaultdict(list)
    wav_stachs = defaultdict(list)
    pk_codes = [(pk.waveform_id.station_code,
                 pk.waveform_id.channel_code) for pk in event.picks]
    wav_codes = [(tr.stats.station, tr.stats.channel) for tr in stream]
    for sta, chan in wav_codes:
        wav_stachs[sta].append(chan)
    for sta, chan in pk_codes:
        pk_stachs[sta].append(chan)
    pk_stream = Stream(traces=[tr for tr in stream
                               if tr.stats.station in pk_stachs
                               and tr.stats.channel in
                               pk_stachs[tr.stats.station]])
    i = 0
    axes = []
    for pk in event.picks:
        sta = pk.waveform_id.station_code
        chan = pk.waveform_id.channel_code
        if sta not in wav_stachs and chan not in wav_stachs[sta]:
            print('No wav for {}.{}'.format(sta, chan))
            continue
        tr = stream.select(station=pk.waveform_id.station_code,
                           channel=pk.waveform_id.channel_code)[0]
        if i == 0:
            ax = fig.add_subplot(len(pk_stream), 1, i + 1)
            axes.append(ax)
            i += 1
        else:
            ax = fig.add_subplot(len(pk_stream), 1, i + 1, sharex=axes[-1])
            i += 1
        x = ((tr.times() / 86400.) + date2num(tr.stats.starttime.datetime))
        ax.plot(x, tr.data, color='k', linewidth=1,
                label='{}.{}'.format(sta, chan))
        ax.legend()
        ax.axvline(date2num(pk.time.datetime), color='red')
    plt.show()
    plt.close('all')
    return

def cat_to_elapsed_days(cat, outfile, mag_multiplier):
    """
    Take catalog and convert the dates to days since start of catalog.

    :param csv_file: File path
    :return:
    """
    from obspy import UTCDateTime

    next_rows = []
    dates = []
    mags = []
    for ev in cat:
        o = ev.preferred_origin()
        try:
            m = ev.magnitudes[-1].mag
            mags.append(m)
        except IndexError:
            print('No magnitude for: {}'.format(ev))
            continue
        dates.append(o.time)
        next_rows.append([o.longitude, o.latitude, o.depth / 1000., m])
    start_date = min(dates)
    mm = max(mags) * mag_multiplier
    with open(outfile, 'w') as nf:
        for date, new_r in zip(dates, next_rows):
            if not new_r[3]: # catch rare NoneType magnitude
                continue
            elapsed = (date.datetime - start_date.datetime).days
            nf.write('{} {} {} {} {}\n'.format(new_r[0], new_r[1], new_r[2],
                                               str(elapsed),
                                               str(float(new_r[3]) / mm)))
    return

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
        leg = ax1.legend()
        leg.get_frame().set_alpha(0.5)
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
        leg = ax.legend((barsP[0], barsS[0]), ('P picks', 'S-picks'))
        leg.get_frame().set_alpha(0.5)
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

    param_list = [ev_dict[param] for rid, ev_dict in diff_dict.iteritems()]
    if param2:
        param_list2 = [ev_dict[param2] for rid, ev_dict in diff_dict.iteritems()]
    ax = sns.distplot(param_list)
    if param2:
        ax = sns.regplot(np.array(param_list), np.array(param_list2))
    if show:
        plt.gcf().show()
    return ax

def bbox_two_cat(cat1, cat2, bbox, depth_thresh):
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
    comm_cat_S = Catalog()
    comm_cat_P = Catalog()
    for i, ev in enumerate(catS):
        if len([pk for pk in ev.picks if pk.phase_hint == 'S']) > 0:
            comm_cat_S.events.append(ev)
            comm_cat_P.events.append(catP[i])
    return comm_cat_P, comm_cat_S

def plot_hypoDD_log(hypoDDpy_dir, save=False):
    # Load log file and make utf-8 text file
    lines = []
    with open('{}/output_files/hypoDD_log.txt'.format(hypoDDpy_dir),
              'rb') as f_in:
        for ln in f_in:
            try:
                lines.append(ln.decode('utf-8'))
            except UnicodeDecodeError:
                continue
    # Create dictionary for each inversion for each cluster
    clusters = {}
    index = 0
    cc_ct, cc_only, ct_only = False, False, False
    for line in lines:
        index += 1
        if line.startswith('RELOCATION OF CLUSTER:'):
            print(line)
            clust_num = int(line.split('RELOCATION OF CLUSTER:')[1][0:4])
            clusters[clust_num] = {}
        if line.startswith('  IT   EV  CT  CC'):
            cc_ct = True
            line_spl = lines[index + 1][5:].split()
            it_num = int(lines[index + 1][:5].split()[0])
            clusters[clust_num][it_num] = {'ev%': int(line_spl[0]),
                                           'ct%': int(line_spl[1]),
                                           'cc%': int(line_spl[2]),
                                           'rmsct': int(line_spl[3]),
                                           'rmscc': int(line_spl[5]),
                                           'rmsst': int(line_spl[7]),
                                           'dx': int(line_spl[8]),
                                           'dy': int(line_spl[9]),
                                           'dz': int(line_spl[10]),
                                           'os': int(line_spl[12]),
                                           'aq': int(line_spl[13]),
                                           'cnd': int(line_spl[14])}
        elif line.startswith('  IT   EV  CT'):
            ct_only = True
            line_spl = lines[index + 1][5:].split()
            it_num = int(lines[index + 1][:5].split()[0])
            clusters[clust_num][it_num] = {'ev%': int(line_spl[0]),
                                           'ct%': int(line_spl[1]),
                                           'rmsct': int(line_spl[2]),
                                           'rmsst': int(line_spl[4]),
                                           'dx': int(line_spl[5]),
                                           'dy': int(line_spl[6]),
                                           'dz': int(line_spl[7]),
                                           'os': int(line_spl[9]),
                                           'aq': int(line_spl[10]),
                                           'cnd': int(line_spl[11])}
        elif line.startswith('  IT   EV  CC'):
            cc_only = True
            line_spl = lines[index + 1][5:].split()
            it_num = int(lines[index + 1][:5].split()[0])
            clusters[clust_num][it_num] = {'ev%': int(line_spl[0]),
                                           'cc%': int(line_spl[1]),
                                           'rmscc': int(line_spl[2]),
                                           'rmsst': int(line_spl[4]),
                                           'dx': int(line_spl[5]),
                                           'dy': int(line_spl[6]),
                                           'dz': int(line_spl[7]),
                                           'os': int(line_spl[9]),
                                           'aq': int(line_spl[10]),
                                           'cnd': int(line_spl[11])}
    print(clusters)
    # Inversion summary plot
    for clust_num, iter_dict in clusters.items():
        # create lists for plotting
        ev, ct, cc, rmsct, rmscc, rmsst, dx, dy, dz, osh, aq, cnd = (
        [], [], [], [], [], [],
        [], [], [], [], [], [])
        for ind, params in iter_dict.items():
            ev.append(params['ev%'])
            if cc_ct:
                ct.append(params['ct%'])
                cc.append(params['cc%'])
                rmsct.append(params['rmsct'])
                rmscc.append(params['rmscc'])
            elif ct_only:
                ct.append(params['ct%'])
                rmsct.append(params['rmsct'])
            elif cc_only:
                cc.append(params['cc%'])
                rmscc.append(params['rmscc'])
            rmsst.append(params['rmsst'])
            dx.append(params['dx'])
            dy.append(params['dy'])
            dz.append(params['dz'])
            osh.append(params['os'])
            aq.append(params['aq'])
            cnd.append(params['cnd'])
        iters = np.arange(1, len(ev) + 1, 1)
        # Event, ct and cc percentage (of initial number of data of each type)
        # and number of air quakes
        ax1 = plt.subplot(611)
        ax1.plot(iters, ev, marker='o', label='Event %')
        if cc_ct:
            ax1.plot(iters, ct, marker='o', label='Catalog %')
            ax1.plot(iters, cc, marker='o', label='Cross-corr %')
        elif ct_only:
            ax1.plot(iters, ct, marker='o', label='Catalog %')
        elif cc_only:
            ax1.plot(iters, cc, marker='o', label='Cross-corr %')
        ax1_2 = ax1.twinx()
        ax1_2.plot(iters, aq, marker='x', label='Air-quakes')
        ax1.set_ylabel('Percent')
        ax1.legend()
        ax1_2.set_ylabel('Count')
        ax1_2.legend()
        ax1.set_title(
            'HypoDD Inversion Stats for Cluster ' + str(clust_num) + '\n',
            horizontalalignment='center', verticalalignment='center',
            fontsize=10)
        # RMS residual for catalog and cross-corr (RMS CT and CC)
        ax2 = plt.subplot(612)
        if cc_ct:
            ax2.plot(iters, rmsct, marker='o', label='Catalog RMS')
            ax2_1 = ax2.twinx()
            ax2_1.plot(iters, rmscc, marker='x', color='g',
                       label='Cross-corr RMS')
            ax2_1.set_ylabel('RMS residual (ms)')
            ax2_1.legend(bbox_to_anchor=(0.2, 0.2))
        elif ct_only:
            ax2.plot(iters, rmsct, marker='o', label='Catalog RMS')
        elif cc_only:
            ax2.plot(iters, rmscc, marker='o', label='Cross-corr RMS')
        ax2.set_ylabel('RMS residual (ms)')
        ax2.legend()
        # Largest rms residual at station (RMS ST)
        ax3 = plt.subplot(613)
        ax3.plot(iters, rmsst, marker='o', label='Largest Stat RMS')
        ax3.set_ylabel('RMS residual (ms)')
        ax3.legend()
        # Average absolute of change in hypocentre (DX, DY, DZ)
        ax4 = plt.subplot(614)
        ax4.plot(iters, dx, marker='o', label='Hypo Shift DX')
        ax4.plot(iters, dy, marker='o', label='Hypo Shift DY')
        ax4.plot(iters, dz, marker='o', label='Hypo Shift DZ')
        ax4.set_ylabel('Hypo Shift (m)')
        ax4.legend()
        # Origin shift of each cluster (OS)
        ax5 = plt.subplot(615)
        ax5.plot(iters, osh, marker='o', label='Ave Cluster Shift')
        ax5.set_ylabel('Cluster Shift (m)')
        # Condition number
        ax6 = plt.subplot(616)
        ax6.plot(iters, cnd, marker='o', label='Condition Number')
        ax6.set_ylabel('Condition Number')
        ax6.set_xlabel('Iteration Number')
        # get figure
        fig = plt.gcf()
        fig.set_size_inches(8.27, 11.69)
        fig.tight_layout()
        if save:
            fig.savefig('{}/log_plot.pdf'.format(hypoDDpy_dir), format='PDF')
        else:
            plt.show()
            plt.close()
    return