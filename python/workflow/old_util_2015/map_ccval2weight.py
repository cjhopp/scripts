#!/usr/bin/env python

"""
Script with functions to plot picks over waveforms and map those to weights
for purposes of locating events.
"""

import sys
sys.path.insert(0, '/home/chet/EQcorrscan')
import matplotlib.pyplot as plt
import numpy as np


def daterange(start_date, end_date):
    # Generator function for looping over dates
    from datetime import timedelta
    for n in range(int((end_date.datetime - start_date.datetime).days)):
        yield start_date + timedelta(n)


def make_temp_dict(temp_dir):
    from obspy import read
    from obspy.core.event import ResourceIdentifier
    from glob import glob
    # Make underlying template directory
    temp_files = glob(temp_dir)
    template_dict = {}
    for filename in temp_files:
        uri_name = 'smi:local/' + \
                   filename.split('/')[-1].rstrip('.mseed')
        uri = ResourceIdentifier(uri_name)
        template_dict[uri] = read(filename)
    return template_dict


def reweight_picks(cat):
    """
    Function to change pick uncertainties based upon correlation values (saved in pick Comment).
    This works in-place on the catalog.

    :type cat: obspy.core.Catalog
    :param cat: catalog of events with ccvals against detecting template saved in Comment
    :return: obspy.core.Catalog
    """
    from obspy.core.event import QuantityError
    for ev in cat:
        for pk in ev.picks:
            if pk.phase_hint == 'P':
                ccval = float(pk.comments[0].text.split('=')[-1])
                # Re-weight based on some scheme (less down-weighting)
                if ccval > 0.3:
                    pk.time_errors = QuantityError(uncertainty=0.05)
    return cat


def rand_cat_sample(cat, n_events, cat2=False):
    from obspy import Catalog
    rand_cat = Catalog()
    indices = np.random.choice(range(len(cat)), n_events, replace=False)
    rand_cat.events = [cat[i] for i in indices]
    if cat2:
        rand_cat2 = Catalog()
        rand_cat2.events = [cat[i] for i in indices]
    return rand_cat


def simple_pick_plot(cat, n_events, template_dict, st_dict, pyasdf=None, savefiles=False):
    """
    Function to plot a random sample from a catalog with picks
    :type: template_list: str
    :param template_list: directory containing all template waveforms
    :type: cat: obspy.core.Catalog
    :param cat: catalog of events of interest
    :type: n_events: int
    :param: n_events: number of random events to select from cat and plot
    :type: pyasdf: str
    :param: pyasdf:
    :return: matplotlib.pyplot.figure
    """
    from obspy import Catalog, UTCDateTime, Stream
    from obspy.core.event import ResourceIdentifier
    if n_events == 'all':
        rand_cat = cat
    else:
        rand_cat = rand_cat_sample(cat, n_events)
    # Make a list of year + julday integers to loop over
    min_date = min([ev.preferred_origin().time for ev in rand_cat])
    max_date = max([ev.preferred_origin().time for ev in rand_cat])
    for date in daterange(min_date, max_date):
        day_cat = rand_cat.filter("time >= " + str(UTCDateTime(date)),
                                  "time <= " + str(UTCDateTime(date) + 86400))
        if len(day_cat) == 0:
            continue
        stachans = {pk.waveform_id.station_code: [] for ev in day_cat for pk in ev.picks}
        for ev in day_cat:
            for pick in ev.picks:
                if pick.waveform_id.channel_code not in stachans[pick.waveform_id.station_code]:
                    stachans[pick.waveform_id.station_code].append(pick.waveform_id.channel_code)
        print(stachans)
        # Read the waveforms for this day
        if pyasdf:
            st = Stream()
            with pyasdf.ASDFDataSet(pyasdf) as ds:
                for sta in stachans:
                    for station in ds.ifilter(ds.q.station == str(sta),
                                              ds.q.channel == stachans[sta],
                                              ds.q.starttime >= UTCDateTime(date),
                                              ds.q.endtime <= UTCDateTime(date) + 86400):
                        st += station.raw_recording
        for ev in day_cat:
            det_st = st_dict[ev.resource_id]
            det_temp = template_dict[ResourceIdentifier('smi:local/' +
                                                 str(ev.resource_id).split('/')[-1].split('_')[0] +
                                                 '_1sec')]
            fig = plot_repicked(det_temp, ev.picks, det_st, size=(21, 15), save=savefiles,
                                savefile=str(ev.resource_id).split('/')[-1] + '.png',
                                title=str(ev.resource_id).split('/')[-1])


def plot_repicked(template, picks, det_stream, size=(10.5, 7.5), save=False,
                  savefile=None, title=False):
    """
    Plot a template over a detected stream, with picks corrected by lag-calc.

    :param template: Template used to make the detection, will be aligned \
        according to picks.
    :type template: obspy.core.stream.Stream
    :param picks: list of corrected picks.
    :type picks: list
    :param det_stream: Stream to plot in the background, should be the \
        detection, data should encompass the time the picks are made.
    :type det_stream: obspy.core.stream.Stream
    :param size: tuple of plot size.
    :type size: tuple
    :param save: To save figure or not, if false, will show to screen.
    :type save: bool
    :param savefile: File name to save file, required if save==True.
    :type savefile: str
    :param title: Title for plot, defaults to None.
    :type title: str

    :return: Figure handle which can be edited.
    :rtype: matplotlib.pyplot.figure
    """
    # _check_save_args(save, savefile)
    fig, axes = plt.subplots(len(det_stream), 1, sharex=True, figsize=size)
    if len(template) > 1:
        axes = axes.ravel()
    mintime = det_stream.sort(['starttime'])[0].stats.starttime
    template.sort(['network', 'station', 'starttime'])
    lengths = []
    lines = []
    labels = []
    n_templates_plotted = 0
    for i, tr in enumerate(det_stream.sort(['starttime'])):
        # Cope with a single channel template case.
        if len(det_stream) > 1:
            axis = axes[i]
        else:
            axis = axes
        tr_picks = [pick for pick in picks if
                    pick.waveform_id.station_code == tr.stats.station and
                    pick.waveform_id.channel_code[0] +
                    pick.waveform_id.channel_code[-1] ==
                    tr.stats.channel[0] + tr.stats.channel[-1]]
        if len(tr_picks) > 1:
            msg = 'Multiple picks on channel %s' % tr.stats.station + ', ' + \
                  tr.stats.channel
            raise NotImplementedError(msg)
        if len(tr_picks) == 0:
            msg = 'No pick for chanel %s' % tr.stats.station + ', ' + \
                  tr.stats.channel
            print(msg)
        else:
            pick = tr_picks[0]
            pick_delay = pick.time - mintime
            delay = tr.stats.starttime - mintime
            y = tr.data
            # Normalise
            if len(tr_picks) > 0 and template:
                y /= max(abs(y[int(pick_delay/tr.stats.delta):int(pick_delay/tr.stats.delta) + len(template[0])]))
            else:
                y /= max(abs(y))
            x = np.linspace(0, (len(y) - 1) * tr.stats.delta, len(y))
            x += delay
            axis.plot(x, y, 'k', linewidth=1.5)
            axis.set_ylim(-max(abs(y)), max(abs(y)))
        if template.select(station=tr.stats.station, channel=tr.stats.channel):
            btr = template.select(station=tr.stats.station,
                                    channel=tr.stats.channel)[0]
            bdelay = pick.time - mintime
            by = btr.data
            by /= max(abs(by))
        bx = np.linspace(0, (len(by) - 1) * btr.stats.delta, len(by))
        bx += bdelay
        if len(tr_picks) > 0:
            # Heads up for the x - 0.1 fudge factor here accounting for template pre-pick time
            template_line, = axis.plot(bx - 0.1, by, 'r', linewidth=1.6, label='Template')
            if not pick.phase_hint:
                pcolor = 'k'
                label = 'Unknown pick'
            elif 'P' in pick.phase_hint.upper():
                pcolor = 'red'
                label = 'P-pick'
            elif 'S' in pick.phase_hint.upper():
                pcolor = 'blue'
                label = 'S-pick'
            else:
                pcolor = 'k'
                label = 'Unknown pick'
            pdelay = pick.time - mintime
            ccval = pick.comments[0].text.split('=')[-1]
            line = axis.axvline(x=pdelay, color=pcolor, linewidth=2,
                                linestyle='--', label=label)
            axis.text(pdelay, max(by), ccval, fontsize=12)
            if label not in labels:
                lines.append(line)
                labels.append(label)
            if n_templates_plotted == 0:
                lines.append(template_line)
                labels.append('Template')
            n_templates_plotted += 1
            lengths.append(max(bx[-1], x[-1]))
        else:
            lengths.append(bx[1])
        axis.set_ylabel('.'.join([tr.stats.station, tr.stats.channel]),
                        rotation=0, horizontalalignment='right')
        axis.yaxis.set_ticks([])
    if len(det_stream) > 1:
        axis = axes[len(det_stream) - 1]
    else:
        axis = axes
    axis.set_xlabel('Time (s) from %s' %
                    mintime.datetime.strftime('%Y/%m/%d %H:%M:%S.%f'))
    plt.figlegend(lines, labels, 'upper right')
    if title:
        if len(template) > 1:
            axes[0].set_title(title)
        else:
            axes.set_title(title)
    else:
        plt.subplots_adjust(top=0.98)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if not save:
        plt.show()
        plt.close()
    else:
        plt.savefig(savefile)
        plt.close()
    return fig