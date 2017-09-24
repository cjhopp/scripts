#!/usr/bin/python
from __future__ import division
from future.utils import iteritems

# import matplotlib
# matplotlib.rcParams['figure.dpi'] = 300

import csv
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from obspy import read
from scipy.signal import argrelextrema
from obspy.imaging.beachball import beach
from eqcorrscan.utils.mag_calc import dist_calc


def foc_mec_from_event(catalog, station_names=False):
    """
    Just taking Tobias' plotting function out of obspyck
    :param catalog:
    :return:
    """

    for ev in catalog:
        fms = ev.focal_mechanisms
        if not fms:
            err = "Error: No focal mechanism data!"
            raise Exception(err)
            return
        # make up the figure:
        fig, tax = plt.subplots()
        ax = fig.add_subplot(111, aspect="equal")
        axs = [ax]
        axsFocMec = axs
        ax.autoscale_view(tight=False, scalex=True, scaley=True)
        width = 2
        plot_width = width * 0.95
        # plot_width = 0.95 * width
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        # plot the selected solution
        av_np1_strike = np.mean([fm.nodal_planes.nodal_plane_1.strike
                                 for fm in fms])
        print('Strike of nodal plane 1: %f' % av_np1_strike)
        fm = sorted([fm for fm in fms], key=lambda x:
                    abs(x.nodal_planes.nodal_plane_1.strike - av_np1_strike))[0]
        np1 = fm.nodal_planes.nodal_plane_1
        if hasattr(fm, "_beachball"):
            beach_ = fm._beachball
        else:
            beach_ = beach([np1.strike, np1.dip, np1.rake],
                           width=plot_width)
            fm._beachball = beach_
        ax.add_collection(beach_)
        # plot the alternative solutions
        if not hasattr(fm, "_beachball2"):
            for fm_ in fms:
                _np1 = fm_.nodal_planes.nodal_plane_1
                beach_ = beach([_np1.strike, _np1.dip, _np1.rake],
                               nofill=True, edgecolor='k', linewidth=1.,
                               alpha=0.3, width=plot_width)
                fm_._beachball2 = beach_
        for fm_ in fms:
            ax.add_collection(fm_._beachball2)
        text = "Focal Mechanism (%i of %i)" % \
               (0 + 1, len(fms))
        text += "\nStrike: %6.2f  Dip: %6.2f  Rake: %6.2f" % \
                (np1.strike, np1.dip, np1.rake)
        if fm.misfit:
            text += "\nMisfit: %.2f" % fm.misfit
        if fm.station_polarity_count:
            text += "\nStation Polarity Count: %i" % fm.station_polarity_count
        # fig.canvas.set_window_title("Focal Mechanism (%i of %i)" % \
        #        (self.focMechCurrent + 1, len(fms)))
        fig.subplots_adjust(top=0.88)  # make room for suptitle
        # values 0.02 and 0.96 fit best over the outer edges of beachball
        # ax = fig.add_axes([0.00, 0.02, 1.00, 0.96], polar=True)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.axison = False
        axFocMecStations = fig.add_axes([0.00, 0.02, 1.00, 0.84], polar=True)
        ax = axFocMecStations
        ax.set_title(text)
        ax.set_axis_off()
        azims = []
        incis = []
        polarities = []
        bbox = dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=1.5,
                    alpha=0.7)
        for pick in ev.picks:
            if pick.phase_hint != "P":
                continue
            wid = pick.waveform_id
            net = wid.network_code
            sta = wid.station_code
            arrival = getArrivalForPick(ev.origins[-1].arrivals, pick)
            if not pick:
                continue
            if pick.polarity is None or arrival is None or arrival.azimuth is None or arrival.takeoff_angle is None:
                continue
            if pick.polarity == "positive":
                polarity = True
            elif pick.polarity == "negative":
                polarity = False
            else:
                polarity = None
            azim = arrival.azimuth
            inci = arrival.takeoff_angle
            # lower hemisphere projection
            if inci > 90:
                inci = 180. - inci
                azim = -180. + azim
            # we have to hack the azimuth because of the polar plot
            # axes orientation
            plotazim = (np.pi / 2.) - ((azim / 180.) * np.pi)
            azims.append(plotazim)
            incis.append(inci)
            polarities.append(polarity)
            if station_names:
                ax.text(plotazim, inci, "  " + sta, va="top", bbox=bbox, zorder=2)
        azims = np.array(azims)
        incis = np.array(incis)
        polarities = np.array(polarities, dtype=bool)
        ax.scatter(azims, incis, marker="o", lw=2, facecolor="w",
                   edgecolor="k", s=200, zorder=3)
        mask = (polarities == True)
        ax.scatter(azims[mask], incis[mask], marker="+", lw=3, color="k",
                   s=200, zorder=4)
        mask = ~mask
        ax.scatter(azims[mask], incis[mask], marker="_", lw=3, color="k",
                   s=200, zorder=4)
        # this fits the 90 degree incident value to the beachball edge best
        ax.set_ylim([0., 91])
        plt.draw()
        plt.show()
    return


def getArrivalForPick(arrivals, pick):
    """
    searches given arrivals for an arrival that references the given
    pick and returns it (empty Arrival object otherwise).
    """
    for a in arrivals:
        if a.pick_id == pick.resource_id:
            return a
    return None

def dec_2_merc_meters(dec_x, dec_y, z):
    """
    Conversion from decimal degrees to meters in Mercury Cartesian Grid.
    This is the same grid used for NLLoc

    **Note** This grid node has been changed from the NLLoc grid origin of
    -38.3724, 175.9577 to ensure that all coordinates are positive
    """
    origin = [-38.8224, 175.9577]
    y = (dec_y - origin[0]) * 111111
    x = (dec_x - origin[1]) * (111111 * np.cos(origin[0]*(np.pi/180)))
    return x, y, z

def write_hybridMT_input(cat, sac_dir, inv, self_files, outfile,
                         file_type='raw', plot=False):
    """
    Umbrella function to handle writing input files for focimt and hybridMT

    :param cat: Catalog of events to write files for
    :param sac_dir: Root directory for detection SAC files
    :param inv: Inventory object containing all necessary station responses
    :param selfs: List containing directory names for template self detections
    :param prefilt: List of 4 corners for preconvolution bandpass
        For details see obspy.core.trace.Trace.remove_response() docs
    :return:
    """
    selfs = []
    for self_file in self_files:
        with open(self_file, 'r') as f:
            rdr = csv.reader(f)
            for row in rdr:
                selfs.append(str(row[0]))
    ev_dict = {}
    # Build prefilt dict (Assuming all wavs have been downsampled to 100 Hz)
    pf_dict = {'MERC': [0.5, 3.5, 40., 49.],
               'GEONET': [0.2, 1.1, 40., 49.]}
    # Loop through events
    for ev in cat:
        ev_id = str(ev.resource_id).split('/')[-1]
        print('Working on {}'.format(ev_id))
        self = [self for self in selfs if self.split('_')[0] == ev_id]
        orig = ev.origins[-1]
        if len(self) == 0: # Skip those with no self detection (outside fields)
            print('No self detection for %s' % ev_id)
            continue
        wavs = glob('%s/%s/*' % (sac_dir, self[0]))
        # Loop through arrivals and populate ev_dict with TOA, Backaz, etc...
        ev_dict[ev_id] = {} # Allocate subdictionary
        ev_dict[ev_id]['phases'] = []
        ev_dict[ev_id]['header'] = None
        for arr in orig.arrivals:
            pick = arr.pick_id.get_referred_object()
            sta = pick.waveform_id.station_code
            chan = pick.waveform_id.channel_code
            if chan[-1] != 'Z':
                continue
            sta_inv = inv.select(station=sta, channel=chan)
            # Do a rough incidence angle calculation based on dist and depth
            dist = dist_calc((orig.latitude, orig.longitude,
                              orig.depth / 1000.), (sta_inv[0][0].latitude,
                                                    sta_inv[0][0].longitude,
                                                    (sta_inv[0][0].elevation -
                                                     sta_inv[0][0][0].depth)
                                                     / 1000.))
            aoi = 90. - np.degrees(np.arcsin(orig.depth / 1000. / dist))
            if np.isnan(aoi):
                aoi = 180. - arr.takeoff_angle
            # Establish which station we're working with
            if sta.endswith('Z'):
                prefilt = pf_dict['GEONET']
            else:
                prefilt = pf_dict['MERC']
            wav_file = [wav for wav in wavs
                        if wav.split('_')[-1].split('.')[0] == chan
                        and wav.split('_')[-2] == sta]
            if len(wav_file) == 0:
                print('Waveform directory not found.')
                continue
            # Read in the corresponding trace
            # Cosine taper and demeaning applied by default
            tr = read(wav_file[0])[0].remove_response(inventory=inv,
                                                      pre_filt=prefilt,
                                                      output='DISP')
            # # Trim once more and detrend again(?)
            # tr.trim(starttime=pick.time - 1., endtime=pick.time + 3).detrend()
            # Trim around P pulse
            tr.trim(starttime=pick.time, endtime=pick.time + 0.12)
            # Find last zero crossing of the trimmed wav, assuming we've
            # trimmed only half a cycle. Then integrate from pick time to
            # first sample with a swapped sign (+/- or -/+)
            if plot:
                tr.plot()
            try:
                pulse = tr.data[:np.where(
                    np.diff(np.sign(tr.data)) != 0)[0][-1] + 2]
            except IndexError as e:
                print('Pulse never crosses zero. Investigate.')
                continue
            omega = np.trapz(pulse)
            # Now determine if the local max is + or -
            try:
                polarity = np.sign(pulse[argrelextrema(np.abs(pulse),
                                                       np.greater,
                                                       order=2)[0][0]])
            except IndexError as e:
                print('Couldnt find acceptable relative min/max, skipping')
                continue
            # Now we can populate the strings in ev_dict
            if file_type == 'raw':
                ev_dict[ev_id]['phases'].append(
                    "  {} {} {} {!s} {!s} {!s} {!s} {!s} {!s} {!s}\n".format(
                        sta, chan[-1], pick.phase_hint, omega * polarity,
                        arr.azimuth, aoi, arr.takeoff_angle, 5000, dist * 1000,
                        2600))
            elif file_type == 'vel1d':
                x, y, z = dec_2_merc_meters(sta_inv[0][0].longitude,
                                            sta_inv[0][0].latitude,
                                            sta_inv[0][0].elevation -
                                            sta_inv[0][0][0].depth)
                ev_dict[ev_id]['phases'].append(
                    "  {} {} {} {!s} {!s} {!s} {!s}\n".format(
                        sta, chan[-1], pick.phase_hint, omega * polarity,
                        y, x, z))
        if len(ev_dict[ev_id]['phases']) > 0:
            if file_type == 'raw':
                ev_dict[ev_id]['header'] = "{} {!s}\n".format(
                    ev_id, len(ev_dict[ev_id]['phases']))
            elif file_type == 'vel1d':
                ex, ey, ez = dec_2_merc_meters(orig.longitude, orig.latitude,
                                               -1 * orig.depth)
                ev_dict[ev_id]['header'] = "{} {!s} {!s} {!s} {!s} {!s}\n".format(
                    ev_id, len(ev_dict[ev_id]['phases']), ey, ex, ez, 2600)
    with open(outfile, 'w') as fo:
        for eid, dict in iteritems(ev_dict):
            if dict['header'] is not None:
                print('Writing event %s' % eid)
                fo.write(dict['header'])
                fo.writelines(dict['phases'])
    return