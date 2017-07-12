#!/usr/bin/python
from __future__ import division
# import matplotlib
# matplotlib.rcParams['figure.dpi'] = 300

def foc_mec_from_event(catalog, station_names=False):
    """
    Just taking Tobias' plotting function out of obspyck
    :param catalog:
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from obspy.imaging.beachball import beach

    fms = catalog[0].focal_mechanisms
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
    print(av_np1_strike)
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
    for pick in catalog[0].picks:
        if pick.phase_hint != "P":
            continue
        wid = pick.waveform_id
        net = wid.network_code
        sta = wid.station_code
        arrival = getArrivalForPick(catalog[0].origins[-1].arrivals, pick)
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
