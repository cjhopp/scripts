#!/usr/bin/python

"""
Set of functions for network design following Zara Rawlinson's MSc thesis
at VUW

Assumes straight ray paths and perfectly known hypocenters
"""

import numpy as np


def calc_amplitude(t, f, q):
    return np.exp(-t * f * np.pi / q) + 0.001


def calc_tt(hypocenters, stations, vp):
    # Need 3D: m stations (rows), 3 coords, n events (cols)
    # Distnces
    diff = stations[:, :, None] - hypocenters.T[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=1))
    return dist / vp


def meas_var(tts, f, q):
    # Measurement variance estimate
    # tts and amps are mxn, m stations, n events
    amps = calc_amplitude(tts, f, q)
    # Sum along n, prod along m
    return np.prod(np.mean(1 / amps**2, axis=1), axis=0)


def hyp_var(tts):
    varcov_tt = np.cov(tts)
    return np.linalg.det(varcov_tt)


def design_criterion(stas, hypocenters, f, q, vp):
    """
    Calculate the design criteria for a set of stations and hypocenters

    :param stas: Array of station locations
    :param hypocenters: Array of hypocenter locations
    :param f: Representative corner freq for events [Hz]
    :param q: Representative attenuation factor
    :param vp: P-wave velocity [m/s]
    :return:
    """
    # Compute travel times
    tts = calc_tt(hypocenters, stas, vp)
    # Hypocenter variance
    Vhyp = hyp_var(tts)
    # Measurement variance
    Vmeas = meas_var(tts, f, q)
    return np.log(Vhyp) - np.log(Vmeas)


def generate_hypocenters():
    """
    Generate randomly distributed hypocenters within a volume

    :return:
    """
    return


def generate_borehole_pts():
    """
    Take borehole trajectories (and depths?) and return an array of xyz points

    :return:
    """
    return


def design_network(geometry, no_sensors, sources=None, source_volume=None,
                   sensors_p_well=None):
    """
    Run Zara's network design implementation

    :param geometry: Accepts 'wells' or 'surface' for now
    :param no_sensors: Maximum number of sensors for the network
    :param sources: Array of source points
    :param source_volume: Give a shapely(?) volume to distribute sources into
    :param sensors_p_well: Optional dict giving the maximum number of sensors
        per well

    :return:
    """
    if not sources and source_volume:
        # TODO Find way of sampling inside a covex hull?
        sources = []
    # Build from one sensor up to no_sensors
    # Preallocate empty network
    old_stas = np.array([])
    for s in range(no_sensors):
        # Make 3d array of no-sensors x pts x iteration
        xs, ys = np.mgrid[0:50:1, 0:50:1]
        zs = np.zeros(xs.shape)
        # TODO Should make this 2D instead of 3D so that apply along
        # TODO will work. Then reshape in the func?

        # TODO https://stackoverflow.com/questions/22424096/apply-functions-to-3d-numpy-array
        trial_pts = np.vstack(xs.flatten(), ys.flatten(), zs.flatten()).T
        stas = np.stack([old_stas, trial_pts], axis=-1)
        target = np.apply_along_axis(design_criterion, 1, stas,
                                     hypocenters=sources, f=f, q=q, vp=vp)
        # Find maximum along the iteration axis
        winner = np.argmin(target, axis='blah')
    return


### Plotting funcs ###

def plot_solution():
    # TODO Replicate Zara's contour plots
    return