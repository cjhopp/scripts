#!/usr/bin/python
"""
Helper functions for subspace detectors
"""

def plot_frac_capture(detector):
    """
    Plot the fractional energy capture of a subspace detector for its design
    set. Include as part of Detector class at some point.
    :param detector:
    :return:
    """
    for dim in len(detector.u):
