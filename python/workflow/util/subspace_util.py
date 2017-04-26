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
    import numpy as np
    import matplotlib.pyplot as plt

    A = detector.sigma*detector.v
    fcs = np.array([len(A.T), detector.dimension])
    for i, ai in enumerate(A.T):
        for j in range(len(ai)):
            fcs[i,j] = ai.T * ai
    avg = [np.mean(dim) for dim in fcs.T]
    fig, ax = plt.subplots()
    for fc in fcs:
        ax.plot(fc, linecolor='grey')
    ax.plot(avg, linecolor='red')
    plt.show()
    return