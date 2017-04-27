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
    import scipy
    import matplotlib.pyplot as plt

    sigma = detector.sigma[0]
    v = detector.v[0]
    u = detector.u[0]
    sig = scipy.linalg.diagsvd(sigma, max(u.shape), max(v.shape))
    A = np.dot(sig, v)
    print('U matrix has shape %s' % str(u.shape))
    print('S matrix has shape %s' % str(sig.shape))
    print('V matrix has shape %s' % str(v.shape))
    print('A matrix has shape %s' % str(A.shape))
    if detector.dimension > 30 or detector.dimension == np.inf:
        dim = max(v.shape)
    else:
        dim = detector.dimension
    fig, ax = plt.subplots()
    av_fc_dict = {i: [] for i in range(dim)}
    for ai in A.T:
        fcs = []
        for j in range(dim):
            av_fc_dict[j].append(float(np.dot(ai[:j].T, ai[:j])))
            fcs.append(float(np.dot(ai[:j].T, ai[:j])))
        ax.plot(fcs, color='grey')
    avg = [np.average(dim[1]) for dim in av_fc_dict.items()]
    ax.plot(avg, color='red', linewidth=3.)
    plt.show()
    return