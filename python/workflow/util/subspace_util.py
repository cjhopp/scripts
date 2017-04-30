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
    if detector.dimension > max(v.shape) or detector.dimension == np.inf:
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

def rewrite_subspace(detector, outfile):
    """
    Rewrite old subspace with U and V matrices switched
    :param detector:
    :return:
    """
    import copy
    from eqcorrscan.core.subspace import Detector

    new_u = copy.deepcopy(detector.v)
    new_v = copy.deepcopy(detector.u)
    final_u = [u.T for u in new_u]
    final_v = [v.T for v in new_v]
    final_data = copy.deepcopy(final_u)
    new_det = Detector(name=detector.name, sampling_rate=detector.sampling_rate,
                       multiplex=detector.multiplex, stachans=detector.stachans,
                       lowcut=detector.lowcut, highcut=detector.highcut,
                       filt_order=detector.filt_order, data=final_data,
                       u=final_u,sigma=detector.sigma,v=final_v,
                       dimension=detector.dimension)
    new_det.write(outfile)
    return