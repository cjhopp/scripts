#!/usr/bin/python

"""
Functions for calculating and working with magnitudes
"""

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


def compare_newberry_magnitudes(catalog1, catalog2):
    """
    Compare the magnitudes in two newberry catalogs
    :param catalog1:
    :param catalog2:
    :return:
    """
    new_mags = []
    old_mags = []
    inds = []
    for ev in catalog1:
        ot = ev.preferred_origin().time
        diffs = np.abs([ot - ev.preferred_origin().time for ev in catalog2])
        if np.min(diffs) > 1.:
            continue
        ind = np.argmin(diffs)
        inds.append(np.abs(ind))
        try:
            old_mag = ev.preferred_magnitude().mag
        except AttributeError:
            continue
        new_ev = catalog2[ind]
        if len(new_ev.magnitudes) == 0:
            continue
        po = new_ev.preferred_origin().resource_id
        nm = [nm.mag for nm in new_ev.magnitudes if nm.magnitude_type == 'MLc' and nm.origin_id == po]
        if len(nm) == 0:
            continue
        new_mags.append(nm)
        old_mags.append(old_mag)
    # Get regression coeficient
    model = LinearRegression()
    model.fit(new_mags, old_mags)
    ax = sns.regplot(x=new_mags, y=old_mags)
    ax.annotate(text=r'$Mw={:.2f}MLc + {:.2f}$'.format(float(model.coef_), float(model.intercept_)),
                xy=(0.5, 0.15), xycoords='axes fraction')
    ax.set_ylabel('Mw [old]')
    ax.set_xlabel('MLc [new]')
    plt.show()
    return
