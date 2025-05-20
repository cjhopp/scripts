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


def plot_radius_vs_magnitude():
    # Define moment magnitude range
    Mw_extended = np.linspace(-5, 3, 200)

    # Define stress drops in Pascals
    stress_drops = [0.1e6, 1e6, 3e6, 10e6]  # 0.1, 1, 3, and 10 MPa
    colors = ['crimson', 'goldenrod', 'mediumseagreen', 'steelblue']
    labels = ['0.1 MPa', '1 MPa', '3 MPa', '10 MPa']

    # Compute seismic moment M0 from Mw
    log_M0_extended = 1.5 * Mw_extended + 9.1
    M0_extended = 10 ** log_M0_extended  # N·m

    # Initialize plot
    plt.figure(figsize=(10, 6))

    # Plot fracture radius curves for each stress drop
    for stress_drop, color, label in zip(stress_drops, colors, labels):
        const_factor = 7 / (16 * stress_drop)
        r = (const_factor * M0_extended) ** (1/3)  # radius in meters
        plt.plot(Mw_extended, r, label=label, color=color)

    # Plot formatting
    plt.yscale('log')
    plt.xlabel("Moment Magnitude (Mw)")
    plt.ylabel("Estimated Fracture Radius (m, log scale)")
    plt.title("Estimated Fracture Radius vs. Moment Magnitude\n(Various Stress Drops)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(title="Stress Drop")
    # plt.ticklabel_format(style='plain', axis='y')  # Standard numeric format on y-axis
    plt.tight_layout()

    # Display the plot
    plt.show()
    return


