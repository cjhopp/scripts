#!/usr/bin/python

"""
Functions for plotting stress and fractures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import eigvalsh

# Local imports
from lbnl.boreholes import otv_to_sdd, depth_to_xyz
from lbnl.coordinates import cartesian_distance


def plot_csd_frac_mohrs(otv_picks, color_by='depth'):
    """
    Make second row of panels for Fig 6 of CSD fiber paper

    :param otv_picks: Path to Quinns wellcad picks
    :param color_by: Color by what variable? Default to 'depth'
    :return:
    """
    # Get strike-dip-depths and combine into three dataframes
    otv_mf, otv_dss, otv_none = otv_to_sdd(otv_picks)
    sdd_mf = pd.concat([df for w, df in otv_mf.items()])
    sdd_dss = pd.concat([df for w, df in otv_dss.items()])
    sdd_none = pd.concat([df for w, df in otv_none.items()])
    sdd_mf['Strike'] = sdd_mf['Azimuth'] - 90.
    sdd_dss['Strike'] = sdd_dss['Azimuth'] - 90.
    sdd_none['Strike'] = sdd_none['Azimuth'] - 90.
    # Set Guglielmi 2020 stress regime
    pf = 1.  # 1 MPa pf
    S = np.array([[6.2, 0, 0], [0., 4.5, 0.], [0., 0., 2.]])
    alpha = np.deg2rad(55.)  # Sig3 azimuth
    beta = np.deg2rad(-90.)
    gamma = 0.
    sig_mf, tau_mf = calculate_frac_stress(
        S, pf, alpha, beta, gamma, np.deg2rad(sdd_mf['Strike'].values),
        np.deg2rad(sdd_mf['Dip'].values))
    sig_dss, tau_dss = calculate_frac_stress(
        S, pf, alpha, beta, gamma, np.deg2rad(sdd_dss['Strike'].values),
        np.deg2rad(sdd_dss['Dip'].values))
    sig_none, tau_none = calculate_frac_stress(
        S, pf, alpha, beta, gamma, np.deg2rad(sdd_none['Strike'].values),
        np.deg2rad(sdd_none['Dip'].values))
    cols = [sdd_mf['Distance'].values, sdd_dss['Distance'].values,
            sdd_none['Distance'].values]
    # Now plot each situation
    fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
    # Make colorbar
    vr = [np.min([c for ds in cols for c in ds]), np.max([c for ds in cols for c in ds])]
    pt1 = plot_mohr3d(S, pf, sig_mf, tau_mf, colors=cols[0], ax=axes[0],
                      vrange=vr)
    pt2 = plot_mohr3d(S, pf, sig_dss, tau_dss, colors=cols[1], ax=axes[1],
                      vrange=vr)
    pt3 = plot_mohr3d(S, pf, sig_none, tau_none, colors=cols[2], ax=axes[2],
                      vrange=vr)
    fig.colorbar(pt1, orientation='horizontal', ax=axes, fraction=0.06,
                 label='Distance to excavation [m]')
    plt.show()
    return


def plot_mohr3d(S, pf, sigma, tau, colors, mu=0.45, ax=None, vrange=(0, 70)):
    r"""Plot 3D Mohr circles."""

    S3, S2, S1 = eigvalsh(S)
    S1 -= pf
    S2 -= pf
    S3 -= pf
    R_maj = 0.5 * (S1 - S3)
    cent_maj = 0.5 * (S1 + S3)

    R_min = 0.5 * (S2 - S3)
    cent_min = 0.5 * (S2 + S3)

    R_mid = 0.5 * (S1 - S2)
    cent_mid = 0.5 * (S1 + S2)
    circ1 = plt.Circle((cent_maj, 0), R_maj, facecolor='steelblue', lw=0,
                       edgecolor='#5c8037', alpha=0.5, zorder=0)
    circ2 = plt.Circle((cent_min, 0), R_min, facecolor='steelblue', lw=0,
                       alpha=0.5, zorder=0)
    circ3 = plt.Circle((cent_mid, 0), R_mid, facecolor='steelblue', lw=0,
                       alpha=0.5, zorder=0)
    if not ax:
        fig, ax = plt.subplots()
    ax.add_artist(circ1)
    ax.add_artist(circ2)
    ax.add_artist(circ3)
    # Plot failure criterion
    x = np.arange(10)
    y = x * mu
    ax.plot(x, y, color='k', linestyle=':', linewidth=1.5, alpha=0.5)
    # Plot fractures
    pts = ax.scatter(sigma, tau, c=colors, label='Fractures', s=30, alpha=0.7,
                     cmap='magma_r', vmin=vrange[0], vmax=vrange[1],
                     edgecolor='k', linewidth=0.5)
    ax.set_xlim(0, 6)
    ax.set_ylim(0., 1.1 * R_maj)
    ax.set_aspect('equal')
    ax.set_xlabel(r"$\sigma$ [MPa]", size=14)
    ax.set_ylabel(r"$\tau$ [MPa]", size=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return pts


def calculate_frac_stress(S, pf, a, b, g, strikes, dips):
    """
    Calculate normal and shear stresses on plane in given stress field

    ..note : Reference:
    https://dnicolasespinoza.github.io/node38.html

    :param S: Stress tensor in stress coordinates
    :param pf: Pore pressure
    :param alpha: Angles to NED coords
    :param beta: Angles to NED coords
    :param gamma: Angles to NED coords
    :param strikes: Array of strikes
    :param dips: Array of dips
    :return:
    """
    # Rotation matrix: stress --> geographic
    rpg = np.array([
        [np.cos(a) * np.cos(b),
         np.sin(a) * np.cos(b),
         -np.sin(b)],
        [(np.cos(a) * np.sin(b) * np.sin(g)) - (np.sin(a) * np.cos(g)),
         (np.sin(a) * np.sin(b) * np.sin(g)) + (np.cos(a) * np.cos(g)),
         np.cos(b) * np.sin(g)],
        [(np.cos(a) * np.sin(b) * np.cos(g)) + (np.sin(a) * np.sin(g)),
         (np.sin(a) * np.sin(b) * np.cos(g)) - (np.cos(a) * np.sin(g)),
         np.cos(b) * np.cos(g)]
    ])
    # Stress in geographic coords
    sg = rpg.T @ S @ rpg
    # Now define plane normals
    norms = np.vstack([
        -np.sin(strikes) * np.sin(dips),
        np.cos(strikes) * np.sin(dips),
        -np.cos(dips)
    ])
    # Stress vector acting on fault
    ts = [sg @ norms[:, i] for i in range(norms.shape[-1])]
    Sn = [np.dot(np.array(t), norms[:, i]) for i, t in enumerate(ts)]
    sig = np.array(Sn) - pf
    tau = np.array([np.sqrt(np.linalg.norm(t)**2 - np.linalg.norm(Sn[i])**2)
                    for i, t in enumerate(ts)])
    return sig, tau