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


def show_magnitude_error_effects():
    # --- Parameters ---
    np.random.seed(42)
    n_stations = 8
    network_radius = 5_000  # meters
    true_depth = 2_000  # meters
    grid_extent = 10_000  # meters for heatmap
    grid_points = 101  # resolution of heatmap

    # --- Reference amplitude function (simplified) ---
    def A0(delta):
        a, b = -1.3, 1.0
        return 10**(a + b*np.log10(delta + 1))
    # --- Station coordinates (uniformly around circle, z=0) ---
    angles = np.linspace(0, 2*np.pi, n_stations, endpoint=False)
    xs = network_radius * np.cos(angles)
    ys = network_radius * np.sin(angles)
    zs = np.zeros(n_stations)
    stations = np.stack([xs, ys, zs], axis=1)

    # --- True hypocenter ---
    hypo_true = np.array([0.0, 0.0, true_depth])
    d_true = np.linalg.norm(stations - hypo_true, axis=1)

    # --- Panel 1: Heatmap in horizontal plane at depth=true_depth ---
    x = np.linspace(-grid_extent, grid_extent, grid_points)
    y = np.linspace(-grid_extent, grid_extent, grid_points)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, true_depth)

    abs_dML = np.zeros_like(X)
    for i in range(grid_points):
        for j in range(grid_points):
            hypo_perturbed = np.array([X[i,j], Y[i,j], Z[i,j]])
            d_perturbed = np.linalg.norm(stations - hypo_perturbed, axis=1)
            mag_change = np.mean(np.abs(np.log10(A0(d_perturbed)) - np.log10(A0(d_true))))
            abs_dML[i,j] = mag_change

    # --- Depth error curve: ML change with depth (0 km to 5 km) ---
    depths = np.linspace(0, true_depth + 3000, 101)  # 0 km to 5 km
    abs_dML_depth = []
    for z in depths:
        hypo_perturbed = np.array([0.0, 0.0, z])
        d_perturbed = np.linalg.norm(stations - hypo_perturbed, axis=1)
        mag_change = np.mean(np.abs(np.log10(A0(d_perturbed)) - np.log10(A0(d_true))))
        abs_dML_depth.append(mag_change)
    abs_dML_depth = np.array(abs_dML_depth)

    # --- Amplitude error curve: ML change with amplitude error (0% to +100%) ---
    amplitude_error_percent = np.linspace(0, 1.0, 101)  # 0% to 100%
    amplitude_error_ratio = 1 + amplitude_error_percent
    abs_dML_amp = np.abs(np.log10(amplitude_error_ratio))

    # --- Effect of 1 or 2 bad amplitude picks in the network ---
    bad_pick_counts = [1, 2]
    bad_errors = [0.5, 1.0]  # 50% and 100% error
    avg_ML_error = np.zeros((len(bad_pick_counts), len(bad_errors)))
    for i, n_bad in enumerate(bad_pick_counts):
        for j, bad_err in enumerate(bad_errors):
            amp_ratios = np.ones(n_stations)
            amp_ratios[:n_bad] = 1 + bad_err  # first n_bad picks are bad
            ML_errors = np.log10(amp_ratios)
            avg_ML_error[i, j] = np.mean(np.abs(ML_errors))

    # --- Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    # Panel 1: Heatmap
    im = axs[0].imshow(abs_dML, extent=(-grid_extent/1000, grid_extent/1000, -grid_extent/1000, grid_extent/1000),
                    origin='lower', cmap='viridis', aspect='equal')
    axs[0].set_title('Avg |ΔML| in Horizontal Plane\n(Depth = 2 km)')
    axs[0].set_xlabel('X (km)')
    axs[0].set_ylabel('Y (km)')
    axs[0].plot(xs/1000, ys/1000, 'wo', markersize=8, markeredgecolor='k', label='Stations')
    axs[0].plot(0, 0, 'r*', markersize=12, label='True Hypocenter')
    axs[0].legend(loc='upper right')
    cb = plt.colorbar(im, ax=axs[0], label='Avg |ΔML|')

    # Panel 2: Depth error, amplitude error, and bad picks
    ax2 = axs[1]
    lns1 = ax2.plot(depths/1000, abs_dML_depth, color='tab:blue', label='Depth error (km)')
    ax2.set_xlabel('Depth (km)')
    ax2.set_ylabel('Avg |ΔML|')
    ax2.invert_xaxis()
    ax2.grid(True)
    ax2.set_title('Avg |ΔML|: Depth vs. Amplitude Error')

    # Twin x-axis for amplitude error
    ax3 = ax2.twiny()
    lns2 = ax3.plot(amplitude_error_percent*100, abs_dML_amp, color='tab:orange', label='Amplitude error (%)')

    # Add points for bad picks
    colors = ['tab:green', 'tab:red']
    markers = ['o', 's']
    for i, n_bad in enumerate(bad_pick_counts):
        for j, bad_err in enumerate(bad_errors):
            ax3.plot(bad_err*100, avg_ML_error[i, j], marker=markers[i], color=colors[i],
                    label=f'{n_bad} bad pick{"s" if n_bad > 1 else ""} ({int(bad_err*100)}%)', linestyle='None', markersize=9, markeredgecolor='k')

    # Combine legends
    lns = lns1 + lns2
    labels = [l.get_label() for l in lns]
    # Add bad pick labels
    labels += [f'1 bad pick (50%)', f'1 bad pick (100%)', f'2 bad picks (50%)', f'2 bad picks (100%)']
    handles = lns + [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', markeredgecolor='k', markersize=9, label='1 bad pick'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:red', markeredgecolor='k', markersize=9, label='2 bad picks')
    ]
    ax2.legend(handles, labels, loc='upper left')

    plt.tight_layout()
    plt.show()