#!/usr/bin/env python3
"""
CASSM per-pair result visualization.
Generates a multi-page PDF from a cassm_dashboard_bundle_full.npz file.

Usage:
    conda run -n ttcr_inv python cussp_cassm_plot_results.py \
        --config cussp_cassm_plot_config.yaml
"""
import argparse
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (activates 3-D projection)
from scipy.ndimage import generic_filter

try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_BOREHOLES = [
    "AML", "AML", "AML", "AML",
    "AMU", "AMU", "AMU", "AMU",
    "DML", "DML", "DML", "DML",
    "DMU", "DMU", "DMU", "DMU",
]
BH_ORDER    = ["AML", "AMU", "DML", "DMU"]
REC_BH_ORDER = ["AML", "AMU", "DML", "DMU", "TS"]
BH_COLORS   = {
    "AML": "#1f77b4", "AMU": "#ff7f0e",
    "DML": "#2ca02c", "DMU": "#d62728", "TS": "#9467bd",
}


def get_rec_borehole(rec_idx: int) -> str:
    ch = rec_idx + 1
    if ch <= 12:  return "AML"
    if ch <= 24:  return "AMU"
    if ch <= 36:  return "DML"
    if ch <= 48:  return "DMU"
    return "TS"


def build_source_labels(source_boreholes: list[str], n_sources: int) -> list[str]:
    """Build stable source labels like AMLS1, AMLS2, ..., DMUS4."""
    labels: list[str] = []
    counts: dict[str, int] = {}
    for i in range(n_sources):
        bh = source_boreholes[i] if i < len(source_boreholes) and source_boreholes[i] else f"SRC{i+1:02d}"
        counts[bh] = counts.get(bh, 0) + 1
        labels.append(f"{bh}S{counts[bh]}")
    return labels


def build_receiver_labels(n_receivers: int) -> list[str]:
    """Build receiver labels: AML1/Z..DMU4/Y, then TS24..TS01."""
    comp = ("Z", "X", "Y")
    accel_bh = ("AML", "AMU", "DML", "DMU")
    labels: list[str] = []
    for ch in range(1, n_receivers + 1):
        if ch <= 48:
            bh_idx = (ch - 1) // 12
            sensor_num = ((ch - 1) % 12) // 3 + 1
            c = comp[(ch - 1) % 3]
            bh = accel_bh[bh_idx] if bh_idx < len(accel_bh) else f"BH{bh_idx+1}"
            labels.append(f"{bh}{sensor_num}/{c}")
        else:
            ts_num = 73 - ch
            labels.append(f"TS{ts_num:02d}")
    return labels


def rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    return generic_filter(arr.astype(float), np.nanmedian, size=window, mode="nearest")


def parse_times(epoch_times):
    # Coerce invalid entries (e.g., "NaT") to NaT while preserving array length.
    return pd.to_datetime(
        [str(t) for t in epoch_times],
        utc=True,
        errors="coerce",
    ).to_numpy(dtype="datetime64[s]")


def t_axis(epoch_times):
    """Return (t_days, t0_str, t_num) robust to invalid timestamps.

    If at least one valid epoch time exists, use the first valid timestamp as t0
    and linearly interpolate any missing entries in index space.  If all times are
    invalid, fall back to epoch index in days so plotting still works.
    """
    t = parse_times(epoch_times)
    # Convert to integer seconds for interpolation; NaT becomes the minimum int.
    sec = t.astype("datetime64[s]").astype(np.int64)
    nat_i64 = np.iinfo(np.int64).min
    valid = sec != nat_i64

    if valid.any():
        x = np.arange(len(sec), dtype=float)
        sec_f = sec.astype(float)
        if not valid.all():
            # Fill missing timestamps by linear interpolation in epoch-index space.
            sec_f[~valid] = np.interp(x[~valid], x[valid], sec_f[valid])
        t0 = float(sec_f[valid][0])
        t_days = (sec_f - t0) / 86400.0
        t0_str = str(np.datetime64(int(t0), "s"))[:10]
        t_num = mdates.date2num(pd.to_datetime(sec_f, unit="s", utc=True).to_pydatetime())
        return t_days, t0_str, t_num

    # Last resort: synthetic time axis by epoch index.
    t_days = np.arange(len(sec), dtype=float) / 86400.0
    t_num = mdates.date2num(pd.to_datetime(np.arange(len(sec)), unit="s", utc=True).to_pydatetime())
    return t_days, "unknown", t_num


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _add_colorbar(fig, ax, im, label):
    plt.colorbar(im, ax=ax, label=label, fraction=0.03, pad=0.02)


def _format_time_axis(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _format_time_axis_full_utc(ax):
    """Format x-axis with full UTC datetime stamps on each tick."""
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax.tick_params(axis="x", labelrotation=35)


def load_injection_series(injection_csv: str | None):
    """Load injection data as matplotlib datenums + pressure/flow arrays.

    Expected columns: Time, PT 503, Net Flow.
    Returns None if input is missing or cannot be parsed.
    """
    if not injection_csv:
        return None
    path = Path(injection_csv)
    if not path.exists():
        print(f"Injection CSV not found: {path} (skipping injection overlay)")
        return None
    try:
        df = pd.read_csv(path)
        if "Time" not in df.columns:
            print(f"Injection CSV missing 'Time' column: {path} (skipping injection overlay)")
            return None
        t = pd.to_datetime(df["Time"], utc=True, errors="coerce")
        ok = t.notna()
        if ok.sum() < 2:
            print(f"Injection CSV has insufficient valid timestamps: {path} (skipping injection overlay)")
            return None

        p = pd.to_numeric(df.get("PT 503", pd.Series(index=df.index, dtype=float)), errors="coerce")
        q = pd.to_numeric(df.get("Net Flow", pd.Series(index=df.index, dtype=float)), errors="coerce")

        t_num = mdates.date2num(t[ok].dt.to_pydatetime())
        return {
            "t_num": t_num,
            "pressure": p[ok].to_numpy(dtype=float),
            "flow": q[ok].to_numpy(dtype=float),
            "label": path.name,
        }
    except Exception as exc:
        print(f"Failed loading injection CSV {path}: {exc} (skipping injection overlay)")
        return None


def plot_dt_heatmap(dt, t_num, pair_src_bh, pair_rec_bh, vmax_pct, ax_title):
    """Return (fig, ax) with a 2-D heatmap of dt sorted by source borehole."""
    n_pairs = dt.shape[0]
    if n_pairs == 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No pairs available for dt heatmap", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Sort: source bh first, then receiver bh
    rec_bh_rank = {b: i for i, b in enumerate(REC_BH_ORDER)}
    src_bh_rank = {b: i for i, b in enumerate(BH_ORDER)}
    sort_key = [(src_bh_rank.get(pair_src_bh[i], 9), rec_bh_rank.get(pair_rec_bh[i], 9), i)
                for i in range(n_pairs)]
    sort_order = [x[2] for x in sorted(sort_key)]
    dt_sorted = dt[sort_order, :]

    vmax = np.nanpercentile(np.abs(dt_sorted[~np.isnan(dt_sorted)]), vmax_pct) if (~np.isnan(dt_sorted)).any() else 50

    fig, ax = plt.subplots(figsize=(18, 9))
    im = ax.imshow(dt_sorted, aspect="auto", origin="upper",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[t_num[0], t_num[-1], n_pairs, 0])
    _add_colorbar(fig, ax, im, "dt (µs)")

    # Draw source-borehole boundaries and labels
    prev_sbh = pair_src_bh[sort_order[0]]
    boundary_start = 0
    for row, idx in enumerate(sort_order):
        sbh = pair_src_bh[idx]
        if sbh != prev_sbh:
            ax.axhline(row, color="k", lw=0.8, alpha=0.6)
            mid = (boundary_start + row) / 2
            ax.text(t_num[-1] + (t_num[-1] - t_num[0]) * 0.005, mid,
                    f"src:{prev_sbh}", va="center", ha="left", fontsize=7, clip_on=False)
            boundary_start = row
            prev_sbh = sbh
    mid = (boundary_start + n_pairs) / 2
    ax.text(t_num[-1] + (t_num[-1] - t_num[0]) * 0.005, mid,
            f"src:{prev_sbh}", va="center", ha="left", fontsize=7, clip_on=False)

    _format_time_axis(ax)
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Pair (sorted: src bh, then rec bh)")
    ax.set_title(ax_title)
    plt.tight_layout()
    return fig


def plot_sensor_heatmaps(dt, t_num, pair_is_hydro, vmax_pct):
    """Side-by-side heatmaps: accel vs hydro."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, is_h, label, color in [
        (axes[0], False, "Accelerometers", "#1f77b4"),
        (axes[1], True,  "Hydrophones",    "#9467bd"),
    ]:
        sel = np.where(pair_is_hydro == is_h)[0]
        if not len(sel):
            ax.set_visible(False)
            continue
        sub = dt[sel, :]
        flat = sub[~np.isnan(sub)]
        vmax = np.nanpercentile(np.abs(flat), vmax_pct) if flat.size else 50
        im = ax.imshow(sub, aspect="auto", origin="upper",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[t_num[0], t_num[-1], len(sel), 0])
        _add_colorbar(fig, ax, im, "dt (µs)")
        ax.set_title(f"dt — {label} ({len(sel)} pairs)  [±{vmax:.0f} µs]")
        _format_time_axis(ax)
        ax.set_xlabel("UTC time")
        ax.set_ylabel("Pair index")
    fig.suptitle("dt heatmap by sensor type", fontsize=13)
    plt.tight_layout()
    return fig


def plot_per_sourcebh(dt, t_num, pair_src_bh, pair_rec_bh, src_bh_name,
                      dt_ylim_us, smooth_window):
    """One page per source borehole: subpanels per receiver borehole."""
    sel_src = [i for i, b in enumerate(pair_src_bh) if b == src_bh_name]
    if not sel_src:
        return None

    rec_bh_groups = {}
    for i in sel_src:
        rec_bh_groups.setdefault(pair_rec_bh[i], []).append(i)
    n_groups = len(rec_bh_groups)

    fig, axes = plt.subplots(n_groups, 1, figsize=(18, 3.8 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]

    for ax, rbh in zip(axes, [b for b in REC_BH_ORDER if b in rec_bh_groups]):
        idxs = rec_bh_groups[rbh]
        color = BH_COLORS.get(rbh, "gray")
        alpha = max(0.08, 0.6 / len(idxs))
        for i in idxs:
            ax.plot(t_num, dt[i, :], color=color, lw=0.5, alpha=alpha)
        # Smoothed median
        med = np.nanmedian(dt[idxs, :], axis=0)
        sm = rolling_median(np.where(np.isnan(med), 0, med), smooth_window)
        ax.plot(t_num, sm, color="k", lw=2.0,
                label=f"smoothed median (n={len(idxs)}, w={smooth_window})")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_ylabel("dt (µs)")
        ax.set_ylim(-dt_ylim_us, dt_ylim_us)
        ax.set_title(f"src:{src_bh_name} → rec:{rbh}  ({len(idxs)} pairs)")
        ax.legend(fontsize=8, loc="upper right")

    _format_time_axis(axes[-1])
    axes[-1].set_xlabel("UTC time")
    fig.suptitle(f"dt per pair — source borehole: {src_bh_name}", fontsize=13)
    plt.tight_layout()
    return fig


def plot_bh_crossing_grid(dt, t_num, pair_src_bh, pair_rec_bh, dt_ylim_us, smooth_window):
    """4×5 grid: source bh (rows) × receiver bh (cols), median + individual pairs."""
    fig, axes = plt.subplots(4, 5, figsize=(22, 14), sharex=True, sharey=True)
    for ri, sbh in enumerate(BH_ORDER):
        for ci, rbh in enumerate(REC_BH_ORDER):
            ax = axes[ri, ci]
            idxs = [i for i in range(len(pair_src_bh))
                    if pair_src_bh[i] == sbh and pair_rec_bh[i] == rbh]
            if not idxs:
                ax.set_visible(False)
                continue
            color = BH_COLORS.get(rbh, "gray")
            alpha = max(0.05, 0.5 / len(idxs))
            for i in idxs:
                ax.plot(t_num, dt[i, :], color=color, lw=0.4, alpha=alpha)
            med = np.nanmedian(dt[idxs, :], axis=0)
            sm = rolling_median(np.where(np.isnan(med), 0, med), smooth_window)
            ax.plot(t_num, sm, color="k", lw=1.8)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_title(f"{sbh}→{rbh}\nn={len(idxs)}", fontsize=7)
            ax.set_ylim(-dt_ylim_us, dt_ylim_us)
            ax.tick_params(labelsize=6)

    for ri, sbh in enumerate(BH_ORDER):
        axes[ri, 0].set_ylabel(f"src:{sbh}\ndt (µs)", fontsize=8)
    for ci, rbh in enumerate(REC_BH_ORDER):
        _format_time_axis(axes[-1, ci])
        axes[-1, ci].set_xlabel(f"UTC\nrec:{rbh}", fontsize=8)

    fig.suptitle(
        "dt by borehole crossing  (black = smoothed median, colour = individual pairs)",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


def plot_metric_by_sensor(metric, t_num, pair_is_hydro, smooth_window,
                          metric_name, unit):
    """Two-panel: accel (top) / hydro (bottom) time series."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    for ax, is_h, label, color in [
        (ax1, False, "Accelerometers", "#1f77b4"),
        (ax2, True,  "Hydrophones",    "#9467bd"),
    ]:
        sel = np.where(pair_is_hydro == is_h)[0]
        if not len(sel):
            ax.set_visible(False)
            continue
        sub = metric[sel, :]
        alpha = max(0.04, 0.4 / len(sel))
        for i in range(len(sel)):
            ax.plot(t_num, sub[i, :], color=color, lw=0.4, alpha=alpha)
        med = np.nanmedian(sub, axis=0)
        ax.plot(t_num, rolling_median(med, smooth_window), color="k", lw=2,
                label=f"median (n={len(sel)}, w={smooth_window})")
        ax.set_ylabel(f"{metric_name} ({unit})")
        ax.set_title(f"{metric_name} — {label}")
        ax.legend(fontsize=8, loc="upper right")
    _format_time_axis(ax2)
    ax2.set_xlabel("UTC time")
    fig.suptitle(f"{metric_name} over time", fontsize=13)
    plt.tight_layout()
    return fig


def plot_each_pair_timeseries(dt, cc, cf, t_num, pair_src, pair_rec, pair_src_name, pair_rec_name,
                              dt_ylim_us, pair_index_full, injection=None,
                              env_cc=None, env_lag_us=None, env_smooth_lag_us=None):
    """Yield one dt-vs-time figure per active pair.

    *env_cc*           : (n_pairs, n_epochs) envelope xcorr peak cc — overlaid on the
                         cc panel alongside xcorr_peak_cc.  None if not available.
    *env_lag_us*       : (n_pairs, n_epochs) raw coarse envelope lag in µs — overlaid
                         on the dt panel as a thin dashed line (noisy, for QC).
    *env_smooth_lag_us*: (n_pairs, n_epochs) causal running-median smoothed coarse lag
                         in µs — overlaid as a heavier dashed line showing what actually
                         guided the fine xcorr search center.
    """
    inj_t = None
    inj_p = None
    inj_q = None
    if injection is not None:
        inj_t = injection.get("t_num")
        inj_p = injection.get("pressure")
        inj_q = injection.get("flow")

    for i in range(dt.shape[0]):
        if inj_t is not None:
            fig, (ax, ax_cc, ax_cf, ax_inj) = plt.subplots(
                4, 1, figsize=(14, 10.4), sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.0, 1.0, 1.1]}
            )
        else:
            fig, (ax, ax_cc, ax_cf) = plt.subplots(
                3, 1, figsize=(14, 7.8), sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.0, 1.0]}
            )
            ax_inj = None

        # ── dt panel ─────────────────────────────────────────────────────────
        ax.plot(t_num, dt[i, :], color="#1f77b4", lw=0.8, label="fine xcorr dt")
        # Raw coarse envelope lag: thin dashed — noisy but shows true envelope output.
        if env_lag_us is not None:
            eli = env_lag_us[i, :]
            eli_plot = np.where(np.isfinite(eli), eli, np.nan)
            ax.plot(t_num, eli_plot, color="#9467bd", lw=0.5, ls=":",
                    alpha=0.5, label="env coarse lag (raw)")
        # Smoothed coarse lag: heavier dashed — this is what guided the fine search.
        if env_smooth_lag_us is not None:
            esli = env_smooth_lag_us[i, :]
            esli_plot = np.where(np.isfinite(esli), esli, np.nan)
            ax.plot(t_num, esli_plot, color="#9467bd", lw=1.2, ls="--",
                    alpha=0.85, label="env smooth lag (guide)")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.legend(fontsize=7, loc="upper right", ncol=2)

        dti = dt[i, :]
        # Include envelope lag values in the y-range so neither trace is clipped.
        all_dt_vals = [dti]
        if env_lag_us is not None:
            all_dt_vals.append(env_lag_us[i, :])
        if env_smooth_lag_us is not None:
            all_dt_vals.append(env_smooth_lag_us[i, :])
        finite = np.concatenate([v[np.isfinite(v)] for v in all_dt_vals])
        if finite.size:
            y_min = float(np.min(finite))
            y_max = float(np.max(finite))
            span = y_max - y_min
            if span < 1.0:
                # Keep a readable vertical scale even for nearly flat traces.
                pad = 0.5
                y_mid = 0.5 * (y_min + y_max)
                ax.set_ylim(y_mid - pad, y_mid + pad)
            else:
                pad = 0.08 * span
                ax.set_ylim(y_min - pad, y_max + pad)
        else:
            # Fallback when a pair has no finite dt samples.
            ax.set_ylim(-dt_ylim_us, dt_ylim_us)
        ax.set_ylabel("dt (µs)")
        ax.set_title(
            f"pair {int(pair_index_full[i])}  src:{pair_src_name[i]}"
            f" -> rec:{pair_rec_name[i]}"
        )

        # ── cc panel — fine xcorr cc + envelope cc overlaid ──────────────────
        cci = cc[i, :] if cc is not None else None
        env_cci = env_cc[i, :] if env_cc is not None else None
        if cci is not None:
            ax_cc.plot(t_num, cci, color="#ff7f0e", lw=0.8, label="xcorr peak cc")
        else:
            ax_cc.text(0.5, 0.5, "xcorr_peak_cc unavailable in bundle",
                       ha="center", va="center", transform=ax_cc.transAxes, fontsize=8)
        if env_cci is not None:
            # Mask NaN (pairs where envelope mode was off or pair was skipped).
            env_cci_plot = np.where(np.isfinite(env_cci), env_cci, np.nan)
            ax_cc.plot(t_num, env_cci_plot, color="#9467bd", lw=0.7, ls="--",
                       alpha=0.85, label="envelope cc")
        if cci is not None or env_cci is not None:
            ax_cc.legend(fontsize=7, loc="lower right", ncol=2)
        ax_cc.set_ylabel("peak cc")
        ax_cc.set_ylim(0.0, 1.0)
        ax_cc.grid(True, alpha=0.2)

        cfi = cf[i, :] if cf is not None else None
        if cfi is not None:
            ax_cf.plot(t_num, cfi, color="#2ca02c", lw=0.8)
            ax_cf.set_ylabel("centfreq (kHz)")
            finite_cf = cfi[np.isfinite(cfi)]
            if finite_cf.size:
                y_min = float(np.min(finite_cf))
                y_max = float(np.max(finite_cf))
                span = y_max - y_min
                pad = 0.08 * span if span > 0.2 else 0.1
                ax_cf.set_ylim(y_min - pad, y_max + pad)
            ax_cf.grid(True, alpha=0.2)
        else:
            ax_cf.text(0.5, 0.5, "centfreq unavailable in bundle", ha="center", va="center")
            ax_cf.set_ylabel("centfreq (kHz)")
            ax_cf.grid(True, alpha=0.2)

        if ax_inj is not None:
            # Plot pressure and flow on twin y-axes under each pair dt panel.
            p_ok = np.isfinite(inj_p)
            q_ok = np.isfinite(inj_q)

            if p_ok.any():
                ax_inj.plot(inj_t[p_ok], inj_p[p_ok], color="#d62728", lw=1.0, label="PT 503")
            ax_inj.set_ylabel("PT 503", color="#d62728")
            ax_inj.tick_params(axis="y", labelcolor="#d62728")
            ax_inj.grid(True, alpha=0.25)

            ax_q = ax_inj.twinx()
            if q_ok.any():
                ax_q.plot(inj_t[q_ok], inj_q[q_ok], color="#2ca02c", lw=1.0, label="Net Flow")
            ax_q.set_ylabel("Net Flow", color="#2ca02c")
            ax_q.tick_params(axis="y", labelcolor="#2ca02c")

            # Combined legend
            h1, l1 = ax_inj.get_legend_handles_labels()
            h2, l2 = ax_q.get_legend_handles_labels()
            if h1 or h2:
                ax_inj.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
            ax_inj.set_xlabel("UTC time")
            _format_time_axis_full_utc(ax_inj)
        else:
            ax_cf.set_xlabel("UTC time")
            _format_time_axis_full_utc(ax_cf)

        fig.tight_layout()
        yield fig


def plot_stats_grid(dt, pair_src_bh, pair_rec_bh):
    """4×5 heatmaps: mean dt and std dt per borehole crossing."""
    std_grid  = np.full((4, 5), np.nan)
    mean_grid = np.full((4, 5), np.nan)
    n_grid    = np.zeros((4, 5), dtype=int)

    for ri, sbh in enumerate(BH_ORDER):
        for ci, rbh in enumerate(REC_BH_ORDER):
            idxs = [i for i in range(len(pair_src_bh))
                    if pair_src_bh[i] == sbh and pair_rec_bh[i] == rbh]
            if not idxs:
                continue
            vals = dt[idxs, :].flatten()
            vals = vals[~np.isnan(vals)]
            if vals.size:
                std_grid[ri, ci]  = float(np.std(vals))
                mean_grid[ri, ci] = float(np.mean(vals))
                n_grid[ri, ci]    = len(idxs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles  = ["dt std dev (µs) — noise/variability", "dt mean (µs) — systematic offset"]
    cmaps   = ["viridis", "RdBu_r"]
    grids   = [std_grid, mean_grid]
    vlims   = [(None, None), (-20, 20)]

    for ax, grid, title, cmap, (vmin, vmax) in zip(axes, grids, titles, cmaps, vlims):
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(5)); ax.set_xticklabels(REC_BH_ORDER)
        ax.set_yticks(range(4)); ax.set_yticklabels(BH_ORDER)
        ax.set_xlabel("Receiver borehole")
        ax.set_ylabel("Source borehole")
        ax.set_title(title)
        _add_colorbar(fig, ax, im, "µs")
        median_val = np.nanmedian(grid)
        for ri in range(4):
            for ci in range(5):
                if not np.isnan(grid[ri, ci]):
                    txt_color = "white" if grid[ri, ci] > median_val else "black"
                    ax.text(ci, ri,
                            f"{grid[ri, ci]:.1f}\n(n={n_grid[ri, ci]})",
                            ha="center", va="center", fontsize=7, color=txt_color)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Geometry loading and 3D/sensitivity coverage
# ---------------------------------------------------------------------------

def _load_geometry_csvs(sources_csv: str, receivers_csv: str,
                        src_bh_list: list, n_rec: int = 72):
    """Load source/receiver coordinates from CSV files.

    Source mapping  : ordered by borehole in *src_bh_list*, depth-ascending.
    Receiver mapping: ch1-48 (accel) → 4 BH × 12-ch groups, 3 ch/sensor.
                      ch49-72 (hydro) → TS01-TS24.

    Returns (src_xyz, rec_xyz) each (N, 3) float64; NaN where coord unknown.
    """
    # ── sources ──────────────────────────────────────────────────────────────
    src_by_bh: dict = {}
    with open(sources_csv) as fh:
        for row in _csv.DictReader(fh):
            bh = row["borehole"]
            src_by_bh.setdefault(bh, []).append(
                (float(row["depth_m"]), float(row["x"]), float(row["y"]), float(row["z"]))
            )
    for bh in src_by_bh:
        src_by_bh[bh].sort()          # depth-ascending → index order matches source_index

    bh_counts: dict = {}
    src_xyz = np.full((len(src_bh_list), 3), np.nan)
    for i, bh in enumerate(src_bh_list):
        n = bh_counts.get(bh, 0)
        bh_counts[bh] = n + 1
        entries = src_by_bh.get(bh, [])
        if n < len(entries):
            _, x, y, z = entries[n]
            src_xyz[i] = [x, y, z]

    # ── receivers ────────────────────────────────────────────────────────────
    rec_by_id: dict = {}
    with open(receivers_csv) as fh:
        for row in _csv.DictReader(fh):
            rec_by_id[row["receiver_id"]] = [float(row["x"]), float(row["y"]), float(row["z"])]

    accel_bh = ["AML", "AMU", "DML", "DMU"]
    rec_xyz = np.full((n_rec, 3), np.nan)
    for rec_idx in range(n_rec):
        ch = rec_idx + 1        # 1-based
        if ch <= 48:
            bh = accel_bh[(ch - 1) // 12]
            sensor_in_bh = ((ch - 1) % 12) // 3   # 0-3
            rid = f"{bh}{sensor_in_bh + 1}"
        else:
            rid = f"TS{ch - 48:02d}"
        if rid in rec_by_id:
            rec_xyz[rec_idx] = rec_by_id[rid]

    return src_xyz, rec_xyz


def plot_3d_coverage(src_xyz, rec_xyz, active_idxs, pair_is_hydro,
                     pair_src_bh, n_rec: int = 72):
    """Two 3D subplots: all active pairs vs hydrophones-only.

    Sources = coloured stars (by borehole); accelerometer receivers = circles;
    hydrophone receivers = diamonds.  Rays coloured by source borehole.
    """
    fig = plt.figure(figsize=(20, 9))
    scenarios = [
        ("All active pairs (accel + hydro)",    np.ones(len(active_idxs), dtype=bool)),
        ("Hydrophones only (accel removed)",      pair_is_hydro.astype(bool)),
    ]

    for col, (title, pmask) in enumerate(scenarios):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        sel = active_idxs[pmask]
        sbhs_sel = [pair_src_bh[i] for i, m in enumerate(pmask) if m]

        # ── rays ─────────────────────────────────────────────────────────────
        for pidx, sbh in zip(sel, sbhs_sel):
            si, ri = divmod(int(pidx), n_rec)
            if np.any(np.isnan(src_xyz[si])) or np.any(np.isnan(rec_xyz[ri])):
                continue
            color = BH_COLORS.get(sbh, "#888888")
            ax.plot([src_xyz[si, 0], rec_xyz[ri, 0]],
                    [src_xyz[si, 1], rec_xyz[ri, 1]],
                    [src_xyz[si, 2], rec_xyz[ri, 2]],
                    color=color, lw=0.35, alpha=0.22)

        # ── receivers ────────────────────────────────────────────────────────
        seen_rec: set = set()
        for pidx in sel:
            ri = int(pidx) % n_rec
            if ri in seen_rec or np.any(np.isnan(rec_xyz[ri])):
                continue
            seen_rec.add(ri)
            is_h = ri >= 48
            ax.scatter(*rec_xyz[ri],
                       color="#9467bd" if is_h else "#1f77b4",
                       marker="D" if is_h else "o",
                       s=20, zorder=5, depthshade=True)

        # ── sources ──────────────────────────────────────────────────────────
        seen_src: set = set()
        for pidx, sbh in zip(sel, sbhs_sel):
            si = int(pidx) // n_rec
            if si in seen_src or np.any(np.isnan(src_xyz[si])):
                continue
            seen_src.add(si)
            ax.scatter(*src_xyz[si],
                       color=BH_COLORS.get(sbh, "#888888"),
                       marker="*", s=80, zorder=6,
                       edgecolors="k", linewidths=0.4)

        ax.set_xlabel("E (m)", fontsize=8); ax.set_ylabel("N (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.set_title(f"{title}\n({pmask.sum()} ray paths)", fontsize=10)
        ax.tick_params(labelsize=7)

    legend_handles = [
        Line2D([0], [0], color=BH_COLORS["AML"], lw=2, label="src: AML"),
        Line2D([0], [0], color=BH_COLORS["AMU"], lw=2, label="src: AMU"),
        Line2D([0], [0], color=BH_COLORS["DML"], lw=2, label="src: DML"),
        Line2D([0], [0], color=BH_COLORS["DMU"], lw=2, label="src: DMU"),
        Line2D([0], [0], marker="o", color="#1f77b4", lw=0, ms=7, label="rec: accel"),
        Line2D([0], [0], marker="D", color="#9467bd", lw=0, ms=7, label="rec: hydro"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("3D ray coverage — spatial sensitivity analysis", fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def plot_sensitivity_maps(src_xyz, rec_xyz, active_idxs, pair_is_hydro, n_rec: int = 72):
    """2D hit-count sensitivity maps (plan, E-Z section, N-Z section).

    Two rows: all pairs (top) vs hydrophones-only (bottom).
    Both rows share the same colour scale so sensitivity loss is immediately
    visible.
    """
    # ── grid bounds ──────────────────────────────────────────────────────────
    valid_pts = np.vstack([
        src_xyz[~np.any(np.isnan(src_xyz), axis=1)],
        rec_xyz[~np.any(np.isnan(rec_xyz), axis=1)],
    ])
    pad = 5.0
    xmin, xmax = valid_pts[:, 0].min() - pad, valid_pts[:, 0].max() + pad
    ymin, ymax = valid_pts[:, 1].min() - pad, valid_pts[:, 1].max() + pad
    zmin, zmax = valid_pts[:, 2].min() - pad, valid_pts[:, 2].max() + pad
    nx, ny, nz = 50, 40, 35
    n_samp = 60          # samples per ray (fast, vectorised per-ray)

    def _hits(mask):
        xy = np.zeros((nx, ny), dtype=np.int32)
        xz = np.zeros((nx, nz), dtype=np.int32)
        yz = np.zeros((ny, nz), dtype=np.int32)
        t_vals = np.linspace(0.0, 1.0, n_samp)
        for pidx in active_idxs[mask]:
            si, ri = divmod(int(pidx), n_rec)
            if np.any(np.isnan(src_xyz[si])) or np.any(np.isnan(rec_xyz[ri])):
                continue
            pts = src_xyz[si] + t_vals[:, None] * (rec_xyz[ri] - src_xyz[si])  # (n_samp,3)
            ix = np.clip(((pts[:, 0] - xmin) / (xmax - xmin) * nx).astype(int), 0, nx - 1)
            iy = np.clip(((pts[:, 1] - ymin) / (ymax - ymin) * ny).astype(int), 0, ny - 1)
            iz = np.clip(((pts[:, 2] - zmin) / (zmax - zmin) * nz).astype(int), 0, nz - 1)
            np.add.at(xy, (ix, iy), 1)
            np.add.at(xz, (ix, iz), 1)
            np.add.at(yz, (iy, iz), 1)
        return xy, xz, yz

    scenarios = [
        ("All pairs",        np.ones(len(active_idxs), dtype=bool)),
        ("Hydro only",       pair_is_hydro.astype(bool)),
    ]
    all_hits = [_hits(m) for _, m in scenarios]

    vmax = max(g.max() for grids in all_hits for g in grids if g.max() > 0)

    views = [
        ("Plan view (E–N)",   [xmin, xmax, ymin, ymax], "E (m)", "N (m)"),
        ("Section (E–Z)",     [xmin, xmax, zmin, zmax], "E (m)", "Z (m)"),
        ("Section (N–Z)",     [ymin, ymax, zmin, zmax], "N (m)", "Z (m)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for row, ((label, mask), grids) in enumerate(zip(scenarios, all_hits)):
        n_rays = mask.sum()
        for col, ((view_title, extent, xl, yl), grid) in enumerate(zip(views, grids)):
            ax = axes[row, col]
            im = ax.imshow(grid.T, origin="lower", aspect="auto", extent=extent,
                           cmap="hot_r", vmin=0, vmax=vmax, interpolation="bilinear")
            plt.colorbar(im, ax=ax, label="Ray hit count", fraction=0.03, pad=0.02)
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
            ax.set_title(f"{label} ({n_rays} rays) — {view_title}", fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Sensitivity maps: ray hit count per grid cell\n"
        "(shared colour scale — bottom row shows coverage lost by removing accelerometers)",
        fontsize=12,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(config_file: Path) -> argparse.Namespace:
    """Load plotting configuration from a YAML file."""
    if not yaml:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")

    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: expected YAML dictionary, got {type(cfg)}")

    args = argparse.Namespace()
    args.bundle_file = cfg.get("bundle_file", "")
    args.output_pdf = cfg.get("output_pdf", "")
    args.dt_max_us = float(cfg.get("dt_max_us", 150.0))
    args.dt_ylim_us = float(cfg.get("dt_ylim_us", 50.0))
    args.smooth_window = int(cfg.get("smooth_window", 12))
    args.vmax_pct = float(cfg.get("vmax_pct", 95.0))
    args.source_boreholes = cfg.get("source_boreholes", DEFAULT_SOURCE_BOREHOLES)
    args.sources_csv = cfg.get("sources_csv", None)
    args.receivers_csv = cfg.get("receivers_csv", None)
    args.per_pair_output_pdf = cfg.get("per_pair_output_pdf", "")
    args.injection_csv = cfg.get("injection_csv", "")
    args.despike_mad_thresh = float(cfg.get("despike_mad_thresh", 5.0))

    if not args.bundle_file:
        raise ValueError("Missing required config key: bundle_file")
    if not args.output_pdf:
        raise ValueError("Missing required config key: output_pdf")

    return args


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example usage:\n"
            "  cussp_cassm_plot_results.py --config cussp_cassm_plot_config.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    return p

def main():
    cli_args = build_arg_parser().parse_args()
    args = load_config(cli_args.config)

    if isinstance(args.source_boreholes, str):
        src_bh = [s.strip() for s in args.source_boreholes.split(",") if s.strip()]
    else:
        src_bh = [str(s).strip() for s in args.source_boreholes if str(s).strip()]

    print(f"Loading {args.bundle_file} ...")
    b = np.load(args.bundle_file, allow_pickle=True)
    n_src = int(b["n_sources"])
    n_rec = int(b["n_receivers"])
    dt_raw      = b["dt_us"].astype(np.float32)
    cc_raw             = b["xcorr_peak_cc"].astype(np.float32) if "xcorr_peak_cc" in b.files else None
    env_cc_raw         = b["envelope_peak_cc"].astype(np.float32) if "envelope_peak_cc" in b.files else None
    env_lag_raw        = b["envelope_lag_us"].astype(np.float32) if "envelope_lag_us" in b.files else None
    env_smooth_lag_raw = b["envelope_smooth_lag_us"].astype(np.float32) if "envelope_smooth_lag_us" in b.files else None
    rms_raw            = b["rms"].astype(np.float32)
    cf_raw      = b["centfreq"].astype(np.float32)

    t_days, t0_str, t_num = t_axis(b["epoch_times"])

    # Active pairs: waveform-backed finite dt only.
    # No fallback behavior: if none exist, fail fast with diagnostics.
    rms_has_data = (rms_raw > 0)
    finite_dt = np.isfinite(dt_raw)
    active_mask = np.any(finite_dt & rms_has_data, axis=1)

    active_idxs = np.where(active_mask)[0]
    if active_idxs.size == 0:
        n_pairs_any_finite_dt = int(np.sum(np.any(finite_dt, axis=1)))
        n_pairs_any_rms = int(np.sum(np.any(rms_has_data, axis=1)))
        raise RuntimeError(
            "No active pairs found with finite dt on nonzero-RMS epochs. "
            f"pairs(any finite dt)={n_pairs_any_finite_dt}, "
            f"pairs(any rms>0)={n_pairs_any_rms}."
        )

    if "xcorr_edge_hit" in b.files and "xcorr_peak_cc" in b.files:
        eh = b["xcorr_edge_hit"].astype(np.int8)
        cc = b["xcorr_peak_cc"].astype(np.float32)
        rms_nz = (rms_raw != 0)
        if np.any(rms_nz):
            edge_frac_valid = float(np.mean(eh[rms_nz] != 0))
            cc_med_valid = float(np.nanmedian(cc[rms_nz]))
            print(
                "DIAG: on nonzero-RMS cells: edge_hit_frac="
                f"{edge_frac_valid:.3f}, median_peak_cc={cc_med_valid:.3f}"
            )
            if edge_frac_valid > 0.95:
                print(
                    "DIAG: nearly all xcorr picks are at lag-window edges; "
                    "increase xcorr.max_lag_ms and/or relax edge_guard_samples in processing config."
                )
    print(f"Active pairs: {len(active_idxs)}")

    pair_src    = active_idxs // n_rec
    pair_rec    = active_idxs % n_rec
    pair_src_bh = [src_bh[s] for s in pair_src]
    pair_rec_bh = [get_rec_borehole(r) for r in pair_rec]
    pair_is_hydro = pair_rec >= 48
    src_labels = build_source_labels(src_bh, n_src)
    rec_labels = build_receiver_labels(n_rec)
    pair_src_name = [src_labels[s] for s in pair_src]
    pair_rec_name = [rec_labels[r] for r in pair_rec]

    dt      = dt_raw[active_idxs, :].astype(np.float32).copy()
    cc      = cc_raw[active_idxs, :].copy() if cc_raw is not None else None
    env_cc         = env_cc_raw[active_idxs, :].copy() if env_cc_raw is not None else None
    env_lag        = env_lag_raw[active_idxs, :].copy() if env_lag_raw is not None else None
    env_smooth_lag = env_smooth_lag_raw[active_idxs, :].copy() if env_smooth_lag_raw is not None else None
    rms            = rms_raw[active_idxs, :].copy(); rms[rms == 0] = np.nan
    cf      = cf_raw[active_idxs, :].copy();  cf[cf == 0]   = np.nan
    if env_cc_raw is not None:
        print(f"Envelope cc loaded: shape {env_cc.shape}, "
              f"finite fraction {np.mean(np.isfinite(env_cc)):.2%}")
    else:
        print("Envelope cc not found in bundle (processed without envelope_guide mode)")
    injection = load_injection_series(args.injection_csv)

    print(f"Writing {args.output_pdf} ...")
    with PdfPages(args.output_pdf) as pdf:

        # Page 1: full dt heatmap (all pairs)
        fig = plot_dt_heatmap(dt, t_num, pair_src_bh, pair_rec_bh,
                              args.vmax_pct,
                              f"dt (µs) — {len(active_idxs)} pairs × {len(t_days)} epochs  "
                              f"[sorted by source bh then receiver bh]")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  Page 1: dt heatmap")

        # Page 2: dt heatmap split by sensor type
        fig = plot_sensor_heatmaps(dt, t_num, pair_is_hydro, args.vmax_pct)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  Page 2: dt heatmap accel vs hydro")

        # Pages 3–6: per source borehole time series
        for page, sbh in enumerate(BH_ORDER, start=3):
            fig = plot_per_sourcebh(dt, t_num, pair_src_bh, pair_rec_bh,
                                    sbh, args.dt_ylim_us, args.smooth_window)
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            print(f"  Page {page}: dt per pair — src:{sbh}")

        # Page 7: 4×5 borehole crossing grid
        fig = plot_bh_crossing_grid(dt, t_num, pair_src_bh, pair_rec_bh,
                                    args.dt_ylim_us, args.smooth_window)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  Page 7: borehole crossing grid")

        # Page 8: RMS by sensor type
        fig = plot_metric_by_sensor(rms, t_num, pair_is_hydro,
                                    args.smooth_window, "RMS amplitude", "a.u.")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  Page 8: RMS")

        # Page 9: Centroid frequency by sensor type
        fig = plot_metric_by_sensor(cf, t_num, pair_is_hydro,
                                    args.smooth_window, "Centroid frequency", "kHz")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  Page 9: centroid frequency")

        # Page 10: Statistics grid (std dev + mean dt per borehole crossing)
        fig = plot_stats_grid(dt, pair_src_bh, pair_rec_bh)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  Page 10: statistics grid")

        # Pages 11-12: 3-D coverage and sensitivity maps (only if geometry CSVs supplied)
        if args.sources_csv and args.receivers_csv:
            src_xyz, rec_xyz = _load_geometry_csvs(
                args.sources_csv, args.receivers_csv, src_bh, n_rec=n_rec
            )
            fig = plot_3d_coverage(src_xyz, rec_xyz, active_idxs,
                                   pair_is_hydro, pair_src_bh, n_rec=n_rec)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            print("  Page 11: 3-D ray coverage")

            fig = plot_sensitivity_maps(src_xyz, rec_xyz, active_idxs,
                                        pair_is_hydro, n_rec=n_rec)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            print("  Page 12: sensitivity maps")

    print(f"\nDone → {args.output_pdf}")

    if args.per_pair_output_pdf:
        per_pair_pdf = Path(args.per_pair_output_pdf)
        per_pair_pdf.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {per_pair_pdf} ...")
        with PdfPages(per_pair_pdf) as pdf:
            for fig in plot_each_pair_timeseries(
                    dt=dt,
                    cc=cc,
                    cf=cf,
                    t_num=t_num,
                    pair_src=pair_src,
                    pair_rec=pair_rec,
                    pair_src_name=pair_src_name,
                    pair_rec_name=pair_rec_name,
                    dt_ylim_us=args.dt_ylim_us,
                    pair_index_full=active_idxs,
                    injection=injection,
                    env_cc=env_cc,
                    env_lag_us=env_lag,
                    env_smooth_lag_us=env_smooth_lag,
                ):
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        print(f"Done → {per_pair_pdf}")


if __name__ == "__main__":
    main()
