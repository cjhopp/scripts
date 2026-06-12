"""
cussp_cassm_plot_inversion.py
Visualise the output of cussp_cassm_ttcr_inversion.py.

Produces a multi-page PDF:
  Page 1  – Residual norm vs time (LSQR data misfit per epoch)
  Page 2  – Mean |ΔVp| vs time (spatial average of perturbation)
  Pages 3+ – Plan-view ΔVp snapshots every --slice-every epochs
              (6 panels per page; wells, sensors, TS well string overlaid)
  Last page – Time-averaged ΔVp in plan + 2 cross-sections

All spatial panels use equal-unit axes (1 m = same pixel in every direction).
Sources, borehole sensor strings, and TS01-24 drift string are overlaid on
every spatial panel when --sources-csv / --receivers-csv are supplied.

Usage
-----
conda run -n ttcr_inv python cussp_cassm_plot_inversion.py \
    --results-npz  /home/chopp/cassm_local/inversion/live/ttcr_timelapse_results.npz \
    --output-pdf   /home/chopp/cassm_local/inversion/live/cassm_inversion_results.pdf \
    --sources-csv  /home/chopp/cassm_local/inversion/input/sources_hmc.csv \
    --receivers-csv /home/chopp/cassm_local/inversion/input/receivers_hmc.csv \
    [--slice-every 100]
    [--dvp-clim 200]
    [--z-index -1]
"""

import argparse
import csv
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -- colour palette -----------------------------------------------------------
_BH_COLORS = {
    "AML": "#1f77b4",   # blue
    "AMU": "#17becf",   # cyan
    "DML": "#d62728",   # red
    "DMU": "#ff7f0e",   # orange
}
_TS_COLOR  = "#2ca02c"   # green  -- drift string / TS hydrophones
_TSS_COLOR = "#9467bd"   # purple -- TSS sources (if present)
_SRC_COLOR = "gold"


# -- time axis ----------------------------------------------------------------
def _t_axis(epoch_labels):
    from datetime import datetime

    def _parse(s):
        s = str(s).strip()
        m = re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$", s)
        if m:
            return datetime(*map(int, m.groups()))
        try:
            return np.datetime64(s, "s").astype("datetime64[s]").item()
        except Exception:
            return None

    times = [_parse(lbl) for lbl in epoch_labels]
    t0 = next(t for t in times if t is not None)
    t_days = np.array([
        (t - t0).total_seconds() / 86400.0 if t is not None else np.nan
        for t in times
    ])
    t0_str = t0.strftime("%Y-%m-%d") if isinstance(t0, datetime) else str(t0)[:10]
    return t_days, t0_str, t0


# -- injection data loader ----------------------------------------------------
def _load_injection(csv_path):
    """Load 1-minute injection CSV (Time, PT 503, Net Flow).  Returns DataFrame or None."""
    p = Path(csv_path)
    if not p.exists():
        LOG.warning("Injection CSV not found: %s", p)
        return None
    try:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
        LOG.info("Loaded %d injection rows from %s", len(df), p.name)
        return df
    except Exception as exc:
        LOG.warning("Failed to load injection CSV %s: %s", p, exc)
        return None


def _overlay_injection(ax, inj_df, t0, pressure_col="PT 503", flow_col="Net Flow"):
    """Overlay injection pressure (right y-axis) and flow (dashed, same axis) on ax."""
    if inj_df is None or len(inj_df) == 0:
        return
    from datetime import timezone

    # Align injection times to same t0 as inversion epochs
    t0_ts = pd.Timestamp(t0)
    if inj_df["Time"].dt.tz is not None:
        t0_ts = t0_ts.tz_localize("UTC")
    t_inj_days = (inj_df["Time"] - t0_ts).dt.total_seconds() / 86400.0

    ax2 = ax.twinx()
    if pressure_col in inj_df.columns:
        p_vals = pd.to_numeric(inj_df[pressure_col], errors="coerce")
        ax2.plot(t_inj_days, p_vals,
                 color="firebrick", lw=0.9, alpha=0.7, label=pressure_col)
    if flow_col in inj_df.columns:
        q_vals = pd.to_numeric(inj_df[flow_col], errors="coerce")
        # scale flow to fit on the pressure axis (rough dual overlay)
        p_range = ax2.get_ylim()
        q_min, q_max = q_vals.min(), q_vals.max()
        if q_max > q_min:
            q_scaled = (q_vals - q_min) / (q_max - q_min) * (p_range[1] - p_range[0]) + p_range[0]
        else:
            q_scaled = q_vals
        ax2.plot(t_inj_days, q_scaled,
                 color="steelblue", lw=0.9, alpha=0.5, ls="--", label=f"{flow_col} (scaled)")
    ax2.set_ylabel(f"{pressure_col} (psi)", color="firebrick", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="firebrick", labelsize=7)
    ax2.legend(fontsize=7, loc="upper right")


# -- grid helpers -------------------------------------------------------------
def _unflatten(vec_flat, cx, cy, cz):
    return vec_flat.reshape((cx, cy, cz), order="F")


def _cell_centres(node_arr):
    return 0.5 * (node_arr[:-1] + node_arr[1:])


# -- geometry loading ---------------------------------------------------------
_FT2M = 0.3048


def _load_one_foot_trajectories(wellbore_dir):
    """
    Load One_foot_interval_well_trajectory_E2_<NAME>_in_HMC.csv files.
    All x/y/z values are in feet regardless of column name suffix; convert to m.
    Returns {name: (N,3) float array in HMC metres}.
    """
    wdir = Path(wellbore_dir)
    trajs = {}
    for f in sorted(wdir.glob("One_foot_interval_well_trajectory_E2_*_in_HMC.csv")):
        # extract well name from filename stem
        parts = f.stem.split("_")
        # stem like: One_foot_interval_well_trajectory_E2_TC_in_HMC
        try:
            name = parts[parts.index("E2") + 1]
        except (ValueError, IndexError):
            continue
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            # accept x/x_ft, y/y_ft, z/z_ft
            xc = next(c for c in ("x", "x_ft") if c in df.columns)
            yc = next(c for c in ("y", "y_ft") if c in df.columns)
            zc = next(c for c in ("z", "z_ft") if c in df.columns)
            xyz = df[[xc, yc, zc]].to_numpy(float) * _FT2M
            trajs[name] = xyz
        except Exception as exc:
            LOG.warning("Skipping trajectory %s: %s", f.name, exc)
    return trajs


def _load_geometry(sources_csv, receivers_csv, wellbore_dir=None,
                   tc_inj_depth_ft=142):
    if not (sources_csv and receivers_csv):
        return None

    def _read(path):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    src_rows = _read(sources_csv)
    rec_rows = _read(receivers_csv)

    src_xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])]
                        for r in src_rows])

    bh_pattern  = re.compile(r"^(AML|AMU|DML|DMU)(\d+)$")
    ts_pattern  = re.compile(r"^TS(\d+)$")
    tss_pattern = re.compile(r"^TSS(\d+)$")

    bh_groups = {k: [] for k in ("AML", "AMU", "DML", "DMU")}
    ts_list   = []
    tss_list  = []

    for r in rec_rows:
        rid = r["receiver_id"]
        x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
        m = bh_pattern.match(rid)
        if m:
            bh_groups[m.group(1)].append((int(m.group(2)), x, y, z))
            continue
        m = ts_pattern.match(rid)
        if m:
            ts_list.append((int(m.group(1)), x, y, z))
            continue
        m = tss_pattern.match(rid)
        if m:
            tss_list.append((int(m.group(1)), x, y, z))

    for bh in bh_groups:
        bh_groups[bh].sort(key=lambda t: t[0])
        bh_groups[bh] = np.array([[t[1], t[2], t[3]] for t in bh_groups[bh]])

    ts_list.sort(key=lambda t: t[0])
    ts_xyz = np.array([[t[1], t[2], t[3]] for t in ts_list]) if ts_list else None

    tss_list.sort(key=lambda t: t[0])
    tss_xyz = np.array([[t[1], t[2], t[3]] for t in tss_list]) if tss_list else None

    # full 1-ft trajectory files (T-wells + overrides for A-wells)
    trajs = {}
    inj_point = None
    if wellbore_dir and Path(wellbore_dir).exists():
        trajs = _load_one_foot_trajectories(wellbore_dir)
        LOG.info("Loaded %d full trajectories: %s", len(trajs), sorted(trajs))

        # override bh_groups with full A-well trajectories if available
        for bh in list(bh_groups.keys()):
            if bh in trajs:
                bh_groups[bh] = trajs[bh]

        # TC injection point
        if "TC" in trajs:
            tc_xyz = trajs["TC"]
            idx = min(int(round(tc_inj_depth_ft)), len(tc_xyz) - 1)
            inj_point = tc_xyz[idx]  # (3,) metres
            LOG.info("TC injection point at %d ft: HMC (%.1f, %.1f, %.1f) m",
                     tc_inj_depth_ft, *inj_point)

    return dict(
        src_xyz=src_xyz,
        bh_groups=bh_groups,
        ts_xyz=ts_xyz,
        tss_xyz=tss_xyz,
        t_wells={k: v for k, v in trajs.items() if k in ("TC", "TU", "TL", "TN")},
        inj_point=inj_point,
    )


# -- geometry overlay ---------------------------------------------------------
_DIM = {"x": 0, "y": 1, "z": 2}


def _overlay_geometry(ax, geom, dim1, dim2):
    if geom is None:
        return
    d1, d2 = _DIM[dim1], _DIM[dim2]

    # A-well CASSM borehole sensor strings (or full trajectory if available)
    for bh_name, xyz in geom["bh_groups"].items():
        if len(xyz) == 0:
            continue
        color = _BH_COLORS.get(bh_name, "gray")
        ax.plot(xyz[:, d1], xyz[:, d2],
                "-", color=color, lw=1.5, zorder=4, alpha=0.85)
        # mark sensor positions only when it's the short 4-point version
        if len(xyz) <= 8:
            ax.scatter(xyz[:, d1], xyz[:, d2],
                       c=color, s=18, zorder=5, edgecolors="k",
                       linewidths=0.4)

    # T-wells (injection / monitoring)
    for tw_name, xyz in geom.get("t_wells", {}).items():
        ax.plot(xyz[:, d1], xyz[:, d2],
                "--", color="steelblue", lw=1.2, zorder=3, alpha=0.75)
        # collar marker
        ax.scatter(xyz[0, d1], xyz[0, d2],
                   marker="s", c="steelblue", s=20, zorder=5,
                   edgecolors="k", linewidths=0.4)

    # TS well (hydrophone string)
    ts = geom["ts_xyz"]
    if ts is not None and len(ts):
        ax.plot(ts[:, d1], ts[:, d2],
                "-", color=_TS_COLOR, lw=1.0, zorder=3, alpha=0.7)
        ax.scatter(ts[:, d1], ts[:, d2],
                   c=_TS_COLOR, s=8, zorder=4, linewidths=0)

    # TSS sources
    tss = geom["tss_xyz"]
    if tss is not None and len(tss):
        ax.scatter(tss[:, d1], tss[:, d2],
                   marker="^", c=_TSS_COLOR, s=30, zorder=5,
                   edgecolors="k", linewidths=0.5)

    # piezo sources
    src = geom["src_xyz"]
    if src is not None and len(src):
        ax.scatter(src[:, d1], src[:, d2],
                   marker="*", c=_SRC_COLOR, s=70, zorder=6,
                   edgecolors="k", linewidths=0.5)

    # TC injection point
    inj = geom.get("inj_point")
    if inj is not None:
        ax.scatter(inj[d1], inj[d2],
                   marker="D", c="red", s=60, zorder=7,
                   edgecolors="k", linewidths=0.7)


def _geom_legend_handles(geom):
    if geom is None:
        return []
    handles = []
    for bh_name, color in _BH_COLORS.items():
        if bh_name in geom["bh_groups"] and len(geom["bh_groups"][bh_name]):
            handles.append(Line2D([0], [0], color=color, lw=1.5, label=bh_name))
    if geom.get("t_wells"):
        handles.append(Line2D([0], [0], color="steelblue", lw=1.2, ls="--",
                               label="T-wells"))
    if geom.get("ts_xyz") is not None:
        handles.append(Line2D([0], [0], color=_TS_COLOR, lw=1.5, label="TS well"))
    if geom.get("src_xyz") is not None:
        handles.append(Line2D([0], [0], marker="*", color=_SRC_COLOR, lw=0,
                               ms=9, mec="k", mew=0.5, label="Sources"))
    if geom.get("inj_point") is not None:
        handles.append(Line2D([0], [0], marker="D", color="red", lw=0,
                               ms=7, mec="k", mew=0.7, label="TC inj. point"))
    return handles


# -- single raster panel ------------------------------------------------------
def _slice_panel(ax, X, Y, img, clim, xlabel, ylabel, title, geom,
                 dim1, dim2, cmap="seismic"):
    vmin, vmax = clim
    im = ax.pcolormesh(X, Y, img, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="nearest", zorder=1)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    _overlay_geometry(ax, geom, dim1, dim2)
    return im


# -- page: residual norm ------------------------------------------------------
def page_residuals(pdf, t_days, t0_str, residual_norm, t0=None, inj_df=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_days, residual_norm, lw=0.8, color="steelblue")
    ax.set_xlabel(f"Days since {t0_str}")
    ax.set_ylabel("LSQR residual norm")
    ax.set_title("Data misfit per epoch")
    ax.grid(True, lw=0.4, alpha=0.5)
    if t0 is not None and inj_df is not None:
        _overlay_injection(ax, inj_df, t0)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    LOG.info("Saved page: residual norm")


# -- page: mean |DVp| ---------------------------------------------------------
def page_mean_dvp(pdf, t_days, t0_str, dvp, t0=None, inj_df=None):
    mean_dvp = np.mean(np.abs(dvp), axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_days, mean_dvp, lw=0.8, color="darkorange")
    ax.set_xlabel(f"Days since {t0_str}")
    ax.set_ylabel("|DVp| mean (m/s)")
    ax.set_title("Spatially averaged |DVp| per epoch  (hit-masked cells only)")
    ax.grid(True, lw=0.4, alpha=0.5)
    if t0 is not None and inj_df is not None:
        _overlay_injection(ax, inj_df, t0)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    LOG.info("Saved page: mean |DVp|")


# -- page: time-mean 3-view map -----------------------------------------------
def page_time_mean_map(pdf, xc, yc, zc, dvp, cx, cy, cz, dvp_clim, z_index, geom):
    dvp_mean = np.mean(dvp, axis=1)
    grid = _unflatten(dvp_mean, cx, cy, cz)

    iz = z_index if z_index >= 0 else cz // 2
    iy = cy // 2
    ix = cx // 2
    clim = (-float(dvp_clim), float(dvp_clim))

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Time-averaged DVp (m/s)", fontsize=11)

    Xxy, Yxy = np.meshgrid(xc, yc, indexing="ij")
    im0 = _slice_panel(axes[0], Xxy, Yxy, grid[:, :, iz], clim,
                       "X - HMC easting (m)", "Y - HMC northing (m)",
                       f"Plan view  Z = {zc[iz]:.1f} m", geom, "x", "y")

    Xxz, Zxz = np.meshgrid(xc, zc, indexing="ij")
    im1 = _slice_panel(axes[1], Xxz, Zxz, grid[:, iy, :], clim,
                       "X - HMC easting (m)", "Z - elevation (m)",
                       f"E-W cross-section  Y = {yc[iy]:.1f} m", geom, "x", "z")

    Xyz, Zyz = np.meshgrid(yc, zc, indexing="ij")
    im2 = _slice_panel(axes[2], Xyz, Zyz, grid[ix, :, :], clim,
                       "Y - HMC northing (m)", "Z - elevation (m)",
                       f"N-S cross-section  X = {xc[ix]:.1f} m", geom, "y", "z")

    for im, ax in zip([im0, im1, im2], axes):
        fig.colorbar(im, ax=ax, label="DVp (m/s)", fraction=0.046, pad=0.04)

    handles = _geom_legend_handles(geom)
    if handles:
        axes[0].legend(handles=handles, fontsize=6, loc="lower right",
                       framealpha=0.7, markerscale=0.9)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    LOG.info("Saved page: time-mean map")


# -- pages: epoch plan-view snapshots -----------------------------------------
def pages_epoch_slices(pdf, t_days, t0_str, xc, yc, zc, dvp, cx, cy, cz,
                       dvp_clim, z_index, slice_every, geom):
    n_epochs = dvp.shape[1]
    iz = z_index if z_index >= 0 else cz // 2
    clim = (-float(dvp_clim), float(dvp_clim))
    Xxy, Yxy = np.meshgrid(xc, yc, indexing="ij")

    epoch_indices = list(range(0, n_epochs, slice_every))
    LOG.info("Generating %d epoch slice pages (every %d epochs)...",
             len(epoch_indices), slice_every)

    geom_handles = _geom_legend_handles(geom)

    n_per_page = 6
    for start in range(0, len(epoch_indices), n_per_page):
        batch = epoch_indices[start:start + n_per_page]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.ravel()

        for k, e in enumerate(batch):
            grid = _unflatten(dvp[:, e], cx, cy, cz)
            _slice_panel(axes[k], Xxy, Yxy, grid[:, :, iz], clim,
                         "X (m)", "Y (m)",
                         f"Epoch {e}  -  Day {t_days[e]:.1f}",
                         geom, "x", "y")

        for k in range(len(batch), n_per_page):
            axes[k].set_visible(False)

        sm = plt.cm.ScalarMappable(
            cmap="seismic", norm=mcolors.Normalize(vmin=clim[0], vmax=clim[1]))
        sm.set_array([])
        visible_axes = [axes[k] for k in range(len(batch))]
        fig.colorbar(sm, ax=visible_axes, label="DVp (m/s)",
                     fraction=0.018, pad=0.02)

        fig.suptitle(
            f"Plan-view DVp at Z = {zc[iz]:.1f} m  (days since {t0_str})",
            fontsize=10)

        if geom_handles:
            axes[0].legend(handles=geom_handles, fontsize=5.5, loc="lower right",
                           framealpha=0.7, markerscale=0.8)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    LOG.info("Saved epoch slice pages")


# -- main ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Visualise CASSM timelapse inversion results")
    p.add_argument("--results-npz",
                   default="/home/chopp/cassm_local/inversion/live/ttcr_timelapse_results.npz")
    p.add_argument("--output-pdf",
                   default="/home/chopp/cassm_local/inversion/live/cassm_inversion_results.pdf")
    p.add_argument("--sources-csv",
                   default="/home/chopp/cassm_local/inversion/input/sources_hmc.csv")
    p.add_argument("--receivers-csv",
                   default="/home/chopp/cassm_local/inversion/input/receivers_hmc.csv")
    p.add_argument("--wellbore-dir",
                   default="/media/chopp/HDD1/chet-collab/boreholes/4100/"
                           "Borehole-trajectories-in-hmc_1ft-spacing",
                   help="Directory of One_foot_interval_well_trajectory_E2_*.csv files")
    p.add_argument("--tc-inj-depth-ft", type=float, default=142.0,
                   help="TC injection depth in feet along borehole (default 142)")
    p.add_argument("--injection-csv",
                   default="/media/chopp/HDD1/chet-cussp/raw-injection/live/latest_INJ_data_1min.csv",
                   help="1-minute injection CSV (Time, PT 503, Net Flow)")
    p.add_argument("--slice-every", type=int, default=100)
    p.add_argument("--dvp-clim", type=float, default=200.0)
    p.add_argument("--z-index", type=int, default=-1)
    args = p.parse_args()

    npz_path = Path(args.results_npz)
    if not npz_path.exists():
        LOG.error("Results file not found: %s", npz_path)
        return 1

    LOG.info("Loading %s ...", npz_path)
    r = np.load(npz_path, allow_pickle=True)

    epoch_labels  = r["epoch_labels"]
    x = r["x"]; y = r["y"]; z_nodes = r["z"]
    cx = int(r["nx"]); cy = int(r["ny"]); cz = int(r["nz"])
    dvp           = r["dvp"]
    residual_norm = r["residual_norm"]
    vp_bg         = float(r["vp_background"])

    n_cells, n_epochs = dvp.shape
    LOG.info("Grid: %d x %d x %d cells, %d epochs, Vp_bg=%.0f m/s",
             cx, cy, cz, n_epochs, vp_bg)

    xc = _cell_centres(x)
    yc = _cell_centres(y)
    zc = _cell_centres(z_nodes)

    t_days, t0_str, t0 = _t_axis(epoch_labels)

    geom = _load_geometry(args.sources_csv, args.receivers_csv,
                          wellbore_dir=args.wellbore_dir,
                          tc_inj_depth_ft=int(args.tc_inj_depth_ft))
    if geom is not None:
        LOG.info("Geometry: %d sources, A-wells %s, T-wells %s, %d TS sensors",
                 len(geom["src_xyz"]),
                 list(geom["bh_groups"].keys()),
                 list(geom["t_wells"].keys()),
                 len(geom["ts_xyz"]) if geom["ts_xyz"] is not None else 0)

    out_path = Path(args.output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inj_df = _load_injection(args.injection_csv) if args.injection_csv else None

    with PdfPages(out_path) as pdf:
        page_residuals(pdf, t_days, t0_str, residual_norm, t0=t0, inj_df=inj_df)
        page_mean_dvp(pdf, t_days, t0_str, dvp, t0=t0, inj_df=inj_df)
        pages_epoch_slices(pdf, t_days, t0_str, xc, yc, zc, dvp, cx, cy, cz,
                           args.dvp_clim, args.z_index, args.slice_every, geom)
        page_time_mean_map(pdf, xc, yc, zc, dvp, cx, cy, cz,
                           args.dvp_clim, args.z_index, geom)

    LOG.info("Written: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
