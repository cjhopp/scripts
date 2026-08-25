#!/usr/bin/env python3
"""Create per-epoch CASSM raypath movies colored by dt.

This is a lightweight, geometry-first "poor man's inversion" visualization:
for each epoch, ray segments are colored by measured dt (microseconds) and
shown in three synchronized panels:
  1) Plan view (E-N)
  2) E-Z cross-section
  3) N-Z cross-section

Wells and receivers are always shown as static context overlays.

Example:
  python cussp_cassm_movie_raypath_per_epoch.py \
      --bundle-file /home/chopp/cassm_local/live/cassm_dashboard_bundle_full.npz \
      --sources-csv /home/chopp/cassm_local/inversion/input/sources_hmc.csv \
      --receivers-csv /home/chopp/cassm_local/inversion/input/receivers_hmc.csv \
      --wellbore-dir /media/chopp/HDD1/chet-collab/boreholes/4100/Borehole-trajectories-in-hmc_1ft-spacing \
      --output-dir /home/chopp/cassm_local/live/raypath_movies \
      --output-mode both --opacity-mode xcorr --stride 5 --fps 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors as mcolors
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter
import numpy as np
import pandas as pd

from cussp_cassm_ttcr_inversion import _build_active_pair_geometry

LOG = logging.getLogger("cussp_cassm_movie_raypath_per_epoch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_trajs(wellbore_dir: Optional[Path]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if not wellbore_dir:
        return out
    if not wellbore_dir.exists():
        LOG.warning("Well trajectory directory does not exist: %s", wellbore_dir)
        return out

    ft2m = 0.3048
    for f in sorted(wellbore_dir.glob("One_foot_interval_well_trajectory_E2_*_in_HMC.csv")):
        parts = f.stem.split("_")
        if "E2" not in parts:
            continue
        try:
            name = parts[parts.index("E2") + 1]
        except Exception:
            continue
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            xc = "x" if "x" in df.columns else "x_ft"
            yc = "y" if "y" in df.columns else "y_ft"
            zc = "z" if "z" in df.columns else "z_ft"
            out[name] = df[[xc, yc, zc]].to_numpy(float) * ft2m
        except Exception as exc:
            LOG.warning("Skipping trajectory %s: %s", f.name, exc)
    return out


def _receiver_xyz_from_csv(receivers_csv: Path, n_rec: int) -> np.ndarray:
    rec_by_id: Dict[str, np.ndarray] = {}
    with open(receivers_csv, newline="") as fh:
        import csv as _csv
        for row in _csv.DictReader(fh):
            rid = str(row["receiver_id"]).strip()
            rec_by_id[rid] = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)

    accel_bh = ["AML", "AMU", "DML", "DMU"]
    rec_xyz = np.full((n_rec, 3), np.nan, dtype=float)
    for rec_idx in range(n_rec):
        ch = rec_idx + 1
        if ch <= 48:
            bh = accel_bh[(ch - 1) // 12]
            sensor_in_bh = ((ch - 1) % 12) // 3
            rid = f"{bh}{sensor_in_bh + 1}"
        else:
            rid = f"TS{ch - 48:02d}"
        if rid in rec_by_id:
            rec_xyz[rec_idx] = rec_by_id[rid]
    return rec_xyz


def _compute_alpha(metric: np.ndarray, mode: str, alpha_min: float, alpha_max: float) -> np.ndarray:
    if mode == "fixed":
        return np.full(metric.shape, alpha_max, dtype=float)

    alpha = np.full(metric.shape, alpha_min, dtype=float)
    finite = np.isfinite(metric)
    if np.any(finite):
        alpha[finite] = alpha_min + (alpha_max - alpha_min) * np.clip(metric[finite], 0.0, 1.0)
    return alpha


def _smooth_dt_us(dt_us: np.ndarray, window: int) -> np.ndarray:
    """Apply per-pair rolling-median smoothing across epochs for visualization.

    Uses a centered window and ignores NaNs. Window=1 disables smoothing.
    """
    w = max(int(window), 1)
    if w <= 1:
        return dt_us
    sm = pd.DataFrame(dt_us.T).rolling(window=w, min_periods=1, center=True).median()
    return sm.to_numpy(dtype=float).T


def _make_segments(a: np.ndarray, b: np.ndarray, i0: int, i1: int) -> np.ndarray:
    return np.stack([a[:, [i0, i1]], b[:, [i0, i1]]], axis=1)


def _draw_static_context(
    axs: np.ndarray,
    trajs: Dict[str, np.ndarray],
    rec_xyz: np.ndarray,
) -> None:
    plan_ax, ez_ax, nz_ax = axs

    for name, xyz in trajs.items():
        color = "steelblue" if name.startswith("T") else "0.3"
        plan_ax.plot(xyz[:, 0], xyz[:, 1], color=color, lw=0.8, alpha=0.5)
        ez_ax.plot(xyz[:, 0], xyz[:, 2], color=color, lw=0.8, alpha=0.5)
        nz_ax.plot(xyz[:, 1], xyz[:, 2], color=color, lw=0.8, alpha=0.5)

    good = ~np.any(np.isnan(rec_xyz), axis=1)
    rec = rec_xyz[good]
    is_h = np.where(good)[0] >= 48

    if rec.size > 0:
        plan_ax.scatter(rec[~is_h, 0], rec[~is_h, 1], s=12, c="tab:blue", alpha=0.7, marker="o", label="Accel")
        plan_ax.scatter(rec[is_h, 0], rec[is_h, 1], s=14, c="tab:purple", alpha=0.8, marker="D", label="Hydro")

        ez_ax.scatter(rec[~is_h, 0], rec[~is_h, 2], s=10, c="tab:blue", alpha=0.6, marker="o")
        ez_ax.scatter(rec[is_h, 0], rec[is_h, 2], s=12, c="tab:purple", alpha=0.7, marker="D")

        nz_ax.scatter(rec[~is_h, 1], rec[~is_h, 2], s=10, c="tab:blue", alpha=0.6, marker="o")
        nz_ax.scatter(rec[is_h, 1], rec[is_h, 2], s=12, c="tab:purple", alpha=0.7, marker="D")

    plan_ax.set_xlabel("Easting (m)")
    plan_ax.set_ylabel("Northing (m)")
    plan_ax.set_title("Plan (E-N)")

    ez_ax.set_xlabel("Easting (m)")
    ez_ax.set_ylabel("Elevation (m)")
    ez_ax.set_title("E-Z")

    nz_ax.set_xlabel("Northing (m)")
    nz_ax.set_ylabel("Elevation (m)")
    nz_ax.set_title("N-Z")

    for ax in axs:
        ax.grid(True, alpha=0.25)

    # Enforce equal-unit scaling so geometry is not visually warped.
    plan_ax.set_aspect("equal", adjustable="box")
    ez_ax.set_aspect("equal", adjustable="box")
    nz_ax.set_aspect("equal", adjustable="box")


def _load_injection_series(injection_csv: str | None):
    """Load injection data as matplotlib datenums + pressure/flow arrays."""
    if not injection_csv:
        return None
    path = Path(injection_csv)
    if not path.exists():
        LOG.warning("Injection CSV not found: %s (skipping panel)", path)
        return None
    try:
        df = pd.read_csv(path)
        if "Time" not in df.columns:
            LOG.warning("Injection CSV missing 'Time' column: %s (skipping panel)", path)
            return None
        t = pd.to_datetime(df["Time"], utc=True, errors="coerce")
        ok = t.notna()
        if ok.sum() < 2:
            LOG.warning("Injection CSV has insufficient valid timestamps: %s (skipping panel)", path)
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
        LOG.warning("Failed loading injection CSV %s: %s (skipping panel)", path, exc)
        return None


def _format_time_axis_full_utc(ax) -> None:
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax.tick_params(axis="x", labelrotation=30)


def _epoch_t_num(bundle: Dict[str, np.ndarray], n_epochs: int) -> np.ndarray:
    if "epoch_times" in bundle:
        t = pd.to_datetime([str(x) for x in bundle["epoch_times"].tolist()], utc=True, errors="coerce")
    elif "epoch_labels" in bundle:
        t = pd.to_datetime([str(x) for x in bundle["epoch_labels"].tolist()], utc=True, errors="coerce")
    else:
        return np.full(n_epochs, np.nan, dtype=float)

    if len(t) != n_epochs:
        return np.full(n_epochs, np.nan, dtype=float)

    ok = pd.notna(t)
    out = np.full(n_epochs, np.nan, dtype=float)
    if np.any(ok):
        # pd.to_datetime(list-like) returns a DatetimeIndex, which does not expose `.dt`.
        out[ok] = mdates.date2num(t[ok].to_pydatetime())
    return out


def _draw_injection_panel(ax, ax_q, injection, frame_t_num: float) -> None:
    ax.cla()
    ax_q.cla()
    inj_t = injection.get("t_num")
    inj_p = injection.get("pressure")
    inj_q = injection.get("flow")

    p_ok = np.isfinite(inj_p)
    q_ok = np.isfinite(inj_q)

    if p_ok.any():
        ax.plot(inj_t[p_ok], inj_p[p_ok], color="#d62728", lw=1.0, label="PT 503")
    ax.set_ylabel("PT 503", color="#d62728")
    ax.tick_params(axis="y", labelcolor="#d62728")
    ax.grid(True, alpha=0.25)

    if q_ok.any():
        ax_q.plot(inj_t[q_ok], inj_q[q_ok], color="#2ca02c", lw=1.0, label="Net Flow")
    ax_q.set_ylabel("Net Flow", color="#2ca02c")
    ax_q.tick_params(axis="y", labelcolor="#2ca02c")

    if np.isfinite(frame_t_num):
        ax.axvline(frame_t_num, color="k", lw=0.35, alpha=0.9, ls="--", label="Frame UTC")
        ax_q.axvline(frame_t_num, color="k", lw=0.35, alpha=0.9, ls="--")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_q.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)

    _format_time_axis_full_utc(ax)
    ax.set_xlabel("UTC time")
    ax.set_title("Injection Context (PT 503 and Net Flow)")


def _plot_epoch(
    fig: plt.Figure,
    axs_top: np.ndarray,
    ax_inj,
    ax_inj_q,
    tx: np.ndarray,
    rx: np.ndarray,
    dt_epoch: np.ndarray,
    alpha_epoch: np.ndarray,
    norm: mcolors.Normalize,
    cmap,
    trajs: Dict[str, np.ndarray],
    rec_xyz: np.ndarray,
    epoch_label: str,
    injection,
    frame_t_num: float,
) -> None:
    for ax in axs_top:
        ax.cla()

    _draw_static_context(axs_top, trajs, rec_xyz)

    valid = np.isfinite(dt_epoch)
    if np.any(valid):
        txv = tx[valid]
        rxv = rx[valid]
        dtv = dt_epoch[valid]
        av = alpha_epoch[valid]

        rgba = cmap(norm(dtv))
        rgba[:, 3] = av

        seg_plan = _make_segments(txv, rxv, 0, 1)
        seg_ez = _make_segments(txv, rxv, 0, 2)
        seg_nz = _make_segments(txv, rxv, 1, 2)

        for ax, seg in zip(axs_top, (seg_plan, seg_ez, seg_nz)):
            lc = LineCollection(seg, colors=rgba, linewidths=0.9)
            ax.add_collection(lc)

    axs_top[0].legend(loc="lower left", fontsize=8)
    if injection is not None and ax_inj is not None:
        _draw_injection_panel(ax_inj, ax_inj_q, injection=injection, frame_t_num=frame_t_num)
    fig.suptitle(f"Per-epoch raypath dt map | {epoch_label}", fontsize=12)


def _epoch_labels(bundle: Dict[str, np.ndarray], n_epochs: int) -> List[str]:
    if "epoch_times" in bundle:
        times = [str(x) for x in bundle["epoch_times"].tolist()]
        if len(times) == n_epochs:
            return times
    if "epoch_labels" in bundle:
        labels = [str(x) for x in bundle["epoch_labels"].tolist()]
        if len(labels) == n_epochs:
            return labels
    return [f"epoch_{i:06d}" for i in range(n_epochs)]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create per-epoch dt-colored raypath movie")
    p.add_argument("--bundle-file", default="/home/chopp/cassm_local/live/cassm_dashboard_bundle_full.npz")
    p.add_argument("--sources-csv", default="/home/chopp/cassm_local/inversion/input/sources_hmc.csv")
    p.add_argument("--receivers-csv", default="/home/chopp/cassm_local/inversion/input/receivers_hmc.csv")
    p.add_argument("--source-boreholes", default="AML,AML,AML,AML,AMU,AMU,AMU,AMU,DML,DML,DML,DML,DMU,DMU,DMU,DMU")
    p.add_argument("--n-receivers", type=int, default=72)
    p.add_argument("--wellbore-dir", default="")
    p.add_argument("--output-dir", default="/home/chopp/cassm_local/live/raypath_movies")
    p.add_argument("--output-prefix", default="raypath_dt")
    p.add_argument("--output-mode", choices=["png", "mp4", "both"], default="both")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=0, help="0 means all")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--dt-clim-us", type=float, default=0.0, help="0 means auto from percentile")
    p.add_argument("--dt-clim-percentile", type=float, default=99.0)
    p.add_argument("--opacity-mode", choices=["fixed", "xcorr", "envelope"], default="xcorr")
    p.add_argument("--opacity-min", type=float, default=0.15)
    p.add_argument("--opacity-max", type=float, default=0.95)
    p.add_argument("--dt-smooth-window", type=int, default=5, help="Rolling-median window in epochs (1 disables)")
    p.add_argument("--injection-csv", default="/media/chopp/HDD1/chet-cussp/raw-injection/live/latest_INJ_data_1min.csv")
    p.add_argument("--omit-accelerometer-paths", dest="omit_accelerometer_paths", action="store_true", default=True,
                   help="When set (default), only hydrophone raypaths are rendered")
    p.add_argument("--include-accelerometer-paths", dest="omit_accelerometer_paths", action="store_false",
                   help="Render both accelerometer and hydrophone raypaths")
    p.add_argument("--min-valid-epochs-per-pair", type=int, default=1)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = np.load(args.bundle_file, allow_pickle=True)
    dt_us_full = np.asarray(bundle["dt_us"], dtype=float)
    n_pairs, n_epochs_total = dt_us_full.shape
    source_boreholes = [s.strip() for s in str(args.source_boreholes).split(",") if s.strip()]

    active_idxs, tx, rx = _build_active_pair_geometry(
        dt_us=dt_us_full,
        sources_csv=Path(args.sources_csv),
        receivers_csv=Path(args.receivers_csv),
        src_bh_list=source_boreholes,
        n_rec=int(args.n_receivers),
        min_valid_epochs_per_pair=int(args.min_valid_epochs_per_pair),
    )

    dt_us = dt_us_full[active_idxs, :]
    pair_rec_idx = active_idxs % int(args.n_receivers)
    if args.omit_accelerometer_paths:
        hydro_mask = pair_rec_idx >= 48
        active_idxs = active_idxs[hydro_mask]
        tx = tx[hydro_mask]
        rx = rx[hydro_mask]
        dt_us = dt_us[hydro_mask, :]
        pair_rec_idx = pair_rec_idx[hydro_mask]

    n_active = dt_us.shape[0]

    if args.opacity_mode == "xcorr" and "xcorr_peak_cc" in bundle.files:
        opacity_metric = np.asarray(bundle["xcorr_peak_cc"], dtype=float)[active_idxs, :]
    elif args.opacity_mode == "envelope" and "envelope_peak_cc" in bundle.files:
        opacity_metric = np.asarray(bundle["envelope_peak_cc"], dtype=float)[active_idxs, :]
    else:
        opacity_metric = np.full_like(dt_us, np.nan, dtype=float)
        if args.opacity_mode != "fixed":
            LOG.warning("Opacity metric '%s' unavailable in bundle; using minimum alpha fallback.", args.opacity_mode)

    epoch_labels = _epoch_labels(bundle, n_epochs_total)
    epoch_t_num = _epoch_t_num(bundle, n_epochs_total)
    injection = _load_injection_series(args.injection_csv)

    if n_active == 0:
        raise RuntimeError("No active pairs remain after path-family filtering.")

    dt_us = _smooth_dt_us(dt_us, window=int(args.dt_smooth_window))

    max_epochs = int(args.max_epochs) if int(args.max_epochs) > 0 else n_epochs_total
    max_epochs = min(max_epochs, n_epochs_total)
    stride = max(int(args.stride), 1)
    frame_epochs = list(range(0, max_epochs, stride))

    finite_dt = dt_us[np.isfinite(dt_us)]
    if finite_dt.size == 0:
        raise RuntimeError("No finite dt values available for movie rendering.")

    if float(args.dt_clim_us) > 0.0:
        clim = float(args.dt_clim_us)
    else:
        clim = float(np.nanpercentile(np.abs(finite_dt), float(args.dt_clim_percentile)))
        if clim <= 0:
            clim = float(np.nanmax(np.abs(finite_dt)))
        if clim <= 0:
            clim = 1.0

    LOG.info(
        "Rendering %d frames from %d active pairs (clim=+/-%.2f us, smooth_window=%d, omit_accel=%s)",
        len(frame_epochs),
        n_active,
        clim,
        int(args.dt_smooth_window),
        bool(args.omit_accelerometer_paths),
    )

    norm = mcolors.Normalize(vmin=-clim, vmax=clim)
    cmap = cm.get_cmap("RdBu_r")

    rec_xyz = _receiver_xyz_from_csv(Path(args.receivers_csv), int(args.n_receivers))
    trajs = _load_trajs(Path(args.wellbore_dir)) if args.wellbore_dir else {}

    fig = plt.figure(figsize=(16, 8), dpi=int(args.dpi), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.25])
    ax_plan = fig.add_subplot(gs[0, 0])
    ax_ez = fig.add_subplot(gs[0, 1])
    ax_nz = fig.add_subplot(gs[0, 2])
    ax_inj = fig.add_subplot(gs[1, :])
    ax_inj_q = ax_inj.twinx()
    axs_top = np.array([ax_plan, ax_ez, ax_nz], dtype=object)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs_top.ravel().tolist(), shrink=0.9)
    cbar.set_label("dt (us)")

    png_dir = out_dir / f"{args.output_prefix}_frames"
    if args.output_mode in ("png", "both"):
        png_dir.mkdir(parents=True, exist_ok=True)

    def _render_one(epoch_idx: int) -> None:
        dt_epoch = dt_us[:, epoch_idx]
        alpha_epoch = _compute_alpha(
            opacity_metric[:, epoch_idx],
            mode=str(args.opacity_mode),
            alpha_min=float(args.opacity_min),
            alpha_max=float(args.opacity_max),
        )
        _plot_epoch(
            fig=fig,
            axs_top=axs_top,
            ax_inj=ax_inj,
            ax_inj_q=ax_inj_q,
            tx=tx,
            rx=rx,
            dt_epoch=dt_epoch,
            alpha_epoch=alpha_epoch,
            norm=norm,
            cmap=cmap,
            trajs=trajs,
            rec_xyz=rec_xyz,
            epoch_label=str(epoch_labels[epoch_idx]),
            injection=injection,
            frame_t_num=float(epoch_t_num[epoch_idx]) if np.isfinite(epoch_t_num[epoch_idx]) else np.nan,
        )

    if args.output_mode in ("png", "both"):
        for i, eidx in enumerate(frame_epochs):
            _render_one(eidx)
            frame_file = png_dir / f"frame_{i:06d}_epoch_{eidx:06d}.png"
            fig.savefig(frame_file, dpi=int(args.dpi))
            if (i + 1) % 50 == 0 or i == len(frame_epochs) - 1:
                LOG.info("PNG frame %d/%d written", i + 1, len(frame_epochs))

    if args.output_mode in ("mp4", "both"):
        mp4_file = out_dir / f"{args.output_prefix}.mp4"
        try:
            writer = FFMpegWriter(fps=int(args.fps), bitrate=3000)
            with writer.saving(fig, str(mp4_file), int(args.dpi)):
                for i, eidx in enumerate(frame_epochs):
                    _render_one(eidx)
                    writer.grab_frame()
                    if (i + 1) % 50 == 0 or i == len(frame_epochs) - 1:
                        LOG.info("MP4 frame %d/%d written", i + 1, len(frame_epochs))
            LOG.info("Wrote movie: %s", mp4_file)
        except Exception as exc:
            LOG.error("Failed to write MP4 (ffmpeg likely missing): %s", exc)

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
