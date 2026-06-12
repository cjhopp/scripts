#!/usr/bin/env python3
"""Plot top-scoring CASSM source-receiver pairs in 3-D.

Uses score table written by dt diagnostics (top_pair_scores.csv), maps each pair
onto Tx/Rx geometry, and renders 3D line plots highlighting the strongest
coherent dt responders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cussp_cassm_ttcr_inversion import _build_active_pair_geometry


def _load_trajs(wellbore_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    ft2m = 0.3048
    for f in sorted(wellbore_dir.glob("One_foot_interval_well_trajectory_E2_*_in_HMC.csv")):
        parts = f.stem.split("_")
        if "E2" not in parts:
            continue
        try:
            name = parts[parts.index("E2") + 1]
        except Exception:
            continue
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        xc = "x" if "x" in df.columns else "x_ft"
        yc = "y" if "y" in df.columns else "y_ft"
        zc = "z" if "z" in df.columns else "z_ft"
        out[name] = df[[xc, yc, zc]].to_numpy(float) * ft2m
    return out


def _build_pair_geometry(bundle_file: Path, sources_csv: Path, receivers_csv: Path, n_rec: int):
    bundle = np.load(bundle_file, allow_pickle=True)
    dt_us = bundle["dt_us"].astype(float)
    src_bh = "AML,AML,AML,AML,AMU,AMU,AMU,AMU,DML,DML,DML,DML,DMU,DMU,DMU,DMU".split(",")
    active_idxs, tx, rx = _build_active_pair_geometry(
        dt_us=dt_us,
        sources_csv=sources_csv,
        receivers_csv=receivers_csv,
        src_bh_list=src_bh,
        n_rec=n_rec,
        min_valid_epochs_per_pair=1,
    )
    g = {int(pidx): (tx[i], rx[i]) for i, pidx in enumerate(active_idxs)}
    return g


def _plot_one(ax, df: pd.DataFrame, geom_map: dict[int, tuple[np.ndarray, np.ndarray]],
              trajs: dict[str, np.ndarray], title: str):
    # wells as context
    for name, xyz in trajs.items():
        color = "steelblue" if name.startswith("T") else "k"
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=0.9, alpha=0.45)

    # pairs
    s = df["score"].to_numpy(float)
    if len(s) == 0:
        return
    smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
    if smax <= smin:
        smax = smin + 1.0

    for _, row in df.iterrows():
        pidx = int(row["pair_index"])
        if pidx not in geom_map:
            continue
        tx, rx = geom_map[pidx]
        score = float(row["score"])
        u = (score - smin) / (smax - smin)
        color = plt.cm.plasma(u)
        lw = 0.7 + 2.8 * u
        ax.plot([tx[0], rx[0]], [tx[1], rx[1]], [tx[2], rx[2]], color=color, lw=lw, alpha=0.95)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Elevation (m)")
    ax.view_init(elev=24, azim=-58)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot top-scoring CASSM pairs in 3-D")
    p.add_argument("--score-csv", default="/home/chopp/cassm_local/inversion/live/top_pair_scores.csv")
    p.add_argument("--bundle-file", default="/home/chopp/cassm_local/live/cassm_dashboard_bundle_full.npz")
    p.add_argument("--sources-csv", default="/home/chopp/cassm_local/inversion/input/sources_hmc.csv")
    p.add_argument("--receivers-csv", default="/home/chopp/cassm_local/inversion/input/receivers_hmc.csv")
    p.add_argument("--wellbore-dir", default="/media/chopp/HDD1/chet-collab/boreholes/4100/Borehole-trajectories-in-hmc_1ft-spacing")
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--out-dir", default="/home/chopp/cassm_local/inversion/live/top_signal_pairs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.score_csv)
    df = df[np.isfinite(df["score"])].copy()
    df = df.sort_values("score", ascending=False)

    top_all = df.head(args.top_n)
    top_h = df[df["is_hydro"]].head(args.top_n)
    top_a = df[~df["is_hydro"]].head(max(20, args.top_n // 2))

    geom = _build_pair_geometry(
        bundle_file=Path(args.bundle_file),
        sources_csv=Path(args.sources_csv),
        receivers_csv=Path(args.receivers_csv),
        n_rec=72,
    )
    trajs = _load_trajs(Path(args.wellbore_dir))

    # 3-panel summary
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    _plot_one(ax1, top_all, geom, trajs, f"Top {len(top_all)} pairs by score (all)")
    _plot_one(ax2, top_h, geom, trajs, f"Top {len(top_h)} hydrophone pairs")
    _plot_one(ax3, top_a, geom, trajs, f"Top {len(top_a)} accelerometer pairs")

    fig.tight_layout()
    out_png = out_dir / "top_signal_pairs_3d.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    # CSV exports for quick inspection
    top_all.to_csv(out_dir / "top_pairs_all.csv", index=False)
    top_h.to_csv(out_dir / "top_pairs_hydro.csv", index=False)
    top_a.to_csv(out_dir / "top_pairs_accel.csv", index=False)

    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
