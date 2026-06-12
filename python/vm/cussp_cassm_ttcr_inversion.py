#!/usr/bin/env python3
"""TTCR-backed CUSSP CASSM timelapse inversion runner.

This script plugs into the existing workflow:
1) Geometry/mask from cussp_cassm_inversion_prep.py
2) Differential delay-times from cassm_dashboard_bundle.npz (dt_us)
3) TTCR kernel G via ttcrpy.rgrid.Grid3d ray tracing
4) Per-epoch solve for slowness perturbation

Solve form:
    [ G ]                 [ dt ]
    [ λD D ] Δs  ~=       [  0 ]
    [ λP P ]              [  0 ]

where P is a diagonal matrix from the prior mask (full-domain by default).
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LOG = logging.getLogger("cussp_cassm_ttcr_inversion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _build_node_gradient_operator(nx: int, ny: int, nz: int):
    """Build first-order finite-difference operator on node-centered model."""
    from scipy.sparse import coo_matrix

    def idx(i, j, k):
        return i + nx * (j + ny * k)

    rows = []
    cols = []
    vals = []
    r = 0

    # x-differences
    for k in range(nz):
        for j in range(ny):
            for i in range(nx - 1):
                a = idx(i, j, k)
                b = idx(i + 1, j, k)
                rows.extend([r, r])
                cols.extend([a, b])
                vals.extend([-1.0, 1.0])
                r += 1

    # y-differences
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx):
                a = idx(i, j, k)
                b = idx(i, j + 1, k)
                rows.extend([r, r])
                cols.extend([a, b])
                vals.extend([-1.0, 1.0])
                r += 1

    # z-differences
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                a = idx(i, j, k)
                b = idx(i, j, k + 1)
                rows.extend([r, r])
                cols.extend([a, b])
                vals.extend([-1.0, 1.0])
                r += 1

    n_model = nx * ny * nz
    D = coo_matrix((vals, (rows, cols)), shape=(r, n_model)).tocsr()
    return D


def _adapt_mask_to_model(mask: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Adapt node/cell mask arrays to cell-centered model size expected by TTCR L."""
    n_node = nx * ny * nz
    cx, cy, cz = nx - 1, ny - 1, nz - 1
    n_cell = cx * cy * cz

    m = np.asarray(mask, dtype=float).reshape(-1)
    if m.size == n_cell:
        return m
    if m.size == n_node:
        m3 = m.reshape((nx, ny, nz), order="C")
        c = (
            m3[:-1, :-1, :-1]
            + m3[1:, :-1, :-1]
            + m3[:-1, 1:, :-1]
            + m3[:-1, :-1, 1:]
            + m3[1:, 1:, :-1]
            + m3[1:, :-1, 1:]
            + m3[:-1, 1:, 1:]
            + m3[1:, 1:, 1:]
        ) / 8.0
        return c.reshape(-1, order="C")
    raise RuntimeError(f"Unsupported mask size {m.size}; expected {n_node} (nodes) or {n_cell} (cells)")


def _load_bundle(bundle_file: Path) -> Dict[str, np.ndarray]:
    obj = np.load(bundle_file, allow_pickle=True)
    return {
        "dt_us": obj["dt_us"],
        "epoch_labels": np.array([str(x) for x in obj["epoch_labels"].tolist()]),
        "n_sources": int(obj["n_sources"]),
        "n_receivers": int(obj["n_receivers"]),
    }


def _load_pair_geometry(pair_file: Path, n_pairs_expected: int) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(pair_file)
    required = {"tx", "ty", "tz", "rx", "ry", "rz"}
    if not required.issubset(df.columns):
        raise ValueError(f"pair geometry missing required columns: {sorted(required)}")

    # Keep deterministic order. If amplifier channel exists, sort sources by channel then receiver id.
    sort_cols = []
    if "amplifier_channel" in df.columns:
        sort_cols.append("amplifier_channel")
    if "source_id" in df.columns:
        sort_cols.append("source_id")
    if "receiver_id" in df.columns:
        sort_cols.append("receiver_id")
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    if len(df) != n_pairs_expected:
        raise ValueError(
            f"pair count mismatch: pair_geometry={len(df)} vs expected={n_pairs_expected}"
        )

    tx = df[["tx", "ty", "tz"]].to_numpy(float)
    rx = df[["rx", "ry", "rz"]].to_numpy(float)
    return tx, rx


def _build_active_pair_geometry(
    dt_us: np.ndarray,
    sources_csv: Path,
    receivers_csv: Path,
    src_bh_list: List[str],
    n_rec: int = 72,
    min_valid_epochs_per_pair: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive source/receiver 3-D coordinates for all active pairs in the bundle.

    Active pairs are those where at least one epoch has a non-zero dt value.
    Returns (active_idxs, tx, rx) — parallel arrays of length K.
    """
    min_valid = max(int(min_valid_epochs_per_pair), 1)
    active_mask = (dt_us != 0).sum(axis=1) >= min_valid
    active_idxs_candidate = np.where(active_mask)[0]

    # --- source coordinates (sorted depth-first per borehole) ---
    src_by_bh: Dict[str, list] = {}
    with open(sources_csv, newline="") as fh:
        for row in _csv.DictReader(fh):
            bh = row.get("borehole", "").strip().upper()
            if not bh:
                continue
            src_by_bh.setdefault(bh, []).append(
                (float(row["depth_m"]), float(row["x"]), float(row["y"]), float(row["z"]))
            )
    for bh in src_by_bh:
        src_by_bh[bh].sort()

    bh_counts: Dict[str, int] = {}
    n_src = len(src_bh_list)
    src_xyz = np.full((n_src, 3), np.nan)
    for i, bh in enumerate(src_bh_list):
        n = bh_counts.get(bh, 0)
        bh_counts[bh] = n + 1
        entries = src_by_bh.get(bh, [])
        if n < len(entries):
            _, x, y, z = entries[n]
            src_xyz[i] = [x, y, z]

    # --- receiver coordinates (channel-indexed) ---
    rec_by_id: Dict[str, np.ndarray] = {}
    with open(receivers_csv, newline="") as fh:
        for row in _csv.DictReader(fh):
            rec_by_id[row["receiver_id"]] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])]
            )

    accel_bh = ["AML", "AMU", "DML", "DMU"]
    rec_xyz = np.full((n_rec, 3), np.nan)
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

    # --- build tx/rx for valid active pairs ---
    valid_idxs = []
    tx_list = []
    rx_list = []
    for pidx in active_idxs_candidate:
        si = int(pidx) // n_rec
        ri = int(pidx) % n_rec
        if si >= n_src:
            continue
        if np.any(np.isnan(src_xyz[si])) or np.any(np.isnan(rec_xyz[ri])):
            continue
        tx_list.append(src_xyz[si])
        rx_list.append(rec_xyz[ri])
        valid_idxs.append(int(pidx))

    if not valid_idxs:
        raise RuntimeError("No active pairs with known geometry found.")

    LOG.info(
        "Active pairs: %d found, %d with valid geometry",
        len(active_idxs_candidate),
        len(valid_idxs),
    )
    return (
        np.array(valid_idxs, dtype=int),
        np.array(tx_list, dtype=float),
        np.array(rx_list, dtype=float),
    )


def _build_sensor_hull_prior(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
    inside_weight: float,
    outside_weight: float,
) -> np.ndarray:
    """Build cell-wise prior weights using the 3-D convex hull of sensor geometry."""
    from scipy.spatial import Delaunay

    cx, cy, cz = len(x) - 1, len(y) - 1, len(z) - 1
    n_model = cx * cy * cz

    pts = np.vstack([tx, rx])
    if pts.shape[0] < 4:
        return np.full(n_model, inside_weight, dtype=float)

    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    zc = 0.5 * (z[:-1] + z[1:])
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing="ij")
    model_pts = np.column_stack(
        [Xc.ravel(order="C"), Yc.ravel(order="C"), Zc.ravel(order="C")]
    )

    try:
        hull = Delaunay(pts)
        inside = hull.find_simplex(model_pts) >= 0
    except Exception as exc:
        LOG.warning("Sensor hull build failed (%s); using uniform prior", exc)
        return np.full(n_model, inside_weight, dtype=float)

    prior = np.full(n_model, outside_weight, dtype=float)
    prior[inside] = inside_weight
    LOG.info(
        "Sensor-hull prior: inside=%d outside=%d (inside_weight=%.3g outside_weight=%.3g)",
        int(np.sum(inside)),
        int(np.sum(~inside)),
        float(inside_weight),
        float(outside_weight),
    )
    return prior


def _compute_siddon_kernel(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
):
    """Build ray-path sensitivity matrix G using the Siddon straight-ray algorithm.

    Returns a scipy CSR sparse matrix of shape (n_pairs, n_cells) where
    G[i, j] is the path length in metres of ray i through cell j.

    Cell ordering matches _build_node_gradient_operator: cidx = ix + cx*(iy + cy*iz).
    No external C extensions — pure NumPy/SciPy.
    """
    from scipy.sparse import lil_matrix

    n_pairs = len(tx)
    cx, cy, cz = len(x) - 1, len(y) - 1, len(z) - 1
    n_model = cx * cy * cz

    G = lil_matrix((n_pairs, n_model), dtype=np.float64)

    for ray_i in range(n_pairs):
        p0 = tx[ray_i].astype(float)
        p1 = rx[ray_i].astype(float)
        diff = p1 - p0
        ray_len = float(np.linalg.norm(diff))
        if ray_len < 1e-9:
            continue

        def _axis_ts(p0c: float, p1c: float, planes: np.ndarray) -> np.ndarray:
            if abs(p1c - p0c) < 1e-12:
                return np.empty(0, dtype=float)
            ts = (planes - p0c) / (p1c - p0c)
            return ts[(ts > -1e-10) & (ts < 1.0 + 1e-10)]

        t_arr = np.unique(np.concatenate([
            [0.0, 1.0],
            _axis_ts(p0[0], p1[0], x),
            _axis_ts(p0[1], p1[1], y),
            _axis_ts(p0[2], p1[2], z),
        ]))
        t_arr = np.clip(t_arr, 0.0, 1.0)

        # Midpoints of each segment
        t_mids = (t_arr[:-1] + t_arr[1:]) / 2.0
        pts_mid = p0 + t_mids[:, None] * diff  # (n_seg, 3)

        # Cell indices via searchsorted on node arrays
        ix = np.clip(np.searchsorted(x, pts_mid[:, 0]) - 1, 0, cx - 1)
        iy = np.clip(np.searchsorted(y, pts_mid[:, 1]) - 1, 0, cy - 1)
        iz = np.clip(np.searchsorted(z, pts_mid[:, 2]) - 1, 0, cz - 1)
        seg_lens = ray_len * (t_arr[1:] - t_arr[:-1])

        # Only accumulate segments whose midpoint is inside grid bounds
        in_bounds = (
            (pts_mid[:, 0] >= x[0]) & (pts_mid[:, 0] <= x[-1]) &
            (pts_mid[:, 1] >= y[0]) & (pts_mid[:, 1] <= y[-1]) &
            (pts_mid[:, 2] >= z[0]) & (pts_mid[:, 2] <= z[-1])
        )
        for k in np.where(in_bounds)[0]:
            cidx = int(ix[k]) + cx * (int(iy[k]) + cy * int(iz[k]))
            G[ray_i, cidx] += seg_lens[k]

    LOG.info("Siddon kernel complete: G shape=%s, nnz=%d", G.shape, G.nnz)
    return G.tocsr()


def _solve_epoch(
    G,
    D,
    p_mask,
    dt_s: np.ndarray,
    lam_d: float,
    lam_p: float,
):
    from scipy.sparse import diags, vstack
    from scipy.sparse.linalg import lsqr

    P = diags(p_mask, 0, format="csr")

    A = vstack([G, lam_d * D, lam_p * P], format="csr")
    b = np.concatenate([dt_s, np.zeros(D.shape[0]), np.zeros(P.shape[0])])

    out = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=1000)
    ds = out[0]
    return ds, out


def _write_quicklook_png(
    out_png: Path,
    dvp_grid: np.ndarray,
    z: np.ndarray,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        LOG.warning("matplotlib not available; skipping quicklook png")
        return

    iz = len(z) // 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=140)
    im = ax.imshow(dvp_grid[:, :, iz].T, origin="lower", cmap="RdBu_r")
    ax.set_title(f"dVp slice @ z index {iz}")
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    fig.colorbar(im, ax=ax, label="dVp (m/s)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run TTCR timelapse inversion for CUSSP CASSM")
    p.add_argument("--bundle-file", default="/home/chopp/cassm_local/live/cassm_dashboard_bundle_full.npz")
    p.add_argument("--pair-file", default="", help="Path to pair_geometry.csv (not needed when --sources-csv/--receivers-csv are given).")
    p.add_argument("--sources-csv", default="/home/chopp/cassm_local/inversion/input/sources_hmc.csv", help="Source coordinate CSV (enables active-pair filtering).")
    p.add_argument("--receivers-csv", default="/home/chopp/cassm_local/inversion/input/receivers_hmc.csv", help="Receiver coordinate CSV (enables active-pair filtering).")
    p.add_argument(
        "--source-boreholes",
        default="AML,AML,AML,AML,AMU,AMU,AMU,AMU,DML,DML,DML,DML,DMU,DMU,DMU,DMU",
        help="Comma-separated borehole name per source, matching bundle source ordering.",
    )
    p.add_argument("--n-receivers", type=int, default=72, help="Number of receiver channels in bundle.")
    p.add_argument(
        "--min-valid-epochs-per-pair",
        type=int,
        default=1,
        help="Require at least this many picked epochs for a pair to be used at all (default 1).",
    )
    p.add_argument("--grid-mask-file", default="/home/chopp/cassm_local/inversion/input/inversion_grid_mask.npz")
    p.add_argument("--out-dir", default="/home/chopp/cassm_local/inversion/live")
    p.add_argument("--vp-background", type=float, default=6900.0)
    p.add_argument("--lambda-d", type=float, default=1.0)
    p.add_argument("--lambda-p", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=0, help="0 means all epochs")
    p.add_argument("--dt-max-us", type=float, default=150.0,
                   help="Hard-reject |dt| > this threshold before solving (µs); default 150")
    p.add_argument("--dt-outlier-nsigma", type=float, default=3.0,
                   help="Per-epoch MAD outlier rejection: reject pairs with |dt - median| > N*1.4826*MAD; 0 = disabled (default 3.0)")
    p.add_argument(
        "--baseline-n-epochs",
        type=int,
        default=0,
        help="If >0, solve baseline slowness from the first N epochs and use it as s0.",
    )
    p.add_argument(
        "--prior-inside-weight",
        type=float,
        default=1.0,
        help="Prior weight inside sensor-defined 3-D hull (default 1).",
    )
    p.add_argument(
        "--prior-outside-weight",
        type=float,
        default=1000.0,
        help="Prior weight outside sensor-defined 3-D hull (default 1000).",
    )
    p.add_argument("--min-hits", type=int, default=0,
                   help="Zero out cells traversed by fewer than this many rays (hit-count mask); 0 = disabled")
    p.add_argument("--quicklook-every", type=int, default=0,
                   help="Write a quicklook PNG every N epochs; 0 = disabled (default)")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    bundle_file = Path(args.bundle_file)
    pair_file = Path(args.pair_file)
    grid_file = Path(args.grid_mask_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load_bundle(bundle_file)
    dt_us = np.asarray(bundle["dt_us"], dtype=float)
    n_pairs, n_epochs = dt_us.shape

    if args.sources_csv and args.receivers_csv:
        src_bh_list = [s.strip() for s in args.source_boreholes.split(",")]
        active_idxs, tx, rx = _build_active_pair_geometry(
            dt_us=dt_us,
            sources_csv=Path(args.sources_csv),
            receivers_csv=Path(args.receivers_csv),
            src_bh_list=src_bh_list,
            n_rec=int(args.n_receivers),
            min_valid_epochs_per_pair=int(args.min_valid_epochs_per_pair),
        )
        dt_us = dt_us[active_idxs, :]
        n_pairs = len(active_idxs)
        LOG.info("Using %d active pairs with geometry from CSVs", n_pairs)
    else:
        if not args.pair_file:
            raise ValueError("Either --sources-csv/--receivers-csv or --pair-file must be provided.")
        tx, rx = _load_pair_geometry(pair_file, n_pairs_expected=n_pairs)

    grid = np.load(grid_file, allow_pickle=True)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid["y"], dtype=float)
    z = np.asarray(grid["z"], dtype=float)
    p_mask = np.asarray(grid["mask"], dtype=float)
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    cx, cy, cz = nx - 1, ny - 1, nz - 1

    n_model = cx * cy * cz
    p_mask = _adapt_mask_to_model(p_mask, nx, ny, nz)

    LOG.info("Building Siddon straight-ray kernel G (%d pairs)...", n_pairs)
    G = _compute_siddon_kernel(
        x=x,
        y=y,
        z=z,
        tx=tx,
        rx=rx,
    )
    LOG.info("G shape: %s", G.shape)

    # Localized prior penalty in sensor-defined 3-D volume (convex hull)
    p_hull = _build_sensor_hull_prior(
        x=x,
        y=y,
        z=z,
        tx=tx,
        rx=rx,
        inside_weight=float(args.prior_inside_weight),
        outside_weight=float(args.prior_outside_weight),
    )
    p_mask = p_mask * p_hull

    # hit-count mask: cells traversed by fewer than --min-hits rays are zeroed
    hit_counts = np.bincount(G.indices, minlength=n_model)
    min_hits = int(args.min_hits)
    if min_hits > 0:
        hit_mask = hit_counts >= min_hits
        n_masked = int(np.sum(~hit_mask))
        LOG.info("Hit-count mask: %d/%d cells below min-hits=%d will be zeroed",
                 n_masked, n_model, min_hits)
    else:
        hit_mask = np.ones(n_model, dtype=bool)

    # Restrict inversion to hit-supported model cells so min-hits improves conditioning,
    # not only post-solve visualization.
    model_mask = hit_mask
    n_model_active = int(np.sum(model_mask))
    if n_model_active == 0:
        raise RuntimeError("No model cells survive --min-hits; lower the threshold.")
    if n_model_active < n_model:
        LOG.info("Active model cells for solve: %d/%d", n_model_active, n_model)

    Gm = G[:, model_mask]

    LOG.info("Building regularization operator D...")
    D = _build_node_gradient_operator(cx, cy, cz)
    Dm = D[:, model_mask]
    p_mask_m = p_mask[model_mask]

    epoch_labels = bundle["epoch_labels"]
    if args.max_epochs and args.max_epochs > 0:
        n_epochs = min(n_epochs, int(args.max_epochs))

    dt_thresh_s = float(args.dt_max_us) * 1.0e-6
    dt_nsigma   = float(args.dt_outlier_nsigma)
    quicklook_every = int(args.quicklook_every)

    s0_m = np.full(n_model_active, 1.0 / float(args.vp_background), dtype=float)

    baseline_n = int(args.baseline_n_epochs)
    if baseline_n > 0:
        n_base = min(baseline_n, n_epochs)
        dt_base_us = np.where(dt_us[:, :n_base] != 0.0, dt_us[:, :n_base], np.nan)
        dt_base_us = np.nanmedian(dt_base_us, axis=1)
        dt_base_us = np.nan_to_num(dt_base_us, nan=0.0)
        dt_base_s = dt_base_us * 1.0e-6

        valid_b = (dt_base_s != 0.0) & (np.abs(dt_base_s) <= dt_thresh_s)
        if dt_nsigma > 0 and valid_b.sum() > 3:
            dt_ok = dt_base_s[valid_b]
            med = np.median(dt_ok)
            mad = np.median(np.abs(dt_ok - med))
            sigma_est = mad * 1.4826
            if sigma_est > 0:
                valid_b &= np.abs(dt_base_s - med) <= dt_nsigma * sigma_est

        if valid_b.any():
            ds_base_m, _ = _solve_epoch(
                G=Gm[valid_b, :],
                D=Dm,
                p_mask=p_mask_m,
                dt_s=dt_base_s[valid_b],
                lam_d=float(args.lambda_d),
                lam_p=float(args.lambda_p),
            )
            s0_m = s0_m + ds_base_m
            s0_m[s0_m <= 1e-9] = 1e-9
            LOG.info("Baseline slowness initialized from first %d epochs", n_base)
        else:
            LOG.warning("Baseline solve requested but no valid baseline pairs; using scalar background")

    ds_all = np.zeros((n_model, n_epochs), dtype=np.float32)
    dvp_all = np.zeros((n_model, n_epochs), dtype=np.float32)
    residual_norm = np.zeros(n_epochs, dtype=np.float64)

    for e in range(n_epochs):
        lbl = str(epoch_labels[e])
        dt_s_full = dt_us[:, e] * 1.0e-6

        # Step 1: hard threshold + zero-pick mask
        valid_e = (dt_s_full != 0.0) & (np.abs(dt_s_full) <= dt_thresh_s)

        # Step 2: per-epoch MAD outlier rejection on the surviving picks
        if dt_nsigma > 0 and valid_e.sum() > 3:
            dt_ok = dt_s_full[valid_e]
            med = np.median(dt_ok)
            mad = np.median(np.abs(dt_ok - med))
            sigma_est = mad * 1.4826          # consistent std estimator
            if sigma_est > 0:
                valid_e &= np.abs(dt_s_full - med) <= dt_nsigma * sigma_est

        if not valid_e.any():
            LOG.warning("Epoch %d/%d (%s): no valid pairs, skipping", e + 1, n_epochs, lbl)
            continue

        G_e  = Gm[valid_e, :]          # subset to valid pairs and active model cells
        dt_s = dt_s_full[valid_e]

        ds_m, lsqr_out = _solve_epoch(
            G=G_e,
            D=Dm,
            p_mask=p_mask_m,
            dt_s=dt_s,
            lam_d=float(args.lambda_d),
            lam_p=float(args.lambda_p),
        )

        s_m = s0_m + ds_m
        s_m[s_m <= 1e-9] = 1e-9
        vp_m = 1.0 / s_m
        dvp_m = vp_m - float(args.vp_background)

        ds = np.zeros(n_model, dtype=float)
        dvp = np.zeros(n_model, dtype=float)
        ds[model_mask] = ds_m
        dvp[model_mask] = dvp_m

        ds_all[:, e] = ds.astype(np.float32)
        dvp_all[:, e] = dvp.astype(np.float32)
        residual_norm[e] = float(lsqr_out[3])

        # quicklook slice (throttled)
        if quicklook_every > 0 and (e % quicklook_every == 0):
            dvp_grid = dvp.reshape((cx, cy, cz), order="C")
            png = out_dir / f"dvp_slice_{lbl}.png"
            _write_quicklook_png(png, dvp_grid, z)

        if (e + 1) % 100 == 0 or e == n_epochs - 1:
            LOG.info("Solved epoch %d/%d: %s", e + 1, n_epochs, lbl)

    out_npz = out_dir / "ttcr_timelapse_results.npz"
    with out_npz.open("wb") as f:
        np.savez_compressed(
            f,
            epoch_labels=epoch_labels[:n_epochs],
            x=x,
            y=y,
            z=z,
            nx=cx,
            ny=cy,
            nz=cz,
            vp_background=float(args.vp_background),
            lambda_d=float(args.lambda_d),
            lambda_p=float(args.lambda_p),
            baseline_n_epochs=np.int32(baseline_n),
            prior_inside_weight=np.float64(args.prior_inside_weight),
            prior_outside_weight=np.float64(args.prior_outside_weight),
            method="straight-ray-Siddon",
            ds=ds_all,
            dvp=dvp_all,
            residual_norm=residual_norm,
            hit_counts=hit_counts.astype(np.int32),
            active_model_cells=np.int32(n_model_active),
            min_hits=np.int32(min_hits),
            dt_outlier_nsigma=np.float64(dt_nsigma),
        )

    summary = {
        "bundle_file": str(bundle_file),
        "sources_csv": str(args.sources_csv) if args.sources_csv else None,
        "receivers_csv": str(args.receivers_csv) if args.receivers_csv else None,
        "pair_file": str(pair_file) if str(pair_file) else None,
        "grid_mask_file": str(grid_file),
        "out_npz": str(out_npz),
        "n_pairs": int(n_pairs),
        "n_epochs": int(n_epochs),
        "n_model": int(n_model),
        "n_model_active": int(n_model_active),
        "grid_shape": [cx, cy, cz],
        "vp_background": float(args.vp_background),
        "lambda_d": float(args.lambda_d),
        "lambda_p": float(args.lambda_p),
        "min_valid_epochs_per_pair": int(args.min_valid_epochs_per_pair),
        "baseline_n_epochs": int(baseline_n),
        "prior_inside_weight": float(args.prior_inside_weight),
        "prior_outside_weight": float(args.prior_outside_weight),
        "method": "straight-ray-Siddon",
        "min_hits": int(min_hits),
        "dt_outlier_nsigma": float(dt_nsigma),
    }
    (out_dir / "ttcr_timelapse_summary.json").write_text(json.dumps(summary, indent=2))

    LOG.info("Wrote results: %s", out_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
