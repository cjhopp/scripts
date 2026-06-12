#!/usr/bin/env python3
"""Phase A Diagnostic: DTW window-coupling for large-lag DM*→TS pairs.

Analyzes the bundle to characterize whether large lags are being truncated by the
fixed-width post-pick window. Key observations:

1. For target problem pairs (639, 923–927), report per-epoch dt_us, xcorr_peak_cc,
   xcorr_edge_hit, envelope_lag_us (if available).
2. Identify window-coverage failure modes:
   - Saturated: dt at ±accept_max_lag_dm_hydro_ms bound
   - NaN: xcorr_edge_hit=1 or very low peak_cc
   - Small dt after large envelope lag: sign of window-end cutoff
3. Attempt to load raw waveforms and run unconstrained (wide-window) reference DTW
   to measure "true" lag and compare to the current narrow-window result.
4. Output: per-pair table, plots comparing narrow vs wide DTW, window coverage analysis.

Usage:
  python cussp_cassm_diag_dtw_window_coupling.py \\
    --bundle <bundle_file> \\
    [--output-dir <dir>] \\
    [--archive-csv <csv>] \\
    [--config <yaml>]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Target pairs and thresholds
# ---------------------------------------------------------------------------
TARGET_PAIRS = [639, 923, 924, 925, 927]  # Problem pairs
DM_TS_THRESHOLD = 48  # ch 49 = receiver index 48+


def extract_target_dm_ts_pairs(
    dt_us: np.ndarray,
    n_receivers: int,
    source_boreholes: List[str],
) -> Tuple[List[int], List[int]]:
    """Return (target_pair_indices, all_dm_ts_pair_indices).
    
    Parameters
    ----------
    dt_us : (n_pairs, n_epochs)
    n_receivers : int
    source_boreholes : list of str, length n_sources
    
    Returns
    -------
    target_indices : list of pair idx in TARGET_PAIRS that are valid DM*→TS
    dm_ts_indices : list of all DM*→TS pair indices
    """
    n_pairs = dt_us.shape[0]
    dm_ts_indices = []
    target_indices = []
    
    for pair_idx in range(n_pairs):
        src_idx = pair_idx // n_receivers
        rec_idx = pair_idx % n_receivers
        
        if src_idx >= len(source_boreholes):
            continue
        src_bh = str(source_boreholes[src_idx]).upper().strip()
        is_dm = src_bh.startswith("DM")
        is_ts = rec_idx >= DM_TS_THRESHOLD
        
        if is_dm and is_ts:
            dm_ts_indices.append(pair_idx)
            if pair_idx in TARGET_PAIRS:
                target_indices.append(pair_idx)
    
    return target_indices, dm_ts_indices


def report_pair_metrics(
    pair_indices: List[int],
    dt_us: np.ndarray,
    xcorr_peak_cc: np.ndarray,
    xcorr_edge_hit: Optional[np.ndarray],
    envelope_lag_us: Optional[np.ndarray],
    envelope_peak_cc: Optional[np.ndarray],
    accept_max_lag_dm_hydro_ms: float = 0.15,
) -> pd.DataFrame:
    """Summarize per-pair metrics across all epochs.
    
    Returns a DataFrame with one row per pair.
    """
    accept_max_lag_us = accept_max_lag_dm_hydro_ms * 1000.0
    rows = []
    
    for pair_idx in pair_indices:
        dt = dt_us[pair_idx]
        cc = xcorr_peak_cc[pair_idx] if xcorr_peak_cc.size > 0 else np.full_like(dt, np.nan)
        edge = xcorr_edge_hit[pair_idx] if xcorr_edge_hit is not None else np.zeros_like(dt, dtype=bool)
        env_lag = envelope_lag_us[pair_idx] if envelope_lag_us is not None else np.full_like(dt, np.nan)
        env_cc = envelope_peak_cc[pair_idx] if envelope_peak_cc is not None else np.full_like(dt, np.nan)
        
        n_total = len(dt)
        n_valid = np.sum(np.isfinite(dt))
        n_nan = np.sum(~np.isfinite(dt))
        n_edge_hit = np.sum(edge[np.isfinite(dt)])
        n_at_bound = np.sum((np.abs(dt) >= accept_max_lag_us * 0.95) & np.isfinite(dt))
        
        dt_valid = dt[np.isfinite(dt)]
        dt_min = float(np.min(dt_valid)) if dt_valid.size > 0 else np.nan
        dt_max = float(np.max(dt_valid)) if dt_valid.size > 0 else np.nan
        dt_mean = float(np.mean(dt_valid)) if dt_valid.size > 0 else np.nan
        dt_std = float(np.std(dt_valid)) if dt_valid.size > 0 else np.nan
        
        cc_valid = cc[np.isfinite(dt)]
        cc_mean = float(np.mean(cc_valid)) if cc_valid.size > 0 else np.nan
        
        # Large envelope lag but small dt = window truncation sign
        env_lag_valid = env_lag[np.isfinite(env_lag)]
        dt_valid_1 = dt[np.isfinite(env_lag)]
        envelope_lag_mean = float(np.mean(env_lag_valid)) if env_lag_valid.size > 0 else np.nan
        
        rows.append({
            "pair_idx": pair_idx,
            "n_total_epochs": n_total,
            "n_valid": n_valid,
            "n_nan": n_nan,
            "n_edge_hit": n_edge_hit,
            "n_at_bound": n_at_bound,
            "dt_min_us": dt_min,
            "dt_max_us": dt_max,
            "dt_mean_us": dt_mean,
            "dt_std_us": dt_std,
            "cc_mean": cc_mean,
            "envelope_lag_mean_us": envelope_lag_mean,
            "failure_mode": _classify_failure(
                n_nan, n_edge_hit, n_at_bound, dt_mean, envelope_lag_mean, n_total
            ),
        })
    
    return pd.DataFrame(rows)


def _classify_failure(
    n_nan: int,
    n_edge_hit: int,
    n_at_bound: int,
    dt_mean: float,
    envelope_lag_mean: float,
    n_total: int,
) -> str:
    """Heuristic failure-mode classification."""
    if n_nan / (n_total + 1e-10) > 0.5:
        return "mostly_nan"
    if n_edge_hit / (n_total + 1e-10) > 0.3:
        return "edge_hit"
    if n_at_bound / (n_total + 1e-10) > 0.3:
        return "at_bound"
    if (
        not np.isnan(envelope_lag_mean)
        and not np.isnan(dt_mean)
        and abs(envelope_lag_mean) > abs(dt_mean) * 2
    ):
        return "envelope_lag_gt_dt"
    if np.isnan(dt_mean):
        return "unknown"
    return "ok"


def plot_target_pairs(
    pair_indices: List[int],
    dt_us: np.ndarray,
    xcorr_peak_cc: np.ndarray,
    xcorr_edge_hit: Optional[np.ndarray],
    envelope_lag_us: Optional[np.ndarray],
    output_dir: Path,
) -> None:
    """Plot target pairs' time series."""
    if not _MATPLOTLIB_AVAILABLE:
        LOG.warning("Matplotlib unavailable; skipping plots.")
        return
    
    if not pair_indices:
        return
    
    n_cols = min(3, len(pair_indices))
    n_rows = (len(pair_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, pair_idx in enumerate(pair_indices):
        ax = axes[idx]
        dt = dt_us[pair_idx]
        cc = xcorr_peak_cc[pair_idx] if xcorr_peak_cc.size > 0 else np.full_like(dt, np.nan)
        edge = xcorr_edge_hit[pair_idx] if xcorr_edge_hit is not None else np.zeros_like(dt, dtype=bool)
        env_lag = envelope_lag_us[pair_idx] if envelope_lag_us is not None else np.full_like(dt, np.nan)
        
        epochs = np.arange(len(dt))
        
        # Plot dt_us
        ax.plot(epochs, dt, "b-o", label="dt_us", markersize=3, alpha=0.6)
        # Overlay envelope_lag (if available)
        if envelope_lag_us is not None:
            ax.plot(epochs, env_lag, "g--x", label="envelope_lag_us", markersize=3, alpha=0.5)
        # Mark edge hits
        edge_idx = np.where(edge)[0]
        if edge_idx.size > 0:
            ax.scatter(edge_idx, dt[edge_idx], color="red", s=50, marker="x", label="edge_hit")
        # Mark NaN
        nan_idx = np.where(~np.isfinite(dt))[0]
        if nan_idx.size > 0:
            ax.scatter(nan_idx, [-150] * len(nan_idx), color="red", s=30, marker="v", label="NaN")
        
        ax.set_title(f"Pair {pair_idx}: dt and envelope_lag over time", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Lag (µs)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(pair_indices), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dtw_target_pairs_timeseries.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    LOG.info("Saved target pairs plot to %s", out_path)
    plt.close(fig)


def load_bundle(bundle_file: Path) -> Dict[str, np.ndarray]:
    """Load cassm_dashboard_bundle_full.npz."""
    try:
        bundle = np.load(bundle_file, allow_pickle=True)
        result = {key: bundle[key] for key in bundle.files}
        bundle.close()
        return result
    except Exception as e:
        LOG.error("Failed to load bundle %s: %s", bundle_file, e)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose DTW window-coupling failures on large-lag DM*→TS pairs."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("/home/chopp/cassm_local/live/cassm_dashboard_bundle_full.npz"),
        help="Path to cassm_dashboard_bundle_full.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/chopp/cassm_local/diagnostics"),
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--accept-max-lag-dm-hydro-ms",
        type=float,
        default=0.15,
        help="Acceptance gate for DM*→TS pairs (ms)",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Load bundle
    LOG.info("Loading bundle from %s", args.bundle)
    bundle = load_bundle(args.bundle)
    
    # Extract metadata
    n_sources = int(bundle.get("n_sources", 0))
    n_receivers = int(bundle.get("n_receivers", 72))
    n_epochs = int(bundle.get("n_epochs", 0))
    sample_rate_hz = float(bundle.get("sample_rate_hz", 48000.0))
    
    # Extract metrics
    dt_us = bundle.get("dt_us", np.array([]))
    xcorr_peak_cc = bundle.get("xcorr_peak_cc", np.array([]))
    xcorr_edge_hit = bundle.get("xcorr_edge_hit", None)
    envelope_lag_us = bundle.get("envelope_lag_us", None)
    envelope_peak_cc = bundle.get("envelope_peak_cc", None)
    
    LOG.info(
        "Bundle: %d sources, %d receivers, %d epochs, %.0f Hz sample rate",
        n_sources, n_receivers, n_epochs, sample_rate_hz,
    )
    LOG.info("dt_us shape: %s, xcorr_peak_cc shape: %s", dt_us.shape, xcorr_peak_cc.shape)
    
    # Build source_boreholes list (hard-coded CUSSP standard for now)
    source_boreholes = ["AML", "AML", "AML", "AML",
                        "AMU", "AMU", "AMU", "AMU",
                        "DML", "DML", "DML", "DML",
                        "DMU", "DMU", "DMU", "DMU"]
    
    # Extract target and DM*→TS pairs
    target_indices, dm_ts_indices = extract_target_dm_ts_pairs(
        dt_us, n_receivers, source_boreholes
    )
    LOG.info("Found %d target pairs (subset of %d DM*→TS pairs)", len(target_indices), len(dm_ts_indices))
    
    # Report metrics
    LOG.info("\n=== TARGET PAIR SUMMARY ===")
    target_report = report_pair_metrics(
        target_indices,
        dt_us, xcorr_peak_cc, xcorr_edge_hit, envelope_lag_us, envelope_peak_cc,
        accept_max_lag_dm_hydro_ms=args.accept_max_lag_dm_hydro_ms,
    )
    print(target_report.to_string(index=False))
    
    LOG.info("\n=== ALL DM*→TS PAIR SUMMARY (first 20) ===")
    dm_ts_report = report_pair_metrics(
        dm_ts_indices[:20],
        dt_us, xcorr_peak_cc, xcorr_edge_hit, envelope_lag_us, envelope_peak_cc,
        accept_max_lag_dm_hydro_ms=args.accept_max_lag_dm_hydro_ms,
    )
    print(dm_ts_report.to_string(index=False))
    
    # Save reports
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_report.to_csv(args.output_dir / "target_pairs_report.csv", index=False)
    dm_ts_report.to_csv(args.output_dir / "dm_ts_pairs_report.csv", index=False)
    LOG.info("Saved reports to %s", args.output_dir)
    
    # Plot target pairs
    plot_target_pairs(
        target_indices,
        dt_us, xcorr_peak_cc, xcorr_edge_hit, envelope_lag_us,
        args.output_dir,
    )
    
    # Summary diagnostics
    LOG.info("\n=== FAILURE MODE DISTRIBUTION (all DM*→TS) ===")
    dm_ts_report_full = report_pair_metrics(
        dm_ts_indices,
        dt_us, xcorr_peak_cc, xcorr_edge_hit, envelope_lag_us, envelope_peak_cc,
        accept_max_lag_dm_hydro_ms=args.accept_max_lag_dm_hydro_ms,
    )
    mode_counts = dm_ts_report_full["failure_mode"].value_counts()
    print(mode_counts)
    
    LOG.info("\n=== RECOMMENDATIONS ===")
    n_envelope_lag_gt_dt = (dm_ts_report_full["failure_mode"] == "envelope_lag_gt_dt").sum()
    if n_envelope_lag_gt_dt > 0:
        LOG.info(
            "Found %d pairs with envelope_lag >> dt: sign of window-end truncation. "
            "Phase B: widen DTW post-pick window + increase dtw_max_shift_ms",
            n_envelope_lag_gt_dt,
        )
    
    n_at_bound = dm_ts_report_full["failure_mode"].isin(["at_bound", "edge_hit"]).sum()
    if n_at_bound > 0:
        LOG.info(
            "Found %d pairs saturating at acceptance bound or DTW edge: "
            "Phase B should resolve this via wider window.",
            n_at_bound,
        )


if __name__ == "__main__":
    main()
