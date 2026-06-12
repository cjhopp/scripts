#!/usr/bin/env python3
"""Diagnostic: per-epoch DTW gate breakdown for target pairs.

Loads the raw gather (not the bundle) and re-runs the DTW acceptance logic
for each epoch of the specified pairs, reporting EXACTLY which gate caused NaN.

Usage:
    python cussp_cassm_diag_dtw_gates.py --config /path/to/cussp_cassm_config.yaml \
        --pair-indices 924 927 923 639 925
"""
import argparse
import sys
import logging
import numpy as np
from typing import List, Tuple
from collections import Counter

sys.path.insert(0, '/home/chopp/scripts/python/vm')

from pathlib import Path
from cussp_cassm_process import (
    MetricConfig, load_config, _dtw_dt_samples, _xcorr_dt_samples,
    _cosine_window, _preprocess_waveform, _window_samples, CASSMTempGather,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)


def run_gate_diag(config_path: str, pair_indices: List[int], n_epochs_max: int = 200):
    args = load_config(config_path)
    cfg = MetricConfig(
        dtw_enabled=args.dtw_enabled,
        dtw_max_shift_ms=args.dtw_max_shift_ms,
        dtw_strain_limit=args.dtw_strain_limit,
        dtw_min_ncc=args.dtw_min_ncc,
        xcorr_edge_guard_samples=args.xcorr_edge_guard_samples,
        xcorr_min_peak_cc=args.xcorr_min_peak_cc,
        window_pre_pick_s=args.window_pre_pick_ms / 1000.0 if args.window_pre_pick_ms is not None else None,
        window_post_pick_s=args.window_post_pick_ms / 1000.0 if args.window_post_pick_ms is not None else None,
        xcorr_fine_half_lag_s=args.xcorr_fine_half_lag_s,
        filter_low_hz=args.filter_low_hz,
        filter_high_hz=args.filter_high_hz,
    )
    dtw_max_shift_ms = cfg.dtw_max_shift_ms

    gather = CASSMTempGather.from_hdf5_compact(Path(args.cache_file))
    sr = gather.sample_rate_hz
    pre_samples = max(int((cfg.window_pre_pick_s or 0.0001) * sr), 1)
    post_samples = max(int((cfg.window_post_pick_s or 0.0005) * sr), 1)
    dtw_max_shift_samples = max(int(cfg.dtw_max_shift_ms * sr / 1000.0), 1)
    dtw_win_extension = dtw_max_shift_samples + 5
    fine_half_lag = max(int(cfg.xcorr_fine_half_lag_s * sr), 1)

    log.info(f"Config: max_shift={dtw_max_shift_ms:.3f}ms={dtw_max_shift_samples}samp "
             f"edge_guard={cfg.xcorr_edge_guard_samples} "
             f"fine_half_lag={fine_half_lag}samp "
             f"min_ncc={cfg.dtw_min_ncc} "
             f"dtw_win_extension={dtw_win_extension}")
    log.info(f"window: pre={pre_samples} post={post_samples} dtw_wide={pre_samples+post_samples+dtw_win_extension}")

    # Baseline picks using the MetricConfig (same as main pipeline)
    baseline_picks = gather._baseline_picks(cfg)

    for p in pair_indices:
        if p >= gather.n_pairs:
            log.warning(f"Pair {p} out of range (n_pairs={gather.n_pairs}), skipping")
            continue

        # Build baseline windows using same stacking as main pipeline
        n_base = cfg.baseline_n_epochs
        if n_base > 1:
            base_raw = np.mean(gather.get_pair(slice(0, n_base), p), axis=0)
        else:
            base_raw = gather.get_pair(0, p)
        base_tr = _preprocess_waveform(base_raw, sr, cfg, pair_index=p, n_receivers=gather.n_receivers)

        pick = baseline_picks[p]
        sw = _window_samples(pick, pre_samples, post_samples, gather.sample_count)
        w_len = sw.stop - sw.start
        taper = _cosine_window(w_len, cfg.window_taper_fraction)
        bl_win = base_tr[sw] * taper

        sw_dtw = _window_samples(pick, pre_samples, post_samples + dtw_win_extension, gather.sample_count)
        w_dtw = sw_dtw.stop - sw_dtw.start
        if w_dtw >= 4:
            taper_dtw_bl = _cosine_window(w_dtw, cfg.window_taper_fraction)
            bl_dtw_win = base_tr[sw_dtw] * taper_dtw_bl
        else:
            bl_dtw_win = np.zeros(pre_samples + post_samples + dtw_win_extension)

        n_epochs = min(gather.n_epochs, n_epochs_max)
        gate_counts = Counter()
        dtw_lags, dtw_nccs = [], []

        for e in range(n_epochs):
            tr = _preprocess_waveform(
                gather.get_pair(e, p), sr, cfg, pair_index=p, n_receivers=gather.n_receivers
            )
            ep_seg = tr[sw]
            if ep_seg.size < 4 or w_len < 4:
                gate_counts['short_narrow'] += 1
                continue
            ep_win = ep_seg * taper[:ep_seg.size]

            ep_dtw_seg = tr[sw_dtw]
            if ep_dtw_seg.size < 4 or w_dtw < 4:
                gate_counts['short_wide'] += 1
                continue

            taper_dtw_ep = _cosine_window(ep_dtw_seg.size, cfg.window_taper_fraction)
            ep_dtw_win = ep_dtw_seg * taper_dtw_ep

            dtw_lag, dtw_ncc, dtw_rejected = _dtw_dt_samples(
                baseline_win=bl_dtw_win,
                epoch_win=ep_dtw_win,
                max_shift=dtw_max_shift_samples,
                strain_limit=cfg.dtw_strain_limit,
                edge_guard_samples=cfg.xcorr_edge_guard_samples,
                signal_end_j=pre_samples + post_samples,
            )
            dtw_lags.append(dtw_lag)
            dtw_nccs.append(dtw_ncc)

            dtw_min_ncc_violated = dtw_ncc < cfg.dtw_min_ncc
            dtw_saturated = dtw_rejected

            if dtw_saturated:
                gate_counts['dtw_saturated'] += 1
                continue
            if dtw_min_ncc_violated:
                gate_counts['dtw_min_ncc'] += 1
                continue

            lag, peak_cc, edge_hit = _xcorr_dt_samples(
                bl_win, ep_win, fine_half_lag,
                cfg.xcorr_edge_guard_samples,
                center_lag=int(round(dtw_lag)),
            )
            if edge_hit:
                gate_counts['fine_xcorr_edge_hit'] += 1
                continue
            if peak_cc < cfg.xcorr_min_peak_cc:
                gate_counts['fine_xcorr_low_cc'] += 1
                continue
            gate_counts['accept'] += 1

        total = sum(gate_counts.values())
        dtw_lags = np.array(dtw_lags)
        dtw_nccs = np.array(dtw_nccs)
        log.info(
            f"\nPair {p} (first {n_epochs} epochs):\n"
            f"  DTW lag: min={dtw_lags.min():.1f} max={dtw_lags.max():.1f} "
            f"mean={dtw_lags.mean():.2f} samp  (max_shift={dtw_max_shift_samples})\n"
            f"  DTW NCC: min={dtw_nccs.min():.3f} max={dtw_nccs.max():.3f} "
            f"mean={dtw_nccs.mean():.3f}  (threshold={cfg.dtw_min_ncc})\n"
            f"  Gate breakdown (n={total}):"
        )
        for gate, count in sorted(gate_counts.items(), key=lambda x: -x[1]):
            log.info(f"    {gate}: {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair-indices", type=int, nargs="+", default=[923, 924, 925, 927, 639])
    ap.add_argument("--n-epochs", type=int, default=300)
    args = ap.parse_args()
    run_gate_diag(args.config, args.pair_indices, args.n_epochs)
