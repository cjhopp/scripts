#!/usr/bin/env python3
"""
Diagnostic script: Investigate baseline picks and dead channels (61/63/64/65).

Load the raw HDF5 cache (from_hdf5_compact), sort by time, derive n_base from
baseline_end_date, and for pairs 635-640 print:
  - Recomputed baseline pick
  - base_raw/base_tr max-abs
  - The bl_wide_win norm
  - Per-epoch max-abs split into baseline vs post-baseline windows

This confirms that channels 61/63/64/65 have zero baseline (dead during baseline
period) while ch 62 (control) has live baseline data.
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Import CASSM process utilities
from cussp_cassm_process import (
    CASSMTempGather,
    MetricConfig,
    _preprocess_waveform,
    _apply_picker,
    _window_samples,
    _cosine_window,
)

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    if yaml is None:
        raise ImportError("PyYAML required for config loading")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("/home/chopp/cassm_local/live/cassm_tempgather_full.h5"),
        help="Path to HDF5 cache file",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("/home/chopp/scripts/python/vm/cussp_cassm_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--baseline-end-date",
        type=str,
        help="Override baseline_end_date from config (ISO-8601)",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=[635, 636, 637, 638, 639, 640],
        help="Pair indices to analyze (default: 635-640)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BASELINE DEAD CHANNEL DIAGNOSTIC")
    print("=" * 80)

    # Load config to get baseline_end_date
    if args.config_file and args.config_file.exists():
        config = load_config(args.config_file)
        baseline_end_date = config.get("picking", {}).get("baseline_end_date")
        if args.baseline_end_date:
            baseline_end_date = args.baseline_end_date
    else:
        if not args.baseline_end_date:
            raise ValueError("Must provide --baseline-end-date or --config-file")
        baseline_end_date = args.baseline_end_date

    print(f"\nCache file: {args.cache_file}")
    print(f"Baseline end date: {baseline_end_date}")

    # Load cache
    print(f"\nLoading cache from {args.cache_file}...")
    tg = CASSMTempGather.from_hdf5_compact(args.cache_file)
    print(f"  Loaded {tg.n_epochs} epochs")
    print(f"  Pairs: {tg.n_pairs} = {tg.n_sources} sources × {tg.n_receivers} receivers")
    print(f"  Samples per epoch: {tg.sample_count}")
    print(f"  Sample rate: {tg.sample_rate_hz:.1f} Hz")
    print(f"  Sample period: {tg.dt * 1e6:.2f} µs")

    # Ensure epochs are sorted by time
    if tg.sort_by_time():
        print("  Epochs were sorted by timestamp")
    else:
        print("  Epochs already sorted by timestamp")

    # Calculate baseline_n_epochs from baseline_end_date
    try:
        cutoff = pd.Timestamp(baseline_end_date).tz_localize("UTC") \
            if pd.Timestamp(baseline_end_date).tzinfo is None \
            else pd.Timestamp(baseline_end_date).tz_convert("UTC")

        def _to_utc(t: pd.Timestamp) -> pd.Timestamp:
            return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

        n_base = sum(1 for t in tg.epoch_times if _to_utc(t) <= cutoff)
    except Exception as exc:
        raise ValueError(f"Failed to parse baseline_end_date: {exc}")

    if n_base < 1:
        raise ValueError("baseline_end_date selects zero baseline epochs")

    print(f"  Baseline epochs: {n_base}/{tg.n_epochs}")
    print(f"  First epoch: {tg.epoch_times[0]}")
    print(f"  Last baseline epoch: {tg.epoch_times[n_base - 1] if n_base > 0 else 'N/A'}")
    if n_base < tg.n_epochs:
        print(f"  First post-baseline epoch: {tg.epoch_times[n_base]}")
    print(f"  Last epoch: {tg.epoch_times[-1]}")

    # Create minimal config for picking
    # Default from config.yaml picking section
    metric_config = MetricConfig(
        pick_search_s=0.05,
        window_s=0.05,
        clip_first_s=0.0,
        mute_first_s=0.0,
        hydro_clip_first_s=0.0,
        hydro_mute_first_s=0.0,
        taper_fraction=0.1,
        filter_low_hz=1.0,
        filter_high_hz=20.0,
        filter_order=4,
        accel_filter_low_hz=1.0,
        accel_filter_high_hz=20.0,
        hydro_filter_low_hz=1.0,
        hydro_filter_high_hz=20.0,
        picker="stalta",
        stalta_short_s=0.001,
        stalta_long_s=0.05,
        stalta_threshold=10.0,
        baseline_n_epochs=n_base,
        dt_method="xcorr",
        xcorr_max_lag_s=0.01,
        xcorr_accept_max_lag_s=0.01,
        xcorr_accept_max_lag_hydro_s=0.01,
        xcorr_accept_max_lag_dm_hydro_s=0.01,
        xcorr_min_peak_cc=0.0,
        xcorr_edge_guard_samples=1,
        window_taper_fraction=0.1,
        window_pre_pick_s=0.005,
        window_post_pick_s=0.02,
        xcorr_despike_single_epoch=False,
        xcorr_despike_mad_thresh=5.0,
        xcorr_mask_short_runs=False,
        xcorr_short_run_max_len_epochs=3,
        xcorr_short_run_min_amp_us=1.0,
        xcorr_short_run_neighbor_tol_us=0.5,
        envelope_guide_xcorr=True,
        envelope_max_lag_s=0.01,
        envelope_smooth_samples=5,
        envelope_min_peak_cc=0.5,
        xcorr_fine_half_lag_s=0.002,
        envelope_guide_smooth_epochs=3,
        dtw_enabled=False,
        dtw_max_shift_ms=10.0,
        dtw_strain_limit=0.1,
        dtw_min_ncc=0.5,
        source_boreholes=None,
    )

    # Load manual picks from JSON if available
    import json
    manual_picks = {}
    mp_file = Path(args.cache_file.parent) / "manual_picks.json"
    if mp_file.exists():
        try:
            with open(mp_file, "r") as f:
                raw_mp = json.load(f)
                manual_picks = {int(k): int(v) for k, v in raw_mp.items()}
            print(f"Loaded {len(manual_picks)} manual picks from {mp_file}")
        except Exception as exc:
            print(f"Warning: Failed to load manual picks: {exc}")
    else:
        print(f"No manual_picks.json found at {mp_file}")

    # Compute baseline picks (with manual overrides)
    baseline_picks = tg._baseline_picks(metric_config, manual_picks=manual_picks)

    # Window parameters
    pre_samples = max(int(metric_config.window_pre_pick_s * tg.sample_rate_hz), 1)
    post_samples = max(int(metric_config.window_post_pick_s * tg.sample_rate_hz), 15)
    envelope_max_lag = max(int(metric_config.envelope_max_lag_s * tg.sample_rate_hz), 1)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC OUTPUT FOR PAIRS 635-640")
    print("=" * 80)

    for pair_idx in args.pairs:
        if pair_idx < 0 or pair_idx >= tg.n_pairs:
            print(f"\nPair {pair_idx}: INVALID (out of range 0-{tg.n_pairs - 1})")
            continue

        src_idx = pair_idx // tg.n_receivers
        rec_idx = pair_idx % tg.n_receivers
        ch_number = rec_idx + 1  # 1-based channel

        print(f"\n{'─' * 80}")
        print(f"Pair {pair_idx}: Source {src_idx}, Receiver {rec_idx} (Channel {ch_number})")
        print(f"{'─' * 80}")

        # Compute baseline mean
        if n_base > 1:
            base_raw = np.mean(tg.get_pair(slice(0, n_base), pair_idx), axis=0)
        else:
            base_raw = tg.get_pair(0, pair_idx)

        # Preprocess
        base_tr = _preprocess_waveform(
            base_raw,
            tg.sample_rate_hz,
            metric_config,
            pair_index=pair_idx,
            n_receivers=tg.n_receivers,
        )

        # Baseline pick
        pick_idx = baseline_picks[pair_idx]

        # Extract windows
        sw = _window_samples(pick_idx, pre_samples, post_samples, tg.sample_count)
        sw_wide = _window_samples(
            pick_idx, pre_samples,
            post_samples + envelope_max_lag, tg.sample_count,
        )

        # Compute baseline window
        w_len = sw.stop - sw.start
        if w_len >= 4:
            taper = _cosine_window(w_len, metric_config.window_taper_fraction)
            bl_win = base_tr[sw] * taper
        else:
            bl_win = np.zeros(pre_samples + post_samples)

        # Compute wide baseline window (for envelope)
        w_wide = sw_wide.stop - sw_wide.start
        if w_wide >= 4:
            taper_wide = _cosine_window(w_wide, metric_config.window_taper_fraction)
            bl_wide_win = base_tr[sw_wide] * taper_wide
        else:
            bl_wide_win = np.zeros(pre_samples + post_samples + envelope_max_lag)

        # Print diagnostics
        print(f"Baseline pick index: {pick_idx} (sample)")
        print(f"  Window slice: [{sw.start}, {sw.stop})")
        print(f"  Wide window slice: [{sw_wide.start}, {sw_wide.stop})")

        print(f"\nBaseline trace statistics:")
        print(f"  base_raw max-abs: {np.max(np.abs(base_raw)):.6e}")
        print(f"  base_tr max-abs: {np.max(np.abs(base_tr)):.6e}")
        print(f"  bl_win max-abs: {np.max(np.abs(bl_win)):.6e}")
        print(f"  bl_win norm (L2): {np.linalg.norm(bl_win):.6e}")
        print(f"  bl_wide_win max-abs: {np.max(np.abs(bl_wide_win)):.6e}")
        print(f"  bl_wide_win norm (L2): {np.linalg.norm(bl_wide_win):.6e}")

        # Check if baseline is all-zero (dead channel)
        is_dead = np.allclose(base_tr, 0.0, atol=1e-10)
        print(f"\n  ✗ DEAD CHANNEL (all zeros)" if is_dead else f"  ✓ ALIVE (non-zero baseline)")

        # Per-epoch statistics
        print(f"\nPer-epoch max-abs amplitude:")
        print(f"  {'Epoch':<7} {'Label':<20} {'Type':<12} {'Max-abs':>12} {'Window':<20}")
        print(f"  {'-' * 75}")

        for e in range(tg.n_epochs):
            tr = _preprocess_waveform(
                tg.get_pair(e, pair_idx),
                tg.sample_rate_hz,
                metric_config,
                pair_index=pair_idx,
                n_receivers=tg.n_receivers,
            )

            # Get max-abs for the epoch
            ep_max = np.max(np.abs(tr))

            # Classify as baseline or post-baseline
            epoch_time = tg.epoch_times[e]
            is_baseline = e < n_base
            epoch_type = "BASELINE" if is_baseline else "POST-BL"

            # Get window max-abs
            if e < n_base:
                # Baseline window (use standard slice)
                ep_seg = tr[sw]
                if len(ep_seg) >= 4:
                    taper = _cosine_window(len(ep_seg), metric_config.window_taper_fraction)
                    ep_win = ep_seg * taper
                else:
                    ep_win = np.zeros(pre_samples + post_samples)
                win_max = np.max(np.abs(ep_win))
                win_label = "narrow"
            else:
                # Post-baseline window (check wide window)
                ep_seg_wide = tr[sw_wide]
                if len(ep_seg_wide) >= 4:
                    taper_wide = _cosine_window(len(ep_seg_wide), metric_config.window_taper_fraction)
                    ep_win_wide = ep_seg_wide * taper_wide
                else:
                    ep_win_wide = np.zeros(pre_samples + post_samples + envelope_max_lag)
                win_max = np.max(np.abs(ep_win_wide))
                win_label = "wide"

            print(f"  {e:<7} {tg.epoch_labels[e]:<20} {epoch_type:<12} "
                  f"{ep_max:>12.6e} {win_label:<20}")

            # Show first few and last few epochs
            if e == 2 and tg.n_epochs > 8:
                print(f"  ...")

        print(f"\nSummary: {n_base} baseline epochs, {tg.n_epochs - n_base} post-baseline epochs")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Dead channel signature:
  - bl_wide_win norm ~ 0.0 (zero baseline trace)
  - Per-epoch max-abs in baseline window: all ~ 0.0
  - Per-epoch max-abs in post-baseline window: 185k-234k (alive later)
  
Control channel (ch 62, pair 637) should show:
  - bl_wide_win norm >> 0.0 (non-zero baseline)
  - Per-epoch max-abs in baseline window: non-zero
  
Failure mechanism for channels 61/63/64/65:
  - Baseline mean is zero → bl_wide_win is zero
  - _envelope_coarse_lag(zero_baseline, ...) returns (0.0, 0.0)
  - All epochs locked to zero lag, causing pick misalignment
""")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
