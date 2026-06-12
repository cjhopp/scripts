#!/usr/bin/env python3
"""Diagnostic: multi-panel xcorr stage visualization for envelope-guided DM*→TS pairs.

Generates detailed plots showing coarse envelope xcorr → fine waveform xcorr stages
for selected epochs, including waveforms, envelopes, cc traces, and parabolic 
interpolation. Helps diagnose cycle-skip, coarse lag accuracy, and edge-hit issues.

Usage:
    python cussp_cassm_diag_xcorr_stages.py \
        --config /path/to/cussp_cassm_config.yaml \
        --pair-indices 923 924 925 927 639 \
        --n-debug-plots 10 \
        --output-dir /home/chopp/cassm_local/live/xcorr_diag_plots
"""
import argparse
import sys
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import random

sys.path.insert(0, '/home/chopp/scripts/python/vm')

from cussp_cassm_process import (
    MetricConfig, load_config, _envelope_coarse_lag, _xcorr_dt_samples,
    _preprocess_waveform, _window_samples, _cosine_window, CASSMTempGather,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.signal import correlate as _sp_correlate
except ImportError as e:
    log.error(f"Missing required package: {e}")
    sys.exit(1)


def _compute_xcorr_trace(baseline_win: np.ndarray, epoch_win: np.ndarray) -> np.ndarray:
    """Compute full normalized xcorr trace (2N-1 length)."""
    b = baseline_win.astype(np.float64)
    e = epoch_win.astype(np.float64)
    b_norm = np.linalg.norm(b)
    e_norm = np.linalg.norm(e)
    if b_norm < 1e-30 or e_norm < 1e-30:
        return np.zeros(2 * len(b) - 1, dtype=np.float64)
    cc = _sp_correlate(e / e_norm, b / b_norm, mode="full")
    return cc.astype(np.float64)


def _lag_to_us(lag_samples: float, sample_rate_hz: float) -> float:
    """Convert lag in samples to microseconds."""
    dt_us = 1e6 / float(sample_rate_hz)
    return float(lag_samples) * dt_us


def _sample_to_us(sample_idx: int, sample_rate_hz: float) -> float:
    """Convert sample index to microseconds from time zero."""
    dt_us = 1e6 / float(sample_rate_hz)
    return float(sample_idx) * dt_us


def run_xcorr_diag(
    config_path: str,
    pair_indices: List[int],
    n_debug_plots: int = 10,
    output_dir: Optional[str] = None,
    explicit_epochs: Optional[List[int]] = None,
):
    """Generate multi-panel xcorr stage plots for selected pairs and epochs."""
    
    args = load_config(config_path)
    
    # Convert window times from ms to seconds if provided
    window_pre_pick_s = None
    window_post_pick_s = None
    if hasattr(args, 'window_pre_pick_ms') and args.window_pre_pick_ms is not None:
        window_pre_pick_s = float(args.window_pre_pick_ms) / 1000.0
    if hasattr(args, 'window_post_pick_ms') and args.window_post_pick_ms is not None:
        window_post_pick_s = float(args.window_post_pick_ms) / 1000.0
    
    cfg = MetricConfig(
        pick_search_s=float(args.pick_search_s),
        window_s=float(args.window_s),
        clip_first_s=float(args.clip_first_s),
        mute_first_s=float(args.mute_first_s),
        hydro_clip_first_s=args.hydro_clip_first_s,
        hydro_mute_first_s=args.hydro_mute_first_s,
        taper_fraction=float(args.taper_fraction),
        filter_low_hz=float(args.filter_low_hz),
        filter_high_hz=float(args.filter_high_hz),
        filter_order=int(args.filter_order),
        accel_filter_low_hz=args.accel_filter_low_hz,
        accel_filter_high_hz=args.accel_filter_high_hz,
        hydro_filter_low_hz=args.hydro_filter_low_hz,
        hydro_filter_high_hz=args.hydro_filter_high_hz,
        window_pre_pick_s=window_pre_pick_s,
        window_post_pick_s=window_post_pick_s,
        envelope_guide_xcorr=bool(args.envelope_guide_xcorr),
        envelope_max_lag_s=float(args.envelope_max_lag_s),
        envelope_smooth_samples=int(args.envelope_smooth_samples),
        envelope_min_peak_cc=float(args.envelope_min_peak_cc),
        xcorr_fine_half_lag_s=float(args.xcorr_fine_half_lag_s),
        envelope_guide_smooth_epochs=int(args.envelope_guide_smooth_epochs),
        xcorr_edge_guard_samples=int(args.xcorr_edge_guard_samples),
        xcorr_min_peak_cc=float(args.xcorr_min_peak_cc),
    )
    
    sr = float(args.sample_rate_hz)
    dt_us = 1e6 / sr
    
    # Window dimensions
    if cfg.window_pre_pick_s is not None and cfg.window_post_pick_s is not None:
        pre_samples = max(int(cfg.window_pre_pick_s * sr), 1)
        post_samples = max(int(cfg.window_post_pick_s * sr), 1)
    else:
        half = max(int(cfg.window_s * sr / 2), 8)
        pre_samples = half
        post_samples = half
    
    # Envelope coarse stage uses wider window: production extends POST-ONLY (not pre).
    # _window_samples(pick, pre_samples, post_samples + envelope_max_lag) => asymmetric.
    envelope_max_lag = max(int(cfg.envelope_max_lag_s * sr), 1)
    pre_wide = pre_samples  # NO extension on pre; matches production
    post_wide = post_samples + envelope_max_lag  # Only post extended
    
    # Fine stage window
    fine_half_lag = max(int(cfg.xcorr_fine_half_lag_s * sr), 1)
    
    log.info(f"Config: narrow=(pre={pre_samples} post={post_samples}) wide=(pre={pre_wide} post={post_wide})")
    log.info(f"        fine_half_lag={fine_half_lag} (~{fine_half_lag*dt_us:.1f} µs)")
    log.info(f"        envelope_max_lag={envelope_max_lag} (~{envelope_max_lag*dt_us:.1f} µs)")
    log.info(f"        guide_smooth_epochs={cfg.envelope_guide_smooth_epochs} envelope_min_peak_cc={cfg.envelope_min_peak_cc}")
    
    gather = CASSMTempGather.from_hdf5_compact(Path(args.cache_file))
    sample_count = gather.sample_count
    n_receivers = gather.n_receivers
    
    # Load baseline picks
    baseline_picks = gather._baseline_picks(cfg)
    
    # Envelope smoothing window state
    env_smooth_w = max(int(cfg.envelope_guide_smooth_epochs), 1)
    
    # Process each pair
    for p in pair_indices:
        if p >= gather.n_pairs:
            log.warning(f"Pair {p} out of range (n_pairs={gather.n_pairs}), skipping")
            continue
        
        log.info(f"\nProcessing pair {p} (investigating root cause of missing/zero lags)...")
        
        # Build baseline windows (stacked)
        n_base = cfg.baseline_n_epochs
        if n_base > 1:
            base_raw = np.mean(gather.get_pair(slice(0, n_base), p), axis=0)
        else:
            base_raw = gather.get_pair(0, p)
        
        base_tr = _preprocess_waveform(
            base_raw, sr, cfg, pair_index=p, n_receivers=n_receivers
        )
        
        pick = baseline_picks[p]
        
        # Narrow xcorr window (used for fine waveform xcorr)
        sw = _window_samples(pick, pre_samples, post_samples, sample_count)
        w_len = sw.stop - sw.start
        taper = _cosine_window(w_len, cfg.window_taper_fraction)
        bl_win = base_tr[sw] * taper
        
        # Wide envelope window (asymmetric: post-only extension to match production).
        # Production: _window_samples(pick, pre_samples, post_samples + envelope_max_lag).
        # This ensures negative lags (speedup) are truncated the same way, which may
        # contribute to bias toward positive dt (slowdown) for decorrelated waveforms.
        sw_wide = _window_samples(pick, pre_wide, post_wide, sample_count)
        w_wide = sw_wide.stop - sw_wide.start
        taper_wide = _cosine_window(w_wide, cfg.window_taper_fraction)
        bl_wide_win = base_tr[sw_wide] * taper_wide
        
        # Select epochs to plot
        if explicit_epochs is not None:
            epochs_to_plot = explicit_epochs
        else:
            n_available = min(gather.n_epochs, 2000)
            epochs_to_plot = sorted(random.sample(range(n_available), min(n_debug_plots, n_available)))
        
        # Per-pair tracking for time-series plots
        coarse_lags_all = []
        smoothed_lags_all = []
        final_dts_all = []
        accept_mask_all = []
        
        # Envelope smoothing state
        env_lag_buf = np.full(env_smooth_w, np.nan, dtype=np.float64)
        env_buf_pos = 0
        env_buf_cnt = 0
        
        for e in epochs_to_plot:
            log.info(f"  Epoch {e}/{epochs_to_plot[-1]}...")
            
            tr = _preprocess_waveform(
                gather.get_pair(e, p), sr, cfg, pair_index=p, n_receivers=n_receivers
            )
            
            # Narrow window
            ep_seg = tr[sw]
            w_len_ep = sw.stop - sw.start
            if ep_seg.size < 4 or w_len_ep < 4:
                log.info(f"    Epoch {e}: narrow window too short, skipping")
                continue
            ep_win = ep_seg * taper[:ep_seg.size]
            
            # Wide window
            ep_wide_seg = tr[sw_wide]
            w_wide_ep = sw_wide.stop - sw_wide.start
            if ep_wide_seg.size < 4 or w_wide_ep < 4:
                log.info(f"    Epoch {e}: wide window too short, skipping")
                continue
            ep_wide_win = ep_wide_seg * taper_wide[:ep_wide_seg.size]
            
            # Trim to consistent length
            trim_len = min(len(bl_wide_win), len(ep_wide_win))
            bl_wide_trim = bl_wide_win[:trim_len]
            ep_wide_trim = ep_wide_win[:trim_len]
            trim_len_narrow = min(len(bl_win), len(ep_win))
            bl_win_trim = bl_win[:trim_len_narrow]
            ep_win_trim = ep_win[:trim_len_narrow]
            
            # ===== COARSE ENVELOPE STAGE =====
            coarse_lag, env_cc_val = _envelope_coarse_lag(
                bl_wide_trim, ep_wide_trim,
                max_lag=envelope_max_lag,
                smooth_samples=cfg.envelope_smooth_samples,
                edge_guard_samples=cfg.xcorr_edge_guard_samples,
            )
            
            coarse_lags_all.append(coarse_lag)
            
            # Update smoothing buffer
            if env_cc_val >= cfg.envelope_min_peak_cc:
                buf = env_lag_buf
                buf[env_buf_pos] = coarse_lag
                env_buf_pos = (env_buf_pos + 1) % env_smooth_w
                env_buf_cnt = min(env_buf_cnt + 1, env_smooth_w)
                valid_buf = buf[~np.isnan(buf)]
                smoothed_lag = float(np.median(valid_buf)) if valid_buf.size else coarse_lag
            else:
                smoothed_lag = coarse_lag
            
            smoothed_lags_all.append(smoothed_lag)
            
            # ===== FINE WAVEFORM STAGE =====
            if env_cc_val >= cfg.envelope_min_peak_cc:
                lag, peak_cc, edge_hit = _xcorr_dt_samples(
                    bl_win_trim, ep_win_trim, fine_half_lag,
                    cfg.xcorr_edge_guard_samples,
                    center_lag=int(round(smoothed_lag)),
                )
                accepted = peak_cc >= cfg.xcorr_min_peak_cc and not edge_hit
                final_dts_all.append(lag * (1e6 / sr))
            else:
                lag, peak_cc, edge_hit = 0.0, 0.0, True
                accepted = False
                final_dts_all.append(np.nan)
            
            accept_mask_all.append(accepted)
            
            # Generate plot
            output_path = Path(output_dir or "/tmp") / f"xcorr_pair{p}_epoch{e}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            _make_xcorr_stage_plot(
                pair_idx=p,
                epoch_idx=e,
                epoch_label=gather.epoch_labels[e] if e < len(gather.epoch_labels) else f"epoch_{e}",
                baseline_wide=bl_wide_trim,
                epoch_wide=ep_wide_trim,
                baseline_narrow=bl_win_trim,
                epoch_narrow=ep_win_trim,
                coarse_lag_samp=coarse_lag,
                smoothed_lag_samp=smoothed_lag,
                env_cc=env_cc_val,
                fine_lag_samp=lag,
                fine_cc=peak_cc,
                fine_edge_hit=edge_hit,
                fine_half_lag_samp=fine_half_lag,
                envelope_max_lag_samp=envelope_max_lag,
                sr_hz=sr,
                pick_sample=pick,
                pre_wide=pre_wide,
                post_wide=post_wide,
                pre_narrow=pre_samples,
                post_narrow=post_samples,
                accepted=accepted,
                output_path=output_path,
            )
            
            log.info(f"    Saved: {output_path}")
        
        # After all epochs for this pair: generate per-pair time-series summary plot (Panel 6 equivalent)
        if len(coarse_lags_all) > 1:
            _make_per_pair_summary_plot(
                pair_idx=p,
                epochs_plotted=epochs_to_plot,
                coarse_lags_samp=coarse_lags_all,
                smoothed_lags_samp=smoothed_lags_all,
                final_dts_us=final_dts_all,
                accept_mask=accept_mask_all,
                sr_hz=sr,
                envelope_min_peak_cc=cfg.envelope_min_peak_cc,
                output_dir=output_dir,
            )


def _make_per_pair_summary_plot(
    pair_idx: int,
    epochs_plotted: list,
    coarse_lags_samp: list,
    smoothed_lags_samp: list,
    final_dts_us: list,
    accept_mask: list,
    sr_hz: float,
    envelope_min_peak_cc: float,
    output_dir: str,
) -> None:
    """Generate per-pair time-series summary plot showing guide-smoothing behavior."""
    import matplotlib.pyplot as plt
    
    dt_us = 1e6 / sr_hz
    coarse_lags_us = [l * dt_us for l in coarse_lags_samp]
    smoothed_lags_us = [l * dt_us for l in smoothed_lags_samp]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f"Pair {pair_idx} Time-Series Summary (Guide Smoothing & Acceptance)\n"
        f"envelope_min_peak_cc={envelope_min_peak_cc}",
        fontsize=12, fontweight="bold"
    )
    
    # Panel 1: Coarse vs Smoothed Lag
    ax = axes[0]
    ax.plot(range(len(coarse_lags_us)), coarse_lags_us, 'g-', alpha=0.6, linewidth=1.0, label='Raw coarse lag')
    ax.plot(range(len(smoothed_lags_us)), smoothed_lags_us, 'b-', linewidth=1.5, label='Smoothed guide lag')
    ax.fill_between(range(len(coarse_lags_us)), coarse_lags_us, smoothed_lags_us, alpha=0.15, color='gray')
    ax.set_xlabel('Epoch index (in plot order)')
    ax.set_ylabel('Lag (µs)')
    ax.set_title('Panel 1: Coarse Envelope Lag vs Smoothed Guide Lag (Hypothesis: freeze near zero?)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Panel 2: Final (Fine) DT with acceptance mask
    ax = axes[1]
    colors = ['green' if acc else 'red' for acc in accept_mask]
    ax.scatter(range(len(final_dts_us)), final_dts_us, c=colors, s=20, alpha=0.7, label='Final dt (green=accept, red=reject)')
    ax.plot(range(len(final_dts_us)), final_dts_us, 'k-', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('Epoch index (in plot order)')
    ax.set_ylabel('Final DT (µs)')
    ax.set_title('Panel 2: Fine XCorr DT with Acceptance Gate (Hypothesis: frozen near zero due to guide stall?)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Panel 3: Correlation trace (guide smear effect)
    ax = axes[2]
    # Calculate how far the smoothed lag deviates from coarse for each epoch
    devs = [abs(s - c) for s, c in zip(smoothed_lags_us, coarse_lags_us)]
    ax.bar(range(len(devs)), devs, alpha=0.6, color='orange', label='|Smoothed - Raw Coarse| deviation')
    ax.set_xlabel('Epoch index (in plot order)')
    ax.set_ylabel('Lag deviation (µs)')
    ax.set_title('Panel 3: Guide-Smoothing Median Effect (Large deviation = median suppressing coarse peaks?)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir or "/tmp") / f"xcorr_pair{pair_idx}_summary.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    log.info(f"  Summary plot: {output_path}")


def _make_xcorr_stage_plot(
    pair_idx: int,
    epoch_idx: int,
    epoch_label: str,
    baseline_wide: np.ndarray,
    epoch_wide: np.ndarray,
    baseline_narrow: np.ndarray,
    epoch_narrow: np.ndarray,
    coarse_lag_samp: float,
    smoothed_lag_samp: float,
    env_cc: float,
    fine_lag_samp: float,
    fine_cc: float,
    fine_edge_hit: bool,
    fine_half_lag_samp: int,
    envelope_max_lag_samp: int,
    sr_hz: float,
    pick_sample: int,
    pre_wide: int,
    post_wide: int,
    pre_narrow: int,
    post_narrow: int,
    accepted: bool,
    output_path: Path,
) -> None:
    """Create 6-panel xcorr stage visualization."""
    
    dt_us = 1e6 / sr_hz
    
    try:
        from scipy.signal import hilbert
    except ImportError:
        hilbert = None
    
    # Compute Hilbert envelopes
    if hilbert is not None:
        env_base_wide = np.abs(hilbert(baseline_wide.astype(np.float64)))
        env_epoch_wide = np.abs(hilbert(epoch_wide.astype(np.float64)))
    else:
        env_base_wide = np.abs(baseline_wide.astype(np.float64))
        env_epoch_wide = np.abs(epoch_wide.astype(np.float64))
    
    # Smooth envelopes
    k = 5
    kernel = np.ones(k, dtype=np.float64) / k
    env_base_wide = np.convolve(env_base_wide, kernel, mode="same")
    env_epoch_wide = np.convolve(env_epoch_wide, kernel, mode="same")
    
    # Compute xcorr traces
    cc_wide = _compute_xcorr_trace(env_base_wide, env_epoch_wide)
    cc_narrow = _compute_xcorr_trace(baseline_narrow, epoch_narrow)
    
    # Sample axes (in microseconds)
    ax_wide_us = np.arange(len(baseline_wide)) * dt_us
    ax_narrow_us = np.arange(len(baseline_narrow)) * dt_us
    
    # Lag axes (in microseconds)
    lag_center_wide = len(cc_wide) // 2
    lags_wide_samp = np.arange(len(cc_wide)) - lag_center_wide
    lags_wide_us = lags_wide_samp * dt_us
    
    lag_center_narrow = len(cc_narrow) // 2
    lags_narrow_samp = np.arange(len(cc_narrow)) - lag_center_narrow
    lags_narrow_us = lags_narrow_samp * dt_us
    
    # Create figure with 6 panels
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f"Pair {pair_idx}, Epoch {epoch_idx} ({epoch_label})\n"
        f"Coarse: lag={coarse_lag_samp:.2f}s→{_lag_to_us(coarse_lag_samp, sr_hz):.1f}µs, "
        f"smoothed={_lag_to_us(smoothed_lag_samp, sr_hz):.1f}µs, cc={env_cc:.3f} | "
        f"Fine: lag={_lag_to_us(fine_lag_samp, sr_hz):.1f}µs, cc={fine_cc:.3f}, "
        f"edge_hit={fine_edge_hit} | Accept={accepted}",
        fontsize=11, fontweight="bold"
    )
    
    # Panel 1: Wide context waveform
    ax = axes[0, 0]
    ax.plot(ax_wide_us, baseline_wide, 'b-', alpha=0.6, label='Baseline', linewidth=0.8)
    ax.plot(ax_wide_us, epoch_wide, 'r-', alpha=0.6, label='Epoch', linewidth=0.8)
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Panel 1: Wide Context Waveforms (±{pre_wide}/{post_wide} samp)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Coarse envelope stage waveforms + envelopes
    ax = axes[0, 1]
    ax.plot(ax_wide_us, baseline_wide, 'b-', alpha=0.4, label='Baseline raw', linewidth=0.8)
    ax.plot(ax_wide_us, epoch_wide, 'r-', alpha=0.4, label='Epoch raw', linewidth=0.8)
    ax.plot(ax_wide_us, env_base_wide, 'b-', alpha=0.9, label='Baseline envelope', linewidth=1.2)
    ax.plot(ax_wide_us, env_epoch_wide, 'r-', alpha=0.9, label='Epoch envelope', linewidth=1.2)
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Panel 2: Coarse Stage Envelopes (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Coarse xcorr trace
    ax = axes[1, 0]
    ax.plot(lags_wide_us, cc_wide, 'k-', linewidth=1.0)
    # Search band
    search_lo_us = -envelope_max_lag_samp * dt_us
    search_hi_us = envelope_max_lag_samp * dt_us
    ax.axvspan(search_lo_us, search_hi_us, alpha=0.1, color='green', label='Search band')
    # Coarse peak
    coarse_peak_us = _lag_to_us(coarse_lag_samp, sr_hz)
    ax.plot(coarse_peak_us, np.interp(coarse_lag_samp, lags_wide_samp, cc_wide), 
            'go', markersize=8, label=f'Raw coarse: {coarse_peak_us:.1f}µs')
    # Smoothed coarse
    smoothed_peak_us = _lag_to_us(smoothed_lag_samp, sr_hz)
    ax.axvline(smoothed_peak_us, color='b', linestyle='--', linewidth=1.5, 
               label=f'Smoothed: {smoothed_peak_us:.1f}µs')
    ax.set_xlabel('Lag (µs)')
    ax.set_ylabel('Normalized CC')
    ax.set_title(f'Panel 3: Coarse Envelope XCorr (cc={env_cc:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Fine stage narrow waveforms
    ax = axes[1, 1]
    ax.plot(ax_narrow_us, baseline_narrow, 'b-', alpha=0.7, label='Baseline', linewidth=1.0)
    ax.plot(ax_narrow_us, epoch_narrow, 'r-', alpha=0.7, label='Epoch', linewidth=1.0)
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Panel 4: Fine Stage Waveforms (±{pre_narrow}/{post_narrow} samp)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Fine xcorr trace + parabolic interpolation
    ax = axes[2, 0]
    ax.plot(lags_narrow_us, cc_narrow, 'k-', linewidth=1.0)
    # Fine search band
    fine_search_lo_us = _lag_to_us(int(round(smoothed_lag_samp)) - fine_half_lag_samp, sr_hz)
    fine_search_hi_us = _lag_to_us(int(round(smoothed_lag_samp)) + fine_half_lag_samp, sr_hz)
    ax.axvspan(fine_search_lo_us, fine_search_hi_us, alpha=0.1, color='orange', 
               label=f'Fine search window')
    # Cycle margin band (±1.8 samp at 10 kHz = ±37 µs)
    cycle_margin_samp = 1.8
    cycle_margin_lo = _lag_to_us(smoothed_lag_samp - cycle_margin_samp, sr_hz)
    cycle_margin_hi = _lag_to_us(smoothed_lag_samp + cycle_margin_samp, sr_hz)
    ax.axvspan(cycle_margin_lo, cycle_margin_hi, alpha=0.05, color='purple', 
               label='±1.8 samp margin')
    # Fine peak
    fine_peak_us = _lag_to_us(fine_lag_samp, sr_hz)
    fine_cc_interp = np.interp(fine_lag_samp, lags_narrow_samp, cc_narrow)
    ax.plot(fine_peak_us, fine_cc_interp, 'mo', markersize=10, 
            label=f'Fine peak: {fine_peak_us:.1f}µs')
    
    # Parabolic interpolation visualization
    lag_int = int(round(fine_lag_samp))
    lag_idx = lag_center_narrow + lag_int
    if 0 < lag_idx < len(cc_narrow) - 1:
        y0, y1, y2 = cc_narrow[lag_idx - 1], cc_narrow[lag_idx], cc_narrow[lag_idx + 1]
        lags_fit = np.array([lag_int - 1, lag_int, lag_int + 1])
        lags_fit_us = lags_fit * dt_us
        # Fit parabola
        z = np.polyfit(lags_fit, [y0, y1, y2], 2)
        lags_smooth = np.linspace(lag_int - 1.5, lag_int + 1.5, 50)
        lags_smooth_us = lags_smooth * dt_us
        parabola = np.polyval(z, lags_smooth)
        ax.plot(lags_smooth_us, parabola, 'm--', linewidth=1.5, alpha=0.7, label='Parabola fit')
    
    # Edge-hit marker
    if fine_edge_hit:
        ax.text(0.5, 0.95, 'EDGE HIT', transform=ax.transAxes, 
                ha='center', va='top', fontsize=12, color='red', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Lag (µs)')
    ax.set_ylabel('Normalized CC')
    ax.set_title(f'Panel 5: Fine XCorr (cc={fine_cc:.3f}, half_lag={fine_half_lag_samp} samp)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Per-pair time-series summary (empty placeholder for now)
    ax = axes[2, 1]
    ax.text(0.5, 0.5, 'Per-pair summary\n(generated after all epochs)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=10, style='italic')
    ax.set_title('Panel 6: (Reserved for time-series)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to cussp_cassm_config.yaml")
    ap.add_argument("--pair-indices", type=int, nargs="+", 
                   default=[923, 924, 925, 927, 639],
                   help="Pair indices to process")
    ap.add_argument("--n-debug-plots", type=int, default=10,
                   help="Number of random epochs to plot per pair")
    ap.add_argument("--output-dir", type=str, 
                   default="/home/chopp/cassm_local/live/xcorr_diag_plots",
                   help="Output directory for PNG plots")
    ap.add_argument("--epoch-indices", type=int, nargs="+", default=None,
                   help="Explicit epoch indices to plot (overrides random selection)")
    
    args = ap.parse_args()
    
    run_xcorr_diag(
        config_path=args.config,
        pair_indices=args.pair_indices,
        n_debug_plots=args.n_debug_plots,
        output_dir=args.output_dir,
        explicit_epochs=args.epoch_indices,
    )
    
    log.info(f"Diagnostic plots saved to {args.output_dir}")
