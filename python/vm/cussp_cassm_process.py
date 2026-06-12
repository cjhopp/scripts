#!/usr/bin/env python3
"""Headless CUSSP CASSM processing pipeline.

This script is intended to run on the recorder-side/headless server.
It ingests new CASSM epoch folders, updates a temp-gather cache, computes
metrics, and publishes a compact bundle for visualization apps running on a
small remote VM.

Published artifacts (defaults under /data/chet-cussp/cassm/live):
- cassm_tempgather_full.h5: full local cache on the processing server
- cassm_dashboard_bundle.npz: compact bundle for visualization
- cassm_dashboard_manifest.json: metadata and update time
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import correlate as _sp_correlate

try:
    import yaml
except ImportError:
    yaml = None

# FWI dt estimator — lazy import so the module is optional at startup.
# Imported inside run_once() when fwi_dt_enabled is True.
_fwi_estimate_dt = None
_fwi_build_context = None


LOG = logging.getLogger("cussp_cassm_process")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _safe_parse_epoch_time(name: str) -> pd.Timestamp:
    try:
        if len(name) == 14 and name.isdigit():
            return pd.to_datetime(name, format="%Y%m%d%H%M%S", utc=True)
        if len(name) == 12 and name.isdigit():
            return pd.to_datetime(name, format="%Y%m%d%H%M", utc=True)
        ts = pd.to_datetime(name, utc=True, errors="coerce")
        if pd.isna(ts):
            return pd.Timestamp("1970-01-01", tz="UTC")
        return ts
    except Exception:
        return pd.Timestamp("1970-01-01", tz="UTC")


def _window_samples(pick_idx: int, pre_samples: int, post_samples: int, n_samples: int) -> slice:
    """Return an asymmetric slice [pick - pre_samples, pick + post_samples], clamped to [0, n_samples)."""
    i0 = max(pick_idx - pre_samples, 0)
    i1 = min(pick_idx + post_samples, n_samples)
    return slice(i0, i1)


def _cosine_window(n: int, taper_frac: float = 0.10) -> np.ndarray:
    """Cosine-tapered (Tukey) window — matches MATLAB cosWindow used in dsiCASSM."""
    w = np.ones(n, dtype=np.float64)
    taper_n = max(int(round(n * taper_frac)), 1)
    if taper_n < n // 2:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper_n) / taper_n))
        w[:taper_n] = ramp
        w[n - taper_n:] = ramp[::-1]
    return w


def _xcorr_dt_samples(
    baseline_win: np.ndarray,
    epoch_win: np.ndarray,
    max_lag: int,
    edge_guard_samples: int = 1,
    center_lag: int = 0,
) -> Tuple[float, float, bool]:
    """Normalized cross-correlation lag (sub-sample precision via parabolic fit).

    Returns (lag_samples, peak_cc, edge_hit) where positive lag means the epoch
    arrives LATER (dt > 0 = slowdown). This matches the MATLAB
    dsiCASSMdelaySequenceEstWinPar convention.

    *center_lag* shifts the search window by that many samples (integer), allowing
    the envelope-guided mode to center the fine waveform search on the coarse
    energy peak.  The returned lag is always relative to zero (not center_lag),
    so the sign convention is preserved regardless of centering.
    """
    b = baseline_win.astype(np.float64)
    e = epoch_win.astype(np.float64)
    b_norm = np.linalg.norm(b)
    e_norm = np.linalg.norm(e)
    if b_norm == 0.0 or e_norm == 0.0:
        return 0.0, 0.0, True
    cc = _sp_correlate(e / e_norm, b / b_norm, mode="full")
    # lags: cc[n-1+k] corresponds to lag k
    n = len(b)
    center = len(cc) // 2
    search_center = center + int(round(center_lag))
    lo = max(search_center - max_lag, 0)
    hi = min(search_center + max_lag + 1, len(cc))
    cc_win = cc[lo:hi]
    if cc_win.size == 0:
        return 0.0, 0.0, True
    peak_idx = int(np.argmax(cc_win))
    peak_cc = float(cc_win[peak_idx])
    edge_guard = max(int(edge_guard_samples), 0)
    edge_hit = peak_idx <= edge_guard or peak_idx >= (cc_win.size - 1 - edge_guard)
    lag_int = (lo + peak_idx) - center  # integer lag in samples, always relative to true zero
    # Parabolic sub-sample refinement
    abs_idx = lo + peak_idx
    if 0 < abs_idx < len(cc) - 1:
        y0, y1, y2 = cc[abs_idx - 1], cc[abs_idx], cc[abs_idx + 1]
        denom = 2.0 * y1 - y0 - y2
        sub = float(np.clip((y2 - y0) / (2.0 * denom), -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
    else:
        sub = 0.0
    return float(lag_int) + sub, peak_cc, edge_hit


def _envelope_coarse_lag(
    baseline_win: np.ndarray,
    epoch_win: np.ndarray,
    max_lag: int,
    smooth_samples: int = 5,
    edge_guard_samples: int = 1,
) -> Tuple[float, float]:
    """Envelope cross-correlation for cycle-skip-resistant coarse lag estimation.

    Computes the analytic signal envelope (Hilbert transform → |z|) of each
    tapered window, applies a uniform smoothing kernel to suppress intra-cycle
    ripple, then cross-correlates the normalized envelopes over ±max_lag samples.

    Returns (coarse_lag_samples, envelope_peak_cc).
    Positive lag = epoch arrives LATER — same sign convention as _xcorr_dt_samples.
    Returns (0.0, 0.0) on failure (zero-norm, too-short window, scipy unavailable).
    """
    try:
        from scipy.signal import hilbert as _hilbert
    except ImportError:
        return 0.0, 0.0

    b = baseline_win.astype(np.float64)
    e = epoch_win.astype(np.float64)

    if b.size < 4 or e.size < 4:
        return 0.0, 0.0

    # Analytic envelope via Hilbert transform
    env_b = np.abs(_hilbert(b))
    env_e = np.abs(_hilbert(e))

    # Uniform smoothing to suppress intra-cycle ripple
    k = max(int(smooth_samples), 1)
    if k > 1:
        kernel = np.ones(k, dtype=np.float64) / k
        env_b = np.convolve(env_b, kernel, mode="same")
        env_e = np.convolve(env_e, kernel, mode="same")

    b_norm = np.linalg.norm(env_b)
    e_norm = np.linalg.norm(env_e)
    if b_norm == 0.0 or e_norm == 0.0:
        return 0.0, 0.0

    cc = _sp_correlate(env_e / e_norm, env_b / b_norm, mode="full")
    center = len(cc) // 2
    lo = max(center - max_lag, 0)
    hi = min(center + max_lag + 1, len(cc))
    cc_win = cc[lo:hi]
    if cc_win.size == 0:
        return 0.0, 0.0

    peak_idx = int(np.argmax(cc_win))
    peak_cc = float(cc_win[peak_idx])

    # Parabolic sub-sample refinement
    abs_idx = lo + peak_idx
    if 0 < abs_idx < len(cc) - 1:
        y0, y1, y2 = cc[abs_idx - 1], cc[abs_idx], cc[abs_idx + 1]
        denom = 2.0 * y1 - y0 - y2
        sub = float(np.clip((y2 - y0) / (2.0 * denom), -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
    else:
        sub = 0.0

    lag_int = (lo + peak_idx) - center
    return float(lag_int) + sub, peak_cc


def _despike_single_epoch_dt(
    dt_us: np.ndarray,
    rms: np.ndarray,
    mad_thresh: float = 5.0,
) -> np.ndarray:
    """Mask isolated single-epoch dt spikes in-place and return spike mask.

    A point is marked as a spike when it differs from the 3-point median
    (e-1, e, e+1) by more than mad_thresh * global MAD for that pair trace.
    Only epochs with rms>0 and finite dt are considered eligible.
    """
    spike_mask = np.zeros(dt_us.shape, dtype=np.uint8)
    if dt_us.size == 0:
        return spike_mask

    # Operate directly on dt_us so masked spikes persist into the published bundle.
    out = dt_us
    n_pairs = out.shape[0]
    for p in range(n_pairs):
        row = out[p, :]
        valid = np.isfinite(row) & (rms[p, :] > 0)
        if np.count_nonzero(valid) < 4:
            continue

        finite = row[valid].astype(np.float64)
        global_mad = float(np.median(np.abs(finite - np.median(finite))))
        if global_mad < 1e-6:
            continue

        work = np.full(row.shape, np.nan, dtype=np.float64)
        work[valid] = row[valid].astype(np.float64)
        left = np.concatenate(([work[0]], work[:-1]))
        right = np.concatenate((work[1:], [work[-1]]))
        with np.errstate(invalid="ignore"):
            med3 = np.nanmedian(np.stack([left, work, right], axis=0), axis=0)
        spike = valid & (np.abs(work - med3) > (mad_thresh * global_mad))
        if np.any(spike):
            row[spike] = np.nan
            spike_mask[p, spike] = 1

    return spike_mask


def _mask_short_branch_runs(
    dt_us: np.ndarray,
    rms: np.ndarray,
    max_run_len_epochs: int = 4,
    min_amp_us: float = 35.0,
    neighbor_tol_us: float = 12.0,
) -> np.ndarray:
    """Mask short branch-like dt runs that depart then return to local level.

    Targets multi-epoch phase-wrap toggles that survive single-epoch despiking.
    A run is masked when:
      1) run length <= max_run_len_epochs,
      2) valid neighbors exist on both sides,
      3) left/right neighbor levels are close (return-to-level), and
      4) run level differs from neighbor level by at least min_amp_us.
    """
    mask = np.zeros(dt_us.shape, dtype=np.uint8)
    if dt_us.size == 0:
        return mask

    n_pairs, n_epochs = dt_us.shape
    for p in range(n_pairs):
        row = dt_us[p, :]
        valid = np.isfinite(row) & (rms[p, :] > 0)
        i = 0
        while i < n_epochs:
            if not valid[i]:
                i += 1
                continue

            j = i
            while j + 1 < n_epochs and valid[j + 1]:
                j += 1

            run_len = j - i + 1
            if run_len <= max_run_len_epochs and i > 0 and j < n_epochs - 1:
                if valid[i - 1] and valid[j + 1]:
                    left = float(row[i - 1])
                    right = float(row[j + 1])
                    run_level = float(np.nanmedian(row[i:j + 1]))
                    if (
                        abs(left - right) <= neighbor_tol_us
                        and abs(run_level - 0.5 * (left + right)) >= min_amp_us
                    ):
                        row[i:j + 1] = np.nan
                        mask[p, i:j + 1] = 1

            i = j + 1

    return mask


def _dtw_dt_samples(
    baseline_win: np.ndarray,
    epoch_win: np.ndarray,
    max_shift: int,
    strain_limit: float = 2.0,
    edge_guard_samples: int = 1,
    signal_end_j: Optional[int] = None,
) -> Tuple[float, float, bool]:
    """Dynamic Time Warping (DTW) for cycle-skip-resistant large-dt estimation.

    Aligns epoch to baseline using accumulated-cost DTW with a Sakoe-Chiba band
    of width max_shift, then extracts the lag from the early-arrival portion of
    the optimal path.  Handles cycle-skipping (large dt over multiple periods)
    and waveform shape change (decorrelation).

    Algorithm:
      1. Normalise both windows to unit L2 norm.
      2. Forward DP over the band |j - i| <= max_shift:
           D[i,j] = (e[i-1] - b[j-1])² + min(D[i-1,j-1], D[i,j-1], D[i-1,j])
         (diagonal, horizontal, vertical steps — standard DTW).
      3. Backtrack from (ne, nb) to (1, 1) along the minimum-cost path.
      4. Lag = median of (path_i - path_j) over the first-arrival region.
         Positive lag = epoch arrives LATER (slowdown), matching CUSSP convention.
      5. Quality = NCC after shifting epoch by -lag_int to align with baseline.

    *strain_limit* is kept as an API parameter (used by MetricConfig) but is
    superseded by the Sakoe-Chiba band as the primary cycle-skip guard; it is
    not applied inside this function.

    Returns (lag_samples, quality, edge_hit) — same signature as _xcorr_dt_samples.
    """
    b = baseline_win.astype(np.float64)
    e = epoch_win.astype(np.float64)
    nb, ne = len(b), len(e)

    if nb < 4 or ne < 4:
        return 0.0, 0.0, True

    b_norm = np.linalg.norm(b)
    e_norm = np.linalg.norm(e)
    if b_norm < 1e-30 or e_norm < 1e-30:
        return 0.0, 0.0, True

    b_n = b / b_norm
    e_n = e / e_norm

    # Forward pass — Sakoe-Chiba band: only fill cells where |j - i| <= max_shift.
    # D[i, j] is the minimum accumulated cost to align epoch[0..i-1] with baseline[0..j-1].
    D = np.full((ne + 1, nb + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0

    for i in range(1, ne + 1):
        j_lo = max(1, i - max_shift)
        j_hi = min(nb, i + max_shift)
        for j in range(j_lo, j_hi + 1):
            cost = (e_n[i - 1] - b_n[j - 1]) ** 2
            prev = D[i - 1, j - 1]                          # diagonal (always in band)
            if j > 1 and D[i, j - 1] < prev:
                prev = D[i, j - 1]                           # horizontal
            if i > 1 and D[i - 1, j] < prev:
                prev = D[i - 1, j]                           # vertical
            if prev < np.inf:
                D[i, j] = cost + prev

    if D[ne, nb] == np.inf:
        return 0.0, 0.0, True

    # Backtrack from (ne, nb) to (1, 1) — collect path in reverse, then flip.
    path_i_rev: List[int] = [ne]
    path_j_rev: List[int] = [nb]
    i, j = ne, nb
    while i > 1 or j > 1:
        if i <= 1:
            j -= 1
        elif j <= 1:
            i -= 1
        else:
            # Choose the predecessor with the lowest accumulated cost.
            best = D[i - 1, j - 1]
            i_next, j_next = i - 1, j - 1
            if D[i, j - 1] < best:
                best = D[i, j - 1]
                i_next, j_next = i, j - 1
            if D[i - 1, j] < best:
                i_next, j_next = i - 1, j
            i, j = i_next, j_next
        path_i_rev.append(i)
        path_j_rev.append(j)

    path_i = np.array(path_i_rev[::-1], dtype=np.int32)
    path_j = np.array(path_j_rev[::-1], dtype=np.int32)

    # Extract lag from the peak-energy region of the alignment path.
    # path_i indexes epoch (1-based), path_j indexes baseline (1-based).
    # epoch[i-1] aligns with baseline[j-1].  If epoch is delayed by k samples
    # then j ≈ i - k, so (path_i - path_j) ≈ k.  Positive = epoch arrives LATER.
    #
    # Using the peak-energy region is robust to windows where the first arrival is
    # offset from the window start by a pre-pick noise buffer (CUSSP: ~5 samples),
    # and to tests with centered wavelets — the first-arrival peak is always in this
    # region regardless of window position.
    # Restrict lag extraction to the signal region only (ignoring the noisy window
    # extension used to capture large-lag arrivals).  signal_end_j is the last
    # 1-based baseline index that contains signal (= pre_samples + post_samples).
    # When None, use the full baseline window (backwards-compatible).
    j_cap = signal_end_j if (signal_end_j is not None and 1 <= signal_end_j <= nb) else nb

    peak_j = int(np.argmax(b_n[:j_cap] ** 2)) + 1   # 1-based, within signal region
    n_around = max(3, j_cap // 8)                    # ±n_around samples around peak
    j_lo = max(1, peak_j - n_around)
    j_hi = min(j_cap, peak_j + n_around)
    signal_mask = (path_j >= j_lo) & (path_j <= j_hi)
    if signal_mask.sum() >= 2:
        warps = (path_i[signal_mask] - path_j[signal_mask]).astype(np.float64)
    else:
        # Fallback: use first half of signal-region path
        path_in_sig = path_j <= j_cap
        pts_sig = path_j[path_in_sig]
        if pts_sig.size >= 2:
            n_use = max(pts_sig.size // 2, 2)
            warps = (path_i[path_in_sig][:n_use] - path_j[path_in_sig][:n_use]).astype(np.float64)
        else:
            n_use = max(len(path_i) // 2, 2)
            warps = (path_i[:n_use] - path_j[:n_use]).astype(np.float64)
    lag_int = int(np.median(warps)) if warps.size > 0 else 0
    lag_int = int(np.clip(lag_int, -max_shift, max_shift))

    # Quality: NCC after undoing the lag, computed over the signal region only.
    if lag_int == 0:
        epoch_shifted = e_n.copy()
    else:
        epoch_shifted = np.roll(e_n, -lag_int)
        if lag_int > 0:
            epoch_shifted[-lag_int:] = 0.0   # clear wrapped-around tail
        else:
            epoch_shifted[:-lag_int] = 0.0   # clear wrapped-around head

    b_sig = b_n[:j_cap]
    e_sig = epoch_shifted[:j_cap]
    b_sig_norm = float(np.linalg.norm(b_sig))
    e_sig_norm = float(np.linalg.norm(e_sig))
    if b_sig_norm > 1e-30 and e_sig_norm > 1e-30:
        cc_val = float(np.dot(b_sig / b_sig_norm, e_sig / e_sig_norm))
    else:
        cc_val = 0.0

    # Edge hit: lag saturated against the band boundary.
    edge_hit = abs(lag_int) >= max_shift - edge_guard_samples

    return float(lag_int), cc_val, edge_hit


def _spectral_ratio_slope(
    baseline_win: np.ndarray,
    epoch_win: np.ndarray,
    sample_rate_hz: float,
    fmin_hz: float = 500.0,
    fmax_hz: float = 4000.0,
) -> float:
    """Slope of log(|FFT(baseline)| / |FFT(epoch)|) vs. frequency.

    Matches MATLAB th(85) computed in testingCentFreq.m / testingspectralratio.m:
      specRat = log(specbase ./ specmon);
      slope   = polyfit(f(range), specRat(range), 1);

    A negative slope means the epoch has LESS high-frequency content relative to
    the baseline (increased attenuation / higher t*).  Units: nepers / Hz.
    """
    if baseline_win.size < 4 or epoch_win.size < 4:
        return 0.0
    n_fft = max(int(2 ** np.ceil(np.log2(max(len(baseline_win), 64)))), 64)
    spec_b = np.abs(np.fft.rfft(baseline_win.astype(np.float64), n=n_fft))
    spec_e = np.abs(np.fft.rfft(epoch_win.astype(np.float64), n=n_fft))
    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate_hz))
    mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz)
    if mask.sum() < 4:
        return 0.0
    eps = np.finfo(np.float64).tiny
    # log(a/b) rewritten as log(a) - log(b) to avoid float64 overflow when a >> b or b << eps.
    log_ratio = (np.log(np.maximum(spec_b[mask], eps))
                 - np.log(np.maximum(spec_e[mask], eps)))
    slope = float(np.polyfit(freqs_hz[mask], log_ratio, 1)[0])
    return slope


@dataclass
class MetricConfig:
    pick_search_s: float = 0.012
    window_s: float = 0.003
    clip_first_s: float = 0.002
    mute_first_s: float = 0.002
    taper_fraction: float = 0.01
    filter_low_hz: float = 0.0
    filter_high_hz: float = 0.0
    filter_order: int = 4
    accel_filter_low_hz: Optional[float] = None
    accel_filter_high_hz: Optional[float] = None
    hydro_filter_low_hz: Optional[float] = None
    hydro_filter_high_hz: Optional[float] = None
    # Per-sensor-type clip and mute overrides for hydrophones.
    # Hydrophones lack the low-frequency electrical noise that contaminates accelerometer
    # channels, so the hard-zero clip and pick-search mute can be reduced or eliminated.
    # When None, falls back to clip_first_s / mute_first_s respectively.
    hydro_clip_first_s: Optional[float] = None
    hydro_mute_first_s: Optional[float] = None
    # Picker selection: 'aic', 'stalta', or 'gradient'
    picker: str = "aic"
    stalta_short_s: float = 0.0002   # STA window (s)
    stalta_long_s: float = 0.0015    # LTA window (s)
    stalta_threshold: float = 3.0    # STA/LTA trigger threshold
    baseline_n_epochs: int = 1       # number of leading epochs to stack for the baseline pick
    aic_margin_samples: int = 10     # edge guard: AIC search restricted to [margin, n-margin]
    aic_min_snr: float = 0.0         # minimum SNR to accept an AIC pick (0 = accept all)
    # dt measurement method
    # "xcorr" (default, matches MATLAB dsiCASSMdelaySequenceEstWinPar): pick baseline ONCE with
    #   the selected picker, then measure per-epoch dt via windowed normalized cross-correlation
    #   against the preprocessed baseline trace window.  Achieves sub-sample precision (±1-5 µs).
    # "pick": legacy independent-pick mode — re-runs the picker on every epoch trace.  Noisy
    #   (±20-100 µs at 50 kHz) but does not require storing baseline waveforms.
    dt_method: str = "xcorr"
    xcorr_max_lag_s: float = 0.001   # ±max xcorr search lag (s)
    xcorr_accept_max_lag_s: float = 0.001        # accept only |lag| <= this bound (s) — accel channels
    xcorr_accept_max_lag_hydro_s: float = 0.001  # accept only |lag| <= this bound (s) — hydro channels (ch≥49)
    xcorr_accept_max_lag_dm_hydro_s: float = 0.0015  # DM* -> hydro override for larger legitimate shifts
    source_boreholes: Optional[Tuple[str, ...]] = None
    xcorr_min_peak_cc: float = 0.0    # reject xcorr peaks below this normalized correlation
    xcorr_edge_guard_samples: int = 1 # reject peaks within this many samples of the lag-window edge
    xcorr_despike_single_epoch: bool = True  # apply isolated-spike masking to dt during processing
    xcorr_despike_mad_thresh: float = 5.0  # spike threshold in units of global MAD
    xcorr_mask_short_runs: bool = True  # mask short branch-like multi-epoch excursions
    xcorr_short_run_max_len_epochs: int = 4
    xcorr_short_run_min_amp_us: float = 35.0
    xcorr_short_run_neighbor_tol_us: float = 12.0
    window_taper_fraction: float = 0.10  # cosine taper fraction for the xcorr/metric window
    # Asymmetric xcorr window: small pre-pick noise buffer + larger post-pick signal region.
    # When both are set, these override window_s.  Recommended: pre=0.0002–0.0005 s,
    # post=0.001–0.002 s — minimises noise while capturing the P-wave wavelet.
    window_pre_pick_s: Optional[float] = None
    window_post_pick_s: Optional[float] = None
    # Envelope-guided xcorr (cycle-skip resistant two-stage correlation)
    # Stage 1: Hilbert envelope xcorr over a wide lag window to find the coarse
    #   energy-peak shift, free of cycle ambiguity.
    # Stage 2: Waveform xcorr centered on that coarse lag with a narrow
    #   ±xcorr_fine_half_lag_s search, locking onto the correct cycle.
    # When envelope_peak_cc < envelope_min_peak_cc the epoch is NaN-rejected
    # (not silently fallen back to unguided xcorr) so that problematic pairs
    # are visible in QC and the dashboard pairwise plots.
    envelope_guide_xcorr: bool = False       # master switch; False = existing behaviour
    envelope_max_lag_s: float = 0.00025      # coarse search half-width (±0.25 ms default)
    envelope_smooth_samples: int = 5         # uniform smoothing kernel half-width for envelope
    envelope_min_peak_cc: float = 0.20       # envelope cc below this → NaN reject
    xcorr_fine_half_lag_s: float = 0.0003   # ±fine waveform search around coarse lag
                                             # (~14 samples at 48 kHz; spans ~6 half-cycles
                                             # at 10 kHz — keep < ½ dominant period to avoid
                                             # cycle-skip within the fine search window)
    envelope_guide_smooth_epochs: int = 7    # causal running-median window (in epochs) applied
                                             # to per-pair coarse lags before using as fine-search
                                             # center.  Suppresses epoch-to-epoch noise in the
                                             # envelope estimate without introducing future-epoch
                                             # bias.  Set to 1 to disable smoothing.
    # FWI-derived dt for DM*→TS hydrophone pairs (Phase 1 CDD-TLFWI hybrid).
    # When fwi_dt_enabled=True and the pair is a DM*→TS hydrophone pair, the
    # xcorr/envelope dt measurement is replaced by a wave-equation forward model
    # + correlative misfit 1D line search.  All other pairs use xcorr as normal.
    # The FWIContext object (pre-computed grid, v_p model, solver, wavelets) is
    # injected via compute_metrics(fwi_context=...) rather than stored here, so
    # MetricConfig remains serialisable.
    # NOTE: Phase 1 disables FWI in favor of DTW for large-dt recovery.
    fwi_dt_enabled: bool = False  # Disabled in Phase 1; kept for Phase 2 (spatial inversion)
    fwi_dt_sources_csv: str = ""
    fwi_dt_receivers_csv: str = ""
    fwi_dt_solver: str = "fd2d"              # "analytic", "fd2d", or "devito"
    fwi_dt_grid_dx_m: float = 0.5
    fwi_dt_grid_dz_m: float = 0.5
    fwi_dt_grid_padding_m: float = 20.0
    fwi_dt_vp_background_mps: float = 3000.0
    fwi_dt_freq_bands: Optional[List[Tuple[float, float]]] = None  # None → default CUSSP bands
    fwi_dt_search_max_ms: float = 2.0        # ±initial dt search range (ms) for coarsest band
    fwi_dt_min_ncc: float = 0.2             # minimum NCC to accept FWI-derived dt
    fwi_dt_gate_pre_ms: Optional[float] = None   # P-wave gate pre-pick (ms); None → use window_pre_pick_s
    fwi_dt_gate_post_ms: Optional[float] = None  # P-wave gate post-pick (ms); None → use window_post_pick_s
    fwi_dt_cpml_thickness: int = 20         # CPML absorbing boundary layers for FD2DSolver
    # Dynamic Time Warping (DTW) for large-dt estimation on DM*→TS pairs.
    # DTW is a cycle-skip-resistant alternative to plain xcorr: it finds the
    # optimal time-axis warping between baseline and epoch via accumulated-error DP,
    # then extracts a cycle-unambiguous lag (integer samples) + a fine xcorr refine.
    # Applied only to DM*→TS hydrophone pairs when dtw_enabled=True;
    # all other pairs use xcorr as normal.
    dtw_enabled: bool = True  # Phase 1 default: enabled for DM*→TS pairs (large-dt recovery)
    dtw_max_shift_ms: float = 0.5  # maximum warp magnitude in milliseconds (~24 samples @ 48 kHz)
    dtw_strain_limit: float = 2.0  # max local warp slope di/dj; ~2 allows ±50% slowdown/speedup
    dtw_min_ncc: float = 0.2  # minimum NCC (after DTW warp) to accept result


def _aic_pick(x: np.ndarray, margin: int = 10, min_snr: float = 0.0) -> Tuple[int, float]:
    """Maeda (1985) AIC first-break picker — faithful translation of SeisComP araic.cpp.

    AIC(k) = k * log10(sum(x²[:k]) / (k-1)) + (n-k-1) * log10(sum(x²[k:]) / (n-k-1))

    The search is constrained to [margin, n-margin] to avoid edge artefacts.
    SNR is computed as SeisComP does: 0.707 * peak(|x[kmin:]|) / RMS(x[margin:kmin]).

    Returns (kmin, snr) where kmin is a 0-based index within x.  If snr < min_snr
    (and min_snr > 0) the pick is rejected and (0, snr) is returned so the caller
    falls back to the start of the search window.
    """
    n = len(x)
    margin = max(margin, 1)
    imin, imax = margin, n - margin
    if n < 2 * margin + 2 or imin >= imax:
        return 0, 0.0

    xf = x.astype(np.float64)
    cf = xf ** 2
    total = cf.sum()
    if total == 0.0:
        return 0, 0.0

    eps = np.finfo(np.float64).tiny
    cs = np.cumsum(cf)                          # cs[i] = sum(cf[:i+1])
    k_arr = np.arange(imin, imax, dtype=np.float64)
    ki = k_arr.astype(np.intp)

    # sum(cf[:k]) = cs[k-1];  sum(cf[k:]) = total - cs[k-1]
    sum1 = cs[ki - 1]
    sum2 = total - sum1
    var1 = np.maximum(sum1 / np.maximum(k_arr - 1.0, 1.0), eps)
    var2 = np.maximum(sum2 / np.maximum(n - k_arr - 1.0, 1.0), eps)
    aic = k_arr * np.log10(var1) + (n - k_arr - 1.0) * np.log10(var2)
    kmin = imin + int(np.argmin(aic))

    # SNR: RMS of x in pre-pick window vs peak of |x| in post-pick window
    pre = cf[margin:kmin]
    post = np.abs(xf[kmin:n - margin])
    noise_rms = float(np.sqrt(np.mean(pre))) if pre.size > 0 else float(eps)
    signal_peak = float(np.max(post)) if post.size > 0 else 0.0
    snr = 0.707 * signal_peak / max(noise_rms, float(eps))

    if min_snr > 0.0 and snr < min_snr:
        return 0, snr  # SNR check failed — caller falls back to start of window

    return kmin, snr


def _stalta_pick(x: np.ndarray, n_sta: int, n_lta: int, threshold: float) -> int:
    """Causal STA/LTA first-break picker on energy characteristic function.

    Returns the first sample where STA/LTA >= *threshold*, or the sample of
    maximum ratio when the threshold is never reached.
    """
    n = len(x)
    cf = x.astype(np.float64) ** 2
    eps = 1e-30

    cs = np.cumsum(cf)
    # STA: causal running mean over n_sta samples
    sta = np.empty(n)
    sta[:n_sta] = cs[:n_sta] / np.arange(1, n_sta + 1, dtype=np.float64)
    sta[n_sta:] = (cs[n_sta:] - cs[:-n_sta]) / n_sta
    # LTA: causal running mean over n_lta samples
    lta = np.empty(n)
    lta[:n_lta] = cs[:n_lta] / np.arange(1, n_lta + 1, dtype=np.float64)
    lta[n_lta:] = (cs[n_lta:] - cs[:-n_lta]) / n_lta

    ratio = sta / np.maximum(lta, eps)
    triggered = np.where(ratio >= threshold)[0]
    return int(triggered[0]) if len(triggered) else int(np.argmax(ratio))


def _apply_picker(x: np.ndarray, config: "MetricConfig", sample_rate_hz: float) -> int:
    """Dispatch to the configured picker; return 0-based index within *x*."""
    if config.picker == "aic":
        k, _snr = _aic_pick(x, margin=config.aic_margin_samples, min_snr=config.aic_min_snr)
        return k
    if config.picker == "stalta":
        n_sta = max(int(config.stalta_short_s * sample_rate_hz), 2)
        n_lta = max(int(config.stalta_long_s * sample_rate_hz), n_sta + 1)
        return _stalta_pick(x, n_sta, n_lta, config.stalta_threshold)
    # gradient fallback
    grad = np.abs(np.diff(x, prepend=x[:1]))
    return int(np.argmax(grad))


def _bandpass_bounds_for_pair(
    pair_index: Optional[int],
    n_receivers: Optional[int],
    config: MetricConfig,
) -> Tuple[float, float]:
    """Return effective (low, high) Hz bounds for a pair.

    Receiver family split assumes CUSSP mapping:
    rec 0..47 accelerometers, rec 48..71 hydrophones.
    """
    low = float(config.filter_low_hz)
    high = float(config.filter_high_hz)
    if pair_index is None or n_receivers is None or n_receivers <= 0:
        return low, high

    rec_idx = int(pair_index) % int(n_receivers)
    is_accel = rec_idx < min(48, int(n_receivers))
    is_hydro = rec_idx >= 48

    if is_accel:
        if config.accel_filter_low_hz is not None:
            low = float(config.accel_filter_low_hz)
        if config.accel_filter_high_hz is not None:
            high = float(config.accel_filter_high_hz)
    elif is_hydro:
        if config.hydro_filter_low_hz is not None:
            low = float(config.hydro_filter_low_hz)
        if config.hydro_filter_high_hz is not None:
            high = float(config.hydro_filter_high_hz)

    return low, high


def _maybe_bandpass(
    x: np.ndarray,
    sample_rate_hz: float,
    config: MetricConfig,
    pair_index: Optional[int] = None,
    n_receivers: Optional[int] = None,
) -> np.ndarray:
    """Optionally apply a low/high/band-pass filter to a 1D waveform segment."""
    low, high = _bandpass_bounds_for_pair(pair_index, n_receivers, config)
    if low <= 0.0 and high <= 0.0:
        return x

    nyq = 0.5 * float(sample_rate_hz)
    lo = max(low / nyq, 1.0e-6) if low > 0.0 else None
    hi = min(high / nyq, 0.999999) if high > 0.0 else None

    if lo is not None and hi is not None and lo >= hi:
        return x

    if x.size < 8:
        return x

    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception:
        return x

    order = max(int(config.filter_order), 1)
    try:
        if lo is not None and hi is not None:
            sos = butter(order, [lo, hi], btype="band", output="sos")
        elif lo is not None:
            sos = butter(order, lo, btype="high", output="sos")
        elif hi is not None:
            sos = butter(order, hi, btype="low", output="sos")
        else:
            return x
        return sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return x


def _preprocess_waveform(
    x: np.ndarray,
    sample_rate_hz: float,
    config: MetricConfig,
    pair_index: Optional[int] = None,
    n_receivers: Optional[int] = None,
) -> np.ndarray:
    """Apply clip, taper, and optional filtering before picking/metrics."""
    y = np.asarray(x, dtype=np.float32).copy()
    if y.size == 0:
        return y

    is_hydro = (
        pair_index is not None
        and n_receivers is not None
        and (pair_index % n_receivers) >= 48
    )
    effective_clip_s = (
        config.hydro_clip_first_s
        if is_hydro and config.hydro_clip_first_s is not None
        else config.clip_first_s
    )
    clip_n = max(int(float(effective_clip_s) * float(sample_rate_hz)), 0)
    clip_n = min(clip_n, y.size)
    if clip_n > 0:
        y[:clip_n] = 0.0

    taper_fraction = max(float(config.taper_fraction), 0.0)
    taper_n = int(taper_fraction * y.size)
    taper_n = min(max(taper_n, 0), max(y.size - clip_n, 0))
    if taper_n > 1 and clip_n < y.size:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, taper_n, dtype=np.float32))
        y[clip_n : clip_n + taper_n] *= ramp

    return _maybe_bandpass(
        y,
        sample_rate_hz,
        config,
        pair_index=pair_index,
        n_receivers=n_receivers,
    )


class CASSMTempGather:
    """Full-resolution temp-gather cache for processing server."""

    def __init__(
        self,
        n_sources: int = 16,
        n_receivers: int = 72,
        sample_count: int = 3840,
        sample_rate_hz: float = 48000.0,
    ):
        self.n_sources = n_sources
        self.n_receivers = n_receivers
        self.n_pairs = n_sources * n_receivers
        self.sample_count = sample_count
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz

        # shape: (n_epochs, n_pairs, n_samples)
        self.data = np.zeros((0, self.n_pairs, self.sample_count), dtype=np.float32)
        self.epoch_labels: List[str] = []
        self.epoch_times: List[pd.Timestamp] = []
        # Actual number of source files found per epoch (may differ from n_sources
        # when the array configuration changes over time).
        self.epoch_source_counts: List[int] = []

        self._metric_cache: Dict[str, np.ndarray] = {}
        self._pick_cache: Dict[str, np.ndarray] = {}
        # Flat pair indices to store/load (None = all pairs).  Set when
        # bad-channel / same-well filtering is active so the HDF5 file only
        # holds the ~536 valid pairs instead of all 1152.
        self._valid_pair_indices: Optional[np.ndarray] = None
        # Inverse index for compact HDF5 loading: maps full pair_idx -> compact axis index.
        # None when data is stored in full (n_pairs,) shape.
        self._compact_inv: Optional[np.ndarray] = None

    @property
    def n_epochs(self) -> int:
        return int(self.data.shape[0])

    def append_epoch(
        self,
        epoch_label: str,
        epoch_cube: np.ndarray,
        actual_n_sources: Optional[int] = None,
    ) -> None:
        expected = (self.n_sources, self.n_receivers, self.sample_count)
        if epoch_cube.shape != expected:
            raise ValueError(f"Epoch shape mismatch: expected {expected}, got {epoch_cube.shape}")
        pair_data = epoch_cube.reshape(self.n_pairs, self.sample_count)[np.newaxis, :, :]
        self.data = np.concatenate([self.data, pair_data.astype(np.float32)], axis=0)
        self.epoch_labels.append(epoch_label)
        self.epoch_times.append(_safe_parse_epoch_time(epoch_label))
        self.epoch_source_counts.append(
            actual_n_sources if actual_n_sources is not None else self.n_sources
        )
        self._metric_cache.clear()
        self._pick_cache.clear()

    def append_many(self, items) -> int:
        """Accept 2-tuples (label, cube) or 3-tuples (label, cube, actual_n_sources).

        Pre-allocates the full expanded array in a single numpy allocation to avoid
        O(N²) memory copies from repeated np.concatenate calls.
        """
        items_list = list(items)  # materialise generator if needed
        if not items_list:
            return 0
        n_new = len(items_list)
        n_old = self.n_epochs
        # Single allocation for old + new data.
        new_data = np.empty((n_old + n_new, self.n_pairs, self.sample_count), dtype=np.float32)
        if n_old > 0:
            new_data[:n_old] = self.data
        # Free the old array before filling new slots to keep peak RAM at ~1× array size.
        self.data = new_data
        for i, item in enumerate(items_list):
            label, cube = item[0], item[1]
            actual_n = item[2] if len(item) > 2 else None
            new_data[n_old + i] = cube.reshape(self.n_pairs, self.sample_count).astype(np.float32)
            self.epoch_labels.append(label)
            self.epoch_times.append(_safe_parse_epoch_time(label))
            self.epoch_source_counts.append(
                actual_n if actual_n is not None else self.n_sources
            )
        self._metric_cache.clear()
        self._pick_cache.clear()
        return n_new

    def sort_by_time(self) -> bool:
        """Sort epochs by parsed timestamp to ensure strict temporal order."""
        n = len(self.epoch_times)
        if n <= 1:
            return False
        times = np.array([int(t.value) for t in self.epoch_times], dtype=np.int64)
        order = np.argsort(times)
        if np.all(order == np.arange(n)):
            return False
        self.data = self.data[order, :, :]
        self.epoch_labels = [self.epoch_labels[i] for i in order]
        self.epoch_times = [self.epoch_times[i] for i in order]
        self.epoch_source_counts = [self.epoch_source_counts[i] for i in order]
        self._metric_cache.clear()
        self._pick_cache.clear()
        return True

    # ------------------------------------------------------------------
    # HDF5 persistence (preferred over NPZ for large caches)
    # ------------------------------------------------------------------

    def to_hdf5(self, out_file: Path) -> None:
        """Write the cache to an HDF5 file (atomic, via temp file).

        If ``_valid_pair_indices`` is set only those pair columns are written,
        halving disk usage.  ``from_hdf5`` reconstructs the full array on load.
        """
        import h5py
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_file.with_suffix(out_file.suffix + ".tmp")
        vpi = self._valid_pair_indices
        data_out = self.data[:, vpi, :] if vpi is not None else self.data
        n_stored = data_out.shape[1]
        with h5py.File(tmp, "w") as f:
            f.attrs["n_sources"] = self.n_sources
            f.attrs["n_receivers"] = self.n_receivers
            f.attrs["sample_count"] = self.sample_count
            f.attrs["sample_rate_hz"] = self.sample_rate_hz
            f.create_dataset("data", data=data_out,
                             chunks=(1, n_stored, self.sample_count))
            if vpi is not None:
                f.create_dataset("valid_pair_indices", data=vpi.astype(np.int32))
            f.create_dataset("epoch_labels",
                             data=np.array(self.epoch_labels, dtype="S32"))
            f.create_dataset("epoch_times",
                             data=np.array([t.isoformat() for t in self.epoch_times], dtype="S64"))
            f.create_dataset("epoch_source_counts",
                             data=np.array(self.epoch_source_counts, dtype=np.int32))
        os.replace(tmp, out_file)

    @classmethod
    def from_hdf5(cls, in_file: Path) -> "CASSMTempGather":
        """Load a cache from an HDF5 file into memory.

        If the file was written with ``valid_pair_indices``, the stored compact
        array is expanded back to full ``(n_epochs, n_pairs, n_samples)`` shape
        (zeros for invalid pair slots) so all downstream code is unaffected.
        """
        import h5py
        with h5py.File(in_file, "r") as f:
            tg = cls(
                n_sources=int(f.attrs["n_sources"]),
                n_receivers=int(f.attrs["n_receivers"]),
                sample_count=int(f.attrs["sample_count"]),
                sample_rate_hz=float(f.attrs["sample_rate_hz"]),
            )
            compact = f["data"][:].astype(np.float32)
            if "valid_pair_indices" in f:
                vpi = f["valid_pair_indices"][:].astype(np.int32)
                tg._valid_pair_indices = vpi
                tg.data = np.zeros(
                    (compact.shape[0], tg.n_pairs, tg.sample_count), dtype=np.float32
                )
                tg.data[:, vpi, :] = compact
            else:
                tg.data = compact
            tg.epoch_labels = [s.decode() if isinstance(s, bytes) else str(s)
                               for s in f["epoch_labels"][:].tolist()]
            tg.epoch_times = [pd.to_datetime(
                s.decode() if isinstance(s, bytes) else s, utc=True)
                for s in f["epoch_times"][:].tolist()]
            tg.epoch_source_counts = f["epoch_source_counts"][:].tolist()
        # Guard against partial writes from a previous crash: truncate data to
        # the number of epochs that have complete metadata.
        n_meta = len(tg.epoch_labels)
        if tg.data.shape[0] != n_meta:
            LOG.warning(
                "HDF5 cache %s: data has %d epoch(s) but metadata has %d — "
                "truncating to %d (likely caused by a previous aborted run).",
                in_file, tg.data.shape[0], n_meta, n_meta,
            )
            tg.data = tg.data[:n_meta, :, :]
            tg.epoch_source_counts = tg.epoch_source_counts[:n_meta]
        return tg

    @classmethod
    def from_hdf5_compact(cls, in_file: Path) -> "CASSMTempGather":
        """Load a compact HDF5 cache without expanding to full pair shape.

        Unlike ``from_hdf5``, this keeps ``data`` in the stored compact shape
        ``(n_epochs, n_valid_pairs, n_samples)`` and builds an inverse lookup
        table so that ``get_pair(epoch_idx, full_pair_idx)`` works correctly.
        Useful for the picker GUI where only one pair is accessed at a time,
        avoiding allocation of the full ~46 GB dense array.
        """
        import h5py
        with h5py.File(in_file, "r") as f:
            tg = cls(
                n_sources=int(f.attrs["n_sources"]),
                n_receivers=int(f.attrs["n_receivers"]),
                sample_count=int(f.attrs["sample_count"]),
                sample_rate_hz=float(f.attrs["sample_rate_hz"]),
            )
            tg.data = f["data"][:].astype(np.float32)
            if "valid_pair_indices" in f:
                vpi = f["valid_pair_indices"][:].astype(np.int32)
                tg._valid_pair_indices = vpi
                inv = np.full(tg.n_pairs, -1, dtype=np.int32)
                inv[vpi] = np.arange(len(vpi), dtype=np.int32)
                tg._compact_inv = inv
            tg.epoch_labels = [s.decode() if isinstance(s, bytes) else str(s)
                               for s in f["epoch_labels"][:].tolist()]
            tg.epoch_times = [pd.to_datetime(
                s.decode() if isinstance(s, bytes) else s, utc=True)
                for s in f["epoch_times"][:].tolist()]
            tg.epoch_source_counts = f["epoch_source_counts"][:].tolist()
        return tg
    def get_pair(self, epoch_idx, pair_idx: int) -> np.ndarray:
        """Return waveform data for one full-index pair across given epochs.

        Works with both full-shape and compact-shape ``data`` arrays.
        ``epoch_idx`` may be an int, a list of ints, or a slice.
        """
        if self._compact_inv is not None:
            cidx = int(self._compact_inv[pair_idx])
            if cidx < 0:
                # This pair was not stored (inactive/same-well) — return zeros.
                if isinstance(epoch_idx, (int, np.integer)):
                    return np.zeros(self.sample_count, dtype=np.float32)
                n = (len(epoch_idx) if hasattr(epoch_idx, '__len__')
                     else self.n_epochs)
                return np.zeros((n, self.sample_count), dtype=np.float32)
            return self.data[epoch_idx, cidx, :]
        return self.data[epoch_idx, pair_idx, :]

    def _baseline_picks(
        self,
        config: MetricConfig,
        valid_pairs_mask: Optional[np.ndarray] = None,
        manual_picks: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        if self.n_epochs == 0:
            return np.zeros(self.n_pairs, dtype=int)
        key = (
            f"{config.pick_search_s:.6f}|{config.clip_first_s:.6f}|{config.mute_first_s:.6f}|"
            f"{config.hydro_clip_first_s}|{config.hydro_mute_first_s}|"
            f"{config.taper_fraction:.6f}|"
            f"{config.filter_low_hz:.3f}|{config.filter_high_hz:.3f}|{int(config.filter_order)}|"
            f"{config.accel_filter_low_hz}|{config.accel_filter_high_hz}|"
            f"{config.hydro_filter_low_hz}|{config.hydro_filter_high_hz}|"
            f"{config.picker}|{config.stalta_short_s:.6f}|{config.stalta_long_s:.6f}|"
            f"{config.stalta_threshold:.3f}|{config.baseline_n_epochs}"
        )
        if key in self._pick_cache:
            return self._pick_cache[key]
        n_search = max(int(config.pick_search_s * self.sample_rate_hz), 20)
        n_search = min(n_search, self.sample_count)
        clip_idx = max(int(config.clip_first_s * self.sample_rate_hz), 0)
        clip_idx = min(clip_idx, n_search - 1)
        mute_idx = max(int(config.mute_first_s * self.sample_rate_hz), 0)
        mute_idx = min(mute_idx, n_search - 1)
        start_idx = max(clip_idx, mute_idx)
        # Per-sensor overrides for hydrophone pairs (receivers 48+, 1-based channels 49–72).
        hydro_clip_s = config.hydro_clip_first_s if config.hydro_clip_first_s is not None else config.clip_first_s
        hydro_mute_s = config.hydro_mute_first_s if config.hydro_mute_first_s is not None else config.mute_first_s
        hydro_clip_idx = min(max(int(hydro_clip_s * self.sample_rate_hz), 0), n_search - 1)
        hydro_mute_idx = min(max(int(hydro_mute_s * self.sample_rate_hz), 0), n_search - 1)
        hydro_start_idx = max(hydro_clip_idx, hydro_mute_idx)

        picks = np.zeros(self.n_pairs, dtype=int)
        for p in range(self.n_pairs):
            if valid_pairs_mask is not None and not valid_pairs_mask[p]:
                continue
            is_hydro = (p % self.n_receivers) >= 48
            p_start = hydro_start_idx if is_hydro else start_idx
            n_base = min(config.baseline_n_epochs, self.n_epochs)
            if n_base > 1:
                base_raw = np.mean(self.get_pair(slice(0, n_base), p), axis=0)
            else:
                base_raw = self.get_pair(0, p)
            base_full = _preprocess_waveform(
                base_raw,
                self.sample_rate_hz,
                config,
                pair_index=p,
                n_receivers=self.n_receivers,
            )
            base = base_full[p_start:n_search]
            if base.size <= 1:
                picks[p] = p_start
                continue
            picks[p] = p_start + _apply_picker(base, config, self.sample_rate_hz)
        # Override with any manual picks supplied by the GUI.
        if manual_picks:
            for pair_idx, sample in manual_picks.items():
                pair_idx = int(pair_idx)
                if 0 <= pair_idx < self.n_pairs:
                    picks[pair_idx] = int(sample)
        self._pick_cache[key] = picks
        return picks

    def compute_metrics(
        self,
        config: MetricConfig,
        valid_pairs_mask: Optional[np.ndarray] = None,
        manual_picks: Optional[Dict[int, int]] = None,
        fwi_context: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        key = (
            f"{config.pick_search_s:.6f}|{config.window_s:.6f}|"
            f"{config.clip_first_s:.6f}|{config.mute_first_s:.6f}|"
            f"{config.hydro_clip_first_s}|{config.hydro_mute_first_s}|"
            f"{config.taper_fraction:.6f}|"
            f"{config.filter_low_hz:.3f}|"
            f"{config.filter_high_hz:.3f}|{int(config.filter_order)}|"
            f"{config.accel_filter_low_hz}|{config.accel_filter_high_hz}|"
            f"{config.hydro_filter_low_hz}|{config.hydro_filter_high_hz}|"
            f"{config.picker}|{config.stalta_short_s:.6f}|{config.stalta_long_s:.6f}|"
            f"{config.stalta_threshold:.3f}|{config.baseline_n_epochs}|"
            f"{config.dt_method}|{config.xcorr_max_lag_s:.6f}|{config.xcorr_accept_max_lag_s:.6f}|{config.xcorr_accept_max_lag_hydro_s:.6f}|{config.xcorr_accept_max_lag_dm_hydro_s:.6f}|"
            f"{','.join(config.source_boreholes) if config.source_boreholes else ''}|{config.xcorr_min_peak_cc:.4f}|"
            f"{config.xcorr_edge_guard_samples}|{config.window_taper_fraction:.4f}|"
            f"{config.window_pre_pick_s}|{config.window_post_pick_s}|"
            f"{config.xcorr_despike_single_epoch}|{config.xcorr_despike_mad_thresh:.4f}|"
            f"{config.xcorr_mask_short_runs}|{config.xcorr_short_run_max_len_epochs}|"
            f"{config.xcorr_short_run_min_amp_us:.4f}|{config.xcorr_short_run_neighbor_tol_us:.4f}|"
            f"{config.envelope_guide_xcorr}|{config.envelope_max_lag_s:.6f}|"
            f"{config.envelope_smooth_samples}|{config.envelope_min_peak_cc:.4f}|"
            f"{config.xcorr_fine_half_lag_s:.6f}|{config.envelope_guide_smooth_epochs}|"
            f"{config.dtw_enabled}|{config.dtw_max_shift_ms:.3f}|{config.dtw_strain_limit:.2f}|{config.dtw_min_ncc:.3f}"
        )
        if f"{key}:rms" in self._metric_cache:
            return {
                "rms": self._metric_cache[f"{key}:rms"],
                "centfreq": self._metric_cache[f"{key}:centfreq"],
                "dt_us": self._metric_cache[f"{key}:dt_us"],
                "xcorr_peak_cc": self._metric_cache[f"{key}:xcorr_peak_cc"],
                "xcorr_edge_hit": self._metric_cache[f"{key}:xcorr_edge_hit"],
                "dt_spike_mask": self._metric_cache[f"{key}:dt_spike_mask"],
                "dt_short_run_mask": self._metric_cache[f"{key}:dt_short_run_mask"],
                "pick_index": self._metric_cache[f"{key}:pick_index"],
                "baseline_pick_index": self._metric_cache[f"{key}:baseline_pick_index"],
                "spec_ratio_slope": self._metric_cache[f"{key}:spec_ratio_slope"],
                "envelope_lag_us": self._metric_cache[f"{key}:envelope_lag_us"],
                "envelope_smooth_lag_us": self._metric_cache[f"{key}:envelope_smooth_lag_us"],
                "envelope_peak_cc": self._metric_cache[f"{key}:envelope_peak_cc"],
            }

        if self.n_epochs == 0:
            z = np.zeros((self.n_pairs, 0), dtype=np.float32)
            zu = np.zeros((self.n_pairs, 0), dtype=np.uint8)
            return {
                "rms": z,
                "centfreq": z,
                "dt_us": z,
                "xcorr_peak_cc": z,
                "xcorr_edge_hit": zu,
                "dt_spike_mask": zu,
                "dt_short_run_mask": zu,
                "spec_ratio_slope": z,
                "envelope_lag_us": z,
                "envelope_smooth_lag_us": z,
                "envelope_peak_cc": z,
            }

        # Asymmetric window: small pre-pick buffer + larger post-pick signal region.
        if config.window_pre_pick_s is not None and config.window_post_pick_s is not None:
            pre_samples = max(int(config.window_pre_pick_s * self.sample_rate_hz), 1)
            post_samples = max(int(config.window_post_pick_s * self.sample_rate_hz), 15)
        else:
            half = max(int(config.window_s * self.sample_rate_hz / 2), 8)
            pre_samples = half
            post_samples = half
        win_samples = pre_samples + post_samples
        # Zero-pad to next power of 2 for spectral metrics (centfreq, spec ratio slope).
        # Improves centroid precision and gives more regression points for the slope fit
        # without changing true spectral resolution (set by win_samples).
        n_fft_cf = max(int(2 ** np.ceil(np.log2(max(win_samples, 64)))), 64)
        freqs_cf = np.fft.rfftfreq(n_fft_cf, d=self.dt)  # padded frequency axis for centfreq / spec-ratio
        clip_idx = max(int(config.clip_first_s * self.sample_rate_hz), 0)
        clip_idx = min(clip_idx, self.sample_count - 1)
        mute_idx = max(int(config.mute_first_s * self.sample_rate_hz), 0)
        mute_idx = min(mute_idx, self.sample_count - 1)
        start_idx = max(clip_idx, mute_idx)
        _h_clip_s = config.hydro_clip_first_s if config.hydro_clip_first_s is not None else config.clip_first_s
        _h_mute_s = config.hydro_mute_first_s if config.hydro_mute_first_s is not None else config.mute_first_s
        hydro_start_idx = max(
            min(max(int(_h_clip_s * self.sample_rate_hz), 0), self.sample_count - 1),
            min(max(int(_h_mute_s * self.sample_rate_hz), 0), self.sample_count - 1),
        )
        n_search = max(int(config.pick_search_s * self.sample_rate_hz), 20)
        n_search = min(n_search, self.sample_count)
        max_lag = max(int(config.xcorr_max_lag_s * self.sample_rate_hz), 1)
        accept_max_lag_accel = max(int(config.xcorr_accept_max_lag_s * self.sample_rate_hz), 1)
        accept_max_lag_hydro = max(int(config.xcorr_accept_max_lag_hydro_s * self.sample_rate_hz), 1)
        accept_max_lag_dm_hydro = max(int(config.xcorr_accept_max_lag_dm_hydro_s * self.sample_rate_hz), 1)
        # Envelope-guided xcorr: pre-compute integer sample counts once
        envelope_max_lag = max(int(config.envelope_max_lag_s * self.sample_rate_hz), 1)
        fine_half_lag = max(int(config.xcorr_fine_half_lag_s * self.sample_rate_hz), 1)
        _env_smooth_w = max(int(config.envelope_guide_smooth_epochs), 1)
        # DTW: pre-compute max shift and wide-window extension (for DM*→TS pairs)
        dtw_max_shift_samples = max(int(config.dtw_max_shift_ms * self.sample_rate_hz / 1000.0), 1)
        dtw_win_extension = dtw_max_shift_samples + 5

        rms = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)
        centfreq = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)
        dt_us = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)
        xcorr_peak_cc = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)
        xcorr_edge_hit = np.zeros((self.n_pairs, self.n_epochs), dtype=np.uint8)
        dt_spike_mask = np.zeros((self.n_pairs, self.n_epochs), dtype=np.uint8)
        dt_short_run_mask = np.zeros((self.n_pairs, self.n_epochs), dtype=np.uint8)
        pick_index = np.zeros((self.n_pairs, self.n_epochs), dtype=np.int32)
        spec_ratio_slope = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)
        # Envelope-guidance diagnostics: NaN = envelope mode off or pair skipped;
        # 0.0 = envelope mode on but pair not processed; valid floats when active.
        # envelope_lag_us      : raw per-epoch coarse lag (noisy, for QC)
        # envelope_smooth_lag_us: causal running-median of coarse lag (used to guide fine xcorr)
        envelope_lag_us = np.full((self.n_pairs, self.n_epochs), np.nan, dtype=np.float32)
        envelope_smooth_lag_us = np.full((self.n_pairs, self.n_epochs), np.nan, dtype=np.float32)
        envelope_peak_cc = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)
        # Per-pair circular buffer for causal running-median of coarse lags.
        # Allocated lazily inside the pair loop so only envelope-mode pairs use RAM.
        _env_lag_buf: dict = {}  # pair_index -> np.ndarray of length _env_smooth_w (float32)
        _env_buf_pos: dict = {}  # pair_index -> int (circular insert position)
        _env_buf_cnt: dict = {}  # pair_index -> int (number of valid entries so far)

        # Baseline picks (one per pair) — cached separately; independent of dt_method.
        baseline_pick_index = self._baseline_picks(
            config, valid_pairs_mask=valid_pairs_mask, manual_picks=manual_picks
        )
        src_boreholes = [w.upper().strip() for w in config.source_boreholes] if config.source_boreholes else []
        fwi_estimate_dt = _fwi_estimate_dt

        if config.dt_method == "xcorr":
            # ---------------------------------------------------------------
            # XCORR MODE — matches MATLAB dsiCASSMdelaySequenceEstWinPar:
            #   1. Pick baseline ONCE on the preprocessed mean of first n_base epochs
            #   2. Extract a cosine-tapered window around that pick
            #   3. Per-epoch dt = normalized xcorr lag (sub-sample, parabolic)
            #   4. RMS, centfreq, spectral-ratio-slope all measured in that same window
            # ---------------------------------------------------------------
            # Build per-pair preprocessed baseline windows
            n_base = min(config.baseline_n_epochs, self.n_epochs)
            bl_wins: List[np.ndarray] = []
            bl_tapers: List[np.ndarray] = []
            bl_slices: List[slice] = []
            # Wide baseline windows for envelope coarse-lag stage: extend the post-pick
            # region by envelope_max_lag so that epoch energy shifted by up to
            # ±envelope_max_lag samples is still inside the comparison window.
            bl_wide_wins: List[np.ndarray] = []
            bl_wide_slices: List[slice] = []
            # DTW-specific wide windows: extend further to cover dtw_max_shift + wavelet.
            # Only used when dtw_enabled=True for DM*→TS pairs.
            bl_dtw_wins: List[np.ndarray] = []
            bl_dtw_slices: List[slice] = []
            for p in range(self.n_pairs):
                if valid_pairs_mask is not None and not valid_pairs_mask[p]:
                    bl_wins.append(np.zeros(win_samples))
                    bl_tapers.append(_cosine_window(win_samples, config.window_taper_fraction))
                    bl_slices.append(slice(0, win_samples))
                    bl_wide_wins.append(np.zeros(win_samples + envelope_max_lag))
                    bl_wide_slices.append(slice(0, win_samples + envelope_max_lag))
                    bl_dtw_wins.append(np.zeros(win_samples + dtw_win_extension))
                    bl_dtw_slices.append(slice(0, win_samples + dtw_win_extension))
                    continue
                if n_base > 1:
                    base_raw = np.mean(self.get_pair(slice(0, n_base), p), axis=0)
                else:
                    base_raw = self.get_pair(0, p)
                base_tr = _preprocess_waveform(
                    base_raw, self.sample_rate_hz, config,
                    pair_index=p, n_receivers=self.n_receivers,
                )
                sw = _window_samples(baseline_pick_index[p], pre_samples, post_samples, self.sample_count)
                bl_slices.append(sw)
                w_len = sw.stop - sw.start
                if w_len < 4:
                    bl_wins.append(np.zeros(win_samples))
                    bl_tapers.append(_cosine_window(win_samples, config.window_taper_fraction))
                else:
                    taper = _cosine_window(w_len, config.window_taper_fraction)
                    bl_tapers.append(taper)
                    bl_wins.append(base_tr[sw] * taper)
                # Wide slice: same pre-pick start, post extended by envelope_max_lag.
                sw_wide = _window_samples(
                    baseline_pick_index[p], pre_samples,
                    post_samples + envelope_max_lag, self.sample_count,
                )
                bl_wide_slices.append(sw_wide)
                w_wide = sw_wide.stop - sw_wide.start
                if w_wide < 4:
                    bl_wide_wins.append(np.zeros(win_samples + envelope_max_lag))
                else:
                    taper_wide = _cosine_window(w_wide, config.window_taper_fraction)
                    bl_wide_wins.append(base_tr[sw_wide] * taper_wide)
                # DTW wide slice: extended to cover dtw_max_shift + wavelet buffer.
                sw_dtw = _window_samples(
                    baseline_pick_index[p], pre_samples,
                    post_samples + dtw_win_extension, self.sample_count,
                )
                bl_dtw_slices.append(sw_dtw)
                w_dtw = sw_dtw.stop - sw_dtw.start
                if w_dtw < 4:
                    bl_dtw_wins.append(np.zeros(win_samples + dtw_win_extension))
                else:
                    taper_dtw = _cosine_window(w_dtw, config.window_taper_fraction)
                    bl_dtw_wins.append(base_tr[sw_dtw] * taper_dtw)

            # Per-epoch xcorr loop
            for p in range(self.n_pairs):
                if valid_pairs_mask is not None and not valid_pairs_mask[p]:
                    continue
                bl_win = bl_wins[p]
                sw = bl_slices[p]
                taper = bl_tapers[p]
                bl_dtw_win = bl_dtw_wins[p]
                sw_dtw = bl_dtw_slices[p]
                pick_index[p, :] = baseline_pick_index[p]
                # Bandpass centre frequency metadata (used for centroid/spectral metrics).
                _src_idx = p // self.n_receivers
                _src_bh = src_boreholes[_src_idx] if _src_idx < len(src_boreholes) else ""
                _is_hydro = (p % self.n_receivers) >= 48
                _is_dm_source = _src_bh.startswith("DM")
                # DTW-derived dt: highest priority for large shifts — checked before envelope/xcorr.
                # Restricted to DM*→TS hydrophone pairs (Phase 1 large-dt recovery).
                _use_dtw_this_pair = (
                    config.dtw_enabled
                    and _is_dm_source
                    and _is_hydro
                )
                # FWI-derived dt: highest priority — checked before envelope/xcorr.
                # Restricted to DM*→TS hydrophone pairs (same family as envelope guidance).
                # fwi_context must be pre-built and passed into compute_metrics().
                # NOTE: FWI path disabled in Phase 1; kept for Phase 2 (spatial inversion).
                _use_fwi_dt_this_pair = (
                    config.fwi_dt_enabled
                    and _is_dm_source
                    and _is_hydro
                    and fwi_context is not None
                    and False  # Phase 1: DTW replaces FWI for dt estimation
                )
                # Envelope guidance is restricted to DM*→TS hydrophone pairs only.
                # All other pairs use standard unguided xcorr.
                _use_envelope_this_pair = (
                    config.envelope_guide_xcorr and _is_dm_source and _is_hydro
                    and not _use_dtw_this_pair  # DTW supersedes envelope when both enabled
                    and not _use_fwi_dt_this_pair  # FWI supersedes envelope when both enabled
                )
                if _is_hydro and _is_dm_source:
                    accept_max_lag = accept_max_lag_dm_hydro
                elif _is_hydro:
                    accept_max_lag = accept_max_lag_hydro
                else:
                    accept_max_lag = accept_max_lag_accel
                if _is_hydro:
                    _flo = config.hydro_filter_low_hz if config.hydro_filter_low_hz is not None and config.hydro_filter_low_hz > 0 else config.filter_low_hz
                    _fhi = config.hydro_filter_high_hz if config.hydro_filter_high_hz is not None and config.hydro_filter_high_hz > 0 else config.filter_high_hz
                else:
                    _flo = config.accel_filter_low_hz if config.accel_filter_low_hz is not None and config.accel_filter_low_hz > 0 else config.filter_low_hz
                    _fhi = config.accel_filter_high_hz if config.accel_filter_high_hz is not None and config.accel_filter_high_hz > 0 else config.filter_high_hz
                _flo = _flo if _flo and _flo > 0 else 1.0
                _fhi = _fhi if _fhi and _fhi > 0 else self.sample_rate_hz * 0.45
                if _use_fwi_dt_this_pair and fwi_estimate_dt is None:
                    try:
                        from cussp_cassm_fwi import fwi_estimate_dt as _lazy_fwi_estimate_dt
                        fwi_estimate_dt = _lazy_fwi_estimate_dt
                    except Exception:
                        _use_fwi_dt_this_pair = False
                # Spectral-ratio and centfreq frequency range (same bandpass as xcorr).
                sr_fmin = _flo
                sr_fmax = _fhi
                # Mask to the filter band for centroid frequency (uses zero-padded freq axis).
                _cf_mask = (freqs_cf >= sr_fmin) & (freqs_cf <= sr_fmax)
                freqs_inband = freqs_cf[_cf_mask]
                # Initialise per-pair causal circular buffer for envelope lag smoothing.
                if _use_envelope_this_pair and _env_smooth_w > 1:
                    _env_lag_buf[p] = np.full(_env_smooth_w, np.nan, dtype=np.float64)
                    _env_buf_pos[p] = 0
                    _env_buf_cnt[p] = 0
                for e in range(self.n_epochs):
                    tr = _preprocess_waveform(
                        self.get_pair(e, p), self.sample_rate_hz, config,
                        pair_index=p, n_receivers=self.n_receivers,
                    )
                    ep_seg = tr[sw]
                    w_len = sw.stop - sw.start
                    if w_len < 4 or ep_seg.size < 4:
                        continue
                    ep_win = ep_seg * taper[:ep_seg.size]

                    _dtw_guided_this_epoch = False
                    _envelope_guided_this_epoch = False
                    _fwi_guided_this_epoch = False
                    _dtw_min_ncc_violated = False
                    _dtw_saturated = False
                    if _use_dtw_this_pair:
                        # --- DTW-derived dt (Phase 1: cycle-skip-resistant large-shift recovery) ---
                        # DTW operates on WIDE windows to avoid truncating arrivals with large lags.
                        # Extract wide epoch window (analogous to envelope's ep_wide_win).
                        ep_dtw_seg = tr[sw_dtw]
                        w_dtw_len = sw_dtw.stop - sw_dtw.start
                        if w_dtw_len < 4 or ep_dtw_seg.size < 4:
                            # Wide window too short; skip DTW
                            pass
                        else:
                            taper_dtw_epoch = _cosine_window(ep_dtw_seg.size, config.window_taper_fraction)
                            ep_dtw_win = ep_dtw_seg * taper_dtw_epoch
                            # Run DTW on wide windows (no redundant recompute of max_shift)
                            dtw_lag, dtw_ncc, dtw_rejected = _dtw_dt_samples(
                                baseline_win=bl_dtw_win,
                                epoch_win=ep_dtw_win,
                                max_shift=dtw_max_shift_samples,
                                strain_limit=config.dtw_strain_limit,
                                edge_guard_samples=config.xcorr_edge_guard_samples,
                                signal_end_j=pre_samples + post_samples,
                            )
                            # Fine xcorr centered on DTW lag for sub-sample phase precision.
                            # If it edge-hits (fine_half_lag too tight for this epoch's DTW
                            # estimate), fall back to the integer DTW lag directly rather than
                            # losing the epoch entirely.
                            lag, peak_cc, edge_hit = _xcorr_dt_samples(
                                bl_win, ep_win, fine_half_lag,
                                config.xcorr_edge_guard_samples,
                                center_lag=int(round(dtw_lag)),
                            )
                            if edge_hit:
                                # Fine xcorr misplaced; use DTW lag directly (integer precision).
                                lag = dtw_lag
                                peak_cc = dtw_ncc
                                edge_hit = False
                            _dtw_guided_this_epoch = True
                            _dtw_min_ncc_violated = (dtw_ncc < config.dtw_min_ncc)
                            _dtw_saturated = dtw_rejected  # edge_hit from DTW
                        if not _dtw_guided_this_epoch:
                            # DTW failed or skipped; fall back to standard xcorr
                            lag, peak_cc, edge_hit = _xcorr_dt_samples(
                                bl_win, ep_win, max_lag, config.xcorr_edge_guard_samples
                            )
                    elif _use_dtw_this_pair:
                        # Unreachable: kept for clarity
                        pass
                        _dtw_guided_this_epoch = True
                    elif _use_fwi_dt_this_pair:
                        # --- FWI-derived dt (Phase 1 CDD-TLFWI hybrid) ---
                        _src_global = p // self.n_receivers
                        _rec_global = p % self.n_receivers
                        src_pos, rec_pos = fwi_context.get_pair_grid_pos(_src_global, _rec_global)
                        if src_pos is not None and rec_pos is not None:
                            _wav = fwi_context.get_source_wavelet(_src_global)
                            _dt_us_fwi, _ncc_fwi, _rejected_fwi = fwi_estimate_dt(
                                bl_win=bl_win,
                                ep_win=ep_win,
                                vp=fwi_context.vp,
                                grid=fwi_context.grid,
                                source_wavelet=_wav,
                                src_ix=src_pos[0],
                                src_iz=src_pos[1],
                                rec_ix=rec_pos[0],
                                rec_iz=rec_pos[1],
                                solver=fwi_context.solver,
                                freq_bands=fwi_context.freq_bands,
                                sample_rate_hz=self.sample_rate_hz,
                                dt_search_max_s=fwi_context.dt_search_max_s,
                                min_ncc=fwi_context.min_ncc,
                            )
                            # Convert FWI dt_us to lag in samples for the acceptance gate
                            lag = _dt_us_fwi / (self.dt * 1e6)
                            peak_cc = _ncc_fwi
                            edge_hit = _rejected_fwi
                            _fwi_guided_this_epoch = True
                        else:
                            # Pair not in FWI context (position lookup failed) — fallback xcorr
                            lag, peak_cc, edge_hit = _xcorr_dt_samples(
                                bl_win, ep_win, max_lag, config.xcorr_edge_guard_samples
                            )
                    elif _use_envelope_this_pair:
                        # For the envelope coarse stage use the WIDE window so that
                        # epoch energy shifted by up to ±envelope_max_lag samples is
                        # still captured.  The narrow ep_win is used only for the fine
                        # waveform xcorr (where the search is already centered on the
                        # coarse lag result).
                        sw_wide = bl_wide_slices[p]
                        ep_wide_seg = tr[sw_wide]
                        bl_wide_win = bl_wide_wins[p]
                        # Trim to the shorter of the two wide windows (edge clamping).
                        wide_len = min(len(bl_wide_win), ep_wide_seg.size)
                        if wide_len >= 4:
                            taper_wide = _cosine_window(wide_len, config.window_taper_fraction)
                            ep_wide_win = ep_wide_seg[:wide_len] * taper_wide
                            bl_wide_win_trim = bl_wide_win[:wide_len]
                        else:
                            ep_wide_win = ep_win
                            bl_wide_win_trim = bl_win
                        coarse_lag, env_cc_val = _envelope_coarse_lag(
                            bl_wide_win_trim, ep_wide_win,
                            max_lag=envelope_max_lag,
                            smooth_samples=config.envelope_smooth_samples,
                            edge_guard_samples=config.xcorr_edge_guard_samples,
                        )
                        # Store raw coarse lag for diagnostics.
                        envelope_lag_us[p, e] = float(coarse_lag * self.dt * 1e6)
                        envelope_peak_cc[p, e] = float(env_cc_val)

                        # Update causal running-median buffer with this epoch's raw lag.
                        # PHASE 2 FIX: Always update buffer to prevent guide median from freezing
                        # when individual epochs have decorrelated envelope cc.
                        # The smoothed lag is what guides the fine xcorr search center.
                        if _env_smooth_w > 1:
                            buf = _env_lag_buf[p]
                            pos = _env_buf_pos[p]
                            buf[pos] = coarse_lag  # Always update, regardless of cc threshold
                            _env_buf_pos[p] = (pos + 1) % _env_smooth_w
                            _env_buf_cnt[p] = min(_env_buf_cnt[p] + 1, _env_smooth_w)
                            valid_buf = buf[~np.isnan(buf)]
                            smoothed_lag = float(np.median(valid_buf)) if valid_buf.size else coarse_lag
                        else:
                            smoothed_lag = coarse_lag
                        envelope_smooth_lag_us[p, e] = float(smoothed_lag * self.dt * 1e6)

                        if env_cc_val >= config.envelope_min_peak_cc:
                            # Fine waveform xcorr centered on the SMOOTHED coarse lag.
                            # The ±fine_half_lag window already constrains the search
                            # so the absolute-lag acceptance gate is not applied —
                            # doing so would re-introduce the cycle-skip rejection we
                            # are trying to prevent.
                            lag, peak_cc, edge_hit = _xcorr_dt_samples(
                                bl_win, ep_win, fine_half_lag,
                                config.xcorr_edge_guard_samples,
                                center_lag=int(round(smoothed_lag)),
                            )
                            _envelope_guided_this_epoch = True
                        else:
                            # PHASE 2 FIX: Allow fallback to smoothed_lag instead of hard reject.
                            # When envelope cc dips (decorrelated frames), still use envelope
                            # guidance center but relax fine xcorr gate to prevent freeze while
                            # maintaining cycle-skip protection from envelope-guided centering.
                            # Use smoothed_lag as center hint with slightly relaxed search window.
                            lag, peak_cc, edge_hit = _xcorr_dt_samples(
                                bl_win, ep_win, fine_half_lag + config.xcorr_edge_guard_samples,
                                config.xcorr_edge_guard_samples,
                                center_lag=int(round(smoothed_lag)),
                            )
                            # Mark as envelope-guided (fallback) so acceptance gate is relaxed
                            _envelope_guided_this_epoch = True
                    else:
                        lag, peak_cc, edge_hit = _xcorr_dt_samples(
                            bl_win, ep_win, max_lag, config.xcorr_edge_guard_samples
                        )
                        env_cc_val = 0.0
                    xcorr_peak_cc[p, e] = float(peak_cc)
                    xcorr_edge_hit[p, e] = 1 if edge_hit else 0

                    # Centroid frequency: power-spectrum weighted mean, zero-padded FFT, restricted to filter band.
                    spec_pow = np.abs(np.fft.rfft(ep_win, n=n_fft_cf)) ** 2
                    spec_inband = spec_pow[_cf_mask]
                    denom = float(spec_inband.sum())
                    if denom > 0:
                        centfreq[p, e] = float(np.sum(freqs_inband * spec_inband) / denom / 1000.0)

                    # Acceptance gate.
                    # DTW-guided mode: check DTW quality (min_ncc) and saturation, then fine xcorr.
                    #   If DTW saturates or has poor NCC, reject explicitly (NaN).
                    #   Otherwise skip the abs(lag) distance gate and rely on DTW cycle-unambiguity.
                    # FWI-guided mode: the NCC gate is already applied inside fwi_estimate_dt();
                    #   edge_hit == rejected flag.  Skip the abs(lag) distance-from-zero gate.
                    # Envelope-guided mode: skip the abs(lag) gate for the same reason.
                    # Unguided xcorr mode: keep the original gate.
                    if _dtw_guided_this_epoch:
                        # DTW-specific gate: check for saturation and min_ncc violation
                        if _dtw_saturated or _dtw_min_ncc_violated:
                            accept = False
                        else:
                            # DTW passed; fine xcorr must also pass
                            accept = peak_cc >= config.xcorr_min_peak_cc and not edge_hit
                    elif _fwi_guided_this_epoch or _envelope_guided_this_epoch:
                        accept = peak_cc >= config.xcorr_min_peak_cc and not edge_hit
                    else:
                        accept = (peak_cc >= config.xcorr_min_peak_cc
                                  and not edge_hit
                                  and abs(lag) <= accept_max_lag)

                    if accept:
                        raw_dt = float(lag * self.dt * 1e6)
                        dt_us[p, e] = raw_dt
                        pick_index[p, e] = int(np.round(baseline_pick_index[p] + lag))
                    else:
                        # Use NaN instead of 0 so inversion/plotting can distinguish
                        # rejected epochs (low cc, edge hit, no waveform) from a true
                        # zero travel-time change.
                        dt_us[p, e] = np.nan
                        pick_index[p, e] = int(baseline_pick_index[p])

                    rms[p, e] = float(np.sqrt(np.mean(ep_win ** 2)))

                    # Spectral ratio slope (MATLAB th(85)) — over the same band as the xcorr filter.
                    spec_ratio_slope[p, e] = _spectral_ratio_slope(
                        bl_win, ep_win, self.sample_rate_hz,
                        fmin_hz=sr_fmin, fmax_hz=sr_fmax,
                    )

        else:
            # ---------------------------------------------------------------
            # LEGACY PICK MODE — independent picker per epoch (noisy, kept for
            # backward compatibility and comparison).
            # ---------------------------------------------------------------
            for p in range(self.n_pairs):
                if valid_pairs_mask is not None and not valid_pairs_mask[p]:
                    continue
                p_start = hydro_start_idx if (p % self.n_receivers) >= 48 else start_idx
                for e in range(self.n_epochs):
                    tr = _preprocess_waveform(
                        self.get_pair(e, p), self.sample_rate_hz, config,
                        pair_index=p, n_receivers=self.n_receivers,
                    )
                    seg = tr[p_start:n_search]
                    pick_index[p, e] = (
                        p_start + _apply_picker(seg, config, self.sample_rate_hz)
                        if seg.size > 1 else p_start
                    )

                n_base = min(config.baseline_n_epochs, self.n_epochs)
                baseline_mean = float(np.mean(pick_index[p, :n_base]))
                dt_us[p, :] = (
                    (pick_index[p, :].astype(np.float64) - baseline_mean) * self.dt * 1e6
                )

                for e in range(self.n_epochs):
                    tr = _preprocess_waveform(
                        self.get_pair(e, p), self.sample_rate_hz, config,
                        pair_index=p, n_receivers=self.n_receivers,
                    )
                    sw = _window_samples(baseline_pick_index[p], pre_samples, post_samples, self.sample_count)
                    w = tr[sw]
                    rms[p, e] = float(np.sqrt(np.mean(np.square(w)))) if w.size else 0.0
                    spec = np.abs(np.fft.rfft(w, n=win_samples)) ** 2
                    denom = float(spec.sum())
                    if denom > 0:
                        centfreq[p, e] = float(np.sum(freqs_pow * spec) / denom / 1000.0)

        self._metric_cache[f"{key}:rms"] = rms
        self._metric_cache[f"{key}:centfreq"] = centfreq

        if config.xcorr_despike_single_epoch:
            dt_spike_mask = _despike_single_epoch_dt(
                dt_us,
                rms,
                mad_thresh=config.xcorr_despike_mad_thresh,
            )
            n_spikes = int(np.sum(dt_spike_mask))
            if n_spikes > 0:
                LOG.info("Processing despike masked %d isolated dt spikes", n_spikes)

        if config.xcorr_mask_short_runs:
            dt_short_run_mask = _mask_short_branch_runs(
                dt_us,
                rms,
                max_run_len_epochs=config.xcorr_short_run_max_len_epochs,
                min_amp_us=config.xcorr_short_run_min_amp_us,
                neighbor_tol_us=config.xcorr_short_run_neighbor_tol_us,
            )
            n_short = int(np.sum(dt_short_run_mask))
            if n_short > 0:
                LOG.info("Processing short-run mask removed %d branch-like dt points", n_short)

        self._metric_cache[f"{key}:dt_us"] = dt_us
        self._metric_cache[f"{key}:xcorr_peak_cc"] = xcorr_peak_cc
        self._metric_cache[f"{key}:xcorr_edge_hit"] = xcorr_edge_hit
        self._metric_cache[f"{key}:dt_spike_mask"] = dt_spike_mask
        self._metric_cache[f"{key}:dt_short_run_mask"] = dt_short_run_mask
        self._metric_cache[f"{key}:pick_index"] = pick_index
        self._metric_cache[f"{key}:baseline_pick_index"] = baseline_pick_index
        self._metric_cache[f"{key}:spec_ratio_slope"] = spec_ratio_slope
        self._metric_cache[f"{key}:envelope_lag_us"] = envelope_lag_us
        self._metric_cache[f"{key}:envelope_smooth_lag_us"] = envelope_smooth_lag_us
        self._metric_cache[f"{key}:envelope_peak_cc"] = envelope_peak_cc

        if config.envelope_guide_xcorr:
            n_env_rejected = int(np.sum(
                np.isfinite(envelope_lag_us) & (envelope_peak_cc < config.envelope_min_peak_cc)
            ))
            n_env_accepted = int(np.sum(
                np.isfinite(envelope_lag_us) & (envelope_peak_cc >= config.envelope_min_peak_cc)
            ))
            LOG.info(
                "Envelope-guided xcorr: %d epoch-pair(s) accepted, %d rejected (env_cc < %.2f)",
                n_env_accepted, n_env_rejected, config.envelope_min_peak_cc,
            )
        return {
            "rms": rms,
            "centfreq": centfreq,
            "dt_us": dt_us,
            "xcorr_peak_cc": xcorr_peak_cc,
            "xcorr_edge_hit": xcorr_edge_hit,
            "dt_spike_mask": dt_spike_mask,
            "dt_short_run_mask": dt_short_run_mask,
            "pick_index": pick_index,
            "baseline_pick_index": baseline_pick_index,
            "spec_ratio_slope": spec_ratio_slope,
            "envelope_lag_us": envelope_lag_us,
            "envelope_smooth_lag_us": envelope_smooth_lag_us,
            "envelope_peak_cc": envelope_peak_cc,
        }


def _load_cache_for_processing(cache_file: Path) -> CASSMTempGather:
    """Load the cache in the most memory-efficient form available.

    The live processing cache is written compactly when invalid/same-well pairs
    are masked out. Loading that file through ``from_hdf5_compact`` avoids
    expanding the pair axis back to the full dense geometry, which is the main
    memory spike that the FWI branch pushed over the edge.
    """
    import h5py

    with h5py.File(cache_file, "r") as f:
        has_valid_pairs = "valid_pair_indices" in f
    if has_valid_pairs:
        return CASSMTempGather.from_hdf5_compact(cache_file)
    return CASSMTempGather.from_hdf5(cache_file)


def _default_receiver_boreholes(n_receivers: int) -> List[str]:
    """Return the well name (borehole prefix) for each receiver index 0..n_receivers-1.

    CUSSP channel map (1-based channel -> receiver index = channel - 1):
      ch  1-12  (rec  0-11 ): AML
      ch 13-24  (rec 12-23 ): AMU
      ch 25-36  (rec 24-35 ): DML
      ch 37-48  (rec 36-47 ): DMU
      ch 49-72  (rec 48-71 ): TS   (hydrophones)
    """
    wells = []
    for ri in range(n_receivers):
        ch = ri + 1  # 1-based channel
        if ch <= 12:
            wells.append("AML")
        elif ch <= 24:
            wells.append("AMU")
        elif ch <= 36:
            wells.append("DML")
        elif ch <= 48:
            wells.append("DMU")
        else:
            wells.append("TS")
    return wells


def _build_same_well_mask(
    n_sources: int,
    n_receivers: int,
    source_boreholes: List[str],
    receiver_boreholes: List[str],
) -> np.ndarray:
    """Return a boolean array of shape (n_sources * n_receivers,) that is True
    for every pair where source and receiver share the same borehole well.

    *source_boreholes*   length n_sources, one well name per source (0-based order).
    *receiver_boreholes* length n_receivers, one well name per receiver channel.

    Well matching is case-insensitive.  E.g. source well 'AML' matches receiver
    borehole 'AML1' because the receiver name *starts with* the source well name
    and the trailing suffix is purely a sensor number.
    """
    mask = np.zeros(n_sources * n_receivers, dtype=bool)
    src_wells = [w.upper().strip() for w in source_boreholes]
    rec_wells  = [w.upper().strip() for w in receiver_boreholes]
    for si, sw in enumerate(src_wells):
        for ri, rw in enumerate(rec_wells):
            # Match if receiver borehole starts with the source well name,
            # e.g. src 'AML' matches rec 'AML1', 'AML2', etc.
            if rw == sw or rw.startswith(sw):
                mask[si * n_receivers + ri] = True
    return mask


def list_epoch_dirs(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def load_epoch_npz(epoch_dir: Path) -> Optional[np.ndarray]:
    for name in ("epoch_data.npz", "dsi_epoch.npz"):
        f = epoch_dir / name
        if f.exists():
            obj = np.load(f, allow_pickle=True)
            if "data" in obj:
                return obj["data"].astype(np.float32)
    return None


def load_epoch_seg2(
    epoch_dir: Path,
    n_sources: int,
    n_receivers: int,
    sample_count: int,
    bad_channels: Optional[frozenset] = None,
) -> Optional[np.ndarray]:
    """Load one epoch from SEG2 .dat files.

    *bad_channels* is a frozenset of 1-based Geode channel numbers (e.g. {72}
    for TS01) that should be zeroed at ingest.  System channels (73-76) are
    excluded automatically because their channel numbers exceed n_receivers.
    """
    dat_files = sorted(epoch_dir.glob("*.dat"))
    seg2_files = sorted(epoch_dir.glob("*.seg2"))
    seg_files = sorted(dat_files + seg2_files)
    if not seg_files:
        return None

    # Prefer explicit source mapping from SIGMAV logs (channel, dat-file, well, depth),
    # e.g. "00,  67317.dat,  B5_1,  20.0". This prevents source-index shifts when
    # the array configuration changes and a subset of source files is present.
    sigmav_map: Dict[str, int] = {}
    sigmav_logs = sorted(epoch_dir.glob("*_SIGMAV.log"))
    if not sigmav_logs:
        sigmav_logs = sorted(epoch_dir.glob("*SIGMAV*.log"))

    # Guardrail: if source mapping log is missing and the epoch is undersampled,
    # skip the epoch to avoid ambiguous source-index assignment.
    if not sigmav_logs and len(dat_files) < 16:
        LOG.warning(
            "Skipping epoch %s: no SIGMAV log and only %d .dat files (<16).",
            epoch_dir.name,
            len(dat_files),
        )
        return None
    _sigmav_line = re.compile(
        r"\b(\d{1,3})\s*,\s*([^,\s]+\.(?:dat|seg2))\s*,\s*([^,]+?)\s*,\s*([-+]?\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    )
    for logf in sigmav_logs:
        try:
            txt = logf.read_text(errors="ignore")
        except Exception:
            continue
        for line in txt.splitlines():
            m = _sigmav_line.search(line)
            if not m:
                continue
            chan = int(m.group(1))
            fname = Path(m.group(2)).name.lower()
            if chan < 0:
                continue
            sigmav_map[fname] = chan

    n_files = len(seg_files)
    if n_files != n_sources:
        LOG.warning(
            "Epoch %s: found %d source file(s), expected %d. "
            "Loading %d source(s); remaining slots will be zero. "
            "If this is a configuration change, update --n-sources and VALID_DAT_COUNTS.",
            epoch_dir.name, n_files, n_sources, min(n_files, n_sources),
        )

    try:
        from obspy import read as obspy_read
    except Exception as exc:
        raise RuntimeError("ObsPy is required for SEG2 ingestion.") from exc

    _bad = bad_channels or frozenset()
    cube = np.zeros((n_sources, n_receivers, sample_count), dtype=np.float32)
    used_src_idx: set = set()
    for default_src_idx, seg_file in enumerate(seg_files[:n_sources]):
        src_idx = default_src_idx
        mapped_src_idx = sigmav_map.get(seg_file.name.lower())
        if mapped_src_idx is not None:
            if 0 <= mapped_src_idx < n_sources:
                src_idx = mapped_src_idx
            else:
                LOG.warning(
                    "Epoch %s: SIGMAV mapped %s to channel %d outside [0,%d); using positional index %d.",
                    epoch_dir.name, seg_file.name, mapped_src_idx, n_sources, default_src_idx,
                )

        if src_idx in used_src_idx:
            LOG.warning(
                "Epoch %s: duplicate source index %d for file %s; skipping this file.",
                epoch_dir.name, src_idx, seg_file.name,
            )
            continue
        used_src_idx.add(src_idx)

        try:
            st = obspy_read(str(seg_file))
        except Exception as exc:
            LOG.warning("Failed reading %s: %s", seg_file, exc)
            continue

        for tr_idx, tr in enumerate(st):
            rec_idx = None
            try:
                ch = int(tr.stats.seg2.get("CHANNEL_NUMBER", tr_idx + 1))
                if 1 <= ch <= n_receivers:
                    rec_idx = ch - 1
            except Exception:
                ch = tr_idx + 1
                rec_idx = tr_idx if tr_idx < n_receivers else None

            if rec_idx is None or rec_idx >= n_receivers:
                continue

            # Zero known-bad channels — leave the slot as zeros in the cube.
            if ch in _bad:
                LOG.debug(
                    "Epoch %s: channel %d (rec_idx %d) zeroed at ingest.",
                    epoch_dir.name, ch, rec_idx,
                )
                continue  # leave cube[src_idx, rec_idx, :] = 0

            data = np.asarray(tr.data, dtype=np.float32)
            if data.size >= sample_count:
                cube[src_idx, rec_idx, :] = data[:sample_count]
            else:
                cube[src_idx, rec_idx, : data.size] = data

    return cube


def load_epoch_cube(
    epoch_dir: Path,
    n_sources: int,
    n_receivers: int,
    sample_count: int,
    bad_channels: Optional[frozenset] = None,
) -> Optional[np.ndarray]:
    cube = load_epoch_npz(epoch_dir)
    if cube is not None:
        return cube
    return load_epoch_seg2(epoch_dir, n_sources, n_receivers, sample_count, bad_channels=bad_channels)


def scan_new_epochs(
    tg: CASSMTempGather,
    data_dir: Path,
    bad_channels: Optional[frozenset] = None,
) -> Generator[Tuple[str, np.ndarray, int], None, None]:
    """Yield 3-tuples of (label, cube, actual_n_sources) for epochs not yet in *tg*.

    This is a generator so each epoch cube is freed after the caller processes it,
    keeping peak RAM to ~1 epoch worth of data at a time during streaming ingest.
    Emits WARNING for any epoch whose source-file count differs from tg.n_sources.
    """
    known = set(tg.epoch_labels)
    all_dirs = list_epoch_dirs(data_dir)
    new_dirs = [ep for ep in all_dirs if ep.name not in known]
    n_new = len(new_dirs)
    LOG.info("scan_new_epochs: %d total dirs, %d already cached, %d to load",
             len(all_dirs), len(all_dirs) - n_new, n_new)
    if n_new == 0:
        return
    t0 = time.monotonic()
    _LOG_INTERVAL = 50
    warned_counts: set = set()
    n_loaded = 0
    for idx, ep in enumerate(new_dirs):
        actual_n = len(
            list(ep.glob("*.dat")) + list(ep.glob("*.seg2"))
        )
        cube = load_epoch_cube(ep, tg.n_sources, tg.n_receivers, tg.sample_count, bad_channels=bad_channels)
        if cube is None:
            continue
        expected = (tg.n_sources, tg.n_receivers, tg.sample_count)
        if cube.shape != expected:
            LOG.warning("Skipping %s: expected shape %s, got %s", ep.name, expected, cube.shape)
            continue
        if actual_n != tg.n_sources and actual_n not in warned_counts:
            LOG.warning(
                "Array configuration mismatch: epoch %s has %d source files, expected %d.",
                ep.name, actual_n, tg.n_sources,
            )
            warned_counts.add(actual_n)
        n_loaded += 1
        yield (ep.name, cube, actual_n)
        done = idx + 1
        if done % _LOG_INTERVAL == 0 or done == n_new:
            elapsed = time.monotonic() - t0
            rate = n_loaded / elapsed if elapsed > 0 else 0
            eta_s = (n_new - done) / rate if rate > 0 else float("inf")
            LOG.info("Scanned %d/%d epochs from disk (%.1f ep/s, ETA %.0fs)",
                     n_loaded, n_new, rate, eta_s)


def make_preview(data: np.ndarray, target_samples: int) -> np.ndarray:
    """Downsample waveforms for lightweight remote visualization.

    Input shape: (n_epochs, n_pairs, n_samples)
    Output shape: (n_epochs, n_pairs, n_preview_samples)
    """
    if data.size == 0:
        return data
    n_samples = data.shape[2]
    if target_samples >= n_samples:
        return data
    idx = np.linspace(0, n_samples - 1, int(target_samples)).astype(int)
    return data[:, :, idx]


def _build_observed_baseline_from_gather(
    tg: CASSMTempGather,
    baseline_n_epochs: int,
) -> np.ndarray:
    """Build the baseline stack while respecting compact pair storage.

    The FWI branch only needs the averaged baseline waveforms, not the full
    epoch cube.  When the cache is compact we rebuild only the baseline matrix
    in dense pair form, which is much smaller than expanding the entire cache.
    """
    n_base = min(int(baseline_n_epochs), tg.n_epochs)
    baseline = np.zeros((tg.n_pairs, tg.sample_count), dtype=np.float32)
    if n_base < 1:
        return baseline

    # Rebuild pair-by-pair so the returned array is always dense full-shape,
    # regardless of whether the cache was stored compactly or not.
    for pair_idx in range(tg.n_pairs):
        pair_stack = np.asarray(tg.get_pair(slice(0, n_base), pair_idx), dtype=np.float32)
        baseline[pair_idx, :] = np.mean(pair_stack, axis=0, dtype=np.float64).astype(np.float32)
    return baseline


def collect_inversion_outputs(
    inversion_dir: Path,
    url_prefix: Optional[str] = None,
    max_items: int = 20,
) -> List[Dict[str, object]]:
    """Collect latest inversion outputs for manifest publication.

    Supported file types: images and common analysis products.
    """
    inversion_dir = Path(inversion_dir)
    if not inversion_dir.exists():
        return []

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.svg", "*.json", "*.csv", "*.npz")
    files: List[Path] = []
    for pat in patterns:
        files.extend(inversion_dir.glob(pat))

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[: max(max_items, 1)]
    outputs: List[Dict[str, object]] = []
    for f in files:
        mtime = pd.Timestamp.utcfromtimestamp(f.stat().st_mtime).isoformat()
        item = {
            "name": f.name,
            "path": str(f),
            "size_bytes": int(f.stat().st_size),
            "updated_utc": mtime,
        }
        if url_prefix:
            base = url_prefix.rstrip("/")
            item["url"] = f"{base}/{f.name}"
        outputs.append(item)
    return outputs


def collect_processing_outputs(
    qc_dir: Path,
    url_prefix: Optional[str] = None,
    max_items: int = 20,
) -> List[Dict[str, object]]:
    qc_dir = Path(qc_dir)
    if not qc_dir.exists():
        return []

    files = sorted(qc_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[: max(max_items, 1)]
    out: List[Dict[str, object]] = []
    for f in files:
        if not f.is_file():
            continue
        item = {
            "name": f.name,
            "path": str(f),
            "size_bytes": int(f.stat().st_size),
            "updated_utc": pd.Timestamp.fromtimestamp(f.stat().st_mtime, tz="UTC").isoformat(),
        }
        if url_prefix:
            item["url"] = f"{url_prefix.rstrip('/')}/{f.name}"
        out.append(item)
    return out


def _build_receiver_labels(n_receivers: int) -> List[str]:
    """Human-readable receiver names for the fixed CUSSP/CASSM Geode layout.

    ch  1-48 : 16 three-component accelerometers across 4 boreholes
               (AML ch1-12, AMU ch13-24, DML ch25-36, DMU ch37-48)
               sensor number within borehole = ((ch-1)%12)//3 + 1
               component = Z/X/Y for (ch-1)%3 = 0/1/2
    ch 49-72 : hydrophones TS24 (ch49, deepest) … TS01 (ch72)
    """
    _comp = ("Z", "X", "Y")
    _accel_bh = ("AML", "AMU", "DML", "DMU")
    labels: List[str] = []
    for ch in range(1, n_receivers + 1):
        if ch <= 48:
            bh_idx = (ch - 1) // 12
            sensor_num = ((ch - 1) % 12) // 3 + 1
            comp = _comp[(ch - 1) % 3]
            bh = _accel_bh[bh_idx] if bh_idx < len(_accel_bh) else f"BH{bh_idx}"
            labels.append(f"{bh}{sensor_num}/{comp}")
        else:
            ts_num = 73 - ch  # ch49→TS24, ch72→TS01
            labels.append(f"TS{ts_num:02d}")
    return labels


def _build_source_labels(
    n_sources: int, source_boreholes: Optional[List[str]] = None
) -> List[str]:
    """Human-readable source names.  Each source within a borehole is numbered
    sequentially in source-index order (AML1, AML2, …).  Falls back to
    "Src0", "Src1", … when *source_boreholes* is not supplied.
    """
    if not source_boreholes or len(source_boreholes) < n_sources:
        return [f"Src{i}" for i in range(n_sources)]
    bh_counts: dict = {}
    labels: List[str] = []
    for bh in source_boreholes[:n_sources]:
        bh_counts[bh] = bh_counts.get(bh, 0) + 1
        labels.append(f"{bh}S{bh_counts[bh]}")
    return labels


def _build_pair_labels(
    n_sources: int,
    n_receivers: int,
    src_labels: Optional[List[str]] = None,
    rec_labels: Optional[List[str]] = None,
) -> List[str]:
    """Return n_sources × n_receivers pair labels 'SrcName→RecName'."""
    sl = src_labels or [f"S{i}" for i in range(n_sources)]
    rl = rec_labels or [f"R{j}" for j in range(n_receivers)]
    return [f"{sl[si]}→{rl[ri]}" for si in range(n_sources) for ri in range(n_receivers)]


def write_processing_qc(
    qc_dir: Path,
    tg: CASSMTempGather,
    metrics: Dict[str, np.ndarray],
    config: MetricConfig,
    max_pairs: int = 60,
    baseline_plot_samples: int = 60,
    pair_labels: Optional[List[str]] = None,
    valid_pairs_mask: Optional[np.ndarray] = None,
) -> None:
    """Write compact QC figures to inspect picking and metric behavior."""
    qc_dir.mkdir(parents=True, exist_ok=True)

    if tg.n_epochs == 0:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        LOG.warning("matplotlib unavailable; skipping processing QC plots")
        return

    # Build human-readable labels from pair_labels or fall back to indices.
    _valid_labels = pair_labels and len(pair_labels) == tg.n_pairs
    if _valid_labels:
        _src_labels = [pair_labels[si * tg.n_receivers].split("→")[0]
                       for si in range(tg.n_sources)]
        _rec_labels = [pair_labels[ri].split("→")[1]
                       for ri in range(tg.n_receivers)]
    else:
        _src_labels = [str(i) for i in range(tg.n_sources)]
        _rec_labels = [str(j) for j in range(tg.n_receivers)]

    def _plabel(p: int) -> str:
        return pair_labels[p] if _valid_labels else f"pair {p}"

    dt_us = metrics["dt_us"]
    rms = metrics["rms"]
    centfreq = metrics["centfreq"]
    pick_index = metrics.get("pick_index")
    baseline_pick_index = metrics.get("baseline_pick_index")

    # Receiver-family masks for CUSSP geometry:
    #   rec 0..47  -> 16 accelerometers x 3 components
    #   rec 48..71 -> 24 hydrophones (if present in this config)
    # For reduced-receiver configs (e.g., 48), hydrophone mask is empty.
    rec_idx_all = np.arange(tg.n_pairs, dtype=np.int32) % max(tg.n_receivers, 1)
    accel_mask = rec_idx_all < min(48, tg.n_receivers)
    hydro_mask = rec_idx_all >= 48
    # Restrict statistics to the valid (active, non-same-well) pairs only.
    if valid_pairs_mask is not None:
        accel_mask = accel_mask & valid_pairs_mask
        hydro_mask = hydro_mask & valid_pairs_mask
    n_accel_pairs = int(np.sum(accel_mask))
    n_hydro_pairs = int(np.sum(hydro_mask))

    top_n = int(min(max(max_pairs, 1), tg.n_pairs))
    order = np.argsort(np.nanmedian(rms, axis=1))[::-1]
    top_pairs = order[:top_n]

    def _top_pairs_for_mask(mask: np.ndarray, n: int) -> np.ndarray:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return np.array([], dtype=np.int32)
        local_order = idx[np.argsort(np.nanmedian(rms[idx, :], axis=1))[::-1]]
        return local_order[: int(min(max(n, 1), idx.size))]

    top_pairs_accel = _top_pairs_for_mask(accel_mask, max_pairs)
    top_pairs_hydro = _top_pairs_for_mask(hydro_mask, max_pairs)

    # Figure 1: delay-time heatmap for strongest channels.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=140)
    im = ax.imshow(dt_us[top_pairs, :], aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title(f"Delay-Time Heatmap (Top {top_n} Pairs by Median RMS)")
    ax.set_xlabel("Epoch Index (time-sorted)")
    ax.set_ylabel("Pair Rank")
    fig.colorbar(im, ax=ax, label="dt_us")
    fig.tight_layout()
    fig.savefig(qc_dir / "qc_dt_heatmap.png")
    plt.close(fig)

    # Figure 1b: delay-time heatmap split by receiver family.
    n_panels = 2 if top_pairs_hydro.size else 1
    fig, axs = plt.subplots(1, n_panels, figsize=(12 if n_panels == 2 else 7, 5), dpi=140)
    axs = np.atleast_1d(axs)
    if top_pairs_accel.size:
        im0 = axs[0].imshow(dt_us[top_pairs_accel, :], aspect="auto", cmap="RdBu_r", interpolation="nearest")
        axs[0].set_title(f"Accelerometers (top {top_pairs_accel.size})")
        axs[0].set_xlabel("Epoch Index")
        axs[0].set_ylabel("Pair Rank")
        fig.colorbar(im0, ax=axs[0], label="dt_us")
        if _valid_labels and top_pairs_accel.size <= 30:
            axs[0].set_yticks(np.arange(top_pairs_accel.size))
            axs[0].set_yticklabels([pair_labels[p] for p in top_pairs_accel], fontsize=6)
    else:
        axs[0].set_title("Accelerometers: none")
        axs[0].axis("off")

    if n_panels == 2:
        if top_pairs_hydro.size:
            im1 = axs[1].imshow(dt_us[top_pairs_hydro, :], aspect="auto", cmap="RdBu_r", interpolation="nearest")
            axs[1].set_title(f"Hydrophones (top {top_pairs_hydro.size})")
            axs[1].set_xlabel("Epoch Index")
            axs[1].set_ylabel("Pair Rank")
            fig.colorbar(im1, ax=axs[1], label="dt_us")
            if _valid_labels and top_pairs_hydro.size <= 30:
                axs[1].set_yticks(np.arange(top_pairs_hydro.size))
                axs[1].set_yticklabels([pair_labels[p] for p in top_pairs_hydro], fontsize=6)
        else:
            axs[1].set_title("Hydrophones: none")
            axs[1].axis("off")

    fig.suptitle("Delay-Time Heatmaps by Receiver Family")
    fig.tight_layout()
    fig.savefig(qc_dir / "qc_dt_heatmap_by_sensor.png")
    plt.close(fig)

    # Figure 2: global metric trends over epochs.
    # Restrict medians to valid (active, non-same-well) pairs so that skipped
    # zero-valued pairs don’t push the median to zero.
    _valid_idx = (
        np.where(valid_pairs_mask)[0]
        if valid_pairs_mask is not None
        else np.arange(tg.n_pairs)
    )
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), dpi=140, sharex=True)
    axs[0].plot(np.nanmedian(dt_us[_valid_idx, :], axis=0), color="k", lw=1.0)
    axs[0].set_ylabel("Median dt_us")
    axs[0].grid(True, alpha=0.3)
    axs[1].plot(np.nanmedian(rms[_valid_idx, :], axis=0), color="tab:blue", lw=1.0)
    axs[1].set_ylabel("Median RMS")
    axs[1].grid(True, alpha=0.3)
    axs[2].plot(np.nanmedian(centfreq[_valid_idx, :], axis=0), color="tab:green", lw=1.0)
    axs[2].set_ylabel("Median Cf (kHz)")
    axs[2].set_xlabel("Epoch Index (time-sorted)")
    axs[2].grid(True, alpha=0.3)
    fig.suptitle("Processing Metric Trends")
    fig.tight_layout()
    fig.savefig(qc_dir / "qc_metric_trends.png")
    plt.close(fig)

    # Figure 2b: metric trends split by receiver family.
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), dpi=140, sharex=True)
    # dt_us
    if n_accel_pairs > 0:
        axs[0].plot(np.nanmedian(dt_us[accel_mask, :], axis=0), color="tab:blue", lw=1.0, label="accel")
    if n_hydro_pairs > 0:
        axs[0].plot(np.nanmedian(dt_us[hydro_mask, :], axis=0), color="tab:orange", lw=1.0, label="hydro")
    axs[0].set_ylabel("Median dt_us")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=8)
    # rms
    if n_accel_pairs > 0:
        axs[1].plot(np.nanmedian(rms[accel_mask, :], axis=0), color="tab:blue", lw=1.0, label="accel")
    if n_hydro_pairs > 0:
        axs[1].plot(np.nanmedian(rms[hydro_mask, :], axis=0), color="tab:orange", lw=1.0, label="hydro")
    axs[1].set_ylabel("Median RMS")
    axs[1].grid(True, alpha=0.3)
    # centfreq
    if n_accel_pairs > 0:
        axs[2].plot(np.nanmedian(centfreq[accel_mask, :], axis=0), color="tab:blue", lw=1.0, label="accel")
    if n_hydro_pairs > 0:
        axs[2].plot(np.nanmedian(centfreq[hydro_mask, :], axis=0), color="tab:orange", lw=1.0, label="hydro")
    axs[2].set_ylabel("Median Cf (kHz)")
    axs[2].set_xlabel("Epoch Index (time-sorted)")
    axs[2].grid(True, alpha=0.3)
    fig.suptitle("Processing Metric Trends by Receiver Family")
    fig.tight_layout()
    fig.savefig(qc_dir / "qc_metric_trends_by_sensor.png")
    plt.close(fig)

    # Figure 3: waveform inspection for one representative pair.
    pair = int(top_pairs[0]) if top_pairs.size else 0
    t_ms = np.arange(tg.sample_count, dtype=float) * tg.dt * 1000.0
    traces = [
        (0, "first"),
        (max(tg.n_epochs // 2, 0), "mid"),
        (max(tg.n_epochs - 1, 0), "last"),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=140)
    for e, lbl in traces:
        tr = tg.get_pair(e, pair)
        tr = _preprocess_waveform(
            tr,
            tg.sample_rate_hz,
            config,
            pair_index=pair,
            n_receivers=tg.n_receivers,
        )
        tr = tr / (np.max(np.abs(tr)) + 1e-9)
        ax.plot(t_ms, tr, lw=1.0, label=f"{lbl} (epoch {e})")

    clip_end_ms = float(config.clip_first_s) * 1000.0
    ax.axvspan(0.0, clip_end_ms, color="tab:purple", alpha=0.10, label="clipped")
    mute_end_ms = float(config.mute_first_s) * 1000.0
    ax.axvspan(0.0, mute_end_ms, color="tab:red", alpha=0.12, label="muted for picks")

    if baseline_pick_index is not None and baseline_pick_index.size > pair:
        b_ms = float(baseline_pick_index[pair]) * tg.dt * 1000.0
        ax.axvline(b_ms, color="k", ls="--", lw=1.0, label="baseline pick")
    if pick_index is not None and pick_index.shape[1] > 0:
        p_last_ms = float(pick_index[pair, -1]) * tg.dt * 1000.0
        ax.axvline(p_last_ms, color="tab:orange", ls=":", lw=1.2, label="last-epoch pick")

    ax.set_xlim(0.0, min(25.0, t_ms[-1]))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(f"{_plabel(pair)} — Pick Region / Crosstalk Zone")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(qc_dir / "qc_waveform_pick_view.png")
    plt.close(fig)

    # Figure 4: preprocessed baseline-pick gallery (pick is on preprocessed trace,
    # so show the preprocessed trace alongside the pick marker).
    sample_n = int(min(max(baseline_plot_samples, 1), tg.n_pairs))
    sample_pairs = top_pairs[:sample_n]
    n_cols = 5
    n_rows = int(np.ceil(sample_n / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 1.8 * n_rows), dpi=140)
    axs_arr = np.atleast_1d(axs).ravel()
    for i, p in enumerate(sample_pairs):
        ax = axs_arr[i]
        tr = _preprocess_waveform(
            tg.get_pair(0, int(p)),
            tg.sample_rate_hz,
            config,
            pair_index=int(p),
            n_receivers=tg.n_receivers,
        )
        tr = tr / (np.max(np.abs(tr)) + 1e-9)
        ax.plot(t_ms, tr, lw=0.8, color="k")
        ax.axvspan(0.0, clip_end_ms, color="tab:purple", alpha=0.10)
        ax.axvspan(0.0, mute_end_ms, color="tab:red", alpha=0.12)
        if (baseline_pick_index is not None
                and baseline_pick_index.size > int(p)
                and (valid_pairs_mask is None or valid_pairs_mask[int(p)])):
            b_ms = float(baseline_pick_index[int(p)]) * tg.dt * 1000.0
            ax.axvline(b_ms, color="tab:orange", ls="-", lw=0.9)
        ax.set_xlim(0.0, min(25.0, t_ms[-1]))
        ax.set_title(_plabel(int(p)), fontsize=6)
        ax.grid(True, alpha=0.2)
        if i % n_cols == 0:
            ax.set_ylabel("norm amp", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)

    for j in range(sample_n, len(axs_arr)):
        axs_arr[j].axis("off")

    fig.suptitle(f"Preprocessed Baseline Pick Gallery (n={sample_n}, epoch 0) — clip+taper+filter applied")
    fig.tight_layout()
    fig.savefig(qc_dir / "qc_baseline_raw_pick_gallery.png")
    plt.close(fig)

    # Figure 5: source × receiver matrix of baseline pick times.
    # Immediately exposes systematic source-level or receiver-level anomalies.
    if baseline_pick_index is not None and baseline_pick_index.size == tg.n_pairs:
        pick_ms_flat = baseline_pick_index.astype(float) * tg.dt * 1000.0
        # Mask invalid (inactive / same-well) pairs so they show as blank in the plot
        # rather than as spurious 0 ms picks.
        if valid_pairs_mask is not None:
            pick_ms_flat[~valid_pairs_mask] = np.nan
        pick_ms_2d = pick_ms_flat.reshape(tg.n_sources, tg.n_receivers)
        fig, ax = plt.subplots(1, 1, figsize=(max(tg.n_receivers * 0.22, 6), max(tg.n_sources * 0.4, 4)), dpi=140)
        im = ax.imshow(pick_ms_2d, aspect="auto", cmap="plasma", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Baseline pick time (ms)")
        ax.set_title("Baseline Pick Time Matrix (epoch 0, preprocessed)")
        ax.set_xlabel("Receiver index")
        ax.set_ylabel("Source index")
        ax.set_yticks(np.arange(tg.n_sources))
        ax.set_yticklabels(_src_labels, fontsize=7)
        step = max(tg.n_receivers // 12, 1)
        xtick_idx = np.arange(0, tg.n_receivers, step)
        ax.set_xticks(xtick_idx)
        ax.set_xticklabels([_rec_labels[j] for j in xtick_idx], rotation=45, ha="right", fontsize=6)
        fig.tight_layout()
        fig.savefig(qc_dir / "qc_baseline_pick_matrix.png")
        plt.close(fig)

    # Figure 6: per-source gather view — one subplot per source, all receiver
    # waveforms overlaid or offset, with the baseline pick marked.
    # Reveals moveout pattern and flags whole-source failures.
    n_src = tg.n_sources
    n_src_cols = min(4, n_src)
    n_src_rows = int(np.ceil(n_src / n_src_cols))
    fig, axs_src = plt.subplots(
        n_src_rows, n_src_cols,
        figsize=(4.5 * n_src_cols, 2.8 * n_src_rows),
        dpi=130,
    )
    axs_src_arr = np.atleast_1d(axs_src).ravel()
    xlim_gather = min(25.0, t_ms[-1])
    for src_idx in range(n_src):
        ax = axs_src_arr[src_idx]
        pair_start = src_idx * tg.n_receivers
        pair_end = pair_start + tg.n_receivers
        for rec_idx in range(tg.n_receivers):
            p_idx = pair_start + rec_idx
            tr = _preprocess_waveform(
                tg.get_pair(0, p_idx),
                tg.sample_rate_hz,
                config,
                pair_index=p_idx,
                n_receivers=tg.n_receivers,
            )
            amax = np.max(np.abs(tr)) + 1e-9
            tr_norm = tr / amax
            # offset traces by receiver index for a gather-style display
            ax.plot(t_ms, tr_norm + rec_idx, lw=0.5, color="k", alpha=0.7)
            if (baseline_pick_index is not None
                    and baseline_pick_index.size > p_idx
                    and (valid_pairs_mask is None or valid_pairs_mask[p_idx])):
                b_ms = float(baseline_pick_index[p_idx]) * tg.dt * 1000.0
                ax.plot(b_ms, float(rec_idx), "r.", ms=3, alpha=0.85)
        ax.axvspan(0.0, clip_end_ms, color="tab:purple", alpha=0.07)
        ax.axvspan(0.0, mute_end_ms, color="tab:red", alpha=0.09)
        ax.set_xlim(0.0, xlim_gather)
        ax.set_title(_src_labels[src_idx], fontsize=8)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)
        if src_idx % n_src_cols == 0:
            ax.set_ylabel("→ rec", fontsize=7)
        if src_idx >= (n_src_rows - 1) * n_src_cols:
            ax.set_xlabel("ms", fontsize=7)

    for j in range(n_src, len(axs_src_arr)):
        axs_src_arr[j].axis("off")

    fig.suptitle("Per-Source Gather — Preprocessed Baseline (epoch 0)\nRed dots = picks, purple/red spans = clip/mute zones")
    fig.tight_layout()
    fig.savefig(qc_dir / "qc_baseline_per_source_gather.png")
    plt.close(fig)

    # Figure 7: baseline pick-time histogram — exposes picks clustering in the
    # crosstalk zone or suspiciously tight / wide distributions.
    if baseline_pick_index is not None and baseline_pick_index.size > 0:
        pick_ms_all = baseline_pick_index.astype(float) * tg.dt * 1000.0
        # Restrict the overall histogram to valid (active, non-same-well) pairs only.
        _hist_mask = valid_pairs_mask if valid_pairs_mask is not None else np.ones(tg.n_pairs, dtype=bool)
        pick_ms_valid = pick_ms_all[_hist_mask]
        n_valid_pairs = int(_hist_mask.sum())
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), dpi=140)
        ax.hist(pick_ms_valid, bins=min(80, n_valid_pairs // 5 + 10), color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvspan(0.0, clip_end_ms, color="tab:purple", alpha=0.20, label=f"clip zone (<{clip_end_ms:.2f} ms)")
        ax.axvspan(0.0, mute_end_ms, color="tab:red", alpha=0.15, label=f"mute zone (<{mute_end_ms:.2f} ms)")
        n_in_mute = int(np.sum(pick_ms_valid <= mute_end_ms))
        ax.set_xlabel("Baseline pick time (ms)")
        ax.set_ylabel("Number of pairs")
        ax.set_title(
            f"Baseline Pick Distribution  |  n_valid_pairs={n_valid_pairs}  |  "
            f"picks ≤ mute zone: {n_in_mute} ({100*n_in_mute/max(n_valid_pairs,1):.1f}%)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(qc_dir / "qc_baseline_pick_histogram.png")
        plt.close(fig)

        # Figure 7b: baseline pick-time histogram by receiver family.
        n_panels = 2 if n_hydro_pairs > 0 else 1
        fig, axs = plt.subplots(1, n_panels, figsize=(11 if n_panels == 2 else 6, 3.5), dpi=140, sharey=True)
        axs = np.atleast_1d(axs)

        if n_accel_pairs > 0:
            pick_ms_accel = pick_ms_all[accel_mask]
            axs[0].hist(pick_ms_accel, bins=min(60, max(12, pick_ms_accel.size // 8)), color="tab:blue", alpha=0.8)
            axs[0].axvspan(0.0, clip_end_ms, color="tab:purple", alpha=0.20)
            axs[0].axvspan(0.0, mute_end_ms, color="tab:red", alpha=0.15)
            axs[0].set_title(f"Accelerometers (n={pick_ms_accel.size})")
            axs[0].set_xlabel("Baseline pick time (ms)")
            axs[0].set_ylabel("Number of pairs")
            axs[0].grid(True, alpha=0.3)
        else:
            axs[0].set_title("Accelerometers: none")
            axs[0].axis("off")

        if n_panels == 2:
            pick_ms_hydro = pick_ms_all[hydro_mask]
            axs[1].hist(pick_ms_hydro, bins=min(60, max(12, pick_ms_hydro.size // 8)), color="tab:orange", alpha=0.8)
            axs[1].axvspan(0.0, clip_end_ms, color="tab:purple", alpha=0.20)
            axs[1].axvspan(0.0, mute_end_ms, color="tab:red", alpha=0.15)
            axs[1].set_title(f"Hydrophones (n={pick_ms_hydro.size})")
            axs[1].set_xlabel("Baseline pick time (ms)")
            axs[1].grid(True, alpha=0.3)

        fig.suptitle("Baseline Pick Distribution by Receiver Family")
        fig.tight_layout()
        fig.savefig(qc_dir / "qc_baseline_pick_histogram_by_sensor.png")
        plt.close(fig)

    # Figure 8: spectral ratio slope trends — MATLAB th(85) equivalent.
    spec_ratio_slope = metrics.get("spec_ratio_slope")
    if spec_ratio_slope is not None and spec_ratio_slope.size > 0 and np.any(spec_ratio_slope != 0):
        n_panels = 2 if top_pairs_hydro.size else 1
        fig, axs = plt.subplots(1, n_panels, figsize=(12 if n_panels == 2 else 7, 4), dpi=140)
        axs = np.atleast_1d(axs)

        if top_pairs_accel.size:
            med_a = np.nanmedian(spec_ratio_slope[top_pairs_accel, :], axis=0)
            axs[0].plot(med_a, color="tab:blue", lw=1.0)
            axs[0].axhline(0, color="k", lw=0.6, ls="--")
            axs[0].set_title(f"Accelerometers (median of top {top_pairs_accel.size})")
            axs[0].set_xlabel("Epoch Index")
            axs[0].set_ylabel("Spectral ratio slope (nepers/Hz)")
            axs[0].grid(True, alpha=0.3)
        else:
            axs[0].axis("off")

        if n_panels == 2 and top_pairs_hydro.size:
            med_h = np.nanmedian(spec_ratio_slope[top_pairs_hydro, :], axis=0)
            axs[1].plot(med_h, color="tab:orange", lw=1.0)
            axs[1].axhline(0, color="k", lw=0.6, ls="--")
            axs[1].set_title(f"Hydrophones (median of top {top_pairs_hydro.size})")
            axs[1].set_xlabel("Epoch Index")
            axs[1].set_ylabel("Spectral ratio slope (nepers/Hz)")
            axs[1].grid(True, alpha=0.3)

        fig.suptitle(
            "Spectral Ratio Slope  log(|FFT(baseline)| / |FFT(epoch)|) vs. f  "
            "— MATLAB th(85)\n"
            "Negative slope = epoch has relatively less high-f content (increased attenuation)"
        )
        fig.tight_layout()
        fig.savefig(qc_dir / "qc_spectral_ratio_slope.png")
        plt.close(fig)

    summary = {
        "n_epochs": int(tg.n_epochs),
        "n_pairs": int(tg.n_pairs),
        "n_accelerometer_pairs": n_accel_pairs,
        "n_hydrophone_pairs": n_hydro_pairs,
        "time_sorted": True,
        "pick_search_s": float(config.pick_search_s),
        "window_s": float(config.window_s),
        "picker": config.picker,
        "dt_method": config.dt_method,
        "xcorr_max_lag_s": float(config.xcorr_max_lag_s),
        "xcorr_accept_max_lag_s": float(config.xcorr_accept_max_lag_s),
        "xcorr_accept_max_lag_hydro_s": float(config.xcorr_accept_max_lag_hydro_s),
        "xcorr_accept_max_lag_dm_hydro_s": float(config.xcorr_accept_max_lag_dm_hydro_s),
        "xcorr_min_peak_cc": float(config.xcorr_min_peak_cc),
        "xcorr_edge_guard_samples": int(config.xcorr_edge_guard_samples),
        "xcorr_despike_single_epoch": bool(config.xcorr_despike_single_epoch),
        "xcorr_despike_mad_thresh": float(config.xcorr_despike_mad_thresh),
        "xcorr_mask_short_runs": bool(config.xcorr_mask_short_runs),
        "xcorr_short_run_max_len_epochs": int(config.xcorr_short_run_max_len_epochs),
        "xcorr_short_run_min_amp_us": float(config.xcorr_short_run_min_amp_us),
        "xcorr_short_run_neighbor_tol_us": float(config.xcorr_short_run_neighbor_tol_us),
        "clip_first_s": float(config.clip_first_s),
        "mute_first_s": float(config.mute_first_s),
        "taper_fraction": float(config.taper_fraction),
        "filter_low_hz": float(config.filter_low_hz),
        "filter_high_hz": float(config.filter_high_hz),
        "filter_order": int(config.filter_order),
        "accel_filter_low_hz": (
            None if config.accel_filter_low_hz is None else float(config.accel_filter_low_hz)
        ),
        "accel_filter_high_hz": (
            None if config.accel_filter_high_hz is None else float(config.accel_filter_high_hz)
        ),
        "hydro_filter_low_hz": (
            None if config.hydro_filter_low_hz is None else float(config.hydro_filter_low_hz)
        ),
        "hydro_filter_high_hz": (
            None if config.hydro_filter_high_hz is None else float(config.hydro_filter_high_hz)
        ),
        "representative_pair": int(pair),
        "baseline_plot_samples": sample_n,
    }
    (qc_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2))


def publish_bundle(
    bundle_path: Path,
    manifest_path: Path,
    tg: CASSMTempGather,
    metrics: Dict[str, np.ndarray],
    preview_samples: int,
    inversion_outputs: Optional[List[Dict[str, object]]] = None,
    processing_outputs: Optional[List[Dict[str, object]]] = None,
) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    preview = make_preview(tg.data, target_samples=preview_samples)
    preview_dt_ms = tg.dt * (tg.sample_count - 1) / max(preview.shape[2] - 1, 1) * 1000.0 if tg.n_epochs else 0.0

    tmp_bundle = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    with tmp_bundle.open("wb") as f:
        np.savez_compressed(
            f,
            epoch_labels=np.array(tg.epoch_labels, dtype=object),
            epoch_times=np.array([t.isoformat() for t in tg.epoch_times], dtype=object),
            rms=metrics["rms"],
            centfreq=metrics["centfreq"],
            dt_us=metrics["dt_us"],
            xcorr_peak_cc=metrics.get("xcorr_peak_cc"),
            xcorr_edge_hit=metrics.get("xcorr_edge_hit"),
            dt_spike_mask=metrics.get("dt_spike_mask"),
            dt_short_run_mask=metrics.get("dt_short_run_mask"),
            envelope_lag_us=metrics.get("envelope_lag_us"),
            envelope_smooth_lag_us=metrics.get("envelope_smooth_lag_us"),
            envelope_peak_cc=metrics.get("envelope_peak_cc"),
            gather_preview=preview,
            valid_pair_indices=tg._valid_pair_indices,
            preview_dt_ms=preview_dt_ms,
            n_sources=tg.n_sources,
            n_receivers=tg.n_receivers,
            sample_count=tg.sample_count,
            sample_rate_hz=tg.sample_rate_hz,
            n_epochs=tg.n_epochs,
        )
    os.replace(tmp_bundle, bundle_path)

    # Build source-count histogram and a list of detected configuration transitions.
    # A transition is any epoch where actual_n_sources differs from the previous epoch's.
    source_count_histogram: Dict[str, int] = {}
    config_transitions: list = []
    prev_n: Optional[int] = None
    for label, n in zip(tg.epoch_labels, tg.epoch_source_counts):
        key = str(n)
        source_count_histogram[key] = source_count_histogram.get(key, 0) + 1
        if prev_n is not None and n != prev_n:
            config_transitions.append(
                {"epoch": label, "prev_n_sources": prev_n, "new_n_sources": n}
            )
        prev_n = n

    if config_transitions:
        LOG.warning(
            "Array configuration changed %d time(s) across the archive: %s",
            len(config_transitions),
            "; ".join(
                f"{t['epoch']}: {t['prev_n_sources']}->{t['new_n_sources']}"
                for t in config_transitions
            ),
        )

    now = pd.Timestamp.now(tz="UTC").isoformat()
    manifest = {
        "updated_utc": now,
        "n_epochs": tg.n_epochs,
        "n_pairs": tg.n_pairs,
        "n_sources": tg.n_sources,
        "n_receivers": tg.n_receivers,
        "sample_count": tg.sample_count,
        "sample_rate_hz": tg.sample_rate_hz,
        "bundle_path": str(bundle_path),
        "source_count_histogram": source_count_histogram,
        "config_transitions": config_transitions,
        "processing_outputs": processing_outputs or [],
        "latest_processing": (processing_outputs or [None])[0],
        "inversion_outputs": inversion_outputs or [],
        "latest_inversion": (inversion_outputs or [None])[0],
    }
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp_manifest, manifest_path)


def _stream_ingest_hdf5(
    tg: CASSMTempGather,
    hdf5_path: Path,
    epochs_gen: Generator,
    n_new: int,
    valid_pair_indices: Optional[np.ndarray] = None,
) -> int:
    """Stream epoch cubes one at a time directly into an HDF5 file.

    If *valid_pair_indices* is provided only those pair columns are written to
    disk, cutting storage roughly in half (~22 GB instead of ~47 GB for the
    full CUSSP geometry).  Invalid pair slots are zero-filled on the final
    in-memory load so all downstream code is unaffected.

    Peak RAM: ~1 epoch cube (~18 MB) during streaming, then one full array load
    at the end.  Returns the number of epochs successfully appended.
    """
    import h5py

    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if hdf5_path.exists() else "w"
    n_start = tg.n_epochs

    # Prefer the pair layout already stored in an existing compact cache.
    # If the file was written as a compact HDF5 cache, appends must preserve
    # that exact stored pair index set.
    vpi = valid_pair_indices  # default for new files
    if mode == "a":
        with h5py.File(hdf5_path, "r") as existing:
            if "valid_pair_indices" in existing:
                stored_vpi = existing["valid_pair_indices"][:].astype(np.int32)
                if vpi is not None and not np.array_equal(stored_vpi, vpi):
                    LOG.warning(
                        "Existing compact cache layout differs from the current pair mask; using the layout already stored in %s.",
                        hdf5_path,
                    )
                vpi = stored_vpi

    n_stored = len(vpi) if vpi is not None else tg.n_pairs

    t0 = time.monotonic()
    n_appended = 0

    with h5py.File(hdf5_path, mode) as f:
        if "data" not in f:
            f.attrs["n_sources"] = tg.n_sources
            f.attrs["n_receivers"] = tg.n_receivers
            f.attrs["sample_count"] = tg.sample_count
            f.attrs["sample_rate_hz"] = tg.sample_rate_hz
            if vpi is not None:
                f.create_dataset("valid_pair_indices", data=vpi.astype(np.int32))
            f.create_dataset(
                "data",
                shape=(n_start, n_stored, tg.sample_count),
                maxshape=(None, n_stored, tg.sample_count),
                dtype="float32",
                chunks=(1, n_stored, tg.sample_count),
            )
            f.create_dataset("epoch_labels",
                             data=np.array(tg.epoch_labels, dtype="S32"),
                             maxshape=(None,), chunks=(256,))
            f.create_dataset("epoch_times",
                             data=np.array([t.isoformat() for t in tg.epoch_times], dtype="S64"),
                             maxshape=(None,), chunks=(256,))
            f.create_dataset("epoch_source_counts",
                             data=np.array(tg.epoch_source_counts, dtype=np.int32),
                             maxshape=(None,), chunks=(256,))
            if n_start > 0:
                src = tg.data[:, vpi, :] if vpi is not None else tg.data
                f["data"][:] = src

        ds = f["data"]
        ds_labels = f["epoch_labels"]
        ds_times = f["epoch_times"]
        ds_counts = f["epoch_source_counts"]

        for label, cube, actual_n in epochs_gen:
            new_size = ds.shape[0] + 1
            ds.resize(new_size, axis=0)
            flat = cube.reshape(tg.n_pairs, tg.sample_count).astype(np.float32)
            ds[new_size - 1] = flat[vpi, :] if vpi is not None else flat

            ts = _safe_parse_epoch_time(label)
            tg.epoch_labels.append(label)
            tg.epoch_times.append(ts)
            tg.epoch_source_counts.append(actual_n)

            ds_labels.resize(new_size, axis=0)
            ds_labels[new_size - 1] = label.encode("ascii")
            ds_times.resize(new_size, axis=0)
            ds_times[new_size - 1] = ts.isoformat().encode("ascii")
            ds_counts.resize(new_size, axis=0)
            ds_counts[new_size - 1] = actual_n

            n_appended += 1
            if n_appended % 50 == 0 or n_appended == n_new:
                elapsed = time.monotonic() - t0
                rate = n_appended / elapsed if elapsed > 0 else 0
                eta_s = (n_new - n_appended) / rate if rate > 0 else float("inf")
                LOG.info("Streamed %d/%d epochs → HDF5 (%.1f ep/s, ETA %.0fs)",
                         n_appended, n_new, rate, eta_s)

    # Load data from HDF5 into memory for downstream computation.
    # If compact storage was used, expand back to full (n_epochs, n_pairs, n_samples).
    LOG.info("Loading %d epochs from HDF5 into memory...", n_start + n_appended)
    with h5py.File(hdf5_path, "r") as f:
        compact = f["data"][:].astype(np.float32)
    if vpi is not None:
        tg._valid_pair_indices = vpi
        tg.data = np.zeros(
            (compact.shape[0], tg.n_pairs, tg.sample_count), dtype=np.float32
        )
        tg.data[:, vpi, :] = compact
        del compact
    else:
        tg.data = compact
    tg._metric_cache.clear()
    tg._pick_cache.clear()
    return n_appended


def run_once(args) -> int:
    cache_file = Path(args.cache_file)
    data_dir = Path(args.data_dir)
    bundle_file = Path(args.bundle_file)
    manifest_file = Path(args.manifest_file)

    if cache_file.suffix not in (".h5", ".hdf5"):
        raise ValueError(f"--cache-file must be an HDF5 file (.h5 or .hdf5), got: {cache_file}")

    if cache_file.exists():
        tg = _load_cache_for_processing(cache_file)
        if tg.sort_by_time():
            LOG.info("Sorted cached epochs by timestamp")
        LOG.info("Loaded cache: %s (epochs=%d)", cache_file, tg.n_epochs)
    else:
        tg = CASSMTempGather(
            n_sources=args.n_sources,
            n_receivers=args.n_receivers,
            sample_count=args.sample_count,
            sample_rate_hz=args.sample_rate_hz,
        )
        LOG.info("Initialized new temp-gather cache")

    bad_channels = frozenset(
        int(x.strip()) for x in args.known_bad_receiver_channels.split(",") if x.strip()
    ) if args.known_bad_receiver_channels else frozenset()

    # If an explicit active-channel list was given, zero every channel NOT on it.
    if args.active_receiver_channels:
        active = frozenset(
            int(x.strip()) for x in args.active_receiver_channels.split(",") if x.strip()
        )
        all_channels = frozenset(range(1, tg.n_receivers + 1))
        inactive = all_channels - active
        if inactive:
            LOG.info(
                "--active-receiver-channels: zeroing %d inactive channels at ingest.",
                len(inactive),
            )
        bad_channels = bad_channels | inactive

    if bad_channels:
        LOG.info("Channels zeroed at ingest: %s", sorted(bad_channels))

    # Build same-well pair mask from source/receiver borehole assignments.
    same_well_mask: Optional[np.ndarray] = None
    if args.source_boreholes:
        src_wells = [w.strip() for w in args.source_boreholes.split(",") if w.strip()]
        # Derive receiver well names from the channel_map in the YAML (or fall back to
        # a simple rule: strip trailing digits from the borehole label).
        # Here we use the hardcoded CUSSP receiver borehole order derived from the YAML.
        rec_boreholes = _default_receiver_boreholes(tg.n_receivers)
        if len(src_wells) != tg.n_sources:
            LOG.warning(
                "--source-boreholes has %d entries but n_sources=%d; skipping same-well masking.",
                len(src_wells), tg.n_sources,
            )
        else:
            same_well_mask = _build_same_well_mask(
                tg.n_sources, tg.n_receivers, src_wells, rec_boreholes
            )
            n_sw = int(np.sum(same_well_mask))
            LOG.info(
                "Same-well pair mask: %d/%d pairs will be zeroed (sources in same borehole as receiver).",
                n_sw, tg.n_pairs,
            )

    # Build the set of flat pair indices to keep for HDF5 compact storage.
    # Pairs whose receiver is inactive OR that are same-well are excluded so
    # they are never written to disk (not even as zeros).
    valid_pair_indices: Optional[np.ndarray] = None
    if bad_channels or same_well_mask is not None:
        invalid_mask = np.zeros(tg.n_pairs, dtype=bool)
        for rec_ch in bad_channels:
            rec_0 = rec_ch - 1  # convert 1-based channel to 0-based receiver index
            if 0 <= rec_0 < tg.n_receivers:
                invalid_mask[rec_0::tg.n_receivers] = True  # all sources × this receiver
        if same_well_mask is not None:
            invalid_mask |= same_well_mask
        valid_pair_indices = np.where(~invalid_mask)[0].astype(np.int32)
        LOG.info(
            "Compact HDF5: storing %d/%d valid pairs (%.1f%% of full array).",
            len(valid_pair_indices), tg.n_pairs,
            100.0 * len(valid_pair_indices) / tg.n_pairs,
        )

    epochs_gen = scan_new_epochs(tg, data_dir, bad_channels=bad_channels)
    # Count new epochs first (just directory listing — no data loaded yet).
    known = set(tg.epoch_labels)
    n_new = sum(1 for ep in list_epoch_dirs(data_dir) if ep.name not in known)
    if n_new > 0:
        n = _stream_ingest_hdf5(tg, cache_file, epochs_gen, n_new,
                                valid_pair_indices=valid_pair_indices)
        if tg.sort_by_time():
            LOG.info("Sorted epochs by timestamp after append")
        LOG.info("Streamed and appended %d new epoch(s) to HDF5 cache", n)
    else:
        LOG.info("No new epochs found")

    # Resolve baseline epoch count from baseline_end_date only.
    if not args.baseline_end_date:
        raise ValueError(
            "Missing required picking.baseline_end_date in config. "
            "baseline_n_epochs has been removed."
        )
    try:
        cutoff = pd.Timestamp(args.baseline_end_date).tz_localize("UTC") \
            if pd.Timestamp(args.baseline_end_date).tzinfo is None \
            else pd.Timestamp(args.baseline_end_date).tz_convert("UTC")

        def _to_utc(t: pd.Timestamp) -> pd.Timestamp:
            return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

        baseline_n_epochs = sum(1 for t in tg.epoch_times if _to_utc(t) <= cutoff)
    except Exception as exc:
        raise ValueError(
            f"Invalid picking.baseline_end_date '{args.baseline_end_date}': {exc}"
        ) from exc

    if baseline_n_epochs < 1:
        raise ValueError(
            "picking.baseline_end_date selects zero baseline epochs; "
            "choose a later baseline_end_date."
        )

    LOG.info(
        "baseline_end_date %s -> %d baseline epoch(s) out of %d total",
        args.baseline_end_date, baseline_n_epochs, tg.n_epochs,
    )

    metric_config = MetricConfig(
        pick_search_s=args.pick_search_s,
        window_s=args.window_s,
        clip_first_s=args.clip_first_s,
        mute_first_s=args.mute_first_s,
        taper_fraction=args.taper_fraction,
        filter_low_hz=args.filter_low_hz,
        filter_high_hz=args.filter_high_hz,
        filter_order=args.filter_order,
        accel_filter_low_hz=args.accel_filter_low_hz,
        accel_filter_high_hz=args.accel_filter_high_hz,
        hydro_filter_low_hz=args.hydro_filter_low_hz,
        hydro_filter_high_hz=args.hydro_filter_high_hz,
        hydro_clip_first_s=args.hydro_clip_first_s,
        hydro_mute_first_s=args.hydro_mute_first_s,
        picker=args.picker,
        stalta_short_s=args.stalta_short_s,
        stalta_long_s=args.stalta_long_s,
        stalta_threshold=args.stalta_threshold,
        baseline_n_epochs=baseline_n_epochs,
        aic_margin_samples=args.aic_margin_samples,
        aic_min_snr=args.aic_min_snr,
        dt_method=args.dt_method,
        xcorr_max_lag_s=args.xcorr_max_lag_ms / 1000.0,
        xcorr_accept_max_lag_s=args.xcorr_accept_max_lag_ms / 1000.0,
        xcorr_accept_max_lag_hydro_s=args.xcorr_accept_max_lag_hydro_ms / 1000.0,
        xcorr_accept_max_lag_dm_hydro_s=args.xcorr_accept_max_lag_dm_hydro_ms / 1000.0,
        xcorr_min_peak_cc=args.xcorr_min_peak_cc,
        xcorr_edge_guard_samples=args.xcorr_edge_guard_samples,
        xcorr_despike_single_epoch=args.xcorr_despike_single_epoch,
        xcorr_despike_mad_thresh=args.xcorr_despike_mad_thresh,
        xcorr_mask_short_runs=args.xcorr_mask_short_runs,
        xcorr_short_run_max_len_epochs=args.xcorr_short_run_max_len_epochs,
        xcorr_short_run_min_amp_us=args.xcorr_short_run_min_amp_us,
        xcorr_short_run_neighbor_tol_us=args.xcorr_short_run_neighbor_tol_us,
        source_boreholes=tuple([w.strip() for w in args.source_boreholes.split(",") if w.strip()]) if args.source_boreholes else None,
        window_pre_pick_s=args.window_pre_pick_ms / 1000.0 if args.window_pre_pick_ms is not None else None,
        window_post_pick_s=args.window_post_pick_ms / 1000.0 if args.window_post_pick_ms is not None else None,
        envelope_guide_xcorr=args.envelope_guide_xcorr,
        envelope_max_lag_s=args.envelope_max_lag_s,
        envelope_smooth_samples=args.envelope_smooth_samples,
        envelope_min_peak_cc=args.envelope_min_peak_cc,
        xcorr_fine_half_lag_s=args.xcorr_fine_half_lag_s,
        envelope_guide_smooth_epochs=args.envelope_guide_smooth_epochs,
        dtw_enabled=args.dtw_enabled,
        dtw_max_shift_ms=args.dtw_max_shift_ms,
        dtw_strain_limit=args.dtw_strain_limit,
        dtw_min_ncc=args.dtw_min_ncc,
        fwi_dt_enabled=args.fwi_dt_enabled,
        fwi_dt_sources_csv=args.fwi_dt_sources_csv,
        fwi_dt_receivers_csv=args.fwi_dt_receivers_csv,
        fwi_dt_solver=args.fwi_dt_solver,
        fwi_dt_grid_dx_m=args.fwi_dt_grid_dx_m,
        fwi_dt_grid_dz_m=args.fwi_dt_grid_dz_m,
        fwi_dt_grid_padding_m=args.fwi_dt_grid_padding_m,
        fwi_dt_vp_background_mps=args.fwi_dt_vp_background_mps,
        fwi_dt_freq_bands=args.fwi_dt_freq_bands,
        fwi_dt_search_max_ms=args.fwi_dt_search_max_ms,
        fwi_dt_min_ncc=args.fwi_dt_min_ncc,
        fwi_dt_gate_pre_ms=args.fwi_dt_gate_pre_ms,
        fwi_dt_gate_post_ms=args.fwi_dt_gate_post_ms,
        fwi_dt_cpml_thickness=args.fwi_dt_cpml_thickness,
    )

    # Build valid-pairs mask: receiver is active (not in bad_channels / inactive) AND
    # pair is not a same-well pair.  This prevents inactive / same-well pairs from
    # wasting compute time and polluting QC statistics.
    # Pair p = src * n_receivers + rec_idx; receiver channel (1-based) = rec_idx + 1.
    valid_pairs_mask = np.ones(tg.n_pairs, dtype=bool)
    if bad_channels:
        for p in range(tg.n_pairs):
            rec_ch = (p % tg.n_receivers) + 1  # 1-based channel
            if rec_ch in bad_channels:
                valid_pairs_mask[p] = False
    if same_well_mask is not None:
        valid_pairs_mask &= ~same_well_mask
    n_valid = int(valid_pairs_mask.sum())
    LOG.info(
        "Valid pairs for metric computation: %d / %d (skipping %d inactive/same-well pairs)",
        n_valid, tg.n_pairs, tg.n_pairs - n_valid,
    )

    # Load manual picks from the GUI picker if provided.
    manual_picks: Optional[Dict[int, int]] = None
    if args.manual_picks_file:
        mp_path = Path(args.manual_picks_file)
        if mp_path.exists():
            try:
                raw = json.loads(mp_path.read_text())
                manual_picks = {int(k): int(v) for k, v in raw.items()}
                LOG.info("Loaded %d manual picks from %s", len(manual_picks), mp_path)
            except Exception as exc:
                LOG.warning("Could not load manual picks file %s: %s", mp_path, exc)
        else:
            LOG.warning("--manual-picks-file path does not exist: %s", mp_path)

    # Restrict to manually-picked pairs only if requested.
    if args.require_manual_picks:
        if manual_picks:
            picked_set = set(manual_picks.keys())
            before = int(valid_pairs_mask.sum())
            for p in range(tg.n_pairs):
                if valid_pairs_mask[p] and p not in picked_set:
                    valid_pairs_mask[p] = False
            after = int(valid_pairs_mask.sum())
            LOG.info(
                "--require-manual-picks: restricted from %d to %d pairs (%d without a pick excluded)",
                before, after, before - after,
            )
        else:
            LOG.warning("--require-manual-picks set but no manual picks loaded; ignoring.")

    # Build FWI context once (before pair loop) when fwi_dt is enabled.
    fwi_context = None
    if metric_config.fwi_dt_enabled:
        try:
            from cussp_cassm_fwi import build_fwi_context as _build_fwi_ctx
            from cussp_cassm_fwi import build_observed_baseline as _build_obs_bl
            LOG.info("Building FWI context for DM*→TS pairs...")
            if not str(metric_config.fwi_dt_sources_csv).strip():
                raise ValueError(
                    "fwi_dt.sources_csv is empty; set it to a CSV file path or disable fwi_dt.enabled."
                )
            if not str(metric_config.fwi_dt_receivers_csv).strip():
                raise ValueError(
                    "fwi_dt.receivers_csv is empty; set it to a CSV file path or disable fwi_dt.enabled."
                )
            _d_obs_bl = _build_observed_baseline_from_gather(tg, metric_config.baseline_n_epochs)
            _bl_picks = tg._baseline_picks(
                metric_config,
                valid_pairs_mask=valid_pairs_mask,
                manual_picks=manual_picks,
            )
            _src_bh_list = list(metric_config.source_boreholes) if metric_config.source_boreholes else []
            _fwi_freq_bands = metric_config.fwi_dt_freq_bands or [
                (250.0, 2000.0), (500.0, 8000.0), (1000.0, 20000.0),
            ]
            fwi_context = _build_fwi_ctx(
                tg_n_sources=tg.n_sources,
                tg_n_receivers=tg.n_receivers,
                tg_sample_rate_hz=tg.sample_rate_hz,
                tg_sample_count=tg.sample_count,
                d_obs_baseline=_d_obs_bl,
                baseline_picks=_bl_picks,
                source_boreholes=_src_bh_list,
                sources_csv=Path(metric_config.fwi_dt_sources_csv),
                receivers_csv=Path(metric_config.fwi_dt_receivers_csv),
                solver_name=metric_config.fwi_dt_solver,
                grid_dx_m=metric_config.fwi_dt_grid_dx_m,
                grid_dz_m=metric_config.fwi_dt_grid_dz_m,
                grid_padding_m=metric_config.fwi_dt_grid_padding_m,
                vp_background_mps=metric_config.fwi_dt_vp_background_mps,
                freq_bands=_fwi_freq_bands,
                dt_search_max_ms=metric_config.fwi_dt_search_max_ms,
                min_ncc=metric_config.fwi_dt_min_ncc,
                gate_pre_ms=metric_config.fwi_dt_gate_pre_ms,
                gate_post_ms=metric_config.fwi_dt_gate_post_ms,
                cpml_thickness=metric_config.fwi_dt_cpml_thickness,
            )
            LOG.info("FWI context ready.")
        except Exception as _fwi_exc:
            LOG.error(
                "Failed to build FWI context — disabling FWI dt for this run: %s",
                _fwi_exc,
            )
            fwi_context = None

    metrics = tg.compute_metrics(
        metric_config,
        valid_pairs_mask=valid_pairs_mask,
        manual_picks=manual_picks,
        fwi_context=fwi_context,
    )

    qc_dir = Path(args.qc_dir) if args.qc_dir else bundle_file.parent / "processing_qc"
    src_labels = _build_source_labels(
        tg.n_sources,
        [w.strip() for w in args.source_boreholes.split(",") if w.strip()] if args.source_boreholes else None,
    )
    rec_labels = _build_receiver_labels(tg.n_receivers)
    pair_labels = _build_pair_labels(tg.n_sources, tg.n_receivers, src_labels, rec_labels)
    write_processing_qc(
        qc_dir=qc_dir,
        tg=tg,
        metrics=metrics,
        config=metric_config,
        max_pairs=args.qc_max_pairs,
        baseline_plot_samples=args.baseline_plot_samples,
        pair_labels=pair_labels,
        valid_pairs_mask=valid_pairs_mask,
    )
    processing_outputs = collect_processing_outputs(
        qc_dir=qc_dir,
        url_prefix=args.qc_url_prefix,
        max_items=args.max_qc_items,
    )

    inversion_outputs = collect_inversion_outputs(
        inversion_dir=Path(args.inversion_dir),
        url_prefix=args.inversion_url_prefix,
        max_items=args.max_inversion_items,
    )
    publish_bundle(
        bundle_path=bundle_file,
        manifest_path=manifest_file,
        tg=tg,
        metrics=metrics,
        preview_samples=args.preview_samples,
        processing_outputs=processing_outputs,
        inversion_outputs=inversion_outputs,
    )
    LOG.info("Published bundle: %s", bundle_file)
    LOG.info("Published manifest: %s", manifest_file)
    return 0


def load_config(config_file: Path) -> argparse.Namespace:
    """Load configuration from a YAML file and return an argparse.Namespace object.
    
    This allows all parameters to be stored in a single editable config file
    instead of scattered across CLI arguments.
    """
    if not yaml:
        raise RuntimeError(
            "PyYAML is required for config file loading. Install with: pip install pyyaml"
        )
    
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: expected YAML dictionary, got {type(cfg)}")
    
    # Flatten nested config structure into argparse-style attributes
    args = argparse.Namespace()
    
    # Data paths
    data = cfg.get("data", {})
    args.data_dir = data.get("data_dir", "/data/chet-cussp/cassm/CASSMdata")
    args.cache_file = data.get("cache_file", "/home/chopp/cassm_local/live/cassm_tempgather_full.h5")
    args.bundle_file = data.get("bundle_file", "/home/chopp/cassm_local/live/cassm_dashboard_bundle_full.npz")
    args.manifest_file = data.get("manifest_file", "/home/chopp/cassm_local/live/cassm_dashboard_manifest.json")
    args.qc_dir = data.get("qc_dir", "")
    args.qc_url_prefix = data.get("qc_url_prefix", "/cassm-processing")
    args.inversion_dir = data.get("inversion_dir", "/data/chet-cussp/cassm/inversion/live")
    args.inversion_url_prefix = data.get("inversion_url_prefix", "/cassm-inversion")
    
    # Geometry
    geom = cfg.get("geometry", {})
    args.n_sources = geom.get("n_sources", 16)
    args.n_receivers = geom.get("n_receivers", 72)
    args.sample_count = geom.get("sample_count", 3840)
    args.sample_rate_hz = geom.get("sample_rate_hz", 48000.0)
    
    # Channels
    channels = cfg.get("channels", {})
    args.known_bad_receiver_channels = channels.get("known_bad_channels", "72")
    args.active_receiver_channels = channels.get("active_channels", "")
    args.source_boreholes = channels.get("source_boreholes", "")
    
    # Preprocessing
    preproc = cfg.get("preprocessing", {})
    args.pick_search_s = preproc.get("pick_search_s", 0.012)
    args.window_s = preproc.get("window_s", 0.003)
    args.window_pre_pick_ms = preproc.get("window_pre_pick_ms")
    args.window_post_pick_ms = preproc.get("window_post_pick_ms")
    args.clip_first_s = preproc.get("clip_first_s", 0.002)
    args.mute_first_s = preproc.get("mute_first_s", 0.002)
    args.hydro_clip_first_s = preproc.get("hydro_clip_first_s")
    args.hydro_mute_first_s = preproc.get("hydro_mute_first_s")
    args.taper_fraction = preproc.get("taper_fraction", 0.01)
    
    # Filters
    filt = cfg.get("filters", {})
    args.filter_low_hz = filt.get("low_hz", 0.0)
    args.filter_high_hz = filt.get("high_hz", 0.0)
    args.filter_order = filt.get("order", 4)
    args.accel_filter_low_hz = filt.get("accel_low_hz")
    args.accel_filter_high_hz = filt.get("accel_high_hz")
    args.hydro_filter_low_hz = filt.get("hydro_low_hz")
    args.hydro_filter_high_hz = filt.get("hydro_high_hz")
    
    # Picking
    pick = cfg.get("picking", {})
    args.picker = pick.get("method", "aic")
    args.stalta_short_s = pick.get("stalta_short_s", 0.0002)
    args.stalta_long_s = pick.get("stalta_long_s", 0.0015)
    args.stalta_threshold = pick.get("stalta_threshold", 3.0)
    args.aic_margin_samples = pick.get("aic_margin_samples", 10)
    args.aic_min_snr = pick.get("aic_min_snr", 0.0)
    if "baseline_n_epochs" in pick:
        LOG.warning(
            "picking.baseline_n_epochs is deprecated and ignored; "
            "use only picking.baseline_end_date."
        )
    args.baseline_end_date = pick.get("baseline_end_date", "")
    
    # Cross-correlation
    xc = cfg.get("xcorr", {})
    args.dt_method = xc.get("method", "xcorr")
    args.xcorr_max_lag_ms = xc.get("max_lag_ms", 1.0)
    args.xcorr_accept_max_lag_ms = xc.get("accept_max_lag_ms", args.xcorr_max_lag_ms)
    args.xcorr_accept_max_lag_hydro_ms = xc.get("accept_max_lag_hydro_ms", args.xcorr_accept_max_lag_ms)
    args.xcorr_accept_max_lag_dm_hydro_ms = xc.get("accept_max_lag_dm_hydro_ms", 0.15)
    args.xcorr_min_peak_cc = xc.get("min_peak_cc", 0.6)
    args.xcorr_edge_guard_samples = xc.get("edge_guard_samples", 1)
    args.xcorr_despike_single_epoch = bool(xc.get("despike_single_epoch", True))
    args.xcorr_despike_mad_thresh = float(xc.get("despike_mad_thresh", 5.0))
    args.xcorr_mask_short_runs = bool(xc.get("mask_short_runs", True))
    args.xcorr_short_run_max_len_epochs = int(xc.get("short_run_max_len_epochs", 4))
    args.xcorr_short_run_min_amp_us = float(xc.get("short_run_min_amp_us", 35.0))
    args.xcorr_short_run_neighbor_tol_us = float(xc.get("short_run_neighbor_tol_us", 12.0))
    # Envelope-guided xcorr parameters
    args.envelope_guide_xcorr = bool(xc.get("envelope_guide", False))
    args.envelope_max_lag_s = float(xc.get("envelope_max_lag_ms", 5.0)) / 1000.0
    args.envelope_smooth_samples = int(xc.get("envelope_smooth_samples", 5))
    args.envelope_min_peak_cc = float(xc.get("envelope_min_peak_cc", 0.20))
    args.xcorr_fine_half_lag_s = float(xc.get("fine_half_lag_ms", 0.3)) / 1000.0
    args.envelope_guide_smooth_epochs = int(xc.get("guide_smooth_epochs", 7))

    # DTW parameters for DM*→TS pairs
    dtw = cfg.get("dtw", {})
    args.dtw_enabled = bool(dtw.get("enabled", True))
    args.dtw_max_shift_ms = float(dtw.get("max_shift_ms", 0.25))
    args.dtw_strain_limit = float(dtw.get("strain_limit", 2.0))
    args.dtw_min_ncc = float(dtw.get("min_ncc", 0.2))

    # FWI-derived dt
    fwi = cfg.get("fwi_dt", {})
    args.fwi_dt_enabled = bool(fwi.get("enabled", False))
    args.fwi_dt_sources_csv = fwi.get("sources_csv", "")
    args.fwi_dt_receivers_csv = fwi.get("receivers_csv", "")
    args.fwi_dt_solver = fwi.get("solver", "fd2d")
    args.fwi_dt_grid_dx_m = float(fwi.get("grid_dx_m", 0.5))
    args.fwi_dt_grid_dz_m = float(fwi.get("grid_dz_m", 0.5))
    args.fwi_dt_grid_padding_m = float(fwi.get("grid_padding_m", 20.0))
    args.fwi_dt_vp_background_mps = float(fwi.get("vp_background_mps", 3000.0))
    _fwi_freq_bands = fwi.get("freq_bands", None)
    args.fwi_dt_freq_bands = None if _fwi_freq_bands is None else [tuple(map(float, band)) for band in _fwi_freq_bands]
    args.fwi_dt_search_max_ms = float(fwi.get("search_max_ms", 2.0))
    args.fwi_dt_min_ncc = float(fwi.get("min_ncc", 0.2))
    args.fwi_dt_gate_pre_ms = fwi.get("gate_pre_ms", None)
    args.fwi_dt_gate_post_ms = fwi.get("gate_post_ms", None)
    args.fwi_dt_cpml_thickness = int(fwi.get("cpml_thickness", 20))

    # Manual picks
    mp = cfg.get("manual_picks", {})
    args.manual_picks_file = mp.get("file", "")
    args.require_manual_picks = mp.get("require", False)
    
    # Output
    out = cfg.get("output", {})
    args.preview_samples = out.get("preview_samples", 400)
    args.max_qc_items = out.get("max_qc_items", 20)
    args.max_inversion_items = out.get("max_inversion_items", 20)
    args.qc_max_pairs = out.get("max_pairs", 60)
    args.baseline_plot_samples = out.get("baseline_plot_samples", 60)
    
    # Watch mode
    wm = cfg.get("watch", {})
    args.watch = wm.get("enabled", False)
    args.period_s = wm.get("period_s", 300)
    
    LOG.info("Loaded configuration from %s", config_file)
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    """Minimal argument parser that only accepts --config and --watch.
    
    All processing parameters are defined in the YAML config file.
    Use --watch to enable continuous monitoring mode.
    """
    p = argparse.ArgumentParser(
        description="Headless CUSSP CASSM processing pipeline",
        epilog=(
            "Example usage:\n"
            "  cussp_cassm_process.py --config cussp_cassm_config.yaml\n"
            "  cussp_cassm_process.py --config cussp_cassm_config.yaml --watch"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file (required). All parameters are defined here.",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously and poll for new epochs (overrides watch.enabled in config)",
    )
    return p


def main() -> int:
    cli_args = build_arg_parser().parse_args()
    args = load_config(cli_args.config)
    
    # Override watch setting if --watch is provided on command line
    if cli_args.watch:
        args.watch = True
    
    if not args.watch:
        return run_once(args)

    LOG.info("Starting watch mode (%ds)", args.period_s)
    while True:
        try:
            run_once(args)
        except Exception as exc:
            LOG.exception("Processing cycle failed: %s", exc)
        time.sleep(max(args.period_s, 5))


if __name__ == "__main__":
    raise SystemExit(main())
