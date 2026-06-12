#!/usr/bin/env python3
"""Phase 0 Diagnostics: Characterize large-dt failure on DM*→TS pairs.

Analyzes existing cassm_dashboard_bundle_full.npz output to:
  1. Classify failure modes: cycle-skip (~100 µs jumps) vs. decorrelation (peak_cc collapse) vs. clipping (bounds).
  2. Histogram dt jumps, NaN patterns, peak_cc distribution.
  3. Plot raw-window amplitude spectra for DM→TS pairs to see sub-5 kHz energy.
  4. Suggest diagnostics: whether low-freq unwrapping is viable, and estimated DTW parameters.

Usage:
  python cussp_cassm_diag_large_dt.py <bundle_file> [--output-dir <dir>] [--n-pairs <N>]

Output:
  <output-dir>/diagnostic_report.txt   - text summary
  <output-dir>/dt_histogram.png        - dt value histogram + epoch-to-epoch jumps
  <output-dir>/peak_cc_vs_dt.png       - scatter: peak_cc vs |dt|
  <output-dir>/spectrum_samples.png    - raw-window spectra for N DM→TS pairs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

# Try matplotlib; if unavailable, skip plots but still do text analysis.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def analyze_dt_failures(
    dt_us: np.ndarray,
    xcorr_peak_cc: np.ndarray,
    envelope_peak_cc: np.ndarray,
    accept_max_lag_dm_hydro_us: float = 150.0,
    period_us: float = 100.0,  # ~100 µs period at 10 kHz
) -> Dict[str, object]:
    """Classify dt failures on DM*→TS pairs.

    Parameters
    ----------
    dt_us : (n_pairs, n_epochs) array; DM*→TS subset dt values in µs
    xcorr_peak_cc : (n_pairs, n_epochs) xcorr peak correlation
    envelope_peak_cc : (n_pairs, n_epochs) envelope-guided cc (may be NaN if not used)
    accept_max_lag_dm_hydro_us : float; acceptance gate upper bound
    period_us : float; nominal signal period (100 µs @ 10 kHz)

    Returns
    -------
    dict with keys:
      n_valid_pairs, n_epochs, n_finite_dt, n_rejected (NaN)
      n_at_bound (clipped at ±accept_max_lag)
      n_cycle_skip_jumps (epoch-to-epoch jump ~ half_period)
      mean_peak_cc_valid, mean_peak_cc_rejected
      dt_range_finite, max_jump_magnitude_us
      recommendations (str)
    """
    n_pairs, n_epochs = dt_us.shape
    dt_valid = np.isfinite(dt_us)
    n_valid = int(np.sum(dt_valid))
    n_rejected = int(np.sum(~dt_valid))

    # Clipping check
    at_bound = np.abs(dt_us) >= (accept_max_lag_dm_hydro_us - 5.0)  # -5 µs tolerance for rounding
    n_at_bound = int(np.sum(at_bound & dt_valid))

    # Cycle-skip pattern: epoch-to-epoch jumps ~ half_period
    half_period = 0.5 * period_us
    epoch_jumps = np.diff(dt_us, axis=1)  # (n_pairs, n_epochs-1)
    jump_mag = np.abs(epoch_jumps)
    # Jumps near a half-period or full period
    cycle_skip_jumps = (
        ((jump_mag >= half_period * 0.8) & (jump_mag <= half_period * 1.2)) |
        ((jump_mag >= period_us * 0.8) & (jump_mag <= period_us * 1.2))
    )
    n_cycle_skip = int(np.sum(cycle_skip_jumps))
    max_jump_mag = float(np.nanmax(jump_mag)) if jump_mag.size > 0 else 0.0

    # Decorrelation: compare peak_cc for valid vs rejected
    cc_valid = xcorr_peak_cc[dt_valid]
    cc_rejected = xcorr_peak_cc[~dt_valid]
    mean_cc_valid = float(np.mean(cc_valid)) if cc_valid.size else np.nan
    mean_cc_rejected = float(np.mean(cc_rejected)) if cc_rejected.size else np.nan

    # dt range
    dt_finite = dt_us[dt_valid]
    dt_min = float(np.min(dt_finite)) if dt_finite.size else np.nan
    dt_max = float(np.max(dt_finite)) if dt_finite.size else np.nan

    # Envelope guidance usage (if available)
    env_used = np.isfinite(envelope_peak_cc)
    n_env_guided = int(np.sum(env_used))

    # Recommendations
    recs = []
    if n_rejected / (n_valid + n_rejected + 1e-10) > 0.3:
        recs.append("• High rejection rate (>30%) → likely decorrelation; DTW + progressive filtering recommended.")
    if n_cycle_skip > n_epochs * n_pairs * 0.01:  # >1% of epoch-to-epoch transitions
        recs.append("• Frequent cycle-skip jumps detected → DTW primary tool (recovers cycle-unambiguous lag).")
    if n_at_bound / (n_valid + 1e-10) > 0.1:
        recs.append(f"• >10% of valid dt at ±{accept_max_lag_dm_hydro_us} µs bound → may need wider acceptance gate or larger-shift estimator.")
    if abs(dt_max - dt_min) > 3.0 * period_us:
        recs.append("• Wide dt range (>3 periods) → cycle-skip/unwrapping definitely a factor; DTW + low-freq check needed.")
    if abs(mean_cc_valid - mean_cc_rejected) < 0.1:
        recs.append("• Valid vs rejected peak_cc similar → not pure decorrelation; likely geometry/picking issue.")

    if not recs:
        recs.append("• Mixture of failure modes or unclear signal. Recommend: DTW hybrid (Phase 1–2) for robust recovery.")

    recommendation_str = "\n".join(recs)

    return {
        "n_valid_pairs": n_pairs,
        "n_epochs": n_epochs,
        "n_finite_dt": n_valid,
        "n_rejected": n_rejected,
        "n_at_bound": n_at_bound,
        "n_cycle_skip_jumps": n_cycle_skip,
        "mean_peak_cc_valid": mean_cc_valid,
        "mean_peak_cc_rejected": mean_cc_rejected,
        "dt_range": (dt_min, dt_max),
        "max_jump_magnitude_us": max_jump_mag,
        "n_envelope_guided": n_env_guided,
        "recommendations": recommendation_str,
    }


def plot_dt_histogram(
    dt_us: np.ndarray,
    xcorr_peak_cc: np.ndarray,
    output_file: Path,
    accept_max_lag_dm_hydro_us: float = 150.0,
) -> None:
    """Histogram of dt values and epoch-to-epoch jumps."""
    if not _MATPLOTLIB_AVAILABLE:
        LOG.warning("Matplotlib not available; skipping dt histogram plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 0: dt histogram
    dt_valid = dt_us[np.isfinite(dt_us)]
    axes[0, 0].hist(dt_valid, bins=50, edgecolor="k", alpha=0.7)
    axes[0, 0].axvline(-accept_max_lag_dm_hydro_us, color="r", linestyle="--", label="Accept bounds")
    axes[0, 0].axvline(accept_max_lag_dm_hydro_us, color="r", linestyle="--")
    axes[0, 0].set_xlabel("dt (µs)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Distribution of valid dt values")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 1: epoch-to-epoch jumps
    jumps = np.diff(dt_us, axis=1)
    jump_valid = jumps[np.isfinite(jumps)]
    axes[0, 1].hist(jump_valid, bins=50, edgecolor="k", alpha=0.7, color="orange")
    axes[0, 1].axvline(50.0, color="r", linestyle="--", label="Half-period (10 kHz)")
    axes[0, 1].axvline(-50.0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("Epoch-to-epoch jump (µs)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Time-series continuity (cycle-skip ~ ±50 µs)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 2: peak_cc distribution (valid vs rejected)
    cc_valid = xcorr_peak_cc[np.isfinite(dt_us)]
    cc_rejected = xcorr_peak_cc[~np.isfinite(dt_us)]
    axes[1, 0].hist(cc_valid, bins=40, alpha=0.6, label="Valid dt", edgecolor="k")
    axes[1, 0].hist(cc_rejected, bins=40, alpha=0.6, label="Rejected dt", edgecolor="k", color="red")
    axes[1, 0].set_xlabel("xcorr peak_cc")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Peak correlation: valid vs rejected")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 3: scatter peak_cc vs |dt|
    valid_idx = np.isfinite(dt_us)
    axes[1, 1].scatter(
        np.abs(dt_us[valid_idx]), xcorr_peak_cc[valid_idx],
        alpha=0.4, s=10, label="Valid"
    )
    axes[1, 1].axvline(accept_max_lag_dm_hydro_us, color="r", linestyle="--", label="Accept bound")
    axes[1, 1].set_xlabel("|dt| (µs)")
    axes[1, 1].set_ylabel("xcorr peak_cc")
    axes[1, 1].set_title("Correlation vs dt magnitude")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    LOG.info("Saved dt histogram to %s", output_file)
    plt.close(fig)


def plot_spectra(
    data: np.ndarray,
    pair_indices: list[int],
    sample_rate_hz: float,
    output_file: Path,
    n_fft: int = 2048,
) -> None:
    """Plot amplitude spectra for sample DM→TS pair windows.

    Parameters
    ----------
    data : (n_samples,) or (n_pairs, n_samples); raw waveform(s)
    pair_indices : list of pair indices to plot (or [] for first N rows)
    sample_rate_hz : float
    output_file : Path
    n_fft : int
    """
    if not _MATPLOTLIB_AVAILABLE:
        LOG.warning("Matplotlib not available; skipping spectrum plots.")
        return

    fig, axes = plt.subplots(
        len(pair_indices), 1,
        figsize=(10, 3 * len(pair_indices))
    )
    if len(pair_indices) == 1:
        axes = [axes]

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate_hz)

    for idx, pair_idx in enumerate(pair_indices):
        if len(data.shape) == 1:
            trace = data
        else:
            trace = data[pair_idx] if pair_idx < data.shape[0] else data[0]

        # Pad and window
        trace_pad = np.zeros(n_fft, dtype=trace.dtype)
        trace_pad[:min(len(trace), n_fft)] = trace[:min(len(trace), n_fft)]
        window = np.hanning(len(trace_pad))
        trace_windowed = trace_pad * window

        spec = np.abs(np.fft.rfft(trace_windowed, n=n_fft))
        spec_db = 20.0 * np.log10(spec / np.max(spec) + 1e-10)

        axes[idx].semilogy(freqs, spec, linewidth=1)
        axes[idx].axvline(5000.0, color="r", linestyle="--", alpha=0.5, label="5 kHz (filter corner)")
        axes[idx].axvline(15000.0, color="r", linestyle="--", alpha=0.5, label="15 kHz (filter corner)")
        axes[idx].set_xlabel("Frequency (Hz)")
        axes[idx].set_ylabel("Amplitude (linear)")
        axes[idx].set_title(f"DM→TS pair {pair_idx}: raw-window spectrum")
        axes[idx].legend(loc="upper right")
        axes[idx].grid(True, alpha=0.3, which="both")
        axes[idx].set_xlim([0, sample_rate_hz / 2.0])

    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches="tight")
    LOG.info("Saved spectra to %s", output_file)
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
        return {}


def extract_dm_ts_subset(
    data_dict: Dict[str, np.ndarray],
    source_boreholes: list[str],
    n_receivers: int = 72,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Extract DM*→TS (ch49+) subset from full arrays.

    Returns
    -------
    dt_dm_ts, xcorr_peak_cc_dm_ts, envelope_peak_cc_dm_ts, pair_indices
    """
    dt_us = data_dict.get("dt_us", np.array([]))
    xcorr_peak_cc = data_dict.get("xcorr_peak_cc", np.array([]))
    envelope_peak_cc = data_dict.get("envelope_peak_cc", np.array([]))

    if dt_us.size == 0:
        LOG.error("No dt_us in bundle.")
        return np.array([]), np.array([]), np.array([]), []

    dm_ts_indices = []
    for pair_idx in range(dt_us.shape[0]):
        src_idx = pair_idx // n_receivers
        rec_idx = pair_idx % n_receivers
        if src_idx < len(source_boreholes):
            src_bh = source_boreholes[src_idx].upper().strip()
            is_dm = src_bh.startswith("DM")
            is_ts = rec_idx >= 48  # ch 49+ = indices 48+
            if is_dm and is_ts:
                dm_ts_indices.append(pair_idx)

    if not dm_ts_indices:
        LOG.warning("No DM*→TS pairs found. Analyzing all pairs instead.")
        dm_ts_indices = list(range(min(100, dt_us.shape[0])))

    dt_subset = dt_us[dm_ts_indices]
    cc_subset = xcorr_peak_cc[dm_ts_indices] if xcorr_peak_cc.size > 0 else np.zeros_like(dt_subset)
    env_subset = envelope_peak_cc[dm_ts_indices] if envelope_peak_cc.size > 0 else np.full_like(dt_subset, np.nan)

    return dt_subset, cc_subset, env_subset, dm_ts_indices


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 Diagnostics: analyze large-dt failures on DM*→TS pairs."
    )
    parser.add_argument("bundle_file", type=Path, help="Path to cassm_dashboard_bundle_full.npz")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots/report.")
    parser.add_argument("--n-pairs", type=int, default=5, help="Number of DM→TS pairs to plot spectra for.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(name)s — %(levelname)s — %(message)s",
    )

    bundle_file = Path(args.bundle_file)
    if not bundle_file.exists():
        LOG.error("Bundle file not found: %s", bundle_file)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else bundle_file.parent / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Loading bundle: %s", bundle_file)

    data = load_bundle(bundle_file)
    if not data:
        return 1

    # Get source_boreholes from config if available (fallback to default)
    source_boreholes = data.get("source_boreholes", None)
    if source_boreholes is None:
        # Fallback from config
        source_boreholes = ["AML", "AML", "AML", "AML", "AMU", "AMU", "AMU", "AMU",
                            "DML", "DML", "DML", "DML", "DMU", "DMU", "DMU", "DMU"]
        LOG.info("Using default source_boreholes; check config if incorrect.")
    elif isinstance(source_boreholes, np.ndarray):
        source_boreholes = source_boreholes.tolist()

    # Extract DM*→TS subset
    sample_rate_hz = float(data.get("sample_rate_hz", 48000.0))
    n_receivers = int(data.get("n_receivers", 72))

    dt_dm_ts, cc_dm_ts, env_dm_ts, pair_indices = extract_dm_ts_subset(
        data, source_boreholes, n_receivers
    )

    if dt_dm_ts.size == 0:
        LOG.error("Could not extract DM*→TS subset.")
        return 1

    LOG.info("Analyzing %d DM*→TS pairs, %d epochs", *dt_dm_ts.shape)

    # Run analysis
    analysis = analyze_dt_failures(
        dt_dm_ts, cc_dm_ts, env_dm_ts,
        accept_max_lag_dm_hydro_us=150.0,
        period_us=100.0,
    )

    # Write text report
    report_file = output_dir / "diagnostic_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("CUSSP CASSM Phase 0 Diagnostics: Large-dt Failure Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Bundle: {bundle_file}\n")
        f.write(f"Sample rate: {sample_rate_hz:.0f} Hz\n")
        f.write(f"Signal band: 5–15 kHz (period ~100 µs @ 10 kHz)\n")
        f.write(f"DM*→TS pairs analyzed: {analysis['n_valid_pairs']}\n")
        f.write(f"Epochs per pair: {analysis['n_epochs']}\n\n")

        f.write("FAILURE CLASSIFICATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Valid dt measurements:  {analysis['n_finite_dt']}\n")
        f.write(f"Rejected (NaN):         {analysis['n_rejected']}\n")
        f.write(f"At acceptance bound:    {analysis['n_at_bound']}\n")
        f.write(f"Cycle-skip jumps (~50 µs): {analysis['n_cycle_skip_jumps']}\n")
        f.write(f"Envelope-guided epochs: {analysis['n_envelope_guided']}\n\n")

        dt_min, dt_max = analysis["dt_range"]
        f.write(f"dt range (finite):      [{dt_min:.1f}, {dt_max:.1f}] µs\n")
        f.write(f"Max epoch-to-epoch jump: {analysis['max_jump_magnitude_us']:.1f} µs\n")
        f.write(f"Mean peak_cc (valid):   {analysis['mean_peak_cc_valid']:.3f}\n")
        f.write(f"Mean peak_cc (rejected):{analysis['mean_peak_cc_rejected']:.3f}\n\n")

        f.write("INTERPRETATION & RECOMMENDATIONS\n")
        f.write("-" * 70 + "\n")
        f.write(analysis["recommendations"])
        f.write("\n\n")

        f.write("NEXT STEPS\n")
        f.write("-" * 70 + "\n")
        f.write("1. Review plots in this directory.\n")
        f.write("2. If cycle-skip jumps are frequent: proceed to Phase 1 (DTW).\n")
        f.write("3. If sub-5 kHz energy is present in spectrum plot: Phase 4 (progressive unwrapping) may help.\n")
        f.write("4. If mostly decorrelation: verify waveform preprocessing; may need shape-change tolerance.\n")

    LOG.info("Saved diagnostic report to %s", report_file)
    print(f"\n{analysis['recommendations']}\n")

    # Generate plots
    if _MATPLOTLIB_AVAILABLE:
        plot_dt_histogram(
            dt_dm_ts, cc_dm_ts,
            output_dir / "dt_histogram.png",
            accept_max_lag_dm_hydro_us=150.0,
        )

        # Plot spectra for sample pairs
        if "raw_data" in data or "cache_file" in data:
            LOG.warning("Raw data not in bundle; spectra plot skipped. (Would need to load raw cache.)")
        else:
            LOG.info("Raw waveform data not available in bundle for spectrum plots.")

    LOG.info("Diagnostics complete. Output dir: %s", output_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
