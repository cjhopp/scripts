#!/usr/bin/env python3
"""Phase A diagnostic: is the TS-string dt trend smooth enough to unwrap?

Cycle skipping on DM*->TS hydrophone pairs is an integer ambiguity: the fine
cross-correlation resolves the sub-cycle phase correctly but locks onto the
wrong cycle, displacing dt by an integer multiple of the dominant period T.
Unwrapping along the hydrophone string can resolve that integer -- but only if
the *true* dt difference between adjacent elements stays below T/2.

This script measures that gradient and reports the go/no-go number:

    fraction of adjacent-element steps with |d(dt)| > T/2

evaluated over the quiet (baseline) epochs, where dt is small and cycle skips
are not expected. A low fraction means a plain unwrap will work; a high one
means the unwrap must be quality-weighted with explicit break detection.

Outputs (default <bundle_dir>/string_unwrap_diag/):
  diag_string_unwrap_summary.json   go/no-go statistics
  diag_period_estimates.png         T from centroid frequency vs. band vs. autocorrelation
  diag_step_histogram.png           |d(dt)|/(T/2) for quiet vs. active epochs
  diag_string_waterfall_<SRC>.png   dt vs. along-string distance, coloured by cc
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cussp_cassm_process import (
    MetricConfig,
    _preprocess_waveform,
    load_config,
)

LOG = logging.getLogger("cussp_cassm_diag_string_unwrap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Receivers 48..71 are the hydrophone string; ch = rec + 1 and TS number = 73 - ch,
# so receiver 48 is TS24 (deepest) and receiver 71 is TS01.
HYDRO_REC_START = 48


@dataclass
class StringGeometry:
    """Hydrophone string ordered along its own arclength, gaps removed."""

    rec_indices: np.ndarray  # 0-based receiver indices, ordered TS01 -> TS24
    labels: List[str]
    arclength_m: np.ndarray  # cumulative distance from the first retained element

    @property
    def n_elements(self) -> int:
        return int(self.rec_indices.size)

    @property
    def gaps_m(self) -> np.ndarray:
        """Spacing between consecutive *retained* elements (length n_elements - 1)."""
        return np.diff(self.arclength_m)


def _ts_label_for_receiver(rec_idx: int) -> str:
    return f"TS{73 - (rec_idx + 1):02d}"


def _parse_channel_list(spec: str) -> set:
    return {int(x.strip()) for x in str(spec).split(",") if x.strip()}


def build_ts_string(
    receivers_csv: Path,
    n_receivers: int,
    active_channels: str,
    bad_channels: str,
) -> StringGeometry:
    """Order the usable TS elements along the string and compute arclength.

    The TS well is deviated, so arclength is the cumulative 3-D distance between
    consecutive elements rather than a depth difference.
    """
    df = pd.read_csv(receivers_csv)
    coords = {
        str(r.receiver_id).strip().upper(): np.array([r.x, r.y, r.z], dtype=np.float64)
        for r in df.itertuples()
    }

    active = _parse_channel_list(active_channels) if str(active_channels).strip() else None
    bad = _parse_channel_list(bad_channels) if str(bad_channels).strip() else set()

    # Walk TS01 -> TS24, which is receiver index 71 -> 48.
    rec_indices: List[int] = []
    labels: List[str] = []
    points: List[np.ndarray] = []
    for rec_idx in range(n_receivers - 1, HYDRO_REC_START - 1, -1):
        ch = rec_idx + 1
        label = _ts_label_for_receiver(rec_idx)
        if ch in bad or (active is not None and ch not in active):
            continue
        if label not in coords:
            LOG.warning("Receiver %s (ch %d) missing from geometry CSV; skipping.", label, ch)
            continue
        rec_indices.append(rec_idx)
        labels.append(label)
        points.append(coords[label])

    if len(rec_indices) < 3:
        raise ValueError(
            f"Only {len(rec_indices)} usable TS elements after applying "
            "active/bad channel filters; cannot assess string continuity."
        )

    pts = np.vstack(points)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(seg)])

    LOG.info(
        "TS string: %d usable elements %s..%s, %.1f m total, %.2f m median spacing.",
        len(rec_indices), labels[0], labels[-1], arclength[-1], float(np.median(seg)),
    )
    return StringGeometry(
        rec_indices=np.array(rec_indices, dtype=np.int32),
        labels=labels,
        arclength_m=arclength,
    )


def dm_source_indices(source_boreholes: str) -> List[int]:
    wells = [w.strip().upper() for w in str(source_boreholes).split(",") if w.strip()]
    return [i for i, w in enumerate(wells) if w.startswith("DM")]


def _source_label(source_boreholes: str, src_idx: int) -> str:
    wells = [w.strip().upper() for w in str(source_boreholes).split(",") if w.strip()]
    if src_idx >= len(wells):
        return f"Src{src_idx}"
    well = wells[src_idx]
    nth = wells[:src_idx].count(well) + 1
    return f"{well}S{nth}"


def band_period_us(args) -> float:
    """Dominant period implied by the hydrophone bandpass centre frequency."""
    lo = args.hydro_filter_low_hz or args.filter_low_hz
    hi = args.hydro_filter_high_hz or args.filter_high_hz
    if not lo or not hi or hi <= lo:
        raise ValueError(
            "Hydrophone bandpass is not configured; cannot derive the cycle-skip "
            "period. Set filters.hydro_low_hz and filters.hydro_high_hz."
        )
    return 1.0e6 / (0.5 * (float(lo) + float(hi)))


def period_from_centfreq(centfreq_khz: np.ndarray) -> np.ndarray:
    """Per-pair period (us) from the published centroid frequency (stored in kHz)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t_us = 1000.0 / centfreq_khz
    t_us[~np.isfinite(t_us)] = np.nan
    return t_us


def period_from_cache(
    cache_file: Path,
    pair_indices: np.ndarray,
    args,
    n_base_epochs: int,
    sample_rate_hz: float,
    n_receivers: int,
) -> Dict[int, float]:
    """Measure the dominant period per pair by autocorrelating baseline waveforms.

    The autocorrelation is taken over a window matching the one the cross-correlation
    actually uses; widening it lets low-frequency coda dominate and biases the period
    high, which would corrupt any downstream cycle arithmetic. Reads only the requested
    pairs from the HDF5 cache so the 28 GB file is never fully materialised.
    """
    import h5py

    periods: Dict[int, float] = {}
    metric_config = MetricConfig(
        clip_first_s=args.clip_first_s,
        mute_first_s=args.mute_first_s,
        hydro_clip_first_s=args.hydro_clip_first_s,
        hydro_mute_first_s=args.hydro_mute_first_s,
        taper_fraction=args.taper_fraction,
        filter_low_hz=args.filter_low_hz,
        filter_high_hz=args.filter_high_hz,
        filter_order=args.filter_order,
        hydro_filter_low_hz=args.hydro_filter_low_hz,
        hydro_filter_high_hz=args.hydro_filter_high_hz,
    )
    if args.window_pre_pick_ms is not None and args.window_post_pick_ms is not None:
        win_samples = int(
            (args.window_pre_pick_ms + args.window_post_pick_ms) / 1000.0 * sample_rate_hz
        )
    else:
        win_samples = int(args.window_s * sample_rate_hz)
    win_samples = max(win_samples, 16)

    with h5py.File(cache_file, "r") as f:
        ds = f["data"]
        if "valid_pair_indices" in f:
            vpi = f["valid_pair_indices"][:].astype(np.int32)
            inv = {int(p): i for i, p in enumerate(vpi)}
        else:
            inv = {p: p for p in range(ds.shape[1])}

        n_base = int(min(max(n_base_epochs, 1), ds.shape[0], 20))
        for pair_idx in pair_indices:
            cidx = inv.get(int(pair_idx))
            if cidx is None:
                continue
            stack = np.asarray(ds[0:n_base, cidx, :], dtype=np.float64)
            trace = _preprocess_waveform(
                stack.mean(axis=0),
                sample_rate_hz,
                metric_config,
                pair_index=int(pair_idx),
                n_receivers=n_receivers,
            ).astype(np.float64)
            t = _dominant_period_from_trace(trace, sample_rate_hz, win_samples)
            if t is not None:
                periods[int(pair_idx)] = t

    LOG.info("Autocorrelation period measured for %d/%d pairs.", len(periods), len(pair_indices))
    return periods


def _dominant_period_from_trace(
    trace: np.ndarray, sample_rate_hz: float, win_samples: int
) -> Optional[float]:
    """Lag (us) of the first secondary autocorrelation maximum within the metric window."""
    energy = trace**2
    if not np.any(energy > 0):
        return None
    peak = int(np.argmax(energy))
    half = max(win_samples // 2, 8)
    seg = trace[max(peak - half, 0) : peak + half]
    seg = seg - seg.mean()
    if seg.size < 16 or np.linalg.norm(seg) == 0.0:
        return None

    ac = np.correlate(seg, seg, mode="full")[seg.size - 1 :]
    ac /= ac[0]
    # First zero crossing marks the end of the central lobe; the next maximum is one period.
    neg = np.where(ac < 0)[0]
    if neg.size == 0:
        return None
    start = int(neg[0])
    if start >= ac.size - 2:
        return None
    lag = start + int(np.argmax(ac[start:]))
    if lag <= 0:
        return None
    return float(lag) * 1.0e6 / float(sample_rate_hz)


def adjacent_steps(
    dt_us: np.ndarray,
    cc: np.ndarray,
    geom: StringGeometry,
    src_idx: int,
    n_receivers: int,
    epoch_mask: np.ndarray,
    min_cc: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (|d(dt)| in us, spacing in m, validity mask) per adjacent element pair.

    Both members of a pair must have a finite dt and cc >= *min_cc*; consecutive
    entries in the string may be separated by more than the nominal spacing when
    intervening elements were dropped, so the true gap is returned alongside.
    """
    pair_rows = src_idx * n_receivers + geom.rec_indices
    dt_str = dt_us[np.ix_(pair_rows, epoch_mask)]
    cc_str = cc[np.ix_(pair_rows, epoch_mask)]

    good = np.isfinite(dt_str) & (cc_str >= min_cc)
    both = good[:-1, :] & good[1:, :]
    steps = np.abs(dt_str[1:, :] - dt_str[:-1, :])
    gaps = np.repeat(geom.gaps_m[:, None], dt_str.shape[1], axis=1)
    return steps[both], gaps[both], both


def _plot_period_estimates(
    out_dir: Path,
    t_centfreq: np.ndarray,
    t_band: float,
    t_autocorr: Dict[int, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    finite = t_centfreq[np.isfinite(t_centfreq)]
    if finite.size:
        ax.hist(finite, bins=80, color="tab:blue", alpha=0.65, density=True,
                label=f"centroid freq (median {np.median(finite):.1f} us)")
    if t_autocorr:
        vals = np.array(list(t_autocorr.values()))
        ax.hist(vals, bins=40, color="tab:green", alpha=0.55, density=True,
                label=f"autocorrelation (median {np.median(vals):.1f} us)")
    ax.axvline(t_band, color="k", ls="--", lw=1.5, label=f"band centre ({t_band:.1f} us)")
    ax.set_xlabel("Dominant period T (us)")
    ax.set_ylabel("Density")
    ax.set_title("Cycle-skip quantum estimates -- DM*->TS pairs, quiet epochs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "diag_period_estimates.png")
    plt.close(fig)


def _plot_step_histogram(
    out_dir: Path,
    quiet_norm: np.ndarray,
    active_norm: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    bins = np.linspace(0, 8, 161)
    if quiet_norm.size:
        ax.hist(quiet_norm, bins=bins, density=True, color="tab:blue", alpha=0.6,
                label=f"quiet epochs (n={quiet_norm.size})")
    if active_norm.size:
        ax.hist(active_norm, bins=bins, density=True, color="tab:red", alpha=0.5,
                label=f"active epochs (n={active_norm.size})")
    ax.axvline(1.0, color="k", ls="--", lw=1.5, label="T/2 (unwrap limit)")
    for k in (2, 4, 6):
        ax.axvline(k, color="grey", ls=":", lw=1.0)
    ax.set_xlabel("|d(dt)| between adjacent TS elements, in units of T/2")
    ax.set_ylabel("Density")
    ax.set_title(
        "Adjacent-element dt steps\n"
        "Mass beyond 1.0 breaks a plain unwrap; modes at 2, 4, 6 are whole-cycle skips"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "diag_step_histogram.png")
    plt.close(fig)


def _plot_string_waterfall(
    out_dir: Path,
    dt_us: np.ndarray,
    cc: np.ndarray,
    geom: StringGeometry,
    src_idx: int,
    src_label: str,
    n_receivers: int,
    t_ref_us: float,
    n_snapshots: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_rows = src_idx * n_receivers + geom.rec_indices
    n_epochs = dt_us.shape[1]
    snap = np.linspace(0, n_epochs - 1, min(n_snapshots, n_epochs)).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(snap):
        y = dt_us[pair_rows, e]
        q = cc[pair_rows, e]
        colour = cmap(i / max(len(snap) - 1, 1))
        ax.plot(geom.arclength_m, y, lw=0.8, alpha=0.5, color=colour)
        ok = np.isfinite(y)
        ax.scatter(geom.arclength_m[ok], y[ok], c=[colour], s=8 + 22 * np.clip(q[ok], 0, 1),
                   edgecolors="none", alpha=0.8)

    for k in range(-4, 5):
        if k:
            ax.axhline(k * t_ref_us, color="grey", ls=":", lw=0.7)
    ax.axhline(0.0, color="k", lw=0.8)

    ax.set_xlabel(f"Along-string distance from {geom.labels[0]} (m)")
    ax.set_ylabel("dt (us)")
    ax.set_title(
        f"{src_label} -> TS string: dt vs. along-string distance\n"
        f"colour = epoch (dark->bright), marker size = xcorr cc, "
        f"dotted lines = integer multiples of T={t_ref_us:.0f} us"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"diag_string_waterfall_{src_label}.png")
    plt.close(fig)


def run(args) -> int:
    cfg = load_config(Path(args.config))
    bundle_file = Path(args.bundle or cfg.bundle_file)
    receivers_csv = Path(args.receivers_csv or cfg.fwi_dt_receivers_csv)
    if not receivers_csv or not receivers_csv.exists():
        raise FileNotFoundError(
            f"Receiver geometry CSV not found: {receivers_csv}. "
            "Pass --receivers-csv explicitly."
        )

    out_dir = Path(args.out_dir) if args.out_dir else bundle_file.parent / "string_unwrap_diag"
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading bundle %s", bundle_file)
    z = np.load(bundle_file, allow_pickle=True)
    dt_us = z["dt_us"]
    cc = z["xcorr_peak_cc"]
    centfreq = z["centfreq"]
    epoch_times = pd.to_datetime([str(t) for t in z["epoch_times"]], utc=True)
    n_receivers = int(z["n_receivers"])
    sample_rate_hz = float(z["sample_rate_hz"])

    geom = build_ts_string(
        receivers_csv, n_receivers, cfg.active_receiver_channels, cfg.known_bad_receiver_channels
    )
    dm_sources = dm_source_indices(cfg.source_boreholes)
    if not dm_sources:
        raise ValueError("No DM* sources found in channels.source_boreholes.")
    LOG.info("DM* sources: %s", [_source_label(cfg.source_boreholes, s) for s in dm_sources])

    if not cfg.baseline_end_date:
        raise ValueError("picking.baseline_end_date is required to select quiet epochs.")
    cutoff = pd.Timestamp(cfg.baseline_end_date)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    quiet_mask = np.asarray(epoch_times <= cutoff)
    active_mask = ~quiet_mask
    LOG.info("Quiet epochs: %d, active epochs: %d", quiet_mask.sum(), active_mask.sum())
    if quiet_mask.sum() < 2:
        raise ValueError("Fewer than 2 quiet epochs; cannot establish the unwrap baseline.")

    dm_pair_rows = np.concatenate(
        [s * n_receivers + geom.rec_indices for s in dm_sources]
    ).astype(np.int32)

    t_band = band_period_us(cfg)
    t_centfreq_all = period_from_centfreq(centfreq[np.ix_(dm_pair_rows, np.where(quiet_mask)[0])])
    # One stable period per pair, used to normalise that pair's steps.
    with np.errstate(invalid="ignore"):
        t_per_pair = np.nanmedian(t_centfreq_all, axis=1)
    t_per_pair = np.where(np.isfinite(t_per_pair) & (t_per_pair > 0), t_per_pair, t_band)

    t_autocorr: Dict[int, float] = {}
    cache_file = Path(args.cache_file or cfg.cache_file)
    if not args.skip_cache_period and cache_file.exists():
        LOG.info("Measuring autocorrelation period from %s", cache_file)
        t_autocorr = period_from_cache(
            cache_file, dm_pair_rows, cfg, int(quiet_mask.sum()), sample_rate_hz, n_receivers
        )
    elif not args.skip_cache_period:
        LOG.warning("Cache file %s not found; skipping autocorrelation period check.", cache_file)

    row_to_t = {int(p): float(t) for p, t in zip(dm_pair_rows, t_per_pair)}

    quiet_norm: List[np.ndarray] = []
    active_norm: List[np.ndarray] = []
    quiet_grad: List[np.ndarray] = []
    per_source: Dict[str, Dict[str, float]] = {}

    for src_idx in dm_sources:
        label = _source_label(cfg.source_boreholes, src_idx)
        rows = src_idx * n_receivers + geom.rec_indices
        # Normalise each step by the mean period of the two elements involved.
        t_elem = np.array([row_to_t.get(int(r), t_band) for r in rows])
        t_step = 0.5 * (t_elem[:-1] + t_elem[1:])

        for mask, sink in ((quiet_mask, quiet_norm), (active_mask, active_norm)):
            if not mask.any():
                continue
            steps, gaps, both = adjacent_steps(
                dt_us, cc, geom, src_idx, n_receivers, mask, args.min_cc
            )
            if steps.size == 0:
                continue
            t_flat = np.repeat(t_step[:, None], both.shape[1], axis=1)[both]
            sink.append(steps / (0.5 * t_flat))
            if mask is quiet_mask:
                quiet_grad.append(steps / np.maximum(gaps, 1e-6))
                n_over = int(np.count_nonzero(steps > 0.5 * t_flat))
                per_source[label] = {
                    "n_steps": int(steps.size),
                    "n_over_half_period": n_over,
                    "frac_over_half_period": float(n_over / max(steps.size, 1)),
                    "median_step_us": float(np.median(steps)),
                    "p95_step_us": float(np.percentile(steps, 95)),
                    "median_gradient_us_per_m": float(np.median(steps / np.maximum(gaps, 1e-6))),
                }

        _plot_string_waterfall(
            out_dir, dt_us, cc, geom, src_idx, label, n_receivers,
            float(np.median(t_elem)), args.waterfall_snapshots,
        )

    quiet_all = np.concatenate(quiet_norm) if quiet_norm else np.array([])
    active_all = np.concatenate(active_norm) if active_norm else np.array([])
    grad_all = np.concatenate(quiet_grad) if quiet_grad else np.array([])
    if quiet_all.size == 0:
        raise ValueError(
            f"No adjacent-element steps passed the cc >= {args.min_cc} filter in the "
            "quiet window; lower --min-cc or check the bundle."
        )

    frac_over = float(np.count_nonzero(quiet_all > 1.0) / quiet_all.size)
    verdict = (
        "GO_SIMPLE" if frac_over < 0.02
        else "GO_VITERBI" if frac_over < 0.10
        else "NO_GO"
    )

    _plot_period_estimates(out_dir, t_centfreq_all, t_band, t_autocorr)
    _plot_step_histogram(out_dir, quiet_all, active_all)

    summary = {
        "bundle_file": str(bundle_file),
        "n_epochs": int(dt_us.shape[1]),
        "n_quiet_epochs": int(quiet_mask.sum()),
        "n_active_epochs": int(active_mask.sum()),
        "baseline_end_date": str(cfg.baseline_end_date),
        "min_cc": float(args.min_cc),
        "string": {
            "n_elements": geom.n_elements,
            "first": geom.labels[0],
            "last": geom.labels[-1],
            "total_length_m": float(geom.arclength_m[-1]),
            "median_spacing_m": float(np.median(geom.gaps_m)),
        },
        "period_us": {
            "band_centre": float(t_band),
            "centfreq_median": float(np.nanmedian(t_centfreq_all)),
            "autocorr_median": (
                float(np.median(list(t_autocorr.values()))) if t_autocorr else None
            ),
            "autocorr_n_pairs": len(t_autocorr),
        },
        "go_no_go": {
            "verdict": verdict,
            "frac_quiet_steps_over_half_period": frac_over,
            "n_quiet_steps": int(quiet_all.size),
            "median_quiet_step_in_half_periods": float(np.median(quiet_all)),
            "p95_quiet_step_in_half_periods": float(np.percentile(quiet_all, 95)),
            "median_quiet_gradient_us_per_m": (
                float(np.median(grad_all)) if grad_all.size else None
            ),
            "frac_active_steps_over_half_period": (
                float(np.count_nonzero(active_all > 1.0) / active_all.size)
                if active_all.size else None
            ),
        },
        "per_source_quiet": per_source,
        "interpretation": {
            "GO_SIMPLE": "<2% of quiet steps exceed T/2: quality-anchored unwrap suffices.",
            "GO_VITERBI": "2-10%: use the candidate-ladder Viterbi unwrap with break detection.",
            "NO_GO": ">10%: the string is not smooth enough at this spacing; "
                     "unwrapping alone will not resolve the ambiguity.",
        }[verdict],
    }
    (out_dir / "diag_string_unwrap_summary.json").write_text(json.dumps(summary, indent=2))

    LOG.info("=" * 72)
    LOG.info("VERDICT: %s", verdict)
    LOG.info("  quiet steps exceeding T/2 : %.3f%% (%d of %d)",
             100.0 * frac_over, int(np.count_nonzero(quiet_all > 1.0)), quiet_all.size)
    LOG.info("  median quiet step         : %.3f x (T/2)", float(np.median(quiet_all)))
    LOG.info("  median quiet gradient     : %s us/m",
             f"{np.median(grad_all):.2f}" if grad_all.size else "n/a")
    if active_all.size:
        LOG.info("  active steps exceeding T/2: %.3f%%",
                 100.0 * np.count_nonzero(active_all > 1.0) / active_all.size)
    LOG.info("  %s", summary["interpretation"])
    LOG.info("Outputs written to %s", out_dir)
    LOG.info("=" * 72)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase A: assess whether TS-string dt is smooth enough to unwrap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Processing YAML config (same file used by cussp_cassm_process.py).")
    p.add_argument("--bundle", type=Path, default=None,
                   help="Bundle NPZ. Defaults to data.bundle_file from the config.")
    p.add_argument("--receivers-csv", type=Path, default=None,
                   help="Receiver geometry CSV. Defaults to fwi_dt.receivers_csv.")
    p.add_argument("--cache-file", type=Path, default=None,
                   help="HDF5 waveform cache, used for the autocorrelation period check.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory. Defaults to <bundle_dir>/string_unwrap_diag.")
    p.add_argument("--min-cc", type=float, default=0.5,
                   help="Minimum xcorr_peak_cc for an element to enter the statistics.")
    p.add_argument("--waterfall-snapshots", type=int, default=40,
                   help="Number of epochs drawn in each per-source waterfall plot.")
    p.add_argument("--skip-cache-period", action="store_true",
                   help="Skip the autocorrelation period check (avoids reading the HDF5 cache).")
    return p


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
