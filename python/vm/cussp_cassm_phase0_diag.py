#!/usr/bin/env python3
"""Phase 0 diagnostic for DM*→TS large-dt failures.

Characterises the failure modes on DM*→TS hydrophone pairs using the
stored dashboard bundle (NPZ) and optionally the raw waveform cache (HDF5).

Outputs a multi-panel figure and a text summary that answers:
  A. Are failures from cycle-skipping?
       Epoch-to-epoch dt jumps clustered near ±100 µs = cycle-skip signature.
  B. Are failures from decorrelation?
       xcorr_peak_cc collapse (NaN epochs with low cc = decorrelation).
  C. Is the accept bound clipping legitimate shifts?
       dt_us piling up at ±accept_max_lag_dm_hydro_ms.
  D. Is envelope_lag_us tracking consistently?
       dt_us vs envelope_lag_us scatter (should be 1:1 if envelope is reliable).
  E. Is there usable energy below 5 kHz in DM→TS waveforms?
       Decides whether progressive low-freq unwrapping (Phase 4) is viable.

Usage
-----
  python cussp_cassm_phase0_diag.py --config cussp_cassm_config.yaml \\
      [--bundle /path/to/bundle.npz] [--h5 /path/to/cache.h5] \\
      [--out-dir /path/to/output]  [--epoch-range 100:200]

The --bundle and --h5 paths default to the values in the config file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("cussp_cassm_phase0_diag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config_minimal(config_path: Path) -> dict:
    """Return only the fields needed for Phase 0 diagnostics."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: expected dict, got {type(cfg)}")

    channels = cfg.get("channels", {})
    geom     = cfg.get("geometry", {})
    xc       = cfg.get("xcorr", {})
    data     = cfg.get("data", {})

    src_bh_str = channels.get("source_boreholes", "")
    src_boreholes = [s.strip().upper() for s in src_bh_str.split(",") if s.strip()] \
        if src_bh_str else []

    return {
        "source_boreholes":           src_boreholes,
        "n_sources":                  int(geom.get("n_sources", 16)),
        "n_receivers":                int(geom.get("n_receivers", 72)),
        "sample_rate_hz":             float(geom.get("sample_rate_hz", 48000.0)),
        "accept_max_lag_dm_hydro_ms": float(xc.get("accept_max_lag_dm_hydro_ms", 0.15)),
        "bundle_file":                str(data.get("bundle_file", "")),
        "cache_file":                 str(data.get("cache_file", "")),
        "hydro_filter_low_hz":        cfg.get("filters", {}).get("hydro_low_hz"),
        "hydro_filter_high_hz":       cfg.get("filters", {}).get("hydro_high_hz"),
    }


# ---------------------------------------------------------------------------
# Pair identification
# ---------------------------------------------------------------------------

def _dm_ts_pair_indices(
    source_boreholes: List[str],
    n_sources: int,
    n_receivers: int,
    hydro_first_ch: int = 48,    # receiver index at which hydrophones start (0-based)
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dm_ts_pair_indices, all_valid_src_indices_for_dm).

    DM*→TS pairs: source_boreholes[src_idx].startswith("DM") and rec_idx >= hydro_first_ch.
    """
    dm_src_idx = [
        i for i, bh in enumerate(source_boreholes[:n_sources]) if bh.startswith("DM")
    ]
    if not dm_src_idx:
        LOG.warning("No DM* sources found in source_boreholes list; check config.")
    pairs = [
        si * n_receivers + ri
        for si in dm_src_idx
        for ri in range(hydro_first_ch, n_receivers)
    ]
    return np.array(pairs, dtype=np.int64), np.array(dm_src_idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------

def _load_bundle(bundle_path: Path) -> dict:
    """Load the dashboard bundle NPZ and return a dict of arrays."""
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    LOG.info("Loading bundle: %s", bundle_path)
    raw = np.load(bundle_path, allow_pickle=True)
    keys = list(raw.files)
    out: dict = {}
    for k in keys:
        try:
            v = raw[k]
            out[k] = v.item() if v.ndim == 0 else v
        except Exception:
            out[k] = raw[k]
    LOG.info(
        "Bundle loaded: %d pairs × %d epochs, keys=%s",
        int(out.get("n_sources", 0)) * int(out.get("n_receivers", 0)),
        int(out.get("n_epochs", 0)),
        sorted(k for k in keys if not k.startswith("gather")),
    )
    return out


# ---------------------------------------------------------------------------
# Spectral analysis helpers
# ---------------------------------------------------------------------------

def _median_spectrum_db(
    waveforms: np.ndarray,   # (n_epochs, n_samples)
    sample_rate_hz: float,
    fmin_hz: float = 100.0,
    fmax_hz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median amplitude spectrum in dB over epochs, restricted to [fmin, fmax]."""
    n_epochs, n_samples = waveforms.shape
    if n_epochs == 0 or n_samples == 0:
        return np.array([]), np.array([])
    n_fft = int(2 ** np.ceil(np.log2(max(n_samples, 64))))
    specs = []
    for e in range(n_epochs):
        s = np.abs(np.fft.rfft(waveforms[e].astype(np.float64), n=n_fft))
        specs.append(s)
    specs_arr = np.stack(specs, axis=0)  # (n_epochs, n_fft//2+1)
    median_amp = np.median(specs_arr, axis=0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    mask = freqs >= fmin_hz
    if fmax_hz is not None:
        mask &= freqs <= fmax_hz
    freqs = freqs[mask]
    median_amp = median_amp[mask]
    eps = np.finfo(np.float64).tiny
    db = 20.0 * np.log10(np.maximum(median_amp, eps))
    db -= db.max()   # normalise to 0 dB peak
    return freqs, db


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_phase0(
    bundle_path: Path,
    config_path: Path,
    h5_path: Optional[Path],
    out_dir: Path,
    epoch_range: Optional[Tuple[int, int]],
    n_spectrum_pairs: int = 5,
) -> None:
    cfg = _load_config_minimal(config_path)

    # Override bundle/h5 from args if the config path matches default
    if not bundle_path.exists() and cfg["bundle_file"]:
        bundle_path = Path(cfg["bundle_file"])
    if h5_path is None and cfg["cache_file"]:
        h5_path = Path(cfg["cache_file"])

    bundle = _load_bundle(bundle_path)

    n_sources   = int(bundle.get("n_sources",   cfg["n_sources"]))
    n_receivers = int(bundle.get("n_receivers", cfg["n_receivers"]))
    n_epochs    = int(bundle.get("n_epochs",    0))
    sample_rate = float(bundle.get("sample_rate_hz", cfg["sample_rate_hz"]))
    n_pairs     = n_sources * n_receivers

    src_bh = cfg["source_boreholes"]
    if len(src_bh) < n_sources:
        # Pad with generic labels if config doesn't cover all sources
        src_bh = src_bh + [f"Src{i}" for i in range(len(src_bh), n_sources)]

    dm_ts_idx, dm_src_idx = _dm_ts_pair_indices(
        src_bh, n_sources, n_receivers, hydro_first_ch=48
    )

    accept_bound_us = cfg["accept_max_lag_dm_hydro_ms"] * 1000.0

    # Pull arrays from bundle — all shaped (n_pairs, n_epochs) as float32.
    def _get(key: str, shape: tuple) -> np.ndarray:
        if key in bundle:
            arr = np.asarray(bundle[key], dtype=np.float32)
            if arr.shape == shape:
                return arr
            if arr.size == np.prod(shape):
                return arr.reshape(shape).astype(np.float32)
        return np.full(shape, np.nan, dtype=np.float32)

    shape = (n_pairs, n_epochs)
    dt_us_full           = _get("dt_us",               shape)
    xcorr_cc_full        = _get("xcorr_peak_cc",       shape)
    xcorr_edge_full      = _get("xcorr_edge_hit",       shape)
    env_lag_full         = _get("envelope_lag_us",      shape)
    env_smooth_lag_full  = _get("envelope_smooth_lag_us", shape)
    env_cc_full          = _get("envelope_peak_cc",     shape)

    # Restrict to requested epoch range
    e0, e1 = 0, n_epochs
    if epoch_range is not None:
        e0 = max(0, epoch_range[0])
        e1 = min(n_epochs, epoch_range[1])
    epoch_slice = slice(e0, e1)

    dt_dm    = dt_us_full[np.ix_(dm_ts_idx, np.arange(e0, e1))].astype(np.float64)
    cc_dm    = xcorr_cc_full[np.ix_(dm_ts_idx, np.arange(e0, e1))].astype(np.float64)
    edge_dm  = xcorr_edge_full[np.ix_(dm_ts_idx, np.arange(e0, e1))]
    env_dm   = env_lag_full[np.ix_(dm_ts_idx, np.arange(e0, e1))].astype(np.float64)
    env_sm   = env_smooth_lag_full[np.ix_(dm_ts_idx, np.arange(e0, e1))].astype(np.float64)
    env_cc_dm = env_cc_full[np.ix_(dm_ts_idx, np.arange(e0, e1))].astype(np.float64)

    n_dm_pairs = len(dm_ts_idx)
    n_ep = e1 - e0
    dt_flat    = dt_dm.ravel()
    cc_flat    = cc_dm.ravel()
    env_flat   = env_dm.ravel()

    # -----------------------------------------------------------------------
    # A. Epoch-to-epoch dt jumps (cycle-skip detection)
    # -----------------------------------------------------------------------
    dt_valid  = np.where(np.isfinite(dt_dm), dt_dm, np.nan)
    dt_jumps  = np.diff(dt_valid, axis=1).ravel()
    dt_jumps  = dt_jumps[np.isfinite(dt_jumps)]

    # -----------------------------------------------------------------------
    # B. NaN/reject characterisation
    # -----------------------------------------------------------------------
    n_total   = dt_dm.size
    n_nan     = int(np.sum(~np.isfinite(dt_dm)))
    n_edge    = int(np.sum(edge_dm > 0))
    # cc at rejected epochs (NaN dt): are they low cc = decorrelation, or
    # non-NaN cc = rejected by lag bound = clipping?
    nan_mask  = ~np.isfinite(dt_flat)
    cc_at_nan = cc_flat[nan_mask & np.isfinite(cc_flat)]
    # "Low cc" = cc < 0.3 at rejected epochs — likely decorrelation.
    # "High cc" = cc >= 0.3 at rejected epochs — likely lag clipping.
    n_low_cc_reject  = int(np.sum(cc_at_nan < 0.3))
    n_high_cc_reject = int(np.sum(cc_at_nan >= 0.3))

    # -----------------------------------------------------------------------
    # C. Accept-bound pile-up (clipping detection)
    # -----------------------------------------------------------------------
    dt_accepted = dt_flat[np.isfinite(dt_flat)]
    clip_tol    = 3.0  # µs tolerance to detect pile-up near bound
    n_near_pos  = int(np.sum(dt_accepted > accept_bound_us - clip_tol))
    n_near_neg  = int(np.sum(dt_accepted < -accept_bound_us + clip_tol))

    # -----------------------------------------------------------------------
    # Text summary
    # -----------------------------------------------------------------------
    summary_lines = [
        "=" * 70,
        "CUSSP CASSM Phase 0 Diagnostic — DM*→TS large-dt failure analysis",
        "=" * 70,
        f"Bundle:       {bundle_path}",
        f"Config:       {config_path}",
        f"Epoch range:  [{e0}, {e1}) of {n_epochs} total",
        f"DM*→TS pairs: {n_dm_pairs}  (sources {dm_src_idx.tolist()}, rec ≥48)",
        f"Accept bound: ±{accept_bound_us:.0f} µs",
        "",
        "A. EPOCH-TO-EPOCH DT JUMP DISTRIBUTION",
        f"   Total finite jumps:   {len(dt_jumps):,}",
        f"   |jump| > 80 µs:       {int(np.sum(np.abs(dt_jumps) > 80)):,}  "
        f"({100*np.mean(np.abs(dt_jumps) > 80):.1f}%)  ← cycle-skip indicator",
        f"   |jump| 80–120 µs:     {int(np.sum((np.abs(dt_jumps) > 80) & (np.abs(dt_jumps) < 120))):,}"
        "  (±100 µs cluster = cycle-skip)",
        "",
        "B. NaN/REJECT CHARACTERISATION",
        f"   Total cells:          {n_total:,}",
        f"   NaN (rejected):       {n_nan:,}  ({100*n_nan/max(n_total,1):.1f}%)",
        f"   Edge-hit flag set:    {n_edge:,}",
        f"   Of NaN — low cc<0.3:  {n_low_cc_reject:,}  ← decorrelation mode",
        f"   Of NaN — cc≥0.3:      {n_high_cc_reject:,}  ← likely lag-clipping mode",
        "",
        "C. ACCEPT-BOUND PILE-UP (±3 µs of bound)",
        f"   Near +{accept_bound_us:.0f} µs:          {n_near_pos:,}",
        f"   Near -{accept_bound_us:.0f} µs:          {n_near_neg:,}",
        "   (large counts → legitimate large dt clipped by xcorr gate)",
        "",
    ]

    # envelope availability
    env_avail = np.sum(np.isfinite(env_flat)) > 0
    if env_avail:
        env_vs_dt_mask = np.isfinite(dt_flat) & np.isfinite(env_flat)
        if env_vs_dt_mask.sum() > 10:
            corr_env_dt = float(np.corrcoef(
                dt_flat[env_vs_dt_mask], env_flat[env_vs_dt_mask]
            )[0, 1])
            summary_lines += [
                "D. ENVELOPE_LAG_US vs DT_US",
                f"   Pearson r:            {corr_env_dt:.3f}  (1.0 = perfect agreement)",
                "   (low r = envelope and xcorr tracking different things → DTW can resolve)",
                "",
            ]
        else:
            summary_lines += ["D. ENVELOPE_LAG_US vs DT_US: insufficient overlap", ""]
    else:
        summary_lines += ["D. ENVELOPE_LAG_US: not available in bundle (env_guide off)", ""]

    # Conclusion
    summary_lines += [
        "E. SPECTRAL ANALYSIS",
        "   (see figure for sub-5 kHz energy check)",
        "",
        "FAILURE MODE ASSESSMENT:",
    ]
    if int(np.sum(np.abs(dt_jumps) > 80)) > 10:
        summary_lines.append("  [CYCLE-SKIP]  dt jumps >80 µs detected — DTW needed.")
    if n_low_cc_reject > n_high_cc_reject:
        summary_lines.append("  [DECORRELATION]  NaN epochs have low cc — DTW NCC approach needed.")
    if n_near_pos + n_near_neg > 50:
        summary_lines.append(
            f"  [CLIPPING]  {n_near_pos+n_near_neg} epochs pile up at ±{accept_bound_us:.0f} µs bound "
            "— widen xcorr_accept_max_lag_dm_hydro_ms."
        )
    summary_lines.append("=" * 70)

    summary_str = "\n".join(summary_lines)
    LOG.info("\n%s", summary_str)

    # -----------------------------------------------------------------------
    # Write text summary
    # -----------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    txt_path = out_dir / f"phase0_diag_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(summary_str + "\n")
    LOG.info("Summary written: %s", txt_path)

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        LOG.warning("matplotlib not available — skipping figure output")
        return

    has_spectrum = (
        h5_path is not None and h5_path.exists() and n_dm_pairs > 0
    )

    n_rows = 3
    n_cols = 2
    if has_spectrum:
        n_rows = 4
    fig = plt.figure(figsize=(14, n_rows * 4))
    gs  = GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.35)

    # -- Panel A: epoch-to-epoch dt jump histogram --
    ax_a = fig.add_subplot(gs[0, 0])
    if dt_jumps.size > 0:
        bins_a = np.arange(-500, 501, 5)
        ax_a.hist(dt_jumps, bins=bins_a, color="steelblue", edgecolor="none", alpha=0.8)
        for v in [-100, 100]:
            ax_a.axvline(v, color="tomato", lw=1.2, ls="--", label=f"{v:+d} µs")
        ax_a.set_xlabel("Epoch-to-epoch Δdt (µs)")
        ax_a.set_ylabel("Count")
        ax_a.set_title("A. Δdt histogram — cycle-skip at ±100 µs")
        ax_a.legend(fontsize=8)
        ax_a.set_xlim(-500, 500)

    # -- Panel B: xcorr_peak_cc distribution at NaN vs accepted epochs --
    ax_b = fig.add_subplot(gs[0, 1])
    bins_cc = np.linspace(0, 1, 41)
    cc_accepted = cc_flat[np.isfinite(dt_flat) & np.isfinite(cc_flat)]
    if cc_accepted.size > 0:
        ax_b.hist(cc_accepted, bins=bins_cc, alpha=0.7, color="seagreen",
                  label=f"Accepted ({len(cc_accepted):,})", density=True)
    if cc_at_nan.size > 0:
        ax_b.hist(cc_at_nan, bins=bins_cc, alpha=0.7, color="tomato",
                  label=f"Rejected/NaN ({len(cc_at_nan):,})", density=True)
    ax_b.axvline(0.3, color="k", lw=1, ls="--", label="cc=0.3 threshold")
    ax_b.set_xlabel("xcorr_peak_cc")
    ax_b.set_ylabel("Density")
    ax_b.set_title("B. cc distribution — decorr: rejected cc<<accepted")
    ax_b.legend(fontsize=8)

    # -- Panel C: dt_us distribution with accept-bound markers --
    ax_c = fig.add_subplot(gs[1, 0])
    if dt_accepted.size > 0:
        bins_dt = np.arange(-300, 301, 5)
        ax_c.hist(dt_accepted, bins=bins_dt, color="mediumpurple", edgecolor="none", alpha=0.8)
        for v in [-accept_bound_us, accept_bound_us]:
            ax_c.axvline(v, color="tomato", lw=1.5, ls="--",
                         label=f"bound ±{accept_bound_us:.0f} µs")
        ax_c.set_xlabel("dt_us (µs)")
        ax_c.set_ylabel("Count")
        ax_c.set_title("C. dt_us distribution — pile-up at bound = clipping")
        handles, labels = ax_c.get_legend_handles_labels()
        ax_c.legend(handles[:1], labels[:1], fontsize=8)
        ax_c.set_xlim(-350, 350)

    # -- Panel D: dt_us vs envelope_lag_us scatter --
    ax_d = fig.add_subplot(gs[1, 1])
    if env_avail:
        mask_d = np.isfinite(dt_flat) & np.isfinite(env_flat)
        if mask_d.sum() > 10:
            subsample = max(1, mask_d.sum() // 4000)
            idx = np.where(mask_d)[0][::subsample]
            ax_d.scatter(env_flat[idx], dt_flat[idx], s=2, alpha=0.4, color="steelblue")
            lim = max(np.nanpercentile(np.abs(dt_flat[mask_d]), 99),
                      np.nanpercentile(np.abs(env_flat[mask_d]), 99))
            lim = min(lim + 20, 400)
            ax_d.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="1:1")
            ax_d.set_xlim(-lim, lim)
            ax_d.set_ylim(-lim, lim)
            ax_d.set_xlabel("envelope_lag_us (µs)")
            ax_d.set_ylabel("dt_us (µs)")
            ax_d.set_title("D. dt_us vs envelope — divergence = cycle-skip")
            ax_d.legend(fontsize=8)
    else:
        ax_d.text(0.5, 0.5, "Envelope guidance\nnot available",
                  ha="center", va="center", transform=ax_d.transAxes, color="grey")
        ax_d.set_title("D. dt_us vs envelope_lag_us")

    # -- Panel E: DM*→TS dt_us time series (median across pairs) --
    ax_e = fig.add_subplot(gs[2, :])
    if n_dm_pairs > 0:
        median_dt = np.nanmedian(dt_dm, axis=0)   # (n_ep,)
        p10 = np.nanpercentile(dt_dm, 10, axis=0)
        p90 = np.nanpercentile(dt_dm, 90, axis=0)
        ep_x = np.arange(e0, e1)
        ax_e.fill_between(ep_x, p10, p90, alpha=0.25, color="steelblue", label="10–90 pctile")
        ax_e.plot(ep_x, median_dt, color="steelblue", lw=1.2, label="Median")
        ax_e.axhline(accept_bound_us, color="tomato", lw=1, ls="--",
                     label=f"±{accept_bound_us:.0f} µs bound")
        ax_e.axhline(-accept_bound_us, color="tomato", lw=1, ls="--")
        ax_e.axhline(0, color="k", lw=0.5, ls=":")
        # NaN fraction per epoch (right axis)
        ax_e2 = ax_e.twinx()
        nan_frac = np.mean(~np.isfinite(dt_dm), axis=0)
        ax_e2.plot(ep_x, nan_frac, color="tomato", lw=0.8, alpha=0.7, label="NaN frac")
        ax_e2.set_ylabel("NaN fraction", color="tomato")
        ax_e2.set_ylim(0, 1)
        ax_e2.tick_params(axis="y", labelcolor="tomato")
        ax_e.set_xlabel("Epoch index")
        ax_e.set_ylabel("dt_us (µs)")
        ax_e.set_title("E. DM*→TS median dt_us over time (red = NaN fraction)")
        ax_e.legend(loc="upper left", fontsize=8)

    # -- Panel F: Amplitude spectra (requires HDF5) --
    if has_spectrum and n_rows > 3:
        ax_f = fig.add_subplot(gs[3, :])
        _add_spectrum_panel(
            ax=ax_f,
            h5_path=h5_path,
            dm_ts_idx=dm_ts_idx,
            n_receivers=n_receivers,
            dm_src_idx=dm_src_idx,
            src_boreholes=src_bh,
            sample_rate_hz=sample_rate,
            n_pairs_plot=n_spectrum_pairs,
            hydro_flo=cfg.get("hydro_filter_low_hz"),
            hydro_fhi=cfg.get("hydro_filter_high_hz"),
            epoch_slice=epoch_slice,
        )

    fig.suptitle(
        f"Phase 0 Diagnostic — DM*→TS Failure Modes  "
        f"(epochs {e0}–{e1-1}, {n_dm_pairs} pairs)",
        fontsize=13, y=0.995,
    )
    fig_path = out_dir / f"phase0_diag_{ts}.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Figure written: %s", fig_path)


def _add_spectrum_panel(
    ax,
    h5_path: Path,
    dm_ts_idx: np.ndarray,
    n_receivers: int,
    dm_src_idx: np.ndarray,
    src_boreholes: List[str],
    sample_rate_hz: float,
    n_pairs_plot: int,
    hydro_flo: Optional[float],
    hydro_fhi: Optional[float],
    epoch_slice: slice,
) -> None:
    """Load a few DM→TS pairs from HDF5 and plot their median spectra."""
    try:
        import h5py
    except ImportError:
        ax.text(0.5, 0.5, "h5py not available — spectrum skipped",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("F. Amplitude spectra (h5py missing)")
        return

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    n_plotted = 0
    with h5py.File(h5_path, "r") as f:
        vpi = f["valid_pair_indices"][:] if "valid_pair_indices" in f else None
        n_src_h5  = int(f.attrs["n_sources"])
        n_rec_h5  = int(f.attrs["n_receivers"])
        srate     = float(f.attrs["sample_rate_hz"])
        data_ds   = f["data"]   # (n_epochs, n_compact_pairs, n_samples)
        n_ep_h5   = data_ds.shape[0]
        ep_lo = max(epoch_slice.start or 0, 0)
        ep_hi = min(epoch_slice.stop  or n_ep_h5, n_ep_h5)
        ep_lo = max(0, ep_lo)
        ep_hi = min(n_ep_h5, ep_hi)

        # Inverse lookup for compact → full pair index
        inv_vpi = None
        if vpi is not None:
            inv_vpi = np.full(n_src_h5 * n_rec_h5, -1, dtype=np.int32)
            for ci, fi in enumerate(vpi):
                inv_vpi[fi] = ci

        for i, full_pair_idx in enumerate(dm_ts_idx[:n_pairs_plot * 4]):
            if n_plotted >= n_pairs_plot:
                break
            src_idx = int(full_pair_idx) // n_receivers
            rec_idx = int(full_pair_idx) % n_receivers
            if vpi is not None:
                if inv_vpi is None or inv_vpi[full_pair_idx] < 0:
                    continue
                compact_idx = int(inv_vpi[full_pair_idx])
            else:
                compact_idx = full_pair_idx

            try:
                wfms = data_ds[ep_lo:ep_hi, compact_idx, :]  # (n_epochs, n_samp)
            except Exception:
                continue

            if wfms.ndim != 2 or wfms.shape[0] == 0:
                continue

            freqs, db = _median_spectrum_db(wfms.astype(np.float64), srate)
            if freqs.size == 0:
                continue

            bh = src_boreholes[src_idx] if src_idx < len(src_boreholes) else f"Src{src_idx}"
            label = f"{bh}→ch{rec_idx+1}"
            ax.plot(freqs / 1000.0, db, lw=0.9, alpha=0.8,
                    color=cmap(n_plotted % 10), label=label)
            n_plotted += 1

    # Mark filter band and 5 kHz threshold
    ax.axvline(5.0, color="tomato", lw=1.2, ls="--", label="5 kHz threshold")
    if hydro_flo:
        ax.axvline(hydro_flo / 1000.0, color="grey", lw=1, ls=":", label=f"flo={hydro_flo/1e3:.0f} kHz")
    if hydro_fhi:
        ax.axvline(hydro_fhi / 1000.0, color="grey", lw=1, ls="-.", label=f"fhi={hydro_fhi/1e3:.0f} kHz")
    ax.axhline(-20, color="k", lw=0.5, ls="--", alpha=0.4, label="-20 dB")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Relative amplitude (dB)")
    ax.set_title(
        "F. DM*→TS waveform spectra — energy below 5 kHz gates Phase 4 (low-freq unwrapping)"
    )
    ax.legend(fontsize=7, ncol=3)
    ax.set_ylim(-60, 3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_epoch_range(s: str) -> Tuple[int, int]:
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("epoch-range must be 'start:stop'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 0: DM*→TS large-dt failure diagnostic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",   required=True,  type=Path,
                   help="YAML config (cussp_cassm_config.yaml)")
    p.add_argument("--bundle",   type=Path, default=None,
                   help="NPZ dashboard bundle (default: from config)")
    p.add_argument("--h5",       type=Path, default=None,
                   help="HDF5 waveform cache for spectrum panel (default: from config)")
    p.add_argument("--out-dir",  type=Path, default=Path("."),
                   help="Directory for output figure and text summary")
    p.add_argument("--epoch-range", type=_parse_epoch_range, default=None,
                   metavar="START:STOP",
                   help="Restrict to epochs [START, STOP); e.g. 100:300")
    p.add_argument("--n-spectra", type=int, default=5,
                   help="Number of DM→TS pairs to plot spectra for")
    args = p.parse_args()

    if args.bundle is None:
        # Will be resolved from config inside run_phase0
        args.bundle = Path("/nonexistent")

    run_phase0(
        bundle_path    = args.bundle,
        config_path    = args.config,
        h5_path        = args.h5,
        out_dir        = args.out_dir,
        epoch_range    = args.epoch_range,
        n_spectrum_pairs = args.n_spectra,
    )


if __name__ == "__main__":
    main()
