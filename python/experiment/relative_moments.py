"""
Functions to calculate magnitudes for families of near-repeating earthquakes
using singular value decomposition. Magnitudes are relative to the template
(event with a local magnitude).

Original author: Calum Chamberlain (28 October 2015) using EQcorrscan.
Python 3 cleanup and refactor.
"""

from __future__ import annotations

import csv
import glob
import functools
import multiprocessing
import os
import pickle
from types import SimpleNamespace
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FutureTimeoutError
from pathlib import Path
from threading import Lock
import time
from typing import List, Optional, Sequence, Tuple

_DAY_LOAD_FAILED = object()   # sentinel stored in raw cache when a day load fails
_MERGE_TIMEOUT_S = 120.0


class DayStreamMergeTimeout(RuntimeError):
    """Raised when st.merge() for a daylong stream exceeds _MERGE_TIMEOUT_S."""

import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, Stream, UTCDateTime
from obspy import Trace

# After this epoch the GP* channels are assigned to station 'HITP2' (4 kHz);
# GK1/GK2 remain on 'HITP'.  Templates were built with station='HITP' throughout.
_HITP2_EPOCH = UTCDateTime("2025-10-23T22:38:19.859000Z")
from obspy.core.event import Magnitude
from obspy.clients.fdsn import Client
from scipy.signal import detrend
from scipy.ndimage import median_filter

from eqcorrscan.utils import clustering, stacking
from eqcorrscan.utils.correlate import get_array_xcorr
from eqcorrscan.utils.pre_processing import multi_process
from eqcorrscan.utils.mag_calc import svd_moments, relative_magnitude as _eqc_relative_magnitude
from eqcorrscan.core.match_filter import match_filter


logger = logging.getLogger(__name__)

_LOG_COUNTERS = {}
_LOG_COUNTERS_LOCK = Lock()
_LOG_THROTTLE_N = 500

_shared_st = None


def _init_worker(stream_data):
    """Initializer for each worker process in the pool."""
    global _shared_st
    _shared_st = stream_data


def _get_geophone_amplitude(det, snippet_before_sec, snippet_after_sec, geophone_chans):
    """Worker function to get the max amplitude on geophone channels for a detection."""
    global _shared_st
    snippet = _shared_st.slice(det.detect_time - snippet_before_sec, det.detect_time + snippet_after_sec)

    if any(len(snippet.select(channel=ch)) == 0 for ch in geophone_chans):
        return (det, np.inf)

    max_amp = 0
    for ch in geophone_chans:
        tr = snippet.select(channel=ch)[0]
        max_amp = max(max_amp, np.max(np.abs(tr.data)))

    return (det, max_amp)


def remove_offsets_rolling_median(tr, window_sec=60.0):
    """
    Fast rolling-median baseline removal using SciPy.
    """

    fs = tr.stats.sampling_rate
    win = max(3, int(window_sec * fs))
    if win % 2 == 0:
        win += 1  # median_filter prefers odd window sizes

    data = tr.data.astype(np.float64, copy=False)
    baseline = median_filter(data, size=win, mode='nearest')
    tr.data = data - baseline


def remove_HITP_spikes(
    stream,
    spike_template_path,
    geophone_chans=['GPZ', 'GP1', 'GP2'],
    ccth=0.90,
    num_quiet_spikes=36,
    min_quiet_spikes=10,
    low_freq_cutoff=2.0,
    plot=True,
    plot_output_dir='.',
    chunk_start=None,
    apply_offset_removal=True,):
    """
    Removes electronic spike noise from an ObsPy Stream from station HITP in-place.

    Args:
        stream (obspy.Stream):
            The Stream object to be modified.
        spike_template_path (str or list of str):
            Path(s) to spike template file(s). If list, templates are applied sequentially.
        geophone_chans (list, optional):
            A list of the geophone channel names to denoise.
            Defaults to ['GPZ', 'GP1', 'GP2'].
        ccth (float, optional):
            Cross-correlation threshold for spike detection. Defaults to 0.90.
        num_quiet_spikes (int, optional):
            Number of quietest spikes for the median transfer function. Defaults to 36.
        min_quiet_spikes (int, optional):
            Minimum number of quiet spikes required to proceed with denoising. Defaults to 10.
        low_freq_cutoff (float, optional):
            Frequency (Hz) to set the transfer function to zero below. Defaults to 2.0.
        plot (bool, optional):
            If True, generate and save diagnostic plots. Defaults to False.
        plot_output_dir (str, optional):
            Directory to save plots to. Defaults to current directory ('.').

    Returns:
        None: The stream object is modified in-place.
    """
    # 1. VALIDATE INPUTS AND SET UP
    ref_chan = 'GK1'
    all_chans = [ref_chan] + geophone_chans
    station = "HITP"
    if chunk_start > UTCDateTime("2025-10-23T22:38:19.859000Z"):
        station = "HITP2"

    if not stream:
        warnings.warn("Input stream is empty. Nothing to do.")
        return

    if ref_chan not in [tr.stats.channel for tr in stream]:
        raise ValueError(f"Required reference channel '{ref_chan}' not found in stream.")

    # Normalize templates input to list
    if isinstance(spike_template_path, str):
        spike_template_paths = [spike_template_path]
    else:
        spike_template_paths = list(spike_template_path)

    original_stream_for_plotting = stream.copy() if plot else None
    denoised_snapshots = []  # snapshots after each template (for comparison plot)

    if plot:
        if not os.path.exists(plot_output_dir):
            os.makedirs(plot_output_dir)
        timestring = stream[0].stats.starttime.strftime("%Y%m%dT%H%M%S")
        chunk_timestring = chunk_start.strftime("%Y%m%dT%H%M%S") if chunk_start else "unknown_chunk"

    if apply_offset_removal:
        logger.debug("Removing slow offsets using rolling median...")
        for tr in stream:
            if tr.stats.channel == ref_chan:
                remove_offsets_rolling_median(tr, window_sec=60.0)

    # Process each template sequentially (steps 3-6 per template)
    for template_idx, tpl_path in enumerate(spike_template_paths, start=1):
        # 2. LOAD TEMPLATE
        try:
            template_data = np.loadtxt(tpl_path).flatten()
            if template_idx == 2:
                template_data *= -1.
            template_st = Stream(Trace(
                data=template_data,
                header={
                    'network': stream[0].stats.network,
                    'station': station,
                    'channel': ref_chan,
                    'sampling_rate': stream[0].stats.sampling_rate
                }
            ))
        except Exception as e:
            raise IOError(f"Error reading template file: {e}")

        # 3. DETECT SPIKES
        logger.debug("Running matched-filter detection on %s (template %s)...", ref_chan, template_idx)
        detections = match_filter(
            template_list=[template_st],
            template_names=['spike'],
            st=stream.select(channel=ref_chan),
            threshold=ccth,
            threshold_type='absolute',
            trig_int=0.5,
            plot=False,
            plotdir=plot_output_dir,
        )
        # Filter for only positive detections (why are negative detections present?)
        detections = [det for det in detections if det.detect_val > 0]

        logger.debug("Found %s initial detections for template %s.", len(detections), template_idx)
        if not detections:
            warnings.warn(f"No detections found for template {template_idx}. Stream will not be modified for this template.")
            continue

        # 4. QC: FIND QUIETEST SPIKES (PARALLELIZED)
        snippet_before_sec=0.1
        snippet_after_sec=0.4
        cpu_cores = os.cpu_count()
        logger.debug(
            "Finding quietest spikes in parallel using %s cores (template %s)...",
            cpu_cores,
            template_idx,
        )
        worker = functools.partial(_get_geophone_amplitude, snippet_before_sec=snippet_before_sec, snippet_after_sec=snippet_after_sec, geophone_chans=geophone_chans)

        with multiprocessing.Pool(initializer=_init_worker, initargs=(stream,)) as pool:
            detection_amplitudes = pool.map(worker, detections)

        valid_detections = [item for item in detection_amplitudes if np.isfinite(item[1])]
        sorted_detections = sorted(valid_detections, key=lambda x: x[1])

        num_to_select = min(num_quiet_spikes, len(sorted_detections))

        # MODIFIED: Add check for minimum number of spikes
        if num_to_select < min_quiet_spikes:
            warnings.warn(
                f"Found only {num_to_select} quiet spikes, which is fewer than the minimum "
                f"of {min_quiet_spikes} required. Aborting denoising for this segment (template {template_idx})."
            )
            continue

        logger.debug(
            "Selected %s quietest spikes for TF calculation (template %s).",
            num_to_select,
            template_idx,
        )
        quiet_detections = [item[0] for item in sorted_detections[:num_to_select]]

        # 5. GATHER SNIPPETS AND CALCULATE MEDIAN TRANSFER FUNCTION
        quiet_snippets = []
        for det in quiet_detections:
            snippet = stream.slice(det.detect_time - snippet_before_sec, det.detect_time + snippet_after_sec).copy()
            n1_before_samples = int(snippet_before_sec * snippet[0].stats.sampling_rate)
            for tr in snippet:
                tr.data = detrend(tr.data)
                tr.data -= np.mean(tr.data[:n1_before_samples - 10])
            quiet_snippets.append(snippet)


        # --- PLOTTING BLOCK 1: QUIET SPIKES (per template) ---
        if plot:
            logger.debug("Plotting spike stack for quality control (template %s)...", template_idx)
            fig_spikes, axes_spikes = plt.subplots(len(all_chans), 1, figsize=(10, 8), sharex=True, sharey=True)
            fig_spikes.suptitle(f'Stacked Quiet Spikes for Station {station} (Template {template_idx})', fontsize=16)
            for i, ch in enumerate(all_chans):
                ax = axes_spikes if len(all_chans) == 1 else axes_spikes[i]
                for snip in quiet_snippets:
                    if ch not in [tr.stats.channel for tr in snip]:
                        continue
                    trace = snip.select(channel=ch)[0]
                    time_axis = trace.times() - snippet_before_sec
                    ax.plot(time_axis, trace.data, 'k-', alpha=0.2)
                ax.set_ylabel(ch)
                ax.grid(True)
            (axes_spikes if len(all_chans) == 1 else axes_spikes[-1]).set_xlabel("Time relative to detection (s)")
            savename_spikes = os.path.join(
                plot_output_dir,
                f"fig_spike_stack_{station}_{chunk_timestring}_template{template_idx}.jpg"
            )
            plt.savefig(savename_spikes, dpi=160)
            logger.debug("Saved spike stack plot to %s", savename_spikes)
            plt.close()

        fft_starttime_offset_sec = 0.2
        total_snip_sec = snippet_before_sec + snippet_after_sec
        max_fft_window = total_snip_sec - fft_starttime_offset_sec
        if max_fft_window <= 0:
            warnings.warn("Snippet too short for FFT window; skipping template.")
            continue
        fft_window_sec = min(0.5, max_fft_window)  # cap at 0.5 s, but fit within snippet
        fs1 = quiet_snippets[0][0].stats.sampling_rate
        winlen_samples = int(fft_window_sec * fs1)
        freq = np.fft.rfftfreq(winlen_samples, d=1 / fs1)

        individual_tfs = {ch: [] for ch in geophone_chans}
        for snip in quiet_snippets:
            starttime_cut = snip[0].stats.starttime + fft_starttime_offset_sec
            endtime_cut = starttime_cut + winlen_samples / fs1
            ref_trace_cut = snip.select(channel=ref_chan)[0].copy().trim(starttime_cut, endtime_cut)

            # Skip snippets that are too short for the fft window
            if ref_trace_cut.stats.npts < winlen_samples:
                continue

            ref_trace_cut.taper(0.04)
            fft_ref = np.fft.rfft(ref_trace_cut.data)

            for ch in geophone_chans:
                if ch not in [tr.stats.channel for tr in snip]:
                    continue
                geo_trace_cut = snip.select(channel=ch)[0].copy().trim(starttime_cut, endtime_cut)

                if geo_trace_cut.stats.npts < winlen_samples:
                    continue

                geo_trace_cut.taper(0.04)
                fft_geo = np.fft.rfft(geo_trace_cut.data)

                # Enforce consistent TF length
                if fft_geo.shape != fft_ref.shape:
                    continue

                tf = np.divide(fft_geo, fft_ref, out=np.zeros_like(fft_geo), where=fft_ref!=0)
                individual_tfs[ch].append(tf)


        transfer_functions = {}
        for ch in geophone_chans:
            if not individual_tfs[ch]:
                continue
            tfs_stack = np.array(individual_tfs[ch])
            median_real = np.median(np.real(tfs_stack), axis=0)
            median_imag = np.median(np.imag(tfs_stack), axis=0)
            median_tf = median_real + 1j * median_imag

            # Use fixed freq derived from winlen_samples
            median_tf[freq <= low_freq_cutoff] = 0 + 0j
            transfer_functions[ch] = median_tf

        # --- PLOTTING BLOCK 2: TRANSFER FUNCTIONS (per template) ---
        if plot:
            logger.debug("Plotting final median transfer functions (template %s)...", template_idx)
            plt.figure(figsize=(10, 8))
            for ch in geophone_chans:
                if ch not in transfer_functions:
                    continue
                plt.subplot(2, 1, 1)
                plt.plot(freq, np.real(transfer_functions[ch]), label=f"Real({ch}/{ref_chan})", alpha=0.5)
                plt.title('Median Transfer Functions (Real Part)'); plt.grid(True); plt.legend()
                plt.subplot(2, 1, 2)
                plt.plot(freq, np.imag(transfer_functions[ch]), label=f"Imag({ch}/{ref_chan})", alpha=0.5)
                plt.title('Median Transfer Functions (Imaginary Part)'); plt.xlabel("Frequency (Hz)"); plt.grid(True); plt.legend()
            savename_tf = os.path.join(
                plot_output_dir,
                f"fig_transfer_functions_{station}_{chunk_timestring}_template{template_idx}.jpg"
            )
            plt.savefig(savename_tf, dpi=160)
            logger.debug("Saved transfer function plot to %s", savename_tf)
            plt.close()

        # 6. DENOISE THE FULL STREAM (IN-PLACE)
        logger.debug("Denoising the full time series in-place (template %s)...", template_idx)
        ref_trace_full = stream.select(channel=ref_chan)[0]
        fft_ref_full = np.fft.rfft(ref_trace_full.data)
        full_freqs = np.fft.rfftfreq(ref_trace_full.stats.npts, d=ref_trace_full.stats.delta)

        for ch in geophone_chans:
            if ch not in transfer_functions:
                continue
            trace_to_denoise = stream.select(channel=ch)[0]
            if trace_to_denoise.stats.npts != ref_trace_full.stats.npts:
                warnings.warn(f"Channel {ch} has a different length than reference. Skipping.")
                continue

            tf_interpolated = np.interp(full_freqs, freq, transfer_functions[ch], left=0, right=0)
            fft_predicted_noise = fft_ref_full * tf_interpolated
            predicted_noise_time = np.fft.irfft(fft_predicted_noise, n=ref_trace_full.stats.npts)

            trace_to_denoise.data = trace_to_denoise.data.astype(np.float64)
            trace_to_denoise.data -= predicted_noise_time

        if plot:
            denoised_snapshots.append(stream.copy())

    # --- PLOTTING BLOCK 3: COMPARISON PLOT (once after all templates) ---
    if plot:
        logger.debug("Plotting detailed before-and-after comparison (final)...")
        wide_plot_start_offset_sec = 500
        wide_plot_duration_sec = 30
        zoom_plot_start_offset_sec = 20
        zoom_plot_duration_sec = 3

        if original_stream_for_plotting and original_stream_for_plotting[0].stats.npts / original_stream_for_plotting[0].stats.sampling_rate > wide_plot_start_offset_sec + wide_plot_duration_sec:
            logger.debug("Denoising comparison plot: start")
            base_time = original_stream_for_plotting[0].stats.starttime
            wide_start_time = base_time + wide_plot_start_offset_sec
            wide_end_time = wide_start_time + wide_plot_duration_sec
            zoom_start_time = wide_start_time + zoom_plot_start_offset_sec
            zoom_end_time = zoom_start_time + zoom_plot_duration_sec

            fig, axes = plt.subplots(len(geophone_chans), 2, figsize=(20, 12), squeeze=False)
            fig.suptitle(f'Denoising Comparison for {station}: Wide and Zoomed Views', fontsize=16)
            logger.debug("Denoising comparison plot: layout")
            for i, ch in enumerate(geophone_chans):
                ax_wide = axes[i, 0]
                ax_zoom = axes[i, 1]
                logger.debug("Denoising comparison plot channel %s", ch)
                original_tr_wide = original_stream_for_plotting.select(channel=ch)[0].copy().trim(wide_start_time, wide_end_time)
                original_tr_zoom = original_tr_wide.copy().trim(zoom_start_time, zoom_end_time)

                time_axis_wide = original_tr_wide.times("matplotlib")
                time_axis_zoom = original_tr_zoom.times("matplotlib")

                ax_wide.plot(time_axis_wide, original_tr_wide.data, 'k-', linewidth=0.5, alpha=0.7, label='Raw')
                logger.debug("Denoising comparison plot: wide")
                if len(denoised_snapshots) >= 1:
                    t1_tr_wide = denoised_snapshots[0].select(channel=ch)[0].copy().trim(wide_start_time, wide_end_time)
                    ax_wide.plot(time_axis_wide, t1_tr_wide.data, 'r-', linewidth=0.8, alpha=0.8, label='Denoised (Template 1)')

                if len(denoised_snapshots) >= 2:
                    tn_tr_wide = denoised_snapshots[-1].select(channel=ch)[0].copy().trim(wide_start_time, wide_end_time)
                    ax_wide.plot(time_axis_wide, tn_tr_wide.data, 'b-', linewidth=0.8, alpha=0.8, label=f'Denoised (Templates 1-{len(denoised_snapshots)})')

                ax_wide.set_title(f'Channel {ch} - Wide View')
                ax_wide.set_ylabel('Amplitude')
                ax_wide.grid(True)

                zoom_start_mpl = original_tr_zoom.times("matplotlib")[0]
                zoom_end_mpl = original_tr_zoom.times("matplotlib")[-1]
                ax_wide.axvspan(zoom_start_mpl, zoom_end_mpl, color='blue', alpha=0.2, label='Zoom Area')
                ax_wide.legend()

                ax_zoom.plot(time_axis_zoom, original_tr_zoom.data, 'k-', linewidth=0.5, alpha=0.7, label='Raw')
                logger.debug("Denoising comparison plot: zoom")
                if len(denoised_snapshots) >= 1:
                    t1_tr_zoom = denoised_snapshots[0].select(channel=ch)[0].copy().trim(zoom_start_time, zoom_end_time)
                    ax_zoom.plot(time_axis_zoom, t1_tr_zoom.data, 'r-', linewidth=0.8, alpha=0.8, label='Denoised (Template 1)')

                if len(denoised_snapshots) >= 2:
                    tn_tr_zoom = denoised_snapshots[-1].select(channel=ch)[0].copy().trim(zoom_start_time, zoom_end_time)
                    ax_zoom.plot(time_axis_zoom, tn_tr_zoom.data, 'b-', linewidth=0.8, alpha=0.8, label=f'Denoised (Templates 1-{len(denoised_snapshots)})')

                ax_zoom.set_title('Zoomed View')
                ax_zoom.grid(True)
                ax_zoom.legend()

                ax_wide.xaxis_date()
                ax_zoom.xaxis_date()
            logger.debug("Denoising comparison plot: finalize")
            axes[-1, 0].set_xlabel('Time'); axes[-1, 1].set_xlabel('Time')
            fig.autofmt_xdate()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            savename_comp = os.path.join(plot_output_dir, f"fig_denoising_comparison_detailed_{station}_{chunk_timestring}.jpg")
            plt.savefig(savename_comp, dpi=160)
            logger.debug("Saved detailed comparison plot to %s", savename_comp)


def _log_every(key: str, level: int, msg: str, *args, every: int = _LOG_THROTTLE_N) -> None:
    with _LOG_COUNTERS_LOCK:
        count = _LOG_COUNTERS.get(key, 0) + 1
        _LOG_COUNTERS[key] = count
    if count == 1 or count % every == 0:
        logger.log(level, msg, *args)


def _plot_gutenberg_richter(
    magnitudes: Sequence[float],
    out_path: Path,
    title: str,
    bin_width: float = 0.1,
    extra_series: Optional[dict] = None,
    template_mags=None,
    self_detect_mags=None,
) -> None:
    """Plot a cumulative Gutenberg-Richter diagram.

    Parameters
    ----------
    magnitudes:
        Primary magnitude series (SVD ML). Plotted in black with b-value fit.
    extra_series:
        Optional dict of ``{label: [magnitudes]}`` for additional series
        (e.g. pairwise estimates).  Each is plotted without a b-value fit.
    template_mags:
        Single float or list of floats — template magnitude(s) shown as
        a vertical dashed gold line with text annotation.
    self_detect_mags:
        List of floats — magnitudes of self-detections (detect_time ≈ template
        event time), shown as gold star markers on the cumulative curve.
    """
    import matplotlib.transforms as _mpl_transforms
    _SERIES_COLORS = ["steelblue", "darkorange", "forestgreen", "firebrick"]

    mags = np.asarray([m for m in magnitudes if m is not None and np.isfinite(m)], dtype=float)
    if mags.size == 0:
        return
    mags = np.sort(mags)
    unique_mags = np.unique(mags)
    if unique_mags.size == 0:
        return
    if bin_width <= 0:
        bin_width = 0.1

    min_mag = np.min(mags)
    max_mag = np.max(mags)
    bins = np.arange(min_mag, max_mag + bin_width, bin_width)
    if bins.size < 2:
        bins = np.array([min_mag, max_mag + bin_width])
    hist, edges = np.histogram(mags, bins=bins)
    max_bin = int(np.argmax(hist)) if hist.size > 0 else 0
    mc = edges[max_bin] + bin_width / 2.0

    mags_mc = mags[mags >= mc]
    counts = np.array([np.sum(mags >= m) for m in unique_mags], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(unique_mags, counts, "o-", color="black", linewidth=0.7,
            markersize=3, label="SVD ML", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Magnitude (ML)")
    ax.set_ylabel("N >= M")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")

    if mags_mc.size >= 2:
        # Least-squares regression on binned cumulative counts above Mc.
        # Treats each magnitude bin equally, avoiding bias toward
        # densely-sampled small magnitudes in the Aki MLE.
        _bin_edges = np.arange(mc, max_mag + bin_width, bin_width)
        _bin_counts = np.array([np.sum(mags >= m) for m in _bin_edges], dtype=float)
        _valid = _bin_counts > 0
        if _valid.sum() >= 2:
            _x = _bin_edges[_valid]
            _y = np.log10(_bin_counts[_valid])
            _coeffs = np.polyfit(_x, _y, 1)
            b_val = -float(_coeffs[0])
            a_val = float(_coeffs[1])
            if np.isfinite(b_val) and b_val > 0:
                m_fit = np.linspace(mc, max_mag, 50)
                y_fit = 10.0 ** (a_val - b_val * m_fit)
                ax.plot(m_fit, y_fit, "r--", linewidth=0.8,
                        label=f"SVD b={b_val:.2f}, Mc={mc:.2f}", alpha=0.5)

    if extra_series:
        for color_idx, (label, extra_mags) in enumerate(extra_series.items()):
            em = np.asarray(
                [m for m in extra_mags if m is not None and np.isfinite(m)], dtype=float
            )
            if em.size == 0:
                continue
            em = np.sort(em)
            unique_em = np.unique(em)
            counts_e = np.array([np.sum(em >= m) for m in unique_em], dtype=float)
            color = _SERIES_COLORS[color_idx % len(_SERIES_COLORS)]
            ax.plot(unique_em, counts_e, "s--", color=color,
                    linewidth=0.7, markersize=3, label=label, alpha=0.5)

    # Self-detection markers (gold stars on the cumulative curve)
    if self_detect_mags:
        _sdm = np.asarray(
            [m for m in self_detect_mags if m is not None and np.isfinite(m)], dtype=float
        )
        if _sdm.size > 0:
            _sd_counts = np.array([np.sum(mags >= m) for m in _sdm], dtype=float)
            _sd_valid = _sd_counts > 0
            if _sd_valid.any():
                ax.scatter(
                    _sdm[_sd_valid], _sd_counts[_sd_valid],
                    marker="*", s=90, color="gold", edgecolors="darkorange",
                    linewidths=0.6, zorder=5, label="Self-detect", alpha=0.9,
                )

    # Template magnitude vertical line(s) with text annotation
    if template_mags is not None:
        _tmpl_list = (
            [template_mags] if isinstance(template_mags, (int, float, np.floating))
            else list(template_mags)
        )
        _trans = _mpl_transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for _tmag in _tmpl_list:
            try:
                _tmag = float(_tmag)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(_tmag):
                continue
            ax.axvline(_tmag, color="goldenrod", linewidth=0.9, linestyle="--", alpha=0.7)
            ax.text(
                _tmag, 0.98, f"Tmpl\n{_tmag:.2f}",
                fontsize=6, color="goldenrod", ha="center", va="top",
                transform=_trans,
            )

    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_magnitude_comparison(all_results: List[dict], plot_dir: str) -> None:
    """Scatter plot comparing pairwise and SVD magnitudes across all detections.

    Two subplots: uncorrected pairwise vs SVD, and bias-corrected pairwise vs SVD.
    Points are coloured by family; a 1:1 reference line is drawn.
    """
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    families = sorted({r.get("family_name", "") for r in all_results if r.get("family_name")})
    cmap = plt.get_cmap("tab10")
    color_map = {f: cmap(i % 10) for i, f in enumerate(families)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    all_plotted: List[float] = []

    for ax, pairwise_key, label in [
        (axes[0], "ml_pairwise", "Pairwise (uncorrected)"),
        (axes[1], "ml_pairwise_corrected", "Pairwise (bias-corrected)"),
    ]:
        for fam in families:
            xs, ys = [], []
            for r in all_results:
                if r.get("family_name") != fam:
                    continue
                try:
                    _x = float(r["ml"])
                    _y = float(r[pairwise_key])
                except (TypeError, ValueError, KeyError):
                    continue
                xs.append(_x)
                ys.append(_y)
            if xs:
                ax.scatter(xs, ys, color=color_map[fam], label=fam, s=15, alpha=0.5)
                all_plotted.extend(xs)
                all_plotted.extend(ys)

        if all_plotted:
            lo = min(all_plotted) - 0.2
            hi = max(all_plotted) + 0.2
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="1:1", alpha=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_xlabel("ML (SVD)")
        ax.set_ylabel(f"ML ({label})")
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=6, ncol=2)
        all_plotted = []  # reset for next subplot

    fig.suptitle("Magnitude method comparison")
    fig.tight_layout()
    out_path = plot_path / "magnitude_comparison.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote magnitude comparison plot %s", out_path)


def _plot_self_detection_sanity(
    all_results: List[dict],
    states: List,
    plot_dir: str,
    tol_s: float = 2.0,
) -> None:
    """For self-detections (detect_time ≈ template event time), compare catalog,
    SVD, and pairwise magnitudes on a grouped bar chart.

    The expected self-detection time is derived from the template trace starttime
    plus the template prepick.
    """
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    template_times: dict = {}
    for state in states:
        prepick = float(getattr(state.template_obj, "prepick", None) or 0.0)
        template_times[state.family_name] = state.template_trace.stats.starttime + prepick

    records = []  # (family_name, template_mag, ml_svd, ml_pairwise, ml_pairwise_corrected)
    for r in all_results:
        fam = r.get("family_name", "")
        expected = template_times.get(fam)
        if expected is None:
            continue
        dt_str = r.get("detect_time", "")
        if not dt_str:
            continue
        try:
            dt = UTCDateTime(dt_str)
        except Exception:
            continue
        if abs(dt - expected) > tol_s:
            continue
        def _safe_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        records.append((
            fam,
            _safe_float(r.get("template_mag")),
            _safe_float(r.get("ml")),
            _safe_float(r.get("ml_pairwise")),
            _safe_float(r.get("ml_pairwise_corrected")),
        ))

    if not records:
        logger.info(
            "No self-detections found within %.1fs tolerance; skipping sanity plot", tol_s
        )
        return

    n = len(records)
    x = np.arange(n)
    width = 0.2

    def _bar(ax, vals, offset, color, label):
        positions = x + offset
        heights = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(positions, heights, width, label=label, color=color, alpha=0.5)
        for bar, v in zip(bars, vals):
            if v is None:
                bar.set_alpha(0.0)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.8), 5))
    _bar(ax, [r[1] for r in records], -1.5 * width, "steelblue", "Catalog ML")
    _bar(ax, [r[2] for r in records],  -0.5 * width, "darkorange", "SVD ML")
    _bar(ax, [r[3] for r in records],   0.5 * width, "forestgreen", "Pairwise ML")
    _bar(ax, [r[4] for r in records],   1.5 * width, "firebrick", "Pairwise ML (corrected)")

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in records], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ML")
    ax.set_title(f"Self-detection sanity check (tol = {tol_s:.1f} s)")
    ax.legend()
    fig.tight_layout()

    out_path = plot_path / "self_detection_sanity.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "Wrote self-detection sanity plot (%d self-detections) → %s", n, out_path
    )


def _plot_processing_debug(payload: dict, out_path: Path) -> None:
    raw = payload.get("raw_data")
    proc = payload.get("proc_data")
    filt = payload.get("filt_data")
    if raw is None or proc is None or filt is None:
        return
    raw_sr = float(payload.get("raw_sr") or 1.0)
    proc_sr = float(payload.get("proc_sr") or raw_sr)
    filt_sr = float(payload.get("filt_sr") or proc_sr)
    label = payload.get("label", "trace")
    stachan = payload.get("stachan", "")
    pick_time = payload.get("pick_time", "")
    family = payload.get("family", "")
    detection = payload.get("detection_label", "")

    def _norm(data):
        data = np.asarray(data, dtype=float)
        scale = np.max(np.abs(data))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        return data / scale

    raw_n = _norm(raw)
    proc_n = _norm(proc)
    filt_n = _norm(filt)

    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(9, 6))
    raw_t = np.arange(raw_n.size) / raw_sr
    proc_t = np.arange(proc_n.size) / proc_sr
    filt_t = np.arange(filt_n.size) / filt_sr
    axes[0].plot(raw_t, raw_n, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Raw")
    axes[1].plot(proc_t, proc_n, color="tab:blue", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Detrend/Resample")
    axes[2].plot(filt_t, filt_n, color="tab:red", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("Filter/Trim")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"{family} {detection} {label} {stachan} pick={pick_time}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_corr_debug(payload: dict, out_path: Path) -> None:
    template = payload.get("template_full")
    detection_data = payload.get("detection_data")
    if template is None or detection_data is None:
        return
    sr = float(payload.get("sr") or 1.0)
    corr_val = payload.get("corr_val")
    corr_abs = payload.get("corr_abs")
    detect_val = payload.get("detect_val")
    stachan = payload.get("stachan", "")
    event_index = payload.get("event_index", "")
    family = payload.get("family", "")
    detection_label = payload.get("detection_label", "")
    corr_series = payload.get("corr_series")
    shift_samples = payload.get("shift_samples")
    det_trace_start = payload.get("det_trace_start")
    requested_detect_time = payload.get("requested_detect_time")

    def _norm(data):
        data = np.asarray(data, dtype=float)
        scale = np.max(np.abs(data))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        return data / scale

    template_n = _norm(template)
    detection_n = _norm(detection_data)
    t_template = np.arange(template_n.size) / sr
    t_detection = np.arange(detection_n.size) / sr

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    axes[0].plot(t_template, template_n, color="black", linewidth=0.8, label="Template", alpha=0.5)
    axes[0].plot(t_detection, detection_n, color="tab:orange", linewidth=0.8, label="Detection", alpha=0.5)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Norm amp")
    if corr_series is not None:
        corr_series = np.asarray(corr_series, dtype=float)
        expected = None
        if shift_samples is not None:
            expected = int(2 * int(shift_samples) + 1)
        if expected and expected == corr_series.size:
            center = int(shift_samples)
        else:
            center = corr_series.size // 2
        lag = (np.arange(corr_series.size) - center) / sr
        axes[1].plot(lag, corr_series, color="tab:purple", linewidth=0.9, alpha=0.5)
        axes[1].axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
        if detect_val is not None:
            axes[1].axhline(detect_val, color="tab:red", linewidth=0.8, linestyle="--", label=f"detect_val={detect_val:.3f}")
            axes[1].legend(loc="upper right", fontsize=7)
        axes[1].set_ylabel("XCorr coeff")
    else:
        axes[1].text(0.5, 0.5, "No corr series", ha="center", va="center")
        axes[1].set_ylabel("XCorr coeff")
    axes[1].set_xlabel("Lag (s)")
    det_str = f" det={detect_val:.3f}" if detect_val is not None else ""
    if corr_abs is None:
        fig.suptitle(f"{family} {detection_label} {stachan} event={event_index} corr={corr_val:.3f}{det_str}")
    else:
        fig.suptitle(
            f"{family} {detection_label} {stachan} event={event_index} corr={corr_val:.3f} abs={corr_abs:.3f}{det_str}"
        )
        time_info_parts = []
        if requested_detect_time is not None:
            time_info_parts.append(f"detect_time={requested_detect_time}")
        if det_trace_start is not None:
            time_info_parts.append(f"loaded={det_trace_start}")
        if time_info_parts:
            fig.text(0.5, 0.01, "  |  ".join(time_info_parts), ha="center", va="bottom", fontsize=6.5, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def local_to_moment(mag: float, m: float = 1.0, c: float = 0.0) -> float:
    """
    Convert local magnitude to seismic moment.

    Default coefficients (m=1.0, c=0.0) reflect Mw ≈ ML for Utah
    (Whidden & Pankow 2012, SRL 83(5), Fig. 10: little systematic bias,
    σ=0.2 around the 1:1 line).  Override via ML_MW_M / ML_MW_C in config.
    """
    mw = (mag - c) / m
    return 10.0 ** (1.5 * mw + 9.0)


def _day_start(t: UTCDateTime) -> UTCDateTime:
    return UTCDateTime(t.year, t.month, t.day)


def _align_time_window(starttime: UTCDateTime, endtime: UTCDateTime, samp_rate: float) -> tuple[UTCDateTime, UTCDateTime]:
    if samp_rate <= 0:
        return starttime, endtime
    duration = float(endtime - starttime)
    target_npts = int(round(duration * samp_rate)) + 1
    aligned_end = starttime + (target_npts - 1) / samp_rate
    return starttime, aligned_end


def _day_station_token(stations: Sequence[str]) -> str:
    station_set = set(stations or [])
    if "HITP" in station_set or "HITP2" in station_set:
        return "HITP,HITP2"
    return ",".join(sorted(stations)) if stations else ""


def _day_station_list(stations: Sequence[str]) -> List[str]:
    station_set = set(stations or [])
    if "HITP" in station_set or "HITP2" in station_set:
        return ["HITP", "HITP2"]
    return sorted(stations) if stations else []


def _day_cache_path(cache_root: str, network: str, stations: Sequence[str], day: UTCDateTime) -> Path:
    station_token = _day_station_token(stations)
    fname = f"{network}.{station_token}.{day.strftime('%Y%j')}.mseed"
    return Path(cache_root) / fname


def _day_cache_path_legacy(cache_root: str, network: str, stations: Sequence[str], day: UTCDateTime) -> Path:
    """Old year-less filename, kept only for migration reads."""
    station_token = _day_station_token(stations)
    fname = f"{network}.{station_token}.{day.strftime('%j')}.mseed"
    return Path(cache_root) / fname



class DaylongStreamCache:
    def __init__(
        self,
        cache_root: str,
        client: Optional[Client],
        prefer_local: bool = True,
        *,
        denoise: bool = False,
        spike_template_path: Optional[Sequence[str]] = None,
        denoise_plot_dir: Optional[str] = None,
        denoise_chunk_s: float = 3600.0,
        denoise_geophone_chans: Optional[Sequence[str]] = None,
    ):
        self.cache_root = cache_root
        self.client = client
        self.prefer_local = prefer_local
        self.denoise = denoise
        self.spike_template_path = list(spike_template_path) if spike_template_path else []
        self.denoise_plot_dir = denoise_plot_dir
        self.denoise_chunk_s = float(denoise_chunk_s) if denoise_chunk_s else 3600.0
        self.denoise_geophone_chans = list(denoise_geophone_chans) if denoise_geophone_chans else ["GPZ"]
        self._raw_cache = {}
        self._processed_cache = {}
        self._raw_locks = {}
        self._processed_locks = {}
        self._lock_guard = Lock()
        self.warn_s = 60.0

    def _get_key_lock(self, lock_map: dict, key) -> Lock:
        with self._lock_guard:
            lock = lock_map.get(key)
            if lock is None:
                lock = Lock()
                lock_map[key] = lock
        return lock

    def _load_day_stream(self, day: UTCDateTime, network: str, stations: Sequence[str], channels: Sequence[str]) -> Stream:
        key = (day.timestamp, network, tuple(sorted(stations)), tuple(sorted(channels)))
        cached = self._raw_cache.get(key)
        if cached is _DAY_LOAD_FAILED:
            raise DayStreamMergeTimeout(
                f"Day {day.strftime('%Y-%m-%d')} {network} previously timed out; skipping"
            )
        if cached is not None:
            _log_every(
                f"day_raw_hit_{day.timestamp}_{network}",
                logging.INFO,
                "Daylong raw cache hit for %s %s",
                day.strftime("%Y-%m-%d"),
                network,
                every=1,
            )
            return cached.copy()
        lock = self._get_key_lock(self._raw_locks, key)
        with lock:
            cached = self._raw_cache.get(key)
            if cached is _DAY_LOAD_FAILED:
                raise DayStreamMergeTimeout(
                    f"Day {day.strftime('%Y-%m-%d')} {network} previously timed out; skipping"
                )
            if cached is not None:
                _log_every(
                    f"day_raw_hit_{day.timestamp}_{network}",
                    logging.INFO,
                    "Daylong raw cache hit for %s %s",
                    day.strftime("%Y-%m-%d"),
                    network,
                    every=1,
                )
                return cached.copy()
            logger.info(
                "Daylong load start for %s %s (%s stations, %s channels)",
                day.strftime("%Y-%m-%d"),
                network,
                len(stations),
                len(channels),
            )
            cache_path = _day_cache_path(self.cache_root, network, stations, day)
            legacy_path = _day_cache_path_legacy(self.cache_root, network, stations, day)
            t0 = time.monotonic()
            if self.prefer_local and legacy_path.exists():
                logger.info("Daylong load using legacy cache %s", legacy_path)
                st = read(str(legacy_path))
                logger.info("Daylong read done for %s", legacy_path)
            elif self.prefer_local and cache_path.exists():
                logger.info("Daylong load using year-qualified cache %s", cache_path)
                st = read(str(cache_path))
                logger.info("Daylong read done for %s", cache_path)
            else:
                st = None
            if st is None:
                if self.client is None:
                    raise ValueError("Client required to fetch daylong data")
                logger.info(
                    "Daylong load using client %s %s stations=%s channels=%s",
                    day.strftime("%Y-%m-%d"),
                    network,
                    ",".join(sorted(stations)),
                    ",".join(sorted(channels)),
                )
                st = self.client.get_waveforms(
                    network,
                    ",".join(sorted(stations)),
                    "*",
                    ",".join(sorted(channels)),
                    day,
                    day + 86400,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                st.write(str(cache_path), format="MSEED")
            logger.info("Daylong merge start for %s %s", day.strftime("%Y-%m-%d"), network)
            merge_start = time.monotonic()
            _merge_st = st
            # Use an executor without a context manager so we can discard the
            # stuck thread immediately on timeout instead of blocking in
            # executor.__exit__(wait=True) forever.
            _ex = ThreadPoolExecutor(max_workers=1)
            _future = _ex.submit(_merge_st.merge, fill_value="latest")
            try:
                _future.result(timeout=_MERGE_TIMEOUT_S)
            except _FutureTimeoutError:
                _ex.shutdown(wait=False)   # abandon the stuck thread
                self._raw_cache[key] = _DAY_LOAD_FAILED
                logger.error(
                    "Daylong merge timed out after %.0fs for %s %s — skipping day",
                    _MERGE_TIMEOUT_S,
                    day.strftime("%Y-%m-%d"),
                    network,
                )
                raise DayStreamMergeTimeout(
                    f"merge timed out after {_MERGE_TIMEOUT_S:.0f}s "
                    f"for {day.strftime('%Y-%m-%d')} {network}"
                )
            _ex.shutdown(wait=False)   # thread already finished; don't linger
            merge_elapsed = time.monotonic() - merge_start
            logger.info(
                "Daylong merge done for %s %s in %.1fs",
                day.strftime("%Y-%m-%d"),
                network,
                merge_elapsed,
            )
            detrend_start = time.monotonic()
            st.detrend("demean")
            detrend_elapsed = time.monotonic() - detrend_start
            logger.info(
                "Daylong detrend done for %s %s in %.1fs",
                day.strftime("%Y-%m-%d"),
                network,
                detrend_elapsed,
            )
            if any(tr.stats.station == "HITP2" for tr in st):
                logger.info("Daylong resample start for %s %s", day.strftime("%Y-%m-%d"), network)
                resample_start = time.monotonic()
                st.resample(1000.0)
                resample_elapsed = time.monotonic() - resample_start
                logger.info(
                    "Daylong resample done for %s %s in %.1fs",
                    day.strftime("%Y-%m-%d"),
                    network,
                    resample_elapsed,
                )
            if self.denoise:
                if not self.spike_template_path:
                    raise ValueError("spike_template_path required when denoise=True")
                logger.info("Daylong denoise start for %s %s", day.strftime("%Y-%m-%d"), network)
                denoise_start = time.monotonic()
                st_denoised = Stream()
                chunk_start = day
                day_end = day + 86400
                while chunk_start < day_end:
                    chunk_end = min(chunk_start + self.denoise_chunk_s, day_end)
                    chunk = st.slice(chunk_start, chunk_end)
                    remove_HITP_spikes(
                        stream=chunk,
                        spike_template_path=self.spike_template_path,
                        geophone_chans=self.denoise_geophone_chans,
                        plot=False,
                        plot_output_dir=self.denoise_plot_dir or ".",
                        chunk_start=chunk_start,
                    )
                    st_denoised += chunk
                    chunk_start = chunk_end
                st_denoised.merge(fill_value="latest")
                st = st_denoised
                denoise_elapsed = time.monotonic() - denoise_start
                logger.info(
                    "Daylong denoise done for %s %s in %.1fs",
                    day.strftime("%Y-%m-%d"),
                    network,
                    denoise_elapsed,
                )
            elapsed = time.monotonic() - t0
            logger.info(
                "Daylong load done for %s %s in %.1fs",
                day.strftime("%Y-%m-%d"),
                network,
                elapsed,
            )
            if elapsed > self.warn_s:
                logger.warning(
                    "Daylong load for %s %s (%s stations, %s channels) took %.1fs",
                    day.strftime("%Y-%m-%d"),
                    network,
                    len(stations),
                    len(channels),
                    elapsed,
                )
            # After _HITP2_EPOCH the GP* channels appear as station='HITP2'.
            # Rename them back to 'HITP' so template station names match.
            # Must happen after denoising (remove_HITP_spikes looks up 'HITP2'
            # internally via chunk_start) and after the resample check above.
            for tr in st:
                if tr.stats.station == "HITP2":
                    tr.stats.station = "HITP"
            self._raw_cache[key] = st.copy()
            return st

    def evict(self) -> None:
        """Clear all in-memory cached streams. On-disk cache files are kept."""
        self._raw_cache.clear()
        self._processed_cache.clear()

    def get_processed(self, day: UTCDateTime, network: str, stations: Sequence[str], channels: Sequence[str],
                      lowcut: Optional[float], highcut: Optional[float], filt_order: int, samp_rate: float,
                      process_length: Optional[float] = None) -> Stream:
        # Always process the full day regardless of process_length.
        # template.process_length is the chunk size used in the original
        # match_filter run, but for extracting any detection window we need
        # the full 86400 s processed (matching the repro script behaviour).
        key = (
            day.timestamp,
            network,
            tuple(sorted(stations)),
            tuple(sorted(channels)),
            lowcut,
            highcut,
            filt_order,
            samp_rate,
        )
        cached = self._processed_cache.get(key)
        if cached is not None:
            _log_every(
                f"day_proc_hit_{day.timestamp}_{network}",
                logging.INFO,
                "Daylong processed cache hit for %s %s",
                day.strftime("%Y-%m-%d"),
                network,
                every=1,
            )
            return cached.copy()
        lock = self._get_key_lock(self._processed_locks, key)
        with lock:
            cached = self._processed_cache.get(key)
            if cached is not None:
                _log_every(
                    f"day_proc_hit_{day.timestamp}_{network}",
                    logging.INFO,
                    "Daylong processed cache hit for %s %s",
                    day.strftime("%Y-%m-%d"),
                    network,
                    every=1,
                )
                return cached.copy()
            logger.info(
                "Daylong processing start for %s %s (%s stations, %s channels)",
                day.strftime("%Y-%m-%d"),
                network,
                len(stations),
                len(channels),
            )
            st = self._load_day_stream(day, network, stations, channels)
            aligned_start, aligned_end = _align_time_window(day, day + 86400.0, samp_rate)
            t0 = time.monotonic()
            processed = multi_process(
                st=st,
                lowcut=lowcut,
                highcut=highcut,
                filt_order=filt_order,
                samp_rate=samp_rate,
                parallel=True,
                num_cores=1,
                starttime=aligned_start,
                endtime=aligned_end,
                fill_gaps=True,
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "Daylong processing done for %s %s in %.1fs",
                day.strftime("%Y-%m-%d"),
                network,
                elapsed,
            )
            if elapsed > self.warn_s:
                logger.warning(
                    "Daylong processing for %s %s (%s stations, %s channels) took %.1fs",
                    day.strftime("%Y-%m-%d"),
                    network,
                    len(stations),
                    len(channels),
                    elapsed,
                )
            for tr in processed:
                tr.stats._daylong_processed = True
            self._processed_cache[key] = processed.copy()
            return processed


def _extract_family_inputs(
    family,
) -> Tuple[float, Sequence, Sequence[Sequence], List[str], List]:
    template = getattr(family, "template", None)
    template_event = getattr(template, "event", None) if template is not None else None
    if template_event is None and template is not None:
        template_event = template
    if template_event is None:
        raise ValueError("Family.template.event or Family.template must be an Event")

    template_mag = family.template.event.preferred_magnitude()
    if template_mag is None:
        raise ValueError("Template event has no preferred magnitude")
    if template_mag.magnitude_type not in {"ML", "MLc", "MLv", "M"}:
        raise ValueError("Template event must have a local magnitude (ML)")
    logger.info(
        "Template preferred magnitude: %s (%s)",
        template_mag.mag,
        template_mag.magnitude_type,
    )

    template_picks = list(getattr(template_event, "picks", []) or [])

    detections = family.detections
    if len(detections) == 0:
        logger.info("Template %s has no detections", family.template.name)

    detections_picks = []
    detection_ids = []
    detection_events = []
    for det in detections:
        event = getattr(det, "event", det)
        detections_picks.append(list(getattr(event, "picks", []) or []))
        detection_ids.append(str(getattr(event, "resource_id", "")) or str(det))
        detection_events.append(det)  # store the Detection object, not event — detect_time/detect_val live here

    logger.info(
        "Template picks: %s, detections: %s",
        len(template_picks),
        len(detections_picks),
    )
    return template_mag, template_picks, detections_picks, detection_ids, detection_events


class _FamilyState:
    """Per-family extraction accumulator, built day-by-day before SVD."""

    def __init__(
        self,
        *,
        family_name: str,
        family,
        template_obj,
        template_trace,
        template_mag: float,
        detection_times: List[UTCDateTime],
        detection_ids: List[str],
        detection_events: List,
        detections_picks: List,
        corr_thresh: float,
        xcorr_shift_len_s: float,
        plotvar: bool,
        debug_plots: bool,
        debug_plot_limit: int,
        debug_plot_dir: Optional[str],
        plot_dir: Optional[str],
        result_mag_type: str,
        update_events: bool,
        stachan_workers: int,
        noise_window_s: float,
        ml_mw_m: float,
        ml_mw_c: float,
    ):
        self.family_name = family_name
        self.family = family
        self.template_obj = template_obj
        self.template_trace = template_trace
        self.template_mag = template_mag
        self.detection_times = detection_times
        self.detection_ids = detection_ids
        self.detection_events = detection_events
        self.detections_picks = detections_picks
        self.corr_thresh = corr_thresh
        self.xcorr_shift_len_s = xcorr_shift_len_s
        self.plotvar = plotvar
        self.debug_plots = debug_plots
        self.debug_plot_limit = debug_plot_limit
        self.debug_plot_dir = debug_plot_dir
        self.plot_dir = plot_dir
        self.result_mag_type = result_mag_type
        self.update_events = update_events
        self.stachan_workers = stachan_workers
        # Empirical ML↔Mw conversion:  ML = ml_mw_m * Mw + ml_mw_c
        self.ml_mw_m: float = ml_mw_m
        self.ml_mw_c: float = ml_mw_c
        # One Stream slot per detection; filled only if loaded and CC-passed.
        n = len(detection_times)
        self.all_detection_streams: List[Stream] = [Stream() for _ in range(n)]
        # Pre-pick noise traces (duration = template signal window).
        self.noise_window_s: float = noise_window_s
        self.template_long_trace: Optional["Trace"] = None
        self.all_detection_long_traces: List[Optional["Trace"]] = [None] * n
        # Debug accumulation
        self.debug_corr_payloads: List[dict] = []
        self._debug_plot_counts: dict = {"corr": 0}
        self._debug_plot_lock: Lock = Lock()
        self.plot_payloads: dict = {}
        # Per-detection CC and diagnostic metadata, keyed by event_index.
        # Populated unconditionally during _family_extract_day.
        self.all_detection_meta: List[dict] = [{} for _ in range(n)]


def _family_init(
    family,
    *,
    corr_thresh: float,
    xcorr_shift_len_s: float,
    plotvar: bool,
    debug_plots: bool,
    debug_plot_limit: int,
    debug_plot_dir: Optional[str],
    plot_dir: Optional[str],
    result_mag_type: str,
    update_events: bool,
    stachan_workers: int,
    ml_mw_m: float = 1 / 0.791,   # UUSS: Mw = 0.791*ML + 0.851
    ml_mw_c: float = -0.851 / 0.791,
) -> "_FamilyState":
    """Decode a family into a _FamilyState ready for day-by-day extraction."""
    (
        template_mag,
        _template_picks,
        detections_picks,
        detection_ids,
        detection_events,
    ) = _extract_family_inputs(family)
    template_obj = getattr(family, "template", None)
    family_name = getattr(template_obj, "name", "") if template_obj is not None else ""

    total = len(detections_picks or [])
    records = []
    for idx in range(total):
        det_obj = detection_events[idx] if detection_events and idx < len(detection_events) else None
        det_time = getattr(det_obj, "detect_time", None) or getattr(det_obj, "detection_time", None)
        if det_time is None:
            raise ValueError(f"Detection {idx} is missing detect_time; strict mode requires detect_time")
        records.append({
            "id": detection_ids[idx] if detection_ids and idx < len(detection_ids) else str(idx),
            "event": det_obj,
            "time": det_time,
            "picks": detections_picks[idx] if idx < len(detections_picks) else [],
        })
    records.sort(key=lambda r: r["time"].timestamp)

    template_stream = getattr(template_obj, "st", None)
    if template_stream is None or len(template_stream) == 0:
        raise ValueError("Family template must include template.st for repro-style correlation")
    template_stream = template_stream.select(station="HITP", channel="GPZ")
    if len(template_stream) != 1:
        raise ValueError(
            f"Strict mode requires exactly one HITP.GPZ template trace, got {len(template_stream)}"
        )
    template_trace = template_stream[0].copy()

    _sr = template_trace.stats.sampling_rate
    noise_window_s = (template_trace.stats.npts - 1) / _sr if _sr > 0 else 0.0
    logger.info(
        "Family %s: noise_window_s=%.3fs (= template duration)",
        family_name, noise_window_s,
    )

    return _FamilyState(
        family_name=family_name,
        family=family,
        template_obj=template_obj,
        template_trace=template_trace,
        template_mag=float(template_mag.mag),
        detection_times=[r["time"] for r in records],
        detection_ids=[r["id"] for r in records],
        detection_events=[r["event"] for r in records],
        detections_picks=[r["picks"] for r in records],
        corr_thresh=corr_thresh,
        xcorr_shift_len_s=xcorr_shift_len_s,
        plotvar=plotvar,
        debug_plots=debug_plots,
        debug_plot_limit=debug_plot_limit,
        debug_plot_dir=debug_plot_dir,
        plot_dir=plot_dir,
        result_mag_type=result_mag_type,
        update_events=update_events,
        stachan_workers=stachan_workers,
        noise_window_s=noise_window_s,
        ml_mw_m=ml_mw_m,
        ml_mw_c=ml_mw_c,
    )


def _family_extract_day(
    state: "_FamilyState",
    day: UTCDateTime,
    trace_loader,
) -> None:
    """For all detections on `day`: load trace, compute CC, store if passing.

    Passing traces are stored in state.all_detection_streams[event_index].
    Failing or unloaded slots remain as empty Stream().
    """
    day_start = _day_start(day)
    day_end = day_start + 86400

    template_trace = state.template_trace
    sr = template_trace.stats.sampling_rate
    if sr <= 0:
        raise ValueError("Template trace has invalid sampling rate")
    duration = (template_trace.stats.npts - 1) / sr
    pad_samples = int(round(state.xcorr_shift_len_s * sr))
    if pad_samples < 1:
        raise ValueError(
            f"xcorr_shift_len_s={state.xcorr_shift_len_s} produces pad_samples={pad_samples}"
        )
    xcorr_func = get_array_xcorr()

    _day_merge_timed_out: bool = False
    for event_index, det_time in enumerate(state.detection_times):
        if not (day_start <= det_time < day_end):
            continue
        if _day_merge_timed_out:
            continue
        endtime = det_time + duration
        wf = SimpleNamespace(
            network_code=template_trace.stats.network or "",
            station_code="HITP",
            channel_code="GPZ",
            location_code="",
        )
        pseudo_pick = SimpleNamespace(phase="P", waveform_id=wf, time=det_time)
        try:
            st = trace_loader(pseudo_pick, det_time, endtime, state.template_obj, False)
        except DayStreamMergeTimeout as exc:
            logger.error(
                "Family %s: skipping remaining detections on %s — %s",
                state.family_name, day.strftime("%Y-%m-%d"), exc,
            )
            _day_merge_timed_out = True
            continue
        if st is None or len(st) < 1:
            raise RuntimeError(
                f"Strict load failed for event={event_index} station=HITP channel=GPZ at {det_time}"
            )
        tr = st[0].copy()
        if tr.data.dtype != np.float64:
            tr.data = tr.data.astype(np.float64)
        if (tr.stats.station or "") != "HITP" or (tr.stats.channel or "") != "GPZ":
            raise RuntimeError(
                f"Strict load returned wrong trace: got {tr.stats.station}.{tr.stats.channel}, expected HITP.GPZ"
            )
        logger.info(
            "ALIGN detect_time=%s tr_start=%s tr_npts=%d template_start=%s template_npts=%d offset_samples=%.3f",
            det_time,
            tr.stats.starttime,
            tr.stats.npts,
            template_trace.stats.starttime,
            template_trace.stats.npts,
            (tr.stats.starttime - det_time) * sr,
        )

        # Compute CC immediately — only needs template + this trace
        npts = min(tr.stats.npts, template_trace.stats.npts)
        tmpl_arr = np.asarray(template_trace.data[:npts], dtype=np.float32)
        det_arr = np.asarray(tr.data[:npts], dtype=np.float32)
        if det_arr.size < tmpl_arr.size:
            det_arr = np.pad(det_arr, (0, tmpl_arr.size - det_arr.size))
        elif det_arr.size > tmpl_arr.size:
            det_arr = det_arr[: tmpl_arr.size]
        data_long = np.pad(det_arr, (pad_samples, pad_samples), mode="constant")
        corr_series, _ = xcorr_func(tmpl_arr[None, :], data_long, [0], cc_squared=False)
        if corr_series is None or len(corr_series) == 0:
            raise RuntimeError("get_array_xcorr returned empty")
        corr_series = np.asarray(corr_series[0], dtype=float)
        expected = 2 * pad_samples + 1
        if corr_series.size != expected:
            raise RuntimeError(
                f"Unexpected correlation length: got {corr_series.size}, expected {expected}"
            )
        corr_val = float(corr_series[pad_samples])
        peak_idx = int(np.argmax(np.abs(corr_series)))
        best_lag_samples = peak_idx - pad_samples
        best_lag_s = best_lag_samples / sr
        corr_at_peak = float(corr_series[peak_idx])
        peak_series = float(np.max(np.abs(corr_series)))
        detect_val = getattr(state.detection_events[event_index], "detect_val", None)
        cc_passed = abs(corr_val) >= state.corr_thresh
        state.all_detection_meta[event_index] = {
            "corr_zero_lag": corr_val,
            "corr_peak": corr_at_peak,
            "corr_peak_abs": peak_series,
            "best_lag_samples": best_lag_samples,
            "best_lag_s": best_lag_s,
            "detect_val": detect_val,
            "cc_passed": cc_passed,
            "data_npts": tr.stats.npts,
            "data_std": float(np.std(tr.data)),
            "samp_rate": sr,
            "template_npts": template_trace.stats.npts,
        }
        det_id = (
            state.detection_ids[event_index]
            if event_index < len(state.detection_ids)
            else str(event_index)
        )
        logger.info(
            "CC stachan=HITP.GPZ event=%s zero_lag=%.4f series_peak=%.4f detect_val=%s",
            event_index,
            corr_val,
            peak_series,
            f"{detect_val:.4f}" if detect_val is not None else "N/A",
        )
        if detect_val is not None and detect_val > 0.8 and abs(corr_val) < 0.4:
            logger.warning(
                "HIGH_DETECT_LOW_CORR event=%s detect_val=%.4f corr=%.4f "
                "detect_time=%s template_st=%s "
                "tmpl_mean=%.4g tmpl_std=%.4g tmpl_max=%.4g "
                "data_mean=%.4g data_std=%.4g data_max=%.4g data_npts=%d",
                event_index,
                detect_val,
                corr_val,
                det_time,
                template_trace.stats.starttime,
                float(np.mean(template_trace.data)),
                float(np.std(template_trace.data)),
                float(np.max(np.abs(template_trace.data))),
                float(np.mean(tr.data)),
                float(np.std(tr.data)),
                float(np.max(np.abs(tr.data))),
                tr.stats.npts,
            )
            logger.warning("HIGH_DETECT_LOW_CORR loaded_trace_start=%s", tr.stats.starttime)

        if state.debug_plots and abs(corr_val) < state.corr_thresh:
            with state._debug_plot_lock:
                if state._debug_plot_counts["corr"] < state.debug_plot_limit:
                    state._debug_plot_counts["corr"] += 1
                    state.debug_corr_payloads.append({
                        "debug_index": state._debug_plot_counts["corr"],
                        "family": state.family_name or "family",
                        "detection_label": det_id,
                        "detect_val": detect_val,
                        "stachan": "HITP.GPZ",
                        "event_index": event_index,
                        "corr_val": corr_val,
                        "corr_abs": abs(corr_val),
                        "sr": sr,
                        "template_full": template_trace.data.copy(),
                        "detection_data": tr.data.copy(),
                        "corr_series": corr_series,
                        "shift_samples": pad_samples,
                        "det_trace_start": str(tr.stats.starttime),
                        "requested_detect_time": str(det_time),
                    })

        if cc_passed:
            state.all_detection_streams[event_index] = Stream([tr])
        # Load pre-pick noise window (same duration as signal) for bias-corrected magnitude.
        noise_start = det_time - state.noise_window_s
        wf_noise = SimpleNamespace(
            network_code=template_trace.stats.network or "",
            station_code="HITP",
            channel_code="GPZ",
            location_code="",
        )
        pseudo_pick_noise = SimpleNamespace(phase="P", waveform_id=wf_noise, time=noise_start)
        try:
            st_long = trace_loader(pseudo_pick_noise, noise_start, endtime, state.template_obj, False)
            if st_long and len(st_long) >= 1:
                state.all_detection_long_traces[event_index] = st_long[0].copy()
        except Exception as exc:
            logger.warning(
                "Could not load noise window for detection %d at %s: %s",
                event_index, det_time, exc,
            )


def _write_family_debug_plots(state: "_FamilyState") -> None:
    if not state.debug_plots:
        return
    debug_root = Path(state.debug_plot_dir or state.plot_dir or "debug_plots")
    corr_dir = debug_root / "corr"
    corr_dir.mkdir(parents=True, exist_ok=True)
    for payload in state.debug_corr_payloads:
        idx = payload.get("debug_index", 0)
        stachan = payload.get("stachan", "trace").replace(".", "_")
        family = payload.get("family", "family")
        detection = payload.get("detection_label", "detection")
        out_path = corr_dir / f"corr_{family}_{detection}_{stachan}_{idx:04d}.png"
        _plot_corr_debug(payload, out_path)


def _family_load_template_noise(state: "_FamilyState", trace_loader) -> None:
    """Load a pre-pick noise window for the template and store on state.

    Must be called while the day cache is still populated (before eviction),
    typically right after the day loop in __main__.
    """
    template_trace = state.template_trace
    tmpl_pick_time = template_trace.stats.starttime
    noise_start = tmpl_pick_time - state.noise_window_s
    duration_s = (template_trace.stats.npts - 1) / template_trace.stats.sampling_rate
    endtime = tmpl_pick_time + duration_s
    wf = SimpleNamespace(
        network_code=template_trace.stats.network or "",
        station_code="HITP",
        channel_code="GPZ",
        location_code="",
    )
    pseudo_pick = SimpleNamespace(phase="P", waveform_id=wf, time=noise_start)
    try:
        st = trace_loader(pseudo_pick, noise_start, endtime, state.template_obj, True)
    except Exception as exc:
        logger.warning(
            "Could not load template noise for family %s: %s", state.family_name, exc
        )
        return
    if st and len(st) > 0:
        state.template_long_trace = st[0].copy()
        logger.info(
            "Loaded template+noise trace for family %s (start=%s)",
            state.family_name, state.template_long_trace.stats.starttime,
        )


def _pairwise_relative_magnitude(
    template_trace: "Trace",
    detection_trace: "Trace",
    corr_peak_abs: float,
    template_long_trace: Optional["Trace"] = None,
    detection_long_trace: Optional["Trace"] = None,
    family_name: str = "",
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute pairwise amplitude-ratio relative magnitude (Schaff & Richards 2014).

    Returns (delta_m_uncorrected, delta_m_corrected).

    delta_m_uncorrected: pure log10(std_det/std_tmpl), no SNR/CC bias correction.
      Computed from signal-only traces; first 10 % of window used as noise proxy.
    delta_m_corrected: full Schaff & Richards bias correction using pre-pick noise
      from template_long_trace / detection_long_trace.  None if long traces are
      not supplied or the correction fails.

    Both return None on error.  Add to template_mag to obtain absolute Ml.
    """
    from obspy.core.event import Event, Pick, WaveformStreamID

    tr_tmpl = template_trace.copy()
    tr_det = detection_trace.copy()
    # Ensure both traces share a seed_id so the intersection is non-empty.
    if tr_tmpl.id != tr_det.id:
        logger.debug(
            "_pairwise_relative_magnitude: seed_id mismatch %s vs %s; skipping",
            tr_tmpl.id, tr_det.id,
        )
        return None, None
    seed_id = tr_tmpl.id
    sr = tr_tmpl.stats.sampling_rate
    duration_s = max((tr_tmpl.stats.npts - 1) / sr, 1.0 / sr)

    # Diagnostic: log std of template vs detection to help diagnose amplitude scale issues.
    _std_tmpl = float(np.std(tr_tmpl.data)) if tr_tmpl.stats.npts > 0 else 0.0
    _std_det = float(np.std(tr_det.data)) if tr_det.stats.npts > 0 else 0.0
    _ratio = _std_det / _std_tmpl if _std_tmpl > 0 else float("nan")
    logger.info(
        "pairwise_amp %s det=%s family=%s: party std_tmpl=%.4g std_det=%.4g ratio=%.4g (delta_m_raw=%.3f)",
        seed_id, tr_det.stats.starttime.isoformat(), family_name,
        _std_tmpl, _std_det, _ratio,
        float(np.log10(abs(_ratio))) if np.isfinite(_ratio) and _ratio > 0 else float("nan"),
    )
    # If template_long_trace is available it was loaded from the same processed
    # daylong stream as the detection, ensuring consistent amplitude scaling.
    # Prefer it over template_trace (which may have been stored with different
    # normalization in the Party file) for the uncorrected estimate.
    if template_long_trace is not None and detection_long_trace is not None:
        _tmpl_pick = template_trace.stats.starttime
        _det_pick = detection_trace.stats.starttime
        _tr_tmpl_sig = template_long_trace.copy().trim(
            starttime=_tmpl_pick, endtime=_tmpl_pick + duration_s
        )
        _tr_det_sig = detection_long_trace.copy().trim(
            starttime=_det_pick, endtime=_det_pick + duration_s
        )
        if _tr_tmpl_sig.stats.npts > 2 and _tr_det_sig.stats.npts > 2:
            _std_tmpl_dl = float(np.std(_tr_tmpl_sig.data))
            _std_det_dl = float(np.std(_tr_det_sig.data))
            _ratio_dl = _std_det_dl / _std_tmpl_dl if _std_tmpl_dl > 0 else float("nan")
            logger.info(
                "pairwise_amp %s (daylong): std_tmpl=%.4g std_det=%.4g ratio=%.4g (delta_m_raw=%.3f)",
                seed_id, _std_tmpl_dl, _std_det_dl, _ratio_dl,
                float(np.log10(abs(_ratio_dl))) if np.isfinite(_ratio_dl) and _ratio_dl > 0 else float("nan"),
            )
            tr_tmpl = _tr_tmpl_sig
            tr_tmpl.stats.station = tr_det.stats.station
            tr_tmpl.stats.network = tr_det.stats.network
            tr_tmpl.stats.channel = tr_det.stats.channel
            tr_det = _tr_det_sig

    # Recompute duration from the (possibly swapped) tr_tmpl.
    sr = tr_tmpl.stats.sampling_rate
    duration_s = max((tr_tmpl.stats.npts - 1) / sr, 1.0 / sr)
    # Noise window: first 10 % of the signal trace as a noise proxy.
    noise_end_proxy = max(duration_s * 0.1, 5.0 / sr)
    noise_window_proxy = (0.0, noise_end_proxy)
    signal_window = (0.0, duration_s)

    def _make_event(tr: "Trace", pick_time) -> "Event":
        pick = Pick()
        pick.time = pick_time
        pick.waveform_id = WaveformStreamID(
            network_code=tr.stats.network,
            station_code=tr.stats.station,
            location_code=tr.stats.location,
            channel_code=tr.stats.channel,
        )
        pick.phase_hint = "P"
        ev = Event()
        ev.picks = [pick]
        return ev

    # --- Uncorrected (bias-free) estimate ---
    ev1_unc = _make_event(tr_tmpl, tr_tmpl.stats.starttime)
    ev2_unc = _make_event(tr_det, tr_det.stats.starttime)
    delta_m_uncorrected: Optional[float] = None
    try:
        rel_mags = _eqc_relative_magnitude(
            st1=Stream([tr_tmpl]),
            st2=Stream([tr_det]),
            event1=ev1_unc,
            event2=ev2_unc,
            noise_window=noise_window_proxy,
            signal_window=signal_window,
            min_snr=1.0,
            min_cc=0.0,
            correlations={seed_id: corr_peak_abs},
            correct_mag_bias=False,
        )
        delta_m_uncorrected = rel_mags.get(seed_id)
    except Exception as exc:
        logger.warning("_pairwise_relative_magnitude (uncorrected) failed for %s: %s", seed_id, exc)

    # --- Corrected (SNR + CC bias) estimate using pre-pick noise traces ---
    delta_m_corrected: Optional[float] = None
    have_long = (
        template_long_trace is not None
        and detection_long_trace is not None
    )
    if have_long:
        tr_tmpl_long = template_long_trace.copy()
        tr_det_long = detection_long_trace.copy()
        tmpl_pick_time = template_trace.stats.starttime
        det_pick_time = detection_trace.stats.starttime
        tmpl_lead = tmpl_pick_time - tr_tmpl_long.stats.starttime
        det_lead = det_pick_time - tr_det_long.stats.starttime
        noise_window_s = min(tmpl_lead, det_lead)  # actual usable pre-pick span
        if noise_window_s < 0.1:
            logger.warning(
                "Long traces for %s lack sufficient pre-pick data "
                "(tmpl_lead=%.2fs det_lead=%.2fs); skipping corrected estimate",
                seed_id, tmpl_lead, det_lead,
            )
        else:
            ev1_corr = _make_event(tr_tmpl_long, tmpl_pick_time)
            ev2_corr = _make_event(tr_det_long, det_pick_time)
            noise_window_corr = (-noise_window_s, -0.1)
            try:
                rel_mags_corr = _eqc_relative_magnitude(
                    st1=Stream([tr_tmpl_long]),
                    st2=Stream([tr_det_long]),
                    event1=ev1_corr,
                    event2=ev2_corr,
                    noise_window=noise_window_corr,
                    signal_window=signal_window,
                    min_snr=1.5,
                    min_cc=0.0,
                    correlations={seed_id: corr_peak_abs},
                    correct_mag_bias=True,
                )
                delta_m_corrected = rel_mags_corr.get(seed_id)
            except Exception as exc:
                logger.warning(
                    "_pairwise_relative_magnitude (corrected) failed for %s: %s", seed_id, exc
                )

    return delta_m_uncorrected, delta_m_corrected


def _family_finalize(state: "_FamilyState") -> List[dict]:
    """Run SVD + svd_moments over all accumulated streams; return per-detection results."""
    template_idx = len(state.detection_times)
    all_streams = state.all_detection_streams + [Stream([state.template_trace.copy()])]

    # Indices of non-empty HITP.GPZ streams, in original order
    passed_indices = [
        i for i, st in enumerate(all_streams)
        if len(st.select(station="HITP", channel="GPZ")) > 0
    ]
    n_dets_passed = sum(1 for i in passed_indices if i != template_idx)
    logger.info(
        "Family %s: %d/%d detections passed CC filter; SVD starting",
        state.family_name,
        n_dets_passed,
        len(state.detection_times),
    )

    if template_idx not in passed_indices:
        logger.warning("Template missing from streams for family %s; skipping", state.family_name)
        _write_family_debug_plots(state)
        return []
    if n_dets_passed < 2:
        logger.warning(
            "Fewer than 2 detections for family %s after CC filter; skipping", state.family_name
        )
        _write_family_debug_plots(state)
        return []

    stachan = "HITP.GPZ"
    stachan_event_list = [(stachan, passed_indices)]
    # Compact stream list for SVD: only non-empty streams, preserving order
    filled_streams = [st for st in all_streams if len(st) > 0]

    if state.plotvar:
        chan_traces = [
            st.select(station="HITP", channel="GPZ")[0]
            for st in filled_streams
            if len(st.select(station="HITP", channel="GPZ")) > 0
        ]
        if chan_traces:
            npts = state.template_trace.stats.npts
            padded = []
            for tr in chan_traces:
                d = np.asarray(tr.data, dtype=np.float64)
                if len(d) < npts:
                    d = np.pad(d, (0, npts - len(d)))
                else:
                    d = d[:npts]
                padded.append(d)
            data = np.vstack(padded)
            max_abs = np.max(np.abs(data), axis=1)
            max_abs[max_abs == 0] = 1.0
            data_norm = data / max_abs[:, None]
            vmax = float(np.percentile(np.abs(data_norm), 99.0)) or 1.0
            state.plot_payloads[stachan] = {
                "data_norm": data_norm,
                "vmax": vmax,
                "template_data": state.template_trace.data.copy(),
                "title": "HITP.GPZ",
            }

    Uvectors, SValues, SVectors, out_stachans = clustering.svd(stream_list=filled_streams)
    event_list = []
    event_stachans = []
    for out_stachan in out_stachans:
        # out_stachan is a (station, channel) tuple from clustering.svd
        out_sc = "%s.%s" % (out_stachan[0], out_stachan[1]) if isinstance(out_stachan, tuple) else out_stachan
        for sc, ev_indices in stachan_event_list:
            if sc == out_sc:
                event_list.append(ev_indices)
                event_stachans.append(out_sc)
    if not event_list or not any(event_list):
        logger.warning("No valid event list for svd_moments for family %s", state.family_name)
        _write_family_debug_plots(state)
        return []

    if len(event_stachans) == 1:
        # EQcorrscan's svd_moments single-stachan shortcut erroneously returns
        # u[0][:, 0] (shape=n_samples) instead of event weights (shape=n_events).
        # The correct weights are in SVectors[0][0, :] (first row of numpy V,
        # i.e. U_harris[:,0] in the Harris decomposition convention).
        svd_weights = np.asarray(SVectors[0][0, :], dtype=float)
        if np.all(svd_weights < 0):
            logger.info("Family %s: all SVD weights negative; flipping sign", state.family_name)
            svd_weights = np.abs(svd_weights)
        relative_moments = svd_weights
        # event_list[0] is already passed_indices; keep it as the flat list
        event_list = event_list[0]
    else:
        relative_moments, event_list = svd_moments(
            u=Uvectors,
            s=SValues,
            v=SVectors,
            stachans=event_stachans,
            event_list=event_list,
        )

    if event_list[-1] != len(state.detection_times):
        logger.error("Template not last in event_list for family %s; aborting", state.family_name)
        logger.error("Largest event in event_list: %s", event_list[-1])
        _write_family_debug_plots(state)
        return False

    template_moment = local_to_moment(state.template_mag, m=state.ml_mw_m, c=state.ml_mw_c)
    norm_moment = template_moment / relative_moments[-1]
    moments = relative_moments * norm_moment
    n_negative = int(np.sum(moments < 0))
    if n_negative:
        logger.info(
            "Family %s: %d/%d relative moments negative (SVD sign ambiguity); "
            "taking abs() for magnitude calculation",
            state.family_name, n_negative, len(moments),
        )
    mw = [2.0 / 3.0 * (np.log10(abs(M)) - 9.0) if np.isfinite(M) and M != 0 else float("nan") for M in moments]
    # ML via empirical regression:  ML = ml_mw_m * Mw + ml_mw_c
    ml = [state.ml_mw_m * M + state.ml_mw_c if np.isfinite(M) else float("nan") for M in mw]
    _finite_ml = [m for m in ml[:-1] if np.isfinite(m)]
    logger.info(
        "Family %s: SVD ML range [%.2f, %.2f] (template_mag=%.2f, n=%d)",
        state.family_name,
        min(_finite_ml) if _finite_ml else float("nan"),
        max(_finite_ml) if _finite_ml else float("nan"),
        state.template_mag,
        len(_finite_ml),
    )

    results = []
    _ml_pairwise_list: list = []
    _ml_pairwise_corrected_list: list = []
    for i, event_id in enumerate(event_list[:-1]):
        event_index = event_id
        label = (
            state.detection_ids[event_id]
            if state.detection_ids and event_id < len(state.detection_ids)
            else str(event_id)
        )
        if state.update_events and state.detection_events and event_index < len(state.detection_events):
            det_obj = state.detection_events[event_index]
            event = getattr(det_obj, "event", det_obj)
            if np.isfinite(ml[i]) and hasattr(event, "magnitudes"):
                mag = Magnitude(mag=ml[i], magnitude_type=state.result_mag_type)
                event.magnitudes.append(mag)
                if getattr(event, "preferred_magnitude_id", None) is None:
                    event.preferred_magnitude_id = mag.resource_id
        detect_time = (
            state.detection_times[event_index].isoformat()
            if event_index < len(state.detection_times)
            else ""
        )
        meta = state.all_detection_meta[event_index] if event_index < len(state.all_detection_meta) else {}
        # Pairwise amplitude-ratio magnitude (Schaff & Richards 2014)
        det_stream = state.all_detection_streams[event_index]
        det_traces = det_stream.select(station="HITP", channel="GPZ")
        delta_m_pairwise: Optional[float] = None
        ml_pairwise: Optional[float] = None
        delta_m_pairwise_corrected: Optional[float] = None
        ml_pairwise_corrected: Optional[float] = None
        if len(det_traces) == 1:
            det_long = (
                state.all_detection_long_traces[event_index]
                if event_index < len(state.all_detection_long_traces)
                else None
            )
            delta_m_pairwise, delta_m_pairwise_corrected = _pairwise_relative_magnitude(
                template_trace=state.template_trace,
                detection_trace=det_traces[0],
                corr_peak_abs=float(meta.get("corr_peak_abs") or 0.0),
                template_long_trace=state.template_long_trace,
                detection_long_trace=det_long,
                family_name=state.family_name,
            )
            # delta_m_pairwise = log10(M0_det/M0_tmpl) = (3/2)*delta_Mw
            # Convert to delta_ML via the empirical regression:
            # delta_ML = ml_mw_m * delta_Mw = ml_mw_m * (2/3) * delta_m_pairwise
            _pairwise_ml_factor = state.ml_mw_m * (2.0 / 3.0)
            if delta_m_pairwise is not None:
                ml_pairwise = state.template_mag + _pairwise_ml_factor * delta_m_pairwise
            if delta_m_pairwise_corrected is not None:
                ml_pairwise_corrected = state.template_mag + _pairwise_ml_factor * delta_m_pairwise_corrected
        if ml_pairwise is not None:
            _ml_pairwise_list.append(ml_pairwise)
        if ml_pairwise_corrected is not None:
            _ml_pairwise_corrected_list.append(ml_pairwise_corrected)
        results.append({
            "family_name": state.family_name,
            "template_mag": state.template_mag,
            "template_event_time": (
                state.template_trace.stats.starttime + float(
                    getattr(state.template_obj, "prepick", None) or 0.0
                )
            ).isoformat(),
            "event_index": event_index,
            "event_id": label,
            "detect_time": detect_time,
            "moment": moments[i] if np.isfinite(moments[i]) else "",
            "mw": mw[i] if np.isfinite(mw[i]) else "",
            "ml": ml[i] if np.isfinite(ml[i]) else "",
            "corr_zero_lag": meta.get("corr_zero_lag", ""),
            "corr_peak": meta.get("corr_peak", ""),
            "corr_peak_abs": meta.get("corr_peak_abs", ""),
            "best_lag_samples": meta.get("best_lag_samples", ""),
            "best_lag_s": meta.get("best_lag_s", ""),
            "detect_val": meta.get("detect_val", ""),
            "data_std": meta.get("data_std", ""),
            "data_npts": meta.get("data_npts", ""),
            "samp_rate": meta.get("samp_rate", ""),
            "template_npts": meta.get("template_npts", ""),
            "delta_m_pairwise": delta_m_pairwise if delta_m_pairwise is not None else "",
            "ml_pairwise": ml_pairwise if ml_pairwise is not None else "",
            "delta_m_pairwise_corrected": delta_m_pairwise_corrected if delta_m_pairwise_corrected is not None else "",
            "ml_pairwise_corrected": ml_pairwise_corrected if ml_pairwise_corrected is not None else "",
        })

    if state.plot_dir:
        plot_path = Path(state.plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)
        safe_family = state.family_name.replace(" ", "_") if state.family_name else "family"
        out_path = plot_path / f"{safe_family}_gr.png"
        _gr_extra: dict = {}
        if _ml_pairwise_list:
            _gr_extra["Pairwise ML"] = _ml_pairwise_list
        if _ml_pairwise_corrected_list:
            _gr_extra["Pairwise ML (corrected)"] = _ml_pairwise_corrected_list
        # Identify self-detections: detect_time within 2 s of template event time
        _tmpl_event_time = state.template_trace.stats.starttime + float(
            getattr(state.template_obj, "prepick", None) or 0.0
        )
        _self_detect_ml = []
        for _r in results:
            try:
                if abs(UTCDateTime(_r["detect_time"]) - _tmpl_event_time) <= 2.0 and _r.get("ml") != "":
                    _self_detect_ml.append(float(_r["ml"]))
            except Exception:
                pass
        _plot_gutenberg_richter(
            ml, out_path, f"Gutenberg-Richter: {state.family_name}",
            extra_series=_gr_extra or None,
            template_mags=state.template_mag,
            self_detect_mags=_self_detect_ml or None,
        )
        logger.info("Wrote Gutenberg-Richter plot %s", out_path)

        # --- Moment sign diagnostic ---
        # Raw signed moments for events only (exclude template = last index)
        _ev_moments = [moments[i] for i, _ in enumerate(event_list[:-1]) if np.isfinite(moments[i])]
        if _ev_moments:
            _pos = [m for m in _ev_moments if m >= 0]
            _neg = [m for m in _ev_moments if m < 0]
            _fig_ms, _ax_ms = plt.subplots(figsize=(7, 4))
            if _pos:
                _ax_ms.hist(_pos, bins=40, color="steelblue", alpha=0.7, label=f"Positive ({len(_pos)})")
            if _neg:
                _ax_ms.hist(_neg, bins=40, color="firebrick", alpha=0.7, label=f"Negative ({len(_neg)})")
            _ax_ms.axvline(0, color="black", linewidth=0.8, linestyle="--")
            _ax_ms.set_xlabel("Relative moment (signed)")
            _ax_ms.set_ylabel("Count")
            _ax_ms.set_title(f"SVD relative moments: {state.family_name}")
            _ax_ms.legend(fontsize=8)
            _fig_ms.tight_layout()
            _ms_path = plot_path / f"{safe_family}_moment_sign.png"
            _fig_ms.savefig(_ms_path, dpi=200, bbox_inches="tight")
            plt.close(_fig_ms)
            logger.info("Wrote moment sign diagnostic plot %s", _ms_path)

        # --- CC vs moment scatter ---
        _cc_vals, _mom_vals, _neg_mask = [], [], []
        for i, ev_id in enumerate(event_list[:-1]):
            _meta_i = state.all_detection_meta[ev_id] if ev_id < len(state.all_detection_meta) else {}
            _cc = _meta_i.get("corr_peak_abs")
            _m = moments[i]
            if _cc is not None and np.isfinite(_m) and _m != 0:
                _cc_vals.append(float(_cc))
                _mom_vals.append(abs(float(_m)))
                _neg_mask.append(_m < 0)
        if _cc_vals:
            _cc_arr = np.array(_cc_vals)
            _mom_arr = np.array(_mom_vals)
            _neg_arr = np.array(_neg_mask)
            _fig_cc, _ax_cc = plt.subplots(figsize=(7, 4))
            _ax_cc.scatter(_cc_arr[~_neg_arr], _mom_arr[~_neg_arr],
                           s=8, alpha=0.5, color="steelblue", label="Positive moment")
            if _neg_arr.any():
                _ax_cc.scatter(_cc_arr[_neg_arr], _mom_arr[_neg_arr],
                               s=12, alpha=0.8, color="firebrick", marker="x", label="Negative moment (abs)")
            _ax_cc.set_xlabel("CC peak (abs)")
            _ax_cc.set_ylabel("|Relative moment|")
            _ax_cc.set_title(f"CC vs moment: {state.family_name}")
            _ax_cc.legend(fontsize=8)
            _fig_cc.tight_layout()
            _cc_path = plot_path / f"{safe_family}_cc_vs_moment.png"
            _fig_cc.savefig(_cc_path, dpi=200, bbox_inches="tight")
            plt.close(_fig_cc)
            logger.info("Wrote CC vs moment plot %s", _cc_path)

    if state.plotvar:
        plot_path = Path(state.plot_dir) if state.plot_dir else None
        stachan_index = {"%s.%s" % (sta, chan): idx for idx, (sta, chan) in enumerate(out_stachans)}
        for sc, payload in state.plot_payloads.items():
            if sc not in stachan_index:
                continue
            idx = stachan_index[sc]
            template_data = payload.get("template_data")
            data_norm = payload.get("data_norm")
            vmax = payload.get("vmax")
            if data_norm is None or template_data is None:
                continue
            uvec = Uvectors[idx]
            if uvec is None or len(uvec.shape) < 2:
                continue
            sv1 = np.asarray(uvec[:, 0]).reshape(-1)
            fig, axes = plt.subplots(
                3, 2, sharex="col", figsize=(9, 10),
                gridspec_kw={"height_ratios": [1, 5, 1], "width_ratios": [1, 0.03]},
            )
            axes[0, 1].axis("off")
            axes[2, 1].axis("off")
            x_vals = np.arange(len(template_data))
            axes[0, 0].plot(x_vals, template_data, "k", linewidth=1.0)
            axes[0, 0].set_title(payload.get("title", sc))
            axes[0, 0].set_ylabel("Template")
            im = axes[1, 0].imshow(
                data_norm, aspect="auto", cmap="seismic",
                vmin=-vmax, vmax=vmax, interpolation="nearest",
                extent=[0, data_norm.shape[1] - 1, data_norm.shape[0] - 1, 0],
            )
            axes[1, 0].set_ylabel("Trace")
            fig.colorbar(im, cax=axes[1, 1], label="Normalized amp")
            sv_x = np.linspace(0, len(sv1) - 1, len(sv1))
            axes[2, 0].plot(sv_x, sv1, color="tab:blue", linewidth=1.0)
            axes[2, 0].set_ylabel("SVD v1")
            axes[2, 0].set_xlabel("Sample")
            if plot_path:
                plot_path.mkdir(parents=True, exist_ok=True)
                safe_stachan = sc.replace(".", "_")
                safe_fname = state.family_name.replace(" ", "_") if state.family_name else "family"
                out_path = plot_path / f"{safe_fname}_{safe_stachan}_svd.png"
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                logger.info("Wrote stachan SVD plot %s", out_path)
                plt.close(fig)
            else:
                plt.show()

    _write_family_debug_plots(state)
    return results

if __name__ == "__main__":
    import argparse
    from eqcorrscan import Party

    _parser = argparse.ArgumentParser(description="Relative moment magnitudes")
    _parser.add_argument(
        "--plots-only", action="store_true",
        help="Skip all computation; regenerate plots from existing RESULTS_CSV.",
    )
    _parser.add_argument(
        "--exclude-families", default="",
        help="Comma-separated family names to omit from party-level plots.",
    )
    _args = _parser.parse_args()
    _exclude_families = {f.strip() for f in _args.exclude_families.split(",") if f.strip()}

    LOG_FILE = "relative_moments.log"
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("Logging to %s", LOG_FILE)


    PARTY_PATH = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/All_detections_HITP-HITP2_MAD20_w-magnitudes.tgz"
    PARTY_OUTPUT = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/All_detections_HITP-HITP2_MAD20_detection-magnitudes.tgz"
    RESULTS_CSV = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/magnitudes.csv"
    CLIENT_URL = 'http://131.243.224.19:8085'

    READ_DETECTION_CATALOG = False
    ESTIMATE_ORIGIN = True
    PREFER_LOCAL = True
    RESULT_MAG_TYPE = "ML"

    CORR_THRESH = 0.6
    XCORR_SHIFT_LEN_S = 0.1
    PLOTVAR = True
    STACHAN_WORKERS = 1
    PLOT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/magnitude_plots"
    DAY_WAVEFORM_CACHE = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/waveform_cache"
    DENOISE_ENABLED = False
    SPIKE_TEMPLATE_PATH = [
        "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
        "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
    ]
    DENOISE_PLOT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/denoiser_plots"
    DENOISE_CHUNK_S = 3600.0
    DENOISE_GEOPHONE_CHANS = ["GPZ"]
    DEBUG_PLOTS = True
    DEBUG_PLOT_LIMIT = 50
    DEBUG_PLOT_DIR = None
    # Empirical ML↔Mw conversion for this study area:  ML = ML_MW_M * Mw + ML_MW_C
    # UUSS regression: Mw = 0.791 * ML + 0.851  →  ML = (1/0.791)*Mw - (0.851/0.791)
    ML_MW_M = 1 / 0.791   # ≈ 1.2642
    ML_MW_C = -0.851 / 0.791   # ≈ -1.0758
    # Set to a file path to enable resume-from-checkpoint; set to None to disable.
    CHECKPOINT_FILE = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/relative_moments_checkpoint.pkl"

    # --- --plots-only shortcut: read CSV → regenerate plots → exit ---
    if _args.plots_only:
        if not RESULTS_CSV or not Path(RESULTS_CSV).exists():
            raise FileNotFoundError(f"PLOTS_ONLY=True but RESULTS_CSV not found: {RESULTS_CSV}")
        logger.info("PLOTS_ONLY mode: loading results from %s", RESULTS_CSV)
        with open(RESULTS_CSV, newline="") as _fh:
            _reader = csv.DictReader(_fh)
            all_results = list(_reader)
        def _f(v):
            try: return float(v)
            except (TypeError, ValueError): return None
        _plot_results = [
            r for r in all_results
            if r.get("family_name", "") not in _exclude_families
        ]
        if _exclude_families:
            logger.info("PLOTS_ONLY: excluding %d families from party plots: %s",
                        len(_exclude_families), _exclude_families)
        party_magnitudes = [x for x in (_f(r.get("ml")) for r in _plot_results) if x is not None]
        party_magnitudes_pairwise = [x for x in (_f(r.get("ml_pairwise")) for r in _plot_results) if x is not None]
        party_magnitudes_pairwise_corrected = [x for x in (_f(r.get("ml_pairwise_corrected")) for r in _plot_results) if x is not None]
        if PLOT_DIR and party_magnitudes:
            plot_path = Path(PLOT_DIR)
            plot_path.mkdir(parents=True, exist_ok=True)
            gr_extra: dict = {}
            if party_magnitudes_pairwise:
                gr_extra["Pairwise ML"] = party_magnitudes_pairwise
            if party_magnitudes_pairwise_corrected:
                gr_extra["Pairwise ML (corrected)"] = party_magnitudes_pairwise_corrected
            _po_tmpl_mags = sorted({
                float(r["template_mag"]) for r in _plot_results
                if r.get("template_mag") not in (None, "")
            })
            # Self-detections: detect_time within 2 s of template_event_time
            _po_self_ml = []
            for _r in _plot_results:
                try:
                    if (abs(UTCDateTime(_r["detect_time"]) - UTCDateTime(_r["template_event_time"])) <= 2.0
                            and _r.get("ml") not in (None, "")):
                        _po_self_ml.append(float(_r["ml"]))
                except Exception:
                    pass
            _gr_suffix = f"_excl{len(_exclude_families)}" if _exclude_families else ""
            _plot_gutenberg_richter(
                party_magnitudes, plot_path / f"party_gr{_gr_suffix}.png",
                "Gutenberg-Richter: Party", extra_series=gr_extra or None,
                template_mags=_po_tmpl_mags or None,
                self_detect_mags=_po_self_ml or None,
            )
            logger.info("PLOTS_ONLY: wrote GR plot")
        if PLOT_DIR and _plot_results:
            _plot_magnitude_comparison(_plot_results, PLOT_DIR)
            logger.info("PLOTS_ONLY: wrote magnitude comparison plot")
        logger.info("PLOTS_ONLY: done")
        raise SystemExit(0)

    party = Party().read(
        PARTY_PATH,
        read_detection_catalog=READ_DETECTION_CATALOG,
        estimate_origin=ESTIMATE_ORIGIN,
    )
    logger.info("Loaded Party: %s", PARTY_PATH)

    client = Client(CLIENT_URL) if CLIENT_URL else None

    families = party.families if hasattr(party, "families") else party
    party_magnitudes = []
    day_cache = DaylongStreamCache(
        DAY_WAVEFORM_CACHE,
        client,
        prefer_local=PREFER_LOCAL,
        denoise=DENOISE_ENABLED,
        spike_template_path=SPIKE_TEMPLATE_PATH,
        denoise_plot_dir=DENOISE_PLOT_DIR,
        denoise_chunk_s=DENOISE_CHUNK_S,
        denoise_geophone_chans=DENOISE_GEOPHONE_CHANS,
    )
    logger.info("Extraction denoise enabled: %s", DENOISE_ENABLED)

    def _day_trace_loader(pick, starttime, endtime, template_obj, is_template):
        wf = getattr(pick, "waveform_id", None)
        if wf is None:
            raise ValueError("Strict mode requires waveform_id")
        if template_obj is None:
            raise ValueError("Strict mode requires template_obj")
        net = wf.network_code or ""
        sta = "HITP"
        cha = "GPZ"
        if not net:
            raise ValueError("Strict mode requires network code")
        day = _day_start(starttime)
        stations = [sta]
        channels = [cha]
        lowcut = getattr(template_obj, "lowcut", None)
        highcut = getattr(template_obj, "highcut", None)
        filt_order = getattr(template_obj, "filt_order", None)
        samp_rate = getattr(template_obj, "samp_rate", None)
        process_length = getattr(template_obj, "process_length", None)
        if lowcut is None or highcut is None or filt_order is None or samp_rate is None:
            raise ValueError("Strict mode requires template lowcut/highcut/filt_order/samp_rate")
        processed = day_cache.get_processed(
            day,
            net,
            stations,
            channels,
            lowcut,
            highcut,
            filt_order,
            samp_rate,
            process_length=process_length,
        )
        selected = processed.select(
            network=net,
            station=sta,
            channel=cha,
            location="",
        )
        if len(selected) != 1:
            raise RuntimeError(
                f"Strict trace selection expected exactly 1 HITP.GPZ trace, got {len(selected)} for {starttime}"
            )
        src = selected[0]
        sr = float(src.stats.sampling_rate)
        if sr <= 0:
            raise RuntimeError("Selected trace has invalid sampling rate")
        npts = int(round((endtime - starttime) * sr)) + 1
        i0 = int(round((starttime - src.stats.starttime) * sr))
        i1 = i0 + npts
        out = np.zeros(npts, dtype=np.float64)
        s0 = max(i0, 0)
        s1 = min(i1, src.stats.npts)
        if s1 > s0:
            out[s0 - i0:s1 - i0] = np.asarray(src.data[s0:s1], dtype=np.float64)
        tr = src.copy()
        tr.data = out
        tr.stats.starttime = starttime
        logger.info(
            "TRACE_INDEX net=%s sta=%s cha=%s req=%s src_start=%s i0=%d npts=%d",
            net,
            sta,
            cha,
            starttime,
            src.stats.starttime,
            i0,
            npts,
        )
        return Stream([tr])

    states = [
        _family_init(
            family,
            corr_thresh=CORR_THRESH,
            xcorr_shift_len_s=XCORR_SHIFT_LEN_S,
            plotvar=PLOTVAR,
            result_mag_type=RESULT_MAG_TYPE,
            update_events=True,
            stachan_workers=STACHAN_WORKERS,
            plot_dir=PLOT_DIR,
            debug_plots=DEBUG_PLOTS,
            debug_plot_limit=DEBUG_PLOT_LIMIT,
            debug_plot_dir=DEBUG_PLOT_DIR,
            ml_mw_m=ML_MW_M,
            ml_mw_c=ML_MW_C,
        )
        for family in families
    ]
    logger.info("Initialized %d family states", len(states))

    day_timestamps: set = set()
    for state in states:
        for t in state.detection_times:
            day_timestamps.add(_day_start(t).timestamp)
    day_keys = sorted(day_timestamps)
    logger.info("Days to process: %d", len(day_keys))

    # --- Checkpoint resume ---
    completed_days: set = set()
    _loaded_from_checkpoint = False
    if CHECKPOINT_FILE and Path(CHECKPOINT_FILE).exists():
        try:
            with open(CHECKPOINT_FILE, "rb") as _f:
                _ckpt = pickle.load(_f)
            completed_days = _ckpt.get("completed_days", set())
            for state, saved in zip(states, _ckpt.get("state_data", [])):
                state.all_detection_streams = saved["all_detection_streams"]
                state.all_detection_meta = saved["all_detection_meta"]
                state.all_detection_long_traces = saved["all_detection_long_traces"]
                if saved.get("template_long_trace") is not None:
                    state.template_long_trace = saved["template_long_trace"]
            _loaded_from_checkpoint = True
            logger.info(
                "Resumed from checkpoint %s (%d/%d days already done)",
                CHECKPOINT_FILE, len(completed_days), len(day_keys),
            )
        except Exception as _exc:
            logger.warning(
                "Could not load checkpoint %s: %s — starting fresh",
                CHECKPOINT_FILE, _exc,
            )
            completed_days = set()
            _loaded_from_checkpoint = False

    # Load pre-pick noise for templates not already restored from checkpoint.
    _noise_needed = [s for s in states if s.template_long_trace is None]
    if _noise_needed:
        logger.info("Loading template noise for %d families", len(_noise_needed))
        for state in _noise_needed:
            _family_load_template_noise(state, _day_trace_loader)
        day_cache.evict()
        logger.info("Template noise loaded; cache evicted")
        # Backfill template_long_trace into the checkpoint so future restarts
        # don't need to reload it.
        if CHECKPOINT_FILE and Path(CHECKPOINT_FILE).exists() and completed_days:
            try:
                with open(CHECKPOINT_FILE, "rb") as _f:
                    _ckpt_backfill = pickle.load(_f)
                for state, saved in zip(states, _ckpt_backfill.get("state_data", [])):
                    saved["template_long_trace"] = state.template_long_trace
                _ckpt_tmp = str(CHECKPOINT_FILE) + ".tmp"
                with open(_ckpt_tmp, "wb") as _f:
                    pickle.dump(_ckpt_backfill, _f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(_ckpt_tmp, CHECKPOINT_FILE)
                logger.info("Checkpoint backfilled with template_long_trace for %d families", len(states))
            except Exception as _exc:
                logger.warning("Could not backfill checkpoint: %s", _exc)
    else:
        logger.info("Template noise restored from checkpoint; skipping reload")

    for day_key in day_keys:
        if day_key in completed_days:
            logger.info("Skipping completed day %s", UTCDateTime(day_key).strftime("%Y-%m-%d"))
            continue
        day = UTCDateTime(day_key)
        logger.info("Processing day %s", day.strftime("%Y-%m-%d"))
        for state in states:
            _family_extract_day(state, day, _day_trace_loader)
        day_cache.evict()
        logger.info("Cache evicted after day %s", day.strftime("%Y-%m-%d"))
        if CHECKPOINT_FILE:
            completed_days.add(day_key)
            _ckpt_data = {
                "completed_days": completed_days,
                "state_data": [
                    {
                        "all_detection_streams": state.all_detection_streams,
                        "all_detection_meta": state.all_detection_meta,
                        "all_detection_long_traces": state.all_detection_long_traces,
                        "template_long_trace": state.template_long_trace,
                    }
                    for state in states
                ],
            }
            _ckpt_tmp = str(CHECKPOINT_FILE) + ".tmp"
            with open(_ckpt_tmp, "wb") as _f:
                pickle.dump(_ckpt_data, _f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(_ckpt_tmp, CHECKPOINT_FILE)
            logger.info(
                "Checkpoint saved: %d/%d days done",
                len(completed_days), len(day_keys),
            )

    all_results: List[dict] = []
    party_magnitudes_pairwise: List[float] = []
    party_magnitudes_pairwise_corrected: List[float] = []
    for state in states:
        family_results = _family_finalize(state)
        if not family_results:
            continue
        for row in family_results:
            all_results.append(row)
            ml_val = row.get("ml") if isinstance(row, dict) else None
            if ml_val is not None:
                party_magnitudes.append(ml_val)
            try:
                party_magnitudes_pairwise.append(float(row["ml_pairwise"]))
            except (TypeError, ValueError, KeyError):
                pass
            try:
                party_magnitudes_pairwise_corrected.append(float(row["ml_pairwise_corrected"]))
            except (TypeError, ValueError, KeyError):
                pass

    if RESULTS_CSV:
        csv_path = Path(RESULTS_CSV)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "family_name", "template_mag", "template_event_time",
            "event_index", "event_id", "detect_time",
            "moment", "mw", "ml",
            "delta_m_pairwise", "ml_pairwise",
            "delta_m_pairwise_corrected", "ml_pairwise_corrected",
            "corr_zero_lag", "corr_peak", "corr_peak_abs",
            "best_lag_samples", "best_lag_s",
            "detect_val", "data_std", "data_npts", "samp_rate", "template_npts",
        ]
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Wrote %d magnitude rows to %s", len(all_results), RESULTS_CSV)

    if PLOT_DIR and party_magnitudes:
        plot_path = Path(PLOT_DIR)
        plot_path.mkdir(parents=True, exist_ok=True)
        _plot_results = [
            r for r in all_results
            if r.get("family_name", "") not in _exclude_families
        ]
        if _exclude_families:
            logger.info("Excluding %d families from party plots: %s",
                        len(_exclude_families), _exclude_families)
        def _pf(v):
            try: return float(v)
            except (TypeError, ValueError): return None
        _plot_mags = [x for x in (_pf(r.get("ml")) for r in _plot_results) if x is not None]
        _plot_mags_pw = [x for x in (_pf(r.get("ml_pairwise")) for r in _plot_results) if x is not None]
        _plot_mags_pwc = [x for x in (_pf(r.get("ml_pairwise_corrected")) for r in _plot_results) if x is not None]
        out_path = plot_path / ("party_gr.png" if not _exclude_families else f"party_gr_excl{len(_exclude_families)}.png")
        gr_extra: dict = {}
        if _plot_mags_pw:
            gr_extra["Pairwise ML"] = _plot_mags_pw
        if _plot_mags_pwc:
            gr_extra["Pairwise ML (corrected)"] = _plot_mags_pwc
        _party_tmpl_mags = sorted({
            float(r["template_mag"]) for r in _plot_results
            if r.get("template_mag") not in (None, "")
        })
        _plot_gutenberg_richter(
            _plot_mags, out_path, "Gutenberg-Richter: Party",
            extra_series=gr_extra or None,
            template_mags=_party_tmpl_mags or None,
        )
        logger.info("Wrote party Gutenberg-Richter plot %s", out_path)

    if PLOT_DIR and all_results:
        _plot_magnitude_comparison(all_results, PLOT_DIR)
        _plot_self_detection_sanity(all_results, states, PLOT_DIR)

    if PARTY_OUTPUT:
        logger.info("Writing Party: %s", PARTY_OUTPUT)
        party.write(PARTY_OUTPUT, overwrite=True)
        logger.info("Wrote Party with updated magnitudes to %s", PARTY_OUTPUT)
    else:
        logger.info("PARTY_OUTPUT not set; skipping Party write")
