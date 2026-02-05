#!/usr/bin/python
import os
import argparse
import random
import warnings
import multiprocessing
import functools
import os
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.signal import detrend
from scipy.ndimage import median_filter
from datetime import timedelta
from matplotlib.lines import Line2D

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, Stream, Trace
from eqcorrscan.core.match_filter import match_filter
from eqcorrscan import Tribe, Party, Family, Template
from eqcorrscan.utils.pre_processing import multi_process
from eqcorrscan.core.match_filter import match_filter


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
    min_quiet_spikes=10, # MODIFIED: Added minimum threshold
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
        station == "HITP2"
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
        print("Removing slow offsets using rolling median...")
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
        print(f"Running matched-filter detection on {ref_chan} (template {template_idx})...")
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

        print(f"Found {len(detections)} initial detections for template {template_idx}.")
        if not detections:
            warnings.warn(f"No detections found for template {template_idx}. Stream will not be modified for this template.")
            continue

        # 4. QC: FIND QUIETEST SPIKES (PARALLELIZED)
        snippet_before_sec=0.1
        snippet_after_sec=0.4
        cpu_cores = os.cpu_count()
        print(f"Finding quietest spikes in parallel using {cpu_cores} cores (template {template_idx})...")
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

        print(f"Selected {num_to_select} quietest spikes for TF calculation (template {template_idx}).")
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
            print(f"Plotting spike stack for quality control (template {template_idx})...")
            fig_spikes, axes_spikes = plt.subplots(len(all_chans), 1, figsize=(10, 8), sharex=True, sharey=True)
            fig_spikes.suptitle(f'Stacked Quiet Spikes for Station {station} (Template {template_idx})', fontsize=16)
            for i, ch in enumerate(all_chans):
                ax = axes_spikes if len(all_chans) == 1 else axes_spikes[i]
                for snip in quiet_snippets:
                    if ch not in [tr.stats.channel for tr in snip]: continue
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
            print(f"Saved spike stack plot to {savename_spikes}")
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
            print(f"Plotting final median transfer functions (template {template_idx})...")
            plt.figure(figsize=(10, 8))
            for ch in geophone_chans:
                if ch not in transfer_functions: continue
                plt.subplot(2, 1, 1)
                plt.plot(freq, np.real(transfer_functions[ch]), label=f"Real({ch}/{ref_chan})")
                plt.title('Median Transfer Functions (Real Part)'); plt.grid(True); plt.legend()
                plt.subplot(2, 1, 2)
                plt.plot(freq, np.imag(transfer_functions[ch]), label=f"Imag({ch}/{ref_chan})")
                plt.title('Median Transfer Functions (Imaginary Part)'); plt.xlabel("Frequency (Hz)"); plt.grid(True); plt.legend()
            savename_tf = os.path.join(
                plot_output_dir,
                f"fig_transfer_functions_{station}_{chunk_timestring}_template{template_idx}.jpg"
            )
            plt.savefig(savename_tf, dpi=160)
            print(f"Saved transfer function plot to {savename_tf}")
            plt.close()

        # 6. DENOISE THE FULL STREAM (IN-PLACE)
        print(f"Denoising the full time series in-place (template {template_idx})...")
        ref_trace_full = stream.select(channel=ref_chan)[0]
        fft_ref_full = np.fft.rfft(ref_trace_full.data)
        full_freqs = np.fft.rfftfreq(ref_trace_full.stats.npts, d=ref_trace_full.stats.delta)

        for ch in geophone_chans:
            if ch not in transfer_functions: continue
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
        print("Plotting detailed before-and-after comparison (final)...")
        wide_plot_start_offset_sec = 500
        wide_plot_duration_sec = 30
        zoom_plot_start_offset_sec = 20
        zoom_plot_duration_sec = 3

        if original_stream_for_plotting and original_stream_for_plotting[0].stats.npts / original_stream_for_plotting[0].stats.sampling_rate > wide_plot_start_offset_sec + wide_plot_duration_sec:
            base_time = original_stream_for_plotting[0].stats.starttime
            wide_start_time = base_time + wide_plot_start_offset_sec
            wide_end_time = wide_start_time + wide_plot_duration_sec
            zoom_start_time = wide_start_time + zoom_plot_start_offset_sec
            zoom_end_time = zoom_start_time + zoom_plot_duration_sec

            fig, axes = plt.subplots(len(geophone_chans), 2, figsize=(20, 12), squeeze=False)
            fig.suptitle(f'Denoising Comparison for {station}: Wide and Zoomed Views', fontsize=16)

            for i, ch in enumerate(geophone_chans):
                ax_wide = axes[i, 0]
                ax_zoom = axes[i, 1]
                
                original_tr_wide = original_stream_for_plotting.select(channel=ch)[0].copy().trim(wide_start_time, wide_end_time)
                original_tr_zoom = original_tr_wide.copy().trim(zoom_start_time, zoom_end_time)

                time_axis_wide = original_tr_wide.times("matplotlib")
                time_axis_zoom = original_tr_zoom.times("matplotlib")

                ax_wide.plot(time_axis_wide, original_tr_wide.data, 'k-', linewidth=0.5, alpha=0.7, label='Raw')

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

            axes[-1, 0].set_xlabel('Time'); axes[-1, 1].set_xlabel('Time')
            fig.autofmt_xdate()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            savename_comp = os.path.join(plot_output_dir, f"fig_denoising_comparison_detailed_{station}_{chunk_timestring}.jpg")
            plt.savefig(savename_comp, dpi=160)
            print(f"Saved detailed comparison plot to {savename_comp}")
            plt.close()
        else:
            warnings.warn("Stream is too short to generate comparison plot.")

    print("Denoising complete. The input Stream object has been modified.")
    return None


def _finalise_figure(fig, **kwargs):  # pragma: no cover
    """
    Internal function to wrap up a figure.
    {plotting_kwargs}
    """

    title = kwargs.get("title")
    show = kwargs.get("show", True)
    save = kwargs.get("save", False)
    savefile = kwargs.get("savefile", "EQcorrscan_figure.png")
    return_figure = kwargs.get("return_figure", False)
    size = kwargs.get("size", (10.5, 7.5))
    fig.set_size_inches(size)
    if title:
        fig.suptitle(title)
    if save:
        fig.savefig(savefile, bbox_inches="tight")
        # Logger.info("Saved figure to {0}".format(savefile))
    if show:
        plt.show(block=True)
    if return_figure:
        return fig
    try:
        fig.clf()
        plt.close(fig)
    except Exception as e:
        import tkinter
        if not (isinstance(e, tkinter.TclError) and 'invalid command name' in str(e)):
            raise
        # Otherwise, ignore
    return None


def detection_multiplot(stream, template=None, times=None, party=None,
                        streamcolour='k', cmap='tab10', template_labels=None,
                        **kwargs):
    """
    Plot a stream with one-or-more templates overlaid at detection times.
    Usage:
      - Old (single template): detection_multiplot(stream, template=Stream, times=[UTCDateTime,...], ...)
      - Multi-template: detection_multiplot(stream, template=[Stream1,Stream2], times=[[t1...],[t2...]], ...)
      - Party (eqcorrscan Party): detection_multiplot(stream, party=party_object, ...)
        For a Party, each Family in the Party is used: Family.template.st is the template Stream,
        Family.detections yields Detection objects with detect_time attribute.
    template / times are ignored if party is provided.
    """

    # --- Build templates and times_list from party if provided ---
    if party is not None:
        templates = []
        times_list = []
        template_labels = template_labels or []
        for fam in party:
            fam_tpl = getattr(fam, 'template', None)
            if fam_tpl is None:
                continue
            # Template stream is either fam.template.st or fam.template (already a Stream)
            tpl_st = fam_tpl.st if hasattr(fam_tpl, 'st') else fam_tpl
            templates.append(tpl_st)
            dets = getattr(fam, 'detections', []) or getattr(fam, 'dets', [])
            # extract detect_time (support different attribute names)
            det_times = []
            for d in dets:
                t = getattr(d, 'detect_time', None) or getattr(d, 'detect_time_utc', None) or getattr(d, 'time', None)
                if t is not None:
                    det_times.append(t)
            times_list.append(det_times)
            # label fallback
            lbl = getattr(fam_tpl, 'name', None) or getattr(fam, 'name', None) or f"Template_{len(templates)}"
            template_labels.append(lbl)
    else:
        # --- Normalize template(s) and times argument ---
        if template is None:
            raise ValueError("Either template+times or party must be provided.")
        if isinstance(template, (list, tuple)):
            templates = list(template)
        else:
            templates = [template]
        # times can be list-of-lists or single list to broadcast
        if times is None:
            times_list = [[] for _ in templates]
        elif isinstance(times, (list, tuple)) and len(templates) > 1 and all(isinstance(t, (list, tuple)) for t in times):
            times_list = [list(t) for t in times]
        else:
            times_list = [list(times) for _ in templates]
        if template_labels is None:
            template_labels = [f"Template {i+1}" for i in range(len(templates))]

    # collect unique station+channel traces across all templates preserving order
    template_stachans = []
    for tpl in templates:
        template_stachans.extend([(tr.stats.station, tr.stats.channel) for tr in tpl])
    seen = set()
    unique_stachans = []
    for stachan in template_stachans:
        if stachan not in seen:
            seen.add(stachan)
            unique_stachans.append(stachan)

    if len(unique_stachans) == 0:
        raise ValueError("No traces present in template(s).")

    # Ensure mean is removed from stream
    stream = stream.detrend('demean')
    # --- Prepare figure/axes ---
    ntraces = len(unique_stachans)
    fig, axes = plt.subplots(ntraces, 1, sharex=True)
    if ntraces == 1:
        axes = [axes]

    # determine global mintime across all template traces for alignment
    all_template_traces = [tr for tpl in templates for tr in tpl]
    mintime = min([tr.stats.starttime for tr in all_template_traces])
    print(f'Mintime: {mintime}')
    # color list for templates
    try:
        cmap_obj = cm.get_cmap(cmap)
        tpl_colors = [cmap_obj(i) for i in range(cmap_obj.N)]
    except Exception:
        tpl_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # make deterministic per-template color list (wrap if necessary)
    tpl_color_list = [tpl_colors[i % len(tpl_colors)] for i in range(len(templates))]

    # --- Loop axes / stachans and plot background + overlays ---
    for i, (station, channel) in enumerate(unique_stachans):
        axis = axes[i]
        image = stream.select(station=station, channel='*' + channel[-1])
        if not image:
            print(f'No data for {station} {channel}')
            continue
        image = image.merge()[0]

        # datetime vector for samples
        image_times = [image.stats.starttime.datetime +
                       timedelta((j * image.stats.delta) / 86400)
                       for j in range(len(image.data))]
        print(f'Image time start: {image_times[0]}')
        # normalize background safely
        denom = max(np.abs(image.data)) if max(np.abs(image.data)) != 0 else 1.0
        image_norm = image.data / denom
        axis.plot(image_times, image_norm, streamcolour, linewidth=1.2)

        handles = []
        # For each template, find matching trace and plot each detection overlay
        for tpl_idx, tpl in enumerate(templates):
            color = tpl_color_list[tpl_idx]
            tpl_tr = None
            for tr in tpl:
                if (tr.stats.station, tr.stats.channel) == (station, channel):
                    tpl_tr = tr
                    break
            if tpl_tr is None:
                continue
            any_plotted = False
            for det_time in times_list[tpl_idx]:
                # compute lag to align template with detection time
                lagged_time = (UTCDateTime(det_time) + (tpl_tr.stats.starttime - tpl[0].stats.starttime))
                # lagged_time = (UTCDateTime(det_time) + (tpl_tr.stats.starttime - mintime))
                lagged_dt = lagged_time.datetime
                template_times = [lagged_dt + timedelta((j * tpl_tr.stats.delta) / 86400)
                                  for j in range(len(tpl_tr.data))]
                # normalize template against local image segment (safe)
                try:
                    start_idx = int((template_times[0] - image_times[0]).total_seconds() / image.stats.delta)
                    end_idx = int((template_times[-1] - image_times[0]).total_seconds() / image.stats.delta)
                    if start_idx < 0 or end_idx <= start_idx:
                        raise ValueError
                    segment = image.data[max(0, start_idx):min(len(image.data), end_idx)]
                    normalizer = max(np.abs(segment)) if len(segment) > 0 and max(np.abs(segment)) != 0 else denom
                except Exception as e:
                    print(e)
                    normalizer = denom
                tpl_denom = max(np.abs(tpl_tr.data)) if max(np.abs(tpl_tr.data)) != 0 else 1.0
                scale = normalizer / tpl_denom
                axis.plot(template_times, (tpl_tr.data * scale) / denom, color=color, linewidth=1.2, alpha=0.9)
                any_plotted = True

            if any_plotted:
                label = template_labels[tpl_idx] if tpl_idx < len(template_labels) else f"Template {tpl_idx+1}"
                handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=label))

        # axis label and legend (include stream handle)
        ylab = f"{station}.{channel}"
        axis.set_ylabel(ylab, rotation=0, horizontalalignment='right')
        if handles:
            bg_handle = Line2D([0], [0], color=streamcolour, linewidth=1.2, label='Stream')
            unique_handles = [bg_handle]
            seen_lbls = set()
            for h in handles:
                if h.get_label() not in seen_lbls:
                    unique_handles.append(h)
                    seen_lbls.add(h.get_label())
            axis.legend(handles=unique_handles, loc='upper right', fontsize=8)

    # final formatting
    if ntraces > 1:
        axes[-1].set_xlabel('Time')
    else:
        axes[0].set_xlabel('Time')
    plt.subplots_adjust(hspace=0, left=0.175, right=0.95, bottom=0.07)
    plt.xticks(rotation=10)
    fig = _finalise_figure(fig=fig, **kwargs)  # pragma: no cover
    return fig


def process_date_range(start_date, end_date, tribe, client, params):
    """
    Process a specific date range for matched filtering.
    """
    current_date = start_date
    while current_date < end_date:
        print(f"\n--- Processing {current_date.strftime('%Y-%m-%d')} ---")
        day_start = current_date
        day_end = current_date + 86400  # Process one day at a time

        try:
            # 1. Download data (or read from cache)
            cache_fname = f"{params['network']}.{params['station']}.{current_date.strftime('%j')}.mseed"
            cache_path = os.path.join(params['waveform_cache'], cache_fname)

            if os.path.exists(cache_path):
                print(f"Reading cached daily data: {cache_path}")
                st = read(cache_path)
            else:
                print("Downloading daily data...")
                st = client.get_waveforms(
                    params['network'], params['station'], '*', ",".join(params['channels']),
                    day_start, day_end
                )
                st.write(cache_path, format="MSEED")
                print(f"Wrote daily data to cache: {cache_path}")

            st.merge(fill_value='interpolate')
            st.detrend('demean')
            
            if 'HITP2' in [tr.stats.station for tr in st]:
                st.resample(1000.)

            # 2. Denoise the stream in chunks, build a new stream
            print("Applying spike removal...")
            chunk_length = params['chunk_length_sec']
            st_denoised = Stream()

            chunk_start = day_start
            while chunk_start < day_end:
                chunk_end = min(chunk_start + chunk_length, day_end)
                print(f"Denoising chunk: {chunk_start} to {chunk_end}")

                chunk = st.slice(chunk_start, chunk_end)

                remove_HITP_spikes(
                    stream=chunk,
                    spike_template_path=params['spike_template_path'],
                    geophone_chans=['GPZ'],
                    plot=True,
                    plot_output_dir=params['plot_output_dir'],
                    chunk_start=chunk_start,
                )

                st_denoised += chunk
                chunk_start = chunk_end

            st_denoised.merge(fill_value='interpolate')
            st = st_denoised

            # 3. Pre-process the denoised data
            print("Running standard dayproc pre-processing...")
            st_processed = multi_process(
                st, lowcut=tribe[0].lowcut, highcut=tribe[0].highcut,
                filt_order=tribe[0].filt_order, samp_rate=tribe[0].samp_rate, parallel=True
            )

            # 4. Run matched filtering
            print("Running matched-filter...")
            detections = match_filter(
                template_list=[t.st for t in tribe],
                template_names=[t.name for t in tribe],
                st=st_processed,
                threshold=params['cc_thresh'],
                threshold_type='MAD',
                trig_int=params['trig_int'],
            )

            print(f"Found {len(detections)} detections for this day.")

            # Create and save a Party for the current day
            daily_party = None
            fam_dict = None
            if detections:
                fam_dict = {t.name: Family(template=t) for t in tribe}
                for d in detections:
                    fam_dict[d.template_name] += d

                daily_party = Party(families=[fam for fam in fam_dict.values() if len(fam.detections) > 0])
                daily_party.decluster(params['trig_int'])

                daily_output_path = os.path.join(
                    params['party_output_dir'], f"detections_{current_date.strftime('%Y%m%d')}.tgz"
                )
                if len(daily_party) > 0:
                    print(f"Saving {len(daily_party)} detections to {daily_output_path}...")
                    daily_party.write(daily_output_path)
                else:
                    print("No families with detections, not writing a file.")
            else:
                print("No detections found for this day.")

            # --- NEW: Plot random subset of individual detections ---
            if detections and fam_dict:
                rng = random.Random(params['random_seed'])
                det_list = list(detections)
                rng.shuffle(det_list)
                n_plot = min(params['n_random_det_plots'], len(det_list))

                print(f"Plotting {n_plot} random detections...")
                for det in det_list[:n_plot]:
                    fam = fam_dict.get(det.template_name)
                    if fam is None:
                        continue
                    template = fam.template

                    template_start = min(tr.stats.starttime for tr in template.st)
                    template_end = max(tr.stats.endtime for tr in template.st)
                    template_duration = template_end - template_start
                    padding = 0.1 * template_duration
                    plot_start = det.detect_time - padding
                    plot_end = det.detect_time + template_duration + padding

                    plot_stream = st.slice(starttime=plot_start, endtime=plot_end).copy()
                    plot_stream.detrend('linear')
                    plot_stream.taper(0.05)
                    plot_stream.filter('bandpass', freqmin=template.lowcut, freqmax=template.highcut, corners=4)
                    plot_stream.detrend('linear')

                    det_id = getattr(det, "id", None) or f"{det.template_name}_{det.detect_time.strftime('%Y%m%dT%H%M%S')}"
                    detection_multiplot(
                        stream=plot_stream,
                        template=template.st,
                        times=[det.detect_time],
                        template_labels=[det.template_name],
                        streamcolour='k',
                        templatecolour='r',
                        show=False,
                        save=True,
                        savefile=os.path.join(params['detection_plot_dir'], f"{det_id}.png")
                    )

            # --- NEW: One multiplot for a random window of denoised data ---
            if daily_party and len(daily_party) > 0:
                rng = random.Random(params['random_seed'] + 1)
                max_start = max(0, int((day_end - day_start) - params['random_window_sec']))
                rand_offset = rng.randint(0, max_start) if max_start > 0 else 0
                win_start = day_start + rand_offset
                win_end = win_start + params['random_window_sec']

                window_stream = st.slice(starttime=win_start, endtime=win_end).copy()

                # Build a sub-party with detections inside the window
                sub_fams = []
                for fam in daily_party.families:
                    new_fam = Family(template=fam.template)
                    for d in fam.detections:
                        if win_start <= d.detect_time <= win_end:
                            new_fam += d
                    if len(new_fam.detections) > 0:
                        sub_fams.append(new_fam)

                if sub_fams:
                    sub_party = Party(families=sub_fams)
                    outpath = os.path.join(
                        params['multiplot_dir'],
                        f"multiplot_{current_date.strftime('%Y%m%d')}_{win_start.strftime('%H%M%S')}_{int(params['random_window_sec'])}s.png"
                    )
                    detection_multiplot(
                        stream=window_stream,
                        party=sub_party,
                        streamcolour='k',
                        show=False,
                        save=True,
                        savefile=outpath
                    )
                    print(f"Saved random-window multiplot: {outpath}")

        except Exception as e:
            print(f"Error processing day {current_date.strftime('%Y-%m-%d')}: {e}")

        current_date += 86400  # Move to the next day

    print("\n--- Processing complete. ---")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run daily matched filtering for a specific SLURM task.")
    parser.add_argument("--splits", type=int, required=True, help="Total number of splits (SLURM array size).")
    parser.add_argument("--instance", type=int, required=True, help="SLURM array task ID.")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD).")
    args = parser.parse_args()

    # Convert start and end dates to UTCDateTime
    start_date = UTCDateTime(args.start)
    end_date = UTCDateTime(args.end)

    # Calculate the time range for this task
    total_duration = (end_date - start_date) / args.splits
    task_start = start_date + args.instance * total_duration
    task_end = min(start_date + (args.instance + 1) * total_duration, end_date)

    # --- USER-DEFINED PARAMETERS ---
    params = {
        'network': '6K',
        'station': 'HITP,HITP2',
        'channels': ['GK1', 'GPZ'],
        'cc_thresh': 10.0,
        'trig_int': 0.75,
        'spike_template_path': [
            "/global/home/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
            "/global/home/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
        ],
        'party_output_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/daily_parties",
        'plot_output_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/denoiser_plots",
        'waveform_cache': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/waveform_cache",
        'detection_plot_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/detection_plots",
        'multiplot_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/multiplots",
        'chunk_length_sec': 1 * 3600,
        'n_random_det_plots': 15,
        'random_window_sec': 60,
        'random_seed': 42,
    }

    os.makedirs(params['party_output_dir'], exist_ok=True)
    os.makedirs(params['plot_output_dir'], exist_ok=True)
    os.makedirs(params['waveform_cache'], exist_ok=True)
    os.makedirs(params['detection_plot_dir'], exist_ok=True)
    os.makedirs(params['multiplot_dir'], exist_ok=True)

    # Load the Tribe
    print("Loading Tribe...")
    tribe = Tribe().read("/global/home/users/chopp/chet-meq/cape_modern/templates/eqcorrscan/HITP_templates_1dayproc_clustered_10-1-25.tgz")
    # Initialize the client
    client = Client("http://131.243.224.19:8085")

    # Process the assigned date range
    process_date_range(task_start, task_end, tribe, client, params)