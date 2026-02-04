#!/usr/bin/python

import warnings
import multiprocessing
import functools
import os
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend

import numpy as np
import matplotlib.pyplot as plt

from obspy import Stream, Trace
from scipy.signal import detrend
from scipy.ndimage import median_filter
from eqcorrscan import Template
from eqcorrscan.core.match_filter import match_filter

# --- Multiprocessing Worker and Initializer for QC ---

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
