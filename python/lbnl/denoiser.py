#!/usr/bin/python

import warnings
import multiprocessing
import functools
import os
import numpy as np
import matplotlib.pyplot as plt
from obspy import Stream, Trace
from scipy.signal import detrend
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

def remove_HITP_spikes(
    stream,
    spike_template_path,
    geophone_chans=['GPZ', 'GP1', 'GP2'],
    ccth=0.97,
    num_quiet_spikes=36,
    min_quiet_spikes=10, # MODIFIED: Added minimum threshold
    low_freq_cutoff=2.0,
    plot=False,
    plot_output_dir='.',
    chunk_start=None):
    """
    Removes electronic spike noise from an ObsPy Stream from station HITP in-place.

    Args:
        stream (obspy.Stream): 
            The Stream object to be modified.
        spike_template_path (str): 
            Path to the spike template file.
        geophone_chans (list, optional): 
            A list of the geophone channel names to denoise. 
            Defaults to ['GPZ', 'GP1', 'GP2'].
        ccth (float, optional): 
            Cross-correlation threshold for spike detection. Defaults to 0.97.
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

    original_stream_for_plotting = stream.copy() if plot else None

    if plot:
        if not os.path.exists(plot_output_dir):
            os.makedirs(plot_output_dir)
        timestring = stream[0].stats.starttime.strftime("%Y%m%dT%H%M%S")

    # 2. LOAD TEMPLATE
    try:
        template_data = np.loadtxt(spike_template_path).flatten()
        template_st = Stream(Trace(data=template_data, header={'network': stream[0].stats.network, 'station': station, 'channel': ref_chan, 'sampling_rate': stream[0].stats.sampling_rate}))
    except Exception as e:
        raise IOError(f"Error reading template file: {e}")

    # 3. DETECT SPIKES
    print(f"Running matched-filter detection on {ref_chan}...")
    detections = match_filter(template_list=[template_st], template_names=['spike'], st=stream.select(channel=ref_chan), threshold=ccth, threshold_type='absolute', trig_int=2.0)
    print(f"Found {len(detections)} initial detections.")
    if not detections:
        warnings.warn("No detections found. Stream will not be modified.")
        return

    # 4. QC: FIND QUIETEST SPIKES (PARALLELIZED)
    snippet_before_sec=0.3
    snippet_after_sec=0.5
    cpu_cores = os.cpu_count()
    print(f"Finding quietest spikes in parallel using {cpu_cores} cores...")
    worker = functools.partial(_get_geophone_amplitude, snippet_before_sec=snippet_before_sec, snippet_after_sec=snippet_after_sec, geophone_chans=geophone_chans)

    with multiprocessing.Pool(initializer=_init_worker, initargs=(stream,)) as pool:
        detection_amplitudes = pool.map(worker, detections)

    valid_detections = [item for item in detection_amplitudes if np.isfinite(item[1])]
    sorted_detections = sorted(valid_detections, key=lambda x: x[1])

    num_to_select = min(num_quiet_spikes, len(sorted_detections))

    # MODIFIED: Add check for minimum number of spikes
    if num_to_select < min_quiet_spikes:
        warnings.warn(f"Found only {num_to_select} quiet spikes, which is fewer than the minimum of {min_quiet_spikes} required. Aborting denoising for this segment.")
        return

    print(f"Selected {num_to_select} quietest spikes for TF calculation.")
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
    
    # --- PLOTTING BLOCK 1: QUIET SPIKES ---
    if plot:
                # Generate a unique timestring for the chunk
        chunk_timestring = chunk_start.strftime("%Y%m%dT%H%M%S") if chunk_start else "unknown_chunk"

        print("Plotting spike stack for quality control...")
        fig_spikes, axes_spikes = plt.subplots(len(all_chans), 1, figsize=(10, 8), sharex=True, sharey=True)
        fig_spikes.suptitle(f'Stacked Quiet Spikes for Station {station}', fontsize=16)
        for i, ch in enumerate(all_chans):
            # Ensure axes_spikes is always indexable
            ax = axes_spikes if len(all_chans) == 1 else axes_spikes[i]
            for snip in quiet_snippets:
                if ch not in [tr.stats.channel for tr in snip]: continue
                trace = snip.select(channel=ch)[0]
                time_axis = trace.times() - snippet_before_sec
                ax.plot(time_axis, trace.data, 'k-', alpha=0.2)
            ax.set_ylabel(ch)
            ax.grid(True)
        (axes_spikes if len(all_chans) == 1 else axes_spikes[-1]).set_xlabel("Time relative to detection (s)")
        savename_spikes = os.path.join(plot_output_dir, f"fig_spike_stack_{station}_{chunk_timestring}.jpg")
        plt.savefig(savename_spikes, dpi=160)
        print(f"Saved spike stack plot to {savename_spikes}")
        plt.close()

    fft_window_sec=0.5
    fft_starttime_offset_sec=0.2
    fs1 = quiet_snippets[0][0].stats.sampling_rate
    winlen_samples = int(fft_window_sec * fs1)
    freq = np.fft.rfftfreq(winlen_samples, d=1 / fs1)

    individual_tfs = {ch: [] for ch in geophone_chans}
    for snip in quiet_snippets:
        starttime_cut = snip[0].stats.starttime + fft_starttime_offset_sec
        endtime_cut = starttime_cut + winlen_samples / fs1
        ref_trace_cut = snip.select(channel=ref_chan)[0].copy().trim(starttime_cut, endtime_cut)
        ref_trace_cut.taper(0.04)
        fft_ref = np.fft.rfft(ref_trace_cut.data)
        for ch in geophone_chans:
            if ch not in [tr.stats.channel for tr in snip]: continue
            geo_trace_cut = snip.select(channel=ch)[0].copy().trim(starttime_cut, endtime_cut)
            geo_trace_cut.taper(0.04)
            fft_geo = np.fft.rfft(geo_trace_cut.data)
            tf = np.divide(fft_geo, fft_ref, out=np.zeros_like(fft_geo), where=fft_ref!=0)
            individual_tfs[ch].append(tf)

    transfer_functions = {}
    for ch in geophone_chans:
        if not individual_tfs[ch]: continue
        tfs_stack = np.array(individual_tfs[ch])
        median_real = np.median(np.real(tfs_stack), axis=0)
        median_imag = np.median(np.imag(tfs_stack), axis=0)
        median_tf = median_real + 1j * median_imag
        median_tf[freq <= low_freq_cutoff] = 0 + 0j
        transfer_functions[ch] = median_tf

    # --- PLOTTING BLOCK 2: TRANSFER FUNCTIONS ---
    if plot:
        print("Plotting final median transfer functions...")
        plt.figure(figsize=(10, 8))
        for ch in geophone_chans:
            if ch not in transfer_functions: continue
            plt.subplot(2, 1, 1)
            plt.plot(freq, np.real(transfer_functions[ch]), label=f"Real({ch}/{ref_chan})")
            plt.title('Median Transfer Functions (Real Part)'); plt.grid(True); plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(freq, np.imag(transfer_functions[ch]), label=f"Imag({ch}/{ref_chan})")
            plt.title('Median Transfer Functions (Imaginary Part)'); plt.xlabel("Frequency (Hz)"); plt.grid(True); plt.legend()
        savename_tf = os.path.join(plot_output_dir, f"fig_transfer_functions_{station}_{chunk_timestring}.jpg")
        plt.savefig(savename_tf, dpi=160)
        print(f"Saved transfer function plot to {savename_tf}")
        plt.close()

    # 6. DENOISE THE FULL STREAM (IN-PLACE)
    print("Denoising the full time series in-place...")
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

    # --- PLOTTING BLOCK 3: COMPARISON PLOT ---
    if plot:
        print("Plotting detailed before-and-after comparison...")
        wide_plot_start_offset_sec = 500
        wide_plot_duration_sec = 60
        zoom_plot_start_offset_sec = 20
        zoom_plot_duration_sec = 5

        if original_stream_for_plotting and original_stream_for_plotting[0].stats.npts / original_stream_for_plotting[0].stats.sampling_rate > wide_plot_start_offset_sec + wide_plot_duration_sec:
            base_time = original_stream_for_plotting[0].stats.starttime
            wide_start_time = base_time + wide_plot_start_offset_sec
            wide_end_time = wide_start_time + wide_plot_duration_sec
            zoom_start_time = wide_start_time + zoom_plot_start_offset_sec
            zoom_end_time = zoom_start_time + zoom_plot_duration_sec

            fig, axes = plt.subplots(len(geophone_chans), 2, figsize=(20, 12), squeeze=False) # MODIFIED: squeeze=False

            fig.suptitle(f'Denoising Comparison for {station}: Wide and Zoomed Views', fontsize=16)

            for i, ch in enumerate(geophone_chans):
                ax_wide = axes[i, 0]
                ax_zoom = axes[i, 1]
                
                original_tr_wide = original_stream_for_plotting.select(channel=ch)[0].copy().trim(wide_start_time, wide_end_time)
                denoised_tr_wide = stream.select(channel=ch)[0].copy().trim(wide_start_time, wide_end_time)
                original_tr_zoom = original_tr_wide.copy().trim(zoom_start_time, zoom_end_time)
                denoised_tr_zoom = denoised_tr_wide.copy().trim(zoom_start_time, zoom_end_time)

                time_axis_wide = original_tr_wide.times("matplotlib")
                ax_wide.plot(time_axis_wide, original_tr_wide.data, 'k-', linewidth=0.5, alpha=0.7, label='Original')
                ax_wide.plot(time_axis_wide, denoised_tr_wide.data, 'r-', linewidth=0.8, alpha=0.8, label='Denoised')
                ax_wide.set_title(f'Channel {ch} - Wide View')
                ax_wide.set_ylabel('Amplitude')
                ax_wide.grid(True)

                zoom_start_mpl = original_tr_zoom.times("matplotlib")[0]
                zoom_end_mpl = original_tr_zoom.times("matplotlib")[-1]
                ax_wide.axvspan(zoom_start_mpl, zoom_end_mpl, color='blue', alpha=0.2, label='Zoom Area')
                ax_wide.legend()

                time_axis_zoom = original_tr_zoom.times("matplotlib")
                ax_zoom.plot(time_axis_zoom, original_tr_zoom.data, 'k-', linewidth=0.5, alpha=0.7, label='Original')
                ax_zoom.plot(time_axis_zoom, denoised_tr_zoom.data, 'r-', linewidth=0.8, alpha=0.8, label='Denoised')
                ax_zoom.set_title(f'Zoomed View')
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
