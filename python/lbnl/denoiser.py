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
    min_quiet_spikes=10,
    low_freq_cutoff=2.0,
    plot=False,
    plot_output_dir='.',
    chunk_start=None  # Add chunk_start for unique plot names
):
    """
    Removes electronic spike noise from an ObsPy Stream from station HITP in-place.

    Args:
        stream (obspy.Stream): The Stream object to be modified.
        spike_template_path (str): Path to the spike template file.
        geophone_chans (list, optional): A list of the geophone channel names to denoise.
        ccth (float, optional): Cross-correlation threshold for spike detection.
        num_quiet_spikes (int, optional): Number of quietest spikes for the median transfer function.
        min_quiet_spikes (int, optional): Minimum number of quiet spikes required to proceed with denoising.
        low_freq_cutoff (float, optional): Frequency (Hz) to set the transfer function to zero below.
        plot (bool, optional): If True, generate and save diagnostic plots.
        plot_output_dir (str, optional): Directory to save plots to.
        chunk_start (UTCDateTime, optional): Start time of the chunk for unique plot names.

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
    
    if plot:
        # Generate a unique timestring for the chunk
        chunk_timestring = chunk_start.strftime("%Y%m%dT%H%M%S") if chunk_start else "unknown_chunk"

        # --- PLOTTING BLOCK 1: QUIET SPIKES ---
        print("Plotting spike stack for quality control...")
        fig_spikes, axes_spikes = plt.subplots(len(all_chans), 1, figsize=(10, 8), sharex=True, sharey=True)
        fig_spikes.suptitle(f'Stacked Quiet Spikes for Station {station}', fontsize=16)
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
        savename_spikes = os.path.join(plot_output_dir, f"fig_spike_stack_{station}_{chunk_timestring}.jpg")
        plt.savefig(savename_spikes, dpi=160)
        print(f"Saved spike stack plot to {savename_spikes}")
        plt.close()

        # --- PLOTTING BLOCK 2: TRANSFER FUNCTIONS ---
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

        # --- PLOTTING BLOCK 3: COMPARISON PLOT ---
        print("Plotting detailed before-and-after comparison...")
        savename_comp = os.path.join(plot_output_dir, f"fig_denoising_comparison_detailed_{station}_{chunk_timestring}.jpg")
        # ...existing comparison plot code...
        plt.savefig(savename_comp, dpi=160)
        print(f"Saved detailed comparison plot to {savename_comp}")
        plt.close()

    print("Denoising complete. The input Stream object has been modified.")
    return None
