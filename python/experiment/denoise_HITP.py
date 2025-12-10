#!/usr/bin/python

import warnings
import multiprocessing
import functools
import os

import numpy as np
import matplotlib.pyplot as plt

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, Trace
from scipy.signal import detrend
from eqcorrscan import Template
from eqcorrscan.core.match_filter import match_filter

# --- Multiprocessing Worker and Initializer for QC ---

shared_st = None

def init_worker(stream_data):
    """
    Initializer for each worker process in the pool.
    This function sets the global 'shared_st' variable for the worker.
    """
    global shared_st
    shared_st = stream_data

def get_geophone_amplitude(det, snippet_before_sec, snippet_after_sec, geophone_chans):
    """
    Worker function to get the max amplitude on geophone channels for a detection.
    Returns a tuple of (detection, max_amplitude).
    """
    global shared_st
    snippet = shared_st.slice(det.detect_time - snippet_before_sec, det.detect_time + snippet_after_sec)

    # Check if snippet contains all required geophone channels
    if any(len(snippet.select(channel=ch)) == 0 for ch in geophone_chans):
        return (det, np.inf) # Return infinity to sort this to the end

    # Find the max amplitude across all geophone channels in the snippet
    max_amp = 0
    for ch in geophone_chans:
        tr = snippet.select(channel=ch)[0]
        max_amp = max(max_amp, np.max(np.abs(tr.data)))
    
    return (det, max_amp)

def main():
    """
    Main function to detect spikes, find the quietest ones, calculate a median
    transfer function, and denoise geophone data.
    """
    params = {
        'network': '6K',
        'station': 'HITP',
        'starttime': UTCDateTime("2025-09-01"),
        'endtime': UTCDateTime("2025-09-02T00:00:00"),
        'ccth': 0.97,
        'spike_template_filename': '/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt',
        'num_quiet_spikes': 36,
    }

    # Time windows
    snippet_before_sec = 0.3
    snippet_after_sec = 0.5
    fft_window_sec = 0.5
    fft_starttime_offset_sec = 0.2
    low_freq_cutoff = 2.0

    # Channels
    ref_chan = 'GK1'
    geophone_chans = ['GPZ', 'GP1', 'GP2']
    all_chans = [ref_chan] + geophone_chans

    # 1. DATA LOADING
    client = Client('http://131.243.224.19:8085')
    print(f"Downloading waveform data for channels: {all_chans}...")
    st = client.get_waveforms(
        params['network'], params['station'], '*', ",".join(all_chans),
        params['starttime'], params['endtime']
    )
    st.merge(fill_value='interpolate')
    st.detrend('demean')
    st_original_full = st.copy()

    print("Loading and processing template...")
    try:
        template_data = np.loadtxt(params['spike_template_filename']).flatten()
        template_st = Stream(Trace(data=template_data, header={'network': '6K', 'station': 'HITP', 'channel': ref_chan, 'sampling_rate': st[0].stats.sampling_rate}))
    except Exception as e:
        print(f"Error reading template file: {e}")
        return

    # 2. SPIKE DETECTION ON REFERENCE CHANNEL
    print(f"Running matched-filter detection on reference channel {ref_chan}...")
    detections = match_filter(template_list=[template_st], template_names=['spike'], st=st.select(channel=ref_chan),
                              threshold=params['ccth'], threshold_type='absolute', trig_int=2.0)
    print(f"Found {len(detections)} initial detections.")

    # 3. QC: FIND THE QUIETEST SPIKES (PARALLELIZED)
    cpu_cores = os.cpu_count()
    print(f"Finding quietest spikes in parallel using {cpu_cores} cores...")

    # Create a partial function for the worker
    worker = functools.partial(get_geophone_amplitude,
                               snippet_before_sec=snippet_before_sec,
                               snippet_after_sec=snippet_after_sec,
                               geophone_chans=geophone_chans)

    with multiprocessing.Pool(initializer=init_worker, initargs=(st,)) as pool:
        # This returns a list of (detection, amplitude) tuples
        detection_amplitudes = pool.map(worker, detections)

    # Filter out any that had missing data
    valid_detections = [item for item in detection_amplitudes if np.isfinite(item[1])]

    # Sort by amplitude (the second element of the tuple)
    sorted_detections = sorted(valid_detections, key=lambda x: x[1])

    num_to_select = min(params['num_quiet_spikes'], len(sorted_detections))
    if num_to_select == 0:
        print("Could not find any valid spikes after QC. Aborting.")
        return

    print(f"Selected the {num_to_select} quietest spikes for TF calculation.")
    quiet_detections = [item[0] for item in sorted_detections[:num_to_select]]

    # 4. PLOT THE QUIET SPIKES
    print("Plotting the selected quiet spike stack...")
    quiet_snippets = []
    for det in quiet_detections:
        snippet = st.slice(det.detect_time - snippet_before_sec, det.detect_time + snippet_after_sec).copy()
        n1_before_samples = int(snippet_before_sec * snippet[0].stats.sampling_rate)
        for tr in snippet:
            tr.data = detrend(tr.data)
            tr.data -= np.mean(tr.data[:n1_before_samples - 10])
        quiet_snippets.append(snippet)

    fig_spikes, axes_spikes = plt.subplots(len(all_chans), 1, figsize=(10, 8), sharex=True, sharey=True)
    fig_spikes.suptitle(f'Stacked Quiet Spikes for Station {params["station"]}', fontsize=16)
    for i, ch in enumerate(all_chans):
        ax = axes_spikes[i]
        for snip in quiet_snippets:
            trace = snip.select(channel=ch)[0]
            time_axis = trace.times() - snippet_before_sec
            ax.plot(time_axis, trace.data, 'k-', alpha=0.2)
        ax.set_ylabel(ch)
        ax.grid(True)
    axes_spikes[-1].set_xlabel("Time relative to detection (s)")
    savename_spikes = f"fig_spike_stack_{params['station']}.jpg"
    plt.savefig(savename_spikes, dpi=160)
    print(f"Saved spike stack plot to {savename_spikes}")
    plt.close()

    # 5. CALCULATE MEDIAN TRANSFER FUNCTION
    print("Calculating median transfer function from quiet spikes...")
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
            geo_trace_cut = snip.select(channel=ch)[0].copy().trim(starttime_cut, endtime=endtime_cut)
            geo_trace_cut.taper(0.04)
            fft_geo = np.fft.rfft(geo_trace_cut.data)
            tf = np.divide(fft_geo, fft_ref, out=np.zeros_like(fft_geo), where=fft_ref!=0)
            individual_tfs[ch].append(tf)

    transfer_functions = {}
    for ch in geophone_chans:
        tfs_stack = np.array(individual_tfs[ch])
        median_real = np.median(np.real(tfs_stack), axis=0)
        median_imag = np.median(np.imag(tfs_stack), axis=0)
        median_tf = median_real + 1j * median_imag
        median_tf[freq <= low_freq_cutoff] = 0 + 0j
        transfer_functions[ch] = median_tf

    # 6. PLOT TRANSFER FUNCTIONS
    print("Plotting final median transfer functions...")
    plt.figure(figsize=(10, 8))
    for ch in geophone_chans:
        plt.subplot(2, 1, 1)
        plt.plot(freq, np.real(transfer_functions[ch]), label=f"Real({ch}/{ref_chan})")
        plt.title('Median Transfer Functions (Real Part)'); plt.grid(True); plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(freq, np.imag(transfer_functions[ch]), label=f"Imag({ch}/{ref_chan})")
        plt.title('Median Transfer Functions (Imaginary Part)'); plt.xlabel("Frequency (Hz)"); plt.grid(True); plt.legend()
    savename_tf = f"fig_transfer_functions_{params['station']}.jpg"
    plt.savefig(savename_tf, dpi=160)
    print(f"Saved transfer function plot to {savename_tf}")
    plt.close()

    # 7. DENOISE THE FULL HOURLY DATA
    print("Denoising the full time series...")
    st_denoised = st_original_full.copy()
    ref_trace_full = st_original_full.select(channel=ref_chan)[0]
    fft_ref_full = np.fft.rfft(ref_trace_full.data)
    full_freqs = np.fft.rfftfreq(ref_trace_full.stats.npts, d=ref_trace_full.stats.delta)

    for ch in geophone_chans:
        trace_to_denoise = st_denoised.select(channel=ch)[0]
        if trace_to_denoise.stats.npts != ref_trace_full.stats.npts:
            warnings.warn(f"Channel {ch} has a different length than reference. Skipping.")
            continue
        tf_interpolated = np.interp(full_freqs, freq, transfer_functions[ch], left=0, right=0)
        fft_predicted_noise = fft_ref_full * tf_interpolated
        predicted_noise_time = np.fft.irfft(fft_predicted_noise, n=ref_trace_full.stats.npts)
        trace_to_denoise.data = trace_to_denoise.data.astype(np.float64)
        trace_to_denoise.data -= predicted_noise_time

    savename_denoised = f"denoised_data_{params['station']}.mseed"
    st_denoised.write(savename_denoised, format="MSEED")
    print(f"Saved denoised data to {savename_denoised}")
    
    # 8. PLOT DETAILED BEFORE-AND-AFTER COMPARISON
    print("Plotting detailed before-and-after comparison...")
    wide_plot_start_offset_sec = 500
    wide_plot_duration_sec = 60
    zoom_plot_start_offset_sec = 20
    zoom_plot_duration_sec = 5
    base_time = st_original_full[0].stats.starttime
    wide_start_time = base_time + wide_plot_start_offset_sec
    wide_end_time = wide_start_time + wide_plot_duration_sec
    zoom_start_time = wide_start_time + zoom_plot_start_offset_sec
    zoom_end_time = zoom_start_time + zoom_plot_duration_sec
    fig, axes = plt.subplots(len(geophone_chans), 2, figsize=(20, 12))
    fig.suptitle('Denoising Comparison: Wide and Zoomed Views', fontsize=16)
    for i, ch in enumerate(geophone_chans):
        ax_wide = axes[i, 0]
        ax_zoom = axes[i, 1]
        original_tr_wide = st_original_full.select(channel=ch)[0].copy().trim(wide_start_time, wide_end_time)
        denoised_tr_wide = st_denoised.select(channel=ch)[0].copy().trim(wide_start_time, wide_end_time)
        original_tr_zoom = original_tr_wide.copy().trim(zoom_start_time, zoom_end_time)
        denoised_tr_zoom = denoised_tr_wide.copy().trim(zoom_start_time, zoom_end_time)
        time_axis_wide = original_tr_wide.times("matplotlib")
        ax_wide.plot(time_axis_wide, original_tr_wide.data, 'k-', linewidth=0.5, alpha=0.5, label='Original')
        ax_wide.plot(time_axis_wide, denoised_tr_wide.data, 'r-', linewidth=0.5, alpha=0.5, label='Denoised')
        ax_wide.set_title(f'Channel {ch} - Wide View ({wide_plot_duration_sec} s)')
        ax_wide.set_ylabel('Amplitude')
        ax_wide.grid(True)
        zoom_start_mpl = original_tr_zoom.times("matplotlib")[0]
        zoom_end_mpl = original_tr_zoom.times("matplotlib")[-1]
        ax_wide.axvspan(zoom_start_mpl, zoom_end_mpl, color='blue', alpha=0.2, label='Zoom Area')
        ax_wide.legend()
        time_axis_zoom = original_tr_zoom.times("matplotlib")
        ax_zoom.plot(time_axis_zoom, original_tr_zoom.data, 'k-', linewidth=0.5, alpha=0.5, label='Original')
        ax_zoom.plot(time_axis_zoom, denoised_tr_zoom.data, 'r-', linewidth=0.5, alpha=0.5, label='Denoised')
        ax_zoom.set_title(f'Zoomed View ({zoom_plot_duration_sec} s)')
        ax_zoom.grid(True)
        ax_zoom.legend()
        ax_wide.xaxis_date()
        ax_zoom.xaxis_date()
    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    savename_comp = f"fig_denoising_comparison_detailed_{params['station']}.jpg"
    plt.savefig(savename_comp, dpi=160)
    print(f"Saved detailed comparison plot to {savename_comp}")
    plt.close()

if __name__ == '__main__':
    main()
