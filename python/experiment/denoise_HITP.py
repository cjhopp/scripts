#!/usr/bin/python

import warnings
import multiprocessing
import functools

import numpy as np
import matplotlib.pyplot as plt

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, Trace
from scipy.signal import detrend
from eqcorrscan import Template
from eqcorrscan.core.match_filter import match_filter


def process_detection(det, st, snippet_before_sec, snippet_after_sec, all_chans):
    """
    Worker function to process a single detection. This function is designed to be
    called by a multiprocessing pool.
    """
    # Slice creates a view, so we copy it to a new object. This is much faster
    # than copying the whole stream and then trimming.
    snippet = st.slice(det.detect_time - snippet_before_sec, det.detect_time + snippet_after_sec).copy()

    # Perform quality control checks on the snippet
    expected_npts = int((snippet_before_sec + snippet_after_sec) * snippet[0].stats.sampling_rate)
    if len(snippet) < len(all_chans) or any(abs(tr.stats.npts - expected_npts) > 1 for tr in snippet):
        return None  # Return None for invalid snippets that will be filtered out later

    # Detrend and demean the snippet
    n1_before_samples = int(snippet_before_sec * snippet[0].stats.sampling_rate)
    for tr in snippet:
        tr.data = detrend(tr.data)
        tr.data -= np.mean(tr.data[:n1_before_samples - 10])

    return snippet


def main():
    """
    Main function to detect spikes, calculate transfer functions, and denoise geophone data.
    """
    params = {
        'network': '6K',
        'station': 'HITP',
        'starttime': UTCDateTime("2025-09-01"),
        'endtime': UTCDateTime("2025-09-02T00:00:00"),
        'ccth': 0.97,
        'spike_template_filename': '/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt',
    }

    # Time windows in seconds
    snippet_before_sec = 0.3
    snippet_after_sec = 0.5
    fft_window_sec = 0.5
    fft_starttime_offset_sec = 0.2

    # Define channels
    ref_chan = 'GK1'
    geophone_chans = ['GPZ', 'GP1', 'GP2']
    all_chans = [ref_chan] + geophone_chans

    # 1. DATA AND TEMPLATE LOADING
    client = Client('http://131.243.224.19:8085')
    print(f"Downloading waveform data for channels: {all_chans}...")
    st = client.get_waveforms(
        params['network'], params['station'], '*', "GPZ,GP1,GP2,GK1",
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

    # 2. SPIKE DETECTION
    template = Template(name='spike_template', st=template_st, samp_rate=template_st[0].stats.sampling_rate)
    print(f"Running matched-filter detection on reference channel {ref_chan}...")
    detections = match_filter(template_list=[template_st], template_names=['spike'], st=st.select(channel=ref_chan),
                              threshold=params['ccth'], threshold_type='absolute', trig_int=2.0)
    print(f"Found {len(detections)} initial detections.")

    # 3. EXTRACT AND PROCESS SPIKE SNIPPETS (PARALLELIZED)
    print("Extracting and processing snippets in parallel...")
    
    # Use functools.partial to create a worker function with fixed arguments
    worker = functools.partial(process_detection, st=st, 
                               snippet_before_sec=snippet_before_sec,
                               snippet_after_sec=snippet_after_sec,
                               all_chans=all_chans)
    
    # Use a multiprocessing Pool to map the worker to the detections
    with multiprocessing.Pool() as pool:
        processed_snippets = pool.map(worker, detections)
    
    # Filter out None results from invalid snippets
    good_snippets = [snip for snip in processed_snippets if snip is not None]

    print(f"Found {len(good_snippets)} 'good' spikes after QC.")
    if not good_snippets:
        print("No good spikes found, cannot proceed with analysis.")
        return

    # 4. CALCULATE AVERAGE SPECTRA FOR THE SPIKES
    fs1 = good_snippets[0][0].stats.sampling_rate
    winlen_samples = int(fft_window_sec * fs1)
    freq = np.fft.rfftfreq(winlen_samples, d=1/fs1)

    avg_ffts = {}
    for ch in all_chans:
        all_ffts_for_chan = []
        for snip in good_snippets:
            starttime_cut = snip[0].stats.starttime + fft_starttime_offset_sec
            endtime_cut = starttime_cut + winlen_samples / fs1
            trace_cut = snip.select(channel=ch)[0].copy().trim(starttime_cut, endtime=endtime_cut)
            trace_cut.taper(0.04)
            all_ffts_for_chan.append(np.fft.rfft(trace_cut.data))
        avg_ffts[ch] = np.mean(all_ffts_for_chan, axis=0)

    # 5. CALCULATE AND PLOT TRANSFER FUNCTIONS
    print("Calculating and plotting transfer functions...")
    transfer_functions = {}
    plt.figure(figsize=(10, 8))
    for ch in geophone_chans:
        transfer_functions[ch] = avg_ffts[ch] / avg_ffts[ref_chan]
        plt.subplot(2,1,1)
        plt.plot(freq, np.real(transfer_functions[ch]), label=f"Real({ch}/{ref_chan})")
        plt.title('Transfer Functions (Real Part)'); plt.grid(True); plt.legend()
        plt.subplot(2,1,2)
        plt.plot(freq, np.imag(transfer_functions[ch]), label=f"Imag({ch}/{ref_chan})")
        plt.title('Transfer Functions (Imaginary Part)'); plt.xlabel("Frequency (Hz)"); plt.grid(True); plt.legend()

    savename_tf = f"fig_transfer_functions_{params['station']}.jpg"
    plt.savefig(savename_tf, dpi=160)
    print(f"Saved transfer function plot to {savename_tf}")
    plt.close()

    # 6. DENOISE THE FULL HOURLY DATA
    print("Denoising the full time series...")
    st_denoised = st_original_full.copy()
    
    ref_trace_full = st_original_full.select(channel=ref_chan)[0]
    fft_ref_full = np.fft.rfft(ref_trace_full.data)
    full_freqs = np.fft.rfftfreq(ref_trace_full.stats.npts, d=ref_trace_full.stats.delta)

    for ch in geophone_chans:
        try:
            trace_index = [tr.stats.channel for tr in st_denoised].index(ch)
        except ValueError:
            warnings.warn(f"Channel {ch} not found to be denoised. Skipping.")
            continue

        trace_to_denoise = st_denoised[trace_index]
        
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
    
    # 7. PLOT DETAILED BEFORE-AND-AFTER COMPARISON
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
