#!/usr/bin/python

import warnings

import numpy as np
import matplotlib.pyplot as plt

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, Stream, Trace
from scipy.signal import detrend
from eqcorrscan import Template
from eqcorrscan.core.match_filter import match_filter

def main():
    """
    Main function to detect spikes, calculate transfer functions, and denoise geophone data.
    """
    params = {
        'network': '6K',
        'station': 'HITP',
        'starttime': UTCDateTime("2025-09-01"),
        'endtime': UTCDateTime("2025-09-01T01:00:00"), # Use one hour of data
        'ccth': 0.97,
        'spike_template_filename': '/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt',
    }

    # Time windows in seconds, as suggested
    snippet_before_sec = 0.3
    snippet_after_sec = 0.5
    fft_window_sec = 0.5
    fft_starttime_offset_sec = 0.2

    # Define reference and target channels
    ref_chan = 'GK1'
    geophone_chans = ['GPZ', 'GP1', 'GP2']
    all_chans = [ref_chan] + geophone_chans

    # 1. DATA AND TEMPLATE LOADING
    client = Client('http://131.243.224.19:8085')
    print(f"Downloading waveform data for channels: {all_chans}...")
    st = Stream()
    st = client.get_waveforms(
        params['network'], params['station'], '*', "GPZ,GP1,GP2,GK1",
        params['starttime'], params['endtime']
    )
    st.merge(fill_value='interpolate')
    st_original_full = st.copy()

    print("Loading and processing template...")
    try:
        template_data = np.loadtxt(params['spike_template_filename']).flatten()
        template_st = Stream(Trace(data=template_data, header={'network': '6K', 'station': 'HITP', 'channel': ref_chan, 'sampling_rate': st[0].stats.sampling_rate}))
    except Exception as e:
        print(f"Error reading template file: {e}. Please update the path in `params`.")
        return

    # 2. SPIKE DETECTION USING EQCORRSCAN
    template = Template(name='spike_template', st=template_st,
                        samp_rate=template_st[0].stats.sampling_rate)

    print(f"Running matched-filter detection on reference channel {ref_chan}...")
    detections = match_filter(template_list=[template_st], template_names=['spike'], st=st.select(channel=ref_chan),
                              threshold=params['ccth'], threshold_type='absolute', trig_int=2.0)
    print(f"Found {len(detections)} initial detections.")

    # 3. EXTRACT AND PROCESS SPIKE SNIPPETS
    good_snippets = []
    for det in detections:
        snippet = st.copy().trim(det.detect_time - snippet_before_sec, det.detect_time + snippet_after_sec)
        
        # Check if snippet has all channels and correct length
        expected_npts = int((snippet_before_sec + snippet_after_sec) * snippet[0].stats.sampling_rate)
        if len(snippet) < len(all_chans) or any(abs(tr.stats.npts - expected_npts) > 1 for tr in snippet):
            continue
        
        n1_before_samples = int(snippet_before_sec * snippet[0].stats.sampling_rate)
        for tr in snippet:
            tr.data = detrend(tr.data)
            tr.data -= np.mean(tr.data[:n1_before_samples - 10])
        good_snippets.append(snippet)

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
        plt.title('Transfer Functions (Real Part)')
        plt.grid(True); plt.legend()
        plt.subplot(2,1,2)
        plt.plot(freq, np.imag(transfer_functions[ch]), label=f"Imag({ch}/{ref_chan})")
        plt.title('Transfer Functions (Imaginary Part)')
        plt.xlabel("Frequency (Hz)"); plt.grid(True); plt.legend()

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

    # Correctly loop through geophone channels and modify traces in-place
    for ch in geophone_chans:
        try:
            # Find the index of the trace we want to modify in the stream
            trace_index = [tr.stats.channel for tr in st_denoised].index(ch)
        except ValueError:
            warnings.warn(f"Channel {ch} not found in the stream to be denoised. Skipping.")
            continue

        # Get a direct reference to the trace object, not a copy
        trace_to_denoise = st_denoised[trace_index]
        
        # Check for length mismatch to prevent broadcasting errors
        if trace_to_denoise.stats.npts != ref_trace_full.stats.npts:
            warnings.warn(f"Channel {ch} has a different length than the reference. Skipping.")
            continue

        # Interpolate transfer function
        tf_interpolated = np.interp(full_freqs, freq, transfer_functions[ch], left=0, right=0)
        # Predict noise in the frequency domain
        fft_predicted_noise = fft_ref_full * tf_interpolated
        # Convert predicted noise back to the time domain
        predicted_noise_time = np.fft.irfft(fft_predicted_noise, n=ref_trace_full.stats.npts)
        # --- FIX THE DTYPE AND SUBTRACTION ---
        # 1. Ensure the original data is float64 to preserve precision.
        trace_to_denoise.data = trace_to_denoise.data.astype(np.float64)
        # 2. Subtract the float64 predicted noise. THIS MODIFIES THE TRACE IN-PLACE.
        trace_to_denoise.data -= predicted_noise_time

    # Save the now-modified stream, ensuring to save as float to keep precision
    savename_denoised = f"denoised_data_{params['station']}.mseed"
    st_denoised.write(savename_denoised, format="MSEED")
    print(f"Saved denoised data to {savename_denoised}")
    
    # Plot a before-and-after comparison
    print("Plotting before-and-after comparison...")
    plot_start = st_original_full[0].stats.starttime + 100
    plot_end = plot_start + 30
    
    fig, axes = plt.subplots(len(geophone_chans), 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Denoising Comparison', fontsize=16)

    for i, ch in enumerate(geophone_chans):
        ax = axes[i]
        original_tr = st_original_full.select(channel=ch)[0].copy().trim(plot_start, plot_end)
        denoised_tr = st_denoised.select(channel=ch)[0].copy().trim(plot_start, plot_end)
        time_axis = original_tr.times("matplotlib")
        ax.plot(time_axis, original_tr.data, 'k-', label='Original')
        ax.plot(time_axis, denoised_tr.data, 'r-', label='Denoised')
        ax.set_ylabel(ch); ax.legend(); ax.grid(True)
        
    axes[-1].xaxis_date()
    fig.autofmt_xdate()
    savename_comp = f"fig_denoising_comparison_{params['station']}.jpg"
    plt.savefig(savename_comp, dpi=160)
    print(f"Saved comparison plot to {savename_comp}")
    plt.close()

if __name__ == '__main__':
    main()
