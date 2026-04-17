#!/usr/bin/python
# filepath: /home/chopp/scripts/python/experiment/plot_party_detections.py
import sys
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, Stream
from eqcorrscan import Tribe, Party, Family
from eqcorrscan.utils.pre_processing import multi_process

sys.path.insert(0, '/home/chopp/scripts/python')
from lbnl.denoiser import remove_HITP_spikes
from lbnl.waveforms import detection_multiplot


def main():
    """
    Load an existing Party, download/cache waveforms, and plot random detections.
    """
    # --- USER-DEFINED PARAMETERS ---
    PARTY_PATH = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/All_detections_HITP_MAD20.tgz"
    TRIBE_PATH = "/media/chopp/HDD1/chet-meq/cape_modern/templates/eqcorrscan/HITP2_templates_1dayproc_clustered_10-1-25.tgz"
    
    SPIKE_TEMPLATE_PATH = [
        "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
        "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
    ]
    
    WAVEFORM_CACHE = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/waveform_cache"
    MULTIPLOT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/detection_plots/MAD20"
    
    # Data-source parameters
    CLIENT_URL = 'http://131.243.224.19:8085'
    NETWORK = '6K'
    STATION = 'HITP,HITP2'
    CHANNELS = ['GK1', 'GPZ']
    
    # Plotting parameters
    N_RANDOM_DET_PLOTS = 100
    RANDOM_SEED = 42
    
    # Processing parameters
    PROCESS_CORES = 30  # Number of cores for multiprocessing waveform processing

    # --- SETUP ---
    os.makedirs(MULTIPLOT_DIR, exist_ok=True)
    os.makedirs(WAVEFORM_CACHE, exist_ok=True)
    
    print("Loading Party...")
    try:
        party = Party().read(PARTY_PATH, read_detection_catalog=False, estimate_origin=False)
        print(f"Loaded Party with {len(party)} detections across {len(party.families)} families.")
    except Exception as e:
        print(f"Error: Could not load Party from {PARTY_PATH}. Aborting.")
        print(e)
        return
    
    print("Loading Tribe...")
    try:
        tribe = Tribe().read(TRIBE_PATH)
        print(f"Loaded {len(tribe)} templates.")
    except Exception as e:
        print(f"Error: Could not load Tribe from {TRIBE_PATH}. Aborting.")
        print(e)
        return
    
    client = Client(CLIENT_URL)
    
    # --- COLLECT ALL DETECTIONS ---
    all_detections = []
    for family in party.families:
        for det in family.detections:
            all_detections.append((family.template, det))
    
    if len(all_detections) == 0:
        print("No detections found in Party. Exiting.")
        return
    
    print(f"Total detections in Party: {len(all_detections)}")
    
    # --- SELECT RANDOM SUBSET ---
    random.seed(RANDOM_SEED)
    n_to_plot = min(N_RANDOM_DET_PLOTS, len(all_detections))
    random_detections = random.sample(all_detections, n_to_plot)
    
    print(f"Selected {n_to_plot} random detections for plotting.")
    
    # --- GROUP DETECTIONS BY DAY ---
    day_groups = {}
    for template, det in random_detections:
        day = det.detect_time.date  # Use Python date object instead of UTCDateTime
        if day not in day_groups:
            day_groups[day] = []
        day_groups[day].append((template, det))
    
    print(f"Detections span {len(day_groups)} unique days.")
    
    # --- PROCESS EACH DAY ---
    for day, detections in sorted(day_groups.items()):
        print(f"\n--- Processing day: {day.strftime('%Y-%m-%d')} ({len(detections)} detections) ---")
        
        day_start = UTCDateTime(day)  # Convert back to UTCDateTime for processing
        day_end = day_start + 86400

        try:
            # 1. Download or load cached waveforms
            cache_fname = f"{NETWORK}.{STATION}.{day.strftime('%j')}.mseed"
            cache_path = os.path.join(WAVEFORM_CACHE, cache_fname)
            
            if os.path.exists(cache_path):
                print(f"  Reading cached waveforms: {cache_path}")
                st = read(cache_path)
            else:
                print(f"  Downloading waveforms from {CLIENT_URL}...")
                st = client.get_waveforms(
                    network=NETWORK, station=STATION, location='*',
                    channel=','.join(CHANNELS), starttime=day_start, endtime=day_end
                )
                st.write(cache_path, format='MSEED')
                print(f"  Cached waveforms to: {cache_path}")
            
            st.merge(fill_value='interpolate')
            st.detrend('demean')
            
            # DO NOT rename raw data - keep original station names
            
            # 2. Denoise the stream in chunks serially (remove_HITP_spikes is already parallel)
            print("  Applying spike removal in serial chunks...")
            chunk_length = 1 * 3600  # 1 hour chunks
            
            st_denoised = Stream()
            chunk_start = day_start
            
            while chunk_start < day_end:
                chunk_end = min(chunk_start + chunk_length, day_end)
                st_chunk = st.slice(chunk_start, chunk_end).copy()
                
                print(f"    Denoising chunk {chunk_start.strftime('%H:%M')} - {chunk_end.strftime('%H:%M')}...")
                remove_HITP_spikes(
                    stream=st_chunk,
                    spike_template_path=SPIKE_TEMPLATE_PATH,
                    plot=False,
                    chunk_start=chunk_start,
                    apply_offset_removal=True
                )
                
                st_denoised += st_chunk
                chunk_start = chunk_end
            
            st_denoised.merge(fill_value='interpolate')
            st = st_denoised
            
            # 3. Rename TEMPLATES to match the raw data station names for this time period
            print("  Adjusting template station names to match data...")
            tribe_copy = tribe.copy()
            if day_start > UTCDateTime("2025-10-23T22:38:19.859000Z"):
                # After Oct 23, data has station HITP2, so rename templates
                for template in tribe_copy.templates:
                    for tr in template.st:
                        if tr.stats.station == 'HITP':
                            tr.stats.station = 'HITP2'
            
            # 4. Pre-process stream to match templates
            print("  Pre-processing stream with multiprocessing...")
            # Get processing parameters from first template
            template = tribe_copy[0]  # Changed from tribe_copy.templates[0]
            st_processed = multi_process(
                st=st.copy(),
                lowcut=template.lowcut,
                highcut=template.highcut,
                filt_order=template.filt_order,
                samp_rate=template.samp_rate,
                num_cores=PROCESS_CORES,
                starttime=day_start,
                endtime=day_end,
                seisan_chan_names=False,
                fill_gaps=True,
                ignore_length=True,
                ignore_bad_data=True
            )
            # 5. Plot each detection for this day
            for template_orig, det in detections:
                template = template_orig
                plot_fname = f"detection_{det.id}_{det.detect_time.strftime('%Y%m%d_%H%M%S')}.png"
                plot_path = os.path.join(MULTIPLOT_DIR, plot_fname)

                print(f"  Plotting detection {det.id} at {det.detect_time}...")
                
                try:
                    # Calculate plot window based on template duration
                    template_start = min(tr.stats.starttime for tr in template.st)
                    template_end = max(tr.stats.endtime for tr in template.st)
                    template_duration = template_end - template_start
                    padding = 0.1 * template_duration
                    plot_start = det.detect_time - padding
                    plot_end = det.detect_time + template_duration + padding
                    
                    # Slice and filter the stream for this detection
                    plot_stream = st_processed.slice(starttime=plot_start, endtime=plot_end).copy()
                    plot_stream.detrend('linear')
                    plot_stream.taper(0.05)
                    plot_stream.filter('bandpass', freqmin=template.lowcut, freqmax=template.highcut, corners=4)
                    plot_stream.detrend('linear')
                    
                    detection_multiplot(
                        stream=plot_stream,
                        template=template.st,
                        times=[det.detect_time],
                        streamcolour='k',
                        templatecolour='r',
                        size=(18.5, 10),
                        title=f"Detection {det.id} | {det.detect_time.strftime('%Y-%m-%d %H:%M:%S')} | Threshold: {det.detect_val:.2f}",
                        show=False,
                        save=True,
                        savefile=plot_path
                    )
                    print(f"    Saved: {plot_path}")
                except Exception as e:
                    print(f"    Error plotting detection {det.id}: {e}")
        
        except Exception as e:
            print(f"  Error processing day {day.strftime('%Y-%m-%d')}: {e}")
            continue
    
    print("\n--- Plotting complete. ---")


if __name__ == '__main__':
    main()