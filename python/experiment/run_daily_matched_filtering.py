#!/usr/bin/python
import sys
sys.path.insert(0, '/home/chopp/scripts/python')

import os
import random
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, Stream
from eqcorrscan.core.match_filter import match_filter
from eqcorrscan import Tribe, Party, Family
from eqcorrscan.utils.pre_processing import multi_process

# Import our custom denoising function
from lbnl.denoiser import remove_HITP_spikes
from lbnl.waveforms import detection_multiplot

def main():
    """
    Main script to run daily matched-filtering with a custom denoising step.
    """
    # --- USER-DEFINED PARAMETERS ---
    # TRIBE_PATH = "/media/chopp/HDD1/chet-meq/cape_modern/templates/eqcorrscan/HITP_templates_1dayproc_clustered_10-1-25.tgz"
    TRIBE_PATH = "/media/chopp/HDD1/chet-meq/cape_modern/templates/eqcorrscan/HITP2_templates_1dayproc_clustered_10-1-25.tgz" 
    SPIKE_TEMPLATE_PATH = [
        "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
        "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
        # "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_1sec.txt",
    ]
    START_DATE = UTCDateTime(2025, 10, 27)
    END_DATE = UTCDateTime(2025, 10, 29)
    
    PARTY_OUTPUT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_2spikes/daily_parties"
    PLOT_OUTPUT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_2spikes/denoiser_plots"
    DETECTION_PLOT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_2spikes/detection_plots"
    MULTIPLOT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_2spikes/multiplots"

    WAVEFORM_CACHE = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/waveform_cache"

    # Matched-filter parameters
    CCTHRESH = 8.  # Detection threshold
    TRIG_INT = 2 # Trigger interval in seconds

    # Plotting parameters
    N_RANDOM_DET_PLOTS = 15
    RANDOM_WINDOW_SEC = 60  # length of random multiplot window (seconds)
    RANDOM_SEED = 42

    # Data-source parameters
    CLIENT_URL = 'http://131.243.224.19:8085'
    NETWORK = '6K'
    STATION = 'HITP,HITP2'
    CHANNELS = ['GK1', 'GPZ']

    # --- SCRIPT LOGIC ---

    os.makedirs(PARTY_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(WAVEFORM_CACHE, exist_ok=True)
    os.makedirs(DETECTION_PLOT_DIR, exist_ok=True)
    os.makedirs(MULTIPLOT_DIR, exist_ok=True)
    print(f"Party files will be saved to: {PARTY_OUTPUT_DIR}")
    print(f"Denoiser plots will be saved to: {PLOT_OUTPUT_DIR}")

    print("Loading Tribe...")
    try:
        tribe = Tribe().read(TRIBE_PATH)
        print(f"Loaded {len(tribe)} templates.")
    except Exception as e:
        print(f"Error: Could not load Tribe from {TRIBE_PATH}. Aborting.")
        print(e)
        return

    client = Client(CLIENT_URL)

    # Loop through each day
    current_date = START_DATE
    while current_date < END_DATE:
        print(f"\n--- Processing {current_date.strftime('%Y-%m-%d')} ---")
        day_start = current_date
        day_end = current_date + 86400

        try:
            # 1. Download data (or read from cache)
            cache_fname = f"{NETWORK}.{STATION}.{current_date.strftime('%j')}.mseed"
            cache_path = os.path.join(WAVEFORM_CACHE, cache_fname)

            if os.path.exists(cache_path):
                print(f"Reading cached daily data: {cache_path}")
                st = read(cache_path)
            else:
                print("Downloading daily data...")
                st = client.get_waveforms(NETWORK, STATION, '*', ",".join(CHANNELS), day_start, day_end)
                st.write(cache_path, format="MSEED")
                print(f"Wrote daily data to cache: {cache_path}")
            st.merge(fill_value='interpolate')
            st.detrend('demean')
            if 'HITP2' in [tr.stats.station for tr in st]:
                st.resample(1000.)

            # 2. Denoise the stream in-place
            print("Applying spike removal...")
            chunk_length = 1 * 3600

            # NEW: build denoised stream incrementally (one full copy total)
            st_denoised = Stream()

            chunk_start = day_start
            while chunk_start < day_end:
                chunk_end = min(chunk_start + chunk_length, day_end)
                print(f"Denoising chunk: {chunk_start} to {chunk_end}")

                chunk = st.slice(chunk_start, chunk_end)

                remove_HITP_spikes(
                    stream=chunk,
                    spike_template_path=SPIKE_TEMPLATE_PATH,
                    geophone_chans=['GPZ'],
                    plot=True,
                    plot_output_dir=PLOT_OUTPUT_DIR,
                    chunk_start=chunk_start,
                )

                st_denoised += chunk
                chunk_start = chunk_end

            st_denoised.merge(fill_value='interpolate')
            st = st_denoised

            # 3. Pre-process the denoised data for matched-filtering
            print("Running standard dayproc pre-processing...")
            st_processed = multi_process(
                st,
                lowcut=tribe[0].lowcut,
                highcut=tribe[0].highcut,
                filt_order=tribe[0].filt_order,
                samp_rate=tribe[0].samp_rate,
                parallel=True
            )
            
            # 4. Run matched-filtering
            print("Running matched-filter...")
            detections = match_filter(
                template_list=[t.st for t in tribe],
                template_names=[t.name for t in tribe],
                st=st_processed,
                threshold=CCTHRESH,
                threshold_type='MAD',
                trig_int=TRIG_INT,
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
                
                # Decluster the party
                daily_party.decluster(TRIG_INT)
                
                daily_output_path = os.path.join(PARTY_OUTPUT_DIR, f"detections_{current_date.strftime('%Y%m%d')}.tgz")
                
                if len(daily_party) > 0:
                    print(f"Saving {len(daily_party)} detections by {len(daily_party.families)} families to {daily_output_path}...")
                    daily_party.write(daily_output_path)
                    print("Save complete.")
                else:
                    print("No families with detections, not writing a file.")
            else:
                print("No detections found for this day.")

            # --- NEW: Plot random subset of individual detections ---
            if detections and fam_dict:
                rng = random.Random(RANDOM_SEED)
                det_list = list(detections)
                rng.shuffle(det_list)
                n_plot = min(N_RANDOM_DET_PLOTS, len(det_list))

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
                        streamcolour='k',
                        templatecolour='r',
                        show=False,
                        save=True,
                        savefile=os.path.join(DETECTION_PLOT_DIR, f"{det_id}.png")
                    )

            # --- NEW: One multiplot for a random window of denoised data ---
            if daily_party and len(daily_party) > 0:
                rng = random.Random(RANDOM_SEED + 1)
                max_start = max(0, int((day_end - day_start) - RANDOM_WINDOW_SEC))
                rand_offset = rng.randint(0, max_start) if max_start > 0 else 0
                win_start = day_start + rand_offset
                win_end = win_start + RANDOM_WINDOW_SEC

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
                        MULTIPLOT_DIR,
                        f"multiplot_{current_date.strftime('%Y%m%d')}_{win_start.strftime('%H%M%S')}_{int(RANDOM_WINDOW_SEC)}s.png"
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
        
        current_date += 86400 # Move to the next day

    print("\n--- Daily processing complete. ---")

if __name__ == '__main__':
    main()
