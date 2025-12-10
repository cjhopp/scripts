#!/usr/bin/python

import os
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from eqcorrscan.core.match_filter import match_filter
from eqcorrscan import Tribe, Party, Family
from eqcorrscan.utils.pre_processing import multi_processing

# Import our custom denoising function
from lbnl.denoiser import remove_HITP_spikes

def main():
    """
    Main script to run daily matched-filtering with a custom denoising step.
    """
    # --- USER-DEFINED PARAMETERS ---
    TRIBE_PATH = "/media/chopp/HDD1/chet-meq/cape_modern/templates/eqcorrscan/HITP_templates_1dayproc_clustered_10-1-25.tgz" 
    SPIKE_TEMPLATE_PATH = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt"
    START_DATE = UTCDateTime(2025, 7, 27)
    END_DATE = UTCDateTime(2025, 10, 8)
    
    PARTY_OUTPUT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data/daily_parties"
    PLOT_OUTPUT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data/denoiser_plots"

    # Matched-filter parameters
    CCTHRESH = 8.  # Detection threshold
    TRIG_INT = 2 # Trigger interval in seconds

    # Data-source parameters
    CLIENT_URL = 'http://131.243.224.19:8085'
    NETWORK = '6K'
    STATION = 'HITP'
    CHANNELS = ['GK1', 'GPZ']

    # --- SCRIPT LOGIC ---

    os.makedirs(PARTY_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
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
            # 1. Download data
            print("Downloading daily data...")
            st = client.get_waveforms(NETWORK, STATION, '*', ",".join(CHANNELS), day_start, day_end)
            st.merge(fill_value='interpolate')
            st.detrend('demean')

            # 2. Denoise the stream in-place
            print("Applying spike removal...")
            remove_HITP_spikes(stream=st, 
                               spike_template_path=SPIKE_TEMPLATE_PATH,
                               geophone_chans=['GPZ'], 
                               plot=True, 
                               plot_output_dir=PLOT_OUTPUT_DIR)

            # 3. Pre-process the denoised data for matched-filtering
            print("Running standard dayproc pre-processing...")
            st_processed = multi_processing(st, lowcut=tribe[0].lowcut, highcut=tribe[0].highcut, filt_order=tribe[0].filt_order, samp_rate=tribe[0].samp_rate, parallel=True)
            
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
            if detections:
                fam_dict = {t.name: Family(template=t) for t in tribe}
                for d in detections:
                    fam_dict[d.template_name] += d
                
                daily_party = Party(families=[fam for fam in fam_dict.values() if len(fam.detections) > 0])
                
                daily_output_path = os.path.join(PARTY_OUTPUT_DIR, f"detections_{current_date.strftime('%Y%m%d')}.tgz")
                
                if len(daily_party) > 0:
                    print(f"Saving {len(daily_party)} families to {daily_output_path}...")
                    daily_party.write(daily_output_path)
                    print("Save complete.")
                else:
                    print("No families with detections, not writing a file.")
            else:
                print("No detections found for this day.")

        except Exception as e:
            print(f"Error processing day {current_date.strftime('%Y-%m-%d')}: {e}")
        
        current_date += 86400 # Move to the next day

    print("\n--- Daily processing complete. ---")

if __name__ == '__main__':
    main()
