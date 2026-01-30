#!/usr/bin/python
import os
import argparse
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from eqcorrscan.core.match_filter import match_filter
from eqcorrscan import Tribe, Party, Family
from eqcorrscan.utils.pre_processing import multi_process
from lbnl.denoiser import remove_HITP_spikes

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
            # 1. Download data
            print("Downloading daily data...")
            st = client.get_waveforms(
                params['network'], params['station'], '*', ",".join(params['channels']),
                day_start, day_end
            )
            st.merge(fill_value='interpolate')
            st.detrend('demean')

            # 2. Denoise the stream in-place
            print("Applying spike removal...")
            chunk_length = 6 * 3600  # 6-hour chunks
            chunk_start = day_start
            while chunk_start < day_end:
                chunk_end = min(chunk_start + chunk_length, day_end)
                chunk = st.slice(chunk_start, chunk_end)
                remove_HITP_spikes(
                    stream=chunk,
                    spike_template_path=params['spike_template_path'],
                    geophone_chans=['GPZ'],
                    plot=True,
                    plot_output_dir=params['plot_output_dir'],
                    chunk_start=chunk_start,
                )
                chunk_start = chunk_end

            st.merge(fill_value='interpolate')

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
        'station': 'HITP',
        'channels': ['GK1', 'GPZ'],
        'cc_thresh': 10.0,
        'trig_int': 0.75,
        'spike_template_path': "/global/home/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
        'party_output_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_6hr/daily_parties",
        'plot_output_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_6hr/denoiser_plots",
    }

    # Ensure output directories exist
    os.makedirs(params['party_output_dir'], exist_ok=True)
    os.makedirs(params['plot_output_dir'], exist_ok=True)

    # Load the Tribe
    print("Loading Tribe...")
    tribe = Tribe().read("/global/home/users/chopp/chet-meq/cape_modern/templates/eqcorrscan/HITP_templates_1dayproc_clustered_10-1-25.tgz")
    # Initialize the client
    client = Client("http://131.243.224.19:8085")

    # Process the assigned date range
    process_date_range(task_start, task_end, tribe, client, params)