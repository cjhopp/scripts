#!/usr/bin/python
import os
import argparse
import random
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, Stream
from eqcorrscan.core.match_filter import match_filter
from eqcorrscan import Tribe, Party, Family
from eqcorrscan.utils.pre_processing import multi_process
from lbnl.denoiser import remove_HITP_spikes
from lbnl.waveforms import detection_multiplot

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
            # 1. Download data (or read from cache)
            cache_fname = f"{params['network']}.{params['station']}.{current_date.strftime('%j')}.mseed"
            cache_path = os.path.join(params['waveform_cache'], cache_fname)

            if os.path.exists(cache_path):
                print(f"Reading cached daily data: {cache_path}")
                st = read(cache_path)
            else:
                print("Downloading daily data...")
                st = client.get_waveforms(
                    params['network'], params['station'], '*', ",".join(params['channels']),
                    day_start, day_end
                )
                st.write(cache_path, format="MSEED")
                print(f"Wrote daily data to cache: {cache_path}")

            st.merge(fill_value='interpolate')
            st.detrend('demean')

            # 2. Denoise the stream in chunks, build a new stream
            print("Applying spike removal...")
            chunk_length = params['chunk_length_sec']
            st_denoised = Stream()

            chunk_start = day_start
            while chunk_start < day_end:
                chunk_end = min(chunk_start + chunk_length, day_end)
                print(f"Denoising chunk: {chunk_start} to {chunk_end}")

                chunk = st.slice(chunk_start, chunk_end)

                remove_HITP_spikes(
                    stream=chunk,
                    spike_template_path=params['spike_template_path'],
                    geophone_chans=['GPZ'],
                    plot=True,
                    plot_output_dir=params['plot_output_dir'],
                    chunk_start=chunk_start,
                )

                st_denoised += chunk
                chunk_start = chunk_end

            st_denoised.merge(fill_value='interpolate')
            st = st_denoised

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
            daily_party = None
            fam_dict = None
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

            # --- NEW: Plot random subset of individual detections ---
            if detections and fam_dict:
                rng = random.Random(params['random_seed'])
                det_list = list(detections)
                rng.shuffle(det_list)
                n_plot = min(params['n_random_det_plots'], len(det_list))

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
                        template_labels=[det.template_name],
                        streamcolour='k',
                        templatecolour='r',
                        show=False,
                        save=True,
                        savefile=os.path.join(params['detection_plot_dir'], f"{det_id}.png")
                    )

            # --- NEW: One multiplot for a random window of denoised data ---
            if daily_party and len(daily_party) > 0:
                rng = random.Random(params['random_seed'] + 1)
                max_start = max(0, int((day_end - day_start) - params['random_window_sec']))
                rand_offset = rng.randint(0, max_start) if max_start > 0 else 0
                win_start = day_start + rand_offset
                win_end = win_start + params['random_window_sec']

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
                        params['multiplot_dir'],
                        f"multiplot_{current_date.strftime('%Y%m%d')}_{win_start.strftime('%H%M%S')}_{int(params['random_window_sec'])}s.png"
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
        'spike_template_path': [
            "/global/home/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
            "/global/home/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
        ],
        'party_output_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/daily_parties",
        'plot_output_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/denoiser_plots",
        'waveform_cache': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/waveform_cache",
        'detection_plot_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/detection_plots",
        'multiplot_dir': "/global/scratch/users/chopp/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/multiplots",
        'chunk_length_sec': 1 * 3600,
        'n_random_det_plots': 15,
        'random_window_sec': 60,
        'random_seed': 42,
    }

    os.makedirs(params['party_output_dir'], exist_ok=True)
    os.makedirs(params['plot_output_dir'], exist_ok=True)
    os.makedirs(params['waveform_cache'], exist_ok=True)
    os.makedirs(params['detection_plot_dir'], exist_ok=True)
    os.makedirs(params['multiplot_dir'], exist_ok=True)

    # Load the Tribe
    print("Loading Tribe...")
    tribe = Tribe().read("/global/home/users/chopp/chet-meq/cape_modern/templates/eqcorrscan/HITP_templates_1dayproc_clustered_10-1-25.tgz")
    # Initialize the client
    client = Client("http://131.243.224.19:8085")

    # Process the assigned date range
    process_date_range(task_start, task_end, tribe, client, params)