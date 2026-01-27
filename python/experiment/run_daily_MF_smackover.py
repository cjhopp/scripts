#!/usr/bin/python

import os
import logging
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from eqcorrscan import Tribe
from eqcorrscan.utils.plotting import detection_multiplot

import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend

def detect_tribe_client_with_lag_calc(tribe, client, start, end, param_dict,
                                      party_dir, plot_dir, waveform_dir, log_path):
    """
    Run detect for tribe on specified wav client, with lag_calc, detection_multiplot,
    and saving detection waveforms.

    :param tribe: eqcorrscan Tribe object.
    :param client: obspy Client object.
    :param start: Start time for detections.
    :param end: End time for detections.
    :param param_dict: Parameters necessary for running detect from client.
    :param party_dir: Directory to save daily parties.
    :param plot_dir: Directory to save detection multiplots.
    :param waveform_dir: Directory to save detection waveforms.
    :param log_path: Path to the log file.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Ensure output directories exist
    os.makedirs(party_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(waveform_dir, exist_ok=True)

    logger.info("Starting detection workflow...")
    logger.info(f"Party directory: {party_dir}")
    logger.info(f"Plot directory: {plot_dir}")
    logger.info(f"Waveform directory: {waveform_dir}")

    # Loop over each day in the specified time range
    current_date = start
    while current_date < end:
        logger.info(f"Processing {current_date.strftime('%Y-%m-%d')}...")
        day_start = current_date
        day_end = min(current_date + 86400, end)  # Process one day at a time

        try:
            # Step 1: Run detection for the day
            logger.info("Running detection...")
            if param_dict['return_stream']:
                day_party, day_stream = tribe.client_detect(
                    client=client,
                    starttime=day_start,
                    endtime=day_end,
                    **param_dict
                )
            else:
                day_party = tribe.client_detect(
                    client=client,
                    starttime=day_start,
                    endtime=day_end,
                    **param_dict
                )

            # Step 2: Refine picks using Party.lag_calc
            logger.info("Refining picks with Party.lag_calc...")
            refined_catalog = day_party.lag_calc(
                stream=day_stream,
                pre_processed=False,
                shift_len=param_dict.get('shift_len', 0.2),
                min_cc=param_dict.get('min_cc', 0.7),
                interpolate=param_dict.get('interpolate', True),
                plot=True,
                plotdir= plot_dir
            )
            # Save the refined picks catalog to a file
            catalog_output_path = os.path.join(
                party_dir,
                f"refined_picks_{current_date.strftime('%Y%m%d')}.xml"
            )
            logger.info(f"Saving refined picks catalog to {catalog_output_path}...")
            refined_catalog.write(catalog_output_path, format="QUAKEML")
            logger.info("Refined picks catalog saved.")

            # Step 3: Generate multiplots for detections
            logger.info("Generating multiplots...")
            for family in day_party.families:
                for detection in family.detections:
                    template_start = min(trace.stats.starttime for trace in family.template.st)
                    template_end = max(trace.stats.endtime for trace in family.template.st)
                    template_duration = template_end - template_start
                    padding = 0.1 * template_duration
                    plot_start = detection.detect_time - padding
                    plot_end = detection.detect_time + template_duration + padding
                    plot_stream = day_stream.slice(starttime=plot_start, endtime=plot_end).copy()
                    plot_stream.detrend('linear')  # Simple detrend for viz (should be proceessed already)
                    plot_stream.taper(0.05)
                    plot_stream.filter('bandpass', freqmin=family.template.lowcut, freqmax=family.template.highcut, corners=4)
                    plot_stream.detrend('linear')
                    detection_multiplot(
                        stream=plot_stream,
                        template=family.template.st,
                        times=[detection.detect_time],
                        streamcolour='k',
                        templatecolour='r',
                        show=False,
                        save=True,
                        savefile=os.path.join(
                            plot_dir,
                            f"{detection.id}.png"
                        )
                    )

            # Step 4: Save waveforms for each detection
            logger.info("Saving waveforms for each detection...")
            for family in day_party.families:
                for detection in family.detections:
                    detection_waveform = day_stream.slice(
                        starttime=detection.detect_time - param_dict['waveform_padding'][0],
                        endtime=detection.detect_time + param_dict['waveform_padding'][1]
                    )
                    waveform_output_path = os.path.join(
                        waveform_dir,
                        f"{detection.id}.mseed"
                    )
                    detection_waveform.write(waveform_output_path, format="MSEED")
                    logger.info(f"Saved waveform to {waveform_output_path}")

            # Step 5: Save the Party for the day
            party_output_path = os.path.join(party_dir, f"party_{current_date.strftime('%Y%m%d')}.tgz")
            logger.info(f"Saving Party to {party_output_path}...")
            day_party.write(party_output_path)
            logger.info("Party saved.")

        except Exception as e:
            logger.error(f"Error processing day {current_date.strftime('%Y-%m-%d')}: {e}")

        # Move to the next day
        current_date = day_end

    logger.info("Processing complete.")


if __name__ == "__main__":
    # --- USER-DEFINED PARAMETERS ---
    TRIBE_PATH = "/media/chopp/HDD1/chet-meq/smackover/templates/Smackover_north_tribe_snr5.tgz"
    CLIENT_URL = "IRIS"
    START_DATE = UTCDateTime("2019-03-07")
    END_DATE = UTCDateTime("2025-12-31")
    PARTY_DIR = "/media/chopp/HDD1/chet-meq/smackover/detections/parties/smackover_north_full_tribe/MAD12_2hr"
    PLOT_DIR = "/media/chopp/HDD1/chet-meq/smackover/detections/plots/smackover_north_full_tribe/MAD12_2hr"
    WAVEFORM_DIR = "/media/chopp/HDD1/chet-meq/smackover/detections/waveforms/smackover_north_full_tribe/MAD12_2hr"
    LOG_PATH = "/media/chopp/HDD1/chet-meq/smackover/detections/logs/smackover_north_full_tribe/MAD12_2hr/detection_log.txt"

    # Detection parameters
    PARAM_DICT = {
        "threshold": 12.0,
        "threshold_type": "MAD",
        "trig_int": 10.0,
        "return_stream": True,
        "concurrent_processing": True,
        "process_cores": 30,
        "shift_len": 0.5,
        "min_cc": 0.7,
        "interpolate": True,
        "waveform_padding": [30, 200]  # Pre- and post-detection padding in seconds
    }

    # Load the Tribe
    tribe = Tribe().read(TRIBE_PATH)

    # Initialize the client
    client = Client(CLIENT_URL)

    # Run the detection workflow
    detect_tribe_client_with_lag_calc(
        tribe=tribe,
        client=client,
        start=START_DATE,
        end=END_DATE,
        param_dict=PARAM_DICT,
        party_dir=PARTY_DIR,
        plot_dir=PLOT_DIR,
        waveform_dir=WAVEFORM_DIR,
        log_path=LOG_PATH
    )