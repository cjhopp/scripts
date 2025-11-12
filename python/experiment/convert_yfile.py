import os
import zipfile
import shutil
import logging

from glob import glob
from obspy import read, read_inventory

# Setup logging configuration
logging.basicConfig(
    filename='process_log.txt',  # Specify the log file name
    level=logging.INFO,  # Log level, can be DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)


mapping = {
    'B': '5B',
    'D': '3B',
    'R': '4B',
    'N': '6B',
    'F': '3I'
}

comp_map = {
    'DPZ': 'DP1',
    'DPN': 'DP2',
    'DPE': 'DP3',
}

non_ncedc_boreholes = ['RRS12', 'RRS13', 'RRS14', 'RRS15', 'BRB10', 'BRB11']

non_ncedc_surface = ['RRS16', 'RRS17', 'RRS18', 'BRP9']

def convert_y_to_miniseed(y_file_path, output_base_path, inventory):
    stream = read(y_file_path)
    tr = stream[0]
    # Add network code
    tr.stats.network = mapping[tr.stats.station[0]]
    # In case of Brady, remove zeros in station name
    if tr.stats.station[0] == 'B' and tr.stats.station[-1] != '0':
        tr.stats.station = tr.stats.station.replace('0', '')
    # Assume these are all geophones at 500 Hz. Downhole sensors are DP[123], loc 40
    tr.stats.channel = 'DP' + tr.stats.channel[2:]
    if tr.stats.station in non_ncedc_boreholes or tr.stats.network == '3I':
        loc = '40'
    elif tr.stats.station in non_ncedc_surface:
        loc = '00'
    else:
        try:
            loc = inventory.select(network=tr.stats.network, station=tr.stats.station,
                                    time=tr.stats.starttime)[0][0][0].location_code
        except IndexError as e:
            logging.error(f"No NCEDC channel for {tr.stats.network}.{tr.stats.station} at {tr.stats.starttime}")
            sta = inventory.select(network=tr.stats.network, station=tr.stats.station)[0][0]
            sta.channels.sort(key=lambda x: x.start_date)
            loc = sta.channels[-1].location_code
            logging.info(f"Using latest location code {loc} for {tr.stats.network}.{tr.stats.station}")
    tr.stats.location = loc
    if loc == '40':
        tr.stats.channel = comp_map[tr.stats.channel]
    # Build output directory structure
    target_dir = os.path.join(
        output_base_path,
        tr.stats.network,
        tr.stats.station,
        tr.stats.channel
    )
    os.makedirs(target_dir, exist_ok=True)
    basename = (
        f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}_"
        f"{tr.stats.starttime.strftime('%Y.%jT%H%M%S')}_"
        f"{tr.stats.endtime.strftime('%Y.%jT%H%M%S')}"
    )
    out_file = os.path.join(target_dir, os.path.basename(basename) + '.mseed')
    if os.path.exists(out_file):
        logging.info(f"File already exists, skipping: {out_file}")
        return
    logging.info(f"Converting {y_file_path} to {out_file}")
    stream.write(out_file, format='MSEED')

def process_zip_directory(zip_dir_path, output_base_path, inventory):
    logging.info(f"Processing zip directory: {zip_dir_path}")
    zip_dirs = glob(f'{zip_dir_path}/*.zip')
    for zd in zip_dirs:
        logging.info(f"Found zip file: {zd}")
        # Create a temporary directory to unzip files
        temp_dir = zd.replace('.zip', '')
        try:
            with zipfile.ZipFile(zd, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            logging.info(f"Extracted {zd} to {temp_dir}")
        except Exception as e:
            logging.error(f"Failed to unzip {zd}: {e}")
            return

        # Loop through unzipped Y files and convert to miniseed
        for root, _, files in os.walk(temp_dir):
            for file in files:
                y_file_path = os.path.join(root, file)
                convert_y_to_miniseed(y_file_path, output_base_path, inventory)

        # Remove the unzipped directory
        shutil.rmtree(temp_dir)
        logging.info(f"Removed temporary directory: {temp_dir}")

def main(input_path, output_base_path):
    logging.info(f"Starting processing with input file: {input_path}")
    
    try:
        with open(input_path, 'r') as file:
            zip_dirs = file.readlines()
        
        inventory = read_inventory(inv_path)
        for zip_dir in zip_dirs:
            zip_dir = zip_dir.strip()
            process_zip_directory(zip_dir, output_base_path, inventory)
        
        logging.info("Completed processing all zip directories.")
    
    except Exception as e:
        logging.error(f"Error in main processing: {e}")

if __name__ == "__main__":
    # Specify input text file containing the paths to the zip directories
    input_path = '/home/chopp/zip_dirs_no_fallon.txt'
    # Specify output base path where miniseed files will be saved
    output_base_path = '/data/HDDs/mseed/taurus-data/round2'
    # inventory
    # inv_path = '/home/chopp/All_arrays_NCEDC.xml'
    inv_path = '/media/chopp/HDD1/chet-meq/All_historic_networks_11-12-2025.xml'
    main(input_path, output_base_path)
