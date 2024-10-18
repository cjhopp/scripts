#!/usr/bin/python
import time
import logging

import numpy as np
import xarray as xr

from glob import glob
from datetime import datetime
from lxml import etree
from lxml.etree import XMLSyntaxError
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def read_XTDTS(path, no_cols):
    # Read single xml file and return array for all values and time
    try:
        dts = etree.parse(path)
    except XMLSyntaxError as e:
        return None
    # Get root element
    root = dts.getroot()
    # Create one string for all values, comma sep
    measurements = np.fromstring(','.join(
        [l.text.replace('\n', '')
         for l in root[0].find('{*}logData').findall('{*}data')]),
        sep=',')
    # 6 columns in original data
    measurements = measurements.reshape(-1, no_cols)
    # Get time
    dto = datetime.strptime(root[0].find('{*}endDateTimeIndex').text, '%Y-%m-%dT%H:%M:%S.%fZ')
    ref = float(root[0].find('{*}customData').find('{*}referenceTemperature').text)
    p1 = float(root[0].find('{*}customData').find('{*}probe1Temperature').text)
    p2 = float(root[0].find('{*}customData').find('{*}probe2Temperature').text)
    return dto, measurements, ref, p1, p2


def read_XTDTS_to_xarray(new_file, no_cols):
    """
    Read a directory of xml files into an xarray Dataset object

    :param directory: Path to file root
    :param no_cols: Number of columns in the xml files (configurable for XTDTS)
    :return:
    """

    times, measures, ref, p1, p2 = read_XTDTS(new_file)
    times = np.array(times)
    measures = np.stack(measures, axis=-1)
    # Only save the temperature DataArray for now; can add stokes arrays if needed
    temp = xr.DataArray(measures[:, no_cols-1, :], name='temperature',
                        coords={'depth': measures[:, 0, 0],
                                'time': times},
                        dims=['depth', 'time'],
                        attrs={'units': 'degrees C'})
    delta = xr.DataArray(measures[:, no_cols-1, :], name='deltaT',
                         coords={'depth': measures[:, 0, 0],
                                 'time': times},
                         dims=['depth', 'time'],
                         attrs={'units': 'degrees C'})
    delta = delta - delta.isel(time=0)
    ds = xr.Dataset({'temperature': temp, 'deltaT': delta})
    return ds


# Monitoring class for new files
class Watcher:
    DIRECTORY_TO_WATCH = "/data/chet-cussp/DTS/raw_data/4100"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):

    @staticmethod
    def on_created(event):
        if not event.is_directory:
            logging.info(f"New file created: {event.src_path}")
            # Read in entire dataset
            ds = xr.open_dataset('/data/chet-cussp/DTS/DTS_all.nc', chunks={'depth': 1000})
            ds += read_XTDTS_to_xarray(event.src_path)
            ds.to_netcdf('/data/chet-cussp/DTS/DTS_all.nc')


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(filename="/data/chet-cussp/DTS/combine_XTDTS.log",
                        level=logging.INFO,
                        format="%(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info("Starting the directory watcher")
    w = Watcher()
    w.run()