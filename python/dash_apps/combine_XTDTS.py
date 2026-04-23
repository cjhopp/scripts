#!/usr/bin/python
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import xarray as xr
import zarr
from lxml import etree
from lxml.etree import XMLSyntaxError
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

WATCH_DIR = Path("/data/chet-cussp/DTS/raw_data/4100")
ZARR_PATH = Path("/data/chet-cussp/DTS/DTS_all.zarr")
NO_COLS = 6
QUEUE_TIMEOUT = 1
RETRY_ATTEMPTS = 3

# Write time as integer seconds since epoch so xarray can decode without cftime.
# Sub-second precision in the reference timestamp (e.g. 'days since 2024-07-11
# 20:02:01.165000') causes a ValueError in the standard decoder.
TIME_ENCODING = {"time": {"units": "seconds since 1970-01-01 00:00:00", "dtype": "int64"}}


def normalize_event_path(raw_path):
    """
    Canonicalize a path to the .xml extension.

    The XTDTS writer first creates/renames to names like 'foo.chan.xml' or 'foo.xml.chan'
    before settling on the final name. Canonicalize to the bare .xml stem.
    """
    path = Path(raw_path)
    # Fast path: already clean
    if path.suffix.lower() == ".xml":
        return path
    # Find the first .xml in the name and truncate there
    xml_index = path.name.lower().find(".xml")
    if xml_index == -1:
        return path
    return path.with_name(path.name[: xml_index + 4])


def read_XTDTS(path, no_cols):
    """Parse a single XTDTS XML file. Returns (datetime, ndarray, ref, p1, p2) or None."""
    try:
        dts = etree.parse(str(path))
    except (OSError, XMLSyntaxError) as exc:
        logging.warning("Failed to parse %s: %s", path, exc)
        return None

    try:
        root = dts.getroot()
        measurements = np.fromstring(
            ",".join(
                [
                    line.text.replace("\n", "")
                    for line in root[0].find("{*}logData").findall("{*}data")
                ]
            ),
            sep=",",
        )
        measurements = measurements.reshape(-1, no_cols)
        dto = datetime.strptime(
            root[0].find("{*}endDateTimeIndex").text,
            "%Y-%m-%dT%H:%M:%S.%fZ",
        )
        custom = root[0].find("{*}customData")
        ref = float(custom.find("{*}referenceTemperature").text)
        p1 = float(custom.find("{*}probe1Temperature").text)
        p2 = float(custom.find("{*}probe2Temperature").text)
    except (AttributeError, TypeError, ValueError) as exc:
        logging.warning("Malformed XML payload in %s: %s", path, exc)
        return None

    return dto, measurements, ref, p1, p2


def make_dataset(parsed, no_cols):
    """Build an xr.Dataset from a parsed XTDTS tuple."""
    dto, measures, ref, p1, p2 = parsed
    # Explicit nanosecond dtype avoids xarray's non-nanosecond conversion warning
    times = np.array([dto], dtype="datetime64[ns]")
    return xr.Dataset(
        {
            "temperature": xr.DataArray(
                measures[:, [no_cols - 1]],
                coords={"depth": measures[:, 0], "time": times},
                dims=["depth", "time"],
                attrs={"units": "degrees C"},
            ),
            "reference_temperature": xr.DataArray(
                np.array([ref]),
                coords={"time": times},
                dims=["time"],
                attrs={"units": "degrees C"},
            ),
            "probe1_temperature": xr.DataArray(
                np.array([p1]),
                coords={"time": times},
                dims=["time"],
                attrs={"units": "degrees C"},
            ),
            "probe2_temperature": xr.DataArray(
                np.array([p2]),
                coords={"time": times},
                dims=["time"],
                attrs={"units": "degrees C"},
            ),
        }
    )


def inspect_store(zarr_path):
    """
    Open the Zarr store and return (known_timestamps: set, store_ready: bool).

    A store is considered broken (not ready) if it is missing, unreadable, or
    has no 'time' dimension (e.g. the empty {} state seen after a first-write crash).
    """
    if not zarr_path.exists():
        return set(), False

    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
    except Exception as exc:
        logging.warning("Treating unreadable Zarr store as broken: %s", exc)
        return set(), False

    if "time" not in ds.sizes or ds.sizes["time"] == 0:
        logging.warning(
            "Zarr store at %s has empty or missing 'time' dimension – will overwrite on first good write",
            zarr_path,
        )
        return set(), False

    known = {np.datetime64(value) for value in ds.time.values}
    logging.info("Loaded %s existing timestamps from Zarr store", len(known))
    return known, True


class Worker(threading.Thread):
    """
    Dedicated I/O thread. Consumes file paths from the queue, parses XML,
    deduplicates, and writes to Zarr. Never runs in the Observer thread.
    """

    def __init__(self, file_queue, stop_event, zarr_path, no_cols, known_timestamps, store_ready):
        super().__init__(daemon=True, name="ZarrWriter")
        self.file_queue = file_queue
        self.stop_event = stop_event
        self.zarr_path = zarr_path
        self.no_cols = no_cols
        self.known_timestamps = known_timestamps
        # Mutable bool boxed in a list so _write_dataset can update it
        self._store_ready = [store_ready]

    def run(self):
        while not self.stop_event.is_set() or not self.file_queue.empty():
            try:
                raw_path = self.file_queue.get(timeout=QUEUE_TIMEOUT)
            except Empty:
                continue
            try:
                self._process(raw_path)
            except Exception:
                logging.exception("Unhandled error processing %s", raw_path)
            finally:
                self.file_queue.task_done()

    def _process(self, raw_path):
        path = normalize_event_path(raw_path)
        if path.suffix.lower() != ".xml":
            return
        if not path.exists():
            logging.info("Skipping missing file %s", path)
            return

        parsed = read_XTDTS(path, self.no_cols)
        if parsed is None:
            return

        timestamp = np.datetime64(parsed[0])
        if timestamp in self.known_timestamps:
            logging.info("Duplicate timestamp %s in %s – skipping", timestamp, path.name)
            return

        ds = make_dataset(parsed, self.no_cols)
        self._write_dataset(ds)
        self.known_timestamps.add(timestamp)
        logging.info("Ingested %s (%s)", path.name, parsed[0].isoformat())

    def _write_dataset(self, dataset):
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                if not self._store_ready[0]:
                    self._nuke_broken_store()
                    dataset.to_zarr(self.zarr_path, mode="w", encoding=TIME_ENCODING)
                    self._store_ready[0] = True
                else:
                    dataset.to_zarr(self.zarr_path, append_dim="time")
                zarr.consolidate_metadata(str(self.zarr_path))
                return
            except Exception as exc:
                if attempt == RETRY_ATTEMPTS:
                    raise
                backoff = 2 ** (attempt - 1)
                logging.warning(
                    "Write attempt %s/%s failed: %s; retrying in %ss",
                    attempt,
                    RETRY_ATTEMPTS,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

    def _nuke_broken_store(self):
        if not self.zarr_path.exists():
            return
        logging.warning("Removing broken Zarr store at %s", self.zarr_path)
        if self.zarr_path.is_dir():
            shutil.rmtree(self.zarr_path)
        else:
            self.zarr_path.unlink()


class Handler(FileSystemEventHandler):
    """
    Watchdog event handler. Queues paths only – no I/O here.
    on_closed fires after the writer has closed the file (inotify IN_CLOSE_WRITE).
    on_moved handles rsync --partial-dir / --delay-updates atomic renames.
    """

    def __init__(self, file_queue):
        self.file_queue = file_queue

    def _enqueue(self, raw_path):
        path = normalize_event_path(raw_path)
        if path.suffix.lower() != ".xml":
            return
        logging.info("Queueing %s", path.name)
        self.file_queue.put(path)

    def on_closed(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._enqueue(event.dest_path)


class Watcher:
    def __init__(
        self,
        directory_to_watch=WATCH_DIR,
        zarr_path=ZARR_PATH,
        no_cols=NO_COLS,
    ):
        self.directory_to_watch = Path(directory_to_watch)
        self.zarr_path = Path(zarr_path)
        self.no_cols = no_cols
        self.file_queue = Queue()
        self.stop_event = threading.Event()

        known_timestamps, store_ready = inspect_store(self.zarr_path)
        self.event_handler = Handler(self.file_queue)
        self.worker = Worker(
            file_queue=self.file_queue,
            stop_event=self.stop_event,
            zarr_path=self.zarr_path,
            no_cols=self.no_cols,
            known_timestamps=known_timestamps,
            store_ready=store_ready,
        )
        self.observer = Observer()

    def run(self):
        self.worker.start()
        self._startup_catchup()
        self._schedule_observer()
        self.observer.start()
        try:
            while True:
                time.sleep(5)
                if not self.observer.is_alive():
                    logging.error("Observer thread died; restarting")
                    self._restart_observer()
        except KeyboardInterrupt:
            logging.info("Stopping watcher on keyboard interrupt")
        finally:
            self.stop_event.set()
            self.observer.stop()
            self.observer.join()
            self.file_queue.join()
            self.worker.join(timeout=5)

    def _startup_catchup(self):
        seen = set()
        count = 0
        for raw_path in sorted(self.directory_to_watch.iterdir()):
            path = normalize_event_path(raw_path)
            if path.suffix.lower() != ".xml" or not path.exists() or path in seen:
                continue
            seen.add(path)
            self.file_queue.put(path)
            count += 1
        logging.info("Startup catchup: queued %s XML file(s)", count)

    def _schedule_observer(self):
        self.observer.schedule(
            self.event_handler,
            str(self.directory_to_watch),
            recursive=False,
        )

    def _restart_observer(self):
        self.observer.stop()
        self.observer.join()
        self.observer = Observer()
        self._schedule_observer()
        self.observer.start()
        logging.info("Observer restarted")


if __name__ == "__main__":
    logging.basicConfig(
        filename="/data/chet-cussp/DTS/combine_XTDTS.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("Starting the directory watcher")
    Watcher().run()
