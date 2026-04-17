#!/usr/bin/python
"""
Reproduce a single detection correlation coefficient from EQcorrscan output.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import read, Stream, UTCDateTime

from eqcorrscan import Party
from eqcorrscan.core.match_filter import match_filter
from eqcorrscan.utils.correlate import get_array_xcorr
from eqcorrscan.utils.pre_processing import multi_process

from relative_moments import remove_HITP_spikes


PARTY_PATH = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/denoised_data_1hr/All_detections_HITP-HITP2_MAD20_w-magnitudes.tgz"
WAVEFORM_CACHE = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/waveform_cache"
OUTPUT_DIR = "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/repro_cc"

TEMPLATE_NAME = "lbnl2025rbjf"
DETECT_TIME = UTCDateTime("2025-08-06T22:15:57.476000Z")
DETECT_VAL = 0.8390554
CHANNEL = ("HITP", "GPZ")

PREFER_LOCAL = True
DENOISE_ENABLED = False
SPIKE_TEMPLATE_PATH = [
    "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
    "/media/chopp/HDD1/chet-meq/cape_modern/matched_filter/HITP_detect/GK1_spikes/spiketemplate_GK1.txt",
]
DENOISE_CHUNK_S = 3600.0
DENOISE_GEOPHONE_CHANS = ["GPZ"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _cache_path(day: UTCDateTime, network: str, stations: str) -> Path:
    return Path(WAVEFORM_CACHE) / f"{network}.{stations}.{day.strftime('%j')}.mseed"


def _load_day_stream(day: UTCDateTime, network: str, stations: str, channels: str) -> Stream:
    cache_path = _cache_path(day, network, stations)
    if PREFER_LOCAL and cache_path.exists():
        logger.info("Reading cached day stream %s", cache_path)
        st = read(str(cache_path))
    else:
        raise RuntimeError(f"Missing cache file: {cache_path}")
    logger.info("Merging day stream")
    st.merge(fill_value="interpolate")
    logger.info("Detrending day stream")
    st.detrend("demean")
    if any(tr.stats.station == "HITP2" for tr in st):
        logger.info("Resampling day stream to 1000 Hz")
        st.resample(1000.0)
    if DENOISE_ENABLED:
        logger.info("Denoising day stream")
        st_denoised = Stream()
        chunk_start = day
        day_end = day + 86400
        while chunk_start < day_end:
            chunk_end = min(chunk_start + DENOISE_CHUNK_S, day_end)
            chunk = st.slice(chunk_start, chunk_end)
            remove_HITP_spikes(
                stream=chunk,
                spike_template_path=SPIKE_TEMPLATE_PATH,
                geophone_chans=DENOISE_GEOPHONE_CHANS,
                plot=False,
                plot_output_dir=".",
                chunk_start=chunk_start,
            )
            st_denoised += chunk
            chunk_start = chunk_end
        st_denoised.merge(fill_value="interpolate")
        st = st_denoised
    return st


def _find_detection(party: Party):
    for family in party.families:
        if family.template.name != TEMPLATE_NAME:
            continue
        for det in family.detections:
            if abs(det.detect_time - DETECT_TIME) < 0.001:
                return family, det
    raise RuntimeError("Detection not found in party")


def _plot_overlay(template_data, data_window, sr, out_path: Path) -> None:
    def _norm(data):
        data = np.asarray(data, dtype=float)
        scale = np.max(np.abs(data))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        return data / scale

    t = np.arange(len(template_data)) / sr
    template_n = _norm(template_data)
    data_n = _norm(data_window)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, data_n, color="black", label="data")
    ax.plot(t, template_n, color="red", alpha=0.8, label="template")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_corr_series(lags, corr_series, detect_lag, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(lags, corr_series, color="black")
    ax.axvline(detect_lag, color="red", linestyle="--", label="detect")
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("XCorr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Party %s", PARTY_PATH)
    party = Party().read(PARTY_PATH, read_detection_catalog=False)
    family, det = _find_detection(party)
    logger.info("Using template %s", family.template.name)
    logger.info("Detection time %s", det.detect_time)
    template = family.template

    template_st = template.st.select(station=CHANNEL[0], channel=CHANNEL[1])
    if len(template_st) != 1:
        raise RuntimeError("Template channel not found")
    template_tr = template_st[0]
    logger.info("Template channel %s.%s", template_tr.stats.station, template_tr.stats.channel)
    logger.info("Template length %s samples", template_tr.stats.npts)

    day = UTCDateTime(DETECT_TIME.date)
    st = _load_day_stream(day, template_tr.stats.network, ",".join([CHANNEL[0]]), CHANNEL[1])

    logger.info("Running dayproc with template parameters")
    processed = multi_process(
        st=st,
        lowcut=template.lowcut,
        highcut=template.highcut,
        filt_order=template.filt_order,
        samp_rate=template.samp_rate,
        parallel=True,
    )

    data_tr = processed.select(
        network=template_tr.stats.network,
        station=CHANNEL[0],
        channel=CHANNEL[1],
        location=template_tr.stats.location,
    )
    if len(data_tr) != 1:
        raise RuntimeError("Processed channel not found")
    data_tr = data_tr[0]
    logger.info("Processed channel %s.%s", data_tr.stats.station, data_tr.stats.channel)

    sr = data_tr.stats.sampling_rate
    template_data = template_tr.data.astype(np.float64)
    template_len = len(template_data) / sr

    data_window = data_tr.slice(DETECT_TIME, DETECT_TIME + template_len).data.astype(np.float64)
    data_window = data_window[: len(template_data)]

    # Normalized correlation at zero lag
    def _norm(x):
        denom = np.sqrt(np.sum(x * x))
        return x / denom if denom else x
    corr_zero = float(np.dot(_norm(template_data), _norm(data_window)))

    # Sliding correlation around detect time for visualization
    pad_s = 2.0
    pad_samples = int(pad_s * sr)
    start = DETECT_TIME - pad_s
    end = DETECT_TIME + template_len + pad_s
    data_long = data_tr.slice(start, end).data.astype(np.float32)

    xcorr_func = get_array_xcorr()
    cccs, _ = xcorr_func(template_data[None, :].astype(np.float32), data_long, [0], cc_squared=False)
    corr_series = np.asarray(cccs[0], dtype=float)
    lags = (np.arange(len(corr_series)) - pad_samples) / sr

    detect_index = pad_samples
    detect_corr = corr_series[detect_index]

    # Run match_filter for confirmation
    detections = match_filter(
        template_list=[template_st],
        template_names=[TEMPLATE_NAME],
        st=processed,
        threshold=0.1,
        threshold_type="absolute",
        trig_int=0.5,
    )
    det_match = None
    for d in detections:
        if abs(d.detect_time - DETECT_TIME) < 0.001:
            det_match = d
            break

    print("Expected detect_val:", DETECT_VAL)
    print("corr_zero:", corr_zero)
    print("corr_series_at_detect:", detect_corr)
    if det_match is not None:
        print("match_filter detect_val:", det_match.detect_val)
    else:
        print("match_filter detection not found at time")

    logger.info("Expected detect_val: %.6f", DETECT_VAL)
    logger.info("corr_zero: %.6f", corr_zero)
    logger.info("corr_series_at_detect: %.6f", detect_corr)

    _plot_overlay(template_data, data_window, sr, out_dir / "overlay.png")
    _plot_corr_series(lags, corr_series, 0.0, out_dir / "corr_series.png")
    logger.info("Wrote plots to %s", out_dir)


if __name__ == "__main__":
    main()
