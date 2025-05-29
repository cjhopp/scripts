#!/usr/bin/env python3
"""
Full‐range beamforming from RUN_START → RUN_END
using Dask LocalCluster (4 workers × 2 threads each).

Dependencies:
  pip install numpy xarray dask[distributed] obspy scipy matplotlib
"""

import os
# throttle BLAS in each worker
os.environ["OMP_NUM_THREADS"]       = "2"
os.environ["OPENBLAS_NUM_THREADS"]  = "2"
os.environ["MKL_NUM_THREADS"]       = "2"

import numpy as np
import xarray as xr
from obspy import UTCDateTime, read_inventory
from dask.distributed import Client, LocalCluster

# -------------------------------------------------------------------
# USER PARAMETERS: edit these
# -------------------------------------------------------------------
# 1) time window for the entire run:
RUN_START    = "2022-02-15T06:00:00"
RUN_END      = "2023-01-07T18:30:00"

# 2) paths & inventory
MSEED_DIR     = "/path/to/your/mseed"
INVENTORY_XML = "/path/to/your/inventory.xml"
OUTDIR        = "/path/to/output_dir"

# 3) beam‐forming “cheap” knobs
SAMP_RATE   = 20.0       # Hz, Nyquist > highest band
CHUNK_S     = 600.0      # data‐chunk size for dask arrays
WIN_S       = 300.0      # window length [s]
OVERLAP     = 0.2        # 20% overlap
PAD_S       = 10.0       # padding for filtfilt
METHOD      = "bartlett" # only Bartlett here

# 4) coarse beam grid
BAZS  = np.arange(0, 360, 5.0)        # every 5°
SLOWS = np.linspace(0.0, 0.003, 30)   # 0 → 3 ms/m in 30 steps

# 5) frequency bands
BANDS = [
    (0.1, 1.0),
    (1.0, 5.0),
    (5.0,10.0),
]

# -------------------------------------------------------------------
# import your beam‐former class
# -------------------------------------------------------------------
from yourmodule import SeismicBeamformer
# -------------------------------------------------------------------

def process_block(inv, mseed_dir, t0_str, t1_str, outdir):
    """
    1) load_data → raw_zarr
    2) preprocess & compute_csd_dask → csd_zarr
    3) compute_beams per‐band → small beam_zarrs
    """
    t0 = UTCDateTime(t0_str)
    t1 = UTCDateTime(t1_str)

    tag = f"{t0.year}_{t0.julday:03d}"
    raw_zarr = os.path.join(outdir, f"raw_{tag}.zarr")
    csd_zarr = os.path.join(outdir, f"csd_{tag}.zarr")

    # -- 1) load & resample
    bf = SeismicBeamformer(
        mseed_dir=mseed_dir,
        inventory=inv,
        t0=t0, t1=t1,
        samp_rate=SAMP_RATE,
        chunk_size_s=CHUNK_S
    )
    bf.load_data(raw_zarr=raw_zarr)
    bf.preprocess(
        band=(0.1, 10.0),
        window_s=WIN_S,
        overlap=OVERLAP,
        pad_s=PAD_S
    )

    # -- 2) compute full‐band CSD to disk
    bf.compute_csd_dask(spill_zarr=csd_zarr)

    # -- 3) beam‐scan each band
    for fmin, fmax in BANDS:
        bf.compute_beams(
            bazs=BAZS,
            slows=SLOWS,
            method=METHOD,
            freq_band=(fmin, fmax)
        )
        # wrap P_fb in xarray and write
        dsb = xr.DataArray(
            bf.P_fb,
            dims=("freq","beam"),
            coords={
                "freq": bf.freqs_sel,
                "beam": np.arange(bf.nbaz * bf.nslow)
            }
        ).to_dataset(name="P_fb")
        out_beam = os.path.join(
            outdir,
            f"beam_{fmin:.2f}-{fmax:.2f}_{tag}.zarr"
        )
        dsb.to_zarr(out_beam, mode="w", consolidated=True)

    return f"[{t0_str} → {t1_str}] done"

def make_monthly_blocks(start_iso, end_iso):
    """
    Slice [start, end) into calendar‐month chunks.
    Partial first/last blocks allowed.
    Returns list of (t0_iso, t1_iso).
    """
    start = UTCDateTime(start_iso)
    end   = UTCDateTime(end_iso)
    blocks = []
    cur = start
    while cur < end:
        # compute next‐month boundary
        year = cur.year
        month = cur.month
        if month == 12:
            nxt = UTCDateTime(f"{year+1}-01-01T00:00:00")
        else:
            nxt = UTCDateTime(f"{year}-{month+1:02d}-01T00:00:00")
        t1 = nxt if nxt < end else end
        blocks.append((cur.isoformat(), t1.isoformat()))
        cur = t1
    return blocks

def main():
    # load inventory once
    inv = read_inventory(INVENTORY_XML)

    # build month‐by‐month blocks between RUN_START→RUN_END
    blocks = make_monthly_blocks(RUN_START, RUN_END)

    os.makedirs(OUTDIR, exist_ok=True)

    # start Dask cluster: 4 workers × 2 threads → 8 total threads
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        memory_limit="12GB",
        dashboard_address=":8787"
    )
    client = Client(cluster)
    print("Dask dashboard at", client.dashboard_link)
    print("Processing blocks:")
    for b in blocks:
        print(" ", b[0], "→", b[1])

    # submit jobs
    futures = [
        client.submit(process_block,
                      inv, MSEED_DIR,
                      t0s, t1s, OUTDIR)
        for t0s, t1s in blocks
    ]

    # wait & report
    for res in client.gather(futures):
        print(res)

    client.close()
    cluster.close()

if __name__ == "__main__":
    main()