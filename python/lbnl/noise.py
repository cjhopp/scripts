import os
import glob
import utm
import numpy as np
import xarray as xr
import dask
import dask.array as da
from datetime import datetime
from obspy import read, UTCDateTime
from obspy.core.inventory import Inventory
from scipy.signal import get_window, butter, sosfiltfilt
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)


class SeismicBeamformer:
    def __init__(self,
                 mseed_dir: str,
                 inventory: Inventory,
                 t0: UTCDateTime,
                 t1: UTCDateTime,
                 samp_rate: float = 50.0,
                 chunk_size_s: float = 100.0):
        """
        mseed_dir    : directory with MiniSEED files
        inventory    : ObsPy Inventory (with lat/lon)
        t0, t1       : UTCDateTime window
        samp_rate    : desired sampling rate [Hz]
        chunk_size_s : size of time‐chunks for Dask
        """
        self.mseed_dir = mseed_dir
        self.inv       = inventory
        self.t0, self.t1 = t0, t1
        self.sr        = samp_rate
        self.chunk_s   = chunk_size_s

        # to be populated:
        self.ds       = None       # xarray Dataset with raw time‐series
        self.coords   = None       # (ntr,2) UTM coords
        self.win_idx  = None       # list of (i0,i1) windows
        self.Rf       = None       # Dask array (nf, ntr, ntr)
        self.freqs    = None       # np.ndarray of length nf
        self.P        = None       # xarray.DataArray (freq,baz,slow)

    @staticmethod
    def _read_trace(station: str,
                    comp:    str,
                    files:   list,
                    t0:      UTCDateTime,
                    t1:      UTCDateTime,
                    sr:      float,
                    npts:    int) -> np.ndarray:
        """
        Reads all files for one (station,comp), merges, trims+pads, resamples.
        Returns a float32 array of length npts.
        """
        from obspy import Stream
        st = Stream()
        # sort by starttime
        def _start(f):
            return read(f, headonly=True)[0].stats.starttime
        for fn in sorted(files, key=_start):
            st += read(fn, headonly=True)
        st.merge(fill_value=0)
        tr = next((t for t in st
                   if t.stats.station == station
                   and t.stats.channel.endswith(comp)),
                  None)
        if tr is None:
            return np.zeros(npts, dtype=np.float32)

        # trim (with padding), then resample
        tr.trim(t0, t1, pad=True, fill_value=0)
        tr.resample(sr)
        data = tr.data.astype(np.float32)

        # final safety‐check
        if data.size < npts:
            data = np.pad(data, (0, npts - data.size), "constant")
        return data[:npts]

    def load_data(self,
                  date_fmt: str = "%Y.%j.%H.%M.%S.%f") -> xr.Dataset:
        """
        1) Discover files in [t0,t1]
        2) Build one delayed merge/trim/resample per trace
        3) Stack into xarray.Dataset with dims (tr, time)
        4) Compute UTM coords
        """
        logging.info("Discovering files…")
        all_files = glob.glob(os.path.join(self.mseed_dir, "**/*"),
                              recursive=True)
        all_files = [f for f in all_files if os.path.isfile(f)]

        # parse times from filename
        basenames = [os.path.basename(f)[:-8] for f in all_files]
        dates = np.array([datetime.strptime(bn, date_fmt)
                          for bn in basenames])
        mask = ((self.t0.datetime <= dates) &
                (dates <= self.t1.datetime))
        files = np.array(all_files)[mask].tolist()

        # list of stations & comps from Inventory
        stations = sorted({sta.code
                           for net in self.inv
                           for sta in net})
        comps    = ["Z","N","E"]

        # total samples from target sr:
        total_s = float(self.t1 - self.t0)
        npts    = int(round(total_s * self.sr))
        dt      = 1.0 / self.sr
        self.npts = npts
        self.dt   = dt

        # map (sta,comp) -> files
        from collections import defaultdict
        fmap = defaultdict(list)
        for fn in files:
            hdrs = read(fn, headonly=True)
            for tr in hdrs:
                sta  = tr.stats.station
                comp = tr.stats.channel[-1]
                if sta in stations and comp in comps:
                    fmap[(sta,comp)].append(fn)

        # build one delayed trace per (sta,comp)
        chunks = int(self.chunk_s * self.sr)
        darrs, names = [], []
        for sta in stations:
            for cp in comps:
                fns = fmap.get((sta,cp), [])
                blk = dask.delayed(self._read_trace)(
                          sta, cp, fns, self.t0, self.t1,
                          self.sr, npts
                      )
                darr = da.from_delayed(blk,
                                       shape=(npts,),
                                       dtype=np.float32)
                darr = darr.rechunk((chunks,))
                darrs.append(darr)
                names.append(f"{sta}-{cp}")

        data = da.stack(darrs, axis=0)  # (ntr, nt)
        time = np.arange(npts) * dt

        ds = xr.Dataset(
            {"u": (("tr","time"), data)},
            coords={
                "tr":      names,
                "station": [nm.split("-")[0] for nm in names],
                "comp":    [nm.split("-")[1] for nm in names],
                "time":    time
            }
        )
        self.ds = ds

        # compute UTM coords
        logging.info("Computing UTM coordinates…")
        utm_pts = []
        for nm in names:
            code = nm.split("-")[0]
            found = False
            for net in self.inv:
                for sta in net:
                    if sta.code == code:
                        e,n,_,_ = utm.from_latlon(sta.latitude,
                                                  sta.longitude)
                        utm_pts.append((e,n))
                        found = True
                        break
                if found: break
            if not found:
                raise ValueError(f"Station {code} not in inventory")
        self.coords = np.array(utm_pts)
        return ds

    def preprocess(self,
                   band:     tuple,
                   window_s: float   = 50.0,
                   overlap:  float   = 0.5,
                   pad_s:    float   = None) -> None:
        """
        1) Bandpass‐filter via map_overlap
        2) Build sliding‐window index list for CSD
        """
        ds  = self.ds
        dt  = float(ds.time.values[1] - ds.time.values[0])
        fs  = 1.0 / dt
        sos = butter(4, band, fs=fs, btype="bandpass", output="sos")
        pad = int(pad_s*fs) if pad_s else 4*3

        def _fblock(x):
            return sosfiltfilt(sos, x, axis=-1)

        logging.info("Applying bandpass filter…")
        u_filt = ds.u.data.map_overlap(_fblock,
                                       depth=(0,pad),
                                       boundary="reflect",
                                       trim=True,
                                       dtype=ds.u.dtype)
        self.ds = ds.assign(u=(("tr","time"), u_filt))

        # sliding‐window indices
        step_s = window_s * (1-overlap)
        nwin   = int(np.floor((float(self.t1-self.t0)-window_s)
                              / step_s)) + 1
        idx = []
        for k in range(nwin):
            i0 = int(round(k*step_s*fs))
            i1 = i0 + int(round(window_s*fs))
            idx.append((i0,i1))
        logging.info(f"Prepared {nwin} windows for CSD")
        self.win_idx = idx

    @staticmethod
    def _csd_block_dask(arr: np.ndarray,
                        win: np.ndarray) -> np.ndarray:
        """
        arr: (ntr, wlen) → returns (nfreq, ntr, ntr)
        """
        X = np.fft.rfft(arr * win[None,:], axis=1)    # (ntr,nf)
        R = np.einsum("if,jf->fij", X, X.conj())      # (nf,ntr,ntr)
        return R

    def compute_csd_dask(self) -> None:
        """
        1) Loops over windows
        2) map_blocks → accumulate sum → normalize
        Stores self.Rf (da.Array) and self.freqs (np.ndarray)
        """
        logging.info("Computing CSD via Dask…")
        raw = self.ds.u.data   # (ntr, nt)
        ntr, nt = raw.shape

        # FFT params
        i0, i1 = self.win_idx[0]
        wlen    = i1 - i0
        freqs   = np.fft.rfftfreq(wlen, self.dt)
        win     = get_window("hann", wlen)
        U       = np.sum(win*win)
        Fs      = 1.0 / self.dt

        R_sum = None
        for i0,i1 in self.win_idx:
            blk = raw[:, i0:i1].rechunk({0:ntr, 1:wlen})
            Rw  = da.map_blocks(self._csd_block_dask,
                                blk, win,
                                dtype=np.complex128,
                                chunks=(freqs.size,ntr,ntr))
            if R_sum is None:
                R_sum = Rw
            else:
                R_sum = R_sum + Rw

        # normalize
        R_sum = R_sum / (len(self.win_idx)*Fs*U)

        self.Rf     = R_sum
        self.freqs  = freqs

    def compute_beams(self,
                      bazs:      np.ndarray,
                      slows:     np.ndarray,
                      method:    str      = "bartlett",
                      nnoise:    int      = None,
                      capon_reg: float    = 1e-3,
                      freq_band: tuple    = None) -> None:
        """
        1) pick freq indices
        2) compute delays
        3) run Bartlett (vectorized) or Capon/MUSIC
        4) wrap into xarray.DataArray self.P(freq,baz,slow)
        """
        if self.Rf is None:
            raise RuntimeError("Call compute_csd_dask() first")

        # select freqs
        f_all = self.freqs
        if freq_band:
            fmin,fmax = freq_band
            fi = np.where((f_all>=fmin)&(f_all<=fmax))[0]
        else:
            fi = np.arange(f_all.size)
        freqs = f_all[fi]

        # get small CSD and compute
        Rsub = self.Rf[fi,:,:].compute()   # (nf,ntr,ntr)
        nfreq, ntr, _ = Rsub.shape

        nbaz, nslow = len(bazs), len(slows)
        # convert slows → s/m
        slows_m = (slows*1e-3
                   if np.nanmax(slows)>1e-2
                   else slows.copy())

        # delays: (nbaz*nslow, ntr)
        th   = np.deg2rad(bazs)
        dirs = np.stack([np.sin(th), np.cos(th)],axis=1)  # (nbaz,2)
        xy   = self.coords                             # (ntr,2)
        delays = np.empty((nbaz*nslow, ntr), dtype=float)
        idx    = 0
        for d in dirs:
            for s in slows_m:
                delays[idx,:] = (xy @ d) * s
                idx += 1

        # ------- Bartlett vectorized -------
        if method.lower()=="bartlett":
            # A: (nf, nbeam, ntr)
            A = np.exp(-2j*np.pi*freqs[:,None,None] * delays[None,:,:])
            # R·Aᴴ → (nf,ntr,nbeam)
            RA = np.einsum("fij,fbj->fbi", Rsub, A.conj())
            P  = np.real(np.einsum("fbj,fbj->fb",
                                   A.conj(), RA))
        else:
            # fallback to your existing loops for capon/music
            from warnings import warn
            warn("Only 'bartlett' is vectorized; "
                 "falling back to slower loops.")
            P = np.zeros((nfreq, nbaz*nslow), dtype=float)
            # ... insert your loop for capon/music here ...
            raise NotImplementedError("Capon/MUSIC not implemented here")

        # reshape → (freq, baz, slow)
        P3 = P.reshape(nfreq, nbaz, nslow)

        da_P = xr.DataArray(
            P3,
            dims=("freq","baz","slow"),
            coords={"freq": freqs,
                    "baz":  bazs,
                    "slow": slows}
        )
        self.P = da_P

    def plot_beam(self,
                  fmin:     float,
                  fmax:     float,
                  ax:        plt.Axes = None,
                  cmap:      str       = "viridis",
                  shading:   str       = "nearest",
                  rmax:      float     = None):
        """
        Plot the 2D polar map for frequencies in [fmin,fmax].
        """
        if self.P is None:
            raise RuntimeError("Call compute_beams() first")

        Psel = self.P.sel(freq=slice(fmin,fmax)).sum(dim="freq")
        thetas = np.deg2rad(self.P.baz.values)
        rs     = self.P.slow.values
        Theta, R = np.meshgrid(thetas, rs, indexing="xy")
        Z = Psel.values.T  # (slow, baz)

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection":"polar"})
        pcm = ax.pcolormesh(Theta, R, Z,
                            cmap=cmap, shading=shading)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, rmax or rs.max())
        ax.set_title(f"{fmin:.2f}–{fmax:.2f} Hz")
        return pcm

    def plot_three_bands(self,
                         bands: list  = [(0.1,1.0),
                                         (1.0,5.0),
                                         (5.0,10.0)],
                         **kwargs):
        """
        Plot three bands side-by-side.
        """
        n = len(bands)
        fig, axs = plt.subplots(1, n,
                                subplot_kw={"projection":"polar"},
                                figsize=(5*n,5))
        if n==1:
            axs = [axs]
        for ax,(f1,f2) in zip(axs,bands):
            pcm = self.plot_beam(f1,f2, ax=ax, **kwargs)
            cb = fig.colorbar(pcm, ax=ax, pad=0.05)
            cb.set_label("Beam power")
        plt.tight_layout()
        plt.show()