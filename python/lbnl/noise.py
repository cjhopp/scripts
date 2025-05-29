import os, glob, utm
import numpy as np
import xarray as xr
import dask
import dask.array as da

from tqdm import tqdm
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
        inventory    : ObsPy Inventory (with lat/lon for each station)
        t0, t1       : UTCDateTime window
        samp_rate    : desired sampling rate [Hz]
        chunk_size_s : approximate chunk length along time (seconds)
        """
        self.mseed_dir    = mseed_dir
        self.inv          = inventory
        self.t0, self.t1  = t0, t1
        self.samp_rate    = samp_rate
        self.chunk_size_s = chunk_size_s

        # Will be filled later
        self.ds        = None      # xarray.Dataset of raw timeseries
        self.coords    = None      # UTM coords for each trace (ntr,2)
        self.win_idx   = None      # list of (i0,i1) for sliding windows
        self.freqs     = None      # FFT freqs
        self.Rf        = None      # CSD cube (dask array)
        self.raw_zarr  = None      # path to raw‐Zarr store if written

    def _read_and_resamp(self,
                         fname: str,
                         t0: UTCDateTime,
                         t1: UTCDateTime,
                         samp_rate: float,
                         station: str,
                         comp: str,
                         npts: int) -> np.ndarray:
        """
        Delayed read+trim+resample helper.
        Always returns a float32 array of length npts.
        """
        st = read(fname)
        tr = next((_ for _ in st
                   if _.stats.station == station
                   and _.stats.channel.endswith(comp)), None)
        if tr is None:
            return np.zeros(npts, dtype=np.float32)

        # resample if requested
        if samp_rate is not None:
            tr.resample(samp_rate)
        data = tr.data.astype(np.float32)

        # trim or pad to exactly npts
        if data.size > npts:
            data = data[:npts]
        elif data.size < npts:
            data = np.pad(data, (0, npts - data.size), mode="constant")
        return data

    def load_data(self,
                  date_fmt: str = "%Y.%j.%H.%M.%S.%f",
                  raw_zarr: str = None) -> xr.Dataset:
        """
        1) reads & trims all MSEED files in [t0,t1]
        2) resamples to self.samp_rate
        3) stacks into an xarray.Dataset with dims (tr, time)
        4) converts station lat/lon → UTM
        5) optionally writes the raw timeseries to Zarr
        """
        logging.info("Discovering files…")
        all_files = glob.glob(os.path.join(self.mseed_dir, "**/*"),
                              recursive=True)
        all_files = [f for f in all_files if os.path.isfile(f)]

        # parse dates from filenames (hard-coded convention)
        basenames = [os.path.basename(f)[:-8] for f in all_files]
        dates = np.array([datetime.strptime(fn, date_fmt)
                          for fn in basenames])
        mask = (self.t0.datetime <= dates) & (dates <= self.t1.datetime)
        files = np.array(all_files)[mask]

        # list of stations & comps from Inventory
        stations = sorted({sta.code
                           for net in self.inv
                           for sta in net})
        comps = ["Z","N","E"]

        # determine sampling & number of points
        hdr = read(files[0], headonly=True)[0]
        fs0 = 1.0 / hdr.stats.delta
        total_s = float(self.t1 - self.t0)
        npts = int(np.round(total_s * fs0))
        self.npts = npts
        self.dt   = 1.0 / self.samp_rate

        # map (station,comp) → list of candidate files
        from collections import defaultdict
        filemap = defaultdict(list)
        for fn in files:
            stmp = read(fn, headonly=True)
            for tr in stmp:
                sta = tr.stats.station
                comp_ = tr.stats.channel[-1]
                if sta in stations and comp_ in comps:
                    filemap[(sta,comp_)].append(fn)

        # build one dask array per trace (sta‐comp)
        logging.info("Building dask arrays for each trace…")
        darr_list, tr_names = [], []
        chunks = int(self.chunk_size_s * self.samp_rate)
        for sta in stations:
            for comp_ in comps:
                key = (sta, comp_)
                fns = sorted(filemap.get(key, []))
                if not fns:
                    darr = da.zeros((npts,),
                                    chunks=(chunks,),
                                    dtype=np.float32)
                else:
                    pieces = []
                    for fn in fns:
                        blk = dask.delayed(self._read_and_resamp)(
                            fn, self.t0, self.t1,
                            self.samp_rate, sta, comp_, npts
                        )
                        darr_blk = da.from_delayed(blk,
                                                   shape=(npts,),
                                                   dtype=np.float32)
                        darr_blk = darr_blk.rechunk((chunks,))
                        pieces.append(darr_blk)
                    darr = da.concatenate(pieces, axis=0)[:npts]
                darr_list.append(darr)
                tr_names.append(f"{sta}-{comp_}")

        # stack into (ntr, nt)
        data = da.stack(darr_list, axis=0)
        time = np.arange(npts) * self.dt

        ds = xr.Dataset(
            {"u": (("tr","time"), data)},
            coords={
                "tr":      tr_names,
                "station": [nm.split("-")[0] for nm in tr_names],
                "comp":    [nm.split("-")[1] for nm in tr_names],
                "time":    time
            }
        )
        self.ds = ds

        # convert station → UTM
        logging.info("Computing UTM coordinates…")
        utm_pts = []
        for nm in tr_names:
            sta_code = nm.split("-")[0]
            found = False
            for net in self.inv:
                for sta in net:
                    if sta.code == sta_code:
                        e,n,_,_ = utm.from_latlon(sta.latitude,
                                                  sta.longitude)
                        utm_pts.append((e,n))
                        found = True
                        break
                if found: break
            if not found:
                raise ValueError(f"Station {sta_code} not in inventory")
        self.coords = np.array(utm_pts)

        # optionally spill raw to Zarr
        if raw_zarr:
            logging.info(f"Writing raw timeseries to {raw_zarr}")
            ds.u.to_dataset(name="u") \
                 .to_zarr(raw_zarr,
                          mode="w",
                          consolidated=True)
            self.raw_zarr = raw_zarr

        return ds

    def preprocess(self,
                   band: tuple,
                   window_s: float  = 50.0,
                   overlap: float   = 0.5,
                   pad_s: float     = None) -> None:
        """
        1) Bandpass‐filter via dask.map_overlap
        2) Build sliding‐window index list for CSD
        """
        ds = self.ds
        dt = float(ds.time.values[1] - ds.time.values[0])
        fs = 1.0 / dt

        # design filter
        sos = butter(4, band, fs=fs, btype="bandpass", output="sos")
        pad = int(pad_s * fs) if pad_s else 4*3  # 3×order

        def _filt_block(x: np.ndarray) -> np.ndarray:
            return sosfiltfilt(sos, x, axis=-1)

        logging.info("Applying bandpass filter with map_overlap…")
        filtered = ds.u.data.map_overlap(
            _filt_block,
            depth=(0, pad),
            boundary="reflect",
            trim=True,
            dtype=ds.u.dtype
        )
        self.ds = ds.assign(u=(("tr", "time"), filtered))

        # sliding‐window indices
        step_s = window_s * (1.0 - overlap)
        total_s = float(self.t1 - self.t0)
        nwin = int(np.floor((total_s - window_s) / step_s)) + 1

        idx = []
        for k in range(nwin):
            i0 = int(np.round(k * step_s * fs))
            i1 = i0 + int(np.round(window_s * fs))
            idx.append((i0, i1))
        self.win_idx = idx
        logging.info(f"Prepared {len(idx)} sliding windows")

    @staticmethod
    def _csd_block_dask(arr: np.ndarray,
                        win: np.ndarray) -> np.ndarray:
        """
        arr: (ntr,wlen)
        win: (wlen,)
        → (nfreq, ntr, ntr)
        """
        X = np.fft.rfft(arr * win[None,:], axis=1)    # (ntr,nf)
        R = np.einsum("if,jf->fij", X, X.conj())      # (nf,ntr,ntr)
        return R

    def compute_csd_dask(self,
                        raw_zarr: str = None,
                        spill_zarr: str = None
                        ) -> tuple[da.Array, np.ndarray]:
        """
        1) open raw Zarr
        2) extract sliding‐windows & do map_blocks
        3) sum & normalize → self.Rf, self.freqs
        4) optionally spill R(f,i,j) to Zarr
        """
        raw_zarr = raw_zarr or self.raw_zarr
        if not raw_zarr:
            raise ValueError("must provide raw_zarr in load_data or here")

        logging.info(f"Opening raw timeseries from {raw_zarr}")
        ds_z = xr.open_zarr(raw_zarr, consolidated=True)
        raw  = ds_z.u.data                     # (ntr, nt)
        ntr, nt = raw.shape

        # FFT params
        i0, i1 = self.win_idx[0]
        wlen    = i1 - i0
        dt      = float(self.ds.time.values[1]
                        - self.ds.time.values[0])
        Fs      = 1.0 / dt
        freqs   = np.fft.rfftfreq(wlen, dt)
        win     = get_window("hann", wlen)
        U       = np.sum(win * win)

        logging.info("Computing CSD for each window…")
        R_blocks = []
        for (i0, i1) in self.win_idx:
            slice_ = raw[:, i0:i1]
            slice_ = slice_.rechunk({0: ntr, 1: wlen})
            Rb = da.map_blocks(
                SeismicBeamformer._csd_block_dask,
                slice_,    # (ntr, wlen)
                win,
                dtype=np.complex128,
                chunks=(freqs.size, ntr, ntr)
            )
            R_blocks.append(Rb)
        R_all = da.stack(R_blocks, axis=0)       # (nwin,nf,ntr,ntr)
        R_sum = (R_all.sum(axis=0)            # (nf,ntr,ntr)
                 / (len(self.win_idx) * Fs * U))

        self.freqs = freqs
        self.Rf     = R_sum

        if spill_zarr:
            logging.info(f"Spilling CSD to {spill_zarr}")
            xr.DataArray(R_sum,
                         dims=("freq","tr","tr2"),
                         coords={"freq": freqs,
                                 "tr":   self.ds.tr,
                                 "tr2":  self.ds.tr}
            ).to_dataset(name="Rf") \
             .to_zarr(spill_zarr, mode="w")
        return R_sum, freqs

    @staticmethod
    def _beamform_bartlett_vec(Rsub: np.ndarray,
                               freqs: np.ndarray,
                               delays: np.ndarray,
                               freq_block: int = 64
                              ) -> np.ndarray:
        """
        Vectorized Bartlett beamformer: P[f,b] = a(f,b)^H Rsub[f] a(f,b)
        Chunks over frequencies in blocks of `freq_block` to limit memory.
        Rsub   : (nfreq, ntr, ntr)
        freqs  : (nfreq,)
        delays : (nbeams, ntr)
        returns P_fb : (nfreq, nbeams)
        """
        nfr, ntr, _ = Rsub.shape
        nbeams      = delays.shape[0]
        P = np.zeros((nfr, nbeams), dtype=float)

        # Loop over frequency‐chunks
        for i0 in tqdm(range(0, nfr, freq_block), desc='Beamforming freqs', ncols=80):
            i1    = min(nfr, i0 + freq_block)
            fs    = freqs[i0:i1]                         # (fchunk,)
            Rblk  = Rsub[i0:i1, :, :]                   # (fchunk,ntr,ntr)

            # build steering matrix for this block:
            # Ablk[f,b,i] = exp(-2j*pi * fs[f]*delays[b,i])
            # result shape = (fchunk, nbeams, ntr)
            phase = -2j * np.pi * fs[:, None, None] * delays[None, :, :]
            Ablk  = np.exp(phase)

            # Bartlett P = Re{ sum_{i,j} conj(A_i) * R * A_j }
            # this is exactly a triple‐contract which np.einsum can do in C/BLAS
            # 'fbm,fmn,fbn->fb'
            P[i0:i1] = np.real(
                np.einsum("fbm,fmn,fbn->fb", Ablk.conj(), Rblk, Ablk)
            )

        return P
    
    @staticmethod
    def _beamform_numpy_freq(Rsub: np.ndarray,
                             freqs: np.ndarray,
                             delays: np.ndarray,
                             method: str        = "bartlett",
                             nnoise: int        = None,
                             capon_reg: float   = 1e-3
                            ) -> np.ndarray:
        """
        Vectorized beamforming per-frequency.
        Rsub   : (nfreq, ntr, ntr) small CSD cube
        freqs  : (nfreq,)
        delays : (nbeams, ntr)
        returns P_fb: (nfreq, nbeams)
        """
        nfreq, ntr, _ = Rsub.shape
        nbeams        = delays.shape[0]
        P_fb = np.zeros((nfreq, nbeams), dtype=float)

        # Precompute inverses / subspaces
        if method == "bartlett":
            return SeismicBeamformer._beamform_bartlett_vec(Rsub, freqs, delays)
        elif method == "capon":
            Rinv = np.empty_like(Rsub)
            for k in range(nfreq):
                M      = ntr
                avg_ev = np.trace(Rsub[k]).real / M
                δ      = capon_reg * avg_ev
                Rl     = Rsub[k] + δ * np.eye(M, dtype=Rsub.dtype)
                try:
                    Rinv[k] = np.linalg.inv(Rl)
                except np.linalg.LinAlgError:
                    Rinv[k] = np.linalg.pinv(Rl)

        elif method == "music":
            En = []
            for k in range(nfreq):
                w, V   = np.linalg.eigh(Rsub[k])
                idx    = np.argsort(w)
                En_k   = V[:, idx[:-nnoise]]   # (ntr, nnoise)
                En.append(En_k)
            En = np.stack(En, axis=0)         # (nfreq, ntr, nnoise)

        # Main loop over frequencies
        for k in range(nfreq):
            f   = freqs[k]
            Rf  = Rsub[k]                   # (ntr,ntr)
            A   = np.exp(-2j*np.pi*f*delays)  # (nbeams,ntr)

            if method == "capon":
                for b in range(nbeams):
                    ak         = A[b][:,None]     # (ntr,1)
                    den        = (ak.conj().T @ Rinv[k] @ ak).real.item()
                    P_fb[k,b] = 1.0 / den

            else:  # MUSIC
                for b in range(nbeams):
                    ak         = A[b][:,None]
                    num        = (ak.conj().T @ ak).real
                    D          = En[k].dot(En[k].conj().T)
                    den        = (ak.conj().T @ D @ ak).real
                    P_fb[k,b] = (num/den).item()

        return P_fb

    def compute_beams(self,
                      bazs:      np.ndarray,
                      slows:     np.ndarray,
                      method:     str     = "bartlett",
                      nnoise:     int     = None,
                      capon_reg:  float   = 1e-3,
                      freq_band:  tuple   = None):
        """
        Compute once and store:
          - self.bazs, self.slows
          - self.freqs_sel  (selected freq indices)
          - self.P_fb       (nfreq_sel x (nbaz*nslow) beam powers)
        You can then call any plot_* method quickly afterwards.
        """
        # store beam grid
        self.bazs, self.slows = bazs, slows
        nbaz, nslow = len(bazs), len(slows)

        # 1) pick freq‐indices
        f_all = self.freqs
        if freq_band:
            fmin,fmax = freq_band
            idx = np.where((f_all>=fmin)&(f_all<=fmax))[0]
        else:
            idx = np.arange(f_all.size)
        self.freqs_sel = f_all[idx]

        # 2) load small CSD cube once
        Rsub = self.Rf[idx,:,:].compute()

        # 3) build delays (nbaz*nslow, ntr)
        if np.nanmax(slows) > 1e-2:
            slows_m = slows * 1e-3
        else:
            slows_m = slows.copy()

        xy   = self.coords            # (ntr,2)
        th   = np.deg2rad(bazs)       # (nbaz,)
        dirs = np.stack([np.sin(th),
                         np.cos(th)], axis=1)  # (nbaz,2)

        ntr = xy.shape[0]
        delays = np.empty((nbaz*nslow, ntr), dtype=float)
        cnt = 0
        for p in dirs:
            for s in slows_m:
                delays[cnt,:] = (xy @ p) * s
                cnt += 1

        # 4) vectorized per-frequency beamforming
        self.P_fb = SeismicBeamformer._beamform_numpy_freq(
                        Rsub,
                        self.freqs_sel,
                        delays,
                        method.lower(),
                        nnoise,
                        capon_reg
                    )

        # stash dims
        self.nbaz, self.nslow = nbaz, nslow
        return

    def plot_band(self,
                  fmin:     float,
                  fmax:     float,
                  ax:       plt.Axes = None,
                  cmap:     str      = "viridis",
                  shading:  str      = "nearest",
                  rmax:     float    = None):
        """
        Plot a single 2D polar map for frequencies in [fmin,fmax].
        Must call compute_beams() first.
        """
        if not hasattr(self, "P_fb"):
            raise RuntimeError("Call compute_beams(...) before plotting")

        # select freq‐slice inside self.freqs_sel
        mask = (self.freqs_sel >= fmin) & (self.freqs_sel <= fmax)
        if not mask.any():
            raise ValueError(f"No freqs in [{fmin},{fmax}]")
        P_slice = self.P_fb[mask,:].sum(axis=0)  # (nbaz*nslow,)
        P2d     = P_slice.reshape(self.nbaz, self.nslow)  # (nbaz,nslow)

        # build grid for pcolormesh
        thetas = np.deg2rad(self.bazs)
        rs     = self.slows
        Theta, R = np.meshgrid(thetas, rs, indexing="xy")
        Z = P2d.T   # (nslow, nbaz)

        # prepare axes
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection":"polar"})
        pcm = ax.pcolormesh(Theta, R, Z,
                            cmap=cmap,
                            shading=shading)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, rmax or rs.max())
        ax.set_title(f"{fmin:.2f}–{fmax:.2f} Hz")
        return pcm

    def plot_three_bands(self,
                         bands:  list    = [(0.1,1.0),
                                           (1.0,5.0),
                                           (5.0,10.0)],
                         cmap:   str     = "viridis",
                         shading:str     = "nearest",
                         rmax:   float   = None):
        """
        Quick convenience: plots all bands side‐by‐side.
        Must call compute_beams() first.
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, "P_fb"):
            raise RuntimeError("Call compute_beams(...) before plotting")

        n = len(bands)
        fig, axes = plt.subplots(1, n,
                                 subplot_kw={"projection":"polar"},
                                 figsize=(5*n,5))
        if n == 1:
            axes = [axes]

        for ax, (f1, f2) in zip(axes, bands):
            pcm = self.plot_band(f1, f2,
                                 ax=ax,
                                 cmap=cmap,
                                 shading=shading,
                                 rmax=rmax)
            # colorbar to the right
            cb = fig.colorbar(pcm, ax=ax, pad=0.05)
            cb.set_label("Beam power")

        plt.tight_layout()
        plt.show()
        return