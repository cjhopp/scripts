import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Replace this with your actual import
# e.g. from mypackage.beamformer import SeismicBeamformer
# -------------------------------------------------------------------
from lbnl.noise import SeismicBeamformer  
# -------------------------------------------------------------------

def make_array_geometry(ntr=10, radius=1000.0):
    """
    place ntr sensors evenly on a circle of given radius (meters)
    """
    thetas = np.linspace(0, 2*np.pi, ntr, endpoint=False)
    x = radius * np.cos(thetas)
    y = radius * np.sin(thetas)
    return np.vstack([x, y]).T  # shape (ntr,2)

def synth_csd_from_sources(coords, freqs, azs, slows):
    """
    coords : (ntr,2) array of [x,y] in meters
    freqs  : length‐M array of frequencies
    azs    : length‐M array of back‐azimuths [deg]
    slows  : length‐M array of slownesses [s/m]
    returns Rf : (M, ntr, ntr) complex ndarray
    """
    ntr = coords.shape[0]
    M   = len(freqs)
    Rf  = np.zeros((M, ntr, ntr), dtype=np.complex128)
    for i, (f, baz, s) in enumerate(zip(freqs, azs, slows)):
        # steering delays
        th = np.deg2rad(baz)
        p  = np.array([np.sin(th), np.cos(th)])  # unit vector
        delays = coords.dot(p) * s               # (ntr,)
        a = np.exp(-2j*np.pi * f * delays)       # steering vector
        Rf[i] = np.outer(a, a.conj())
    return Rf

def plot_synthetic_band(coords,
                        Rsub,       # (nfr,ntr,ntr) numpy array
                        freqs,      # (nfr,)
                        grid_baz,   # 1D array of candidate back‐azimuths
                        grid_slow,  # 1D array of candidate slownesses
                        band_label):
    """
    Wrap the CSD in dask, run compute_beams + plot_band, show plot.
    """
    nfr, ntr, _ = Rsub.shape

    # make a minimal bf instance
    bf = SeismicBeamformer.__new__(SeismicBeamformer)
    bf.coords = coords
    bf.freqs  = freqs
    # wrap Rsub in a tiny dask array so compute_beams works unchanged
    bf.Rf     = da.from_array(Rsub, chunks=(nfr, ntr, ntr))

    # run beam‐scan (bartlett)
    bf.compute_beams(grid_baz,
                     grid_slow,
                     method="bartlett",
                     freq_band=(freqs.min(), freqs.max()))
    P = bf.P_fb.sum(axis=0)   # P is 1D of length nbeams
    idx = np.argmax(P)
    ib, islow = np.unravel_index(idx, (len(grid_baz), len(grid_slow)))
    print("Peak at baz=", grid_baz[ib], " slow=", grid_slow[islow],
        " power=", P[idx])
    # now plot
    vmax = P.max()
    fig, ax = plt.subplots(subplot_kw={"projection":"polar"},
                           figsize=(5,4))
    pcm = bf.plot_band(freqs.min(), freqs.max(),
                       ax=ax,
                       cmap="viridis",
                       shading="nearest",
                       rmax=grid_slow.max())
    pcm.set_clim(vmax*.95, vmax)
    ax.set_title(band_label, pad=20)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.1)
    cbar.set_label("Beam power")
    plt.show()


if __name__ == "__main__":

    # 1) build geometry
    coords = make_array_geometry(ntr=10, radius=1000.0)

    # 2) define the scan grid
    grid_baz  = np.linspace(0,360,181)         # every 2°
    grid_slow = np.linspace(1e-4,0.001,101)       # up to 3 ms/m

    # Band #1: 0.1–1 Hz → Gaussian cloud around 0° at 0.5Hz
    M0     = 300
    f0     = 0.5
    az0    = np.random.normal(0, 30, M0)  % 360
    s0     = np.random.uniform(0, 0.0025, M0)
    # build a *single* CSD by summing all outer‐products at the same f0
    Rsingle = np.zeros((1, coords.shape[0], coords.shape[0]), dtype=complex)
    for baz, slow in zip(az0, s0):
        th     = np.deg2rad(baz)
        p      = np.array([np.sin(th), np.cos(th)])
        delays = coords.dot(p) * slow
        a      = np.exp(-2j*np.pi*f0*delays)
        Rsingle[0] += np.outer(a, a.conj())
    freqs0 = np.array([f0])
    plot_synthetic_band(coords, Rsingle, freqs0,
                        grid_baz, grid_slow,
                        "0.1–1 Hz: Gaussian cloud @0°")

    # 1) BAND #2 : single plane‐wave at 3 Hz FROM 180°, v=2000 m/s
    f2    = 3.0
    az2   = np.array([180.0])
    s2    = np.array([1/2000.0])
    # build one CSD at 3 Hz by summing all sources (here just one!)
    R2 = np.zeros((1,coords.shape[0],coords.shape[0]), dtype=np.complex128)
    for baz, slow in zip(az2, s2):
        th    = np.deg2rad(baz)
        p     = np.array([np.sin(th), np.cos(th)])
        delays= coords.dot(p)*slow
        a     = np.exp(-2j*np.pi*f2*delays)
        R2[0] += np.outer(a, a.conj())
    plot_synthetic_band(coords, R2, np.array([f2]),
                        grid_baz, grid_slow,
                        "1–5 Hz : single source @180°, 2 km/s")


    # 2) BAND #3 : four plane‐waves at 7 Hz from N/E/S/W, v=4000
    f3    = 7.0
    az3   = np.array([0, 90,180,270], dtype=float)
    s3    = np.full(4, 1/4000.0)
    # again, build one CSD at 7 Hz by summing the four outer‐products
    R3 = np.zeros((1,coords.shape[0],coords.shape[0]), dtype=np.complex128)
    for baz, slow in zip(az3, s3):
        th     = np.deg2rad(baz)
        p      = np.array([np.sin(th), np.cos(th)])
        delays = coords.dot(p)*slow
        a      = np.exp(-2j*np.pi*f3*delays)
        R3[0] += np.outer(a, a.conj())
    plot_synthetic_band(coords, R3, np.array([f3]),
                        grid_baz, grid_slow,
                        "5–10 Hz : four cardinal sources @4 km/s")