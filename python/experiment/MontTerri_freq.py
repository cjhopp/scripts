import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from obspy import read, read_events


# ---------------------------------------------------------------------------
# Receiver coordinates (Easting, Northing, Elevation), same CRS as event XYZ
# ---------------------------------------------------------------------------
STATION_COORDS = {
    # --- Sensors B8x / B9x ---
    "B81": (2579328.52044, 1247577.1019, 488.108),
    "B82": (2579328.42564, 1247576.4328, 487.3709),
    "B83": (2579328.1421, 1247574.4265, 485.1586),
    "B84": (2579328.04796, 1247573.7583, 484.4205),
    "B85": (2579327.76506, 1247571.7556, 482.2049),
    "B86": (2579327.67112, 1247571.0884, 481.466),
    "B91": (2579332.34591, 1247583.0604, 476.5779),
    "B92": (2579332.44796, 1247582.4988, 475.7568),
    "B93": (2579332.75492, 1247580.8165, 473.292),
    "B94": (2579332.85762, 1247580.2557, 472.4704),
    "B95": (2579333.16559, 1247578.5735, 470.0056),
    "B96": (2579333.26661, 1247578.0125, 469.184),

    # --- Stations B301-B322 ---
    "B301": (2579324.496867, 1247592.9675, 491.9781),
    "B302": (2579324.474816, 1247591.3557, 490.0671),
    "B303": (2579324.451961, 1247589.7436, 488.1565),
    "B304": (2579324.428664, 1247588.132, 486.2454),
    "B305": (2579324.405486, 1247586.5206, 484.3342),
    "B306": (2579324.384642, 1247584.9097, 482.4225),
    "B307": (2579324.365244, 1247583.3001, 480.5097),
    "B308": (2579324.345286, 1247581.6901, 478.5972),
    "B309": (2579324.324412, 1247580.0796, 476.6852),
    "B310": (2579324.304438, 1247578.4679, 474.7741),
    "B311": (2579324.286077, 1247576.8557, 472.8636),
    "B312": (2579324.267101, 1247575.2429, 470.9534),
    "B313": (2579324.248199, 1247573.6299, 469.0435),
    "B314": (2579324.22945, 1247572.0169, 467.1336),
    "B315": (2579324.20813, 1247570.4047, 465.223),
    "B316": (2579324.185952, 1247568.7916, 463.3131),
    "B317": (2579324.163273, 1247567.1781, 461.4037),
    "B318": (2579324.14131, 1247565.5641, 459.4946),
    "B319": (2579324.118188, 1247563.9509, 457.5849),
    "B320": (2579324.091171, 1247562.3397, 455.6735),
    "B321": (2579324.062039, 1247560.7305, 453.7606),
    "B322": (2579324.032701, 1247559.1223, 451.8467),

    # --- Stations B401-B422 ---
    "B401": (2579329.17688, 1247597.8316, 492.0198),
    "B402": (2579329.55651, 1247596.495, 489.9415),
    "B403": (2579329.93608, 1247595.1581, 487.8633),
    "B404": (2579330.31607, 1247593.8213, 485.7852),
    "B405": (2579330.69621, 1247592.4842, 483.7073),
    "B406": (2579331.07555, 1247591.1471, 481.6292),
    "B407": (2579331.45355, 1247589.8114, 479.5501),
    "B408": (2579331.83095, 1247588.475, 477.4713),
    "B409": (2579332.20912, 1247587.1386, 475.3925),
    "B410": (2579332.58727, 1247585.8011, 473.3145),
    "B411": (2579332.96631, 1247584.4626, 471.2373),
    "B412": (2579333.34493, 1247583.1243, 469.16),
    "B413": (2579333.72263, 1247581.7854, 467.0827),
    "B414": (2579334.09847, 1247580.4462, 465.0054),
    "B415": (2579334.47514, 1247579.1063, 462.9287),
    "B416": (2579334.85297, 1247577.7667, 460.852),
    "B417": (2579335.23231, 1247576.4266, 458.7759),
    "B418": (2579335.6125, 1247575.0857, 456.7004),
    "B419": (2579335.9923, 1247573.7446, 454.625),
    "B420": (2579336.3728, 1247572.4026, 452.5504),
    "B421": (2579336.7543, 1247571.0595, 450.4766),
    "B422": (2579337.1362, 1247569.7158, 448.4033),

    # --- Extra individual stations ---
    "B31": (2579324.461294, 1247590.3884, 488.9207),
    "B34": (2579324.127923, 1247564.5962, 458.3487),
    "B42": (2579329.86015, 1247595.4255, 488.279),
    "B43": (2579335.9162, 1247574.0129, 455.0401),
    "B551": (2579328.42237, 1247584.2444, 500.604),
    "B585": (2579321.582, 1247563.2318, 478.8632),
    "B647": (2579334.79024, 1247589.14718, 501.6884),
    "B659": (2579337.27833, 1247570.6516, 476.9351),
    "B748": (2579340.15296, 1247593.54974, 503.041),
    "B75": (2579351.2349, 1247579.5168, 477.7178),
}


def _euclidean_distance(p1, p2):
    """3D Euclidean distance between two (x, y, z) points."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))


def _get_trace_distance(tr, event_location, station_coords=STATION_COORDS):
    """
    Resolve source-receiver distance (in the same units as your CRS,
    presumably meters) for a trace.

    Priority:
      1. tr.stats.distance, if already set explicitly.
      2. Computed from event_location=(x, y, z) and station_coords lookup
         keyed by tr.stats.station.
    """
    if getattr(tr.stats, "distance", None) is not None:
        return tr.stats.distance

    station = tr.stats.station
    if station.startswith('C'):
        return np.nan
    if station not in station_coords:
        raise KeyError(
            f"Station '{station}' (trace {tr.id}) not found in "
            "station_coords. Add its (Easting, Northing, Elevation) or "
            "set tr.stats.distance directly."
        )

    return _euclidean_distance(event_location, station_coords[station])


def estimate_centroid_frequency(
    tr,
    pick_time=None,
    window=0.01,
    freq_min=None,
    freq_max=None,
    ax=None,
):
    """
    Estimate the centroid (amplitude-weighted mean) frequency of a trace.

    Parameters
    ----------
    tr : obspy.Trace
    pick_time : obspy.UTCDateTime or None
        If given, trim to [pick_time, pick_time + window] before analysis.
    window : float
        Trim window length in seconds.
    freq_min, freq_max : float or None
        Frequency band to compute the centroid over (Hz).
        Defaults to [1 Hz, Nyquist].
    ax : matplotlib.axes.Axes or None
        If given, plot the spectrum and mark the centroid frequency.

    Returns
    -------
    f_centroid : float
        Centroid frequency in Hz.
    """
    tr = tr.copy()
    if pick_time is not None:
        tr.trim(starttime=pick_time, endtime=pick_time + window)

    data = tr.data.astype(float)
    npts = len(data)
    nyquist = 0.5 * tr.stats.sampling_rate
    freqs = np.fft.rfftfreq(npts, d=tr.stats.delta)
    spectrum = np.abs(np.fft.rfft(data))

    fmin = freq_min if freq_min is not None else 1.0
    fmax = freq_max if freq_max is not None else nyquist
    mask = (freqs > fmin) & (freqs <= fmax)
    f_band = freqs[mask]
    s_band = spectrum[mask]

    f_centroid = np.sum(f_band * s_band) / np.sum(s_band)

    if ax is not None:
        ax.loglog(f_band, s_band, label=f"{tr.stats.station} spectrum")
        ax.axvline(
            f_centroid, color="red", linestyle="--",
            label=f"centroid = {f_centroid:.1f} Hz",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Centroid frequency — {tr.id}")
        ax.legend()

    return f_centroid


def plot_stream_by_distance(
    stream,
    event_location,
    event=None,
    window=0.01,
    station_coords=STATION_COORDS,
    cmap="viridis",
    distance_units="m",
    ax=None,
    linewidth=0.8,
    alpha=0.9,
):
    """
    Plot the amplitude spectrum of all traces in a Stream on a single axis,
    colored by source-receiver distance computed from local Cartesian coordinates.
    Traces whose station is not in station_coords (distance=NaN) are skipped.

    Parameters
    ----------
    stream : obspy.Stream
        Stream containing the traces to plot.
    event_location : tuple(float, float, float)
        (Easting, Northing, Elevation) of the event/source, in the same
        CRS/units as station_coords.
    event : obspy.core.event.Event or None
        If provided, each trace is trimmed to [pick_time, pick_time + window]
        using the pick matching that trace's station. Traces with no matching
        pick are skipped.
    window : float
        Length of the trim window in seconds (default 0.01 s).
    station_coords : dict
        Mapping of station code -> (Easting, Northing, Elevation).
        Defaults to the built-in STATION_COORDS table.
    cmap : str or matplotlib.colors.Colormap
        Colormap used to encode distance.
    distance_units : str
        Label for the colorbar (e.g. 'm').
    ax : matplotlib.axes.Axes or None
        Axis to plot on. A new figure/axis is created if None.
    linewidth : float
        Line width for each trace.
    alpha : float
        Line transparency.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Build station -> pick time lookup from event picks (first pick per station)
    pick_times = {}
    if event is not None:
        for pick in event.picks:
            sta = pick.waveform_id.station_code
            if sta not in pick_times:
                pick_times[sta] = pick.time

    distances = np.array([
        _get_trace_distance(tr, event_location, station_coords)
        for tr in stream
    ])

    valid = ~np.isnan(distances)
    valid_distances = distances[valid]
    valid_stream = [tr for tr, v in zip(stream, valid) if v]

    norm = Normalize(vmin=valid_distances.min(), vmax=valid_distances.max())
    colormap = plt.get_cmap(cmap)

    for tr, dist in zip(valid_stream, valid_distances):
        if pick_times:
            pick_time = pick_times.get(tr.stats.station)
            if pick_time is None:
                continue
            tr = tr.copy().trim(starttime=pick_time, endtime=pick_time + window)
            if tr.stats.npts == 0:
                continue

        data = tr.data.astype(float)
        npts = len(data)
        nyquist = 0.5 * tr.stats.sampling_rate
        freqs = np.fft.rfftfreq(npts, d=tr.stats.delta)
        spectrum = np.abs(np.fft.rfft(data))
        mask = (freqs > 0) & (freqs <= nyquist)
        freqs, spectrum = freqs[mask], spectrum[mask]

        color = colormap(norm(dist))
        ax.plot(freqs, spectrum, color=color, linewidth=linewidth, alpha=alpha)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude spectra colored by source-receiver distance")

    sm = ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"Distance ({distance_units})")

    return fig, ax


if __name__ == "__main__":
    st = read("/media/chopp/HDD1/chet-FS-B/dug-seis/denoise/event_wavs/event_37.ms")

    # Event location in the same CRS as the receiver table above
    event_xyz = (2579323.38996, 1247581.43652, 466.723648976)  # Event 37
    ev37 = read_events('/media/chopp/HDD1/chet-FS-B/dug-seis/denoise/cycle_events/event_37.xml')[0]

    # Build pick-time lookup
    pick_times = {p.waveform_id.station_code: p.time for p in ev37.picks}

    # Corner frequency estimate for B91
    tr_b91 = st.select(station="B91")[0]
    fig_fc, ax_fc = plt.subplots(figsize=(8, 5))
    fc = estimate_centroid_frequency(
        tr_b91,
        pick_time=pick_times.get("B91"),
        window=0.01,
        ax=ax_fc,
        freq_min=400.0,
    )
    print(f"B91 centroid frequency: {fc:.1f} Hz")

    fig, ax = plot_stream_by_distance(st, event_location=event_xyz, event=ev37)
    plt.show()