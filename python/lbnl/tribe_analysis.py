#!/usr/bin/env python
"""
tribe_analysis.py
-----------------
Quantitative tools for selecting which SEED channels (picks) to include in
an EQcorrscan Tribe.  Produces diagnostic plots and a ranked channel list
before any waveform downloads for template construction.

Workflow
--------
1.  Call ``analyze_picks()`` on your catalog.  A short waveform window is
    fetched around each event to compute per-pick SNR (no full-day downloads
    during analysis).  Summary statistics and six diagnostic plots are written
    to ``out_dir``.

2.  Inspect the plots and choose thresholds.

3.  Call ``select_seeds()`` to produce a filtered list of SEED IDs, optionally
    limited by minimum event-coverage fraction, minimum median SNR, maximum
    epicentral distance, and a hard cap on the total number of channels.

4.  Call ``build_tribe()`` (thin wrapper around ``template_gen``) to construct
    the Tribe with only the chosen channels.

Quick-start
-----------
::

    from obspy import read_events
    from obspy.clients.fdsn import Client
    from tribe_analysis import analyze_picks, select_seeds, build_tribe

    catalog = read_events("events.xml")
    client  = Client("IRIS")

    # Parameters forwarded to template_gen; also used to size the SNR window.
    parameters = dict(
        lowcut=2.0, highcut=15.0, samp_rate=50.0, filt_order=4,
        length=6.0, prepick=0.5, swin="all", process_len=86400,
    )

    summary = analyze_picks(
        catalog=catalog,
        client=client,
        parameters=parameters,
        out_dir="tribe_analysis",
        noise_dur=20.0,     # seconds of pre-pick noise for SNR
        max_pick_distance_km=300.0,   # exclude stations > 300 km away
        per_template_plots=True,      # write per-template quality plots
        per_template_top_n=20,        # channels shown per template plot
        per_template_min_snr=3.0,     # pass/fail guideline on per-template plots
        # Checkpoints are always used; set force_recompute=True to refresh.
    )

    seeds = select_seeds(
        summary,
        min_event_frac=0.30,   # pick present on ≥30 % of events
        min_median_snr=3.0,
        max_dist_km=200.0,
        max_seeds=20,
    )
    print("Selected seeds:", seeds)

    tribe = build_tribe(catalog, client, parameters, seeds)
    tribe.write("tribe.tgz")

Notes
-----
* SNR is defined as  max(|signal|) / RMS(noise),  matching the internal
  definition used by eqcorrscan's ``_template_gen``.
* The noise window is  [pick.time - prepick - noise_dur,  pick.time - prepick].
* The signal window is [pick.time - prepick,             pick.time - prepick + length].
* Only the channels that appear in ``seeds`` are kept in the catalog before
  the final ``template_gen`` call; events with no remaining picks are dropped.
"""

import os
import re
import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

log = logging.getLogger(__name__)

# FDSN channel codes: 1–3 chars of A-Z, 0-9, or wildcard (*/?)
# "--" (SEED blank-location sentinel stored in the wrong field) and other
# dash-only values must be excluded before any FDSN query is attempted.
_VALID_CHA = re.compile(r'^[A-Z0-9?*]{1,3}$', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rms(array):
    """RMS amplitude of a numpy array (fallback if eqcorrscan is absent)."""
    return float(np.sqrt(np.mean(np.square(array))))


def _pick_seed_id(pick) -> str | None:
    """Return SEED id for a pick only when station/channel codes are usable."""
    wid = getattr(pick, "waveform_id", None)
    if not wid:
        return None
    if not wid.station_code or not wid.channel_code:
        return None
    if not _VALID_CHA.match(wid.channel_code):
        return None
    return wid.get_seed_string()


try:
    from eqcorrscan.core.template_gen import _rms as _eq_rms
    _rms = _eq_rms  # prefer the upstream version
except ImportError:
    pass


def _event_id(ev):
    """Return a short string identifier for an event."""
    rid = ev.resource_id.id
    # Try to extract a meaningful fragment rather than the full URI
    for sep in ("=", "/", "#"):
        parts = rid.rsplit(sep, 1)
        if len(parts) == 2 and parts[1]:
            return parts[1]
    return rid


def _preferred_origin(ev):
    """Return preferred origin; raises ValueError if none is present."""
    org = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)
    if org is None:
        raise ValueError(f"Event {_event_id(ev)} has no origin.")
    return org


def _resolve_station_coords(
    catalog,
    inventory=None,
    client=None,
) -> dict[str, tuple[float, float]]:
    """Resolve station coordinates for all usable seeds in a catalog."""
    if inventory is None and client is None:
        return {}

    all_seeds: set[str] = set()
    for ev in catalog:
        for p in ev.picks:
            seed = _pick_seed_id(p)
            if seed is None:
                continue
            all_seeds.add(seed)

    sta_coords: dict[str, tuple[float, float]] = {}
    _fdsn_client_used = False

    for seed in sorted(all_seeds):
        net, sta, loc, cha = seed.split(".")
        lat, lon = None, None

        if inventory is not None:
            chans = inventory.select(
                network=net, station=sta, location=loc, channel=cha
            ).networks
            for netobj in chans:
                for staobj in netobj.stations:
                    for chaobj in staobj.channels:
                        lat, lon = chaobj.latitude, chaobj.longitude
                        break
                    if lat is not None:
                        break
                if lat is not None:
                    break
            if lat is None:
                chans2 = inventory.select(network=net, station=sta).networks
                for netobj in chans2:
                    for staobj in netobj.stations:
                        lat, lon = staobj.latitude, staobj.longitude
                        break
                    if lat is not None:
                        break

        if lat is None and client is not None:
            try:
                inv_q = client.get_stations(
                    network=net, station=sta,
                    starttime=UTCDateTime(0), endtime=UTCDateTime(),
                    level="channel",
                )
                for netobj in inv_q.networks:
                    for staobj in netobj.stations:
                        for chaobj in staobj.channels:
                            if chaobj.code == cha:
                                lat, lon = chaobj.latitude, chaobj.longitude
                                break
                        if lat is None:
                            lat, lon = staobj.latitude, staobj.longitude
                        break
                    if lat is not None:
                        break
                if not _fdsn_client_used:
                    log.info("Querying FDSN client for station coordinates …")
                _fdsn_client_used = True
            except Exception as exc:
                log.debug(f"  Could not resolve {seed} from client: {exc}")

        if lat is None:
            log.debug(f"  Could not resolve coordinates for {seed}; skipping distance.")
            continue
        sta_coords[seed] = (float(lat), float(lon))

    return sta_coords


def _savefig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


def _make_map_axes(figsize=(10, 8)):
    """Create a cartopy map axes when available, else a plain matplotlib axes."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        have_cartopy = True
    except ImportError:
        ccrs = None
        cfeature = None
        have_cartopy = False

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()} if have_cartopy else {},
    )

    if have_cartopy:
        ax.add_feature(cfeature.LAND, facecolor="lightgrey")
        ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        transform_kw = {"transform": ccrs.PlateCarree()}
    else:
        transform_kw = {}
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    return fig, ax, transform_kw


# ---------------------------------------------------------------------------
# Step 1 – Catalog-level pick statistics (no waveforms needed)
# ---------------------------------------------------------------------------

def _catalog_pick_stats(catalog) -> pd.DataFrame:
    """
    Build pick-count statistics from the catalog without fetching any data.

    Returns a DataFrame with one row per SEED ID containing:
      seed_id, n_picks, n_events_with_pick, event_coverage_frac, phase_types.
    ``n_events_total`` is also stored as a scalar attribute on the returned
    object via ``df.attrs``.
    """
    n_events = len(catalog)
    per_seed: dict[str, dict] = {}

    for ev in catalog:
        seeds_this_event: set[str] = set()
        for pick in ev.picks:
            seed = _pick_seed_id(pick)
            if seed is None:
                continue
            phase = pick.phase_hint or "?"
            if seed not in per_seed:
                per_seed[seed] = {"n_picks": 0, "events": set(), "phases": set()}
            per_seed[seed]["n_picks"] += 1
            per_seed[seed]["phases"].add(phase)
            seeds_this_event.add(seed)
        for seed in seeds_this_event:
            per_seed[seed]["events"].add(_event_id(ev))

    rows = []
    for seed, d in per_seed.items():
        rows.append({
            "seed_id": seed,
            "n_picks": d["n_picks"],
            "n_events_with_pick": len(d["events"]),
            "event_coverage_frac": len(d["events"]) / n_events if n_events else 0.0,
            "phase_types": ",".join(sorted(d["phases"])),
        })
    df = pd.DataFrame(rows).sort_values("n_picks", ascending=False).reset_index(drop=True)
    df.attrs["n_events_total"] = n_events
    return df


# ---------------------------------------------------------------------------
# Step 2 – Per-pick SNR via short waveform fetches
# ---------------------------------------------------------------------------

def _compute_pick_snrs(
    catalog,
    client,
    length: float,
    prepick: float,
    noise_dur: float = 20.0,
) -> pd.DataFrame:
    """
    Fetch a short waveform window around each event and compute per-pick SNR.

    SNR = max(|signal|) / RMS(noise), consistent with eqcorrscan's
    ``_template_gen`` internal calculation.

    Parameters
    ----------
    catalog : obspy.Catalog
    client : FDSN client with ``get_waveforms`` / ``get_waveforms_bulk``
    length : float   template length in seconds
    prepick : float  pre-pick time in seconds
    noise_dur : float  noise window duration (seconds before pick)
    Returns
    -------
    pd.DataFrame  columns: event_id, seed_id, phase_hint, snr, pick_time
    """
    events = sorted(catalog.events, key=lambda e: _preferred_origin(e).time)

    records = []
    n = len(events)

    for i, ev in enumerate(events):
        eid = _event_id(ev)
        valid_picks = [p for p in ev.picks if _pick_seed_id(p) is not None]
        if not valid_picks:
            continue

        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"  SNR fetch: event {i+1}/{n}  ({eid})")

        pick_times = [p.time for p in valid_picks]
        t_start = min(pick_times) - prepick - noise_dur - 5.0
        t_end   = max(pick_times) + length + 5.0

        # Build bulk request (deduplicated)
        # Use float timestamps in the hash key because UTCDateTime.__hash__
        # does not return an integer in all obspy versions.
        # Picks with invalid channel codes (e.g. "--") are skipped; they
        # would cause a 422 from the EarthScope FDSN dataselect endpoint.
        bulk = []
        seen: set[tuple] = set()
        for p in valid_picks:
            wid = p.waveform_id
            net = wid.network_code  or "*"
            sta = wid.station_code
            loc = wid.location_code or "*"
            cha = wid.channel_code
            key = (net, sta, loc, cha, float(t_start), float(t_end))
            if key not in seen:
                seen.add(key)
                bulk.append((net, sta, loc, cha, t_start, t_end))

        if not bulk:
            log.debug(f"  Event {eid}: no valid picks after channel filtering; skipping.")
            continue

        try:
            if hasattr(client, "get_waveforms_bulk"):
                st = client.get_waveforms_bulk(bulk)
            else:
                # Fall back to sequential requests for clients without bulk
                from obspy import Stream
                st = Stream()
                for net, sta, loc, cha, ts, te in bulk:
                    try:
                        st += client.get_waveforms(
                            net, sta, loc, cha, UTCDateTime(ts), UTCDateTime(te)
                        )
                    except Exception as exc_inner:
                        log.debug(
                            f"  Could not fetch {net}.{sta}.{loc}.{cha}: {exc_inner}"
                        )
        except Exception as exc:
            log.warning(f"  Event {eid}: waveform request failed ({exc}); skipping.")
            continue

        if not st:
            continue

        for pick in valid_picks:
            seed = _pick_seed_id(pick)
            if seed is None:
                continue
            net, sta, loc, cha = seed.split(".")
            trs = st.select(network=net, station=sta, location=loc, channel=cha)
            if not trs:
                continue
            tr = trs.merge()[0]

            sig_start = pick.time - prepick
            sig_end   = sig_start + length
            noise_start = sig_start - noise_dur
            noise_end   = sig_start

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                noise_sl = tr.slice(
                    starttime=UTCDateTime(noise_start),
                    endtime=UTCDateTime(noise_end)
                )
                sig_sl = tr.slice(
                    starttime=UTCDateTime(sig_start),
                    endtime=UTCDateTime(sig_end)
                )

            if noise_sl is None or len(noise_sl.data) < 5:
                log.debug(f"  Insufficient noise data: {seed} event {eid}")
                continue
            if sig_sl is None or len(sig_sl.data) < 5:
                log.debug(f"  Insufficient signal data: {seed} event {eid}")
                continue

            noise_amp = _rms(noise_sl.data)
            if not np.isfinite(noise_amp) or noise_amp <= 0:
                log.debug(f"  Zero/nan noise for {seed} event {eid}; skipping.")
                continue

            sig_amp = float(np.max(np.abs(sig_sl.data)))
            snr     = sig_amp / noise_amp

            records.append({
                "event_id":   eid,
                "seed_id":    seed,
                "phase_hint": pick.phase_hint or "",
                "snr":        snr,
                "pick_time":  pick.time.datetime,
            })

    df = pd.DataFrame(records)
    log.info(
        f"SNR computed for {len(df):,} picks across "
        f"{df['event_id'].nunique() if not df.empty else 0} events."
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 – Epicentral distances
# ---------------------------------------------------------------------------

def _compute_distances(
    catalog,
    inventory=None,
    client=None,
) -> pd.DataFrame:
    """
    Compute median epicentral distance (km) from catalog events to each station.

    Station coordinates are looked up in ``inventory`` (StationXML); if absent
    for a given station, an FDSN ``client`` is queried as fallback.  Channels
    whose coordinates cannot be resolved are excluded with a warning.

    Returns
    -------
    pd.DataFrame  columns: seed_id, sta_lat, sta_lon,
                           median_dist_km, mean_dist_km, min_dist_km
    """
    if inventory is None and client is None:
        log.warning(
            "No inventory or client provided; distance analysis skipped."
        )
        return pd.DataFrame()

    sta_coords = _resolve_station_coords(catalog, inventory=inventory, client=client)
    if not sta_coords:
        return pd.DataFrame()

    # ── Compute per-event distances ───────────────────────────────────────────
    dist_accum: dict[str, list[float]] = defaultdict(list)

    for ev in catalog:
        try:
            org = _preferred_origin(ev)
        except ValueError:
            continue
        if org.latitude is None or org.longitude is None:
            continue
        ev_lat, ev_lon = org.latitude, org.longitude

        for p in ev.picks:
            seed = _pick_seed_id(p)
            if seed is None:
                continue
            if seed not in sta_coords:
                continue
            sta_lat, sta_lon = sta_coords[seed]
            dist_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)
            dist_accum[seed].append(dist_m / 1000.0)

    rows = []
    for seed, dists in dist_accum.items():
        sta_lat, sta_lon = sta_coords[seed]
        rows.append({
            "seed_id":        seed,
            "sta_lat":        sta_lat,
            "sta_lon":        sta_lon,
            "median_dist_km": float(np.median(dists)),
            "mean_dist_km":   float(np.mean(dists)),
            "min_dist_km":    float(np.min(dists)),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "seed_id", "sta_lat", "sta_lon",
                "median_dist_km", "mean_dist_km", "min_dist_km",
            ]
        )

    df = pd.DataFrame(rows).sort_values("median_dist_km").reset_index(drop=True)
    log.info(f"Distances resolved for {len(df):,} SEED IDs.")
    return df


def _compute_pick_distances(
    catalog,
    inventory=None,
    client=None,
) -> pd.DataFrame:
    """
    Compute event-specific epicentral distance for each picked seed.

    Returns
    -------
    pd.DataFrame  columns: event_id, seed_id, dist_km
    """
    if inventory is None and client is None:
        return pd.DataFrame(columns=["event_id", "seed_id", "dist_km"])

    sta_coords = _resolve_station_coords(catalog, inventory=inventory, client=client)
    if not sta_coords:
        return pd.DataFrame(columns=["event_id", "seed_id", "dist_km"])

    records = []
    for ev in catalog:
        try:
            org = _preferred_origin(ev)
        except ValueError:
            continue
        if org.latitude is None or org.longitude is None:
            continue
        eid = _event_id(ev)
        ev_lat, ev_lon = org.latitude, org.longitude

        seen: set[str] = set()
        for p in ev.picks:
            seed = _pick_seed_id(p)
            if seed is None or seed in seen:
                continue
            if seed not in sta_coords:
                continue
            seen.add(seed)
            sta_lat, sta_lon = sta_coords[seed]
            dist_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)
            records.append({
                "event_id": eid,
                "seed_id": seed,
                "dist_km": dist_m / 1000.0,
            })

    if not records:
        return pd.DataFrame(columns=["event_id", "seed_id", "dist_km"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 4 – Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_pick_counts(pick_stats: pd.DataFrame, out_dir: str) -> None:
    """Horizontal bar – total pick count per SEED ID."""
    df = pick_stats.sort_values("n_picks", ascending=True)
    n = len(df)
    fig_h = max(5, n * 0.22 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # Colour by dominant phase type
    palette = {"P": "#1f77b4", "S": "#d62728", "?": "#7f7f7f"}
    colors = []
    for ph in df["phase_types"]:
        if "P" in ph and "S" not in ph:
            colors.append(palette["P"])
        elif "S" in ph and "P" not in ph:
            colors.append(palette["S"])
        elif "P" in ph and "S" in ph:
            colors.append("#ff7f0e")   # both
        else:
            colors.append(palette["?"])

    y = np.arange(n)
    ax.barh(y, df["n_picks"].values, color=colors, edgecolor="none", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["seed_id"].values, fontsize=7)
    ax.set_xlabel("Total picks in catalog", fontsize=11)
    ax.set_title(
        f"Picks per SEED ID  (N events = {pick_stats.attrs.get('n_events_total','?')})",
        fontsize=12,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palette["P"],  label="P only"),
        Patch(facecolor=palette["S"],  label="S only"),
        Patch(facecolor="#ff7f0e",     label="P + S"),
        Patch(facecolor=palette["?"],  label="unknown phase"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "01_pick_counts.png"))


def _plot_event_coverage(pick_stats: pd.DataFrame, out_dir: str) -> None:
    """Horizontal bar – fraction of events with ≥1 pick per SEED ID."""
    df = pick_stats.sort_values("event_coverage_frac", ascending=True)
    n = len(df)
    fig_h = max(5, n * 0.22 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    colors = plt.cm.RdYlGn(df["event_coverage_frac"].values)
    y = np.arange(n)
    ax.barh(y, df["event_coverage_frac"].values * 100, color=colors, edgecolor="none", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["seed_id"].values, fontsize=7)
    ax.set_xlabel("Events with ≥ 1 pick  (%)", fontsize=11)
    ax.set_title("Event coverage per SEED ID", fontsize=12)
    ax.axvline(30, color="orange", linestyle="--", linewidth=1, label="30 %")
    ax.axvline(50, color="red",    linestyle="--", linewidth=1, label="50 %")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "02_event_coverage.png"))


def _plot_snr_distributions(snr_df: pd.DataFrame, out_dir: str) -> None:
    """Violin plot of SNR distributions per SEED ID, sorted by median SNR."""
    if snr_df.empty:
        log.warning("No SNR data; skipping SNR distribution plot.")
        return

    medians = snr_df.groupby("seed_id")["snr"].median().sort_values(ascending=False)
    ordered_seeds = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(max(10, len(ordered_seeds) * 0.5 + 2), 6))

    data_groups = [
        np.log10(np.maximum(snr_df.loc[snr_df["seed_id"] == s, "snr"].values, 1e-3))
        for s in ordered_seeds
    ]

    parts = ax.violinplot(
        data_groups,
        positions=np.arange(len(ordered_seeds)),
        showmedians=True,
        showextrema=False,
        widths=0.7,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f77b4")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("red")
    parts["cmedians"].set_linewidth(1.5)

    ax.axhline(np.log10(3), color="orange", linestyle="--", linewidth=1, label="SNR = 3")
    ax.axhline(np.log10(5), color="red",    linestyle="--", linewidth=1, label="SNR = 5")
    ax.set_xticks(np.arange(len(ordered_seeds)))
    ax.set_xticklabels(ordered_seeds, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("log₁₀(SNR)", fontsize=11)
    ax.set_title("SNR distribution per SEED ID  (sorted by median)", fontsize=12)
    ax.legend(fontsize=9)

    # Annotate median values
    for xi, seed in enumerate(ordered_seeds):
        med = medians[seed]
        ax.text(
            xi, np.log10(max(med, 1e-3)) + 0.05,
            f"{med:.1f}",
            ha="center", va="bottom", fontsize=5.5, color="black",
        )

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "03_snr_distributions.png"))


def _plot_snr_vs_distance(
    snr_df: pd.DataFrame, dist_df: pd.DataFrame, out_dir: str
) -> None:
    """Scatter: median SNR vs median epicentral distance per SEED ID."""
    if snr_df.empty or dist_df.empty:
        log.warning("Insufficient data for SNR vs distance plot; skipping.")
        return

    med_snr = snr_df.groupby("seed_id")["snr"].median().rename("median_snr").reset_index()
    merged  = med_snr.merge(dist_df[["seed_id", "median_dist_km"]], on="seed_id", how="inner")
    if merged.empty:
        log.warning("No overlapping seeds between SNR and distance data; skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        merged["median_dist_km"],
        merged["median_snr"],
        c=merged["median_snr"].values,
        cmap="viridis",
        s=60,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.4,
    )
    fig.colorbar(sc, ax=ax, label="Median SNR")

    for _, row in merged.iterrows():
        ax.annotate(
            row["seed_id"].split(".")[1],            # station code only
            (row["median_dist_km"], row["median_snr"]),
            textcoords="offset points", xytext=(4, 2),
            fontsize=6, alpha=0.7,
        )

    ax.axhline(3, color="orange", linestyle="--", linewidth=1, label="SNR = 3")
    ax.axhline(5, color="red",    linestyle="--", linewidth=1, label="SNR = 5")
    ax.set_xlabel("Median epicentral distance (km)", fontsize=11)
    ax.set_ylabel("Median SNR", fontsize=11)
    ax.set_title("Median SNR vs epicentral distance per SEED ID", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "04_snr_vs_distance.png"))


def _plot_coverage_matrix(
    catalog,
    snr_df: pd.DataFrame,
    pick_stats: pd.DataFrame,
    out_dir: str,
    max_events: int = 80,
    max_seeds: int = 40,
) -> None:
    """
    Heatmap of per-event, per-seed SNR (or binary pick/no-pick if SNR absent).

    Columns = SEED IDs (top N by pick count).
    Rows    = events (up to max_events, chronological).
    Cell value = log₁₀(SNR) when available, else 1 for has-pick, 0 for no-pick.
    """
    # Choose top seeds by coverage
    top_seeds = pick_stats.head(max_seeds)["seed_id"].tolist()

    # Collect events in time order
    events_sorted = sorted(
        catalog.events,
        key=lambda e: _preferred_origin(e).time,
    )
    if len(events_sorted) > max_events:
        step = max(1, len(events_sorted) // max_events)
        events_sorted = events_sorted[::step][:max_events]

    ev_ids  = [_event_id(ev) for ev in events_sorted]
    n_ev    = len(ev_ids)
    n_seeds = len(top_seeds)

    # Initialise matrix with NaN (= no pick on that channel for that event)
    mat = np.full((n_ev, n_seeds), np.nan)

    # Fill with SNR values if available
    if not snr_df.empty:
        for ei, eid in enumerate(ev_ids):
            sub = snr_df[snr_df["event_id"] == eid]
            for si, seed in enumerate(top_seeds):
                row = sub[sub["seed_id"] == seed]["snr"]
                if not row.empty:
                    mat[ei, si] = np.log10(max(row.mean(), 1e-3))
    else:
        # Binary has-pick matrix
        seed_set_per_event: list[set[str]] = []
        for ev in events_sorted:
            seeds_ev = {
                seed
                for p in ev.picks
                for seed in [_pick_seed_id(p)]
                if seed is not None
            }
            seed_set_per_event.append(seeds_ev)
        for ei, seeds_ev in enumerate(seed_set_per_event):
            for si, seed in enumerate(top_seeds):
                mat[ei, si] = 1.0 if seed in seeds_ev else np.nan

    # Shorten event labels
    ev_labels_short = [eid[:16] for eid in ev_ids]
    seed_labels_short = [s.rsplit(".", 1)[0] for s in top_seeds]  # drop CHA code

    fig_h = max(6, n_ev * 0.13 + 2)
    fig_w = max(8, n_seeds * 0.25 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = matplotlib.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="#e8e8e8")   # grey for missing picks

    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    if not np.isfinite(vmin):
        vmin, vmax = 0, 1

    im = ax.imshow(
        mat, aspect="auto", cmap=cmap, origin="upper",
        interpolation="nearest", vmin=vmin, vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar_label = "log₁₀(SNR)" if not snr_df.empty else "has pick"
    cbar.set_label(cbar_label, fontsize=9)

    ax.set_xticks(np.arange(n_seeds))
    ax.set_xticklabels(seed_labels_short, rotation=70, ha="right", fontsize=6)
    ax.set_yticks(np.arange(n_ev))
    ax.set_yticklabels(ev_labels_short, fontsize=5)
    ax.set_xlabel("SEED ID", fontsize=10)
    ax.set_ylabel("Event", fontsize=10)
    ax.set_title(
        f"Coverage matrix — top {n_seeds} seeds × {n_ev} events  "
        f"(grey = no pick)",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "05_coverage_matrix.png"))


def _plot_selection_summary(
    summary_df: pd.DataFrame,
    out_dir: str,
    min_event_frac: float | None = None,
    min_median_snr: float | None = None,
    max_dist_km: float | None = None,
) -> None:
    """
    Multi-panel bar chart showing each seed and whether it passes/fails each
    of the three selection criteria.  Seeds are sorted by a composite score.
    """
    df = summary_df.copy().reset_index(drop=True)
    if df.empty:
        return

    # Compute composite score (all criteria normalised 0–1 and equally weighted)
    cols_present = []
    if "event_coverage_frac" in df.columns:
        df["_score_cov"] = df["event_coverage_frac"]
        cols_present.append("_score_cov")
    if "median_snr" in df.columns and df["median_snr"].notna().any():
        snr_max = df["median_snr"].replace([np.inf, -np.inf], np.nan).max()
        if snr_max and snr_max > 0:
            df["_score_snr"] = df["median_snr"].fillna(0).clip(lower=0) / snr_max
            cols_present.append("_score_snr")
    if "median_dist_km" in df.columns and df["median_dist_km"].notna().any():
        dist_max = df["median_dist_km"].replace([np.inf, -np.inf], np.nan).max()
        if dist_max and dist_max > 0:
            df["_score_dist"] = 1.0 - df["median_dist_km"].fillna(dist_max).clip(upper=dist_max) / dist_max
            cols_present.append("_score_dist")

    if cols_present:
        df["_composite"] = df[cols_present].mean(axis=1)
    else:
        df["_composite"] = 0.0

    df = df.sort_values("_composite", ascending=True).reset_index(drop=True)
    n = len(df)

    n_panels = 1 + sum([
        "event_coverage_frac" in df.columns,
        "median_snr" in df.columns and df["median_snr"].notna().any(),
        "median_dist_km" in df.columns and df["median_dist_km"].notna().any(),
    ])
    fig_h = max(6, n * 0.22 + 2)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 6, fig_h), sharey=True)
    if n_panels == 1:
        axes = [axes]

    y         = np.arange(n)
    ax_idx    = 0

    # Panel A – composite score
    ax = axes[ax_idx]; ax_idx += 1
    ax.barh(y, df["_composite"].values, color="#9467bd", edgecolor="none", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["seed_id"].values, fontsize=7)
    ax.set_xlabel("Composite score", fontsize=10)
    ax.set_title("Composite", fontsize=10)

    # Panel B – event coverage
    if "event_coverage_frac" in df.columns:
        ax = axes[ax_idx]; ax_idx += 1
        pass_mask = (
            df["event_coverage_frac"] >= (min_event_frac or 0)
        ).values
        colors = np.where(pass_mask, "#2ca02c", "#d62728")
        ax.barh(y, df["event_coverage_frac"].values * 100, color=colors,
                edgecolor="none", alpha=0.85)
        if min_event_frac is not None:
            ax.axvline(min_event_frac * 100, color="k", linestyle="--",
                       linewidth=1.2, label=f"min {min_event_frac*100:.0f}%")
            ax.legend(fontsize=8)
        ax.set_xlabel("Event coverage (%)", fontsize=10)
        ax.set_title("Coverage", fontsize=10)

    # Panel C – median SNR
    if "median_snr" in df.columns and df["median_snr"].notna().any():
        ax = axes[ax_idx]; ax_idx += 1
        snr_vals = df["median_snr"].fillna(0).values
        pass_mask = (snr_vals >= (min_median_snr or 0))
        colors = np.where(pass_mask, "#2ca02c", "#d62728")
        ax.barh(y, snr_vals, color=colors, edgecolor="none", alpha=0.85)
        if min_median_snr is not None:
            ax.axvline(min_median_snr, color="k", linestyle="--",
                       linewidth=1.2, label=f"min {min_median_snr}")
            ax.legend(fontsize=8)
        ax.set_xlabel("Median SNR", fontsize=10)
        ax.set_title("Median SNR", fontsize=10)
        ax.set_xscale("symlog", linthresh=1)

    # Panel D – median distance
    if "median_dist_km" in df.columns and df["median_dist_km"].notna().any():
        ax = axes[ax_idx]; ax_idx += 1
        dist_vals = df["median_dist_km"].fillna(9999).values
        pass_mask = (dist_vals <= (max_dist_km or np.inf))
        colors = np.where(pass_mask, "#2ca02c", "#d62728")
        ax.barh(y, dist_vals, color=colors, edgecolor="none", alpha=0.85)
        if max_dist_km is not None:
            ax.axvline(max_dist_km, color="k", linestyle="--",
                       linewidth=1.2, label=f"max {max_dist_km} km")
            ax.legend(fontsize=8)
        ax.set_xlabel("Median distance (km)", fontsize=10)
        ax.set_title("Distance", fontsize=10)

    fig.suptitle(
        "Channel selection summary  (green = passes criterion, red = fails)",
        fontsize=12,
    )
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "06_selection_summary.png"))


def _plot_station_map(
    dist_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    catalog,
    out_dir: str,
) -> None:
    """
    Station map with bubble size ∝ pick count and colour = median SNR.
    Requires 'sta_lat'/'sta_lon' columns in dist_df.
    Falls back silently if cartopy is not available.
    """
    if dist_df.empty or "sta_lat" not in dist_df.columns:
        return

    # Keep only channels that survived earlier filtering and are present
    # in the final summary table.
    keep_cols = ["seed_id", "n_picks"]
    if "median_snr" in summary_df.columns:
        keep_cols.append("median_snr")
    plot_df = dist_df.merge(summary_df[keep_cols], on="seed_id", how="inner")
    if plot_df.empty:
        return
    if "median_snr" not in plot_df.columns:
        plot_df["median_snr"] = 0.0
    plot_df["median_snr"] = plot_df["median_snr"].fillna(0)
    plot_df = plot_df[plot_df["n_picks"].fillna(0) > 0]
    if plot_df.empty:
        return

    # Aggregate to station level (one physical station may have multiple channels)
    plot_df["station"] = plot_df["seed_id"].apply(lambda s: s.split(".")[1])
    sta_df = (
        plot_df.groupby("station")
        .agg(
            lat=("sta_lat", "first"),
            lon=("sta_lon", "first"),
            n_picks=("n_picks", "sum"),
            median_snr=("median_snr", "median"),
        )
        .reset_index()
    )

    # Event locations
    ev_lats, ev_lons = [], []
    for ev in catalog:
        try:
            org = _preferred_origin(ev)
            if org.latitude is not None:
                ev_lats.append(org.latitude)
                ev_lons.append(org.longitude)
        except ValueError:
            pass

    max_n = sta_df["n_picks"].max() or 1
    sizes = 30 + 300 * (sta_df["n_picks"] / max_n)

    fig, ax, transform_kw = _make_map_axes(figsize=(10, 8))
    point_transform_kw = {**transform_kw, "zorder": 5} if transform_kw else {}

    sc = ax.scatter(
        sta_df["lon"], sta_df["lat"],
        s=sizes, c=np.log10(sta_df["median_snr"].clip(lower=0.1)),
        cmap="plasma", edgecolors="k", linewidths=0.5, alpha=0.9,
        **point_transform_kw,
    )
    fig.colorbar(sc, ax=ax, label="log₁₀(median SNR)", shrink=0.6)

    for _, row in sta_df.iterrows():
        ax.annotate(
            row["station"],
            (row["lon"], row["lat"]),
            textcoords="offset points", xytext=(4, 3),
            fontsize=7,
            **transform_kw,
        )

    if ev_lats:
        ax.scatter(
            ev_lons, ev_lats, marker="*", s=80,
            c="yellow", edgecolors="k", linewidths=0.4, zorder=6,
            label="Events",
            **transform_kw,
        )
        ax.legend(fontsize=9)

    ax.set_title("Station map  (bubble size ∝ pick count, colour = median SNR)", fontsize=11)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "07_station_map.png"))


def _plot_template_station_map(
    event_id: str,
    template_df: pd.DataFrame,
    event_lat: float | None,
    event_lon: float | None,
    ax,
    transform_kw: dict,
) -> None:
    """Draw a per-template map showing all candidate stations and selected ones."""
    if template_df.empty:
        return

    map_df = template_df.dropna(subset=["sta_lat", "sta_lon"]).copy()
    if map_df.empty:
        return

    map_df["station"] = map_df["seed_id"].apply(lambda s: s.split(".")[1])
    sta_df = (
        map_df.groupby("station")
        .agg(
            lat=("sta_lat", "first"),
            lon=("sta_lon", "first"),
            median_snr=("median_snr", "max"),
            dist_km=("dist_km", "min"),
            selected=("pass_all", "max"),
        )
        .reset_index()
    )

    ax.scatter(
        sta_df["lon"],
        sta_df["lat"],
        s=30,
        facecolors="none",
        edgecolors="#7f7f7f",
        linewidths=0.7,
        alpha=0.9,
        **transform_kw,
    )

    selected_df = sta_df[sta_df["selected"]]
    if not selected_df.empty:
        sc = ax.scatter(
            selected_df["lon"],
            selected_df["lat"],
            s=80,
            c=np.log10(selected_df["median_snr"].clip(lower=0.1)),
            cmap="plasma",
            edgecolors="k",
            linewidths=0.5,
            alpha=0.95,
            **transform_kw,
        )
        ax.figure.colorbar(sc, ax=ax, label="log10(selected median SNR)", shrink=0.7)

    for _, row in sta_df.iterrows():
        ax.annotate(
            row["station"],
            (row["lon"], row["lat"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6,
            **transform_kw,
        )

    if event_lat is not None and event_lon is not None:
        ax.scatter(
            [event_lon], [event_lat],
            marker="*", s=120,
            c="yellow", edgecolors="k", linewidths=0.5,
            label="Template event",
            **transform_kw,
        )
        ax.legend(fontsize=8)

    ax.set_title(
        f"Template {event_id}: station map (grey=available, filled=selected)",
        fontsize=10,
    )


def _plot_selected_station_overview(
    selected_df: pd.DataFrame,
    catalog,
    out_dir: str,
) -> None:
    """Plot all template events with stations selected after thresholding."""
    if selected_df.empty:
        return

    plot_df = selected_df.dropna(subset=["sta_lat", "sta_lon"]).copy()
    if plot_df.empty:
        return

    plot_df["station"] = plot_df["seed_id"].apply(lambda s: s.split(".")[1])
    sta_df = (
        plot_df.groupby("station")
        .agg(
            lat=("sta_lat", "first"),
            lon=("sta_lon", "first"),
            n_templates=("event_id", "nunique"),
            median_snr=("median_snr", "median"),
        )
        .reset_index()
    )
    if sta_df.empty:
        return

    fig, ax, transform_kw = _make_map_axes(figsize=(11, 8))

    max_n = sta_df["n_templates"].max() or 1
    sizes = 40 + 260 * (sta_df["n_templates"] / max_n)
    sc = ax.scatter(
        sta_df["lon"],
        sta_df["lat"],
        s=sizes,
        c=np.log10(sta_df["median_snr"].clip(lower=0.1)),
        cmap="plasma",
        edgecolors="k",
        linewidths=0.5,
        alpha=0.9,
        **transform_kw,
    )
    fig.colorbar(sc, ax=ax, label="log10(selected median SNR)", shrink=0.7)

    for _, row in sta_df.iterrows():
        ax.annotate(
            row["station"],
            (row["lon"], row["lat"]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=6,
            **transform_kw,
        )

    ev_lats, ev_lons = [], []
    for ev in catalog:
        try:
            org = _preferred_origin(ev)
        except ValueError:
            continue
        if org.latitude is None or org.longitude is None:
            continue
        ev_lats.append(org.latitude)
        ev_lons.append(org.longitude)

    if ev_lats:
        ax.scatter(
            ev_lons, ev_lats,
            marker="*", s=70,
            c="yellow", edgecolors="k", linewidths=0.4,
            label="Templates",
            **transform_kw,
        )
        ax.legend(fontsize=8)

    ax.set_title(
        "Selected stations across all templates (size = templates using station)",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "selected_station_overview.png"))


def _plot_per_template_quality(
    snr_df: pd.DataFrame,
    pick_dist_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    catalog,
    out_dir: str,
    min_snr: float | None = None,
    top_n: int | None = 25,
) -> pd.DataFrame:
    """Write per-template quality plots into per-template subdirectories."""
    if snr_df.empty:
        log.warning("No SNR data; skipping per-template quality plots.")
        return pd.DataFrame()

    root = os.path.join(out_dir, "per_template")
    os.makedirs(root, exist_ok=True)

    event_ids = list(dict.fromkeys(snr_df["event_id"].tolist()))
    coord_df = dist_df[["seed_id", "sta_lat", "sta_lon"]].drop_duplicates() if not dist_df.empty else pd.DataFrame(columns=["seed_id", "sta_lat", "sta_lon"])
    event_coords = {}
    for ev in catalog:
        eid = _event_id(ev)
        try:
            org = _preferred_origin(ev)
        except ValueError:
            event_coords[eid] = (None, None)
            continue
        event_coords[eid] = (org.latitude, org.longitude)

    selected_records = []

    for eid in event_ids:
        sub_snr = snr_df[snr_df["event_id"] == eid]
        if sub_snr.empty:
            continue

        df = (
            sub_snr.groupby("seed_id")["snr"]
            .median()
            .rename("median_snr")
            .reset_index()
        )

        if not pick_dist_df.empty:
            sub_dist = (
                pick_dist_df[pick_dist_df["event_id"] == eid]
                .groupby("seed_id")["dist_km"]
                .median()
                .reset_index()
            )
            df = df.merge(sub_dist, on="seed_id", how="left")
        else:
            df["dist_km"] = np.nan

        if not coord_df.empty:
            df = df.merge(coord_df, on="seed_id", how="left")
        else:
            df["sta_lat"] = np.nan
            df["sta_lon"] = np.nan

        if min_snr is None:
            df["pass_snr"] = True
        else:
            df["pass_snr"] = df["median_snr"] >= min_snr

        df["pass_all"] = df["pass_snr"]
        df = df.sort_values(
            ["pass_all", "median_snr", "dist_km"],
            ascending=[False, False, True],
        )
        if top_n is not None and len(df) > top_n:
            df = df.head(top_n)

        df_plot = df.sort_values("median_snr", ascending=True)
        y = np.arange(len(df_plot))
        labels = df_plot["seed_id"].tolist()

        fig_h = max(5, len(df_plot) * 0.28 + 2)
        fig = plt.figure(figsize=(19, fig_h))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.35])
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1], sharey=fig.axes[0] if fig.axes else None)]
        map_fig, map_ax, transform_kw = _make_map_axes(figsize=(1, 1))
        plt.close(map_fig)
        if transform_kw:
            axes.append(fig.add_subplot(gs[0, 2], projection=map_ax.projection))
            map_transform_kw = transform_kw
            try:
                import cartopy.feature as cfeature
                axes[2].add_feature(cfeature.LAND, facecolor="lightgrey")
                axes[2].add_feature(cfeature.OCEAN, facecolor="aliceblue")
                axes[2].add_feature(cfeature.COASTLINE, linewidth=0.5)
                axes[2].add_feature(cfeature.BORDERS, linewidth=0.4)
                axes[2].add_feature(cfeature.STATES, linewidth=0.3)
            except ImportError:
                pass
        else:
            axes.append(fig.add_subplot(gs[0, 2]))
            axes[2].set_xlabel("Longitude")
            axes[2].set_ylabel("Latitude")
            axes[2].grid(True, linewidth=0.4, alpha=0.5)
            map_transform_kw = {}

        snr_colors = np.where(df_plot["pass_snr"].values, "#2ca02c", "#d62728")
        axes[0].barh(y, df_plot["median_snr"].values, color=snr_colors, alpha=0.85)
        if min_snr is not None:
            axes[0].axvline(min_snr, color="k", linestyle="--", linewidth=1.0)
        axes[0].set_xlabel("Median SNR (template)")
        axes[0].set_title("Per-template SNR")
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=7)
        axes[0].set_xscale("symlog", linthresh=1)

        dist_vals = df_plot["dist_km"].fillna(np.nan).values
        axes[1].barh(y, dist_vals, color="#4c72b0", alpha=0.85)
        axes[1].set_xlabel("Epicentral distance (km)")
        axes[1].set_title("Per-template distance")
        axes[1].tick_params(labelleft=False)

        ev_lat, ev_lon = event_coords.get(eid, (None, None))
        _plot_template_station_map(
            event_id=eid,
            template_df=df,
            event_lat=ev_lat,
            event_lon=ev_lon,
            ax=axes[2],
            transform_kw=map_transform_kw,
        )

        fig.suptitle(
            f"Template {eid}: channel quality (green=passes criterion)",
            fontsize=11,
        )
        fig.tight_layout()

        event_dir = os.path.join(root, eid)
        os.makedirs(event_dir, exist_ok=True)
        _savefig(fig, os.path.join(event_dir, "quality_summary.png"))
        df.to_csv(os.path.join(event_dir, "channel_quality.csv"), index=False)

        selected = df[df["pass_all"]].copy()
        if not selected.empty:
            selected["event_id"] = eid
            selected_records.append(selected)

    log.info(f"Per-template quality plots written under {root}")
    if not selected_records:
        return pd.DataFrame()
    return pd.concat(selected_records, ignore_index=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_picks(
    catalog,
    client=None,
    parameters: dict | None = None,
    out_dir: str = "tribe_analysis",
    inventory=None,
    noise_dur: float = 20.0,
    max_pick_distance_km: float | None = None,
    per_template_plots: bool = False,
    per_template_top_n: int | None = 25,
    per_template_min_snr: float | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Compute pick statistics, SNRs, and distances across the catalog, then
    write diagnostic plots to ``out_dir``.

    Parameters
    ----------
    catalog : obspy.Catalog
        Full event catalog with picks.
    client : FDSN client, optional
        Required for SNR computation and (if ``inventory`` not given) for
        station-coordinate queries.
    parameters : dict, optional
        Template-generation parameter dict.  Must contain at least 'length'
        and 'prepick' for SNR computation; if absent, SNR is skipped.
    out_dir : str
        Directory where plots are written.
    inventory : obspy.Inventory, optional
        StationXML used for distance computation.  If both this and ``client``
        are None, distances are skipped.
    noise_dur : float
        Duration (s) of the noise window used for SNR calculation.
    max_pick_distance_km : float or None
        Filter out picks from stations > this epicentral distance (km).
        Useful to exclude distant/continental stations.  If specified,
        distances are computed and picks beyond this threshold are removed
        before SNR/plot analyses.
    per_template_plots : bool
        If True, write per-template channel-quality plots under
        ``out_dir/per_template/<event_id>/``.
    per_template_top_n : int or None
        Number of channels shown in each per-template plot.
    per_template_min_snr : float or None
        Optional SNR threshold overlaid in per-template plots.
    force_recompute : bool
        If True, ignore existing checkpoint tables and recompute all analysis
        products.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per SEED ID.  Columns:
        seed_id, n_picks, n_events_with_pick, event_coverage_frac,
        phase_types, median_snr, mean_snr, median_dist_km, mean_dist_km,
        min_dist_km (distance columns present only if coords are available).
    """
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"Analyzing {len(catalog)} events → output in {out_dir}")

    dist_tag = "all" if max_pick_distance_km is None else f"maxdist_{max_pick_distance_km:g}km"
    pick_ckpt = os.path.join(out_dir, f"checkpoint_pick_stats_{dist_tag}.csv")
    snr_ckpt = os.path.join(out_dir, f"checkpoint_snr_{dist_tag}.csv")
    dist_ckpt = os.path.join(out_dir, "checkpoint_distances.csv")
    pick_dist_ckpt = os.path.join(out_dir, f"checkpoint_pick_distances_{dist_tag}.csv")

    def _load_ckpt(path: str, label: str) -> pd.DataFrame | None:
        if force_recompute or not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            log.info(f"Loaded {label} checkpoint: {path}")
            return df
        except Exception as exc:
            log.warning(f"Could not read {label} checkpoint ({exc}); recomputing.")
            return None

    def _save_ckpt(df: pd.DataFrame, path: str, label: str) -> None:
        try:
            df.to_csv(path, index=False)
            log.info(f"Wrote {label} checkpoint: {path}")
        except Exception as exc:
            log.warning(f"Could not write {label} checkpoint ({exc}).")

    dist_df = pd.DataFrame()

    # ── Distance-based pick filtering ───────────────────────────────────────
    if max_pick_distance_km is not None:
        dist_df = _load_ckpt(dist_ckpt, "distance")
        if dist_df is None:
            log.info("Computing distances for pick filtering …")
            dist_df = _compute_distances(catalog, inventory=inventory, client=client)
            if not dist_df.empty:
                _save_ckpt(dist_df, dist_ckpt, "distance")
        if not dist_df.empty:
            allowed_seeds = set(
                dist_df[dist_df["median_dist_km"] <= max_pick_distance_km]["seed_id"]
            )
            catalog_working = catalog.copy()
            n_removed = 0
            for ev in catalog_working:
                before = len(ev.picks)
                ev.picks = [
                    p for p in ev.picks
                    if _pick_seed_id(p) in allowed_seeds
                ]
                n_removed += before - len(ev.picks)
            if n_removed:
                log.info(
                    f"Removed {n_removed:,} picks from stations "
                    f"> {max_pick_distance_km} km."
                )
            catalog = catalog_working
        else:
            log.warning(
                "Could not compute distances; skipping pick distance filtering."
            )

    # ── Pick statistics (no waveforms) ──────────────────────────────────────
    pick_stats = _load_ckpt(pick_ckpt, "pick-stats")
    if pick_stats is None:
        log.info("Computing pick statistics …")
        pick_stats = _catalog_pick_stats(catalog)
        _save_ckpt(pick_stats, pick_ckpt, "pick-stats")
    else:
        pick_stats.attrs["n_events_total"] = len(catalog)
    n_events_total = pick_stats.attrs.get("n_events_total", len(catalog))
    log.info(
        f"  {len(pick_stats):,} unique SEED IDs, "
        f"{n_events_total} events, "
        f"{pick_stats['n_picks'].sum():,} total picks."
    )

    # ── SNR computation ──────────────────────────────────────────────────────
    snr_df = pd.DataFrame()
    pick_dist_df = pd.DataFrame()
    if client is not None and parameters is not None:
        length  = parameters.get("length")
        prepick = parameters.get("prepick")
        if length is None or prepick is None:
            log.warning(
                "parameters dict missing 'length' or 'prepick'; SNR skipped."
            )
        else:
            snr_df = _load_ckpt(snr_ckpt, "snr")
            if snr_df is None:
                log.info("Computing per-pick SNRs …")
                snr_df = _compute_pick_snrs(
                    catalog, client, length, prepick,
                    noise_dur=noise_dur,
                )
                if not snr_df.empty:
                    _save_ckpt(snr_df, snr_ckpt, "snr")

            if per_template_plots:
                pick_dist_df = _load_ckpt(pick_dist_ckpt, "pick-distance")
                if pick_dist_df is None:
                    log.info("Computing event-specific pick distances …")
                    pick_dist_df = _compute_pick_distances(
                        catalog,
                        inventory=inventory,
                        client=client,
                    )
                    if not pick_dist_df.empty:
                        _save_ckpt(pick_dist_df, pick_dist_ckpt, "pick-distance")
    else:
        log.warning(
            "No client or parameters provided; SNR analysis skipped."
        )

    # ── Distances ────────────────────────────────────────────────────────────
    # Skip if already computed for filtering above
    if max_pick_distance_km is not None:
        log.info("Using pre-computed distances from filtering step.")
    else:
        dist_df = _load_ckpt(dist_ckpt, "distance")
        if dist_df is None:
            log.info("Computing epicentral distances …")
            dist_df = _compute_distances(catalog, inventory=inventory, client=client)
            if not dist_df.empty:
                _save_ckpt(dist_df, dist_ckpt, "distance")

    # ── Build merged summary DataFrame ───────────────────────────────────────
    summary = pick_stats.copy()
    summary.attrs = {}   # clear non-serialisable attrs before merge

    if not snr_df.empty:
        snr_agg = (
            snr_df.groupby("seed_id")["snr"]
            .agg(median_snr="median", mean_snr="mean")
            .reset_index()
        )
        summary = summary.merge(snr_agg, on="seed_id", how="left")
    else:
        summary["median_snr"] = np.nan
        summary["mean_snr"]   = np.nan

    if not dist_df.empty:
        summary = summary.merge(
            dist_df[["seed_id", "median_dist_km", "mean_dist_km", "min_dist_km"]],
            on="seed_id", how="left",
        )

    # ── Plots ────────────────────────────────────────────────────────────────
    # New workflow: per-template diagnostics only.
    log.info("Generating per-template diagnostic plots …")
    selected_template_channels = _plot_per_template_quality(
        snr_df=snr_df,
        pick_dist_df=pick_dist_df,
        dist_df=dist_df,
        catalog=catalog,
        out_dir=out_dir,
        min_snr=per_template_min_snr,
        top_n=per_template_top_n,
    )
    _plot_selected_station_overview(
        selected_df=selected_template_channels,
        catalog=catalog,
        out_dir=out_dir,
    )

    # ── Write CSV summary ────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "pick_summary.csv")
    summary.to_csv(csv_path, index=False)
    log.info(f"Summary CSV written to {csv_path}")

    if not selected_template_channels.empty:
        selected_template_channels_path = os.path.join(
            out_dir, "selected_template_channels.csv"
        )
        selected_template_channels.to_csv(selected_template_channels_path, index=False)
        log.info(
            f"Per-template selected channels written to {selected_template_channels_path}"
        )

    log.info("analyze_picks complete.")
    return summary


def select_seeds(
    summary: pd.DataFrame,
    min_event_frac: float | None = None,
    min_median_snr: float | None = None,
    max_dist_km: float | None = None,
    max_seeds: int | None = None,
    sort_by: str = "composite",
) -> list[str]:
    """
    Apply threshold filters to the summary DataFrame and return a ranked list
    of seed IDs to pass into ``build_tribe``.

    Parameters
    ----------
    summary : pd.DataFrame
        Output of ``analyze_picks``.
    min_event_frac : float (0–1), optional
        Drop seeds whose event-coverage fraction < this value.
    min_median_snr : float, optional
        Drop seeds whose median SNR < this value.
    max_dist_km : float, optional
        Drop seeds whose median epicentral distance > this value.
    max_seeds : int, optional
        Hard cap on the number of returned seeds.
    sort_by : str
        One of 'composite' (default), 'coverage', 'snr', 'distance'.
        Controls how surviving seeds are ranked before the max_seeds cap.

    Returns
    -------
    list of str
        SEED IDs that pass all filters, ranked by ``sort_by``.
    """
    df = summary.copy()
    n_initial = len(df)

    if min_event_frac is not None:
        before = len(df)
        df = df[df["event_coverage_frac"] >= min_event_frac]
        log.info(
            f"Event coverage filter (≥{min_event_frac*100:.0f}%): "
            f"{before - len(df):,} seeds removed; {len(df):,} remain."
        )

    if min_median_snr is not None and "median_snr" in df.columns:
        before = len(df)
        df = df[df["median_snr"].fillna(0) >= min_median_snr]
        log.info(
            f"Median SNR filter (≥{min_median_snr}): "
            f"{before - len(df):,} seeds removed; {len(df):,} remain."
        )

    if max_dist_km is not None and "median_dist_km" in df.columns:
        before = len(df)
        df = df[df["median_dist_km"].fillna(9999) <= max_dist_km]
        log.info(
            f"Distance filter (≤{max_dist_km} km): "
            f"{before - len(df):,} seeds removed; {len(df):,} remain."
        )

    # ── Sort ──────────────────────────────────────────────────────────────────
    sort_cols: list[str] = []
    sort_asc:  list[bool] = []

    if sort_by == "composite":
        # Build a mini composite on the surviving subset
        for col, asc in [
            ("event_coverage_frac", False),
            ("median_snr",          False),
            ("median_dist_km",      True),
        ]:
            if col in df.columns:
                sort_cols.append(col)
                sort_asc.append(asc)
        if not sort_cols:
            sort_cols = ["n_picks"]
            sort_asc  = [False]
    elif sort_by == "coverage":
        sort_cols = ["event_coverage_frac", "n_picks"]
        sort_asc  = [False, False]
    elif sort_by == "snr" and "median_snr" in df.columns:
        sort_cols = ["median_snr"]
        sort_asc  = [False]
    elif sort_by == "distance" and "median_dist_km" in df.columns:
        sort_cols = ["median_dist_km"]
        sort_asc  = [True]
    else:
        sort_cols = ["n_picks"]
        sort_asc  = [False]

    df = df.sort_values(sort_cols, ascending=sort_asc)

    if max_seeds is not None and len(df) > max_seeds:
        log.info(
            f"Capping at {max_seeds} seeds (dropped {len(df) - max_seeds:,})."
        )
        df = df.head(max_seeds)

    seeds = df["seed_id"].tolist()
    log.info(
        f"select_seeds: {n_initial} → {len(seeds)} seeds selected "
        f"(filters: event_frac≥{min_event_frac}, SNR≥{min_median_snr}, "
        f"dist≤{max_dist_km} km, max={max_seeds})."
    )
    for s in seeds:
        row = df[df["seed_id"] == s].iloc[0]
        parts = [
            f"cov={row['event_coverage_frac']:.2f}",
        ]
        if "median_snr" in row.index and np.isfinite(row["median_snr"]):
            parts.append(f"SNR={row['median_snr']:.1f}")
        if "median_dist_km" in row.index and np.isfinite(row.get("median_dist_km", np.nan)):
            parts.append(f"dist={row['median_dist_km']:.0f}km")
        log.info(f"  + {s}  ({', '.join(parts)})")

    return seeds


def build_tribe(
    catalog,
    client,
    parameters: dict,
    seeds: list | None = None,
    template_seeds: dict | pd.DataFrame | None = None,
):
    """
    Build an EQcorrscan Tribe from the catalog using either global or
    template-specific SEED selection.

    This is a thin, focused wrapper around ``template_gen``.  It filters picks
    to those in ``seeds`` or ``template_seeds``, drops events left with no
    picks, then calls ``template_gen`` once on the cleaned catalog.

    Parameters
    ----------
    catalog : obspy.Catalog
    client : FDSN client
    parameters : dict
        Forwarded verbatim to ``template_gen`` (except 'process_len' which is
        also passed as ``process_len``).  Required keys:
        lowcut, highcut, samp_rate, filt_order, length, prepick, process_len.
    seeds : list of str, optional
        SEED IDs to keep (e.g. output of ``select_seeds``).  If None, all
        picks in the catalog are used.
    template_seeds : dict or pd.DataFrame, optional
        Template-first channel selection. If a dict, it must map
        ``event_id -> iterable[seed_id]``. If a DataFrame, it must contain
        ``event_id`` and ``seed_id`` columns; if ``pass_all`` is present, only
        rows with ``pass_all == True`` are kept. This takes precedence over
        the global ``seeds`` list.

    Returns
    -------
    eqcorrscan.core.match_filter.Tribe
    """
    try:
        from eqcorrscan.core.template_gen import template_gen
        from eqcorrscan.core.match_filter import Template, Tribe
    except ImportError as exc:
        raise ImportError("eqcorrscan is required for build_tribe.") from exc

    tribe   = Tribe()
    working = catalog.copy()
    working.events.sort(key=lambda e: _preferred_origin(e).time)

    template_seed_map: dict[str, set[str]] | None = None
    if template_seeds is not None:
        if isinstance(template_seeds, pd.DataFrame):
            if "event_id" not in template_seeds.columns or "seed_id" not in template_seeds.columns:
                raise ValueError(
                    "template_seeds DataFrame must contain 'event_id' and 'seed_id' columns."
                )
            template_df = template_seeds.copy()
            if "pass_all" in template_df.columns:
                template_df = template_df[template_df["pass_all"].fillna(False)]
            template_seed_map = {
                str(event_id): set(group["seed_id"].tolist())
                for event_id, group in template_df.groupby("event_id")
            }
        else:
            template_seed_map = {
                str(event_id): set(seed_ids)
                for event_id, seed_ids in template_seeds.items()
            }

    # Remove picks with unusable channel codes (e.g. '--') before any further
    # filtering or waveform requests.
    n_invalid_removed = 0
    for ev in working:
        before = len(ev.picks)
        ev.picks = [p for p in ev.picks if _pick_seed_id(p) is not None]
        n_invalid_removed += before - len(ev.picks)
    if n_invalid_removed:
        log.warning(
            f"build_tribe: removed {n_invalid_removed} picks with invalid channel codes."
        )

    if template_seed_map is not None:
        for ev in working:
            eid = _event_id(ev)
            allowed = template_seed_map.get(eid, set())
            ev.picks = [
                p for p in ev.picks
                if _pick_seed_id(p) in allowed
            ]
        n_before = len(working)
        working.events = [ev for ev in working if ev.picks]
        n_dropped = n_before - len(working)
        if n_dropped:
            log.warning(
                f"build_tribe: {n_dropped} events dropped (no picks after "
                f"template-specific filtering)."
            )
    elif seeds is not None:
        seeds_set = set(seeds)
        for ev in working:
            ev.picks = [
                p for p in ev.picks
                if _pick_seed_id(p) in seeds_set
            ]
        n_before = len(working)
        working.events = [ev for ev in working if ev.picks]
        n_dropped = n_before - len(working)
        if n_dropped:
            log.warning(
                f"build_tribe: {n_dropped} events dropped (no picks after "
                f"filtering to {len(seeds)} seeds)."
            )

    if not working:
        log.warning("build_tribe: catalog is empty after pick filtering; returning empty Tribe.")
        return tribe

    log.info(
        f"build_tribe: calling template_gen on {len(working)} events, "
        f"{'template-specific' if template_seed_map is not None else (len(seeds) if seeds else 'all')} seeds …"
    )

    temp_list = template_gen(
        method="from_client",
        client_id=client,
        catalog=working,
        data_pad=parameters.get("data_pad", 20),
        lowcut=parameters["lowcut"],
        highcut=parameters["highcut"],
        samp_rate=parameters["samp_rate"],
        filt_order=parameters["filt_order"],
        length=parameters["length"],
        prepick=parameters["prepick"],
        swin=parameters.get("swin", "all"),
        process_len=parameters["process_len"],
        min_snr=parameters.get("min_snr"),
    )

    for i, st in enumerate(temp_list):
        ev  = working[i]
        eid = _event_id(ev)
        tribe.templates.append(
            Template(
                name=eid,
                event=ev,
                st=st,
                lowcut=parameters["lowcut"],
                highcut=parameters["highcut"],
                samp_rate=parameters["samp_rate"],
                filt_order=parameters["filt_order"],
                prepick=parameters["prepick"],
                process_length=parameters["process_len"],
            )
        )

    log.info(f"build_tribe: {len(tribe)} templates created.")
    return tribe


def compare_tribe_to_analysis(
    tribe,
    summary: pd.DataFrame,
    template_seeds: "pd.DataFrame | None" = None,
    min_event_frac: float | None = None,
    min_median_snr: float | None = None,
    max_dist_km: float | None = None,
) -> pd.DataFrame:
    """
    Compare channels in an existing Tribe against the per-template analysis.

    Parameters
    ----------
    tribe : eqcorrscan Tribe
        The existing Tribe to compare.
    summary : pd.DataFrame
        Output of ``analyze_picks`` (``pick_summary.csv``).  Used for the
        metric columns (SNR, coverage, distance) that appear in the plots.
    template_seeds : pd.DataFrame, optional
        **Recommended.**  Output of ``analyze_picks``
        (``selected_template_channels.csv``), which is a *per-template*
        selection based on per-event SNR and distance thresholds — no global
        event-coverage filter is applied.  If a ``pass_all`` column is
        present, only rows with ``pass_all == True`` are counted.  A seed is
        "suggested" if it appears in any template's selected set.

        When this is supplied, ``min_event_frac``, ``min_median_snr``, and
        ``max_dist_km`` are **ignored** — the per-template CSV already encodes
        those decisions at the event level.

        If ``None``, falls back to applying the global threshold arguments
        against ``summary``.
    min_event_frac : float (0–1), optional
        Fallback: minimum global event coverage fraction.  Ignored when
        ``template_seeds`` is provided.
    min_median_snr : float, optional
        Fallback: minimum global median SNR.  Ignored when ``template_seeds``
        is provided.
    max_dist_km : float, optional
        Fallback: maximum global median distance.  Ignored when
        ``template_seeds`` is provided.

    Returns
    -------
    pd.DataFrame  with a ``comparison`` column:
      kept_and_suggested          – in Tribe and selected by analysis
      kept_but_not_suggested      – in Tribe but analysis would drop it
      suggested_but_missing_in_tribe – analysis selects it, not in Tribe
      excluded_by_both            – neither the Tribe nor analysis wants it
    """
    if summary.empty:
        return pd.DataFrame()

    tribe_seeds: set[str] = set()
    for tmpl in tribe:
        st = getattr(tmpl, "st", None)
        if st is None:
            continue
        for tr in st:
            stats = tr.stats
            net = str(getattr(stats, "network", "") or "")
            sta = str(getattr(stats, "station", "") or "")
            loc = str(getattr(stats, "location", "") or "")
            cha = str(getattr(stats, "channel", "") or "")
            if not sta or not cha or not _VALID_CHA.match(cha):
                continue
            tribe_seeds.add(f"{net}.{sta}.{loc}.{cha}")

    df = summary.copy()
    df["in_existing_tribe"] = df["seed_id"].isin(tribe_seeds)

    if template_seeds is not None:
        # Per-template path: a seed is "suggested" if it appears in any
        # template's selected set.  Event-coverage is irrelevant here —
        # a channel only needs to earn its place in its own template.
        ts_df = template_seeds.copy()
        if "pass_all" in ts_df.columns:
            ts_df = ts_df[ts_df["pass_all"].fillna(False)]
        suggested_seeds = set(ts_df["seed_id"].dropna().tolist())
        df["suggested_by_analysis"] = df["seed_id"].isin(suggested_seeds)
        log.info(
            f"compare_tribe_to_analysis: using per-template selection "
            f"({len(suggested_seeds)} unique suggested seeds)."
        )
    else:
        # Fallback: global threshold path
        if min_event_frac is None:
            df["pass_coverage"] = True
        else:
            df["pass_coverage"] = df["event_coverage_frac"] >= min_event_frac

        if min_median_snr is None or "median_snr" not in df.columns:
            df["pass_snr"] = True
        else:
            df["pass_snr"] = df["median_snr"].fillna(0) >= min_median_snr

        if max_dist_km is None or "median_dist_km" not in df.columns:
            df["pass_dist"] = True
        else:
            df["pass_dist"] = df["median_dist_km"].fillna(np.inf) <= max_dist_km

        df["suggested_by_analysis"] = (
            df["pass_coverage"] & df["pass_snr"] & df["pass_dist"]
        )

    df["comparison"] = np.select(
        [
            df["in_existing_tribe"] & df["suggested_by_analysis"],
            df["in_existing_tribe"] & ~df["suggested_by_analysis"],
            ~df["in_existing_tribe"] & df["suggested_by_analysis"],
        ],
        [
            "kept_and_suggested",
            "kept_but_not_suggested",
            "suggested_but_missing_in_tribe",
        ],
        default="excluded_by_both",
    )

    sort_col = "event_coverage_frac" if "event_coverage_frac" in df.columns else "n_picks"
    df = df.sort_values(
        ["in_existing_tribe", "suggested_by_analysis", sort_col],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return df


def plot_tribe_comparison(
    comparison_df: pd.DataFrame,
    catalog=None,
    dist_df: pd.DataFrame | None = None,
    out_dir: str = "tribe_analysis",
) -> None:
    """
    Visualise the output of ``compare_tribe_to_analysis``.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of ``compare_tribe_to_analysis``.
    catalog : obspy.Catalog, optional
        Used to draw template event locations on the map.
    dist_df : pd.DataFrame, optional
        Output of ``_compute_distances`` (from ``analyze_picks`` internals or
        ``checkpoint_distances.csv``).  Must contain ``sta_lat`` / ``sta_lon``
        columns for the station map to be drawn.
    out_dir : str
        Directory where plots are written.

    Writes
    ------
    tribe_comparison_summary.png
        4-panel figure: category counts, SNR boxplot, coverage boxplot,
        distance boxplot — one box per comparison category.
    tribe_comparison_map.png
        Station map with each station coloured by comparison category.
        Written only when ``dist_df`` provides coordinate columns.
    """
    os.makedirs(out_dir, exist_ok=True)

    if comparison_df.empty:
        log.warning("comparison_df is empty; nothing to plot.")
        return

    cat_colors = {
        "kept_and_suggested":             "#2ca02c",
        "kept_but_not_suggested":         "#ff7f0e",
        "suggested_but_missing_in_tribe": "#1f77b4",
        "excluded_by_both":               "#c7c7c7",
    }
    cat_labels = {
        "kept_and_suggested":             "Kept & suggested",
        "kept_but_not_suggested":         "Kept, not suggested",
        "suggested_but_missing_in_tribe": "Suggested, missing",
        "excluded_by_both":               "Excluded by both",
    }
    categories = list(cat_colors.keys())

    df = comparison_df.copy()
    df["comparison"] = df["comparison"].fillna("excluded_by_both")

    # ── Figure 1: metric distributions per category ───────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # Panel 0: count per category
    counts = df["comparison"].value_counts().reindex(categories, fill_value=0)
    ax = axes[0]
    y_pos = np.arange(len(categories))
    bars = ax.barh(
        y_pos,
        counts.values,
        color=[cat_colors[c] for c in categories],
        edgecolor="none",
        alpha=0.85,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([cat_labels[c] for c in categories], fontsize=9)
    ax.set_xlabel("Number of SEED IDs")
    ax.set_title("Category counts")
    for bar, v in zip(bars, counts.values):
        if v > 0:
            ax.text(
                v + 0.2, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", ha="left", fontsize=9,
            )

    def _boxplot_panel(ax, col, ylabel, title, log_scale=False):
        groups, labels, colors = [], [], []
        for c in categories:
            if col not in df.columns:
                break
            vals = df.loc[df["comparison"] == c, col].dropna().values
            if len(vals) == 0:
                continue
            groups.append(vals)
            labels.append(cat_labels[c])
            colors.append(cat_colors[c])
        if not groups:
            ax.text(0.5, 0.5, f"No {col} data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        bp = ax.boxplot(
            groups,
            labels=labels,
            patch_artist=True,
            medianprops={"color": "k", "linewidth": 1.5},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        if log_scale:
            ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    _boxplot_panel(axes[1], "median_snr",          "Median SNR",            "SNR per category",      log_scale=True)
    _boxplot_panel(axes[2], "event_coverage_frac",  "Event coverage (frac)", "Coverage per category")
    _boxplot_panel(axes[3], "median_dist_km",        "Median distance (km)",  "Distance per category")

    fig.suptitle("Tribe vs. analysis comparison — metric distributions", fontsize=13)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "tribe_comparison_summary.png"))

    # ── Figure 2: station map coloured by category ────────────────────────
    if dist_df is None or dist_df.empty or "sta_lat" not in dist_df.columns:
        log.info("No dist_df with coordinates; tribe comparison map skipped.")
        return

    plot_df = dist_df.merge(
        df[["seed_id", "comparison", "n_picks"]],
        on="seed_id",
        how="inner",
    )
    if plot_df.empty:
        log.info("No overlapping seeds between comparison_df and dist_df; map skipped.")
        return

    fig, ax, transform_kw = _make_map_axes(figsize=(11, 9))
    max_picks = plot_df["n_picks"].max() or 1

    # Offset overlapping channels at the same station slightly so they don't
    # sit exactly on top of each other.  Within each station group, channels
    # are spread in a small circle (radius ~0.05°).
    import math
    station_channel_count: dict[str, int] = {}
    for seed in plot_df["seed_id"]:
        sta = seed.split(".")[1]
        station_channel_count[sta] = station_channel_count.get(sta, 0) + 1

    station_index: dict[str, int] = {}   # tracks which offset slot we're on

    for cat in categories:
        sub = plot_df[plot_df["comparison"] == cat].copy()
        if sub.empty:
            continue
        lons, lats, sizes_list = [], [], []
        labels_list = []
        for _, row in sub.iterrows():
            sta = row["seed_id"].split(".")[1]
            n_ch = station_channel_count[sta]
            idx  = station_index.get(sta, 0)
            station_index[sta] = idx + 1

            if n_ch > 1:
                angle = 2 * math.pi * idx / n_ch
                d_lon = 0.05 * math.cos(angle)
                d_lat = 0.05 * math.sin(angle)
            else:
                d_lon = d_lat = 0.0

            lons.append(row["sta_lon"] + d_lon)
            lats.append(row["sta_lat"] + d_lat)
            sizes_list.append(40 + 240 * (row["n_picks"] / max_picks))
            # Label = station.channel (last two parts of SEED id)
            labels_list.append(".".join(row["seed_id"].split(".")[-2:]))  # loc.cha

        ax.scatter(
            lons, lats,
            s=sizes_list,
            color=cat_colors[cat],
            edgecolors="k",
            linewidths=0.4,
            alpha=0.85,
            label=cat_labels[cat],
            **transform_kw,
        )
        for lon, lat, label in zip(lons, lats, labels_list):
            ax.annotate(
                label,
                (lon, lat),
                textcoords="offset points", xytext=(4, 2),
                fontsize=5,
                **transform_kw,
            )

    if catalog is not None:
        ev_lats, ev_lons = [], []
        for ev in catalog:
            try:
                org = _preferred_origin(ev)
                if org.latitude is not None:
                    ev_lats.append(org.latitude)
                    ev_lons.append(org.longitude)
            except ValueError:
                pass
        if ev_lats:
            ax.scatter(
                ev_lons, ev_lats,
                marker="*", s=80,
                c="yellow", edgecolors="k", linewidths=0.4,
                zorder=6, label="Template events",
                **transform_kw,
            )

    ax.legend(fontsize=8, loc="best")
    ax.set_title(
        "Station map: Tribe vs. analysis  (size ∝ pick count)",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "tribe_comparison_map.png"))


def _plot_scan_timeline(
    chunk_df: pd.DataFrame,
    windows: pd.DataFrame,
    event_times: list,
    active_seeds: set,
    min_channels: int,
    out_dir: str,
) -> None:
    """Write scan_timeline.png showing event density and channel availability."""
    if chunk_df.empty and not event_times:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 0: event density histogram
    ax = axes[0]
    if event_times:
        t_datetimes = [t.datetime for t in event_times]
        ax.hist(t_datetimes, bins=min(100, max(10, len(event_times))),
                color="#1f77b4", edgecolor="none", alpha=0.8)
    ax.set_ylabel("Template events per bin", fontsize=10)
    ax.set_title("Template event distribution", fontsize=11)
    if not windows.empty:
        for _, row in windows.iterrows():
            ax.axvspan(pd.Timestamp(row["start"]), pd.Timestamp(row["end"]),
                       color="green", alpha=0.12, zorder=0)

    # Panel 1: active channel count over time
    ax = axes[1]
    if not chunk_df.empty and active_seeds:
        t_mids = [
            pd.Timestamp(r["start"]) + (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])) / 2
            for _, r in chunk_df.iterrows()
        ]
        n_vals = chunk_df["n_channels_active"].values
        ax.step(t_mids, n_vals, where="mid", color="#2ca02c", linewidth=1.2)
        ax.fill_between(t_mids, n_vals, step="mid", color="#2ca02c", alpha=0.3)
        if min_channels > 0:
            ax.axhline(min_channels, color="k", linestyle="--", linewidth=1.0,
                       label=f"min {min_channels} channels")
            ax.legend(fontsize=9)
        ax.set_ylabel("Active selected channels", fontsize=10)
        ax.set_title("Selected channel availability over time", fontsize=11)
    else:
        ax.text(0.5, 0.5, "No seeds specified; availability not tracked.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Channel availability", fontsize=11)

    if not windows.empty:
        first = True
        for _, row in windows.iterrows():
            axes[1].axvspan(
                pd.Timestamp(row["start"]), pd.Timestamp(row["end"]),
                color="green", alpha=0.12, zorder=0,
                label="Scan window" if first else None,
            )
            first = False

    axes[1].set_xlabel("Date", fontsize=10)
    fig.autofmt_xdate()
    fig.suptitle(
        "Suggested matched-filter scan windows  (green shading)", fontsize=12
    )
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "scan_timeline.png"))


def suggest_scan_windows(
    catalog,
    client=None,
    inventory=None,
    seeds: list | None = None,
    template_seeds: "pd.DataFrame | dict | None" = None,
    out_dir: str | None = None,
    inventory_path: str | None = None,
    pre_buffer_days: float = 0.0,
    post_buffer_days: float = 0.0,
    min_channels: int = 1,
    chunk_days: float = 1.0,
) -> pd.DataFrame:
    """
    Determine time windows suitable for running the matched filter.

    Uses the temporal span of the template catalog plus optional leading/
    trailing buffers.  If channel-epoch information is available (via
    ``inventory`` or ``client``), the function counts how many selected
    channels are simultaneously active in each daily chunk and returns only
    the windows where that count meets ``min_channels``.  Adjacent passing
    chunks are merged into contiguous scan windows.

    Parameters
    ----------
    catalog : obspy.Catalog
        Template catalog.  Event origin times define the scan extent.
    client : FDSN client, optional
        Queried for channel availability when ``inventory`` is absent.
    inventory : obspy.Inventory, optional
        StationXML with channel-level epoch data (start_date / end_date).
    seeds : list of str, optional
        Global seed list (e.g. output of ``select_seeds``).
    template_seeds : pd.DataFrame or dict, optional
        Per-template selected channels.  If a DataFrame, must contain
        ``seed_id`` (and optionally ``pass_all``); the union of selected
        seeds is used.  If a dict, must map event_id → iterable[seed_id].
        Takes precedence over ``seeds`` when both are provided.
    out_dir : str, optional
        If supplied, writes ``scan_windows.csv`` and ``scan_timeline.png``
        into this directory.
    inventory_path : str, optional
        If supplied, the StationXML queried from ``client`` is written to
        this path (e.g. ``"/path/to/inventory.xml"``).  Any inventory
        supplied via the ``inventory`` argument is not re-written.
        Ignored when no client queries are made.
    pre_buffer_days : float
        Days to extend the window before the earliest template event.
    post_buffer_days : float
        Days to extend the window after the latest template event.
    min_channels : int
        Minimum number of simultaneously active selected channels required
        for a chunk to be included.  Ignored (set to 0 internally) when no
        seed list is provided.
    chunk_days : float
        Size of each evaluation chunk in days (default 1.0 = daily).

    Returns
    -------
    pd.DataFrame
        One row per contiguous scan window.  Columns:
        start, end, n_days, min_channels_active, mean_channels_active,
        n_events.
    """
    # ── Resolve the active seed set ──────────────────────────────────────
    active_seeds: set[str] = set()
    if template_seeds is not None:
        if isinstance(template_seeds, pd.DataFrame):
            ts_df = template_seeds.copy()
            if "pass_all" in ts_df.columns:
                ts_df = ts_df[ts_df["pass_all"].fillna(False)]
            active_seeds = set(ts_df["seed_id"].dropna().tolist())
        else:
            for v in template_seeds.values():
                active_seeds.update(v)
    elif seeds is not None:
        active_seeds = set(seeds)

    # ── Catalog time range ───────────────────────────────────────────────
    event_times = []
    for ev in catalog:
        try:
            org = _preferred_origin(ev)
            event_times.append(org.time)
        except ValueError:
            pass

    if not event_times:
        log.warning("No events with origins found; cannot suggest scan windows.")
        return pd.DataFrame(
            columns=["start", "end", "n_days",
                     "min_channels_active", "mean_channels_active", "n_events"]
        )

    t_cat_start = min(event_times)
    t_cat_end   = max(event_times)
    t_start     = t_cat_start - pre_buffer_days * 86400.0
    t_end       = t_cat_end   + post_buffer_days * 86400.0

    log.info(
        f"Catalog spans {t_cat_start.date} – {t_cat_end.date}  "
        f"({len(event_times)} events).  "
        f"Scan range: {t_start.date} – {t_end.date}."
    )

    # ── Channel epoch resolution ─────────────────────────────────────────
    # For each seed, determine the total operational span as
    # (earliest start_date, latest end_date) across all matching channels.
    # Seeds absent from inventory are assumed to span the full scan range.
    channel_epochs: dict[str, tuple] = {}   # seed -> (UTCDateTime, UTCDateTime)

    EPOCH_OPEN = UTCDateTime(9999, 1, 1)

    # Accumulate inventory fragments returned by client queries so we can
    # optionally write them out after the loop.
    try:
        from obspy import Inventory as _Inventory
        _queried_inv = _Inventory(networks=[], source="suggest_scan_windows")
    except ImportError:
        _queried_inv = None

    if active_seeds:
        for seed in sorted(active_seeds):
            net, sta, loc, cha = seed.split(".")
            found = False
            cs_best = UTCDateTime()    # start = now   (will be min'd down)
            ce_best = UTCDateTime(0)   # end   = epoch (will be max'd up)

            if inventory is not None:
                sel = inventory.select(network=net, station=sta,
                                       location=loc, channel=cha)
                for netobj in sel.networks:
                    for staobj in netobj.stations:
                        for chaobj in staobj.channels:
                            found = True
                            cs = chaobj.start_date or UTCDateTime(0)
                            ce = chaobj.end_date   or EPOCH_OPEN
                            if cs < cs_best:
                                cs_best = cs
                            if ce > ce_best:
                                ce_best = ce

            if not found and client is not None:
                try:
                    inv_q = client.get_stations(
                        network=net, station=sta, location=loc, channel=cha,
                        starttime=t_start, endtime=t_end,
                        level="channel",
                    )
                    if _queried_inv is not None:
                        _queried_inv += inv_q
                    for netobj in inv_q.networks:
                        for staobj in netobj.stations:
                            for chaobj in staobj.channels:
                                found = True
                                cs = chaobj.start_date or UTCDateTime(0)
                                ce = chaobj.end_date   or EPOCH_OPEN
                                if cs < cs_best:
                                    cs_best = cs
                                if ce > ce_best:
                                    ce_best = ce
                except Exception as exc:
                    log.debug(f"  Could not query availability for {seed}: {exc}")

            if not found:
                log.debug(f"  {seed}: no epoch info; assuming always active.")
                channel_epochs[seed] = (UTCDateTime(0), EPOCH_OPEN)
            else:
                channel_epochs[seed] = (cs_best, ce_best)

    # ── Build daily chunks ───────────────────────────────────────────────
    chunk_secs = chunk_days * 86400.0
    chunks = []
    t = t_start
    while t < t_end:
        t_next = min(t + chunk_secs, t_end)
        t_mid  = t + (t_next - t) / 2.0
        if channel_epochs:
            n_active = sum(
                1 for cs, ce in channel_epochs.values()
                if cs <= t_mid <= ce
            )
        else:
            n_active = 0
        n_ev = sum(1 for et in event_times if t <= et < t_next)
        chunks.append({
            "start":             t.datetime,
            "end":               t_next.datetime,
            "n_channels_active": n_active,
            "n_events":          n_ev,
        })
        t = t_next

    chunk_df = pd.DataFrame(chunks)

    # ── Filter and merge into contiguous windows ─────────────────────────
    effective_min = min_channels if active_seeds else 0
    if effective_min > 0:
        passing = chunk_df[chunk_df["n_channels_active"] >= effective_min].reset_index(drop=True)
    else:
        passing = chunk_df.reset_index(drop=True)

    if passing.empty:
        log.warning(
            f"No scan windows with >= {effective_min} active channels found."
        )
        windows = pd.DataFrame(
            columns=["start", "end", "n_days",
                     "min_channels_active", "mean_channels_active", "n_events"]
        )
    else:
        windows = []
        block_start = passing["start"].iloc[0]
        block_end   = passing["end"].iloc[0]
        block_n_ch  = [passing["n_channels_active"].iloc[0]]
        block_n_ev  = passing["n_events"].iloc[0]

        for i in range(1, len(passing)):
            row      = passing.iloc[i]
            prev_end = passing["end"].iloc[i - 1]
            if pd.Timestamp(row["start"]) <= pd.Timestamp(prev_end):
                # contiguous
                block_end = row["end"]
                block_n_ch.append(row["n_channels_active"])
                block_n_ev += row["n_events"]
            else:
                windows.append({
                    "start":                block_start,
                    "end":                  block_end,
                    "n_days":               (pd.Timestamp(block_end) - pd.Timestamp(block_start)).days,
                    "min_channels_active":  int(min(block_n_ch)),
                    "mean_channels_active": float(np.mean(block_n_ch)),
                    "n_events":             int(block_n_ev),
                })
                block_start = row["start"]
                block_end   = row["end"]
                block_n_ch  = [row["n_channels_active"]]
                block_n_ev  = row["n_events"]

        windows.append({
            "start":                block_start,
            "end":                  block_end,
            "n_days":               (pd.Timestamp(block_end) - pd.Timestamp(block_start)).days,
            "min_channels_active":  int(min(block_n_ch)),
            "mean_channels_active": float(np.mean(block_n_ch)),
            "n_events":             int(block_n_ev),
        })
        windows = pd.DataFrame(windows)

    total_days = windows["n_days"].sum() if not windows.empty else 0
    log.info(
        f"suggest_scan_windows: {len(windows)} window(s) identified "
        f"covering {total_days} days, "
        f"{windows['n_events'].sum() if not windows.empty else 0} template events."
    )
    for _, row in windows.iterrows():
        log.info(
            f"  {pd.Timestamp(row['start']).date()} – "
            f"{pd.Timestamp(row['end']).date()}  "
            f"({row['n_days']} days, "
            f"min {row['min_channels_active']} / "
            f"mean {row['mean_channels_active']:.1f} channels, "
            f"{row['n_events']} template events)"
        )

    # ── Write queried inventory ──────────────────────────────────────────
    if inventory_path is not None and _queried_inv is not None and _queried_inv.networks:
        try:
            _queried_inv.write(inventory_path, format="STATIONXML")
            log.info(f"Queried inventory written to {inventory_path}")
        except Exception as exc:
            log.warning(f"Could not write inventory to {inventory_path}: {exc}")

    # ── Optional output ───────────────────────────────────────────────────
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if not windows.empty:
            csv_path = os.path.join(out_dir, "scan_windows.csv")
            windows.to_csv(csv_path, index=False)
            log.info(f"Scan windows written to {csv_path}")
        _plot_scan_timeline(
            chunk_df=chunk_df,
            windows=windows,
            event_times=event_times,
            active_seeds=active_seeds,
            min_channels=effective_min,
            out_dir=out_dir,
        )

    return windows
