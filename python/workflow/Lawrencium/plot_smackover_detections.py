#!/usr/bin/env python
"""
plot_smackover_detections.py
----------------------------
Assessment plots for Smackover North MAD-12 matched-filter detections.

Reads the pre-built declustered Party file directly (read_detection_catalog=False
skips the per-template catalog XML, making the load fast).
Waveform shot-gather plots load the saved detection .mseed files.

Outputs are written to OUTPUT_DIR and labeled by stage:
  01_temporal_overview.png
  02_cc_quality.png
  03_template_stats.png
  04_activity_heatmap.png
  05_interevent_times.png
  06_daily_patterns.png
  07_template_map.png
  08_spacetime_migration.png
  stack_{template_name}.png   (one per top-N template)

Usage:
    /home/chopp/miniconda3/envs/py311/bin/python plot_smackover_detections.py
    /home/chopp/miniconda3/envs/py311/bin/python plot_smackover_detections.py --no-stacks
    /home/chopp/miniconda3/envs/py311/bin/python plot_smackover_detections.py --min-chans 3
"""

import argparse
import logging
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as mpe
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAVE_CARTOPY = True
except ImportError:
    _HAVE_CARTOPY = False
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read as obspy_read
from eqcorrscan import Tribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
PARTY_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/Smackover_all_dets_MAD12_decluster10.tgz"
)
WAVEFORM_DIR = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/waveforms/smackover_north_full_tribe/MAD12_2hr"
)
TRIBE_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/templates"
    "/Smackover_north_tribe_snr5.tgz"
)
OUTPUT_DIR = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/assessment_plots"
)

# Shot-gather parameters
STACK_PRE = 2.0     # seconds before detect_time
STACK_POST = 15.0   # seconds after detect_time
STACK_FMIN = 1.0
STACK_FMAX = 20.0
TOP_N_TEMPLATES = None    # None = all templates; set to an int to limit
MAX_DET_PER_PLOT = None    # subsample if a family has more detections
MAX_ALIGN_S = 1.0         # maximum CC-alignment shift applied to stack traces (seconds)
ALIGN_WINDOW_S = 2.0      # half-width of the reference window around t=0 used for CC alignment (seconds)
# Minimum number of channels required to keep a detection (≥ MIN_CHANS channels).
# Set to None to disable filtering. Uses Party.min_chans(MIN_CHANS - 1) internally
# (which keeps detections with no_chans > MIN_CHANS - 1, i.e. ≥ MIN_CHANS).
MIN_CHANS = 2
# Glitch / spike rejection: a detection is removed if the fraction of channels
# with excess kurtosis > GLITCH_KURT_THRESHOLD exceeds GLITCH_CHAN_FRAC.
# Constant / all-zero traces count as glitchy regardless of kurtosis.
# Gaussian noise kurtosis ≈ 0; transient spikes typically >> 10.
# With only 2 matched channels a single spikey channel inflates the CC sum
# enough to produce fictitious detections, so a low fraction threshold is
# appropriate.  Set GLITCH_KURT_THRESHOLD to None to disable entirely.
GLITCH_KURT_THRESHOLD = 10.0   # excess kurtosis limit (Fisher; Gaussian ≈ 0)
GLITCH_CHAN_FRAC = 0.75          # reject if fraction of glitchy channels ≥ this value
# Station inventory (StationXML) for nearest-station Z-channel selection.
INVENTORY_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/instruments/Smackover_10pick.xml"
)
# Minimum |CC| required for the self-detection (the template event run against
# its own template).  Raise an error if no detection in the family reaches this;
# that should never happen and indicates a corrupted or mismatched catalog.
SELF_DETECT_MIN_CC = 0.8


# ── Catalog builder ────────────────────────────────────────────────────────────


def filter_glitch_detections(
    df: pd.DataFrame,
    waveform_dir: str,
    tribe_path: str,
    kurt_threshold: float = GLITCH_KURT_THRESHOLD,
    chan_frac: float = GLITCH_CHAN_FRAC,
) -> pd.DataFrame:
    """
    Remove detections whose waveforms are dominated by glitch/spike artifacts.

    For each detection only the channels that belong to its template are
    evaluated — channels in the saved mseed that are NOT part of the template
    are ignored.  This prevents the many-channel saved waveform files from
    diluting the glitch fraction when the template itself only used a few
    channels (e.g. 2 channels → a single spikey trace is 50 % glitchy).

    A detection is REJECTED when the fraction of glitchy template channels
    meets or exceeds chan_frac.  Glitchy = constant/all-zero (std < 1e-10) OR
    excess kurtosis (Fisher; Gaussian ≈ 0) > kurt_threshold.

    Traces shorter than 10 samples are excluded from the count.
    If the waveform file is missing the detection is kept (cannot verify).
    """
    from scipy.stats import kurtosis as sp_kurt
    from eqcorrscan import Tribe

    if kurt_threshold is None:
        return df

    # ── Build per-template channel sets from the Tribe ────────────────────────
    log.info("Glitch filter: loading tribe to resolve template channel sets …")
    try:
        tribe = Tribe().read(tribe_path)
    except Exception as exc:
        log.warning(f"Glitch filter: could not load tribe ({exc}); skipping filter.")
        return df

    tmpl_chans: dict[str, set[str]] = {}
    for t in tribe:
        if t.st:
            tmpl_chans[t.name] = {tr.id for tr in t.st}

    keep = np.ones(len(df), dtype=bool)
    n = len(df)
    log.info(
        f"Glitch filter: checking {n:,} detections "
        f"(kurtosis threshold = {kurt_threshold}, chan_frac = {chan_frac}) …"
    )

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 2000 == 0 and i > 0:
            log.info(f"  … {i}/{n}")
        fpath = os.path.join(waveform_dir, f"{row['id']}.mseed")
        if not os.path.exists(fpath):
            continue  # no file → keep
        try:
            st = obspy_read(fpath)
        except Exception:
            continue  # unreadable → keep
        if not st:
            continue

        # Narrow to template channels; fall back to all channels if unknown
        allowed = tmpl_chans.get(row["template_name"])
        traces_to_check = [tr for tr in st if allowed is None or tr.id in allowed]
        if not traces_to_check:
            # Template channels absent in file — keep but log once at debug level
            continue

        n_eval = 0
        n_glitch = 0
        for tr in traces_to_check:
            data = tr.data.astype(float)
            if len(data) < 10:
                continue  # too short — exclude from count
            n_eval += 1
            if np.std(data) < 1e-10:  # constant / all-zero → glitchy
                n_glitch += 1
                continue
            k = sp_kurt(data, fisher=True)
            if np.isnan(k) or k > kurt_threshold:
                n_glitch += 1

        if n_eval > 0 and (n_glitch / n_eval) >= chan_frac:
            keep[i] = False

    df_out = df[keep].copy()
    n_bad = (~keep).sum()
    log.info(
        f"Glitch filter: removed {n_bad:,}/{n:,} detections ({100*n_bad/n:.1f}%); "
        f"{len(df_out):,} remain."
    )
    if n_bad > 0:
        by_tmpl = (
            df[~keep]
            .groupby("template_name")
            .size()
            .sort_values(ascending=False)
        )
        for tmpl, cnt in by_tmpl.items():
            log.info(f"  removed {cnt:4d}  {tmpl}")
    return df_out

def party_to_dataframe(party_path: str, min_chans: int | None = None) -> pd.DataFrame:
    """
    Read a declustered Party .tgz (read_detection_catalog=False skips per-template
    catalog XML, keeping load time short) and return a flat DataFrame.

    If min_chans is given, detections with fewer than min_chans channels are
    removed via Party.min_chans(min_chans - 1) before building the DataFrame.
    """
    from eqcorrscan import Party

    log.info(f"Reading party: {party_path}")
    party = Party().read(party_path, read_detection_catalog=False)

    if min_chans is not None:
        n_before = sum(len(f.detections) for f in party.families)
        party = party.min_chans(min_chans - 1)
        n_after = sum(len(f.detections) for f in party.families)
        log.info(
            f"min_chans filter (≥{min_chans}): {n_before:,} → {n_after:,} detections "
            f"({n_before - n_after:,} removed)"
        )

    records = []
    for fam in party.families:
        tmpl_name = fam.template.name
        for d in fam.detections:
            sta0, cha0 = d.chans[0] if d.chans else (None, None)
            records.append({
                "template_name": tmpl_name,
                "detect_time":   d.detect_time.datetime,
                "detect_val":    d.detect_val,
                "threshold":     d.threshold,
                "no_chans":      d.no_chans,
                "trig_chan":      f"{sta0}.{cha0}" if sta0 else None,
                "id":            d.id,
            })

    if not records:
        log.warning("No detections found in party.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["detect_time"] = pd.to_datetime(df["detect_time"], utc=True)
    df["cc_abs"]      = df["detect_val"].abs()
    df["cc_ratio"]    = df["cc_abs"] / df["threshold"].replace(0, np.nan)
    dt_naive = df["detect_time"].dt.tz_convert(None)
    df["year"]       = dt_naive.dt.year
    df["month"]      = dt_naive.dt.month
    df["hour"]       = dt_naive.dt.hour
    df["dayofweek"]  = dt_naive.dt.dayofweek  # 0=Mon
    df["yearmonth"]  = dt_naive.dt.to_period("M")
    df["net_prefix"] = df["template_name"].str.extract(r"^([a-zA-Z]+)", expand=False)

    log.info(
        f"Catalog ready: {len(df):,} detections across "
        f"{df['template_name'].nunique()} templates "
        f"({df['detect_time'].min().date()} – {df['detect_time'].max().date()})"
    )
    return df


# ── Plot helpers ───────────────────────────────────────────────────────────────

NET_COLORS = {
    "us":  "#1f77b4",
    "nm":  "#2ca02c",
    "tx":  "#d62728",
    "usb": "#9467bd",
}


def _net_color(prefix: str) -> str:
    return NET_COLORS.get(str(prefix).lower(), "#8c564b")


def _savefig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")


# ── Figure 1: Temporal overview ────────────────────────────────────────────────

def plot_temporal_overview(df: pd.DataFrame, out_dir: str) -> None:
    """Monthly detection rate + cumulative count."""
    monthly = (
        df.groupby("yearmonth")
        .size()
        .rename("count")
        .reset_index()
    )
    monthly["date"] = monthly["yearmonth"].dt.to_timestamp()
    monthly = monthly.sort_values("date")
    monthly["cumulative"] = monthly["count"].cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    ax1, ax2 = axes

    ax1.bar(monthly["date"], monthly["count"], width=25, color="#1f77b4",
            alpha=0.8, label="Monthly detections")
    ax1.set_ylabel("Detections / month", fontsize=11)
    ax1.set_title("Smackover North — Matched-Filter Detection Rate (MAD 12)", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.step(monthly["date"], monthly["cumulative"], where="post",
             color="#d62728", linewidth=1.5, label="Cumulative detections")
    ax2.set_ylabel("Cumulative detections", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "01_temporal_overview.png"))


# ── Figure 2: CC quality ───────────────────────────────────────────────────────

def plot_cc_quality(df: pd.DataFrame, out_dir: str) -> None:
    """CC ratio histogram + CC ratio vs time scatter + no_chans histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    ax_hist, ax_scat, ax_nch = axes

    bins = np.linspace(1.0, max(df["cc_ratio"].quantile(0.99), 3.0), 60)
    ax_hist.hist(df["cc_ratio"], bins=bins, color="#1f77b4", edgecolor="none", alpha=0.8)
    ax_hist.axvline(1.0, color="k", linestyle="--", linewidth=1, label="Threshold = 1×")
    ax_hist.axvline(1.5, color="orange", linestyle="--", linewidth=1, label="1.5×")
    ax_hist.axvline(2.0, color="red", linestyle="--", linewidth=1, label="2×")
    ax_hist.set_xlabel("|CC| / threshold", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_title("CC quality distribution", fontsize=11)
    ax_hist.legend(fontsize=9)

    scatter_df = df.sample(min(len(df), 10000), random_state=42)
    sc = ax_scat.scatter(
        scatter_df["detect_time"].dt.tz_localize(None).to_numpy(),
        scatter_df["cc_ratio"],
        c=scatter_df["cc_ratio"],
        cmap="plasma",
        s=4,
        alpha=0.5,
        vmin=1.0,
        vmax=df["cc_ratio"].quantile(0.95),
        rasterized=True,
    )
    ax_scat.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax_scat.set_xlabel("Date", fontsize=11)
    ax_scat.set_ylabel("|CC| / threshold", fontsize=11)
    ax_scat.set_title("|CC| / threshold over time (10 k random sample)", fontsize=11)
    fig.colorbar(sc, ax=ax_scat, label="|CC| / threshold")

    # ── no_chans histogram ────────────────────────────────────────────────────
    max_ch = int(df["no_chans"].max())
    ch_bins = np.arange(0.5, max_ch + 1.5, 1)
    ax_nch.hist(df["no_chans"], bins=ch_bins, color="#2ca02c", edgecolor="none", alpha=0.8)
    ax_nch.set_xlabel("Channels per detection", fontsize=11)
    ax_nch.set_ylabel("Count", fontsize=11)
    ax_nch.set_title("Channel count distribution", fontsize=11)
    ax_nch.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Annotate cumulative fraction on right y-axis
    ch_counts = df["no_chans"].value_counts().sort_index()
    cumfrac = ch_counts.cumsum() / len(df)
    ax_nch2 = ax_nch.twinx()
    ax_nch2.step(cumfrac.index, cumfrac.values, where="post",
                 color="#d62728", linewidth=1.2, label="Cumulative fraction")
    ax_nch2.set_ylim(0, 1.05)
    ax_nch2.set_ylabel("Cumulative fraction", fontsize=9, color="#d62728")
    ax_nch2.tick_params(axis="y", labelcolor="#d62728")

    fig.suptitle("Detection Quality — CC ratio", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "02_cc_quality.png"))


# ── Figure 3: Template statistics ─────────────────────────────────────────────

def plot_template_stats(df: pd.DataFrame, out_dir: str) -> None:
    """Ranked horizontal bar chart + active time span per template."""
    counts = df.groupby("template_name").size().sort_values(ascending=True)
    templates = counts.index.tolist()
    colors = [_net_color(df.loc[df["template_name"] == t, "net_prefix"].iloc[0])
              for t in templates]

    # Active time span
    spans = df.groupby("template_name")["detect_time"].agg(["min", "max"])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(templates) * 0.25 + 2)))
    ax_bar, ax_span = axes

    y = np.arange(len(templates))
    ax_bar.barh(y, counts.values, color=colors, edgecolor="none", alpha=0.85)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(templates, fontsize=7)
    ax_bar.set_xlabel("Total detections", fontsize=11)
    ax_bar.set_title("Detections per template (ranked)", fontsize=11)
    # legend for network prefixes
    seen = set()
    for tmpl, col in zip(templates, colors):
        pfx = df.loc[df["template_name"] == tmpl, "net_prefix"].iloc[0]
        if pfx not in seen:
            ax_bar.barh([], [], color=col, label=pfx)
            seen.add(pfx)
    ax_bar.legend(fontsize=9, title="Network")

    # Time span: horizontal line from first to last detection
    for i, tmpl in enumerate(templates):
        row = spans.loc[tmpl]
        t0 = row["min"].to_pydatetime()
        t1 = row["max"].to_pydatetime() if pd.notna(row["max"]) else t0
        ax_span.plot([t0, t1], [i, i], color=colors[i], linewidth=3, solid_capstyle="round")

    ax_span.set_yticks(y)
    ax_span.set_yticklabels(templates, fontsize=7)
    ax_span.set_xlabel("Date", fontsize=11)
    ax_span.set_title("Active detection span per template", fontsize=11)

    fig.suptitle("Template-level statistics", fontsize=13)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "03_template_stats.png"))


# ── Figure 4: Activity heatmap ─────────────────────────────────────────────────

def plot_activity_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """Template × year heatmap (annual totals)."""
    pivot = (
        df.groupby(["template_name", "year"])
        .size()
        .reset_index(name="count")
        .pivot(index="template_name", columns="year", values="count")
        .fillna(0)
    )
    # Sort templates by total detections descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    n_templates, n_years = pivot.shape
    fig_h = max(5, n_templates * 0.22 + 2)
    fig_w = max(10, n_years * 0.45 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = np.log10(pivot.values + 1)
    im = ax.imshow(data, aspect="auto", cmap="hot_r", origin="upper",
                   interpolation="nearest")
    ax.set_xticks(np.arange(n_years))
    ax.set_xticklabels(pivot.columns.astype(int), rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_templates))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title("Annual detection count per template  (colour = log₁₀(1+N))", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("log₁₀(1 + count)", fontsize=9)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "04_activity_heatmap.png"))


# ── Figure 5: Interevent times ─────────────────────────────────────────────────

def plot_interevent_times(df: pd.DataFrame, out_dir: str) -> None:
    """Interevent time histogram (overall and per top-5 template)."""
    df_sorted = df.sort_values(["template_name", "detect_time"])
    df_sorted["iet"] = (
        df_sorted.groupby("template_name")["detect_time"]
        .diff()
        .dt.total_seconds()
    )
    iet_all = df_sorted["iet"].dropna()

    top5 = df["template_name"].value_counts().head(5).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_all, ax_top = axes

    bins = np.logspace(np.log10(max(iet_all.min(), 1)), np.log10(iet_all.max()), 60)
    ax_all.hist(iet_all, bins=bins, color="#1f77b4", edgecolor="none", alpha=0.8)
    ax_all.set_xscale("log")
    ax_all.set_xlabel("Interevent time (s)", fontsize=11)
    ax_all.set_ylabel("Count", fontsize=11)
    ax_all.set_title("All-family interevent times", fontsize=11)
    # Mark key intervals
    for val, label in [(10, "10 s"), (3600, "1 hr"), (86400, "1 day")]:
        ax_all.axvline(val, color="gray", linestyle="--", linewidth=0.8)
        ax_all.text(val * 1.05, ax_all.get_ylim()[1] * 0.9, label, fontsize=7, color="gray")

    for tmpl in top5:
        iet_t = df_sorted.loc[df_sorted["template_name"] == tmpl, "iet"].dropna()
        if len(iet_t) < 2:
            continue
        ax_top.hist(iet_t, bins=bins, histtype="step", linewidth=1.2, label=tmpl)
    ax_top.set_xscale("log")
    ax_top.set_xlabel("Interevent time (s)", fontsize=11)
    ax_top.set_ylabel("Count", fontsize=11)
    ax_top.set_title("Interevent times — top-5 templates", fontsize=11)
    ax_top.legend(fontsize=8)

    fig.suptitle("Inter-detection time distribution", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "05_interevent_times.png"))


# ── Figure 6: Daily patterns ───────────────────────────────────────────────────

def plot_daily_patterns(df: pd.DataFrame, out_dir: str) -> None:
    """Hour-of-day and day-of-week detection distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_hr, ax_dow = axes

    hours = np.arange(24)
    hr_counts = df["hour"].value_counts().reindex(hours, fill_value=0)
    ax_hr.bar(hours, hr_counts.values, color="#1f77b4", edgecolor="none", alpha=0.8)
    ax_hr.set_xlabel("Hour of day (UTC)", fontsize=11)
    ax_hr.set_ylabel("Detections", fontsize=11)
    ax_hr.set_title("Hour-of-day distribution", fontsize=11)
    ax_hr.set_xticks(hours[::2])

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_counts = df["dayofweek"].value_counts().reindex(range(7), fill_value=0)
    colors_dow = ["#1f77b4"] * 5 + ["#d62728"] * 2  # weekdays blue, weekends red
    ax_dow.bar(dow_labels, dow_counts.values, color=colors_dow, edgecolor="none", alpha=0.8)
    ax_dow.set_xlabel("Day of week", fontsize=11)
    ax_dow.set_ylabel("Detections", fontsize=11)
    ax_dow.set_title("Day-of-week distribution", fontsize=11)

    fig.suptitle("Temporal patterns (anthropogenic noise check)", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "06_daily_patterns.png"))


# ── Figure 7+: Waveform shot-gather ───────────────────────────────────────────

def _nearest_z_chan(tmpl_obj, inventory, restrict_to: set | None = None) -> str:
    """
    Return the full SEED ID of the Z channel at the station nearest to the
    template's hypocenter, considering only Z channels present in tmpl_obj.st
    AND (if given) in restrict_to.

    Coordinates are looked up in the local inventory first.  Channels absent
    from it are queried via IRIS FDSNWS (network-level, no waveforms needed).
    Channels whose coordinates cannot be resolved are skipped with a warning.

    Raises ValueError if origin coordinates are missing, or if no Z channel's
    coordinates could be resolved at all.
    """
    from obspy.clients.fdsn import Client as FDSNClient
    from obspy.geodetics import gps2dist_azimuth

    org = (
        tmpl_obj.event.preferred_origin()
        or (tmpl_obj.event.origins[0] if tmpl_obj.event.origins else None)
    )
    if org is None or org.latitude is None:
        raise ValueError(
            f"Template {tmpl_obj.name}: no origin coordinates available."
        )

    _fdsn_client = None  # lazy-initialised only if needed

    best_id = None
    best_dist = float("inf")
    for tr in tmpl_obj.st:
        if not tr.stats.channel.endswith("Z"):
            continue
        if restrict_to is not None and tr.id not in restrict_to:
            continue
        # 1. Try local inventory
        try:
            coords = inventory.get_coordinates(tr.id, datetime=tr.stats.starttime)
        except Exception:
            # 2. Try IRIS FDSNWS
            try:
                if _fdsn_client is None:
                    _fdsn_client = FDSNClient("IRIS")
                net, sta, loc, cha = tr.id.split(".")
                remote_inv = _fdsn_client.get_stations(
                    network=net, station=sta, location=loc, channel=cha,
                    starttime=tr.stats.starttime, level="channel",
                )
                coords = remote_inv.get_coordinates(tr.id, datetime=tr.stats.starttime)
            except Exception as exc:
                log.warning(
                    f"  {tmpl_obj.name}: cannot resolve coordinates for {tr.id} "
                    f"(local inventory and IRIS both failed: {exc}); skipping channel."
                )
                continue

        dist, _, _ = gps2dist_azimuth(
            org.latitude, org.longitude,
            coords["latitude"], coords["longitude"],
        )
        if dist < best_dist:
            best_dist = dist
            best_id = tr.id

    if best_id is None:
        candidate_z = [tr.id for tr in tmpl_obj.st if tr.stats.channel.endswith("Z")]
        if restrict_to is not None:
            candidate_z = [c for c in candidate_z if c in restrict_to]
        raise ValueError(
            f"Template {tmpl_obj.name}: could not resolve coordinates for any "
            f"Z channel common to template and detection stream "
            f"(candidates: {candidate_z})."
        )
    return best_id


def _load_stack_traces(
    det_ids: list, waveform_dir: str, detect_times: list,
    trig_chans: list | None = None,
    fmin: float = STACK_FMIN, fmax: float = STACK_FMAX,
    target_samp_rate: float | None = None,
) -> tuple:
    """
    Load and trim waveform files for a list of detection IDs.
    Returns (traces_list, times_used, samp_rate, n_samples).
    Each trace is a numpy array of length n_samples.
    Detections whose mseed file does not contain the requested channel are skipped.
    """
    traces = []
    det_times_out = []

    samp_out = None
    n_target = None

    if trig_chans is None:
        trig_chans = [None] * len(det_ids)

    for det_id, det_time, trig_chan in zip(det_ids, detect_times, trig_chans):
        fpath = os.path.join(waveform_dir, f"{det_id}.mseed")
        if not os.path.exists(fpath):
            continue
        try:
            st = obspy_read(fpath)
        except Exception as exc:
            log.debug(f"  Cannot read {det_id}.mseed: {exc}")
            continue

        if trig_chan is None:
            continue  # no channel specified — skip rather than guess
        sel = st.select(id=trig_chan)
        if not sel:
            continue
        tr = sel[0]

        tr = tr.copy()
        tr.data = tr.data.astype(float)  # avoid int32 padding errors
        if target_samp_rate is not None and abs(tr.stats.sampling_rate - target_samp_rate) > 0.5:
            tr.resample(target_samp_rate)
        detect_utc = UTCDateTime(pd.Timestamp(det_time).timestamp())
        t0 = detect_utc - STACK_PRE
        t1 = detect_utc + STACK_POST
        tr = tr.copy().trim(t0, t1, pad=True, fill_value=0.0)
        tr.detrend("demean")
        tr.taper(0.05)
        nyq = tr.stats.sampling_rate / 2.0
        safe_fmax = min(fmax, nyq * 0.999)
        tr.filter("bandpass", freqmin=fmin, freqmax=safe_fmax, corners=4, zerophase=True)

        data = tr.data.astype(float)
        norm = np.max(np.abs(data))
        if norm == 0:
            continue
        data /= norm

        if samp_out is None:
            samp_out = tr.stats.sampling_rate
            n_target = len(data)

        # Pad or trim to consistent length
        if len(data) < n_target:
            data = np.pad(data, (0, n_target - len(data)))
        else:
            data = data[:n_target]

        traces.append(data)
        det_times_out.append(det_time)

    return traces, det_times_out, samp_out, (n_target or 0)


def _align_traces(
    traces_arr: np.ndarray, ref_data: np.ndarray, max_shift_samples: int,
    t_axis: np.ndarray, t_lo: float, t_hi: float,
) -> np.ndarray:
    """
    Shift each row of traces_arr to maximise cross-correlation with ref_data.
    Only the ref_data samples in [t_lo, t_hi] (seconds) are used as the
    alignment kernel — should span the expected phase onset
    (e.g. -prepick to -prepick + ALIGN_WINDOW_S).
    Shift is capped at ±max_shift_samples; shifted-in samples are zero-padded.
    """
    from scipy.signal import correlate

    mask = (t_axis >= t_lo) & (t_axis <= t_hi)
    kernel = ref_data[mask]
    if kernel.size == 0:
        kernel = ref_data  # fallback

    N = traces_arr.shape[1]
    aligned = np.zeros_like(traces_arr)
    for i, det in enumerate(traces_arr):
        cc = correlate(det, kernel, mode="valid")
        # In 'valid' mode len(cc) = N - len(kernel) + 1
        # center = first index in det where the kernel (starting at t_lo) aligns.
        center = np.where(mask)[0][0]
        lo = max(0, center - max_shift_samples)
        hi = min(len(cc), center + max_shift_samples + 1)
        if lo >= hi:
            aligned[i] = det
            continue
        best = np.argmax(cc[lo:hi]) + lo
        lag = best - center
        if lag == 0:
            aligned[i] = det
        elif lag > 0:                           # detection arrived late → shift left
            aligned[i, :N - lag] = det[lag:]
        else:                                   # detection arrived early → shift right
            aligned[i, -lag:] = det[:N + lag]
    return aligned


def plot_waveform_stack(
    df: pd.DataFrame,
    waveform_dir: str,
    tribe_path: str,
    out_dir: str,
    top_n: int = TOP_N_TEMPLATES,
    max_det: int = MAX_DET_PER_PLOT,
) -> None:
    """
    Shot-gather style waveform stack for the top-N templates.
    Traces sorted chronologically, coloured by |CC|/threshold.
    Template waveform overlaid at top; mean stack shown below.
    """
    log.info(f"Loading tribe for template waveforms: {tribe_path}")
    tribe = Tribe().read(tribe_path)
    tribe_dict = {t.name: t for t in tribe.templates}

    log.info(f"Loading station inventory: {INVENTORY_PATH}")
    from obspy import read_inventory
    inventory = read_inventory(INVENTORY_PATH)

    top_templates = df["template_name"].value_counts()
    if top_n is not None:
        top_templates = top_templates.head(top_n)
    top_templates = top_templates.index.tolist()

    for tmpl_name in top_templates:
        sub = df[df["template_name"] == tmpl_name].sort_values("detect_time")
        log.info(f"  Shot gather for {tmpl_name}: {len(sub)} detections")

        # Self-detection: detect_val / no_chans should be ~1.0.
        # detect_val is the sum of per-channel CCs; dividing by no_chans gives
        # the average per-channel CC, which equals 1.0 when the template
        # correlates perfectly with its own stored waveform.
        sub = sub.copy()
        sub["cc_per_chan"] = sub["detect_val"] / sub["no_chans"].replace(0, np.nan)
        sub["cc_per_chan_dist"] = (sub["cc_per_chan"] - 1.0).abs()
        peak_row = sub.nsmallest(1, "cc_per_chan_dist").iloc[0]
        if peak_row["cc_per_chan"] < SELF_DETECT_MIN_CC:
            log.warning(
                f"Template {tmpl_name}: no self-detection found "
                f"(best detect_val/no_chans = {peak_row['cc_per_chan']:.3f}, "
                f"required ≥ {SELF_DETECT_MIN_CC}). "
                "Skipping — the template event should always detect itself."
            )
            continue

        if max_det is not None and len(sub) > max_det:
            sub = sub.sample(max_det, random_state=42).sort_values("detect_time")
            log.info(f"    Subsampled to {max_det}")

        det_ids = sub["id"].tolist()
        det_times = sub["detect_time"].dt.tz_localize(None).to_numpy()
        cc_ratios = sub["cc_ratio"].values

        tmpl_obj = tribe_dict.get(tmpl_name)
        if tmpl_obj is None:
            raise ValueError(
                f"Template {tmpl_name} found in detections but not in tribe."
            )
        proc_fmin = tmpl_obj.lowcut or STACK_FMIN
        proc_fmax = tmpl_obj.highcut or STACK_FMAX
        proc_samp_rate = tmpl_obj.samp_rate or None

        # ── Load self-detection mseed first to know which channels are available
        fpath_peak = os.path.join(waveform_dir, f"{peak_row['id']}.mseed")
        if not os.path.exists(fpath_peak):
            log.warning(
                f"    {tmpl_name}: self-detection waveform file missing "
                f"({peak_row['id']}.mseed); skipping."
            )
            continue
        st_peak = obspy_read(fpath_peak)
        det_chans = {tr.id for tr in st_peak}

        try:
            best_chan = _nearest_z_chan(tmpl_obj, inventory, restrict_to=det_chans)
        except ValueError as exc:
            log.warning(f"    {tmpl_name}: skipping — {exc}")
            continue
        chan_ids = [best_chan] * len(det_ids)

        # ── Template waveform on best_chan ──────────────────────────────────
        tmpl_tr_sel = tmpl_obj.st.select(id=best_chan)
        if not tmpl_tr_sel:
            raise ValueError(
                f"Template {tmpl_name}: nearest-Z channel {best_chan} not found "
                f"in template stream ({[tr.id for tr in tmpl_obj.st]})."
            )
        tmpl_tr_raw = tmpl_tr_sel[0].copy()
        # Template.st is already fully processed by EQcorrscan (detrended, filtered,
        # resampled). Do not re-process; only resample if sample rate mismatches.
        if proc_samp_rate is not None and abs(tmpl_tr_raw.stats.sampling_rate - proc_samp_rate) > 0.5:
            tmpl_tr_raw.resample(proc_samp_rate)
        td = tmpl_tr_raw.data.astype(float)
        # t=0 on the plot is detect_time. The template's first sample IS at
        # detect_time (since detect_time = pick_time - prepick). The onset
        # therefore appears at t=+prepick_s on the detection-relative axis.
        t_tmpl = np.linspace(
            0.0,
            len(td) / tmpl_tr_raw.stats.sampling_rate,
            len(td),
        )

        traces, times_used, samp_rate, n_samples = _load_stack_traces(
            det_ids, waveform_dir, det_times, chan_ids,
            fmin=proc_fmin, fmax=proc_fmax, target_samp_rate=proc_samp_rate,
        )
        if len(traces) < 2:
            log.warning(f"    Not enough waveforms for {tmpl_name}, skipping.")
            continue

        traces_arr = np.array(traces)   # shape (N, n_samples)

        # Match cc_ratios to loaded traces (some files may be missing)
        # Build mapping by time
        loaded_times = set(str(t)[:22] for t in times_used)
        cc_arr = []
        for t, cc in zip(det_times, cc_ratios):
            if str(t)[:22] in loaded_times:
                cc_arr.append(cc)
        cc_arr = np.array(cc_arr[:len(traces_arr)])
        if len(cc_arr) < len(traces_arr):
            cc_arr = np.ones(len(traces_arr))  # fallback

        t_axis = np.linspace(-STACK_PRE, STACK_POST, n_samples)

        # Self-detection trace: st_peak already loaded above; re-use it.
        ref_interp = None
        try:
            _sel = st_peak.select(id=best_chan)
            if not _sel:
                raise ValueError(f"{best_chan} not found in self-detection stream")
            tr_peak = _sel[0].copy()
            tr_peak.data = tr_peak.data.astype(float)
            detect_utc_peak = UTCDateTime(pd.Timestamp(peak_row["detect_time"]).timestamp())
            tr_peak.trim(detect_utc_peak - STACK_PRE,
                         detect_utc_peak + STACK_POST,
                         pad=True, fill_value=0.0)
            tr_peak.detrend("demean")
            tr_peak.taper(0.05)
            if proc_samp_rate is not None and abs(tr_peak.stats.sampling_rate - proc_samp_rate) > 0.5:
                tr_peak.resample(proc_samp_rate)
            nyq_peak = tr_peak.stats.sampling_rate / 2.0
            safe_fmax_peak = min(proc_fmax, nyq_peak * 0.999)
            tr_peak.filter("bandpass", freqmin=proc_fmin, freqmax=safe_fmax_peak,
                           corners=4, zerophase=True)
            rd = tr_peak.data.astype(float)
            sr_peak = tr_peak.stats.sampling_rate
            n_peak = len(rd)
            t_peak = np.linspace(-STACK_PRE, STACK_POST, n_peak)
            ref_interp = np.interp(t_axis, t_peak, rd)
            # Normalize template by its own maximum.
            norm_t = np.max(np.abs(td))
            if norm_t > 1e-10:
                td /= norm_t
            # Normalize self-detection by its maximum within the window that
            # overlaps the template [0, t_tmpl[-1]].
            t_overlap_end = t_tmpl[-1]
            mask_d = (t_axis >= 0.0) & (t_axis <= t_overlap_end)
            if mask_d.any():
                norm_d = np.max(np.abs(ref_interp[mask_d]))
            else:
                norm_d = np.max(np.abs(ref_interp))
            if norm_d > 1e-10:
                ref_interp /= norm_d
        except Exception as exc:
            log.warning(f"    Cannot process self-detection trace for {tmpl_name}: {exc}")

        # CC-based alignment: window starts at -prepick (template onset), length = ALIGN_WINDOW_S
        align_prepick = tmpl_obj.prepick if tmpl_obj is not None else 0.0
        if ref_interp is not None and samp_rate is not None:
            max_shift = int(MAX_ALIGN_S * samp_rate)
            t_lo_align = -align_prepick
            t_hi_align = -align_prepick + ALIGN_WINDOW_S
            traces_arr = _align_traces(traces_arr, ref_interp, max_shift,
                                       t_axis, t_lo_align, t_hi_align)

        # Find which row in the loaded-traces array is the self-detection
        peak_row_idx = None
        if peak_row["id"] in det_ids:
            peak_t = pd.Timestamp(peak_row["detect_time"])
            if peak_t.tzinfo is not None:
                peak_t = peak_t.tz_localize(None)
            for idx, t in enumerate(times_used):
                if abs((pd.Timestamp(t) - peak_t).total_seconds()) < 1.0:
                    peak_row_idx = idx
                    break

        mean_stack = np.mean(traces_arr, axis=0)
        n_det = len(traces_arr)

        # Amplitude colour limits: symmetric, clipped at 95th percentile
        amp_lim = np.percentile(np.abs(traces_arr), 95)
        amp_lim = max(amp_lim, 1e-6)

        # ── Figure: three rows × two columns
        # Row 0: template/self-detection trace (thin)
        # Row 1: waveform matrix (tall) ← colorbar spans rows 1-2
        # Row 2: mean stack panel (short)
        fig_h = max(6, min(n_det * 0.07 + 5, 32))
        mat_h = max(4, n_det * 0.07)
        fig = plt.figure(figsize=(13, fig_h), layout="constrained")
        gs = gridspec.GridSpec(
            3, 2,
            width_ratios=[30, 1],
            height_ratios=[1, mat_h, 2],
            figure=fig,
        )
        ax_tmpl  = fig.add_subplot(gs[0, 0])
        ax_mat   = fig.add_subplot(gs[1, 0], sharex=ax_tmpl)
        ax_stack = fig.add_subplot(gs[2, 0], sharex=ax_tmpl)
        cax      = fig.add_subplot(gs[1:, 1])  # colorbar spans matrix + stack rows

        # ── Reference panel: self-detection (black) + template.st (red) ───────
        if ref_interp is None:
            raise ValueError(
                f"Template {tmpl_name}: failed to load self-detection waveform "
                f"from {peak_row['id']}.mseed on channel {best_chan}."
            )
        ax_tmpl.plot(t_axis, ref_interp, color="k", linewidth=1.2,
                     label="Self-detection", zorder=3)
        valid = (t_tmpl >= t_axis[0]) & (t_tmpl <= t_axis[-1])
        if valid.any():
            ax_tmpl.plot(t_tmpl[valid], td[valid], color="#d62728", linewidth=1.2,
                         alpha=0.5, label="Template", zorder=4)
        ax_tmpl.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_tmpl.set_ylabel("norm.", fontsize=9)
        ax_tmpl.set_yticks([])
        ax_tmpl.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax_tmpl.set_title(
            f"Shot gather — {tmpl_name}  (N={n_det}, chan: {best_chan})",
            fontsize=11,
        )

        # ── Matrix panel ──────────────────────────────────────────────────────
        # traces_arr shape: (n_det, n_samples); rows = detections, oldest at top
        im = ax_mat.imshow(
            traces_arr,
            aspect="auto",
            extent=[-STACK_PRE, STACK_POST, n_det, 0],  # y: 0=top (oldest)
            cmap="RdBu_r",
            vmin=-amp_lim,
            vmax=amp_lim,
            interpolation="nearest",
            origin="upper",
            rasterized=True,
        )
        ax_mat.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_mat.set_ylabel("Detection index (chronological)", fontsize=10)
        ax_mat.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax_mat.tick_params(labelbottom=False)  # x labels on ax_stack instead

        # Self-detection marker: white line with black stroke, label inside axes
        if peak_row_idx is not None:
            _stroke = [mpe.withStroke(linewidth=3.0, foreground="black")]
            ax_mat.axhline(peak_row_idx + 0.5, color="white", linewidth=1.8,
                           linestyle="-", zorder=6,
                           path_effects=_stroke)
            # axes-fraction y: imshow extent is inverted (0 at top, n_det at bottom)
            # so fraction = 1 - (data_y / n_det)
            ann_y_frac = 1.0 - (peak_row_idx + 0.5) / n_det
            ann = ax_mat.annotate(
                "self ▶",
                xy=(0.0, ann_y_frac),
                xycoords=("axes fraction", "axes fraction"),
                xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8, color="white",
                path_effects=_stroke,
                annotation_clip=False,
                zorder=7,
            )

        # Colourbar spans matrix + stack rows
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Normalised amplitude", fontsize=9)

        # ── Mean stack panel ──────────────────────────────────────────────
        ax_stack.fill_between(t_axis, mean_stack, alpha=0.35, color="#1f77b4")
        ax_stack.plot(t_axis, mean_stack, color="#1f77b4", linewidth=1.2)
        ax_stack.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_stack.axhline(0.0, color="k", linewidth=0.4, alpha=0.4)
        ax_stack.set_xlabel("Time relative to detection (s)", fontsize=11)
        ax_stack.set_ylabel("Mean stack", fontsize=9)

        _savefig(fig, os.path.join(out_dir, f"stack_{tmpl_name}.png"))


# ── Summary text report ────────────────────────────────────────────────────────


def _load_template_locs(tribe_path: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with one row per template that appears in df.
    Columns: template_name, lat, lon, depth_km, net_prefix, n_det, mean_cc_ratio.
    Templates with no origin are silently dropped.
    """
    from eqcorrscan import Tribe
    tribe = Tribe().read(tribe_path)
    counts   = df.groupby("template_name").size().rename("n_det")
    mean_cc  = df.groupby("template_name")["cc_ratio"].mean().rename("mean_cc_ratio")
    rows = []
    for t in tribe.templates:
        try:
            org = t.event.preferred_origin() or t.event.origins[0]
            if org is None or org.latitude is None:
                continue
        except (IndexError, AttributeError):
            continue
        pfx = re.match(r'^([a-zA-Z]+)', t.name)
        pfx = pfx.group(1).lower() if pfx else "other"
        rows.append({
            "template_name": t.name,
            "lat":           org.latitude,
            "lon":           org.longitude,
            "depth_km":      (org.depth or 0) / 1000.0,
            "net_prefix":    pfx,
        })
    loc_df = pd.DataFrame(rows)
    if loc_df.empty:
        return loc_df
    loc_df = loc_df.merge(counts,  on="template_name", how="left")
    loc_df = loc_df.merge(mean_cc, on="template_name", how="left")
    loc_df["n_det"] = loc_df["n_det"].fillna(0).astype(int)
    return loc_df


# ── Figure 7: Detection bubble map ────────────────────────────────────────────

def plot_map(df: pd.DataFrame, tribe_path: str, out_dir: str) -> None:
    """
    Two-panel figure:
      Left  – map of template locations; bubble size ∝ √(n_det),
               colour = network prefix, annotated with template name.
      Right – depth cross-section (lon vs depth) with the same bubbles.
    Requires cartopy; falls back to plain scatter if not available.
    """
    log.info("Building template location data …")
    loc_df = _load_template_locs(tribe_path, df)
    if loc_df.empty:
        log.warning("No template locations found — skipping map.")
        return

    # Colour by network prefix
    prefixes = sorted(loc_df["net_prefix"].unique())
    cmap_net = plt.get_cmap("tab10")
    pfx_color = {p: cmap_net(i / max(len(prefixes) - 1, 1)) for i, p in enumerate(prefixes)}
    colors = loc_df["net_prefix"].map(pfx_color)

    # Bubble size: scale so the largest template gets a 400 pt² marker
    max_n = loc_df["n_det"].max() or 1
    sizes = 20 + 380 * (loc_df["n_det"] / max_n)

    lon_pad = max(0.3, (loc_df["lon"].max() - loc_df["lon"].min()) * 0.15)
    lat_pad = max(0.2, (loc_df["lat"].max() - loc_df["lat"].min()) * 0.15)
    extent = [
        loc_df["lon"].min() - lon_pad, loc_df["lon"].max() + lon_pad,
        loc_df["lat"].min() - lat_pad, loc_df["lat"].max() + lat_pad,
    ]

    fig = plt.figure(figsize=(16, 8))
    if _HAVE_CARTOPY:
        proj = ccrs.PlateCarree()
        ax_map  = fig.add_subplot(1, 2, 1, projection=proj)
        ax_xsec = fig.add_subplot(1, 2, 2)

        ax_map.set_extent(extent, crs=proj)
        ax_map.add_feature(cfeature.LAND,   facecolor="#f5f5f0", zorder=0)
        ax_map.add_feature(cfeature.OCEAN,  facecolor="#d0e8f5", zorder=0)
        ax_map.add_feature(cfeature.STATES, linewidth=0.6, edgecolor="#888888", zorder=1)
        ax_map.add_feature(cfeature.RIVERS, linewidth=0.4, edgecolor="#6699cc",
                           alpha=0.6, zorder=1)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
        sc = ax_map.scatter(
            loc_df["lon"], loc_df["lat"],
            s=sizes, c=colors,
            transform=proj, zorder=4,
            edgecolors="k", linewidths=0.5, alpha=0.85,
        )
        gl = ax_map.gridlines(draw_labels=True, linewidth=0.4,
                              color="gray", alpha=0.6, linestyle="--")
        gl.top_labels = gl.right_labels = False
    else:
        ax_map  = fig.add_subplot(1, 2, 1)
        ax_xsec = fig.add_subplot(1, 2, 2)
        sc = ax_map.scatter(
            loc_df["lon"], loc_df["lat"],
            s=sizes, c=colors,
            edgecolors="k", linewidths=0.5, alpha=0.85,
        )
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_aspect("equal")

    # Annotate with template name (small font, offset)
    for _, row in loc_df.iterrows():
        kw = dict(fontsize=5.5, color="#333333", clip_on=True)
        if _HAVE_CARTOPY:
            ax_map.text(row["lon"] + 0.02, row["lat"] + 0.02,
                        row["template_name"], transform=proj, **kw)
        else:
            ax_map.text(row["lon"] + 0.02, row["lat"] + 0.02,
                        row["template_name"], **kw)

    ax_map.set_title("Template locations  (bubble ∝ √N detections)", fontsize=11)

    # Network legend — tiny, inside map, just coloured squares
    for pfx in prefixes:
        ax_map.scatter([], [], s=30, color=pfx_color[pfx],
                       edgecolors="k", linewidths=0.4, label=pfx.upper())
    ax_map.legend(title="Network", fontsize=7, title_fontsize=7,
                  loc="lower right", framealpha=0.7,
                  handlelength=0.8, handletextpad=0.4, borderpad=0.4,
                  markerscale=0.8)

    # Depth cross-section: lon vs depth
    ax_xsec.scatter(
        loc_df["lon"], loc_df["depth_km"],
        s=sizes, c=colors,
        edgecolors="k", linewidths=0.5, alpha=0.85,
    )
    ax_xsec.invert_yaxis()
    ax_xsec.set_xlabel("Longitude", fontsize=11)
    ax_xsec.set_ylabel("Depth (km)", fontsize=11)
    ax_xsec.set_title("Depth cross-section", fontsize=11)
    ax_xsec.grid(True, linewidth=0.4, alpha=0.5)

    # Size legend — compact, inside cross-section panel
    for n_ref in [10, 100, 500, 2000]:
        if n_ref > max_n * 1.5:
            continue
        s_ref = 20 + 380 * (n_ref / max_n)
        ax_xsec.scatter([], [], s=s_ref, color="#aaaaaa",
                        edgecolors="k", linewidths=0.5,
                        label=f"{n_ref}")
    ax_xsec.legend(title="N det", fontsize=7, title_fontsize=7,
                   loc="lower right", framealpha=0.7,
                   handlelength=0.8, handletextpad=0.4, borderpad=0.4,
                   markerscale=0.7)

    fig.suptitle("Smackover North — Template Catalogue Geography", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig(fig, os.path.join(out_dir, "07_template_map.png"))


# ── Figure 8: Space-time migration ────────────────────────────────────────────

def plot_spacetime(df: pd.DataFrame, tribe_path: str, out_dir: str) -> None:
    """
    Space-time migration plot: each DETECTION is a point at
      x = detect_time,  y = template latitude  (or distance from centroid)
    coloured by template, sized by CC ratio.
    A second panel shows longitude vs time for E-W migration.
    """
    log.info("Building space-time migration plot …")
    loc_df = _load_template_locs(tribe_path, df)
    if loc_df.empty:
        log.warning("No template locations — skipping space-time plot.")
        return

    # Merge detection times with template locations
    # df already has net_prefix; drop it from loc_df to avoid _x/_y suffix collision
    merged = df.merge(loc_df[["template_name", "lat", "lon", "depth_km"]],
                      on="template_name", how="inner")
    if merged.empty:
        log.warning("No overlapping templates/detections — skipping space-time plot.")
        return

    # Colour by template (tab20 if ≤20 templates, else cycle)
    tmpl_list = merged["template_name"].unique().tolist()
    n_t = len(tmpl_list)
    cmap_t = plt.get_cmap("tab20" if n_t <= 20 else "nipy_spectral")
    tmpl_color = {t: cmap_t(i / max(n_t - 1, 1)) for i, t in enumerate(tmpl_list)}
    t_colors = merged["template_name"].map(tmpl_color)

    # Marker size proportional to CC ratio, capped for readability
    cc_clipped = merged["cc_ratio"].clip(upper=merged["cc_ratio"].quantile(0.95))
    cc_norm = (cc_clipped - cc_clipped.min()) / (cc_clipped.max() - cc_clipped.min() + 1e-9)
    sizes = (4 + 18 * cc_norm).values

    times = merged["detect_time"].dt.tz_localize(None).to_numpy()

    # Compute distance (km) from centroid for N-S and E-W
    lat0 = merged["lat"].mean()
    lon0 = merged["lon"].mean()
    # 1° lat ≈ 111 km;  1° lon ≈ 111 * cos(lat) km
    cos_lat = np.cos(np.radians(lat0))
    merged["dist_ns"] = (merged["lat"] - lat0) * 111.0
    merged["dist_ew"] = (merged["lon"] - lon0) * 111.0 * cos_lat

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    ax_ns, ax_ew, ax_depth = axes

    kw = dict(alpha=0.4, linewidths=0, rasterized=True)

    ax_ns.scatter(times, merged["dist_ns"], s=sizes, c=t_colors, **kw)
    ax_ns.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax_ns.set_ylabel("N–S distance from\ncentroid (km)", fontsize=10)
    ax_ns.set_title("Space-time migration  (colour = template, size ∝ |CC|/threshold)",
                    fontsize=12)
    ax_ns.grid(True, linewidth=0.4, alpha=0.4)

    ax_ew.scatter(times, merged["dist_ew"], s=sizes, c=t_colors, **kw)
    ax_ew.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax_ew.set_ylabel("E–W distance from\ncentroid (km)", fontsize=10)
    ax_ew.grid(True, linewidth=0.4, alpha=0.4)

    ax_depth.scatter(times, merged["depth_km"], s=sizes, c=t_colors, **kw)
    ax_depth.invert_yaxis()
    ax_depth.set_ylabel("Template depth (km)", fontsize=10)
    ax_depth.set_xlabel("Date", fontsize=11)
    ax_depth.grid(True, linewidth=0.4, alpha=0.4)

    # Network-colour legend patches (one per template would be too many)
    prefixes = sorted(loc_df["net_prefix"].unique())
    cmap_net = plt.get_cmap("tab10")
    pfx_color = {p: cmap_net(i / max(len(prefixes) - 1, 1)) for i, p in enumerate(prefixes)}
    # Build per-template colour legend grouped by network
    from matplotlib.lines import Line2D
    handles, labels_ = [], []
    for tmpl in tmpl_list:
        pfx = merged.loc[merged["template_name"] == tmpl, "net_prefix"].iloc[0]
        # Use the template colour (not network colour) for the legend dot
        handles.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=tmpl_color[tmpl],
                               markersize=5, label=tmpl))
        labels_.append(tmpl)
    if len(handles) <= 30:
        ax_ns.legend(handles=handles, labels=labels_,
                     fontsize=6, ncol=max(1, n_t // 10 + 1),
                     loc="upper right", framealpha=0.7, title="Template",
                     title_fontsize=7)

    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "08_spacetime_migration.png"))



def write_summary(df: pd.DataFrame, out_dir: str) -> None:
    """Write a plain-text summary of the detection catalog."""
    path = os.path.join(out_dir, "detection_summary.txt")
    top10 = df["template_name"].value_counts().head(10)
    with open(path, "w") as fh:
        fh.write("Smackover North — Matched-Filter Detection Summary\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Total detections      : {len(df):,}\n")
        fh.write(f"Templates active       : {df['template_name'].nunique()}\n")
        fh.write(
            f"Date range             : "
            f"{df['detect_time'].min().date()} – "
            f"{df['detect_time'].max().date()}\n"
        )
        fh.write(f"Median |CC|/threshold  : {df['cc_ratio'].median():.3f}\n")
        fh.write(f"Mean   |CC|/threshold  : {df['cc_ratio'].mean():.3f}\n")
        fh.write(f"Detections (no_chans=1): {(df['no_chans'] == 1).sum():,}\n")
        fh.write(f"Detections (no_chans>1): {(df['no_chans'] > 1).sum():,}\n")
        fh.write("\nTop 10 templates by detection count:\n")
        for i, (tmpl, n) in enumerate(top10.items(), 1):
            fh.write(f"  {i:2d}. {tmpl:<22} {n:5d}\n")
        fh.write("\nTrigger channel distribution (top 10):\n")
        for chan, n in df["trig_chan"].value_counts().head(10).items():
            fh.write(f"  {chan:<25} {n:5d}\n")
    log.info(f"Summary written to {path}")



# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smackover MF detection assessment plots")
    parser.add_argument(
        "--no-stacks", action="store_true",
        help="Skip waveform shot-gather plots (faster)"
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_TEMPLATES,
        help=f"Number of templates to produce shot gathers for (default: {TOP_N_TEMPLATES})"
    )
    parser.add_argument(
        "--max-det", type=int, default=MAX_DET_PER_PLOT,
        help=f"Max detections per shot gather (default: {MAX_DET_PER_PLOT})"
    )
    parser.add_argument(
        "--min-chans", type=int, default=MIN_CHANS,
        help=(
            "Keep only detections with ≥ this many channels "
            "(uses Party.min_chans internally; default: no filter)"
        ),
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──── Load declustered party ──────────────────────────────────────────────
    df = party_to_dataframe(PARTY_PATH, min_chans=args.min_chans)
    if df.empty:
        log.error("Empty catalog — nothing to plot.")
        return
    if GLITCH_KURT_THRESHOLD is not None:
        df = filter_glitch_detections(
            df, WAVEFORM_DIR, TRIBE_PATH, GLITCH_KURT_THRESHOLD, GLITCH_CHAN_FRAC
        )
    if df.empty:
        log.error("All detections removed by glitch filter — nothing to plot.")
        return

    # ──── Statistical plots ───────────────────────────────────────────────────
    log.info("Generating statistical plots …")
    plot_temporal_overview(df, OUTPUT_DIR)
    plot_cc_quality(df, OUTPUT_DIR)
    plot_template_stats(df, OUTPUT_DIR)
    plot_activity_heatmap(df, OUTPUT_DIR)
    plot_interevent_times(df, OUTPUT_DIR)
    plot_daily_patterns(df, OUTPUT_DIR)
    write_summary(df, OUTPUT_DIR)
    plot_map(df, TRIBE_PATH, OUTPUT_DIR)
    plot_spacetime(df, TRIBE_PATH, OUTPUT_DIR)

    # ──── Waveform stacks ─────────────────────────────────────────────────────
    if not args.no_stacks:
        log.info("Generating waveform shot-gather plots …")
        plot_waveform_stack(
            df, WAVEFORM_DIR, TRIBE_PATH, OUTPUT_DIR,
            top_n=args.top_n, max_det=args.max_det,
        )
    else:
        log.info("Skipping waveform stacks (--no-stacks).")

    log.info(f"All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
