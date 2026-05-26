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
  stack_{template_name}.png   (one per top-N template)

Usage:
    /home/chopp/miniconda3/envs/py311/bin/python plot_smackover_detections.py
    /home/chopp/miniconda3/envs/py311/bin/python plot_smackover_detections.py --no-stacks
"""

import argparse
import logging
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
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
# Minimum number of channels that must correlate above threshold for a detection
# to be retained.  Change this to tighten or relax the quality filter.
MIN_CHANS = 3        # no_chans >= MIN_CHANS

PARTY_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/Smackover_analyzed_raw.tgz"
)
WAVEFORM_DIR = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/waveforms/smackover_north_analyzed/MAD12_2hr"
)
TRIBE_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/templates"
    "/Smackover_north_tribe_analyzed_4-17-2026.tgz"
)
# Per-template best-channel lookup from tribe_analysis output.
# columns: event_id, seed_id, median_snr, pass_all …
# When present, this takes precedence over the on-the-fly SNR calculation.
TEMPLATE_CHAN_CSV = (
    "/media/chopp/HDD1/chet-meq/smackover/templates/tribe_analysis"
    "/selected_template_channels.csv"
)
OUTPUT_DIR = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/assessment_plots_analyzed/no_chans3/"
)
SWD_MAT_PATH = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/assessment_plots_analyzed/smackover_swd.mat"
)
SWD_RADIUS_KM = 20.0  # wells within this distance of a template contribute to injection total

# Fixed time bounds for all per-template quality plots (matches the scan window)
SCAN_START = pd.Timestamp("2009-02-12", tz="UTC").tz_convert(None).to_pydatetime()
SCAN_END   = pd.Timestamp("2026-03-31", tz="UTC").tz_convert(None).to_pydatetime()

# ── Per-template spike-day exclusions ─────────────────────────────────────────
# Days where detections are almost certainly noise (e.g. glitches, network
# outages, or single-channel false-positive bursts) that would overwhelm the
# rest of the catalog.  Add entries as:
#   "template_name": ["YYYY-MM-DD", "YYYY-MM-DD", ...]
# Applied before all analysis (including df_raw), so excluded days vanish
# entirely.  Check quality_vs_time plots to identify candidates.
SPIKE_DAY_EXCLUSIONS: dict[str, list[str]] = {
    # Identified automatically: days where per-template detections are >10x
    # the template's median daily rate (no_chans >= 3 filter applied).
    # Three catalog-wide events dominate (likely large regional earthquakes
    # flooding many templates simultaneously):
    #   2012-12-25  — hits 7 templates (up to 64x median)
    #   2013-02-07  — hits 6 templates (up to 35x median)
    #   2015-03-08  — hits 2 templates (up to 294x median)
    # Per-template spikes:
    #   2020-09-14  — nm60163943, nm60080523 (15–18x)
    #   2023-06-26  — nm60120628 (161x)
    #   2023-08-08  — nm60120628 (13x)
    "nm60081223": ["2015-03-08", "2013-02-14"],
    "nm60351847": ["2015-03-08", "2013-02-14", "2013-02-07"],
    "nm60120628": ["2023-06-26", "2023-08-08"],
    "nm60080523": ["2012-12-25", "2013-02-07", "2013-02-14", "2020-09-14"],
    "us70003tzm": ["2012-12-25", "2013-02-07"],
    "us7000rfpr": ["2012-12-25"],
    "us6000m33c": ["2012-12-25"],
    "us6000pkzk": ["2012-12-25", "2013-02-07"],
    "nm60163943": ["2020-09-14"],
    "tx2024ywip": ["2012-12-25"],
    "us70008ee1": ["2012-12-25"],
    "us6000e1z3": ["2013-02-07"],
    "us6000dy5c": ["2013-02-07"],
}

# Templates whose detections look like noise/artefacts — excluded entirely.
TEMPLATE_EXCLUSIONS: list[str] = [
    "us2000h85v",
    "tx2025qlwgec",
    "us70003tzm",
    "us6000e1q3",
    "tx2023zock",
    "tx2024ywip",
    "tx2024zbdb",
    "tx2024zocv",
    "tx2024yvww",
    "us6000pi49",
    "us70008ee1",
]

# Shot-gather parameters
STACK_PRE = 2.0     # seconds before detect_time
STACK_POST = 15.0   # sanalyzeder detect_time
STACK_FMIN = 1.0
STACK_FMAX = 20.0
TOP_N_TEMPLATES = None    # None = all templates; set to an analyzedto limit
MAX_DET_PER_PLOT = 500    # subsample if a family has more detections
MAX_ALIGN_S = 1.0         # max_analyzedimum CC-alignment shift applied to stack traces (seconds)
ALIGN_WINDOW_S = 2.0      # half-width of the reference window around t=0 used for CC alignment (seconds)
# Preferred channels to use in shot-gather (tried in order).
# Also used as the label in the stack plot title.
PREFERRED_CHANS = [
    "US.NATX.00.BHZ",
    "AG.WLAR.00.HHZ",
    "AG.WLAR.00.HHE",
    "AG.FCAR.00.HHZ",
    "AG.WHAR.00.HHZ",
    "AG.CCAR.00.HHZ",
    "NM.UALR..BHZ",
]


# ── Per-template best-channel lookup ─────────────────────────────────────────
def _load_template_best_chans(csv_path: str | None) -> dict:
    """
    Load selected_template_channels.csv and return a dict mapping
    event_id -> best seed_id (highest median_snr among pass_all==True rows).
    Returns an empty dict if the file is missing or unreadable.
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "pass_all" in df.columns:
            df = df[df["pass_all"].fillna(False)]
        best = (
            df.sort_values("median_snr", ascending=False)
            .groupby("event_id", sort=False)
            .first()
            .reset_index()[["event_id", "seed_id"]]
        )
        mapping = dict(zip(best["event_id"].astype(str), best["seed_id"].astype(str)))
        log.info(f"Loaded best-channel lookup for {len(mapping)} templates from {csv_path}")
        return mapping
    except Exception as exc:
        log.warning(f"Could not load template channel CSV ({exc}); falling back to on-the-fly SNR.")
        return {}


TEMPLATE_BEST_CHAN: dict = _load_template_best_chans(TEMPLATE_CHAN_CSV)


# ── Catalog builder ────────────────────────────────────────────────────────────

def party_to_dataframe(party_path: str) -> pd.DataFrame:
    """
    Read a declustered Party .tgz (read_detection_catalog=False skips per-template
    catalog XML, keeping load time short) and return a flat DataFrame.
    """
    from eqcorrscan import Party

    log.info(f"Reading party: {party_path}")
    party = Party().read(party_path, read_detection_catalog=False)

    records = []
    for fam in party.families:
        tmpl_name = fam.template.name
        for d in fam.detections:
            sta0, cha0 = d.chans[0] if d.chans else (None, None)
            n_stations = len(set(s for s, c in d.chans)) if d.chans else 0
            records.append({
                "template_name": tmpl_name,
                "detect_time":   d.detect_time.datetime,
                "detect_val":    d.detect_val,
                "threshold":     d.threshold,
                "no_chans":      d.no_chans,
                "n_stations":    n_stations,
                "trig_chan":      f"{sta0}.{cha0}" if sta0 else None,
                "id":            d.id,
            })

    if not records:
        log.warning("No detections found in party.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["detect_time"] = pd.to_datetime(df["detect_time"], utc=True)
    df["cc_abs"]      = df["detect_val"].abs()
    df["avg_chan_corr"] = df["cc_abs"] / df["no_chans"].replace(0, np.nan)
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
    """CC ratio histogram + CC ratio vs time scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_hist, ax_scat = axes

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

    fig.suptitle("Detection Quality — CC ratio", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, "02_cc_quality.png"))


# ── Figure 02b: Detection quality vs time ──────────────────────────────────────

def _plot_template_quality(sub_raw: pd.DataFrame, tmpl_name: str, tmpl_dir: str,
                          x0=None, x1=None) -> None:
    """
    Three-panel per-template quality diagnostic in a single column:
      Panel 1: Cumulative detections by no_chans threshold
      Panel 2: Daily avg_chan_corr with IQR band
      Panel 3: Individual detection no_chans vs time (scatter)
    sub_raw should be the UNFILTERED family slice (from df_raw).
    """
    if sub_raw.empty:
        return

    s = sub_raw.copy()
    s["date"] = s["detect_time"].dt.tz_convert(None).dt.normalize()

    thresh_colors = {
        3: "#1b9e77",
        4: "#7570b3",
        5: "#d95f02",
    }

    daily_cc = (
        s.groupby("date")["avg_chan_corr"]
        .agg(
            p10=lambda x: x.quantile(0.10),
            p25=lambda x: x.quantile(0.25),
            p50="median",
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 8), sharex=False,
        gridspec_kw={"height_ratios": [1.1, 1.0, 0.85], "hspace": 0.12},
    )
    ax1, ax2, ax3 = axes
    ax2.sharex(ax1)
    ax3.sharex(ax1)

    # Panel 1: cumulative detections by no_chans threshold
    all_sub = s.sort_values("detect_time")
    for thresh, col in thresh_colors.items():
        grp = all_sub[all_sub["no_chans"] >= thresh]
        if grp.empty:
            continue
        times_t = grp["detect_time"].dt.tz_convert(None).sort_values()
        ax1.step(
            times_t,
            np.arange(1, len(times_t) + 1),
            where="post",
            color=col,
            linewidth=1.5,
            label=f"no_chans ≥ {thresh} (N={len(times_t):,})",
        )
    ax1.set_ylabel("Cumulative detections", fontsize=9)
    ax1.set_title("Cumulative detections by no_chans threshold", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax1.grid(axis="y", lw=0.4, alpha=0.4)

    # Panel 2: daily avg_chan_corr over time
    dates = daily_cc["date"].values
    ax2.fill_between(dates, daily_cc["p10"], daily_cc["p90"],
                     alpha=0.15, color="#1f77b4", label="P10–P90")
    ax2.fill_between(dates, daily_cc["p25"], daily_cc["p75"],
                     alpha=0.35, color="#1f77b4", label="IQR")
    ax2.plot(dates, daily_cc["p50"], color="#1f77b4", lw=1.4, label="Median")
    ax2.set_ylabel("avg_chan_corr", fontsize=9)
    ax2.set_title("Daily avg_chan_corr — median ± IQR / P10–P90", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(axis="y", lw=0.4, alpha=0.4)

    # Panel 3: daily median no_chans as a line
    daily_nc = (
        s.assign(_date=s["detect_time"].dt.tz_convert(None).dt.normalize())
        .groupby("_date")["no_chans"]
        .median()
        .reset_index()
        .rename(columns={"_date": "date"})
    )
    nc_dates = daily_nc["date"].tolist()
    ax3.plot(nc_dates, daily_nc["no_chans"], color="#444444", lw=1.2)
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.set_ylabel("no_chans", fontsize=9)
    ax3.set_title("Daily median no_chans", fontsize=9)
    ax3.grid(axis="y", lw=0.4, alpha=0.4)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax3.set_xlabel("Date", fontsize=9)
    for ax in (ax1, ax2):
        ax.tick_params(labelbottom=False)

    _x0 = x0 if x0 is not None else (pd.Timestamp(daily_cc["date"].min()).to_pydatetime() if len(dates) > 0 else None)
    _x1 = x1 if x1 is not None else (pd.Timestamp(daily_cc["date"].max()).to_pydatetime() if len(dates) > 0 else None)
    if _x0 is not None and _x1 is not None:
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(_x0, _x1)

    fig.suptitle(f"{tmpl_name} — detection quality diagnostics", fontsize=11)
    fig.tight_layout()
    _savefig(fig, os.path.join(tmpl_dir, "quality_vs_time.png"))


# ── Figure 4: Activity heatmap ─────────────────────────────────────────────────

def plot_template_stats(df: pd.DataFrame, out_dir: str) -> None:
    """Ranked template detection counts with active time span."""
    counts = df.groupby("template_name").size().sort_values(ascending=True)
    templates = counts.index.tolist()
    colors = [
        _net_color(df.loc[df["template_name"] == tmpl, "net_prefix"].iloc[0])
        for tmpl in templates
    ]

    spans = df.groupby("template_name")["detect_time"].agg(["min", "max"])

    y = np.arange(len(templates))
    fig, (ax_bar, ax_span) = plt.subplots(1, 2, figsize=(14, max(6, len(templates) * 0.22 + 2)))

    ax_bar.barh(y, counts.values, color=colors, edgecolor="none", alpha=0.85)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(templates, fontsize=7)
    ax_bar.set_xlabel("Detections", fontsize=11)
    ax_bar.set_title("Detections per template", fontsize=11)
    ax_bar.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_bar.grid(axis="x", lw=0.4, alpha=0.4)

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

def _best_chan_from_waveform(
    det_id: str, waveform_dir: str, detect_time,
    fmin: float, fmax: float, pre: float, post: float,
) -> str | None:
    """
    Load a detection waveform file and return the full SEED ID of the channel
    with the highest SNR.  SNR = RMS(0–5 s post detection) / RMS(pre-detection
    noise window), measured after bandpass filtering.  Falls back to None if the
    file is missing or no channel has enough data.
    """
    fpath = os.path.join(waveform_dir, f"{det_id}.mseed")
    if not os.path.exists(fpath):
        return None
    try:
        st = obspy_read(fpath)
    except Exception:
        return None

    detect_utc = UTCDateTime(pd.Timestamp(detect_time).timestamp())
    t0, t1 = detect_utc - pre, detect_utc + post
    best_id, best_snr = None, -1.0
    for tr in st:
        try:
            tr2 = tr.copy()
            tr2.data = tr2.data.astype(float)
            tr2.trim(t0, t1, pad=True, fill_value=0.0)
            tr2.detrend("demean")
            tr2.filter("bandpass", freqmin=fmin, freqmax=fmax,
                       corners=4, zerophase=True)
        except Exception:
            continue
        sr = tr2.stats.sampling_rate
        n_pre = int(pre * sr)
        n_sig = int(min(5.0, post) * sr)
        data = tr2.data
        if n_pre < 5 or len(data) < n_pre + n_sig:
            continue
        noise_rms  = np.sqrt(np.mean(data[:n_pre] ** 2))
        signal_rms = np.sqrt(np.mean(data[n_pre:n_pre + n_sig] ** 2))
        snr = signal_rms / (noise_rms + 1e-12)
        if snr > best_snr:
            best_snr = snr
            best_id = tr2.id
    return best_id


def _pick_trace(st, chan_id: str | None = None):
    """
    Pick the best trace from a Stream.
    Priority: chan_id (full SEED ID) > PREFERRED_CHANS list > first trace.
    """
    # 1. Try the requested channel by full SEED ID
    if chan_id:
        sel = st.select(id=chan_id)
        if sel:
            return sel[0]
    # 2. Try preferred channel list (full SEED IDs)
    for cid in PREFERRED_CHANS:
        sel = st.select(id=cid)
        if sel:
            return sel[0]
    # 3. Fall back to first trace
    return st[0] if len(st) else None


def _load_stack_traces(
    det_ids: list, waveform_dir: str, detect_times: list,
    trig_chans: list | None = None,
) -> tuple:
    """
    Load and trim waveform files for a list of detection IDs.
    Returns (traces_list, times_used, samp_rate, n_samples).
    Each trace is a numpy array of length n_samples.
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

        tr = _pick_trace(st, trig_chan)
        if tr is None:
            continue
        # Enforce strict channel match: if trig_chan was specified but a
        # different channel was returned by fallback, skip this detection.
        # Mixing channels in the same shot gather corrupts the matrix.
        if trig_chan is not None and tr.id != trig_chan:
            log.debug(f"  {det_id}: requested {trig_chan} not found, skipping row")
            continue

        tr = tr.copy()
        tr.data = tr.data.astype(float)  # avoid int32 padding errors
        detect_utc = UTCDateTime(pd.Timestamp(det_time).timestamp())
        t0 = detect_utc - STACK_PRE
        t1 = detect_utc + STACK_POST
        tr = tr.copy().trim(t0, t1, pad=True, fill_value=0.0)
        tr.detrend("demean")
        tr.taper(0.05)
        tr.filter("bandpass", freqmin=STACK_FMIN, freqmax=STACK_FMAX, corners=4, zerophase=True)

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
    try:
        tribe = Tribe().read(tribe_path)
        tribe_dict = {t.name: t for t in tribe.templates}
    except Exception as exc:
        log.warning(f"Cannot load tribe: {exc}; template overlays will be skipped.")
        tribe_dict = {}

    top_templates = df["template_name"].value_counts()
    if top_n is not None:
        top_templates = top_templates.head(top_n)
    top_templates = top_templates.index.tolist()

    for tmpl_name in top_templates:
        sub = df[df["template_name"] == tmpl_name].sort_values("detect_time")
        log.info(f"  Shot gather for {tmpl_name}: {len(sub)} detections")

        # Self-detection: match detect_time to the expected detection time of the
        # template event detecting itself.
        #
        # detect_time = pick_time - prepick  (NOT origin_time — travel time offset
        # means origin_time can be 5–30 s before detect_time).
        # Use the earliest pick in the template's event minus prepick as the
        # expected detect_time; tolerance ±2 s covers sampling/timing jitter.
        # Fall back to highest cc_abs if picks are absent.
        # Done before subsampling so we always search the full family.
        tmpl_obj_pre = tribe_dict.get(tmpl_name)
        _expected_dt = None
        if tmpl_obj_pre is not None:
            try:
                _prepick = tmpl_obj_pre.prepick or 0.0
                _pick_times = [
                    p.time for p in tmpl_obj_pre.event.picks if p.time is not None
                ]
                if _pick_times:
                    _earliest_pick = min(_pick_times)
                    _expected_dt = pd.Timestamp(
                        (_earliest_pick - _prepick).datetime, tz="UTC"
                    )
            except Exception:
                pass

        if _expected_dt is not None:
            _dt = (sub["detect_time"] - _expected_dt).dt.total_seconds().abs()
            _close = sub[_dt <= 2.0]
            if not _close.empty:
                _close = _close.copy()
                _close["_cc_pc"] = _close["cc_abs"] / _close["no_chans"].replace(0, np.nan)
                peak_row = _close.nlargest(1, "_cc_pc").iloc[0]
                log.info(
                    f"    {tmpl_name}: self-detection found "
                    f"(Δt={_dt[_close.index].min():.2f}s, id={peak_row['id']})"
                )
            else:
                peak_row = sub.nlargest(1, "cc_abs").iloc[0]
                log.warning(
                    f"    {tmpl_name}: no detection within 2 s of expected "
                    f"detect_time {_expected_dt.isoformat()[:19]} "
                    f"(earliest pick - prepick={_prepick}s); "
                    f"nearest Δt={_dt.min():.1f}s; using peak cc_abs as reference."
                )
        else:
            peak_row = sub.nlargest(1, "cc_abs").iloc[0]
            log.debug(f"    {tmpl_name}: no picks in tribe event; using peak cc_abs.")

        if max_det is not None and len(sub) > max_det:
            sub = sub.sample(max_det, random_state=42).sort_values("detect_time")
            log.info(f"    Subsampled to {max_det}")

        det_ids = sub["id"].tolist()
        det_times = sub["detect_time"].dt.tz_localize(None).to_numpy()
        cc_ratios = sub["cc_ratio"].values

        # Choose the channel with highest SNR.
        # Priority: pre-computed tribe_analysis CSV > on-the-fly waveform SNR.
        tmpl_obj = tmpl_obj_pre  # already resolved above for origin-time lookup
        best_chan = TEMPLATE_BEST_CHAN.get(tmpl_name)
        if best_chan is not None:
            log.info(f"    {tmpl_name}: best_chan from tribe_analysis CSV: {best_chan}")
        else:
            best_chan = _best_chan_from_waveform(
                peak_row["id"], waveform_dir, peak_row["detect_time"],
                STACK_FMIN, STACK_FMAX, STACK_PRE, STACK_POST,
            )
            if best_chan is None:
                log.debug(f"    Peak-CC waveform missing for {tmpl_name}; using PREFERRED_CHANS fallback")
            else:
                log.info(f"    {tmpl_name}: best_chan from on-the-fly SNR: {best_chan}")
        chan_ids = [best_chan] * len(det_ids)

        traces, times_used, samp_rate, n_samples = _load_stack_traces(
            det_ids, waveform_dir, det_times, chan_ids
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

        # Self-detection trace: load and process the peak-CC detection waveform
        # for the reference panel and CC alignment (avoids zero-padding of template).
        ref_trace = None
        ref_interp = None
        fpath_peak = os.path.join(waveform_dir, f"{peak_row['id']}.mseed")
        if os.path.exists(fpath_peak):
            try:
                st_peak = obspy_read(fpath_peak)
                tr_peak = _pick_trace(st_peak, best_chan)
                if tr_peak is not None:
                    tr_peak = tr_peak.copy()
                    tr_peak.data = tr_peak.data.astype(float)
                    detect_utc_peak = UTCDateTime(pd.Timestamp(peak_row["detect_time"]).timestamp())
                    tr_peak.trim(detect_utc_peak - STACK_PRE,
                                 detect_utc_peak + STACK_POST,
                                 pad=True, fill_value=0.0)
                    tr_peak.detrend("demean")
                    tr_peak.taper(0.05)
                    tr_peak.filter("bandpass", freqmin=STACK_FMIN, freqmax=STACK_FMAX,
                                   corners=4, zerophase=True)
                    rd = tr_peak.data.astype(float)
                    norm = np.max(np.abs(rd))
                    if norm > 0:
                        rd /= norm
                    # Interpolate onto the common t_axis
                    sr_peak = tr_peak.stats.sampling_rate
                    n_peak = len(rd)
                    t_peak = np.linspace(-STACK_PRE, STACK_POST, n_peak)
                    ref_interp = np.interp(t_axis, t_peak, rd)
                    ref_trace = tr_peak
            except Exception as exc:
                log.debug(f"    Cannot load peak-CC trace for {tmpl_name}: {exc}")

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

        # ── Figure: two rows × two columns; col 1 = narrow colorbar for mat row only
        fig_h = max(5, min(n_det * 0.07 + 3, 28))
        mat_h = max(4, n_det * 0.07)
        fig = plt.figure(figsize=(13, fig_h))
        gs = gridspec.GridSpec(
            2, 2,
            width_ratios=[30, 1],
            height_ratios=[1, mat_h],
            hspace=0.05, wspace=0.06,
        )
        ax_tmpl = fig.add_subplot(gs[0, 0])
        ax_mat  = fig.add_subplot(gs[1, 0], sharex=ax_tmpl)
        cax     = fig.add_subplot(gs[1, 1])  # dedicated colorbar column

        # ── Template panel (self-detection trace) ────────────────────────────
        if ref_interp is not None:
            ax_tmpl.fill_between(t_axis, ref_interp, alpha=0.25, color="#d62728")
            ax_tmpl.plot(t_axis, ref_interp, color="#d62728", linewidth=1.0)
        ax_tmpl.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_tmpl.set_ylabel("Template\n(norm.)", fontsize=9)
        ax_tmpl.set_yticks([])
        chan_label = best_chan if best_chan else "?"
        ax_tmpl.set_title(
            f"Shot gather — {tmpl_name}  (N={n_det}, chan: {chan_label})",
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
        ax_mat.set_xlabel("Time relative to detection (s)", fontsize=11)
        ax_mat.set_ylabel("Detection index (chronological)", fontsize=10)
        ax_mat.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

        # Self-detection marker: horizontal line + right-margin tick label
        if peak_row_idx is not None:
            ax_mat.axhline(peak_row_idx + 0.5, color="gold", linewidth=1.5,
                           linestyle="-", alpha=0.9, zorder=5)
            ax_mat.annotate(
                "▶ self",
                xy=(STACK_POST, peak_row_idx + 0.5),
                xycoords="data",
                xytext=(3, 0), textcoords="offset points",
                va="center", ha="left", fontsize=7, color="gold",
                annotation_clip=False,
            )

        # Colourbar in dedicated column — does not borrow space from ax_mat
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Normalised amplitude", fontsize=9)

        # Mean stack overlay on matrix (scaled to row-height units)
        stack_scaled = mean_stack / amp_lim * (n_det * 0.08)   # ~8% of height
        ax_mat.plot(t_axis, n_det / 2 + stack_scaled,
                    color="k", linewidth=0.8, alpha=0.7, label="Mean stack")
        ax_mat.legend(loc="upper right", fontsize=8, framealpha=0.7)

        fig.tight_layout()
        _savefig(fig, os.path.join(out_dir, f"stack_{tmpl_name}.png"))


# ── Summary text report ────────────────────────────────────────────────────────

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
        fh.write(f"Quality filter applied : no_chans >= {MIN_CHANS}\n")
        if SPIKE_DAY_EXCLUSIONS:
            fh.write(f"Spike-day exclusions   : {len(SPIKE_DAY_EXCLUSIONS)} template(s)\n")
            for tmpl, days in sorted(SPIKE_DAY_EXCLUSIONS.items()):
                fh.write(f"  {tmpl}: {', '.join(sorted(days))}\n")
        fh.write(f"no_chans distribution  : {df['no_chans'].value_counts().sort_index().to_dict()}\n")
        fh.write("\nTop 10 templates by detection count:\n")
        for i, (tmpl, n) in enumerate(top10.items(), 1):
            fh.write(f"  {i:2d}. {tmpl:<22} {n:5d}\n")
        fh.write("\nTrigger channel distribution (top 10):\n")
        for chan, n in df["trig_chan"].value_counts().head(10).items():
            fh.write(f"  {chan:<25} {n:5d}\n")
    log.info(f"Summary written to {path}")


# ── Figure 7: Template location map ──────────────────────────────────────────

def plot_template_map(df: pd.DataFrame, tribe_path: str, out_dir: str) -> dict:
    """
    Map of template event locations, sized by filtered detection count.
    Only templates that appear in df (i.e. have ≥1 filtered detection) are shown.

    Dense clusters are detected automatically (DBSCAN, ~30 km radius) and shown
    as inset zoom panels with connecting boxes drawn on the main map.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from sklearn.cluster import DBSCAN

    log.info("Generating template location map …")
    try:
        tribe = Tribe().read(tribe_path)
        tribe_dict = {t.name: t for t in tribe.templates}
    except Exception as exc:
        log.warning(f"Cannot load tribe for map: {exc}")
        return

    det_counts = df["template_name"].value_counts()
    lats, lons, counts, names = [], [], [], []
    for tmpl_name, n in det_counts.items():
        tmpl = tribe_dict.get(tmpl_name)
        if tmpl is None:
            continue
        try:
            orig = tmpl.event.preferred_origin() or tmpl.event.origins[0]
            if orig.latitude is None or orig.longitude is None:
                continue
            lats.append(float(orig.latitude))
            lons.append(float(orig.longitude))
            counts.append(n)
            names.append(tmpl_name)
        except (IndexError, AttributeError):
            pass

    if not lats:
        log.warning("No template event locations found in tribe, skipping map.")
        return

    lats   = np.array(lats)
    lons   = np.array(lons)
    counts = np.array(counts, dtype=float)

    # ── Marker sizes: sqrt-scaled, 30–400 pt² ────────────────────────────────
    sqc = np.sqrt(counts)
    s_norm = (sqc - sqc.min()) / (sqc.max() - sqc.min() + 1e-9)
    s = 30 + 370 * s_norm

    # ── DBSCAN clustering in geographic coordinates ───────────────────────────
    # Use haversine metric on radians; eps = 30 km / Earth radius
    CLUSTER_RADIUS_KM = 30.0
    MIN_CLUSTER_SIZE  = 3
    coords_rad = np.deg2rad(np.column_stack([lats, lons]))
    db = DBSCAN(
        eps=CLUSTER_RADIUS_KM / 6371.0,
        min_samples=MIN_CLUSTER_SIZE,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords_rad)
    labels = db.labels_
    cluster_ids = sorted(set(labels) - {-1})
    log.info(
        f"  Map clustering: {len(cluster_ids)} dense cluster(s) found "
        f"(DBSCAN, r={CLUSTER_RADIUS_KM} km, min={MIN_CLUSTER_SIZE})"
    )

    # ── Main map ──────────────────────────────────────────────────────────────
    crs = ccrs.PlateCarree()
    pad_lat = max(0.5, (lats.max() - lats.min()) * 0.20)
    pad_lon = max(0.5, (lons.max() - lons.min()) * 0.20)
    extent = [
        lons.min() - pad_lon, lons.max() + pad_lon,
        lats.min() - pad_lat, lats.max() + pad_lat,
    ]

    def _add_map_features(ax_m, scale="10m"):
        ax_m.add_feature(cfeature.LAND.with_scale(scale),   facecolor="#f5f5e8", zorder=0)
        ax_m.add_feature(cfeature.OCEAN.with_scale(scale),  facecolor="#ddeeff", zorder=0)
        ax_m.add_feature(cfeature.STATES.with_scale(scale), linewidth=0.6,
                         edgecolor="dimgray", zorder=1)
        ax_m.add_feature(cfeature.RIVERS.with_scale(scale), linewidth=0.4,
                         edgecolor="#99bbff", zorder=1)
        ax_m.add_feature(cfeature.LAKES.with_scale(scale),  facecolor="#ddeeff", zorder=1)
        ax_m.coastlines(scale, linewidth=0.7)

    # Layout constants
    MAX_INSET_PAD  = 0.10  # degrees — cap so insets stay tight
    INSETS_PER_COL = 3    # max insets stacked vertically before adding another column

    # Lay out: main map takes left 60%; insets fill the right side
    # Figure width scales with the number of inset columns needed
    n_clusters  = len(cluster_ids)
    n_cols_pre  = int(np.ceil(n_clusters / INSETS_PER_COL)) if n_clusters else 0
    if n_clusters:
        fig_w = 11 + 4 * n_cols_pre   # 15 for 1 col, 19 for 2 cols, etc.
        fig = plt.figure(figsize=(fig_w, 9))
        ax = fig.add_axes([0.02, 0.06, 0.60, 0.88], projection=crs)
    else:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(1, 1, 1, projection=crs)

    ax.set_extent(extent, crs=crs)
    _add_map_features(ax)

    sc = ax.scatter(
        lons, lats, s=s, c=counts, cmap="hot_r",
        transform=crs, zorder=5,
        edgecolors="k", linewidths=0.5,
        vmin=0, vmax=counts.max(), alpha=0.88,
    )

    for name, lat, lon, n in zip(names, lats, lons, counts):
        ax.text(
            lon, lat, f"{name}\n({int(n)})",
            transform=crs, fontsize=5.5, ha="left", va="bottom",
            clip_on=True,
        )

    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("Filtered detections (n_stations ≥ 3)", fontsize=9)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(
        "Smackover North — Active template event locations\n"
        "(size ∝ √detections, colour = detection count)",
        fontsize=11,
    )

    # ── Cluster insets ────────────────────────────────────────────────────────
    INSET_COLORS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    ]

    for ci, cid in enumerate(cluster_ids):
        mask = labels == cid
        c_lats = lats[mask]
        c_lons = lons[mask]
        c_counts = counts[mask]
        c_names  = [names[i] for i, m in enumerate(mask) if m]
        color = INSET_COLORS[ci % len(INSET_COLORS)]

        # Bounding box on main map — pad = half cluster span, capped at MAX_INSET_PAD
        lon_span = c_lons.max() - c_lons.min()
        lat_span = c_lats.max() - c_lats.min()
        pad_lon_c = min(MAX_INSET_PAD, max(0.02, lon_span * 0.5))
        pad_lat_c = min(MAX_INSET_PAD, max(0.02, lat_span * 0.5))
        bb_lon0 = c_lons.min() - pad_lon_c
        bb_lon1 = c_lons.max() + pad_lon_c
        bb_lat0 = c_lats.min() - pad_lat_c
        bb_lat1 = c_lats.max() + pad_lat_c

        # Draw rectangle on main map
        import matplotlib.patches as mpatches
        rect = mpatches.FancyBboxPatch(
            (bb_lon0, bb_lat0),
            bb_lon1 - bb_lon0, bb_lat1 - bb_lat0,
            boxstyle="square,pad=0",
            linewidth=1.8, edgecolor=color, facecolor="none",
            transform=crs, zorder=8,
        )
        ax.add_patch(rect)
        ax.text(
            (bb_lon0 + bb_lon1) / 2, bb_lat1 + 0.05,
            f"Cluster {ci + 1}",
            transform=crs, fontsize=7.5, color=color,
            ha="center", va="bottom", fontweight="bold",
            zorder=9,
        )

        # Inset axes: grid of INSETS_PER_COL rows × as many columns as needed
        n_insets   = len(cluster_ids)
        n_cols_ins = int(np.ceil(n_insets / INSETS_PER_COL))
        col_w      = 0.33 / n_cols_ins          # each column gets equal share of right 33%
        row_h      = 0.82 / min(n_insets, INSETS_PER_COL)
        ins_col    = ci // INSETS_PER_COL
        ins_row    = ci %  INSETS_PER_COL       # 0 = top row
        ins_left   = 0.65 + ins_col * col_w
        ins_bottom = 0.09 + (INSETS_PER_COL - 1 - ins_row) * row_h
        ax_ins = fig.add_axes(
            [ins_left, ins_bottom, col_w * 0.92, row_h * 0.88],
            projection=crs,
        )
        ins_extent = [bb_lon0, bb_lon1, bb_lat0, bb_lat1]
        ax_ins.set_extent(ins_extent, crs=crs)
        _add_map_features(ax_ins)

        # Rescale marker sizes for the inset (same sqrt scaling, local range)
        sqc_ins = np.sqrt(c_counts)
        sn_ins  = (sqc_ins - sqc_ins.min()) / (sqc_ins.max() - sqc_ins.min() + 1e-9)
        s_ins   = 40 + 300 * sn_ins

        ax_ins.scatter(
            c_lons, c_lats, s=s_ins, c=c_counts, cmap="hot_r",
            transform=crs, zorder=5,
            edgecolors="k", linewidths=0.5,
            vmin=0, vmax=counts.max(), alpha=0.9,
        )
        for name, lat, lon, n in zip(c_names, c_lats, c_lons, c_counts):
            ax_ins.text(
                lon, lat, f"{name}\n({int(n)})",
                transform=crs, fontsize=5.5, ha="left", va="bottom",
                clip_on=True,
            )

        gl_ins = ax_ins.gridlines(
            draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--"
        )
        gl_ins.top_labels = False
        gl_ins.right_labels = False
        gl_ins.xlabel_style = {"size": 6}
        gl_ins.ylabel_style = {"size": 6}

        # Coloured border to match rectangle on main map
        for spine in ax_ins.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
        ax_ins.set_title(f"Cluster {ci + 1}  ({mask.sum()} templates)",
                         fontsize=8, color=color, pad=3)

    _savefig(fig, os.path.join(out_dir, "07_template_map.png"))

    # Return 1-indexed cluster assignments matching the map's "Cluster N" labels.
    # Templates not in any cluster (noise, label=-1) are excluded.
    return {names[i]: int(labels[i]) + 1
            for i in range(len(names)) if labels[i] >= 0}


# ── Per-template: waterfalls for all channels + cumulative step plot ──────────

def plot_per_template_waterfalls(
    df: pd.DataFrame,
    waveform_dir: str,
    tribe_path: str,
    out_dir: str,
    max_det: int = MAX_DET_PER_PLOT,
    df_raw: pd.DataFrame = None,
) -> None:
    """
    For each template with filtered detections:
      - One cumulative step plot saved to per_template/{tmpl}/step_plot.png
        (shows raw family in grey + n_stations>=3 filtered in blue)
      - One shot-gather waterfall per template channel saved to
        per_template/{tmpl}/{NET_STA_LOC_CHA}.png

    Waterfalls use strict channel matching: only detections whose saved
    waveform contains that specific SEED channel are included in the panel.
    """
    log.info(f"Loading tribe for per-channel waterfalls: {tribe_path}")
    try:
        tribe = Tribe().read(tribe_path)
        tribe_dict = {t.name: t for t in tribe.templates}
    except Exception as exc:
        log.warning(f"Cannot load tribe: {exc}")
        return

    active_tmpls = df["template_name"].value_counts()
    n_total = len(active_tmpls)
    log.info(f"Per-template subdirs for {n_total} templates …")

    for i_t, (tmpl_name, n_fam) in enumerate(active_tmpls.items(), 1):
        sub = df[df["template_name"] == tmpl_name].sort_values("detect_time")
        tmpl_obj = tribe_dict.get(tmpl_name)
        if tmpl_obj is None:
            log.warning(f"  [{i_t}/{n_total}] {tmpl_name}: not in tribe, skipping.")
            continue

        tmpl_dir = os.path.join(out_dir, "per_template", tmpl_name)
        os.makedirs(tmpl_dir, exist_ok=True)

        # ── Cumulative step plot — three no_chans thresholds ─────────────────
        _THRESH_COLORS = {
            3: "#1b9e77",   # teal
            4: "#7570b3",   # purple
            5: "#d95f02",   # orange
        }
        fig, ax_step = plt.subplots(figsize=(10, 3.5))
        all_sub = df_raw[df_raw["template_name"] == tmpl_name] if df_raw is not None else sub
        for thresh, col in _THRESH_COLORS.items():
            grp = all_sub[all_sub["no_chans"] >= thresh].sort_values("detect_time")
            if grp.empty:
                continue
            times_t = grp["detect_time"].dt.tz_localize(None).sort_values()
            ax_step.step(
                times_t, np.arange(1, len(times_t) + 1),
                where="post", color=col, linewidth=1.5,
                label=f"no_chans \u2265 {thresh}  (N={len(times_t):,})",
            )
        ax_step.set_xlabel("Date", fontsize=10)
        ax_step.set_ylabel("Cumulative detections", fontsize=10)
        ax_step.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax_step.set_title(f"{tmpl_name} \u2014 cumulative detections by no_chans threshold", fontsize=10)
        ax_step.legend(fontsize=8, loc="upper left")
        ax_step.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_step.xaxis.set_major_locator(mdates.YearLocator())
        ax_step.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        fig.autofmt_xdate(rotation=30, ha="right")
        ax_step.grid(axis="y", linewidth=0.4, alpha=0.4)
        fig.tight_layout()
        _savefig(fig, os.path.join(tmpl_dir, "step_plot.png"))

        # ── cc_ratio / no_chans quality diagnostic ────────────────────────────
        _plot_template_quality(
            df_raw[df_raw["template_name"] == tmpl_name] if df_raw is not None else all_sub,
            tmpl_name, tmpl_dir,
            x0=SCAN_START, x1=SCAN_END,
        )

        # ── Per-channel waterfalls ────────────────────────────────────────────
        chan_ids = [tr.id for tr in tmpl_obj.st]
        log.info(
            f"  [{i_t}/{n_total}] {tmpl_name}: {n_fam} dets, "
            f"{len(chan_ids)} channels"
        )

        det_ids   = sub["id"].tolist()
        det_times = sub["detect_time"].dt.tz_localize(None).to_numpy()
        cc_ratios = sub["cc_ratio"].values

        # Resolve self-detection / reference row once per template
        _expected_dt = None
        try:
            _prepick = tmpl_obj.prepick or 0.0
            _pick_times = [p.time for p in tmpl_obj.event.picks if p.time is not None]
            if _pick_times:
                _expected_dt = pd.Timestamp(
                    (min(_pick_times) - _prepick).datetime, tz="UTC"
                )
        except Exception:
            pass

        if _expected_dt is not None:
            _dt = (sub["detect_time"] - _expected_dt).dt.total_seconds().abs()
            _close = sub[_dt <= 2.0]
            peak_row = (
                _close.nlargest(1, "cc_abs").iloc[0]
                if not _close.empty
                else sub.nlargest(1, "cc_abs").iloc[0]
            )
        else:
            peak_row = sub.nlargest(1, "cc_abs").iloc[0]

        # Subsample if needed (same RNG seed as main stack for consistency)
        if max_det is not None and len(det_ids) > max_det:
            rng = np.random.RandomState(42)
            idx = np.sort(rng.choice(len(det_ids), max_det, replace=False))
            det_ids   = [det_ids[i] for i in idx]
            det_times = det_times[idx]
            cc_ratios = cc_ratios[idx]

        for chan_id in chan_ids:
            chan_fname = chan_id.replace(".", "_") + ".png"
            out_path   = os.path.join(tmpl_dir, chan_fname)

            traces, _times_used, samp_rate, n_samples = _load_stack_traces(
                det_ids, waveform_dir, det_times,
                [chan_id] * len(det_ids),
            )
            if len(traces) < 2:
                log.debug(f"    {tmpl_name}/{chan_id}: <2 traces, skipping.")
                continue

            traces_arr = np.array(traces)
            t_axis     = np.linspace(-STACK_PRE, STACK_POST, n_samples)
            amp_lim    = max(np.percentile(np.abs(traces_arr), 95), 1e-6)
            n_det      = len(traces_arr)
            mean_stack = np.mean(traces_arr, axis=0)

            # Load reference trace for this channel from the peak (self-)detection
            ref_interp = None
            fpath_peak = os.path.join(waveform_dir, f"{peak_row['id']}.mseed")
            if os.path.exists(fpath_peak):
                try:
                    st_ref = obspy_read(fpath_peak)
                    tr_ref = _pick_trace(st_ref, chan_id)
                    if tr_ref is not None and tr_ref.id == chan_id:
                        tr_ref = tr_ref.copy()
                        tr_ref.data = tr_ref.data.astype(float)
                        t_utc = UTCDateTime(pd.Timestamp(peak_row["detect_time"]).timestamp())
                        tr_ref.trim(t_utc - STACK_PRE, t_utc + STACK_POST,
                                    pad=True, fill_value=0.0)
                        tr_ref.detrend("demean")
                        tr_ref.taper(0.05)
                        tr_ref.filter("bandpass", freqmin=STACK_FMIN, freqmax=STACK_FMAX,
                                      corners=4, zerophase=True)
                        rd = tr_ref.data.astype(float)
                        norm = np.max(np.abs(rd))
                        if norm > 0:
                            rd /= norm
                            t_ref = np.linspace(-STACK_PRE, STACK_POST, len(rd))
                            ref_interp = np.interp(t_axis, t_ref, rd)
                except Exception:
                    pass

            # Self-detection row index in the loaded matrix
            peak_row_idx = None
            if peak_row["id"] in det_ids:
                peak_t = pd.Timestamp(peak_row["detect_time"])
                if peak_t.tzinfo is not None:
                    peak_t = peak_t.tz_localize(None)
                for idx2, t in enumerate(_times_used):
                    if abs((pd.Timestamp(t) - peak_t).total_seconds()) < 1.0:
                        peak_row_idx = idx2
                        break

            # ── Figure: 2-row × 2-col  (template trace top, matrix bottom) ──
            fig_h = max(5, min(n_det * 0.07 + 3, 28))
            mat_h = max(4, n_det * 0.07)
            fig   = plt.figure(figsize=(13, fig_h))
            gs    = gridspec.GridSpec(
                2, 2,
                width_ratios=[30, 1],
                height_ratios=[1, mat_h],
                hspace=0.05, wspace=0.06,
            )
            ax_tmpl = fig.add_subplot(gs[0, 0])
            ax_mat  = fig.add_subplot(gs[1, 0], sharex=ax_tmpl)
            cax     = fig.add_subplot(gs[1, 1])

            # Template panel
            if ref_interp is not None:
                ax_tmpl.fill_between(t_axis, ref_interp, alpha=0.25, color="#d62728")
                ax_tmpl.plot(t_axis, ref_interp, color="#d62728", linewidth=1.0)
            ax_tmpl.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_tmpl.set_ylabel("Template\n(norm.)", fontsize=9)
            ax_tmpl.set_yticks([])
            ax_tmpl.set_title(
                f"Shot gather — {tmpl_name}  ·  {chan_id}  (N={n_det})",
                fontsize=11,
            )

            # Matrix panel
            im = ax_mat.imshow(
                traces_arr,
                aspect="auto",
                extent=[-STACK_PRE, STACK_POST, n_det, 0],
                cmap="RdBu_r",
                vmin=-amp_lim, vmax=amp_lim,
                interpolation="nearest",
                origin="upper",
                rasterized=True,
            )
            ax_mat.axvline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_mat.set_xlabel("Time relative to detection (s)", fontsize=11)
            ax_mat.set_ylabel("Detection index (chronological)", fontsize=10)
            ax_mat.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

            # Self-detection marker
            if peak_row_idx is not None:
                ax_mat.axhline(peak_row_idx + 0.5, color="gold", linewidth=1.5,
                               linestyle="-", alpha=0.9, zorder=5)
                ax_mat.annotate(
                    "▶ self",
                    xy=(STACK_POST, peak_row_idx + 0.5),
                    xycoords="data",
                    xytext=(3, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7, color="gold",
                    annotation_clip=False,
                )

            # Mean stack overlay
            stack_scaled = mean_stack / amp_lim * (n_det * 0.08)
            ax_mat.plot(
                t_axis, n_det / 2 + stack_scaled,
                color="k", linewidth=0.8, alpha=0.7, label="Mean stack",
            )
            ax_mat.legend(loc="upper right", fontsize=8, framealpha=0.7)

            cb = fig.colorbar(im, cax=cax)
            cb.set_label("Normalised amplitude", fontsize=9)

            fig.tight_layout()
            _savefig(fig, out_path)

        log.info(f"    {tmpl_name}: per-channel waterfalls done ({i_t}/{n_total})")


# ── Injection vs detections ───────────────────────────────────────────────────

def plot_injection_vs_detections(
    df: pd.DataFrame,
    tribe_path: str,
    swd_mat_path: str,
    out_dir: str,
    radii_km: tuple = (20.0, 10.0, 5.0, 1.0),
    df_raw: pd.DataFrame = None,
) -> None:
    """
    For each template, plot monthly SWD injection rate (barrels/month) summed
    over wells within each radius in `radii_km` as overlaid lines on the same
    injection panel, with cumulative detections below.
    The detection panel shows four no_chans threshold lines (2/3/4/5).
    Saves one PNG per template into out_dir/injection_vs_detections/.
    Also saves a summary figure (08_injection_summary.png) aggregating across
    all templates using the largest radius.
    """
    import scipy.io as sio
    from datetime import datetime, timedelta

    log.info("Generating injection vs. detections plots …")

    # ── Load SWD data ─────────────────────────────────────────────────────────
    if not os.path.exists(swd_mat_path):
        log.warning(f"SWD mat file not found: {swd_mat_path} — skipping injection plots.")
        return
    mat = sio.loadmat(swd_mat_path, squeeze_me=True, struct_as_record=False)
    m = mat["monthly"]

    def _matlab2ts(dn):
        return pd.Timestamp(
            datetime(1, 1, 1) + timedelta(days=float(dn) - 367)
        ).tz_localize("UTC")

    inj_times  = pd.DatetimeIndex([_matlab2ts(d) for d in m.date])  # length T
    well_lats  = m.Latitude   # shape (W,)
    well_lons  = m.Longitude  # shape (W,)
    vol        = m.vol        # shape (T, W)
    vol_clean  = np.where(np.isnan(vol), 0.0, vol)
    # Measured depth in feet per well (static); used for volume-weighted avg depth
    mdft_raw   = getattr(m, "mdft", None)
    mdft_clean = np.where(np.isnan(mdft_raw.astype(float)), np.nan, mdft_raw.astype(float)) \
                 if mdft_raw is not None else None

    def _calc_depth_series(mask):
        """Volume-weighted average injection depth (ft) per time step."""
        if mdft_clean is None or not mask.any():
            return np.full(len(inj_times), np.nan)
        v = vol_clean[:, mask]          # (T, W_r)
        d = mdft_clean[mask]            # (W_r,)
        total_vol = v.sum(axis=1)       # (T,)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_d = np.where(total_vol > 0,
                             (v * d).sum(axis=1) / total_vol,
                             np.nan)
        return avg_d

    # ── Load tribe ────────────────────────────────────────────────────────────
    try:
        tribe = Tribe().read(tribe_path)
        tribe_dict = {t.name: t for t in tribe.templates}
    except Exception as exc:
        log.warning(f"Cannot load tribe for injection plots: {exc}")
        return

    # ── Haversine helper ──────────────────────────────────────────────────────
    def _hav_km(lat1, lon1, lat2_arr, lon2_arr):
        R = 6371.0
        dlat = np.radians(lat2_arr - lat1)
        dlon = np.radians(lon2_arr - lon1)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_arr))
             * np.sin(dlon / 2) ** 2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Categorical colours for radii — visually distinct at all sizes
    RADIUS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]  # Set1: red, blue, green, purple

    out_subdir = os.path.join(out_dir, "injection_vs_detections")
    os.makedirs(out_subdir, exist_ok=True)

    scan_start = pd.Timestamp("2009-02-12", tz="UTC").to_pydatetime()
    scan_end   = pd.Timestamp("2026-03-31", tz="UTC").to_pydatetime()
    inj_times_mpl = inj_times.to_pydatetime()

    templates_in_df = df["template_name"].unique()
    # Track which wells fall within the largest radius of any template (for summary)
    max_radius = max(radii_km)
    all_nearby_mask = np.zeros(len(well_lats), dtype=bool)

    # Three no_chans thresholds for the detection panel
    _THRESH_COLORS = {
        3: "#1b9e77",   # teal
        4: "#7570b3",   # purple
        5: "#d95f02",   # orange
    }

    def _make_figure(ax_inj, ax_depth, ax_det, radius_series, title, det_times, n_det,
                      raw_det_times=None, n_det_raw=0, depth_series=None,
                      df_tmpl=None):
        """Draw injection rate, depth, and cumulative detections onto the given axes."""
        # Panel 1: one line per radius, largest drawn as filled area, rest as lines
        for i, (r, series) in enumerate(zip(radii_km, radius_series)):
            color = RADIUS_COLORS[i % len(RADIUS_COLORS)]
            n_w = (series > 0).any()  # any data at this radius?
            label = f"\u2264{r:.0f} km"
            if i == 0:
                # Largest radius: filled area for visual weight
                ax_inj.fill_between(
                    inj_times_mpl, series / 1e3, alpha=0.18, color=color, step="post",
                )
            ax_inj.step(
                inj_times_mpl, series / 1e3, where="post",
                color=color, lw=1.6 - i * 0.3, alpha=0.92,
                label=label if n_w else f"{label} (no wells)",
            )
        ax_inj.set_ylabel("Injection rate (10\u00b3 bbl/month)", fontsize=9)
        ax_inj.set_title(title, fontsize=10)
        ax_inj.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
        ax_inj.grid(axis="y", lw=0.4, alpha=0.4)

        # Separate depth panel
        _COL_DEPTH = "#7b3294"   # purple
        ax_inj.legend(fontsize=8, loc="upper left", ncol=2)
        if depth_series is not None and ax_depth is not None:
            for i, (r, d_arr) in enumerate(zip(radii_km, depth_series)):
                color = RADIUS_COLORS[i % len(RADIUS_COLORS)]
                valid = ~np.isnan(d_arr)
                if valid.any():
                    ax_depth.step(
                        np.array(inj_times_mpl)[valid], d_arr[valid] / 1000.0,
                        where="post", color=color, lw=1.3, alpha=0.85,
                        label=f"\u2264{r:.0f} km",
                    )
            ax_depth.set_ylabel("Avg depth (10\u00b3 ft)", fontsize=9, color=_COL_DEPTH)
            ax_depth.tick_params(axis="y", colors=_COL_DEPTH)
            ax_depth.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
            ax_depth.legend(fontsize=7, loc="upper left", ncol=2)
            ax_depth.grid(axis="y", lw=0.4, alpha=0.4)
            ax_depth.set_title("Vol-weighted avg injection depth", fontsize=8)

        # Panel 2: four no_chans threshold lines on a single y-axis
        if df_tmpl is not None and not df_tmpl.empty:
            for thresh, col in _THRESH_COLORS.items():
                grp = df_tmpl[df_tmpl["no_chans"] >= thresh].sort_values("detect_time")
                if grp.empty:
                    continue
                t_arr = pd.to_datetime(grp["detect_time"].values)
                ax_det.step(
                    t_arr, np.arange(1, len(t_arr) + 1), where="post",
                    color=col, lw=1.5,
                    label=f"no_chans \u2265 {thresh}  (N={len(t_arr):,})",
                )
        elif n_det > 0:
            # fallback: just the already-filtered series
            ax_det.step(
                det_times, np.arange(1, n_det + 1), where="post",
                color="#1b9e77", lw=1.5, label=f"no_chans \u2265 {MIN_CHANS}  (N={n_det:,})",
            )
        ax_det.set_ylabel("Cumulative detections", fontsize=9)
        ax_det.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax_det.legend(fontsize=8, loc="upper left")
        ax_det.grid(axis="y", lw=0.4, alpha=0.4)

        # Shared formatting
        ax_det.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_det.xaxis.set_major_locator(mdates.YearLocator(5))
        ax_det.xaxis.set_minor_locator(mdates.YearLocator(1))
        plt.setp(ax_det.get_xticklabels(), fontsize=8)
        _span_axes = [ax_inj, ax_det]
        if ax_depth is not None:
            _span_axes.append(ax_depth)
        for ax in _span_axes:
            ax.axvspan(scan_start, scan_end, color="gold", alpha=0.12, zorder=0)

    for tmpl_name in sorted(templates_in_df):
        tmpl = tribe_dict.get(tmpl_name)
        if tmpl is None:
            continue
        try:
            orig = tmpl.event.preferred_origin() or tmpl.event.origins[0]
            if orig.latitude is None or orig.longitude is None:
                continue
            t_lat = float(orig.latitude)
            t_lon = float(orig.longitude)
        except (IndexError, AttributeError):
            continue

        dists = _hav_km(t_lat, t_lon, well_lats, well_lons)

        # Build one series per radius (volume and depth)
        radius_series = []
        depth_series  = []
        any_wells = False
        for r in radii_km:
            mask = dists <= r
            if r == max_radius:
                all_nearby_mask |= mask
            series = vol_clean[:, mask].sum(axis=1) if mask.any() else np.zeros(len(inj_times))
            radius_series.append(series)
            depth_series.append(_calc_depth_series(mask))
            if mask.any():
                any_wells = True

        if not any_wells:
            log.info(f"  {tmpl_name}: no injection wells within {max_radius} km — skipping.")
            continue

        n_wells_max = (dists <= max_radius).sum()
        sub = df[df["template_name"] == tmpl_name].sort_values("detect_time")
        det_times = pd.to_datetime(sub["detect_time"].values)
        n_det = len(sub)
        log.info(f"  {tmpl_name}: {n_wells_max} wells within {max_radius} km, {n_det} detections")

        # Build per-template slice of the full catalog for multi-threshold panel
        df_tmpl = df_raw[df_raw["template_name"] == tmpl_name] if df_raw is not None else None

        fig, (ax_inj, ax_depth_p, ax_det) = plt.subplots(
            3, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 2], "hspace": 0.08},
        )
        title = (
            f"{tmpl_name}  —  SWD injection at multiple radii  "
            f"(≤{max_radius:.0f} km: {n_wells_max} well{'s' if n_wells_max != 1 else ''})"
        )
        _make_figure(ax_inj, ax_depth_p, ax_det, radius_series, title, det_times, n_det,
                     depth_series=depth_series, df_tmpl=df_tmpl)
        _savefig(fig, os.path.join(out_subdir, f"{tmpl_name}_injection.png"))

    # ── Summary: total injection near any template (largest radius) ───────────
    n_union = all_nearby_mask.sum()
    if n_union > 0:
        # Build per-radius series summed over the UNION of all-template wells
        # (for the summary we recompute with the union mask, then tighten per radius
        #  by also computing per-radius globally across all template distances)
        # Simpler: recompute dists from every template, keep the min distance per well
        min_dists = np.full(len(well_lats), np.inf)
        for tmpl_name in sorted(templates_in_df):
            tmpl = tribe_dict.get(tmpl_name)
            if tmpl is None:
                continue
            try:
                orig = tmpl.event.preferred_origin() or tmpl.event.origins[0]
                if orig.latitude is None or orig.longitude is None:
                    continue
                d = _hav_km(float(orig.latitude), float(orig.longitude), well_lats, well_lons)
                min_dists = np.minimum(min_dists, d)
            except (IndexError, AttributeError):
                continue

        summary_series = []
        summary_depths = []
        for r in radii_km:
            mask = min_dists <= r
            series = vol_clean[:, mask].sum(axis=1) if mask.any() else np.zeros(len(inj_times))
            summary_series.append(series)
            summary_depths.append(_calc_depth_series(mask))

        all_det = df.sort_values("detect_time")
        all_det_times = pd.to_datetime(all_det["detect_time"].values)
        n_det_total = len(all_det)

        fig, (ax_inj, ax_depth_p, ax_det) = plt.subplots(
            3, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 2], "hspace": 0.08},
        )
        title = (
            f"All templates — SWD injection at multiple radii  "
            f"(≤{max_radius:.0f} km: {n_union} wells near any template)"
        )
        # Pass full catalog so the panel can draw all three thresholds
        df_summary = df_raw if df_raw is not None else df
        _make_figure(ax_inj, ax_depth_p, ax_det, summary_series, title, all_det_times, n_det_total,
                     depth_series=summary_depths, df_tmpl=df_summary)
        _savefig(fig, os.path.join(out_dir, "08_injection_summary.png"))

    log.info(f"  Injection plots saved to {out_subdir}")


# ── Injection vs detections — per DBSCAN cluster ─────────────────────────────

def plot_injection_vs_detections_by_cluster(
    df: pd.DataFrame,
    tribe_path: str,
    swd_mat_path: str,
    out_dir: str,
    cluster_radius_km: float = 30.0,
    min_cluster_size: int = 3,
    radii_km: tuple = (20.0, 10.0, 5.0, 1.0),
    df_raw: pd.DataFrame = None,
    cluster_map: dict = None,
) -> None:
    """
    One injection-vs-detections figure per DBSCAN spatial cluster of templates.
    Uses the same radii, SWD data, and 3-panel layout as
    plot_injection_vs_detections, but pools all templates within each cluster.
    Injection wells are selected by minimum distance to any cluster template.
    Saves to out_dir/cluster_injection/cluster_{N:02d}_injection.png.
    """
    from sklearn.cluster import DBSCAN
    import scipy.io as sio
    from datetime import datetime, timedelta

    log.info("Generating per-cluster injection vs. detections plots …")

    # ── Load SWD data ─────────────────────────────────────────────────────────
    if not os.path.exists(swd_mat_path):
        log.warning(f"SWD mat file not found: {swd_mat_path} — skipping cluster injection plots.")
        return
    mat = sio.loadmat(swd_mat_path, squeeze_me=True, struct_as_record=False)
    m = mat["monthly"]

    def _matlab2ts(dn):
        return pd.Timestamp(
            datetime(1, 1, 1) + timedelta(days=float(dn) - 367)
        ).tz_localize("UTC")

    inj_times  = pd.DatetimeIndex([_matlab2ts(d) for d in m.date])
    well_lats  = m.Latitude
    well_lons  = m.Longitude
    vol        = m.vol
    vol_clean  = np.where(np.isnan(vol), 0.0, vol)
    mdft_raw   = getattr(m, "mdft", None)
    mdft_clean = np.where(np.isnan(mdft_raw.astype(float)), np.nan, mdft_raw.astype(float)) \
                 if mdft_raw is not None else None

    def _calc_depth_series(mask):
        if mdft_clean is None or not mask.any():
            return np.full(len(inj_times), np.nan)
        v = vol_clean[:, mask]
        d = mdft_clean[mask]
        total_vol = v.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_d = np.where(total_vol > 0, (v * d).sum(axis=1) / total_vol, np.nan)
        return avg_d

    # ── Load tribe ────────────────────────────────────────────────────────────
    try:
        tribe = Tribe().read(tribe_path)
        tribe_dict = {t.name: t for t in tribe.templates}
    except Exception as exc:
        log.warning(f"Cannot load tribe for cluster injection plots: {exc}")
        return

    def _hav_km(lat1, lon1, lat2_arr, lon2_arr):
        R = 6371.0
        dlat = np.radians(lat2_arr - lat1)
        dlon = np.radians(lon2_arr - lon1)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_arr))
             * np.sin(dlon / 2) ** 2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # ── Collect template locations — same order as plot_template_map (descending
    #    detection count) so DBSCAN labels match the map insets exactly.
    templates_in_df = df["template_name"].value_counts().index
    lats_list, lons_list, names_list = [], [], []
    for tmpl_name in templates_in_df:
        tmpl = tribe_dict.get(tmpl_name)
        if tmpl is None:
            continue
        try:
            orig = tmpl.event.preferred_origin() or tmpl.event.origins[0]
            if orig.latitude is None or orig.longitude is None:
                continue
            lats_list.append(float(orig.latitude))
            lons_list.append(float(orig.longitude))
            names_list.append(tmpl_name)
        except (IndexError, AttributeError):
            pass

    if not lats_list:
        log.warning("No template locations found — skipping cluster injection plots.")
        return

    lats_arr  = np.array(lats_list)
    lons_arr  = np.array(lons_list)
    names_arr = np.array(names_list)

    # ── Cluster assignments ───────────────────────────────────────────────────
    if cluster_map is not None:
        # Use pre-computed 1-indexed assignments from plot_template_map so that
        # cluster numbers match the map insets exactly.
        labels = np.array([cluster_map.get(n, -1) for n in names_list])
        cluster_ids = sorted(set(labels) - {-1})
        log.info(
            f"  Cluster injection: using pre-computed cluster map — "
            f"{len(cluster_ids)} cluster(s); "
            f"{(labels == -1).sum()} noise templates"
        )
    else:
        # Fallback: run DBSCAN independently
        coords_rad = np.deg2rad(np.column_stack([lats_arr, lons_arr]))
        db = DBSCAN(
            eps=cluster_radius_km / 6371.0,
            min_samples=min_cluster_size,
            algorithm="ball_tree",
            metric="haversine",
        ).fit(coords_rad)
        labels = db.labels_
        cluster_ids = sorted(set(labels) - {-1})
        log.info(
            f"  Cluster injection: {len(cluster_ids)} cluster(s) found "
            f"(DBSCAN, r={cluster_radius_km} km, min={min_cluster_size}); "
            f"{(labels == -1).sum()} noise templates"
        )
    if not cluster_ids:
        log.info("  No dense clusters found — skipping cluster injection plots.")
        return

    max_radius = max(radii_km)
    RADIUS_COLORS  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    _THRESH_COLORS = {3: "#1b9e77", 4: "#7570b3", 5: "#d95f02"}
    _COL_DEPTH     = "#7b3294"

    scan_start    = pd.Timestamp("2009-02-12", tz="UTC").to_pydatetime()
    scan_end      = pd.Timestamp("2026-03-31", tz="UTC").to_pydatetime()
    inj_times_mpl = inj_times.to_pydatetime()

    out_subdir = os.path.join(out_dir, "cluster_injection")
    os.makedirs(out_subdir, exist_ok=True)

    for cid in cluster_ids:
        clust_names = names_arr[labels == cid]
        n_templates = len(clust_names)
        log.info(f"  Cluster {cid}: {n_templates} templates — {list(clust_names)}")

        # Minimum distance from each well to any template in this cluster
        min_dists = np.full(len(well_lats), np.inf)
        for tmpl_name in clust_names:
            idx = np.where(names_arr == tmpl_name)[0]
            if not idx.size:
                continue
            d = _hav_km(lats_arr[idx[0]], lons_arr[idx[0]], well_lats, well_lons)
            min_dists = np.minimum(min_dists, d)

        if not (min_dists <= max_radius).any():
            log.info(f"  Cluster {cid}: no wells within {max_radius} km — skipping.")
            continue

        n_wells_max = int((min_dists <= max_radius).sum())

        radius_series, depth_series = [], []
        for r in radii_km:
            mask   = min_dists <= r
            series = vol_clean[:, mask].sum(axis=1) if mask.any() else np.zeros(len(inj_times))
            radius_series.append(series)
            depth_series.append(_calc_depth_series(mask))

        # Pool detections from all cluster templates
        df_clust_raw = (
            df_raw[df_raw["template_name"].isin(clust_names)].copy()
            if df_raw is not None else None
        )
        df_clust = df[df["template_name"].isin(clust_names)].sort_values("detect_time")
        n_det    = len(df_clust)

        # Build title — abbreviate template list if more than 5
        if n_templates <= 5:
            tmpl_str = ", ".join(clust_names)
        else:
            tmpl_str = ", ".join(clust_names[:5]) + f" … +{n_templates - 5} more"
        title = f"Cluster {cid}"

        fig, (ax_inj, ax_depth, ax_det) = plt.subplots(
            3, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 2], "hspace": 0.08},
        )

        # Panel 1: injection rate lines per radius
        for i, (r, series) in enumerate(zip(radii_km, radius_series)):
            color = RADIUS_COLORS[i % len(RADIUS_COLORS)]
            label = f"\u2264{r:.0f} km"
            if i == 0:
                ax_inj.fill_between(
                    inj_times_mpl, series / 1e3, alpha=0.18, color=color, step="post",
                )
            ax_inj.step(
                inj_times_mpl, series / 1e3, where="post",
                color=color, lw=1.6 - i * 0.3, alpha=0.92, label=label,
            )
        ax_inj.set_ylabel("Injection rate (10\u00b3 bbl/month)", fontsize=9)
        ax_inj.set_title(title, fontsize=9)
        ax_inj.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
        ax_inj.grid(axis="y", lw=0.4, alpha=0.4)
        ax_inj.legend(fontsize=8, loc="upper left", ncol=2)

        # Panel 2: volume-weighted average injection depth
        for i, (r, d_arr) in enumerate(zip(radii_km, depth_series)):
            color = RADIUS_COLORS[i % len(RADIUS_COLORS)]
            valid = ~np.isnan(d_arr)
            if valid.any():
                ax_depth.step(
                    np.array(inj_times_mpl)[valid], d_arr[valid] / 1000.0,
                    where="post", color=color, lw=1.3, alpha=0.85,
                    label=f"\u2264{r:.0f} km",
                )
        ax_depth.set_ylabel("Avg depth (10\u00b3 ft)", fontsize=9, color=_COL_DEPTH)
        ax_depth.tick_params(axis="y", colors=_COL_DEPTH)
        ax_depth.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
        ax_depth.legend(fontsize=7, loc="upper left", ncol=2)
        ax_depth.grid(axis="y", lw=0.4, alpha=0.4)
        ax_depth.set_title("Vol-weighted avg injection depth", fontsize=8)

        # Panel 3: cumulative detections by no_chans threshold
        df_for_thresh = df_clust_raw if df_clust_raw is not None else df_clust
        for thresh, col in _THRESH_COLORS.items():
            grp = df_for_thresh[df_for_thresh["no_chans"] >= thresh].sort_values("detect_time")
            if grp.empty:
                continue
            t_arr = pd.to_datetime(grp["detect_time"].values)
            ax_det.step(
                t_arr, np.arange(1, len(t_arr) + 1), where="post",
                color=col, lw=1.5,
                label=f"no_chans \u2265 {thresh}  (N={len(t_arr):,})",
            )
        ax_det.set_ylabel("Cumulative detections", fontsize=9)
        ax_det.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax_det.legend(fontsize=8, loc="upper left")
        ax_det.grid(axis="y", lw=0.4, alpha=0.4)

        # Shared x-axis formatting
        ax_det.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_det.xaxis.set_major_locator(mdates.YearLocator(5))
        ax_det.xaxis.set_minor_locator(mdates.YearLocator(1))
        plt.setp(ax_det.get_xticklabels(), fontsize=8)
        for ax in [ax_inj, ax_depth, ax_det]:
            ax.axvspan(scan_start, scan_end, color="gold", alpha=0.12, zorder=0)

        _savefig(fig, os.path.join(out_subdir, f"cluster_{cid:02d}_injection.png"))
        log.info(
            f"  Cluster {cid}: saved — {n_templates} templates, "
            f"{n_det:,} pooled detections, {n_wells_max} wells within {max_radius:.0f} km"
        )

    log.info(f"  Cluster injection plots saved to {out_subdir}")


# ── Per-well injection vs detections — per DBSCAN cluster ─────────────────────

def plot_per_well_injection_by_cluster(
    df: pd.DataFrame,
    tribe_path: str,
    swd_mat_path: str,
    out_dir: str,
    max_radius_km: float = 20.0,
    df_raw: pd.DataFrame = None,
    cluster_map: dict = None,
) -> None:
    """
    For each DBSCAN cluster (identified via cluster_map from plot_template_map),
    produce a 2-panel figure:
      Panel 1: individual SWD well injection rates as separate lines, coloured
               by distance from the nearest cluster template. Well names annotate
               the closest N wells. Wells sorted nearest-first.
      Panel 2: cumulative detections by no_chans threshold.

    Only wells within max_radius_km of any template in the cluster are shown.
    Saves to out_dir/cluster_injection/cluster_{N:02d}_per_well.png.
    """
    import scipy.io as sio
    from datetime import datetime, timedelta
    from itertools import chain

    log.info("Generating per-well injection plots by cluster …")

    if cluster_map is None:
        log.warning("No cluster_map supplied — skipping per-well injection plots.")
        return
    if not os.path.exists(swd_mat_path):
        log.warning(f"SWD mat file not found: {swd_mat_path} — skipping per-well plots.")
        return

    mat = sio.loadmat(swd_mat_path, squeeze_me=True, struct_as_record=False)
    m = mat["monthly"]

    def _matlab2ts(dn):
        return pd.Timestamp(
            datetime(1, 1, 1) + timedelta(days=float(dn) - 367)
        ).tz_localize("UTC")

    inj_times     = pd.DatetimeIndex([_matlab2ts(d) for d in m.date])
    inj_times_mpl = inj_times.to_pydatetime()
    well_lats     = m.Latitude
    well_lons     = m.Longitude
    vol           = m.vol
    vol_clean     = np.where(np.isnan(vol), 0.0, vol)
    well_names    = np.array([str(n) for n in m.wellname])
    well_apis     = np.array([str(a) for a in m.API])
    well_counties = np.array([str(c) for c in m.county])
    mdft_raw      = getattr(m, "mdft", None)
    mdft_clean    = np.where(
        np.isnan(mdft_raw.astype(float)), np.nan, mdft_raw.astype(float)
    ) if mdft_raw is not None else None

    # Load tribe for template locations
    try:
        tribe = Tribe().read(tribe_path)
        tribe_dict = {t.name: t for t in tribe.templates}
    except Exception as exc:
        log.warning(f"Cannot load tribe for per-well plots: {exc}")
        return

    def _hav_km(lat1, lon1, lat2_arr, lon2_arr):
        R = 6371.0
        dlat = np.radians(lat2_arr - lat1)
        dlon = np.radians(lon2_arr - lon1)
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2_arr))
             * np.sin(dlon / 2) ** 2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Group templates by cluster
    from collections import defaultdict
    clusters: dict[int, list[str]] = defaultdict(list)
    for tmpl_name, cid in cluster_map.items():
        clusters[cid].append(tmpl_name)

    scan_start = pd.Timestamp("2009-02-12", tz="UTC").to_pydatetime()
    scan_end   = pd.Timestamp("2026-03-31", tz="UTC").to_pydatetime()
    _THRESH_COLORS = {3: "#1b9e77", 4: "#7570b3", 5: "#d95f02"}

    out_subdir = os.path.join(out_dir, "cluster_injection")
    os.makedirs(out_subdir, exist_ok=True)

    for cid in sorted(clusters.keys()):
        clust_names = clusters[cid]

        # Minimum distance from each well to any template in this cluster
        min_dists = np.full(len(well_lats), np.inf)
        for tmpl_name in clust_names:
            tmpl = tribe_dict.get(tmpl_name)
            if tmpl is None:
                continue
            try:
                orig = tmpl.event.preferred_origin() or tmpl.event.origins[0]
                if orig.latitude is None or orig.longitude is None:
                    continue
                d = _hav_km(float(orig.latitude), float(orig.longitude),
                             well_lats, well_lons)
                min_dists = np.minimum(min_dists, d)
            except (IndexError, AttributeError):
                continue

        near_idx = np.where(min_dists <= max_radius_km)[0]
        if near_idx.size == 0:
            log.info(f"  Cluster {cid}: no wells within {max_radius_km} km — skipping per-well.")
            continue

        # Sort by distance (nearest first)
        near_idx = near_idx[np.argsort(min_dists[near_idx])]
        n_wells  = len(near_idx)
        log.info(f"  Cluster {cid}: {n_wells} wells within {max_radius_km} km")

        # Build per-well injection series
        well_series = []
        for wi in near_idx:
            well_series.append(vol_clean[:, wi])

        # Pool detections for this cluster
        df_for_thresh = (
            df_raw[df_raw["template_name"].isin(clust_names)]
            if df_raw is not None
            else df[df["template_name"].isin(clust_names)]
        )

        # ── Figure — one row per well + detection panel at bottom ────────────
        n_rows   = n_wells + 1
        _well_h  = 1.6
        row_h    = [_well_h] * n_wells + [_well_h * 3]
        fig, axes = plt.subplots(
            n_rows, 1, figsize=(14, _well_h * n_wells + _well_h * 3 + 1.0), sharex=True,
            gridspec_kw={"height_ratios": row_h, "hspace": 0.08},
        )
        well_axes = axes[:n_wells]
        ax_det    = axes[-1]

        # Categorical colours — tab20 + tab20b gives 40 distinct hues
        _palette = list(chain(
            [plt.get_cmap("tab20")(i) for i in range(20)],
            [plt.get_cmap("tab20b")(i) for i in range(20)],
        ))
        well_colors = [_palette[rank % len(_palette)] for rank in range(n_wells)]

        for rank, (wi, series, ax_w) in enumerate(zip(near_idx, well_series, well_axes)):
            dist  = min_dists[wi]
            color = well_colors[rank]
            wname = well_names[wi] if well_names[wi] not in ("", "nan") else f"API {well_apis[wi]}"
            depth_str = (
                f"  {mdft_clean[wi]/1000:.1f}k ft"
                if mdft_clean is not None and not np.isnan(mdft_clean[wi])
                else ""
            )

            ax_w.plot(inj_times_mpl, series / 1e3, color=color, lw=1.4)
            ax_w.set_ylabel("10³ bbl\n/mo", fontsize=7, labelpad=2)
            ax_w.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=False))
            ax_w.tick_params(axis="y", labelsize=7)
            ax_w.grid(axis="y", lw=0.4, alpha=0.4)
            ax_w.axvspan(scan_start, scan_end, color="gold", alpha=0.12, zorder=0)

            # Well label on the right margin
            ax_w.text(
                1.002, 0.5, f"{wname}\n({dist:.1f} km{depth_str})",
                transform=ax_w.transAxes, fontsize=7, va="center",
                color=color, clip_on=False,
            )

        # Super-title for the injection block
        well_axes[0].set_title(
            f"Cluster {cid}  —  individual SWD wells within {max_radius_km:.0f} km  "
            f"({n_wells} wells,  {len(clust_names)} templates)",
            fontsize=10,
        )

        # Detection panel
        for thresh, col in _THRESH_COLORS.items():
            grp = df_for_thresh[df_for_thresh["no_chans"] >= thresh].sort_values("detect_time")
            if grp.empty:
                continue
            t_arr = pd.to_datetime(grp["detect_time"].values)
            ax_det.step(
                t_arr, np.arange(1, len(t_arr) + 1), where="post",
                color=col, lw=1.5,
                label=f"no_chans ≥ {thresh}  (N={len(t_arr):,})",
            )
        ax_det.set_ylabel("Cumulative detections", fontsize=9)
        ax_det.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax_det.legend(fontsize=8, loc="upper left")
        ax_det.grid(axis="y", lw=0.4, alpha=0.4)
        ax_det.axvspan(scan_start, scan_end, color="gold", alpha=0.12, zorder=0)

        ax_det.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_det.xaxis.set_major_locator(mdates.YearLocator(5))
        ax_det.xaxis.set_minor_locator(mdates.YearLocator(1))
        plt.setp(ax_det.get_xticklabels(), fontsize=8)

        fig.tight_layout()
        _savefig(fig, os.path.join(out_subdir, f"cluster_{cid:02d}_per_well.png"))
        log.info(f"  Cluster {cid}: per-well figure saved.")

    log.info(f"  Per-well cluster injection plots saved to {out_subdir}")


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
        "--no-per-template", action="store_true",
        help="Skip per-channel waterfall plots and per-template step plots"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──── Load declustered party ──────────────────────────────────────────────
    df = party_to_dataframe(PARTY_PATH)
    if df.empty:
        log.error("Empty catalog — nothing to plot.")
        return

    # ──── Whole-template exclusions ──────────────────────────────────────────────
    if TEMPLATE_EXCLUSIONS:
        n_before_tmpl = len(df)
        df = df[~df["template_name"].isin(TEMPLATE_EXCLUSIONS)].copy()
        log.info(
            f"Template exclusions: dropped {n_before_tmpl - len(df):,} detections "
            f"for {len(TEMPLATE_EXCLUSIONS)} excluded template(s): {TEMPLATE_EXCLUSIONS}"
        )

    # ──── Spike-day exclusions (applied before everything, including df_raw) ────
    if SPIKE_DAY_EXCLUSIONS:
        n_before_spike = len(df)
        date_col = df["detect_time"].dt.tz_convert(None).dt.normalize()
        mask_keep = pd.Series(True, index=df.index)
        for tmpl, bad_days in SPIKE_DAY_EXCLUSIONS.items():
            bad_ts = pd.to_datetime(bad_days).normalize()
            in_tmpl = df["template_name"] == tmpl
            on_bad_day = date_col.isin(bad_ts)
            mask_keep &= ~(in_tmpl & on_bad_day)
        df = df[mask_keep].copy()
        n_dropped_spike = n_before_spike - len(df)
        log.info(
            f"Spike-day exclusions: dropped {n_dropped_spike:,} detections "
            f"({n_before_spike:,} → {len(df):,}) across "
            f"{len(SPIKE_DAY_EXCLUSIONS)} template(s)"
        )
    else:
        log.info("No spike-day exclusions configured (SPIKE_DAY_EXCLUSIONS is empty).")

    # ──── Quality filter: require at least MIN_CHANS channels correlated ────────
    df_raw = df.copy()   # full catalog — used for the multi-threshold (≥2/3/4/5) panels
    n_before = len(df)
    df = df[df["no_chans"] >= MIN_CHANS].copy()
    n_dropped = n_before - len(df)
    log.info(
        f"no_chans >= {MIN_CHANS} filter: kept {len(df):,} / {n_before:,} detections "
        f"(dropped {n_dropped:,})"
    )

    # ──── Copy individual detection plots for filtered detections ────────────
    src_plot_dir = (
        "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
        "/plots/smackover_north_analyzed/MAD12_2hr"
    )
    det_plot_dest = os.path.join(OUTPUT_DIR, "detection_plots")
    os.makedirs(det_plot_dest, exist_ok=True)
    copied, missing = 0, 0
    for det_id in df["id"]:
        src = os.path.join(src_plot_dir, f"{det_id}.png")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(det_plot_dest, f"{det_id}.png"))
            copied += 1
        else:
            missing += 1
    log.info(f"Detection plots copied: {copied} (missing: {missing}) → {det_plot_dest}")

    # ──── Statistical plots ───────────────────────────────────────────────────
    log.info("Generating statistical plots …")
    plot_temporal_overview(df, OUTPUT_DIR)
    plot_cc_quality(df, OUTPUT_DIR)
    plot_template_stats(df, OUTPUT_DIR)
    plot_activity_heatmap(df, OUTPUT_DIR)
    plot_interevent_times(df, OUTPUT_DIR)
    plot_daily_patterns(df, OUTPUT_DIR)
    write_summary(df, OUTPUT_DIR)

    # ──── Template location map ─────────────────────────────────────────────
    # Returns 1-indexed cluster assignments (matching the map's "Cluster N" labels)
    cluster_map = plot_template_map(df, TRIBE_PATH, OUTPUT_DIR)

    # ──── Injection vs detections ─────────────────────────────────────────────
    plot_injection_vs_detections(
        df, TRIBE_PATH, SWD_MAT_PATH, OUTPUT_DIR,
        radii_km=(20.0, 10.0, 5.0, 1.0), df_raw=df_raw,
    )

    # ──── Per-cluster injection vs detections ─────────────────────────────────
    plot_injection_vs_detections_by_cluster(
        df, TRIBE_PATH, SWD_MAT_PATH, OUTPUT_DIR,
        cluster_radius_km=30.0, min_cluster_size=3,
        radii_km=(20.0, 10.0, 5.0, 2.0, 1.0), df_raw=df_raw,
        cluster_map=cluster_map,
    )

    # ──── Per-well injection vs detections (individual wells per cluster) ──────
    plot_per_well_injection_by_cluster(
        df, TRIBE_PATH, SWD_MAT_PATH, OUTPUT_DIR,
        max_radius_km=3.0, df_raw=df_raw,
        cluster_map=cluster_map,
    )

    # ──── Per-template step plots + per-channel waterfalls ────────────────────
    if not args.no_per_template:
        log.info("Generating per-template step plots and per-channel waterfalls …")
        plot_per_template_waterfalls(
            df, WAVEFORM_DIR, TRIBE_PATH, OUTPUT_DIR,
            max_det=args.max_det, df_raw=df_raw,
        )
    else:
        log.info("Skipping per-template plots (--no-per-template).")

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
