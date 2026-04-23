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

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.cm as cm
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
OUTPUT_DIR = (
    "/media/chopp/HDD1/chet-meq/smackover/detections/lawrencium"
    "/assessment_plots_analyzed"
)

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

        # Choose the channel with highest SNR using the self-detection waveform
        # (has a proper noise window before detection time; template pre-event is too short)
        tmpl_obj = tmpl_obj_pre  # already resolved above for origin-time lookup
        best_chan = _best_chan_from_waveform(
            peak_row["id"], waveform_dir, peak_row["detect_time"],
            STACK_FMIN, STACK_FMAX, STACK_PRE, STACK_POST,
        )
        if best_chan is None:
            log.debug(f"    Peak-CC waveform missing for {tmpl_name}; using PREFERRED_CHANS fallback")
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
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──── Load declustered party ──────────────────────────────────────────────
    df = party_to_dataframe(PARTY_PATH)
    if df.empty:
        log.error("Empty catalog — nothing to plot.")
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
