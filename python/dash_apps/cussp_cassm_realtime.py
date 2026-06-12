"""CUSSP CASSM real-time dashboard (Panel + Plotly).

This app migrates the main MATLAB AppDesigner workflow in
FSB_CASSM_RealTime to Python for CUSSP-style web deployment:

1) Build an initial temp-gather from epoch folders.
2) Append only new epochs during manual or automatic updates.
3) Compute and plot per source/receiver metrics:
   - RMS amplitude
   - Centroid frequency
   - Relative delay time (microseconds)
4) Save/load a temp-gather cache to avoid repeated SEG2 parsing.

Data model assumptions
----------------------
- Epoch folders are direct children of a data directory and are sorted by
  folder name (typically timestamp strings like YYYYmmddHHMMSS).
- Each epoch folder can contain:
  - SEG2 files (*.dat, *.seg2), one file per source shot, or
  - a precomputed epoch NPZ with key "data" and shape
    (n_sources, n_receivers, n_samples).

The SEG2 reader path intentionally stays lightweight: ObsPy can parse SEG2
without custom converters for most CASSM datasets.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import panel as pn
import pandas as pd
import plotly.graph_objects as go

pn.extension("plotly")

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


Pair = Tuple[int, int]


def _safe_parse_epoch_time(name: str) -> pd.Timestamp:
    """Parse timestamp-like folder names with a fallback to lexical order only."""
    try:
        if len(name) == 14 and name.isdigit():
            return pd.to_datetime(name, format="%Y%m%d%H%M%S", utc=True)
        if len(name) == 12 and name.isdigit():
            return pd.to_datetime(name, format="%Y%m%d%H%M", utc=True)
        ts = pd.to_datetime(name, utc=True, errors="coerce")
        if pd.isna(ts):
            return pd.Timestamp("1970-01-01", tz="UTC")
        return ts
    except Exception:
        return pd.Timestamp("1970-01-01", tz="UTC")


def _build_pair_index(n_sources: int, n_receivers: int) -> List[Pair]:
    return [(src + 1, rec + 1) for src in range(n_sources) for rec in range(n_receivers)]


def _window_samples(center_idx: int, width: int, n_samples: int) -> slice:
    half = max(width // 2, 1)
    i0 = max(center_idx - half, 0)
    i1 = min(center_idx + half, n_samples)
    return slice(i0, i1)


@dataclass
class MetricConfig:
    pick_search_s: float = 0.012
    window_s: float = 0.003


class CASSMTempGather:
    """In-memory temp-gather equivalent for MATLAB dsitempgath."""

    def __init__(
        self,
        n_sources: int = 24,
        n_receivers: int = 44,
        sample_count: int = 3840,
        sample_rate_hz: float = 48000.0,
    ):
        self.n_sources = n_sources
        self.n_receivers = n_receivers
        self.n_pairs = n_sources * n_receivers
        self.sample_count = sample_count
        self.sample_rate_hz = sample_rate_hz
        self.dt = 1.0 / sample_rate_hz

        self.pairs: List[Pair] = _build_pair_index(n_sources, n_receivers)

        # data shape: (n_epochs, n_pairs, n_samples)
        self.data = np.zeros((0, self.n_pairs, self.sample_count), dtype=np.float32)
        self.epoch_labels: List[str] = []
        self.epoch_times: List[pd.Timestamp] = []

        self._metric_cache: Dict[str, np.ndarray] = {}
        self._pick_cache: Optional[np.ndarray] = None

    @property
    def n_epochs(self) -> int:
        return int(self.data.shape[0])

    def append_epoch(self, epoch_label: str, epoch_cube: np.ndarray) -> None:
        """Append one epoch cube of shape (n_sources, n_receivers, n_samples)."""
        if epoch_cube.shape != (self.n_sources, self.n_receivers, self.sample_count):
            raise ValueError(
                "epoch_cube shape mismatch. "
                f"Expected {(self.n_sources, self.n_receivers, self.sample_count)}, got {epoch_cube.shape}."
            )
        pair_data = epoch_cube.reshape(self.n_pairs, self.sample_count)
        pair_data = pair_data[np.newaxis, :, :].astype(np.float32)
        self.data = np.concatenate([self.data, pair_data], axis=0)
        self.epoch_labels.append(epoch_label)
        self.epoch_times.append(_safe_parse_epoch_time(epoch_label))
        self._metric_cache.clear()
        self._pick_cache = None

    def append_many(self, epoch_items: Sequence[Tuple[str, np.ndarray]]) -> int:
        before = self.n_epochs
        for label, cube in epoch_items:
            self.append_epoch(label, cube)
        return self.n_epochs - before

    def to_npz(self, out_file: Path) -> None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_file,
            data=self.data,
            epoch_labels=np.array(self.epoch_labels, dtype=object),
            epoch_times=np.array([t.isoformat() for t in self.epoch_times], dtype=object),
            n_sources=self.n_sources,
            n_receivers=self.n_receivers,
            sample_count=self.sample_count,
            sample_rate_hz=self.sample_rate_hz,
        )

    @classmethod
    def from_npz(cls, in_file: Path) -> "CASSMTempGather":
        obj = np.load(in_file, allow_pickle=True)
        tg = cls(
            n_sources=int(obj["n_sources"]),
            n_receivers=int(obj["n_receivers"]),
            sample_count=int(obj["sample_count"]),
            sample_rate_hz=float(obj["sample_rate_hz"]),
        )
        tg.data = obj["data"].astype(np.float32)
        tg.epoch_labels = [str(x) for x in obj["epoch_labels"].tolist()]
        tg.epoch_times = [pd.to_datetime(x, utc=True) for x in obj["epoch_times"].tolist()]
        return tg

    def _baseline_picks(self, pick_search_s: float) -> np.ndarray:
        """Pick first-break proxies from baseline epoch for each pair."""
        if self.n_epochs == 0:
            return np.zeros(self.n_pairs, dtype=int)
        if self._pick_cache is not None:
            return self._pick_cache

        n_search = max(int(pick_search_s * self.sample_rate_hz), 20)
        n_search = min(n_search, self.sample_count)

        base = self.data[0, :, :n_search]
        grad = np.abs(np.diff(base, axis=1, prepend=base[:, :1]))
        picks = np.argmax(grad, axis=1).astype(int)
        self._pick_cache = picks
        return picks

    def compute_metrics(self, config: Optional[MetricConfig] = None) -> Dict[str, np.ndarray]:
        if config is None:
            config = MetricConfig()
        key = f"{config.pick_search_s:.6f}|{config.window_s:.6f}"
        if key in self._metric_cache:
            return {
                "rms": self._metric_cache[f"{key}:rms"],
                "centfreq": self._metric_cache[f"{key}:centfreq"],
                "dt_us": self._metric_cache[f"{key}:dt_us"],
            }

        if self.n_epochs == 0:
            z = np.zeros((self.n_pairs, 0), dtype=np.float32)
            return {"rms": z, "centfreq": z, "dt_us": z}

        picks = self._baseline_picks(config.pick_search_s)
        win_samples = max(int(config.window_s * self.sample_rate_hz), 16)

        rms = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)
        centfreq = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)
        dt_us = np.zeros((self.n_pairs, self.n_epochs), dtype=np.float32)

        freqs = np.fft.rfftfreq(win_samples, d=self.dt)

        for p in range(self.n_pairs):
            p0 = int(picks[p])
            s0 = _window_samples(p0, win_samples, self.sample_count)
            base_trace = self.data[0, p, s0]
            base_grad = np.abs(np.diff(base_trace, prepend=base_trace[:1]))
            base_pick_local = int(np.argmax(base_grad))
            base_pick_global = s0.start + base_pick_local

            for e in range(self.n_epochs):
                tr = self.data[e, p, :]
                sw = _window_samples(base_pick_global, win_samples, self.sample_count)
                w = tr[sw]

                # Delay proxy: shift of strongest derivative within analysis window.
                grad = np.abs(np.diff(w, prepend=w[:1]))
                pick_local = int(np.argmax(grad))
                pick_global = sw.start + pick_local
                dt_us[p, e] = (pick_global - base_pick_global) * self.dt * 1e6

                # RMS amplitude
                rms[p, e] = float(np.sqrt(np.mean(np.square(w)))) if w.size else 0.0

                # Centroid frequency in kHz
                spec = np.abs(np.fft.rfft(w, n=win_samples))
                denom = float(np.sum(spec))
                if denom > 0:
                    centfreq[p, e] = float(np.sum(freqs * spec) / denom / 1000.0)
                else:
                    centfreq[p, e] = 0.0

        self._metric_cache[f"{key}:rms"] = rms
        self._metric_cache[f"{key}:centfreq"] = centfreq
        self._metric_cache[f"{key}:dt_us"] = dt_us
        return {"rms": rms, "centfreq": centfreq, "dt_us": dt_us}


def _list_epoch_dirs(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def _load_epoch_npz(epoch_dir: Path) -> Optional[np.ndarray]:
    for name in ("epoch_data.npz", "dsi_epoch.npz"):
        f = epoch_dir / name
        if f.exists():
            obj = np.load(f, allow_pickle=True)
            if "data" in obj:
                return obj["data"]
    return None


def _load_epoch_seg2(
    epoch_dir: Path,
    n_sources: int,
    n_receivers: int,
    sample_count: int,
) -> Optional[np.ndarray]:
    """Best-effort SEG2 loader with minimal assumptions.

    Expected mapping: sorted SEG2 files correspond to source index order.
    Receiver index inferred from trace CHANNEL_NUMBER when available, else
    sequential trace order.
    """
    seg_files = sorted(list(epoch_dir.glob("*.dat")) + list(epoch_dir.glob("*.seg2")))
    if not seg_files:
        return None

    try:
        from obspy import read as obspy_read
    except Exception as exc:
        raise RuntimeError("ObsPy is required for SEG2 ingestion.") from exc

    cube = np.zeros((n_sources, n_receivers, sample_count), dtype=np.float32)

    for src_idx, seg_file in enumerate(seg_files[:n_sources]):
        try:
            st = obspy_read(str(seg_file))
        except Exception as exc:
            LOG.warning("Failed reading SEG2 file %s: %s", seg_file, exc)
            continue

        for tr_idx, tr in enumerate(st):
            rec_idx = None
            try:
                ch = int(tr.stats.seg2.get("CHANNEL_NUMBER", tr_idx + 1))
                if 1 <= ch <= n_receivers:
                    rec_idx = ch - 1
            except Exception:
                rec_idx = tr_idx if tr_idx < n_receivers else None

            if rec_idx is None or rec_idx >= n_receivers:
                continue

            data = np.asarray(tr.data, dtype=np.float32)
            if data.size >= sample_count:
                cube[src_idx, rec_idx, :] = data[:sample_count]
            else:
                cube[src_idx, rec_idx, : data.size] = data

    return cube


def load_epoch_cube(
    epoch_dir: Path,
    n_sources: int,
    n_receivers: int,
    sample_count: int,
) -> Optional[np.ndarray]:
    cube = _load_epoch_npz(epoch_dir)
    if cube is not None:
        return cube.astype(np.float32)
    return _load_epoch_seg2(epoch_dir, n_sources, n_receivers, sample_count)


class CUSSPCASSMDashboard:
    def __init__(self):
        self._tg: Optional[CASSMTempGather] = None
        self._auto_callback = None
        self._update_lock = threading.Lock()

        # Controls
        self.parent_dir = pn.widgets.TextInput(
            name="Parent Folder",
            value="/data/chet-cussp/cassm",
            placeholder="Folder containing CASSMdata and tempgathers",
            sizing_mode="stretch_width",
        )
        self.data_subdir = pn.widgets.TextInput(name="Data Subfolder", value="CASSMdata")
        self.cache_subdir = pn.widgets.TextInput(name="Cache Subfolder", value="tempgathers")
        self.cache_filename = pn.widgets.TextInput(name="Cache File", value="cussp_tempgather.npz")

        self.n_sources = pn.widgets.IntInput(name="Sources", value=24, start=1, end=256)
        self.n_receivers = pn.widgets.IntInput(name="Receivers", value=44, start=1, end=256)
        self.sample_count = pn.widgets.IntInput(name="Samples", value=3840, start=128, end=65536)
        self.sample_rate = pn.widgets.FloatInput(name="Sample rate (Hz)", value=48000.0, step=1000.0)

        self.pick_search_s = pn.widgets.FloatInput(name="Pick search (s)", value=0.012, step=0.001)
        self.window_s = pn.widgets.FloatInput(name="Metric window (s)", value=0.003, step=0.001)

        self.source_sel = pn.widgets.IntInput(name="Source", value=1, start=1, end=24)
        self.receiver_sel = pn.widgets.IntInput(name="Receiver", value=1, start=1, end=44)

        self.auto_update = pn.widgets.Toggle(name="Auto Update", value=False)
        self.auto_period_s = pn.widgets.IntInput(name="Auto period (s)", value=300, start=10, end=3600)

        self.btn_create = pn.widgets.Button(name="Create New", button_type="primary")
        self.btn_load = pn.widgets.Button(name="Load Cache", button_type="default")
        self.btn_save = pn.widgets.Button(name="Save Cache", button_type="default")
        self.btn_update = pn.widgets.Button(name="Update", button_type="success")
        self.btn_plot_gather = pn.widgets.Button(name="Plot Current Temp-Gather", button_type="light")

        self.status = pn.pane.Alert("Ready.", alert_type="secondary")
        self.summary = pn.pane.Markdown("No data loaded.")

        # Figures
        self.fig_rms = pn.pane.Plotly(height=260, config={"responsive": True})
        self.fig_centfreq = pn.pane.Plotly(height=260, config={"responsive": True})
        self.fig_dt = pn.pane.Plotly(height=260, config={"responsive": True})
        self.fig_gather = pn.pane.Plotly(height=320, config={"responsive": True})

        # Callbacks
        self.btn_create.on_click(self._create_new)
        self.btn_load.on_click(self._load_cache)
        self.btn_save.on_click(self._save_cache)
        self.btn_update.on_click(self._update)
        self.btn_plot_gather.on_click(self._plot_current_gather)
        self.source_sel.param.watch(self._refresh_metric_plots, "value")
        self.receiver_sel.param.watch(self._refresh_metric_plots, "value")
        self.pick_search_s.param.watch(self._refresh_metric_plots, "value")
        self.window_s.param.watch(self._refresh_metric_plots, "value")
        self.auto_update.param.watch(self._toggle_auto_update, "value")

    def _set_status(self, msg: str, kind: str = "secondary") -> None:
        self.status.object = msg
        self.status.alert_type = kind

    def _parent_path(self) -> Path:
        return Path(self.parent_dir.value).expanduser()

    def _data_path(self) -> Path:
        return self._parent_path() / self.data_subdir.value

    def _cache_path(self) -> Path:
        return self._parent_path() / self.cache_subdir.value / self.cache_filename.value

    def _metric_config(self) -> MetricConfig:
        return MetricConfig(
            pick_search_s=max(float(self.pick_search_s.value), 0.001),
            window_s=max(float(self.window_s.value), 0.0005),
        )

    def _scan_new_epochs(self) -> List[Tuple[str, np.ndarray]]:
        if self._tg is None:
            return []
        data_dir = self._data_path()
        epoch_dirs = _list_epoch_dirs(data_dir)
        known = set(self._tg.epoch_labels)
        new_items: List[Tuple[str, np.ndarray]] = []

        for ep in epoch_dirs:
            if ep.name in known:
                continue
            cube = load_epoch_cube(
                ep,
                n_sources=self._tg.n_sources,
                n_receivers=self._tg.n_receivers,
                sample_count=self._tg.sample_count,
            )
            if cube is None:
                continue
            if cube.shape != (self._tg.n_sources, self._tg.n_receivers, self._tg.sample_count):
                LOG.warning("Skipping %s due to shape mismatch: %s", ep, cube.shape)
                continue
            new_items.append((ep.name, cube))
        return new_items

    def _update_summary(self) -> None:
        if self._tg is None:
            self.summary.object = "No data loaded."
            return

        if self._tg.n_epochs:
            t0 = self._tg.epoch_times[0]
            t1 = self._tg.epoch_times[-1]
            tmsg = f"{t0} to {t1}"
        else:
            tmsg = "n/a"

        self.summary.object = (
            f"**Epochs:** {self._tg.n_epochs}  \\n"
            f"**Pairs:** {self._tg.n_pairs} ({self._tg.n_sources}x{self._tg.n_receivers})  \\n"
            f"**Samples/trace:** {self._tg.sample_count} at {self._tg.sample_rate_hz:.1f} Hz  \\n"
            f"**Time span:** {tmsg}"
        )

        self.source_sel.end = self._tg.n_sources
        self.receiver_sel.end = self._tg.n_receivers
        self.source_sel.value = int(np.clip(self.source_sel.value, 1, self._tg.n_sources))
        self.receiver_sel.value = int(np.clip(self.receiver_sel.value, 1, self._tg.n_receivers))

    def _pair_idx(self) -> Optional[int]:
        if self._tg is None:
            return None
        src = int(self.source_sel.value)
        rec = int(self.receiver_sel.value)
        if not (1 <= src <= self._tg.n_sources and 1 <= rec <= self._tg.n_receivers):
            return None
        return (src - 1) * self._tg.n_receivers + (rec - 1)

    def _epoch_x(self) -> List[str]:
        if self._tg is None:
            return []
        return self._tg.epoch_labels

    def _refresh_metric_plots(self, *_events) -> None:
        if self._tg is None or self._tg.n_epochs == 0:
            return
        pidx = self._pair_idx()
        if pidx is None:
            return

        m = self._tg.compute_metrics(self._metric_config())
        x = self._epoch_x()
        src = self.source_sel.value
        rec = self.receiver_sel.value
        title_suffix = f"S{src:02d}-R{rec:02d}"

        self.fig_rms.object = go.Figure(
            data=[go.Scatter(x=x, y=m["rms"][pidx, :], mode="lines+markers", name="RMS")],
            layout=dict(title=f"RMS Amplitude ({title_suffix})", margin=dict(l=40, r=20, t=35, b=30)),
        )
        self.fig_centfreq.object = go.Figure(
            data=[go.Scatter(x=x, y=m["centfreq"][pidx, :], mode="lines+markers", name="Centroid Freq")],
            layout=dict(title=f"Centroid Frequency (kHz) ({title_suffix})", margin=dict(l=40, r=20, t=35, b=30)),
        )
        self.fig_dt.object = go.Figure(
            data=[go.Scatter(x=x, y=m["dt_us"][pidx, :], mode="lines+markers", name="Delay")],
            layout=dict(title=f"Relative Delay (microseconds) ({title_suffix})", margin=dict(l=40, r=20, t=35, b=30)),
        )

    def _plot_current_gather(self, _event=None) -> None:
        if self._tg is None or self._tg.n_epochs == 0:
            return
        pidx = self._pair_idx()
        if pidx is None:
            return
        traces = self._tg.data[:, pidx, :]
        self.fig_gather.object = go.Figure(
            data=[
                go.Heatmap(
                    z=traces,
                    x=np.arange(self._tg.sample_count) * self._tg.dt * 1000.0,
                    y=self._epoch_x(),
                    colorscale="RdBu",
                    reversescale=True,
                    colorbar=dict(title="Amp"),
                )
            ],
            layout=dict(
                title=f"Temp-gather waveform panel (S{self.source_sel.value:02d}-R{self.receiver_sel.value:02d})",
                xaxis_title="Time (ms)",
                yaxis_title="Epoch",
                margin=dict(l=45, r=20, t=35, b=40),
            ),
        )

    def _create_new(self, _event=None) -> None:
        with self._update_lock:
            self._set_status("Creating initial temp-gather...", "warning")
            tg = CASSMTempGather(
                n_sources=int(self.n_sources.value),
                n_receivers=int(self.n_receivers.value),
                sample_count=int(self.sample_count.value),
                sample_rate_hz=float(self.sample_rate.value),
            )

            items: List[Tuple[str, np.ndarray]] = []
            for ep in _list_epoch_dirs(self._data_path()):
                cube = load_epoch_cube(ep, tg.n_sources, tg.n_receivers, tg.sample_count)
                if cube is None:
                    continue
                if cube.shape != (tg.n_sources, tg.n_receivers, tg.sample_count):
                    LOG.warning("Skipping %s due to shape mismatch: %s", ep, cube.shape)
                    continue
                items.append((ep.name, cube))

            n_added = tg.append_many(items)
            self._tg = tg
            self._update_summary()
            self._refresh_metric_plots()
            self._plot_current_gather()

            if n_added == 0:
                self._set_status("No readable epochs found under data directory.", "warning")
            else:
                self._set_status(f"Created temp-gather with {n_added} epochs.", "success")

    def _update(self, _event=None) -> None:
        with self._update_lock:
            if self._tg is None:
                self._set_status("Create or load a temp-gather first.", "warning")
                return
            self._set_status("Checking for new epochs...", "warning")
            items = self._scan_new_epochs()
            if not items:
                self._set_status("No new epochs found.", "secondary")
                return

            n = self._tg.append_many(items)
            self._update_summary()
            self._refresh_metric_plots()
            self._plot_current_gather()
            self._set_status(f"Added {n} new epoch(s).", "success")

    def _save_cache(self, _event=None) -> None:
        if self._tg is None:
            self._set_status("Nothing to save. Create or load data first.", "warning")
            return
        out = self._cache_path()
        self._tg.to_npz(out)
        self._set_status(f"Saved cache: {out}", "success")

    def _load_cache(self, _event=None) -> None:
        path = self._cache_path()
        if not path.exists():
            self._set_status(f"Cache not found: {path}", "warning")
            return
        self._tg = CASSMTempGather.from_npz(path)
        self._update_summary()
        self._refresh_metric_plots()
        self._plot_current_gather()
        self._set_status(f"Loaded cache: {path}", "success")

    def _toggle_auto_update(self, event) -> None:
        on = bool(event.new)
        if on:
            period_ms = int(max(self.auto_period_s.value, 10) * 1000)
            self._auto_callback = pn.state.add_periodic_callback(self._update, period=period_ms)
            self._set_status(f"Auto-update enabled ({self.auto_period_s.value}s).", "success")
        else:
            if self._auto_callback is not None:
                self._auto_callback.stop()
                self._auto_callback = None
            self._set_status("Auto-update disabled.", "secondary")

    def panel(self):
        controls = pn.Column(
            pn.pane.Markdown("### CUSSP CASSM Controls"),
            self.parent_dir,
            pn.Row(self.data_subdir, self.cache_subdir),
            self.cache_filename,
            pn.Row(self.n_sources, self.n_receivers),
            pn.Row(self.sample_count, self.sample_rate),
            pn.Row(self.pick_search_s, self.window_s),
            pn.Row(self.source_sel, self.receiver_sel),
            pn.Row(self.btn_create, self.btn_update),
            pn.Row(self.btn_load, self.btn_save),
            self.btn_plot_gather,
            pn.Row(self.auto_update, self.auto_period_s),
            self.status,
            self.summary,
            width=460,
        )

        plots = pn.Column(
            self.fig_rms,
            self.fig_centfreq,
            self.fig_dt,
            self.fig_gather,
            sizing_mode="stretch_both",
        )

        return pn.Row(controls, plots, sizing_mode="stretch_both")


dashboard = CUSSPCASSMDashboard()

pn.template.VanillaTemplate(
    title="CUSSP CASSM Realtime",
    logo="/CUSSP.png",
    main=dashboard.panel(),
).servable()
