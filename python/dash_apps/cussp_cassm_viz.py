"""CUSSP CASSM visualization-only dashboard (Panel + Plotly).

This app is intended for the lightweight remote VM. It reads precomputed
artifacts produced by the headless pipeline:

- bundle NPZ: metrics + preview waveforms
- manifest JSON: metadata and update time

No SEG2 reading, processing, or inversion occurs in this process.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import panel as pn
import plotly.graph_objects as go

pn.extension("plotly")

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class CUSSPCASSMViz:
    def __init__(self):
        self.bundle_path = pn.widgets.TextInput(
            name="Bundle NPZ",
            value="/data/chet-cussp/cassm/live/cassm_dashboard_bundle.npz",
            sizing_mode="stretch_width",
        )
        self.manifest_path = pn.widgets.TextInput(
            name="Manifest JSON",
            value="/data/chet-cussp/cassm/live/cassm_dashboard_manifest.json",
            sizing_mode="stretch_width",
        )

        self.source_sel = pn.widgets.IntInput(name="Source", value=1, start=1, end=24)
        self.receiver_sel = pn.widgets.IntInput(name="Receiver", value=1, start=1, end=44)

        self.refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary")
        self.auto_refresh = pn.widgets.Toggle(name="Auto Refresh", value=True)
        self.auto_period_s = pn.widgets.IntInput(name="Refresh period (s)", value=60, start=10, end=3600)

        self.status = pn.pane.Alert("Waiting for bundle...", alert_type="secondary")
        self.summary = pn.pane.Markdown("No data loaded.")
        self.inversion_summary = pn.pane.Markdown("No inversion outputs listed.")
        self.inversion_preview = pn.Column(sizing_mode="stretch_width")

        self.fig_rms = pn.pane.Plotly(height=260, config={"responsive": True})
        self.fig_centfreq = pn.pane.Plotly(height=260, config={"responsive": True})
        self.fig_dt = pn.pane.Plotly(height=260, config={"responsive": True})
        self.fig_gather = pn.pane.Plotly(height=330, config={"responsive": True})

        self._bundle_mtime: Optional[float] = None
        self._manifest_mtime: Optional[float] = None
        self._auto_cb = None

        self._bundle = None
        self._manifest = {}

        self.refresh_btn.on_click(self._refresh)
        self.source_sel.param.watch(self._replot, "value")
        self.receiver_sel.param.watch(self._replot, "value")
        self.auto_refresh.param.watch(self._toggle_auto, "value")

        self._refresh()
        self._toggle_auto(type("_evt", (), {"new": bool(self.auto_refresh.value)})())

    def _set_status(self, text: str, kind: str = "secondary") -> None:
        self.status.object = text
        self.status.alert_type = kind

    def _load_manifest(self) -> dict:
        p = Path(self.manifest_path.value)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}

    def _load_bundle(self) -> Optional[dict]:
        p = Path(self.bundle_path.value)
        if not p.exists():
            self._set_status(f"Bundle not found: {p}", "warning")
            return None

        try:
            obj = np.load(p, allow_pickle=True)
            out = {
                "epoch_labels": [str(x) for x in obj["epoch_labels"].tolist()],
                "epoch_times": [str(x) for x in obj["epoch_times"].tolist()],
                "rms": obj["rms"],
                "centfreq": obj["centfreq"],
                "dt_us": obj["dt_us"],
                "gather_preview": obj["gather_preview"],
                "preview_dt_ms": float(obj["preview_dt_ms"]),
                "n_sources": int(obj["n_sources"]),
                "n_receivers": int(obj["n_receivers"]),
                "sample_count": int(obj["sample_count"]),
                "sample_rate_hz": float(obj["sample_rate_hz"]),
                "n_epochs": int(obj["n_epochs"]),
            }
            return out
        except Exception as exc:
            LOG.exception("Failed loading bundle: %s", exc)
            self._set_status(f"Failed loading bundle: {exc}", "danger")
            return None

    def _pair_idx(self) -> Optional[int]:
        if self._bundle is None:
            return None
        nsrc = self._bundle["n_sources"]
        nrec = self._bundle["n_receivers"]
        src = int(self.source_sel.value)
        rec = int(self.receiver_sel.value)
        if not (1 <= src <= nsrc and 1 <= rec <= nrec):
            return None
        return (src - 1) * nrec + (rec - 1)

    def _refresh(self, _event=None) -> None:
        bundle_file = Path(self.bundle_path.value)
        manifest_file = Path(self.manifest_path.value)

        b_mtime = bundle_file.stat().st_mtime if bundle_file.exists() else None
        m_mtime = manifest_file.stat().st_mtime if manifest_file.exists() else None

        if b_mtime is None:
            self._set_status(f"Bundle missing: {bundle_file}", "warning")
            return

        changed = (b_mtime != self._bundle_mtime) or (m_mtime != self._manifest_mtime) or (self._bundle is None)
        if not changed:
            self._set_status("No new bundle update.", "secondary")
            return

        bundle = self._load_bundle()
        if bundle is None:
            return

        manifest = self._load_manifest()
        self._bundle = bundle
        self._manifest = manifest
        self._bundle_mtime = b_mtime
        self._manifest_mtime = m_mtime

        self.source_sel.end = bundle["n_sources"]
        self.receiver_sel.end = bundle["n_receivers"]
        self.source_sel.value = int(np.clip(self.source_sel.value, 1, bundle["n_sources"]))
        self.receiver_sel.value = int(np.clip(self.receiver_sel.value, 1, bundle["n_receivers"]))

        updated_utc = manifest.get("updated_utc", "unknown")
        self.summary.object = (
            f"**Updated (UTC):** {updated_utc}  \\n"
            f"**Epochs:** {bundle['n_epochs']}  \\n"
            f"**Pairs:** {bundle['n_sources'] * bundle['n_receivers']} ({bundle['n_sources']}x{bundle['n_receivers']})  \\n"
            f"**Sample rate:** {bundle['sample_rate_hz']:.1f} Hz  \\n"
            f"**Preview dt:** {bundle['preview_dt_ms']:.4f} ms"
        )

        outputs = manifest.get("inversion_outputs", []) if isinstance(manifest, dict) else []
        latest = manifest.get("latest_inversion", None) if isinstance(manifest, dict) else None
        self.inversion_summary.object = (
            f"**Inversion outputs listed:** {len(outputs)}  \\n"
            f"**Latest inversion:** {latest.get('name', 'n/a') if isinstance(latest, dict) else 'n/a'}"
        )
        self._render_inversion_previews(outputs)

        self._replot()
        self._set_status("Bundle loaded.", "success")

    def _render_inversion_previews(self, outputs) -> None:
        panes = [pn.pane.Markdown("### Latest Inversion Products")]
        if not outputs:
            panes.append(pn.pane.Markdown("No inversion outputs available."))
            self.inversion_preview.objects = panes
            return

        image_ext = (".png", ".jpg", ".jpeg", ".webp", ".svg")
        shown = 0
        max_show = 3
        for item in outputs:
            if shown >= max_show:
                break
            if not isinstance(item, dict):
                continue
            name = item.get("name", "unknown")
            path = item.get("path")
            url = item.get("url")
            updated = item.get("updated_utc", "")
            if not str(name).lower().endswith(image_ext):
                continue

            target = url or path
            panes.append(pn.pane.Markdown(f"**{name}**  \\nUpdated: {updated}"))

            if url:
                panes.append(
                    pn.pane.HTML(
                        f'<img src="{url}" style="max-width:100%; height:auto; border:1px solid #ddd;" />',
                        sizing_mode="stretch_width",
                    )
                )
            elif path and Path(path).exists():
                panes.append(pn.pane.Image(str(path), sizing_mode="stretch_width"))
            else:
                panes.append(pn.pane.Markdown(f"Preview unavailable. Artifact: {target}"))

            shown += 1

        if shown == 0:
            first = outputs[0] if isinstance(outputs[0], dict) else {}
            panes.append(pn.pane.Markdown(f"No image outputs found. Latest artifact: {first.get('name', 'n/a')}"))

        self.inversion_preview.objects = panes

    def _replot(self, *_events) -> None:
        if self._bundle is None:
            return
        pidx = self._pair_idx()
        if pidx is None:
            return

        x = self._bundle["epoch_labels"]
        src = int(self.source_sel.value)
        rec = int(self.receiver_sel.value)
        title_suffix = f"S{src:02d}-R{rec:02d}"

        self.fig_rms.object = go.Figure(
            data=[go.Scatter(x=x, y=self._bundle["rms"][pidx, :], mode="lines+markers")],
            layout=dict(title=f"RMS Amplitude ({title_suffix})", margin=dict(l=40, r=20, t=35, b=30)),
        )
        self.fig_centfreq.object = go.Figure(
            data=[go.Scatter(x=x, y=self._bundle["centfreq"][pidx, :], mode="lines+markers")],
            layout=dict(title=f"Centroid Frequency (kHz) ({title_suffix})", margin=dict(l=40, r=20, t=35, b=30)),
        )
        self.fig_dt.object = go.Figure(
            data=[go.Scatter(x=x, y=self._bundle["dt_us"][pidx, :], mode="lines+markers")],
            layout=dict(title=f"Relative Delay (microseconds) ({title_suffix})", margin=dict(l=40, r=20, t=35, b=30)),
        )

        preview = self._bundle["gather_preview"][:, pidx, :]
        n_cols = preview.shape[1]
        tx = np.arange(n_cols) * self._bundle["preview_dt_ms"]
        self.fig_gather.object = go.Figure(
            data=[
                go.Heatmap(
                    z=preview,
                    x=tx,
                    y=x,
                    colorscale="RdBu",
                    reversescale=True,
                    colorbar=dict(title="Amp"),
                )
            ],
            layout=dict(
                title=f"Waveform Preview ({title_suffix})",
                xaxis_title="Time (ms)",
                yaxis_title="Epoch",
                margin=dict(l=45, r=20, t=35, b=40),
            ),
        )

    def _toggle_auto(self, event) -> None:
        on = bool(event.new)
        if on:
            period_ms = int(max(self.auto_period_s.value, 10) * 1000)
            if self._auto_cb is not None:
                self._auto_cb.stop()
            self._auto_cb = pn.state.add_periodic_callback(self._refresh, period=period_ms)
            self._set_status(f"Auto refresh enabled ({self.auto_period_s.value}s).", "success")
        else:
            if self._auto_cb is not None:
                self._auto_cb.stop()
                self._auto_cb = None
            self._set_status("Auto refresh disabled.", "secondary")

    def panel(self):
        controls = pn.Column(
            pn.pane.Markdown("### CUSSP CASSM Viz"),
            self.bundle_path,
            self.manifest_path,
            pn.Row(self.source_sel, self.receiver_sel),
            pn.Row(self.refresh_btn, self.auto_refresh, self.auto_period_s),
            self.status,
            self.summary,
            self.inversion_summary,
            self.inversion_preview,
            width=480,
        )

        plots = pn.Column(
            self.fig_rms,
            self.fig_centfreq,
            self.fig_dt,
            self.fig_gather,
            sizing_mode="stretch_both",
        )
        return pn.Row(controls, plots, sizing_mode="stretch_both")


app = CUSSPCASSMViz()

pn.template.VanillaTemplate(
    title="CUSSP CASSM",
    logo="/CUSSP.png",
    main=app.panel(),
).servable()
