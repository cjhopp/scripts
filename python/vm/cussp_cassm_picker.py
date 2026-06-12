"""CASSM manual P-wave arrival picker.

Pair-by-pair interactive GUI for picking baseline P-arrivals on stacked
waveforms.  Writes picks to a JSON file consumable by cussp_cassm_process.py
via --manual-picks-file.

Usage
-----
conda run -n ttcr_inv python cussp_cassm_picker.py \
  --cache-file /path/to/cassm_tempgather.npz \
  --picks-file  /path/to/manual_picks.json \
  --n-sources 16 --n-receivers 72 \
  --accel-filter-low-hz 1000 --accel-filter-high-hz 12000 \
  --hydro-filter-low-hz 5000 --hydro-filter-high-hz 20000 \
  --clip-first-s 0.002 --mute-first-s 0.002 \
  --active-receiver-channels 1,5,7,12,13,16,19,22,25,28,31,37,44,46,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71 \
  --known-bad-receiver-channels 72 \
  --source-boreholes AML,AML,AML,AML,AMU,AMU,AMU,AMU,DML,DML,DML,DML,DMU,DMU,DMU,DMU \
  --port 8052

Then open http://localhost:8052 in a browser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Reuse processing helpers from the sibling pipeline script.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from cussp_cassm_process import (  # noqa: E402
    CASSMTempGather,
    MetricConfig,
    _apply_picker,
    _build_pair_labels,
    _build_receiver_labels,
    _build_source_labels,
    _default_receiver_boreholes,
    _build_same_well_mask,
    _preprocess_waveform,
)

import pandas as pd  # noqa: E402
import dash  # noqa: E402
from dash import Input, Output, State, ctx, dcc, html  # noqa: E402

import plotly.graph_objects as go  # noqa: E402

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CASSM manual P-wave picker GUI")
    p.add_argument("--cache-file", required=True,
                   help="Path to the tempgather .npz cache produced by cussp_cassm_process.py")
    p.add_argument("--picks-file", default="manual_picks.json",
                   help="JSON file to load/save picks (default: manual_picks.json)")
    p.add_argument("--n-sources", type=int, default=16)
    p.add_argument("--n-receivers", type=int, default=72)
    p.add_argument("--source-boreholes", default=None,
                   help="Comma-separated borehole name per source (e.g. AML,AML,AMU,...)")
    p.add_argument("--active-receiver-channels", default=None,
                   help="Comma-separated 1-based channel numbers that are active")
    p.add_argument("--known-bad-receiver-channels", default=None,
                   help="Comma-separated 1-based channel numbers to zero regardless")
    p.add_argument("--accel-filter-low-hz",  type=float, default=1000.0)
    p.add_argument("--accel-filter-high-hz", type=float, default=12000.0)
    p.add_argument("--hydro-filter-low-hz",  type=float, default=5000.0)
    p.add_argument("--hydro-filter-high-hz", type=float, default=20000.0)
    p.add_argument("--filter-order", type=int, default=4)
    p.add_argument("--clip-first-s", type=float, default=0.002)
    p.add_argument("--mute-first-s", type=float, default=0.002)
    p.add_argument("--hydro-clip-first-s", type=float, default=None,
                   help="Clip window (s) for hydrophone pairs; defaults to --clip-first-s")
    p.add_argument("--hydro-mute-first-s", type=float, default=None,
                   help="Pick-search mute (s) for hydrophone pairs; defaults to --mute-first-s")
    p.add_argument(
        "--baseline-end-date",
        default="",
        help=(
            "ISO-8601 date (UTC) marking the end of the baseline period, e.g. '2026-05-06'. "
            "All epochs with timestamps <= this value are stacked as the baseline waveform. "
            "Overrides --baseline-n-epochs when set."
        ),
    )
    p.add_argument("--baseline-n-epochs", type=int, default=0,
                   help="Number of leading epochs to use as baseline (0 = all epochs). "
                        "Overridden by --baseline-end-date.")
    p.add_argument("--port", type=int, default=8052)
    p.add_argument("--host", default="127.0.0.1")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_gather(args: argparse.Namespace) -> CASSMTempGather:
    p = Path(args.cache_file)
    if p.suffix in (".h5", ".hdf5"):
        # Compact load: keeps only the ~536 valid pairs in memory (~5 GB)
        # instead of expanding to the full 1152-pair array (~46 GB).
        tg = CASSMTempGather.from_hdf5_compact(p)
    else:
        tg = CASSMTempGather.from_npz(p)
    return tg


def _build_valid_mask(
    tg: CASSMTempGather,
    args: argparse.Namespace,
) -> np.ndarray:
    """Replicate the valid-pairs mask from run_once()."""
    bad_channels: set = set()
    if args.known_bad_receiver_channels:
        bad_channels = {int(x.strip()) for x in args.known_bad_receiver_channels.split(",") if x.strip()}
    if args.active_receiver_channels:
        active = {int(x.strip()) for x in args.active_receiver_channels.split(",") if x.strip()}
        all_ch = set(range(1, tg.n_receivers + 1))
        bad_channels |= all_ch - active

    valid = np.ones(tg.n_pairs, dtype=bool)
    for p in range(tg.n_pairs):
        if (p % tg.n_receivers) + 1 in bad_channels:
            valid[p] = False

    if args.source_boreholes:
        src_wells = [w.strip() for w in args.source_boreholes.split(",") if w.strip()]
        rec_bh = _default_receiver_boreholes(tg.n_receivers)
        if len(src_wells) == tg.n_sources:
            sw = _build_same_well_mask(tg.n_sources, tg.n_receivers, src_wells, rec_bh)
            valid &= ~sw
    return valid


def _make_config(args: argparse.Namespace) -> MetricConfig:
    return MetricConfig(
        clip_first_s=args.clip_first_s,
        mute_first_s=args.mute_first_s,
        hydro_clip_first_s=args.hydro_clip_first_s,
        hydro_mute_first_s=args.hydro_mute_first_s,
        filter_order=args.filter_order,
        accel_filter_low_hz=args.accel_filter_low_hz,
        accel_filter_high_hz=args.accel_filter_high_hz,
        hydro_filter_low_hz=args.hydro_filter_low_hz,
        hydro_filter_high_hz=args.hydro_filter_high_hz,
    )


def _preprocess_pair(
    tg: CASSMTempGather,
    pair_idx: int,
    config: MetricConfig,
) -> np.ndarray:
    """Return (n_epochs, n_samples) preprocessed float64 array for one pair."""
    out = np.zeros((tg.n_epochs, tg.sample_count), dtype=np.float64)
    for e in range(tg.n_epochs):
        out[e] = _preprocess_waveform(
            tg.get_pair(e, pair_idx),
            tg.sample_rate_hz,
            config,
            pair_index=pair_idx,
            n_receivers=tg.n_receivers,
        )
    return out


# ---------------------------------------------------------------------------
# Plotly figure builder
# ---------------------------------------------------------------------------

def _build_figure(
    traces: np.ndarray,            # (n_epochs, n_samples)
    dt_s: float,
    pair_label: str,
    clip_s: float,
    mute_s: float,
    hydro_clip_s: Optional[float] = None,
    hydro_mute_s: Optional[float] = None,
    pick_sample: Optional[int] = None,
    aic_pick_sample: Optional[int] = None,
    xlim_ms: float = 15.0,
    accel_filter_low: float = 1000.0,
    accel_filter_high: float = 12000.0,
    hydro_filter_low: float = 5000.0,
    hydro_filter_high: float = 20000.0,
    pair_idx: int = 0,
    n_receivers: int = 72,
    baseline_indices: Optional[List[int]] = None,
    show_traces: bool = False,
) -> go.Figure:
    is_hydro = (pair_idx % n_receivers) >= 48
    eff_clip_s = (hydro_clip_s if is_hydro and hydro_clip_s is not None else clip_s)
    eff_mute_s = (hydro_mute_s if is_hydro and hydro_mute_s is not None else mute_s)
    filt_lo = hydro_filter_low if is_hydro else accel_filter_low
    filt_hi = hydro_filter_high if is_hydro else accel_filter_high
    color_stack = "darkorange" if is_hydro else "royalblue"
    sensor_label = "Hydro" if is_hydro else "Accel"

    n_epochs, n_samp = traces.shape
    t_ms = np.arange(n_samp) * dt_s * 1000.0
    mask_t = t_ms <= xlim_ms

    b_idx = baseline_indices if baseline_indices else list(range(n_epochs))
    b_idx = [i for i in b_idx if 0 <= i < n_epochs]
    post_idx = [i for i in range(n_epochs) if i not in set(b_idx)]

    stack = np.mean(traces[b_idx], axis=0) if b_idx else np.mean(traces, axis=0)
    amax = np.max(np.abs(stack)) + 1e-9
    stack_norm = stack / amax

    fig = go.Figure()

    # Clip/mute shading using effective (sensor-aware) values.
    if eff_clip_s > 0:
        fig.add_vrect(x0=0, x1=eff_clip_s * 1000.0,
                      fillcolor="mediumpurple", opacity=0.12, line_width=0,
                      annotation_text="clip", annotation_position="top left",
                      annotation_font_size=9)
    if eff_mute_s > eff_clip_s:
        fig.add_vrect(x0=eff_clip_s * 1000.0, x1=eff_mute_s * 1000.0,
                      fillcolor="salmon", opacity=0.10, line_width=0,
                      annotation_text="mute", annotation_position="top left",
                      annotation_font_size=9)
    elif eff_mute_s > 0:
        fig.add_vrect(x0=0, x1=eff_mute_s * 1000.0,
                      fillcolor="salmon", opacity=0.10, line_width=0,
                      annotation_text="mute", annotation_position="top left",
                      annotation_font_size=9)

    # Post-baseline epoch traces (faint red)
    _clip = 1.0  # clip individual traces to ±1× stack amplitude
    if show_traces:
        for e in post_idx:
            tr = np.clip(traces[e] / amax, -_clip, _clip)
            fig.add_trace(go.Scatter(
            x=t_ms[mask_t], y=tr[mask_t],
            mode="lines",
            line=dict(color="salmon", width=0.6),
            opacity=0.20,
            showlegend=(e == post_idx[0]) if post_idx else False,
            name=f"Post-baseline ({len(post_idx)})",
            legendgroup="post",
            ))

    # Baseline epoch traces (faint blue/grey)
    if show_traces:
        for e in b_idx:
            tr = np.clip(traces[e] / amax, -_clip, _clip)
            fig.add_trace(go.Scatter(
            x=t_ms[mask_t], y=tr[mask_t],
            mode="lines",
            line=dict(color="steelblue" if not is_hydro else "peru", width=0.7),
            opacity=0.20,
            showlegend=(e == b_idx[0]) if b_idx else False,
            name=f"Baseline epoch ({len(b_idx)})",
            legendgroup="baseline_epochs",
        ))

    # Baseline stack trace
    fig.add_trace(go.Scatter(
        x=t_ms[mask_t], y=stack_norm[mask_t],
        mode="lines",
        line=dict(color=color_stack, width=2.5),
        name=f"Baseline stack (n={len(b_idx)})",
    ))

    # AIC baseline pick (reference — orange dashed)
    if aic_pick_sample is not None and aic_pick_sample > 0:
        aic_ms = float(aic_pick_sample) * dt_s * 1000.0
        fig.add_vline(x=aic_ms, line=dict(color="goldenrod", dash="dash", width=1.5),
                      annotation_text="AIC", annotation_position="top right",
                      annotation_font_size=9)

    # Manual pick (red solid)
    if pick_sample is not None:
        pick_ms = float(pick_sample) * dt_s * 1000.0
        fig.add_vline(x=pick_ms, line=dict(color="crimson", dash="solid", width=2.0),
                      annotation_text=f"pick {pick_ms:.3f} ms",
                      annotation_position="top left",
                      annotation_font_size=10,
                      annotation_font_color="crimson")

    fig.update_layout(
        title=dict(
            text=f"{pair_label}  [{sensor_label}  {filt_lo/1000:.0f}–{filt_hi/1000:.0f} kHz]",
            font_size=13,
        ),
        xaxis=dict(title="Time (ms)", range=[0, xlim_ms], showgrid=True),
        yaxis=dict(title="Norm. amplitude", range=[-1.0, 1.0], showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=55, r=20, t=60, b=45),
        height=420,
        hovermode="closest",
        clickmode="event",
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def build_app(args: argparse.Namespace) -> dash.Dash:
    tg = _load_gather(args)
    config = _make_config(args)
    valid_mask = _build_valid_mask(tg, args)
    valid_indices = np.where(valid_mask)[0].tolist()  # pair indices that are pickable
    n_valid = len(valid_indices)

    src_bh = [w.strip() for w in args.source_boreholes.split(",") if w.strip()] \
        if args.source_boreholes else None
    src_labels = _build_source_labels(tg.n_sources, src_bh)
    rec_labels = _build_receiver_labels(tg.n_receivers)
    pair_labels = _build_pair_labels(tg.n_sources, tg.n_receivers, src_labels, rec_labels)

    # Determine which epoch indices belong to the baseline period.
    def _to_utc(t: pd.Timestamp) -> pd.Timestamp:
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

    if args.baseline_end_date:
        try:
            cutoff = pd.Timestamp(args.baseline_end_date)
            cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
            baseline_indices: List[int] = [
                i for i, t in enumerate(tg.epoch_times) if _to_utc(t) <= cutoff
            ]
        except Exception:
            baseline_indices = list(range(tg.n_epochs))
    elif args.baseline_n_epochs > 0:
        baseline_indices = list(range(min(args.baseline_n_epochs, tg.n_epochs)))
    else:
        baseline_indices = list(range(tg.n_epochs))  # use all epochs

    n_baseline = len(baseline_indices)

    # Config used when computing per-pair AIC reference picks on demand.
    config_baseline = MetricConfig(
        clip_first_s=config.clip_first_s,
        mute_first_s=config.mute_first_s,
        hydro_clip_first_s=config.hydro_clip_first_s,
        hydro_mute_first_s=config.hydro_mute_first_s,
        filter_order=config.filter_order,
        accel_filter_low_hz=config.accel_filter_low_hz,
        accel_filter_high_hz=config.accel_filter_high_hz,
        hydro_filter_low_hz=config.hydro_filter_low_hz,
        hydro_filter_high_hz=config.hydro_filter_high_hz,
        baseline_n_epochs=n_baseline,
    )

    # Lazy per-pair AIC cache: computed on first view of each pair, not at startup.
    _aic_cache: Dict[int, Optional[int]] = {}

    def _aic_pick_one(pair_idx: int) -> Optional[int]:
        """Return the AIC baseline pick (sample index) for one pair, cached."""
        if pair_idx in _aic_cache:
            return _aic_cache[pair_idx]
        try:
            n_search = max(int(config_baseline.pick_search_s * tg.sample_rate_hz), 20)
            n_search = min(n_search, tg.sample_count)
            is_hydro = (pair_idx % tg.n_receivers) >= 48
            if is_hydro:
                _cs = config_baseline.hydro_clip_first_s if config_baseline.hydro_clip_first_s is not None else config_baseline.clip_first_s
                _ms = config_baseline.hydro_mute_first_s if config_baseline.hydro_mute_first_s is not None else config_baseline.mute_first_s
            else:
                _cs = config_baseline.clip_first_s
                _ms = config_baseline.mute_first_s
            start_idx = max(
                min(max(int(_cs * tg.sample_rate_hz), 0), n_search - 1),
                min(max(int(_ms * tg.sample_rate_hz), 0), n_search - 1),
            )
            b_idx = [i for i in baseline_indices if 0 <= i < tg.n_epochs]
            if not b_idx:
                _aic_cache[pair_idx] = None
                return None
            base_raw = (np.mean(tg.get_pair(b_idx, pair_idx), axis=0)
                        if len(b_idx) > 1 else tg.get_pair(b_idx[0], pair_idx))
            base_full = _preprocess_waveform(
                base_raw, tg.sample_rate_hz, config_baseline,
                pair_index=pair_idx, n_receivers=tg.n_receivers,
            )
            base = base_full[start_idx:n_search]
            result = (start_idx if base.size <= 1
                      else start_idx + _apply_picker(base, config_baseline, tg.sample_rate_hz))
            _aic_cache[pair_idx] = int(result)
            return int(result)
        except Exception:
            _aic_cache[pair_idx] = None
            return None

    # Load existing manual picks.
    picks_path = Path(args.picks_file)
    picks: Dict[str, int] = {}
    if picks_path.exists():
        try:
            picks = {str(k): int(v) for k, v in json.loads(picks_path.read_text()).items()}
        except Exception:
            pass

    def _save_picks(p: Dict[str, int]) -> None:
        picks_path.write_text(json.dumps(p, indent=2))

    # Pair selector options
    pair_options = [
        {"label": f"[{vi}] {pair_labels[vi]}", "value": vi}
        for vi in valid_indices
    ]

    app = dash.Dash(__name__, title="CASSM Picker")
    app.layout = html.Div([
        dcc.Store(id="picks-store", data=picks),
        dcc.Store(id="valid-indices", data=valid_indices),

        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.H3("CASSM P-wave Picker", style={"margin": "0 12px 0 0", "display": "inline"}),
            html.Span(id="progress-label", style={"fontSize": 14, "color": "#555"}),
        ], style={"padding": "10px 16px 4px", "borderBottom": "1px solid #ddd",
                  "display": "flex", "alignItems": "center"}),

        # ── Controls row ─────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Pair", style={"fontSize": 12}),
                dcc.Dropdown(
                    id="pair-dropdown",
                    options=pair_options,
                    value=valid_indices[0] if valid_indices else None,
                    clearable=False,
                    style={"width": 280},
                ),
            ], style={"marginRight": 16}),

            html.Button("◀ Prev", id="btn-prev", n_clicks=0,
                        style={"marginRight": 6}),
            html.Button("Next ▶", id="btn-next", n_clicks=0,
                        style={"marginRight": 16}),
            html.Button("✕ Clear pick", id="btn-clear", n_clicks=0,
                        style={"marginRight": 16, "color": "crimson"}),

            html.Div([
                html.Label("X-axis limit (ms)", style={"fontSize": 12}),
                dcc.Input(id="xlim-input", type="number", value=15.0,
                          min=2, max=50, step=0.5,
                          style={"width": 70, "marginLeft": 6}),
            ], style={"display": "flex", "alignItems": "center", "marginRight": 20}),

            html.Div([
                html.Label("Accel filter (kHz)", style={"fontSize": 12, "marginRight": 6}),
                dcc.Input(id="accel-lo", type="number",
                          value=args.accel_filter_low_hz / 1000.0, min=0.1, max=20, step=0.1,
                          style={"width": 60}),
                html.Span("–", style={"margin": "0 4px"}),
                dcc.Input(id="accel-hi", type="number",
                          value=args.accel_filter_high_hz / 1000.0, min=1, max=24, step=0.5,
                          style={"width": 60}),
            ], style={"display": "flex", "alignItems": "center", "marginRight": 20}),

            html.Div([
                html.Label("Hydro filter (kHz)", style={"fontSize": 12, "marginRight": 6}),
                dcc.Input(id="hydro-lo", type="number",
                          value=args.hydro_filter_low_hz / 1000.0, min=0.1, max=24, step=0.1,
                          style={"width": 60}),
                html.Span("–", style={"margin": "0 4px"}),
                dcc.Input(id="hydro-hi", type="number",
                          value=args.hydro_filter_high_hz / 1000.0, min=1, max=24, step=0.5,
                          style={"width": 60}),
            ], style={"display": "flex", "alignItems": "center", "marginRight": 20}),

            dcc.Checklist(
                id="show-traces",
                options=[{"label": " Show traces", "value": "show"}],
                value=[],
                style={"fontSize": 13},
            ),
        ], style={"padding": "8px 16px", "display": "flex", "alignItems": "flex-end",
                  "flexWrap": "wrap", "gap": 8, "borderBottom": "1px solid #eee"}),

        # ── Main waveform plot ───────────────────────────────────────────────
        dcc.Graph(id="waveform-graph", config={"scrollZoom": True,
                                                "modeBarButtonsToAdd": ["drawline"],
                                                "displaylogo": False}),

        # ── Status bar ──────────────────────────────────────────────────────
        html.Div(id="status-bar",
                 style={"padding": "4px 16px", "fontSize": 12,
                        "color": "#555", "borderTop": "1px solid #eee"}),
    ])

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output("pair-dropdown", "value"),
        Input("btn-prev", "n_clicks"),
        Input("btn-next", "n_clicks"),
        State("pair-dropdown", "value"),
        prevent_initial_call=True,
    )
    def navigate(n_prev, n_next, current_val):
        if current_val is None or not valid_indices:
            return current_val
        try:
            pos = valid_indices.index(current_val)
        except ValueError:
            pos = 0
        if ctx.triggered_id == "btn-prev":
            pos = max(pos - 1, 0)
        else:
            pos = min(pos + 1, n_valid - 1)
        return valid_indices[pos]

    @app.callback(
        Output("picks-store", "data"),
        Input("waveform-graph", "clickData"),
        Input("btn-clear", "n_clicks"),
        State("picks-store", "data"),
        State("pair-dropdown", "value"),
        prevent_initial_call=True,
    )
    def handle_click(click_data, n_clear, current_picks, pair_val):
        if current_picks is None:
            current_picks = {}
        triggered = ctx.triggered_id

        if triggered == "btn-clear":
            current_picks.pop(str(pair_val), None)
            _save_picks(current_picks)
            return current_picks

        if triggered == "waveform-graph" and click_data and pair_val is not None:
            x_ms = click_data["points"][0]["x"]
            sample = int(round(float(x_ms) / 1000.0 * tg.sample_rate_hz))
            sample = max(0, min(sample, tg.sample_count - 1))
            current_picks[str(pair_val)] = sample
            _save_picks(current_picks)
        return current_picks

    @app.callback(
        Output("waveform-graph", "figure"),
        Output("progress-label", "children"),
        Output("status-bar", "children"),
        Input("pair-dropdown", "value"),
        Input("picks-store", "data"),
        Input("xlim-input", "value"),
        Input("accel-lo", "value"),
        Input("accel-hi", "value"),
        Input("hydro-lo", "value"),
        Input("hydro-hi", "value"),
        Input("show-traces", "value"),
    )
    def update_figure(pair_val, current_picks, xlim, accel_lo, accel_hi, hydro_lo, hydro_hi, show_traces_val):
        if pair_val is None or tg.n_epochs == 0:
            return go.Figure(), "", "No data."

        if current_picks is None:
            current_picks = {}

        # Build config with the current (possibly UI-adjusted) filter settings
        live_config = MetricConfig(
            clip_first_s=config.clip_first_s,
            mute_first_s=config.mute_first_s,
            hydro_clip_first_s=config.hydro_clip_first_s,
            hydro_mute_first_s=config.hydro_mute_first_s,
            filter_order=config.filter_order,
            accel_filter_low_hz=float(accel_lo or 1.0) * 1000.0,
            accel_filter_high_hz=float(accel_hi or 12.0) * 1000.0,
            hydro_filter_low_hz=float(hydro_lo or 5.0) * 1000.0,
            hydro_filter_high_hz=float(hydro_hi or 20.0) * 1000.0,
        )

        traces = _preprocess_pair(tg, pair_val, live_config)
        pick_s = current_picks.get(str(pair_val), None)
        aic_s = _aic_pick_one(pair_val)

        fig = _build_figure(
            traces=traces,
            dt_s=tg.dt,
            pair_label=pair_labels[pair_val],
            clip_s=config.clip_first_s,
            mute_s=config.mute_first_s,
            hydro_clip_s=config.hydro_clip_first_s,
            hydro_mute_s=config.hydro_mute_first_s,
            pick_sample=pick_s,
            aic_pick_sample=aic_s,
            xlim_ms=float(xlim or 15.0),
            accel_filter_low=float(accel_lo or 1.0) * 1000.0,
            accel_filter_high=float(accel_hi or 12.0) * 1000.0,
            hydro_filter_low=float(hydro_lo or 5.0) * 1000.0,
            hydro_filter_high=float(hydro_hi or 20.0) * 1000.0,
            pair_idx=pair_val,
            n_receivers=tg.n_receivers,
            baseline_indices=baseline_indices,
            show_traces="show" in (show_traces_val or []),
        )

        n_picked = len(current_picks)
        pos = valid_indices.index(pair_val) + 1 if pair_val in valid_indices else "?"
        progress = (
            f"{n_picked} / {n_valid} valid pairs picked  |  pair {pos}/{n_valid}"
            f"  |  baseline: {n_baseline}/{tg.n_epochs} epochs"
        )

        pick_info = (
            f"Manual pick: {float(pick_s) * tg.dt * 1000.0:.4f} ms (sample {pick_s})"
            if pick_s is not None
            else "No manual pick set — click on the waveform to pick.  "
                 "Use scroll/box-select to zoom; double-click to reset zoom."
        )

        return fig, progress, pick_info

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    app = build_app(args)
    print(f"\nCASSM Picker running at http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
