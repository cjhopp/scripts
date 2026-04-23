import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

from obspy import read_events

# HMC coordinate converter (same library used on the mine server)
sys.path.insert(0, "/home/chopp/scripts/python")
try:
    from lbnl.coordinates import SURF_converter
    _SURF = SURF_converter()
except Exception:
    _SURF = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG_FILE = Path("/data/chet-cussp/seismicity/catalog.quakeml")
WELLBORE_DIR = Path("/data/chet-cussp/wellbores")
# Trimesh hull JSON for the 4100L drift (set to None to disable)
HULL_FILE = Path("/data/chet-cussp/seismicity/drift_hull.json")

# HMC axis limits (matches plot_4100)
HMC_XLIM = [1215, 1265]   # Easting [HMC m]
HMC_YLIM = [-905, -855]   # Northing [HMC m]
HMC_ZLIM = [305, 355]     # Elevation [HMC m]

# HMC z of the Earth surface above the 4100L volume (metres).
# Calibrate with: SURF_SURFACE_HMC_Z_M = known_hmc_elev + origin.depth for one event.
# Rough estimate: borehole tops ~355 m HMC + ~1250 m to surface ≈ 1605 m.
SURF_SURFACE_HMC_Z_M = 1605.0

# Auto-refresh interval (milliseconds)
REFRESH_MS = 5 * 60 * 1000   # 5 minutes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_catalog(path):
    """Read a QuakeML file and return a DataFrame with HMC x/y/z/time/mag columns.

    Expects events with hmc_east / hmc_north / hmc_elev stored as extra
    attributes on the preferred origin (as written by the mine push script).
    Returns an empty DataFrame if the file is missing, unreadable, or empty.
    """
    empty = pd.DataFrame(columns=["x", "y", "z", "time", "mag", "hover"])

    if not Path(path).exists():
        log.warning("Catalog not found: %s", path)
        return empty

    try:
        cat = read_events(str(path))
    except Exception as exc:
        log.error("Failed to read catalog %s: %s", path, exc)
        return empty

    rows = []
    for ev in cat:
        try:
            orig = ev.preferred_origin() or ev.origins[0]
            # Prefer HMC coordinates if the push script annotated them
            if hasattr(orig, "extra") and "hmc_east" in orig.extra:
                x = float(orig.extra.hmc_east.value)
                y = float(orig.extra.hmc_north.value)
                z = float(orig.extra.hmc_elev.value)
            elif _SURF is not None:
                # Fallback: convert lat/lon via SURF_converter; z from depth.
                # origin.depth is metres positive-downward (ObsPy / QuakeML convention).
                x, y, _ = _SURF.to_HMC((orig.longitude, orig.latitude, 0.0))
                z = SURF_SURFACE_HMC_Z_M - orig.depth
            else:
                log.debug("No HMC attributes and no SURF_converter — skipping event")
                continue
            mag_obj = ev.preferred_magnitude()
            mag = mag_obj.mag if mag_obj else np.nan
            rows.append(dict(x=x, y=y, z=z, time=orig.time.datetime, mag=mag))
        except (AttributeError, KeyError, TypeError) as exc:
            log.debug("Skipping event: %s", exc)
            continue

    if not rows:
        log.warning("No events with HMC coordinates found in %s", path)
        return empty

    df = pd.DataFrame(rows)
    mag_str = df["mag"].apply(lambda m: f"M{m:.1f}" if pd.notna(m) and np.isfinite(m) else "M?")
    df["hover"] = df["time"].astype(str) + "<br>" + mag_str
    log.info("Loaded %d events from %s", len(df), path)
    return df


def load_wellbores(directory):
    """Load *.csv wellbore trajectories from directory.

    Accepted HMC column sets (case-insensitive):
      • easting_m, northing_m, elevation_m
      • easting, northing, elevation
      • x_m, y_m, z_m

    Well names starting with 'T' are coloured steelblue; all others black
    (matching the plot_4100 convention).

    Returns a dict {name: DataFrame(x, y, z)}.
    """
    wellbores = {}
    wdir = Path(directory)
    if not wdir.exists():
        log.info("Wellbore directory not found: %s", wdir)
        return wellbores

    for csv_file in sorted(wdir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            df.columns = [c.strip().lower() for c in df.columns]
            wdf = None

            if {"easting_m", "northing_m", "elevation_m"}.issubset(df.columns):
                wdf = df.rename(
                    columns={"easting_m": "x", "northing_m": "y", "elevation_m": "z"}
                )[["x", "y", "z"]]
            elif {"easting", "northing", "elevation"}.issubset(df.columns):
                wdf = df.rename(
                    columns={"easting": "x", "northing": "y", "elevation": "z"}
                )[["x", "y", "z"]]
            elif {"x_m", "y_m", "z_m"}.issubset(df.columns):
                wdf = df.rename(
                    columns={"x_m": "x", "y_m": "y", "z_m": "z"}
                )[["x", "y", "z"]]
            elif {"longitude", "latitude"}.issubset(df.columns) and _SURF is not None:
                # lat/lon input — convert horizontals with SURF_converter.
                # Vertical: use 'elevation_m' if present, else depth (positive down).
                rows = []
                for _, row in df.iterrows():
                    ex, ey, _ = _SURF.to_HMC((row["longitude"], row["latitude"], 0.0))
                    if "elevation_m" in df.columns:
                        ez = SURF_SURFACE_HMC_Z_M - (SURF_SURFACE_HMC_Z_M - row["elevation_m"])
                        # elevation_m is already absolute, map to HMC z via offset
                        # HMC z ≈ elevation_m - (known_abs_elev_at_surf - SURF_SURFACE_HMC_Z_M)
                        # Simplification: pass elevation_m directly (same units)
                        ez = row["elevation_m"]
                    elif "depth_m" in df.columns:
                        ez = SURF_SURFACE_HMC_Z_M - row["depth_m"]
                    else:
                        ez = np.nan
                    rows.append({"x": ex, "y": ey, "z": ez})
                wdf = pd.DataFrame(rows)
            else:
                log.warning("Unrecognised columns in %s: %s", csv_file.name, list(df.columns))
                continue

            if wdf is not None:
                wellbores[csv_file.stem] = wdf
        except Exception as exc:
            log.error("Failed to load wellbore CSV %s: %s", csv_file.name, exc)

    log.info("Loaded %d wellbore(s) from %s", len(wellbores), wdir)
    return wellbores


def load_hull(path):
    """Load trimesh JSON hull.  Returns (vertices ndarray, faces ndarray) or (None, None)."""
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        with open(p, "r") as f:
            data = json.load(f)
        vertices = np.array(data["vertices"], dtype=float)
        faces = np.array(data["faces"], dtype=int)
        log.info("Loaded drift hull: %d vertices, %d faces", len(vertices), len(faces))
        return vertices, faces
    except Exception as exc:
        log.warning("Failed to load hull %s: %s", path, exc)
        return None, None


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(cat_df, wellbores, hull_verts, hull_faces, last_updated):
    n_events = len(cat_df)
    fig = go.Figure()

    # Drift hull — semi-transparent mesh
    if hull_verts is not None and len(hull_verts) > 0:
        fig.add_trace(
            go.Mesh3d(
                x=hull_verts[:, 0],
                y=hull_verts[:, 1],
                z=hull_verts[:, 2],
                i=hull_faces[:, 0],
                j=hull_faces[:, 1],
                k=hull_faces[:, 2],
                color="darkgray",
                opacity=0.25,
                name="Drift",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Wellbore traces — T-prefix = steelblue, others = black (plot_4100 convention)
    for name, wdf in wellbores.items():
        color = "steelblue" if name[0].upper() == "T" else "black"
        fig.add_trace(
            go.Scatter3d(
                x=wdf["x"].values,
                y=wdf["y"].values,
                z=wdf["z"].values,
                mode="lines",
                line=dict(color=color, width=3),
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "E=%{x:.1f} m<br>N=%{y:.1f} m<br>Elev=%{z:.1f} m<extra></extra>"
                ),
            )
        )

    # Seismicity scatter — coloured by time, sized by magnitude
    if n_events > 0:
        mag = cat_df["mag"].fillna(0.0)
        sizes = np.clip((mag - mag.min()) ** 2 + 4, 4, 20).values

        t_sec = (
            pd.to_datetime(cat_df["time"]) - pd.to_datetime(cat_df["time"]).min()
        ).dt.total_seconds().values

        fig.add_trace(
            go.Scatter3d(
                x=cat_df["x"].values,
                y=cat_df["y"].values,
                z=cat_df["z"].values,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=t_sec,
                    colorscale="Plasma",
                    colorbar=dict(title="Time →", len=0.45, thickness=12, x=1.0),
                    opacity=0.75,
                ),
                text=cat_df["hover"].values,
                hovertemplate=(
                    "%{text}<br>"
                    "E=%{x:.1f} m, N=%{y:.1f} m, Elev=%{z:.1f} m<extra></extra>"
                ),
                name=f"Seismicity ({n_events})",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Easting [HMC m]", range=HMC_XLIM),
            yaxis=dict(title="Northing [HMC m]", range=HMC_YLIM),
            zaxis=dict(title="Elevation [HMC m]", range=HMC_ZLIM),
            aspectmode="cube",
            bgcolor="white",
        ),
        title=dict(
            text=f"CUSSP 4100L — {n_events} event{'s' if n_events != 1 else ''}",
            font=dict(size=14),
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
        template="plotly_white",
        uirevision="layout",   # preserve camera angle across refreshes
    )
    return fig


# ---------------------------------------------------------------------------
# Panel app
# ---------------------------------------------------------------------------

class SeismicityDashboard(pn.viewable.Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self._wellbores = load_wellbores(WELLBORE_DIR)
        self._hull_verts, self._hull_faces = load_hull(HULL_FILE)
        cat_df, last_updated = self._fetch()
        self._header = pn.pane.Markdown(
            self._header_md(len(cat_df), last_updated),
            sizing_mode="stretch_width",
        )
        self._plot = pn.pane.Plotly(
            build_figure(cat_df, self._wellbores, self._hull_verts, self._hull_faces, last_updated),
            sizing_mode="stretch_both",
            min_height=700,
        )
        pn.state.add_periodic_callback(self._refresh, period=REFRESH_MS)

    @staticmethod
    def _fetch():
        cat_df = load_catalog(CATALOG_FILE)
        last_updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return cat_df, last_updated

    @staticmethod
    def _header_md(n, last_updated):
        return (
            f"**CUSSP 4100L Seismicity** &nbsp;|&nbsp; "
            f"{n} event{'s' if n != 1 else ''} &nbsp;|&nbsp; "
            f"Last updated: {last_updated} &nbsp;*(auto-refreshes every 5 min)*"
        )

    def _refresh(self):
        cat_df, last_updated = self._fetch()
        self._header.object = self._header_md(len(cat_df), last_updated)
        self._plot.object = build_figure(
            cat_df, self._wellbores, self._hull_verts, self._hull_faces, last_updated
        )

    def __panel__(self):
        return pn.Column(
            self._header,
            self._plot,
            sizing_mode="stretch_both",
        )


pn.extension("plotly")

app = SeismicityDashboard()
pn.template.VanillaTemplate(
    title="CUSSP Seismicity",
    logo="/home/chopp/CUSSP.png",
    main=app,
).servable()
