import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
import pyproj

from obspy import read_events

# ---------------------------------------------------------------------------
# Configuration — adjust to match site layout
# ---------------------------------------------------------------------------
CATALOG_FILE = Path("/data/chet-cussp/seismicity/catalog.quakeml")
WELLBORE_DIR = Path("/data/chet-cussp/wellbores")

# UTM Zone 13N (South Dakota / SURF)
_PROJ = pyproj.Proj("EPSG:26913")

# Local coordinate reference: approximate centre of the 4100L experiment volume
# All x/y/z in the plot are metres relative to this point.
REF_LAT = 44.3517
REF_LON = -103.7508
REF_ELEV_M = -1250.0      # ~4100 ft below surface in metres (negative = underground)

_REF_E, _REF_N = _PROJ(REF_LON, REF_LAT)

# Auto-refresh interval (milliseconds)
REFRESH_MS = 5 * 60 * 1000   # 5 minutes

# Wellbore line colours (cycled)
_WELL_COLOURS = ["steelblue", "firebrick", "darkgreen", "darkorchid", "orange", "saddlebrown"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def latlon_depth_to_local(lats, lons, depths_km):
    """Convert WGS84 lat/lon + depth-km to local XYZ (metres).

    x = Easting  − reference Easting
    y = Northing − reference Northing
    z = −depth_km * 1000   (positive up from REF_ELEV_M)
    """
    e, n = _PROJ(np.asarray(lons), np.asarray(lats))
    x = e - _REF_E
    y = n - _REF_N
    z = -np.asarray(depths_km) * 1000.0
    return x, y, z


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_catalog(path):
    """Read a QuakeML file and return a DataFrame with x/y/z/time/mag columns.

    Returns an empty DataFrame (with the right columns) if the file is missing
    or unparseable — so the dashboard still renders.
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
            mag_obj = ev.preferred_magnitude()
            mag = mag_obj.mag if mag_obj else np.nan
            rows.append(
                dict(
                    lat=orig.latitude,
                    lon=orig.longitude,
                    depth_km=orig.depth / 1000.0,
                    time=orig.time.datetime,
                    mag=mag,
                )
            )
        except (IndexError, AttributeError):
            continue

    if not rows:
        return empty

    df = pd.DataFrame(rows)
    df["x"], df["y"], df["z"] = latlon_depth_to_local(
        df["lat"].values, df["lon"].values, df["depth_km"].values
    )
    mag_str = df["mag"].apply(lambda m: f"M{m:.1f}" if np.isfinite(m) else "Munk")
    df["hover"] = df["time"].astype(str) + "<br>" + mag_str
    return df


def load_wellbores(directory):
    """Load all *.csv files in directory as wellbore trajectories.

    Accepted column sets (case-insensitive, stripped):
      • x_m, y_m, z_m          — already in local mine coordinates (metres)
      • easting_m, northing_m, elevation_m  — UTM Zone 13N; shifted by reference point

    The filename stem becomes the borehole name in the legend.
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

            if {"x_m", "y_m", "z_m"}.issubset(df.columns):
                wellbores[csv_file.stem] = df.rename(
                    columns={"x_m": "x", "y_m": "y", "z_m": "z"}
                )[["x", "y", "z"]]

            elif {"easting_m", "northing_m", "elevation_m"}.issubset(df.columns):
                wdf = pd.DataFrame()
                wdf["x"] = df["easting_m"] - _REF_E
                wdf["y"] = df["northing_m"] - _REF_N
                wdf["z"] = df["elevation_m"] - REF_ELEV_M
                wellbores[csv_file.stem] = wdf

            else:
                log.warning("Unrecognised column layout in %s: %s", csv_file.name, list(df.columns))
        except Exception as exc:
            log.error("Failed to load wellbore CSV %s: %s", csv_file.name, exc)

    log.info("Loaded %d wellbore(s) from %s", len(wellbores), wdir)
    return wellbores


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(cat_df, wellbores, last_updated):
    fig = go.Figure()

    # Wellbore traces
    for i, (name, wdf) in enumerate(wellbores.items()):
        colour = _WELL_COLOURS[i % len(_WELL_COLOURS)]
        fig.add_trace(
            go.Scatter3d(
                x=wdf["x"],
                y=wdf["y"],
                z=wdf["z"],
                mode="lines",
                line=dict(color=colour, width=4),
                name=name,
                hovertemplate=f"<b>{name}</b><br>x=%{{x:.1f}} m<br>y=%{{y:.1f}} m<br>z=%{{z:.1f}} m<extra></extra>",
            )
        )

    # Earthquake scatter
    n_events = len(cat_df)
    if n_events > 0:
        mag = cat_df["mag"].fillna(1.0)
        sizes = np.clip(3 + 3 * mag, 3, 18).values

        # Colour by time (seconds since earliest event)
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
                    colorbar=dict(title="Time →", len=0.5, thickness=12),
                    opacity=0.75,
                ),
                text=cat_df["hover"].values,
                hovertemplate="%{text}<br>x=%{x:.1f} m, y=%{y:.1f} m, z=%{z:.1f} m<extra></extra>",
                name=f"Seismicity ({n_events} events)",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="Elevation (m)",
            aspectmode="data",
            bgcolor="white",
        ),
        title=dict(
            text=f"CUSSP 4100L — {n_events} events",
            font=dict(size=14),
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
        template="plotly_white",
        uirevision="layout",   # preserve camera angle across data refreshes
    )
    return fig


# ---------------------------------------------------------------------------
# Panel app
# ---------------------------------------------------------------------------

class SeismicityDashboard(pn.viewable.Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        self._wellbores = load_wellbores(WELLBORE_DIR)
        cat_df, last_updated = self._fetch()
        self._header = pn.pane.Markdown(
            self._header_md(len(cat_df), last_updated),
            sizing_mode="stretch_width",
        )
        self._plot = pn.pane.Plotly(
            build_figure(cat_df, self._wellbores, last_updated),
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
            f"**CUSSP EGS Collab 4100L** &nbsp;|&nbsp; "
            f"{n} events &nbsp;|&nbsp; "
            f"Last updated: {last_updated} &nbsp;*(auto-refreshes every 5 min)*"
        )

    def _refresh(self):
        cat_df, last_updated = self._fetch()
        self._header.object = self._header_md(len(cat_df), last_updated)
        self._plot.object = build_figure(cat_df, self._wellbores, last_updated)

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
