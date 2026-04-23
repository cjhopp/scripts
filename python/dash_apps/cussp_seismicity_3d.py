import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

pn.extension("plotly")

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
HULL_FILE = Path("/data/chet-cussp/seismicity/drift_hull.npy")

# HMC axis limits (matches plot_4100)
HMC_XLIM = [1195, 1275]   # Easting [HMC m]  (+10 m West)
HMC_YLIM = [-935, -845]   # Northing [HMC m]  (+20 m South)
HMC_ZLIM = [295, 365]     # Elevation [HMC m]

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
    """Load wellbore trajectories from directory.

    Accepts two formats:

    1. One_foot_<WELLNAME>_*.csv  (raw SURF as-built files, HMC feet)
       Positional: col 2 = depth(ft), col 3 = easting(ft),
                   col 4 = northing(ft), col 5 = elevation(ft)
       Converted to metres automatically.

    2. Any other *.csv with named HMC columns (already in metres):
       • easting_m, northing_m, elevation_m
       • easting,   northing,   elevation
       • x_m, y_m, z_m
       • longitude, latitude  (+ optional depth_m or elevation_m)

    Well name = filename stem (or embedded name from One_foot files).
    T-prefix wells → steelblue; all others → black.
    """
    wellbores = {}
    wdir = Path(directory)
    if not wdir.exists():
        log.info("Wellbore directory not found: %s", wdir)
        return wellbores

    for csv_file in sorted(wdir.glob("*.csv")):
        try:
            stem = csv_file.stem
            wdf = None

            if stem.startswith("One_foot"):
                # Raw SURF 1-ft trajectory: col 2=depth, 3=east, 4=north, 5=elev (feet)
                arr = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=[2, 3, 4, 5])
                # reorder to [easting, northing, elevation, depth] then convert ft→m
                arr = arr[:, [1, 2, 3, 0]]
                arr[:, :3] *= 0.3048
                wdf = pd.DataFrame(arr[:, :3], columns=["x", "y", "z"])
                # Extract well name the same way make_4100_boreholes does
                parts = stem.split("_")
                well_name = parts[-3] if len(parts) >= 3 else stem
            else:
                df = pd.read_csv(csv_file)
                df.columns = [c.strip().lower() for c in df.columns]
                well_name = stem

                if {"easting_m", "northing_m", "elevation_m"}.issubset(df.columns):
                    wdf = df.rename(columns={"easting_m": "x", "northing_m": "y",
                                             "elevation_m": "z"})[["x", "y", "z"]]
                elif {"easting", "northing", "elevation"}.issubset(df.columns):
                    wdf = df.rename(columns={"easting": "x", "northing": "y",
                                             "elevation": "z"})[["x", "y", "z"]]
                elif {"x_m", "y_m", "z_m"}.issubset(df.columns):
                    wdf = df.rename(columns={"x_m": "x", "y_m": "y",
                                             "z_m": "z"})[["x", "y", "z"]]
                elif {"longitude", "latitude"}.issubset(df.columns) and _SURF is not None:
                    rows = []
                    for _, row in df.iterrows():
                        ex, ey, _ = _SURF.to_HMC((row["longitude"], row["latitude"], 0.0))
                        if "elevation_m" in df.columns:
                            ez = row["elevation_m"]
                        elif "depth_m" in df.columns:
                            ez = SURF_SURFACE_HMC_Z_M - row["depth_m"]
                        else:
                            ez = np.nan
                        rows.append({"x": ex, "y": ey, "z": ez})
                    wdf = pd.DataFrame(rows)
                else:
                    log.warning("Unrecognised columns in %s: %s",
                                csv_file.name, list(df.columns))
                    continue

            if wdf is not None:
                wellbores[well_name] = wdf

        except Exception as exc:
            log.error("Failed to load wellbore CSV %s: %s", csv_file.name, exc)

    log.info("Loaded %d wellbore(s) from %s", len(wellbores), wdir)
    return wellbores

    log.info("Loaded %d wellbore(s) from %s", len(wellbores), wdir)
    return wellbores


def load_hull(path):
    """Load drift hull mesh.  Returns (vertices ndarray, faces ndarray) or (None, None).

    Supported formats (detected by extension):

    • .npy   — numpy archive saved as np.save('hull.npy', {'vertices': V, 'faces': F})
               or a (N,6) array with columns [x,y,z, i,j,k] (vertices + face indices)
    • .csv   — two sections: vertex rows (3 cols x,y,z) then face rows (3 cols i,j,k),
               separated by a blank line; OR a single file with a 'section' column
    • .json  — trimesh-exported JSON with 'vertices' and 'faces' keys
    • .stl/.ply/.obj/.glb/.off — any format supported by trimesh (must be installed)

    Fastest to load at runtime: .npy  ~instant, .stl ~1 s, .json ~10 s for 80 MB.
    To convert the 80 MB JSON once:
        import trimesh, numpy as np
        m = trimesh.load('4100_TriMesh.json')
        np.save('drift_hull.npy', {'vertices': m.vertices, 'faces': m.faces})
        # or: m.export('drift_hull.stl')
    """
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        suffix = p.suffix.lower()

        if suffix == ".npy":
            data = np.load(p, allow_pickle=True).item()
            vertices = np.array(data["vertices"], dtype=float)
            faces = np.array(data["faces"], dtype=int)

        elif suffix == ".csv":
            # Expect two sections separated by a blank line: vertices then faces
            text = p.read_text()
            sections = [s.strip() for s in text.split("\n\n") if s.strip()]
            if len(sections) == 2:
                vertices = np.loadtxt(sections[0].splitlines(), delimiter=",")
                faces = np.loadtxt(sections[1].splitlines(), delimiter=",", dtype=int)
            else:
                # Single CSV: x,y,z,i,j,k per row
                arr = np.loadtxt(p, delimiter=",", skiprows=1)
                vertices = arr[:, :3]
                faces = arr[:, 3:].astype(int)

        elif suffix == ".json":
            import base64 as _b64
            with open(p, "r") as f:
                data = json.load(f)
            def _decode(sub):
                if isinstance(sub, dict) and "base64" in sub:
                    return np.frombuffer(_b64.b64decode(sub["base64"]),
                                         dtype=sub["dtype"]).reshape(sub["shape"])
                return np.array(sub)
            vertices = _decode(data["vertices"]).astype(float)
            faces = _decode(data["faces"]).astype(int)

        else:
            # trimesh handles STL, PLY, OBJ, GLB, OFF, etc.
            import trimesh as _trimesh
            obj = _trimesh.load(str(p), force="mesh")
            if hasattr(obj, "geometry"):
                obj = max(obj.geometry.values(), key=lambda m: len(m.faces))
            vertices = np.array(obj.vertices, dtype=float)
            faces = np.array(obj.faces, dtype=int)

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


def build_magnitude_figure(cat_df):
    """Magnitude vs time scatter plot."""
    fig = go.Figure()
    if len(cat_df) > 0:
        times = pd.to_datetime(cat_df["time"])
        mag = cat_df["mag"].fillna(float("nan"))
        fig.add_trace(go.Scatter(
            x=times,
            y=mag,
            mode="markers",
            marker=dict(
                size=6,
                color=mag,
                colorscale="Plasma",
                cmin=-5,
                cmax=0,
                showscale=False,
            ),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>M%{y:.2f}<extra></extra>",
            name="Magnitude",
        ))
    fig.update_layout(
        height=440,
        margin=dict(l=60, r=20, t=30, b=40),
        template="plotly_white",
        title=dict(text="Magnitude", font=dict(size=12)),
        yaxis=dict(title="M", range=[-5, 0]),
        xaxis=dict(title=""),
        uirevision="mag",
    )
    return fig


def build_injection_figure(inj_df=None):
    """Injection parameters vs time.  Placeholder until data is provided."""
    fig = go.Figure()
    if inj_df is not None and len(inj_df) > 0:
        for col in [c for c in inj_df.columns if c != "time"]:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(inj_df["time"]),
                y=inj_df[col],
                mode="lines",
                name=col,
            ))
    else:
        fig.add_annotation(
            text="Injection data not yet available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="gray"),
        )
    fig.update_layout(
        height=440,
        margin=dict(l=60, r=20, t=30, b=40),
        template="plotly_white",
        title=dict(text="Injection Parameters", font=dict(size=12)),
        xaxis=dict(title="Time"),
        uirevision="inj",
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
            sizing_mode="stretch_width",
            height=750,
        )
        self._mag_plot = pn.pane.Plotly(
            build_magnitude_figure(cat_df),
            sizing_mode="stretch_width",
        )
        self._inj_plot = pn.pane.Plotly(
            build_injection_figure(),
            sizing_mode="stretch_width",
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
        self._mag_plot.object = build_magnitude_figure(cat_df)

    def __panel__(self):
        return pn.Column(
            self._header,
            self._plot,
            self._mag_plot,
            self._inj_plot,
            sizing_mode="stretch_width",
        )


app = SeismicityDashboard()
pn.template.VanillaTemplate(
    title="CUSSP Seismicity",
    logo="/CUSSP.png",
    main=app,
).servable()
