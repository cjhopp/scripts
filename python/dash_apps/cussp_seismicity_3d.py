import json
import logging
import math
import io
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Proj, Transformer

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
STATION_FILE = Path("/data/chet-cussp/seismicity/stations_hmc.csv")
WELLBORE_DIR = Path("/data/chet-cussp/wellbores")
# Trimesh hull JSON for the 4100L drift (set to None to disable)
HULL_FILE = Path("/data/chet-cussp/seismicity/drift_hull.npy")
INJ_LIVE_DIR = Path('/data/chet-cussp/injection/live')
SNAP_STATIONS_TO_WELLS = True

# Canonical plotting frame for seismicity/stations exported by push pipeline.
CANONICAL_FRAME = "hmc-grid-nad27-utm13"
# Set source frame for engineering products.
# Use "hmc-true-north" only if survey metadata explicitly says true-north axes.
WELLBORE_SOURCE_FRAME = "hmc-grid-nad27-utm13"
HULL_SOURCE_FRAME = "hmc-grid-nad27-utm13"
# Reference point used to compute meridian convergence of the UTM grid.
FRAME_REF_LAT_WGS84 = 44.35105719
FRAME_REF_LON_WGS84 = -103.75035647

# HMC axis limits (matches plot_4100)
HMC_XLIM = [1195, 1275]   # Easting [HMC m]  (+10 m West)
HMC_YLIM = [-935, -845]   # Northing [HMC m]  (+20 m South)
HMC_ZLIM = [295, 365]     # Elevation [HMC m]

# HMC z of the Earth surface above the 4100L volume (metres).
# Calibrate with: SURF_SURFACE_HMC_Z_M = known_hmc_elev + origin.depth for one event.
# Rough estimate: borehole tops ~355 m HMC + ~1250 m to surface ≈ 1605 m.
SURF_SURFACE_HMC_Z_M = 1605.0

# Auto-refresh interval (milliseconds)
REFRESH_MS = 1 * 60 * 1000   # 1 minute
REFRESH_LABEL = "1 min"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _grid_convergence_deg(lon_wgs84, lat_wgs84):
    """Return UTM13 meridian convergence (degrees) at a WGS84 location.

    This is a deterministic projection property that maps true north to grid north.
    """
    wgs84_to_nad27 = Transformer.from_crs("EPSG:4326", "EPSG:4267", always_xy=True)
    lon27, lat27 = wgs84_to_nad27.transform(float(lon_wgs84), float(lat_wgs84))
    proj = Proj("EPSG:26713")
    factors = proj.get_factors(lon27, lat27)
    return float(factors.meridian_convergence)


GRID_CONVERGENCE_DEG = _grid_convergence_deg(FRAME_REF_LON_WGS84, FRAME_REF_LAT_WGS84)
_WGS84_TO_NAD27_GEO = Transformer.from_crs("EPSG:4326", "EPSG:4267", always_xy=True)


def _rotate_xy(x, y, angle_deg):
    theta = math.radians(float(angle_deg))
    ct = math.cos(theta)
    st = math.sin(theta)
    xr = ct * x - st * y
    yr = st * x + ct * y
    return xr, yr


def _transform_xy_to_canonical(x, y, source_frame):
    """Map source XY into CANONICAL_FRAME using projection-derived math only."""
    if source_frame == CANONICAL_FRAME:
        return x, y

    if source_frame == "hmc-true-north" and CANONICAL_FRAME == "hmc-grid-nad27-utm13":
        # True-north frame -> UTM grid frame: rotate by +meridian convergence.
        return _rotate_xy(x, y, GRID_CONVERGENCE_DEG)

    raise ValueError(f"Unsupported frame conversion: {source_frame} -> {CANONICAL_FRAME}")


def _transform_df_xy_to_canonical(df, source_frame):
    if len(df) == 0:
        return df
    x, y = _transform_xy_to_canonical(df["x"].to_numpy(float), df["y"].to_numpy(float), source_frame)
    out = df.copy()
    out["x"] = x
    out["y"] = y
    return out


def _transform_vertices_xy_to_canonical(vertices, source_frame):
    if vertices is None or len(vertices) == 0:
        return vertices
    x, y = _transform_xy_to_canonical(vertices[:, 0], vertices[:, 1], source_frame)
    out = np.array(vertices, copy=True)
    out[:, 0] = x
    out[:, 1] = y
    return out


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

    def _extract_event_magnitude(event):
        mag_obj = event.preferred_magnitude()
        if mag_obj is not None and getattr(mag_obj, "mag", None) is not None:
            return float(mag_obj.mag)
        for candidate in getattr(event, "magnitudes", []) or []:
            if getattr(candidate, "mag", None) is not None:
                return float(candidate.mag)
        return np.nan

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
                lon27, lat27 = _WGS84_TO_NAD27_GEO.transform(orig.longitude, orig.latitude)
                x, y, _ = _SURF.to_HMC((lon27, lat27, 0.0))
                z = SURF_SURFACE_HMC_Z_M - orig.depth
            else:
                log.debug("No HMC attributes and no SURF_converter — skipping event")
                continue
            mag = _extract_event_magnitude(ev)
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
                        lon27, lat27 = _WGS84_TO_NAD27_GEO.transform(
                            float(row["longitude"]), float(row["latitude"])
                        )
                        ex, ey, _ = _SURF.to_HMC((lon27, lat27, 0.0))
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
                wdf = _transform_df_xy_to_canonical(wdf, WELLBORE_SOURCE_FRAME)
                wellbores[well_name] = wdf

        except Exception as exc:
            log.error("Failed to load wellbore CSV %s: %s", csv_file.name, exc)

    log.info("Loaded %d wellbore(s) from %s", len(wellbores), wdir)
    return wellbores


def load_stations(path):
    """Load station/channel HMC CSV exported by cussp_push_catalog.py."""
    p = Path(path)
    empty = pd.DataFrame(columns=["x", "y", "z", "label"])

    if not p.exists():
        log.warning("Station CSV not found: %s", p)
        return empty

    try:
        df = pd.read_csv(p)
    except Exception as exc:
        log.error("Failed to read station CSV %s: %s", p, exc)
        return empty

    required = {"hmc_east_m", "hmc_north_m"}
    if not required.issubset(df.columns):
        log.warning("Station CSV missing required columns: %s", sorted(required))
        return empty

    z_col = "hmc_z_minus_depth_m" if "hmc_z_minus_depth_m" in df.columns else "hmc_z_m_asl"
    if z_col not in df.columns:
        log.warning("Station CSV missing Z column (hmc_z_minus_depth_m or hmc_z_m_asl)")
        return empty

    out = pd.DataFrame(
        {
            "x": pd.to_numeric(df["hmc_east_m"], errors="coerce"),
            "y": pd.to_numeric(df["hmc_north_m"], errors="coerce"),
            "z": pd.to_numeric(df[z_col], errors="coerce"),
            "network": df.get("network", ""),
            "station": df.get("station", ""),
            "channel": df.get("channel", ""),
        }
    ).dropna(subset=["x", "y", "z"])

    out["label"] = (
        out["network"].astype(str)
        + "."
        + out["station"].astype(str)
        + "."
        + out["channel"].astype(str)
    )

    # Plot one point per station to avoid dense per-channel overlap.
    out = out.sort_values(["label"]).drop_duplicates(subset=["network", "station"], keep="first")
    log.info("Loaded %d station point(s) from %s", len(out), p)
    return out[["x", "y", "z", "label"]]


def snap_stations_to_wellbores(station_df, wellbores):
    """Snap each station point to its nearest borehole sample point.

    This is an empirical display alignment step used when exact visual overlap
    with borehole trajectories is required.
    """
    if len(station_df) == 0 or not wellbores:
        return station_df

    wells = []
    for name, wdf in wellbores.items():
        if len(wdf) == 0:
            continue
        arr = wdf[["x", "y", "z"]].to_numpy(float)
        wells.append((name, arr))

    if not wells:
        return station_df

    rows = []
    for _, sta in station_df.iterrows():
        s = np.array([float(sta["x"]), float(sta["y"]), float(sta["z"])])
        best_d = np.inf
        best_p = None
        best_name = ""
        for name, warr in wells:
            d = np.sqrt(((warr - s) ** 2).sum(axis=1))
            i = int(np.argmin(d))
            if d[i] < best_d:
                best_d = float(d[i])
                best_p = warr[i]
                best_name = name
        rows.append(
            {
                "x": float(best_p[0]),
                "y": float(best_p[1]),
                "z": float(best_p[2]),
                "label": f"{sta['label']} [snapped:{best_name}]",
            }
        )

    snapped = pd.DataFrame(rows)
    log.info("Snapped %d station(s) to nearest well points", len(snapped))
    return snapped


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
                    return np.frombuffer(_b64.b64decode(sub["base64"]), dtype=sub["dtype"]).reshape(sub["shape"])
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

        vertices = _transform_vertices_xy_to_canonical(vertices, HULL_SOURCE_FRAME)
        log.info("Loaded drift hull: %d vertices, %d faces", len(vertices), len(faces))
        return vertices, faces
    except Exception as exc:
        log.warning("Failed to load hull %s: %s", path, exc)
        return None, None


def _resolve_metadata_path(data_path):
    return data_path.with_name(data_path.name.replace('data', 'metadata'))


def _parse_injection_metadata(metadata_path):
    """Return a best-effort mapping of column -> unit from metadata CSV."""
    if metadata_path is None or not metadata_path.exists():
        return {}
    try:
        meta_df = pd.read_csv(metadata_path, nrows=1, low_memory=False)
        if meta_df.empty:
            return {}
        row = meta_df.iloc[0]
        units = {}
        for col, val in row.items():
            if pd.notna(val) and str(val).strip():
                units[col] = str(val).strip()
        return units
    except Exception:
        return {}


def _find_latest_injection_pair(live_dir):
    latest_data = live_dir / 'latest_INJ_data.csv'
    latest_meta = live_dir / 'latest_INJ_metadata.csv'
    if latest_data.exists() and latest_meta.exists():
        return latest_data, latest_meta

    data_files = sorted(live_dir.glob('*INJ_data.csv'))
    if not data_files:
        return None, None
    latest = max(data_files, key=lambda p: p.stat().st_mtime)
    metadata = _resolve_metadata_path(latest)
    if not metadata.exists():
        return latest, None
    return latest, metadata


def _date_from_injection_filename(data_path):
    """Extract base date from filenames like CUSSP2026_05_08.INJ_data.csv."""
    m = re.search(r'(\d{4})_(\d{2})_(\d{2})', data_path.name)
    if not m:
        return None
    try:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
    except ValueError:
        return None


def _parse_time_series(series, data_path):
    """Parse a candidate time series with multiple fallbacks."""
    time_raw = series.astype(str).str.strip()
    parsed_time = pd.to_datetime(
        time_raw,
        format='%m/%d/%y %H:%M:%S',
        errors='coerce',
    )
    if parsed_time.isna().all():
        parsed_time = pd.to_datetime(time_raw, errors='coerce')
    if parsed_time.isna().all():
        serial = pd.to_numeric(time_raw, errors='coerce')
        parsed_time = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')

        frac_mask = serial.notna() & (serial >= 0) & (serial < 2)
        file_date = _date_from_injection_filename(data_path)
        if frac_mask.any() and file_date is not None:
            parsed_time.loc[frac_mask] = file_date + pd.to_timedelta(serial[frac_mask], unit='D')

        excel_mask = serial.notna() & (serial >= 20000)
        if excel_mask.any():
            parsed_time.loc[excel_mask] = pd.Timestamp('1899-12-30') + pd.to_timedelta(
                serial[excel_mask], unit='D'
            )

        parsed_time = pd.DatetimeIndex(parsed_time)
    return pd.Series(parsed_time, index=series.index)


def _choose_time_column(df, data_path):
    """Choose the most likely time column by parse success rate."""
    candidate_cols = []
    if 'Time' in df.columns:
        candidate_cols.append('Time')
    for col in df.columns:
        lc = str(col).lower()
        if col not in candidate_cols and ('time' in lc or lc.startswith('unnamed')):
            candidate_cols.append(col)
    if not candidate_cols:
        candidate_cols = list(df.columns)

    best_col = None
    best_parsed = None
    best_score = -1.0
    for col in candidate_cols:
        parsed = _parse_time_series(df[col], data_path)
        score = float(parsed.notna().mean())
        if score > best_score:
            best_score = score
            best_col = col
            best_parsed = parsed

    if best_col is not None and best_col != 'Time':
        log.warning("Using '%s' as injection time column (instead of 'Time')", best_col)
    return best_col, best_parsed


def load_injection_dataframe(
    live_dir=INJ_LIVE_DIR,
    pressure_col='PT 503',
    flow_col='Net Flow',
    filename='latest_INJ_data_1min.csv',
):
    """Load injection CSV pair and return dataframe + labels."""
    data_path = live_dir / filename
    if not data_path.exists():
        if filename != 'latest_INJ_data.csv':
            return load_injection_dataframe(
                live_dir=live_dir,
                pressure_col=pressure_col,
                flow_col=flow_col,
                filename='latest_INJ_data.csv',
            )
        log.warning("No injection data file found at %s", data_path)
        return None, None

    if filename == 'latest_INJ_data_1min.csv':
        metadata_path = live_dir / 'latest_INJ_metadata.csv'
    elif filename.endswith('_data.csv'):
        metadata_path = _resolve_metadata_path(data_path)
    else:
        metadata_path = None

    try:
        with open(data_path, 'r') as f:
            lines = [line.rstrip('\r\n').rstrip(',') + '\n' for line in f]
        csv_text = ''.join(lines)
    except Exception as exc:
        log.warning("Failed to read injection file %s: %s", data_path, exc)
        return None, None

    attempts = [
        ('strict-skiprows', dict(skiprows=[1, 2], low_memory=False)),
        ('flexible', dict(low_memory=False)),
    ]

    chosen_df = None
    chosen_parsed_time = None
    chosen_mode = None
    chosen_score = -1.0

    for mode, kwargs in attempts:
        try:
            df_try = pd.read_csv(io.StringIO(csv_text), **kwargs)
        except Exception as exc:
            log.warning("Injection read mode %s failed for %s: %s", mode, data_path, exc)
            continue

        df_try.columns = [str(c).strip().replace('\ufeff', '') for c in df_try.columns]
        _, parsed_try = _choose_time_column(df_try, data_path)
        if parsed_try is None:
            continue

        score = float(parsed_try.notna().mean())
        if score > chosen_score:
            chosen_df = df_try
            chosen_parsed_time = parsed_try
            chosen_mode = mode
            chosen_score = score

        if mode == 'strict-skiprows' and score >= 0.80:
            break

    if chosen_df is None or chosen_parsed_time is None:
        log.warning("Injection file %s has no parsable time column", data_path)
        return None, None

    log.info(
        'Injection CSV parse mode=%s file=%s valid_time_fraction=%.3f',
        chosen_mode,
        filename,
        chosen_score,
    )

    df = chosen_df
    df['Time'] = chosen_parsed_time
    df = df.dropna(subset=['Time'])
    if df.empty:
        log.warning("Injection file %s has no valid timestamp rows", data_path)
        return None, None

    if pressure_col not in df.columns:
        log.warning("Injection file %s has no pressure column '%s'", data_path, pressure_col)
        return None, None
    if flow_col not in df.columns:
        log.warning("Injection file %s has no flow column '%s'", data_path, flow_col)
        return None, None

    df[pressure_col] = pd.to_numeric(df[pressure_col], errors='coerce')
    df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce')
    df = df.dropna(subset=[pressure_col, flow_col]).sort_values('Time')
    if df.empty:
        log.warning("Injection file %s has no numeric pressure/flow rows", data_path)
        return None, None

    units = _parse_injection_metadata(metadata_path) if metadata_path else {}
    labels = {
        'pressure_col': pressure_col,
        'flow_col': flow_col,
        'pressure_unit': units.get(pressure_col, 'psi'),
        'flow_unit': units.get(flow_col, 'L/min'),
    }
    return df, labels


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(cat_df, station_df, wellbores, hull_verts, hull_faces, last_updated):
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
        mag_raw = cat_df["mag"]
        has_mag = mag_raw.notna()
        # Fill NaN with the minimum real magnitude so unknown events get the
        # *smallest* marker; without this, NaN→0 sits above all negative real
        # magnitudes and gets the maximum size (backwards).
        if has_mag.any():
            mag = mag_raw.fillna(float(mag_raw.min()))
        else:
            mag = mag_raw.fillna(0.0)
        sizes = np.clip(1.05 * (mag - mag.min()) ** 2 + 3.1, 3.1, 14).values

        t_datetimes = pd.to_datetime(cat_df["time"])
        t_min_dt = t_datetimes.min()
        t_sec = (t_datetimes - t_min_dt).dt.total_seconds().values
        t_max_sec = float(t_sec.max()) if t_sec.max() > 0 else 1.0

        n_ticks = 5
        tick_sec_vals = np.linspace(0.0, t_max_sec, n_ticks)
        tick_labels = [
            (t_min_dt + pd.Timedelta(seconds=float(ts))).strftime("%Y-%m-%d\n%H:%M UTC")
            for ts in tick_sec_vals
        ]

        fig.add_trace(
            go.Scatter3d(
                x=cat_df["x"].values,
                y=cat_df["y"].values,
                z=cat_df["z"].values,
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=t_sec,
                    colorscale=[
                        [0.0, "rgb(135,135,135)"],
                        [1.0, "rgb(20,20,20)"],
                    ],
                    cmin=0.0,
                    cmax=t_max_sec,
                    colorbar=dict(
                        title="",
                        len=0.45,
                        thickness=12,
                        x=1.0,
                        tickvals=tick_sec_vals.tolist(),
                        ticktext=tick_labels,
                    ),
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

    # Station markers from SeisComP inventory export
    if len(station_df) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=station_df["x"].values,
                y=station_df["y"].values,
                z=station_df["z"].values,
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    color="red",
                    size=3,
                    opacity=0.5,
                ),
                text=station_df["label"].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "E=%{x:.1f} m, N=%{y:.1f} m, Elev=%{z:.1f} m<extra></extra>"
                ),
                name=f"Stations ({len(station_df)})",
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


def build_time_panels_figure(cat_df, inj_df=None, inj_labels=None, date_range=None):
    """Build combined magnitude + injection panels with shared x-axis."""
    # Keep these in one Plotly figure so x-axis pan/zoom is natively linked,
    # mirroring the paired time-panel behavior used in fiboreglass.
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=("Magnitude", "Injection Parameters"),
    )

    # Top row: magnitude scatter
    mag_df = cat_df[cat_df["mag"].notna()].copy() if len(cat_df) > 0 else pd.DataFrame()
    if len(mag_df) > 0:
        times = pd.to_datetime(mag_df["time"])
        mag = mag_df["mag"].astype(float)
        mag_df = mag_df.assign(time_dt=times).sort_values("time_dt")
        times = mag_df["time_dt"]
        mag = mag_df["mag"].astype(float)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=mag,
                mode="markers",
                marker=dict(
                    size=10,
                    color=mag,
                    colorscale="Plasma",
                    cmin=float(mag.min()),
                    cmax=float(mag.max()) if float(mag.max()) > float(mag.min()) else float(mag.min()) + 1.0,
                    showscale=False,
                    opacity=0.95,
                    line=dict(color="rgba(20,20,20,0.7)", width=0.8),
                ),
                hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>M%{y:.2f}<extra></extra>",
                name="Magnitude",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No magnitudes in selected time window",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12, color="gray"),
            row=1,
            col=1,
        )

    # Bottom row: injection pressure + flow (multi y)
    if inj_df is not None and inj_labels is not None and len(inj_df) > 0:
        pressure_col = inj_labels["pressure_col"]
        flow_col = inj_labels["flow_col"]
        pressure_unit = inj_labels["pressure_unit"]
        flow_unit = inj_labels["flow_unit"]

        df = inj_df.copy()
        times = pd.to_datetime(df["Time"])
        if date_range is not None:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            mask = (times >= start) & (times <= end)
            df = df.loc[mask]
            times = times.loc[mask]

        if len(df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=df[pressure_col],
                    mode="lines",
                    name=f"{pressure_col} [{pressure_unit}]",
                    line=dict(color="firebrick", width=2),
                    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Pressure=%{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=df[flow_col],
                    mode="lines",
                    name=f"{flow_col} [{flow_unit}]",
                    line=dict(color="steelblue", width=2),
                    hovertemplate="%{x|%Y-%m-%d %H:%M:%S}<br>Flow=%{y:.2f}<extra></extra>",
                ),
                row=2,
                col=1,
                secondary_y=True,
            )
        else:
            fig.add_annotation(
                text="No injection data in selected time window",
                xref="x2 domain",
                yref="y2 domain",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=13, color="gray"),
            )
    else:
        fig.add_annotation(
            text="Injection data not yet available",
            xref="x2 domain",
            yref="y2 domain",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13, color="gray"),
        )

    if date_range is not None:
        fig.update_xaxes(range=[pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])], row=1, col=1)
        fig.update_xaxes(range=[pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])], row=2, col=1)

    fig.update_layout(
        height=900,
        margin=dict(l=60, r=60, t=40, b=40),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        uirevision="time-panels",
    )
    fig.update_xaxes(title="Time", row=2, col=1)
    fig.update_yaxes(title="M", autorange=True, showgrid=True, zeroline=True, row=1, col=1)
    if inj_labels is not None:
        fig.update_yaxes(title_text=f"Pressure [{inj_labels['pressure_unit']}]", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text=f"Flow [{inj_labels['flow_unit']}]", row=2, col=1, secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Panel app
# ---------------------------------------------------------------------------

class SeismicityDashboard(pn.viewable.Viewer):
    def __init__(self, **params):
        super().__init__(**params)
        log.info(
            "Frame setup: canonical=%s, wellbores=%s, hull=%s, grid convergence=%.6f deg",
            CANONICAL_FRAME,
            WELLBORE_SOURCE_FRAME,
            HULL_SOURCE_FRAME,
            GRID_CONVERGENCE_DEG,
        )
        self._wellbores = load_wellbores(WELLBORE_DIR)
        self._stations = load_stations(STATION_FILE)
        if SNAP_STATIONS_TO_WELLS:
            self._stations = snap_stations_to_wellbores(self._stations, self._wellbores)
        self._hull_verts, self._hull_faces = load_hull(HULL_FILE)
        self._inj_data_path = None
        self._inj_mtime = None
        self._inj_df = None
        self._inj_labels = None
        self._refresh_injection_data()
        cat_df, last_updated = self._fetch()
        self._cat_df_full = cat_df

        min_t, max_t = self._catalog_time_bounds(cat_df)
        self._date_start = pn.widgets.DatePicker(
            name="Start date",
            value=min_t.date(),
        )
        self._date_end = pn.widgets.DatePicker(
            name="End date",
            value=max_t.date(),
        )
        self._time_start = pn.widgets.TextInput(
            name="Start time (UTC)",
            value=min_t.strftime("%H:%M"),
            width=120,
        )
        self._time_end = pn.widgets.TextInput(
            name="End time (UTC)",
            value=max_t.strftime("%H:%M"),
            width=120,
        )
        self._date_start.param.watch(self._on_slider_change, "value")
        self._date_end.param.watch(self._on_slider_change, "value")
        self._time_start.param.watch(self._on_slider_change, "value")
        self._time_end.param.watch(self._on_slider_change, "value")

        cat_filtered = self._apply_date_filter(cat_df)
        self._header = pn.pane.Markdown(
            self._header_md(len(cat_filtered), last_updated, n_total=len(cat_df)),
            sizing_mode="stretch_width",
        )
        self._plot = pn.pane.Plotly(
            build_figure(
                cat_filtered,
                self._stations,
                self._wellbores,
                self._hull_verts,
                self._hull_faces,
                last_updated,
            ),
            sizing_mode="stretch_width",
            height=750,
        )
        self._mag_plot = pn.pane.Plotly(
            build_time_panels_figure(cat_filtered, self._inj_df, self._inj_labels, date_range=self._date_range()),
            sizing_mode="stretch_width",
        )
        pn.state.add_periodic_callback(self._refresh, period=REFRESH_MS)

    @staticmethod
    def _fetch():
        cat_df = load_catalog(CATALOG_FILE)
        last_updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return cat_df, last_updated

    @staticmethod
    def _header_md(n_visible, last_updated, n_total=None):
        if n_total is not None and n_total != n_visible:
            count_str = f"{n_visible} of {n_total} event{'s' if n_total != 1 else ''}"
        else:
            count_str = f"{n_visible} event{'s' if n_visible != 1 else ''}"
        return (
            f"**CUSSP 4100L Seismicity** &nbsp;|&nbsp; "
            f"{count_str} &nbsp;|&nbsp; "
            f"Last updated: {last_updated} &nbsp;*(auto-refreshes every {REFRESH_LABEL})*"
        )

    @staticmethod
    def _catalog_time_bounds(df):
        """Return (min_datetime, max_datetime) as naive Python datetimes for slider bounds."""
        now = datetime.utcnow()
        if len(df) == 0:
            return now - timedelta(days=30), now
        t = pd.to_datetime(df["time"])
        if t.dt.tz is not None:
            t = t.dt.tz_convert(None)
        return t.min().to_pydatetime(), max(t.max().to_pydatetime(), now)

    @staticmethod
    def _parse_time_text(value, default_time):
        text = (value or "").strip()
        try:
            parsed = datetime.strptime(text, "%H:%M")
            return parsed.strftime("%H:%M")
        except ValueError:
            return default_time

    def _date_range(self):
        """Return (start, end) as Timestamps from date and time filter widgets."""
        start_time = self._parse_time_text(self._time_start.value, "00:00")
        end_time = self._parse_time_text(self._time_end.value, "23:59")
        start = pd.Timestamp(f"{self._date_start.value} {start_time}")
        end = pd.Timestamp(f"{self._date_end.value} {end_time}")
        if end < start:
            start, end = end, start
        return start, end

    def _apply_date_filter(self, df):
        """Filter df to the current picker selection."""
        if len(df) == 0:
            return df
        start, end = self._date_range()
        t = pd.to_datetime(df["time"])
        if t.dt.tz is not None:
            t = t.dt.tz_convert(None)
        mask = (t >= start) & (t <= end)
        return df[mask].reset_index(drop=True)

    def _on_slider_change(self, event):
        cat_filtered = self._apply_date_filter(self._cat_df_full)
        self._header.object = self._header_md(len(cat_filtered), datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), n_total=len(self._cat_df_full))
        self._plot.object = build_figure(
            cat_filtered,
            self._stations,
            self._wellbores,
            self._hull_verts,
            self._hull_faces,
            None,
        )
        self._mag_plot.object = build_time_panels_figure(cat_filtered, self._inj_df, self._inj_labels, date_range=self._date_range())

    def _refresh_injection_data(self):
        data_path, _ = _find_latest_injection_pair(INJ_LIVE_DIR)
        if data_path is None:
            self._inj_df = None
            self._inj_labels = None
            self._inj_data_path = None
            self._inj_mtime = None
            return

        oneminfile = INJ_LIVE_DIR / 'latest_INJ_data_1min.csv'
        watch_path = oneminfile if oneminfile.exists() else data_path
        mtime = watch_path.stat().st_mtime
        if self._inj_data_path == watch_path and self._inj_mtime == mtime:
            return

        self._inj_df, self._inj_labels = load_injection_dataframe(INJ_LIVE_DIR, filename='latest_INJ_data_1min.csv')
        self._inj_data_path = watch_path
        self._inj_mtime = mtime

    def _refresh(self):
        cat_df, last_updated = self._fetch()
        self._refresh_injection_data()
        self._cat_df_full = cat_df
        cat_filtered = self._apply_date_filter(cat_df)
        self._header.object = self._header_md(len(cat_filtered), last_updated, n_total=len(cat_df))
        self._plot.object = build_figure(
            cat_filtered,
            self._stations,
            self._wellbores,
            self._hull_verts,
            self._hull_faces,
            last_updated,
        )
        self._mag_plot.object = build_time_panels_figure(cat_filtered, self._inj_df, self._inj_labels, date_range=self._date_range())

    def __panel__(self):
        date_row = pn.Row(
            pn.pane.Markdown("**Filter time window (UTC):**", margin=(8, 8, 0, 0)),
            self._date_start,
            self._time_start,
            self._date_end,
            self._time_end,
        )
        return pn.Column(
            self._header,
            date_row,
            self._plot,
            self._mag_plot,
            sizing_mode="stretch_width",
        )


app = SeismicityDashboard()
pn.template.VanillaTemplate(
    title="CUSSP Seismicity",
    logo="/CUSSP.png",
    main=app,
).servable()
