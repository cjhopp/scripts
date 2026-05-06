#!/usr/bin/env python3
"""
plot_event_locations.py

Map view (XY) and two cross-sections (XZ, YZ) for CASS-DTS shot locations.

All coordinates are in HMC metres (SURF mine coordinate system) via
lbnl.coordinates.SURF_converter.  Z is HMC elevation (positive upward).

Overlays borehole trajectories and the drift hull mesh from
~/chet-cussp/model/.

Usage:
    python plot_event_locations.py [--outdir /tmp/plots]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import obspy

# Coordinate conversion: inventory/event lat/lons are NAD27 geographic.
# Empirically verified (RMS 0.07 m vs borehole CSV ground truth):
#   1. datum-shift NAD27 geographic → WGS84 geographic
#   2. project WGS84 UTM Z13N (EPSG:32613)
#   3. subtract WGS84 UTM origin of the HMC frame
from pyproj import Proj as _Proj, Transformer as _Transformer
_UTM_WGS84   = _Proj(init="EPSG:32613")
_ORIG_WGS84  = (598334.1035272244, 4912479.756701191)
_NAD27_TO_WGS84 = _Transformer.from_crs("EPSG:4267", "EPSG:4326", always_xy=True)

def wgs84_to_hmc(lon, lat):
    """Convert NAD27 lon/lat (stored in inventory) → HMC metres."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo84, la84 = _NAD27_TO_WGS84.transform(lon, lat)
        e, n = _UTM_WGS84(lo84, la84)
    return e - _ORIG_WGS84[0], n - _ORIG_WGS84[1]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARRAY_STATIONS = {
    "AML1", "AML2", "AML3", "AML4",
    "AMU1", "AMU2", "AMU4",
    "DML1", "DML2", "DML3", "DML4",
    "DMU1", "DMU2", "DMU3", "DMU4",
}
WINDOW_COLORS = {1: "royalblue", 2: "tomato", 3: "forestgreen"}
WINDOW_LABELS = {1: "Window 1 (3 shots)", 2: "Window 2 (5 shots)", 3: "Window 3 (6 shots)"}

MODEL_DIR    = Path.home() / "chet-cussp" / "model"
BOREHOLE_DIR = MODEL_DIR / "boreholes"
HULL_FILE    = MODEL_DIR / "drift_hull_50k.npy"

# Borehole colours: sensor boreholes black, target/tiltmeter boreholes dimgrey
def _bh_color(name):
    return "dimgrey" if name[0].upper() in ("T",) else "0.35"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_boreholes(directory):
    """Return dict name → (x_arr, y_arr, z_arr) in HMC metres."""
    wells = {}
    bdir = Path(directory)
    if not bdir.exists():
        print(f"Borehole dir not found: {bdir}")
        return wells
    for csv in sorted(bdir.glob("One_foot_*.csv")):
        try:
            arr = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=[2, 3, 4, 5])
            # cols: Depth_m, x_ft, y_ft, z_ft  →  convert x/y/z ft→m
            x = arr[:, 1] * 0.3048
            y = arr[:, 2] * 0.3048
            z = arr[:, 3] * 0.3048
            # Well name embedded in filename: One_foot_interval_well_trajectory_E2_<NAME>_in_HMC
            parts = csv.stem.split("_")
            name = parts[-3] if len(parts) >= 3 else csv.stem
            wells[name] = (x, y, z)
            print(f"  borehole {name}: {len(x)} points  "
                  f"x=[{x.min():.0f},{x.max():.0f}]  "
                  f"z=[{z.min():.0f},{z.max():.0f}]")
        except Exception as e:
            print(f"  WARNING: failed to load {csv.name}: {e}")
    return wells


def load_hull(path):
    """Return (vertices, faces) or (None, None).  Hull npy is a pickled dict."""
    p = Path(path)
    if not p.exists():
        print(f"Hull file not found: {p}")
        return None, None
    try:
        data = np.load(p, allow_pickle=True).item()
        verts = np.array(data["vertices"], dtype=float)
        faces = np.array(data["faces"],    dtype=int)
        print(f"  hull: {len(verts)} vertices, {len(faces)} faces  "
              f"x=[{verts[:,0].min():.0f},{verts[:,0].max():.0f}]  "
              f"z=[{verts[:,2].min():.0f},{verts[:,2].max():.0f}]")
        return verts, faces
    except Exception as e:
        print(f"  WARNING: failed to load hull: {e}")
        return None, None


def hull_outline(verts, faces, axis="z"):
    """Project hull faces onto a 2-D plane and return boundary line segments.

    For each cross-section we collect all triangle edges and plot them as
    thin grey lines — this gives a filled-outline silhouette of the drift.
    axis='z' → XY map;  axis='y' → XZ section;  axis='x' → YZ section.
    """
    if verts is None:
        return None, None
    axis_map = {"z": (0, 1), "y": (0, 2), "x": (1, 2)}
    i0, i1 = axis_map[axis]
    a = verts[faces[:, 0]][:, [i0, i1]]
    b = verts[faces[:, 1]][:, [i0, i1]]
    c = verts[faces[:, 2]][:, [i0, i1]]
    # Return as (N,2,2) segments: [[[x0,y0],[x1,y1]], ...]
    segs = np.stack([a, b], axis=1)
    segs = np.concatenate([segs,
                           np.stack([b, c], axis=1),
                           np.stack([c, a], axis=1)], axis=0)
    return segs[:, :, 0], segs[:, :, 1]   # xs, ys  each (N,2)


def draw_hull_outline(ax, verts, faces, proj_axis, color="0.75", lw=0.3, alpha=0.5):
    if verts is None:
        return
    xs, ys = hull_outline(verts, faces, axis=proj_axis)
    if xs is None:
        return
    # plot as LineCollection for speed
    from matplotlib.collections import LineCollection
    segs = np.stack([xs, ys], axis=2)   # (N, 2, 2) → [start,end] per seg
    lc = LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=1)
    ax.add_collection(lc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir",      default="/tmp/plots")
    ap.add_argument("--events-dir",  default="/tmp")
    ap.add_argument("--inventory",   default="/tmp/inventory.xml")
    ap.add_argument("--borehole-dir", default=str(BOREHOLE_DIR))
    ap.add_argument("--hull",        default=str(HULL_FILE))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model overlays
    # ------------------------------------------------------------------
    print("Loading boreholes…")
    wells = load_boreholes(args.borehole_dir)
    print("Loading drift hull…")
    hull_verts, hull_faces = load_hull(args.hull)

    # ------------------------------------------------------------------
    # Load inventory → station HMC coordinates
    # ------------------------------------------------------------------
    inv = obspy.read_inventory(args.inventory)
    sta_hmc = {}    # code → (hmc_x, hmc_y, hmc_z)
    for net in inv:
        for sta in net:
            if sta.code in ARRAY_STATIONS:
                hx, hy = wgs84_to_hmc(sta.longitude, sta.latitude)
                sta_hmc[sta.code] = (hx, hy, sta.elevation)

    if not sta_hmc:
        raise RuntimeError("No array stations found in inventory.")

    sx = np.array([v[0] for v in sta_hmc.values()])
    sy = np.array([v[1] for v in sta_hmc.values()])
    sz = np.array([v[2] for v in sta_hmc.values()])

    # ------------------------------------------------------------------
    # Load events → HMC coordinates
    # ------------------------------------------------------------------
    ev_data = []
    for w in [1, 2, 3]:
        ev_file = Path(args.events_dir) / f"pbevents_w{w}.xml"
        if not ev_file.exists():
            print(f"w{w}: {ev_file} not found, skipping")
            continue
        cat = obspy.read_events(str(ev_file))
        for ev in cat:
            o = ev.preferred_origin() or ev.origins[0]
            hx, hy = wgs84_to_hmc(o.longitude, o.latitude)
            hz = -o.depth   # ObsPy depth is +ve downward, HMC z is elevation
            ev_data.append(dict(
                w=w, x=hx, y=hy, z=hz,
                time=o.time, n_arrivals=len(o.arrivals),
            ))
        print(f"w{w}: {len(cat)} events loaded")

    if not ev_data:
        raise RuntimeError("No events found.")

    # ------------------------------------------------------------------
    # Figure: 3 panels  (XY map | XZ E–W section | YZ N–S section)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 6))
    ax_xy = fig.add_subplot(1, 3, 1)
    ax_xz = fig.add_subplot(1, 3, 2)
    ax_yz = fig.add_subplot(1, 3, 3)

    # --- drift hull outline ---
    draw_hull_outline(ax_xy, hull_verts, hull_faces, proj_axis="z")
    draw_hull_outline(ax_xz, hull_verts, hull_faces, proj_axis="y")
    draw_hull_outline(ax_yz, hull_verts, hull_faces, proj_axis="x")

    # --- boreholes ---
    for name, (bx, by, bz) in wells.items():
        c = _bh_color(name)
        lkw = dict(color=c, lw=0.8, alpha=0.7, zorder=2)
        ax_xy.plot(bx, by, **lkw)
        ax_xz.plot(bx, bz, **lkw)
        ax_yz.plot(by, bz, **lkw)

    # --- stations ---
    mkw = dict(marker="^", c="k", s=50, zorder=5, linewidths=0.5)
    ax_xy.scatter(sx, sy, **mkw)
    ax_xz.scatter(sx, sz, **mkw)
    ax_yz.scatter(sy, sz, **mkw)
    for code, (x, y, _) in sta_hmc.items():
        ax_xy.annotate(code, (x, y), fontsize=5,
                       xytext=(2, 2), textcoords="offset points",
                       ha="left", va="bottom", color="0.3")

    # --- events ---
    for d in ev_data:
        c = WINDOW_COLORS[d["w"]]
        kw = dict(c=c, s=35, zorder=10, alpha=0.85, linewidths=0.4, edgecolors="k")
        ax_xy.scatter(d["x"], d["y"], **kw)
        ax_xz.scatter(d["x"], d["z"], **kw)
        ax_yz.scatter(d["y"], d["z"], **kw)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("HMC Easting (m)",   fontsize=9)
    ax_xy.set_ylabel("HMC Northing (m)",  fontsize=9)
    ax_xy.set_title("Map view (XY)",      fontsize=10)

    ax_xz.set_xlabel("HMC Easting (m)",   fontsize=9)
    ax_xz.set_ylabel("HMC Elevation (m)", fontsize=9)
    ax_xz.set_title("E–W cross-section (X–Z)", fontsize=10)

    ax_yz.set_xlabel("HMC Northing (m)",  fontsize=9)
    ax_yz.set_ylabel("HMC Elevation (m)", fontsize=9)
    ax_yz.set_title("N–S cross-section (Y–Z)", fontsize=10)

    for ax in [ax_xy, ax_xz, ax_yz]:
        ax.autoscale_view()
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.tick_params(labelsize=8)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=WINDOW_COLORS[w], edgecolor="k",
                       linewidth=0.4, label=WINDOW_LABELS[w])
        for w in [1, 2, 3] if any(d["w"] == w for d in ev_data)
    ]
    legend_handles += [
        mlines.Line2D([], [], marker="^", color="k", linestyle="None",
                      markersize=7, label="Station"),
        mlines.Line2D([], [], color="0.35", lw=1.0, label="Borehole"),
        mlines.Line2D([], [], color="0.75", lw=1.0, label="Drift hull"),
    ]
    ax_xy.legend(handles=legend_handles, fontsize=7, loc="best", framealpha=0.7)

    fig.suptitle("CASS-DTS shot locations — 2026-05-01  (HMC coordinates)", fontsize=12)
    plt.tight_layout(rect=[0, 0.0, 1, 0.97])

    out_path = outdir / "event_locations.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'W':>2}  {'Time':>26}  {'hmc_x':>8}  {'hmc_y':>8}  {'hmc_z':>7}  {'#arr':>4}")
    print("-" * 68)
    for d in ev_data:
        print(f"{d['w']:>2}  {str(d['time']):>26}  "
              f"{d['x']:>8.1f}  {d['y']:>8.1f}  {d['z']:>7.1f}  "
              f"{d['n_arrivals']:>4}")


if __name__ == "__main__":
    main()
