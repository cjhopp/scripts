#!/usr/bin/env python3
"""Prepare CUSSP CASSM inversion geometry and prior mask artifacts.

This script intentionally uses the same receiver station schema used by the
CUSSP seismicity dashboard (stations_hmc.csv), and creates full-domain prior
mask artifacts suitable for new-site commissioning.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("cussp_cassm_inversion_prep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_stations(station_file: Path) -> pd.DataFrame:
    df = pd.read_csv(station_file)
    required = {"hmc_east_m", "hmc_north_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"Station CSV missing required columns: {sorted(required)}")

    z_col = "hmc_z_minus_depth_m" if "hmc_z_minus_depth_m" in df.columns else "hmc_z_m_asl"
    if z_col not in df.columns:
        raise ValueError("Station CSV missing z column: hmc_z_minus_depth_m or hmc_z_m_asl")

    out = pd.DataFrame(
        {
            "station": df.get("station", "").astype(str),
            "channel": df.get("channel", "").astype(str),
            "x": pd.to_numeric(df["hmc_east_m"], errors="coerce"),
            "y": pd.to_numeric(df["hmc_north_m"], errors="coerce"),
            "z": pd.to_numeric(df[z_col], errors="coerce"),
        }
    ).dropna(subset=["x", "y", "z"])

    # One point per station to avoid per-channel duplicates for inversion geometry.
    out = out.sort_values(["station", "channel"]).drop_duplicates(subset=["station"], keep="first")
    out = out.rename(columns={"station": "station_id"})
    return out[["station_id", "x", "y", "z"]].reset_index(drop=True)


def _load_sources(source_file: Path) -> pd.DataFrame:
    df = pd.read_csv(source_file)
    required = {"source_id", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"Source CSV missing required columns: {sorted(required)}")

    out = pd.DataFrame(
        {
            "source_id": df["source_id"].astype(str),
            "x": pd.to_numeric(df["x"], errors="coerce"),
            "y": pd.to_numeric(df["y"], errors="coerce"),
            "z": pd.to_numeric(df["z"], errors="coerce"),
        }
    ).dropna(subset=["x", "y", "z"])
    return out.reset_index(drop=True)


def _split_sources_receivers(
    station_points: pd.DataFrame,
    source_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    src_re = re.compile(source_pattern)
    mask = station_points["station_id"].astype(str).apply(lambda s: bool(src_re.match(s)))

    sources = station_points[mask].copy().rename(columns={"station_id": "source_id"})
    receivers = station_points[~mask].copy().rename(columns={"station_id": "receiver_id"})

    sources = sources[["source_id", "x", "y", "z"]].reset_index(drop=True)
    receivers = receivers[["receiver_id", "x", "y", "z"]].reset_index(drop=True)
    return sources, receivers


def _load_amplifier_map(amplifier_map_file: Path) -> pd.DataFrame:
    df = pd.read_csv(amplifier_map_file)
    required = {"channel", "borehole", "depth_m"}
    if not required.issubset(df.columns):
        raise ValueError(f"Amplifier map missing required columns: {sorted(required)}")

    out = pd.DataFrame(
        {
            "channel": pd.to_numeric(df["channel"], errors="coerce"),
            "borehole": df["borehole"].astype(str).str.upper().str.strip(),
            "depth_m": pd.to_numeric(df["depth_m"], errors="coerce"),
        }
    ).dropna(subset=["channel", "depth_m"])
    out["channel"] = out["channel"].astype(int)

    out = out.sort_values(["borehole", "depth_m"]).reset_index(drop=True)
    out["source_index"] = out.groupby("borehole").cumcount() + 1
    return out


def _attach_amplifier_channels(
    sources: pd.DataFrame,
    amplifier_map: pd.DataFrame,
    borehole_alias: dict[str, str],
) -> pd.DataFrame:
    src = sources.copy()
    src["_key"] = src["source_id"].astype(str).str.upper().str.strip()
    src["amplifier_channel"] = np.nan
    src["borehole"] = ""
    src["depth_m"] = np.nan
    src["source_index"] = np.nan

    for _, row in amplifier_map.iterrows():
        borehole = str(row["borehole"]).upper().strip()
        idx = int(row["source_index"])

        prefixes = [borehole]
        alias = borehole_alias.get(borehole)
        if alias:
            prefixes.append(str(alias).upper().strip())

        candidates = []
        for pre in prefixes:
            candidates.extend([f"{pre}S{idx}", f"{pre}{idx}"])

        matched = src["_key"].isin(candidates)
        if matched.any():
            src.loc[matched, "amplifier_channel"] = int(row["channel"])
            src.loc[matched, "borehole"] = borehole
            src.loc[matched, "depth_m"] = float(row["depth_m"])
            src.loc[matched, "source_index"] = idx

    src.drop(columns=["_key"], inplace=True)
    src["amplifier_channel"] = src["amplifier_channel"].astype("Int64")
    src["source_index"] = src["source_index"].astype("Int64")
    return src


def _build_pair_table(sources: pd.DataFrame, receivers: pd.DataFrame) -> pd.DataFrame:
    src = sources.assign(_k=1)
    rec = receivers.assign(_k=1)
    pairs = src.merge(rec, on="_k", suffixes=("_src", "_rec")).drop(columns=["_k"])
    pairs = pairs.rename(
        columns={
            "x_src": "tx",
            "y_src": "ty",
            "z_src": "tz",
            "x_rec": "rx",
            "y_rec": "ry",
            "z_rec": "rz",
        }
    )
    keep = ["source_id", "receiver_id", "tx", "ty", "tz", "rx", "ry", "rz"]
    if "amplifier_channel" in pairs.columns:
        keep.insert(1, "amplifier_channel")
    if "borehole" in pairs.columns:
        keep.insert(2, "borehole")
    if "depth_m" in pairs.columns:
        keep.insert(3, "depth_m")
    return pairs[keep]


def _build_grid_and_mask(
    pairs: pd.DataFrame,
    spacing_xyz: tuple[float, float, float],
    margin_m: float,
    explicit_bounds: tuple[float, float, float, float, float, float] | None,
) -> dict:
    coords = np.vstack(
        [
            pairs[["tx", "ty", "tz"]].to_numpy(float),
            pairs[["rx", "ry", "rz"]].to_numpy(float),
        ]
    )

    if explicit_bounds is None:
        xmin, ymin, zmin = coords.min(axis=0) - margin_m
        xmax, ymax, zmax = coords.max(axis=0) + margin_m
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = explicit_bounds

    dx, dy, dz = spacing_xyz
    nx = int(np.floor((xmax - xmin) / dx)) + 1
    ny = int(np.floor((ymax - ymin) / dy)) + 1
    nz = int(np.floor((zmax - zmin) / dz)) + 1

    x = xmin + np.arange(nx) * dx
    y = ymin + np.arange(ny) * dy
    z = zmin + np.arange(nz) * dz

    # Full-domain commissioning mask: allow changes everywhere in the inversion volume.
    mask = np.ones(nx * ny * nz, dtype=np.float32)

    return {
        "x": x,
        "y": y,
        "z": z,
        "mask": mask,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
        "zmin": float(zmin),
        "zmax": float(zmax),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare inversion geometry and full-domain prior mask")
    p.add_argument("--station-file", default="/data/chet-cussp/seismicity/stations_hmc.csv")
    p.add_argument(
        "--source-file",
        default="",
        help="Optional explicit source CSV with columns source_id,x,y,z. If omitted, sources are inferred from station names.",
    )
    p.add_argument(
        "--source-pattern",
        default=r".*S[1-4]$",
        help="Regex pattern to classify station IDs as sources when source-file is omitted.",
    )
    p.add_argument(
        "--amplifier-map-file",
        default="/home/chopp/scripts/python/vm/source_channel_depth_map.csv",
        help="CSV with columns channel,borehole,depth_m for source channel mapping.",
    )
    p.add_argument(
        "--borehole-alias-json",
        default='{"WMW":"TSS"}',
        help="JSON dict mapping borehole names to station prefix aliases (e.g. WMW->TSS).",
    )
    p.add_argument("--out-dir", default="/data/chet-cussp/cassm/inversion/input")
    p.add_argument("--vp-mps", type=float, default=6900.0)
    p.add_argument("--spacing-x", type=float, default=0.5)
    p.add_argument("--spacing-y", type=float, default=0.5)
    p.add_argument("--spacing-z", type=float, default=0.5)
    p.add_argument("--margin-m", type=float, default=0.5)
    p.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        default=None,
        help="Optional explicit bounds instead of auto bounds.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    station_file = Path(args.station_file)
    source_file = Path(args.source_file) if args.source_file else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_points = _load_stations(station_file)
    inferred_sources, receivers = _split_sources_receivers(station_points, args.source_pattern)

    if source_file is not None and source_file.exists():
        sources = _load_sources(source_file)
    else:
        sources = inferred_sources.copy()

    amplifier_map_file = Path(args.amplifier_map_file)
    if amplifier_map_file.exists() and not sources.empty:
        try:
            alias = json.loads(args.borehole_alias_json)
            alias = {str(k).upper(): str(v).upper() for k, v in alias.items()}
            amp_map = _load_amplifier_map(amplifier_map_file)
            sources = _attach_amplifier_channels(sources, amp_map, alias)
        except Exception as exc:
            LOG.warning("Failed to attach amplifier map (%s): %s", amplifier_map_file, exc)

    if receivers.empty:
        raise RuntimeError("No valid receivers loaded.")
    if sources.empty:
        raise RuntimeError("No valid sources loaded.")

    sources_file = out_dir / "sources_hmc.csv"
    receivers_file = out_dir / "receivers_hmc.csv"
    sources.to_csv(sources_file, index=False)
    receivers.to_csv(receivers_file, index=False)

    pairs = _build_pair_table(sources, receivers)
    pair_file = out_dir / "pair_geometry.csv"
    pairs.to_csv(pair_file, index=False)

    grid = _build_grid_and_mask(
        pairs=pairs,
        spacing_xyz=(args.spacing_x, args.spacing_y, args.spacing_z),
        margin_m=float(args.margin_m),
        explicit_bounds=tuple(args.bounds) if args.bounds else None,
    )

    slowness_background = 1.0 / float(args.vp_mps)

    np.savez_compressed(
        out_dir / "inversion_grid_mask.npz",
        x=grid["x"],
        y=grid["y"],
        z=grid["z"],
        mask=grid["mask"],
        nx=grid["nx"],
        ny=grid["ny"],
        nz=grid["nz"],
        dx=grid["dx"],
        dy=grid["dy"],
        dz=grid["dz"],
    )

    summary = {
        "station_file": str(station_file),
        "source_file": str(source_file) if source_file is not None else "derived_from_station_pattern",
        "source_pattern": str(args.source_pattern),
        "amplifier_map_file": str(amplifier_map_file) if amplifier_map_file.exists() else None,
        "sources_file": str(sources_file),
        "receivers_file": str(receivers_file),
        "pair_geometry_file": str(pair_file),
        "grid_mask_file": str(out_dir / "inversion_grid_mask.npz"),
        "n_sources": int(len(sources)),
        "n_receivers": int(len(receivers)),
        "n_pairs": int(len(pairs)),
        "vp_mps": float(args.vp_mps),
        "slowness_background_s_per_m": float(slowness_background),
        "prior_mask_mode": "full_domain",
        "bounds": {
            "xmin": grid["xmin"],
            "xmax": grid["xmax"],
            "ymin": grid["ymin"],
            "ymax": grid["ymax"],
            "zmin": grid["zmin"],
            "zmax": grid["zmax"],
        },
        "grid_shape": [grid["nx"], grid["ny"], grid["nz"]],
    }
    (out_dir / "inversion_prep_summary.json").write_text(json.dumps(summary, indent=2))

    LOG.info("Wrote pair geometry: %s", pair_file)
    LOG.info("Wrote sources: %s", sources_file)
    LOG.info("Wrote receivers: %s", receivers_file)
    LOG.info("Wrote grid/mask: %s", out_dir / "inversion_grid_mask.npz")
    LOG.info("Wrote summary: %s", out_dir / "inversion_prep_summary.json")
    LOG.info("Done: %d sources, %d receivers, %d pairs", len(sources), len(receivers), len(pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
