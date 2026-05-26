#!/usr/bin/env python

"""Build a normalized Cape injection stage table from mixed timing/location sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj


DEFAULT_INJECTION_ROOT = Path("/media/chopp/HDD1/chet-meq/cape_modern/injection")
DEFAULT_OUTPUT_DIR = DEFAULT_INJECTION_ROOT / "timing_location"
FTUS_TO_M = 1200.0 / 3937.0
FT_TO_M = 0.3048
TABLE_COLUMNS = [
    "field",
    "well",
    "source_well",
    "stage",
    "start_time",
    "end_time",
    "interval_begin_m",
    "interval_end_m",
    "interval_mid_m",
    "x_m",
    "y_m",
    "z_m",
    "timing_source",
    "interval_source",
    "position_source",
    "notes",
]
MANUAL_COLUMNS = TABLE_COLUMNS + ["manual_fill_fields"]


def clean_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def parse_stage(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def canonical_source_well(raw_well: str | None) -> str | None:
    text = clean_text(raw_well)
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text)
    text = text.replace("1l", "1-I").replace("1L", "1-I")
    text = re.sub(r"\b([123])([IP])\b", r"\1-\2", text)
    text = re.sub(r"\b([48])([A-Z]{2})\b", r"\1-\2", text)
    if re.fullmatch(r"[123]-[IP]", text):
        text = f"Frisco {text}"
    if re.fullmatch(r"[48]-PB", text):
        text = f"Gold {text}"
    match = re.fullmatch(r"Bearskin[- ]?(\d+[A-Z]+)", text, flags=re.IGNORECASE)
    if match:
        return f"Bearskin {match.group(1).upper()}"
    return text


def canonical_well_label(raw_well: str | None) -> str | None:
    text = canonical_source_well(raw_well)
    if text is None:
        return None
    match = re.fullmatch(r"Frisco\s+(\d)-[IP]", text, flags=re.IGNORECASE)
    if match:
        return f"Frisco-{match.group(1)}"
    match = re.fullmatch(r"Gold\s+(\d+)-([A-Z]+)", text, flags=re.IGNORECASE)
    if match:
        return f"Gold-{match.group(1)}{match.group(2).upper()}"
    match = re.fullmatch(r"Bearskin\s+(\d+[A-Z]+)", text, flags=re.IGNORECASE)
    if match:
        return f"Bearskin-{match.group(1).upper()}"
    return text.replace(" ", "-")


def field_from_well(well: str | None) -> str | None:
    if not well:
        return None
    return well.split("-", 1)[0]


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[0]


def deduplicate_stage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    key_cols = ["well", "stage"]
    value_cols = [column for column in frame.columns if column not in key_cols]
    aggregations = {column: first_non_null for column in value_cols}
    deduplicated = frame.groupby(key_cols, dropna=False, as_index=False).agg(aggregations)
    return deduplicated[frame.columns]


def load_trajectory_points(path: Path) -> np.ndarray:
    frame = pd.read_csv(path, header=None)
    numeric = frame.apply(pd.to_numeric, errors='coerce')
    numeric = numeric.dropna(how='any')
    if numeric.shape[1] < 3:
        raise ValueError(f'Expected at least 3 numeric columns in {path}')
    return numeric.iloc[:, :3].to_numpy(dtype=float)


_UTM26912 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:26912", always_xy=True)


def load_gps_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load GPS-accurate well trajectory from a plain {well}.csv file.

    Returns (md_m, east_m, north_m, elev_m) arrays aligned by measured depth,
    where elev_m is positive above sea level (= -sstvd_m).
    Returns empty arrays if the file has no usable data.
    """
    frame = pd.read_csv(path)
    md = pd.to_numeric(frame["md_m"], errors="coerce")
    lat = pd.to_numeric(frame["lat_deg"], errors="coerce")
    lon = pd.to_numeric(frame["lon_deg"], errors="coerce")
    sstvd = pd.to_numeric(frame["sstvd_m"], errors="coerce")
    valid = md.notna() & lat.notna() & lon.notna() & sstvd.notna()
    if valid.sum() < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    md_v = md[valid].to_numpy(dtype=float)
    lat_v = lat[valid].to_numpy(dtype=float)
    lon_v = lon[valid].to_numpy(dtype=float)
    sstvd_v = sstvd[valid].to_numpy(dtype=float)
    east_v, north_v = _UTM26912.transform(lon_v, lat_v)
    elev_v = -sstvd_v  # sstvd positive = below sea level; elev positive = above sea level
    order = np.argsort(md_v)
    return md_v[order], east_v[order], north_v[order], elev_v[order]


def interpolate_along_trajectory(points: np.ndarray, target_distance_m: float) -> tuple[float, float, float]:
    if len(points) == 0:
        raise ValueError('Empty trajectory point array')
    if len(points) == 1:
        return tuple(points[0])
    deltas = np.diff(points, axis=0)
    arc_length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(deltas, axis=1))))
    target_distance_m = float(np.clip(target_distance_m, float(arc_length[0]), float(arc_length[-1])))
    x = np.interp(target_distance_m, arc_length, points[:, 0])
    y = np.interp(target_distance_m, arc_length, points[:, 1])
    z = np.interp(target_distance_m, arc_length, points[:, 2])
    return float(x), float(y), float(z)


def merge_stage_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    key_cols = ["well", "stage"]
    merged = None
    for frame in frames:
        if frame is None or frame.empty:
            continue
        frame = deduplicate_stage_frame(frame.copy())
        if merged is None:
            merged = frame
            continue
        merged = merged.merge(frame, on=key_cols, how="outer", suffixes=("", "__new"))
        for column in list(merged.columns):
            if not column.endswith("__new"):
                continue
            base_column = column[:-5]
            if base_column in merged.columns:
                merged[base_column] = merged[base_column].where(merged[base_column].notna(), merged[column])
            else:
                merged.rename(columns={column: base_column}, inplace=True)
        merged.drop(columns=[col for col in merged.columns if col.endswith("__new")], inplace=True)
    if merged is None:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    return merged


def load_frisco_stage_times(injection_root: Path) -> pd.DataFrame:
    path = injection_root / "timing_location" / "first_times.csv"
    frame = pd.read_csv(path)
    frame["source_well"] = frame["Well Name"].map(lambda value: canonical_source_well(f"Frisco {value}"))
    frame["well"] = frame["source_well"].map(canonical_well_label)
    frame["stage"] = frame["Stage Number"].map(parse_stage)
    frame["start_time"] = pd.to_datetime(frame["First Time"], errors="coerce")
    frame["timing_source"] = path.name
    frame["field"] = frame["well"].map(field_from_well)
    return ensure_columns(frame, TABLE_COLUMNS)


def parse_frisco_stage_file_well(file_name: str) -> str | None:
    lower_name = file_name.lower()
    if "frisco-2p" in lower_name or " 2-p" in lower_name:
        return "Frisco 2-P"
    if "frisco 3i" in lower_name or "frisco 3-i" in lower_name:
        return "Frisco 3-I"
    if "frisco 1i" in lower_name or "frisco 1l" in lower_name or "frisco 1-i" in lower_name:
        return "Frisco 1-I"
    return None


def parse_frisco_stage_file_stage(file_name: str) -> int | None:
    for pattern in [r"stage_(\d+)", r"stage\s+(\d+)\s+of"]:
        match = re.search(pattern, file_name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_first_timestamp(frame: pd.DataFrame) -> pd.Timestamp | pd.NaT:
    for column in ["Time", "Job Time", "Time.1"]:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(str).str.strip()
        if values.empty:
            continue
        for date_format in ["%Y-%m-%d %H:%M:%S", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"]:
            parsed = pd.to_datetime(values, format=date_format, errors="coerce")
            if parsed.notna().any():
                return parsed.dropna().iloc[0]
    return pd.NaT


def load_frisco_raw_stage_times(injection_root: Path) -> pd.DataFrame:
    roots = [
        injection_root / "Frisco Inj data" / "Frisco 1I_3I",
        injection_root / "Frisco Inj data" / "Frisco 2-P",
    ]
    records = []
    for root in roots:
        for path in sorted(root.glob("*.csv")):
            source_well = canonical_source_well(parse_frisco_stage_file_well(path.name))
            stage = parse_frisco_stage_file_stage(path.name)
            if source_well is None or stage is None:
                continue
            frame = pd.read_csv(path, nrows=10)
            start_time = extract_first_timestamp(frame)
            if pd.isna(start_time):
                continue
            records.append({
                "field": "Frisco",
                "well": canonical_well_label(source_well),
                "source_well": source_well,
                "stage": stage,
                "start_time": start_time,
                "timing_source": str(path.relative_to(injection_root)),
            })
    return ensure_columns(pd.DataFrame.from_records(records), TABLE_COLUMNS)


def load_gold_stage_times(injection_root: Path) -> pd.DataFrame:
    path = injection_root / "Gold" / "Gold Frac Phase 1 Start-Stop Times.xlsx"
    frame = pd.read_excel(path)
    frame["source_well"] = frame["Well"].map(canonical_source_well)
    frame["well"] = frame["source_well"].map(canonical_well_label)
    frame["stage"] = frame["Stage"].map(parse_stage)
    dates = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    start_offset = pd.to_timedelta(frame["Start Time"].astype(str), errors="coerce")
    end_offset = pd.to_timedelta(frame["End Time"].astype(str), errors="coerce")
    frame["start_time"] = dates + start_offset
    frame["end_time"] = dates + end_offset
    overnight = frame["end_time"].notna() & frame["start_time"].notna() & (frame["end_time"] < frame["start_time"])
    frame.loc[overnight, "end_time"] = frame.loc[overnight, "end_time"] + pd.Timedelta(days=1)
    frame["notes"] = frame["Notes"]
    frame["timing_source"] = path.name
    frame["field"] = frame["well"].map(field_from_well)
    return ensure_columns(frame, TABLE_COLUMNS)


def load_frisco_stage_intervals(injection_root: Path) -> pd.DataFrame:
    path = injection_root / "timing_location" / "Frisco Pad Perf Intervals_es.xlsx"
    raw = pd.read_excel(path, header=None)
    meter_block_starts = [15, 19, 23]
    records = []
    for start_col in meter_block_starts:
        source_well = canonical_source_well(raw.iat[1, start_col])
        if source_well is None:
            continue
        block = raw.iloc[3:, start_col:start_col + 4].copy()
        block.columns = ["stage", "interval_begin_m", "interval_end_m", "interval_length_m"]
        block = block[pd.to_numeric(block["stage"], errors="coerce").notna()].copy()
        if block.empty:
            continue
        block["well"] = canonical_well_label(source_well)
        block["source_well"] = source_well
        block["stage"] = block["stage"].map(parse_stage)
        for column in ["interval_begin_m", "interval_end_m", "interval_length_m"]:
            block[column] = pd.to_numeric(block[column], errors="coerce")
        block["interval_mid_m"] = (block["interval_begin_m"] + block["interval_end_m"]) / 2.0
        block["interval_source"] = path.name
        block["field"] = block["well"].map(field_from_well)
        records.append(ensure_columns(block, TABLE_COLUMNS))
    if not records:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    return pd.concat(records, ignore_index=True)


def load_bearskin_stage_locations(injection_root: Path) -> pd.DataFrame:
    path = injection_root / "Bearskin" / "Bearskin_well_trajectories.csv"
    borehole_dir = injection_root.parent / "spatial_data" / "vector" / "boreholes"
    frame = pd.read_csv(path)
    frame["stage"] = pd.to_numeric(frame["STAGENO"], errors="coerce").astype("Int64")
    frame = frame[frame["stage"].notna() & (frame["stage"] != 999)].copy()
    staged_rows = []
    for (source_well, stage), group in frame.groupby(["WELLNAME", "stage"]):
        depth_ft = pd.to_numeric(group["DEPTHMD_FT"], errors="coerce")
        midpoint_ft = (depth_ft.min() + depth_ft.max()) / 2.0
        midpoint_index = (depth_ft - midpoint_ft).abs().idxmin()
        midpoint_row = group.loc[midpoint_index]
        interval_mid_m = midpoint_ft * FT_TO_M
        well_label = canonical_well_label(source_well)
        well_csv_name = f"{well_label.replace('-', '_')}.csv"
        well_csv_path = borehole_dir / well_csv_name
        x_m, y_m, z_m, position_source = None, None, None, None
        if well_csv_path.exists():
            md, east, north, elev = load_gps_trajectory(well_csv_path)
            if len(md) >= 2:
                x_m = float(np.interp(interval_mid_m, md, east))
                y_m = float(np.interp(interval_mid_m, md, north))
                z_m = float(np.interp(interval_mid_m, md, elev))
                position_source = well_csv_name
        if x_m is None:
            # No GPS-accurate source for this well; leave position as NaN
            z_m = float(midpoint_row["Z_FT"]) * FT_TO_M
        staged_rows.append({
            "field": "Bearskin",
            "well": well_label,
            "source_well": canonical_source_well(source_well),
            "stage": int(stage),
            "interval_begin_m": depth_ft.min() * FT_TO_M,
            "interval_end_m": depth_ft.max() * FT_TO_M,
            "interval_mid_m": interval_mid_m,
            "x_m": x_m,
            "y_m": y_m,
            "z_m": z_m,
            "interval_source": path.name,
            "position_source": position_source,
            "notes": group.get("PLUGTYPE").dropna().iloc[0] if group.get("PLUGTYPE").notna().any() else pd.NA,
        })
    return ensure_columns(pd.DataFrame.from_records(staged_rows), TABLE_COLUMNS)


FRISCO_PLAN_XLSX = {
    'Frisco-1': 'Frisco 1-I_Plan #2.xlsx',
    'Frisco-2': 'Frisco 2-P_Plan #2.xlsx',
    'Frisco-3': 'Frisco 3-I_Plan #2.xlsx',
}


def load_frisco_plan_trajectory(borehole_dir: Path, well_label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Frisco well trajectory from Plan xlsx (the same source as the well-trace GeoJSONs).

    The Plan xlsx contains MD (ft), Latitude, Longitude, and SubSea TVD (ft) columns, so
    stage positions interpolated from this will fall on the displayed well traces.
    Returns (md_m, east_m, north_m, elev_m), or four empty arrays if unavailable.
    """
    xlsx_name = FRISCO_PLAN_XLSX.get(well_label)
    if xlsx_name is None:
        return np.array([]), np.array([]), np.array([]), np.array([])
    path = borehole_dir / xlsx_name
    if not path.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])
    raw = pd.read_excel(path, header=None)
    # Find the data-header row (contains "Measured Depth" in column 0)
    header_row = None
    for i, row in raw.iterrows():
        if str(row.iloc[0]).strip().lower().startswith('measured depth'):
            header_row = i
            break
    if header_row is None:
        return np.array([]), np.array([]), np.array([]), np.array([])
    # Skip header row + abbreviation row; data starts two rows after the header
    data = raw.iloc[header_row + 2:].reset_index(drop=True)
    md_ft = pd.to_numeric(data.iloc[:, 0], errors='coerce')
    sstvd_ft = pd.to_numeric(data.iloc[:, 5], errors='coerce')
    lat = pd.to_numeric(data.iloc[:, 10], errors='coerce')
    lon = pd.to_numeric(data.iloc[:, 11], errors='coerce')
    valid = md_ft.notna() & sstvd_ft.notna() & lat.notna() & lon.notna()
    if valid.sum() < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    md_v = (md_ft[valid] * FT_TO_M).to_numpy(dtype=float)
    elev_v = (sstvd_ft[valid] * FT_TO_M).to_numpy(dtype=float)
    east_v, north_v = _UTM26912.transform(lon[valid].to_numpy(dtype=float), lat[valid].to_numpy(dtype=float))
    order = np.argsort(md_v)
    return md_v[order], east_v[order], north_v[order], elev_v[order]


def interpolate_frisco_positions(stage_table: pd.DataFrame, injection_root: Path) -> pd.DataFrame:
    borehole_dir = injection_root.parent / "spatial_data" / "vector" / "boreholes"
    welpaths_path = injection_root / "Frisco Inj data" / "fromEricS" / "Frisco Welpaths.csv"
    welpaths = pd.read_csv(welpaths_path)
    welpaths["source_well"] = welpaths["Well ID"].map(canonical_source_well)
    welpaths["well"] = welpaths["source_well"].map(canonical_well_label)
    records = []
    frisco_wells = stage_table.loc[stage_table["field"] == "Frisco", "well"].dropna().unique()
    for well in frisco_wells:
        subset = stage_table[(stage_table["well"] == well) & stage_table["interval_mid_m"].notna()]
        if subset.empty:
            continue
        # Prefer Plan xlsx — same source as the well-trace GeoJSONs — so stage markers
        # fall on the displayed well trace rather than a different survey's UTM positions.
        md, east, north, elev = load_frisco_plan_trajectory(borehole_dir, well)
        if len(md) >= 2:
            position_source = FRISCO_PLAN_XLSX[well]
        else:
            # Fall back to Welpaths for wells without a Plan xlsx (e.g. Frisco-4)
            group = welpaths[welpaths["well"] == well].sort_values("Depth MD_m")
            if group.empty:
                continue
            md = pd.to_numeric(group["Depth MD_m"], errors="coerce").to_numpy()
            east = pd.to_numeric(group["UTM x (m)"], errors="coerce").to_numpy()
            north = pd.to_numeric(group["UTM y (m)"], errors="coerce").to_numpy()
            elev = pd.to_numeric(group["elev z (m)"], errors="coerce").to_numpy()
            position_source = welpaths_path.name
        for row in subset.itertuples(index=False):
            records.append({
                "well": well,
                "stage": int(row.stage),
                "x_m": float(np.interp(row.interval_mid_m, md, east)),
                "y_m": float(np.interp(row.interval_mid_m, md, north)),
                "z_m": float(np.interp(row.interval_mid_m, md, elev)),
                "position_source": position_source,
            })
    if not records:
        return pd.DataFrame(columns=["well", "stage", "x_m", "y_m", "z_m", "position_source"])
    return pd.DataFrame.from_records(records)


def interpolate_gold_positions(stage_table: pd.DataFrame, injection_root: Path) -> pd.DataFrame:
    borehole_dir = injection_root.parent / "spatial_data" / "vector" / "boreholes"
    records = []
    gold_wells = stage_table.loc[stage_table["field"] == "Gold", "well"].dropna().unique()
    for well in gold_wells:
        well_csv_name = f"{well.replace('-', '_')}.csv"
        well_csv_path = borehole_dir / well_csv_name
        if not well_csv_path.exists():
            continue
        md, east, north, elev = load_gps_trajectory(well_csv_path)
        if len(md) < 2:
            continue
        subset = stage_table[(stage_table["well"] == well) & stage_table["interval_mid_m"].notna()]
        for row in subset.itertuples(index=False):
            records.append({
                "well": well,
                "stage": int(row.stage),
                "x_m": float(np.interp(row.interval_mid_m, md, east)),
                "y_m": float(np.interp(row.interval_mid_m, md, north)),
                "z_m": float(np.interp(row.interval_mid_m, md, elev)),
                "position_source": well_csv_name,
            })
    if not records:
        return pd.DataFrame(columns=["well", "stage", "x_m", "y_m", "z_m", "position_source"])
    return pd.DataFrame.from_records(records)


def load_manual_overrides(manual_path: Path) -> pd.DataFrame:
    if not manual_path.exists() or manual_path.stat().st_size == 0:
        return pd.DataFrame(columns=MANUAL_COLUMNS)
    frame = pd.read_csv(manual_path)
    for column in ["start_time", "end_time"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if "stage" in frame.columns:
        frame["stage"] = frame["stage"].map(parse_stage)
    if "well" in frame.columns:
        frame["field"] = frame["well"].map(field_from_well)
    return ensure_columns(frame, MANUAL_COLUMNS)


def build_stage_table(injection_root: Path, manual_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frisco_times = load_frisco_stage_times(injection_root)
    frisco_raw_times = load_frisco_raw_stage_times(injection_root)
    gold_times = load_gold_stage_times(injection_root)
    frisco_intervals = load_frisco_stage_intervals(injection_root)
    bearskin_locations = load_bearskin_stage_locations(injection_root)

    stage_table = merge_stage_frames([
        frisco_times,
        frisco_raw_times,
        gold_times,
        frisco_intervals,
        bearskin_locations,
    ])
    frisco_positions = interpolate_frisco_positions(stage_table, injection_root)
    gold_positions = interpolate_gold_positions(stage_table, injection_root)
    stage_table = merge_stage_frames([stage_table, frisco_positions, gold_positions])
    manual_overrides = load_manual_overrides(manual_path)
    if not manual_overrides.empty:
        stage_table = merge_stage_frames([stage_table, ensure_columns(manual_overrides, TABLE_COLUMNS)])

    stage_table["field"] = stage_table["well"].map(field_from_well)
    stage_table["stage"] = pd.to_numeric(stage_table["stage"], errors="coerce").astype("Int64")
    stage_table["timing_status"] = np.where(stage_table["start_time"].notna(), "present", "missing")
    location_present = stage_table[["interval_begin_m", "interval_end_m", "x_m", "y_m", "z_m"]].notna().any(axis=1)
    stage_table["location_status"] = np.where(location_present, "present", "missing")
    stage_table = stage_table.sort_values(["field", "well", "stage"], kind="stable").reset_index(drop=True)

    gaps = stage_table[(stage_table["timing_status"] == "missing") | (stage_table["location_status"] == "missing")].copy()
    missing_fields = []
    for row in gaps.itertuples(index=False):
        needs = []
        if row.timing_status == "missing":
            needs.append("start/end time")
        if row.location_status == "missing":
            needs.append("stage location")
        missing_fields.append(", ".join(needs))
    gaps["manual_fill_fields"] = missing_fields
    return stage_table, ensure_columns(gaps, MANUAL_COLUMNS)


def write_manual_scaffold(manual_path: Path) -> None:
    if manual_path.exists():
        return
    pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(manual_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injection-root", type=Path, default=DEFAULT_INJECTION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manual_path = output_dir / "cape_injection_stage_manual.csv"
    write_manual_scaffold(manual_path)
    stage_table, gaps = build_stage_table(args.injection_root, manual_path)

    table_path = output_dir / "cape_injection_stage_table.csv"
    gaps_path = output_dir / "cape_injection_stage_gaps.csv"
    stage_table.to_csv(table_path, index=False)
    gaps.to_csv(gaps_path, index=False)

    field_counts = stage_table.groupby("field").size().to_dict()
    print(f"Wrote {len(stage_table)} stage rows to {table_path}")
    print(f"Wrote {len(gaps)} gap rows to {gaps_path}")
    print(f"Manual overrides: {manual_path}")
    print(f"Rows by field: {field_counts}")


if __name__ == "__main__":
    main()