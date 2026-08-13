import panel as pn
import xarray as xr
import holoviews as hv
import numpy as np
import pandas as pd
import logging
import re
import zarr
import dask.array as dask_array
from pathlib import Path

import param

from holoviews.operation.datashader import rasterize

hv.extension('bokeh', config=dict(image_rtol=10000))
log = logging.getLogger(__name__)

chan_map_4100 = {'AMU': 146.445, 'AML': 282.68, 'DMU': 439.905, 'DML': 560.765, 'Whole fiber': 718.4}

fiber_depth_4100 = {'AMU': 60, 'AML': 60, 'DMU': 55, 'DML': 55, 'Whole fiber': 941.8}

buttons = list(chan_map_4100.keys())
buttons.insert(0, 'Whole fiber')


def get_start(direction, well):
    if direction == 'Downgoing':
        return chan_map_4100[well] - fiber_depth_4100[well]
    elif direction == 'Upgoing':
        return chan_map_4100[well]


def get_end(direction, well):
    if direction == 'Downgoing':
        return chan_map_4100[well]
    elif direction == 'Upgoing':
        return chan_map_4100[well] + fiber_depth_4100[well]


_DTS_DROP_VARS = ['probe1_temperature', 'probe2_temperature', 'reference_temperature']


def _decode_seconds_since(raw_seconds, units):
    """Decode an int64 'seconds since <epoch>' array into datetime64[ns].

    Mirrors the TIME_ENCODING used to write this store in combine_XTDTS.py
    ("seconds since 1970-01-01 00:00:00"), without depending on xarray's
    internal CF-decoding API.
    """
    m = re.match(r'seconds since (.+)$', units.strip())
    epoch = np.datetime64(m.group(1).replace(' ', 'T')) if m else np.datetime64('1970-01-01T00:00:00')
    return epoch + raw_seconds.astype('timedelta64[s]')


def _open_dts_dataset(zarr_path):
    """Open the DTS Zarr store, tolerating a mismatched trailing timestep.

    combine_XTDTS.py continuously appends to this store. If that ingest
    process is interrupted mid-write (systemd restart, or resuming after a
    long outage of the upstream DTS source), the 'time' coordinate and the
    'temperature' data variable can end up differing in length by a few
    entries. xr.open_dataset() raises ValueError("conflicting sizes...")
    in that case with no way to relax the check, so fall back to reading
    the raw Zarr arrays directly and truncating everything to the
    shortest common 'time' length.
    """
    try:
        return xr.open_dataset(
            str(zarr_path), chunks={}, engine='zarr', drop_variables=_DTS_DROP_VARS,
        )
    except ValueError as exc:
        if 'conflicting sizes' not in str(exc):
            raise
        log.warning(
            "DTS Zarr store has conflicting 'time' sizes (%s); "
            "falling back to truncated read of raw arrays", exc,
        )

    group = zarr.open_group(str(zarr_path), mode='r')
    time_arr = group['time']
    depth_vals = group['depth'][:]
    temp_arr = group['temperature']
    temp_dims = list(temp_arr.attrs['_ARRAY_DIMENSIONS'])
    time_axis = temp_dims.index('time')

    n_time = min(time_arr.shape[0], temp_arr.shape[time_axis])
    log.warning(
        "Truncating DTS store to %d common timestep(s) (time=%d, temperature=%d)",
        n_time, time_arr.shape[0], temp_arr.shape[time_axis],
    )

    raw_time = time_arr[:n_time]
    time_units = time_arr.attrs.get('units', 'seconds since 1970-01-01 00:00:00')
    time_vals = _decode_seconds_since(raw_time, time_units)

    temp_dask = dask_array.from_zarr(str(zarr_path), component='temperature')
    slicer = [slice(None)] * temp_dask.ndim
    slicer[time_axis] = slice(0, n_time)
    temp_dask = temp_dask[tuple(slicer)]
    temp_attrs = {k: v for k, v in temp_arr.attrs.items() if k != '_ARRAY_DIMENSIONS'}

    return xr.Dataset(
        {'temperature': (temp_dims, temp_dask, temp_attrs)},
        coords={'time': time_vals, 'depth': depth_vals},
    )


_MAX_WINDOW = np.timedelta64(60, 'D')
_raw = _open_dts_dataset('/data/chet-cussp/DTS/DTS_all.zarr')
_time_end = _raw.time[-1].values
_DS = _raw.sel(time=slice(_time_end - _MAX_WINDOW, None)).load()
del _raw

INJ_LIVE_DIR = Path('/data/chet-cussp/injection/live')


def _coerce_plot_time(value):
    """Convert stream/range values into pandas.Timestamp.

    HoloViews/Bokeh range streams may provide datetime axis values as
    milliseconds since epoch (float). Normalize all variants here.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return pd.to_datetime(value, unit='ms', errors='coerce')
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def _resolve_metadata_path(data_path):
    return data_path.with_name(data_path.name.replace('data', 'metadata'))


def _parse_injection_metadata(metadata_path):
    """Return a best-effort mapping of column -> unit from metadata CSV."""
    if not metadata_path.exists():
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


def load_injection_dataframe(live_dir=INJ_LIVE_DIR, pressure_col='PT 503', flow_col='Net Flow', filename=None):
    """Load injection CSV pair and return dataframe + labels.
    
    Args:
        live_dir: Directory with injection files
        pressure_col: Column name for pressure (e.g., 'PT 503' for TC interval)
        flow_col: Column name for flow (e.g., 'Net Flow')
        filename: Optional override filename to load (default: latest_INJ_data.csv)
    """
    import io
    
    if filename is None:
        filename = 'latest_INJ_data.csv'
    
    data_path = live_dir / filename
    if not data_path.exists():
        # Fall back to raw if downsampled doesn't exist
        if filename != 'latest_INJ_data.csv':
            return load_injection_dataframe(live_dir, pressure_col, flow_col, 'latest_INJ_data.csv')
        log.warning("No injection data file found at %s", data_path)
        return None, None

    # For the 1-min downsampled file, point at the raw metadata alias directly.
    # _resolve_metadata_path() does a naive 'data'->'metadata' replace that would
    # corrupt '_1min' filenames, and the file doesn't end with '_data.csv' anyway.
    if filename == 'latest_INJ_data_1min.csv':
        metadata_path = live_dir / 'latest_INJ_metadata.csv'
    elif filename.endswith('_data.csv'):
        metadata_path = _resolve_metadata_path(data_path)
    else:
        metadata_path = None

    # Read file and strip trailing commas to handle malformed CSVs
    try:
        with open(data_path, 'r') as f:
            lines = [line.rstrip('\r\n').rstrip(',') + '\n' for line in f]
        csv_text = ''.join(lines)
    except Exception as exc:
        log.warning("Failed to read injection file %s: %s", data_path, exc)
        return None, None

    # Prefer expected schema first: header row + units row + blank row.
    # Fall back to flexible parsing only if the strict parse quality is poor.
    attempts = [
        ("strict-skiprows", dict(skiprows=[1, 2], low_memory=False)),
        ("flexible", dict(low_memory=False)),
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

        # If strict mode already looks good, use it and stop.
        if mode == "strict-skiprows" and score >= 0.80:
            break

    if chosen_df is None or chosen_parsed_time is None:
        log.warning("Injection file %s has no parsable time column", data_path)
        return None, None

    log.info("Injection CSV parse mode=%s file=%s valid_time_fraction=%.3f", chosen_mode, filename, chosen_score)
    df = chosen_df
    df['Time'] = chosen_parsed_time
    df = df.dropna(subset=['Time'])
    if df.empty:
        log.warning("Injection file %s has no valid timestamp rows", data_path)
        return None, None

    # Verify requested columns exist
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


def get_data(variable, well, direction, length):
    start = get_start(direction, well)
    end = get_end(direction, well)
    no, unit = length.split()
    no = int(no)
    if unit == 'M':
        timedelta = np.timedelta64(no * 30, 'D')
    else:
        timedelta = np.timedelta64(no, unit)
    time_end = _DS.time[-1].values
    da = _DS['temperature'].sel(depth=slice(start, end), time=slice(time_end - timedelta, None))
    da = da.assign_coords(depth=da['depth'] - da['depth'][0])
    if variable == 'deltaT':
        da = da - da.isel(time=0)
        da.name = 'deltaT'
    return da



class Fiboreglass(pn.viewable.Viewer):
    variable = param.Selector(objects=['temperature', 'deltaT'], default='temperature')
    color_selector = param.Range((17, 28), bounds=(-10, 40), step=1)
    length_selector = param.Selector(objects=['12 h', '1 D', '2 D', '1 W', '3 W', '1 M', '2 M'], default='2 D')
    well_selector = param.Selector(objects=buttons, default='Whole fiber')
    direction_selector = param.Selector(objects=['Downgoing', 'Upgoing'], default='Downgoing')

    def __init__(self, **params):
        super().__init__(**params)
        self.da = get_data(self.variable, self.well_selector, self.direction_selector, self.length_selector)
        self._inj_data_path = None
        self._inj_mtime = None
        self.injection = None
        self.injection_labels = None
        self._refresh_injection_data()
        self._plot_pane = pn.panel(self._update_plot)
        self._layout = pn.Column(
            pn.Row(
                self.param.variable,
                        self.param.well_selector,
                        self.param.direction_selector,
                        align='center'),
                    pn.Row(
                self.param.length_selector,
                        self.param.color_selector,
                        align='center'),
            self._plot_pane
        )

    def _refresh_injection_data(self):
        data_path, _ = _find_latest_injection_pair(INJ_LIVE_DIR)
        if data_path is None:
            self.injection = None
            self.injection_labels = None
            self._inj_data_path = None
            self._inj_mtime = None
            return

        # Use the mtime of the 1-min downsampled file for staleness detection,
        # because that is the file actually served to the dashboard. The raw alias
        # (latest_INJ_data.csv) may be unchanged even when the pull script regenerates
        # the downsampled version from newly synced files.
        oneminfile = INJ_LIVE_DIR / 'latest_INJ_data_1min.csv'
        watch_path = oneminfile if oneminfile.exists() else data_path
        mtime = watch_path.stat().st_mtime
        if self._inj_data_path == watch_path and self._inj_mtime == mtime:
            return

        # Load 1-min downsampled version to get full data across all time windows
        df, labels = load_injection_dataframe(INJ_LIVE_DIR, filename='latest_INJ_data_1min.csv')
        self.injection = df
        self.injection_labels = labels
        self._inj_data_path = watch_path
        self._inj_mtime = mtime
        if df is None or labels is None:
            log.warning("Injection load failed for %s", data_path)
            return
        log.info(
            "Loaded injection data from 1-min downsampled file: rows=%d time_min=%s time_max=%s pressure_col=%s flow_col=%s",
            len(df),
            df['Time'].min(),
            df['Time'].max(),
            labels.get('pressure_col'),
            labels.get('flow_col'),
        )

    def _build_injection_plot(self, x_range=None):
        """Build combined pressure + flow injection panel with multi_y.

        The injection DataFrame's 'Time' column is renamed to 'time' (lowercase)
        before constructing the hv.Curve objects.  HoloViews dimension matching
        for shared_axes is case-sensitive: the heatmap x-dimension is 'time'
        (from the xarray coordinate), so using 'time' here causes HoloViews to
        wire the Bokeh Range1d objects natively and give synchronous pan/zoom.
        """
        if self.injection is None or self.injection_labels is None:
            return hv.Curve([], kdims=['time'], vdims=['pressure']).opts(
                responsive=True, show_grid=True,
            )

        pressure_col = self.injection_labels['pressure_col']
        flow_col = self.injection_labels['flow_col']
        pressure_unit = self.injection_labels['pressure_unit']
        flow_unit = self.injection_labels['flow_unit']

        # Rename 'Time' -> 'time' so the x-dimension matches the heatmap kdim.
        df = self.injection.rename(columns={'Time': 'time'})

        pressure = hv.Curve(
            df, 'time', pressure_col,
            label=f'{pressure_col} [{pressure_unit}]',
        ).opts(responsive=True, show_grid=True, color='firebrick')
        flow = hv.Curve(
            df, 'time', flow_col,
            label=f'{flow_col} [{flow_unit}]',
        ).opts(color='steelblue')
        return (pressure * flow).opts(multi_y=True, responsive=True)

    @param.depends('variable', 'color_selector', 'well_selector', 'direction_selector',
                   'length_selector')
    def _update_plot(self):
        # Any of the selections should produce a new set of plots
        self.da = get_data(self.variable, self.well_selector, self.direction_selector, self.length_selector)
        self._refresh_injection_data()
        # Reset colorbar values based on variable selection
        if self.variable == 'deltaT':
            self.color_selector = (-2, 2)
        elif self.variable == 'temperature':
            self.color_selector = (17, 28)
        dmap = rasterize(hv.QuadMesh(self.da, kdims=['time', 'depth']))
        dmap = dmap.apply.opts(clim=self.color_selector, cmap='BuRd_r', clabel=self.variable, apply_hard_bounds=True)
        # Make pointer stream
        pointer = hv.streams.Tap(x=self.da.time.values[0], y=self.da.depth.values[0], source=dmap)
        # Sections
        tsec = hv.DynamicMap(self.tap_timeseries, streams=[pointer])
        dsec = hv.DynamicMap(self.tap_depth_curve, streams=[pointer])

        # Gridspec
        gspec = pn.GridSpec(max_height=2000)
        main_plot = dmap.opts(tools=['hover'], responsive=True, colorbar=True, invert_yaxis=True, shared_axes=True)
        gspec[0, 1:4] = main_plot
        # Depth section
        if self.variable == 'temperature':
            gspec[:, 0] = dsec.opts(responsive=True, invert_axes=True, show_grid=True).redim.range(
                temperature=self.color_selector)
        elif self.variable == 'deltaT':
            gspec[:, 0] = dsec.opts(responsive=True, invert_axes=True, show_grid=True).redim.range(
                deltaT=self.color_selector)
        # Time section — shares x-axis with main_plot via shared_axes=True
        gspec[1, 1:4] = tsec.opts(responsive=True, ylim=self.color_selector, show_grid=True, shared_axes=True)

        # Injection panel — combined pressure+flow with multi_y.
        # 'time' kdim (lowercase) matches the heatmap dimension so HoloViews
        # wires the Bokeh Range1d natively via shared_axes=True.
        x_range_stream = hv.streams.RangeX(source=main_plot)
        injection_dmap = hv.DynamicMap(self._build_injection_plot, streams=[x_range_stream])
        gspec[2, 1:4] = injection_dmap.opts(show_grid=True, shared_axes=True)
        return gspec

    def tap_timeseries(self, x, y):
        return hv.Curve(self.da.sel(depth=y, method='nearest'), kdims=['time'], label=f'Depth: {y:0.3f}')

    def tap_depth_curve(self, x, y):
        return hv.Curve(self.da.sel(time=x, method='nearest'), kdims=['depth'], label=f'Time: {x}')

    def __panel__(self):
        return self._layout


fbg = Fiboreglass()
app = pn.template.VanillaTemplate(
    title='DTS Data Viewer', logo='/CUSSP.png', main=fbg).servable()
