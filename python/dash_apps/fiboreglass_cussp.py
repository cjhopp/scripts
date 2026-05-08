import panel as pn
import xarray as xr
import holoviews as hv
import numpy as np
import pandas as pd
import logging
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


_MAX_WINDOW = np.timedelta64(60, 'D')
_raw = xr.open_dataset('/data/chet-cussp/DTS/DTS_all.zarr', chunks={}, engine='zarr')
_time_end = _raw.time[-1].values
_DS = _raw.sel(time=slice(_time_end - _MAX_WINDOW, None)).load()
del _raw

INJ_LIVE_DIR = Path('/data/chet-cussp/injection/live')


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


def load_injection_dataframe(live_dir=INJ_LIVE_DIR):
    """Load latest injection CSV pair and return dataframe + labels."""
    data_path, metadata_path = _find_latest_injection_pair(live_dir)
    if data_path is None:
        log.warning("No injection data file found in %s", live_dir)
        return None, None

    flow_candidates = ['Triplex Flow', 'TV Flow', 'Net Flow', 'Quizix Flow']
    df = pd.read_csv(data_path, skiprows=[1, 2], low_memory=False)
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]

    if 'Time' not in df.columns:
        time_like = [c for c in df.columns if c.lower() == 'time' or c.lower().endswith(' time')]
        if time_like:
            df = df.rename(columns={time_like[0]: 'Time'})

    if 'Time' not in df.columns:
        log.warning("Injection file %s has no Time column", data_path)
        return None, None

    df['Time'] = pd.to_datetime(
        df['Time'],
        format='%m/%d/%y %H:%M:%S',
        errors='coerce',
    )
    df = df.dropna(subset=['Time'])
    if df.empty:
        log.warning("Injection file %s has no valid timestamp rows", data_path)
        return None, None

    pressure_col = 'PT 403' if 'PT 403' in df.columns else None
    if pressure_col is None:
        pt_candidates = [c for c in df.columns if c.upper().startswith('PT ')]
        pressure_col = pt_candidates[0] if pt_candidates else None

    flow_col = next((c for c in flow_candidates if c in df.columns), None)
    if flow_col is None:
        generic_flows = [c for c in df.columns if 'flow' in c.lower()]
        flow_col = generic_flows[0] if generic_flows else None

    if pressure_col is None:
        log.warning("Injection file %s has no pressure column", data_path)
        return None, None
    if flow_col is None:
        log.warning("Injection file %s has no flow column", data_path)
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
        'flow_unit': units.get(flow_col, 'LPM'),
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

        mtime = data_path.stat().st_mtime
        if self._inj_data_path == data_path and self._inj_mtime == mtime:
            return

        df, labels = load_injection_dataframe(INJ_LIVE_DIR)
        self.injection = df
        self.injection_labels = labels
        self._inj_data_path = data_path
        self._inj_mtime = mtime

    def _build_injection_plot(self):
        if self.injection is None or self.injection_labels is None:
            return hv.Text(0.5, 0.5, 'No injection data available').opts(
                responsive=True,
                xaxis=None,
                yaxis=None,
            )

        pressure_col = self.injection_labels['pressure_col']
        flow_col = self.injection_labels['flow_col']
        pressure_unit = self.injection_labels['pressure_unit']
        flow_unit = self.injection_labels['flow_unit']

        pressure = hv.Curve(
            self.injection,
            'Time',
            pressure_col,
            label=f'{pressure_col} [{pressure_unit}]',
        ).opts(responsive=True, show_grid=True, color='firebrick')
        flow = hv.Curve(
            self.injection,
            'Time',
            flow_col,
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
        gspec[0, 1:4] = dmap.opts(tools=['hover'], responsive=True, colorbar=True, invert_yaxis=True)
        # Depth section (relim just this panel, since the time series should twin this range)
        if self.variable == 'temperature':
            gspec[:, 0] = dsec.opts(responsive=True, invert_axes=True, show_grid=True).redim.range(
                temperature=self.color_selector)
        elif self.variable == 'deltaT':
            gspec[:, 0] = dsec.opts(responsive=True, invert_axes=True, show_grid=True).redim.range(
                deltaT=self.color_selector)
        # Time section
        gspec[1, 1:4] = tsec.opts(responsive=True, ylim=self.color_selector, show_grid=True)
        # Accessory plot
        gspec[2, 1:4] = self._build_injection_plot()
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
