import panel as pn
import xarray as xr
import holoviews as hv
import numpy as np

import param

from holoviews.operation.datashader import rasterize

hv.extension('bokeh', config=dict(image_rtol=10000))

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


def get_data(variable, well, direction, length):
    ds = xr.open_dataset('/media/chopp/Data1/chet-cussp/DTS/DTS_all.zarr', chunks={'depth': 1000}, engine='zarr')
    ds['deltaT'] = ds['temperature'] - ds['temperature'].isel(time=0)
    start = get_start(direction, well)
    end = get_end(direction, well)
    no, unit = length.split()
    timedelta = np.timedelta64(int(no), unit)
    time_end = ds.time[-1].values
    da = ds[variable].sel(depth=slice(start, end), time=slice(time_end - timedelta, None))
    da['depth'] = da['depth'] - da['depth'][0]
    return da



class Fiboreglass(pn.viewable.Viewer):
    variable = param.Selector(objects=['temperature', 'deltaT'], default='temperature')
    color_selector = param.Range((17, 28), bounds=(-10, 40), step=1)
    delta_selector = param.Range((-.5, .5), bounds=(-2, 2), step=0.1)
    length_selector = param.Selector(objects=['12 h', '1 D', '2 D', '1 W', '3 W', '10 W', '20 W', '56 W'], default='2 D')
    well_selector = param.Selector(objects=buttons, default='Whole fiber')
    direction_selector = param.Selector(objects=['Downgoing', 'Upgoing'], default='Downgoing')

    def __init__(self, **params):
        super().__init__(**params)
        self.da = get_data(self.variable, self.well_selector, self.direction_selector, self.length_selector)
        self.injection = xr.open_dataset('/media/chopp/Data1/chet-cussp/injection_FTES/netcdf/FTES_injection_1min.nc')
        self._plot_pane = self._update_plot
        self._layout = pn.Column(
            pn.Row(
                self.param.variable,
                        self.param.well_selector,
                        self.param.direction_selector,
                        align='center'),
                    pn.Row(
                self.param.length_selector,
                        self.param.color_selector,
                        self.param.delta_selector,
                        align='center'),
            self._plot_pane
        )

    @param.depends('variable', 'color_selector', 'well_selector', 'direction_selector',
                   'length_selector')
    def _update_plot(self):
        # Any of the selections should produce a new set of plots
        self.da = get_data(self.variable, self.well_selector, self.direction_selector, self.length_selector)
        # Reset colorbar values based on variable selection
        if self.variable == 'deltaT':
            self.delta_selector = (-0.5, 0.5)
        elif self.variable == 'temperature':
            self.color_selector = (17, 28)
        dmap = rasterize(hv.QuadMesh(self.da, kdims=['time', 'depth']))
        if self.variable == 'temperature':
            dmap = dmap.apply.opts(clim=self.color_selector, cmap='BuRd_r', clabel=self.variable,
                                   apply_hard_bounds=True)
        elif self.variable == 'deltaT':
            dmap = dmap.apply.opts(clim=self.delta_selector, cmap='BuRd_r', clabel=self.variable,
                                   apply_hard_bounds=True)
        # Make pointer stream
        pointer = hv.streams.Tap(x=self.da.time.values[0], y=self.da.depth.values[0], source=dmap)
        # Sections
        tsec = hv.DynamicMap(self.tap_timeseries, streams=[pointer])
        dsec = hv.DynamicMap(self.tap_depth_curve, streams=[pointer])
        ref_dsec = hv.Curve(self.da.isel(time=0), kdims=['depth'], label='Reference')
        # Gridspec
        gspec = pn.GridSpec(max_height=2000)
        gspec[0, 1:4] = dmap.opts(tools=['hover'], responsive=True, colorbar=True, invert_yaxis=True)
        # Depth section (relim just this panel, since the time series should twin this range)
        if self.variable == 'temperature':
            gspec[:, 0] = (ref_dsec * dsec).opts(responsive=True, invert_axes=True, show_grid=True).redim.range(
                temperature=self.color_selector)
            # Time section
            gspec[1, 1:4] = tsec.opts(responsive=True, ylim=self.color_selector, show_grid=True)
        elif self.variable == 'deltaT':
            gspec[:, 0] = (ref_dsec * dsec).opts(responsive=True, invert_axes=True, show_grid=True).redim.range(
                deltaT=self.delta_selector)
            # Time section
            gspec[1, 1:4] = tsec.opts(responsive=True, ylim=self.delta_selector, show_grid=True)
        # Accessory plot
        gspec[2, 1:4] = (hv.Curve(self.injection, 'time', 'PT 403', label='Pressure').opts(
                            responsive=True, show_grid=True, color='firebrick') *
                         hv.Curve(self.injection, 'time', 'Triplex Flow', label='Flow').opts(
                             color='steelblue'
                         )).opts(multi_y=True)
        return gspec

    def tap_timeseries(self, x, y):
        return hv.Curve(self.da.sel(depth=y, method='nearest'), kdims=['time'], label=f'Depth: {y:0.3f}')

    def tap_depth_curve(self, x, y):
        return hv.Curve(self.da.sel(time=x, method='nearest'), kdims=['depth'], label=f'Time: {x}')

    def __panel__(self):
        return self._layout


fbg = Fiboreglass()
app = pn.template.VanillaTemplate(
    title='DTS Data Viewer', logo='/home/chopp/CUSSP.png', main=fbg).servable()
