#!/usr/bin/python

import pyproj

import panel as pn
import holoviews as hv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from glob import glob
from datetime import datetime
from scipy.interpolate import interp1d
from holoviews.selection import link_selections
from holoviews.core.data.interface import DataError


wellpath = '/home/chopp/data/chet-meq/newberry/boreholes/55-29/GDR_submission/Deviation_corrected_with-depth.csv'
injection_path = '/home/chopp/data/chet-meq/newberry/boreholes/injection_data/One Second Data Listing - Time - Pressure - Rate Example.csv'
old_injection_path = '/home/chopp/data/chet-meq/newberry/boreholes/injection_data/2014'

# Stim depths converted to meters
stages = np.array([9005., 9070., 9170., 9270., 9370., 9470., 9570., 9670., 9770., 9870.]) * 0.3048

hv.extension('bokeh', 'plotly', 'matplotlib')

def get_data():
    cli = Client('http://131.243.224.19:8085')
    starttime = UTCDateTime(2014, 1, 1)
    endtime = UTCDateTime(2015, 1, 1)
    # starttime = UTCDateTime(2024, 11, 1)
    # endtime = UTCDateTime()  # Current time
    newb_catalog = cli.get_events(starttime=starttime, endtime=endtime,
                                  latitude=43.726, longitude=-121.316, maxradius=0.05,
                                  includeallmagnitudes=True)
    newb_catalog.events.sort(key=lambda x: x.preferred_origin().time)
    return newb_catalog


def get_injection(path):
    injection = pd.read_csv(path, skiprows=15, names=['time', 'psi', 'bpm'], index_col=0, usecols=[0, 1, 2])
    injection['bps'] = injection['bpm'] / 60
    injection['cumulative [bbl]'] = injection['bps'].cumsum()  # Assumes 1 Hz data
    return hv.Dataset(injection)


def get_old_injection(directory):
    injection = pd.DataFrame()
    for excel in glob('{}/*.xlsx'.format(directory)):
        injection = pd.concat([injection, pd.read_excel(excel)])
    ## Something abovwe here isn't working right. Cumulative sum returning NaN
    injection = injection.set_index('Date + Time')
    injection['cumulative [gal]'] = injection['Corrected UltraSonic (gpm)'].cumsum()
    return hv.Dataset(injection)


def ecdf_transform(data):
    return len(data) - data.rank(method="first")


def calc_max_curv(magnitudes):
    """
    Stolen from EQcorrscan to avoid dependency
    """
    bin_size = 0.1
    min_bin, max_bin = int(min(magnitudes)), int(max(magnitudes) + 1)
    bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    df, bins = np.histogram(magnitudes, bins)
    grad = (df[1:] - df[0:-1]) / bin_size
    # Need to find the second order derivative
    curvature = (grad[1:] - grad[0:-1]) / bin_size
    max_curv = bins[np.argmax(np.abs(curvature))] + bin_size
    return float(max_curv)


def bval_wrapper(ser: pd.Series, df: pd.DataFrame):
    df_roll = df.loc[ser.index]
    return calc_bval(df_roll)


def calc_bval(df):
    mc = calc_max_curv(df['magnitude'])
    filtered = df.loc[(df['magnitude'] > mc) & (df['cumulative number'] > 0.)]
    b, a = np.polyfit(filtered['magnitude'], np.log10(filtered['cumulative number']), 1)
    return b


def trajectory_wrapper(ser: pd.Series, df: pd.DataFrame, injection_pt):
    df_roll = df.loc[ser.index]
    return calc_trajectory(df_roll, injection_pt)


def calc_trajectory(dataset, injection_pt):
    pts = dataset[['east', 'north', 'elevation']].values
    centroid = np.median(pts, axis=0)
    vector = injection_pt - centroid
    azrad = np.rad2deg(np.arctan2(vector[1], vector[0]))
    plunge = np.rad2deg(np.arctan2(vector[2], np.sqrt(np.sum(vector[:2]**2))))
    if azrad < 0:
        azrad += 360
        plunge *= -1
    return azrad


def plunge_wrapper(ser: pd.Series, df: pd.DataFrame, injection_pt):
    df_roll = df.loc[ser.index]
    return calc_plunge(df_roll, injection_pt)


def calc_plunge(dataset, injection_pt):
    pts = dataset[['east', 'north', 'elevation']].values
    centroid = np.median(pts, axis=0)
    vector = injection_pt - centroid
    azrad = np.rad2deg(np.arctan2(vector[1], vector[0]))
    plunge = -np.rad2deg(np.arctan2(vector[2], np.sqrt(np.sum(vector[:2]**2))))
    if azrad < 0:
        azrad += 360
    return plunge


def get_seismic_events(catalog):
    utm = pyproj.Proj("EPSG:32610")
    well = np.loadtxt(wellpath, delimiter=',', skiprows=1)
    params = np.array([(ev.preferred_origin().longitude, ev.preferred_origin().latitude, ev.preferred_origin().depth,
                        ev.preferred_magnitude().mag, float(ev.preferred_origin().time.timestamp))
                        for ev in catalog if ev.preferred_origin() != None and ev.preferred_magnitude() != None])
    dataset = pd.DataFrame({'latitude': params[:, 1], 'longitude': params[:, 0], 'depth': params[:, 2],
                            'magnitude': params[: , 3], 'marker size': (params[: , 3] + 1)**2,
                            'timestamp': params[:, 4]})
    dataset["cumulative number"] = dataset['magnitude'].transform(ecdf_transform)
    # Rolling b values (this overwrites all columns with the result for some reason...take a random column
    dataset['b'] = -1. * dataset.rolling(100, center=False).apply(bval_wrapper, args=(dataset,))['latitude']
    # Overall bval calculation
    Mc = calc_max_curv(dataset['magnitude'].values)
    points = dataset.loc[(dataset['magnitude'] > Mc) & (dataset['cumulative number'] > 0.)]
    east, north = utm(dataset['longitude'], dataset['latitude'])
    east -= well[0][0]
    north -= well[0][1]
    elevation = params[:, 2] * -1.
    dataset['east'] = east
    dataset['north'] = north
    dataset['elevation'] = elevation
    # Rolling trajectory
    dataset['trend'] = dataset.rolling(25, center=False).apply(
        trajectory_wrapper, args=(dataset, np.array([511, -10, -1184.])))['latitude']
    dataset['plunge'] = dataset.rolling(25, center=False).apply(
        plunge_wrapper, args=(dataset, np.array([511, -10, -1184.])))['latitude']
    b, a = np.polyfit(points['magnitude'], np.log10(points['cumulative number']), 1)
    dataset['b label'] = ['b value: {}'.format(-b) for i in range(len(dataset))]
    dataset['bfit'] = 10**(np.log10(len(points)) + b * points['magnitude'])
    dataset['cumulative max mag'] = dataset['magnitude'].cummax()
    return hv.Dataset(dataset)


def get_wellpath(wellpath):
    liner_color = {'cased': 'black', 'slotted': 'red'}
    liner_style = {'cased': 'solid', 'slotted': 'dotted'}
    # This should produce a DataFrame with
    slotted = {'cased': [(0, 1912), (2289, 2493)],
               'slotted': [(1912, 2289), (2493, 3045)]}
    wellpath = np.loadtxt(wellpath, delimiter=',', skiprows=1)
    wellpath[:, 0] -= wellpath[0, 0]
    wellpath[:, 1] -= wellpath[0, 1]
    fx = interp1d(wellpath[:, 2], wellpath[:, 0])
    fy = interp1d(wellpath[:, 2], wellpath[:, 1])
    fd = interp1d(wellpath[:, 2], wellpath[:, 4])
    # Interpolate onto meter depth spacing
    new_z = np.linspace(wellpath[-1, 2], wellpath[0, 2], 3009)[::-1]
    well_ds = pd.DataFrame({'east': fx(new_z), 'north': fy(new_z), 'elevation': new_z, 'depth': fd(new_z)})
    map_paths = []
    NS_paths = []
    EW_paths = []
    for liner, intervals in slotted.items():
        for s in intervals:
            curve = hv.Curve(
                well_ds.query('{} > depth and {} < depth'.format(s[1], s[0])),
                'east', 'north').opts(color=liner_color[liner], line_dash=liner_style[liner], line_width=3.)
            map_paths.append(curve)
            curve = hv.Curve(
                well_ds.query('{} > depth and {} < depth'.format(s[1], s[0])),
                'north', 'elevation').opts(color=liner_color[liner], line_dash=liner_style[liner], line_width=3.)
            NS_paths.append(curve)
            curve = hv.Curve(
                well_ds.query('{} > depth and {} < depth'.format(s[1], s[0])),
                'east', 'elevation').opts(color=liner_color[liner], line_dash=liner_style[liner], line_width=3.)
            EW_paths.append(curve)
    for stage in stages:
        idx = (well_ds['depth'] - stage).abs().idxmin()
        row = well_ds.loc[idx]
        scat_map = hv.Scatter((row['east'], row['north'])).opts(size=15., color='black', marker='dash')
        map_paths.append(scat_map)
        scat_NS = hv.Scatter((row['north'], row['elevation'])).opts(size=15., color='black', marker='dash')
        NS_paths.append(scat_NS)
        scat_EW = hv.Scatter((row['east'], row['elevation'])).opts(size=15., color='black', marker='dash')
        EW_paths.append(scat_EW)
    return map_paths, NS_paths, EW_paths


def injection_plot(injection):
    """
    Plot flow rate and pressure
    :return:
    """
    try:
        flow = hv.Curve(injection, 'time', vdims=['bpm']).opts(color='firebrick')
        pressure = hv.Curve(injection, 'time', vdims=['psi']).opts(color='steelblue')
    except DataError:
        flow = hv.Curve(injection, 'Date + Time', vdims=['Corrected UltraSonic (gpm)']).opts(color='firebrick')
        pressure = hv.Curve(injection, 'Date + Time', vdims=['WHP Corrected (psi)']).opts(color='steelblue')
    return (flow * pressure).opts(backend='bokeh', multi_y=True, bgcolor='whitesmoke', responsive=True)


def plot_current_trajectory(dataset):
    """
    Calculate the average trajectory of the last 25(?) events
    :param dataset:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.scatter(np.deg2rad(dataset['trend']), dataset['plunge'], marker='o', alpha=0.5, s=5.,
               c=dataset['timestamp'], cmap='cividis')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim([90, 0])
    ax.set_title('Seismic trajectory (upper hemisphere)')
    return fig


def plot_vol_moment(dataset, injection):
    gamma = 1.5e8
    gar_max = []
    galis_max = []
    vols = np.linspace(1E2, 3E7, 100)
    for v in vols:
        gar_max.append(v * 3E10)  # G = 3E10 Pa from McGarr 2014
        galis_max.append(v**(3/2) * gamma)
    garr = hv.Curve(zip(vols, gar_max), label='McGarr [30 GPa]').opts(color='black', responsive=True, backend='bokeh')
    galis = hv.Curve(zip(vols, galis_max), label='Galis [y=1.5e8]').opts(color='black', line_dash='dashed', backend='bokeh')
    m3 = 0.1589872949  # Conversion factor to m3 from bbl
    m3 = 0.00378541
    mw_max = np.max(dataset['magnitude'])
    m0 = 10.0 ** (1.5 * mw_max + 9.0 )
    cum_vol = np.nanmax(injection['cumulative [gal]']) * m3
    # Interpolate cumulative max mag onto cumulative volume
    vol = injection.dframe()['cumulative [gal]'] * m3
    vol = vol[~vol.index.duplicated(keep='first')]
    cat = dataset.dframe()
    cat['Date + Time'] = pd.to_datetime(cat['timestamp'], unit='s')
    cat = cat.set_index('Date + Time')
    cat['cumulative max m0'] = 10.0 ** (1.5 * cat['cumulative max mag'] + 9.0 )
    df = pd.concat([vol, cat['cumulative max m0']], axis=1)
    df = df.rename(columns={'cumulative [gal]': 'volume', 'cumulative max m0': 'magnitude'})
    df = df.assign(magnitude=lambda x: x['magnitude'].interpolate())
    df = df.dropna()
    max = hv.Scatter([(cum_vol, m0)]).opts(
        logx=True, logy=True, marker='circle_x', size=10, color='red', alpha=0.2,
        xlabel='Cumulative volume [m^3]', ylabel='Maximum moment', xlim=(1e2, 3e7),
        bgcolor='whitesmoke')
    max_running = hv.Curve(hv.Dataset(df), 'volume', vdims=['magnitude'])
    return garr * galis * max * max_running


def gr_plot(dataset):
    Mc = calc_max_curv(dataset['magnitude'])
    gr = hv.Scatter(dataset, 'magnitude', vdims=['cumulative number']).opts(
        responsive=True, logy=True, bgcolor='whitesmoke', color='darkgray', backend='bokeh')
    fit = hv.Curve(dataset.iloc[dataset['magnitude'] > Mc].sort('magnitude'),
                   'magnitude', vdims=['bfit']).opts(
        color='red', logy=True, title='b value: {:.2f}'.format(float(dataset['b label'][0].split()[-1])),
        backend='bokeh', ylim=(0, 3000))
    return gr * fit


def mag_time_plot(dataset):
    cat = dataset.dframe()
    cat['Date + Time'] = pd.to_datetime(cat['timestamp'], unit='s')
    cat = cat.set_index('Date + Time')
    plot = hv.Scatter(hv.Dataset(cat), 'Date + Time', vdims=['magnitude']).opts(
        responsive=True, bgcolor='whitesmoke', color='black', backend='bokeh')
    return plot


def bval_time_plot(dataset):
    cat = dataset.dframe()
    cat['Date + Time'] = pd.to_datetime(cat['timestamp'], unit='s')
    cat = cat.set_index('Date + Time')
    plot = hv.Curve(hv.Dataset(cat), 'Date + Time', vdims=['b']).opts(
        responsive=True, bgcolor='whitesmoke', color='firebrick', backend='bokeh')
    return plot


def all_time_plot(dataset):
    bplot = bval_time_plot(dataset)
    mplot = mag_time_plot(dataset)
    return (bplot * mplot).opts(backend='bokeh', multi_y=True)


def seismicity_3d(dataset):
    tickvals = np.linspace(dataset.range('timestamp')[0], dataset.range('timestamp')[1], 10)
    ticktext = [UTCDateTime(t).strftime('%d %b %Y: %H:%M')
                for t in tickvals]
    cticks = [(tv, ticktext[i]) for i, tv in enumerate(tickvals)]
    # Map view, two cross-sections, and 3D
    mapview = hv.Scatter(
        dataset, 'east', vdims=['north', 'timestamp', 'magnitude', 'marker size']).opts(
        marker='circle',
        color='timestamp',
        size='marker size',
        colorbar=True,
        cticks=cticks,
        cmap='Cividis',
        xlim=(-2000, 2000), ylim=(-2000, 2000),
        responsive=True,
        bgcolor='whitesmoke',
        backend='bokeh',
    )
    NS = hv.Scatter(
        dataset, 'north', vdims=['elevation', 'timestamp', 'magnitude', 'marker size']).opts(
        marker='circle',
        color='timestamp',
        size='marker size',
        colorbar=True,
        cticks=cticks,
        cmap='Cividis',
        xlim=(-2000, 2000), ylim=(-2000, 2000),
        responsive=True,
        bgcolor='whitesmoke',
        backend='bokeh',
    )
    EW = hv.Scatter(
        dataset, 'east', vdims=['elevation', 'timestamp', 'magnitude', 'marker size']).opts(
        marker='circle',
        color='timestamp',
        size='marker size',
        colorbar=True,
        cticks=cticks,
        cmap='Cividis',
        xlim=(-2000, 2000), ylim=(-2000, 2000),
        responsive=True,
        bgcolor='whitesmoke',
        backend='bokeh',
    )
    map_paths, NS_paths, EW_paths = get_wellpath(wellpath)
    well_map = hv.Overlay(map_paths, 'east', 'north').opts(
        xlim=(-2000, 2000), ylim=(-2000, 2000), backend='bokeh')
    well_NS = hv.Overlay(NS_paths).opts(
        xlim=(-2000, 2000), ylim=(-2200, 1800), backend='bokeh')
    well_EW = hv.Overlay(EW_paths).opts(
        xlim=(-2000, 2000), ylim=(-2200, 1800), backend='bokeh')
    return well_map, mapview, well_NS, NS, well_EW, EW


class DailyReport(pn.viewable.Viewer):

    def __init__(self, **params):
        super().__init__(**params)
        self.catalog = get_data()
        self.wellpath = get_injection(wellpath)
        self.dataset = get_seismic_events(self.catalog)
        # self.injection = get_injection(injection_path)
        self.injection = get_old_injection(old_injection_path)
        # Full configuration
        full_config = pn.widgets.TextEditor(toolbar=[
            ['bold', 'italic', 'underline'],  # toggled buttons
            ['blockquote', 'code-block'],

            [{'header': 1}, {'header': 2}],  # custom button values
            [{'list': 'ordered'}, {'list': 'bullet'}],
            [{'script': 'sub'}, {'script': 'super'}],  # superscript/subscript
            [{'indent': '-1'}, {'indent': '+1'}],  # outdent/indent
            [{'direction': 'rtl'}],  # text direction

            [{'size': ['small', False, 'large', 'huge']}],  # custom dropdown
            [{'header': [1, 2, 3, 4, 5, 6, False]}],

            [{'color': []}, {'background': []}],  # dropdown with defaults from theme
            [{'font': []}],
            [{'align': []}],

            ['clean']  # remove formatting button
        ])
        linked_plots, time_plots, polar_plot, inj_plot = self._link_plots()
        injection_panel = injection_plot(self.injection)
        save_button = pn.widgets.Button(name='Save report', button_type='primary')
        save_button.on_click(self._save)
        self.button_pane = pn.Row(save_button)
        self.row1 = pn.Row(linked_plots, height=1500)
        self.row2 = pn.Row(inj_plot, polar_plot, height=700)
        self.row3 = pn.Row(time_plots, height=500)
        self.row4 = pn.Row(injection_panel, height=500)
        self.row5 = pn.Row('# Discussion', full_config, align='center', height=500)
        self._layout = pn.Column(
            self.button_pane,
            self.row1, self.row2, self.row3, self.row4, self.row5,
            align='center', scroll=True,
        )


    def _save(self, event):
        from bokeh.resources import INLINE, CDN
        print('Saving file to Newberry_report.html')
        # row3 = self.row3
        # row3.embed()
        col = pn.Column('Newberry Report: {}'.format(UTCDateTime().date),
                        self.row1, self.row2, self.row3, self.row4)
        col.save('Newberry_report.html', resources=INLINE)


    def _link_plots(self):
        sel = link_selections.instance()
        map_well, map_seis, NS_well, NS_seis, EW_well, EW_seis = seismicity_3d(self.dataset)
        grich = gr_plot(self.dataset)
        injection_plot = pn.pane.HoloViews(plot_vol_moment(self.dataset, self.injection))
        polar_plot = pn.pane.Matplotlib(plot_current_trajectory(self.dataset), dpi=200, tight=True,
                                        sizing_mode="stretch_height")
        all_time = all_time_plot(self.dataset)
        layout = (sel(map_seis, index_cols=['timestamp']) * map_well + grich +
                  sel(NS_seis, index_cols=['timestamp']) * NS_well +
                  sel(EW_seis, index_cols=['timestamp']) * EW_well).cols(2)# + polar_plot).cols(2)
        return layout, all_time, polar_plot, injection_plot


    def __panel__(self):
        return self._layout


pan = DailyReport()
app = pn.template.VanillaTemplate(
    title='Newberry Report: {}'.format(UTCDateTime().date), logo='', main=pan).servable()
