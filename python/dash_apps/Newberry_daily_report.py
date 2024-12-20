#!/usr/bin/python

import param
import pyproj

import panel as pn
import seaborn as sns
import holoviews as hv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from copy import deepcopy
from datetime import datetime
from matplotlib import gridspec
from holoviews.selection import link_selections
from bokeh.palettes import RdYlBu
from bokeh.models import DatetimeTickFormatter


wellpath = '/media/chopp/Data1/chet-meq/newberry/boreholes/55-29/GDR_submission/Deviation_corrected.csv'
injection_path = '/media/chopp/Data1/chet-meq/newberry/boreholes/injection_data/One Second Data Listing - Time - Pressure - Rate Example.csv'

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
    return newb_catalog


def get_injection(path):
    injection = pd.read_csv(path, skiprows=15, names=['time', 'psi', 'bpm'], index_col=0, usecols=[0, 1, 2])
    injection['bps'] = injection['bpm'] / 60
    injection['cumulative'] = injection['bps'].cumsum()  # Assumes 1 Hz data
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


def cantor_pairing(x, y):
    return (x + y) * (x + y + 1) // 2 + y


def cantor_unpairing(z):
    try:
        w = int((pow(8 * z + 1, 0.5) - 1) // 2)
    except ValueError:
        return (np.nan, np.nan)
    t = (w * w + w) // 2
    y = z - t
    x = w - y
    return x, y


def trajectory_wrapper(ser: pd.Series, df: pd.DataFrame):
    df_roll = df.loc[ser.index]
    return calc_trajectory(df_roll)


def calc_trajectory(dataset):
    pts = dataset[['east', 'north', 'elevation']].values
    centroid = np.mean(pts)
    pts_centered = pts - centroid
    U, S, Vt = np.linalg.svd(pts_centered)
    vector = Vt[0]
    azrad = np.rad2deg(np.arctan2(vector[1], vector[0]))
    plunge = np.rad2deg(np.arctan2(vector[2], np.sqrt(np.sum(vector[:2]**2))))
    if azrad < 0:
        azrad += 360
        plunge *= -1
    paired = cantor_pairing(azrad, plunge)
    return paired


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
    dataset['b'] = -1. * dataset.rolling(100).apply(bval_wrapper, args=(dataset,))['latitude']
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
    dataset['trajectory encoded'] = dataset.rolling(25).apply(trajectory_wrapper, args=(dataset,))['latitude']
    trend, plunge = zip(*dataset['trajectory encoded'].apply(cantor_unpairing).values)
    dataset['trend'] = np.deg2rad(trend)
    dataset['plunge'] = np.deg2rad(plunge)
    b, a = np.polyfit(points['magnitude'], np.log10(points['cumulative number']), 1)
    dataset['b label'] = ['b value: {}'.format(-b) for i in range(len(dataset))]
    dataset['bfit'] = 10**(np.log10(len(points)) + b * points['magnitude'])
    return hv.Dataset(dataset)


def get_wellpath(wellpath):
    elev_wh = 1703.2488  # Wellhead elevation
    slotted = [(0, 1912), (1912, 2289), (2289, 2493), (2493, 3045)]
    section_color = ['black', 'red', 'black', 'red']
    section_styles = ['solid', 'dot', 'solid', 'dot']
    wellpath = np.loadtxt(wellpath, delimiter=',', skiprows=1)
    wellpath[:, 0] -= wellpath[0, 0]
    wellpath[:, 1] -= wellpath[0, 1]
    well_ds = pd.DataFrame({'east': wellpath[:, 0], 'north': wellpath[:, 1], 'elevation': wellpath[:, 2]})
    paths = []
    for i, s in enumerate(slotted):
        ds = hv.Dataset(
            well_ds.query('{} > elevation and {} < elevation'.format(elev_wh - s[0], elev_wh - s[1])))
        paths.append(ds)
    return paths


def injection_plot(injection):
    """
    Plot flow rate and pressure
    :return:
    """
    flow = hv.Curve(injection, 'time', vdims=['bpm']).opts(color='firebrick')
    pressure = hv.Curve(injection, 'time', vdims=['psi']).opts(color='steelblue')
    return (flow * pressure).opts(backend='bokeh', multi_y=True, bgcolor='whitesmoke', responsive=True)


def plot_current_trajectory(dataset):
    """
    Calculate the average trajectory of the last 25(?) events
    :param dataset:
    :return:
    """
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='polar')
    # ax.scatter(dataset['trend'], dataset['plunge'])
    # plt.show()
    polar = hv.Scatter(dataset, 'trend', vdims=['plunge', 'timestamp']).opts(
        responsive=True,
        color='timestamp',
        cmap='Cividis',
        bgcolor='whitesmoke',
        backend='bokeh',
        projection='polar',
    )
    return polar


def plot_vol_moment(dataset, injection):
    mmax = []
    vols = np.linspace(1E2, 3E7, 100)
    for v in vols:
        mmax.append(v * 3E10)  # G = 3E10 Pa from McGarr 2014
    garr = hv.Curve(zip(vols, mmax)).opts(color='lightgray', responsive=True, backend='bokeh')
    m3 = 0.1589872949  # Conversion factor to m3 from bbl
    max = hv.Scatter([(np.max(dataset['magnitude']), injection['cumulative'][-1] * m3)])
    # ax.text(x=5E2, y=3E13, s='$M_{o}=GV$', rotation=28., color='dimgray',
    #         horizontalalignment='center', verticalalignment='center')
    return garr * max


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
    tickvals = np.linspace(dataset.range('timestamp')[0], dataset.range('timestamp')[1], 10)
    ticktext = [UTCDateTime(t).strftime('%d %b %Y: %H:%M')
                for t in tickvals]
    xticks = [(tv, ticktext[i]) for i, tv in enumerate(tickvals)]
    plot = hv.Scatter(dataset, 'timestamp', vdims=['magnitude']).opts(
        responsive=True, bgcolor='whitesmoke', color='timestamp', cmap='Cividis', backend='bokeh', xticks=xticks)
    return plot


def bval_time_plot(dataset):
    tickvals = np.linspace(dataset.range('timestamp')[0], dataset.range('timestamp')[1], 10)
    ticktext = [UTCDateTime(t).strftime('%d %b %Y: %H:%M')
                for t in tickvals]
    xticks = [(tv, ticktext[i]) for i, tv in enumerate(tickvals)]
    plot = hv.Curve(dataset, 'timestamp', vdims=['b']).opts(
        responsive=True, bgcolor='whitesmoke', color='firebrick', xticks=xticks)
    return plot


def all_time_plot(dataset):
    bplot = bval_time_plot(dataset)
    mplot = mag_time_plot(dataset)
    tickvals = np.linspace(dataset.range('timestamp')[0], dataset.range('timestamp')[1], 10)
    ticktext = [UTCDateTime(t).strftime('%d %b %Y: %H:%M')
                for t in tickvals]
    xticks = [(tv, ticktext[i]) for i, tv in enumerate(tickvals)]
    return (bplot * mplot).opts(xticks=xticks, backend='bokeh', multi_y=True)


def seismicity_3d(dataset):
    section_color = ['black', 'red', 'black', 'red']
    section_styles = ['solid', 'dot', 'solid', 'dot']
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
    seis3D = hv.Scatter3D(
        dataset, kdims=['east', 'north', 'elevation'], vdims=['timestamp', 'magnitude']).opts(
        marker='circle',
        color='timestamp',
        size=dataset['marker size'],
        colorbar=True,
        colorbar_opts=dict(ticktext=ticktext, tickvals=tickvals),
        cmap='Cividis',
        responsive=True,
        # bgcolor='whitesmoke',
        backend='plotly'
        # colorbar_opts=dict(
        #     title='Timestamp', ticktext=ticktext, tickvals=tickvals)
    )
    paths = get_wellpath(wellpath)
    well_map = hv.Curve(paths, 'east', 'north').opts(
        # color=section_color, dash=section_styles, line_width=3.,
        xlim=(-2000, 2000), ylim=(-2000, 2000), backend='bokeh')
    well_NS = hv.Curve(paths, 'north', 'elevation').opts(
        # color=section_color, dash=section_styles, line_width=3.,
        xlim=(-2000, 2000), ylim=(-2200, 1800), backend='bokeh')
    well_EW = hv.Curve(paths, 'east', 'elevation').opts(
        # color=section_color, dash=section_styles, line_width=3.,
        xlim=(-2000, 2000), ylim=(-2200, 1800), backend='bokeh')
    well_3d = hv.Path3D(paths, kdims=['east', 'north', 'elevation']).opts(
        # color=section_color, dash=section_styles, line_width=3.,
        xlim=(-2000, 2000), ylim=(-2000, 2000), zlim=(-2200, 1800),
        width=800, height=1200, responsive=True, backend='plotly')
    return well_map, mapview, well_NS, NS, well_EW, EW#, well_3d * seis3D


class DailyReport(pn.viewable.Viewer):

    def __init__(self, **params):
        super().__init__(**params)
        self.catalog = get_data()
        self.wellpath = get_injection(wellpath)
        self.dataset = get_seismic_events(self.catalog)
        self.injection = get_injection(injection_path)
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
        linked_plots, time_plots = self._link_plots()
        injection_panel = injection_plot(self.injection)
        save_button = pn.widgets.Button(name='Save report', button_type='primary')
        save_button.on_click(self._save)
        self.button_pane = pn.Row(save_button)
        self.row1 = pn.Row(linked_plots, height=2000)
        self.row2 = pn.Row(time_plots, height=500)
        self.row3 = pn.Row(injection_panel, height=500)
        self.row4 = pn.Row('# Discussion', full_config, align='center', height=500)
        self._layout = pn.Column(
            self.button_pane,
            self.row1, self.row2, self.row3, self.row4,
            align='center', scroll=True,
        )


    def _save(self, event):
        from bokeh.resources import INLINE, CDN
        print('Saving file to Newberry_report.png')
        row3 = self.row3
        row3.embed()
        col = pn.Column('Newberry Report: {}'.format(UTCDateTime().date),
                        self.row1, self.row2, row3)
        col.save('Newberry_report.html', resources=INLINE)


    def _link_plots(self):
        sel = link_selections.instance()
        # Gridspec
        gspec = pn.GridSpec(max_height=2000)
        # map, NS, EW, threeD = seismicity_3d(self.dataset)
        map_well, map_seis, NS_well, NS_seis, EW_well, EW_seis = seismicity_3d(self.dataset)
        grich = gr_plot(self.dataset)
        injection_plot = plot_vol_moment(self.dataset, self.injection)
        polar_plot = plot_current_trajectory(self.dataset)
        all_time = all_time_plot(self.dataset)
        layout = (sel(map_seis, index_cols=['timestamp']) * map_well + grich +
                  sel(NS_seis, index_cols=['timestamp']) * NS_well +
                  sel(EW_seis, index_cols=['timestamp']) * EW_well
                  + injection_plot + polar_plot).cols(2)
        return layout, all_time


    def __panel__(self):
        return self._layout


pan = DailyReport()
app = pn.template.VanillaTemplate(
    title='Newberry Report: {}'.format(UTCDateTime().date), logo='', main=pan).servable()
