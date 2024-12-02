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
from datetime import datetime
from matplotlib import gridspec
from holoviews.selection import link_selections


wellpath = '/media/chopp/Data1/chet-meq/newberry/boreholes/55-29/GDR_submission/Deviation_corrected.csv'

hv.extension('plotly', 'matplotlib')

def get_data():
    cli = Client('http://131.243.224.19:8085')
    newb_catalog = cli.get_events(starttime=UTCDateTime(2012, 1, 1), latitude=43.726, longitude=-121.316,
                                  maxradius=0.22)
    return newb_catalog


def ecdf_transform(data):
    return len(data) - data.rank(method="first")


def calc_max_curv(magnitudes, bin_size=0.1, plotvar=False):
    """
    Stolen from EQcorrscan to avoid dependency
    """
    min_bin, max_bin = int(min(magnitudes)), int(max(magnitudes) + 1)
    bins = np.arange(min_bin, max_bin + bin_size, bin_size)
    df, bins = np.histogram(magnitudes, bins)
    grad = (df[1:] - df[0:-1]) / bin_size
    # Need to find the second order derivative
    curvature = (grad[1:] - grad[0:-1]) / bin_size
    max_curv = bins[np.argmax(np.abs(curvature))] + bin_size
    if plotvar:
        fig, ax = plt.subplots()
        ax.scatter(bins[:-1] + bin_size / 2, df, color="k",
                   label="Magnitudes")
        ax.axvline(x=max_curv, color="red", label="Maximum curvature")
        ax1 = ax.twinx()
        ax1.plot(bins[:-1] + bin_size / 2, np.cumsum(df[::-1])[::-1],
                 color="k", label="Cumulative distribution")
        ax1.scatter(bins[1:-1], grad, color="r", label="Gradient")
        ax2 = ax.twinx()
        ax2.scatter(bins[1:-2] + bin_size, curvature, color="blue",
                    label="Curvature")
        # Code borrowed from https://matplotlib.org/3.1.1/gallery/ticks_and_
        # spines/multiple_yaxis_with_spines.html#sphx-glr-gallery-ticks-and-
        # spines-multiple-yaxis-with-spines-py
        ax2.spines["right"].set_position(("axes", 1.2))
        ax2.set_frame_on(True)
        ax2.patch.set_visible(False)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.spines["right"].set_visible(True)

        ax.set_ylabel("N earthquakes in bin")
        ax.set_xlabel("Magnitude")
        ax1.set_ylabel("Cumulative events and gradient")
        ax2.set_ylabel("Curvature")
        fig.legend()
        fig.show()
    return float(max_curv)


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
    print(dataset)
    Mc = calc_max_curv(dataset['magnitude'].values)
    points = dataset.loc[(dataset['magnitude'] > Mc) & (dataset['cumulative number'] > 0.)]
    print(points)
    b, a = np.polyfit(points['magnitude'], np.log10(points['cumulative number']), 1)
    dataset['b label'] = ['b value: {}'.format(-b) for i in range(len(dataset))]
    dataset['bfit'] = 10**(np.log10(len(points)) + b * points['magnitude'])
    east, north = utm(dataset['longitude'], dataset['latitude'])
    east -= well[0][0]
    north -= well[0][1]
    elevation = params[:, 2] * -1.
    dataset['east'] = east
    dataset['north'] = north
    dataset['elevation'] = elevation
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


def PGA_graph():
    return


def gr_plot(dataset):
    Mc = calc_max_curv(dataset['magnitude'])
    gr = hv.Scatter(dataset, 'magnitude', vdims=['cumulative number']).opts(
        responsive=True, logy=True, bgcolor='whitesmoke', color='darkgray')
    fit = hv.Curve(dataset.iloc[dataset['magnitude'] > Mc].sort('magnitude'),
                   'magnitude', vdims=['bfit']).opts(
        color='red', logy=True, title='b value: {:.2f}'.format(float(dataset['b label'][0].split()[-1])))
    return gr * fit


def mag_time_plot(dataset):
    tickvals = np.linspace(dataset.range('timestamp')[0], dataset.range('timestamp')[1], 10)
    ticktext = [UTCDateTime(t).strftime('%d %b %Y: %H:%M')
                for t in tickvals]
    xticks = [(tv, ticktext[i]) for i, tv in enumerate(tickvals)]
    plot = hv.Scatter(dataset, 'timestamp', vdims=['magnitude']).opts(
        responsive=True, bgcolor='whitesmoke', color='timestamp', cmap='Cividis', xticks=xticks)
    return plot


def seismicity_3d(dataset):
    section_color = ['black', 'red', 'black', 'red']
    section_styles = ['solid', 'dot', 'solid', 'dot']
    tickvals = np.linspace(dataset.range('timestamp')[0], dataset.range('timestamp')[1], 10)
    ticktext = [UTCDateTime(t).strftime('%d %b %Y: %H:%M')
                for t in tickvals]
    # Map view, two cross-sections, and 3D
    mapview = hv.Scatter(
        dataset, kdims=['east', 'north'], vdims=['timestamp', 'magnitude']).opts(
        marker='circle',
        color='timestamp',
        size=dataset['marker size'],
        colorbar=True,
        colorbar_opts=dict(ticktext=ticktext, tickvals=tickvals),
        cmap='Cividis',
        xlim=(-2000, 2000), ylim=(-2000, 2000),
        responsive=True,
        bgcolor='whitesmoke',
    )
    NS = hv.Scatter(
        dataset, kdims=['north', 'elevation'], vdims=['timestamp', 'magnitude']).opts(
        marker='circle',
        color='timestamp',
        size=dataset['marker size'],
        colorbar=True,
        colorbar_opts=dict(ticktext=ticktext, tickvals=tickvals),
        cmap='Cividis',
        xlim=(-2000, 2000), ylim=(-2000, 2000),
        responsive=True,
        bgcolor='whitesmoke',
    )
    EW = hv.Scatter(
        dataset, kdims=['east', 'elevation'], vdims=['timestamp', 'magnitude']).opts(
        marker='circle',
        color='timestamp',
        size=dataset['marker size'],
        colorbar=True,
        colorbar_opts=dict(ticktext=ticktext, tickvals=tickvals),
        cmap='Cividis',
        xlim=(-2000, 2000), ylim=(-2000, 2000),
        responsive=True,
        bgcolor='whitesmoke',
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
        bgcolor='whitesmoke',
        # colorbar_opts=dict(
        #     title='Timestamp', ticktext=ticktext, tickvals=tickvals)
    )
    paths = get_wellpath(wellpath)
    well_map = hv.Curve(paths, 'east', 'north').opts(
        # color=section_color, dash=section_styles, line_width=3.,
        xlim=(-2000, 2000), ylim=(-2000, 2000))
    well_NS = hv.Curve(paths, 'north', 'elevation').opts(
        # color=section_color, dash=section_styles, line_width=3.,
        xlim=(-2000, 2000), ylim=(-2200, 1800))
    well_EW = hv.Curve(paths, 'east', 'elevation').opts(
        # color=section_color, dash=section_styles, line_width=3.,
        xlim=(-2000, 2000), ylim=(-2200, 1800))
    well_3d = hv.Path3D(paths, kdims=['east', 'north', 'elevation']).opts(
        # color=section_color, dash=section_styles, line_width=3.,
        xlim=(-2000, 2000), ylim=(-2000, 2000), zlim=(-2200, 1800),
        width=800, height=1200, responsive=True)
    return well_map * mapview, well_NS * NS, well_EW * EW, well_3d * seis3D


class DailyReport(pn.viewable.Viewer):

    def __init__(self, **params):
        super().__init__(**params)
        self.catalog = get_data()
        self.dataset = get_seismic_events(self.catalog)
        self._layout = pn.Column(self._link_plots(),
            align='center'
        )

    def _link_plots(self):
        # Gridspec
        gspec = pn.GridSpec(max_height=2000)
        sel = link_selections.instance(unselected_alpha=0.5)
        map, NS, EW, threeD = seismicity_3d(self.dataset)
        grich = gr_plot(self.dataset)
        mag_time = mag_time_plot(self.dataset)
        sel(map + NS + EW + threeD + grich, index_cols=['timestamp']).cols(2)
        gspec[:4, :2] = map
        gspec[:4, 2:4] = grich
        gspec[4:8, :2] = NS
        gspec[4:8, 2:4] = EW
        gspec[8:12, :4] = threeD
        gspec[12:14, :] = mag_time
        return gspec

    def __panel__(self):
        return self._layout


pan = DailyReport()
app = pn.template.VanillaTemplate(
    title='Newberry Report: {}'.format(UTCDateTime().date), logo='', main=pan).servable()
