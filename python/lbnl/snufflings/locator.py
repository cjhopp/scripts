#!/usr/bin/python

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyrocko.snuffling import Snuffling, Param
from pyrocko.pile_viewer import PhaseMarker, EventMarker
from glob import glob

fsb_well_colors = {'B1': 'k', 'B2': 'steelblue', 'B3': 'goldenrod',
                   'B4': 'goldenrod', 'B5': 'goldenrod', 'B6': 'goldenrod',
                   'B7': 'goldenrod', 'B8': 'firebrick', 'B9': 'firebrick',
                   'B10': 'k'}


class Locate(Snuffling):
    '''Steal dug-seis location algorithm for manual picks'''
    def __init__(self):
        Snuffling.__init__(self)

    def setup(self):
        '''Customization of the snuffling.'''
        self.set_name('Locate active event')
        self.add_parameter(Param('P velocity', 'vp', 3000, 0, 10000))
        self.add_parameter(Param('S velocity', 'vs', 1875, 0, 6250))
        self.add_parameter(Param('Damping', 'damp', 0.01, 0.001, 0.1))


    def call(self):
        '''Main work routine of the snuffling.'''
        # gather events to be processed
        viewer = self.get_viewer()
        event = viewer.get_active_event()
        if event is None:
            self.fail('No event selected')
        stations = {s.station: np.array([s.lon,
                                         s.lat,
                                         s.elevation])
                    for s in viewer.stations.values()}
        if not stations:
            self.fail('No station information')
        # Grab picks
        tobs = []
        trig_ch = []
        phase = []
        ev_t = [np.mean([m.tmax, m.tmin]) for m in viewer.markers
                if isinstance(m, EventMarker)][0]
        for m2 in viewer.markers:
            if isinstance(m2, PhaseMarker) and m2.kind == 0:
                if m2.get_event() == event:
                    net, sta, _, _ = m2.one_nslc()
                    trig_ch.append(sta)
                    tobs.append((np.mean([m2.tmax + m2.tmin]) - ev_t) * 1000)
                    phase.append(m2.get_phasename())
        npicks = len(tobs)
        sensor_coords = np.array([stations[tch] for tch in trig_ch])
        # Allow for S picks
        vels = {'P': self.vp, 'S': self.vs}
        v = np.array([vels[phs] / 1000 for phs in phase])
        # Starting location is earliest pick
        loc = sensor_coords[tobs.index(min(tobs)), :] + 0.1
        t0 = min(tobs)
        nit = 0
        jacobian = np.zeros([len(tobs), 4])
        dm = 1. * np.ones(4)
        while nit < 100 and np.linalg.norm(dm) > 0.00001:
            nit = nit + 1
            dist = [np.linalg.norm(loc - sensor_coords[i, :])
                    for i in range(npicks)]
            tcalc = [dist[i] / v[i] + t0 for i in range(npicks)]
            res = [tobs[i] - tcalc[i] for i in range(npicks)]
            rms = np.linalg.norm(res) / npicks
            for j in range(3):
                for i in range(npicks):
                    jacobian[i, j] = -((sensor_coords[i, j] - loc[j]) /
                                       (v[i] * dist[i]))
            jacobian[:, 3] = np.ones(npicks)
            dm = np.matmul(
                np.matmul(
                    np.linalg.inv(np.matmul(np.transpose(jacobian),
                                            jacobian) +
                                  pow(self.damp, 2) * np.eye(4)),
                    np.transpose(jacobian)), res)
            loc = loc + dm[0:3]
            t0 = t0 + dm[3]
        x = loc[0]
        y = loc[1]
        z = loc[2]
        t0 = t0 / 1000.
        print(x, y, z, t0)
        # Plot with the wellbores
        well_dict = create_FSB_boreholes()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, s=10., color='r', alpha=0.7)
        for w, pts in well_dict.items():
            if w.startswith('B'):
                wx = pts[:, 0]
                wy = pts[:, 1]
                wz = pts[:, 2]
                ax.scatter(wx[0], wy[0], wz[0], s=10., marker='s',
                           color=fsb_well_colors[w])
                ax.plot(wx, wy, wz, color=fsb_well_colors[w])
        ax.set_xlim([2579310, 2579355])
        ax.set_ylim([1247555, 1247600])
        ax.set_zlim([485, 530])
        plt.show()


def __snufflings__():
    '''Returns a list of snufflings to be exported by this module.'''

    return [ Locate() ]


## Utility funcs

def create_FSB_boreholes(gocad_dir='/media/chet/data/chet-FS-B/Mont_Terri_model/',
                         asbuilt_dir='/media/chet/data/chet-FS-B/wells/'):
    """
    Return dictionary of FSB well coordinates

    :param asbuilt_dir: Directory holding the gamma logs for each well
    """
    if not os.path.isdir(asbuilt_dir):
        asbuilt_dir = '/media/chet/hdd/seismic/chet_FS-B/wells/'
    if not os.path.isdir(asbuilt_dir):
        asbuilt_dir = 'data/chet-FS-B/wells'
    excel_asbuilts = glob('{}/**/*Gamma_Deviation.xlsx'.format(asbuilt_dir))
    well_dict = {}
    if not os.path.isdir(gocad_dir):
        gocad_dir = '/media/chet/hdd/seismic/chet_FS-B/Mont_Terri_model'
    if not os.path.isdir(gocad_dir):
        gocad_dir = 'data/chet-FS-B/Mont_Terri_model'
    gocad_asbuilts =  glob('{}/*.wl'.format(gocad_dir))
    for gocad_f in gocad_asbuilts:
        name = str(gocad_f).split('-')[-1].split('.')[0]
        well_dict[name] = []
        # Multispace delimiter
        top = pd.read_csv(gocad_f, header=None, skiprows=np.arange(13),
                          delimiter='\s+', index_col=False, engine='python',
                          nrows=1)
        rows = pd.read_csv(gocad_f, header=None, skiprows=np.arange(14),
                           delimiter='\s+', index_col=False, engine='python',
                           skipfooter=1)
        lab, x_top, y_top, z_top = top.values.flatten()
        well_dict[name] = np.stack(((x_top + rows.iloc[:, 3]).values,
                                    (y_top + rows.iloc[:, 4]).values,
                                    rows.iloc[:, 2].values,
                                    rows.iloc[:, 1].values)).T
        if well_dict[name].shape[0] < 1000:  # Read in gamma instead
            # If so, make a more highly-sampled interpolation
            x, y, z, d = zip(*well_dict[name])
            td = d[-1]
            if td == 'Top':
                td = float(d[-3])
            well_dict[name] = np.stack((np.linspace(x_top, x[-1], 1000),
                                        np.linspace(y_top, y[-1], 1000),
                                        np.linspace(z_top, z[-1], 1000),
                                        np.linspace(0, td, 1000))).T
    return well_dict
