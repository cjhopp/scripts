#!/usr/bin/env python

"""
Script to handle pick refinement/removal and relocation of catalog earthquakes.
"""

import os
import locale
import warnings
import uuid
import copy
import pyproj

import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from subprocess import call
from datetime import datetime
from scipy.io import loadmat
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from obspy import UTCDateTime, Catalog
from obspy.core.event import Arrival, QuantityError, ResourceIdentifier, \
    OriginUncertainty, Origin, CreationInfo, OriginQuality, Event, \
    WaveformStreamID, Pick, Comment
from obspy.core import AttribDict
from obspy.geodetics import kilometer2degrees
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from sklearn.cluster import KMeans

# Local import
try:
    from lbnl.boreholes import create_FSB_boreholes
except ImportError:
    print('Something wrong with local import')

fsb_coords = {
    'FS.B31..XNZ': [2579324.461294, 1247590.3884, 488.9207],
    'FS.B32..XNZ': [2579324.127923, 1247564.5962, 458.3487],
    'FS.B42..XNZ': [2579329.86015, 1247595.4255, 488.2790],
    'FS.B43..XNZ': [2579335.9162, 1247574.0129, 455.0401],
    'FS.B551..XNZ': [2579328.42237, 1247584.2444, 500.6040],
    'FS.B585..XNZ': [2579321.582, 1247563.2318, 478.8632],
    'FS.B647..XNZ': [2579334.79024, 1247589.14718, 501.6884],
    'FS.B659..XNZ': [2579337.27833, 1247570.6516, 476.9351],
    'FS.B748..XNZ': [2579340.15296, 1247593.54974, 503.0410],
    'FS.B75..XNZ': [2579351.2349, 1247579.5168, 477.7178],
    # Hydrophones
    'FS.B301..XN1': [2579324.496867, 1247592.9675, 491.9781],
    'FS.B302..XN1': [2579324.474816, 1247591.3557, 490.0691],
    'FS.B303..XN1': [2579324.451961, 1247589.7436, 488.1565],
    'FS.B304..XN1': [2579324.4286640002, 1247588.132, 486.2473],
    'FS.B305..XN1': [2579324.405486, 1247586.5206, 484.3342],
    'FS.B306..XN1': [2579324.384642, 1247584.9097, 482.4244],
    'FS.B307..XN1': [2579324.365244, 1247583.3001, 480.5097],
    'FS.B308..XN1': [2579324.345286, 1247581.6901, 478.59919999999994],
    'FS.B309..XN1': [2579324.324412, 1247580.0796, 476.6852],
    'FS.B310..XN1': [2579324.304438, 1247578.4679, 474.7741],
    'FS.B311..XN1': [2579324.286077, 1247576.8557, 472.8636],
    'FS.B312..XN1': [2579324.267101, 1247575.2429, 470.9534],
    'FS.B313..XN1': [2579324.248199, 1247573.6299, 469.04549999999995],
    'FS.B314..XN1': [2579324.22945, 1247572.0169, 467.1336],
    'FS.B315..XN1': [2579324.20813, 1247570.4047, 465.22499999999997],
    'FS.B316..XN1': [2579324.185952, 1247568.7916, 463.3131],
    'FS.B317..XN1': [2579324.163273, 1247567.1781000001, 461.40569999999997],
    'FS.B318..XN1': [2579324.14131, 1247565.5641, 459.4946],
    'FS.B319..XN1': [2579324.118188, 1247563.9509, 457.58689999999996],
    'FS.B320..XN1': [2579324.091171, 1247562.3397, 455.6735],
    'FS.B321..XN1': [2579324.062039, 1247560.7305, 453.76259999999996],
    'FS.B322..XN1': [2579324.032701, 1247559.1223, 451.8467],
    'FS.B401..XN1': [2579329.17688, 1247597.8316, 492.0198],
    'FS.B402..XN1': [2579329.55651, 1247596.4949999999, 489.93850000000003],
    'FS.B403..XN1': [2579329.93608, 1247595.1581, 487.8633],
    'FS.B404..XN1': [2579330.31607, 1247593.8213, 485.7822],
    'FS.B405..XN1': [2579330.69621, 1247592.4842, 483.7073],
    'FS.B406..XN1': [2579331.07555, 1247591.1471, 481.6262],
    'FS.B407..XN1': [2579331.45355, 1247589.8114, 479.5501],
    'FS.B408..XN1': [2579331.83095, 1247588.4749999999, 477.4683],
    'FS.B409..XN1': [2579332.20912, 1247587.1386, 475.3925],
    'FS.B410..XN1': [2579332.58727, 1247585.8011, 473.3145],
    'FS.B411..XN1': [2579332.96631, 1247584.4626, 471.2373],
    'FS.B412..XN1': [2579333.34493, 1247583.1243, 469.1600],
    'FS.B413..XN1': [2579333.7226299997, 1247581.7854, 467.0797],
    'FS.B414..XN1': [2579334.09847, 1247580.4462, 465.0054],
    'FS.B415..XN1': [2579334.4751399998, 1247579.1062999999, 462.9257],
    'FS.B416..XN1': [2579334.85297, 1247577.7667, 460.8520],
    'FS.B417..XN1': [2579335.23231, 1247576.4266, 458.7729],
    'FS.B418..XN1': [2579335.6125, 1247575.0857, 456.7004],
    'FS.B419..XN1': [2579335.9923, 1247573.7445999999, 454.622],
    'FS.B420..XN1': [2579336.3728, 1247572.4026, 452.5504],
    'FS.B421..XN1': [2579336.7543, 1247571.0595, 450.47360000000003],
    'FS.B422..XN1': [2579337.1362, 1247569.7158, 448.4033],
    # AE sensors
    'FS.B81..XN1': [2579328.4993, 1247576.6989, 487.7793],
    'FS.B82..XN1': [2579328.4046, 1247576.0296, 487.0424],
    'FS.B83..XN1': [2579328.1213, 1247574.0238, 484.8296],
    'FS.B84..XN1': [2579328.0271, 1247573.3558, 484.0914],
    'FS.B85..XN1': [2579327.7444, 1247571.3531, 481.8757],
    'FS.B86..XN1': [2579327.6501, 1247570.6865, 481.1363],
    'FS.B91..XN1': [2579332.2996, 1247583.0459, 476.5476],
    'FS.B92..XN1': [2579332.4017, 1247582.4843, 475.7265],
    'FS.B93..XN1': [2579332.7086, 1247580.8020, 473.2617],
    'FS.B94..XN1': [2579332.8113, 1247580.2412, 472.4401],
    'FS.B95..XN1': [2579333.1193, 1247578.5590, 469.9753],
    'FS.B96..XN1': [2579333.2203, 1247577.9980, 469.1537]
}

source_coords = {
    'S1': [2579328.2022, 1247583.5615, 499.90950000000004],
    'S2': [2579327.21299, 1247580.4926, 496.77040000000005],
    'S3': [2579326.22299, 1247577.4319000002, 493.62370000000004],
    'S4': [2579325.22981, 1247574.3779000002, 490.4714],
    'S5': [2579324.2343, 1247571.3303, 487.31370000000004],
    'S6': [2579323.24033, 1247568.2886, 484.1498],
    'S7': [2579322.2461, 1247565.2523, 480.9809],
    'S8': [2579321.25, 1247562.2222000002, 477.80660000000006],
    'S9': [2579334.87098, 1247588.55173, 500.89000000000004],
    'S10': [2579335.2342499997, 1247585.8715, 497.2936],
    'S11': [2579335.59656, 1247583.1888, 493.69890000000004],
    'S12': [2579335.95866, 1247580.5041, 490.1057],
    'S13': [2579336.3183399998, 1247577.8177999998, 486.51340000000005],
    'S14': [2579336.67692, 1247575.1325, 482.9203],
    'S15': [2579337.0372099997, 1247572.4442999999, 479.32950000000005],
    'S16': [2579337.39851, 1247569.7554, 475.73940000000005],
    'S17': [2579340.51113, 1247593.09901, 502.2253],
    'S18': [2579342.12396, 1247591.0687199999, 498.5474],
    'S19': [2579343.73466, 1247589.0359, 494.87],
    'S20': [2579345.3438600004, 1247587.0004, 491.1935],
    'S21': [2579346.9505000003, 1247584.9625, 487.5171],
    'S22': [2579348.5567, 1247582.9235999999, 483.8411],
    'S23': [2579350.1638, 1247580.8810999999, 480.1675],
    'S24': [2579351.7701000003, 1247578.8351999999, 476.4954]
}

fsb_well_colors = {'B1': 'k', 'B2': 'steelblue', 'B3': 'goldenrod',
                   'B4': 'goldenrod', 'B5': 'goldenrod', 'B6': 'goldenrod',
                   'B7': 'goldenrod', 'B8': 'firebrick', 'B9': 'firebrick',
                   'B10': 'k'}


# origin = [-38.3724, 175.9577]

def my_conversion(x, y, z):
    origin = [-38.3724, 175.9577]
    new_y = origin[0] + ((y * 1000) / 111111)
    new_x = origin[1] + ((x * 1000) /
                         (111111 * np.cos(origin[0] * (np.pi/180))))
    return new_x, new_y, z

def casc_xyz2latlon(x, y):
    """
    Convert from scaled surf xyz (in km) to lat lon
    :param x:
    :param y:
    :return:
    """
    pts = zip(x, y)
    orig_utm = (239200, 5117300)
    utm = pyproj.Proj(init="EPSG:32610")
    pts_utm = [(orig_utm[0] + (pt[0] * 1000), orig_utm[1] + (pt[1] * 1000))
               for pt in pts]
    utmx, utmy = zip(*pts_utm)
    lon, lat = utm(utmx, utmy, inverse=True)
    return (lon, lat)

def surf_xyz2latlon(x, y):
    """
    Convert from scaled surf xyz (in km) to lat lon
    :param x:
    :param y:
    :return:
    """
    # Descale (/10) and convert to meters
    x *= 10
    y *= 10
    pts = zip(x, y)
    orig_utm = (598420.3842806489, 4912272.275375654)
    utm = pyproj.Proj(init="EPSG:26713")
    pts_utm = [(orig_utm[0] + pt[0], orig_utm[1] + pt[1])
               for pt in pts]
    utmx, utmy = zip(*pts_utm)
    lon, lat = utm(utmx, utmy, inverse=True)
    return (lon, lat)

def fsb_xyz2latlon(x, y):
    """
    Convert from scaled surf xyz (in km) to lat lon
    :param x:
    :param y:
    :return:
    """
    # Descale (/10) and convert to meters
    x *= 10
    y *= 10
    pts = zip(x, y)
    orig_utm = (2579255., 1247501.)
    utm = pyproj.Proj(init='EPSG:2056')
    pts_utm = [(orig_utm[0] + pt[0], orig_utm[1] + pt[1])
               for pt in pts]
    utmx, utmy = zip(*pts_utm)
    lon, lat = utm(utmx, utmy, inverse=True)
    return (lon, lat)


def fsb_simple_conversion(x, y):
    """
    Convert from fsb xyz to lat lon
    """
    utm = pyproj.Proj(init='EPSG:2056')
    lon, lat = utm(x, y, inverse=True)
    return (lon, lat)


def relocate(cat, root_name, in_file, pick_uncertainty, location='SURF'):
    """
    Run NonLinLoc relocations on a catalog.

    :type cat: obspy.Catalog
    :param cat: catalog of events with picks to relocate
    :type root_name: str
    :param root_name: String specifying where the nlloc.obs files will be
        written from the catalog
    :type in_file: str
    :param in_file: NLLoc input file
    :type pick_uncertainty: dict
    :param pick_uncertainty: Dictionary mapping uncertainties to sta/chans
    :param location: Which coordinate conversion to use

    :return: same catalog with new origins appended to each event
    """
    for ev in cat:
        if len(ev.picks) < 5:
            print('Fewer than 5 picks for {}. Will not locate.'.format(
                ev.resource_id.id))
            continue
        for pk in ev.picks:
            # Assign arrival time uncertainties if mapping provided
            if (not pk.time_errors.upper_uncertainty
                and not pk.time_errors.uncertainty) or pick_uncertainty:
                sta = pk.waveform_id.station_code[:2]
                chan = pk.waveform_id.channel_code[-1]
                try:
                    pk.time_errors.uncertainty = pick_uncertainty[sta][chan]
                except (TypeError, KeyError) as e:
                    try:
                        pk.time_errors.uncertainty = pick_uncertainty[pk.phase_hint[0]]
                    except (TypeError, KeyError) as e:
                        pk.time_errors.uncertainty = pick_uncertainty
            # For cases of specific P or S phases, just convert to P or S
            if pk.phase_hint not in ['P', 'S']:
                pk.phase_hint = pk.phase_hint[0]
        id_str = str(ev.resource_id).split('/')[-1]
        if len(id_str.split('=')) > 1 and location == 'cascadia':
            # For FDSN pulled events from USGS
            id_str = ev.resource_id.id.split('=')[-2].split('&')[0]
        filename = '{}/obs/{}.nll'.format(root_name, id_str)
        outfile = '{}/loc/{}'.format(root_name, id_str)
        # TODO This clause needs faster file existece check. Do 25-7.
        if os.path.isfile(filename):
            # if len(glob(outfile + '.????????.??????.grid0.loc.hyp')) > 0:
            print('LOC file already written, reading output to catalog')
        else:
            # Here forego obspy write func in favor of obspyck dicts2NLLocPhases
            phases = dicts2NLLocPhases(ev, location)
            with open(filename, 'w') as f:
                f.write(phases)
            # Specify awk command to edit NLLoc .in file
            # Write to unique tmp file (just in_file.bak) so as not to
            # overwrite if multiple instances running.
            cmnd = """awk '$1 == "LOCFILES" {$2 = "%s"; $5 = "%s"}1' %s > %s.bak && mv %s.bak %s""" % (
                filename, outfile, in_file, in_file, in_file, in_file)
            call(cmnd, shell=True)
            # Call to NLLoc
            call('NLLoc %s' % in_file, shell=True)
        # Now reading NLLoc output back into catalog as new origin
        # XXX BE MORE CAREFUL HERE. CANNOT GRAB BOTH SUM AND NON-SUM
        out_w_ext = glob(outfile + '.????????.??????.grid0.loc.hyp')
        try:
            loadNLLocOutput(ev=ev, infile=out_w_ext[0], location=location)
        except (ValueError, IndexError) as ve:
            print(ve)
            continue
        # ev.origins.append(new_o_obj)
        # ev.preferred_origin_id = new_o_obj.resource_id.id
    return cat


def thomsen_full(x, delta, epsilon, vp0):
    """
    Thomsens full anisotropy treatment for P waves
    """
    VpVs = 1.77
    VsVp = 1 / VpVs
    dstar = (1 - VsVp**2) * ((2 * delta) - epsilon)
    Dstar = 0.5 * (1 - VsVp**2) * (np.sqrt((1 + (4 * dstar * np.sin(x)**2 *
                                                 np.cos(x)**2) /
                                            (1 - VsVp**2)**2) +
                                           (4 * (1 - VsVp**2 + epsilon) * epsilon *
                                            np.sin(x)**4 / (1 - VsVp**2)**2)) - 1)
    return np.sqrt(vp0**2 * (1 + (epsilon * np.sin(x)**2) + Dstar))


def thomsen_weak(x, delta, epsilon, vp0):
    """
    Thomsens weak anisotropy for P waves
    """
    return np.sqrt(vp0**2 * (1 + (delta * np.sin(x)**2 * np.cos(x)**2) +
                             (epsilon * np.sin(x)**4)))


def tt_to_cat(tt_file):
    """
    Return a synthetic catalog of picks from Tanners sample CASSM travel times
    """
    cat = Catalog()
    tt_array = loadmat(tt_file)['data']
    src_names = np.repeat(np.arange(24), 44)
    src_names = np.array(['S{:d}'.format(src_names[i] + 1)
                          for i in range(src_names.shape[0])])
    twtwos = np.arange(22) + 1
    b3 = ['B3{:02d}'.format(twtwos[i]) for i in range(22)]
    b4 = ['B4{:02d}'.format(twtwos[i]) for i in range(22)]
    b3.extend(b4)
    rec_names = np.tile(np.array(b3), 24)
    counter = 1
    for i in range(tt_array.shape[0]):
        if i % 44 == 0:
            if i != 0:
                cat.append(ev)
            ev = Event(id=ResourceIdentifier(id=str(counter)))
            counter += 1
            ot = UTCDateTime(1970, 1, 1)
        if counter in [10, 24]:  # Bad sources
            continue
        pt = ot + tt_array[i][0]
        pk = Pick(time=pt, time_errors=QuantityError(1e-5), phase_hint='P',
                  waveform_id=WaveformStreamID(
                      network_code='FS', station_code=rec_names[i],
                      location_code='', channel_code='XN1'))
        if pk.waveform_id.id not in fsb_coords:
            continue
        ev.picks.append(pk)
        if len(ev.comments) == 0:
            extra = AttribDict({'CASSM': {'value': source_coords[src_names[i]],
                                          'namespace': 'smi:local/cassm_loc'}})
            ev.extra = extra
    cat.append(ev)
    return cat


def fit_thomsen(tt_file, aniso_azi=323, aniso_inc=44, plot_fit=True,
                plot_paths=False):
    """
    Fit thomsen weak anisotropy to list of cassm travel times
    """
    aniso_angle = 450 - aniso_azi
    if aniso_angle > 360:
        aniso_angle -= 360
    strike = aniso_azi + 90
    if strike > 360:
        strike -= 360.
    dip = 90 - aniso_inc
    # To radians
    ani_az_rad = np.deg2rad(aniso_angle)
    ani_inc_rad = np.deg2rad(aniso_inc)
    aniso_pole = np.array([np.cos(ani_inc_rad) * np.cos(ani_az_rad),
                           np.cos(ani_inc_rad) * np.sin(ani_az_rad),
                           -np.sin(ani_inc_rad)])
    aniso_pole /= np.linalg.norm(aniso_pole)
    aniso_pole_ten = aniso_pole * 10
    tt_array = loadmat(tt_file)['data']
    twtwos = np.arange(22) + 1
    b3 = ['FS.B3{:02d}..XN1'.format(twtwos[i]) for i in range(22)]
    b4 = ['FS.B4{:02d}..XN1'.format(twtwos[i]) for i in range(22)]
    b3.extend(b4)
    rec_names = np.tile(np.array(b3), 24)
    src_names = np.repeat(np.arange(24), 44)
    src_names = np.array(['S{:d}'.format(src_names[i] + 1)
                          for i in range(src_names.shape[0])])
    rec_locs = [np.array(fsb_coords[nm]) for nm in rec_names]
    src_locs = [np.array(source_coords[nm]) for nm in src_names]
    dists = [np.sqrt(np.sum((rec_locs[i] - src_locs[i])**2))
             for i in range(src_names.shape[0])]
    dists = np.array(dists)
    avg_depths = (tt_array[:, 3] + tt_array[:, 6]) / 2
    src_dep = tt_array[:, 3]
    rec_dep = tt_array[:, 6]
    src_east = tt_array[:, 2]
    Vps = dists / tt_array[:, 0]
    cluster_arr = tt_array[:, 1:4].copy()
    cluster_arr[:, 0] -= cluster_arr[0, 0]
    cluster_arr[:, 1] -= cluster_arr[0, 1]
    cluster_arr[:, 2] -= cluster_arr[0, 2]
    well5 = np.ones(352) * 5
    well6 = well5 + 1
    well7 = well6 + 1
    well_no = np.hstack([well5, well6, well7])
    hyd_well_3 = np.ones(22)
    hyd_well_4 = hyd_well_3 + 1
    hyd_well_no = np.tile(np.hstack([hyd_well_3, hyd_well_4]), 24)
    hydro_no = np.tile(np.arange(44), 24)
    x = tt_array[:, 5] - tt_array[:, 2]
    y = tt_array[:, 4] - tt_array[:, 1]
    z = tt_array[:, 6] - tt_array[:, 3]
    paz = np.rad2deg(np.arctan(x / y))
    paths = np.vstack([x, y, z]).T
    # L2 norm along rows
    pnorm = paths / np.sqrt((paths * paths).sum(axis=1))[:, np.newaxis]
    angles = np.arccos(np.dot(pnorm, aniso_pole))
    angles = np.rad2deg(angles)

    # Now select points to fit
    fig, ax_pick = plt.subplots()
    pts = ax_pick.scatter(angles, Vps)
    selector = SelectFromCollection(ax_pick, pts)
    selection = {}
    def accept(event):
        if event.key == "enter":
            print("Selected points:")
            indices = selector.ind
            selector.disconnect()
            ax_pick.set_title("")
            selection['indices'] = indices

    fig.canvas.mpl_connect("key_press_event", accept)
    plt.show(block=True)
    indices = selection['indices']

    # Take only the selected indices for fitting
    Vps = Vps[indices]
    well_no = well_no[indices]
    hydro_no = hydro_no[indices]
    hyd_well_no = hyd_well_no[indices]
    angles = angles[indices]
    dists = dists[indices]
    avg_depths = avg_depths[indices]
    src_dep = src_dep[indices]
    rec_dep = rec_dep[indices]
    src_east = src_east[indices]
    if plot_paths:
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(projection='3d')
        ax3d.plot(xs=np.array([0, aniso_pole_ten[0]]) + tt_array[0, 2],
                  ys=np.array([0, aniso_pole_ten[1]]) + tt_array[0, 1],
                  zs=np.array([0, aniso_pole_ten[2]]) + tt_array[0, 3],
                  color='b')
        ax3d.scatter(tt_array[indices, 2], tt_array[indices, 1],
                     tt_array[indices, 3], c=well_no)
        # Plot up the well bores
        for w, pts in create_FSB_boreholes().items():
            if w.startswith('B'):
                wx = pts[:, 0]  # + 579300
                wy = pts[:, 1]  # + 247500
                wz = pts[:, 2]  # + 500
                ax3d.scatter(wx[0], wy[0], wz[0], s=10., marker='s',
                             color=fsb_well_colors[w])
                ax3d.plot(wx, wy, wz, color=fsb_well_colors[w])
        for i in range(tt_array.shape[0]):
            if i not in indices:
                continue
            ax3d.plot(xs=[tt_array[i, 2], tt_array[i, 5]],
                      ys=[tt_array[i, 1], tt_array[i, 4]],
                      zs=[tt_array[i, 3], tt_array[i, 6]], color='k',
                      alpha=0.1)
    if plot_fit:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        vars = [('Rec. well no', hyd_well_no), ('Source easting', src_east),
                ('Receiver no', hydro_no), ('Well no', well_no)]
        for i, ax in enumerate(axes.flatten()):
            mpbl = ax.scatter(angles, Vps, alpha=0.2, c=vars[i][1])
            fig.colorbar(mpbl, label=vars[i][0], ax=ax)
    # Fit a curve
    fit_vps = Vps[np.isfinite(Vps)]
    fit_bed_rad = np.deg2rad(angles[np.isfinite(Vps)])
    popt_f, pcov_f = curve_fit(thomsen_full, fit_bed_rad, fit_vps)
    std_f = np.sqrt(np.diag(pcov_f))
    if plot_fit:
        for ax in axes.flatten():
            ax.annotate(
                'Vp0: {:.2f}$\pm${:.2f}\ndelta: {:.2f}$\pm${:.2f}\nepsilon: {:.2f}$\pm${:.2f}'.format(
                popt_f[2], std_f[2], popt_f[0], std_f[0], popt_f[1], std_f[1]),
                xy=(0.6, 0.75), xytext=(0.55, 0.05), xycoords='axes fraction')
            df, ef, vp0f = popt_f
            xline = np.arange(0, np.pi / 2, 0.1)
            yline_f = thomsen_full(xline, df, ef, vp0f)
            ax.plot(np.rad2deg(xline), yline_f)
            ax.set_ylabel('Vp [m/s]')
            ax.set_xlabel('Angle to bedding normal [degrees]')
            ax.set_title('Bedding Strike: {} Dip: {}'.format(strike, dip))
        plt.show()
    return popt_f, pcov_f


def search_aniso(tt_file, min_az, max_az, min_inc, max_inc):
    """
    Grid search orientation of anisotropy for best fitting Thomsen
    """
    azs = np.arange(min_az, max_az, 2)
    incs = np.arange(min_inc, max_inc, 2)
    Azs, Incs = np.meshgrid(azs, incs)
    results = []
    for a, i in zip(Azs.flatten(), Incs.flatten()):
        print('Searching az: {} inc: {}'.format(a, i))
        popt, pcov = fit_thomsen(tt_file, a, i, plot_fit=False)
        results.append(np.sqrt(np.diag(pcov))[0])
    best = np.array(results).argmin()
    print(best)
    popt, pcov = fit_thomsen(tt_file, Azs.flatten()[best], Incs.flatten()[best])
    return


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def search_thomsen(cassm_cat, Vp_range, delta_range, epsilon_range, cores=8):
    """
    Search the Thomsen parameters to get lowest misfit for each CASSM source
    at FSB
    """
    vps, deltas, epsilons = np.meshgrid(
        np.arange(Vp_range[0], Vp_range[1], 100),
        np.arange(delta_range[0], delta_range[1], 0.05),
        np.arange(epsilon_range[0], epsilon_range[1], 0.05))
    variables = np.vstack([vps.flatten(), deltas.flatten(), epsilons.flatten()])
    results = Parallel(n_jobs=cores, verbose=10)(
        delayed(relocate_thomsen)(cassm_cat, fsb_coords, fsb_simple_conversion,
                                  variables[0, i], 0.01, variables[1, i],
                                  variables[2, i], 323., 46.)
        for i in range(variables.shape[1]))
    return variables, results, variables[:, np.argmin(results)]


def relocate_thomsen(catalog, coordinates, conversion, Vp0, damping,
                     delta, epsilon, aniso_azi, aniso_inc):
    """
    Locate an event in a homogeneous background medium from a list of picks
    using travel times.

    This version will only consider P phase picks and ignores all other picks.

    Args:
        catalog: Catalog of events to relocate
        coordinates: Dictionary mapping channel ids to cartesian coordinates.
        conversion: Function to convert xyz to lat/lon
        Vp0: P wave velocity or the minimum velocity in the anisotropic case.
        damping: Damping.
        delta: Thomsen parameter delta describing break from elipticity
        epsilon: Thomsen parameter epsilon
        aniso_azi: Azimuth of pole to the plane of anisotropy
        aniso_inc: Inclination of the pole to plane of anisotropy
        verbose: Print a short summary when an event is found.

    Returns:
        A complete event with a location origin. Will be returned regardless of
        how well the location works so a subsequent QC check is advisable.
    """
    misfits = []
    for event in catalog:
        print('Relocating {}'.format(event.resource_id.id))
        # Filter to only use P-phase picks.
        event_picks = []
        for pick in event.picks:
            pick = copy.deepcopy(pick)
            if pick.phase_hint and pick.phase_hint.lower() != "p":
                continue
            event_picks.append(pick)

        if len(event_picks) < 3:
            print(">= 3 P phase picks are required for an event location.")
            continue
        starttime = min([p.time for p in event_picks])

        # time relative to startime of snippets, in milliseconds.
        t_relative = []
        for pick in event_picks:
            t_relative.append((pick.time - starttime) * 1000)

        npicks = len(t_relative)

        # Assemble sensor coordinate array for the actually used picks.
        sensor_coords = np.zeros((npicks, 3), dtype=np.float64)
        for i, p in enumerate(event_picks):
            sensor_coords[i, :] = coordinates[p.waveform_id.id]

        vp = Vp0 * np.ones([npicks]) / 1000.0

        loc = sensor_coords[t_relative.index(min(t_relative)), :] + 0.1
        t0 = min(t_relative)
        nit = 0
        jacobian = np.zeros([npicks, 4])
        dm = 1.0 * np.ones(4)

        # Actual optimization.
        while nit < 100 and np.linalg.norm(dm) > 0.00001:
            nit = nit + 1
            for i in range(npicks):
                azi = np.arctan2(
                    sensor_coords[i, 0] - loc[0], sensor_coords[i, 1] - loc[1]
                )
                inc = np.arctan2(
                    sensor_coords[i, 2] - loc[2],
                    np.linalg.norm(sensor_coords[i, range(2)] - loc[range(2)]),
                )
                theta = np.arccos(
                    np.cos(inc)
                    * np.cos(azi)
                    * np.cos(aniso_inc)
                    * np.cos(aniso_azi)
                    + np.cos(inc)
                    * np.sin(azi)
                    * np.cos(aniso_inc)
                    * np.sin(aniso_azi)
                    + np.sin(inc) * np.sin(aniso_inc)
                )
                vp[i] = (
                        Vp0
                        / 1000.0
                        * (
                                1.0
                                + delta
                                * np.sin(theta) ** 2
                                * np.cos(theta) ** 2
                                + epsilon * np.sin(theta) ** 4
                        )
                )

            dist = [np.linalg.norm(loc - sensor_coords[i, :]) for i in range(npicks)]
            tcalc = [dist[i] / vp[i] + t0 for i in range(npicks)]

            res = [t_relative[i] - tcalc[i] for i in range(npicks)]
            rms = np.linalg.norm(res) / npicks
            for j in range(3):
                for i in range(npicks):
                    jacobian[i, j] = -(sensor_coords[i, j] - loc[j]) / (vp[i] * dist[i])
            jacobian[:, 3] = np.ones(npicks)

            dm = np.matmul(
                np.matmul(
                    np.linalg.inv(
                        np.matmul(np.transpose(jacobian), jacobian)
                        + pow(damping, 2) * np.eye(4)
                    ),
                    np.transpose(jacobian),
                ),
                res,
            )
            loc = loc + dm[0:3]
            t0 = t0 + dm[3]

        # Finally create the event object with the used picks and arrivals.
        # Try to specify as many details as possible.
        origin_time = starttime + t0 / 1000.0

        # Convert local coordinates to WGS84.
        longitude, latitude = conversion(np.array([loc[0]]),
                                         np.array([loc[1]]))
        depth = loc[-1]
        print(latitude, longitude, depth)
        s = "anisotropic"
        earth_model_id = ResourceIdentifier(
            id=f"earth_model/homogeneous/{s}/velocity={int(round(Vp0))}"
        )
        method_id = "method/p_wave_travel_time/homogeneous_model"

        extra = AttribDict({'ch1903_east': {'value': loc[0],
                                            'namespace': 'smi:local/ch1903'},
                            'ch1903_north': {'value': loc[1],
                                             'namespace': 'smi:local/ch1903'},
                            'ch1903_elev': {'value': depth,  # Extra attribs maintain absolute elevation
                                            'namespace': 'smi:local/ch1903'}})
        try:
            source = np.array(event.extra.CASSM.value)
            location = np.array([loc[0], loc[1], loc[-1]])
            misfit = np.sqrt(np.sum((location - source)**2))
            misfits.append(misfit)
        except AttributeError as e:
            pass
        # Create origin.
        o = Origin(
            resource_id=f"origin/p_wave_travel_time/homogeneous_model/{uuid.uuid4()}",
            time=origin_time,
            longitude=longitude[0],
            latitude=latitude[0],
            depth=depth,
            depth_type="from location",
            time_fixed=False,
            epicenter_fixed=False,
            method_id=method_id,
            earth_model_id=earth_model_id,
        )
        o.extra = extra
        # And fill with arrivals.
        for _i, pick in enumerate(event_picks):
            o.arrivals.append(
                Arrival(
                    resource_id=f"arrival/{_i}/{o.resource_id.id}",
                    pick_id=pick.resource_id,
                    time_residual=res[_i] / 1000,
                    phase="P",
                    earth_model_id=earth_model_id,
                )
            )

        o.time_errors = QuantityError(uncertainty=rms / 1000)

        event.origins.append(o)
        event.preferred_origin_id = o.resource_id
    total_rms = np.sqrt(np.mean(np.array(misfits)**2))
    return total_rms


def dicts2NLLocPhases(ev, location):
    """
    *********
    CJH Stolen from obspyck to use a scaling hack for 6 decimal precision
    *********

    Returns the pick information in NonLinLoc's own phase
    file format as a string. This string can then be written to a file.
    Currently only those fields really needed in location are actually used
    in assembling the phase information string.

    Information on the file formats can be found at:
    http://alomax.free.fr/nlloc/soft6.00/formats.html#_phase_

    Quote:
    NonLinLoc Phase file format (ASCII, NLLoc obsFileType = NLLOC_OBS)

    The NonLinLoc Phase file format is intended to give a comprehensive
    phase time-pick description that is easy to write and read.

    For each event to be located, this file contains one set of records. In
    each set there is one "arrival-time" record for each phase at each seismic
    station. The final record of each set is a blank. As many events as desired can
    be included in one file.

    Each record has a fixed format, with a blank space between fields. A
    field should never be left blank - use a "?" for unused characther fields and a
    zero or invalid numeric value for numeric fields.

    The NonLinLoc Phase file record is identical to the first part of each
    phase record in the NLLoc Hypocenter-Phase file output by the program NLLoc.
    Thus the phase list output by NLLoc can be used without modification as time
    pick observations for other runs of NLLoc.

    NonLinLoc phase record:
    Fields:
    Station name (char*6)
        station name or code
    Instrument (char*4)
        instument identification for the trace for which the time pick
        corresponds (i.e. SP, BRB, VBB)
    Component (char*4)
        component identification for the trace for which the time pick
        corresponds (i.e. Z, N, E, H)
    P phase onset (char*1)
        description of P phase arrival onset; i, e
    Phase descriptor (char*6)
        Phase identification (i.e. P, S, PmP)
    First Motion (char*1)
        first motion direction of P arrival; c, C, u, U = compression;
        d, D = dilatation; +, -, Z, N; . or ? = not readable.
    Date (yyyymmdd) (int*6)
        year (with century), month, day
    Hour/minute (hhmm) (int*4)
        Hour, min
    Seconds (float*7.4)
        seconds of phase arrival
    Err (char*3)
        Error/uncertainty type; GAU
    ErrMag (expFloat*9.2)
        Error/uncertainty magnitude in seconds
    Coda duration (expFloat*9.2)
        coda duration reading
    Amplitude (expFloat*9.2)
        Maxumim peak-to-peak amplitude
    Period (expFloat*9.2)
        Period of amplitude reading
    PriorWt (expFloat*9.2)

    A-priori phase weight Currently can be 0 (do not use reading) or
    1 (use reading). (NLL_FORMAT_VER_2 - WARNING: under development)

    Example:

    GRX    ?    ?    ? P      U 19940217 2216   44.9200 GAU  2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    GRX    ?    ?    ? S      ? 19940217 2216   48.6900 GAU  4.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    CAD    ?    ?    ? P      D 19940217 2216   46.3500 GAU  2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    CAD    ?    ?    ? S      ? 19940217 2216   50.4000 GAU  4.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    BMT    ?    ?    ? P      U 19940217 2216   47.3500 GAU  2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
    """
    nlloc_str = ""

    for pick in ev.picks:
        if pick.waveform_id.station_code == 'NSMTC':
            sta = 'NSMT{}'.format(pick.waveform_id.location_code)
        else:
            sta = pick.waveform_id.station_code.ljust(6)
        inst = "?".ljust(4)
        comp = "?".ljust(4)
        onset = "?"
        try:
            phase = pick.phase_hint.ljust(6)
        except AttributeError:
            phase = 'P'.ljust(6)
        pol = "?"
        t = pick.time
        if location in ['SURF', 'FSB']:
            # CJH Hack to accommodate full microsecond precision...
            t = datetime.fromtimestamp(t.datetime.timestamp() * 100)
        date = t.strftime("%Y%m%d")
        hour_min = t.strftime("%H%M")
        sec = "%7.4f" % (t.second + t.microsecond / 1e6)
        error_type = "GAU"
        error = None
        # XXX check: should we take only half of the complete left-to-right error?!?
        if location == 'cascadia':
            error = pick.time_errors.uncertainty
        elif pick.time_errors.upper_uncertainty and pick.time_errors.lower_uncertainty:
            error = (pick.time_errors.upper_uncertainty + pick.time_errors.lower_uncertainty) * 100
        elif pick.time_errors.uncertainty:
            error = 200 * pick.time_errors.uncertainty
        error = "%9.2e" % error
        coda_dur = "-1.00e+00"
        ampl = "-1.00e+00"
        period = "-1.00e+00"
        fields = [sta, inst, comp, onset, phase, pol, date, hour_min,
                  sec, error_type, error, coda_dur, ampl, period]
        phase_str = " ".join(fields)
        nlloc_str += phase_str + "\n"
    return nlloc_str

def loadNLLocOutput(ev, infile, location):
    lines = open(infile, "rt").readlines()
    if not lines:
        err = "Error: NLLoc output file (%s) does not exist!" % infile
        print(err)
        return
    # goto signature info line
    try:
        line = lines.pop(0)
        while not line.startswith("SIGNATURE"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return

    line = line.rstrip().split('"')[1]
    signature, nlloc_version, date, time = line.rsplit(" ", 3)
    # new NLLoc > 6.0 seems to add prefix 'run:' before date
    if date.startswith('run:'):
        date = date[4:]
    saved_locale = locale.getlocale()
    try:
        locale.setlocale(locale.LC_ALL, ('en_US', 'UTF-8'))
    except:
        creation_time = None
    else:
        creation_time = UTCDateTime().strptime(date + time,
                                               str("%d%b%Y%Hh%Mm%S"))
    finally:
        locale.setlocale(locale.LC_ALL, saved_locale)
    # goto maximum likelihood origin location info line
    try:
        line = lines.pop(0)
        while not line.startswith("HYPOCENTER"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return

    line = line.split()
    x = float(line[2])
    y = float(line[4])
    depth = float(line[6]) * 1000 # depth: negative down!
    if location == 'cascadia':
        lon, lat = casc_xyz2latlon(np.array([x]), np.array([y]))
    # Convert coords
    elif location in ['SURF', 'FSB']:
        # CJH I reported depths at SURF in meters below 130 m so positive is
        # down in this case
        depth = float(line[6])
        print('Doing hypo conversion')
        # Descale first
        depth *= 10
        if location == 'SURF':
            lon, lat = surf_xyz2latlon(np.array([x]), np.array([y]))
        else:
            lon, lat = fsb_xyz2latlon(np.array([x]), np.array([y]))
    else:
        print('Location: {} not supported'.format(location))
        return
    # goto origin time info line
    try:
        line = lines.pop(0)
        while not line.startswith("GEOGRAPHIC  OT"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return
    line = line.split()
    year = int(line[2])
    month = int(line[3])
    day = int(line[4])
    hour = int(line[5])
    minute = int(line[6])
    seconds = float(line[7])
    time = UTCDateTime(year, month, day, hour, minute, seconds)
    if location in ['SURF', 'FSB']:
        # Convert to actual time
        time = UTCDateTime(datetime.fromtimestamp(
            time.datetime.timestamp() / 100.
        ))
    # goto location quality info line
    try:
        line = lines.pop(0)
        while not line.startswith("QUALITY"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return

    line = line.split()
    rms = float(line[8])
    gap = float(line[12])

    # goto location quality info line
    try:
        line = lines.pop(0)
        while not line.startswith("STATISTICS"):
            line = lines.pop(0)
    except:
        err = "Error: No correct location info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return
    line = line.split()
    # # read in the error ellipsoid representation of the location error.
    # # this is given as azimuth/dip/length of axis 1 and 2 and as length
    # # of axis 3.
    # azim1 = float(line[20])
    # dip1 = float(line[22])
    # len1 = float(line[24])
    # azim2 = float(line[26])
    # dip2 = float(line[28])
    # len2 = float(line[30])
    # len3 = float(line[32])
    #
    # # XXX TODO save original nlloc error ellipse?!
    # # errX, errY, errZ = errorEllipsoid2CartesianErrors(azim1, dip1, len1,
    # #                                                   azim2, dip2, len2,
    # #                                                   len3)
    # # NLLOC uses error ellipsoid for 68% confidence interval relating to
    # # one standard deviation in the normal distribution.
    # # We multiply all errors by 2 to approximately get the 95% confidence
    # # level (two standard deviations)...
    # errX *= 2
    # errY *= 2
    # errZ *= 2
    # if location == 'SURF':
    #     # CJH Now descale to correct dimensions
    #     errX /= 100
    #     errY /= 100
    #     errZ /= 100
    # Take covariance approach from obspy
    covariance_xx = float(line[8])
    covariance_yy = float(line[14])
    covariance_zz = float(line[18])
    # determine which model was used:
    # XXX handling of path extremely hackish! to be improved!!
    dirname = os.path.dirname(infile)
    controlfile = os.path.join(dirname, "last.in")
    lines2 = open(controlfile, "rt").readlines()
    line2 = lines2.pop()
    while not line2.startswith("LOCFILES"):
        line2 = lines2.pop()
    line2 = line2.split()
    model = line2[3]
    model = model.split("/")[-1]
    event = ev
    if event.creation_info is None:
        event.creation_info = CreationInfo()
        event.creation_info.creation_time = UTCDateTime()
    o = Origin()
    event.origins = [o]
    # event.set_creation_info_username('cjhopp')
    # version field has 64 char maximum per QuakeML RNG schema
    o.creation_info = CreationInfo(creation_time=creation_time,
                                   version=nlloc_version[:64])
    # assign origin info
    o.method_id = "/".join(["smi:de.erdbeben-in-bayern", "location_method",
                            "nlloc", "7"])
    o.latitude = lat[0]
    o.longitude = lon[0]
    o.depth = depth
    if location in ['SURF', 'FSB']:
        print('Creating origin uncertainty')
        o.longitude = lon[0]
        o.latitude = lat[0]
        print('Assigning depth {}'.format(depth))
        o.depth = depth# * (-1e3)  # meters positive down!
        print('Creating extra AttribDict')
        # Attribute dict for actual hmc coords
        if location == 'FSB':
            extra = AttribDict({
                'ch1903_east': {
                    'value': 2579255. + (x * 10),
                    'namespace': 'smi:local/ch1903'
                },
                'ch1903_north': {
                    'value': 1247501. + (y * 10),
                    'namespace': 'smi:local/ch1903'
                },
                'ch1903_elev': {
                    'value': 547. - depth, # Extra attribs maintain absolute elevation
                    'namespace': 'smi:local/ch1903'
                }
            })
        else:
            extra = AttribDict({
                'hmc_east': {
                    'value': x * 10,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_north': {
                    'value': y * 10,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_elev': {
                    'value': 130 - depth, # Extra attribs maintain absolute elevation
                    'namespace': 'smi:local/hmc'
                }
            })
        o.extra = extra
    o.origin_uncertainty = OriginUncertainty()
    o.quality = OriginQuality()
    ou = o.origin_uncertainty
    oq = o.quality
    # negative values can appear on diagonal of covariance matrix due to a
    # precision problem in NLLoc implementation when location coordinates are
    # large compared to the covariances.
    try:
        o.longitude_errors.uncertainty = kilometer2degrees(np.sqrt(covariance_xx))
    except ValueError:
        if covariance_xx < 0:
            msg = ("Negative value in XX value of covariance matrix, not "
                   "setting longitude error (epicentral uncertainties will "
                   "still be set in origin uncertainty).")
            warnings.warn(msg)
        else:
            raise
    try:
        o.latitude_errors.uncertainty = kilometer2degrees(np.sqrt(covariance_yy))
    except ValueError:
        if covariance_yy < 0:
            msg = ("Negative value in YY value of covariance matrix, not "
                   "setting longitude error (epicentral uncertainties will "
                   "still be set in origin uncertainty).")
            warnings.warn(msg)
        else:
            raise
    o.depth_errors.uncertainty = np.sqrt(covariance_zz) * 1e3  # meters!
    o.depth_errors.confidence_level = 68
    o.depth_type = str("from location")
    # if errY > errX:
    #     ou.azimuth_max_horizontal_uncertainty = 0
    # else:
    #     ou.azimuth_max_horizontal_uncertainty = 90
    # ou.min_horizontal_uncertainty, \
    #         ou.max_horizontal_uncertainty = \
    #         sorted([errX * 1e3, errY * 1e3])
    # ou.preferred_description = "uncertainty ellipse"
    # o.depth_errors.uncertainty = errZ * 1e3
    oq.standard_error = rms #XXX stimmt diese Zuordnung!!!?!
    oq.azimuthal_gap = gap
    # o.depth_type = "from location"
    o.earth_model_id = "%s/earth_model/%s" % ("smi:de.erdbeben-in-bayern",
                                              model)
    o.time = time
    # goto synthetic phases info lines
    try:
        line = lines.pop(0)
        while not line.startswith("PHASE ID"):
            line = lines.pop(0)
    except:
        err = "Error: No correct synthetic phase info found in NLLoc " + \
              "outputfile (%s)!" % infile
        print(err)
        return

    # remove all non phase-info-lines from bottom of list
    try:
        badline = lines.pop()
        while not badline.startswith("END_PHASE"):
            badline = lines.pop()
    except:
        err = "Error: Could not remove unwanted lines at bottom of " + \
              "NLLoc outputfile (%s)!" % infile
        print(err)
        return

    o.quality.used_phase_count = 0
    o.quality.extra = AttribDict()
    o.quality.extra.usedPhaseCountP = {'value': 0,
                                       'namespace': "http://erdbeben-in-bayern.de/xmlns/0.1"}
    o.quality.extra.usedPhaseCountS = {'value': 0,
                                       'namespace': "http://erdbeben-in-bayern.de/xmlns/0.1"}

    # go through all phase info lines
    """
    Order of fields:
    ID Ins Cmp On Pha FM Q Date HrMn Sec Coda Amp Per PriorWt > Err ErrMag
    TTpred Res Weight StaLoc(X Y Z) SDist SAzim RAz RDip RQual Tcorr
    TTerrTcorr

    Fields:
    ID (char*6)
        station name or code
    Ins (char*4)
        instrument identification for the trace for which the time pick corresponds (i.e. SP, BRB, VBB)
    Cmp (char*4)
        component identification for the trace for which the time pick corresponds (i.e. Z, N, E, H)
    On (char*1)
        description of P phase arrival onset; i, e
    Pha (char*6)
        Phase identification (i.e. P, S, PmP)
    FM (char*1)
        first motion direction of P arrival; c, C, u, U = compression; d, D = dilatation; +, -, Z, N; . or ? = not readable.
    Date (yyyymmdd) (int*6)
        year (with century), month, day
    HrMn (hhmm) (int*4)
        Hour, min
    Sec (float*7.4)
        seconds of phase arrival
    Err (char*3)
        Error/uncertainty type; GAU
    ErrMag (expFloat*9.2)
        Error/uncertainty magnitude in seconds
    Coda (expFloat*9.2)
        coda duration reading
    Amp (expFloat*9.2)
        Maxumim peak-to-peak amplitude
    Per (expFloat*9.2)
        Period of amplitude reading
    PriorWt (expFloat*9.2)
        A-priori phase weight
    > (char*1)
        Required separator between first part (observations) and second part (calculated values) of phase record.
    TTpred (float*9.4)
        Predicted travel time
    Res (float*9.4)
        Residual (observed - predicted arrival time)
    Weight (float*9.4)
        Phase weight (covariance matrix weight for LOCMETH GAU_ANALYTIC, posterior weight for LOCMETH EDT EDT_OT_WT)
    StaLoc(X Y Z) (3 * float*9.4)
        Non-GLOBAL: x, y, z location of station in transformed, rectangular coordinates
        GLOBAL: longitude, latitude, z location of station
    SDist (float*9.4)
        Maximum likelihood hypocenter to station epicentral distance in kilometers
    SAzim (float*6.2)
        Maximum likelihood hypocenter to station epicentral azimuth in degrees CW from North
    RAz (float*5.1)
        Ray take-off azimuth at maximum likelihood hypocenter in degrees CW from North
    RDip (float*5.1)
        Ray take-off dip at maximum likelihood hypocenter in degrees upwards from vertical down (0 = down, 180 = up)
    RQual (float*5.1)
        Quality of take-off angle estimation (0 = unreliable, 10 = best)
    Tcorr (float*9.4)
        Time correction (station delay) used for location
    TTerr (expFloat*9.2)
        Traveltime error used for location
    """
    used_stations = set()
    for line in lines:
        line = line.split()
        # check which type of phase
        if line[4] == "P":
            type = "P"
        elif line[4] == "S":
            type = "S"
        else:
            print("Encountered a phase that is not P and not S!! "
                  "This case is not handled yet in reading NLLOC "
                  "output...")
            continue
        # get values from line
        station = line[0]
        epidist = float(line[21])
        azimuth = float(line[23])
        ray_dip = float(line[24])
        # if we do the location on traveltime-grids without angle-grids we
        # do not get ray azimuth/incidence. but we can at least use the
        # station to hypocenter azimuth which is very close (~2 deg) to the
        # ray azimuth
        if azimuth == 0.0 and ray_dip == 0.0:
            azimuth = float(line[22])
            ray_dip = np.nan
        if line[3] == "I":
            onset = "impulsive"
        elif line[3] == "E":
            onset = "emergent"
        else:
            onset = None
        if line[5] == "U":
            polarity = "positive"
        elif line[5] == "D":
            polarity = "negative"
        else:
            polarity = None
        # predicted travel time is zero.
        # seems to happen when no travel time cube is present for a
        # provided station reading. show an error message and skip this
        # arrival.
        if float(line[15]) == 0.0:
            msg = ("Predicted travel time for station '%s' is zero. "
                   "Most likely the travel time cube is missing for "
                   "this station! Skipping arrival for this station.")
            print(msg % station)
            continue
        res = float(line[16])
        weight = float(line[17])

        # assign synthetic phase info
        pick = [p for p in ev.picks if p.waveform_id.station_code == station
                and p.phase_hint == type]
        if station.startswith('NSMT'):
            pick = [p for p in ev.picks
                    if p.waveform_id.station_code == 'NSMTC'
                    and p.waveform_id.location_code == station[-2:]
                    and p.phase_hint == type]
        if len(pick) == 0:
            msg = "This should not happen! Location output was read and a corresponding pick is missing!"
            raise NotImplementedError(msg)
        arrival = Arrival(pick_id=pick[0].resource_id.id)
        o.arrivals.append(arrival)
        # residual is defined as P-Psynth by NLLOC!
        arrival.distance = kilometer2degrees(epidist)
        arrival.phase = type
        arrival.time_residual = res
        if location in ['SURF', 'FSB']:
            arrival.time_residual = res / 1000. # CJH descale time too (why 1000)??
        arrival.azimuth = azimuth
        if not np.isnan(ray_dip):
            arrival.takeoff_angle = ray_dip
        if onset and not pick.onset:
            pick.onset = onset
        if polarity and not pick.polarity:
            pick.polarity = polarity
        # we use weights 0,1,2,3 but NLLoc outputs floats...
        arrival.time_weight = weight
        o.quality.used_phase_count += 1
        if type == "P":
            o.quality.extra.usedPhaseCountP['value'] += 1
        elif type == "S":
            o.quality.extra.usedPhaseCountS['value'] += 1
        else:
            print("Phase '%s' not recognized as P or S. " % type +
                  "Not incrementing P nor S phase count.")
        used_stations.add(station)
    o.used_station_count = len(used_stations)
    try:
        update_origin_azimuthal_gap(ev)
    except IndexError as e:
        print('Invalid resource ids breaking Arrival-->Pick lookup')
    print('Made it through location reading')
    # read NLLOC scatter file
    data = readNLLocScatter(infile.replace('hyp', 'scat'), location)
    print('Read in scatter')
    o.nonlinloc_scatter = data

def getPickForArrival(picks, arrival):
    """
    searches list of picks for a pick that matches the arrivals pick_id
    and returns it (empty Pick object otherwise).
    """
    pick = None
    for p in picks:
        if arrival.pick_id == p.resource_id:
            pick = p
            break
    return pick

def update_origin_azimuthal_gap(ev):
    origin = ev.origins[0]
    arrivals = origin.arrivals
    picks = ev.picks
    azims = {}
    for a in arrivals:
        p = getPickForArrival(picks, a)
        if p is None:
            msg = ("Could not find pick for arrival. Aborting calculation "
                   "of azimuthal gap.")
            print(msg)
            return
        netsta = ".".join([p.waveform_id.network_code, p.waveform_id.station_code])
        azim = a.azimuth
        if azim is None:
            msg = ("Arrival's azimuth is 'None'. "
                   "Calculated azimuthal gap might be wrong")
            print(msg)
        else:
            azims.setdefault(netsta, []).append(azim)
    azim_list = []
    for netsta in azims:
        tmp_list = azims.get(netsta, [])
        if not tmp_list:
            msg = ("No azimuth information for station %s. "
                   "Aborting calculation of azimuthal gap.")
            print(msg)
            return
        azim_list.append((np.median(tmp_list), netsta))
    azim_list = sorted(azim_list)
    azims = np.array([azim for azim, netsta in azim_list])
    azims.sort()
    # calculate azimuthal gap
    gaps = azims - np.roll(azims, 1)
    gaps[0] += 360.0
    gap = gaps.max()
    i_ = gaps.argmax()
    if origin.quality is None:
        origin.quality = OriginQuality()
    origin.quality.azimuthal_gap = gap
    # calculate secondary azimuthal gap
    gaps = azims - np.roll(azims, 2)
    gaps[0] += 360.0
    gaps[1] += 360.0
    gap = gaps.max()
    origin.quality.secondary_azimuthal_gap = gap

def getPick(event, network=None, station=None, phase_hint=None,
            waveform_id=None, seed_string=None):
    """
    returns first matching pick, does NOT ensure there is only one!
    if setdefault is True then if no pick is found an empty one is returned and inserted into self.picks.
    """
    picks = event.picks
    for p in picks:
        if network is not None and network != p.waveform_id.network_code:
            continue
        if station is not None and station != p.waveform_id.station_code:
            continue
        if phase_hint is not None and phase_hint != p.phase_hint:
            continue
        if waveform_id is not None and waveform_id != p.waveform_id:
            continue
        if seed_string is not None and seed_string != p.waveform_id.get_seed_string():
            continue
    return p

def readNLLocScatter(scat_filename, location):
    """
    ****
    Stolen from obspyck
    ****

    This function reads location and values of pdf scatter samples from the
    specified NLLoc *.scat binary file (type "<f4", 4 header values, then 4
    floats per sample: x, y, z, pdf value) and converts X/Y Gauss-Krueger
    coordinates (zone 4, central meridian 12 deg) to Longitude/Latitude in
    WGS84 reference ellipsoid.
    Messages on stderr are written to specified GUI textview.
    Returns an array of xy pairs.
    """
    # read data, omit the first 4 values (header information) and reshape
    print('Reading scatter')
    data = np.fromfile(scat_filename, dtype="<f4").astype("float")[4:]
    # Explicit floor divide or float--> integer error
    print('Reshaping')
    data = data.reshape((data.shape[0] // 4, 4)).swapaxes(0, 1)
    # data[0], data[1] = gk2lonlat(data[0], data[1])
    print('Converting scatter coords')
    if location == 'SURF':
        data[0], data[1] = surf_xyz2latlon(data[0], data[1])
        data[2] *= 10
        data[2] = 130 - data[2]
    elif location == 'FSB': # go straight to ch1903 for this
        print(data.shape)
        data[0] *= 10
        data[0] = 2579255. + data[0]
        data[1] *= 10
        data[1] = 1247501. + data[1]
        data[2] *= 10
        data[2] = 547 - data[2]
    # Descale depth too and convert to m (* 100 / 1000 = * 10)
    return data.T


def errorEllipsoid2CartesianErrors(azimuth1, dip1, len1, azimuth2, dip2, len2,
                                   len3):
    """
    This method converts the location error of NLLoc given as the 3D error
    ellipsoid (two azimuths, two dips and three axis lengths) to a cartesian
    representation.
    We calculate the cartesian representation of each of the ellipsoids three
    eigenvectors and use the maximum of these vectors components on every axis.
    """
    z = len1 * np.sin(np.radians(dip1))
    xy = len1 * np.cos(np.radians(dip1))
    x = xy * np.sin(np.radians(azimuth1))
    y = xy * np.cos(np.radians(azimuth1))
    v1 = np.array([x, y, z])

    z = len2 * np.sin(np.radians(dip2))
    xy = len2 * np.cos(np.radians(dip2))
    x = xy * np.sin(np.radians(azimuth2))
    y = xy * np.cos(np.radians(azimuth2))
    v2 = np.array([x, y, z])

    v3 = np.cross(v1, v2)
    v3 /= np.sqrt(np.dot(v3, v3))
    v3 *= len3

    v1 = np.abs(v1)
    v2 = np.abs(v2)
    v3 = np.abs(v3)

    error_x = max([v1[0], v2[0], v3[0]])
    error_y = max([v1[1], v2[1], v3[1]])
    error_z = max([v1[2], v2[2], v3[2]])

    return (error_x, error_y, error_z)


def dd_time2EQ(catalog, nlloc_root, in_file):
    """
    Takes a catalog with hypoDD-defined origins and populates the arrivals
    attribute for that origin using specified NLLoc Grid files through
    time2EQ

    :param catalog: Catalog containing events which we need Arrivals for
    :param nlloc_root: Root directory for file IO
    :param in_file: NLLoc/Time2EQ run file. User is responsible for defining
        the path to grid files in this control file. This file will be modified
        in-place as this function runs.
    :return:
    """
    # Temp ctrl file overwritten each iteration
    new_ctrl = '{}.new'.format(in_file)
    for ev in catalog:
        eid = ev.resource_id.id.split('/')[-1]
        o = ev.preferred_origin()
        if not o or not o.method_id:
            print('Preferred origin not DD: {}'.format(eid))
            continue
        if len(o.arrivals) > 0:
            print('DD origin has some Arrivals. '
                  + 'Removing and adding again.')
            o.arrivals = []
        print('Raytracing for: {}'.format(eid))
        obs_file = '{}/obs/{}'.format(nlloc_root, eid)
        new_obs = '{}.obs'.format(obs_file) # Only real picks in this one
        print(new_obs)
        loc_file = '{}/loc/{}'.format(nlloc_root, eid)
        out_file_hyp = glob(
            '{}.????????.??????.grid0.loc.hyp'.format(loc_file))
        # Edit the ctrl file for both Time2EQ and NLLoc statements
        if len(out_file_hyp) == 0:
            with open(in_file, 'r') as f, open(new_ctrl, 'w') as fo:
                for line in f:
                    # Time2EQ
                    if line.startswith('EQFILES'):
                        line = line.split()
                        line = '{} {} {}\n'.format(line[0], line[1], obs_file)
                    elif line.startswith("EQSRCE"):
                        line = "EQSRCE {} LATLON {} {} {} 0.0\n".format(
                            eid, o.latitude, o.longitude, o.depth / 1000.)
                    # NLLoc
                    elif line.startswith('LOCFILES'):
                        ln = line.split()
                        line = ' '.join([ln[0], new_obs, ln[2],
                                         ln[3], loc_file])
                    fo.write(line)
            call(["Time2EQ", new_ctrl])
            # Edit obs_file to have just the Time2EQ phases for which we
            # have picks!
            # Make list of sta.phase
            sta_phz = {'{}.{}'.format(pk.waveform_id.station_code,
                                      pk.phase_hint): pk
                       for pk in ev.picks}
            # Also will add the polarities in here to eliminate separate func
            with open(obs_file, 'r') as of, open(new_obs, 'w') as nof:
                for line in of:
                    ln = line.split()
                    # Write the first line
                    if ln[0] == '#':
                        nof.write(' '.join(ln) + '\n')
                        continue
                    staph = '{}.{}'.format(ln[0], ln[4])
                    # Now only write phases we picked to the obs file
                    if staph in sta_phz:
                        if sta_phz[staph].polarity == 'positive':
                            ln[5] = 'U'
                        elif sta_phz[staph].polarity == 'negative':
                            ln[5] = 'D'
                        nof.write(' '.join(ln) + '\n')
            call(["NLLoc", new_ctrl])
            out_file_hyp = glob(
                '{}.????????.??????.grid0.loc.hyp'.format(loc_file))
            if len(out_file_hyp) == 0:
                print('No observations produced. Skip.')
                continue
        pk_stas = [pk.waveform_id.station_code for pk in ev.picks]
        # Instead of using the obspy 'read_nlloc_hyp' method, like above,
        # we'll just take the toa and dip from the phases. There was some
        # weirdness with bad microseconds being read into datetime objs
        # possibly linked to origins at 1900?
        try:
            with open(out_file_hyp[0], 'r') as f:
                for i, line in enumerate(f):
                    if (i > 15 and not line.startswith('END')
                        and not line.startswith('\n')):
                        ln = line.split()
                        pha = ln[4]
                        sta = ln[0]
                        dist = kilometer2degrees(float(ln[-6]))
                        if sta not in pk_stas:
                            continue
                        toa = ln[-3]
                        to_az = ln[-4]
                        try:
                            pk = [pk for pk in ev.picks
                                  if pk.waveform_id.station_code == sta][0]
                        except IndexError:
                            continue
                        ev.preferred_origin().arrivals.append(
                            Arrival(phase=pha, pick_id=pk.resource_id.id,
                                    takeoff_angle=toa, azimuth=to_az,
                                    distance=dist))
        except:
            print('Issue opening file. Event may not have been located')
            continue
    return

def write_xyz(cat, outfile):
    import csv
    with open(outfile, 'wb') as f:
        writer = csv.writer(f, delimiter=' ')
        for ev in cat:
            if ev.preferred_origin():
                writer.writerow([ev.preferred_origin().latitude,
                                 ev.preferred_origin().longitude,
                                 ev.preferred_origin().depth / 1000])

############## GrowClust Functions ############################################

def hypoDD_to_GrowClust(in_dir, out_dir):
    """
    Helper to take input files from hypoDD and convert them for use with
    GrowClust

    :param in_dir: Path to the HypoDD input directory
    :param out_dir: Path to the GrowClust input directory
    :return:
    """
    # First, convert phase.dat to evlist.txt
    with open('{}/phase.dat'.format(in_dir), 'r') as in_f:
        with open('{}/evlist.txt'.format(out_dir), 'w') as out_f:
            for ln in in_f:
                if ln.startswith('#'):
                    out_f.write('{}\n'.format(' '.join(ln.split()[1:])))
    # Now remove occurrences of network string from dt.cc and write to
    # xcordata.txt (use sed via system call as much faster)
    sed_str = "sed 's/NZ.//g' {}/dt.cc > {}/xcordata.txt".format(in_dir,
                                                                 out_dir)
    call(sed_str, shell=True)
    return

def GrowClust_to_Catalog(hypoDD_cat, out_dir):
    """
    Take the original catalog used in generating dt's with HypoDDpy and read
    the output of GrowClust into the appropriate events as new origins.

    This is probably going to borrow heavily from hypoDDpy...
    :param hypoDD_cat: Same catalog used in hypoDDpy to generate dt's
    :param out_dir: GrowClust output directory
    :return:
    """
    # Catalog is sorted by time in hypoDDpy before event map is generated
    hypoDD_cat.events.sort(key=lambda x: x.preferred_origin().time)
    new_o_map = {}
    with open('{}/out.growclust_cat'.format(out_dir), 'r') as f:
        for ln in f:
            ln.strip()
            line = ln.split()
            # First determine if it was relocated
            # Default is line[19] == -1 for no, but also should beware of
            # unceratintites of 0.000. Deal with these later?
            eid = int(line[6]) # Event id before clustering
            if line[13] == '1' and line[19] == '-1.000':
                print('Event {} not relocated, keep original location'.format(
                    eid
                ))
                continue
            re_lat = float(line[7])
            re_lon = float(line[8])
            re_dep = float(line[9]) * 1000 # meters bsl
            x_uncert = float(line[19]) * 1000 # in m
            z_uncert = float(line[20]) * 1000 # in m
            t_uncert = float(line[21])
            o_uncert = OriginUncertainty(horizontal_uncertainty=x_uncert)
            t_uncert = QuantityError(uncertainty=t_uncert)
            d_uncert = QuantityError(uncertainty=z_uncert)
            sec = int(line[5].split('.')[0])
            microsec = int(line[5].split('.')[1]) * 1000
            method_id = ResourceIdentifier(id='GrowClust')
            re_time = UTCDateTime(year=int(line[0]), month=int(line[1]),
                                  day=int(line[2]), hour=int(line[3]),
                                  minute=int(line[4]), second=sec,
                                  microsecond=microsec)
            new_o_map[eid] = Origin(time=re_time, latitude=re_lat,
                                    longitude=re_lon, depth=re_dep,
                                    time_errors=t_uncert,
                                    depth_errors=d_uncert,
                                    origin_uncertainty=o_uncert,
                                    method_id=method_id)
    for i, ev in enumerate(hypoDD_cat):
        id = i + 1 # Python indexing
        if id in new_o_map:
            ev.origins.append(new_o_map[id])
            ev.preferred_origin_id = new_o_map[id].resource_id.id
    return hypoDD_cat