#!/usr/bin/python
"""
Functions for working with instrumentation for EGS Collab, FS-B, etc...

IMPORTANT
***********************************************
For SURF, the arbitrary zero depth point is elev = 130 m
***********************************************

"""
import os

import numpy as np
import pandas as pd
import math as M
import matplotlib.pyplot as plt

from glob import glob
from obspy import read_inventory
from obspy.core.util import AttribDict
from obspy.core.inventory import Inventory, Network, Station, Channel, Response
from obspy.core.inventory import ResponseListResponseStage
from obspy.core.inventory.response import ResponseListElement, InstrumentPolynomial
from obspy.core.inventory.response import InstrumentSensitivity
from obspy.core.inventory.util import Latitude, Longitude, Equipment

from lbnl.coordinates import SURF_converter, FSB_converter
from lbnl.boreholes import create_FSB_boreholes


fsb_accelerometers = ['B31', 'B34', 'B42', 'B43', 'B551', 'B585', 'B647',
                      'B659', 'B748', 'B75']

nsmtc_orientation = {'NSMTC.G1': {'E': 90., 'N': 0., 'Z': 0},
                     'NSMTC.B1': {'1': 56.27, '2': 146.27, 'Z': 0},
                     'NSMTC.B2': {'1': 303.99, '2': 33.99, 'Z': 0},
                     'NSMTC.B3': {'1': 84.39, '2': 174.39, 'Z': 0},
                     'NSMTC.G2': {'1': 326.88, '2': 56.88, 'Z': 0}}

resp_labl_map = {'RESP.XX.NS491..BNZ.LowNoise.0_005_1000.60V.2G': 'Silicon Audio ULN Accelerometer',
                 'RESP.XX.NS126..BNZ.Titan.DC_430.20V.0_5G': 'Nanometrics Titan Accelerometer',
                 'RESP.XX.NS380..SLZ.HS1LT.3810.115000.2.76': 'Geospace HS-1-LT Geophone',
                 'RESP.XX.NS391..SHZ.GS11D.10.380.NONE.32': 'Geospace GS-11D Geophone',
                 'RESP.XX.NS539..BHZ.Trillium120Q.120.1500': 'Nanometrics Trillium 120s PH broadband'}

resp_outp_map = {'RESP.XX.NS491..BNZ.LowNoise.0_005_1000.60V.2G': 'ACC',
                 'RESP.XX.NS126..BNZ.Titan.DC_430.20V.0_5G': 'ACC',
                 'RESP.XX.NS380..SLZ.HS1LT.3810.115000.2.76': 'VEL',
                 'RESP.XX.NS391..SHZ.GS11D.10.380.NONE.32': 'VEL',
                 'RESP.XX.NS539..BHZ.Trillium120Q.120.1500': 'VEL'}

resp_color_map = {'Silicon Audio ULN Accelerometer': '#2070b4',
                  'Nanometrics Titan Accelerometer': '#ee854a',
                  'Geospace HS-1-LT Geophone': '#6acc64',
                  'Geospace GS-11D Geophone': '#d65f5f',
                  'Nanometrics Trillium 120s PH broadband': '#ee854a'}

resp_ls_map = {'Silicon Audio ULN Accelerometer': '-',
               'Nanometrics Titan Accelerometer': ':',
               'Geospace HS-1-LT Geophone': '-',
               'Geospace GS-11D Geophone': '-',
               'Nanometrics Trillium 120s PH broadband': '-'}

def plot_resp(resp_dir, min_freq, sampling_rate):
    """
    Plot the RESP curves in a directory

    :param resp_dir: Directory containing RESP files
    :param min_freq: Minimum frequency to plot
    :param sampling_rate: Sampling rate for theoretical data
    """
    resps = glob('{}/RESP*'.format(resp_dir))
    file_order = []
    fig, axes = plt.subplots(nrows=2, figsize=(9, 9))
    for i, resp in enumerate(resps):
        base = os.path.basename(resp)
        file_order.append(base)
        lab = resp_labl_map[base]
        output = resp_outp_map[base]
        read_inventory(resp)[0][0][0].response.plot(
            output=output,
            min_freq=min_freq, sampling_rate=sampling_rate,
            label=lab, axes=fig.axes, unwrap_phase=True, show=False)
    # Edit linestyle after the fact
    inds = []  # Labels are only assigned to amplitude axes, save inds of lines
    ind_labs = []
    for i, ln in enumerate(axes[0].get_lines()):
        label = ln.get_label()
        ind_labs.append(label)
        if label in resp_color_map:
            inds.append(i)
            ln.set_color(resp_color_map[label])
            ln.set_linestyle(resp_ls_map[label])
    ax2_lines = axes[1].get_lines()
    for ind in inds:
        ln = ax2_lines[ind]
        ln.set_color(resp_color_map[ind_labs[ind]])
        ln.set_linestyle(resp_ls_map[ind_labs[ind]])
    axes[0].grid(True)
    axes[1].grid(True)
    axes[1].set_yticks([-2 * np.pi, -(3/2) * np.pi, -np.pi, -np.pi / 2, 0,
                        np.pi / 2, np.pi])
    axes[1].set_yticklabels([r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$\pi$',
                             r'$\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$',
                             r'$\pi$'])
    axes[0].set_ylabel('Amplitude')
    axes[1].set_ylabel('Phase [rad]')
    fig.legend()
    plt.show()
    return


def MMF_calibration_to_response(directory, plot=False):
    """
    Take directory of MMF calibration spreadsheets and convert to Obspy
    inventory object
    """
    inv = Inventory(networks=[Network(code='MMF')])
    lat = Latitude(0.)
    lon = Longitude(0.)
    chan_map = {'Tabellenblatt3': 'X',
                'Tabellenblatt4': 'Y',
                'Tabellenblatt5': 'Z'}
    calibs = glob('{}/*.xls'.format(directory))
    for c in calibs:
        serial = c.split()[-2]
        sta = Station(code=serial[1:], latitude=lat, longitude=lon,
                      elevation=0.)
        # Tables slightly shifted for each channel due to comments
        dict_xyz = pd.read_excel(
            c, sheet_name=['Tabellenblatt3'], header=14,
            usecols=list(np.arange(4, 14)), nrows=37)
        dict_xyz.update(pd.read_excel(
            c, sheet_name=['Tabellenblatt4'], header=14,
            usecols=list(np.arange(5, 15)), nrows=37))
        dict_xyz.update(pd.read_excel(
            c, sheet_name=['Tabellenblatt5'], header=13,
            usecols=list(np.arange(9, 20)), nrows=37))
        # Get array of sensitivities at 80 Hz for X, Y, Z
        sens = pd.read_excel(
            c, sheet_name=['Tabellenblatt2'], header=84,
            usecols=[27], nrows=3)['Tabellenblatt2'].values.squeeze()
        # mV/m/s**2 to V/m/s**2
        sens_dict = {'Tabellenblatt3': float(sens[0].replace(',', '.')) * 1e-3,
                     'Tabellenblatt4': float(sens[1].replace(',', '.')) * 1e-3,
                     'Tabellenblatt5': float(sens[2].replace(',', '.')) * 1e-3}
        # Resp for each channel
        for nm, df in dict_xyz.items():
            # Dummy channel
            # Set samp_rate to 40 kHz so that Nyquist is below max shake freq
            chan = Channel(code='XN{}'.format(chan_map[nm]), location_code='',
                           latitude=lat, longitude=lon, elevation=0., depth=0.,
                           sample_rate=40000.,
                           sensor=Equipment(
                               type='IEPE Accelerometer',
                               description='Piezoelectric accelerometer',
                               manufacturer='MMF',
                               model='KS943B.100',
                               serial_number=serial))
            values = df[['[Hz]', '[m/s²]', '[°]']].values
            response_elements = [ResponseListElement(
                frequency=values[i][0], amplitude=values[i][1],
                phase=values[i][2])
                                 for i in range(values.shape[0])]
            resp_stage = ResponseListResponseStage(
                response_list_elements=response_elements, stage_gain=1,
                stage_gain_frequency=80., input_units='M/S**2',
                output_units='V',
                stage_sequence_number=1
            )
            sensitivity = InstrumentSensitivity(
                value=float(sens_dict[nm]), frequency=80.,
                input_units='M/S**2', output_units='V', frequency_range_start=5,
                frequency_range_end=15850,
                frequency_range_db_variation=3)
            response = Response(instrument_sensitivity=sensitivity,
                                response_stages=[resp_stage])
            chan.response = response
            sta.channels.append(chan)
            # chan.response.plot(min_freq=2.4, sampling_rate=40000.)
        inv[0].stations.append(sta)
    if plot:
        inv.plot_response(min_freq=2.4, plot_degrees=True)
    return inv


def read_fsb_asbuilt(excel_path, gocad_dir, asbuilt_dir):
    """
    Read the as-built excel spreadsheet for FSB and return a dictionary of all
    the stations and sources containing locations

    :param path: Path to excel spreadsheet
    :return:
    """
    sens_dict = {}
    # Read excel spreadsheet of sensor wells and depths
    sensors = pd.read_excel(excel_path, sheet_name=None, skiprows=np.arange(5),
                            usecols=np.arange(1, 10), header=None)
    well_dict = create_FSB_boreholes(gocad_dir=gocad_dir,
                                     asbuilt_dir=asbuilt_dir)
    # Hydrophones first
    for i, sens in sensors['Hydrophones'].iterrows():
        if sens[3] != ' -- ': # B3
            dep = float(sens[3])
            easts, norths, zs, deps = np.hsplit(well_dict['B3'], 4)
            # Get closest depth point
            dists = np.squeeze(np.abs(dep - deps))
            name = 'B3{:02d}'.format(sens[5])
        else: #B4
            dep = float(sens[4])
            easts, norths, zs, deps = np.hsplit(well_dict['B4'], 4)
            # Get closest depth point
            dists = np.squeeze(np.abs(dep - deps))
            # Use convention that hydrophone string #s zero-padded
            name = 'B4{:02d}'.format(sens[5])
        x = easts[np.argmin(dists)][0]
        y = norths[np.argmin(dists)][0]
        z = zs[np.argmin(dists)][0]
        sens_dict[name] = (x, y, z)
    for i, sens in sensors['Accelerometers'].iterrows():
        if sens[4] == 'Z': # All info in Z chan row
            bh = sens[7]
            dep = float(sens[9])
            easts, norths, zs, deps = np.hsplit(well_dict[bh], 4)
            # Name accelerometers after serial # (non zero-padded to keep
            # namespace clean for hydro strings)
            no = sens[6].split('_')[1].lstrip('0')
            name = '{}{}'.format(bh, no)
            dists = np.squeeze(np.abs(dep - deps))
            x = easts[np.argmin(dists)][0]
            y = norths[np.argmin(dists)][0]
            z = zs[np.argmin(dists)][0]
            sens_dict[name] = (x, y, z)
    # Do CASSMs
    for i, sens in sensors['Sources'].iterrows():
        try:
            dep = float(sens[2])
            bh = sens[4]
            easts, norths, zs, deps = np.hsplit(well_dict[bh], 4)
            name = 'S{}'.format(sens[1])
            dists = np.squeeze(np.abs(dep - deps))
            x = easts[np.argmin(dists)][0]
            y = norths[np.argmin(dists)][0]
            z = zs[np.argmin(dists)][0]
            sens_dict[name] = (x, y, z)
        except KeyError as e:
            print(e)
            continue
    # Do AE's
    for i, sens in sensors['AEs'].iterrows():
        if sens[2] != ' -- ': # B8
            dep = float(sens[2])
            easts, norths, zs, deps = np.hsplit(well_dict['B8'], 4)
            # Get closest depth point
            dists = np.squeeze(np.abs(dep - deps))
            name = sens[4]
        else: # B9
            dep = float(sens[3])
            easts, norths, zs, deps = np.hsplit(well_dict['B9'], 4)
            # Get closest depth point
            dists = np.squeeze(np.abs(dep - deps))
            # Use convention that hydrophone string #s zero-padded
            name = sens[4]
        x = easts[np.argmin(dists)][0]
        y = norths[np.argmin(dists)][0]
        z = zs[np.argmin(dists)][0]
        sens_dict[name] = (x, y, z)
    return sens_dict


def fsb_to_inv(path, gocad_dir, asbuilt_dir, orientations=False, debug=0):
    """
    Take excel file of sensor locations and build an Inventory

    :param path: Path to excel spreadsheet
    :param orientations: False or dict of orientation info
    :param debug:
    :return:
    """
    converter = FSB_converter()
    sens_dict = read_fsb_asbuilt(path, gocad_dir, asbuilt_dir)
    # Assemble dictionary of {station: {channel: infoz}}
    # Create dict before, then build inventory from channel level upwards
    sta_dict = {}
    extra_dict = {}
    for sta, loc in sens_dict.items():
        # Station location
        # Convert from SURF coords to lat lon, but keep local for actual use
        lon, lat, elev = converter.to_lonlat((loc[0], loc[1], loc[2]))
        depth = 0.0  # Until we do any orientations?
        # Save HMC coords to custom attributes of Station and Channel
        extra = AttribDict({
            'ch1903_east': {
                'value': loc[0],
                'namespace': 'smi:local/hmc'
            },
            'ch1903_north': {
                'value': loc[1],
                'namespace': 'smi:local/hmc'
            },
            'ch1903_elev': {
                'value': loc[2], # extra will preserve absolute elev
                'namespace': 'smi:local/hmc'
            }
        })
        # Not yet implemented; Pass orientations dict when we do
        if orientations:
            # TODO Something is real effed here. Answers are right though.
            dip_rad = np.arcsin(-orientations[sta]['Sz'])
            az_rad = np.arcsin(orientations[sta]['Sx'] / np.cos(dip_rad))
            dip = np.rad2deg(dip_rad)
            az = np.rad2deg(az_rad)
            # Force positive
            if az < 0:
                az += 360.
            # Correct
            if orientations[sta]['Sx'] < 0 and orientations[sta]['Sy'] < 0:
                az -= 270.
                az = 270. - az
            elif orientations[sta]['Sy'] < 0:
                az = 180 - az
            if debug > 0:
                print(np.array((orientations[sta]['Sx'],
                                orientations[sta]['Sy'],
                                orientations[sta]['Sz'])))
                print(az, dip)
        try:
            if orientations[sta]['Sensor'].endswith(('Z', 'X', 'Y')):
                chan = 'XN{}'.format(orientations[sta]['Sensor'][-1])
                # Geophones
                if orientations[sta]['Sensor'].startswith('G'):
                    no = orientations[sta]['Sensor'][-3]
                # Accelerometers
                else:
                    no = orientations[sta]['Sensor'].split('_')[1]
                sta_name = '{}{}'.format(orientations[sta]['Desc'], no)
                channel = Channel(code=chan, location_code='', latitude=lat,
                                  longitude=lon, elevation=elev, depth=depth,
                                  azimuth=az, dip=dip, response=Response())
                # channel.extra = extra
            elif orientations[sta]['Sensor'].startswith('Hydro'):
                chan = 'XN1'
                sta_name = '{}{}'.format(
                    orientations[sta]['Desc'],
                    orientations[sta]['Sensor'].split('-')[-1].zfill(2))
                channel = Channel(code=chan, location_code='', latitude=lat,
                                  longitude=lon, elevation=elev, depth=depth,
                                  response=Response())
        except TypeError as e:
            sta_name = sta
            if sta in fsb_accelerometers:
                channels = []
                for chan in ['XNZ', 'XNX', 'XNY']:
                    channels.append(Channel(code=chan, location_code='',
                                            latitude=lat, longitude=lon,
                                            elevation=elev, depth=depth,
                                            response=Response()))
            else:
                channel = Channel(code='XN1', location_code='', latitude=lat,
                                  longitude=lon, elevation=elev, depth=depth,
                                  response=Response())
                channels = [channel]
        extra_dict[sta_name] = extra
        sta_dict[sta_name] = channels
    stas = []
    for nm, chans in sta_dict.items():
        station = Station(code=nm, latitude=chans[0].latitude,
                          longitude=chans[0].longitude,
                          elevation=chans[0].elevation,
                          channels=chans)
        station.extra = extra_dict[nm]
        stas.append(station)
    inventory = Inventory(networks=[Network(code='FS', stations=stas)],
                          source='FSB')
    return inventory


def surf_stations_to_inv(excel_file, debug=0):
    """
    Take Petrs orientation excel file for the hydrophones/accelerometers
    and build an inventory for later use.
    :param excel_file: path to Petr's excel file (formatting hard-coded)
    :return: obspy.core.Inventory
    """
    # Call coordinate converter
    converter = SURF_converter()
    sta_df = pd.read_excel(excel_file, skiprows=[0,1,2,3], header=1, nrows=90)
    # Assemble dictionary of {station: {channel: infoz}}
    # Create dict before, then build inventory from channel level upwards
    sta_dict = {}
    extra_dict = {}
    for i, row in sta_df.iterrows():
        # Station location
        # Convert from SURF coords to lat lon, but keep local for actual use
        lon, lat, elev = converter.to_lonlat((row['Easting(m)'],
                                              row['Northing(m)'],
                                              row['Elev(m)']))
        # Correct for arbitrary zero 'depth' of 130m
        elev -= 130
        # Already accounted for in the elevation but will include here as its
        # ...a required arg for Channel()
        depth = row['Depth (m)']
        # Save HMC coords to custom attributes of Station and Channel
        extra = AttribDict({
            'hmc_east': {
                'value': row['Easting(m)'],
                'namespace': 'smi:local/hmc'
            },
            'hmc_north': {
                'value': row['Northing(m)'],
                'namespace': 'smi:local/hmc'
            },
            'hmc_elev': {
                'value': row['Elev(m)'], # extra will preserve absolute elev
                'namespace': 'smi:local/hmc'
            }
        })
        # Sort out azimuth and dip for this channel (if it exists)
        if not np.isnan(row['Sx']):
            # TODO Something is real effed here. Answers are right though.
            dip_rad = np.arcsin(-row['Sz'])
            az_rad = np.arcsin(row['Sx'] / np.cos(dip_rad))
            dip = np.rad2deg(dip_rad)
            az = np.rad2deg(az_rad)
            # Force positive
            if az < 0:
                az += 360.
            # Correct
            if row['Sx'] < 0 and row['Sy'] < 0:
                az -= 270.
                az = 270. - az
            elif row['Sy'] < 0:
                az = 180 - az
            if debug > 0:
                print(np.array((row['Sx'], row['Sy'], row['Sz'])))
                print(az, dip)
        if row['Sensor'].endswith(('Z', 'X', 'Y')):
            chan = 'XN{}'.format(row['Sensor'][-1])
            # Geophones
            if row['Sensor'].startswith('G'):
                continue
            # Accelerometers
            else:
                no = row['Sensor'].split('_')[1]
            sta_name = '{}{}'.format(row['Desc'], no)
            if sta_name in ['OB14', 'OT17', 'PDT2', 'PDT5', 'PSB8', 'PST11']:
                # These are geode stations only, skip
                continue
            channel = Channel(code=chan, location_code='', latitude=lat,
                              longitude=lon, elevation=elev, depth=depth,
                              azimuth=az, dip=dip, response=Response())
            # channel.extra = extra
        elif row['Sensor'].startswith('Hydro'):
            chan = 'XN1'
            sta_name = '{}{}'.format(row['Desc'],
                                     row['Sensor'].split('-')[-1].zfill(2))
            channel = Channel(code=chan, location_code='', latitude=lat,
                              longitude=lon, elevation=elev, depth=depth,
                              response=Response())
        extra_dict[sta_name] = extra
            # channel.extra = extra
        if sta_name in sta_dict.keys():
            sta_dict[sta_name].append(channel)
        else:
            sta_dict[sta_name] = [channel]
    # Now loop station dict to create inventory
    stas = []
    for nm, chans in sta_dict.items():
        station = Station(code=nm, latitude=chans[0].latitude,
                          longitude=chans[0].longitude,
                          elevation=chans[0].elevation,
                          channels=chans)
        station.extra = extra_dict[nm]
        stas.append(station)
    # Build inventory
    inventory = Inventory(networks=[Network(code='SV', stations=stas)],
                          source='SURF')
    return inventory


def cassm_to_smf_dist(well_dict, inventory, outfile):
    """Calculate the minimum distance between each CASSM source and each
    fiber-installed borehole"""

    smf_dists = {}
    cassms = inventory.select(station='S*')
    with open(outfile, 'w') as f:
        for cassm in cassms[0]:
            ch1903e = cassm.extra['ch1903_east'].value
            ch1903n = cassm.extra['ch1903_north'].value
            ch1903el = cassm.extra['ch1903_elev'].value
            f.write('\n{},{},{},{}\n'.format(cassm.code, ch1903e,
                                         ch1903n, ch1903el))
            for well in ['B1', 'B2', 'B3', 'B4', 'B5',
                         'B6', 'B7', 'B8', 'B9', 'B10']:
                pts = well_dict[well]
                dists = [np.sqrt((pt[0] - ch1903e)**2 + (pt[1] - ch1903n)**2 +
                                 (pt[2] - ch1903el)**2)
                         for pt in pts]
                close_pt = pts[np.argmin(dists)]
                f.write('{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(
                    well, np.min(dists), close_pt[0], close_pt[1],
                    close_pt[2], close_pt[3]))
    return


def consolidate_inv_channels(inventory):
    """
    Check for same channels with contiguous times and merge into same chan

    :param inventory: Inventory object
    :return:
    """
    # Unique chans
    new_chans = []
    loc_chans = list(set(['{}.{}'.format(chan.location_code, chan.code)
                          for sta in inventory[0] for chan in sta]))
    for lc in loc_chans:
        lc_inv = inventory.select(location=lc.split('.')[0],
                                  channel=lc.split('.')[-1])
        start = min([c.start_date for c in lc_inv[0][0]])
        nc = lc_inv[0][0][0]
        nc.start_date = start
        nc.end_date = None
        new_chans.append(nc)
    inventory[0][0].channels = new_chans
    return inventory


def update_G2(inv):
    """Helper to modify response for G2 borehole geophone"""
    G_mod3 = 81.82  # V/m/s
    fc_mod3 = 8.0  # corner frequency
    damp_mod3 = 0.70  # damping
    # pole & zero
    poles_mod3 = [-(damp_mod3 + M.sqrt(1 - damp_mod3 ** 2) * 1j) *
                  2 * np.pi * fc_mod3,
                  -(damp_mod3 - M.sqrt(1 - damp_mod3 ** 2) * 1j) *
                  2 * np.pi * fc_mod3]
    for net in inv:
        for sta in net:
            if sta.code == 'NSMTC':
                for chan in sta:
                    staloc = '{}.{}'.format(sta.code, chan.location_code)
                    chan.azimuth = nsmtc_orientation[staloc][chan.code[-1]]
                    if chan.location_code == 'G2':
                        # update pole & zero
                        # chan.response.response_stages[0]._poles = poles_mod3
                        # update sensitivity value
                        # also x -1 to polarity flip for Z comp
                        if chan.code == 'CHZ':
                            chan.response.response_stages[0].stage_gain = (
                                    G_mod3 * -1)
                            chan.dip = 90.
                        else:
                            chan.response.response_stages[0].stage_gain = G_mod3
                        # # Normalization factor
                        # chan.response.response_stages[0].normalization_factor = 0.9998924742191032
                        # Recompute sensitivity
                    if chan.location_code == 'G1' and chan.code[-1] == 'Z':
                        chan.response.response_stages[0].stage_gain *= -1.
                        chan.dip = 90.
    return inv


def update_BX(inv):
    """Helper for modifying the B* channels"""
    for net in inv:
        for sta in net:
            for chan in sta:
                if sta.code == 'NSMTC' and chan.code == 'CNZ':
                    chan.response.response_stages[0].stage_gain = (40.0 / 9.80665) * -1
                    chan.dip = 90.
                elif sta.code == 'NSMTC' and chan.code in ['CN2', 'CN1']:
                    chan.response.response_stages[0].stage_gain = (40.0 / 9.80665)
    return inv


def modify_SAULN_inventory(inv):
    inv = update_BX(inv)
    inv = update_G2(inv)
    # Lowercase units
    for net in inv:
        for sta in net:
            for chan in sta:
                for stage in chan.response.response_stages:
                    if stage.input_units != 'V':
                        stage.input_units = stage.input_units.lower()
                    if stage.output_units != 'V':
                        stage.output_units = stage.output_units.lower()
    return inv