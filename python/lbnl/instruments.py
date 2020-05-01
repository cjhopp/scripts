#!/usr/bin/python
"""
Functions for working with instrumentation for EGS Collab, FS-B, etc...

IMPORTANT
***********************************************
For SURF, the arbitrary zero depth point is elev = 130 m
***********************************************

"""

import obspy

import numpy as np
import pandas as pd

from lbnl.coordinates import SURF_converter, FSB_converter
from lbnl.boreholes import create_FSB_boreholes
from obspy.core.util import AttribDict
from obspy.core.inventory import Inventory, Network, Station, Channel, Response

fsb_accelerometers = ['B31', 'B34', 'B42', 'B43', 'B551', 'B585', 'B647',
                      'B659', 'B748', 'B75']

def read_fsb_asbuilt(path):
    """
    Read the as-built excel spreadsheet for FSB and return a dictionary of all
    the stations and sources containing locations

    :param path: Path to excel spreadsheet
    :return:
    """
    sens_dict = {}
    # Read excel spreadsheet of sensor wells and depths
    sensors = pd.read_excel(path, sheet_name=None, skiprows=np.arange(5),
                            usecols=np.arange(1, 10), header=None)
    well_dict = create_FSB_boreholes()
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


def fsb_to_inv(path, orientations=False, debug=0):
    """
    Take excel file of sensor locations and build an Inventory

    :param path: Path to excel spreadsheet
    :param orientations: False or dict of orientation info
    :param debug:
    :return:
    """
    converter = FSB_converter()
    sens_dict = read_fsb_asbuilt(path)
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
                no = row['Sensor'][-3]
            # Accelerometers
            else:
                no = row['Sensor'].split('_')[1]
            sta_name = '{}{}'.format(row['Desc'], no)
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