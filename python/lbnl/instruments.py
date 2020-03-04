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

from lbnl.coordinates import SURF_converter
from lbnl.boreholes import create_FSB_boreholes
from obspy.core.util import AttribDict
from obspy.core.inventory import Inventory, Network, Station, Channel, Response


def read_fsb_asbuilt(path):
    sens_dict = {}
    inv = Inventory(networks=[Network(stations=[], code='FS')],
                    source='FSB')
    """Read excel spreadsheet of sensor wells and depths"""
    sensors = pd.read_excel(path, sheet_name=None, skiprows=np.arange(5),
                            usecols = np.arange(1, 8), header = None)
    well_dict = create_FSB_boreholes()
    # Hydrophones first
    for i, sens in sensors['Hydrophones'].iterrows():
        if sens[2] != ' -- ': # B3
            dep = float(sens[2])
            easts, norths, zs, deps = np.hsplit(well_dict['B3'], 4)
            # Get closest depth point
            dists = np.squeeze(np.abs(dep - deps))
            name = 'B3{:02d}'.format(sens[4])
        else: #B4
            dep = float(sens[3])
            easts, norths, zs, deps = np.hsplit(well_dict['B4'], 4)
            # Get closest depth point
            dists = np.squeeze(np.abs(dep - deps))
            # Use convention that hydrophone string #s zero-padded
            name = 'B4{:02d}'.format(sens[4])
        x = easts[np.argmin(dists)][0]
        y = norths[np.argmin(dists)][0]
        z = zs[np.argmin(dists)][0]
        sens_dict[name] = (x, y, z)
    for i, sens in sensors['Accelerometers'].iterrows():
        if sens[2] == 'Z': # All info in Z chan row
            bh = sens[5]
            dep = float(sens[7])
            easts, norths, zs, deps = np.hsplit(well_dict[bh], 4)
            # Name accelerometers after serial # (non zero-padded to keep
            # namespace clean for hydro strings)
            no = sens[4].split('_')[1].lstrip('0')
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
            name = sens[3]
            dists = np.squeeze(np.abs(dep - deps))
            x = easts[np.argmin(dists)][0]
            y = norths[np.argmin(dists)][0]
            z = zs[np.argmin(dists)][0]
            sens_dict[name] = (x, y, z)
        except KeyError:
            continue
    # Do AE's
    for i, sens in sensors['AEs'].iterrows():
        if sens[2] != ' -- ': # B8
            dep = float(sens[2])
            easts, norths, zs, deps = np.hsplit(well_dict['B8'], 4)
            # Get closest depth point
            dists = np.squeeze(np.abs(dep - deps))
            name = 'B8{:02d}'.format(sens[4])
        else: # B9
            dep = float(sens[3])
            easts, norths, zs, deps = np.hsplit(well_dict['B9'], 4)
            # Get closest depth point
            dists = np.squeeze(np.abs(dep - deps))
            # Use convention that hydrophone string #s zero-padded
            name = 'B9{:02d}'.format(sens[4])
        x = easts[np.argmin(dists)][0]
        y = norths[np.argmin(dists)][0]
        z = zs[np.argmin(dists)][0]
        sens_dict[name] = (x, y, z)
    return sens_dict


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
    network = Network(code='SV')
    inventory = Inventory(networks=[network], source=obspy.__version__)
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
