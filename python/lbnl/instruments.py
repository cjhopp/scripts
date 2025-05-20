#!/usr/bin/python
"""
Functions for working with instrumentation for EGS Collab, FS-B, etc...

IMPORTANT
***********************************************
For SURF, the arbitrary zero depth point is elev = 130 m
***********************************************

"""
import os
import pyproj

import numpy as np
import pandas as pd
import math as M
import matplotlib.pyplot as plt

from glob import glob
from copy import deepcopy
from obspy import read_inventory, UTCDateTime
from obspy.core.util import AttribDict
from obspy.core.inventory import Inventory, Network, Station, Channel, Response
from obspy.core.inventory import ResponseListResponseStage
from obspy.core.inventory.response import ResponseListElement, InstrumentPolynomial
from obspy.core.inventory.response import InstrumentSensitivity, ResponseStage
from obspy.core.inventory.util import Latitude, Longitude, Equipment
from mpl_toolkits.mplot3d import Axes3D

from lbnl.coordinates import SURF_converter, FSB_converter
from lbnl.boreholes import create_FSB_boreholes

dac_locations = {
    'DAC01': (-118.33740, 38.85207, 1275),
    'DAC02': (-118.30798, 38.84438, 1281),
    'DAC03': (-118.31027, 38.85007, 1277),
    'DAC04': (-118.32735, 38.84718, 1275),
    'DAC05': (-118.31956, 38.84067, 1285),
    'DAC06': (-118.33182, 38.84059, 1287),
    'DAC07': (-118.30060, 38.83677, 1290),
    'DAC08': (-118.32736, 38.82941, 1307),
    'DAC09': (-118.31497, 38.83273, 1300)
}

jv_locations = {
    'JV01': (-117.4805, 40.1777, 1429, 200.),
    #### JV02 has moved since this point
    'JV02': (-117.4747, 40.1687, 1474, 101.5),
    'JV03': (-117.4698, 40.1761, 1519, 100.),
    'JV04': (-117.4725, 40.1837, 1468, 169.5),
    'JV05': (-117.4835, 40.1709, 1426, 150.2664),
    'JV06': (-117.4910, 40.1736, 1376, 202.4),
    'JV07': (-117.4679, 40.1632, 1533, 52.7),
    'JV08': (-117.5024, 40.1657, 1321, 202.4)
}

fsb_accelerometers = ['B31', 'B34', 'B42', 'B43', 'B551', 'B585', 'B647',
                      'B659', 'B748', 'B75']

nsmtc_orientation = {'NSMTC.G1': {'E': 90., 'N': 0., 'Z': 0},
                     'NSMTC.B1': {'1': 231.31, '2': 321.31, 'Z': 0},
                     'NSMTC.B2': {'1': 119.56, '2': 209.56, 'Z': 0},
                     'NSMTC.B3': {'1': 252.60, '2': 342.60, 'Z': 0},
                     'NSMTC.G2': {'1': 147.75, '2': 237.75, 'Z': 0}}

resp_labl_map = {'RESP.XX.NS491..BNZ.LowNoise.0_005_1000.60V.2G': 'Silicon Audio ULN Accelerometer',
                 'RESP.XX.NS129..BNZ.Titan.DC_430.20V.4G': 'Nanometrics Titan Accelerometer',
                 'RESP.XX.NS380..SLZ.HS1LT.3810.115000.2.76': 'Geospace HS-1-LT Geophone',
                 'RESP.XX.NS391..SHZ.GS11D.10.380.NONE.32': 'Geospace GS-11D Geophone',
                 'RESP.XX.NS539..BHZ.Trillium120Q.120.1500': 'Nanometrics Trillium 120s PH broadband'}

resp_outp_map = {'RESP.XX.NS491..BNZ.LowNoise.0_005_1000.60V.2G': 'VEL',
                 'RESP.XX.NS129..BNZ.Titan.DC_430.20V.4G': 'VEL',
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


# Because I screwed up the SS miniseed convertion, GB01 is now the western, 1C station
# ...GB02 is the eastern, 3C station
oee_station_map = {
    'GB2': {'latitude': 40.4860, 'longitude': -88.4813, 'elev': 231.},
    'GB1': {'latitude': 40.4860, 'longitude': -88.4685, 'elev': 241.},
}


oee_channel_map = {
    'GB2': {1: ['DPZ'], 2: ['DPZ'], 3: ['DPZ'], 4: ['DPZ'], 5: ['DPZ'], 6: ['DPZ'], 7: ['DPZ'], 8: ['DPZ'],
            9: ['DPZ'], 10: ['DPZ'], 11: ['DPZ'], 12: ['DPZ'], 13: ['DPZ'], 14: ['DPZ']},
    'GB1': {1: ['DPZ', 'DP1', 'DP2'], 2: ['DPZ', 'DP1', 'DP2'], 3: ['DPZ', 'DP1', 'DP2'], 4: ['DPZ', 'DP1', 'DP2'],
             5: ['DPZ', 'DP1', 'DP2'], 6: ['DPZ', 'DP1', 'DP2'], 7: ['DPZ', 'DP1', 'DP2'], 8: ['DPZ', 'DP1', 'DP2'],
             9: ['DPZ', 'DP1', 'DP2'], 10: ['DPZ', 'DP1', 'DP2'], 11: ['DPZ', 'DP1', 'DP2'], 12: ['DPZ', 'DP1', 'DP2'],
             13: ['DPZ', 'DP1', 'DP2'], 14: ['DPZ', 'DP1', 'DP2']}
}

oee_depth_map = np.linspace(6.7, 136.7, 14)



class Inventory3D(Inventory):
    """
    Subclass of Inventory to add 3D plotting functionality
    
    """
    def __init__(self, inventory):
        # Directly initialize the Inventory part by extracting networks
        super().__init__(networks=inventory.networks, source=inventory.source, sender=inventory.sender,
                         created=inventory.created, module=inventory.module, module_uri=inventory.module_uri)
    
    def plot_3d(self, utm_zone, elev_scale=1.0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Define a pyproj Proj for the UTM zone for central Illinois
        utm_proj = pyproj.Proj(proj="utm", zone=utm_zone, datum="WGS84")
        
        # To store unique channel combinations and their assigned colors
        channel_combinations = {}
        colors = plt.cm.get_cmap("tab10")  # Use a colormap for distinct colors
        
        color_index = 0  # Index to go through available colors
        labels_added = set()  # To avoid duplicate labels in the legend

        # Loop over all stations in the inventory
        for network in self:
            for station in network:
                east, north = utm_proj(station.longitude, station.latitude)  # Convert to UTM
                depth = station.elevation - station[0].depth
                
                # Get the unique combination of channels for this station
                channel_names = set([ch.code for ch in station.channels])
                channel_combination = tuple(sorted(channel_names))  # Sort to ensure uniqueness
                # If this combination hasn't been encountered, assign it a new color
                if channel_combination not in channel_combinations:
                    channel_combinations[channel_combination] = colors(color_index % 10)  # Use next color in colormap
                    color_index += 1

                # Plot with color corresponding to the channel combination
                color = channel_combinations[channel_combination]
                if channel_combination not in labels_added:
                    ax.scatter(east, north, depth, c=[color], marker='v', label=str(channel_combination))
                    labels_added.add(channel_combination)
                else:
                    ax.scatter(east, north, depth, c=[color], marker='v', label="_nolegend_")

        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_zlabel('Elevation (m)')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        ax.set_aspect('equal')  # Make axes units equal
        plt.show()


def edit_oee_inventory(old_inventory):
    """
    Change the original OEE inventory to include DAS and geophone channels as separate stations
    """
    response = old_inventory.select(station='GB01')[0][0][-1].response
    new_inventory = Inventory(networks=[Network(code='ZF', start_date=UTCDateTime(2024, 8, 22))])
    for sta, chans in oee_channel_map.items():
        # Create a new station for each channel
        for no, chans in chans.items():
            new_sta = Station(code=f'{sta}{no:02d}', latitude=oee_station_map[sta]['latitude'],
                              longitude=oee_station_map[sta]['longitude'],
                              elevation=oee_station_map[sta]['elev'],
                              start_date=UTCDateTime(2024, 8, 22))
            for c in chans:
                channel = Channel(code=c, location_code='', latitude=new_sta.latitude,
                                  longitude=new_sta.longitude, elevation=new_sta.elevation,
                                  depth=oee_depth_map[no-1], sample_rate=1000.,
                                  response=deepcopy(response),
                                  start_date=UTCDateTime(2024, 8, 22))
                new_sta.channels.append(channel)
            new_inventory[0].stations.append(new_sta)

    # for sta in old_inventory[0]:
    #     if sta.code == 'GB01':
    #         new_sta.channels.sort(key=lambda x: x.location_code)
    #         for i, chan in enumerate(new_sta.channels):
    #             try:
    #                 loc = int(chan.location_code.lstrip('0'))
    #             except ValueError:
    #                 continue  # SAULN
    #             new_sta = sta.copy()
    #             new_sta.code = f'GB1{i-2:02d}'
    #             chan.code = oee_channel_map['GB01'][i-2]
    #             chan.location_code = ''
    #             chan.depth = oee_depth_map[loc-1]
    #             chan.latitude = oee_station_map['GB01']['latitude']
    #             chan.longitude = oee_station_map['GB01']['longitude']
    #             chan.elevation = oee_station_map['GB01']['elev']
    #     elif sta.code == 'GB02':
    #         for chan in sta.channels:
    #             try:
    #                 loc = int(chan.location_code.lstrip('0'))
    #             except ValueError:
    #                 continue  # SAULN
    #             new_sta = sta.copy()
    #             new_sta.code = f'GB1{i-2:02d}'
    #             chan.code = oee_channel_map['GB02'][int(chan.location_code)]
    #             chan.location_code = ''
    #             chan.depth = oee_depth_map[loc-1]
    #             chan.latitude = oee_station_map['GB02']['latitude']
    #             chan.longitude = oee_station_map['GB02']['longitude']
    #             chan.elevation = oee_station_map['GB02']['elev']
    # Add DAS stations
    das500 = Station(code='D0500', latitude=oee_station_map['GB2']['latitude'],
                     longitude=oee_station_map['GB1']['longitude'],
                     elevation=oee_station_map['GB1']['elev'])
    das500chan = old_inventory.select(station='GB01')[0][0][0].copy()
    das500chan.code = 'FSF'
    das500chan.location_code = ''
    das500chan.depth = 500.
    das500.channels = [das500chan]
    das1200 = Station(code='D1200', latitude=oee_station_map['GB1']['latitude'],
                    longitude=oee_station_map['GB1']['longitude'],
                    elevation=oee_station_map['GB1']['elev'])
    das1200chan = old_inventory.select(station='GB01')[0][0][0].copy()
    das1200chan.code = 'FSF'
    das1200chan.location_code = ''
    das1200chan.depth = 1200.
    das1200.channels = [das1200chan]
    new_inventory[0].stations.extend([das500, das1200])
    return new_inventory


def nodal_to_inv(excel_sheet, channel_inv):
    """
    Take Nori's nodal spreadsheet and return an inventory object

    :param excel_sheet: Path to the sheet
    :param channel_inv: Inventory containing the response to be used for all channels (not correct)
    :return:
    """
    nodal_sheet = pd.read_excel(excel_sheet, sheet_name='All sensors')
    # Select only rows where
    imu_sheet = nodal_sheet[nodal_sheet['flag (=0: IGU, =1: IGU_EB, =2: Sercel, =3: IMU, =4: IMU_3C)'] > 2]
    inv_nodes = Inventory(networks=[Network(code='SS')])
    used_stas = []
    for i, station in imu_sheet.iterrows():
        if np.isnan(station['Elevation_m_1']):
            continue
        start = UTCDateTime(station['Start_time_1'])
        try:
            end = UTCDateTime(station['End_time_1'])
        except TypeError:
            end = None
        code = str(int(station['IMU_id']))[-5:]
        if not code in used_stas:
            sta = Station(code=code, latitude=station['Latitude_deg_1'], longitude=station['Longitude_deg_1'],
                          elevation=station['Elevation_m_1'], start_date=start)
            chan = channel_inv[0][0][0].copy()
            chan.location_code = ''
            chan.code = 'GP{}'.format(station['Channel code'])
            chan.start_date = start
            chan.end_date = end
            chan.latitude = station['Latitude_deg_1']
            chan.longitude = station['Longitude_deg_1']
            chan.elevation = station['Elevation_m_1']
            chan.sample_rate = 1000.
            chan.depth = 0.
            if chan.code[-1] == 'Z':
                chan.azimuth = 0.0
                chan.dip = -90.
            elif chan.code[-1] == 'N':
                chan.azimuth = 0.
                chan.dip = 0.
            elif chan.code[-1] == 'E':
                chan.azimuth = 90.
                chan.dip = 0.
            sta.channels.append(chan)
            inv_nodes[0].stations.append(sta)
            used_stas.append(code)
        else:
            which = [i for i, sta in enumerate(inv_nodes[0].stations) if sta.code == code][0]
            sta = inv_nodes[0][which]
            chan = channel_inv[0][0][0].copy()
            chan.location_code = ''
            chan.code = 'GP{}'.format(station['Channel code'])
            chan.start_date = start
            chan.end_date = end
            chan.latitude = sta.latitude
            chan.longitude = sta.longitude
            chan.elevation = sta.elevation
            chan.sample_rate = 1000.
            chan.depth = 0.
            if chan.code[-1] == 'Z':
                chan.azimuth = 0.0
                chan.dip = -90.
            elif chan.code[-1] == 'N':
                chan.azimuth = 0.
                chan.dip = 0.
            elif chan.code[-1] == 'E':
                chan.azimuth = 90.
                chan.dip = 0.
            sta.channels.append(chan)
            starts = [c.start_date for c in sta.channels]
            sta.start_date = min(starts)
    return inv_nodes


def plot_resp(resp_dir, min_freq, sampling_rate):
    """
    Plot the RESP curves in a directory

    :param resp_dir: Directory containing RESP files
    :param min_freq: Minimum frequency to plot
    :param sampling_rate: Sampling rate for theoretical data
    """
    resps = glob('{}/RESP*'.format(resp_dir))
    file_order = []
    fig, axes = plt.subplots(nrows=4, figsize=(9, 12))
    for i, resp in enumerate(resps):
        base = os.path.basename(resp)
        file_order.append(base)
        lab = resp_labl_map[base]
        output = resp_outp_map[base]
        read_inventory(resp)[0][0][0].response.plot(
            output=output, min_freq=min_freq, sampling_rate=sampling_rate,
            label=lab, axes=[axes[0], axes[2]], unwrap_phase=True, show=False)
        read_inventory(resp)[0][0][0].response.plot(
            output='ACC', min_freq=min_freq, sampling_rate=sampling_rate,
            axes=[axes[1], axes[3]], unwrap_phase=True, show=False)
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
    for i, ln in enumerate(axes[1].get_lines()):
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
    axes[2].grid(True)
    axes[2].set_yticks([-2 * np.pi, -(3/2) * np.pi, -np.pi, -np.pi / 2, 0,
                        np.pi / 2, np.pi])
    axes[2].set_yticklabels([r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$\pi$',
                             r'$\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$',
                             r'$\pi$'])
    axes[3].grid(True)
    axes[3].set_yticks([-2 * np.pi, -(3/2) * np.pi, -np.pi, -np.pi / 2, 0,
                        np.pi / 2, np.pi])
    axes[3].set_yticklabels([r'$-2\pi$', r'$-\frac{3\pi}{2}$', r'$\pi$',
                             r'$\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$',
                             r'$\pi$'])
    axes[0].set_ylabel('Amplitude [VEL]')
    axes[1].set_ylabel('Amplitude [ACC]')
    axes[2].set_ylabel('Velocity phase [rad]')
    axes[3].set_ylabel('Acceleration phase [rad]')
    axes[3].set_xlabel('Frequency [Hz]')
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
    avg_amp = {'XNZ': [], 'XNY': [], 'XNX': []}
    avg_phase = {'XNZ': [], 'XNY': [], 'XNX': []}
    avg_sensitivity = {'XNZ': [], 'XNY': [], 'XNX': []}
    avg_freq = []
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
            chan_code = 'XN{}'.format(chan_map[nm])
            # Set samp_rate to 40 kHz so that Nyquist is below max shake freq
            chan = Channel(code=chan_code, location_code='',
                           latitude=lat, longitude=lon, elevation=0., depth=0.,
                           sample_rate=40000.,
                           sensor=Equipment(
                               type='IEPE Accelerometer',
                               description='Piezoelectric accelerometer',
                               manufacturer='MMF',
                               model='KS943B.100',
                               serial_number=serial))
            values = df[['[Hz]', '[m/s²]', '[°]']].values
            # Add to dict for average channel estimate later
            avg_amp[chan_code].append(values[:, 1])
            avg_phase[chan_code].append(values[:, 2])
            avg_sensitivity[chan_code].append(float(sens_dict[nm]))
            avg_freq = values[:, 0]
            response_elements = [ResponseListElement(
                frequency=values[i][0], amplitude=values[i][1],
                phase=values[i][2])
                                 for i in range(values.shape[0])]
            # Add a value at zero to avoid deconvolution errors
            response_elements.insert(0, ResponseListElement(
                frequency=0., amplitude=values[0][1], phase=values[0][2]
            ))
            # Add a stage for the Vibbox (gain 100)
            vbox_stage = ResponseStage(
                stage_sequence_number=2, stage_gain=100, stage_gain_frequency=80,
                input_units='COUNTS', output_units='V'
            )
            resp_stage = ResponseListResponseStage(
                response_list_elements=response_elements, stage_gain=1,
                stage_gain_frequency=80., input_units='M/S**2',
                output_units='V',
                stage_sequence_number=1
            )
            sensitivity = InstrumentSensitivity(
                value=float(sens_dict[nm]), frequency=80.,
                input_units='M/S**2', output_units='COUNTS', frequency_range_start=5,
                frequency_range_end=15850,
                frequency_range_db_variation=3)
            response = Response(instrument_sensitivity=sensitivity,
                                response_stages=[resp_stage, vbox_stage])
            response.recalculate_overall_sensitivity(80.)
            chan.response = response
            sta.channels.append(chan)
            # chan.response.plot(min_freq=2.4, sampling_rate=40000.)
        inv[0].stations.append(sta)
    # Now make an 'average' channel for the other sensors
    avg_sta = Station(code='AVG', latitude=lat, longitude=lon,
                      elevation=0.)
    for c in ['XNX', 'XNY', 'XNZ']:
        chan = Channel(code=c, location_code='',
                       latitude=lat, longitude=lon, elevation=0., depth=0.,
                       sample_rate=40000.,
                       sensor=Equipment(
                           type='IEPE Accelerometer',
                           description='Piezoelectric accelerometer',
                           manufacturer='MMF',
                           model='KS943B.100',
                           serial_number='9999'))
        amp = np.array(avg_amp[c]).mean(axis=0)
        pha = np.array(avg_phase[c]).mean(axis=0)
        response_elements = [ResponseListElement(
            frequency=avg_freq[i], amplitude=amp[i],
            phase=pha[i])
            for i in range(avg_freq.size)]
        # Add a value at zero to avoid deconvolution errors
        response_elements.insert(0, ResponseListElement(
            frequency=0., amplitude=amp[0], phase=pha[0]
        ))
        # Add a stage for the Vibbox (gain 100)
        vbox_stage = ResponseStage(
            stage_sequence_number=2, stage_gain=100, stage_gain_frequency=80,
            input_units='COUNTS', output_units='V'
        )
        resp_stage = ResponseListResponseStage(
            response_list_elements=response_elements, stage_gain=1,
            stage_gain_frequency=80., input_units='M/S**2',
            output_units='V',
            stage_sequence_number=1
        )
        sensitivity = InstrumentSensitivity(
            value=np.array(avg_sensitivity[c]).mean(), frequency=80.,
            input_units='M/S**2', output_units='COUNTS', frequency_range_start=5,
            frequency_range_end=15850,
            frequency_range_db_variation=3)
        response = Response(instrument_sensitivity=sensitivity,
                            response_stages=[resp_stage, vbox_stage])
        response.recalculate_overall_sensitivity(80.)
        chan.response = response
        avg_sta.channels.append(chan)
    inv[0].stations.append(avg_sta)
    if plot:
        inv.plot_response(min_freq=2.4, plot_degrees=True)
    return inv


def read_fsb_asbuilt(excel_path):
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
    inventory = Inventory()
    inventory.networks = [Network(code='FS')]
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
    for nm, chans in sta_dict.items():
        station = Station(code=nm, latitude=chans[0].latitude,
                          longitude=chans[0].longitude,
                          elevation=chans[0].elevation,
                          channels=chans)
        station.extra = extra_dict[nm]
        inventory[0].stations.append(station)
    return inventory


def surf_4100_to_inv(location_file, response_inv, plot=False):
    """
    Combine the xyz Homestake locations and MMF calibration responses into
    an Inventory object for the 4100L
    """
    converter = SURF_converter()
    sta_df = pd.read_csv(location_file)
    inv = Inventory()
    serial_map = {'GMF1': '21010', 'GMF2': '21015', 'GMF3': '21027'}
    inv.networks = [Network(code='CB')]
    for _, row in sta_df.iterrows():
        print(row)
        sta_code = row['Sensor name']
        # Station location
        # Convert from SURF coords to lat lon, but keep local for actual use
        lon, lat, elev = converter.to_lonlat((row['x_ft'] * 0.3048,
                                              row['y_ft'] * 0.3048,
                                              row['z_ft'] * 0.3048))
        print(lon, lat, elev)
        # Just leave as zero here and convert HMC feet elevation to m
        depth = 0.0
        # Save HMC coords to custom attributes of Station and Channel
        extra = AttribDict({
            'hmc_east': {
                'value': row['x_ft'],
                'namespace': 'smi:local/hmc'
            },
            'hmc_north': {
                'value': row['y_ft'],
                'namespace': 'smi:local/hmc'
            },
            'hmc_elev': {
                'value': row['z_ft'] * 0.3048,
                'namespace': 'smi:local/hmc'
            }
        })
        if sta_code.startswith('TS'):
            # Hydrophone or CASSM, wet well
            if 'SS' in sta_code:
                # Cassm (Y for unspecified instrument)
                chan_code = 'XY1'
                chans = [Channel(code=chan_code, location_code='', latitude=lat,
                                 longitude=lon, elevation=elev, depth=depth,
                                 response=Response())]
            else:
                # Hydrophone (D), Downhole (H) per SEED manual
                chan_code = 'XDH'
                chans = [Channel(code=chan_code, location_code='', latitude=lat,
                                 longitude=lon, elevation=elev, depth=depth,
                                 response=Response())]
        elif 'S' in sta_code:
            # Grouted CASSM
            chan_code = 'XY1'
            chans = [Channel(code=chan_code, location_code='', latitude=lat,
                             longitude=lon, elevation=elev, depth=depth,
                             response=Response())]
        else:
            # Grouted accelerometer
            chans = []
            try:
                serial = serial_map[sta_code]
            except KeyError:
                serial = '9999'
            for chan_code in ['XNX', 'XNY', 'XNZ']:
                # Set samp_rate to 40 kHz so that Nyquist is below max shake f
                chan = Channel(code=chan_code, location_code='',
                               latitude=lat, longitude=lon, elevation=elev,
                               depth=0., sample_rate=40000.,
                               sensor=Equipment(
                                   type='IEPE Accelerometer',
                                   description='Piezoelectric accelerometer',
                                   manufacturer='MMF',
                                   model='KS943B.100',
                                   serial_number=serial))
                # Apply exact response for the three tested sensors,
                # ...otherwise use the average
                avg_resp = response_inv.select(
                    station='AVG', channel=chan_code)[0][0][0].response
                chan.response = avg_resp
                chans.append(chan)
        sta = Station(code=sta_code, latitude=chans[0].latitude,
                      longitude=chans[0].longitude,
                      elevation=chans[0].elevation,
                      channels=chans)
        sta.extra = extra
        inv[0].stations.append(sta)
    return inv


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
    for net in inv:
        for sta in net:
            if sta.code == 'NSMTC':
                for chan in sta:
                    staloc = '{}.{}'.format(sta.code, chan.location_code)
                    chan.azimuth = nsmtc_orientation[staloc][chan.code[-1]]
                    if chan.location_code == 'G2':
                        if chan.code == 'CHZ':
                            chan.response.response_stages[0].stage_gain = G_mod3
                            chan.dip = 90.
                        else:
                            chan.response.response_stages[0].stage_gain = G_mod3
                    if chan.location_code == 'G1' and chan.code[-1] == 'Z':
                        chan.dip = 90.
                    chan.response.recalculate_overall_sensitivity()
    return inv


def update_BX(inv):
    """Helper for modifying the B* channels"""
    for net in inv:
        for sta in net:
            if not sta.code == 'NSMTC':
                continue
            for chan in sta:
                if chan.location_code[0] != 'B':
                    continue
                if chan.code == 'CNZ':
                    chan.dip = 90.
                if chan.start_date == UTCDateTime(2017, 8, 20):
                    chan.response.response_stages[0].stage_gain = (60.0 /
                                                                   9.80665)
                else:
                    chan.response.response_stages[0].stage_gain = (40.0 /
                                                                   9.80665)
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