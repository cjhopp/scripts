#!/usr/bin/python

"""
Functions for the processing, reading, converting of events and event files

IMPORTANT
***********************************************
Arbitrary zero depth point is elev = 130 m
***********************************************
"""

import os

import numpy as np

from glob import glob
from subprocess import call
from obspy import UTCDateTime, Catalog, read, read_inventory, Stream, Trace
from obspy.core.util import AttribDict
from obspy.core.event import Pick, Origin, Arrival, Event, Magnitude,\
    WaveformStreamID, ResourceIdentifier, OriginQuality, OriginUncertainty,\
    QuantityError
from obspy.signal.rotate import rotate2zne
from lbnl.coordinates import SURF_converter


def surf_events_to_cat(loc_file, pick_file):
    """
    Take location files (hypoinverse formatted) and picks (format TBD)
    and creates a single obspy catalog for later use and dissemination.

    :param loc_file: File path
    :param pick_file: File path
    :return: obspy.core.Catalog
    """
    # Read/parse location file and create Events for each
    surf_cat = Catalog()
    # Parse the pick file to a dictionary
    pick_dict = parse_picks(pick_file)
    with open(loc_file, 'r') as f:
        next(f)
        for ln in f:
            ln = ln.strip('\n')
            line = ln.split(',')
            eid = line[0]
            if eid not in pick_dict:
                print('No picks for this location, skipping for now.')
                continue
            ot = UTCDateTime(line[1])
            hmc_east = float(line[2])
            hmc_north = float(line[3])
            hmc_elev = float(line[4])
            gap = float(line[-5])
            rms = float(line[-3])
            errXY = float(line[-2])
            errZ = float(line[-1])
            converter = SURF_converter()
            lon, lat, elev = converter.to_lonlat((hmc_east, hmc_north,
                                                  hmc_elev))
            o = Origin(time=ot, longitude=lon, latitude=lat, depth=130 - elev)
            o.origin_uncertainty = OriginUncertainty()
            o.quality = OriginQuality()
            ou = o.origin_uncertainty
            oq = o.quality
            ou.horizontal_uncertainty = errXY * 1e3
            ou.preferred_description = "horizontal uncertainty"
            o.depth_errors.uncertainty = errZ * 1e3
            oq.standard_error = rms
            oq.azimuthal_gap = gap
            extra = AttribDict({
                'hmc_east': {
                    'value': hmc_east,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_north': {
                    'value': hmc_north,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_elev': {
                    'value': hmc_elev,
                    'namespace': 'smi:local/hmc'
                }
            })
            o.extra = extra
            rid = ResourceIdentifier(id=ot.strftime('%Y%m%d%H%M%S%f'))
            # Dummy magnitude of 1. for all events until further notice
            mag = Magnitude(mag=1., mag_errors=QuantityError(uncertainty=1.))
            ev = Event(origins=[o], magnitudes=[mag],
                       picks=pick_dict[eid], resource_id=rid)
            surf_cat.append(ev)
    return surf_cat

def parse_picks(pick_file):
    """
    Helper for parsing file with pick information

    :param pick_file: Path to the file
    :return: Dictionary of pick info: {eid: {sta.chan: list of picks}}
    """
    pick_dict = {}
    with open(pick_file, 'r') as f:
        next(f)
        for ln in f:
            ln = ln.strip('\n')
            line = ln.split(',')
            eid = line[0]
            time = UTCDateTime(line[-3])
            phase = line[-2]
            # Placing these per standard convention but not true!!
            # TODO Maybe these should all be Z? Careful when doing correlations
            if phase == 'P':
                chan = 'XNZ'
            else:
                chan = 'XNX'
            snr = float(line[-1])
            if snr == 0.:
                method = 'manual'
            else:
                method = 'phasepapy'
            wf_id = WaveformStreamID(network_code='SV', station_code=line[2],
                                     location_code='', channel_code=chan)
            pk = Pick(time=time, method=method, waveform_id=wf_id,
                      phase_hint=phase)
            if eid not in pick_dict:
                pick_dict[eid] = [pk]
            else:
                pick_dict[eid].append(pk)
    return pick_dict


def obspyck_from_local(inv_path, wav_dir, catalog):
    """
    Function to take local catalog, inventory and waveforms for picking.

    This has been gutted from scripts.python.workflow.obspyck_util for use
    with SURF/FS-B networks.

    :param inv: Station inventory
    :param wav_dir: Directory of mseeds named according to timestamp
        eid convention
    :param catalog: catalog of events to pick
    :return:
    """

    # Grab all stationxml files
    inv_files = [inv_path]
    inv = read_inventory(inv_path)
    all_wavs = glob('{}/*'.format(wav_dir))
    # Sort events, although they should already be sorted and it doesnt matter
    catalog.events.sort(key=lambda x: x.origins[-1].time)
    if len(catalog) == 0:
        print('No events in catalog')
        return
    eids = [ev.resource_id.id.split('/')[-1] for ev in catalog]
    wav_files = [p for p in all_wavs if p.split('/')[-1].split('_')[0] in eids]
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    for ev in catalog:
        o = ev.origins[0]
        eid = ev.resource_id.id.split('/')[-1]
        wav_file = [f for f in wav_files if f.split('/')[-1].split('_')[0]
                    == eid]
        # Create temporary mseed without the superfluous non-seis traces
        st = read(wav_file[0])
        rms = [tr for tr in st
               if tr.stats.station in ['CMon', 'CTrig', 'CEnc', 'PPS']]
        for rm in rms:
            st.traces.remove(rm)
        # Rotate to ZNE not in obspyck so do it here.
        rotated_st = Stream()
        # Loop each station in inv and append to new st
        for sta in inv[0]:
            sta_st = st.select(station=sta.code)
            if len(sta_st) < 3 and len(sta_st) > 0:
                # Ignore hydrophones here
                rotated_st += sta_st
                continue
            elif len(sta_st) == 0:
                continue
            data1 = sta_st.select(channel='*Z')[0]
            dip1 = sta.select(channel='*Z')[0].dip
            az1 = sta.select(channel='*Z')[0].azimuth
            data2 = sta_st.select(channel='*X')[0]
            dip2 = sta.select(channel='*X')[0].dip
            az2 = sta.select(channel='*X')[0].azimuth
            data3 = sta_st.select(channel='*Y')[0]
            dip3 = sta.select(channel='*Y')[0].dip
            az3 = sta.select(channel='*Y')[0].azimuth
            rot_np = rotate2zne(data_1=data1.data, azimuth_1=az1, dip_1=dip1,
                                data_2=data2.data, azimuth_2=az2, dip_2=dip2,
                                data_3=data3.data, azimuth_3=az3, dip_3=dip3,
                                inverse=False)
            # Check that traces are indeed different from before
            print(not np.all(data1.data == rot_np[0]))
            print(not np.all(data2.data == rot_np[1]))
            print(not np.all(data3.data == rot_np[2]))
            # Reassemble rotated stream
            # TODO Without renaming XYZ, just assuming user understands
            # TODO that X is North, Y is East....fix this by adding channels
            # TODO to inventory later!
            new_trZ = Trace(data=rot_np[0], header=data1.stats)
            # new_trZ.stats.channel = 'XNZ'
            new_trN = Trace(data=rot_np[1], header=data2.stats)
            # new_trN.stats.channel = 'XNN'
            new_trE = Trace(data=rot_np[2], header=data3.stats)
            # new_trE.stats.channel = 'XNE'
            rot_st = Stream(traces=[new_trZ, new_trN, new_trE])
            rotated_st += rot_st
        tmp_wav_file = ['tmp/tmp_wav.mseed']
        rotated_st.write(tmp_wav_file[0], format="MSEED")
        # If not pick uncertainties, assign some arbitrary ones
        for pk in ev.picks:
            if not pk.time_errors:
                pk.time_errors.uncertainty = 0.0001
        tmp_name = 'tmp/%s' % str(ev.resource_id).split('/')[-1]
        ev.write(tmp_name, format='QUAKEML')
        print('Launching obspyck for ev: {}' .format(
              str(ev.resource_id).split('/')[-1]))
        input_file = '/home/chet/obspyck/hoppch_surf.obspyckrc17'
        root = ['obspyck -c {} -t {} -d 0.01 -s SV --event {}'.format(
            input_file, str(o.time - 0.0002), tmp_name)]
        cmd = ' '.join(root + tmp_wav_file + inv_files)
        print(cmd)
        call(cmd, shell=True)
    return