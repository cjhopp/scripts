#!/usr/bin/python

"""
Functions for the processing, reading, converting of events and event files
"""

from obspy import UTCDateTime, Catalog
from obspy.core.util import AttribDict
from obspy.core.event import Pick, Origin, Arrival, Event, Magnitude,\
    WaveformStreamID
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
            print(line)
            eid = line[0]
            if eid not in pick_dict:
                print('No picks for this location, skipping for now.')
                continue
            ot = UTCDateTime(line[1])
            hmc_east = float(line[2])
            hmc_north = float(line[3])
            hmc_elev = float(line[4])
            converter = SURF_converter()
            lon, lat, elev = converter.to_lonlat((hmc_east, hmc_north,
                                                  hmc_elev))
            o = Origin(time=ot, longitude=lon, latitude=lat, depth=elev)
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
            # Dummy magnitude of 1. for all events until further notice
            ev = Event(origins=[o], magnitudes=[Magnitude(mag=1.)],
                       picks=pick_dict[eid])
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
