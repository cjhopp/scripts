#!/usr/bin/python

"""
Functions for the processing, reading, converting of events and event files
"""

import os

from glob import glob
from subprocess import call
from obspy import UTCDateTime, Catalog, read
from obspy.core.util import AttribDict
from obspy.core.event import Pick, Origin, Arrival, Event, Magnitude,\
    WaveformStreamID, ResourceIdentifier, OriginQuality, OriginUncertainty,\
    QuantityError
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
            o = Origin(time=ot, longitude=lon, latitude=lat, depth=elev)
            o.origin_uncertainty = OriginUncertainty()
            o.quality = OriginQuality()
            ou = o.origin_uncertainty
            oq = o.quality
            ou.horizontal_uncertainty = errXY * 1e3
            ou.preferred_description = "horizontal uncertainty"
            o.depth_errors.uncertainty = errZ * 1e3
            oq.standard_error = rms  # XXX stimmt diese Zuordnung!!!?!
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
        tmp_wav_file = ['tmp/tmp_wav.mseed']
        st.write(tmp_wav_file[0], format="MSEED")
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