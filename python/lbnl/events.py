#!/usr/bin/python

"""
Functions for the processing, reading, converting of events and event files

IMPORTANT
***********************************************
Arbitrary zero depth point is elev = 130 m
***********************************************
"""

import os
import shutil

import numpy as np

from glob import glob
from subprocess import call
from obspy import UTCDateTime, Catalog, read, read_inventory, Stream, Trace
from obspy.core.util import AttribDict
from obspy.core.event import Pick, Origin, Arrival, Event, Magnitude,\
    WaveformStreamID, ResourceIdentifier, OriginQuality, OriginUncertainty,\
    QuantityError
from lbnl.coordinates import SURF_converter
from lbnl.waveforms import rotate_channels

three_comps = ['OB13', 'OB15', 'OT16', 'OT18', 'PDB3', 'PDB4', 'PDB6', 'PDT1',
               'PSB7', 'PSB9', 'PST10', 'PST12']

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
                },
                'hmc_eid': {
                    'value': eid,
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
            if line[2] not in three_comps: # Hydrophone channels
                chan = 'XN1'
            elif phase == 'P':
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

def add_pols_to_Time2EQ_hyp(catalog, nlloc_dir, outdir, hydrophones=False):
    """
    Add polarities to the nlloc hyp files produced from Time2EQ. This is the
    last part of the workflow which takes hypoDD locations, retraces the
    raypaths with Time2EQ, relocates these with NLLoc and then repopulates
    the PHASE lines in the .hyp file with the polarities picked in Obspyck
    (this function). These are then fed into the Arnold focmec stuff.

    :param catalog: Catalog with polarity picks to use
    :param nlloc_dir: Path to the NLLoc loc/ directory with corresponding
        location files for the catalog provided
    :param outdir: Path to output directory for the .scat, .hdr and .hyp files
        Usually this will be in an Arnold_Townend projects/ directory
    :param hydrophones: Whether to include polarities measured on hydrophones.
        Defaults to False as I'm not sure how to handle these yet.
    :return:
    """
    for ev in catalog:
        print('{}'.format(str(ev.resource_id).split('/')[-1]))
        nlloc_fs = glob('{}/{}*'.format(
            nlloc_dir,
            str(ev.resource_id).split('/')[-1].split('_')[0]))
        try:
            hyp_path = [path for path in nlloc_fs
                        if path.endswith('.hyp')
                        and 'sum' not in path.split('.')][0]
        except IndexError as msg:
            print('No NLLoc location for this event. Probably low SNR?')
            continue
        print(hyp_path)
        # Move hdr and scat files to outdir
        scat_hdr = [path for path in nlloc_fs
                    if (path.endswith('.hdr')
                        or path.endswith('.scat'))
                    and 'sum' not in path.split('.')]
        for fl in scat_hdr:
            shutil.copyfile(fl, '{}/{}'.format(outdir, fl.split('/')[-1]))
        # Now edit the loc file and write it to outdir
        with open(hyp_path, 'r') as orig:
            with open('{}/{}'.format(outdir,
                                     hyp_path.split('/')[-1]), 'w') as new:
                phase = False
                for ln in orig:
                    line = ln.rstrip()
                    line = line.split()
                    if len(line) == 0:
                        print('End of file')
                        break
                    # Write top of file as usual until we get to PHASE lines
                    if line[0] == 'PHASE':
                        phase = True
                        new.write(' '.join(line) + '\n')
                        continue
                    elif line[0] == 'END_PHASE':
                        phase = False
                    if phase:
                        # Skip all the S phases
                        if line[4] == 'S':
                            print('Ignore S phases')
                            new.write(' '.join(line) + '\n')
                            continue
                        # If hydrophone == False, don't bother with those
                        if not hydrophones and line[0] not in three_comps:
                            new.write(' '.join(line) + '\n')
                            continue
                        # Try to find a corresponding polarity pick in catalog
                        # Because P and S traced to all stations, we find only
                        # phase lines corresponding to actual picks in the
                        # catalog and populate the FM column. These will be the
                        # only ones used by the Focal mech package anyways.
                        print('Try adding for {}'.format(line[0]))
                        try:
                            pk = [pk for pk in ev.picks
                                  if pk.waveform_id.station_code == line[0]
                                  and line[4] == 'P'][0]
                        except IndexError:
                            print('No polarity pick for {}'.format(line[0]))
                            new.write(' '.join(line) + '\n')
                            continue
                        if pk.polarity not in ['positive', 'negative']:
                            print('No polarity for station {}'.format(line[0]))
                            new.write(' '.join(line) + '\n')
                            continue
                        if pk.polarity == 'positive':
                            line[5] = 'U'
                        elif pk.polarity == 'negative':
                            line[5] = 'D'
                    new.write(' '.join(line) + '\n')
    return

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
        rotated_st = rotate_channels(st, inv)
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