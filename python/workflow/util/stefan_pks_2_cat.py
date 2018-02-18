#!/usr/bin/env python

"""
Taking stefans csv files and putting the picks into the catalog
"""
from glob import glob
from obspy import UTCDateTime, read
from obspy.core.event import ResourceIdentifier, Pick, WaveformStreamID
from obspy.core.event.base import QuantityError


def make_alph():
    # Build alphabet mapping dict
    import string
    alph = {}
    for x, y in zip(string.ascii_uppercase, range(0, 26)):
        alph[x] = y
    return alph

def make_pk_dict(pk_file, name_map=None):
    """
    Make dictionary of all picks in the provided sPickerC output file

    If name_map provided, will map sPickerC event names to actual names
    expected in the catalog

    :param pk_file:
    :param name_map:
    :return:
    """
    # Mapping of event new_name --> old name from stefan txt file
    names = {}
    if name_map:
        with open(name_map, 'r') as f:
            f_str = [line for line in f]
            for ln in f_str:
                ln = ln.rstrip('\n').split(',')
                if ln[1] not in names:
                    names[ln[1]] = ln[0]
                else:
                    print(ln[0] + ' used miltiple times?')
                    print(ln[0], names[ln[0]])
                    print(ln)
                    continue
    # Now make a dictionary of all the Spicks keyed to event rid
    picks = {}
    with open(pk_file, 'r') as f2:
        next(f2)
        pk_str = [line for line in f2]
        for ln2 in pk_str:
            ln2 = ln2.rstrip('\n').split(',')
            if len(ln2) < 2:
                continue
            if ln2[1] == 'EVENT':
                continue
            if name_map:
                if ln2[1].split('/')[-1] in names:
                    rid = ResourceIdentifier('smi:local/{}'.format(
                        names[ln2[1].split('/')[-1]]))
                else:
                    print('Wonky event name in pick file. Skipping.')
                    continue
            else:
                rid = ResourceIdentifier('smi:local/{}'.format(
                    ln2[1].split('/')[-1]))
            try:
                if rid not in picks:
                    picks[rid] = [{'time': ln2[-6],
                                   'error': float(ln2[-3]) / 1000.,
                                   'sta': ln2[1].split('/')[0]}]
                else:
                    picks[rid].append({'time': ln2[-6],
                                       'error': float(ln2[-3]) / 1000.,
                                       'sta': ln2[1].split('/')[0]})
            except ValueError as e:
                print(e)
    return picks


def assign_stefan_picks(cat, pk_file, uncert_cutoff, name_map=None,
                        temps=False, temp_sac_dir=False):
    """
    Take output from Stefans Spicker and add to catalog (in place)
    :param cat: Catalog which we want to populate with S-picks
    :param pk_file: File including all of the S-picks
    :param uncert_cutoff: Cutoff for the pick error in seconds
    :param name_map: In the case of detections, we need to map new eids back
        to the original based on this file provided by stefan
    :param temps: Whether or not we are using template resource_id and not
        detection resource_id
    :param temp_sac_dir: Directory of self detections for templates. This is
        so that we can map the self_detection name (which may be the basis
        for rids in a catalog) to the basic template name.
    :return:
    """

    boreholes = ['NS12', 'NS13', 'NS14', 'THQ2'] # For channel naming
    alph = make_alph()
    picks = make_pk_dict(pk_file, name_map)
    if temps and temp_sac_dir:
        self_names = [nm.split('/')[-1] for nm in
                      glob('{}/*'.format(temp_sac_dir))]
        temp_map = {ResourceIdentifier('smi:local/{}'.format(nm)):
                    ResourceIdentifier('smi:local/{}'.format(nm.split('_')[0]))
                    for nm in self_names}
    for ev in cat:
        print('For ev: %s' % str(ev.resource_id))
        eid = str(ev.resource_id).split('/')[-1]
        ev.resource_id = ResourceIdentifier('smi:local/{}'.format(eid))
        if temps and temp_sac_dir:
            if ev.resource_id in temp_map:
                print(ev.resource_id)
                id = temp_map[ev.resource_id]
            else:
                print('Event not in SAC directory')
                continue
        else:
            id = ResourceIdentifier('smi:local/{}'.format(eid))
        print(id)
        if id in picks:
            for pk in picks[id]:
                # Build the datetime from the time string...
                pk_date = ev.picks[-1].time
                hour = int(pk['time'].split(':')[0])
                minute = int(pk['time'].split(':')[1])
                second = int(pk['time'].split(':')[2].split('.')[0])
                sta_nm = '{}{}{}'.format(pk['sta'][:2],
                                         str(alph[pk['sta'][2]]),
                                         str(alph[pk['sta'][3]]))
                if sta_nm in boreholes:
                    chan_nm = 'EH1'
                else:
                    chan_nm = 'EHE'
                if len(pk['time'].split(':')[2].split('.')) == 1:
                    microsecond = 0
                else:
                    microsecond = int(
                        pk['time'].split(':')[2].split('.')[1]) * 1000
                pk_time = UTCDateTime(year=pk_date.year, month=pk_date.month,
                                      day=pk_date.day, hour=hour, minute=minute,
                                      second=second, microsecond=microsecond)
                if pk['sta'][0] == 'N' or pk['sta'][0] == 'R':
                    wv_id = WaveformStreamID(network_code='NZ',
                                             station_code=sta_nm,
                                             channel_code=chan_nm)
                else:
                    wv_id = WaveformStreamID(network_code='NZ',
                                             station_code=pk['sta'],
                                             channel_code=chan_nm)
                if float(pk['error']) < uncert_cutoff:
                    uncert = QuantityError(uncertainty=float(pk['error']))
                    pk = Pick(time=pk_time, waveform_id=wv_id, phase_hint='S',
                              time_errors=uncert)
                    ev.picks.append(pk)
        else:
            print('id not in picks')
    return cat

def cat_2_sac_dir(cat, sac_dir, dry_run=True):
    """
    Take a catalog with S-picks from Stefan and add them to the headers of
    the SAC files in the given directory.
    :param cat: Catalog with picks which we want to add to SAC files
    :param sac_dir: Directory of SAC files which we wrote for stefan which we
        now want to add S-picks to.

    .. NOTE:: This assumes a catalog with full eids for self_detections
    :return:
    """
    sac_names = glob('{}/*'.format(sac_dir))
    for ev in cat:
        eid = str(ev.resource_id).split('/')[-1]
        # Find directory for this event, if it exists
        ev_sacs = []
        for sac in sac_names:
            if sac.split('/')[-1] == eid:
                ev_sacs.extend(glob('{}/*'.format(sac)))
        if len(ev_sacs) == 0:
            print('No events in directory match this event.')
            continue
        # Loop over picks, find S-picks, and then add them to the appropriate
        # SAC file headers
        for pk in ev.picks:
            if pk.phase_hint == 'S':
                sta = pk.waveform_id.station_code
                chan = pk.waveform_id.channel_code
                tr_sac = [s for s in ev_sacs
                          if s.split('_')[-2] == sta and
                          s.split('_')[-1].rstrip('.sac') == chan]
                if len(tr_sac) == 0:
                    print('No sac files exist for this sta/chan')
                    continue
                # Read in the data and add the s-pick to the header
                tr = read(tr_sac[0])[0]
                tr.stats.sac['t0'] = pk.time - tr.stats.starttime
                tr.stats.sac['kt0'] = 'S'
                if dry_run:
                    print('Would write file to: {}'.format(tr_sac[0]))
                    print('SAC header for new trace:\n{}'.format(tr.stats.sac))
                else:
                    print('Writing file: {}'.format(tr_sac[0]))
                    tr.write(tr_sac[0], format='MSEED')
    return