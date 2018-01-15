#!/usr/bin/env python

"""
Taking stefans csv files and putting the picks into the catalog
"""
from obspy import UTCDateTime
from glob import glob
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


def assign_stefan_picks(cat, name_map, pk_file, uncert_cutoff, temps=False,
                        temp_sac_dir=False):
    """
    Take output from Stefans Spicker and add to catalog (in place)
    :param cat:
    :param name_map:
    :param pk_file:
    :param uncert_cutoff:
    :return:
    """

    boreholes = ['NS12', 'NS13', 'NS14', 'THQ2'] # For channel naming
    alph = make_alph()
    picks = make_pk_dict(name_map, pk_file)
    if temps and temp_sac_dir:
        self_names = [nm.split('/')[-1] for nm in
                      glob('{}/*'.format(temp_sac_dir))]
        temp_map = {ResourceIdentifier(
            'smi:de.erdbeben-in-bayern/event/{}'.format(nm.split('_')[0])):
            ResourceIdentifier('smi:local/{}'.format(nm)) for nm in self_names}
    for ev in cat:
        print('For ev: %s' % str(ev.resource_id))
        if temps and temp_sac_dir:
            if ev.resource_id in temp_map:
                id = temp_map[ev.resource_id]
            else:
                print('Event not in SAC directory')
                continue
        else:
            id = ev.resource_id
        print(id)
        if id in picks:
            for pk in picks[id]:
                # Build the datetime from the time string...
                o_time = ev.preferred_origin().time
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
                pk_time = UTCDateTime(year=o_time.year, month=o_time.month,
                                      day=o_time.day, hour=hour, minute=minute,
                                      second=second, microsecond=microsecond)
                if pk['sta'][0] == 'N' or pk['sta'][0] == 'R':
                    wv_id = WaveformStreamID(station_code=sta_nm,
                                             channel_code=chan_nm)
                else:
                    wv_id = WaveformStreamID(station_code=pk['sta'],
                                             channel_code=chan_nm)
                if float(pk['error']) < uncert_cutoff:
                    uncert = QuantityError(uncertainty=float(pk['error']))
                    pk = Pick(time=pk_time, waveform_id=wv_id, phase_hint='S',
                              time_errors=uncert)
                    ev.picks.append(pk)
        else:
            print('id not in picks')
    return cat