#!/usr/bin/env python

"""
Taking stefans csv files and putting the picks into the catalog
"""
from obspy import UTCDateTime
from obspy.core.event import ResourceIdentifier, Pick, WaveformStreamID
from obspy.core.event.base import QuantityError


def make_alph():
    # Build alphabet mapping dict
    import string
    alph = {}
    for x, y in zip(string.ascii_uppercase, range(0, 26)):
        alph[x] = y
    return alph

def make_pk_dict(name_map, pk_file):
    # Mapping of event new_name --> old name from stefan txt file
    names = {}
    with open(name_map, 'rb') as f:
        f_str = [line for line in f]
        for ln in f_str:
            ln.split(',')
            if ln[0] not in names:
                names[ln[0]] = ln[1]
            else:
                print(line[0] + ' used miltiple times?')
                print(line[0], names[line[0]])
                print(line)
                continue
    # Now make a dictionary of all the Spicks keyed to event rid
    picks = {}
    with open(pk_file, 'rb') as f2:
        pk_str = [line for line in f2[1:]]
        for ln2 in pk_str:
            ln2.split(',')
            rid = ResourceIdentifier('smi:local/{}'.format(
                names[ln2[1].split('/')[-1]]))
            if rid not in picks:
                picks[rid] = [{'time': ln2[-6], 'error': int(ln2[-3]) / 1000.,
                               'sta': ln2[0].split('/')[0]}]
            else:
                picks[rid].append({'time': ln2[-6],
                                   'error': int(ln2[-3]) / 1000,
                                   'sta': ln2[1].split('/')[0]})
    return picks


def assign_stefan_picks(cat, name_map, pk_file, uncert_cutoff):
    """
    Take output from Stefans Spicker and add to catalog
    :param cat:
    :param name_map:
    :param pk_file:
    :param uncert_cutoff:
    :return:
    """

    boreholes = ['NS12', 'NS13', 'NS14', 'THQ2'] # For channel naming
    alph = make_alph()
    picks = make_pk_dict(name_map, pk_file)
    for ev in cat:
        print('For ev: %s' % str(ev.resource_id))
        if ev.resource_id in picks:
            for pk in picks[ev.resource_id]:
                # (Sigh) Build the datetime from the time string...
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
    return cat