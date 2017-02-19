#!/usr/bin/env python

"""
Taking stefans csv files and putting the picks into the catalog
"""
import csv
from obspy import read_events, UTCDateTime
from obspy.core.event import ResourceIdentifier, Pick, WaveformStreamID, Event
from obspy.core.event.base import QuantityError

def stefan_pk_dicts():
    # Read in his mapping of event new_name --> old name
    names = {}
    with open('/home/chet/data/mrp_data/stefan_spicks/try_2/eventlist.txt', 'rb') as f:
        reader = csv.reader(f)
        for line in reader:
            if line[0] not in names:
                names[line[0]] = line[1]
            else:
                print(line[0] + ' used miltiple times!')
                print(line[0], names[line[0]])
                print(line)
                continue

    picks = {}
    with open('/home/chet/data/mrp_data/stefan_spicks/try_2/stefan_picks_all.csv', 'rb') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row[0]) > 0:
                if row[0][:2] in ['RT', 'NS', 'AR', 'HR', 'PR', 'AL']:
                    if len(row[-4]) > 0:
                        if ResourceIdentifier('smi:local/' + names[row[0].split('/')[-1]]) not in picks:
                            picks[ResourceIdentifier('smi:local/' + names[row[0].split('/')[-1]])] = [{'time': row[-7],
                                                                   'error': int(row[-4]) / 1000.00,
                                                                   'sta': row[0].split('/')[0]}]
                        else:
                            picks[ResourceIdentifier('smi:local/' + names[row[0].split('/')[-1]])].append({'time': row[-7],
                                                                                        'error': int(row[-4]) / 1000.00,
                                                                                        'sta': row[0].split('/')[0]})
    return picks


def make_alph():
    # Add the info to the catalog
    # Build alphabet mapping dict
    import string
    alph = {}
    for x, y in zip(string.ascii_uppercase, range(0, 26)):
        alph[x] = y
    return alph

def assign_stefan_picks(cat, uncert_cutoff):
    alph = make_alph()
    picks = stefan_pk_dicts()
    for ev in cat:
        print('For ev: %s' % str(ev.resource_id))
        if ev.resource_id in picks:
            for pk in picks[ev.resource_id]:
                # (Sigh) Build the datetime from the time string...
                # XXX FUCKED THIS UP IN ORIGINAL. NEED TO ASSIGN UTCDateTime(kwargs)...
                #     ...instead of assigning utc_dto.hour = 12, etc...

                # FURTHER NOTE: Many picks have errors of 0.50 seconds which need to be removed
                o_time = ev.preferred_origin().time
                hour = int(pk['time'].split(':')[0])
                minute = int(pk['time'].split(':')[1])
                second = int(pk['time'].split(':')[2].split('.')[0])
                if len(pk['time'].split(':')[2].split('.')) == 1:
                    microsecond = 0
                else:
                    microsecond = int(pk['time'].split(':')[2].split('.')[1]) * 1000
                pk_time = UTCDateTime(year=o_time.year, month=o_time.month, day=o_time.day,
                                      hour=hour, minute=minute, second=second, microsecond=microsecond)
                if pk['sta'][0] == 'N' or pk['sta'][0] == 'R':
                    wv_id = WaveformStreamID(station_code=pk['sta'][:2] + str(alph[pk['sta'][2]]) +
                                                          str(alph[pk['sta'][3]]), channel_code='EHE')
                else:
                    wv_id = WaveformStreamID(station_code=pk['sta'], channel_code='EHE')
                if float(pk['error']) < float(uncert_cutoff):
                    uncert = QuantityError(uncertainty=float(pk['error']))
                    pk = Pick(time=pk_time, waveform_id=wv_id, phase_hint='S', time_errors=uncert)
                    ev.picks.append(pk)
    return cat

def remove_s_picks(cat):
    for ev in cat:
        for pk in list(ev.picks):
            if pk.phase_hint == 'S':
                ev.picks.remove(pk)
    return cat