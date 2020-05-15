#!/usr/bin/python

"""
Set of functions wrapping obspy triggering and phasepapy picking/association
"""

import yaml

from glob import glob
from obspy import UTCDateTime, read, Stream
from obspy.signal.trigger import coincidence_trigger


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def trigger(param_file):
    """
    Wrapper on obspy coincidence trigger for a directory of waveforms

    :param param_file: Path to a yaml with the necessary parameters

    :return:
    """
    with open(param_file, 'r') as f:
        paramz = yaml.load(f)
    trig_p = paramz['Trigger']
    pick_p = paramz['Picker']
    trigs = []
    start = UTCDateTime(trig_p['start_time']).datetime
    end = UTCDateTime(trig_p['end_time']).datetime
    for date in date_generator(start, end):
        jday = UTCDateTime(date).julday
        day_wavs = glob('{}/**/*{}.ms'.format(
            paramz['Trigger']['raw_wav_dir'], jday))
        st = Stream()
        for w in day_wavs:
            # TODO Do some checks for continuity, gaps, etc...
            st += read(w)
        trigs += coincidence_trigger(
            trigger_type='recstalta',
            stream=st, thr_on=trig_p['threshold_on'],
            thr_off=trig_p['threshold_off'],
            thr_coincidence_sum=trig_p['coincidence_sum'],
            sta=trig_p['sta'], lta=trig_p['lta'], details=True)
    return trigs