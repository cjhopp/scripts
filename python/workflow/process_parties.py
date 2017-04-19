#!/usr/bin/python
from __future__ import division


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def grab_day_wavs(wav_dirs, dto, stachans):
    # Helper to recursively crawl paths searching for waveforms for a dict of
    # stachans for one day
    import os
    import fnmatch
    from itertools import chain
    from obspy import read, Stream

    st = Stream()
    wav_files = []
    for path, dirs, files in chain.from_iterable(os.walk(path)
                                                 for path in wav_dirs):
        print('Looking in %s' % path)
        for sta, chans in iter(stachans.items()):
            for chan in chans:
                for filename in fnmatch.filter(files,
                                               '*.%s.*.%s*%d.%03d'
                                                       % (
                                               sta, chan, dto.year,
                                               dto.julday)):
                    wav_files.append(os.path.join(path, filename))
    print('Reading into memory')
    for wav in wav_files:
        st += read(wav)
    return st

def lag_calc_daylong(wav_dir, party, start, end):
    """
    Essentially just a day loop to grab the day's waveforms and the day's
    party and then perform the lag calc
    :param wav_dir:
    :param party:
    :return:
    """
    import datetime
    from obspy import UTCDateTime

    cat_start = datetime.datetime.strptime(start, '%d/%m/%Y')
    cat_end = datetime.datetime.strptime(end, '%d/%m/%Y')
    for date in date_generator(cat_start, cat_end):
        # Find waveforms and find pertinent party
        dto = UTCDateTime(date)

    return