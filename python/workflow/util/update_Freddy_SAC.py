#!/usr/bin/env python

from __future__ import division

from glob import glob
from obspy import read, Stream


def update_freddy_SAC(cat, wav_dirs):
    """
    Take a directory of SAC files and update them with picks from an xml

    :param cat: Catalog of events used to update the SAC headers
    :param wav_dirs: Path to directory
    :return:
    """

    # Sort events by time to aid in counting any that occur in same minute
    cat.events.sort(key=lambda x: x.origins[-1].time)
    cnt = 0 # Silly counter for event in same minute...
    for event in cat:
        rest_o = event.origins[-1]
        dir_srch = '{}.{}.{}.{}*'.format(rest_o.time.year, rest_o.time.jday,
                                           rest_o.time.hour,
                                           rest_o.time.minute)
        # Find any directories matching pattern
        min_dirs = glob('{}/{}'.format(wav_dirs, dir_srch))
        min_dirs.sort()
        st = Stream()
        if len(min_dirs) == 0:
            print('No directories match. Moving on')
            continue
        elif len(min_dirs) == 1:
            f_names = glob('{}/*'.format(min_dirs[0]))
            st1 = read('{}/*'.format(min_dirs[0]))
            picks_2_header(st1, event, f_names)
        else:
            # Here use a crude counter to keep track of which directory to
            # use of the multiple for this minute. If we're at the final one,
            # reset cnt
            f_names = glob('{}/*'.format(min_dirs[cnt]))
            st1 = read('{}/*'.format(min_dirs[cnt]))
            picks_2_header(st1, event, f_names)
            cnt += 1
            if cnt == len(min_dirs):
                cnt = 0
    return

def picks_2_header(st1, event, f_names):
    """
    For each pick, find the Trace in st1, populate the header and overwrite
    the original file.

    :param st1:
    :param event:
    :param f_names:
    :return:
    """

    for pick in event.picks:
        pk_sta = pick.waveform_id.station_code
        pk_chan = pick.waveform_id.channel_code
        # Grab just this sta/chan
        chan_wav = st1.select(station=pk_sta, channel=pk_chan)[0]
        # Copy it out of the way
        work_tr = chan_wav.copy()
        print('Populating SAC header for {}.{}'.format(work_tr.stats.station,
                                                       work_tr.stats.channel))
        # Assign the pick time and type
        if pick.phase_hint == 'P':
            print('Writing pick to "a" header')
            work_tr.stats.sac['a'] = pick.time - work_tr.stats.starttime
            work_tr.stats.sac['ka'] = pick.phase_hint
        elif pick.phase_hint == 'S':
            work_tr.stats.sac['t0'] = pick.time - work_tr.stats.starttime
            work_tr.stats.sac['kt0'] = pick.phase_hint
        # Find the original filename which corresponds to this sta/chan
        filename = [nm for nm in f_names if nm.split('.')[-2] == pk_sta
                    and nm.split('.')[-1] == pk_chan][0]
        print('Writing file {}'.format(filename))
        work_tr.write(filename, format="SAC")
    return