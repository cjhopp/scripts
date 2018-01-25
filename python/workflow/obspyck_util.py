#!/usr/bin/env python

"""
Wrapper functions for obspyck and seishub
"""
import os
import fnmatch
import datetime
import shutil
import numpy as np

from obspy import read, UTCDateTime
from subprocess import call
from glob import glob
from itertools import chain


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def pick_event(event):
    """
    Pull wavs from around event from seishub
    :param event: obspy.core.Event
    :return:
    """

    #Write event to temporary qml to feed to obspyck
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    tmp_name = 'tmp/%s' % str(event.resource_id).split('/')[-1]
    event.write(tmp_name, format='QUAKEML')
    o_time = event.origins[-1].time
    input_file = '/home/chet/obspyck/hoppch.obspyckrc17'
    call('obspyck -c %s -t %s -s NS --event %s' % (input_file,
                                                   str(o_time - 5),
                                                   tmp_name),
         shell=True)
    shutil.rmtree('tmp')
    return

def pick_catalog(catalog, rand_choice=False, ev_ids=False):
    """
    Pick all events in catalog in a loop. Continues after closing obspyck?
    :param catalog: obspy.core.Catalog
    :param ev_ids: list of event identifier strings to match
    :return:
    """

    if rand_choice:
        inds = np.random.choice(range(len(catalog)),
                                rand_choice, replace=False)
        catalog = [catalog[i] for i in inds]
    if ev_ids:
        for ev in catalog:
            if str(ev.resource_id).split('/')[-1] in ev_ids:
                pick_event(ev)
    else:
        for ev in catalog:
            print('Launching obspyck for ev: %s' %
                  str(ev.resource_id).split('/')[-1])
            pick_event(ev)
    return

def obspyck_from_local(inv_dir, wav_dirs=None, wav_file=None, catalog=None,
                       wav_format='mseed', utcdto=None, start=False,
                       end=False):
    """
    Function to take local files instead of seishub inv/wavs
    :return:
    """

    # Grab all stationxml files
    inv_files = glob(inv_dir)
    # Work out what the args are telling us to do
    if not catalog:
        if not utcdto:
            utcdto = read(wav_file)[0].stats.starttime
            # msg = 'Without a catalog you need to specify a datetime'
            # raise Exception(msg)
        cat_start = utcdto.date
        cat_end = utcdto.date
    else:
        if len(catalog[0].origins) > 0:
            catalog.events.sort(key=lambda x: x.origins[-1].time)
        else:
            catalog.events.sort(key=lambda x: x.picks[0].time)
        if start:
            cat_start = datetime.datetime.strptime(start, '%d/%m/%Y')
            cat_end = datetime.datetime.strptime(end, '%d/%m/%Y')
        else:
            cat_start = catalog[0].origins[-1].time.date
            cat_end = catalog[-1].origins[-1].time.date
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        if catalog:
            print('Running for date: %s' % str(dto))
            sch_str_start = 'time >= %s' % str(dto)
            sch_str_end = 'time <= %s' % str(dto + 86400)
            tmp_cat = catalog.filter(sch_str_start, sch_str_end)
            if len(tmp_cat) == 0:
                print('No events on: %s' % str(dto))
                continue
            else:
                print('%d events for this day' % len(tmp_cat))
            stas = list(set([pk.waveform_id.station_code for ev in tmp_cat
                             for pk in ev.picks]))
        else:
            stas = ['ALRZ','ARAZ','HRRZ','NS01','NS02','NS03','NS04','NS05',
                    'NS06','NS07','NS08','NS09','NS10','NS11','NS12','NS13',
                    'NS14','NS15','NS16','NS18','PRRZ','RT01','RT02','RT03',
                    'RT05','RT06','RT07','RT08','RT09','RT10','RT11','RT12',
                    'RT13','RT14','RT15','RT16','RT17','RT18','RT19','RT20',
                    'RT21','RT22','RT23','THQ2','WPRZ']
        # Grab day's wav files
        if wav_format == 'mseed' and not wav_file:
            wav_ds = ['%s/%d' % (d, dto.year) for d in wav_dirs]
            wav_files = []
            for path, dirs, files in chain.from_iterable(os.walk(path)
                                                         for path in wav_ds):
                print('Looking in %s' % path)
                for sta in stas:
                    for filename in fnmatch.filter(files,
                                                   '*.%s.*%d.%03d'
                                                           % (sta, dto.year,
                                                              dto.julday)):
                        wav_files.append(os.path.join(path, filename))
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        if catalog:
            for ev in tmp_cat:
                # If getting from SAC directory, grab wavs
                if wav_format == 'SAC' and not wav_file:
                    sac_dirs = glob(wav_dirs[0] + '/*')
                    try:
                        wav_files = [
                            glob('{}/*'.format(dir)) for dir
                            in sac_dirs if dir.split('/')[-1].split('_')[0] ==
                            str(ev.resource_id).split('/')[-1]][0]
                    except IndexError:
                        try:
                            wav_files = [
                                glob('{}/*'.format(dir)) for dir
                                in sac_dirs if
                                dir.split('/')[-1].split('_')[0] ==
                                str(ev.resource_id).split('/')[-1].split('_')[0]][0]
                        except IndexError:
                            print('No SAC file for this event.')
                            continue
                elif wav_file:
                    wav_files = [wav_file]
                # First, remove amplitudes and station mags not set with obspyck
                rm_amps = []
                rm_sta_mags = []
                for amp in ev.amplitudes:
                    if "/obspyck/" not in str(amp.method_id) or str(
                        amp.method_id).endswith("/obspyck/1"):
                        rm_amps.append(amp)
                for ampl in rm_amps:
                    ev.amplitudes.remove(ampl)
                tmp_name = 'tmp/%s' % str(ev.resource_id).split('/')[-1]
                ev.write(tmp_name, format='QUAKEML')
                if len(ev.origins) > 0:
                    utc_dt = ev.origins[-1].time
                else:
                    utc_dt = ev.picks[0].time
                print('Finding waveform files')
                # Limit wav_dirs
                print('Launching obspyck for ev: %s' %
                      str(ev.resource_id).split('/')[-1])
                input_file = '/home/chet/obspyck/hoppch_local.obspyckrc17'
                root = ['obspyck -c %s -t %s -d 360 -s NS --event %s' % (input_file,
                                                                         str(utc_dt - 20),
                                                                         tmp_name)]
                cmd = ' '.join(root + wav_files + inv_files)
                call(cmd, shell=True)
        else:
            print('Launching obspyck:')
            input_file = '/home/chet/obspyck/hoppch_local.obspyckrc17'
            root = ['obspyck -c {} -t {} -d 360 -s NS'.format(
                input_file, str(utcdto - 20))]
            if not wav_file and wav_files:
                cmd = ' '.join(root + wav_files + inv_files)
            else:
                cmd = ' '.join(root + [wav_file] + inv_files)
            call(cmd, shell=True)
    return

