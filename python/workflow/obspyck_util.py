#!/usr/bin/env python

"""
Wrapper functions for obspyck and seishub
"""
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
    import os
    import shutil
    from subprocess import call

    #Write event to temporary qml to feed to obspyck
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    tmp_name = 'tmp/%s' % str(event.resource_id).split('/')[-1]
    event.write(tmp_name, format='QUAKEML')
    o_time = event.origins[-1].time
    input_file = '/home/chet/obspyck/hoppch.obspyckrc17'
    call('obspyck -c %s -t %s -s NS --event %s' % (input_file,
                                                   str(o_time - 20),
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
    import numpy as np

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


def obspyck_from_local(wav_dirs, inv_dir, catalog, start=False, end=False):
    """
    Function to take local files instead of seishub inv/wavs
    :return:
    """
    import fnmatch
    import datetime
    import os
    from subprocess import call
    from glob import glob
    from itertools import chain
    from obspy import UTCDateTime

    # Grab all stationxml files
    inv_files = glob(inv_dir)
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
        # Grab day's wav files
        wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
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
        for ev in tmp_cat:
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
            root = ['obspyck -c %s -t %s -s NS --event %s' % (input_file,
                                                              str(utc_dt - 20),
                                                              tmp_name)]
            cmd = ' '.join(root + wav_files + inv_files)
            call(cmd, shell=True)
    return

