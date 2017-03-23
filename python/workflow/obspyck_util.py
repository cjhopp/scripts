#!/usr/bin/env python

"""
Wrapper functions for obspyck and seishub
"""

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


def obspyck_from_local(wav_dirs, inv_dir, catalog):
    """
    Function to take local files instead of seishub Inv/wavs
    :return:
    """
    import fnmatch
    import os
    from subprocess import call
    from glob import glob
    from itertools import chain

    # Grab all stationxml files
    inv_files = glob(inv_dir)
    catalog.events.sort(key=lambda x: x.origins[-1].time)
    for ev in catalog:
        if not os.path.isdir('tmp'):
            os.mkdir('tmp')
        tmp_name = 'tmp/%s' % str(ev.resource_id).split('/')[-1]
        ev.write(tmp_name, format='QUAKEML')
        dto = ev.origins[-1].time
        # Create stachan dict
        stachans = {pk.waveform_id.station_code: [] for pk in ev.picks}
        for pk in ev.picks:
            chan_code = pk.waveform_id.channel_code
            if chan_code not in stachans[pk.waveform_id.station_code]:
                stachans[pk.waveform_id.station_code].append(chan_code)
        wav_files = []
        print('Finding waveform files')
        # Limit wav_dirs
        wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
        for path, dirs, files in chain.from_iterable(os.walk(path)
                                                     for path in wav_ds):
            print('Looking in %s' % path)
            for sta, chans in iter(stachans.items()):
                for chan in chans:
                    for filename in fnmatch.filter(files,
                                                   '*.%s.*.%s*%d.%03d'
                                                           % (sta, chan,
                                                              dto.year,
                                                              dto.julday)):
                        wav_files.append(os.path.join(path, filename))
        print('Launching obspyck for ev: %s' %
              str(ev.resource_id).split('/')[-1])
        input_file = '/home/chet/obspyck/hoppch.obspyckrc17'
        root = ['obspyck -c %s -t %s -s NS --event %s' % (input_file,
                                                          str(dto - 20),
                                                          tmp_name)]
        cmd = ' '.join(root + wav_files + inv_files)
        call(cmd, shell=True)

