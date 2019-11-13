#!/usr/bin/python

"""
Set of functions wrapping Lomax java tools: SeisGram2K and Seismicity viewer
"""
import os
import subprocess
import fnmatch

from datetime import timedelta
from itertools import chain
from obspy import UTCDateTime

def date_generator(start_date, end_date):
    # Generator for date looping
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def seis_viewer_compare(ev1, ev2):
    """
    Hard-coded crap to launch seismicity viewer for two events (one with S-picks, one without)
    :param ev1:
    :param ev2:
    :return:
    """
    filename1 = '/media/chet/hdd/seismic/NZ/NLLoc/mrp/2015_Rawlinson_S_9-21/loc/%s.*.*.grid0.loc.hyp' % \
                str(ev1.resource_id).split('/')[-1]
    filename2 = '/media/chet/hdd/seismic/NZ/NLLoc/mrp/2015_Rawlinson_S_9-21/rewt_0.05_test/loc/%s.*.*.grid0.loc.hyp' % \
                str(ev2.resource_id).split('/')[-1]
    print(filename1)
    cmnd = 'java net.alomax.seismicity.Seismicity %s %s' % (filename1, filename2)
    subprocess.call(cmnd, shell=True)
    return

def seis_view_catalogs(cat1, cat2):
    for i, ev in enumerate(cat1):
        seis_viewer_compare(ev, cat2[i])
    return

def view_waveforms(wav_dirs, start, end, stations):
    """
    Wrapper on SeisGram2K 7.0 to view any number of waveforms interactively
    :param wav_dirs: waveform directories
    :param start: start UTCDateTime
    :param end: end UTCDateTime
    :param stations: list of strings defining stations to plot
    :return:
    """
    # Establish date
    start_date = start.datetime
    end_date = end.datetime
    wav_files = []
    for dto in date_generator(start_date, end_date):
        utcd = UTCDateTime(dto)
        for path, dirs, files in chain.from_iterable(os.walk(path)
                                                     for path in wav_dirs):
            print('Looking in %s' % path)
            for sta in stations:
                for filename in fnmatch.filter(files, '*.%s.*%d.%03d'
                        % (sta, utcd.year, utcd.julday)):
                    wav_files.append(os.path.join(path, filename))
    cmnd = 'java {} net.alomax.seisgram2k.SeisGram2K {}'.format(
        '-classpath /home/chet/seisgram2K/SeisGram2K70.jar',
        ' '.join(wav_files))
    print(cmnd)
    subprocess.call(cmnd, shell=True)
    return