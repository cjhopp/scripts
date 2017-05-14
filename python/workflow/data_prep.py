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

def sc3ml2qml(zipdir, outdir, stylesheet, prog='xalan'):
    """
    Converting Steve's zipped sc3ml to individual-event qml files
    :param zipdir: directory of zipped sc3ml files
    :param outdir: directory to output qml files to
    :param stylesheet: conversion stylesheet path
    :param prog: which conversion program to use. Defaults to xalan. Can
        also use xsltproc.
    :return:
    """
    import os
    import fnmatch

    raw_files = []
    # raw_dir = '/home/chet/data/mrp_data/sherburn_catalog/quake-ml/xsl_test/sc3ml_test'
    for root, dirnames, filenames in os.walk(zipdir):
        for filename in fnmatch.filter(filenames, '*.xml.zip'):
            raw_files.append(os.path.join(root, filename))
    # Running sczip from SC3
    os.chdir('/home/chet/seiscomp3/lib/')
    for afile in raw_files:
        name = afile.rstrip('.zip')
        cmd_str = ' '.join(['/home/chet/seiscomp3/bin/sczip', '-d', afile,
                            '-o', name])
        os.system(cmd_str)
        # Convert sc3ml to QuakeML
        # Put new files in separate directory
        new_name = ''.join([outdir, os.path.basename(afile).rstrip('.xml.zip'),
                            '_QML.xml'])
        if prog == 'xsltproc':
            cmd_str2 = ' '.join(['xsltproc', '-o', new_name,
                                 stylesheet, name])
        elif prog == 'xalan':
            cmd_str2 = ' '.join(['xalan', '-xsl', stylesheet, '-in', name,
                                 '-out', new_name])
        else:
            print('Invalid program type. Use xalan or xsltproc')
        os.system(cmd_str2)
    #Remove all '#' from QuakeML (shady way of circumventing validation issues)
    # qml_files = glob('/home/chet/data/mrp_data/sherburn_catalog/quake-ml/*QML.xml')
    # for one_file in qml_files:
    #     command = "sed -i 's/#//g' " + one_file
    #     os.system(command)
    return


def run_qml_with_obspy(dir, outdir):
    """
    Trying to address schema validation issues in seishub by reading qml from
    above funtion into obspy, then rewriting to new qml
    :param dir: Directory of qmls
    :param outdir: Output directory for new events
    :return:
    """
    from glob import glob
    from obspy import read_events
    qmls = glob(dir)
    for qml in qmls:
        ev_name = qml.split('/')[-1].split('.')[1].rstrip('_QML')
        ev_cat = read_events(qml)
        ev_cat.write('%s/%s.xml' % (outdir, ev_name), format='QUAKEML')


def consolidate_qmls(directory, outfile=False):
    """
    Take directory of single-event qmls from above function and consolidate
    into one, year-long Catalog.write() qml file.
    :param directory: Directory of qml files
    :param outfile: Defaults to False, else is path to new outfile
    :return: obspy.core.Catalog
    """
    from glob import glob
    from obspy import read_events, Catalog
    qmls = glob(directory)
    cat = Catalog()
    for qml in qmls:
        cat += read_events(qml)
    if outfile:
        cat.write(outfile)
    return cat


def remove_staxml_end_date(inv):
    """
    Loop through Merc inventory and remove end date of latest channel
    They appear to be incorrect anyways.
    :param inv:
    :return:
    """
    for net in inv:
        for sta in net:
            if len(sta.channels) > 3:
                # Find chan with latest end date, and remove it
                for chan in set([chan.code for chan in sta.channels]):
                    max(sta.select(channel=chan).channels,
                        lambda c: c.end_time)[0].end_date = None
    return


def make_franny_symlinks(src_dirs, out_dir):
    """
    Make symlinks for NS12, NS13, NS14 with correct component naming for
    horizontal components
    :param src_dir:
    :param out_dir:
    :return:
    """
    import os
    import fnmatch
    import string
    from itertools import chain

    for path, dirs, files in chain.from_iterable(os.walk(path)
                                                 for path in src_dirs):
        print('Looking in %s' % path)
        for sta in ['NS12', 'NS13', 'NS14']:
            for filename in fnmatch.filter(files, '*.%s*' % sta):
                net = filename.split('.')[-7]
                chan = filename.split('.')[-4]
                if chan[-1] == 'N':
                    new_chan = 'EH1'
                elif chan[-1] == 'E':
                    new_chan = 'EH2'
                else:
                    continue
                mseed_nm = filename.split('/')[-1]
                new_mseed = string.replace(mseed_nm, chan, new_chan)
                old_path = os.path.join(path, filename)
                new_path = '%s/%s/%s/%s.D/%s' % (out_dir, net,
                                                 sta, new_chan, new_mseed)

                print('Creating symlink for file %s at %s'
                      % (old_path, new_path))
                spwd = '*blackmore89'
                cmnd = 'sudo -S ln %s %s' % (old_path, new_path)
                os.system('echo %s | %s' % (spwd, cmnd))
    return


def asdf_create(asdf_name, wav_dirs, sta_dir):
    """
    Wrapper on ASDFDataSet to create a new HDF5 file which includes
    all waveforms in a directory and stationXML directory
    :param asdf_name: Full path to new asdf file
    :param wav_dir: List of directories of waveform files (will grab all files)
    :param sta_dir: Directory of stationXML files (will grab all files)
    :return:
    """
    import pyasdf
    import os
    from glob import glob
    from obspy import read

    with pyasdf.ASDFDataSet(asdf_name) as ds:
        wav_files = []
        for wav_dir in wav_dirs:
            wav_files.extend([os.path.join(root, a_file)
                              for root, dirs, files in os.walk(wav_dir)
                              for a_file in files])
        for _i, filename in enumerate(wav_files):
            print("Adding mseed file %i of %i..." % (_i+1, len(wav_files)))
            st = read(filename)
            #Add waveforms
            ds.add_waveforms(st, tag="raw_recording")
        sta_files = glob('%s/*' % sta_dir)
        for filename in sta_files:
            ds.add_stationxml(filename)
    return


def test_snr_distribution(cat, wav_dirs, prepick=0.05, length=1.0,
                          start=False, end=False):
    """
    Get a feel for what the distribution of SNRs for events in a catalog is.
    This should give us an idea of what our ideal threshold should be in
    generating templates
    :param cat: Catalog of interest
    :param wav_dirs: Waveform directories
    :return:
    """
    import datetime
    from obspy import UTCDateTime
    from eqcorrscan.utils import pre_processing
    from eqcorrscan.core.bright_lights import _rms

    # Establish date range for template creation
    cat.events.sort(key=lambda x: x.origins[-1].time)
    if start:
        cat_start = datetime.datetime.strptime(start, '%d/%m/%Y')
        cat_end = datetime.datetime.strptime(end, '%d/%m/%Y')
    else:
        cat_start = cat[0].origins[-1].time.date
        cat_end = cat[-1].origins[-1].time.date
    # Preallocate snr dict
    snrs = {}
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        print('Processing templates for: %s' % str(dto))
        q_start = dto - 10
        q_end = dto + 86410
        # Establish which events are in this day
        sch_str_start = 'time >= %s' % str(dto)
        sch_str_end = 'time <= %s' % str(dto + 86400)
        tmp_cat = cat.filter(sch_str_start, sch_str_end)
        if len(tmp_cat) == 0:
            print('No events on: %s' % str(dto))
            continue
        # Which stachans we got?
        stachans = {pk.waveform_id.station_code: [] for ev in tmp_cat
                    for pk in ev.picks}
        for ev in tmp_cat:
            for pk in ev.picks:
                chan_code = pk.waveform_id.channel_code
                if chan_code not in stachans[pk.waveform_id.station_code]:
                    stachans[pk.waveform_id.station_code].append(chan_code)
        print('Reading waveforms')
        wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
        st = grab_day_wavs(wav_ds, dto, stachans)
        print('Merging')
        st.merge(fill_value='interpolate')
        print('Preprocessing')
        try:
            st1 = pre_processing.dayproc(st, lowcut=1., highcut=20.,
                                         filt_order=3, samp_rate=100.,
                                         num_cores=6, starttime=dto,
                                         ignore_length=True)
        except NotImplementedError or Exception as e:
            print('Found error in dayproc, noting date and continuing')
            print(e)
            continue
        for tr in st1:
            stachan = '%s.%s' % (tr.stats.station, tr.stats.channel)
            print('Working on %s' % stachan)
            if stachan not in snrs.keys():
                snrs[stachan] = []
            noise_amp = _rms(tr.data)
            for ev in tmp_cat:
                for pk in ev.picks:
                    if pk.waveform_id.station_code == tr.stats.station and \
                            pk.waveform_id.channel_code == tr.stats.channel:
                        starttime = pk.time - prepick
                        tr_cut = tr.copy().trim(starttime=starttime,
                                                endtime=starttime + length,
                                                nearest_sample=False)
                        if len(tr_cut.data) == 0:
                            print('No data provided for %s.%s starting at %s' %
                                  (tr.stats.station, tr.stats.channel,
                                   str(starttime)))
                            continue
                        # Ensure that the template is the correct length
                        if len(tr_cut.data) == (tr_cut.stats.sampling_rate *
                                                    length) + 1:
                            tr_cut.data = tr_cut.data[0:-1]
                        snr = max(tr_cut.data) / noise_amp
                        snrs[stachan].append(snr)
    return snrs


def mseed_2_templates(wav_dirs, cat, outdir, length, prepick,
                      highcut=None, lowcut=None, f_order=None,
                      samp_rate=None, min_snr=2.,
                      start=None, end=None, miniseed=True,
                      asdf_file=False, debug=1):
    """
    Function to generate individual mseed files for each event in a catalog
    from a pyasdf file or continuous data.
    :param asdf_file: ASDF file with waveforms and stations
    :param cat: path to xml of Catalog of events for which we'll create
        templates
    :param outdir: output directory for miniseed files
    :param length: length of templates in seconds
    :param prepick: prepick time for waveform trimming
    :param highcut: Filter highcut (if desired)
    :param lowcut: Filter lowcut (if desired)
    :param f_order: Filter order
    :param samp_rate: Sampling rate for the templates
    :param start: start date as %Y/%m/%d if desired
    :param end: same as above. Defaults to full length of catalog.
    :return:
    """
    import pyasdf
    import collections
    import copy
    import datetime
    from obspy import UTCDateTime, Stream, read
    from eqcorrscan.core.template_gen import template_gen
    from eqcorrscan.utils import pre_processing
    from timeit import default_timer as timer

    # Establish date range for template creation
    cat.events.sort(key=lambda x: x.origins[-1].time)
    if start:
        cat_start = datetime.datetime.strptime(start, '%d/%m/%Y')
        cat_end = datetime.datetime.strptime(end, '%d/%m/%Y')
    else:
        cat_start = cat[0].origins[-1].time.date
        cat_end = cat[-1].origins[-1].time.date
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        print('Processing templates for: %s' % str(dto))
        q_start = dto - 10
        q_end = dto + 86410
        # Establish which events are in this day
        sch_str_start = 'time >= %s' % str(dto)
        sch_str_end = 'time <= %s' % str(dto + 86400)
        tmp_cat = cat.filter(sch_str_start, sch_str_end)
        if len(tmp_cat) == 0:
            print('No events on: %s' % str(dto))
            continue
        # Which stachans we got?
        stachans = {pk.waveform_id.station_code: [] for ev in tmp_cat
                    for pk in ev.picks}
        for ev in tmp_cat:
            for pk in ev.picks:
                chan_code = pk.waveform_id.channel_code
                if chan_code not in stachans[pk.waveform_id.station_code]:
                    stachans[pk.waveform_id.station_code].append(chan_code)
        wav_read_start = timer()
        # Be sure to go +/- 10 sec to account for GeoNet shit timing
        if asdf_file:
            with pyasdf.ASDFDataSet(asdf_file) as ds:
                st = Stream()
                for sta, chans in iter(stachans.items()):
                    for station in ds.ifilter(ds.q.station == sta,
                                              ds.q.channel == chans,
                                              ds.q.starttime >= q_start,
                                              ds.q.endtime <= q_end):
                        st += station.raw_recording
        elif miniseed:
            wav_ds = ['%s%d' % (d, dto.year) for d in wav_dirs]
            st = grab_day_wavs(wav_ds, dto, stachans)
        wav_read_stop = timer()
        print('Reading waveforms took %.3f seconds' % (wav_read_stop
                                                       - wav_read_start))
        print('Looping through stachans to merge/resamp')
        stachans = [(tr.stats.station, tr.stats.channel) for tr in st]
        for stachan in list(set(stachans)):
            tmp_st = st.select(station=stachan[0], channel=stachan[1])
            if len(tmp_st) > 1 and len(set([tr.stats.sampling_rate for tr in tmp_st])) > 1:
                print('Traces from %s.%s have differing samp rates' % (stachan[0], stachan[1]))
                for tr in tmp_st:
                    st.remove(tr)
                tmp_st.resample(sampling_rate=samp_rate)
                st += tmp_st
        st.merge(fill_value='interpolate')
        resamp_stop = timer()
        print('Resample/merge took %s secs' % str(resamp_stop - wav_read_stop))
        print('Preprocessing...')
        # Process the stream
        try:
            st1 = pre_processing.dayproc(st, lowcut=lowcut, highcut=highcut,
                                         filt_order=f_order, samp_rate=samp_rate,
                                         starttime=dto, debug=debug, ignore_length=True,
                                         num_cores=4)
        except NotImplementedError or Exception as e:
            print('Found error in dayproc, noting date and continuing')
            print(e)
            with open('%s/dayproc_errors.txt' % outdir, mode='a') as fo:
                fo.write('%s\n%s\n' % (str(date), e))
            continue
        print('Feeding stream to template_gen...')
        for event in tmp_cat:
            print('Copying stream to keep away from the trim...')
            trim_st = copy.deepcopy(st1)
            ev_name = str(event.resource_id).split('/')[-1]
            pk_stachans = ['%s.%s' % (pk.waveform_id.station_code,
                                      pk.waveform_id.channel_code)
                           for pk in event.picks]
            # Run check to ensure that there is only one pick for each channel
            dups = [pk for pk, count
                    in collections.Counter(pk_stachans).items() if count > 1]
            if len(dups) > 0:
                print('Event %s still has dup picks. Skipping' % ev_name)
                continue
            template = template_gen(event.picks, trim_st, length=length,
                                    prepick=prepick, min_snr=min_snr)
            if len([tr for tr in template
                    if tr.stats.channel[-1] == 'Z']) < 6:
                print('Skipping template with fewer than 6 Z-comp traces')
                continue
            # temp_list.append(template)
            print('Writing event %s to file...' % ev_name)
            template.write('%s/%s.mseed' % (outdir, ev_name),
                           format="MSEED")
            del trim_st
        del tmp_cat, st1, st


def template_spectrograms(temp_dir, num_evs):
    """
    Visualize spectrograms of template events to determine best
    passband for filters
    :param temp_dir: Directory where raw temp waveforms live
    :param num_evs: How many events to randomly select from all temps
    :return:
    """
    from glob import glob
    import numpy as np
    from obspy import read

    files = glob('%s/*' % temp_dir)
    rands = np.random.choice(range(len(files)), num_evs, replace=False)
    rand_files = [fl for i, fl in enumerate(files) if i in rands]
    for fl in rand_files:
        st = read(fl)
        for tr in st:
            tr.spectrogram()

##############################################################################
# vv Duplicate pick related BS vv #

def remove_temp_dups(templates, cat, bad_list):
    """
    Used to remove traces in template on same channel. Will keep pick closest
    to the average pick time for the other picks of the same phase
    :param templates: list of template files
    :param cat: catalog with duplicate picks removed
    :param bad_list: list of eid's for events to throw out
    :return:
    """
    from obspy import read
    import collections

    for ev in cat:
        if str(ev.resource_id).split('/')[-1] not in bad_list:
            temp = read([temp for temp in templates
                         if temp.split('/')[-1].rstrip('.mseed')
                         == str(ev.resource_id).split('/')[-1]][0])
            ev_stachans = [(pk.waveform_id.station_code,
                            pk.waveform_id.channel_code)
                           for pk in ev.picks]
            temp_stachans = [(tr.stats.station,
                              tr.stats.channel)
                             for tr in temp]
            # Find which stachans are duplicates
            dups = [t_stachan for t_stachan, count
                    in collections.Counter(temp_stachans).items() if
                    count > 1]
            for dup in dups:
                temp.traces.remove(dup)
            if len(dups) > 1:
                for dup in dups:
                    perps = temp.select(station=dup[0],
                                        channel=dup[1]).sort('starttime')
                    temp.remove(perps[-1])
                temp.write(temp_file.rstrip('.mseed') + '_nodups.mseed',
                           format='MSEED')
    return


def remove_dups_TauP(cat, input):
    """
    Taking stefan's TauP arrivals and chosing only the arrival closest to
    them
    :param cat: obspy.core.Catalog
    :param input: stefan's text file
    :return: 'Fixed' catalog
    """
    import csv
    from obspy import UTCDateTime, Catalog

    fixed_cat = cat.copy()
    with open(input, 'r') as f:
        lines = csv.reader(f)
        next(lines, None) # Skipping header
        eid = None # For keeping track of which event we're on
        for line in lines:
            if line[0] != eid:
                eid = line[0]
                ev = [ev for ev in fixed_cat
                      if str(ev.resource_id).split('/')[-1] == line[0]][0]
            picks = [pk for pk in ev.picks
                     if pk.waveform_id.station_code == line[1]
                     and pk.phase_hint == 'P']
            while len(picks) > 1:
                # If there are duplicates, remove them
                tauP_time = UTCDateTime(line[4])
                picks.remove(max(picks, lambda p: abs((p.time - tauP_time)))[0])
                ev.picks.remove(max(picks, lambda p: abs((p.time - tauP_time)))[0])
    return fixed_cat


def replace_dup_events(orig_cat, replacement_cat, bad_cat):
    """
    Take the original catalog and replace events with duplicate picks with
    the same events reviewed and corrected (via obspyck or otherwise)
    :param orig_cat: Original catalog
    :param replacement_cat: Catalog of corrected events
    :param bad_cat: Catalog of all events which had duplicate events
    :return:
    """
    from obspy import Catalog

    rev_ids = [str(ev.resource_id).split('/')[-1]
               for ev in replacement_cat.events]
    bad_ids = [str(ev.resource_id).split('/')[-1]
               for ev in bad_cat.events]
    remove_ids = list(set(bad_ids) - set(rev_ids))
    for ev in orig_cat:
        eid = str(ev.resource_id).split('/')[-1]
        if eid in rev_ids:
            rev_ev = [evnt for evnt in replacement_cat.events
                      if str(evnt.resource_id).split('/')[-1] == eid][0]
            ev.picks = rev_ev.picks
            ev.origins = rev_ev.origins
    for id in remove_ids:
        orig_cat.events.remove([ev for ev in orig_cat
                                if str(ev.resource_id).split('/')[-1]
                                == id][0])
    return

##############################################################################

def sync_temps_catalogs(cat, temp_dir):
    # Remove the events from catalog which didn't get made into temps due
    # to low SNR
    from glob import glob

    temp_files = glob(temp_dir)
    temp_names = [nm.split('/')[-1].split('.')[0] for nm in temp_files]
    rm_evs = []
    for ev in cat:
        if str(ev.resource_id).split('/')[-1] not in temp_names:
            rm_evs.append(ev)
    for rm_ev in rm_evs:
        cat.events.remove(rm_ev)
    return


def mseed_2_Tribe(temp_dir, cat, swin='all', tar_name=None):
    """
    Take a directory of templates and make them into a Tribe object
    :param temp_dir: Directory containing templates
    :param cat: catalog coresponding to templates
    :return:
    """
    from eqcorrscan.core.match_filter import Template, Tribe
    from glob import glob
    from obspy import read

    temp_files = glob('%s/*' % temp_dir)
    Tribe = Tribe()
    for ev in cat:
        eid = str(ev.resource_id).split('/')[-1]
        print('Adding event: %s' % eid)
        temp = [read(temp_file) for temp_file in temp_files
                if temp_file.split('/')[-1].split('.')[0] == eid][0]
        if swin == 'P':
            for tr in temp.copy():
                if tr.stats.channel[-1] != 'Z':
                    temp.remove(tr)
        T_o = Template(name=eid, st=temp, lowcut=3., highcut=20.,
                       samp_rate=50., filt_order=3, process_length=86400,
                       prepick=0.1, event=ev)
        Tribe += T_o
    if tar_name:
        Tribe.write(tar_name)
    return Tribe


def mseed_2_Party(wav_dir, cat, temp_cat, lowcut, highcut, filt_order,
                  process_length, prepick):
    """
    Take waveforms and catalog and create a Party object
    :param wav_dir:
    :param cat:
    :return:
    """
    from eqcorrscan.core.match_filter import Detection, Family, Party, Template
    from obspy import read, UTCDateTime, Catalog

    partay = Party()
    # Get templates first
    temp_tup = [(ev, str(ev.resource_id).split('/')[-1].split('_')[0])
                for ev in cat
                if str(ev.resource_id).split('/')[-1].split('_')[-1]=='self']
    temp_evs, temp_ids = zip(*temp_tup)
    temp_evs = list(temp_evs)
    wav_files = ['%s/%s.mseed' % (wav_dir, str(ev.resource_id).split('/')[-1])
                 for ev in temp_evs]
    temp_wavs = [read(wav) for wav in wav_files]
    for temp_wav, temp_ev in zip(temp_wavs, temp_evs):
        #Create a Template object, assign it to Family and then to Party
        tid = str(temp_ev.resource_id).split('/')[-1].split('_')[0]
        if len([ev for ev in temp_cat
                if str(ev.resource_id).split('/')[-1] == tid]) > 0:
            temp_ev = [ev for ev in temp_cat
                           if str(ev.resource_id).split('/')[-1] == tid][0]
        tmp = Template(name=tid, st=temp_wav, lowcut=lowcut, highcut=highcut,
                       samp_rate=temp_wav[0].stats.sampling_rate,
                       filt_order=filt_order, process_length=process_length,
                       prepick=prepick, event=temp_ev)
        fam_det_evs = [ev for ev in cat
                       if str(ev.resource_id).split('/')[-1].split('_')[-1]!='self'
                       and str(ev.resource_id).split('/')[-1].split('_')[0]==tid]
        fam_dets = [Detection(template_name=str(ev.resource_id).split('/')[-1].split('_')[0],
                              detect_time=UTCDateTime([com.text.split('=')[-1]
                                                       for com in ev.comments
                                                       if com.text.split('=')[0]=='det_time'][0]),
                              no_chans=len(ev.picks),
                              chans=[pk.waveform_id.station_code
                                     for pk in ev.picks],
                              detect_val=float([com.text.split('=')[-1]
                                                for com in ev.comments
                                                if com.text.split('=')[0]=='detect_val'][0]),
                              threshold=float([com.text.split('=')[-1]
                                               for com in ev.comments
                                               if com.text.split('=')[0]=='threshold'][0]),
                              typeofdet='corr',
                              threshold_type='MAD',
                              threshold_input=8.0,
                              event=ev, id=str(ev.resource_id).split('/')[-1])
                    for ev in fam_det_evs]
        fam_cat = Catalog(events=[det.event for det in fam_dets])
        fam = Family(template=tmp, detections=fam_dets, catalog=fam_cat)
        partay.families.append(fam)
    return partay


def party_svd_moments(party, shift_len, length, reject, wav_dir,
                      calibrate=False):
    """
    Calculate the relative moments for detections in a Family using
    mag_calc.svd_moments()

    :type calibrate: bool
    :param calibrate: If present, will scale moments to template magnitude
    :return: Party
    """
    import copy
    import numpy as np
    from obspy import read, Catalog
    from obspy.core.event import Magnitude
    from eqcorrscan.core.subspace import _subspace_process
    from eqcorrscan.utils.clustering import svd
    from eqcorrscan.utils.mag_calc import svd_moments

    for fam in party.families:
        prepick = fam.template.prepick
        events = [det.event for det in fam.detections]
        wav_files = ['%s/%s.mseed' % (wav_dir,
                                      str(ev.resource_id).split('/')[-1])
                     for ev in events]
        streams = [read(wav_file) for wav_file in wav_files]
        streams.insert(0, fam.template.st)
        front_clip = prepick - (shift_len / 2.) - 0.05
        back_clip = prepick - 0.05 + length + (shift_len / 2.)
        wrk_streams = copy.deepcopy(streams)
        for st in wrk_streams:
            for tr in st:
                tr.trim(starttime=tr.stats.starttime + front_clip,
                        endtime=tr.stats.starttime + back_clip)
        tmp = fam.template
        p_streams,stachans = _subspace_process(streams=copy.deepcopy(wrk_streams),
                                               lowcut=tmp.lowcut,
                                               highcut=tmp.highcut,
                                               filt_order=tmp.filt_order,
                                               sampling_rate=tmp.samp_rate,
                                               multiplex=False, align=True,
                                               shift_len=shift_len,
                                               reject=reject)
        # Remove zero-padded traces
        for stream in p_streams:
            for tr in stream.copy():
                if np.count_nonzero(tr.data) == 0:
                    stream.traces.remove(tr)
        event_list = []
        for stachan in stachans:
            st_list = []
            for i, st in enumerate(p_streams):
                if len(st.select(station=stachan[0],
                                 channel=stachan[-1])) > 0:
                    st_list.append(i)
            event_list.append(st_list)
        # event_list = np.asarray(event_list).tolist()
        u, sigma, v, stachans = svd(stream_list=p_streams, full=True)
        try:
            M, events_out = svd_moments(u, sigma, v, stachans, event_list)
        except IOError as e:
            print('Family %s raised error %s' % (fam.template.name, e))
            continue
        # If we have a Mag for template, calibrate moments
        if calibrate and len(fam.template.event.magnitudes) > 0:
            temp_mag = fam.template.event.preferred_magnitude().mag
            #Normalize moments to template mag
            M_normed = [m/M[0] for m in M]
            M_calib = [temp_mag + ((2/3)*np.log10(m/M[0])) for m in M_normed]
            # M_calib = [(temp_mag / M[0]) * m for m in M]
            # Add calibrated mags to detection events
            for i, eind in enumerate(events_out):
                fam.detections[eind - 1].event.magnitudes.append(Magnitude(mag=M_calib[i]))
            fam.catalog = Catalog(events=[det.event for det in fam.detections])
    return party


def shelly_mags(party, wav_dir):
    """
    Implementation of David Shelly's relative amplitude-based calculations
    :param party:
    :return:
    """
    from obspy import read

    for fam in party:
        prepick = fam.template.prepick
        events = [det.event for det in fam.detections]
        wav_files = ['%s/%s.mseed' % (wav_dir,
                                      str(ev.resource_id).split('/')[-1])
                     for ev in events]
        streams = [read(wav_file) for wav_file in wav_files]

    return


def make_dist_mat(directory, highcut, lowcut, samp_rate,
                  filt_order, raw_prepick, corr_prepick,
                  length, shift, outfile, cores):
    """
    Taking a directory of templates, processing wavs and computing correlation
    clustering prior to creating subspace
    :param directory: Directory of template mseeds
    :param highcut: filter highcut
    :param lowcut: filter lowcut
    :param samp_rate: output sample rate
    :param filt_order: filter corners
    :param raw_prepick: Prepick time of template files
    :param corr_prepick: Output prepick before correlations
    :param length: Length of temp to be correlating
    :param shift: Shift length in secs allowed during correlations
    :param outfile: Filename for output distance matrix
    :param method: Method for heirarchical clustering
    :return:
    """
    from glob import glob
    import numpy as np
    from obspy import read
    from eqcorrscan.utils.pre_processing import shortproc
    from eqcorrscan.utils.clustering import distance_matrix

    temp_files = glob('%s/*' % directory)
    temp_list = [(shortproc(read(tmp),lowcut=lowcut, highcut=highcut,
                            samp_rate=samp_rate, filt_order=filt_order,
                            parallel=True, num_cores=cores),
                  tmp.split('/')[-1].split('.')[0])
                 for tmp in temp_files]
    front_clip = raw_prepick - corr_prepick
    back_clip = front_clip + length
    for temp in temp_list:
        for tr in temp[0]:
            tr.trim(starttime=tr.stats.starttime + front_clip,
                    endtime=tr.stats.starttime + back_clip)
    temp_sts = [x[0] for x in temp_list]
    print('Starting distance matrix computations')
    dist_mat = distance_matrix(temp_sts, allow_shift=True, shift_len=shift,
                               cores=cores)
    print('Saving matrix to %s' % outfile)
    np.save(outfile, dist_mat)
    return

def cluster_temp_list(directory, dist_mat, method):
    """
    Do event clustering from pre-saved dist_mat
    :param directory:
    :param dist_mat:
    :param method:
    :return:
    """
    from glob import glob
    from obspy import read
    from scipy.cluster.hierarchy import fcluster
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    temp_files = glob('%s/*' % directory)
    temp_list = [(read(tmp), tmp.split('/')[-1].split('.')[0])
                 for tmp in temp_files]
    dist_df = pd.DataFrame(np.load(dist_mat))
    clust_grd = sns.clustermap(dist_df, method=method, vmin=0., vmax=1.)
    plt.show()
    Z = clust_grd.dendrogram_row.linkage
    indices = fcluster(Z, t=1.0, criterion='inconsistent')
    group_ids = list(set(indices))  # Unique list of group ids
    # Convert to tuple of (group id, stream id)
    indices = [(indices[i], i) for i in range(len(indices))]
    # Sort by group id
    indices.sort(key=lambda tup: tup[0])
    groups = []
    for group_id in group_ids:
        group = []
        for ind in indices:
            if ind[0] == group_id:
                group.append(temp_list[ind[1]])
            elif ind[0] > group_id:
                # Because we have sorted by group id, when the index is greater
                # than the group_id we can break the inner loop.
                # Patch applied by CJC 05/11/2015
                groups.append(group)
                break
    # Catch the final group
    groups.append(group)
    return groups