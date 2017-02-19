#!/usr/bin/python


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


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


def mseed_2_templates(wav_dirs, cat, outdir, length, prepick,
                      highcut=None, lowcut=None, f_order=None,
                      samp_rate=None, start=None, end=None,
                      miniseed=True, asdf_file=False, debug=1):
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
    import os
    import fnmatch
    from itertools import chain
    import collections
    import copy
    import datetime
    from obspy import UTCDateTime, Stream, read
    from eqcorrscan.core.template_gen import template_gen
    from eqcorrscan.utils import pre_processing
    from timeit import default_timer as timer

    # Establish date range for template creation
    cat.events.sort(key=lambda x: x.preferred_origin().time)
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
            st = Stream()
            wav_files = []
            for path, dirs, files in chain.from_iterable(os.walk(path)
                                                         for path in wav_dirs):
                for sta, chans in iter(stachans.items()):
                    for chan in chans:
                        for filename in fnmatch.filter(files,
                                                       '*.%s.*.%s*%03d'
                                                       % (sta, chan, dto.julday)):
                            wav_files.append(os.path.join(path, filename))
            for wav in wav_files:
                st += read(wav)
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
                                         starttime=dto, debug=debug, ignore_length=True)
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
                for dup in dups:
                    NotImplementedError('More than one pick on a channel: ' +
                                        '%s: %s' % (ev_name, dup))
            template = template_gen(event.picks, trim_st, length=length,
                                    prepick=prepick)
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


def remove_temp_overlaps(templates, temp_files):
    """
    Used to remove traces in template on same channel. Will keep earliest
    pick unless Steve Sherb says otherwise
    :param templates: list of Stream objects
    :return:
    """
    import collections

    for temp, temp_file in zip(templates, temp_files):
        stachans = [(tr.stats.station, tr.stats.channel)
                    for tr in temp]
        dups = [tr for tr, count
                in collections.Counter(stachans).items() if count > 1]
        if len(dups) > 1:
            for dup in dups:
                perps = temp.select(station=dup[0],
                                    channel=dup[1]).sort('starttime')
                temp.remove(perps[-1])
            temp.write(temp_file.rstrip('.mseed') + '_nodups.mseed',
                       format='MSEED')
    return


def remove_duplicate_picks(cat, outfile=None):
    """
    Search through a catalog and remove duplicate picks
    :param cat:
    :param outfile: path to new catalog file
    :return:
    """
    import collections

    write_flag=False
    for ev in cat:
        stachans = [(pk.waveform_id.station_code,
                     pk.waveform_id.channel_code,
                     pk.phase_hint) for pk in ev.picks]
        dups = [pk for pk, count
                in collections.Counter(stachans).items() if count > 1]
        if len(dups) > 1:
            write_flag=True
            print('Fixing event %s' % str(ev.resource_id))
            for dup in dups:
                perp_pks = [pk for pk in ev.picks
                            if pk.waveform_id.station_code == dup[0]
                            and pk.waveform_id.channel_code == dup[1]
                            and pk.phase_hint == dup[2]]
                # Sort by time and remove all but first dup
                srtd_perps = sorted(perp_pks, key=lambda x: x.time)
                for perp in srtd_perps[1:]:
                    ev.picks.remove(perp)
    if write_flag:
        if outfile:
            cat.write(outfile, format='QUAKEML')
    else:
        print('No duplicate picks in this catalog.')
    return
