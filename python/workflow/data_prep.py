#!/usr/bin/python


def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int ((end_date - start_date).days)):
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


def pyasdf_2_templates(asdf_file, cat_path, outdir, length, prepick,
                       highcut=None, lowcut=None, f_order=None,
                       samp_rate=None, debug=1):
    """
    Function to generate individual mseed files for each event in a catalog
    from a pyasdf file of continuous data.
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
    :return:
    """
    import pyasdf
    import copy
    from obspy import UTCDateTime, Stream, read_events
    from eqcorrscan.core.template_gen import template_gen
    from eqcorrscan.utils import pre_processing
    from timeit import default_timer as timer

    # Read in catalog
    cat = read_events(cat_path)
    # Establish date range for template creation
    cat.events.sort(key=lambda x: x.preferred_origin().time)
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
        with pyasdf.ASDFDataSet(asdf_file) as ds:
            st = Stream()
            for sta, chans in stachans.iteritems():
                for station in ds.ifilter(ds.q.station == sta,
                                          ds.q.channel == chans,
                                          ds.q.starttime >= q_start,
                                          ds.q.endtime <= q_end):
                    st += station.raw_recording
        wav_read_stop = timer()
        print('Reading waveforms took %.3f seconds' % (wav_read_stop
                                                       - wav_read_start))
        print('Merging stream...')
        if debug > 1:
            print('Length of st pre-merge: %d' % len(st))
        st.merge(fill_value='interpolate')
        if debug > 1:
            print('Length of st post-merge: %d' % len(st))
            print()
        print('Preprocessing...')
        # Process the stream
        # First check that all traces are len() == 1
        if debug > 1:
            tr_lens = ['%s.%s: %s %s' % (tr.stats.station, tr.stats.channel,
                                         len(tr), type(tr))
                       for tr in st]
            print(tr_lens)
        st1 = pre_processing.dayproc(st, lowcut=lowcut, highcut=highcut,
                                     filt_order=f_order, samp_rate=samp_rate,
                                     starttime=dto, debug=debug)
        print('Feeding stream to _template_gen...')
        for event in tmp_cat:
            print('Copying stream to keep away from the trim...')
            trim_st = copy.deepcopy(st1)
            ev_name = str(event.resource_id).split('/')[-1]
            template = template_gen(event.picks, trim_st, length=length,
                                    prepick=prepick)
            # temp_list.append(template)
            print('Writing event %s to file...' % ev_name)
            template.write('%s/%s_raw.mseed' % (outdir, ev_name),
                           format="MSEED")
            del trim_st
        del tmp_cat, st1, st
