#/usr/bin/env python

"""
Create pyasdf file
"""

def asdf_create(asdf_name, wav_dir, sta_dir):
    """
    Wrapper on ASDFDataSet to create a new HDF5 file which includes
    all waveforms in a directory and stationXML directory
    :param asdf_name: Full path to new asdf file
    :param wav_dir: Directory of waveform files (will grab all files)
    :param sta_dir: Directory of stationXML files (will grab all files)
    :return:
    """
    import pyasdf
    from obspy import read
    from glob import glob

    with pyasdf.ASDFDataSet(asdf_name) as ds:
        wav_files = glob('%s/*/*/*' % wav_dir)
        for _i, filename in enumerate(wav_files):
            print("Adding mseed file %i of %i..." % (_i+1, len(wav_files)))
            st = read(filename)
            #Add waveforms
            ds.add_waveforms(st, tag="raw_recording")
        sta_files = glob('%s/*' % sta_dir)
        for filename in sta_files:
            ds.add_stationxml(filename)
    return