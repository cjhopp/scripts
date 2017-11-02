#!/usr/bin/python

"""
Fedes intro to obspy Inventory object fun and string parsing/formatting

Hopefully this works. I think your notebook is python3.
"""

import os
from datetime import timedelta
from obspy import read, read_inventory, UTCDateTime
from obspy.clients.fdsn import Client

def date_generator(start_date, end_date):
    # Generator for date looping
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def freddy_sac_convert(root_dir, outdir, inv_file):
    """
    Function to do mseed to sac convertion on fede's machine
    :param root_dir: root directory path (day_volumes)
    :param outdir: path to output directory for sac files
    :param inv_file: path to stationxml metadata file
    :return:
    """
    # Read inventory from StationXML. This will recognize the format automatically
    inv = read_inventory(inv_file)
    # Loop through just immediate subdirectoies (stations)
    for root, dirs, wav_files in os.walk(root_dir):
        sta = root.split('/')[-1]
        # Glob creates a list of all of the files in a dir that match a pattern
        # Check if a subdirectory exists with this station name
        # in the output directory. If it doesn't, make one.
        # This just mimics your original structure. Change it if you want.
        if not os.path.exists('{}/{}'.format(outdir, sta)):
            os.mkdir('{}/{}'.format(outdir, sta))
        # Loop through each of those files
        for wav_file in wav_files:
            # Splits the file path by '/' and takes the last element
            # Splits the file name by '.' for use when nameing the output file
            name_parts = wav_file.split('.')
            # Read into Stream, merge, take first (only) trace
            tr = read(os.path.join(root, wav_file)).merge(
                fill_value='interpolate')[0]
            tr = attach_sac_header(tr, inv)
            chan = tr.stats.channel
            # So now we've got the trace with all of the necessary
            # header info. We'll create the filename first, then save it
            out_path = '{}/{}/{}.{}.{}.{}.sac'.format(outdir, sta,
                                                      name_parts[-2],
                                                      name_parts[-1], sta,
                                                      chan)
            tr.write(out_path, format='SAC')

def get_geonet_wavs(cat, inv, outdir):
    # Establish the client
    cli = Client('GEONET')
    cat.events.sort(key=lambda x: x.origins[-1].time)
    cat_start = cat[0].origins[-1].time.date
    cat_end = cat[-1].origins[-1].time.date
    for date in date_generator(cat_start, cat_end):
        dto = UTCDateTime(date)
        # Loop over all stations in the network
        # (assuming only one network: 'NZ')
        for sta in inv[0]:
            # This just mimics your original structure. Change it if you want.
            if not os.path.exists('{}/{}'.format(outdir, sta.code)):
                os.mkdir('{}/{}'.format(outdir, sta.code))
            try:
                st = cli.get_waveforms(network='NZ', station=sta.code,
                                       starttime=dto, endtime=dto + 86400)
            except FDSNException as e:
                print(e)
                continue
            st.merge(fill_value='interpolate')
            for tr in st:
                chan = tr.stats.channel
                tr = attach_sac_header(tr, inv)
                out_path = '{}/{}/{!s}.{!s}.{}.{}.sac'.format(
                    outdir, sta.code, (dto + 10).year, (dto + 10).julday,
                    sta.code, chan)
                tr.write(out_path, format='SAC')

def attach_sac_header(tr, inv):
    sta_inv = inv.select(station=tr.stats.station)
    tr.stats.sac = {}
    tr.stats['sac']['knetwk'] = sta_inv[0].code
    tr.stats['sac']['kstnm'] = sta_inv[0][0].code
    tr.stats['sac']['stla'] = sta_inv[0][0].latitude
    tr.stats['sac']['stlo'] = sta_inv[0][0].longitude
    tr.stats['sac']['stel'] = sta_inv[0][0].elevation
    # Loop over every channel in the station
    for chan in sta_inv[0][0]:
        # If it matches the channel of the trace, use it
        if chan.code == tr.stats.channel:
            tr.stats['sac']['stdp'] = chan.depth
            tr.stats['sac']['cmpaz'] = chan.azimuth
            tr.stats['sac']['kcmpnm'] = chan.code
            # SAC cmpinc is deg from vertical (not horiz)
            if chan.dip == -90.0:
                tr.stats['sac']['cmpinc'] = 180.0
                tr.stats['sac']['lpspol'] = False
            elif chan.dip == 90.0:
                tr.stats['sac']['cmpinc'] = 0.0
                tr.stats['sac']['lpspol'] = True
            else:
                tr.stats['sac']['cmpinc'] = 90.0
    return tr
