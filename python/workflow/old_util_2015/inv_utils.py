#!/usr/bin/env python

"""Standard things that I like to do with obspy.core.inventory objects"""

# Read in one of the cats
# cat = read_events('/media/chet/hdd/seismic/NZ/catalogs/2015_dets_nlloc/2015_dets_nlloc_Sherburn_no_dups.xml')

# staml_dir = '/home/chet/data/GeoNet_catalog/stations/station_xml/*'
def files2inv(directory, source=''):
    # Read individual stationxmls to inventory
    from obspy import read_inventory, Inventory
    from glob import glob
    files = glob(directory)
    inv = Inventory(networks=[], source=source)
    for filename in files:
        sing_sta_inv = read_inventory(filename)
        if sing_sta_inv[0].code in inv.get_contents()['networks']:
            inv[0].stations += sing_sta_inv[0].stations
        else:
            inv += sing_sta_inv
    return inv


def stas_w_picks(inv, cat, start_date, end_date):
    stas = list(set([pk.waveform_id.station_code for ev in cat for pk in ev.picks
                     if start_date < ev.preferred_origin().time < end_date]))
    for net in inv:
        for sta in net:
            if sta.code not in stas:
                net.stations.remove(sta)
    return inv


def sta_available_plot(inv, size=(18.5, 10.5)):
    from obspy import UTCDateTime
    import matplotlib.pyplot as plt
    sta_list = [sta for net in inv for sta in net]
    sta_list = sorted(sta_list, key=lambda k: k.code)
    fig, axes = plt.subplots(len(sta_list), 1, sharex=True, figsize=size)
    for i, sta in enumerate(sta_list):
        start = sta.start_date
        if sta.termination_date:
            end = sta.termination_date
        else:
            end = UTCDateTime(year=2015, month=12, day=31)
        axes[i].plot([start.datetime, end.datetime], [0, 0], color='red', linewidth=10.0)
        axes[i].set_ylabel(sta.code, rotation=0, horizontalalignment='right', verticalalignment='center')
        axes[i].yaxis.set_ticks([])
    plt.show()
    return fig


def dataless2xseed(indir, outdir):
    """
    Function for taking a directory of dataless files, parsing them with obspy.io.xseed and writing to file
    :type indir: str
    :param indir: Input directory with wildcards
    :type outdir: str
    :param outdir: Output directory for xseed
    :return: nuthin
    """
    from glob import glob
    from obspy.io.xseed import Parser
    files = glob(indir)
    for filename in files:
        sp = Parser(filename)
        sp.writeXSEED('%s%s.xseed' % (outdir, filename.split('/')[-1].strip()))
    return