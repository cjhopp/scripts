#!/usr/bin/python

"""
Script to pull down a large amount of waveform data. Edit as needed and run from command line
"""

from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import Restrictions, GlobalDomain, MassDownloader

domain = GlobalDomain()

restrictions = Restrictions(starttime=UTCDateTime(2014, 9, 1), endtime=UTCDateTime(2015, 2, 14),
                            chunklength_in_sec=86400, network='9G', station='NN*', location='*', channel='*',
                            reject_channels_with_gaps=False, minimum_length=0.0)

mdl = MassDownloader(providers=['IRIS'])
mdl.download(
    domain, restrictions,
    mseed_storage=('/Data2/old-newberry-waveforms/{network}/{station}/{channel}.{location}.{starttime}.{endtime}.mseed'),
    stationxml_storage=('chet-meq/newberry/instruments/stationxml')
)

