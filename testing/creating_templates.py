#template creation
#Works with GEONET

import sys
sys.path.insert(0, '/Users/home/taylorte/EQcorrscan')

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from eqcorrscan.core.template_gen import from_client
from eqcorrscan.tutorials.get_geonet_events import get_geonet_events
from eqcorrscan.utils.catalog_utils import filter_picks
from eqcorrscan.core import template_gen


catalog = get_geonet_events(
        startdate = UTCDateTime('2016-11-13T11:00:00'), 
        enddate = UTCDateTime('2016-11-14T11:00:00'), 
        maxlat = -42.0, minlat = -43.0, minlon = 173, maxlon = 174)

print catalog


filtered_catalog = filter_picks(catalog, stations=['WEL'], top_n_picks=2) 
print filtered_catalog

templates = from_client(
        catalog = filtered_catalog, client_id='GEONET', lowcut = 2.0, 
        highcut = 9.0, samp_rate = 20.0, filt_order = 4, length = 2.0, 
        prepick = 0.15, swin = 'all', process_len = 86400)
print templates

templates[0].plot()
templates[1].plot()
templates[2].plot()
templates[3].plot()

templates[0].write('kaik_eq-WEL.ms', format = 'MSEED')
templates[1].write('kaik_eq-WEL2.ms', format = 'MSEED')
templates[2].write('kaik_eq-WEL3.ms', format = 'MSEED')
templates[3].write('kaik_eq-WEL4.ms', format = 'MSEED')


