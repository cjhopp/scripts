#!/usr/bin/env python

"""
Creating stationxml files for 'newer' stations:

Excerpt from SC3 station file:

<Station code="new_inv" startDate="2015-05-01T00:00:00" restrictedStatus="open">
<Latitude>-38.5764</Latitude>
<Longitude>176.2154</Longitude>
<Elevation>382</Elevation>
<Site>
<Name>MRP Ngatamariki Seismic 15</Name>
</Site>
<CreationDate>2015-05-01T00:00:00</CreationDate>
</Station>
<Station code="NS16" startDate="2015-04-23T00:00:00" restrictedStatus="open">
<Latitude>-38.5751</Latitude>
<Longitude>176.162</Longitude>
<Elevation>450</Elevation>
<Site>
<Name>MRP Ngatamariki 16</Name>
</Site>
<CreationDate>2015-04-23T00:00:00</CreationDate>
</Station>
<Station code="NS18" startDate="2015-04-23T00:00:00" restrictedStatus="open">
<Latitude>-38.5307</Latitude>
<Longitude>176.1852</Longitude>
<Elevation>348</Elevation>
<Site>
<Name>MRP Ngatamariki 18</Name>
</Site>
<CreationDate>2015-04-23T00:00:00</CreationDate>
</Station>
<Station code="RT22" startDate="2015-06-24T00:00:00" restrictedStatus="open">
<Latitude>-38.580367</Latitude>
<Longitude>176.186985</Longitude>
<Elevation>423</Elevation>
<Site>
<Name>MRP Rotokawa 22</Name>
</Site>
<CreationDate>2015-06-24T00:00:00</CreationDate>
</Station>
<Station code="RT23" startDate="2015-06-25T00:00:00" restrictedStatus="open">
<Latitude>-38.600085</Latitude>
<Longitude>176.160466</Longitude>
<Elevation>447</Elevation>
<Site>
<Name>MRP Rotokawa 23</Name>
</Site>
<CreationDate>2015-06-25T00:00:00</CreationDate>
</Station>
"""

from obspy import read_inventory, UTCDateTime
from obspy.core.inventory import Inventory
#Dictionary of dictionaries of new stations
new_stas = {}
new_stas['NS15'] = {}
new_stas['NS15']['code'] = 'NS15'
new_stas['NS15']['start'] = UTCDateTime('2015-05-01T00:00:00')
new_stas['NS15']['name'] = 'MRP Ngatamariki 15'
new_stas['NS15']['lat'] = -38.5764
new_stas['NS15']['lon'] = 176.2154
new_stas['NS15']['elev'] = 382

new_stas['NS16'] = {}
new_stas['NS16']['code'] = 'NS16'
new_stas['NS16']['start'] = UTCDateTime('2015-04-23T00:00:00')
new_stas['NS16']['name'] = 'MRP Ngatamariki 16'
new_stas['NS16']['lat'] = -38.5751
new_stas['NS16']['lon'] = 176.162
new_stas['NS16']['elev'] = 450

new_stas['NS18'] = {}
new_stas['NS18']['code'] = 'NS18'
new_stas['NS18']['start'] = UTCDateTime('2015-04-23T00:00:00')
new_stas['NS18']['name'] = 'MRP Ngatamariki 18'
new_stas['NS18']['lat'] = -38.5307
new_stas['NS18']['lon'] = 176.1852
new_stas['NS18']['elev'] = 348

new_stas['RT22'] = {}
new_stas['RT22']['code'] = 'RT22'
new_stas['RT22']['start'] = UTCDateTime('2015-06-24T00:00:00')
new_stas['RT22']['name'] = 'MRP Rotokawa 22'
new_stas['RT22']['lat'] = -38.580367
new_stas['RT22']['lon'] = 176.186985
new_stas['RT22']['elev'] = 423

new_stas['RT23'] = {}
new_stas['RT23']['code'] = 'RT23'
new_stas['RT23']['start'] = UTCDateTime('2015-06-25T00:00:00')
new_stas['RT23']['name'] = 'MRP Rotokawa 23'
new_stas['RT23']['lat'] = -38.600085
new_stas['RT23']['lon'] = 176.160466
new_stas['RT23']['elev'] = 447

starting_inv = read_inventory('/home/chet/data/GeoNet_catalog/stations/station_xml/NS10_STATIONXML.xml')
station_template = starting_inv[0]

for new_sta in new_stas:
    new_sta_dict = new_stas[new_sta]
    #Copy network
    temp_network = [station_template.copy()]
    #Put into new inventory object
    temp_inv = Inventory(temp_network, 'VUW')
    temp_sta = temp_inv[0].stations[0]
    temp_sta.code = unicode(new_sta_dict['code'], 'utf-8')
    temp_sta.start_date = new_sta_dict['start']
    temp_sta.creation_date = new_sta_dict['start']
    temp_sta.site.name = new_sta_dict['name']
    temp_sta.latitude = new_sta_dict['lat']
    temp_sta.longitude = new_sta_dict['lon']
    temp_sta.elevation = new_sta_dict['elev']
    #Loop through channel and response info to change minor naming issues
    for chan in temp_sta.channels:
        chan.start_date = new_sta_dict['start']
        chan.latitude = new_sta_dict['lat']
        chan.longitude = new_sta_dict['lon']
        chan.elevation = new_sta_dict['elev']
        #Roundabout replacing of station name in descriptions
        dl_desc_split = chan.data_logger.description.split('.')
        dl_desc_split[0] = new_sta_dict['code']
        chan.data_logger.description = '.'.join(dl_desc_split)
        #Do the same replacement as above for all response stages
        for stage in chan.response.response_stages:
            if stage.name:
                tmp_name = stage.name.split('.')
                tmp_name[0] = new_sta_dict['code']
                stage.name = '.'.join(tmp_name)
    temp_inv.write('/home/chet/data/GeoNet_catalog/stations/station_xml/' +
                   new_sta_dict['code'] + '_STATIONXML.xml',
                   format='STATIONXML')
