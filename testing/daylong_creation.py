
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


client = Client('GEONET')
t1 = UTCDateTime('2016-11-13T11:00:00')
t2 = UTCDateTime('2016-11-14T11:00:00')
st = client.get_waveforms('NZ', 'WEL', '*', 'HH*', t1, t2)

print st
st.write('daylong-data.ms', format='MSEED')
