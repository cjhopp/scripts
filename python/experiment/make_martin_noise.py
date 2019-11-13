from obspy import read, read_events, read_inventory, Catalog, Stream, Trace, UTCDateTime
from surf_seis.vibbox import vibbox_read
from glob import glob

vbboxes = glob('/media/chet/data/chet-collab/wavs/test_vbox_raw/vbox_2018051714*')
st_raw = Stream()
for vb in vbboxes:
    st_raw += vibbox_read(vb).select(station='OT16').copy()
st_interp = st_raw.copy()
# Fill gaps via interpolation
st_interp.merge(fill_value='interpolate')
# Demean each trace
for tr in st_interp:
    tr.detrend(type='demean')
# Traces too large to write, break into 10-minute chunks
start = UTCDateTime(2018, 5, 17, 14)
for i in range(6):
    start_slice = start + (600 * i)
    end_slice = start_slice + 600.
    for tr in st_interp:
        nm = '{}.{}..{}.{}.mseed'.format(tr.stats.network, tr.stats.station,
                                         tr.stats.channel, start_slice)
        tr.slice(starttime=start_slice, endtime=end_slice).write(nm) 
