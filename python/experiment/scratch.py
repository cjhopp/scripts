"""
Random shit for standard use in ipython. Meaningless as a script
"""

for pick in ev2.picks:
    if not 'pick_list' in locals():
        pick_list = [pick.waveform_id.station_code]
    else:
        pick_list.append(pick.waveform_id.station_code)


import fnmatch
import os

raw_files = []
raw_dir = '/Volumes/Taranaki_01/data/hoppche/waveform_data/2015'
for root, dirnames, filenames in os.walk(raw_dir):
    for filename in fnmatch.filter(filenames, '*2015.001'):
        raw_files.append(os.path.join(root, filename))

#Read os.walk-ed file list to obspy.Stream
for filename in raw_files:
    if not 'st' in locals():
        st = obspy.read(filename)
    else:
        st += obspy.read(filename)
st.merge(fill_value='interpolate')

#Add waveforms to pyasdf
for filename in raw_files:
    ds.add_waveforms(filename, tag="raw_recording")

#Messing with duplicate picks in quakeml from Steve
for pick in ev.picks:
    if pick.waveform_id.station_code == 'WPRZ':
        if not 'mult_picks' in locals():
            mult_picks = [pick]
        else:
            mult_picks.append(pick)

#Messing with picks
for pick in picks:
    if pick.waveform_id.station_code == 'WPRZ':
        picks.remove(pick)

#Extract all stations from pyasdf
for station in ds.ifilter(ds.q.station == "*"):
    print(station)
    if not 'inv' in locals():
        inv = station.StationXML
    else:
        inv += station.StationXML

# Extract certain day from pyasdf
from obspy import UTCDateTime
import pyasdf
starttime = UTCDateTime(2015, 7, 31)
endtime = UTCDateTime(2015, 8, 1)
for station in ds.ifilter(ds.q.station == "RT01",
                          ds.q.channel == "EHZ",
                          ds.q.starttime >= starttime,
                          ds.q.endtime <= endtime):
    if 'st' not in locals():
        st = station.raw_recording
    else:
        st += station.raw_recording

#Read individual stationxmls to inventory
from obspy import read_inventory
from glob import glob
files = glob('/home/chet/data/GeoNet_catalog/stations/station_xml/*')
for filename in files:
    if not 'inv' in locals():
        inv = read_inventory(filename)
    else:
        inv += read_inventory(filename)

# Trim individual traces
for tr in st1:
    tr.trim(tr.stats.starttime + 0.1, tr.stats.endtime - 0.1)

# PPSD and spectra stuff
from obspy.signal import PPSD
file_root = '/home/chet/figures/NZ/network_info/'
for tr in st:
    pdf_name = file_root + 'PDFs/' + tr.stats.station + tr.stats.channel + '.png'
    tr_ppsd = PPSD(tr.stats, metadata=inv)
    tr_ppsd.add(tr)
    try:
        tr_ppsd.plot(pdf_name)
    except:
        continue
    del tr_ppsd
    # tr.spectrogram(title=str(tr.stats.station) + str(tr.stats.starttime))

# What the memory use of an obspy stream?
num_bytes = 0
for tr in st:
    num_bytes += tr.data.nbytes

# Catalog switch for match_filter
picks = [Pick(time=detecttime + (tr.stats.starttime - detecttime)]

### Testing mayavi plotting from stackoverflow
import numpy as np
from scipy import stats
from mayavi import mlab

mu=np.array([1,10,20])
# Let's change this so that the points won't all lie in a plane...
sigma=np.matrix([[20,10,10],
                 [10,25,1],
                 [10,1,50]])

data=np.random.multivariate_normal(mu,sigma,1000)
values = data.T

kde = stats.gaussian_kde(values)

# Create a regular 3D grid with 50 points in each dimension
xmin, ymin, zmin = data.min(axis=0)
xmax, ymax, zmax = data.max(axis=0)
xi, yi, zi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:50j]

# Evaluate the KDE on a regular grid...
coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
density = kde(coords).reshape(xi.shape)

# Visualize the density estimate as isosurfaces
mlab.contour3d(xi, yi, zi, density, opacity=0.5)
mlab.axes()
mlab.show()

# Pickling shiz
import cPickle
with open(r"/media/chet/hdd/seismic/NZ/clustering/nlloc_dets_Sherburn_km_mat.npy", 'rb') as f:
    dist_mat = cPickle.load(f)

with open(r"/media/chet/hdd/seismic/NZ/clustering/nlloc_dets_Sherburn_linkage.npy", 'wb') as f:
    cPickle.dump(Z, f)

catN = []
catS = []
for i, ev in enumerate(catalog):
    if ev.preferred_origin().latitude < -38.580367:
        catS.append(str(i))
    else:
        catN.append(str(i))

# Catalog error uncertainties
uncertainties = [(ev.preferred_origin().origin_uncertainty.min_horizontal_uncertainty,
                 ev.preferred_origin().origin_uncertainty.max_horizontal_uncertainty,
                 ev.preferred_origin().origin_uncertainty.azimuth_max_horizontal_uncertainty)
                 for ev in cat]

