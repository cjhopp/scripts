#!/usr/bin/env python

"""Simple script to merge a directory of catalogs"""

from glob import glob
import seaborn as sns
from obspy import read_events, Catalog

cat_dir = '/media/chet/hdd/seismic/NZ/catalogs/2015_det2cat/*'
cats = glob(cat_dir)

master_cat = Catalog()
for cat in cats:
    master_cat += read_events(cat)

# Check pick quality distribution for pick exclusion
pk_qual = [float(pk.comments[0].text.split('=')[-1]) for ev in master_cat for pk in ev.picks]