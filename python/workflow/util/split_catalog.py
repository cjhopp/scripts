#/usr/bin/env python

"""
Script to split an Obspy Catalog into manageable parts
"""

from obspy import read_events

quakeml = '/home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotnga/final_cat/rotnga_qml_nodupsATALL'
outdir = '/home/chet/data/mrp_data/sherburn_catalog/quake-ml/rotnga/final_cat/'

cat = read_events(quakeml)
#Dividing catalog into 10 parts
part_size = len(cat)/10

chunks = [cat[i:i+part_size] for i in xrange(0, len(cat), part_size)]

count = 1
for temp_cat in chunks:
    cat_name = outdir + 'rotnga_2015_part_' + str(count) + '.xml'
    temp_cat.write(outdir + 'rotnga_2015_part_' + str(count) + '.xml',
                   format="QUAKEML")
    count += 1
