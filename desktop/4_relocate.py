#!/usr/bin/env python

r"""
Script to handle pick refinement/removal and relocation of catalog earthquakes.
"""

from obspy import read_events
from glob import glob
from subprocess import call
import numpy as np
from obspy.io.nlloc.core import read_nlloc_hyp


"""
Now running NLLoc from subprocess call and reading new origin back into catalog
"""
# Read catalog
# cat = read_events('/media/chet/hdd/seismic/NZ/catalogs/2015_det2cat/2015_1sec_dets_all.xml')
# test_cat = cat[:100].copy()
origin = [-38.3724, 175.9577]


def my_conversion(x, y, z):
    new_y = origin[0] + ((y * 1000) / 111111)
    new_x = origin[1] + ((x * 1000) / (111111 * np.cos(origin[0] * (np.pi/180))))
    return new_x, new_y, z

root_name = '/media/chet/hdd/seismic/NZ/NLLoc/mrp/2015_Rawlinson_reweighted/obs/'
for ev in cat_rewt:
    # for pk in ev.picks:
        # if float(pk.comments[0].text.split('=')[-1]) < 0.40:
        #     # print('Removing pick below ccval threshold for %s' % str(ev.resource_id))
        #     ev.picks.remove(pk)
        # pk.time_errors.uncertainty = 1 - float(pk.comments[0].text.split('=')[-1])
        # pk.time_errors.uncertainty = 0.01
    if len(ev.picks) < 8:
        print('Fewer than 8 picks for %s. Will not locate.' % str(ev.resource_id))
        continue
    id_str = str(ev.resource_id).split('/')[-1]
    filename = root_name + id_str + '.nll'
    ev.write(filename, format="NLLOC_OBS")
    # Specify awk command to edit NLLoc .in file
    in_file = '/home/chet/NLLoc/mrp/run/nlloc_mrp.in'
    outfile = '/media/chet/hdd/seismic/NZ/NLLoc/mrp/2015_Rawlinson_reweighted/loc/' + id_str
    cmnd = """awk '$1 == "LOCFILES" {$2 = "%s"; $5 = "%s"}1' %s > tmp && mv tmp %s""" % (filename, outfile, in_file, in_file)
    call(cmnd, shell=True)
    # Call to NLLoc
    call('NLLoc /home/chet/NLLoc/mrp/run/nlloc_mrp.in', shell=True)
    # Now reading NLLoc output back into catalog as new origin
    # XXX BE MORE CAREFUL HERE. CANNOT GRAB BOTH SUM AND NON-SUM
    out_w_ext = glob(outfile + '.????????.??????.grid0.loc.hyp')
    try:
        new_o = read_nlloc_hyp(out_w_ext[0], coordinate_converter=my_conversion,
                               picks=ev.picks)
    except ValueError as ve:
        print(ve)
        continue
    ev.origins.append(new_o[0].origins[0])
    ev.preferred_origin_id = str(new_o[0].origins[0].resource_id)

cat_rewt.write('/media/chet/hdd/seismic/NZ/catalogs/2015_dets_nlloc/2015_nlloc_Rawlinson_reweight.xml', format='QUAKEML')
cat_rewt.write('/media/chet/hdd/seismic/NZ/catalogs/2015_dets_nlloc/2015_nlloc_Rawlinson_reweight.shp', format='SHAPEFILE')

import csv
with open('/media/chet/hdd/seismic/NZ/catalogs/2015_dets_nlloc/2015_dets_nlloc_Sherburn_nodups.xyz', 'wb') as f:
    writer = csv.writer(f, delimiter=' ')
    for ev in cat:
        if ev.preferred_origin():
            writer.writerow([ev.preferred_origin().latitude, ev.preferred_origin().longitude, ev.preferred_origin().depth / 1000])
