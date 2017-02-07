#!/usr/bin/env python

"""Creating catalog from pre-created NLLoc output"""
from obspy import read_events
from glob import glob
import numpy as np
from obspy.io.nlloc.core import read_nlloc_hyp

# Read catalog
cat = read_events('/media/chet/hdd/seismic/NZ/catalogs/2015_det2cat/2015_1sec_dets_all.xml')
origin = [-38.3724, 175.9577]


def my_conversion(x, y, z):
    new_y = origin[0] + ((y * 1000) / 111111)
    new_x = origin[1] + ((x * 1000) / (111111 * np.cos(origin[0] * (np.pi/180))))
    return new_x, new_y, z

root_name = '/media/chet/hdd/seismic/NZ/NLLoc/mrp/2015_dets_Sherburn/obs/'
for ev in cat:
    for pk in ev.picks:
        if float(pk.comments[0].text.split('=')[-1]) < 0.30:
            # print('Removing pick below ccval threshold for %s' % str(ev.resource_id))
            ev.picks.remove(pk)
    if len(ev.picks) < 6:
        print('Fewer than 6 picks for %s. Will not locate.' % str(ev.resource_id))
        continue
    id_str = str(ev.resource_id).split('/')[-1]
    outfile = '/media/chet/hdd/seismic/NZ/NLLoc/mrp/2015_dets_Sherburn/loc/' + id_str
    out_w_ext = glob(outfile + '.????????.??????.grid0.loc.hyp')
    try:
        new_o = read_nlloc_hyp(out_w_ext[0], coordinate_converter=my_conversion,
                               picks=ev.picks)
    except ValueError as ve:
        print(ve)
        continue
    ev.origins.append(new_o[0].origins[0])
    ev.preferred_origin_id = str(new_o[0].origins[0].resource_id)
