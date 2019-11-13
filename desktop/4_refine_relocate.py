#!/usr/bin/env python

r"""
Script to handle pick refinement/removal and relocation of catalog earthquakes.
"""

from obspy import read, read_events, Catalog
from glob import glob
from obspy.core.event import ResourceIdentifier
# from eqcorrscan.utils import cat_util
from eqcorrscan.utils import clustering, pre_processing
from eqcorrscan.core import template_gen
from subprocess import call
import numpy as np
from obspy.io.nlloc.core import read_nlloc_hyp

# # First establish a dictionary of template waveforms
# temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/events_raw/*'
# temp_files = glob(temp_dir)
#
# template_dict = {}
# for filename in temp_files:
#     uri_name = 'smi:org.gfz-potsdam.de/geofon/' +\
#                filename.split('/')[-1].split('_')[-1].rstrip('.mseed')
#     uri = ResourceIdentifier(uri_name)
#     template_dict[uri] = read(filename)
#
# # Read in the full catalog
# cat = read_events('/home/chet/data/mrp_data/catalogs/2015/final/qml/bbox_avgarr_filt.xml')
#
# # Cluster events by distance
# groups = clustering.space_cluster(cat, d_thresh=2.0)
#
# # Run the pick refinement on each group and add events to new catalog
# refined_cat = Catalog()
# for group in groups:
#     refined_cat += cat_util.refine_picks(group, template_dict, pre_pick=0.1,
#                                          post_pick=0.9, shift_len=0.05,
#                                          cc_thresh=0.4, lowcut=1.0)

"""
Now running NLLoc from subprocess call and reading new origin back into catalog
"""
# Read catalog
cat = read_events('/Users/home/hoppche/data/2015_1sec_dets_all.xml')

origin = [-38.3724, 175.9577]


def my_conversion(x, y, z):
    new_y = origin[0] + ((y * 1000) / 111111)
    new_x = origin[1] + ((x * 1000) / (111111 * np.cos(origin[0] * (np.pi/180))))
    return new_x, new_y, z

root_name = '/Users/home/hoppche/NLLoc/mrp/obs/'
for ev in cat:
    for pk in ev.picks:
        if float(pk.comments[0].text.split('=')[-1]) < 0.30:
            # print('Removing pick below ccval threshold for %s' % str(ev.resource_id))
            ev.picks.remove(pk)
    if len(ev.picks) < 6:
        print('Fewer than 6 picks for %s. Will not locate.' % str(ev.resource_id))
        continue
    id_str = str(ev.resource_id).split('/')[-1]
    filename = root_name + id_str + '.nll'
    ev.write(filename, format="NLLOC_OBS")
    # Specify awk command to edit NLLoc .in file
    in_file = '/Users/home/hoppche/NLLoc/mrp/run/nlloc_mrp.in'
    outfile = '/Users/home/hoppche/NLLoc/mrp/loc/' + id_str
    cmnd = """awk '$1 == "LOCFILES" {$2 = "%s"; $5 = "%s"}1' %s > tmp && mv tmp %s""" % (filename, outfile, in_file, in_file)
    call(cmnd, shell=True)
    # Call to NLLoc
    call('NLLoc /Users/home/hoppche/NLLoc/mrp/run/nlloc_mrp.in', shell=True)
    # Now reading NLLoc output back into catalog as new origin
    out_w_ext = glob(outfile + '*.grid0.loc.hyp')
    new_o = read_nlloc_hyp(out_w_ext[0], coordinate_converter=my_conversion,
                           picks=ev.picks)
    ev.origins.append(new_o[0].origins[0])
    ev.preferred_origin_id = str(new_o[0].origins[0].resource_id)

# Cut templates for each new event based on new picks
for event in refined_cat:
    ev_name = str(event.resource_id).split('/')[2]
    st = template_dict[event.resource_id]
    st1 = pre_processing.shortproc(st, lowcut=1.0, highcut=20.0,
                                   filt_order=3, samp_rate=50, debug=0)
    print('Feeding stream to _template_gen...')
    template = template_gen._template_gen(event.picks, st1, length=4.0,
                                          swin='all', prepick=0.5)
    print('Writing event ' + ev_name + ' to file...')
    template.write('/media/chet/hdd/seismic/NZ/templates/rotnga_2015/' +
                   'refined_picks/' + ev_name + '_50Hz.mseed', format="MSEED")
    del st, st1, template
