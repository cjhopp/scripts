#!/usr/bin/env python

r"""
Script to handle pick refinement/removal and relocation of catalog earthquakes.
"""

import os
from glob import glob
from subprocess import call
import numpy as np
from obspy.io.nlloc.core import read_nlloc_hyp


"""
Now running NLLoc from subprocess call and reading new origin back into catalog
"""

# origin = [-38.3724, 175.9577]

def my_conversion(x, y, z):
    origin = [-38.3724, 175.9577]
    new_y = origin[0] + ((y * 1000) / 111111)
    new_x = origin[1] + ((x * 1000) / (111111 * np.cos(origin[0] * (np.pi/180))))
    return new_x, new_y, z


def relocate(cat, root_name, in_file, pick_uncertainty=0.1):
    """
    Run NonLinLoc relocations on a catalog. This is a function hardcoded for my laptop only.
    :type cat: obspy.Catalog
    :param cat: catalog of events with picks to relocate
    :type root_name: str
    :param root_name: String specifying where the nlloc.obs files will be written from the catalog
    :type outfiles: str
    :param outfiles: Output directory for location files
    :type in_file: str
    :param in_file: NLLoc input file
    :return: same catalog with new origins appended to each event
    """
    for ev in cat:
        if len(ev.picks) < 5:
            print('Fewer than 5 picks for %s. Will not locate.' % str(ev.resource_id))
            continue
        for pk in ev.picks:
            pk.time_errors.uncertainty = pick_uncertainty
        id_str = str(ev.resource_id).split('/')[-1]
        filename = root_name + 'obs/' + id_str + '.nll'
        outfile = root_name + 'loc/' + id_str
        if os.path.isfile(outfile):
            print('LOC file already written, reading output to catalog')
        else:
            ev.write(filename, format="NLLOC_OBS")
            # Specify awk command to edit NLLoc .in file
            outfile = root_name + 'loc/' + id_str
            cmnd = """awk '$1 == "LOCFILES" {$2 = "%s"; $5 = "%s"}1' %s > tmp.txt && mv tmp.txt %s""" % (filename, outfile, in_file, in_file)
            call(cmnd, shell=True)
            # Call to NLLoc
            call('NLLoc %s' % in_file, shell=True)
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
    return cat

def write_xyz(cat, outfile):
    import csv
    with open(outfile, 'wb') as f:
        writer = csv.writer(f, delimiter=' ')
        for ev in cat:
            if ev.preferred_origin():
                writer.writerow([ev.preferred_origin().latitude, ev.preferred_origin().longitude,
                                 ev.preferred_origin().depth / 1000])
