#!/usr/bin/env python

"""
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
    new_x = origin[1] + ((x * 1000) /
                         (111111 * np.cos(origin[0] * (np.pi/180))))
    return new_x, new_y, z

def relocate(cat, root_name, in_file, pick_uncertainty):
    """
    Run NonLinLoc relocations on a catalog. This is a function hardcoded for
    my laptop only.
    :type cat: obspy.Catalog
    :param cat: catalog of events with picks to relocate
    :type root_name: str
    :param root_name: String specifying where the nlloc.obs files will be
        written from the catalog
    :type in_file: str
    :param in_file: NLLoc input file
    :type pick_uncertainty: dict
    :param pick_uncertainty: Dictionary mapping uncertainties to sta/chans
    :return: same catalog with new origins appended to each event
    """
    for ev in cat:
        if len(ev.picks) < 5:
            print('Fewer than 5 picks for {}. Will not locate.'.format(
                ev.resource_id.id))
            continue
        for pk in ev.picks:
            if not pk.time_errors.upper_uncertainty:
                sta = pk.waveform_id.station_code[:2]
                chan = pk.waveform_id.channel_code[-1]
                pk.time_errors.uncertainty = pick_uncertainty[sta][chan]
        id_str = str(ev.resource_id).split('/')[-1]
        filename = root_name + 'obs/' + id_str + '.nll'
        outfile = root_name + 'loc/' + id_str
        # TODO This clause needs faster file existece check. Do 25-7.
        if os.path.isfile(outfile):
            if len(glob(outfile + '.????????.??????.grid0.loc.hyp')) > 0:
                print('LOC file already written, reading output to catalog')
        else:
            ev.write(filename, format="NLLOC_OBS")
            # Specify awk command to edit NLLoc .in file
            cmnd = """awk '$1 == "LOCFILES" {$2 = "%s"; $5 = "%s"}1' %s > tmp.txt && mv tmp.txt %s""" % (filename, outfile, in_file, in_file)
            call(cmnd, shell=True)
            # Call to NLLoc
            call('NLLoc %s' % in_file, shell=True)
        # Now reading NLLoc output back into catalog as new origin
        # XXX BE MORE CAREFUL HERE. CANNOT GRAB BOTH SUM AND NON-SUM
        out_w_ext = glob(outfile + '.????????.??????.grid0.loc.hyp')
        try:
            new_o = read_nlloc_hyp(out_w_ext[0],
                                   coordinate_converter=my_conversion,
                                   picks=ev.picks)
        except ValueError as ve:
            print(ve)
            continue
        ev.origins.append(new_o[0].origins[0])
        ev.preferred_origin_id = str(new_o[0].origins[0].resource_id)
    return cat


def hypoDD_time2EQ(catalog, nlloc_root, in_file):
    """
    Takes a catalog with hypoDD-defined origins and populates the arrivals
    attribute for that origin using specified NLLoc Grid files through
    time2EQ

    :param catalog: Catalog containing evnets which we need Arrivals for
    :param nlloc_root: Root directory for file IO
    :param in_file: NLLoc/Time2EQ run file. User is responsible for defining
        the path to grid files in this control file. This file will be modified
        in-place as this function runs.
    :return:
    """
    # Temp ctrl file overwritten each iteration
    new_ctrl = '{}.new'.format(in_file)
    for ev in catalog:
        eid = ev.resource_id.id.split('/')[-1]
        o = ev.preferred_origin()
        if not o or not o.method_id:
            print('Preferred origin not hypoDD: {}'.format(eid))
            continue
        if len(o.arrivals) > 0:
            print('HypoDD origin has some Arrivals. Moving on.')
            continue
        print('Raytracing for: {}'.format(eid))
        obs_file = '{}/obs/{}'.format(nlloc_root, eid)
        loc_file = '{}/loc/{}'.format(nlloc_root, eid)
        with open(in_file, 'r') as f, open(new_ctrl, 'w') as fo:
            for line in f:
                if line.startswith('EQFILES'):
                    line = line.split()
                    line = '{} {} {}'.format(line[0], line[1], obs_file)
                elif line.startswith("EQSRCE"):
                    line = "EQSRCE {} LATLON {} {} {} 0.0\n".format(
                        eid, o.latitude, o.longitude, o.depth / 1000.)
                elif line.startswith('LOCFILES'):
                    ln = line.split()
                    line = ' '.join([ln[0], obs_file, ln[2], ln[3], loc_file])
                fo.write(line)
        call(["Time2EQ", new_ctrl])
        call(["NLLoc", new_ctrl])
        out_file_hyp = glob(
            '{}.????????.??????.grid0.loc.hyp'.format(loc_file))
        new_o = read_nlloc_hyp(out_file_hyp[0],
                               coordinate_converter=my_conversion,
                               picks=ev.picks)
        print(new_o)
        ev.preferred_origin().arrivals = new_o[0].origins[0].arrivals
    return


def write_xyz(cat, outfile):
    import csv
    with open(outfile, 'wb') as f:
        writer = csv.writer(f, delimiter=' ')
        for ev in cat:
            if ev.preferred_origin():
                writer.writerow([ev.preferred_origin().latitude,
                                 ev.preferred_origin().longitude,
                                 ev.preferred_origin().depth / 1000])