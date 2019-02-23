#!/usr/bin/env python

"""
Script to handle pick refinement/removal and relocation of catalog earthquakes.
"""

import os
import numpy as np

from glob import glob
from subprocess import call
from obspy import UTCDateTime
from obspy.core.event import Arrival, QuantityError, ResourceIdentifier, \
    OriginUncertainty, Origin
from obspy.geodetics import kilometer2degrees
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
            if (not pk.time_errors.upper_uncertainty
                and not pk.time_errors.uncertainty):
                sta = pk.waveform_id.station_code[:2]
                chan = pk.waveform_id.channel_code[-1]
                pk.time_errors.uncertainty = pick_uncertainty[sta][chan]
        id_str = str(ev.resource_id).split('/')[-1]
        filename = '{}/obs/{}.nll'.format(root_name, id_str)
        outfile = '{}/loc/{}'.format(root_name, id_str)
        # TODO This clause needs faster file existece check. Do 25-7.
        if os.path.isfile(filename):
            # if len(glob(outfile + '.????????.??????.grid0.loc.hyp')) > 0:
            print('LOC file already written, reading output to catalog')
        else:
            ev.write(filename, format="NLLOC_OBS")
            # Specify awk command to edit NLLoc .in file
            # Write to unique tmp file (just in_file.bak) so as not to
            # overwrite if multiple instances running.
            cmnd = """awk '$1 == "LOCFILES" {$2 = "%s"; $5 = "%s"}1' %s > %s.bak && mv %s.bak %s""" % (
                filename, outfile, in_file, in_file, in_file, in_file)
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


def dd_time2EQ(catalog, nlloc_root, in_file):
    """
    Takes a catalog with hypoDD-defined origins and populates the arrivals
    attribute for that origin using specified NLLoc Grid files through
    time2EQ

    :param catalog: Catalog containing events which we need Arrivals for
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
            print('Preferred origin not DD: {}'.format(eid))
            continue
        if len(o.arrivals) > 0:
            print('DD origin has some Arrivals. '
                  + 'Removing and adding again.')
            o.arrivals = []
        print('Raytracing for: {}'.format(eid))
        obs_file = '{}/obs/{}'.format(nlloc_root, eid)
        new_obs = '{}.obs'.format(obs_file) # Only real picks in this one
        print(new_obs)
        loc_file = '{}/loc/{}'.format(nlloc_root, eid)
        out_file_hyp = glob(
            '{}.????????.??????.grid0.loc.hyp'.format(loc_file))
        # Edit the ctrl file for both Time2EQ and NLLoc statements
        if len(out_file_hyp) == 0:
            with open(in_file, 'r') as f, open(new_ctrl, 'w') as fo:
                for line in f:
                    # Time2EQ
                    if line.startswith('EQFILES'):
                        line = line.split()
                        line = '{} {} {}\n'.format(line[0], line[1], obs_file)
                    elif line.startswith("EQSRCE"):
                        line = "EQSRCE {} LATLON {} {} {} 0.0\n".format(
                            eid, o.latitude, o.longitude, o.depth / 1000.)
                    # NLLoc
                    elif line.startswith('LOCFILES'):
                        ln = line.split()
                        line = ' '.join([ln[0], new_obs, ln[2],
                                         ln[3], loc_file])
                    fo.write(line)
            call(["Time2EQ", new_ctrl])
            # Edit obs_file to have just the Time2EQ phases for which we
            # have picks!
            # Make list of sta.phase
            sta_phz = {'{}.{}'.format(pk.waveform_id.station_code,
                                      pk.phase_hint): pk
                       for pk in ev.picks}
            # Also will add the polarities in here to eliminate separate func
            with open(obs_file, 'r') as of, open(new_obs, 'w') as nof:
                for line in of:
                    ln = line.split()
                    # Write the first line
                    if ln[0] == '#':
                        nof.write(' '.join(ln) + '\n')
                        continue
                    staph = '{}.{}'.format(ln[0], ln[4])
                    # Now only write phases we picked to the obs file
                    if staph in sta_phz:
                        if sta_phz[staph].polarity == 'positive':
                            ln[5] = 'U'
                        elif sta_phz[staph].polarity == 'negative':
                            ln[5] = 'D'
                        nof.write(' '.join(ln) + '\n')
            call(["NLLoc", new_ctrl])
            out_file_hyp = glob(
                '{}.????????.??????.grid0.loc.hyp'.format(loc_file))
            if len(out_file_hyp) == 0:
                print('No observations produced. Skip.')
                continue
        pk_stas = [pk.waveform_id.station_code for pk in ev.picks]
        # Instead of using the obspy 'read_nlloc_hyp' method, like above,
        # we'll just take the toa and dip from the phases. There was some
        # weirdness with bad microseconds being read into datetime objs
        # possibly linked to origins at 1900?
        try:
            with open(out_file_hyp[0], 'r') as f:
                for i, line in enumerate(f):
                    if (i > 15 and not line.startswith('END')
                        and not line.startswith('\n')):
                        ln = line.split()
                        pha = ln[4]
                        sta = ln[0]
                        dist = kilometer2degrees(float(ln[-6]))
                        if sta not in pk_stas:
                            continue
                        toa = ln[-3]
                        to_az = ln[-4]
                        try:
                            pk = [pk for pk in ev.picks
                                  if pk.waveform_id.station_code == sta][0]
                        except IndexError:
                            continue
                        ev.preferred_origin().arrivals.append(
                            Arrival(phase=pha, pick_id=pk.resource_id.id,
                                    takeoff_angle=toa, azimuth=to_az,
                                    distance=dist))
        except:
            print('Issue opening file. Event may not have been located')
            continue
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

############## GrowClust Functions ############################################

def hypoDD_to_GrowClust(in_dir, out_dir):
    """
    Helper to take input files from hypoDD and convert them for use with
    GrowClust

    :param in_dir: Path to the HypoDD input directory
    :param out_dir: Path to the GrowClust input directory
    :return:
    """
    # First, convert phase.dat to evlist.txt
    with open('{}/phase.dat'.format(in_dir), 'r') as in_f:
        with open('{}/evlist.txt'.format(out_dir), 'w') as out_f:
            for ln in in_f:
                if ln.startswith('#'):
                    out_f.write('{}\n'.format(' '.join(ln.split()[1:])))
    # Now remove occurrences of network string from dt.cc and write to
    # xcordata.txt (use sed via system call as much faster)
    sed_str = "sed 's/NZ.//g' {}/dt.cc > {}/xcordata.txt".format(in_dir,
                                                                 out_dir)
    call(sed_str, shell=True)
    return

def GrowClust_to_Catalog(hypoDD_cat, out_dir):
    """
    Take the original catalog used in generating dt's with HypoDDpy and read
    the output of GrowClust into the appropriate events as new origins.

    This is probably going to borrow heavily from hypoDDpy...
    :param hypoDD_cat: Same catalog used in hypoDDpy to generate dt's
    :param out_dir: GrowClust output directory
    :return:
    """
    # Catalog is sorted by time in hypoDDpy before event map is generated
    hypoDD_cat.events.sort(key=lambda x: x.preferred_origin().time)
    new_o_map = {}
    with open('{}/out.growclust_cat'.format(out_dir), 'r') as f:
        for ln in f:
            ln.strip()
            line = ln.split()
            # First determine if it was relocated
            # Default is line[19] == -1 for no, but also should beware of
            # unceratintites of 0.000. Deal with these later?
            eid = int(line[6]) # Event id before clustering
            if line[13] == '1' and line[19] == '-1.000':
                print('Event {} not relocated, keep original location'.format(
                    eid
                ))
                continue
            re_lat = float(line[7])
            re_lon = float(line[8])
            re_dep = float(line[9]) * 1000 # meters bsl
            x_uncert = float(line[19]) * 1000 # in m
            z_uncert = float(line[20]) * 1000 # in m
            t_uncert = float(line[21])
            o_uncert = OriginUncertainty(horizontal_uncertainty=x_uncert)
            t_uncert = QuantityError(uncertainty=t_uncert)
            d_uncert = QuantityError(uncertainty=z_uncert)
            sec = int(line[5].split('.')[0])
            microsec = int(line[5].split('.')[1]) * 1000
            method_id = ResourceIdentifier(id='GrowClust')
            re_time = UTCDateTime(year=int(line[0]), month=int(line[1]),
                                  day=int(line[2]), hour=int(line[3]),
                                  minute=int(line[4]), second=sec,
                                  microsecond=microsec)
            new_o_map[eid] = Origin(time=re_time, latitude=re_lat,
                                    longitude=re_lon, depth=re_dep,
                                    time_errors=t_uncert,
                                    depth_errors=d_uncert,
                                    origin_uncertainty=o_uncert,
                                    method_id=method_id)
    for i, ev in enumerate(hypoDD_cat):
        id = i + 1 # Python indexing
        if id in new_o_map:
            ev.origins.append(new_o_map[id])
            ev.preferred_origin_id = new_o_map[id].resource_id.id
    return hypoDD_cat