#!/usr/bin/python

"""
Functions for the processing, reading, converting of events and event files

IMPORTANT
***********************************************
Arbitrary zero depth point is elev = 130 m
***********************************************
"""

import io
import os
import re
import shutil
import pyproj

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import dask.dataframe as dd

from glob import glob
from itertools import cycle
from subprocess import call
from scipy.linalg import lstsq
from dask import delayed
from sklearn.cluster import DBSCAN
try:
    from mplstereonet import StereonetAxes
except ImportError:
    print('No mplstereonet. Not plotting stereonets')
    StereonetAxes = None
from obspy import UTCDateTime, Catalog, read, read_inventory, Stream, Trace,\
    read_events, Inventory
from obspy.core.util import AttribDict
from obspy.core.event import Pick, Origin, Arrival, Event, Magnitude,\
    WaveformStreamID, ResourceIdentifier, OriginQuality, OriginUncertainty,\
    QuantityError
from obspy.clients.fdsn import Client

## Locals
from lbnl.coordinates import SURF_converter
from lbnl.boreholes import depth_to_xyz, parse_surf_boreholes
try:
    from libcomcat.exceptions import ContentNotFoundError, ProductNotFoundError
    from libcomcat.search import get_event_by_id
    from libcomcat.dataframes import get_phase_dataframe, get_detail_data_frame
except ModuleNotFoundError:
    print('No libcomcat. Dont import from usgs')
try:
    from lbnl.waveforms import rotate_channels
except ImportError as e:
    print('Not dependencies for rotate_channels. Youre in the wrong env')

three_comps = ['OB13', 'OB15', 'OT16', 'OT18', 'PDB3', 'PDB4', 'PDB6', 'PDT1',
               'PSB7', 'PSB9', 'PST10', 'PST12']


# Planes and shiz taken from workflow.focal_mecs
def cluster_catalog(catalog):
    # Assuming Newberry right now, allow argument for this later
    utm = pyproj.Proj("EPSG:32610")
    cmap = sns.color_palette('deep', 10)
    cycler = cycle(cmap)
    locs = np.array([(ev.preferred_origin().longitude, ev.preferred_origin().latitude, ev.preferred_origin().depth)
                     for ev in catalog])
    east, north = utm(locs[:, 0], locs[:, 1])
    XYZ = np.vstack([east, north, locs[:, 2]]).T
    db = DBSCAN(eps=20, min_samples=5).fit(XYZ)
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_axes(rect=[0.1, 0.5, 0.8, 0.4], projection='3d')
    ax_stereo = StereonetAxes(rect=[0.1, 0.1, 0.8, 0.4], fig=fig)
    fig.add_axes(ax_stereo)
    ax.scatter(xs=XYZ[:, 0], ys=XYZ[:, 1], zs=XYZ[:, 2], c=db.labels_, s=5.)
    for lab in set(db.labels_):
        color = next(cycler)
        if lab == -1:
            continue
        cluster_mask = db.labels_ == lab
        xyz = XYZ[cluster_mask]
        Xs, Ys, Zs, strike, dip = pts_to_plane(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        print(strike, dip)
        ax.plot_surface(Xs, Ys, Zs, color=color)
        ax_stereo.plane(strike, dip, color=color)
        ax_stereo.pole(strike, dip, color=color)
    ax.axis('equal')
    ax_stereo.grid(True)
    plt.show()
    return db


def pts_to_plane(x, y, z, method='lstsq'):
    # Create a grid over the desired area
    # Here just define it over the x and y range of the cluster (100 pts)
    x_ran = max(x) - min(x)
    y_ran = max(y) - max(y)
    if method == 'lstsq':
        # Add 20 percent to x and y dimensions for viz purposes
        X, Y = np.meshgrid(np.arange(min(x) - (0.2 * x_ran),
                                     max(x) + (0.2 * x_ran),
                                     (max(x) - min(x)) / 10.),
                           np.arange(min(y) - (0.2 * y_ran),
                                     max(y) + (0.2 * y_ran),
                                     (max(y) - min(y)) / 10.))
        # Now do the linear fit and generate the coefficients of the plane
        A = np.c_[x, y, np.ones(len(x))]
        C, _, _, _ = lstsq(A, z)  # Coefficients (also the vector normal?)
    elif method == 'svd':
        print('SVD not implemented')
        return
    # Evaluate the plane for the points of the grid
    Z = C[0] * X + C[1] * Y + C[2]
    # strike and dip
    pt1 = (X[0][2], Y[0][2], Z[0][2])
    pt2 = (X[3][1], Y[3][1], Z[3][1])
    pt3 = (X[0][0], Y[0][0], Z[0][0])
    strike, dip = strike_dip_from_pts(pt1, pt2, pt3)
    return X, Y, Z, strike, dip


def pts_to_ellipsoid(x, y, z):
    # Function from:
    # https://github.com/aleksandrbazhin/ellipsoid_fit_python/blob/master/ellipsoid_fit.py
    # http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
    # for arbitrary axes
    D = np.array([x * x,
                  y * y,
                  z * z,
                  2 * x * y,
                  2 * x * z,
                  2 * y * z,
                  2 * x,
                  2 * y,
                  2 * z])
    DT = D.conj().T
    v = np.linalg.solve(D.dot(DT), D.dot(np.ones(np.size(x))))
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], -1]])
    center = np.linalg.solve(- A[:3, :3], [[v[6]], [v[7]], [v[8]]])
    T = np.eye(4)
    T[3, :3] = center.T
    R = T.dot(A).dot(T.conj().T)
    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    radii = np.sqrt(1. / np.abs(evals)) # Absolute value to eliminate imaginaries?
    return center, radii, evecs, v


def strike_dip_from_pts(pt1, pt2, pt3):
    # Take the output from the best fit plane and calculate strike and dip
    vec_1 = np.array(pt3) - np.array(pt1)
    vec_2 = np.array(pt3) - np.array(pt2)
    U = np.cross(vec_1, vec_2)
    # Standard rectifying for right-hand rule
    if U[2] < 0:
        easting = U[1]
        northing = -U[0]
    else:
        easting = -U[1]
        northing = U[0]
    if easting >= 0:
        partA_strike = easting**2 + northing**2
        strike = np.rad2deg(np.arccos(northing / np.sqrt(partA_strike)))
    else:
        partA_strike = northing / np.sqrt(easting**2 + northing**2)
        strike = 360. - np.rad2deg(np.arccos(partA_strike))
    part1_dip = np.sqrt(U[1]**2 + U[0]**2)
    part2_dip = np.sqrt(part1_dip**2 + U[2]**2)
    dip = np.rad2deg(np.arcsin(part1_dip / part2_dip))
    return strike, dip


def parse_filenames_to_eid(path, method='SURF', cassm=False):
    if method == 'cascadia':
        name = path.split('/')[-1].rstrip('.ms')
    else:
        if cassm:
            str_int = 1
        else:
            str_int = 0
        name = path.split('/')[-1].split('_')[str_int]
    return name


def parse_resource_id_to_eid(ev, method='SURF'):
    if method == 'cascadia':
        name = ev.resource_id.id.split('=')[-2].split('&')[0]
    else:
        name = ev.resource_id.id.split('/')[-1]
    return name


def cat_dist_to_notch(
        cat, well, depth, starttime, endtime,
        well_file='data/chet-collab/boreholes/surf_4850_wells.csv'):
    """
    Return dict of distances to a notch for a catalog.

    Used as an argument for plotly_timeseries

    :param cat: Catalog object
    :param well: String of well
    :param depth: Depth of notch in well
    :param starttime: Start time of catalog
    :param endtime: End time of catalog
    :return:
    """
    times = [ev.origins[0].time.datetime for ev in cat
             if starttime < ev.origins[-1].time < endtime]
    xyzs = [(float(ev.origins[-1].extra.hmc_east.value),
             float(ev.origins[-1].extra.hmc_north.value),
             float(ev.origins[-1].extra.hmc_elev.value))
             for ev in cat if starttime < ev.origins[-1].time < endtime]
    well_dict = parse_surf_boreholes(well_file)
    notch = depth_to_xyz(well_dict, well, depth)
    dists = [np.sqrt((p[0] - notch[0])**2 + (p[1] - notch[1])**2 +
                     (p[2] - notch[2])**2) for p in xyzs]
    return {'times': times, 'dists': dists}


def clean_cat_picks(cat, PS_one_chan=True):
    """
    Clean irregularities in a catalog before it is used for tribe creation

    Works in-place on catalog

    :param cat: obspy.core.Catalog
    :param PS_one_chan: Allow P and S picks on same channel?

    :return:
    """
    for ev in cat:
        # Remove non-obspyck picks
        ev.picks = [pk for pk in ev.picks if pk.creation_info]
        # Allow P and S on same channel?
        if not PS_one_chan:
            used_seeds = []
            pks = []
            for pk in ev.picks:
                if '{}.{}.{}.{}'.format(
                        pk.waveform_id.network_code,
                        pk.waveform_id.station_code,
                        pk.waveform_id.location_code,
                        pk.waveform_id.channel_code) in used_seeds:
                    continue
                else:
                    used_seeds.append('{}.{}.{}.{}'.format(
                        pk.waveform_id.network_code,
                        pk.waveform_id.station_code,
                        pk.waveform_id.location_code,
                        pk.waveform_id.channel_code))
                    pks.append(pk)
            ev.picks = pks
        # Now remove same phases on single channel
        used_seeds = []
        pks = []
        for pk in ev.picks:
            if '{}.{}.{}.{}.{}'.format(
                    pk.waveform_id.network_code,
                    pk.waveform_id.station_code,
                    pk.waveform_id.location_code,
                    pk.waveform_id.channel_code,
                    pk.phase_hint) in used_seeds:
                continue
            else:
                used_seeds.append('{}.{}.{}.{}.{}'.format(
                    pk.waveform_id.network_code,
                    pk.waveform_id.station_code,
                    pk.waveform_id.location_code,
                    pk.waveform_id.channel_code,
                    pk.phase_hint))
                pks.append(pk)
        ev.picks = pks
    return cat


def ORNL_events_to_cat(ornl_file):
    """Make Catalog from ORNL locations"""
    cat = Catalog()
    loc_df = pd.read_csv(ornl_file, infer_datetime_format=True)
    loc_df = loc_df.set_index('event_datetime')
    eid = 0
    for dt, row in loc_df.iterrows():
        ot = UTCDateTime(dt)
        hmc_east = row['x(m)']
        hmc_north = row['y(m)']
        hmc_elev = row['z(m)']
        errX = row['error_x (m)']
        errY = row['error_y (m)']
        errZ = row['error_z (m)']
        rms = row['rms (millisecond)']
        converter = SURF_converter()
        lon, lat, elev = converter.to_lonlat((hmc_east, hmc_north,
                                              hmc_elev))
        o = Origin(time=ot, latitude=lat, longitude=lon, depth=130 - elev)
        o.origin_uncertainty = OriginUncertainty()
        o.quality = OriginQuality()
        ou = o.origin_uncertainty
        oq = o.quality
        ou.max_horizontal_uncertainty = np.max([errX, errY])
        ou.min_horizontal_uncertainty = np.min([errX, errY])
        o.depth_errors.uncertainty = errZ
        oq.standard_error = rms * 1e3
        extra = AttribDict({
            'hmc_east': {
                'value': hmc_east,
                'namespace': 'smi:local/hmc'
            },
            'hmc_north': {
                'value': hmc_north,
                'namespace': 'smi:local/hmc'
            },
            'hmc_elev': {
                'value': hmc_elev,
                'namespace': 'smi:local/hmc'
            },
            'hmc_eid': {
                'value': eid,
                'namespace': 'smi:local/hmc'
            }
        })
        o.extra = extra
        rid = ResourceIdentifier(id=ot.strftime('%Y%m%d%H%M%S%f'))
        # Dummy magnitude of 1. for all events until further notice
        mag = Magnitude(mag=1., mag_errors=QuantityError(uncertainty=1.))
        ev = Event(origins=[o], magnitudes=[mag], resource_id=rid)
        ev.preferred_origin_id = o.resource_id.id
        cat.events.append(ev)
        eid += 1
    return cat


def surf_events_to_cat(loc_file, pick_file):
    """
    Take location files (hypoinverse formatted) and picks (format TBD)
    and creates a single obspy catalog for later use and dissemination.

    :param loc_file: File path
    :param pick_file: File path
    :return: obspy.core.Catalog
    """
    # Read/parse location file and create Events for each
    surf_cat = Catalog()
    # Parse the pick file to a dictionary
    pick_dict = parse_picks(pick_file)
    with open(loc_file, 'r') as f:
        next(f)
        for ln in f:
            ln = ln.strip('\n')
            line = ln.split(',')
            eid = line[0]
            if eid not in pick_dict:
                print('No picks for this location, skipping for now.')
                continue
            ot = UTCDateTime(line[1])
            hmc_east = float(line[2])
            hmc_north = float(line[3])
            hmc_elev = float(line[4])
            gap = float(line[-5])
            rms = float(line[-3])
            errXY = float(line[-2])
            errZ = float(line[-1])
            converter = SURF_converter()
            lon, lat, elev = converter.to_lonlat((hmc_east, hmc_north,
                                                  hmc_elev))
            o = Origin(time=ot, longitude=lon, latitude=lat, depth=130 - elev)
            o.origin_uncertainty = OriginUncertainty()
            o.quality = OriginQuality()
            ou = o.origin_uncertainty
            oq = o.quality
            ou.horizontal_uncertainty = errXY * 1e3
            ou.preferred_description = "horizontal uncertainty"
            o.depth_errors.uncertainty = errZ * 1e3
            oq.standard_error = rms
            oq.azimuthal_gap = gap
            extra = AttribDict({
                'hmc_east': {
                    'value': hmc_east,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_north': {
                    'value': hmc_north,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_elev': {
                    'value': hmc_elev,
                    'namespace': 'smi:local/hmc'
                },
                'hmc_eid': {
                    'value': eid,
                    'namespace': 'smi:local/hmc'
                }
            })
            o.extra = extra
            rid = ResourceIdentifier(id=ot.strftime('%Y%m%d%H%M%S%f'))
            # Dummy magnitude of 1. for all events until further notice
            mag = Magnitude(mag=1., mag_errors=QuantityError(uncertainty=1.))
            ev = Event(origins=[o], magnitudes=[mag],
                       picks=pick_dict[eid], resource_id=rid)
            surf_cat.append(ev)
    return surf_cat


def parse_picks(pick_file):
    """
    Helper for parsing file with pick information

    :param pick_file: Path to the file
    :return: Dictionary of pick info: {eid: {sta.chan: list of picks}}
    """
    pick_dict = {}
    with open(pick_file, 'r') as f:
        next(f)
        for ln in f:
            ln = ln.strip('\n')
            line = ln.split(',')
            eid = line[0]
            time = UTCDateTime(line[-3])
            phase = line[-2]
            # Placing these per standard convention but not true!!
            # TODO Maybe these should all be Z? Careful when doing correlations
            if line[2] not in three_comps: # Hydrophone channels
                chan = 'XN1'
            elif phase == 'P':
                chan = 'XNZ'
            else:
                chan = 'XNX'
            snr = float(line[-1])
            if snr == 0.:
                method = 'manual'
            else:
                method = 'phasepapy'
            wf_id = WaveformStreamID(network_code='SV', station_code=line[2],
                                     location_code='', channel_code=chan)
            pk = Pick(time=time, method=method, waveform_id=wf_id,
                      phase_hint=phase)
            if eid not in pick_dict:
                pick_dict[eid] = [pk]
            else:
                pick_dict[eid].append(pk)
    return pick_dict


def martin_cassm_to_loc_hyp(cassm_dir):
    """
    One-time use (probably) to take a number of files and output one location
    and one pick file for the cassm sources which can then by input into
    surf_events_to_cat()
    """
    # First get ids of only cassm events from catalog
    cassm_mat = np.genfromtxt('{}/cassm_events_mat.txt'.format(cassm_dir),
                              skip_header=1, delimiter=',', dtype='str')
    cassm_ids = cassm_mat[:, 0]
    # Now put hyp origin times for each cassm event into keys of dict with eid
    # as value
    hyp_dict = {}
    new_loc_dict = {}
    with open(glob('{}/hyp_vibbox*'.format(cassm_dir))[0], 'r') as f:
        for ln in f:
            line = ln.split()
            if len(line) == 6 and line[-1] in cassm_ids:
                time_str = '{}-{}-{}T{}:{}:{}.{}'.format(
                    line[0][:4], line[0][4:6], line[0][6:8], line[0][8:10],
                    line[0][10:12], line[0][12:14], line[0][14])
                time_str_full = '{}-{}-{}T{}:{}:{}.{}'.format(
                    line[0][:4], line[0][4:6], line[0][6:8], line[0][8:10],
                    line[0][10:12], line[0][12:14], line[0][14:20])
                hyp_dict[time_str] = line[-1]
                new_loc_dict[line[-1]] = time_str_full
    # Now write new loc file with corrected times
    with open('{}/cassm_events.loc'.format(cassm_dir), 'w') as outf:
        for row in cassm_mat:
            row = list(row)
            row[1] = new_loc_dict[row[0]]
            row[-1] += '\n'
            outf.write(','.join(row))
    # Now build array of lines for new pick file
    new_pick_lines = []
    with open(glob('{}/picks_auto*'.format(cassm_dir))[0], 'r') as pf:
        for ln in pf:
            line = ln.split()
            if line[1][:-6] in hyp_dict:
                # Edit station info and eid for this line
                line[0] = hyp_dict[line[1][:-6]]
                # Replace the different naming convention for stations
                line[2] = line[2].replace('_A.', '').replace('_H.', '')
                new_pick_lines.append(line)
    with open('{}/picks_cassm.txt'.format(cassm_dir), 'w') as of:
        # Add header (even if we'll ignore it later)
        of.write('eventid,origin,station,pick,phase,snr\n')
        for out_line in new_pick_lines:
            of.write(','.join(out_line) + '\n')
    return


def add_pols_to_Time2EQ_hyp(catalog, nlloc_dir, outdir, hydrophones=False):
    """
    Add polarities to the nlloc hyp files produced from Time2EQ. This is the
    last part of the workflow which takes hypoDD locations, retraces the
    raypaths with Time2EQ, relocates these with NLLoc and then repopulates
    the PHASE lines in the .hyp file with the polarities picked in Obspyck
    (this function). These are then fed into the Arnold focmec stuff.

    :param catalog: Catalog with polarity picks to use
    :param nlloc_dir: Path to the NLLoc loc/ directory with corresponding
        location files for the catalog provided
    :param outdir: Path to output directory for the .scat, .hdr and .hyp files
        Usually this will be in an Arnold_Townend projects/ directory
    :param hydrophones: Whether to include polarities measured on hydrophones.
        Defaults to False as I'm not sure how to handle these yet.
    :return:
    """
    for ev in catalog:
        print('{}'.format(str(ev.resource_id).split('/')[-1]))
        nlloc_fs = glob('{}/{}*'.format(
            nlloc_dir,
            str(ev.resource_id).split('/')[-1].split('_')[0]))
        try:
            hyp_path = [path for path in nlloc_fs
                        if path.endswith('.hyp')
                        and 'sum' not in path.split('.')][0]
        except IndexError as msg:
            print('No NLLoc location for this event. Probably low SNR?')
            continue
        print(hyp_path)
        # Move hdr and scat files to outdir
        scat_hdr = [path for path in nlloc_fs
                    if (path.endswith('.hdr')
                        or path.endswith('.scat'))
                    and 'sum' not in path.split('.')]
        for fl in scat_hdr:
            shutil.copyfile(fl, '{}/{}'.format(outdir, fl.split('/')[-1]))
        # Now edit the loc file and write it to outdir
        with open(hyp_path, 'r') as orig:
            with open('{}/{}'.format(outdir,
                                     hyp_path.split('/')[-1]), 'w') as new:
                phase = False
                for ln in orig:
                    line = ln.rstrip()
                    line = line.split()
                    if len(line) == 0:
                        print('End of file')
                        break
                    # Write top of file as usual until we get to PHASE lines
                    if line[0] == 'PHASE':
                        phase = True
                        new.write(' '.join(line) + '\n')
                        continue
                    elif line[0] == 'END_PHASE':
                        phase = False
                    if phase:
                        # Skip all the S phases
                        if line[4] == 'S':
                            print('Ignore S phases')
                            new.write(' '.join(line) + '\n')
                            continue
                        # If hydrophone == False, don't bother with those
                        if not hydrophones and line[0] not in three_comps:
                            new.write(' '.join(line) + '\n')
                            continue
                        # Try to find a corresponding polarity pick in catalog
                        # Because P and S traced to all stations, we find only
                        # phase lines corresponding to actual picks in the
                        # catalog and populate the FM column. These will be the
                        # only ones used by the Focal mech package anyways.
                        print('Try adding for {}'.format(line[0]))
                        try:
                            pk = [pk for pk in ev.picks
                                  if pk.waveform_id.station_code == line[0]
                                  and line[4] == 'P'][0]
                        except IndexError:
                            print('No polarity pick for {}'.format(line[0]))
                            new.write(' '.join(line) + '\n')
                            continue
                        if pk.polarity not in ['positive', 'negative']:
                            print('No polarity for station {}'.format(line[0]))
                            new.write(' '.join(line) + '\n')
                            continue
                        if pk.polarity == 'positive':
                            line[5] = 'U'
                        elif pk.polarity == 'negative':
                            line[5] = 'D'
                    new.write(' '.join(line) + '\n')
    return


def obspyck_from_local(config_file, inv_paths, location, wav_dir=None,
                       catalog=None, wav_file=None, cassm=False, rotate=False,
                       length=0.03, prepick=0.003, pick_error=0.0001):
    """
    Function to take local catalog, inventory and waveforms for picking.

    This has been gutted from scripts.python.workflow.obspyck_util for use
    with SURF/FS-B networks.

    :param inv: list of paths to StationXML files
    :param wav_dir: Directory of mseeds named according to timestamp
        eid convention
    :param catalog: catalog of events to pick (optional)
    :param wav_file: If not passing a directory, pass single waveform file path
    :param cassm: Bool for string parsing of cassm event files
    :param rotate: If orientation information is saved in the inventory,
        rotate the channels into ZNE. Defaults to False.
    :param length: Length (seconds) of wave to plot
    :param prepick: Seconds before pick of wav to plot
    :param pick_error: Default pick error to assign if none exists

    :return:
    """

    # Sort network name
    if location == 'cascadia':
        net = 'UW'
    elif location == '4100':
        net = 'CB'
    elif location == 'fsb':
        net = 'FS'
    else:
        net = 'SV'
    # Grab all stationxml files
    inv = Inventory()
    for inv_f in inv_paths:
        inv += read_inventory(inv_f)
    # For the case of a single wav file with no catalog (probably a stack)
    if not catalog and wav_file:
        st = read(wav_file)
        st.traces.sort(key=lambda x: x.stats.starttime) # sort first
        utcdto = st[0].stats.starttime
        root = ['obspyck -c {} -t {} -d {} -s {}'.format(config_file,
                                                         utcdto - prepick,
                                                         length, net)]
        cmd = ' '.join(root + [wav_file] + inv_paths)
        print(cmd)
        call(cmd, shell=True)
        return
    all_wavs = glob('{}/*'.format(wav_dir))
    # Sort events, although they should already be sorted and it doesnt matter
    catalog.events.sort(key=lambda x: x.origins[-1].time)
    if len(catalog) == 0:
        print('No events in catalog')
        return
    ## Old workflow for SURF 4850
    # eids = [parse_resource_id_to_eid(ev, method=location) for ev in catalog]
    # wav_files = [
    #     p for p in all_wavs
    #     if parse_filenames_to_eid(p, method=location, cassm=cassm) in eids]
    eids = [ev.resource_id.id.split('/')[-1] for ev in catalog]
    wav_files = [f for f in all_wavs if f.split('/')[-1].rstrip('.ms') in eids]
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    for ev in catalog:
        pk1 = min([pk.time for pk in ev.picks])
        # eid = parse_resource_id_to_eid(ev, method=location)
        # wav_file = [
        #     f for f in wav_files if parse_filenames_to_eid(f, method=location,
        #                                                    cassm=cassm) == eid]
        eid = ev.resource_id.id.split('/')[-1]
        wav_file = [f for f in wav_files
                    if f.split('/')[-1].rstrip('.ms') == eid]
        # Create temporary mseed without the superfluous non-seis traces
        try:
            st = read(wav_file[0])
        except IndexError as e:
            print('No waveform for this event')
            continue
        # Vibbox specific channels, not for picking
        rms = [tr for tr in st
               if tr.stats.station in ['CMon', 'CTrg', 'CEnc', 'PPS']]
        for rm in rms:
            st.traces.remove(rm)
        tmp_wav_file = ['tmp/tmp_wav.mseed']
        if rotate:
            # Rotate to ZNE not in obspyck so do it here.
            rotated_st = rotate_channels(st, inv)
            rotated_st.write(tmp_wav_file[0], format="MSEED")
        else:
            st.write(tmp_wav_file[0], format="MSEED")
        # If not pick uncertainties, assign some arbitrary ones
        for pk in ev.picks:
            if not pk.time_errors:
                pk.time_errors.uncertainty = pick_error
        tmp_name = 'tmp/{}_repicked.xml'.format(eid)
        # tmp_name = 'tmp/{}.xml'.format(
        #     parse_resource_id_to_eid(ev, method=location))
        ev.write(tmp_name, format='QUAKEML')
        print('Launching obspyck for ev: {}' .format(
              str(ev.resource_id).split('/')[-1]))
        root = ['obspyck -c {} -t {} -d {} -s {} --event {}'.format(
            config_file, pk1 - prepick, length, net, tmp_name)]
        cmd = ' '.join(root + tmp_wav_file + inv_paths)
        print(cmd)
        call(cmd, shell=True)
    return


def get_dataframe_parties(base_dir):
    """
    Read the detection csv files from an arbitrary number of eqcorrscan Parties to a dataframe.
    
    :param base_dir: Path to a directory of extracted party subdirs.
    :return: A Dask DataFrame with proper column types and additional threshold columns.
    """
    # Find all `*detections.csv` files in the directory structure
    detection_files = glob(os.path.join(base_dir, "**", "*detections.csv"), recursive=True)
    if not detection_files:
        raise FileNotFoundError(f"No `*detections.csv` files found in {base_dir}")

    # Function to parse a single file into a Pandas DataFrame
    def parse_file(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Extract column names from the first line
        columns = [field.split(":")[0].strip() for field in lines[0].split(";") if ":" in field]
        
        # Extract data rows
        data = []
        for line in lines:
            row = [field.split(":", 1)[1].strip() for field in line.split(";") if ":" in field]
            data.append(row)
        
        # Create a Pandas DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Convert column types
        if 'detect_time' in df.columns:
            df['detect_time'] = pd.to_datetime(df['detect_time'], errors='coerce')
        if 'no_chans' in df.columns:
            df['no_chans'] = pd.to_numeric(df['no_chans'], errors='coerce', downcast='integer')
        for col in ['detect_val', 'threshold', 'threshold_input']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
        
        # Add columns for different threshold values
        if 'threshold' in df.columns:
            df['MAD'] = df['threshold'] / 8  # Calculate MAD from threshold (threshold = 8 * MAD)
            df['MAD*10'] = df['MAD'] * 10
            df['MAD*15'] = df['MAD'] * 15
            df['MAD*20'] = df['MAD'] * 20
        
        return df

    # Use Dask to read and concatenate all files
    delayed_dfs = [delayed(parse_file)(file) for file in detection_files]
    dask_df = dd.from_delayed(delayed_dfs)
    return dask_df


def get_events_seiscomp(status, location_methods, url=None, **kwargs):
    """
    Get events from seiscomp database using obspy.client.fdsn.Client

    :param status: Status of events to fetch (e.g. 'manual', 'automatic')
    :param location_method: Location method of origins to fetch (LOCSAT, NonLinLoc, or RTDD)

    :return obspy.core.events.Catalog
    """
    if not all(method in ['NonLinLoc', 'RTDD', 'LOCSAT', 'HYPO71'] for method in location_methods):
        raise ValueError('All location methods must be NonLinLoc, RTDD, or LOCSAT')
    if status not in ['automatic', 'manual']:
        raise ValueError('Status must be automatic or manual')
    if not url:
        cli = Client("http://131.243.224.19:8085", timeout=3600)
    else:
        cli = Client(url, timeout=3600)
    hungry_cat = cli.get_events(**kwargs)
    full_cat = Catalog()
    for ev in hungry_cat:
        print(f'Pulling all ojects for event {ev.resource_id.id}')
        full_ev = cli.get_events(eventid=ev.resource_id.id.split('/')[-1], includeallorigins=True,
                                 includearrivals=True)[0]
        print(full_ev)
        origin = full_ev.preferred_origin()
        if origin.method_id is not None:
            if origin.method_id.id in location_methods and origin.evaluation_mode == status:
                full_ev.origins = [origin]
        else:
            print(origin)
            try:
                origins = [o for o in full_ev.origins if o.method_id.id in location_methods
                        and o.evaluation_mode == status]
                origins.sort(key=lambda x: x.creation_info.creation_time)
            except AttributeError:
                origins = full_ev.origins
            try:
                origin = origins[-1]
                full_ev.origins = [origin]
            except IndexError:
                print('No origins found for this event')
                continue
        full_cat.append(full_ev)
    return full_cat


def retrieve_usgs_catalog(**kwargs):
    """
    Wrapper on obspy.clients.fdsn.Client and libcomcat (usgs) to retrieve a full
    catalog, including phase picks (that otherwise are not supported by the usgs
    fdsn implementation)

    :param kwargs: Will be passed to the Client (e.g. minlongitude, maxmagnitude
        etc...)
    :return: obspy.core.events.Catalog
    """
    cli = Client('https://earthquake.usgs.gov')
    # cli = Client('NCEDC')
    cat = cli.get_events(**kwargs)
    print('{} events in catalog'.format(len(cat)))
    # Now loop over each event and grab the phase dataframe using libcomcat
    rms = []  # List of events with no comcat arrival info to remove
    for ev in cat:
        print(ev.resource_id.id)
        eid = ev.resource_id.id.split('=')[-2].split('&')[0]
        # eid = 'nc{}'.format(ev.resource_id.id.split('/')[-1])
        print(eid)
        detail = get_event_by_id(eid, includesuperseded=True)
        try:
            phase_df = get_phase_dataframe(detail)
        except ProductNotFoundError:
            rms.append(ev)
            continue
        o = ev.preferred_origin()
        try:
            for i, phase_info in phase_df.iterrows():
                seed_id = phase_info['Channel'].split('.')
                loc = seed_id[-1]
                if loc == '--':
                    loc = ''
                wf_id = WaveformStreamID(network_code=seed_id[0],
                                         station_code=seed_id[1],
                                         location_code=loc,
                                         channel_code=seed_id[2])
                pk = Pick(time=UTCDateTime(phase_info['Arrival Time']),
                          method=phase_info['Status'], waveform_id=wf_id,
                          phase_hint=phase_info['Phase'])
                try:
                    arr = Arrival(pick_id=pk.resource_id.id, phase=pk.phase_hint,
                                  azimuth=phase_info['Azimuth'],
                                  distance=phase_info['Distance'],
                                  time_residual=phase_info['Residual'],
                                  time_weight=phase_info['Weight'])
                    o.arrivals.append(arr)
                    ev.picks.append(pk)
                except ValueError:
                    continue
        except AttributeError:
            rms.append(ev)
            continue
        # Try to read focal mechanisms/moment tensors
        try:
            if 'moment-tensor' in detail.products:
                # Always take MT where available
                mt_xml = detail.getProducts(
                    'moment-tensor')[0].getContentBytes('quakeml.xml')[0]
            elif 'focal-mechanism' in detail.products:
                mt_xml = detail.getProducts(
                    'focal-mechanism')[0].getContentBytes('quakeml.xml')[0]
            else:
                continue
        except ContentNotFoundError:
            continue
        mt_ev = read_events(io.TextIOWrapper(io.BytesIO(mt_xml),
                                             encoding='utf-8'))
        if len(mt_ev[0].focal_mechanisms) > 0:
            FM = mt_ev[0].preferred_focal_mechanism()
        if FM is None:
            FM = mt_ev[0].focal_mechanisms[0]
        FM.triggering_origin_id = ev.preferred_origin().resource_id.id
        ev.focal_mechanisms = [FM]
    for rm in rms:
        cat.events.remove(rm)
    return cat


def parse_eq_Canada(file_path):
    """
    Parse eqcanada text file to an obspy Catalog

    :param file_path: Path to downloaded file
    :return:
    """
    cat = Catalog()
    with open(file_path, 'r') as f:
        next(f)
        for ln in f:
            line = ln.split('|')
            rid = ResourceIdentifier(id='smi:local/{}'.format(line[0]))
            o = Origin(time=UTCDateTime(line[1]), latitude=float(line[2]),
                       longitude=float(line[3]), depth=float(line[4]))
            m = Magnitude(magnitude_type=line[5], mag=float(line[6]))
            e = Event(resource_id=rid, force_resource_id=False,
                      origins=[o], magnitudes=[m])
            cat.events.append(e)
    return cat

def ngds_to_cat(path):
    """
    Read the excel file from LLNL for Newberry 2012-2013 to a catalog
    :param path:
    :return:
    """
    df = pd.read_excel(path)
    return


def ncedc_dd_to_cat(path):
    """
    Read csv of ncedc dd locations to catalog

    :param path:
    :return:
    """
    cat = Catalog()
    with open(path, 'r') as f:
        next(f)
        for ln in f:
            ot, lat, lon, dp, mag, mt, _, _, _, _, _, eid = ln.split(',')
            ot = UTCDateTime('T'.join(ot.split()))
            o = Origin(latitude=lat, longitude=lon,
                       depth=float(dp) * 1000, time=ot)
            m = Magnitude(mag=mag, magnitude_type=mt)
            ev = Event(origins=[o], magnitudes=[m],
                       resource_id=ResourceIdentifier(id=eid))
            cat.events.append(ev)
    return cat


def fervo_to_cat(path):
    # Fervo catalog relative to 16B wellhead (in feet)
    X0 = 334641.1891
    Y0 = 4263443.693
    Z0 = 1650.0249

    df_test = pd.read_csv(path, skipinitialspace=True)
    df_test.columns = df_test.columns.str.strip()
    df_test['Easting (m)'] = X0 + df_test['X'] * 0.3048
    df_test['Northing (m)'] = Y0 + df_test['Y'] * 0.3048
    df_test['Elevation (m)'] = Z0 - pd.to_numeric(df_test['Depth'], errors='coerce') * 0.3048
    # Convert Easting/Northing (EPSG:6341) to lat/lon
    proj = pyproj.Proj("EPSG:6341")
    df_test['Longitude'], df_test['Latitude'] = proj(df_test['Easting (m)'], df_test['Northing (m)'], inverse=True)
    cat = Catalog()
    for _, row in df_test.iterrows():
        try:
            ot = UTCDateTime().strptime(row['Origin Date'] + row['Origin Time'], '%d/%m/%Y%H:%M:%S.%f')
        except:
            continue
        o = Origin(
            time=ot,
            longitude=row['Longitude'],
            latitude=row['Latitude'],
            depth= -1. * row['Elevation (m)']
        )
        m = Magnitude(
            mag=row['MomMag'] if 'MomMag' in row else 1.0
        )
        ev = Event(origins=[o], magnitudes=[m])
        ev.preferred_origin_id = o.resource_id.id
        cat.events.append(ev)
    return cat


def parse_pyrocko_markers(marker_file):
    """Parse picks in Pyrocko markers format"""
    picks = []
    with open(marker_file, 'r') as f:
        for ln in f:
            line = ln.split()
            if line[0] in ['#', 'event:']:
                continue
            time = UTCDateTime('T'.join(line[1:3]))
            nslc = line[4].split('.')
            pk = Pick(time=time, phase_hint=line[-3],
                      waveform_id=WaveformStreamID(network_code=nslc[0],
                                                   station_code=nslc[1],
                                                   location_code=nslc[2],
                                                   channel_code=nslc[3]))
            picks.append(pk)
    return picks


def combine_xml_pyrocko_markers(event_file, marker_file):
    """Combine a QuakeML file with associated pyrocko markers"""
    ev = read_events(event_file)
    picks = parse_pyrocko_markers(marker_file)
    # Replace original picks with new ones (only bc we use all of them in NLLoc)
    ev[0].picks = picks
    return ev

#### PLOTTTING ####

def plot_cumulative_catalog(catalogs, xlim=None, title=None):
    """Simple cumulative number of events with time"""
    fig, ax = plt.subplots()
    for cat in catalogs:
        times = [ev.picks[-1].time.datetime for ev in cat if len(ev.picks) > 0]
        times.sort()
        vals = np.arange(len(times))
        ax.step(times, vals)
    # Formatting
    ax.set_ylim(bottom=0.)
    if xlim:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title, fontsize=15)
    fig.autofmt_xdate()
    ax.set_ylabel('Cumulative number')
    ax.set_facecolor('whitesmoke')
    plt.show()
    return


def extract_lbnl_template(comment_text):
    """
    Extracts the first occurrence of 'lbnl202' followed by letters/digits from a string.
    Returns None if not found.
    """
    match = re.search(r'(lbnl202\w+)', comment_text)
    if match:
        return match.group(1)
    else:
        return None


def plot_template_event_bipartite(cat, max_labels=50):
    """
    Plot a bipartite graph of template-event connections from an ObsPy Catalog,
    extracting template names of the form 'lbnl202...' from Origin comments.
    """
    edges = []
    for event in cat:
        event_id = event.resource_id.id.split('/')[-1]
        for origin in event.origins:
            template_name = extract_lbnl_template(origin.comments[0].text)
            if template_name:
                edges.append((template_name, event_id))

    if not edges:
        print("No template-event edges found. Check your comment format.")
        return

    # Unique lists for coloring and layout
    template_names = sorted(set(template for template, _ in edges))
    event_ids = sorted(set(event for _, event in edges))
    print(len(edges))

    # Build a DataFrame: rows=templates, cols=events, values=1/0
    df = pd.DataFrame(0, index=template_names, columns=event_ids)
    for template, event in edges:
        df.loc[template, event] = 1

    # Clustered heatmap
    sns.clustermap(df, cmap="Blues", figsize=(12, 8))
    plt.title("Template-Event Detection Matrix")
    plt.show()


def plot_cumulative_detections(df, group_by_template=False, exclude_templates=None, title=None):
    """
    Plot the cumulative number of detections over time.

    :param df: A Pandas or Dask DataFrame containing the detection data. 
               Must include 'detect_time' and 'template_name' columns.
    :param group_by_template: If True, plot one line for each template. 
                              If False, plot all detections together.
    :param exclude_templates: A list of template names to exclude from the plot.
    :param title: Title for the plot (optional).
    """

    # Ensure the DataFrame is computed if it's a Dask DataFrame
    if isinstance(df, dd.DataFrame):
        df = df.compute()

    # Ensure 'detect_time' is sorted and converted to datetime
    df['detect_time'] = pd.to_datetime(df['detect_time'], errors='coerce')
    df = df.sort_values(by='detect_time')

    # Exclude specified templates
    if exclude_templates:
        df = df[~df['template_name'].isin(exclude_templates)]

    # Initialize the plot
    plt.figure(figsize=(12, 6))

    if group_by_template:
        # Group by template and calculate cumulative counts
        grouped = df.groupby('template_name')
        for template, group in grouped:
            group = group.sort_values(by='detect_time')
            cumulative_counts = group['detect_time'].value_counts().sort_index().cumsum()
            plt.plot(cumulative_counts.index, cumulative_counts.values, label=template)
    else:
        # Plot all detections together
        cumulative_counts = df['detect_time'].value_counts().sort_index().cumsum()
        plt.plot(cumulative_counts.index, cumulative_counts.values, label='All Templates')

    # Formatting
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Cumulative Detections', fontsize=12)
    plt.title(title or 'Cumulative Detections Over Time', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()