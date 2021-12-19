#!/usr/bin/python

"""
Utilities for handling various formats of borehole trajectory/orientation files
"""
import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from itertools import cycle
from glob import glob
try:
    from scipy.interpolate import Rbf
    from skspatial.objects import Plane
except ModuleNotFoundError:
    print('No skspatial here')
from pathlib import Path
from matplotlib.dates import date2num, num2date

# local imports
from lbnl.coordinates import cartesian_distance

fsb_wellheads = {'B1': (2579341.176, 1247584.889, 513.378),
                 'B2': (2579334.480, 1247570.980, 513.200),
                 'B3': (2579324.915, 1247611.678, 514.132),
                 'B4': (2579325.497, 1247612.048, 514.067),
                 'B5': (2579332.571, 1247597.289, 513.782),
                 'B6': (2579333.568, 1247598.048, 513.701),
                 'B7': (2579335.555, 1247599.383, 513.702),
                 'B8': (2579331.968, 1247600.540, 513.757),
                 'B9': (2579327.803, 1247608.908, 513.951),
                 'B10': (2579335.483, 1247593.244, 513.951)}


def depth_to_xyz(well_dict, well, depth):
    """
    Return xyz coords for depth in a given borehole

    :param well: Well string
    :param depth: Depth float
    :return:
    """
    easts, norths, zs, deps = np.hsplit(well_dict[well], 4)
    # Get closest depth point
    dists = np.squeeze(np.abs(depth - deps))
    x = easts[np.argmin(dists)][0]
    y = norths[np.argmin(dists)][0]
    z = zs[np.argmin(dists)][0]
    return (x, y, z)


def borehole_plane_intersect(well_dict, well, pt_on_plane, strike, dip,
                             epsilon=1e-6):
    """
    Return the intersection point of a borehole with a plane

    :param well_dict: Dict of well path from create_FSB_boreholes
    :param well: Name of well in question
    :param pt_on_plane: Point on the plane defined by strike-dip
    :param strike: Strike of plane
    :param dip: Dip of plane (RHR)
    :param epsilon: Near-zero approximation for near-parallel line/plane

    Return a Vector or None (when the intersection can't be found).

    ..note Injection point FSB: (2579330.559, 1247576.248, 472.775)
    """
    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    # Define fault normal
    p_no = np.array((np.sin(d) * np.cos(s),
                     -np.sin(d) * np.sin(s),
                     np.cos(d)))
    p_no /= np.linalg.norm(p_no)
    well_pts = well_dict[well]
    p0 = well_pts[0][:3]
    p1 = well_pts[-1][:3]
    u = p1 - p0
    dot = np.dot(p_no, u)
    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = p0 - pt_on_plane
        fac = -np.dot(p_no, w) / dot
        u *= fac
        pt = p0 + u
        # Now calculate closest depth point
        closest_pt = np.argmin(np.abs(np.sum(pt - well_pts[:, :3], axis=1)))
        apx_depth = well_pts[closest_pt, -1]
        return pt, fac, apx_depth
    else:
        # The segment is parallel to plane.
        return None


def distance_to_borehole(well_dict, well, depth, gallery_pts,
                         excavation_times):
    """
    Calculate distance, x, y, azimuth and dip to SIMFIP from Gallery excavation
    front at Mont Terri

    :param well_dict: Output of parse_FSB_wells
    :param well: String of well to calculate distance to
    :param depth: Depth in well to calculate distances to
    :param gallery_pts: Path to file with 1-m spacing gallery distances
    :param excavation_times: Path to file with times and distances along gallery
        for G18 excavation
    :return:
    """
    borehole_xyz = depth_to_xyz(well_dict, well, depth)
    df_gallery = read_gallery_distances(gallery_pts)
    gallery_dist = df_gallery['distance']
    df_excavation = read_gallery_excavation(excavation_times).copy()
    exc_dist = df_excavation['distance [m]']
    # Calculate value of closest point along my line to measured dist
    dists_along = np.array([gallery_dist[np.argmin(np.abs(gallery_dist - d))]
                            for d in exc_dist])
    # Get X, Y, Z as array for each of these distances
    pts_along = np.vstack([df_gallery.loc[df_gallery['distance'] == d].values[0]
                           for d in dists_along])
    df_excavation['X'] = pts_along[:, 0]
    df_excavation['Y'] = pts_along[:, 1]
    df_excavation['Z'] = pts_along[:, 2]
    # Pythagoras dance
    bh_dx = borehole_xyz[0] - pts_along[:, 0]
    bh_dy = borehole_xyz[1] - pts_along[:, 1]
    bh_dz = borehole_xyz[2] - pts_along[:, 2]
    bh_dists = np.sqrt(bh_dx**2 + bh_dy**2 + bh_dz**2)
    bh_az = np.rad2deg(np.arctan(bh_dx / bh_dy))
    bh_plunge = np.rad2deg(np.arcsin(bh_dz / bh_dists))
    # Put back into DataFrame
    df_excavation.insert(0, 'dh', np.sqrt(bh_dx**2 + bh_dy**2))
    df_excavation.insert(0, 'dz', bh_dz)
    df_excavation.insert(0, 'Distance to SIMFIP', bh_dists)
    df_excavation.insert(0, 'Azimuth to SIMFIP', bh_az)
    df_excavation.insert(0, 'Plunge to SIMFIP', bh_plunge)
    df_excavation = df_excavation.rename(
        mapper={'distance [m]': 'Excavation distance [m]'}, axis='columns')
    return df_excavation


def read_gallery_excavation(path):
    """Parse gallery excavation progress"""
    df = pd.read_csv(path, parse_dates=[0])
    df = df.set_index('time')
    return df


def read_gallery_distances(path):
    """Helper to read points along gallery"""
    df = pd.read_csv(path)
    # Call elevation of center of gallery ~513m
    df.insert(2, 'Z', np.ones(df['X'].values.shape[0]) * 515.)
    return df[['X', 'Y', 'Z', 'distance']]


def calc_mesh_area(X, Y, Z):
    """Assume rotated mesh is rhombus and calculate area"""
    v1 = (X[0, 0], Y[0, 0], Z[0, 0])
    v2 = (X[0, 2], Y[0, 2], Z[0, 2])
    v3 = (X[2, 0], Y[2, 0], Z[2, 0])
    v4 = (X[2, 2], Y[2, 2], Z[2, 2])
    p = np.sqrt((v1[0] - v4[0])**2 +
                (v1[1] - v4[1])**2 +
                (v1[2] - v4[2])**2)
    q = np.sqrt((v2[0] - v3[0])**2 +
                (v2[1] - v3[1])**2 +
                (v2[2] - v3[2])**2)
    return p * q / 2


def scale_mesh(X, Y, Z, center):
    """Scale points of mesh to area of 1"""
    A = calc_mesh_area(X, Y, Z)
    scale = 1. / A
    Xs = ((X - center[0]) * scale) + center[0]
    Ys = ((Y - center[1]) * scale) + center[1]
    Zs = ((Z - center[2]) * scale) + center[2]
    return Xs.flatten(), Ys.flatten(), Zs.flatten()


def make_frac_mesh(center, dip_direction, dip):
    # Dip direction to strike
    strike = dip_direction - 90.
    if strike < 0.:
        strike += 360.
    # Helper to return X, Y, Z arrays to be plotted
    # 2m square grid around point
    x = np.linspace(center[0] - 0.5, center[0] + 0.5, 3)
    y = np.linspace(center[1] - 0.5, center[1] + 0.5, 3)
    # Mesh x and y arrays
    X, Y = np.meshgrid(x, y)
    dip_rad = np.deg2rad(dip)
    strike_rad = np.deg2rad(strike)
    # Normal to plane
    a = np.sin(dip_rad) * np.cos(strike_rad)  # East
    b = -np.sin(dip_rad) * np.sin(strike_rad)  # North
    c = np.cos(dip_rad)
    # Evaluate d at borehole xyz
    d = np.dot(np.array([a, b, c]), center)
    Z = (d - a * X - b * Y) / c
    return X, Y, Z


def structures_to_planes(path, well_dict):
    """
    Take the Optical TV picked structures for a well (dip direction-dip)
    and return lists of X, Y, Z arrays defining a plane for plotting.

    :param path: Path to Terratek excel
    :return: list of (X, Y, Z)
    """
    # Custom color palette similar to wellcad convention
    cols = {'open/undif. fracture': 'blue',
            'sealed fracture / vein': 'lightblue',
            'foliation / bedding': 'red',
            'induced fracture': 'magenta',
            'sedimentary structures/color changes undif.': 'green',
            'uncertain type': 'orange',
            'lithology change': 'yellow'}
    # Read excel sheet
    well = path.split('_')[-2]
    fracs = pd.read_excel(path, skiprows=np.arange(9),
                          usecols=np.arange(1, 9), header=None)
    frac_planes = []
    for i, frac in fracs.iterrows():
        dep = frac[1]
        dd = frac[4]
        dip = frac[5]
        frac_type = frac[7]
        # Get borehole xyz of feature
        try:
            bh_point = depth_to_xyz(well_dict, well, dep)
        except KeyError:
            print('No borehole info yet for {}'.format(well))
            return []
        X, Y, Z = make_frac_mesh(center=bh_point, dip_direction=dd, dip=dip)
        Xs, Ys, Zs = scale_mesh(X, Y, Z, bh_point)
        frac_planes.append((Xs, Ys, Zs, frac_type, cols[frac_type]))
    return frac_planes


def read_frac_cores(path, well):
    """
    Return dictionary of {well: {'fracture type': fracture density array}}

    :param path: Path to core fracture density excel file
        Fracture_Density_Core.xlsx
    :return: dict
    """
    well_dict = {'All fractures': {}}
    fracs = pd.read_excel(path, sheet_name=None, skiprows=[2], header=1)
    for full_name, items in fracs.items():
        if 'BCS-{}'.format(well) == full_name:
            bin_centers = ((items['TopDepth'] +
                            items['BottomDepth']) / 2).values
            well_dict['All fractures'] = np.stack(
                (bin_centers, items['Total Counts Core 1m'])).T
    return well_dict


def read_frac_quinn(path, well):
    """
    Read Quinns TV picks

    :return:
    """
    well_dict = {'All fractures': {}}
    fracs = pd.read_excel(path, sheet_name=None, skiprows=[1], header=0)
    for full_name, items in fracs.items():
        if 'BCS-{}'.format(well) == full_name:
            bin_centers = items['Depth'].values
            well_dict['All fractures'] = np.stack(
                (bin_centers, items['Dip'], items['Azimuth'],
                 items['Mont Terri Structure'])).T
    return well_dict


def otv_to_sdd(path):
    """
    Take Quinns wellcad picks and parse to strike-dip-dep DataFrames
    for fracture groups

    :param path: Path to wellcad.xlsx file
    :return:
    """
    exc_frnt = np.array([2.57931745e+06, 1.24755756e+06, 5.15000000e+02])
    otv_picks = pd.read_excel(path, sheet_name=None, skiprows=[1],
                              header=0)
    well_dict = create_FSB_boreholes()
    # Get dip dir, dip, and depth for each subset
    otv_MF = {w[-2:]: d.loc[d['Main Fault'] == 'f'][['Azimuth', 'Dip', 'Depth']]
              for w, d in otv_picks.items()
              if w[-2:] in ['D4', 'D5', 'D6']}
    otv_DSS = {w[-2:]: d.loc[d['DSS'] == 's'][['Azimuth', 'Dip', 'Depth']]
               for w, d in otv_picks.items()
               if w[-2:] in ['D4', 'D5', 'D6']}
    otv_none = {w[-2:]: d.loc[(d['Main Fault'] != 'f') &
                              (d['DSS'] != 's')][['Azimuth', 'Dip', 'Depth']]
                for w, d in otv_picks.items()
                if w[-2:] in ['D4', 'D5', 'D6']}
    for thing in [otv_MF, otv_DSS, otv_none]:
        for w, df in thing.items():
            df['Distance'] = [cartesian_distance(
                pt1=depth_to_xyz(well_dict, w, d),
                pt2=exc_frnt) for d in df['Depth'].values]
    return otv_MF, otv_DSS, otv_none


def read_frac_otv(path, well):
    """
    Return dict of {well: {'fracture type': fracture density array}}

    :param path:
    :param well:
    :return:
    """
    well_dict = {'All fractures': {}}
    fracs = pd.read_excel(path, sheet_name=None, skiprows=[2], header=1)
    for full_name, items in fracs.items():
        if 'BCS-{}'.format(well) == full_name:
            bin_centers = ((items['TopDepth'] +
                            items['BottomDepth']) / 2).values
            well_dict['All fractures'] = np.stack(
                (bin_centers, items['Total Counts Core 1m'])).T
    return well_dict


def calculate_frac_density(path, well_dict):
    """
    Return dict of {well: {'fracture type': fracture density array}}

    :param path:
    :param well_dict:
    :return:
    """
    well = path.split('_')[-2]
    fracs = pd.read_excel(path, skiprows=np.arange(9),
                          usecols=np.arange(1, 9), header=None)
    deps = fracs[1].values
    types = fracs[7].values
    unique_types = list(set(types))
    # 0.5 step and 0.5 overlap
    dep_bins = np.arange(0, well_dict[well][-1, -1], 0.5)
    frac_dict = {}
    # All fractures first
    total_density = np.array(
        [deps[np.where(np.logical_and(a - 1 <= deps, deps < a + 1))].shape[0]
         for a in dep_bins])
    frac_dict['All fractures'] = np.stack((dep_bins, total_density)).T
    for t in unique_types:
        ds = fracs[fracs[7] == t][1].values
        t_dens = np.array(
            [deps[np.where(np.logical_and(a - 1 <= ds, ds < a + 1))].shape[0]
             for a in dep_bins])
        frac_dict[t] = np.stack((dep_bins, t_dens)).T
    return frac_dict


def create_FSB_boreholes(gocad_dir='/media/chet/data/chet-FS-B/Mont_Terri_model/',
                         asbuilt_dir='/media/chet/data/chet-FS-B/wells/'):
    """
    Return dictionary of FSB well coordinates

    :param asbuilt_dir: Directory holding the gamma logs for each well
    """
    if not os.path.isdir(asbuilt_dir):
        asbuilt_dir = '/media/chet/hdd/seismic/chet_FS-B/wells/'
    if not os.path.isdir(asbuilt_dir):
        asbuilt_dir = 'data/chet-FS-B/wells'
    excel_asbuilts = glob('{}/**/*Gamma_Deviation.xlsx'.format(asbuilt_dir),
                          recursive=True)
    well_dict = {}
    if not os.path.isdir(gocad_dir):
        gocad_dir = '/media/chet/hdd/seismic/chet_FS-B/Mont_Terri_model'
    if not os.path.isdir(gocad_dir):
        gocad_dir = 'data/chet-FS-B/Mont_Terri_model'
    gocad_asbuilts =  glob('{}/*.wl'.format(gocad_dir))
    for gocad_f in gocad_asbuilts:
        name = str(gocad_f).split('-')[-1].split('.')[0]
        well_dict[name] = []
        # Multispace delimiter
        top = pd.read_csv(gocad_f, header=None, skiprows=np.arange(13),
                          delimiter='\s+', index_col=False, engine='python',
                          nrows=1)
        rows = pd.read_csv(gocad_f, header=None, skiprows=np.arange(14),
                           delimiter='\s+', index_col=False, engine='python',
                           skipfooter=1)
        lab, x_top, y_top, z_top = top.values.flatten()
        try:
            x_top, y_top, z_top = fsb_wellheads[name]
        except (ValueError, KeyError):
            print('Cant read top for {}'.format(name))
            pass
        well_dict[name] = np.stack(((x_top + rows.iloc[:, 3]).values,
                                    (y_top + rows.iloc[:, 4]).values,
                                    rows.iloc[:, 2].values,
                                    rows.iloc[:, 1].values)).T
        try:
            excel_f = [f for f in excel_asbuilts
                       if f.split('-')[-1].split('_')[0] == name][0]
        except IndexError:
            # If so, make a more highly-sampled interpolation
            x, y, z, d = zip(*well_dict[name])
            td = d[-1]
            if td == 'Top':
                td = float(d[-3])
            well_dict[name] = np.stack((np.linspace(x_top, x[-1], 1000),
                                        np.linspace(y_top, y[-1], 1000),
                                        np.linspace(z_top, z[-1], 1000),
                                        np.linspace(0, td, 1000))).T
            continue
        df = pd.read_excel(excel_f, header=6, skiprows=[7])
        well_dict[name] = np.stack(((x_top + df['Easting']).values,
                                    (y_top + df['Northing']).values,
                                    (z_top - df['TVD']).values,
                                    df['Depth'].values)).T
    return well_dict


def parse_surf_boreholes(file_path):
    """
    Parse the surf 4850 xyz file to dict of hole: {[(x, y, z), (x1, y1, z1)]}

    :param file_path: Path to borehole text file
    :return: dict
    """
    well_dict = {}
    with open(file_path, 'r') as f:
        next(f)
        for ln in f:
            ln = ln.rstrip('\n')
            line = ln.split(',')
            xm = float(line[3]) / 3.28084
            ym = float(line[4]) / 3.28084
            zm = float(line[5]) / 3.28084
            dp = float(line[2])
            name = line[0].split('-')[1]
            if name in well_dict:
                well_dict[name] = np.concatenate(
                    (well_dict[name],
                     np.array([xm, ym, zm, dp]).reshape(1, 4)))
            else:
                well_dict[name] = np.array([xm, ym, zm, dp]).reshape(1, 4)
    return well_dict


def make_4100_boreholes(path, plot=False):
    """
    Take as-planned for 4100L boreholes and return array of xyz pts

    :param path: Path to excel file from Paul's drilling plan

    :return:
    """
    df = pd.read_excel(path)
    # Do the iterrows cause who cares
    well_dict = {}
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    for i, row in df.iterrows():
        strt = row[['Easting (ft)', 'Northing (ft)',
                    'Height (ft)']].values * 0.3048
        bearing = np.deg2rad(row['Bearing (deg)'])
        tilt = np.deg2rad(row['Tilt (deg)'])
        unit_v = np.array([np.sin(bearing) * np.cos(tilt),
                           np.cos(bearing) * np.cos(tilt),
                           np.sin(tilt)])
        unit_v /= np.linalg.norm(unit_v)
        vect = unit_v * row['Length (ft)'] * 0.3048
        end = strt + vect
        pts = np.vstack(
            [np.linspace(strt[0], end[0], 200),
             np.linspace(strt[1], end[1], 200),
             np.linspace(strt[2], end[2], 200)]).T
        well_dict[row['New Name']] = pts
        if plot:
            ax.plot(xs=pts[:, 0], ys=pts[:, 1], zs=pts[:, 2], label=row['New Name'])
    if plot:
        fig.legend()
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_zlabel('Elevation')
        plt.show()
    return well_dict


def wells_4850_to_gmt(outfile):
    colors = cycle(sns.color_palette())
    well_file = '/media/chet/data/chet-collab/boreholes/surf_4850_wells.csv'
    wells = parse_surf_boreholes(well_file)
    with open(outfile, 'w') as f:
        for key, pts in wells.items():
            col = np.array(next(colors))
            col *= 255
            col_str = '{}/{}/{}'.format(int(col[0]), int(col[1]), int(col[2]))
            f.write('>-W1.0,{} -L{}\n'.format(col_str, key))
            for pt in pts:
                f.write('{} {}\n'.format(pt[0], pt[1]))
    return


def fsb_to_xyz(well_dict, strike, dip, outfile):
    """
    Write out the xyz points of various important FSB features

    :param outfile: Path to output .csv
    :return:
    """
    # Fault intersection pts
    fault_int = {w: borehole_plane_intersect(
        well_dict, w, (2579330.559, 1247576.248, 472.775), strike, dip)
        for w in ['B1', 'B2', 'B10', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']}
    lines = ['Feature, Borehole, Depth [m], X, Y, Z']
    # D7 SIMFIP
    for feat, dep in zip(('Top SIMFIP', 'Bottom SIMFIP'), (21.55, 28.75)):
        pt = depth_to_xyz(well_dict, 'D7', dep)
        lines.append('{}, D7, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
            feat, dep, pt[0], pt[1], pt[2]))
    # B2 SIMFIP
    for feat, dep in zip(('Top SIMFIP', 'Bottom SIMFIP'), (40.47, 41.47)):
        pt = depth_to_xyz(well_dict, 'B2', dep)
        lines.append('{}, B2, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
            feat, dep, pt[0], pt[1], pt[2]))
    # B2 Screens
    for i, dep in enumerate([25.31, 335.4, 41.47, 45.76, 51.34]):
        pt_top = depth_to_xyz(well_dict, 'B2', dep - 1)
        lines.append('Screen {} top, B2, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
            i + 1, dep - 1, pt_top[0], pt_top[1], pt_top[2]))
        pt_bot = depth_to_xyz(well_dict, 'B2', dep)
        lines.append(
            'Screen {} bottom, B2, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
                i + 1, dep, pt_bot[0], pt_bot[1], pt_bot[2]))
    # B1 pressure transducers
    for i, dep in enumerate([29., 31., 34.9, 42.2]):
        pt = depth_to_xyz(well_dict, 'B1', dep)
        lines.append('Pressure {}, B1, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
            i + 1, dep, pt[0], pt[1], pt[2]))
    # B1 Displacement sensors
    for i, dep in enumerate([31., 34.9, 38.55, 42.2]):
        pt = depth_to_xyz(well_dict, 'B1', dep)
        lines.append('Displacement {}, B1, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
            i + 1, dep, pt[0], pt[1], pt[2]))
    # All Fault intervals
    for well, fault_pt in fault_int.items():
        lines.append('Fault top, {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
            well, fault_pt[-1], fault_pt[0][0],
            fault_pt[0][1], fault_pt[0][2]))
    with open(outfile, 'w') as f:
        f.write('\n'.join(lines))
    return


def fit_mt_main_fault(well_dict, section='all', function='thin_plate_spline'):
    fault_depths = {'D1': (14.34, 19.63), 'D2': (11.04, 16.39),
                    'D3': (17.98, 20.58), 'D4': (27.05, 28.44),
                    'D5': (19.74, 22.66), 'D6': (28.5, 31.4),
                    'D7': (22.46, 25.54), 'B2': (41.25, 45.65),
                    'B1': (34.8, 42.25), 'B9': (None, 55.7),
                    'B10': (17.75, 21.7), '1': (38.15, 45.15),
                    '2': (44.23, 49.62), '3': (38.62, 43.39)}
    if section == 'FS':
        fault_depths = {k: fault_depths[k] for k in ('1', '2', '3')}
    elif section == 'FSB':
        fault_depths = {k: fault_depths[k] for k in fault_depths.keys()
                        if k[0] == 'B'}
    elif section == 'CSD':
        fault_depths = {k: fault_depths[k] for k in fault_depths.keys()
                        if k[0] == 'D'}
    elif section == 'west':
        fault_depths = {k: fault_depths[k] for k in fault_depths.keys()
                        if k[0] == 'D' or k in ('B2')}
    elif section == 'east':
        fault_depths = {k: fault_depths[k] for k in fault_depths.keys()
                        if k[0] == 'B' or k in ('1', '2', '3')}
    elif section == 'all':
        pass
    else:
        print('Section {} is invalid'.format(section))
        return
    print('Section: {}'.format(section))
    # Do best fit plane
    tops = [depth_to_xyz(well_dict, well, d[0])
            for well, d in fault_depths.items() if d[0]
            and well in fault_depths]
    bottoms = [depth_to_xyz(well_dict, well, d[1])
               for well, d in fault_depths.items() if well in fault_depths]
    A_top = np.array(tops).T
    A_bot = np.array(bottoms).T
    c_top = A_top.sum(axis=1) / A_top.shape[1]
    c_bot = A_top.sum(axis=1) / A_top.shape[1]
    # Top first
    u, s, v = np.linalg.svd(A_top - c_top[:, np.newaxis])
    # Lsqr quadratic fit
    X, Y = np.meshgrid(np.arange(np.min(A_top[0, :]), np.max(A_top[0, :]), 2),
                       np.arange(np.min(A_top[1, :]), np.max(A_top[1, :]), 2))
    tops_array = np.array(tops)
    spline = Rbf(tops_array[:, 0], tops_array[:, 1],
                 tops_array[:, 2], function=function,
                 smooth=0)
    Z = spline(X, Y)
    u1, u2, u3 = u[:, -1]
    if u3 < 0:
        easting = u2
    else:
        easting = -u2
    if u3 > 0:
        northing = u1
    else:
        northing = -u1
    dip = np.rad2deg(np.arctan(np.sqrt(easting**2 + northing**2) / u3))
    if easting >= 0:
        partA_strike = easting**2 + northing**2
        strike = np.rad2deg(np.arccos(northing / np.sqrt(partA_strike)))
    else:
        partA_strike = northing / np.sqrt(easting**2 + northing**2)
        strike = np.rad2deg(2 * np.pi - np.arccos(partA_strike))
    print('SVD strike Top: {}'.format(strike))
    print('SVD dip Top: {}'.format(dip))
    # Now bottom
    u, s, v = np.linalg.svd(A_bot - c_bot[:, np.newaxis])
    u1, u2, u3 = u[:, -1]
    if u3 < 0:
        easting = u2
    else:
        easting = -u2
    if u3 > 0:
        northing = u1
    else:
        northing = -u1
    dip = np.rad2deg(np.arctan(np.sqrt(easting**2 + northing**2) / u3))
    if easting >= 0:
        partA_strike = easting**2 + northing**2
        strike = np.rad2deg(np.arccos(northing / np.sqrt(partA_strike)))
    else:
        partA_strike = northing / np.sqrt(easting**2 + northing**2)
        strike = np.rad2deg(2 * np.pi - np.arccos(partA_strike))
    print('SVD strike Bottom: {}'.format(strike))
    print('SVD dip Bottom: {}'.format(dip))
    # Now compute fit for all possible planes
    dips = np.arange(90)
    strikes = np.arange(360)
    dip_rads = np.deg2rad(dips)
    strike_rads = np.deg2rad(strikes)
    S, D = np.meshgrid(strike_rads, dip_rads)
    # Normal to plane
    a = np.sin(D.flatten()) * np.cos(S.flatten())  # East
    b = -np.sin(D.flatten()) * np.sin(S.flatten())  # North
    c = np.cos(D.flatten())
    c_top = c_top.squeeze()
    planes = [Plane(point=c_top.squeeze(), normal=np.array([a[i], b[i], c[i]]))
              for i in range(a.shape[0])]
    rmss = np.array([np.sqrt(np.mean([p.distance_point_signed(t)**2
                                      for t in tops]))
                     for p in planes])
    rmss = rmss.reshape(S.shape)
    print('Gridsearch Strike: {}'.format(np.rad2deg(S.flatten()[np.argmin(rmss)])))
    print('Grid search Dip: {}'.format(np.rad2deg(D.flatten()[np.argmin(rmss)])))
    print('Grid search RMS: {}'.format(np.min(rmss)))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(np.rad2deg(S), np.rad2deg(D), rmss, cmap='viridis')
    ax.set_xlabel('Strike [deg]')
    ax.set_ylabel('Dip [deg]')
    ax.set_zlabel('RMS for plane [meters]')
    plt.show()
    return X, Y, Z

# Plotting

def plot_excavation_vector(df_excavation):
    """Plot excavation progress with time and lower hemi projection"""
    fig = plt.figure(figsize=(7, 10))
    ax1 = fig.add_subplot(211)
    df_excavation[['Distance to SIMFIP', 'Excavation distance [m]',
                   'dh', 'dz']].plot(ax=ax1)
    ax1.set_ylabel('Distance [m]')
    ax2 = fig.add_subplot(212, projection='polar')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    times = df_excavation.index.values
    colors = date2num(times)
    dots = ax2.scatter(
        np.deg2rad(df_excavation['Azimuth to SIMFIP']),
        df_excavation['Plunge to SIMFIP'],
        c=colors)
    ticks = np.linspace(np.nanmin(colors),
                        np.nanmax(colors), 5)
    cax = plt.colorbar(dots, ax=ax2, ticks=ticks)
    ticklabs = [num2date(t).strftime('%b-%d') for t in ticks]
    cax.ax.set_yticklabels(ticklabs)
    ax2.set_ylim([-90., 0])
    ax2.set_yticklabels([])
    plt.subplots_adjust(hspace=0.4, wspace=0.6)
    plt.show()
    return