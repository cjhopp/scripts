#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from glob import glob
from scipy.linalg import lstsq, norm
from scipy.io import loadmat
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from scipy.spatial.transform import Rotation as R



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

csd_well_colors = {'D1': 'blue', 'D2': 'blue', 'D3': 'green',
                   'D4': 'green', 'D5': 'green', 'D6': 'green', 'D7': 'black'}

# Mont Terri fault depths by borehole
fault_depths = {'D1': (14.34, 19.63), 'D2': (11.04, 16.39), 'D3': (17.98, 20.58),
                'D4': (27.05, 28.44), 'D5': (19.74, 22.66), 'D6': (28.5, 31.4),
                'D7': (22.46, 25.54), 'B2': (41.25, 45.65), 'B1': (34.8, 42.25),
                'B9': (55.7, 55.7), 'B10': (17.75, 21.7), '1': (38.15, 45.15),
                '2': (44.23, 49.62), '3': (38.62, 43.39)}

def plot_CSD_2D(autocad_path, gamma_path, strike=305.,
                origin=np.array([2579325., 1247565., 514.])):
    """
    Plot the Mont Terri lab in a combination of 3D, map view, and cross section

    :param autocad_path: Path to file with arcs and lines etc
    :param gamma_path: Path to Gamma deviation logs and .wl files for borehole trajectories
    :param strike: Strike of main fault to project piercepoints onto
    :param origin: Origin point for the cross section

    :return:
    """
    fig = plt.figure(figsize=(12, 12))
    spec = gridspec.GridSpec(ncols=8, nrows=8, figure=fig)
    ax3d = fig.add_subplot(spec[:4, :4], projection='3d')
    ax_x = fig.add_subplot(spec[:4, 4:])
    ax_map = fig.add_subplot(spec[4:, :4])
    ax_fault = fig.add_subplot(spec[4:, 4:])
    well_dict = create_FSB_boreholes(gamma_path)
    # Cross section plane (strike 320)
    r = np.deg2rad(360 - strike)
    normal = np.array([-np.sin(r), -np.cos(r), 0.])
    normal /= norm(normal)
    new_strk = np.array([np.sin(r), -np.cos(r), 0.])
    new_strk /= norm(new_strk)
    change_b_mat = np.array([new_strk, [0, 0, 1], normal])
    for afile in glob('{}/*.csv'.format(autocad_path)):
        # if 'FSB' in afile:
        #     continue
        df_cad = pd.read_csv(afile)
        lines = df_cad.loc[df_cad['Name'] == 'Line']
        arcs = df_cad.loc[df_cad['Name'] == 'Arc']
        for i, line in lines.iterrows():
            xs = np.array([line['Start X'], line['End X']])
            ys = np.array([line['Start Y'], line['End Y']])
            zs = np.array([line['Start Z'], line['End Z']])
            # Proj
            pts = np.column_stack([xs, ys, zs])
            proj_pts = np.dot(pts - origin, normal)[:, None] * normal
            proj_pts = pts - origin - proj_pts
            proj_pts = np.matmul(change_b_mat, proj_pts.T)
            ax3d.plot(xs, ys, zs, color='darkgray', zorder=110)
            ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='darkgray',
                      zorder=110)
            ax_map.plot(xs, ys, color='darkgray')
        for i, arc in arcs.iterrows():
            # Stolen math from Melchior
            if not np.isnan(arc['Extrusion Direction X']):
                rotaxang = [arc['Extrusion Direction X'],
                            arc['Extrusion Direction Y'],
                            arc['Extrusion Direction Z'],
                            arc['Total Angle']]
                rad = np.linspace(arc['Start Angle'], arc['Start Angle'] +
                                  arc['Total Angle'])
                dx = np.sin(np.deg2rad(rad)) * arc['Radius']
                dy = np.cos(np.deg2rad(rad)) * arc['Radius']
                dz = np.zeros(dx.shape[0])
                phi1 = -np.arctan2(
                    norm(np.cross(np.array([rotaxang[0], rotaxang[1], rotaxang[2]]),
                                  np.array([0, 0, 1]))),
                    np.dot(np.array([rotaxang[0], rotaxang[1], rotaxang[2]]),
                           np.array([0, 0, 1])))
                DX = dx * np.cos(phi1) + dz * np.sin(phi1)
                DY = dy
                DZ = dz * np.cos(phi1) - dx * np.sin(phi1)
                # ax.plot(DX, DY, DZ, color='r')
                phi2 = np.arctan(rotaxang[1] / rotaxang[0])
                fdx = (DX * np.cos(phi2)) - (DY * np.sin(phi2))
                fdy = (DX * np.sin(phi2)) + (DY * np.cos(phi2))
                fdz = DZ
                x = fdx + arc['Center X']
                y = fdy + arc['Center Y']
                z = fdz + arc['Center Z']
                # projected pts
                pts = np.column_stack([x, y, z])
                proj_pts = np.dot(pts - origin, normal)[:, None] * normal
                proj_pts = pts - origin - proj_pts
                proj_pts = np.matmul(change_b_mat, proj_pts.T)
                ax3d.plot(x, y, z, color='darkgray', zorder=110)
                ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='darkgray',
                          zorder=110)
                ax_map.plot(x, y, color='darkgray')
            elif not np.isnan(arc['Start X']):
                v1 = -1. * np.array([arc['Center X'] - arc['Start X'],
                                     arc['Center Y'] - arc['Start Y'],
                                     arc['Center Z'] - arc['Start Z']])
                v2 = -1. * np.array([arc['Center X'] - arc['End X'],
                                     arc['Center Y'] - arc['End Y'],
                                     arc['Center Z'] - arc['End Z']])
                rad = np.linspace(0, np.deg2rad(arc['Total Angle']), 50)
                # get rotation vector (norm is rotation angle)
                rotvec = np.cross(v2, v1)
                rotvec /= norm(rotvec)
                rotvec = rotvec[:, np.newaxis] * rad[np.newaxis, :]
                Rs = R.from_rotvec(rotvec.T)
                pt = np.matmul(v1, Rs.as_matrix())
                # Projected pts
                x = arc['Center X'] + pt[:, 0]
                y = arc['Center Y'] + pt[:, 1]
                z = arc['Center Z'] + pt[:, 2]
                pts = np.column_stack([x, y, z])
                proj_pts = np.dot(pts - origin, normal)[:, None] * normal
                proj_pts = pts - origin - proj_pts
                proj_pts = np.matmul(change_b_mat, proj_pts.T)
                ax3d.plot(x, y, z, color='darkgray', zorder=110)
                ax_x.plot(proj_pts[0, :], proj_pts[1, :], color='darkgray',
                          zorder=110)
                ax_map.plot(x, y, color='darkgray')
    # Fault model
    fault_mod = '{}/faultmod.mat'.format(autocad_path)
    faultmod = loadmat(fault_mod, simplify_cells=True)['faultmod']
    x = faultmod['xq']
    y = faultmod['yq']
    zt = faultmod['zq_top']
    zb = faultmod['zq_bot']
    ax3d.plot_surface(x, y, zt, color='bisque', alpha=.5)
    ax3d.plot_surface(x, y, zb, color='bisque', alpha=.5)
    # Proj
    pts_t = np.column_stack([x.flatten(), y.flatten(), zt.flatten()])
    proj_pts_t = np.dot(pts_t - origin, normal)[:, None] * normal
    proj_pts_t = pts_t - origin - proj_pts_t
    proj_pts_t = np.matmul(change_b_mat, proj_pts_t.T)
    pts_b = np.column_stack([x.flatten(), y.flatten(), zb.flatten()])
    proj_pts_b = np.dot(pts_b - origin, normal)[:, None] * normal
    proj_pts_b = pts_b - origin - proj_pts_b
    proj_pts_b = np.matmul(change_b_mat, proj_pts_b.T)
    ax_x.fill(proj_pts_t[0, :], proj_pts_t[1, :], color='bisque', alpha=0.7,
              label='Main Fault')
    ax_x.fill(proj_pts_b[0, :], proj_pts_b[1, :], color='bisque', alpha=0.7)
    for well, pts in well_dict.items():
        if well[0] not in ['D', 'B']:
            continue
        try:
            col = csd_well_colors[well]
            zdr = 109
        except KeyError:
            col = 'lightgray'
            zdr = 90
        # Proj
        pts = pts[:, :3]
        proj_pts = np.dot(pts - origin, normal)[:, None] * normal
        proj_pts = pts - origin - proj_pts
        proj_pts = np.matmul(change_b_mat, proj_pts.T)
        ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col,
                  linewidth=1.5, zorder=zdr)
        ax_x.plot(proj_pts[0, :], proj_pts[1, :], color=col, zorder=zdr)
        ax_map.scatter(pts[:, 0][0], pts[:, 1][0], color=col, s=15.,
                       zorder=111)
        ax_map.annotate(text=well, xy=(pts[:, 0][0], pts[:, 1][1]), fontsize=10,
                        weight='bold', xytext=(3, 0),
                        textcoords="offset points", color=col)
    # Plot fault coords and piercepoints
    plot_pierce_points(x, y, zt, strike=47, dip=57, ax=ax_fault, well_dict=well_dict)
    # Formatting
    ax3d.set_xlim([2579295, 2579340])
    ax3d.set_ylim([1247555, 1247600])
    ax3d.set_zlim([485, 530])
    # ax3d.view_init(elev=30., azim=-112)
    ax3d.view_init(elev=32, azim=-175)
    ax3d.margins(0.)
    ax3d.set_xticks([])
    ax3d.set_xticklabels([])
    ax3d.set_yticks([])
    ax3d.set_yticklabels([])
    ax3d.set_zticks([])
    ax3d.set_zticklabels([])
    # Overview map
    ax_map.axis('equal')
    ax_map.axis('off')
    ax_map.set_xlim([2579300, 2579338])
    ax_map.set_ylim([1247540, 1247582])
    # Fault map
    ax_fault.axis('equal')
    ax_fault.spines['top'].set_visible(False)
    ax_fault.spines['left'].set_visible(False)
    ax_fault.spines['right'].set_visible(False)
    ax_fault.spines['bottom'].set_bounds(-5, 5)
    ax_fault.tick_params(direction='in', left=False, labelleft=False)
    ax_fault.set_xticks([-5, 0, 5])
    ax_fault.set_xticklabels(['0', '5', '10'])
    ax_fault.set_xlabel('Meters')
    # Cross section
    ax_x.set_xlim([-30, 5])
    ax_x.axis('equal')
    ax_x.spines['top'].set_visible(False)
    ax_x.spines['bottom'].set_visible(False)
    ax_x.spines['left'].set_visible(False)
    ax_x.yaxis.set_ticks_position('right')
    ax_x.tick_params(direction='in', bottom=False, labelbottom=False)
    ax_x.set_yticks([-30, -20, -10, 0])
    ax_x.set_yticklabels(['30', '20', '10', '0'])
    ax_x.set_ylabel('Meters', labelpad=15)
    ax_x.yaxis.set_label_position("right")
    ax_x.spines['right'].set_bounds(0, -30)
    fig.legend()
    plt.show()
    return


def create_FSB_boreholes(asbuilt_dir):
    """
    Return dictionary of FSB well coordinates

    :param asbuilt_dir: Directory holding the gamma logs for each well
    """
    well_dict = {}
    excel_asbuilts = glob('{}/**/*Gamma_Deviation.xlsx'.format(asbuilt_dir),
                          recursive=True)
    gocad_asbuilts =  glob('{}/*.wl'.format(asbuilt_dir))
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


def get_well_piercepoint(wells, well_dict):
    """
    Return the xyz points of the main fault for a list of wells

    :param wells: List
    :return:
    """
    pierce_dict = {}
    for well in wells:
        pierce_dict[well] = {'top': depth_to_xyz(well_dict, well,
                                                 fault_depths[well][0])}
        pierce_dict[well]['bottom'] = depth_to_xyz(well_dict, well,
                                                   fault_depths[well][1])
    return pierce_dict


def plot_pierce_points(x, y, z, strike, dip, ax, well_dict):
    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    origin = np.array((np.nanmean(x), np.nanmean(y), np.nanmean(z)))
    normal = np.array((np.sin(d) * np.cos(s), -np.sin(d) * np.sin(s),
                       np.cos(d)))
    strike_new = np.array([np.sin(s), np.cos(s), 0])
    up_dip = np.array([-np.cos(s) * np.cos(d), np.sin(s) * np.cos(d), np.sin(d)])
    change_B_mat = np.array([strike_new, up_dip, normal])
    grid_pts = np.subtract(np.column_stack([x, y, z]), origin)
    newx, newy, newz = change_B_mat.dot(grid_pts.T)
    newx = newx[~np.isnan(newx)]
    newy = newy[~np.isnan(newy)]
    pts = np.column_stack([newx, newy])
    hull = ConvexHull(pts)
    pierce_points = get_well_piercepoint(['D1', 'D2', 'D3', 'D4', 'D5',
                                          'D6', 'D7'], well_dict)
    ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color='white',
            alpha=0.0)
    size = 20.
    fs = 10
    # Plot well pierce points
    projected_pts = {}
    for well, pts in pierce_points.items():
        try:
            col = csd_well_colors[well]
        except KeyError as e:
            col = cols_4850[well]
        p = np.array(pts['top'])
        # Project onto plane in question
        proj_pt = p - (normal.dot(p - origin)) * normal
        trans_pt = proj_pt - origin
        new_pt = change_B_mat.dot(trans_pt.T)
        ax.scatter(new_pt[0], new_pt[1], marker='+', color='k', s=size,
                   zorder=103)
        ax.annotate(text=well, xy=(new_pt[0], new_pt[1]), fontsize=fs,
                    weight='bold', xytext=(3, 0),
                    textcoords="offset points", color=col)
        projected_pts[well] = new_pt
    return projected_pts


if __name__ in '__main__':
    plot_CSD_2D('/media/chopp/Data1/chet-FS-B/Mont_Terri_model/APR_gallery_files',
                '/media/chopp/Data1/chet-FS-B/wells/FSB_gamma')