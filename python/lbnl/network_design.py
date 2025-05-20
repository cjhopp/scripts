#!/usr/bin/python

"""
Set of functions for network design following Zara Rawlinson's MSc thesis
at VUW

Assumes straight ray paths and perfectly known hypocenters
"""
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from numpy.linalg import LinAlgError
from scipy.spatial import ConvexHull, Delaunay
from numpy.linalg import det
from scipy.stats import dirichlet
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union

# Local imports
from lbnl.boreholes import make_4100_boreholes

def calc_amplitude(t, f, q):
    return np.exp(-t * f * np.pi / q) + 0.00001


def calc_tt(hypocenters, stations, vp):
    # Need 3D: m stations (rows), 3 coords, n events (cols)
    # Distances
    diff = stations[:, :, None] - hypocenters.T[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=1))
    return dist / vp


def meas_var(tts, f, q):
    # Measurement variance estimate
    # tts and amps are mxn, m stations, n events
    amps = calc_amplitude(tts, f, q)
    # Sum along n, prod along m
    return np.prod(np.mean(1 / amps**2, axis=1), axis=0)


def hyp_var(tts):
    varcov_tt = np.cov(tts)
    return np.abs(np.linalg.det(varcov_tt))


def design_criterion(stas, hypocenters, f, q, vp):
    """
    Calculate the design criteria for a set of stations and hypocenters

    :param stas: Array of station locations
    :param hypocenters: Array of hypocenter locations
    :param f: Representative corner freq for events [Hz]
    :param q: Representative quality factor
    :param vp: P-wave velocity [m/s]
    :return:
    """
    # Reshape to n x 3
    stas = stas.reshape(-1, 3)
    # Compute travel times
    tts = calc_tt(hypocenters, stas, vp)
    # Measurement variance
    Vmeas = meas_var(tts, f, q)
    # print(np.none(np.isnan(Vmeas)))
    try:
        # Hypocenter variance
        Vhyp = hyp_var(tts)
        return np.log(Vhyp) - np.log(Vmeas)
    except LinAlgError as e:
        # Case of one station, one source
        return -np.log(Vmeas)


def generate_hypocenters(points, n):
    """
    Generate randomly distributed hypocenters within a volume

    From SO answer:
    https://stackoverflow.com/questions/59073952/how-to-get-uniformly-distributed-points-in-convex-hull

    :param well_dict: Dictionary of well points
    :param n: Number of points to draw

    :return:
    """
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = points[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] -
                      deln[:, dims:, :])) / np.math.factorial(dims)
    sample = np.random.choice(len(vols), size=n, p=vols / vols.sum())

    pts = np.einsum('ijk, ij -> ik',
                    deln[sample], dirichlet.rvs([1] * (dims + 1), size=n))
    return pts


def generate_4100_source_verts(well_dict):
    """
    Take borehole trajectories and return a set of outer points defining
    a volume where we expect events

    :return:
    """
    dml_pts = well_dict['E2-DML']
    dmu_pts = well_dict['E2-DMU']
    aml_pts = well_dict['E2-AML']
    amu_pts = well_dict['E2-AMU']
    hull_pts = []
    # First get well intersection pt depths
    for well in ['E2-TN', 'E2-TL', 'E2-TU', 'E2-TS']:
        all_pts = well_dict[well]
        dists_dml = np.sqrt(np.sum((dml_pts - all_pts)**2, axis=1))
        dists_dmu = np.sqrt(np.sum((dmu_pts - all_pts)**2, axis=1))
        dists_aml = np.sqrt(np.sum((aml_pts - all_pts)**2, axis=1))
        dists_amu = np.sqrt(np.sum((amu_pts - all_pts)**2, axis=1))
        closest_shallow = np.argmin(np.min(np.vstack([dists_dml, dists_dmu]),
                                           axis=0))
        closest_deep = np.argmin(np.min(np.vstack([dists_aml, dists_amu]),
                                        axis=0))
        hull_pts.append(all_pts[closest_shallow, :])
        # Deeper pt
        hull_pts.append(all_pts[closest_deep, :])
    return np.array(hull_pts)


def design_network(trial_pts, no_sensors, sources, f=10000, q=200, vp=3000,
                   plot=True, well_dict=None, outdir=None):
    """
    Run Zara's network design implementation

    :param geometry: Accepts 'wells' or 'surface' for now
    :param no_sensors: Maximum number of sensors for the network
    :param sources: Array of source points
    :param source_volume: Give a shapely(?) volume to distribute sources into
    :param sensors_p_well: Optional dict giving the maximum number of sensors
        per well

    :return:
    """
    # Build from one sensor up to no_sensors
    # Preallocate 1d array len n stations x 3 coordinates
    old_stas = np.array([])
    targets = []
    for s in range(no_sensors):
        print('Sensor {}'.format(s))
        # TODO https://stackoverflow.com/questions/22424096/apply-functions-to-3d-numpy-array
        if old_stas.size == 0:
            stas = trial_pts
        else:
            stas = np.broadcast_to(old_stas, (trial_pts.shape[0],
                                              old_stas.shape[0]))
            stas = np.concatenate((stas, trial_pts), axis=1)
        target = np.apply_along_axis(design_criterion, 1, stas,
                                     hypocenters=sources, f=f, q=q, vp=vp)
        # Save resulting image
        targets.append(target)
        loc = np.nanargmax(target)
        final_sta = trial_pts[loc, :]
        print('Next station: {}'.format(final_sta))
        old_stas = np.concatenate((old_stas, final_sta))
    if plot:
        plot_solution(targets, trial_pts, old_stas, sources,
                      well_dict=well_dict, outdir=outdir)
    return old_stas


def test_design():
    # Make 3d array of no-sensors x pts x iteration
    xs, ys = np.mgrid[-100:100:1, -100:100:1]
    zs = np.zeros(xs.shape)
    trial_pts = np.vstack([xs.flatten(), ys.flatten(), zs.flatten()]).T
    # Test sources
    sources = np.array([[0, 0, 100], [10, 10, 50], [50, 50, 25], [10, 0, 50],
                        [0, 10, 400], [0, 0, 5], [0, 0, 43]])
    design_network(trial_pts=trial_pts, no_sensors=6, sources=sources,
                   plot=True)
    return


def design_4100(well_file, n_sources, n_sensors, f, q, vp, outdir=None):
    """
    Run sensor placement algorithm for Collab 4100L

    :param well_file: Path to well trajectory file
    :param n_sources: Integer number of sources to populate
    :param n_sensors: Integer number of sensors to place
    :param f: Representative corner freq
    :param q: Representative quality factor
    :param vp: Homogenous p velocity
    :return:
    """
    well_dict = make_4100_boreholes(well_file)
    verts = generate_4100_source_verts(well_dict=well_dict)
    sources = generate_hypocenters(verts, n_sources)
    # Just try all points in monitoring wells
    trial_pts = np.vstack([pts for w, pts in well_dict.items()
                           if w in ['E2-AML', 'E2-AMU', 'E2-DML', 'E2-DMU']])
    stations = design_network(trial_pts, no_sensors=n_sensors,
                              sources=sources, f=f, q=q, vp=vp,
                              well_dict=well_dict, plot=True,
                              outdir=outdir)
    stations = stations.reshape(-1, 3)
    if outdir:
        np.savetxt('{}/station_locations.txt'.format(outdir), stations)
    return stations


def _grid_on_region(region, step, origin):
    """
    Perfect rectilinear grid of spacing=step, aligned so that
    origin lies on a grid node.  Returns all pts inside region.
    """
    minx, miny, maxx, maxy = region.bounds
    # offset so origin sits on a node
    dx = (origin.x - minx) % step
    dy = (origin.y - miny) % step
    xs = np.arange(minx + dx, maxx + step/2, step)
    ys = np.arange(miny + dy, maxy + step/2, step)
    pts = []
    for x in xs:
        for y in ys:
            p = Point(x, y)
            if region.contains(p):
                pts.append(p)
    return pts

def count_for_radius(poly, center, 
                     R_inner, s1, s2, R_max):
    """Return number of points if you clip to R_max, do two-zone grids."""
    clip_poly = poly.intersection(center.buffer(R_max))
    inner_reg = clip_poly.intersection(center.buffer(R_inner))
    outer_reg = clip_poly.difference(inner_reg)
    pts_in = _grid_on_region(inner_reg, s1, origin=center)
    pts_out = _grid_on_region(outer_reg, s2, origin=center)
    return len(pts_in) + len(pts_out)

def find_two_zone_grid(
    poly, center, 
    R_inner, s1, s2, 
    N_target,
    R_min=None, R_max=None,
    tol=0, max_iter=25,
    plot=False
):
    """
    Find a max_radius so that a perfect s1/s2 two-zone grid
    clipped to max_radius yields as close to N_target points
    as possible.  Returns (final_radius, pts_inner, pts_outer).
    """
    # 1) bounding search interval for R
    if R_min is None:
        R_min = R_inner
    if R_max is None:
        # worst-case upper bound: distance to farthest vertex of poly
        minx, miny, maxx, maxy = poly.bounds
        # circle covering entire bbox center→corner
        import math
        dx = max(abs(center.x - minx), abs(center.x - maxx))
        dy = max(abs(center.y - miny), abs(center.y - maxy))
        R_max = math.hypot(dx, dy) * 1.01

    # 2) bisection on R_max
    best = None  # (R, count, pts_in, pts_out)
    low, high = R_min, R_max
    for i in range(max_iter):
        mid = 0.5*(low + high)
        clip = poly.intersection(center.buffer(mid))
        inner = clip.intersection(center.buffer(R_inner))
        outer = clip.difference(inner)

        pts_in  = _grid_on_region(inner, s1, origin=center)
        pts_out = _grid_on_region(outer, s2, origin=center)
        cnt = len(pts_in) + len(pts_out)

        # keep track of best so far (closest to target)
        if best is None or abs(cnt - N_target) < abs(best[1] - N_target):
            best = (mid, cnt, pts_in, pts_out)

        if abs(cnt - N_target) <= tol:
            break

        # if we have too many points, shrink radius
        if cnt > N_target:
            high = mid
        else:
            low = mid

    R_final, cnt_final, pts_in_final, pts_out_final = best

    if plot:
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        # plot poly
        def _plot_poly(p, **kw):
            if isinstance(p, Polygon):
                x,y = p.exterior.xy; ax.plot(x,y,**kw)
                for hole in p.interiors:
                    xh,yh = hole.xy; ax.plot(xh,yh,**kw,linestyle='--')
            else:
                for sub in p.geoms:
                    _plot_poly(sub, **kw)

        _plot_poly(poly, color='k')
        # circles
        for r,c,ls in [(R_inner,'orange',':'),
                       (R_final,'gray','--')]:
            xc,yc = center.buffer(r).exterior.xy
            ax.plot(xc,yc,color=c,linestyle=ls)
        # points
        ax.scatter([p.x for p in pts_out_final],
                   [p.y for p in pts_out_final],
                   c='C0',s=20,alpha=0.6,label=f'outer ({len(pts_out_final)})')
        ax.scatter([p.x for p in pts_in_final],
                   [p.y for p in pts_in_final],
                   c='C1',marker='^',s=30,alpha=0.8,
                   label=f'inner ({len(pts_in_final)})')
        ax.scatter(center.x, center.y,
                   c='gold',marker='*',s=150,edgecolor='k')
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'N={cnt_final} @ R_max={R_final:.1f}')
        plt.show()

    return R_final, pts_in_final, pts_out_final


### Plotting funcs ###

def plot_solution(targets, points, stas, sources, well_dict=None,
                  outdir=None):
    for i, t in enumerate(targets):
        # Plot each on its own for now
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title('Sensor {}'.format(i + 1), fontsize=16)
        # Station pts
        plot_prev_stas = stas[:i*3].reshape(-1, 3)
        plot_current_sta = stas[i*3:i*3 + 3].reshape(-1, 3)
        cs = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=t,
                        s=0.5, alpha=0.8, cmap='magma')
        ax.scatter(plot_prev_stas[:, 0], plot_prev_stas[:, 1],
                   plot_prev_stas[:, 2], marker='^', color='k')
        ax.scatter(plot_current_sta[:, 0], plot_current_sta[:, 1],
                   plot_current_sta[:, 2], marker='^', color='b')
        ax.scatter(sources[:, 0], sources[:, 1], sources[:, 2],
                   marker='o', color='r', s=0.3)
        if well_dict:
            for w, pts in well_dict.items():
                if w[-2] == 'T':
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                            color='k', linewidth=0.75)
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.set_zlabel('Elevation')
        fig.colorbar(cs, ax=ax)
        if outdir:
            plt.savefig('{}/Sensor_{}.png'.format(outdir, i + 1), dpi=300)
            plt.close('all')
        else:
            plt.show()
    return