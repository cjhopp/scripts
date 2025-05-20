import math
import random
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union


def grid_candidates(poly: Polygon, step: float) -> list:
    """Rectilinear grid clipped to `poly`."""
    minx, miny, maxx, maxy = poly.bounds
    nx = max(2, int(math.ceil((maxx - minx) / step)) + 1)
    ny = max(2, int(math.ceil((maxy - miny) / step)) + 1)
    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    pts = [Point(x, y) for x in xs for y in ys]
    return [p for p in pts if poly.contains(p)]


def polar_candidates(poly: Polygon,
                     center: Point,
                     R1: float,
                     Rmax: float,
                     dr: float,
                     M: int) -> list:
    """
    Build a polar grid in the annulus [R1, Rmax]:
      - M rays (angles) evenly spaced over [0,2π)
      - rings at radii R1 + k*dr
    Returns a list of M lists: each sublist is the points on that ray.
    """
    thetas = np.linspace(0, 2*math.pi, M, endpoint=False)
    n_rings = int(math.floor((Rmax - R1) / dr)) + 1
    radii = [R1 + k*dr for k in range(n_rings)]

    rays = []
    for θ in thetas:
        cosθ, sinθ = math.cos(θ), math.sin(θ)
        pts_on_ray = []
        for r in radii:
            x = center.x + r*cosθ
            y = center.y + r*sinθ
            p = Point(x, y)
            if poly.contains(p):
                pts_on_ray.append(p)
        rays.append(pts_on_ray)
    return rays


def two_zone_polar_grid(
    poly: Polygon,
    center: Point,
    N_total: int,
    R1: float,
    s1: float,
    Rmax: float,
    dr: float,
    M: int,
    roads_gdf: gpd.GeoDataFrame = None,
    road_buffer: float = 0.0,
    plot: bool = False
):
    """
    1) Clip poly to circle(Rmax) and optional roads.buffer(road_buffer)
    2) Carve inner = ∩ circle(R1), outer = difference
    3) Build ALL inner-grid pts at spacing=s1, take up to N_total of them
    4) If there's remainder N_rem, build polar outer rays and round‐robin
       pick exactly N_rem from them.
    Returns (inner_pts, outer_pts).
    """
    # 0) clip to max‐radius
    working = poly.intersection(center.buffer(Rmax))

    # 1) carve out inner/outer from the study polygon
    inner_poly = poly.intersection(center.buffer(R1))
    outer_poly = poly.intersection(center.buffer(Rmax)).difference(inner_poly)

    # 2) sample the inner grid
    cand_in = grid_candidates(inner_poly, s1)
    if len(cand_in) >= N_total:
        # pick exactly N_total *without* numpy
        inner_pts = random.sample(cand_in, N_total)
        outer_pts = []
        return inner_pts, outer_pts

    inner_pts = cand_in.copy()
    N_rem    = N_total - len(inner_pts)

    # 3) clip outer to roads corridor *after* filling inner
    if roads_gdf is not None and road_buffer > 0:
        roads_union = unary_union(roads_gdf.geometry)
        corridor    = roads_union.buffer(road_buffer)
        outer_poly  = outer_poly.intersection(corridor)

    # 4) build polar‐grid rays & round-robin sample N_rem
    rays = polar_candidates(outer_poly, center, R1, Rmax, dr, M)
    outer_pts = []
    idx = [0]*M
    while len(outer_pts) < N_rem:
        any_picked = False
        for i in range(M):
            if len(outer_pts) >= N_rem:
                break
            if idx[i] < len(rays[i]):
                outer_pts.append(rays[i][idx[i]])
                idx[i] += 1
                any_picked = True
        if not any_picked:
            break

    # 5) fallback fill from leftovers, also with random.sample
    if len(outer_pts) < N_rem:
        leftovers = []
        for i, ray in enumerate(rays):
            leftovers.extend(ray[idx[i]:])
        need = N_rem - len(outer_pts)
        if len(leftovers) < need:
            raise ValueError("Not enough outer candidates to fill remainder")
        outer_pts.extend(random.sample(leftovers, need))

    # 5) optional plot
    if plot:
        fig, ax = plt.subplots(figsize=(6, 6))

        def plot_geom(g, **kw):
            if isinstance(g, Polygon):
                xs, ys = g.exterior.xy
                ax.plot(xs, ys, **kw)
                for h in g.interiors:
                    xh, yh = h.xy
                    ax.plot(xh, yh, **kw, linestyle="--")
            elif isinstance(g, MultiPolygon):
                for part in g.geoms:
                    plot_geom(part, **kw)

        # base polygon
        plot_geom(poly, color="k", linewidth=1)

        # optional corridor
        if roads_gdf is not None and road_buffer > 0:
            plot_geom(corridor, color="brown", alpha=0.3)

        # circles
        x1, y1 = center.buffer(R1).exterior.xy
        ax.plot(x1, y1, "C1:", label=f"R1={R1}")
        x2, y2 = center.buffer(Rmax).exterior.xy
        ax.plot(x2, y2, "C1--", label=f"Rmax={Rmax}")

        # plot pts
        ax.scatter([p.x for p in inner_pts],
                   [p.y for p in inner_pts],
                   c="C1", s=20, label=f"inner ({len(inner_pts)})")
        ax.scatter([p.x for p in outer_pts],
                   [p.y for p in outer_pts],
                   c="C0", s=20, label=f"outer ({len(outer_pts)})")

        # center marker
        ax.scatter(center.x, center.y,
                   marker="*", c="gold", s=150,
                   edgecolor="k", label="center")

        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        # plt.show()
        return inner_pts, outer_pts, fig, ax
    return inner_pts, outer_pts


# --------------------
# Example of use:
# --------------------
if __name__ == '__main__':
    # 1) read the study polygon(s)
    pg = gpd.read_file("/home/chopp/QGIS/projects/chet-ingeneous/GSV_nodable_land.shp")
    study_poly = unary_union(pg.geometry)

    # 2) read your roads
    roads = gpd.read_file("/home/chopp/QGIS/projects/chet-ingeneous/Nodable_roads_UTM.shp")
    point = gpd.read_file('/home/chopp/QGIS/projects/chet-ingeneous/Center_of_anolmaly.shp').geometry.values[0]
    # 3) call the sampler, passing roads & a buffer distance
    res = two_zone_polar_grid(
        poly      = study_poly,
        center    = point,
        N_total   = 50,
        R1        = 5000,
        s1        = 1500,
        Rmax      = 15000,
        dr        = 100,
        M         = 36,
        roads_gdf = roads,          # <<< here!
        road_buffer = 50.0,         # 50 m around roads
        plot      = True
    )
    if len(res) == 4:
        inner, outer, fig, ax = res
        # now explicitly show (or save) your figure:
        plt.show()       # or plt.show()
    else:
        inner, outer = res
    # Write inner and out points to a joint shapefile
    inner_gdf = gpd.GeoDataFrame(geometry=inner)
    outer_gdf = gpd.GeoDataFrame(geometry=outer)
    inner_gdf.to_file("/home/chopp/QGIS/projects/chet-ingeneous/inner_points.shp")
    outer_gdf.to_file("/home/chopp/QGIS/projects/chet-ingeneous/outer_points.shp")
    print("got", len(inner), "inner and", len(outer), "outer points")
