#!/usr/bin/python

"""
Grid read/write utilities for NLLoc, hypoDD, etc... using xarray
"""

import pyproj

import numpy as np
import xarray as xr


def read_array(path):
    arr = xr.open_dataset(path)
    return arr


def write_simul2000(dataset, outfile):
    """
    Write a simul2000-formatted file from cascadia xarray for hypoDD 2.1b

    :param vp_array: Xarray DataSet for cascadia 3D model
    :param outfile: Path to model file
    :return:
    """
    # Hardcoded
    vp = dataset['Vp'][1300::80, 500::80, 0::10].copy()
    vs = dataset['Vs'][1300::80, 500::80, 0::10].copy()
    vp = xr.concat([vp.isel(depth=0), vp], dim='depth')
    vs = xr.concat([vs.isel(depth=0), vs], dim='depth')
    vp = xr.concat([vp.isel(Northing=0), vp], dim='Northing')
    vs = xr.concat([vs.isel(Northing=0), vs], dim='Northing')
    vp = xr.concat([vp.isel(Easting=0), vp], dim='Easting')
    vs = xr.concat([vs.isel(Easting=0), vs], dim='Easting')
    vp = xr.concat([vp, vp.isel(depth=-1)], dim='depth')
    vs = xr.concat([vs, vs.isel(depth=-1)], dim='depth')
    vp = xr.concat([vp, vp.isel(Northing=-1)], dim='Northing')
    vs = xr.concat([vs, vs.isel(Northing=-1)], dim='Northing')
    vp = xr.concat([vp, vp.isel(Easting=-1)], dim='Easting')
    vs = xr.concat([vs, vs.isel(Easting=-1)], dim='Easting')
    # Edit coordinates for the periphery planes
    new_dc = vp.coords['depth'].values
    new_dc[0] = -2000
    new_dc[-1] = 600000
    new_east = vp.coords['Easting'].values
    new_east[0] = -500000
    new_east[-1] = 900000
    new_north = vp.coords['Northing'].values
    new_north[0] = -500000
    new_north[-1] = 900000
    vp.assign_coords(depth=new_dc)
    vs.assign_coords(depth=new_dc)
    vp.assign_coords(Easting=new_east)
    vs.assign_coords(Easting=new_east)
    vp.assign_coords(Northing=new_north)
    vs.assign_coords(Northing=new_north)
    # With above indexing, SW vertex is: (-126.3779, 46.1593, -2.5)
    # SE vertex (considered origin in simul) is: (-121.1441, 46.1593, -2.5)
    # Make origin 0, 0, 0 at SE corner (and West is positive!!)
    utm_grid = np.meshgrid((vp.coords['Easting'].values[-1] -
                            vp.coords['Easting'].values)[::-1],
                           (vp.coords['Northing'].values -
                            vp.coords['Northing'].values[0]))
    # Now write the file
    # Loop over Y inside Z with X (cartesian) varying along file row
    with open(outfile, 'w') as f:
        f.write('{:4.1f}{:3d}{:3d}{:3d}\n'.format(
            1.0, vp.coords['Easting'].size,
            vp.coords['Northing'].size,
            vp.coords['depth'].size))
        np.savetxt(f, utm_grid[0][0, :].reshape(
            1, utm_grid[0].shape[1]) / 1000., fmt='%6.1f')
        np.savetxt(f, utm_grid[1][:, 0].reshape(
            1, utm_grid[0].shape[0]) / 1000., fmt='%6.1f')
        np.savetxt(f, (new_dc / 1000.).reshape(
            1, new_dc.shape[0]), fmt='%6.1f')
        f.write('  0  0  0\n  0  0  0\n')  # Whatever these are...
        for i, z in enumerate(vp.coords['depth']):
            for j, y in enumerate(vp.coords['Northing']):
                row_vals = vp.isel(depth=i, Northing=j).values[::-1] / 1000.
                row_vals = row_vals.reshape(1, row_vals.shape[0])
                np.savetxt(f, row_vals, fmt='%5.2f')
        # Finally Vp/Vs ratio
        for i, z in enumerate(vp.coords['depth']):
            for j, y in enumerate(vp.coords['Northing']):
                row_vals = (vp.isel(depth=i, Northing=j) /
                            vs.isel(depth=i, Northing=j)).values[::-1]
                row_vals = row_vals.reshape(1, row_vals.shape[0])
                row_vals[row_vals == np.inf] = 0.0
                np.savetxt(f, row_vals, fmt='%5.2f')
    return


def write_NLLoc_grid(dataset, outdir):
    """
    Write NLLoc grids for cascadia array
    :return:
    """
    return