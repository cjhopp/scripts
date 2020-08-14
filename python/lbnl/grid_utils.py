#!/usr/bin/python

"""
Grid read/write utilities for NLLoc, hypoDD, etc... using xarray
"""

import pyproj

import numpy as np
import xarray as xr

from pyproj import Proj


def read_array(path):
    arr = xr.open_dataset(path)
    return arr


def write_simul2000(dataset, outfile, resample_factor=2):
    """
    Write a simul2000-formatted file from cascadia xarray for hypoDD 2.1b

    :param vp_array: Xarray DataSet for cascadia 3D model
    :param outfile: Path to model file
    :return:
    """
    rf = resample_factor
    # Hardcoded
    vp = dataset['Vp'][1300::rf, 500::rf, 0::rf].copy()
    vs = dataset['Vs'][1300::rf, 500::rf, 0::rf].copy()
    # Add far-field planes to models (6 faces, 2 models)
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
    orig = (np.median(vp.coords['Easting'].values),
            np.median(vp.coords['Northing'].values),
            np.median(vp.coords['depth'].values))
    new_dc_p = vp.coords['depth'].values
    new_dc_p[0] = -20000
    new_dc_p[-1] = 999000
    new_east_p = vp.coords['Easting'].values
    new_east_p -= np.median(new_east_p).astype(np.int64)
    new_east_p[0] = -999000
    new_east_p[-1] = 999000
    new_north_p = vp.coords['Northing'].values
    new_north_p -= np.median(new_north_p).astype(np.int64)
    new_north_p[0] = -999000
    new_north_p[-1] = 999000
    new_dc_s = vs.coords['depth'].values
    new_dc_s[0] = -20000
    new_dc_s[-1] = 999000
    new_east_s = vs.coords['Easting'].values
    new_east_s -= np.median(new_east_s).astype(np.int64)
    new_east_s[0] = -999000
    new_east_s[-1] = 999000
    new_north_s = vs.coords['Northing'].values
    new_north_s -= np.median(new_north_s).astype(np.int64)
    new_north_s[0] = -999000
    new_north_s[-1] = 999000
    vp.assign_coords(Easting=new_east_p[::-1],
                     Northing=new_north_p[::-1],
                     depth=new_dc_p[::-1])
    vs.assign_coords(Easting=new_east_s[::-1],
                     Northing=new_north_s[::-1],
                     depth=new_dc_s[::-1])
    # With above indexing, SW vertex is: (-126.3779, 46.1593, -2.5)
    # SE vertex (considered origin in simul) is: (-121.1441, 46.1593, -2.5)
    # Make origin 0, 0, 0 at SE corner (and West is positive!!)
    utm_grid = np.meshgrid(vp.coords['Easting'].values,
                           vp.coords['Northing'].values)
    # Now write the file
    # Loop over Y inside Z with X (cartesian) varying along file row
    with open(outfile, 'w') as f:
        f.write('{:4.1f}{:3d}{:3d}{:3d}\n'.format(
            1.0, vp.coords['Easting'].size,
            vp.coords['Northing'].size,
            vp.coords['depth'].size))
        np.savetxt(f, utm_grid[0][0, :].reshape(
            1, utm_grid[0].shape[1]) / 1000., fmt='%4.0f.',
                   delimiter=' ')
        np.savetxt(f, utm_grid[1][:, 0].reshape(
            1, utm_grid[0].shape[0]) / 1000., fmt='%4.0f.',
                   delimiter=' ')
        np.savetxt(f, (new_dc_p / 1000.).reshape(
            1, new_dc_p.shape[0]), fmt='%4.0f.',
                   delimiter=' ')
        f.write('  0  0  0\n\n')  # Whatever these are...
        for i, z in enumerate(vp.coords['depth']):
            for j, y in enumerate(vp.coords['Northing']):
                row_vals = vp.isel(depth=i, Northing=j).values[::-1] / 1000.
                row_vals = row_vals.reshape(1, row_vals.shape[0])
                np.savetxt(f, row_vals, fmt='%4.2f',
                           delimiter=' ')
        # Finally Vp/Vs ratio
        for i, z in enumerate(vp.coords['depth']):
            for j, y in enumerate(vp.coords['Northing']):
                row_vals = (vp.isel(depth=i, Northing=j) /
                            vs.isel(depth=i, Northing=j)).values[::-1]
                row_vals = row_vals.reshape(1, row_vals.shape[0])
                row_vals[row_vals == np.inf] = 1.78
                np.savetxt(f, row_vals, fmt='%5.3f',
                           delimiter=' ')
    # Return grid origin
    p = Proj(init="EPSG:32610")
    return p(orig[0], orig[1], inverse=True), orig[-1]


def write_NLLoc_grid(dataset, outdir):
    """
    Write NLLoc grids for cascadia array
    :return:
    """
    return