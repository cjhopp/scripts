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
    vp = dataset['Vp'][1300:, 500:, :].copy()
    vs = dataset['Vs'][1300:, 500:, :].copy()
    # Add five depth layers to dataset (ugly as), theres a Dataset concat...
    dep_coords = vp.coords['depth'].values
    new_dc = np.insert(dep_coords, 0, np.array([-2500, -2000, -1500,
                                                -1000, -500]))
    for i in range(5):
        vp = xr.concat([vp[:, :, 0], vp], dim='depth')
        vs = xr.concat([vs[:, :, 0], vs], dim='depth')
    vp.assign_coords(depth=new_dc)
    vs.assign_coords(depth=new_dc)
    # With above indexing, SW vertex is: (-126.3779, 46.1593, -2.5)
    # SE vertex (considered origin in simul) is: (-121.1441, 46.1593, -2.5)
    # Make origin 0, 0, 0 at SE corner (and West is positive!!)
    utm_grid = np.meshgrid(vp.coords['Easting'].values[-1] - vp.coords['Easting'].values,
                           vp.coords['Northing'].values - vp.coords['Northing'].values[0])
    # Now write the file
    # Loop over Y inside Z with X (cartesian) varying along file row
    with open(outfile, 'w') as f:
        f.write('{} {} {} {}\n'.format(1.0, vp.coords['Easting'].size,
                                       vp.coords['Northing'].size,
                                       vp.coords['depth'].size))
        np.savetxt(f, utm_grid[0][0, :].reshape(
            1, utm_grid[0].shape[1]) / 1000., fmt='%6.1f')
        np.savetxt(f, utm_grid[1][:, 0].reshape(
            1, utm_grid[0].shape[0]) / 1000., fmt='%6.1f')
        np.savetxt(f, (new_dc / 1000.).reshape(
            1, new_dc.shape[0]), fmt='%6.1f')
        f.write('0 0 0\n0 0 0\n')  # Whatever these are...
        for i, z in enumerate(vp.coords['depth']):
            for j, y in enumerate(vp.coords['Northing']):
                row_vals = vp.isel(depth=i, Northing=j).values[::-1] / 1000.
                np.savetxt(f, row_vals.reshape(1, row_vals.shape[0]),
                           fmt='%5.2f')
        # Finally Vp/Vs ratio
        for i, z in enumerate(vp.coords['depth']):
            for j, y in enumerate(vp.coords['Northing']):
                row_vals = (vp.isel(depth=i, Northing=j) /
                            vs.isel(depth=i, Northing=j)).values[::-1]
                np.savetxt(f, row_vals.reshape(1, row_vals.shape[0]),
                           fmt='%5.2f')
    return


def write_NLLoc_grid(dataset, outdir):
    """
    Write NLLoc grids for cascadia array
    :return:
    """
    return