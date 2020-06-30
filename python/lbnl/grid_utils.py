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
    new_dc = np.insert(dep_coords, 0, np.array([-2.5, -2., -1.5, -1., -0.5]))
    for i in range(5):
        vp = xr.concat([vp[:, :, 0], vp], dim='depth')
        vs = xr.concat([vs[:, :, 0], vs], dim='depth')
    vp.assign_coords(depth=new_dc)
    vs.assign_coords(depth=new_dc)
    # With above indexing, grid origin is: (-126.3779, 46.1593, -2.5)
    utm = pyproj.Proj(init="EPSG:32610")
    utm_grid = np.meshgrid(vp.coords['Easting'].values,
                           vp.coords['Northing'].values)
    lon, _ = utm(utm_grid[0][0, :], utm_grid[1][0, :], inverse=True)
    _, lat = utm(utm_grid[0][:, 0], utm_grid[1][:, 0], inverse=True)
    # Now write the file
    # Loop over Y inside Z with X (cartesian) varying along file row
    with open(outfile, 'w') as f:
        f.write('{},{},{},{}\n'.format(1.0, vp.coords['Easting'].size,
                                       vp.coords['Northing'].size,
                                       vp.coords['depth'].size))
        np.savetxt(f, lon, delimiter=',', newline=" ", fmt='%0.4f')
        np.savetxt(f, lat, delimiter=',', newline=" ", fmt='%0.4f')
        np.savetxt(f, vp.coords['depth'].values, delimiter=',',
                   newline=" ", fmt='%0.4f')
        f.write('0,0,0\n0,0,0\n')  # Whatever these are...
        for i, z in enumerate(vp.coords['depth']):
            for j, y in enumerate(vp.coords['Northing']):
                np.savetxt(f, vp.isel(depth=i, Northing=j).values / 1000.,
                           delimiter=',', newline=" ", fmt='%0.3f')
        # Finally Vp/Vs ratio
        for i, z in enumerate(vp.coords['depth']):
            for j, y in enumerate(vp.coords['Northing']):
                np.savetxt(f, vp.isel(depth=i, Northing=j) /
                           vs.isel(depth=i, Northing=j),
                           delimiter=',', newline=" ", fmt='%0.3f')
    return


def write_NLLoc_grid(dataset, outdir):
    """
    Write NLLoc grids for cascadia array
    :return:
    """
    return