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
    vp = dataset['Vp'][1300:, 500:, :]
    # Add five depth layers to dataset (ugly as)
    for i in range(5):
        vp.values = np.insert(vp.values, 0, vp.values[:, :, 0], axis=2)
    # With above indexing, grid origin is: (-126.3779, 46.1593, -2.5)
    utm = pyproj.Proj(init="EPSG:32610")
    lon, lat = utm(vp.coords['Easting'].values, vp.coords['Northing'].values,
                   inverse=True)
    # Now write the file
    # Loop over Y inside Z with X (cartesian) varying along file row
    with open(outfile, 'w') as f:
        f.write('{},{},{},{}\n'.format(1.0, vp['Vp'].coords['Easting'].size,
                                     vp['Vp'].coords['Northing'].size,
                                     vp['Vp'].coords['depth'].size))
        np.savetxt(lon, delimiter=',', newline='\n', format='{:0.4f}')
        np.savetxt(lat, delimiter=',', newline='\n', format='{:0.4f}')
        np.savetxt(vp['Vp'].coords['depth'].values, delimiter=',',
                   newline='\n', format='{:0.4f}')
        f.write('0,0,0\n0,0,0\n')  # Whatever these are...
        for z in vp['Vp'].coords['depth'].values:
            for y in lat:
                np.savetxt(vp.isel(depth=z, Northing=y)['Vp'] / 1000.,
                           delimiter=',', newline='\n', format='{:0.3f}')
        # Finally Vp/Vs ratio
        for z in vp['Vp'].coords['depth'].values:
            for y in lat:
                np.savetxt(vp.isel(depth=z, Northing=y)['Vp'] /
                           vp.isel(depth=z, Northing=y)['Vs'],
                           delimiter=',', newline='\n', format='{:0.3f}')
    return


def write_NLLoc_grid(dataset, outdir):
    """
    Write NLLoc grids for cascadia array
    :return:
    """
    return