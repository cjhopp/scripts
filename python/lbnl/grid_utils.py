#!/usr/bin/python

"""
Grid read/write utilities for NLLoc, hypoDD, etc... using xarray
"""

import os
import gemgis
import pyproj

from nllgrid import NLLGrid

try:
    import rioxarray
except ImportError:
    pass

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from pyproj import Proj
from scipy.ndimage import shift
from pathlib import Path
from typing import Union, List, Sequence, Optional, Iterable, Dict, Tuple



# Gillian Foulger final 1D model at Newberry; 2100 m elevation as surface
foulger_elev = [2.1, 1.8, 1.65, 1.5, 1.35, 1.1, 0.2, -3.]
foulger_vp = [2.034, 3.33, 3.507, 3.673, 3.767, 3.8, 4.7]
foulger_vs = [1.236, 2.015, 2.122, 2.222, 2.279, 2.299, 2.843]



def read_array(path):
    arr = xr.open_dataset(path)
    return arr


def basin_gradient_p(depth):
    # Vp gradient model for Cape Modern basin sediments
    return 1.47 * np.log(7.0 * depth + 0.9, out=np.zeros_like(depth) - 0.107, where=(depth > 0.)) + 0.89


def basin_gradient_s(depth):
    # Vs gradient model for Cape Modern basin sediments
    return 1.57 * np.log(3.01 * depth + 1.79, out=np.zeros_like(depth) + 0.586, where=(depth > 0.)) - 0.52


def shift_model(topo, velocity):
    # Shift top of model to conform to topography
    return shift(velocity, topo, mode='nearest')


def extrapolate_tob_cape(path):
    tob_raw = rioxarray.open_rasterio(path)
    tob_50m = tob_raw.interp(x=np.linspace(315300, 345300, 601), y=np.linspace(4246200, 4276200, 601),
                             kwargs={'fill_value': 'extrapolate'})
    return tob_50m


def read_cape_topo(path):
    topo = rioxarray.open_rasterio(path)
    topo_reproj = topo.rio.reproject('epsg:26912', nodata=np.nan)
    topo_50m = topo_reproj.interp(x=np.linspace(315300, 345300, 601), y=np.linspace(4246200, 4276200, 601))
    return topo_50m


def read_ppmod_newberry(path):
    """
    Read an earth model file from sw4 pfile format (seems to be hardcoded for Newberry depths, etc...from E Matzel)

    :param path: Path to file
    :return: xarray.Dataset with a DataArray for each variable in the pfile
    """
    variables = ['ind', 'depth_from_surface', 'Vp', 'Vs', 'density', 'Qp', 'Qs']
    with open(path, 'r') as f:
        lines = f.readlines()
    ds = xr.Dataset()
    lat_no, lat_start, lat_end = lines[2].split()
    latitude = np.linspace(float(lat_start), float(lat_end), int(lat_no))
    lon_no, lon_start, lon_end = lines[3].split()
    longitude = np.linspace(float(lon_start), float(lon_end), int(lon_no))
    dep_no, dep_start, dep_end = lines[4].split()
    depth = np.array([0., 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])
    skippers = [i for i in np.arange(7, int(lat_no) * int(lon_no) * (int(dep_no)+1), int(dep_no) + 1)]
    header = list(np.arange(7))
    data = pd.read_csv(path, sep=' ', header=None, skiprows=header+skippers, dtype=np.float64).values
    arrays = np.hsplit(data, 7)
    for i, arr in enumerate(arrays):
        if i < 1:
            continue
        da = xr.DataArray(arr.reshape((int(lat_no), int(lon_no), int(dep_no))),
                          coords={'latitude': latitude, 'longitude': longitude, 'elevation': depth * -1000},
                          name=variables[i])
        ds[variables[i]] = da
    return ds


def read_ts(path: Union[str, Path]) -> Tuple[list, list]:
    """Function to read GoCAD .ts files (hacked for slb Petrel export 12-11-2025 cjh)

    Parameters
    __________

        path : Union[str, Path]
            Path to ts file, e.g. ``path='mesh.ts'``

    Returns
    _______

        vertices : list
            Pandas DataFrames containing the vertex data

        faces : list
            NumPy arrays containing the faces data

    .. versionadded:: 1.0.x

    """


    # Checking that the path is of type string or a path
    if not isinstance(path, (str, Path)):
        raise TypeError("Path must be of type string")

    # Getting the absolute path
    path = os.path.abspath(path=path)

    # Checking that the file has the correct file ending
    if not path.endswith(".ts"):
        raise TypeError("The mesh must be saved as .ts file")

    # Creating empty lists to store data
    vertices, faces = [], []

    # Creating empty lists to store data
    vertices_list, faces_list = [], []

    # Creating column names
    columns = ["id", "X", "Y", "Z"]

    # Opening file
    with open(path) as f:
        # Extracting data from every line
        for line in f:
            if not line.strip():
                continue
            line_type, *values = line.split()

            if line_type == "PROPERTIES":
                columns += values

                # Deleting duplicate column names
                columns = list(dict.fromkeys(columns))

            elif line_type == "END":
                # Creating array for faces
                faces = np.array(faces, dtype=np.int32)

                # Creating DataFrame for vertices
                vertices = pd.DataFrame(vertices, columns=columns).apply(pd.to_numeric)

                vertices_list.append(vertices)
                faces_list.append(faces)

                del vertices
                del faces

                vertices = []
                faces = []

            elif line_type == "VRTX" or line_type == "PVRTX":
                vertices.append(values)
            elif line_type == "TRGL":
                faces.append(values)


    return vertices_list, faces_list


def combine_ts_files_into_xarray(directory):
    datasets = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ts"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                vertices, faces = read_ts(file_path)
                
                # Create a DataArray for this file
                da = xr.DataArray(
                    np.column_stack([vertices[0]['X'], vertices[0]['Y'], vertices[0]['Z']]),
                    dims=['points', 'coordinate'],
                    coords={'points': range(len(vertices[0])), 
                            'coordinate': ['X', 'Y', 'Z']}
                )
                datasets.append(da)
    
    if datasets:
        # Combine into a single Dataset
        combined_da = xr.concat(datasets, dim='points')
        return combined_da
    else:
        return None


def create_3d_grid(x_spacing, y_spacing, z_spacing, extent):
    x = np.arange(extent['xmin'], extent['xmax'], x_spacing)
    y = np.arange(extent['ymin'], extent['ymax'], y_spacing)
    z = np.arange(extent['zmin'], extent['zmax'], z_spacing)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return xr.DataArray(X, dims=['z', 'y', 'x']), xr.DataArray(Y, dims=['z', 'y', 'x']), xr.DataArray(Z, dims=['z', 'y', 'x'])


def interpolate_surfaces_to_grid(da, X, Y, Z):
    points = da.sel(coordinate=['x', 'y']).values.T
    values = da.sel(coordinate='z').values
    
    # Interpolate the surfaces onto the grid
    interpolated_values = griddata(points, values, (X.ravel(), Y.ravel(), Z.ravel()), method='linear')
    interpolated_values = interpolated_values.reshape(X.shape)
    
    # Create an xarray Dataset
    ds = xr.Dataset()
    ds['interpolated_values'] = xr.DataArray(interpolated_values, dims=['z', 'y', 'x'], 
                                               coords={'x': X[0, 0, :], 'y': Y[:, 0, 0], 'z': Z[:, :, 0]})
    return ds


def ts_to_xarray_grid(directory_path, xyspacing=100, zspacing=50):
    da = combine_ts_files_into_xarray(directory_path)
    if da is not None:
        extent = {'xmin': 0, 'xmax': 100, 'ymin': 0, 'ymax': 100, 'zmin': 0, 'zmax': 1000}
        x_spacing = 10
        y_spacing = 10
        z_spacing = 50
        
        X, Y, Z = create_3d_grid(x_spacing, y_spacing, z_spacing, extent)
        ds = interpolate_surfaces_to_grid(da, X, Y, Z)
        return ds


def create_newberry_1d(topo_path):
    simple_x = np.linspace(-121.38, -121.38 + 0.12444560483672994, 200)
    simple_y = np.linspace(43.68, 43.77, 200)
    vp_1d = np.hstack(
        [np.zeros(int((foulger_elev[i] - foulger_elev[i+1]) / 0.05)) + foulger_vp[i]
        for i in range(len(foulger_vp))])
    vs_1d = np.hstack(
        [np.zeros(int((foulger_elev[i] - foulger_elev[i+1]) / 0.05)) + foulger_vs[i]
        for i in range(len(foulger_vp))])
    air_p = xr.DataArray(np.ones([200, 200, 50]) * foulger_vp[0], name='Vp',
                         coords={'latitude': simple_y,
                                 'longitude': simple_x,
                                 'elevation': np.arange(0, 2500., 50.)[::-1]})
    air_s = xr.DataArray(np.ones([200, 200, 50]) * foulger_vs[0], name='Vs',
                         coords={'latitude': simple_y,
                                 'longitude': simple_x,
                                 'elevation': np.arange(0, 2500., 50.)[::-1]})
    Vp = xr.DataArray(np.ones([200, 200, 100]) * vp_1d[np.newaxis, np.newaxis, :], name='Vp',
                      coords={'latitude': simple_y,
                              'longitude': simple_x,
                              'elevation': np.arange(-5000., 0., 50.)[::-1]})
    Vs = xr.DataArray(np.ones([200, 200, 100]) * vs_1d[np.newaxis, np.newaxis, :], name='Vs',
                      coords={'latitude': simple_y,
                              'longitude': simple_x,
                              'elevation': np.arange(-5000., 0., 50.)[::-1]})
    ds = xr.concat([xr.Dataset({'Vp': air_p, 'Vs': air_s}), xr.Dataset({'Vp': Vp, 'Vs': Vs})], dim='elevation')
    # Upsample the grid to make the shift calculation more precise
    new_elev = np.linspace(ds.elevation.min(), ds.elevation.max(), int(ds.elevation.max() - ds.elevation.min()))
    ds = ds.interp(elevation=new_elev)
    # Read in the topography
    topo = rioxarray.open_rasterio(topo_path)
    # topo = topo.rolling(y=25, x=25).mean()
    topo = topo.interp(x=ds.longitude, y=ds.latitude).drop(['band', 'x', 'y', 'spatial_ref']).squeeze('band')
    ds = xr.apply_ufunc(shift_model, topo, ds, input_core_dims=[[], ['elevation']],
                        output_core_dims=[['elevation']], vectorize=True)
    # Back to coarser model
    # ds = ds.interp(elevation=np.linspace(ds.elevation.min(), ds.elevation.max(), 50))
    ds['topography'] = topo
    return ds


def create_newberry_3d(ppmod_path, topo_path):
    """
    Create a 3D model for Newberry from the sw4 output of LLNL (Eric Matzel) and the DEM. Used the DEM to convert
    the model into absolute elevation instead of depth from surface.

    :param ppmod_path:
    :param topo_path:
    :return:
    """
    # New, empty arrays to stack on top of model (elements to be shifted into and eliminated)
    new_Vp = xr.DataArray(np.zeros([41, 57, 24]) + 0.5, name='Vp',
                          coords={'latitude': np.linspace(43.68, 43.78, 41),
                                  'longitude': np.linspace(-121.38, -121.24, 57),
                                  'elevation': np.arange(100., 2500., 100.)[::-1]})
    new_Vs = xr.DataArray(np.zeros([41, 57, 24]) + 0.3, name='Vs',
                          coords={'latitude': np.linspace(43.68, 43.78, 41),
                                  'longitude': np.linspace(-121.38, -121.24, 57),
                                  'elevation': np.arange(100., 2500., 100.)[::-1]})
    new_rho = xr.DataArray(np.zeros([41, 57, 24]), name='density',
                          coords={'latitude': np.linspace(43.68, 43.78, 41),
                                  'longitude': np.linspace(-121.38, -121.24, 57),
                                  'elevation': np.arange(100., 2500., 100.)[::-1]})
    new_Qp = xr.DataArray(np.zeros([41, 57, 24]), name='Qp',
                          coords={'latitude': np.linspace(43.68, 43.78, 41),
                                  'longitude': np.linspace(-121.38, -121.24, 57),
                                  'elevation': np.arange(100., 2500., 100.)[::-1]})
    new_Qs = xr.DataArray(np.zeros([41, 57, 24]), name='Qs',
                          coords={'latitude': np.linspace(43.68, 43.78, 41),
                                  'longitude': np.linspace(-121.38, -121.24, 57),
                                  'elevation': np.arange(100., 2500., 100.)[::-1]})
    new_ds = xr.Dataset({'Vp': new_Vp, 'Vs': new_Vs, 'density': new_rho, 'Qp': new_Qp, 'Qs': new_Qs})
    vmod = read_ppmod(ppmod_path)
    vmod = xr.concat([new_ds, vmod], dim='elevation')
    topo = rioxarray.open_rasterio(topo_path)
    # Project topography onto WGS84 lat/lon
    topo = topo.rio.reproject('EPSG:4326')
    # Upsample the grid to make the shift calculation more precise
    new_elev = np.linspace(vmod.elevation.min(), vmod.elevation.max(), int(vmod.elevation.max() - vmod.elevation.min()))
    vmod = vmod.interp(elevation=new_elev)
    # Interpolate onto vmod grid
    topo = topo.rolling(y=25, x=25).mean()
    topo = topo.interp(x=vmod.longitude, y=vmod.latitude).drop(['band', 'x', 'y', 'spatial_ref'])
    # The elevation node spacing must be 1 meter for this to work!
    vmod = xr.apply_ufunc(shift_model, topo, vmod, input_core_dims=[[], ['elevation']],
                          output_core_dims=[['elevation']], vectorize=True)
    vmod['topography'] = topo
    # Back to coarser model
    vmod = vmod.interp(elevation=np.linspace(vmod.elevation.min(), vmod.elevation.max(), 100))
    vmod['VpVs'] = vmod['Vp'] / vmod['Vs']
    return vmod.squeeze('band')


def write_cape_array(topo, tob):
    """
    Create a 3D grid for Cape Modern using topography, Top-of-basement (TOB) and curves fir to velocity logs
    in the basin

    :param topo: Path to topography file (netcdf 20 m grid)
    :param tob: Path to top of basement file (netcdf 20 m grid)
    :return:
    """

    topo = xr.load_dataarray(topo)
    tob = xr.load_dataarray(tob) + 50
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    tob.plot.surface(ax=ax)
    topo.plot.surface(ax=ax)
    plt.show()
    # thickness
    thick = topo - tob
    cape_model_p = xr.DataArray(np.zeros([601, 601, 201]) + 5.8, name='Vp',
                                coords={'easting': np.linspace(315300., 345300., 601),
                                        'northing': np.linspace(4246200., 4276200., 601),
                                        'elevation': np.linspace(-7000., 3000., 201)})
    cape_model_s = xr.DataArray(np.zeros([601, 601, 201]) + 3.392, name='Vs',
                                coords={'easting': np.linspace(315300., 345300., 601),
                                        'northing': np.linspace(4246200., 4276200., 601),
                                        'elevation': np.linspace(-7000., 3000., 201)})
    # Create a scaled depth dataarray to feed to the basin functions
    depth = topo - cape_model_p.elevation
    scaled_depth = 1.828 * (depth / thick)  # Gives depth as if the basin is always 6000 ft deep
    basin_da_gradient_p = xr.apply_ufunc(basin_gradient_p, scaled_depth, input_core_dims=[['elevation']],
                                         output_core_dims=[['elevation']], vectorize=True)
    basin_da_gradient_s = xr.apply_ufunc(basin_gradient_s, scaled_depth, input_core_dims=[['elevation']],
                                         output_core_dims=[['elevation']], vectorize=True)
    # Clip above tob
    basin_da_gradient_p = basin_da_gradient_p.where(cape_model_p.elevation >= tob)
    basin_da_gradient_s = basin_da_gradient_s.where(cape_model_s.elevation >= tob)
    # Combine with basement velocities
    basement_p = cape_model_p.where(cape_model_p.elevation < tob)
    cape_model_p = basement_p.combine_first(basin_da_gradient_p)
    air_p = cape_model_p.where(cape_model_p.elevation <= topo).min(dim='elevation', skipna=True)
    basement_s = cape_model_s.where(cape_model_s.elevation < tob)
    cape_model_s = basement_s.combine_first(basin_da_gradient_s)
    air_s = cape_model_s.where(cape_model_s.elevation <= topo).min(dim='elevation', skipna=True)
    cape_model_p = xr.where(((cape_model_p.elevation > topo) & (air_p == 5.8)), air_p, cape_model_p)
    cape_model_s = xr.where(((cape_model_s.elevation > topo) & (air_s == 3.392)), air_s, cape_model_s)
    cape_model_p.sel(northing=4252500, method='nearest').plot.imshow()
    plt.show()
    cape_model_p.sel(easting=330000, method='nearest').plot.imshow()
    plt.show()
    ## This is working EOD 2-7 Need to combine these DataArrays into a Dataset, write to disk, then generate NLLoc grids
    ## to see if this worked or not
    ds = xr.Dataset({'Vp': cape_model_p, 'Vs': cape_model_s})
    # Get back to north, east, elev dimension order
    ds = ds.transpose('northing', 'easting', 'elevation')
    return ds


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
        f.write('{:4.1f} {:3d} {:3d} {:3d}\n'.format(
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


def write_newberry1d_NLLoc_grid(dataset):
    """
    Write NLLoc grids from xarray Dataset of LLNL 3D model
    :return:
    """
    # Assume that sampling is uniform
    # Start a new grid
    grd_P = NLLGrid()
    grd_S = NLLGrid()
    # NLLoc models require units of slowness*length and cubic nodes
    origin = [dataset.longitude[0].values, dataset.latitude[0].values]
    # Change from lat/lon to meters (SIMPLE TRANS from NLLoc)
    x = (dataset.longitude - dataset.longitude[0]) * 111.111 * np.cos(np.deg2rad(dataset.latitude[100]))
    y = (dataset.latitude - dataset.latitude[0]) * 111.111
    dataset = dataset.rename({'latitude': 'y', 'longitude': 'x', 'elevation': 'depth'})
    dataset = dataset.assign_coords(x=x.values*1000, y=y.values*1000, depth=dataset.depth.values*-1)
    # Interpolate onto 200 m grid
    dataset = dataset.interp(x=np.arange(dataset.x.min(), dataset.x.max(), step=50.),
                             y=np.arange(dataset.y.min(), dataset.y.max(), step=50.),
                             depth=np.arange(dataset.depth.min(), dataset.depth.max(), step=50.)[::-1])
    # NLLGrid indexing is column-first
    dataset = dataset.transpose('x', 'y', 'depth')
    print(dataset)
    Pslow_len = 0.05 / dataset.Vp.values
    Sslow_len = 0.05 / dataset.Vs.values
    # Populate grid headers
    for grd in [grd_P, grd_S]:
        grd.dx = 0.05
        grd.dy = 0.05
        grd.dz = 0.05
        grd.x_orig = 0.0
        grd.y_orig = 0.0
        grd.z_orig = dataset.depth.min().values / 1000.
        grd.type = 'SLOW_LEN'
        grd.orig_lat = origin[1]
        grd.orig_lon = origin[0]
        grd.proj_name = 'SIMPLE'
    # After transpose, still need to flip Z dim for each array
    grd_P.array = np.flip(Pslow_len, axis=2)
    grd_P.basename = 'Newberry_1d-topo.P.mod'
    grd_S.array = np.flip(Sslow_len, axis=2)
    grd_S.basename = 'Newberry_1d-topo.S.mod'
    # Write to file
    grd_P.write_buf_file()
    grd_P.write_hdr_file()
    grd_S.write_buf_file()
    grd_S.write_hdr_file()
    return grd_P, grd_S, Pslow_len, Sslow_len


def write_newberry3d_NLLoc_grid(dataset, outdir):
    """
    Write NLLoc grids from xarray Dataset of LLNL 3D model
    :return:
    """
    # Assume that sampling is uniform
    # Start a new grid
    grd_P = NLLGrid()
    grd_S = NLLGrid()
    # NLLoc models require units of slowness*length and cubic nodes
    origin = [dataset.longitude[0].values, dataset.latitude[0].values]
    # Change from lat/lon to meters
    x = np.arange(len(dataset.longitude)) * 161.
    y = np.arange(len(dataset.latitude)) * 222.128
    dataset = dataset.rename({'latitude': 'y', 'longitude': 'x', 'elevation': 'depth'})
    dataset = dataset.assign_coords(x=x, y=y, depth=dataset.depth.values*-1)
    # Interpolate onto 200 m grid
    dataset = dataset.interp(x=np.arange(dataset.x.min(), dataset.x.max(), step=200.),
                             y=np.arange(dataset.y.min(), dataset.y.max(), step=200.),
                             depth=np.arange(dataset.depth.min(), dataset.depth.max(), step=200.)[::-1])
    # NLLGrid indexing is column-first
    dataset = dataset.transpose('x', 'y', 'depth')
    print(dataset)
    Pslow_len = 0.2 / dataset.Vp.values
    Sslow_len = 0.2 / dataset.Vs.values
    # Populate grid headers
    for grd in [grd_P, grd_S]:
        grd.dx = 0.2
        grd.dy = 0.2
        grd.dz = 0.2
        grd.x_orig = 0.0
        grd.y_orig = 0.0
        grd.z_orig = dataset.depth.min().values / 1000.
        grd.type = 'SLOW_LEN'
        grd.orig_lat = origin[1]
        grd.orig_lon = origin[0]
        grd.proj_name = 'SIMPLE'
    # After transpose, still need to flip Z dim for each array
    grd_P.array = np.flip(Pslow_len, axis=2)
    grd_P.basename = 'Newberry_LLNL-3d.P.mod'
    grd_S.array = np.flip(Sslow_len, axis=2)
    grd_S.basename = 'Newberry_LLNL-3d.S.mod'
    # Write to file
    grd_P.write_buf_file()
    grd_P.write_hdr_file()
    grd_S.write_buf_file()
    grd_S.write_hdr_file()
    return grd_P, grd_S, Pslow_len, Sslow_len