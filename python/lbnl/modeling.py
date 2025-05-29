#!/usr/bin/python


"""
Utilities for modeling wavefields with sw4 and maybe some other codes later
"""

import struct
import datetime
import dask

import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter
from pathlib import Path
from math import cos, radians, degrees

try:
    import pySW4 as sw4
except ImportError:
    print("pySW4 not found, SW4 functionality disabled")


def frac_to_ifile(
    frac_path,
    origin=(1200, -910, 360),
    vertical_thickness=0.2,
    grid_spacing=1.0,
    smoothing=True,
    smooth_sigma=1.0,
    plot=False,
    extrapolate=False,
    extrap_method='rbf'
):
    """
    Interpolate fracture aperture data from an irregular CSV file and 
    generate an SW4 material interface file.

    The CSV file is expected to contain columns for X, Y, and Z coordinates.
    The function rebases the coordinates to a local system, performs interpolation
    to a regular grid, optionally extrapolates to fill missing data, and optionally
    smooths the grid. Finally, it writes an SW4 interface file containing top and 
    bottom elevations of the interface layer.

    Parameters
    ----------
    frac_path : str or Path
        Path to the CSV file with fracture data (columns: X, Y, Z).
    origin : tuple of float, optional
        The reference origin (x0, y0, z0) in meters. The coordinates in the file
        will be rebased with respect to this origin. Default is (1200, -910, 360).
    vertical_thickness : float, optional
        The vertical thickness (in meters) of the interface layer to add to the top.
        Default is 0.2.
    grid_spacing : float, optional
        The spacing (in meters) for the regular grid into which data is interpolated.
        Default is 1.0.
    smoothing : bool, optional
        If True, apply a Gaussian filter to smooth the interpolated grid. Default is True.
    smooth_sigma : float, optional
        The sigma (in meters) for the Gaussian smoothing. Default is 1.0.
    plot : bool, optional
        If True, display a 3D surface plot of the interpolated (and extrapolated) grid.
        Default is False.
    extrapolate : bool, optional
        If True, fill missing values (NaNs) in the interpolated grid using the specified 
        extrapolation method. Default is False.
    extrap_method : {'nearest', 'rbf'}, optional
        The extrapolation method to use if extrapolate is True.
        'nearest' uses nearest-neighbor interpolation and 'rbf' fits a global surface.
        Default is 'rbf'.

    Returns
    -------
    None
        The function writes an interface file (<stem>_ifile1.txt) to the same directory as frac_path.
    """
    # -- load & rebase ---------------------------------------------------
    arr = np.loadtxt(frac_path, skiprows=1, delimiter=',')
    arr *= 0.3048   # ft → m
    arr[:, 0] -= origin[0]
    arr[:, 1] -= origin[1]
    arr[:, 2] = origin[2] - arr[:, 2]

    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]

    # -- define regular grid --------------------------------------------
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_regular = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_regular = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    Xi, Yi = np.meshgrid(x_regular, y_regular, indexing='xy')

    # -- interpolation --------------------------------------------------
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

    # -- extrapolation fill ---------------------------------------------
    if extrapolate:
        if extrap_method == 'nearest':
            Zi_nn = griddata((x, y), z, (Xi, Yi), method='nearest')
            Zi = np.where(np.isnan(Zi), Zi_nn, Zi)
        elif extrap_method == 'rbf':
            # RBF fits a global surface and will extrapolate outside the convex hull.
            rbf = Rbf(x, y, z, function='thin_plate')
            Zi_rbf = rbf(Xi, Yi)
            Zi = np.where(np.isnan(Zi), Zi_rbf, Zi)
        else:
            raise ValueError("extrap_method must be 'nearest' or 'rbf'")
    
    if smoothing:
        Zi = gaussian_filter(Zi, sigma=smooth_sigma)
    
    # -- optional plot --------------------------------------------------
    if plot:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Interpolated (and extrapolated) Fracture Surface")
        plt.tight_layout()
        plt.show()

    # -- write interface file -------------------------------------------
    stem = Path(frac_path).stem
    ifile1 = Path(frac_path).with_name(f"{stem}_ifile1.txt")
    Nx, Ny = len(x_regular), len(y_regular)

    with open(ifile1, "w") as f:
        # Write header: numbers of grid points and number of layers (2)
        f.write(f"{Nx} {Ny} 2\n")
        for j in range(Ny):
            for i in range(Nx):
                top_z = Zi[j, i]
                bot_z = top_z + vertical_thickness
                f.write(f"{x_regular[i]:.3f} {y_regular[j]:.3f} {top_z:.3f} {bot_z:.3f}\n")

    print(f"Wrote interface file: {ifile1}")


def write_sw4_input(path, inventory, outdir, sources=['DMUS3'], origin=(1200, -910, 360),
                    tmax=1.0, write_model=False, ifiles=None, material=None, pfile=None):
    """
    Create an SW4 input file for simulation, defining sources, receivers, 
    and material interfaces.

    This function writes a configuration file that tells SW4 where to find 
    material interface files or a pfile. It uses the provided inventory to 
    determine the positions of sources and receivers. For each station-like 
    entity in the inventory, the function writes either a source (if the station 
    code matches one of the provided source codes) or a receiver.

    Parameters
    ----------
    path : str or Path
        Output file path for the SW4 input file.
    inventory : obspy Inventory
        An Obspy Inventory object containing station metadata.
    outdir : str
        Output directory path for SW4 file I/O.
    sources : list of str, optional
        List of station codes to be considered as sources. Default is ['DMUS3'].
    origin : tuple of float, optional
        The model origin (x, y, z) in meters. Default is (1200, -910, 360).
    tmax : float, optional
        Maximum simulation time in seconds. Default is 1.0.
    write_model : bool, optional
        If True, include commands in the input file to output model fields. Default is False.
    ifiles : list of str or Path, optional
        List of interface file paths to include in the input file.
    material : dict, optional
        Dictionary containing material properties (keys: Vp, Vs, rho).
        Must be provided along with ifiles if pfile is not provided.
    pfile : str or Path, optional
        Path to a precomputed pfile. If provided, ifiles and material are ignored.

    Returns
    -------
    None
        The function writes the SW4 input file to the specified path.
    """
    # Check that either (ifiles and material) or pfile is provided.
    if not ((ifiles and material) or pfile):
        raise ValueError("Either 'ifiles' and 'material' must be provided, or 'pfile' must be provided.")

    with open(path, "w") as f:
        # Write file I/O settings and simulation time.
        f.write("# SW4 input file\n")
        f.write("# Generated by write_sw4_input\n")
        f.write("\n")
        f.write(f"fileio path={outdir} verbose=2 printcycle=100 pfs=0\n")
        f.write(f"time t={tmax}\n")
        f.write("grid x=80 y=50 z=60 h=.1\n")
        # Write source and receiver definitions.
        for sta in inventory[0]:  # Assumes only one network in the inventory.
            if sta.code[-2] == 'S':  # If station is a source.
                coords = sta.extra
                x = (coords['hmc_east'].value * 0.3048) - origin[0]
                y = (coords['hmc_north'].value * 0.3048) - origin[1]
                z = origin[2] - (coords['hmc_elev'].value)
                if sta.code in sources:
                    f.write(f"source x={x} y={y} z={z} freq=400. type=RickerInt t0=0.0165\n")
                else:
                    f.write(f"#source x={x} y={y} z={z} freq=400. type=RickerInt t0=0.0165\n")
            else:  # Receiver station.
                coords = sta.extra
                x = (coords['hmc_east'].value * 0.3048) - origin[0]
                y = (coords['hmc_north'].value * 0.3048) - origin[1]
                z = origin[2] - (coords['hmc_elev'].value)
                f.write(f"rec x={x} y={y} z={z} sta={sta.code} variables=velocity\n")
        # Write material or pfile specifications.
        if ifiles and material:
            for i, ifile in enumerate(ifiles):
                f.write(f"ifile filename={ifile} input=cartesian\n")
                f.write(f"material id={i} vp={material['Vp']} vs={material['Vs']} rho={material['rho']}\n")
                f.write(f"material id={i+1} vp=1500 vs=10 rho=1000\n")
            f.write(f"block vp={material['Vp']} vs={material['Vs']} rho={material['rho']}\n")
        elif pfile:
            f.write(f"pfile filename={pfile} style=cartesian\n")
        # Set boundary condition.
        f.write("boundary_conditions lz=2\n")
        if write_model:
            f.write("volimage file=model mode=rho cycle=0\n")
            f.write("volimage file=model mode=p cycle=0\n")
            f.write("volimage file=model mode=s cycle=0\n")
    return


def write_pfile_from_fracture_fullgrid(
    ifile_path,
    origin=(1200.0, -910.0, 360.0),
    rebased=True,
    model_depth=60.0,
    dxy=0.5,
    dz_coarse=0.5,
    dz_fine=0.01,
    grid_bounds=None,
    contiguous: bool = False,
    contiguous_method: str = "linear",
    mat_above=None,
    mat_inside=None,
    mat_below=None,
    coord_system="cartesian",
    lon0_deg=0.0,
    lat0_deg=0.0,
    earth_radius=6_371_000.0,
    model_name="LocalModel",
    discontinuity_idx=(-99, -99, -99, -99),
    q_available=False,
    qp_const=1000.0,
    qs_const=600.0,
    outfile=None,
    snap_tol=1e-6
):
    """
    Build and write an SW4 p-file from fracture data distributed on a full XY grid.

    This function reads an SW4 interface file (ifile) containing top and bottom 
    elevations for each grid cell, and then constructs a detailed p-file that 
    includes a global vertical grid with both coarse and fine spacing. Optional 
    parameters allow for coordinate rebasing and interpolation to fill contiguous 
    blocks.

    Parameters
    ----------
    ifile_path : str or Path
        Path to the SW4 interface file containing columns for x, y, z_top, and z_bottom.
    origin : tuple of float, optional
        Global origin (x0, y0, z0) in meters (HMC; elevation in meters above sea level).
        Default is (1200.0, -910.0, 360.0).
    rebased : bool, optional
        If True, assume the CSV is already in local XY coordinates and z values are positive-down.
        Default is True.
    model_depth : float, optional
        Maximum model depth in meters. Default is 60.0.
    dxy : float, optional
        Horizontal grid spacing in meters. Default is 0.5.
    dz_coarse : float, optional
        Coarse vertical spacing in meters. Default is 0.5.
    dz_fine : float, optional
        Fine vertical spacing in meters. Default is 0.01.
    grid_bounds : tuple of tuple, optional
        Optional override for grid bounds given as ((xmin, xmax), (ymin, ymax)).
    contiguous : bool, optional
        If True, use interpolation to fill gaps in the grid. Default is False.
    contiguous_method : str, optional
        Method for interpolation when contiguous is True ('linear' or 'cubic'). Default is 'linear'.
    mat_above : dict, optional
        Material properties for the layer above the fracture (keys: Vp, Vs, rho). Defaults provided if None.
    mat_inside : dict, optional
        Material properties for the fracture zone. Defaults provided if None.
    mat_below : dict, optional
        Material properties for the layer below the fracture. Defaults provided if None.
    coord_system : str, optional
        Coordinate system for output ('cartesian' or 'geographic'). Default is 'cartesian'.
    lon0_deg : float, optional
        Reference longitude in degrees (used when coord_system is 'geographic'). Default is 0.0.
    lat0_deg : float, optional
        Reference latitude in degrees (used when coord_system is 'geographic'). Default is 0.0.
    earth_radius : float, optional
        Earth radius in meters. Default is 6_371_000.0.
    model_name : str, optional
        A label for the model, written in the header of the p-file.
    discontinuity_idx : tuple of int, optional
        Indices for geological discontinuities (Ised, IMoho, I410, I660). Default is (-99, -99, -99, -99).
    q_available : bool, optional
        Flag indicating if quality factors (Q) are available. Default is False.
    qp_const : float, optional
        Default value for P-wave quality factor. Default is 1000.0.
    qs_const : float, optional
        Default value for S-wave quality factor. Default is 600.0.
    outfile : str or Path, optional
        Output file path for the generated p-file. If None, the output filename is derived from ifile_path.
    snap_tol : float, optional
        Tolerance used when snapping grid values. Default is 1e-6.

    Returns
    -------
    None
        The function writes the generated p-file to disk.
    """
    # ---- defaults for materials
    if mat_above   is None: mat_above   = dict(Vp=6900., Vs=3730., rho=2950.)
    if mat_inside  is None: mat_inside  = dict(Vp=1000.,  Vs=100., rho=1000.)
    if mat_below   is None: mat_below   = dict(Vp=6900., Vs=3730., rho=2950.)

    # determine outfile
    if outfile is None:
        outfile = Path(ifile_path).with_suffix("_pfile.txt")

    # ---- read the 4-column ifile: x y z_top z_bottom
    df = pd.read_csv(
        ifile_path,
        delim_whitespace=True,
        names=["x", "y", "z1", "z2"],
        comment="#",
        dtype=float,
    )

    # ---- optional rebasing into local coords + positive-down depth
    if not rebased:
        x0, y0, z0 = origin
        df["x"] -= x0
        df["y"] -= y0
        # elevation->depth below origin
        df["z1"] = z0 - df["z1"]
        df["z2"] = z0 - df["z2"]
    # else: assume x,y already in [0–80]