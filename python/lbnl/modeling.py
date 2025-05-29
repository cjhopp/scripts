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
import numpy as np
import struct
import datetime
import dask
import pandas as pd

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
    # else: assume x,y already in [0–80],[0–50] and z1,z2 in [0–model_depth]


    # ---- horizontal grid bounds
    if grid_bounds is None:
        xmin, xmax = 0.0, 80.0
        ymin, ymax = 0.0, 50.0
    else:
        (xmin, xmax), (ymin, ymax) = grid_bounds


    x_vals = np.arange(xmin, xmax + snap_tol, dxy)
    y_vals = np.arange(ymin, ymax + snap_tol, dxy)
    nx, ny = len(x_vals), len(y_vals)


    # prepare bin-grid storage
    z1_grid = np.full((nx, ny), np.nan)
    z2_grid = np.full((nx, ny), np.nan)


    # ---- bin each fracture point into the nearest cell
    mask_ok = (df.z2 > 0) & (df.z1 < model_depth) & (df.z2 > df.z1)
    df = df[mask_ok]
    raw_x = df.x.values
    raw_y = df.y.values
    ix = np.rint((raw_x - xmin) / dxy).astype(int)
    iy = np.rint((raw_y - ymin) / dxy).astype(int)
    ix = np.clip(ix, 0, nx-1)
    iy = np.clip(iy, 0, ny-1)
    z1_grid[ix, iy] = df.z1.values
    z2_grid[ix, iy] = df.z2.values


    # ---- optional contiguous fill
    if contiguous:
        Xi, Yi = np.meshgrid(x_vals, y_vals, indexing="ij")
        pts = np.column_stack((raw_x, raw_y))
        z1_int = griddata(pts, df.z1.values, (Xi, Yi), method=contiguous_method, fill_value=np.nan)
        z2_int = griddata(pts, df.z2.values, (Xi, Yi), method=contiguous_method, fill_value=np.nan)
        mask1 = np.isnan(z1_int)
        mask2 = np.isnan(z2_int)
        z1_int[mask1] = z1_grid[mask1]
        z2_int[mask2] = z2_grid[mask2]
        z1_grid, z2_grid = z1_int, z2_int


    # ---- ensure we have at least one fracture point in the grid
    m1 = ~np.isnan(z1_grid)
    m2 = ~np.isnan(z2_grid)
    if not (m1.any() and m2.any()):
        raise RuntimeError("No fracture points landed inside your XY grid!")


    # ---- clamp all depths to [0, model_depth]
    z1c = np.clip(z1_grid[m1], 0.0, model_depth)
    z2c = np.clip(z2_grid[m2], 0.0, model_depth)
    top_depth = z1c.min()      # shallowest top ≥ 0
    bot_depth = z2c.max()      # deepest bottom ≤ model_depth


    # ---- build the ONE global vertical grid
    z_above = np.arange(0.0,       top_depth, dz_coarse)
    z_fine  = np.arange(top_depth, bot_depth,  dz_fine)
    z_below = np.arange(bot_depth, model_depth + snap_tol, dz_coarse)
    z_all   = np.unique(np.concatenate([z_above, z_fine, z_below]))
    nz      = len(z_all)


    # ---- header coords Xgrid, Ygrid
    if coord_system.lower() == 'cartesian':
        Xgrid, Ygrid = x_vals, y_vals
    else:
        lat0 = np.radians(lat0_deg)
        dlat = (y_vals - y_vals[0]) / earth_radius
        dlon = (x_vals - x_vals[0]) / (earth_radius * np.cos(lat0))
        Ygrid = lat0_deg + np.degrees(dlat)
        Xgrid = lon0_deg + np.degrees(dlon)


    # ---- write the SW4 p-file
    NY, NX, Ndep = ny, nx, nz
    Ymin, Ymax = Ygrid.min(), Ygrid.max()
    Xmin, Xmax = Xgrid.min(), Xgrid.max()
    Ised, IMoho, I410, I660 = discontinuity_idx
    qflag = 'T' if q_available else 'F'


    with open(outfile, "w") as f:
        # header
        f.write(f"{model_name}\n")
        f.write(f"{dxy:.4f}\n")
        f.write(f"{NX:d} {Xmin:.4f} {Xmax:.4f}\n")
        f.write(f"{NY:d} {Ymin:.4f} {Ymax:.4f}\n")
        f.write(f"{Ndep:d} {z_all.min():.3f} {z_all.max():.3f}\n")
        f.write(f"{Ised:d} {IMoho:d} {I410:d} {I660:d}\n")
        f.write(qflag + "\n")


        buf = []
        BATCH = 250
        hdr_fmt = "{:.3f} {:.3f} {}\n".format
        row_fmt = "{:7d} {:7.3f} {:7.1f} {:7.1f} {:7.1f}\n".format


        for iy, Y in enumerate(Ygrid):
            for ix, X in enumerate(Xgrid):
                ztop = z1_grid[ix, iy]
                if np.isnan(ztop):
                    k1 = k2 = 0
                else:
                    zbot = z2_grid[ix, iy]


                    # require a real overlap
                    if ztop >= model_depth or zbot <= 0 or zbot <= ztop:
                        # no intersection
                        k1 = k2 = 0
                    else:
                        # compute only once, from the clipped-but-tested values
                        z1c = max(0.0, ztop)
                        z2c = min(model_depth, zbot)
                        k1 = np.searchsorted(z_all, z1c, side="left")
                        k2 = np.searchsorted(z_all, z2c, side="left")


                buf.append(hdr_fmt(X, Y, nz))
                for k, zm in enumerate(z_all):
                    if k < k1:
                        m = mat_above
                    elif k < k2:
                        m = mat_inside
                    else:
                        m = mat_below
                    buf.append(row_fmt(k, zm, m["Vp"], m["Vs"], m["rho"]))


                if len(buf) > BATCH*(nz+1):
                    f.write("".join(buf))
                    buf.clear()


        if buf:
            f.write("".join(buf))


    print(f"Wrote SW4 pfile to: {outfile}")


def _parse_header(lines):
    """
    Parse the header lines of an SW4 p-file.

    This helper function reads the first 7 lines of a p-file and extracts
    metadata such as model name, grid dimensions, depth limits, and discontinuity indices.

    Parameters
    ----------
    lines : list of str
        A list of 7 header lines read from the p-file.

    Returns
    -------
    dict
        A dictionary containing the parsed header information.
    """
    model_name = lines[0].strip()
    delta_deg = float(lines[1])
    NX, Xmin, Xmax = map(float, lines[2].split())
    NY, Ymin, Ymax = map(float, lines[3].split())
    Ndep, dmin_m, dmax_m = map(float, lines[4].split())
    Ised, IMoho, I410, I660 = map(int, lines[5].split())
    qflag = lines[6].strip()
    NY = int(NY); NX = int(NX); Ndep = int(Ndep)
    header_info = dict(
        model_name=model_name,
        delta_deg=delta_deg,
        NY=NY,
        NX=NX,
        Ndep=Ndep,
        Ymin=Ymin,
        Ymax=Ymax,
        Xmin=Xmin,
        Xmax=Xmax,
        dmin_m=dmin_m,
        dmax_m=dmax_m,
        Ised=Ised,
        IMoho=IMoho,
        I410=I410,
        I660=I660,
        q_available=(qflag.upper() == 'T')
    )
    return header_info


def _scan_file_for_offsets(path):
    """
    Scan an SW4 p-file to determine the file offsets for each latitude-row block.

    This function reads the header and then records the byte offset for each 
    latitude row in the p-file. These offsets allow for efficient delayed reading 
    of subsets of the file.

    Parameters
    ----------
    path : str or Path
        Path to the SW4 p-file.

    Returns
    -------
    tuple
        Contains the header dictionary, a list of byte offsets for each latitude row,
        and the reconstructed latitude, longitude, and depth arrays.
    """
    path = Path(path)
    offsets = []
    with open(path, 'rb') as f:
        # Read global header.
        lines = [f.readline().decode('ascii') for _ in range(7)]
        hdr = _parse_header(lines)
        NY, NX, Nz = hdr['NY'], hdr['NX'], hdr['Ndep']
        hdr_q = hdr['q_available']

        # Reconstruct lat/lon arrays.
        Ymin, Ymax = hdr['Ymin'], hdr['Ymax']
        Xmin, Xmax = hdr['Xmin'], hdr['Xmax']
        Y_grid = np.linspace(Ymin, Ymax, NY)
        X_grid = np.linspace(Xmin, Xmax, NX)

        for j in range(NY):
            offsets.append(f.tell())
            for _ in range(NX):
                _ = f.readline()  # Skip profile header.
                for _ in range(Nz):
                    _ = f.readline()  # Skip profile data.
    with open(path, 'rb') as f:
        f.seek(offsets[0])
        _ = f.readline().decode('ascii')
        raw = np.asarray([f.readline().decode('ascii').split() for _ in range(Nz)])
        data0 = np.asarray(raw, dtype=float)
        depth_m = data0[:, 1]
    return hdr, offsets, Y_grid, X_grid, depth_m


@dask.delayed
def _read_latitude_row(path, byte_offset, Nz, NX, q_available):
    """
    Delayed function to read one latitude-row block from a SW4 p-file.

    Reads one row of profiles (each of depth Nz) starting from the given byte 
    offset. This function returns the arrays for Vp, Vs, density, and optionally 
    quality factors Qp and Qs.

    Parameters
    ----------
    path : str or Path
        Path to the SW4 p-file.
    byte_offset : int
        The byte offset in the file where the latitude row begins.
    Nz : int
        Number of depth levels in each profile.
    NX : int
        Number of profiles (longitude points) in this latitude row.
    q_available : bool
        Flag indicating whether quality factor data (Qp, Qs) is available.

    Returns
    -------
    tuple of numpy.ndarray
        Returns (vp, vs, rh) if q_available is False, or (vp, vs, rh, qp, qs) if True.
        Each array has shape (NX, Nz) and dtype float32.
    """
    path = Path(path)
    with open(path, 'rb') as f:
        f.seek(byte_offset)
        rows_vp, rows_vs, rows_rh = [], [], []
        rows_qp, rows_qs = [], []
        for _ in range(NX):
            f.readline().decode('ascii')  # Skip profile header.
            raw = np.asarray([f.readline().decode('ascii').split() for __ in range(Nz)])
            data = np.asarray(raw, dtype=float)
            rows_vp.append(data[:, 2])
            rows_vs.append(data[:, 3])
            rows_rh.append(data[:, 4])
            if q_available:
                rows_qp.append(data[:, 5])
                rows_qs.append(data[:, 6])

    vp = np.stack(rows_vp, axis=0).astype(np.float32)
    vs = np.stack(rows_vs, axis=0).astype(np.float32)
    rh = np.stack(rows_rh, axis=0).astype(np.float32)
    if q_available:
        qp = np.stack(rows_qp, axis=0).astype(np.float32)
        qs = np.stack(rows_qs, axis=0).astype(np.float32)
        return vp, vs, rh, qp, qs
    else:
        return vp, vs, rh


def read_sw4_pfile_dask(path, return_q=False, chunk_rows=1):
    """
    Read an SW4 p-file into an xarray Dataset using Dask for out-of-core processing.

    This function uses delayed functions to read the p-file in chunks along the 
    latitude axis. It concatenates the delayed arrays for material properties into 
    Dask arrays and builds an xarray Dataset with appropriate coordinates.

    Parameters
    ----------
    path : str or Path
        Path to the SW4 p-file.
    return_q : bool, optional
        If True and if quality factor data is available, include Qp and Qs in the Dataset.
        Default is False.
    chunk_rows : int, optional
        Number of latitude rows to read per Dask chunk. Default is 1.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing material properties (Vp, Vs, density, and optionally Qp and Qs)
        with dimensions ('latitude', 'longitude', 'depth') and relevant coordinate information.
    """
    path = Path(path)
    hdr, offsets, Y_grid, X_grid, depth_m = _scan_file_for_offsets(path)
    Nz = hdr['Ndep']
    Ny = hdr['NY']
    Nx = hdr['NX']
    q_avail = hdr['q_available'] and return_q

    # Build delayed tasks chunked by latitude rows.
    row_tasks = []
    for start_j in range(0, Ny, chunk_rows):
        end_j = min(start_j + chunk_rows, Ny)
        offs = offsets[start_j:end_j]
        row_tasks.append(_read_latitude_row(path, offs[0], Nz, Nx, q_avail))

    vp_chunks = []
    vs_chunks = []
    rh_chunks = []
    if q_avail:
        qp_chunks = []
        qs_chunks = []

    for task in row_tasks:
        vp_t = task[0]
        vs_t = task[1]
        rh_t = task[2]

        vp_da = da.from_delayed(vp_t, shape=(Nx, Nz), dtype=np.float32)
        vs_da = da.from_delayed(vs_t, shape=(Nx, Nz), dtype=np.float32)
        rh_da = da.from_delayed(rh_t, shape=(Nx, Nz), dtype=np.float32)

        vp_chunks.append(vp_da[np.newaxis, ...])
        vs_chunks.append(vs_da[np.newaxis, ...])
        rh_chunks.append(rh_da[np.newaxis, ...])

        if q_avail:
            qp_da = da.from_delayed(task[3], shape=(Nx, Nz), dtype=np.float32)
            qs_da = da.from_delayed(task[4], shape=(Nx, Nz), dtype=np.float32)
            qp_chunks.append(qp_da[np.newaxis, ...])
            qs_chunks.append(qs_da[np.newaxis, ...])

    Vp = da.concatenate(vp_chunks, axis=0)
    Vs = da.concatenate(vs_chunks, axis=0)
    Rh = da.concatenate(rh_chunks, axis=0)
    if q_avail:
        Qp = da.concatenate(qp_chunks, axis=0)
        Qs = da.concatenate(qs_chunks, axis=0)

    coords = dict(
        latitude=('latitude', Y_grid),
        longitude=('longitude', X_grid),
        depth=('depth', depth_m),
    )
    data_vars = dict(
        Vp=(('latitude', 'longitude', 'depth'), Vp),
        Vs=(('latitude', 'longitude', 'depth'), Vs),
        density=(('latitude', 'longitude', 'depth'), Rh),
    )
    if q_avail:
        data_vars['Qp'] = (('latitude', 'longitude', 'depth'), Qp)
        data_vars['Qs'] = (('latitude', 'longitude', 'depth'), Qs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=hdr)
    return ds


def read_sw4img_volume(filename, verbose=False, machineformat='native'):
    """
    Read a volumetric .sw4img file and return its contents as an xarray Dataset.

    This function reads the header and volumetric data from a .sw4img file,
    verifies that the file contains a full 3D volume (plane = -1), and constructs
    an xarray Dataset with the volume and its corresponding spatial coordinates.

    Parameters
    ----------
    filename : str or Path
        Path to the .sw4img file.
    verbose : bool, optional
        If True, output header details to the console. Default is False.
    machineformat : {'native', 'little', 'big'}, optional
        Byte order specifier. Default is 'native'.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the 3D volume under the key 'volume' along with
        coordinate arrays 'x', 'y', and 'z'. Additional header attributes are stored in attrs.

    Raises
    ------
    ValueError
        If the file does not represent a full 3D volume (i.e. if plane != -1).
    """
    byte_order = {'native': '=', 'little': '<', 'big': '>'}.get(machineformat, '=')

    with open(filename, 'rb') as f:
        prec = struct.unpack(byte_order + 'i', f.read(4))[0]
        npatches = struct.unpack(byte_order + 'i', f.read(4))[0]
        t = struct.unpack(byte_order + 'd', f.read(8))[0]
        plane = struct.unpack(byte_order + 'i', f.read(4))[0]
        coord = struct.unpack(byte_order + 'd', f.read(8))[0]
        mode = struct.unpack(byte_order + 'i', f.read(4))[0]
        gridinfo = struct.unpack(byte_order + 'i', f.read(4))[0]
        timecreated = f.read(25).decode('ascii').strip()

        if plane != -1:
            raise ValueError("This function only supports volumetric files with plane = -1")

        if verbose:
            print(f"Precision: {prec}")
            print(f"Patches: {npatches}")
            print(f"Time: {t}")
            print(f"Plane: {plane}")
            print(f"Coord: {coord}")
            print(f"Time created: {timecreated}")

        h = struct.unpack(byte_order + 'd', f.read(8))[0]
        zmin = struct.unpack(byte_order + 'd', f.read(8))[0]
        ib = struct.unpack(byte_order + 'i', f.read(4))[0]
        ni = struct.unpack(byte_order + 'i', f.read(4))[0]
        jb = struct.unpack(byte_order + 'i', f.read(4))[0]
        nj = struct.unpack(byte_order + 'i', f.read(4))[0]
        kb = struct.unpack(byte_order + 'i', f.read(4))[0]
        nk = struct.unpack(byte_order + 'i', f.read(4))[0]

        nx = ni - ib + 1
        ny = nj - jb + 1
        nz = nk - kb + 1

        dtype = np.float32 if prec == 4 else np.float64
        volume = np.fromfile(f, dtype=dtype, count=nx * ny * nz).reshape((nx, ny, nz), order='F')

        x = np.arange(ib, ni + 1) * h
        y = np.arange(jb, nj + 1) * h
        z = zmin + np.arange(kb, nk + 1) * h

        ds = xr.Dataset(
            {"volume": (("x", "y", "z"), volume)},
            coords={"x": x, "y": y, "z": z},
            attrs={
                "precision": prec,
                "time": t,
                "coord": coord,
                "grid_spacing": h,
                "time_created": timecreated,
                "mode": mode,
                "gridinfo": gridinfo,
                "ib": ib, "ni": ni, "jb": jb, "nj": nj, "kb": kb, "nk": nk
            }
        )
    return ds