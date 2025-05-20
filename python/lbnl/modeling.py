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
    grid_spacing=1.0,      # meters
    smoothing=True,
    smooth_sigma=1.,      # meters
    plot=False,
    # ---- new extrapolation options ----------------------------------
    extrapolate=False,
    extrap_method='rbf'  # 'nearest' or 'rbf'
):
    """
    Interpolates irregular fracture data onto a regular grid and writes to
    an SW4 material interface file.  Optionally extrapolates beyond the
    convex hull of the input points.

    Parameters
    ----------
    frac_path         : path to the frac file (.csv with XYZ columns)
    origin            : (x0, y0, z0) origin in meters
    vertical_thickness: thickness of the interface layer in meters
    grid_spacing      : regular grid step (meters)
    plot              : if True, show a quick surface plot
    extrapolate       : if True, fill NaN's by nearest/RBF extrapolation
    extrap_method     : 'nearest' | 'rbf' (only if extrapolate=True)

    Returns
    -------
    None  (writes <stem>_ifile1.txt to disk)
    """

    # -- load & rebase ---------------------------------------------------
    arr = np.loadtxt(frac_path, skiprows=1, delimiter=',')
    arr *= 0.3048   # ft → m
    arr[:,0] -= origin[0]
    arr[:,1] -= origin[1]
    arr[:,2]  = origin[2] - arr[:,2]

    x, y, z = arr[:,0], arr[:,1], arr[:,2]

    # -- define regular grid --------------------------------------------
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_regular = np.arange(x_min, x_max+grid_spacing, grid_spacing)
    y_regular = np.arange(y_min, y_max+grid_spacing, grid_spacing)
    Xi, Yi = np.meshgrid(x_regular, y_regular, indexing='xy')

    # -- first pass: linear, then cubic fallback -------------------------
    Zi  = griddata((x, y), z, (Xi, Yi), method='cubic')

    # -- extrapolation fill ------------------------------------------------
    if extrapolate:
        if extrap_method == 'nearest':
            Zi_nn = griddata((x, y), z, (Xi, Yi), method='nearest')
            Zi = np.where(np.isnan(Zi), Zi_nn, Zi)

        elif extrap_method == 'rbf':
            # RBF fits a global surface and will extrapolate outside hull
            rbf = Rbf(x, y, z, function='thin_plate')
            Zi_rbf = rbf(Xi, Yi)
            Zi = np.where(np.isnan(Zi), Zi_rbf, Zi)

        else:
            raise ValueError("extrap_method must be 'nearest' or 'rbf'")
    if smoothing:
            Zi = gaussian_filter(Zi, sigma=smooth_sigma)
    # -- optional plot -----------------------------------------------------
    if plot:
        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xi, Yi, Zi, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Interpolated (and extrapolated) Fracture Surface")
        plt.tight_layout()
        plt.show()

    # -- write interface file ---------------------------------------------
    stem   = Path(frac_path).stem
    ifile1 = Path(frac_path).with_name(f"{stem}_ifile1.txt")
    Nx, Ny = len(x_regular), len(y_regular)

    with open(ifile1, "w") as f:
        # header: N_x, N_y, 2 layers
        f.write(f"{Nx} {Ny} 2\n")
        for j in range(Ny):
            for i in range(Nx):
                top_z = Zi[j, i]
                bot_z = top_z + vertical_thickness
                f.write(f"{x_regular[i]:.3f} {y_regular[j]:.3f} "
                        f"{top_z:.3f} {bot_z:.3f}\n")

    print(f"Wrote interface file: {ifile1}")


def write_sw4_input(path, inventory, outdir, sources=['DMUS3'], origin=(1200, -910, 360),
                    tmax=1.0, write_model=False, ifiles=None, material=None, pfile=None):
    """
    Write a collab rfile for SW4
    
    :param path: path to write the input file to
    :param ifiles: list of interface files
    :param material: Dict with keys Vp, Vs, rho
    :param inventory: obspy Inventory object
    :param origin: (x, y, z) origin of the model (meters HMC; elevation masl)
    :param dt: time step (seconds)
    :param tmax: max time (seconds)
    :return: None
    """
    # Check that either (ifiles and material) or pfile is provided
    if not ((ifiles and material) or pfile):
        raise ValueError("Either 'ifiles' and 'material' must be provided, or 'pfile' must be provided.")

    with open(path, "w") as f:
        # Write the header
        f.write("# SW4 input file\n")
        f.write("# Generated by write_sw4_input\n")
        f.write("#\n")
        f.write(f'fileio path={outdir} verbose=2 printcycle=100 pfs=0\n')
        f.write(f'time t={tmax}\n')
        f.write('grid x=80 y=50 z=60 h=.1\n')
        # Now the sources/stations
        for sta in inventory[0]:  # Assumes only one network in this inventory
            if sta.code[-2] == 'S':  # Source
                coords = sta.extra
                x = (coords['hmc_east'].value * .3048) - origin[0]
                y = (coords['hmc_north'].value * .3048) - origin[1]
                z = origin[2] - (coords['hmc_elev'].value)
                if sta.code in sources:
                    f.write(f'source x={x} y={y} z={z} freq=400. type=RickerInt t0=0.0165\n')
                else:
                    f.write(f'#source x={x} y={y} z={z} freq=400. type=RickerInt t0=0.0165\n')
            else:  # Hydrophone or accelerometer
                coords = sta.extra
                x = (coords['hmc_east'].value * .3048) - origin[0]
                y = (coords['hmc_north'].value * .3048) - origin[1]
                z = origin[2] - (coords['hmc_elev'].value)
                # Note here that, for grouted hydrophones you should use strain then convert to stress tensor
                # ...pressure in a solid is the mean compressive stress.
                f.write(f'rec x={x} y={y} z={z} sta={sta.code} variables=velocity\n')
        # Now the material properties
        if ifiles and material:
            for i, ifile in enumerate(ifiles):
                f.write(f'ifile filename={ifile} input=cartesian\n')
                f.write(f'material id={i} vp={material["Vp"]} vs={material["Vs"]} rho={material["rho"]}\n')
                f.write(f'material id={i+1} vp=1500 vs=10 rho=1000\n')
            f.write(f'block vp={material["Vp"]} vs={material["Vs"]} rho={material["rho"]}\n')
        elif pfile:
            f.write(f'pfile filename={pfile} style=cartesian\n')
        # Need to change lz boundary condition to 2 (sepergrid; same as all others)
        f.write('boundary_conditions lz=2\n')
        if write_model:
            f.write(f'volimage file=model mode=rho cycle=0\n')
            f.write(f'volimage file=model mode=p cycle=0\n')
            f.write(f'volimage file=model mode=s cycle=0\n')
    return


def write_pfile_from_fracture_fullgrid(
    ifile_path,
    # ------------- global origin ----------------------------------------
    origin=(1200.0, -910.0, 360.0),  # (x0,y0,z0) in m (HMC; elevation masl)
    rebased=True,                    # If True, assume CSV is already local XY (0–80,0–50)
                                     # and Z positive-down [0–model_depth].
    # ------------- geometry & mesh ---------------------------------------
    model_depth=60.0,                # m, positive down
    dxy=0.5,                         # horizontal spacing
    dz_coarse=0.5,                   # vertical coarse spacing
    dz_fine=0.01,                    # vertical fine spacing
    # ---------- optional grid override ----------------------------------
    grid_bounds=None,                # ((xmin,xmax),(ymin,ymax)) after rebasing
    contiguous: bool = False,
    contiguous_method: str = "linear",  # 'linear' or 'cubic'
    # ------------- materials ---------------------------------------------
    mat_above=None,
    mat_inside=None,
    mat_below=None,
    # ------------- coordinate system -------------------------------------
    coord_system="cartesian",        # 'cartesian' or 'geographic'
    lon0_deg=0.0,
    lat0_deg=0.0,
    earth_radius=6_371_000.0,
    # ------------- p-file header extras ---------------------------------
    model_name="LocalModel",
    discontinuity_idx=(-99, -99, -99, -99),
    q_available=False,
    qp_const=1000.0,
    qs_const=600.0,
    # ------------- housekeeping ------------------------------------------
    outfile=None,
    snap_tol=1e-6
):
    """
    Build an SW4 p-file (7-line global header + per-profile blocks).
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
    """Exactly your old header parser."""
    model_name = lines[0].strip()
    delta_deg  = float(lines[1])
    NX, Xmin, Xmax = map(float, lines[2].split())
    NY, Ymin, Ymax = map(float, lines[3].split())
    Ndep, dmin_m, dmax_m = map(float, lines[4].split())
    Ised, IMoho, I410, I660 = map(int, lines[5].split())
    qflag = lines[6].strip()
    NY = int(NY); NX = int(NX); Ndep = int(Ndep)
    header_info = dict(
        model_name=model_name, delta_deg=delta_deg,
        NY=NY, NX=NX, Ndep=Ndep,
        Ymin=Ymin, Ymax=Ymax,
        Xmin=Xmin, Xmax=Xmax,
        dmin_m=dmin_m, dmax_m=dmax_m,
        Ised=Ised, IMoho=IMoho, I410=I410, I660=I660,
        q_available=(qflag.upper()=='T')
    )
    return header_info


def _scan_file_for_offsets(path):
    """
    Do one quick Python pass: read the first 7 lines, record header;
    then record the byte offset *in the file* where each latitude-row
    of Nx profiles begins.
    Returns (hdr, byte_offset, lat_grid, lon_grid, depth_km)
    """
    path = Path(path)
    Y_grid = X_grid = depth_m = None
    offsets = []
    with open(path, 'rb') as f:
        # read global header
        lines = [f.readline().decode('ascii') for _ in range(7)]
        hdr = _parse_header(lines)
        NY, NX, Nz = hdr['NY'], hdr['NX'], hdr['Ndep']
        hdr_q = hdr['q_available']

        # read the lat/lon arrays (we don't actually have them in the ASCII,
        # so we must reconstruct them here exactly as you did in write)
        Ymin, Ymax = hdr['Ymin'], hdr['Ymax']
        Xmin, Xmax = hdr['Xmin'], hdr['Xmax']
        Y_grid = np.linspace(Ymin, Ymax, NY)
        X_grid = np.linspace(Xmin, Xmax, NX)

        # now for each latitude j, record file_offset
        for j in range(NY):
            offsets.append(f.tell())
            # skip Nx profiles
            for i in range(NX):
                # skip one profile‐header line
                _ = f.readline()
                # skip Nz data lines
                for _ in range(Nz):
                    _ = f.readline()
        # we also want the depth_km from the very first profile
    # get depth_km out of the first profile cheaply (reuse the offset)
    with open(path, 'rb') as f:
        f.seek(offsets[0])
        _ = f.readline().decode('ascii')          # profile header
        raw = np.asarray([f.readline().decode('ascii').split() for _ in range(Nz)])
        data0 = np.asarray(raw, dtype=float)
        depth_m = data0[:,1]
    return hdr, offsets, Y_grid, X_grid, depth_m


# change the signature to accept Nlon explicitly
@dask.delayed
def _read_latitude_row(path, byte_offset, Nz, NX, q_available):
    """Read one latitude‐row of Nlon profiles, each Nz deep."""
    path = Path(path)
    with open(path, 'rb') as f:
        f.seek(byte_offset)
        rows_vp, rows_vs, rows_rh = [], [], []
        rows_qp, rows_qs = [], []
        for _ in range(NX):
            f.readline().decode('ascii')  # skip profile header
            raw = np.asarray([f.readline().decode('ascii').split() for __ in range(Nz)])
            data = np.asarray(raw, dtype=float)
            rows_vp.append(data[:,2])
            rows_vs.append(data[:,3])
            rows_rh.append(data[:,4])
            if q_available:
                rows_qp.append(data[:,5])
                rows_qs.append(data[:,6])

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
    A Dask‐ified version of your read.  We chunk in blocks of `chunk_rows`
    along the latitude axis.
    """
    path = Path(path)
    hdr, offsets, Y_grid, X_grid, depth_m = _scan_file_for_offsets(path)
    Nz = hdr['Ndep']
    Ny = hdr['NY']
    Nx = hdr['NX']
    q_avail = hdr['q_available'] and return_q

    # Build a list of delayed tasks, one per *chunk* of rows
    row_tasks = []
    for start_j in range(0, Ny, chunk_rows):
        end_j = min(start_j + chunk_rows, Ny)
        offs = offsets[start_j:end_j]
        # one delayed call returns tuples of numpy arrays
        row_tasks.append(_read_latitude_row(path, offs[0], Nz, Nx, q_avail))

    # Now turn them into dask arrays and stack them
    vp_chunks = []
    vs_chunks = []
    rh_chunks = []
    if q_avail:
        qp_chunks = []
        qs_chunks = []

    for task in row_tasks:
    # pick off each output by index
        vp_t = task[0]
        vs_t = task[1]
        rh_t = task[2]

        if q_avail:
            qp_t = task[3]
            qs_t = task[4]
        # Each vp_t is shape (Nx, Nz) → turn that into a delayed 3D block (1, Nx, Nz)
        vp_da = da.from_delayed(vp_t, shape=(Nx, Nz), dtype=np.float32)
        vs_da = da.from_delayed(vs_t, shape=(Nx, Nz), dtype=np.float32)
        rh_da = da.from_delayed(rh_t, shape=(Nx, Nz), dtype=np.float32)

        # add a newaxis for the latitude chunk
        vp_chunks.append(vp_da[np.newaxis, ...])
        vs_chunks.append(vs_da[np.newaxis, ...])
        rh_chunks.append(rh_da[np.newaxis, ...])

        if q_avail:
            qp_da = da.from_delayed(qp_t, shape=(Nx, Nz), dtype=np.float32)
            qs_da = da.from_delayed(qs_t, shape=(Nx, Nz), dtype=np.float32)
            qp_chunks.append(qp_da[np.newaxis, ...])
            qs_chunks.append(qs_da[np.newaxis, ...])

    # concatenate all latitude chunks
    Vp = da.concatenate(vp_chunks, axis=0)
    Vs = da.concatenate(vs_chunks, axis=0)
    Rh = da.concatenate(rh_chunks, axis=0)
    if q_avail:
        Qp = da.concatenate(qp_chunks, axis=0)
        Qs = da.concatenate(qs_chunks, axis=0)

    # Now Vp.shape == (Ny, Nx, Nz), chunked in latitude‐blocks of size chunk_rows
    depth = depth_m  # negative up, metres
    coords = dict(
      latitude  = ('latitude', Y_grid),
      longitude = ('longitude', X_grid),
      depth = ('depth', depth),
    )
    data_vars = dict(
      Vp      = (('latitude','longitude','depth'), Vp),
      Vs      = (('latitude','longitude','depth'), Vs),
      density = (('latitude','longitude','depth'), Rh),
    )
    if q_avail:
      data_vars['Qp'] = (('latitude','longitude','depth'), Qp)
      data_vars['Qs'] = (('latitude','longitude','depth'), Qs)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=hdr)
    return ds


def read_sw4img_volume(filename, verbose=False, machineformat='native'):
    """
    Reads a full 3D volume from a .sw4img file (with plane = -1) and returns an xarray Dataset.

    Parameters:
    - filename: Path to the .sw4img file.
    - verbose: If True, prints header information.
    - machineformat: Byte order ('native', 'little', 'big').

    Returns:
    - ds: xarray Dataset containing the volume and metadata.
    """
    byte_order = {'native': '=', 'little': '<', 'big': '>'}.get(machineformat, '=')

    with open(filename, 'rb') as f:
        # Header
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

        # Only supporting one patch for now
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

        # Define coordinate arrays
        x = np.arange(ib, ni + 1) * h
        y = np.arange(jb, nj + 1) * h
        z = zmin + np.arange(kb, nk + 1) * h

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                "volume": (("x", "y", "z"), volume)
            },
            coords={
                "x": x,
                "y": y,
                "z": z
            },
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