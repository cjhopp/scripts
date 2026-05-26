#!/usr/bin/env python
"""Convert depth-domain SEG-Y velocity cubes to NonLinLoc NLLGrid files."""

import argparse
import sys
from pathlib import Path

import numpy as np
from nllgrid import NLLGrid

try:
    from lbnl.grid_utils import read_segy_velocity_to_xarray
except ModuleNotFoundError:
    # Support direct script execution: python /path/to/lbnl/segy_to_nllgrid.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from grid_utils import read_segy_velocity_to_xarray


def _coord_spacing(values):
    """Return median spacing for sorted coordinate values."""
    vals = np.asarray(values, dtype=np.float64)
    if vals.size < 2:
        raise ValueError('Need at least two coordinate values to estimate spacing')
    diffs = np.diff(np.sort(vals))
    diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > 0)]
    if diffs.size == 0:
        raise ValueError('Coordinate spacing is zero or undefined')
    return float(np.median(diffs))


def _utm_to_latlon(x_m, y_m, crs='EPSG:26910'):
    """Convert projected coordinates to lon/lat using pyproj if available."""
    try:
        from pyproj import Transformer
    except ImportError:
        return None, None

    transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(float(x_m), float(y_m))
    return float(lat), float(lon)


def _write_nllgrid(
    array_slow_len,
    basename,
    outdir,
    dx_km,
    dy_km,
    dz_km,
    z_orig_km,
    x_orig_km=0.0,
    y_orig_km=0.0,
    orig_lat=None,
    orig_lon=None,
    proj_name='SIMPLE',
):
    """Write one NLLGrid model (BUF + HDR)."""
    grid = NLLGrid()
    grid.dx = dx_km
    grid.dy = dy_km
    grid.dz = dz_km
    grid.x_orig = x_orig_km
    grid.y_orig = y_orig_km
    grid.z_orig = z_orig_km
    grid.type = 'SLOW_LEN'
    grid.proj_name = proj_name
    if orig_lat is not None and orig_lon is not None:
        grid.orig_lat = float(orig_lat)
        grid.orig_lon = float(orig_lon)
    grid.array = np.flip(array_slow_len, axis=2)
    grid.basename = str(Path(outdir) / basename)
    grid.write_buf_file()
    grid.write_hdr_file()
    return grid


def _plot_nllgrid_builtin(grid, out_png, slice_index='max', cmap='viridis'):
    """Plot grid using NLLGrid built-in plot method and save to file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Plotting requested but matplotlib is not installed') from exc

    axes, _ = grid.plot(slice_index=slice_index, handle=True, cmap=cmap)
    fig = axes[0].get_figure()
    fig.savefig(str(out_png), dpi=180, bbox_inches='tight')
    plt.close(fig)


def export_segy_to_nllgrid(
    segy_path,
    outdir,
    basename='velocity',
    datum_asl_m=2000.0,
    sample_interval_m=5.0,
    velocity_var='Vp',
    vpvs_ratio=None,
    infer_xy_from_corners=False,
    sw_corner=(632800.0, 4839600.0),
    bin_size_m=10.0,
    iline_origin=941,
    xline_origin=941,
    input_crs='EPSG:26910',
    proj_name='SIMPLE',
    orig_lat=None,
    orig_lon=None,
    write_plots=False,
    plot_slice='max',
    plot_cmap='viridis',
):
    """Read SEG-Y into xarray and export NonLinLoc SLOW_LEN grids."""
    ds = read_segy_velocity_to_xarray(
        segy_path=segy_path,
        datum_asl_m=datum_asl_m,
        sample_interval_m=sample_interval_m,
        velocity_name=velocity_var,
        infer_xy_from_corners=infer_xy_from_corners,
        sw_corner=sw_corner,
        bin_size_m=bin_size_m,
        iline_origin=iline_origin,
        xline_origin=xline_origin,
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if velocity_var not in ds:
        raise KeyError(f"Velocity variable '{velocity_var}' not found in dataset")

    vp = ds[velocity_var].transpose('iline', 'xline', 'depth_m')

    if 'easting' in ds.coords and 'northing' in ds.coords:
        x_vals = ds['easting'].isel(iline=0).values
        y_vals = ds['northing'].isel(xline=0).values
        dx_m = _coord_spacing(x_vals)
        dy_m = _coord_spacing(y_vals)
        x0_m = float(np.nanmin(x_vals))
        y0_m = float(np.nanmin(y_vals))
    else:
        dx_m = float(bin_size_m)
        dy_m = float(bin_size_m)
        x0_m = 0.0
        y0_m = 0.0

    dz_m = float(sample_interval_m)
    z0_km = float(np.nanmin(ds['depth_m'].values)) / 1000.0

    if orig_lat is None or orig_lon is None:
        auto_lat, auto_lon = _utm_to_latlon(x0_m, y0_m, crs=input_crs)
        if auto_lat is not None and auto_lon is not None:
            orig_lat = auto_lat
            orig_lon = auto_lon

    vp_vals = vp.values.astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        pslow_len = np.where(vp_vals > 0, dz_m / vp_vals, np.nan)

    p_grid = _write_nllgrid(
        array_slow_len=pslow_len,
        basename=f'{basename}.P.mod',
        outdir=outdir,
        dx_km=dx_m / 1000.0,
        dy_km=dy_m / 1000.0,
        dz_km=dz_m / 1000.0,
        z_orig_km=z0_km,
        x_orig_km=0.0,
        y_orig_km=0.0,
        orig_lat=orig_lat,
        orig_lon=orig_lon,
        proj_name=proj_name,
    )

    if write_plots:
        _plot_nllgrid_builtin(
            p_grid,
            out_png=outdir / f'{basename}.P.mod.png',
            slice_index=plot_slice,
            cmap=plot_cmap,
        )

    s_grid = None
    if vpvs_ratio is not None:
        vpvs = float(vpvs_ratio)
        if vpvs <= 0:
            raise ValueError('vpvs_ratio must be > 0')
        vs_vals = vp_vals / vpvs
        with np.errstate(divide='ignore', invalid='ignore'):
            sslow_len = np.where(vs_vals > 0, dz_m / vs_vals, np.nan)
        s_grid = _write_nllgrid(
            array_slow_len=sslow_len,
            basename=f'{basename}.S.mod',
            outdir=outdir,
            dx_km=dx_m / 1000.0,
            dy_km=dy_m / 1000.0,
            dz_km=dz_m / 1000.0,
            z_orig_km=z0_km,
            x_orig_km=0.0,
            y_orig_km=0.0,
            orig_lat=orig_lat,
            orig_lon=orig_lon,
            proj_name=proj_name,
        )

        if write_plots:
            _plot_nllgrid_builtin(
                s_grid,
                out_png=outdir / f'{basename}.S.mod.png',
                slice_index=plot_slice,
                cmap=plot_cmap,
            )

    return ds, p_grid, s_grid


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Convert depth-domain SEG-Y velocity to NonLinLoc NLLGrid files'
    )
    parser.add_argument('segy_path', help='Path to depth-domain SEG-Y velocity file')
    parser.add_argument('--outdir', default='.', help='Output directory for .buf/.hdr files')
    parser.add_argument('--basename', default='velocity', help='Output basename prefix')
    parser.add_argument('--datum-asl-m', type=float, default=2000.0, help='Datum elevation in meters ASL')
    parser.add_argument('--sample-interval-m', type=float, default=5.0, help='Depth sample interval in meters')
    parser.add_argument('--velocity-var', default='Vp', help='Variable name to store in xarray')
    parser.add_argument('--vpvs-ratio', type=float, default=None, help='Optional constant Vp/Vs to write S grid')
    parser.add_argument('--infer-xy-from-corners', action='store_true', help='Infer XY from corner/bin metadata')
    parser.add_argument('--sw-easting', type=float, default=632800.0, help='SW corner easting (m)')
    parser.add_argument('--sw-northing', type=float, default=4839600.0, help='SW corner northing (m)')
    parser.add_argument('--bin-size-m', type=float, default=10.0, help='Bin size in meters')
    parser.add_argument('--iline-origin', type=int, default=941, help='Inline index at SW corner')
    parser.add_argument('--xline-origin', type=int, default=941, help='Crossline index at SW corner')
    parser.add_argument('--input-crs', default='EPSG:26910', help='CRS for easting/northing (used for lat/lon origin)')
    parser.add_argument('--proj-name', default='SIMPLE', help='NLLGrid projection name in header')
    parser.add_argument('--orig-lat', type=float, default=None, help='Override origin latitude for NLLGrid header')
    parser.add_argument('--orig-lon', type=float, default=None, help='Override origin longitude for NLLGrid header')
    parser.add_argument('--plot', action='store_true', help='Write QC PNGs using NLLGrid built-in plot')
    parser.add_argument('--plot-slice', default='max', help="Slice index for NLLGrid plot (e.g., 'max', 'min', or i,j,k)")
    parser.add_argument('--plot-cmap', default='viridis', help='Colormap name for built-in NLLGrid plot')
    return parser


def _parse_slice_arg(value):
    """Parse --plot-slice argument into accepted NLLGrid plot input."""
    txt = str(value).strip().lower()
    if txt in ('max', 'min'):
        return txt
    if ',' in txt:
        parts = [p.strip() for p in txt.split(',')]
        if len(parts) != 3:
            raise ValueError("--plot-slice with commas must be 'i,j,k'")
        return [int(parts[0]), int(parts[1]), int(parts[2])]
    return int(txt)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    plot_slice = _parse_slice_arg(args.plot_slice)

    _, _, s_grid = export_segy_to_nllgrid(
        segy_path=args.segy_path,
        outdir=args.outdir,
        basename=args.basename,
        datum_asl_m=args.datum_asl_m,
        sample_interval_m=args.sample_interval_m,
        velocity_var=args.velocity_var,
        vpvs_ratio=args.vpvs_ratio,
        infer_xy_from_corners=args.infer_xy_from_corners,
        sw_corner=(args.sw_easting, args.sw_northing),
        bin_size_m=args.bin_size_m,
        iline_origin=args.iline_origin,
        xline_origin=args.xline_origin,
        input_crs=args.input_crs,
        proj_name=args.proj_name,
        orig_lat=args.orig_lat,
        orig_lon=args.orig_lon,
        write_plots=args.plot,
        plot_slice=plot_slice,
        plot_cmap=args.plot_cmap,
    )

    if s_grid is None:
        print('Wrote P grid only (no --vpvs-ratio provided).')
    else:
        print('Wrote P and S grids.')


if __name__ == '__main__':
    main()
