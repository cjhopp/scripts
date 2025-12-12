"""
Tools for creating 3D velocity models from geologic surfaces.
This module provides functions to read GoCAD .ts surface files,
build a 3D grid, and populate it with seismic velocities based on
the stratigraphic order of the surfaces.
"""
import os
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, generate_binary_structure
from matplotlib.tri import Triangulation, LinearTriInterpolator
import datetime
import rioxarray
import plotly.graph_objects as go
from pyproj import Proj, Transformer
import pyvista as pv


def read_ts(path: Union[str, Path]) -> List[Dict[str, Union[pd.DataFrame, np.ndarray]]]:
    """
    Reads a GoCAD .ts file and extracts vertex and face data for each object.
    Handles non-sequential and non-numeric vertex IDs.
    """
    path = Path(path)
    if not path.is_file() or path.suffix != ".ts":
        raise FileNotFoundError(f"File not found or is not a .ts file: {path}")

    objects = []
    vertices, faces = [], []
    vrtx_map = {} # Map from file vertex ID to 0-based index
    vrtx_idx = 0
    columns = ["X", "Y", "Z"] # Base columns
    prop_columns = []

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            line_type = parts[0]

            if line_type == "PROPERTIES":
                prop_columns = parts[1:]

            elif line_type in ("VRTX", "PVRTX"):
                v_id = parts[1]
                v_coords = parts[2:5]
                v_props = parts[5:]
                
                if v_id not in vrtx_map:
                    vrtx_map[v_id] = vrtx_idx
                    full_vertex_data = v_coords + v_props
                    vertices.append(full_vertex_data)
                    vrtx_idx += 1

            elif line_type == "TRGL":
                try:
                    face_indices = [vrtx_map[p] for p in parts[1:4]]
                    faces.append(face_indices)
                except KeyError as e:
                    print(f"Warning: Vertex ID {e} in TRGL not found in VRTX block. Skipping face.")

            elif line_type == "END":
                if vertices:
                    all_columns = columns + prop_columns
                    vert_df = pd.DataFrame(vertices).apply(pd.to_numeric, errors='coerce')
                    vert_df.columns = all_columns[:len(vert_df.columns)]
                    
                    face_arr = np.array(faces, dtype=np.int32)
                    objects.append({"vertices": vert_df, "faces": face_arr})

                    vertices, faces, vrtx_map, prop_columns = [], [], {}, []
                    vrtx_idx = 0
    return objects


def plot_gridded_surfaces_3d(
    surfaces: Dict[str, np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    title: str = "Gridded Geologic Surfaces"
):
    """
    Creates a 3D plot of gridded geologic surfaces and saves it as an HTML file.
    """
    fig = go.Figure()
    colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd']
    
    for i, (name, z_grid) in enumerate(surfaces.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=z_grid,
            name=name,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=0.8,
            hoverinfo='name+z',
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], 
            mode='markers',
            marker=dict(size=10, color=color),
            name=name,
            showlegend=True
        ))

    fig.update_layout(
        title_text=title,
        showlegend=True,
        legend=dict(title='Layers'),
        scene=dict(
            xaxis_title='X (Easting)',
            yaxis_title='Y (Northing)',
            zaxis_title='Z (Elevation)',
            aspectratio=dict(x=1.5, y=1.5, z=0.5)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    filename = "debug_plot_gridded_surfaces.html"
    fig.write_html(filename)
    print(f"\n---> Debug plot of interpolated surfaces saved to: {os.path.abspath(filename)}\n")


def load_surfaces_from_directory(
    directory: Union[str, Path], 
    velocity_map: Dict[str, float]
) -> Tuple[List[Tuple[str, pd.DataFrame, np.ndarray]], Dict[str, float]]:
    """
    Loads all .ts surfaces from a directory, including their triangulation (faces).
    """
    directory = Path(directory)
    surfaces = []
    
    for file_path in directory.glob("*.ts"):
        surface_name = file_path.stem
        if surface_name in velocity_map:
            print(f"Processing: {file_path.name}")
            ts_object = read_ts(file_path)[0]
            vertices_df = ts_object['vertices']
            faces_arr = ts_object['faces']
            
            mean_elevation = vertices_df['Z'].mean()
            surfaces.append((mean_elevation, surface_name, vertices_df, faces_arr))
        else:
            print(f"Warning: No velocity found for '{surface_name}' in velocity_map. Skipping.")

    if not surfaces:
        raise ValueError("No valid surfaces found in the directory that match the velocity_map.")

    surfaces.sort(key=lambda x: x[0], reverse=True)
    sorted_surfaces = [(name, df, faces) for _, name, df, faces in surfaces]

    all_verts = pd.concat([df for _, df, _ in sorted_surfaces])
    extent = {
        'xmin': all_verts['X'].min(), 'xmax': all_verts['X'].max(),
        'ymin': all_verts['Y'].min(), 'ymax': all_verts['Y'].max(),
        'zmin': all_verts['Z'].min(), 'zmax': all_verts['Z'].max(),
    }

    return sorted_surfaces, extent

def _stitch_gaps(
    grid_z: np.ndarray,
    X_grid: np.ndarray, 
    Y_grid: np.ndarray, 
    iterations: int = 3
) -> np.ndarray:
    """
    Stitch local gaps by iteratively filling NaNs at the fringe of valid data.
    This provides controlled gap-filling without large-scale extrapolation or artifacts.
    """
    if iterations == 0:
        return grid_z

    filled_grid = np.copy(grid_z)
    struct = generate_binary_structure(2, 1)  # 4-connectivity

    for i in range(iterations):
        nan_mask = np.isnan(filled_grid)
        if not np.any(nan_mask):
            break  # Exit if there are no NaNs left

        # Identify NaN pixels that are directly adjacent to valid data points
        border_nans = binary_dilation(~nan_mask, structure=struct) & nan_mask
        if not np.any(border_nans):
            break  # Exit if no more gaps can be filled

        # These are the valid points on the grid to source data from
        valid_points = np.array([X_grid[~nan_mask], Y_grid[~nan_mask]]).T
        valid_values = filled_grid[~nan_mask]

        # The coordinates of the specific fringe NaNs we want to fill
        fill_points = np.array([X_grid[border_nans], Y_grid[border_nans]]).T
        
        # Use griddata with 'nearest' to fill ONLY these border NaNs
        # This propagates the value from the single nearest valid grid point.
        filled_values = griddata(valid_points, valid_values, fill_points, method='nearest')
        
        # Update the grid for the next iteration
        filled_grid[border_nans] = filled_values

    return filled_grid

def surfaces_to_velocity_volume(
    sorted_surfaces: List[Tuple[str, pd.DataFrame, np.ndarray]],
    velocity_map: Dict[str, float],
    grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    fill_velocity_top: float = 500.0,
    plot_debug: bool = False,
    precision_decimals: int = 1,
    stitching_iterations: int = 3
) -> xr.DataArray:
    """
    Creates a 3D velocity volume using triangulation and controlled gap stitching.
    """
    X, Y, Z = grid_coords
    nz, ny, nx = Z.shape
    
    velocity_grid = np.full(Z.shape, np.nan, dtype=np.float32)
    X_grid_2d, Y_grid_2d = X[0, :, :], Y[0, :, :]

    interpolated_surfs = {}
    for name, verts, faces in sorted_surfaces:
        points = verts[['X', 'Y']].values
        values = verts['Z'].values
        
        # Step 1: Initial interpolation using triangulation. This is fast and accurate
        # but will leave NaNs outside the data's convex hull, which is what we want.
        tri = Triangulation(points[:, 0], points[:, 1], triangles=faces)
        interpolator = LinearTriInterpolator(tri, values)
        grid_z_flat = interpolator(X_grid_2d.flatten(), Y_grid_2d.flatten())
        grid_z = grid_z_flat.filled(np.nan).reshape((ny, nx))
        
        # Step 2: **Stitch Gaps.** Iteratively fill NaNs at the edges of the valid data area.
        # This closes small gaps between adjacent surfaces without extrapolating wildly.
        grid_z_stitched = _stitch_gaps(
            grid_z=grid_z,
            X_grid=X_grid_2d,
            Y_grid=Y_grid_2d,
            iterations=stitching_iterations
        )
        
        interpolated_surfs[name] = grid_z_stitched
        
    if plot_debug:
        plot_gridded_surfaces_3d(interpolated_surfs, X_grid_2d, Y_grid_2d)

    surf_names = [name for name, _, _ in sorted_surfaces]
    name_to_global_rank = {name: i for i, name in enumerate(surf_names)}
    all_surfs_z = np.array([interpolated_surfs[name] for name in surf_names])

    for y_idx in range(ny):
        for x_idx in range(nx):
            column_z_values = all_surfs_z[:, y_idx, x_idx]
            local_surfs = [
                (z_val, name) for z_val, name in zip(column_z_values, surf_names)
                if not np.isnan(z_val)
            ]

            if not local_surfs:
                continue

            # Correctly implement a stable sort for tie-breaking
            local_surfs.sort(
                key=lambda item: (round(item[0], precision_decimals), -name_to_global_rank[item[1]]),
                reverse=True
            )

            Z_col = Z[:, y_idx, x_idx]
            top_z, _ = local_surfs[0]
            velocity_grid[:, y_idx, x_idx][Z_col > top_z] = fill_velocity_top

            for i, (top_surf_z, name) in enumerate(local_surfs):
                velocity = velocity_map[name]
                if i + 1 < len(local_surfs):
                    bottom_surf_z, _ = local_surfs[i+1]
                else:
                    bottom_surf_z = -np.inf
                
                mask = (Z_col <= top_surf_z) & (Z_col > bottom_surf_z)
                velocity_grid[:, y_idx, x_idx][mask] = velocity

    da = xr.DataArray(
        velocity_grid,
        dims=['z', 'y', 'x'],
        coords={'z': Z[:, 0, 0], 'y': Y[0, :, 0], 'x': X[0, 0, :]},
        name='velocity'
    )
    da.attrs['units'] = 'm/s'
    
    return da


def build_velocity_model(
    surfaces_directory: Union[str, Path],
    velocity_map: Dict[str, float],
    grid_spacing: Tuple[float, float, float] = (50.0, 50.0, 25.0),
    extent_buffer: float = 0.0,
    manual_extent: Optional[Dict[str, float]] = None,
    input_crs: str = "EPSG:26911",
    output_crs: Optional[str] = "EPSG:4326",
    plot_debug: bool = False,
    stitching_iterations: int = 3
) -> xr.Dataset:
    """
    Main function to build a 3D velocity model from a directory of .ts surface files.
    """
    print("1. Loading and sorting surfaces...")
    sorted_surfaces, auto_extent = load_surfaces_from_directory(surfaces_directory, velocity_map)

    if manual_extent:
        extent = manual_extent
    else:
        extent = {
            'xmin': auto_extent['xmin'] - extent_buffer,
            'xmax': auto_extent['xmax'] + extent_buffer,
            'ymin': auto_extent['ymin'] - extent_buffer,
            'ymax': auto_extent['ymax'] + extent_buffer,
            'zmin': auto_extent['zmin'] - extent_buffer,
            'zmax': auto_extent['zmax'] + extent_buffer,
        }

    print("2. Creating 3D grid...")
    dx, dy, dz = grid_spacing
    x_coords = np.arange(extent['xmin'], extent['xmax'], dx)
    y_coords = np.arange(extent['ymin'], extent['ymax'], dy)
    z_coords = np.arange(extent['zmax'], extent['zmin'], -dz)
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    X, Y, Z = X.transpose(2,1,0), Y.transpose(2,1,0), Z.transpose(2,1,0)

    print("3. Interpolating surfaces and building velocity volume...")
    velocity_da = surfaces_to_velocity_volume(
        sorted_surfaces,
        velocity_map,
        (X, Y, Z),
        plot_debug=plot_debug,
        stitching_iterations=stitching_iterations
    )

    print("4. Finalizing dataset...")
    ds = velocity_da.to_dataset()
    ds.attrs['created_at'] = datetime.datetime.utcnow().isoformat()
    ds.attrs['source_directory'] = str(Path(surfaces_directory).resolve())
    ds.attrs['grid_spacing_xyz'] = str(grid_spacing)
    ds.attrs['stratigraphic_order'] = [name for name, _, _ in sorted_surfaces]
    ds.attrs['velocity_map'] = str(velocity_map)

    # --- MOVE PLOTTING HERE (before reprojection) ---
    if plot_debug:
        plot_velocity_model_3d(ds['velocity'], isosurface_levels=[2000, 4000], slice_plane='z')

    print("5. Reprojecting coordinates (if specified)...")
    ds.rio.write_crs(input_crs, inplace=True)

    if output_crs and input_crs.lower() != output_crs.lower():
        ds = ds.rio.reproject(output_crs)
        print(f"Dataset reprojected to {output_crs}")
        ds.attrs['crs'] = output_crs
    else:
        ds.attrs['crs'] = input_crs
        print("No reprojection needed.")

    print("Done.")
    return ds


def plot_velocity_model_3d(
    velocity_da: xr.DataArray,
    colormap: str = 'viridis',
    opacity: float = 0.5,
    isosurface_levels: Optional[List[float]] = None,
    slice_plane: Optional[str] = None,
    downsample_factor: Optional[int] = None
):
    """
    Interactive 3D plotter for velocity model using PyVista (volume rendering the whole cube).
    
    Parameters:
    - velocity_da: xarray DataArray with velocity data (dims: ['z', 'y', 'x']).
    - colormap: Matplotlib colormap name (e.g., 'viridis', 'plasma').
    - opacity: Opacity for volume rendering (0-1).
    - isosurface_levels: List of velocity values for isosurface contours (optional).
    - slice_plane: Add a slice plane ('x', 'y', or 'z') for cross-section viewing (optional).
    - downsample_factor: Factor to downsample the grid for performance. If None, auto-set.
    
    Usage: Call after building the model, e.g., plot_velocity_model_3d(velocity_da).
    """
    # Extract 1D coordinates
    x = velocity_da.coords['x'].values
    y = velocity_da.coords['y'].values
    z = velocity_da.coords['z'].values
    velocity = velocity_da.values
    
    # Auto-set downsample_factor if not provided (aim for <1M points)
    if downsample_factor is None:
        total_points = len(x) * len(y) * len(z)
        if total_points > 1e6:
            downsample_factor = int(np.ceil(total_points / 1e6) ** (1/3))
            print(f"Auto-downsampling grid by factor {downsample_factor} to reduce size for plotting.")
        else:
            downsample_factor = 1
    
    # Downsample
    x = x[::downsample_factor]
    y = y[::downsample_factor]
    z = z[::downsample_factor]
    velocity = velocity[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    # Center coordinates
    x_offset, y_offset, z_offset = x.min(), y.min(), z.min()
    x -= x_offset
    y -= y_offset
    z -= z_offset
    print(f"Coordinates centered by offsets: X-{x_offset:.0f}, Y-{y_offset:.0f}, Z-{z_offset:.0f}")
    
    # Cast and clean data
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = z.astype(np.float32)
    velocity = velocity.astype(np.float32)
    print(f"Velocity shape: {velocity.shape}, min: {velocity.min()}, max: {velocity.max()}, mean: {np.nanmean(velocity)}")
    velocity = np.nan_to_num(velocity, nan=np.nanmean(velocity))
    velocity = np.clip(velocity, 100, 10000)
    
    # Create PyVista StructuredGrid for uniform grid (correct way)
    grid = pv.StructuredGrid()
    grid.dimensions = np.array([len(x), len(y), len(z)])
    grid.origin = (x.min(), y.min(), z.min())
    grid.spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
    grid['velocity'] = velocity.flatten(order='F')  # Match dims ['z', 'y', 'x']
    
    # Plot the whole cube with volume rendering, colored by velocity
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars='velocity', cmap=colormap, opacity=opacity)
    plotter.add_scalar_bar(title="Velocity (m/s)", vertical=True)
    
    # Add isosurfaces if requested
    if isosurface_levels:
        for level in isosurface_levels:
            if velocity.min() <= level <= velocity.max():
                isosurface = grid.contour([level], scalars='velocity')
                plotter.add_mesh(isosurface, color='black', opacity=0.7)
    
    # Add slice plane if requested
    if slice_plane:
        if slice_plane == 'x':
            slice = grid.slice(normal='x')
        elif slice_plane == 'y':
            slice = grid.slice(normal='y')
        elif slice_plane == 'z':
            slice = grid.slice(normal='z')
        plotter.add_mesh(slice, scalars='velocity', cmap=colormap, show_scalar_bar=False)
    
    # Add axes
    plotter.add_axes(
        xlabel=f'X (scaled, offset {x_offset:.0f})',
        ylabel=f'Y (scaled, offset {y_offset:.0f})',
        zlabel=f'Z (scaled, offset {z_offset:.0f})'
    )
    
    plotter.show()
    return plotter

