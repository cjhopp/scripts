
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
from matplotlib.tri import Triangulation, LinearTriInterpolator
import datetime
import rioxarray
import plotly.graph_objects as go
from pyproj import Proj, Transformer


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


def surfaces_to_velocity_volume(
    sorted_surfaces: List[Tuple[str, pd.DataFrame, np.ndarray]],
    velocity_map: Dict[str, float],
    grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    fill_velocity_top: float = 500.0,
    plot_debug: bool = False,
    precision_decimals: int = 4
) -> xr.DataArray:
    """
    Creates a 3D velocity volume by using a stable triangulation-based interpolation.
    """
    X, Y, Z = grid_coords
    nz, ny, nx = Z.shape
    
    velocity_grid = np.full(Z.shape, np.nan, dtype=np.float32)
    grid_points_2d_flat = (X[0, :, :].flatten(), Y[0, :, :].flatten())

    interpolated_surfs = {}
    for name, verts, faces in sorted_surfaces:
        points = verts[['X', 'Y']].values
        values = verts['Z'].values
        
        # Use the triangulation from the file for stable interpolation
        tri = Triangulation(points[:, 0], points[:, 1], triangles=faces)
        interpolator = LinearTriInterpolator(tri, values)
        
        grid_z_flat = interpolator(grid_points_2d_flat[0], grid_points_2d_flat[1])
        grid_z = grid_z_flat.filled(np.nan).reshape((ny, nx))
        
        interpolated_surfs[name] = grid_z
        
    if plot_debug:
        plot_gridded_surfaces_3d(interpolated_surfs, X[0, :, :], Y[0, :, :])

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

            # Stable sort as a defense-in-depth
            local_surfs.sort(
                key=lambda item: (round(item[0], precision_decimals), name_to_global_rank[item[1]]),
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
    plot_debug: bool = False
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
        plot_debug=plot_debug
    )

    print("4. Finalizing dataset...")
    ds = velocity_da.to_dataset()
    ds.attrs['created_at'] = datetime.datetime.utcnow().isoformat()
    ds.attrs['source_directory'] = str(Path(surfaces_directory).resolve())
    ds.attrs['grid_spacing_xyz'] = str(grid_spacing)
    ds.attrs['stratigraphic_order'] = [name for name, _, _ in sorted_surfaces]
    ds.attrs['velocity_map'] = str(velocity_map)

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
