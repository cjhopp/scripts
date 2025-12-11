
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
import datetime
import rioxarray
import plotly.graph_objects as go
from pyproj import Proj, Transformer


def read_ts(path: Union[str, Path]) -> List[Dict[str, Union[pd.DataFrame, np.ndarray]]]:
    """
    Reads a GoCAD .ts file and extracts vertex and face data for each object.

    Args:
        path (Union[str, Path]): Path to the .ts file.

    Returns:
        List[Dict[str, Union[pd.DataFrame, np.ndarray]]]: A list of dictionaries,
        where each dictionary represents a distinct object in the file and
        contains 'vertices' (a DataFrame) and 'faces' (a NumPy array).
    """
    path = Path(path)
    if not path.is_file() or path.suffix != ".ts":
        raise FileNotFoundError(f"File not found or is not a .ts file: {path}")

    objects = []
    vertices, faces = [], []
    columns = ["id", "X", "Y", "Z"]
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            line_type = parts[0]

            if line_type == "PROPERTIES":
                # Add any additional properties to the columns
                columns.extend(p for p in parts[1:] if p not in columns)

            elif line_type in ("VRTX", "PVRTX"):
                if len(parts) - 1 != len(columns):
                    # Handle cases where properties might be missing for some vertices
                    # Pad with NaNs
                    padded_parts = parts + [np.nan] * (len(columns) - (len(parts) -1))
                    vertices.append(padded_parts[1:])
                else:
                    vertices.append(parts[1:])

            elif line_type == "TRGL":
                faces.append(parts[1:])

            elif line_type == "END":
                if vertices:
                    # Create DataFrame for the collected vertices
                    vert_df = pd.DataFrame(vertices, columns=columns).apply(pd.to_numeric)
                    
                    # Create array for the collected faces
                    face_arr = np.array(faces, dtype=np.int32) - 1 # Adjust for 0-based indexing

                    objects.append({"vertices": vert_df, "faces": face_arr})

                    # Reset for the next object in the file
                    vertices, faces = [], []
                    columns = ["id", "X", "Y", "Z"]
    return objects

def plot_surfaces_3d(surfaces: List[Tuple[str, pd.DataFrame, np.ndarray]], title: str = "Geologic Surfaces"):
    """
    Creates a 3D plot of geologic surfaces and saves it as an HTML file.

    Args:
        surfaces (List[Tuple[str, pd.DataFrame, np.ndarray]]): A list containing tuples of
            (surface_name, vertices_dataframe, faces_numpy_array).
        title (str): The title for the plot.
    """
    fig = go.Figure()
    # Colorblind-friendly palette
    colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd']

    for i, (name, verts, faces) in enumerate(surfaces):
        fig.add_trace(go.Mesh3d(
            x=verts['X'],
            y=verts['Y'],
            z=verts['Z'],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=colors[i % len(colors)],
            opacity=0.7,
            hoverinfo='name'
        ))

    fig.update_layout(
        title_text=title,
        scene=dict(
            xaxis_title='X (Easting)',
            yaxis_title='Y (Northing)',
            zaxis_title='Z (Elevation)',
            aspectratio=dict(x=1, y=1, z=0.5) # Exaggerate Z for clarity
        )
    )
    
    filename = f"debug_plot_{title.replace(' ', '_')}.html"
    fig.write_html(filename)
    print(f"\n---> Debug plot saved to: {os.path.abspath(filename)}\n")


def load_surfaces_from_directory(
    directory: Union[str, Path], 
    velocity_map: Dict[str, float],
    plot_debug: bool = False
) -> Tuple[List[Tuple[str, pd.DataFrame, np.ndarray]], Dict[str, float]]:
    """
    Loads all .ts surfaces from a directory, sorts them by elevation, and determines the model extent.

    Args:
        directory (Union[str, Path]): Directory containing .ts files.
        velocity_map (Dict[str, float]): A dictionary mapping surface names (file basenames)
                                         to the seismic velocity of the layer *below* that surface.
        plot_debug (bool): If True, display an interactive 3D plot of the surfaces.

    Returns:
        Tuple containing:
        - List[Tuple[str, pd.DataFrame, np.ndarray]]: A list of (name, vertices_df, faces_arr) tuples,
                                                       sorted from highest to lowest elevation.
        - Dict[str, float]: The calculated spatial extent of the model (xmin, xmax, etc.).
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
            
            # Convert Z from depth to elevation
            vertices_df['Z'] = -vertices_df['Z']
            
            mean_elevation = vertices_df['Z'].mean()
            surfaces.append((mean_elevation, surface_name, vertices_df, faces_arr))
        else:
            print(f"Warning: No velocity found for '{surface_name}' in velocity_map. Skipping.")

    if not surfaces:
        raise ValueError("No valid surfaces found in the directory that match the velocity_map.")

    # Sort surfaces by mean elevation (highest first)
    surfaces.sort(key=lambda x: x[0], reverse=True)
    
    sorted_surfaces = [(name, df, faces) for _, name, df, faces in surfaces]

    if plot_debug:
        plot_surfaces_3d(sorted_surfaces, title="Surfaces After Loading and Sorting")

    # Determine overall extent from all surfaces
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
    fill_velocity_top: float = 1500.0
) -> xr.DataArray:
    """
    Creates a 3D velocity volume by filling the space between interpolated surfaces.

    Args:
        sorted_surfaces (List[Tuple[str, pd.DataFrame, np.ndarray]]): Surfaces sorted by elevation.
        velocity_map (Dict[str, float]): Map of surface name to the velocity below it.
        grid_coords (Tuple): Tuple of 3D NumPy arrays (X, Y, Z) representing the grid.
        fill_velocity_top (float): Velocity for the volume above the top surface.

    Returns:
        xr.DataArray: A 3D DataArray containing the velocity model.
    """
    X, Y, Z = grid_coords
    # Initialize velocity grid with the value for material above the top layer
    velocity_grid = np.full(X.shape, fill_value=fill_velocity_top, dtype=np.float32)

    # Grid points for interpolation
    grid_points_2d = (X[0, :, :], Y[0, :, :])

    # Interpolate each surface onto the grid
    interpolated_surfs = {}
    for name, verts, _ in sorted_surfaces:
        points = verts[['X', 'Y']].values
        values = verts['Z'].values
        # Use nearest neighbor to fill gaps at the edges of the convex hull
        grid_z = griddata(points, values, grid_points_2d, method='cubic', fill_value=np.nan)
        grid_z_nearest = griddata(points, values, grid_points_2d, method='nearest')
        grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]
        interpolated_surfs[name] = grid_z
        
    # Fill velocity layers from top (highest elevation) to bottom (lowest elevation)
    for i, (name, _, _) in enumerate(sorted_surfaces):
        top_surf_z = interpolated_surfs[name]
        velocity = velocity_map[name]
        
        if i + 1 < len(sorted_surfaces):
            # It's a layer bounded by two surfaces
            bottom_surf_name, _, _ = sorted_surfaces[i+1]
            bottom_surf_z = interpolated_surfs[bottom_surf_name]
            # Find where the grid Z is below the top surface and above the bottom surface
            mask = (Z <= top_surf_z) & (Z > bottom_surf_z)
        else:
            # It's the last layer, extending downwards
            mask = Z <= top_surf_z
            
        velocity_grid[mask] = velocity

    # Create xarray DataArray
    da = xr.DataArray(
        velocity_grid,
        dims=['z', 'y', 'x'],
        coords={'z': Z[:, 0, 0], 'y': Y[0, :, 0], 'x': X[0, 0, :]},
        name='velocity'
    )
    return da


def build_velocity_model(
    surfaces_directory: Union[str, Path],
    velocity_map: Dict[str, float],
    grid_spacing: Tuple[float, float, float] = (100.0, 100.0, 50.0),
    extent_buffer: float = 500.0,
    manual_extent: Optional[Dict[str, float]] = None,
    input_crs: str = "EPSG:26911",
    output_crs: Optional[str] = "EPSG:4326",
    plot_debug: bool = False
) -> xr.Dataset:
    """
    Main function to build a 3D velocity model from a directory of .ts surface files.

    Args:
        surfaces_directory (Union[str, Path]): Path to the directory with .ts files.
        velocity_map (Dict[str, float]): Maps surface names to the velocity of the unit below.
        grid_spacing (Tuple[float, float, float]): Spacing for the grid (dx, dy, dz).
        extent_buffer (float): Buffer to add to the automatically calculated model extent.
        manual_extent (Optional[Dict[str, float]]): Manually define the model's bounding box.
        input_crs (str): The EPSG code for the source coordinate system (default: "EPSG:26911").
        output_crs (Optional[str]): The EPSG code for the target coordinate system. If provided,
                                     the model will be reprojected (default: "EPSG:4326").
        plot_debug (bool): If True, display an interactive 3D plot of the loaded surfaces.

    Returns:
        xr.Dataset: A dataset containing the 3D velocity model and metadata.
    """
    print("1. Loading and sorting surfaces...")
    sorted_surfaces, auto_extent = load_surfaces_from_directory(surfaces_directory, velocity_map, plot_debug=plot_debug)

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
    z_coords = np.arange(extent['zmax'], extent['zmin'], -dz) # Grid from top to bottom
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Transpose to get z, y, x order
    X, Y, Z = X.transpose(2,1,0), Y.transpose(2,1,0), Z.transpose(2,1,0)

    print("3. Interpolating surfaces and building velocity volume...")
    velocity_da = surfaces_to_velocity_volume(
        sorted_surfaces,
        velocity_map,
        (X, Y, Z)
    )

    print("4. Finalizing dataset...")
    ds = velocity_da.to_dataset()
    
    # Add metadata
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
