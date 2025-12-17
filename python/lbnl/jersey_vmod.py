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
from scipy.ndimage import binary_dilation, generate_binary_structure, gaussian_filter
from matplotlib.tri import Triangulation, LinearTriInterpolator
import datetime
import rioxarray
import plotly.graph_objects as go
from pyproj import Proj, Transformer

try:
    from nllgrid import NLLGrid
except ImportError:
    raise RuntimeError("The `nllgrid` package is not installed. Please install it to use this function.")


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

def get_grid_coords(extent: Dict[str, float], grid_spacing: Tuple[float, float, float]):
    """
    Create 3D grid arrays X,Y,Z used everywhere. Ensure deterministic endpoints
    (include xmax/ymax) so repeated calls produce identical shapes.
    """
    dx, dy, dz = grid_spacing
    nx = int(round((extent['xmax'] - extent['xmin']) / dx)) + 1
    ny = int(round((extent['ymax'] - extent['ymin']) / dy)) + 1
    nz = int(round((extent['zmax'] - extent['zmin']) / dz)) + 1
    x_coords = np.linspace(extent['xmin'], extent['xmax'], nx)
    y_coords = np.linspace(extent['ymin'], extent['ymax'], ny)
    z_coords = np.linspace(extent['zmax'], extent['zmin'], nz)
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    X, Y, Z = X.transpose(2,1,0), Y.transpose(2,1,0), Z.transpose(2,1,0)
    return X, Y, Z

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

def _interpolate_and_stitch_surface(
    verts: pd.DataFrame,
    faces: np.ndarray,
    X_grid_2d: np.ndarray,
    Y_grid_2d: np.ndarray,
    stitching_iterations: int = 3,
    allow_extrapolate: bool = False,
    plot_debug: bool = False
) -> np.ndarray:
    """
    Centralized surface interpolation helper:
      - Linear triangulation interpolation
      - iterative stitching via _stitch_gaps
      - optional nearest-vertex extrapolation (guarded)
    Returns gridded Z (ny, nx) with NaNs preserved unless allow_extrapolate=True.
    """
    pts = verts[['X', 'Y']].values
    vals = verts['Z'].values
    tri = Triangulation(pts[:, 0], pts[:, 1], triangles=faces)
    interp = LinearTriInterpolator(tri, vals)
    flat = interp(X_grid_2d.flatten(), Y_grid_2d.flatten())
    gridz = flat.filled(np.nan).reshape(X_grid_2d.shape)
    gridz = _stitch_gaps(gridz, X_grid_2d, Y_grid_2d, iterations=stitching_iterations)

    # optional nearest-vertex fill for remaining interior NaNs (only when explicitly allowed)
    if allow_extrapolate and np.any(np.isnan(gridz)):
        from scipy.spatial import cKDTree
        nan_idx = np.argwhere(np.isnan(gridz))
        if nan_idx.size:
            tree = cKDTree(pts)
            query_pts = np.column_stack((X_grid_2d[nan_idx[:, 0], nan_idx[:, 1]],
                                         Y_grid_2d[nan_idx[:, 0], nan_idx[:, 1]]))
            _, nn = tree.query(query_pts, k=1)
            gridz[nan_idx[:, 0], nan_idx[:, 1]] = vals[nn]
    elif np.any(np.isnan(gridz)) and plot_debug:
        print(f"{len(np.argwhere(np.isnan(gridz)))} NaNs remain after stitching (extrapolate disabled)")

    return gridz

def surfaces_to_velocity_volume(
    sorted_surfaces: List[Tuple[str, pd.DataFrame, np.ndarray]],
    velocity_map: Dict[str, float],
    grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    fill_velocity_top: float = 500.0,
    plot_debug: bool = False,
    precision_decimals: float = 1,
    stitching_iterations: int = 3,
    smoothing_sigma_m: float = 0.0,
    snap_tol_m: float = 0.0,
    min_thickness_multiplier: float = 2.0,
    vol_thresh_fraction: float = 0.001,
    allow_extrapolate: bool = False,           # if True, fill remaining NaNs from nearest source vertex
    final_fill: bool = True                    # if True, perform final safety fill of remaining NaNs
) -> xr.DataArray:
    """
    Creates a 3D velocity volume using triangulation and controlled gap stitching.
    """
    X, Y, Z = grid_coords
    nz, ny, nx = Z.shape

    # Compute dz from the Z grid (assumes uniform spacing)
    if nz > 1:
        dz = abs(Z[1,0,0] - Z[0,0,0])
    else:
        dz = 1.0  # fallback if only one layer
    
    velocity_grid = np.full(Z.shape, np.nan, dtype=np.float32)
    X_grid_2d, Y_grid_2d = X[0, :, :], Y[0, :, :]

    interpolated_surfs = {}
    for name, verts, faces in sorted_surfaces:
        grid_z_stitched = _interpolate_and_stitch_surface(
            verts, faces, X_grid_2d, Y_grid_2d,
            stitching_iterations=stitching_iterations,
            allow_extrapolate=allow_extrapolate,
            plot_debug=plot_debug
        )
        interpolated_surfs[name] = grid_z_stitched

    # Optional nan-aware Gaussian smoothing of gridded surfaces (disabled by default)
    if smoothing_sigma_m and smoothing_sigma_m > 0.0:
        # estimate grid spacing (assumes uniform)
        dx = float(np.abs(X_grid_2d[0, 1] - X_grid_2d[0, 0]))
        dy = float(np.abs(Y_grid_2d[1, 0] - Y_grid_2d[0, 0]))
        sigma_x = max(0.5, smoothing_sigma_m / dx)
        sigma_y = max(0.5, smoothing_sigma_m / dy)

        def _nan_gaussian_smooth(arr2d, sigma):
            nan_mask = np.isnan(arr2d)
            weights = (~nan_mask).astype(float)
            filled = np.where(nan_mask, 0.0, arr2d)
            num = gaussian_filter(filled, sigma=sigma, mode='nearest')
            den = gaussian_filter(weights, sigma=sigma, mode='nearest')
            with np.errstate(invalid='ignore', divide='ignore'):
                result = num / den
            result[den == 0] = np.nan
            return result

        for name in list(interpolated_surfs.keys()):
            interpolated_surfs[name] = _nan_gaussian_smooth(interpolated_surfs[name], sigma=(sigma_y, sigma_x))
        print(f"Applied nan-aware Gaussian smoothing to surfaces (sigma px = {sigma_x:.2f},{sigma_y:.2f})")

    if plot_debug:
        plot_gridded_surfaces_3d(interpolated_surfs, X_grid_2d, Y_grid_2d)

    surf_names = [name for name, _, _ in sorted_surfaces]
    name_to_global_rank = {name: i for i, name in enumerate(surf_names)}
    all_surfs_z = np.array([interpolated_surfs[name] for name in surf_names])

    # --- CLEANUP: remove tiny interior NaN holes and protect Qa at contacts ---
    from scipy.spatial import cKDTree
    from scipy.ndimage import label as ndi_label

    # 1) Fill very small NaN islands inside each surface using nearest SAME-surface valid cell
    small_hole_thresh = 4  # max pixels to auto-fill (tune: 1-10)
    nsurf = all_surfs_z.shape[0]
    for si in range(nsurf):
        grid = all_surfs_z[si]
        nan_mask = np.isnan(grid)
        if not np.any(nan_mask):
            continue
        labeled, ncomp = ndi_label(nan_mask)
        valid_pts = np.argwhere(~nan_mask)
        if valid_pts.size == 0:
            continue
        tree = cKDTree(valid_pts)
        for comp in range(1, ncomp + 1):
            comp_idx = np.argwhere(labeled == comp)
            if comp_idx.shape[0] <= small_hole_thresh:
                # fill this small hole from nearest same-surface valid cell (index-space)
                _, nn = tree.query(comp_idx, k=1)
                nearest_valid = valid_pts[nn]
                grid[comp_idx[:,0], comp_idx[:,1]] = grid[nearest_valid[:,0], nearest_valid[:,1]]
        all_surfs_z[si] = grid

    # 2) Enforce Qa shallowness: if Qa is present but is deeper than any other SURFACE at that column,
    #    treat Qa as missing there (prevent Qa from undercutting other units).
    if "Qa" in surf_names:
        qa_i = surf_names.index("Qa")
        tol = dz * 0.0  # tolerance
        for yy in range(ny):
            for xx in range(nx):
                qz = all_surfs_z[qa_i, yy, xx]
                if np.isnan(qz):
                    continue
                # max elevation of other valid surfaces at this column
                other = np.nanmax(np.delete(all_surfs_z[:, yy, xx], qa_i))
                if not np.isnan(other) and (qz < other - tol):
                    # Qa is deeper than another surface — ignore Qa here
                    all_surfs_z[qa_i, yy, xx] = np.nan
    # --- end cleanup ---

    # --- Remove tiny Qa surface islands so they can't create vertical shafts ---
    from scipy.ndimage import label as ndi_label
    small_qa_area = 2  # remove Qa patches <= this many pixels (tune: 1-4)
    if "Qa" in surf_names:
        qi = surf_names.index("Qa")
        qa_mask = ~np.isnan(all_surfs_z[qi])
        labeled, ncomp = ndi_label(qa_mask)
        for comp in range(1, ncomp + 1):
            comp_area = np.count_nonzero(labeled == comp)
            if comp_area <= small_qa_area:
                all_surfs_z[qi][labeled == comp] = np.nan
        if plot_debug:
            print(f"Removed {np.count_nonzero(qa_mask & np.isnan(all_surfs_z[qi]))} tiny Qa pixels")
    # --- end tiny-Qa removal ---

    # optional snapping: make nearly-coincident surfaces equal (reduces tiny inverted layers)
    if snap_tol_m and snap_tol_m > 0.0:
        n_surf = all_surfs_z.shape[0]
        for a in range(n_surf - 1):
            for b in range(a + 1, n_surf):
                diff = np.abs(all_surfs_z[a] - all_surfs_z[b])
                close = (~np.isnan(diff)) & (diff <= snap_tol_m)
                if np.any(close):
                    meanvals = np.nanmean(np.stack([all_surfs_z[a][close], all_surfs_z[b][close]]), axis=0)
                    all_surfs_z[a][close] = meanvals
                    all_surfs_z[b][close] = meanvals
        if plot_debug:
            print(f"Snapped surfaces closer than {snap_tol_m} m")

    # determine dz and min thickness
    if nz > 1:
        dz = abs(Z[1,0,0] - Z[0,0,0])
    else:
        dz = 1.0
    min_layer_thickness = max(min_thickness_multiplier * dz, dz)  # safeguard >= dz

    # compute 3D small-component volume threshold
    total_vox = int(nx * ny * nz)
    vol_thresh = max(10, int(round(vol_thresh_fraction * total_vox)))

    # --- Diagnostics: detect columns with inverted or very thin layers and save CSV ---
    small_thickness_mask = np.zeros((ny, nx), dtype=int)
    inverted_mask = np.zeros((ny, nx), dtype=bool)
    for yy in range(ny):
        for xx in range(nx):
            col = all_surfs_z[:, yy, xx]
            valid = ~np.isnan(col)
            nvalid = np.count_nonzero(valid)
            if nvalid < 2:
                continue
            # sort surfaces top (largest z) -> bottom (smallest z)
            zsorted = np.sort(col[valid])[::-1]
            # compute positive thicknesses between adjacent surfaces
            thicknesses = zsorted[:-1] - zsorted[1:]
            # count thin layers (< min thickness) and detect inverted/non-positive thickness
            small_thickness_mask[yy, xx] = np.count_nonzero(thicknesses < min_layer_thickness)
            inverted_mask[yy, xx] = np.any(thicknesses <= 0.0)

    n_inverted = int(np.count_nonzero(inverted_mask))
    n_small = int(np.count_nonzero(small_thickness_mask))
    if n_inverted or n_small:
        rows = []
        for yy in range(ny):
            for xx in range(nx):
                if small_thickness_mask[yy, xx] or inverted_mask[yy, xx]:
                    rows.append({
                        'x_idx': int(xx), 'y_idx': int(yy),
                        'lon': float(X_grid_2d[yy, xx]), 'lat': float(Y_grid_2d[yy, xx]),
                        'thin_count': int(small_thickness_mask[yy, xx]),
                        'inverted': bool(inverted_mask[yy, xx])
                    })
        import pandas as _pd
        _pd.DataFrame(rows).to_csv('surface_diagnostics.csv', index=False)
        print(f"Surface diagnostics: {n_inverted} columns with inverted thicknesses, {n_small} columns with thin layers (<{min_layer_thickness} m). Saved surface_diagnostics.csv")
    # --- end diagnostics ---

    for y_idx in range(ny):
        for x_idx in range(nx):
            column_z_values = all_surfs_z[:, y_idx, x_idx]
            local_surfs = [
                (z_val, name) for z_val, name in zip(column_z_values, surf_names)
                if not np.isnan(z_val)
            ]

            if not local_surfs:
                continue

            # Sort locally by elevation (top -> bottom).  Tie-breaker is the
            # stable Python sort order (deterministic) so columns get true
            # local stratigraphy instead of a global ordering.
            local_surfs.sort(key=lambda item: round(item[0], precision_decimals), reverse=True)

            Z_col = Z[:, y_idx, x_idx]
            top_z, _ = local_surfs[0]
            velocity_grid[:, y_idx, x_idx][Z_col > top_z] = fill_velocity_top

            for i, (top_surf_z, name) in enumerate(local_surfs):
                velocity = velocity_map[name]
                if i + 1 < len(local_surfs):
                    bottom_surf_z, _ = local_surfs[i+1]
                else:
                    bottom_surf_z = -np.inf

                # --- NEW: Only assign if layer is thick enough and not inverted ---
                if (top_surf_z - bottom_surf_z) < min_layer_thickness or top_surf_z <= bottom_surf_z:
                    continue  # Skip zero/negative thickness or inverted layers

                mask = (Z_col <= top_surf_z) & (Z_col > bottom_surf_z)
                velocity_grid[:, y_idx, x_idx][mask] = velocity

    # --- Enforce local cap/floor rules: if Qa present, nothing above Qa; if M present, nothing below M ---
    # This overrides previous assignments for columns where those surfaces exist.
    cap_name = "Qa"
    floor_name = "M"
    cap_idx = surf_names.index(cap_name) if cap_name in surf_names else None
    floor_idx = surf_names.index(floor_name) if floor_name in surf_names else None
    if cap_idx is not None or floor_idx is not None:
        for yy in range(ny):
            for xx in range(nx):
                Z_col = Z[:, yy, xx]
                col_surfs = all_surfs_z[:, yy, xx]
                # enforce Qa cap: always use the explicit top-fill value, not Qa's velocity
                if cap_idx is not None and not np.isnan(col_surfs[cap_idx]):
                    qa_z = float(col_surfs[cap_idx])
                    above_mask = Z_col > qa_z
                    velocity_grid[above_mask, yy, xx] = float(fill_velocity_top)
                # enforce M floor
                if floor_idx is not None and not np.isnan(col_surfs[floor_idx]):
                    m_z = float(col_surfs[floor_idx])
                    below_mask = Z_col <= m_z
                    velocity_grid[below_mask, yy, xx] = float(velocity_map.get(floor_name, velocity_grid[below_mask, yy, xx]))

    # --- Enforce stock-underlies rule ---
    # Stocks (Stock_West / Stock_East) must underlie all other layers where present,
    # except M is allowed to underlie stocks. For each column: below the (shallowest)
    # stock surface, set voxels to stock velocity unless the voxel is below M (then use M).
    stock_names = [n for n in surf_names if n in ("Stock_West", "Stock_East")]
    stock_idxs = [surf_names.index(n) for n in stock_names] if stock_names else []
    m_idx = floor_idx  # M index if present (already computed)
    for yy in range(ny):
        for xx in range(nx):
            if not stock_idxs:
                break
            col_surfs = all_surfs_z[:, yy, xx]
            # collect any stock surface elevations at this column
            stock_zs = [float(col_surfs[i]) for i in stock_idxs if not np.isnan(col_surfs[i])]
            if not stock_zs:
                continue
            # use the shallowest (highest elevation) stock surface as the enforcing level
            stock_top_z = max(stock_zs)
            # get M elevation if present and valid
            m_z = float(col_surfs[m_idx]) if (m_idx is not None and not np.isnan(col_surfs[m_idx])) else None
            # enforce: for voxels at or below stock_top_z, set to stock velocity unless voxel is below M
            stock_vel = None
            # prefer Stock_East if present, otherwise Stock_West (deterministic pick)
            chosen_stock_name = None
            for nm in ("Stock_East", "Stock_West"):
                if nm in surf_names:
                    idx_nm = surf_names.index(nm)
                    if not np.isnan(col_surfs[idx_nm]):
                        chosen_stock_name = nm
                        stock_vel = float(velocity_map.get(nm, np.nan))
                        break
            Z_col = Z[:, yy, xx]
            below_stock_mask = Z_col <= stock_top_z
            if not np.any(below_stock_mask):
                continue
            if stock_vel is None or np.isnan(stock_vel):
                # no valid stock velocity (shouldn't happen) — skip enforcing here
                continue
            if m_z is None:
                # no M at this column: everything below stock_top_z becomes stock
                velocity_grid[below_stock_mask, yy, xx] = stock_vel
            else:
                # if M exists: voxels deeper than or equal to M_z stay M; voxels between stock_top_z and M_z become stock
                below_m_mask = Z_col <= m_z
                # voxels deeper than M keep M (do nothing)
                # voxels between stock_top_z (inclusive) and M_z (exclusive) become stock
                middle_mask = below_stock_mask & (~below_m_mask)
                if np.any(middle_mask):
                    velocity_grid[middle_mask, yy, xx] = stock_vel
    # --- end stock rule ---

    # --- POST-PROCESS: 3D small-component relabeling (surface-index aware, conservative) ---
    from scipy.ndimage import label, binary_dilation, generate_binary_structure, distance_transform_edt
    from scipy.spatial import cKDTree

    # Get the pre-postprocess assignment (surface indices) so relabeling uses surface identity
    src_idx_grid_pre, _, surf_names = compute_assignment_sources(
        sorted_surfaces, (X, Y, Z),
        precision_decimals=precision_decimals,
        stitching_iterations=stitching_iterations,
        allow_extrapolate=False,
        smoothing_sigma_m=smoothing_sigma_m,
        snap_tol_m=snap_tol_m
    )
    # src_idx_grid: -1 unassigned, -2 top_fill, >=0 surface index into surf_names
    src_idx_grid = src_idx_grid_pre
    int_grid = src_idx_grid_pre.copy()  # working copy we will modify during relabel

    # --- define 3D struct and conservative relabel thresholds (was missing) ---
    struct3 = generate_binary_structure(3, 1)  # 6-connectivity in 3D
    horiz_area_thresh = max(3, int(round(min(nx, ny) * 0.01)))  # small horizontal footprint (cells)
    vert_extent_thresh = max(3, 10)  # max vertical extent (in cells) to consider for relabeling; tuneable
    # vol_thresh was computed earlier

    # Qa protections
    qa_idx = surf_names.index("Qa") if "Qa" in surf_names else None
    qa_vel = float(velocity_map.get("Qa", np.nan)) if qa_idx is not None else np.nan

    for cls in np.unique(int_grid):
        if cls < 0:
            continue  # skip unassigned/top_fill
        cls_mask3 = int_grid == cls
        if not np.any(cls_mask3):
            continue
        labeled, ncomp = label(cls_mask3, structure=struct3)
        if ncomp == 0:
            continue
        for comp_id in range(1, ncomp + 1):
            comp_mask = labeled == comp_id
            comp_vol = np.count_nonzero(comp_mask)
            if comp_vol == 0:
                continue
            # horizontal footprint = count of unique (y,x) where any z True
            footprint_area = int(np.count_nonzero(np.any(comp_mask, axis=0)))
            # vertical extent in cells
            z_inds = np.where(np.any(np.any(comp_mask, axis=2), axis=1))[0]  # z indices with any True
            if z_inds.size:
                vert_span = int(z_inds.max() - z_inds.min() + 1)
            else:
                vert_span = 0

            # Conservative criteria: require all three to be small
            if not (comp_vol <= vol_thresh and footprint_area <= horiz_area_thresh and vert_span <= vert_extent_thresh):
                continue  # keep larger/tall components

            # find 3D neighbors and candidate neighbor classes
            nbr = binary_dilation(comp_mask, structure=struct3) & (~comp_mask)
            neighbor_idxs = int_grid[nbr]
            neighbor_idxs = neighbor_idxs[neighbor_idxs >= 0]  # drop unassigned/topfill
            if neighbor_idxs.size == 0:
                continue

            vals, counts = np.unique(neighbor_idxs, return_counts=True)

            # Enforce strict rule: never relabel a component TO Qa if the component was not Qa originally.
            if qa_idx is not None and qa_idx in vals and int(cls) != qa_idx:
                # remove Qa from candidate list
                mask_keep = vals != qa_idx
                vals = vals[mask_keep]
                counts = counts[mask_keep]
                if vals.size == 0:
                    continue  # no valid replacement candidates (skip relabel)

            # Prefer neighbor by local stratigraphic proximity (closest surface elevation at the component XY)
            # compute the XY centroid (use integer grid indices) robustly
            # comp_mask shape: (z,y,x). Project to (y,x) to find footprint indices.
            footprint = np.any(comp_mask, axis=0)  # (y,x)
            ys, xs = np.nonzero(footprint)
            if ys.size:
                yc = int(np.round(ys.mean()))
                xc = int(np.round(xs.mean()))
            else:
                # should not happen, but fall back to the mean of all indices in the mask
                z_inds, y_inds, x_inds = np.nonzero(comp_mask)
                yc = int(np.round(y_inds.mean())) if y_inds.size else 0
                xc = int(np.round(x_inds.mean())) if x_inds.size else 0

            # compute component mean depth directly (safer and faster)
            comp_mean_z = float(np.nanmean(Z[comp_mask]))

            candidate_scores = []
            for v in vals:
                v = int(v)
                # try to get local surface z; if NaN use a large penalty
                try:
                    surf_z = float(all_surfs_z[v, yc, xc])
                except Exception:
                    surf_z = np.nan
                if np.isnan(surf_z):
                    # use modal neighbor fallback score (inverse of count)
                    score = 1e6 + (1.0 / (counts[np.where(vals == v)][0] + 1e-6))
                else:
                    score = abs(comp_mean_z - surf_z)
                candidate_scores.append((v, score))

            # pick candidate with smallest score (closest in depth); tie-break by neighbor support
            candidate_scores = sorted(candidate_scores, key=lambda t: (t[1], -counts[np.where(vals == t[0])[0][0]]))
            chosen_cls = int(candidate_scores[0][0])

            # additional safety: do not pick Qa (should not happen due to removal above), but double-check
            if qa_idx is not None and chosen_cls == qa_idx and int(cls) != qa_idx:
                continue

            # assign chosen class to component (both int_grid and velocity_grid)
            int_grid[comp_mask] = chosen_cls
            chosen_name = surf_names[chosen_cls]
            chosen_vel = float(velocity_map.get(chosen_name, np.nan))
            velocity_grid[comp_mask] = chosen_vel

    # --- continue with column-wise interpolation and lateral NN fill as before ---
    # 1) Column-wise (vertical) fill: interpolate along z within each (x,y) column
    z_idx = np.arange(nz)
    for yy in range(ny):
        for xx in range(nx):
            col = velocity_grid[:, yy, xx]
            good = ~np.isnan(col)
            if not np.any(good):
                continue
            if np.all(good):
                continue
            # nearest-in-column fill: pick nearest valid depth but avoid copying Qa into voxels
            good_idx = np.where(good)[0]
            missing_idx = np.where(~good)[0]
            if good_idx.size == 0:
                continue
            col_interp = col.copy()
            for m in missing_idx:
                # allow Qa only if this voxel was originally Qa
                allow_qa_here = (qa_idx is not None and src_idx_grid[m, yy, xx] == qa_idx)
                # search good_idx in order of distance, pick first that is allowed
                order = np.argsort(np.abs(good_idx - m))
                chosen = None
                for idx in order:
                    cand = good_idx[idx]
                    cand_val = float(col[cand])
                    if np.isnan(cand_val):
                        continue
                    if not (np.isclose(cand_val, qa_vel) and not allow_qa_here):
                        chosen = cand
                        break
                if chosen is not None:
                    col_interp[m] = col[chosen]
                else:
                    # leave as NaN for lateral fill to handle (we refuse to copy Qa here)
                    col_interp[m] = np.nan
            velocity_grid[:, yy, xx] = col_interp
 
    # 2) Lateral fill for any remaining missing voxels using nearest neighbor in 3D
    valid_mask = ~np.isnan(velocity_grid)
    if not np.all(valid_mask):
        # Build KDTree of valid donors excluding Qa (we'll allow Qa donors only for voxels that were originally Qa)
        valid_coords_nonqa = np.argwhere(valid_mask & ~(np.isclose(velocity_grid, qa_vel)))
        valid_coords_all = np.argwhere(valid_mask)

        # Trees may be empty; only build if there are donors
        tree_nonqa = cKDTree(valid_coords_nonqa) if valid_coords_nonqa.size else None
        tree_all = cKDTree(valid_coords_all) if valid_coords_all.size else None

        missing_coords = np.argwhere(~valid_mask)
        for z0, y0, x0 in missing_coords:
            # allow Qa donor only if original preassignment at this voxel was Qa
            allow_qa = (qa_idx is not None and src_idx_grid[z0, y0, x0] == qa_idx)
            chosen_val = np.nan
            if not allow_qa and tree_nonqa is not None:
                # query nearest non-Qa donor
                dist, idx = tree_nonqa.query([z0, y0, x0], k=1)
                donor = tuple(valid_coords_nonqa[int(idx)])
                chosen_val = float(velocity_grid[donor])
            elif allow_qa and tree_all is not None:
                dist, idx = tree_all.query([z0, y0, x0], k=1)
                donor = tuple(valid_coords_all[int(idx)])
                chosen_val = float(velocity_grid[donor])
            # if no donor found under rules, leave NaN
            if not np.isnan(chosen_val):
                velocity_grid[z0, y0, x0] = chosen_val
 
    # final safety fill (optional). If disabled, remaining voxels stay NaN.
    if final_fill and np.isnan(velocity_grid).any():
        nonqa_vals = velocity_grid[~np.isnan(velocity_grid) & ~(np.isclose(velocity_grid, qa_vel))]
        if nonqa_vals.size:
            fallback = float(np.nanmean(nonqa_vals))
        else:
            fallback = float(fill_velocity_top)
        velocity_grid[np.isnan(velocity_grid)] = fallback
    # --- end post-process ---c

    # Final enforcement: remove any Qa values that did NOT originate from Qa preassignment.
    # Replace them with nearest non-Qa donor (or non-Qa fallback).
    if qa_idx is not None:
        # mask of voxels that are Qa now but were not preassigned as Qa
        bad_qa_mask = (np.isclose(velocity_grid, qa_vel)) & (src_idx_grid != qa_idx)
        if np.any(bad_qa_mask):
            donors = np.argwhere(~np.isnan(velocity_grid) & ~(np.isclose(velocity_grid, qa_vel)))
            if donors.size:
                tree = cKDTree(donors)
                targets = np.argwhere(bad_qa_mask)
                dists, idxs = tree.query(targets, k=1)
                donor_coords = donors[idxs]
                for (zt, yt, xt), (zd, yd, xd) in zip(targets, donor_coords):
                    velocity_grid[zt, yt, xt] = velocity_grid[zd, yd, xd]
            else:
                # no non‑Qa donors — use fallback non‑Qa mean or top fill
                nonqa_vals = velocity_grid[~np.isnan(velocity_grid) & ~(np.isclose(velocity_grid, qa_vel))]
                fallback = float(np.nanmean(nonqa_vals)) if nonqa_vals.size else float(fill_velocity_top)
                velocity_grid[bad_qa_mask] = fallback

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
    output_crs: Optional[str] = None,
    plot_debug: bool = False,
    stitching_iterations: int = 3,
    # new params exposed here (defaults match surfaces_to_velocity_volume)
    smoothing_sigma_m: float = 0.0,
    snap_tol_m: float = 0.0,
    min_thickness_multiplier: float = 2.0,
    vol_thresh_fraction: float = 0.001,
    vp_vs_ratio: float = 1.66
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
    X, Y, Z = get_grid_coords(extent, grid_spacing)

    print("3. Interpolating surfaces and building velocity volume...")
    velocity_da = surfaces_to_velocity_volume(
        sorted_surfaces,
        velocity_map,
        (X, Y, Z),
        plot_debug=plot_debug,
        precision_decimals=1,
        stitching_iterations=stitching_iterations,
        smoothing_sigma_m=smoothing_sigma_m,
        snap_tol_m=snap_tol_m,
        min_thickness_multiplier=min_thickness_multiplier,
        vol_thresh_fraction=vol_thresh_fraction
    )

    print("4. Finalizing dataset...")
    ds = velocity_da.to_dataset()
    # Add Vp as explicit variable (alias for 'velocity') and compute Vs with constant Vp/Vs ratio
    ds['Vp'] = ds['velocity'].copy()
    ds['Vp'].attrs['units'] = 'm/s'
    ds['Vp'].attrs['description'] = 'P-wave velocity (alias of velocity)'
    ds.attrs['vp_vs_ratio'] = float(vp_vs_ratio)
    # compute Vs (avoid division by zero)
    with np.errstate(invalid='ignore', divide='ignore'):
        ds['Vs'] = ds['Vp'] / float(vp_vs_ratio)
    ds['Vs'].attrs['units'] = 'm/s'
    ds['Vs'].attrs['description'] = f"S-wave velocity computed from Vp using constant Vp/Vs ratio = {vp_vs_ratio}"

    ds.attrs['created_at'] = datetime.datetime.utcnow().isoformat()
    ds.attrs['source_directory'] = str(Path(surfaces_directory).resolve())
    ds.attrs['grid_spacing_xyz'] = str(grid_spacing)
    ds.attrs['stratigraphic_order'] = [name for name, _, _ in sorted_surfaces]
    ds.attrs['velocity_map'] = str(velocity_map)

    # --- MOVE PLOTTING HERE (before reprojection) ---
    # if plot_debug:
    #     plot_velocity_model_3d(ds['velocity'], isosurface_levels=[2000, 4000])

    print("5. Reprojecting coordinates (if specified)...")
    ds.rio.write_crs(input_crs, inplace=True)

    if output_crs and input_crs.lower() != output_crs.lower():
        ds = ds.rio.reproject(output_crs)
        print(f"Dataset reprojected to {output_crs}")
        ds.attrs['crs'] = output_crs
    else:
        ds.attrs['crs'] = input_crs
        print("No reprojection needed.")

    # Add lat/lon as additional 2D coords (keep x/y as native grid coordinates)
    try:
        # transformer: native grid CRS -> geographic (lon/lat)
        transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
        # X, Y are shape (z, y, x); take the surface-level 2D grid used elsewhere
        X2d = X[0, :, :]  # easting / native x
        Y2d = Y[0, :, :]  # northing / native y
        lon2d, lat2d = transformer.transform(X2d, Y2d)
        ds = ds.assign_coords({
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d)
        })
        ds.attrs['native_crs'] = input_crs
        ds.attrs['lonlat_crs'] = "EPSG:4326"
    except Exception as e:
        print(f"Warning: failed to compute lon/lat coords: {e}")

    print("Done.")
    return ds


def plot_velocity_model_3d(
    velocity_da: xr.DataArray,
    colormap: str = 'viridis',
    opacity: float = 0.2,
    isosurface_levels: Optional[List[float]] = None,
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
    
    # Create 3D coordinate arrays for PyVista StructuredGrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = X.transpose(2, 1, 0)  # Match dims ['z', 'y', 'x']
    Y = Y.transpose(2, 1, 0)
    Z = Z.transpose(2, 1, 0)
    
    # Create PyVista StructuredGrid
    grid = pv.StructuredGrid(X, Y, Z)
    grid['velocity'] = velocity.flatten(order='F')  # Match dims ['z', 'y', 'x']
    
    # Plot the whole cube with volume rendering, colored by velocity
    plotter = pv.Plotter()
    vol = plotter.add_volume(grid, scalars='velocity', cmap=colormap, opacity=[0, 0.01, 0.01, 0, 0])  # Direct low-opacity transfer function
    plotter.add_scalar_bar(title="Velocity (m/s)", vertical=True)
    
    # Add isosurfaces if requested
    if isosurface_levels:
        for level in isosurface_levels:
            if velocity.min() <= level <= velocity.max():
                isosurface = grid.contour([level], scalars='velocity')
                plotter.add_mesh(isosurface, color='black', opacity=0.7)
    
    # Add three interactive slicers (one for each axis)
    def slice_callback(normal, origin):
        slice_mesh = grid.slice(normal=normal, origin=origin)
        plotter.clear()  # Clear previous renders
        vol = plotter.add_volume(grid, scalars='velocity', cmap=colormap, opacity=[0, 0.01, 0.01, 0, 0])  # Same low opacity
        plotter.add_scalar_bar(title="Velocity (m/s)", vertical=True)
        if isosurface_levels:
            for level in isosurface_levels:
                if velocity.min() <= level <= velocity.max():
                    isosurface = grid.contour([level], scalars='velocity')
                    plotter.add_mesh(isosurface, color='black', opacity=0.7)
        if slice_mesh.n_points > 0:  # Avoid plotting empty meshes
            plotter.add_mesh(slice_mesh, scalars='velocity', cmap=colormap, show_scalar_bar=False)
        plotter.add_axes(
            xlabel=f'X (scaled, offset {x_offset:.0f})',
            ylabel=f'Y (scaled, offset {y_offset:.0f})',
            zlabel=f'Z (scaled, offset {z_offset:.0f})'
        )
    
    for normal in ['x', 'y', 'z']:
        plotter.add_plane_widget(slice_callback, normal=normal, implicit=False)
    
    # Add axes
    plotter.add_axes(
        xlabel=f'X (scaled, offset {x_offset:.0f})',
        ylabel=f'Y (scaled, offset {y_offset:.0f})',
        zlabel=f'Z (scaled, offset {z_offset:.0f})'
    )
    
    # Enable turntable-style rotation  # Removed to allow widgets
    # plotter.enable_joystick_actor_style()
    
    plotter.show()
    return plotter


def compute_assignment_sources(
    sorted_surfaces: List[Tuple[str, pd.DataFrame, np.ndarray]],
    grid_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    precision_decimals: int = 1,
    stitching_iterations: int = 3,
    allow_extrapolate: bool = False,
    smoothing_sigma_m: float = 0.0,
    snap_tol_m: float = 0.0
):
    """
    Reproduce the per-column assignment (before post-processing) and return:
      - src_idx_grid: int array (z,y,x) where value = surface index used for that voxel
        (-1 = unassigned, -2 = top fill)
      - vel_grid: float array (z,y,x) velocities assigned by that pass
      - surf_names: list of surface names (index -> name)
    This is cheap relative to full post-processing and useful for debugging which
    surface provided the voxel assignment.
    """
    from scipy.ndimage import label as ndi_label  # <-- add this import to avoid NameError

    X, Y, Z = grid_coords
    nz, ny, nx = Z.shape

    # interpolate surfaces same as surfaces_to_velocity_volume
    X_grid_2d, Y_grid_2d = X[0, :, :], Y[0, :, :]
    interp_surfs = []
    for name, verts, faces in sorted_surfaces:
        gridz = _interpolate_and_stitch_surface(
            verts, faces, X_grid_2d, Y_grid_2d,
            stitching_iterations=stitching_iterations,
            allow_extrapolate=allow_extrapolate,
            plot_debug=False
        )
        interp_surfs.append(gridz)

    surf_names = [name for name, _, _ in sorted_surfaces]
    all_surfs_z = np.array(interp_surfs)  # (nsurf, ny, nx)

    # Apply the same nan-aware Gaussian smoothing used by surfaces_to_velocity_volume
    if smoothing_sigma_m and smoothing_sigma_m > 0.0:
        dx = float(np.abs(X_grid_2d[0, 1] - X_grid_2d[0, 0]))
        dy = float(np.abs(Y_grid_2d[1, 0] - Y_grid_2d[0, 0]))
        sigma_x = max(0.5, smoothing_sigma_m / dx)
        sigma_y = max(0.5, smoothing_sigma_m / dy)

        def _nan_gaussian_smooth_local(arr2d, sigma):
            nan_mask = np.isnan(arr2d)
            weights = (~nan_mask).astype(float)
            filled = np.where(nan_mask, 0.0, arr2d)
            num = gaussian_filter(filled, sigma=sigma, mode='nearest')
            den = gaussian_filter(weights, sigma=sigma, mode='nearest')
            with np.errstate(invalid='ignore', divide='ignore'):
                result = num / den
            result[den == 0] = np.nan
            return result

        for i in range(len(interp_surfs)):
            interp_surfs[i] = _nan_gaussian_smooth_local(interp_surfs[i], sigma=(sigma_y, sigma_x))

    # Apply the same snap_tol_m surface snapping as final pipeline (avoid small inversions)
    if snap_tol_m and snap_tol_m > 0.0:
        surf_array = np.array(interp_surfs)
        n_surf = surf_array.shape[0]
        for a in range(n_surf - 1):
            for b in range(a + 1, n_surf):
                diff = np.abs(surf_array[a] - surf_array[b])
                close = (~np.isnan(diff)) & (diff <= snap_tol_m)
                if np.any(close):
                    meanvals = np.nanmean(np.stack([surf_array[a][close], surf_array[b][close]]), axis=0)
                    surf_array[a][close] = meanvals
                    surf_array[b][close] = meanvals
        interp_surfs = [surf_array[i] for i in range(surf_array.shape[0])]

    name_to_global_rank = {name: i for i, name in enumerate(surf_names)}
    all_surfs_z = np.array(interp_surfs)  # (nsurf, ny, nx)

    src_idx = -np.ones(Z.shape, dtype=int)   # -1 = unassigned
    vel = np.full(Z.shape, np.nan, dtype=float)

    # helper to map surface index -> velocity (requires velocity_map ordering)
    # user must map names externally if needed

    # build a name->index map from sorted_surfaces
    name_to_idx = {name: i for i, (name, _, _) in enumerate(sorted_surfaces)}

    for y_idx in range(ny):
        for x_idx in range(nx):
            column_z_values = all_surfs_z[:, y_idx, x_idx]
            local = [(zv, name) for zv, name in zip(column_z_values, surf_names) if not np.isnan(zv)]
            if not local:
                continue
            local.sort(key=lambda item: round(item[0], precision_decimals), reverse=True)
            Z_col = Z[:, y_idx, x_idx]
            top_z, _ = local[0]
            # mark above top fill (use index -2)
            above_mask = Z_col > top_z
            src_idx[above_mask, y_idx, x_idx] = -2
            vel[above_mask, y_idx, x_idx] = np.nan  # caller can interpret top fill separately

            for i, (top_surf_z, name) in enumerate(local):
                if i+1 < len(local):
                    bottom_surf_z, _ = local[i+1]
                else:
                    bottom_surf_z = -np.inf
                mask = (Z_col <= top_surf_z) & (Z_col > bottom_surf_z)
                if mask.any():
                    sid = name_to_idx[name]
                    src_idx[mask, y_idx, x_idx] = sid
                    # velocity_map not imported here; caller can map sid->velocity as needed
    # Remove tiny Qa columns from preassignment so "pre" matches final behavior
    if "Qa" in surf_names:
        qi = surf_names.index("Qa")
        # presence mask of Qa columns (any depth)
        qa_cols = np.any(src_idx == qi, axis=0)
        labeled, ncomp = ndi_label(qa_cols)
        small_qa_area = 2  # same threshold as surfaces_to_velocity_volume
        for comp in range(1, ncomp + 1):
            comp_area = np.count_nonzero(labeled == comp)
            if comp_area <= small_qa_area:
                # clear all voxels in these columns
                src_idx[:, labeled == comp] = -1
                vel[:, labeled == comp] = np.nan
        if True:  # optional debug
            removed = np.count_nonzero(qa_cols & (src_idx[0] != qi))
            # print small debug only when needed
            # print(f"compute_assignment_sources: removed {removed} tiny Qa columns")
    return src_idx, vel, surf_names


def compare_prepost_for_location(
    surfaces_dir: Union[str, Path],
    velocity_map: Dict[str, float],
    grid_spacing: Tuple[float, float, float],
    lon: float,
    lat: float,
    stitching_iterations: int = 3,
    # forwarded kwargs to reproduce the final model exactly
    smoothing_sigma_m: float = 0.0,
    snap_tol_m: float = 0.0,
    min_thickness_multiplier: float = 1.0,
    vol_thresh_fraction: float = 0.001,
    fill_velocity_top: float = 500.0,
    query_crs: Optional[str] = None,
    surfaces_crs: Optional[str] = None,
    show_df: bool = True
):
    """
    Rebuild pre-assignment (compute_assignment_sources) and the final velocity volume
    (surfaces_to_velocity_volume) and print/return a per-depth table for the nearest
    grid cell to (lon,lat). Use the same parameters you used to build the model.
    """
    # load surfaces and extent
    sorted_surfaces, extent = load_surfaces_from_directory(surfaces_dir, velocity_map)
    X, Y, Z = get_grid_coords(extent, grid_spacing)
    X2d = X[0,:,:]; Y2d = Y[0,:,:]
    ny, nx = X2d.shape

    # Determine native CRS
    native_crs = surfaces_crs if surfaces_crs is not None else "EPSG:26911"

    # Heuristic: if query_crs not given, treat large numeric inputs as UTM/native CRS
    if query_crs is None:
        if abs(lon) > 10000 or abs(lat) > 10000:
            query_crs = native_crs
        else:
            query_crs = "EPSG:4326"

    # transform query to native CRS if needed
    try:
        if query_crs != native_crs:
            tf = Transformer.from_crs(query_crs, native_crs, always_xy=True)
            qx, qy = tf.transform(lon, lat)
        else:
            qx, qy = lon, lat
    except Exception:
        qx, qy = lon, lat

    yy, xx = np.unravel_index(np.argmin((X2d - qx)**2 + (Y2d - qy)**2), X2d.shape)

    # 1) pre-postprocess sources
    src_idx, vel_pre_dummy, surf_names = compute_assignment_sources(
        sorted_surfaces, (X, Y, Z),
        precision_decimals=1,
        stitching_iterations=stitching_iterations,
        allow_extrapolate=False,
        smoothing_sigma_m=smoothing_sigma_m,
        snap_tol_m=snap_tol_m
    )

    # Map src_idx -> name/velocity
    def src_name(v):
        if v == -2:
            return "TOP_FILL"
        if v == -1:
            return None
        return surf_names[int(v)]
    def src_velocity(v):
        if v == -2:
            return float(fill_velocity_top)
        if v == -1:
            return np.nan
        name = surf_names[int(v)]
        return float(velocity_map.get(name, np.nan))

    src_col_idx = src_idx[:, yy, xx]
    src_col_names = [src_name(int(v)) for v in src_col_idx]
    src_col_vels = [src_velocity(int(v)) for v in src_col_idx]

    # 2) final velocity volume (recompute with same params)
    vel_da = surfaces_to_velocity_volume(
        sorted_surfaces,
        velocity_map,
        (X, Y, Z),
        fill_velocity_top=fill_velocity_top,
        plot_debug=False,
        precision_decimals=1,
        stitching_iterations=stitching_iterations,
        smoothing_sigma_m=smoothing_sigma_m,
        snap_tol_m=snap_tol_m,
        min_thickness_multiplier=min_thickness_multiplier,
        vol_thresh_fraction=vol_thresh_fraction
    )
    final_col = vel_da.values[:, yy, xx].astype(float)
    z_coords_out = Z[:, yy, xx].astype(float)

    # build class-value mapping used in postprocess for quick "final source guess"
    class_values = np.unique(list(velocity_map.values()) + [fill_velocity_top])
    # for mapping final velocity to closest class value
    def guess_class_name_from_velocity(v):
        if np.isnan(v):
            return None
        diffs = np.abs(class_values - v)
        idx = int(np.argmin(diffs))
        val = float(class_values[idx])
        # find corresponding surface name(s) for this velocity (may be multiple)
        names = [n for n, vv in velocity_map.items() if float(vv) == val]
        if val == float(fill_velocity_top):
            return "TOP_FILL"
        return ",".join(names) if names else f"vel_{val}"

    guessed_final_names = [guess_class_name_from_velocity(v) for v in final_col]

    # compare and assemble DataFrame
    import pandas as _pd
    rows = []
    for k in range(len(z_coords_out)):
        rows.append({
            'z': float(z_coords_out[k]),
            'pre_source': src_col_names[k],
            'pre_vel': src_col_vels[k],
            'final_vel': float(final_col[k]),
            'final_guess_source': guessed_final_names[k],
            'changed': (not (np.isclose(src_col_vels[k], final_col[k], equal_nan=True)))
        })
    df = _pd.DataFrame(rows)
    df = df.sort_values('z', ascending=False).reset_index(drop=True)

    if show_df:
        print(f"Debug comparison at nearest grid cell x={xx}, y={yy} (grid lon={X2d[yy,xx]:.6f}, lat={Y2d[yy,xx]:.6f})")
        print(df.to_string(index=False, float_format='{:.3f}'.format))

    # summarize where changes happened
    n_changed = int(df['changed'].sum())
    print(f"\nTotal depths changed by postprocessing or fill: {n_changed} / {len(df)}")
    if n_changed:
        # show first few changed rows
        print("\nFirst changed rows (top->bottom):")
        print(df[df['changed']].head(10).to_string(index=False, float_format='{:.3f}'.format))

    return {
        "grid_x": int(xx), "grid_y": int(yy),
        "grid_lon": float(X2d[yy,xx]), "grid_lat": float(Y2d[yy,xx]),
        "comparison_df": df,
        "pre_src_idx_col": src_col_idx.tolist(),
        "final_col": final_col.tolist()
    }


def write_nlloc_grid(
    ds: xr.Dataset,
    path: Union[str, Path],
    var: str = "Vp",
    *,
    grid_type: str = "SLOW_LEN",
    proj_name: str = "SIMPLE",
    float_type: str = "FLOAT",
    overwrite: bool = True
):
    """
    Write dataset to an NLLoc grid file using the `nllgrid` package.

    Parameters:
    - ds: xarray Dataset containing coordinate dims ['x', 'y', 'z'] and variable `var`.
    - path: output basename for the NLLoc grid files (e.g., 'mygrid' for 'mygrid.hdr' and 'mygrid.buf').
    - var: variable name in the dataset to write (default 'Vp').
    - grid_type: type of the grid (default 'SLOW_LEN').
    - proj_name: projection name (default 'SIMPLE').
    - float_type: precision of the grid ('FLOAT' or 'DOUBLE', default 'FLOAT').
    - overwrite: if False, raises an error if the files already exist.

    Raises:
    - ValueError: if grid spacing (dx, dy, dz) is not uniform.
    - RuntimeError: if the `nllgrid` package is not installed or its API is unavailable.
    - KeyError: if the specified variable or required coordinates are missing in the dataset.
    """
    path = Path(path)
    hdr_path = path.with_suffix(".hdr")
    buf_path = path.with_suffix(".buf")

    if not overwrite and (hdr_path.exists() or buf_path.exists()):
        raise FileExistsError(f"Files already exist and overwrite=False: {hdr_path}, {buf_path}")

    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset")

    # Extract coordinates and data
    x = np.asarray(ds.coords["x"].values, dtype=float)
    y = np.asarray(ds.coords["y"].values, dtype=float)
    z = np.asarray(ds.coords["z"].values, dtype=float) * -1
    values = ds[var].values

    # Ensure uniform grid spacing
    dx = np.unique(np.diff(x))
    dy = np.unique(np.diff(y))
    dz = np.abs(np.unique(np.diff(z)))

    # if len(dx) != 1 or len(dy) != 1 or len(dz) != 1 or not np.isclose(dx[0], dy[0], atol=1e-3) or not np.isclose(dx[0], dz[0], atol=1e-3):
    #     raise ValueError(f"Grid spacing must be uniform and equal in all dimensions (dx={dx}, dy={dy}, dz={dz}).")

    dx, dy, dz = dx[0], dy[0], dz[0]  # Extract the single spacing value

    # Ensure data ordering matches NLLoc expectations (x, y, z)
    values = ds[var].transpose("x", "y", "z").values

    # Convert velocity to slowness*length (SLOW_LEN)
    if grid_type == "SLOW_LEN":
        values = dz / values  # Convert to slowness*length (dz is in km)

    # Reproject origin into Lat/Lon
    transformer = Transformer.from_crs("EPSG:26911", "EPSG:4326", always_xy=True)  # Example: UTM Zone 11N to WGS84
    origin_x, origin_y = x[0], y[0]
    origin_lon, origin_lat = transformer.transform(origin_x, origin_y)

    # Create an NLLoc grid object
    grid = NLLGrid()
    grid.array = values  # No flipping, as the order is already x, y, z
    grid.dx = round(dx, 0) * 1e-3
    grid.dy = round(dy, 0) * 1e-3
    grid.dz = round(dz, 0) * 1e-3
    grid.x_orig = 0.0  # Origin in grid coordinates
    grid.y_orig = 0.0
    grid.z_orig = z.min() / 1000.0  # Convert to kilometers
    grid.type = grid_type
    grid.float_type = float_type
    grid.orig_lat = origin_lat
    grid.orig_lon = origin_lon
    grid.proj_name = proj_name
    grid.basename = str(path)

    # Write the grid files
    try:
        grid.write_hdr_file()
        grid.write_buf_file()
        print(f"Successfully wrote NLLoc grid to: {hdr_path} and {buf_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to write NLLoc grid files: {e}")

def smooth_dataset(
    ds: xr.Dataset,
    var: str = "Vp",
    sigma: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    inplace: bool = False
) -> xr.Dataset:
    """
    Apply Gaussian smoothing to a variable in the dataset.

    Parameters:
    - ds: xarray Dataset containing the variable to smooth.
    - var: Name of the variable to smooth (default 'Vp').
    - sigma: Standard deviation for Gaussian kernel in grid units (z, y, x).
    - inplace: If True, modify the dataset in place. Otherwise, return a new dataset.

    Returns:
    - Smoothed xarray Dataset.
    """
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in the dataset.")

    # Extract the variable to smooth
    data = ds[var].values

    # Apply Gaussian smoothing
    smoothed_data = gaussian_filter(data, sigma=sigma, mode="nearest")

    # Update the dataset
    if inplace:
        ds[var].values = smoothed_data
        return ds
    else:
        smoothed_ds = ds.copy()
        smoothed_ds[var].values = smoothed_data
        return smoothed_ds
