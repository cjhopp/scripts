import dash
import numpy as np
import pandas as pd
import xarray as xr
import pyproj

import plotly.graph_objs as go

from dash import dcc, html
from obspy import read_events
from osgeo import gdal
from datetime import datetime
from shapely.geometry import Polygon

# Set your local data directory here
data_directory = '/media/chopp/HDD1/chet-meq'

site_polygons = {
    'Newberry': Polygon([(-121.0736, 43.8988), (-121.0736, 43.5949), (-121.4918, 43.5949), (-121.4918, 43.8988)]),
    'JV': Polygon([(-117.40, 40.2357), (-117.5692, 40.2357), (-117.5692, 40.107), (-117.40, 40.107)]),
    'DAC': Polygon([(-118.1979, 38.9604), (-118.1979, 38.7943), (-118.4046, 38.7943), (-118.4046, 9604)]),
    'TM': Polygon([(-117.5956, 39.7353), (-117.5956, 39.6056), (-117.7649, 39.6056), (-117.7649, 39.7353)]),
    'Cape': Polygon([(-112.6924, 38.3912), (-112.6924, 38.6512), (-113.1358, 38.6512), (-113.1358, 38.3912)])
}

datasets = {
    'Newberry': [
        f'{data_directory}/newberry/vector/boreholes/Deviation_corrected.csv',
        f'{data_directory}/newberry/DEM/USGS_13_merged_epsg-26910_just_edifice_very-coarse.tif'
    ],
    'JV': [
        f'{data_directory}/JV/vector/boreholes/Offset_Wells_Surveys_JV.csv',
    ],
    'DAC': [
        f'{data_directory}/DAC/vector/boreholes/Offset_Wells_Surveys_DAC.csv'
    ],
    'Cape': {
        'Topography': f'{data_directory}/cape_modern/spatial_data/DEM/Cape-modern_Lidar_downsample.tif',
        'Basement': f'{data_directory}/cape_modern/spatial_data/vmods/ToB_50m_grid_3-1-24.nc',
        'Frisco-1': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Frisco-1_trajectory.csv',
        'Frisco-2': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Frisco-2_trajectory.csv',
        'Frisco-3': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Frisco-3_trajectory.csv',
        'Frisco-4': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Frisco-4_trajectory.csv',
        'Bearskin-1IA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_1IA_xyz.csv',
        'Bearskin-2IB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_2IB_xyz.csv',
        'Bearskin-4PB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_4PB_xyz.csv',
        'Bearskin-6IB': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_6IB_xyz.csv',
        'Bearskin-7PA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_7PA_xyz.csv',
        'Bearskin-8IA': f'{data_directory}/cape_modern/spatial_data/vector/boreholes/Bearskin_8IA_xyz.csv'
    }
}

projections = {'cape': pyproj.Proj("EPSG:26912"),
               'newberry': pyproj.Proj("EPSG:32610"),
               'JV': pyproj.Proj("EPSG:32611"),
               'DAC': pyproj.Proj("EPSG:26911"),}


color_dict = {
    'JV': {
        ('14-34'): 'black',
        ('18A-27', '46-28', '14-27', '81-28', '81A-28'): 'steelblue',
        ('86-28', '87-28', '77A-28'): 'firebrick',
    },
    'DAC': {
        ('68-1RD'): 'black',
        ('24-6', '24A-6', '26-6', '26A-6', '36-6', '24-6', '24A-6'): 'steelblue',
        ('64-11', '64A-11', '64B-11', '64C-11', '65-11', '65A-11', '85-11', '85A-11', '54-11', '54A-11'): 'firebrick',
    },
}

depth_correction = {
    'JV': 1446.,
    'DAC': 1286.,
}

catalog_colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

def get_pixel_coords(dataset):
    band = dataset.GetRasterBand(1)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    transform = dataset.GetGeoTransform()
    xo = transform[0]
    yo = transform[3]
    pixw = transform[1]
    pixh = transform[5]
    return (np.arange(cols) * pixw) + xo, (np.arange(rows) * pixh) + yo, band

def plot_datasets_3d(datasets):
    objects = []
    for label, data in datasets.items():
        if not data.endswith(('tif', 'nc')):
            wellpath = np.loadtxt(data, delimiter=',', skiprows=1)
            east = wellpath[:, 0]
            north = wellpath[:, 1]
            dep_m = wellpath[:, 2]
            objects.append(go.Scatter3d(
                x=east, y=north, z=dep_m,
                name=label,
                mode='lines',
                line=dict(color='black', width=6),
                hoverinfo='skip'
            ))
        elif data.endswith('tif'):
            topo = gdal.Open(data, gdal.GA_ReadOnly)
            x, y, band = get_pixel_coords(topo)
            X, Y = np.meshgrid(x, y, indexing='xy')
            raster_values = band.ReadAsArray()
            topo_mesh = go.Mesh3d(
                x=X.flatten(), y=Y.flatten(), z=raster_values.flatten(),
                name=label, color='gray', opacity=0.3, delaunayaxis='z', showlegend=True,
                hoverinfo='skip'
            )
            objects.append(topo_mesh)
        elif data.endswith('nc'):
            tob = xr.load_dataarray(data)
            tob = tob.interp(easting=tob.easting[::10], northing=tob.northing[::10])
            X, Y = np.meshgrid(tob.easting, tob.northing, indexing='xy')
            Z = tob.values.flatten()
            tob_mesh = go.Mesh3d(
                x=X.flatten(), y=Y.flatten(), z=Z,
                name=label, color='gray', opacity=0.5, delaunayaxis='z', showlegend=True,
                hoverinfo='skip'
            )
            objects.append(tob_mesh)
    return objects

def get_catalog_params(catalog, utm):
    params = []
    for ev in catalog:
        o = ev.preferred_origin()
        try:
            m = ev.preferred_magnitude().mag
        except Exception:
            m = 0.5
        params.append([ev.resource_id.id, o.time.timestamp, o.latitude, o.longitude, o.depth, m])
    params = np.array(params)
    if len(params) == 0:
        return None
    id, t, lat, lon, depth, m = np.split(params, 6, axis=1)
    t = t.astype('f').flatten()
    lat = lat.astype('f').flatten()
    lon = lon.astype('f').flatten()
    depth = depth.astype('f').flatten()
    m = m.astype('f').flatten()
    ev_east, ev_north = utm(lon, lat)
    depth = np.array(depth) * -1
    return id, t, lat, lon, depth, m, ev_east, ev_north

def make_3d_figure(catalogs, catalog_names, datasets, field='cape', scale_by_magnitude=False, color_by_time=False):
    objects = plot_datasets_3d(datasets)
    mfact = 2.5
    utm = projections[field]
    for i, catalog in enumerate(catalogs):
        result = get_catalog_params(catalog, utm)
        if result is None:
            continue
        id, t, lat, lon, depth, m, ev_east, ev_north = result
        tickvals = np.linspace(min(t), max(t), 10)
        ticktext = [datetime.fromtimestamp(int(tv)).strftime('%d %b %Y: %H:%M') for tv in tickvals]
        if scale_by_magnitude:
            marker_size = (mfact * np.array(m)) ** 2
        else:
            marker_size = np.full_like(m, 2.)
        if color_by_time:
            marker_color = t
            marker_dict = dict(
                color=marker_color,
                cmin=min(tickvals),
                cmax=max(tickvals),
                size=marker_size,
                symbol='circle',
                line=dict(color=marker_color, width=1, colorscale='Cividis'),
                colorbar=dict(
                    title=dict(text='Timestamp', font=dict(size=18)),
                    x=-0.2,
                    ticktext=ticktext,
                    tickvals=tickvals
                ),
                colorscale='Bluered',
                opacity=0.5
            )
        else:
            marker_color = catalog_colors[i % len(catalog_colors)]
            marker_dict = dict(
                color=marker_color,
                size=marker_size,
                symbol='circle',
                line=dict(color=marker_color, width=1),
                opacity=0.5
            )
        scat_obj = go.Scatter3d(
            x=ev_east, y=ev_north, z=depth,
            mode='markers',
            name=catalog_names[i],
            hoverinfo='text',
            text=np.array(id),
            marker=marker_dict
        )
        objects.append(scat_obj)
    fig = go.Figure(data=objects)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Easting (m)'),
            yaxis=dict(title='Northing (m)'),
            zaxis=dict(title='Elevation (m)'),
            aspectmode='data',
            bgcolor="rgb(244, 244, 248)"
        ),
        title='3D Seismicity',
        legend=dict(itemsizing='constant', bgcolor='whitesmoke', bordercolor='gray', borderwidth=1),
        height=900  # Double the height
    )
    return fig

def make_cumulative_figure(catalogs, catalog_names):
    curves = []
    pick_curves = []
    for i, catalog in enumerate(catalogs):
        times = [ev.preferred_origin().time.datetime for ev in catalog]
        times = sorted(times)
        if not times:
            continue
        df = pd.DataFrame({'time': times, 'count': np.arange(1, len(times)+1)})
        curves.append(go.Scatter(
            x=df['time'], y=df['count'],
            mode='lines+markers',
            name=f'{catalog_names[i]} Events',
            yaxis='y1',
            line=dict(color=catalog_colors[i % len(catalog_colors)])
        ))
        # Cumulative picks
        pick_times = []
        for ev in catalog:
            for arr in ev.origins[0].arrivals:
                try:
                    pick = arr.pick_id.get_referred_object()
                    pick_times.append(pick.time.datetime)
                except Exception:
                    continue
        pick_times = sorted(pick_times)
        if pick_times:
            dfp = pd.DataFrame({'time': pick_times, 'count': np.arange(1, len(pick_times)+1)})
            pick_curves.append(go.Scatter(
                x=dfp['time'], y=dfp['count'],
                mode='lines',
                name=f'{catalog_names[i]} Picks',
                yaxis='y2',
                line=dict(dash='dot', color=catalog_colors[i % len(catalog_colors)])
            ))
    fig = go.Figure(data=curves + pick_curves)
    fig.update_layout(
        title='Cumulative Number of Events and Picks',
        xaxis_title='Time',
        yaxis=dict(title='Cumulative Events'),
        yaxis2=dict(title='Cumulative Picks', overlaying='y', side='right', showgrid=False),
        legend=dict(itemsizing='constant', bgcolor='whitesmoke', bordercolor='gray', borderwidth=1),
        height=450
    )
    return fig

def make_arrivals_histogram(catalogs, catalog_names):
    hists = []
    for i, catalog in enumerate(catalogs):
        seed_ids = []
        for ev in catalog:
            for arr in ev.origins[0].arrivals:
                try:
                    pick = arr.pick_id.get_referred_object()
                    seed_ids.append(pick.waveform_id.get_seed_string())
                except Exception:
                    continue
        if not seed_ids:
            continue
        s, counts = np.unique(seed_ids, return_counts=True)
        hists.append(go.Bar(
            x=s, y=counts,
            name=catalog_names[i],
            marker=dict(color=catalog_colors[i % len(catalog_colors)])
        ))
    fig = go.Figure(data=hists)
    fig.update_layout(
        title='Histogram of Arrivals by SEED ID',
        xaxis_title='SEED ID',
        yaxis_title='Arrivals',
        barmode='group',
        legend=dict(itemsizing='constant', bgcolor='whitesmoke', bordercolor='gray', borderwidth=1),
        height=450
    )
    return fig

# --- MAIN DASH APP ---

# Specify your local catalog files here
catalog_paths = [
    # '/media/chopp/HDD1/chet-meq/cape_modern/seiscomp_output/dlpick_testing/test_1-day/final_relocations_autopicks_qml.xml',
    # '/media/chopp/HDD1/chet-meq/cape_modern/seiscomp_output/dlpick_testing/test_1-day/final_relocations_qml.xml',
    # Add more as needed
    '/home/chopp/Cape_noise.xml',
    '/home/chopp/Cape_events.xml',
]
catalog_names = ['Catalog 1', 'Catalog 2']  # Match order to catalog_paths
field = 'cape'
datas = datasets['Cape']

catalogs = [read_events(path) for path in catalog_paths]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Seismic Catalog Comparison Dashboard"),
    html.Label("Scale 3D markers by magnitude:"),
    dcc.RadioItems(
        id='scale-mag-toggle',
        options=[
            {'label': 'Yes', 'value': 'yes'},
            {'label': 'No', 'value': 'no'}
        ],
        value='no',
        inline=True
    ),
    html.Label("Color 3D markers by time:"),
    dcc.RadioItems(
        id='color-by-time-toggle',
        options=[
            {'label': 'Yes', 'value': 'yes'},
            {'label': 'No', 'value': 'no'}
        ],
        value='no',
        inline=True
    ),
    dcc.Graph(id='3d-plot', figure=make_3d_figure(catalogs, catalog_names, datas, field=field)),
    dcc.Graph(id='cumulative-plot', figure=make_cumulative_figure(catalogs, catalog_names)),
    dcc.Graph(id='arrivals-hist', figure=make_arrivals_histogram(catalogs, catalog_names)),
])

@app.callback(
    dash.dependencies.Output('3d-plot', 'figure'),
    [
        dash.dependencies.Input('scale-mag-toggle', 'value'),
        dash.dependencies.Input('color-by-time-toggle', 'value')
    ]
)
def update_3d_plot(scale_mag_value, color_by_time_value):
    scale_by_magnitude = (scale_mag_value == 'yes')
    color_by_time = (color_by_time_value == 'yes')
    return make_3d_figure(
        catalogs, catalog_names, datas, field=field,
        scale_by_magnitude=scale_by_magnitude,
        color_by_time=color_by_time
    )

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)

