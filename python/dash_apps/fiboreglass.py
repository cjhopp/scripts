import h5py
import json
import os

from glob import glob
from datetime import datetime
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import xarray as xr
import holoviews as hv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc

from plotly.subplots import make_subplots
from obspy import UTCDateTime
from flask_caching import Cache
from dash import Dash, html, dcc, Input, ctx, Output, State, callback

# Neubrex mapping
neubrex_mapping = {
    'B1': (56.36, 156.25),
    # 'B2': (1796.50, 1907.02),
    'B3': (1140.27, 1311.10),
    'B4': (1316.81, 1477.03),
    'B5': (982.92, 1101.35),
    'B6': (1499.42, 1600.32),
    'B7': (1605.37, 1700.63),
    'B9': (215.51, 331.46),
    'B10': (796.38, 932.75),
    'B11': (635.68, 707.95),
    # 'B12': (412.43, 512.04)  #OG mapping with Antonio
    'B12': (405.82, 512.04)
}

fsc23_mapping_bottom = {
    'B1': 106.305,
    'B2': 1834.105,
    'B3': 1225.685,
    'B4': 1396.92,
    'B5': 1042.135,
    'B6': 1549.87,
    'B7': 1653.0,
    'B9': 273.485,
    'B10': 864.565,
    'B11': 671.815,
    'B12': 458.93
}


fiber_depths = {'D1': 21.26, 'D2': 17.1, 'D3': 31.42, 'D4': 35.99, 'D5': 31.38,
                'D6': 36.28, 'D7': 29.7, 'B1': 51.5, 'B2': 53.3, 'B3': 84.8,
                'B4': 80., 'B5': 59., 'B6': 49.5, 'B7': 49.3, 'B8': 61.,
                'B9': 61., 'B10a': 35.5, 'B10b': 35.5, 'B11': 36.25, 'B12': 50, 'Tank': 30}

fiber_winding = 5  # Degrees from borehole axis

def read_fsb_hydro(path, year=2023, B1_path=False, B12_path=False):
    """Helper to read in Pauls hydraulic data"""
    if year == 2020:
        df = pd.read_csv(path, names=['Time', 'Pressure', 'Flow'], header=0)
        df['dt'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S.%f')
        tz = 'CET'
        df = df.set_index('dt')
        df = df.drop(['Time'], axis=1)
        df.index = df.index.tz_localize(tz)
        df.index = df.index.tz_convert('UTC')
        df.index = df.index.tz_convert(None)
    elif year == 2023:
        df = pd.read_csv(path, names=['Time', 'Flow', 'Pressure', 'CO2'], header=0)
        df['dt'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M:%S.%f')
        tz = 'UTC'
        df = df.set_index('dt')
        df = df.drop(['Time'], axis=1)
        df.index = df.index.tz_localize(tz)
        df.index = df.index.tz_convert('UTC')
        df.index = df.index.tz_convert(None)
        df = df.resample('5s').mean()
        if B1_path:
            pkls = glob('{}/*.pkl'.format(B1_path))
            for pkl in pkls:
                with open(pkl, 'rb') as f:
                    df1 = pickle.load(f)
                    df1 = df1.rename({'Pressure': '{} Pressure'.format(pkl.split('/')[-1]),
                                      'Temperature': '{} Temperature'.format(pkl.split('/')[-1])}, axis=1)
                df1 = df1.resample('5s').mean()
                df = pd.concat([df, df1], join='outer')
        if B12_path:
            df12 = pd.read_csv(B12_path, names=['Time', 'B12 pressure [kPa]', 'CO2 pp', 'pH'], header=0)
            df12['dt'] = pd.to_datetime(df12['Time'], format='%m/%d/%Y %H:%M:%S.%f')
            tz = 'UTC'
            df12 = df12.set_index('dt')
            df12 = df12.drop(['Time'], axis=1)
            df12 = df12.resample('5s').mean().interpolate('linear')
            df12.plot()
            plt.show()
            df12.index = df12.index.tz_localize(tz)
            df12.index = df12.index.tz_convert('UTC')
            df12.index = df12.index.tz_convert(None)
            df = pd.concat([df, df12], join='outer')
    elif year == 2021:
        df = pd.read_csv(path, names=['Time', 'Pressure', 'Packer', 'Flow', 'Qin', 'Qout', 'Hz'], header=0)
        df['dt'] = pd.to_datetime(df['Time'], format='%d-%b-%Y %H:%M:%S')
        tz = 'CET'  # ????
        df = df.set_index('dt')
        df = df.drop(['Time'], axis=1)
        df.index = df.index.tz_localize(tz)
        df.index = df.index.tz_convert('UTC')
        df.index = df.index.tz_convert(None)
    return df


# Read in NetCDF file as Dataset now
ds = xr.open_dataset('/media/chopp/Data1/chet-FS-B/DSS/FSC23/FSC_Omnisens.nc')
# Convert to relative strain and slice in global env now. Should move these to some sort of interactive thing
ds['microstrain'] -= ds['microstrain'].isel(time=0)
ds = ds.sel(time=slice(datetime(2023, 5, 1), datetime(2023, 5, 15)))
# Also read in FSC injections for now
df_fsc = read_fsb_hydro('/media/chopp/Data1/chet-FS-B/pump/MtTerriInjectionMay2023_BFSB2_PQ_CO2_1Hz.csv')

subplots = make_subplots(rows=3, cols=3, shared_xaxes='columns',
                         specs=[[{'rowspan': 3, 'colspan': 1}, {'colspan': 2, 'rowspan': 1}, None],
                                [None, {'colspan': 2, 'rowspan': 1}, None],
                                [None, {'colspan': 2, 'rowspan': 1, 'secondary_y': True}, None]]
                         )

f = go.FigureWidget(subplots)
# Plot dummies to start
f.add_annotation(x=1, y=1, text='No data', font=dict(size=50, color='gray', variant='small-caps'),
                 textangle=-90, xref='x', yref='y', row=1, col=1)
f.add_annotation(x=1, y=1, text='No data', font=dict(size=50, color='gray', variant='small-caps'),
                 textangle=0, xref='x', yref='y', row=3, col=2)
f.add_annotation(x=1, y=1, text='No data', font=dict(size=50, color='gray', variant='small-caps'),
                 textangle=0, xref='x', yref='y', row=1, col=2)
f.add_annotation(x=1, y=1, text='No data', font=dict(size=50, color='gray', variant='small-caps'),
                 textangle=0, xref='x', yref='y', row=2, col=2)
f.update_layout(dict(xaxis2=dict(matches="x3"),
                     yaxis2=dict(matches="y3", autorange="reversed", title="Depth [m]"),
                     yaxis3=dict(autorange="reversed", title="Depth [m]"),
                     xaxis4=dict(matches="x3"),
                     yaxis5=dict(domain=[0.0, 0.26666666666666666]),
                     coloraxis=dict(colorscale="RdBu_r"),
                     coloraxis_colorbar=dict(title="Microstrain", len=0.33),
                     template='ggplot2'))
f.update_yaxes(showline=True, linecolor='gray')
f.update_xaxes(showline=True, linecolor='gray')


CACHE_CONFIG = {
    # try 'FileSystemCache' if you don't want to setup redis
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
}

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)


# Define layout
app.layout = dbc.Container(children=[
    html.H1(children='Fiboreglass', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(
            id='interactive-plot',
            figure=f,
            style={'width': '75vh', 'height': '75vh', 'marginLeft': 'auto', 'marginRight': 'auto'},
        ),
    ], id='graph-container'),
    html.Hr(),
    dbc.Row(children=[
        dbc.Col([
            dcc.Upload(
                dbc.Button('Upload Mapping', color='secondary', className='me-1'),
                       id='upload-mapping')],
                md=4
        ),
        dbc.Col([
            dcc.Dropdown(
                value='Wells',
                options=[well for well in fsc23_mapping_bottom.keys()],
                id='well-dropdown'
            ),
        ], md=4),
        dbc.Col([
            dcc.Slider(0, 1000, 100, value=500, id='color-slider')
        ], md=4)
    ], align='center'),
    dcc.Store(id='signal')
])


@callback(
    Output('interactive-plot', 'figure'),
    Input('well-dropdown', 'value'),
    Input('interactive-plot', 'clickData'),
    Input('color-slider', 'value'),
    State('interactive-plot', 'figure'),
    State('color-slider', 'value'),
    State('well-dropdown', 'value'),
    State('graph-container', 'n_clicks'),
    prevent_initial_call=True
)
def update_figure(b1, b2, b3, fig, color, well, clicks):
    triggered_id = ctx.triggered_id
    if triggered_id == 'well-dropdown':
        return select_well(b1, color)
    elif triggered_id == 'interactive-plot':
        return pick_data(fig, b2, well, clicks)
    elif triggered_id == 'color-slider':
        return recolor(fig, b3)


def recolor(fig, color):
    fig = go.FigureWidget(fig)
    fig.update_layout(
        xaxis=dict(title="Microstrain", range=[-color, color]),
        yaxis4=dict(title='Microstrain', range=[-color, color]),
        coloraxis=dict(colorscale="RdBu_r", cmin=-color, cmax=color)
    )
    return fig


@cache.memoize()
def extract_well(well):
    dt_array = pd.to_datetime(ds.time)
    mapping = fsc23_mapping_bottom[well]
    fiber_depth = (fiber_depths[well] /
                   np.cos(np.deg2rad(fiber_winding)))
    # Downgoing fiber
    hole_depths = ds.depth.sel(depth=slice(mapping - fiber_depth, mapping)).values.copy()
    hole_depths -= hole_depths[0]
    down_data = ds.sel(depth=slice(mapping - fiber_depth, mapping))['microstrain'].values
    up_data = ds.sel(depth=slice(mapping, mapping + fiber_depth))['microstrain'].values
    data_dict = {'time': dt_array.astype(str), 'depth': hole_depths, 'down_data': down_data, 'up_data': up_data}
    return data_dict


def select_well(well, color):
    """
    Plot the waterfalls for the selected well
    :param well:
    :return:
    """
    subplots = make_subplots(rows=3, cols=3, shared_xaxes='columns',
                             specs=[[{'rowspan': 3, 'colspan': 1}, {'colspan': 2, 'rowspan': 1}, None],
                                    [None, {'colspan': 2, 'rowspan': 1}, None],
                                    [None, {'colspan': 2, 'rowspan': 1, 'secondary_y': True}, None]]
                             )
    fig = go.FigureWidget(subplots)
    print(fig.layout)
    data_dict = extract_well(well)
    dt_array = [UTCDateTime(dt).datetime for dt in data_dict['time']]
    hole_depths = data_dict['depth']
    down_data = data_dict['down_data']
    up_data = data_dict['up_data']
    fig.add_heatmap(z=down_data[::-1, :], y=hole_depths[::-1], x=dt_array, row=1, col=2, coloraxis="coloraxis")
    fig.add_heatmap(z=up_data, y=hole_depths[::-1], x=dt_array, row=2, col=2, coloraxis="coloraxis")
    # Hard coded FSC flow/CO2 for now
    fig.add_scatter(x=df_fsc.index, y=df_fsc['Flow'], name='Flow [L/min]', legendgroup='group3',
                    legendgrouptitle_text='Injection parameters', yaxis='y5', xaxis='x4', line=dict(color='steelblue'))
    fig.add_scatter(x=df_fsc.index, y=df_fsc['CO2'] * 100, line=dict(color='purple'), name=r'CO2 [g/g]x100',
                    legendgroup='group3', yaxis='y5', xaxis='x4')
    fig.update_layout(dict(yaxis=dict(title="Depth [m]", range=[hole_depths[-1], 0]),
                           xaxis=dict(title="Microstrain", range=[-color, color]),
                           xaxis2=dict(matches="x3"),
                           yaxis2=dict(matches="y3", autorange="reversed", title="Depth [m]"),
                           yaxis3=dict(autorange="reversed", title="Depth [m]"),
                           xaxis4=dict(matches="x3", range=[dt_array[0], dt_array[-1]]),
                           yaxis4=dict(title='Microstrain', range=[-color, color]),
                           yaxis5=dict(range=[0, 10], domain=[0.0, 0.26666666666666666], title='Flow [L/min]'),
                           coloraxis=dict(colorscale="RdBu_r", cmin=-color, cmax=color),
                           coloraxis_colorbar=dict(title="Microstrain", len=0.33),
                           template='ggplot2'))
    fig.update_yaxes(showline=True, linecolor='gray')
    fig.update_xaxes(showline=True, linecolor='gray')
    return fig


def pick_data(fig, clickData, well, clicks):
    """
    Take click data, check if it falls in a waterfall, then plot to the depth and time graph
    :param clickData:
    :return:
    """
    print(clickData)
    fig = go.FigureWidget(fig)
    data_dict = extract_well(well)
    dt_array = [UTCDateTime(dt).datetime for dt in data_dict['time']]
    hole_depths = data_dict['depth']
    down_data = data_dict['down_data']
    up_data = data_dict['up_data']
    which_waterfall = clickData['points'][0]['curveNumber']
    # Check if in waterfalls
    if not which_waterfall in [0, 1]:
        # Output some message somewhere?
        return
    data = [down_data, up_data]
    click_time = UTCDateTime(clickData['points'][0]['x']).datetime
    click_t_i = np.argmin(np.abs(np.array(dt_array) - click_time))
    click_depth = clickData['points'][0]['y']
    click_d_i = np.argmin(np.abs(click_depth - hole_depths))
    if which_waterfall == 1:
        click_d_i = up_data.shape[0] - click_d_i
    time_trace = data[clickData['points'][0]['curveNumber']][click_d_i, :]
    depth_trace = data[clickData['points'][0]['curveNumber']][:, click_t_i]
    if which_waterfall == 1:
        depth_trace = depth_trace[::-1]
    # Update graphs
    fig.add_scatter(x=depth_trace, y=hole_depths, name=str(click_time), legendgroup="group",
                    legendgrouptitle_text="Depth traces", showlegend=True, yaxis='y')
    fig.add_scatter(x=dt_array, y=time_trace, name='{:.3f} m'.format(click_depth),
                    legendgroup="group2", legendgrouptitle_text="Time traces",
                    yaxis='y4', xaxis='x4', showlegend=True)
    fig.update_layout(dict(uirevision=True))  # Keep user interactions from before update (like zoom)
    return fig


# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)