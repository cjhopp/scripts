#!/usr/bin/python

"""
Script to pull down Michigan-Huron water level data and plot to html.

Intended to be posted on website
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from datetime import datetime


def read_waterlevels(path):
    df = pd.read_csv(path, header=2, index_col='year')
    start = datetime(year=1918, month=1, day=1)
    end_yr = df.index.values[-1]
    # Whats the current (last) month in the data? Future are NaN
    end_month = df.iloc[-1].last_valid_index()
    end = datetime.strptime('{}-{}'.format(end_yr, end_month), '%Y-%b')
    months = pd.date_range(start, end, freq='BM')
    levels = df.values.flatten()
    return months, levels


def plot_waterlevels(path):
    """
    Make plotly figure of the water levels (and eventually other stuff)

    :return:
    """
    fig = go.Figure()
    # Read the file
    months, levels = read_waterlevels(path)
    fig.add_trace(go.Scatter(x=months, y=levels, name='Monthly Avg. Level',
                             line=dict(color='navy', width=1.),
                             mode='lines'))
    fig.add_trace(go.Scatter(x=[months[0], months[-1]],
                             y=[np.nanmean(levels), np.nanmean(levels)],
                             name='Avg. Level 1918-present',
                             line=dict(color='red'),
                             mode='lines'))
    fig.update_layout(template='plotly',
                      xaxis=dict(
                          rangeselector=dict(
                              buttons=list([
                                  dict(count=5,
                                       label="5y",
                                       step="year",
                                       stepmode="backward"),
                                  dict(count=10,
                                       label="10y",
                                       step="year",
                                       stepmode="backward"),
                                  dict(count=30,
                                       label="30y",
                                       step="year",
                                       stepmode="backward"),
                                  dict(count=50,
                                       label="50y",
                                       step="year",
                                       stepmode="backward"),
                                  dict(step="all"),
                              ])
                          ),
                          rangeslider=dict(
                              visible=True
                          ),
                          type="date"
                      )
                      )
    fig.show(renderer='firefox')
    return