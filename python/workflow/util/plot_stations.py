#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_station_times(inventory, dates=None, ax=None, color=None, show=False):
    """
    Plot a horizontal bar chart of the operational time for each station
    in an Inventory object

    :param inventory: obspy Inventory object
    :return:
    """
    # Sort by name first (assuming one network)
    inventory[0].stations.sort(key=lambda x: x.code, reverse=True)
    if not ax:
        fig, axes = plt.subplots()
    else:
        axes = ax
    # Get station labels, y positions and date ranges
    sta_labs = [sta.code for net in inventory for sta in net]
    ypos = np.arange(len(sta_labs))
    width = [mdates.date2num(sta.end_date.datetime) -
             mdates.date2num(sta.start_date.datetime)
             for net in inventory for sta in net]
    left_edge = [mdates.date2num(sta.start_date.datetime)
                 for net in inventory for sta in net]
    print(left_edge)
    if not color:
        color = 'dimgray'
    axes.barh(ypos, width=width, left=left_edge, align='center',
              color=color, alpha=0.8)
    axes.set_yticks(ypos)
    axes.set_yticklabels(sta_labs)
    axes.set_xlabel('Date', fontsize=16)
    if dates:
        axes.set_xlim([mdates.date2num(dates[0]), mdates.date2num(dates[1])])
    locator = mdates.AutoDateLocator(minticks=3)
    formatter = mdates.AutoDateFormatter(locator)
    axes.xaxis.set_major_locator(locator)
    axes.xaxis.set_major_formatter(formatter)
    if show:
        plt.show()
    return axes