#!/usr/bin/env python

"""
Given the poor quality of picks in the catalog, we need to remove a certain
number with residuals over a threshold.
"""
import numpy as np
from obspy import read_events, Catalog

def cat_stats(cat):
# Getting some catalog statistics (residuals, RMS, etc)
    [x for event in cat for x in event.preferred_origin().arrivals]
    for event in cat:
        if 'time_res' not in locals():
            time_res = [x.time_residual for x in event.preferred_origin().arrivals]
        else:
            time_res += [x.time_residual for x in event.preferred_origin().arrivals]
    the_mean = np.mean(time_res)
    st_dev = np.std(time_res)
    print('Catalog has the following stats:')
    print('Mean arrival time residual: %0.3f' % the_mean)
    print('Standard deviation of arrival time residuals: %0.3f' % st_dev)
    return time_res, the_mean, st_dev

def cat_stat_ev_avg(cat):
    filtered_cat = Catalog()
    avg_arr_res = []
    for event in cat:
        pref_o = event.preferred_origin()
        # Calculate average arrival time residual for origin
        avg_arr_res.append(sum([x.time_residual for
                           i, x in enumerate(pref_o.arrivals)]) / i)
    mean_avg_ev_res = np.mean(avg_arr_res)
    std_avg_ev_res = np.std(avg_arr_res)
    print('Catalog mean avg event arr. time residual of: %0.3f' % mean_avg_ev_res)
    print('Catalog st_dev avg event arr residual of: %0.3f' % std_avg_ev_res)
    for event in cat:
        pref_o = event.preferred_origin()
        avg_arr_res = sum([x.time_residual for
                           i, x in enumerate(pref_o.arrivals)]) / i
        if avg_arr_res < std_avg_ev_res:
            filtered_cat.append(event)
        else:
            continue
    return filtered_cat, avg_arr_res, mean_avg_ev_res, std_avg_ev_res

def remove_bad_picks(cat, st_dev):
    # For removing events with 1 or more bad picks
    filtered_cat = Catalog()
    for event in cat:
        pref_o = event.preferred_origin()
        bad_arrivals = [x for x in pref_o.arrivals if x.time_residual > st_dev]
        if bad_arrivals:
            del bad_arrivals
            continue
        else:
            filtered_cat.append(event)
            del bad_arrivals

# Assign pick time errors
def assign_pk_err(cat):
    for ev in cat:
        for pk in ev.picks:
            if float(pk.comments[0].text.split('=')[-1]) > 0.90:
                pk.time_errors.uncertainty = 0.01
            elif float(pk.comments[0].text.split('=')[-1]) > 0.70:
                pk.time_errors.uncertainty = 0.05
            elif float(pk.comments[0].text.split('=')[-1]) > 0.50:
                pk.time_errors.uncertainty = 0.10
            elif float(pk.comments[0].text.split('=')[-1]) > 0.30:
                pk.time_errors.uncertainty = 0.30
            else:
                pk.time_errors.uncertainty = 0.5
    return cat

