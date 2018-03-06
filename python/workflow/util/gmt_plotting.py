#!/usr/bin/python

"""
Functions to replace gmt shell scripts
"""
import gmt
import numpy as np

from glob import glob
from subprocess import call
from obspy import read_events, UTCDateTime
from gmt.clib import LibGMT
from gmt.base_plotting import BasePlotting
from gmt.utils import build_arg_string, dummy_context, data_kind

def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def catalog_arrays(catalog):
    # Make array of elapsed seconds since start of catalog and normalized magnitudes
    dates = [ev.preferred_origin().time.datetime for ev in catalog]
    dates.sort()
    secs = [(d - dates[0]).total_seconds() for d in dates]
    secs /= np.max(secs)
    # Magnitude scaling as well
    mags = [ev.magnitudes[-1].mag for ev in catalog]
    mags /= np.max(mags)
    return secs, mags

def gmt_project(catalog, center, end, mags, secs):
    """
    System call of gmt project at this point is best way to get what we want
    """
    # Write temporary file
    with open('/home/chet/gmt/tmp/cat.tmp', 'w') as f:
        for ev, mag, sec in zip(catalog, mags, secs):
            f.write('{} {} {} {} {}\n'.format(ev.preferred_origin().longitude,
                                        ev.preferred_origin().latitude,
                                        ev.preferred_origin().depth,
                                        mag, sec))
    outfile = '/home/chet/gmt/tmp/cat_proj.tmp'
    cmd = 'gmt project /home/chet/gmt/tmp/cat.tmp'
    args = '-C{:.3f}/{:.3f} -E{}/{} -Fpz -Q -V > {}'.format(center[0], center[1],
                                                    end[0], end[1],
                                                    outfile)
    call(' '.join([cmd, args]), shell=True)
    return outfile

def plot_Nga_well_map(fig):
    # Well tracks
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_tracks_injection.gmt',
             W='1.5,blue', S='qn1:+f8p,blue+Lh')
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_tracks_production.gmt',
             W='1.5,red', S='qn1:+f8p,red+Lh')
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM06_track.gmt', W='1.5,blue',
             S='qn1:+f8p,blue+Lh+n-.2/0.2')
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM08_track.gmt', W='1.5,blue',
             S='qn1:+f8p,blue+Lh+n0/-0.15')
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM10_track.gmt', W='1.5,blue',
             S='qn1:+f8p,blue+Lh+n0/0')
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM09_track.gmt', W='1.5,blue',
             S='qn1:+f8p,blue+Lh+n0/0')
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM12_track.gmt', W='1.5,red',
             S='qn1:+f8p,red+Lh')
    # Wellhead points
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_injection_pts.gmt',
             color='blue', style='c0.2c')
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_production_pts.gmt',
             color='red', style='c0.2c')
    return

def plot_Nga_well_depth(fig):
    proj_wells = glob('/home/chet/gmt/data/NZ/wells/*project.txt')
    for well in proj_wells:
        well_nm = well.split('/')[-1].split('_')[0]
        if well_nm in ['NM10', 'NM06', 'NM08', 'NM09']:
            fig.plot(data=well, W='0.9p,blue')
        else:
            fig.plot(data=well, W='0.9p,red')
    proj_fzs = glob('/home/chet/gmt/data/NZ/wells/feedzones/*project.csv')
    for well in proj_fzs:
        fig.plot(data=well, W='6.5p,blue')
    return

def plot_earthquakes_map(catalog, mags, secs, fig):
    lons = np.asarray([ev.preferred_origin().longitude for ev in catalog])
    lats = np.asarray([ev.preferred_origin().latitude for ev in catalog])
    fig.plot(x=lons, y=lats, sizes=mags/4, color=secs,
             style='cc', cmap='cool')
    return

def plot_earthquakes_depth(catalog, mags, secs, center_pt, end_pt,
                           region, scale, B_list, Y, fig):
    tmpfile = gmt_project(catalog, center_pt, end_pt, mags, secs)
    with open(tmpfile, 'r') as f:
        x = []; y = []; pmags = []; psecs = []
        for line in f:
            ln = line.split('\t')
            x.append(float(ln[0]))
            y.append(float(ln[1]) / 1000.)
            pmags.append(float(ln[2]))
            psecs.append(float(ln[3].rstrip('\n')))
    fig.plot(x=np.array(x), y=np.array(y), sizes=np.array(pmags)/4,
             color=np.array(psecs), cmap='cool', style='cc', R=region,
             J=scale, B=B_list, Y=Y)
    return

def plot_water(fig):
    # Water
    fig.plot(data='/home/chet/gmt/data/NZ/water/taupo_lakes.gmt',
             color='cornflowerblue')
    fig.plot(data='/home/chet/gmt/data/NZ/water/taupo_river_polygons.gmt',
             color='cornflowerblue')
    return

def plot_background_datasets(fig, region):
    fig.plot(region=region, projection='M13', Y='10',
             data='/home/chet/gmt/data/NZ/resistivity/RT_Boundary_Risk_2000.gmt',
             color='lightgray')
    fig.plot(
        data='/home/chet/gmt/data/NZ/resistivity/NM_Boundary_Bosely2010.gmt',
        color='lightgray')
    # Water
    fig.plot(data='/home/chet/gmt/data/NZ/water/taupo_lakes.gmt',
             color='cornflowerblue')
    fig.plot(data='/home/chet/gmt/data/NZ/water/taupo_river_polygons.gmt',
             color='cornflowerblue')
    # Faults
    fig.plot(data='/home/chet/gmt/data/NZ/faults/NZ_active_faults.gmt',
             W='0.85')
    fig.basemap(L='x11.5/1/-38/1', frame=['0.02', 'wsEN'])
    return

def plot_stations(fig):
    fig.plot(data='/home/chet/gmt/data/NZ/stations/mrp_inventory.gmt',
             style='t0.3c', color='black', W='0.07')
    return

def plot_Nga_section_line(start_pt, end_pt, fig):
    fig.plot(x=np.asarray([start_pt[0], end_pt[0]]),
             y=np.asarray([start_pt[1], end_pt[1]]),
             W='1.0,black,-')
    labels_file = '/home/chet/gmt/data/NZ/misc/section_labels.txt'
    with LibGMT() as lib:
        label_args = {'F': '+f16p,black,-'}
        file_context = dummy_context(labels_file)
        with file_context as fname:
            arg_str = ' '.join([fname, build_arg_string(label_args)])
            lib.call_module('pstext', arg_str)
    return

def plot_Nga_scale():
    scale_args = {'C': 'cool', 'D': '10.25/13.75/3.5/0.5h',
                  'B': 'px1+l"Date of occurrence"',
                  'F': '+gwhite+p1.2k'}
    with LibGMT() as lib:
        lib.call_module('psscale', ' '.join([build_arg_string(scale_args),
                                             '--FONT_ANNOT_PRIMARY=7p',
                                             '--FONT_LABEL=9p']))
    return

def plot_NM5_7_4():
    pts5_7 = '/home/chet/gmt/data/NZ/wells/NM05_NM07_pts.gmt'
    args5_7 = {'D': '0.25/0.15', 'F': '+jCB+f8p,red'}
    pt4 = '/home/chet/gmt/data/NZ/wells/NM04_pt.gmt'
    args4 = {'D': '0.35/-0.35'}
    with LibGMT() as lib:
        for pts, kwargs in zip([pts5_7, pt4], [args5_7, args4]):
            file_context = dummy_context(pts)
            with file_context as fname:
                arg_str = ' '.join([fname, build_arg_string(kwargs)])
                lib.call_module('pstext', arg_str)
    return

def plot_Nga_static(catalog, start_pt, end_pt, secs=[],
                    mags=[], show=True, outfile=None):
    """
    Main code for plotting Ngatamariki seismicity
    :param catalog: Catalog of events to plot
    :param center_pt: Center point for the cross section (tup)
    :param end_pt: End point for the cross_section (tup)
    :return:
    """
    # Arg check
    if outfile:
        show = False
    # Sort catalog
    catalog.events.sort(key=lambda x: x.picks[-1].time)
    # Calculate mid-point of cross_section
    c_lon = start_pt[0] + ((end_pt[0] - start_pt[0]) / 2.)
    c_lat = start_pt[1] + ((end_pt[1] - start_pt[1]) / 2.)
    if len(secs) == 0 and len(mags) == 0:
        secs, mags = catalog_arrays(catalog)
    # Set prefs
    with LibGMT() as lib:
        lib.call_module('gmtset', 'FONT_ANNOT_PRIMARY 10p')
        lib.call_module('gmtset', 'FORMAT_GEO_MAP ddd.xx')
        lib.call_module('gmtset', 'MAP_FRAME_TYPE plain')
    region = [176.15, 176.23, -38.58, -38.51]
    # Set up figure
    fig = gmt.Figure()
    plot_background_datasets(fig, region=region)
    plot_earthquakes_map(catalog, mags, secs, fig)
    plot_Nga_well_map(fig)
    plot_NM5_7_4()
    plot_stations(fig)
    plot_Nga_scale()
    plot_Nga_section_line(start_pt=start_pt, end_pt=end_pt, fig=fig)
    # Cross section
    scale = 'X13/-6'
    B_list = ['px1.0+l"Distance from center of x-section (km)"',
              'py1.0+l"Depth (km)"', 'WeSn']
    Y = '-7.25'
    # Set new prefs
    with LibGMT() as lib:
        lib.call_module('gmtset', 'FONT_ANNOT_PRIMARY 10p')
        lib.call_module('gmtset', 'FONT_LABEL 12p')
    # Eq cross section
    x_region = [-3.635, 3.635, -0.5, 4.5]
    plot_earthquakes_depth(catalog, mags, secs, center_pt=(c_lon, c_lat),
                           end_pt=end_pt, region=x_region, scale=scale,
                           B_list=B_list, Y=Y, fig=fig)
    plot_Nga_well_depth(fig)
    if show:
        fig.show(external=True)
    elif outfile:
        fig.savefig(outfile)
    return

def earthquake_video(catalog, outdir, field='Nga', buffer=100):
    """
    Overarching function for plotting a series of pngs and compiling videos
    :param catalog:
    :param steps:
    :return:
    """
    # Sort catalog
    catalog.events.sort(key=lambda x: x.preferred_origin().time)
    # Establish size/color arrays
    secs, mags = catalog_arrays(catalog)
    # Loop over specified steps
    counter = 0
    for date in date_generator(catalog[0].origins[-1].time.date,
                               catalog[-1].origins[-1].time.date):
        dto = UTCDateTime(date)
        print('Plotting {}'.format(dto))
        start_str = 'time >= %s' % str(dto)
        end_str = 'time <= %s' % str(dto + 86400)
        day_cat = catalog.filter(start_str, end_str)
        no_evs = len(day_cat)
        counter += no_evs
        plot_cat = catalog[:counter]
        tmp_mags = mags[:counter]
        tmp_secs = secs[:counter]
        if field == 'Nga':
            start_pt = (176.171, -38.517)
            end_pt = (176.209, -38.575)
            plot_Nga_static(plot_cat, start_pt, end_pt, tmp_secs, tmp_mags,
                            outfile='{}/Nga_static_{:s}.png'.format(
                                outdir, str(dto)))
    return

def plot_flow_rates():
    return