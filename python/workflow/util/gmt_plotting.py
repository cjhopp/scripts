#!/usr/bin/python

"""
Functions to replace gmt shell scripts
"""
import gmt
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from glob import glob
from subprocess import call
from obspy import UTCDateTime, Catalog
from focal_mecs import format_arnold_to_gmt
from gmt.clib import LibGMT
from itertools import cycle
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


def gmt_project(catalog, center, end, mags=None, secs=None, fm_file=None):
    """
    System call of gmt project at this point is best way to get what we want
    """
    # Write temporary file
    if not (mags is None and secs is None):
        with open('/home/chet/gmt/tmp/cat.tmp', 'w') as f:
            for ev, mag, sec in zip(catalog, mags, secs):
                f.write('{} {} {} {} {}\n'.format(ev.preferred_origin().longitude,
                                            ev.preferred_origin().latitude,
                                            ev.preferred_origin().depth,
                                            mag, sec))
        outfile = '/home/chet/gmt/tmp/cat_proj.tmp'
        cmd = 'gmt project /home/chet/gmt/tmp/cat.tmp'
    elif fm_file:
        outfile = '/home/chet/gmt/tmp/fm_proj.tmp'
        cmd = 'gmt project {}'.format(fm_file)
    else:
        # If plotting seis but no events in catalog, write empty file to proj
        outfile = '/home/chet/gmt/tmp/cat_proj.tmp'
        infile = '/home/chet/gmt/tmp/cat.tmp'
        cmd = 'gmt project {}'.format(infile)
        with open(infile, 'w') as f:
            f.write('\n')
    args = '-C{:.3f}/{:.3f} -E{}/{} -Fpz -Q -V > {}'.format(center[0],
                                                            center[1],
                                                            end[0], end[1],
                                                            outfile)
    call(' '.join([cmd, args]), shell=True)
    if fm_file:
        # Need to put a dummy 'depth' column in the projected fm file
        with open(outfile, 'r') as f:
            with open('{}.new'.format(outfile), 'w') as fo:
                for line in f:
                    ln = line.rstrip('\n').split()
                    ln.insert(2, '0.0')
                    fo.write('{}\n'.format(' '.join(ln)))
        outfile = '{}.new'.format(outfile)
    return outfile


def plot_Nga_well_map(fig, dto):
    # Well tracks
    if dto:
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_tracks_injection.gmt',
                 W='1.5,blue')
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_tracks_production.gmt',
                 W='1.5,red')
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM10_track.gmt',
                 W='1.5,blue')
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM09_track.gmt',
                 W='1.5,blue')
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM12_track.gmt',
                 W='1.5,red')
    else:
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_tracks_injection.gmt',
                 W='1.5,blue', S='qn1:+f8p,blue+Lh')
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_tracks_production.gmt',
                 W='1.5,red', S='qn1:+f8p,red+Lh')
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM10_track.gmt', W='1.5,blue',
                 S='qn1:+f8p,blue+Lh+n0/0')
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM09_track.gmt', W='1.5,blue',
                 S='qn1:+f8p,blue+Lh+n0/0')
        fig.plot(data='/home/chet/gmt/data/NZ/wells/NM12_track.gmt', W='1.5,red',
                 S='qn1:+f8p,red+Lh')
    # Wellhead points
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_injection_pts.gmt',
             color='blue', style='c0.1')
    fig.plot(data='/home/chet/gmt/data/NZ/wells/NM_production_pts.gmt',
             color='red', style='c0.1')
    return


def plot_Nga_well_depth(fig):
    proj_wells = glob('/home/chet/gmt/data/NZ/wells/*project.txt')
    for well in proj_wells:
        well_nm = well.split('/')[-1].split('_')[0]
        if well_nm in ['NM10', 'NM06', 'NM08', 'NM09', 'NM04']:
            fig.plot(data=well, W='0.9p,blue')
        else:
            fig.plot(data=well, W='0.9p,red')
    proj_fzs = glob('/home/chet/gmt/data/NZ/wells/feedzones/*project.csv')
    for well in proj_fzs:
        fig.plot(data=well, W='6.5p,blue')
    return


def plot_earthquakes_map(catalog, mags, secs, fig, old_cat=None,
                         old_mags=None):
    lons = np.asarray([ev.preferred_origin().longitude for ev in catalog])
    lats = np.asarray([ev.preferred_origin().latitude for ev in catalog])
    if old_cat:
        lons_old = np.asarray([ev.preferred_origin().longitude
                               for ev in old_cat])
        lats_old = np.asarray([ev.preferred_origin().latitude
                               for ev in old_cat])
        fig.plot(x=lons_old, y=lats_old, sizes=old_mags / 2, style='cc',
                 color='grey')
    fig.plot(x=lons, y=lats, sizes=mags / 2, color=secs,
             style='cc', cmap='cool')
    return


def plot_earthquakes_depth(catalog, mags, secs, center_pt, end_pt,
                           region, scale, B_list, Y, fig, old_cat=None,
                           old_mags=None, old_secs=None):
    if old_cat:
        tmpfile_old = gmt_project(old_cat, center_pt, end_pt, old_mags,
                                  old_secs)
        with open(tmpfile_old, 'r') as f:
            xo = []; yo = []; pmagso = []; psecso = []
            for line in f:
                ln = line.split('\t')
                xo.append(float(ln[0]))
                yo.append(float(ln[1]) / 1000.)
                pmagso.append(float(ln[2]))
                psecso.append(float(ln[3].rstrip('\n')))
        fig.plot(x=np.array(xo), y=np.array(yo), sizes=np.array(pmagso) / 2,
                 color='grey', style='cc', R=region, J=scale, B=B_list, Y=Y)
        Y = 0
    tmpfile = gmt_project(catalog, center_pt, end_pt, mags, secs)
    with open(tmpfile, 'r') as f:
        x = []; y = []; pmags = []; psecs = []
        for line in f:
            ln = line.split('\t')
            x.append(float(ln[0]))
            y.append(float(ln[1]) / 1000.)
            pmags.append(float(ln[2]))
            psecs.append(float(ln[3].rstrip('\n')))
    fig.plot(x=np.array(x), y=np.array(y), sizes=np.array(pmags) / 2,
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
    # AFZ label
    with LibGMT() as lib:
        label_args = {'F': '+f10p+a50'}
        file_context = dummy_context('/home/chet/gmt/data/NZ/faults/afz_text.txt')
        with file_context as fname:
            arg_str = ' '.join([fname, build_arg_string(label_args)])
            lib.call_module('pstext', arg_str)
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


def plot_date_text(dto):
    pt = '176.21 -38.515'
    with open('tmp_text.csv', 'w') as f:
        f.write('{} @:14:Date: {}@::'.format(pt, dto.strftime('%d-%m-%Y')))
    with LibGMT() as lib:
        file_context = dummy_context('tmp_text.csv')
        with file_context as fname:
            lib.call_module('pstext', fname)
    os.remove('tmp_text.csv')
    return


def plot_date_line(dto, fig):
    with open('tmp_line.csv', 'w') as f:
        f.write('{} 0\n'.format(dto.strftime('%Y-%m-%dT%H:%M:%S')))
        f.write('{} 1300\n'.format(dto.strftime('%Y-%m-%dT%H:%M:%S')))
    fig.plot(data='tmp_line.csv', W='1.0,black,--')
    return


def injection_legend(leg_file):
    args = {'D': 'x0.1/7.5+w4/2/TC',
            'C': '0.1i/0.1i', 'F': '+gwhite+p'}
    with LibGMT() as lib:
        file_context = dummy_context(leg_file)
        with file_context as fname:
            arg_str = ' '.join([fname, build_arg_string(args)])
            lib.call_module('pslegend', arg_str)
    return


def plot_injection_rates(fig, dto=None, field='North', data='Flow'):
    """
    Plot injection rates on separate axis, as well as time of plot as vertical
    line for video purposes
    :return:
    """
    well_dir = '/home/chet/data/mrp_data/well_data/flow_rates/July_2017_final'
    well_fs = glob('{}/*flow_all.csv'.format(well_dir))
    well_whps = glob('{}/*WHP_all.csv'.format(well_dir))
    well_fs.sort()
    colors = cycle(['darkturquoise', 'lightblue', 'darkseagreen', 'lawngreen'])
    region = '2012T/2016T/0/1300' # Time on x, flow rate on y
    proj = 'X15/9.7'
    with LibGMT() as lib:
        lib.call_module('gmtset', 'FONT_ANNOT_PRIMARY 12p')
        lib.call_module('gmtset', 'FORMAT_DATE_MAP o')
        lib.call_module('gmtset', 'FORMAT_TIME_PRIMARY_MAP Character')
        lib.call_module('gmtset', 'FONT_TITLE 16p')
    if dto: # Set parameters for animation next to map
        fig.plot(data=well_fs[1], Y=12, X=16, R=region,
                 projection=proj, W='0.7,{}'.format(next(colors)))
        fig.plot(data=well_fs[2], W='0.7,{}'.format(next(colors)))
        plot_date_line(dto, fig)
        # Legend North
        injection_legend('/home/chet/gmt/data/NZ/misc/ngaN_legend_flow.txt')
        fig.basemap(B=['pxa1Y', 'pya100+l"Flow rate (t/h)"',
                       'SEwn+t"Ngatamariki North"'])
        fig.plot(data=well_fs[0], Y=-12, R=region,
                 projection=proj, W='0.7,{}'.format(next(colors)))
        fig.plot(data=well_fs[-1], W='0.7,{}'.format(next(colors)))
        if dto:
            plot_date_line(dto, fig)
            # Legend South
        injection_legend('/home/chet/gmt/data/NZ/misc/ngaS_legend_flow.txt')
        fig.basemap(B=['pxa1Y', 'pya100+l"Flow rate (t/h)"',
                       'wSEn+t"Ngatamariki South"'])
    else: # Standalone plot of flow rates
        if data == 'Flow':
            for f in well_fs:
                well = f.split('/')[-1].split('_')[0]
                if well in ['NM08', 'NM09'] and field == 'North':
                    fig.plot(data=f, R=region, projection=proj,
                             W='0.7,{}'.format(next(colors)))
                elif well in ['NM06', 'NM10'] and field == 'South':
                    fig.plot(data=f, R=region, projection=proj,
                             W='0.7,{}'.format(next(colors)))
            fig.basemap(B=['pxa1Y', 'pya100+l"Flow rate (t/h)"',
                           'SEwn+t"Ngatamariki {}"'.format(field)])
        elif data == 'WHP':
            for f in well_whps:
                region = '2012T/2016T/0/35' # Time on x, WHP on y
                well = f.split('/')[-1].split('_')[0]
                if well in ['NM08', 'NM09'] and field == 'North':
                    fig.plot(data=f, R=region, projection=proj,
                             W='0.7,{}'.format(next(colors)))
                elif well in ['NM06', 'NM10'] and field == 'South':
                    fig.plot(data=f, R=region, projection=proj,
                             W='0.7,{}'.format(next(colors)))
                fig.basemap(B=['pxa1Y', 'pya5+l"WHP (barg)"',
                               'syf1', 'WSen+t"Ngatamariki {}"'.format(field)],
                            R=region)
    return


def plot_well_files(well_list, params, show=True, outfile=None):
    """
    Plot well data from file (with matplotlib) to avoid issues with multiple
    y axes in gmt...

    :param welldir:
    :return:
    """
    well_dir = '/home/chet/data/mrp_data/well_data/flow_rates/July_2017_final'
    well_fs = glob('{}/*_all.csv'.format(well_dir))
    well_fs.sort()
    colors = cycle(['darkturquoise', 'purple', 'lightblue', 'darkred'])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax2 = ax.twinx()
    for f in well_fs:
        well = f.split('/')[-1].split('_')[0]
        param = f.split('/')[-1].split('_')[-2]
        dtos = []
        vals = []
        if well in well_list and param in params:
            with open(f, 'r') as f:
                for line in f:
                    ln = line.split()
                    dtos.append(
                        UTCDateTime().strptime(ln[0],
                                               '%Y-%m-%dT%H:%M:%S').datetime)
                    vals.append(float(ln[-1].rstrip('\n')))
            if param == 'flow':
                ln = ax.plot(dtos, vals, label='{}: Flow (t/h)'.format(well),
                             color=next(colors), linewidth=1.0)
            elif param == 'WHP':
                ln = ax2.plot(dtos, np.array(vals) / 10.,
                              label='{}: WHP (MPa)'.format(well),
                              color=next(colors), linewidth=1.0)
    ax.set_ylim([0, 1300])
    ax2.set_ylim([0, 3.5])
    ax.set_ylabel('Flow rate (t/h)', fontsize=16)
    ax2.set_ylabel('WHP (MPa)', fontsize=16)
    ax2.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    fig.autofmt_xdate()
    ax2.tick_params(axis='x', labelsize=14)
    handles = ax.legend().get_lines()  # Grab these lines for legend
    if isinstance(ax.legend_, matplotlib.legend.Legend):
        ax.legend_.remove()  # Need to manually remove this, apparently
    handles.extend(ax2.legend().get_lines())
    plt.legend(handles=handles, loc=2, fontsize=12)
    if show:
        plt.show()
    elif outfile:
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
    return


def plot_fm_map(catalog, fm_file, color, old_cat=False):
    # Write temporary outfile for psmeca plotting
    tmp_file = '/home/chet/gmt/tmp/tmp.psmeca'
    if old_cat:
        tmp_file = '/home/chet/gmt/tmp/tmp_old.psmeca'
    format_arnold_to_gmt(fm_file, catalog, outfile=tmp_file, names=False,
                         id_type='detection')
    args = {'S': 'a2.0', 'G': color, 'C': '0.5P0.1'}
    with LibGMT() as lib:
        file_context = dummy_context(tmp_file)
        with file_context as fname:
            arg_str = ' '.join([fname, build_arg_string(args)])
            lib.call_module('psmeca', arg_str)
    return


def plot_fm_depth(catalog, center_pt, end_pt, region,
                  scale, Y, B_list, fig, old_cat):
    if old_cat:
        color = 'grey'
        tmp_file = gmt_project(old_cat, center_pt, end_pt,
                               fm_file='/home/chet/gmt/tmp/tmp_old.psmeca')
        args = {'S': 'a2.0', 'G': color, 'C': '0.5P0.1', 'J': scale,
                'R': '{}/{}/{}/{}'.format(region[0], region[1], region[2],
                                          region[3]),
                'Y': Y}
        with LibGMT() as lib:
            file_context = dummy_context(tmp_file)
            with file_context as fname:
                arg_str = ' '.join([fname, build_arg_string(args)])
                lib.call_module('psmeca', arg_str)
        color = 'blue'
        tmp_filen = gmt_project(catalog, center_pt, end_pt,
                               fm_file='/home/chet/gmt/tmp/tmp.psmeca')
        args = {'S': 'a2.0', 'G': color, 'C': '0.5P0.1'}
        with LibGMT() as lib:
            file_context = dummy_context(tmp_filen)
            with file_context as fname:
                arg_str = ' '.join([fname, build_arg_string(args)])
                lib.call_module('psmeca', arg_str)
        fig.basemap(B=B_list)
    else:
        color = 'blue'
        tmp_file = gmt_project(catalog, center_pt, end_pt,
                               fm_file='/home/chet/gmt/tmp/tmp.psmeca')
        args = {'S': 'a2.0', 'G': color, 'C': '0.5P0.1', 'J': scale,
                'R': '{}/{}/{}/{}'.format(region[0], region[1], region[2],
                                          region[3]),
                'Y': Y}
        with LibGMT() as lib:
            file_context = dummy_context(tmp_file)
            with file_context as fname:
                arg_str = ' '.join([fname, build_arg_string(args)])
                lib.call_module('psmeca', arg_str)
        fig.basemap(B=B_list)
    return

def plot_Nga_static(cat, start_pt=None, end_pt=None, secs=[], mags=[],
                    dto=None, flows=False, show=True, outfile=None,
                    old_cat=None, old_mags=None, old_secs=None, fm_file=None,
                    dd_only=True):
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
    if dd_only:
        catalog = Catalog(events=[ev for ev in cat
                                  if ev.preferred_origin().method_id])
    else:
        catalog = cat
    # Sort catalog
    catalog.events.sort(key=lambda x: x.origins[-1].time)
    # Calculate mid-point of cross_section
    if not start_pt and not end_pt:
        start_pt = [176.171, -38.517]
        end_pt = [176.209, -38.575]
    c_lon = start_pt[0] + ((end_pt[0] - start_pt[0]) / 2.)
    c_lat = start_pt[1] + ((end_pt[1] - start_pt[1]) / 2.)
    if len(secs) == 0 and len(mags) == 0 and len(catalog) > 0:
        secs, mags = catalog_arrays(catalog)
    # Scale down mags
    mags = np.array(mags) * 0.5
    # Set prefs
    with LibGMT() as lib:
        lib.call_module('gmtset', 'FONT_ANNOT_PRIMARY 10p')
        lib.call_module('gmtset', 'FORMAT_GEO_MAP ddd.xx')
        lib.call_module('gmtset', 'MAP_FRAME_TYPE plain')
        lib.call_module('gmtset', 'PS_MEDIA A3')
    region = [176.15, 176.23, -38.58, -38.51]
    # Set up figure
    fig = gmt.Figure()
    plot_background_datasets(fig, region=region)
    if not fm_file:
        plot_earthquakes_map(catalog, mags, secs, fig, old_cat, old_mags)
    else:
        if old_cat:
            # Old guys
            plot_fm_map(old_cat, fm_file, color='grey', old_cat=True)
        # New guys
        plot_fm_map(catalog, fm_file, color='blue')
    if dto:
        plot_Nga_well_map(fig, dto=dto)
    else:
        plot_Nga_well_map(fig, dto=None)
        plot_NM5_7_4()
    plot_stations(fig)
    if dto:
        # pstext the date instead of the color scale
        plot_date_text(dto)
    else:
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
    x_region = [-3.635, 3.635, -1.0, 4.5]
    if not fm_file:
        plot_earthquakes_depth(catalog, mags, secs, center_pt=(c_lon, c_lat),
                               end_pt=end_pt, region=x_region, scale=scale,
                               B_list=B_list, Y=Y, fig=fig, old_cat=old_cat,
                               old_mags=old_mags, old_secs=old_secs)
    else:
        plot_fm_depth(catalog, center_pt=(c_lon, c_lat), end_pt=end_pt,
                      region=x_region, scale=scale, Y=Y, B_list=B_list,
                      fig=fig, old_cat=old_cat)
    plot_Nga_well_depth(fig)
    if flows:
        plot_injection_rates(fig, dto)
    if show:
        fig.show(external=True)
    elif outfile:
        fig.savefig(outfile)
    return


def earthquake_video(catalog, outdir, flows=True, field='Nga', fm_file=None):
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
    last_10_cnt = []
    for i, date in enumerate(date_generator(UTCDateTime(2012, 5, 1).date,
                             catalog[-1].origins[-1].time.date)):
        dto = UTCDateTime(date)
        print('Plotting {}'.format(dto))
        start_str = 'time >= %s' % str(dto)
        end_str = 'time <= %s' % str(dto + 86400)
        day_cat = catalog.filter(start_str, end_str)
        no_evs = len(day_cat)
        counter += no_evs
        last_10_cnt.append(no_evs)
        # Plot old events as grey, color for events in last 10 days
        if i < 10:
            new_cat = catalog[:counter]
            tmp_mags_new = mags[:counter]
            tmp_secs_new = secs[:counter]
            old_cat = None
            tmp_mags_old = None
            tmp_secs_old = None
        else:
            last_10 = np.sum(last_10_cnt[-10:])
            new_cat = catalog[counter - last_10:counter]
            tmp_mags_new = mags[counter - last_10:counter]
            tmp_secs_new = secs[counter - last_10:counter]
            old_cat = catalog[:counter - last_10]
            tmp_mags_old = mags[:counter - last_10]
            tmp_secs_old = secs[:counter - last_10]
        if field == 'Nga':
            start_pt = (176.171, -38.517)
            end_pt = (176.209, -38.575)
            plot_Nga_static(new_cat, start_pt, end_pt, tmp_secs_new,
                            tmp_mags_new, flows=flows, dto=dto,
                            outfile='{}/img{}.png'.format(outdir, i + 1),
                            old_cat=old_cat, old_mags=tmp_mags_old,
                            old_secs=tmp_secs_old, fm_file=fm_file)
    return