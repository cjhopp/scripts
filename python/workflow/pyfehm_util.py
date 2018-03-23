#!/usr/bin/python

"""
Functions for running creating models and running pyFEHM simulations
"""
import fdata
import fpost
import numpy as np
import pandas as pd

from glob import glob
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt

def latlon_2_grid(x, y, z, origin):
    """
    Conversion from lat/lon to grid coords in meters
    :param x: Longitude
    :param y: Latitude
    :param z: Depth (untouched)
    :param origin: Origin pt for the cartesian grid (lon, lat)
    :return:
    """
    new_y = (y - origin[1]) * 111111
    new_x = (x - origin[0]) * (111111 * np.cos(origin[1] * (np.pi/180)))
    return new_x, new_y, z

def feedzones_2_rects(fz_file_pattern, surf_loc=None):
    """
    Take feedzone files and convert them into node-bounding boxes in grid
    coordinates
    :param fz_file_pattern: Glob string to find all feedzone files
    :param surf_loc: Optional, sets surfact grid location of straight hole
        as location for all well nodes (forced perfectly straight hole)
    :return:
    """
    grid_origin = (176.16, -38.578, 0)

    fz_rects = []
    fz_files = glob(fz_file_pattern)
    for fz_file in fz_files:
        lines = []
        with open(fz_file, 'r') as f:
            for ln in f:
                lines.append(ln.split(' '))
        rect = []
        # Make sure to convert from mCT to elevation, idiot
        for pt_ln in [lines[0], lines[-1]]:
            if not surf_loc:
                rect.append(latlon_2_grid(float(pt_ln[0]),
                                          float(pt_ln[1]),
                                          (float(pt_ln[3]) * -1.) + 350.,
                                          origin=grid_origin))
            else:
                rect.append([surf_loc[0], surf_loc[1],
                             (float(pt_ln[3]) * -1.) + 350.])
        fz_rects.append(rect)
    return fz_rects

def run_initial_conditions(dat):
    # initial run allowing equilibration
    dat.tf = 365.25
    dat.dtmax = dat.tf
    # rsto specify the initial conditions file
    dat.files.rsto = '{}/NM08_INCON.ini'.format(dat.work_dir)
    dat.run('{}/NM08_INPUT.dat'.format(dat.work_dir),
            use_paths=True)
    dat.incon.read('{}/NM08_INCON.ini'.format(dat.work_dir))
    dat.ti = 0.
    dat.delete(dat.preslist)
    dat.delete(
        [zn for zn in dat.zonelist if (zn.index >= 600 and zn.index <= 700)])
    return dat

def set_stress(dat):
    """
    Set the initial stress conditions
    :param dat:
    :return:
    """
    # stress parameters
    rho = 2700.
    xgrad = 0.61
    ygrad = (0.61 + 1) / 2
    dat.strs.on()
    dat.strs.bodyforce = False
    dat.add(fdata.fmacro('stressboun', zone='XMIN',
                         param=(('direction', 1), ('value', 0))))
    dat.add(fdata.fmacro('stressboun', zone='YMIN',
                         param=(('direction', 2), ('value', 0))))
    dat.add(fdata.fmacro('stressboun', zone='ZMIN',
                         param=(('direction', 3), ('value', 0))))
    dat.add(fdata.fmacro('stressboun', zone='XMAX',
                         param=(('direction', 1), ('value', 0))))
    dat.add(fdata.fmacro('stressboun', zone='YMAX',
                         param=(('direction', 2), ('value', 0))))
    dat.zone[0].poissons_ratio = 0.2
    dat.zone[0].thermal_expansion = 3.5e-5
    dat.zone[0].pressure_coupling = 1.
    # Model starts at surface, so no overburden (specified as zgrad when
    # 'calculate_vertical' set to True)
    dat.incon.stressgrad(xgrad=xgrad, ygrad=ygrad, zgrad=0,
                         calculate_vertical=True, vertical_fraction=True)
    return dat

def define_well_nodes(dat, well_file_pattern, well_name, surf_loc=None,
                      type='injection'):
    """
    Define zones of injection and production from permeable zone files
    ..Note:
    Should approximate feedzones even in deviated wells
    :param dat: pyFEHM fdata object
    :param well_file_pattern: Glob string for all files containing feedzone
        information
    :param well_name: Used for zone naming
    :param surf_loc: Optional, forces a perfectly vertical hole at the
        specified location
    :param type: Default to injection. Can also be production.
    :return:
    """
    if type == 'production': ind = 40
    else: ind = 30
    # Convert well fz files to grid coordinate rectangles
    fz_rects = feedzones_2_rects(well_file_pattern, surf_loc)
    for i, zone in enumerate(fz_rects):
        # Add buffer around rectangle
        rect = [[zone[0][0] - 0.1, zone[0][1] - 0.1, zone[0][2] + 0.1],
                [zone[1][0] + 0.1, zone[1][1] + 0.1, zone[1][2] - 0.1]]
        dat.new_zone(ind + i, name='{}_fzone_{}'.format(well_name, i),
                     rect=rect)
    return dat

def set_well_boundary(dat, excel_file, sheet_name, well_name,
                      dates, parameters=['Flow (t/h)', 'WHP (barg)'],
                      t_step='day', temp=75., decimate=False,
                      debug=0):
    """
    Set the boundary conditions for flow in a single well. Currently assumes
    uniform flow across all feedzones...??
    :param dat: pyFEHM data object
    :param flow_xls: Excel sheet with flow rates
    :param sheet_name: Sheet we want in the excel file
    :param well_name: Name of well in this sheet
    :param dates: list of start and end date obj for truncating flow data
    :param parameters: List specifying which well parameters to set for the
        model
    :param t_step: Specify the time step for data
    :param decimate: If not False, will decimate data by factor specified
    :return:
    """
    # Read in excel file
    df = pd.read_excel(excel_file, header=[0, 1], sheetname=sheet_name)
    # All flow info is local time
    df.index = df.index.tz_localize('Pacific/Auckland')
    print('Flow data tz set to: {}'.format(df.index.tzinfo))
    # Truncate to desired dates
    start = dates[0]
    end = dates[1]
    df = df.truncate(before=start, after=end)
    dtos = df.xs((well_name, parameters[0]), level=(0, 1),
                 axis=1).index.to_pydatetime()
    flows = df.xs((well_name, parameters[0]), level=(0, 1), axis=1)
    # Convert t/h to kg/sec (injection is negative)
    flows /= -3.6
    flow_list = flows.values.tolist()
    # Flatten this for some dumb (self-imposed) reason
    flow_list = [lst[0] for lst in flow_list]
    pres = df.xs((well_name, parameters[1]), level=(0, 1), axis=1)
    # Convert bar to MPa
    pres /= 10
    pres_list = pres.values.tolist()
    pres_list = [lst[0] for lst in pres_list]
    # Convert dtos to elapsed time
    if t_step == 'min':
        times = [(dt - dtos[0]).total_seconds() / 60. for dt in dtos]
    elif t_step == 'day':
        times = [(dt - dtos[0]).total_seconds() / 86400. for dt in dtos]
    well_zones = [key for key in dat.zone.keys() if type(key) == str]
    zone_list = [zone for zone in well_zones if zone.startswith(well_name)]
    if decimate:
        times = times[::decimate]
        flow_list = flow_list[::decimate]
        pres_list = pres_list[::decimate]
    if debug > 0:
        plt.plot(times, flow_list)
        plt.plot(times, pres_list)
        plt.show()
    temps = [temp for i in range(len(flow_list))]
    # Create boundary
    flow_list.insert(0, 'dsw')
    pres_list.insert(0, 'pw')
    temps.insert(0, 'ft')
    bound = fdata.fboun(type='ti_linear', zone=zone_list, times=times,
                        variable=[flow_list, pres_list, temps])
    # Add it
    dat.add(bound)
    return dat

def set_fgeneral(dat, zonelist, general_dict):
    """
    General function to set an fgeneral macro for non-implemented
    :param dat:
    :param zonelist:
    :param general_dict:
    :return:
    """

    return

def set_permmodel(dat, zonelist, index, permmodel_dict):
    """
    Setup a permeability model as in Dempsey et al., 2013 for Desert Peak

    :type dat: pyfehm.fdata
    :param dat: fdata object to add this model to
    :type zonelist: list or str
    :param zonelist: Zone(s) to apply the model to
    :type index: int
    :param index: Index of the stress-permeability model to use
    :type permmodel_dict: dict
    :param permmodel_dict: Dictionary of parameters for this particular model.
        All of the available parameters must be assigned (see fdata.py lines
        70-90). No defaults will be set.
    :return:
    """
    perm_mod = fdata.fmodel('permmodel', index=index,
                            zonelist=zonelist)
    # Set required permeability
    for key, value in permmodel_dict.iteritems():
        perm_mod.param[key] = value
    dat.add(perm_mod)
    return

def make_NM08_grid(work_dir):
    base_name = 'NM08'
    dat = fdata.fdata(work_dir=work_dir)
    dat.files.root = base_name
    pad_1 = [1500., 1500.]
    # Symmetric grid in x-y
    base = 3
    dx = pad_1[0]
    x1 = dx ** (1 - base) * np.linspace(0, dx, 10) ** base
    X = np.sort(list(pad_1[0] - x1) + list(pad_1[0] + x1)[1:] + [pad_1[0]])
    # If no. z nodes > 100, temperature_gradient will not like it...
    surface_deps = np.linspace(350, -750, 10)
    cap_grid = np.linspace(-750, -1200, 10)
    perm_zone = np.linspace(-1200., -2100., 60)
    lower_reservoir = np.linspace(-2100, -3000, 20)
    Z = np.sort(list(surface_deps) + list(cap_grid) + list(perm_zone)
                + list(lower_reservoir))
    dat.grid.make('{}_GRID.inp'.format(base_name), x=X, y=X, z=Z,
                  full_connectivity=True)
    grid_dims = [3000., 3000.] # 5x7x5 km grid
    # Geology time
    dat.new_zone(1, 'suface_units', rect=[[-0.1, -0.1, 350 + 0.1],
                                          [grid_dims[0] + 0.1,
                                           grid_dims[1] + 0.1,
                                           -750 - 0.1]],
                 permeability=[1.e-15, 1.e-15, 1.e-15], porosity=0.1,
                 density=2477, specific_heat=800., conductivity=2.2)
    dat.new_zone(2, 'clay_cap', rect=[[-0.1, -0.1, -750],
                                      [grid_dims[0] + 0.1,
                                       grid_dims[1] + 0.1,
                                       -1200 - 0.1]],
                 permeability=1.e-18, porosity=0.01, density=2500,
                 specific_heat=1200., conductivity=2.2)
    return dat

def reservoir_params(dat, temp_file, reservoir_dict, show=False):
    grid_dims = [3000., 3000.] # 5x7x5 km grid
    # Intrusive properties from Cant et al., 2018
    dat.new_zone(3, 'tahorakuri', rect=[[-0.1, -0.1, -1200.],
                                        [grid_dims[0] + 0.1,
                                         grid_dims[1] + 0.1,
                                         -2300 - 0.1]],
                 permeability=reservoir_dict['tahorakuri']['perms'],
                 porosity=0.1, density=2500, specific_heat=1200.,
                 conductivity=2.2, youngs_modulus=40., poissons_ratio=0.26)
    # Intrusive properties from Cant et al., 2018
    dat.new_zone(4, 'intrusive', rect=[[-0.1, -0.1, -2300.],
                                       [grid_dims[0] + 0.1,
                                        grid_dims[1] + 0.1,
                                        -3000 - 0.1]],
                 permeability=reservoir_dict['intrusive']['perms'],
                 porosity=0.03, density=2500, specific_heat=1200.,
                 conductivity=2.2, youngs_modulus=33., poissons_ratio=0.33)
    # Assign temperature profile
    dat.temperature_gradient(temp_file, hydrostatic=0.1,
                             offset=0.1, first_zone=600,
                             auxiliary_file='NM08_temp.macro')
    dat.zone['ZMAX'].fix_pressure(0.1) # Surface pressure set to 1 atm
    # dat.zone['intrusive'].fix_temperature(260.)
    # dat.zone['ZMIN'].fix_temperature(270.)
    if show:
        print('Launching paraview')
        dat.paraview()
    return dat

def model_run(dat, param_dict, verbose=True, diagnostic=False):
    # run simulation
    dat.ti = 0
    dat.tf = 40
    dat.dtn = 5000
    dat.dtmax = param_dict['dtmax']
    dat.cont.variables.append(
        ['xyz', 'pressure', 'liquid', 'temperature', 'stress', 'displacement',
         'permeability'])
    dat.cont.format = 'surf'
    dat.cont.time_interval = param_dict['output_interval']
    dat.hist.variables.append(['temperature', 'pressure', 'flow', 'zfl'])
    dat.hist.time_interval = param_dict['output_interval']
    dat.hist.format = 'surf'
    dat.hist.zonelist = [dat.zone[30], dat.zone[31], dat.zone[32],
                         dat.zone[33], dat.zone['tahorakuri']]
    # Now run this thing
    dat.run('{}/{}_INPUT.dat'.format(dat.work_dir, dat.files.root),
            use_paths=True, files=['hist', 'outp', 'check'], verbose=verbose,
            diagnostic=diagnostic)
    return dat

def NM08_model_loop(root, run_dict, res_dict, machine, decimate=100):
    """
    Function to run multiple models in parallel with differing perms (for now)
    on sgees018
    :return:
    """
    # Making the directory
    perm_xx, perm_yy, perm_zz = res_dict['tahorakuri']['perms']
    if machine == 'laptop':
        fz_file_pat = '/home/chet/gmt/data/NZ/wells/feedzones/' \
                      'NM08_feedzones_?.csv'
        T_file = '/home/chet/data/mrp_data/Steve_Sewell_MRP_PhD_Data/' \
                 'Natural_State_Temperatures/NM08_profile_pyfehm_comma.txt'
        excel_file = '/home/chet/data/mrp_data/well_data/flow_rates/' \
                     'July_2017_final/Merc_Ngatamariki.xlsx'
    elif machine == 'server':
        fz_file_pat = '/Users/home/hoppche/data/merc_data/wells/' \
                      'NM08_feedzones_?.csv'
        T_file = '/Users/home/hoppche/data/merc_data/temps/' \
                 'NM08_profile_pyfehm_comma.txt'
        excel_file = '/Users/home/hoppche/data/merc_data/flows/' \
                     'Merc_Ngatamariki.xlsx'
    # Make the directory for this object
    print('Making grid')
    # Extract just floats and exponent from perms
    px = '{:.1E}'.format(perm_xx)[:3]
    py = '{:.1E}'.format(perm_yy)[:3]
    pz = '{:.1E}'.format(perm_zz)[:3]
    exp = '{:.1E}'.format(perm_xx)[-4:]
    work_dir = '{}/perms_{}x{}x{}{}'.format(root, px, py, pz, exp)
    dat = make_NM08_grid(work_dir=work_dir)
    print('Assigning reservoir parameters')
    dat = reservoir_params(dat, temp_file=T_file, reservoir_dict=res_dict,
                           show=False)
    print('Defining well nodes')
    dat = define_well_nodes(
        dat, well_file_pattern=fz_file_pat,
        well_name='NM08', type='injection', surf_loc=[1500., 1500.])
    print('Running initial condition')
    dat = run_initial_conditions(dat)
    dat = set_well_boundary(
        dat, excel_file=excel_file, sheet_name='NM08 Stimulation',
        well_name='NM08', dates=[datetime(2012, 6, 7), datetime(2012, 7, 12)],
        t_step='day', decimate=decimate, debug=0)
    dat = set_stress(dat)
    dat = model_run(dat, run_dict)
    return dat

def model_multiprocess(reservoir_dicts, root, run_dict, machine='laptop',
                       parallel=False):
    if parallel:
        cores = len(reservoir_dicts)
        pool = Pool(processes=cores)
        res = [pool.apply_async(NM08_model_loop, (root, run_dict, res_dict,
                                                  machine))
               for i, res_dict in enumerate(reservoir_dicts)]
    else:
        for r_dict in reservoir_dicts:
            NM08_model_loop(root, run_dict, r_dict, machine)
    return

def process_output(outdirs, contour=True, history=False, elevation=-1300):
    for outdir in outdirs:
        if contour:
            cont = fpost.fcontour('{}/*sca_node.csv'.format(outdir),
                                  latest=True)
            # Slice plots of T, P and stress
            cont.slice_plot(
                save='{}/T_slice_{}.png'.format(outdir, elevation), cbar=True,
                levels=10, slice=['z', elevation], divisions=[150, 150],
                variable='T', xlims=[0, 3000], ylims=[0, 3000],
                title='NM08 T slice: {} m'.format(elevation))
            cont.slice_plot(
                save='{}/P_slice_{}.png'.format(outdir, elevation), cbar=True,
                levels=10, slice=['z', elevation], divisions=[150, 150],
                variable='P', xlims=[0, 3000], ylims=[0, 3000],
                title='NM08 P slice: {} m'.format(elevation))
            cont.slice_plot(
                save='{}/strs_xx_slice_{}.png'.format(outdir, elevation),
                cbar=True, levels=10, slice=['z', elevation],
                divisions=[150, 150], variable='strs_xx', xlims=[0, 3000],
                ylims=[0, 3000],
                title='NM08 strs_xx slice: {} m'.format(elevation))
            # Cutaway plots
            cont.cutaway_plot(
                save='{}/T_cutaway_{}.png'.format(outdir, elevation),
                cbar=True, levels=np.linspace(240, 270, 10),
                variable='T', xlims=[1500, 2000], ylims=[1500, 2000],
                zlims=[-2000, -1200], grid_lines='k:',
                title='NM08 T cutaway / $^o$C'.format(elevation))
            cont.cutaway_plot(
                save='{}/P_cutaway_{}.png'.format(outdir, elevation),
                cbar=True, levels=np.linspace(9, 15, 10),
                variable='P', xlims=[1500, 2000], ylims=[1500, 2000],
                zlims=[-2000, -1200], grid_lines='k:',
                title='NM08 P cutaway / MPa'.format(elevation))
            cont.cutaway_plot(
                save='{}/strs_xx_cutaway_{}.png'.format(outdir, elevation),
                cbar=True, levels=np.linspace(20, 40, 10),
                variable='strs_xx', xlims=[1500, 2000], ylims=[1500, 2000],
                zlims=[-2000, -1200], grid_lines='k:',
                title='NM08 strs_xx cutaway / MPa'.format(elevation))
            # Profile plots
            cont.profile_plot(
                save='{}/T_prof_from_NM08_{}.png'.format(outdir, elevation),
                profile=np.array([[1500, 1500, elevation],
                                  [3000, 3000, elevation]]),
                variable='T', ylabel='Temperature $^o$C',
                title='Temp from well', color='g',
                marker='o--', method='linear')
            cont.profile_plot(
                save='{}/P_prof_from_NM08_{}.png'.format(outdir, elevation),
                profile=np.array([[1500, 1500, elevation],
                                  [3000, 3000, elevation]]),
                variable='P', ylabel='Pressure MPa',
                title='Pressure from well', color='g',
                marker='o--', method='linear')
            cont.profile_plot(
                save='{}/H_stress_prof_from_NM08_{}.png'.format(outdir,
                                                                elevation),
                profile=np.array([[1500, 1500, elevation],
                                  [3000, 3000, elevation]]),
                variable='strs_xx', ylabel='Stress_xx MPa',
                title='Stress from well', color='g',
                marker='o--', method='linear')
        if history:
            continue
        plt.close('all')
    return