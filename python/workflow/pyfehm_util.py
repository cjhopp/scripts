#!/usr/bin/python

"""
Functions for running creating models and running pyFEHM simulations
"""
from fdata import *  # Gross
from fpost import *
from glob import glob

import pandas as pd

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

def run_initial_conditions(dat, root):
    # initial run allowing equilibration
    dat.files.rsto = '{}_INCON.ini'.format(root)
    dat.tf = 365.25
    dat.dtmax = dat.tf
    dat.cont.variables.append(
        ['xyz', 'pressure', 'temperature', 'stress', 'permeability'])
    dat.run(root + '_INPUT.dat', use_paths=True)
    dat.incon.read('{}/{}_INCON.ini'.format(dat.work_dir, root))
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
        print(rect)
        dat.new_zone(ind + i, name='{}_fzone_{}'.format(well_name, i),
                     rect=rect)
    return dat

def set_well_boundary(dat, excel_file, sheet_name, well_name,
                      dates, parameters=['Flow (t/h)', 'WHP (barg)']):
    """
    Set the boundary conditions for flow in a single well. Currently assumes
    uniform flow across all feedzones...??
    :param dat: pyFEHM data object
    :param flow_xls: Excel sheet with flow rates
    :param sheet_name: Sheet we want in the excel file
    :param well_name: Name of well in this sheet
    :param dates: list of start and end date obj for truncating flow data
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
    # Convert t/h to kg/sec
    flows /= 3.6
    flow_list = flows.values.tolist()
    flow_list.insert(0, 'dsw')
    pres = df.xs((well_name, parameters[1]), level=(0, 1), axis=1)
    # Convert bar to MPa
    pres /= 10
    pres_list = pres.values.tolist()
    pres_list.insert(0, 'pw')
    # Convert dtos to elapsed minutes
    mins = [(dt - dtos[0]).seconds / 60. for dt in dtos]
    mins.insert(0, 'min')
    well_zones = [key for key in dat.zone.keys() if type(key) == str]
    zone_list = [zone for zone in well_zones if zone.startswith(well_name)]
    # Create boundary
    bound = fboun(zone=zone_list, times=mins, variable=[flow_list, pres_list])
    # Add it
    dat.add(bound)
    return dat

def make_NM08_grid(temp_file, root, show=True):
    dat = fdata(work_dir='/home/chet/pyFEHM')
    dat.files.root = root
    pad_1 = [1500., 1500.]
    print('Grid location of pad 1:\n{}'.format(pad_1))
    grid_dims = [3000., 3000.] # 5x7x5 km grid
    # Symmetric grid in x-y
    base = 3
    dx = pad_1[0]
    x1 = dx ** (1 - base) * np.linspace(0, dx, 10) ** base
    X = np.sort(list(pad_1[0] - x1) + list(pad_1[0] + x1)[1:] + [pad_1[0]])
    surface_deps = np.linspace(350, -750, 10)
    cap_grid = np.linspace(-750, -1200, 10)
    perm_zone = np.linspace(-1200., -2100., 90)
    lower_reservoir = np.linspace(-2100, -3000, 20)
    Z = np.sort(list(surface_deps) + list(cap_grid) + list(perm_zone)
                + list(lower_reservoir))
    dat.grid.make('{}_GRID.inp'.format(root), x=X, y=X, z=Z)
    # Assign temperature profile
    dat.temperature_gradient(temp_file, hydrostatic=0.1,
                             offset=0., first_zone=600,
                             auxiliary_file='temp_NM08.macro')
    # Geology time
    # Assign default params for boundary zones
    dat.zone[0].permeability = 1.e-15
    dat.zone[0].permeability = 1.e-15
    dat.zone[0].porosity = 0.1
    rho = 2477.
    dat.zone[0].density = rho
    dat.zone[0].specific_heat = 800.
    dat.zone[0].conductivity = 2.7
    dat.new_zone(1, 'suface_units', rect=[[-0.1, -0.1, 300 - 0.1],
                                          [grid_dims[0] + 0.1,
                                           grid_dims[1] + 0.1,
                                           -750 - 0.1]],
                 permeability=[5.e-16, 8.e-16, 5.e-16], porosity=0.1,
                 density=2477, specific_heat=800., conductivity=2.7)
    dat.new_zone(2, 'clay_cap', rect=[[-0.1, -0.1, -750],
                                      [grid_dims[0] + 0.1,
                                       grid_dims[1] + 0.1,
                                       -1200 - 0.1]],
                 permeability=1.e-20, porosity=0.01, density=2477,
                 specific_heat=800., conductivity=2.7)
    # Intrusive properties from Cant et al., 2018
    dat.new_zone(3, 'tahorakuri', rect=[[-0.1, -0.1, -1200.],
                                        [grid_dims[0] + 0.1,
                                         grid_dims[1] + 0.1,
                                         -2300 - 0.1]],
                 permeability=[1.5e-16, 3.e-16, 1.5e-16], porosity=0.1,
                 density=2500, specific_heat=1200., conductivity=2.7,
                 youngs_modulus=40., poissons_ratio=0.26)
    # Intrusive properties from Cant et al., 2018
    dat.new_zone(4, 'intrusive', rect=[[-0.1, -0.1, -2300.],
                                       [grid_dims[0] + 0.1,
                                        grid_dims[1] + 0.1,
                                        -4500 - 0.1]],
                 permeability=[1.5e-18, 3.e-18, 1.5e-18], porosity=0.03,
                 density=2500, specific_heat=800., conductivity=2.7,
                 youngs_modulus=33., poissons_ratio=0.33)
    # Assign temperature profile
    dat.temperature_gradient(temp_file, hydrostatic=0.1,
                             offset=-49., first_zone=600,
                             auxiliary_file='temp.macro')
    # Set up permeability model!
    #
    dat.zone['ZMAX'].fix_pressure(0.1)
    if show:
        print('Launching paraview')
        dat.paraview()
    return dat

def make_Nga_grid(temp_file, root, show=True):
    dat = fdata(work_dir='/home/chet/pyFEHM')
    dat.files.root = root
    # Bottom left of Nga Grid
    grid_origin = (176.16, -38.578, 0)
    pad_1_latlon = (176.178, -38.533, 0) # Pad 1 (NM08, 09) point
    pad_D_latlon = (176.196, -38.564, 0) # Pad D (NM06, 10) point
    pad_1 = latlon_2_grid(pad_1_latlon[0], pad_1_latlon[1], pad_1_latlon[2],
                          grid_origin)
    pad_D = latlon_2_grid(pad_D_latlon[0], pad_D_latlon[1], pad_D_latlon[2],
                          grid_origin)
    grid_dims = [4200., 6000., 3300.] # 5x7x5 km grid
    # Setup grid vects with power law spacing towards injection regions
    base = 3
    dx1 = pad_1[0]
    dx2 = (pad_D[0] - pad_1[0]) / 2.
    dx3 = grid_dims[0] - pad_D[0]
    dy1 = pad_D[1]
    dy2 = (pad_1[1] - pad_D[1]) / 2.
    dy3 = grid_dims[1] - pad_1[1]
    x1 = dx1 ** (1 - base) * np.linspace(0, dx1, 10) ** base
    x2 = dx2 ** (1 - base) * np.linspace(0, dx2, 12) ** base
    x3 = dx3 ** (1 - base) * np.linspace(0, dx3, 10) ** base
    X = np.sort(list(pad_1[0] - x1)[:-1] + list(pad_1[0] + x2)[1:] +
                list(pad_D[0] - x2)[:-1] + list(pad_D[0] + x3)[1:])
    y1 = dy1 ** (1 - base) * np.linspace(0, dy1, 8) ** base
    y2 = dy2 ** (1 - base) * np.linspace(0, dy2, 10) ** base
    y3 = dy3 ** (1 - base) * np.linspace(0, dy3, 8) ** base
    Y = np.sort(list(pad_D[1] - y1)[:-1] + list(pad_D[1] + y2)[1:] +
                list(pad_1[1] - y2)[:-1] + list(pad_1[1] + y3)[1:])
    # Now linear spacing between different depth intervals
    surface_deps = np.linspace(350, -750, 10)
    cap_grid = np.linspace(-750, -1200, 10)
    perm_zone = np.linspace(-1200., -2100., 90)
    lower_reservoir = np.linspace(-2100, -3000, 20)
    Z = np.sort(list(surface_deps) + list(cap_grid) + list(perm_zone)
                + list(lower_reservoir))
    dat.grid.make('{}_GRID.inp'.format(root), x=X, y=Y, z=Z)
    # Geology time
    # Assign default params for boundary zones
    dat.zone[0].permeability = 1.e-15
    dat.zone[0].permeability = 1.e-15
    dat.zone[0].porosity = 0.1
    rho = 2477.
    dat.zone[0].density = rho
    dat.zone[0].specific_heat = 800.
    dat.zone[0].conductivity = 2.7
    dat.new_zone(1, 'suface_units', rect=[[-0.1, -0.1, 300 - 0.1],
                                          [grid_dims[0] + 0.1,
                                           grid_dims[1] + 0.1,
                                           -750 - 0.1]],
                 permeability=[5.e-16, 8.e-16, 5.e-16], porosity=0.1,
                 density=2477, specific_heat=800., conductivity=2.7)
    dat.new_zone(2, 'clay_cap', rect=[[-0.1, -0.1, -750],
                                      [grid_dims[0] + 0.1,
                                       grid_dims[1] + 0.1,
                                       -1500 - 0.1]],
                 permeability=1.e-20, porosity=0.01, density=2477,
                 specific_heat=800., conductivity=2.7)
    # Intrusive properties from Siratovich et al., 2018
    dat.new_zone(3, 'tahorakuri', rect=[[-0.1, -0.1, -1500.],
                                        [grid_dims[0] + 0.1,
                                         grid_dims[1] + 0.1,
                                         -2300 - 0.1]],
                 permeability=[1.5e-16, 3.e-16, 1.5e-16], porosity=0.05,
                 density=2500, specific_heat=800., conductivity=2.7,
                 youngs_modulus=40., poissons_ratio=0.23)
    # Intrusive properties from Cant et al., 2018
    dat.new_zone(4, 'intrusive', rect=[[-0.1, 4000. - 0.1, -2300.],
                                       [grid_dims[0] + 0.1,
                                        grid_dims[1] + 0.1,
                                        -3000 - 0.1]],
                 permeability=[1.5e-18, 3.e-18, 1.5e-18], porosity=0.03,
                 density=2500, specific_heat=800., conductivity=2.7,
                 youngs_modulus=33., poissons_ratio=0.33)
    # Rotokawa Andesite properties from Siratovich et al., 2016
    dat.new_zone(5, 'andesite', rect=[[-0.1, -0.1, -2300.],
                                      [grid_dims[0] + 0.1,
                                       4000., -3000 - 0.1]],
                 permeability=[1.5e-16, 3.e-16, 1.5e-16], porosity=0.03,
                 density=2300, specific_heat=800., conductivity=2.7,
                 youngs_modulus=33., poissons_ratio=0.33)
    # Torlesse greywacke properties from:
    # Rock Properties of Greywacke Basement Hosting Geothermal Reservoirs,
    # New Zealand: Preliminary Results; Mcnamara 2014
    dat.new_zone(6, 'greywacke', rect=[[-0.1, -0.1, -3000.],
                                       [grid_dims[0] + 0.1,
                                        4000., -4000 - 0.1]],
                 permeability=[1.5e-16, 1.5e-16, 1.5e-16], porosity=0.03,
                 density=2620, specific_heat=800., conductivity=2.7,
                 youngs_modulus=4., poissons_ratio=0.23)
    # Assign temperature profile
    dat.temperature_gradient(temp_file, hydrostatic=0.1,
                             offset=-49., first_zone=600,
                             auxiliary_file='temp.macro')
    dat.zone['ZMAX'].fix_pressure(0.1)
    if show:
        print('Launching paraview')
        dat.paraview()
    return dat


def model_setup(temp_file):
    """
    Create mesh and initialize PyFEHM object with zones.
    """
    root = 'NgaN'
    dat = fdata(work_dir='/home/chet/pyFEHM/NgaN')
    dat.files.root = root
    initialConditions = True
    # grid generation - 1/8 symmetry
    #  - z = vertical, well, Sv
    #  - y = parallel to fracture, SHmax
    #  - x = perpendicular to fracture, Shmin
    # variable parameters
    dx = 3.
    # three zones of stimulation
    depth = 1035. - 115.  # depth of injection point
    dim = 2000.  # dimensions of box
    # find pair of zones which are closest
    xmin, xmid, xmax = 0., 45., dim  # dimensions of domain (high res region)
    Nx, xPower = 6, 1.5
    x = (list(np.linspace(xmin, xmid, xmid / dx + 1)) +
         list(powspace(xmid, xmax, Nx + 1, xPower))[1:])
    z = (list(np.linspace(-depth, -depth + xmid, xmid / dx + 1)) +
         list(powspace(-depth + xmid, 0, Nx + 1, xPower))[1:])
    dat.grid.make('{}_GRID.inp'.format(root), x=x, y=x, z=z)
    dat.paraview() # Visualize
    # Steady-state stuff?
    # dat.work_dir = '/home/chet/pyFEHM/NgaN_steady'
    print('grid contains {} nodes'.format(len(x) ** 3))
    # define zones
    dat.new_zone(1, 'damage', rect=[[0, 0, -depth], [xmid - 0.1, xmid - 0.1,
                                                     xmid - 0.1 - depth]])
    injNode = dat.grid.node_nearest_point([0, 0, -depth])
    dat.new_zone(2, 'injection', nodelist=injNode)
    print('damage zone contains {} nodes'.format(
        len(dat.zone['damage'].nodelist)))
    # material properties
    # kx0 = 1.e-16*3.	# permeability
    # ky0 = 5.e-16*3.	# permeability
    # kz0 = 1.e-16*3.	# permeability
    # dat.zone[0].permeability = [kx0,ky0,kz0]
    dat.zone[0].permeability = 1.e-15
    dat.zone[0].porosity = 0.1
    rho = 2477.
    dat.zone[0].density = rho
    dat.zone[0].specific_heat = 800.
    dat.zone[0].conductivity = 2.7
    # initial temperature/pressure conditions
    dat.temperature_gradient(temp_file, hydrostatic=0.1,
                             offset=0., first_zone=600,
                             auxiliary_file='temp.macro')
    dat.zone['ZMAX'].fix_pressure(0.1)
    # initial run allowing equilibration
    dat.files.rsto = root + '_INCON.ini'
    dat.tf = 365.25
    dat.dtmax = dat.tf
    dat.cont.variables.append(
        ['xyz', 'pressure', 'temperature', 'stress', 'permeability'])
    if initialConditions:
        dat.run(root + '_INPUT.dat', use_paths=True)
    dat.incon.read(dat.work_dir + os.sep + root + '_INCON.ini')
    dat.ti = 0.
    dat.delete(dat.preslist)
    dat.delete(
        [zn for zn in dat.zonelist if (zn.index >= 600 and zn.index <= 700)])

    # stress parameters
    mu = 0.75
    xgrad = 0.61
    ygrad = (0.61 + 1) / 2
    dat.strs.on()
    dat.strs.bodyforce = False
    dat.add(fmacro('stressboun', zone='XMIN',
                   param=(('direction', 1), ('value', 0))))
    dat.add(fmacro('stressboun', zone='YMIN',
                   param=(('direction', 2), ('value', 0))))
    dat.add(fmacro('stressboun', zone='ZMIN',
                   param=(('direction', 3), ('value', 0))))
    dat.add(fmacro('stressboun', zone='XMAX',
                   param=(('direction', 1), ('value', 0))))
    dat.add(fmacro('stressboun', zone='YMAX',
                   param=(('direction', 2), ('value', 0))))
    dat.zone[0].youngs_modulus = 25.e3
    dat.zone[0].poissons_ratio = 0.2
    dat.zone[0].thermal_expansion = 3.5e-5
    dat.zone[0].pressure_coupling = 1.
    dat.incon.stressgrad(xgrad=xgrad, ygrad=ygrad,
                         zgrad=115. * rho * 9.81 / 1e6,
                         calculate_vertical=True, vertical_fraction=True)
    dat.incon.write(root + '_INCON.ini')
    dat.incon.read(dat.work_dir + os.sep + root + '_INCON.ini')
    nd = dat.grid.node_nearest_point([0, 0, -(930 - 115)])
    dat.work_dir = None
    return dat
