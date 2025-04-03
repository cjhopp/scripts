#!/usr/bin/env python3


"""
Utilities for modeling wavefields with sw4 and maybe some other codes later
"""

import numpy as np


try:
    import pySW4 as sw4
except ImportError:
    print("pySW4 not found, SW4 functionality disabled")



def frac_to_ifile(frac_path, origin=(1200, -120, 360), vertical_thickness=0.2):
    """
    Write a file with frac xyz to an sw4 material interface file

    :param frac_path: path to the frac file
    :param origin: (x, y, z) origin of the model (meters HMC; elevation masl)
    :param vertical_thickness: vertical distance between the two interfaces (meters)
    :return: None
    """
    array = np.loadtxt(frac_path)
    array[:, 0] = array[:, 0] - origin[0]
    array[:, 1] = array[:, 1] - origin[1]
    array[:, 2] = array[:, 2] - origin[2]
    # Sort the array so that column 1 changes fastest, then column 2
    array = array[np.lexsort((array[:, 1], array[:, 0]))]
    # Write the first interface
    ifile1 = frac_path.replace(".txt", "_ifile1.txt")
    with open(ifile1, "w") as f:
        for i in range(len(array)):
            f.write(f"{array[i][0]} {array[i][1]} {array[i][2]} {array[i][2] - vertical_thickness}\n")
    return


def write_sw4_input(path, ifiles, materials, inventory, origin=(1200, -120, 360), dt=0.01, tmax=10.0):
    """
    Write a collab rfile for SW4
    
    :param path: path to write the input file to
    :param ifiles: list of interface files
    :param materials: list of material properties
    :param inventory: obspy Inventory object
    :param origin: (x, y, z) origin of the model (meters HMC; elevation masl)
    :param dt: time step (seconds)
    :param tmax: max time (seconds)
    :return: None
    """
    with open(path, "wa") as f:
        # Write the header
        
        f.write('')
        # Need to change lz boundary condition to 2 (sepergrid; same as all others)
        f.write('boundary_conditions lz=2\n')
    continue