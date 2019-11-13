#!/usr/bin/env python

"""
Take indices for groups from clustering.py and extract these entries from hypoDD
input files
"""

import csv
import numpy as np
from obspy import read_events

# Create generator expression
def group_by_heading(file):
    buffer = []
    for line in file:
        if line.startswith( "#" ):
            if buffer: yield buffer
            buffer = [ line ]
        else:
            buffer.append( line )
    yield buffer


# Need to shrink size of catS to allow hypoDD to actually run
# catS_rand = [catS[i] for i in np.random.choice(range(len(catS)), 8000, replace=False)]
# Handle each file in separate loops, shouldn't be long
## dt.cc
def separate_hypoDD_input(cat, rootdir):
    # This seperates the events into N and S for hypoDD stupid memory limitations
    catN = []
    catS = []
    for i, ev in enumerate(cat):
        if ev.preferred_origin().latitude < -38.580367:
            catS.append(str(i))
        else:
            catN.append(str(i))
    with open(rootdir+ 'dt.cc', 'rb') as dt_cc_in:
        with open(rootdir + 'dt.cc_N', 'wb') as dt_cc_N:
            with open(rootdir + 'dt.cc_S', 'wb') as dt_cc_S:
                for head_and_lines in group_by_heading(dt_cc_in):
                    if head_and_lines[0].split()[1] in catN and head_and_lines[0].split()[2] in catN:
                        for line in head_and_lines:
                            dt_cc_N.write(line)
                    elif head_and_lines[0].split()[1] in catS and head_and_lines[0].split()[2] in catS:
                        for line in head_and_lines:
                            dt_cc_S.write(line)
    ## dt.ct
    with open(rootdir + 'dt.ct') as dt_ct_in:
        with open(rootdir + 'dt.ct_N', 'wb') as dt_ct_N:
            with open(rootdir + 'dt.ct_S', 'wb') as dt_ct_S:
                for head_and_lines in group_by_heading(dt_ct_in):
                    if head_and_lines[0].split()[1] in catN and head_and_lines[0].split()[2] in catN:
                        for line in head_and_lines:
                            dt_ct_N.write(line)
                    elif head_and_lines[0].split()[1] in catS and head_and_lines[0].split()[2] in catS:
                        for line in head_and_lines:
                            dt_ct_S.write(line)
    ## event.dat
    with open(rootdir + 'event.dat', 'rb') as event_in:
        with open(rootdir + 'event_N.dat', 'wb') as event_N_out:
            with open(rootdir + 'event_S.dat', 'wb') as event_S_out:
                for line in event_in:
                    if line.split()[-1] in catN:
                        event_N_out.write(line)
                    elif line.split()[-1] in catS:
                        event_S_out.write(line)
    return

