#!/usr/bin/python

"""
Funtions for working with various stress inversion codes
e.g. SATSI, STRESSINVERSE

TODO Migrate funcs for John/Richards code as well
"""

def cmt_to_stressinverse(input, output):
    """Parse CMT from NCEDC query to stressinverse input file"""
    sdrs = []
    with open(input, 'r') as f:
        for ln in f:
            if ln.startswith('Fault plane'):
                line = ln.split('=')
                strike = line[1].split()[0]
                dip = line[2].split()[0]
                rake = line[-1].split()[0]
                sdrs.append([strike, dip, rake])
                next(f)
    with open(output, 'w') as of:
        for sdr in sdrs:
            of.write(' '.join(sdr) + '\n')
    return
