#!/usr/bin/python

"""
Functions for processing and plotting DTS data
"""

from scipy.io import loadmat


def read_struct(f):
    struct = loadmat(f, struct_as_record=False,
                     squeeze_me=True)['monthSet'].dayCell
    return struct

