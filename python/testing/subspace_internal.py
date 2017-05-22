#!/usr/bin/python
"""
Python and cython functions for subspace internals
for testing purposes
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
cimport numpy as np
import cython
from scipy.linalg.cython_blas cimport ddot
from scipy.linalg.blas import sgemv, sdot

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def det_statistic(float[:,:] detector,
                  float[:] data,
                  size_t inc):
    cdef size_t i, datamax = data.shape[0]
    cdef size_t ulen = detector.shape[0]
    cdef size_t stat_imax = (datamax // inc) - (ulen // inc) + 1
    cdef size_t dat_imax = (datamax - ulen) + 1
    cdef float[:] stats = np.zeros(dat_imax, dtype=DTYPE)
    cdef float[:,:] uut = np.dot(detector, detector.T)
    # Actual loop after static typing
    for i in range(0, dat_imax, inc):
        xp = np.dot(data[i:i+ulen].T, np.dot(uut, data[i:i+ulen]))
        xt = np.dot(data[i:i+ulen].T, data[i:i+ulen])
        stats[i] = (xp / xt)
    # Downsample stats
    stats = stats[::inc]
    # Cope with case of errored internal loop
    if np.all(np.isnan(stats)):
        return np.zeros(stat_imax, dtype=DTYPE)
    else:
        return np.array(stats)


def det_stat(detector, data, inc):
    ulen = detector.shape[0]
    datamax = data.shape[0]
    dat_imax = datamax - ulen + 1
    stats = np.zeros(dat_imax, dtype=np.float32)
    uut = np.dot(detector, detector.T)
    # gemm = get_blas_funcs('gemm', [uut, uut.T])
    for i in range(0, dat_imax, inc):
        xp2 = sgemv(1, uut, data[i:i + ulen])
        xp = sdot(data[i:i+ulen].T, xp2)
        xt = sdot(data[i:i+ulen].T, data[i:i+ulen])
        stats[i] = (xp / xt)
    stats = stats[::inc]
    if np.all(np.isnan(stats)):
        return np.zeros(stat_imax, dtype=DTYPE)
    else:
        return np.array(stats)


