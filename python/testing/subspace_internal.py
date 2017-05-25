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

def det_stat_freq(det_time, det_freq, data_time, data_freq, inc):
    ulen = np.int32(det_time.shape[0])
    mean = np.mean(rolling_window(data_time, ulen), 1)
    mean = mean.reshape(1, len(mean))
    var = np.var(rolling_window(data_time, ulen), 1)
    var *= ulen
    det_sum = np.sum(det_time, axis=0)
    det_sum = det_sum.reshape(len(det_sum), 1)
    av_norm = np.multiply(mean, det_sum)
    freq_cor = np.multiply(det_freq, data_freq)
    #Do ifft
    iff = scipy.real(scipy.fftpack.ifft(freq_cor))[:, ulen - 1:len(data_time)] - av_norm
    result = np.sum(np.square(iff), axis=0) / var
    return result[::inc]


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _MPXDS(self, MPcon, reqlen, ssTD, ssFD, Nc, MPconFD):
    """
    Function to preform subspace detection on multiplexed data
    MPcon is time domain rep of data block, MPconFD is freq. domain,
    ssTD is time domain rep of subspace, ssFD is freq domain rep,
    Nc is the number of channels in the multiplexed stream
    """
    n = np.int32(np.shape(ssTD)[1])  # length of each basis vector
    a = pd.rolling_mean(MPcon, n)[n - 1:]  # rolling mean of data block
    b = pd.rolling_var(MPcon, n)[n - 1:]  # rolling var of data block
    b *= n  # rolling power in vector
    sum_ss = np.sum(ssTD, axis=1)  # the sum of all the subspace basis vects
    ares = a.reshape(1, len(a))  # reshaped a
    sumres = sum_ss.reshape(len(sum_ss), 1)  # reshaped sum
    av_norm = np.multiply(ares, sumres)  # to account for non 0 mean vects
    m1 = np.multiply(ssFD, MPconFD)  # fd correlation with each basis vect
    # preform inverse fft
    if1 = scipy.real(scipy.fftpack.ifft(m1))[:, n - 1:len(MPcon)] - av_norm
    result = np.sum(np.square(if1), axis=0) / b  # get detection statistcs
    return result[::Nc]  # account for multiplexing


def _getRA(self, ssTD, ssFD, st, Nc, reqlen, contrim, names, sta):
    """
    Function to make DataFrame of this datachunk with all subspaces and
    singles that act on it
    """
    cols = ['SSdetect', 'STALTA', 'TimeStamp', 'SampRate', 'MaxDS',
            'MaxSTALTA', 'Nc', 'File']
    CorDF = pd.DataFrame(index=names, columns=cols)
    utc1 = st[0].stats.starttime
    utc2 = st[0].stats.endtime
    try:
        conSt = _applyFilter(st, self.filt, self.decimate, self.dtype,
                             fillZeros=self.fillZeros)
    except Exception:
        msg = 'failed to filter %s, skipping' % st
        detex.log(__name__, msg, level='warning', pri=True)
        return None, None, None
    if len(conSt) < 1:
        return None, None, None
    sr = conSt[0].stats.sampling_rate
    CorDF.SampRate = sr
    MPcon, ConDat, TR = multiplex(conSt, Nc, returnlist=True, retst=True)
    CorDF.TimeStamp = TR[0].stats.starttime.timestamp
    if isinstance(contrim, dict):
        ctrim = np.median(contrim.values())
    else:
        ctrim = contrim

    # Trim continuous data to avoid overlap
    if ctrim < 0:
        MPconcur = MPcon[:len(MPcon) - int(ctrim * sr * Nc)]
    else:
        MPconcur = MPcon

    # get freq. domain rep of data
    rele = 2 ** np.max(reqlen.values()).bit_length()
    MPconFD = scipy.fftpack.fft(MPcon, n=rele)

    # loop through each subpsace/single and calc sd
    for ind, row in CorDF.iterrows():
        # make sure the template is shorter than continuous data else skip

        if len(MPcon) <= np.max(np.shape(ssTD[ind])):
            msg = ('current data block on %s ranging from %s to %s is '
                   'shorter than %s, skipping') % (sta, utc1, utc2, ind)
            detex.log(__name__, msg, level='warning')
            return None, None, None
        ssd = self._MPXDS(MPconcur, reqlen[ind], ssTD[ind],
                          ssFD[ind], Nc, MPconFD)
        CorDF.SSdetect[ind] = ssd  # set detection statistic
        if len(ssd) < 10:
            msg = ('current data block on %s ranging from %s to %s is too '
                   'short, skipping') % (sta, utc1, utc2, ind)
            detex.log(__name__, msg, level='warning')
            return None, None, None
        CorDF.MaxDS[ind] = ssd.max()
        CorDF.Nc[ind] = Nc
        # If an infinity value occurs, zero it.
        if CorDF.MaxDS[ind] > 1.1:
            ssd[np.isinf(ssd)] = 0
            CorDF.SSdetect[ind] = ssd
            CorDF.MaxDS[ind] = ssd.max()
        if not self.fillZeros:  # dont calculate sta/lta if zerofill used
            try:
                CorDF.STALTA[ind] = self._getStaLtaArray(
                    CorDF.SSdetect[ind],
                    self.triggerLTATime * CorDF.SampRate[0],
                    self.triggerSTATime * CorDF.SampRate[0])
                CorDF.MaxSTALTA[ind] = CorDF.STALTA[ind].max()

            except Exception:
                msg = ('failing to calculate sta/lta of det. statistic'
                       ' on %s for %s start at %s') % (sta, ind, utc1)
                detex.log(__name__, msg, level='warn')
                # else:
                # return None, None, None
    return CorDF, MPcon, ConDat
