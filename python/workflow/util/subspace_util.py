#!/usr/bin/python
"""
Helper functions for subspace detectors
"""
def date_generator(start_date, end_date):
    # Generator for date looping
    from datetime import timedelta
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

def grab_day_wavs(wav_dirs, dto, stachans):
    # Helper to recursively crawl paths searching for waveforms for a dict of
    # stachans for one day
    import os
    import fnmatch
    from itertools import chain
    from obspy import read, Stream

    st = Stream()
    wav_files = []
    for path, dirs, files in chain.from_iterable(os.walk(path)
                                                 for path in wav_dirs):
        print('Looking in %s' % path)
        for sta, chans in iter(stachans.items()):
            for chan in chans:
                for filename in fnmatch.filter(files,
                                               '*.%s.*.%s*%d.%03d'
                                                       % (
                                               sta, chan, dto.year,
                                               dto.julday)):
                    wav_files.append(os.path.join(path, filename))
    print('Reading into memory')
    for wav in wav_files:
        st += read(wav)
    st.merge(fill_value='interpolate')
    print('Checking for trace length. Removing if too short')
    rm_trs = []
    for tr in st:
        if len(tr.data) < (86400 * tr.stats.sampling_rate * 0.8):
            rm_trs.append(tr)
        if tr.stats.starttime != dto:
            print('Trimming trace %s.%s with starttime %s to %s'
                  % (tr.stats.station, tr.stats.channel,
                     str(tr.stats.starttime), str(dto)))
            tr.trim(starttime=dto, endtime=dto + 86400,
                    nearest_sample=False)
    if len(rm_trs) != 0:
        print('Removing traces shorter than 0.8 * daylong')
        for tr in rm_trs:
            st.remove(tr)
    else:
        print('All traces long enough to proceed to dayproc')
    return st

def plot_frac_capture(detector):
    """
    Plot the fractional energy capture of a subspace detector for its design
    set. Include as part of Detector class at some point.
    :param detector:
    :return:
    """
    import numpy as np
    import scipy
    import matplotlib.pyplot as plt

    sigma = detector.sigma[0]
    v = detector.v[0]
    u = detector.u[0]
    sig = scipy.linalg.diagsvd(sigma, max(u.shape), max(v.shape))
    A = np.dot(sig, v)
    if detector.dimension > max(v.shape) or detector.dimension == np.inf:
        dim = max(v.shape)
    else:
        dim = detector.dimension
    fig, ax = plt.subplots()
    av_fc_dict = {i: [] for i in range(dim)}
    for ai in A.T:
        fcs = []
        for j in range(dim):
            av_fc_dict[j].append(float(np.dot(ai[:j].T, ai[:j])))
            fcs.append(float(np.dot(ai[:j].T, ai[:j])))
        ax.plot(fcs, color='grey')
    avg = [np.average(dim[1]) for dim in av_fc_dict.items()]
    ax.plot(avg, color='red', linewidth=3.)
    plt.show()
    return

def rewrite_subspace(detector, outfile):
    """
    Rewrite old subspace with U and V matrices switched
    :param detector:
    :return:
    """
    import copy
    from eqcorrscan.core.subspace import Detector

    new_u = copy.deepcopy(detector.v)
    new_v = copy.deepcopy(detector.u)
    final_u = [u.T for u in new_u]
    final_v = [v.T for v in new_v]
    final_data = copy.deepcopy(final_u)
    new_det = Detector(name=detector.name, sampling_rate=detector.sampling_rate,
                       multiplex=detector.multiplex, stachans=detector.stachans,
                       lowcut=detector.lowcut, highcut=detector.highcut,
                       filt_order=detector.filt_order, data=final_data,
                       u=final_u,sigma=detector.sigma,v=final_v,
                       dimension=detector.dimension)
    new_det.write(outfile)
    return

def get_nullspace(wav_dirs, start, end, n):
    """
    Function to grab a random sample of data from our dataset, check that
    it doesn't contain amplitude spikes (STA/LTA?), then feed it to subspace
    threshold calculation
    :type wav_dir: str
    :param wav_dir: Where the wavs live
    :type start: obspy.core.event.UTCDateTime
    :param start: Start of range from which to draw random samples
    :type end: obspy.core.event.UTCDateTime
    :param end: End of range for random samples
    :type
    :return: list of obspy.core.stream.Stream
    """
    import numpy as np

    range = (end - start).days  # Number of days in range
    # Take a random sample of days since start of range
    rands = np.random.choice(range, size=n, replace=False)
    dtos = [start + (86400 * rand) for rand in rands]
    for dto in dtos:
        day_wavs = []
    return