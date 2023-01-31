import numpy as np
import matplotlib.pyplot as plt
from obspy import Trace
from scipy import signal
from scipy.signal import spectrogram
from obspy import read


def piezo_dataread(numfile, datfile):

    # This little loop opens the .DAT file that is to be used for calibration of shot data
    calibration = {}
    fields = {'XINCR', 'XZERO', 'YZERO', 'YMULT'}
    marker = []
    pts = []
    with open(datfile, 'r') as f:
        for ln in f:
            ln = ln.strip()
            line = ln.split(',')
            if line[0].startswith('Trace'):
                calibration[line[0]] = {}
                marker.append(line[0])
                continue
            if line[-1].strip() in fields:
                calibration[marker[-1]][line[-1].strip()] = float(line[0])
                continue
            if line[-1].endswith('array'):
                pts.append(int(line[0]))
                continue
    # Read in data (of known length)
    print(pts)
    lengths = list(set(pts))
    try:
        length = lengths[0]
    except IndexError:
        print('Not all traces are the same length. Check these files.')
        return
    # This line reads in and reshapes the binary data file
    array = np.fromfile(numfile, dtype=np.int16).reshape(-1, length)
    # Create output dictionary
    output_dict = {}
    print(calibration)
    for trace_name, calib_dict in calibration.items():
        row = int(trace_name.split()[1]) - 1
        print(row)
        tvec = (np.arange(array.shape[-1]) *
                calib_dict['XINCR']) + calib_dict['XZERO']
        trace = (array[row] * calib_dict['YMULT']) + calib_dict['YZERO']
        trace = np.vstack((tvec * 1e6, trace))
        output_dict[trace_name] = trace
    return output_dict



