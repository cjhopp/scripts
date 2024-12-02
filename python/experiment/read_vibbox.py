#!/usr/bin/python

try:
    from scipy.stats import median_absolute_deviation
except ImportError:
    from scipy.stats import median_abs_deviation as median_absolute_deviation
from obspy import Stream, Trace, UTCDateTime
from obspy.core.trace import Stats

import matplotlib.pyplot as plt


fsb_seeds = [
    'FS.B31..XNZ', 'FS.B31..XNX', 'FS.B31..XNY', 'FS.B32..XNZ', 'FS.B32..XNX', 'FS.B32..XNY', 'FS.B42..XNZ',
    'FS.B42..XNX', 'FS.B42..XNY', 'FS.B43..XNZ', 'FS.B43..XNX', 'FS.B43..XNY', 'FS.B551..XNZ', 'FS.B551..XNX',
    'FS.B551..XNY', 'FS.B585..XNZ', 'FS.B585..XNX', 'FS.B585..XNY', 'FS.B647..XNZ', 'FS.B647..XNX', 'FS.B647..XNY',
    'FS.B659..XNZ', 'FS.B659..XNX', 'FS.B659..XNY', 'FS.B748..XNZ', 'FS.B748..XNX', 'FS.B748..XNY', 'FS.B75..XNZ',
    'FS.B75..XNX', 'FS.B75..XNY', 'FS.B301..XN1', 'FS.B303..XN1', 'FS.B305..XN1', 'FS.B307..XN1', 'FS.B309..XN1',
    'FS.B310..XN1', 'FS.B311..XN1', 'FS.B312..XN1', 'FS.B314..XN1', 'FS.B316..XN1', 'FS.B318..XN1', 'FS.B320..XN1',
    'FS.B322..XN1', 'FS.B401..XN1', 'FS.B403..XN1', 'FS.B405..XN1', 'FS.B407..XN1', 'FS.B409..XN1', 'FS.B410..XN1',
    'FS.B411..XN1', 'FS.B412..XN1', 'FS.B414..XN1', 'FS.B416..XN1', 'FS.B418..XN1', 'FS.B420..XN1', 'FS.B422..XN1',
    'FS.CTrg..', 'FS.CEnc..', 'FS.PPS..', 'FS.CMon..', 'FS.B81..XN1', 'FS.B82..XN1', 'FS.B83..XN1', 'FS.B91..XN1'
]


def vibbox_read(fname, seeds, debug=0):
    """
    Read function for raw VIBbox, 32-bit binary data files

    :param fname: Name of the file to read
    :param seeds: Iterable of the SEED ids in the order stored on file
    :param debug: Debugging flag, basically to check time signal is being read
    :return:
    """
    network, stations, locations, channels = zip(*[s.split('.') for s in seeds])
    network = network[0]
    # Find channel PPS (pulse per second)
    try:
        clock_channel = np.where(np.array(stations) == 'PPS')[0][0]
    except IndexError:
        print('No PPS channel in file. Not reading')
        return
    HEADER_SIZE=4
    HEADER_OFFSET=27
    DATA_OFFSET=148
    VOLTAGE_RANGE=10  # +/- Volts
    with open(fname, "rb") as f:
        f.seek(HEADER_OFFSET, os.SEEK_SET)
        # read header
        H = np.fromfile(f, dtype=np.uint32, count=HEADER_SIZE)
        BUFFER_SIZE=H[0]
        FREQUENCY=H[1]
        NUM_OF_BUFFERS=H[2]
        no_channels=H[3]
        # read data
        f.seek(DATA_OFFSET, os.SEEK_SET)
        A = np.fromfile(f, dtype=np.uint32,
                        count=BUFFER_SIZE * NUM_OF_BUFFERS)
        try:
            A = A.reshape(int(len(A) / no_channels), no_channels)
        except ValueError as e:
            print(e)
            # File was interrupted mid-write. Return empty stream
            return Stream()
    # Sanity check on number of channels provided in yaml
    if len(channels) != no_channels:
        print('Number of channels in config file not equal to number in data')
        return
    A = A / 2**32  # Norm to 32-bit
    A *= (2 * VOLTAGE_RANGE)
    A -= VOLTAGE_RANGE  # Demean
    path, fname = os.path.split(fname)
    try:
        # Clock signal attenuated at FSB, use 70xMAD
        if 'FS' in seeds[0]:
            thresh = 500
        else:
            thresh = 30
        # Use derivative of PPS signal to find pulse start
        dt = np.diff(A[:, clock_channel])
        # Use 70 * MAD threshold
        samp_to_first_full_second = np.where(
            dt > np.mean(dt) + thresh * median_absolute_deviation(dt))[0][0]
        # Condition where PPS not recorded properly
        if samp_to_first_full_second > 101000:
            print('Cannot read time signal')
        # If we start during the time pulse, use end of pulse for timing
        if samp_to_first_full_second > 90000:
            print('Start of data is during time pulse. Using end of pulse.')
            # Negative dt
            samp_to_first_full_second = np.where(
                dt < np.mean(dt) - thresh *
                median_absolute_deviation(dt))[0][0] + 90000
        if debug > 0:
            fig, ax = plt.subplots()
            ax.plot(dt, color='r')
            ax.plot(A[:, clock_channel], color='k')
            ax.axhline(y=np.mean(dt) + thresh * median_absolute_deviation(dt),
                       color='magenta', linestyle='--')
            ax.axvline(x=samp_to_first_full_second, color='magenta',
                       linestyle='--')
            fig.text(x=0.75, y=0.75, s=samp_to_first_full_second,
                     fontsize=14)
            plt.show()
        starttime = UTCDateTime(
            np.int(fname[5:9]), np.int(fname[9:11]), np.int(fname[11:13]),
            np.int(fname[13:15]), np.int(fname[15:17]), np.int(fname[17:19]),
            np.int(1e6 * (1 - (np.float(samp_to_first_full_second) /
                               FREQUENCY))))
    except Exception as e:
        print(e)
        print('Cannot read exact time signal: ' + fname +
              '. Taking an approximate one instead')
        starttime = UTCDateTime(
            np.int(fname[5:9]), np.int(fname[9:11]), np.int(fname[11:13]),
            np.int(fname[13:15]), np.int(fname[15:17]), np.int(fname[17:19]),
            np.int(1e2 * np.int(fname[19:23])))
    # arrange it in an obspy stream
    st = Stream()
    for i, sta in enumerate(stations):
        stats = Stats()
        # stats.sampling_rate = round(H[1], 1)
        stats.delta = 1. / H[1]
        stats.npts = A.shape[0]
        stats.network = network
        stats.station = sta
        stats.channel = channels[i]
        stats.location = locations[i]
        stats.starttime = starttime
        # Create new array to avoid non-contiguous warning in obspy.core.mseed
        st.traces.append(Trace(data=np.array(A[:, i]), header=stats))
    return st