#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import circmean

def MFAST_to_stats(file_path, bin_size=3, overlap=1, debug=1):
    """
    Functions to parse MFAST output and turn it into useful statistics

    Dict will be keyed like this:
        results[julday][parameter] = [values]

    Where julday is the julian day of the measurement and parameter is
    either 'fast' or 'Dfast' for the fast direction or the error
    respectively.

    :param file_path: Path to the MFAST output file
    :param bin_size: Window size in days
    :param overlap: Window overlap in days
    :return:
    """

    # While file is open, read the important information into a dict
    with open(file_path, 'r') as f:
        # Initialize main nested dictionary
        results = {int(line.rstrip().split(',')[6].split('.')[0]): {}
                   for line in f}
    # Go back to start of file and loop again
    with open(file_path, 'r') as f:
        # Now populate the nested dictionaries with the fast and Dfast values
        # for each day
        for line in f:
            line = line.rstrip().split(',')
            # If you were hoping to run this on a dataset that bridged two
            # or more years, then the julday key should be converted to a
            # Python datetime object
            # year = line[5]
            julday = int(line[6].split('.')[0])
            Dfast = float(line[-19])
            fast = float(line[-20])
            if fast < 0:
                fast += 180
            # Check if the 'fast' key is already in the dictionary
            if 'fast' in results[julday]:
                results[julday]['fast'].append(fast)
            # If not, create it and make its value a list with fast in it
            else:
                results[julday]['fast'] = [fast]
            # Same thing for Dfast
            if 'Dfast' in results[julday]:
                results[julday]['Dfast'].append(Dfast)
            else:
                results[julday]['Dfast'] = [Dfast]
    avg = []
    start = min(results.keys())
    end = max(results.keys())
    # For convenience, create empty dicts for gaps in julday keys
    for jday in range(start, end):
        if jday not in results:
            results[jday] = {'fast': [], 'Dfast': []}
    if debug > 0:
        print('Results dictionary:\n{}'.format(results))
    # Now do a moving average
    # Only really relevant if you have consecutive days in your dataset
    # Loop over all days by increments of (bin_size - overlap)
    step = bin_size - overlap
    if step == 0:
        step = 1
    for j in range(start, end, step):
        window_fast = []
        # Take all measurements for all days in the window
        for k in range(bin_size):
            window_fast.extend(results[j + k]['fast'])
        # Append the mean for this window to the list 'avg'
        avg.append(circmean(window_fast))
    print(avg)
    if debug > 1:
        fig, ax = plt.subplots()
        ax.plot([d for d in range(start, end, step)], avg)
        plt.title('Average fast direction')
        ax.set_xlabel('Julian day')
        ax.set_ylabel('Fast direction')
        plt.show()
        plt.close()
    return np.rad2deg(avg)