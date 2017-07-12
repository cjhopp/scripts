#!/usr/bin/python

"""
Find P-S times for groups from emipirical_SVD
Trying to verify if linear stacking is valid for these events or if I need
to account for variations in station P-S times within groups when stacking
"""

import pickle
import numpy as np

with open('/home/chet/data/hierarchy_cluster.pickle', 'r') as f1:
    groups = pickle.load(f1)

PS_dict = {}
grp_cnt = 0
for group in groups:
    #For each group a dictionary containing a dict of P-S times at each station
    PS_dict[grp_cnt] = {}
    for st in group:
        #For each stream, a dictionary of starttimes for each station
        starttimes = {}
        chan_dict = {}
        for tr in st:
            station = str(tr.stats.station)
            channel = str(tr.stats.channel)
            chan_dict[channel] = tr.stats.starttime
            starttimes[station] = chan_dict
        #Calculate S minus P times for each station and put them in PS_dict
        for sta in starttimes:
            if 'EE' in starttimes[sta]:
                P_S = starttimes[sta]['EE'] - starttimes[sta]['EZ']
                if sta in PS_dict[grp_cnt]:
                    PS_dict[grp_cnt][sta].append(P_S)
                else:
                    PS_dict[grp_cnt][sta] = [P_S]
                break
    grp_cnt+=1

#Can now use PS_dict to explore spread of S-P times across all groups
#Convert lists in PS_dict to numpy arrays
sd_dict = {}
for group in PS_dict:
    sd_dict[group] = {}
    for sta in PS_dict[group]:
        std = np.std(np.asarray(PS_dict[group][sta]))
        sd_dict[group][sta] = std
