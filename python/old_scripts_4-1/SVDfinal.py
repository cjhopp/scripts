#!/usr/bin/python

## This script runs the SVD on a set of earthquake events 
## (of similar waveforms) and then decides on their relative moments.
## SARAH JURY updated 5 Feb 2013

import subprocess
import obspy.core
import numpy as np
import matplotlib.pyplot as plt
import math
import mpl_toolkits.basemap
import random
import itertools
import os
import scipy.linalg

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)
    
##                    Important 

## Seismograms must be aligned for this to work correctly.
## Must set the below information
## For this script all events must be detected at all stations

#############################
path = './okmok_data/SARAH_test/c*/filt.' # where the data is located
path2 = '.Z.SAC' # the end of the fileames
station_list = ['OKER', 'OKSP', 'OKWE'] # choose which stations you want

# list of events for naming purposes later (in order they are read in)
events = ['c10021', 'c10054', 'c10146', 'c10327', 'c10346', 'c10365'] 
#############################

DataStream =[]

# Read in data
for StationIndex in range(len(station_list)):
    station = station_list[StationIndex]
    st = obspy.core.read(path + station + path2)
    DataStream.append(st) #add all streams to one matrix

# Define empty matrix to be filled with relative moments
K = [] #Matrix for the following station kernels to go into

# Loop through stations
for StreamIndex in range(len(DataStream)):
    print
    print "For station", station_list[StreamIndex]
    print "Reading in data"
    st = DataStream[StreamIndex]
    DataMatrix = []
    for EventIndex in range(len(st)):
        tr = st[EventIndex]
        raw = tr.data[0:len(tr)]
        DataMatrix.append(raw)
    
    # Do the SVD
    print "Doing SVD"
    U, s, V = scipy.linalg.svd(DataMatrix,full_matrices=False)
    
    # Decide how many singular values you want to use
    NumberSVs=4;
    # Set the rest to zero
    s[NumberSVs:len(s)]=0
    # Make s into a diagonal matrix
    s=np.diag(s)
    
    # Make the SVD output into matrices
    U=np.matrix(U)
    s=np.matrix(s)
    V=np.matrix(V)
    
    # Extract weights corresponding to first SV
    SVD_weights = U[:,0] #Gives the relative amplitude weights
    
    # estimates and residuals using the SVD
    est=U*s*V
    est=np.array(est)
    resid=DataMatrix-est
    print "Plotting Graph"
    for EventIndex in range(len(st)):
        
        # Define time values
        t = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, 
                                                        tr.stats.delta)
        
        # define and normalise weight labels to be plotted 
        weights = abs(SVD_weights/max(abs(SVD_weights)))
        
        # Plot data
        maxAmp=np.amax(abs(DataMatrix[EventIndex])) # normalising factor
        # Raw data, SVD estimate, residuals
        plt.plot(t,DataMatrix[EventIndex]/maxAmp+EventIndex,'k')
        plt.plot(t,est[EventIndex]/maxAmp+EventIndex, 'm--') 
        plt.plot(t,resid[EventIndex]/maxAmp+EventIndex,'dodgerblue') 
        plt.text(0.5,+EventIndex+.1,str("%.3f" % weights[EventIndex]) , 
                                    bbox=dict(facecolor='plum', alpha=1))
        
        # Graph titles and axes labels
        station_name = station_list[StreamIndex]
        plt.suptitle('SVD analysis at station '+ station_name)
        plt.xlabel('Time (s)')
    plt.show()
    
    ## Create the kernel matrix  
    
    # turn the SVD_weights matrix into a list
    SVD_weights = np.array(SVD_weights).reshape(-1).tolist()
    
    # Shuffle a copy of the SVD_weights for pairing
    random_SVD_weights = np.copy(SVD_weights)
    random.shuffle(random_SVD_weights)
    
    # Add the first element to the end so all elements will be paired twice
    random_SVD_weights = np.append(random_SVD_weights,[random_SVD_weights[0]])


    # Take pairs of all the SVD_weights (each weight appears in 2 pairs)
    pairs = []
    for pair in pairwise(random_SVD_weights):
        pairs.append(pair)
    pairs = np.array(pairs,dtype=float)


    # Deciding values for each place in kernel matrix using the pairs
    for pairsIndex in range(len(pairs)):
        min_weight = min(pairs[pairsIndex])
        max_weight = max(pairs[pairsIndex])
        row = []
        # Working out values for each row of kernel matrix
        for i in range(len(SVD_weights)):
            if SVD_weights[i] == max_weight:
                result = -1
            elif SVD_weights[i] == min_weight:
                normalised = max_weight/min_weight
                result = float(normalised)
            else:
                result = 0
            row.append(result)
        # Add each row to the K matrix   
        K.append(row)
        
# Add an extra row to K, so average moment = 1 
K.append(np.ones(len(SVD_weights)) * (1. / len(SVD_weights)))

print
print "Created Kernel matrix"
print
Krounded = np.around(K, decimals = 4)
print Krounded
print

# Create a weighting matrix to put emphasis on the final row.
W = np.matrix(np.identity(len(K)))
# the final element of W = the number of stations*number of events
W[-1,-1] = len(K)-1

# Make K into a matrix 
K = np.matrix(K)

############

# Solve using the weighted least squares equation, K.T is K transpose
Kinv = np.array(np.linalg.inv(K.T*W*K) * K.T * W)

# M = Kinv*D
# D = [0,0,......,0,1] (column) where the no. of rows = no. of rows of K
# Therefore M is the last column of Kinv (by matrix multiplication)
# M are the relative moments of the events
M = Kinv[:, -1]

# Store the relative moments and events in an ouput file
dstore = []
# Now print out final result
fout = open('SVDfinal.txt', 'w')

# List the results
for i, rel_mom in enumerate(M):
    dstore.append([events[i], rel_mom])
    print 'The relative moment of %s is %.2f' % (events[i], rel_mom)
    fout.write('The relative moment of %s is %.2f\n'% (events[i], rel_mom))
fout.close()

# Calculate the Mean of the relative moments
print 'The average moment is %5.2f'% M.mean()
print 
