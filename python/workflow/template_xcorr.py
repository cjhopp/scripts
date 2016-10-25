#!/user/bin/python

"""
Take a set of templates and cross-correlate them with each other using obspy.cross_correlation.xcorr

Will be used for identifying multiplets in MRP dataset
"""
import os
import cPickle as pickle
import pylab as pl
from obspy import Stream, read, UTCDateTime
from obspy.signal.cross_correlation import xcorr
from glob import glob
#from core.match_filter import normxcorr2, _channel_loop
import numpy as np

#shift_file = open('/home/chet/data/template_pha_shift.txt', 'w')
temp_dir = '/home/chet/data/templates/'
os.chdir(temp_dir)
ms_files = glob('*.ms')
#Sort files in time
files = ms_files.sort()
####################################################################
"""
Function for running template auto correlations
    Can also write out shift 'i.e. travel time delays' to file
"""

def template_auto_corr(files, write_shifts=False):
    if write_shifts:
        shift_file = open('/home/chet/data/template_pha_shift.txt', 'w')
    xcorrs = np.zeros((len(files), len(files)))
    file_cnt = 0
    #For each template, correlate with each other template and write value to xcorrs
    for j in range(len(files)):
        print('Running template '+files[j])
        temp1 = read(files[j])
        temp1.resample(50)
        for i in range(len(files)):
            temp2 = read(files[i])
            temp2.resample(50)
            #print('correlating with '+files[i])
            #Make list of common sta.chans between both templates
            temp1_stachan = []
            temp2_stachan = []
            for tr1 in temp1:
                temp1_stachan.append(tr1.stats.station+'.'+tr1.stats.channel)
            for tr2 in temp2:
                temp2_stachan.append(tr2.stats.station+'.'+tr2.stats.channel)
            com_stachan = set(temp1_stachan).intersection(temp2_stachan)
            #Run the cross-correlation loop
            temp_xcorrs = []
            #shifts = []
            for stachan in com_stachan:
                #Use tr.select() to specify sta and chan from stachan list
                temp1_data = temp1.select(station = stachan[0:4], channel = stachan[5:])
                temp2_data = temp2.select(station = stachan[0:4], channel = stachan[5:])
                [index, ccc] = xcorr(temp1_data[0], temp2_data[0], 50)
                temp_xcorrs.append(ccc)
                if write_shifts:
                    #Write phase shifts to file for possible use at later date
                    shift_file.write('%s %s %s %s\n' %(files[j], stachan, ccc, index))
            #What sort of correlation are we doing? Stacked CCC? Mean CCC?
            xcorrs[j, i] = np.mean(temp_xcorrs)
            file_cnt += 1
    if write_shifts:
        shift_file.close()
    return xcorrs

#Call function on miniseed file list
xcorrs = template_auto_corr(files)
#If NaNs in xcorrs, replace with 0 or something more meaningful
xcorrs[np.isnan(xcorrs)] = 0
#Plot xcorrs as 'Tartan' diagram using pylab
pl.pcolor(xcorrs)
pl.colorbar()
pl.show()

with open('/home/chet/data/xcorr_1025.pickle', 'w') as f:
    pickle.dump(xcorrs, f)
###########################################################
"""
Now we create families of events using a technique similar to Petersen
"""
##If reading xcorrs from xcorr_*.pickle
with open('/home/chet/data/xcorr_1025.pickle') as f:
    xcorrs = pickle.load(f)

#Take average ccc for all rows
template_avg = np.mean(xcorrs, axis=0)
#Loop over desired number of families, finding 'master' and 'fam' each iteration
fams = 50
fam_list = []
for i in range(fams):
    index_master = np.argmax(template_avg)
    ##Set threshold value for families
    ##Option 1: master avg * scale
    scale = 3.0
    thresh = template_avg[index_master] * scale
    ##Option 2: constant
    # thresh = 0.20
    print('Threshold for family '+ str(i) + ' set at: ' + str(thresh))
    new_ind = np.asarray(np.where(xcorrs[index_master] > thresh))
    if i == 0:
        # total_ind = new_ind
        fam_list.append(new_ind[0])
        remade_list = new_ind[0]
    else:
        #New fam of indices which don't already belong to a group
        unique_ind = np.setdiff1d(new_ind[0], remade_list)
        fam_list.append(unique_ind)
        remade_list = np.concatenate((remade_list, unique_ind), axis=1)
        #Add indices of new fam to array of indices which belong to a family
        #total_ind = np.concatenate((total_ind, new_ind), axis=1)
    template_avg[remade_list] = 0
    #Make a cuttoff at which creating multiplets is meaningless
    if len(np.where(template_avg == 0)[0]) > (len(xcorrs[0])*0.95):
        print('Too few events left to be of use')
        break
    #Rewrite the list of filenames in the new order
    fam_file_list = []
    for ind in remade_list:
        fam_file_list.append(files[ind])
#######################################################################
"""
Rerun the correlation function for events reordered by family
"""

##Re-run the correlations over newly ordered list (same as above)
fam_xcorrs = template_auto_corr(fam_file_list)

#If NaNs in xcorrs, replace with 0 or something more meaningful
fam_xcorrs[np.isnan(fam_xcorrs)] = 0
#Plot xcorrs as 'Tartan' diagram using pylab
pl.pcolor(fam_xcorrs)
pl.colorbar()
pl.show()

with open('/home/chet/data/fam_const_'+str(thresh)+'.pickle', 'w') as f:
    pickle.dump([fam_xcorrs, fam_list, fam_file_list], f)
