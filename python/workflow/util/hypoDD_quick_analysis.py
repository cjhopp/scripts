#!/usr/bin/env python

""" This program analyses output from hypoDD and tomoDD. 
    """
    
import os
import subprocess
import time
import shutil
import matplotlib.pyplot as plt

# USER INPUT ---------------------------------------------------------------

hypoDD_out_dir = r'/home/steve/PhD_Unix/Tomography/RK-NM_2012-2015/7_hypoDD/input_files_bup/'
hypoDD_files = {'loc' : 'hypoDD.loc',
                'reloc' : 'hypoDD.reloc',
                'log' : 'hypoDD.log'}

gmt_script_dir = r'/home/steve/PhD_Unix/Tomography/gmt_scripts/hypo_plot'
gmt_script = r'plot_hypos_hypoDD.bsh'
file_suffix = '_run1'

# --------------------------------------------------------------------------

# Matplotlib settings for plots
# font sizes
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

# Copy hypoDD loc and reloc to gmt folder
shutil.copy(os.path.join(hypoDD_out_dir, hypoDD_files['loc']),
            os.path.join(gmt_script_dir, hypoDD_files['loc']))
shutil.copy(os.path.join(hypoDD_out_dir, hypoDD_files['reloc']),
            os.path.join(gmt_script_dir, hypoDD_files['reloc']))

# Run gmt script
st_time = time.time()
subprocess.call(os.path.join(gmt_script_dir, gmt_script), shell=True,
                cwd=gmt_script_dir)
end_time = time.time()

# Move and rename output pdf
for filename in os.listdir(os.path.join(gmt_script_dir, 'PostScripts')):
    # check if created recently
    f_time = abs(os.path.getmtime(os.path.join(gmt_script_dir, 'PostScripts', filename)))
    if f_time >= st_time and f_time <= end_time:
            shutil.move(os.path.join(gmt_script_dir, 'PostScripts', filename), 
                                     hypoDD_out_dir)
            new_filename = (filename.split('.')[0] + file_suffix + 
                            '.' + filename.split('.')[1])
            os.rename(os.path.join(hypoDD_out_dir, filename), 
                      os.path.join(hypoDD_out_dir, new_filename))

# Load log file and make utf-8 text file
f_in = open(os.path.join(hypoDD_out_dir, hypoDD_files['log']), 'rb')
f_out = open(os.path.join(hypoDD_out_dir, hypoDD_files['log'] + '.fix'), 'w')
bts = f_in.read()
f_in.close()

for index in range(len(bts)):
    try:
        st = bts[index:index+1].decode('utf-8')
        if st != '\0':
            f_out.write(st)
    except:
        continue

f_out.close()
f = open(os.path.join(hypoDD_out_dir, hypoDD_files['log'] + '.fix'), 'r')
lines = f.readlines()
f.close()

# Create dictionary for each inversion for each cluster
clusters = {}
index = 0
cc_ct, cc_only, ct_only = False, False, False
for line in lines:
    index += 1
    if line.startswith('RELOCATION OF CLUSTER:'):
        clust_num = int(line.split('RELOCATION OF CLUSTER:')[1][0:4])
        clusters[clust_num] = {}
    if line.startswith('  IT   EV  CT  CC'):
        cc_ct = True
        line_spl = lines[index + 1][5:].split()
        it_num = int(lines[index + 1][:5].split()[0])
        clusters[clust_num][it_num] = {'ev%' : int(line_spl[0]),
                                       'ct%' : int(line_spl[1]),
                                       'cc%' : int(line_spl[2]),
                                       'rmsct' : int(line_spl[3]),
                                       'rmscc' : int(line_spl[5]),
                                       'rmsst' : int(line_spl[7]),
                                       'dx' : int(line_spl[8]),
                                       'dy' : int(line_spl[9]),
                                       'dz' : int(line_spl[10]),
                                       'os' : int(line_spl[12]),
                                       'aq' : int(line_spl[13]),
                                       'cnd' : int(line_spl[14])}
    elif line.startswith('  IT   EV  CT'):
        ct_only = True
        line_spl = lines[index + 1][5:].split()
        it_num = int(lines[index + 1][:5].split()[0])
        clusters[clust_num][it_num] = {'ev%' : int(line_spl[0]),
                         'ct%' : int(line_spl[1]),
                         'rmsct' : int(line_spl[2]),
                         'rmsst' : int(line_spl[4]),
                         'dx' : int(line_spl[5]),
                         'dy' : int(line_spl[6]),
                         'dz' : int(line_spl[7]),
                         'os' : int(line_spl[9]),
                         'aq' : int(line_spl[10]),
                         'cnd' : int(line_spl[11])}   
    elif line.startswith('  IT   EV  CC'):
        cc_only = True
        line_spl = lines[index + 1][5:].split()
        it_num = int(lines[index + 1][:5].split()[0])
        clusters[clust_num][it_num] = {'ev%' : int(line_spl[0]),
                                       'cc%' : int(line_spl[1]),
                                       'rmscc' : int(line_spl[2]),
                                       'rmsst' : int(line_spl[4]),
                                       'dx' : int(line_spl[5]),
                                       'dy' : int(line_spl[6]),
                                       'dz' : int(line_spl[7]),
                                       'os' : int(line_spl[9]),
                                       'aq' : int(line_spl[10]),
                                       'cnd' : int(line_spl[11])}
        
# Inversion summary plot
for clust_num in clusters.keys():
    # create lists for plotting
    ev, ct, cc, rmsct, rmscc, rmsst, dx, dy, dz, osh, aq, cnd = ([],[], [], [], [], [],
                                                            [], [], [], [], [], []) 
    iters = clusters[clust_num].keys()
    for iter in iters:
        ev.append(clusters[clust_num][iter]['ev%'])
        if cc_ct:
            ct.append(clusters[clust_num][iter]['ct%'])
            cc.append(clusters[clust_num][iter]['cc%'])
            rmsct.append(clusters[clust_num][iter]['rmsct'])
            rmscc.append(clusters[clust_num][iter]['rmscc'])
        elif ct_only:
            ct.append(clusters[clust_num][iter]['ct%'])
            rmsct.append(clusters[clust_num][iter]['rmsct'])       
        elif cc_only:
            cc.append(clusters[clust_num][iter]['cc%'])
            rmscc.append(clusters[clust_num][iter]['rmscc'])            
        rmsst.append(clusters[clust_num][iter]['rmsst'])
        dx.append(clusters[clust_num][iter]['dx'])
        dy.append(clusters[clust_num][iter]['dy'])
        dz.append(clusters[clust_num][iter]['dz'])
        osh.append(clusters[clust_num][iter]['os'])
        aq.append(clusters[clust_num][iter]['aq'])
        cnd.append(clusters[clust_num][iter]['cnd'])
    # Event, ct and cc percentage (of initial number of data of each type)
    # and number of air quakes
    ax1 = plt.subplot(611)
    ax1.plot(iters, ev, marker='o', label='Event %')
    if cc_ct:
        ax1.plot(iters, ct, marker='o', label ='Catalog %')
        ax1.plot(iters, cc, marker='o', label ='Cross-corr %')
    elif ct_only:
        ax1.plot(iters, ct, marker='o', label ='Catalog %')
    elif cc_only:
        ax1.plot(iters, cc, marker='o', label ='Cross-corr %')
    ax1_2 = ax1.twinx()
    ax1_2.plot(iters, aq, marker= 'x', label='Air-quakes')               
    ax1.set_ylabel('Percent')
    ax1.legend()
    ax1_2.set_ylabel('Count')
    ax1_2.legend()
    ax1.set_title('HypoDD Inversion Stats for Cluster ' + str(clust_num) + '\n',
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=10)
    
    # RMS residual for catalog and cross-corr (RMS CT and CC)
    ax2 = plt.subplot(612)
    if cc_ct:
        ax2.plot(iters, rmsct, marker='o', label ='Catalog RMS')
        ax2_1 = ax2.twinx()
        ax2_1.plot(iters, rmscc, marker='x', color='g', label ='Cross-corr RMS')
        ax2_1.set_ylabel('RMS residual (ms)')
        ax2_1.legend(bbox_to_anchor=(0.2, 0.2)) 
    elif ct_only:
        ax2.plot(iters, rmsct, marker='o', label ='Catalog RMS')
    elif cc_only:
        ax2.plot(iters, rmscc, marker='o', label ='Cross-corr RMS')            
    ax2.set_ylabel('RMS residual (ms)')
    ax2.legend()    
 
    
    # Largest rms residual at station (RMS ST)
    ax3 = plt.subplot(613)
    ax3.plot(iters, rmsst, marker='o', label='Largest Stat RMS')
    ax3.set_ylabel('RMS residual (ms)')
    ax3.legend()
    
    # Average absolute of change in hypocentre (DX, DY, DZ)
    ax4 = plt.subplot(614)
    ax4.plot(iters, dx, marker='o', label='Hypo Shift DX')
    ax4.plot(iters, dy, marker='o', label='Hypo Shift DY')
    ax4.plot(iters, dz, marker='o', label='Hypo Shift DZ')
    ax4.set_ylabel('Hypo Shift (m)')
    ax4.legend()    
    
    # Origin shift of each cluster (OS)
    ax5 = plt.subplot(615)
    ax5.plot(iters, osh, marker='o', label='Ave Cluster Shift')
    ax5.set_ylabel('Cluster Shift (m)')

    # Condition number 
    ax6 = plt.subplot(616)
    ax6.plot(iters, cnd, marker='o', label='Condition Number')
    ax6.set_ylabel('Condition Number')
    ax6.set_xlabel('Iteration Number')    
    
    # get figure
    fig = plt.gcf()
    fig.set_size_inches(8.27, 11.69)
    
    # format
    fig.tight_layout()
    
    # save figure
    fig.savefig(os.path.join(hypoDD_out_dir, 'Cluster' + str(clust_num) + 
                             '_hypoDDInvStats_' + file_suffix + '.pdf'), 
                format='PDF')
    
    # close figure
    plt.close()

