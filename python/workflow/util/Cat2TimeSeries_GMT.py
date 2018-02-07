#!/usr/bin/env python

""" This program takes a QML catalog and produces a time series mp4 movie
    of earthquake locations."""
    
import os
import shutil
import csv
from subprocess import call
from obspy import read_events, UTCDateTime

# USER INPUT--------------------------------------------------------------------
# Input catalog file
in_dir = r'/home/steve/PhD_Unix/Tomography/NM_GNS_manpicks_2012-15/7_hypoDD/'
in_file = r'manpicksGNS-ngatamariki-may2012-sep2014_filt-10obs-NMlats-2012_fix2.xml'

# Output directory for time series movie
out_dir = r'/home/steve/PhD_Unix/Tomography/NM_GNS_manpicks_2012-15/graphics/'
out_file = r'NM_2012-14_GNSmanpicks.mpg'

# Time increment of each frame (in hours)
time_inc = 4

# GMT directory and script
gmt_dir = r'/home/steve/PhD_Unix/Tomography/gmt_scripts/hypo_time_series/'
gmt_script = r'plot_hypos_timeseries.bsh'

# EQ FILTERING
filt_events = False
# latitude (decimal degrees (WGS84), north then south). Leave as '' if no bound
lat_filt = ['-38.50', '-38.582']
# longitude (decimal degrees (WGS84), west then east). Leave as '' if no bound
long_filt = ['176.163', '176.226']
# depth range (mbsl - +ive down, from then to depth), Leave as '' if no bound
depth_filt = ['', '']
# magnitude range (mag from then to)
mag_filt = ['', '']
# azimuthal gap (az gap min then max - degrees)
azgap_filt = ['', '']
# station count (minimum then maximum number of stations used to locate events)
statcnt_filt = ['', '']
# phase count (minimum then maximum number of picks for events)
phcnt_filt = ['12', '']
# date (earliest then latest time of events, date form is YYYY-MM-DD)  
date_filt = ['2012-05-21', '']
# time (times corresponding to dates, time form is HH:MM:SS)
time_filt = ['00:00:00', '']

# -----------------------------------------------------------------------------

def filter_cat(cat):  
    # Filter events in catalog
    # create filter string
    print('Filtering events...')
    filt_str = []
    # latitude
    if lat_filt[0] != '':
        filt_str.append("latitude <= " + lat_filt[0])
    if lat_filt[1] != '':
        filt_str.append("latitude >= " + lat_filt[1])
    # longitude
    if long_filt[0] != '':
        filt_str.append("longitude >= " + long_filt[0])
    if long_filt[1] != '':
        filt_str.append("longitude <= " + long_filt[1])
    # depth
    if depth_filt[0] != '':
        filt_str.append("depth >= " + depth_filt[0])
    if depth_filt[1] != '':
        filt_str.append("depth <= " + depth_filt[1])
    # magnitude
    if mag_filt[0] != '':
        filt_str.append("magnitude >= " + mag_filt[0])
    if mag_filt[1] != '':
        filt_str.append("magnitude <= " + mag_filt[1])
    # azimuthal gap
    if azgap_filt[0] != '':
        filt_str.append("azimuthal_gap >= " + azgap_filt[0])
    if azgap_filt[1] != '':
        filt_str.append("azimuthal_gap <= " + azgap_filt[1])
    # station count
    if statcnt_filt[0] != '':
        filt_str.append("used_station_count >= " + statcnt_filt[0])
    if statcnt_filt[1] != '':
        filt_str.append("used_station_count <= " + statcnt_filt[1])
    # phase count
    if phcnt_filt[0] != '':
        filt_str.append("used_phase_count >= " + phcnt_filt[0])
    if phcnt_filt[1] != '':
        filt_str.append("used_phase_count <= " + phcnt_filt[1])
    # date and time filter
    if date_filt[0] != '' and time_filt[0] != '':
        filt_str.append("time >= " + date_filt[0] + 'T' + time_filt[0])
    if date_filt[1] != '' and time_filt[1] != '':
        filt_str.append("time <= " + date_filt[1] + 'T' + time_filt[1])
    
    # create new filtered obspy catalog
    first = 0
    for i in filt_str:
        if first == 0: 
            cat_filt = cat.filter(i)
            first = 1 
            print('Event count after ' + i + ' = ' + str(cat_filt.count()))
        else:
            cat_filt = cat_filt.filter(i)
            print('Event count after ' + i + ' = ' + str(cat_filt.count()))
            
    print('Original number of events = ' + str(cat.count()) + '\n' + 
           'Number of events after filter = ' + str(cat_filt.count()) + '\n' +
           'Number of events filtered = ' + str((cat.count() - cat_filt.count())) + '\n')
    
    return cat_filt
    
# Load QML file into obspy catalog
print('Importing Catalog...')
cat = read_events(os.path.join(in_dir + in_file))

if filt_events:
    cat_filt = filter_cat(cat)
    # export filtered catalog in QML format
    cat_filt.write(outqml_dir + outqml_name, format="QUAKEML")
    cat = cat_filt

# make list of event times, lon, lat, depth, mag
evt_list = []
index = 0
for evt in cat:
    evt_list.append([])
    evt_list[index].append(evt.origins[0].time)
    evt_list[index].append(evt.origins[0].longitude)
    evt_list[index].append(evt.origins[0].latitude)
    evt_list[index].append(evt.origins[0].depth / 1000)
    try:
        evt_list[index].append(evt.origins[0].magnitudes[0].mag)
    except:
        evt_list[index].append(1)
    index += 1
    
# get starttime and endtime
starttime, endtime = evt_list[0][0], evt_list[0][0]
for evt in evt_list:
    if evt[0] < starttime:
        starttime = evt[0]
    if evt[0] > endtime:
        endtime = evt[0]
        
# loop over each time period and produce image
# convert time_inc to seconds
time_inc = time_inc*60*60
time_from = starttime - time_inc
time_to = starttime
tot_frames = int((endtime - starttime) / time_inc)
frame_num = 0
# create frames directory
try:
    os.mkdir(os.path.join(gmt_dir, 'frames'))
except:
    print('Warning! - Frames directory already exists')
    pass
frames_dir = os.path.join(gmt_dir, 'frames')

# remove existing cumulative catalog
try:
    os.remove(os.path.join(gmt_dir + '/cum_catalog.gmt'))
except:
    pass

while time_from <= endtime:
    time_from += time_inc
    time_to += time_inc
    frame_num += 1
    print('Creating frame ' + str(frame_num) + ' of ' 
          + str(tot_frames) + ' total frames.')
    # get list of events that occurred during this period
    inc_evt_list = []
    for evt in evt_list:
        if evt[0] > time_from and evt[0] <= time_to:
            inc_evt_list.append(evt)
    # write to inc_catalog.gmt
    try:
        os.remove(os.path.join(gmt_dir + '/inc_catalog.gmt'))
    except:
        pass
    f_inc = open(os.path.join(gmt_dir + '/inc_catalog.gmt'), 'w')
    writer_inc = csv.writer(f_inc)
    for evt in inc_evt_list:
        writer_inc.writerow(evt)
    f_inc.close()
    # append evts to cum_catalog.gmt
    with open(os.path.join(gmt_dir + '/cum_catalog.gmt'), 'a') as f_cum:
        writer_cum = csv.writer(f_cum)
        for evt in inc_evt_list:
            writer_cum.writerow(evt)

    # update gmt script
    f = open(os.path.join(gmt_dir + gmt_script), 'r')
    lines = f.readlines()
    f.close()
    os.remove(os.path.join(gmt_dir + gmt_script))
    f = open(os.path.join(gmt_dir + gmt_script), 'w')
    for line in lines:
        if line.startswith('time='):
            line = ('time=' + '"' + time_from.isoformat().split('.')[0] + 
                    ' to ' + time_to.isoformat().split('.')[0] + '"')
            line = line.replace('-', '.')
            line = line.replace(' ', '_')
            f.write(line  + '\n')
        elif line.startswith('PS='):
            line = 'PS=' + str(frame_num) + '.ps'
            f.write(line  + '\n')
        else:
            f.write(line)
    f.close()
    
    # make gmt script executable
    st = os.stat(os.path.join(gmt_dir, gmt_script))
    os.chmod(os.path.join(gmt_dir, gmt_script), st.st_mode | 0o111)
    # Run gmt script
    call(os.path.join(gmt_dir, gmt_script), shell=True,
                cwd=gmt_dir)
    
    # remove ps and move jpeg to frames directory
    os.remove(os.path.join(gmt_dir, str(frame_num) + '.ps'))
    shutil.move(os.path.join(gmt_dir, str(frame_num) + '.jpg'),
                frames_dir)
           
# run ffmpeg
print('Creating time series movie...')
call(['avconv', '-r 25', '-f', 'image2', 
      '-i', '%d.jpeg', '-b', '300000k', 
      os.path.join(out_dir + out_file)], shell=True, cwd=frames_dir)
print('Time series movie complete')
