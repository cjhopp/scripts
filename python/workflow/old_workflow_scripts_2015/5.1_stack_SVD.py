#!/usr/bin/env python

"""
Handles alligning, stacking and, if desired, single value decomposition \
of template groups created in 5_hierarchy.py
"""
import sys
sys.path.insert(0, '/home/chet/EQcorrscan')
from glob import glob
from eqcorrscan.utils import stacking, clustering
from obspy import read_events, read
from obspy.core.event import ResourceIdentifier

# Grabbing correlation catalogs
# group_cats = glob('/media/chet/hdd/seismic/NZ/catalogs/qml/corr_groups/4_sec_temps/*')
# Specific cat debug
group_cats = glob('/media/chet/hdd/seismic/NZ/catalogs/qml/corr_groups/4_sec_temps/spacegrp_063_corrgrp_018*')
# Make the background template dictionary
temp_dir = '/media/chet/hdd/seismic/NZ/templates/rotnga_2015/dayproc_4-27/*'
temp_files = glob(temp_dir)
template_dict = {}
for filename in temp_files:
    uri_name = 'smi:org.gfz-potsdam.de/geofon/' +\
               filename.split('/')[-1].split('_')[0]
    uri = ResourceIdentifier(uri_name)
    template_dict[uri] = read(filename)
# Set dimension of the subspace detector (# of left singular vectors)
svst_dim = 3
for grp_cat in group_cats:
    cat = read_events(grp_cat)
    grp_name = grp_cat.split('/')[-1].rstrip('.xml')
    temp_root = '/media/chet/hdd/seismic/NZ/detectors/'
    if len(cat) == 1:
        print('Catalog %s contains single event' % grp_name)
        temp_name = temp_root + 'singleton/' + grp_name + '_single.mseed'
        # print('Writing single event as template: %s' % temp_name)
        # template_dict[cat[0].resource_id].write(temp_name, format='MSEED')
    elif len(cat) <= svst_dim:
        print('Not much point creating subspace detectors for cats of len %02d \
              when were going to create detector of dimension %02d'
              % (len(cat), svst_dim))
    elif len(cat) > svst_dim:
        print('Working on group: %s' % grp_cat)
        # Create group template dict with stream and shifts keyed to id
        templates = {ev.resource_id: {'stream': template_dict[ev.resource_id],
                                      'shifts': {}}
                     for ev in cat}
        # Also make list in order of cat for plotting
        temp_list = [template_dict[ev.resource_id] for ev in cat]
        # Create stachans dict to loop over in aligning traces
        stachans = {tr.stats.station: []
                    for tid, template in templates.iteritems()
                    for tr in template['stream']}
        for tid, t_dict in templates.iteritems():
            for tr in t_dict['stream']:
                chan_code = 'E' + tr.stats.channel[1]
                if chan_code not in stachans[tr.stats.station]:
                    stachans[tr.stats.station].append(chan_code)
        # Seperate out common stachan traces, allign them, store lag in stachan
        for sta, chans in stachans.iteritems():
            for chan in chans:
                trace_ids = []
                chan_traces = []
                for tid, t_dict in templates.iteritems():
                    if t_dict['stream'].select(station=sta, channel=chan):
                        tr = t_dict['stream'].select(station=sta,
                                                     channel=chan)[0]
                        trace_ids.append(tid)
                        chan_traces.append(tr)
                if chan_traces and len(chan_traces) > 1:
                    shifts, ccs = stacking.align_traces(chan_traces,
                                                        shift_len=25)
                    for i, shift in enumerate(shifts):
                        if sta not in templates[trace_ids[i]]['shifts']:
                            templates[trace_ids[i]]['shifts'][sta] = {chan:
                                                                      shift}
                        else:
                            templates[trace_ids[i]]['shifts'][sta][chan] = shift
        # Now loop through templates and shift waveforms
        aligned_temps = []
        for tid, t_dict in templates.iteritems():
            new_st = t_dict['stream'].copy()
            for tr in new_st:
                sta = tr.stats.station
                chan = tr.stats.channel
                start = tr.stats.starttime
                if sta in t_dict['shifts'] and chan in t_dict['shifts'][sta]:
                    tr.stats.starttime = start + t_dict['shifts'][sta][chan]
            aligned_temps.append(new_st)
        # Actual SVD and create streams for convenience
        SVectors, SValues, Uvectors, stachans = clustering.SVD(aligned_temps)
        SVstreams = clustering.SVD_2_stream(SVectors, stachans, svst_dim, 50.0)
        for i, stream in enumerate(SVstreams):
            svec_name = temp_root + '/SVD/%s_n%02d_svec%02d.mseed' % (grp_name,
                                                                      svst_dim,
                                                                      i)
            stream.write(svec_name, format='MSEED')
