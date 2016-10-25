#!/usr/bin/env python

"""
Workflow script to take detected events, cluster them, and calculate local magnitudes
"""
import numpy as np
import copy
import warnings
import ipdb
import matplotlib
from eqcorrscan.core import subspace
from eqcorrscan.utils import mag_calc, stacking, clustering
from glob import glob
from obspy import read_events, read, UTCDateTime, Catalog, Stream
from obspy.core.event import ResourceIdentifier, Magnitude, CreationInfo

# Laptop specific scaling requirement
matplotlib.rcParams['figure.dpi'] = 150

###### Block for detecting self detections more robustly than previous attempts ##########
def template_dict_only(temp_dir):
    temp_dict = {}
    temp_files = glob(temp_dir)
    for filename in temp_files:
        temp_rid = filename.split('/')[-1].split('_')[0]
        tmp_st = read(filename)
        tmp_st.sort(['starttime'])
        starttime = tmp_st[0].stats.starttime
        temp_dict[temp_rid] = {'starttime': starttime,
                               'stream': tmp_st}
    return temp_dict


def find_self_dets(det_cat, temp_dir):
    temp_dict = template_dict_only(temp_dir)
    selfs = []
    for ev in det_cat:
        temp_str, det_time_str = str(ev.resource_id).split('/')[-1].split('_')
        if det_time_str == 'self':
            selfs.append(ev.resource_id)
            continue
        det_time = UTCDateTime(det_time_str)
        if abs(det_time - temp_dict[temp_str]['starttime']) < 1.0:
            print('Found self detection for %s' % str(ev.resource_id))
            selfs.append(ev.resource_id)
    return selfs


def reassign_selfs(cat, det_cat, temp_dir):
    selfs = find_self_dets(det_cat, temp_dir)
    for ev in cat:
        temp_str, det_time_str = str(ev.resource_id).split('/')[-1].split('_')
        if ev.resource_id in selfs and det_time_str != 'self':
            ev.resource_id = ResourceIdentifier('smi:local/%s_self' % temp_str)
    return cat

###########################################################################################

def temp_det_dict_shifts(cat, temp_cat, det_dir, ccs_thresh, shift_len, plot=False):
    """

    :param cat: catalog of detections
    :param temp_cat: catalog of templates
    :param det_dir: directory of detection waveforms
    :param ccs_thresh: cc threshold for alignment
    :param shift_len: allowed shift for alignment (sec)
    :param plot: plot bool
    :return: template dictionary
    """
    # Make convenience dict of {temp_id: index} for use in creating template_dict
    ind_dict = {}
    for i, ev in enumerate(cat):
        ind_dict[ev.resource_id] = i
    temp_files = glob(det_dir)
    template_dict = {}
    for filename in temp_files:
        temp_rid = ResourceIdentifier('smi:org.gfz-potsdam.de/geofon/' + filename.split('/')[-1].split('_')[0])
        det_rid = ResourceIdentifier('smi:local/' + filename.split('/')[-1].rstrip('.mseed'))
        if det_rid in ind_dict:
            if temp_rid not in template_dict and str(det_rid).split('/')[-1].split('_')[-1] == 'self':
                template_dict[temp_rid] = {det_rid: {'stream': read(filename),
                                                     'shifts': {},
                                                     'ind': ind_dict[det_rid]},
                                           'temp_mag': [ev.preferred_magnitude().mag for ev in temp_cat
                                                        if ev.resource_id == temp_rid][0],
                                           'temp_ind': ind_dict[det_rid]}
            elif temp_rid not in template_dict:
                template_dict[temp_rid] = {det_rid: {'stream': read(filename),
                                                     'shifts': {},
                                                     'ind': ind_dict[det_rid]},
                                           'temp_mag': None}
            elif str(det_rid).split('/')[-1].split('_')[-1] == 'self':
                template_dict[temp_rid][det_rid] = {'stream': read(filename),
                                                     'shifts': {},
                                                     'ind': ind_dict[det_rid]}
                template_dict[temp_rid]['temp_mag'] = [ev.preferred_magnitude().mag for ev in temp_cat
                                                        if ev.resource_id == temp_rid][0]
                template_dict[temp_rid]['temp_ind'] = ind_dict[det_rid]
            else:
                template_dict[temp_rid][det_rid] = {'stream': read(filename),
                                                     'shifts': {},
                                                     'ind': ind_dict[det_rid]}
    # Trim the waveforms to shorter lengths
    # Templates from /2015_det2cats/* are 3 sec pre-pick and 7 sec post-pick
    for tid, det_dict in template_dict.iteritems():
        for evid, ev_dict in det_dict.iteritems():
            if evid != 'temp_mag' and evid != 'temp_ind':
                for tr in ev_dict['stream']:
                    tr.trim(starttime=tr.stats.starttime + 2.5,
                            endtime=tr.stats.endtime - 3)
    # Variable of random keys for plotting
    samp_ids = [id for i, id in enumerate(template_dict.keys())
                if i in np.random.choice(range(len(template_dict)),
                                         len(template_dict) // 20,
                                         replace=False)]
    # Now shift and
    for tid, det_dict in template_dict.iteritems():
        design_set = []
        design_inds = []
        print('Shifting waveforms for template %s' % str(tid))
        if plot:
            if tid in samp_ids: plotvar=True
            else: plotvar=False
        else: plotvar=False
        for eid, ev_dict in det_dict.iteritems():
            if eid != 'temp_mag' and eid != 'temp_ind':
                design_set.append(ev_dict['stream'])
                design_inds.append(ev_dict['ind'])
        aligned_streams = subspace.align_design(design_set,
                                                reject=ccs_thresh,
                                                multiplex=False,
                                                shift_len=shift_len,
                                                plot=plotvar)
        det_dict['aligned'] = aligned_streams
        det_dict['aligned_inds'] = design_inds
    return template_dict


def relative_mag_calc(cat, template_dict, n_SVs=4, plot=False, debug=1):
    """
    Now we're going to loop through templates, filter out poorly correlated waveforms,
    compute SVD and relative magnitudes using EQcorrscan functions, then map relative
    mags to real magnitudes using template local magnitudes
    """
    from eqcorrscan.utils.plotting import multi_trace_plot
    from obspy.core.event import ResourceIdentifier
    import matplotlib.pyplot as plt
    # Assign shifts for detections to template dictionary
    new_cat = Catalog()
    # Random sample of template ids for plotting
    samp_ids = [id for i, id in enumerate(template_dict.keys())
                if i in np.random.choice(range(len(template_dict)),
                                         len(template_dict) // 20,
                                         replace=False)]
    for tid, det_dict in template_dict.iteritems():
        # Perform some checks on the dictionary first
        if len(det_dict) <= 1:
            print('%s has <= one detection. No magnitude will be calculated.' % str(tid))
            continue
        else:
            print('Working on detections for template: %s' % str(tid))
            if 'self' not in [str(key).split('/')[-1].split('_')[-1] for key in det_dict.keys()]:
                print('Self detection not located in catalog. Moving to next template.')
                continue
            inds = det_dict['aligned_inds']
            stream_list = det_dict['aligned']
            # Do SVD
            if len(stream_list) <= n_SVs:
                warnings.warn('Fewer streams then nSVs passed to SVD. Moving to next template')
                continue
            svd_dict = clustering.svd(stream_list, full=True)
            if plot:
                if tid in samp_ids:
                    for stachan in svd_dict:
                        if 'svectors' in svd_dict[stachan]:
                            if len(svd_dict[stachan]['svectors']) > 0:
                                if len(svd_dict[stachan]['events']) < 5:
                                    # we will not plot stachans with only one event
                                    continue
                                fig, axes = plt.subplots(len(svd_dict[stachan]['events']), 1, sharex=True,
                                                             figsize=(14, 24), squeeze=False)
                                first_SV = svd_dict[stachan]['svectors'][0]
                                first_SVal = svd_dict[stachan]['svalues'][0]
                                for i, ev_ind in enumerate(svd_dict[stachan]['events']):
                                    data_tr = stream_list[ev_ind].select(station=stachan.split('.')[0],
                                                                         channel=stachan.split('.')[1])[0]
                                    # TODO Also, add U weight text.
                                    samp_rate = data_tr.stats.sampling_rate
                                    SV_y = first_SV * first_SVal
                                    SV_x = np.arange(len(SV_y))
                                    SV_x = SV_x / samp_rate
                                    dat_y = data_tr.data
                                    U_wt = np.matrix(copy.deepcopy(svd_dict[stachan]['uvectors']))
                                    # TODO Does something need to happen to the amplitudes of the streams
                                    # TODO in stream_list??
                                    svd_wts = np.array(U_wt[:, 0]).reshape(-1).tolist()
                                    axes[i, 0].plot(SV_x, SV_y * svd_wts[i], color='r')
                                    axes[i, 0].plot(SV_x, dat_y, color='k')
                                    axes[i, 0].text(0.9, 0.15,
                                                    str(svd_wts[i]),
                                                    bbox=dict(facecolor='white', alpha=0.95),
                                                    transform=axes[i, 0].transAxes)
                                    axes[i, 0].text(0.7, 0.85, data_tr.stats.starttime.datetime.
                                                   strftime('%Y/%m/%d %H:%M:%S'),
                                                   bbox=dict(facecolor='white', alpha=0.95),
                                                   transform=axes[i, 0].transAxes)
                                fig.suptitle('%s\nChannel: %s First SVal: %f'
                                             % (str(tid), stachan, first_SVal))
                                fig.show()
            # Feed output vectors and values to mag_calc.SVD_moments
            M, events_out = mag_calc.SVD_moments(svd_dict, n_SVs, debug=debug)
            # Find rel_amp of self detection
            try:
                rel_amp_t = [M[i] for i, cat_ind in enumerate(inds)
                             if i in events_out and cat_ind == det_dict['temp_ind']][0]
            except:
                warnings.warn('Relative amp not calculated for template in this case....investigate')
                continue
            # Convert relative values to template values
            Mls = [np.log10(rel_amp_i / rel_amp_t) + det_dict['temp_mag']
                   for rel_amp_i in M]
            if len(Mls) != len(events_out):
                warnings.warn('Not same number of local mags and out events')
            for i, cat_ind in enumerate(inds):
                if i in events_out:
                    Mls_ind = [k for k, ev in enumerate(events_out) if ev == i][0]
                    if cat_ind == det_dict['temp_ind']:
                        event = cat[cat_ind].copy()
                        event.magnitudes.append(Magnitude(mag=det_dict['temp_mag'],
                                                          creation_info=(CreationInfo(author='SeisComp'))))
                        new_cat.append(event)
                    else:
                        event = cat[cat_ind].copy()
                        event.magnitudes.append(Magnitude(mag=Mls[Mls_ind],
                                                          creation_info=(CreationInfo(author='eqcorrscan.utils.mag_calc.SVD_moment'))))
                        new_cat.append(event)

    return new_cat