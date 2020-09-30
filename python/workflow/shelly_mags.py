#!/usr/bin/python
"""
Spitballing recreating David Shelly's methodology on the Long Valley catalog
for magnitudes and focal mechanisms
"""

import csv
import copy
import scipy
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from collections import Counter
from obspy import Stream, read, Catalog
from obspy.core.event import Magnitude, Comment
from eqcorrscan.utils import stacking
from eqcorrscan.utils.mag_calc import svd_moments
from eqcorrscan.utils.pre_processing import shortproc
from eqcorrscan.utils.clustering import cross_chan_correlation, svd


def CUSP_to_SC3_rel_mags(det_cat, temp_cat, selfs):
    """
    Take a catalog with relative magnitudes calculated using the Ristau 2009
    CUSP equation and correct them using the updated SeisComP3 scale

    :param det_cat: Catalog of detections and templates with magnitudes
    :param selfs: List of strings for self detection ids
    :return:
    """
    # Make a dictionary of the CUSP-derived moment, SeisComP M0 for templates
    temp_mag_dict = {ev.resource_id.id.split('/')[-1]:
                        {'Old M0':
                              local_to_moment(ev.magnitudes[0].mag,
                                              m=0.88, c=0.73),
                         'New M0':
                              local_to_moment(ev.magnitudes[0].mag,
                                              m=0.97, c=0.14)}
                     for ev in temp_cat}
    # Now loop the catalog and redo the calculations
    for det in det_cat:
        # First determine the relative moment (I didn't save these anywhere...)
        eid = det.resource_id.id.split('/')[-1]
        if eid in selfs:
            print('Template event: Adding a Mw magnitude')
            det.magnitudes.append(
                Magnitude(mag=ML_to_Mw(det.magnitudes[0].mag, m=0.97, c=0.14),
                          magnitude_type='Mw',
                          comments=[Comment(text='Ristau et al., 2016 BSSA')]))
            continue
        tid = det.resource_id.id.split('/')[-1].split('_')[0]
        det_mo = Mw_to_M0([m.mag for m in det.magnitudes
                           if m.magnitude_type == 'Mw'][0])
        rel_mo = det_mo / temp_mag_dict[tid]['Old M0']
        new_det_mo = rel_mo * temp_mag_dict[tid]['New M0']
        new_det_Mw = (2. / 3. * np.log10(new_det_mo)) - 9.
        new_det_ML = (0.97 * new_det_Mw) + 0.14
        det.magnitudes.append(
            Magnitude(mag=new_det_Mw, magnitude_type='Mw',
                      comments=[Comment(text='rel_mo={}'.format(rel_mo))]))
        det.magnitudes.append(
            Magnitude(mag=new_det_ML, magnitude_type='ML',
                      comments=[Comment(text='rel_mo={}'.format(rel_mo))]))
        det.preferred_magnitude_id = det.magnitudes[-1].resource_id.id
    return

def Mw_to_M0(Mw):
    """Simple Hanks and Kanamori calc"""
    M0 = 10 ** (1.5 * (Mw + 9.0))
    return M0

def ML_to_Mw(ML, m=0.97, c=0.14):
    """Simple calc for ML to Mw using Ristau et al., 2016 scaling relation"""
    Mw = (ML - c) * m
    return Mw

def local_to_moment(mag, m=0.88, c=0.73):
    """
    From Gabe and/or Calum?
    Function to convert local magnitude to seismic moment - defaults to use
    the linear estimate from Ristau 2009 (BSSA) for shallow earthquakes in
    New Zealand.

    :type mag: float
    :param mag: Local Magnitude
    :type m: float
    :param m: The m in the relationship Ml = m * Mw + c
    :type c: constant
    :param c: See m
    """
    # Fist convert to moment magnitude
    Mw = ( mag - c ) / m
    # Then convert to seismic moment following (Hanks Kanamori)
    Moment = 10.0 ** (1.5 * Mw + 9.0 )
    return Moment

def local_to_moment_Majer(Ml):
    """
    Bypasses the calculation of Mw above and just uses the relation of Mo to Ml
    of Majer et all 1979 for the Geysers.
    :param Ml: Local magnitude
    :return:
    """
    Mo = (10**(17.27 + 0.77 * Ml)) * 1E-7 # Dyne.cm to N.m
    return Mo

def remove_outliers(M, ev_out, m=4):
    """
    helper function for outliers
    :param M:
    :param ev_out:
    :return:
    """
    import numpy as np
    new_M = []
    new_evs = []
    for i, m in enumerate(list(M)):
        if abs(m) <= 4.:
            print(m)
            new_M.append(M[i])
            new_evs.append(ev_out[i])
    return np.array(new_M), new_evs

def cc_coh_dets(streams, detections, length, corr_prepick, shift):
    # Loop over detections and return list of ccc with template
    # Trim all wavs to desired length
    # Assumes the first entry is template
    for i, st in enumerate(streams):
        det = detections[i]
        for tr in st:
            pk = [pk for pk in det.event.picks
                  if pk.waveform_id.get_seed_string() == tr.id][0]
            strt = pk.time - corr_prepick - (shift / 2.)
            tr.trim(starttime=strt, endtime=strt + length + (shift / 2.),
                    nearest_sample=True)
    # Clean out traces of different lengths
    len = Counter([(tr.id, tr.stats.npts)
                   for st in streams for tr in st]).most_common(1)[0][0][1]
    for st in streams:
        rms = [tr for tr in st if tr.stats.npts != len]
        for rm in rms:
            st.traces.remove(rm)
    coh, i = cross_chan_correlation(st1=streams[0], streams=streams[1:],
                                    shift_len=shift)
    return coh


def party_relative_mags(party, self_files, shift_len, align_len, svd_len,
                        reject, wav_dir, min_amps, calibrate=False,
                        method='PCA', plot_svd=False):
    """
    Calculate the relative moments for detections in a Family using
    mag_calc.svd_moments()

    :param party: Party of detections
    :param self_files: List of self-detection wav files (in order of families)
    :param shift_len: Maximum shift length used in waveform alignment
    :param align_len: Length of waveform used for correlation in alignment
    :param svd_len: Length of waveform used in relative amplitude calc
    :param reject: Min cc threshold for accepted measurement
    :param wav_dir: Root directory of waveforms
    :param min_amps: Minimum number of relative measurements per pair
    :param calibrate: Flag for calibration to a priori Ml's
    :param method: 'PCA' or 'LSQR'
    :param plot_svd: Bool to plot results of svd relative amplitude calcs
    :return:
    """
    pty = party.copy()
    for i, fam in enumerate(pty.families):
        print('Starting work on family %s' % fam.template.name)
        if len(fam) == 1:
            print('Only self-detection. Moving on.')
            continue
        temp = fam.template
        prepick = temp.prepick
        # Here we'll read in the waveforms and trim from stefan's directory
        # of SAC files so as not to duplicate data
        ev_files = glob('{}/*'.format(wav_dir))
        ev_files.sort()
        streams = []
        self_ind = [j for j, ev_f in enumerate(ev_files)
                    if ev_f == self_files[i]][0]
        # Read in Z components of events which we wrote for stefan
        # Many of these ev_dirs will not exist!
        for ev_w in ev_files:
            print('Reading %s' % ev_w)
            streams.append(read(ev_w))
        print('Moving self detection to top of list')
        # Move the self detection to the first element
        streams.insert(0, streams.pop(self_ind))
        print('Template Stream: %s' % str(streams[0]))
        if len(streams[0]) == 0:
            print('Template %s waveforms did not get written. Investigate.' %
                  temp.name)
            continue
        # Process streams then copy to both ccc_streams and svd_streams
        print('Shortproc-ing streams')
        for st in streams:
            rms = [tr for tr in st if tr.stats.sampling_rate < temp.samp_rate]
            for rm in rms:
                st.traces.remove(rm)
            shortproc(st=st, lowcut=temp.lowcut,
                      highcut=temp.highcut, filt_order=temp.filt_order,
                      samp_rate=temp.samp_rate)
        # Remove all traces with no picks before copying
        for str_ind, st in enumerate(streams):
            det = fam.detections[str_ind]
            rms = []
            for tr in st:
                try:
                    [pk for pk in det.event.picks
                     if pk.waveform_id.get_seed_string() == tr.id][0]
                except IndexError:
                    rms.append(tr)
            for rm in rms:
                st.traces.remove(rm)
        print('Copying streams')
        wrk_streams = copy.deepcopy(streams)
        svd_streams = copy.deepcopy(streams) # For svd
        ccc_streams = copy.deepcopy(streams)
        # work out cccoh for each event with template
        cccohs = cc_coh_dets(streams=ccc_streams, detections=fam.detections,
                             length=svd_len, corr_prepick=prepick,
                             shift=shift_len)
        for eind, st in enumerate(wrk_streams):
            det = fam.detections[eind]
            for tr in st:
                pk = [pk for pk in det.event.picks
                      if pk.waveform_id.get_seed_string() == tr.id][0]
                tr.trim(starttime=pk.time - prepick - shift_len,
                        endtime=pk.time + shift_len + align_len)
        st_seeds = list(set([tr.id for st in wrk_streams for tr in st]))
        st_seeds.sort()
        # Align streams with just P arrivals, then use longer st for svd
        print('Now aligning svd_streams')
        shift_inds = int(shift_len * fam.template.samp_rate)
        for st_seed in st_seeds:
            trs = []
            for i, st in enumerate(wrk_streams):
                if len(st.select(id=st_seed)) > 0:
                    trs.append((i, st.select(id=st_seed)[0]))
            inds, traces = zip(*trs)
            shifts, ccs = stacking.align_traces(trace_list=list(traces),
                                                shift_len=shift_inds,
                                                positive=True,
                                                master=traces[0].copy())
            # We now have shifts based on P correlation, shift and trim
            # larger wavs for svd
            for j, shift in enumerate(shifts):
                st = svd_streams[inds[j]]
                det = fam.detections[inds[j]]
                if ccs[j] < reject:
                    svd_streams[inds[j]].remove(st.select(id=st_seed)[0])
                    print('Removing trace due to low cc value: %s' % ccs[j])
                    continue
                pk = [pk for pk in det.event.picks
                      if pk.waveform_id.get_seed_string() == st_seed][0]
                strt_tr = pk.time - prepick - shift
                st.select(id=st_seed)[0].trim(strt_tr, strt_tr + svd_len)
        if method == 'LSQR':
            print('Using least-squares method')
            event_list = []
            for st_id in st_seeds:
                st_list = []
                for stind, st in enumerate(svd_streams):
                    if len(st.select(id=st_id)) > 0:
                        st_list.append(stind)
                event_list.append(st_list)
            # event_list = np.asarray(event_list).tolist()
            u, sigma, v, sta_chans = svd(stream_list=svd_streams, full=True)
            try:
                M, events_out = svd_moments(u, sigma, v, sta_chans, event_list)
            except IOError as e:
                print('Family %s raised error %s' % (fam.template.name, e))
                return
        elif method == 'PCA':
            print('Using principal component method')
            M, events_out = svd_relative_amps(fam, svd_streams, min_amps,
                                              plot=plot_svd)
            print(M, events_out)
        else:
            print('{} not valid argument for mag calc method'.format(method))
            return
        # If we have a Mag for template, calibrate moments
        if calibrate and len(fam.template.event.magnitudes) > 0:
            print('Converting relative amps to magniutdes')
            # Convert the template magnitude to seismic moment
            temp_mag = fam.template.event.magnitudes[-1].mag
            temp_mo = local_to_moment(temp_mag)
            # Extrapolate from the template moment - relative moment relationship to
            # Get the moment for relative moment = 1.0
            norm_mo = temp_mo / M[0]
            # Template is the last event in the list
            # Now these are weights which we can multiple the moments by
            moments = np.multiply(M, norm_mo)
            # Now convert to Mw
            Mw = [2.0 / 3.0 * (np.log10(m) - 9.0) for m in moments]
            # Mw2, evs2 = remove_outliers(Mw, events_out)
            # Convert to local
            Ml = [0.88 * m + 0.73 for m in Mw]
            #Normalize moments to template mag
            # Add calibrated mags to detection events
            for jabba, eind in enumerate(events_out):
                fam.detections[eind].event.magnitudes = [
                    Magnitude(mag=Mw[jabba], magnitude_type='Mw')]
                fam.detections[eind].event.comments.append(
                    Comment(text=str(cccohs[eind])))
                fam.detections[eind].event.magnitudes.append(
                    Magnitude(mag=Ml[jabba], magnitude_type='ML'))
                fam.detections[eind].event.preferred_magnitude_id = (
                    fam.detections[eind].event.magnitudes[-1].resource_id.id)
    return pty, cccohs


def svd_relative_amps(fam, streams, min_amps, plot):
    """
    Calculate the relative amplitudes using svd between template and detections

    :param streams: List of streams with first being the template
        These need to be prepared beforehand
    :return:
    """
    template = streams[0]
    M = []
    events_out = []
    for svd_ind, st in enumerate(streams):
        if len(st) == 0:
            print('Event not located, skipping')
            continue
        ev_r_amps = []
        # For each pair of template:detection (including temp:temp)
        if plot:
            subplots = len([t for t in template if len(st.select(id=t.id)) > 0])
            fig, ax = plt.subplots(nrows=subplots * 2,
                                   figsize=(5, 15), sharex='col')
        ctr = 0  # Axes incrementer
        for tr_ind, tr in enumerate(template):
            if len(st.select(id=tr.id)) > 0:
                det_tr = st.select(id=tr.id)[0]
                # Convoluted way of getting two 'vert' vectors
                data_mat = np.vstack((tr.data, det_tr.data)).T
                U, sig, Vt = scipy.linalg.svd(data_mat,
                                              full_matrices=True)
                # Vt is 2x2 for two events
                # Per Shelly et al., 2016 eq. 4
                ev_r_amps.append(Vt[0][1] / Vt[0][0])
                if plot:
                    ax_i = ctr
                    # Time vector
                    time = np.arange(tr.data.shape[0]) * tr.stats.delta
                    ax[ax_i].plot(time, tr.data, color='r',
                                  label='Template' if tr_ind == 0 else "")
                    ax[ax_i].plot(time, det_tr.data, color='b',
                                  label='Detection' if tr_ind == 0 else "")
                    ax[ax_i].annotate(xy=(0.03, 0.7), text=tr.id, fontsize=8,
                                        xycoords='axes fraction')
                    ax[ax_i + 1].plot(time, tr.data / np.linalg.norm(tr.data),
                                      color='k')
                    ax[ax_i + 1].plot(time,
                                      det_tr.data / np.linalg.norm(det_tr.data),
                                      color='steelblue')
                    ax[ax_i + 1].plot(time, U[0] * Vt[0][0], color='goldenrod',
                                      label='1st SV' if tr_ind == 0 else "")
                    ax[ax_i].set_yticklabels([])
                    ax[ax_i + 1].set_yticklabels([])
                    ctr += 2
        if len(ev_r_amps) < min_amps:
            print('Fewer than {} amp picks, skipping.'.format(min_amps))
            if plot:
                plt.close('all')
            continue
        M.append(np.median(ev_r_amps))
        events_out.append(svd_ind)
        if plot:
            fig.legend()
            fig.suptitle('{}: {:0.3f}'.format(
                fam.detections[svd_ind].detect_time.strftime(
                    '%Y/%m/%dT%H:%M:%S'),
                np.median(ev_r_amps)))
            p_nm = '{}_svd_plot.png'.format(fam.detections[svd_ind].detect_time)
            ax[-1].set_xlabel('Time [sec]')
            ax[-1].margins(x=0)
            plt.savefig(p_nm, dpi=300)
    return M, events_out

