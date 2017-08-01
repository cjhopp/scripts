#!/usr/bin/python
from __future__ import division

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
    # Then convert to seismic moment following standard convention
    Moment = 10.0 ** (1.5 * Mw + 9.0 )
    return Moment


def cc_coh_dets(streams, length, wav_prepick, corr_prepick, shift):
    # Loop over detections and return list of cc_coh with template
    # Trim all wavs to desired length
    # Assumes the first entry is template
    from eqcorrscan.utils.clustering import cross_chan_coherence

    for st in streams:
        for tr in st:
            strt = tr.stats.starttime + wav_prepick - corr_prepick - (shift / 2.)
            tr.trim(starttime=strt, endtime=strt + length + (shift / 2.))
    cccohs = []
    for stream in streams[1:]:
        coh, i = cross_chan_coherence(st1=streams[0], st2=stream,
                                      allow_shift=True, shift_len=shift)
        cccohs.append(coh)
    return cccohs


def party_relative_mags(party, shift_len, align_len, svd_len, reject, sac_dir,
                        calibrate=False, method='PCA'):
    """
    Calculate the relative moments for detections in a Family using
    mag_calc.svd_moments()

    :param party: Party of detections
    :param shift_len: Maximum shift length used in waveform alignment
    :param align_len: Length of waveform used for correlation in alignment
    :param svd_len: Length of waveform used in relative amplitude calc
    :param reject: Min cc threshold for accepted measurement
    :param sac_dir: Root directory of waveforms
    :param calibrate: Boolean flag for calibration to a priori Ml's
    :return:
    """
    import copy
    import scipy
    import numpy as np
    from glob import glob
    from obspy import read, Catalog, Stream
    from obspy.core.event import Magnitude, Comment
    from eqcorrscan.core.match_filter import normxcorr2
    from eqcorrscan.utils import stacking
    from eqcorrscan.utils.pre_processing import shortproc
    from eqcorrscan.utils.clustering import svd
    from eqcorrscan.utils.mag_calc import svd_moments

    for fam in party.families:
        temp = fam.template
        prepick = temp.prepick
        events = [det.event for det in fam.detections]
        # Find out which detection is the self detection and its index
        self_det = [(det, i) for i, det in enumerate(fam.detections)
                    if det.detect_val / det.no_chans == 1.0][0]
        # Here we'll read in the waveforms and trim from stefan's directory
        # of SAC files so as not to duplicate data
        ev_dirs = ['%s/%s' % (sac_dir, str(ev.resource_id).split('/')[-1])
                   for ev in events]
        # Maybe have to write these as list comprehensions...
        streams = []
        for ev_dir in ev_dirs:
            raw_st = Stream()
            for wav_file in glob('%s/*EHZ*' % ev_dir):
                raw_tr = read(wav_file)[0]
                print(raw_tr.stats.sac['a'])
                raw_tr.trim(starttime=raw_tr.stats.starttime +
                                      raw_tr.stats.sac['a'] - 3.,
                            endtime=raw_tr.stats.endtime)
                raw_st += raw_tr
            streams.append(raw_st)
        streams.insert(0, fam.template.st)
        front_clip = prepick - (shift_len) - 0.05
        back_clip = prepick - 0.05 + align_len + (shift_len)
        wrk_streams = copy.deepcopy(streams) # For aligning
        # Process streams then copy to both ccc_streams and svd_streams
        for st in streams:
            shortproc(st=st, lowcut=temp.lowcut, highcut=temp.highcut,
                      filt_order=temp.filt_order, samp_rate=temp.samp_rate)
        svd_streams = copy.deepcopy(streams) # For svd
        ccc_streams = copy.deepcopy(streams)
        # work out cccoh for each event with template
        cccohs = cc_coh_dets(streams=ccc_streams, shift=shift_len,
                             length=svd_len, wav_prepick=3.,
                             corr_prepick=0.05)
        for st in wrk_streams:
            for tr in st:
                tr.trim(starttime=tr.stats.starttime + front_clip,
                        endtime=tr.stats.starttime + back_clip)
        st_chans = list(set([(tr.stats.station, tr.stats.channel)
                             for st in wrk_streams for tr in st]))
        st_chans.sort()
        # Align streams with just P arrivals, then use longer st for svd
        shift_inds = int(shift_len * fam.template.samp_rate)
        for st_chan in st_chans:
            self_det_trace = False
            trs = []
            for i, st in enumerate(wrk_streams):
                if len(st.select(station=st_chan[0], channel=st_chan[-1])) > 0:
                    trs.append((i, st.select(station=st_chan[0],
                                             channel=st_chan[-1])[0]))
                    # If template has trace for stachan, note the index
                    if i == self_det[1]:
                        self_det_trace = True
                        mast_ind = len(trs) - 1
            inds, traces = zip(*trs)
            if self_det_trace:
                shifts, ccs = stacking.align_traces(trace_list=list(traces),
                                                    shift_len=shift_inds,
                                                    positive=True,
                                                    master=traces[mast_ind].copy)
            else:
                shifts, ccs = stacking.align_traces(trace_list=list(traces),
                                                    shift_len=shift_inds,
                                                    positive=True)
            # We not have shifts based on P correlation, shift and trim
            # larger wavs for svd
            for j, shift in enumerate(shifts):
                st = svd_streams[inds[j]]
                if ccs[j] < reject:
                    svd_streams[inds[j]].remove(st.select(
                        station=st_chan[0], channel=st_chan[-1])[0])
                    print('Removing stream due to low cc value: %s' % ccs[j])
                    continue
                strt_tr = st.select(
                    station=st_chan[0], channel=st_chan[-1])[0].stats.starttime
                strt_tr += prepick
                strt_tr -= shift
                st.select(station=st_chan[0],
                          channel=st_chan[-1])[0].trim(strt_tr,strt_tr
                                                       + svd_len)
        if method == 'LSQR':
            event_list = []
            for stachan in st_chans:
                st_list = []
                for i, st in enumerate(svd_streams):
                    if len(st.select(station=stachan[0],
                                     channel=stachan[-1])) > 0:
                        st_list.append(i)
                event_list.append(st_list)
            # event_list = np.asarray(event_list).tolist()
            u, sigma, v, sta_chans = svd(stream_list=svd_streams, full=True)
            try:
                M, events_out = svd_moments(u, sigma, v, sta_chans, event_list)
            except IOError as e:
                print('Family %s raised error %s' % (fam.template.name, e))
                continue
        elif method == 'PCA':
            # Now loop over all detections and do svd for each matching
            # chan with temp
            template = svd_streams[self_det[1]]
            M = []
            for i, st in enumerate(svd_streams):
                ev_r_amps = []
                for tr in template:
                    if len(st.select(station=tr.stats.station,
                                     channel=tr.stats.channel)) > 0:
                        det_tr = st.select(station=tr.stats.station,
                                           channel=tr.stats.channel)[0]
                        # Check if the ccval is above thresh
                        if normxcorr2(tr.data, det_tr.data) > reject:
                            data_mat = np.hstack(tr.data, det_tr)
                            U, sig, Vt = scipy.linalg.svd(data_mat,
                                                          full_matrices=True)
                            ev_r_amps.append(Vt[0][1] / Vt[0][0])
                if len(ev_r_amps) < 6:
                    print('Fewer than 6 amplitude picks, skipping.')
                    continue
                M.append(np.median(ev_r_amps))
                events_out.append(i)
        # If we have a Mag for template, calibrate moments
        if calibrate and len(fam.template.event.magnitudes) > 0:
            # Convert the template magnitude to seismic moment
            temp_mag = fam.template.event.magnitudes[-1].mag
            temp_mo = local_to_moment(temp_mag)
            # Extrapolate from the template moment - relative moment relationship to
            # Get the moment for relative moment = 1.0
            norm_mo = temp_mo / M[0]
            # Template is the last event in the list
            # Now these are weights which we can multiple the moments by
            moments = M * norm_mo
            # Now convert to Mw
            Mw = [2.0 / 3.0 * (np.log10(m) - 9.0) for m in moments]
            Mw2, evs2 = remove_outliers(Mw, events_out)
            # Convert to local
            Ml = [0.88 * m + 0.73 for m in Mw2]
            #Normalize moments to template mag
            # Add calibrated mags to detection events
            for i, eind in enumerate(evs2):
                fam.detections[eind-1].event.magnitudes = [Magnitude(mag=Mw2[i],
                                                                     magnitude_type='Mw')]
                fam.detections[eind-1].event.comments.append(
                    Comment(text=str(cccohs[eind-1])))
                # fam.detections[eind-1].event.magnitudes.append(Magnitude(mag=Ml[i],
                #                                                          magnitude_type='ML'))
            fam.catalog = Catalog(events=[det.event for det in fam.detections])
    return party, cccohs


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


def shelly_mags(party, wav_dir):
    """
    Implementation of David Shelly's relative amplitude-based calculations
    :param party:
    :return:
    """
    from obspy import read

    for fam in party:
        prepick = fam.template.prepick
        events = [det.event for det in fam.detections]
        wav_files = ['%s/%s.mseed' % (wav_dir,
                                      str(ev.resource_id).split('/')[-1])
                     for ev in events]
        streams = [read(wav_file) for wav_file in wav_files]
    return