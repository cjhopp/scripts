#!/usr/bin/env python
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150

"""
This is a series of functions for weighting of picks made by cross-correlation of template
and network waveforms at time of matched filter detection
"""


def create_template_dict(temp_dir):
    from glob import glob
    from obspy import read

    temp_files = glob(temp_dir)
    template_dict = {}
    for filename in temp_files:
        temp_name = filename.split('/')[-1].rstrip('.mseed')
        template_dict[temp_name] = read(filename)
    return template_dict

def weight_corr_picks(cat, temp_dict=None, stream_dict=None, method='SNR_temp', temp_cat=False, show_ccs=False):
    """
    Implementing pick weighting by SNR of cc function
    :param cat: catalog of detections
    :param template_dir: directory where the templates live
    :param stream: directory where the longer event waveforms
    :param method: 'SNR_temp', 'SNR_ccval', 'doub_diff_ccval'
    :return: obspy.core.event.Catalog

    ** Note: Hypoellipse default quality mapping: 0.05, 0.1, 0.2, 0.4 sec (0, 1, 2, 3?)
             We can go with these values based on SNR of cc function, I suppose.
    """
    import warnings
    import numpy as np
    from eqcorrscan.core.match_filter import normxcorr2
    import matplotlib.pyplot as plt
    from obspy.core.event import QuantityError

    SNRs = []
    if temp_cat:
        temp_SNRs = {str(ev.resource_id).split('/')[-1].split('_')[0]:
                         {amp.waveform_id.station_code + '.EZ':
                              amp.snr for amp in ev.amplitudes} for ev in temp_cat}
    for ev in cat:
        det_rid = str(ev.resource_id).split('/')[-1]
        temp_rid = str(ev.resource_id).split('/')[-1].split('_')[0]
        picks = list(ev.picks)
        for i, pk in enumerate(picks):
            if pk.phase_hint == 'P':
                sta = pk.waveform_id.station_code
                chan = pk.waveform_id.channel_code
                if method == 'SNR_ccval':
                    temp = temp_dict[temp_rid + '_1sec']
                    stream = stream_dict[det_rid]
                    tr = temp.select(station=sta,
                                     channel=chan)[0]
                    st_tr = stream.select(station=sta,
                                          channel=chan)[0]
                    ccc = normxcorr2(tr.data, st_tr.data)[0]
                    pk_samp = int(tr.stats.sampling_rate * (pk.time - st_tr.stats.starttime) - 5)
                    sta_start = pk_samp - 5
                    sta_end = pk_samp + 5
                    LTA = np.std(ccc)
                    STA = np.std(ccc[sta_start:sta_end])
                    # STA = abs(ccc[pk_samp])
                    SNR = STA / LTA
                    # Here we map the ccval SNR to time uncertainty in the original catalog
                    orig_pk = ev.picks[i]
                    if SNR < 0.75: orig_pk.time_errors = QuantityError(uncertainty=0.40)
                    elif SNR < 1.25: orig_pk.time_errors = QuantityError(uncertainty=0.20)
                    elif SNR < 1.75: orig_pk.time_errors = QuantityError(uncertainty=0.10)
                    elif SNR < 2.25: orig_pk.time_errors = QuantityError(uncertainty=0.05)
                    else: orig_pk.time_errors = QuantityError(uncertainty=0.01)
                    SNRs.append(SNR)
                    if show_ccs:
                        fig, ax = plt.subplots()
                        ax.plot(ccc)
                        ax.set_title('%s.%s: %f' % (sta, chan, SNR))
                        fig.show()
                elif method == 'SNR_temp':
                    orig_pk = ev.picks[i]
                    try:
                        SNR = temp_SNRs[temp_rid]['%s.%s' % (sta, chan)]
                    except KeyError:
                        warnings.warn('%s.%s has no amplitude pick' % (sta, chan))
                        orig_pk.time_errors = QuantityError(uncertainty=0.10)
                        continue
                    if SNR < 1.0:
                        orig_pk.time_errors = QuantityError(uncertainty=0.20)
                    elif SNR < 2.:
                        orig_pk.time_errors = QuantityError(uncertainty=0.10)
                    elif SNR < 5.:
                        orig_pk.time_errors = QuantityError(uncertainty=0.05)
                    else:
                        orig_pk.time_errors = QuantityError(uncertainty=0.01)
    return cat


def remove_unwanted_picks(cat, cc_thresh):
    for ev in cat:
        cp_picks = list(ev.picks)
        for pk in cp_picks:
            if pk.phase_hint == 'P':
                if float(pk.comments[0].text.split('=')[-1]) < cc_thresh:
                    ev.picks.remove(pk)
    return cat


def plot_rand_correlation(cat, d_thresh, temp_dict, stream_dict):
    """
    Calculate cross correlation coefficients for events located further than d_thresh from
    each other. This should represent correlation of uncorrelated signals and help determine
    ccval_cutoff
    :param cat:
    :param dist_thresh:
    :param temp_dict:
    :param stream_dict:
    :return:
    """
    from eqcorrscan.utils.mag_calc import dist_calc
    from eqcorrscan.core.match_filter import normxcorr2
    import numpy as np

    corrs = []
    for i, ev in enumerate(cat):
        print i
        ev_tup = (ev.preferred_origin().latitude, ev.preferred_origin().longitude,
                  ev.preferred_origin().depth / 1000.)
        for ev2 in cat[i+1:]:
            ev_tup2 = (ev2.preferred_origin().latitude, ev2.preferred_origin().longitude,
                      ev2.preferred_origin().depth / 1000.)
            dist = dist_calc(ev_tup, ev_tup2)
            if dist > d_thresh:
                det_rid = str(ev2.resource_id).split('/')[-1]
                temp_rid = str(ev.resource_id).split('/')[-1].split('_')[0]
                temp = temp_dict[temp_rid + '_1sec']
                stream = stream_dict[det_rid]
                for pk in ev.picks:
                    if pk.phase_hint == 'P':
                        sta = pk.waveform_id.station_code
                        chan = pk.waveform_id.channel_code
                        if len(temp.select(station=sta, channel=chan)) > 0:
                            tr = temp.select(station=sta,
                                             channel=chan)[0]
                        else:
                            continue
                        if len(stream.select(station=sta, channel=chan)) > 0:
                            st_tr = stream.select(station=sta,
                                                  channel=chan)[0]
                        else:
                            continue
                        # # still correcting for 0.1 sec pre-pick time here...gross
                        # pk_samp =
                        # corr_start = pk_samp - 5
                        # corr_end = pk_samp + 6
                        ccc = normxcorr2(tr.data, st_tr.data[140:201])[0]
                        corrs.append(max(ccc.max(), ccc.min(), key=abs))
    return corrs
