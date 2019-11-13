"""
Functions to calculate magnitudes for families of near-repeating earthquakes
using singular value decomposition.  Magnitudes will be relative to the template
event, so ensure that the template has a local magnitude calculated for it.

Written by Calum Chamberlain 28 October 2015 - uses the EQcorrscan package.
"""

import sys
# sys.path.insert(0, '/home/calumch/my_programs/Building/EQcorrscan')
import matplotlib.pyplot as plt
from eqcorrscan.utils.mag_calc import SVD_moments
from eqcorrscan.utils import Sfile_util

def family_calc(template, detections, wavdir, cut=(-0.5, 3.0),\
                freqmin=5.0, freqmax=15.0, corr_thresh=0.9, \
                template_pre_pick=0.1, samp_rate=100.0, plotvar=False,\
                resample=True):
    """
    Function to calculate the magnitudes for a given family, where the template
    is an s-file with a magnitude (and an appropriate waveform in the same
    directory), and the detections is a list of s-files for that template.

    :type template: str
    :param template: path to the template for this family
    :type detections: List of str
    :param detections: List of paths for s-files detected for this family
    :type wavdir: str
    :param wavdir: Path to the detection waveforms
    :type cut: tuple of float
    :param cut: Cut window around P-pick
    :type freqmin: float
    ;param freqmin: Low-cut in Hz
    :type freqmax: float
    :param freqmin: High-cut in Hz
    :type corr_thresh: float
    :param corr:thresh: Minimum correlation (with stack) for use in SVD
    :type template_pre_pick: float
    :param template_pre_pick: Pre-pick used for template in seconds
    :type samp_rate: float
    :param samp_rate: Desired sampling rate in Hz

    :returns: np.ndarry of relative magnitudes
    """
    from obspy import read, Stream
    from eqcorrscan.utils import stacking, clustering
    from eqcorrscan.core.match_filter import normxcorr2
    import numpy as np
    from obspy.signal.cross_correlation import xcorr

    # First read in the template and check that is has a magnitude
    template_mag = Sfile_util.readheader(template).Mag_1
    template_magtype = Sfile_util.readheader(template).Mag_1_type
    if template_mag=='nan' or template_magtype != 'L':
        raise IOError('Template does not have a local magnitude, calculate this')

    # Now we need to load all the waveforms and picks
    all_detection_streams=[] # Empty list for all the streams
    all_p_picks=[] # List for all the P-picks
    event_headers=[] # List of event headers which we will return
    for detection in detections:
        event_headers.append(Sfile_util.readheader(detection))
        d_picks=Sfile_util.readpicks(detection)
        try:
            d_stream=read(wavdir+'/'+Sfile_util.readwavename(detection)[0])
        except IOError:
            # Allow for seisan year/month directories
            d_stream=read(wavdir+'/????/??/'+Sfile_util.readwavename(detection)[0])
        except:
            raise IOError('Cannot read waveform')
        # Resample the stream
        if resample:
            d_stream = d_stream.detrend('linear')
            d_stream = d_stream.resample(samp_rate)
        # We only want channels with a p-pick, these should be vertical channels
        picked=[]
        p_picks=[]
        for pick in d_picks:
            if pick.phase=='P':
                p_picks.append(pick)
                tr=d_stream.select(station=pick.station,\
                                   channel='??'+pick.channel[-1])
                if len(tr) >= 1:
                    tr=tr[0]
                else:
                    print 'No channel for pick'
                    print pick
                    break
                # Filter the trace
                tr=tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
                # Trim the trace around the P-pick
                tr.trim(pick.time+cut[0]-0.05, pick.time+cut[1]+0.5)
                picked.append(tr)
        picked=Stream(picked)
        # Add this to the list of streams
        all_detection_streams.append(picked)
        all_p_picks.append(p_picks)
    # Add the template in
    template_stream = read('/'.join(template.split('/')[0:-1])+'/'+\
                           Sfile_util.readwavename(template)[0])
    # Resample
    if resample:
        template_stream = template_stream.detrend('linear')
        template_stream = template_stream.resample(samp_rate)
    template_picks = Sfile_util.readpicks(template)
    picked=[]
    p_picks=[]
    for pick in template_picks:
        pick.time-=template_pre_pick
        if pick.phase=='P':
            p_picks.append(pick)
            tr=template_stream.select(station=pick.station,\
                                   channel='??'+pick.channel[-1])
            if len(tr) >= 1:
                tr=tr[0]
                # Filter the trace
                tr=tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
                # Trim the trace around the P-pick
                tr.trim(pick.time+cut[0]-0.05, pick.time+cut[1]+0.5)
                picked.append(tr)
            else:
                print 'No channel for pick'
                print pick
    all_detection_streams.append(Stream(picked))
    print ' I have read in '+str(len(all_detection_streams))+' streams of data'
    all_p_picks.append(p_picks)
    # We now have a list of bandpassed, trimmed streams for all P-picked channels
    # Lets align them
    stachans=[tr.stats.station+'.'+tr.stats.channel\
              for st in all_detection_streams for tr in st]
    stachans=list(set(stachans))
    for i in range(len(stachans)):
        chan_traces=[]
        chan_pick_indexes=[] # Need this for next crop
        for j, detection_stream in enumerate(all_detection_streams):
            stachan=stachans[i]
            # If there is a pick/data for this channel then add it to the list
            detection_trace=detection_stream.select(station=stachan.split('.')[0],\
                                                    channel=stachan.split('.')[1])
            if len(detection_trace)==1:
                chan_traces.append(detection_trace[0])
                chan_pick_indexes.append(j)
            elif len(detection_trace) > 1:
                print 'More than one trace for '+stachan
                chan_traces.append(detection_trace[0])
                chan_pick_indexes.append(j)
        # shiftlen=int(0.4 * (cut[1] - cut[0]) * chan_traces[0].stats.sampling_rate)
        # shiftlen=400
        shiftlen=10
        shifts, ccs = stacking.align_traces(chan_traces, shiftlen,\
                                       master=chan_traces[-1])
                                       # master=master)
        # Shift by up to 0.5s
        # Ammend the picks using the shifts
        for j in range(len(shifts)):
            shift=shifts[j]
            pick_index=chan_pick_indexes[j] # Tells me which stream to look at
            for pick in all_p_picks[pick_index]:
                if pick.station==stachan.split('.')[0]:# and\
                   # pick.channel=='*'+stachan.split('.')[1][-1]:
                    pick.time-=shift
                    print 'Shifting '+pick.station+' by '+str(shift)+\
                            ' for correlation at '+str(ccs[j])
    # We now have amended picks, now we need to re-trim to complete the alignment
    for i in range(len(all_detection_streams)):
        for j in range(len(all_detection_streams[i])):
            all_detection_streams[i][j].trim(all_p_picks[i][j].time+cut[0], \
                                             all_p_picks[i][j].time+cut[1], \
                                             pad=True, fill_value=0,\
                                             nearest_sample=True)
    # Do a real small-scale adjustment, the stack will be better now
    # for i in range(len(stachans)):
        # chan_traces=[]
        # chan_pick_indexes=[] # Need this for next crop
        # for j, detection_stream in enumerate(all_detection_streams):
            # stachan=stachans[i]
            # # If there is a pick/data for this channel then add it to the list
            # detection_trace=detection_stream.select(station=stachan.split('.')[0],\
                                                    # channel=stachan.split('.')[1])
            # if len(detection_trace)==1:
                # chan_traces.append(detection_trace[0])
                # chan_pick_indexes.append(j)
            # elif len(detection_trace) > 1:
                # print 'More than one trace for '+stachan
                # chan_traces.append(detection_trace[0])
                # chan_pick_indexes.append(j)
        # master=stacking.linstack([Stream(tr) for tr in chan_traces])[0]
        # shifts, ccs = stacking.align_traces(chan_traces, 10,\
                                       # master=master)
        # # Shift by up to 0.5s
        # # Ammend the picks using the shifts
        # for j in range(len(shifts)):
            # shift=shifts[j]
            # pick_index=chan_pick_indexes[j] # Tells me which stream to look at
            # for pick in all_p_picks[pick_index]:
                # if pick.station==stachan.split('.')[0]:# and\
                   # # pick.channel=='*'+stachan.split('.')[1][-1]:
                    # pick.time-=shift
                    # print 'Shifting '+pick.station+' by '+str(shift)+\
                            # ' for correlation at '+str(ccs[j])
    # # We now have amended picks, now we need to re-trim to complete the alignment
    # for i in range(len(all_detection_streams)):
        # for j in range(len(all_detection_streams[i])):
            # all_detection_streams[i][j].trim(all_p_picks[i][j].time+cut[0], \
                                             # all_p_picks[i][j].time+cut[1], \
                                             # pad=True, fill_value=0,\
                                             # nearest_sample=True)


    #--------------------------------------------------------------------------
    # Now we have completely aligned traces:
    # We need to remove poorly correlated traces before we compute the SVD
    # We also want to record which stachans have channels for which events
    stachan_event_list=[]
    for stachan in stachans:
        chan_traces=[]
        event_list=[]
        final_event_list=[] # List for the final indexes of events for this stachan
        for i in range(len(all_detection_streams)):
            # Extract channel
            st=all_detection_streams[i]
            tr=st.select(station=stachan.split('.')[0],\
                         channel=stachan.split('.')[1])
            if not len(tr) == 0:
                chan_traces.append(tr[0])
                event_list.append(i)
        # enforce fixed length
        for tr in chan_traces:
            tr.data=tr.data[0:int( tr.stats.sampling_rate * \
                                  ( cut[1] - cut[0] ))]
        # Compute the stack and compare to this
        chan_traces=[Stream(tr) for tr in chan_traces]
        # stack=stacking.linstack(chan_traces)
        stack=chan_traces[-1]
        chan_traces=[st[0] for st in chan_traces]
        if plotvar:
            fig, axes = plt.subplots(len(chan_traces)+1, 1, sharex=True,\
                                     figsize=(7, 12))
            axes=axes.ravel()
            axes[0].plot(stack[0].data, 'r', linewidth=1.5)
            axes[0].set_title(chan_traces[0].stats.station+'.'+\
                              chan_traces[0].stats.channel)
            axes[0].set_ylabel('Stack')
        for i, tr in enumerate(chan_traces):
            if plotvar:
                axes[i+1].plot(tr.data, 'k', linewidth=1.5)
            # corr = normxcorr2(tr.data.astype(np.float32),\
                              # stack[0].data.astype(np.float32))
            dummy, corr = xcorr(tr.data.astype(np.float32),\
                                 stack[0].data.astype(np.float32), 1)
            corr=np.array(corr).reshape(1,1)
            if plotvar:
                axes[i+1].set_ylabel(str(round(corr[0][0],2)))
            if corr[0][0] < corr_thresh:
                # Remove the channel
                print str(corr)+' for channel '+tr.stats.station+'.'+\
                        tr.stats.channel+' event '+str(i)
                all_detection_streams[event_list[i]].remove(tr)
            else:
                final_event_list.append(event_list[i])
        if plotvar:
           plt.show()
        # We should require at-least three detections per channel used
        # Compute the SVD
        if len(final_event_list) >= 3:
            stachan_event_list.append((stachan, final_event_list))
        else:
            for i in range(len(all_detection_streams)):
                tr=all_detection_streams[i].select(station=stachan.split('.')[0])
                if not len(tr) == 0:
                    all_detection_streams[i].remove(tr[0])
    # Remove empty streams
    filled_streams=[]
    for stream in all_detection_streams:
        if not len(stream) == 0:
            filled_streams.append(stream)
    all_detection_streams = filled_streams
    # Now we have the streams that are highly enough correlated and the list of
    # which events these correspond to
    print len(all_detection_streams)
    print stachan_event_list
    if len(all_detection_streams) > 0 and len(all_detection_streams[0]) > 0:
        V, s, U, out_stachans = clustering.SVD(all_detection_streams)
        # Reorder the event list
        event_list=[]
        event_stachans=[]
        for out_stachan in out_stachans:
            for stachan in stachan_event_list:
                if stachan[0] == out_stachan:
                    event_list.append(stachan[1])
                    event_stachans.append(stachan[0])
                    print len(stachan[1])
        print event_list
        relative_moments, event_list = SVD_moments(U, s, V, event_stachans,\
                                                   event_list)
        print '\n\nRelative moments: '
        print relative_moments
        for stachan in stachan_event_list:
            print stachan
        # Now we have the relative moments for all appropriate events - this should
        # include the template event also, which has a manually determined magnitude
        # Check that we have got the template event
        if not event_list[-1] == len(detections):
            print 'Template not included in relative magnitude, fail'
            print 'Largest event in event_list: '+str(event_list[-1])
            print 'You gave me '+str(len(detections))+' detections'
            return False
        # Convert the template magnitude to seismic moment
        template_moment = local_to_moment(template_mag)
        # Extrapolate from the template moment - relative moment relationship to
        # Get the moment for relative moment = 1.0
        norm_moment = template_moment / relative_moments[-1]
        # Template is the last event in the list
        # Now these are weights which we can multiple the moments by
        moments = relative_moments * norm_moment
        print 'Moments '
        print moments
        # Now convert to Mw
        Mw = [2.0/3.0 * (np.log10(M) - 9.0 ) for M in moments]
        print 'Moment magnitudes: '
        print Mw
        # Convert to local
        Ml = [ 0.88 * M + 0.73 for M in Mw ]
        print 'Local magnitudes: '
        print Ml
        print 'Template_magnitude: '
        print template_mag
        i=0
        for event_id in event_list[0:-1]:
            print event_id
            print Ml[i]
            event_headers[event_id].Mag_2=Ml[i]
            event_headers[event_id].Mag_2_type='S'
            i+=1
        # return event_headers
    else:
        print 'No useful channels'
        print all_detection_streams
        # return False
def local_to_moment(mag, m=0.88, c=0.73):
    """
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


if __name__=='__main__':
    print 'Testing'
    import glob
    import datetime as dt
    import numpy as np
    all_events=[]
    # templates=glob.glob('../Re_weighted_masters/*L.S??????')
    templates=glob.glob('../Re_weighted_masters/08-0830-50L.S201303')
    # templates=glob.glob('../Re_weighted_masters/17-1336-15L.S201304')
    for template in templates:
        print template
        template_time=dt.datetime.strptime(template.split('/')[-1],\
                                           '%d-%H%M-%SL.S%Y%m')
        template_name=template_time.strftime('%Y-%m-%d-')+\
                str(int((template_time - dt.datetime.combine(template_time.date(),\
                                                         dt.time(0))).total_seconds()))
        detections=glob.glob('../QC_matlab_final_reweighted/REA/'+\
                             template_name+'/????/??/*L.S??????')
        print 'I have read in '+str(len(detections))+' detections'
        if len(detections) > 3:
            wavdir='../QC_matlab_final_reweighted/WAV/'+template_name
            try:
                events=family_calc(template, detections, wavdir, corr_thresh=0.80,\
                            template_pre_pick=-0.05, cut=(0.5, 3.5), freqmin=2.0\
                                   ,freqmax=10.0, plotvar=True, resample=True,\
                                   samp_rate=400)
                if events:
                    all_events.append(events)
            except IOError:
                print 'No template magnitude'
            # except np.linalg.linalg.LinAlgError as err:
                # if 'Singular matrix' in err.message:
                    # print 'Singular Matrix, trying again'
                    # events=family_calc(template, detections, wavdir, corr_thresh=0.85,\
                            # template_pre_pick=-0.1, cut=(-0.5, 3.5), freqmin=2.0, \
                                       # freqmax=10.0, plotvar=False, resample=True)
                    # if events:
                        # all_events.append(events)
                # else:
                    # raise np.linalg.LinAlgError('Lin-alg error, but not a singular matrix')
        else:
            print 'Which is not enough!'
    f=open(templates[0].split('/')[-1]+'_SVD_magnitude.txt','w')
    # f=open('SVD_magnitude.txt','w')
    for event in all_events:
        for ev in event:
            print ev
            f.write(str(ev.time)+', '+str(ev.Mag_2)+'\n')
    f.close()
