#!/usr/bin/python
from __future__ import division
from future.utils import iteritems

import csv
import copy
import shutil
import numpy as np
import matplotlib.pyplot as plt

try:
    from plotFMC import circles
except:
    print('FMC files not on your path')
from glob import glob
from subprocess import Popen, PIPE
from shelly_focmecs import cluster_to_consensus
from obspy import read, Catalog
from scipy.signal import argrelmax, argrelmin
from scipy.stats import circmean
from obspy.imaging.beachball import beach
from eqcorrscan.utils import pre_processing
from eqcorrscan.utils.mag_calc import dist_calc
# Try to import hashpype if in active env
try:
    from hashpy.hashpype import HashPype, HashError
    from hashpy.plotting.focalmechplotter import FocalMechPlotter
except:
    print('HashPy not installed in this env, fool.')
# Try to import mtfit if in active env
try:
    from mtfit import mtfit
    from mtfit.utilities.file_io import parse_hyp
except:
    print('MTfit not installed in this env, fool')

def grab_day_wavs(wav_dirs, dto, stachans):
    # Helper to recursively crawl paths searching for waveforms from a dict of
    # stachans for one day
    import os
    import fnmatch
    from itertools import chain
    from obspy import read, Stream

    st = Stream()
    wav_files = []
    wav_ds = ['%s/%d' % (d, dto.year) for d in wav_dirs]
    for path, dirs, files in chain.from_iterable(os.walk(path)
                                                 for path in wav_ds):
        print('Looking in %s' % path)
        for sta, chans in iter(stachans.items()):
            for chan in chans:
                for filename in fnmatch.filter(files,
                                               '*.%s.*.%s*%d.%03d'
                                                       % (
                                               sta, chan, dto.year,
                                               dto.julday)):
                    wav_files.append(os.path.join(path, filename))
    print('Reading into memory')
    for wav in wav_files:
        st += read(wav)
    stachans = [(tr.stats.station, tr.stats.channel) for tr in st]
    for stachan in list(set(stachans)):
        tmp_st = st.select(station=stachan[0], channel=stachan[1])
        if len(tmp_st) > 1 and len(set([tr.stats.sampling_rate
                                        for tr in tmp_st])) > 1:
            print('Traces from %s.%s have differing samp rates'
                  % (stachan[0], stachan[1]))
            for tr in tmp_st:
                st.remove(tr)
            tmp_st.resample(sampling_rate=100.)
            st += tmp_st
    st.merge(fill_value='interpolate')
    print('Checking for trace length. Removing if too short')
    rm_trs = []
    for tr in st:
        if len(tr.data) < (86400 * tr.stats.sampling_rate * 0.8):
            rm_trs.append(tr)
        if tr.stats.starttime != dto:
            print('Trimming trace %s.%s with starttime %s to %s'
                  % (tr.stats.station, tr.stats.channel,
                     str(tr.stats.starttime), str(dto)))
            tr.trim(starttime=dto, endtime=dto + 86400,
                    nearest_sample=False)
    if len(rm_trs) != 0:
        print('Removing traces shorter than 0.8 * daylong')
        for tr in rm_trs:
            st.remove(tr)
    else:
        print('All traces long enough to proceed to dayproc')
    return st
########################## FMC PLOTTING ######################################

def arnold2FMC(input_files, outfile='test.png', show=True):
    """
    Wrapper on FMC to take control of the output and have access to the
    matplotlib Figure instance for custom plotting

    :param input_files: List of input psmeca files for plotting
    :param outfile: Optional output figure
    :return:
    """
    # Example using subprocess and grabbing stdout for use in plotting
    for input in input_files:
        cmd = '/home/chet/FMC_1.01/FMC.py -i AR -o K {}'.format(input)
        print(cmd)
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        out_list = stdout.decode('utf-8').split('\n')
        X_kaverina = [float(ln.split()[0]) for ln in out_list[2:-1]]
        Y_kaverina = [float(ln.split()[1]) for ln in out_list[2:-1]]
    # TODO Still not complete, more functionality and formatting needed
    # TODO Also should add time dependent capability...
    # TODO ...prefereably static but possibly video?
    fig = circles(X_kaverina, Y_kaverina, size=0.5, color='b',
                  plotname=outfile)
    if show:
        plt.show()
    return fig

########################## MTFIT STUFF #######################################

def run_mtfit(catalog, nlloc_dir, parallel=True, n=8, algorithm='iterate',
              phy_mem=1, inversion_options='PPolarity',
              number_location_samples=5000, MT=True, DC=True):
    """
    Wrapper on mtfit to run over a catalog for which there are already
    polarity picks and .scatangle nlloc files in the specified dir
    :param catalog: Catalog of events
    :param nlloc_dir: Directory with the necessary nlloc and .scatangle files
    :param parallel: Run in parallel?
    :param n: Number of cores
    :param algorithm: MTfit inversion algorithm
    :param phy_mem: A soft memory limit of 1Gb of RAM for estimating the
        sample sizes. This is only a soft limit, so no errors are thrown
        if the memory usage increases above this.
    :param inversion_options: What data to include in the inversion
    :param number_location_samples: How many random samples to draw from the
        NLLoc location PDF

    :return:
    """
    for ev in catalog:
        eid = str(ev.resource_id).split('/')[-1]
        print('Running mtfit for {}'.format(eid))
        nlloc_fs = glob('{}/{}*'.format(
            nlloc_dir,
            str(ev.resource_id).split('/')[-1].split('_')[0]))
        # Find the hyp file with update pol information
        print(nlloc_fs)
        try:
            hyp_path = [path for path in nlloc_fs
                        if path.endswith('.hyp')
                        and 'sum' not in path.split('.')
                        and path.split('_')[-1].startswith('pol')][0]
        except IndexError as msg:
            print('No NLLoc location for this event. Probably low SNR?')
            continue
        print(hyp_path)
        # Read in data dict
        data = parse_hyp(hyp_path)
        print(data)
        print(type(data))
        print(data['PPolarity'])
        data['UID'] = '{}_ppolarity'.format(eid)
        # Set the convert flag to convert the output to other source parameterisations
        convert = True
        # Set location uncertainty file path
        location_pdf_file_path = [path for path in nlloc_fs
                                  if path.endswith('.scatangle')][0]
        # Handle location uncertainty
        # Set number of location samples to use (randomly sampled from PDF) as this
        #    reduces calculation time
        # (each location sample is equivalent to running an additional event)
        bin_scatangle = True
        if DC:
            ### First run for DC contrained solution
            max_samples = 100000
            dc = True
            print('Running DC for {}'.format(eid))
            mtfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm,
                  parallel=parallel, inversion_options=inversion_options, phy_mem=phy_mem, dc=dc,
                  max_samples=max_samples, convert=convert, bin_scatangle=bin_scatangle,
                  number_location_samples=number_location_samples, n=n)
        if MT:
            ### Now for full MT
            # Change max_samples for MT inversion
            max_samples = 1000000
            dc = False
            print('Running full MT for {}'.format(eid))
            # Create the inversion object with the set parameters.
            mtfit(data, location_pdf_file_path=location_pdf_file_path, algorithm=algorithm,
                  parallel=parallel, inversion_options=inversion_options, phy_mem=phy_mem,
                  max_samples=max_samples, convert=convert, dc=dc,
                  bin_scatangle=bin_scatangle,
                  number_location_samples=number_location_samples, n=n)
    return

def plot_mtfit_output(directory, outdir):
    return

##############################################################################
################# Richard's focmec and stress inversion formatting ############

def format_arnold_to_gmt(arnold_file, catalog, outfile, names=False,
                         id_type='detection', date_range=[]):
    """
    Take *_sdr.dat output file from Arnold FM software
    add magnitudes, and output to psmeca format
    :param arnold_file: Output from arnold
    :param catalog: catalog including events in arnold file which need mags
    :param outfile: Name of output file to be used by psmeca
    :param names: Whether to include event names in psmeca file
    :param id_type: Whether catalog ids are in detection or template format
    :return:
    """
    # If len 0 catalog, warn and write empty file for gmt-plotting loop
    # conveneience
    if len(catalog) == 0:
        with open(outfile, 'w') as of:
            of.write('')
        print('Length 0 catalog: writing empty output file.')
        return
    if date_range:
        dates = date_range
    else:
        dats = [ev.picks[-1].time for ev in catalog]
        dates = [min(dats), max(dats)]
    if id_type == 'detection':
        # Dict keyed to detection id formatting from focmec package
        id_dict = {'{}.{}.{}'.format(
            ev.resource_id.id.split('/')[-1].split('_')[0],
            ev.resource_id.id.split('_')[-2],
            ev.resource_id.id.split('_')[-1][:6]): ev
            for ev in catalog
            if dates[0] < ev.picks[-1].time < dates[1]}
    with open(arnold_file, 'r') as f:
        next(f)
        with open(outfile, 'w') as of:
            for line in f:
                line = line.rstrip('\n')
                line = line.split(',')
                if id_type == 'detection':
                    try:
                        ev = [id_dict[line[0]]]
                    except KeyError:
                        ev = []
                elif id_type == 'template':
                    ev = [ev for ev in catalog
                          if str(ev.resource_id).split('/')[-1]
                          == line[0].split('.')[0]
                          if dates[0] < ev.picks[-1].time < dates[1]]
                if len(ev) > 0:
                    ev = ev[0]
                    if len(ev.magnitudes) == 0:
                        print('No mags for event: {}'.format(ev.resource_id))
                        print(ev)
                        continue
                    o = ev.preferred_origin()
                    if names:
                        name = str(ev.resource_id).split('/')[-1]
                    else:
                        name = ''
                    of.write('{} {} {} {} {} {} {} 0 0 {}\n'.format(
                        o.longitude, o.latitude, o.depth / 1000., line[1],
                        line[2], line[3], ev.preferred_magnitude().mag, name))
    return

def arnold_focmec_2_clust(sdr_err_file, clust_dict, outdir, window=None):
    """
    Function to break output file from arnold focmec into clusters

    :param sdr_err_file: Output from afmec (projname_scalar_err_degrees.csv)
    :param clust_dict: Dict with clust name as keys, lists of ev name as value
    :param outdir: Directory to put the separated files into
    :param time_dict (optional): Dict with the size of window and overlap
    :return:
    """
    for clust_name, big_ev_list in clust_dict.items():
        print('Doing cluster: {}'.format(clust_name))
        with open(sdr_err_file, 'r') as f:
            clust_ev_list = [line
                             for line in f if line.split(',')[0].split('.')[0]
                             in big_ev_list]
        print(len(clust_ev_list))
        if window:
            sub_clusts = [clust_ev_list[i:i+window]
                          for i in range(0, len(clust_ev_list) - window)]
        else:
            sub_clusts = [clust_ev_list]
        for i, ev_list in enumerate(sub_clusts):
            new_fname = '{}_{}.csv'.format(clust_name, i)
            with open('{}/{}'.format(outdir, new_fname), 'w') as of:
                for line in ev_list:
                    ln = line.rstrip('\n').split(',')
                    of.write('{},{},{},{}\n'.format(ln[1], ln[2],
                                                    ln[3], ln[-1]))
    return

##############################################################################

def write_obspy_focmec_2_gmt(catalog, outfile, names=False, strike_range=45,
                             format='Aki'):
    """
    Take a catalog which contains focal mechanisms and write a file formatted
    for input into GMT.

    :param catalog: Catalog of events
    :param outfile: Name of the output file
    :param format: GMT format to write to. Just 'Aki' for now.
    :return:
    """
    with open(outfile, 'w') as f:
        for ev in catalog:
            eid = str(ev.resource_id).split('/')[-1]
            # The origin should probably be the NLLoc manual one
            orig = ev.origins[-1]
            mag = ev.preferred_magnitude()
            # Here will have to determine preferred FM
            fms = [(fm.nodal_planes.nodal_plane_1.strike,
                    fm.nodal_planes.nodal_plane_1.dip,
                    fm.nodal_planes.nodal_plane_1.rake)
                   for fm in ev.focal_mechanisms]
            fms = list(zip(*fms))
            # Determine the difference between the min and max
            angle = min(fms[0]) - max(fms[0])
            diff = abs((angle + 180) % 360 - 180)
            print('Angle difference: {}'.format(diff))
            if diff <= strike_range:
                avg_fm = [circmean(fms[0], high=360), np.mean(fms[1]),
                          np.mean(fms[2])]
                if names:
                    name=eid
                else:
                    name = ''
                f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(
                    orig.longitude, orig.latitude, orig.depth / 1000., avg_fm[0],
                    avg_fm[1], avg_fm[2], mag.mag, 0, 0, name))
            else:
                print(
                    'Range of strikes for {} too large. Skipping.'.format(eid))
    return

def add_pols_to_hyp(catalog, nlloc_dir, outdir, ev_type='temp'):
    """
    Add polarities to the nlloc hyp files to be used by Arnold focmec stuff
    :param catalog:
    :param nlloc_dir:
    :return:
    """
    for ev in catalog:
        print('{}'.format(str(ev.resource_id).split('/')[-1]))
        if ev_type == 'temp':
            nlloc_fs = glob('{}/{}*'.format(
                nlloc_dir,
                str(ev.resource_id).split('/')[-1].split('_')[0]))
        elif ev_type == 'det':
            nlloc_fs = glob('{}/{}*'.format(
                nlloc_dir,
                str(ev.resource_id).split('/')[-1]))
        try:
            hyp_path = [path for path in nlloc_fs
                        if path.endswith('.hyp')
                        and 'sum' not in path.split('.')][0]
        except IndexError as msg:
            print('No NLLoc location for this event. Probably low SNR?')
            continue
        print(hyp_path)
        # Move hdr and scat files to outdir
        scat_hdr = [path for path in nlloc_fs
                    if (path.endswith('.hdr')
                        or path.endswith('.scat'))
                    and 'sum' not in path.split('.')]
        for fl in scat_hdr:
            shutil.copyfile(fl, '{}/{}'.format(outdir, fl.split('/')[-1]))
        with open(hyp_path, 'r') as orig:
            with open('{}/{}'.format(outdir,
                                     hyp_path.split('/')[-1]), 'w') as new:
                phase = False
                for ln in orig:
                    line = ln.rstrip()
                    line = line.split()
                    if len(line) == 0:
                        print('End of file')
                        break
                    if line[0] == 'PHASE':
                        phase = True
                        new.write(' '.join(line) + '\n')
                        continue
                    elif line[0] == 'END_PHASE':
                        phase = False
                    if phase:
                        print('Try adding for {}'.format(line[0]))
                        try:
                            pk = [pk for pk in ev.picks
                                  if pk.waveform_id.station_code == line[0]
                                  and line[2] == 'Z'][0]
                        except IndexError:
                            print('No pick for this.....pick??')
                            continue
                        if pk.polarity not in ['positive', 'negative']:
                            print('No polarity for station {}'.format(line[0]))
                            continue
                        if pk.polarity == 'positive':
                            line[5] = 'U'
                        elif pk.polarity == 'negative':
                            line[5] = 'D'
                    new.write(' '.join(line) + '\n')
    return


def foc_mec_from_event(catalog, station_names=False, picks_only=False,
                       outdir=False):
    """
    Just taking Tobias' plotting function out of obspyck
    :param catalog: Catalog of events with focmec info
    :param station_names: Whether or not to plot the station names
    :param wavdir: False or the root of the waveform directory
    :param outdir: False or the directory to write the files to
    :return:
    """

    for ev in catalog:
        eid = str(ev.resource_id).split('/')[-1]
        print('Plotting eid: {}'.format(eid))
        fms = ev.focal_mechanisms
        if not fms:
            err = "Error: No focal mechanism data!"
            print(err)
            continue
        # make up the figure:
        fig, tax = plt.subplots(figsize=(7, 7))
        ax = fig.add_subplot(111, aspect="equal")
        ax.autoscale_view(tight=False, scalex=True, scaley=True)
        width = 2
        plot_width = width * 0.95
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        # plot the selected solution
        av_np1_strike = np.mean([fm.nodal_planes.nodal_plane_1.strike
                                 for fm in fms])
        print('Strike of nodal plane 1: %f' % av_np1_strike)
        fm = sorted([fm for fm in fms], key=lambda x:
                    abs(x.nodal_planes.nodal_plane_1.strike - av_np1_strike))[0]
        np1 = fm.nodal_planes.nodal_plane_1
        if hasattr(fm, "_beachball"):
            beach_ = copy.copy(fm._beachball)
        else:
            beach_ = beach([np1.strike, np1.dip, np1.rake],
                           width=plot_width)
            fm._beachball = copy.copy(beach_)
        ax.add_collection(beach_)
        # plot the alternative solutions
        if not hasattr(fm, "_beachball2"):
            for fm_ in fms:
                _np1 = fm_.nodal_planes.nodal_plane_1
                beach_ = beach([_np1.strike, _np1.dip, _np1.rake],
                               nofill=True, edgecolor='k', linewidth=1.,
                               alpha=0.3, width=plot_width)
                fm_._beachball2 = copy.copy(beach_)
        for fm_ in fms:
            ax.add_collection(fm_._beachball2)
        text = "Focal Mechanism (%i of %i)" % \
               (0 + 1, len(fms))
        text += "\nStrike: %6.2f  Dip: %6.2f  Rake: %6.2f" % \
                (np1.strike, np1.dip, np1.rake)
        if fm.misfit:
            text += "\nMisfit: %.2f" % fm.misfit
        if fm.station_polarity_count:
            text += "\nStation Polarity Count: %i" % fm.station_polarity_count
        # fig.canvas.set_window_title("Focal Mechanism (%i of %i)" % \
        #        (self.focMechCurrent + 1, len(fms)))
        fig.subplots_adjust(top=0.88)  # make room for suptitle
        # values 0.02 and 0.96 fit best over the outer edges of beachball
        # ax = fig.add_axes([0.00, 0.02, 1.00, 0.96], polar=True)
        ax.set_ylim(-1, 1)
        ax.set_xlim(-1, 1)
        ax.axison = False
        axFocMecStations = fig.add_axes([0.00, 0.02, 1.00, 0.84], polar=True)
        ax = axFocMecStations
        # ax.set_title(text)
        ax.set_axis_off()
        azims = []
        incis = []
        polarities = []
        bbox = dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=1.5,
                    alpha=0.7)
        for pick in ev.picks:
            if pick.phase_hint != "P":
                continue
            wid = pick.waveform_id
            net = wid.network_code
            sta = wid.station_code
            # Something screwy in this not plotting
            arrival = getArrivalForPick(ev.origins[-2].arrivals, pick)
            if not pick:
                continue
            if pick.polarity is None or arrival is None or \
                arrival.azimuth is None or arrival.takeoff_angle is None:
                continue
            if pick.polarity == "positive":
                polarity = True
            elif pick.polarity == "negative":
                polarity = False
            else:
                polarity = None
            azim = arrival.azimuth
            inci = arrival.takeoff_angle
            # lower hemisphere projection
            if inci > 90:
                inci = 180. - inci
                azim = -180. + azim
            # we have to hack the azimuth because of the polar plot
            # axes orientation
            plotazim = (np.pi / 2.) - ((azim / 180.) * np.pi)
            azims.append(plotazim)
            incis.append(inci)
            polarities.append(polarity)
            if station_names:
                ax.text(plotazim, inci, "  " + sta, va="top", bbox=bbox, zorder=2)
        azims = np.array(azims)
        incis = np.array(incis)
        polarities = np.array(polarities, dtype=bool)
        ax.scatter(azims, incis, marker="o", lw=2, facecolor="w",
                   edgecolor="k", s=200, zorder=3)
        mask = (polarities == True)
        ax.scatter(azims[mask], incis[mask], marker="+", lw=3, color="k",
                   s=200, zorder=4)
        mask = ~mask
        ax.scatter(azims[mask], incis[mask], marker="_", lw=3, color="k",
                   s=200, zorder=4)
        # this fits the 90 degree incident value to the beachball edge best
        ax.set_ylim([0., 91])
        if not outdir:
            plt.draw()
            plt.show()
            plt.close('all')
        else:
            fig.savefig('{}/{}_focmec.png'.format(outdir, eid),
                        dpi=500)
            plt.close('all')
    return


def getArrivalForPick(arrivals, pick):
    """
    searches given arrivals for an arrival that references the given
    pick and returns it (empty Arrival object otherwise).
    """
    for a in arrivals:
        if a.pick_id == pick.resource_id:
            return a
    return None

def dec_2_merc_meters(dec_x, dec_y, z):
    """
    Conversion from decimal degrees to meters in Mercury Cartesian Grid.
    This is the same grid used for NLLoc

    **Note** This grid node has been changed from the NLLoc grid origin of
    -38.3724, 175.9577 to ensure that all coordinates are positive
    """
    origin = [-38.8224, 175.9577]
    y = (dec_y - origin[0]) * 111111
    x = (dec_x - origin[1]) * (111111 * np.cos(origin[0]*(np.pi/180)))
    return x, y, z

############################### HASHPY STUFF #################################


def clusts_to_hashpy(clust_cats, config, outdir):
    """
    Loop over cluster catalogs and compute consensus mechanisms for each
    :return:
    """
    for i, clust in enumerate(clust_cats):
        cons, stach_dict, arr_dict = cluster_to_consensus(clust)
        soln_cat = run_hashpy(cons, config,
                              outfile='{}/Cat_consensus_{}.xml'.format(
                                  outdir, i))
    return


def run_hashpy(catalog, config, outfile, mode=None):
    """
    Wrapper on hashpy for calculating HASH focal mechanisms
    :param catalog: :class: obspy.core.event.Catalog
    :param config: Configuration dict for hashpy
    :return:
    """
    new_cat = Catalog()
    hp = HashPype(**config)
    hp.load_velocity_models()
    if mode == 'consensus':
        hp.input(catalog, format="OBSPY_CONSENSUS")
        hp.generate_trial_data()
        try:
            hp.calculate_takeoff_angles()
        except:
            print('Error in toa calc for consensus')
        pass1 = hp.check_minimum_polarity()
        pass2 = hp.check_maximum_gap()
        if pass1 and pass2:
            try:
                hp.calculate_hash_focalmech()
                hp.calculate_quality()
            except:
                print('Error in fm calc for consensus')
        else:
            print("Minimum polarity and/or maximum gap check failed")
        new_cat += hp.output(format="OBSPY")
    elif mode == 'composite':
        hp.input(catalog, format='OBSPY_COMPOSITE')
        print(hp.p_azi_mc)
        print(hp.p_the_mc)
        print(hp.magap)
        print(hp.mpgap)
        # pass1 = hp.check_minimum_polarity()
        # pass2 = hp.check_maximum_gap()
        # pass1 = True, pass2 = True
        # if pass1 and pass2:
        try:
            hp.calculate_hash_focalmech()
            hp.calculate_quality()
        except:
            print('Error in fm calc for composite')
        # else:
        #     print('Minimum polarity and/or maximum gap check failed')
        new_cat += hp.output(format="OBSPY")
    else:
        for ev in catalog:
            eid = str(ev.resource_id).split('/')[-1]
            # Set up hashpy object
            hp.input(ev, format="OBSPY")
            hp.generate_trial_data()
            try:
                hp.calculate_takeoff_angles()
            except:
                print('Error in toa calc for eid: {}'.format(eid))
                continue
            pass1 = hp.check_minimum_polarity()
            pass2 = hp.check_maximum_gap()
            if pass1 and pass2:
                try:
                    hp.calculate_hash_focalmech()
                    hp.calculate_quality()
                except:
                    print('Error in fm calc for eid: {}'.format(eid))
                    continue
            else:
                print("Minimum polarity and/or maximum gap check failed")
                continue
            new_cat += hp.output(format="OBSPY")
    new_cat.write(outfile, format="QUAKEML")
    return new_cat

def plot_hashpy(catalog, outdir):
    """
    Take a catalog of events with foc mecs defined by hashpy and save the
    plots to a file
    :param catalog:
    :param outdir:
    :return:
    """
    for ev in catalog:
        eid = str(ev.resource_id).split('/')[-1]
        fmp = FocalMechPlotter(ev)
        fmp.fig.savefig('{}/{}'.format(outdir, eid), dpi=500)
    return

##############################################################################

def plot_network_arrivals(wav_dirs, lowcut, highcut, start, end, sta_list=None,
                          remove_resp=False, inv=None, dto=None, ev=None):
    """
    Plot data for the whole network at a given dto

    This is intended for plotting teleseismic arrivals to check polarities
    :param dto:
    :param wav_dirs:
    :return:
    """
    if not sta_list:
        sta_list = ['ALRZ','ARAZ','HRRZ','NS01','NS02','NS03','NS04','NS05','NS06',
                    'NS07','NS08','NS09','NS10','NS11','NS12','NS13','NS14','NS15',
                    'NS16','NS18','PRRZ','RT01','RT02','RT03','RT05','RT06','RT07',
                    'RT08','RT09','RT10','RT11','RT12','RT13','RT14','RT15','RT16',
                    'RT17','RT18','RT19','RT20','RT21','RT22','RT23','THQ2','WPRZ']
    stachans = {sta: ['EHZ'] for sta in sta_list}
    if ev:
        dto = ev.origins[-1].time
    # Get start of day
    dto_start = copy.deepcopy(dto)
    dto_start.hour = 0
    dto_start.minute = 0
    dto_start.second = 0
    dto_start.microsecond = 0
    st = grab_day_wavs(wav_dirs, dto_start, stachans)
    pf_dict = {'MERC': [0.001, 1.0, 35., 45.],
               'WPRZ': [0.001, 0.5, 35., 45.],
               'GEONET': [0.001, 0.01, 40., 48.]}
    st.traces.sort(key=lambda x:
                   inv.select(station=x.stats.station)[0][0].latitude)
    st1 = pre_processing.dayproc(st, lowcut, highcut, 3, 100., starttime=dto,
                                 num_cores=4)
    trimmed = st1.trim(starttime=dto+start, endtime=dto+end)
    for tr in trimmed:
        sta = tr.stats.station
        if sta.endswith('Z'):
            if sta == 'WPRZ':
                prefilt = pf_dict['WPRZ']
            else:
                prefilt = pf_dict['GEONET']
        else:
            prefilt = pf_dict['MERC']
        if remove_resp:
            # Cosine taper and demeaning applied by default
            tr.remove_response(inventory=inv, pre_filt=prefilt, output='DISP')
    labels=[]
    for tr in trimmed:
        labels.append(tr.stats.station)
        tr.data = tr.data / max(tr.data)
    fig, ax = plt.subplots(figsize=(3, 6))
    vert_steps = np.linspace(0, len(trimmed), len(trimmed))
    for tr, vert_step in zip(trimmed, vert_steps):
        ax.plot(tr.data + vert_step, color='k', linewidth=0.3)
    ax.yaxis.set_ticks(vert_steps)
    ax.set_yticklabels(labels, fontsize=8)
    plt.show()
    return

######## HYBRIDMT STUFF ##########

def write_hybridMT_input(cat, sac_dir, inv, self_files, outfile,
                         prepick, postpick, file_type='raw', plot=False):
    """
    Umbrella function to handle writing input files for focimt and hybridMT

    :param cat: Catalog of events to write files for
    :param sac_dir: Root directory for detection SAC files
    :param inv: Inventory object containing all necessary station responses
    :param selfs: List containing directory names for template self detections
    :param prefilt: List of 4 corners for preconvolution bandpass
        For details see obspy.core.trace.Trace.remove_response() docs
    :return:
    """
    selfs = []
    for self_file in self_files:
        with open(self_file, 'r') as f:
            rdr = csv.reader(f)
            for row in rdr:
                selfs.append(str(row[0]))
    ev_dict = {}
    # Build prefilt dict (Assuming all wavs have been downsampled to 100 Hz)
    pf_dict = {'MERC': [0.001, 1.0, 35., 45.],
               'WPRZ': [0.001, 0.5, 35., 45.],
               'GEONET': [0.001, 0.01, 40., 48.]}
    # Loop through events
    for ev in cat:
        ev_id = str(ev.resource_id).split('/')[-1]
        print('Working on {}'.format(ev_id))
        self = [self for self in selfs if self.split('_')[0] == ev_id]
        orig = ev.origins[-1]
        if len(self) == 0: # Skip those with no self detection (outside fields)
            print('No self detection for %s' % ev_id)
            continue
        wavs = glob('%s/%s/*' % (sac_dir, self[0]))
        # Loop through arrivals and populate ev_dict with TOA, Backaz, etc...
        ev_dict[ev_id] = {} # Allocate subdictionary
        ev_dict[ev_id]['phases'] = []
        ev_dict[ev_id]['header'] = None
        for arr in orig.arrivals:
            pick = arr.pick_id.get_referred_object()
            sta = pick.waveform_id.station_code
            chan = pick.waveform_id.channel_code
            print('{}.{}'.format(sta, chan))
            if chan[-1] != 'Z':
                continue
            sta_inv = inv.select(station=sta, channel=chan)
            # Do a rough incidence angle calculation based on dist and depth
            dist = dist_calc((orig.latitude, orig.longitude,
                              orig.depth / 1000.), (sta_inv[0][0].latitude,
                                                    sta_inv[0][0].longitude,
                                                    (sta_inv[0][0].elevation -
                                                     sta_inv[0][0][0].depth)
                                                     / 1000.))
            aoi = 90. - np.degrees(np.arcsin(orig.depth / 1000. / dist))
            if np.isnan(aoi):
                aoi = 180. - arr.takeoff_angle
            # Establish which station we're working with
            if sta.endswith('Z'):
                if sta == 'WPRZ':
                    prefilt = pf_dict['WPRZ']
                else:
                    prefilt = pf_dict['GEONET']
            else:
                prefilt = pf_dict['MERC']
            wav_file = [wav for wav in wavs
                        if wav.split('_')[-1].split('.')[0] == chan
                        and wav.split('_')[-2] == sta]
            if len(wav_file) == 0:
                print('Waveform directory not found.')
                continue
            # Read in the corresponding trace
            # Cosine taper and demeaning applied by default
            raw = read(wav_file[0])[0]
            tr = read(wav_file[0])[0].remove_response(inventory=inv,
                                                      pre_filt=prefilt,
                                                      output='DISP')
            # Invert polarity of SP instruments
            if not sta.endswith('Z'):
                tr.data *= -1
            # Trim around P pulse
            raw_sliced = raw.slice(starttime=pick.time - 0.2,
                                   endtime=pick.time + 1).copy()
            whole_tr = tr.slice(starttime=pick.time - 0.2,
                                endtime=pick.time + 1).copy()
            tr.trim(starttime=pick.time - prepick,
                    endtime=pick.time + postpick)
            pick_sample = int(prepick * tr.stats.sampling_rate)
            # Find the next index where trace crosses the 'zero' value
            # which we assume is the value at time of pick.
            # Take last 'zero' crossing of the trimmed wav, assuming we've
            # trimmed only half a cycle. Then integrate from pick time to
            # first sample with a swapped sign (+/- or -/+)
            # Make pick value zero
            leveled = tr.data - tr.data[pick_sample]
            # Determine some polarity info
            rel_min_max = argrelmax(np.abs(leveled)) #Relative peaks
            print(rel_min_max)
            try:
                if rel_min_max[0].shape[0] > 1:
                    print(leveled[rel_min_max])
                    rel_pk = np.argmax(np.abs(leveled[rel_min_max]))
                    print(rel_pk)
                    print(rel_min_max[0][rel_pk])
                    peak = leveled[rel_min_max[0][rel_pk]] # Largest peak
                else:
                    try:
                        peak = leveled[rel_min_max][0]
                    except IndexError:
                        print('No relative maxima or minima')
                        continue
            except ValueError:
                print('No relative maxima or minima')
                continue
            print('Peak value: {!s}'.format(peak))
            polarity = np.sign(peak) # Sign of largest peak
            print('Zero crossings at: {!s}'.format(
                np.where(np.diff(np.sign(leveled[pick_sample + 1:])) != 0)[0]))
            try:
                pulse = leveled[pick_sample:pick_sample + 1 + np.where(
                    np.diff(np.sign(leveled[pick_sample + 1:]))
                    != 0)[0][-1] + 2] # 2-sample fudge factor over crossing
            except IndexError as i:
                print('IndexError: {}'.format(i))
                if polarity == 1:
                    try:
                        pulse = leveled[pick_sample:argrelmin(leveled)[0][-1] + 1]
                    except IndexError:
                        print('No zero crossing OR relative minimum.')
                        continue
                elif polarity == -1:
                    try:
                        pulse = leveled[pick_sample:argrelmax(leveled)[0][-1] + 1]
                    except IndexError:
                        print('No zero crossing OR relative maximum.')
                        continue
            # Try to catch case where small min/max just post pick
            if len(pulse) < 6:
                print('{}'.format(
                      'Pulse is too short: likely due to small rel peak'))
                pulse = leveled[pick_sample:]
            omega = np.trapz(pulse)
            if plot:
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
                fig.suptitle('{}.{}'.format(sta, chan))
                ax1.plot(raw_sliced.data, label='raw')
                ax2.plot(whole_tr.data, label='Displacement')
                ax3.plot(leveled, color='k', label='Pulse')
                ax3.plot(np.arange(pick_sample, pick_sample + len(pulse),
                                   step=1),
                         pulse, color='r')
                ax3.axvline(pick_sample, linestyle='--',
                            color='grey', label='Pick')
                plt.legend()
                plt.show()
                plt.close()
            # Now we can populate the strings in ev_dict
            if file_type == 'raw':
                ev_dict[ev_id]['phases'].append(
                    "  {} {} {} {!s} {!s} {!s} {!s} {!s} {!s} {!s}\n".format(
                        sta, chan[-1], pick.phase_hint, omega * polarity,
                        arr.azimuth, aoi, arr.takeoff_angle, 5000, dist * 1000,
                        2600))
            elif file_type == 'vel1d':
                x, y, z = dec_2_merc_meters(sta_inv[0][0].longitude,
                                            sta_inv[0][0].latitude,
                                            sta_inv[0][0].elevation -
                                            sta_inv[0][0][0].depth)
                ev_dict[ev_id]['phases'].append(
                    "  {} {} {} {!s} {!s} {!s} {!s}\n".format(
                        sta, chan[-1], pick.phase_hint, omega * polarity,
                        y, x, z))
        if len(ev_dict[ev_id]['phases']) > 0:
            if file_type == 'raw':
                ev_dict[ev_id]['header'] = "{} {!s}\n".format(
                    ev_id, len(ev_dict[ev_id]['phases']))
            elif file_type == 'vel1d':
                ex, ey, ez = dec_2_merc_meters(orig.longitude, orig.latitude,
                                               -1 * orig.depth)
                ev_dict[ev_id]['header'] = "{} {!s} {!s} {!s} {!s} {!s}\n".format(
                    ev_id, len(ev_dict[ev_id]['phases']), ey, ex, ez, 2600)
    with open(outfile, 'w') as fo:
        for eid, dict in iteritems(ev_dict):
            if dict['header'] is not None:
                print('Writing event %s' % eid)
                fo.write(dict['header'])
                fo.writelines(dict['phases'])
    return