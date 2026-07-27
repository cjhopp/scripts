#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 18:12:11 2024

@author: hilaryc
"""
import numpy as np
from scipy import signal
import sys, obspy, os
import matplotlib.pyplot as plt
import pandas as pd
from obspy.core import UTCDateTime
from obspy.core import Stream,Trace
from obspy.geodetics import gps2dist_azimuth as gps2DistAzimuth # depends on obspy version; this is for v1.1.0
from datetime import datetime, timedelta
from obspy.core.util.attribdict import AttribDict
import scipy.fftpack as sfft
from scipy.signal import hilbert,convolve,detrend, butter, lfilter,  correlate, deconvolve
from scipy.signal.windows import hann
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D
import scipy
from obspy.geodetics.base import calc_vincenty_inverse,degrees2kilometers,gps2dist_azimuth
from obspy.signal.filter import bandpass
sys.path.append('/home/hilarych/packages/') # For engaging, fairweather
sys.path.append('/Users/hilaryc/Research/') # For local
# import seisproc as sep
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import lmfit
import pykonal
sampleFrequency=1000.
gaugeLength=10.
conversionFactor    = 116.0 * 2**-13 * sampleFrequency / gaugeLength

def get_tt_raypath_list(Cgrid,
                   dx,xaxis,zaxis,evdp,rec_izs, return_tt_matrix=False):
    '''
    Given a 2D grid (Cgrid) filled with velocity, calculate the eikonal travel time using Pykonal.
    Assuming source is at [0,z at evdp,0]
    evdp: EQ depth.
    Cgrid: A 2D grid with 1D velocity profile.
    dx: Grid spacing.
    xaxis: horizontal axis of the grid
    zaxis: vertical axis of the grid
    rec_izs: Locations of the receiver. Assuming receivers are aligned in vertical direction. These are the indices of along the z direction.
    See the documentation of pykonal for more customations.    
    '''
    # Instantiate EikonalSolver object using Cartesian coordinates.
    solver = pykonal.EikonalSolver(coord_sys="cartesian")
    solver.velocity.min_coords = 0, 0, 0
    solver.velocity.node_intervals = dx, dx, dx
    solver.velocity.npts = len(xaxis), len(zaxis), 1
    Ct=Cgrid.T
    Ct1=Ct[:,:,np.newaxis]
    solver.velocity.values = Ct1#np.ones(solver.velocity.npts)
    assert np.min(abs(zaxis-evdp))< dx, "The depth of the grid probably does not reach source depth. Make the grid deeper."
    idep_src=np.nanargmin(abs(zaxis-evdp))
    src_idx = 0, idep_src, 0
    solver.traveltime.values[src_idx] = 0
    solver.unknown[src_idx] = False
    solver.trial.push(*src_idx)
    solver.solve()
    # print(solver.traveltime.values)
    tt_matrix=solver.traveltime.values
    
    rec_ix=-1
    rec_iy= 0
    # Initialize
    tt_list=np.full(len(rec_izs),np.nan)
    ray_path_list=[]
    rec_loc_list=[]
    for i,rec_iz in enumerate(rec_izs):
        if ~np.isnan(rec_iz):
            rec_idx=rec_ix,int(rec_iz),rec_iy
            # print(rec_idx)
            rec_loc=xaxis[rec_idx[0]],zaxis[rec_idx[1]],rec_iy
            tt_rec=tt_matrix[rec_idx]
            # print(rec_loc)
            ray_path=solver.trace_ray(np.array(rec_loc))
            tt_list[i]=tt_rec
            ray_path_list.append(ray_path)
            rec_loc_list.append(rec_loc)
        else:
            ray_path_list.append(np.nan)
            rec_loc_list.append(np.nan)
    if return_tt_matrix:
        return tt_list,ray_path_list,rec_loc_list, tt_matrix
    else:
        return tt_list,ray_path_list,rec_loc_list, np.nan
    
def get_att_along_path_list(Qgrid_z,tt_matrix,ray_path_list,xaxis,zaxis):
    '''
    Given the Q profile (Qgrid_z), tt_matix, and raypath, calculate the accumulated attenuation
    Qgrid: 1D Q profile
    tt_list,ray_path_list: from get_tt_raypath_list
    xaxis: horizontal axis of the grid
    zaxis: vertical axis of the grid
        
    '''
    # Initialize
    att_list=np.full(len(ray_path_list),np.nan)
    for ir,ray_path in enumerate(ray_path_list):
        if type(ray_path)!=float: # ==np.nan
            # Initialize
            q_list=[] # Q along the ray path
            iz_list=[]
            ix_list=[]
            tt_pt_list=[]
    
            for i,rp in enumerate(ray_path):
                (x,z,_)=rp
                iz=np.nanargmin(abs(zaxis-z))
                ix=np.nanargmin(abs(xaxis-x))
                q=Qgrid_z[iz]
                tt_pt=tt_matrix[ix,iz,0]
                q_list.append(q)
                ix_list.append(ix)
                iz_list.append(iz)
                tt_pt_list.append(tt_pt)
               
            dif_tt=np.diff(tt_pt_list)
            att=0 # Initialize
            for i in range(len(dif_tt)):
                if ~np.isnan(q_list[i]):
                    att+= dif_tt[i]/q_list[i+1]
            att_list[ir]=att
        else:
            att_list[ir]=np.nan
    att_list[att_list==0]=np.nan
    return att_list
    # return np.array(i_enter_layers),tt_in_layers

def angle_btw_vectors(vector1, vector2):
    """
    Calculate the angle between two 3D vectors.

    Args:
        vector1 (list or numpy.array): The first 3D vector.
        vector2 (list or numpy.array): The second 3D vector.

    Returns:
        float: The angle in radians.
    """
    # Ensure the input vectors are numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)
    magnitude_vector1 = np.linalg.norm(vector1)
    magnitude_vector2 = np.linalg.norm(vector2)

    # Check for division by zero
    if magnitude_vector1 == 0 or magnitude_vector2 == 0:
        raise ValueError("Both input vectors must have a non-zero magnitude.")

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_vector1 * magnitude_vector2)

    # Calculate the angle using the arccosine function
    angle_rad = np.arccos(cos_angle)

    return angle_rad

def rawDAS2acc_factor_via_v(v, incident_rad,phase):
    '''
    Channel-dependent, frequency-independent(assuming same velocity for all frequency) factor.
    v is the medium velocity at the channel location.
    '''
    return v/np.cos(incident_rad)/particle_motion_sensitivity(incident_rad,phase)*conversionFactor  


def rawDAS2acc_factor_via_vapp(v_app, incident_rad,phase):
    '''
    Channel-dependent, frequency-independent(assuming same velocity for all frequency) factor.
    v_app is the apparent velocity of the target wave perceived by the cable.
    '''
    return v_app/particle_motion_sensitivity(incident_rad,phase)*conversionFactor  

def particle_motion_sensitivity(incident_rad,phase):
    # if phase=='P':
        # res=(np.cos(incident_rad))**2
    # elif phase =='S':
        # res=(np.cos(incident_rad)*np.sin(incident_rad))
    # else:
        # raise "P or S??"
    if phase=='P':
        res=np.cos(incident_rad)
    elif phase =='S':
        res=np.sin(incident_rad)
    else:
        raise "P or S??"
    return res

def calculate_azimuth(vector):
    """
    Calculate the azimuth of a 2D or 3D vector.

    Args:
        vector (list or tuple or numpy array): A 2D or 3D vector.

    Returns:
        float: The azimuth of the vector in radians.

    Raises:
        ValueError: If the input vector is not 2D or 3D.
    """
    # Convert the vector to a NumPy array
    vector = np.array(vector)
    # Check if the vector is 2D or 3D
    if len(vector) not in [2, 3]:
        raise ValueError("Input vector must be 2D or 3D")
    # If the vector is 3D, project it onto the xy-plane
    if len(vector) == 3:
        vector = vector[:2]
    # Calculate the azimuth
    azimuth = np.arctan2(vector[0], vector[1])
    # Convert the azimuth to a positive angle in [0, 2π)
    if azimuth < 0:
        azimuth += 2 * np.pi
    return azimuth

def att_term(tt,
             Q, f):
    kappa=tt/Q
    return np.exp(-kappa*np.pi*f)


def residual_att_term(params, f, data, tt,
                      # dx, v, 
                      fit_frange):
    Q = params['Q']
    Omega0 = params['Omega0']
    
    ind=np.where( (f>=fit_frange[0])&(f<=fit_frange[1]) )[0]
    model = Omega0*att_term(tt=tt,  #dx=dx,v=v, 
                            Q=Q, f=f[ind])
    return np.log10(data[ind])-np.log10(model)

def fit_att_term(x,y,tt,#dx,v,  
                 init_Q,init_Omega0,fit_frange,
            Q_range,
            varying_Omega0,
            # Omega0_range,
            method='least_sqaures',
            nan_policy='propagate'):
    import lmfit
    isNan=np.isnan(y)
    if all(isNan):
        return np.nan
    
    else:
        params=lmfit.Parameters()
        params.add('Q',value=init_Q,min=Q_range[0],max=Q_range[1])
        # if np.diff(Omega0_range)[0]==0:
        if varying_Omega0:
            # init_Omega0=np.nanmean(y[np.where(x<low_f_lv_for_Omega0)[0]])
            # print(np.nanmin(y))
            # print(np.nanmax(y))
            params.add('Omega0',value=init_Omega0,min=np.nanmin(y),max=np.nanmax(y)*10)
        else:
            params.add('Omega0',value=init_Omega0,vary=False)
        mini = lmfit.Minimizer(residual_att_term, params, nan_policy=nan_policy,
                                fcn_kws=dict(f=x[~isNan], \
                                            data=y[~isNan],
                                            tt=tt,
                                            # dx=dx, v=v,
                                            fit_frange=fit_frange   \
                                            )
                                )
        result = mini.minimize(method=method)
        return result#,mini
def gauge_length_effect(L,k_app):
    return np.abs(np.sin(np.pi*k_app*L)/np.pi/k_app)/L
def gauge_length_effect_spectra(f,app_vp,L):
    # theta=inc_rad_chs[i]
    # v=vp_profile_well[i]
    # k_app=f/v*np.cos(theta)
    
    # v_app=app_vp_list[i]
    k_app=f/app_vp
    res=gauge_length_effect(L,k_app)
    return res
def moving_average_smoothing(series, window_size):
    """
    Smooth the input series using a moving average with the specified window size.

    Parameters:
    series (list or np.ndarray): The input series to be smoothed.
    window_size (int): The size of the moving window.

    Returns:
    np.ndarray: The smoothed series.
    """
    if not isinstance(series, (list, np.ndarray)):
        raise ValueError("Input series must be a list or numpy array.")
    
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    
    series = np.array(series)
    smoothed_series = np.zeros_like(series, dtype=float)
    
    for i in range(len(series)):
        start = max(0, i - window_size // 2)
        end = min(len(series), i + window_size // 2 + 1)
        smoothed_series[i] = np.nanmean(series[start:end])
    
    return smoothed_series

def valid_f_range(signal, noise, snr_limit):
    '''
    Given the signal and noise spectrum (precalculated from data of the same length) and
    the desired minimum SNR, output the valid spectrum length and range (in indices) above this SNR threshold.
    
    '''
    ev_snr = signal/noise



    # Only keep the data that is above the SNR limit; pad 0s otherwise
    sn_range= (ev_snr>=snr_limit)
    #np.where(ev_snr>=snr_limit, ev_snr, 0)
    
    # Add additional 1s in the beginning for calculating diff later
    for i in range(len(sn_range)-1):
        if sn_range[i] ==0 and sn_range[i+1]==1:
            sn_range[i] = 1
    
    sn_range_diff = np.diff(np.pad(sn_range,(1,1),'constant',constant_values=(0,0)).astype(int)) # shape = len(sn_range)+1
    p = np.where(sn_range_diff ==1)[0]  # Start of a valid band
    q = np.where(sn_range_diff ==-1)[0] # End of a valid band
    
    try:

        maxlen, ix = max(q-p) , np.argmax(q-p)
        # print(type(ix))
        if type(ix) == int or type(ix) == np.int64:
            imin, imax = p[ix], q[ix]-1
        else:
            #print('All frequency bands are below noise level.')
            imin, imax = 1,1
            
        
        return {'imin':imin, 'imax':imax}
    
    except ValueError:
        
        # print('No valid frequency band under this snr_limit.')
        # print('Maximum SNR =',max(ev_snr))
        
        return {'max_SNR':max(ev_snr)}

def get_f_range_EV(EV_signal,EV_noise, freq, sampling_rate, snr_limit = 9):
    '''
    The single event version of get_f_range_Main_EGF.
    '''
    f_Ny = sampling_rate/2
    
    
    r_EV = valid_f_range(EV_signal,EV_noise, snr_limit)
    
    try:
        imin_EV, imax_EV = r_EV['imin'], r_EV['imax']
        
        imin = max(imin_EV,2) # Start at at least 3 samples from the begining
        imax = imax_EV
        
        fmin = freq[imin]
        fmax = min(freq[imax],0.8*f_Ny) # End at at most 0.8 Nyquist
        if fmax<=fmin:# Added by HC on 2021.9.26
            fmin, fmax = np.nan, np.nan 
    except KeyError: # Situation where valid_f_range returned max_SNR, not imin and imax
        #print('np.nan')
        fmin, fmax = np.nan, np.nan 
        
        
    return {'fmin':fmin, 'fmax':fmax}#, fmin_Main, fmax_Main, fmin_EGF, fmax_EGF


def log_resample(S2, F2, lowf, highf, deltaf):
    '''
    Program to resample displacement spectrum
    Input : 
        S2 - amplitude of the displacement spectrum 
        F2 - frequency of the displacement spectrum 
        lowf - lower frequency of interest (in log units)
        highf - higher frequency of interest (in log units)
        deltaf - frequency intervals in log scale
    Output:
        f_new_dec - new freq points
        amp - corresponding amplitude values
    
    Hilary's version of Rachel's log_resample.m and resampler_dispt.m from Gisela, March 2009
    '''
    f_int = deltaf/2.
    
    f_new_ord = np.arange(lowf, highf+deltaf, deltaf)
    
    # Initialize
    f_new_dec = np.zeros(shape = len(f_new_ord))
    amp = np.zeros(shape = len(f_new_ord))
    
    for j,f in enumerate(f_new_ord):
        count = 0
        amp_sum = 0
        
        for i  in range(len(F2)):
            
            if (F2[i] > 10.**(f-f_int) ) and (F2[i] <= 10.**(f+f_int) ):
                amp_sum += S2[i]
                count += 1
        
        # If no sample is in this freq interval         
        if count == 0:
            count = 1
            for i  in range(len(F2)-1):
                if (F2[i] < 10.**f) and (F2[i+1] > 10.**f):
                    # print('------------')
                    decile = (S2[i+1]-S2[i]) / (F2[i+1]-F2[i])
                    ordinal = S2[i] - decile* F2[i]
                    amp_sum = decile*(10.**f) + ordinal
                    
        f_new_dec[j] = 10.**f
        amp[j] = amp_sum/count
        
    return f_new_dec, amp

def spectra_single_att(f,fc,Omega0,att,n=2,gamma=2):
    
    nom=Omega0*np.exp(-np.pi* f*( att ))
    den=(1+(f/fc)**(gamma*n))**(1/gamma)
    return nom/den

def residual_spec_single_att(params, f, data, att,fmin=0, fmax=100000):
    fc = params['fc']
    Omega0 = params['Omega0']
    n = params['n']
    gamma = params['gamma']
    
    model = spectra_single_att(f,fc,Omega0,att,n=n,gamma=gamma)
    # if fmin!=None:
    ind=np.where( (f>=fmin)&(f<=fmax))[0]
    
    return abs((np.log10(data[ind])-np.log10(model[ind])))

def fit_spec_single_att(x,y,att,
            init_fc,low_f_lv_for_Omega0,
            gamma,n,
            fc_range,Omega0_range,
            method='least_sqaures',
            nan_policy='propagate',print_reports=False,conf_int=False,fmin=0, fmax=100000):
    '''
    Single spectrum version of fit_spec_ratio
    
    '''
    import lmfit
     
    if Omega0_range== None:
        lowf_amps=y[np.where(x<low_f_lv_for_Omega0)[0]]
        init_Omega0=np.nanmean(lowf_amps)
        # Omega0_range=[np.nanmin(lowf_amps)/10,np.nanmax(lowf_amps)*10]
        Omega0_range=[np.nanmin(lowf_amps),np.nanmax(lowf_amps)]
    else:
        init_Omega0=np.nanmean(y[np.where(x<low_f_lv_for_Omega0)[0]])
    
    # print('!!!!!',init_Omega0,Omega0_range)
    if np.isnan(init_Omega0):
         return np.nan,np.nan
      
    # print(lowf_amps)
    # print(Omega0_range)
    # print(init_Omega0)
    # print(y)
    # print('init:\nfc:%s, Omega0:%s'%(init_fc,init_Omega0))
    isNan=np.isnan(y)
    params=lmfit.Parameters()
    #print(Omega0_range)
    params.add('fc',value=init_fc,min=fc_range[0], max=fc_range[1]) #EV
    if Omega0_range[0]< Omega0_range[1]:
        params.add('Omega0',value=init_Omega0,min=Omega0_range[0],max=Omega0_range[1])
    elif Omega0_range[0]== Omega0_range[1]:
        params.add('Omega0',value=init_Omega0,vary=False)

    params.add('n',value=n,vary=False)
    params.add('gamma',value=gamma,vary=False)
    # params.add('gamma',value=gamma,min=1,max=2)
    
        
    mini = lmfit.Minimizer(residual_spec_single_att, 
                           params, nan_policy=nan_policy,
                            fcn_kws=dict(f=x[~isNan], \
                                        data=y[~isNan],
                                        att=att,
                                        fmin=fmin, fmax=fmax\
                                        )
                            )
    try:
        result = mini.minimize(method=method)
    except ValueError: # ValueError: `x0` is infeasible.
        
        print('*******************',init_Omega0)
        print('*******************',params)
        print('*******************',np.isnan(init_Omega0))

        
        raise Exception( 'The ValueError of "x0 is infeasible" is coming because your initial values violate the bounds. Check the parameters values and bounds. ')
    if conf_int==True:
        try:
            ci, trace = lmfit.conf_interval(mini, result, sigmas=[1, 2], trace=True)
        except: # MinimizerException: 
            # Cannot determine Confidence Intervals without sensible uncertainty estimates
            # The result that are bad if it is at the limits
            print('Bad estimate.')
            lmfit.report_fit(result.params, min_correl=0.5) # result
            lmfit.printfuncs.report_ci(ci) # conf. int.
        if print_reports==True:
            lmfit.report_fit(result.params, min_correl=0.5) # result
            lmfit.printfuncs.report_ci(ci) # conf. int.

    return result,mini

def seismic_moment(Omega0,rho,c,R,Ur):
    '''
    R : Hypocentral distance
    rho: Density at the hypocenter 
    c: P wave velocity at the hypocenter
    Ur: mean radiation pattern coefficient (0.52 for P wave from Madariaga, 1976)
    
    density, ρ, is set to 2,790 kg/m3, velocity c for P and S waves is chosen 
    according to the value in the velocity model at the focal depth of respective 
    event, and R is the hypocentral distance. The mean radiation pattern, Ur, 
    is set to be 0.52 and 0.63 for P and S waves, respectively (Aki & Richards, 1980).
    
    
    '''
    return 4*np.pi*rho* (c)**3 *R *Omega0 / Ur

def stress_drop(m0,fc,c,k):
    '''
    https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020JB020103
    k is a constant related to the reciprocal relation between fc and r.
    '''
    return (7/16)*(m0/ ((k*c/fc)**3)) /1e6
    
def slip_from_m0(m0,r,mu):
    return m0/mu/np.pi/(r**2)

def m02Mw_Nm(m0):
    return (np.log10(m0)/1.5)-6.07

def fc_2_src_radius(fc,k,beta):
    return k*beta/fc


def hann_taper(data,percentage=0.1,wlen=None,left_right='both'):
    '''
    Design a Hann taper that tapers the first and last wlen samples.
    Default: Taper length based on percentage of the total points
            If wlen != None, use length as the taper length.
    
    '''
    npts = np.size(data,-1)  # Default: Apply taper to the last dimension
    
    if wlen == None:
        wlen = int(round(npts*percentage))
    
    window = hann(wlen*2)
    #window = scipy.signal.hann(int(0.05 * npts))
    if left_right=='both':
        left = window[:wlen]
        right = window[wlen:]
    elif left_right=='left':
        left = window[:wlen]
        right = np.ones(wlen)
    elif left_right=='right':
        left = np.ones(wlen)
        right = window[wlen:]
    else:
        raise Exception('Available options: "both", "left", "right"')
        
    middle = np.ones(int(npts-wlen*2))
    window = np.concatenate((left, middle, right))
    data_tapered = data* window
    return data_tapered
