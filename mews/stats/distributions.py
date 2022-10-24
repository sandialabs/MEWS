#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 10:47:41 2022

Copyright 2021 National Technology and Engineering Solutions of Sandia, LLC. 
Under the terms of Contract DE-NA0003525, there is a non-exclusive license 
for use of this work by or on behalf of the U.S. Government. 
Export of this program may require a license from the 
United States Government.

Please refer to the LICENSE.md file for a full description of the license
terms for MEWS. 

The license for MEWS is the Modified BSD License and copyright information
must be replicated in any derivative works that use the source code.

@author: dlvilla
"""

from scipy.special import erf
from scipy.optimize import curve_fit, bisect
from scipy.interpolate import splev, splrep
import numpy as np
import matplotlib.pyplot as plt


def pdf_density_given_value(x, pdf):
    bins = pdf[1]
    dens = pdf[0]
    len_prob = len(prob)
    idi = np.array([np.where(np.logical_and(bins[0:-1]<=binx,bins[1:]>binx))[0][0] for binx in x])
    
    interp = np.array([ bins[idix] + (Pr - prob[idix])/(prob[idix+1]-prob[idix]) * (bins[idix+1]-bins[idix])
                     if idix < len_prob
                     else 
                      bins[-1]
                     for Pr, idix in zip(P,idi) ])



def cdf_exponential(x,lamb):
    return 1-np.exp(-lamb * x)

# TODO make this a class such that erf(a) and erf(b) are not recalculated.
def cdf_truncnorm(x,mu,sig,a,b):
    #https://en.wikipedia.org/wiki/Truncated_normal_distribution
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        xi = (x - mu)/sig
        alpha = (a - mu)/sig
        beta = (b - mu)/sig
        erf_alpha = erf(alpha)
        return (erf(xi) - erf_alpha)/(erf(beta) - erf_alpha)

def offset_cdf_truncnorm(x,mu,sig,a,b,rnd):
    #https://en.wikipedia.org/wiki/Truncated_normal_distribution
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        xi = (x - mu)/sig
        alpha = (a - mu)/sig
        beta = (b - mu)/sig
        erf_alpha = erf(alpha)
        return (erf(xi) - erf_alpha)/(erf(beta) - erf_alpha) - rnd
    
def trunc_norm_dist(rnd,mu,sig,a,b,minval,maxval):
    # inverse lookup from cdf
    x, r = bisect(offset_cdf_truncnorm, a, b, args=(mu,sig,a,b,rnd),full_output=True)
    if r.converged:
        return inverse_transform_fit(x,maxval,minval)
    else:
        raise ValueError("The bisection method failed to converge! Please investigate")
    
def transform_fit(value,minval,maxval):
    # this function maps minval to maxval from -1 to 1
    return 2 *  (value - minval)/(maxval - minval) - 1
                 
# TODO - merge these functions with those in ExtremeTemperatureWaves
def inverse_transform_fit(norm_signal, signal_max, signal_min):
    return (norm_signal + 1)*(signal_max - signal_min)/2.0 + signal_min 

def fit_exponential_distribution(month_duration_hr,include_plots):
    """
    

    Parameters
    ----------
    month_duration_hr : np.array
        for a given month, a list of 
    include_plots : bool
        True = plot histogram of results and exponential fit

    Returns
    -------
    None.

    """
    
    
    hour_in_day = 24
    # this always comes out to a positive integer under the constraints present 
    num_bin = int((month_duration_hr.max() - month_duration_hr.min())/hour_in_day)+1
    month_dur_histogram = np.histogram(month_duration_hr,num_bin,range=(month_duration_hr.min()-hour_in_day/2,
                                                                        month_duration_hr.max()+hour_in_day/2))
    bin_avg = (month_dur_histogram[1][1:]+month_dur_histogram[1][0:-1])/2
    hist_norm = (month_dur_histogram[0]/month_dur_histogram[0].sum()).cumsum()
    
    lamb, lcov = curve_fit(cdf_exponential,bin_avg,hist_norm,0)
    
    pvalue = 1.96 * np.max(lcov[0][0])/lamb[0]
    
    if include_plots:
        plt.plot(bin_avg,hist_norm,np.arange(0,month_duration_hr.max()),cdf_exponential(np.arange(0,month_duration_hr.max()),lamb[0]))
        
    P0 = 1 - lamb[0]
    
    return P0,lamb,lcov,pvalue