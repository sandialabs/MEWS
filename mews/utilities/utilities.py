#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:10:32 2022

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

import os
from warnings import warn
import numpy as np

def dict_key_equal(dict_key,dict_check):
    # This function is recursive
    
    if dict_key.keys() == dict_check.keys():
        for key,val in dict_key.items():
            if isinstance(val,dict) and not isinstance(dict_check[key], dict):
                raise ValueError("The dictionary being checked has a value that should be a dictionary")
            elif isinstance(val,dict):
                dict_key_equal(val,dict_check[key])
                
    else:
        raise ValueError("The dictionary being checked does not have"+
                         " the same keys. It must have the key:\n\n{0}"+
                         "\n\n".format(str(dict_key.keys())))
        
    

def filter_cpu_count(cpu_count):
    """
    Filters the number of cpu's to use for parallel runs such that 
    the maximum number of cpu's is properly constrained.

    Parameters
    ----------
    cpu_count : int or None :
        int : number of proposed cpu's to use in parallel runs
        None : choose the number of cpu's based on os.cpu_count()
    Returns
    -------
    int
        Number of CPU's to use in parallel runs.

    """
    
    if isinstance(cpu_count,(type(None), int)):
        
        max_cpu = os.cpu_count()
        
        if cpu_count is None:
            if max_cpu == 1:
                return 1
            else:
                return max_cpu -1
        
        elif max_cpu <= cpu_count:
            warn("The requested cpu count is greater than the number of "
                 +"cpu available. The count has been reduced to the maximum "
                 +"number of cpu's ({0:d}) minus 1 (unless max cpu's = 1)".format(max_cpu))
            if max_cpu == 1:
                return 1
            else:
                return max_cpu - 1
        else:
            return cpu_count
        
def check_for_nphistogram(hist):
    if not isinstance(hist, tuple):
        raise TypeError("The histogram must be a 2-tuple of two arrays")
    elif not len(hist) == 2:
        raise ValueError("The histogram must be a tuple of length 2")
    elif not isinstance(hist[0],np.ndarray) or not isinstance(hist[1],np.ndarray):
        raise TypeError("The histogram tuples elements must be of type np.ndarray")
    elif len(hist[0]) != len(hist[1]) - 1:
        raise ValueError("The histogram tuple first entry must be one element smaller in lenght than the second entry!")

def histogram_area(hist):
    bin_ = bin_avg(hist)
    h_ = hist[0]
    
    return np.trapz(h_,bin_)

def histogram_step_wise_integral(hist,a=None,b=None):
    """
    Calculates the integral assuming that a constant value is sustained
    for each bin in a histogram

    Parameters
    ----------
    hist : 2-tuple from np.histogram first entry is one less long as second
           first element are values and second element are bin edges
        DESCRIPTION.
    a : float, optional
        lower integration boundary. The default is None. If None, then just
        use the lowest bound of the histogram bins
    b : float, optional
        upper integration boundary. The default is None. If None, then just
        use the highest bound of the histogram bins

    Raises
    ------
    ValueError
        raised if a > b.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if a is None:
        a = hist[1].min()
    if b is None:
        b = hist[1].max()
    
    if a > b:
        raise ValueError("Input 'a' must be greater than input 'b'.")
    if a == b:
        return 0.0
    
    bin_ = hist[1]
    h_ = hist[0]    
    
    if a > bin_[-1]:
        return 0.0
    elif b < bin_[0]:
        return 0.0
    
    
    if a < bin_[0]:
        beg_area = 0.0
        beg_bin = 0
    else:
        # this is the first bin boundary division ahead of a
        diva = (bin_ < a).argmin()
        beg_area = h_[diva-1] * (bin_[diva] - a)
        beg_bin = diva
        
    if b > bin_[-1]:
        end_area = 0.0
        end_bin = len(bin_)
    else:
        divb = (bin_ < b).argmin()
        end_area = h_[divb-1] * (b - bin_[divb-1])
        end_bin = divb-1
        
    return (np.diff(bin_)[beg_bin:end_bin] * h_[beg_bin:end_bin]).sum() + beg_area + end_area
    

def histogram_intersection(hist1,hist2):
    def int_boundaries(bin_,maxb,minb):
        bin_ = bin_[bin_ <= maxb]
        return bin_[bin_ >= minb]
    
    bin1 = bin_avg(hist1)
    bin2 = bin_avg(hist2)
    h1 = hist1[0]
    h2 = hist2[0]
    
    maxintb = np.min([bin1.max(),bin2.max()])
    minintb = np.max([bin1.min(),bin2.min()])
    
    bin1int = int_boundaries(bin1,maxintb,minintb)
    bin2int = int_boundaries(bin2,maxintb,minintb)
    
    intpoint = np.unique(np.concatenate([bin1int,bin2int]))
    
    interp_vals1 = np.interp(intpoint,bin1,h1)
    interp_vals2 = np.interp(intpoint,bin2,h2)
    
    minvals = np.min(np.concatenate([interp_vals1.reshape([len(interp_vals1),1]),
                           interp_vals2.reshape([len(interp_vals2),1])],axis=1),axis=1)
    
    return np.trapz(minvals,intpoint)


def histogram_non_overlapping(hist1,hist2,return_min_max=False):
    bin1 = bin_avg(hist1)
    bin2 = bin_avg(hist2)
    h1 = hist1[0]
    h2 = hist2[0]
    
    # find max boundaries 
    maxintb = np.min([bin1.max(),bin2.max()])
    minintb = np.max([bin1.min(),bin2.min()])
    
    maxb = np.max([bin1.max(),bin2.max()])
    minb = np.min([bin1.min(),bin2.min()])
    
    
    # establish what points to integrate
    if maxb in bin1:
        elem = bin1 >= maxintb
        intpoint_max = bin1[elem]
        h_max = h1[elem]
    else:
        elem = bin2 >= maxintb
        intpoint_max = bin2[elem]
        h_max = h2[elem]
    
    if minb in bin1:
        elem = bin1 <= minintb
        intpoint_min = bin1[elem]
        h_min = h1[elem]
    else:
        elem = bin2 <= minintb
        intpoint_min = bin2[elem]
        h_min = h2[elem]
        
    # integrate
    min_side_area = np.trapz(h_min,intpoint_min)
    max_side_area = np.trapz(h_max,intpoint_max) 
    
    if return_min_max:
        return min_side_area, max_side_area
    else:
        return min_side_area + max_side_area


def histogram_non_intersection(hist1,hist2):
    """
    Calculates a numeric approximation of the non-intersecting area of 
    two histograms. values are assumed to occur at the centriod of each bin
    significant errors may result for low-resolution histograms

    Parameters
    ----------
    hist1 : tuple output from numpy.histogram
        first histogram
    hist2 : tuple output from numpy.histogram
        second histogram

    Returns
    -------
    float
        Non-intersecting area of a histogram.

    """
    
    intersect_area = histogram_intersection(hist1,hist2)
    non_overlapping_area = histogram_non_overlapping(hist1,hist2)
    
    bin1 = bin_avg(hist1)
    bin2 = bin_avg(hist2)
    
    area1 = np.trapz(hist1[0],bin1)
    area2 = np.trapz(hist2[0],bin2)
    
    return area1 + area2 - 2 * intersect_area
    

def create_complementary_histogram(sample, hist0):
    
    """
    This function recieves a sample from a dynamic random process and then creates
    a histogram of the sample using the same spacing interval as a histogram 'hist0'
    It then returns the histogram as a discreet probability distribution function tuple
    and as a cumulative probability distribution (CDF) function tuple
    
    Parameters
    ----------
    
    sample : np.array
        a 1-d array of sample values for a dynamic process.
        
    hist0 : tuple returned by np.histogram
    
    Returns
    -------
    tup1, tup2 - tup1 is the sample discreet pdf and tup2 is the sample 
                 discreet cdf
                 
    Raises
    -----
    TypeError - if sample is not an np.array or hist0 is not a tuple of arrays
    ValueError - if the input histogram 
    
    """
    if len(sample) == 0:
        # no waves occured and all zeros must be used to evaluate
        hist1 = np.zeros(len(hist0[0]))
        bin_avg = (hist0[1][1:]+hist0[1][0:-1])/2
        return (hist1,hist0[1]), (np.ones(len(hist0[0])), bin_avg)
    else:
    
        check_for_nphistogram(hist0)
        
        bin_spacing = np.diff(hist0[1]).mean()
        
        start = np.floor(sample.min() / bin_spacing)
        end = np.ceil(sample.max()/ bin_spacing)
        
        num_bin = int(end - start)
        
        hist1 = np.histogram(sample,num_bin,range=(start*bin_spacing,end*bin_spacing))
        
        bin_avg = (hist1[1][1:]+hist1[1][0:-1])/2
        
        hist1_prob = hist1[0]/hist1[0].sum() 
        
        cdf = (hist1_prob).cumsum()
        
        return (hist1_prob,hist1[1]), (cdf, bin_avg)

def linear_interp_discreet_func(vals, discreet_func_tup, is_x_interp=False):
    """
    linear interpolation of a function. Naturally written for y interpolation
    of a cdf but just have to reverse terms for x-interpolation
    
    
    """
    
    if is_x_interp:
        xf = discreet_func_tup[0]
        yf = discreet_func_tup[1]
    else:
        xf = discreet_func_tup[1]
        yf = discreet_func_tup[0]

    len_s = len(xf)
    # must handle values beyond the range of yf (=xf if is_x_interp=True)
    idi = np.array([0 if va <= yf[0] else len_s if va >= yf[-1] else np.where(np.logical_and(yf[0:-1]<=va,yf[1:]>va))[0][0] for va in vals])
    interp = np.array([ xf[idix] + (va - yf[idix])/(yf[idix+1]-yf[idix]) * (xf[idix+1]-xf[idix])
                     if idix < len_s
                     else 
                      xf[-1]
                     for va, idix in zip(vals,idi) ])
    return interp
            
def find_extreme_intervals(states_arr,states):
    """
    This function returns a dictionary whose entry keys are 
    the "states" input above. Each dictionary element contains 
    a list of tuples. Each tuple contains the start and end times 
    of an event where "states_arr" was equal to the corresponding state

    Parameters
    ----------
    states_arr : array-like
        a 1-D array of integers of values that are only in the states input
    states : array-like,list-like
        a 1-D array of values to look for in states_arr. 

    Returns
    -------
    state_int_dict : TYPE
        A dictionary with key "state" each value contains a list of tuples
        that indicate the start and end time of an extreme event.

    """ 
    
    diff_states = np.concatenate((np.array([0]),np.diff(states)))
    state_int_dict = {}
    for state in states:
        state_ind = [i for i, val in enumerate(states_arr==state) if val]
        end_points = [i for i, val in enumerate(np.diff(state_ind)>1) if val]
        start_point = 0
        if len(end_points) == 0 and len(state_ind) > 0:
            ep_list = [(state_ind[0],state_ind[-1])]
        elif len(end_points) == 0 and len(state_ind) == 0:
            ep_list = []
        else:
            ep_list = []
            for ep in end_points:
                ep_list.append((state_ind[start_point],state_ind[ep]))
                start_point = ep+1
        state_int_dict[state] = ep_list
    return state_int_dict    

def _bin_avg(hist):
    return (hist[1][1:] + hist[1][0:-1])/2

def bin_avg(hist):
    if len(hist[1]) == len(hist[0]) + 1:
        bin0 = _bin_avg(hist)
    else:
        bin0 = hist[1]
    return bin0