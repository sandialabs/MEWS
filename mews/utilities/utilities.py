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
    check_for_nphistogram(hist0)
    
    bin_spacing = np.diff(hist0[1])
    
    # assure a constant
    if (bin_spacing < bin_spacing[0] *1.00001).any() or (bin_spacing > bin_spacing[0]).any():
        raise ValueError("The histogram input 'hist0' must have a constant bin spacing")
    


    # this always comes out to a positive integer under the constraints present 
    num_bin = int((sample.max() - sample.min())/bin_spacing[0])+1
    
    hist1 = np.histogram(sample,num_bin,range=(sample.min()-bin_spacing[0]/2,
                                               sample.max()+bin_spacing[0]/2))
    
    bin_avg = (hist1[1][1:]+hist1[1][0:-1])/2
    
    hist1_prob = hist1[0]/hist1[0].sum() 
    
    cdf = (hist1_prob).cumsum()
    
    return (hist1_prob,hist1[1]), (cdf, bin_avg)
            
    