#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:10:32 2022

@author: dlvilla
"""

import os
from warnings import warn

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
            
    