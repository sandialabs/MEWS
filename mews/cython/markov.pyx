# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:20:54 2021

Copyright Notice
=================

Copyright 2023 National Technology and Engineering Solutions of Sandia, LLC. 
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
import numpy as np
cimport numpy as np


DTYPE = np.float64
ctypedef np.float_t DTYPE_t


cpdef np.ndarray[int, ndim=1] markov_chain(np.ndarray[DTYPE_t, ndim=2] cdf, 
                                               np.ndarray[DTYPE_t, ndim=1] rand, 
                                               int state0):
    # assign initial values
    cdef int num_step = len(rand)
    cdef np.ndarray[int, ndim=1] yy = np.zeros(num_step,dtype=int)
    
    
    cdef int num_state = cdf.shape[0]
    cdef int idx
    cdef int idy
    
    for idx in range(num_step):
        for idy in range(num_state):
            if cdf[state0,idy] > rand[idx]:
                yy[idx] = idy
                state0 = idy
                break
            if idy == num_state-1:
                yy[idx] = idy
    
    return yy
    
   
