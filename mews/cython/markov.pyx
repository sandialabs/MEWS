# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:20:54 2021

@author: dlvilla
"""
import numpy as np
cimport numpy as np


DTYPE = np.float
ctypedef np.float_t DTYPE_t


cpdef np.ndarray[np.int_t, ndim=1] markov_chain(np.ndarray[DTYPE_t, ndim=2] cdf, 
                                               np.ndarray[DTYPE_t, ndim=1] rand, 
                                               np.int_t state0):
        
    # assign initial values
    cdef np.int_t num_step = len(rand)
    cdef np.ndarray[np.int_t, ndim=1] yy = np.zeros(num_step,dtype=np.int)
    
    
    cdef np.int_t num_state = cdf.shape[0]
    cdef np.int_t idx
    cdef np.int_t idy
    
    
    for idx in range(num_step):
        for idy in range(num_state):
            if cdf[state0,idy] > rand[idx]:
                yy[idx] = idy
                state0 = idy
                break
            if idy == num_state-1:
                yy[idx] = idy
    
    return yy
    
    