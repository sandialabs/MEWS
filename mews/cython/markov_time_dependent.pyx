#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:35:07 2022

@author: dlvilla

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

This time dependent markov process takes on a decay term

These functions must exactly replicate the mews.stats.markov_time_dependent

python based functions


The functions here are for a very specific form of Markov process and 
will not work for a generalized transition matrix. The first row is 
assumed to be fully populated but all subsequent rows only have 2 terms
in the transitions matrix with all other terms equal to zero. A correct example
form of the transition matrix is shown below. 

M = [[0.2, 0.2, 0.2, 0.2, 0.2],
     [0.5, 0.5, 0.0, 0.0, 0.0],
     [0.5, 0.0, 0.5, 0.0, 0.0],
     [0.5, 0.0, 0.0, 0.5, 0.0],
     [0.5, 0.0, 0.0, 0.0, 0.5]]

The input is not the transition matrix though but the cumulative distribution
function M.cumsum(axis=1)

"""

import numpy as np
cimport numpy as np
from math import exp




DTYPE = np.float
ctypedef np.float_t DTYPE_t

cpdef DTYPE_t exponential_decay(DTYPE_t time_in_state,
                                DTYPE_t lamb):
    """
    This provides an exponential decay of probability of a specific state
    continuing
    
    """
    cdef DTYPE_t val
    val = exp(-time_in_state * lamb)
    return val

cpdef DTYPE_t exponential_decay_with_cutoff(DTYPE_t time_in_state,
                                DTYPE_t lamb,
                                DTYPE_t cutoff):
    """
    This provides an exponential decay of probability of a specific state
    continuing
    
    TODO - markov processes cutoff in cutoff + 2. Fixing this in cython
    is finicky so I am delaying fixing for now.
    
    """
    cdef DTYPE_t val
    cdef DTYPE_t zero = 0.0
    
    if time_in_state >= cutoff:
        val = zero
    else:
        val = exp(-time_in_state * lamb)
        
    return val

cpdef DTYPE_t delayed_exponential_decay_with_cutoff(DTYPE_t time_in_state,
                                DTYPE_t lamb,
                                DTYPE_t cutoff,
                                DTYPE_t delay):
    """
    This provides an exponential decay of probability after a delay time "delay"
    of a specific state continuing. Probability reduces to zero at cutoff time
    "cutoff"
    
    TODO - THIS FUNCTION IS NOT DONEmarkov processes cutoff in cutoff + 2. Fixing this in cython
    is finicky so I am delaying fixing for now.
    
    """
    cdef DTYPE_t val
    cdef DTYPE_t zero = 0.0
    cdef DTYPE_t one = 1.0
    
    if time_in_state >= cutoff + delay:
        val = zero
    elif time_in_state <= delay:
        val = one
    else:
        val = exp(-(time_in_state - delay) * lamb)
        
    return val

cpdef DTYPE_t linear_decay(DTYPE_t time_in_state,
                                DTYPE_t slope):
    """
    This provides a linear decay of probability of a specific state
    continuing. This could enable a complete prohibition of continuing
    
    """
    cdef DTYPE_t val
    cdef DTYPE_t one = 1.0
    cdef DTYPE_t zero = 0.0
    
    val = one - slope * time_in_state
    if val < zero:
        val = zero
    return val


cpdef DTYPE_t linear_decay_with_cutoff(DTYPE_t time_in_state,
                                DTYPE_t slope,
                                DTYPE_t cutoff):
    """
    This provides a linear decay of probability of a specific state
    continuing. If the time_in_state exceeds cutoff+2, then the 
    probability of continuing is set to zero.
    
    TODO - fix this so that it is cutoff rather than cutoff+2
    
    """
    cdef DTYPE_t val
    cdef DTYPE_t one = 1.0
    cdef DTYPE_t zero = 0.0
    
    if time_in_state >= cutoff:
        val = zero
    else:    
        val = one - slope * time_in_state
        if val < zero:
            val = zero
    
    return val

cpdef DTYPE_t quadratic_times_exponential_decay_with_cutoff(DTYPE_t time_in_state,
                                DTYPE_t time_to_peak,
                                DTYPE_t Pmax,
                                DTYPE_t P0,
                                DTYPE_t cutoff_time):
    
    
    """
    This function provides a controlled method to increase probability of
    sustaining a heat wave until 'time_to_peak' after this time, the heat
    wave sustaining probability drops. Pmax is the maximum <= 1.0 and 
    P0 is the initial probability. There are therefore 4 parameters 
    for this fit including a cutoff time at which probability drops to zero.
    
    """
    cdef DTYPE_t val
    cdef DTYPE_t one = 1.0
    cdef DTYPE_t two = 2.0
    cdef DTYPE_t zero = 0.0
    cdef DTYPE_t t_dimensionless = time_in_state / time_to_peak
    
    if time_in_state > cutoff_time:
        val = zero
    else:
        # no multiplication of P0 here because that is done in
        # evaluate_decay_function
        val = (one + (t_dimensionless)**two * exp(two) * 
                    (Pmax/P0 - 1)* exp(-two * t_dimensionless))
        
        if time_in_state >= two * time_to_peak:
            val = val * exp(-exp(1)**(-2) * abs(Pmax/P0-1) * (t_dimensionless-2)**2)
    
    return val
    
    

cpdef double[:] evaluate_decay_function(np.ndarray[DTYPE_t,ndim=1] cdf0,
                                    np.int_t func_type,
                                    np.ndarray[DTYPE_t, ndim=2] coef,
                                    np.int_t idy,
                                    DTYPE_t time_in_state):
    
    cdef DTYPE_t one = 1.0
    cdef DTYPE_t func_eval = one
    cdef DTYPE_t P0 = one - cdf0[0]
    cdef np.ndarray[DTYPE_t, ndim=1] cdf1 = np.zeros(len(cdf0))
    cdef np.int_t idym1 = idy - 1
    
    if time_in_state <= 0.0:
        return cdf0 # no change to the cdf
    
    if func_type == 0:
        # use exponential decay
        func_eval = exponential_decay(time_in_state,coef[idym1,0])
    elif func_type == 1:
        # use linear decay
        func_eval = linear_decay(time_in_state,coef[idym1,0])
    elif func_type == 2:
        func_eval = exponential_decay_with_cutoff(time_in_state,coef[idym1,0],coef[idym1,1])
    elif func_type == 3:
        func_eval = linear_decay_with_cutoff(time_in_state,coef[idym1,0],coef[idym1,1])
    elif func_type == 4:
        func_eval = quadratic_times_exponential_decay_with_cutoff(time_in_state,
                                                                  coef[idym1,0],
                                                                  coef[idym1,1],
                                                                  P0,
                                                                  coef[idym1,2])
    elif func_type == 5:
        func_eval = delayed_exponential_decay_with_cutoff(time_in_state,
                                                          coef[idym1,0],
                                                          coef[idym1,1],
                                                          coef[idym1,2])
    else:
        print("func_type must be 0,1,2,3,4 or 5...upredictable behavior is resulting!")
        
    cdf1[0] = cdf0[0] + P0 * func_eval
    cdf1[idy] = one - P0 * func_eval
    
    return cdf1

cpdef np.ndarray[np.int_t, ndim=1] markov_chain_time_dependent(np.ndarray[DTYPE_t, ndim=2] cdf, 
                                               np.ndarray[DTYPE_t, ndim=1] rand, 
                                               np.int_t state0,
                                               np.ndarray[DTYPE_t, ndim=2] coef,
                                               np.ndarray[np.int_t, ndim=1] func_type):
    
    """
    This function creates a Markov chain whose 2nd ... num row rows
    represent extreme event types whose probability of continuing decays 
    with each step in that state. The first row is constant probability
    whereas the 2nd .. num rows are of the form
    
    row n = [1-P0 + P0*f(t), 0, ...(to position n) .., 1-P0*f(t), 0, ...]
    
    The decay functions f(t) must be of the form where f(0) = 1 and then
    monotonically decreases to zero or assymptotically approaches a number
    greater than or equal to zero that is less than or equal to 1
    
    Parameters
    ----------
    cdf : a discreet cumulative probability distribution function. This will
          be altered for rows 2..num rows as indicated above
          
    rand : an array of random numbers that indicates the number of steps to
           take.
           
    state0 : the initial state of the random process
    
    coef : an array of coefficients that must be of size  (m-1)xp where 
           m is the number of states (cdf.shape[0]) and p is the number of 
           coefficients that the decay functions require. A new function
           has to be created in this module if a new function type is needed.
           for example the function "exponential_decay" has already been added
           and p = 1 since only one lambda exponent is needed.
           
           Note: The first state does not need coefficients because it is 
           assumed to be a constant markov process. Only rows 2...m have 
           time decay.
           
    func_type : an integer array that indicates what function type to use. 
           
           0 - use exponential_decay
           1 - use linear_decay
           2 - use exponential_decay with cut-off point
           3 - use linear_decay with cut-off point
           4 - use quadratic time exponential that peaks at a specific time and
               then decays. This function increases probability of sustaining
               a heat wave and then decays after the peak
           5 - use delayed exponential with cutoff.
               
               for func_type 0
                   coef is 1 element = lambda for exp(-lambda * t)
                   
               1: coef has 1 element = slope for slope * t
               2: coef has 2 elements = lambda and a cutoff time
               3: coef has 2 elements = slope and a cutoff time
               4: coef has 3 elelments = 1) time_to_peak, 2) Peak Maximum Probability,
                                         3) cutoff time at which probability drops
                                         to zero.
               5: coef has 3 elements = 1) lambda for exp (-lambda * t),
                                        2) cutoff time where probability = 0
                                        3) delay time before exponential decay begins
                                
    
    
    Returns
    -------
    yy : state vector of len(rand)
         if the function returns all -999,
         then an incorrect input for the func_type was given.
    
    This function is intended to be used inside: 
        mews.stats.markov_time_dependent.markov_chain_time_dependent_wrapper
    which thoroughly checks the inputs so that difficult cython type errors
    do not occur. The function does not protect against bad inputs on its own.
    
    """
    # yy is the output sample of states.
    # assign initial values
    cdef np.int_t num_step = len(rand)
    cdef np.ndarray[np.int_t,ndim=1] yy = np.zeros(num_step,dtype=np.int)
    
    cdef np.int_t one = 1
    cdef np.int_t num_state = cdf.shape[0]
    cdef np.int_t idx
    cdef np.int_t idy
    cdef np.int_t idym1 = idy - one

    cdef np.int_t step_in_cur_state
    cdef double[:] cdf_local
    
    # assign first value the initial value.
    yy[0] = state0
    step_in_cur_state = 0
    

    for idx in range(1, num_step):
        
        # threshold = 0.005
        # if rand[idx] < threshold:
        #     breakpoint()
        
        for idy in range(num_state):
            
            # for rows other than the first row this will bring 
            # significant savings because our cdf is only
            # two terms. The first is always the only change before 
            # state0 
            if state0 > 0 and idy > 0:
                # no further evaluation needed, for the form of cdf used
                # here where the first row is fully populated and all subsequent6
                # rows only have two values that change, we know that the 
                # the evaluation at idy=0 is the only transition out of 
                # the current state. We can immediately conclude we stay in the 
                # current state.
                yy[idx] = state0
                step_in_cur_state += 1
                break 
            else:
                if state0 > 0:
                    # func_type is mapped at a lag of 1 from num_state
                    # since state=0 does not have a decay function.
                    idym1 = idy - 1 
                    cdf_local = evaluate_decay_function(cdf[state0,:],
                                                        func_type[idym1],
                                                        coef,
                                                        idy,
                                                        step_in_cur_state)
                else:
                    cdf_local = cdf[state0,:]
                    
                # evaluate if whether next cdf value is greater.
                if cdf_local[idy] > rand[idx]:
                    yy[idx] = idy
                    if idy != yy[idx-1]:
                        step_in_cur_state = 0
                    else:
                        step_in_cur_state += 1
                    
                    state0 = idy
                    break
                if idy == num_state-1:
                    yy[idx] = idy
                    state0 = idy
                    if idy != yy[idx-1]:
                        step_in_cur_state = 0
                    else:
                        step_in_cur_state += 1
    
    return yy
