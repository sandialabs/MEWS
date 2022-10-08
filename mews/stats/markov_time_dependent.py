#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:35:07 2022

@author: dlvilla

This time dependent markov process takes on a decay term

These functions must exactly replicate the mews.cython.markov_time_dependent.pyx

cython based functions

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
from math import exp


# All of these functions must have f(time_in_state=0) = 1 and must monotonically
# decrease or stay constant never going below zero.
def exponential_decay(time_in_state,
                      lamb):
    """
    This provides an exponential decay of probability of a specific state
    continuing
    
    """
    val = exp(-time_in_state * lamb)
    return val

def exponential_decay_with_cutoff(time_in_state,
                                  lamb,
                                  cutoff):
    """
    This provides an exponential decay of probability of a specific state
    continuing
    
    """
    zero = 0.0
    
    if time_in_state > cutoff:
        val = zero
    else:
        val = exp(-time_in_state * lamb)
        
    return val

def linear_decay(time_in_state,
                 slope):
    """
    This provides a linear decay of probability of a specific state
    continuing. This could enable a complete prohibition of continuing
    
    """

    one = 1.0
    zero = 0.0
    
    val = one - slope * time_in_state
    if val < zero:
        val = zero
    return val


def linear_decay_with_cutoff(time_in_state,
                             slope,
                             cutoff):
    """
    This provides a linear decay of probability of a specific state
    continuing. If the time_in_state exceeds cutoff, then the 
    probability of continuing is set to zero.
    
    """
    #cdef DTYPE_t val
    one = 1.0
    zero = 0.0
    
    if time_in_state > cutoff:
        val = zero
    else:    
        val = one - slope * time_in_state
        if val < zero:
            val = zero
    
    return val


def evaluate_decay_function(cdf0,
                            func_type,
                            coef,
                            idy,
                            time_in_state):
    
    one = 1.0
    P0 = one - cdf0[0]
    cdf1 = np.zeros(cdf0.shape)
    idym1 = idy - 1
    
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
    else:
        return np.zeros(len(cdf0))
        
    cdf1[0] = cdf0[0] + P0 * func_eval
    cdf1[idy] = one - P0 * func_eval
    
    return cdf1

def markov_chain_time_dependent_py(cdf, 
                                rand, 
                                state0,
                                coef,
                                func_type):
# cpdef np.ndarray[np.int_t, ndim=1] markov_chain_time_dependent(np.ndarray[DTYPE_t, ndim=2] cdf, 
#                                                np.ndarray[DTYPE_t, ndim=1] rand, 
#                                                np.int_t state0,
#                                                np.ndarray[DTYPE_t, ndim=2] coef,
#                                                np.int_t func_type):
    
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
           
    func_type : an integer that indicates what function type to use. 
           
           0 - use exponential_decay
           1 - use linear_decay
           2 - use exponential_decay with cut-off point
           3 - use linear_decay with cut-off point
    
    
    Returns
    -------
    
    """
    
    # yy is the output sample of states.
    # assign initial values
    #cdef np.int_t num_step = len(rand)
    #cdef np.ndarray[np.int_t, ndim=1] yy = np.zeros(num_step,dtype=np.int)
    num_step = len(rand)
    yy = np.zeros(num_step,dtype=int)
    
    num_state = cdf.shape[0]
    #cdef np.int_t num_state = cdf.shape[0]
    #cdef np.int_t idx
    #cdef np.int_t idy
    #cdef np.int_t step_in_cur_state
    #cdef np.ndarray[DTYPE_t, ndim=1] cdf_local
    
    # assign first value the initial value.
    yy[0] = state0
    step_in_cur_state = 0
    invalid_input = -999
    
    for idx in range(1, num_step):
        
        # threshold = 0.005
        # if rand[idx] < threshold:
        #     breakpoint()
        
        for idy in range(num_state):
            
            # for rows other than the first row this will bring 
            # significant savings because our cdf is only
            # two terms. The first is always the only change before 
            # state0 
            if state0 > 0 and idy >0:
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
                    cdf_local = evaluate_decay_function(cdf[state0,:],
                                                        func_type,
                                                        coef,
                                                        idy,
                                                        step_in_cur_state)
                    # this is a mechanism for indicating that the 
                    # function type is invalid. An all -999 response is
                    # incorrect.
                    if cdf_local.sum() == 0.0:
                        return invalid_input*np.ones(num_step,dtype=np.int)
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
    
    return yy