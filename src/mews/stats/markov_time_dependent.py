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

------------------



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
from mews.cython.markov_time_dependent import markov_chain_time_dependent

def cython_function_input_checks(cdf, 
                                rand, 
                                state0,
                                coef,
                                func_type):
    def array_check(arr,type_needed,inp_name,dim_needed):
        if not isinstance(arr,np.ndarray):
            raise TypeError("input '"+inp_name+"' must be a numpy array (i.e. type np.ndarray)")
        else:
            if not isinstance(arr[tuple([0 for idx in range(dim_needed)])], type_needed):
                raise TypeError("Array input '"+inp_name+"' must have elements of type "+str(type_needed))
                
        if len(arr.shape) != dim_needed:
            raise ValueError("Array input '"+inp_name+"' must be of dimension {0:d}".format(dim_needed))
            
    def int_check(intvar,name,bounds):
        if not isinstance(intvar, int):
            try:
                intnew = int(intvar)
            except:
                raise ValueError("Integer input '"+name+"' must be convertable into an integer.")
        else:
            intnew = intvar
            
        if intnew > bounds[1] or intnew < bounds[0]:
            raise ValueError("Integer input '"+name+"' must be within the interval "+str(bounds))
            
        return intvar
        
        
    # Thorough protection of inputs is required to avoid strange cython errors that the user will not understand
    array_check(cdf,float,"cdf",2)
    array_check(rand,float,"rand",1)
    state0 = int_check(state0,'state0',[0,cdf.shape[0]-1])
    array_check(func_type,np.integer,'func_type',1)
    for idx, ifunc in enumerate(func_type):
        int_check(ifunc, 'func_type[{0:d}]'.format(idx), [0,5])

    if (rand > 1).any() or (rand < 0).any():
        raise ValueError("The rand input vector elements must be a probabilities (i.e. 0<=rand<=1)")
    elif (cdf > 1.0+1e-10).any() or (cdf < 0.0-1e-10).any():
        print("BEGIN CDF\n\n")
        print(cdf)
        print("END CDF\n\n")
        raise ValueError("The cdf input matrix's elements must be probabilities (i.e. 0<=cdf<=1)")
    elif cdf.shape[0] != cdf.shape[1]:
        raise ValueError("The cdf must be a square matrix!")
    

    for idx, col in enumerate(coef):
        ftype = func_type[idx]
        if ftype == 0 or ftype == 1 or ftype == 2 or ftype == 3:
            if col[0] > 1.0 or col[0] < 0.0:
                raise ValueError("Exponent/slope coefficients must be between 0 and 1 for reasonable decays")
                
        if ftype == 2 or ftype == 3:
            if col[1] < 0.0:
                raise ValueError("The cutoff time must be greater than zero.")
                
        if ftype == 4:
            #coef - 1) time_to_peak
            #       2) maximum probability
            #       3) cutoff time!!
            if col[0] < 0.0:
                raise ValueError("The peak time must be greater than zero!")
            elif col[2] < 0.0:
                raise ValueError("The cutoff time must be greater than zero!")
            elif col[1] < 0.0 or col[1] > 1.0:
                raise ValueError("The maximum probability must be between 0 and 1.")
                
        if ftype == 5:
            # coef 1) lambda for exp (-lambda * t),
            #      2) cutoff time where probability = 0
            #      3) delay time before exponential decay begins
            if col[0] > 1.0 or col[0] < 0.0:
                raise ValueError("Exponent/slope coefficients must be between 0 and 1 for reasonable decays")
            if col[1] < 0.0:
                raise ValueError("Cutoff time must be greater than 0.0.")
            if col[2] < 0.0:
                raise ValueError("Delay time must be greater than 0.0")
                
            
        if ftype > 5:
            raise ValueError("function types of 0 to 5 are allowed! A higher value of {0:d} was given".format(ftype))
    

    func_type_int32 = np.array([np.long(ftype) for ftype in func_type])

            
    return state0, func_type_int32


def markov_chain_time_dependent_wrapper(cdf, 
                                rand, 
                                state0,
                                coef,
                                func_type,
                                check_inputs=True):
    """
    This function wraps around the cython implementation of 
    mews.cython.markov_time_dependent.markov_chain_time_dependent to handle 
    its error conditions gracefully and provide proper raising of python 
    exceptions
    
    See the function explanation for 
    mews.cython.markov_time_dependent.markov_chain_time_dependent
    for clear documentation of the function.
    
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
           4 - use quadratic time exponential that peaks at a specific time and
               then decays. This function increases probability of sustaining
               a heat wave and then decays after the peak
           5 - use delayed exponential decay with cutoff
               
               for func_type 0
                   coef is 1 element = lambda for exp(-lambda * t)
                   
               1: coef has 1 element = slope for slope * t
               2: coef has 2 elements = lambda and a cutoff time
               3: coef has 2 elements = slope and a cutoff time
               4: coef has 4 elelments = 1) time_to_peak, 2) Peak Maximum Probability,
                                         and 3) cutoff time at which probability drops
                                         to zero.
                                         
               5: coef has 3 elements = 1) lambda for exp (-lambda * t),
                                         2) cutoff time where probability = 0
                                         3) delay time before exponential decay begins
           
    check_inputs : bool : optional : Default = True
        Check all of the inputs types to assure that cython
        c code will not throw a strange error that is hard to understand 
        for python coders.
    
    
    Returns
    -------
    yy : state vector of len(rand)
         if the function returns all -999,
         then an incorrect input for the func_type was given.
    
    
    """
    # this should not change the value of state0 and func_type but will
    # change from float to int if the wrong type is passed
    if check_inputs:
        state0, func_type = cython_function_input_checks(cdf, rand, state0, coef, func_type)

    
    yy = markov_chain_time_dependent(cdf, 
                                    rand, 
                                    state0,
                                    coef,
                                    np.long(func_type))
    return yy


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
    
    if time_in_state >= cutoff:
        val = zero
    else:
        val = exp(-time_in_state * lamb)
        
    return val

def delayed_exponential_decay_with_cutoff(time_in_state,
                                          lamb,
                                          cutoff,
                                          delay):
    """
    This provides an exponential decay of probability after a delay time "delay"
    of a specific state continuing. Probability reduces to zero at cutoff time
    "cutoff"
    
    TODO - THIS FUNCTION IS NOT DONEmarkov processes cutoff in cutoff + 2. Fixing this in cython
    is finicky so I am delaying fixing for now.
    
    """
    zero = 0.0
    one = 1.0
    
    if time_in_state >= cutoff + delay:
        val = zero
    elif time_in_state <= delay:
        val = one
    else:
        val = exp(-(time_in_state - delay) * lamb)
        
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
    
    if time_in_state >= cutoff:
        val = zero
    else:    
        val = one - slope * time_in_state
        if val < zero:
            val = zero
    
    return val

def quadratic_times_exponential_decay_with_cutoff(time_in_state,
                                time_to_peak,
                                Pmax,
                                P0,
                                cutoff_time):
    """
    This function provides a controlled method to increase probability of
    sustaining a heat wave until 'time_to_peak' after this time, the heat
    wave sustaining probability drops. Pmax is the maximum <= 1.0 and 
    P0 is the initial probability. There are therefore 4 parameters 
    for this fit including a cutoff time at which probability drops to zero.
    
    """
    one = 1.0
    two = 2.0
    zero = 0.0
    t_dimensionless = time_in_state / time_to_peak
    
    if time_in_state > cutoff_time:
        val = zero
    else:
        # no multiplication of P0 here because that is done in
        # evaluate_decay_function
        val = (one + (t_dimensionless)**two * exp(two) * 
                    (Pmax/P0 - 1)* exp(-two * t_dimensionless))
        if time_in_state >= 2 * time_to_peak:
            val = val * exp(-exp(1)**(-2) * abs(Pmax/P0-1) * (t_dimensionless-2)**2)
    
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
        raise ValueError("func_type must be 0,1,2,3,4, or 5...upredictable behavior is resulting!")
        
    cdf1[0] = cdf0[0] + P0 * func_eval
    cdf1[idy] = one - P0 * func_eval
    
    return cdf1

def markov_chain_time_dependent_py(cdf, 
                                rand, 
                                state0,
                                coef,
                                func_type,
                                check_input=False):
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
           
           for func_type 0
               coef is 1 element = lambda for exp(-lambda * t)
               
           1: coef has 1 element = slope for slope * t
           2: coef has 2 elements = lambda and a cutoff time
           3: coef has 2 elements = slope and a cutoff time
           4: coef has 3 elelments = 1) time_to_peak, 2) Peak Maximum Probability
                                     and 3) cutoff time at which probability drops
                                     to zero.
           different numbers of coefficients are needed if 'func_type' has 
           different decay functions for heat waves versus cold snaps.
           
                            
           
    func_type : np.array of integers that indicate what function type to use. 
           
           0 - use exponential_decay
           1 - use linear_decay
           2 - use exponential_decay with cut-off point
           3 - use linear_decay with cut-off point
           4 - use quadratic time exponential that peaks at a specific time and
               then decays. This function increases probability of sustaining
               a heat wave and then decays after the peak

          
               
    
    
    
    Returns
    -------
    
    """
    if check_input:
        state0,func_type = cython_function_input_checks(cdf, 
                                        rand, 
                                        state0,
                                        coef,
                                        func_type)
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
    # assign first value the initial value.
    yy[0] = state0
    step_in_cur_state = 0
    
    for idx in range(1, num_step):
        
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
                    cdf_local = evaluate_decay_function(cdf[state0,:],
                                                        func_type[idy-1],
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
