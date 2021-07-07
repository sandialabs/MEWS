# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:20:54 2021

Copyright Notice
=================

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
from numpy import zeros

class MarkovPy():
    @staticmethod
    def markov_chain_py(cdf, rand, state0):
        """
        markov_chain_py(cdf, rand, state0)
        
        evaluate a chain of markov states based on cumulative probability matrix,
        random numbers and initial state
        
        Parameters
        ----------
        cfd : np.ndarray : 
            must be a 2-D square matrix whose elements are the 
            cumulative sum of a markov right stochastic matrix
        rand : np.ndarray : 
            1-D array of random probabilities 0 <= p <= 1
        state0 : int, np.integer :
            Integer indicating the initial markov state (determines row of cfd to
            start on). 0 <= state0 <= cfd.shape[0]
        
        This function does not have Type and Value protections and should only
        be used in the context of such checks having already occured!
        """
        
        # assign initial values
        num_step = len(rand)
        yy = zeros(num_step)
        
        num_state = cdf.shape[0]
        
        
        for idx in range(num_step):
            for idy in range(num_state):
                if cdf[state0,idy] > rand[idx]:
                    yy[idx] = idy
                    state0 = idy
                    break
                if idy == num_state-1:
                    yy[idx] = idy
        
        return yy
    
    