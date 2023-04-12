# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:22:19 2021

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

from mews.weather import Alter
from mews.cython.markov import markov_chain
from mews.stats.markov import MarkovPy
from mews.stats.markov_time_dependent import (markov_chain_time_dependent_wrapper,
                                              markov_chain_time_dependent_py)
from mews.constants.data_format import (INVALID_VALUE, WAVE_MAP, DEFAULT_NONE_DECAY_FUNC,
                                       DECAY_FUNC_TYPES, ABREV_WAVE_NAMES)
from mews.errors.exceptions import ExtremesIntegrationError
from numpy.random  import default_rng, seed
from scipy.optimize import curve_fit, minimize
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from mews.utilities.utilities import filter_cpu_count
from mews.utilities.utilities import create_output_weather_file_name
from mews.utilities.utilities import find_extreme_intervals
from copy import deepcopy
from warnings import warn


def _transform_fit_sup(value,maxval,minval):
    return 2 * (value - minval)/(maxval - minval) - 1

def _inverse_transform_fit(norm_signal, signal_max, signal_min):
    return (norm_signal + 1)*(signal_max - signal_min)/2.0 + signal_min 

class DiscreteMarkov():
    """
    >>> obj = MarkovChain(transition_matrix,
                          state_names,
                          use_cython=True)
    
    Object to work with discrete Markov chains.
    
    Parameters
    ----------
    rng : numpy.random.default_rng()
        Random number generator
        
    transition_matrix : np.ndarray of np.float
        Square matrix that is a right stochastic matrix (i.e. the rows 
        sum to 1).
        
    state_names : list of strings : optional : Default = None
        list of names that must be of the same dimension as the dimension
        as the transition_matrix. The default is None and a list of 
        ["0","1",...] is assigned.
        
    use_cython : bool : optional : Default = True
        Use cython to calculate markov chain histories (~20x faster). 
        The default is True.
        
    decay_func_type: str : optional : Default = None
        If None - no decay in markov process will be included.
        
        Otherwise, this string must be one of the following. 
        
        1: "exponential"
        2: "linear"
        3: "exponential_cutoff"
        4: "linear_cutoff"
        5: "quadratic_times_exponential_decay_with_cutoff"
        
        see the "coef" input description to see how each of these
        options work.
        
    coef : np.ndarray : optional : Default = None
        This controls if a markov process with time dependent decay of the
        extreme event probability of continuing is used. The form depends on the
        decay_func_type input.
        
        decay_func_type:
            
            "exponential" need input like: np.array([[lamb_cs],[lamb_hw]])
            where lamb_hw is the decay rate for heat waves and lamb_cs
            is the decay rate for cold snaps.
            lamb is an exponential decay coefficient. The value must be 
            positive and smaller values decay slower.
            
            Probability decays according to 1 - exp(-lamb*time_in_extreme_wave)
            
            "linear" need input like: np.array([[slope_cs],[slope_hw]])
            
            Probability decays according to 1 - slope * time_in_extreme_wave
            and does not go below zero.
            
            "exponential_cutoff" need input like np.array([[lamb_cs,delt_cutoff_cs],
                                                          [lamb_hw,delt_cutoff_hw]])
            
            same behavior as exponential but with a cutoff where delt_cutoff_hw is the
            number of time steps in a heat wave at which the heat wave is forced to end.
            Currently the algorith allows +2 time steps beyond this cutoff. This is a known
            issue.
            
            "linear_cutoff" needs input like np.array([[slope_cs,delt_cutoff_cs],
                                                          [slope_hw,delt_cutoff_hw]])
            
            ""quadratic_times_exponential_decay_with_cutoff"" needs input like
            
            np.array([[time_to_max_prob_cs,delt_cutoff_cs,max_probability_cs],
                      [time_to_max_prob_hw,delt_cutoff_hw,max_probability_hw]])
            
            ""

    Returns
    -------
    None

    """
    # this list must be updated if new function types are added to the markov_time_dependent_wrapper functions
    _func_types = DECAY_FUNC_TYPES
    # DO NOT CHANGE THE ORDER OF THIS LIST! 
    _events = ABREV_WAVE_NAMES
    
    def __init__(self,rng,
                      transition_matrix,
                      state_names=None,
                      use_cython=True,
                      decay_func_type=DEFAULT_NONE_DECAY_FUNC,
                      coef=DEFAULT_NONE_DECAY_FUNC):
        if not decay_func_type is None:
           for wt in self._events:
               if not decay_func_type[wt] is None and not decay_func_type[wt] in self._func_types:
                   raise ValueError("The decay_func_type['{0}'] input must be a string equal to one of the following: \n\n".format(wt) + str(self._func_types.keys()))
               if not decay_func_type[wt] is None and coef[wt] is None:
                    raise ValueError("If a 'decay_func_type' is provided, then coefficients input 'coef' is needed.")
        
        
        self.rng = rng
        # special handling
        if state_names is None:
            try:
                state_names = [str(idx) for idx in range(transition_matrix.shape[0])]
            except:
                raise Exception("Something is wrong with the transition_matrix"
                                +". It must be a right stochastic"
                                +" square matrix!")
        
        if not isinstance(use_cython,bool):
            raise TypeError("The use_cython input must be a boolean!")
        # type and value checks
        self._type_checks(transition_matrix, state_names)
        self._value_checks(state_names,transition_matrix)
        
        self._mat = transition_matrix
        self._cdf = transition_matrix.cumsum(axis=1)
        names = {}
        for idx, name in enumerate(state_names):
            names[name] = idx
        self._names = names
        self.use_cython = use_cython
        self.tol = 1e-10
        
        # convert coefficients to a pure array rather than a dictionary
        # where extra entries are -999 if there is a mismatch in
        # decay function type.
        max_len = 0
        num_event = len(coef)
        for wt in self._events:
            cf = coef[wt]

            if cf is None:
                len_cf = 0
            else:
                len_cf = len(cf)
            max_len = np.max([max_len,len_cf])
        coef_arr = INVALID_VALUE * np.ones((num_event,max_len),dtype=float)

        for idx, wt in enumerate(self._events):
            cf = coef[wt]
            if not cf is None:
                coef_arr[idx,0:len(cf)] = cf
        
        self.coef = coef_arr
        self.decay_func_type = decay_func_type
        
    def history(self,num_step,state0,min_steps=24,skip_steps=0,count_in_min_steps_intervals=False):
        """
        >>> obj.history(num_step,state0, 
                        min_steps, 
                        skip_steps)
        
        Calculate a history for a discrete markov chain of 'num_step' length
        and initial 'state0'.
        
        Parameters
        ----------
        num_step : int or np.int
            Number of steps to simulate for the object's defined transition_matrix.
        state0 : int or str
            Initial state of the history.
        min_steps : int : optional
            Minimum number of steps before a state can transition.
        skip_steps : int
            Number of steps at the beginning of the markov chain to assign to
            state0 (this is used for weather files that may begin in the middle
            of the day to avoid applying heat waves out of sync with
            day-night cycles).
        count_in_min_steps_intervals : bool : optional : Default = False
            True = apply markov transition matrix as a discrete process that 
            is randomly tested every min_steps interval (i.e. min_steps = 24
            means the markov chain state is only tested 1 in 24 steps).
            False = apply markov transition matrix every step but only transition
            at every min_steps step.
            
        Raises
        ------
        ValueError
            Error in input for num_step or state0. State0 must be a value less
            than the dimension-1 of the transition_matrix or a string that is 
            in the self._names dictionary.
        TypeError
            All incorrect types are rejected from this method.

        Returns
        -------
        None

        """
        # input checks
        if isinstance(state0,str):
            if not state0 in self._names:
                raise ValueError("The 'state0' input must be a valid state_name={}".format(str(self._names)))
            else:
                # convert to an index
                state0 = self._names[state0]
        elif not isinstance(state0,(np.integer,int)):
            raise TypeError("The 'state0' input must be a valid string name or an integer!")
        else:
            nxn = len(self._names)
            if state0 > len(self._names):
                raise ValueError("The transition_matrix is {0:d}x{1:d} ".format(nxn,nxn)
                                 +"but a state = {2:d} was entered")
        if count_in_min_steps_intervals: 
            num_real_step = int(np.floor((num_step - skip_steps)/min_steps))
        else:
            num_real_step = num_step
        
        if num_real_step < 0:
            raise ValueError("The number of skip_steps must not be greater than the num_steps!")
        elif num_real_step == 0:
            raise ValueError("There are no real steps. The weather history"
                             +" provided is too short to sample in min_steps"
                             +"={0:d} size steps!".format(min_steps))
        
        
        if count_in_min_steps_intervals:
            remain_steps = num_step - num_real_step * min_steps - skip_steps
        else:
            remain_steps = 0
            
        prob = self.rng.random(num_real_step)
        
        cdf = self._cdf
        
        int_func_list = []
        for wt in self._events:
            if self.decay_func_type[wt] is None:
                int_func_list.append(None)
            else:
                int_func_list.append(self._func_types[self.decay_func_type[wt]])
        
        # The Markov Chain is about 10x slower in Python
        if self.use_cython:
            if self.decay_func_type is None or (
                    self.decay_func_type[ABREV_WAVE_NAMES[0]] is None 
                    and self.decay_func_type[ABREV_WAVE_NAMES[1]] is None):
                state = markov_chain(cdf,prob,state0)
            else:
                

                
                state = markov_chain_time_dependent_wrapper(cdf,
                                                            prob,
                                                            state0,
                                                            self.coef,
                                                            np.array(int_func_list),
                                                            check_inputs=True)
        else:
            if self.decay_func_type is None:
                state = MarkovPy.markov_chain_py(cdf,prob,state0)
            else:
                state = markov_chain_time_dependent_py(cdf,
                                                       prob,
                                                       state0,
                                                       self.coef,
                                                       np.array(int_func_list),
                                                       check_input=True)

        if count_in_min_steps_intervals:
            # translate state into an array of min_step length segments
            state_extended = np.repeat(state,min_steps)
            ending_state = np.repeat(np.array([state_extended[-1]]),remain_steps)
            begin_state = np.repeat(np.array([state0]),skip_steps)
            
            final_state = np.concatenate((begin_state,state_extended,ending_state))
        else:
            # TODO - this is slow!
            # move state changes to min_step intervalsstate_count = np.array([(day == state_val).sum() for name,state_val in self._names.items()])
            # for each day, assign the state that has the majority of hours in
            # the day, except that there must be a 24 hour pause between
            # events.
            

            big_step = int(np.floor(num_step/min_steps))
            prev_state = 0            
            nstate = deepcopy(state)
            for idx in range(big_step):
                
                day = nstate[min_steps * idx:min_steps*(idx+1)]
                state_count = np.array([(day == state_val).sum() for name,state_val in self._names.items()])
                
                new_state = state_count.argmax()
                
                if prev_state != 0 and new_state != 0 and state_count[0] > 0:
                    # There must be a 24 hour break between events regardless of 
                    # if there is a majority
                    nstate[min_steps * idx:min_steps*(idx+1)] = 0
                elif prev_state == 0 and new_state == 0 and state_count[0] != min_steps:
                    nstate[min_steps * idx:min_steps*(idx+1)] = state_count[1:].argmax() + 1
                else:
                    nstate[min_steps * idx:min_steps*(idx+1)] = state_count.argmax()
                    
                prev_state = new_state


        return nstate
            
    def steady(self):
        """
        >>> obj.steady()
        
        Calculates a stationary probability vector for each state based on the
        transition probability matrix.
        
        Parameters
        ----------
        None

        Returns
        -------
        steady : np.ndarray
            Vector of stationary probabilities of each state given the 
            transition matrix assigned to the process.

        """
        # transpose needed because eig must work with a left hand stochastic
        # matrix
        if self.decay_func_type is None or self.decay_func_type == DEFAULT_NONE_DECAY_FUNC:
            val,vec = np.linalg.eig(np.transpose(self._mat))
            steady = vec[:,0]/vec.sum(axis=0)[0]
            
            if np.abs(np.imag(steady).sum() > len(steady) * self.tol):
                raise UserWarning("The eigenvector has non-zero imaginary parts!")
            
            return np.real(steady)
        else:
            raise NotImplementedError("The 'steady' method does not handle "+
                                      "cases where time dependent decay of the"+
                                      " transition matrix is included!")
            
                
    def _type_checks(self,transition_matrix,state_names):
        if not isinstance(state_names,list):
            raise TypeError("The 'state_names' input must be a list")
        else:
            if np.array([not isinstance(name,str) for name in state_names]).any():
                raise TypeError("The 'state_names' elements must be strings!")
        
        if not isinstance(transition_matrix,np.ndarray):
            raise TypeError("The 'transition_matrix' input must be a numpy array!")
        else:
            if not np.issubdtype(transition_matrix.dtype,np.number):
                raise TypeError("The 'transition_matrix' must have elements that are numeric!")
    
    
    def _value_checks(self,state_names,transition_matrix):
        epsilon = 1e-6
        if transition_matrix.ndim != 2:
            raise ValueError("The 'transition_matrix' input must be of dimension=2!")
        
        if (transition_matrix < 0.0).any():
            raise ValueError("All entries to the transition matrix are "+
                             "probabilities and must be positive!\n\n"+str(transition_matrix))
        
        if len(state_names) != transition_matrix.shape[0]:
            raise ValueError("The number of statenames must match the size of the transition_matrix!")
        elif transition_matrix.shape[1] != transition_matrix.shape[0]:
            raise ValueError("The 'transition_matrix' input must be a square matrix!")
        
        row_sum = transition_matrix.sum(axis=1)
        if (row_sum > 1.0+epsilon).any() or (row_sum < 1.0 - epsilon).any():
            raise ValueError("The rows of the transition matrix must sum to "
                             + "one. At least one row is "
                             +"{0:5.3e} away from 1.0!".format(epsilon))
        # renormalize
        for idx,rsum in enumerate(row_sum):
            transition_matrix[idx,:] = transition_matrix[idx,:]/rsum
        

        

# This is a preliminary class for the toy model. Much more thorough review of
# best methods for heat wave generation and of using historical data as a 
# starting point for statistical distributions is needed.
class Extremes():
    """
    Fit weather data with a model for extreme hot and cold that can
    then be modified to produce increasing intensity and duration
    by adjustment of the statistical parameters involved.
    
    
    >>> obj = Extremes(start_year,
                       max_avg_dist,
                       max_avg_delta,
                       min_avg_dist,
                       min_avg_delta,
                       transition_matrix,
                       transition_matrix_delta,
                       weather_files,
                       num_realizations,
                       num_repeat=1,
                       use_cython=True,
                       column='Dry Bulb Temperature',
                       tzname=None,
                       write_results=True,
                       results_folder="",
                       results_append_to_name="",
                       run_parallel=True,
                       min_steps=24,
                       test_shape_func=False,
                       doe2_input=None,
                       random_seed=None,
                       max_E_dist=None,
                       del_max_E_dist=None,
                       min_E_dist=None,
                       del_min_E_dist=None,
                       frac_long_wave=0.8,
                       current_year=None,
                       climate_temp_func=None,
                       averaging_steps=1,
                       no_climate_trend=False,
                       use_global=True,
                       baseline_year=2014)
    
    look for results in obj.results - files are output as specified in the inputs.
    
    Parameters
    ----------
    start_year : int
        Year to begin a new history that will replace other years in 
        weather files.
    
    max_avg_dist : function or dict
        If function: 
            High extreme (for temperature extreme heat) statistical
            distribution function that returns random numbers that indicate
            how much energy/hr is associated with an extreme event.
        If dict: 
            A dictionary of functions where each...
        
    max_avg_delta : float or dict
        If float:
            Gradient of change for the average value of 
            max_avg_dist per year. Change is included step-wise on a yearly
            basis.
        If dict:
            Contains elements 'func' and 'param' where 'param' is a dictionary
            of deltas on each parameter input to the function in 'func'
            set of all monthly parameters needed by the max_avg_dist 
            function. The number of parameters is distribution dependent.
        
    min_avg_dist : function or dict
        If function:
            A CDF that recieves one input.
            
            Low extreme (for temperature extreme cold) statistical distribution
            function that returns random numbers that indicate how much 
            reduction in energy/hr is associated with an extreme event.
        
        If dict:
            ...
        
    min_avg_delta : float
        Gradient of change for the average value of min_avg_dist per year.
        Change is included step-wise on a yearly basis
        
    transition_matrix : np.array
        Markov chain 3x3 matrix where index 0 = probability of "normal 
        conditions", 1 = probability of maximum extreme conditions, and
        2 = probability of minimum extreme conditions. Rows number is associated
        with the probabilities for a given state (i.e. row 0 applies when the
        the current state is 0). Rows must sum to 1.
        
    transition_matrix_delta : np.array
        Rate per year that transition matrix probabilities are changing in
        absolute (not percentage) change in probability. Rows must sum to zero so that the
        transition matrix always has rows continue to sum to 1.
        
    weather_files : list 
        Weather files that will be read and altered per the statistical specs above.
        
    num_realizations : int
        The number of realizations per weather file that will be output for the
        statistics. Number of output weather files is num_realizations * len(weather_files).
        
    num_repeat : int : optional : Default = 1
        DEPRICATED - LEAVE = 1 Number of repeat years that will be processed
        into a longer duration than the original weather file.
        
    use_cython : bool : optional : Default = True
        Flag to enable cython. If False, then much slower native python
        is used for the Markov chain. 
        
    column : str : optional : Default = 'Dry Bulb Temperature'
        Column for weather signal that will be altered. This input needs
        to be different for a DOE2 weather file.
        
    tzname : str : optional : Default = None
        Time zone name to apply the weather files to. Must be a valid
        time zone for the tz Python library.
        
    write_results : bool : optional : Default = True
        Flag to either write (True) or not write the weather files.
        results can be accessed through obj.results if not written.
        
    results_folder : str : optional : Default = ""
        String of a path to a location to write weather files.
    
    results_append_to_name : str : optional : Default = ""
        String to append to the file names of output weather files.
        
    run_parallel : bool : optional : Default = True
        Flag to either run parallel (True) or run on a single processor (False)
        set to False if parallel run engine fails.
        
    min_steps : int > 0 : optional : Default=24
        Minimum number of time steps for an extreme event .
        
    test_shape_func : bool : optional : Default = False
        For testing purposes:
            Boolean indicating whether computationally expensive testing is 
            done to assure the shape function used integrates to the original
            heat_added. Default value is False - True is needed for unittesting.
        
    doe2_input : dict : optional : Default = None
       | Optional input required to perform the analysis using DOE2
       | bin files. See mews.weather.alter. needs:
       | {'doe2_bin2txt_path':OPTIONAL - path to bin2txt.exe DOE2 utility
       | MEWS has a default location for this utility 
       | which can be obtained from DOE2 (www.doe2.com),
       | 'doe2_start_datetime':datetime indicating the start date and time
       | for the weather DOE2 weather file,
       | 'doe2_tz'=time zone for doe2 file,
       | 'doe2_hour_in_file'=8760 or 8784 for leap year,
       | 'doe2_dst'= Start and end of daylight savings time for the 
       | doe2_start_datetime year,
       | 'txt2bin_exepath' : OPTIONAL - path to txt2bin.exe DOE2 utility}
    
    random_seed : int : optional : Default = None
        Use this to fix an analysis so that pseudo-random numbers are sampled
        as the same sequence so that results can be replicated.
        
     max_E_dist : dict : optional : Default = None
         A distribution that defines random sampling for energy of an 
         extreme wave. If this is None, then the old way of using
         Extreme applies if it is a dictionary, then it must be a CDF function
         that allows random numbers to be used and has parameters as independent
         inputs so that they can be shifted via the del_max_E_dist input.
         
    del_max_E_dist : dict : optional : Default = None
        For every parameter in max_E_dist, a del_param is needed that shows
        how the max_E_dist is stretched/shifted with changes in climate.
        
    frac_long_wave : float <= 1 > 0 
        Old input style feature. The amount of energy that goes into a sinusoidal peak. 
        
    current_year : int : optional 
        New input style feature. Used to add a global warming offset to the entire dataset.
     
     no_climate_trend : Bool : Optional : Default = False
          Exclude the climate trend gradual increase in temperature
          while including the heat wave effects.
          
    use_global : bool : optional : Default = True
         Controls whether to use the old use case where MEWS calculates
         from global surface temperature averages or to use the new case where
         MEWS uses CMIP6 data from a specific lat/lon.
         All new use shoudl keep this False but it is defaulted to True
         so that old scripts don't have to change.'
         
    baseline_year: numeric : optional : Default = 2014
        Only used with use_global=True
        
        This is the CMIP6 baseline_year. It should always stay equal to 2014.
        
    num_cpu : int : optional : Default = None
        None : use the number of cpu's available minus one
        int : use num_cpu if it is <= number of cpu's available minus one
        
    test_markov : bool : optional : Default = False
        Set to true if you want to override raised errors due to only running
        a small time frame. This is for test purposes only. MEWS ussually
        needs 50+ years of run to create distributions on the markov process
        
    confidence_interval : str : optional : Default = ""
        String to add to the output files for the IPCC confidence interval
        selected. If this is left blank, then files will overwrite eachother if
        more than one confidence interval is being calculated.
        
    overwrite_existing : bool : optional : Default = True
        Indicate whether to overwrite an existing file
    
    Returns
    -------
    None 
    """
    
    """
    TODO finish documentation
     ,
     min_E_dist=None - optional, cold snap analog to max_E_dist
     del_min_E_dist - optional, cold snap analog to del_max_E_dist
     climate_temp_func
     averaging_steps (see alter object)
    """
    states = {"normal":0,"cold":1,"hot":2}
    max_recursion = 100
    
    def __init__(self,start_year,
                 max_avg_dist,
                 max_avg_delta,
                 min_avg_dist,
                 min_avg_delta,
                 transition_matrix,
                 transition_matrix_delta,
                 weather_files,
                 num_realizations,
                 num_repeat=1,
                 use_cython=True,
                 column='Dry Bulb Temperature',
                 tzname=None,
                 write_results=True,
                 results_folder="",
                 results_append_to_name="",
                 run_parallel=True,
                 min_steps=24,
                 test_shape_func=False,
                 doe2_input=None,
                 random_seed=None,
                 max_E_dist=None,
                 del_max_E_dist=None,
                 min_E_dist=None,
                 del_min_E_dist=None,
                 frac_long_wave=0.8,
                 current_year=None,
                 climate_temp_func=None,
                 averaging_steps=1,
                 no_climate_trend=False,
                 use_global=True,
                 baseline_year=2014,
                 norms_hourly=None,
                 num_cpu=None,
                 test_markov=False,
                 confidence_interval="",
                 overwrite_existing=True):
        
        self._num_cpu = filter_cpu_count(num_cpu)
        self.use_global = use_global
        self.baseline_year = baseline_year
        self._delTmax_verification_data = {"delTmax":[],"freq_s":[]} # used in unit testing
        self._objDM = None
        self._overwrite_existing = overwrite_existing
        
        # input checking
        if not isinstance(weather_files,list):
            raise TypeError("The 'weather_files' input must be a list of strings!")
        
        self.wfile_names = []
        # perform some input checking
        new_input_format = self._monthly_input_checking([max_avg_dist, 
                                      max_avg_delta, 
                                      min_avg_dist, 
                                      min_avg_delta, 
                                      transition_matrix, 
                                      transition_matrix_delta, 
                                      max_E_dist, 
                                      del_max_E_dist, 
                                      min_E_dist, 
                                      del_min_E_dist])
        


        
        # set the random seed - only for non-parallel application!
        if not random_seed is None:
            self.rng = default_rng(seed=random_seed)
        else:
            self.rng = default_rng()
        
        # turn off parallel for the DOE2 case or setup parallel processing otherwise.
        if run_parallel and doe2_input is None:
            
            try:
                import multiprocessing as mp
                pool = mp.Pool(self._num_cpu)
            except:
                warnings.warn("Something went wrong with importing "
                            +"multiprocessing and creating a pool "
                            +"of asynchronous processes reverting to"
                            +" non-parallel run!",UserWarning)
                run_parallel = False
        elif run_parallel and not doe2_input is None:
            run_parallel = False
            warnings.warn("The program does not handle DOE2 files with"+
                              " multiple processors run_parallel set to 'False'",
                              UserWarning)
        
        # Constants that should not be changed.
        state_int = [1,2] # no changes to non-selected intervals
        state_name = [name for name in self.states.keys()]
        num_wfile = len(weather_files)
        self.recursive_depth = 0
        
        results = {}
        objDM = {}
        for id0 in range(num_realizations):
            for idx in range(num_repeat):
                for idy,wfile in enumerate(weather_files):
                     
                    if new_input_format:
                        year = current_year
                        
                    else:
                        year = start_year + idx * num_wfile + idy
                        
                    key_name = (wfile, id0, year)
                    if run_parallel:
                        if random_seed is None:
                            rng = default_rng()
                        else: 
                            rng = default_rng(seed=(random_seed + int(1010 * self.rng.random(1))))
                        args = (start_year,idx,num_wfile,idy,wfile,column,
                                tzname,transition_matrix,transition_matrix_delta,
                                state_name,use_cython,state_int, 
                                min_avg_dist, min_avg_delta, 
                                max_avg_dist, max_avg_delta, results_folder, 
                                results_append_to_name, id0, write_results,year,
                                frac_long_wave,min_steps,test_shape_func,doe2_input,
                                max_E_dist,del_max_E_dist,min_E_dist,del_min_E_dist,
                                new_input_format,climate_temp_func,averaging_steps,rng,
                                no_climate_trend,norms_hourly,key_name,test_markov,confidence_interval)
                        results[key_name] = pool.apply_async(self._process_wfile,
                                                         args=args)
                    else:
                        (results[key_name],objDM[key_name]) = self._process_wfile(start_year,idx,
                                num_wfile,idy,wfile,column,
                                tzname,transition_matrix,transition_matrix_delta,
                                state_name,use_cython,state_int,
                                min_avg_dist, min_avg_delta, 
                                max_avg_dist, max_avg_delta, results_folder, 
                                results_append_to_name, id0, write_results, year,
                                frac_long_wave,min_steps,test_shape_func,doe2_input,
                                max_E_dist,del_max_E_dist,min_E_dist,del_min_E_dist,
                                new_input_format,climate_temp_func,averaging_steps,
                                self.rng,no_climate_trend,norms_hourly,key_name,
                                test_markov,confidence_interval)
        
        if run_parallel:
            results_get = {}
            for tup,poolObj in results.items():
                try:
                    (results_get[tup],objDM[tup]) = poolObj.get()
                except AttributeError:
                    raise AttributeError("The multiprocessing module will not"
                                         +" handle lambda functions or any"
                                         +" other locally defined functions!")
               
            pool.close()
            self.results = results_get
            self._objDM = objDM
        else:
            self.results = results
            self._objDM = objDM
            
            
            
    
    def _monthly_input_checking(self,dict_list):
        incorrect_type = 0
        dict_found = False
        for dict_e in dict_list:
            if not isinstance(dict_e,dict) and not dict_e is None:
                incorrect_type += 1
            else:
                if not dict_e is None and len(dict_e) > 0:
                    dict_found=True
                    if 'func' in dict_e.keys():
                        dict_e2 = dict_e['param']
                    else:
                        dict_e2 = dict_e
                    
                    month_list = [1,2,3,4,5,6,7,8,9,10,11,12]
                    if list(dict_e2.keys()) != month_list:
                        raise ValueError("All dictionary inputs must have an entry for every month!" +
                                         "The following entries must be present:\n\n "+str(month_list) +
                                         "\n\nThe following are in the dict_e2 variable:\n\n " + str(dict_e2.keys()))
        
        
        if incorrect_type > 0 and incorrect_type != 6:
            raise TypeError("You must either define all inputs as dictionaires"
                            +" with 12 months as the key {1:,2:, etc..} or as"
                            +" functions and floats per the documentation")
        
        return dict_found
            
    def _alter_if_leap_year(self,df_one_year,year):
        hour_in_day = 24
        hour_in_year = 8760
        hour_in_leapyear = hour_in_year + hour_in_day
        
        feb_28 = pd.Timestamp(year,2,28)
        feb_28_day = feb_28.day_of_year
        if len(df_one_year) == hour_in_year and feb_28.is_leap_year:
            # repeat Feb. 28th and call it Feb 29th
            df_list = []

            df_feb28 = df_one_year.iloc[0:feb_28_day*hour_in_day,:]
            df_feb28.index = pd.date_range(pd.Timestamp(year,1,1),periods=feb_28_day*hour_in_day,freq='H')
            df_list.append(df_feb28)
            
            # repeat february 28th as a proxy for February 29th
            df_feb29 = df_one_year.iloc[feb_28_day*hour_in_day:(feb_28_day+1)*hour_in_day,:]
            df_feb29.index = pd.date_range(pd.Timestamp(year,2,29),periods=hour_in_day,freq='H')
            df_list.append(df_feb29)
            
            # tag on the rest of the year.
            df_after_feb29 = df_one_year.iloc[feb_28_day*hour_in_day:(len(df_one_year)+1)*hour_in_day]
            df_after_feb29.index = pd.date_range(pd.Timestamp(year,3,1),periods=hour_in_leapyear-len(df_feb28)-len(df_feb29),freq='H')
            df_list.append(df_after_feb29)
            return pd.concat(df_list,axis=0)
        elif len(df_one_year) != hour_in_year and len(df_one_year) != hour_in_leapyear:
            raise ValueError("The 'df_one_year' input must be an hourly yearly signal!")
        else:
            # no changes needed!
            return df_one_year
            
        
        
    
    
    # this function was created to make parallelization possible.
    def _process_wfile(self,start_year,idx,num_wfile,idy,wfile,column,tzname,
                       transition_matrix,transition_matrix_delta,state_name,
                       use_cython,state_int,min_avg_dist,
                       min_avg_delta,max_avg_dist,max_avg_delta,
                       results_folder, results_append_to_name,id0, 
                       write_results, year, frac_long_wave,min_steps,
                       test_shape_func,doe2_in, max_E_dist,del_max_E_dist,
                       min_E_dist,del_min_E_dist, new_input_format,
                       climate_temp_func,averaging_steps,rng, no_climate_trend,
                       norms_hourly,key_name,test_markov,confidence_interval):
        # THIS IS USUALLY IN A PARALLEL mode so debugging can be hard unless
        # you set run_parallel to False.

        new_wfile_name = create_output_weather_file_name(os.path.basename(wfile),
                                                         results_append_to_name,
                                                         year,
                                                         confidence_interval,
                                                         id0) 

        new_wfile_path = os.path.join(results_folder,new_wfile_name)
        
        # new feature to skip re-writing files that have already been written
        if ((os.path.exists(new_wfile_path) and self._overwrite_existing) or
            (not os.path.exists(new_wfile_path))):
        
            objDM_dict = {}
            
            if self.use_global == False:
                # add Feb29 if a leap year
                norms_hourly = self._alter_if_leap_year(norms_hourly,year)
            
            
            if doe2_in is None:
                objA = Alter(wfile, year)
            else:
                objA = self._DOE2Alter(wfile,year,doe2_in)
                
            org_series = objA.epwobj.dataframe[column]
            org_dates = objA.reindex_2_datetime(tzname).index
            num_step = len(objA.epwobj.dataframe.index)
    
            if new_input_format:
                state0 = "normal"
                states_arr = None
                for month, trans_matrix in transition_matrix.items():
                    month_num_steps = (org_dates.month == month).sum()
                    
                    modified_trans_matrix = trans_matrix + transition_matrix_delta[month]
                    
                    if not self.use_global:
                        coef, decay_func_type = self._coef_form(max_avg_dist['param'][month], min_avg_dist['param'][month])
                    else:
                        coef = DEFAULT_NONE_DECAY_FUNC
                        decay_func_type = DEFAULT_NONE_DECAY_FUNC
                    
                    objDM = DiscreteMarkov(rng, modified_trans_matrix, 
                                           state_name, use_cython, decay_func_type=decay_func_type,
                                           coef=coef)
                    
                    if test_markov:
                        # adjust to a longer time period so that it is nearly
                        # certain another heat wave will occur.
                        adj_num_step = 8760 # simulate an entire year
                    else:
                        adj_num_step = month_num_steps
    
                    states_arr0 = objDM.history(adj_num_step, state0,count_in_min_steps_intervals=False)    
    
                    if test_markov:
                        # this testing allows the markov process to continue beyond the extend of the current month
                        # so that the heat wave duration and time between heat waves can be quantified for an unchanging Markov process.
                        # this information is used to properly validate the frequency and duration characteristics of 10 and 50 year events that 
                        # MEWS focuses on.
    
                        state_intervals = find_extreme_intervals(states_arr0, state_int)
                        delt_between_hw = [tup1[0]-tup0[0] for tup1,tup0 in zip(state_intervals[1][1:],state_intervals[1][0:-1]) if tup0[0] <= month_num_steps]
                        delt_in_hw = [tup[1]-tup[0]+1 for tup in state_intervals[1] if tup[0] <= month_num_steps]
                        self._delTmax_verification_data["freq_s"].append({"key_name":key_name,
                                                                          "time delta between consecutive heat waves":delt_between_hw,"month":month,
                                                                          "heat wave duration":delt_in_hw})
                        states_arr0 = states_arr0[0:month_num_steps]
                        
                    if states_arr is None:
                        states_arr = states_arr0
                    else:
                        states_arr = np.concatenate((states_arr, states_arr0))
                        
                    objDM_dict[month] = objDM 
                    state0 = state_name[states_arr[-1]]
                
            else:
    
                
                objDM = DiscreteMarkov(rng,transition_matrix 
                                   + transition_matrix_delta 
                                   * (year - start_year),
                                   state_name,use_cython)
                
                objDM_dict["no months"] = objDM
                
                # generate a history
                states_arr = objDM.history(num_step,"normal",min_steps=min_steps)
                
            # distinguish the shape function type
            if new_input_format:
                shape_function_type = "heat_wave_shape_func"
            else:
                shape_function_type = "double_shape_func"
            
    
            # separate history into extreme states.
            state_intervals = find_extreme_intervals(states_arr, state_int)
            for (junk,state),s_ind in zip(state_intervals.items(),state_int):
    
                if s_ind==1:
                    is_hw = False
                    # shift cold snaps to start at noon whereas heat waves 
                    # start at mid-night
                    shifted_state = []
                    for tup in state:
                        tup_try = (int(tup[0] + min_steps/2), int(tup[1] + min_steps/2))
                        if tup_try[0] >= len(states_arr) or tup_try[1] > len(states_arr): # go back
                            tup_try = (int(tup[0]-min_steps/2),int(tup[1]-min_steps/2))
                        shifted_state.append(tup_try)
                    state = shifted_state
    
    
                    if new_input_format:
                        avg_dist = min_E_dist 
                        peak_dist = min_avg_dist  # minimum temperature
                        peak_delta = min_avg_delta
                        avg_delta = del_min_E_dist
                    else:
                        avg_dist = min_avg_dist
                        peak_dist = None
                        peak_delta = None
                        avg_delta = (year - start_year) * min_avg_delta
                else:
                    is_hw = True
                    # heat waves for s_ind==2
                    wave_shift = 0.0
                    if new_input_format:
                        avg_dist = max_E_dist
                        peak_dist = max_avg_dist
                        peak_delta = max_avg_delta
                        avg_delta = del_max_E_dist
                    else:    
                        avg_dist = max_avg_dist
                        peak_dist = None
                        peak_delta = None
                        avg_delta = (year - start_year) * max_avg_delta
    
                for tup in state:
                    if self.use_global == False:
                        local_norms_hourly = norms_hourly.iloc[tup[0]:tup[1]+1,:]
                    else:
                        local_norms_hourly = None
                    
                    new_vals,heat_added_0,delT_max_0,norm_temp, norm_dur = self._add_extreme(org_series.iloc[tup[0]:tup[1]+1], 
                                      avg_dist, avg_delta,
                                      frac_long_wave=frac_long_wave,
                                      min_steps=min_steps,
                                      shape_func_type=shape_function_type,
                                      test_shape_func=test_shape_func,
                                      peak_dist=peak_dist,
                                      peak_delta=peak_delta,
                                      org_dates=org_dates[tup[0]:tup[1]+1],
                                      rng=rng,
                                      norms_hourly=local_norms_hourly,
                                      is_hw=is_hw,
                                      key_name=key_name)
    
                    new_date_start = org_dates[new_vals.index[0]]
                    duration = len(new_vals)
                    
                    if s_ind == 1:
                        peak_delta_val = new_vals.min()
                    else:
                        peak_delta_val = new_vals.max()
                        
                    alt_name = objA.add_alteration(year, 
                                        new_date_start.day, 
                                        new_date_start.month, 
                                        new_date_start.hour, 
                                        duration, peak_delta_val,
                                        shape_func=new_vals.values,
                                        column=column,
                                        averaging_steps=averaging_steps)
                    
                    # this is used to figure out what the actual delT is for each
                    # heat wave as sampled before renormalizing to the climate normals.
                    objA._unit_test_data[alt_name] = (heat_added_0,delT_max_0,norm_temp,norm_dur)
                    
            if not climate_temp_func is None and not no_climate_trend:
                # TODO - add the ability to continuously change the trend or to
                #        just make it constant like it is now.
                if self.use_global:
                    ctf = climate_temp_func(year)
                else:
                    ctf = climate_temp_func(year - self.baseline_year)
                objA.add_alteration(year,1,1,1,num_step,ctf,np.ones(num_step),
                                    alteration_name="global temperature trend")
                # added 4/5/2023
                self._add_to_ground_temperatures(ctf,objA)
                    
                    
            if write_results:
                # TODO, this writing needs to be standardized accross all
                # alteration combinations!
                if doe2_in is None:
                    simple_call = True
                elif isinstance(doe2_in,dict) and not 'txt2bin_exepath' in doe2_in:
                    simple_call = True
                else:
                    simple_call = False 
                
                if simple_call:
                    objA.write(new_wfile_path, 
                                  overwrite=True, create_dir=True)
                else:
                    objA.write(new_wfile_path, 
                                  overwrite=True, create_dir=True,
                                  txt2bin_exepath=doe2_in['txt2bin_exepath'])
                self.wfile_names.append(new_wfile_name)
                
            return objA,objDM_dict
        else:
            return None,None
    
    
    def _add_to_ground_temperatures(self, air_temp_change, alter_obj):
        """
        
        See https://bigladdersoftware.com/epx/docs/8-2/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html
        
        for specification of the ground temperatures field
        
        There can be several depths of ground temperature measurements.
        
        We assume all of them elevate by the rise in average air temperature.
        
        This is a broad assumption that could be a function of many issues
        
        Including land-coverage changes. Changes to cloud cover etc....
        
        
        """
        if "epwobj" in dir(alter_obj):
            # assure case insensitivity
            capitalized_headers = {header.upper():header for header in alter_obj.epwobj.headers}
            if "GROUND TEMPERATURES" in capitalized_headers:
                ground_temps = alter_obj.epwobj.headers[
                        capitalized_headers["GROUND TEMPERATURES"]]

                new_ground_temps = self._ground_temp_data_dictionary_unfold(ground_temps,air_temp_change)

                alter_obj.epwobj.headers[capitalized_headers["GROUND TEMPERATURES"]] = new_ground_temps
                
                msg = ("This epw file's dry bulb and ground temperatures have been altered "+
                "by the multi-scenario extreme weather simulator https://github.com/sandialabs/MEWS")
               
                if "COMMENTS 2" in capitalized_headers:
                    alter_obj.epwobj.headers[capitalized_headers["COMMENTS 2"]].append(
                        msg)
                else:
                    alter_obj.epwobj.headers["COMMENTS 2"] = msg
                
            else:
                warn("Ground temperatures were not present in the epw headers and have not been changed!")
        else:
            raise AttributeError("The Extreme class object does not yet have an epwobj read in!")
        pass
    
    def _ground_temp_data_dictionary_unfold(self,ground_temps,air_temp_change):
        idl = 0

        new_ground_temps = []
        new_ground_temps.append(ground_temps[0])
        for idx, gtemp in enumerate(ground_temps[1:]):

            if idx > 0 and idl > 3:
                new_ground_temps.append("{0:5.2f}".format(float(gtemp) + air_temp_change))   
            else:
                new_ground_temps.append(gtemp)
            if np.mod(idx+1,16)==0:
                idl = 0
            else:
                idl += 1
                
        return new_ground_temps
    
    def _DOE2Alter(self,wfile,year,doe2_in):
        if not isinstance(doe2_in,dict):
            raise TypeError("The doe2 input must be a dictionary!")

        if 'doe2_bin2txt_path' in doe2_in:
            doe2_bin2txt_path=doe2_in['doe2_bin2txt_path']
            objA = Alter(wfile,year,
                         isdoe2=True,
                         doe2_bin2txt_path=doe2_bin2txt_path,
                         doe2_start_datetime=doe2_in['doe2_start_datetime'],
                         doe2_hour_in_file=doe2_in['doe2_hour_in_file'],
                         doe2_tz=doe2_in['doe2_tz'],
                         doe2_dst=doe2_in['doe2_dst'])
        else:
            objA = Alter(wfile,year,
                         isdoe2=True,
                         doe2_start_datetime=doe2_in['doe2_start_datetime'],
                         doe2_hour_in_file=doe2_in['doe2_hour_in_file'],
                         doe2_tz=doe2_in['doe2_tz'],
                         doe2_dst=doe2_in['doe2_dst'])
        
        return objA



    @staticmethod
    def double_shape_func(t,A,B,D,min_s):
        return A * np.sin(t * np.pi/D) + B * (1.0-np.cos(t * np.pi / (0.5*min_s)))

    @staticmethod
    def heat_wave_shape_func(t,A,B,D,D_odd,min_s):
        
        return np.where(t>D_odd,
                 B*(1-np.cos(2*np.pi*t/min_s)),
                 A*np.sin(np.pi * t / D_odd) + B * (1-np.cos(2*np.pi * t/min_s)))
    @staticmethod
    def total_energy_integral(A,B,D,D_odd,min_s,is_hw):
        if is_hw:
            A = np.abs(A)
            B = np.abs(B)
        else:
            A = -np.abs(A)
            B = -np.abs(B)
        return 2 * A * D_odd/np.pi + B * D - B * min_s / (
            2 * np.pi) * np.sin(2 * np.pi * D / min_s)
        
    
    def _add_extreme(self,org_vals,integral_dist,integral_delta,frac_long_wave,min_steps,
                     shape_func_type=None, test_shape_func=False, peak_dist=None, peak_delta=None,
                     org_dates=None,rng=None,norms_hourly=None,is_hw=True,key_name=None):
        """
        >>> obj._add_extreme(org_vals,
                             integral_dist,
                             integral_delta,
                             frac_long_wave=1.0,
                             min_steps,
                             shape_func_type=None,
                             test_shape_func=False,
                             peak_dist=None,
                             peak_delta=None,
                             norms_hourly=None,
                             is_hw=True,
                             key_name=None)

        Parameters
        ----------
        org_vals : pandas.Series
            Original values for weather over the duration of an inserted 
            extreme event.
            
        integral_dist : function
            Random number generating function for the integral of total energy
            over the duration of the extreme event (returns total energy).
            
        integral_delta : float
            Delta amount to energy intensity of extreme event (due to climate
            change).
            
        frac_long_wave : float 1 > frac_long_wave > 0
            Amount of heat added that is attributed to a sinusoidal function 
            accross the entire duration of the heat wave. The remainder is 
            applied to increases in daily heat.
            
        min_steps : int > 0
            Minimum number of steps heat waves must begin and end on (e.g. if
            min_steps=24, then a heat wave can only begin and end at the 
            beginning of a new day).
            
        shape_func_type : str : optional : Default = None
            The name of the function that is to be used. If a new shape
            is desired, then a new function has to be hard coded into extremes
            TODO - generalize this to any shape function input.
            
        test_shape_func : bool : optional  
            For testing purposes:
                Boolean indicating whether computationally expensive testing is 
                done to assure the shape function used integrates to the original
                heat_added. Default value is False - True is needed for unittesting.

        peak_dist : function : optional 
            Only for new input structure:
                Random number generating function for the peak temperature per duration.
            
        peak_delta : dict : optional
            Only for new input structure:
                Delta on each parameter of peak_dist due to climate change.
            
        org_dates : pd.Series : optional 
            Only for new input structure:
                Allows calculation of which months the heat wave start date began in.
            
        rng : optional : Default = None
            Random number generator. If None, then use self.rng
            
        norms_hourly : optional : Default = None
            These are the climate norms from NOAA that are the reference for
            how much heat is being added via a heat wave. This is required 
            on new new use case use_global=False
            
        is_hw : optional : Boolean : Default = True
            If false, a cold snap is being added 
            If true , a heat wave is being added.
            
        key_name : optional : tuple : Default = None
            Passes on the tuple of realization number, weather file and year
            for identifying the exact placement of a heat wave in validation data

        Returns
        -------
        new_vals :  
            Values to add to org_vals in the weather history to produce a new
            extreme event.
        """        
        duration = len(org_vals)
        
        if rng is None:
            rng = self.rng
            
        
        # these are determined by taking the integral of "double_shape_func" from
        # 0 to D and equating it to heat_added times frac_long_wave for the sine term
        # and to (1-frac_long_wave) for the 1-cos term. 
        if shape_func_type is None or shape_func_type == "double_shape_func":
            # this is the old method code - eventually we want to depricate this
            # but it is retained so that MEWS can replicate earlier study results.
            
            # heat added per hour duration of the heat wave
            heat_added = (integral_dist(None)+integral_delta) * duration
            term1 = np.pi / (0.5 * min_steps)
            Acoef = frac_long_wave * heat_added * np.pi / (2.0 * duration)
            Bcoef = term1*(1.0 - frac_long_wave) * heat_added / (
                duration*term1-np.sin(term1*duration))
            
            # for unit testing, assure the shape function used integrates 
            # correctly
            if test_shape_func:
                acceptable_error_percent = 0.1
                # 100 steps per day.
                xval = np.array(np.arange(0,duration,duration/(24*100)))

                h_test = Extremes.double_shape_func(xval,Acoef,Bcoef,duration,min_steps)
                h_test_heat = np.trapz(h_test,xval)
                
                if 100 * (h_test_heat - heat_added)/heat_added > acceptable_error_percent:
                    raise ExtremesIntegrationError("The heat wave increased "
                                               +"temperature did not integrate to"
                                               +" the heat_added. This should"
                                               +" never happen. There is a bug.")
            
            h_t = Extremes.double_shape_func(np.array(range(duration)),Acoef,Bcoef,duration,min_steps)
            # renormalize to make hourly approximation exactly fit heat_added
            h_t = heat_added * h_t / h_t.sum()
            
            new_vals = pd.Series(h_t,index=org_vals.index,name=org_vals.name,dtype=org_vals.dtype)
            
            # these are not needed by this use case but are required in the output of the
            # more recent use case.
            heat_added_0 = None
            delT_max_0 = None
            norm_temperature = None
            norm_duration = None
            
        elif shape_func_type == "heat_wave_shape_func":
            # THIS IS HIGHLY REPETITIVE - perhaps some code refactoring would be appropriate
            # the month of the start date determines the sampling parameters

            # statistics come from the month with he majority of hours in.
            s_month = org_dates.month.unique()[np.array([(org_dates.month == 
                        month).sum() for month in org_dates.month.unique()]
                                                        ).argmax()]
            
            if is_hw:
                wstr = ABREV_WAVE_NAMES[1]
            else:
                wstr = ABREV_WAVE_NAMES[0]
            
            param = integral_dist['param'][s_month]
            
            # trunc_norm_dist(rnd,mu,sig,a,b,minval,maxval)
            mu_E = param['energy_normal_param']['mu']
            sig_E = param['energy_normal_param']['sig']
            maxval_E = param['max energy per duration']
            minval_E = param['min energy per duration']
            norm_energy = param['normalizing energy']
            norm_duration = param['normalizing duration']
            alpha_E = param['energy linear slope']
            mu_T = param['extreme_temp_normal_param']['mu']
            sig_T = param['extreme_temp_normal_param']['sig']
            maxval_T = param['max extreme temp per duration']
            minval_T = param['min extreme temp per duration']
            norm_temperature = param['normalizing extreme temp']
            alpha_T = param['normalized extreme temp duration fit slope']
            beta_T = param['normalized extreme temp duration fit intercept']
            
            # changes to the -1..1 boundary of the distributions.
            if self.use_global:
                Sm1_E = 0.0
                Sm1_T = 0.0
                S_E = 0.0
                S_T = 0.0
            else:
                # remember we are looking for the difference from the -1..1 interval
                # so -(-1) on the minus 1 (m1) entries and -1 on the positive side entries S_E and S_T
                Sm1_E = _transform_fit_sup(param['min energy per duration'],
                                           param['hist max energy per duration'],
                                           param['hist min energy per duration']) + 1
                Sm1_T = _transform_fit_sup(param['min extreme temp per duration'],
                                           param['hist max extreme temp per duration'],
                                           param['hist min extreme temp per duration']) + 1
                S_E = _transform_fit_sup(param['max energy per duration'],
                                           param['hist max energy per duration'],
                                           param['hist min energy per duration']) - 1
                S_T = _transform_fit_sup(param['max extreme temp per duration'],
                                           param['hist max extreme temp per duration'],
                                           param['hist min extreme temp per duration']) - 1 
                
            abs_maxval_delT = param['normalizing extreme temp']

            # introduce a delta if it exists.  - this feature is not used
            # by the new analysis which just changes the actual param values
            # rather than supplying deltas.
            if self.use_global:
                if not integral_delta is None:
                    param_delta = integral_delta[s_month]
                    mu_E += param_delta['del_mu']
                    sig_E += param_delta['del_sig']
                    Sm1_E += param_delta['del_a']
                    S_E += param_delta['del_b']
                if not peak_delta is None:
                    peak_param_delta = peak_delta[s_month]
                    mu_T += peak_param_delta['del_mu']
                    sig_T += peak_param_delta['del_sig']
                    Sm1_T += peak_param_delta['del_a']
                    S_T += peak_param_delta['del_b']   
                    # Temperature increase limit.
                    # this is a use_global issue.
                    if isinstance(peak_param_delta['delT_increase_abs_max'],float):
                        abs_maxval_delT += peak_param_delta['delT_increase_abs_max']
                    else:
                        abs_maxval_delT += peak_param_delta['delT_increase_abs_max'][wstr]

            # integral_dist - includes the inverse transform back to 
            # energy per duration from the -1..1 space (-1..1 gets shifted so it
            # is not strictly -1..1)
            iter0 = 0
            max_iter = 20
            
            #TODO - you need more work on the relationship between peak 
            #       temperature and total energy
            
            E_per_duration = integral_dist['func'](rng.random(1)[0],
                                                   mu_E,
                                                   sig_E,
                                                   -1+Sm1_E,
                                                   1+S_E,
                                                   minval_E,
                                                   maxval_E)
            T_per_duration = integral_dist['func'](rng.random(1)[0],
                                                   mu_T,
                                                   sig_T,
                                                   -1+Sm1_T,
                                                   1+S_T,
                                                   minval_T,
                                                   maxval_T)
            
            # columns = "HLY-TEMP-10PCTL","HLY-TEMP-NORMAL","HLY-TEMP-90PCTL"
            # not adjustment here because 
            if self.use_global == False:
                if norm_temperature < 0.0:
                    # heat between current weather and climate norms. This is heat that already
                    # counts for heat waves that are baselined off the climate norms
                    heat_added_0 = (org_vals.values - norms_hourly["HLY-TEMP-NORMAL"].values).sum()
                    delT_max_0 = (org_vals.values - norms_hourly["HLY-TEMP-NORMAL"].values).mean()
                    
                else:
                    heat_added_0 = (org_vals.values - norms_hourly["HLY-TEMP-NORMAL"].values).sum()
                    delT_max_0 = (org_vals.values - norms_hourly["HLY-TEMP-NORMAL"].values).mean()
            else:
                heat_added_0 = 0.0
                delT_max_0 = 0.0
                
            # these are both normalized off of the original climate normals that were used as the heat wave
            # standard. 
            heat_added = E_per_duration * norm_energy * alpha_E * (duration/norm_duration) - heat_added_0
            delTmax = T_per_duration * norm_temperature * (alpha_T * (duration/norm_duration) + beta_T) - delT_max_0
            
            # We reduce the peak temperature increase to limit it to IPCC feasible temperature changes regardless of
            # how long of a duration heat wave the Markov process creates.
            # The energy is maintained but some may be shed if the solution cannot produce the energy requested.
            if (is_hw and (delTmax > abs_maxval_delT)) or (not is_hw and (delTmax < abs_maxval_delT)):
                delTmax = abs_maxval_delT
            
            
            D = duration
            
            # IF heat_added_0 is big enough it can actually serve to make things more mild. Either way, existing heat
            # waves and cold snaps do not have to be taken out because the original weather data is being referenced through
            # heat_added_0 and delT_max_0 to the climate norms 
            
            # calculate D_odd
            Dint = np.floor(duration/min_steps)
            if np.mod(Dint,2)==0 and Dint != 0:
                D_odd = Dint - 1
            elif Dint != 0:
                D_odd = Dint
            else:
                Dint = 1
            D_odd = D_odd * min_steps
            
            # determine E and delTmax coefficients
            Acoef = (delTmax - np.pi/(2*D_odd) * heat_added)/(2 - np.pi * D / (2 * D_odd)
                                                  + (min_steps/(2*D_odd)) * 
                                                  np.sin(2*np.pi * D/min_steps))
            Bcoef = (delTmax - Acoef)/2
            
            if test_shape_func:
                
                acceptable_error_percent = 0.1
                # 100 steps per day.
                xval = np.array(np.arange(0,duration,duration/(24*100)))
    
                h_test = Extremes.heat_wave_shape_func(xval,Acoef,Bcoef,D,D_odd,min_steps)
                h_test_heat = np.trapz(h_test,xval)
                
                if 100 * (h_test_heat - heat_added)/heat_added > acceptable_error_percent:
                    raise ExtremesIntegrationError("The heat wave increased "
                                               +"temperature did not integrate to"
                                               +" the heat_added. This should"
                                               +" never happen. There is a bug.")
            
            # adjust to meet delTmax but not E if the solution is non-physical.
            # Acoef and Bcoef > 0 for a physically real solution.
            dual_solution = False
            if heat_added >= 0 and Acoef > delTmax:
                Acoef = delTmax
                Bcoef = 0.0
            elif heat_added >= 0 and Acoef < 0.0:
                Bcoef = delTmax / 2
                Acoef = 0.0
            elif heat_added <= 0 and Acoef < delTmax:
                Acoef = delTmax
                Bcoef = 0.0
            elif heat_added <=0 and Acoef > 0:
                Acoef = 0.0
                Bcoef = delTmax / 2
            else:
                dual_solution = True
                Bcoef = (delTmax - Acoef)/2
            
            if iter0 >= max_iter:
                raise ValueError("The E and delTmax sampling could not find a physically real solution!")
            
            if is_hw:
                self._delTmax_verification_data["delTmax"].append({"delT_max_0":delT_max_0,
                                                                   "delTmax":delTmax,
                                                                   "coef_delTmax":Acoef + 2*Bcoef,
                                                                   "norm_temperature":norm_temperature,
                                                                   "maxval_T":maxval_T,
                                                                   "T_per_duration":T_per_duration,
                                                                   "duration":duration,
                                                                   "norm_duration":norm_duration,
                                                                   "minval_T":minval_T,
                                                                   "mu_T":mu_T,
                                                                   "sig_T":sig_T,
                                                                   "abs_maxval_delT":abs_maxval_delT,
                                                                   "s_month":s_month,
                                                                   "org_dates":org_dates,
                                                                   "key_name":key_name})    
            
            # for unit testing, assure the shape function used integrates 
            # correctly
     
            h_t = Extremes.heat_wave_shape_func(np.array(range(duration)),Acoef,Bcoef,D,D_odd,min_steps)
            
            # renormalize to make hourly approximation exactly fit heat_added
            if dual_solution:
                h_t = heat_added * h_t / h_t.sum()
            
            new_vals = pd.Series(h_t,index=org_vals.index,name=org_vals.name,dtype=org_vals.dtype) 
        else:
            raise ValueError("MEWS currently only can have 'double_shape_func' or 'heat_wave_shape_func'" +
                             " as values for the shape_func_type input")
        

        
        return new_vals, heat_added_0, delT_max_0, norm_temperature, norm_duration
    
    
    def _coef_form(self,hw_param,cs_param):
        """
        Form coefficients array and decay_func_type dictionary for input to
        DiscreteMarkov class
        
        """
        hwcoef = hw_param['decay function coef']
        cscoef = cs_param['decay function coef']
        
        coef = {ABREV_WAVE_NAMES[0]:cscoef, ABREV_WAVE_NAMES[1]:hwcoef}
        
        decay_func_type = {ABREV_WAVE_NAMES[0]:cs_param['decay function'],
                           ABREV_WAVE_NAMES[1]:hw_param['decay function']}
        
        return coef, decay_func_type
        
        
        
                
                
            
        
        
        
    
    
    

    

