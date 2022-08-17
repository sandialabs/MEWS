# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:22:19 2021

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

from mews.weather import Alter
from mews.cython.markov import markov_chain
from mews.stats.markov import MarkovPy
from mews.errors.exceptions import ExtremesIntegrationError
from numpy.random  import default_rng, seed
from scipy.optimize import curve_fit, minimize
import pandas as pd
import numpy as np
import os
import warnings

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

    Returns
    -------
    None

    """
    
    def __init__(self,rng,transition_matrix,state_names=None,use_cython=True):

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
        self.use_cython = True
        self.tol = 1e-10
        
    def history(self,num_step,state0,min_steps=24,skip_steps=0,count_in_min_steps_intervals=True):
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
        count_in_min_steps_intervals : bool : optional : Default = True
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
            num_real_step = np.int(np.floor((num_step - skip_steps)/min_steps))
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
        # The Markov Chain is NOT efficient in Python 
        if self.use_cython:
            state = markov_chain(cdf,prob,state0)
        else:
            state = MarkovPy.markov_chain_py(cdf,prob,state0)
        
        if count_in_min_steps_intervals:
            # translate state into an array of min_step length segments
            state_extended = np.repeat(state,min_steps)
            ending_state = np.repeat(np.array([state_extended[-1]]),remain_steps)
            begin_state = np.repeat(np.array([state0]),skip_steps)
            
            final_state = np.concatenate((begin_state,state_extended,ending_state))
        else:
            # TODO - this is slow!
            # move state changes to min_step intervals
            # for each day, assign the state that has the majority of hours in
            # the day.
            

            big_step = np.int(np.floor(num_step/min_steps))

            for idx in range(big_step):
                day = state[min_steps * idx:min_steps*(idx+1)]
                state_count = np.array([(day == state_val).sum() for name,state_val in self._names.items()])
                state[min_steps * idx:min_steps*(idx+1)] = state_count.argmax()

            final_state = state
        
        return final_state
            
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
        val,vec = np.linalg.eig(np.transpose(self._mat))
        steady = vec[:,0]/vec.sum(axis=0)[0]
        
        if np.abs(np.imag(steady).sum() > len(steady) * self.tol):
            raise UserWarning("The eigenvector has non-zero imaginary parts!")
        
        return np.real(steady)
            
                
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
            raise ValueError("All entries to the transition matrix are probabilities and must be positive!")
        
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
                       no_climate_trend=False)
    
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
        Number of repeat years that will be processed into a longer duration
        than the original weather file.
        
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
                 no_climate_trend=False):
        
        
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
                pool = mp.Pool(mp.cpu_count()-1)
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
                                new_input_format,climate_temp_func,averaging_steps,rng,no_climate_trend)
                        results[key_name] = pool.apply_async(self._process_wfile,
                                                         args=args)
                    else:
                        results[key_name] = self._process_wfile(start_year,idx,
                                num_wfile,idy,wfile,column,
                                tzname,transition_matrix,transition_matrix_delta,
                                state_name,use_cython,state_int,
                                min_avg_dist, min_avg_delta, 
                                max_avg_dist, max_avg_delta, results_folder, 
                                results_append_to_name, id0, write_results, year,
                                frac_long_wave,min_steps,test_shape_func,doe2_input,
                                max_E_dist,del_max_E_dist,min_E_dist,del_min_E_dist,
                                new_input_format,climate_temp_func,averaging_steps,self.rng,no_climate_trend)
        
        if run_parallel:
            results_get = {}
            for tup,poolObj in results.items():
                try:
                    results_get[tup] = poolObj.get()
                except AttributeError:
                    raise AttributeError("The multiprocessing module will not"
                                         +" handle lambda functions or any"
                                         +" other locally defined functions!")
               
            pool.close()
            self.results = results_get
        else:
            self.results = results
    
    def _monthly_input_checking(self,dict_list):
        incorrect_type = 0
        dict_found = False
        for dict_e in dict_list:
            if not isinstance(dict_e,dict) and not dict_e is None:
                incorrect_type += 1
            else:
                if not dict_e is None:
                    dict_found=True
                    if 'func' in dict_e.keys():
                        dict_e2 = dict_e['param']
                    else:
                        dict_e2 = dict_e
                    
                    if list(dict_e2.keys()) != [1,2,3,4,5,6,7,8,9,10,11,12]:
                        raise ValueError("All dictionary inputs must have an entry for every month!")
        
        
        if incorrect_type > 0 and incorrect_type != 6:
            raise TypeError("You must either define all inputs as dictionaires"
                            +" with 12 months as the key {1:,2:, etc..} or as"
                            +" functions and floats per the documentation")
        
        return dict_found
            
    
    # this function was created to make parallelization possible.
    def _process_wfile(self,start_year,idx,num_wfile,idy,wfile,column,tzname,
                       transition_matrix,transition_matrix_delta,state_name,
                       use_cython,state_int,min_avg_dist,
                       min_avg_delta,max_avg_dist,max_avg_delta,
                       results_folder, results_append_to_name,id0, 
                       write_results, year, frac_long_wave,min_steps,
                       test_shape_func,doe2_in, max_E_dist,del_max_E_dist,
                       min_E_dist,del_min_E_dist, new_input_format,
                       climate_temp_func,averaging_steps,rng, no_climate_trend):
        
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
                
                objDM = DiscreteMarkov(rng, modified_trans_matrix, 
                                       state_name, use_cython)
                
                if states_arr is None:
                    states_arr = objDM.history(month_num_steps, state0,count_in_min_steps_intervals=False)
                else:
                    states_arr = np.concatenate((states_arr, objDM.history(
                                    month_num_steps, 
                                    state0,
                                    count_in_min_steps_intervals=False)))
                
                state0 = state_name[states_arr[-1]]
            
            #import matplotlib.pyplot as plt
            #plt.plot(states_arr)
        else:
            objDM = DiscreteMarkov(rng,transition_matrix 
                               + transition_matrix_delta 
                               * (year - start_year),
                               state_name,use_cython)
        # distinguish the shape function type
        if new_input_format:
            shape_function_type = "heat_wave_shape_func"
        else:
            shape_function_type = "double_shape_func"
        
            # generate a history
            states_arr = objDM.history(num_step,"normal",min_steps=min_steps)
        # separate history into extreme states.
        state_intervals = self._find_extreme_intervals(states_arr, state_int)
        for state, s_ind in zip(state_intervals,state_int):

            if s_ind==1:
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
                new_vals = self._add_extreme(org_series.iloc[tup[0]:tup[1]+1], 
                                  avg_dist, avg_delta,
                                  frac_long_wave=frac_long_wave,
                                  min_steps=min_steps,
                                  shape_func_type=shape_function_type,
                                  test_shape_func=test_shape_func,
                                  peak_dist=peak_dist,
                                  peak_delta=peak_delta,
                                  org_dates=org_dates[tup[0]:tup[1]+1],
                                  rng=rng)

                new_date_start = org_dates[new_vals.index[0]]
                duration = len(new_vals)
                
                if s_ind == 1:
                    peak_delta_val = new_vals.min()
                else:
                    peak_delta_val = new_vals.max()
                    
                objA.add_alteration(year, 
                                    new_date_start.day, 
                                    new_date_start.month, 
                                    new_date_start.hour, 
                                    duration, peak_delta_val,
                                    shape_func=new_vals.values,
                                    column=column,
                                    averaging_steps=averaging_steps)
        if not climate_temp_func is None and not no_climate_trend:
            # TODO - add the ability to continuously change the trend or to
            #        just make it constant like it is now.
            objA.add_alteration(year,1,1,1,num_step,climate_temp_func(year),np.ones(num_step),
                                alteration_name="global temperature trend")
                
                
        if write_results:
            # TODO, this writing needs to be standardized accross all
            # alteration combinations!
            if doe2_in is None:
                simple_call = True
            elif isinstance(doe2_in,dict) and not 'txt2bin_exepath' in doe2_in:
                simple_call = True
            else:
                simple_call = False 
            
            
            new_wfile_name = (os.path.basename(wfile)[:-4] 
                         + results_append_to_name 
                         + "_{0:d}".format(year) 
                         + "_r{0:d}".format(id0)
                         + wfile[-4:])
            
            if simple_call:
                objA.write(os.path.join(results_folder,new_wfile_name), 
                              overwrite=True, create_dir=True)
            else:
                objA.write(os.path.join(results_folder,new_wfile_name), 
                              overwrite=True, create_dir=True,
                              txt2bin_exepath=doe2_in['txt2bin_exepath'])
            self.wfile_names.append(new_wfile_name)
        return objA
    
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

    def _find_extreme_intervals(self,states_arr,states):
        diff_states = np.concatenate((np.array([0]),np.diff(states)))
        state_int_list = []
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
            state_int_list.append(ep_list)
        return state_int_list

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
                     org_dates=None,rng=None):
        """
        >>> obj._add_extreme(org_vals,
                             integral_dist,
                             integral_delta,
                             frac_long_wave=1.0,
                             min_steps,
                             shape_func_type=None,
                             test_shape_func=False,
                             peak_dist=None,
                             peak_delta=None)

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
            
        elif shape_func_type == "heat_wave_shape_func":
            # THIS IS HIGHLY REPETITIVE - perhaps some code refactoring would be appropriate
            # the month of the start date determines the sampling parameters

            # statistics come from the month with he majority of hours in.
            s_month = org_dates.month.unique()[np.array([(org_dates.month == 
                        month).sum() for month in org_dates.month.unique()]
                                                        ).argmax()]

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
            Sm1_E = 0.0
            Sm1_T = 0.0
            S_E = 0.0
            S_T = 0.0
            
            # introduce a delta if it exists.
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
                
                
            # integral_dist - includes the inverse transform back to 
            # energy per duration from the -1..1 space (-1..1 gets shifted so it
            # is not strictly -1..1)
            iter0 = 0
            max_iter = 20
            found_physical_solution = False
            
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
            
            heat_added = E_per_duration * norm_energy * alpha_E * (duration/norm_duration)
            Tmax = T_per_duration * norm_temperature * (alpha_T * (duration/norm_duration) + beta_T)
            D = duration
            
            # calculate D_odd
            Dint = np.floor(duration/min_steps)
            if np.mod(Dint,2)==0 and Dint != 0:
                D_odd = Dint - 1
            elif Dint != 0:
                D_odd = Dint
            else:
                Dint = 1
            D_odd = D_odd * min_steps
            
            # determine E and Tmax coefficients
            Acoef = (Tmax - np.pi/(2*D_odd) * heat_added)/(2 - np.pi * D / (2 * D_odd)
                                                  + (min_steps/(2*D_odd)) * 
                                                  np.sin(2*np.pi * D/min_steps))
            Bcoef = (Tmax - Acoef)/2
            
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
            
            # adjust to meet Tmax but not E if the solution is non-physical.
            dual_solution = False
            if heat_added >= 0 and Acoef > Tmax:
                Acoef = Tmax
                Bcoef = 0.0
            elif heat_added >= 0 and Acoef < 0.0:
                Bcoef = Tmax / 2
                Acoef = 0.0
            elif heat_added <= 0 and Acoef < Tmax:
                Acoef = Tmax
                Bcoef = 0.0
            elif heat_added <=0 and Acoef > 0:
                Acoef = 0.0
                Bcoef = Tmax / 2
            else:
                dual_solution = True
                Bcoef = (Tmax - Acoef)/2
            
            if iter0 >= max_iter:
                raise ValueError("The E and Tmax sampling could not find a physically real solution!")
                
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
        

        
        return new_vals
        
        
        
                
                
            
        
        
        
    
    
    

    

