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
from mews.cython import markov_chain
from mews.stats.markov import MarkovPy
from mews.errors.exceptions import ExtremesIntegrationError
from numpy.random  import default_rng, seed
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import os
import warnings

class DiscreteMarkov():
    """
    obj = MarkovChain(transition_matrix,state_names,use_cython=True)
    
    Object to work with Discrete Markov Chains
    
    Methods
    ----------
    .history(self, num_step, state0):
        Calculate a history of num_step for the markov process
    
    .steady(self)
        Calculate steady-state (stationary) probabilities of each state
        given the transition matrix
    
    """
    
    def __init__(self,rng,transition_matrix,state_names=None,use_cython=True):
        """
        DiscreteMarkov

        Parameters
        ----------
        rng : numpy.random.default_rng() :
            Random number generator
        transition_matrix : np.ndarray of np.float
            Square matrix that is a right stochastic matrix (i.e. the rows 
            sum to 1)
        state_names : list of strings, optional
            list of names that must be of the same dimension as the dimension
            as the transition_matrix
            The default is None and a list of ["0","1",...] is assigned 
        use_cython : bool, optional
            Use cython to calculate markov chain histories (~20x faster). 
            The default is True.

        Returns
        -------
        DiscreteMarkof Object

        """
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
        
    def history(self,num_step,state0,min_steps=24,skip_steps=0):
        """
        obj.history(num_step,state0, min_steps, skip_steps)
        
        calculate a history for a discrete markov chain of 'num_step' length
        and initial 'state0'
        
        Parameters
        ----------
        num_step : int or np.int
            number of steps to simulate for the object's defined transition_matrix
        state0 : int or str
            initial state of the history
        min_steps : int : optional
            minimum number of steps before a state can transition
        skip_steps : int :
            Number of steps at the beginning of the markov chain to assign to
            state0 (this is used for weather files that may begin in the middle
                    of the day to avoid applying heat waves out of sync with
                    day-night cycles)

        Raises
        ------
        ValueError
            Error in input for num_step or state0. State0 must be a value less
            than the dimension-1 of the transition_matrix or a string that is 
            in the self._names dictionary.
        TypeError
            All incorrect types are rejected from this method

        Returns
        -------
        None.

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
          
        num_real_step = np.int(np.floor((num_step - skip_steps)/min_steps))
        
        if num_real_step < 0:
            raise ValueError("The number of skip_steps must not be greater than the num_steps!")
        elif num_real_step == 0:
            raise ValueError("There are no real steps. The weather history"
                             +" provided is too short to sample in min_steps"
                             +"={0:d} size steps!".format(min_steps))
            
        remain_steps = num_step - num_real_step * min_steps - skip_steps
        prob = self.rng.random(num_real_step)
        
        cdf = self._cdf
        # The Markov Chain is NOT efficient in Python 
        if self.use_cython:
            state = markov_chain(cdf,prob,state0)
        else:
            state = MarkovPy.markov_chain_py(cdf,prob,state0)

        # translate state into an array of min_step length segments
        state_extended = np.repeat(state,min_steps)
        ending_state = np.repeat(np.array([state_extended[-1]]),remain_steps)
        begin_state = np.repeat(np.array([state0]),skip_steps)
        
        final_state = np.concatenate((begin_state,state_extended,ending_state))
        
        return final_state
            
    def steady(self):
        """
        obj.steady()
        
        Calculate stationary probabilities vector for each state based on the
        transition probabilities matrix
        
        Parameter:
            None

        Returns
        -------
        steady : np.ndarray
            vector of stationary probabilities of each state given the 
            transition matrix assigned to the process

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
    by adjustment of the statistical parameters involved
    
    for now this is written like it is the 
    
    
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
                 frac_long_wave=0.8,
                 min_steps=24,
                 test_shape_func=False,
                 doe2_input=None,
                 random_seed=None):
        
        """
        
        obj = Extremes(start_year,
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
                 frac_long_wave=0.8,
                 min_steps=24,
                 test_shape_func=False,
                 doe2_input=None)
        
        Parameters
        ----------
        
        start_year : int
            Year to begin a new history that will replace other years in 
            weather files
        
        max_avg_dist : function
            high extreme (for temperature extreme heat) statistical
            distribution function that returns random numbers that indicate
            how much energy/hr is associated with an extreme event
            
        max_avg_delta : float
            gradient of change for the average value of 
            max_avg_dist per year. Change is included step-wise on a yearly
            basis
            
        min_avg_dist : function
            low extreme (for temperature extreme cold) statistical distribution
            function that returns random numbers that indicate how much 
            reduction in energy/hr is associated with an extreme event
            
        min_avg_delta : float
            gradient of change for the average value of min_avg_dist per year.
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
            weather files that will be read and altered per the statistical specs above
            
        num_realizations : int
            The number of realizations per weather file that will be output for the
            statistics. Number of output weather files is num_realizations * len(weather_files)
            
        num_repeat : int : optional : Default = 1
            number of repeat years that will be processed into a longer duration
            than the original weather file.
            
        use_cython : bool : optional : Default = True
            flag to enable cython. If False, then much slower native python
            is used for the Markov chain 
            
        column : str : optional : Default = 'Dry Bulb Temperature'
            column for weather signal that will be altered. This input needs
            to be different for a DOE2 weather file.
            
        tzname : str : optional : Default = None
            time zone name to apply the weather files to. Must be a valid
            time zone for the tz Python library.
            
        write_results : bool : optional : Default = True
            flag to either write (True) or not write the weather files.
            results can be accessed through obj.results if not written.
            
        results_folder : str : optional : Default = ""
            string of a path to a location to write weather files
        
        results_append_to_name : str : optional : Default = ""
            string to append to the file names of output weather files
            
        run_parallel : bool : optional : Default = True
            flag to either run parallel (True) or run on a single processor (False)
            set to False if parallel run engine fails.
            
        frac_long_wave : float : optional : Default = 0.8
            Amount of variable added that is attributed to a sinusoidal function 
            accross the entire duration of the extreme event. The remainder is 
            applied to a constant increase for the entire extreme event.
            
        min_steps : int > 0 : optional : Default=24
            Minimum number of time steps for an extreme event 
            
        test_shape_func : bool : optional : Default = False
            for testing purposes
            Boolean indicating whether computationally expensive testing is 
            done to assure the shape function used integrates to the original
            heat_added. Default value is False - True is needed for unittesting
            
        doe2_input : dict : optional : Default = None
            Optional input required to perform the analysis using DOE2
            bin files. See mews.weather.alter:
                needs:
                    {'doe2_bin2txt_path':OPTIONAL - path to bin2txt.exe DOE2 utility
                                MEWS has a default location for this utility 
                                which can be obtained from DOE2 (www.doe2.com),
                    'doe2_start_datetime':datetime indicating the start date and time
                                          for the weather DOE2 weather file,
                     'doe2_tz'=time zone for doe2 file,
                     'doe2_hour_in_file'=8760 or 8784 for leap year,
                     'doe2_dst'= Start and end of daylight savings time for the 
                                 doe2_start_datetime year,
                      'txt2bin_exepath' : OPTIONAL - path to txt2bin.exe DOE2 utility}

        Returns
        -------
        
        None - look for results in obj.results - files are output as specified
               in the inputs.
        """
        if not random_seed is None:
            self.rng = default_rng(seed=random_seed)
        else:
            self.rng = default_rng()
        
        
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
        else:
            run_parallel = False
            warnings.warn("The program does not handle DOE2 files with"+
                              " multiple processors run_parallel set to 'False'",
                              UserWarning)
        
        
        state_int = [1,2] # no changes to non-selected intervals
        state_name = [name for name in self.states.keys()]
        num_wfile = len(weather_files)
        self.recursive_depth = 0
        
        results = {}
        for id0 in range(num_realizations):
            for idx in range(num_repeat):
                for idy,wfile in enumerate(weather_files):
                    year = start_year + idx * num_wfile + idy
                    key_name = (wfile, id0, year)
                    if run_parallel:
                        args = (start_year,idx,num_wfile,idy,wfile,column,
                                tzname,transition_matrix,transition_matrix_delta,
                                state_name,use_cython,state_int, 
                                min_avg_dist, min_avg_delta, 
                                max_avg_dist, max_avg_delta, results_folder, 
                                results_append_to_name, id0, write_results,year,
                                frac_long_wave,min_steps,test_shape_func,doe2_input)
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
                                frac_long_wave,min_steps,test_shape_func,doe2_input)
        
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
    
    # this function was created to make parallelization possible.
    def _process_wfile(self,start_year,idx,num_wfile,idy,wfile,column,tzname,
                       transition_matrix,transition_matrix_delta,state_name,
                       use_cython,state_int,min_avg_dist,
                       min_avg_delta,max_avg_dist,max_avg_delta,
                       results_folder, results_append_to_name,id0, 
                       write_results, year, frac_long_wave,min_steps,
                       test_shape_func,doe2_in):
        
        if doe2_in is None:
            objA = Alter(wfile, year)
        else:
            objA = self._DOE2Alter(wfile,year,doe2_in)
            
        org_series = objA.epwobj.dataframe[column]
        org_dates = objA.reindex_2_datetime(tzname).index
        num_step = len(objA.epwobj.dataframe.index)
        
        objDM = DiscreteMarkov(self.rng,transition_matrix 
                               + transition_matrix_delta 
                               * (year - start_year),
                               state_name,use_cython)
        # generate a history
        states_arr = objDM.history(num_step,"normal",min_steps=min_steps)
        # separate history into extreme states.
        state_intervals = self._find_extreme_intervals(states_arr, state_int)
        for state, s_ind in zip(state_intervals,state_int):
            if s_ind == 1:
                avg_dist = min_avg_dist
                avg_delta = (year - start_year) * min_avg_delta
            else:
                avg_dist = max_avg_dist
                avg_delta = (year - start_year) * max_avg_delta
                
            for tup in state:
                new_vals = self._add_extreme(org_series.iloc[tup[0]:tup[1]+1], 
                                  avg_dist, avg_delta,
                                  frac_long_wave=frac_long_wave,
                                  min_steps=min_steps,
                                  test_shape_func=test_shape_func)
                new_date_start = org_dates[new_vals.index[0]]
                duration = len(new_vals)
                if s_ind == 1:
                    peak_delta = new_vals.min()
                else:
                    peak_delta = new_vals.max()
                    
                objA.add_alteration(year, 
                                    new_date_start.day, 
                                    new_date_start.month, 
                                    new_date_start.hour, 
                                    duration, peak_delta,
                                    shape_func=new_vals.values,
                                    column=column)
                
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
                objA.write(os.path.join(results_folder,os.path.basename(wfile)[:-4] 
                             + results_append_to_name 
                             + "_{0:d}".format(year) 
                             + "_r{0:d}".format(id0)
                             + wfile[-4:]), 
                              overwrite=True, create_dir=True)
            else:
                objA.write(os.path.join(results_folder,os.path.basename(wfile)[:-4] 
                             + results_append_to_name 
                             + "_{0:d}".format(year) 
                             + "_r{0:d}".format(id0)
                             + wfile[-4:]), 
                              overwrite=True, create_dir=True,
                              txt2bin_exepath=doe2_in['txt2bin_exepath'])
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
    
    def _add_extreme(self,org_vals,integral_dist,integral_delta,frac_long_wave,min_steps,
                     test_shape_func=False):
        """
        obj._add_extreme(org_vals,integral_dist,integral_delta,frac_long_wave=1.0,min_steps=12)

        Parameters
        ----------
        org_vals : pandas.Series
            Original values for weather over the duration of an inserted 
            extreme event
            
        integral_dist : function
            random number generating function for the integral of total energy
            over the duration of the extreme event (returns total energy)
            
        integral_delta : float
            delta amount to energy intensity of extreme event (due to climate
            change)
            
        frac_long_wave : float 1 > frac_long_wave > 0
            Amount of heat added that is attributed to a sinusoidal function 
            accross the entire duration of the heat wave. The remainder is 
            applied to increases in daily heat.
            
        min_steps : int > 0
            Minimum number of steps heat waves must begin and end on (e.g. if
            min_steps=24, then a heat wave can only begin and end at the 
            beginning of a new day)
            
        test_shape_func : Bool, optional - for testing purposes
            Boolean indicating whether computationally expensive testing is 
            done to assure the shape function used integrates to the original
            heat_added. Default value is False - True is needed for unittesting

        Returns
        -------
        new_vals :  
            values to add to org_vals in the weather history to produce a new
            extreme event
        """        
        duration = len(org_vals)
        # heat added per hour duration of the heat wave
        heat_added = (integral_dist(None)+integral_delta) * duration
        
        # these are determined by taking the integral of "double_shape_func" from
        # 0 to D and equating it to heat_added times frac_long_wave for the sine term
        # and to (1-frac_long_wave) for the 1-cos term.        
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
        
        return new_vals
        
        
        
                
                
            
        
        
        
    
    
    

    

