# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:59:38 2021

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
from mews.stats import Extremes
from mews.stats.solve import SolveDistributionShift, shift_a_b_of_trunc_gaussian
from mews.weather.psychrometrics import relative_humidity
from mews.weather.climate import ClimateScenario
from mews.utilities.utilities import filter_cpu_count
from mews.stats.distributions import (cdf_truncnorm, trunc_norm_dist, 
                                      inverse_transform_fit, transform_fit)
from mews.constants.data_format import (WAVE_MAP, VALID_SOLVE_OPTIONS, DEFAULT_SOLVE_OPTIONS,
                                        VALID_SOLVE_OPTION_TIMES,VALID_SOLVE_INPUTS)

from mews.utilities.utilities import (write_readable_python_dict,
                                      read_readable_python_dict,
                                      dict_key_equal, bin_avg)

from copy import deepcopy
from datetime import datetime
from scipy.optimize import bisect, fsolve
from mews.constants.physical import (HOURS_IN_YEAR, DAYS_IN_YEAR, HOURS_IN_DAY)
from mews.constants.analysis import (DEFAULT_SOLVER_NUMBER_STEPS, 
                                     DEFAULT_RANDOM_SEED)
from shutil import which
import traceback
import io
import pandas as pd
import numpy as np
import os
import urllib
from urllib.parse import urlparse

import statsmodels.api as sm
from warnings import warn

import matplotlib.pyplot as plt

def _calculate_shift(obj2,first_try_solution_location,month_factors,sol_dir,cii,new_solution_name):
    
    # brings in the historic solution
    stats = obj2.read_solution(first_try_solution_location)
    stats_new = deepcopy(stats)
    
    for month in np.arange(1,13):                            
        stat = stats['heat wave'][month]
        stat_new = stats_new['heat wave'][month]
        alpha = stat['normalized extreme temp duration fit slope']
        beta = stat['normalized extreme temp duration fit intercept']
        hist0 = stat['historical durations (durations0)']
        avgbin = bin_avg(hist0)
        Dmean = (hist0[0] * avgbin).sum()/hist0[0].sum()
        Dmax = stat['normalizing duration']
        delTmax = stat['normalizing extreme temp'] 
        delTipcc = obj2.ipcc_results['ipcc_fact'][month]['10 year event'][cii + " CI Increase in Intensity"]
        
        delms = delTipcc / ((alpha * Dmean/Dmax + beta) * delTmax) / 2
        
        mu_T = stat['extreme_temp_normal_param']['mu']
        sig_T = stat['extreme_temp_normal_param']['sig']
        
        dela, delb = shift_a_b_of_trunc_gaussian(delms*month_factors[month-1], mu_T, delms*month_factors[month-1], sig_T )
        
        stat_new['extreme_temp_normal_param']['mu'] = mu_T + month_factors[month-1] * delms
        stat_new['extreme_temp_normal_param']['sig'] = sig_T + month_factors[month-1] * delms
        stat_new['min extreme temp per duration'] = stat_new['min extreme temp per duration'] + dela
        stat_new['max extreme temp per duration'] = stat_new['max extreme temp per duration'] + delb
        
        # now do energy assuming that energy grows proportionately with temperature (i.e. duration is not)
        # increasing for this case.
        
        # ASSUME A SINUSOIDAL FORM! the area under a single sinusoidal wave 
        # is 2 times the amplitude delTipcc.
        alpha_E = stat['energy linear slope']
        Emax = stat['normalizing energy']
        
        delms_E = 2 * delTipcc / ((alpha_E * Dmean/Dmax) * Emax) / 2

        mu_E = stat['energy_normal_param']['mu']
        sig_E = stat['energy_normal_param']['sig']
        
        dela_E, delb_E = shift_a_b_of_trunc_gaussian(delms_E*month_factors[month-1], mu_E, 
                                                     delms_E*month_factors[month-1], sig_E)
        
        
        
        stat_new['energy_normal_param']['mu'] = mu_E + month_factors[month-1] * delms_E
        stat_new['energy_normal_param']['sig'] = sig_E + month_factors[month-1] * delms_E
        stat_new['min energy per duration'] = stat_new['min energy per duration'] + dela_E
        stat_new['max energy per duration'] = stat_new['max energy per duration'] + delb_E
    
    obj2.stats = stats_new
    obj2.write_solution(os.path.join(sol_dir,new_solution_name))


def _assemble_x_solution(stats,month):
    """
    This must follow the x input vector convention for 
    ObjectiveFunction.markov_gaussian_model_for_peak_temperature
    in solve.py

    Returns
    -------
    x_solution : TYPE
        DESCRIPTION.

    """
    # no deltas on mean and std.
    x_solution = [0.0,0.0,0.0,0.0]
    # proba of waves
    x_solution.append(stats['cold snap'][month]['hourly prob of heat wave'])
    x_solution.append(stats['heat wave'][month]['hourly prob of heat wave'])
    # probab sustain waves
    x_solution.append(stats['cold snap'][month]['hourly prob stay in heat wave'])
    x_solution.append(stats['heat wave'][month]['hourly prob stay in heat wave'])
    for wave_type in ['cold snap', 'heat wave']:
        if 'decay function coef' in stats[wave_type][month]:
            for coef in stats[wave_type][month]['decay function coef']:
                x_solution.append(coef)

    return np.array(x_solution)

def _mix_user_and_default(default_vals,solve_type,solve_options):
    opt_val = {}
    for key,def_val in default_vals.items():
        if key in solve_options[solve_type]:
            opt_val[key] = solve_options[solve_type][key]
        else:
            opt_val[key] = def_val
    return opt_val

def _process_extra_columns(ext_col,month,fut_year=None,clim_scen=None,ci_interval=None):
    #{'future year':None,'climate scenario':None, 'threshold confidence interval': None}
    
    # This function is only used in historic and future context.
    for key, val in zip(['future year', 'climate scenario', 'threshold confidence interval','month'],[fut_year,clim_scen,ci_interval,month]):
        
        if (not val is None):
            ext_col[key] = val
        else:
            ext_col[key] = 'historic'
    
    return ext_col

def _create_figure_out_name(dirname,basename,month,is_dir,dir_exists,scen=None,ci_int=None,year=None,is_historic=True):

    if not dir_exists:
        if not os.path.exists(dirname) and len(dirname) > 0:
            os.mkdir(dirname)
                
    if not scen is None:
        ext_str = "_{0}_{1}_{2}".format(scen,ci_int,str(year))
    else:
        ext_str = ""
    if is_historic:
        hstr = "historic"
    else:
        hstr = "future"
        
    if is_dir:
        filename = os.path.join(dirname,basename,hstr+"_month_{0:d}{1}.png".format(month,ext_str))
    else:
        filename = os.path.join(dirname,basename.split(".")[0]+"_"+hstr+"_month_{0:d}{1}.png".format(month,ext_str))
    
    return filename

class ExtremeTemperatureWaves(Extremes):
    
    """
    >>> ExtremeTemperatureWaves(station,
                                weather_files,
                                unit_conversion,
                                num_year,
                                use_local,
                                include_plots,
                                doe2_inputs,
                                results_folder,
                                proxy,
                                use_global,
                                delT_ipcc_min_frac)
    
    This initializer reads and processes the heat waves and cold snap statistics
    for a specific NOAA station and corresponding weather data. After instantiation,
    the "create_scenario" method can be used to create weather files.
    
    Parameters
    ----------
    
    station : str or dict
        str: Must be a valid NOAA station number that has both daily-summary
        and 1991-2020 hourly climate norms data. If use_local=True, then
        this can be the path to a local csv file and <station>_norms.csv
        is expected in the same location for the climate norm data.
        
        dict: must be a dictionary of the following form
            {'norms':<str with path to norms file>,
             'summaries':<str with path to the daily summaries file>}
    
    weather_files : list
        List of path and file name strings that include all the weather files
        to alter
        
    unit_conversion : tuple
        Index 0 = scale factor for NOAA daily summaries data to convert to Celcius
        Index 1 = offset for NOAA daily summaries to convert to Celcius
        For example if daily summarries are Fahrenheit, then (5/9, -(5/9)*32)
        needs to be input. If Celcius then (1,0). Tenths of Celcius (1/10, 0)
        
    use_local : Bool : Optional: Default = False
        Flag to indicate that that "station" input is actually a path to
        local <station>.csv and <station>_norms.csv files for the NOAA data
        
    include_plots : Bool : Optional : Default = False
        True : plot all kinds of diagnostic information to help determine 
        if heat waves are well characterized statistically by the data. 
        This adds run time but is highly advised for new weather stations
        not previously analyzed.
        
    doe2_input : dict : Optional : Default = None
       | If none - process the run as E+.
       |
       | Optional input required to perform the analysis using DOE2
       | bin files. See mews.weather.alter. needs:
       |     
       | {'doe2_bin2txt_path':OPTIONAL - path to bin2txt.exe DOE2 utility
       |  MEWS has a default location for this utility 
       |  which can be obtained from DOE2 (www.doe2.com),
       | 'doe2_start_datetime':datetime indicating the start date and time
       | for the weather DOE2 weather file,
       | 'doe2_tz'=time zone for doe2 file,
       | 'doe2_hour_in_file'=8760 or 8784 for leap year,
       | 'doe2_dst'= Start and end of daylight savings time for the 
       | doe2_start_datetime year,
       | 'txt2bin_exepath' : OPTIONAL - path to txt2bin.exe DOE2 utility}
    
    results_folder : str : Optional : Default = "mews_results"
        Path to the location where MEWS will write all of the output files
        for the requested analysis. Files will be the original weather file
        name with "_<realization>_<year>" appended to it.
        
    random_seed : int : optional : Default : None
        None : allow mews to generate heat waves whose random seed are not
               controlled. Sequential runs will not produce the same
               result.
        int < 2^32-1 : Random seed will enable sequential runs to produce
              the same result.
              
    run_parallel : Bool : optional : Default = True
        True : use parallel processing 
        False : do not use parallel processing - this is useful for 
                debugging or if MEWS is being used in another parallel process
        see num_cpu
        
    proxy : str : Optional : Default = None
        proxy server and port to access url's behind a proxy server
        example: https://proxy.institution.com:8080
        
    use_global : bool : optional : Default = False
        True means use the old use case where temperature anomaly
        is based on global average mean surfacne temperature curves rather
        specific curves in CMIP6.
        
    delT_ipcc_min_frac : float : optional : Default = 1 Range 0 - 1 or ValueError raised
        The fraction of the interpolated change in maximum temperature increase
        for a heat wave applicable on the lowest probability of heat waves month.
        The highest probability of heat waves month gets the full delT from the
        IPCC data. Default is set to a value of 1 for worst case where every
        month gets the full shift. Must range between 0 and 1.
        
    norms_unit_conversion : tuple : optional : Default = (5/9, -(5/9) * 32)
        conversion multiplier and offset needed to convert the climate normals 
        NOAA data to degrees Celcius.
        
    num_cpu : int or None : optional : Default = None
        None : use the maximum number of cpu available minus 1
        int : use num_cpu cpu's. MEWS will select the max available minus one if this 
              is too large.
              
    write_results : Bool : optional : Default = True
        if True, write the energyplus or DOE2 files for each realization.
        if False, do not write (saves time when testing or when the 
                                MEWS results are going to be used directly in python)
        
    test_markov : Bool : optional : Default = False
        Keep False for all practical use of MEWS. This is a unit testing feature
        that will make run times longer.
        if True the process will run the markov process until another heat wave
        event occurs so that convergence studies can accurately assess time gaps
        between heat waves for a given month and reconstruct if the process
        is working as intended.
        
    solve_options : dict : Optional, Default = None
        options to be passed to mews.stats.solve.SolveDistributionShift.
        There are two levels to this dictionary:
            ['historic','future'] is the first level key
        A subdictionary then gives the name of an input parameter as the key and then
        the value should be the desired input. Valid input parameters are: 
            
            ['problem_bounds',
             'decay_func_type',
             'use_cython',
             'num_cpu',
             'plot_results',
             'max_iter',
             'plot_title',
             'out_path',
             'weights',
             'limit_temperatures',
             'delT_above_shifted_extreme',
             'num_step',
             'min_num_waves',
             'x_solution',
             'test_mode',
             'num_postprocess',
             'extra_output_columns']
            
        See the documentation for mews.stats.solve.SolveDistributionShift 
        to understand these parameters.
        
    solution_file : str : optional : Default = ""
        Path and name of a file that contains a solution either historic
        or for the current scenario. This enables by-passing the optimization
        problem that takes significant computational resources so that
        MEWS can implement a solution already computed quickly
        
    overwrite_existing : bool : optional : Default = True
       If epw or bin files are being written, designate if existing files
       will be overwritten. This is useful if MEWS only creates some 
       of a large number of requested files.
        
    Returns
    -------
    None
    
    """
    #  These url's must end with "/" !
    
    # norms are provided in Fahrenheit!
    norms_url  = "https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/"
    # daily data are provided in Celcius!
    daily_url = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/"
    
    def __init__(self,station,
                 weather_files,
                 unit_conversion,
                 use_local=False,
                 include_plots=False,
                 doe2_input=None,
                 results_folder="mews_results",
                 random_seed=None,
                 run_parallel=True,
                 proxy=None,
                 use_global=False,
                 delT_ipcc_min_frac=1.0,
                 norms_unit_conversion=(5/9,-(5/9)*32),
                 num_cpu=None,
                 write_results=True,
                 test_markov=False,
                 solve_options=None,
                 solution_file="",
                 overwrite_existing=True):
        
        self._config_latex()
        self._overwrite_existing = overwrite_existing
        
        # This does the baseline heat wave analysis on historical data
        # "create_scenario" moves this into the future for a specific scenario
        # and confidence interval factor.
        solve_options = self._check_solve_options(solve_options)
        
        self._num_cpu = filter_cpu_count(num_cpu)
        self._test_markov = test_markov
        
        if delT_ipcc_min_frac > 1.0 or delT_ipcc_min_frac < 0.0:
            raise ValueError("The input 'delT_ipcc_min_frac' must be between 0.0 and 1.0!")
        
        self._proxy = proxy
        self.use_global = use_global
        np.random.seed(random_seed)
        self._random_seed = random_seed
        
        # consistency checks
        self._check_NOAA_url_validity()
        
        # temperature specific tasks
        
        if include_plots:
            # ! TODO - move all plotting out of this class
            plt.close("all")

        # The year input 2001 is arbitrary and should simply not be a leap year
        self._read_and_curate_NOAA_data(station,2001,unit_conversion,norms_unit_conversion,use_local)
        
        # produce the initial values of several parameters. The optimization
        # will actually produce the statistics needed.
        stats, hours_per_year = self._wave_stats(self.NOAA_data,include_plots)
        
        # self.stats will be overridden later on!
        self.stats = stats
        
        if not use_global:
            new_stats = self._solve_historic_distributions(stats, solve_options, hours_per_year,solution_file)
        else:
            new_stats = stats
            
        self._hours_per_year = hours_per_year
            
        self.stats = new_stats
        
        if include_plots:
            self._plot_stats_by_month(stats["heat wave"],"Heat Waves")
            self._plot_stats_by_month(stats["cold snap"],"Cold Snaps")
        
        self._results_folder = results_folder
        self._run_parallel = run_parallel
        self._doe2_input = doe2_input
        self._weather_files = weather_files
        self._delT_ipcc_min_frac = delT_ipcc_min_frac
        self.ext_obj = {}
        self._verification_data = {}
        

        self.extreme_results = {}
        self.extreme_delstats = {}
        self.ipcc_results = {"ipcc_fact":{},'durations':{}}
        self._create_scenario_has_been_run = False
        self._write_results = write_results
        self.solve_options = solve_options
        self.future_solve_obj = {}
        self._station = station
        self._unit_conversion = unit_conversion
        self._use_local = use_local
        self._include_plots = include_plots
        self._norms_unit_conversion = norms_unit_conversion
        self._num_cpu = num_cpu
        self._test_markov = test_markov
        self._solution_file = solution_file

    def _config_latex(self):
        if not which("latex") is None:
            plt.rcParams.update({
                                "text.usetex": True,
                                "font.family": "Helvetica"})
        else:
            plt.rcParams.update({
                        "text.usetex": False,
                        "font.family": "Helvetica"})
        
    
    def _parallel_create_solution_func(self,
                                       syear,
                                       scen,
                                       cii,
                                       obj,
                                       historic_solution,
                                       scen_dict,
                                       cold_snap_shift,
                                       sol_dir,
                                       filename,
                                       overwrite):
        """
        Calculate solutions in parallel
        
        This function is just an interative process. The location of change is in
        _calculate_shift where obj.stats values are changed and a solution file 
        is written. The function then uses obj.create_scenario with the new solution 
        file to assess if a shift has improved the fit.
        
        COLD SNAP SHIFT IS NOT IMPLEMENTED HERE AND WOULD REQUIRE A LOOP 
        OVER COLD SNAP FOLLOWED BY THE CURRENTLY IMPLEMENTED LOOP OVER 
        HEAT WAVES. IMPLEMENT A UNIT TEST FOR THIS CAPABILITY!
        
        """
        # restart capability
        num_cases = 3  # the stochastic nature of the model requires 
                       # evaluating the model 8 times to get different actual
                       # shift results.
        
        cancel_run = False
        fname = filename+"final_solution_{0:d}_{1}_{2}.txt".format(syear,scen,cii)
        if (os.path.exists(fname) and not overwrite):
            return None, None
        
        # start fresh with the historical solution
        obj2 = deepcopy(obj)
        
        if not os.path.exists(historic_solution):
            obj2.write_solution(os.path.join(historic_solution))
        
        # look at three cases to make a better interpolation.
        if 'future' in obj2.solve_options:
            if 'num_postprocess' in obj2.solve_options['future']:
                org_num_postprocess = obj2.solve_options['future']['num_postprocess']
            if 'plot_results' in obj2.solve_options['future']:
                org_plot_results = obj2.solve_options['future']['plot_results']
                
        obj2.solve_options['future']['num_postprocess'] = num_cases
        obj2.solve_options['future']['plot_results'] = False
        
        ## ONLY FOR TROUBLESHOOTING!!
        #obj2.solve_options['future']['test_mode'] = True
        #obj2.solve_options['future']['num_steps'] = 1e5 # much fewer steps to reduce run time.

    
        # Question - do you want to look across the extreme event confidence
        #            intervals or just stick to the 50%?
        #          - do you want a cold snap shift? I don't have information for
        #            how much cold snaps will change with increasing global warming
        
        # THIS IS THE HISTORICAL SOLUTION
        org_write_results = obj2._write_results
        obj2._write_results = False
        result = obj2.create_scenario(scenario_name=scen,
                                                        year=syear,
                                                        climate_temp_func=scen_dict,
                                                        num_realization=1,
                                                        climate_baseyear=2014,
                                                        increase_factor_ci=cii,
                                                        cold_snap_shift=cold_snap_shift,
                                                        solution_file=historic_solution)
        
        if result[syear] is None:
            cancel_run = True
            return result,fname,cancel_run
        else:
        
            obj2_iter_prev = deepcopy(obj2)
            # These are an initial guess with positive value to represent stepping forward.
            month_factors = [1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0,
                             1.0]
            
            temp_new_solution_name = "iterate_{0}_{1}_{2}.txt".format(str(scen),str(syear),str(cii))
            
            error_more_than_1_percent = True
            max_iter = 5
            iter_ = 0
    
            while error_more_than_1_percent:
                # historical is just the previous iteration
                
                # TAKE A CALCULATED SHIFT THAT HAS LINEAR VARIATION
                # This writes a solution file used by the next create_scenario command
                # calculate_shift always goes to the historic average
                _calculate_shift(obj2,historic_solution,month_factors,sol_dir,cii,temp_new_solution_name)
                
                
                result = obj2.create_scenario(scenario_name=scen,
                                                                year=syear,
                                                                climate_temp_func=scen_dict,
                                                                num_realization=1,
                                                                climate_baseyear=2014,
                                                                increase_factor_ci=cii,
                                                                cold_snap_shift=cold_snap_shift,
                                                                solution_file=os.path.join(sol_dir,temp_new_solution_name))
                
                # calculate the sensitivity and then new month factors that will produce an exact solution on the linear
                # variation.
                new_month_factors = []
                for month in np.arange(1,13):
                    # h_thresh is the previous iteration's solution (first time through its the historical solution)
                    h_thresh = [obj2_iter_prev.future_solve_obj[scen][cii][syear][month].obj_solve.thresholds[case+1]['hw']
                                for case in range(num_cases)]
                    # next iteration.
                    f_thresh = [obj2.future_solve_obj[scen][cii][syear][month].obj_solve.thresholds[1]['hw']
                                for case in range(num_cases)]
                    
                    h_gap = np.array([0.5*((h_thresh[case]['target'][0] - h_thresh[case]['actual'][0]) 
                                           + (h_thresh[case]['target'][1] - h_thresh[case]['actual'][1]))
                                      for case in range(num_cases)]).mean()
                    f_gap = np.array([0.5*((f_thresh[case]['target'][0] - f_thresh[case]['actual'][0]) 
                                           + (f_thresh[case]['target'][1] - f_thresh[case]['actual'][1]))
                                      for case in range(num_cases)]).mean()
                    
                    new_month_factors.append(month_factors[month-1] * h_gap / (h_gap - f_gap))
                    
                max_percent_error = np.abs(100*(np.array(new_month_factors) - np.array(month_factors))/np.array(month_factors)).max()
                #print("\n\n")
                #print("max_percent_error: {0:5.2f}".format(max_percent_error))
                #print("f_gap: {0:5.2f}".format(f_gap))
    
                if iter_ > max_iter:
                    error_more_than_1_percent = False
                elif max_percent_error < 1.0:
                    error_more_than_1_percent = False
                else:
                    iter_ += 1
                    
                month_factors = new_month_factors
            
            
            obj2.write_solution(os.path.join(sol_dir,fname))
            
            # remove the temporary file
            if os.path.exists(os.path.join(sol_dir,temp_new_solution_name)):
                os.remove(os.path.join(sol_dir,temp_new_solution_name))
            
            if 'future' in obj2.solve_options:
                if 'num_postprocess' in obj2.solve_options['future']:
                    org_num_postprocess = obj2.solve_options['future']['num_postprocess']
                if 'plot_results' in obj2.solve_options['future']:
                    org_plot_results = obj2.solve_options['future']['plot_results']
            
            # restore original solve options
            if 'future' in obj2.solve_options:
                if 'num_postprocess' in obj2.solve_options['future']:
                    if "org_num_postprocess" in locals():
                        obj2.solve_options['future']['num_postprocess'] = org_num_postprocess 
                if 'plot_results' in obj2.solve_options['future']:
                    if "org_plot_results" in locals():
                        obj2.solve_options['future']['plot_results'] = org_plot_results 

            obj2._write_results = org_write_results
            
        return result,fname,cancel_run
        
    
    
    def create_solutions(self,future_years,scenarios,ci_intervals,historic_solution,scen_dict,cold_snap_shift=None,filename="",
                         run_parallel=None,num_cpu=None,overwrite=True):
        """
        
        Write solution files for a broad range of future years, ssp scenarios, 
        and confidence intervals. A historic solution must be run first. This is 
        much quicker than "create_scenario" and uses
        algebraic solution that shifts mean and stretches standard deviation
        equally to exactly meet the average of the 10 and 50 year IPCC events.

        Parameters
        ----------
        future_years : array-like
            list of integers years > 2014
        scenarios : array-like
            list of valid SSP names ["SSP119","SSP126","SSP245","SSP370","SSP585"] are the only valid entries
        ci_intervals : array-like maximum of 3 elements
            list of confidence intervals to calculate ['5%','50%', and '95%'] are the
            only valid entries
        historic_solution : str
            Must provide a path to a solution_file generated by "write_solution"
            of the historic optimization solution
        scen_dict : dict
            Must provide a dictionary with keys = {historic, ssp names...} of 
            numpy.poly1d climate surface temperature change polynomials 
        cold_snap_shift : dict, optional
            NOT YET IMPLEMENTED 
            manually input equivalent to the IPCC heat wave shift table. If none, 
            cold snaps will be kept at their historical values
        filename : str, optional
            string to prepend to the filename output The default is "".
        run_parallel : bool : Optional : default = None
            None - use the original objects input for run_parallel
            Value - overide and use a local run_parallel with the 
            number of processors from num_processor input
        num_cpu : int : Optional : default = None
            None = use the original objects input for num_cpu
            Value - override with a new number of processors
        overwrite : bool : Optional : default = True
            If true overwrite existing files, otherwise,
            do not overwrite, just move on to the next file that doesn't exist
        

        Raises
        ------
        ValueError
            Whenever the function is called multiple times or "create_scenario"
            has already been run.

        Returns
        -------
        results : dictionary of dictionary of dictionary with ci, year, and scenario
            as keys. with mew.stats.Alter type objects for actual realizations
            THIS IS NOT THAT USEFUL
            
        filenames : list
            a list of all the filenames with MEWS solution parameters written.

        """
        if run_parallel is None:
            run_parallel = self._run_parallel
        if num_cpu is None:
            num_cpu = self._num_cpu 
            
        num_cpu = filter_cpu_count(num_cpu)
        
        if run_parallel:
            try:
                import multiprocessing as mp
                pool = mp.Pool(num_cpu)
            except:
                warn("Something went wrong with importing "
                            +"multiprocessing and creating a pool "
                            +"of asynchronous processes reverting to"
                            +" non-parallel run!",UserWarning)
                run_parallel = False
        
        results = {}

        filenames=[]
        run_canceled = []
        if self._create_scenario_has_been_run:
            raise ValueError("This function cannot be run if create_scenario "+
                             " has overwritten the historic solution on "+
                             "initialization. The best use case is to run the"+
                             " initiazation to produce historic fit (takes a "+
                             "long time), use obj.write_solution, and then"+
                             " reinitialize using a solution file multiple time"+
                             " and running this routine 1 time.")

        sol_dir = os.path.dirname(historic_solution)
        
        obj = deepcopy(self)

        # still working on this!
        # this is the longest step.
        for syear in future_years:
            results[syear] = {}
            for scen in scenarios:
                results[syear][scen] = {}
                for cii in ci_intervals:
                    
                    tup = (syear,
                            scen,
                            cii,
                            obj,
                            historic_solution,
                            scen_dict,
                            cold_snap_shift,
                            sol_dir,
                            filename,
                            overwrite)
                    if run_parallel:
                        results[syear][scen][cii] = pool.apply_async(self._parallel_create_solution_func,
                                                          args=tup)
                    else:
                        results[syear][scen][cii],fname,cancel_run = self._parallel_create_solution_func(*tup)
                        filenames.append(fname)
                        run_canceled.append(cancel_run)
                    
        if run_parallel:
            presults = {}
            for syear in future_years:
                presults[syear] = {}
                for scen in scenarios:
                    presults[syear][scen] = {}
                    for cii in ci_intervals:
                        poolObj = results[syear][scen][cii]
                        try:
                            presults[syear][scen][cii],fname,cancel_run = poolObj.get()
                            filenames.append(fname)
                            run_canceled.append(cancel_run)
                        except AttributeError:
                            raise AttributeError("The multiprocessing module will not"
                                                 +" handle lambda functions or any"
                                                 +" other locally defined functions!")

        # TODO - add run_canceled to output. Avoiding this for version 1.1 
        # so that interfaces do not change.
        return results,filenames
        
    def create_scenario(self,scenario_name,year,climate_temp_func,
                        num_realization=1,climate_baseyear=None,increase_factor_ci="50%",
                        cold_snap_shift=None,solution_file="",random_seed=None,output_dir=None):
        
        """
        >>> obj.create_scenario(scenario_name,start_year,num_year, climate_temp_func,
                                num_realization,climate_baseyear,increase_factor_ci)
        
        Places results into self.extreme_results and self.ext_obj
        
        This function extends the heat wave analysis for ExtremeTemperatureWaves
        into the future for a specific shared socioeconomic pathway (SSP) scenario
        and a specific confidence interval (ci) track (5%, 50%, or 95%) for
        the heat wave intensity and frequency parameters taken from Figure SPM6
        
        Parameters
        ----------
        
        scenario_name : str 
            A string indicating the name of a scenario
        
        year : int 
            A year (must be >= 2014) that is the starting point of the analysis
        
        climate_temp_func : dict of func 
            Must contain key 'historic' and 'scenario_name' so that
            the historic baseline and increase in temperature can be 
            quantified.
            self.use_global = True
                A function that provides a continuous change in temperature that
                is scaled to a yearly time scale. time = 2020 is the begin of the 
                first year that is valid for the function.
                
                No valid return for values beyond 4 C due to lack of data from
                IPCC for higher values.
            self.use_global = False
                A function that provides a continuous change in temperature that
                baselined from 2014. Input must be years - 2014.
            
        num_realization : int : Optional : Default = 1
            Number of times to repeat the entire analysis of each weather file 
            so that stochastic analysis can be carried out.
            
        climate_baseyear : int : optional : Default =None
            Only used when self.use_global = False
            Required if use_global = True. Should be 2014 for CMIP6.
            
        increase_factor_ci : str : optional : Default = "50%"
            Choose one of "[5%, 50%, 95%]" indicating what values out of table SPM6
            to use 5% is for the lower bound 95% Confidence inverval (CI)
                   50% is for the mean
                   95% is the upper bound of the 95% CI
            all other strings will raise a ValueError
            
        cold_snap_shift : dict : optional : Default = None
            NOT YET FUNCTIONAL - must calculate cold snap shift and assure
                                 cold snap thresholds become part of the 
                                 optimization problem.
            MEWS does not include a shift in statistics for cold snaps due to
            lack of information. This feature allows the user to enter a dictionary 
            of the required form:
            
            cold_snap_shift = {'temperature':{'10 year': <10 year event 
                                                          shift in degC + means 
                                                          less severe cold snaps>,
                                              '50 year': <50 year event...},
                               'frequency':{'10 year': <10 year multiplication
                                                        factor on frequency of 
                                                        events < 1 decreases 
                                                        frequency of events>,
                                            '50 year': <50 year ...}}
            
        solution_file : str : optional : Default = ""
            Path and name of a file that contains a solution either historic
            or for the current scenario. This enables by-passing the optimization
            problem that takes significant computational resources so that
            MEWS can implement a solution already computed quickly
            
        random_seed : int : optional : Default = None
            Use this input if this function is being called repeatedly to avoid
            using the same original random seed every time.
            
        output_dir : str :optional : Default = None
            indicate where to put the output .epw results from the creation of 
            this scenario.
            If not entered use the original value provided results_folder input
            to the instantiation of this object.
            
        Returns
        -------    
        results_dict : dict : 
            A dictionary whose key is the year analyzed
            
        
        """
        if not output_dir is None:
            self._results_folder = output_dir
        
        if not random_seed is None:
            self._random_seed = random_seed
            np.random.seed(random_seed)
        
        self.extreme_results[scenario_name] = {}
        self.extreme_delstats[scenario_name] = {}
        
        
        if not increase_factor_ci in _DeltaTransition_IPCC_FigureSPM6._valid_increase_factor_tracks:
            raise ValueError("The input 'increase_factor_ci' must be a string with one of the following three values: \n\n{0}".format(
                str(_DeltaTransition_IPCC_FigureSPM6._valid_increase_factor_tracks.keys())))
        
        
        ext_obj_dict = {}
        results_dict = {}
        
        if not scenario_name in self.future_solve_obj:
            self.future_solve_obj[scenario_name] = {}
            
        if not increase_factor_ci in self.future_solve_obj[scenario_name]:
            self.future_solve_obj[scenario_name][increase_factor_ci] = {}
        

            
        if not year in self.future_solve_obj[scenario_name][increase_factor_ci]:
            self.future_solve_obj[scenario_name][increase_factor_ci][year] = {}
        
        self.extreme_delstats[scenario_name][increase_factor_ci] = {}
        

        # If you are using a solution_file, then self.stats is changed
        # by the function the "delta" and "del" values below will be zero.

        (transition_matrix, 
         transition_matrix_delta,
         del_E_dist,
         del_delTmax_dist,
         cancel_run) = self._create_transition_matrix_dict(self.stats,
                                                                 climate_temp_func,
                                                                 year,
                                                                 climate_baseyear,
                                                                 scenario_name,
                                                                 increase_factor_ci, 
                                                                 cold_snap_shift,
                                                                 solution_file)
                                                           
        if cancel_run:
            results_dict[year] = None
            ext_obj_dict[year] = None
            self._verification_data[year] = None
        else:
            
            if self.use_global == False:
                base_year = climate_baseyear
            else:
                base_year = None
            
            if hasattr(self,"_DO_NOT_RERUN_PARALLEL_"):
                old_run_parallel = self._run_parallel
                self._run_parallel = False
                
            # now initiate use of the Extremes class to unfold the process
            ext_obj_dict[year] = super().__init__(year,
                     {'func':trunc_norm_dist, 'param':self.stats['heat wave']},
                     del_delTmax_dist,
                     {'func':trunc_norm_dist, 'param':self.stats['cold snap']},
                     None,
                     transition_matrix,
                     transition_matrix_delta,
                     self._weather_files,
                     num_realizations=num_realization,
                     num_repeat=1,
                     use_cython=True,
                     column='Dry Bulb Temperature',
                     tzname=None,
                     write_results=self._write_results,
                     results_folder=self._results_folder,
                     results_append_to_name=scenario_name,
                     run_parallel=self._run_parallel,
                     min_steps=HOURS_IN_DAY,
                     test_shape_func=False,
                     doe2_input=self._doe2_input,
                     random_seed=self._random_seed,
                     max_E_dist={'func':trunc_norm_dist,'param':self.stats['heat wave']},
                     del_max_E_dist=del_E_dist,
                     min_E_dist={'func':trunc_norm_dist,'param':self.stats['cold snap']},
                     del_min_E_dist=None,
                     current_year=int(year),
                     climate_temp_func=climate_temp_func[scenario_name],
                     averaging_steps=HOURS_IN_DAY,
                     use_global=self.use_global,
                     baseline_year=base_year,
                     norms_hourly=self.df_norms_hourly,
                     num_cpu=self._num_cpu,
                     test_markov=self._test_markov,
                     confidence_interval=increase_factor_ci,
                     overwrite_existing=self._overwrite_existing)
            results_dict[year] = self.results
            
            if hasattr(self,"_DO_NOT_RERUN_PARALLEL_"):
                self._run_parallel = old_run_parallel
        
            self.extreme_delstats[scenario_name][increase_factor_ci][year] = {"E":del_E_dist,"delT":del_delTmax_dist}
            # this is happening 
            self._verification_data[year] = self._delTmax_verification_data
                
                
        self.extreme_results[scenario_name][increase_factor_ci] = results_dict
        self.ext_obj[scenario_name] = ext_obj_dict
        self._create_scenario_has_been_run = True

        return results_dict
    
    def get_results(self, scenario, year, confidence_interval):
        
        """
        obj.get_results(scenario,year,confidence_interval)
        
        This function retrieves results from an ExtremeTemperatureWaves object
        that has had the 'create_scenario' method run at least once. It provides
        a strongly checked method for getting results without having to. use the
        3-deep dictionary structure used by ExtremeTemperatureWaves
        
        Inputs
        ------
        
        scenario : str : a valid IPCC SSP scenario that has already been run.
                     valid scenario names include {0}
        
        year : int : a year between 1900 and 2100 that has already been run
                     using 'create_scenario'
        
        confidence_interval : str : A confidence interval string designation.
                    valid CI names include {1}
        
        Returns
        -------
        
        dict - a dictionary containing the mews.weather.alter.Alter objects
               associated with the selected (scenario,year,confidence_interval)
               these objects contain all of the alterations that allow exploration
               of specific heat waves.
               
        Raises
        ------
        
        TypeError - One of the inputs is an invalid type
        
        ValueError - One of the inputs is an invalid value
        
        """.format(str(_DeltaTransition_IPCC_FigureSPM6._valid_scenario_names),
        _DeltaTransition_IPCC_FigureSPM6._valid_increase_factor_tracks)
                                                                     
        if not isinstance(scenario,str):
            raise TypeError("The 'scenario' input must be a string.")
            
        if not isinstance(confidence_interval,str):
            raise TypeError("The 'confidence_interval' input must be a string.")
        
        if not scenario in _DeltaTransition_IPCC_FigureSPM6._valid_scenario_names:
            raise ValueError("The scenario {0} is not a valid scenario name. The only values permitted are: ".format(scenario) +
                             + str(_DeltaTransition_IPCC_FigureSPM6._valid_scenario_names))
            
        if not isinstance(year,(int)):
            raise TypeError("The 'year' input must be an integer!")
            
        if (year < 1900 or year > 2100):
            raise ValueError("The 'year' input is invalid. MEWS only can analyze 1900 to 2100!")
        
        if not confidence_interval in _DeltaTransition_IPCC_FigureSPM6._valid_increase_factor_tracks:
            raise ValueError("The input 'confidence_interval' must be a string with one of the following three values: \n\n{0}".format(
                str(_DeltaTransition_IPCC_FigureSPM6._valid_increase_factor_tracks.keys())))
        
        
        message = "You must run the 'create_scenario' to generate results"
        
        if self._create_scenario_has_been_run:
            if scenario in self.extreme_results:
                scen_results = self.extreme_results[scenario]
                
                if confidence_interval in scen_results:
                    
                    if year in scen_results:
                        return scen_results[year]
                    else:
                        raise ValueError("The year {0:d} has not yet been analyzed. ".format(year) 
                                         + message+" for {0:d}".format(year))
                
                else:
                    raise ValueError("The confidence interval position {0} has not yet been run. ".format(confidence_interval) +
                                     + message + " for confidence interval position {0}.".format(confidence_interval))
                
            else:
                raise ValueError("The scenario {0} has not yet been analyzed. ".format(scenario)
                                 + message + " for scenario {0}!".format(scenario))
        else:
            raise ValueError(message+"!")
    
    def _check_solve_options(self,solve_options):
        
        if solve_options is None:
            solve_options = DEFAULT_SOLVE_OPTIONS
        
        if not isinstance(solve_options, dict):
            raise TypeError("The input 'solve_options' must be a dictionary of a specific type")
            
        for tim, subdict in solve_options.items():
            if not tim in VALID_SOLVE_OPTION_TIMES:
                raise ValueError("The input 'solve_options' must be a dictionary"+
                                 " with specific keys. An entry "+
                                 "'{0}' is not part of the valid entries list = \n\n{1}".format(tim, str(VALID_SOLVE_OPTION_TIMES)))
            for opt, val in subdict.items():
                if not opt in VALID_SOLVE_OPTIONS:
                    raise ValueError("The input 'solve_options['{0}'] has an invalid solve option = '{1}'".format(tim,opt) +
                                     "\n\nValid options are:\n\n{0}".format(",\n".join(VALID_SOLVE_OPTIONS)))
            # REQUIRED solve options must be given defaults here.
            for opt in [VALID_SOLVE_INPUTS[16]]:
                if not opt in subdict:
                    subdict[opt] = DEFAULT_SOLVE_OPTIONS[tim][opt]
                
        
        return solve_options
    
    def _solve_historic_distributions(self,stats, solve_options, frac_hours_per_year, solution_file):
        # solve options unpacked
        
        # loop over heat waves/ cold snaps
        new_stats = {}
        for wt1,wt2 in WAVE_MAP.items():
            new_stats[wt1] = {}
        
        # This simply allows the user to evaluate a solution 
        # and overrides performing an optimization
        if len(solution_file) != 0:
            override_stats = self.read_solution(solution_file)
            stats = override_stats
        else:
            override_stats = None
                
        random_seed = self._random_seed
        
        # These defaults do not match all of the defaults in solve.py
        # there are specific differences that are intentional and they 
        # should be kept separate.
        default_vals = DEFAULT_SOLVE_OPTIONS['historic']
        
        historic_time_interval = int(stats['heat wave'][1]['historic time interval'])
        
        # bring in values assigned by the user. Use defaults for unassigned values.
        opt_val = _mix_user_and_default(default_vals, 'historic', solve_options)

        fig_out_dir_name = os.path.dirname(opt_val['out_path'])
        fig_out_base_name = os.path.basename(opt_val['out_path'])
        fig_is_dir = os.path.isdir(opt_val['out_path'])
        fig_dir_exists = os.path.exists(fig_out_dir_name)
        
        obj_solve = None
        sobj_dict = {}
        for month in range(1,13):
            
            hist0 = {}
            durations0 = {}
            param0 = {}
            for wname1,wname2 in WAVE_MAP.items():
                hist0[wname2] = stats[wname1][month]['historical temperatures (hist0)'] 
                durations0[wname2] = stats[wname1][month]['historical durations (durations0)']
                param0[wname2] = stats[wname1][month]
            
            
            
            
            out_path_month = _create_figure_out_name(fig_out_dir_name,
                                                        fig_out_base_name,
                                                        month,
                                                        fig_is_dir,
                                                        fig_dir_exists)
            extra_columns = _process_extra_columns(opt_val['extra_output_columns'],month)
            
            if not override_stats is None:
                # must override some solve options (decay function does not change with month)
                for wt,wave_type in zip(['cs','hw'],['cold snap','heat wave']):
                    if 'decay function' in stats[wave_type][month]:
                        opt_val['decay_func_type'][wt] = stats[wave_type][month]['decay function']
                    else:
                        opt_val['decay_func_type'][wt] = None
            
            if obj_solve is None:
                
                if not override_stats is None:
                    opt_val['x_solution'] = _assemble_x_solution(override_stats, month)
                
                obj_solve = SolveDistributionShift(opt_val['num_step'], 
                                       param0, 
                                       random_seed, 
                                       hist0, 
                                       durations0, 
                                       opt_val['delT_above_shifted_extreme'], 
                                       historic_time_interval,
                                       int(frac_hours_per_year[month-1] * HOURS_IN_YEAR),
                                       problem_bounds=opt_val['problem_bounds'],
                                       ipcc_shift={'cs':None,'hw':None},
                                       decay_func_type=opt_val['decay_func_type'],
                                       use_cython=opt_val['use_cython'],
                                       num_cpu=opt_val['num_cpu'],
                                       plot_results=opt_val['plot_results'],
                                       max_iter=opt_val['max_iter'],
                                       plot_title=opt_val['plot_title'],
                                       out_path=out_path_month,
                                       weights=opt_val['weights'],
                                       limit_temperatures=opt_val['limit_temperatures'],
                                       min_num_waves=opt_val['min_num_waves'],
                                       x_solution=opt_val['x_solution'],
                                       test_mode=opt_val['test_mode'],
                                       num_postprocess=opt_val['num_postprocess'],
                                       extra_output_columns=extra_columns,
                                       org_samples={"Temperature":{'hw':stats['heat wave'][month]['historical delTmax'],
                                                                   'cs':stats['cold snap'][month]['historical delTmax']},
                                                    "Duration":{'hw':stats['heat wave'][month]['historical durations'],
                                                                'cs':stats['cold snap'][month]['historical durations']}})

                if opt_val['test_mode']:
                    # this makes all other runs just be evaluations when 
                    # we just want to run quick tests.
                    if not override_stats is None:
                        x_solution = opt_val['x_solution']
                    else:
                        x_solution = obj_solve.optimize_result.x
                else:
                    x_solution = None
                
                param = obj_solve.param
            else:
        
                if not override_stats is None:
                    x_solution = _assemble_x_solution(override_stats, month)
                
                if month == 12:
                    write_csv = True
                else:
                    write_csv = False
                    
                # only reassign values that change by month (and x_solution for testing purposes
                # to reduce run time)
                inputs = {"param0":param0,
                          "hist0":hist0,
                          "durations0":durations0,
                          "hours_per_year":int(frac_hours_per_year[month-1] * HOURS_IN_YEAR),
                          "x_solution":x_solution,
                          "out_path":out_path_month,
                          "extra_output_columns":extra_columns,
                          "org_samples":{"Temperature":{'hw':stats['heat wave'][month]['historical delTmax'],
                                                      'cs':stats['cold snap'][month]['historical delTmax']},
                                       "Duration":{'hw':stats['heat wave'][month]['historical durations'],
                                                   'cs':stats['cold snap'][month]['historical durations']}}}
                param = obj_solve.reanalyze(inputs, write_csv)
            
            for wt1, wt2 in WAVE_MAP.items():
                new_stats[wt1][month] = param[wt2]
            sobj_dict[month] = deepcopy(obj_solve)
            
            
        self.hist_obj_solve = sobj_dict
        return new_stats
    

            
        
        
    
    def _real_value_stats(self,wave_type,scenario_name,year,ci_interval,stat_name,duration):
        
        
        
        if not isinstance(duration,np.ndarray):
            raise TypeError("The duration input must be a numpy array of values!")
        
        
        stats = self.stats[wave_type]

        for month,subdict in stats.items():
            
            if stat_name == "delT":
                minval = subdict['hist min extreme temp per duration']
                maxval = subdict['hist max extreme temp per duration']
                param = subdict['extreme_temp_normal_param']
                mu = param['mu']
                sig = param['sig']
                norm0 = subdict['normalizing extreme temp']
                slope = subdict['normalized extreme temp duration fit slope']
                interc = subdict['normalized extreme temp duration fit intercept']
                
            elif stat_name == "E":
                minval = subdict['min energy per duration']
                maxval = subdict['max energy per duration']
                param = subdict['energy_normal_param']
                mu = param['mu']
                sig = param['sig']
                norm0 = subdict['normalizing energy']    
                slope = subdict['energy linear slope']
                interc = 0.0
            else:
                raise ValueError("Only 'delT' or 'E' are accepted inputs for 'stat_name'")
            norm_d = subdict['normalizing duration']
            
            # The original average Tmax
            values = {'mu':mu,'mu+sig':mu+sig,"a":-1,"b":1}
            
            delstat = self.extreme_delstats[scenario_name][ci_interval][year][stat_name][month]
            
            del_values = {'mu':delstat["del_sig"],
                          'mu+sig':delstat["del_mu"]+delstat["del_sig"]
                          ,"a":delstat["del_a"],"b":delstat["del_b"]}
            result = {}
            
            for key,val in values.items():
                per_dur = inverse_transform_fit(val,maxval,minval) 
                res_0 = per_dur * norm0 * (slope * duration/norm_d + interc)
                if len(self.extreme_delstats) > 0:
                    
                
                    per_dur_del = inverse_transform_fit(val + del_values[key],
                                                maxval,
                                                minval)
                    res_del = per_dur_del * norm0 * (slope * duration/norm_d + interc)
                else:
                    res_del = np.nan
                
                
                    
                
                result[key] = {"0":res_0,"del":res_del}
        
            return result
            
            
        
    
    def _create_transition_matrix_dict(self,stats,climate_temp_func,year,
                                       climate_baseyear,scenario,increase_factor_ci,
                                       cold_snap_shift,solution_file):
        
        """
        Here is where the delta T, delta E and delta Markov matrix values are calculated
        using class _DeltaTransition_IPCC_FigureSPM6
        
        """
        
        transition_matrix = {}
        transition_matrix_delta = {}
        del_E_dist = {}
        del_delTmax_dist = {}
        cancel_run = False
        
        # This simply allows the user to evaluate a solution 
        # and overrides performing an optimization
        if len(solution_file) != 0:
            override_stats = self.read_solution(solution_file)
            stats = override_stats
            
        else:
            override_stats = None
        
        
        if not self.use_global:
            # need a new copy that will not be altered.
            solve_options = deepcopy(self.solve_options)    
            fig_out_dir_name = os.path.dirname(solve_options['future']['out_path'])
            fig_out_base_name = os.path.basename(solve_options['future']['out_path'])
            fig_is_dir = os.path.isdir(solve_options['future']['out_path'])
            fig_dir_exists = os.path.exists(fig_out_dir_name)
        else:
            solve_options = None
        
        
        
        # gather the probability of a heat wave for each month.
        prob_hw = {}
        for month,stat in stats['heat wave'].items():
            prob_hw[month] = stat['hourly prob of heat wave']
            
        solution_obtained = False

        obj = None
        # Form the parameters needed by Extremes but on a monthly basis.
        solve_obj = None
        for hot_tup,cold_tup in zip(stats['heat wave'].items(), stats['cold snap'].items()):
            
            month = cold_tup[0]
            cold_param = cold_tup[1]
            hot_param = hot_tup[1]
            
            if len(solution_file) > 0:
                solve_options['future']['x_solution'] = _assemble_x_solution(stats, month)
            
            Pwh = hot_param['hourly prob of heat wave']
            Pwsh = hot_param['hourly prob stay in heat wave']
            Pwc = cold_param['hourly prob of heat wave']
            Pwsc = cold_param['hourly prob stay in heat wave']
            transition_matrix[month] = np.array([[1-Pwh-Pwc,Pwc,Pwh],
                                                 [1-Pwsc, Pwsc, 0.0],
                                                 [1-Pwsh, 0.0, Pwsh]])
            # just to speed up testing! This makes everything after month 1
            # just an evaluation of month 1 instead of an optimization for a
            # new month.
            if not self.use_global:
                if solution_obtained:
                    if 'future' in solve_options:
                        if 'test_mode' in solve_options['future']:
                            if solve_options['future']['test_mode']:
                                if override_stats is None:
                                    solve_options['future']['x_solution'] = obj.obj_solve.optimize_result.x
                                else:
                                    pass # this has already been assigned above.
                                
                if not 'future' in solve_options:
                    solve_options[VALID_SOLVE_OPTION_TIMES[1]] = {}
    
                if not VALID_SOLVE_INPUTS[23] in solve_options[VALID_SOLVE_OPTION_TIMES[1]]:
                    
                    solve_options[VALID_SOLVE_OPTION_TIMES[1]][VALID_SOLVE_INPUTS[23]] = {} 
                
                
                extra_columns = _process_extra_columns(solve_options[VALID_SOLVE_OPTION_TIMES[1]][VALID_SOLVE_INPUTS[23]], 
                                                       month, year, scenario, increase_factor_ci)  


                solve_options['future']['out_path'] = _create_figure_out_name(fig_out_dir_name, 
                                        fig_out_base_name, 
                                        month, 
                                        fig_is_dir, 
                                        fig_dir_exists, 
                                        scenario, 
                                        increase_factor_ci, 
                                        year, 
                                        is_historic=False)
                
                
            else:
                extra_columns = None

            
            # Due to not finding information in IPCC yet, we assume that cold events
            # do not increase in magnitude or frequency.
            obj = _DeltaTransition_IPCC_FigureSPM6(hot_param,
                                               cold_param,
                                               climate_temp_func,
                                               year,
                                               self._hours_per_year[month-1],
                                               self.use_global,
                                               climate_baseyear,scenario,
                                               self,
                                               increase_factor_ci,
                                               prob_hw,
                                               self._delT_ipcc_min_frac,
                                               month,
                                               self._random_seed,
                                               solve_options,
                                               cold_snap_shift,
                                               write_csv=self._write_results,
                                               extra_columns=extra_columns,
                                               solve_obj=solve_obj)
            
            if not obj.cancel_run:
                
                solve_obj = obj.obj_solve
                solution_obtained = True
                
                self.future_solve_obj[scenario][increase_factor_ci][year][month] = deepcopy(obj)
    
                transition_matrix_delta[month] = obj.transition_matrix_delta
                del_E_dist[month] = obj.del_E_dist
                del_delTmax_dist[month] = obj.del_delTmax_dist
                self.ipcc_results['ipcc_fact'][month] = obj.ipcc_fact
            else:
                cancel_run = True
                
            # IMPORTANT! - THIS IS HOW CHANGES ARE TRANSFERRED WHEN USING
            # The solution file!
            
            self.stats = stats


        return (transition_matrix,
                transition_matrix_delta,
                del_E_dist,
                del_delTmax_dist,
                cancel_run)
        
    
    def _read_and_curate_NOAA_data(self,station,year,unit_conversion,norms_unit_conversion,use_local=False):
    
        """
        read_NOAA_data(station,year=None,use_local=False)
        
        Reads National Oceanic and Atmospheric Association (NOAA)
        daily-summaries data over a long time period and
        NOAA climate norms hourly data from 1991-2020 from the url's
        self.norms_url and self.daily_url. Creates self.NOAA_data dataframe
        
        
        This funciton is highly specific to the data formats that worked on 
        10/14/2021 for the NOAA data repositories and will likely need significant
        updating if new data conventions are used by NOAA.
        
        Parameters
        ----------
        station : str or dict
            str: Must be a valid weather station ID that has a valid representation 
            for both the self.norms_url and self.daily_url web locations
            the error handling provides a list of valid station ID's that
            meets this criterion if a valid ID is not provded.
            
            dict: must be a dictionary of the following form
                {'norms':<str with path to norms file>,
                 'summaries':<str with path to the daily summaries file>}
        
        year : int 
            The year to be assigned to the climate norms data for 
            dataframe purposes. This year is later overlaid accross
            the range of the daily data
            
        unit_conversion : tuple
            a tuple allowing index 0 to be a scale factor on the units in the
            daily summary data and index 1 to be an offset factor. Must convert
            to degrees Celcius
            
        norms_unit_conversion : tuple
            a tuple allowing index 0 to be a scale factor on the units in the
            climate norms data and index 1 to be an offset factor. Must convert
            to degrees Celcius            
        
        use_local 
            If MEWS is not reading web-urls use this to indicate to look
            for local files at "station" file path location. 
            if true 
                    
        Returns
        -------
        NOAA_data : DataFrame 
            Contains the daily summaries with statistical summary
            daily data from the climate normals overlaid so that heat and cold 
            wave comparisons can be made.
            
        self.mid_date : datetime - provides the middle date of the range
            of time convered by the daily summaries being analyzed.
            
        
        
        """
        if not self._proxy is None:
            os.environ['http_proxy'] = self._proxy 
            os.environ['HTTP_PROXY'] = self._proxy
            os.environ['https_proxy'] = self._proxy
            os.environ['HTTPS_PROXY'] = self._proxy
        
        
        df_temp = []
        for url,isdaily in zip([self.daily_url,self.norms_url],[True,False]):
    
            if isinstance(station,dict):
                
                if isdaily:
                    df = pd.read_csv(station['summaries'],low_memory=False)
                else:
                    df = pd.read_csv(station['norms'],low_memory=False)
                
            elif isinstance(station,str):
    
                if station[-4:] == ".csv":
                    ending = ""
                else:
                    ending = ".csv"
            
                if use_local:
                    if isdaily:
                        df = pd.read_csv(station + ending,low_memory=False)
                    else:
                        if len(ending) == 0:
                            station = station[:-4]
                            ending = ".csv"
                        df = pd.read_csv(station + "_norms" + ending)
            
                else:
                    
                    # give a good error if the http and station number are not working.
                    try:
                        filestr = urllib.request.urlopen(urllib.parse.urljoin(url,station+ending)).read().decode()
                    except urllib.error.HTTPError as exc:
                        exc.msg = "The link to \n\n'" + exc.filename + "'\n\ncould not be found."
                        raise(exc)
                    except Exception as exc_unknown:
                        raise(exc_unknown) 
                    
                    df = pd.read_csv(io.StringIO(filestr))
            else:
                raise TypeError("The station input must be a dict with two entries 'summaries', and 'norms' " +
                                "or a string that indicates a valid station ID for NOAA summaries and " +
                                "and norms data")
        
            
            if isdaily:
                # specific processing for daily summary data.
                df['DATE'] = pd.to_datetime(df['DATE'])
                df.index = df['DATE']
        
                # numbers provided are degrees Celcius in tenths.
                df['TMAX'] = unit_conversion[0] * df['TMAX'] + unit_conversion[1]
                df['TMIN'] = unit_conversion[0] * df['TMIN'] + unit_conversion[1]
                meta_data = {'STATION':df['STATION'].iloc[0],
                             'LONGITUDE':df['LONGITUDE'].iloc[0],
                             'LATITUDE':df['LATITUDE'].iloc[0],
                             'ELEVATION':df['ELEVATION'].iloc[0],
                             'NAME':df['NAME'].iloc[0]}
                
                df_new = df[['TMIN','TMAX']]
                self.meta = meta_data
                
                def time_in_year_decimal(timestamp):
                    year = timestamp.year
                    day = timestamp.dayofyear
                    numday = pd.Timestamp(year,12,31).dayofyear
                    return year + day/numday
                    
                
                
                # Calculate the middle date in a decimal form for years
                # for interpolation in _DeltaTransition_IPCC_FigureSPM6
                mid_timestamp = pd.to_datetime(df.index[0].to_datetime64() + 
                                    (df.index[-1].to_datetime64()-
                                     df.index[0].to_datetime64())/2.0)
                
                self.hw_beg_date = time_in_year_decimal(df.index[0])
                self.hw_mid_date = time_in_year_decimal(mid_timestamp)
                self.hw_end_date = time_in_year_decimal(df.index[-1])
             
            else:
                df.index = pd.to_datetime(df["DATE"].apply(lambda x: str(year)+"-"+x))
                keep = ["HLY-TEMP-10PCTL","HLY-TEMP-NORMAL","HLY-TEMP-90PCTL",
                        "HLY-DEWP-10PCTL","HLY-DEWP-NORMAL","HLY-DEWP-90PCTL",
                        "HLY-PRES-10PCTL","HLY-PRES-NORMAL","HLY-PRES-90PCTL"]
                df = df[keep]
                df["HLY-RELH-10PCTL"] = df[["HLY-TEMP-10PCTL","HLY-DEWP-10PCTL"]].apply(lambda x: relative_humidity(x[1],x[0]),axis=1)
                df["HLY-RELH-NORMAL"] = df[["HLY-TEMP-NORMAL","HLY-DEWP-NORMAL"]].apply(lambda x: relative_humidity(x[1],x[0]),axis=1)
                df["HLY-RELH-90PCTL"] = df[["HLY-TEMP-90PCTL","HLY-DEWP-90PCTL"]].apply(lambda x: relative_humidity(x[1],x[0]),axis=1)
                
                # This is needed and goes to the Extremes class so that heat wave stats continue
                # to be added against the norm rather than the actual weather data.
                # you must convert to degrees celcius
                df_C = df[["HLY-TEMP-10PCTL","HLY-TEMP-NORMAL","HLY-TEMP-90PCTL"]].apply(lambda x: norms_unit_conversion[0] * x + norms_unit_conversion[1])
                self.df_norms_hourly = df_C
                
                df_max = df.resample('1D').max()
                df_min = df.resample('1D').min()
                df_avg = df.resample('1D').mean()
                
                # now reconstruct the comparisons to 90% TMAX and 10% TMIN needed for hot and cold waves. No other data is needed for now 
                df_new = pd.concat([df_max["HLY-TEMP-90PCTL"],df_avg["HLY-TEMP-NORMAL"],df_min["HLY-TEMP-10PCTL"],df_max["HLY-TEMP-10PCTL"],df_min["HLY-TEMP-90PCTL"],
                                    df_max["HLY-RELH-90PCTL"],df_avg["HLY-RELH-NORMAL"],df_min["HLY-RELH-10PCTL"],df_max["HLY-RELH-10PCTL"],df_min["HLY-RELH-90PCTL"],
                                    df_max["HLY-PRES-90PCTL"],df_avg["HLY-PRES-NORMAL"],df_min["HLY-PRES-10PCTL"],df_max["HLY-PRES-10PCTL"],df_min["HLY-PRES-90PCTL"]],axis=1)
                df_new.columns = ["TMAX_B","TAVG_B","TMIN_B","TMAXMIN_B","TMINMAX_B",
                                  "HMAX_B","HAVG_B","HMIN_B","HMAXMIN_B","HMINMAX_B",
                                  "PMAX_B","PAVG_B","PMIN_B","PMAXMIN_B","PMINMAX_B"]
                # convert from degrees Fahrenheit TODO - are all norms provided in Fahrenheit???
                df_new[["TMAX_B","TAVG_B","TMIN_B","TMAXMIN_B","TMINMAX_B"]] = df_new[[
                    "TMAX_B","TAVG_B","TMIN_B","TMAXMIN_B","TMINMAX_B"]].apply(
                        lambda x: norms_unit_conversion[0] * x + norms_unit_conversion[1])
                        
            df_temp.append(df_new)
        df_daily = df_temp[0]
        df_norms = df_temp[1]
        
        # Assure no repeat dates are present. DAte gaps are addressed in _extend_boundary_df_to_daily_range.
        for df_,dtype in zip([df_daily,df_norms],['daily summaries', 'climate norms']):
        # assure no duplicate datres occur:
            first_duplicate_index = df_.index.duplicated().argmax()
            if first_duplicate_index != 0:
                raise ValueError("The following date is duplicated in the {1}.\n\n {0}".format(
                    str(df_.index[first_duplicate_index]),dtype)
                                 +"\n\nThis is not allowed. Please clean the data!")
        
        # projects the norms on the daily summaries date range it a repetitive cycle.
        self.NOAA_data = self._extend_boundary_df_to_daily_range(df_norms,df_daily)
    
    def _check_NOAA_url_validity(self):
        def is_valid(url):
            """
            Checks whether `url` is a valid URL.
            """
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        
        err_msg = ("MEWS: Test to "+
                    "see if this url is still valid on an internet connected"+
                    " computer for NOAA and contact them if it is no longer"+
                    " working and update the source code in the "+
                    "ExtremeTemperatureWave.daily_url or .norms_url constants!")
        
        if not is_valid(self.daily_url):
            raise urllib.error.HTTPError(self.daily_url,None,err_msg)
        elif not is_valid(self.norms_url):
            raise urllib.error.HTTPError(self.norms_url,None,err_msg)
    
            
    def _extend_boundary_df_to_daily_range(self,df_norms,df_daily):
        """
        This function overlays the climate norms 90th percentile and all other columns 
        over the daily data so that heat wave statistics can be quantified. 
        """
        
        df_combined = df_daily
        
        unique_year = df_daily.index.year.unique()
        
        # verify that every year is consecutive
        if np.diff(unique_year).sum() + 1 != len(unique_year):
            pass
            # THIS WAS AN OVERLY RESTRICTIVE TEST
            #raise ValueError("The daily data from NOAA has one or more gaps of a year! Please use a station that has continuous data!")
        
    
        
        feb_28_day = pd.Timestamp(2020,2,28).day_of_year
    
        df_list = []
    
        for year in unique_year:

            # change the df_norms so that it reflects the current year being focussed on 
            df_norms.index = pd.DatetimeIndex([datetime(year,date.month,date.day) for date in df_norms.index])
            
            ind = np.argwhere(df_daily.index.year == year)
            
            # assure the data is complete.
            if np.diff(ind[:,0]).sum() + 1 != len(ind):
                raise ValueError("There is a missing day in year + {0:5d}. This algorithm cannot handle missing days".format(year))
            else:
                b_ind = ind[0]
                e_ind = ind[-1]
                
            first_date = df_daily.index[b_ind]
            end_date = df_daily.index[e_ind]
            
            first_day = first_date.day_of_year[0]-1
            end_day = end_date.day_of_year[0]-1
    
            
            if first_date.is_leap_year and (first_day <= feb_28_day and end_day > feb_28_day):
                # Feb 29 must be added 
                df_list.append(df_norms.iloc[first_day:feb_28_day,:])
                # repeat february 28th as a proxy for February 29th
                df_feb29 = df_norms.iloc[feb_28_day:feb_28_day+1,:]
                df_feb29.index = pd.DatetimeIndex([datetime(year,2,29)])
                df_list.append(df_feb29)
                df_list.append(df_norms.iloc[feb_28_day:end_day+1])
            else:
                # just add the range.
                df_list.append(df_norms.iloc[first_day:end_day+1,:])
    
        df_norms_overlaid = pd.concat(df_list,axis=0)
        df_combined = pd.concat([df_daily,df_norms_overlaid],axis=1)

        return df_combined
    
    def _isolate_waves(self,season,extreme_days):
        
        if self.use_global:
            num_days_for_hw = 2
        else:
            num_days_for_hw = 1 # this used to be 2 but the statistics work out
                            # much better if we include single hot days.
                            # or days with single hot nights.
        waves_by_month = {}
        
        taken = []
        for month in season:
            if month == 12:
                prev_month = 11
                next_month = 1
            elif month == 1:
                prev_month = 12
                next_month = 2
            else:
                prev_month = month - 1
                next_month = month + 1
            
            # reduce heat wave days to only the current and previous months.
            df_wm = extreme_days[(extreme_days.index.month == prev_month) | 
                                    (extreme_days.index.month == month) |
                                    (extreme_days.index.month == next_month)] 
            
            # identify heat wave days and whether they are 2 or more consecutive days
            date_diff = df_wm.index.to_series().diff()
            
            prev_date_1_day_ago = date_diff == np.timedelta64(1,'D')
            
            potential_start_days = np.argwhere((prev_date_1_day_ago == False).values)[:,0]
            wave_consecutive_days = np.argwhere(prev_date_1_day_ago.values)[:,0]
            
            is_start_day = np.concatenate([np.diff(potential_start_days) > num_days_for_hw-1,np.array([True])])
            
            if len(is_start_day) != 1:
                start_days = potential_start_days[is_start_day]
            else:
                start_days = np.array([],dtype=np.int64)
                
            all_wave_days = np.concatenate([wave_consecutive_days,start_days])
            all_wave_days.sort()
            
            
            # Quantify all heat waves in previous, current, and next months as a list
            # this does not capture the last heat wave
            waves = [all_wave_days[np.where(all_wave_days==s1)[0][0]:np.where(all_wave_days==s2)[0][0]] 
                          for s1,s2 in zip(start_days[:-1],start_days[1:])]
            # add the last heat wave.
            if len(start_days) > 1:
                waves.append(np.arange(start_days[-1],wave_consecutive_days[-1]))
            
            # Get rid of any heat waves that are outside or mostly in months. Any heat wave 
            # directly cut in half by between months is assigned to the current month but is added to
            # "taken" so that it will not be added to the next month in the next iteration.
            waves_by_month[month] = (self._delete_waves_fully_outside_this_month(
                waves,df_wm,month,prev_month,next_month,taken),df_wm)

        return waves_by_month
    
    def _determine_norm_param(self,dist_sample):
            mu = dist_sample.mean()
            sig = dist_sample.std()        
            
            
            return {'mu':mu,
                    'sig':sig}
    
    def _plot_fit(self,xdata,ydata,func,pvalues,fit_name,ax=None):
        
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            
        ax.scatter(xdata,ydata,marker="x",color="k")
        
        xline = np.arange(xdata.min(),1.01*xdata.max(),(xdata.max() - xdata.min())/100)
    
        yfunc = func(xline)
            
        ax.plot(xline,yfunc,color="k",linestyle="-")
        
        ax.grid("on")
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_title(fit_name)
    
        
    def _plot_linear_fit(self,xdata,ydata,params,pvalues,fit_name):
        
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        
        ax.scatter(xdata,ydata,marker="x",color="k")
        
        xline = np.arange(xdata.min(),1.5*xdata.max(),(xdata.max() - xdata.min())/2)
        if len(params) == 1:
            yline = params[0] * xline
        else:
            yline = params[1] * xline + params[0]
            
        ax.plot(xline,yline,color="k",linestyle="-")
        
        ax.grid("on")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(fit_name+"Pvalues = " + str(pvalues))
        
    
    def _transform_fit(self,signal):
        # this function maps linearly from -1 to 1
        return 2 *  (signal - signal.min())/(signal.max() - signal.min()) - 1
    
    def _inverse_transform_fit(self,norm_signal, signal_max, signal_min):
        return (norm_signal + 1)*(signal_max - signal_min)/2.0 + signal_min 
        
    
    def _calculate_wave_stats(self,waves,
                              waves_other,
                              frac_tot_days,
                              time_hours,
                              is_hw,
                              hours_with_data,
                              include_plots=True):
        
        # waves_other is for hw if in cs and for cs if in hw.
        if include_plots:
            fig1, ax1 = plt.subplots(4,3,figsize=(5,5))
            fig2, ax2 = plt.subplots(4,3,figsize=(5,5))
            fig3, ax3 = plt.subplots(4,3,figsize=(5,5))
            fig4, ax4 = plt.subplots(4,3,figsize=(5,5))
            fig1.subplots_adjust(wspace=0, hspace=0)
            fig2.subplots_adjust(wspace=0, hspace=0)
            fig3.subplots_adjust(wspace=0, hspace=0)
            fig4.subplots_adjust(wspace=0, hspace=0)
        
        hour_in_day = 24
        stats = {}
        row = 0;col=0
        for month,tup_waves_cur in waves.items():

            if col == 3:
                row += 1
                col = 0
            # duration
            waves_cur = tup_waves_cur[0]
            waves_oth = waves_other[month][0]
            df_wm = tup_waves_cur[1]
            frac = frac_tot_days[month]
            
            # calculate duration stats
            month_duration = np.array([len(arr) for arr in waves_cur])
            month_duration_oth = np.array([len(arr) for arr in waves_oth])
            total_hours_in_extreme = (month_duration.sum() + month_duration_oth.sum())*hour_in_day

            
            if len(month_duration) == 0:
                raise ValueError("No heat waves were identified in {0}".format(month)
                                 +" perhaps there is a unit mismatch in"
                                 +" the daily summaries and climate norms?")
            else:
                max_duration = month_duration.max()

                
            num_day = np.arange(2,max_duration+1)
            duration_history = np.array([(month_duration == x).sum() for x in num_day])
            temp_dict = {}
            
            # calculate log-normal for extreme temperature difference from average conditions and total wave energy
            if is_hw:
                extreme_temp = np.array([(df_wm.iloc[waves_cur[idx],:]["TMAX"]-df_wm.iloc[waves_cur[idx],:]["TAVG_B"]).max() for idx in range(len(waves_cur))])
            else:
                extreme_temp = np.array([(df_wm.iloc[waves_cur[idx],:]["TMIN"]-df_wm.iloc[waves_cur[idx],:]["TAVG_B"]).min() for idx in range(len(waves_cur))])
            
            if np.isnan(extreme_temp.sum()):
                # No Nan allowed during heat waves! throw error, the data needs to be fixed manually!
                isna_idx = [idx for idx, bool_ in enumerate(np.isnan(extreme_temp)) if bool_]
                
                raise ValueError("The daily summaries has a data gap during an extreme temperature event!\n" +
                                 "\n" +
                                 "The following dates are NaN values and must be filled manually for MEWS to work:\n\n"+
                                 str([df_wm.iloc[waves_cur[idx],:].index for idx in isna_idx]))
            
            # This produces negative energy for cold snaps and positive energy for heat waves.
            wave_energy = np.array([((df_wm.iloc[waves_cur[idx],:]["TMAX"] 
                                    + df_wm.iloc[waves_cur[idx],:]["TMIN"])/2 
                                   - df_wm.iloc[waves_cur[idx],:]["TAVG_B"]).sum() 
                                   for idx in range(len(waves_cur))])
            
            # convert from C*day to C*hr and day to hr
            wave_energy_C_hr = wave_energy * hour_in_day
            month_duration_hr = month_duration * hour_in_day
        
            # choose the extremum that is appropriate for the two types of waves (hot/cold)
            if is_hw:
                norm_extreme_temp = extreme_temp.max()
                norm_energy = wave_energy_C_hr.max()
            else:
                norm_extreme_temp = extreme_temp.min()
                norm_energy = wave_energy_C_hr.min()
            
            month_duration_norm = month_duration_hr / month_duration_hr.max()
            wave_energy_norm = wave_energy_C_hr/norm_energy
            extreme_temp_norm = extreme_temp/norm_extreme_temp
            
            # since all hw are exactly one day, the second term reduces to a constant offset term.
            size_tup = (len(month_duration),1)
            energy_regression_vars = np.concatenate(
                [(month_duration_norm).reshape(size_tup)],axis=1)
            max_temp_regression_vars = np.concatenate(
                [np.ones((size_tup)),
                 (month_duration_norm).reshape(size_tup)],axis=1)
            
    
            
            y_regression = [wave_energy_norm,extreme_temp_norm]
            X_regression = [energy_regression_vars,max_temp_regression_vars]
            
            results_shape_coef = []
            par = []
            for y_reg,X_reg in zip(y_regression,X_regression):
            
                model_shape_coef = sm.OLS(y_reg,X_reg)
            
                results_shape_coef.append(model_shape_coef.fit())
                
                par.append(results_shape_coef[-1].params)
            
            # be careful, the function assumes that the first term is the constant term
            # but here it is the slope making reversal of the params fit 
            
    
            E_func = lambda D: (par[0][0] * D)
            T_func = lambda D: (par[1][0] + par[1][1] * D)
    
            if include_plots:
                self._plot_fit(month_duration_norm,wave_energy_norm,
                                E_func, 
                                results_shape_coef[0].pvalues, str(month),ax1[row,col])
                
                self._plot_fit(month_duration_norm,extreme_temp_norm,
                                T_func, 
                                results_shape_coef[1].pvalues, str(month),ax2[row,col])

            # Calculate the linear growth of energy with duration based on the mean
            wave_energy_per_duration = wave_energy_C_hr / (norm_energy * E_func(month_duration_norm))   # deg C * hr / hr
            extreme_temp_per_duration = extreme_temp / (norm_extreme_temp * T_func(month_duration_norm))
            
            # transform to a common interval
            wave_energy_per_duration_norm = self._transform_fit(wave_energy_per_duration)
            extreme_temp_per_duration_norm = self._transform_fit(extreme_temp_per_duration)
            
            
            if include_plots:
                ax3[row,col].hist(wave_energy_per_duration_norm)
                ax3[row,col].set_xticks([])
                ax3[row,col].set_yticks([])
                ax4[row,col].hist(extreme_temp_per_duration_norm)
                ax4[row,col].set_xticks([])
                ax4[row,col].set_yticks([])

            temp_dict['help'] = ("These statistics are already mapped"+
                                " from -1 ... 1 and _inverse_transform_fit is"+
                                " needed to return to actual degC and"+
                                " degC*hr values. If input of actual values is desired use transform_fit(X,max,min)")
            temp_dict['energy_normal_param'] = self._determine_norm_param(wave_energy_per_duration_norm)
            temp_dict['extreme_temp_normal_param'] = self._determine_norm_param(extreme_temp_per_duration_norm)
            temp_dict['max extreme temp per duration'] = extreme_temp_per_duration.max()
            temp_dict['min extreme temp per duration'] = extreme_temp_per_duration.min()
            temp_dict['hist max extreme temp per duration'] = extreme_temp_per_duration.max()
            temp_dict['hist min extreme temp per duration'] = extreme_temp_per_duration.min()
            temp_dict['max energy per duration'] = wave_energy_per_duration.max()
            temp_dict['min energy per duration'] = wave_energy_per_duration.min()
            temp_dict['hist max energy per duration'] = wave_energy_per_duration.max()
            temp_dict['hist min energy per duration'] = wave_energy_per_duration.min()
            temp_dict['energy linear slope'] = par[0][0]
            temp_dict['normalized extreme temp duration fit slope'] = par[1][1]
            temp_dict['normalized extreme temp duration fit intercept'] = par[1][0]
            temp_dict['normalizing energy'] = norm_energy
            temp_dict['normalizing extreme temp'] = norm_extreme_temp
            temp_dict['normalizing duration'] = month_duration_hr.max()
            # historic time interval
            temp_dict['historic time interval'] =  hours_with_data
            #
            # Markov chain model parameter estimation
            #
            hour_in_cur_month = time_hours * frac
            # for Markov chain model of heat wave initiation. 
            prob_of_wave_in_any_hour = len(month_duration)/(hour_in_cur_month-total_hours_in_extreme)
            
            # for Markov chain model of probability a heat wave will continue into 
            # the next hour a linear regression is needed here
            
            num_hour_per_event_duration = (duration_history * num_day)*hour_in_day
            
            #if self.use_global:
            # We are fitting P(n) = P0^n, where P0 is the variable desired to determine and is the
            # Markov probability of transition out of the heat wave state based on the duration data.
            # this is a linear fit if we take the logarithm.
            prob_per_event = num_hour_per_event_duration/num_hour_per_event_duration.sum()
            
            included = prob_per_event != 0.0 # drop any heat wave durations that have no data!
            
            log_prob_duration = np.log(prob_per_event[included])
            
            num_hour_passed = num_day[included] * hour_in_day
            
            num_hour_passed = np.insert(num_hour_passed,0,0)
            log_prob_duration = np.insert(log_prob_duration,0,0)

            model = sm.OLS(log_prob_duration,num_hour_passed)
            
            results = model.fit()

            P0 = np.exp(results.params[0])
            
            # verify that the result is significant by p-value < 0.05 i.e. 
            # the probability that the null hypothesis is true (i.e. your results 
            # is a random chance occurance is <5%)
            pvalue = results.pvalues[0]
                    
            #else:
                # 10/4/2022 - we need the discreet geometric distribution since we
                # are working with discreet Markov processes
                # this always comes out to a positive integer under the constraints present 
                #P0,lamb,lcov,pvalue = fit_exponential_distribution(month_duration_hr,include_plots)
                
            if pvalue > 0.05:
                if self.use_global and self.include_plots:
                    self._plot_linear_fit(log_prob_duration,num_hour_passed,results.params,results.pvalues,
                                    "Month=" + str(month)+" Markov probability")
                warn(results.summary())
                warn("The weather data has produced a low p-value fit"+
                                 " for Markov process fits! More data is needed "+
                                 "to produce a statistically significant result!")
                
                    
            # normalize and assume that the probability of going to a cold snap 
            # directly from a heat wave is zero.
            temp_dict['hourly prob stay in heat wave'] = P0 
            temp_dict['hourly prob of heat wave'] = prob_of_wave_in_any_hour
            
            # adding duration and temperature histograms
            hrs_range = np.arange(24,month_duration_hr.max()+24,24)
            temp_dict['historical durations (durations0)'] = np.histogram(month_duration_hr,
                                                             range=(12,month_duration_hr.max()+12),
                                                             bins=len(hrs_range))
            nbins = int(np.ceil(len(extreme_temp)/10))
            bin_delT = (extreme_temp.max() - extreme_temp.min())/nbins
            
            temp_dict['historical temperatures (hist0)'] = np.histogram(extreme_temp,
                                                             range=(extreme_temp.min()-0.5*bin_delT,extreme_temp.max()+0.5*bin_delT),
                                                             bins=nbins)
            temp_dict['historical delTmax'] = extreme_temp
            temp_dict['historical durations'] = month_duration_hr
            
            stats[month] = temp_dict
            
            col+=1
            
        if include_plots:
            if is_hw:
                fig1.savefig("historic_heat_wave_energy.png",dpi=300)
                fig2.savefig("historic_heat_wave_temperature.png",dpi=300)
                fig3.savefig("energy_per_duration_heat_wave_histogram",dpi=300)
                fig4.savefig("maxdelT_per_duration_heat_wave_histogram",dpi=300)
            else:
                fig1.savefig("historic_cold_snap_energy.png",dpi=300)
                fig2.savefig("historic_cold_snap_temperature.png",dpi=300)
                fig3.savefig("energy_per_duration_cold_snap_histogram",dpi=300)
                fig4.savefig("maxdelT_per_duration_cold_snap_histogram",dpi=300)                
                
            
        return stats
    
    
    def _wave_stats(self,df_combined,include_plots=False):
        
        """
        wave_stats(df_combined,is_heat)
        
        Calculates the algebraically estimated statistical parameter per month for heat waves or cold snaps
        
        Parameters
        ----------    
        df_combined : pd.DataFrame
            A combined data set of NOAA historical
            daily data and NOAA climate 10%, 50%, and 90% data
            Violation of the 90% (for heat waves) or 10% data
            is how heat wave days are identified.
            
        Returns
        -------
        mews_stats : dict 
            A dictionary of mews statstics.
        """
        
        # total time covered by the dataset
        seconds_in_hour = 3600.0
        months_per_year = 12
        hours_per_day = 24.0
        time_hours = (df_combined.index[-1] - df_combined.index[0]).total_seconds()/seconds_in_hour + hours_per_day
        hours_with_data = hours_per_day * len(df_combined.index)
        frac_hours_in_month = hours_per_day * np.array([len(df_combined[df_combined.index.month == month]) for month in range(1,13)])/hours_with_data
        
        
        
        stats = {}
        
        is_heat_wave = [True,False]
        extreme_days = {}
        waves_all_year = {}
        
        
        # above 90% criterion for TMAX_B or above the 90% criterion for the hourly maximum minimum temperature
        extreme_days['hw'] = df_combined[(df_combined["TMAX"] > df_combined["TMAX_B"])|
                                     (df_combined["TMIN"] > df_combined["TMINMAX_B"])]

        # below 10% criterion TMIN or below 10% criterion for the minimum hourly maximum temperature
        extreme_days['cs'] = df_combined[(df_combined["TMIN"] < df_combined["TMIN_B"])|
                                     (df_combined["TMAX"] < df_combined["TMAXMIN_B"])]

        # do a different assessment for each month in the heat wave season because 
        # the statistics show a significant peak at the end of summer and we do not
        # want the probability to be smeared out as a result.
        
        # we need to determine if the month starts or ends with heat waves. If it does,
        # the month with more days gets the heat wave. If its a tie, then the current month
        # gets the heat wave and the heat wave is marked as taken so that it is 
        # not double counted
            
        # now the last false before a true is the start of each heat wave 
        # then true until the next false is the duration of the heat wave
        
        months = np.arange(1,months_per_year+1)
        waves_all_year['hw'] = self._isolate_waves(months,extreme_days['hw']) 
        waves_all_year['cs'] = self._isolate_waves(months,extreme_days['cs']) 
        
        # this is the total days in each month. Nothing to do with waves.
        num_total_days = df_combined.groupby(df_combined.index.month).count()["TMIN"]
        frac_tot_days = num_total_days/num_total_days.sum()
            
        for is_hw in is_heat_wave:
            if is_hw:
                waves_all_year_cur = waves_all_year['hw']
                waves_all_year_other = waves_all_year['cs']
            else:
                waves_all_year_cur = waves_all_year['cs']
                waves_all_year_other = waves_all_year['hw']

            if is_hw:
                description = "heat wave"
            else:
                description = "cold snap"
            
            stats[description] = self._calculate_wave_stats(waves_all_year_cur,
                                                            waves_all_year_other,
                                                            frac_tot_days,
                                                            time_hours,
                                                            is_hw,
                                                            hours_with_data,
                                                            include_plots)
        return stats, frac_hours_in_month
    
    def _plot_stats_by_month(self,stats,title_string):
        
        mu = []
        sig = []
        muT = []
        sigT = []
        p_hw = []
        ps_hw = []
        months = []
        
        for month,modat in stats.items():
            logdat = modat['energy_normal_param']
            logdelT = modat['extreme_temp_normal_param']
            mu.append(logdat['mu'])
            sig.append(logdat['sig'])
            muT.append(logdelT['mu'])
            sigT.append(logdelT['sig'])
            months.append(month)
            
            p_hw.append(modat['hourly prob of heat wave'])
            ps_hw.append(modat['hourly prob stay in heat wave'])
        
        fontsize={'font.size':16}
        
        plt.rcParams.update(fontsize)    
           
        fig,axl = plt.subplots(6,1,figsize=(5,8))
        
        dat = [mu,sig,muT,sigT,p_hw,ps_hw]
        name = ["$\mu_{\Delta E}$","$\sigma_{\Delta E}$","$\mu_{\Delta T}$","$\sigma_{\Delta T}$","$P_{w}$","$P_{sw}$"]
        
        idx = 0
        for ax,da,na in zip(axl,dat,name):
            idx+=1
            ax.plot(months,da)
            ax.set_ylabel(na)
            ax.grid("on")
            ax.set_xticks(range(1,13))
            if idx < 5:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
        axl[0].set_title(title_string)
        axl[-1].set_xlabel("Month")
        
        try:
            #This can cause a latex error in certain cases if latex cannot install
            # new packages
            plt.tight_layout()
        except:
            pass
        plt.savefig(title_string+"_monthly_MEWS_parameter_results.png",dpi=1000)
        
    
    def read_solution(self,solution_file):
        
        file_exists = os.path.exists(solution_file)
        if file_exists:
            new_stats = read_readable_python_dict(solution_file)   
        else:
            raise FileNotFoundError("The solution file '{0}' does not exist. An existing file must be input!".format(solution_file))
        
        return new_stats
    
    def write_solution(self,solution_file,is_historic=True,overwrite=True):

        file_exists = os.path.exists(solution_file)
        if file_exists and not overwrite:
            raise FileExistsError("The solution file '{0}' exists and overwrite has been set = False. Set overwrite to True if you want to overwrite!".format(solution_file))
        else:
            if file_exists:
                os.remove(solution_file)

            write_readable_python_dict(solution_file, self.stats)
    
    
    
    def _delete_waves_fully_outside_this_month(self,heat_waves,df_wm,month,prev_month,next_month,hw_taken):
        
        keep_ind = []
        
        for idx,hw in enumerate(heat_waves):
    
            df_wave_ind = df_wm.iloc[hw,0].index
            # on boundary
            numday_in_month = (df_wave_ind.month == month).sum()
            if numday_in_month == 0:
                pass # do nothing, the entire heat wave is in a previous
                     # or posthumous month. this heat wave will not be included in keep_ind
            else:        
                if len(df_wave_ind) != numday_in_month:
                    # we have a boundary violation and we need to deal with it
                    num_prev = (df_wave_ind.month == prev_month).sum()
                    num_next = (df_wave_ind.month == next_month).sum()    
                    
                    if num_prev > numday_in_month and num_prev > num_next:
                        # the heat wave belongs in the previous month
                        pass
                    elif num_next > numday_in_month and num_next > num_prev:
                        # the heat wave belongs to the next month
                        pass
                    elif ((num_next == numday_in_month and num_next > num_prev) or
                         (num_prev == numday_in_month and num_prev > num_next) or
                         (num_next == num_prev)):
                        # this heat wave is given to the current month but must also be marked as taken since
                        # it will also qualify to be taken by the next month.
                        wave_is_taken = False
                        for dfhw in hw_taken:
                            if df_wave_ind.equals(dfhw):
                                wave_is_taken = True
                        
                        if not wave_is_taken:
                            hw_taken.append(df_wave_ind)
                            keep_ind.append(idx)
                            
                    else: # the majority of days are in the current month and the heat wave belongs 
                          # to the current month.
                        keep_ind.append(idx)
    
                else:
                    # no changes needed, the heat wave is fully in the current month.
                    keep_ind.append(idx)
        
        cur_month_heat_waves = [heat_waves[idx] for idx in keep_ind]
    
        return cur_month_heat_waves
            
class _DeltaTransition_IPCC_FigureSPM6():
    """
    >>> obj = _DeltaTransition_IPCC_FigureSPM6(hot_param,
                                               cold_param,
                                               climate_temp_func,
                                               year,
                                               month_hours_per_year,
                                               use_global=False,
                                               climate_baseyear=None,
                                               scenario=None,
                                               ext_temp_waves_obj=None,
                                               increase_factor_ci="50%",
                                               prob_hw=None,
                                               delT_ipcc_min_frac=1.0,
                                               month=None,
                                               random_seed=None,
                                               solve_options=None,
                                               cold_snap_shift=None,
                                               write_csv=False,
                                               extra_columns=None,
                                               solve_obj=None)
    
    This function has two use cases depending on whether MEWS is being used
    with the old global CMIP6 data or for actual CMIP6 lat/lon projections.
    
    if use_global = True:
        This class assumes that we can divide by the 1.0 C multipliers for the present
        day and then multiply. We interpolate linearly or extrapolate linearly from
        the sparse data available.
        
        This is called within the context of a specific month.
    else:
        The begin and end years of the heat wave data are used to find the
        middle time of the NOAA daily summaries. This date is then interpolated 
        between the 1850-1900 (i.e. 1875). If we have 1930-1980 the middle date 
        is 1955. We use the CMIP historical polynomial fit to determine the baseline
        temperature change from 1850-1900 for the actual data. We then interpolate
        a baseline factor instead of simply using the 1.0C factor.
    
    """
    
    _valid_increase_factor_tracks = {"5%":['5% CI Increase in Intensity','5% CI Increase in Frequency'],
                                     "50%":['50% CI Increase in Intensity','50% CI Increase in Frequency'],
                                     "95%":['95% CI Increase in Intensity','95% CI Increase in Frequency']}
    _valid_scenario_names = ClimateScenario._valid_scenario_names
    
    def __init__(self,hot_param,cold_param,climate_temp_func,year,month_hours_per_year,
                 use_global=False,
                 climate_baseyear=None,
                 scenario=None,
                 ext_temp_waves_obj=None,
                 increase_factor_ci="50%",
                 prob_hw=None,
                 delT_ipcc_min_frac=1.0,
                 month=None,
                 random_seed=None,
                 solve_options=None,
                 cold_snap_shift=None,
                 write_csv=False,
                 extra_columns=None,
                 solve_obj=None):
        if not scenario is None:
            identifier = scenario
            identifier = identifier + "_" + str(year)
        if not month is None:
            identifier = identifier + "_" + str(month) + increase_factor_ci
        
        if solve_obj is None:
            self.obj_solve = None
        else:
            self.obj_solve = solve_obj
        
        #input validation
        if use_global==False:
            if climate_baseyear is None:
                raise ValueError("'climate_baseyear' input must not be None if 'use_global'=False")
            if scenario is None:
                raise ValueError("'scenario' input must not be None if 'use_global'=False")
            if ext_temp_waves_obj is None:
                raise ValueError("'ext_temp_waves_obj' input must not be None if 'use_global'=False")
            if prob_hw is None:
                raise ValueError("'prob_hw' input must not be None if 'use_global'=False")
            if month is None:
                raise ValueError("'month' input must not be None if 'use_global'=False")
        
        # increase_factor_ci_val is checked when coming into Extreme
        if use_global == False:
            # unpack.
            hist_str = self._valid_scenario_names[0]
            baseline_year = climate_baseyear

            # CMIP data goes back to 1850.
            historic_temp_func = climate_temp_func['historical']
            # average delT during IPCC baseline of 1850-1900
            avg_delT_1850_1900 = np.array([historic_temp_func(yr-baseline_year) for yr in np.arange(1850,1900.01,0.01)]).mean()
            # average delT during heat wave data used
            avg_delT_hw_period = np.array([historic_temp_func(yr-baseline_year)
                                           for yr in np.arange(ext_temp_waves_obj.hw_beg_date,
                                                               ext_temp_waves_obj.hw_end_date+.01,0.01)]).mean()
            # remember these numbers are negative
            hw_delT = avg_delT_hw_period-avg_delT_1850_1900
            baseline_delT = -avg_delT_1850_1900
            
            #calculate heat wave probability shape function used to decrease
            #ipcc increases for months with lower probability of heat waves.
            prob_hw_arr = np.array([val for val in prob_hw.values()])
            self._hw_maxprob = prob_hw_arr.max()
            self._hw_minprob = prob_hw_arr.min()
            self._delT_ipcc_min_frac = delT_ipcc_min_frac
            self.delT_ipcc_frac = self._hw_probability_shape_func(prob_hw_arr)
            delT_ipcc_frac_month = self.delT_ipcc_frac[int(month-1)]
            
        else:
            hw_delT = None 
            baseline_delT = None
            delT_ipcc_frac_month = None
            
        self.use_global = use_global
        # bring in the ipcc data and process it.
        ipcc_data =  pd.read_csv(os.path.join(os.path.dirname(__file__),"data","IPCC_FigureSPM_6.csv"))
        

        try:
            if use_global:
                # This is change in global temperature from 2020!
                delta_TG = climate_temp_func[scenario](year)
                
                # now interpolate from the IPCC tables 
                
                (ipcc_val_10, ipcc_val_50) = self._interpolate_ipcc_data(ipcc_data, delta_TG, hw_delT, baseline_delT)
                
            else:
                delta_TG = climate_temp_func[scenario](year-baseline_year)
                (ipcc_val_10, ipcc_val_50) = self._interpolate_ipcc_data(ipcc_data, delta_TG, hw_delT, baseline_delT)
            cancel_run = False
        except ValueError as excep:
            cancel_run = True
            warn("A value error occured when trying to interpolate the ipcc table.!\n\n Traceback:\n\n" + 
                 str(traceback.print_stack()) + "\n\n" + str(excep))
        except Exception as excep:
            raise excep
        
        self.cancel_run = cancel_run
        
        if not cancel_run:
            
            f_ipcc_ci_10 = ipcc_val_10[self._valid_increase_factor_tracks[increase_factor_ci][1]] 
            f_ipcc_ci_50 = ipcc_val_50[self._valid_increase_factor_tracks[increase_factor_ci][1]]
    
            #bring the multiplication factors to the surface.
            dfraw = pd.concat([ipcc_val_10,ipcc_val_50],axis=1)
            dfraw.columns = ["10 year event","50 year event"]
            self.ipcc_fact = dfraw
    
           
            if use_global:
                tup = self._old_analysis(use_global,f_ipcc_ci_50,f_ipcc_ci_10,
                                   hot_param,cold_param,delT_ipcc_frac_month,
                                   ipcc_val_10,ipcc_val_50,increase_factor_ci,
                                   delta_TG)
            
            else:
                tup = self._new_analysis(hot_param,cold_param,delT_ipcc_frac_month,
                                   ipcc_val_10,ipcc_val_50,increase_factor_ci,
                                   delta_TG, solve_options,random_seed, 
                                   month_hours_per_year,cold_snap_shift,write_csv,
                                   extra_columns, identifier)
                
            
            (Phwm, Pcsm, P_prime_hwm, P_prime_csm, Pcssm, P_prime_cssm, Phwsm, 
               P_prime_hwsm, del_mu_E_hw_m, del_sig_E_hw_m, del_a_E_hw_m, 
               del_b_E_hw_m, del_mu_delT_max_hwm, del_sig_delT_max_hwm,
               del_a_delT_max_hwm, del_b_delT_max_hwm, delT_abs_max 
             ) = tup
            
            self.transition_matrix_delta = np.array(
                [[Phwm + Pcsm - P_prime_hwm - P_prime_csm, P_prime_csm - Pcsm, P_prime_hwm - Phwm],
                 [Pcssm - P_prime_cssm, P_prime_cssm - Pcssm, 0.0],
                 [Phwsm - P_prime_hwsm, 0.0, P_prime_hwsm - Phwsm]])
            self.del_E_dist = {'del_mu':del_mu_E_hw_m,
                     'del_sig':del_sig_E_hw_m,
                     'del_a':del_a_E_hw_m,
                     'del_b':del_b_E_hw_m}
            self.del_delTmax_dist = {'del_mu':del_mu_delT_max_hwm,
                     'del_sig':del_sig_delT_max_hwm,
                     'del_a':del_a_delT_max_hwm,
                     'del_b':del_b_delT_max_hwm,
                     'delT_increase_abs_max':delT_abs_max}
            


    def _interpolate_ipcc_data(self,ipcc_data,delta_TG,hw_delT,baseline_delT=None):
        
        # this function is dependent on the format of the table in IPCC_FigureSPM_6.csv
        if self.use_global:
            present_tempanomal = ipcc_data['Global Warming Levels (⁰C)'].values[0]
        else:
            present_tempanomal = baseline_delT

        future_tempanomal = present_tempanomal + delta_TG
        
        if self.use_global:
            if future_tempanomal > 4.0:
                raise ValueError("The current IPCC data only includes changes in temperature of 4C for global warming!")
        else:
            if future_tempanomal > 4.0:
                warn("The mews analysis allows extrapolation beyond the IPCC info that gives factors to 4.0C. The growth in the "+
                     "IPCC data is nearly linear. Your temperature anomaly is at {0:5.2f}C".format(future_tempanomal))
            if future_tempanomal > 6.05:

                raise ValueError("The current MEWS analysis only allows extrapolation to 6.05C and the IPCC info only goes to 4C")
        
            
        ipcc_num = ipcc_data.drop(["Event","Units"],axis=1) 
        
        def interp_func(ipcc_num,use_global,tempanomal):
            
            #global warming delta T
            gwDT = ipcc_num['Global Warming Levels (⁰C)'].values
            
            ind = 0
            
            for ind in range(4):
                if tempanomal <= ipcc_data['Global Warming Levels (⁰C)'].values[ind]:
                    break
            
            if tempanomal > 4.0:
                # must extrapolate
                ipcc_val_10_u = (ipcc_num.loc[3,:]-ipcc_num.iloc[2,:])/(gwDT[3] - gwDT[2])*(tempanomal-gwDT[3])+ipcc_num.loc[3,:]
                ipcc_val_50_u = (ipcc_num.iloc[-1,:]-ipcc_num.iloc[-2,:])/(gwDT[-1] - gwDT[-2])*(tempanomal-gwDT[-1])+ipcc_num.iloc[-1,:]
            
            elif ind > 0:
                interp_fact = (tempanomal - 
                   ipcc_data['Global Warming Levels (⁰C)'].values[ind-1])/(
                   ipcc_data['Global Warming Levels (⁰C)'].values[ind] - 
                   ipcc_data['Global Warming Levels (⁰C)'].values[ind-1])
                 
                   
                     
                ipcc_val_10_u = (ipcc_num.loc[ind,:] - ipcc_num.loc[ind-1,:]) * interp_fact + ipcc_num.loc[ind-1,:]
                ipcc_val_50_u = (ipcc_num.loc[ind+4,:] - ipcc_num.loc[ind+3,:]) * interp_fact + ipcc_num.loc[ind+3,:]
            else:
                if use_global:
                    ipcc_val_10_u = ipcc_num.loc[0,:]
                    ipcc_val_50_u = ipcc_num.loc[4,:]
                else:
                    def inte_freq_extrap_zero(tempanomal,gwDT,ipcc_num,zero_freq,zero_inte,ind):
                        intensity = (tempanomal / gwDT)*(ipcc_num.iloc[ind,1:4]-zero_inte) + zero_inte
                        frequency = (tempanomal / gwDT)*(ipcc_num.iloc[ind,4:]-zero_freq) + zero_freq
                        return pd.concat([ipcc_num.iloc[ind,0:1],intensity,frequency])
                    
                    freq_zero = 1.0
                    inte_zero = 0.0
                    
                    ipcc_val_10_u = inte_freq_extrap_zero(tempanomal,gwDT[0],ipcc_num,freq_zero,inte_zero,0)
                    ipcc_val_50_u = inte_freq_extrap_zero(tempanomal,gwDT[4],ipcc_num,freq_zero,inte_zero,4)
            
            return ipcc_val_10_u, ipcc_val_50_u
        
        ipcc_val_10_u, ipcc_val_50_u = interp_func(ipcc_num,self.use_global, future_tempanomal)
        
        if self.use_global == False:
            # hwd = heat wave data
            ipcc_val_10_hwd, ipcc_val_50_hwd = interp_func(ipcc_num,self.use_global, hw_delT)
            
            
        
        # TODO - if less recent data is available, this (Below) assumption
        # is non-conversative and will underestimate shifts in climate.
        
        # 9/9/2022 - with use_global=False, this assumption below no longer applies. The heat wave
        # data time interval is now considered so that the factor taken away is
        # an exact interpolation between 0 and the first value in the table.
        
        # XXXXXNO LONGER APPLIES EXCEPT FOR use_global=True
        # Assumption: Because, these values are being based off of data that is more recent,
        # XXXXXthe amplification/offset has to be based on current 1.0C warming levels.
        num_std_in_95ci = 1.96
        def divide_CI(val,val_hw):
            # divide one confidence interval by another assuming 
            # two independent random variables.
            
            num = np.random.normal(val["50% CI Increase in Frequency"],
                                   ((val['95% CI Increase in Frequency']-val["5% CI Increase in Frequency"])/2)/num_std_in_95ci,
                                   100000)
            num_hw = np.random.normal(val_hw["50% CI Increase in Frequency"],
                                               ((val_hw['95% CI Increase in Frequency']-val_hw["5% CI Increase in Frequency"])/2)/num_std_in_95ci,
                                               100000)
            num_new = num/num_hw
            
            mean = num_new.mean()
            std = num_new.std()
            return mean,std
        
        
        ipcc_val_10 = deepcopy(ipcc_val_10_u)
        ipcc_val_50 = deepcopy(ipcc_val_50_u)
        
        for lab,val in ipcc_val_10_u.items():
            if "Intensity" in lab:
                if self.use_global:
                    ipcc_val_10[lab] = ipcc_val_10_u[lab] - ipcc_num.loc[0,lab]
                    ipcc_val_50[lab] = ipcc_val_50_u[lab] - ipcc_num.loc[4,lab]
                else:
                    ipcc_val_10[lab] = ipcc_val_10_u[lab] - ipcc_val_10_hwd[lab]
                    ipcc_val_50[lab] = ipcc_val_50_u[lab] - ipcc_val_50_hwd[lab]
            #
            #  DIVIDSION OF CONFIDENCE INTERVALS IS MORE COMPLEX!
            elif "Frequency" in lab:
                if self.use_global:
                    ipcc_val_10[lab] = ipcc_val_10_u[lab] / ipcc_num.loc[0,lab]
                    ipcc_val_50[lab] = ipcc_val_50_u[lab] / ipcc_num.loc[4,lab]
            #     else:
            #         ipcc_val_10[lab] = ipcc_val_10_u[lab] / ipcc_val_10_hwd[lab]
            #         ipcc_val_50[lab] = ipcc_val_50_u[lab] / ipcc_val_50_hwd[lab]                
            
            # We assume independent random variables!
        if self.use_global == False:
            mean_ci_10,std_ci_10 = divide_CI(ipcc_val_10,ipcc_val_10_hwd)
            mean_ci_50,std_ci_50 = divide_CI(ipcc_val_50,ipcc_val_50_hwd)
        
            ipcc_val_10["50% CI Increase in Frequency"] = mean_ci_10
            ipcc_val_50["50% CI Increase in Frequency"] = mean_ci_50
            
            ipcc_val_10["5% CI Increase in Frequency"] = mean_ci_10 - std_ci_10 * num_std_in_95ci
            ipcc_val_50["5% CI Increase in Frequency"] = mean_ci_50 - std_ci_50 * num_std_in_95ci
            
            ipcc_val_10["95% CI Increase in Frequency"] = mean_ci_10 + std_ci_10 * num_std_in_95ci
            ipcc_val_50["95% CI Increase in Frequency"] = mean_ci_50 + std_ci_50 * num_std_in_95ci
            
        return ipcc_val_10, ipcc_val_50    
    
    
    def _hw_probability_shape_func(self,hw_prob):
        #Returns a mapping from probability to fraction of delT ipcc added 
        # to a given probability hw_maxprob maps to 1.0 and hw_minprob maps to
        # delT_ipcc_min_frac which is a user input between 0 and 1
        return (1.0-self._delT_ipcc_min_frac)/(self._hw_maxprob - self._hw_minprob) * (hw_prob-self._hw_minprob) + self._delT_ipcc_min_frac
    
    def _new_analysis(self, hot_param, cold_param, delT_ipcc_frac_month, 
                      ipcc_val_10, ipcc_val_50,
                      increase_factor_ci, delta_TG, solve_options, random_seed,
                      frac_month_hours_per_year,cold_snap_shift,write_csv,
                      extra_columns,identifier):

        # 1. arrange inputs
        if random_seed is None:
            random_seed = DEFAULT_RANDOM_SEED
        
        ipcc_shift = {}
        ipcc_shift['hw'] = {}
        ipcc_shift['hw']['temperature'] = {'10 year':ipcc_val_10[self._valid_increase_factor_tracks[increase_factor_ci][0]],
                                     '50 year':ipcc_val_50[self._valid_increase_factor_tracks[increase_factor_ci][0]]
                                     }
        ipcc_shift['hw']['frequency']   = {'10 year':ipcc_val_10[self._valid_increase_factor_tracks[increase_factor_ci][1]],
                                     '50 year':ipcc_val_50[self._valid_increase_factor_tracks[increase_factor_ci][1]]
                                     }
        # allow manually input cold snap shift.
        ipcc_shift['cs'] = cold_snap_shift
        
        
        historic_time_interval = int(hot_param['historic time interval'])
        
        # These defaults do not match all of the defaults in solve.py
        # there are specific differences that are intentional and they 
        # should be kept separate.
        default_vals = DEFAULT_SOLVE_OPTIONS['future']
        
        opt_val = _mix_user_and_default(default_vals,'future',solve_options)
        
        # already at the monthly level
        mstats = {'heat wave': hot_param, 'cold snap': cold_param}
        hist0 = {}
        durations0 = {}
        param0 = {}
        for wn1,wn2 in WAVE_MAP.items():
            hist0[wn2] = mstats[wn1]['historical temperatures (hist0)'] 
            durations0[wn2] = mstats[wn1]['historical durations (durations0)']
            param0[wn2] = mstats[wn1]
        
        # END OF ARRANGING INPUTS
        if self.obj_solve is None:
            obj_solve = SolveDistributionShift(opt_val['num_step'], 
                                   param0, 
                                   random_seed, 
                                   hist0, 
                                   durations0, 
                                   opt_val['delT_above_shifted_extreme'], 
                                   historic_time_interval, 
                                   int(frac_month_hours_per_year*HOURS_IN_YEAR),
                                   problem_bounds=opt_val['problem_bounds'],
                                   ipcc_shift=ipcc_shift,
                                   decay_func_type=opt_val['decay_func_type'],
                                   use_cython=opt_val['use_cython'],
                                   num_cpu=opt_val['num_cpu'],
                                   plot_results=opt_val['plot_results'],
                                   max_iter=opt_val['max_iter'],
                                   plot_title=opt_val['plot_title'],
                                   out_path=opt_val['out_path'],
                                   weights=opt_val['weights'],
                                   limit_temperatures=opt_val['limit_temperatures'],
                                   min_num_waves=opt_val['min_num_waves'],
                                   x_solution=opt_val['x_solution'],
                                   test_mode=opt_val['test_mode'],
                                   num_postprocess=opt_val["num_postprocess"],
                                   extra_output_columns=extra_columns,
                                   identifier=identifier)
            self.obj_solve = obj_solve
        else:
            inputs = {"param0":param0,
                      "hist0":hist0,
                      "durations0":durations0,
                      "hours_per_year":int(frac_month_hours_per_year * HOURS_IN_YEAR),
                      "x_solution":opt_val['x_solution'],
                      "out_path":opt_val['out_path'],
                      "extra_output_columns":extra_columns,
                      "ipcc_shift":ipcc_shift}
            self.obj_solve.reanalyze(inputs,write_csv)

        
        
        par = self.obj_solve.param
        des = self.obj_solve.del_shifts
        
        Phwm = hot_param["hourly prob of heat wave"]
        # probability a heat wave is sustainted
        Phwsm = hot_param["hourly prob stay in heat wave"]
        # probability of a cold snap
        Pcsm = cold_param["hourly prob of heat wave"]
        # probability of sustaining a cold snap
        Pcssm = cold_param["hourly prob stay in heat wave"]
        
        P_prime_hwm = par['hw']['hourly prob of heat wave']
        P_prime_csm = par['cs']['hourly prob of heat wave']
        P_prime_hwsm = par['hw']['hourly prob stay in heat wave']
        P_prime_cssm = par['cs']['hourly prob stay in heat wave']
        
        # no change to the duration normalized energy distributions.
        (del_mu_E_hw_m, del_sig_E_hw_m, del_a_E_hw_m, 
        del_b_E_hw_m) = (0,0,0,0)

        del_mu_delT_max_hwm = des['hw']["del_mu_T"] 
        del_sig_delT_max_hwm = des['hw']["del_sig_T"]
        del_a_delT_max_hwm = des['hw']['del_a']
        del_b_delT_max_hwm = des['hw']['del_b']
        
        delT_abs_max = self.obj_solve.abs_max_temp  # this is a dictionary which is new

        return (Phwm, Pcsm, P_prime_hwm, P_prime_csm, Pcssm, P_prime_cssm, Phwsm, 
           P_prime_hwsm, del_mu_E_hw_m, del_sig_E_hw_m, del_a_E_hw_m, 
           del_b_E_hw_m, del_mu_delT_max_hwm, del_sig_delT_max_hwm,
           del_a_delT_max_hwm, del_b_delT_max_hwm, delT_abs_max
         ) 

    def _old_analysis(self, use_global, f_ipcc_ci_50, f_ipcc_ci_10, hot_param,
                      cold_param, delT_ipcc_frac_month, ipcc_val_10, ipcc_val_50,
                      increase_factor_ci, delta_TG):
               
        # neglect leap years
        hours_in_10_years = 10 * DAYS_IN_YEAR * HOURS_IN_DAY  # hours in 10 years
        hours_in_50_years = 5 * hours_in_10_years
        
        # switch to the paper notation
        N10 = hours_in_10_years
        N50 = hours_in_50_years
        # probability a heat wave begins
        Phwm = hot_param["hourly prob of heat wave"]
        # probability a heat wave is sustainted
        Phwsm = hot_param["hourly prob stay in heat wave"]
        # probability of a cold snap
        Pcsm = cold_param["hourly prob of heat wave"]
        # probability of sustaining a cold snap
        Pcssm = cold_param["hourly prob stay in heat wave"]
        # unpack gaussian distribution parameters
        normalized_ext_temp = hot_param['extreme_temp_normal_param']
        normalized_energy = hot_param['energy_normal_param']
        maxtemp = hot_param['max extreme temp per duration']
        mintemp = hot_param['min extreme temp per duration']
        
        #norm_energy = hot_param['normalizing energy']
        norm_temp = hot_param['normalizing extreme temp']
        norm_duration = hot_param['normalizing duration']
        alphaT = hot_param['normalized extreme temp duration fit slope']
        betaT = hot_param['normalized extreme temp duration fit intercept']
        
        
        if use_global:
            P_prime_hwm = Phwm * (f_ipcc_ci_50 * N10 + f_ipcc_ci_10 * N50)/(N10 + N50)
            
            # Estimate the interval positions of the 10 year and 50 year changes
            # in temperature.
            # equation 23 - all statistics have been translated to -1, 1, -1 is the minimum 
            #               extreme temperature and 1 is the maximum extreme temperature
            
            mu_norm = normalized_ext_temp['mu']
            sig_norm = normalized_ext_temp['sig']
            
            # solve for the 10 year and 50 year expected peak temperature per duration
            F0 = lambda x: cdf_truncnorm(transform_fit(x,mintemp,maxtemp),
                                         mu_norm,
                                         sig_norm,
                                         -1,
                                         1)
            
            S10 = - 1 + 1/(N10 * Phwm)
            S50 = - 1 + 1/(N50 * Phwm)
            
            F10 = lambda x:F0(x) + S10
            F50 = lambda x:F0(x) + S50
            delTmax10_hwm, r10 = bisect(F10,mintemp,maxtemp,full_output=True)
            delTmax50_hwm, r50 = bisect(F50,mintemp,maxtemp,full_output=True)
    
            if not r10.converged:
                raise ValueError("Bisection method failed to find 10 year expected value for heat wave temperature")
            elif not r50.converged:
                raise ValueError("Bisection method failed to find 50 year expected value for heat wave temperature")
                
            # Find the -1..1 interval shift parameters that reflect the IPCC shift amounts
            # in temperature for 10 and 50 year events. This can shift and stretch the distribution.
            # solve for the shift in mean and standard deviation (2 equations two unknowns)
            # equation 24 in the writeup
            
            # sig_s and mu_s are the independent shift and stretch variables that 
            # are solvede as two unknowns. They are calculated within the original -1..1
            # interval and are not dimensional.. use inverse_transform_fit to give them
            # dimensions.
            
            # function for the establishment of truncated Gaussian shifting 
            # due to increasing maximum temperatures from
            # the original -1 .. 1 interval to a new interval S_m1 to S_1
            # this funciton is used by fsolve below
            
            # must normalize by duration
            # differentiate between the different months!!!!
            D10 = np.log((1/(Phwm * N10)))/np.log(Phwsm)  # in hours - expected value
            D50 = np.log((1/(Phwm * N50)))/np.log(Phwsm)  # in hours - expected value
        else:
            Davg_hw = np.log(0.5)/np.log(Phwsm)
            Davg_cs = np.log(0.5)/np.log(Pcssm)
            D10 = np.log(1/(Phwm*(N10 - N10*(Phwm*Davg_hw + Pcsm*Davg_cs))))/np.log(Phwsm)
            D50 = np.log(1/(Phwm*(N50 - N50*(Phwm*Davg_hw + Pcsm*Davg_cs))))/np.log(Phwsm)
        
        # IPCC values must be normalized per duration to assure the correct amount
        # is added.
        if use_global == False:
            abs_delT_10 = delT_ipcc_frac_month * ipcc_val_10[self._valid_increase_factor_tracks[increase_factor_ci][0]]
            abs_delT_50 = delT_ipcc_frac_month * ipcc_val_50[self._valid_increase_factor_tracks[increase_factor_ci][0]]
        else:
            abs_delT_10 = ipcc_val_10[self._valid_increase_factor_tracks[increase_factor_ci][0]]
            abs_delT_50 = ipcc_val_50[self._valid_increase_factor_tracks[increase_factor_ci][0]]
            
        
        if use_global == False:
            # The new model only shifts probabilities!
            del_mu_delT_max_hwm = 0 #npar[0]
            del_sig_delT_max_hwm = 0 #npar[1]
            # delta from -1...1 boundaries of the original transformed delT_max distribution.
            del_a_delT_max_hwm = 0
            del_b_delT_max_hwm = 0
        else:
            
            new_delT_10 = delTmax10_hwm + abs_delT_10/(
                norm_temp * (alphaT * (D10/norm_duration) + betaT))
            new_delT_50 = delTmax50_hwm + abs_delT_50/(
                norm_temp * (alphaT * (D50/norm_duration) + betaT))
            def F10_50_S(npar):
                
                mu_s,sig_s = npar
                S_m1 = -1 + mu_s - (1 + mu_norm)/(sig_norm) * sig_s
                S_1 = 1 + mu_s + (1 - mu_norm)/(sig_norm) * sig_s
                return [
                cdf_truncnorm(
                    transform_fit(new_delT_10,
                                    mintemp,
                                    maxtemp),
                    mu_norm + mu_s,
                    sig_norm + sig_s,
                    S_m1,
                    S_1) + S10
                ,
                cdf_truncnorm(
                    transform_fit(new_delT_50,
                                        mintemp,
                                        maxtemp),
                    mu_norm + mu_s,
                    sig_norm + sig_s,
                    S_m1,
                    S_1) + S50]
        
            
            mu_guess = transform_fit(new_delT_10,mintemp,maxtemp) - transform_fit(
                delTmax10_hwm,mintemp,maxtemp)
            
            # if the new_delT_10 and new_delT_50 are nearly identifical, then the
            # solution becomes unstable because only the mean is shifting and
            # the standard deviation does not matter.
            
            npar, infodict_s, ier_s, mesg_s = fsolve(F10_50_S, (mu_guess,0.0),full_output=True)
            
            if ier_s != 1:
                err_tol = 0.0001
                # if the new_delT_10 and new_delT_50 are nearly identifical, then the
                # solution becomes unstable because only the mean is shifting and
                # the standard deviation does not matter.
                # Also if deltaTG is nearly zero the solution doesn't matter as well.
                # it will be unussually small.
                # Otherwise, there is a problem with non-convergence.
                if(np.abs(new_delT_50 - new_delT_10) > err_tol) and delta_TG > err_tol:
                    raise ValueError("The solution for change in mean and standard deviation did not converge!")
            
            del_mu_delT_max_hwm = npar[0]
            del_sig_delT_max_hwm = npar[1]
        

            # delta from -1...1 boundaries of the original transformed delT_max distribution.
            del_a_delT_max_hwm = del_mu_delT_max_hwm - (1 + mu_norm)/(sig_norm) * del_sig_delT_max_hwm
            del_b_delT_max_hwm = del_mu_delT_max_hwm + (1 - mu_norm)/(sig_norm) * del_sig_delT_max_hwm
        
        # adjusted durations - assume durations increase proportionally with temperature duration
        # regression alpha_T, beta_T
        # temperature.
        
        if self.use_global:
            S_D_10 = new_delT_10 / delTmax10_hwm
        else:
            S_D_10 = 1+(abs_delT_10 * norm_duration)/(alphaT * norm_temp * D10)
        if S_D_10 < 1.0:
            raise ValueError("A decrease in 10 year durations is not expected for the current analysis!")
            
        if self.use_global:
            S_D_50 = new_delT_50 / delTmax50_hwm
        else:
            S_D_50 = 1+(abs_delT_50 * norm_duration) / (alphaT * norm_temp * D50)
        if S_D_50 < 1.0:
            raise ValueError("A decrease in 50 year durations is not expected for the current analysis!")
        
        D10_prime = D10 * S_D_10
        D50_prime = D50 * S_D_50
        
        if self.use_global:
            # old method being retained for replication of the Albuquerque Study.
            P_prime_hwsm = (N10 * Phwsm ** (1/S_D_50) + N50 * Phwsm ** (1/S_D_10))/(N10 + N50)
            delT_abs_max = abs_delT_50 
        else:
            # this new equation is correct!
            # 2 equations and two unknowns to enforce 10 and 50 year event durations
            # - 1. The hourly probability of a 10 year heat wave event is multiplied by the frequency multiplier and 
            #      increases in duration sufficiently to lead to a delT per IPCC's increase in intensity projection.
            # - 2. Same but for a 50 year event.
            #
            f_ipcc_ci_10 * Phwm * Phwsm**D10
            P_prime_hwsm = np.exp((np.log(f_ipcc_ci_50/f_ipcc_ci_10)+(D50-D10)*np.log(Phwsm))/(D50_prime - D10_prime))
            P_prime_hwm = f_ipcc_ci_10 * Phwm * (Phwsm)**D10/(P_prime_hwsm)**(D10_prime)
 
            # Verify that I did my algebra correctly
            #P_check = f_ipcc_ci_50 * Phwm * (Phwsm)**D50/(P_prime_hwsm)**(D50*S_D_50)
            
            delT_abs_max = abs_delT_50
            
        # equation 31 scaling of energy
        if self.use_global:
            epsilon = 1.0e-6
            if P_prime_hwsm+epsilon < Phwsm:
                raise ValueError("The probability of sustaining a heat wave has decreased. "+
                                 "This should not happen in the current analysis!")
            
            # equation 30 optimal scaling of D_HW_Pm
            S_D_m = np.log(Phwsm)/np.log(P_prime_hwsm)
            if S_D_m + epsilon < 1.0:
                raise ValueError("The scaling factor on heat wave sustainment"+
                                 " must be greater than 1.0")
            
            S_E_m = S_D_m * (new_delT_10/delTmax10_hwm + new_delT_50/delTmax50_hwm)/2
            
            del_mu_E_hw_m = transform_fit(
                S_E_m * inverse_transform_fit(
                    normalized_energy['mu'], 
                    hot_param['max energy per duration'], 
                    hot_param['min energy per duration'])
                ,hot_param['min energy per duration'],
                 hot_param['max energy per duration'])-normalized_energy['mu']     
            # transformation is not needed here, 
            # be careful here! inverse transform and transform have different 
            # orders for the min and max inputs!
            del_sig_E_hw_m = transform_fit(
                (inverse_transform_fit(
                    normalized_ext_temp['sig'] + del_sig_delT_max_hwm, 
                    maxtemp, 
                    mintemp)/
                  inverse_transform_fit(normalized_ext_temp['sig'],
                                              maxtemp, 
                                              mintemp))*
                  inverse_transform_fit(normalized_energy['sig'], 
                                              hot_param['max energy per duration'],
                                              hot_param['min energy per duration']),
                  hot_param['min energy per duration'],
                  hot_param['max energy per duration'])-normalized_energy['sig']
            
        else:
            # no changes to truncated Gaussian for new use case     
            del_mu_E_hw_m = 0.0
            del_sig_E_hw_m = 0.0
        
        # delta from the -1..1 boundaries of the transformed Energy distribution (still in transformed space but no 
        # longer on the -1...1 interval.)
        if self.use_global:
            del_a_E_hw_m = del_mu_E_hw_m - (1 + normalized_energy['mu'])/(normalized_energy['sig']) * del_sig_E_hw_m
            del_b_E_hw_m = del_mu_E_hw_m + (1 - normalized_energy['mu'])/(normalized_energy['sig']) * del_sig_E_hw_m
        else:
            del_a_E_hw_m = 0.0
            del_b_E_hw_m = 0.0
        
        # for the current work, assume cold snaps do not change with climate
        # TODO - add cold snap changes (decreases?)
        P_prime_csm = Pcsm
        P_prime_cssm = Pcssm
        
        # NEXT STEPS - GATHER ALL YOUR VARIABLES AND FORMULATE THE DELTA M matrix
        # RETURN THEM SO YOU CAN GET THEM INTO MEWS' original EXTREMES class.
        if abs(P_prime_hwm) > 1 or abs(P_prime_hwsm) > 1:
            raise ValueError("The adjusted probabilities must be less than one!")
    
        return (Phwm, Pcsm, P_prime_hwm, P_prime_csm, Pcssm, P_prime_cssm, Phwsm, 
         P_prime_hwsm, del_mu_E_hw_m, del_sig_E_hw_m, del_a_E_hw_m, 
         del_b_E_hw_m, del_mu_delT_max_hwm, del_sig_delT_max_hwm,
         del_a_delT_max_hwm, del_b_delT_max_hwm, delT_abs_max)


        
    
    
        
            
