#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:22:21 2022

@author: dlvilla


This is a script for applying the MEWS model to Worcester,MA

It is accomplished through 3 steps that a are manually run and adjusted

rather than through a single run. Start by setting "step = 1".

STEP1 - set "step = 1" 

"""
import os
import numpy as np
from mews.weather.climate import ClimateScenario
from mews.events import ExtremeTemperatureWaves
import pickle as pkl
from mews.utilities.utilities import bin_avg
from mews.stats.solve import shift_a_b_of_trunc_gaussian
from copy import deepcopy

def calculate_shift(obj2,first_try_solution_location,month_factors,sol_dir):

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
    obj2.write_solution(os.path.join(sol_dir,"iterate.txt"))


if __name__ == "__main__":
    """
    INPUT DATA
    
    adjust all of these to desired values if analyzing a new location. Paths have
    to be set and folders created if functioning outside of the MEWS repository 
    structure.
    
    
    
    """
    
    step = 3
    # STEP 1 to using MEWS, create a climate increase in surface temperature
    #        set of scenarios,
    #
    # you must download https://osf.io/ts9e8/files/osfstorage or else
    # endure the process of all the CMIP 6 files downloading (I had to restart ~100 times)
    # with proper proxy settings.

    # HERE CS stands for climate scenario (not cold snap)
    future_years = [2020,2040,2060,2080] #2014, 2020, 2040, 2060, 

    # CI interval valid = ['5%','50%','95%']
    ci_intervals = ["5%","50%","95%"]
    
    lat = 42.268
    lon = 360-71.8763

    # Station - Worcester Regional Airport
    station = os.path.join("example_data", "Worcester", "USW00094746.csv")
    
    first_try_solution_location = os.path.join("example_data","Worcester","results","worcester_historical_solution.txt")

    # change this to the appropriate unit conversion (5/9, -(5/9)*32) is for F going to C
    unit_conversion = (5/9, -(5/9)*32)
    unit_conv_norms = (5/9, -(5/9)*32)

    # gives consistency in run results
    random_seed = 7293821

    num_realization_per_scenario = 10

    # plot_results
    plot_results = True
    run_parallel = True
    num_cpu = 20

    # No need to input "historical"
    # valid names for scenarios are: ["historical","SSP119","SSP126","SSP245","SSP370","SSP585"]
    scenarios = ['SSP585',"SSP245", 'SSP370'] 
    output_folder_cs = os.path.join(
        "example_data", "Worcester", "climate_scenario_results")

    weather_files = [os.path.join(
        "example_data", "Worcester", "USA_MA_Worcester.Rgnl.AP.725095_TMY3.epw")]
    
    """
    STEP 1
    
    Calculate the polynomials for Global Warming based on the selected CMIP6 ensemble.
    
    """
    
    if step == 1:
        cs_obj = ClimateScenario(use_global=False,
                                 lat=lat,
                                 lon=lon,
                                 end_year=future_years[-1],
                                 output_folder=output_folder_cs,
                                 write_graphics_path=output_folder_cs,
                                 num_cpu=num_cpu,
                                 run_parallel=run_parallel,
                                 model_guide=os.path.join(
                                     "example_data", "Models_Used_alpha.xlsx"),
                                 data_folder=os.path.join(
                                     "..", "..", "CMIP6_Data_Files"),
                                 align_gcm_to_historical=True)
        scen_dict = cs_obj.calculate_coef(scenarios)

        if not os.path.exists("temp_pickles"):
            os.mkdir("temp_pickles")
        pkl.dump([cs_obj], open(os.path.join(
            "temp_pickles", "cs_obj.pickle"), 'wb'))

    else:
        # this is the results obtained by dlvilla 10/31/2022
        scen_dict = {'historical': np.poly1d([1.35889778e-10, 7.56034740e-08, 1.55701410e-05, 1.51736807e-03,
                                              7.20591313e-02, 4.26377339e-06]),
                     'SSP245': np.poly1d([-0.00019697,  0.04967771, -0.09146572]),
                     'SSP370': np.poly1d([0.00017505,  0.03762937, -0.07095332]),
                     'SSP585': np.poly1d([0.00027245, 0.05231527, 0.0031547])}

        cs_obj = pkl.load(
            open(os.path.join("temp_pickles", "cs_obj.pickle"), 'rb'))[0]
    """
    # STEP 2. FIT THE HISTORIC DATA.
    # USE THESE TO CONTROL OPTIMIZATION...
    # You must provide a proxy.txt file if needed or change proxy=None if the computer
    # system does not have a proxy server.
    """
    if step == 2:
        solve_options = {'historic': {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                      'decay_func_type': {'cs': 'quadratic_times_exponential_decay_with_cutoff', 'hw': "quadratic_times_exponential_decay_with_cutoff"},
                                      'max_iter': 30,
                                      'limit_temperatures': False,
                                      'num_cpu': -1,
                                      'plot_results': plot_results,
                                      'num_step': 2500000,
                                      'test_mode': False,
                                      'min_num_waves': 10,
                                      'weights': np.array([1, 1, 1, 1, 1]),
                                      'out_path': os.path.join("example_data", "Worcester", "results", "worcester.png")},
                         'future': {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                    'max_iter': 30,
                                    'limit_temperatures': False,
                                    'num_cpu': -1,
                                    'num_step': 2500000,
                                    'plot_results': plot_results,
                                    'decay_func_type': {'cs': 'quadratic_times_exponential_decay_with_cutoff', 'hw': "quadratic_times_exponential_decay_with_cutoff"},
                                    'test_mode': False,
                                    'min_num_waves': 10,
                                    'out_path': os.path.join("example_data", "Worcester", "results", "worcester_future.png")}}

        # I had to manually add the daily summaries and norms
        obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=unit_conversion,
                                      use_local=True, random_seed=random_seed,
                                      include_plots=plot_results,
                                      run_parallel=run_parallel, use_global=False, delT_ipcc_min_frac=1.0,
                                      num_cpu=num_cpu, write_results=True, test_markov=False,
                                      solve_options=solve_options, proxy=os.path.join(
                                          "..", "..", "proxy.txt"),
                                      norms_unit_conversion=unit_conv_norms)
        
        # add quadratic_times_exponential_decay_with_cutoff
        obj.write_solution(first_try_solution_location)
        
        
        if not os.path.exists("temp_pickles"):
            os.mkdir("temp_pickles")
        pkl.dump([obj, solve_options], open(
            os.path.join("temp_pickles", "obj.pickle"), 'wb'))
    else:

        blob = pkl.load(open(os.path.join("temp_pickles", "obj.pickle"), 'rb'))
        obj = blob[0]
        solve_options = blob[1]


    """
    STEP 3
    
    Shift historic distributions to future
    
    """
    if step == 3:
    
        results = {}
        #     _bound_desc = ['delT_mu',
                           # 'delT_sig multipliers',
                           # 'P_event', 'P_sustain',
                           # 'multipliers to max probability time',
                           # 'slope or exponent multipliers',
                           # 'cutoff time multipliers',
                           # 'max peak prob for quadratic model',
                           # 'delay_time_multipliers']
        #
        
        # obj2.solve_options['future']['problem_bounds'] = {'cs':{'delT_mu': (-4.0, 10.0),
        #                      'delT_sig multipliers': (0.05,3),
        #                      'P_event': (0.001, 0.02),
        #                      'P_sustain': (0.97, 0.999999),
        #                      'multipliers to max probability time': (0,1),
        #                      'cutoff time multipliers':(2,3),
        #                      'max peak prob for quadratic model': (0.98, 1.0)},
        #                'hw':{'delT_mu': (-4.0, 10.0),
        #                      'delT_sig multipliers': (-2,3),
        #                      'P_event': (0.0005,0.02),
        #                      'P_sustain': (0.97,0.999999),
        #                      'multipliers to max probability time': (0.1,2),
        #                      'cutoff time multipliers':(1,3),
        #                      'max peak prob for quadratic model': (0.97, 1.0)}}
        # obj2.solve_options['future']['max_iter'] = 30
        # obj2.solve_options['future']['num_cpu'] = -1
        # obj2.solve_options['future']['weights'] = np.array([3.0,1,1,1,3.0])
        # 

        sol_dir = os.path.dirname(first_try_solution_location)
        
        # still working on this!
        # this is the longest step.
        for syear in future_years:
            results[syear] = {}
            for scen in scenarios:
                results[syear][scen] = {}
                for cii in ci_intervals:
                    
                    # restart capability
                    if syear in results:
                        if scen in results[syear]:
                            if cii in results[syear][scen]:
                                continue
                            
                    # start fresh with the historical solution
                    obj2 = deepcopy(obj)
                    
                    if not os.path.exists(first_try_solution_location):
                        obj2.write_solution(os.path.join(first_try_solution_location))
                    
                    obj2.solve_options['future']['num_postprocess'] = 1
                    
                    file_path = first_try_solution_location
                
                    # Question - do you want to look across the extreme event confidence
                    #            intervals or just stick to the 50%?
                    #          - do you want a cold snap shift? I don't have information for
                    #            how much cold snaps will change with increasing global warming
                    
                    # THIS IS THE HISTORICAL SOLUTION
                    obj2._write_results = False
                    results[syear][scen][cii] = obj2.create_scenario(scenario_name=scen,
                                                                    year=syear,
                                                                    climate_temp_func=scen_dict,
                                                                    num_realization=num_realization_per_scenario,
                                                                    climate_baseyear=2014,
                                                                    increase_factor_ci=cii,
                                                                    cold_snap_shift=None,
                                                                    solution_file=file_path)
                    # These must be one for the calculation to work!
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
                    
                    obj2_hist = deepcopy(obj2)
                    # TAKE A CALCULATED SHIFT THAT HAS LINEAR VARIATION
                    calculate_shift(obj2,first_try_solution_location,month_factors,sol_dir)
                    
                    #month_factors = [1.6,
                    #                 2.5,
                    #                 2.5,
                    #                 2.5,
                    #                 2.0,
                    #                 1.75,
                    #                 3.0,
                    #                 2.5,
                    #                 3.0,
                    #                 2.5,
                    #                 2.25,
                    #                 2.25]

                    
                    
                    results[syear][scen][cii] = obj2.create_scenario(scenario_name=scen,
                                                                    year=syear,
                                                                    climate_temp_func=scen_dict,
                                                                    num_realization=num_realization_per_scenario,
                                                                    climate_baseyear=2014,
                                                                    increase_factor_ci=cii,
                                                                    cold_snap_shift=None,
                                                                    solution_file=os.path.join(sol_dir,"iterate.txt"))
            

                    
                    new_month_factors = []
                    for month in np.arange(1,13):
                        h_thresh = obj2_hist.future_solve_obj[scen][cii][syear][month].obj_solve.thresholds[1]['hw']
                        f_thresh = obj2.future_solve_obj[scen][cii][syear][month].obj_solve.thresholds[1]['hw']
                        
                        h_gap = 0.5*((h_thresh['target'][0] - h_thresh['actual'][0]) + (h_thresh['target'][1] - h_thresh['actual'][1]))       
                        f_gap = 0.5*((f_thresh['target'][0] - f_thresh['actual'][0]) + (f_thresh['target'][1] - f_thresh['actual'][1]))
                        
                        new_month_factors.append(h_gap / (h_gap - f_gap))
                        
                    # now produce a solution that is reasonably good only based on changing temperature distribution parameters from the 
                    # historic solution.

                    calculate_shift(obj2_hist,first_try_solution_location,new_month_factors,sol_dir)
                    
                    
                    results[syear][scen][cii] = obj2_hist.create_scenario(scenario_name=scen,
                                                                    year=syear,
                                                                    climate_temp_func=scen_dict,
                                                                    num_realization=num_realization_per_scenario,
                                                                    climate_baseyear=2014,
                                                                    increase_factor_ci=cii,
                                                                    cold_snap_shift=None,
                                                                    solution_file=os.path.join(sol_dir,"iterate.txt"))
                    obj2_hist.write_solution(os.path.join(sol_dir,"final_solution_{0:d}_{1}_{2}.txt".format(syear,scen,cii)))
                    
                    
                    
                        
                        
                    #pkl.dump([obj2,results],open(
                    #    os.path.join("temp_pickles", "worcester_incomplete_results.pickle"), 'wb'))
        #pkl.dump([obj,results],open(
        #    os.path.join("temp_pickles", "worcester_results.pickle"), 'wb'))
                

    #
