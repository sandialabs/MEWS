#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:24:45 2022

This shows non-MEWS users how to take a solution from the worcester example
and to create EnergyPlus files using it.

@author: dlvilla
"""

import os
from mews.events import ExtremeTemperatureWaves
import numpy as np
from numpy import poly1d
from numpy.random import default_rng
import pickle as pkl

"""
INPUTS:

"""
# These solution files require a multi-hour run using the 3 steps in worcester_example.py
# add to the list or use output from ExtremeTemperatureWaves.create_solutions to 
# make a file list.

# Already done
"""

'worcester_final_solution_2080_SSP585_95%.txt'


'worcester_final_solution_2020_SSP585_5%.txt',
'worcester_final_solution_2040_SSP370_50%.txt',
'worcester_final_solution_2040_SSP585_50%.txt',
'worcester_final_solution_2080_SSP245_95%.txt',
'worcester_final_solution_2060_SSP585_50%.txt',
'worcester_final_solution_2080_SSP585_50%.txt',
'worcester_final_solution_2080_SSP245_50%.txt',
'worcester_final_solution_2060_SSP245_95%.txt',
'worcester_final_solution_2060_SSP370_50%.txt',
'worcester_final_solution_2080_SSP370_50%.txt',
'worcester_final_solution_2080_SSP370_95%.txt',
'worcester_final_solution_2080_SSP585_95%.txt',
'worcester_final_solution_2060_SSP585_95%.txt',
'worcester_final_solution_2060_SSP585_5%.txt',
'worcester_final_solution_2080_SSP585_5%.txt',
'worcester_final_solution_2060_SSP245_50%.txt',
'worcester_final_solution_2040_SSP245_95%.txt',
'worcester_final_solution_2040_SSP585_95%.txt',
'worcester_final_solution_2040_SSP370_95%.txt',
'worcester_final_solution_2020_SSP245_95%.txt',
'worcester_final_solution_2020_SSP370_95%.txt',
'worcester_final_solution_2080_SSP245_5%.txt',
'worcester_final_solution_2020_SSP370_50%.txt',
'worcester_final_solution_2040_SSP245_5%.txt',
'worcester_final_solution_2020_SSP585_50%.txt',
'worcester_final_solution_2020_SSP245_5%.txt',
'worcester_final_solution_2040_SSP585_5%.txt',
'worcester_final_solution_2020_SSP370_5%.txt',
'worcester_final_solution_2020_SSP585_95%.txt',
'worcester_final_solution_2060_SSP370_5%.txt',
'worcester_final_solution_2080_SSP370_5%.txt',
'worcester_final_solution_2060_SSP370_95%.txt',
'worcester_final_solution_2060_SSP245_5%.txt',
'worcester_final_solution_2040_SSP245_50%.txt',
'worcester_final_solution_2040_SSP370_5%.txt',
'worcester_final_solution_2020_SSP245_50%.txt'

"""

solution_files = ['worcester_final_solution_2020_SSP585_5%.txt',
'worcester_final_solution_2040_SSP370_50%.txt',
'worcester_final_solution_2040_SSP585_50%.txt',
'worcester_final_solution_2080_SSP245_95%.txt',
'worcester_final_solution_2060_SSP585_50%.txt',
'worcester_final_solution_2080_SSP585_50%.txt',
'worcester_final_solution_2080_SSP245_50%.txt',
'worcester_final_solution_2060_SSP245_95%.txt',
'worcester_final_solution_2060_SSP370_50%.txt',
'worcester_final_solution_2080_SSP370_50%.txt',
'worcester_final_solution_2080_SSP370_95%.txt',
'worcester_final_solution_2080_SSP585_95%.txt',
'worcester_final_solution_2060_SSP585_95%.txt',
'worcester_final_solution_2060_SSP585_5%.txt',
'worcester_final_solution_2080_SSP585_5%.txt',
'worcester_final_solution_2060_SSP245_50%.txt',
'worcester_final_solution_2040_SSP245_95%.txt',
'worcester_final_solution_2040_SSP585_95%.txt',
'worcester_final_solution_2040_SSP370_95%.txt',
'worcester_final_solution_2020_SSP245_95%.txt',
'worcester_final_solution_2020_SSP370_95%.txt',
'worcester_final_solution_2080_SSP245_5%.txt',
'worcester_final_solution_2020_SSP370_50%.txt',
'worcester_final_solution_2040_SSP245_5%.txt',
'worcester_final_solution_2020_SSP585_50%.txt',
'worcester_final_solution_2020_SSP245_5%.txt',
'worcester_final_solution_2040_SSP585_5%.txt',
'worcester_final_solution_2020_SSP370_5%.txt',
'worcester_final_solution_2020_SSP585_95%.txt',
'worcester_final_solution_2060_SSP370_5%.txt',
'worcester_final_solution_2080_SSP370_5%.txt',
'worcester_final_solution_2060_SSP370_95%.txt',
'worcester_final_solution_2060_SSP245_5%.txt',
'worcester_final_solution_2040_SSP245_50%.txt',
'worcester_final_solution_2040_SSP370_5%.txt',
'worcester_final_solution_2020_SSP245_50%.txt']

# must provide the directory to the solution files 
inp_dir = os.path.join(os.path.dirname(__file__),"example_data","Worcester","results2_2x")

# Station - Worcester Regional Airport - must reflect the same station as the solution files used
station = os.path.join(inp_dir,"..", "USW00094746.csv")

# solution output from step 2 of worcester_example.py
historical_solution = os.path.join(inp_dir,"worcester_historical_solution.txt")

# change this to the appropriate unit conversion (5/9, -(5/9)*32) is for F going to C
unit_conversion = (5/9, -(5/9)*32)
unit_conv_norms = (5/9, -(5/9)*32)

# gives consistency in run results
random_seedling = 7293821
rng = default_rng(random_seedling)

# this will be the number of files output per solution file need 100+ for statistical studies.
num_realization_per_scenario = 100

# plot_results
plot_results = True
run_parallel = True
num_cpu = 6
num_pool_cpu = 10

# this is the file that serves as the base-weather 
weather_files = [os.path.join(inp_dir,"..", "USA_MA_Worcester.Rgnl.AP.725095_TMY3.epw")]

# you must have the polynomials for the specific location
#lat = 42.268
#lon = 360-71.8763

# this is the output of step 1 of worcester_example.py
scen_dict = {'historical': poly1d([1.35889778e-10, 7.56034740e-08, 1.55701410e-05, 1.51736807e-03,
        7.20591313e-02, 4.26377339e-06]),
 'SSP585': poly1d([ 4.00596448e-04,  4.52765361e-02, -6.44939852e-07]),
 'SSP245': poly1d([-6.96226200e-05,  4.12639281e-02, -7.51405631e-08]),
 'SSP370': poly1d([ 2.23138299e-04,  3.47465086e-02, -7.67683705e-07])}


# No solve is happening but you still need to control a couple of variables. out_path indicates where all the output files will go.
solve_options = {'historic': {'plot_results': plot_results,
                              'out_path': os.path.join("mews_results_2x", "worcester.png")},
                 'future': {'plot_results': plot_results,
                            'out_path': os.path.join("mews_results_2x", "worcester_future.png")}}

"""

CODE

"""
obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=unit_conversion,
                          use_local=True, random_seed=random_seedling,
                          include_plots=plot_results,
                          run_parallel=run_parallel, use_global=False, delT_ipcc_min_frac=1.0,
                          num_cpu=num_cpu, write_results=True, test_markov=False,
                          solve_options=solve_options,
                          norms_unit_conversion=unit_conv_norms,
                          solution_file=historical_solution)


hist_stats = obj.read_solution(historical_solution)

if run_parallel:
    import multiprocessing as mp
    pool = mp.Pool(num_pool_cpu)

results = {}
for file in solution_files:
    filepath = os.path.join(inp_dir,file)
    
    ## cleaning up an error I made - DELETE AFTER VERIFICATION
    # fstats = obj.read_solution(filepath)
    # for wtype, subdict in fstats.items():
    #     for month, subsub in subdict.items():
    #         if not "hist max energy per duration" in subsub:
    #             fstats[wtype][month]['hist max energy per duration'] = hist_stats[wtype][month]['hist max energy per duration']
    #         if not "hist min energy per duration" in subsub:
    #             fstats[wtype][month]['hist min energy per duration'] = hist_stats[wtype][month]['hist min energy per duration']
    # obj.stats = fstats
    # obj.write_solution(filepath, is_historic=False)     
    
    random_seed = int(rng.random(1)[0]*1000000)
    brstr = file.split(".")[0].split("_")
    
    year = int(brstr[3])
    scen_name = brstr[4]
    cii = brstr[-1]

    args = (scen_name,
            year,
            scen_dict,
            num_realization_per_scenario,
            2014,
            cii,
            None,
            filepath,
            random_seed,
            "mews_results_2x")
    
    if run_parallel:
        results[file] = pool.apply_async(obj.create_scenario,
                                         args=args)
    else:
        results[file] = obj.create_scenario(*args)

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
else:
    results_get = results
    
pkl.dump([results_get],open("temp_pickles/obj_worcester_alter.pickle",'wb'))
    
    
"""
output - results .epw will be output to the mews_results file that is created 
         Also output are *.png showing the closeness of fit, and *.csv with
         all of the actual data output for the *.png files so that someone 
         can plot the closeness of fit themselves.

"""