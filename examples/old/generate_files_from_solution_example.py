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


"""
INPUTS:

"""
# These solution files require a multi-hour run using the 3 steps in worcester_example.py
# add to the list or use output from ExtremeTemperatureWaves.create_solutions to 
# make a file list.
solution_files = ['final_solution_2080_SSP370_95%.txt',  'final_solution_2080_SSP585_95%.txt']


# must provide the directory to the solution files 
inp_dir = os.path.join(os.path.dirname(__file__),"example_data","Worcester","solution_files")

# Station - Worcester Regional Airport - must reflect the same station as the solution files used
station = os.path.join(inp_dir,"..", "USW00094746.csv")

# solution output from step 2 of worcester_example.py
historical_solution = os.path.join(inp_dir,"worcester_historical_solution.txt")

# change this to the appropriate unit conversion (5/9, -(5/9)*32) is for F going to C
unit_conversion = (5/9, -(5/9)*32)
unit_conv_norms = (5/9, -(5/9)*32)

# gives consistency in run results
random_seed = 7293821

# this will be the number of files output per solution file need 100+ for statistical studies.
num_realization_per_scenario = 3 

# plot_results
plot_results = True
run_parallel = True
num_cpu = 20

# this is the file that serves as the base-weather 
weather_files = [os.path.join(inp_dir,"..", "USA_MA_Worcester.Rgnl.AP.725095_TMY3.epw")]

# you must have the polynomials for the specific location
#lat = 42.268
#lon = 360-71.8763

# this is the output of step 1 of worcester_example.py
scen_dict = {'historical': np.poly1d([1.35889778e-10, 7.56034740e-08, 1.55701410e-05, 1.51736807e-03,
                                      7.20591313e-02, 4.26377339e-06]),
             'SSP245': np.poly1d([-0.00019697,  0.04967771, -0.09146572]),
             'SSP370': np.poly1d([0.00017505,  0.03762937, -0.07095332]),
             'SSP585': np.poly1d([0.00027245, 0.05231527, 0.0031547])}


# No solve is happening but you still need to control a couple of variables. out_path indicates where all the output files will go.
solve_options = {'historic': {'plot_results': plot_results,
                              'out_path': os.path.join("mews_results", "worcester.png")},
                 'future': {'plot_results': plot_results,
                            'out_path': os.path.join("mews_results", "worcester_future.png")}}

"""

CODE

"""
# This enables skipping the first step if you need to write a bunch of files.
if not "obj" in globals():
    pass
obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=unit_conversion,
                              use_local=True, random_seed=random_seed,
                              include_plots=plot_results,
                              run_parallel=run_parallel, use_global=False, delT_ipcc_min_frac=1.0,
                              num_cpu=num_cpu, write_results=True, test_markov=False,
                              solve_options=None,
                              norms_unit_conversion=unit_conv_norms,
                              solution_file=historical_solution)

for file in solution_files:
    brstr = file.split(".")[0].split("_")
    
    year = int(brstr[2])
    scen_name = brstr[3]
    cii = brstr[-1]
    
    obj.create_scenario(scenario_name=scen_name,
                        year=year,
                        climate_temp_func=scen_dict,
                        num_realization=num_realization_per_scenario,
                        climate_baseyear=2014,
                        increase_factor_ci=cii,
                        cold_snap_shift=None,
                        solution_file=os.path.join(inp_dir,file))
    
"""
output - results .epw will be output to the mews_results file that is created 
         Also output are *.png showing the closeness of fit, and *.csv with
         all of the actual data output for the *.png files so that someone 
         can plot the closeness of fit themselves.

"""