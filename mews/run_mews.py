#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:55:01 2023

This is the main interface to create a set of solution files via an input script

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

"""
import os
import numpy as np
from numpy import poly1d
from mews.weather.climate import ClimateScenario
from mews.events import ExtremeTemperatureWaves
from mews.utilities.utilities import (bin_avg, read_readable_python_dict, 
                                      create_smirnov_table, write_readable_python_dict)
from mews.constants.data_format import (VALID_RUN_MEWS_ENTRIES, 
                                        REQUIRED_STRING, 
                                        TEMPLATE_ID_STRING,
                                        SOLVE_OPTIONS_STRING)

from copy import deepcopy
from mews.errors.exceptions import MEWSInputTemplateError
from mews.utilities.utilities import filter_cpu_count
import pickle as pkl
import pandas as pd


def _check_and_arrange_run_dict_inputs(run_dict):
    """
    

    Parameters
    ----------
    run_dict : run_dict as described in run_mews_extreme_temperature
    
    Raises
    ------
    TypeError - indicates that an input key value in run_dict is not the correct type
    
    MEWSInputTemplateError - indicates that the input template name is not 
                             art of the input dictionary
    
    ValueError - Indicates that a required value in the input is not present

    Returns
    -------
    run_dict_modified - a completed input object that has template values for
                        anything missing from run_dict or has default values
                        if a value is not required.

    """
    
    
    
    def _type_error(run_name,entry_name,type_check):
        raise TypeError("The run name '{0}'".format(run_name) 
                        + " has an erroneous type for key "
                        + "'{0}'".format(entry_name) + " it "
                        + "must be of type(s) '{0}'".format(type_check))
    
    def _missing_required_value_error(run_name,entry_name):
        raise MEWSInputTemplateError("The run name '{0}'".format(run_name) 
                        + " has a missing required key "
                        + "'{0}'".format(entry_name) + " please add this required key to the inputs")
        
        
    master_run_dict_modified = {}
    for run_name, run_val in run_dict.items():
        # place template values in a template
        if TEMPLATE_ID_STRING in run_val:
            if run_val[TEMPLATE_ID_STRING] in run_dict:
                run_val_template = run_dict[run_val[TEMPLATE_ID_STRING]]
            else:
                raise ValueError("INPUT TEMPLATE NAME ERROR: The {0} entry must be another key within".format(TEMPLATE_ID_STRING)
                                 +" the mews input dictionary! The run name '{0}'".format(run_name) +
                                 " has a '{0}' value of '{1}' but".format(TEMPLATE_ID_STRING,
                                                                          run_val[TEMPLATE_ID_STRING]) +
                                 " this is not contained in the dictionary"+
                                 " key equal to \n\n:{0} \n\nPlease fix".format(str(run_dict.keys()))+
                                 " the erroneous input!")
            
        else:
            run_val_template = {}
        
        
        run_dict_modified = {}
        for entry_name, default_val in VALID_RUN_MEWS_ENTRIES.items():
            required_status = default_val[0]
            type_check = default_val[1]
            for tdict in [run_val,run_val_template]:                        
                if entry_name in tdict:
                    if not isinstance(tdict[entry_name],type_check):
                        _type_error(run_name,entry_name,type_check)
                    run_dict_modified[entry_name] = tdict[entry_name]
                    break
            if (not entry_name in run_dict_modified) and (
                    required_status == REQUIRED_STRING):
                _missing_required_value_error(run_name,entry_name)
            elif (not entry_name in run_dict_modified) and (
                    required_status != REQUIRED_STRING):
                default = required_status
                run_dict_modified[entry_name] = default
                
        if SOLVE_OPTIONS_STRING in run_val and SOLVE_OPTIONS_STRING in run_val_template:
            if "future" not in run_val[SOLVE_OPTIONS_STRING] or (
               "historic" not in run_val[SOLVE_OPTIONS_STRING]):
                
                raise KeyError("The {0} entry must have key=['future','historic']".format(SOLVE_OPTIONS_STRING))
            else:
                for vald in ['future','historic']:
                    for keyf,valf in run_val_template[SOLVE_OPTIONS_STRING][vald].items():
                        if not keyf in run_val[SOLVE_OPTIONS_STRING][vald]:
                            run_dict_modified[SOLVE_OPTIONS_STRING][vald][keyf] = valf
        master_run_dict_modified[run_name] = run_dict_modified
                    
    
    return master_run_dict_modified


def generate_epw_files(obj,solution_files,solution_path,num_files_per_solution,
                       output_dir,random_seed,scen_dict,run_parallel,num_cpu):
    """
    
    For a list of solution files, generate EnergyPlus .epw weather files.
    that only have dry-bulb temperature altered per the MEWS algorithm.
    
    This function will overwrite any prevous results

    Parameters
    ----------
    obj : mews.events.ExtremeTemperatureWaves 
        An object of the ExtremeTemperatureWaves class that has already
        had obj.create_solutions run.    
    solution_files : list
        A list of solution file names (no path) that are contained in 
        solution_path
    solution_path : str
        A string giving the path to a folder that contains all names
        in solution_files
    num_files_per_solution : int
        The number of ".epw" realization files to generate for each solution file
    output_dir : str
        Path to location that .epw files will be output to. 
    random_seed : int
        A pseudo-random number generator random seed
    scen_dict : dict
        a dictionary containing keys for climate scenarios with numpy.poly1d
        values that represent a polynomial for the average of the cmip6 ensemble
        specified for the mews run.

    Returns
    -------
    epw_file_list - a list of all EnergyPlus files inside output_dir. This may be
                more than those just generated by this function if other
                epw files were generated at an earlier time.
    
    This function creates EnergyPlus files in output_dir

    """
    
    if run_parallel:
        import multiprocessing as mp
        pool = mp.Pool(num_cpu)
    
    results = {}
    for file in solution_files:

        brstr = file.split(".")[0].split("_")
        
        year = int(brstr[-3])
        scen_name = brstr[-2]
        cii = brstr[-1]

        args = (scen_name,
                year,
                scen_dict,
                num_files_per_solution,
                2014,
                cii,
                None,
                os.path.join(solution_path,file),
                random_seed,
                output_dir)
        
        if run_parallel:
            results[file] = pool.apply_async(obj.create_scenario,
                                             args=args)
        else:
            obj.create_scenario(*args)

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
    

    epw_file_list = [os.path.join(os.path.abspath(output_dir),file) for file 
                     in os.listdir(output_dir) if ".epw" in file]
    
    return epw_file_list, results_get


def extreme_temperature(run_dict, run_dict_var, run_parallel=True, num_cpu=-1, 
                                 plot_results=True,generate_epw=True,proxy=None,
                                 abs_path_start="",only_generate_files=[],
                                 skip_runs=[]):

    """
    This function Runs MEWS for a parameter study of daily summary stations
    either defined as a dictionary in python or else placed as text file
    with the dictionary text in it. The input is complex and is thoroughly
    documented below.
    
    This function will overwrite any prevous results
    
    Parameters
    ----------
    
    run_dict : str or dict : Required
        Either 
        
        1) A string path to an ASCII file that contains a python dictionary string
           that can be read directly to a dictionary described in 2)
               
                 or 
                                
        2) A dictionary of dictionaries that has entries with all of the information needed
        to make one or more mews runs using the ClimateScenario and
        EtremeTemperatureWaves classes:
        
        The text below surrounds an entry that needs to be given a Python input
        by "<>" 
        
        {<RunName1 - str a unique identifier for a run>: 
         
             {"template_id": <(optional) a RunName inside this dictionary that
                              will populate a new entry. Only entries that
                              need to be changed from this template need to be
                              input>,
              "future_years": <a list of integer future years>,
              "ci_intervals": <a list of any of the set ["5%", "50%", "95%"]>, 
              "latitude_longitude" : <a tuple of latitude (N) and longitude (E)
                                      take extra care to give the North (N) and
                                      east (E) orientation to this vaule. Many
                                      times a (W) orientation is given for longitude.>,
            "scenarios" :         < SSP scenarios list to create polynomial fits
                                    and weather files for. Valid entries are:
                                   ["historical","SSP119","SSP126","SSP245",
                                    "SSP370","SSP585"]     
            "polynomial_order" :   < (optional) A dictionary that has every scenario in "scenarios"
                                 included. Each entry must be an integer indicating 
                                 what order of polynomial to fit CMIP6 data with
                                 e.g. recommended values={'historical':7,
                                                          'SSP585':4,
                                                          'SSP245':4,
                                                          'SSP370':4}
            "weather_files" :    (optional) < a list of weather files to use as the base weather
                                  for outputting MEWS realizations if nothing is given
                                  then weather files will not be generated. >,
            "num_files_per_solution" : < an integer that indicates how many mews 
                                         realizations to output >, 
            "daily_summaries_path" : <a string that gives a path to at least 
                                      50 years of NOAA daily summary data. >
            "climate_normals_path" : <a string that gives a path to the 1991-2020
                                      climate normals for the same location as the
                                      daily summaries dataset.>,
            "daily_summaries_unit_conversion": <tuple to convert daily summary data
                                     to degrees Celcius F->C = (5/9, -(5/9)*32),
                                     give a value of 'None' if no conversion is needed>,
            "climate_normals_unit_conversion": <tuple to convert climate norms data
                                     to degrees Celcius F->C = (5/9, -(5/9)*32),
                                     give a value of 'None' if no conversion is needed>,
            "historic_solution_save_location": <a string that gives a path to the location
                                     historic fits will be output future solutions will also
                                     be output to this location along with plots and csv files
                                     of the results.>,
            "random_seed" : <An integer used for generating pseodorandom numbers,
            "cmip6_model_guide" : <(optional) string path to a model guide xlsx spreadsheet for CMIP6 
                                models to use. The mews default is included in the repository.
                             mews default is in "examples/example_data/"
            "cmip6_data_folder" : <string path to a folder that contains all of the
                                   cmip6 surface temperature data. If not downloaded,
                                   this data has been stored at 
                                   
                                   https://osf.io/ts9e8/files/osfstorage
                                   
                                   which is 24Gb of data.

            "solve_options": <A dictionary with two entries "historic" and "future"
                                    which are both valid solve_options inputs for the
                                    SolveDistributionShift class: e.g.
                                    
                                    {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                     'decay_func_type': {'cs': 'quadratic_times_exponential_decay_with_cutoff', 
                                                         'hw': "quadratic_times_exponential_decay_with_cutoff"},
                                     'max_iter': 30,
                                     'limit_temperatures': False,
                                     'num_cpu': -1,
                                     'plot_results': plot_results,
                                     'num_step': 2500000,
                                     'test_mode': False,
                                     'min_num_waves': 10,
                                     'weights': np.array([1, 1, 1, 1, 1]),
                                     'out_path': os.path.join("example_data", "Kodiak", "results", "kodiak.png")}
                                    
                                   if specific values are given such as 'num_cpu', 
                                   these override the values given as input elsewhere>,
                                   
            "clim_scen_out_folder" : <string path to a folder name that will be created
                                      and climate scenario outputs written to it this will only
                                      be a filename in the current directory unless it is given
                                      more than one component name.>,
            "epw_out_folder" : <string path to a folder name that will be created
                                and where all EPW files will be output for RunName1
                                
            ""
    run_dict_var : dict : (Optional) Default = {}
        Only useful if accessing run_dict through a python readable dictionary       
        Provides a way to introduce text variables into the run_dict file.
        key value is a variable name that will be in the text of the file
        val is the value that will be used in the dictionary.
           
                                   
    run_parallel : bool : (Optional) DEfault = True
           Indicate whether to use parallel processing. MEWS takes very long 
           on a single processor (weeks?). Not running parallel is mostly for
           debugging purposes.
           
    num_cpu : int : (Optional) Default = -1,
          number of cpu's to run mews on. -1 indicates to use the maximum number
          of processors available less one.
          
    plot_results : bool (Optional) Default = True
          If you want output graphics leave as True
    
    generate_epw : bool (Optional) Default = True
          Determines if .epw EnergyPlus output files will be written. 
          If False, then generate_epw_files function can be used to 
          generate the files. This is desirable if the mews solution
          is run on one computer but that the files (which use a lot of 
          memory storage) need to be generated elsewhere. The solution_files
          are easy to send to someone via email whereas the .epw files take
          up many Gb of data.
          
    proxy : str (optional) : Default = None
          path to a text file that contains a proxy server name
          This is only needed if MEWS is used to directly download climate norms
          , daily summaries or CMIP6 data. These features are not tested and may
          not work. It is recommended that files be downloaded manually and
          put in local folders for a mews analysis.
          
    only_generate_files : list :Default = []
         Enter any name in the run_dict dictionary key and that run
         will skip the historic and future solution steps that take a long time
         and just generate the files. This is for runs that have already been
         finished but that need new files to be generated.
         
    skip_runs : list : Default = []
         Any run name in run_dict entered here will be completely skipped.
         This is useful if one run completed but another failed and you do 
         not want to have to reconfigure the input deck.
          
    Raises
    ------
        This function tries on each RunName given in run_dict. If an error
        is raised, it takes the exception object and puts it into the results
        in place of the actual results and moves on to the next RunName
        
        
    Returns
    -------
        This function creates solution files, csv files of historic and shifted
        heat wave distributions, and plots output that illustrates how good
        the fits are. It also outputs 
    
    """
    # only change this to true for debugging. The program will not work unless
    # you have stepped through the solution and created specific pickle file names
    troubleshoot = False
    
    if isinstance(run_dict,str):
        run_dict = read_readable_python_dict(run_dict,run_dict_var)
        
    run_dict_modified = _check_and_arrange_run_dict_inputs(run_dict)
    num_cpu = filter_cpu_count(num_cpu)
    results = {}
    
    for run_name, run_val in run_dict_modified.items():
        # Keep on going if one run fails.
        try:
            # unpack
            future_years = run_val["future_years"]
            ci_intervals = run_val["ci_intervals"]
            latlon = run_val["latitude_longitude"]
            scenarios = run_val["scenarios"]
            lat = latlon[0]
            lon = latlon[1]
            random_seed = run_val["random_seed"]
            polynomial_order = run_val["polynomial_order"]
            weather_files = run_val["weather_files"]
            historic_solution_save_location = run_val["historic_solution_save_location"]
            clim_scen_out_folder = run_val["clim_scen_out_folder"]
            cmip6_model_guide = run_val["cmip6_model_guide"]
            cmip6_data_folder = run_val["cmip6_data_folder"]
            daily_and_norms_paths = {"summaries":run_val["daily_summaries_path"],
                                     "norms": run_val["climate_normals_path"]}
            daily_summaries_unit_conversion = run_val["daily_summaries_unit_conversion"]
            climate_normals_unit_conversion = run_val["climate_normals_unit_conversion"]
            solve_options = run_val[SOLVE_OPTIONS_STRING]
            sol_dir = os.path.dirname(historic_solution_save_location)
            num_files_per_solution = run_val["num_files_per_solution"]
            epw_output_dir = run_val["epw_out_folder"]
            scen_dict_name_path = os.path.join(sol_dir,"scen_dict_"+run_name+".dict")
            
            if run_name in skip_runs:
                continue
            elif not run_name in only_generate_files:
                if not os.path.exists(sol_dir):
                    os.mkdir(sol_dir)
                
                
                """
                STEP 1 Create a climate increase in surface temperature
                        set of scenarios,
                        
                        ~ 15 min run time on 60 processors linux server
                """
                if troubleshoot:
                    # For Troubleshooting
                    tup = pkl.load(open("temp_cs_obj.pickle",'rb'))
                    cs_obj, scen_dict = tup
                else:
                    cs_obj = ClimateScenario(use_global=False,
                                              lat=lat,
                                              lon=lon,
                                              end_year=future_years[-1],
                                              output_folder=clim_scen_out_folder,
                                              write_graphics_path=clim_scen_out_folder,
                                              num_cpu=num_cpu,
                                              run_parallel=run_parallel,
                                              model_guide=cmip6_model_guide,
                                              data_folder=cmip6_data_folder,
                                              align_gcm_to_historical=True,
                                              polynomial_order=polynomial_order,
                                              proxy=proxy)
                    
                    
                    scen_dict = cs_obj.calculate_coef(scenarios)
                    write_readable_python_dict(scen_dict_name_path, scen_dict)
                    
                """
                STEP 2 FIT THE HISTORIC DATA. - This takes 6 hours on 60 processors!
                       It is the longest operation and uses optimization.
                 
                """
                if troubleshoot:
                    # For troubleshooting
                    obj = pkl.load(open("obj_pickle.pkl",'rb'))
                else:
                    obj = ExtremeTemperatureWaves(daily_and_norms_paths, 
                                                  weather_files, 
                                                  unit_conversion=daily_summaries_unit_conversion,
                                                  use_local=True, 
                                                  random_seed=random_seed,
                                                  include_plots=plot_results,
                                                  run_parallel=run_parallel, 
                                                  use_global=False, 
                                                  delT_ipcc_min_frac=1.0,
                                                  num_cpu=num_cpu, 
                                                  write_results=True, 
                                                  test_markov=False,
                                                  solve_options=solve_options, 
                                                  proxy=proxy,
                                                  norms_unit_conversion=climate_normals_unit_conversion)
                    
                    obj.write_solution(historic_solution_save_location)
        
                
    
                
                smirnov_df = create_smirnov_table(obj, os.path.join(sol_dir,run_name + "_kolmogorov_smirnov_test_statistic.tex" ))
                    
                """
                STEP 3 Create future solutions (and epw files if requested)
                """
                if troubleshoot:
                    # For Troubleshooting
                    solution_files = ["final_solution_2020_SSP245_5%.txt","final_solution_2080_SSP245_50%.txt"]
                    alter_results = None       
                else:
                    alter_results,solution_files = obj.create_solutions(future_years, scenarios, 
                                                                   ci_intervals, 
                                                                   historic_solution_save_location, 
                                                                   scen_dict)
            else: # gain context from only_generate_files
                alter_results = None
                cs_obj = None
                # adding the solution file shortens this to a matter of minutes
                # instead of hours. The solution file must exist though!
                scen_dict = read_readable_python_dict(scen_dict_name_path)
                obj = ExtremeTemperatureWaves(daily_and_norms_paths, 
                                              weather_files, 
                                              unit_conversion=daily_summaries_unit_conversion,
                                              use_local=True, 
                                              random_seed=random_seed,
                                              include_plots=plot_results,
                                              run_parallel=run_parallel, 
                                              use_global=False, 
                                              delT_ipcc_min_frac=1.0,
                                              num_cpu=num_cpu, 
                                              write_results=True, 
                                              test_markov=False,
                                              solve_options=solve_options, 
                                              proxy=proxy,
                                              norms_unit_conversion=climate_normals_unit_conversion,
                                              solution_file=historic_solution_save_location)
                solution_files = [file for file in os.listdir(sol_dir) if "final_solution_" in file]
                
                smirnov_df = create_smirnov_table(obj, os.path.join(sol_dir,run_name + "_kolmogorov_smirnov_test_statistic.tex" ))
                
            # begin where only generate files is executed
            if generate_epw and len(weather_files) > 0:
                epw_files, results_get = generate_epw_files(obj,solution_files,sol_dir,num_files_per_solution,
                                               epw_output_dir,random_seed, scen_dict,run_parallel,num_cpu)
            else:
                epw_files = []
                results_get = {}
                
            
            results[run_name] = {"Alter objects dict":alter_results,
                       "ExtremeTemperatureWave object":obj,
                       "epw files":epw_files,
                       "solution_files":solution_files,
                       "ClimateScenario object":cs_obj,
                       "scen_dict":scen_dict,
                       "run input values":run_val,
                       "kolomogorov smirnov":smirnov_df,
                       "epw alter results":results_get}
            
        except Exception as excep:
            results[run_name] = {"Run Failed Exception":excep}
            
        
    return results
        
        
        
if __name__ == "__main__":
    example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","examples"))
    run_dict = {"Chicago":{"future_years":[2020,2040,2060,2080],
                "ci_intervals":["5%","50%","95%"],
                "latitude_longitude":(41.78300,360-87.75000),
                "scenarios":["SSP245","SSP370","SSP585"],
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone5A_Chicago","Chicago_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone5A_Chicago","Chicago_norms.csv"),
                "daily_summaries_unit_conversion":(5/9, -(5/9)*32),
                "climate_normals_unit_conversion":(5/9, -(5/9)*32),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone5A_Chicago","results","chicago_midway_historical_solution.txt"),
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone5A_Chicago","USA_IL_Chicago.Midway.Intl.AP.725340_TMY3.epw"),
                                 os.path.join(example_dir,"example_data","ClimateZone5A_Chicago","USA_IL_Chicago.OHare.Intl.AP.725300_TMY3.epw")],
                "random_seed":349082,
                "cmip6_data_folder":os.path.join("..","..","CMIP6_Data_Files"),
                "solve_options":{'historic': {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                              'decay_func_type': {'cs': 'quadratic_times_exponential_decay_with_cutoff', 
                                                                  'hw': "quadratic_times_exponential_decay_with_cutoff"},
                                              'max_iter': 60,
                                              'limit_temperatures': False,
                                              'num_cpu': -1,
                                              'plot_results': True,
                                              'num_step': 2500000,
                                              'test_mode': False,
                                              'min_num_waves': 10,
                                              'weights': np.array([1, 1, 1, 3, 1]),
                                              'out_path': os.path.join(example_dir,"example_data", "ClimateZone5A_Chicago", "results", "chicago.png")},
                                 'future': {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                            'max_iter': 60,
                                            'limit_temperatures': False,
                                            'num_cpu': -1,
                                            'num_step': 2500000,
                                            'plot_results': True,
                                            'decay_func_type': {'cs': 'quadratic_times_exponential_decay_with_cutoff', 
                                                                'hw': "quadratic_times_exponential_decay_with_cutoff"},
                                            'test_mode': False,
                                            'min_num_waves': 10,
                                            'out_path': os.path.join(example_dir,"example_data", "ClimateZone5A_Chicago", "results", "chicago_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone5A_Chicago","clim_scen_results"),
                "num_files_per_solution": 200,
                "epw_out_folder":os.path.join(example_dir,"example_data","ClimateZone5A_Chicago","mews_epw_results")}, 
     "Phoenix":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw")],
                "latitude_longitude":(33.4352,360-112.0101),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","Phoenix_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","Phoenix_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","results","phoenix_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone2B_Phoenix", "results", "phoenix_airport.png"),
                                              'weights':np.array([1, 1, 1, 5, 1]),
                                              'max_iter':60,
                                              'decay_func_type':{'cs':'exponential_cutoff',
                                                                 'hw':'exponential_cutoff'}},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone2B_Phoenix", "results", "phoenix_airport_future.png"),
                                            'weights':np.array([1, 1, 1, 5, 1]),
                                            'max_iter':60,
                                            'decay_func_type':{'cs':'exponential_cutoff',
                                                               'hw':'exponential_cutoff'}}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","mews_epw_results")},
     "Minneapolis":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone6A_Minneapolis","USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3.epw")],
                "latitude_longitude":(44.8848,360-93.2223),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone6A_Minneapolis","Minneapolis_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone6A_Minneapolis","Minneapolis_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone6A_Minneapolis","results","minneapolis_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone6A_Minneapolis", "results", "minneapolis_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone6A_Minneapolis", "results", "minneapolis_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone6A_Minneapolis","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone6A_Minneapolis","mews_epw_results")},
     "Baltimore":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone4A_Baltimore","USA_MD_Baltimore-Washington.Intl.Marshall.AP.724060_TMY3.epw")],
                "latitude_longitude":(39.1774,360-76.6684),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone4A_Baltimore","Baltimore_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone4A_Baltimore","Baltimore_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone4A_Baltimore","results","baltimore_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone4A_Baltimore", "results", "baltimore_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone4A_Baltimore", "results", "baltimore_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone4A_Baltimore","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone4A_Baltimore","mews_epw_results")},
     "Miami":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone1A_Miami","USA_FL_Miami-Opa.Locka.Exec.AP.722024_TMY3.epw")],
                "latitude_longitude":(25.90000,360.0-80.28300),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone1A_Miami","Miami_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone1A_Miami","Miami_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone1A_Miami","results","Miami_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone1A_Miami", "results", "Miami_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone1A_Miami", "results", "Miami_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone1A_Miami","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone1A_Miami","mews_epw_results")},
     "Houston":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone2A_Houston","USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3.epw")],
                "latitude_longitude":(30.00000,360.0-95.36700),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone2A_Houston","Houston_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone2A_Houston","Houston_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone2A_Houston","results","Houston_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone2A_Houston", "results", "Houston_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone2A_Houston", "results", "Houston_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone2A_Houston","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone2A_Houston","mews_epw_results")},
     "Atlanta":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone3A_Atlanta","USA_GA_Atlanta-Hartsfield-Jackson.Intl.AP.722190_TMY3.epw")],
                "latitude_longitude":(33.63300,360.0-84.43300),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone3A_Atlanta","Atlanta_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone3A_Atlanta","Atlanta_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone3A_Atlanta","results","Atlanta_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone3A_Atlanta", "results", "Atlanta_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone3A_Atlanta", "results", "Atlanta_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone3A_Atlanta","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone3A_Atlanta","mews_epw_results")},
     "LasVegas":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone3B_LasVegas","USA_NV_Las.Vegas-McCarran.Intl.AP.723860_TMY3.epw")],
                "latitude_longitude":(36.08300,360.0-115.1500),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone3B_LasVegas","LasVegas_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone3B_LasVegas","LasVegas_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone3B_LasVegas","results","LasVegas_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone3B_LasVegas", "results", "LasVegas_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone3B_LasVegas", "results", "LasVegas_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone3B_LasVegas","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone3B_LasVegas","mews_epw_results")},
     "LosAngeles":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone3B_LosAngeles","USA_CA_Los.Angeles.Intl.AP.722950_TMY3.epw")],
                "latitude_longitude":(33.93300,360.0-118.4000),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone3B_LosAngeles","LosAngeles_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone3B_LosAngeles","LosAngeles_norms.csv"),
                "daily_summaries_unit_conversion":(0.1, 0.0),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone3B_LosAngeles","results","LosAngeles_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone3B_LosAngeles", "results", "LosAngeles_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone3B_LosAngeles", "results", "LosAngeles_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone3B_LosAngeles","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone3B_LosAngeles","mews_epw_results")},
     "SanFrancisco":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone3C_SanFrancisco","USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw")],
                "latitude_longitude":(37.62,360.0-122.40),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone3C_SanFrancisco","SanFrancisco_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone3C_SanFrancisco","SanFrancisco_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone3C_SanFrancisco","results","SanFrancisco_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone3C_SanFrancisco", "results", "SanFrancisco_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone3C_SanFrancisco", "results", "SanFrancisco_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone3C_SanFrancisco","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone3C_SanFrancisco","mews_epw_results")},
     "Albuquerque":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone4B_Albuquerque","USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")],
                "latitude_longitude":(35.04,360.0-106.62),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone4B_Albuquerque","USW00023050.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone4B_Albuquerque","USW00023050_norms.csv"),
                "daily_summaries_unit_conversion":(0.1, 0.0),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone4B_Albuquerque","results","Albuquerque_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone4B_Albuquerque", "results", "Albuquerque_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone4B_Albuquerque", "results", "Albuquerque_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone4B_Albuquerque","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone4B_Albuquerque","mews_epw_results")},
     "Seattle":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone4C_Seattle","USA_WA_Seattle-Tacoma.Intl.AP.727930_TMY3.epw")],
                "latitude_longitude":(47.44300,360.0-122.3060),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone4C_Seattle","Seattle_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone4C_Seattle","Seattle_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone4C_Seattle","results","Seattle_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone4C_Seattle", "results", "Seattle_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone4C_Seattle", "results", "Seattle_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone4C_Seattle","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone4C_Seattle","mews_epw_results")},
     "Denver":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone5B_Denver","USA_CO_Aurora-Buckley.AFB.724695_TMY3.epw")],
                "latitude_longitude":(39.71700,360.0-104.7500),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone5B_Denver","Denver_Cheesman_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone5B_Denver","Denver_AURORA_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone5B_Denver","results","Denver_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone5B_Denver", "results", "Denver_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone5B_Denver", "results", "Denver_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone5B_Denver","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone5B_Denver","mews_epw_results")},
     "Helena":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone6B_Helena","USA_MT_Helena.Rgnl.AP.727720_TMY3.epw")],
                "latitude_longitude":(46.60000,360.0-111.9670),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone6B_Helena","Helena_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone6B_Helena","Helena_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone6B_Helena","results","Helena_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone6B_Helena", "results", "Helena_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone6B_Helena", "results", "Helena_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone6B_Helena","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone6B_Helena","mews_epw_results")},
     "Duluth":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone7_Duluth","USA_MN_Duluth.Intl.AP-Duluth.ANGB.727450_TMY3.epw")],
                "latitude_longitude":(46.83300,360.0-92.21700),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone7_Duluth","Duluth_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone7_Duluth","Duluth_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone7_Duluth","results","Duluth_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone7_Duluth", "results", "Duluth_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone7_Duluth", "results", "Duluth_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone7_Duluth","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone7_Duluth","mews_epw_results")},
     "Fairbanks":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone8_Fairbanks","USA_AK_Fairbanks.Intl.AP.702610_TMY3.epw")],
                "latitude_longitude":(64.81700,360.0-147.8500),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone8_Fairbanks","Fairbanks_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone8_Fairbanks","Fairbanks_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone8_Fairbanks","results","Fairbanks_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone8_Fairbanks", "results", "Fairbanks_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone8_Fairbanks", "results", "Fairbanks_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone8_Fairbanks","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone8_Fairbanks","mews_epw_results")},
     "McAllen":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","ClimateZone2A_McAllen","USA_TX_McAllen.Miller.Intl.AP.722506_TMY3.epw")],
                "latitude_longitude":(26.17600,360.0-98.24000),
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone2A_McAllen","McAllen_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone2A_McAllen","McAllen_norms.csv"),
                "daily_summaries_unit_conversion":(0.1, 0.0),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone2A_McAllen","results","McAllen_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone2A_McAllen", "results", "McAllen_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone2A_McAllen", "results", "McAllen_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","ClimateZone2A_McAllen","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone2A_McAllen","mews_epw_results")},
     "Kodiak":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","Kodiak","USA_AK_Kodiak.AP.703500_TMY3.epw")],
                "latitude_longitude":(57.741461,360.0-152.5032979),
                "daily_summaries_path":os.path.join(example_dir,"example_data","Kodiak","Kodiak.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","Kodiak","Kodiak_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","Kodiak","results","Kodiak_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "Kodiak", "results", "Kodiak_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "Kodiak", "results", "Kodiak_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","Kodiak","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","Kodiak","mews_epw_results")},
     "Worcester":{"template_id":"Chicago",
                "weather_files":[os.path.join(example_dir,"example_data","Worcester","USA_MA_Worcester.Rgnl.AP.725095_TMY3.epw")],
                "latitude_longitude":(42.27,360-71.88),
                "daily_summaries_path":os.path.join(example_dir,"example_data","Worcester","USW00094746.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","Worcester","USW00094746_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","Worcester","results","Worcester_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "Worcester", "results", "Worcester_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "Worcester", "results", "Worcester_airport_future.png")}},
                "clim_scen_out_folder": os.path.join(example_dir,"example_data","Worcester","clim_scen_results"),
                "epw_out_folder": os.path.join(example_dir,"example_data","Worcester","mews_epw_results")}}
    
    
    
    # This runs in several hours and is too long for making into a unit test
    #input_file = os.path.join(os.path.dirname(__file__),"..","examples","mews_input_example.txt")
    results = extreme_temperature(run_dict,run_dict_var={"example_dir":example_dir},only_generate_files=[ 
                                                                'LasVegas',
                                                                'LosAngeles',
                                                                'SanFrancisco',
                                                                'Albuquerque',
                                                                'Seattle', 
                                                                'Denver',
                                                                'Helena', 
                                                                'Duluth',
                                                                'Fairbanks',
                                                                'McAllen',
                                                                'Kodiak',
                                                                'Worcester'],
                                        skip_runs=["Chicago",
                                                                                        'Houston', 
                                                                                        'Atlanta',
                                                                                        "Baltimore",
                                                                                        "Minneapolis",
                                                                                        "Phoenix",
                                                                                        'Miami'])
    pkl.dump([results], open(os.path.join(example_dir,"example_data","STBE_study_results.pkl"),'wb'))
    
