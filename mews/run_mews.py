#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:55:01 2023

This is the main interface to create a set of solution files via an input script

@author: dlvilla
"""
import os
import numpy as np
from numpy import poly1d
from mews.weather.climate import ClimateScenario
from mews.events import ExtremeTemperatureWaves
from mews.utilities.utilities import bin_avg, read_readable_python_dict, create_smirnov_table
from mews.constants.data_format import (VALID_RUN_MEWS_ENTRIES, 
                                        REQUIRED_STRING, 
                                        TEMPLATE_ID_STRING,
                                        SOLVE_OPTIONS_STRING)

from copy import deepcopy
from mews.errors.exceptions import MEWSInputTemplateError

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
                       output_dir,random_seed,scen_dict):
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
    for file in solution_files:
        brstr = file.split(".")[0].split("_")
        
        year = int(brstr[-3])
        scen_name = brstr[-2]
        cii = brstr[-1]
        
        obj.create_scenario(scenario_name=scen_name,
                            year=year,
                            climate_temp_func=scen_dict,
                            num_realization=num_files_per_solution,
                            climate_baseyear=2014,
                            increase_factor_ci=cii,
                            cold_snap_shift=None,
                            solution_file=os.path.join(solution_path,file),
                            output_dir=output_dir,
                            random_seed=random_seed)
        
    epw_file_list = [os.path.join(os.path.abspath(output_dir),file) for file in os.listdir(output_dir) if ".epw" in file]
    
    return epw_file_list


def extreme_temperature(run_dict, run_parallel=True, num_cpu=-1, 
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
          
    only_generate_files : list :Default = None
         Enter any name 
          
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
    if isinstance(run_dict,str):
        run_dict = read_readable_python_dict(run_dict)
        
    run_dict_modified = _check_and_arrange_run_dict_inputs(run_dict)
    
    results = {}
    
    for run_name, run_val in run_dict_modified.items():
        # Keep on going if one run fails.
        try:
            troubleshoot = False
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
            

            
            if generate_epw and len(weather_files) > 0:
                epw_files = generate_epw_files(obj,solution_files,sol_dir,num_files_per_solution,
                                               epw_output_dir,random_seed, scen_dict)
            else:
                epw_files = []
                
            
            results[run_name] = {"Alter objects dict":alter_results,
                       "ExtremeTemperatureWave object":obj,
                       "epw files":epw_files,
                       "solution_files":solution_files,
                       "ClimateScenario object":cs_obj,
                       "scen_dict":scen_dict,
                       "run input values":run_val,
                       "kolomogorov smirnov":smirnov_df}
            
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
                                              'max_iter': 30,
                                              'limit_temperatures': False,
                                              'num_cpu': -1,
                                              'plot_results': True,
                                              'num_step': 2500000,
                                              'test_mode': False,
                                              'min_num_waves': 10,
                                              'weights': np.array([1, 1, 1, 1, 1]),
                                              'out_path': os.path.join(example_dir,"example_data", "ClimateZone5A_Chicago", "results", "chicago.png")},
                                 'future': {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                            'max_iter': 30,
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
                "daily_summaries_path":os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","Chicago_daily.csv"),
                "climate_normals_path":os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","Chicago_norms.csv"),
                "historic_solution_save_location":os.path.join(example_dir,"example_data","ClimateZone2B_Phoenix","results","phoenix_airport_historical_solution.txt"),
                "solve_options":{'historic': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone2B_Phoenix", "results", "phoenix_airport.png")},
                                 'future': {'out_path': os.path.join(example_dir,"example_data", "ClimateZone2B_Phoenix", "results", "phoenix_airport_future.png")}},
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
                "epw_out_folder": os.path.join(example_dir,"example_data","ClimateZone4A_Baltimore","mews_epw_results")}}
    
    
    
    # This runs in several hours and is too long for making into a unit test
    #input_file = os.path.join(os.path.dirname(__file__),"..","examples","mews_input_example.txt")
    extreme_temperature(run_dict)
    
    
    invalid_input = True
    while invalid_input:
        yesno = input("Are you sure that you want to start this study?"+
                      " It takes several days to run on 60 processors. Enter (y/n)")
        if yesno == "y":
            
            extreme_temperature(run_dict)
            invalid_input = False
        elif yesno == "n":
            invalid_input = False
        else:
            print("You must enter 'y' or 'n'\n\n.")
    
    
        





