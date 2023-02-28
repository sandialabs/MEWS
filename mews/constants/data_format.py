#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:40:43 2022

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
import numpy as np
import os
from mews.constants.analysis import DEFAULT_SOLVER_NUMBER_STEPS

ABREV_WAVE_NAMES = ["cs","hw"]
WAVE_NAMES = ["cold snap","heat wave"]

WAVE_MAP = {}
for awn,wn in zip(ABREV_WAVE_NAMES,WAVE_NAMES):
    WAVE_MAP[wn] = awn

VALID_SOLVE_INPUTS = ['num_step',    #0
                 'param0',#1
                 'random_seed',#2
                 'hist0',#3
                 'durations0',#4
                 'delT_above_shifted_extreme',#5
                 'historic_time_interval',#6
                 'hours_per_year',#7
                 'problem_bounds',#8
                 'ipcc_shift',#9
                 'decay_func_type',#10
                 'use_cython',#11
                 'num_cpu',#12
                 'plot_results',#13
                 'max_iter',#14
                 'plot_title',#15
                 'out_path',#16
                 'weights',#17
                 'limit_temperatures',#18
                 'min_num_waves',#19
                 'x_solution',#20
                 'test_mode',#21
                 'num_postprocess',#22
                 'extra_output_columns',#23
                 "org_samples"]#24

RESIDUAL_TERM_NAMES = ["Future temperature thresholds","Penalty on high temperatures",
                       "Penalty on inverted waves (i.e. cold heat waves and hot cold snaps)",
                       "Duration residuals multiplier", "Future stretched distribution residuals"]
NUM_RESIDUAL_TERMS = len(RESIDUAL_TERM_NAMES)
WEIGHTS_DEFAULT_ARRAY = np.ones(NUM_RESIDUAL_TERMS)

VALID_SOLVE_OPTIONS = [VALID_SOLVE_INPUTS[8],#'problem_bounds
                       VALID_SOLVE_INPUTS[10],#'decay_func_type
                VALID_SOLVE_INPUTS[11],#'use_cython',
                VALID_SOLVE_INPUTS[12],#'num_cpu',
                VALID_SOLVE_INPUTS[13],#'plot_results',
                VALID_SOLVE_INPUTS[14],#'max_iter',
                VALID_SOLVE_INPUTS[15],#'plot_title',
                VALID_SOLVE_INPUTS[16],#'out_path',
                VALID_SOLVE_INPUTS[17],#'weights',
                VALID_SOLVE_INPUTS[18],#'limit_temperatures',
                VALID_SOLVE_INPUTS[5],#'delT_above_shifted_extreme',
                VALID_SOLVE_INPUTS[0],#'num_step',
                VALID_SOLVE_INPUTS[19],#'min_num_waves',
                VALID_SOLVE_INPUTS[20],#'x_solution',
                VALID_SOLVE_INPUTS[21],#'test_mode',
                VALID_SOLVE_INPUTS[22],#'num_postprocess'
                VALID_SOLVE_INPUTS[23]]#extra_output_columns
VALID_SOLVE_OPTION_TIMES = ['historic','future']

_DEFAULT_S_OPT = {VALID_SOLVE_OPTIONS[0]:None,
                VALID_SOLVE_OPTIONS[1]:{'cs':"quadratic_times_exponential_decay_with_cutoff",
                                        'hw':"quadratic_times_exponential_decay_with_cutoff"},
                VALID_SOLVE_OPTIONS[2]:True,
                VALID_SOLVE_OPTIONS[3]:-1,
                VALID_SOLVE_OPTIONS[4]:False,
                VALID_SOLVE_OPTIONS[5]:20,
                VALID_SOLVE_OPTIONS[6]:'',
                VALID_SOLVE_OPTIONS[7]:"",
                VALID_SOLVE_OPTIONS[8]:WEIGHTS_DEFAULT_ARRAY,
                VALID_SOLVE_OPTIONS[9]:False,
                VALID_SOLVE_OPTIONS[10]:{ABREV_WAVE_NAMES[0]:-20,ABREV_WAVE_NAMES[1]:20},
                VALID_SOLVE_OPTIONS[11]:DEFAULT_SOLVER_NUMBER_STEPS,
                VALID_SOLVE_OPTIONS[12]:25,
                VALID_SOLVE_OPTIONS[13]:None,
                VALID_SOLVE_OPTIONS[14]:False,
                VALID_SOLVE_OPTIONS[15]:3,
                VALID_SOLVE_OPTIONS[16]:{'future year':None,'climate scenario':None, 'threshold confidence interval': None, 'month':None}}
DEFAULT_SOLVE_OPTIONS = {VALID_SOLVE_OPTION_TIMES[0]:_DEFAULT_S_OPT,
                         VALID_SOLVE_OPTION_TIMES[1]:_DEFAULT_S_OPT}

INVALID_VALUE = -99

DECAY_FUNC_TYPES =  {"exponential":0,"linear":1,"exponential_cutoff":2,
               "linear_cutoff":3,
               "quadratic_times_exponential_decay_with_cutoff":4,
               "delayed_exponential_decay_with_cutoff":5}
DECAY_FUNC_NAMES = list(DECAY_FUNC_TYPES.keys())
DECAY_FUNC_NAMES.insert(0,None)

DEFAULT_NONE_DECAY_FUNC = {}
for awn in ABREV_WAVE_NAMES:
    DEFAULT_NONE_DECAY_FUNC[awn] = None

REQUIRED_STRING = "Required Property"
TEMPLATE_ID_STRING = "template_id"
SOLVE_OPTIONS_STRING = "solve_options"
# If None, then a value is not optional 

# SET A LIST WITH FIRST ENTRY "REQUIRED_STRING" if an entry must be included
# in either the template or in the actual entry itself
VALID_RUN_MEWS_ENTRIES = {TEMPLATE_ID_STRING:["",str],
              "future_years":[REQUIRED_STRING,list],
              "ci_intervals":[REQUIRED_STRING,list], 
              "latitude_longitude":[REQUIRED_STRING,(tuple,list)],
            "scenarios":[REQUIRED_STRING,list],     
            "polynomial_order" :[{'historical':7,
                                 'SSP119':4,
                                 'SSP126':4,
                                 'SSP585':4,
                                 'SSP245':4,
                                 'SSP370':4},dict],
            "weather_files" :[[],list],
            "daily_summaries_path" : [REQUIRED_STRING,str],
            "climate_normals_path" : [REQUIRED_STRING,str],
            "daily_summaries_unit_conversion": [REQUIRED_STRING,tuple],
            "climate_normals_unit_conversion": [REQUIRED_STRING,tuple],
            "historic_solution_save_location": [REQUIRED_STRING,str],
            "random_seed" : [REQUIRED_STRING,int],
            "cmip6_model_guide" : [os.path.join("data","Models_Used_alpha.xlsx"),str],
            "cmip6_data_folder" : [REQUIRED_STRING,str],
            SOLVE_OPTIONS_STRING: [REQUIRED_STRING,dict],
            "num_files_per_solution":[REQUIRED_STRING,int],
            "clim_scen_out_folder" : [REQUIRED_STRING,str],
            "epw_out_folder" :["",str]}
VALID_RUN_MEWS_ENTRIES_LIST = [val for key,val in VALID_RUN_MEWS_ENTRIES.items()]