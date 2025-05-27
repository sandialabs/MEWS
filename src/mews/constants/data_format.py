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
from mews.constants.physical import C_2_K_OFFSET

SMALL_VALUE = np.finfo(np.float64).tiny

EPW_DATA_DICT_URL = ("https://bigladdersoftware.com/epx/docs/8-3/"
                     +"auxiliary-programs/energyplus-weather-file-"
                     +"epw-data-dictionary.html")

ABREV_WAVE_NAMES = ["cs", "hw"]
WAVE_NAMES = ["cold snap", "heat wave"]

WAVE_MAP = {}
for awn, wn in zip(ABREV_WAVE_NAMES, WAVE_NAMES):
    WAVE_MAP[wn] = awn

VALID_SOLVE_INPUTS = [
    "num_step",  # 0
    "param0",  # 1
    "random_seed",  # 2
    "hist0",  # 3
    "durations0",  # 4
    "delT_above_shifted_extreme",  # 5
    "historic_time_interval",  # 6
    "hours_per_year",  # 7
    "problem_bounds",  # 8
    "ipcc_shift",  # 9
    "decay_func_type",  # 10
    "use_cython",  # 11
    "num_cpu",  # 12
    "plot_results",  # 13
    "max_iter",  # 14
    "plot_title",  # 15
    "out_path",  # 16
    "weights",  # 17
    "limit_temperatures",  # 18
    "min_num_waves",  # 19
    "x_solution",  # 20
    "test_mode",  # 21
    "num_postprocess",  # 22
    "extra_output_columns",  # 23
    "org_samples",
]  # 24

RESIDUAL_TERM_NAMES = [
    "Future temperature thresholds",
    "Penalty on high temperatures",
    "Penalty on inverted waves (i.e. cold heat waves and hot cold snaps)",
    "Duration residuals multiplier",
    "Future stretched distribution residuals",
]
NUM_RESIDUAL_TERMS = len(RESIDUAL_TERM_NAMES)
WEIGHTS_DEFAULT_ARRAY = np.ones(NUM_RESIDUAL_TERMS)

VALID_SOLVE_OPTIONS = [
    VALID_SOLVE_INPUTS[8],  #'problem_bounds
    VALID_SOLVE_INPUTS[10],  #'decay_func_type
    VALID_SOLVE_INPUTS[11],  #'use_cython',
    VALID_SOLVE_INPUTS[12],  #'num_cpu',
    VALID_SOLVE_INPUTS[13],  #'plot_results',
    VALID_SOLVE_INPUTS[14],  #'max_iter',
    VALID_SOLVE_INPUTS[15],  #'plot_title',
    VALID_SOLVE_INPUTS[16],  #'out_path',
    VALID_SOLVE_INPUTS[17],  #'weights',
    VALID_SOLVE_INPUTS[18],  #'limit_temperatures',
    VALID_SOLVE_INPUTS[5],  #'delT_above_shifted_extreme',
    VALID_SOLVE_INPUTS[0],  #'num_step',
    VALID_SOLVE_INPUTS[19],  #'min_num_waves',
    VALID_SOLVE_INPUTS[20],  #'x_solution',
    VALID_SOLVE_INPUTS[21],  #'test_mode',
    VALID_SOLVE_INPUTS[22],  #'num_postprocess'
    VALID_SOLVE_INPUTS[23],
]  # extra_output_columns
VALID_SOLVE_OPTION_TIMES = ["historic", "future"]

_DEFAULT_S_OPT = {
    VALID_SOLVE_OPTIONS[0]: None,
    VALID_SOLVE_OPTIONS[1]: {
        "cs": "quadratic_times_exponential_decay_with_cutoff",
        "hw": "quadratic_times_exponential_decay_with_cutoff",
    },
    VALID_SOLVE_OPTIONS[2]: True,
    VALID_SOLVE_OPTIONS[3]: -1,
    VALID_SOLVE_OPTIONS[4]: False,
    VALID_SOLVE_OPTIONS[5]: 20,
    VALID_SOLVE_OPTIONS[6]: "",
    VALID_SOLVE_OPTIONS[7]: "",
    VALID_SOLVE_OPTIONS[8]: WEIGHTS_DEFAULT_ARRAY,
    VALID_SOLVE_OPTIONS[9]: False,
    VALID_SOLVE_OPTIONS[10]: {ABREV_WAVE_NAMES[0]: -20, ABREV_WAVE_NAMES[1]: 20},
    VALID_SOLVE_OPTIONS[11]: DEFAULT_SOLVER_NUMBER_STEPS,
    VALID_SOLVE_OPTIONS[12]: 25,
    VALID_SOLVE_OPTIONS[13]: None,
    VALID_SOLVE_OPTIONS[14]: False,
    VALID_SOLVE_OPTIONS[15]: 3,
    VALID_SOLVE_OPTIONS[16]: {
        "future year": None,
        "climate scenario": None,
        "threshold confidence interval": None,
        "month": None,
    },
}
DEFAULT_SOLVE_OPTIONS = {
    VALID_SOLVE_OPTION_TIMES[0]: _DEFAULT_S_OPT,
    VALID_SOLVE_OPTION_TIMES[1]: _DEFAULT_S_OPT,
}

INVALID_VALUE = -99

DECAY_FUNC_TYPES = {
    "exponential": 0,
    "linear": 1,
    "exponential_cutoff": 2,
    "linear_cutoff": 3,
    "quadratic_times_exponential_decay_with_cutoff": 4,
    "delayed_exponential_decay_with_cutoff": 5,
}
DECAY_FUNC_NAMES = list(DECAY_FUNC_TYPES.keys())
DECAY_FUNC_NAMES.insert(0, None)

DEFAULT_NONE_DECAY_FUNC = {}
for awn in ABREV_WAVE_NAMES:
    DEFAULT_NONE_DECAY_FUNC[awn] = None

REQUIRED_STRING = "Required Property"
TEMPLATE_ID_STRING = "template_id"
SOLVE_OPTIONS_STRING = "solve_options"
# If None, then a value is not optional

# SET A LIST WITH FIRST ENTRY "REQUIRED_STRING" if an entry must be included
# in either the template or in the actual entry itself
VALID_RUN_MEWS_ENTRIES = {
    TEMPLATE_ID_STRING: ["", str],
    "future_years": [REQUIRED_STRING, list],
    "ci_intervals": [REQUIRED_STRING, list],
    "latitude_longitude": [REQUIRED_STRING, (tuple, list)],
    "scenarios": [REQUIRED_STRING, list],
    "polynomial_order": [
        {
            "historical": 7,
            "SSP119": 4,
            "SSP126": 4,
            "SSP585": 4,
            "SSP245": 4,
            "SSP370": 4,
        },
        dict,
    ],
    "weather_files": [[], list],
    "daily_summaries_path": [REQUIRED_STRING, str],
    "climate_normals_path": [REQUIRED_STRING, str],
    "daily_summaries_unit_conversion": [REQUIRED_STRING, tuple],
    "climate_normals_unit_conversion": [REQUIRED_STRING, tuple],
    "historic_solution_save_location": [REQUIRED_STRING, str],
    "random_seed": [REQUIRED_STRING, int],
    "cmip6_model_guide": [os.path.join("data", "Models_Used_alpha.xlsx"), str],
    "cmip6_data_folder": [REQUIRED_STRING, str],
    SOLVE_OPTIONS_STRING: [REQUIRED_STRING, dict],
    "num_files_per_solution": [REQUIRED_STRING, int],
    "clim_scen_out_folder": [REQUIRED_STRING, str],
    "epw_out_folder": ["", str],
}
VALID_RUN_MEWS_ENTRIES_LIST = [val for key, val in VALID_RUN_MEWS_ENTRIES.items()]

# data dictionary obtained from:
# https://bigladdersoftware.com/epx/docs/8-3/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html
# Actual data does not have a descriptor
# dlvilla added other fields to help with validation.
EPW_PRESSURE_COLUMN_NAME = "Atmospheric Station Pressure"
EPW_DB_TEMP_COLUMN_NAME = "Dry Bulb Temperature"
EPW_DP_TEMP_COLUMN_NAME = "Dew Point Temperature"
EPW_RH_COLUMN_NAME = "Relative Humidity"

EPW_PSYCH_NAMES = [
    EPW_PRESSURE_COLUMN_NAME,
    EPW_DB_TEMP_COLUMN_NAME,
    EPW_DP_TEMP_COLUMN_NAME,
    EPW_RH_COLUMN_NAME,
]
#The Location header record duplicates the information required for the 
# Location Object. When only a Run Period object is used (i.e. a weather file), 
# then the Location Object Is not needed. When a Run Period and Design Day objects
# are entered, then the Location on the weather file (as described previously) is
#  used and overrides any Location Object entry.
# THIS IS INCOMPLETE! TODO - complete this and provide more thorough checks in the EPW 
# package read/write functions
EPW_HEADER_DICTIONARY = {"LOCATION":[{"type":str,"description":"City","notes":""},
                                     {"type":str,"description":"State Province Region",
                                      "notes":""},
                                     {"type":str,"description":"Country","notes":""},
                                     {"type":str,"description":"Source","notes":""},
                                     {"type":str,"description":"WMO","notes":("World"
                                                +" Meteorological Organization's Station"
                                                +" Number. WMO is usually a 6 digit field."
                                                +" Used as a string in EnergyPlus")},
                                    {"type":float,"description":"Latitude","units":"degrees",
                                     "minimum":-90.0,"maximum":90.0,"default":0.0,
                                     "notes":("+ is North, - is South, degrees minutes"
                                              +" represented in decimal (i.e. 30 "
                                              +"minutes is .5)")},
                                    {"type":float,"description":"Longitude","units":"degrees",
                                     "minimum":-180.0,"maximum":180.0,"default":0.0,
                                     "notes":("- is West, + is East, degrees minutes"
                                              +" represented in decimal (i.e. 30 "
                                              +"minutes is .5)")},
                                    {"type":float,"description":"TimeZone","units":"hr",
                                     "minimum":-12.0,"maximum":12.0,"default":0.0,
                                     "notes":("Time relative to Greenwich Mean Time (GMT)")},
                                    {"type":float,"description":"Elevation","units":"m",
                                     "minimum":-1000.0,"maximum":1000.0,"default":0.0,
                                     "notes":("")}]}


EPW_DATA_DICTIONARY = {
    "Year": {
        "column": 1,
        "type": int,
        "maximum": 2200,
        "minimum": 1700,
        "psychrometric": False,
        "note": "",
        "units": None,
    },
    "Month": {
        "column": 2,
        "type": int,
        "maximum": 12,
        "minimum": 1,
        "psychrometric": False,
        "note": "",
        "units": None,
    },
    "Day": {
        "column": 3,
        "type": int,
        "maximum": 31,
        "minimum": 1,
        "psychrometric": False,
        "note": "",
        "units": None,
    },
    "Hour": {
        "column": 4,
        "type": int,
        "maximum": 23,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "units": None,
    },
    "Minute": {
        "column": 5,
        "type": int,
        "maximum": 59,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "units": None,
    },
    "Data Source and Uncertainty Flags": {
        "column": 6,
        "type": str,
        "note": (
            "Initial day of weather"
            + " file is checked by EnergyPlus for validity (as shown below)."
            + " Each field is checked for 'missing' as shown below. Reasonable values, calculated"
            + " values or the last 'good' value is substituted."
        ),
        "units": None,
        "psychrometric": False,
    },
    EPW_DB_TEMP_COLUMN_NAME: {
        "column": 7,
        "type": float,
        "maximum": 70,
        "minimum": -70,
        "psychrometric": True,
        "note": "",
        "missing": 99.9,
        "units": "C",
        "cool_prop": {
            "symbol": "T",
            "convert_units": lambda x: x + C_2_K_OFFSET,
            "unconvert_units": lambda x: x - C_2_K_OFFSET,
        },
    },
    EPW_DP_TEMP_COLUMN_NAME: {
        "column": 8,
        "type": float,
        "maximum": 70,
        "minimum": -70,
        "psychrometric": True,
        "note": "",
        "missing": 99.9,
        "units": "C",
        "cool_prop": {
            "symbol": "D",
            "convert_units": lambda x: x + C_2_K_OFFSET,
            "unconvert_units": lambda x: x - C_2_K_OFFSET,
        },
    },
    EPW_RH_COLUMN_NAME: {
        "column": 9,
        "type": float,
        "maximum": 110,
        "minimum": 0,
        "psychrometric": True,
        "note": "",
        "missing": 999,
        "units": "%",
        "cool_prop": {
            "symbol": "R",
            "convert_units": lambda x: x / 100,
            "unconvert_units": lambda x: 100 * x,
        },
    },
    EPW_PRESSURE_COLUMN_NAME: {
        "column": 10,
        "type": float,
        "maximum": 120000,
        "minimum": 31000,
        "psychrometric": True,
        "note": "",
        "missing": 999999,
        "units": "Pa",
        "cool_prop": {
            "symbol": "P",
            "convert_units": lambda x: x,
            "unconvert_units": lambda x: x,
        },
    },
    "Extraterrestrial Horizontal Radiation": {
        "column": 11,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 9999,
        "units": "Wh/m2",
    },
    "Extraterrestrial Direct Normal Radiation": {
        "column": 12,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 9999,
        "units": "Wh/m2",
    },
    "Horizontal Infrared Radiation Intensity": {
        "column": 13,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 9999,
        "units": "Wh/m2",
    },
    "Global Horizontal Radiation": {
        "column": 14,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 9999,
        "units": "Wh/m2",
    },
    "Direct Normal Radiation": {
        "column": 15,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 9999,
        "units": "Wh/m2",
    },
    "Diffuse Horizontal Radiation": {
        "column": 16,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 9999,
        "units": "Wh/m2",
    },
    "Global Horizontal Illuminance": {
        "column": 17,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "will be missing if >= 999900",
        "missing": 999999,
        "units": "lux",
    },
    "Direct Normal Illuminance": {
        "column": 18,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "will be missing if >= 999900",
        "missing": 999999,
        "units": "lux",
    },
    "Diffuse Horizontal Illuminance": {
        "column": 19,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "will be missing if >= 999900",
        "missing": 999999,
        "units": "lux",
    },
    "Zenith Luminance": {
        "column": 20,
        "type": float,
        "maximum": None,
        "minimum": 0,
        "psychrometric": False,
        "note": "will be missing if >= 9999",
        "missing": 9999,
        "units": "Cd/m2",
    },
    "Wind Direction": {
        "column": 21,
        "type": float,
        "maximum": 360,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 999,
        "units": "degrees",
    },
    "Wind Speed": {
        "column": 22,
        "type": float,
        "maximum": 40,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 999,
        "units": "m/s",
    },
    "Total Sky Cover": {
        "column": 23,
        "type": float,
        "maximum": 10,
        "minimum": 0,
        "psychrometric": False,
        "note": "",
        "missing": 99,
        "units": None,
    },
    "Opaque Sky Cover": {
        "column": 24,
        "type": float,
        "maximum": 10,
        "minimum": 0,
        "psychrometric": False,
        "note": "used if Horizontal IR Intensity missing",
        "missing": 99,
        "units": None,
    },
    "Visibility": {
        "column": 25,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 9999,
        "units": "km",
    },
    "Ceiling Height": {
        "column": 26,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 99999,
        "units": "m",
    },
    "Present Weather Observation": {
        "column": 27,
        "type": int,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": None,
        "units": None,
    },
    "Present Weather Codes": {
        "column": 28,
        "type": int,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": None,
        "units": None,
    },
    "Precipitable Water": {
        "column": 29,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 999,
        "units": "mm",
    },
    "Aerosol Optical Depth": {
        "column": 30,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 0.999,
        "units": "thousandths",
    },
    "Snow Depth": {
        "column": 31,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 999,
        "units": "cm",
    },
    "Days Since Last Snowfall": {
        "column": 32,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 99,
        "units": None,
    },
    "Albedo": {
        "column": 33,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 999,
        "units": None,
    },
    "Liquid Precipitation Depth": {
        "column": 34,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 999,
        "units": "mm",
    },
    "Liquid Precipitation Quantity": {
        "column": 35,
        "type": float,
        "maximum": None,
        "minimum": None,
        "psychrometric": False,
        "note": "",
        "missing": 99,
        "units": "hr",
    },
}

EPW_PSYCHROMETRIC_DICTIONARY = {
    key: val for key, val in EPW_DATA_DICTIONARY.items() if val["psychrometric"]
}
