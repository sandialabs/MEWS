#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:40:43 2022

@author: dlvilla
"""
import numpy as np
from mews.constants.analysis import DEFAULT_SOLVER_NUMBER_STEPS

ABREV_WAVE_NAMES = ["cs","hw"]
WAVE_NAMES = ["cold snap","heat wave"]

WAVE_MAP = {}
for awn,wn in zip(ABREV_WAVE_NAMES,WAVE_NAMES):
    WAVE_MAP[wn] = awn

VALID_SOLVE_OPTIONS = ['problem_bounds',
                       'decay_func_type',
                'use_cython',
                'num_cpu',
                'plot_results',
                'max_iter',
                'plot_title',
                'fig_path',
                'weights',
                'limit_temperatures',
                'delT_above_shifted_extreme',
                'num_step',
                'min_num_waves',
                'x_solution',
                'test_mode']
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
                VALID_SOLVE_OPTIONS[8]:np.array([1.0,1.0,1.0,1.0]),
                VALID_SOLVE_OPTIONS[9]:False,
                VALID_SOLVE_OPTIONS[10]:{ABREV_WAVE_NAMES[0]:-20,ABREV_WAVE_NAMES[1]:20},
                VALID_SOLVE_OPTIONS[11]:DEFAULT_SOLVER_NUMBER_STEPS,
                VALID_SOLVE_OPTIONS[12]:25,
                VALID_SOLVE_OPTIONS[13]:None,
                VALID_SOLVE_OPTIONS[14]:False}
DEFAULT_SOLVE_OPTIONS = {VALID_SOLVE_OPTION_TIMES[0]:_DEFAULT_S_OPT,
                         VALID_SOLVE_OPTION_TIMES[1]:_DEFAULT_S_OPT}

INVALID_VALUE = -99


DEFAULT_NONE_DECAY_FUNC = {}
for awn in ABREV_WAVE_NAMES:
    DEFAULT_NONE_DECAY_FUNC[awn] = None