#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:12:15 2022

@author: dlvilla
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:07:57 2021

@author: dlvilla
"""

from mews.stats.solve import SolveDistributionShift, duration_residuals_func, calculated_hist0_stretched
from mews.stats.extreme import DiscreteMarkov
from mews.stats.solve import ObjectiveFunction
from numpy.random import default_rng
from matplotlib import pyplot as plt
from mews.utilities.utilities import find_extreme_intervals
from mews.constants.data_format import DEFAULT_NONE_DECAY_FUNC, ABREV_WAVE_NAMES
from shutil import rmtree

import numpy as np
import unittest
import os
from warnings import warn


class Test_SolveDistributionShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = False
        cls.write_results = False
        cls._run_all_tests = True
        cls.rng = default_rng(23985)
        plt.close('all')
        if os.path.exists("testing_output"):
            rmtree(("testing_output"),ignore_errors=False,onerror=None)
        os.mkdir("testing_output")
        
    
    @classmethod
    def tearDownClass(cls):
        
        if os.path.exists("testing_output"):
            rmtree(("testing_output"),ignore_errors=False,onerror=None)
    
    
    def test_calculated_hist0_stretched(self):
        run_test = self._run_all_tests
        if run_test:
        
            hist0 = {'hw':(np.array([1,2,3,4,5,4,3,2,1]), np.array([5,6,7,8,9,10,11,12,13,14])),
                     'cs':(np.array([1,2,3,4,5,4,3,2,1,1]), np.array([-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4]))}
            num_step = 2e6
            ipcc_shift = {'hw':{'frequency':{'10 year':5.0,'50 year':15.0},
                            'temperature':{'10 year':3.0,'50 year':4.2}},
                          'cs':{'frequency':{'10 year':0.9,'50 year':0.8},
                                          'temperature':{'10 year':0.2,'50 year':0.3}}}
            
            historic_time_interval = 8760 # this is just a reasonble number given that only 25 waves have occured
                                          # in the histogram.
            hours_per_year = 8760/12
            
            (hist0_stretched, asol_val, objective_function, param_val, prob_vals) = calculated_hist0_stretched(
                                                        hist0, ipcc_shift, 
                                                        historic_time_interval, 
                                                        hours_per_year,
                                                        self.plot_results)
            epsilon = 0.05
            for wt in ABREV_WAVE_NAMES:
                for dur in ["10 year", "50 year"]:
                    in_range = ((prob_vals["hw"]["actual"]["10 year"] > prob_vals["hw"]["target"]["10 year"] - epsilon) and
                                (prob_vals["hw"]["actual"]["10 year"] < prob_vals["hw"]["target"]["10 year"] + epsilon))
                    self.assertTrue(in_range) 

        else:
            warn("The test 'test_calculated_hist0_stretched' was not run because run_test=False")

    def test_decay_func_types(self):
        
        """

        
        """

        run_test = self._run_all_tests
        if run_test:

            default_problem_bounds = {'cs':{'delT_mu': (0.0, 20.0),
                                             'delT_sig multipliers': (-0.1,4),
                                             'P_event': (0.00025, 0.0125),
                                             'P_sustain': (0.975, 0.99999),
                                             'multipliers to max probability time': (0,2),
                                             'cutoff time multipliers':(1,4),
                                             'max peak prob for quadratic model': (0.976, 1.0)},
                                       'hw':{'delT_mu': (0.0, 20.0),
                                             'delT_sig multipliers': (-0.1,10),
                                             'P_event': (0.00025,0.0125),
                                             'P_sustain': (0.985,0.99999),
                                             'slope or exponent multipliers' : (0,1),
                                             'cutoff time multipliers':(2,6),
                                             'delay_time_multipliers':(0.1,3)}}
            
            decay_func_type = {'hw':""}
            
            weights = np.array([100.0,1.0,1.0,100.0,1.0])
            
            hours_per_year = 31 * 24
            
            resid_vals = []
            delT_above_50_year = {'hw':20.0,'cs':-20.0} # degrees celcius
            num_step = 8760*50  # provides 11 50 year periods
            rng = default_rng(29023430)
            decay_func_type = [{'cs':'quadratic_times_exponential_decay_with_cutoff',
                                'hw':'delayed_exponential_decay_with_cutoff'}]#[None,"linear","exponential","exponential_cutoff","quadratic_times_exponential_decay_with_cutoff"]  # 8 parameters

            for decay_func in decay_func_type:
                param = {}
                param['hw'] = {}
                param['cs'] = {}
                # taken from extreme._add_extreme
                param['hw']['normalizing duration'] = 192
                param['hw']['extreme_temp_normal_param'] = {}
                param['hw']['extreme_temp_normal_param']['mu'] = 0.0
                param['hw']['extreme_temp_normal_param']['sig'] = 0.3
                param['hw']['max extreme temp per duration'] = 0.55
                param['hw']['hist max extreme temp per duration'] = 0.55
                param['hw']['min extreme temp per duration'] = 0.1
                param['hw']['hist min extreme temp per duration'] = 0.1
                param['hw']['normalizing extreme temp'] = 18
                param['hw']['normalized extreme temp duration fit slope'] = 1.0
                param['hw']['normalized extreme temp duration fit intercept'] = 0.2
                param['hw']['hourly prob stay in heat wave'] = 0.98 
                param['hw']['hourly prob of heat wave'] = 0.002
                
                param['cs']['normalizing duration'] = 192
                param['cs']['extreme_temp_normal_param'] = {}
                param['cs']['extreme_temp_normal_param']['mu'] = 0.0
                param['cs']['extreme_temp_normal_param']['sig'] = 0.3
                param['cs']['max extreme temp per duration'] = 0.55
                param['cs']['hist max extreme temp per duration'] = 0.55
                param['cs']['min extreme temp per duration'] = 0.1
                param['cs']['hist min extreme temp per duration'] = 0.1
                param['cs']['normalizing extreme temp'] = -18
                param['cs']['normalized extreme temp duration fit slope'] = 1.0
                param['cs']['normalized extreme temp duration fit intercept'] = 0.2
                param['cs']['hourly prob stay in heat wave'] = 0.98 
                param['cs']['hourly prob of heat wave'] = 0.002
        
                # None indicates to just fit the historical pdf
                ipcc_shift = {'hw':{'frequency':{'10 year':8.5,'50 year':39.5},
                                'temperature':{'10 year':4.5,'50 year':6.12}},
                              'cs':None}
                
                hist0 = {}
                durations0 = {}
                hist0['hw'] = (np.array([16, 22, 25, 15, 16, 13,  9,  5,  5,  2, 1,  0, 1,  2,  1, 0,0,  1, 0,0,1, 0,1]),
                                  np.array(np.arange(0.0,12,0.5)))
                hist0['cs'] = (np.array([1,0,1,0,0,1,2,0,1,2,1,0,3,2,5,7,11,14,17,22,32,32,21]),
                                  np.array(np.arange(-12,-0.0,0.5)))
                
                durations0['hw'] = (np.array([50,35,15,13,10,9,3,1]),
                                  np.array(np.arange(12,228,24)))
                durations0['cs'] = (np.array([70,40,25,15,12,6,5,2]),
                                  np.array(np.arange(24,228,24)))
                
                # 70 years of records assumed.
                historic_time_interval = 70 * 24 * 365
                
                random_seed = 5618578
                
                # # negative one indicates use all processors available.
                num_cpu = -1
                use_cython = True
                # use this to analyze a specific case found by the solution for troubleshooting.
                # 
                #
                # x0 = np.array([5.83891836e-02, 1.04536792e+00, 5.59722081e+00, 7.98558457e-01,
                #        6.52315993e-03, 1.06182262e-02, 9.84923406e-01, 9.86482344e-01,
                #        7.71808163e+01, 4.82722068e+02, 9.93025112e-01, 1.30386822e+02,
                #        6.52913909e+02, 2.04991220e+00])
                # # run this once so that any obvious bugs will come out 
                # # before it goes into parallel mode in differential_evolution.
                # obj_func = ObjectiveFunction(['cs','hw'],random_seed)
                # resid = obj_func.markov_gaussian_model_for_peak_temperature(x0,
                #                                             num_step,
                #                                             param,
                #                                             hist0,
                #                                             durations0,
                #                                             historic_time_interval,
                #                                             hours_per_year,
                #                                             ipcc_shift,
                #                                             decay_func_type=decay_func,
                #                                             use_cython=use_cython,    
                #                                             output_hist=False,
                #                                             delT_above_shifted_extreme=delT_above_50_year,
                #                                             weights=weights,
                #                                             min_num_waves=25,
                #                                             hist0_stretched=DEFAULT_NONE_DECAY_FUNC)
                
                if decay_func is None:
                    decay_func_str = "no decay"
                else:
                    decay_func_str = decay_func['hw']
                
                # SHIFT PER IPCC
                # low number of iterations because this is unit testing.
                obj = SolveDistributionShift(num_step,
                                       param,
                                       random_seed,
                                       hist0, 
                                       durations0,
                                       delT_above_50_year,
                                       historic_time_interval,
                                       hours_per_year,
                                       default_problem_bounds,
                                       ipcc_shift,
                                       decay_func_type=decay_func,
                                       use_cython=use_cython,
                                       plot_results=self.plot_results,
                                       max_iter=1,
                                       plot_title="Shift to future "+decay_func_str,
                                       num_cpu=num_cpu,
                                       weights=weights,
                                       limit_temperatures=False,
                                       min_num_waves=25)

                # Just fit the histograms - no ipcc_shift factors
                ipcc_shift = {'hw':None, 'cs':None}
                obj2 = SolveDistributionShift(num_step,  
                                       param,
                                       random_seed,
                                       hist0, 
                                       durations0,
                                       delT_above_50_year,
                                       historic_time_interval,
                                       hours_per_year,
                                       default_problem_bounds,
                                       ipcc_shift,
                                       decay_func_type=decay_func,
                                       use_cython=use_cython,
                                       plot_results=self.plot_results,
                                       max_iter=1,
                                       plot_title="Fit histogram "+decay_func_str,
                                       num_cpu=num_cpu,
                                       weights=weights,
                                       limit_temperatures=False,
                                       min_num_waves=25)
                

                resid_vals.append([val for key,val in obj.residuals.items()])
                resid_vals.append([val for key,val in obj2.residuals.items()])

        else:
            warn("\n\nUnit test 'test_decay_func_types' was not run. It has been manually turned off!\n\n")

    def test_single_known_case(self):
        run_test = self._run_all_tests
        if run_test:
            random_seed = 7293821
            
            
            opt_val = {'problem_bounds': None, 
                       'decay_func_type': {'cs': 'exponential_cutoff', 'hw': 'exponential_cutoff'}, 
                       'use_cython': True, 
                       'num_cpu': -1, 
                       'plot_results': self.plot_results, 
                       'max_iter': 25, 
                       'plot_title': '', 'out_path': '', 
                       'weights': np.array([1., 1., 1., 1.,1.]), 
                       'limit_temperatures': False, 
                       'delT_above_shifted_extreme': {'cs': -10, 'hw': 10}, 
                       'num_step': 2000000, 
                       'min_num_waves': 10, 
                       'x_solution': None, 
                       'test_mode': False}
            
            inputs = {'param0': {'cs': {'help': 'These statistics are already mapped from -1 ... 1 and'+
                                        '_inverse_transform_fit is needed to return to actual degC and'+
                                        ' degC*hr values. If input of actual values is desired use trans'+
                                        'form_fit(X,max,min)', 'energy_normal_param': {'mu': 0.0038899912491902897,
                                                                                       'sig': 0.4154535225139539}, 
                                        'extreme_temp_normal_param': {'mu': 0.0075494672194406755, 'sig': 0.3399804140386147}, 
                                        'max extreme temp per duration': 1.4307574699210692, 'min extreme temp per duration': 
                                            0.5585639494466237,'hist max extreme temp per duration': 1.4307574699210692, 'hist min extreme temp per duration': 
                                                0.5585639494466237, 'max energy per duration': 1.3562919483136582, 
                                            'min energy per duration': 0.3990299129968476, 'energy linear slope': 0.9768230669124248, 'normalized extreme temp duration fit slope': 0.6946813967764691, 'normalized extreme temp duration fit intercept': 0.46494070822925054, 'normalizing energy': -3775.333333333333, 'normalizing extreme temp': -25.3125, 'normalizing duration': 384, 'historic time interval': 650040.0, 'hourly prob stay in heat wave': 0.9863164139915251, 'hourly prob of heat wave': 0.005069067413290484, 'historical durations (durations0)': (np.array([53, 59, 37, 19,  9,  3,  2,  3,  1,  0,  0,  0,  0,  0,  0,  1]), np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204., 228., 252.,
                   276., 300., 324., 348., 372., 396.])), 'historical temperatures (hist0)': (np.array([ 1,  0,  0,  2,  6,  2,  7, 13, 14, 11, 25, 22, 29, 20, 25,  6,  2,
                    1,  1]), np.array([-25.78947368, -24.78531856, -23.78116343, -22.77700831,
                   -21.77285319, -20.76869806, -19.76454294, -18.76038781,
                   -17.75623269, -16.75207756, -15.74792244, -14.74376731,
                   -13.73961219, -12.73545706, -11.73130194, -10.72714681,
                    -9.72299169,  -8.71883657,  -7.71468144,  -6.71052632]))}, 'hw': {'help': 'These statistics are already mapped from -1 ... 1 and _inverse_transform_fit is needed to return to actual degC and degC*hr values. If input of actual values is desired use transform_fit(X,max,min)', 'energy_normal_param': {'mu': 0.07612295120388764, 'sig': 0.26004106669177984}, 'extreme_temp_normal_param': {'mu': -0.33020629173650035, 'sig': 0.2687845192334446}, 
                                                                                      'max extreme temp per duration': 1.8604981436271297, 
                                                                                      'min extreme temp per duration': 0.5661091912436009, 
                                                                                      'hist max extreme temp per duration': 1.8604981436271297, 
                                                                                      'hist min extreme temp per duration': 0.5661091912436009, 'max energy per duration': 1.8468741653468745, 'min energy per duration': -0.284239504158205, 'energy linear slope': 0.7404780352640733, 'normalized extreme temp duration fit slope': 0.4823740200889345, 'normalized extreme temp duration fit intercept': 0.45709478834728723, 'normalizing energy': 1648.666666666667, 'normalizing extreme temp': 23.157407407407405, 'normalizing duration': 144, 'historic time interval': 650040.0, 'hourly prob stay in heat wave': 0.9777962801504562, 'hourly prob of heat wave': 0.004445599228768125, 'historical durations (durations0)': (np.array([92, 49, 14,  6,  2,  1]), np.array([ 12.,  36.,  60.,  84., 108., 132., 156.])), 'historical temperatures (hist0)': (np.array([ 1,  2,  2, 11, 35, 19, 18, 19, 13, 17, 10,  8,  6,  0,  2,  0,  1]), np.array([ 6.57244009,  7.57590029,  8.5793605 ,  9.58282071, 10.58628092,
                   11.58974113, 12.59320133, 13.59666154, 14.60012175, 15.60358196,
                   16.60704216, 17.61050237, 18.61396258, 19.61742279, 20.62088299,
                   21.6243432 , 22.62780341, 23.63126362]))}}, 'hist0': {'cs': (np.array([ 1,  0,  0,  2,  6,  2,  7, 13, 14, 11, 25, 22, 29, 20, 25,  6,  2,
                    1,  1]), np.array([-25.78947368, -24.78531856, -23.78116343, -22.77700831,
                   -21.77285319, -20.76869806, -19.76454294, -18.76038781,
                   -17.75623269, -16.75207756, -15.74792244, -14.74376731,
                   -13.73961219, -12.73545706, -11.73130194, -10.72714681,
                    -9.72299169,  -8.71883657,  -7.71468144,  -6.71052632])), 'hw': (np.array([ 1,  2,  2, 11, 35, 19, 18, 19, 13, 17, 10,  8,  6,  0,  2,  0,  1]), np.array([ 6.57244009,  7.57590029,  8.5793605 ,  9.58282071, 10.58628092,
                   11.58974113, 12.59320133, 13.59666154, 14.60012175, 15.60358196,
                   16.60704216, 17.61050237, 18.61396258, 19.61742279, 20.62088299,
                   21.6243432 , 22.62780341, 23.63126362]))}, 'durations0': {'cs': (np.array([53, 59, 37, 19,  9,  3,  2,  3,  1,  0,  0,  0,  0,  0,  0,  1]), np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204., 228., 252.,
                   276., 300., 324., 348., 372., 396.])), 'hw': (np.array([92, 49, 14,  6,  2,  1]), np.array([ 12.,  36.,  60.,  84., 108., 132., 156.]))}, 'hours_per_year': 741, 'x_solution': np.array([3.37869749e-01, 1.35605901e+00, 1.71737277e+00, 4.94912986e-01,
                   5.39332295e-03, 5.34034912e-03, 9.89053004e-01, 9.60857492e-01,
                   2.60122650e-03, 5.65243511e+02, 1.74433104e-04, 3.01323399e+02])}
            
            opt_val["x_solution"] = inputs['x_solution']
            opt_val["plot_results"] = self.plot_results
            param0 = inputs['param0']
            hist0 = inputs['hist0']
            durations0 = inputs['durations0']
            historic_time_interval = 650040
            frac_hours_per_year = np.array([0.08469633, 0.07716448, 0.08469633, 0.08196419, 0.08469633,
                   0.08307181, 0.08569319, 0.08469633, 0.08196419, 0.08469633,
                   0.08196419, 0.08469633])
            HOURS_IN_YEAR = 8760
            month = 1
            
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
                                   out_path=opt_val['out_path'],
                                   weights=opt_val['weights'],
                                   limit_temperatures=opt_val['limit_temperatures'],
                                   min_num_waves=opt_val['min_num_waves'],
                                   x_solution=opt_val['x_solution'],
                                   test_mode=opt_val['test_mode']) 
            
            resid1 = obj_solve.residuals
            
            inputs["x_solution"] = np.array([0.37869749e-01, 0.15605901e+00, 0.01737277e+00, 0.04912986e-01,
                   5.39332295e-03, 5.34034912e-03, 9.89053004e-01, 9.60857492e-01,
                   2.60122650e-03, 5.65243511e+02, 1.74433104e-04, 3.01323399e+02])
            
            obj_solve.reanalyze(inputs)
            
            resid2 = obj_solve.residuals
            self.assertTrue(resid1[1] > resid2[1])
        else:
            warn("\n\nThe test_single_known_case test was not run because run_test = False!\n\n")
        
    def test_single_known_case2(self):
        run_test = self._run_all_tests
        if run_test:
            random_seed = 7293821
            
            
            opt_val = {'problem_bounds': None, 
                       'decay_func_type': {'cs': 'exponential_cutoff', 'hw': 'exponential_cutoff'}, 
                       'use_cython': True, 
                       'num_cpu': -1, 
                       'plot_results': self.plot_results, 
                       'max_iter': 25, 
                       'plot_title': '', 'out_path': '', 
                       'weights': np.array([1., 1., 1., 1.,1.]), 
                       'limit_temperatures': False, 
                       'delT_above_shifted_extreme': {'cs': -10, 'hw': 10}, 
                       'num_step': 2000000, 
                       'min_num_waves': 10, 
                       'x_solution': None, 
                       'test_mode': False}
            
            inputs = {'num_step': 2000000,
             'param0': ({'cs': {'help': 'These statistics are already mapped from -1 ... 1 and _inverse_transform_fit is needed to return to actual degC and degC*hr values. If input of actual values is desired use transform_fit(X,max,min)',
                'energy_normal_param': {'mu': -0.09637132972696814,
                 'sig': 0.33367351187294125},
                'extreme_temp_normal_param': {'mu': -0.3739023995793261,
                 'sig': 0.33027180813677953},
                'max extreme temp per duration': 1.8135697651556644,
                'min extreme temp per duration': 0.6265700963010938,
                'hist max extreme temp per duration': 1.8135697651556644,
                'hist min extreme temp per duration': 0.6265700963010938,
                'max energy per duration': 1.59786617708341,
                'min energy per duration': 0.26493943258502145,
                'energy linear slope': 0.9147515369179842,
                'normalized extreme temp duration fit slope': 0.6093184608559103,
                'normalized extreme temp duration fit intercept': 0.4765308166244309,
                'normalizing energy': -4346.722222222223,
                'normalizing extreme temp': -23.31712962962963,
                'normalizing duration': 456,
                'historic time interval': 650040.0,
                'hourly prob stay in heat wave': 0.9884606468375257,
                'hourly prob of heat wave': 0.005318791400334364,
                'historical durations (durations0)': (np.array([55, 68, 34, 12,  9,  3,  5,  2,  0,  1,  0,  2,  0,  0,  0,  0,  0,
                         0,  1]),
                 np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204., 228., 252.,
                        276., 300., 324., 348., 372., 396., 420., 444., 468.])),
                'historical temperatures (hist0)': (np.array([ 2,  2,  0,  1,  1,  3, 10,  7,  4, 10, 12, 15, 14, 21, 26, 35, 14,
                        13,  1,  1]),
                 np.array([-23.71429398, -22.88024884, -22.0462037 , -21.21215856,
                        -20.37811343, -19.54406829, -18.71002315, -17.87597801,
                        -17.04193287, -16.20788773, -15.37384259, -14.53979745,
                        -13.70575231, -12.87170718, -12.03766204, -11.2036169 ,
                        -10.36957176,  -9.53552662,  -8.70148148,  -7.86743634,
                         -7.0333912 ]))},
               'hw': {'help': 'These statistics are already mapped from -1 ... 1 and _inverse_transform_fit is needed to return to actual degC and degC*hr values. If input of actual values is desired use transform_fit(X,max,min)',
                'energy_normal_param': {'mu': 0.08962260244583249,
                 'sig': 0.3761662661828063},
                'extreme_temp_normal_param': {'mu': -0.4156148791873582,
                 'sig': 0.37141737443335276},
                'max extreme temp per duration': 1.6056618887870147,
                'min extreme temp per duration': 0.7493661855698713,
                'hist max extreme temp per duration': 1.6056618887870147,
                'hist min extreme temp per duration': 0.7493661855698713,
                'max energy per duration': 1.4918141998166654,
                'min energy per duration': 0.12910763742847048,
                'energy linear slope': 0.8642296404774669,
                'normalized extreme temp duration fit slope': 0.4022972440636707,
                'normalized extreme temp duration fit intercept': 0.4806872353625872,
                'normalizing energy': 1557.9444444444448,
                'normalizing extreme temp': 22.606481481481485,
                'normalizing duration': 168,
                'historic time interval': 650040.0,
                'hourly prob stay in heat wave': 0.9790749455390084,
                'hourly prob of heat wave': 0.004321518012771671,
                'historical durations (durations0)': (np.array([80, 47, 18,  6,  1,  3,  1]),
                 np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180.])),
                'historical temperatures (hist0)': (np.array([ 7, 19, 23, 24, 18, 19, 12, 12,  6,  8,  2,  4,  0,  0,  1,  1]),
                 np.array([ 8.98300058,  9.86027018, 10.73753979, 11.61480939, 12.49207899,
                        13.3693486 , 14.2466182 , 15.1238878 , 16.00115741, 16.87842701,
                        17.75569661, 18.63296622, 19.51023582, 20.38750543, 21.26477503,
                        22.14204463, 23.01931424]))}},),
             'random_seed': 7293821,
             'hist0': {'cs': (np.array([ 2,  2,  0,  1,  1,  3, 10,  7,  4, 10, 12, 15, 14, 21, 26, 35, 14,
                      13,  1,  1]),
               np.array([-23.71429398, -22.88024884, -22.0462037 , -21.21215856,
                      -20.37811343, -19.54406829, -18.71002315, -17.87597801,
                      -17.04193287, -16.20788773, -15.37384259, -14.53979745,
                      -13.70575231, -12.87170718, -12.03766204, -11.2036169 ,
                      -10.36957176,  -9.53552662,  -8.70148148,  -7.86743634,
                       -7.0333912 ])),
              'hw': (np.array([ 7, 19, 23, 24, 18, 19, 12, 12,  6,  8,  2,  4,  0,  0,  1,  1]),
               np.array([ 8.98300058,  9.86027018, 10.73753979, 11.61480939, 12.49207899,
                      13.3693486 , 14.2466182 , 15.1238878 , 16.00115741, 16.87842701,
                      17.75569661, 18.63296622, 19.51023582, 20.38750543, 21.26477503,
                      22.14204463, 23.01931424]))},
             'durations0': {'cs': (np.array([55, 68, 34, 12,  9,  3,  5,  2,  0,  1,  0,  2,  0,  0,  0,  0,  0,
                       0,  1]),
               np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204., 228., 252.,
                      276., 300., 324., 348., 372., 396., 420., 444., 468.])),
              'hw': (np.array([80, 47, 18,  6,  1,  3,  1]),
               np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180.]))},
             'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
             'historic_time_interval': 650040,
             'hours_per_year': 741,
             'problem_bounds': None,
             'ipcc_shift': {'cs': None, 'hw': None},
             'decay_func_type': {'cs': 'exponential_cutoff', 'hw': 'exponential_cutoff'},
             'use_cython': True,
             'num_cpu': -1,
             'plot_results': False,
             'max_iter': 25,
             'plot_title': '',
             'out_path': '',
             'weights': np.array([1., 1., 1., 1.,1.]),
             'limit_temperatures': False,
             'min_num_waves': 10,
             'x_solution': None,
             'test_mode': False,
             'num_postprocess': 3,
             'extra_output_columns': {'future year': 'historic',
              'climate scenario': 'historic',
              'threshold confidence interval': 'historic',
              'month': 12}}
            
            opt_val["x_solution"] = np.array([1.74067384e+00, 1.30160996e+00, 6.05887379e-01, 7.41466612e-01,
                   4.92829943e-03, 4.21899149e-03, 9.90012012e-01, 9.66459463e-01,
                   1.17871443e-03, 5.82290685e+02, 2.94837586e-04, 2.01724077e+02])
            opt_val["plot_results"] = self.plot_results
            param0 = inputs['param0'][0]
            hist0 = inputs['hist0']
            durations0 = inputs['durations0']
            historic_time_interval = 650040
            frac_hours_per_year = np.array([0.08469633, 0.07716448, 0.08469633, 0.08196419, 0.08469633,
                   0.08307181, 0.08569319, 0.08469633, 0.08196419, 0.08469633,
                   0.08196419, 0.08469633])
            HOURS_IN_YEAR = 8760
            month = 1
            
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
                                   out_path=opt_val['out_path'],
                                   weights=opt_val['weights'],
                                   limit_temperatures=opt_val['limit_temperatures'],
                                   min_num_waves=opt_val['min_num_waves'],
                                   x_solution=opt_val['x_solution'],
                                   test_mode=opt_val['test_mode']) 
            
            resid1 = obj_solve.residuals
    
            inputs["x_solution"] = np.array([0.37869749e-01, 0.15605901e+00, 0.01737277e+00, 0.04912986e-01,
                   5.39332295e-03, 5.34034912e-03, 9.89053004e-01, 9.60857492e-01,
                   2.60122650e-03, 5.65243511e+02, 1.74433104e-04, 3.01323399e+02])
            inputs["param0"] = inputs["param0"][0]
            obj_solve.reanalyze(inputs)
            
            resid2 = obj_solve.residuals
    
            self.assertTrue(resid1[1] > resid2[1])  
    
    def test_known_worcester_case_June_2014_future(self):
        
        "This comes from a real case in Worcester MA in June 2014 shift"
        
        hist0 = {'hw':(np.array([ 1,  0,  0,  0,  1,  0, 10, 16, 29, 31, 25, 22, 12, 10,  9,  3,  2,1]),
         np.array([ 3.36374743,  4.1596222 ,  4.95549697,  5.75137174,  6.54724651,
                 7.34312128,  8.13899606,  8.93487083,  9.7307456 , 10.52662037,
                11.32249514, 12.11836991, 12.91424468, 13.71011946, 14.50599423,
                15.301869  , 16.09774377, 16.89361854, 17.68949331])),
                  'cs':(np.array([ 2,  1,  2,  0,  6,  4,  6,  7, 16, 10, 20, 25, 22, 20, 19, 27, 26,
                           8,  1,  4,  2,  2,  2,  2]),
                   np.array([-15.31915509, -14.89055266, -14.46195023, -14.0333478 ,
                          -13.60474537, -13.17614294, -12.74754051, -12.31893808,
                          -11.89033565, -11.46173322, -11.03313079, -10.60452836,
                          -10.17592593,  -9.7473235 ,  -9.31872106,  -8.89011863,
                           -8.4615162 ,  -8.03291377,  -7.60431134,  -7.17570891,
                           -6.74710648,  -6.31850405,  -5.88990162,  -5.46129919,
                           -5.03269676]))}

        
        durations0 = {'hw':(np.array([76, 44, 24, 15,  9,  1,  2,  1]),
         np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204.])),
                      'cs':(np.array([79, 71, 46, 18,  8,  3,  4,  4,  0,  0,  1]),
                       np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204., 228., 252.,
                              276.]))}
        
        historic_time_interval = 650040
        
        hours_per_year = 741
        
        random_seed = 7293821
        
        param0 = {'hw':{'help': 'These statistics are already mapped from -1 ... 1 and _inverse_transform_fit is needed to return to actual degC and degC*hr values. If input of actual values is desired use transform_fit(X,max,min)',
         'energy_normal_param': {'mu': -0.10226493816492782,
          'sig': 0.37642169697450595},
         'extreme_temp_normal_param': {'mu': 0.23483388928821727,
          'sig': 0.4613883374783218},
         'max extreme temp per duration': 1.7862463896925065,
         'min extreme temp per duration': -0.1406602593461735,
         'hist max extreme temp per duration': 1.7862463896925065,
         'hist min extreme temp per duration': -0.1406602593461735,
         'max energy per duration': 1.6310344056194355,
         'min energy per duration': 0.1718715046306651,
         'energy linear slope': 0.7603350767529872,
         'normalized extreme temp duration fit slope': 0.39821590280496677,
         'normalized extreme temp duration fit intercept': 0.5579202449045396,
         'normalizing energy': 1340.0000000000005,
         'normalizing extreme temp': 17.3125,
         'normalizing duration': 192,
         'historic time interval': 650040.0,
         'hourly prob stay in heat wave': 0.9788886199655599,
         'hourly prob of heat wave': 0.005405981846722733,
         'historical durations (durations0)': (np.array([76, 44, 24, 15,  9,  1,  2,  1]),
          np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204.])),
         'historical temperatures (hist0)': (np.array([ 1,  0,  0,  0,  1,  0, 10, 16, 29, 31, 25, 22, 12, 10,  9,  3,  2,
                  1]),
          np.array([ 3.36374743,  4.1596222 ,  4.95549697,  5.75137174,  6.54724651,
                  7.34312128,  8.13899606,  8.93487083,  9.7307456 , 10.52662037,
                 11.32249514, 12.11836991, 12.91424468, 13.71011946, 14.50599423,
                 15.301869  , 16.09774377, 16.89361854, 17.68949331])),
         'decay function': 'quadratic_times_exponential_decay_with_cutoff',
         'decay function coef': np.array([ 96.91670234,   0.97129716, 197.65052132])},
                  'cs':{'help': 'These statistics are already mapped from -1 ... 1 and _inverse_transform_fit is needed to return to actual degC and degC*hr values. If input of actual values is desired use transform_fit(X,max,min)',
        'energy_normal_param': {'mu': 0.012583096121127626,
         'sig': 0.3865940662591064},
        'extreme_temp_normal_param': {'mu': 0.2071786056066263,
         'sig': 0.48418059193534846},
        'max extreme temp per duration': 1.7795213892594484,
        'min extreme temp per duration': 0.4087431054359999,
        'hist max extreme temp per duration': 1.7795213892594484,
        'hist min extreme temp per duration': 0.4087431054359999,
        'max energy per duration': 1.6589417225644925,
        'min energy per duration': -0.025886010311640867,
        'energy linear slope': 0.8396179089345742,
        'normalized extreme temp duration fit slope': 0.47674102276569175,
        'normalized extreme temp duration fit intercept': 0.5526128780127326,
        'normalizing energy': -1518.3333333333333,
        'normalizing extreme temp': -15.113425925925927,
        'normalizing duration': 264,
        'historic time interval': 650040.0,
        'hourly prob stay in heat wave': 0.9692740128468249,
        'hourly prob of heat wave': 0.008634356940133602,
        'historical durations (durations0)': (np.array([79, 71, 46, 18,  8,  3,  4,  4,  0,  0,  1]),
         np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204., 228., 252.,
                276.])),
        'historical temperatures (hist0)': (np.array([ 2,  1,  2,  0,  6,  4,  6,  7, 16, 10, 20, 25, 22, 20, 19, 27, 26,
                 8,  1,  4,  2,  2,  2,  2]),
         np.array([-15.31915509, -14.89055266, -14.46195023, -14.0333478 ,
                -13.60474537, -13.17614294, -12.74754051, -12.31893808,
                -11.89033565, -11.46173322, -11.03313079, -10.60452836,
                -10.17592593,  -9.7473235 ,  -9.31872106,  -8.89011863,
                 -8.4615162 ,  -8.03291377,  -7.60431134,  -7.17570891,
                 -6.74710648,  -6.31850405,  -5.88990162,  -5.46129919,
                 -5.03269676])),
        'decay function': 'quadratic_times_exponential_decay_with_cutoff',
        'decay function coef': np.array([229.04473382,   0.9770868 , 534.34596956])}}
        
        
        opt_val =  {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                    'max_iter': 30,
                                    'limit_temperatures': False,
                                    'num_cpu': -1,
                                    'num_step': 2500000,
                                    'plot_results': self.plot_results,
                                    'decay_func_type': {'cs': 'quadratic_times_exponential_decay_with_cutoff', 'hw': "quadratic_times_exponential_decay_with_cutoff"},
                                    'test_mode': False,
                                    'min_num_waves': 10,
                                    'out_path': os.path.join("testing_output", "worcester_future_6_50%_2014_SSP245.png")}
        opt_val['problem_bounds'] = {'cs':{'delT_mu': (-4.0, 10.0),
                             'delT_sig multipliers': (0.05,3),
                             'P_event': (0.001, 0.02),
                             'P_sustain': (0.97, 0.999999),
                             'multipliers to max probability time': (0,1),
                             'cutoff time multipliers':(2,3),
                             'max peak prob for quadratic model': (0.98, 1.0)},
                       'hw':{'delT_mu': (-4.0, 10.0),
                             'delT_sig multipliers': (-2,3),
                             'P_event': (0.0005,0.02),
                             'P_sustain': (0.97,0.999999),
                             'multipliers to max probability time': (0.1,2),
                             'cutoff time multipliers':(1,3),
                             'max peak prob for quadratic model': (0.97, 1.0)}}
        opt_val['max_iter'] = 125
        opt_val['num_cpu'] = -1
        opt_val['weights'] = np.array([3.0,1,1,1,3.0])
        opt_val['use_cython'] = True
        opt_val['plot_title'] = "test_known_case_future"
        opt_val['x_solution'] = np.array([0.37869749e-01, 0.15605901e+00, 0.01737277e+00, 0.04912986e-01,
               5.39332295e-03, 5.34034912e-03, 9.89053004e-01, 9.60857492e-01,
               277.23720951,   0.99576397, 287.62982799,
               277.23720951,   0.99576397, 287.62982799])
        
        ipcc_shift = {'cs':None,
                      'hw':{'temperature':{'10 year':0.982942, '50 year':1.027711},
                            'frequency':{'10 year':1.898539, '50 year':2.510386}}}
        
        obj_solve = SolveDistributionShift(opt_val['num_step'], 
                               param0, 
                               random_seed, 
                               hist0, 
                               durations0, 
                               opt_val['delT_above_shifted_extreme'], 
                               historic_time_interval,
                               hours_per_year,
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
                               test_mode=opt_val['test_mode']) 
        resid1 = obj_solve.residuals[1]
        inputs = obj_solve.inputs
        
        inputs['out_path'] = os.path.join("testing_output", "worcester_future_6_50%_2014_SSP245_ADJUSTED.png")
        
        inputs['x_solution'] =  np.array([-1, -0.15605901e+00, 0.12737277e+00, -.25912986e-00,
               5.39332295e-03, 5.34034912e-03, 9.89053004e-01, 9.60857492e-01,
               277.23720951,   0.99576397, 287.62982799,
               277.23720951,   0.99576397, 287.62982799])
        
        obj_solve.reanalyze(inputs)
        resid2 = obj_solve.residuals[1]
        
        self.assertTrue(resid1 > resid2)
        
    
    def test_assert_warns_on_solution_boundary(self):
        run_test = self._run_all_tests
        if run_test:

            default_problem_bounds = {'cs':{'delT_mu': (0.0, 2.0),
                                             'delT_sig multipliers': (-0.1,4),
                                             'P_event': (0.00025, 0.0125),
                                             'P_sustain': (0.975, 0.999999),
                                             'multipliers to max probability time': (0,2),
                                             'cutoff time multipliers':(1,4),
                                             'max peak prob for quadratic model': (0.976, 1.0)},
                                       'hw':{'delT_mu': (0.0, 20.0),
                                             'delT_sig multipliers': (-0.1,10),
                                             'P_event': (0.00025,0.0325),
                                             'P_sustain': (0.985,0.999999),
                                             'slope or exponent multipliers' : (0,1),
                                             'cutoff time multipliers':(2,6),
                                             'delay_time_multipliers':(0.1,3)}}
            
            decay_func_type = {'hw':""}
            
            weights = np.array([100.0,1.0,1.0,1.0,1.0])
            
            hours_per_year = 31 * 24
            
            resid_vals = []
            delT_above_50_year = {'hw':20.0,'cs':-20.0} # degrees celcius
            num_step = 8760*50  # provides 11 50 year periods
            rng = default_rng(29023430)
            decay_func_type = [{'cs':'quadratic_times_exponential_decay_with_cutoff',
                                'hw':'delayed_exponential_decay_with_cutoff'}]#[None,"linear","exponential","exponential_cutoff","quadratic_times_exponential_decay_with_cutoff"]  # 8 parameters

            for decay_func in decay_func_type:
                param = {}
                param['hw'] = {}
                param['cs'] = {}
                # taken from extreme._add_extreme
                param['hw']['normalizing duration'] = 192
                param['hw']['extreme_temp_normal_param'] = {}
                param['hw']['extreme_temp_normal_param']['mu'] = 0.0
                param['hw']['extreme_temp_normal_param']['sig'] = 0.3
                param['hw']['max extreme temp per duration'] = 0.55
                param['hw']['min extreme temp per duration'] = 0.1
                param['hw']['hist max extreme temp per duration'] = 0.55
                param['hw']['hist min extreme temp per duration'] = 0.1
                param['hw']['normalizing extreme temp'] = 18
                param['hw']['normalized extreme temp duration fit slope'] = 1.0
                param['hw']['normalized extreme temp duration fit intercept'] = 0.2
                param['hw']['hourly prob stay in heat wave'] = 0.98 
                param['hw']['hourly prob of heat wave'] = 0.002
                
                param['cs']['normalizing duration'] = 192
                param['cs']['extreme_temp_normal_param'] = {}
                param['cs']['extreme_temp_normal_param']['mu'] = 0.0
                param['cs']['extreme_temp_normal_param']['sig'] = 0.3
                param['cs']['max extreme temp per duration'] = 0.55
                param['cs']['min extreme temp per duration'] = 0.1
                param['cs']['hist max extreme temp per duration'] = 0.55
                param['cs']['hist min extreme temp per duration'] = 0.1
                param['cs']['normalizing extreme temp'] = -18
                param['cs']['normalized extreme temp duration fit slope'] = 1.0
                param['cs']['normalized extreme temp duration fit intercept'] = 0.2
                param['cs']['hourly prob stay in heat wave'] = 0.98 
                param['cs']['hourly prob of heat wave'] = 0.002
        
                # None indicates to just fit the historical pdf
                ipcc_shift = {'hw':{'frequency':{'10 year':8.5,'50 year':39.5},
                                'temperature':{'10 year':4.5,'50 year':6.12}},
                              'cs':None}
                
                hist0 = {}
                durations0 = {}
                hist0['hw'] = (np.array([16, 22, 25, 15, 16, 13,  9,  5,  5,  2, 1,  0, 1,  2,  1, 0,0,  1, 0,0,1, 0,1]),
                                  np.array(np.arange(0.0,12,0.5)))
                hist0['cs'] = (np.array([1,0,1,0,0,1,2,0,1,2,1,0,3,2,5,7,11,14,17,22,32,32,21]),
                                  np.array(np.arange(-12,-0.0,0.5)))
                
                durations0['hw'] = (np.array([50,35,15,13,10,9,3,1]),
                                  np.array(np.arange(12,228,24)))
                durations0['cs'] = (np.array([70,40,25,15,12,6,5,2]),
                                  np.array(np.arange(24,228,24)))
                
                # 70 years of records assumed.
                historic_time_interval = 70 * 24 * 365
                
                random_seed = 5618578
                
                # # negative one indicates use all processors available.
                num_cpu = 1
                use_cython = True
                
                if decay_func is None:
                    decay_func_str = "no decay"
                else:
                    decay_func_str = decay_func['hw']
                
                x0 = np.array([1.13285305e+00, 8.30364720e-02, 1.24392011e+01, 1.63707546e+00,
                               7.75621568e-03, 3.11526403e-03, 9.84492160e-01, 9.87468495e-01,
                               0.0, 9.94307859e-01, 2.07898714e+02, 
                               1.05371387e-04, 4.77138188e+02, 2.21193289e+00])
            
                
                # SHIFT PER IPCC
                # low number of iterations because this is unit testing.
                with self.assertWarns(UserWarning):
                    obj1 = SolveDistributionShift(num_step,
                                           param,
                                           random_seed,
                                           hist0, 
                                           durations0,
                                           delT_above_50_year,
                                           historic_time_interval,
                                           hours_per_year,
                                           default_problem_bounds,
                                           ipcc_shift,
                                           decay_func_type=decay_func,
                                           use_cython=use_cython,
                                           plot_results=self.plot_results,
                                           max_iter=1,
                                           plot_title="Shift to future "+decay_func_str,
                                           num_cpu=num_cpu,
                                           weights=weights,
                                           limit_temperatures=False,
                                           min_num_waves=25,
                                           x_solution=x0)
            


    
    
    def test_duration_residuals_func(self):
        run_test = self._run_all_tests
        if run_test:
            duration0 = (np.array([30.0,20.3,8.0]),np.array([1,2,3,4]))
            ipcc = {'frequency': {'10 year': 2.0, '50 year': 39.5}, 'temperature': {'10 year': 3.0, '50 year': 6.12}}
            
            duration_too_low = (np.array([20.0,10.0,11.0]),np.array([1,2,3,4]))
            
            # must give positive penalty.
            penalty = duration_residuals_func(duration_too_low,duration0,ipcc,np.array([1.0,1.0,1.0,1.0]))
            
            self.assertTrue(penalty > 0.0)
        
            duration_too_high = (duration_too_low[0]*10,duration_too_low[1])
            
            penalty = duration_residuals_func(duration_too_high,duration0,ipcc,np.array([1.0,1.0,1.0,1.0]))
            
            self.assertTrue(penalty > 0.0)
            
            # now try the boundaries and assure penalty = 0
            
            penalty = duration_residuals_func(duration0,duration0,ipcc,np.array([1.0,1.0,1.0,1.0]))
            
            self.assertAlmostEqual(penalty, 0.0)
            
            
            # exactly right on the high side
            penalty = duration_residuals_func((duration0[0]*ipcc['frequency']['10 year'], duration0[1]),duration0,ipcc,np.array([1.0,1.0,1.0,1.0]))
            self.assertAlmostEqual(penalty, 0.0)
            
            # in between
            penalty = duration_residuals_func((np.array([40.0,25.1,8.0]),duration0[1]),duration0,ipcc,np.array([1.0,1.0,1.0,1.0]))
            self.assertAlmostEqual(penalty, 0.0)
            
            # penalty for nonoverlapping content
            penalty = duration_residuals_func((np.array([40.0,25.1,8.0,1.0]),np.array([1,2,3,4,5])),duration0,ipcc,np.array([1.0,1.0,1.0,1.0]))
            self.assertTrue(penalty > 0.0)
        else:
            warn("Test test_duration_residuals_func was not executed because run_test was manually set to False")
        
if __name__ == "__main__":
    
    profile = False
    
    if profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_SolveDistributionShift())
    
    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        with open('solve_test_profile.txt', 'w+') as f:
            f.write(s.getvalue())
    #
