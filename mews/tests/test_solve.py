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

from mews.stats.solve import SolveDistributionShift
from mews.stats.extreme import DiscreteMarkov
from mews.stats.solve import ObjectiveFunction
from numpy.random import default_rng
from matplotlib import pyplot as plt
from mews.utilities.utilities import find_extreme_intervals

import numpy as np
import unittest
import os


class Test_SolveDistributionShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = False
        cls.write_results = False
        cls.rng = default_rng(23985)
        plt.close('all')
        if os.path.exists("testing_output"):
            os.removedirs(("testing_output"))
        os.mkdir("testing_output")
        
    
    @classmethod
    def tearDownClass(cls):
        
        if os.path.exists("testing_output"):
            os.removedirs(("testing_output"))
        pass
    
    
    def test_decay_func_types(self):
        # keep in mind
        """

        
        """
        
        # decay_func_type = {'cs': 'quadratic_times_exponential_decay_with_cutoff', 'hw': 'exponential_cutoff'}
        # tran_matrix = np.array([[0.95571133, 0.02439947, 0.01988919],
        #        [0.01266096, 0.98733904, 0.        ],
        #        [0.00515797, 0.        , 0.99484203]])
        # coef = {'cs': np.array([290.63051208, 331.69958068,   0.99506999]), 'hw': np.array([2.15768819e-03, 2.22306236e+02])}
        # objDM = DiscreteMarkov(self.rng, 
        #                        tran_matrix, 
        #                        decay_func_type=decay_func_type,
        #                        coef=coef,
        #                        use_cython=False)
        # states_arr = objDM.history(1000000, 0)
        
        # state_intervals = find_extreme_intervals(states_arr, [1,2])
        
        # durations = {}
        # for wave_type,state_int in zip(['cs','hw'],[1,2]):
            
        #     durations[wave_type] = np.array([tup[1]-tup[0]+1 for tup in state_intervals[state_int]])   
            
        #     if durations[wave_type].max() > coef[wave_type][1]:
        #         print("violation!\n\n")
        #         breakpoint()
        
        
        default_problem_bounds = {'cs':{'delT_mu': (0.0, 2.0),
                                         'delT_sig multipliers': (-0.1,4),
                                         'P_event': (0.00025, 0.0125),
                                         'P_sustain': (0.975, 0.999999),
                                         'multipliers to max probability time': (0,2),
                                         'slope or exponent multipliers' : (0,1),
                                         'cutoff time multipliers':(1,4),
                                         'max peak prob for quadratic model': (0.976, 1.0)},
                                   'hw':{'delT_mu': (0.0, 2.0),
                                         'delT_sig multipliers': (-0.1,4),
                                         'P_event': (0.00025,0.0125),
                                         'P_sustain': (0.985,0.999999),
                                         'multipliers to max probability time': (0.1,2),
                                         'slope or exponent multipliers' : (0,1),
                                         'cutoff time multipliers':(1,4),
                                         'max peak prob for quadratic model': (0.986, 1.0)}}
        
        decay_func_type = {'hw':""}
        
        weights = np.array([100.0,1.0,1.0,1.0/12])
        
        hours_per_year = 31 * 24
        
        resid_vals = []
        delT_above_50_year = {'hw':20.0,'cs':-20.0} # degrees celcius
        num_step = 8760*50  # provides 11 50 year periods
        rng = default_rng(29023430)
        decay_func_type = [{'cs':'quadratic_times_exponential_decay_with_cutoff',
                            'hw':'quadratic_times_exponential_decay_with_cutoff'}]#[None,"linear","exponential","exponential_cutoff","quadratic_times_exponential_decay_with_cutoff"]  # 8 parameters
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
            hist0['hw'] = (np.array([140,120,85,80,90,70,50,25,30,10,6,8,9,5,4,3,2,1,1,1,1,1,1]),
                              np.array(np.arange(0.0,12,0.5)))
            hist0['cs'] = (np.array([1,1,1,1,1,1,2,3,4,5,9,8,6,10,30,25,50,70,90,80,85,120,140]),
                              np.array(np.arange(-12,-0.0,0.5)))
            
            durations0['hw'] = (np.array([250,175,155,79,50,29,5]),
                              np.array(np.arange(24,192,24)))
            durations0['cs'] = (np.array([250,175,155,79,50,29,5]),
                              np.array(np.arange(24,192,24)))
            
            # 70 years of records assumed.
            historic_time_interval = 70 * 24 * 365
            
            random_seed = 5618578
            
            # # negative one indicates use all processors available.
            num_cpu = -1
            use_cython = True
            # # use this to analyze a specific case found by the solution for troubleshooting.
            # # 
            # #
            # x0 = np.array([1.45938105e-01, 4.73714254e-02, 5.75799556e-01, 1.08612636e-01,
            #                 1.64842479e-03, 0.002, 0.992, 9.83958194e-01,
            #                 2.19334366e+02, 1.01405802e+03, 0.999, 2.19334366e+02, 1.01405802e+03, 0.999])
            # # run this once so that any obvious bugs will come out 
            # # before it goes into parallel mode in differential_evolution.
            # obj_func = ObjectiveFunction(['cs','hw'],random_seed)
            # resid = obj_func.markov_gaussian_model_for_peak_temperature(x0,
            #                                             num_step,
            #                                             param,
            #                                             hist0,
            #                                             durations0,
            #                                             historic_time_interval,
            #                                             ipcc_shift,
            #                                             decay_func_type=decay_func,
            #                                             use_cython=use_cython,    
            #                                             output_hist=False,
            #                                             delT_above_shifted_extreme=delT_above_50_year,
            #                                             weights=weights)
            
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

        self.assertTrue(np.concatenate(resid_vals).max() < 10.0)


              
        

        
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
