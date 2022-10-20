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
from mews.stats.solve import markov_gaussian_model_for_peak_temperature
from numpy.random import default_rng
from matplotlib import pyplot as plt

import numpy as np
import unittest
import os


class Test_SolveDistributionShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = False
        cls.write_results = False
        cls.rng = default_rng()
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
        
        default_problem_bounds = {'cs':{'delT_mu': (0.0, 0.7),
                                         'delT_sig multipliers': (-0.1,2),
                                         'P_event': (0.00001, 0.03),
                                         'P_sustain': (0.958, 0.999999),
                                         'multipliers to max probability time': (0,2),
                                         'slope or exponent multipliers' : (0,1),
                                         'cutoff time multipliers':(1,3),
                                         'max peak prob for quadratic model': (0.97, 1.0)},
                                   'hw':{'delT_mu': (0.0, 0.7),
                                         'delT_sig multipliers': (-0.1,2),
                                         'P_event': (0.00001,0.03),
                                         'P_sustain': (0.958,0.999999),
                                         'multipliers to max probability time': (0.1,2),
                                         'slope or exponent multipliers' : (0,1),
                                         'cutoff time multipliers':(1,3),
                                         'max peak prob for quadratic model': (0.97, 1.0)}}
            
        resid_vals = []
        delT_above_50_year = {'hw':20.0,'cs':-20.0} # degrees celcius
        num_step = 500000  # provides 11 50 year periods
        rng = default_rng(29023430)
        decay_func_type = [None,"linear","exponential","exponential_cutoff","quadratic_times_exponential_decay_with_cutoff"]  # 8 parameters
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
            param['hw']['normalized extreme temp duration fit intercept'] = 0.0
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
            param['cs']['normalized extreme temp duration fit intercept'] = 0.0
            param['cs']['hourly prob stay in heat wave'] = 0.98 
            param['cs']['hourly prob of heat wave'] = 0.002
    
            # None indicates to just fit the historical pdf
            ipcc_shift = {'hw':{'frequency':{'10 year':8.5,'50 year':39.5},
                            'temperature':{'10 year':4.5,'50 year':6.12}},
                          'cs':None}
            
            hist0 = {}
            hist0['hw'] = (np.array([50,65,75,80,90,70,50,25,30,10,6,8,9,5,4,3,2,1,1,1,1,1,1]),
                              np.array(np.arange(2.0,14,0.5)))
            hist0['cs'] = (np.array([1,1,1,1,1,1,2,3,4,5,9,8,6,10,30,25,50,70,90,80,75,65,50]),
                              np.array(np.arange(-14,-2.0,0.5)))
            
            random_seed = 561854
            
            if decay_func is None:
                x0 = np.array([0.1,
                 0.05,
                 0.1,
                 0.05,
                 0.001,
                 0.002,
                 0.98,
                 0.985])
            elif "cutoff" in decay_func and not "quadratic" in decay_func:
                x0 = np.array([0.1,
                     0.05,
                     0.1,
                     0.05,
                     0.001,
                     0.002,
                     0.98,
                     0.985,
                     0.02,
                     0.02,
                     1000,
                     1000])
            elif "quadratic" in decay_func:
                x0 = np.array([0.1,
                     0.05,
                     0.1,
                     0.05,
                     0.001,
                     0.002,
                     0.98,
                     0.985,
                     0.02,
                     0.02,
                     1000,
                     1000,
                     0.997,
                     0.999])
            else:
                x0 = np.array([0.1,
                     0.05,
                     0.1,
                     0.05,
                     0.001,
                     0.002,
                     0.98,
                     0.985,
                     0.02,
                     0.02])
            
            # run this once so that any obvious bugs will come out 
            # before it goes into parallel mode in differential_evolution.
            resid = markov_gaussian_model_for_peak_temperature(x0,
                                                       num_step,
                                                       random_seed,
                                                       param,
                                                       hist0,
                                                       ipcc_shift,
                                                       decay_func_type=decay_func,
                                                       use_cython=True,
                                                       output_hist=True,
                                                       delT_above_shifted_extreme=delT_above_50_year)
            
            fit_hist = False
            
            if decay_func is None:
                decay_func_str = "no decay"
            else:
                decay_func_str = decay_func
            
            # SHIFT PER IPCC
            # low number of iterations because this is unit testing.
            obj = SolveDistributionShift(x0,
                                   num_step,  
                                   param,
                                   random_seed,
                                   hist0, 
                                   delT_above_50_year,
                                   default_problem_bounds,
                                   ipcc_shift,
                                   decay_func_type=decay_func,
                                   use_cython=True,
                                   plot_results=self.plot_results,
                                   max_iter=1,
                                   plot_title="Fit histogram "+decay_func_str)
            # Just fit the histograms - no ipcc_shift factors
            ipcc_shift = {'hw':None, 'cs':None}
            obj2 = SolveDistributionShift(x0,
                                   num_step,  
                                   param,
                                   random_seed,
                                   hist0, 
                                   delT_above_50_year,
                                   default_problem_bounds,
                                   ipcc_shift,
                                   decay_func_type=decay_func,
                                   use_cython=True,
                                   plot_results=self.plot_results,
                                   max_iter=1,
                                   plot_title="Shift to future "+decay_func_str)
    
            optimize_result = obj.optimize_result

            resid_vals.append([val for key,val in obj.residuals.items()])
            resid_vals.append([val for key,val in obj2.residuals.items()])
        self.assertTrue(np.concatenate(resid_vals).max() < 1.0)


              
        

        
if __name__ == "__main__":
    o = unittest.main(Test_SolveDistributionShift())
