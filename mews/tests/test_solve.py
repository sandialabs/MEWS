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
        x[0] = del_mu_T
        x[1] = del_sig_T
        x[2] = Pcs 
        x[3] = Phw
        x[4] = Pcss - probability cold snap is sustained
        x[5] = Phws - probability heat wave is sustained
        x[6] = lamb_cs or slope_cs
        x[7] = lamb_hw or slope_hw 
        x[8] = cutoff_cs
        x[9] = cutoff_hw
        
        """
        delT_above_50_year = 5.0 # degrees celcius
        num_step = 1000000  # provides 11 50 year periods
        rng = default_rng(29023430)
        decay_func_type = [None,"linear","exponential","exponential_cutoff"]  # 8 parameters
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
            param['cs']['normalizing extreme temp'] = 18
            param['cs']['normalized extreme temp duration fit slope'] = 1.0
            param['cs']['normalized extreme temp duration fit intercept'] = 0.0
            param['cs']['hourly prob stay in heat wave'] = 0.98 
            param['cs']['hourly prob of heat wave'] = 0.002
    
            
            ipcc_shift = {'frequency':{'10 year':8.5,'50 year':39.5},
                          'temperature':{'10 year':4.5,'50 year':6.12}}
            
            
            hist0 = (np.array([50,65,75,80,90,70,50,25,30,10,6,8,9,5,4,3,2,1,1,1,1,1,1]),
                              np.array(np.arange(2.0,14,0.5)))
            
            random_seed = 561854
            
            if decay_func is None:
                x0 = np.array([0.1,
                 0.05,
                 0.001,
                 0.002,
                 0.98,
                 0.985])
            elif "cutoff" in decay_func:
                x0 = np.array([0.1,
                     0.05,
                     0.001,
                     0.002,
                     0.98,
                     0.985,
                     0.02,
                     0.02,
                     1000,
                     1000])
            else:
                x0 = np.array([0.1,
                     0.05,
                     0.001,
                     0.002,
                     0.98,
                     0.985,
                     0.02,
                     0.02])
            wave_type = 'hw'
            
            resid = markov_gaussian_model_for_peak_temperature(x0,
                                                       num_step,
                                                       random_seed,
                                                       param,
                                                       wave_type,
                                                       hist0,
                                                       ipcc_shift,
                                                       decay_func_type=decay_func,
                                                       use_cython=True,
                                                       output_hist=True)
            
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
                                   wave_type,
                                   hist0, 
                                   fit_hist,
                                   ipcc_shift,
                                   decay_func_type=decay_func,
                                   use_cython=True,
                                   plot_results=self.plot_results,
                                   max_iter=6,
                                   plot_title="Fit historgram "+decay_func_str)
            # Just fit the histograms
            fit_hist = True
            obj2 = SolveDistributionShift(x0,
                                   num_step,  
                                   param,
                                   random_seed,
                                   wave_type,
                                   hist0, 
                                   fit_hist,
                                   ipcc_shift,
                                   decay_func_type=decay_func_type,
                                   use_cython=True,
                                   plot_results=self.plot_results,
                                   max_iter=6,
                                   plot_title="Shift to future "+decay_func_str)
    
            optimize_result = obj.optimize_result


              
        

        
if __name__ == "__main__":
    o = unittest.main(Test_SolveDistributionShift())
