# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:11:22 2022

@author: dlvilla
"""

from mews.tests.test_extreme_temperature_waves import Test_ExtremeTemperatureWaves

obj = Test_ExtremeTemperatureWaves()

# frequency convergence
convergence_array = [10,20,40,80,160,320,640]
random_seed = [2293,23354,23123,46234,1325,897,234908]

metric_meta_dict = {}
for ca,rs in zip(convergence_array,random_seed):

    metric_meta_dict[(ca,rs)] = obj._run_verification_study_of_use_global_False(ca,
                                                    2080,
                                                    rs,
                                                    False,
                                                    "SSP585")
    