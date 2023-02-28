# -*- coding: utf-8 -*-
"""
Created on Wed July 20 13:24:14 2022

@author: tschoste

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

import unittest
from mews.data_requests.CMIP6 import CMIP_Data
import os

class TestCMIP(unittest.TestCase):
    
# OBJECT CREATION
    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        fpath = os.path.dirname(__file__)
        cls.model_guide = os.path.join(fpath,"data_for_testing","Models_Used_Simplified.xlsx")
        cls.data_folder = os.path.join(fpath,"data_for_testing","CMIP6_Data_Files")
        os.chdir(fpath)
        
        
    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        
    def test_temperature_data_collection(self):
        #Testing invalid coordinates
        with self.assertRaises(ValueError):
                CMIP_Data(lat_desired = 40,
                          lon_desired = 1000,
                          baseline_year = 2014,
                          end_year = 2100,
                          file_path = os.path.join(__file__),
                          data_folder = self.data_folder,
                          model_guide = self.model_guide,
                          scenario_list = ["SSP585"],
                          world_map = False,
                          calculate_error = False)
    
        #Testing all calculations have been computed
        test_obj = CMIP_Data(lat_desired = 40,
                             lon_desired = -100,
                             baseline_year = 2014,
                             end_year = 2100,
                             file_path = os.path.join(__file__),
                             data_folder = self.data_folder,
                             model_guide = self.model_guide,
                             scenario_list = ["historical","SSP585"],
                             calculate_error = True,
                             display_plots = False,
                             run_parallel=True)
        self.assertIsNotNone(test_obj.total_model_data["SSP585"].avg_error)
        self.assertNotEqual([],test_obj.total_model_data["SSP585"].delT_list)
        self.assertNotEqual([],test_obj.total_model_data["SSP585"].delT_list_reg)
        self.assertNotEqual([],test_obj.total_model_data["SSP585"].CI_list)
        
        #Testing plotting
        test_obj.results1(scatter_display=[True,True,True,True,True,True])
        test_obj.results2(desired_scenario = "SSP585",resolution = "test")

        

if __name__=='__main__':
    o=unittest.main(TestCMIP())  