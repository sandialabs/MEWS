# -*- coding: utf-8 -*-
"""
Created on Wed July 20 13:24:14 2022

@author: tschoste
"""

import unittest
from mews.requests.CMIP6 import CMIP_Data
import os

class TestCMIP(unittest.TestCase):
    
# OBJECT CREATION
    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        
    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        
    
    def test_temperature_data_collection(self):
        #Testing invalid coordinates
        with self.assertRaises(ValueError):
                CMIP_Data(lat_desired = 40,
                          lon_desired = 1000,
                          year_baseline = 2014,
                          year_desired = 2100,
                          file_path = os.path.join(__file__),
                          data_folder = "data_for_testing",
                          model_guide = "Models_Used_Simplified.xlsx",
                          scenario_list = ["SSP585"],
                          world_map = False,
                          calculate_error = False)
    
        #Testing all calculations have been computed
        test_obj = CMIP_Data(lat_desired = 40,
                             lon_desired = -100,
                             year_baseline = 2014,
                             year_desired = 2100,
                             file_path = os.path.join(__file__),
                             data_folder = "data_for_testing",
                             model_guide = "Models_Used_Simplified.xlsx",
                             scenario_list = ["historical","SSP585"],
                             calculate_error = True)
        self.assertIsNotNone(test_obj.total_model_data["SSP585"].avg_error)
        self.assertNotEqual([],test_obj.total_model_data["SSP585"].delT_list)
        self.assertNotEqual([],test_obj.total_model_data["SSP585"].delT_list_reg)
        self.assertNotEqual([],test_obj.total_model_data["SSP585"].CI_list)
        
        #Testing plotting
        test_obj.results1(scatter_display=[True,True,True,True,True,True])
        test_obj.results2(desired_scenario = "SSP585",resolution = "test")
        

if __name__=='__main__':
    o=unittest.main(TestCMIP())  