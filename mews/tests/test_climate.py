#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:22:44 2022

@author: dlvilla
"""

from numpy.random import default_rng

import unittest
import os
from matplotlib import pyplot as plt
from matplotlib import rc

import warnings

from mews.weather.climate import ClimateScenario


rng = default_rng()

class Test_ClimateScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = False
        cls.write_results = False
        cls.rng = default_rng()
        fpath = os.path.dirname(__file__)
        cls.path_to_file = os.path.join(fpath,"write.png")
        cls.model_guide = os.path.join(fpath,"data_for_testing","Models_Used_Simplified.xlsx")
        cls.data_folder = os.path.join(fpath,"data_for_testing","CMIP6_Data_Files")
        
        if os.path.exists(cls.path_to_file):
            os.remove(cls.path_to_file)
        
        
        # for the proxy settings to work, a file external to the repository
        # must call out the proxy server specifics!
        proxy_location = os.path.join("..","..","..","proxy.txt")
        
        if os.path.exists(proxy_location):
            with open(proxy_location,'r') as f:
                cls.proxy = f.read()
        else:
            warnings.warn("No proxy settings! If you need for proxy settings to be" +
                          " active then you need to place the correct proxy server in "+
                          os.path.abspath(proxy_location) + " for MEWS to download CMIP6 data.")
            cls.proxy = None
        
               

        plt.close('all')
        font = {'size':16}
        rc('font', **font)   
    
    @classmethod
    def tearDownClass(cls):
        if cls.plot_results:
            plt.close('all')

    def test_old_use_case(self):
        clim_scen = ClimateScenario(use_global=True,lat=45,lon=-105,
                                    data_folder=self.data_folder,
                                    run_parallel=True,proxy=self.proxy,gcm_to_skip=["NorESM2-MM"],
                                    write_graphics_path=None,
                                    align_gcm_to_historical=True,model_guide =self.model_guide) 
        # Errors
        with self.assertRaises(ValueError):
            clim_func_2 = clim_scen.calculate_coef("SSP-5.85")
            # test the old use case

        
        clim_func = clim_scen.calculate_coef("SSP585")  

    def test_climate_scenario_with_specific_lat_lon(self):
        """
        This tests extreme temperature with a specific latitude and longitude
        
        
        """

        clim_scen = ClimateScenario(use_global=False,lat=45,lon=-105,
                                    data_folder=self.data_folder,
                                    run_parallel=True,proxy=self.proxy,gcm_to_skip=["NorESM2-MM"],
                                    write_graphics_path=self.path_to_file,
                                    align_gcm_to_historical=True,model_guide =self.model_guide)  
        
        clim_func = clim_scen.calculate_coef(["SSP585"])  
    
        self.assertTrue(os.path.exists(self.path_to_file))
        
        with self.assertRaises(TypeError):
            clim_func_2 = clim_scen.calculate_coef("SSP585")
        
        os.remove(self.path_to_file)
        clim_scen_low = ClimateScenario(use_global=False,lat=0.0,lon=0.0,
                                    data_folder=self.data_folder,
                                    run_parallel=False,proxy=self.proxy,gcm_to_skip=["NorESM2-MM"],baseline_year=2014,
                                    write_graphics_path=None,
                                    align_gcm_to_historical=True,model_guide =self.model_guide) 
        
        # make sure write_graphics_path does not write anyway.
        self.assertFalse(os.path.exists(self.path_to_file))
        
        clim_scen_high = ClimateScenario(use_global=False,lat=50.0,lon=0.0,
                                    data_folder=self.data_folder,
                                    run_parallel=True,proxy=None,gcm_to_skip=["NorESM2-MM"],
                                    write_graphics_path=None,
                                    align_gcm_to_historical=True,model_guide ="Models_Used_Simplified.xlsx") 
        
        
        
        
        
        
if __name__ == "__main__":
    o = unittest.main(Test_ClimateScenario())