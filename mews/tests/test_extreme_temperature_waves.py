# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:07:57 2021

@author: dlvilla
"""


from numpy.random import default_rng

import unittest
import os
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np

import warnings

from mews.weather.climate import ClimateScenario
from mews.events import ExtremeTemperatureWaves
from mews.graphics.plotting2D import Graphics
import matplotlib.pyplot as plt
import os
import warnings

rng = default_rng()

class Test_ExtremeTemperatureWaves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = False
        cls.write_results = False
        cls.rng = default_rng()
        
        try:
            os.removedirs("mews_results")
        except:
            warnings.warn("The testing could not remove the temporary directory ./mews/tests/mews_results")
        
        proxy_location = os.path.join("..","..","..","proxy.txt")
        
        if os.path.exists(proxy_location):
            with open(proxy_location,'r') as f:
                cls.proxy = f.read()
        else:
            warnings.warn("No proxy settings! If you need for proxy settings to be" +
                          " active then you need to place the correct proxy server in "+
                          os.path.abspath(proxy_location) + " for MEWS to download CMIP6 data.")
            cls.proxy = None
        
        if not os.path.exists("data_for_testing"):
            os.chdir(os.path.join(".","mews","tests"))
            cls.from_main_dir = True
        else:
            cls.from_main_dir = False
               
        cls.test_weather_path = os.path.join(".","data_for_testing")
        erase_me_file_path = os.path.join(cls.test_weather_path,"erase_me_file.epw")
        if os.path.exists(erase_me_file_path):
            os.remove(erase_me_file_path)

        cls.test_weather_file_path = os.path.join(".",
                                                  cls.test_weather_path,
                                                  "USA_NM_Santa.Fe.County.Muni.AP.723656_TMY3.epw")

        plt.close('all')
        font = {'size':16}
        rc('font', **font)   
        
        fpath = os.path.dirname(__file__)
        cls.model_guide = os.path.join(fpath,"data_for_testing","Models_Used_Simplified.xlsx")
        cls.data_folder = os.path.join(fpath,"data_for_testing","CMIP6_Data_Files")
    
    @classmethod
    def tearDownClass(cls):

        for file_name in os.listdir():
            if ("NM_Albuquerque_Intl_ArptTMY3" in file_name or
                "USA_NM_Santa.Fe.County.Muni.AP.723656" in file_name or
                "WEATHER.FMT" in file_name or
                "TXT2BIN.EXE" in file_name or
                "INPUT.DAT" in file_name or
                "USA_NM_Alb" in file_name or 
                ".epw" in file_name):
                    try:
                        os.remove(file_name)
                    except:
                        warnings.warn("The testing could not clean up files"
                                      +" and the ./mews/tests folder has residual "
                                      +"*.bin, *.epw, *.EXE, or *.DAT files"
                                      +" that need to be removed!")
        if os.path.exists(os.path.join(".","mews_results")):
            for file_name in os.listdir(os.path.join(".","mews_results")):
                if (".epw" in file_name):
                    try:
                        os.remove(os.path.join("mews_results",file_name))
                    except:
                        warnings.warn("The testing could not clean up files"
                                          +" and the tests folders have residual "
                                          +"*.epw files in ./mews/tests/mews_results"
                                          +" that need to be removed!")
        try:
            os.removedirs("mews_results")
        except:
            warnings.warn("The testing could not remove the temporary directory ./mews/tests/mews_results")
                        
        if cls.from_main_dir:
            os.chdir(os.path.join("..",".."))
            
                        
    def test_albuquerque_extreme_waves(self):
        
        station = os.path.join(self.test_weather_path,"USW00023050.csv")
        weather_files = [os.path.join(self.test_weather_path,"USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
        
        climate_temp_func = lambda years: 0.03 * (years-2020) + 0.1


        obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0), 
                                      use_local=True, run_parallel=False,use_global=True)
        obj.create_scenario("test", 2020, 4, climate_temp_func)

                        
    def test_ipcc_increases_in_temperature_and_frequency(self):
    
        """
        This test verifies that the statistics coming out of the new
        ExtremeTemperatureWaves class roughly converge to the results indicated
        in Figure SPM.6 that the entire analysis method has been based off of.
        
        The complexity of the transformations and regressions required to fit
        NOAA data and then change then shift that data's distributions makes
        it necessary to verify that no major error have been introduced.
        
        """
    
        clim_scen = ClimateScenario()
        station = os.path.join("data_for_testing","USW00023050.csv")
        weather_files = [os.path.join("data_for_testing","USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
        
        num_year = 1
        start_years = [2020,2050]
        
        # should see increases on average of 1.08C and increase in frequency of 
        # 2.4 between the years.
        
        
        random_seed = 54564863
        
        # subtle difference with scenario names in ClimateScenario!
        scenario = 'SSP585'
        
        obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0),
                                          use_local=True,random_seed=random_seed,
                                          include_plots=self.plot_results,
                                          run_parallel=True,use_global=True)
        results_dict = {}
        

        results_start_year_dict = {}
        
        scen_results = {}
        
        for start_year in start_years:
            

            freq_hw = []
            freq_cs = []
            avg_delT_hw = []
            avg_delT_cs = []
            
            clim_scen.calculate_coef(scenario)
            climate_temp_func = clim_scen.climate_temp_func
            results = obj.create_scenario(scenario, start_year, num_year, 
                                          climate_temp_func, num_realization=10)

            for tup,objA in results[start_year].items():
                
                delT_hw = []
                delT_cs = []
                num_hw = 0
                num_cs = 0

                for alt_name, alteration in objA.alterations.items():
                    if alt_name != 'global temperature trend':
                        minval = alteration.min().values[0]
                        maxval = alteration.max().values[0]
    
                        if np.abs(minval) > np.abs(maxval):
                            is_hw = False
                            num_cs += 1
                        else:
                            is_hw = True
                            num_hw += 1
                        
                        if is_hw:
                            delT_hw.append(maxval)
                        else:
                            delT_cs.append(minval)
                            
                avg_delT_hw.append(np.array(delT_hw).mean())
                avg_delT_cs.append(np.array(delT_cs).mean())
                freq_hw.append(num_hw)
                freq_cs.append(num_cs)
            
            scen_results[start_year] = (avg_delT_hw,avg_delT_cs,freq_hw,freq_cs)
            
        # now let's see if the statistics are coming out over many realizations:
        avg_delT_increase_hw = (np.array(scen_results[2050][0]) - np.array(scen_results[2020][0])).mean()
        avg_delT_decrease_cs = (np.array(scen_results[2050][1]) - np.array(scen_results[2020][1])).mean()
        
        freq_increase_hw = (np.array(scen_results[2050][2]) / np.array(scen_results[2020][2])).mean()
        freq_increase_cs = (np.array(scen_results[2050][3]) / np.array(scen_results[2020][3])).mean()
        
        expected_delT = climate_temp_func(2050) - climate_temp_func(2020)
        
        # the temperature change has converged on the expected change in temperature
        self.assertTrue((avg_delT_increase_hw >= expected_delT - 0.2) and
                        (avg_delT_increase_hw <= expected_delT + 0.2))
        
        # cold snaps should not be changing!
        self.assertTrue(avg_delT_decrease_cs < 0.2 and avg_delT_decrease_cs > -0.2)
        
        # frequency should be increasing by a factor close to 2.4 (this is a hybrid between the 
        # frequency incrase of 2 (i.e. 5.6/2.8 since our baseline is 2020
        # (See Figure SPM.6 of the IPCC technical summary)
        # for 10 year events and 2.89 for 50 year events - see the 
        # documentation.)
        self.assertTrue((freq_increase_hw <= 2.45 + 0.45) and (freq_increase_hw >= 2.45 - 0.45))
        
        # cold snap frequency should be close to 1.0
        self.assertTrue((freq_increase_cs <= 1.0 + 0.1) and (freq_increase_cs >= 1.0 - 0.1))
        
        with self.assertRaises(ValueError):
            # make sure that the unit_conversion error is raised 
            obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/1000,0),
                                          use_local=True,random_seed=random_seed,
                                          include_plots=self.plot_results,
                                          run_parallel=True,use_global=True)
            
    def test_extreme_temperature_waves_local_lat_lon(self):
        # target albuquerque, NM
        clim_scen = ClimateScenario(use_global=False,lat=35.0844,lon=106.6504,end_year=2100,
                                    model_guide=self.model_guide,data_folder=self.data_folder,
                                    run_parallel=True,gcm_to_skip=["NorESM2-MM"])
        
        scen_dict = clim_scen.calculate_coef(["SSP585"])
        
        station = os.path.join("data_for_testing","USW00023050.csv")
        weather_files = [os.path.join("data_for_testing","USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
        
        num_year = 1
        
        # should see increases on average of 1.08C and increase in frequency of 
        # 2.4 between the years.
        num_realizations = 10
        
        random_seed = 5450963  # must be < 2**32-1 for numpy
        
        # subtle difference with scenario names in ClimateScenario!
        scenario = 'SSP585'
        
        obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0),
                                          use_local=True,random_seed=random_seed,
                                          include_plots=self.plot_results,
                                          run_parallel=True,use_global=False)
        # LEFT OFF - you need to figure out why the 5% is producing higher mean
        # temperature than the 50%.
        
        # ALSO - You need to figure out why 95% isn't converging. Start with 
        # finding out if the multiplication factors are higher for 50% than 
        # 5%. If they are, then Figure out if the shifts occuring are 
        # accurate or not.
        results0 = obj.create_scenario(scenario, 2014, num_year, 
                                          scen_dict["SSP585"], 
                                          num_realization=num_realizations,
                                          obj_clim=clim_scen,
                                          increase_factor_ci="5%")
        
        real_stats = obj.real_value_stats('heat wave',"SSP585","delT",192)
        ipcc0 = obj.ipcc_fact
        results5 = obj.create_scenario(scenario, 2080, num_year, 
                                          scen_dict["SSP585"], 
                                          num_realization=num_realizations,
                                          obj_clim=clim_scen,
                                          increase_factor_ci="5%")
        ipcc5 = obj.ipcc_fact
        results50 = obj.create_scenario(scenario, 2080, num_year, 
                                          scen_dict["SSP585"], 
                                          num_realization=num_realizations,
                                          obj_clim=clim_scen,
                                          increase_factor_ci="50%")
        ipcc50 = obj.ipcc_fact
        results95 = obj.create_scenario(scenario, 2080, num_year, 
                                          scen_dict["SSP585"], 
                                          num_realization=num_realizations,
                                          obj_clim=clim_scen,
                                          increase_factor_ci="95%")
        ipcc95 = obj.ipcc_fact
        
        def get_mean(res,nr):
            sum_val = 0
            for year,alter_dict in res.items():
                for tup, alter_obj in alter_dict.items():
                    sum_val = sum_val + alter_obj.epwobj.dataframe['Dry Bulb Temperature'].mean()
            return sum_val/nr   
        
        mean5 = get_mean(results5,num_realizations)
        mean50 = get_mean(results50,num_realizations)
        mean95 = get_mean(results95,num_realizations)
        # basic test making sure the multiplying factors have the right monotonic
        # trend on the CI. A more specific test is needed though verifying that
        # actual content is correct.
        self.assertGreater(mean50,mean5)
        self.assertGreater(mean95, mean50)
        
        # Manually entered values that depend on the shift table.
        res_dict = {"5%":[results5, {'freq_mult':1.5,'intensity_shift':4.9}],
                    "50%":[results50, {'freq_mult':1.5,'intensity_shift':4.9}],
                    "95%":[results95, {'freq_mult':1.5,'intensity_shift':4.9}],
                    "0":[results0, {'freq_mult':1.0,'intensity_shift':0.0}]}
        
        
        
        
        num_hw_dict = {}
        for key,tup in res_dict.items():
            num_hw = 0
            Tmax_sum = 0
            for key2, res in tup[0].items():
                for key3, objA in res.items():
                    
                    df_status = objA.status()
                    num_hw_alt = (df_status.diff() == 1).sum().values[0]
                    num_hw += num_hw_alt
                    
                    for key4, alt in objA.alterations.items():
                        maxval = alt.max().values[0]
                        if maxval > 0:
                            Tmax_sum += maxval
                    
            avg_num_hw = num_hw / num_realizations
            avg_Tmax = Tmax_sum / num_hw
            num_hw_dict[key] = [avg_num_hw,Tmax_sum/num_hw]
            
        
                    
        mult5 = num_hw_dict["5%"][0]/num_hw_dict['0'][0]
        mult50 = num_hw_dict["50%"][0]/num_hw_dict['0'][0]
        mult95 = num_hw_dict["95%"][0]/num_hw_dict['0'][0]
        
        breakpoint()
        # These results are not what you expected. The temperature increase 
        # is much too big per heat wave.
        fig,ax = Graphics.plot_realization(results95,"Dry Bulb Temperature",1)
        pass
            
            
        
        
        
            
        

        
if __name__ == "__main__":
    o = unittest.main(Test_ExtremeTemperatureWaves())
