# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:07:57 2021

@author: dlvilla
"""


from numpy.random import default_rng
import pandas as pd

import unittest
import os
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np

import warnings

from mews.weather.climate import ClimateScenario
from mews.events import ExtremeTemperatureWaves
from mews.events.extreme_temperature_waves import fit_exponential_distribution
from mews.graphics.plotting2D import Graphics
import matplotlib.pyplot as plt
import os
import warnings
from copy import deepcopy
from calendar import monthrange
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
            
                        
    # def test_albuquerque_extreme_waves(self):
        
    #     station = os.path.join(self.test_weather_path,"USW00023050.csv")
    #     weather_files = [os.path.join(self.test_weather_path,"USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
        
    #     climate_temp_func = lambda years: 0.03 * (years-2020) + 0.1


    #     obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0), 
    #                                   use_local=True, run_parallel=False,use_global=True)
    #     obj.create_scenario("test", 2020, 4, climate_temp_func)

                        
    # def test_ipcc_increases_in_temperature_and_frequency(self):
    
    #     """
    #     This test verifies that the statistics coming out of the new
    #     ExtremeTemperatureWaves class roughly converge to the results indicated
    #     in Figure SPM.6 that the entire analysis method has been based off of.
        
    #     The complexity of the transformations and regressions required to fit
    #     NOAA data and then change then shift that data's distributions makes
    #     it necessary to verify that no major error have been introduced.
        
    #     """
    
    #     clim_scen = ClimateScenario()
    #     station = os.path.join("data_for_testing","USW00023050.csv")
    #     weather_files = [os.path.join("data_for_testing","USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
        
    #     num_year = 1
    #     start_years = [2020,2050]
        
    #     # should see increases on average of 1.08C and increase in frequency of 
    #     # 2.4 between the years.
        
        
    #     random_seed = 54564863
        
    #     # subtle difference with scenario names in ClimateScenario!
    #     scenario = 'SSP585'
        
    #     obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0),
    #                                       use_local=True,random_seed=random_seed,
    #                                       include_plots=self.plot_results,
    #                                       run_parallel=True,use_global=True)
    #     results_dict = {}
        

    #     results_start_year_dict = {}
        
    #     scen_results = {}
        
    #     for start_year in start_years:
            

    #         freq_hw = []
    #         freq_cs = []
    #         avg_delT_hw = []
    #         avg_delT_cs = []
            
    #         clim_scen.calculate_coef(scenario)
    #         climate_temp_func = clim_scen.climate_temp_func
    #         results = obj.create_scenario(scenario, start_year, num_year, 
    #                                       climate_temp_func, num_realization=10)
            
            

    #         for tup,objA in results[start_year].items():
                
    #             delT_hw = []
    #             delT_cs = []
    #             num_hw = 0
    #             num_cs = 0

    #             for alt_name, alteration in objA.alterations.items():
    #                 if alt_name != 'global temperature trend':
    #                     minval = alteration.min().values[0]
    #                     maxval = alteration.max().values[0]
    
    #                     if np.abs(minval) > np.abs(maxval):
    #                         is_hw = False
    #                         num_cs += 1
    #                     else:
    #                         is_hw = True
    #                         num_hw += 1
                        
    #                     if is_hw:
    #                         delT_hw.append(maxval)
    #                     else:
    #                         delT_cs.append(minval)
                            
    #             avg_delT_hw.append(np.array(delT_hw).mean())
    #             avg_delT_cs.append(np.array(delT_cs).mean())
    #             freq_hw.append(num_hw)
    #             freq_cs.append(num_cs)
            
    #         scen_results[start_year] = (avg_delT_hw,avg_delT_cs,freq_hw,freq_cs)
            
    #     # now let's see if the statistics are coming out over many realizations:
    #     avg_delT_increase_hw = (np.array(scen_results[2050][0]) - np.array(scen_results[2020][0])).mean()
    #     avg_delT_decrease_cs = (np.array(scen_results[2050][1]) - np.array(scen_results[2020][1])).mean()
        
    #     freq_increase_hw = (np.array(scen_results[2050][2]) / np.array(scen_results[2020][2])).mean()
    #     freq_increase_cs = (np.array(scen_results[2050][3]) / np.array(scen_results[2020][3])).mean()
        
    #     expected_delT = climate_temp_func(2050) - climate_temp_func(2020)
        
    #     # the temperature change has converged on the expected change in temperature
    #     self.assertTrue((avg_delT_increase_hw >= expected_delT - 0.2) and
    #                     (avg_delT_increase_hw <= expected_delT + 0.2))
        
    #     # cold snaps should not be changing!
    #     self.assertTrue(avg_delT_decrease_cs < 0.2 and avg_delT_decrease_cs > -0.2)
        
    #     # frequency should be increasing by a factor close to 2.4 (this is a hybrid between the 
    #     # frequency incrase of 2 (i.e. 5.6/2.8 since our baseline is 2020
    #     # (See Figure SPM.6 of the IPCC technical summary)
    #     # for 10 year events and 2.89 for 50 year events - see the 
    #     # documentation.)
    #     self.assertTrue((freq_increase_hw <= 2.45 + 0.45) and (freq_increase_hw >= 2.45 - 0.45))
        
    #     # cold snap frequency should be close to 1.0
    #     self.assertTrue((freq_increase_cs <= 1.0 + 0.1) and (freq_increase_cs >= 1.0 - 0.1))
        
    #     with self.assertRaises(ValueError):
    #         # make sure that the unit_conversion error is raised 
    #         obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/1000,0),
    #                                       use_local=True,random_seed=random_seed,
    #                                       include_plots=self.plot_results,
    #                                       run_parallel=True,use_global=True)
            
        
            
            

        
        
            
    def test_extreme_temperature_waves_local_lat_lon(self):
        """
        This is a very low order convergence check for the new
        use case where use_global==False between IPCC factors
        from Figure SPM.6 and the statistics of MEWs' output
        More extensive convergence checking cannot be in the 
        unit testing but is performed with the same function
        
        self._run_verification_study_of_use_global_False(num_realizations,
            future_year,random_seed,plot_results,scenario)
        
        where the num_realizations are increased (for frequency) but for
        delT assessments, you want to run the num_realizations=10 and
        num_realizations=50 over and over to assess 10 year and 50 year
        heat wave events generated by MEWS. convergence is expected over many 
        samples of such a type.

        Returns
        -------
        None.

        """
        
        num_realizations = 10  
        
        future_year = 2080
        
                
        random_seed = 1455992  # must be < 2**32-1 for numpy
        
        # subtle difference with scenario names in ClimateScenario!
        scenario = 'SSP585'
        
        # You cannot expect convergence here without a whole lot more runs
        # because every month has to converge. The perc-error allowed has to be 
        # very large here as a result. Especially for frequency where 
        # heat wave probabilities are much lower in 2014.
        perc_err_allowed = 9000.0
        perc_err_allowed_delT = 300.0
        
        run_parallel = False
        
        station = os.path.join("data_for_testing","USW00023050.csv")
        weather_files = [os.path.join("data_for_testing","USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
        
        
        metric_dict, meanval, obj, clim_scen, scen_dict  = self._run_verification_study_of_use_global_False(
            num_realizations, 
            future_year, 
            random_seed, 
            self.plot_results, 
            scenario,
            self.model_guide,
            self.data_folder,
            weather_files,
            station,
            run_parallel = run_parallel,
            use_breakpoint=False)
        
        self.assertGreater(meanval["50%"],meanval["5%"])
        self.assertGreater(meanval["95%"], meanval["50%"])
        
        for key,subdict in metric_dict.items():
            for key2, subdict2 in subdict.items():
                if key == "Intensity":
                    for month, ipcc_delT_estimated in subdict2["MEWS statistical value"].items():
                        if not ipcc_delT_estimated is None:
                            ipcc_delT_exact = (subdict2["IPCC actual value"][month][1]+subdict2["IPCC actual value"][month][0])/2
                            self.assertGreaterEqual(perc_err_allowed_delT, 100 * np.abs((ipcc_delT_estimated-ipcc_delT_exact) / ipcc_delT_exact))
                else:
                    for month, sim_result in subdict2["MEWS statistical value"].items():
                        if not sim_result is None:
                            act_result = subdict2["IPCC actual value"][month]
                            self.assertGreaterEqual(perc_err_allowed, 100 * np.abs(sim_result / act_result-1))
                    
        # verify uniqueness and that random seed keeps random processes consistent
        metric_dict_try1, meanval, obj0, clim_scen, scen_dict = self._run_verification_study_of_use_global_False(
            1, 
            future_year, 
            random_seed, 
            self.plot_results, 
            scenario,
            self.model_guide,
            self.data_folder,
            weather_files,
            station)
        
        metric_dict_try2, meanval, obj1, clim_scen, scen_dict = self._run_verification_study_of_use_global_False(
            1, 
            future_year, 
            random_seed, 
            self.plot_results, 
            scenario,
            self.model_guide,
            self.data_folder,
            weather_files,
            station)
        
        self.assertEqual(metric_dict_try1,metric_dict_try2)
        
        # check get_results
        es000 = obj1.get_results(scenario,future_year,"95%")
        
        
        
        
        
        
    
    @staticmethod
    def _run_verification_study_of_use_global_False(num_realizations,future_year,random_seed,plot_results,scenario,
                                                    model_guide,data_folder,weather_files,station,print_progress=False,run_parallel=True,
                                                    number_cores=20,write_results=False,scen_dict=None,clim_scen=None,obj=None,
                                                    ci_interval=["5%","50%","95%"],use_breakpoint=False):
        

        
        # this provides a template for the new use case where use_global = False. # target albuquerque, NM
        
        if clim_scen is None:
            clim_scen = ClimateScenario(use_global=False,lat=35.0844,lon=106.6504,end_year=2100,
                                        model_guide=model_guide,data_folder=data_folder,
                                        run_parallel=run_parallel,gcm_to_skip=["NorESM2-MM"],num_cpu=number_cores)
            scen_dict = clim_scen.calculate_coef([scenario])
        
        if print_progress:
            print("Finished climate scenario calculations")
        
        num_year = 1
        
        hour_in_day = 24
        # neglect leap years
        hours_in_10_years = 10 * 365 * hour_in_day  # hours in 10 years
        hours_in_50_years = 5 * hours_in_10_years
        # switch to the paper notation
        N10 = hours_in_10_years
        N50 = hours_in_50_years
        
        ipcc = {}
        real_stats = {}
        prob_state = {}
        
        def get_mean(res,nr):
            sum_val = 0
            for year,alter_dict in res.items():
                for tup, alter_obj in alter_dict.items():
                    sum_val = sum_val + alter_obj.epwobj.dataframe['Dry Bulb Temperature'].mean()
            return sum_val/nr  
        
        if obj is None:
            obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0),
                                              use_local=True,random_seed=random_seed,
                                              include_plots=plot_results,
                                              run_parallel=run_parallel,use_global=False,delT_ipcc_min_frac=1.0,
                                              num_cpu=number_cores,write_results=write_results,test_markov=True)
        
        ipcc = {}
        real_stats = {}
        prob_state = {}
        res_dict = {}
        mean = {}
        vdata = {}
        
        for ci_int in ci_interval:
            id0 = "0_"+ci_int
            
            res_dict[id0] = obj.create_scenario(scenario, 2014, num_year, 
                                          scen_dict[scenario], 
                                          num_realization=num_realizations,
                                          obj_clim=clim_scen,
                                          increase_factor_ci=ci_int)
            ipcc[id0]= deepcopy(obj.ipcc_results)

            
            prob_state[id0] = obj._objDM  

            # Get temperature delT's in degrees centigrade

            real_stats[id0] = obj._real_value_stats('heat wave',scenario,2014,ci_int,"delT",np.array([192]))
        


            res_dict[ci_int] = obj.create_scenario(scenario, future_year, num_year, 
                                          scen_dict[scenario], 
                                          num_realization=num_realizations,
                                          obj_clim=clim_scen,
                                          increase_factor_ci=ci_int)
        
            ipcc[ci_int] = deepcopy(obj.ipcc_results)
            
            prob_state[ci_int] = obj._objDM
            # Get temperature delT's in degrees centigrade
            real_stats[ci_int] = obj._real_value_stats('heat wave',scenario,future_year,ci_int,"delT",np.array([192]))
            
            vdata[ci_int] = obj._verification_data
        
            mean[ci_int] = get_mean(res_dict[ci_int],num_realizations)
        

        # basic test making sure the multiplying factors have the right monotonic
        # trend on the CI. A more specific test is needed though verifying that
        # actual content is correct.
        
        #num_hw_dict = Test_ExtremeTemperatureWaves()._get_hw_statistics(res_dict,num_realizations)
        
        
        # Create metrics needed.
        metric_dict = {"Frequency":{},"Intensity":{}}
        

        temp_diff = {}
        for key,tup in res_dict.items(): # confidence interval loop [5%,50%,95%]
            if key[0] != '0':  

                ipcc_delT_exact = {}
                freq_result_act = {}
                freq_result_sim = {}
                
                for month in range(1,13):  # month loop

                    mean_val = []
                    max_val = []
                    
                    # get durations of extreme events.
                    #D10 = ipcc[key]['durations'][month][0]
                    fut_D10_prime = ipcc[key]['durations'][month][1]
                    #D50 = ipcc[key]['durations'][month][2]
                    fut_D50_prime = ipcc[key]['durations'][month][3]
                    #D10 = ipcc['0_'+key]['durations'][month][0]
                    base_D10_prime = ipcc['0_'+key]['durations'][month][1]
                    #D50 = ipcc['0_'+key]['durations'][month][2]
                    base_D50_prime = ipcc['0_'+key]['durations'][month][3]

                    # get heat wave duration data
                    
                    # future year 
                    fut_hws_arr = np.array([])
                    for realization in range(num_realizations):
                        # realization heat wave sustained array
                        try:
                            hws_arr0 = np.concatenate([np.array(li["heat wave duration"]) for 
                                                       li in vdata[key][2080]['freq_s'] if li["month"]==month 
                                                       and li["key_name"][1]==realization])
                        except ValueError:
                            hws_arr0 = np.array([])
                        fut_hws_arr = np.concatenate([fut_hws_arr,hws_arr0])

                    fut_num_10_events = (fut_hws_arr > fut_D10_prime).sum()
                    fut_num_50_events = (fut_hws_arr > fut_D50_prime).sum()
                    
                    # base year 
                    base_hws_arr = np.array([])
                    for realization in range(num_realizations):
                        # realization heat wave sustained array
                        try:
                            hws_arr0 = np.concatenate([np.array(li["heat wave duration"]) for 
                                                       li in vdata[key][2014]['freq_s'] if li["month"]==month 
                                                       and li["key_name"][1]==realization])
                        except ValueError:
                            hws_arr0 = np.array([])
                        base_hws_arr = np.concatenate([base_hws_arr,hws_arr0])
                    
                    base_num_10_events = (base_hws_arr > base_D10_prime).sum()
                    base_num_50_events = (base_hws_arr > base_D50_prime).sum()
                    

                    # get ipcc factors.  
                    base_freq_incr_10 = ipcc['0_'+key]['ipcc_fact'][1].loc[key+" CI Increase in Frequency","10 year event"]
                    base_freq_incr_50 = ipcc['0_'+key]['ipcc_fact'][1].loc[key+" CI Increase in Frequency","50 year event"]
                    
                    fut_freq_incr_10 = ipcc[key]['ipcc_fact'][1].loc[key+" CI Increase in Frequency","10 year event"]
                    fut_freq_incr_50 = ipcc[key]['ipcc_fact'][1].loc[key+" CI Increase in Frequency","50 year event"]

                    for year in [2014,2080]:
                        df_temp = pd.DataFrame(np.array([[val['delT_max_0']+
                                                     val['delTmax']]
                                   for val in vdata[key][year]['delTmax'] if val['s_month']==month]),columns=[
                                           "delTmax original"])

                        mean_val.append(df_temp.mean().values[0])
                        max_val.append(df_temp.max().values[0])
                    temp_diff[month] = {'mean':mean_val[1] - mean_val[0],'max':max_val[1] - max_val[0]}
                    
                                        
                    # original IPCC check 
                    ipcc_delT_exact[month] = (ipcc[key]['ipcc_fact'][month].loc[key+" CI Increase in Intensity","10 year event"]-
                                       ipcc['0_'+key]['ipcc_fact'][month].loc[key+" CI Increase in Intensity","10 year event"],
                                       ipcc[key]['ipcc_fact'][month].loc[key+" CI Increase in Intensity","50 year event"]-
                                       ipcc['0_'+key]['ipcc_fact'][month].loc[key+" CI Increase in Intensity","50 year event"])
                    
                    freq_result_sim[month] = (fut_num_10_events/base_num_10_events, fut_num_50_events/base_num_50_events)
                    freq_result_act[month] = (fut_freq_incr_10/base_freq_incr_10, fut_freq_incr_50/base_freq_incr_50)

                # collect all of the metrics into a dictionary useful for the convergence study.
                metric_dict["Frequency"][key] = {"MEWS statistical value": freq_result_sim,"IPCC actual value":freq_result_act}
                metric_dict["Intensity"][key] = {"MEWS statistical value": temp_diff,"IPCC actual value":ipcc_delT_exact}
                
                    
                    # if num_hw_dict['0_'+key][0]['0_'+key][month] == 0:
                    #     mult[month] = None
                    # else:
                    #     mult[month] = num_hw_dict[key][0][key][month]/num_hw_dict['0_'+key][0]['0_'+key][month]
                    
                    # # this isn't exact. There is some variation month to month  but it is minor variation
                    # # so we don't integrate across the months. 
                    # # base_freq_incr_10 = ipcc['0_'+key][1].loc[key+" CI Increase in Frequency","10 year event"]
                    # # base_freq_incr_50 = ipcc['0_'+key][1].loc[key+" CI Increase in Frequency","50 year event"]
                    
                    # # fut_freq_incr_10 = ipcc[key][1].loc[key+" CI Increase in Frequency","10 year event"]
                    # # fut_freq_incr_50 = ipcc[key][1].loc[key+" CI Increase in Frequency","50 year event"]
            
                    # # This adjustment is needed because we are only assessing times when the possibility of
                    # # a heat wave is the adjusted probability. This excludes times inside a heat wave and
                    # # also excludes times in cold snaps.
                    # hour_in_month = monthrange(future_year,month)[1]*hour_in_day
                    # adj[month] = ((1 - (num_hw_dict[key][2][key][month]+num_hw_dict[key][3][key][month])/hour_in_month)/
                    #       (1 - (num_hw_dict['0_'+key][2]['0_'+key][month]+num_hw_dict['0_'+key][3]['0_'+key][month])/hour_in_month))
                    
                    # # These raw factors have been verified to be correctly passed.
                    # #res_ = (fut_freq_incr_10 * N50 + fut_freq_incr_50 * N10)/(N10 + N50)
                    # #res_0 = (base_freq_incr_10 * N50 + base_freq_incr_50 * N10)/(N10 + N50)
                    
                    # skey = (weather_files[0], 0, future_year)
                    # okey = (weather_files[0], 0, 2014)
                    
                    # act_result[month] = prob_state[key][skey][month]._mat[0,2]/prob_state['0_'+key][okey][month]._mat[0,2]
                        
                    # if mult[month] is None:
                    #     sim_result[month] = None
                    # else:
                    #     sim_result[month] = mult[month]/adj[month]
    
                    # #print("Error between table lookup and \n\n {0:5.2f}".format(100 * np.abs(sim_result / act_result-1)))
                    
                    
                    # # now focus on temperature
                    # if (not num_hw_dict[key][1][key][month] is None) and (not num_hw_dict['0_'+key][1]['0_'+key][month] is None):
                    #     delTmax0 = num_hw_dict['0_'+key][1]['0_'+key][month] #(np.array(num_hw_dict['0_'+key][4]['0_'+key][month])/np.array(num_hw_dict['0_'+key][5]['0_'+key][month]))
                    #     delTmax = num_hw_dict[key][1][key][month]
                    #     # difference in the 10 year event!

                    #     ipcc_delT_estimated[month] = delTmax.max() - delTmax0.max()
                        
                    # else:

                    #     ipcc_delT_estimated[month] = None
                    
                    # # original IPCC check 
                    # ipcc_delT_exact[month] = (ipcc[key][month].loc[key+" CI Increase in Intensity","10 year event"]-
                    #                    ipcc['0_'+key][month].loc[key+" CI Increase in Intensity","10 year event"],
                    #                    ipcc[key][month].loc[key+" CI Increase in Intensity","50 year event"]-
                    #                    ipcc['0_'+key][month].loc[key+" CI Increase in Intensity","50 year event"])
                    
                    


        
        # These results are not what you expected. The temperature increase 
        # is much too big per heat wave.
        if plot_results:
            fig,ax = Graphics.plot_realization(res_dict[ci_interval[0]],"Dry Bulb Temperature",1)

        if use_breakpoint:
            breakpoint()
        return metric_dict, mean, obj, clim_scen, scen_dict
            
    @staticmethod
    def _get_hw_statistics(res_dict,num_realizations):
        # post process to get heat wave statistics between 2014 and 2080
        num_hw_dict = {}
        
        num_hw_d = {}
        Tmax_d = {}
        hw_duration = {}
        num_hr_in_hw_d = {}
        num_hr_in_cs_d = {}
        
        avg_hr_in_hw = {}    
        avg_num_hw = {}
        max_Tmax = {}
        avg_hr_in_cs = {}

        for key, resubdict in res_dict.items():
            #LOOP OVER CI AND 2014 (0_) vs 2080 
            num_hw_d[key] = {}
            Tmax_d[key] = {}
            hw_duration[key] = {}
            num_hr_in_hw_d[key] = {}
            num_hr_in_cs_d[key] = {}  
            
            avg_hr_in_hw[key] = {}    
            avg_num_hw[key] = {}
            max_Tmax[key] = {}
            avg_hr_in_cs[key] = {}
            
            for key2, res in resubdict.items():
                # NULL LOOP - just for 2014 and 2080 that are alone
                for key3, objA in res.items():
                    # LOOP OVER REALIZATIONS
                    if len(num_hw_d[key]) == 0:
                        for month in range(1,13):
                            num_hw_d[key][month] = 0
                            num_hw_d[key][month] = 0
                            Tmax_d[key][month] = []
                            hw_duration[key][month] = []
                            num_hr_in_hw_d[key][month] = 0
                            num_hr_in_cs_d[key][month] = 0    

                    df_status = objA.status()
                    df_status['month'] = objA.reindex_2_datetime().index.month
                    
                    total_hw_hr = (df_status["Status"]==1).sum()
                    
                    for month in range(1,13):
                        num_hr_in_hw_d[key][month] += ((df_status["Status"]==1) & (df_status["month"]==month)).sum()
                        num_hr_in_cs_d[key][month] += ((df_status["Status"]==-1) & (df_status["month"]==month)).sum()
                    
                    
                    for key4, alt in objA.alterations.items():
                        if key4 != "global temperature trend":
                            for month in range(1,13):
                                if month == 12:
                                    next_month = 1
                                else:
                                    next_month = month + 1
                                
                                num_in_month = Test_ExtremeTemperatureWaves().get_hw_hour_count(df_status,alt,month)
                                
                                if num_in_month == 0:
                                    continue
                                else:
                                    num_in_next_month = Test_ExtremeTemperatureWaves().get_hw_hour_count(df_status,alt,next_month)
                                    # then we have a heat wave or cold snap that belongs to this month.    
                                    if alt.sum().values[0] > 0.0:
                                        if num_in_month >= num_in_next_month:
                                            num_hw_d[key][month] += 1
                                            Tmax_d[key][month].append(alt.max().values[0] + objA._unit_test_data[key4][1])
                                            hw_duration[key][month].append(len(alt)/objA._unit_test_data[key4][3])
                                        else:
                                            num_hw_d[key][next_month] += 1
                                            Tmax_d[key][next_month].append(alt.max().values[0] + objA._unit_test_data[key4][1])
                                            hw_duration[key][next_month].append(len(alt)/objA._unit_test_data[key4][3])
                                        
              
                    check_hr_count = np.array([num_hr_in_hw_d[key][mon] for mon in range(1,13)]).sum()

            

            for month in range(1,13):
                
                avg_hr_in_hw[key][month] = num_hr_in_hw_d[key][month] / num_realizations   
                avg_num_hw[key][month] = num_hw_d[key][month] / num_realizations
                if len(Tmax_d[key][month]) > 0:
                    max_Tmax[key][month] = np.array(Tmax_d[key][month])
                else:
                    max_Tmax[key][month] = None
                avg_hr_in_cs[key][month] = num_hr_in_cs_d[key][month] / num_realizations
            
            num_hw_dict[key] = [avg_num_hw,max_Tmax,avg_hr_in_hw,avg_hr_in_cs,Tmax_d,hw_duration]
            
        return num_hw_dict
    
    @staticmethod
    def get_hw_hour_count(df_status,alt,month):

        df_pass_ind = df_status.loc[alt.index,"month"] == month
        num_in_month = len(alt.loc[df_pass_ind])

        return num_in_month
            
        
        
            
        

        
if __name__ == "__main__":
    o = unittest.main(Test_ExtremeTemperatureWaves())
