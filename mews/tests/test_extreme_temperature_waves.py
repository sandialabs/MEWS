# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:07:57 2021

@author: dlvilla

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


from numpy.random import default_rng

import unittest
import os
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from shutil import rmtree

import warnings


from mews.weather.climate import ClimateScenario
from mews.events import ExtremeTemperatureWaves
from mews.graphics.plotting2D import Graphics


from copy import deepcopy
rng = default_rng()

class Test_ExtremeTemperatureWaves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.run_all_tests = True
        cls.plot_results = False
        cls.write_results = False
        cls.run_parallel = True
        cls.rng = default_rng()

        try:
            os.removedirs("mews_results")
            os.removedirs("temp_out")
        except:
            warnings.warn(
                "The testing could not remove the temporary directory ./mews/tests/mews_results or ./mews/tests/temp_out")

        proxy_location = os.path.join("..", "..", "..", "proxy.txt")

        if os.path.exists(proxy_location):
            with open(proxy_location, 'r') as f:
                cls.proxy = f.read()
        else:
            warnings.warn("No proxy settings! If you need for proxy settings to be" +
                          " active then you need to place the correct proxy server in " +
                          os.path.abspath(proxy_location) + " for MEWS to download CMIP6 data.")
            cls.proxy = None

        if not os.path.exists("data_for_testing"):
            os.chdir(os.path.join(".", "mews", "tests"))
            cls.from_main_dir = True
        else:
            cls.from_main_dir = False

        cls.test_weather_path = os.path.join(".", "data_for_testing")
        erase_me_file_path = os.path.join(
            cls.test_weather_path, "erase_me_file.epw")
        if os.path.exists(erase_me_file_path):
            try:
                os.remove(erase_me_file_path)
            except:
                pass
            
        cls.test_weather_file_path = os.path.join(".",
                                                  cls.test_weather_path,
                                                  "USA_NM_Santa.Fe.County.Muni.AP.723656_TMY3.epw")

        plt.close('all')
        font = {'size': 16}
        rc('font', **font)

        fpath = os.path.dirname(__file__)
        cls.model_guide = os.path.join(
            fpath, "data_for_testing", "Models_Used_Simplified.xlsx")
        cls.data_folder = os.path.join(
            fpath, "data_for_testing", "CMIP6_Data_Files")

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
                                  + " and the ./mews/tests folder has residual "
                                  + "*.bin, *.epw, *.EXE, or *.DAT files"
                                  + " that need to be removed!")
            
        file_dir = os.path.join(os.path.dirname(__file__))
        
        if os.path.exists(os.path.join(file_dir,"temp_out")):
            rmtree(os.path.join(file_dir,"temp_out"),ignore_errors=False,onerror=None)
        
        for name in os.listdir(file_dir):
            if "_future_month_" in name or "_historic_month_" in name:
                try:
                    os.remove(os.path.join(file_dir,name))
                except:
                    pass
            elif ".log" in name or ".csv" in name or ".txt" in name or ".png" in name:
                try:
                    os.remove(os.path.join(file_dir,name))
                except:
                    pass
                
                
        if os.path.exists(os.path.join(".", "mews_results")):
            for file_name in os.listdir(os.path.join(".", "mews_results")):
                if (".epw" in file_name):
                    try:
                        os.remove(os.path.join("mews_results", file_name))
                    except:
                        warnings.warn("The testing could not clean up files"
                                      + " and the tests folders have residual "
                                      + "*.epw files in ./mews/tests/mews_results"
                                      + " that need to be removed!")
        try:
            os.rmdir("mews_results")
            os.rmdir("temp_out")
        except:
            warnings.warn(
                "The testing could not remove the temporary directory ./mews/tests/mews_results or ./mews/tests/temp_out")
        if hasattr(cls,"from_main_dir"):
            if cls.from_main_dir:
                os.chdir(os.path.join("..", ".."))

    def test_repeated_date(self):
        run_test = self.run_all_tests
        
        if run_test:
            # this test assures MEWS will reject files with a repeated date in them.
     
            station = {'summaries':os.path.join("data_for_testing", "example_with_repeated_date.csv"),
                       'norms':os.path.join("data_for_testing", "repeated_date_norms.csv")}
            
    
            # change this to the appropriate unit conversion (5/9, -(5/9)*32) is for F going to C
            unit_conversion = (5/9, -(5/9)*32)
            unit_conv_norms = (5/9, -(5/9)*32)
    
            # gives consistency in run results
            random_seed = 564489
    
            # plot_results
            plot_results = False
            run_parallel = False
            num_cpu = 30
    
            weather_files = [os.path.join(
                "data_for_testing", "USA_AK_Kodiak.AP.703500_TMY3.epw")]
            
            with self.assertRaises(ValueError):
                # I had to manually add the daily summaries and norms
                ExtremeTemperatureWaves(station, weather_files, unit_conversion=unit_conversion,
                                              use_local=True, random_seed=random_seed,
                                              include_plots=plot_results,
                                              run_parallel=run_parallel, use_global=False, delT_ipcc_min_frac=1.0,
                                              num_cpu=num_cpu, write_results=True, test_markov=False,
                                              norms_unit_conversion=unit_conv_norms)

    def test_albuquerque_extreme_waves(self):
        run_test = self.run_all_tests
        if run_test:
        
            # OLD TEST
            station = os.path.join(self.test_weather_path,"USW00023050.csv")
            weather_files = [os.path.join(self.test_weather_path,"USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
            
            climate_temp_func = {}
            
            climate_temp_func["test"] = lambda years: 0.03 * (years-2020) + 0.1
    
            obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0),
                                          use_local=True, run_parallel=False,use_global=True)
            obj.create_scenario("test", 2020, climate_temp_func)

    def test_ipcc_increases_in_temperature_and_frequency(self):

        """
        This test verifies that the statistics coming out of the new
        ExtremeTemperatureWaves class roughly converge to the results indicated
        in Figure SPM.6 that the entire analysis method has been based off of.

        The complexity of the transformations and regressions required to fit
        NOAA data and then change then shift that data's distributions makes
        it necessary to verify that no major error have been introduced.

        OLD TEST
        """
        run_test = self.run_all_tests
        if run_test:
            
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
                climate_temp_func = {}
                climate_temp_func[scenario] = clim_scen.climate_temp_func
                results = obj.create_scenario(scenario, start_year,
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
    
            expected_delT = climate_temp_func[scenario](2050) - climate_temp_func[scenario](2020)
    
            # the temperature change has converged on the expected change in temperature
            self.assertTrue((avg_delT_increase_hw >= expected_delT - 0.2) and
                            (avg_delT_increase_hw <= expected_delT + 0.2))
    
            # cold snaps should not be changing!
            self.assertTrue(avg_delT_decrease_cs < 0.4 and avg_delT_decrease_cs > -0.4)
    
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

        """
        Solve options:

            
        """
        run_test = self.run_all_tests
        if run_test:
            random_seed = 1455992
            
            # this is setup not to validate but just to exercise the code. Many more steps
            # are ussually needed for a good analysis.
            solve_options = {'historic': {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                          'decay_func_type': {'cs':'exponential_cutoff','hw':'exponential_cutoff'},
                                          'max_iter': 1,
                                          'limit_temperatures': True,
                                          'num_cpu':10,
                                          'plot_results':self.plot_results,
                                          'num_step':20000,
                                          'test_mode':True,
                                          'min_num_waves':10,
                                          'out_path':os.path.join("temp_out","test_output.png")},
                              'future': {'delT_above_shifted_extreme': {'cs': -10, 'hw': 10},
                                        'max_iter': 1,
                                        'limit_temperatures': True,
                                        'num_cpu':10,
                                        'num_step':20000,
                                        'plot_results':self.plot_results,
                                        'test_mode':True,
                                        'min_num_waves':10,
                                        'out_path':os.path.join("temp_out","future_test_output.png")}}  # extra_output_columns must be altered later.
    
            num_realizations = 1
    
            future_year = 2080
    
            # must be < 2**32-1 for numpy
    
            # subtle difference with scenario names in ClimateScenario!
            scenario = 'SSP585'
    
            run_parallel = True
    
            station = os.path.join("data_for_testing", "USW00023050.csv")
            weather_files = [os.path.join(
                "data_for_testing", "USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
    
            metric_dict, meanval, obj, clim_scen, scen_dict = self._run_verification_study_of_use_global_False(
                num_realizations,
                future_year,
                random_seed,
                self.plot_results,
                scenario,
                self.model_guide,
                self.data_folder,
                weather_files,
                station,
                run_parallel=run_parallel,
                use_breakpoint=False,
                solve_options=solve_options)
        

    @staticmethod
    def _run_verification_study_of_use_global_False(num_realizations, future_year, random_seed, plot_results, scenario,
                                                    model_guide, data_folder, weather_files, station, print_progress=False, run_parallel=True,
                                                    number_cores=20, write_results=False, scen_dict=None, clim_scen=None, obj=None,
                                                    ci_interval=["5%", "50%", "95%"], use_breakpoint=False, solve_options=None):

        # this provides a template for the new use case where use_global = False. # target albuquerque, NM

        if clim_scen is None:
            clim_scen = ClimateScenario(use_global=False, lat=35.0844, lon=106.6504, end_year=2100,
                                        model_guide=model_guide, data_folder=data_folder,
                                        run_parallel=run_parallel, gcm_to_skip=["NorESM2-MM"], num_cpu=number_cores)
            scen_dict = clim_scen.calculate_coef(['historical',scenario])

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

        def get_mean(res, nr):
            sum_val = 0
            for year, alter_dict in res.items():
                for tup, alter_obj in alter_dict.items():
                    sum_val = sum_val + \
                        alter_obj.epwobj.dataframe['Dry Bulb Temperature'].mean(
                        )
            return sum_val/nr

        if obj is None:
            obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10, 0),
                                          use_local=True, random_seed=random_seed,
                                          include_plots=plot_results,
                                          run_parallel=run_parallel, use_global=False, delT_ipcc_min_frac=1.0,
                                          num_cpu=number_cores, write_results=write_results, test_markov=True,
                                          solve_options=solve_options)

        ipcc = {}
        real_stats = {}
        prob_state = {}
        res_dict = {}
        mean = {}
        vdata = {}

        for ci_int in ci_interval:
            id0 = "0_"+ci_int

            res_dict[id0] = obj.create_scenario(scenario, 2014,
                                                scen_dict,
                                                num_realization=num_realizations,
                                                climate_baseyear=2014,
                                                increase_factor_ci=ci_int)
            ipcc[id0] = deepcopy(obj.ipcc_results)

            prob_state[id0] = obj._objDM

            # Get temperature delT's in degrees centigrade

            real_stats[id0] = obj._real_value_stats(
                'heat wave', scenario, 2014, ci_int, "delT", np.array([192]))

            res_dict[ci_int] = obj.create_scenario(scenario, future_year,
                                                   scen_dict,
                                                   num_realization=num_realizations,
                                                   climate_baseyear=2014,
                                                   increase_factor_ci=ci_int)

            ipcc[ci_int] = deepcopy(obj.ipcc_results)

            prob_state[ci_int] = obj._objDM
            # Get temperature delT's in degrees centigrade
            real_stats[ci_int] = obj._real_value_stats(
                'heat wave', scenario, future_year, ci_int, "delT", np.array([192]))

            vdata[ci_int] = obj._verification_data

            mean[ci_int] = get_mean(res_dict[ci_int], num_realizations)

        metric_dict = None

        # These results are not what you expected. The temperature increase
        # is much too big per heat wave.
        if plot_results:
            fig, ax = Graphics.plot_realization(
                res_dict[ci_interval[0]], "Dry Bulb Temperature", 1)

        if use_breakpoint:
            breakpoint()
        return metric_dict, mean, obj, clim_scen, scen_dict

    @staticmethod
    def _get_hw_statistics(res_dict, num_realizations):
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
            # LOOP OVER CI AND 2014 (0_) vs 2080
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
                        for month in range(1, 13):
                            num_hw_d[key][month] = 0
                            num_hw_d[key][month] = 0
                            Tmax_d[key][month] = []
                            hw_duration[key][month] = []
                            num_hr_in_hw_d[key][month] = 0
                            num_hr_in_cs_d[key][month] = 0

                    df_status = objA.status()
                    df_status['month'] = objA.reindex_2_datetime().index.month

                    total_hw_hr = (df_status["Status"] == 1).sum()

                    for month in range(1, 13):
                        num_hr_in_hw_d[key][month] += (
                            (df_status["Status"] == 1) & (df_status["month"] == month)).sum()
                        num_hr_in_cs_d[key][month] += (
                            (df_status["Status"] == -1) & (df_status["month"] == month)).sum()

                    for key4, alt in objA.alterations.items():
                        if key4 != "global temperature trend":
                            for month in range(1, 13):
                                if month == 12:
                                    next_month = 1
                                else:
                                    next_month = month + 1

                                num_in_month = Test_ExtremeTemperatureWaves(
                                ).get_hw_hour_count(df_status, alt, month)

                                if num_in_month == 0:
                                    continue
                                else:
                                    num_in_next_month = Test_ExtremeTemperatureWaves(
                                    ).get_hw_hour_count(df_status, alt, next_month)
                                    # then we have a heat wave or cold snap that belongs to this month.
                                    if alt.sum().values[0] > 0.0:
                                        if num_in_month >= num_in_next_month:
                                            num_hw_d[key][month] += 1
                                            Tmax_d[key][month].append(
                                                alt.max().values[0] + objA._unit_test_data[key4][1])
                                            hw_duration[key][month].append(
                                                len(alt)/objA._unit_test_data[key4][3])
                                        else:
                                            num_hw_d[key][next_month] += 1
                                            Tmax_d[key][next_month].append(
                                                alt.max().values[0] + objA._unit_test_data[key4][1])
                                            hw_duration[key][next_month].append(
                                                len(alt)/objA._unit_test_data[key4][3])

                    check_hr_count = np.array(
                        [num_hr_in_hw_d[key][mon] for mon in range(1, 13)]).sum()

            for month in range(1, 13):

                avg_hr_in_hw[key][month] = num_hr_in_hw_d[key][month] / \
                    num_realizations
                avg_num_hw[key][month] = num_hw_d[key][month] / \
                    num_realizations
                if len(Tmax_d[key][month]) > 0:
                    max_Tmax[key][month] = np.array(Tmax_d[key][month])
                else:
                    max_Tmax[key][month] = None
                avg_hr_in_cs[key][month] = num_hr_in_cs_d[key][month] / \
                    num_realizations

            num_hw_dict[key] = [avg_num_hw, max_Tmax,
                                avg_hr_in_hw, avg_hr_in_cs, Tmax_d, hw_duration]

        return num_hw_dict

    def test_read_and_write_solution_file(self):
        run_test = self.run_all_tests
        if run_test:
            
            solve_options = {'historic':{'num_step':20000,
                                         'test_mode':True},
                             'future':{'num_step':20000,
                                       'test_mode':True}}
            
            # worcester, MA climate function solution.
            scen_dict = {'historical': np.poly1d([1.35889778e-10, 7.56034740e-08, 1.55701410e-05, 1.51736807e-03,
                                                  7.20591313e-02, 4.26377339e-06]),
                         'SSP245': np.poly1d([-0.00019697,  0.04967771, -0.09146572]),
                         'SSP370': np.poly1d([0.00017505,  0.03762937, -0.07095332]),
                         'SSP585': np.poly1d([0.00027245, 0.05231527, 0.0031547])}
            #use the old analysis method because it is fast
            station = os.path.join(self.test_weather_path,"USW00023050.csv")
            weather_files = [os.path.join(self.test_weather_path,"USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
    
            obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0),
                                          use_local=True, run_parallel=True,use_global=True,
                                          write_results=False,solve_options=solve_options,test_markov=True)
            if not os.path.exists("temp_out"):
                os.mkdir("temp_out")
                
            obj.write_solution(os.path.join("temp_out","test_solution_file.txt"))
            
            new_stats = obj.read_solution(os.path.join("temp_out","test_solution_file.txt"))
            
            
            obj2 = ExtremeTemperatureWaves(station, weather_files, unit_conversion=(1/10,0),
                                          use_local=True, run_parallel=True,use_global=False,
                                          solution_file=os.path.join("temp_out","test_solution_file.txt"),
                                          write_results=False,solve_options=solve_options,test_markov=True)
            
            for wt in ['cold snap','heat wave']:
                for month in np.arange(1,13):
                    new_stats[wt][month]['decay function'] = "exponential"
                    new_stats[wt][month]['decay function coef'] = np.array([0.0005])
    
                    
            obj2.stats = new_stats 
            
            obj2.write_solution(os.path.join("temp_out","test_solution_file2.txt"))

            obj2.create_scenario('SSP585',2080,scen_dict,1,2014,
                                 solution_file=os.path.join("temp_out","test_solution_file2.txt"))
            
    def test_create_solutions(self):
        run_test = self.run_all_tests
        if run_test:
            inp_dir = os.path.join(os.path.dirname(__file__),"data_for_testing")
            """
            INPUT DATA
            
            adjust all of these to desired values if analyzing a new location. Paths have
            to be set and folders created if functioning outside of the MEWS repository 
            structure.
            
            
            
            """
            # STEP 1 to using MEWS, create a climate increase in surface temperature
            #        set of scenarios,
            #
            # you must download https://osf.io/ts9e8/files/osfstorage or else
            # endure the process of all the CMIP 6 files downloading (I had to restart ~100 times)
            # with proper proxy settings.
    
            # HERE CS stands for climate scenario (not cold snap)
            future_years = [2063] 
    
            # CI interval valid = ['5%','50%','95%']
            ci_intervals = ["95%"]
            
            lat = 42.268
            lon = 360-71.8763
    
            # Station - Worcester Regional Airport
            station = os.path.join(inp_dir, "USW00094746.csv")
            
            historical_solution = os.path.join(inp_dir,"worcester_historical_solution.txt")
    
            # change this to the appropriate unit conversion (5/9, -(5/9)*32) is for F going to C
            unit_conversion = (5/9, -(5/9)*32)
            unit_conv_norms = (5/9, -(5/9)*32)
    
            # gives consistency in run results
            random_seed = 7293821
    
            num_realization_per_scenario = 10
    
            # plot_results
            plot_results = True
            run_parallel = True
            num_cpu = 20
    
            # No need to input "historical"
            # valid names for scenarios are: ["historical","SSP119","SSP126","SSP245","SSP370","SSP585"]
            scenarios = ['SSP370'] 
    
            weather_files = [os.path.join(inp_dir, "USA_MA_Worcester.Rgnl.AP.725095_TMY3.epw")]
            
            scen_dict = {'historical': np.poly1d([1.35889778e-10, 7.56034740e-08, 1.55701410e-05, 1.51736807e-03,
                                                  7.20591313e-02, 4.26377339e-06]),
                         'SSP245': np.poly1d([-0.00019697,  0.04967771, -0.09146572]),
                         'SSP370': np.poly1d([0.00017505,  0.03762937, -0.07095332]),
                         'SSP585': np.poly1d([0.00027245, 0.05231527, 0.0031547])}
            
            obj = ExtremeTemperatureWaves(station, weather_files, unit_conversion=unit_conversion,
                                          use_local=True, random_seed=random_seed,
                                          include_plots=plot_results,
                                          run_parallel=run_parallel, use_global=False, delT_ipcc_min_frac=1.0,
                                          num_cpu=num_cpu, write_results=True, test_markov=False,
                                          solve_options=None,
                                          norms_unit_conversion=unit_conv_norms,
                                          solution_file=historical_solution)
            
            results,filenames = obj.create_solutions(future_years, scenarios, ci_intervals, historical_solution, scen_dict)
            
            self.assertTrue(len(filenames)==1)
            self.assertTrue(os.path.exists(os.path.join(os.path.dirname(historical_solution),filenames[0])))
            try:
                os.remove(os.path.join(os.path.dirname(historical_solution),filenames[0]))
            except:
                pass

        else:
            warnings.warn("The test_create_solutions unittest was not run because it is turned off!")


    @staticmethod
    def get_hw_hour_count(df_status, alt, month):

        df_pass_ind = df_status.loc[alt.index, "month"] == month
        num_in_month = len(alt.loc[df_pass_ind])

        return num_in_month


if __name__ == "__main__":
    profile = False
    
    if profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_ExtremeTemperatureWaves())
    

