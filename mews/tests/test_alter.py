# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:55:34 2021

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

@author: dlvilla
"""

import unittest
import os
import numpy as np
import pandas as pd
from mews.weather.alter import Alter
from mews.errors.exceptions import (EPWMissingDataFromFile, EPWFileReadFailure,
                                    EPWRepeatDateError)
from mews.weather.doe2weather import DataFormats
from matplotlib import pyplot as plt
from matplotlib import rc
from datetime import datetime

import warnings

class Test_Alter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot_results = False
        
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
        
        erase_me_file = os.path.join(cls.test_weather_path,"erase_me_file","erase_me_Altered.epw")
        if os.path.exists(erase_me_file):
            os.remove(erase_me_file)
        
        erase_dir = os.path.join(cls.test_weather_path,"erase_me_file")
        if os.path.exists(erase_dir):
            os.removedirs(os.path.join(erase_dir))
            
        plt.close('all')
        font = {'size':16}
        rc('font', **font)

        
    @classmethod
    def tearDownClass(cls):
        if cls.from_main_dir:
            os.chdir(os.path.join("..",".."))
        if not cls.plot_results:
            plt.close('all')
        
        
    def test_leapyear(self):
        # test with a leap year replacement on standard year long files
        wg_obj_leap = Alter(self.test_weather_file_path, 2020, True)
        self.assertTrue(len(wg_obj_leap.epwobj.dataframe) == 8784)
        
        # test with a non-leap year
        wg_obj = Alter(self.test_weather_file_path, 2019, True)
        self.assertTrue(len(wg_obj.epwobj.dataframe) == 8760)     
        
        wg_obj_test_leap_snip = Alter(os.path.join(self.test_weather_path,
                                    "URY_Montevideo.865800_IWEC_Feb28_snippet.epw"),
                                 2024, True)
        
        # assure leap years work for a shorter than year file
        wg_obj_test_snip = Alter(os.path.join(self.test_weather_path,
                                    "URY_Montevideo.865800_IWEC_Feb28_snippet.epw"),
                                 2023, True)
        
        # assure multi-year (spanning 2+ leap years) work correctly
        twenty_four_hours = (len(wg_obj_test_leap_snip.epwobj.dataframe) 
                             - len(wg_obj_test_snip.epwobj.dataframe))
        self.assertTrue(twenty_four_hours == 24)
        num_line = 50154-8
        wg_long = Alter(os.path.join(self.test_weather_path,"Combined5YearWeather.epw"),2020)
        
        wg_long.add_alteration(2023, 10, 1, 3, 20000, -30, 
                               lambda x: 1-np.sin(x*np.pi),
                               alteration_name="Ice Age")
        
        
        df_long = wg_long.epwobj.dataframe
        
        # assure 5 leap year substitutions have occured.
        self.assertTrue(len(df_long) == num_line + 24*2)
        
        
        
        df_long = wg_long.reindex_2_datetime("US/Mountain")
        if self.plot_results:
            fig,ax = plt.subplots(1,1)
            df_long['Dry Bulb Temperature'].plot(ax=ax)
            ax.set_title("Multi-year leap year file test")
            
            
    def _Alter_config_use_exe(self,use_exe,binfilename,year,DST):
    
        if use_exe:
            hour_in_file = 8760+24
            start_datetime = datetime(year,1,1,1,0,0,0)
            tz = "America/Denver"

            obj_doe2 = Alter(binfilename,
                  year,
                  True,
                  isdoe2=True,
                  use_exe=use_exe,
                  doe2_start_datetime=start_datetime,
                  doe2_tz=tz,
                  doe2_dst=DST,
                  doe2_hour_in_file=hour_in_file)
        else:
            obj_doe2 = Alter(binfilename,
                  year,
                  True,
                  isdoe2=True,
                  use_exe=use_exe)
            
        return obj_doe2
    
    def test_doe2(self):
        # This won't work unless you put TXT2BIN.EXE and BIN2TXT.EXE 
        # into their respective positions
        
        # original testing using use_exe = True which is no longer the default
        use_exe_list = [False,True]
        
        binfilename = os.path.join(self.test_weather_path,"NM_Albuquerque_Intl_ArptTMY3.bin")
        years = [2020,2017]
        
        DSTs = [[datetime(years[0],3, 8,2,0,0,0), datetime(years[0],11,1,2,0,0,0)],
                [datetime(years[1],3,12,2,0,0,0), datetime(years[1],11,5,2,0,0,0)]]
        
        exe_path = os.path.join(os.path.dirname(__file__),"../third_party_software")
        exe_exist = os.path.exists(os.path.join(exe_path,"BIN2TXT.EXE")) and os.path.exists(os.path.join(exe_path,"TXT2BIN.EXE"))
        
        # run four tests with leap year and without etc...
        for use_exe in use_exe_list:
            for year,DST in zip(years,DSTs): 
                if use_exe:
                    # test to see if DOE-2 utilities are available.
                    if exe_exist:
                        testDoe2 = True
                    else:
                        testDoe2 = False
                        warnings.warn("The DOE-2 features of MEWS cannot be tested"
                                      +" because the 'third_part_software' folder does not"
                                      +" contain 'BIN2TXT.EXE' and 'TXT2BIN.EXE.' These must"
                                      +" be obtained from James Hirsch and Associates "
                                      +" who can be contacted through www.doe2.com"
                                      +". The utilities are free but require a separate"
                                      +" license agreement.")
                else:
                    testDoe2 = True
                    
                if testDoe2:
    
                    obj_doe2 = self._Alter_config_use_exe(use_exe,binfilename,year,DST)
                        
                    
                    df_pre, df = self._add_cold_hot_and_climate(obj_doe2,'DRY BULB TEMP (DEG F)',year)
                    
                    if self.plot_results:
                        fix, ax = plt.subplots(1,1,figsize=(10,10))
                    
                    self.assertTrue(df['DRY BULB TEMP (DEG F)'].max() > 100)
                    self.assertTrue(df['DRY BULB TEMP (DEG F)'].min() < 5)
                    
                    if self.plot_results:
                        df_pre['DRY BULB TEMP (DEG F)'].plot(ax=ax)
                        df['DRY BULB TEMP (DEG F)'].plot(ax=ax)
                        ax.set_ylabel('Dry bulb temperature $(^{\\circ}F)$')
                        ax.grid("on")
                        ax.set_title("Test DOE2 BIN Albuquerque, NM TMY3 with extreme sinusoidal heat wave, \ncold snap and exponential climate trend added")
                    
                    obj_doe2.write("erase_me.bin",overwrite=True,use_exe=use_exe)
                    
                    self.assertTrue(os.path.exists("erase_me.bin"))
                    
                    # double check we can re-read something we just wrote
                    try:
                        obj_doe2_2 = self._Alter_config_use_exe(use_exe,"erase_me.bin",year,DST)
                    except Exception as e:
                        warnings.warn("The erase_me.bin file we just wrote cannot"+
                                      " be read. The DOE2Weather functions must "+
                                      "have a problem!")
                        raise e

                    os.remove("erase_me.bin")
        

        
        
    def _add_cold_hot_and_climate(self,wg_obj,column,year):
        df_pre = wg_obj.reindex_2_datetime()
        
        start_day = 10
        start_month = 7
        start_hour = 12
        duration_hours = 120
        delta_T = 20
        func = lambda x: np.sin(np.pi*x)
        # add a heat wave 
        
        
        wg_obj.add_alteration(year,start_day,start_month,start_hour, 
                           duration_hours, delta_T,func,
                           column=column)
        # add a cold snap
        start_month = 1
        start_day = 20
        delta_T = -20
        wg_obj.add_alteration(year,start_day,start_month,start_hour, 
                           duration_hours, delta_T,func,
                           column=column)
        
        

        
        # add a climate trend all that matters is intercept to maximum ratio.
        # this will be renormalized to maximum being 1
        func = lambda x: 1.2*np.exp(x)-1.2
        
        start_month = 1
        start_day = 1
        start_hour = 1
        delta_T_per_year = 6
        duration_hours = -1 # indicates to continue to the end of the weather file
        wg_obj.add_alteration(year,start_day,start_month,start_hour, 
                           duration_hours, delta_T_per_year,func,
                           column=column)
        
        df = wg_obj.reindex_2_datetime()
        
        return df_pre,df
        
                
    def test_add_remove(self):
        wg_obj = Alter(self.test_weather_file_path, 2019, True)
        
        df_pre, df = self._add_cold_hot_and_climate(wg_obj,'Dry Bulb Temperature',2019)
        # df_pre = wg_obj.reindex_2_datetime()
        
        # start_day = 10
        # start_month = 7
        # start_hour = 12
        # duration_hours = 120
        # delta_T = 20
        # func = lambda x: np.sin(np.pi*x)
        # # add a heat wave 
        
        
        # wg_obj.add_alteration(2019,start_day,start_month,start_hour, 
        #                    duration_hours, delta_T,func,
        #                    column='Dry Bulb Temperature')
        # # add a cold snap
        # start_month = 1
        # start_day = 20
        # delta_T = -20
        # wg_obj.add_alteration(2019,start_day,start_month,start_hour, 
        #                    duration_hours, delta_T,func,
        #                    column='Dry Bulb Temperature')
        
        

        
        # # add a climate trend all that matters is intercept to maximum ratio.
        # # this will be renormalized to maximum being 1
        # func = lambda x: 1.2*np.exp(x)-1.2
        

        # wg_obj.add_alteration(2019,start_day,start_month,start_hour, 
        #                    duration_hours, delta_T_per_year,func,
        #                    column='Dry Bulb Temperature')
        
        # df = wg_obj.reindex_2_datetime()
        
        if self.plot_results:
            fix, ax = plt.subplots(1,1,figsize=(10,10))
        
        self.assertTrue(df['Dry Bulb Temperature'].max() > 50)
        self.assertTrue(df['Dry Bulb Temperature'].min() < -25)
        
        if self.plot_results:
            df_pre['Dry Bulb Temperature'].plot(ax=ax)
            df['Dry Bulb Temperature'].plot(ax=ax)
            ax.set_ylabel('Dry bulb temperature $(^{\\circ}C)$')
            ax.grid("on")
            ax.set_title("Test Santa Fe, NM TMY3 with extreme sinusoidal heat wave, \ncold snap and exponential climate trend added")
        
        # now take away the cold snap
        wg_obj.remove_alteration("Alteration 2")        
        self.assertTrue(wg_obj.epwobj.dataframe['Dry Bulb Temperature'].min() > -25)
        
        # remove the heat wave
        wg_obj.remove_alteration("Alteration 1")
        self.assertTrue(wg_obj.epwobj.dataframe['Dry Bulb Temperature'].max() < 50)
        
        # remove the climate trend
        wg_obj.remove_alteration("Alteration 3")
        
        # we should have returned to the original weather to numeric precision
        should_be_near_zero = (wg_obj.reindex_2_datetime()['Dry Bulb Temperature'] - df_pre['Dry Bulb Temperature']).sum().sum()
        tol = 1e-8
        self.assertTrue(np.abs(should_be_near_zero) < tol)
        
        start_month = 1
        start_day = 1
        start_hour = 1   
        delta_T = -20
        # test using a list input
        wg_obj.add_alteration(2019,start_day,start_month,start_hour, 
                           5, delta_T,[1,2,3,4,5])
        
    def test_Exceptions(self):
        # test bad weather_file_path:
        with self.assertRaises(FileNotFoundError):
            file_name = r"C:\dsklfjsdlfjkds;lfkj.epw"
            wobj = Alter(file_name)
            
        file_n = self.test_weather_file_path
        
        # test check_type must be boolean
        with self.assertRaises(TypeError):
            wobj = Alter(file_n,None,"True")
        
        # test replace_year must be positive
        with self.assertRaises(ValueError):
            wobj = Alter(file_n,-2001,True)
        
        # test replace year must be an integer
        with self.assertRaises(TypeError):
            wobj = Alter(file_n,"1000",True)        
            
        # test alteration name not present
        with self.assertRaises(ValueError):
            wobj = Alter(file_n,1000,True)
            wobj.add_alteration(1000, 1, 1, 1, 10, 100,alteration_name="HeatWave1")
            wobj.remove_alteration("HeatWave")
            
            
        # test csv file with missing entries    
        with self.assertRaises(EPWMissingDataFromFile):
            file_path = os.path.join(self.test_weather_path,"ShorterThanAYearWeather.epw_messed_up")
            wg_obj = Alter(file_path, 2021, False)
        
        # test binary file that does not read
        with self.assertRaises(EPWFileReadFailure):
            file_path = os.path.join(self.test_weather_path, "DOE2_Must_Fail_LivermoreCA_HeatWave_Scenario2.bin")
            wg_obj = Alter(file_path, 2021, False)
        
        
        wg_obj = Alter(self.test_weather_file_path,2020)
        
        # test negative duration input
        with self.assertRaises(ValueError):
            wg_obj.add_alteration(2020, 1, 1, 1, -10, 1, lambda x:x)
        
        # test shape function incorrect length
        with self.assertRaises(ValueError):
            wg_obj.add_alteration(2020, 1, 1, 1, -1, 1, np.ones(200))
            pass
        
        # test shape function wrong type
        with self.assertRaises(TypeError):
            wg_obj.add_alteration(2020,1,1,1,1,1,"Not a function")
            
        # test try to add the same alteration name twice
        with self.assertRaises(ValueError):
            wg_obj.add_alteration(2020,1,1,1,100,10,alteration_name="DontRepeatNames!")
            wg_obj.add_alteration(2020,1,1,1,100,10,alteration_name="DontRepeatNames!")
            
        # test too short of alteration with zero begin
        with self.assertRaises(ZeroDivisionError):
            wg_obj.add_alteration(2020,1,1,1,1,1,alteration_name="MiniBump")
            
        # test incorrect column name
        with self.assertRaises(ValueError):
            wg_obj.add_alteration(2020,1,1,1,1,1,column='Dry Bulb')
            
        # test alteration out of date range
        with self.assertRaises(Exception):
            wg_obj.add_alteration(2019,10,10,10,100,4)
            
        # test repeat date error
        with self.assertRaises(EPWRepeatDateError):
            wg_repeat = Alter(
                os.path.join(self.test_weather_path,
                             "URY_Montevideo.865800_IWEC_Feb29_included_snippet.epw_REPEAT_DATE"))
            wg_repeat.add_alteration(2012, 29, 2, 18, 10, 10)
        
        # test alterations that starts in bounds but then goes beyond the 
        # length of the file
        with self.assertRaises(pd.errors.OutOfBoundsTimedelta):
            wg_obj.add_alteration(2020,1,1,1,1e6,10)
            
        # assure not a directory error is thrown
        with self.assertRaises(NotADirectoryError):
            wg_obj.write(os.path.join("klsdfjdslfj","myfile.epw"))
        
        # assure 
        with self.assertRaises(FileExistsError):
            erase_me2 = os.path.join(self.test_weather_path,"erase_me2.epw")
            wg_obj.write(erase_me2)
            wg_obj.write(erase_me2,overwrite=True)
            wg_obj.write(erase_me2)
        os.remove(erase_me2)
        
            
            
    def test_str_func(self):
        Alter.__str__("")
        obj = Alter(self.test_weather_file_path, 2020, True)
        obj.add_alteration(2020,31,8,10,100,10)
        mystr = str(obj)

             
            
    def test_all_weather_files(self):
        for file in os.listdir(self.test_weather_path):
            file_path = os.path.join(self.test_weather_path,file)
            if file_path[-4:] == ".epw" and os.path.isfile(file_path):
                wg_obj = Alter(file_path, 2021, False)
                
    def test_read_write(self):
        obj = Alter(self.test_weather_file_path, 2020, True)
        # add a hurricane!
        obj.add_alteration(2020,10,12,1,48,100,column='Wind Speed')
        # now overwrite with a new read and find out if you get the same
        # peak on a new history
        obj.read(os.path.join(self.test_weather_path,'URY_Montevideo.865800_IWEC.epw'),2021)
        
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        obj.reindex_2_datetime()['Wind Speed'].plot(ax=ax)
        
        obj.write(os.path.join(self.test_weather_path,'erase_me.epw'), 
                  overwrite=True, create_dir=False)
        obj.read(os.path.join(self.test_weather_path,'erase_me.epw'))
        obj.write()
        os.remove(os.path.join(self.test_weather_path,'erase_me.epw'))
        os.remove(os.path.join(self.test_weather_path,'erase_me.epw_Altered'))
        erase_me_file = os.path.join(self.test_weather_path,"erase_me_file","erase_me_Altered.epw")
        obj.write(erase_me_file,
                  False,True)
        os.remove(erase_me_file)
        os.removedirs(os.path.join(self.test_weather_path,"erase_me_file"))
        
        obj.read(os.path.join(self.test_weather_path,"URY_Montevideo.865800_IWEC_Feb28_snippet.epw"),
                 replace_year=2020)
        
    def test_future_ability_to_translate_epw_to_doe2(self):
        
        # this just illustrates that a more work can make this a good
        # translator between file types, units and some logic are still needed
        # that are overlooked here.
        
        obj = Alter(self.test_weather_file_path, 2020, True)
        # add a hurricane!
        obj.add_alteration(2020,10,12,1,48,100,column='Wind Speed')
        
        
        # For this to work you have to add headers
        
        # np.dtype([('location_IWDID','a20'),
        #                                     ('year_IWYR', 'i4'),
        #                                     ('latitude_WLAT','f4'),
        #                                     ('longitude_WLONG','f4'),
        #                                     ('timezone_IWTZN','i4'),
        #                                     ('record_length_LRECX','i4'),
        #                                     ('number_days_NUMDAY','i4'),
        #                                     ('clearness_number_CLN_IM1','f4'),
        #                                     ('ground_temperature_GT_IM1','f4'),
        #                                     ('solar_flag_IWSOL','i4')])        
        
        
        clearness_number = np.ones(12)
        
        # TODO - need to produce wetbulb from humidity, dry bulb and others available
        
        # TODO - calculate humidity ratio from relative humidity and dry-bulb or other props,
                 #calculate enthalpy
                 #calculate density of air
                 
                 # calculate correct radiation for DOE2.2
        
        # TODO - tranlsate between SI and IP
        
        
        # THIS HAS ERRORs because the mapping is NOT direct
        translation_map = {'MONTH (1-12)':"Month", 'DAY OF MONTH':"Day",
                          'HOUR OF DAY':"Hour", 'WET BULB TEMP (DEG F)':"Dew Point Temperature", 
                          'DRY BULB TEMP (DEG F)':"Dry Bulb Temperature",
                          'PRESSURE (INCHES OF HG)':"Atmospheric Station Pressure",
                          'CLOUD AMOUNT (0 - 10)':"Total Sky Cover",
                          'SNOW FLAG (1 = SNOWFALL)':"Snow Depth",
                          'RAIN FLAG (1 = RAINFALL)':"Precipitable Water",
                          'WIND DIRECTION (0 - 15; 0=N, 1=NNE, ETC)':"Wind Direction",
                          'HUMIDITY RATIO (LB H2O/LB AIR)':"Relative Humidity", 
                          'DENSITY OF AIR (LB/CU FT)': "Liquid Precipitation Quantity",
                          'SPECIFIC ENTHALPY (BTU/LB)':"Snow Depth", 
                          'TOTAL HOR. SOLAR (BTU/HR-SQFT)': "Horizontal Infrared Radiation Intensity",
                          'DIR. NORMAL SOLAR (BTU/HR-SQFT)':'Direct Normal Radiation', 
                          'CLOUD TYPE (0 - 2)':'Albedo', 
                          'WIND SPEED (KNOTS)':"Wind Speed"}
        
        obj.reindex_2_datetime() 
        
        df = pd.DataFrame(data=None,index=obj.epwobj.dataframe.index)
        df["Date"] = obj.epwobj.dataframe.index
        for key,val in translation_map.items():
            df[key] = obj.epwobj.dataframe[val]
        
        ground_temps = np.ones(12) * 500.0
        description = "This is a test"
        headers = []
        month = 0
        for num,temp in zip(clearness_number,ground_temps):
            header_list = [(description,
                                 np.int32(1901),
                                 np.float32(36.0),
                                 np.float32(-111.0),
                                 np.int32(7.0),
                                 np.int32(1),
                                 np.int32(31),
                                 np.float32(clearness_number[month]),
                                 np.float32(ground_temps[month]),
                                 np.int32(5))]
            lheaders = np.array(header_list,dtype=DataFormats.header_dtype)
 
            headers.append(lheaders)
            month += 1
        
        obj.isdoe2 = True  # switching this 
        
        # Do not allow warning output. I want to add an attribute to the
        # pandas dataframe. I know...I should create my own class to do this
        # correctly. For now, I am just going to do it this way though.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df.headers = headers

        obj.epwobj.dataframe = df
        
        warnings.simplefilter("ignore")
        obj.write(self.test_weather_file_path[:-4] + ".BIN",overwrite=True)
        
if __name__ == "__main__":
    o = unittest.main(Test_Alter())