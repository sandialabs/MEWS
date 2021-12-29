# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:39:27 2021

This is a set of functions and scripts that quantify heat wave frequency,
duration, and severity based on:
    1. NOAA daily summary data of daily maximum temperature (TMAX) and
       minimum temperature (TMIN).
    2. NOAA climate normals data for 1991 to 2020. The 90th percentile is
       used on both upper and lower bound temperatures to classify heat wave days
       
Heat wave definition.
    A heat wave commences with a day that has temperature above the 90th 
    percentile at any time during the year. The heat wave continues for consecutive days as long as
    either the night time low is above the 90th percentile for lows or the daytime high
    is above the 90th percentile. This provides data for frequency and duration
    of heat waves. The same logic applies for cold snaps but for lows instead of highs.
    
    The severity of heat waves are then quantified by looking at the total degree days
    added in comparison to the average temperature for climate conditions.
    
    The resulting distributions are fit with statistical distributions so that 
    10 year and 50 year events can be estimated and the resulting rates of 
    increase for severity and frequency from IPCC applied to these distributions
    by shifting the statistical parameters.

@author: dlvilla
"""
from mews.events import ExtremeTemperatureWaves
from mews.graphics import Graphics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


        


if __name__ == "__main__":
    
    clim_scen = ClimateScenario()
    station = os.path.join("example_data","USW00023050.csv")
    weather_files = [os.path.join("example_data","USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
    
    
    num_year = 2
    start_year = 2075
    random_seed = 54564863#2938958
    rc_input = {'size':16}
    
    scenarios = clim_scen.df_temp_anomaly["Climate Scenario"].unique()
    
    fig,axl = plt.subplots(len(scenarios),1)
    
    ETW_dict = {}
    
    for scenario,ax in zip(scenarios,axl):
        clim_scen.calculate_coef(scenario)
        climate_temp_func = clim_scen.climate_temp_func
        
        obj = ExtremeTemperatureWaves(station, weather_files, climate_temp_func, num_year, start_year,
                                 use_local=True,random_seed=random_seed,num_realization=1,scenario_name=scenario,
                                 include_plots=False,run_parallel=False)

        Graphics.plot_realization(obj.extreme_results[scenario], "Dry Bulb Temperature", 0,ax=ax, legend_labels=("extreme","normal"),
                                  title=scenario,rc_input=rc_input)
        
        ETW_dict[scenario] = obj
        

    fig.show()
    
    clim_scen.write_coef_to_table()
    
    
    
    # run a 
    

    

    


 
        










