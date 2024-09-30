#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:54:56 2023

@author: dlvilla
"""

import os
from mews.epw import epw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mews.utilities.utilities import list_epws_stats

epw_file_directory = os.path.join(os.path.dirname(__file__),"example_data","ClimateZone4B_Albuquerque","mews_epw_results_2035_2050")
temp_name = 'Dry Bulb Temperature'
out_file = "epw_file_stats_summary_2035_2050.csv"

rerun = False

if rerun:

    df_out = list_epws_stats(epw_file_directory,temp_name,out_file)

else:

    df_out = pd.read_csv(os.path.join(epw_file_directory,"..",out_file))

# verify there are no repeats
unique = df_out['mean'].unique()

common_list = []
for uniq in unique:
    if len(df_out[df_out['max'] == uniq])>1:
        common_list.append(df_out[df_out['max'] == uniq])

# plot the statistics

    
fig,ax = plt.subplots(1,1,figsize=(5,5))



df_out[["max","min","mean","std","skew","kurtosis"]].hist(bins=20,ax=ax)
plt.tight_layout()
fig.savefig("stats_of_weather_files.png")
