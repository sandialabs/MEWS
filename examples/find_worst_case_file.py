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

epw_file_directory = os.path.join(os.path.dirname(__file__),"example_data","ClimateZone5A_Chicago","mews_epw_results")
temp_name = 'Dry Bulb Temperature'



# this finds the year with the absolute maximum peak temperature
file_list = os.listdir(epw_file_directory)

temp_dict = {}


for file in file_list:

    epw_obj = epw()
    if ".epw" == file[-4:]:        
        epw_obj.read(os.path.join(epw_file_directory,file))
        
        # gather statistics on each file
        df = epw_obj.dataframe[temp_name]
        
        temp_dict[file] = np.array([df.max(),df.min(),df.sum(),df.mean(),df.std(),df.skew(),df.kurtosis()])


df_out = pd.DataFrame(temp_dict,index=["max","min","sum","mean","std",'skew','kurtosis']).T
df_out.to_csv(os.path.join(epw_file_directory,"..","epw_file_stats_summary.csv"))