# -*- coding: utf-8 -*-
"""
Created on Wed July 20 11:59:04 2022

@author: tschoste
"""

from mews.data_requests.CMIP6 import CMIP_Data

import os

def main():
    obj = CMIP_Data(lat_desired = 35.0433,
                    lon_desired = -106.6129,
                    year_baseline = 2014,
                    year_desired = 2050,
                    file_path = os.path.abspath(os.getcwd()),
                    model_guide = "Models_Used_Simplified.xlsx",
                    data_folder = "example_data",
                    world_map=True,
                    calculate_error=False,
                    display_logging=False)
    obj.results1(scatter_display=[True,True,True,True,True,True],CI_display=[False,False,False,False,False,False])
    #obj.results2(desired_scenario = "SSP119",resolution = "low")
    

if __name__ == '__main__':
    main()

