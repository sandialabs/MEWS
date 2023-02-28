# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 12:48:51 2021

See https://towardsdatascience.com/getting-weather-data-in-3-easy-steps-8dc10cc5c859

and 

https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation

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

THIS MODULE IS NOT USED AND IS UNDER CONSTRUCTION

"""
import requests


class NOAA_API():
    base_request_ncdc = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data?"
    base_request_ncei = "https://www.ncei.noaa.gov/access/services/data/v1?"
    
    def __init__(self):
        
        self.ncei_options = {"dataset"}
        
        test = "https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-marine&dataTypes=WIND_DIR,WIND_SPEED&stations=AUCE&startDate=2016-01-01&endDate=2016-01-02&boundingBox=90,-180,-90,180"
        
        r = requests.get(test)
        
        
        pass
    
    
    
if __name__ == "__main__":
    obj = NOAA_API()