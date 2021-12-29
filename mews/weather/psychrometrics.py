# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:15:33 2021

@author: dlvilla
"""

import numpy as np

def relative_humidity(dew_point_F, dry_bulb_F):
    def claussius_clapeyron(Temp_C):
        kelvin_offset = 273.15
        latent_heat = 2.453e6 #J/kg
        moist_air_gas_constant = 461 # J/kg
        
        T_kelvin = Temp_C + kelvin_offset
        
        # outputs units in hectopascal
        vapor_pressure = 6.11 *np.exp((latent_heat / moist_air_gas_constant)*(1/kelvin_offset - 1/T_kelvin))
        
        return vapor_pressure
    
    # convert from Fahrenheit to Celcius
    dew_point_C = 5/9 * (dew_point_F - 32.0)
    dry_bulb_C = 5/9 * (dry_bulb_F - 32.0)
    
    vp_sat = claussius_clapeyron(dry_bulb_C)
    vp = claussius_clapeyron(dew_point_C)
    
    relative_humidity = 100.0 * vp / vp_sat
    
    return relative_humidity