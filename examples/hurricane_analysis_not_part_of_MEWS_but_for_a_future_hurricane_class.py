# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:40:04 2022

@author: dlvilla
"""

import pandas as pd
import os 
import math
import numpy as np
from scipy.stats import kstest, expon
from scipy.optimize import curve_fit

# from https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude#:~:text=You%20can%20use%20Uber%27s%20H3,The%20default%20unit%20is%20Km.
def latlon_distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d
# see for columns https://journals.ametsoc.org/view/journals/mwre/141/10/mwr-d-12-00254.1.xml

record_identifiers = {"L":"Landfall (center of system crossing a coastline)",
                      "W":"Peak maximum sustained wind speed",
                      "P":"Minimum in central pressure",
                      "I":"An intensity peak in terms of both pressure and wind",
                      "C":"Closest approach to a cost not followed by a landfall",
                      "S":"Change of status of the system",
                      "G":"Genesis",
                      "T":{"help":"Provides additional detail on the track "+
                           "(position) of the cyclone; spaces 20-21, before "+
                           "third comma--status of system options are as follows",
                           "TD":"Tropical cyclone of tropical depression intensity (<34kt)",
                           "TS":"Tropical cyclone of tropical storm intensity",
                           "HU":"Tropical cyclone of hurricane intensity",
                           "EX":"Extratropical cyclone (of any intensity)",
                           "SD":"Subtropical cyclone of subtropical storm intensity (>=34kt)",
                           "SS":"Subtropical clcylone of subtropical storm intensity (>=34kt)",
                           "LO":"A low that is neither a tropical cyclone, a subtropical cyclone, nor an extratropical cyclone (of any intensity)",
                           "WV":"Tropical wave (of any intensity)",
                           "DB":"Disturbance (of any intensity)"}}

columns = ["date","time","record identifier", "additional detail", "Latitude", 
           "Longitude","Max Wind Speed (knots)","Minimum Pressure (millibars)",
           "34 knot wind radii maximum extent in northeastern quadrant (miles)",
           "34 knot wind radii maximum extent in southeastern quadrant (miles)",
           "34 knot wind radii maximum extent in southwestern quadrant (miles)",
           "34 knot wind radii maximum extent in northwestern quadrant (miles)",
           "50 knot wind radii maximum extent in northeastern quadrant (miles)",
           "50 knot wind radii maximum extent in southeastern quadrant (miles)",
           "50 knot wind radii maximum extent in southwestern quadrant (miles)",
           "50 knot wind radii maximum extent in northwestern quadrant (miles)",
           "64 knot wind radii maximum extent in northeastern quadrant (miles)",
           "64 knot wind radii maximum extent in southeastern quadrant (miles)",
           "64 knot wind radii maximum extent in southwestern quadrant (miles)",
           "64 knot wind radii maximum extent in northwestern quadrant (miles)"]

data = pd.read_csv(os.path.join("..","InputData","weather","hurricanes","hurdat2-1851-100522_with_column_names_added.txt"))

pr_lat_lons = {"Aguadilla":[18.50,-67.13],
            "San Juan":[18.43,-65.99],
            "Penuelas":[17.97,-66.77]}

roman_et_al_data = {"Aguadilla":87,
            "San Juan":111,
            "Penuelas":68}

start_row_1950 = 22685
start_row_1982 = 36508
years_in_record = 40
pr_radius = 200 # km
avg_day_per_year = 365.25


distance = {}
for key,latlon in pr_lat_lons.items():
    
    distance[key] = {}
    storm_num = 0
    is_first = True
    dist_list = []
    for row_num, row in data.loc[start_row_1982:].iterrows():

        if row["date"][0] != "A":
            # land fall or closest approach to a coast point
            if "L" in row['record identifier'] or "C" in row['record identifier']: 
                if "W" in row['Longitude']:
                    lon = -float(row['Longitude'].split("W")[0])
                else:
                    lon = float(row['Longitude'].split("E")[0])
                temp_dist = latlon_distance(latlon, 
                                 [float(row[' "Latitude"'].split("N")[0]),
                                  lon])
                if temp_dist < pr_radius:
                    dist_list.append([temp_dist,row["Max Wind Speed (knots)"],row['date']])
        else:
            if is_first:
                storm_name = row['time'] + "_" + row['date']
            if "MARIA" in storm_name:
                pass
                #breakpoint()
            if not is_first and len(dist_list) > 0:
                distance[key][storm_name + "_" + str(storm_num)] = np.array(dist_list)
            is_first = False
            dist_list = []
            storm_num += 1
            if not is_first:
                storm_name = row['time'] + "_" + row['date']

# now analyze severity
Maria_ws = 135

time_between_cyclones = {}
ws = {}
lamb_dist = {}
estimated_average_power_outage_duration = {}
for key,dist in distance.items():
    wind_speeds = []
    time_between_cyclones[key] = avg_day_per_year*years_in_record/(len(dist))
    for name, arr in dist.items():
        wind_speeds.append(np.array([float(ar) for ar in arr[:,1]]).mean())
    ws[key] = np.array(wind_speeds)
    hist = np.histogram(ws[key])
    cdf = hist[0].cumsum()/hist[0].sum()
    cdf_x = (hist[1][1:] + hist[1][0:-1])/2
    
    fit = curve_fit(expon.cdf,cdf_x,cdf,np.array([0.0,100]),bounds=[(0.0,10.0),(0.0000001,1000)],full_output=True)
# see whether wind speeds follow an exponential distribution
    
    lamb = fit[0][1]
    
    result = kstest(ws[key],expon.cdf,args=(0,lamb))

    lamb_dist[key] = fit[0][1],result.pvalue
    pass
    estimated_average_power_outage_duration[key] = roman_et_al_data[key] * (fit[0][1] / Maria_ws)
    
    