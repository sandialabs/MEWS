# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:59:38 2021

@author: dlvilla
"""
from mews.stats import Extremes
from mews.weather.psychrometrics import relative_humidity
from mews.data_requests.CMIP6 import CMIP_Data

from scipy.optimize import fsolve
from scipy.optimize import bisect
from scipy.special import erf

import io
from calendar import monthrange
import pandas as pd
import numpy as np
import os
import shutil
import urllib
from urllib.parse import urlparse

from datetime import datetime, timedelta
from contextlib import closing
import statsmodels.api as sm
from warnings import warn

import matplotlib.pyplot as plt

# TODO make this a class such that erf(a) and erf(b) are not recalculated.
def cdf_truncnorm(x,mu,sig,a,b):
    #https://en.wikipedia.org/wiki/Truncated_normal_distribution
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        xi = (x - mu)/sig
        alpha = (a - mu)/sig
        beta = (b - mu)/sig
        erf_alpha = erf(alpha)
        return (erf(xi) - erf_alpha)/(erf(beta) - erf_alpha)

def offset_cdf_truncnorm(x,mu,sig,a,b,rnd):
    #https://en.wikipedia.org/wiki/Truncated_normal_distribution
    if x < a:
        return 0
    elif x > b:
        return 1
    else:
        xi = (x - mu)/sig
        alpha = (a - mu)/sig
        beta = (b - mu)/sig
        erf_alpha = erf(alpha)
        return (erf(xi) - erf_alpha)/(erf(beta) - erf_alpha) - rnd
    
def trunc_norm_dist(rnd,mu,sig,a,b,minval,maxval):
    # inverse lookup from cdf
    x, r = bisect(offset_cdf_truncnorm, a, b, args=(mu,sig,a,b,rnd),full_output=True)
    if r.converged:
        return inverse_transform_fit(x,maxval,minval)
    else:
        raise ValueError("The bisection method failed to converge! Please investigate")
    
def transform_fit(value,minval,maxval):
    # this function maps a logarithm from 0 to interval making for a good log-normal fit
    return 2 *  (value - minval)/(maxval - minval) - 1
                 
# TODO - merge these functions with those in ExtremeTemperatureWaves
def inverse_transform_fit(norm_signal, signal_max, signal_min):
    return (norm_signal + 1)*(signal_max - signal_min)/2.0 + signal_min 



class ExtremeTemperatureWaves(Extremes):
    
    """
    >>> ExtremeTemperatureWaves(station,
                                weather_files,
                                unit_conversion,
                                num_year,
                                use_local,
                                include_plots)
    
    This initializer reads and processes the heat waves and cold snap statistics
    for a specific NOAA station and corresponding weather data. After instantiation,
    the "create_scenario" method can be used to create weather files.
    
    Parameters
    ----------
    
    station : str
        Must be a valid NOAA station number that has both daily-summary
        and 1991-2020 hourly climate norms data. If use_local=True, then
        this can be the path to a local csv file and <station>_norms.csv
        is expected in the same location for the climate norm data.
    
    weather_files : list
        List of path and file name strings that include all the weather files
        to alter
        
    unit_conversion : tuple
        Index 0 = scale factor for NOAA daily summaries data to convert to Celcius
        Index 1 = offset for NOAA daily summaries to convert to Celcius
        For example if daily summarries are Fahrenheit, then (5/9, -(5/9)*32)
        needs to be input. If Celcius then (1,0). Tenths of Celcius (1/10, 0)
        
    num_year : int
        Number of years to simulate extreme wave events and global climate
        change into the future.
        
    start_year : int
        Year to start the weather files in.
        
    use_local : Bool : Optional: Default = False
        Flag to indicate that that "station" input is actually a path to
        local <station>.csv and <station>_norms.csv files for the NOAA data
        
    include_plots : Bool : Optional : Default = False
        True : plot all kinds of diagnostic information to help determine 
        if heat waves are well characterized statistically by the data. 
        This adds run time but is highly advised for new weather stations
        not previously analyzed.
        
    doe2_input : dict : Optional : Default = None
       | If none - process the run as E+.
       |
       | Optional input required to perform the analysis using DOE2
       | bin files. See mews.weather.alter. needs:
       |     
       | {'doe2_bin2txt_path':OPTIONAL - path to bin2txt.exe DOE2 utility
       |  MEWS has a default location for this utility 
       |  which can be obtained from DOE2 (www.doe2.com),
       | 'doe2_start_datetime':datetime indicating the start date and time
       | for the weather DOE2 weather file,
       | 'doe2_tz'=time zone for doe2 file,
       | 'doe2_hour_in_file'=8760 or 8784 for leap year,
       | 'doe2_dst'= Start and end of daylight savings time for the 
       | doe2_start_datetime year,
       | 'txt2bin_exepath' : OPTIONAL - path to txt2bin.exe DOE2 utility}
    
    results_folder : str : Optional : Default = "mews_results"
        Path to the location where MEWS will write all of the output files
        for the requested analysis. Files will be the original weather file
        name with "_<realization>_<year>" appended to it.
    
    Returns
    -------
    None
    
    """
    
    #  These url's must end with "/" !
    
    # norms are provided in Fahrenheit!
    norms_url  = "https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/"
    # daily data are provided in Celcius!
    daily_url = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/"
    
    def __init__(self,station,
                 weather_files,
                 unit_conversion,
                 use_local=False,
                 include_plots=False,
                 doe2_input=None,
                 results_folder="mews_results",
                 random_seed=None,
                 run_parallel=True):
        
        # consistency checks
        self._check_NOAA_url_validity()
        
        # temperature specific tasks
        
        if include_plots:
            # ! TODO - move all plotting out of this class
            plt.close("all")

        # The year is arbitrary and should simply not be a leap year
        self._read_and_curate_NOAA_data(station,2001,unit_conversion,use_local)
        
        stats = self._wave_stats(self.NOAA_data,include_plots)
        self.stats = stats
        
        if include_plots:
            self._plot_stats_by_month(stats["heat wave"],"Heat Waves")
            self._plot_stats_by_month(stats["cold snap"],"Cold Snaps")
        
        self._results_folder = results_folder
        self._random_seed = random_seed
        self._run_parallel = run_parallel
        self._doe2_input = doe2_input
        self._weather_files = weather_files
        self.ext_obj = {}
        

        self.extreme_results = {}
        
        
    def create_scenario(self,scenario_name,start_year,num_year,climate_temp_func,
                        num_realization=1):
        
        """
        >>> obj.create_scenario(scenario_name,start_year,num_year, climate_temp_func)
        
        Places results into self.extreme_results and self.ext_obj
        
        Parameters
        ----------
        
        scenario_name : str 
            A string indicating the name of a scenario
        
        start_year : int 
            A year (must be >= 2020) that is the starting point of the analysis
                     
        num_year : int 
            Number of subsequent years to include in the analysis
        
        climate_temp_func : func 
            A function that provides a continuous change in temperature that
            is scaled to a yearly time scale. time = 2020 is the begin of the 
            first year that is valid for the function.
            
            No valid return for values beyond 4 C due to lack of data from
            IPCC for higher values.
            
        num_realization : int : Optional : Default = 1
            Number of times to repeat the entire analysis of each weather file 
            so that stochastic analysis can be carried out.
        
        Returns
        -------    
        None
        
        """
        

        
        ext_obj_dict = {}
        results_dict = {}
        
        for year in np.arange(start_year, start_year + num_year,1):
            
            
            
            year_no = year
            (transition_matrix, 
             transition_matrix_delta,
             del_E_dist,
             del_delTmax_dist
             ) = self._create_transition_matrix_dict(self.stats,climate_temp_func,year)
            

            # now initiate use of the Extremes class to unfold the process
            ext_obj_dict[year] = super().__init__(start_year,
                     {'func':trunc_norm_dist, 'param':self.stats['heat wave']},
                     del_delTmax_dist,
                     {'func':trunc_norm_dist, 'param':self.stats['cold snap']},
                     None,
                     transition_matrix,
                     transition_matrix_delta,
                     self._weather_files,
                     num_realizations=num_realization,
                     num_repeat=1,
                     use_cython=True,
                     column='Dry Bulb Temperature',
                     tzname=None,
                     write_results=True,
                     results_folder=self._results_folder,
                     results_append_to_name=scenario_name,
                     run_parallel=self._run_parallel,
                     min_steps=24,
                     test_shape_func=False,
                     doe2_input=self._doe2_input,
                     random_seed=self._random_seed,
                     max_E_dist={'func':trunc_norm_dist,'param':self.stats['heat wave']},
                     del_max_E_dist=del_E_dist,
                     min_E_dist={'func':trunc_norm_dist,'param':self.stats['cold snap']},
                     del_min_E_dist=None,
                     current_year=int(year),
                     climate_temp_func=climate_temp_func,
                     averaging_steps=24)
            results_dict[year] = self.results
        

        self.extreme_results[scenario_name] = results_dict
        
        self.ext_obj[scenario_name] = ext_obj_dict
        
        return results_dict
        


    def _create_transition_matrix_dict(self,stats,climate_temp_func,year):
        
        transition_matrix = {}
        transition_matrix_delta = {}
        del_E_dist = {}
        del_delTmax_dist = {}

        # Form the parameters needed by Extremes but on a monthly basis.
        for hot_tup,cold_tup in zip(stats['heat wave'].items(), stats['cold snap'].items()):
            
            month = cold_tup[0]
            cold_param = cold_tup[1]
            hot_param = hot_tup[1]

            Pwh = hot_param['hourly prob of heat wave']
            Pwsh = hot_param['hourly prob stay in heat wave']
            Pwc = cold_param['hourly prob of heat wave']
            Pwsc = cold_param['hourly prob stay in heat wave']
            transition_matrix[month] = np.array([[1-Pwh-Pwc,Pwc,Pwh],
                                                 [1-Pwsc, Pwsc, 0.0],
                                                 [1-Pwsh, 0.0, Pwsh]])
            # Due to not finding information in IPCC yet, we assume that cold events
            # do not increase in magnitude or frequency.
            obj = DeltaTransition_IPCC_FigureSPM6(hot_param,
                                               cold_param,
                                               climate_temp_func,
                                               year)
            transition_matrix_delta[month] = obj.transition_matrix_delta
            del_E_dist[month] = obj.del_E_dist
            del_delTmax_dist[month] = obj.del_delTmax_dist
             
            
        return (transition_matrix,
                transition_matrix_delta,
                del_E_dist,
                del_delTmax_dist)
        

    def _read_and_curate_NOAA_data(self,station,year,unit_conversion,use_local=False):
    
        """
        read_NOAA_data(station,year=None,use_local=False)
        
        Reads National Oceanic and Atmospheric Association (NOAA)
        daily-summaries data over a long time period and
        NOAA climate norms hourly data from 1991-2020 from the url's
        self.norms_url and self.daily_url. Creates self.NOAA_data dataframe
        
        
        This funciton is highly specific to the data formats that worked on 
        10/14/2021 for the NOAA data repositories and will likely need significant
        updating if new data conventions are used by NOAA.
        
        Parameters
        ----------
        station : str or dict
            str: Must be a valid weather station ID that has a valid representation 
            for both the self.norms_url and self.daily_url web locations
            the error handling provides a list of valid station ID's that
            meets this criterion if a valid ID is not provded.
            
            dict: must be a dictionary of the following form
                {'norms':<str with path to norms file>,
                 'summaries':<str with path to the daily summaries file>}
        
        year : int 
            The year to be assigned to the climate norms data for 
            dataframe purposes. This year is later overlaid accross
            the range of the daily data
            
        unit_conversion : tuple
            a tuple allowing index 0 to be a scale factor on the units in the
            daily summary data and index 1 to be an offset factor. 
        
        use_local 
            If MEWS is not reading web-urls use this to indicate to look
            for local files at "station" file path location. 
            if true 
                    
        Returns
        -------
        NOAA_data : DataFrame 
            Contains the daily summaries with statistical summary
            daily data from the climate normals overlaid so that heat and cold 
            wave comparisons can be made.
        
        """
        df_temp = []
        for url,isdaily in zip([self.daily_url,self.norms_url],[True,False]):
    
            if isinstance(station,dict):
                
                if isdaily:
                    df = pd.read_csv(station['summaries'],low_memory=False)
                else:
                    df = pd.read_csv(station['norms'],low_memory=False)
                
            elif isinstance(station,str):
    
                if station[-4:] == ".csv":
                    ending = ""
                else:
                    ending = ".csv"
            
                if use_local:
                    if isdaily:
                        df = pd.read_csv(station + ending,low_memory=False)
                    else:
                        if len(ending) == 0:
                            station = station[:-4]
                            ending = ".csv"
                        df = pd.read_csv(station + "_norms" + ending)
            
                else:
                    
                    # give a good error if the http and station number are not working.
                    try:
                        filestr = urllib.request.urlopen(urllib.parse.urljoin(url,station+ending)).read().decode()
                    except urllib.error.HTTPError as exc:
                        exc.msg = "The link to \n\n'" + exc.filename + "'\n\ncould not be found."
                        raise(exc)
                    except Exception as exc_unknown:
                        raise(exc_unknown) 
                    
                    df = pd.read_csv(io.StringIO(filestr))
            else:
                raise TypeError("The station input must be a dict with two entries 'summaries', and 'norms' " +
                                "or a string that indicates a valid station ID for NOAA summaries and " +
                                "and norms data")
        
            
            if isdaily:
                # specific processing for daily summary data.
                df['DATE'] = pd.to_datetime(df['DATE'])
                df.index = df['DATE']
        
                # numbers provided are degrees Celcius in tenths.
                df['TMAX'] = unit_conversion[0] * df['TMAX'] + unit_conversion[1]
                df['TMIN'] = unit_conversion[0] * df['TMIN'] + unit_conversion[1]
                meta_data = {'STATION':df['STATION'].iloc[0],
                             'LONGITUDE':df['LONGITUDE'].iloc[0],
                             'LATITUDE':df['LATITUDE'].iloc[0],
                             'ELEVATION':df['ELEVATION'].iloc[0],
                             'NAME':df['NAME'].iloc[0]}
                
                df_new = df[['TMIN','TMAX']]
                self.meta = meta_data
            
            else:
                df.index = pd.to_datetime(df["DATE"].apply(lambda x: str(year)+"-"+x))
                keep = ["HLY-TEMP-10PCTL","HLY-TEMP-NORMAL","HLY-TEMP-90PCTL",
                        "HLY-DEWP-10PCTL","HLY-DEWP-NORMAL","HLY-DEWP-90PCTL",
                        "HLY-PRES-10PCTL","HLY-PRES-NORMAL","HLY-PRES-90PCTL"]
                df = df[keep]
                df["HLY-RELH-10PCTL"] = df[["HLY-TEMP-10PCTL","HLY-DEWP-10PCTL"]].apply(lambda x: relative_humidity(x[1],x[0]),axis=1)
                df["HLY-RELH-NORMAL"] = df[["HLY-TEMP-NORMAL","HLY-DEWP-NORMAL"]].apply(lambda x: relative_humidity(x[1],x[0]),axis=1)
                df["HLY-RELH-90PCTL"] = df[["HLY-TEMP-90PCTL","HLY-DEWP-90PCTL"]].apply(lambda x: relative_humidity(x[1],x[0]),axis=1)
                
                df_max = df.resample('1D').max()
                df_min = df.resample('1D').min()
                df_avg = df.resample('1D').mean()
                
                # now reconstruct the comparisons to 90% TMAX and 10% TMIN needed for hot and cold waves. No other data is needed for now 
                df_new = pd.concat([df_max["HLY-TEMP-90PCTL"],df_avg["HLY-TEMP-NORMAL"],df_min["HLY-TEMP-10PCTL"],df_max["HLY-TEMP-10PCTL"],df_min["HLY-TEMP-90PCTL"],
                                    df_max["HLY-RELH-90PCTL"],df_avg["HLY-RELH-NORMAL"],df_min["HLY-RELH-10PCTL"],df_max["HLY-RELH-10PCTL"],df_min["HLY-RELH-90PCTL"],
                                    df_max["HLY-PRES-90PCTL"],df_avg["HLY-PRES-NORMAL"],df_min["HLY-PRES-10PCTL"],df_max["HLY-PRES-10PCTL"],df_min["HLY-PRES-90PCTL"]],axis=1)
                df_new.columns = ["TMAX_B","TAVG_B","TMIN_B","TMAXMIN_B","TMINMAX_B",
                                  "HMAX_B","HAVG_B","HMIN_B","HMAXMIN_B","HMINMAX_B",
                                  "PMAX_B","PAVG_B","PMIN_B","PMAXMIN_B","PMINMAX_B"]
                # convert from degrees Fahrenheit
                df_new[["TMAX_B","TAVG_B","TMIN_B","TMAXMIN_B","TMINMAX_B"]] = df_new[[
                    "TMAX_B","TAVG_B","TMIN_B","TMAXMIN_B","TMINMAX_B"]].apply(
                        lambda x: (5/9) * (x-32.0))
                        
            df_temp.append(df_new)
        df_daily = df_temp[0]
        df_norms = df_temp[1]
        
        self.NOAA_data = self._extend_boundary_df_to_daily_range(df_norms,df_daily)
    
    def _check_NOAA_url_validity(self):
        def is_valid(url):
            """
            Checks whether `url` is a valid URL.
            """
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        
        err_msg = ("MEWS: Test to "+
                    "see if this url is still valid on an internet connected"+
                    " computer for NOAA and contact them if it is no longer"+
                    " working and update the source code in the "+
                    "ExtremeTemperatureWave.daily_url or .norms_url constants!")
        
        if not is_valid(self.daily_url):
            raise urllib.error.HTTPError(self.daily_url,None,err_msg)
        elif not is_valid(self.norms_url):
            raise urllib.error.HTTPError(self.norms_url,None,err_msg)
            
    def _extend_boundary_df_to_daily_range(self,df_norms,df_daily):
        """
        This function overlays the climate norms 90th percentile and all other columns 
        over the daily data so that heat wave statistics can be quantified. 
        """
        
        df_combined = df_daily
        
        unique_year = df_daily.index.year.unique()
        
        # verify that every year is consecutive
        if np.diff(unique_year).sum() + 1 != len(unique_year):
            pass
            # THIS WAS AN OVERLY RESTRICTIVE TEST
            #raise ValueError("The daily data from NOAA has one or more gaps of a year! Please use a station that has continuous data!")
        
    
        
        feb_28_day = pd.Timestamp(2020,2,28).day_of_year
    
        df_list = []
    
        for year in unique_year:
            # change the df_norms so that it reflects the current year being focussed on 
            df_norms.index = pd.DatetimeIndex([datetime(year,date.month,date.day) for date in df_norms.index])
            
            ind = np.argwhere(df_daily.index.year == year)
            
            # assure the data is complete.
            if np.diff(ind[:,0]).sum() + 1 != len(ind):
                raise ValueError("There is a missing day in year + {0:5d}. This algorithm cannot handle missing days".format(year))
            else:
                b_ind = ind[0]
                e_ind = ind[-1]
                
            first_date = df_daily.index[b_ind]
            end_date = df_daily.index[e_ind]
            
            first_day = first_date.day_of_year[0]-1
            end_day = end_date.day_of_year[0]-1
    
            
            if first_date.is_leap_year and (first_day <= feb_28_day and end_day > feb_28_day):
                # Feb 29 must be added 
                df_list.append(df_norms.iloc[first_day:feb_28_day,:])
                # repeat february 28th as a proxy for February 29th
                df_feb29 = df_norms.iloc[feb_28_day:feb_28_day+1,:]
                df_feb29.index = pd.DatetimeIndex([datetime(year,2,29)])
                df_list.append(df_feb29)
                df_list.append(df_norms.iloc[feb_28_day:end_day+1])
            else:
                # just add the range.
                df_list.append(df_norms.iloc[first_day:end_day+1,:])
    
        df_norms_overlaid = pd.concat(df_list,axis=0)
        df_combined = pd.concat([df_daily,df_norms_overlaid],axis=1)
        
        return df_combined
    
    def _isolate_waves(self,season,extreme_days):
        
        waves_by_month = {}
        
        taken = []
        for month in season:
            if month == 12:
                prev_month = 11
                next_month = 1
            elif month == 1:
                prev_month = 12
                next_month = 2
            else:
                prev_month = month - 1
                next_month = month + 1
            
            # reduce heat wave days to only the current and previous months.
            df_wm = extreme_days[(extreme_days.index.month == prev_month) | 
                                    (extreme_days.index.month == month) |
                                    (extreme_days.index.month == next_month)] 
            
            # identify heat wave days and whether they are 2 or more consecutive days
            date_diff = df_wm.index.to_series().diff()
            
            prev_date_1_day_ago = date_diff == np.timedelta64(1,'D')
            
            potential_start_days = np.argwhere((prev_date_1_day_ago == False).values)[:,0]
            wave_consecutive_days = np.argwhere(prev_date_1_day_ago.values)[:,0]
            
            is_start_day = np.concatenate([np.diff(potential_start_days) > 1,np.array([True])])
            
            if len(is_start_day) != 1:
                start_days = potential_start_days[is_start_day]
            else:
                start_days = np.array([],dtype=np.int64)
                
            all_wave_days = np.concatenate([wave_consecutive_days,start_days])
            all_wave_days.sort()
            
            
            # Quantify all heat waves in previous, current, and next months as a list
            # this does not capture the last heat wave
            waves = [all_wave_days[np.where(all_wave_days==s1)[0][0]:np.where(all_wave_days==s2)[0][0]] 
                          for s1,s2 in zip(start_days[:-1],start_days[1:])]
            # add the last heat wave.
            if len(start_days) > 1:
                waves.append(np.arange(start_days[-1],wave_consecutive_days[-1]))
            
            # Get rid of any heat waves that are outside or mostly in months. Any heat wave 
            # directly cut in half by between months is assigned to the current month but is added to
            # "taken" so that it will not be added to the next month in the next iteration.
            waves_by_month[month] = (self._delete_waves_fully_outside_this_month(
                waves,df_wm,month,prev_month,next_month,taken),df_wm)

        return waves_by_month
    
    def _determine_norm_param(self,dist_sample):
            mu = dist_sample.mean()
            sig = dist_sample.std()        
            
            
            return {'mu':mu,
                    'sig':sig}
    
    def _plot_fit(self,xdata,ydata,func,pvalues,fit_name,ax=None):
        
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            
        ax.scatter(xdata,ydata,marker="x",color="k")
        
        xline = np.arange(xdata.min(),1.01*xdata.max(),(xdata.max() - xdata.min())/100)
    
        yfunc = func(xline)
            
        ax.plot(xline,yfunc,color="k",linestyle="-")
        
        ax.grid("on")
        
        ax.set_title(fit_name)
    
        
    def _plot_linear_fit(self,xdata,ydata,params,pvalues,fit_name):
        
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        
        ax.scatter(xdata,ydata,marker="x",color="k")
        
        xline = np.arange(xdata.min(),1.5*xdata.max(),(xdata.max() - xdata.min())/2)
        if len(params) == 1:
            yline = params[0] * xline
        else:
            yline = params[1] * xline + params[0]
            
        ax.plot(xline,yline,color="k",linestyle="-")
        
        ax.grid("on")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(fit_name+"Pvalues = " + str(pvalues))
        
    
    def _transform_fit(self,signal):
        # this function maps a logarithm from 0 to interval making for a good log-normal fit
        return 2 *  (signal - signal.min())/(signal.max() - signal.min()) - 1
    
    def _inverse_transform_fit(self,norm_signal, signal_max, signal_min):
        return (norm_signal + 1)*(signal_max - signal_min)/2.0 + signal_min 
        #(np.exp(norm_signal/interval) - 1.0)/(np.exp(1.0) - 1) * (signal_max - signal_min) + signal_min
    
    def _calculate_wave_stats(self,waves,frac_tot_days,time_hours,is_hw,include_plots=True):
        
        if include_plots:
            fig1, ax1 = plt.subplots(4,3,figsize=(10,10))
            fig2, ax2 = plt.subplots(4,3,figsize=(10,10))
            fig3, ax3 = plt.subplots(4,3,figsize=(10,10))
            fig4, ax4 = plt.subplots(4,3,figsize=(10,10))
        
        hour_in_day = 24
        stats = {}
        row = 0;col=0
        for month,tup_waves_cur in waves.items():
            if col == 3:
                row += 1
                col = 0
            # duration
            waves_cur = tup_waves_cur[0]
            df_wm = tup_waves_cur[1]
            frac = frac_tot_days[month]
            
            # calculate duration stats
            month_duration = np.array([len(arr) for arr in waves_cur])
            
            try:
                max_duration = month_duration.max()
            except:
                import pdb;pdb.set_trace()
                
            num_day = np.arange(2,month_duration.max()+1)
            duration_history = np.array([(month_duration == x).sum() for x in num_day])
            temp_dict = {}
            
            # calculate log-normal for extreme temperature difference from average conditions and total wave energy
            if is_hw:
                extreme_temp = np.array([(df_wm.iloc[waves_cur[idx],:]["TMAX"]-df_wm.iloc[waves_cur[idx],:]["TAVG_B"]).max() for idx in range(len(waves_cur))])
            else:
                extreme_temp = np.array([(df_wm.iloc[waves_cur[idx],:]["TMIN"]-df_wm.iloc[waves_cur[idx],:]["TAVG_B"]).min() for idx in range(len(waves_cur))])
            
            # This produces negative energy for cold snaps and positive energy for heat waves.
            wave_energy = np.array([((df_wm.iloc[waves_cur[idx],:]["TMAX"] 
                                    + df_wm.iloc[waves_cur[idx],:]["TMIN"])/2 
                                   - df_wm.iloc[waves_cur[idx],:]["TAVG_B"]).sum() 
                                   for idx in range(len(waves_cur))])
            
            # convert from C*day to C*hr and day to hr
            wave_energy_C_hr = wave_energy * hour_in_day
            month_duration_hr = month_duration * hour_in_day
            
            # choose the extremum that is appropriate for the two types of waves (hot/cold)
            if is_hw:
                norm_extreme_temp = extreme_temp.max()
                norm_energy = wave_energy_C_hr.max()
            else:
                norm_extreme_temp = extreme_temp.min()
                norm_energy = wave_energy_C_hr.min()
            
            month_duration_norm = month_duration_hr / month_duration_hr.max()
            wave_energy_norm = wave_energy_C_hr/norm_energy
            extreme_temp_norm = extreme_temp/norm_extreme_temp
            
            # since all hw are exactly one day, the second term reduces to a constant offset term.
            size_tup = (len(month_duration),1)
            energy_regression_vars = np.concatenate(
                [(month_duration_norm).reshape(size_tup)],axis=1)
            max_temp_regression_vars = np.concatenate(
                [np.ones((size_tup)),
                 (month_duration_norm).reshape(size_tup)],axis=1)
            
    
            
            y_regression = [wave_energy_norm,extreme_temp_norm]
            X_regression = [energy_regression_vars,max_temp_regression_vars]
            
            results_shape_coef = []
            par = []
            for y_reg,X_reg in zip(y_regression,X_regression):
            
                model_shape_coef = sm.OLS(y_reg,X_reg)
            
                results_shape_coef.append(model_shape_coef.fit())
                
                par.append(results_shape_coef[-1].params)
            
            # be careful, the function assumes that the first term is the constant term
            # but here it is the slope making reversal of the params fit 
            
    
            E_func = lambda D: (par[0][0] * D)
            T_func = lambda D: (par[1][0] + par[1][1] * D)
    
            if include_plots:
                self._plot_fit(month_duration_norm,wave_energy_norm,
                                E_func, 
                                results_shape_coef[0].pvalues, str(month),ax1[row,col])
                
                self._plot_fit(month_duration_norm,extreme_temp_norm,
                                T_func, 
                                results_shape_coef[1].pvalues, str(month),ax2[row,col])

            # Calculate the linear growth of energy with duration based on the mean
            wave_energy_per_duration = wave_energy_C_hr / (norm_energy * E_func(month_duration_norm))   # deg C * hr / hr
            extreme_temp_per_duration = extreme_temp / (norm_extreme_temp * T_func(month_duration_norm))
            
            # transform to a common interval
            wave_energy_per_duration_norm = self._transform_fit(wave_energy_per_duration)
            extreme_temp_per_duration_norm = self._transform_fit(extreme_temp_per_duration)
            
            
            if include_plots:
                ax3[row,col].hist(wave_energy_per_duration_norm)
                ax4[row,col].hist(extreme_temp_per_duration_norm)

            temp_dict['help'] = ("These statistics are already mapped"+
                                " from -1 ... 1 and _inverse_transform_fit is"+
                                " needed to return to actual degC and"+
                                " degC*hr values. If input of actual values is desired use transform_fit(X,max,min)")
            temp_dict['energy_normal_param'] = self._determine_norm_param(wave_energy_per_duration_norm)
            temp_dict['extreme_temp_normal_param'] = self._determine_norm_param(extreme_temp_per_duration_norm)
            temp_dict['max extreme temp per duration'] = extreme_temp_per_duration.max()
            temp_dict['min extreme temp per duration'] = extreme_temp_per_duration.min()
            temp_dict['max energy per duration'] = wave_energy_per_duration.max()
            temp_dict['min energy per duration'] = wave_energy_per_duration.min()
            temp_dict['energy linear slope'] = par[0][0]
            temp_dict['normalized extreme temp duration fit slope'] = par[1][1]
            temp_dict['normalized extreme temp duration fit intercept'] = par[1][0]
            temp_dict['normalizing energy'] = norm_energy
            temp_dict['normalizing extreme temp'] = norm_extreme_temp
            temp_dict['normalizing duration'] = month_duration_hr.max()
            
            #
            # Markov chain model parameter estimation
            #
            hour_in_cur_month = time_hours * frac
            # for Markov chain model of heat wave initiation. 
            prob_of_wave_in_any_hour = len(month_duration)/hour_in_cur_month
            
            # for Markov chain model of probability a heat wave will continue into 
            # the next hour a linear regression is needed here
            
            num_hour_per_event_duration = (duration_history * num_day)*hour_in_day
            
            # We are fitting P(n) = P0^n, where P0 is the variable desired to determine and is the
            # Markov probability of transition out of the heat wave state based on the duration data.
            # this is a linear fit if we take the logarithm.
            prob_per_event = num_hour_per_event_duration/num_hour_per_event_duration.sum()
            
            included = prob_per_event != 0.0 # drop any heat wave durations that have no data!
            
            log_prob_duration = np.log(prob_per_event[included])
            
            num_hour_passed = num_day[included] * hour_in_day
            
            num_hour_passed = np.insert(num_hour_passed,0,0)
            log_prob_duration = np.insert(log_prob_duration,0,0)

            model = sm.OLS(log_prob_duration,num_hour_passed)
            
            results = model.fit()
            
            P0 = np.exp(results.params[0])
            
            # verify that the result is significant by p-value < 0.05 i.e. 
            # the probability that the null hypothesis is true (i.e. your results 
            # is a random chance occurance is <5%)
            if results.pvalues[0] > 0.05:
                self._plot_linear_fit(log_prob_duration,num_hour_passed,results.params,results.pvalues,
                                "Month=" + str(month)+" Markov probability")
                warnings.warn(results.summary())
                raise ValueError("The weather data has produced a low p-value fit"+
                                 " for Markov process fits! More data is needed "+
                                 "to produce a statistically significant result!")
                    
            # normalize and assume that the probability of going to a cold snap 
            # directly from a heat wave is zero.
            temp_dict['hourly prob stay in heat wave'] = P0 
            temp_dict['hourly prob of heat wave'] = prob_of_wave_in_any_hour
            
            stats[month] = temp_dict
            
            col+=1
            
        return stats
    
    
    def _wave_stats(self,df_combined,include_plots=False):
        
        """
        wave_stats(df_combined,is_heat)
        
        Calculates the statistical parameter per month for heat waves or cold snaps
        
        Parameters
        ----------    
        df_combined : pd.DataFrame
            A combined data set of NOAA historical
            daily data and NOAA climate 10%, 50%, and 90% data
            Violation of the 90% (for heat waves) or 10% data
            is how heat wave days are identified.
            
        Returns
        -------
        mews_stats : dict 
            A dictionary of mews statstics.
        """
        
        # total time covered by the dataset
        seconds_in_hour = 3600.0
        months_per_year = 12
        time_hours = (df_combined.index[-1] - df_combined.index[0]).total_seconds()/seconds_in_hour
        stats = {}
        
        is_heat_wave = [True,False]
        for is_hw in is_heat_wave:
            
            if is_hw:
                # above 90% criterion for TMAX_B or above the 90% criterion for the hourly maximum minimum temperature
                extreme_days = df_combined[(df_combined["TMAX"] > df_combined["TMAX_B"])|
                                             (df_combined["TMIN"] > df_combined["TMINMAX_B"])]
            else:
                # below 10% criterion TMIN or below 10% criterion for the minimum hourly maximum temperature
                extreme_days = df_combined[(df_combined["TMIN"] < df_combined["TMIN_B"])|
                                             (df_combined["TMAX"] < df_combined["TMAXMIN_B"])]
            
            # this is the number of heat wave days each month over the entire time period
            num_days_each_month = extreme_days.groupby(extreme_days.index.month).count()["TMIN"] #TMIN just makes it a series
            
            # this is the total days in each month.
            num_total_days = df_combined.groupby(df_combined.index.month).count()["TMIN"]
            frac_tot_days = num_total_days/num_total_days.sum()
            
            months = np.arange(1,months_per_year+1)
            
            # do a different assessment for each month in the heat wave season because 
            # the statistics show a significant peak at the end of summer and we do not
            # want the probability to be smeared out as a result.
            
            # we need to determine if the month starts or ends with heat waves. If it does,
            # the month with more days gets the heat wave. If its a tie, then the current month
            # gets the heat wave and the heat wave is marked as taken so that it is 
            # not double counted
                
        
            # now the last false before a true is the start of each heat wave 
            # then true until the next false is the duration of the heat wave
            waves_all_year = self._isolate_waves(months,extreme_days) 
            
            if is_hw:
                description = "heat wave"
            else:
                description = "cold snap"
            
            stats[description] = self._calculate_wave_stats(waves_all_year,
                                                            frac_tot_days,
                                                            time_hours,
                                                            is_hw,
                                                            include_plots)
        
        
        return stats
    
    def _plot_stats_by_month(self,stats,title_string):
        
        mu = []
        sig = []
        muT = []
        sigT = []
        p_hw = []
        ps_hw = []
        months = []
        
        for month,modat in stats.items():
            logdat = modat['energy_normal_param']
            logdelT = modat['extreme_temp_normal_param']
            mu.append(logdat['mu'])
            sig.append(logdat['sig'])
            muT.append(logdelT['mu'])
            sigT.append(logdelT['sig'])
            months.append(month)
            
            p_hw.append(modat['hourly prob of heat wave'])
            ps_hw.append(modat['hourly prob stay in heat wave'])
        
        fontsize={'font.size':16,'text.usetex':True}
        
        plt.rcParams.update(fontsize)    
           
        fig,axl = plt.subplots(6,1,figsize=(5,8))
        
        dat = [mu,sig,muT,sigT,p_hw,ps_hw]
        name = ["$\mu_{\Delta E}$","$\sigma_{\Delta E}$","$\mu_{\Delta T}$","$\sigma_{\Delta T}$","$P_{w}$","$P_{sw}$"]
        
        idx = 0
        for ax,da,na in zip(axl,dat,name):
            idx+=1
            ax.plot(months,da)
            ax.set_ylabel(na)
            ax.grid("on")
            ax.set_xticks(range(1,13))
            if idx < 5:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
        axl[0].set_title(title_string)
        axl[-1].set_xlabel("Month")
        
        plt.tight_layout()
        plt.savefig(title_string+"_monthly_MEWS_parameter_results.png",dpi=1000)
        
        
    
    
    def _delete_waves_fully_outside_this_month(self,heat_waves,df_wm,month,prev_month,next_month,hw_taken):
        
        keep_ind = []
        
        for idx,hw in enumerate(heat_waves):
    
            df_wave_ind = df_wm.iloc[hw,0].index
            # on boundary
            numday_in_month = (df_wave_ind.month == month).sum()
            if numday_in_month == 0:
                pass # do nothing, the entire heat wave is in a previous
                     # or posthumous month. this heat wave will not be included in keep_ind
            else:        
                if len(df_wave_ind) != numday_in_month:
                    # we have a boundary violation and we need to deal with it
                    num_prev = (df_wave_ind.month == prev_month).sum()
                    num_next = (df_wave_ind.month == next_month).sum()    
                    
                    if num_prev > numday_in_month and num_prev > num_next:
                        # the heat wave belongs in the previous month
                        pass
                    elif num_next > numday_in_month and num_next > num_prev:
                        # the heat wave belongs to the next month
                        pass
                    elif ((num_next == numday_in_month and num_next > num_prev) or
                         (num_prev == numday_in_month and num_prev > num_next) or
                         (num_next == num_prev)):
                        # this heat wave is given to the current month but must also be marked as taken since
                        # it will also qualify to be taken by the next month.
                        wave_is_taken = False
                        for dfhw in hw_taken:
                            if df_wave_ind.equals(dfhw):
                                wave_is_taken = True
                        
                        if not wave_is_taken:
                            hw_taken.append(df_wave_ind)
                            keep_ind.append(idx)
                            
                    else: # the majority of days are in the current month and the heat wave belongs 
                          # to the current month.
                        keep_ind.append(idx)
    
                else:
                    # no changes needed, the heat wave is fully in the current month.
                    keep_ind.append(idx)
        
        cur_month_heat_waves = [heat_waves[idx] for idx in keep_ind]
    
        return cur_month_heat_waves
            
class DeltaTransition_IPCC_FigureSPM6():
    """
    >>> obj = DeltaTransition_IPCC_FigureSPM6(hot_param,
                                              cold_param,
                                              climate_temp_func,
                                              year)
    
    This class assumes that we can divide by the 1.0 C multipliers for the present
    day and then multiply. We interpolate linearly or extrapolate linearly from
    the sparse data available.
    
    This is called within the context of a specific month.
    
    """
    
    
    
    def __init__(self,hot_param,cold_param,climate_temp_func,year):
                
        # bring in the ipcc data and process it.
        print(__file__)
        ipcc_data =  pd.read_csv(os.path.join(os.path.dirname(__file__),"data","IPCC_FigureSPM_6.csv"))
        
        # neglect leap years
        hours_in_10_years = 10 * 365 * 24  # hours in 10 years
        hours_in_50_years = 5 * hours_in_10_years
        
        # switch to the paper notation
        N10 = hours_in_10_years
        N50 = hours_in_50_years
        # probability a heat wave begins
        Phwm = hot_param["hourly prob of heat wave"]
        # probability a heat wave is sustainted
        Phwsm = hot_param["hourly prob stay in heat wave"]
        # probability of a cold snap
        Pcsm = cold_param["hourly prob of heat wave"]
        # probability of sustaining a cold snap
        Pcssm = cold_param["hourly prob stay in heat wave"]
        
        # This is change in global temperature from 2020!
        delta_TG = climate_temp_func(year)
        
        
        # now interpolate from the IPCC tables 
        (ipcc_val_10, ipcc_val_50) = self._interpolate_ipcc_data(ipcc_data, delta_TG)
        f_ipcc_50_10 = ipcc_val_10["Average Increase in Frequency"] 
        f_ipcc_50_50 = ipcc_val_50["Average Increase in Frequency"]
        
        # equation 22 (number may change)
        P_prime_hwm = Phwm * (f_ipcc_50_50 * N10 + f_ipcc_50_10 * N50)/(N10 + N50)
        
        
        
        # Estimate the interval positions of the 10 year and 50 year changes
        # in temperature.
        # equation 23 - all statistics have been translated to -1, 1, -1 is the minimum 
        #               extreme temperature and 1 is the maximum extreme temperature
        normalized_ext_temp = hot_param['extreme_temp_normal_param']
        normalized_energy = hot_param['energy_normal_param']
        maxtemp = hot_param['max extreme temp per duration']
        mintemp = hot_param['min extreme temp per duration']
        
        norm_energy = hot_param['normalizing energy']
        norm_temp = hot_param['normalizing extreme temp']
        norm_duration = hot_param['normalizing duration']
        alphaT = hot_param['normalized extreme temp duration fit slope']
        betaT = hot_param['normalized extreme temp duration fit intercept']
        
        mu_norm = normalized_ext_temp['mu']
        sig_norm = normalized_ext_temp['sig']
        
        # solve for the 10 year and 50 year expected peak temperature per duration
        F0 = lambda x: cdf_truncnorm(transform_fit(x,mintemp,maxtemp),
                                     mu_norm,
                                     sig_norm,
                                     -1,
                                     1)
        
        S10 = - 1 + 1/(N10 * Phwm)
        S50 = - 1 + 1/(N50 * Phwm)
        
        F10 = lambda x:F0(x) + S10
        F50 = lambda x:F0(x) + S50
        delTmax10_hwm, r10 = bisect(F10,mintemp,maxtemp,full_output=True)
        delTmax50_hwm, r50 = bisect(F50,mintemp,maxtemp,full_output=True)

        if not r10.converged:
            raise ValueError("Bisection method failed to find 10 year expected value for heat wave temperature")
        elif not r50.converged:
            raise ValueError("Bisection method failed to find 50 year expected value for heat wave temperature")
            
        # Find the -1..1 interval shift parameters that reflect the IPCC shift amounts
        # in temperature for 10 and 50 year events. This can shift and stretch the distribution.
        # solve for the shift in mean and standard deviation (2 equations two unknowns)
        # equation 24 in the writeup
        
        # sig_s and mu_s are the independent shift and stretch variables that 
        # are solvede as two unknowns. They are calculated within the original -1..1
        # interval and are not dimensional.. use inverse_transform_fit to give them
        # dimensions.
        
        # function for the establishment of truncated Gaussian shifting 
        # due to increasing maximum temperatures from
        # the original -1 .. 1 interval to a new interval S_m1 to S_1
        # this funciton is used by fsolve below
        
        # must normalize by duration

        D10 = np.log((1/(Phwm * N10)))/np.log(Phwsm)  # in hours - expected value
        D50 = np.log((1/(Phwm * N50)))/np.log(Phwsm)  # in hours - expected value
        
        # IPCC values must be normalized per duration to assure the correct amount
        # is added.
        new_delT_10 = delTmax10_hwm + ipcc_val_10["Avg Increase in Intensity"]/(
            norm_temp * (alphaT * (D10/norm_duration) + betaT))
        new_delT_50 = delTmax50_hwm + ipcc_val_50["Avg Increase in Intensity"]/(
            norm_temp * (alphaT * (D50/norm_duration) + betaT))
        def F10_50_S(npar):
            
            mu_s,sig_s = npar
            S_m1 = -1 + mu_s - (1 + mu_norm)/(sig_norm) * sig_s
            S_1 = 1 + mu_s + (1 - mu_norm)/(sig_norm) * sig_s
            return [
            cdf_truncnorm(
                transform_fit(new_delT_10,
                                mintemp,
                                maxtemp),
                mu_norm + mu_s,
                sig_norm + sig_s,
                S_m1,
                S_1) + S10
            ,
            cdf_truncnorm(
                transform_fit(new_delT_50,
                                    mintemp,
                                    maxtemp),
                mu_norm + mu_s,
                sig_norm + sig_s,
                S_m1,
                S_1) + S50]
    
        mu_guess = transform_fit(new_delT_10,mintemp,maxtemp) - transform_fit(
            delTmax10_hwm,mintemp,maxtemp)
        
        npar, infodict_s, ier_s, mesg_s = fsolve(F10_50_S, (mu_guess,0.0),full_output=True)
        
        if ier_s != 1:
            raise ValueError("The solution for change in mean and standard deviation did not converge!")
        
        del_mu_delT_max_hwm = npar[0]
        del_sig_delT_max_hwm = npar[1]
        
        # delta from -1...1 boundaries of the original transformed delT_max distribution.
        del_a_delT_max_hwm = -1 + del_mu_delT_max_hwm - (1 + mu_norm)/(sig_norm) * del_sig_delT_max_hwm
        del_b_delT_max_hwm = 1 + del_mu_delT_max_hwm + (1 - mu_norm)/(sig_norm) * del_sig_delT_max_hwm
        
        # adjusted durations - assume durations increase proportionally with 
        # temperature.
        S_D_10 = new_delT_10 / delTmax10_hwm
        if S_D_10 < 1.0:
            raise ValueError("A decrease in 10 year durations is not expected for the current analysis!")
        S_D_50 = new_delT_50 / delTmax50_hwm
        if S_D_50 < 1.0:
            raise ValueError("A decrease in 50 year durations is not expected for the current analysis!")
        
        D10_prime = D10 * S_D_10
        D50_prime = D50 * S_D_50
        
        P_prime_hwsm = (N10 * Phwsm ** (1/S_D_50) + N50 * Phwsm ** (1/S_D_10))/(N10 + N50)
        epsilon = 1.0e-6
        if P_prime_hwsm+epsilon < Phwsm:
            pdb;pdb.set_trace()
            raise ValueError("The probability of sustaining a heat wave has decreased. "+
                             "This should not happen in the current analysis!")
        
        # equation 30 optimal scaling of D_HW_Pm
        S_D_m = np.log(Phwsm)/np.log(P_prime_hwsm)
        if S_D_m + epsilon < 1.0:
            raise ValueError("The scaling factor on heat wave sustainment"+
                             " must be greater than 1.0")
        
        # equation 31 scaling of energy
        S_E_m = S_D_m * (new_delT_10/delTmax10_hwm + new_delT_50/delTmax50_hwm)/2

        del_mu_E_hw_m = transform_fit(
            S_E_m * inverse_transform_fit(
                normalized_energy['mu'], 
                hot_param['max energy per duration'], 
                hot_param['min energy per duration'])
            ,hot_param['min energy per duration'],
             hot_param['max energy per duration'])-normalized_energy['mu']     
        # transformation is not needed here, 
        # be careful here! inverse transform and transform have different 
        # orders for the min and max inputs!
        del_sig_E_hw_m = transform_fit(
            (inverse_transform_fit(
                normalized_ext_temp['sig'] + del_sig_delT_max_hwm, 
                maxtemp, 
                mintemp)/
              inverse_transform_fit(normalized_ext_temp['sig'],
                                          maxtemp, 
                                          mintemp))*
              inverse_transform_fit(normalized_energy['sig'], 
                                          hot_param['max energy per duration'],
                                          hot_param['min energy per duration']),
              hot_param['min energy per duration'],
              hot_param['max energy per duration'])-normalized_energy['sig']
        
        # delta from the -1..1 boundaries of the transformed Energy distribution (still in transformed space but no 
        # longer on the -1...1 interval.)
        del_a_E_hw_m = del_mu_E_hw_m - (1 + normalized_energy['mu'])/(normalized_energy['sig']) * del_sig_E_hw_m
        del_b_E_hw_m = del_mu_E_hw_m + (1 - normalized_energy['mu'])/(normalized_energy['sig']) * del_sig_E_hw_m
        
        # for the current work, assume cold snaps do not change with climate
        # TODO - add cold snap changes (decreases?)
        P_prime_csm = Pcsm
        P_prime_cssm = Pcssm
        
        # NEXT STEPS - GATHER ALL YOUR VARIABLES AND FORMULATE THE DELTA M matrix
        # RETURN THEM SO YOU CAN GET THEM INTO MEWS' original EXTREMES class.
        self.transition_matrix_delta = np.array(
            [[Phwm + Pcsm - P_prime_hwm - P_prime_csm, P_prime_csm - Pcsm, P_prime_hwm - Phwm],
             [Pcssm - P_prime_cssm, P_prime_cssm - Pcssm, 0.0],
             [Phwsm - P_prime_hwsm, 0.0, P_prime_hwsm - Phwsm]])
        self.del_E_dist = {'del_mu':del_mu_E_hw_m,
                 'del_sig':del_sig_E_hw_m,
                 'del_a':del_a_E_hw_m,
                 'del_b':del_b_E_hw_m}
        self.del_delTmax_dist = {'del_mu':del_mu_delT_max_hwm,
                 'del_sig':del_sig_delT_max_hwm,
                 'del_a':del_a_delT_max_hwm,
                 'del_b':del_b_delT_max_hwm}
        
    def _interpolate_ipcc_data(self,ipcc_data,delta_TG):
        
        # this function is dependent on the format of the table in IPCC_FigureSPM_6.csv
        
        present_tempanomal = ipcc_data['Global Warming Levels (⁰C)'].values[0]
        
        future_tempanomal = present_tempanomal + delta_TG
        
        if future_tempanomal > 4.0:
            raise ValueError("The current IPCC data only includes changes in temperature of 4C for global warming!")
        
        ind = 0
        
        for ind in range(4):
            if future_tempanomal <= ipcc_data['Global Warming Levels (⁰C)'].values[ind]:
                break
            
        ipcc_num = ipcc_data.drop(["Event","Units"],axis=1) 
        if ind > 0:
            interp_fact = (future_tempanomal - 
               ipcc_data['Global Warming Levels (⁰C)'].values[ind-1])/(
               ipcc_data['Global Warming Levels (⁰C)'].values[ind] - 
               ipcc_data['Global Warming Levels (⁰C)'].values[ind-1])
             
               
                 
            ipcc_val_10_u = (ipcc_num.loc[ind,:] - ipcc_num.loc[ind-1,:]) * interp_fact + ipcc_num.loc[ind-1,:]
            ipcc_val_50_u = (ipcc_num.loc[ind+4,:] - ipcc_num.loc[ind+3,:]) * interp_fact + ipcc_num.loc[ind+3,:]
        else:
            ipcc_val_10_u = ipcc_num.loc[0,:]
            ipcc_val_50_u = ipcc_num.loc[4,:]
        
        # TODO - if less recent data is available, this (Below) assumption
        # is non-conversative and will underestimate shifts in climate.
        
        # Assumption: Because, these values are being based off of data that is more recent,
        # the amplification/offset has to be based on current 1.0C warming levels.
        
        
        
        
        ipcc_val_10 = ipcc_val_10_u
        ipcc_val_50 = ipcc_val_50_u
        
        for lab,val in ipcc_val_10_u.iteritems():
            if "Intensity" in lab:
                ipcc_val_10[lab] = ipcc_val_10_u[lab] - ipcc_num.loc[0,lab]
                ipcc_val_50[lab] = ipcc_val_50_u[lab] - ipcc_num.loc[4,lab]
            elif "Frequency" in lab:
                ipcc_val_10[lab] = ipcc_val_10_u[lab] / ipcc_num.loc[0,lab]
                ipcc_val_50[lab] = ipcc_val_50_u[lab] / ipcc_num.loc[4,lab]

        return ipcc_val_10, ipcc_val_50    
    
    # this is the same as the function for ExtremeTemperatureWaves but with
    # min and max as inputs
    

class ClimateScenario(object):
    
    def __init__(self,use_global=True, 
                      lat=None,
                      lon=None,
                      baseline_year=2020,
                      end_year=2060,
                      model_guide="Models_Used_Alpha.xlsx",
                      data_folder="data",
                      run_parallel=True,
                      output_folder=None,
                      proxy=None):
        """
        Parameters
        ----------
        use_global : Bool, optional
            True: This defaults to the old MEWS v0.0.1 behavior where the global
            average from the IPCC Physical Basis Technical Summary is used for
            temperature increase. The default is True.
        lat : float, optional
            lattitude of the location where a temperature trajectory is needed. The default is None.
            Only takes effect if use_global_average = False. 
        lon : float, optional
            longitude of the location where a temperature trajectory is needed.
            Only takes effect if use_global_average = False. The default is None.
        baseline_year : int, optional
            Indicates the start year where temperature anomaly is measured from.
            The IPCC multiplication factors are reduced as baseline year becomes
            more recent in comparison to the 1850-1900 baseline.
        end_year: int, optional
            Indicates the end year for which CMIP6 evaluations will stop
        run_parallel : bool, optional
            Indicates whether to speed up process by parallelization by asyncrhonous pool
            and threading. 
        output_folder : str, optional
            path that indicates where all of the CMIP6 data files should end up.
            Default=None .. will install in folder "CMIP6_Data_Files" one level 
            above the MEWS root directory.
        proxy : str, optional
            name of proxy server that enables internet downloading of CMIP6
            data sets. Default = None .. indicates that no proxy server is 
            needed. The program may freeze if a proxy is needed and no
            proxy is given. 
        

        Returns
        -------
        None.

        """
        self.use_global = use_global
        if use_global:
            file_dir = os.path.dirname(__file__)
            self.df_temp_anomaly = pd.read_csv(os.path.join(file_dir,"data",
                                                            "IPCC2021_ThePhysicalBasis_TechnicalSummaryFigSPM8.csv"))
        else:
            self.lat = lat
            self.lon = lon
            self.end_year = end_year
            self.baseline_year = baseline_year
        self.table = [] # last column is the R2 value
        self.scenarios = []
        self.columns = [r'$t^3$',r'$t^2$',r'$t$',r'$1$',r'$R^2$']
        self.model_guide = model_guide
        self.data_folder = data_folder
        self.run_parallel = run_parallel
        self.output_folder = output_folder
        self.proxy = proxy
        
    def calculate_coef(self,scenario,lat=None,lon=None,scen_func_dict=None,years_per_calc=1):
        
        # use lat/lon 
        if not self.use_global:
            if lat is None:
                lat = self.lat
            if lon is None:
                lon = self.lon
        
            obj_dict = {}
            for year in np.arange(self.baseline_year,self.end_year+1):
                obj_dict[year] = CMIP_Data(lat_desired = lat,
                                            lon_desired = lon,
                                            year_baseline = self.baseline_year,
                                            year_desired = year,
                                            file_path = os.path.abspath(os.getcwd()),
                                            model_guide = self.model_guide,
                                            data_folder = self.data_folder,
                                            world_map=True,
                                            calculate_error=False,
                                            display_logging=False,
                                            display_plots=False,
                                            output_folder=self.output_folder,
                                            run_parallel=self.run_parallel,
                                            proxy=self.proxy)
                
        else:
            df = self.df_temp_anomaly[self.df_temp_anomaly["Climate Scenario"]==scenario]
        
        Xi = ((df["Year"]-2020)/50).values
        # 1.0 is the amount of heating assumed from 1850-1900 to 2020.
        Yi = (df["Temperature Anomaly degC (baseline 1850-1900)"]-1.0).values
        
        p_coef, residuals, rank, singular_values, rcond = np.polyfit(Xi,Yi,3,full=True)
        self.p_coef = p_coef
        
        SSres = np.sum((Yi-np.polyval(p_coef,Xi))**2)
        SStot = np.sum((Yi - np.mean(Yi))**2)
        
        R2 = np.array([1 - (SSres/SStot)])
        
        self.table.append(np.concatenate((p_coef, R2)))
        self.scenarios.append(scenario)
        
        return scen_func_dict

    # 0.267015 shifts to make the baseline year 2020 we assume a 1.0C affect 
    def climate_temp_func(self,year):
        return np.polyval(self.p_coef,(year-2020)/50)-0.267015
    
    def write_coef_to_table(self):
        table_df = pd.DataFrame(self.table,
                                index=self.scenarios,columns=self.columns)
        table_df.to_latex(r"C:\Users\dlvilla\Documents\BuildingEnergyModeling\MEWS\mews_temp\SimbuildPaper\from_python\cubic_coef_table.tex",
                          float_format="%.6G", escape=False)
        
    
    
        
            
