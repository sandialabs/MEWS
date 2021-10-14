# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 09:59:38 2021

@author: dlvilla
"""
from mews.stats import Extremes
from mews.weather.psychrometrics import relative_humidity

import io
from calendar import monthrange
import requests as rqs
import pandas as pd
import numpy as np
import os
import requests as rqs
import shutil
import urllib
from urllib.parse import urlparse


from datetime import datetime, timedelta
from contextlib import closing
from scipy import stats
import statsmodels.api as sm
from warnings import warn


class ExtremeTemperatureWave():
    
    #  These url's must end with "/" !
    
    # norms are provided in Fahrenheit!
    norms_url  = "https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/"
    # daily data are provided in Celcius!
    daily_url = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/"
    
    def __init__(self,station,weather_files,use_local=False):
        
        # consistency checks
        self._check_NOAA_url_validity()
        
        # temperature specific tasks
        
        # ! TODO - move all plotting out of this class
        plt.close("all")

        
        # The year is arbitrary and should simply not be a leap year
        self._read_and_curate_NOAA_data(station,2001, use_local)
        
        stats = self._wave_stats(df_combined)
        
        self._plot_stats_by_month(stats["heat wave"],"Heat Waves")
        self._plot_stats_by_month(stats["cold snap"],"Cold Snaps")
        
        # Here is where development stopped, I need to map between the inputs
        # to extremes and 
        pdb.set_trace()
        
        # now initiate use of the Extremes class to unfold the process
        ext_obj = Extremes(start_year,
                 max_avg_dist,
                 max_avg_delta,
                 min_avg_dist,
                 min_avg_delta,
                 transition_matrix,
                 transition_matrix_delta,
                 weather_files,
                 num_realizations,
                 num_repeat=1,
                 use_cython=True,
                 column='Dry Bulb Temperature',
                 tzname=None,
                 write_results=True,
                 results_folder="",
                 results_append_to_name="",
                 run_parallel=True,
                 frac_long_wave=0.8,
                 min_steps=24,
                 test_shape_func=False,
                 doe2_input=None,
                 random_seed=None)
    
    pass

    def _read_and_curate_NOAA_data(self,station,year,use_local=False):
    
        """
        read_NOAA_data(station,year=None,use_local=False)
        
        Reads National Oceanic and Atmospheric Association (NOAA)
        daily-summaries data over a long time period and
        NOAA climate norms hourly data from 1991-2020 from the url's
        self.norms_url and self.daily_url. Creates self.NOAA_data dataframe
        
        
        This funciton is highly specific to the data formats that worked on 
        10/14/2021 for the NOAA data repositories and will likely need significant
        updating if new data conventions are used by NOAA.
        
        Inputs:
        -------
        
        station : str : must be a valid weather station ID that has a valid representation 
                  for both the self.norms_url and self.daily_url web locations
                  the error handling provides a list of valid station ID's that
                  meets this criterion if a valid ID is not provded.
        
        year : int : The year to be assigned to the climate norms data for 
                     dataframe purposes. This year is later overlaid accross
                     the range of the daily data
        
        use_local : if MEWS is not reading web-urls use this to indicate to look
                    for local files at "station" file path location. The
                    
        Returns:
        --------
        
        NOAA_data : DataFrame : contains the daily summaries with statistical summary
                    daily data from the climate normals overlaid so that heat and cold 
                    wave comparisons can be made.
        
        """
        df_temp = []
        for url in [self.daily_url,self.norms_url]:
    
            if station[-4:] == ".csv":
                ending = ""
            else:
                ending = ".csv"
        
            if use_local:
                if isdaily:
                    df = pd.read_csv(station + ending)
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
        
            
            if isdaily:
                # specific processing for daily summary data.
                df['DATE'] = pd.to_datetime(df['DATE'])
                df.index = df['DATE']
        
                # numbers provided are degrees Celcius in tenths.
                df['TMAX'] = df['TMAX']/10.0
                df['TMIN'] = df['TMIN']/10.0
                meta_data = {'STATION':df['STATION'].iloc[0],
                             'LONGITUDE':df['LONGITUDE'].iloc[0],
                             'LATITUDE':df['LATITUDE'].iloc[0],
                             'ELEVATION':df['ELEVATION'].iloc[0],
                             'NAME':df['NAME'].iloc[0]}
                
                df_new = df[['TMIN','TMAX']]
                df_new.meta = meta_data
            
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
        
        self.NOAA_data = self.extend_boundary_df_to_daily_range(df_norms,df_daily)
        return self.NOAA_data
    
    def _check_NOAA_url_validity(self):
        def is_valid(url):
            """
            Checks whether `url` is a valid URL.
            """
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        
        err_msg = "MEWS: Test to "+
                    "see if this url is still valid on an internet connected"+
                    " computer for NOAA and contact them if it is no longer"+
                    " working and update the source code in the "+
                    "ExtremeTemperatureWave.daily_url or .norms_url constants!"
        
        if not is_valid(self.daily_url):
            raise urllib.error.HTTPError(self.daily_url,,err_msg)
        elif not is_valid(self.norms_url):
            raise urllib.error.HTTPError(self.norms_url,,err_msg)
            
    def _extend_boundary_df_to_daily_range(self,df_norms,df_daily):
        """
        This function overlays the climate norms 90th percentile and all other columns 
        over the daily data so that heat wave statistics can be quantified. 
        
        
        """
        
        df_combined = df_daily
        
        unique_year = df_daily.index.year.unique()
        
        # verify that every year is consecutive
        if np.diff(unique_year).sum() + 1 != len(unique_year):
            raise ValueError("The daily data from NOAA has one or more gaps of a year! Please use a station that has continuous data!")
        
    
        
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
            
            start_days = potential_start_days[is_start_day]
            
            all_wave_days = np.concatenate([wave_consecutive_days,start_days])
            all_wave_days.sort()
            
            
            # Quantify all heat waves in previous, current, and next months as a list
            # this does not capture the last heat wave
            waves = [all_wave_days[np.where(all_wave_days==s1)[0][0]:np.where(all_wave_days==s2)[0][0]] 
                          for s1,s2 in zip(start_days[:-1],start_days[1:])]
            # add the last heat wave.
            waves.append(np.arange(start_days[-1],wave_consecutive_days[-1]))
            
            # Get rid of any heat waves that are outside or mostly in months. Any heat wave 
            # directly cut in half by between months is assigned to the current month but is added to
            # "taken" so that it will not be added to the next month in the next iteration.
            waves_by_month[month] = (delete_waves_fully_outside_this_month(
                waves,df_wm,month,prev_month,next_month,taken),df_wm)
            
            # now calculate the statistics 
            
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
        
    
    def _transform_fit_log_norm(self,signal):
        # this function maps a logarithm from 0 to interval making for a good log-normal fit
        return 2 *  (signal - signal.min())/(signal.max() - signal.min()) - 1
    
    def _inverse_transform_fit_log_norm(self,norm_signal, signal_max, signal_min):
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
            
            max_duration = month_duration.max()
            
            num_day = np.arange(2,month_duration.max()+1)
            duration_history = np.array([(month_duration == x).sum() for x in num_day])
            temp_dict = {}
            
            # calculate log-normal for extreme temperature and total wave energy
            if is_hw:
                extreme_temp = np.array([df_wm.iloc[waves_cur[idx],:]["TMAX"].max() for idx in range(len(waves_cur))])
            else:
                extreme_temp = np.array([df_wm.iloc[waves_cur[idx],:]["TMIN"].min() for idx in range(len(waves_cur))])
            
            
            
            
            # This produces negative energy for cold snaps and positive energy for heat waves.
            wave_energy = np.array([((df_wm.iloc[waves_cur[idx],:]["TMAX"] 
                                    + df_wm.iloc[waves_cur[idx],:]["TMIN"])/2 
                                   - df_wm.iloc[waves_cur[idx],:]["TAVG_B"]).sum() 
                                   for idx in range(len(waves_cur))])
            
            wave_energy_C_hr = wave_energy * hour_in_day
            month_duration_hr = month_duration * hour_in_day
            
            # The following is a fit over the Extremes.double_shape_func:
                #shape(t,D) = A * np.sin(t * np.pi/D) + B * (1.0-np.cos(t * np.pi / (0.5*min_s)))
            # Two constraints come from the heat wave data. 
            # 1. Total energy E/Emax = (2 * A * D / pi + B)/Emax  , we take the data for the heat
            #    wave energy and fit 
            # 2. Peak temperature T/Tmax = (A + B)/Tmax
            # 
            # A and B are both functions of D where
            # A = p0 * D ^ (1/log(Dmax))
            # B = p1 * D ^ (1/log(Dmax))
            #
            # once p0 and p1 are determined, the way that heat waves will grow with respect to
            #
            # a sinusoidal crest (A) and day-to-day heat increase (B) will be fully determined
            # based on the duration of the heat wave. A and B are fit to best reflect historical
            # energy and peak temperature data.
    
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
    
            
            plot_fit(month_duration_norm,wave_energy_norm,
                            E_func, 
                            results_shape_coef[0].pvalues, str(month),ax1[row,col])
            
            plot_fit(month_duration_norm,extreme_temp_norm,
                            T_func, 
                            results_shape_coef[1].pvalues, str(month),ax2[row,col])
            
            # Calculate the linear growth of energy with duration based on the mean
            wave_energy_per_duration = wave_energy_C_hr / (norm_energy * E_func(month_duration_norm))   # deg C * hr / hr
            extreme_temp_per_duration = extreme_temp / (norm_extreme_temp * T_func(month_duration_norm))
            
            # transform to a common interval
            wave_energy_per_duration_norm = transform_fit_log_norm(wave_energy_per_duration)
            extreme_temp_per_duration_norm = transform_fit_log_norm(extreme_temp_per_duration)
            
            if include_plots:
                ax3[row,col].hist(wave_energy_per_duration_norm)
                ax4[row,col].hist(extreme_temp_per_duration_norm)
    
            
            temp_dict['energy_normal_param'] = determine_norm_param(wave_energy_per_duration_norm)
            temp_dict['extreme_temp_normal_param'] = determine_norm_param(extreme_temp_per_duration_norm)
            temp_dict['max extreme temp per duration'] = extreme_temp_per_duration.max()
            temp_dict['min extreme temp per duration'] = extreme_temp_per_duration.min()
            temp_dict['max energy per duration'] = wave_energy_per_duration.max()
            temp_dict['min energy per duration'] = wave_energy_per_duration.min()
            temp_dict['energy linear slope'] = par[0][0]
            temp_dict['normalized extreme temp duration fit slope'] = par[1][1]
            temp_dict['normalized extreme temp duration fit intercept'] = par[1][0]
            
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
                plot_linear_fit(log_prob_duration,num_hour_passed,results.params,results.pvalues,
                                "Month=" + str(month)+" Markov probability")
                print(results.summary())
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
    
    
    def _wave_stats(self,df_combined):
        
        """
        wave_stats(df_combined,is_heat)
        
        Calculate the statistical parameter per month for heat waves or cold snaps
        
        Inputs:
        -------    
            df_combined : pd.DataFrame : A combined data set of NOAA historical
                          daily data and NOAA climate 10%, 50%, and 90% data
                          Violation of the 90% (for heat waves) or 10% data
                          is how heat wave days are identified.
        Returns:
        --------
            mews_stats : dict : A dictionary 
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
            waves_all_year = isolate_waves(months,extreme_days) 
            
            if is_hw:
                description = "heat wave"
            else:
                description = "cold snap"
            
            stats[description] = calculate_wave_stats(waves_all_year,frac_tot_days,time_hours,is_hw)
        
        return stats
    
    def _plot_stats_by_month(self,stats,title_string):
        
        mu = []
        sig = []
        p_hw = []
        ps_hw = []
        months = []
        
        for month,modat in stats.items():
            logdat = modat['energy_normal_param']
            mu.append(logdat['mu'])
            sig.append(logdat['sig'])
            months.append(month)
            
            p_hw.append(modat['hourly prob of heat wave'])
            ps_hw.append(modat['hourly prob stay in heat wave'])
        
        fontsize={'font.size':16,'text.usetex':True}
        
        plt.rcParams.update(fontsize)    
           
        fig,axl = plt.subplots(4,1,figsize=(5,6))
        
        dat = [mu,sig,p_hw,ps_hw]
        name = ["$\mu$","$\sigma$","$P_{w}$","$P_{sw}$"]
        
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
        plt.savefig(title_string+"_monthly_MEWS_parameter_results.png",dpi=400)
        
        
    
    
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
            
        
