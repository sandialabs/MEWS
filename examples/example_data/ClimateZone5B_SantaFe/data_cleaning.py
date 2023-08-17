# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:53:48 2023

@author: dlvilla
"""

import pandas as pd
import os
from scipy import stats
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from copy import deepcopy
from matplotlib import pyplot as plt

def data_cleaning(random_seed, date_format, in_file, out_file, 
                  high_temp_record, low_temp_record, training_intervals):

    np.random.seed(seed=random_seed)
    def filter_df_for_boundary_violation(df_VIOL,dfT,qTMAX,qTMIN):
        for index, row in df_VIOL.iterrows():
            tmax_val = df.loc[index,"TMAX"]
            tmin_val = df.loc[index,"TMIN"]
            if tmax_val > qTMAX:
                dfT.at[index,"TMAX"] = np.nan
            if tmin_val > qTMAX:
                dfT.at[index,"TMIN"] = np.nan
            if tmax_val < qTMIN:
                dfT.at[index,"TMAX"] = np.nan
            if tmin_val < qTMIN:
                dfT.at[index,"TMIN"] = np.nan
    

    
    df = pd.read_csv(in_file)
    index = pd.to_datetime(df["DATE"],format=date_format).values
    
    df.index = index
    
    dfT = df[["TMAX","TMIN"]]

    qTMAX = high_temp_record#
    qTMIN = low_temp_record # 
    
    df_MIN_VIOL = dfT[(dfT["TMAX"] < qTMIN) | (dfT["TMIN"] < qTMIN)]
    df_MAX_VIOL = dfT[(dfT["TMAX"] > qTMAX) | (dfT["TMIN"] > qTMAX)]
    
    filter_df_for_boundary_violation(df_MIN_VIOL,dfT,qTMAX,qTMIN)
    filter_df_for_boundary_violation(df_MAX_VIOL,dfT,qTMAX,qTMIN) 
    
    train_list = [dfT.loc[index[intv[0]]:index[intv[1]]].fillna(method="ffill") 
                  for intv in training_intervals]
    df_train = pd.concat(train_list)    
    
    
    decompose_data = seasonal_decompose(df_train["TMAX"], model="additive",period=365)
    decompose_data.plot();  
    
    decompose_data = seasonal_decompose(df_train["TMIN"], model="additive",period=365)
    decompose_data.plot();  
    
    dftest = adfuller(df_train["TMAX"], autolag = 'AIC')
    print("1. ADF : ",dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)
    
    df_train_max_monthly = df_train["TMAX"].resample("M").mean().dropna()
    df_train_min_monthly = df_train["TMIN"].resample("M").mean().dropna()
    
    model_tmax_w_tmin=sm.tsa.statespace.SARIMAX(df_train_max_monthly,df_train_min_monthly,order=(1, 1, 1),seasonal_order=(1,1,1,12),)
    model_tmin_w_tmax=sm.tsa.statespace.SARIMAX(df_train_min_monthly,df_train_max_monthly,order=(1, 1, 1),seasonal_order=(1,1,1,12),)
    model_tmax=sm.tsa.statespace.SARIMAX(df_train_max_monthly,order=(1, 1, 1),seasonal_order=(1,1,1,12),)
    model_tmin=sm.tsa.statespace.SARIMAX(df_train_min_monthly,order=(1, 1, 1),seasonal_order=(1,1,1,12),)
    results_tmax_w_tmin=model_tmax_w_tmin.fit()
    results_tmin_w_tmax = model_tmin_w_tmax.fit()
    results_tmax = model_tmax.fit()
    results_tmin = model_tmin.fit()
    
    max_forecast = results_tmax.get_forecast(12)
    min_forecast = results_tmin.get_forecast(12)
    
    dfT_no_gaps = dfT.resample("D").asfreq(np.nan)
    df_before = deepcopy(dfT_no_gaps)
    df_no_gaps = df.resample("D").asfreq(np.nan)
    
    
    for dat,row in dfT_no_gaps.iterrows():
        if row["TMIN"] == 0.0:
            row.at["TMIN"] = np.nan
        if row["TMAX"] == 0.0:
            row.at["TMAX"] = np.nan
    
        if row.isna().sum() == 0:
            continue
        elif row.isna().sum() == 2:
            min_month = min_forecast.predicted_mean.index.month == dat.month
            max_month = max_forecast.predicted_mean.index.month == dat.month
            min_std = np.sqrt(min_forecast.var_pred_mean[min_month].values[0])
            max_std = np.sqrt(max_forecast.var_pred_mean[max_month].values[0])
            row.at["TMIN"] = min_forecast.predicted_mean[min_month].values[0] + np.random.normal(0.0,min_std)
            row.at["TMAX"] = max_forecast.predicted_mean[max_month].values[0] + np.random.normal(0.0,max_std)
        elif np.isnan(row["TMAX"]):
            forecast = results_tmax_w_tmin.get_forecast(12,exog=np.ones(12)*row["TMIN"])
            row.at["TMAX"] = forecast.predicted_mean[forecast.predicted_mean.index.month == dat.month].values[0]
        elif np.isnan(row["TMIN"]):     
            forecast = results_tmin_w_tmax.get_forecast(12,exog=np.ones(12)*row["TMAX"])
            row.at["TMIN"] = forecast.predicted_mean[forecast.predicted_mean.index.month == dat.month].values[0]
        else:
            raise ValueError("this should never happen!")
            
        if row["TMAX"] < row["TMIN"]:
            raise ValueError("The approach is flawed. You need to regress TMAX - TMIN instead!")
    
    df_no_gaps["TMAX"] = dfT_no_gaps["TMAX"]
    df_no_gaps["TMIN"] = dfT_no_gaps["TMIN"]
    fig,ax = plt.subplots(1,1)
    
    def apply_func(row):
        row["DATE"] = "{0:d}-{1:0>2}-{2:0>2}".format(
                row.name.year,row.name.month,row.name.day)
        return row
        
    df_no_gaps = df_no_gaps.apply(apply_func,axis = 1)
    
    df_no_gaps[["TMAX","TMIN"]].plot(ylabel="Temperature ($^\\circ$C)",xlabel="Year",ax=ax,legend=None) 
    df_before.plot(ax=ax)       
    ax.legend(["TMAX fill","TMIN fill","TMAX data","TMIN data"],loc="upper left", ncol=2)
    fig.savefig("noaa_daily_summaries_gap_fill.png",dpi=300)
    breakpoint()
    df_no_gaps.to_csv(out_file,index=False)

if __name__ == "__main__":
    random_seed = 54564891
    date_format = '%Y-%m-%d'

    daily_file = "USW00023049_daily.csv"
    out_file = "USW00023049_daily_filtered.csv"
    high_temp_record = 394 #tenths of C  highest temperature recorded in Santa Fe June14,2021
    low_temp_record = -311 # tenths of C coldest temperature ever recorded in Santa Fe Feb 3rd, 2011
    
    #,[-2000,-1]
    
    bb = [[0,3650]]
    
    data_cleaning(random_seed, date_format, daily_file, out_file, 
                      high_temp_record, low_temp_record, bb)
