#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:44:54 2022

@author: dlvilla

This script verifies a case to have the Kolmogorov-Smirnov statistics for closeness of fit
between the historic model and 

You must run worcester_example.py and generate_files_from_solution_example.py

It also cross checks by loading all files generated and the climate norms, 

and the climate change polynomial 


"""
import os
import pickle
import numpy as np
import pandas as pd

def create_smirnov_table(obj):
    
    hist_obj_solve = obj.hist_obj_solve
    np.zeros([0,13])
    
    
    table_dict = {}
    for variable in hist_obj_solve[1].kolmogorov_smirnov.keys():
        table_dict[variable] = np.zeros([0,13])
    
    
    # establish columns for both tables.
    var1 = list(hist_obj_solve[1].kolmogorov_smirnov.keys())[0]
    first_column = ("hw",1,"month")
    columns = [first_column]
    num_cs = 0
    for trial_num, wave_type_dict in hist_obj_solve[1].kolmogorov_smirnov[var1].items():
        for wave_type, Ktest_obj in wave_type_dict.items():
            if wave_type == "cs":
                columns.append((wave_type.upper(), trial_num, 'statistic'))
                columns.append((wave_type.upper(), trial_num, 'p-value'))
                num_cs += 2
            else:
                columns.insert(len(columns)-num_cs, (wave_type.upper(), trial_num, 'statistic'))
                columns.insert(len(columns)-num_cs, (wave_type.upper(), trial_num, 'p-value'))
    
    for month, solve_obj in hist_obj_solve.items():
        for variable, random_trial_dict in solve_obj.kolmogorov_smirnov.items():
            next_row = np.zeros([1,13],dtype=float)
            next_row[0,0] = month
            for trial_num, wave_type_dict in random_trial_dict.items():
                for wave_type, Ktest_obj in wave_type_dict.items():
                    if wave_type == "hw":
                        base_ind = 1
                    else:
                        base_ind = 7
                    
                    next_row[0,base_ind + 2*(trial_num-1)] = Ktest_obj.statistic
                    next_row[0,base_ind + 2*(trial_num-1)+1] = Ktest_obj.pvalue
                    
                    
                    
            table_dict[variable] = np.concatenate([table_dict[variable],next_row],axis=0)
        
    
    df_dict = {}
    for variable, arr in table_dict.items():
        df = pd.DataFrame(arr,columns=columns)
        df.index = df[first_column]
        df.index = [int(ind) for ind in df.index]
        df.index.name = "month"
        df.drop(first_column,axis=1,inplace=True)
        df.columns=pd.MultiIndex.from_tuples(columns[1:])
        
        df.to_latex("worcester_kolmogorov_smirnov_"+variable+"_2x.tex")#,
                    #caption="Kolmogorov-Smirnov distribution fit statistics for "+variable
                    #+ ". The columns are by wave type, random trial, and "
                    #+"statistic type. 'statistic' is the supremum of differences"
                    #+" between the distributions and 'p-value' is the "
                    #+"acceptance value p-value > 0.05 rejects the null "
                    #+"hypothesis and represents 95\% confidence that the "
                    #+"distributions match" )
        
        pass
    
    
    pass

def quantify_event_errors_in_temperature(file_path,base_name,ssp,ci_,years):
    #reads csv files output 
    
    multi_col = []
    for year in years:
        multi_col.append((year,"10 yr","mean"))
        multi_col.append((year,"10 yr","std"))
        multi_col.append((year,"10 yr","min"))
        multi_col.append((year,"10 yr","max"))
        multi_col.append((year,"50 yr","mean"))
        multi_col.append((year,"50 yr","std"))
        multi_col.append((year,"50 yr","min"))
        multi_col.append((year,"50 yr","max"))
    
    
    max_val = [-1]
    min_val = [1]
    
    all_errors = np.array([0,0])
    
    multi_index = []
    data_table = []
    for ssp_ in ssp:
        for ci in ci_:
            multi_index.append((ssp_,ci)) 
            next_row = []
            
            for year in years:

                
                # the December csv has all months in it.
                month = 12
                df = pd.read_csv(os.path.join(file_path,base_name 
                                              + str(month) + "_" 
                                              + str(ssp_) + "_"
                                              + str(ci) + "_" 
                                              + str(year) + ".csv"))
                
                df_10_actual = df[(df["type"] == "10 year actual")]
                df_10_target = df[(df["type"] == "10 year target")]
                df_50_actual = df[(df["type"] == "50 year actual")]
                df_50_target = df[(df["type"] == "50 year target")]
                
                err_10 = df_10_actual["threshold"].values - df_10_target['threshold'].values
                err_50 = df_50_actual["threshold"].values - df_50_target['threshold'].values
                
                all_errors = np.concatenate([all_errors,err_10,err_50])
                
                
                next_row.append(err_10.mean())
                next_row.append(err_10.std())
                next_row.append(err_10.min())
                next_row.append(err_10.max())
                next_row.append(err_50.mean())
                next_row.append(err_50.std())
                next_row.append(err_50.min())
                next_row.append(err_50.max())
                
                if max_val[0] < err_10.max():
                    max_val = [err_10.max(),ssp_,ci,year,err_10.argmax()+1,"10 year"]
                if max_val[0] < err_50.max():
                    max_val = [err_50.max(),ssp_,ci,year,err_50.argmax()+1,"50 year"]
                    
                if min_val[0] > err_10.min():
                    min_val = [err_10.min(),ssp_,ci,year,err_10.argmin()+1,"10 year"]
                if min_val[0] > err_50.min():
                    min_val = [err_50.min(),ssp_,ci,year,err_50.argmin()+1,"50 year"]
                
                                
            data_table.append(next_row)
    print("min_val = " + str(min_val))
    print("max_val = " + str(max_val))
    
    print("mean of all errors: " + str(all_errors.mean()))
    print("standard deviation of all errors" + str(all_errors.std()))
    final_df = pd.DataFrame(data_table,index=pd.MultiIndex.from_tuples(multi_index), 
                            columns = pd.MultiIndex.from_tuples(multi_col))
    final_df.T.to_latex("future_temperature_errors.tex")


if __name__ == "__main__":
    results_file = os.path.join("temp_pickles","obj_worcester_2x.pickle")
    
    file_path = os.path.join("example_data","Worcester","results2")
    ssp = ["SSP245","SSP370","SSP585"]
    ci_ = ["5%","50%","95%"]
    years = [2020,2040,2060,2080]
    base_name = "worcester_future_future_month_"
    
    tup = pickle.load(open(results_file,'rb'))
    
    quantify_event_errors_in_temperature(file_path,base_name,ssp,ci_,years)
    
    obj = tup[0]
    
    create_smirnov_table(obj)