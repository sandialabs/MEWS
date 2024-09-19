# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:11:22 2022

@author: dlvilla
"""



from mews.tests.test_extreme_temperature_waves import Test_ExtremeTemperatureWaves
import os
import pickle as pkl
from time import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def single_row_of_avg_vals(subdict):
    arr = None
    columns = []
    for key,sub2dict in subdict.items():
        for key2, sub3dict in sub2dict.items():
            
            df = pd.DataFrame(sub3dict)
            if key == "Intensity":
                df["IPCC actual value"] = df["IPCC actual value"].apply(lambda x: x[1])
                columns.append((key,key2,df.columns[0],"50 year"))
                columns.append((key,key2,df.columns[1],"50 year"))
            else:
                columns.append((key,key2,df.columns[0],"10 year"))
                columns.append((key,key2,df.columns[0],"50 year"))
                columns.append((key,key2,df.columns[1],"10 year"))
                columns.append((key,key2,df.columns[1],"50 year"))
            
            if arr is None:
                if key=="Intensity":
                    temp_arr = np.array([[row[0]['mean'],row[1]]for row in df.values])
                else:
                    temp_arr = np.array([[row[0][0],row[0][1],row[1][0],row[1][1]] for row in df.values])
                arr = temp_arr.mean(axis=0).reshape([1,temp_arr.shape[1]])
                #arr = df.mean().values.reshape([1,2])
            else:
                if key=="Intensity":
                    temp_arr = np.array([[row[0]['mean'],row[1]]for row in df.values])
                else:
                    temp_arr = np.array([[row[0][0],row[0][1],row[1][0],row[1][1]] for row in df.values])
                arr = np.concatenate([arr,temp_arr.mean(axis=0).reshape([1,temp_arr.shape[1]])],axis=1)
            
    return arr,columns
    

obj = Test_ExtremeTemperatureWaves()


fpath = os.path.dirname(__file__)
model_guide = os.path.join(fpath,"..","mews","data_requests","data","Models_Used_alpha.xlsx")
data_folder = os.path.join(fpath,"..","..","CMIP6_Data_Files")
station = os.path.join(fpath,"..","mews","tests","data_for_testing","USW00023050.csv")
weather_files = [os.path.join(fpath,"..","mews","tests","data_for_testing","USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]

# frequency convergence
convergence_array = [25,50,100,200,300,400,500,750,1000]#,400,500,600,700,800,900,1000]
random_seed = [38405,50647,708349,3123201,443005,65574,73365,8746,9870]#,23354,23123,46234]


seconds_in_minute = 60
runme = True
restart = False
current_file_name = "convergence_study_7.pickle"
next_file_name = "convergence_study_7.pickle"

if runme:
    if restart:
        (metric_meta_dict,restart_tup) = pkl.load(open(current_file_name,'rb'))[0]
    else:
        metric_meta_dict = {}
        restart_tup = None
    
    
    clim_scen = None
    scen_dict = None
    objExtW = None
    
    
    
    initiate_runs = False
    for ca,rs in zip(convergence_array,random_seed):
        if (restart_tup is None) or (restart_tup == (ca,rs)):
            initiate_runs = True
            
        if initiate_runs:
            start = time()
            metric_meta_dict[(ca,rs)], meanvals, objExtW, clim_scen, scen_dict = obj._run_verification_study_of_use_global_False(ca,
                                                                2080,
                                                                rs,
                                                                False,
                                                                "SSP585",
                                                                model_guide,
                                                                data_folder,
                                                                weather_files,
                                                                station,
                                                                print_progress=True,
                                                                number_cores=30,
                                                                write_results=False,
                                                                run_parallel=True,
                                                                scen_dict=scen_dict,
                                                                clim_scen=clim_scen,
                                                                obj=objExtW,
                                                                ci_interval=["95%"])
            end = time()
            print("\nThe {0} case took {1:5.2f} minutes to run.\n\n".format(str((ca,rs)),(end-start)/seconds_in_minute))
            #print("\n\n\nPattern!@#$ \n\n\nRun "+ str((ca,rs))+" failed! \n\n\n")
            
            pkl.dump([metric_meta_dict,(ca,rs)],open(next_file_name,'wb'))
    
else:
    metric_meta_dict_post = pkl.load(open(next_file_name,'rb'))[0]
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib','qt')
    

num_realizations = [] 

post_process = not runme
arr = None
if post_process:
    for tup, subdict in metric_meta_dict_post.items():
        (num_realization, random_seed) = tup
        
        num_realizations.append(num_realization)
        
        
        row, columns = single_row_of_avg_vals(subdict)
        if arr is None:
            arr = row
        else:
            arr = np.concatenate([arr,row],axis=0)
    
    df = pd.DataFrame(arr, index=num_realizations, columns=columns)
    fig,axl = plt.subplots(2,1,sharex=True,figsize=(10,10))

int_col = []
freq_col = []
for col in df.columns:   
    if "Intensity" in col:
        int_col.append(col)
    else:
        freq_col.append(col)
        

df[int_col].plot(ax=axl[0])
df[freq_col].plot(ax=axl[1])
    