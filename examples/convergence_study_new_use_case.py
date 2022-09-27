# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:11:22 2022

@author: dlvilla
"""

if not "Test_ExtremeTemperatureWaves" in globals():
    from mews.tests.test_extreme_temperature_waves import Test_ExtremeTemperatureWaves
import os
import pickle as pkl

runme = True
if runme:

    obj = Test_ExtremeTemperatureWaves()
    
    
    fpath = os.path.dirname(__file__)
    model_guide = os.path.join(fpath,"..","mews","data_requests","data","Models_Used_alpha.xlsx")
    data_folder = os.path.join(fpath,"..","..","CMIP6_Data_Files")
    station = os.path.join(fpath,"..","mews","tests","data_for_testing","USW00023050.csv")
    weather_files = [os.path.join(fpath,"..","mews","tests","data_for_testing","USA_NM_Albuquerque.Intl.AP.723650_TMY3.epw")]
    
    # frequency convergence
    convergence_array = [40,80,120,160,200,240,280,320,400,500,600,700,800,900,1000]
    random_seed = [2293,123,345,567,789,321,432,543,654,765,876,987,23354,23123,46234]
    clim_scen = None
    scen_dict = None
    
    metric_meta_dict = {}
    for ca,rs in zip(convergence_array,random_seed):

        metric_meta_dict[(ca,rs)], mean5, mean50, mean95, obj, clim_scen, scen_dict = obj._run_verification_study_of_use_global_False(ca,
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
                                                            run_parallel=True,
                                                            scen_dict=scen_dict,
                                                            clim_scen=clim_scen)

        #print("\n\n\nPattern!@#$ \n\n\nRun "+ str((ca,rs))+" failed! \n\n\n")
        
    pkl.dump([metric_meta_dict],open("convergence_study_2.pickle",'wb'))
    
else:
    metric_meta_dict = pkl.load(open("convergence_study.pickle",'rb'))[0]
    