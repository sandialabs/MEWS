#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 08:04:11 2023

@author: dlvilla
"""

import os

def post_process_eplus_run(path):
    pass


results_dir = r"/gpfs/dlvilla/HuiOHauula_ResilienceCenter_04142023"

btypedir = os.listdir(results_dir)

failed_runs = []
succeeded_runs = []

for btyp in btypedir:
    path = os.path.join(results_dir,btyp)
    
    run_list = os.listdir(path)
    
    for run in run_list:
        path2 = os.path.join(path,run)
        files = os.listdir(path2)
        if not "mews_study_complete.out" in files and len(files)>0:
            failed_runs.append(path2)
        else:
            succeeded_runs.append(path2)
            
            
empty_run = '/gpfs/dlvilla/HuiOHauula_ResilienceCenter_04142023/ResilienceCenter_normal_business_as_usual/USA_HI_MCB.Hawaii-Kaneohe.Bay.MCAS.Oahu.911760_TMYx.2004-2018_Adjusted_for_HuiOHauulaSSP585_2020_95%_r222'