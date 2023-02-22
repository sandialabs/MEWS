#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:34:40 2023

@author: dlvilla
"""
import os
import pickle as pkl
from mews.run_mews import extreme_temperature

# You must change example_dir to an absolute path to a copy of the MEWS repository 
# examples folder if you move this script. 
# any variable you put in the run_dict file location must be defined before calling extreme temperature
example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# This must be a path to a valid run python dictionary text file. You can change the values 
# in this long input file (very carefully to not corrupt the dictionary syntax) or you can just copy
# and paste the dictionary here so that you can edit this input in an IDE.
# The "only1file.dict" has the number of files to generate per case reduced to 1
# use the "mews_run.dict" for a version that generate 200. You can change the input
# to any value you want depending on how many files are needed.
run_dict_file_path = os.path.join(__file__,"example_data","mews_run_only1file.dict")

"""
This is the way to run mews. All the names available are

"""

results = extreme_temperature(run_dict=run_dict_file_path,
                              only_generate_files=['Houston'],
                              skip_runs=["Chicago",
                                         "Baltimore",
                                         "Minneapolis",
                                         "Phoenix",
                                         'Miami',
                                         'Atlanta', 
                                         'LasVegas',
                                         'LosAngeles',
                                         'SanFrancisco',
                                         'Albuquerque',
                                         'Seattle', 
                                         'Denver',
                                         'Helena', 
                                         'Duluth',
                                         'Fairbanks',
                                         'McAllen',
                                         'Kodiak',
                                         'Worcester'],num_cpu=1, run_parallel=False)

# this saves your results as a pickle so that you can see what went wrong if
# you run this script as a batch script.
pkl.dump([results], open(example_dir,"example_data","study_results.pkl",'wb'))