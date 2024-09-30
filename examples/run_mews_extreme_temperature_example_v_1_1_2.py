#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 2024

@author: dlvilla

IF YOU JUST WANT TO GENERATE FILES:

You must download the solution files in https://osf.io/ts9e8/files/osfstorage  Solution_File_Results
or else the needed weather files and/or solution files (if you want to skip steps) will not be available.
This is the way to run mews. All the names available in the existing dataset are

    "Chicago",
    "Baltimore",
    "Minneapolis",
    "Phoenix",
    'Miami',
    'Houston'
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
    'Worcester'

Any setting for Chicago applies for every city unless a command is repeated with 
a new value. Some of the cities have to repeat different variables because of 
differences in units.

If you want to rerun an optimization (takes a full day on 60 processors for 
the machines used so far), then do not include a city in neither of "skip_runs" 
"only_generate_files". If you only want to generate files (much shorter run time
in a matter of hours, then you only need to ) 

IF YOU WANT TO CREATE A COMPLETELY NEW ANALYSIS:

Then you will need to obtain an epw file (or files and add to or create a 
new /examples/example_data/mews_run.dict) file. 

since I could not finish documentation, Your best shot is to 
watch the video at:

https://drive.google.com/file/d/1B-G5yGu0BFXCqj0BYfu_e8XFliAoeoRi/view?usp=drive_link

"""


from time import time
start_time = time()


import os
import pickle as pkl
import numpy as np

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
run_dict_file_path = os.path.join(os.path.dirname(__file__),"example_data","mews_run_only1file.dict")

# ONLY SET TO TRUE IF YOU WANT TO SEE HOW LONG DIFFERENT FUNCTIONS IN MEWS TAKE
profile = False

if profile:
    import cProfile
    import pstats
    import io
    
    pr = cProfile.Profile()
    pr.enable()
    





results = extreme_temperature(run_dict=run_dict_file_path,
                              only_generate_files=[],
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
                                         'Worcester',
                                         'Houston'],num_cpu=1, run_parallel=False,
                              run_dict_var={"example_dir":example_dir})
if profile:

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    
    with open('extreme_temperature_profile_3.txt', 'w+') as f:
        f.write(s.getvalue())

# this saves your results as a pickle so that you can see what went wrong if
# you run this script as a batch script.

end_time = time()
results['run_time in seconds'] = end_time - start_time
pkl.dump([results], open(os.path.join(example_dir,"example_data","study_results.pkl"),'wb'))
