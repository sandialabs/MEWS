# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:43:12 2021

@author: dlvilla
"""

from mews.stats.extreme import Extremes
from mews.graphics.plotting2D import plot_realization
from numpy.random import default_rng
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import os

# setup random number generation
rng = default_rng()

# graphics stuff
plt.close('all')
font = {'size':16}
rc('font', **font)  

# define statistical distributions for heat waves (arbitrary here):
# 
def max_avg_dist(size):
    mu_integral = 0.7 # three day average heat wave with 2C average 
                             # higher temperature C*hr/hr
    sig_integral = 0.5  #
    return rng.lognormal(mu_integral,sig_integral,size)
    
def min_avg_dist(size):
    mu_integral = 0.5 # three day average heat wave with 2C average 
                             # higher temperature C*hr/hr
    sig_integral = 0.5  #
    return -rng.lognormal(mu_integral,sig_integral,size)


mu_integral = 0.5 # three day average heat wave with 2C average 
                         # higher temperature C*hr/hr
sig_integral = 0.5  #

mu_min = 1
mu_max = 1.2
sig_min = 1
sig_max = 1

p_ee = 1e-2 

# this matrices rows must add to one - gives equal preference to hot and cold waves
trans_mat = np.array([[1.0-2*p_ee,p_ee,p_ee],[0.15,0.85,0.0],[0.15,0.0,0.85]])

# this matrices columns must add to zero - it represents shifting of probabilities
# between heat waves, cold waves, and normal weather.
trans_mat_delta = np.array([[p_ee/100,-p_ee/200,-p_ee/200],[-0.0001,0.00005,0.00005],[-0.0001,0.00005,0.00005]])

wfiles = [os.path.join(".","example_data","USA_NM_Santa.Fe.County.Muni.AP.723656_TMY3.epw")]

 
max_avg_delta = 1.0   # increase in C*hr/hr per year of heat waves
min_avg_delta = -1.0  # decrease in C*hr/hr per year of cold waves

# 100 different realizations of the random history 
num_real = 100

# 25 years of history created (climate change is not included in this example)
num_repeat = 25

obj = Extremes(2021, max_avg_dist, max_avg_delta, min_avg_dist, 
               min_avg_delta, trans_mat, trans_mat_delta, wfiles, num_real, 
               num_repeat=num_repeat,write_results=True,
               run_parallel=True,test_shape_func=False,results_folder="heat_waves_output")

# visualize the resulting files

res = obj.results

fig,axl = plt.subplots(7,1,sharex=True,sharey=True,figsize=(10,10))

for realization_number,ax in enumerate(axl):
    if realization_number == len(axl):
        legend_labels = ("extreme","normal")
    plot_realization(res,"Dry Bulb Temperature",realization_number,ax=ax,legend_labels=("extreme","normal"))
    ax.set_ylabel('R{0:d}'.format(realization_number))

fig.text(0.01, 0.5, "Dry Bulb Temperature (${^\\circ}C$)", va='center', rotation='vertical')

plt.legend(bbox_to_anchor=(1.1,-0.8),ncol=5)
plt.tight_layout()

