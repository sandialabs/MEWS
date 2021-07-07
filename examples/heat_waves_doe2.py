# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:47:59 2021

@author: dlvilla
"""

from mews.stats.extreme import Extremes
from mews.graphics.plotting2D import plot_realization
from numpy.random import default_rng
from matplotlib import pyplot as plt
from matplotlib import rc
from datetime import datetime
import numpy as np
import os

def convert_axis(ax,axtw,conv_func,is_x=False):
    if is_x:
        x1,x2 = ax.get_xlim()
        axtw.set_xlim(conv_func(x1),conv_func(x2))        
    else:
        y1,y2 = ax.get_ylim()
        axtw.set_ylim(conv_func(y1),conv_func(y2))
    axtw.figure.canvas.draw()



# graphics stuff
plt.close('all')
font = {'size':17}
rc('font', **font)  




mu_integral = 0.5 # three day average heat wave with 2C average 
                         # higher temperature C*hr/hr
sig_integral = 0.5  #

mu_min = 1
mu_max = 1.2
sig_min = 1
sig_max = 1

p_ee = 1e-2 

# DO NOT CHANGE THIS IT IS WHAT MAKES THE CALCULATION REPEATABLE FOR THE PUBLICATION
random_seed = 434186856231

# setup random number generation
rng = default_rng(seed=random_seed)

# define statistical distributions for heat waves (arbitrary here):
# 
def max_avg_dist(size):
    mu_integral = 1.8 # three day average heat wave with 2C average 
                             # higher temperature C*hr/hr
    sig_integral = 0.5  #
    return rng.lognormal(mu_integral,sig_integral,size)
    
def min_avg_dist(size):
    mu_integral = 1.0 # three day average heat wave with 2C average 
                             # higher temperature C*hr/hr
    sig_integral = 0.5 #
    return -rng.lognormal(mu_integral,sig_integral,size)


# this matrices rows must add to one - gives equal preference to hot and cold waves
trans_mat = np.array([[1.0-2*p_ee,p_ee,p_ee],[0.15,0.85,0.0],[0.15,0.0,0.85]])

# this matrices columns must add to zero - it represents shifting of probabilities
# between heat waves, cold waves, and normal weather.
trans_mat_delta = np.array([[p_ee/100,-p_ee/200,-p_ee/200],[-0.0001,0.00005,0.00005],[-0.0001,0.00005,0.00005]])

wfiles = [os.path.join(".","example_data","NM_Albuquerque_Intl_ArptTMY3.bin")]

 
max_avg_delta = 1.0   # increase in C*hr/hr per year of heat waves
min_avg_delta = -1.0  # decrease in C*hr/hr per year of cold waves

# 100 different realizations of the random history 
num_real = 3
year = 2021

# 25 years of history created (climate change is not included in this example)
num_repeat = 1

doe_in = {'doe2_hour_in_file':8760,
          'doe2_start_datetime':datetime(year,1,1,1,0,0,0),
          'doe2_dst':[datetime(year,3,14,2,0,0,0), datetime(year,11,7,2,0,0,0)],
          'doe2_tz':"America/Denver"}

obj = Extremes(year, max_avg_dist, max_avg_delta, min_avg_dist, 
               min_avg_delta, trans_mat, trans_mat_delta, wfiles, num_real, 
               num_repeat=num_repeat,write_results=True,
               run_parallel=False,test_shape_func=False,
               results_folder="heat_waves_output_doe2",
               doe2_input=doe_in,column='DRY BULB TEMP (DEG F)',
               random_seed=random_seed)

# visualize the resulting files

res = obj.results

fig,axl = plt.subplots(3,1,sharex=True,sharey=True,figsize=(10,10))

def F_to_C(F):
    return 5/9 * (F - 32)

for realization_number,ax in enumerate(axl):
    if realization_number == len(axl):
        legend_labels = ("extreme","normal")
    plot_realization(res,'DRY BULB TEMP (DEG F)',realization_number,ax=ax,legend_labels=("extreme","TMY3"))
    ax.set_title('R{0:d}'.format(realization_number))
    axtwy = ax.twinx()
    ax.callbacks.connect("ylim_changed",convert_axis(ax,axtwy,lambda x:F_to_C(x)))
    if realization_number == 1:
        axtwy.set_ylabel("Dry Bulb Temperature (${^\\circ}C$)")

axl[1].set_ylabel("Dry Bulb Temperature (${^\\circ}F$)")
axl[1].legend()
#plt.legend()
plt.tight_layout()
plt.savefig('',dpi=300,'Albuquerque_TMY3_weather_realizations.png')