# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:22:19 2021

Copyright Notice
=================

Copyright 2021 National Technology and Engineering Solutions of Sandia, LLC. 
Under the terms of Contract DE-NA0003525, there is a non-exclusive license 
for use of this work by or on behalf of the U.S. Government. 
Export of this program may require a license from the 
United States Government.

Please refer to the LICENSE.md file for a full description of the license
terms for MEWS. 

The license for MEWS is the Modified BSD License and copyright information
must be replicated in any derivative works that use the source code.

@author: dlvilla
"""

from mews.weather import Alter
from mews.cython import markov_chain
from mews.stats.markov import MarkovPy
from mews.stats.extreme import DiscreteMarkov
from mews.stats.extreme import Extremes
from mews.graphics.plotting2D import Graphics
from numpy.random import default_rng
import numpy as np
import unittest
import os
from matplotlib import pyplot as plt
from matplotlib import rc
from time import perf_counter_ns
import scipy as sp
from datetime import datetime
import warnings
import logging
rng = default_rng()


class Test_Extreme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = False
        cls.write_results = False
        cls.rng = default_rng()
        
        
        if not os.path.exists("data_for_testing"):
            os.chdir(os.path.join(".","mews","tests"))
            cls.from_main_dir = True
        else:
            cls.from_main_dir = False
               
        cls.test_weather_path = os.path.join(".","data_for_testing")
        erase_me_file_path = os.path.join(cls.test_weather_path,"erase_me_file.epw")
        if os.path.exists(erase_me_file_path):
            os.remove(erase_me_file_path)

        cls.test_weather_file_path = os.path.join(".",
                                                  cls.test_weather_path,
                                                  "USA_NM_Santa.Fe.County.Muni.AP.723656_TMY3.epw")
        cls.tran_mat = np.array([[0.1,0.2,0.3,0.4],[0.4,0.3,0.2,0.1],
                                 [0.2,0.3,0.4,0.1],[0.3,0.4,0.2,0.1]])
        plt.close('all')
        font = {'size':16}
        rc('font', **font)   
    
    @classmethod
    def tearDownClass(cls):
        if cls.from_main_dir:
            os.chdir(os.path.join("..",".."))
        for file_name in os.listdir():
            if ("NM_Albuquerque_Intl_ArptTMY3" in file_name or
                "USA_NM_Santa.Fe.County.Muni.AP.723656" in file_name or
                "WEATHER.FMT" in file_name or
                "TXT2BIN.EXE" in file_name or
                "INPUT.DAT" in file_name):
                    try:
                        os.remove(file_name)
                    except:
                        warnings.warn("The testing could not clean up files"
                                      +" and the tests folders have residual "
                                      +"*.bin, *.epw, *.EXE, or *.DAT files"
                                      +" that need to be removed!")

            
    def test_MarkovChain(self):
        always0 = np.array([[1,0],[1,0]],dtype=np.float)
        state0 = np.int32(0)
        rand = self.rng.random(10000)

        value = markov_chain(always0,rand,state0)
        value_py = MarkovPy.markov_chain_py(always0,rand,state0)
        np.testing.assert_array_equal(value,value_py)
        self.assertTrue(value_py.sum() == 0.0)
        
        two_state_equal = np.array([[0.5,0.5],[0.5,0.5]])
        rand = self.rng.random(100000)
        value = markov_chain(two_state_equal.cumsum(axis=1),rand,state0)
        value_py = MarkovPy.markov_chain_py(two_state_equal.cumsum(axis=1),rand,state0)
        np.testing.assert_array_equal(value,value_py)
        # in the limit, the average value should approach 0.5.
        self.assertTrue(np.abs(value.sum()/100000 - 0.5) < 0.01)
        
        always1 = np.array([[0,1],[0,1]],dtype=np.float)
        state0 = 1
        rand = self.rng.random(10000)
        
        value = markov_chain(always1,rand,state0)
        self.assertTrue(value.sum() == 10000)
        
        
        # now do a real calculation
        three_state = np.array([[0.9,0.04,0.06],[0.5,0.495,0.005],[0.5,0.005,0.495]])
        val,vec = np.linalg.eig(np.transpose(three_state))
        rand = self.rng.random(np.int(1e6))
        
        tic_c = perf_counter_ns()
        value = markov_chain(three_state.cumsum(axis=1),rand,state0)
        toc_c = perf_counter_ns()
        
        tic_py = perf_counter_ns()
        value_py = MarkovPy.markov_chain_py(three_state.cumsum(axis=1),rand,state0)
        toc_py = perf_counter_ns()
        
        self.assertAlmostEqual((value-value_py).sum(),0.0)

        ctime = toc_c - tic_c
        pytime = toc_py - tic_py

        self.assertTrue(ctime < pytime)
        
        # calculate the steady state via taking the eigenvector of eigenvalue 1
        # eig will handle a left-stochastic matrix so a transpose operation
        # is needed.
        val,vec = np.linalg.eig(np.transpose(three_state))
        steady = vec[:,0]/vec.sum(axis=0)[0]
        
        fig,axl = plt.subplots(1,2,figsize=(20,10))
        nn = axl[1].hist(value,bins=[-0.5,0.5,1.5,2.5],density=True)
        axl[1].set_ylabel("Fraction of time in state")
        axl[0].plot(value[0:1000])
        if not self.plot_results:
            plt.close(fig=fig)
        
        # verify loose convergence.
        self.assertTrue(np.linalg.norm(nn[0] - steady) < 0.01)
        
        logging.log(0,"\n\n")
        logging.log(0,"For 1e6 random numbers a 3-state Markov"
               +" Chain in python took: {0:5.3e} seconds".format(
                   (toc_py - tic_py)/1e9))
        logging.log(0,"For 1e6 random numbers a 3-state Markov"
               +" Chain in cython took: {0:5.3e} seconds".format(
                   (toc_c - tic_c)/1e9))
        
        if pytime/10 < ctime:  
            raise UserWarning("The cython markov_chain function is less than"
                              +" 10x faster than native python!")

        state0 = 1 # 2 on wikipedia
        cat_and_mouse = np.array([[0,0,0.5,0,0.5],
                                  [0,0,1,0,0],
                                  [0.25,0.25,0,0.25,0.25],
                                  [0,0,0.5,0,0.5],
                                  [0,0,0,0,1.0]],dtype=np.float)                  
        rand = self.rng.random(1000)
        
        # this is a markov process which must reach a stationary state of 4 
        # no matter what see https://en.wikipedia.org/wiki/Stochastic_matrix
        value = markov_chain(cat_and_mouse.cumsum(axis=1),rand,state0)[-1]
        
        self.assertEqual(value,4)
    
    def test_extreme_exceptions(self):
        
        # incorrect type for transition matrix with None for state_names
        obj = DiscreteMarkov(rng,self.tran_mat)
        with self.assertRaises(Exception):
            obj = DiscreteMarkov(rng,"A")
        
        # incorrect transition matrix type with specified state_names
        with self.assertRaises(TypeError):
            obj = DiscreteMarkov(rng,"Wrong!",["A"],False)
            
        # non-list for state-names
        with self.assertRaises(TypeError):
            obj = DiscreteMarkov(rng,self.tran_mat,"not a list",True)
            
        # incorrect type in state_names list
        with self.assertRaises(TypeError):
            obj = DiscreteMarkov(rng,self.tran_mat,["ok",1,"ok"],True)
        
        # incorrect type for transition matrix
        with self.assertRaises(TypeError):
            obj = DiscreteMarkov(rng,[[1,2,3],[1,2,3]],["A"],True)
        
        # incorrect type for transition matrix elements
        with self.assertRaises(TypeError):
            obj = DiscreteMarkov(rng,np.array([[1,obj,3],[1,2,3],[1,2,3]]),["A","B","C","D"],True)
            
        # non-sum to 1 rows in transition matrix
        with self.assertRaises(ValueError):
            obj = DiscreteMarkov(rng,np.array([[1,2,3],[1,2,3],[1,2,3]],dtype=np.float),["A","B","C","D"],True)
        
        with self.assertRaises(TypeError):
            obj = DiscreteMarkov(rng,self.tran_mat,["A","B","C","D"],"Not True")
        
        # test exception in methods
        obj = DiscreteMarkov(rng,self.tran_mat,["A","B","C","D"],True)
        
        # incorrect state name str
        with self.assertRaises(ValueError):
            val = obj.history(12,"A")
            obj.history(0,"Z")
        
        # incorrect state name type
        with self.assertRaises(TypeError):
            obj.history(0,np.array([1]))
            
        # incorrect state name index value
        with self.assertRaises(ValueError):
            obj.history(0,100)
        
        # incorrect run index type
        with self.assertRaises(TypeError):
            obj.history("100","A")
        
        # correct use (assure does not Raise an error)
        val = obj.history(100, "A")
        total_probabilities = obj.steady()
    
    
    def test_extreme_heat_history_EP_and_Doe2(self):
        isdoe2 = [True,False]
        col = ['DRY BULB TEMP (DEG F)','Dry Bulb Temperature']
        fname = [[os.path.join(self.test_weather_path,"NM_Albuquerque_Intl_ArptTMY3.bin")],[self.test_weather_file_path]]
        
        # test to see if DOE-2 utilities are available.
        if os.path.exists(r"../third_party_software/BIN2TXT.EXE") and os.path.exists(r"../third_party_software/TXT2BIN.EXE"):
            testDoe2 = True
        else:
            testDoe2 = False
            warnings.warn("The DOE-2 features of MEWS cannot be tested"
                          +" because the 'third_part_software' folder does not"
                          +" contain 'BIN2TXT.EXE' and 'TXT2BIN.EXE.' These must"
                          +" be obtained from James Hirsch and Associates "
                          +" who can be contacted through www.doe2.com"
                          +". The utilities are free but require a separate"
                          +" license agreement.")
            
        
        for ido,wfiles,column in zip(isdoe2,fname,col):
            
            mu_integral = 0.5 # three day average heat wave with 2C average 
                                     # higher temperature C*hr/hr
            sig_integral = 0.5  #
            
            mu_min = 1
            mu_max = 1.2
            sig_min = 1
            sig_max = 1
            
            p_ee = 1e-2 
            
            # this matrices rows must add to one
            trans_mat = np.array([[1.0-2*p_ee,p_ee,p_ee],[0.1,0.9,0.0],[0.1,0.0,0.9]])
            # this matrices columns must add to zero
            trans_mat_delta = np.array([[p_ee/10,-p_ee/20,-p_ee/20],[-0.0001,0.00005,0.00005],[-0.0001,0.00005,0.00005]])
 
            max_avg_delta = 1.0
            min_avg_delta = -1.0
            min_delta = -1.0
            max_delta = 2.0
            
            num_real = 2 # 2 different realizations of the random history 
                          # (i.e. 2 years into the future)
            num_repeat = 3 # number of times the weather file is repeated to form a history.
            
            year = 2021
            
            if ido:
                if testDoe2:
                    doe_in = {'doe2_hour_in_file':8760,
                              'doe2_start_datetime':datetime(year,1,1,1,0,0,0),
                              'doe2_dst':[datetime(year,3,14,2,0,0,0), datetime(year,11,7,2,0,0,0)],
                              'doe2_tz':"America/Denver"}
                else:
                    continue
            else:
                doe_in = None
                
            obj = Extremes(2021, max_avg_dist, max_avg_delta, min_avg_dist, 
                           min_avg_delta, trans_mat, trans_mat_delta, wfiles, num_real, 
                           num_repeat=num_repeat,write_results=self.write_results,
                           run_parallel=False,test_shape_func=True,doe2_input=doe_in,
                           column=column)
            
            
            obj = Extremes(2021,  max_avg_dist, max_avg_delta, min_avg_dist, 
                           min_avg_delta, trans_mat, trans_mat_delta, wfiles, num_real, 
                           num_repeat=num_repeat,write_results=self.write_results,
                           run_parallel=True,test_shape_func=True,doe2_input=doe_in,
                           column=column)
            
            obj = Extremes(2021,  max_avg_dist, max_avg_delta, min_avg_dist, 
                           min_avg_delta, trans_mat, trans_mat_delta, wfiles, num_real, 
                           num_repeat=num_repeat,write_results=True,
                           run_parallel=True,test_shape_func=False,doe2_input=doe_in,
                           column=column)
        # visualize
    
            res = obj.results
            
            fig,axl = plt.subplots(7,1,sharex=True,sharey=True,figsize=(10,10))
            
            for realization_number,ax in enumerate(axl):
                if realization_number == len(axl):
                    legend_labels = ("extreme","normal")
                Graphics.plot_realization(res,column,realization_number,ax=ax,legend_labels=("extreme","normal"))
                ax.set_ylabel('R{0:d}'.format(realization_number))
            
            if column == "Dry Bulb Temperature":
                figtxt1 = "Dry Bulb Temperature (${^\\circ}C$)"
            else:
                figtxt1 = column
            
            fig.text(0.01, 0.5, figtxt1, va='center', rotation='vertical')
            
            plt.legend(bbox_to_anchor=(1.1,-0.8),ncol=5)
            plt.tight_layout()
            
            if not self.plot_results:
                plt.close('all')
        
    
    

def max_avg_dist(size):
    mu_integral = 0.5 # three day average heat wave with 2C average 
                             # higher temperature C*hr/hr
    sig_integral = 0.5  #
    return rng.lognormal(mu_integral,sig_integral,size)
    
def min_avg_dist(size):
    mu_integral = 0.5 # three day average heat wave with 2C average 
                             # higher temperature C*hr/hr
    sig_integral = 0.5  #
    return -rng.lognormal(mu_integral,sig_integral,size)
            
        
            
        
            
        
        
            
if __name__ == "__main__":
    o = unittest.main(Test_Extreme())