#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:20:49 2022

@author: dlvilla
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:12:15 2022

@author: dlvilla
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:07:57 2021

@author: dlvilla
"""

from mews.cython import markov_chain
from mews.stats.markov import MarkovPy

from mews.stats.markov_time_dependent import markov_chain_time_dependent_py
from mews.stats.extreme import DiscreteMarkov
from numpy.random import default_rng 
import os
import numpy as np
import unittest  
from matplotlib import pyplot as plt
from matplotlib import rc 
from mews.cython.markov_time_dependent import markov_chain_time_dependent
from time import perf_counter_ns
import logging
    

class Test_Markov(unittest.TestCase):
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
        if cls.plot_results:
            plt.close('all')
            font = {'size':16}
            rc('font', **font)   
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    
    def test_time_dependent(self):
        
        random_seed = 22305982
        rand_big = default_rng(random_seed).random(100000)
        tran_matrix = np.array([[0.9,.02,.08],
                                [0.02,0.98,0.0],
                                [0.005,0.0,0.995]])
        cdf = tran_matrix.cumsum(axis=1)

        # test exponential decay where exponent is so small that this should produce a constant response
        coef_negligible_1term = np.array([[1e-10],[1e-10]])
        coef_1term = np.array([[0.1],[0.1]])
        
        # these are either a lambda for exponential or slope for linear decay
        # for the first term and a maximum steps term for the second.
        coef_negligible_2terms = np.array([[1e-10,10000],[1e-10,10000]])
        coef_2terms = np.array([[0.1,9],[0.1,9]]) # should limit steps in a state to 9
        
        
        # loop over python verses cython implementations
        for markov_func in [markov_chain_time_dependent_py, markov_chain_time_dependent]:
            yy_const = markov_chain(cdf, rand_big[0:1000], 0)
            for func_type in range(4):
                if func_type < 2:
                    coef_neg = coef_negligible_1term
                    coef = coef_1term
                else:
                    coef_neg = coef_negligible_2terms
                    coef = coef_2terms
                    
                yy_py_negligible = markov_func(cdf, rand_big[0:1000], 0, coef_neg, func_type)
                
                yy_py = markov_func(cdf, rand_big[0:1000], 0, coef, func_type)
                
                # TEST 1 - test negligible decay produces a constant markov process
                if (yy_py_negligible == yy_const).all()==False:
                    breakpoint()
                self.assertTrue((yy_py_negligible == yy_const).all())    
    
                # TEST 2 - non-negligible decay does not produce a constant Markov process
                #          and linear and exponential decay produce different results
                self.assertFalse((yy_py == yy_const).all())
        
        
        pass
    
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
        

        
if __name__ == "__main__":
    o = unittest.main(Test_Markov())
