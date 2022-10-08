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

from mews.stats.solve import SolveDistributionShift
from mews.stats.extreme import DiscreteMarkov
from numpy.random import default_rng



def example_markov_stat_model(param, random_seed, num_time_step=8760*100):
    
    rng = default_rng(random_seed)
    
    transition_matrix = np.array([[1-.001-.002,.001,.002],
                                  [1-0.98, 0.98, 0.0],
                                  [1-0.985, 0.0, 0.985]])
    
    objDM = DiscreteMarkov(rng, transition_matrix)
    
    
    
    

class Test_ExtremeTemperatureWaves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = False
        cls.write_results = False
        cls.rng = default_rng()
        

    
    @classmethod
    def tearDownClass(cls):
        pass
    
    
    def test_residuals(self):
        
        objDM = DiscreteMarkov
        
        SolveDistributionShift(markov_stat_model, param0, hist0)
        
    def _example        
        

        
if __name__ == "__main__":
    o = unittest.main(Test_ExtremeTemperatureWaves())
