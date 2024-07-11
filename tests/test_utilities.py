#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:14:55 2022

@author: dlvilla

Copyright Notice
=================

Copyright 2023 National Technology and Engineering Solutions of Sandia, LLC. 
Under the terms of Contract DE-NA0003525, there is a non-exclusive license 
for use of this work by or on behalf of the U.S. Government. 
Export of this program may require a license from the 
United States Government.

Please refer to the LICENSE.md file for a full description of the license
terms for MEWS. 

The license for MEWS is the Modified BSD License and copyright information
must be replicated in any derivative works that use the source code.

"""


from mews.utilities.utilities import (histogram_intersection, 
                                      histogram_non_intersection, 
                                      histogram_non_overlapping,
                                      histogram_step_wise_integral,
                                      write_readable_python_dict,
                                      read_readable_python_dict,
                                      dict_key_equal)
from numpy.random import default_rng
from matplotlib import pyplot as plt
from mews.utilities.utilities import find_extreme_intervals

import numpy as np
import unittest
import os
from warnings import warn

class Test_SolveDistributionShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # clean this up HOW MUCH of this from Test_Alter is needed?
        cls.plot_results = True
        cls.write_results = False
        cls.rng = default_rng(23985)
        plt.close('all')
        
        
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    
    def test_read_and_write_of_readable_python_dict(self):
        ex_dict = {'cs': {'help': 'These statistics are already mapped from -1 ... 1 and _inverse_transform_fit is needed to return to actual degC and degC*hr values. If input of actual values is desired use transform_fit(X,max,min)', 'energy_normal_param': {'mu': 0.0038899912491902897, 'sig': 0.4154535225139539}, 'extreme_temp_normal_param': {'mu': 0.061888990278578374, 'sig': 0.690956053333432}, 'max extreme temp per duration': 1.9012564475129312, 'min extreme temp per duration': 0.12866200452373405, 'max energy per duration': 1.3562919483136582, 'min energy per duration': 0.3990299129968476, 'energy linear slope': 0.9768230669124248, 'normalized extreme temp duration fit slope': 0.6946813967764691, 'normalized extreme temp duration fit intercept': 0.46494070822925054, 'normalizing energy': -3775.333333333333, 'normalizing extreme temp': -25.3125, 'normalizing duration': 384, 'historic time interval': 650040.0, 'hourly prob stay in heat wave': 0.9623388098739797, 'hourly prob of heat wave': 0.004296022992350157, 'historical durations (durations0)': (np.array([53, 59, 37, 19,  9,  3,  2,  3,  1,  0,  0,  0,  0,  0,  0,  1]), np.array([ 12.,  36.,  60.,  84., 108., 132., 156., 180., 204., 228., 252.,
               276., 300., 324., 348., 372., 396.])), 'historical temperatures (hist0)': 
                          (np.array([ 1,  0,  0,  2,  6,  2,  7, 13, 14, 11, 25, 22, 29, 20, 25,  6,  2,
                1,  1]), np.array([-25.78947368, -24.78531856, -23.78116343, -22.77700831,
               -21.77285319, -20.76869806, -19.76454294, -18.76038781,
               -17.75623269, -16.75207756, -15.74792244, -14.74376731,
               -13.73961219, -12.73545706, -11.73130194, -10.72714681,
                -9.72299169,  -8.71883657,  -7.71468144,  -6.71052632])), 
                                   'decay function': 'quadratic_times_exponential_decay_with_cutoff', 
                                   'decay function coef': np.array([6.92547674e+02, 9.79605373e-01, 1.10730892e+03])}, 
                                   'hw': {'help': 'These statistics are already mapped from -1 ... 1 and _inverse_transform_fit'+
            ' is needed to return to actual degC and degC*hr values. If input of actual values is desired use transform_fit(X,max,min)',
            'energy_normal_param': {'mu': 0.07612295120388764, 'sig': 0.26004106669177984}, 'extreme_temp_normal_param': {'mu': 
        -0.3279009953134448, 'sig': 0.38427114502670034}, 'max extreme temp per duration': 2.2318874862128877, 'min extreme temp per duration':
                0.3813481535873679, 'max energy per duration': 1.8468741653468745, 'min energy per duration': -0.284239504158205, 'energy linear slope': 0.7404780352640733, 'normalized extreme temp duration fit slope': 0.4823740200889345, 'normalized extreme temp duration fit intercept': 0.45709478834728723, 'normalizing energy': 1648.666666666667, 'normalizing extreme temp': 23.157407407407405, 'normalizing duration': 144, 'historic time interval': 650040.0, 'hourly prob stay in heat wave': 0.960346272307918, 'hourly prob of heat wave': 0.005757951778158901, 'historical durations (durations0)': (np.array([92, 49, 14,  6,  2,  1]), np.array([ 12.,  36.,  60.,  84., 108., 132., 156.])), 'historical temperatures (hist0)': (np.array([ 1,  2,  2, 11, 35, 19, 18, 19, 13, 17, 10,  8,  6,  0,  2,  0,  1]), np.array([ 6.57244009,  7.57590029,  8.5793605 ,  9.58282071, 10.58628092,
               11.58974113, 12.59320133, 13.59666154, 14.60012175, 15.60358196,
               16.60704216, 17.61050237, 18.61396258, 19.61742279, 20.62088299,
               21.6243432 , 22.62780341, 23.63126362])), 'decay function': 'quadratic_times_exponential_decay_with_cutoff', 
                'decay function coef': np.array([229.41181563,   0.98662924, 406.27566607])}}
        
        write_readable_python_dict("test.txt",ex_dict)
        
        my_new_dict = read_readable_python_dict("test.txt")
        
        try:
            dict_key_equal(ex_dict,my_new_dict)
        except:
            self.fail("The output dictionaries are not the same!")

        
        

                                   
    def test_histogram_step_wise_integral(self):
        hist = (np.array([1,2,3,2,1]),np.array([0,1,2,3,4,5]))
        area = histogram_step_wise_integral(hist,0.7,4.2)
        self.assertAlmostEqual(area, 7.5)
        
        # case where b is only on the first bin
        area = histogram_step_wise_integral(hist,-3,0.1)
        self.assertAlmostEqual(area, 0.1)
        
        # case where a is only on the last bin
        area = histogram_step_wise_integral(hist,4.9,10.0)
        self.assertAlmostEqual(area,0.1)
        
        # case where no overlap after
        area = histogram_step_wise_integral(hist,5.0,10.0)
        self.assertAlmostEqual(area,0.0)
        
        # case where off before 
        area = histogram_step_wise_integral(hist,-3.0,-0.1)
        self.assertAlmostEqual(area,0.0)
        
        # case where a off but b is in
        area = histogram_step_wise_integral(hist,-3.0,2.1)
        self.assertAlmostEqual(area,3.3)
        
        # case where a on and b is off
        area = histogram_step_wise_integral(hist,2.5,100.0)
        self.assertAlmostEqual(area,4.5)
            
        # incorrect integration order error
        with self.assertRaises(ValueError):
            area = histogram_step_wise_integral(hist,2.5,-100.0)
            
        # subsumed case
        area = histogram_step_wise_integral(hist,-2.5,100.0)
        self.assertAlmostEqual(area,9.0)
        
    
    def test_histogram_non_intersection(self):
        hist1 = (np.array([1,2,3,2,1]),np.array([0,1,2,3,4,5]))
        hist2 = (np.array([1,2,3,2,1]),np.array([0,1,2,3,4,5]))
        # completely overlapping
        
        area = histogram_non_intersection(hist1,hist2)
        self.assertAlmostEqual(area, 0.0)
        
        # no overlap
        hist2 = (np.array([1,2,3,2,1]),np.array([10,11,12,13,14,15]))    
        area = histogram_non_intersection(hist1,hist2)
        self.assertAlmostEqual(area, 16.0)
        
        # partial overlap - slightly different because of intersection
        # area.
        # remember discrete histogram integration here. Trapezoidal 
        # integration of midpoints is not included. If it were,
        # the answer would be 13.5.
        hist2 = (np.array([1,2,3,2,1]),np.array([3,4,5,6,7,8]))    
        area = histogram_non_intersection(hist1,hist2)
        self.assertAlmostEqual(area, 14.0)
    
    
    def test_histogram_non_overlapping(self):
        hist1 = (np.array([1,2,3,2,1]),np.array([0,1,2,3,4,5]))
        hist2 = (np.array([1,2,3,2,1]),np.array([0,1,2,3,4,5]))
        # completely overlapping
        
        area = histogram_non_overlapping(hist1,hist2)
        self.assertAlmostEqual(area, 0.0)
        
        # no overlap
        hist2 = (np.array([1,2,3,2,1]),np.array([10,11,12,13,14,15]))    
        area = histogram_non_overlapping(hist1,hist2)
        self.assertAlmostEqual(area, 16.0)
        
        # partial overlap
        hist2 = (np.array([1,2,3,2,1]),np.array([3,4,5,6,7,8]))    
        area = histogram_non_overlapping(hist1,hist2)
        self.assertAlmostEqual(area, 13.0)
                
        
    
    def test_histogram_intersection(self):
        
        hist1 = [np.array([ 2,  2,  0,  1,  1,  3, 10,  7,  4, 10, 12, 15, 14, 21, 26, 35, 14,
                 13,  1,  1]),
          np.array([-23.71429398, -22.88024884, -22.0462037 , -21.21215856,
                 -20.37811343, -19.54406829, -18.71002315, -17.87597801,
                 -17.04193287, -16.20788773, -15.37384259, -14.53979745,
                 -13.70575231, -12.87170718, -12.03766204, -11.2036169 ,
                 -10.36957176,  -9.53552662,  -8.70148148,  -7.86743634,
                  -7.0333912 ])]
        hist1[0] = hist1[0]/hist1[0].sum()
        hist1 = tuple(hist1)
        
        hist2 = (np.array([0.00014633, 0.        , 0.        , 0.        , 0.        ,
               0.00014633, 0.00029265, 0.00043898, 0.00029265, 0.00073164,
               0.00029265, 0.00043898, 0.00087796, 0.00131694, 0.00146327,
               0.00058531, 0.00131694, 0.00204858, 0.00234124, 0.00365818,
               0.00263389, 0.00380451, 0.00395083, 0.00775534, 0.00760901,
               0.00819432, 0.01126719, 0.01126719, 0.01360843, 0.01551068,
               0.01785192, 0.01726661, 0.02107112, 0.02472929, 0.02107112,
               0.02633889, 0.02999707, 0.03424056, 0.03599649, 0.03585016,
               0.03263096, 0.03775241, 0.03687445, 0.03848405, 0.03892303,
               0.03804507, 0.03628914, 0.03906936, 0.03599649, 0.03380158,
               0.03292362, 0.03102136, 0.02341235, 0.02238806, 0.02429031,
               0.01931519, 0.01799824, 0.01712028, 0.01112087, 0.01419374,
               0.01082821, 0.00907229, 0.00541411, 0.00702371, 0.00541411,
               0.00380451, 0.00424349, 0.00248756, 0.00102429, 0.00117062,
               0.00102429, 0.00043898]), np.array([-62.91441556, -62.06878646, -61.22315736, -60.37752826,
               -59.53189916, -58.68627006, -57.84064096, -56.99501186,
               -56.14938276, -55.30375366, -54.45812456, -53.61249546,
               -52.76686636, -51.92123726, -51.07560816, -50.22997906,
               -49.38434996, -48.53872086, -47.69309176, -46.84746266,
               -46.00183356, -45.15620446, -44.31057536, -43.46494626,
               -42.61931716, -41.77368806, -40.92805896, -40.08242986,
               -39.23680076, -38.39117166, -37.54554256, -36.69991346,
               -35.85428436, -35.00865526, -34.16302616, -33.31739706,
               -32.47176796, -31.62613886, -30.78050976, -29.93488066,
               -29.08925156, -28.24362246, -27.39799336, -26.55236426,
               -25.70673516, -24.86110606, -24.01547696, -23.16984786,
               -22.32421876, -21.47858966, -20.63296056, -19.78733146,
               -18.94170236, -18.09607326, -17.25044416, -16.40481506,
               -15.55918596, -14.71355686, -13.86792776, -13.02229866,
               -12.17666956, -11.33104046, -10.48541136,  -9.63978226,
                -8.79415316,  -7.94852406,  -7.10289496,  -6.25726586,
                -5.41163676,  -4.56600766,  -3.72037856,  -2.87474946,
                -2.02912036]))
                                                
        area = histogram_intersection(hist1,hist2)   
        farea = histogram_non_intersection(hist1,hist2)
        self.assertAlmostEqual(area, 0.1897621324044904)
        self.assertAlmostEqual(farea, 1.2933865021655309)                    
        
        histzero1 = (np.array([1,1,1]),np.array([1,2,3,4]))
        histzero2 = (np.array([2,3,4]),np.array([-7,-6,-5,-4]))
        
        area0 = histogram_intersection(histzero1,histzero2)
        areafull = histogram_non_intersection(histzero1,histzero2)
        
        self.assertTrue(area0 == 0.0)
        self.assertTrue(areafull == 8.0)
        

if __name__ == "__main__":
    profile = False
    
    if profile:
        import cProfile
        import pstats
        import io
        
        pr = cProfile.Profile()
        pr.enable()
        
    o = unittest.main(Test_SolveDistributionShift())
    
    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        
        with open('utilities_test_profile.txt', 'w+') as f:
            f.write(s.getvalue())