# -*- coding: utf-8 -*-
"""
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

This code comes from a third party open-source library. The license is 
repeated below per the terms of redistribution.

EPW License
===========

MIT License

Copyright (c) 2019 Building Energy Research Group (BERG)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import unittest
from mews.epw import epw
import pandas as pd
import os

class TestEpw(unittest.TestCase):
    
# OBJECT CREATION
    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))
        
    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        
    
    def test_epw___init__(self):
        a=epw()
        self.assertIsInstance(a,epw)


    def test_epw_read(self):
        a=epw()
        a.read(r'data_for_testing/new_epw_file.epw')
        

    def test_modify(self):
        a=epw()
        a.read(r'data_for_testing/new_epw_file.epw')
        
        a.headers['LOCATION'][0]='New_location'

        a.dataframe['Dry Bulb Temperature']=a.dataframe['Dry Bulb Temperature']+1.0
        
        a.write(r'data_for_testing/new_epw_file_2.epw')

if __name__=='__main__':
    
    o=unittest.main(TestEpw())  
    