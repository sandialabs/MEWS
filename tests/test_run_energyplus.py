#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:44:00 2023

@author: dlvilla
"""

import os
import numpy as np
from mews.run_energyplus import RunEnergyPlusStudy
import unittest
import shutil


class Test_Run_EnergyPlus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # this test is not usually run unless a valid path to energyplus is
        # supplied
        energyplus_path = os.path.join(
            "/ascldap",
            "users",
            "dlvilla",
            "EnergyPlus",
            "EnergyPlus-22.2.0",
            "build",
            "Products",
            "energyplus",
        )
        if os.path.exists(energyplus_path):
            cls.run_tests = True
        else:
            cls.run_tests = False
        cls.energyplus_path = energyplus_path

        cls.example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        cls.out_path = os.path.join(cls.example_dir, "tmp_out_test_run_energyplus")
        if os.path.exists(cls.out_path):
            shutil.rmtree(cls.out_path)
        os.mkdir(cls.out_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out_path)

    def test_run_energyplus(self):

        if self.run_tests:

            idfs_path = os.path.join(self.example_dir, "data_for_testing", "idfs_path")
            epws_path = os.path.join(self.example_dir, "data_for_testing", "epws_path")

            # IN ORDER FOR THIS TEST TO RUN, YOU MUST HAVE ACCESS TO ENERGYPLUS!
            # with the pyenergyplus API folder being the root directory
            # or, if the API is not available, this must be the path to an
            # executable.

            obj_run_ep = RunEnergyPlusStudy(
                epws_path,
                idfs_path,
                self.energyplus_path,
                self.out_path,
                restart=True,
                run_parallel=False,
            )
            # This should not reconduct the study.
            obj_run_ep = RunEnergyPlusStudy(
                epws_path,
                idfs_path,
                self.energyplus_path,
                self.out_path,
                restart=False,
                run_parallel=False,
            )

            # This will reconduct the study.
            obj_run_ep = RunEnergyPlusStudy(
                epws_path,
                idfs_path,
                self.energyplus_path,
                self.out_path,
                restart=True,
                run_parallel=True,
                num_cpu=2,
            )

            for key, val in obj_run_ep.results.items():
                if isinstance(val, Exception):
                    raise val


if __name__ == "__main__":
    profile = False

    if profile:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()

    o = unittest.main(Test_Run_EnergyPlus())

    if profile:

        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()

        with open("utilities_test_profile.txt", "w+") as f:
            f.write(s.getvalue())
