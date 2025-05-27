#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:22:21 2022

@author: dlvilla


This is a script for applying the MEWS model to Worcester,MA

It is accomplished through 3 steps that a are manually run and adjusted

rather than through a single run. Start by setting "step = 1".

STEP1 - set "step = 1"

"""
import os
import numpy as np
from numpy import poly1d
from mews.weather.climate import ClimateScenario
from mews.events import ExtremeTemperatureWaves
import pickle as pkl
from mews.utilities.utilities import bin_avg
from mews.stats.solve import shift_a_b_of_trunc_gaussian
from copy import deepcopy


if __name__ == "__main__":
    """
    INPUT DATA

    adjust all of these to desired values if analyzing a new location. Paths have
    to be set and folders created if functioning outside of the MEWS repository
    structure.



    """

    step = 3
    # STEP 1 to using MEWS, create a climate increase in surface temperature
    #        set of scenarios,
    #
    # you must download https://osf.io/ts9e8/files/osfstorage or else
    # endure the process of all the CMIP 6 files downloading (I had to restart ~100 times)
    # with proper proxy settings.

    # HERE CS stands for climate scenario (not cold snap)
    future_years = [
        2025,
        2050,
        2075,
        2100,
    ]  # [2020,2040,2060,2080] #2014, 2020, 2040, 2060,

    # CI interval valid = ['5%','50%','95%']
    ci_intervals = ["5%", "50%", "95%"]

    lat = 29.9902
    lon = 360 - 95.3368

    # Station - Houston
    station = os.path.join("example_data", "ClimateZone2A_Houston", "Houston.csv")

    first_try_solution_location = os.path.join(
        "example_data",
        "ClimateZone2A_Houston",
        "results",
        "houston_historical_solution.txt",
    )

    # change this to the appropriate unit conversion (5/9, -(5/9)*32) is for F going to C
    unit_conversion = (5 / 9, -(5 / 9) * 32)
    unit_conv_norms = (5 / 9, -(5 / 9) * 32)

    # gives consistency in run results
    random_seed = 564489

    num_realization_per_scenario = 10

    # plot_results
    plot_results = True
    run_parallel = True
    num_cpu = 30

    # No need to input "historical"
    # valid names for scenarios are: ["historical","SSP119","SSP126","SSP245","SSP370","SSP585"]
    scenarios = ["SSP245", "SSP370", "SSP585"]
    output_folder_cs = os.path.join(
        "example_data", "ClimateZone2A_Houston", "climate_scenario_results"
    )

    weather_files = [
        os.path.join(
            "example_data",
            "ClimateZone2A_Houston",
            "USA_TX_Houston-Bush.Intercontinental.AP.722430_TMY3.epw",
        )
    ]

    """
    STEP 1
    
    Calculate the polynomials for Global Warming based on the selected CMIP6 ensemble.
    
    """

    if step == 1:
        cs_obj = ClimateScenario(
            use_global=False,
            lat=lat,
            lon=lon,
            end_year=future_years[-1],
            output_folder=output_folder_cs,
            write_graphics_path=output_folder_cs,
            num_cpu=num_cpu,
            run_parallel=run_parallel,
            model_guide=os.path.join(
                os.path.dirname(__file__), "example_data", "Models_Used_alpha.xlsx"
            ),
            data_folder=os.path.join("..", "..", "CMIP6_Data_Files"),
            align_gcm_to_historical=True,
            polynomial_order={"historical": 7, "SSP585": 4, "SSP245": 4, "SSP370": 4},
        )
        scen_dict = cs_obj.calculate_coef(scenarios)

        if not os.path.exists("temp_pickles"):
            os.mkdir("temp_pickles")
        pkl.dump(
            [cs_obj], open(os.path.join("temp_pickles", "cs_obj_houston.pickle"), "wb")
        )

    else:
        # this is the results obtained by dlvilla 12/13/2022
        scen_dict = {
            "historical": poly1d(
                [
                    -2.24978562e-12,
                    -7.51866322e-10,
                    -3.87159360e-08,
                    1.17025265e-05,
                    1.74926157e-03,
                    8.59155012e-02,
                    2.86358606e-05,
                ]
            ),
            "SSP585": poly1d(
                [-1.67479380e-06, 4.36826716e-04, 3.16965065e-02, -1.09939702e-09]
            ),
            "SSP245": poly1d(
                [-5.16184849e-07, -5.05898173e-05, 3.24613177e-02, -3.80784746e-07]
            ),
            "SSP370": poly1d(
                [9.95348537e-08, 1.16654213e-04, 3.27642362e-02, -3.22124297e-07]
            ),
        }

        cs_obj = pkl.load(
            open(os.path.join("temp_pickles", "cs_obj_houston.pickle"), "rb")
        )[0]
    """
    # STEP 2. FIT THE HISTORIC DATA.
    # USE THESE TO CONTROL OPTIMIZATION...
    # You must provide a proxy.txt file if needed or change proxy=None if the computer
    # system does not have a proxy server.
    """
    if step == 2:
        solve_options = {
            "historic": {
                "delT_above_shifted_extreme": {"cs": -10, "hw": 10},
                "decay_func_type": {
                    "cs": "quadratic_times_exponential_decay_with_cutoff",
                    "hw": "quadratic_times_exponential_decay_with_cutoff",
                },
                "max_iter": 30,
                "limit_temperatures": False,
                "num_cpu": -1,
                "plot_results": plot_results,
                "num_step": 2500000,
                "test_mode": False,
                "min_num_waves": 10,
                "weights": np.array([1, 1, 1, 1, 1]),
                "out_path": os.path.join(
                    "example_data", "ClimateZone2A_Houston", "results", "houston.png"
                ),
            },
            "future": {
                "delT_above_shifted_extreme": {"cs": -10, "hw": 10},
                "max_iter": 30,
                "limit_temperatures": False,
                "num_cpu": -1,
                "num_step": 2500000,
                "plot_results": plot_results,
                "decay_func_type": {
                    "cs": "quadratic_times_exponential_decay_with_cutoff",
                    "hw": "quadratic_times_exponential_decay_with_cutoff",
                },
                "test_mode": False,
                "min_num_waves": 10,
                "out_path": os.path.join(
                    "example_data",
                    "ClimateZone2A_Houston",
                    "results",
                    "houston_future.png",
                ),
            },
        }

        # I had to manually add the daily summaries and norms
        obj = ExtremeTemperatureWaves(
            station,
            weather_files,
            unit_conversion=unit_conversion,
            use_local=True,
            random_seed=random_seed,
            include_plots=plot_results,
            run_parallel=run_parallel,
            use_global=False,
            delT_ipcc_min_frac=1.0,
            num_cpu=num_cpu,
            write_results=True,
            test_markov=False,
            solve_options=solve_options,
            proxy=os.path.join("..", "..", "proxy.txt"),
            norms_unit_conversion=unit_conv_norms,
        )

        # add quadratic_times_exponential_decay_with_cutoff
        obj.write_solution(first_try_solution_location)

        if not os.path.exists("temp_pickles"):
            os.mkdir("temp_pickles")
        pkl.dump(
            [obj, solve_options],
            open(os.path.join("temp_pickles", "obj_houston.pickle"), "wb"),
        )
    else:

        blob = pkl.load(open(os.path.join("temp_pickles", "obj_houston.pickle"), "rb"))
        obj = blob[0]
        solve_options = blob[1]

    """
    STEP 3
    
    Shift historic distributions to future
    
    """
    if step == 3:

        historical_solution = os.path.join(first_try_solution_location)

        results, filenames = obj.create_solutions(
            future_years, scenarios, ci_intervals, historical_solution, scen_dict
        )

    #
