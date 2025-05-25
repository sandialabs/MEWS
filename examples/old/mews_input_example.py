#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:00:00 2023

@author: dlvilla
"""
import os
import numpy as np

run_dict = {
    "Chicago": {
        "future_years": [2020, 2040, 2060, 2080],
        "ci_intervals": ["5%", "50%", "95%"],
        "latitude_longitude": (41.78300, 360 - 87.75000),
        "daily_summaries_path": os.path.join(
            "example_data", "ClimateZone5A_Chicago", "Chicago_daily.csv"
        ),
        "climate_normals_path": os.path.join(
            "example_data", "ClimateZone5A_Chicago", "Chicago_norms.csv"
        ),
        "daily_summaries_unit_conversion": (5 / 9, -(5 / 9) * 32),
        "climate_normals_unit_conversion": (5 / 9, -(5 / 9) * 32),
        "historic_solution_save_location": os.path.join(
            "example_data",
            "ClimateZone5A_Chicago",
            "results",
            "chicago_midway_historical_solution.txt",
        ),
        "weather_files": [
            os.path.join(
                "example_data",
                "ClimateZone5A_Chicago",
                "USA_IL_Chicago.Midway.Intl.AP.725340_TMY3.epw",
            ),
            os.path.join(
                "example_data",
                "ClimateZone5A_Chicago",
                "USA_IL_Chicago.OHare.Intl.AP.725300_TMY3.epw",
            ),
        ],
        "random_seed": 349082,
        "cmip6_data_folder": os.path.join("..", "..", "CMIP6_Data_Files"),
        "solve_options": {
            "historic": {
                "delT_above_shifted_extreme": {"cs": -10, "hw": 10},
                "decay_func_type": {
                    "cs": "quadratic_times_exponential_decay_with_cutoff",
                    "hw": "quadratic_times_exponential_decay_with_cutoff",
                },
                "max_iter": 30,
                "limit_temperatures": False,
                "num_cpu": -1,
                "plot_results": True,
                "num_step": 2500000,
                "test_mode": False,
                "min_num_waves": 10,
                "weights": np.array([1, 1, 1, 1, 1]),
                "out_path": os.path.join(
                    "example_data", "ClimateZone5A_Chicago", "results", "chicago.png"
                ),
            },
            "future": {
                "delT_above_shifted_extreme": {"cs": -10, "hw": 10},
                "max_iter": 30,
                "limit_temperatures": False,
                "num_cpu": -1,
                "num_step": 2500000,
                "plot_results": True,
                "decay_func_type": {
                    "cs": "quadratic_times_exponential_decay_with_cutoff",
                    "hw": "quadratic_times_exponential_decay_with_cutoff",
                },
                "test_mode": False,
                "min_num_waves": 10,
                "out_path": os.path.join(
                    "example_data",
                    "ClimateZone5A_Chicago",
                    "results",
                    "chicago_future.png",
                ),
            },
        },
        "clim_scen_out_folder": os.path.join(
            "example_data", "ClimateZone5A_Chicago", "clim_scen_results"
        ),
        "num_files_per_scenario": 250,
    },
    "Phoenix": {
        "template_id": "Chicago",
        "weather_files": [
            os.path.join(
                "example_data",
                "ClimateZone2B_Phoenix",
                "USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw",
            )
        ],
        "latitude_longitude": (33.4352, 360 - 112.0101),
        "daily_summaries_path": os.path.join(
            "example_data", "ClimateZone2B_Phoenix", "Chicago_daily.csv"
        ),
        "climate_normals_path": os.path.join(
            "example_data", "ClimateZone2B_Phoenix", "Chicago_norms.csv"
        ),
        "historic_solution_save_location": os.path.join(
            "example_data",
            "ClimateZone2B_Phoenix",
            "results",
            "phoenix_airport_historical_solution.txt",
        ),
        "solve_options": {
            "historic": {
                "out_path": os.path.join(
                    "example_data",
                    "ClimateZone2B_Phoenix",
                    "results",
                    "phoenix_airport.png",
                )
            },
            "future": {
                "out_path": os.path.join(
                    "example_data",
                    "ClimateZone2B_Phoenix",
                    "results",
                    "phoenix_airport_future.png",
                )
            },
        },
        "clim_scen_out_folder": os.path.join(
            "example_data", "ClimateZone2B_Phoenix", "clim_scen_results"
        ),
    },
    "Minneapolis": {
        "template_id": "Chicago",
        "weather_files": [
            os.path.join(
                "example_data",
                "ClimateZone6A_Minneapolis",
                "USA_MN_Minneapolis-St.Paul.Intl.AP.726580_TMY3.epw",
            )
        ],
        "latitude_longitude": (44.8848, 360 - 93.2223),
        "daily_summaries_path": os.path.join(
            "example_data", "ClimateZone6A_Minneapolis", "Minneapolis_daily.csv"
        ),
        "climate_normals_path": os.path.join(
            "example_data", "ClimateZone6A_Minneapolis", "Minneapolis_norms.csv"
        ),
        "historic_solution_save_location": os.path.join(
            "example_data",
            "ClimateZone6A_Minneapolis",
            "results",
            "minneapolis_airport_historical_solution.txt",
        ),
        "solve_options": {
            "historic": {
                "out_path": os.path.join(
                    "example_data",
                    "ClimateZone6A_Minneapolis",
                    "results",
                    "minneapolis_airport.png",
                )
            },
            "future": {
                "out_path": os.path.join(
                    "example_data",
                    "ClimateZone6A_Minneapolis",
                    "results",
                    "minneapolis_airport_future.png",
                )
            },
        },
        "clim_scen_out_folder": os.path.join(
            "example_data", "ClimateZone6A_Minneapolis", "clim_scen_results"
        ),
    },
    "Baltimore": {
        "template_id": "Chicago",
        "weather_files": [
            os.path.join(
                "example_data",
                "ClimateZone4A_Baltimore",
                "USA_MD_Baltimore-Washington.Intl.Marshall.AP.724060_TMY3.epw",
            )
        ],
        "latitude_longitude": (39.1774, 360 - 76.6684),
        "daily_summaries_path": os.path.join(
            "example_data", "ClimateZone4A_Baltimore", "Baltimore_daily.csv"
        ),
        "climate_normals_path": os.path.join(
            "example_data", "ClimateZone4A_Baltimore", "Baltimore_norms.csv"
        ),
        "historic_solution_save_location": os.path.join(
            "example_data",
            "ClimateZone4A_Baltimore",
            "results",
            "baltimore_airport_historical_solution.txt",
        ),
        "solve_options": {
            "historic": {
                "out_path": os.path.join(
                    "example_data",
                    "ClimateZone4A_Baltimore",
                    "results",
                    "baltimore_airport.png",
                )
            },
            "future": {
                "out_path": os.path.join(
                    "example_data",
                    "ClimateZone4A_Baltimore",
                    "results",
                    "baltimore_airport_future.png",
                )
            },
        },
        "clim_scen_out_folder": os.path.join(
            "example_data", "ClimateZone4A_Baltimore", "clim_scen_results"
        ),
    },
}
