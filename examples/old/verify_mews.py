#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:44:54 2022

@author: dlvilla

This script verifies a case to have the Kolmogorov-Smirnov statistics for closeness of fit
between the historic model and

You must run worcester_example.py and generate_files_from_solution_example.py

It also cross checks by loading all files generated and the climate norms,

and the climate change polynomial


"""
import os
import pickle
import numpy as np
import pandas as pd


def create_smirnov_table(obj):

    hist_obj_solve = obj.hist_obj_solve
    np.zeros([0, 13])

    table_dict = {}
    for variable in hist_obj_solve[1].kolmogorov_smirnov.keys():
        table_dict[variable] = np.zeros([0, 13])

    # establish columns for both tables.
    var1 = list(hist_obj_solve[1].kolmogorov_smirnov.keys())[0]
    first_column = ("hw", 1, "month")
    columns = [first_column]
    num_cs = 0
    for trial_num, wave_type_dict in hist_obj_solve[1].kolmogorov_smirnov[var1].items():
        for wave_type, Ktest_obj in wave_type_dict.items():
            if wave_type == "cs":
                columns.append((wave_type.upper(), trial_num, "statistic"))
                columns.append((wave_type.upper(), trial_num, "p-value"))
                num_cs += 2
            else:
                columns.insert(
                    len(columns) - num_cs, (wave_type.upper(), trial_num, "statistic")
                )
                columns.insert(
                    len(columns) - num_cs, (wave_type.upper(), trial_num, "p-value")
                )

    for month, solve_obj in hist_obj_solve.items():
        for variable, random_trial_dict in solve_obj.kolmogorov_smirnov.items():
            next_row = np.zeros([1, 13], dtype=float)
            next_row[0, 0] = month
            for trial_num, wave_type_dict in random_trial_dict.items():
                for wave_type, Ktest_obj in wave_type_dict.items():
                    if wave_type == "hw":
                        base_ind = 1
                    else:
                        base_ind = 7

                    next_row[0, base_ind + 2 * (trial_num - 1)] = Ktest_obj.statistic
                    next_row[0, base_ind + 2 * (trial_num - 1) + 1] = Ktest_obj.pvalue

            table_dict[variable] = np.concatenate(
                [table_dict[variable], next_row], axis=0
            )

    df_dict = {}
    for variable, arr in table_dict.items():
        df = pd.DataFrame(arr, columns=columns)
        df.index = df[first_column]
        df.index = [int(ind) for ind in df.index]
        df.index.name = "month"
        df.drop(first_column, axis=1, inplace=True)
        df.columns = pd.MultiIndex.from_tuples(columns[1:])

        df.to_latex("worcester_kolmogorov_smirnov_" + variable + ".tex")  # ,
        # caption="Kolmogorov-Smirnov distribution fit statistics for "+variable
        # + ". The columns are by wave type, random trial, and "
        # +"statistic type. 'statistic' is the supremum of differences"
        # +" between the distributions and 'p-value' is the "
        # +"acceptance value p-value > 0.05 rejects the null "
        # +"hypothesis and represents 95\% confidence that the "
        # +"distributions match" )

        pass

    pass


if __name__ == "__main__":
    results_file = os.path.join("temp_pickles", "obj_worcester.pickle")

    tup = pickle.load(open(results_file, "rb"))

    obj = tup[0]

    create_smirnov_table(obj)
