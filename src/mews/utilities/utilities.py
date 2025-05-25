#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:10:32 2022

Copyright 2023 National Technology and Engineering Solutions of Sandia, LLC.
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

import os
from warnings import warn
import numpy as np
import pandas as pd
from numpy import poly1d
from mews.epw import epw
from mews.constants.data_format import EPW_PSYCHROMETRIC_DICTIONARY as EPW_PSYCH_DICT
from mews.constants.data_format import (
    EPW_PRESSURE_COLUMN_NAME,
    EPW_DB_TEMP_COLUMN_NAME,
    EPW_DP_TEMP_COLUMN_NAME,
    EPW_RH_COLUMN_NAME,
    EPW_PSYCH_NAMES,
)

from CoolProp.HumidAirProp import HAPropsSI


def calculate_psychrometrics(epw_df, in_columns, out_cool_prop_name=""):
    f"""
    This function takes 3 input psychrometric
    variables in the CoolProp HAPropsSI function and calculates a 
    psychrometric output variable in an EPW file. You can see which variables are input/output
    or strictly output in the "Table of inputs/outputs to HAPropsSI" at the bottom 
    of http://www.coolprop.org/fluid_properties/HumidAir.html#id9

    For EPW files the 4 psychrometric variables are:
    1) Pressure (input)
    2) Dry Bulb Temperature (input/output)
    3) Dew Point Temperature (input/output)
    4) Relative Humidity (input/output)

    Pressure must be an input since the other variables cannot determine it.
    Two other variables are needed so one variable must be recalculated if a change is 
    made to any one of these four. Up to three can be changed and then the fourth 
    recalculated.

    Many other variables can be output if they are needed.

    Inputs
    ======

    epw_df : pandas.Dataframe :
        this must be a dataframe that has been read from an EPW file using an object 
        instantiated from obj = mews.epw.epw() and obj.read(epw_weather_path) obj.epwobj.dataframe

    in_columns : list : len(list) == 2 or 3, all elements in list in {EPW_PSYCH_NAMES}
        always includes '{EPW_PRESSURE_COLUMN_NAME}' if len(list) == 3, each entry
        must be unique (no repeated names).

    out_cool_prop_name : str (optional)
        Give a correct name in http://www.coolprop.org/fluid_properties/HumidAir.html#id9 to
        output something in CoolProp units other than the outputs needed by EPW weather files.

    Returns
    =======
    Whatever column name in {EPW_PSYCH_NAMES} is not in in_columns (+Pressure) is output
    as a timeseries and the epw_df is updated with these new values

    Raises
    ======
    TypeError, ValueError - data validation does not allow invalid inputs.
        
    """
    in_columns = ValidationPsychCalcs._in_column_validation(in_columns)
    ValidationPsychCalcs._epw_df_validation(epw_df, in_columns)

    var_out = [column for column in EPW_PSYCH_NAMES if column not in in_columns][0]
    var_out_symbol = EPW_PSYCH_DICT[var_out]["cool_prop"]["symbol"]
    var_in_symbols = [
        EPW_PSYCH_DICT[column]["cool_prop"]["symbol"] for column in in_columns
    ]
    convert = [
        EPW_PSYCH_DICT[column]["cool_prop"]["convert_units"] for column in in_columns
    ]
    unconvert = EPW_PSYCH_DICT[var_out]["cool_prop"]["unconvert_units"]

    if len(out_cool_prop_name) != 0:
        # this is a shortcircuit that can cause an error.
        var_out_symbol = out_cool_prop_name

    # NOT SURE WHICH VERSION IS FASTER! THIS IS SLOW THOUGH
    # MOST LIKELY ITS BECAUSE COOLPROPS takes some time!
    # def psych_func(row):
    #     return unconvert(HAPropsSI(
    #                         var_out_symbol,
    #                         var_in_symbols[0],
    #                         float(convert[0](row[in_columns[0]])),
    #                         var_in_symbols[1],
    #                         float(convert[1](row[in_columns[1]])),
    #                         var_in_symbols[2],
    #                         float(convert[2](row[in_columns[2]]))))
    # new_values = epw_df.apply(psych_func,axis=1)
    var_in = [
        epw_df[in_columns[0]].values,
        epw_df[in_columns[1]].values,
        epw_df[in_columns[2]].values,
    ]

    new_values = unconvert(
        HAPropsSI(
            var_out_symbol,
            var_in_symbols[0],
            convert[0](var_in[0]),
            var_in_symbols[1],
            convert[1](var_in[1]),
            var_in_symbols[2],
            convert[2](var_in[2]),
        )
    )

    epw_df[var_out] = new_values
    return new_values


def dict_key_equal(dict_key, dict_check):
    # This function is recursive

    if dict_key.keys() == dict_check.keys():
        for key, val in dict_key.items():
            if isinstance(val, dict) and not isinstance(dict_check[key], dict):
                raise ValueError(
                    "The dictionary being checked has a value that should be a dictionary"
                )
            elif isinstance(val, dict):
                dict_key_equal(val, dict_check[key])

    else:
        raise ValueError(
            "The dictionary being checked does not have"
            + " the same keys. It must have the key:\n\n{0}"
            + "\n\n".format(str(dict_key.keys()))
        )


def filter_cpu_count(cpu_count):
    """
    Filters the number of cpu's to use for parallel runs such that
    the maximum number of cpu's is properly constrained.

    Parameters
    ----------
    cpu_count : int or None :
        int : number of proposed cpu's to use in parallel runs
        None : choose the number of cpu's based on os.cpu_count()
    Returns
    -------
    int
        Number of CPU's to use in parallel runs.

    """

    if isinstance(cpu_count, (type(None), int)):

        max_cpu = os.cpu_count()

        if cpu_count is None:
            if max_cpu == 1:
                return 1
            else:
                return max_cpu - 1

        elif max_cpu <= cpu_count:
            warn(
                "The requested cpu count is greater than the number of "
                + "cpu available. The count has been reduced to the maximum "
                + "number of cpu's ({0:d}) minus 1 (unless max cpu's = 1)".format(
                    max_cpu
                )
            )
            if max_cpu == 1:
                return 1
            else:
                return max_cpu - 1
        elif cpu_count == -1:
            if max_cpu == 1:
                return 1
            else:
                return max_cpu - 1
        elif cpu_count <= 0:
            raise ValueError(
                "The CPU count must be a positive number or -1.\n"
                + "-1 indicates to use the maximum number of cpu's minus 1"
            )
        else:
            return cpu_count


def check_for_nphistogram(hist):
    if not isinstance(hist, tuple):
        raise TypeError("The histogram must be a 2-tuple of two arrays")
    elif not len(hist) == 2:
        raise ValueError("The histogram must be a tuple of length 2")
    elif not isinstance(hist[0], np.ndarray) or not isinstance(hist[1], np.ndarray):
        raise TypeError("The histogram tuples elements must be of type np.ndarray")
    elif len(hist[0]) != len(hist[1]) - 1:
        raise ValueError(
            "The histogram tuple first entry must be one element smaller in lenght than the second entry!"
        )


def histogram_area(hist):
    bin_ = bin_avg(hist)
    h_ = hist[0]

    return np.trapezoid(h_, bin_)


def histogram_step_wise_integral(hist, a=None, b=None):
    """
    Calculates the integral assuming that a constant value is sustained
    for each bin in a histogram

    Parameters
    ----------
    hist : 2-tuple from np.histogram first entry is one less long as second
           first element are values and second element are bin edges
        DESCRIPTION.
    a : float, optional
        lower integration boundary. The default is None. If None, then just
        use the lowest bound of the histogram bins
    b : float, optional
        upper integration boundary. The default is None. If None, then just
        use the highest bound of the histogram bins

    Raises
    ------
    ValueError
        raised if a > b.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if a is None:
        a = hist[1].min()
    if b is None:
        b = hist[1].max()

    if a > b:
        raise ValueError("Input 'a' must be greater than input 'b'.")
    if a == b:
        return 0.0

    bin_ = hist[1]
    h_ = hist[0]

    if a > bin_[-1]:
        return 0.0
    elif b < bin_[0]:
        return 0.0

    if a < bin_[0]:
        beg_area = 0.0
        beg_bin = 0
    else:
        # this is the first bin boundary division ahead of a
        diva = (bin_ < a).argmin()
        beg_area = h_[diva - 1] * (bin_[diva] - a)
        beg_bin = diva

    if b > bin_[-1]:
        end_area = 0.0
        end_bin = len(bin_)
    else:
        divb = (bin_ < b).argmin()
        end_area = h_[divb - 1] * (b - bin_[divb - 1])
        end_bin = divb - 1

    return (
        (np.diff(bin_)[beg_bin:end_bin] * h_[beg_bin:end_bin]).sum()
        + beg_area
        + end_area
    )


def histogram_intersection(hist1, hist2):
    def int_boundaries(bin_, maxb, minb):
        bin_ = bin_[bin_ <= maxb]
        return bin_[bin_ >= minb]

    bin1 = bin_avg(hist1)
    bin2 = bin_avg(hist2)
    h1 = hist1[0]
    h2 = hist2[0]

    maxintb = np.min([bin1.max(), bin2.max()])
    minintb = np.max([bin1.min(), bin2.min()])

    bin1int = int_boundaries(bin1, maxintb, minintb)
    bin2int = int_boundaries(bin2, maxintb, minintb)

    intpoint = np.unique(np.concatenate([bin1int, bin2int]))

    interp_vals1 = np.interp(intpoint, bin1, h1)
    interp_vals2 = np.interp(intpoint, bin2, h2)

    minvals = np.min(
        np.concatenate(
            [
                interp_vals1.reshape([len(interp_vals1), 1]),
                interp_vals2.reshape([len(interp_vals2), 1]),
            ],
            axis=1,
        ),
        axis=1,
    )

    return np.trapezoid(minvals, intpoint)


def histogram_non_overlapping(hist1, hist2, return_min_max=False):
    bin1 = bin_avg(hist1)
    bin2 = bin_avg(hist2)
    h1 = hist1[0]
    h2 = hist2[0]

    # find max boundaries
    maxintb = np.min([bin1.max(), bin2.max()])
    minintb = np.max([bin1.min(), bin2.min()])

    maxb = np.max([bin1.max(), bin2.max()])
    minb = np.min([bin1.min(), bin2.min()])

    # establish what points to integrate
    if maxb in bin1:
        elem = bin1 >= maxintb
        intpoint_max = bin1[elem]
        h_max = h1[elem]
    else:
        elem = bin2 >= maxintb
        intpoint_max = bin2[elem]
        h_max = h2[elem]

    if minb in bin1:
        elem = bin1 <= minintb
        intpoint_min = bin1[elem]
        h_min = h1[elem]
    else:
        elem = bin2 <= minintb
        intpoint_min = bin2[elem]
        h_min = h2[elem]

    # integrate
    min_side_area = np.trapezoid(h_min, intpoint_min)
    max_side_area = np.trapezoid(h_max, intpoint_max)

    if return_min_max:
        return min_side_area, max_side_area
    else:
        return min_side_area + max_side_area


def histogram_non_intersection(hist1, hist2):
    """
    Calculates a numeric approximation of the non-intersecting area of
    two histograms. values are assumed to occur at the centriod of each bin
    significant errors may result for low-resolution histograms

    Parameters
    ----------
    hist1 : tuple output from numpy.histogram
        first histogram
    hist2 : tuple output from numpy.histogram
        second histogram

    Returns
    -------
    float
        Non-intersecting area of a histogram.

    """

    intersect_area = histogram_intersection(hist1, hist2)
    non_overlapping_area = histogram_non_overlapping(hist1, hist2)

    bin1 = bin_avg(hist1)
    bin2 = bin_avg(hist2)

    area1 = np.trapezoid(hist1[0], bin1)
    area2 = np.trapezoid(hist2[0], bin2)

    return area1 + area2 - 2 * intersect_area


def create_complementary_histogram(sample, hist0):
    """
    This function recieves a sample from a dynamic random process and then creates
    a histogram of the sample using the same spacing interval as a histogram 'hist0'
    It then returns the histogram as a discreet probability distribution function tuple
    and as a cumulative probability distribution (CDF) function tuple

    Parameters
    ----------

    sample : np.array
        a 1-d array of sample values for a dynamic process.

    hist0 : tuple returned by np.histogram

    Returns
    -------
    tup1, tup2 - tup1 is the sample discreet pdf and tup2 is the sample
                 discreet cdf

    Raises
    -----
    TypeError - if sample is not an np.array or hist0 is not a tuple of arrays
    ValueError - if the input histogram

    """
    if len(sample) == 0:
        # no waves occured and all zeros must be used to evaluate
        hist1 = np.zeros(len(hist0[0]))
        bin_avg = (hist0[1][1:] + hist0[1][0:-1]) / 2
        return (hist1, hist0[1]), (np.ones(len(hist0[0])), bin_avg)
    else:

        check_for_nphistogram(hist0)

        bin_spacing = np.diff(hist0[1]).mean()

        start = np.floor(sample.min() / bin_spacing)
        end = np.ceil(sample.max() / bin_spacing)

        num_bin = int(end - start)

        hist1 = np.histogram(
            sample, num_bin, range=(start * bin_spacing, end * bin_spacing)
        )

        bin_avg = (hist1[1][1:] + hist1[1][0:-1]) / 2

        hist1_prob = hist1[0] / hist1[0].sum()

        cdf = (hist1_prob).cumsum()

        return (hist1_prob, hist1[1]), (cdf, bin_avg)


def linear_interp_discreet_func(vals, discreet_func_tup, is_x_interp=False):
    """
    linear interpolation of a function. Naturally written for y interpolation
    of a cdf but just have to reverse terms for x-interpolation


    """

    if is_x_interp:
        xf = discreet_func_tup[0]
        yf = discreet_func_tup[1]
    else:
        xf = discreet_func_tup[1]
        yf = discreet_func_tup[0]

    len_s = len(xf)
    # must handle values beyond the range of yf (=xf if is_x_interp=True)
    idi = np.array(
        [
            (
                0
                if va <= yf[0]
                else (
                    len_s
                    if va >= yf[-1]
                    else np.where(np.logical_and(yf[0:-1] <= va, yf[1:] > va))[0][0]
                )
            )
            for va in vals
        ]
    )
    interp = np.array(
        [
            (
                xf[idix]
                + (va - yf[idix])
                / (yf[idix + 1] - yf[idix])
                * (xf[idix + 1] - xf[idix])
                if idix < len_s
                else xf[-1]
            )
            for va, idix in zip(vals, idi)
        ]
    )
    return interp


def write_readable_python_dict(filepath, dictionary, overwrite=True):
    """
    Write a python dictionary as a string in an ascii file that
    has a pleasantly readable format and can be evaluated using eval
    to reproduce the dictionary when the file is read back into python.

    Parameters
    ----------
    filepath : str
        Valid filepath to a file
    dictionary : dict
        Any python dictionary to be written to a file
    overwrite : bool, optional
        Indicate whether to overwrite existing files. The default is True.

    Raises
    ------
    FileExistsError
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            "The file:\n\n '{0}'\n\n".format(filepath)
            + "Already exists. Set overwrite=True to overwrite it."
        )
    else:
        if os.path.exists(filepath):
            os.remove(filepath)

        inquote = False
        is_single_quote = False
        inarray = False
        newstr = ""
        indent_level = 0
        for char in str(dictionary).replace("array", "np.array"):
            if char == "," and not inquote and not inarray:
                newstr = newstr + char + "\n" + indent_level * " "
            elif char == "{" and not inquote:
                indent_level = indent_level + 4
                newstr = newstr + "\n" + indent_level * " " + char
            elif char == "}" and not inquote:
                indent_level = indent_level - 4
                newstr = newstr + char + "\n" + indent_level * " "
            elif char == "(" or char == "[" and not inquote:
                inarray = True
                newstr = newstr + char
            elif char == ")" or char == "]" and not inquote:
                inarray = False
                newstr = newstr + char
            elif char == "'" and not inquote:
                inquote = True
                is_single_quote = True
                newstr = newstr + char
            elif char == '"' and not inquote:
                inquote = True
                is_single_quote = False
                newstr = newstr + char
            elif char == "'" and inquote and is_single_quote:
                inquote = False
                newstr = newstr + char
            elif char == '"' and inquote and not is_single_quote:
                inquote = False
                newstr = newstr + char
            else:
                newstr = newstr + char

        with open(filepath, "w") as fref:
            fref.write(newstr)


def read_readable_python_dict(filepath, var={}):
    """
    Parameters
    ----------
    filepath : str
        Valid file path to a text file that contains a readable python dictionary
        string written by "write_readable_python_dict"

    var : dict
       This is only certain to handle string inputs
       It provides a way to put variable names in the readable python dict
       that can then be filled in. Local versions of all variables in
       kwargs are generated.

    Raises
    ------
    FileNotFoundError
        If filepath does not exist, this exception is raised
    SyntaxError
        If the contents of the file fails to evaluate
    ValueError
        If the contents of the file evaluate to a non-dictionary

    Returns
    -------
    A Python dictionary expressed in the file found at filepath

    """

    if os.path.exists(filepath):
        try:
            for key, val in var.items():
                exec("{0} = r'{1}'".format(key, str(val)))

            with open(filepath, "r") as file:
                fstr = file.read()

            inquote = False
            is_single_quote = False
            newstr = ""
            for char in fstr:
                if (char == " " and not inquote) or (char == "\n" and not inquote):
                    pass  # do nothing to add these.
                elif char == "'" and not inquote:
                    inquote = True
                    is_single_quote = True
                    newstr = newstr + char
                elif char == '"' and not inquote:
                    inquote = True
                    is_single_quote = False
                    newstr = newstr + char
                elif char == "'" and inquote and is_single_quote:
                    inquote = False
                    newstr = newstr + char
                elif char == '"' and inquote and not is_single_quote:
                    inquote = False
                    newstr = newstr + char
                else:
                    newstr = newstr + char

            pdict = eval(newstr)

            if not isinstance(pdict, dict):
                raise ValueError("The string read did not create a dictionary")

            return pdict
        except Exception as e:
            raise SyntaxError(
                "The designated file '{0}' does not contain a ".format(filepath)
                + "readable python dictionary from write_readable_python_dict."
            ) from e
    else:
        raise FileNotFoundError(
            "The file\n\n  '{0}'\n\ndoes not exist!".format(filepath)
        )


def find_extreme_intervals(states_arr, states):
    """
    This function returns a dictionary whose entry keys are
    the "states" input above. Each dictionary element contains
    a list of tuples. Each tuple contains the start and end times
    of an event where "states_arr" was equal to the corresponding state

    Parameters
    ----------
    states_arr : array-like
        a 1-D array of integers of values that are only in the states input
    states : array-like,list-like
        a 1-D array of values to look for in states_arr.

    Returns
    -------
    state_int_dict : TYPE
        A dictionary with key "state" each value contains a list of tuples
        that indicate the start and end time of an extreme event.

    """

    diff_states = np.concatenate((np.array([0]), np.diff(states)))
    state_int_dict = {}
    for state in states:
        state_ind = [i for i, val in enumerate(states_arr == state) if val]
        end_points = [i for i, val in enumerate(np.diff(state_ind) > 1) if val]
        start_point = 0
        if len(end_points) == 0 and len(state_ind) > 0:
            ep_list = [(state_ind[0], state_ind[-1])]
        elif len(end_points) == 0 and len(state_ind) == 0:
            ep_list = []
        else:
            ep_list = []
            for ep in end_points:
                ep_list.append((state_ind[start_point], state_ind[ep]))
                start_point = ep + 1
        state_int_dict[state] = ep_list
    return state_int_dict


def _bin_avg(hist):
    return (hist[1][1:] + hist[1][0:-1]) / 2


def bin_avg(hist):
    if len(hist[1]) == len(hist[0]) + 1:
        bin0 = _bin_avg(hist)
    else:
        bin0 = hist[1]
    return bin0


def create_smirnov_table(obj, output_table_location):

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

        if output_table_location[-4:] == ".tex":
            out_name = output_table_location[:-4] + "_" + variable + ".tex"
        else:
            out_name = output_table_location + "_" + variable + ".tex"

        df.to_latex(out_name)

        df_dict[variable] = df

    return df_dict


def quantify_event_errors_in_temperature(file_path, base_name, ssp, ci_, years):
    # reads csv files output

    multi_col = []
    for year in years:
        multi_col.append((year, "10 yr", "mean"))
        multi_col.append((year, "10 yr", "std"))
        multi_col.append((year, "10 yr", "min"))
        multi_col.append((year, "10 yr", "max"))
        multi_col.append((year, "50 yr", "mean"))
        multi_col.append((year, "50 yr", "std"))
        multi_col.append((year, "50 yr", "min"))
        multi_col.append((year, "50 yr", "max"))

    max_val = [-1]
    min_val = [1]

    all_errors = np.array([0, 0])

    multi_index = []
    data_table = []
    for ssp_ in ssp:
        for ci in ci_:
            multi_index.append((ssp_, ci))
            next_row = []

            for year in years:

                # the December csv has all months in it.
                month = 12
                df = pd.read_csv(
                    os.path.join(
                        file_path,
                        base_name
                        + str(month)
                        + "_"
                        + str(ssp_)
                        + "_"
                        + str(ci)
                        + "_"
                        + str(year)
                        + ".csv",
                    )
                )

                df_10_actual = df[(df["type"] == "10 year actual")]
                df_10_target = df[(df["type"] == "10 year target")]
                df_50_actual = df[(df["type"] == "50 year actual")]
                df_50_target = df[(df["type"] == "50 year target")]

                err_10 = (
                    df_10_actual["threshold"].values - df_10_target["threshold"].values
                )
                err_50 = (
                    df_50_actual["threshold"].values - df_50_target["threshold"].values
                )

                all_errors = np.concatenate([all_errors, err_10, err_50])

                next_row.append(err_10.mean())
                next_row.append(err_10.std())
                next_row.append(err_10.min())
                next_row.append(err_10.max())
                next_row.append(err_50.mean())
                next_row.append(err_50.std())
                next_row.append(err_50.min())
                next_row.append(err_50.max())

                if max_val[0] < err_10.max():
                    max_val = [
                        err_10.max(),
                        ssp_,
                        ci,
                        year,
                        err_10.argmax() + 1,
                        "10 year",
                    ]
                if max_val[0] < err_50.max():
                    max_val = [
                        err_50.max(),
                        ssp_,
                        ci,
                        year,
                        err_50.argmax() + 1,
                        "50 year",
                    ]

                if min_val[0] > err_10.min():
                    min_val = [
                        err_10.min(),
                        ssp_,
                        ci,
                        year,
                        err_10.argmin() + 1,
                        "10 year",
                    ]
                if min_val[0] > err_50.min():
                    min_val = [
                        err_50.min(),
                        ssp_,
                        ci,
                        year,
                        err_50.argmin() + 1,
                        "50 year",
                    ]

            data_table.append(next_row)
    print("min_val = " + str(min_val))
    print("max_val = " + str(max_val))

    print("mean of all errors: " + str(all_errors.mean()))
    print("standard deviation of all errors" + str(all_errors.std()))
    final_df = pd.DataFrame(
        data_table,
        index=pd.MultiIndex.from_tuples(multi_index),
        columns=pd.MultiIndex.from_tuples(multi_col),
    )
    final_df.T.to_latex("future_temperature_errors.tex")


def create_output_weather_file_name(
    wfile, scenario_name, year, confidence_interval, realization_num
):
    return (
        os.path.basename(wfile)[:-4]
        + scenario_name
        + "_{0:d}".format(year)
        + "_{0}".format(confidence_interval)
        + "_r{0:d}".format(realization_num)
        + wfile[-4:]
    )


def list_epws_stats(epws_dir, stat_col="Dry Bulb Temperature", out_file=None):
    """
    create a pandas dataframe of summary statistics across all energy plus weather
    (epw) files in a directory. Output as a csv if requested

    Inputs
    ------

    epws_dir : str : A valid path to a folder that contains 1 or more epw files

    stat_col : str : The valid name for a column for which summary statistics
                      will be calculated

    out_file : str : A file name (no path) for which a ".csv" of the output
                     dataframe will be output to one directory back from the
                     epws_dir

    Returns
    -------

    df_out : pandas.DataFrame : index= epw file names, columns equal
                                the maximum, minimum, sum, mean, standard
                                deviation, skew, and kurtosis of the stat_col

    """

    file_list = os.listdir(epws_dir)

    temp_dict = {}

    for file in file_list:

        epw_obj = epw()
        if ".epw" == file[-4:]:
            epw_obj.read(os.path.join(epws_dir, file))

            # gather statistics on each file
            df = epw_obj.dataframe[stat_col]

            temp_dict[file] = np.array(
                [
                    df.max(),
                    df.min(),
                    df.sum(),
                    df.mean(),
                    df.std(),
                    df.skew(),
                    df.kurtosis(),
                ]
            )

    df_out = pd.DataFrame(
        temp_dict, index=["max", "min", "sum", "mean", "std", "skew", "kurtosis"]
    ).T

    if not out_file is None:
        df_out.to_csv(os.path.join(epws_dir, "..", out_file))

    return df_out


class ValidationPsychCalcs:
    @staticmethod
    def _in_column_validation(in_columns):
        # assure correct type
        if not isinstance(in_columns, list):
            raise TypeError(
                f"The 'in_columns' input must be a list you input: {in_columns}"
            )
        # assure list of strings
        non_string_entries = [
            entry for entry in in_columns if not isinstance(entry, str)
        ]
        if len(non_string_entries) != 0:
            raise TypeError(
                f"The 'in_columns' list must be a list of strings. You input: {in_columns}"
            )

        # assure the right number of entries
        len_in_columns = len(in_columns)
        if len_in_columns not in [2, 3]:
            raise ValueError("The 'in_columns' input must be a list of length 2 or 3")
        # assure correct names:
        incorrect_column_names = [
            in_column for in_column in in_columns if in_column not in EPW_PSYCH_NAMES
        ]
        if len(incorrect_column_names) != 0:
            raise ValueError(
                "The 'in_columns' input has invalid one or more invalid"
                + f" entries: {incorrect_column_names}. The only valid "
                + f"inputs are: {EPW_PSYCH_NAMES}"
            )
        # assure unique names
        in_columns_unique = list(set(in_columns))
        if len(in_columns_unique) != len_in_columns:
            raise ValueError(
                "The 'in_columns' input must have unique members. "
                + f"You input: {in_columns}."
            )
        # assure pressure is included
        if EPW_PRESSURE_COLUMN_NAME not in in_columns and len(in_columns) == 2:
            in_columns.append(EPW_PRESSURE_COLUMN_NAME)
        if EPW_PRESSURE_COLUMN_NAME not in in_columns and len(in_columns) == 3:
            raise ValueError(
                f"You must include {EPW_PRESSURE_COLUMN_NAME} when you give 3 entries, you entered: {in_columns}"
            )

        # assure db and dp or db and rh
        good_combo1 = (EPW_DB_TEMP_COLUMN_NAME in in_columns) and (
            EPW_DP_TEMP_COLUMN_NAME in in_columns
        )
        good_combo2 = (EPW_DB_TEMP_COLUMN_NAME in in_columns) and (
            EPW_RH_COLUMN_NAME in in_columns
        )
        if not (good_combo1 or good_combo2):
            raise ValueError(
                "You must have 1 of 2 combinations of inputs for 'in_column': 1 - "
                + f"[{EPW_DB_TEMP_COLUMN_NAME},{EPW_DP_TEMP_COLUMN_NAME},{EPW_PRESSURE_COLUMN_NAME}], or 2- ["
                + f"[{EPW_DB_TEMP_COLUMN_NAME},{EPW_RH_COLUMN_NAME},{EPW_PRESSURE_COLUMN_NAME}]."
            )
        return in_columns

    @staticmethod
    def _epw_df_validation(epw_df, in_columns):
        # assure correct type
        if not isinstance(epw_df, pd.DataFrame):
            raise TypeError(
                f"The input 'epw_df' must be a pandas.DataFrame! You input: {epw_df}"
            )

        # assure columns exist in the DataFrame
        mismatched_columns = [
            in_column for in_column in in_columns if in_column not in epw_df
        ]
        if len(mismatched_columns) != 0:
            raise ValueError(
                "The following columns do not exist in the 'epw_df' input dataframe:"
                + f" {mismatched_columns}. Something is wrong with your EPW file!"
            )

        # assure columns are numeric
        is_not_numeric = [
            in_column
            for in_column in in_columns
            if not pd.api.types.is_numeric_dtype(epw_df[in_column])
        ]
        if len(is_not_numeric) != 0:
            raise ValueError(
                "The 'epw_df' columns must be numeric. The following"
                + f" requested columns are not numeric: {is_not_numeric}."
            )

        # assure columns have no missing values and are in-bounds to maximum and minimum requirement
        for in_column in in_columns:
            missing = EPW_PSYCH_DICT[in_column]["missing"]
            max = EPW_PSYCH_DICT[in_column]["maximum"]
            min = EPW_PSYCH_DICT[in_column]["minimum"]
            if missing is not None:
                is_missing = epw_df[in_column] == missing
                if is_missing.sum() > 0:
                    raise ValueError(
                        f"The '{in_column}' column has missing values equal to {missing}! "
                        + "No missing values are allowed. Please fix the weather data being used!"
                    )
            if max is not None:
                over_max = epw_df[in_column] > max
                if over_max.sum() > 0:
                    raise ValueError(
                        f"The maximum value allowed for column '{in_column}' has been exceeded"
                        + f" by the following values:{epw_df[in_column][over_max]}"
                    )

            if min is not None:
                under_min = epw_df[in_column] < min
                if under_min.sum() > 0:
                    raise ValueError(
                        f"The maximum value allowed for column '{in_column}' has been exceeded"
                        + f" by the following values:{epw_df[in_column][under_min]}"
                    )
