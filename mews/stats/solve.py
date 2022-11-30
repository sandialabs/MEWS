#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:28:02 2022



@author: dlvilla
"""

from mews.stats.extreme import DiscreteMarkov
from mews.utilities.utilities import (linear_interp_discreet_func, bin_avg,
                                      histogram_non_intersection,
                                      histogram_area,
                                      histogram_intersection,
                                      histogram_non_overlapping,
                                      histogram_step_wise_integral)
from numpy.random import default_rng
from mews.stats.distributions import trunc_norm_dist, inverse_transform_fit, transform_fit
from mews.utilities.utilities import find_extreme_intervals, create_complementary_histogram, dict_key_equal
import matplotlib.pyplot as plt
from mews.graphics.plotting2D import Graphics
from scipy.optimize import minimize, LinearConstraint, differential_evolution, NonlinearConstraint, fsolve
from copy import deepcopy
from warnings import warn
from mews.constants.data_format import (DECAY_FUNC_NAMES, ABREV_WAVE_NAMES, 
                                        VALID_SOLVE_INPUTS, 
                                        DEFAULT_NONE_DECAY_FUNC, 
                                        WEIGHTS_DEFAULT_ARRAY)
from mews.constants.physical import HOURS_IN_LEAP_YEAR, HOURS_IN_YEAR

import numpy as np
import pandas as pd
import os
import numbers


def x_offset_for_func_type_combo(decay_func, events):
    wtoff = []
    # must use list events so that 'cs' is first.
    for wtype in events:
        decfunc = decay_func[wtype]
        if decfunc is None:
            wtoff.append(0)
        elif decfunc == "linear" or decfunc == "exponential":
            wtoff.append(1)  # wave type offset
        elif decfunc == "linear_cutoff" or decfunc == "exponential_cutoff":
            wtoff.append(2)
        elif decfunc == "quadratic_times_exponential_decay_with_cutoff":
            wtoff.append(3)
    return wtoff


def shift_a_b_of_trunc_gaussian(del_mu_T, mu_T, del_sig_T, sig_T):
    return (del_mu_T - (1 + mu_T)/(sig_T) * del_sig_T,
            del_mu_T + (1 - mu_T)/(sig_T) * del_sig_T)


def unpack_params(param, wave_type):
    return {'norm_duration':    param[wave_type]['normalizing duration'],
            'mu_T':             param[wave_type]['extreme_temp_normal_param']['mu'],
            'sig_T':            param[wave_type]['extreme_temp_normal_param']['sig'],
            'hist_maxval_T':         param[wave_type]['hist max extreme temp per duration'],
            'hist_minval_T':         param[wave_type]['hist min extreme temp per duration'],
            'maxval_T':         param[wave_type]['max extreme temp per duration'],
            'minval_T':         param[wave_type]['min extreme temp per duration'],
            'norm_temperature': param[wave_type]['normalizing extreme temp'],
            'alpha_T':          param[wave_type]['normalized extreme temp duration fit slope'],
            'beta_T':           param[wave_type]['normalized extreme temp duration fit intercept']}


def temperature(T_per_durations, durations, pval):
    return T_per_durations * pval['norm_temperature'] * (
        pval['alpha_T'] * (durations/pval['norm_duration']) + pval['beta_T'])


def evaluate_temperature(durations, param, del_mu_T, del_sig_T, rng, wave_type):

    # taken from extreme._add_extreme
    pval = unpack_params(param, wave_type)

    Sm1_T, S_T = shift_a_b_of_trunc_gaussian(
        del_mu_T, pval['mu_T'], del_sig_T, pval['sig_T'])

    min_boundary = transform_fit(pval['minval_T'],pval['hist_minval_T'],pval['hist_maxval_T'])
    max_boundary = transform_fit(pval['maxval_T'],pval['hist_minval_T'],pval['hist_maxval_T'])

    T_per_durations = np.array([trunc_norm_dist(rand,
                                                pval['mu_T'] + del_mu_T,
                                                pval['sig_T'] + del_sig_T,
                                                min_boundary+Sm1_T,
                                                max_boundary+S_T,
                                                pval['hist_minval_T'],
                                                pval['hist_maxval_T']) for rand in rng.random(len(durations))])
    # this returns actual temperatures.
    return temperature(T_per_durations, durations, pval)


def calculated_hist0_stretched(hist0, ipcc_shift, historic_time_interval, hours_per_year, plot_results=False):
    """

    This function calculates a stretched form of the historical wave distribtuion
    Such that it fulfills IPCC requirements but retains the generalized shape
    of the historical distribution. A 4th order polynomial kernel function is used to 
    integrate and find an optimal point at which the IPCC criteria are satisfied
    
    This stretched shape becomes the target shape for the entire future distribution
    when solving for an IPCC shift in addition to direct residuals between the thresholds.

    Parameters
    ----------
    hist0 : TYPE
        DESCRIPTION.
    ipcc_shift : TYPE
        DESCRIPTION.
    num_step : TYPE
        DESCRIPTION.
    historic_time_interval : TYPE
        DESCRIPTION.
    hours_per_year : TYPE
        DESCRIPTION.
    plot_result : bool : Optional
        Output a plot of the old pdf versus the new pdf that has been stretched to
        meet IPCC.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    def bilinear_kernel(T, A, T0, T10, T50):
        return np.array([A[0]*(temp - T0)+1.0 if temp < T10
                         else
                         A[1]*(temp - T10) + A[0]*(T10 - T0) + 1.0
                         if temp < T50
                         else A[1]*(T50-T10) + A[0]*(T10 - T0) + 1.0
                         for temp in T])

    def cubic_kernel(T, A):
        return np.polyval(np.poly1d([A[0], A[1], A[2], A[3], 1.0]), T)

    def objective_function(x, param):

        P10 = param[0]
        P50 = param[1]
        T10 = param[2]
        T50 = param[3]
        T10_shifted = param[4]
        T50_shifted = param[5]
        # must use normalized hist0
        hist0 = param[6]
        wt = param[7]  # wt = wave_type
        return_new_pdf = param[8]

        norm_area = histogram_step_wise_integral(
            hist0, hist0[1].min(), hist0[1].max())
        pdf0 = (hist0[0]/norm_area, hist0[1])
        bin_ = pdf0[1]

        if wt == "hw":
            mult = 1.0
        else:
            mult = -1.0
        
        derivative_vals = 4*x[0]*bin_**3 + 3*x[1]*bin_**2 + 2*x[2]*bin_**2 + x[3]
        
        neg_deriv_penalty = -1e6 * derivative_vals[derivative_vals < 0.0].sum()

        new_bin = bin_ * cubic_kernel(bin_, mult*x[0:4])

        if wt == "hw":
            
            new_bin = np.concatenate([new_bin, np.array([new_bin[-1] + x[4]])])
        else:
            new_bin = np.concatenate([np.array([new_bin[0]-x[4]]), new_bin])
        # add an extra free input to the pdf
        if wt == "hw":
            pdf0_0 = np.concatenate([pdf0[0], np.array([pdf0[0][-1]*x[5]])])
        else:
            pdf0_0 = np.concatenate([np.array([pdf0[0][0]*x[5]]), pdf0[0]])

        # now recalculate and renormalize with the new bin spacing.

        temp_pdf = (pdf0_0, new_bin)
        norm_area_2 = histogram_step_wise_integral(
            temp_pdf, temp_pdf[1].min(), temp_pdf[1].max())
        pdf_new = (temp_pdf[0]/norm_area_2, new_bin)

        if wt == "hw":
            P10_actual = histogram_step_wise_integral(
                pdf_new, bin_[0], T10_shifted)
            P50_actual = histogram_step_wise_integral(
                pdf_new, bin_[0], T50_shifted)
        elif wt == "cs":
            P10_actual = histogram_step_wise_integral(
                pdf_new, bin_[0],T10_shifted)
            P50_actual = histogram_step_wise_integral(
                pdf_new, bin_[0],T50_shifted)
        else:
            raise ValueError("MEWS only handles 'cs', and 'hw' for wave types presently!")
        
        if wt == "hw":
            residual2 = (P10 - P10_actual)**2 + (P50 - P50_actual)**2
        else:
            residual2 = 20**2*(P10 - P10_actual)**2 + 200**2*(P50 - P50_actual)**2
            
        # penalties if the distribution doesn't even contain the shifted T10 and T50
        if (T10_shifted > pdf_new[1][-1] and wt == "hw"):
            residual2 += (T10_shifted - pdf_new[1][-1])**2
        elif (T10_shifted < pdf_new[1][0] and wt == "cs"):
            residual2 += 20**2*(T10_shifted - pdf_new[1][0])**2

        # larger terms are needed on "cs" because the probabilities
        # are being assessed at 0.005, 0.00005 rather than at 0.98, 0.99 etc...
        if (T50_shifted > pdf_new[1][-1] and wt == "hw"):
            residual2 += (T50_shifted - pdf_new[1][-1])**2
        elif (T50_shifted < pdf_new[1][0] and wt == "cs"):
            residual2 += 200**2*(T50_shifted - pdf_new[1][0])**2

        if return_new_pdf:
            return residual2 + neg_deriv_penalty, pdf_new, P10_actual, P50_actual
        else:
            return residual2 + neg_deriv_penalty
    # beginning of function.
    hist0_stretched = {}
    prob_vals = {}
    param_val = {}
    asol_val = {}

    for wt, hist0_wt in hist0.items():

        if not ipcc_shift[wt] is None:
            
            num_hist_wave = hist0_wt[0].sum()

            cdf0 = ((hist0_wt[0]/hist0_wt[0].sum()).cumsum(),
                    (hist0_wt[1][1:] + hist0_wt[1][0:-1])/2)

            (P0_less_than_extreme, P_T_target_less_than_extreme,
             P10_given_wave, P50_given_wave) = calculate_probabilities(num_hist_wave,
                                                                       historic_time_interval,
                                                                       hours_per_year,
                                                                       ipcc_shift[wt],
                                                                       wt)
            Tthresh_historic = linear_interp_discreet_func(
                P0_less_than_extreme,
                cdf0)

            # These are the exact values needed
            T_shifted_target = Tthresh_historic + \
                np.array([ipcc_shift[wt]['temperature']['10 year'],
                         ipcc_shift[wt]['temperature']['50 year']])

            param = [P_T_target_less_than_extreme[0],
                     P_T_target_less_than_extreme[1],
                     Tthresh_historic[0],
                     Tthresh_historic[1],
                     T_shifted_target[0],
                     T_shifted_target[1],
                     hist0_wt,
                     wt,
                     False]
            # THIS SHOULD EXHIBIT A UNIQUE PERFECT SOLUTION UNLESS THE HISTOGRAM IS TOO COARSE
            Asol = differential_evolution(objective_function, bounds=[
                                          [-1e-3, 1e-3], [-5e-2, 5e-2], [-1e-1, 1e-1], [-1.0, 1.0], [0, 5], [0, 1]], args=(param,),tol = 0.000001)
            param[-1] = True
            residuals2, pdf_new, P10_actual, P50_actual = objective_function(
                Asol.x, param)
            
            if Asol.fun > 1e-5:
                warn("A perfect solution for shifting the temperature "+
                     "distribution was not found. A square residual on 2 "+
                     "probability targets of {0:10.8e} remains!".format(residuals2))
                    
            # return to a quantity.
            hist0_stretched[wt] = (pdf_new[0]*hist0[wt][0].sum(), pdf_new[1])
            
            if plot_results:
                pdf_norm_area = histogram_step_wise_integral(hist0[wt], hist0[wt][1].min(), hist0[wt][1].max())
                pdf = (hist0[wt][0]/pdf_norm_area, hist0[wt][1])
                fig,ax = plt.subplots(1,1)
                ax.stairs(pdf[0], pdf[1])
                ax.stairs(pdf_new[0],pdf_new[1])
                ax.grid("on")
                ax.set_xlabel("$\Delta$T ($^{\circ}$C)")
                ax.set_ylabel("Probability Density Function")
            
            prob_vals[wt] = {"actual":{"10 year":P10_actual,"50 year":P50_actual},
                             "target":{"10 year":P_T_target_less_than_extreme[0],"50 year":P_T_target_less_than_extreme[1]}}
            param_val[wt] = param
            asol_val[wt] = Asol
        else:
            hist0_stretched[wt] = None
            prob_vals[wt] = None
            param_val[wt] = None
            asol_val[wt] = None
            
    return hist0_stretched, asol_val, objective_function, param_val, prob_vals

def calculate_probabilities(num_hist_wave, historic_time_interval, hours_per_year, ipcc_shift, wave_type):
    # Calculate probabilities
    P10_given_wave, P50_given_wave = probability_of_extreme_10_50(
        num_hist_wave, historic_time_interval, hours_per_year)
    # probabilities a heat wave being less than the 10 and 50 year events
    if wave_type == "cs":
        P0_less_than_extreme = np.array([P10_given_wave,P50_given_wave])
        P_T_target_less_than_extreme = np.array([P10_given_wave * ipcc_shift['frequency']['10 year'],
                                                 P50_given_wave * ipcc_shift['frequency']['50 year']])
    else:
        P0_less_than_extreme = np.array([1.0-P10_given_wave, 1.0-P50_given_wave])
        P_T_target_less_than_extreme = np.array([1.0 - P10_given_wave * ipcc_shift['frequency']['10 year'],
                                                 1.0 - P50_given_wave * ipcc_shift['frequency']['50 year']])

    # target probablities
    

    return P0_less_than_extreme, P_T_target_less_than_extreme, P10_given_wave, P50_given_wave


def ipcc_shift_comparison_residuals(hist0, ipcc_shift,
                                    histT_tuple,
                                    durations,
                                    output_hist,
                                    delT_above_shifted_extreme,
                                    num_hist_wave,
                                    historic_time_interval,
                                    hours_per_year,
                                    wave_type,
                                    weights,
                                    hist0_stretched=None):
    # get and calculate cdf's for historical (0) and future (_T) distributions
    # The hist0_strechted=None is for ObjectiveFunction.__init__ to
    # go ahead and calculate and return a strethced hist0 that becomes the
    # target distribution.

    cdf0 = ((hist0[0]/hist0[0].sum()).cumsum(),
            (hist0[1][1:] + hist0[1][0:-1])/2)
    cdf_T = histT_tuple[1]

    # produce a large negative penalty for distributions that produce negative
    # temperature heat waves.
    if wave_type == "hw":
        negative_penalty = (
            np.max([float((cdf_T[1] < 0).sum()) - 1.0, 0.0]))**2 * weights[2]
    else:
        negative_penalty = (
            np.max([float((cdf_T[1] > 0).sum()) - 1.0, 0.0]))**2 * weights[2]

    (P0_less_than_extreme, P_T_target_less_than_extreme,
     P10_given_wave, P50_given_wave) = calculate_probabilities(num_hist_wave,
                                                               historic_time_interval,
                                                               hours_per_year,
                                                               ipcc_shift,
                                                               wave_type)
    if len(cdf_T[0]) == 1:
        T_shifted_actual = np.diff(histT_tuple[0][1])[0]*P0_less_than_extreme + histT_tuple[0][1][0]
    else:
        T_shifted_actual = linear_interp_discreet_func(
            P_T_target_less_than_extreme, cdf_T, False)

    Tthresh_historic = linear_interp_discreet_func(
        P0_less_than_extreme,
        cdf0)

    # These are the exact values needed
    T_shifted_target = Tthresh_historic + \
        np.array([ipcc_shift['temperature']['10 year'],
                 ipcc_shift['temperature']['50 year']])

    # penalize temperatures much hotter than 50 year peak temperature
    # The square root amplifies the low probability numbers to create increased residuals
    if wave_type == "hw":
        positive_penalty = ((histT_tuple[0][0][cdf_T[1] > (
            T_shifted_target[1] + delT_above_shifted_extreme)]).sum())**0.5 * weights[1]
    else:
        positive_penalty = ((histT_tuple[0][0][cdf_T[1] < (
            T_shifted_target[1] + delT_above_shifted_extreme)]).sum())**0.5 * weights[1]

    if not hist0_stretched is None:

        shape_penalty = histogram_comparison_residuals(histT_tuple[0],hist0_stretched, weights[4], normalize=True)
    else:
        
        shape_penalty = histogram_comparison_residuals(histT_tuple[0],hist0, weights[4], normalize=True)
        
        
    residuals = ((T_shifted_actual - T_shifted_target) /
                 T_shifted_target)**2 * weights[0]

    if output_hist:
        return (residuals + negative_penalty + positive_penalty,
                {'actual': T_shifted_actual, 'target': T_shifted_target},
                residuals,
                negative_penalty,
                positive_penalty)

    return residuals + negative_penalty + positive_penalty + shape_penalty


def probability_of_extreme_10_50(num_hist_wave, historic_time_interval, hours_per_year):
    
    """
    This is the probability that a wave is the 10 year or 50 year wave given 
    that an event has occurred. Since MEWS is monthly, the probability the
    event will occur in January is 
    
    wave_rate_current_month = (number historical waves in current month/
                               total historical time in current month)
    P10_given_wave = 1 / (wave_rate_current_month * total time in 10 years)
    
    Notice that we multiply the wave rate by the total time in 10 years. 
    Otherwise, you would be looking at 12 50 year events in the next 50 years
    if you multiplied by the total time in 10 years in the current month.
    
    """

    num_hour_in_year = 8760

    if num_hist_wave == 0:
        return 0.0, 0.0
    else:
        N10 = 10 * num_hour_in_year
        N50 = 50 * num_hour_in_year

        # TODO - hours per year is not even needed.
        P10_given_wave = historic_time_interval * \
            (hours_per_year / num_hour_in_year) / (num_hist_wave * N10)
        P50_given_wave = historic_time_interval * \
            (hours_per_year / num_hour_in_year) / (num_hist_wave * N50)

        return P10_given_wave, P50_given_wave


def duration_residuals_func(duration, duration0, ipcc_shift, weights):
    """
    This function penalizes incorrect changes in the number of waves.
    If ipcc_shift['frequency']['10 year'] > 1, it penalizes decrease in the number of heat
    waves and also penalizes increase beyond num_waves0 * ipcc_shift['frequency']['10 year']

    If ipcc_shift['frequency']['10 year'] < 1, it penalizes increase in the number of heat
    waves and also penalizes decreases beyond num_waves0 * ipcc_shift['frequency']['10 year']

    if ipcc_shift['frequency']['10 year'] it calculates a residual according to histogram comparison residuals
    in an attempt to make the number of waves equivalent.
    """
    def bounding_penalty(duration0, duration, ipcc_shift, normalizing_factor, is_increase):

        bound_duration = (
            duration0[0]*ipcc_shift['frequency']['10 year'], duration0[1])
        area = histogram_area(duration)

        if is_increase:
            area0 = histogram_area(duration0)
            int_area = histogram_intersection(duration0, duration)
            int_area_max = histogram_intersection(duration, bound_duration)
        else:
            area0 = histogram_area(bound_duration)
            int_area = histogram_intersection(bound_duration, duration)
            int_area_max = histogram_intersection(duration0, duration)

        # lower_bound
        if area0 - int_area > 0:
            penalty = (area0 - int_area)/normalizing_factor
        else:
            penalty = 0.0

        # upper bound
        if area - int_area_max > 0:
            penalty += (area - int_area_max)/normalizing_factor

        return penalty

    match = (ipcc_shift is None) or (ipcc_shift['frequency'] == 1)

    normalizing_factor = 0.5*np.min([duration[0].max()*(duration[1].max()-duration[1].min()),
                                     duration0[0].max()*(duration0[1].max() - duration0[1].min())])

    if match:
        # we want actual numbers. - weight not included here but at return of penalty.
        residuals = histogram_comparison_residuals(
            duration, duration0, 1.0, False)/(normalizing_factor)
        penalty = residuals.sum()
    else:
        is_increase = ipcc_shift['frequency']['10 year'] > 1
        penalty = bounding_penalty(
            duration0, duration, ipcc_shift, normalizing_factor, is_increase)
        # add additional penalty for non-overlapping area.
        penalty += histogram_non_overlapping(duration0,
                                             duration)/normalizing_factor

    return penalty * weights[3]


def histogram_comparison_residuals(histT, hist0, weight, normalize=True):
    """
    This function calculates the difference between two probability density functions
    of discrete historgrams. The histograms must have equal spacing for their
    bin edges.

    histT
    hist0 are both np.historgram tuple output.

    """

    if normalize:
        area0 = histogram_step_wise_integral(hist0)
        areaT = histogram_step_wise_integral(histT)
        hist0_temp = hist0[0]/area0
        histT_temp = histT[0]/areaT
    else:
        hist0_temp = hist0[0]
        histT_temp = histT[0]

    area_residual = histogram_non_intersection(
        (hist0_temp, hist0[1]), (histT_temp, histT[1]))

    # need an array so that other functions work out fine
    return np.array([area_residual * weight])


def unpack_coef_from_x(x, decay_func_type, events):
    coef = {}
    xtoff = x_offset_for_func_type_combo(decay_func_type, events)
    for eid, wt in enumerate(events):
        decay_func = decay_func_type[wt]

        if not decay_func is None:
            offset = np.array(xtoff)[:eid].sum()
            if decay_func == "exponential" or decay_func == "linear":
                coef[wt] = np.array([x[8+offset]])
            elif decay_func == "exponential_cutoff" or decay_func == "linear_cutoff":
                coef[wt] = np.array([x[8+offset], x[9+offset]])
            elif decay_func == "quadratic_times_exponential_decay_with_cutoff":
                # remember the order of the coefficient input for this function is
                # 1) time to maximum 2) maximum prob, 3) cutoff time
                # the order has to be 6,10,8 as a result instead of 6,8,10
                coef[wt] = np.array([x[8+offset], x[9+offset], x[10+offset]])
            elif decay_func == DECAY_FUNC_NAMES[6]:
                coef[wt] = np.array([x[8+offset], x[9+offset], x[10+offset]])
            else:  # should never happen:
                raise ValueError("Invalid decay_func_type. '{0}' was input, but only valid values are : \n\n{1}".format(
                    decay_func, str(SolveDistributionShift._decay_func_types_entries.keys())))
        else:
            coef[wt] = None
    return coef


class ObjectiveFunction():

    def __init__(self, events, random_seed):
        self.iterations = 0
        self._events = events
        if random_seed is None:
            self._rng = default_rng()
        else:
            self._rng = default_rng(random_seed)
        self._hours_in_year = 8760

    def markov_gaussian_model_for_peak_temperature(self, x,
                                                   num_step,
                                                   hist_param,
                                                   hist0,
                                                   durations0,
                                                   historic_time_interval,
                                                   hours_per_year,
                                                   ipcc_shift=DEFAULT_NONE_DECAY_FUNC,
                                                   decay_func_type=DEFAULT_NONE_DECAY_FUNC,
                                                   use_cython=True,
                                                   output_hist=False,
                                                   delT_above_shifted_extreme=DEFAULT_NONE_DECAY_FUNC,
                                                   weights=WEIGHTS_DEFAULT_ARRAY,
                                                   min_num_waves=100,
                                                   hist0_stretched=DEFAULT_NONE_DECAY_FUNC):
        """
        This is an objective function. It finds residuals on one of the following
        criterion
            If ipcc_shift = None.
                Historic Calibration
                --------------------
               The function takes the histogram in "hist0" and the Markov-Truncated
               guassian temperature profiles generated by the input vector x is used to generate "num_step" of a 
               dynamic heat wave process. The histogram of heat wave events for
               this is then produced and aligned to hist0 bin intervals. Residuals
               are then calculated based on the sum of the square of difference in magnitude of the 
               normalized histograms. 
           else
                Future Calibration
                ------------------
               The historical histogram is used to find 10 and 50 year Temperature 
               thresholds (i.e. the temperature at which all events happen in more than
                           10 year and in more than 50 years)
               and then shifts them by ipcc_shift['temperature']['10 year'] and
               ipcc_shift['temperature']['50 year']. It then multiplies the probability
               of the historic 10 and 50 year events by ipcc_shift['frequency']['10 year']
               and ipcc['frequency']['50 year'] and runs the Markov-Trunc-Guassian process
               reflected by the input vector x. The resulting temperature thresholds for the 
               shifted probabilities are found and compared to the ipcc shifted temperatures
               the sum of square of differences between these is the residual.

        Parameters
        ----------

        x : array-like : 
            Must be a 1-D array with the following entries
            x[0] = Change in cold snap mean of truncated guassian for scaled, duration 
                   normalized temperature. For historic case this should always be
                   zero.

            x[1] = Change in cold snap standard deivation of truncated... (see above)

            x[2], x[3] same as x[0] and x[1] but for heat waves.

            x[4] = Probability of a cold snap occuring when markov state = 
                   not in cold snap or heat wave (i.e. 0)

            x[5] = Probability of a heat wave occuring when markov state = not in
                   a cold snap or heat wave (i.e. 0)

            x[6] = Probability of sustaining a cold snap at time in wave = 0

            x[7] = Probability of sustaining a heat wave at time in wave = 0

            All other variables depend on the decay_func_type dictionary

            if decay_func_type['cs'] is None and decay_func_type['hw'] is None:

                There are no other variables

            if decay_func_type['cs'] == "exponential" or "linear" and decay_func_type['hw'] is None

                x[8] = slope or exponential parameter (depending on 
                    'decay_func_type') decay of probability sustaining cold snaps

            if decay_func_type['cs'] == "exponential_cutoff" or "linear_cutoff" and decay_func_type['hw'] is None

                x[8] = slope or exponential parameter (depending on 
                    'decay_func_type') decay of probability sustaining cold snaps
                x[9] = cutoff time at which probability of sustaining a cold snap
                       drops to zero

            if decay_func_type['cs'] == "quadratic_times_exponential_decay_with_cutoff" and decay_func_type['hw'] is None

                x[8] = time to peak probability of sustaining a cold snap
                x[9] = cutoff time at which probability of sustaining a cold snap
                       drops to zero
                x[10] = Maximum probability of sustaining a cold snap

            .
            .   Not listing all 16 combinations but the idea is that 'cs' comes first then 'hw'
            .

            if  decay_func_type['cs'] == "quadratic_times_exponential_decay_with_cutoff" 
            and decay_func_type['hw'] == "quadratic_times_exponential_decay_with_cutoff"

                x[8] = time to peak probability of sustaining a cold snap
                x[9] = cutoff time at which probability of sustaining a cold snap
                       drops to zero
                x[10] = Maximum probability of sustaining a cold snap

                x[11] = Time to peak probability of sustaining a heat wave
                x[12] = Maximum probability of sustaining a heat wave
                x[13] = Cutoff time at which probability of sustaining a heat wave
                       drops to zero





         Other inputs are documented in input to "SolveDistributionShift.__init__"

        """
        # We want the same set of random numbers every time so that the
        # Markov process is not causing as much variation
        rng = self._rng

        # this function is used in optimization routines so it needs to be efficient!

        # first two parameters are always duration normalized and scaled temperature
        # mean and standard deviation
        del_mu_T = {}
        del_sig_T = {}
        del_mu_T['cs'] = x[0]
        del_sig_T['cs'] = x[1]
        del_mu_T['hw'] = x[2]
        del_sig_T['hw'] = x[3]

        Pcs = x[4]
        Phw = x[5]
        Pcss = x[6]
        Phws = x[7]

        # next parameters are the probability of sustainting a cold snap and then a
        # heat wave
        tran_matrix = np.array([[1-Pcs-Phw, Pcs,  Phw],
                                [1-Pcss,    Pcss, 0.0],
                                [1-Phws,    0.0,  Phws]])

        # next parameters depend on the func_type. They do not exist if
        # there is no decay function
        coef = unpack_coef_from_x(x, decay_func_type, self._events)

        # now run the Markov process
        objDM = DiscreteMarkov(rng,
                               tran_matrix,
                               decay_func_type=decay_func_type,
                               coef=coef,
                               use_cython=use_cython)

        states_arr = objDM.history(num_step, 0)

        state_intervals = find_extreme_intervals(states_arr, [1, 2])
        durations = {}
        Tsample = {}
        residuals = {}
        histT_tuple = {}
        thresholds = {}
        duration_residuals = {}
        duration_histograms = {}
        unnormalized_duration_histograms = {}
        temp_resid = {}
        negative_penalty = {}
        positive_penalty = {}

        for wave_type, state_int in zip(ABREV_WAVE_NAMES, [1, 2]):

            # The number of historic heat waves.
            num_hist_wave = hist0[wave_type][0].sum()

            # only look at heat wave effects. Cold snaps do effect things but only
            # in the sense that they displace heat waves.

            durations[wave_type] = np.array(
                [tup[1]-tup[0]+1 for tup in state_intervals[state_int]])

            Tsample[wave_type] = evaluate_temperature(durations[wave_type], hist_param,
                                                      del_mu_T[wave_type],
                                                      del_sig_T[wave_type], rng, wave_type)

        # this returns a tuple the first element is the histogram of temperatures
        # the second element returns the cdf with values mapped to the bin averages.
            if len(Tsample[wave_type]) < min_num_waves:
                # for cases where Markov process produces not enough heat waves. return
                # 99999 residual recognizable at solution. We cannot get statistics on
                # processes that are not creating events.
                residuals[wave_type] = np.array([44444, 55555])*weights.sum()
                thresholds[wave_type] = None
                histT_tuple[wave_type] = None
                duration_histograms[wave_type] = None
                duration_residuals[wave_type] = 0.0
                negative_penalty[wave_type] = None
                positive_penalty[wave_type] = None
                temp_resid[wave_type] = None

            else:
                if Tsample[wave_type].max() > 60:
                    warn("Heat waves of greater than 60 C are not realistic")

                # normal evaluation cases.

                histT_tuple[wave_type] = create_complementary_histogram(
                    Tsample[wave_type], hist0[wave_type])

                # convert the durations to duration histograms for plotting later on.
                # You must normalize num_step vs. historic_time_interval * hours_per_year/hours_in_year
                duration_normalizing_factor = (
                    historic_time_interval * hours_per_year / self._hours_in_year) / num_step

                unnormalized_duration_histograms[wave_type] = np.histogram(
                    durations[wave_type], bins=np.arange(24-12, durations[wave_type].max()+12+24, 24))

                duration_histograms[wave_type] = (unnormalized_duration_histograms[wave_type][0]*duration_normalizing_factor,
                                                  unnormalized_duration_histograms[wave_type][1])

                if duration_histograms[wave_type][0].sum() > min_num_waves:

                    duration_residuals[wave_type] = duration_residuals_func(
                        duration_histograms[wave_type], durations0[wave_type], ipcc_shift[wave_type], weights)

                else:
                    # this means very few waves are coming through and a statistical analysis is invalid. Huge Penalty
                    duration_residuals[wave_type] = 1e5/duration_histograms[wave_type][0].sum()

                if ipcc_shift[wave_type] is None:

                    residuals[wave_type] = histogram_comparison_residuals(
                        histT_tuple[wave_type][0], hist0[wave_type], weights[0])
                    thresholds[wave_type] = None
                    negative_penalty[wave_type] = None
                    positive_penalty[wave_type] = None
                    temp_resid[wave_type] = None

                else:

                    if output_hist:
                        residuals[wave_type], thresholds[wave_type], temp_resid[wave_type], negative_penalty[wave_type], positive_penalty[wave_type] = ipcc_shift_comparison_residuals(
                            hist0[wave_type], ipcc_shift[wave_type],
                            histT_tuple[wave_type], durations[wave_type],
                            output_hist, delT_above_shifted_extreme[wave_type],
                            num_hist_wave, historic_time_interval, hours_per_year, 
                            wave_type, weights, hist0_stretched[wave_type])
                    else:
                        residuals[wave_type] = ipcc_shift_comparison_residuals(hist0[wave_type], ipcc_shift[wave_type],
                                                                               histT_tuple[wave_type], durations[wave_type],
                                                                               output_hist, delT_above_shifted_extreme[
                                                                                   wave_type],
                                                                               num_hist_wave, historic_time_interval, 
                                                                               hours_per_year, wave_type, weights, hist0_stretched[wave_type])

        sum_resid = 0.0
        for wt, resid in residuals.items():
            sum_resid = sum_resid + resid.sum()
            sum_resid = sum_resid + duration_residuals[wt]

        # if you need more output after differential_evolution
        self.iterations += 1

        if output_hist:
            residuals_dict = {}
            residuals_dict['total temperature'] = {'cs': residuals['cs'].sum(),
                                                   'hw': residuals['hw'].sum()}
            residuals_dict['temperature'] = {}
            for wt in self._events:
                if not temp_resid[wt] is None:
                    residuals_dict['temperature'][wt] = temp_resid[wt].sum()

            residuals_dict['temperature negative penalty'] = negative_penalty
            residuals_dict['temperature positive penalty'] = positive_penalty
            residuals_dict['durations'] = duration_residuals

            return sum_resid, Tsample, duration_histograms, thresholds, histT_tuple, residuals_dict

        else:
            return sum_resid


class SolveDistributionShift(object):

    """
    This class numerically evaluates how a combined markov chain and 
    duration normalized truncated gaussian distribution on temperature
    (but not energy) shift the frequency and severity of maximum temperature
    events. 

    It starts with the historical 10 year and 50 year peak temperatures 
    defined as the temperatures : P(T < T_50_max) = 1 - 1/(50*number_of_heat_waves_per_year)
                                  P(T < T_10_max) = 1 - 1/(10*number_of_heat_waves_per_year)


    """

    _variable_order_string = """ x0[0] - change in average of scaled, duration normalized temperature
                                 x0[1] - change in standard deviation of scaled, duration normalized temperature
                                 x0[2] - hourly probability of a cold snap at non-extreme times 
                                 x0[3] - hourly probability of a heat wave at non-extreme times
                                 x0[4] - hourly probability of sustaining a cold snap when in a cold snap
                                 x0[5] - hourly probability of sustaining a heat wave when in a heat wave
                                 if not decay_func_type is None:
                                     if decay_func_type == 'exponential'
                                         x0[6] - lambda of exp(-lambda * time_in_wave) for decaying probability of sustaining a cold snap
                                         x0[7] - same as x0[6] for a heat wave
                                    elif decay_func_type == 'linear'
                                        x0[6] - slope of 1 - slope*time_in_wave for decaying probability of sustaining a cold snap
                                        x0[7] - same as x0[6] for a heat wave
                                    else
                                        if decay_func_type == 'quadratic_times_exponential_decay_with_cutoff':
                                            x0[6:7] = time to maximum probability for cold snaps and heat waves 
                                        else:
                                            x0[6:7] - lambda or slope as defined above
                                       
                                        x0[8:9] - cutoff times at which probability is set to zero
                                        
                                        if decay_func_type == 'quadratic_times_exponential_decay_with_cutoff':
                                            x0[10:11] - maximum sustained probability of cold snaps and heat waves.
                                        
    
                             """

    _ipcc_shift_struct = """
                            ipcc_shift['temperature'] = {'10 year':'=IPCC predicted change in temperature for future 10 year heat wave events',
                                                         '50 year':'=IPCC predicted change in temperature for future 50 year heat wave events'
                                                         }
                            ipcc_shift['frequency'] = {'10 year':'=IPCC predicted increase of frequency of future 10 year heat wave events',
                                                       '50 year':'=IPCC predicted increase of frequency of future 50 year heat wave events'
                                                       }
    
                        """
    _ipcc_shift_dict_str = {'temperature': [
        '10 year', '50 year'], 'frequency': ['10 year', '50 year']}

    _param0_struct = """    param0['normalizing duration'] = maximum duration heat wave in the historic record
                            param0['extreme_temp_normal_param']['mu'] = mean of trucated Gaussian distribution 
                                                                        for scaled, duration normalized temperature 
                                                                        calculated from MEWS historic algorithm
                            param0['extreme_temp_normal_param']['sig'] = standard deviation of the same truncated
                                                                        Gaussian distribution
                                                                        
                            param0['max extreme temp per duration'] = maximum value for temperature per duration 
                                                                      for historic heat waves. This value is used
                                                                      to reverse scale from the interval -1...1                                           
                            param0['min extreme temp per duration'] = minimum value for temperature per duration
                                                                      for historic cold snaps
                            param0['normalizing extreme temp'] = Maximum historic heat wave temperature
                            param0['normalized extreme temp duration fit slope'] = Linear regression slope of heat wave normalized temperature [0..1] versus
                                                                                   normalized heat wave duration [0..1]
                            param0['normalized extreme temp duration fit intercept'] = Linear regression intercept
                            
                        """

    _bound_desc = ['delT_mu',
                   'delT_sig multipliers',
                   'P_event', 'P_sustain',
                   'multipliers to max probability time',
                   'slope or exponent multipliers',
                   'cutoff time multipliers',
                   'max peak prob for quadratic model',
                   'delay_time_multipliers']

    _default_problem_bounds_all = {ABREV_WAVE_NAMES[0]: {_bound_desc[0]: (0.0, 2.0),
                                                         _bound_desc[1]: (-0.1, 4),
                                                         _bound_desc[2]: (0.00001, 0.0125),
                                                         _bound_desc[3]: (0.958, 0.999999),
                                                         _bound_desc[4]: (0, 2),
                                                         _bound_desc[5]: (0, 1),
                                                         _bound_desc[6]: (1, 3),
                                                         _bound_desc[7]: (0.97, 1.0),
                                                         _bound_desc[8]: (0, 3)},
                                   ABREV_WAVE_NAMES[1]: {_bound_desc[0]: (0.0, 2.0),
                                                         _bound_desc[1]: (-0.1, 2),
                                                         _bound_desc[2]: (0.00001, 0.0125),
                                                         _bound_desc[3]: (0.958, 0.999999),
                                                         _bound_desc[4]: (0.1, 2),
                                                         _bound_desc[5]: (0, 1),
                                                         _bound_desc[6]: (1, 3),
                                                         _bound_desc[7]: (0.97, 1.0),
                                                         _bound_desc[8]: (0, 3)}}
    _lin_exp_list = _bound_desc[0:4]
    _lin_exp_list.append(_bound_desc[5])
    _lin_exp_cutoff_list = _lin_exp_list
    _lin_exp_cutoff_list.append(_bound_desc[6])
    _quad_list = [_bound_desc[0], _bound_desc[1], _bound_desc[2],
                  _bound_desc[3], _bound_desc[4], _bound_desc[6], _bound_desc[7]]
    _delay_list = [_bound_desc[0], _bound_desc[1], _bound_desc[2],
                   _bound_desc[3], _bound_desc[5], _bound_desc[6], _bound_desc[8]]

    # this provides a basis for evaluating if problem_bounds has all the entries needed.
    _decay_func_types_entries = {DECAY_FUNC_NAMES[0]: _bound_desc[0:4],
                                 DECAY_FUNC_NAMES[2]: _bound_desc[0:4],
                                 DECAY_FUNC_NAMES[1]: _lin_exp_list,
                                 DECAY_FUNC_NAMES[4]: _lin_exp_cutoff_list,
                                 DECAY_FUNC_NAMES[3]: _lin_exp_cutoff_list,
                                 DECAY_FUNC_NAMES[5]: _quad_list,
                                 DECAY_FUNC_NAMES[6]: _delay_list}

    # cs = cold snap, hw = heat wave.
    # THE ORDER OF THIS LIST MUST NOT BE CHANGED!!! IT WILL SCRAMBLE THE ORDER
    # OF VARIABLES X FOR THE OBJECTIVE FUNCTION
    _events = ABREV_WAVE_NAMES

    _valid_inputs = VALID_SOLVE_INPUTS

    def __init__(self, num_step,
                 param0,
                 random_seed,
                 hist0,
                 durations0,
                 delT_above_shifted_extreme,
                 historic_time_interval,
                 hours_per_year,
                 problem_bounds=None,
                 ipcc_shift={'cs': None, 'hw': None},
                 decay_func_type={'cs': None, 'hw': None},
                 use_cython=True,
                 num_cpu=-1,
                 plot_results=True,
                 max_iter=20,
                 plot_title="",
                 out_path="",
                 weights=WEIGHTS_DEFAULT_ARRAY,
                 limit_temperatures=True,
                 min_num_waves=100,
                 x_solution=None,
                 test_mode=False,
                 num_postprocess=3,
                 extra_output_columns={}):
        """
        This class solves one of two problems. 

        1. The first is to fit the extreme event
        histograms for heat waves and cold snaps to output from a markov-truncated-gaussian (MTG)
        an evoluationary optimization is used to vary the parameters of the MTG model 
        to minimize the residual between the normalized histograms to the histograms created
        by running the MTG model num_steps

        2. The second is to minimize residuals between shifted 10 and 50 year
        heat wave and cold snap events which have increased in frequency and
        in intensity.


        Parameters
        ----------

        num_step : int : 
            number of hourly steps to run the MTG model. Must be at least be
            10 50 year periods or > 50*8760*10 more steps smooths out the 
            optimization residuals but increases run time.

        param0 : dict :
            dictionary of specific form. See self._param0_struct. It contains
            historical values from the mews.events.extreme_temperature_waves 
            analysis.

        random_seed : int :
            An integer < 2**32-1 that enables selection of quasi-random numbers

        hist0 : dict :
            dictionary with key 'hw' and 'cs' that give histograms of 
            heat wave/cold snap temperatures from np.histogram. This is the 
            target distribution for optimization (problem 1)
            or from which 10 and 50 year tempeatures are derived (problem 2)

        durations0 : dict :
            same dictionary form as hist0. This is a histogram of heat wave 
            durations for the historic record.  

        delT_above_shifted_extreme : dict :
            dictionary wit key 'hw' and 'cs' which contain float values that
            limit how hot a heat wave can get above the shifted 50 year event
            and how colder temperatures can get below the 50 year cold snap event.
            Only applies to problem 1. Assign "None" for problem 1.

        historic_time_interval : int :
            The number of hours included in the historic (hist0, durations0) data
            analyzed. This is the length of the daily summaries file times 24.

        hours_per_year : int 
            For a full year analysis, this should be 8760 or 8784 on a leap year
            For the mews monthly analysis, it is the number of hours in the 
            current month being analyzed.

        problem_bounds : dict : Optional: Default value = None
            list of tuples of length 8 for both heat waves 'hw' and cold snaps 'cs'
            depending on input 'decay_func_type'
            A special form of dictionary that contains bounds on the parameters 
            for the variable vector 'x'. If None, values from 
            self._default_problem_bounds are used.
            which contains all options regardless of whether they are needed for a 
            specific decay_func_type. All entries are needed regardless of 
            the function type

            _default_problem_bounds = {'cs':{'delT_mu': (0.0, 2.0),
                                 'delT_sig multipliers': (-0.1,4),
                                 'P_event': (0.00001, 0.03),
                                 'P_sustain': (0.958, 0.999999),
                                 'multipliers to max probability time': (0,2),
                                 'slope or exponent multipliers' : (0,1),
                                 'cutoff time multipliers':(1,3),
                                 'max peak prob for quadratic model': (0.97, 1.0)},
                           'hw':{'delT_mu': (0.0, 0.7),
                                 'delT_sig multipliers': (-0.1,2),
                                 'P_event': (0.00001,0.03),
                                 'P_sustain': (0.958,0.999999),
                                 'multipliers to max probability time': (0.1,2),
                                 'slope or exponent multipliers' : (0,1),
                                 'cutoff time multipliers':(1,3),
                                 'max peak prob for quadratic model': (0.97, 1.0)}}


        ipcc_shift : dict : optional : default value = None
            A special form of dictionary which contains shifts in frequency
            and intensity of heat waves and cold snaps. If none, Problem 1 is
            solved if not None, problem 2 is solved. See self._ipcc_shift_struct
            for the structure needed.

        decay_func_type : str : optional : Default = None
            one of several strings indicating the type of decay probability of 
            sustaining an extreme event undergoes:

            valid values are : [None,     8
                                'linear',   10
                                'exponential',   10
                                'linear_cutoff',   12
                                'exponential_cutoff',  12
                                "quadratic_times_exponential_decay_with_cutoff" 14]
            Each of these change the number of parameters needed for x0
                None - a markov process with no decay or cutoff time for 
                       sustaining a heat wave
                linear - a drop in probability by P0 * (1-slope*time_in_wave) occurs
                         positive slope values are needed for cold snaps and heat waves

                exponential - exponential drop according to (1 - exp(-lambda * time_in_wave))

                linear_cutoff - same as linear but at a second time t_cutoff probability
                                drops to zero

                exponential_cutoff - same as exponential with t_cutoff

                quadratic... - Allows an increase to probability for a specified time 

                               P0 * (one + (time_in_wave / time_to_peak)**two * exp(two) * 
                                             (Pmax/P0 - 1)* exp(-two * time_in_state / time_to_peak))

                               time_to_peak : indicates the time required to
                                              reach Pmax

                               Pmax : indicates the maximum probability reached

                               t_cutoff is also used making this function have
                               3 parameters per wave type.

        use_cython : bool : optional: default = True
            Allows use of native python if False--which is much slower

        num_cpu : int : optional : default = -1
            indicate the number of processors to use in the differential_evolution
            optimization. If = -1 the maximum number available is used.

        plot_results : bool : optional: Default = True
            output plots from matplotlib.pyplot showing the optimization result

        max_iter : int : optional : Default = 20
            maximum number of generations that the differential_evolution
            algorithm will run. The problem stabilizes fairly quickly but will
            not converge because the optimization is stochastic.

        plot_title : str : optional : Default = ""
            Give a custom plot title to the output graphs from this analysis

        out_path : str : optional : Default = ""
            If a png file is desired for output then
            this will output the file to the indicated location.

        weights : np.array : optional : Default = np.array([1.0,1.0,1.0,1.0])
            Allows the four terms being optimized to be emphasized more or less 
            in the multi-objective optimization.

             weights[0] = multiplier on temperature pdf differences residuals
                          or IPCC residuals

             weights[1] = multiplier on penalty for overly high temperatures

             weights[2] = multiplier on penalty for negative temperature heat waves
                          (or positive temperature cold snaps)

             weights[3] = multiplier on duration residuals
             
             weights[4] = Future stretched distribution residuals

        limit_temperatures : bool : optional : Default = True,
            Controls whether to use delT_above_shifted_extreme as a hard 
            limit via a nonlinear constraint on cutoff time, and others. 
            This only has an effect on models that have a cutoff time.

        min_num_waves : int : optional : Default = 100
            The minimum number of waves (both heat wave and cold snap) 
            that must be achieved by 'num_steps' in the sample space for the 
            analysis to not send back a high penalty in the residuals. 
            The analysis being done is statistical and a good sample of
            heat waves is needed for the analysis to be meaningful.

        x_solution : np.array : optional : Default = None
            If not None, A solution x_solution, is immediately applied rather 
            than running the differential_evolution optimization.

        test_mode : bool : optional : Default = False
            Enter into a test mode that overrides some check_inputs
            so that faster evaluations of the functionality of this
            class can be accomplished.

        num_postprocess : int : optional : Default = 3
            The number of times the optimal solution is evaluated by the 
            stochastic model for postprocessing after the optimization.
            This allows visualization and output of the variation between
            stochastic realizations to verify that the solution is stable
            stochastically.

        extra_output_columns : dict : optional : Default = {}
            This allows for extra columns to be added to the output 
            of the total results table. keys are column names and values
            are values that will be applied to every row so that a larger
            data table can be assembled easily across a broad range of 
            categories such as confidence interval, climate scenario, and
            future year.


        Raises
        ------
        _check_ipcc_shift - Raises many TypeError and ValueError exceptions
                            to guard against incorrect input. 

        Returns
        -------
        object with old obj.param0 values and new obj.param results.
        several other values are stored in the object for downstream use.

           obj.df - a dataframe that contains all of the distribution information
                    calculated by this function

        """
        if not hasattr(self, 'inputs'):
            self.df = None
            self._write_csv = False

        self._default_problem_bounds = self._grab_relevant_bounds(
            decay_func_type, problem_bounds)
        self._test_mode = test_mode
        self._check_inputs(num_step,
                           random_seed,
                           param0,
                           hist0,
                           durations0,
                           delT_above_shifted_extreme,
                           historic_time_interval,
                           hours_per_year,
                           problem_bounds,
                           ipcc_shift,
                           decay_func_type,
                           use_cython,
                           num_cpu,
                           plot_results,
                           max_iter,
                           plot_title,
                           out_path,
                           weights,
                           limit_temperatures,
                           min_num_waves,
                           x_solution,
                           test_mode,
                           num_postprocess,
                           extra_output_columns)

        # determine limits on temperature
        abs_max_temp = {}
        # establish the absolute maximum temperature nonlinear constraint boundary on the optimization
        for wtype in self._events:
            if ipcc_shift[wtype] is None:
                abs_max_temp[wtype] = param0[wtype]['normalizing extreme temp'] + \
                    delT_above_shifted_extreme[wtype]  # this parameter
            else:
                abs_max_temp[wtype] = (param0[wtype]['normalizing extreme temp'] + ipcc_shift[wtype]['temperature']['50 year'] +
                                       delT_above_shifted_extreme[wtype])
                
        
        if problem_bounds is None:
            self._prob_bounds = self._default_problem_bounds
        else:
            self._prob_bounds = problem_bounds

        # linear constraint - do not allow Pcs and Phw to sum to more than specified amounts
        # nonlinear constraint - do not allow maximum possible sampleable temperature to exceed a bound.
        bounds_x0, linear_constraint, nonlinear_constraint = self._problem_bounds(decay_func_type,
                                                                                  num_step,
                                                                                  param0,
                                                                                  abs_max_temp,
                                                                                  limit_temperatures)

        # x_solution is a way to bypass optimization and to just evaluate the model on
        # a specific x_solution.
        if x_solution is None:

            constraints = [linear_constraint]
            # the nonlinear constrain is only enforceable when cutoff parameters are present
            for wt, nlc in nonlinear_constraint.items():
                if not nlc is None:
                    constraints.append(nlc)

            obj_func = ObjectiveFunction(
                self._events, random_seed)
            
            
            # Create reasonable future histogram targets that reflect the IPCC shift but smoothly stretch
            # the entire historical distribution
            (hist0_stretched, asol_val, objective_function, param_val, prob_vals
             ) = calculated_hist0_stretched(hist0,ipcc_shift, historic_time_interval,hours_per_year,plot_results)
            
            # this optimization is not expected to converge. The objective function is stochastic. The
            # optimization still finds a solution but the population does not tend to stabilize because
            # of the stochastic objective function.

            iterations = 0
            optimize_result = differential_evolution(obj_func.markov_gaussian_model_for_peak_temperature,
                                                     bounds_x0,
                                                     args=(num_step,
                                                           param0,
                                                           hist0,
                                                           durations0,
                                                           historic_time_interval,
                                                           hours_per_year,
                                                           ipcc_shift,
                                                           decay_func_type,
                                                           use_cython,
                                                           False,
                                                           delT_above_shifted_extreme,
                                                           weights,
                                                           min_num_waves,
                                                           hist0_stretched),
                                                     constraints=tuple(
                                                         constraints),
                                                     workers=num_cpu,
                                                     maxiter=max_iter,
                                                     seed=random_seed,
                                                     disp=False,
                                                     polish=False)  # polishing is a waste of time on a stochastic function.

            xf0 = optimize_result.x
        else:
            xf0 = x_solution
            obj_func = ObjectiveFunction(self._events, random_seed)
            if hasattr(self, 'optimize_result'):
                optimize_result = self.optimize_result
            else:
                optimize_result = None

        # TODO - establish level of convergence.
        
        self._is_solution_on_bounds(xf0,bounds_x0)
        
        Tsample = {}
        durations = {}
        resid = {}
        thresholds = {}
        histT_tuple = {}
        residuals_dict = {}
        for rand_adj in np.arange(1, num_postprocess+1):
            (resid[rand_adj], Tsample[rand_adj], durations[rand_adj],
             thresholds[rand_adj], histT_tuple[rand_adj], residuals_dict[rand_adj]) = obj_func.markov_gaussian_model_for_peak_temperature(
                xf0,
                num_step,
                param0,
                hist0,
                durations0,
                historic_time_interval,
                hours_per_year,
                ipcc_shift,
                decay_func_type,
                use_cython,
                output_hist=True,
                delT_above_shifted_extreme=delT_above_shifted_extreme,
                weights=weights,
                min_num_waves=min_num_waves)

        if plot_results or len(out_path) > 0:
            Graphics.plot_sample_dist_shift(
                hist0, histT_tuple, ipcc_shift, thresholds, self._events, plot_title, out_path, True, plot_results)

            if len(out_path) > 0:
                if len(out_path) > 4:
                    dur_out_path = out_path[:-4] + "_durations.png"
                else:
                    dur_out_path = out_path + "_durations.png"
            else:
                dur_out_path = ''

            Graphics.plot_sample_dist_shift(
                durations0, durations, ipcc_shift, thresholds, self._events, plot_title, dur_out_path, False, plot_results)

        # pack all results into a single self.df and write to csv if out_path != ""
        self._output_table(hist0, histT_tuple, durations0, durations,
                           ipcc_shift, thresholds, self._events, out_path,
                           extra_output_columns)

        # repackage everything for use in other parts of MEWS.
        self.optimize_result = optimize_result
        self.thresholds = thresholds
        self.residuals = resid
        self.Tsample = Tsample
        self.durations = durations
        self.histT_tuple = histT_tuple
        self.residuals_breakdown = residuals_dict

        param_new = deepcopy(param0)

        coef = unpack_coef_from_x(xf0, decay_func_type, self._events)

        # this is just another way of storing the results in the old "delta"
        # form.
        del_shifts = {}

        for wave_type, eid, idx in zip(self._events, [[0, 1, 4, 6], [2, 3, 5, 7]], [0, 1]):
            # eid - extreme event index - these indicate what indices apply to heat waves and cold snaps
            pval = unpack_params(param0, wave_type)
            del_mu_T = xf0[eid[0]]
            del_sig_T = xf0[eid[1]]
            param_new[wave_type]['extreme_temp_normal_param']['mu'] = pval["mu_T"] + del_mu_T
            param_new[wave_type]['extreme_temp_normal_param']['sig'] = pval["sig_T"] + del_sig_T
            param_new[wave_type]['hourly prob of heat wave'] = xf0[eid[2]]
            param_new[wave_type]['hourly prob stay in heat wave'] = xf0[eid[3]]
            param_new[wave_type]['decay function'] = decay_func_type[wave_type]

            # shift the absolute extremes of the truncated gaussian distributions!
            (del_a, del_b) = shift_a_b_of_trunc_gaussian(
                del_mu_T, pval["mu_T"], del_sig_T, pval["sig_T"])
            
            min_boundary = transform_fit(pval['minval_T'],pval['hist_minval_T'],pval['hist_maxval_T'])
            max_boundary = transform_fit(pval['maxval_T'],pval['hist_minval_T'],pval['hist_maxval_T'])
            
            param_new[wave_type]['min extreme temp per duration'] = inverse_transform_fit(min_boundary+del_a,
                                                                                          pval['maxval_T'],
                                                                                          pval['minval_T'])

            param_new[wave_type]['max extreme temp per duration'] = inverse_transform_fit(max_boundary+del_b,
                                                                                          pval['maxval_T'],
                                                                                          pval['minval_T'])
            param_new[wave_type]['decay function coef'] = coef[wave_type]

            del_shifts[wave_type] = {"del_mu_T": del_mu_T,
                                     "del_sig_T": del_sig_T,
                                     "del_a": del_a,
                                     "del_b": del_b}

        self.param0 = param0
        self.param = param_new
        self.del_shifts = del_shifts
        self.abs_max_temp = abs_max_temp
        self.inputs = {}

        # this enables the "reanalyze" function so that incremental
        # changes can be made to just this dictionary after the intitial call
        # to the class.
        self.inputs['num_step'] = num_step
        self.inputs['param0'] = param0
        self.inputs['random_seed'] = random_seed
        self.inputs['hist0'] = hist0
        self.inputs['durations0'] = durations0
        self.inputs['delT_above_shifted_extreme'] = delT_above_shifted_extreme
        self.inputs['historic_time_interval'] = historic_time_interval
        self.inputs['hours_per_year'] = hours_per_year
        self.inputs['problem_bounds'] = problem_bounds
        self.inputs['ipcc_shift'] = ipcc_shift
        self.inputs['decay_func_type'] = decay_func_type
        self.inputs['use_cython'] = use_cython
        self.inputs['num_cpu'] = num_cpu
        self.inputs['plot_results'] = plot_results
        self.inputs['max_iter'] = max_iter
        self.inputs['plot_title'] = plot_title
        self.inputs['out_path'] = out_path
        self.inputs['weights'] = weights
        self.inputs['limit_temperatures'] = limit_temperatures
        self.inputs['min_num_waves'] = min_num_waves
        self.inputs['x_solution'] = x_solution
        self.inputs['test_mode'] = test_mode
        self.inputs['num_postprocess'] = num_postprocess
        self.inputs['extra_output_columns'] = extra_output_columns
        self.bounds = bounds_x0

    def write_csv(self, out_path):
        """
        Incomplete or untested!
        
        Write a csv from self.df - as an alternative to using reanalyze. 
        This helps cases where you just create the object once in a loop.

        Parameters
        ----------
        out_path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
    
        if not os.path.exists(os.path.dirname(out_path)):
            if len(os.path.dirname(out_path)) != 0:
                os.mkdir(os.path.dirname(out_path))

        one_or_more_period = "".join(out_path.split(".")[0:-1])
        if len(one_or_more_period) == 0:
            basestr = out_path
        else:
            basestr = one_or_more_period

        self.df.to_csv(basestr + ".csv", na_rep="NaN")

    def reanalyze(self, inputs, write_csv=False):
        """


        Parameters
        ----------
        inputs : dict
            A dictionary whose keys are only allowed to be input parameters names
            for the initialization of self (i.e. "SolveDistributionShift")

        write_csv : bool, optional : Default = False
            Set to true to output the reanalysis (and all previous analyses.
            as a comma separated values (csv) file in the same directory location
            as the 'out_path' for figure output. The default is False.

        Raises
        ------
        TypeError
            For cases where the inputs are incorrect types
        ValueError
            For any invalid dictionary input to 'inputs'

        Returns
        -------
        dict
            Dictionary of parameters needed by Extreme object and that
            connects the MEWS analysis together.

        """

        if not isinstance(inputs, dict):
            raise TypeError("The input 'inputs' must be a dictionary!")
        if not isinstance(write_csv, bool):
            raise TypeError(
                "the input 'write_csv' must be a boolean (True/False)!")

        self._write_csv = write_csv

        for key, val in inputs.items():
            if key in self.inputs:
                self.inputs[key] = val
            else:
                # no extraneous values allowed.
                raise ValueError("SolveDistributionShift.reanalyze: An input option {0} was given but that is not a valid input. \n".format(key) +
                                 " valid inputs are: \n\n {0}".format(str(self._valid_inputs)))

        tup = tuple([self.inputs[inp] for inp in self._valid_inputs])

        (num_step,
         param0,
         random_seed,
         hist0,
         durations0,
         delT_above_shifted_extreme,
         historic_time_interval,
         hours_per_year,
         problem_bounds,
         ipcc_shift,
         decay_func_type,
         use_cython,
         num_cpu,
         plot_results,
         max_iter,
         plot_title,
         out_path,
         weights,
         limit_temperatures,
         min_num_waves,
         x_solution,
         test_mode,
         num_postprocess,
         extra_output_columns) = tup

        self.__init__(num_step,
                      param0,
                      random_seed,
                      hist0,
                      durations0,
                      delT_above_shifted_extreme,
                      historic_time_interval,
                      hours_per_year,
                      problem_bounds,
                      ipcc_shift,
                      decay_func_type,
                      use_cython,
                      num_cpu,
                      plot_results,
                      max_iter,
                      plot_title,
                      out_path,
                      weights,
                      limit_temperatures,
                      min_num_waves,
                      x_solution,
                      test_mode,
                      num_postprocess,
                      extra_output_columns)

        return self.param

    def _check_ipcc_shift(self, ipcc_shift, key, ukey):
        if not key in ipcc_shift:
            raise ValueError(
                "The input 'ipcc_shift['{0}']' must be a dictionary with key '{1}'".format(ukey, key))
        else:
            for yr_str in ["10 year", "50 year"]:
                if not yr_str in ipcc_shift[key]:
                    raise ValueError(
                        "The input subdictionary 'ipcc_shift[{0}] must have include a key '{1}'".format(key, yr_str))

                if not isinstance(ipcc_shift[key][yr_str], (int, float)):
                    raise TypeError(
                        "The input subdictionary 'ipcc_shift['{0}']['{1}'] must be a positive number".format(key, yr_str))

                if ipcc_shift[key][yr_str] < 0.0 or ipcc_shift[key][yr_str] > 100.0:
                    raise ValueError(
                        "The input subdictionary 'ipcc_shift['{0}']['{1}'] is constrained to the interval 0 - 100".format(key, yr_str))

    def _check_inputs(self,
                      num_step,
                      random_seed,
                      param0,
                      hist0,
                      durations0,
                      delT_above_shifted_extreme,
                      historic_time_interval,
                      hours_per_year,
                      problem_bounds,
                      ipcc_shift,
                      decay_func_type,
                      use_cython,
                      num_cpu,
                      plot_result,
                      max_iter,
                      plot_title,
                      out_path,
                      weights,
                      limit_temperatures,
                      min_num_waves,
                      x_solution,
                      test_mode,
                      num_postprocess,
                      extra_output_columns):
        """
        Input checking is very important because this class uses  
        parallel computing in differential_evolution where debugging bad inputs is much harder!
        """
        if not isinstance(x_solution, (type(None), np.ndarray)):
            raise TypeError(
                "The input 'x_solution' must be None or an np.array of correct size for the decay_func_types requested.")
        if not isinstance(test_mode, bool):
            raise TypeError(
                "The input 'test_mode' must be a boolean (True/False)!")
        if not isinstance(random_seed, (type(None),int)):
            raise TypeError(
                "The input 'random_seed' must be a 32 bit integer or None (i.e. any integer < 2**32-1 works)")
        elif not isinstance(param0, dict):
            raise TypeError(
                "The input 'param0' must be a dictionary with specific structure:\n\n" + self._param0_struct)
        elif not isinstance(hist0, dict):
            raise TypeError(
                "The input 'hist0' must be dict with key {0}.".format(str(self._events)))
        elif not isinstance(durations0, dict):
            raise TypeError(
                "The input 'durations0' must be dict with key {0}.".format(str(self._events)))
        elif not isinstance(ipcc_shift, dict):
            raise TypeError("The ipcc_shift input must be a dictionary with key {0}.".format(
                str(self._events)))
        elif not isinstance(decay_func_type, dict):
            raise TypeError("The input 'decay_func_type' must be a dictionary with keys = {0} and values from the list = {1}".format(
                str(self._events), str(self._decay_func_types)))
        elif not isinstance(use_cython, bool):
            raise TypeError(
                "The input 'use_cython' must be a boolean (i.e. True or False)")
        elif not isinstance(num_cpu, (int)):
            raise TypeError(
                "The input 'num_cpu' must be an integer > 0 or give it a value = -1 to use all cpus available.")
        elif not isinstance(plot_result, bool):
            raise TypeError(
                "The input 'plot_result' must be a boolean (i.e. True or False)")
        elif not isinstance(max_iter, int):
            raise TypeError("The input 'max_iter' must be an integer > 0")
        elif not isinstance(plot_title, str):
            raise TypeError("The input 'plot_title' must be a string!")
        elif not isinstance(out_path, str):
            raise TypeError(
                "The input 'out_path' must be a string that is a valid path to a file that is writeable.")
        elif not isinstance(delT_above_shifted_extreme, dict):
            raise TypeError("the input 'delT_above_shifted_extreme' must be a dictionary with key = {0}.".format(
                str(self._events)))
        elif not isinstance(historic_time_interval, int):
            raise TypeError("The input 'historic_time_interval' must be an integer equal to 24 * the number of days in the NOAA daily summaries for the month being analyzed (or all days if entire year is being lumped).")
        elif not isinstance(hours_per_year, int):
            raise TypeError(
                "The input 'hours_per_year' must be an integer < 8684 and ussually ~ 8760/12 equal to the number of hours in the current month being analyzed in the historic record.")
        elif not isinstance(problem_bounds, (type(None), dict)):
            raise TypeError(
                "The input 'problem_bounds' must be None or a dictionary!")
        elif not isinstance(weights, np.ndarray):
            raise TypeError("The input 'weights' must be a numpy array!")
        elif not isinstance(limit_temperatures, bool):
            raise TypeError(
                "The input 'limit_temperatures' must be a boolean!")
        elif not isinstance(min_num_waves, int):
            raise TypeError("The input 'min_num_waves' must be an integer!")
        elif not isinstance(num_postprocess, int):
            raise TypeError("The input 'num_postprocess' must be an integer!")
        elif not isinstance(extra_output_columns, dict):
            raise TypeError(
                "The input 'extra_output_columns' must be a dictionary!")
        # look at total hours in events by durations0 These must be bounded by a fraction 
        # of the total hours in the data.
        hours_in_event = ((bin_avg(durations0['cs'])*durations0['cs'][0]).sum() + 
                          (bin_avg(durations0['hw'])*durations0['hw'][0]).sum())
        historic_hours = historic_time_interval * hours_per_year / HOURS_IN_YEAR
        if historic_hours < hours_in_event:
            raise ValueError("The historic record has more than 50% in extreme"+
                             " states for total hours. This is not physically "+
                             "realistic for 90% CI being used. Please investigate "+
                             " the heat wave calculations")
        
        for key, val in extra_output_columns.items():
            if not isinstance(key, str):
                raise TypeError(
                    "All keys to 'extra_output_columns' must be strings")
            if not isinstance(val, (str, numbers.Number, bool)):
                raise TypeError(
                    "All values in 'extra_output_columns' must be string, numeric, or boolean!")

        if min_num_waves < 10:
            raise ValueError("The analysis being done is statistical. The minimum" +
                             " number of waves is not allowed to be below a 10! " +
                             "{0:d} has been input.".format(min_num_waves))

        if num_postprocess < 1 or num_postprocess > 100:
            raise ValueError(
                "The input 'num_postprocess' must be between 1 and 100!")

        if hours_per_year > HOURS_IN_LEAP_YEAR or hours_per_year < 0:
            raise ValueError("The input 'hours_per_year' must be between 0 to" +
                             " 8784 and is ususally the number of hours in the" +
                             " current month being analyzed in the historic record.")

        for wt in weights:
            if wt < 0.0:
                raise ValueError(
                    "The elements of input 'weights' must be positive")
        if len(weights) != 5:
            raise ValueError(
                "The input 'weights' must have 5 elements.")

        if isinstance(problem_bounds, dict):
            dict_key_equal(self._default_problem_bounds, problem_bounds)

        for wt, ipcc_shift_ in ipcc_shift.items():
            if not isinstance(ipcc_shift_, (dict, type(None))):
                raise TypeError(
                    "The input 'ipcc_shift' must be a dictionary with specific structure:\n\n"+self._ipcc_shift_struct)

        for wt, hist0_ in hist0.items():
            if not wt in self._events:
                raise ValueError(
                    "The only valid keys for 'hist0' are {0}.".format(str(self._events)))
            if not isinstance(hist0_, tuple):
                raise TypeError(
                    "The input hist0['{0}'] must be a tuple output from np.histogram!".format(wt))
        for wt, durations0_ in durations0.items():
            if not wt in self._events:
                raise ValueError(
                    "The only valid keys for 'durations0' are {0}.".format(str(self._events)))
            if not isinstance(durations0_, tuple):
                raise TypeError(
                    "The input durations0['{0}'] must be a tuple output from np.histogram!".format(wt))
            if durations0_[0].sum() != hist0[wt][0].sum():
                raise ValueError(
                    "The inputs durations0 and hist0 must sum to the same number of heat waves!")

        for wt, delT_ in delT_above_shifted_extreme.items():
            if not wt in self._events:
                raise ValueError(
                    "The only valid keys for 'delT_above_shifted_extreme' are {0}.".format(str(self._events)))
            if not isinstance(delT_, (int, float)):
                raise TypeError(
                    "the input 'delT_above_shifted_extreme['{0}']' must be a number > 0".format(wt))
            # 20.0 C above the 50 year event is super unlikely physically
            if wt == 'hw' and (delT_ < 0 or delT_ > 20.0):
                raise ValueError(
                    "The input 'delT_above_shifted_extreme['{0}'] is in degrees Celcius and must be between 0 and 20.".format(wt))
            elif wt == 'cs' and (delT_ > 0 or delT_ < -20.0):
                raise ValueError(
                    "The input 'delT_above_shifted_extreme['{0}'] is in degrees Celcius and must be between -20 and 0.".format(wt))

        for wt, decay_func in decay_func_type.items():
            if not wt in self._events:
                raise ValueError(
                    "The only valid keys for 'decay_func_type' are {0}.".format(str(self._events)))
            if not decay_func in self._decay_func_types_entries:
                raise ValueError(
                    "The input 'decay_func_type' must be one of the following " + str(self._decay_func_types_entries))

        # test IPCC shift factors.
        for wt, ipcc_shift_subdict in ipcc_shift.items():
            if not ipcc_shift_subdict is None:
                for key in ['temperature', 'frequency']:
                    self._check_ipcc_shift(ipcc_shift_subdict, key, wt)

        # ASSURE CORRECT INPUT VECTOR LENGTH
        if problem_bounds is None:
            problem_bounds = self._default_problem_bounds
        for wt, pb in problem_bounds.items():
            if not wt in self._events:
                raise ValueError(
                    "The problem bounds must have entries for both heat waves and cold snaps ({0})".format(str(self._events)))
            else:
                decay_func = decay_func_type[wt]
                entries_needed = self._decay_func_types_entries[decay_func]
                numvar = len(entries_needed)
                for entry in entries_needed:
                    if not entry in pb:
                        if decay_func is None:
                            func_descr_str = "with no decay function "
                        else:
                            func_descr_str = "with a " + decay_func + " function "
                        raise ValueError("The problem formulation "+func_descr_str+" must have {0:d} variables.".format(numvar) +
                                         "\nThe formaulation for each variable is:\n\n" + self._variable_order_string)
        if not self._test_mode:
            if num_step < 8760*50:
                raise ValueError(
                    "The input 'num_step' needs to be a large value > 8760*50 since a Markov process statistics must be characterized for 50 year events.")

        if historic_time_interval < 0:
            raise ValueError(
                "The historic time interval must be greater than zero!")
        elif historic_time_interval < 24 * 20 * 365:
            warn(
                "The historic time interval is less than 20 years. This is not advised!", UserWarning)

        # TODO continue to make input checking stronger

    def _problem_bounds(self, decay_func_type, num_step, param0, abs_max_temp, limit_temperatures):
        # TODO - bring this to the surface of MEWS' input structure.
        """
        This entire function is very dependent on the problem formulation and
        needs extensive updating if the variable space for MEWS' optimization problem
        is being shrunk or expanded'

        _default_problem_bounds = {'cs':{'delT_mu': (0.0, 0.7),
                                         'delT_sig multipliers': (-0.1,2),
                                         'P_event': (0.00001, 0.03),
                                         'P_sustain': (0.958, 0.999999),
                                         'multipliers to max probability time': (0,2),
                                         'slope or exponent multipliers' : (0,1),
                                         'cutoff time multipliers':(1,3),
                                         'max peak prob for quadratic model': (0.97, 1.0)},
                                   'hw':{'delT_mu': (0.0, 0.7),
                                         'delT_sig multipliers': (-0.1,2),
                                         'P_event': (0.00001,0.03),
                                         'P_sustain': (0.958,0.999999),
                                         'multipliers to max probability time': (0.1,2),
                                         'slope or exponent multipliers' : (0,1),
                                         'cutoff time multipliers':(1,3),
                                         'max peak prob for quadratic model': (0.97, 1.0)}}       """
        # this must be in the same order as variables in "markov_guassian_model_for_peak_temperature"
        dpb = self._prob_bounds
        bounds_x0 = [dpb['cs']['delT_mu'],
                     (dpb['cs']['delT_sig multipliers'][0]*param0['cs']['extreme_temp_normal_param']['sig'],   # narrowing of standard deviation is not expected.
                      dpb['cs']['delT_sig multipliers'][1]*param0['cs']['extreme_temp_normal_param']['sig']),
                     dpb['hw']['delT_mu'],
                     (dpb['hw']['delT_sig multipliers'][0]*param0['hw']['extreme_temp_normal_param']['sig'],
                      dpb['hw']['delT_sig multipliers'][1]*param0['hw']['extreme_temp_normal_param']['sig']),
                     dpb['cs']['P_event'],
                     dpb['hw']['P_event'],
                     # 0.958 gives an average 24 hour heat wave which is the minimum.
                     dpb['cs']['P_sustain'],
                     dpb['hw']['P_sustain']]
        # establish a constraint such that Phw and Pcs cannot add to more than one.
        constrain_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])

        # this is awkward but we need the bounds to be added in 'cs', 'hw' order
        # eventually the order of the variable vector x may need to be reordered

        nlc = {}

        # must do by list to keep 'cs' first!
        for wt in self._events:
            decay_func = decay_func_type[wt]
            if not decay_func is None:
                max_duration = param0[wt]['normalizing duration']

                if decay_func == DECAY_FUNC_NAMES[1] or decay_func == DECAY_FUNC_NAMES[2]:
                    bounds_x0.append(
                        tuple(np.array(dpb[wt]['slope or exponent multipliers'])/max_duration))
                    num_added = 1

                elif decay_func == DECAY_FUNC_NAMES[3] or decay_func == DECAY_FUNC_NAMES[4]:
                    bounds_x0.append(
                        tuple(np.array(dpb[wt]['slope or exponent multipliers'])/max_duration))
                    # cutoff times can be much larger
                    bounds_x0.append(
                        tuple(max_duration*np.array(dpb[wt]['cutoff time multipliers'])))
                    num_added = 2
                elif decay_func == DECAY_FUNC_NAMES[5]:
                    bounds_x0.append(
                        tuple(max_duration * np.array(dpb[wt]['multipliers to max probability time'])))
                    bounds_x0.append(
                        tuple(dpb[wt]['max peak prob for quadratic model']))
                    # cutoff times can be much larger
                    bounds_x0.append(
                        tuple(max_duration * np.array(dpb[wt]['cutoff time multipliers'])))
                    num_added = 3
                elif decay_func == DECAY_FUNC_NAMES[6]:
                    bounds_x0.append(
                        tuple(np.array(dpb[wt]['slope or exponent multipliers'])/max_duration))
                    # cutoff times can be much larger
                    bounds_x0.append(
                        tuple(max_duration * np.array(dpb[wt]['cutoff time multipliers'])))
                    bounds_x0.append(tuple(dpb[wt]['delay_time_multipliers']))
                    num_added = 3
                else:
                    raise ValueError("Invalid decay function type '{0}': only valid values are: \n\n{1}".format(
                        decay_func, str(self._decay_func_types_entries.keys())))

                constrain_matrix = np.concatenate([constrain_matrix, np.zeros((1, num_added))],
                                                  axis=1)

                # these nonlinear constraints on temperature are dependent on having a hard cutoff temperature
                # this is highly dependent on the number of variables added. and the cutoff times are always the second for a given event type.
                if "cutoff" in decay_func:
                    if limit_temperatures:
                        if wt == "cs":
                            lower_bound = (abs_max_temp[wt], abs_max_temp[wt])
                            upper_bound = (0.0, 0.0)
                        else:
                            lower_bound = (0.0, 0.0)
                            upper_bound = (abs_max_temp[wt], abs_max_temp[wt])

                        nlc_obj = MaxTemperatureNonlinearConstraint(
                            param0, decay_func_type, wt, self._events)
                        nlc[wt] = NonlinearConstraint(
                            nlc_obj.func, lower_bound, upper_bound)
                    else:
                        nlc[wt] = None
                else:
                    nlc[wt] = None

        # do not allow zero probability of heat or cold snap solutions or all
        # heat wave and cold snap solution.
        lc = LinearConstraint(constrain_matrix, np.array(
            [1/(num_step/25)]), np.array([1.0-1/(num_step/25)]))

        return bounds_x0, lc, nlc

    def _grab_relevant_bounds(self, decay_func_type, problem_bounds):
        allb = self._default_problem_bounds_all

        new_dict = {}

        for wv, subdict in allb.items():
            # somne input checking need because this is ahead of input checks
            if wv in decay_func_type:
                decay_name = decay_func_type[wv]
            else:
                raise ValueError(
                    "The decay_func_type input must be a dictionary over all wave names: " + str(ABREV_WAVE_NAMES))

            if not problem_bounds is None:
                if wv in problem_bounds:
                    pbs = problem_bounds[wv]
                else:
                    raise ValueError(
                        "The problem_bounds input must be a dictionary over all wave names:" + str(ABREV_WAVE_NAMES))
            else:
                pbs = {}

            new_dict[wv] = {}
            for entry in self._decay_func_types_entries[decay_name]:
                if not entry in pbs:
                    new_dict[wv][entry] = allb[wv][entry]
                else:
                    new_dict[wv][entry] = pbs[entry]
            # now check that no superfluous values exist
            for key, val in pbs.items():
                if not key in self._decay_func_types_entries[decay_name]:
                    raise ValueError("\n\nThe problem bounds for {0} type decay functions does not include '{2}' only contains the following entries: {1}".format(
                        decay_name, str(self._decay_func_types_entries[decay_name]), key))
        return new_dict
    
    
    def _is_solution_on_bounds(self,solution,bounds):
        
        idx = 0
        for sol,bound in zip(solution,bounds):
            bound_range = bound[1] - bound[0]
            if (sol - bound[0])/bound_range < 0.001:
                warn("Variable x[{0:d}] is on its lower bound. Consider".format(idx)+
                     " expanding the boundaries through the solve_options"+
                     "['problem_bounds'] dictionary input.")
            elif (bound[1]-sol)/bound_range < 0.001:
                warn("Variable x[{0:d}] is on its upper bound. Consider".format(idx)+
                     " expanding the boundaries through the solve_options"+
                     "['problem_bounds'] dictionary input.")
        pass
    
    def _output_table(self, hist0, histT_tuple, durations0, durations, ipcc_shift, thresholds, events, out_path, extra_output_columns):

        def add_histogram_rows(hist, table, event, extra_output_columns, unit, valtype, postprocess_num, threshold, normalize=False, write_csv=False):
            if normalize:
                norm = hist[0].sum()
            else:
                norm = 1.0
            for idx, val in enumerate(hist[0]):
                crow = [val/norm, valtype, unit, hist[1][idx], hist[1]
                        [idx+1], event, postprocess_num, threshold]
                for key, val in extra_output_columns.items():
                    crow.append(val)

                table.append(crow)

        def add_threshold_rows(table, thre, event, postproc_num, extra_output_columns):
            for key, thr in thre.items():
                for idx, val in enumerate(thr):
                    if idx == 0:
                        tstr = "10 year "
                    else:
                        tstr = "50 year "
                    lis = [np.nan, tstr + key, "degC", np.nan,
                           np.nan, event, postproc_num, val]
                    for key2, val2 in extra_output_columns.items():
                        lis.append(val2)
                    table.append(lis)

        def add_shift_rows(table, shift, event, extra_output_columns):
            for key, shi in shift.items():
                for key2, val in shi.items():
                    if key == "temperature":
                        unit = "degC"
                    else:
                        unit = "frequency multiplier"
                    lis = [np.nan, key2 + " " + key + " shift",
                           unit, np.nan, np.nan, event, np.nan, val]
                    for key3, val2 in extra_output_columns.items():
                        lis.append(val2)
                    table.append(lis)

        table = []
        columns = ['histogram value', 'type', 'unit', 'bin start',
                   'bin end', 'event', 'postprocess num', 'threshold']
        for key in extra_output_columns.keys():
            columns.append(key)

        for event in events:
            h0e = hist0[event]
            d0e = durations0[event]

            add_histogram_rows(h0e, table, event, extra_output_columns,
                               "degC", "data temperature", np.nan, np.nan, True)
            add_histogram_rows(
                d0e, table, event, extra_output_columns, "hr", "data duration", np.nan, np.nan)

            # remember histT_tuple has two entries one with averaged bin locations
            for postproc_num, hTpost in histT_tuple.items():
                hTe = hTpost[event]
                dTe = durations[postproc_num][event]
                # now do the fit or future results histograms
                if not hTe is None:
                    add_histogram_rows(hTe[0], table, event, extra_output_columns,
                                       "$^{\circ}$C", "fit temperature", postproc_num, np.nan, True)
                if not dTe is None:
                    add_histogram_rows(dTe, table, event, extra_output_columns,
                                       "hr", "fit duration", postproc_num, np.nan)

                if not thresholds[postproc_num][event] is None:
                    add_threshold_rows(
                        table, thresholds[postproc_num][event], event, postproc_num, extra_output_columns)

            if not ipcc_shift[event] is None:
                add_shift_rows(
                    table, ipcc_shift[event], event, extra_output_columns)

        df = pd.DataFrame(table, columns=columns)

        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df])

        if self._write_csv and len(out_path) > 0:
            self.write_csv(out_path)


class MaxTemperatureNonlinearConstraint():

    def __init__(self, param, decay_func, wtype, events):
        self.param = param

        wtoff = x_offset_for_func_type_combo(decay_func, events)

        # this offset allows navigation of the
        self._wtoff = wtoff
        self._decay_func = decay_func
        self._wtype = wtype

    def func(self, x):
        temp_list = []

        if self._wtype == "cs":
            eid = [0, 1]
        else:
            eid = [2, 3]

        pval = unpack_params(self.param, self._wtype)
        del_mu_T = x[eid[0]]
        del_sig_T = x[eid[1]]
        del_a, del_b = shift_a_b_of_trunc_gaussian(del_mu_T,
                                                   pval['mu_T'],
                                                   del_sig_T,
                                                   pval['sig_T'])
        min_boundary = transform_fit(pval['minval_T'],pval['hist_minval_T'],pval['hist_maxval_T'])
        max_boundary = transform_fit(pval['maxval_T'],pval['hist_minval_T'],pval['hist_maxval_T'])
        
        Xb = inverse_transform_fit(min_boundary+del_b, pval['maxval_T'], pval['minval_T'])
        Xa = inverse_transform_fit(max_boundary+del_a,
                                   pval['maxval_T'], pval['minval_T'])

        if self._wtype == 'cs':
            cutoff = x[9]
        else:
            cutoff = x[8+self._wtoff[0]+1]

        temp_list.append(temperature(Xb, cutoff, pval))
        temp_list.append(temperature(Xa, cutoff, pval))

        return tuple(temp_list)
