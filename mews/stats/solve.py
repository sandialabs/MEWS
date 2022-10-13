#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:28:02 2022



@author: dlvilla
"""

from mews.stats.extreme import DiscreteMarkov
from mews.utilities.utilities import linear_interp_discreet_func, bin_avg
from numpy.random import default_rng
from mews.stats.distributions import trunc_norm_dist, inverse_transform_fit
from mews.utilities.utilities import find_extreme_intervals, create_complementary_histogram
import matplotlib.pyplot as plt
from mews.graphics.plotting2D import Graphics
from scipy.optimize import minimize, LinearConstraint, differential_evolution, NonlinearConstraint
from copy import deepcopy

import numpy as np

def shift_a_b_of_trunc_gaussian(del_mu_T,mu_T,del_sig_T,sig_T):
    return (del_mu_T - (1 + mu_T)/(sig_T) * del_sig_T, 
           del_mu_T + (1 - mu_T)/(sig_T) * del_sig_T)

def unpack_params(param,wave_type):
    return {'norm_duration':    param[wave_type]['normalizing duration'],
            'mu_T':             param[wave_type]['extreme_temp_normal_param']['mu'],
            'sig_T':            param[wave_type]['extreme_temp_normal_param']['sig'],
            'maxval_T':         param[wave_type]['max extreme temp per duration'],
            'minval_T':         param[wave_type]['min extreme temp per duration'],
            'norm_temperature': param[wave_type]['normalizing extreme temp'],
            'alpha_T':          param[wave_type]['normalized extreme temp duration fit slope'],
            'beta_T':           param[wave_type]['normalized extreme temp duration fit intercept']}
    

def temperature(T_per_durations,durations,pval):
    return T_per_durations * pval['norm_temperature'] * (
        pval['alpha_T'] * (durations/pval['norm_duration']) + pval['beta_T'])

def evaluate_temperature(durations,param,del_mu_T,del_sig_T,rng, wave_type):

    # taken from extreme._add_extreme
    pval = unpack_params(param,wave_type)

    Sm1_T, S_T = shift_a_b_of_trunc_gaussian(del_mu_T,pval['mu_T'],del_sig_T,pval['sig_T']) 
    
    T_per_durations = np.array([trunc_norm_dist(rand,
                                           pval['mu_T'] + del_mu_T,
                                           pval['sig_T'] + del_sig_T,
                                           -1+Sm1_T,
                                           1+S_T,
                                           pval['minval_T'],
                                           pval['maxval_T']) for rand in rng.random(len(durations))])
    # this returns actual temperatures.
    return temperature(T_per_durations,durations,pval)


def ipcc_shift_comparison_residuals(hist0,ipcc_shift, 
                                    histT_tuple,
                                    durations,
                                    num_step,
                                    output_hist,
                                    delT_above_50_year):
    # get and calculate cdf's for historical (0) and future (_T) distributions
    
    cdf0 = ((hist0[0]/hist0[0].sum()).cumsum(), (hist0[1][1:] + hist0[1][0:-1])/2)
    cdf_T = histT_tuple[1]
    
    # produce a large negative penalty for distributions that produce negative 
    # temperature heat waves.
    negative_penalty = (np.max([float((cdf_T[1] < 0).sum()) - 1.0,0.0]))**2
    
    # Calculate probabilities
    P10_given_hw, P50_given_hw = probability_of_extreme_10_50(durations,num_step)
    
    # probabilities a heat wave being less than the 10 and 50 year events
    P0_less_than_extreme = np.array([1.0-P10_given_hw,1.0-P50_given_hw])
    
    # target probablities 
    P_T_target_less_than_extreme = np.array([1.0 - P10_given_hw * ipcc_shift['frequency']['10 year'],
                                             1.0 - P50_given_hw * ipcc_shift['frequency']['50 year']])
    
    T_shifted_actual = linear_interp_discreet_func(P_T_target_less_than_extreme, cdf_T, False)
    
    
    Tthresh_historic = linear_interp_discreet_func(
        P0_less_than_extreme,
        cdf0)
    
    # These are the exact values needed
    T_shifted_target = Tthresh_historic + np.array([ipcc_shift['temperature']['10 year'], ipcc_shift['temperature']['50 year']])
    
    # penalize temperatures 5 C hotter than 50 year peak temperature
    # The square root amplifies the low probability numbers to create increased residuals
    positive_penalty = ((histT_tuple[0][0][cdf_T[1] > (T_shifted_target[1] + delT_above_50_year)]).sum())**0.5
    
    
    residuals = ((T_shifted_actual - T_shifted_target)/T_shifted_target)**2
    
    if output_hist:
        return residuals + negative_penalty + positive_penalty, {'actual':T_shifted_actual, 'target':T_shifted_target}
    return residuals + negative_penalty + positive_penalty

def probability_of_extreme_10_50(durations,num_step):
    
    if len(durations) == 0:
        return 0.0, 0.0
    else:
        hour_in_year = 8760
        N10 = 10 * hour_in_year
        N50 = 50 * hour_in_year
        num_10_year_intervals = num_step / N10
        num_50_year_intervals = num_step / N50
        P10_given_hw = num_10_year_intervals / len(durations)
        P50_given_hw = num_50_year_intervals / len(durations)
        
        return P10_given_hw, P50_given_hw



def histogram_comparison_residuals(histT,hist0):
    """
    This function calculates the difference between two probability density functions
    of discrete historgrams. The histograms must have equal spacing for their
    bin edges.
    
    histT
    hist0 are both np.historgram tuple output.
    
    """
    
    avg_bin0 = bin_avg(hist0)
    avg_binT = bin_avg(histT)
    all_bin = np.unique(np.concatenate([avg_bin0,avg_binT]))
    
    normT = histT[0]/histT[0].sum()
    norm0 = hist0[0]/hist0[0].sum()
    
    # a rather tedious list comprehension. Saves computation time though.
    residuals = np.array(
    [(normT[np.where(avg_binT==abi)[0][0]] - norm0[np.where(avg_bin0==abi)[0][0]])**2 
        if ((abi in avg_bin0) and (abi in avg_binT)) 
        else
    normT[np.where(avg_binT==abi)[0][0]]**2 
        if (abi in avg_binT) 
        else
    norm0[np.where(avg_bin0==abi)[0][0]]**2 
        for abi in all_bin])
    
    return residuals



def markov_gaussian_model_for_peak_temperature(x,
                                               num_step,
                                               random_seed,
                                               hist_param,
                                               wave_type,
                                               hist0,
                                               ipcc_shift=None,
                                               decay_func_type=None,
                                               use_cython=True,
                                               output_hist=False,
                                               delT_above_50_year=5):
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
        x[0] = Change in 'wave_type' mean of truncated guassian for scaled, duration 
               normalized temperature. For historic case this should always be
               zero.
               
        x[1] = Change in 'wave_type' standard deivation of truncated... (see above)
        
        x[2] = Probability of a cold snap occuring when markov state = 
               not in cold snap or heat wave (i.e. 0)
               
        x[3] = Probability of a heat wave occuring when markov state = not in
               a cold snap or heat wave (i.e. 0)
        
        x[4] = Probability of sustaining a cold snap at time in wave = 0
        
        x[5] = Probability of sustaining a heat wave at time in wave = 0
        If not decay_func_type is None:
            x[6] = slope or exponential parameter (depending on 
                    'decay_func_type') decay of probability sustaining cold snaps
            
            x[7] = same as x[6] for heat waves
            
        If "cutoff" in decay_func_type:
            x[8] = cutoff time at which probability of sustaining a cold snap
                   drops to zero
            x[9] = same as x[8] for heat waves.
               
      .... Finish docs later.
    
    """
    # We want the same set of random numbers every time so that the
    # Markov process is not causing as much 
    print_results = False
    
    rng = default_rng(random_seed)
    if print_results:
        print("trying " + str(x))
    # this function is used in optimization routines so it needs to be efficient!
    
    # first two parameters are always duration normalized and scaled temperature
    # mean and standard deviation.
    del_mu_T = x[0]
    del_sig_T = x[1]
    
    Pcs = x[2]
    Phw = x[3]
    Pcss = x[4]
    Phws = x[5]
    
    # next parameters are the probability of sustainting a cold snap and then a 
    # heat wave
    tran_matrix  = np.array([[1-Pcs-Phw, Pcs,  Phw ],
                             [1-Pcss,    Pcss, 0.0 ],
                             [1-Phws,    0.0,  Phws]])

    
    # next parameters depend on the func_type. They do not exist if
    # there is no decay function
    if not decay_func_type is None:
        if decay_func_type == "exponential":
            lamb_cs = x[6]
            lamb_hw = x[7]
            coef = np.array([[lamb_cs],[lamb_hw]])
        elif decay_func_type == "linear":
            slope_cs = x[6]
            slope_hw = x[7]
            coef = np.array([[slope_cs],[slope_hw]])
        else:
            coef = np.array([[x[6],x[8]],[x[7],x[9]]])
    else:
        coef = None
        
    # now run the Markov process
    objDM = DiscreteMarkov(rng, 
                           tran_matrix, 
                           decay_func_type=None,#decay_func_type,
                           coef=coef,
                           use_cython=use_cython)
    
    states_arr = objDM.history(num_step, 0)
    
    state_intervals = find_extreme_intervals(states_arr, [1,2])
    
    # only look at heat wave effects. Cold snaps do effect things but only
    # in the sense that they displace heat waves.
    durations = np.array([tup[1]-tup[0]+1 for tup in state_intervals[2]])
    
    # probabilities that a heat wave is a ten year event (hottest in 10 years)
    # or a 50 year event
    Tsample = evaluate_temperature(durations, hist_param, 
                                   del_mu_T, del_sig_T, rng, wave_type)
    # this returns a tuple the first element is the histogram of temperatures
    # the second element returns the cdf with values mapped to the bin averages.
    if len(Tsample) == 0:
        # for cases where Markov process produces no heat waves. return 
        # 99999 residual recognizable at solution.
        residuals = np.array([44444,55555])
    else:
        # normal evaluation cases.
        histT_tuple = create_complementary_histogram(Tsample,hist0)    
        
        if ipcc_shift is None:
            residuals = histogram_comparison_residuals(histT_tuple[0],hist0)
        else:
            if output_hist:
                residuals, thresholds = ipcc_shift_comparison_residuals(hist0,ipcc_shift, 
                                                        histT_tuple, durations, 
                                                        num_step,output_hist,delT_above_50_year)
            else:
                residuals = ipcc_shift_comparison_residuals(hist0,ipcc_shift, 
                                                        histT_tuple, durations, 
                                                        num_step,output_hist)
    if print_results:
        print("residuals = {0:10.5f}".format(residuals.sum()))
        
    if output_hist:
        return residuals.sum(), Tsample, durations, thresholds, histT_tuple
    else:
        return residuals.sum()
    


        
        
    



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
                                        x0[6:7] - lambda or slope as defined above
                                        x0[8:9] - cutoff times at which probability is set to zero
    
                             """
    
    _ipcc_shift_struct = """
                            ipcc_shift['temperature'] = {'10 year':'=IPCC predicted change in temperature for future 10 year heat wave events',
                                                         '50 year':'=IPCC predicted change in temperature for future 50 year heat wave events'
                                                         }
                            ipcc_shift['frequency'] = {'10 year':'=IPCC predicted increase of frequency of future 10 year heat wave events',
                                                       '50 year':'=IPCC predicted increase of frequency of future 50 year heat wave events'
                                                       }
    
                        """
    _ipcc_shift_dict_str = {'temperature':['10 year','50 year'],'frequency':['10 year','50 year']}
    
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
    
    def __init__(self, x0, num_step,
                            param0,
                            random_seed,
                            wave_type,
                            hist0,
                            fit_hist,
                            delT_above_50_year,
                            ipcc_shift=None,
                            decay_func_type=None,
                            use_cython=True,
                            num_cpu=-1,
                            plot_results=True,
                            max_iter=20,
                            plot_title="",
                            fig_path=""):
        """
        
        Parameters
        ----------

        x0 : np.array
            initial guess of input vector (see docs for markov_gaussian_model_for_peak_temperature)
            for what each element of this vector represents.
                               num_step, 
                               rng, 
                               param0,
                               wave_type,
                               hist0, 
                               ipcc_shift,
                               decay_func_type=decay_func_type,
                               use_cython=True

        
        
        """
        self._check_inputs(x0, num_step,
                                random_seed,
                                param0,
                                wave_type,
                                hist0,
                                fit_hist,
                                delT_above_50_year,
                                ipcc_shift,
                                decay_func_type,
                                use_cython,
                                num_cpu,
                                plot_results,
                                max_iter,
                                plot_title,
                                fig_path)
        
        # establish the absolute maximum temperature nonlinear constraint boundary on the optimization
        if fit_hist:
            abs_max_temp = delT_above_50_year # this parameter 
        else:
            abs_max_temp = ipcc_shift['temperature']['50 year'] + delT_above_50_year
        
        # linear constraint - do not allow Pcs and Phw to sum to more than specified amounts
        # nonlinear constraint - do not allow maximum possible sampleable temperature to exceed a bound.
        bounds_x0, linear_constraint, nonlinear_constraint = self._problem_bounds(decay_func_type,
                                                                                  num_step,
                                                                                  param0,
                                                                                  wave_type,
                                                                                  abs_max_temp)
        
        # the nonlinear constrain is only enforceable when cutoff parameters are present
        if nonlinear_constraint is None:
            constraints = [linear_constraint]
        else:
            constraints = [linear_constraint, nonlinear_constraint]
            
        # this optimization is not expected to converge. The objective function is stochastic. The
        # optimization still finds a solution but the population does not tend to stabilize because
        # of the stochastic objective function.
        optimize_result = differential_evolution(markov_gaussian_model_for_peak_temperature,
                                         bounds_x0,
                                         args=(num_step,
                                               random_seed,
                                               param0,
                                               wave_type,
                                               hist0,
                                               ipcc_shift,
                                               decay_func_type,
                                               use_cython,
                                               fit_hist,
                                               delT_above_50_year),
                                         constraints=constraints,
                                         workers=num_cpu,
                                         maxiter=max_iter,
                                         disp=False,
                                         polish=False) # polishing is a waste of time on a stochastic function.

        xf0 = optimize_result.x 
        
        # TODO - establish level of convergence.
        
        Tsample = {}
        durations = {}
        resid = {}
        thresholds = {}
        histT_tuple = {}
        for rand_adj in [1253, 9098, 3562]:
            resid[rand_adj], Tsample[rand_adj], durations[rand_adj], thresholds[rand_adj], histT_tuple[rand_adj] = markov_gaussian_model_for_peak_temperature(xf0,num_step,
                  random_seed-rand_adj,
                  param0,
                  wave_type,
                  hist0,
                  ipcc_shift,
                  decay_func_type,
                  use_cython,
                  output_hist=True)
        
        res0 = optimize_result.fun
        plot_result = True
        if plot_results:
            Graphics.plot_sample_dist_shift(hist0,histT_tuple, ipcc_shift, thresholds, plot_title, fig_path)
            

        self.optimize_result = optimize_result
        param_new = deepcopy(param0)
        pval = unpack_params(param0, wave_type)
        param_new[wave_type]['extreme_temp_normal_param']['mu'] = pval["mu_T"] + xf0[0]
        param_new[wave_type]['extreme_temp_normal_param']['sig'] = pval["sig_T"] + xf0[1]
         
        param_new['cs']['hourly prob of heat wave'] = xf0[2]
        param_new['hw']['hourly prob of heat wave'] = xf0[3]
        param_new['cs']['hourly prob stay in heat wave'] = xf0[4]
        param_new['hw']['hourly prob stay in heat wave'] = xf0[5]
        param_new['cs']['decay function'] = decay_func_type
        param_new['hw']['decay function'] = decay_func_type
        if decay_func_type == "linear" or decay_func_type == "exponential":    
            # hw and cold snaps 
            param_new['cs']['decay func coef'] = xf0[6]
            param_new['cs']['decay func coef'] = xf0[7]
        elif "cutoff" in decay_func_type:
            param_new['cs']['decay func coef'] = xf0[6],xf0[8]
            param_new['cs']['decay func coef'] = xf0[7],xf0[9]
        self.param0 = param0
        self.param = param_new
        
    def _check_inputs(self,x0, 
                           num_step,
                           random_seed,
                           param0,
                           wave_type,
                           hist0,
                           fit_hist,
                           delT_above_50_year,
                           ipcc_shift,
                           decay_func_type,
                           use_cython,
                           num_cpu,
                           plot_result,
                           max_iter,
                           plot_title,
                           fig_path):    
    
        if not isinstance(x0,np.ndarray):
            raise TypeError("The input 'x0' must be a numpy array.")
        elif not isinstance(random_seed,int):
            raise TypeError("The input 'random_seed' must be a 32 bit integer (i.e. any integer < 2**32-1 works)")
        elif not isinstance(param0,dict):
            raise TypeError("The input 'param0' must be a dictionary with specific structure:\n\n"+ self._param0_struct)
        elif not isinstance(hist0, tuple):
            raise TypeError("The histogram input 'hist0' must be tuple of lenght 2 from numpy.histogram")
        elif not isinstance(ipcc_shift, dict):
            raise TypeError("The input 'ipcc_shift' must be a dictionary with specific structure:\n\n"+self._ipcc_shift_struct)
    
        # ASSURE CORRECT INPUT VECTOR LENGTH    
        for decay_type,numvar in zip([None,'exponential','linear','exponential_cutoff','linear_cutoff'],[6,8,8,10,10]):
            if decay_func_type == decay_type and len(x0) != numvar:
                if decay_type is None:
                    func_descr_str = "with no decay function "
                else:
                    func_descr_str = "with a" + decay_type + "function "
                raise ValueError("The problem formulation "+func_descr_str+" must have {0:d} variables.".format(numvar) + 
                                 "\nThe formaulation for each variable is:\n\n" + self._variable_order_string)
                
        if num_step < 8760*50:
            raise ValueError("The input 'num_step' needs to be a large value > 8760*50 since a Markov process statistics must be characterized for 50 year events.")
        
        
        # TODO continue to make input checking stronger
                
        
    def _problem_bounds(self,decay_func_type,num_step,param0,wave_type, abs_max_temp):
        bounds_x0 = [(0.0,0.7),
                     (-0.1*param0[wave_type]['extreme_temp_normal_param']['sig'],
                      2*param0[wave_type]['extreme_temp_normal_param']['sig']),
                     (0,0.025),
                     (0,0.025),
                     (0.9,0.999999),
                     (0.9,0.999999)]
        # establish a constraint such that Phw and Pcs cannot add to more than one.                               
        constrain_matrix = np.array([[0.0,0.0,1.0,1.0,0.0,0.0]])
        
        max_duration = param0[wave_type]['normalizing duration']
        
        if "exponential" in decay_func_type or "linear" in decay_func_type:
            # only consider slopes that will cut the probability to 0 by the 
            # maximum duration of historic heat waves. Larger slopes lead to
            # too much shift on the mean and averages of the temperature distribution.
            bounds_x0.append((0,1/max_duration))
            bounds_x0.append((0,1/max_duration))  # slope and exponents should be very small
                                     # numbers a value of 1 makes decay fully occur
                                     # in 1 step.
            constrain_matrix = np.concatenate([constrain_matrix, np.zeros((1,2))],
                                              axis=1)
        if "cutoff" in decay_func_type:
            # limiting heat wave lengths to no more than 10 times the historic duration heat wave found.
            bounds_x0.append((param0[wave_type]['normalizing duration'],param0[wave_type]['normalizing duration']*10)) # cutoff times can be much larger
            bounds_x0.append((param0[wave_type]['normalizing duration'],param0[wave_type]['normalizing duration']*10)) # more than a year makes no sense in this problem
                                       # content.
            constrain_matrix = np.concatenate([constrain_matrix, np.zeros((1,2))],
                                              axis=1)
            
            nlc_obj = MaxTemperatureNonlinearConstraint(param, wave_type)
            nlc = NonlinearConstraint(nlc_obj.func, (0.0,0.0), (abs_max_temp,abs_max_temp))
            # we need a nonlinear constraint that sets a boundary on the absolute maximum temperature a heat wave can 
            # exhibit. We do this by creating a nonlinear constrain between the cutoff time, and the 
        else:
            nlc = None
            
        # do not allow zero probability of heat or cold snap solutions or all 
        # heat wave and cold snap solution.
        lc = LinearConstraint(constrain_matrix, np.array([1/(num_step/25)]), np.array([1.0-1/(num_step/25)]))
        
        
        return bounds_x0, lc, nlc
    
class MaxTemperatureNonlinearConstraint():
    
    def __init__(self,param, wave_type):
        self.param = param
        self.wave_type = wave_type
        
        
    def func(self,x):
        wave_type = self.wave_type
        pval = unpack_params(self.param, wave_type)
        del_mu_T = x[0]
        del_sig_T = x[1]
        del_a, del_b = shift_a_b_of_trunc_guassian(del_mu_T,
                                                   pval['mu_T'],
                                                   del_sig_T,
                                                   pval['sig_T'])
        Xb = inverse_transform_fit(1+del_b, pval['maxval_T'], pval['minval_T'])
        Xa = inverse_transform_fit(-1+del_a, pval['maxval_T'], pval['minval_T'])
        
        if wave_type == "cs":
            cutoff = x[8]
        else:
            cutoff = x[9]
            
        return (temperature(Xb, cutoff, pval),temperature(Xa, cutoff, pval))  
        
        
        
        
        