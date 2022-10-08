#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:28:02 2022

@author: dlvilla
"""



class SolveDistributionShift(object):
    
    
    def __init__(self, markov_stat_model, param0, hist0):
        """
        
        Parameters
        ----------
        
        markov_stat_model : function 
            This function must return a histogram of peak temperature for heat
            waves with bin intervals equivalent to hist0 
            (i.e. for MEWS this is 24 hr intervals for all bins), it must 
            The function must evaluate a history of 
            
        param0 : array-like
            contains a list of the initial values for input parameters
            to markov_stat_model
            
        hist0 : 2-tuple - as returned by np.histogram but with probabilities
            hist0[0] is the histogram probabilities (not count )
            hist0[1] are the historgram bin edges of dimension n+1
            
        
        
        """
        
        
        self._markov_stat_model = markov_stat_model
        
        
        
        
        
        
        
    def _residuals(self,hist1,hist2, P_thresh, delT,f_mult):
        """
        This function evaluates whether two heat wave maximum temperature
        histograms (hist1 and hist2) are shifted by delT at specific
        probability thresholds "P_thresh" multiplied by f_mult.
        An array of thresholds temperatures
        delT_Pthresh is solved by finding the Cumulative Distribution Function 
        (CDF) of hist 1 and then solving for delT_Pthresh such that
        
        P_hist1(T >= delT_Pthresh) = P_thresh. 
        
        Each of these thresholds are then shifted by delT (also an array) and 
        resultant probabilities P_thresh_2 solved such that
        
        P_hist2(T >= delT_Pthresh + delT) = P_thresh_2
        
        The function then returns residuals between P_thresh and 
        P_thresh_2 * f_mult
        
        residuals = (P_thresh - fmult*P_thresh_2)**2
        
        This function is intended to evaluate how well Intergovernmental Panel
        on climate change intensity (delT) and frequency multipliers have been
        accomplished by a dynamic markov process that simulates normal times,
        cold snap times, and heat wave times. Wave events are sampled from a 
        truncated Gaussian process for duration normalized temperature these
        are then used to evaluate maximum (heat waves) or minimum (cold snaps)
        temperatures for each wave event.
        
        Parameters
        ----------
        hist1 : 
            
        hist2 :
            
        delT : array
        
        
        
        
        
        """
        pass
    