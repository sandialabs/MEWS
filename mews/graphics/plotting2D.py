# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:49:21 2021

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
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib._color_data as mcd
from mews.utilities.utilities import linear_interp_discreet_func, bin_avg, histogram_step_wise_integral


class Graphics():

    @staticmethod
    def plot_realization(extremes_result, column, realization_number, ax=None, title="", ylabel="",
                         grid_status=True, rc_input={}, legend_labels=None,fig=None):

        # change fontsize and graphical attributes of the figure see documentation
        # for matplotlib.rc
        rc(rc_input)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        for key0, subdict in extremes_result.items():
            for key, objA in subdict.items():
                wfile = key[0]
                real_id = key[1]
                if real_id == realization_number:
                    year = key[2]
                    date_time1 = objA.reindex_2_datetime()
                    date_time2 = objA.reindex_2_datetime(original=True)
                    ser = date_time1[column]
                    ser2 = date_time2[column]
                    if legend_labels is None:
                        l1 = None
                        l2 = None
                    else:
                        l1 = legend_labels[0]
                        l2 = legend_labels[1]
                    ser.plot(ax=ax, label=l1)
                    ser2.plot(ax=ax, label=l2)
                    ax.set_title(title)
                    ax.set_ylabel(ylabel)
                    ax.grid(grid_status)
        return fig, ax

    @staticmethod
    def plot_sample_dist_shift(hist0, histT_tuple_dict, ipcc_shift, thresholds_dict, events, plot_title=None, fig_path=None, is_temperature=True, plot_results=False):
        """


        Parameters
        ----------


        Returns
        -------
        None.

        """
        font = {'size': 16}
        rc('font', **font)
        fig, axl = plt.subplots(len(histT_tuple_dict), 1, figsize=(
            10, 4*len(histT_tuple_dict)), sharex=True, sharey=True)
        if len(histT_tuple_dict) == 1:
            axl = [axl]
        # adjust historical bins if needed.
        

        for histT_tup, thresh_tup, ax in zip(histT_tuple_dict.items(), thresholds_dict.items(), axl):
            labels_given = False
            bin_prev = None
            for wtype in events:
                
                if is_temperature:
                    if labels_given:
                        labels = [None,None]
                    else:
                        labels_given = True
                        labels = ['historic','future']
                else:
                    labels = ['historic','future']
                
                bin0 = bin_avg(hist0[wtype])
                fun0 = hist0[wtype][0]
                
                # this is needed because durations comes from np.histogram whereas
                # the histT_tuple comes with a cdf and pdf together.

                if not histT_tup[1][wtype] is None:
                    if type(histT_tup[1][wtype][0]) is tuple:
                        histT = histT_tup[1][wtype][0]
                        ttup = thresh_tup[1][wtype]
                    else:
                        histT = histT_tup[1][wtype]
                        ttup = thresh_tup[1]
                    
                    binT = bin_avg(histT)
                else:
                    binT = None
                    ttup = None
                
                if bin_prev is None:
                    second_condition = False
                else:
                    second_condition = bin0.sum() > 0 and bin_prev.sum() < 0
                
                if (bin0.sum() < 0 and bin_prev is None) or second_condition:
                    # cold snap, heat wave plot for temperature does not need
                    # different colors and labels because cs are negative and
                    # heat waves are positive.
                    
                    areaT = histogram_step_wise_integral(histT)
                    area0 = histogram_step_wise_integral(hist0[wtype])
                    
                    ax.plot(bin0, fun0/area0, label=labels[0], color='blue')
                    
                    if not binT is None: 
                        ax.plot(binT, histT[0]/areaT, label=labels[1], color='orange')
                        
                    bin_prev = bin0
                elif bin_prev is None:
                    # duration plotting.
                    # no normalization because we are duration plotting.
                    ax.plot(bin0, fun0, label=labels[0] + " " + wtype, color='blue')
                    
                    if not binT is None:
                        ax.plot(binT, histT[0], label=labels[1] + " " + wtype, color='orange')
                    
                    bin_prev = binT
                    
                else:
                    # duration second plot for hw.
                    ax.plot(bin0, fun0, label=labels[0] + " " + wtype, color='green')
                    
                    if not binT is None:
                        ax.plot(binT, histT[0], label=labels[1] + " " + wtype, color='red')
                        

                
                ylim = ax.get_ylim()
    
                #ax.set_ylim((0, ylim[1]))
                ylim = (0, ylim[1])
                xlim = ax.get_xlim()
    
                on_first = 0
                
                if not ttup is None and is_temperature:
                    for tstr, ystr, pos, color, linestyle in zip(['actual', 'target', 'actual', 'target'],
                                                                 ['10 year', '10 year',
                                                                     '50 year', '50 year'],
                                                                 [0, 0, 1, 1], 
                                                                 ['blue', 
                                                                  mcd.CSS4_COLORS['darkblue'], 
                                                                  mcd.CSS4_COLORS['salmon'], 'red'], 
                                                                 [':', "--", ":", "--"]):
                        thresh = ttup[tstr][pos]
                        ax.plot([thresh, thresh], ylim, label=ystr + " " +
                                tstr, linestyle=linestyle, color=color, linewidth=1)
                        if tstr == 'target':
                            if on_first == 0:
                                on_first = 1
                                color2 = mcd.CSS4_COLORS['turquoise']
                                arr_post = 1/3
                                if not labels[0] is None:
                                    label = "10 year historic"
                                else:
                                    label = None
                            else:
                                color2 = mcd.CSS4_COLORS['orangered']
                                arr_post = 2/3
                                if labels[0] is None:
                                    label = None
                                else:
                                    label = "50 year historic"
                            hist_thresh = thresh-ipcc_shift[wtype]['temperature'][ystr]
                            ax.plot([hist_thresh, hist_thresh], ylim, linestyle=linestyle,
                                    color=color2, linewidth=1., label=label)
                            ax.arrow(hist_thresh, arr_post * ylim[1],
                                     thresh-hist_thresh-1.0,
                                     0.0,
                                     width=ylim[1]/150,
                                     head_width=ylim[1]/30,
                                     head_length=xlim[1]/60)

                ax.grid('on')
                if is_temperature:
                    ax.set_ylabel('Probability Density')
                else:
                    ax.set_ylabel('Number of waves')
            if is_temperature:
                ax.set_xlabel(
                    "Peak temperature increase from daily climate norms average ($^{\circ}C$)")
            else:
                ax.set_xlabel("Duration of wave (hr)")
            ax.legend(fontsize=12)
        
        plt.tight_layout()
        if not plot_title is None:
            axl[0].set_title(plot_title)
        if not fig_path is None:
            plt.savefig(fig_path, dpi=300)
            
        if not plot_results:
            plt.close(fig)
