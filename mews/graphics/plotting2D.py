# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:49:21 2021

@author: dlvilla
"""
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib._color_data as mcd
from mews.utilities.utilities import linear_interp_discreet_func, bin_avg


class Graphics():

    @staticmethod
    def plot_realization(extremes_result, column, realization_number, ax=None, title="", ylabel="",
                         grid_status=True, rc_input={}, legend_labels=None):

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
    def plot_sample_dist_shift(hist0, histT_tuple_dict, ipcc_shift, thresholds_dict, plot_title=None, fig_path=None):
        """


        Parameters
        ----------
        funcs : list 
            list of 2-tuples of arrays of equal length representing a function.
            where tup[0] gives y values and tup[1] gives x values the same 
            as the np.histogram function. If tup[1] is 1 element longer, the
            function assumes that bin edges have been given rather than direct 
            values and the average between edges is used.
        ipcc_data : dict
            dictionary with key 'temperature' and 'frequency' subdictionary must
            contain keys describing what kind of event (ussually '10 year' and 
                                                        '50 year')
        thresh : dict
            dictionary with same structure at the second level as ipcc_data 
            values represent the actual temperature for '10 year' and '50 year'
            events.
            temperature this is the historic 10 or 50 year event. This event
            is then shifted forward by ipcc_data['temperature']['10 year'] or '
            50 year' 


        Returns
        -------
        None.

        """
        font = {'size': 16}
        rc('font', **font)
        fig, axl = plt.subplots(len(histT_tuple_dict), 1, figsize=(
            10, 10), sharex=True, sharey=True)
        if len(histT_tuple_dict) == 1:
            axl = [axl]
        # adjust historical bins if needed.
        bin0 = bin_avg(hist0)
        fun0 = hist0[0]

        for histT_tup, thresh_tup, ax in zip(histT_tuple_dict.items(), thresholds_dict.items(), axl):
            histT = histT_tup[1][0]
            binT = bin_avg(histT)
            ax.plot(bin0, fun0/fun0.sum(), label='historic')

            ax.plot(binT, histT[0], label='shifted')

            ylim = ax.get_ylim()

            ax.set_ylim((0, ylim[1]))
            ylim = (0, ylim[1])

            on_first = 0
            
            if not thresh_tup[1] is None:
                for tstr, ystr, pos, color, linestyle in zip(['actual', 'target', 'actual', 'target'],
                                                             ['10 year', '10 year',
                                                                 '50 year', '50 year'],
                                                             [0, 0, 1, 1], 
                                                             ['blue', 
                                                              mcd.CSS4_COLORS['darkblue'], 
                                                              mcd.CSS4_COLORS['salmon'], 'red'], 
                                                             [':', "--", ":", "--"]):
                    thresh = thresh_tup[1][tstr][pos]
                    ax.plot([thresh, thresh], ylim, label=ystr + " " +
                            tstr, linestyle=linestyle, color=color, linewidth=1)
                    if tstr == 'target':
                        if on_first == 0:
                            on_first = 1
                            color2 = mcd.CSS4_COLORS['turquoise']
                            arr_post = 1/3
                            label = "10 year historic"
                        else:
                            color2 = mcd.CSS4_COLORS['orangered']
                            arr_post = 2/3
                            label = "50 year historic"
                        hist_thresh = thresh-ipcc_shift['temperature'][ystr]
                        ax.plot([hist_thresh, hist_thresh], ylim, linestyle=linestyle,
                                color=color2, linewidth=1., label=label)
                        ax.arrow(hist_thresh, arr_post * ylim[1],
                                 thresh-hist_thresh-1.0,
                                 0.0,
                                 width=0.0005,
                                 head_width=0.005,
                                 head_length=1.0)

            ax.grid('on')
            ax.set_ylabel('Probability Density')
        ax.set_xlabel(
            "Peak temperature increase from daily climate norms average ($^{\circ}C$)")
        ax.legend(fontsize=12)

        if not plot_title is None:
            axl[0].set_title(plot_title)
        if not fig_path is None:
            plt.savefig(fig_path, dpi=300)
