# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:49:21 2021

@author: dlvilla
"""
from matplotlib import pyplot as plt
from matplotlib import rc

class Graphics():
    
    @staticmethod
    def plot_realization(extremes_result,column,realization_number,ax=None,title="",ylabel="",
                     grid_status=True,rc_input={},legend_labels=None):
    
        # change fontsize and graphical attributes of the figure see documentation
        # for matplotlib.rc
        rc(rc_input)
        
        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=(10,10))
        
        for key,objA in extremes_result.items():
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
                ser.plot(ax=ax,label=l1)
                ser2.plot(ax=ax,label=l2)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.grid(grid_status)
        
        
        
        
    