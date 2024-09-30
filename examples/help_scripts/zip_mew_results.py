# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:34:43 2023

@author: dlvilla
"""

from zipfile import ZipFile
from zipfile import ZIP_DEFLATED
import os
import threading


def thread_function(realization,name,base_name,base_folder,files):
    count = 0
    if realization < 10:
        end_ind = -5
    elif realization < 100:
        end_ind = -6
    elif realization < 1000:
        end_ind = -7
    else:
        raise ValueError("The script only allows realizations up to 999 expand the code to include higher cases")
    
    os.chdir(base_folder)
    
    with ZipFile('Kodiak' + str(realization) + ".zip",'w', ZIP_DEFLATED) as zippy:
        for file in files:
            if (base_name in file) and str(realization) in file[end_ind:] and file[end_ind-1] == "r":
                zippy.write(file) 
                print("Count={0:d},Realization={1:d},File={2}".format(count,realization,file))
                count+=1
                
    print(name + str(realization) + ".zip is complete\n\n\n\n\n\n" )


base_name = "USA_MA_Worcester.Rgnl.AP.725095_TMY3"

name = "Worcester"

base_folder = os.path.join(os.path.dirname(__file__),"example_data",
                           "Worcester","mews_results_krissy2")

files = os.listdir(base_folder)

threads = list()
for realization in range(100):
    x = threading.Thread(target=thread_function, args=(realization,name,base_name,base_folder,files))
    threads.append(x)
    x.start()
    
for realization, thread in enumerate(threads):
    thread.join()
    

