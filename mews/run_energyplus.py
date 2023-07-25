#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:25:21 2023

@author: dlvilla
"""
import os
import warnings
import subprocess
from mews.errors.exceptions import EnergyPlusRunFailed
import shutil



class RunEnergyPlusStudy(object):
    
    _complete_file_ = "mews_study_complete.out"
    
    def __init__(self,weather_files_path,
                      idfs_path,
                      energyplus_path,
                      output_path,
                      use_python_api = False,
                      post_process_func = None,
                      restart = False,
                      run_parallel = True,
                      num_cpu = -1):
        """
        This class runs energy plus through the python API to energy plus 
        it creates a directory structure in "out_path" of the idf name (minus idf)
        and then the weather file names. 
        
        Inputs
        ------
        weather_files_path : str  
            A path where all weather files to be run are located. Every entry
            that ends in *.epw will be run in the study
        
        idfs_path : str
            A path where all energyplus idf input files are located. All files
            ending in *.idf will be run
            
        energyplus_path
            A path that is the root of the energyplus repository (not to an
            an executable)
            
        output_path
            A path for the root of output of the study
            
        use_python_api : bool : optional : Default = False
            Indicates whether to use the python API for energyplus 
            to run energy plus directly.
            
        post_process_func : func : optional : NOT IMPLEMENTED YET!
            A function that will be applied to every run of the study after
            completion of the energyplus run
            If = None, no postprocessing is done
            
        restart : bool :optional
            If True, then previous runs will be overwritten
            If False, then previous runs will be assessed for completion
            and not re-run if a file mews_study_complete.out has been written by this program
            
        run_parallel : bool : optional
            Indicates whether to run parallel using the multiprocessing library
            
        num_cpu : int : optional
            Number of cpus to run parallel -1 indicates maximum number available
            
        Returns
        -------
        
        self.results - a dictionary with tuples of the idf and epw file as the
                       key. If a value is None, then there were no errors
                       and the run was successful. If an Exception is present
                       for a given tuple, that run failed and explanation 
                       can be found by raising the exception.
        
        Exceptions
        ----------
        EnergyPlusRunFailed - indicates that energyplus did not succeed 
                              and that an error message from its output is
                              issued with the exception.
        
        """
        # internal variables
        self.energyplus_path = energyplus_path
        self.out_path = output_path
        self.restart = restart
        self.use_python_api = use_python_api

        # setup parallel processing
        if run_parallel:
            pool = self._setup_parallel(num_cpu)
            if isinstance(pool,Exception):
                raise pool
                
        # build parameter study lists (epw = energyplus weather, idf = energyplus input)
        idf_list = [os.path.join(idfs_path,file) for file in os.listdir(idfs_path)  if file[-4:] == ".idf"]
        epw_list = [os.path.join(weather_files_path,file) for file in os.listdir(weather_files_path) if file[-4:] == ".epw"]

        results = {}
        # loop over idfs and weather files.
        for idf in idf_list:
            
            for epw in epw_list:
                
                tup = (idf,
                       epw)

                if run_parallel:

                    results[tup] = pool.apply_async(self._run,
                                                         args=tup)
                else:
                    results[tup] = self._run(*tup)

        # gather any results passed through python.      
        if run_parallel:
            results_get = {}
            for tup,poolObj in results.items():
                try:
                    results_get[tup] = poolObj.get()
                except Exception as excep:
                    results_get[tup] = excep   
                    
            pool.close()

            # Any exceptions can be found here
            self.results = results_get
        else:
            self.results = results
            
    def _setup_parallel(self,num_cpu):
        try:
            import multiprocessing as mp
            machine_num_cpu = mp.cpu_count()
            too_many_requested = num_cpu > machine_num_cpu
            
            if too_many_requested:
                warnings.warn("The number of CPU's requested was '{0:d}' but only '{1:d}' were available!".format(
                    num_cpu,machine_num_cpu))
            
            if num_cpu == -1 or too_many_requested:
                pool = mp.Pool(machine_num_cpu-1)
            else:
                pool = mp.Pool(num_cpu)
            return pool
        except Exception as excep:
            warnings.warn("Setting up a pool for multi-processing failed!")
            return excep

    def _directory_creation(self,idf,wfile):
        
        idf_out_path = os.path.join(self.out_path,os.path.basename(idf[:-4]))
        epw_out_path = os.path.join(idf_out_path,os.path.basename(wfile[:-4]))
        
        if not os.path.exists(idf_out_path):
            try:
                os.mkdir(idf_out_path)
            except Exception as excep:
                warnings.warn("Failed to create directory '" + idf_out_path + "'")
                return excep
        
        for opath in [epw_out_path]:
            path_exists = os.path.exists(opath)
            if self.restart and path_exists:
                try:
                    shutil.rmtree(opath)
                except Exception as excep:
                    breakpoint()
                    warnings.warn("Failed to remove an output path '"+opath+"'")
                    return excep
                execute_run = True
            elif not path_exists:
                execute_run = True
            else:
                execute_run = False
            
            if not os.path.exists(opath):
                try:
                    os.mkdir(opath)
                except Exception as excep:
                    warnings.warn("Failed to create directory '" + opath + "'")
                    return excep
        return epw_out_path, execute_run
    
    def _import_energyplus_api(self):
        
        curpath=os.getcwd()
        try:
            os.chdir(self.energyplus_path)
            from pyenergyplus import api as ep_api
            from pyenergyplus import state as ep_state
        except Exception as excep:
            warnings.warn("The energy plus api failed to import! The "+
                          "energyplus_path must have a subdirectory"+
                          " 'pyenergyplus'.")
            os.chdir(curpath)
            return excep
        else:
            os.chdir(curpath)
            
        return ep_api, ep_state
            
    def _run(self,idf,wfile):
        
        if self.use_python_api:
            tup = self._import_energyplus_api()        
            if isinstance(tup,tuple):
                ep_api, ep_state = tup
            else:
                return tup
        
        
        out_path, execute_run = self._directory_creation(idf,wfile)
        
        if isinstance(out_path,str):
            try: 
                if execute_run:
                    if self.use_python_api:
                        ep_obj = ep_api.EnergyPlusAPI()
                        
                        state_manager = ep_state.StateManager(ep_obj.api)
            
                        state = state_manager.new_state()
                        
                        # TODO implement callbacks
                        #ep_obj.runtime.callback_progress(state,example_callback)
            
                        ep_obj.runtime.run_energyplus(state,["-d",out_path,"-w",
                                                      wfile,
                                                      idf])
                    else:
                        # Now let's run EP  
                        run_succeeded = self._execute_command([self.energyplus_path,"-d",out_path ,"-w", wfile, idf])
                        if not run_succeeded[0]:
                            raise EnergyPlusRunFailed(run_succeeded[1]+" \n\nPath="+out_path)
                    
                
            except Exception as excep:
                warnings.warn("Energy plus failed to run successfully! Path="+out_path)
                return excep
            
            try:
                with open(os.path.join(out_path,self._complete_file_),'w') as fi:
                    fi.write("The run completed successfully!")
            except Exception as excep:
                warnings.warn("The completion file '"+self._complete_file_+
                              "' failed to write!")
            
        else:
            # an exception was returned
            return out_path
        
        
        return None
    
    def _execute_command(self,input0):
        df = subprocess.Popen(input0, stdout=subprocess.DEVNULL)  
        output, err = df.communicate()

        #mesg1 = output.decode('utf-8').split('\n')[1]
        #mesg2 = output.decode('utf-8').split('\n')[-2]
        if not err is None:
            errmsg = err.decode('utf-8')
        else:
            errmsg = None
        if not errmsg is None:
            return False, errmsg
        else:
            return True, errmsg
            