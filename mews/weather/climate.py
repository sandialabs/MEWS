# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:43:41 2021

@author: dlvilla
"""

from warnings import warn
from mews.data_requests.CMIP6 import CMIP_Data

import pandas as pd
import numpy as np
import os


class ClimateScenario(object):
    
    # the first value must be "historical" or some scenario that represents past
    # behavior of models!
    _valid_scenario_names = ["historical","SSP119","SSP126","SSP245","SSP370","SSP585"]
    _required_baseline_year_for_cmip6 = 2014
    _default_poly_order = {_valid_scenario_names[0]:6,
                           _valid_scenario_names[1]:3,
                           _valid_scenario_names[2]:2,
                           _valid_scenario_names[3]:2,
                           _valid_scenario_names[4]:2,
                           _valid_scenario_names[5]:2,}
    
    def __init__(self,use_global=True, 
                      lat=None,
                      lon=None,
                      baseline_year=2020,
                      end_year=2060,
                      model_guide="Models_Used_alpha.xlsx",
                      data_folder="data",
                      run_parallel=True,
                      output_folder=None,
                      proxy=None,
                      gcm_to_skip=[],
                      align_gcm_to_historical=False,
                      polynomial_order=_default_poly_order,
                      write_graphics_path=None):
        """
        Parameters
        ----------
        use_global : Bool, optional
            True: This defaults to the old MEWS v0.0.1 behavior where the global
            average from the IPCC Physical Basis Technical Summary is used for
            temperature increase. The default is True.
        lat : float, optional
            lattitude of the location where a temperature trajectory is needed. The default is None.
            Only takes effect if use_global_average = False. 
        lon : float, optional
            longitude of the location where a temperature trajectory is needed.
            Only takes effect if use_global_average = False. The default is None.
        baseline_year : int, optional - OBSOLETE FOR THE NEW USE CASE- this will 
            not change anything for the new use case!
            CMIP6 sets this to 2014 for all cases!            
                For use_global = True (old use case)
                Indicates the start year where temperature anomaly is measured from.
                The IPCC multiplication factors are reduced as baseline year becomes
                more recent in comparison to the 1850-1900 baseline.
        end_year: int, optional
            Indicates the end year for which CMIP6 evaluations will stop
        run_parallel : bool, optional
            Indicates whether to speed up process by parallelization by asyncrhonous pool
            and threading. 
        output_folder : str, optional
            path that indicates where all of the CMIP6 data files should end up.
            Default=None .. will install in folder "CMIP6_Data_Files" one level 
            above the MEWS root directory.
        proxy : str, optional
            name of proxy server that enables internet downloading of CMIP6
            data sets. Default = None .. indicates that no proxy server is 
            needed. The program may freeze if a proxy is needed and no
            proxy is given. 
        gcm_to_skip : list of str, optional
            list of GCM's to skip in an analysis. If a GCM can no longer download
            because of a server problem, then it should be listed here'
        align_gcm_to_historical : bool, optional : Default = False
            False : allows there to be a discontinuity between historic GCM ensemble
                    and the future GCM projections. There are significant differences
                    in the scenario start points though.
            True : shifts all GCM data for each scenario to start at the historic
                   average for the baseline year. GCM data is shifted up or down
                   to accomplish this.
        polynomial_order : dict: Default = self._default_poly_order 
            dict: must contain a scenario name list and an integer. 
        write_graphics_path : str : Default = None, 
            write out png files if this is not None. 

        

        Returns
        -------
        None.

        """
        self.use_global = use_global
        if use_global:
            file_dir = os.path.dirname(__file__)
            self.df_temp_anomaly = pd.read_csv(os.path.join(file_dir,"data",
                                                            "IPCC2021_ThePhysicalBasis_TechnicalSummaryFigSPM8.csv"))
        else:
            
            self.lat = lat
            self.lon = lon
            self.end_year = end_year
            warn("The baseline year input has no affect when use_global=False")
            self.baseline_year = self._required_baseline_year_for_cmip6
        self.table = [] # last column is the R2 value
        self.scenarios = []
        self.columns = [r'$t^3$',r'$t^2$',r'$t$',r'$1$',r'$R^2$']
        self.model_guide = model_guide
        self.data_folder = data_folder
        self.run_parallel = run_parallel
        self.output_folder = output_folder
        self.proxy = proxy
        self.gcm_to_skip = gcm_to_skip
        self.align_gcm_to_historical = align_gcm_to_historical
        self.write_graphics_path = write_graphics_path
        self._polynomial_order = polynomial_order
        
    def calculate_coef(self,scenario,lat=None,lon=None,scen_func_dict=None,years_per_calc=1):
        
        
        
        # use lat/lon 
        # not global is the new use case
        if not self.use_global:
            if lat is None:
                lat = self.lat
            if lon is None:
                lon = self.lon
            if not isinstance(scenario,list):
                raise TypeError("For 'self.use_global=True' the scenario input"+
                                " must be a list of strings that include any"+
                                " of " + str(self._valid_scenario_names))
            else:
                for scen in scenario:
                    if not scen in self._valid_scenario_names:
                        raise ValueError("An invalid scenario name '{0}' was entered".format(scen) +
                                         "\n\n Valid names are:\n\n" + str(self._valid_scenario_names))
                if not self._valid_scenario_names[0] in scenario:
                    scenario.insert(0,self._valid_scenario_names[0])
                        
            obj = CMIP_Data(lat_desired = lat,
                                        lon_desired=lon,
                                        baseline_year=self.baseline_year,
                                        end_year=self.end_year,
                                        file_path=os.path.abspath(os.getcwd()),
                                        model_guide=self.model_guide,
                                        data_folder=self.data_folder,
                                        world_map=True,
                                        calculate_error=False,
                                        display_logging=False,
                                        display_plots=False,
                                        output_folder=self.output_folder,
                                        run_parallel=self.run_parallel,
                                        proxy=self.proxy,
                                        gcm_to_skip=self.gcm_to_skip,
                                        scenario_list=scenario,
                                        polynomial_fit_order=self._polynomial_order,
                                        align_gcm_to_historical=self.align_gcm_to_historical)
            if not self.write_graphics_path is None:
                obj.display_plots = True
                obj.results1(write_png=self.write_graphics_path)
                
            
            # all polynomials are offset by this.

            
            for scen,cmip_data_obj in obj.total_model_data.items():
                # this gives a function that only needs the year as an input
                # to get SSP deltaT from the 1850-1900 baseline.
                import pdb;pdb.set_trace()
                polycoef = cmip_data_obj.total_model_data[scen].delT_polyfit
                
                scen_func_dict[scen] = np.poly1d(
                    cmip_data_obj.total_model_data[scen].delT_polyfit)
                
                
        else:
            # # TODO remove: THIS IS OLD BUT KEPT FOR LEGACY RESULTS FROM v0.0.1 - it will 
            # not be kept for v1.0.0
            df = self.df_temp_anomaly[self.df_temp_anomaly["Climate Scenario"]==scenario]
            
            Xi = ((df["Year"]-2020)/50).values
            # 1.0 is the amount of heating assumed from 1850-1900 to 2020.
            Yi = (df["Temperature Anomaly degC (baseline 1850-1900)"]-1.0).values
            
            p_coef, residuals, rank, singular_values, rcond = np.polyfit(Xi,Yi,3,full=True)
            self.p_coef = p_coef
            
            SSres = np.sum((Yi-np.polyval(p_coef,Xi))**2)
            SStot = np.sum((Yi - np.mean(Yi))**2)
            
            R2 = np.array([1 - (SSres/SStot)])
            
            self.table.append(np.concatenate((p_coef, R2)))
            self.scenarios.append(scenario)
            
            return scen_func_dict

    # 0.267015 shifts to make the baseline year 2020 we assume a 1.0C affect 
    def climate_temp_func(self,year):
        return np.polyval(self.p_coef,(year-2020)/50)-0.267015
    
    def write_coef_to_table(self,tex_file_path):
        table_df = pd.DataFrame(self.table,
                                index=self.scenarios,columns=self.columns)
        table_df.to_latex(tex_file_path,
                          float_format="%.6G", escape=False)