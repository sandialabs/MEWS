# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:50:39 2021

Copyright Notice
=================

Copyright 2021 National Technology and Engineering Solutions of Sandia, LLC. 
Under the terms of Contract DE-NA0003525, there is a non-exclusive license 
for use of this work by or on behalf of the U.S. Government. 
Export of this program may require a license from the 
United States Government.

Please refer to the LICENSE.md file for a full description of the license
terms for MEWS. 

The license for MEWS is the Modified BSD License and copyright information
must be replicated in any derivative works that use the source code.

@author: dlvilla

MEWS = Multi-senario Extreme Weather Simulator
"""

import os
from datetime import datetime
import types
import numpy as np
import pandas as pd
from calendar import isleap
from copy import deepcopy
from mews.epw import epw
from mews.errors import EPWMissingDataFromFile, EPWFileReadFailure, EPWRepeatDateError
from mews.weather.doe2weather import DOE2Weather
import warnings
from copy import deepcopy

class Alter(object):

    """
    Alter energy plus or doe2 weather files - shorter or longer than a year permitted.
    
    >>> obj = Alter(weather_file_path,
                    replace_year,
                    check_types)
    
    This is effectively the "read" function.
    
    If replace_year causes new coincidence with leap years, then an 
    extra day is added (Feb 28th if repeated). If a replace_year moves
    from a leap year to a non-leap year, then Feb 29th is deleted if it 
    is present.
    
    Returns the altered object.
    
    Parameters
    ----------
    weather_file_path : str 
        A valid path to an energy plus weather file.
    replace_year : int : optional : Default = None
        If None:
            Leave the year in the energy plus file unaltered.
        If int: 
            Change the year to the value for all rows.
    check_types : bool : optional : Default = True
        If True:
            Check types in all functions (slow but safe).
        If False: 
            Do not check types (likely to get less understandable errors
            if something goes wrong)
    clear_alterations : optional: bool 
        NOT CURRENTLY USED.
        Remove previous alterations from the current 
        object. Reissuing a read command to an Alter object allows a new set
        of weather to recieve the same alterations a previous set recieved.            
    
    isdoe2 : optional: bool : ONLY NEEDED IF DOE2
        Read in a DOE2 weather file instead of an EPW and then
        wrangle that data into an EP format for the self.epwobj database.
    use_exe : optional : bool : ONLY NEEDED IF DOE2
        etermine whether to use BIN2TXT and TXT2BIN
        executables (True) or native Python to read BIN files.
    doe2_bin2txt_path : str : optional : ONLY NEEDED IF DOE2
        A valid path to an executable
        that converts a DOE2 filename.BIN weather file to an ASCII text file. 
        The executable is assumed to be named BIN2TXT.EXE and comes with
        any distribution of DOE2 which can be obtained by forming a license
        agreement with James Hirsch and Associates (www.doe2.com). A folder
        in mews "third_party_software" can be used to put the filename.EXE
    doe2_start_datetime : datetime : optional : ONLY NEEDED IF DOE2
    Input the start time for the weather file.
    doe2_hour_in_file : optional : ONLY NEEDED IF DOE2
        Must be 8760 or 8784 for leap years.
    doe2_timezone : str : optional : ONLY NEEDED IF DOE2
        Name of time zone applicable to the doe2 file.
    doe2_dst : optional : ONLY NEEDED IF DOE2
        List/tuple with 2 entries that are datetimes with begin and 
        end times for day light savings time.
    
    Returns
    ------
    None
   
    """
    
    def __init__(self,weather_file_path,replace_year=None,check_types=True,
                 isdoe2=False,use_exe=False,doe2_bin2txt_path=r"../third_party_software/BIN2TXT.EXE",
                 doe2_start_datetime=None,doe2_tz=None,doe2_hour_in_file=8760,doe2_dst=None):
        

        obj = DOE2Weather()
        self.df2bin = obj.df2bin
        self.bin2df = obj.bin2df
        
        self.read(weather_file_path,replace_year,check_types,True,isdoe2,use_exe,
                  doe2_bin2txt_path,doe2_start_datetime,doe2_tz,
                  doe2_hour_in_file,doe2_dst)
        
        
    def _leap_year_replacements(self,df,year,isdoe2):
            Feb28 = df[(df["Day"] == 28) & (df["Month"] == 2) & (df["Year"]==year)]
            Mar1 = df[(df["Month"] == 3) & (df["Day"] == 1) & (df["Year"]==year)]
            Feb29 = df[(df["Month"] == 2) & (df["Day"] == 29) & (df["Year"]==year)]
            hasFeb29 = len(Feb29) > 0
            hasMar1 = len(Mar1) > 0
            hasFeb28 = len(Feb28) > 0
            
            
            if isleap(year) and (not hasFeb29) and (hasMar1 and hasFeb28):
                # add Feb29 by replicating Feb28th
                indFeb28 = Feb28.index[-1]
                sdict = {}

                for name,col in Feb28.iteritems():
                    col.index = np.arange(indFeb28+1,indFeb28+25)
                    sdict[name] = col

                Feb29 = pd.DataFrame(sdict)
                Feb29["Day"] = 29
                
                sdict = {}
                for name,col in df.loc[indFeb28+1:].iteritems():
                    col.index = np.arange(indFeb28+25,
                                          indFeb28+25 + (df.index[-1] - indFeb28))
                    sdict[name] = col
                    
                restOfYear = pd.DataFrame(sdict)
                df_new = pd.concat([df.loc[0:indFeb28],Feb29,restOfYear])
            elif not isleap(year) and hasFeb29:
                df_new = df.drop(Feb29.index)
            else:
                df_new = df
            
            if isdoe2:
                # keep custom attributes like "headers" that were added elsewhere
                df_new.headers = df.headers
            
            
              
            return df_new

                
        
    def _init_check_inputs(self,replace_year,check_types):
        if not isinstance(check_types,bool):
            raise TypeError("'check_types' input must be a boolean!")
        if check_types:       
            if isinstance(replace_year, int):
                if replace_year < 0:
                    positive_message = "'replace_year' must be positive"
                    raise ValueError(positive_message)
            elif replace_year is None:
                pass
            else:
                raise TypeError("'replace_year' must be a positive integer or None")

    
    def __str__(self):
        wfp_str = ""
        alt_str = "None"
        if hasattr(self,'alterations'):
            if len(self.alterations) > 0:
                alt_str = str(self.alterations)
        if hasattr(self,'wfp'):
            wfp_str = self.wfp
        return ("mews.weather.alter.Alter: \n\nWeather for: \n\n'{0}'\n\nwith".format(wfp_str)
                +" alterations:\n\n{0}".format(alt_str))
    
    
    def _get_date_ind(self,year=None,month=None,day=None,hour=None,num=None):
        """
        Query for year, month, day, hour. If an input is not provided
                 query for any value of that entry
        """
        df = self.epwobj.dataframe
        
        all_true = df["Year"] == df["Year"]
        if year is None:
            yr_comparison = all_true
        else:
            yr_comparison = df["Year"] == year
        
        if month is None:
            mo_comparison = all_true
        else:
            mo_comparison = df["Month"] == month
            
        if day is None:
            day_comparison = all_true
        else:
            day_comparison = df["Day"] == day
        
        if hour is None:
            hr_comparison = all_true
        else:
            hr_comparison = df["Hour"] == hour
            
        
        ind_list = df[yr_comparison & mo_comparison & day_comparison 
                 & hr_comparison].index
        
        if num is None:
            return ind_list
        else:
            return ind_list[:num]
        
        
    def add_alteration(self,
                      year,
                      day,
                      month,
                      hour,
                      duration,
                      peak_delta,
                      shape_func=lambda x: np.sin(x*np.pi),
                      column='Dry Bulb Temperature',
                      alteration_name=None,
                      averaging_steps=1):
        """
        add_alteration(year, day, month, hour, duration, peak_delta, shape_func, column)
        
        Alter weather file by adding a shape function and delta to one weather variable.
        
        Internal epwobj is altered use "write" to write the result.
            
        Parameters
        ----------
        year : int 
            Year of start date.
        day : int 
            Day of month on which the heat wave starts.     
        month : int 
            Month of year on which the heat wave starts.
        hour : int 
            Hour of day on which the heat wave starts (1-24).
        Duration : int 
            Number of hours that the heat wave lasts. if = -1
            then the change is applied to the end of the weather file.
        peak_delta : float 
            Peak value change from original weather at "shape_func" maximum.
        shape_func : function|list|tuple|np.array 
            A function or array whose range interval [0,1] will be mapped to 
            [0,duration] in hours. The function will be normalized to have a peak 
            of 1 for its peak value over [0,1]. For example, a sine function could 
            be lambda x: sin(pi*x). This shape is applied in adding the heat wave.
            from the start time to duration_hours later. If the input is an
            array, it must have 'duration' number of entries.
        column : str : optional : Default = 'Dry Bulb Temperature'
            Must be an entry in the column names of the energy plus weather
            file.
        alteration_name : any type that can be a key for dict : optional : Default = None
                If None:
                    Name is "Alteration X" where X is the current
                    number of alterations + 1.
                If any other type: 
                    Must not be a repeat of previously added alterations.
        averaging_steps : int : optional : Default = 1
             The number of steps to average the weather signal over when
             adding the heat wave. For example, if heat wave statistics
             come from daily data, then additions need to be made w/r to
             the daily average and this should be 24.
                
        Returns
        -------
        None 
            
        """
        df = self.epwobj.dataframe
             
        
        # special handling
        if alteration_name is None:
            num_alterations = len(self.alterations)
            alteration_name = "Alteration {0:d}".format(num_alterations + 1)
        if duration == -1:
            duration = len(df) - self._get_date_ind(year,month,day,hour)[-1]
            
        ## TYPE CHECKING
        if self.check_types:
            try:
                start_date = datetime(year,month,day,hour)
            except ValueError:
                raise ValueError("hour must be in 0 .. 23")
            if duration < 0:
                raise ValueError("The 'duration' input must be positive and less than a year (in hours)")
                
            correct_type_found = True
            if not isinstance(shape_func,(types.FunctionType)):
                if isinstance(shape_func,list):
                    shape_func = np.array(shape_func)
                if isinstance(shape_func,np.ndarray):
                    num = len(shape_func)
                    if num != duration:
                        raise ValueError("If the shape_func is provided as a "
                                         +"list, it must have length of the"
                                         +" 'duration' input")
                else:
                    correct_type_found = False
                    
            if not correct_type_found:
                raise TypeError("The shape_func must be of type 'function','list', or 'numpy.ndarray'")
            if not column in df.columns:
                raise ValueError("The 'column' input must be within the list:" + str(df.columns))
            
            if alteration_name in self.alterations:
                raise ValueError("The 'alteration_name' {} is already taken!".format(alteration_name))
        # END TYPE CHECKING
        
        if self.base1:
            hourin = hour + 1 
        else:
            hourin = hour
        bind_list = self._get_date_ind(year,month,day,hourin)
        if len(bind_list) > 1:
            raise EPWRepeatDateError("The date: {0} has a repeat entry!".format(
                str(datetime(year,month,day,hour))))
        elif len(bind_list) == 0:
            raise Exception("The requested alteration dates are outside the"+
                            " range of weather data in the current data!")
        else:
            bind = bind_list[0]
        
        
        eind = bind + duration
        
        if eind > len(df)+1:  # remember eind is not included in the range.
            raise pd.errors.OutOfBoundsTimedelta("The specified start time and"
                                                 +"duration exceed the bounds"
                                                 +"of the weather file's data!")
        if isinstance(shape_func,types.FunctionType):
            func_values = np.array([shape_func(x/duration) for x in range(duration)])
        else:
            func_values = shape_func
        extremum = max(abs(func_values.max()),abs(func_values.min()))
        if extremum == 0:
            raise ZeroDivisionError("The shape_func input has an extremum of"
                                    +" zero. This can occur for very short"
                                    +" alterations where the shape function"
                                    +" begins and ends with zero or by"
                                    +" passing all zeros to a shape function.")
        else:
            normalized_func_values = peak_delta * np.abs(func_values) / extremum
        
        addseg = pd.DataFrame(normalized_func_values,index=range(bind,eind),columns=[column])
        if averaging_steps > 1:
            #TODO - this needs to be moved elsewhere. You have a lot of work
            # to do to add daily average based heat waves correctly.
            df_avg = df.loc[:,column].rolling(window=averaging_steps).mean()
            # assure first steps have numeric values
            df_avg.iloc[:averaging_steps] = df.loc[:,column].iloc[:averaging_steps]
            
            df_diff = df.loc[bind:eind-1,column] - df_avg.loc[bind:eind-1]
            if addseg.sum().values[0] < 0:
                # cold snap
                scale = ((addseg.min() - df_diff.min())/addseg.min()).values[0]

            else:
                # heat wave
                
                if addseg.max()[0] <= 0:
                    scale = 0.0
                else:
                    scale = ((addseg.max() - df_diff.max())/addseg.max()).values[0]
            if scale > 0:
                addsegmod = addseg * scale
            else:
                addsegmod = addseg * 0.0
            
           
                
        else:
            addsegmod = addseg
        
        #df_org = deepcopy(df.loc[bind:eind-1,column])
        df.loc[bind:eind-1,column] = df.loc[bind:eind-1,column] + addsegmod.loc[bind:eind-1,column]
        self.alterations[alteration_name] = addseg
        
    def read(self,weather_file_path,replace_year=None,check_types=True,
             clear_alterations=False,isdoe2=False,use_exe=False,doe2_bin2txt_path=r"../third_party_software/BIN2TXT.EXE",
             doe2_start_datetime=None,doe2_tz=None,doe2_hour_in_file=8760,doe2_dst=None):
        
        """
        Reads an new Energy Plus Weather (epw) file (or doe2 filename.bin) while optionally 
        keeping previously added alterations in obj.alterations.
        
        >>> obj.read(weather_file_path,
                     replace_year=None,
                     check_types=True,
                     clear_alterations=False,
                     isdoe2=False,
                     use_exe=False,
                     doe2_bin2txt_path=r"../third_party_software/BIN2TXT.EXE",
                     doe2_start_datetime=None,
                     doe2_tz=None,
                     doe2_hour_in_file=8760,
                     doe2_dst=None)
        
        The input has three valid modes:
            Energy Plus: only input weather_file_path and optionally:
                replace_year, check_types, and clear_alterations
            DOE2 native python: All Energy Plus AND: isdoe2=True:
                use_exe=False (default) all other inputs are optional:
            DOE2 using exe: All inputs (even optional ones) are REQUIRED
                except doe2_bin2txt_path, replace_year, check_types, and
                clear_alterations remain optional.
                
        Once you have used one mode, there is no crossing over to another mode
        
        Warning: 
            This function resets the entire object and is equivalent 
            to creating a new object except that the previously entered 
            alterations are left intact. This allows for these altertions
            to be applied in proportionately the same positions for a new
            weather history.
                
        If replace_year causes new coincidence with leap years, then an 
        extra day is added (Feb 28th if repeated). If a replace_year moves
        from a leap year to a non-leap year, then Feb 29th is deleted if it 
        is present.
        
        obj.epwobj has a new dataset afterwards.
        
        Parameters
        ----------
        weather_file_path : str 
            Avalid path to an energy plus weather file
            or, if isdoe2=True, then to a DOE2 bin weather file. Many additioanl
            inputs are needed for the doe2 option.
        replace_year : int : optional : Default = None : REQUIRED IF isdoe2=True
            If None:
                Leave the year in the energy plus file unaltered.
            If an int:
                Change the year to the value for all rows.
            If a tup:
                First entry is the begin year and second is the hour
                at which to change to the next year.
                
            This is useful to give TMY or other multi-year compilations a single
            year within a scenario history. 
        check_types : bool : optional : Default = True
            If True: 
                Check types in all functions (slower but safer).
            If False:
                Do not check types (likely to get less understandable errors
                if something goes wrong).
        clear_alterations : bool : optional : Default = False
            Remove previous alterations from the current 
            object. Reissuing a read command to an Alter object allows a new set
            of weather to recieve the same alterations a previous set recieved.
        isdoe2 : bool : optional 
            Read in a DOE2 weather file instead of an EPW and then
            wrangle that data into an EP format for the self.epwobj database.
        use_exe : bool : optional
            If True: 
                Use BIN2TXT.EXE to read DOE-2 BIN file.
            If False:
                Use Python to read DOE-2 BIN (PREFERRED).
        doe2_bin2txt_path : str : optional : 
            A valid path to an executable
            that converts a DOE2 filename.BIN weather file to an ASCII text file. 
            The executable is assumed to be named BIN2TXT.EXE and comes with
            any distribution of DOE2 which can be obtained by forming a license
            agreement with James Hirsch and Associates (www.doe2.com). A folder
            in mews "third_party_software" can be used to put the filename.EXE.
        doe2_start_datetime : datetime : optional 
            Input the start time for the weather file. If not entered, then the 
            value in the BIN file is used. 
        doe2_hour_in_file : optional : REQUIRED IF isdoe2=True
            Must be 8760 or 8784 for leap years. This allows a non-leap year to 
            be forced into a leap year for consistency. Feb28th is just repeated 
            for such cases.
        doe2_timezone : str: optional : REQUIRED IF isdoe2=True
            Name of time zone applicable to the doe2 file.
        doe2_dst : optional : REQUIRED IF isdoe2=True
            List/tuple with 2 entries that are datetimes with begin and end 
            times for day light savings time.
        
        Returns
        -------
        None 
        """
        
        epwobj = epw()
        self.epwobj = epwobj
        self.isdoe2 = isdoe2
        
        if isdoe2:
            if doe2_bin2txt_path == r"../third_party_software/BIN2TXT.EXE":
                doe2_bin2txt_path = os.path.join(os.path.dirname(__file__),doe2_bin2txt_path)
            
            
            self._doe2_check_types(check_types,weather_file_path,doe2_start_datetime, doe2_hour_in_file,
                        doe2_bin2txt_path,doe2_tz,doe2_dst,use_exe)
            
            df = self.bin2df(weather_file_path,doe2_start_datetime, doe2_hour_in_file,
                        doe2_bin2txt_path,doe2_tz,doe2_dst,use_exe)
            
            # add Year column which is expected by the routine
            df["Year"] = int(replace_year)
            df["Month"] = df["MONTH (1-12)"].astype(int)
            df["Day"] = df["DAY OF MONTH"].astype(int)
            df["Hour"] = df["HOUR OF DAY"].astype(int)
            df["Date"] = df.index
            df.index = pd.RangeIndex(start=0,stop=len(df.index))            
            # this must be done now as well as later to keep the code 
            # consistent.
            epwobj.dataframe = df
        else:
            try:
                epwobj.read(weather_file_path)
            except UnicodeDecodeError:
                raise EPWFileReadFailure("The file '{0}' was not read successfully "
                                         +"by the epw package it is corrupt or "
                                         +"the wrong format!".format(weather_file_path))
            except FileNotFoundError as fileerror:
                raise fileerror
            except:
                raise EPWFileReadFailure("The file '{0}' was not read successfully "
                                         +"by the epw package for an unknown "
                                         +"reason!".format(weather_file_path))        
            df = epwobj.dataframe
        
        
        # verify no NaN's
        if df.isna().sum().sum() != 0:
            raise EPWMissingDataFromFile("NaN's are present after reading the "
                                         +"weather file. Only fully populated "
                                         +"data sets are allowed!")

        self._init_check_inputs(replace_year, check_types)
        
        if not replace_year is None:
            # Prepare for leap year alterations
            new_year_ind = self._get_date_ind(month=1,day=1,hour=1)
            
            if len(new_year_ind) == 0:
                df["Year"] = replace_year
                df = self._leap_year_replacements(df, replace_year, isdoe2)
            else:
                # ADD A START POINT FOR THE REPLACE YEAR IF THE FILE DOES NOT BEGIN WITH 
                # JAN 1ST
                if new_year_ind[0] != 0:
                    new_year_ind = new_year_ind.insert(0,0)
                # loop over years.    
                for idx, ind in enumerate(new_year_ind):
                    if idx < len(new_year_ind) - 1:
                        df.loc[ind:new_year_ind[idx+1],"Year"] = replace_year + idx
                    else:
                        df.loc[ind:,"Year"] = replace_year + idx
                
                # This has to be done separately or the new_year_ind will be 
                # knocked out of place.
                for idx, ind in enumerate(new_year_ind):
                    df = self._leap_year_replacements(df, replace_year + idx, isdoe2)
        
        epwobj.dataframe = df
        epwobj.original_dataframe = deepcopy(df)

        self.check_types = check_types
        self.wfp = weather_file_path
        if (hasattr(self,'alterations') and clear_alterations) or not hasattr(self,'alterations'):
            self.alterations = {} # provides a registry of alterations 
            #TODO -bring WNTR registry class into this tool.
        else:
            for name,alteration in self.alterations.items():
                for col in alteration.columns:
                    common_ind = alteration.index.intersection(df.index)
                    if len(common_ind) == 0:
                        UserWarning("None of the alteration '{0}' intersects" 
                                   +" with the newly read-in epw file!")
                    else:
                        if len(common_ind) != len(alteration.index):
                            UserWarning("Only some of the alteration '{0}' intersects"
                                      + "with the newly read-in epw file!")
                        df.loc[common_ind,col] = (df.loc[common_ind,col] + 
                                                  alteration.loc[common_ind,col])
        first_hour_val = np.unique(self.epwobj.dataframe["Hour"].values)[0]
        if first_hour_val == 1 or first_hour_val == 0:
            self.base1 = bool(first_hour_val)
        else:
            # in the rare case that a file does not even have 24 hours assume
            # that it is base 1 (i.e. 1..24 rather than 0..23)
            self.base1 = True
            
    def _check_string_path(self,string_path):
        if isinstance(string_path,str):
            if not os.path.exists(string_path):
                raise FileNotFoundError("The path "+string_path+ 
                                        " does not exist!")
        else:
            raise TypeError("The input 'weather_file_path' must be a string!")

    def _doe2_check_types(self,check_types,weather_file_path,doe2_start_datetime, 
                          doe2_hour_in_file,doe2_bin2txt_path,
                          doe2_tz,doe2_dst,use_exe):
        
        if check_types:
            self._check_string_path(weather_file_path)
            
            if isinstance(weather_file_path,str):
                if not os.path.exists(weather_file_path):
                    raise FileNotFoundError("The path "+weather_file_path+ 
                                            " does not exist!")
            else:
                raise TypeError("The input 'weather_file_path' must be a string!")
            
            
            if use_exe:
                self._check_string_path(doe2_bin2txt_path)
                if not isinstance(doe2_start_datetime,datetime):
                    raise TypeError("The input 'doe2_start_datetime' must be a datetime object!")
                
                if doe2_hour_in_file != 8760 and doe2_hour_in_file != 8784:
                    raise ValueError("The input 'doe2_hour_in_file' must be an "+
                                     "integer of value 8760 for normal years or"+
                                     " 8784 for leap years")
                    
                if not isinstance(doe2_tz,str):
                    raise TypeError("The input 'doe2_tz' must be a string!")
                    
                if not isinstance(doe2_dst,(list,tuple)):
                    raise TypeError("The input 'doe2_dst' must be a list or tuple of 2-elements")
                if len(doe2_dst) != 2:
                    raise ValueError("The input 'doe2_dst' must have 2-elements")
                if not isinstance(doe2_dst[0],datetime) or not isinstance(doe2_dst[1],datetime):
                    raise TypeError("The input 'doe2_dst' must have 2-elements that are datetime objects!")
            
            
    def remove_alteration(self,alteration_name):
        """
        Removes an alteration that has already been added.
        
        >>> obj.remove_alteration(alteration_name)
        
        Parameters
        ----------
        alteration_name : str
            A name that must exist in the obj.alterations already added.

        Returns
        -------
        None

        """
        if alteration_name in self.alterations:
            df = self.epwobj.dataframe
            addseg = self.alterations.pop(alteration_name) # pop returns and removes
            column = addseg.columns[0]
            bind = addseg.index[0]
            eind = addseg.index[-1]
            df.loc[bind:eind,column] = df.loc[bind:eind,column] - addseg.loc[bind:eind,column]

        else:
            raise ValueError("The alteration {0} does not exist. Valid"
                             +" alterations names are:\n\n{1}"
                             .format(alteration_name,str(self.alterations)))
        
    def reindex_2_datetime(self,tzname=None,original=False):
        """
        >>> obj.reindex_2_datetime(tzname=None)
        
        Parameters
        ----------
        tzname : str : optional : Default = None
            A valid time zone name. When None, the data is kept in the native
            time zone.
        
        Returns
        -------
        df_out
            Dataframe with a DatetimeIndex index and weather data.
        
        """
        
        if original:
            df = self.epwobj.original_dataframe
        else:
            df = self.epwobj.dataframe
        
        begin_end_times = [datetime(df["Year"].values[ind], df["Month"].values[ind],
                                   df["Day"].values[ind], df["Hour"].values[ind]-1) for ind in [0,-1]]
        df_out = deepcopy(df)

        datind = pd.date_range(begin_end_times[0],begin_end_times[1], freq='H',tz=tzname)
        df_diff = len(df_out) - len(datind)
        if df_diff == 1: # a lost hour from daylight savings time must be removed from the end
            #TODO - actually find dates of Daylight savings time and verify this is always
            #       what is happening so that this does not create a way for other bugs
            #       to pass through.
            df_out.drop(df_out.index[-1],inplace=True)     
        elif df_diff != 0:
            raise Exception("There is a datetime error that is unknown!")

        df_out.index = datind

        return df_out
    
    def _write_prep(self,out_file_name=None,overwrite=False,create_dir=False):
        if out_file_name is None:
            out_file_name = self.wfp + "_Altered"
        base_dir = os.path.dirname(out_file_name)
        if len(base_dir) == 0:
            base_dir = "."
        # deal with folder/file existence issues
        if os.path.exists(base_dir):
            if os.path.exists(out_file_name):
                if overwrite:
                    os.remove(out_file_name)
                else:
                    raise FileExistsError("The file '{0}' already ".format(out_file_name)
                                          +"exists!")
        else:
            if create_dir:
                os.makedirs(base_dir)
            else:
                raise NotADirectoryError("The folder '{0}' does not".format(base_dir)
                                         +" exist")
    
    def write(self,out_file_name=None,overwrite=False,create_dir=False,
              use_exe=False,txt2bin_exepath=r"../third_party_software/TXT2BIN.EXE"):
        """
        Writes to an Energy Plus Weather file with alterations.
        
        >>> obj.write(out_file_name=None,
                      overwrite=False,
                      create_dir=False,
                      use_exe=False,
                      txt2bin_exepath=r"../third_party_software/TXT2BIN.EXE")

        Parameters
        ----------
        out_file_name : str : optional : Default = None
            File name and path to output altered epw weather to. 
            If None, then use the original file name with "_Altered" appended to it.
        overwrite : bool : optional : Default = False
            If True, overwrite an existing file otherwise throw an error if the 
            file exists.
        create_dir : bool : optional : Default = False
            If True create a new directory, otherwise throw an error if the 
            folder does not exsit.
        txt2bin_exepath : str : optional
            ...

        Returns
        -------
        None

        """
        if out_file_name is None:
            out_file_name = self.wfp + "_Altered"
        base_dir = os.path.dirname(out_file_name)
        if len(base_dir) == 0:
            base_dir = "."
        # deal with folder/file existence issues
        if os.path.exists(base_dir):
            if os.path.exists(out_file_name):
                if overwrite:
                    os.remove(out_file_name)
                else:
                    raise FileExistsError("The file '{0}' already ".format(out_file_name)
                                          +"exists!")
        else:
            if create_dir:
                os.makedirs(base_dir)
            else:
                raise NotADirectoryError("The folder '{0}' does not".format(base_dir)
                                         +" exist")
        
        if self.isdoe2:
            if txt2bin_exepath==r"../third_party_software/TXT2BIN.EXE":
                txt2bin_exepath = os.path.join(os.path.dirname(__file__),txt2bin_exepath)
            if use_exe:
                self._check_string_path(txt2bin_exepath)
            start_datetime = self.epwobj.dataframe["Date"].iloc[0]
            hour_in_file = len(self.epwobj.dataframe)
            
            self.df2bin(self.epwobj.dataframe, out_file_name, use_exe, start_datetime, 
                   hour_in_file, txt2bin_exepath)
        else:
            self.epwobj.write(out_file_name)
        
