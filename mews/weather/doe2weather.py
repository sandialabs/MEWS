# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:26:52 2021

@author: dlvilla


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

METHODS   mews.weather.DOE2Weather.read_doe2_bin and  
          mews.weather.DOE2Weather.write_doe2_bin
 
   Are translations of parts of BIN2TXT.F and TXT2BIN.F translated with permission
   from James and Jeff Hirsch and Associates (JJH&A). The license for these utilities 
   must be formed with (JJH&A) before they are distributed in any other package
   besides MEWS


"""

from numpy import zeros,cumsum,arange, int64, float64,array
import numpy as np
import pandas as pd
from subprocess import Popen, PIPE
from os.path import isfile, dirname,basename, join
from os import chdir as cd
from os import remove as rm
from os import getcwd as pwd
import os
from shutil import copy as cp
from pandas import DataFrame, DatetimeIndex, DateOffset, Series
from datetime import datetime, timedelta
import warnings
import logging
import struct

class DataFormats():
    # changing this will cause significant portions of the code to fail
    header_dtype = np.dtype([('location_IWDID','a20'),
                                            ('year_IWYR', 'i4'),
                                            ('latitude_WLAT','f4'),
                                            ('longitude_WLONG','f4'),
                                            ('timezone_IWTZN','i4'),
                                            ('record_length_LRECX','i4'),
                                            ('number_days_NUMDAY','i4'),
                                            ('clearness_number_CLN_IM1','f4'),
                                            ('ground_temperature_GT_IM1','f4'),
                                            ('solar_flag_IWSOL','i4')])
    column_description = ['MONTH (1-12)',
        'DAY OF MONTH',
        'HOUR OF DAY',
        'WET BULB TEMP (DEG F)',
        'DRY BULB TEMP (DEG F)',
        'PRESSURE (INCHES OF HG)',
        'CLOUD AMOUNT (0 - 10)',
        'SNOW FLAG (1 = SNOWFALL)',
        'RAIN FLAG (1 = RAINFALL)',
        'WIND DIRECTION (0 - 15; 0=N, 1=NNE, ETC)',
        'HUMIDITY RATIO (LB H2O/LB AIR)',
        'DENSITY OF AIR (LB/CU FT)',
        'SPECIFIC ENTHALPY (BTU/LB)',
        'TOTAL HOR. SOLAR (BTU/HR-SQFT)',
        'DIR. NORMAL SOLAR (BTU/HR-SQFT)',
        'CLOUD TYPE (0 - 2)',
        'WIND SPEED (KNOTS)']

class DOE2_Weather_Error(Exception):
    def __init__(self,error_message):
        self.error_message = error_message
        print(error_message)

class DOE2Weather(object):
    
    def __init__(self):
        self.column_description = DataFormats.column_description
    
    def _rm_file(self,filenamepath):
        if isfile(filenamepath):
            try:
                rm(filenamepath)
            except:
                raise DOE2_Weather_Error("The operating system will not allow python to remove the " +
                            filenamepath + " file!")

    def bin2txt(self,binfilename,bin2txtpath):
        txtfilename = ""
        if isfile(bin2txtpath) and isfile(binfilename):
            curdir = pwd()
            try:
                cd(dirname(bin2txtpath))
                self._rm_file("WEATHER.FMT")
                self._rm_file("WEATHER.BIN")
                if not os.path.isabs(binfilename):
                    binfilename2 = os.path.join(curdir,binfilename)
                else:
                    binfilename2 = binfilename
                cp(binfilename2,"WEATHER.BIN")
                    
                # no arguments needed
                pp = Popen(basename(bin2txtpath),stdout=PIPE, stderr=PIPE, shell=True)
                output, errors = pp.communicate()
                if not errors == b'':
                    warnings.warn("WARNING! An error was recorded by Popen.communicate but this does not mean the BIN2TXT did not work. Investigate further to verify it worked.")
                txtfilename = join(dirname(binfilename2) , basename(binfilename2).split(".")[0] + ".txt")
                cp("WEATHER.FMT",txtfilename)
                cd(curdir)
            except:
                # return to the correct directory
                try:
                    cd(curdir)
                    raise DOE2_Weather_Error("The bin to text process failed!")
                except:
                    raise DOE2_Weather_Error("The OS will not allow return to the original directory. " + curdir)
        else:
            if not isfile(bin2txtpath):
                raise DOE2_Weather_Error("doe2bin2txt.conver_bin_to_txt: the requested bin2txtpath" + 
                        " executable does not exist! A valid path to the BIN2TXT.EXE" +
                        " utility must be provided.")
            else:
                raise DOE2_Weather_Error("doe2bin2txt.conver_bin_to_txt: the requested binfilename" + 
                        " does not exist! A valid path to a valid DOE2 weather binary" +
                        " file must be provided.")
        return txtfilename
    
    
    def txt2bin(self,txtfilename,txt2binpath,binfilename):
        if isfile(txt2binpath) and isfile(txtfilename):
            curdir = pwd()
            try:
                change_dir = len(dirname(txt2binpath)) != 0
                if change_dir:
                    cd(dirname(txt2binpath))
                self._rm_file("WEATHER.BIN")
                if txtfilename != "WEATHER.FMT":
                   cp(txtfilename,"WEATHER.FMT")
                # no arguments needed
                p = Popen(basename(txt2binpath),stdout=PIPE, stderr=PIPE, shell=True)
                output, errors = p.communicate()
                if not errors == b'':
                    warnings.warn("The process produced an error make sure the process worked!\n\n" + str(errors) + "\n\n" + str(output))
                cp("WEATHER.BIN",binfilename)
                self._rm_file("WEATHER.BIN")
                cd(curdir)
            except:
                # return to the correct directory
                try:
                    cd(curdir)
                except:
                    raise DOE2_Weather_Error("The OS will not allow return to the original directory.\n\n " + curdir)
        else:
            if not isfile(txt2binpath):
                raise DOE2_Weather_Error("doe2bin2txt.txt2bin: the requested bin2txtpath" + 
                        " executable does not exist! A valid path to the TXT2BIN.EXE" +
                        " utility must be provided.")
            else:
                raise DOE2_Weather_Error("doe2bin2txt.txt2bin: the requested txtfilename" + 
                        " does not exist! A valid path to a valid DOE2 weather textfile" +
                        " file must be provided.")
    
    
    def df2bin(self,df, binfilename, use_exe=False, start_datetime=None, hour_in_file=None, txt2bin_exepath=None,
               location=None,fyear=None,latitude=None,longitude=None,timezone=None,
               iwsz=2,iftyp=3,clearness_number=None,ground_temp=None):
    
        """
        df2bin(df, binfilename, start_datetime, hour_in_file, txt2bin_exepath,
               location=None,fyear=None,latitude=None,longitude=None,timezone=None,
               iwsz=2,iftyp=3,clearness_number=None,ground_temp=None)
        
        Parameters
        ----------
            df              : pd.Dataframe: must contain the columns originally read from a
                                   DOE2 BIN weather file format (in the table below)
                                   if use_exe = False, df must have a 
            binfilename     : str : path and filename that will be output with the
                                   weather signals in df.
            use_exe         : bool : optional : Default=False
                                    Set to True if using TXT2BIN is desired 
                                    instead of python. Changing use_exe between
                                    reads and writes is not allowed
            
            All other parameters only apply if use_exe = True                         
                                    
            start_datetime  : datetime :start date and time to be output in the weather file
                                    typically this is Jan 1st, fyear
            hour_in_file    : int : Either 8760 or 8784
            txt2bin_exepath : str : path and filename that point to TXT2BIN.EXE
                                    DOE2 utility that can be obtained from www.doe2.com
                                    after forming a license agreement with James Hirsch
                                    and associates.
            location .. and all other inputs
            
            
        Returns
        =======
        
        None
        
        
        %% From TXT2BIN.FMT - this gives the exact format needed to output a
        %                     text file that TXT2BIN.EXE can process.
        %       THIS DOCUMENTS A FORMATTED WEATHER FILE (WEATHER.FMT) MADE FROM
        %       A PACKED BINARY DOE2 WEATHER FILE (WEATHER.BIN) USING WTHFMT2.EXE
        %       AND THE EXTRA FILE NEEDED (not really!) TO PACK IT WITH FMTWTH2.EXE
        %  
        % on input.dat:              
        % 
        % Record 1       IWSZ,IFTYP
        %                FORMAT(12X,I1,17X,I1)     
        % 
        %       IWSZ           WORD SIZE          1 = 60-BIT, 2 = 30-BIT
        %       IFTYP          FILE TYPE          1 = OLD, 2 = NORMAL (NO SOLAR),
        %                                         3 = THE DATA HAS SOLAR
        % on weather.fmt:              
        % 
        % Record 1       (IWDID(I),I=1,5),IWYR,WLAT,WLONG,IWTZN,IWSOL
        %                FORMAT(5A4,I5,2F8.2,2I5)     
        % 
        % Record 2       (CLN(I),I=1,12)
        %                FORMAT(12F6.2)      
        % 
        % Record 3       (GT(I),I=1,12)
        %                FORMAT(12F6.1)
        % 
        % Records 4,8763
        %                KMON, KDAY, KH, WBT, DBT, PATM, CLDAMT, ISNOW, 
        %                IRAIN, IWNDDR, HUMRAT, DENSTY, ENTHAL, SOLRAD,
        %                DIRSOL, ICLDTY, WNDSPD
        %                FORMAT(3I2,2F5.0,F6.1,F5.0,2I3,I4,F7.4,F6.3,F6.1,2F7.1,I3,F5.0)      
        %       IWDID          LOCATION I.D.
        %       IWYR           YEAR
        %       WLAT           LATITUDE
        %       WLONG          LONGITUDE
        %       IWTZN          TIME ZONE NUMBER
        %       IWSOL          SOLAR FLAG         IWSOL = IWSZ + (IFTYP-1)*2 - 1
        %       CLN            CLEARNESS NO.
        %       GT             GROUND TEMP.       (DEG R)
        %       KMON           MONTH              (1-12)
        %       KDAY           DAY OF MONTH
        %       KH             HOUR OF DAY
        %       WBT            WET BULB TEMP      (DEG F)
        %       DBT            DRY BULB TEMP      (DEG F)
        %       PATM           PRESSURE           (INCHES OF HG)
        %       CLDAMT         CLOUD AMOUNT       (0 - 10)
        %       ISNOW          SNOW FLAG          (1 = SNOWFALL)
        %       IRAIN          RAIN FLAG          (1 = RAINFALL)
        %       IWNDDR         WIND DIRECTION     (0 - 15; 0=N, 1=NNE, ETC)
        %       HUMRAT         HUMIDITY RATIO     (LB H2O/LB AIR)
        %       DENSTY         DENSITY OF AIR     (LB/CU FT)
        %       ENTHAL         SPECIFIC ENTHALPY  (BTU/LB)
        %       SOLRAD         TOTAL HOR. SOLAR   (BTU/HR-SQFT)
        %       DIRSOL         DIR. NORMAL SOLAR  (BTU/HR-SQFT)
        %       ICLDTY         CLOUD TYPE         (0 - 2)
        %       WNDSPD         WIND SPEED         KNOTS"""
        if hasattr(self,'use_exe'):
            if self.use_exe != use_exe:
                raise ValueError("This class does not support switching between"+
                                 " using Python and using the BIN2TXT.F and TXT2BIN.F"+
                                 " executables!")
        else:
            self.use_exe = use_exe
        

        if use_exe:        
            cdir = pwd()
            #try:
            change_dir = len(dirname(binfilename)) != 0 
            if change_dir:
                cd(dirname(binfilename))
            self._rm_file("WEATHER.FMT")
            self._rm_file("WEATHER.BIN")
            with open("INPUT.DAT",'w') as dat:
                dat.write('            {0:1.0f}                 {1:1.0f}'.format(iwsz,iftyp))
            with open("WEATHER.FMT",'w') as fmt:
                # 3 header rows
    # header_dtype = np.dtype([('location_IWDID','a20'),
    #                                         ('year_IWYR', 'i4'),
    #                                         ('latitude_WLAT','f4'),
    #                                         ('longitude_WLONG','f4'),
    #                                         ('timezone_IWTZN','i4'),
    #                                         ('record_length_LRECX','i4'),
    #                                         ('number_days_NUMDAY','i4'),
    #                                         ('clearness_number_CLN_IM1','f4'),
    #                                         ('ground_temperature_GT_IM1','f4'),
    #                                         ('solar_flag_IWSOL','i4')])                
                
                if isinstance(df.headers[0]['location_IWDID'][0],np.bytes_):
                    location_str = df.headers[0]['location_IWDID'][0].decode('ascii')
                else:
                    location_str = df.headers[0]['location_IWDID'][0]
                    
                row1 = '{0:20s}{1:5d}{2:8.2f}{3:8.2f}{4:5d}{5:5d}\n'.format(
                    location_str,
                    df.headers[0]['year_IWYR'][0],
                    df.headers[0]['latitude_WLAT'][0], 
                    df.headers[0]['longitude_WLONG'][0], 
                    df.headers[0]['timezone_IWTZN'][0],
                    df.headers[0]['solar_flag_IWSOL'][0])
                fmt.write(row1)
                clearness_number = [head['clearness_number_CLN_IM1'][0] for head in df.headers]
                ground_temp = [head['ground_temperature_GT_IM1'][0] for head in df.headers]
                
                fmt.write((12*'{:6.2f}'+'\n').format(*clearness_number))
                fmt.write((12*'{:6.2f}'+'\n').format(*ground_temp))
                
                for index, row in df.iterrows():
                    fmt.write((3*'{:2.0f}'+2*'{:5.0f}' + '{:6.1f}{:5.0f}' + 2*'{:3.0f}' + 
                     '{:4.0f}{:7.4f}{:6.3f}{:6.1f}' + 2*'{:7.1f}'+'{:3.0f}{:5.0f}\n').format(
                             *row.tolist()))
            if isfile(txt2bin_exepath):
                if change_dir:
                    cp(txt2bin_exepath,".")
                else:
                    cp(txt2bin_exepath,".")
                new_exe_path = join(".",basename(txt2bin_exepath))
                self.txt2bin("WEATHER.FMT",new_exe_path,os.path.basename(binfilename))
            else:
                cd(cdir)
                raise DOE2_Weather_Error("The txt2bin.exe utility is not present at: \n\n" + txt2bin_exepath)        
            cd(cdir)    
        
        else:
            # size and type checking
            m = df[self.column_description].values
            
            if m.shape[1] != 17:
                raise ValueError("This function only handles dataframes with"+
                                 " 17 columns as defined for DOE-2 BIN weather files!")
            elif not hasattr(df,'headers'):
                raise ValueError("The input dataframe df must have an attribute "+
                                 "'headers' that contains a list of ")
            headers = df.headers
            DOE2Weather.write_doe2_bin(m, headers, binfilename)


    def bin2df(self,binfilename, start_datetime=None, hour_in_file=None, bin2txt_exepath=None, timezone=None, dst=None, use_exe=False):
        """ This function was originally written in matlab and is the 
        "ReadDOE2BINTXTFile.m" function except that it also includes conversion
        of the BIN file into a text file  
        
        DST is a list of the start of daylight savings and end so that
        adjustments can be made and the time stamps adjusted for daylight savings.
    % the input *.txt "filename" must come from a DOE2 bin file that has been
    %  converted to a text file. It has the following columns of information:
    % Column Number   Variable      Description         Units
    %C 1              IM2            MOMTH              (1-12)
    %C 2             ID             DAY OF MONTH
    %C 3             IH             HOUR OF DAY
    %C 4             CALC(1)        WET BULB TEMP      (DEG F)
    %C 5             CALC(2)        DRY BULB TEMP      (DEG F)
    %C 6             CALC(3)        PRESSURE           (INCHES OF HG)
    %C 7             CALC(4)        CLOUD AMOUNT       (0 - 10)
    %C 8             ISNOW          SNOW FLAG          (1 = SNOWFALL)
    %C 9             IRAIN          RAIN FLAG          (1 = RAINFALL)
    %C 10            IWNDDR         WIND DIRECTION     (0 - 15; 0=N, 1=NNE, ETC)
    %C 11            CALC(8)        HUMIDITY RATIO     (LB H2O/LB AIR)
    %C 12            CALC(9)        DENSITY OF AIR     (LB/CU FT)
    %C 13            CALC(10)       SPECIFIC ENTHALPY  (BTU/LB)
    %C 14            CALC(11)       TOTAL HOR. SOLAR   (BTU/HR-SQFT)
    %C 15            CALC(12)       DIR. NORMAL SOLAR  (BTU/HR-SQFT)
    %C 16            ICLDTY         CLOUD TYPE         (0 - 2)
    %C 17            CALC(14)       WIND SPEED         KNOTS    
        
        
        """
        if (timezone is None and not dst is None) or (not timezone is None and dst is None):
            raise ValueError("The timezone and dst values must be specified together!")
        
        
        hour_in_day = 24
        if hasattr(self,'use_exe'):
            if use_exe != self.use_exe:
                raise ValueError("This class does not support switching between" +
                                 " using BIN2TXT.EXE and TXT2BIN.EXE and using" +
                                 " Python for translation! None returned as a result!")
        else:
            # set the mode of operation of the class
            self.use_exe = use_exe
        
        if use_exe:
            # this is the old way of doing things.
            txtname = self.bin2txt(binfilename, bin2txt_exepath)
            
            if len(txtname)==0:
                raise DOE2_Weather_Error("bom2df:The bin file was not successfully converted please troubleshoot!")

            num = 0
            m = zeros((hour_in_file,17))
            # this is specific to the conversion utility and how it writes out ASCII.
            EntryLength = [0, 2, 2, 2, 5, 5, 6, 5, 3, 3, 4, 7, 6, 6, 7, 7, 3, 5]
            j = 0
            i = 0
            b_lines = []
            
            with open(txtname,'r') as h:
                for text_line in h:
                    if num <= 2:
                        num += 1 # skip three lines
                        b_lines.append(text_line)
                    else:  
                        for mm,nn in zip(cumsum(EntryLength[0:-1]),cumsum(EntryLength[1:])):
                            m[j,i] = float(text_line[mm:nn])
                            i+=1
                        j +=1
                        i = 0
        else:
            # use pure python to do this 
            m, headers = self.read_doe2_bin(binfilename)
        
        # adjust for leap year by repeating February 28th on February 29th.
        if hour_in_file == 8784:
            # February 28th is the 59th day of the year
            # Febrary 29th is the 60th day of a year
            sid = 59 * 24
            
            # shift all of March1st to December 31 over 24 hours
            m[sid+24:] = m[sid:-24]
            # repeat February 28th 
            m[sid:sid+24] = m[sid-24:sid]
            # reassign Feb 28th to Feb 29th - 
            # day of month column
            ind = self.column_description.index("DAY OF MONTH")
            m[sid:sid+24,ind] = 29
            MDAYS = [31,29,31,30,31,30,31,31,30,31,30,31] 
        else:
            MDAYS = [31,28,31,30,31,30,31,31,30,31,30,31]
        
        
        
        dateVec = []
        reached_hours_to_next = True
        start_datetime_was_None = False
        get_year_from_headers = False
        
        month = 0  # 0 = Jan
        for i in arange(m.shape[0]):
            
            # see whether a replacement year has been provided - if not, use what
            # is in the BIN file - if Python is used, the the header of each
            # month can have a different year if it is TMY3
            if start_datetime is None and use_exe:
                start_datetime = datetime(year=int(b_lines[0][20:25]),day=1,month=1)
                start_datetime_was_None = True
            elif start_datetime is None and reached_hours_to_next and not use_exe:
                hour_count = 0
                hours_to_next = MDAYS[month]* hour_in_day

                year = headers[month]['year_IWYR'][0]
                month += 1
                start_datetime_was_None = True
                get_year_from_headers = True
            elif not use_exe and start_datetime is None:
                hour_count += 1

            
            if get_year_from_headers:
                if hour_count > hours_to_next:    
                    reached_hours_to_next = True
                else:
                    reached_hours_to_next = False

            
            # for cases with a replacement year OR use_exe where the text file 
            # does not convey the year at every header
            if use_exe or not start_datetime_was_None:

                current_time = start_datetime+timedelta(hours=float(i))
                dateVec.append(datetime(current_time.year,current_time.month,
                                        current_time.day,current_time.hour,0,0))
            # 
            else:
                dateVec.append(datetime(year,int(m[i,0]),int(m[i,1]),int(m[i,2])-1,0,0))
            
            if not dst is None:
                #Handle daylight savings correctly
                if dateVec[-1] == dst[0]:
                    dateVec[-1] = dateVec[-1] + DateOffset(hour=1)
                elif dateVec[-1] == dst[1]:
                    dateVec[-1] = dateVec[-1] - DateOffset(hour=1)
            
            
                
        dateTimeIn = DatetimeIndex(dateVec)
        if not timezone is None:
            try:
                dateTimeIn = dateTimeIn.tz_localize(timezone, ambiguous=True)
            except Exception as e:
                warnings.warn("The time zone localization process failed. " 
                              + " This probably happened because the time zone " 
                              + "input is incorrect!")
                raise e

        df = DataFrame(index=dateTimeIn,data=m,columns=self.column_description,dtype=float64)
    
        # add header information (different depending on use_exe)
        
        # suppress warnings here. - We want to add attributes to the dataframe
        # and are NOT trying to add columns
        warnings.simplefilter("ignore",category=UserWarning)

        if use_exe:
            clearness_number = [float(x) for x in b_lines[1].split(" ") if len(x)!=0]
            ground_temps = [float(x) for x in b_lines[2].split(" ") if len(x)!=0]
            headers = []
            month = 0
            
            for num,temp in zip(clearness_number,ground_temps):
                header_list = [(b_lines[0][0:20],
                                     np.int32(b_lines[0][20:25]),
                                     np.float32(b_lines[0][25:33]),
                                     np.float32(b_lines[0][33:41]),
                                     np.int32(b_lines[0][41:46]),
                                     np.int32(1),
                                     np.int32(MDAYS[month]),
                                     np.float32(clearness_number[month]),
                                     np.float32(ground_temps[month]),
                                     np.int32(b_lines[0][46:51]))]
                lheaders = np.array(header_list,dtype=DataFormats.header_dtype)
 
                headers.append(lheaders)


        df.headers = headers
            
        return df
    
    @staticmethod
    def read_doe2_bin(binfilename):
        with open(binfilename,mode='rb') as bh:
            bin_content = bh.read()
        """
        obj.read_doe2_bin(binfilename)
        
        Parameters
        ==========
        
        binfilename : str : valid path and file name to a DOE-2 *.BIN weather 
                            file.
        
        Returns
        =======
        
        m : np.array(x,17) : x = length of BIN file (ussually 8760)
            The columns of this array are:
                        
            % Column Number   Variable      Description         Units
            %C 1              IM2            MOMTH              (1-12)
            %C 2             ID             DAY OF MONTH
            %C 3             IH             HOUR OF DAY
            %C 4             CALC(1)        WET BULB TEMP      (DEG F)
            %C 5             CALC(2)        DRY BULB TEMP      (DEG F)
            %C 6             CALC(3)        PRESSURE           (INCHES OF HG)
            %C 7             CALC(4)        CLOUD AMOUNT       (0 - 10)
            %C 8             ISNOW          SNOW FLAG          (1 = SNOWFALL)
            %C 9             IRAIN          RAIN FLAG          (1 = RAINFALL)
            %C 10            IWNDDR         WIND DIRECTION     (0 - 15; 0=N, 1=NNE, ETC)
            %C 11            CALC(8)        HUMIDITY RATIO     (LB H2O/LB AIR)
            %C 12            CALC(9)        DENSITY OF AIR     (LB/CU FT)
            %C 13            CALC(10)       SPECIFIC ENTHALPY  (BTU/LB)
            %C 14            CALC(11)       TOTAL HOR. SOLAR   (BTU/HR-SQFT)
            %C 15            CALC(12)       DIR. NORMAL SOLAR  (BTU/HR-SQFT)
            %C 16            ICLDTY         CLOUD TYPE         (0 - 2)
            %C 17            CALC(14)       WIND SPEED         KNOTS    
            
        headers : list : list of np.array with specialized dtype to capture
            the header for each month of data in the DOE-2 *.BIN file.
            The dtype spec is as follows:
                np.dtype([('location_IWDID','a20'),
                          ('year_IWYR', 'i4'),
                          ('latitude_WLAT','f4'),
                          ('longitude_WLONG','f4'),
                          ('timezone_IWTZN','i4'),
                          ('record_length_LRECX','i4'),
                          ('number_days_NUMDAY','i4'),
                          ('clearness_number_CLN_IM1','f4'),
                          ('ground_temperature_GT_IM1','f4'),
                          ('solar_flag_IWSOL','i4')])
        
        License Note:
        =============
        
        # THIS IS A TRANSLATION OF BIN2TXT.F except the TXT file is never written
        # because data for direct use in Python is desired.
        
        # THIS reverse engineering was approved by Jeff Hirsch on behalf of
        # James and Jeff Hirsch and Associates (JJH&A)
        # along with putting the code on GITHUB with the contingency that the
        # JJH&A license be acknowledged. The original correspondence is provided 
        # below:
        
        # Wed 7/7/2021 4:10 PM
        Yes, you have my permission to distribute your DOE-2 weather file 
        python libraries with TXT2BIN and BIN2TXT executables on Github or 
        also translate the fortran source versions of those apps into python 
        and then distribute on Github as long as your acknowledge the 
        JJH&A license
 
        ________________________________________
        Jeff Hirsch
        James J. Hirsch & Associates
        Voice mail: (XXX) XXX-XXXX
        mobile: (XXX) XXX-XXXX
        -----------------------------------------------------
        From: Villa, Daniel L 
        Sent: Wednesday, July 7, 2021 11:02 AM
        To: Jeff.Hirsch@DOE2.com 
        Subject: Tranlate TXT2BIN.F and BIN2TXT.F into Python and distribute 
                 as open source??
         
        Jeff,
         
        I have built a tool that inputs and outputs DOE-2 weather files with 
        extreme events. I use the TXT2BIN.F and BIN2TXT.F executables with 
        the version of DOE-2 that I have licensed with Hirsch and Associates. 
        I would like to be able to distribute the python libraries with these 
        executables on Github or else be able to translate the *.F files into 
        python but know that the code is distributed under your license. 
         
        Would Hirsch and Associates be willing to let me create Python 
        versions of TXT2BIN.F and BIN2TXT.F and to distribute them as open 
        source code on GitHUB? I understand and respect Hirsch and Associate’s 
        decision if this is not allowed. Thank you.
         
        Daniel Villa
        Energy-Water System Integration Department 08825
        Sandia National Laboratories
        dlvilla@sandia.gov
        XXX-XXX-XXXX

        """
        
        # initiate constants in BIN2TXT
        MDAYS = [31,28,31,30,31,30,31,31,30,31,30,31]
        XMASK_1D = np.array([-99., -99., 15., 0., 0., 0., 0., 0., .02, -30., 0.,
                          .0, .0, .0, .0, 10., 1., 1., .1, 1., 1., 1., 1.,
                          .0001, .001, .5, 1., 1., 1., 1., 0., 0.])
        XMASK = XMASK_1D.reshape((2,16)).T
        
        # record length is 6200 and Fortran puts a 4 byte buffer at the beginning
        # and end of the record. making each record a total of 6208 bytes.
        #
        recl = 6200 
        num_buf_bytes = 4
        tot_recl = recl + 2 * num_buf_bytes
        blocksize=148992
        
        header_length = 56
        num_month_in_year = 12
        num_hour_in_day = 24
        headers = []
        
        # read in headers
      #   DO 100 IM1=1,12
      #   READ (10) (IWDID(I),I=1,5),IWYR,WLAT,WLONG,IWTZN,LRECX,NUMDAY,
      # _          CLN(IM1),GT(IM1),IWSOL
      #   READ (10) IDUM
      #   100 CONTINUE
        
        for IM1 in range(num_month_in_year):
            head_start_byte = (recl + 2*num_buf_bytes) * IM1 + num_buf_bytes
            headers.append(np.frombuffer(bin_content[head_start_byte:head_start_byte+header_length+1],
                                         dtype=DataFormats.header_dtype
                                         ,count=1)
                            )
        # Read and process data
        LRECX = 0
        byte_position = num_buf_bytes
        IWTH = np.zeros(15)  # keep 1 indexing and leave the first element 0
        CALC = np.zeros(15)
        data_records = []
        IDAT30 = np.zeros(1537)
        iterIH = 0
        iter_max = 1e6
        for IM2 in range(1,num_month_in_year+1):
            IDE = MDAYS[IM2-1]
            for ID in range(1,IDE+1):
                IH = 1
                while IH <= num_hour_in_day and iterIH < iter_max:
                    IRECX = int(IM2 * 2 + (ID-1)/16 - 1)   #105
                    IDX = int(np.mod(ID-1,16) + 1)
                    comparison = int(IRECX-LRECX)
                    if comparison < 0:
                        IDIF = int(LRECX - IRECX + 1)
                        for I in range(IDIF):
                            if np.mod(byte_position,tot_recl)==0 and byte_position > num_buf_bytes:
                                byte_position = byte_position - tot_recl
                            elif byte_position > num_buf_bytes:
                                byte_position = byte_position - np.mod(byte_position,tot_recl)
                        next_record = byte_position + tot_recl
                        cur_dat = DOE2Weather._doe2_bin_data_format(bin_content,byte_position,next_record)
                        LRECX = cur_dat['record_length_LRECX']
                        byte_position = next_record
                    elif comparison == 0:
                        IDAT30[1:] = cur_dat['IDAT30']
                        IP1 = int(96*(IDX-1) + 4*IH - 3)
                        IWTH[3] = IDAT30[IP1]/65536
                        IWTH[1] = np.mod(IDAT30[IP1],65536)/256
                        IWTH[2] = np.mod(IDAT30[IP1],256)
                        IWTH[11] = IDAT30[IP1+1]/1048576
                        IWTH[12] = np.mod(IDAT30[IP1+1],1048576)/1024
                        IWTH[4] = np.mod(IDAT30[IP1+1],1024)/64
                        IWTH[5] = np.mod(IDAT30[IP1+1],64)/32
                        IWTH[6] = np.mod(IDAT30[IP1+1],32)/16
                        IWTH[7] = np.mod(IDAT30[IP1+1],16)
                        IWTH[8] = IDAT30[IP1+2]/128
                        IWTH[9] = np.mod(IDAT30[IP1+2],128)
                        IWTH[10] = IDAT30[IP1+3]/2048
                        IWTH[13] = np.mod(IDAT30[IP1+3],2048)/128
                        IWTH[14] = np.mod(IDAT30[IP1+3],128)
                        for I in range(1,15):
                            CALC[I] = float(IWTH[I])*XMASK[I-1,2-1] + XMASK[I-1,1-1]
                        
                        
                        ISNOW = int(CALC[5] + .01)
                        IRAIN = int(CALC[6] + .01)
                        IWNDDR = int(CALC[7] + .01)
                        ICLDTY = int(CALC[13] + .01)
                        
                        data_records.append([IM2,ID,IH,CALC[1],CALC[2],CALC[3],
                                             CALC[4],ISNOW,IRAIN,IWNDDR,CALC[8],
                                             CALC[9],CALC[10],CALC[11],CALC[12],
                                             ICLDTY,CALC[14]])
                        IH += 1
               
                    elif comparison > 0:
                        next_record = byte_position + tot_recl
                        cur_dat = DOE2Weather._doe2_bin_data_format(bin_content,byte_position,next_record)
                        LRECX = cur_dat['record_length_LRECX']
                        byte_position = next_record
                        
                    iterIH += 1
                if iterIH >= iter_max:
                    raise StopIteration("The while loop over days has gotten"
                                        +" stuck! The file being read may not"
                                        +" be the correct BIN Format for DOE-2.")
        m = np.array(data_records)
        return m, headers 
                        
    @staticmethod
    def _doe2_bin_data_format(bin_content,byte_position,next_record):
        return np.frombuffer(bin_content[byte_position:next_record+1],
                             np.dtype([('location_IWDID','a20'),
                                            ('year_IWYR', 'i4'),
                                            ('latitude_WLAT','f4'),
                                            ('longitude_WLONG','f4'),
                                            ('timezone_IWTZN','i4'),
                                            ('record_length_LRECX','i4'),
                                            ('number_days_NUMDAY','i4'),
                                            ('clearness_number_CLN_IM1','f4'),
                                            ('ground_temperature_GT_IM1','f4'),
                                            ('solar_flag_IDUM','i4'),
                                            ('IDAT30','1536i4')]),count=1)[0]    
    @staticmethod
    def write_doe2_bin(m,headers,binfilename,IWSZ=1,IFTYP=2):
        """
        DOE2Weather.write_doe2_bin(m,headers,binfilename,IWSZ=1,IFTYP=2)
        
        Parameters
        ==========

        m : np.array(x,17) : x = length of BIN file (ussually 8760)
            The columns of this array are:
                        
            % Column Number   Variable      Description         Units
            %C 1              IM2            MOMTH              (1-12)
            %C 2             ID             DAY OF MONTH
            %C 3             IH             HOUR OF DAY
            %C 4             CALC(1)        WET BULB TEMP      (DEG F)
            %C 5             CALC(2)        DRY BULB TEMP      (DEG F)
            %C 6             CALC(3)        PRESSURE           (INCHES OF HG)
            %C 7             CALC(4)        CLOUD AMOUNT       (0 - 10)
            %C 8             ISNOW          SNOW FLAG          (1 = SNOWFALL)
            %C 9             IRAIN          RAIN FLAG          (1 = RAINFALL)
            %C 10            IWNDDR         WIND DIRECTION     (0 - 15; 0=N, 1=NNE, ETC)
            %C 11            CALC(8)        HUMIDITY RATIO     (LB H2O/LB AIR)
            %C 12            CALC(9)        DENSITY OF AIR     (LB/CU FT)
            %C 13            CALC(10)       SPECIFIC ENTHALPY  (BTU/LB)
            %C 14            CALC(11)       TOTAL HOR. SOLAR   (BTU/HR-SQFT)
            %C 15            CALC(12)       DIR. NORMAL SOLAR  (BTU/HR-SQFT)
            %C 16            ICLDTY         CLOUD TYPE         (0 - 2)
            %C 17            CALC(14)       WIND SPEED         KNOTS    
            
        headers : list : list of np.array with specialized dtype to capture
            the header for each month of data in the DOE-2 *.BIN file.
            The dtype spec is as follows:
                np.dtype([('location_IWDID','a20'),
                          ('year_IWYR', 'i4'),
                          ('latitude_WLAT','f4'),
                          ('longitude_WLONG','f4'),
                          ('timezone_IWTZN','i4'),
                          ('record_length_LRECX','i4'),
                          ('number_days_NUMDAY','i4'),
                          ('clearness_number_CLN_IM1','f4'),
                          ('ground_temperature_GT_IM1','f4'),
                          ('solar_flag_IWSOL','i4')])
                
        binfilename : str : valid path/filename for which a file will be 
            written.
                
        Returns
        =======
        None 
        
        License Note:
        =============
        
        # THIS IS A TRANSLATION OF TXT2BIN.F except the TXT file is never read
        # from because data is coming from Python.
        
        # THIS reverse engineering was approved by Jeff Hirsch on behalf of
        # James and Jeff Hirsch and Associates (JJH&A)
        # along with putting the code on GITHUB with the contingency that the
        # JJH&A license be acknowledged. The original correspondence is provided 
        # below:
        
        # Wed 7/7/2021 4:10 PM
        Yes, you have my permission to distribute your DOE-2 weather file 
        python libraries with TXT2BIN and BIN2TXT executables on Github or 
        also translate the fortran source versions of those apps into python 
        and then distribute on Github as long as your acknowledge the 
        JJH&A license
 
        ________________________________________
        Jeff Hirsch
        James J. Hirsch & Associates
        Voice mail: (XXX) XXX-XXXX
        mobile: (XXX) XXX-XXXX
        -----------------------------------------------------
        From: Villa, Daniel L 
        Sent: Wednesday, July 7, 2021 11:02 AM
        To: Jeff.Hirsch@DOE2.com 
        Subject: Tranlate TXT2BIN.F and BIN2TXT.F into Python and distribute 
                 as open source??
         
        Jeff,
         
        I have built a tool that inputs and outputs DOE-2 weather files with 
        extreme events. I use the TXT2BIN.F and BIN2TXT.F executables with 
        the version of DOE-2 that I have licensed with Hirsch and Associates. 
        I would like to be able to distribute the python libraries with these 
        executables on Github or else be able to translate the *.F files into 
        python but know that the code is distributed under your license. 
         
        Would Hirsch and Associates be willing to let me create Python 
        versions of TXT2BIN.F and BIN2TXT.F and to distribute them as open 
        source code on GitHUB? I understand and respect Hirsch and Associate’s 
        decision if this is not allowed. Thank you.
         
        Daniel Villa
        Energy-Water System Integration Department 08825
        Sandia National Laboratories
        dlvilla@sandia.gov
        XXX-XXX-XXXX
        
        FORTRAN VARIABLE MEANING KEY:
            
            C              
            C     IWSZ           WORD SIZE          1 = 60-BIT, 2 = 30-BIT
            C     IFTYP          FILE TYPE          1 = OLD, 2 = NORMAL (NO SOLAR),
            C                                       3 = THE DATA HAS SOLAR
            C     IWDID          LOCATION I.D.
            C     IWYR           YEAR
            C     WLAT           LATITUDE
            C     WLONG          LONGITUDE
            C     IWTZN          TIME ZONE NUMBER
            C     IWSOL          SOLAR FLAG         FUNCTION OF IWSZ + IFTYP
            C     CLN            CLEARNESS NO.
            C     GT             GROUND TEMP.       (DEG R)
            C     KMON           MONTH              (1-12)
            C     KDAY           DAY OF MONTH
            C     KH             HOUR OF DAY
            C     WBT            WET BULB TEMP      (DEG F)
            C     DBT            DRY BULB TEMP      (DEG F)
            C     PATM           PRESSURE           (INCHES OF HG)
            C     CLDAMT         CLOUD AMOUNT       (0 - 10)
            C     ISNOW          SNOW FLAG          (1 = SNOWFALL)
            C     IRAIN          RAIN FLAG          (1 = RAINFALL)
            C     IWNDDR         WIND DIRECTION     (0 - 15; 0=N, 1=NNE, ETC)
            C     HUMRAT         HUMIDITY RATIO     (LB H2O/LB AIR)
            C     DENSTY         DENSITY OF AIR     (LB/CU FT)
            C     ENTHAL         SPECIFIC ENTHALPY  (BTU/LB)
            C     SOLRAD         TOTAL HOR. SOLAR   (BTU/HR-SQFT)
            C     DIRSOL         DIR. NORMAL SOLAR  (BTU/HR-SQFT)
            C     ICLDTY         CLOUD TYPE         (0 - 2)
            C     WNDSPD         WIND SPEED         KNOTS

        """ 
        # record length is 6200 and Fortran puts a 4 byte buffer at the beginning
        # and end of the record. making each record a total of 6208 bytes.
        #
        recl = 6200 
        num_buf_bytes = 4
        tot_recl = recl + 2 * num_buf_bytes
        num_month_in_year = 12
        num_hour_in_day = 24
        
        CLN = []; GT = []; IWYR = []; WLAT = []; WLONG = []; IWTZN = []; 
        LRECX = []; NUMDAY = []; IWSOL = []; IWDID = [];
        
        # translate to the original syntax for clarity
        for header in headers:
            IWDID.append(header['location_IWDID'][0])
            CLN.append(header["clearness_number_CLN_IM1"][0])
            GT.append(header["ground_temperature_GT_IM1"][0])
            IWYR.append(header["year_IWYR"][0])
            WLAT.append(header["latitude_WLAT"][0])
            WLONG.append(header["longitude_WLONG"][0])
            IWTZN.append(header["timezone_IWTZN"][0])
            LRECX.append(header["record_length_LRECX"][0])  #not needed
            NUMDAY.append(header["number_days_NUMDAY"][0]) # not needed
            IWSOL.append(header["solar_flag_IWSOL"][0])  # not needed recalculated here
        
        MDAYS = np.zeros(num_month_in_year+1)
        MDAYS[1:] = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        IDAT = np.zeros(1536+1,dtype=np.int32)

        # record buffer. TODO- find out the exact 4 bytes FORTRAN puts here!
        IDUM = np.int32(recl)  # Fortran buffers with 4 bytes at the begin and
                            # end of each record indicating the length in bytes 
                            # of the record.
        
        if (IWSZ == 0):
            raise ValueError("IWSZ = 0 does not work in Python because the WEATHER.FMT file does not exist!")
        else:
            IWSOL = np.int32(IWSZ + (IFTYP-1)*2 - 1)
        
        count = 0
        # ARRAYS are zero base! - The fortran is base 1 - watch out!
        with open(binfilename,'wb',buffering=tot_recl) as file:
            for IM in range(1,num_month_in_year+1):
                IDE = np.int32(MDAYS[IM])
                for ID in range(1,IDE+1):
                    
                    IRECXO = np.int32(IM*2 + (ID-1)/16 - 1)
                    IDXO = np.int32(np.mod(ID-1,16) + 1)
                    
                    for IH in range(1,num_hour_in_day+1):

                        KMON = m[count,0]      # Month
                        KDAY = m[count,1]      # Day
                        KH = m[count,2]        # Hour
                        WBT = m[count,3]       # Wet bulb temperature (F)
                        DBT = m[count,4]       # Dry bulb temperature (F)
                        PATM = m[count,5]      # Atmospheric Pressure (in Hg)
                        CLDAMT = m[count,6]    # Cloud amount (0-10)
                        ISNOW = m[count,7]     # Snow flag (1=snowing)
                        IRAIN = m[count,8]     # Rainfall flag (1=precipitation happening)
                        IWNDDR = m[count,9]    # Wind Direction (0 - 15; 0=N, 1=NNE, ETC)
                        HUMRAT = m[count,10]   # Humidity ratio (lb h2o/lb air)
                        DENSTY = m[count,11]   # Air density (lb/ft3)
                        ENTHAL = m[count,12]   # Enthalpy (BTU/LB)
                        SOLRAD = m[count,13]   # Diffuse solar radiation (BTU/HR-SQFT)
                        DIRSOL = m[count,14]   # Direct solar radiation ((BTU/HR-SQFT))
                        ICLDTY = m[count,15]   # Cloud type (0-2)
                        WNDSPD = m[count,16]   # Wind speed (knots)
                        
                        ISOL = np.int32(SOLRAD + .5)
                        IDN = np.int32(DIRSOL + .5)
                        IWET = np.int32(WBT+99.5)
                        IDRY = np.int32(DBT+99.5)
                        IPRES = np.int32(PATM*10.-149.5)
                        ICLDAM = np.int32(CLDAMT)
                        IWNDSP = np.int32(WNDSPD+0.5)
                        IHUMRT = np.int32(HUMRAT*10000.+0.5)
                        IDENS = np.int32(DENSTY*1000.-19.5)
                        IENTH = np.int32(ENTHAL*2.0+60.5)
                        IP1 = np.int32((IDXO-1)*96 + IH*4 - 3)
                        IDAT[IP1] = np.int32(IPRES*65536 + IWET*256 + IDRY)
                        IDAT[IP1+1] = np.int32(ISOL*1048576 + IDN*1024 + 
                                      ICLDAM*64 + ISNOW*32 + IRAIN*16 + IWNDDR) 
                        IDAT[IP1+2] = np.int32(IHUMRT*128 + IDENS)
                        IDAT[IP1+3] = np.int32(IENTH*2048 + ICLDTY*128 + IWNDSP)
                        
                    if ID != 16 and ID != IDE:
                        pass # keep the loop 
                    else:
                        # IDUM is the padding written by Fortran that
                        #   has to be explicitly included here.
                        byte_arr = struct.pack('i',IDUM)
                        byte_arr = byte_arr + struct.pack('20s',IWDID[IM-1])
                        
                        byte_arr = (byte_arr +
                                    struct.pack('i',IWYR[IM-1]) +
                                    struct.pack('f',WLAT[IM-1]) +
                                    struct.pack('f',WLONG[IM-1]) +
                                    struct.pack('i',IWTZN[IM-1]) +
                                    struct.pack('i',IRECXO) +
                                    struct.pack('i',IDE) +
                                    struct.pack('f',CLN[IM-1]) +
                                    struct.pack('f',GT[IM-1]) +
                                    struct.pack('i',IWSOL)) 
                        byte_arr = byte_arr + b"".join([struct.pack('i',i4) for i4 in IDAT[1:]]) 
                        byte_arr = byte_arr + struct.pack('i',IDUM)
                        
                        file.write(byte_arr)
                            
                            
                        count += 1
        return # end of write_doe2_bin     