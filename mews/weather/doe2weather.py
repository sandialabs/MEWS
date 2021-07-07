# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 10:26:52 2021

@author: dlvilla
"""

from numpy import zeros,cumsum,arange, int64, float64,array
from subprocess import Popen, PIPE
from os.path import isfile, dirname,basename, join
from os import chdir as cd
from os import remove as rm
from os import getcwd as pwd
import os
from shutil import copy as cp
from pandas import DataFrame, DatetimeIndex, DateOffset, Series
from datetime import datetime
import warnings
from mews.errors.exceptions import DOE2_Weather_Error


def _rm_file(filenamepath):
    if isfile(filenamepath):
        try:
            rm(filenamepath)
        except:
            raise DOE2_Weather_Error("The operating system will not allow python to remove the " +
                        filenamepath + " file!")

def bin2txt(binfilename,bin2txtpath):
    txtfilename = ""
    if isfile(bin2txtpath) and isfile(binfilename):
        curdir = pwd()
        try:
            cd(dirname(bin2txtpath))
            _rm_file("WEATHER.FMT")
            _rm_file("WEATHER.BIN")
            if not os.path.isabs(binfilename):
                binfilename2 = os.path.join(curdir,binfilename)
            else:
                binfilename2 = binfilename
            cp(binfilename2,"WEATHER.BIN")
                
            # no arguments needed
            pp = Popen(basename(bin2txtpath),stdout=PIPE, stderr=PIPE, shell=True)
            output, errors = pp.communicate()
            if not errors == b'':
                print("WARNING! An error was recorded by Popen.communicate but this does not mean the BIN2TXT did not work. Investigate further to verify it worked.")
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

def txt2bin(txtfilename,txt2binpath,binfilename):
    if isfile(txt2binpath) and isfile(txtfilename):
        curdir = pwd()
        try:
            change_dir = len(dirname(txt2binpath)) != 0
            if change_dir:
                cd(dirname(txt2binpath))
            _rm_file("WEATHER.BIN")
            if txtfilename != "WEATHER.FMT":
               cp(txtfilename,"WEATHER.FMT")
            # no arguments needed
            p = Popen(basename(txt2binpath),stdout=PIPE, stderr=PIPE, shell=True)
            output, errors = p.communicate()
            if not errors == b'':
                print("The process produced an error make sure the process worked!\n\n" + str(errors) + "\n\n" + str(output))
            cp("WEATHER.BIN",binfilename)
            _rm_file("WEATHER.BIN")
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


def df2bin(df, binfilename, start_datetime, hour_in_file, txt2bin_exepath,
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
        binfilename     : str : path and filename that will be output with the
                               weather signals in df.
        start_datetime  : datetime :start date and time to be output in the weather file
                                typically this is Jan 1st, fyear
        hour_in_file    : int : Either 8760 or 8784
        txt2bin_exepath : str : path and filename that point to TXT2BIN.EXE
                                DOE2 utility that can be obtained from www.doe2.com
                                after forming a license agreement with James Hirsch
                                and associates.
        location .. and all other inputs
        are normally read with the header information from a BIN file but may 
        be needed if a *.BIN file is being derived from scratch with data.
        Plese consult the DOE2 documentation below.
    
    
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
    
    _check_attrib(df, location, fyear,latitude,longitude,timezone,iwsz,iftyp,
                  clearness_number,ground_temp)
    
    cdir = pwd()
    #try:
    change_dir = len(dirname(binfilename)) != 0 
    if change_dir:
        cd(dirname(binfilename))
    _rm_file("WEATHER.FMT")
    _rm_file("WEATHER.BIN")
    with open("INPUT.DAT",'w') as dat:
        dat.write('            {0:1.0f}                 {1:1.0f}'.format(df.iwsz,df.iftyp))
    with open("WEATHER.FMT",'w') as fmt:
        # 3 header rows
        row1 = '{0:20s}{1:5d}{2:8.2f}{3:8.2f}{4:5d}{5:5d}\n'.format(df.location,df.fyear,
                df.latitude, df.longitude, df.timezone, df.solar_flag)
        fmt.write(row1)
        fmt.write((12*'{:6.2f}'+'\n').format(*df.clearness_number))
        fmt.write((12*'{:6.2f}'+'\n').format(*df.ground_temp))
        
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
        txt2bin("WEATHER.FMT",new_exe_path,os.path.basename(binfilename))
    else:
        cd(cdir)
        raise DOE2_Weather_Error("The txt2bin.exe utility is not present at: \n\n" + txt2bin_exepath)        
    cd(cdir)    
    #except:
    #    try:
    #        cd(curdir)
    #        print("Something went wrong!")
"""    final_df.location = df_tmy3Alb.location
    final_df.fyear = year
    final_df.latitude = latitude 
    final_df.longitude = longitude
    final_df.timezone = df_tmy3Alb.timezone
    final_df.iwsz = 2
    final_df.iftyp = 3
    final_df.solar_flag = final_df.iwsz + (final_df.iftyp-1) * 2 - 1    # this is IFTYPE = 3 and IWSZ = 2 an indicator of what is expected in input.dat
    final_df.clearness_number = df_tmy3Alb.clearness_number
    final_df.ground_temp = df_tmy3Alb.ground_temp  # Need ground temperature data    """    


def _check_attrib(df,location,fyear,latitude,longitude,timezone,iwsz,iftyp,clearness_number,ground_temp):
    
    attr_needed = ['location','fyear','latitude','longitude','timezone','clearness_number','ground_temp']
    for attr in attr_needed:
        if not hasattr(df,attr) and eval(attr) is None:
            raise TypeError("The dataframe input for writing a DOE2 file "+
                            "does not have the attribute '"+attr+"' and this"+
                            " attribute must therefore be specifically"+
                            " entered into the write function")
        elif not eval(attr) is None:
            # overright or create the needed attribute
            setattr(df,attr,eval(attr))
        
    if iwsz != 1 and iwsz != 2:
        raise ValueError("The input iwsz must be 1 (Word size =30bit) or 2 (Word size = 60 bit)")
    if iftyp != 1 and iftyp != 2 and iftyp !=3:
        raise ValueError("The input iftyp must be 1=OLD DATA FORMAT, "+
                         "2= NORMAL (NO SOLAR DATA), 3=THE DATA HAS SOLAR")
         
    df.iwsz = iwsz
    df.iftyp = iftyp
    df.solar_flag = df.iwsz + (df.iftyp-1) * 2 - 1    # this is IFTYPE = 3 and IWSZ = 2 an indicator of what is expected in input.dat
    

def bin2df(binfilename, start_datetime, hour_in_file, bin2txt_exepath, timezone, dst):
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
    
    txtname = bin2txt(binfilename, bin2txt_exepath)
    if len(txtname)==0:
        raise DOE2_Weather_Error("bom2df:The bin file was not successfully converted please troubleshoot!")
    
    ColumnDescription = ['MONTH (1-12)',
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
    start_datetime.year
    
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
        ind = ColumnDescription.index("DAY OF MONTH")
        m[sid:sid+24,ind] = 29
        
    
    dateVec = []

    for i in arange(m.shape[0]):

        dateVec.append(datetime(start_datetime.year,int(m[i,0]),int(m[i,1]),int(m[i,2])-1,0,0))

        #Handle daylight savings correctly
        if dateVec[-1] == dst[0]:
            dateVec[-1] = dateVec[-1] + DateOffset(hour=1)
        elif dateVec[-1] == dst[1]:
            dateVec[-1] = dateVec[-1] - DateOffset(hour=1)

            
    dateTimeIn = DatetimeIndex(dateVec)
    dateTimeIn = dateTimeIn.tz_localize(timezone, ambiguous=True)
    #dtypelist= ['int64','int64','int64','float64','float64','float64','int64','int64','int64','int64','float64','float64','float64',
            #'float64','float64','int64','float64']
    
    #dtypedict = dict(zip(ColumnDescription[3:],dtypelist))
    df = DataFrame(index=dateTimeIn,data=m,columns=ColumnDescription,dtype=float64)

    # suppress warnings here.
    warnings.simplefilter("ignore",category=UserWarning)
    df.begin_lines = b_lines

    df.location = str(b_lines[0][0:20])
    df.fyear = int(b_lines[0][20:25])
    df.latitude = float(b_lines[0][25:33])
    df.longitude = float(b_lines[0][33:41])
    df.timezone = int(b_lines[0][41:46])
    df.solar_flag = int(b_lines[0][46:51])
    df.clearness_number = [float(x) for x in b_lines[1].split("  ") if len(x)!=0]
    df.ground_temp = [float(x) for x in b_lines[2].split(" ") if len(x)!=0]        
    df.binfilename = binfilename
    return df