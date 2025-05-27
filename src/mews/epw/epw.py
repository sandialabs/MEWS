# -*- coding: utf-8 -*-
"""

EPW - Lightweight Python package for editing EnergyPlus Weather (epw) files.
EPW is not the EPW downloaded from pypi.org, it must be downloaded
from https://github.com/building-energy/epw

EPW License
===========

MIT License

Copyright (c) 2019 Building Energy Research Group (BERG)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Author - Building Energy Research Group (BERG)
Used in MEWS as allowed by the license.

Revisions have been made to integrate permanently into MEWS 5/27/2025
"""

from copy import deepcopy
import pandas as pd
import csv

from mews.constants.data_format import EPW_DATA_DICT_URL
from mews.constants.data_format import EPW_DATA_DICTIONARY
from mews.utilities.utilities import is_numeric


class epw:
    """
    A class which represents an EnergyPlus weather (epw) file.
    """

    def __init__(self):
        """ """
        self.headers = {}
        self.dataframe = pd.DataFrame()

    def read(self, fp):
        """
        Reads an epw file.

        Parameters
        ----------
        fp : str
            The file path of the epw file

        """

        self.headers = self._read_headers(fp)
        self.dataframe = self._read_data(fp)

    def _read_headers(self, fp):
        """
        Reads the headers of an epw file.

        Parameters
        ----------
        fp : str
            The file path of the epw file

        Returns
        -------
        d : dict
            A dictionary containing the header rows

        """

        d = {}
        with open(fp, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
            for row in csvreader:
                if row[0].isdigit():
                    break
                else:
                    d[row[0]] = row[1:]
            dcheck = deepcopy(d)
            key,val = dcheck.popitem()
            if key != "DATA PERIODS" or len(val) != 6:
                raise ValueError("The last entry of EPW headers must be 'DATA PERIODS'"
                                 +" whose value is a list with 6 elements similar to:"
                                 +" ['1', '1', 'Data', 'Sunday', ' 1/ 1', '12/31']."
                                +f" Please refer to: {EPW_DATA_DICT_URL} for the"
                                +" exact EPW data dictionary" )
        return d

    def _read_data(self, fp):
        """
        Reads the climate data of an epw file.

        Parameters
        ----------
        fp : str
            The file path of the epw file

        Returns
        -------
        df : pd.DataFrame
            A DataFrame comtaining the climate data

        """
        names = list(EPW_DATA_DICTIONARY.keys())
        
        # [
        #     "Year",
        #     "Month",
        #     "Day",
        #     "Hour",
        #     "Minute",
        #     "Data Source and Uncertainty Flags",
        #     "Dry Bulb Temperature",
        #     "Dew Point Temperature",
        #     "Relative Humidity",
        #     "Atmospheric Station Pressure",
        #     "Extraterrestrial Horizontal Radiation",
        #     "Extraterrestrial Direct Normal Radiation",
        #     "Horizontal Infrared Radiation Intensity",
        #     "Global Horizontal Radiation",
        #     "Direct Normal Radiation",
        #     "Diffuse Horizontal Radiation",
        #     "Global Horizontal Illuminance",
        #     "Direct Normal Illuminance",
        #     "Diffuse Horizontal Illuminance",
        #     "Zenith Luminance",
        #     "Wind Direction",
        #     "Wind Speed",
        #     "Total Sky Cover",
        #     "Opaque Sky Cover (used if Horizontal IR Intensity missing)",
        #     "Visibility",
        #     "Ceiling Height",
        #     "Present Weather Observation",
        #     "Present Weather Codes",
        #     "Precipitable Water",
        #     "Aerosol Optical Depth",
        #     "Snow Depth",
        #     "Days Since Last Snowfall",
        #     "Albedo",
        #     "Liquid Precipitation Depth",
        #     "Liquid Precipitation Quantity",
        # ]

        first_row = self._first_row_with_climate_data(fp)
        df = pd.read_csv(fp, skiprows=first_row, header=None, names=names)
        return df

    def _first_row_with_climate_data(self, fp):
        """
        Finds the first row with the climate data of an epw file.

        Parameters
        ----------
        fp : str
            The file path of the epw file

        Returns
        -------
        i : int
            The row number

        """

        with open(fp, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
            for i, row in enumerate(csvreader):
                if row[0].isdigit():
                    break
        return i

    def write(self, fp):
        """
        Writes to an epw file.

        Parameters
        ----------
        fp : str
            The file path of the new epw file

        """

        with open(fp, "w", newline="") as csvfile:
            csvwriter = csv.writer(
                csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for k, v in self.headers.items():
                csvwriter.writerow([k] + v)
            for row in self.dataframe.itertuples(index=False):
                csvwriter.writerow(i for i in row)
