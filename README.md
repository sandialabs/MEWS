![MEWS](information/figures/logo.png)
[![Documentation Status](https://readthedocs.org/projects/mews/badge/?version=latest)](https://mews.readthedocs.io/en/latest/?badge=latest)

![workflow](https://github.com/sandialabs/mews/actions/workflows/pytest.yml/badge.svg)

# Introduction

The Multi-scenario Extreme Weather Simulator (MEWS) is a Python package designed to change EPW files including adding extreme weather events to existing weather data or projections. **MEWS does not simulate weather** but rather adds variations in weather for the purpose of probabilistic analyses of infrastructure or environmental systems. Currently MEWS works for extreme temperature and for adjusting weather files to approximate local weather conditions. Other enhancements to MEWS are envisioned that provide reasonably realistic selection of hurricane futures (see some preliminary work in /examples/hurricane_analysis...) and Villa and Quiroz, 2023 (https://doi.org/10.1080/23744731.2023.2253087).

So far, the infrastructure focus has been for Building Energy Simulation and MEWS can read/write altered weather files for Energy Plus (https://energyplus.net/) and DOE-2 (https://www.doe2.com/) weather files. DOE-2 has not been tested since the first version though. Both of these tools have a rich library of historic and Typical Meteorological weather inputs around the world. See Crawly and Lawrie's web site for weather files to start from in MEWS: https://climate.onebuilding.org/.  

# License

See the LICENSE.md file for license details. This package has third party packages that have their own licenses that are appended to the MEWS license.

# Organization

Directories
  * src/mews - Python package
  * dist - wheel and tar.gz binaries for installing different releases of mews
  * docs - INCOMPLETE DOCUMENTATION - THIS README IS THE BEST PLACE TO GET INFO
  * information - contains some presentation with general information about MEWS. These are older so some of the presentations may be depricated.
  * examples - current working example for extreme temperatures is run_mews_extreme_temperature_example_v_1_1_2.py. All others are deprecated or use older techniques

# Installation

  * To install the latest released version:
    
    ```
    pip install mews
    ```
    
    For the current code:
    
    ```bash
    cd /a/directory/you/want
    python -m venv <a virtual environment name>
    <a virtual environment name>/Scripts/activate
    git clone -b <a branch name or tag name> https://github.com/sandialabs/MEWS.git
    cd MEWS
    pip install -e .[test]
    ```

  * Then run the following to ensure the code passes unit testing
    
    ```bash
    pytest
    ```
    
  * All tests should pass. Sometimes failures occur if you have tex on your computer.
    
    The API for MEWS is only documented in the code and has many inputs that can be contained in a *.dict file (see /examples/example_data/mews_run.dict). This file is just a python dictionary in text format (yes, as I have learned I now know I should have used yaml or JSON but its too late for now)
    

## Other Installation Requirements

  * If your having problems and you are not a developer you may need to add a C Compiler to your computer. MEWS requires Cython which needs a C compiler in place. For windows, this can be the visual studio free Microsoft Visual C++ 14.0 Build Tools that are freely available at https://visualstudio.microsoft.com/visual-cpp-build-tools/. Download the build tools and install them. It is necessary to assure the correct version of the build tools is installed. The stack exchange thread below shows how to verify the correct version is installed.

https://stackoverflow.com/questions/66838238/cython-setup-py-cant-find-installed-visual-c-build-tools

   * MEWS downloads CMIP6 data when using the ClimateScenario class. This step can be messy though and requires many retries when downloading the data live from multiple servers. As a result, the entire dataset (~24Gb) has been uploaded to https://osf.io/ts9e8/files/osfstorage and is publicly available to manually download instead of letting the code do it.

Download the CMIP6_Data_Files file and then make its local path equal to the "output_folder" parameter for the ClimateScenario class in

mews.weather.climate.ClimateScenario

# Using MEWS

## Extreme temperature analyses

A training video has been made available at: https://drive.google.com/file/d/1B-G5yGu0BFXCqj0BYfu_e8XFliAoeoRi/view?usp=drive_link. The MEWS function for heat waves is:

```python
from mews.run_mews import extreme_temperature
```

The example in MEWS/examples/run_mews_extreme_temperature_v_1_1_2.py shows how to use extreme_temperature. A dataset with pre-processed solution files is available at https://osf.io/ts9e8/files/osfstorage in the Solution_File_Results file for the following cities:         

```python
cities = ["Chicago",
          "Baltimore",
          "Minneapolis",
          "Phoenix",
          'Miami',
          'Houston'
          'Atlanta', 
          'LasVegas',
          'LosAngeles',
          'SanFrancisco',
          'Albuquerque',
          'Seattle', 
          'Denver',
          'Helena', 
          'Duluth',
          'Fairbanks',
          'McAllen',
          'Kodiak',
          'Worcester']
 ```
         
The extreme_temperature input parameters can be used to only generate files from the solutions rather than running the lengthy optimization process again.

Inside "MEWS/examples/example_data" are folders for each city and inside these folders you can find the solution files in "results" and "mews_epw_results" folder for EnergyPlus epw files. 

## Altering EPW weather files however you like

MEWS can also be used to alter weather files in whatever way you like through the "Alter" class. This functionality is very similar to what Big Ladder Software has provided in its [Elements](https://bigladdersoftware.com/projects/elements/) software. If you want a GUI driven process, it would be the preferred way to go. MEWS' Alter class makes the functionality scriptable across many weather files though.  This class has the capacity to readjust psychrometric variables and shift the average of time series while holding either the maximum or minimum of the time series constant. The script below shows my own application of MEWS to accomplish this through adjusting nearly all of the important weather variables for BEM for a site that did not have weather data. I used surrounding TMYx, TMY3 data and also NASA POWER MERRA-2 gridded EPW files to estimate a local climate for a specific site. 

```python
# This script shows an example of how I shifted one location's weather
# to reflect an unmeasured location based on various different factors
# such as change in altitude, wind pattern differences, and information
# from the NASA POWER MERRA-2 reanalysis EPW files (which was too coarse)
# to capture local effects such as elevation (site was 530 m and MERRA-2)
# grid was (240 m). Some of the numbers have been changed to keep the
# actual site location anonymous.

# you have to define "path_to_epw_file" for this script to work
from mews.weather.alter import Alter
from mews.constants.data_format import (EPW_PRESSURE_COLUMN_NAME, 
                                        EPW_DB_TEMP_COLUMN_NAME, 
                                        EPW_RH_COLUMN_NAME)

alter_obj = Alter(path_to_epw_file)
df = alter_obj.epwobj.dataframe
# decrease barometric pressure by 1,500 Pa
df[EPW_PRESSURE_COLUMN_NAME] = df[EPW_PRESSURE_COLUMN_NAME] - 1500.0
# increase relative humidity mean by 2% while holding the maximum (100%) constant
df[EPW_RH_COLUMN_NAME] = alter_obj.shift_function(2.0,"Relative Humidity",False)
# decrease db temperature by 0.51 C
df[EPW_DB_TEMP_COLUMN_NAME] = df[EPW_DB_TEMP_COLUMN_NAME] - db_temp_drop
# recalculate dew-point temperature:
alter_obj.recalculate_psychrometrics()
# increase wind speed by 3.0 m/s but keep minimum wind speed equal to the minimum
df["Wind Speed"] = alter_obj.shift_function(3.0,"Wind Speed",True)
# offset wind direction from northeast to southeast by taking -15 degrees from Lanai data
# avoid crossing the zero point
df["Wind Direction"] = alter_obj.shift_function(-15.0,"Wind Direction",True)
# decrease precipitable water by 2mm 
df["Precipitable Water"] = df["Precipitable Water"] - 2.0
# we neglect changes in zenith angle due to increased elevation of 130 m.
cos_zenith = ((df["Global Horizontal Radiation"] - df["Diffuse Horizontal Radiation"])
                  /df["Direct Normal Radiation"])
# increase direct normal radiation by 6.7 W/m2 while keeping minimum of 0 = 0
direct_factor = 6.7 / df["Direct Normal Radiation"].mean()
df["Direct Normal Radiation"] = alter_obj.shift_function(6.7,"Direct Normal Radiation",True)
# increase diffuse horizontal radiation
diffuse_factor = 1.0 / df["Diffuse Horizontal Radiation"].mean()
df["Diffuse Horizontal Radiation"] = alter_obj.shift_function(1.0,"Diffuse Horizontal Radiation",True)
df["Global Horizontal Radiation"] = df["Diffuse Horizontal Radiation"] + df["Direct Normal Radiation"] * cos_zenith
df["Direct Normal Illuminance"] = direct_factor * df["Direct Normal Illuminance"] 
df["Diffuse Horizontal Illuminance"] = diffuse_factor * df["Diffuse Horizontal Illuminance"]
df["Global Horizontal Illuminance"] = df["Diffuse Horizontal Illuminance"] + df["Direct Normal Illuminance"] * cos_zenith
# neglect Zenith illuminance (illuminance from directly overhead)
# CLUES to get this done in the future: https://publications.ibpsa.org/proceedings/esim/2002/papers/esim2002_o2.pdf

# update headers:
headers['LOCATION'][-1] = 530.0 # m elevation
headers['LOCATION'][0] = "Kahikinui Community Resilience Hub Site"
headers['LOCATION'][5] = 20.63556 #latitude
headers['LOCATION'][6] = -156.2908
#If adding a new comment, you MUST make sure that "DATA PERIODS" is the last entry
# in the dictionary, or else the EPW read will fail for the written file.
headers['COMMENTS 2'] = [headers["COMMENTS 2"][0] + f'"Updated from original file {wfile} with the following changes:'
                              +'  1) Barometric pressure decreased by 1,500 Pa'
                              +f'  2) Dry Bulb Temperature decreased by {db_temp_drop} C'
                              +'. also ground temperatures reduced by the same amount.'
                              +'  3) Increased relative humidity by 2 percent and reclaculated dewpoint'
                              +' temperature from new pressure temperature and humidities.'
                              +'  4) Shift and stretched (not a simple offset) wind speed such'
                              +' that 0 m/s stays 0 but average wind speed increased by 3.0 m/s.'
                              +'  5) Offset wind direction by -15 degrees'
                              +'  6) Decreased precipitable water by 2mm'
                              +'  7) Increased DNI by 6.7 W/m2 and DHI by 1.0 W/m2. '
                              +'Used the 6.7/mean(DNI) and 1.0/mean(DHI) to apply factors to illuminance.'
                              +' Recalculated Global variables based on updated horizontal and direct '
                              +'radiation/illuminance values."']
gt =  headers['GROUND TEMPERATURES']
gt_new = (gt[0:5] 
          + [str(float(gt[idx]) - db_temp_drop) for idx in range(5,17)] 
          + gt[17:21] 
          + [str(float(gt[idx]) - db_temp_drop) for idx in range(21,33)] 
          + gt[33:37] 
          + [str(float(gt[idx]) - db_temp_drop) for idx in range(37,49)])
new_wfile = f"Kahikinui_Adjusted_{wfile}"
if os.path.exists(new_wfile):
    os.remove(new_wfile)
alter_obj.write(new_wfile)
alter_obj.read(new_wfile)
```


# Contact 

   * Daniel Villa, Sandia National Laboratories (SNL) dlvilla@sandia.gov
   
# Citing MEWS

You can cite MEWS with one or more of the following:

* Villa, Daniel L., Nathan T. Hahn, John K. Grey, and Frances Pavich. 2024. "Futures for electrochromic windows on high performance houses in arid, cold climates." _Energy and Buildings_ https://doi.org/10.1016/j.enbuild.2024.114293.

* Macmillan, Madeline, Alexander Zolan, Morgan Bazilian and Daniel L. Villa. 2024. "Microgrid design and multi-year dispatch optimization under climate-informed load and renewable resource uncertainty." _Applied Energy_ https://doi.org/10.1016/j.apenergy.2024.123355

* Villa, Daniel L., Sang Hoon Lee, Carlo Bianchi, Juan Pablo Carvallo, Illya Azaroff, Andrea Mammoli and Tyler Schostek 2023. "Multi-scenario Extreme Weather Simulator Application to Heat Waves: Ko’olauloa Community Resilience Hub," _Science and Technology for the Built Environment_ https://doi.org/10.1080/23744731.2023.2279467

* Villa, Daniel L., Tyler J. Schostek, Krissy Govertsen, and Madeline Macmillan. 2023. "A Stochastic Model of Future Extreme Temperature Events for Infrastructure Analysis." _Environmental Modeling & Software_ https://doi.org/10.1016/j.envsoft.2023.105663.

* Villa, Daniel L., Juan Carvallo, Carlo Bianchi, and Sang Hoon Lee. 2022. "Multi-scenario Extreme Weather Simulator Application to Heat Waves." _2022 Building Performance Analysis Conference and SimBuild co-organized by ASHRAE and IBPSA-USA_ https://doi.org/10.26868/25746308.2022.C006


# Sandia Funding Statement

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA-0003525.

