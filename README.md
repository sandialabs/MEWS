![MEWS](information/figures/logo.png)
[![Documentation Status](https://readthedocs.org/projects/mews/badge/?version=latest)](https://mews.readthedocs.io/en/latest/?badge=latest)

![workflow](https://github.com/sandialabs/mews/actions/workflows/pytest.yml/badge.svg)

The Multi-scenario Extreme Weather Simulator (MEWS) is a Python package designed to add
extreme weather events to existing weather data or projections. MEWS does not simulate
weather but rather adds variations in weather for the purpose of probabilistic analyses 
of infrastructure or environmental systems. 

Currently, MEWS provides the capacity to add extreme events that are initiated via a Markov
chain. When an extreme events occurs, sampling from a probability distribution indicates 
the total integrated level of additional energy added to the weather signal 
(wind,temperature,humidity etc...) beyond the dataset. MEWS also provides parameters to
increase the frequency (Markov chain probabilities) and severity (shifting parameters for
probability distributions) of extreme events. 

In general MEWS can add any function to a time series allowing gradual climate trends to also
be included in the overall weather signals.

Significant enhancements to MEWS are envisioned that provide reasonably realistic selection
of hurricane futures, extreme precipitation, and extreme heat and cold scenarios for 
resilience analysis.

Currently the infrastructure focus has been for Building Energy Simulation and MEWS can read/write
altered weather files for Energy Plus (https://energyplus.net/) and DOE-2 (https://www.doe2.com/)
weather files. Both of these provide a rich library of historic and Typical Meteorological weather
inputs around the world. Future connections will link MEWS to NOAA data sources.

The DOE-2 capability is contingent on forming a legal license agreement with 
James Hirsch and Associates (www.doe2.com) and placing the BIN2TXT.EXE and TXT2BIN.EXE
utilities that come with DOE-2.2 into the 'third_party_software' folder.

MEWS has only been tested on Windows using Python 3.8.5 more robust testing is under development. 
Documentation will also follow in the near future.

License
------------

See the LICENSE.md file for license details

Organization
------------

Directories
  * mews - Python package
  * docs - UNDER CONSTRUCTION - inital build available on ReadTheDocs (https://mews.readthedocs.io/en/latest/)
  * information - contains general information about MEWS
  * examples - current working example is run_mews_extreme_temperature_example_v_1_1.py. All others are deprecated or use older techniques

Installation
------------
  * To install the latest released version:
    
    ```
    pip install mews
    ```
    
    For the current code:
    
    ```
    cd < a directory you want to work with >
    python -m venv <a virtual environment name>
    <a virtual environment name>/Scripts/activate
    git clone git@github.com:sandialabs/MEWS.git
    cd MEWS
    pip install -e .
    ```
    If this does not work an alternative method is to:
    
    ```
    cd < a directory you want to work with >
    python -m venv <a virtual environment name>
    <a virtual environment name>/Scripts/activate
    git clone git@github.com:sandialabs/MEWS.git
    cd MEWS
    pip install -r requirements.txt
    python setup.py develop
    ```
    
    Then run the following to ensure the code passes unit testing
    
    ```
    pip install pytest
    pytest
    ```
    
    All tests should pass. If not, contact the dlvilla@sandia.gov. Sometimes failures are driven by a tex failure if you have tex on your computer.
    
    The API for MEWS is not yet documented and has many inputs. The best example of how to use the latest version is available in examples/worcester_example.py
    the other examples are either depricated or are not being kept up to date presently.

  * MEWS requires Cython which needs a C compiler in place. For windows, this can be the visual studio free Microsoft Visual C++ 14.0 Build Tools 
that are freely available at https://visualstudio.microsoft.com/visual-cpp-build-tools/. Download the build tools and install them. It is necessary
to assure the correct version of the build tools is installed. The stack exchange thread below shows how to verify the correct version is installed.

https://stackoverflow.com/questions/66838238/cython-setup-py-cant-find-installed-visual-c-build-tools

   * MEWS downloads CMIP6 data when using the ClimateScenario class. This step can be messy though and requires many retries when downloading the data live from multiple servers. As a result, the entire dataset (~24Gb) has been uploaded to https://osf.io/ts9e8/files/osfstorage and is publicly available to manually download.

Download the CMIP6_Data_Files file and then make its local path equal to the "output_folder" parameter for the ClimateScenario class in

mews.weather.climate.ClimateScenario

Using MEWS
--------
MEWS has many classes that have their API's documented but that have specialized functions that most users will not want to work with.
The MEWS function for heat waves is:

```
from mews.run_mews import extreme_temperature
```

The example in MEWS/examples/run_mews_extreme_temperature_v_1_1.py provides an example of how to use extreme_temperature. The repository now contains
pre-processed solution files for the following cities            

```
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
         
The extreme_mews input parameters can be used to only generate files from the solutions rather than running the lengthy optimization process again.

Inside "MEWS/examples/examplecd _data" are folders for each city and inside these folders you can find the solution files in "results" and "mews_epw_results" folder for EnergyPlus epw files. 

Contact 
--------

   * Daniel Villa, Sandia National Laboratories (SNL) dlvilla@sandia.gov
