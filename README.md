![MEWS](information/figures/logo.png)
[![Documentation Status](https://readthedocs.org/projects/mews/badge/?version=latest)](https://mews.readthedocs.io/en/latest/?badge=latest)

![workflow](https://github.com/sandialabs/mews/actions/workflows/pytest.yml/badge.svg)

The Multi-scenario Extreme Weather Simulator (MEWS) is a Python package designed to add extreme weather events to existing weather data or projections. **MEWS does not simulate weather** but rather adds variations in weather for the purpose of probabilistic analyses of infrastructure or environmental systems. Currently MEWS works for extreme temperature. Other enhancements to MEWS are envisioned that provide reasonably realistic selection of hurricane futures (see some preliminary work in /examples/hurricane_analysis...).

So far, the infrastructure focus has been for Building Energy Simulation and MEWS can read/write altered weather files for Energy Plus (https://energyplus.net/) and DOE-2 (https://www.doe2.com/) weather files. DOE-2 has not been tested since the first version though. Both of these tools have a rich library of historic and Typical Meteorological weather inputs around the world. See Crawly and Lawrie's web site for weather files to start from in MEWS: https://climate.onebuilding.org/.  

License
------------

See the LICENSE.md file for license details. This package has third party packages that have their own licenses that are appended to the MEWS license.

Organization
------------

Directories
  * src/mews - Python package
  * dist - wheel and tar.gz binaries for installing different releases of mews
  * docs - INCOMPLETE DOCUMENTATION - THIS README IS THE BEST PLACE TO GET INFO
  * information - contains some presentation with general information about MEWS. These are older so some of the presentations may be depricated.
  * examples - current working example is run_mews_extreme_temperature_example_v_1_1_2.py. All others are deprecated or use older techniques

Installation
------------
  * To install the latest released version:
    
    ```
    pip install mews
    ```
    
    For the current code:
    
    ```bash
    cd /a/directory/you/want
    python -m venv <a virtual environment name>
    <a virtual environment name>/Scripts/activate
    git clone https://github.com/sandialabs/MEWS.git
    cd MEWS
    pip install -e .[test]
    ```

    Then run the following to ensure the code passes unit testing
    
    ```bash
    pytest
    ```
    
    All tests should pass. Sometimes failures occur if you have tex on your computer.
    
    The API for MEWS is only documented in the code and has many inputs that can be contained in a *.dict file (see /examples/example_data/mews_run.dict). This file is just a python dictionary in text format (yes, as I have learned I now know I should have used yaml or JSON but its too late for now)
    

Other Installation Requirements
-------------------------------
  * If your having problems and you are not a developer you may need to add a C Compiler to your computer. MEWS requires Cython which needs a C compiler in place. For windows, this can be the visual studio free Microsoft Visual C++ 14.0 Build Tools that are freely available at https://visualstudio.microsoft.com/visual-cpp-build-tools/. Download the build tools and install them. It is necessary to assure the correct version of the build tools is installed. The stack exchange thread below shows how to verify the correct version is installed.

https://stackoverflow.com/questions/66838238/cython-setup-py-cant-find-installed-visual-c-build-tools

   * MEWS downloads CMIP6 data when using the ClimateScenario class. This step can be messy though and requires many retries when downloading the data live from multiple servers. As a result, the entire dataset (~24Gb) has been uploaded to https://osf.io/ts9e8/files/osfstorage and is publicly available to manually download instead of letting the code do it.

Download the CMIP6_Data_Files file and then make its local path equal to the "output_folder" parameter for the ClimateScenario class in

mews.weather.climate.ClimateScenario

Using MEWS
--------
A training video has been made available at: https://drive.google.com/file/d/1B-G5yGu0BFXCqj0BYfu_e8XFliAoeoRi/view?usp=drive_link

MEWS has many classes that have their API's documented in the code. These classes have specialized functions that most users will not want to work with. The doc strings are where you need to go to understand the large number of inputs. The MEWS function for heat waves is:

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

Contact 
--------

   * Daniel Villa, Sandia National Laboratories (SNL) dlvilla@sandia.gov
   
Citing MEWS
-----------
You can cite MEWS with one or more of the following:

* Villa, Daniel L., Nathan T. Hahn, John K. Grey, and Frances Pavich. 2024. "Futures for electrochromic windows on high performance houses in arid, cold climates." _Energy and Buildings_ https://doi.org/10.1016/j.enbuild.2024.114293.

* Macmillan, Madeline, Alexander Zolan, Morgan Bazilian and Daniel L. Villa. 2024. "Microgrid design and multi-year dispatch optimization under climate-informed load and renewable resource uncertainty." _Applied Energy_ https://doi.org/10.1016/j.apenergy.2024.123355

* Villa, Daniel L., Sang Hoon Lee, Carlo Bianchi, Juan Pablo Carvallo, Illya Azaroff, Andrea Mammoli and Tyler Schostek 2023. "Multi-scenario Extreme Weather Simulator Application to Heat Waves: Ko’olauloa Community Resilience Hub," _Science and Technology for the Built Environment_ https://doi.org/10.1080/23744731.2023.2279467

* Villa, Daniel L., Tyler J. Schostek, Krissy Govertsen, and Madeline Macmillan. 2023. "A Stochastic Model of Future Extreme Temperature Events for Infrastructure Analysis." _Environmental Modeling & Software_ https://doi.org/10.1016/j.envsoft.2023.105663.

* Villa, Daniel L., Juan Carvallo, Carlo Bianchi, and Sang Hoon Lee. 2022. "Multi-scenario Extreme Weather Simulator Application to Heat Waves." _2022 Building Performance Analysis Conference and SimBuild co-organized by ASHRAE and IBPSA-USA_ https://doi.org/10.26868/25746308.2022.C006


Sandia Funding Statement
------------------------
Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's National Nuclear Security Administration under contract DE-NA-0003525.

