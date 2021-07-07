![MEWS](documentation/figures/logo.png)

The Multi-scenario Extreme Weather Simulator (MEWS) is a Python package designed to add
extreme weather events to existing weather data or projections. MEWS does not simulate
weather but rather adds variations in weather for the purpose of probabilistic resilience
analyses of infrastructure systems. 

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
  * ci - UNDER CONSTRUCTION - eventually testing via travis-ci
  * mews - Python package
  * documentation - UNDER CONSTRUCTION - eventually documentation on ReadTheDocs
  * examples - two scripts showing how to use the 'Extremes' class to create
               heat waves 


Contact
--------

   * Daniel Villa, Sandia National Laboratories (SNL) dlvilla@sandia.gov
